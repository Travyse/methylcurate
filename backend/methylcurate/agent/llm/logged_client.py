"""Provenance-aware LLM client wrapper.

Intercepts acall_structured() to emit PromptRendered, LLMRequestStarted,
LLMResponseReceived, and LLMTokenUsage provenance events transparently.

Uses a ContextVar for per-thread/per-task provenance isolation.
Call ``set_active_provenance(provenance)`` before running work.
"""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from ...utils.helper import compute_sha256
from ...utils.provenance import get_active_provenance

T = TypeVar("T", bound=BaseModel)


def _schema_name(schema: type) -> str:
    name = getattr(schema, "__name__", None)
    if name is not None:
        return name
    return str(schema)


def _message_sha256(messages: str | list) -> str:
    if isinstance(messages, (str,)):
        return compute_sha256(messages)
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            content = msg.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(str(block["text"]))
        elif isinstance(msg, dict):
            parts.append(json.dumps(msg, sort_keys=True, default=str))
    return compute_sha256("".join(parts))


def _message_count(messages: str | list) -> int:
    if isinstance(messages, str):
        return 1
    return len(messages)


def _render_preview(messages: str | list, max_chars: int = 200) -> str:
    if isinstance(messages, str):
        text = messages
    else:
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                content = msg.content
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            parts.append(str(block["text"]))
        text = "".join(parts)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def _extract_token_usage(result: Any) -> dict[str, int] | None:
    usage = getattr(result, "usage_metadata", None)
    if usage is None:
        try:
            response_metadata = getattr(result, "response_metadata", {})
            token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
            if token_usage:
                return {
                    "prompt_tokens": int(token_usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(token_usage.get("completion_tokens", 0)),
                    "total_tokens": int(token_usage.get("total_tokens", 0)),
                }
        except Exception:
            pass
        return None
    try:
        return {
            "prompt_tokens": int(usage.get("input_tokens", 0)),
            "completion_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
    except Exception:
        return None


class LoggedLLMClient:
    def __init__(self, client: Any):
        self._client = client
        self.config = client.config

    def _get_provenance(self) -> Any:
        return get_active_provenance()

    async def acall_structured(self, prompt: str | list, schema: type[T]) -> T:
        provenance = self._get_provenance()
        cfg = self.config
        prompt_sha = _message_sha256(prompt)
        schema_str = _schema_name(schema)
        msg_count = _message_count(prompt)

        if provenance is not None:
            provenance.emit_prompt_rendered(
                template_name="acall_structured",
                template_path="",
                rendered_sha256=prompt_sha,
                rendered_preview=_render_preview(prompt),
            )
            provenance.emit_llm_request_started(
                model=cfg.model,
                provider=cfg.provider,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                attempt=1,
                message_count=msg_count,
                structured_output_schema=schema_str,
                prompt_sha256=prompt_sha,
            )

        start = time.perf_counter()
        try:
            result = await self._client.acall_structured(prompt, schema)
            latency_ms = (time.perf_counter() - start) * 1000
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            if provenance is not None:
                provenance.emit_llm_response_received(
                    latency_ms=latency_ms,
                    output_schema=schema_str,
                    parsed_success=False,
                    attempt=1,
                )
            raise

        if provenance is not None:
            provenance.emit_llm_response_received(
                latency_ms=latency_ms,
                output_schema=schema_str,
                parsed_success=True,
                attempt=1,
            )
            token_usage = _extract_token_usage(result)
            if token_usage is not None:
                provenance.emit_llm_token_usage(
                    prompt_tokens=token_usage["prompt_tokens"],
                    completion_tokens=token_usage["completion_tokens"],
                    total_tokens=token_usage["total_tokens"],
                )

        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
