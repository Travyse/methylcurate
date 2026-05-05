# agent/llm/client.py
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, SecretStr

T = TypeVar("T", bound=BaseModel)

Provider = Literal["openai", "azure_openai", "anthropic", "ollama"]


@dataclass
class LLMConfig:
    """
    Configuration for the LLM client.

    Attributes:
        provider (Provider): The LLM provider.
        model (str): The model name.
        openai_api_key (Optional[str]): OpenAI API key.
        openai_base_url (Optional[str]): OpenAI base URL.
        azure_api_key (Optional[str]): Azure API key.
        azure_endpoint (Optional[str]): Azure endpoint.
        azure_deployment (Optional[str]): Azure deployment.
        azure_api_version (Optional[str]): Azure API version.
        anthropic_api_key (Optional[str]): Anthropic API key.
        ollama_base_url (Optional[str]): Ollama base URL.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling.
        top_p (float): Top-p sampling.
        timeout_s (int): Timeout in seconds.
        max_retries (int): Maximum retries.
        reasoning (Optional[bool]): Enable reasoning.
        streaming (bool): Enable streaming.
    """

    provider: Provider
    model: str

    # OpenAI
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # Azure OpenAI
    azure_api_key: str | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    azure_api_version: str | None = None

    # Anthropic
    anthropic_api_key: str | None = None

    # Ollama
    ollama_base_url: str | None = None  # e.g. http://localhost:11434

    # Runtime behavior
    temperature: float = 0.0
    top_k: int = 40
    top_p: float = 0.9
    timeout_s: int = 120
    max_retries: int = 2
    reasoning: bool | None = None

    # Streaming
    streaming: bool = True


class LLMClient:
    """
    LLM client wrapper that abstracts over different providers and enforces certain capabilities:
        Native structured output (tool/function calling) via with_structured_output()
        Token streaming via astream()
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.

        Args:
            config (LLMConfig): The configuration for the LLM client.
        """
        self.config = config
        self._llm = self._build_llm(config)

        # Enforce "native structured output only"
        if not hasattr(self._llm, "with_structured_output"):
            raise RuntimeError(
                f"Model wrapper for provider={config.provider} does not expose with_structured_output(). Pick a tool-calling-capable backend/version."
            )

        # Enforce "token streaming"
        if not hasattr(self._llm, "astream"):
            raise RuntimeError(
                f"Model wrapper for provider={config.provider} does not expose astream(). "
                "Pin a LangChain version that supports async streaming for this backend."
            )

    # ----------------------------
    # Construction
    # ----------------------------
    def _build_llm(self, cfg: LLMConfig):
        """
        Build the LLM instance based on the configuration.

        Args:
            cfg (LLMConfig): The configuration for the LLM client.

        Returns:
            An instance of the LLM client.
        """
        if cfg.provider == "openai":
            api_key = cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OpenAI API key (OPENAI_API_KEY or config.openai_api_key).")

            kwargs: dict[str, Any] = dict(
                model=cfg.model,
                api_key=api_key,
                temperature=cfg.temperature,
                timeout=cfg.timeout_s,
                max_retries=cfg.max_retries,
                streaming=cfg.streaming,
            )
            if cfg.openai_base_url:
                kwargs["base_url"] = cfg.openai_base_url

            return ChatOpenAI(**kwargs)

        if cfg.provider == "azure_openai":
            api_key = cfg.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = cfg.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = cfg.azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            api_version = cfg.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION")

            missing = [
                k
                for k, v in {
                    "AZURE_OPENAI_API_KEY": api_key,
                    "AZURE_OPENAI_ENDPOINT": endpoint,
                    "AZURE_OPENAI_DEPLOYMENT": deployment,
                    "AZURE_OPENAI_API_VERSION": api_version,
                }.items()
                if not v
            ]
            if missing:
                raise ValueError(f"Azure OpenAI config missing: {', '.join(missing)}")

            return AzureChatOpenAI(
                azure_endpoint=endpoint,
                azure_deployment=deployment,
                api_version=api_version,
                api_key=SecretStr(api_key) if api_key else None,
                temperature=cfg.temperature,
                timeout=cfg.timeout_s,
                max_retries=cfg.max_retries,
                streaming=cfg.streaming,
            )

        if cfg.provider == "anthropic":
            api_key = cfg.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Missing Anthropic API key (ANTHROPIC_API_KEY or config.anthropic_api_key).")

            return ChatAnthropic(
                model_name=cfg.model,
                api_key=SecretStr(api_key),
                temperature=cfg.temperature,
                timeout=cfg.timeout_s,
                max_retries=cfg.max_retries,
                streaming=cfg.streaming,
            )

        if cfg.provider == "ollama":
            kwargs: dict[str, Any] = dict(model=cfg.model, temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p, seed=42)
            if cfg.ollama_base_url:
                kwargs["base_url"] = cfg.ollama_base_url
            # Many Ollama setups support token streaming; native structured output (tools) is not guaranteed.
            return ChatOllama(**kwargs)

        raise ValueError(f"Unsupported provider: {cfg.provider}")

    # ----------------------------
    # Plain (non-streaming) calls
    # ----------------------------
    def call_text(self, prompt: str) -> str:
        """
        Synchronous full response text.
        (Not streaming; use astream_text for streaming.)

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The full response text.
        """
        msg = self._llm.invoke(prompt)
        return self._stringify_content(getattr(msg, "content", msg))

    async def acall_text(self, prompt: str) -> str:
        """
        Async full response text.

        Args:
            prompt (str): The input prompt.
        Returns:
            str: The full response text.
        """
        msg = await self._llm.ainvoke(prompt)
        return self._stringify_content(getattr(msg, "content", msg))

    # ----------------------------
    # Native structured output
    # ----------------------------
    def call_structured(self, prompt: str, schema: type[T]) -> T:
        """
        Native structured output using tool/function calling.
        Returns a Pydantic model instance of type `schema`.

        Args:
            prompt (str): The input prompt.
            schema (Type[T]): The Pydantic model class to validate the output against.

        Returns:
            T: An instance of the Pydantic model `schema`.
        """
        runnable = self._llm.with_structured_output(schema)
        out = runnable.invoke(prompt)

        # LangChain usually returns an instance of schema, but be defensive.
        if isinstance(out, schema):
            return out
        return schema.model_validate(out)

    async def acall_structured(self, prompt: str | list[AnyMessage], schema: type[T]) -> T:
        """
        Async native structured output using tool/function calling.
        Returns a Pydantic model instance of type `schema`.

        Args:
            prompt (str): The input prompt.
            schema (Type[T]): The Pydantic model class to validate the output against.

        Returns:
            T: An instance of the Pydantic model `schema`.
        """
        runnable = self._llm.with_structured_output(schema)
        out = await runnable.ainvoke(prompt)
        if isinstance(out, schema):
            return out
        return schema.model_validate(out)

    # ----------------------------
    # Token streaming
    # ----------------------------
    async def astream_text(self, prompt: str) -> AsyncIterator[str]:
        """
        Async generator yielding token deltas (text chunks).

        Args:
            prompt (str): The input prompt.

        Returns:
            AsyncIterator[str]: An async iterator yielding text chunks.
        """
        async for chunk in self._llm.astream(prompt):
            # chunk is usually an AIMessageChunk with .content
            content = getattr(chunk, "content", chunk)
            text = self._stringify_content(content)
            if text:
                yield text

    async def astream_structured_events(self, prompt: str, schema: type[T]) -> AsyncIterator[dict[str, Any]]:
        """
        If you want to stream *events* for structured calls (tool call events), you typically do this at the
        LangGraph/LangChain runnable layer with astream_events.

        This method provides a streaming events iterator for the runnable produced by with_structured_output.
        You can feed these events into SSE for debugging/progress UI.

        Args:
            prompt (str): The input prompt.
            schema (Type[T]): The Pydantic model class to validate the output against.

        Returns:
            AsyncIterator[Dict[str, Any]]: An async iterator yielding structured events.
        """
        runnable = self._llm.with_structured_output(schema)
        if not hasattr(runnable, "astream_events"):
            raise RuntimeError("Runnable does not support astream_events() in this pinned version.")
        async for ev in runnable.astream_events(prompt, version="v2"):
            yield ev

    # ----------------------------
    # Utils
    # ----------------------------
    def _stringify_content(self, content: Any) -> str:
        """
        Normalize message content to text.
        Handles providers that return list-of-blocks.

        Args:
            content (Any): The raw content from the LLM response.

        Returns:
            str: The normalized text content.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    # last resort
                    parts.append(str(item))
            return "".join(parts)
        return str(content)
