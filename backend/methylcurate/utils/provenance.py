"""Structured provenance logging for LLM-powered workflows.

Emits JSONL events capturing run metadata, artifact tracking,
LLM interactions, retrieval provenance, and error information.
"""

__all__ = ["ProvenanceLogger", "ProvenanceRegistry"]

import contextvars
import hashlib
import json
import os
import re
import threading
from datetime import UTC, datetime
from typing import Any

_current_provenance: contextvars.ContextVar[Any] = contextvars.ContextVar("_current_provenance", default=None)


def set_active_provenance(provenance: Any) -> None:
    _current_provenance.set(provenance)


def get_active_provenance() -> Any:
    return _current_provenance.get()


def _compute_sha256(content: str) -> str:
    h = hashlib.sha256()
    h.update(content.encode("utf-8"))
    return h.hexdigest()


_SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(sk-ant-[a-zA-Z0-9_-]{10,})"), "sk-ant-<REDACTED>"),
    (re.compile(r"\b(sk-(?!ant-|<)[a-zA-Z0-9_-]{10,})"), "sk-<REDACTED>"),
    (re.compile(r"([\w.-]+@[\w.-]+\.\w+)"), "<EMAIL_REDACTED>"),
]


def _redact_sensitive(text: str) -> str:
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _safe_truncate(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


class ProvenanceLogger:
    def __init__(self, log_path: str, run_id: str, subgraph: str, *, thread_id: str | None = None):
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._path = log_path
        self._run_id = run_id
        self._subgraph = subgraph
        self._thread_id = thread_id
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._file: Any = None

    def _open(self) -> None:
        if self._file is None:
            self._file = open(self._path, "a", encoding="utf-8")

    def _emit(self, event: str, **fields: Any) -> None:
        if self._closed:
            return
        with self._lock:
            self._open()
            payload: dict[str, Any] = {
                "event": event,
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": self._run_id,
                "subgraph": self._subgraph,
                **fields,
            }
            if self._thread_id is not None:
                payload["thread_id"] = self._thread_id
            self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._file.flush()

    def emit_run_started(
        self,
        *,
        provider: str,
        model: str,
        temperature: float,
        top_p: float,
        top_k: int,
        timeout_s: int,
        max_retries: int,
        streaming: bool,
        accessions: list[str],
        output_root: str,
        git_commit: str | None = None,
        python_version: str | None = None,
    ) -> None:
        self._started = True
        self._emit(
            "RunStarted",
            config_snapshot={
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "timeout_s": timeout_s,
                "max_retries": max_retries,
                "streaming": streaming,
            },
            accessions=accessions,
            output_root=output_root,
            git_commit=git_commit,
            python_version=python_version,
        )

    def emit_input_loaded(
        self,
        *,
        artifact_kind: str,
        artifact_path: str,
        artifact_sha256: str,
        accession_code: str | None = None,
        step: str | None = None,
        num_samples: int | None = None,
        key_names: list[str] | None = None,
    ) -> None:
        self._emit(
            "InputLoaded",
            artifact_kind=artifact_kind,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            accession_code=accession_code,
            step=step,
            num_samples=num_samples,
            key_names=key_names,
        )

    def emit_prompt_rendered(
        self,
        *,
        template_name: str,
        template_path: str,
        rendered_sha256: str,
        template_sha256: str | None = None,
        rendered_preview: str | None = None,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        safe_preview = _redact_sensitive(_safe_truncate(rendered_preview, 200)) if rendered_preview else None
        self._emit(
            "PromptRendered",
            template_name=template_name,
            template_path=template_path,
            template_sha256=template_sha256,
            rendered_sha256=rendered_sha256,
            rendered_preview=safe_preview,
            accession_code=accession_code,
            step=step,
        )

    def emit_llm_request_started(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        attempt: int = 1,
        message_count: int | None = None,
        structured_output_schema: str | None = None,
        prompt_sha256: str | None = None,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "LLMRequestStarted",
            model=model,
            provider=provider,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            attempt=attempt,
            message_count=message_count,
            structured_output_schema=structured_output_schema,
            prompt_sha256=prompt_sha256,
            accession_code=accession_code,
            step=step,
        )

    def emit_llm_response_received(
        self,
        *,
        latency_ms: float | None = None,
        output_schema: str | None = None,
        output_sha256: str | None = None,
        parsed_success: bool = True,
        finish_reason: str | None = None,
        attempt: int = 1,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "LLMResponseReceived",
            latency_ms=latency_ms,
            output_schema=output_schema,
            output_sha256=output_sha256,
            parsed_success=parsed_success,
            finish_reason=finish_reason,
            attempt=attempt,
            accession_code=accession_code,
            step=step,
        )

    def emit_llm_token_usage(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "LLMTokenUsage",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            accession_code=accession_code,
            step=step,
        )

    def emit_retrieval_query_issued(
        self,
        *,
        query: str,
        ontology: str,
        k: int,
        api_url: str,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "RetrievalQueryIssued",
            query=query,
            ontology=ontology,
            k=k,
            api_url=api_url,
            accession_code=accession_code,
            step=step,
        )

    def emit_retrieval_candidates_returned(
        self,
        *,
        query: str,
        ontology: str,
        num_candidates: int,
        candidate_ids: list[str],
        candidate_labels: list[str],
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "RetrievalCandidatesReturned",
            query=query,
            ontology=ontology,
            num_candidates=num_candidates,
            candidate_ids=candidate_ids,
            candidate_labels=candidate_labels,
            accession_code=accession_code,
            step=step,
        )

    def emit_harmonization_mapping_proposed(
        self,
        *,
        source_label: str,
        target_label: str,
        ontology: str,
        mapping_type: str,
        candidates: list[str] | None = None,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "HarmonizationMappingProposed",
            source_label=source_label,
            target_label=target_label,
            ontology=ontology,
            mapping_type=mapping_type,
            candidates=candidates,
            accession_code=accession_code,
            step=step,
        )

    def emit_validation_completed(
        self,
        *,
        concept: str,
        parse_rate: float | None = None,
        num_resolved: int | None = None,
        num_missing: int | None = None,
        num_error: int | None = None,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "ValidationCompleted",
            concept=concept,
            parse_rate=parse_rate,
            num_resolved=num_resolved,
            num_missing=num_missing,
            num_error=num_error,
            accession_code=accession_code,
            step=step,
        )

    def emit_artifact_written(
        self,
        *,
        artifact_kind: str,
        artifact_path: str,
        artifact_sha256: str | None = None,
        artifact_bytes: int | None = None,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "ArtifactWritten",
            artifact_kind=artifact_kind,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            artifact_bytes=artifact_bytes,
            accession_code=accession_code,
            step=step,
        )

    def emit_warning_raised(
        self,
        *,
        warning: str,
        context: str | None = None,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "WarningRaised",
            warning=warning,
            context=context,
            accession_code=accession_code,
            step=step,
        )

    def emit_retry_scheduled(
        self,
        *,
        error_type: str,
        error_message: str | None = None,
        retry_count: int = 0,
        retry_limit: int = 5,
        backoff_s: float | None = None,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "RetryScheduled",
            error_type=error_type,
            error_message=error_message,
            retry_count=retry_count,
            retry_limit=retry_limit,
            backoff_s=backoff_s,
            accession_code=accession_code,
            step=step,
        )

    def emit_retries_exhausted(
        self,
        *,
        error_type: str,
        error_message: str | None = None,
        total_attempts: int = 0,
        max_retries: int = 0,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "RetriesExhausted",
            error_type=error_type,
            error_message=error_message,
            total_attempts=total_attempts,
            max_retries=max_retries,
            accession_code=accession_code,
            step=step,
        )

    def emit_error_raised(
        self,
        *,
        error_type: str,
        error_message: str | None = None,
        error_code: str | None = None,
        stack_trace: str | None = None,
        context: str | None = None,
        accession_code: str | None = None,
        step: str | None = None,
    ) -> None:
        self._emit(
            "ErrorRaised",
            error_type=error_type,
            error_message=error_message,
            error_code=error_code,
            stack_trace=stack_trace,
            context=context,
            accession_code=accession_code,
            step=step,
        )

    def emit_run_completed(
        self,
        *,
        status: str = "completed",
        total_artifacts: int = 0,
        total_llm_calls: int = 0,
        total_tokens: int | None = None,
        total_latency_ms: float | None = None,
        artifact_manifest: list[dict[str, Any]] | None = None,
        accession_code: str | None = None,
    ) -> None:
        self._emit(
            "RunCompleted",
            status=status,
            total_artifacts=total_artifacts,
            total_llm_calls=total_llm_calls,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            artifact_manifest=artifact_manifest or [],
            accession_code=accession_code,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> "ProvenanceLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ProvenanceRegistry:
    """Registry that creates and caches per-thread ProvenanceLogger instances.

    Each thread gets its own provenance file at
    ``{output_root}/provenance/{thread_id}.jsonl``.

    Thread-safe for concurrent access.
    """

    def __init__(self) -> None:
        self._loggers: dict[str, ProvenanceLogger] = {}
        self._lock = threading.Lock()

    def get_or_create(self, thread_id: str, output_root: str, subgraph: str = "") -> ProvenanceLogger:
        """Return (or create) the ProvenanceLogger for *thread_id*.

        The log path is derived from *output_root* and *thread_id*.
        """
        with self._lock:
            if thread_id not in self._loggers:
                log_path = os.path.join(output_root, "logs", "provenance.jsonl")
                self._loggers[thread_id] = ProvenanceLogger(
                    log_path=log_path,
                    run_id=thread_id,
                    subgraph=subgraph,
                    thread_id=thread_id,
                )
            logger = self._loggers[thread_id]
        return logger

    def get(self, thread_id: str) -> ProvenanceLogger | None:
        """Return the logger for *thread_id* without creating one."""
        with self._lock:
            return self._loggers.get(thread_id)

    def close(self, thread_id: str) -> None:
        """Close and remove the logger for *thread_id*."""
        with self._lock:
            logger = self._loggers.pop(thread_id, None)
            if logger is not None:
                logger.close()

    def close_all(self) -> None:
        """Close all cached loggers."""
        with self._lock:
            for logger in self._loggers.values():
                logger.close()
            self._loggers.clear()
