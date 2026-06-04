"""Thread-safe token usage accumulator for LLM calls.

Provides a ContextVar-based tracker that accumulates prompt and completion
tokens across all LLM calls within a run. Used by LoggedLLMClient to record
usage and by the runtime to surface totals in RunCompleted provenance events.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass

_Tracker: contextvars.ContextVar[TokenUsageTracker | None] = contextvars.ContextVar("_llm_token_tracker", default=None)


def get_tracker() -> TokenUsageTracker | None:
    """Return the active TokenUsageTracker, if one has been set."""
    return _Tracker.get()


def set_tracker(tracker: TokenUsageTracker | None) -> None:
    """Set the active TokenUsageTracker for the current context."""
    _Tracker.set(tracker)


@dataclass
class TokenUsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0

    def record(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.call_count += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
