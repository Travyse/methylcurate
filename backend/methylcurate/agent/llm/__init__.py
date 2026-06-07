"""LLM client abstraction for MethylCurate.

Provides configuration-driven LLM clients with structured output,
streaming, token tracking, and provenance-aware logging.
"""

__all__ = ["client", "logged_client", "token_tracker"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"client", "logged_client", "token_tracker"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.agent.llm' has no attribute {name!r}")
    return __import__(f"methylcurate.agent.llm.{name}", fromlist=[name])
