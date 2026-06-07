"""REST API server for MethylCurate.

FastAPI-based server with LangGraph-SDK-compatible SSE streaming for
assistant-ui integration.
"""

__all__ = ["file_parser", "schemas", "server", "session"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"file_parser", "schemas", "server", "session"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.api' has no attribute {name!r}")
    return __import__(f"methylcurate.api.{name}", fromlist=[name])
