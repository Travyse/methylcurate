"""Runtime components for the MethylCurate agent.

Executes graph steps, manages tool calls, coordinates state transitions,
and handles streaming with human-in-the-loop interrupts.
"""

__all__ = ["chat_runner", "session_store"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"chat_runner", "session_store"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.agent.runtime' has no attribute {name!r}")
    return __import__(f"methylcurate.agent.runtime.{name}", fromlist=[name])
