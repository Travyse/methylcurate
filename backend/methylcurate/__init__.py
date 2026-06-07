"""Top-level package for methylcurate."""

__all__ = ["api", "agent", "contracts", "tools", "utils"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"api", "agent", "contracts", "tools", "utils"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate' has no attribute {name!r}")
    mod = __import__(f"methylcurate.{name}", fromlist=[name])
    return mod
