"""Epigenetic clock models and inference tools.

Provides clock model definitions, prediction inference, age acceleration
computation, and benchmark metric calculations.
"""

__all__ = ["clock_models", "inference"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"clock_models", "inference"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.tools.clocks' has no attribute {name!r}")
    return __import__(f"methylcurate.tools.clocks.{name}", fromlist=[name])
