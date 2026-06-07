"""Workflow state definitions for the MethylCurate agent.

Defines state models for the main graph and each subgraph (GEO,
harmonization, QC, benchmarking) along with state construction utilities.
"""

__all__ = ["models", "utils"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"models", "utils"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.agent.state' has no attribute {name!r}")
    return __import__(f"methylcurate.agent.state.{name}", fromlist=[name])
