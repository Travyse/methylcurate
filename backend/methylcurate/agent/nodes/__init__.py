"""Agent nodes for the MethylCurate workflow.

Each node coordinates a module-level action: routing, GEO retrieval,
metadata harmonization, quality control, benchmarking, and help.
"""

__all__ = ["benchmarking", "geo", "harmonize", "help", "qc", "router"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"benchmarking", "geo", "harmonize", "help", "qc", "router"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.agent.nodes' has no attribute {name!r}")
    return __import__(f"methylcurate.agent.nodes.{name}", fromlist=[name])
