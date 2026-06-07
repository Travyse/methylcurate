"""Agent layer for MethylCurate.

Coordinates workflows across GEO retrieval, metadata harmonization,
quality control, and benchmarking modules through LangGraph-based
graphs, nodes, and runtime machinery.
"""

__all__ = ["graphs", "llm", "nodes", "registry", "runtime", "state"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"graphs", "llm", "nodes", "registry", "runtime", "state"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.agent' has no attribute {name!r}")
    return __import__(f"methylcurate.agent.{name}", fromlist=[name])
