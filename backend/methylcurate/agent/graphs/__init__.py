"""Graph definitions for the MethylCurate agent layer.

Encodes valid workflow transitions among module-level nodes:
GEO retrieval → metadata harmonization → quality control → benchmarking.
"""

__all__ = [
    "benchmarking",
    "deps",
    "geo",
    "harmonization",
    "help",
    "qualitycontrol",
    "router",
    "subgraphs",
]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {
        "benchmarking",
        "deps",
        "geo",
        "harmonization",
        "help",
        "qualitycontrol",
        "router",
        "subgraphs",
    }
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.agent.graphs' has no attribute {name!r}")
    return __import__(f"methylcurate.agent.graphs.{name}", fromlist=[name])
