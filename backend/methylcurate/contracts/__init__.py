"""Structured input/output contracts for MethylCurate.

Defines Pydantic models for module-level I/O: GEO downloads, metadata
extraction, harmonization mappings, QC operations, and clock predictions.
"""

__all__ = ["clocks", "common", "geo", "harmonize", "qc", "router"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"clocks", "common", "geo", "harmonize", "qc", "router"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.contracts' has no attribute {name!r}")
    return __import__(f"methylcurate.contracts.{name}", fromlist=[name])
