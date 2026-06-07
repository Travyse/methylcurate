"""Deterministic tool implementations for MethylCurate.

Subpackages provide GEO download/extraction, metadata harmonization,
quality control, and epigenetic clock inference functionality.
"""

__all__ = ["clocks", "geo", "harmonize", "qc"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"clocks", "geo", "harmonize", "qc"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.tools' has no attribute {name!r}")
    return __import__(f"methylcurate.tools.{name}", fromlist=[name])
