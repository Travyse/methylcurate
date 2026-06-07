"""Shared utility functions for MethylCurate.

Provides error codes, exception handling, prompt rendering, provenance
logging, memory tracking, data sampling, and file I/O helpers.
"""

__all__ = [
    "error_codes",
    "exception_handling",
    "examples",
    "helper",
    "logging",
    "memory",
    "prompting",
    "provenance",
]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {
        "error_codes",
        "exception_handling",
        "examples",
        "helper",
        "logging",
        "memory",
        "prompting",
        "provenance",
    }
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.utils' has no attribute {name!r}")
    return __import__(f"methylcurate.utils.{name}", fromlist=[name])
