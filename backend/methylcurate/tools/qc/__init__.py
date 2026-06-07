"""Quality control tools for methylation data preparation.

Handles data type detection/conversion, CpG/sample-level QC, feature
selection, missing-value imputation, and inter-array correlation checks.
"""

__all__ = ["data_type_conversion", "feature_selection", "impute", "qc", "workflow"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"data_type_conversion", "feature_selection", "impute", "qc", "workflow"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.tools.qc' has no attribute {name!r}")
    return __import__(f"methylcurate.tools.qc.{name}", fromlist=[name])
