# Mean inter-correlation > cutoff (0.97)
# Maximum beta-value > 0.99 or M-value equivalent
# Any sample wiht > 10% of probes with detection p-value > 0.01 removed (default)
# All probes with missing beta-values, with detection -lvaue > 0.01, and non-CG probes
# Probes on sex chromosomes removed

__all__ = []
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer, KNNImputer
from typing import List, Tuple, Dict, Any
from ...contracts.qc import (
    SampleLevelQCInput,
    SampleLevelQCResult,
    CpGLevelQCInput,
    CpGLevelQCResult,
    DNAmQCInput,
    DNAmQCResult,
    InterarrayCorrelationQCInput,
    InterarrayCorrelationQCResult,
)


def handle_sample_level_missingness(
    qc_input: SampleLevelQCInput, data_df: pd.DataFrame
) -> Tuple[SampleLevelQCResult, pd.DataFrame]:
    """Filter samples whose missing-value rate exceeds the configured cutoff.

    Args:
        qc_input: Sample-level QC parameters including the
            ``missing_cutoff`` threshold.
        data_df: Methylation data frame (samples × probes).

    Returns:
        A tuple of (QC result listing removed sample IDs,
        filtered DataFrame).
    """
    sample_missing_rate = data_df.isnull().sum(axis=1) / len(data_df.columns)
    samples_to_keep = sample_missing_rate <= qc_input.missing_cutoff
    data_df_filtered = data_df[samples_to_keep].copy()
    removed_samples = data_df.index[~samples_to_keep].tolist()
    state_result = SampleLevelQCResult.model_validate({"removed_samples": removed_samples})
    return state_result, data_df_filtered


def handle_cpg_level_missingness(
    qc_input: CpGLevelQCInput, data_df: pd.DataFrame
) -> Tuple[CpGLevelQCResult, pd.DataFrame]:
    """Filter CpG probes with high missingness and impute remaining gaps.

    Probes exceeding the missingness cutoff are dropped.  Surviving
    probes are imputed using the configured imputer (KNN or simple).

    Args:
        qc_input: CpG-level QC parameters including missingness
            cutoff and imputation strategy/model.
        data_df: Methylation data frame (samples × probes).

    Returns:
        A tuple of (QC result listing removed CpG IDs and the
        pre-imputation missingness fraction, filtered + imputed DataFrame).
    """
    if data_df.empty:
        return CpGLevelQCResult.model_validate({"removed_cpgs": [], "missing_before_imputation": 0.0}), data_df
    cpg_missing_rate = data_df.isnull().sum(axis=0) / len(data_df)
    cpgs_to_keep = cpg_missing_rate <= qc_input.missing_cutoff
    high_quality_cpgs = data_df.columns[cpgs_to_keep].tolist()
    removed_cpgs = data_df.columns[~cpgs_to_keep].tolist()
    data_df_filtered = data_df[high_quality_cpgs].copy()
    overall_missing_rate = data_df_filtered.isna().to_numpy().mean()

    if qc_input.imputation_strategy.imputation_model.concept == "knn":
        imputer = KNNImputer(
            n_neighbors=qc_input.imputation_strategy.imputation_model.n_neighbors,
            weights=qc_input.imputation_strategy.imputation_model.weights,
        )
    else:
        imputer = SimpleImputer(strategy=qc_input.imputation_strategy.imputation_model.strategy)

    data_df_filtered.loc[:, high_quality_cpgs] = imputer.fit_transform(data_df_filtered[high_quality_cpgs])
    state_result = CpGLevelQCResult.model_validate(
        {"removed_cpgs": removed_cpgs, "missing_before_imputation": overall_missing_rate}
    )
    return state_result, data_df_filtered


def maximum_dnam_filter(qc_input: DNAmQCInput, data_df: pd.DataFrame) -> Tuple[DNAmQCResult, pd.DataFrame]:
    """Keep only samples whose maximum DNAm value exceeds the cutoff.

    Args:
        qc_input: DNAm QC parameters including the ``dnam_cutoff``
            threshold.
        data_df: Methylation data frame (samples × probes).

    Returns:
        A tuple of (QC result listing removed sample IDs,
        filtered DataFrame).
    """
    if data_df.empty:
        return DNAmQCResult.model_validate({"removed_samples": []}), data_df
    max_dnam_per_sample = np.nanmax(data_df.values, axis=1)
    dnam_filter = max_dnam_per_sample > qc_input.dnam_cutoff
    data_df_filtered = data_df[dnam_filter].copy()
    removed_samples = data_df.index[~dnam_filter].tolist()
    state_result = DNAmQCResult.model_validate({"removed_samples": removed_samples})
    return state_result, data_df_filtered


def interarray_correlation(
    qc_input: InterarrayCorrelationQCInput, data_df: pd.DataFrame
) -> Tuple[InterarrayCorrelationQCResult, pd.DataFrame]:
    """Filter samples whose mean inter-array correlation falls below the cutoff.

    Computes the full pairwise sample correlation matrix, derives each
    sample's mean correlation to all other samples, and retains only
    those above the configured threshold.

    Args:
        qc_input: Inter-array correlation QC parameters including the
            ``correlation_cutoff`` threshold.
        data_df: Methylation data frame (samples × probes).

    Returns:
        A tuple of (QC result listing removed sample IDs,
        filtered DataFrame).
    """
    if data_df.empty:
        return InterarrayCorrelationQCResult.model_validate({"removed_samples": []}), data_df
    with np.errstate(invalid="ignore"):
        corr_matrix = np.corrcoef(data_df.values)
    if np.any(np.isnan(corr_matrix)):
        return InterarrayCorrelationQCResult.model_validate(
            {
                "removed_samples": [],
                "notes": "Correlation computation produced NaN (possibly due to constant-variance samples).",
            }
        ), data_df
    mean_inter_corr = (np.sum(corr_matrix, axis=1) - 1) / (data_df.shape[0] - 1)
    corr_filter = mean_inter_corr > qc_input.correlation_cutoff
    data_df_filtered = data_df[corr_filter].copy()
    removed_samples = data_df.index[~corr_filter].tolist()
    state_result = InterarrayCorrelationQCResult.model_validate({"removed_samples": removed_samples})
    return state_result, data_df_filtered
