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
from ...contracts.preprocess import (
    SampleLevelQCInput,
    SampleLevelQCResult,
    CpGLevelQCInput,
    CpGLevelQCResult,
    DNAmQCInput,
    DNAmQCResult,
    InterarrayCorrelationQCInput,
    InterarrayCorrelationQCResult
)

def handle_sample_level_missingness(qc_input: Any, data_df: pd.DataFrame) -> Tuple[SampleLevelQCResult, pd.DataFrame]:
    sample_missing_rate = data_df.isnull().sum(axis=1) / len(data_df.columns)
    samples_to_keep = sample_missing_rate <= qc_input.missing_cutoff
    data_df_filtered = data_df[samples_to_keep].copy()
    removed_samples = data_df.index[~samples_to_keep].tolist()
    state_result = SampleLevelQCResult.model_validate({
        "removed_samples": removed_samples})
    return state_result, data_df_filtered

def handle_cpg_level_missingness(state: CpGLevelQCInput, data_df: pd.DataFrame) -> Tuple[CpGLevelQCResult, pd.DataFrame]:
    if data_df.empty:
        return CpGLevelQCResult.model_validate({
            "removed_cpgs": [],
            "missing_before_imputation": 0.0
        }), data_df
    cpg_missing_rate = data_df.isnull().sum(axis=0) / len(data_df)
    cpgs_to_keep = cpg_missing_rate <= state.missing_cutoff
    high_quality_cpgs = data_df.columns[cpgs_to_keep].tolist()
    removed_cpgs = data_df.columns[~cpgs_to_keep].tolist()
    data_df_filtered = data_df[high_quality_cpgs].copy()
    overall_missing_rate = data_df_filtered.isna().to_numpy().mean()
    
    if state.imputation_strategy.imputation_model.concept == "knn":
        imputer = KNNImputer(
            n_neighbors=state.imputation_strategy.imputation_model.n_neighbors,
            weights=state.imputation_strategy.imputation_model.weights)
    else:
        imputer = SimpleImputer(strategy=state.imputation_strategy.imputation_model.strategy)

    data_df_filtered.loc[:, high_quality_cpgs] = imputer.fit_transform(data_df_filtered[high_quality_cpgs])
    state_result = CpGLevelQCResult.model_validate({
        "removed_cpgs": removed_cpgs,
        "missing_before_imputation": overall_missing_rate
    })
    return state_result, data_df_filtered

def maximum_dnam_filter(state: DNAmQCInput, data_df: pd.DataFrame) -> Tuple[DNAmQCResult, pd.DataFrame]:
    if data_df.empty:
        return DNAmQCResult.model_validate({
            "removed_samples": []
        }), data_df
    max_dnam_per_sample = np.nanmax(data_df.values, axis=1)
    dnam_filter = max_dnam_per_sample > state.dnam_cutoff
    data_df_filtered = data_df[dnam_filter].copy()
    removed_samples = data_df.index[~dnam_filter].tolist()
    state_result = DNAmQCResult.model_validate({
        "removed_samples": removed_samples})
    return state_result, data_df_filtered

def interarray_correlation(state: InterarrayCorrelationQCInput, data_df: pd.DataFrame) -> Tuple[InterarrayCorrelationQCResult, pd.DataFrame]:
    if data_df.empty:
        return InterarrayCorrelationQCResult.model_validate({
            "removed_samples": []
        }), data_df
    corr_matrix = np.corrcoef(data_df.values)
    mean_inter_corr = (np.sum(corr_matrix, axis = 1) - 1) / (data_df.shape[0] - 1)
    corr_filter = mean_inter_corr > state.correlation_cutoff
    data_df_filtered = data_df[corr_filter].copy()
    removed_samples = data_df.index[~corr_filter].tolist()
    state_result = InterarrayCorrelationQCResult.model_validate({
        "removed_samples": removed_samples})
    return state_result, data_df_filtered
