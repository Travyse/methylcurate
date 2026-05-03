__all__ = []
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Type
from sklearn.impute import SimpleImputer, KNNImputer
from ...contracts.qc import ImputerModelInput, ImputationInput

ESTIMATOR_REGISTRY: Dict[str, Type] = {
    "simple": SimpleImputer,
    "knn": KNNImputer,
}

def _impute_per_split(
        reference_df: pd.DataFrame, target_df: pd.DataFrame, reference_metadata_df: pd.DataFrame,
        target_metadata_df: pd.DataFrame, imputer_input: ImputerModelInput, split_strategy: ImputationInput) -> pd.DataFrame:
    """Impute missing values within strata defined by a metadata column.

    One imputer is fitted and applied **per stratum** so that
    imputation respects the grouping (e.g. sex or tissue type).

    Args:
        reference_df: Reference methylation matrix used for fitting.
        target_df: Target methylation matrix to be imputed.
        reference_metadata_df: Metadata for the reference samples.
        target_metadata_df: Metadata for the target samples.
        imputer_input: Specification of the imputer model and its
            hyper-parameters.
        split_strategy: Stratification strategy including the column
            name used for splitting (``stratify_by``).

    Returns:
        A ``DataFrame`` with the same shape as *target_df* where
        missing entries have been replaced by imputed values.
    """
    imputer_kwargs = imputer_input.model_dump()
    concept = imputer_kwargs.pop("concept")
    unique_split_values = list(target_metadata_df[split_strategy.stratify_by].unique())
    imputers = {
        split_value: ESTIMATOR_REGISTRY[concept](**imputer_kwargs)
        for split_value in unique_split_values
    }
    imputed_target_dfs = []
    for split_value in unique_split_values:
        ref_indices = reference_metadata_df[
            reference_metadata_df[split_strategy.stratify_by] == split_value
        ].index
        tgt_indices = target_metadata_df[
            target_metadata_df[split_strategy.stratify_by] == split_value
        ].index
        ref_split_df = reference_df.loc[ref_indices]
        tgt_split_df = target_df.loc[tgt_indices]
        imputers[split_value].fit(ref_split_df.values)
        imputed_values = imputers[split_value].transform(tgt_split_df.values)
        imputed_split_df = pd.DataFrame(imputed_values, index=tgt_split_df.index, columns=tgt_split_df.columns)
        imputed_target_dfs.append(imputed_split_df)
    imputed_target_df = pd.concat(imputed_target_dfs, axis=0)
    return imputed_target_df

def _impute_whole(
        reference_df: pd.DataFrame, target_df: pd.DataFrame, imputer_input: ImputerModelInput) -> pd.DataFrame:
    """Impute missing values across the entire dataset without stratification.

    A single imputer is fitted on the reference and applied to the target.

    Args:
        reference_df: Reference methylation matrix used for fitting.
        target_df: Target methylation matrix to be imputed.
        imputer_input: Specification of the imputer model and its
            hyper-parameters.

    Returns:
        A ``DataFrame`` with the same shape as *target_df* where
        missing entries have been replaced by imputed values.
    """
    imputer_kwargs = imputer_input.model_dump()
    concept = imputer_kwargs.pop("concept")
    imputer = ESTIMATOR_REGISTRY[concept](**imputer_kwargs)
    imputer.fit(reference_df.values)
    imputed_values = imputer.transform(target_df.values)
    imputed_target_df = pd.DataFrame(imputed_values, index=target_df.index, columns=target_df.columns)
    return imputed_target_df
    

def impute_missing_values(
        reference_df: pd.DataFrame, target_df: pd.DataFrame, reference_metadata_df: pd.DataFrame,
        target_metadata_df: pd.DataFrame, imputer_input: ImputerModelInput, split_strategy: ImputationInput) -> pd.DataFrame:
    """Impute missing methylation values with optional stratification.

    Dispatches to whole-dataset or per-stratum imputation based on the
    chosen strategy.

    Args:
        reference_df: Reference methylation matrix used for fitting.
        target_df: Target methylation matrix to be imputed.
        reference_metadata_df: Metadata for the reference samples.
        target_metadata_df: Metadata for the target samples.
        imputer_input: Specification of the imputer model
            (``"simple"`` or ``"knn"``) and its hyper-parameters.
        split_strategy: Imputation strategy (``"whole"`` or
            ``"stratify"``) and, when ``"stratify"``, the metadata
            column to partition on.

    Returns:
        A ``DataFrame`` with the same shape as *target_df* where
        missing entries have been replaced by imputed values.
    """
    if split_strategy.strategy == "stratify":
        imputed_target_df = _impute_per_split(
            reference_df, target_df, reference_metadata_df, target_metadata_df,
            imputer_input, split_strategy)
    else:
        imputed_target_df = _impute_whole(
            reference_df, target_df, imputer_input)
    return imputed_target_df