__all__ = [
    "get_metadata_dataframe",
    "get_available_methylation_dataframe",
    "get_all_methylation_aging_clocks",
    "compute_age_acceleration",
    "get_dataset_predictions",
    "compute_mae",
    "compute_medae",
    "compute_pearson_r",
    "get_extraction_protocol",
    "make_internal_clock_predictions",
]

import json
import os
from collections.abc import Sequence
from functools import reduce
from typing import Any, get_args

import pandas as pd
import pyaging as pya
import statsmodels.api as sm
import torch
from scipy import stats

from ...contracts.clocks import MethylationAgingClock, MethylationClocks
from ...utils.helper import PROJECT_ROOT, load_metadata_aligned_methylation_data, read_feather
from .clock_models import CorticalAge, PCBrainAge


def get_extraction_protocol(accession_code: str, artifacts: list[Any]) -> Any:
    """
    Retrieve the metadata extraction protocol for a given accession code from a list of artifacts.

    Args:
        accession_code (str): The accession code for which to retrieve the extraction protocol.
        artifacts (List[Any]): A list of artifact references containing metadata.

    Returns:
        Any: The metadata extraction protocol.
    """
    extraction_protocol_artifact = next(
        (a for a in artifacts if a.kind == "metadata_extraction_protocol" and a.accession_code == accession_code), None
    )
    if extraction_protocol_artifact is None:
        raise ValueError("No metadata extraction protocol found")
    extraction_protocol = None
    with open(extraction_protocol_artifact.path) as f:
        extraction_protocol = json.load(f)
    return extraction_protocol


def _get_healthy_subset(prediction_df, extraction_protocol):
    """Extract the healthy control subset and return target disease labels.

    Args:
        prediction_df: Prediction DataFrame with "Accession_Code" and
            "Disease_Status" columns.
        extraction_protocol: Dict with
            extraction_protocol["disease_status"]["extraction"]["control_value"].

    Returns:
        Tuple of (accession_code, control_label, healthy_subset, target_labels).
        healthy_subset is the DataFrame filtered to control samples.
        target_labels lists unique non-control disease statuses.
    """
    accession_code = prediction_df["Accession_Code"].unique()[0]
    control_label = extraction_protocol["disease_status"]["extraction"]["control_value"]
    healthy_subset = prediction_df[prediction_df["Disease_Status"] == control_label]
    target_labels = [x for x in prediction_df["Disease_Status"].unique().tolist() if x != control_label]
    return accession_code, control_label, healthy_subset, target_labels


def get_metadata_dataframe(accession_code: str, artifacts: list[Any]) -> pd.DataFrame:
    """
    Retrieve the metadata DataFrame for a given accession code from a list of artifacts.

    Args:
        accession_code (str): The accession code for which to retrieve the metadata.
        artifacts (List[Any]): A list of artifact references containing metadata.

    Returns:
        pd.DataFrame: The metadata DataFrame.
    """
    metadata_artifact = next((a for a in artifacts if a.kind == "dataset_metadata" and a.accession_code == accession_code), None)
    if metadata_artifact is None:
        raise ValueError("No metadata")
    return pd.read_csv(metadata_artifact.path, index_col=0)


def get_available_methylation_dataframe(accession_code: str, artifacts: list[Any]) -> pd.DataFrame:
    """
    Retrieve the available methylation DataFrame for a given accession code from a list of artifacts.

    Args:
        accession_code (str): The accession code for which to retrieve the methylation data.
        artifacts (List[Any]): A list of artifact references containing methylation data.

    Returns:
        pd.DataFrame: The methylation DataFrame.
    """
    # Check for post-quality control data
    postqc_methylation_data = next((a for a in artifacts if a.kind == "postqc_methylation_data" and a.accession_code == accession_code), None)
    if postqc_methylation_data is not None:
        return read_feather(postqc_methylation_data.path, index_name="subject_id")

    # Check for pre-quality control data
    preqc_methylation_data = next((a for a in artifacts if a.kind == "preqc_methylation_data" and a.accession_code == accession_code), None)
    if preqc_methylation_data is not None:
        return load_metadata_aligned_methylation_data(accession_code, artifacts)

    raise ValueError("No methylation data")


def get_dataset_predictions(accession_code: str, artifacts: list[Any]) -> pd.DataFrame:
    """
    Retrieve the dataset predictions for a given accession code from a list of artifacts.

    Args:
        accession_code (str): The accession code for which to retrieve the dataset predictions.
        artifacts (List[Any]): A list of artifact references containing dataset predictions.

    Returns:
        pd.DataFrame: The dataset predictions DataFrame.
    """
    dataset_predictions_artifact = next((a for a in artifacts if a.kind == "dataset_benchmark" and a.accession_code == accession_code), None)
    if dataset_predictions_artifact is not None:
        return pd.read_csv(dataset_predictions_artifact.path, index_col=0)

    raise ValueError("No dataset predictions")


def get_all_methylation_aging_clocks(output_dir: str) -> list[MethylationAgingClock]:
    """
    Retrieve all available methylation aging clocks from the specified output directory.

    Args:
        output_dir (str): The directory containing the clock metadata.

    Returns:
        List[MethylationAgingClock]: A list of available methylation aging clocks.
    """
    methylation_clocks = set()
    pya.utils.show_all_clocks(os.path.join(output_dir))
    clock_metadata = torch.load(os.path.join(output_dir, "all_clock_metadata.pt"), weights_only=False)
    for clock_name in clock_metadata.keys():
        if clock_name in get_args(MethylationClocks):
            # if metadata.get("data_type") == "methylation" and metadata.get("species") == "Homo sapiens":
            methylation_clocks.add(clock_name)
    methylation_clocks = [MethylationAgingClock.model_validate({"clock_name": clock_name}) for clock_name in sorted(list(methylation_clocks))]
    return sorted(methylation_clocks, key=lambda c: c.clock_name.lower())


def compute_age_acceleration(adata: Any, clock_names: Sequence[str]):
    """
    Compute age acceleration for the specified clocks in the given AnnData object.

    Args:
        adata (Any): The AnnData object containing the methylation data.
        clock_names (List[str]): A list of clock names for which to compute age acceleration.

    Returns:
        Any: The AnnData object with age acceleration columns added.
    """
    for clock in clock_names:
        if clock not in adata.obs.columns:
            continue
        # Drop NA for this clock and age
        valid = adata.obs[[clock, "age"]].dropna()
        accel_col = f"{clock}_accel"
        if clock.lower() == "dunedinpace":
            adata.obs.loc[valid.index, accel_col] = adata.obs.loc[valid.index, clock]
            adata.obs[accel_col] = adata.obs[accel_col].astype(float)
            continue
        y = valid[clock]
        X = sm.add_constant(valid["age"])
        model = sm.OLS(y, X).fit()
        residuals = y - model.predict(X)
        # Assign residuals back to the full obs DataFrame
        accel_col = f"{clock}_accel"
        adata.obs[accel_col] = None
        adata.obs.loc[valid.index, accel_col] = residuals
        adata.obs[accel_col] = adata.obs[accel_col].astype(float)
    return adata


def _compute_hc_metric(prediction_df, extraction_protocol, clocks, metric_fn, metric_name):
    """Compute a per-clock metric restricted to healthy control samples.

    Args:
        prediction_df: Prediction DataFrame.
        extraction_protocol: Metadata extraction protocol.
        clocks: List of clock names (or None, which defaults to all).
        metric_fn: Callable(clock_subset, clock_name) -> float.
        metric_name: Column name for the result in the output DataFrame.

    Returns:
        DataFrame with columns ["Accession_Code", "Clock", metric_name].
    """
    accession_code, control_label, healthy_subset, _ = _get_healthy_subset(
        prediction_df,
        extraction_protocol,
    )
    rows = []
    if clocks is None:
        clocks = []
    for clock in clocks:
        if clock not in prediction_df.columns:
            continue
        clock_subset = healthy_subset.dropna(subset=[clock.lower(), "age"])
        if len(clock_subset) < 2:
            continue
        score = metric_fn(clock_subset, clock)
        rows.append(
            {
                "Accession_Code": accession_code,
                "Clock": clock,
                metric_name: score,
            }
        )
    return pd.DataFrame(rows)


def compute_mae(
    prediction_df: pd.DataFrame,
    extraction_protocol: dict[str, Any],
    clocks: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute the mean absolute error (MAE) for each clock.

    Args:
        prediction_df: DataFrame with predictions and metadata.
        extraction_protocol: Metadata extraction protocol.
        clocks: Clock names to analyze (uses all columns if None).

    Returns:
        DataFrame with MAE scores per clock.
    """
    return _compute_hc_metric(
        prediction_df,
        extraction_protocol,
        clocks,
        metric_fn=lambda clock_subset, clock: (abs(clock_subset[clock.lower() + "_accel"] - clock_subset["age"])).mean(),
        metric_name="MAE_score",
    )


def compute_medae(
    prediction_df: pd.DataFrame,
    extraction_protocol: dict[str, Any],
    clocks: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute the median absolute error (MedAE) for each clock.

    Args:
        prediction_df: DataFrame with predictions and metadata.
        extraction_protocol: Metadata extraction protocol.
        clocks: Clock names to analyze (uses all columns if None).

    Returns:
        DataFrame with MedAE scores per clock.
    """
    return _compute_hc_metric(
        prediction_df,
        extraction_protocol,
        clocks,
        metric_fn=lambda clock_subset, clock: (abs(clock_subset[clock.lower() + "_accel"] - clock_subset["age"])).median(),
        metric_name="MedAE_score",
    )


def compute_pearson_r(
    prediction_df: pd.DataFrame,
    extraction_protocol: dict[str, Any],
    clocks: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute the Pearson correlation for each clock vs chronological age.

    Args:
        prediction_df: DataFrame with predictions and metadata.
        extraction_protocol: Metadata extraction protocol.
        clocks: Clock names to analyze (uses all columns if None).

    Returns:
        DataFrame with Pearson_R scores per clock.
    """
    return _compute_hc_metric(
        prediction_df,
        extraction_protocol,
        clocks,
        metric_fn=lambda clock_subset, clock: stats.pearsonr(clock_subset[clock.lower() + "_accel"], clock_subset["age"])[0],
        metric_name="Pearson_R_score",
    )


def _make_pcbrainage_prediction(df, metadata_cols=None, imputer_strategy="knn"):
    """
    Make predictions using the PCBrainAge model.

    Args:
        df (pd.DataFrame): The DataFrame containing the input data.
        metadata_cols (list, optional): A list of metadata columns to exclude from the input data. Defaults to None.
        imputer_strategy (str, optional): The imputation strategy to use. Defaults to 'knn'.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    pcbrainage_model = None
    model_path = os.path.join(str(PROJECT_ROOT), "data", "pcbrainage_model.pkl")
    pcbrainage_model = PCBrainAge.load_state(model_path)
    metadata_cols = metadata_cols or []
    dnam = df[[x for x in df.columns.tolist() if x not in metadata_cols]].copy()
    metadata = df[metadata_cols].copy()
    res = pcbrainage_model.predict(dnam, pheno=metadata, user_imputation=None)
    return res


def _make_corticalage_prediction(df, metadata_cols=None, imputer_strategy="knn"):
    """
    Make predictions using the CorticalAge model.

    Args:
        df (pd.DataFrame): The DataFrame containing the input data.
        metadata_cols (list, optional): A list of metadata columns to exclude from the input data. Defaults to None.
        imputer_strategy (str, optional): The imputation strategy to use. Defaults to 'knn'.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    corticalage_model = None
    model_path = os.path.join(str(PROJECT_ROOT), "data", "corticalage_model.pkl")
    corticalage_model = CorticalAge.load_state(model_path)
    metadata_cols = metadata_cols or []
    dnam = df[[x for x in df.columns.tolist() if x not in metadata_cols]].copy()
    metadata = df[metadata_cols].copy()
    res = corticalage_model.predict(dnam, pheno=metadata, user_imputation=None)
    return res


def make_internal_clock_predictions(df, clocks, metadata_cols=None, imputer_strategy="knn"):
    """
    Make predictions using the specified internal clocks.

    Args:
        df (pd.DataFrame): The DataFrame containing the input data.
        clocks (list): A list of clocks to use for predictions.
        metadata_cols (list, optional): A list of metadata columns to exclude from the input data. Defaults to None.
        imputer_strategy (str, optional): The imputation strategy to use. Defaults to 'knn'.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    results = []
    if "corticalage" in clocks:
        results.append(_make_corticalage_prediction(df, metadata_cols=metadata_cols, imputer_strategy=imputer_strategy))
    if "pcbrainage" in clocks:
        results.append(_make_pcbrainage_prediction(df, metadata_cols=metadata_cols, imputer_strategy=imputer_strategy))
    if len(results) > 1:
        res = reduce(lambda left, right: pd.merge(left, right, on=metadata_cols), results)
    else:
        res = results[0]
    return res
