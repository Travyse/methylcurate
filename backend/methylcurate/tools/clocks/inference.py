__all__ = [
    "get_metadata_dataframe", "get_available_methylation_dataframe",
    "get_all_methylation_aging_clocks", "compute_age_acceleration", "predict_with_aging_clocks", "get_dataset_predictions",
    "compute_mae", "compute_medae", "compute_pearson_r", "bootstrap_aa1_test", "bootstrap_welch_one_sided_aac_gt_hc",
    "merge_and_process_computation_dfs", "get_extraction_protocol", "make_internal_clock_predictions"]

import os
import json
import pickle
import torch
import pyaging as pya
import pandas as pd
from functools import reduce
from typing import List, Any, Tuple, get_args
from scipy import stats
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_absolute_error, median_absolute_error
from .clock_models import CorticalAge, PCBrainAge
from ...contracts.clocks import MethylationAgingClock, PredictionInput, MethylationClocks
from ...utils.helper import read_feather, load_metadata_aligned_methylation_data, PROJECT_ROOT


def get_extraction_protocol(accession_code: str, artifacts: List[Any]) -> Any:
    """
    Retrieve the metadata extraction protocol for a given accession code from a list of artifacts.

    Args:
        accession_code (str): The accession code for which to retrieve the extraction protocol.
        artifacts (List[Any]): A list of artifact references containing metadata.

    Returns:
        Any: The metadata extraction protocol.
    """
    extraction_protocol_artifact = next((a for a in artifacts if a.kind == "metadata_extraction_protocol" and a.accession_code == accession_code), None)
    if extraction_protocol_artifact is None:
        raise ValueError("No metadata extraction protocol found")
    extraction_protocol = None
    with open(extraction_protocol_artifact.path, "r") as f:
        extraction_protocol = json.load(f)
    return extraction_protocol

def get_metadata_dataframe(accession_code: str, artifacts: List[Any]) -> pd.DataFrame:
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

def get_available_methylation_dataframe(accession_code:str, artifacts: List[Any]) -> pd.DataFrame:
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
    
    return ValueError("No methylation data")

def get_dataset_predictions(accession_code: str, artifacts: List[Any]) -> pd.DataFrame:
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
    
    return ValueError("No dataset predictions")

def get_all_methylation_aging_clocks(output_dir: str) -> List[MethylationAgingClock]:
    """
    Retrieve all available methylation aging clocks from the specified output directory.
    
    Args:
        output_dir (str): The directory containing the clock metadata.

    Returns:
        List[MethylationAgingClock]: A list of available methylation aging clocks.
    """
    methylation_clocks = set()
    pya.utils.show_all_clocks(os.path.join(output_dir))
    clock_metadata = torch.load(
        os.path.join(output_dir, "all_clock_metadata.pt"), weights_only=False)
    for clock_name in clock_metadata.keys():
        metadata = clock_metadata[clock_name]
        if clock_name in get_args(MethylationClocks):
        #if metadata.get("data_type") == "methylation" and metadata.get("species") == "Homo sapiens":
            methylation_clocks.add(clock_name)
    methylation_clocks = [
        MethylationAgingClock.model_validate({"clock_name": clock_name}) for clock_name in sorted(list(methylation_clocks))
    ]
    return sorted(methylation_clocks, key=lambda c: c.clock_name.lower())

def compute_age_acceleration(adata: Any, clock_names: List[str]):
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
        valid = adata.obs[[clock, 'age']].dropna()
        accel_col = f"{clock}_accel"
        if clock.lower() == "dunedinpace":
            adata.obs.loc[valid.index, accel_col] = adata.obs.loc[valid.index, clock]
            adata.obs[accel_col] = adata.obs[accel_col].astype(float)
            continue
        y = valid[clock]
        X = sm.add_constant(valid['age'])
        model = sm.OLS(y, X).fit()
        residuals = y - model.predict(X)
        # Assign residuals back to the full obs DataFrame
        accel_col = f"{clock}_accel"
        adata.obs[accel_col] = None
        adata.obs.loc[valid.index, accel_col] = residuals
        adata.obs[accel_col] = adata.obs[accel_col].astype(float)
    return adata

def welch_one_sided_aac_gt_hc(
    df,
    shuffled_labels: list,
    value_col: str,
    control_label: str = "Control",
    group_col: str = "cohort",
    bootstrap_id: int = 1
) -> Tuple[float, float]:
    """
    Two-sample Welch t-test for a single dataset with one-sided alternative:
        H_A: mean(AAC) > mean(HC)

    Returns a dict with summary stats, t, Welch df, and one-sided p-value.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        shuffled_labels (list): The shuffled labels for the bootstrap iteration.
        value_col (str): The column name for the values to be tested.
        control_label (str): The label for the control group.
        group_col (str): The column name for the group labels.
        bootstrap_id (int): The bootstrap iteration ID.

    Returns:
        Tuple[float, float]: The t-statistic and one-sided p-value.
    """
    print(f"Performing AA2 {bootstrap_id}")
    df = df.copy()
    df[group_col] = shuffled_labels
    df[group_col] = df[group_col].apply(lambda x: "Control" if x == control_label else "Other")

    aac = df.loc[df[group_col] == "Other", value_col].dropna().to_numpy()
    hc  = df.loc[df[group_col] == "Control",  value_col].dropna().to_numpy()

    if len(aac) < 2 or len(hc) < 2:
        #raise ValueError(f"Need >=2 non-NA values per group. Got AAC={len(aac)}, HC={len(hc)}")
        return None, None

    # Welch's t-test with one-sided alternative
    res = stats.ttest_ind(aac, hc, equal_var=False, alternative="greater")
    t_stat = float(res.statistic)
    p_one  = float(res.pvalue)
    return t_stat, p_one

def bootstrap_welch_one_sided_aac_gt_hc(
    prediction_df: pd.DataFrame,
    extraction_protocol: Any,
    clocks: list = [],
    n_bootstraps: int = 1000
) -> pd.DataFrame:
    """
    Perform a bootstrap analysis using Welch's one-sided t-test for age acceleration differences between disease and control groups across multiple clocks.

    Args:
        prediction_df (pd.DataFrame): The DataFrame containing the predictions and metadata for the dataset.
        extraction_protocol (Any): The metadata extraction protocol containing information about disease status and control values.
        clocks (list): A list of clock names to analyze.
        n_bootstraps (int): The number of bootstrap iterations to perform.
    Returns:
        pd.DataFrame: A DataFrame containing the results of the bootstrap analysis, including t-stat
    """
    rows = []
    accession_code = prediction_df['Accession_Code'].unique()[0]
    control_label = extraction_protocol["disease_status"]["extraction"]["control_value"]
    target_labels = [x for x in prediction_df["Disease_Status"].unique().tolist() if x != control_label]
    for target_label in target_labels:
        sub_prediction_df = prediction_df[prediction_df["Disease_Status"].isin([control_label, target_label])]

        control_count = sub_prediction_df[sub_prediction_df["Disease_Status"] == control_label].shape[0]
        target_count = sub_prediction_df[sub_prediction_df["Disease_Status"] == target_label].shape[0]
        if any(count < 10 for count in [control_count, target_count]):
            continue

        shuffled_labels = [
            sub_prediction_df["Disease_Status"].sample(frac=1, random_state=i).to_numpy()
            for i in range(1000)
        ]
        for clock in clocks:
            if clock not in sub_prediction_df.columns:
                continue
            clock_t_stats = []

            bootstrapped_results = Parallel(n_jobs=-1)(
                delayed(welch_one_sided_aac_gt_hc)(
                    sub_prediction_df, shuffled_labels[i], f'{clock.lower()}_accel', group_col='Disease_Status',
                    control_label=control_label, bootstrap_id=i) for i in range(n_bootstraps))
            clock_t_stats.extend([t for t, p in bootstrapped_results if t is not None])
        
            aatwo_t_stat, aatwo_p = welch_one_sided_aac_gt_hc(
                sub_prediction_df,
                sub_prediction_df["Disease_Status"].to_numpy(),
                control_label=control_label,
                value_col=f'{clock.lower()}_accel',
                group_col='Disease_Status',
                bootstrap_id=-1)

            rows.append({
                "Accession_Code": accession_code,
                "Disease": target_label,
                "Disease_Group": sub_prediction_df["Disease_Group"].unique()[0],
                "Clock": clock,
                "AA2": aatwo_p,
                "AA2_Empirical_p": (sum(1 for t in clock_t_stats if t >= aatwo_t_stat) + 1)/(len(clock_t_stats) + 1)
            })
    if not rows:
        return pd.DataFrame(columns=["Accession_Code", "Disease", "Disease_Group", "Clock", "AA2", "AA2_Empirical_p"])
    return pd.DataFrame(rows)

def one_sample_t_test(
    df,
    shuffled_labels: list,
    value_col: str,
    control_label: str = "Control",
    group_col: str = "cohort",
    bootstrap_id: int = 1
):
    """
    Two-sample Welch t-test for a single dataset with one-sided alternative:
        H_A: mean(AAC) > mean(HC)

    Returns a dict with summary stats, t, Welch df, and one-sided p-value.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        shuffled_labels (list): The shuffled labels for the bootstrap iteration.
        value_col (str): The column name for the values to be tested.
        control_label (str): The label for the control group.
        group_col (str): The column name for the group labels.
        bootstrap_id (int): The bootstrap iteration ID.
    
    Returns:
        Tuple[float, float]: The t-statistic and one-sided p-value.
    """
    print(f"Performing AA2 {bootstrap_id}")
    mu0 = 0.0
    df = df.copy()
    df[group_col] = shuffled_labels
    # For all values in group_col, if they are not equal to control_label, set value to "Other"
    df[group_col] = df[group_col].apply(lambda x: "Control" if x == control_label else "Other")

    aac = df.loc[df[group_col] == "Other", value_col].dropna().to_numpy()
    if not aac.size > 0:
        return None, None

    t_test_res = stats.ttest_1samp(aac, popmean=mu0, alternative="greater")
    t_stat, p_one_sided = t_test_res.statistic, t_test_res.pvalue

    return t_stat, p_one_sided

def bootstrap_aa1_test(
    prediction_df: pd.DataFrame,
    extraction_protocol: Any,
    clocks: list = [],
    n_bootstraps: int = 1000
) -> pd.DataFrame:
    """
    Perform a bootstrap analysis for AA1 test.

    Args:
        prediction_df (pd.DataFrame): The DataFrame containing the predictions and metadata for the dataset.
        extraction_protocol (Any): The metadata extraction protocol containing information about disease status and control values.
        clocks (list): A list of clock names to analyze.
        n_bootstraps (int): The number of bootstrap iterations to perform.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the bootstrap analysis, including t-statistics and p-values.
    """
    rows = []
    accession_code = prediction_df['Accession_Code'].unique()[0]
    control_label = extraction_protocol["disease_status"]["extraction"]["control_value"]
    target_labels = [x for x in prediction_df["Disease_Status"].unique().tolist() if x != control_label]
    for target_label in target_labels:
        sub_prediction_df = prediction_df[prediction_df["Disease_Status"].isin([control_label, target_label])]

        control_count = sub_prediction_df[sub_prediction_df["Disease_Status"] == control_label].shape[0]
        target_count = sub_prediction_df[sub_prediction_df["Disease_Status"] == target_label].shape[0]
        if any(count < 10 for count in [control_count, target_count]):
            continue

        shuffled_labels = [
            sub_prediction_df["Disease_Status"].sample(frac=1, random_state=i).to_numpy()
            for i in range(1000)
        ]
        
        for clock in clocks:
            if clock not in sub_prediction_df.columns:
                continue
            clock_t_stats = []

            bootstrapped_results = Parallel(n_jobs=-1)(
                delayed(one_sample_t_test)(
                    sub_prediction_df, shuffled_labels[i], f'{clock.lower()}_accel', group_col='Disease_Status',
                    control_label=control_label, bootstrap_id=i) for i in range(n_bootstraps))
            clock_t_stats.extend([t for t, p in bootstrapped_results if t is not None])
        
            aaone_t_stat, aaone_p = one_sample_t_test(
                sub_prediction_df,
                sub_prediction_df["Disease_Status"].tolist(),
                control_label=control_label,
                value_col=f'{clock.lower()}_accel',
                group_col='Disease_Status',
                bootstrap_id=-1)

            rows.append({
                "Accession_Code": accession_code,
                "Disease": target_label,
                "Disease_Group": sub_prediction_df["Disease_Group"].unique()[0],
                "Clock": clock,
                "AA1": aaone_p,
                "AA1_Empirical_p": (sum(1 for t in clock_t_stats if t >= aaone_t_stat) + 1)/(len(clock_t_stats) + 1)
            })
    if not rows:
        return pd.DataFrame(columns=["Accession_Code", "Disease", "Disease_Group", "Clock", "AA1", "AA1_Empirical_p"])
    return pd.DataFrame(rows)

def compute_mae(
    prediction_df: pd.DataFrame,
    extraction_protocol: Any,
    clocks: list = [],
) -> pd.DataFrame:
    """
    Compute the mean absolute error (MAE) for each clock in the dataset.

    Args:
        prediction_df (pd.DataFrame): The DataFrame containing the predictions and metadata for the dataset.
        extraction_protocol (Any): The metadata extraction protocol containing information about disease status and control values.
        clocks (list): A list of clock names to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing the MAE for each clock.
    """
    accession_code = prediction_df['Accession_Code'].unique()[0]
    control_label = extraction_protocol["disease_status"]["extraction"]["control_value"]
    healthy_subset = prediction_df[prediction_df['Disease_Status'] == control_label]
    rows = []
    for clock in clocks:
        if clock not in prediction_df.columns:
            continue
        clock_subset = healthy_subset.dropna(subset=[clock.lower(), 'age'])
        if len(clock_subset) < 2:
            rows.append({
                "Accession_Code": accession_code,
                "Clock": clock,
                "Pearson_R": 0.0
            })
            continue
        age = clock_subset['age']
        clock_pred_age = clock_subset[clock.lower()]
        mae = mean_absolute_error(age, clock_pred_age)
        rows.append({
            "Accession_Code": accession_code,
            "Clock": clock,
            "MAE": mae
        })
    results_df = pd.DataFrame(rows)
    return results_df

def compute_medae(
    prediction_df: pd.DataFrame,
    extraction_protocol: Any,
    clocks: list = [],
) -> pd.DataFrame:
    """
    Compute the median absolute error (MedAE) for each clock in the dataset.

    Args:
        prediction_df (pd.DataFrame): The DataFrame containing the predictions and metadata for the dataset.
        extraction_protocol (Any): The metadata extraction protocol containing information about disease status and control values.
        clocks (list): A list of clock names to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing the MedAE for each clock.
    """
    accession_code = prediction_df['Accession_Code'].unique()[0]
    control_label = extraction_protocol["disease_status"]["extraction"]["control_value"]
    healthy_subset = prediction_df[prediction_df['Disease_Status'] == control_label]
    rows = []
    for clock in clocks:
        if clock not in prediction_df.columns:
            continue
        clock_subset = healthy_subset.dropna(subset=[clock.lower(), 'age'])
        if len(clock_subset) < 2:
            rows.append({
                "Accession_Code": accession_code,
                "Clock": clock,
                "Pearson_R": 0.0
            })
            continue
        age = clock_subset['age']
        clock_pred_age = clock_subset[clock.lower()]
        medae = median_absolute_error(age, clock_pred_age)
        rows.append({
            "Accession_Code": accession_code,
            "Clock": clock,
            "MedAE": medae
        })
    results_df = pd.DataFrame(rows)
    return results_df

def compute_pearson_r(
    prediction_df: pd.DataFrame,
    extraction_protocol: Any,
    clocks: list = [],
) -> pd.DataFrame:
    """
    Compute the Pearson correlation coefficient (R) for each clock in the dataset.

    Args:
        prediction_df (pd.DataFrame): The DataFrame containing the predictions and metadata for the dataset.
        extraction_protocol (Any): The metadata extraction protocol containing information about disease status and control values.
        clocks (list): A list of clock names to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing the Pearson R for each clock.
    """
    accession_code = prediction_df['Accession_Code'].unique()[0]
    control_label = extraction_protocol["disease_status"]["extraction"]["control_value"]
    healthy_subset = prediction_df[prediction_df['Disease_Status'] == control_label]
    rows = []
    for clock in clocks:
        if clock not in prediction_df.columns:
            continue
        clock_subset = healthy_subset.dropna(subset=[clock.lower(), 'age'])
        # If there are less than 2 samples, skip this clock for this dataset
        if len(clock_subset) < 2:
            rows.append({
                "Accession_Code": accession_code,
                "Clock": clock,
                "Pearson_R": 0.0
            })
            continue
        age = clock_subset['age']
        clock_pred_age = clock_subset[clock.lower()]
        pearson_r, _ = stats.pearsonr(age, clock_pred_age)
        rows.append({
            "Accession_Code": accession_code,
            "Clock": clock,
            "Pearson_R": pearson_r
        })
    results_df = pd.DataFrame(rows)
    return results_df

def merge_and_process_computation_dfs(
    dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge and process multiple computation DataFrames, adjusting p-values and computing scores.

    Args:
        dfs (List[pd.DataFrame]): A list of DataFrames containing computation results.

    Returns:
        pd.DataFrame: A merged and processed DataFrame with adjusted p-values and scores.
    """

    full_df = pd.concat(dfs, ignore_index=True, axis=0)
    print(f"\nFull merged df before p-value adjustment and scoring: \n{full_df.head()}\n")
    # Adjust p-values for AA1 and AA2 across all clocks and datasets
    full_df['AA1'] = full_df.groupby(['Accession_Code', 'Disease'])['AA1'].transform(
        lambda x: multipletests(x, method='fdr_bh')[1]
    )
    full_df['AA1_Empirical_p'] = full_df.groupby(['Accession_Code', 'Disease'])['AA1_Empirical_p'].transform(
        lambda x: multipletests(x, method='fdr_bh')[1]
    )
    full_df['AA2'] = full_df.groupby(['Accession_Code', 'Disease'])['AA2'].transform(
        lambda x: multipletests(x, method='fdr_bh')[1]
    )
    full_df['AA2_Empirical_p'] = full_df.groupby(['Accession_Code', 'Disease'])['AA2_Empirical_p'].transform(
        lambda x: multipletests(x, method='fdr_bh')[1]
    )

    # Compute Score
    full_df['AA1_score'] = full_df.groupby(['Clock', 'Disease'])['AA1'].transform(lambda x: (x < 0.05).sum())
    aa1_emp_by_clock = full_df.groupby(['Clock', 'Disease']).apply(
        lambda g: ((g['AA1'] < 0.05) & (g['AA1_Empirical_p'] < 0.05)).sum()
    ).reset_index(name='AA1_emp_score')
    full_df = full_df.merge(aa1_emp_by_clock, on=['Clock', 'Disease'], how='left')
    full_df['AA2_score'] = full_df.groupby(['Clock', 'Disease'])['AA2'].transform(lambda x: (x < 0.05).sum())
    aa2_emp_by_clock = full_df.groupby(['Clock', 'Disease']).apply(
        lambda g: ((g['AA2'] < 0.05) & (g['AA2_Empirical_p'] < 0.05)).sum()
    ).reset_index(name='AA2_emp_score') 
    full_df = full_df.merge(aa2_emp_by_clock, on=['Clock', 'Disease'], how='left')
    full_df['MAE_score'] = full_df.groupby('Clock')['MAE'].transform('mean')
    full_df['MedAE_score'] = full_df.groupby('Clock')['MedAE'].transform('mean')
    full_df['Pearson_R_score'] = full_df.groupby('Clock')['Pearson_R'].transform('mean')
    full_df['Overall_score'] = full_df[['AA1_emp_score', 'AA2_emp_score']].sum(axis=1)
    return full_df

def _make_pcbrainage_prediction(df, metadata_cols=None, imputer_strategy='knn'):
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
    dnam = df[[x for x in df.columns.tolist() if x not in metadata_cols]].copy()
    metadata = df[metadata_cols].copy()
    res = pcbrainage_model.predict(dnam, pheno=metadata, user_imputation=None)
    return res

def _make_corticalage_prediction(df, metadata_cols=None, imputer_strategy='knn'):
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
    dnam = df[[x for x in df.columns.tolist() if x not in metadata_cols]].copy()
    metadata = df[metadata_cols].copy()
    res = corticalage_model.predict(dnam, pheno=metadata, user_imputation=None)
    return res

def make_internal_clock_predictions(df, clocks, metadata_cols=None, imputer_strategy='knn'):
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
    if 'corticalage' in clocks:
        results.append(
            _make_corticalage_prediction(df, metadata_cols=metadata_cols, imputer_strategy=imputer_strategy))
    if 'pcbrainage' in clocks:
        results.append(
            _make_pcbrainage_prediction(df, metadata_cols=metadata_cols, imputer_strategy=imputer_strategy))
    if len(results) > 1:
        res = reduce(lambda left, right: pd.merge(left, right, on=metadata_cols), results)
    else:
        res = results[0]
    return res
