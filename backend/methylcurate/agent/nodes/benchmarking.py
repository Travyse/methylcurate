__all__ = ["clock_retrieval_node", "benchmarking_node", "task_computation_node", "summarize_benchmarking_results"]

import hashlib
import json
import os
from datetime import UTC, datetime

import pandas as pd
import pyaging as pya
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from ...contracts.common import ArtifactRef
from ...tools.clocks.inference import (
    bootstrap_aa1_test,
    bootstrap_welch_one_sided_aac_gt_hc,
    compute_age_acceleration,
    compute_mae,
    compute_medae,
    compute_pearson_r,
    get_dataset_predictions,
    get_extraction_protocol,
    get_metadata_dataframe,
    make_internal_clock_predictions,
    merge_and_process_computation_dfs,
)
from ...utils.helper import (
    PROJECT_ROOT,
    benchmarking_progress,
    compute_sha256,
    consolidate_artifacts,
    get_accession_codes,
    load_metadata_aligned_methylation_data,
)
from ...utils.logging import setup_logger
from ..state.models import BenchmarkingSubgraphState


def _get_harmonized_metadata(accession_code: str, artifacts: list[ArtifactRef]) -> pd.DataFrame:
    """
    Retrieve harmonized metadata for a given accession code from a list of artifacts.

    Args:
        accession_code (str): The accession code for which to retrieve harmonized metadata.
        artifacts (List[ArtifactRef]): A list of artifact references containing metadata.

    Returns:
        pd.DataFrame: A DataFrame containing the harmonized metadata.
    """
    target_dataframe_kinds = ["disease_harmonization_mapping", "tissue_harmonization_mapping"]
    target_json_kinds = ["sex_label_harmonization", "cell_type_label_guessing", "cell_type_label_harmonization"]
    dfs = []
    for kind in target_dataframe_kinds:
        artifact = next((a for a in artifacts if a.kind == kind and a.accession_code == accession_code), None)
        if artifact:
            df = pd.read_csv(artifact.path, index_col=0)
            dfs.append(df)
    for kind in target_json_kinds:
        artifact = next((a for a in artifacts if a.kind == kind and a.accession_code == accession_code), None)
        if artifact:
            with open(artifact.path) as f:
                mapping = json.load(f)
            for m in mapping["mappings"]:
                if m.get("target_label", None) is None:
                    pass
            df = pd.DataFrame(
                [
                    {
                        "original_label": m["source_label"],
                        "harmonized_label": m.get("target_label", m["source_label"]),
                        "harmonized_group_label": m.get("target_label", m["source_label"]),
                    }
                    for m in mapping["mappings"]
                ]
            )
            dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def _get_harmonized_full_data_if_available(
    accession_code: str, metadata_df: pd.DataFrame, methylation_df: pd.DataFrame, artifacts: list[ArtifactRef]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieve harmonized full data for a given accession code if available.

    Args:
        accession_code (str): The accession code for which to retrieve harmonized data.
        metadata_df (pd.DataFrame): The metadata DataFrame.
        methylation_df (pd.DataFrame): The methylation DataFrame.
        artifacts (List[ArtifactRef]): A list of artifact references containing metadata.

    Returns:
        pd.DataFrame: A DataFrame containing the harmonized full data.
    """
    harmonized_metadata = _get_harmonized_metadata(accession_code, artifacts)
    if harmonized_metadata.empty:
        merged = metadata_df.merge(methylation_df, left_index=True, right_index=True)
        return merged, merged.columns.tolist()
    individual_mapper = {}
    group_mapper = {}
    for _, row in harmonized_metadata.iterrows():
        individual_mapper[row["original_label"]] = row["harmonized_label"]
        group_mapper[row["original_label"]] = row["harmonized_group_label"]
    metadata_df["Disease_Group"] = metadata_df["Disease_Status"].map(group_mapper).fillna(metadata_df["Disease_Status"])
    metadata_df["Disease_Status"] = (
        metadata_df["Disease_Status"].map(individual_mapper).fillna(metadata_df["Disease_Status"])
    )
    metadata_df["Tissue_Group"] = metadata_df["Tissue"].map(group_mapper).fillna(metadata_df["Tissue"])
    metadata_df["Tissue"] = metadata_df["Tissue"].map(individual_mapper).fillna(metadata_df["Tissue"])
    metadata_df["Cell_Type"] = metadata_df["Cell_Type"].map(individual_mapper).fillna(metadata_df["Cell_Type"])
    metadata_df["Sex"] = metadata_df["Sex"].map(individual_mapper).fillna(metadata_df["Sex"])
    metadata_df["female"] = (metadata_df["Sex"] == "Female").astype(int)
    metadata_df.drop(columns=["Subject"], inplace=True, errors="ignore")
    metadata_columns = metadata_df.columns.tolist()
    return metadata_df.merge(methylation_df, left_index=True, right_index=True), metadata_columns


def clock_retrieval_node(state: BenchmarkingSubgraphState, config: RunnableConfig) -> Command:
    """
    Retrieve and validate clock models for benchmarking.

    Args:
        state (BenchmarkingSubgraphState): The current state of the benchmarking subgraph.
        config (RunnableConfig): The configuration for the runnable.

    Returns:
        Command: A command containing the updated state and any messages.
    """
    return_dict = {"config": state.config.model_dump()}
    artifacts = []
    clock_dir = os.path.join(state.config.output_root, "clocks")
    pyalogger = pya.logger.LoggerManager.gen_logger("predict_age")
    logger = setup_logger(clock_dir, "clock_retrieval_node", "clock_retrieval.log")
    logger.info("Starting clock retrieval node...")
    logger.info(f"Current clock_dir: {clock_dir}")
    clocks = sorted(state.config.clock_list)
    clocks_to_retrieve = [
        clock
        for clock in clocks
        if not any(
            a
            for a in state.config.artifacts
            if a.kind == "clock"
            and (a.path.endswith(f"{clock.lower()}.pt") or a.path.endswith(f"{clock.lower()}_model.pkl"))
        )
    ]
    if not clocks_to_retrieve:
        return Command(update={})
    logger.info(f"Clocks to retrieve: {clocks_to_retrieve}")

    clock = clocks_to_retrieve[0]
    logger.info(f"Retrieving clock: {clock}")
    if clock.lower() in {"corticalage", "pcbrainage"}:
        model_dir = os.path.join(str(PROJECT_ROOT), "data")
        logger.info(f"Looking for {clock.lower()} in {model_dir}")
        artifacts.append(
            ArtifactRef.model_validate(
                {
                    "path": os.path.join(model_dir, f"{clock.lower()}_model.pkl"),
                    "kind": "clock",
                    "accession_code": None,
                    "sha256": compute_sha256(os.path.join(model_dir, f"{clock.lower()}_model.pkl"), is_path=True),
                    "bytes": os.path.getsize(os.path.join(model_dir, f"{clock.lower()}_model.pkl")),
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
        )
        return_dict["config"]["artifacts"] = consolidate_artifacts(state.config.artifacts, artifacts)
        return_dict["messages"] = [benchmarking_progress(state)]
        return_dict["main_messages"] = [benchmarking_progress(state)]
        return Command(update=return_dict)
    try:
        pya.predict._pred_utils.load_clock(clock.lower(), "cpu", clock_dir, pyalogger, indent_level=2)  # type: ignore
        artifacts.append(
            ArtifactRef.model_validate(
                {
                    "path": os.path.join(clock_dir, f"{clock.lower()}.pt"),
                    "kind": "clock",
                    "accession_code": None,
                    "sha256": compute_sha256(os.path.join(clock_dir, f"{clock.lower()}.pt"), is_path=True),
                    "bytes": os.path.getsize(os.path.join(clock_dir, f"{clock.lower()}.pt")),
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
        )
    except Exception as e:
        logger.error(f"Error loading clock {clock} from pre-existing inference: {e}")

    if artifacts:
        return_dict["config"]["artifacts"] = consolidate_artifacts(state.config.artifacts, artifacts)
    return_dict["messages"] = [benchmarking_progress(state)]
    return_dict["main_messages"] = [benchmarking_progress(state)]
    return Command(update=return_dict)


def benchmarking_node(state: BenchmarkingSubgraphState, config: RunnableConfig) -> Command:
    """
    Perform benchmarking for a given state and configuration.

    Args:
        state (BenchmarkingSubgraphState): The current state of the benchmarking subgraph.
        config (RunnableConfig): The configuration for the runnable.

    Returns:
        Command: A command containing the updated state and any messages.
    """
    # Self-looping
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted(
        [
            accession_code
            for accession_code in accession_codes
            if state.datasets[accession_code].steps["make_predictions"].status in {"running", "not_started"}
        ]
    )
    if not running_accession_codes:
        return Command(update={})

    accession_code = running_accession_codes[0]
    output_dir = os.path.join(state.config.output_root, accession_code)
    logger = setup_logger(output_dir, f"{accession_code}_benchmark", "benchmark.log")
    os.makedirs(output_dir, exist_ok=True)
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {accession_code: state.datasets[accession_code].model_dump()},
    }
    methylation_metadata_col = ["Subject"]
    clock_list = state.config.clock_list
    internal_clock_list = [x.lower() for x in clock_list if x.lower() in {"corticalage", "pcbrainage"}]
    clock_list = [x for x in clock_list if x not in internal_clock_list]

    try:
        metadata_df = get_metadata_dataframe(accession_code, state.config.artifacts)
        logger.info(f"\nMetadata data:\n {metadata_df.head().to_markdown(index=False)}")
        methylation_df = load_metadata_aligned_methylation_data(
            accession_code, state.config.artifacts, methylation_data_type="postqc_methylation_data"
        )
        logger.info(
            f"\nMethylation data:\n {methylation_df[methylation_df.columns.tolist()[:20]].head().to_markdown(index=True)}"
        )
    except Exception as e:
        logger.error(f"Error loading data for accession code {accession_code}: {e}")
        return_dict["datasets"][accession_code]["steps"]["make_predictions"]["status"] = "completed"
        return_dict["datasets"][accession_code]["steps"]["make_computations"]["status"] = "canceled"
        return Command(update=return_dict)
    if methylation_df.empty:
        logger.error(
            f"No methylation data found for accession code {accession_code}. Skipping benchmarking for this dataset."
        )
        return_dict["datasets"][accession_code]["steps"]["make_predictions"]["status"] = "completed"
        return_dict["datasets"][accession_code]["steps"]["make_computations"]["status"] = "canceled"
        return Command(update=return_dict)
    cpg_columns = [col for col in methylation_df.columns if col.startswith("cg")]
    if not cpg_columns:
        logger.error(
            f"No CpG columns found in methylation data for accession code {accession_code}. Skipping benchmarking for this dataset."
        )
        return_dict["datasets"][accession_code]["steps"]["make_predictions"]["status"] = "completed"
        return_dict["datasets"][accession_code]["steps"]["make_computations"]["status"] = "canceled"
        return Command(update=return_dict)
    if "Subject" not in methylation_df.columns.tolist():
        methylation_metadata_col = []
    full_data, metadata_columns = _get_harmonized_full_data_if_available(
        accession_code, metadata_df, methylation_df, state.config.artifacts
    )

    if full_data["Sex"].isnull().any():
        if set([x.lower() for x in clock_list]).intersection({"grimage", "grimage2", "pcgrimage"}):
            clock_list = [x for x in clock_list if x.lower() not in {"grimage", "grimage2", "pcgrimage"}]
        full_data.drop(columns=["Sex", "female"], inplace=True, errors="ignore")

    logger.info(
        f"\nDataset after merging metadata and methylation data: {full_data[full_data.columns.tolist()[:30]].head().to_markdown(index=True)}"
    )
    adata = pya.pp.df_to_adata(
        full_data, metadata_cols=metadata_columns + methylation_metadata_col, imputer_strategy="knn", verbose=False
    )
    pya.pred.predict_age(adata, clock_list, dir=os.path.join(state.config.output_root, "clocks"), verbose=False)  # type: ignore
    if internal_clock_list:
        internal_clock_predictions = make_internal_clock_predictions(
            full_data,
            internal_clock_list,
            metadata_cols=metadata_columns + methylation_metadata_col,
            imputer_strategy="knn",
        )
        adata.obs = adata.obs.merge(  # type: ignore
            internal_clock_predictions, on=metadata_columns + methylation_metadata_col, how="left"
        )
    adata = compute_age_acceleration(adata, state.config.clock_list)

    prediction_output_path = os.path.join(output_dir, "predictions.csv")
    adata.obs.to_csv(prediction_output_path, index=True)
    artifact = ArtifactRef.model_validate(
        {
            "path": prediction_output_path,
            "kind": "dataset_benchmark",
            "accession_code": accession_code,
            "sha256": compute_sha256(prediction_output_path, is_path=True),
            "bytes": os.path.getsize(prediction_output_path),
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    return_dict["config"]["artifacts"] = consolidate_artifacts(state.config.artifacts, [artifact])
    return_dict["datasets"][accession_code]["steps"]["make_predictions"]["status"] = "completed"
    return_dict["messages"] = [benchmarking_progress(state)]
    return_dict["main_messages"] = [benchmarking_progress(state)]
    return Command(update=return_dict)


def task_computation_node(state: BenchmarkingSubgraphState, config: RunnableConfig) -> Command:
    """
    Perform computations for a given state and configuration.

    Args:
        state (BenchmarkingSubgraphState): The current state of the benchmarking subgraph.
        config (RunnableConfig): The configuration for the runnable.

    Returns:
        Command: A command containing the updated state and any messages.
    """
    # Self-looping
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted(
        [
            accession_code
            for accession_code in accession_codes
            if state.datasets[accession_code].steps["make_computations"].status in {"running", "not_started"}
        ]
    )
    if not running_accession_codes:
        return Command(update={})
    accession_code = running_accession_codes[0]
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {accession_code: state.datasets[accession_code].model_dump()},
    }
    logger = setup_logger(
        os.path.join(state.config.output_root, accession_code), f"{accession_code}_tasks", "tasks.log"
    )
    for a in state.config.artifacts:
        logger.info(f"\nArtifact: {a.model_dump()}\n")
    extraction_protocol = get_extraction_protocol(accession_code, state.config.artifacts)
    prediction_df = get_dataset_predictions(accession_code, state.config.artifacts)
    control_label = extraction_protocol["disease_status"]["extraction"]["control_value"]
    control_count = prediction_df[prediction_df["Disease_Status"] == control_label].shape[0]
    other_count = prediction_df[prediction_df["Disease_Status"] != control_label].shape[0]
    if not (control_count > 2) and not (other_count > 2):
        return_dict["datasets"][accession_code]["steps"]["make_computations"]["status"] = "canceled"
        return_dict["datasets"][accession_code]["status"] = "completed"
        return Command(update=return_dict)

    computation_output_path = os.path.join(state.config.output_root, accession_code, "benchmarking_results.csv")
    mae = compute_mae(prediction_df, extraction_protocol, clocks=state.config.clock_list)
    medae = compute_medae(prediction_df, extraction_protocol, clocks=state.config.clock_list)
    pearson_r = compute_pearson_r(prediction_df, extraction_protocol, clocks=state.config.clock_list)
    aa1_task = bootstrap_aa1_test(prediction_df, extraction_protocol, clocks=state.config.clock_list, n_bootstraps=1000)
    aa2_task = bootstrap_welch_one_sided_aac_gt_hc(
        prediction_df, extraction_protocol, clocks=state.config.clock_list, n_bootstraps=1000
    )

    all_results = aa1_task.merge(aa2_task, on=["Accession_Code", "Clock", "Disease", "Disease_Group"], how="outer")
    for res in [mae, medae, pearson_r]:
        all_results = all_results.merge(res, on=["Accession_Code", "Clock"], how="outer")

    all_results.to_csv(computation_output_path, index=True)
    artifact = ArtifactRef.model_validate(
        {
            "path": computation_output_path,
            "kind": "benchmark_summary",
            "accession_code": accession_code,
            "sha256": compute_sha256(computation_output_path, is_path=True),
            "bytes": os.path.getsize(computation_output_path),
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    return_dict["config"]["artifacts"] = consolidate_artifacts(state.config.artifacts, [artifact])
    return_dict["datasets"][accession_code]["steps"]["make_computations"]["status"] = "completed"
    return_dict["datasets"][accession_code]["status"] = "completed"
    return_dict["messages"] = [benchmarking_progress(state)]
    return_dict["main_messages"] = [benchmarking_progress(state)]
    return Command(update=return_dict)


def summarize_benchmarking_results(state: BenchmarkingSubgraphState, config: RunnableConfig) -> Command:
    """
    Summarize benchmarking results for a given state and configuration.

    Args:
        state (BenchmarkingSubgraphState): The current state of the benchmarking subgraph.
        config (RunnableConfig): The configuration for the runnable.

    Returns:
        Command: A command containing the updated state and any messages.
    """

    def _create_hash(accession_codes: list[str], identifier: str):
        m = hashlib.sha256()
        for accession_code in sorted(accession_codes):
            m.update(accession_code.encode("utf-8"))
        m.update(b"Benchmarking")
        m.update(identifier.encode("utf-8"))
        return m.hexdigest()

    predictive_performance_payload = {
        "id": "predictive-quality-control-summary",
        "rowIdKey": "accession_code",
        "columns": [
            {
                "key": "clock",
                "label": "Clock",
                "priority": "primary",
                "align": "left",
            },
            {
                "key": "medae",
                "label": "MedAE",
                "priority": "primary",
                "align": "right",
                "format": {"kind": "number", "decimals": 2},
            },
            {
                "key": "pearson_r",
                "label": "Pearson R",
                "priority": "primary",
                "align": "right",
                "format": {"kind": "number", "decimals": 2},
            },
        ],
        "rows": [],
    }
    aa1_payload = {
        "id": "aa1-quality-control-summary",
        "rowIdKey": "accession_code",
        "columns": [
            {
                "key": "clock",
                "label": "Clock",
                "priority": "primary",
                "align": "left",
            },
            {
                "key": "dx",
                "label": "Disease",
                "priority": "primary",
                "align": "left",
            },
            {
                "key": "aa1",
                "label": "AA1",
                "priority": "primary",
                "align": "right",
                "format": {"kind": "number", "decimals": 2},
            },
            {
                "key": "aa1_shuffled",
                "label": "AA1 Shuffled",
                "priority": "primary",
                "align": "right",
                "format": {"kind": "number", "decimals": 2},
            },
        ],
        "rows": [],
    }
    aa2_payload = {
        "id": "quality-control-summary",
        "rowIdKey": "accession_code",
        "columns": [
            {
                "key": "clock",
                "label": "Clock",
                "priority": "primary",
                "align": "left",
            },
            {
                "key": "dx",
                "label": "Disease",
                "priority": "primary",
                "align": "left",
            },
            {
                "key": "aa2",
                "label": "AA2",
                "priority": "primary",
                "align": "right",
                "format": {"kind": "number", "decimals": 2},
            },
            {
                "key": "aa2_shuffled",
                "label": "AA2 Shuffled",
                "priority": "primary",
                "align": "right",
                "format": {"kind": "number", "decimals": 2},
            },
        ],
        "rows": [],
    }
    logger = setup_logger(state.config.output_root, "overall_benchmark_summary", "benchmark_summary.log")
    accession_codes = sorted(
        [x for x in state.datasets.keys() if state.datasets[x].steps["make_computations"].status == "completed"]
    )
    computation_dfs = []
    for accession_code in accession_codes:
        computation_artifact = next(
            (a for a in state.config.artifacts if a.kind == "benchmark_summary" and a.accession_code == accession_code),
            None,
        )
        if computation_artifact is None:
            continue
        computation_df = pd.read_csv(computation_artifact.path, index_col=0)
        computation_dfs.append(computation_df)

    computation_df = merge_and_process_computation_dfs(computation_dfs)
    # performance
    performance_subset = computation_df.copy().drop_duplicates(subset=["Clock"])
    for _, row in performance_subset.iterrows():
        predictive_performance_payload["rows"].append(
            {"clock": row["Clock"], "medae": row["MedAE_score"], "pearson_r": row["Pearson_R_score"]}
        )

    # task
    task_subset = computation_df.copy().drop_duplicates(subset=["Clock", "Disease"])
    task_subset = task_subset[task_subset["Disease"] != "Control"]
    # aa1
    for _, row in task_subset.dropna(subset=["AA1_score", "AA1_emp_score"]).iterrows():
        aa1_payload["rows"].append(
            {"clock": row["Clock"], "dx": row["Disease"], "aa1": row["AA1_score"], "aa1_shuffled": row["AA1_emp_score"]}
        )

    # aa2
    for _, row in task_subset.dropna(subset=["AA2_score", "AA2_emp_score"]).iterrows():
        aa2_payload["rows"].append(
            {"clock": row["Clock"], "dx": row["Disease"], "aa2": row["AA2_score"], "aa2_shuffled": row["AA2_emp_score"]}
        )

    # predictive_message
    predictive_performance_payload["message"] = (
        f"We benchmarked {len(state.config.clock_list)} clocks on {len(accession_codes)} datasets from GEO. Here is a summary of their performance:"
    )
    pp_hash = _create_hash(accession_codes, "predictive_performance")
    pp_message = ToolMessage(
        id=pp_hash,
        content=predictive_performance_payload["message"],
        tool_call_id=pp_hash,
        artifact=predictive_performance_payload,
        additional_kwargs={
            "name": "geoDatasetSummary",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )

    aa1_payload["message"] = (
        f"Here is a summary of AA1 performance across the {len(state.config.clock_list)} clocks and {len(accession_codes)} datasets we benchmarked:"
    )
    aa1_hash = _create_hash(accession_codes, "aa1_performance")
    aa1_message = ToolMessage(
        id=aa1_hash,
        content=aa1_payload["message"],
        tool_call_id=aa1_hash,
        artifact=aa1_payload,
        additional_kwargs={
            "name": "geoDatasetSummary",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )

    aa2_payload["message"] = (
        f"Here is a summary of AA2 performance across the {len(state.config.clock_list)} clocks and {len(accession_codes)} datasets we benchmarked:"
    )
    aa2_hash = _create_hash(accession_codes, "aa2_performance")
    aa2_message = ToolMessage(
        id=aa2_hash,
        content=aa2_payload["message"],
        tool_call_id=aa2_hash,
        artifact=aa2_payload,
        additional_kwargs={
            "name": "geoDatasetSummary",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )

    progress_update = benchmarking_progress(state)
    logger.info(f"\nFinal progress update message: {json.dumps(progress_update.model_dump(), indent=4)}\n")

    return_dict = {
        "main_messages": [pp_message, aa1_message, aa2_message, progress_update],
        "messages": [pp_message, aa1_message, aa2_message, progress_update],
    }
    return Command(update=return_dict)
