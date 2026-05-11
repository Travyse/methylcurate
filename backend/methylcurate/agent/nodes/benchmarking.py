__all__ = ["clock_retrieval_node", "benchmarking_node", "summarize_benchmarking_results"]

import hashlib
import json
import os
from datetime import UTC, datetime

import pandas as pd
import pyaging as pya
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from scipy import stats

from ...contracts.common import ArtifactRef
from ...tools.clocks.inference import (
    compute_age_acceleration,
    compute_mae,
    compute_medae,
    compute_pearson_r,
    get_dataset_predictions,
    get_extraction_protocol,
    get_metadata_dataframe,
    make_internal_clock_predictions,
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
    metadata_df["Disease_Status"] = metadata_df["Disease_Status"].map(individual_mapper).fillna(metadata_df["Disease_Status"])
    metadata_df["Tissue_Group"] = metadata_df["Tissue"].map(group_mapper).fillna(metadata_df["Tissue"])
    metadata_df["Tissue"] = metadata_df["Tissue"].map(individual_mapper).fillna(metadata_df["Tissue"])
    metadata_df["Cell_Type"] = metadata_df["Cell_Type"].map(individual_mapper).fillna(metadata_df["Cell_Type"])
    metadata_df["Sex"] = metadata_df["Sex"].map(individual_mapper).fillna(metadata_df["Sex"])
    metadata_df["female"] = (metadata_df["Sex"] == "Female").astype(int)
    metadata_df.drop(columns=["Subject"], inplace=True, errors="ignore")
    metadata_columns = metadata_df.columns.tolist()
    return metadata_df.merge(methylation_df, left_index=True, right_index=True), metadata_columns


def clock_retrieval_node(state: BenchmarkingSubgraphState, config: RunnableConfig) -> Command:
    return_dict = {"config": state.config.model_dump()}
    artifacts = []
    clock_dir = os.path.join(state.config.output_root, "clocks")
    pyalogger = pya.logger.LoggerManager.gen_logger("predict_age")
    logger = setup_logger(os.path.join(os.path.dirname(state.config.output_root), "logs"), "clock_retrieval_node", "clock_retrieval.log")
    logger.info("Starting clock retrieval node...")
    logger.info(f"Current clock_dir: {clock_dir}")
    clocks = sorted(state.config.clock_list)
    clocks_to_retrieve = [
        clock
        for clock in clocks
        if not any(
            a
            for a in state.config.artifacts
            if a.kind == "clock" and (a.path.endswith(f"{clock.lower()}.pt") or a.path.endswith(f"{clock.lower()}_model.pkl"))
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
        deps_init = config["configurable"].get("deps")
        if deps_init is not None and getattr(deps_init, "provenance", None) is not None:
            provenance = deps_init.get_provenance(config["configurable"]["thread_id"])
            if provenance is not None:
                for art in artifacts:
                    provenance.emit_artifact_written(
                        artifact_kind=art.kind,
                        artifact_path=art.path,
                        artifact_sha256=art.sha256,
                        artifact_bytes=art.bytes,
                        accession_code=art.accession_code,
                        step="clock_retrieval",
                    )
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
        deps_init = config["configurable"].get("deps")
        if deps_init is not None and getattr(deps_init, "provenance", None) is not None:
            provenance = deps_init.get_provenance(config["configurable"]["thread_id"])
            if provenance is not None:
                for art in artifacts:
                    provenance.emit_artifact_written(
                        artifact_kind=art.kind,
                        artifact_path=art.path,
                        artifact_sha256=art.sha256,
                        artifact_bytes=art.bytes,
                        accession_code=art.accession_code,
                        step="clock_retrieval",
                    )
    return_dict["messages"] = [benchmarking_progress(state)]
    return_dict["main_messages"] = [benchmarking_progress(state)]
    return Command(update=return_dict)


def benchmarking_node(state: BenchmarkingSubgraphState, config: RunnableConfig) -> Command:
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
    logger = setup_logger(
        os.path.join(os.path.dirname(state.config.output_root), "logs"),
        f"{accession_code}_benchmark",
        f"{accession_code}_benchmark.log",
    )
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
        logger.info(f"\nMethylation data:\n {methylation_df[methylation_df.columns.tolist()[:20]].head().to_markdown(index=True)}")
    except Exception as e:
        logger.error(f"Error loading data for accession code {accession_code}: {e}")
        return_dict["datasets"][accession_code]["steps"]["make_predictions"]["status"] = "failed"
        return Command(update=return_dict)
    if methylation_df.empty:
        logger.error(f"No methylation data found for accession code {accession_code}. Skipping benchmarking for this dataset.")
        return_dict["datasets"][accession_code]["steps"]["make_predictions"]["status"] = "failed"
        return Command(update=return_dict)
    cpg_columns = [col for col in methylation_df.columns if col.startswith("cg")]
    if not cpg_columns:
        logger.error(f"No CpG columns found in methylation data for accession code {accession_code}. Skipping benchmarking for this dataset.")
        return_dict["datasets"][accession_code]["steps"]["make_predictions"]["status"] = "failed"
        return Command(update=return_dict)
    if "Subject" not in methylation_df.columns.tolist():
        methylation_metadata_col = []
    full_data, metadata_columns = _get_harmonized_full_data_if_available(accession_code, metadata_df, methylation_df, state.config.artifacts)

    if full_data["Sex"].isnull().any():
        if set([x.lower() for x in clock_list]).intersection({"grimage", "grimage2", "pcgrimage"}):
            clock_list = [x for x in clock_list if x.lower() not in {"grimage", "grimage2", "pcgrimage"}]
        full_data.drop(columns=["Sex", "female"], inplace=True, errors="ignore")

    logger.info(f"\nDataset after merging metadata and methylation data: {full_data[full_data.columns.tolist()[:30]].head().to_markdown(index=True)}")
    adata = pya.pp.df_to_adata(full_data, metadata_cols=metadata_columns + methylation_metadata_col, imputer_strategy="knn", verbose=False)
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
    deps_init = config["configurable"].get("deps")
    if deps_init is not None and getattr(deps_init, "provenance", None) is not None:
        provenance = deps_init.get_provenance(config["configurable"]["thread_id"])
        if provenance is not None:
            provenance.emit_artifact_written(
                artifact_kind=artifact.kind,
                artifact_path=artifact.path,
                artifact_sha256=artifact.sha256,
                artifact_bytes=artifact.bytes,
                accession_code=artifact.accession_code,
                step="make_predictions",
            )
    return_dict["datasets"][accession_code]["steps"]["make_predictions"]["status"] = "completed"
    return_dict["messages"] = [benchmarking_progress(state)]
    return_dict["main_messages"] = [benchmarking_progress(state)]
    return Command(update=return_dict)


def summarize_benchmarking_results(state: BenchmarkingSubgraphState, config: RunnableConfig) -> Command:
    logger = setup_logger(os.path.join(os.path.dirname(state.config.output_root), "logs"), "overall_benchmark_summary", "benchmark_summary.log")
    accession_codes = sorted([x for x in state.datasets if state.datasets[x].steps["make_predictions"].status == "completed"])

    payload = {
        "id": "benchmark-summary",
        "rowIdKey": "accession_code",
        "columns": [
            {
                "key": "accession_code",
                "label": "Accession Code",
                "priority": "primary",
                "align": "left",
            },
            {
                "key": "clock",
                "label": "Clock",
                "priority": "primary",
                "align": "left",
            },
            {
                "key": "medae",
                "label": "Median Absolute Error",
                "priority": "primary",
                "align": "right",
                "format": {"kind": "number", "decimals": 2},
            },
            {
                "key": "mae",
                "label": "Mean Absolute Error",
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

    all_healthy_predictions = []

    for accession_code in accession_codes:
        try:
            prediction_df = get_dataset_predictions(accession_code, state.config.artifacts)
            extraction_protocol = get_extraction_protocol(accession_code, state.config.artifacts)
        except Exception as e:
            logger.warning(f"Skipping dataset {accession_code} in summary: {e}")
            continue

        control_label = extraction_protocol["disease_status"]["extraction"]["control_value"]
        healthy_subset = prediction_df[prediction_df["Disease_Status"] == control_label].copy()
        if not healthy_subset.empty:
            all_healthy_predictions.append(healthy_subset)

        mae_df = compute_mae(prediction_df, extraction_protocol, clocks=state.config.clock_list)
        medae_df = compute_medae(prediction_df, extraction_protocol, clocks=state.config.clock_list)
        pearson_r_df = compute_pearson_r(prediction_df, extraction_protocol, clocks=state.config.clock_list)

        merged = mae_df.merge(medae_df, on=["Accession_Code", "Clock"]).merge(pearson_r_df, on=["Accession_Code", "Clock"])
        for _, row in merged.iterrows():
            payload["rows"].append(
                {
                    "accession_code": accession_code,
                    "clock": row["Clock"],
                    "medae": row["MedAE_score"],
                    "mae": row["MAE_score"],
                    "pearson_r": row["Pearson_R_score"],
                }
            )

    if all_healthy_predictions:
        pooled = pd.concat(all_healthy_predictions, ignore_index=True)
        overall_rows = []
        for clock in state.config.clock_list:
            accel_col = f"{clock.lower()}_accel"
            if accel_col not in pooled.columns:
                continue
            subset = pooled.dropna(subset=[accel_col, "age"])
            if len(subset) < 2:
                continue
            errors = abs(subset[accel_col] - subset["age"])
            overall_rows.append(
                {
                    "accession_code": "Overall",
                    "clock": clock,
                    "medae": errors.median(),
                    "mae": errors.mean(),
                    "pearson_r": stats.pearsonr(subset[accel_col], subset["age"])[0],
                }
            )
        payload["rows"] = overall_rows + payload["rows"]

    def _create_hash(accession_codes: list[str], identifier: str):
        m = hashlib.sha256()
        for accession_code in sorted(accession_codes):
            m.update(accession_code.encode("utf-8"))
        m.update(b"Benchmarking")
        m.update(identifier.encode("utf-8"))
        return m.hexdigest()

    message_text = (
        f"We benchmarked {len(state.config.clock_list)} clocks on {len(accession_codes)} datasets from GEO. Here is a summary of their performance:"
    )
    payload["message"] = message_text
    payload_hash = _create_hash(accession_codes, "benchmark_summary")
    message = ToolMessage(
        id=payload_hash,
        content=message_text,
        tool_call_id=payload_hash,
        artifact=payload,
        additional_kwargs={
            "name": "geoDatasetSummary",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )

    progress_update = benchmarking_progress(state)
    logger.info(f"\nFinal progress update message: {json.dumps(progress_update.model_dump(), indent=4)}\n")

    return_dict = {
        "main_messages": [message, progress_update],
        "messages": [message, progress_update],
    }
    return Command(update=return_dict)
