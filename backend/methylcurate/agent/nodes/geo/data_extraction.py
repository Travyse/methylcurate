__all__ = [
    "extract_sample_metadata",
    "generate_metadata_extraction_summary",
    "format_supplementary_data",
    "merge_supplementary_file_data",
    "refine_extracted_columns",
    "summarize_geo_findings",
]
import gc
import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from ....contracts.common import ArtifactRef
from ....tools.geo import (
    _create_subject_id_mapping,
    extract_dataset_metadata,
    format_individual_methylation_data,
    generate_summary_data,
)
from ....utils.helper import (
    check_step_completion,
    compute_sha256,
    consolidate_artifacts,
    get_accession_codes,
    get_correct_methylation_data,
    read_feather,
    set_step_status,
    update_progress_tracker,
    write_feather,
)
from ....utils.memory import shutdown as mem_shutdown
from ....utils.memory import snapshot as mem_snap
from ....utils.memory import tracemalloc_snapshot as mem_trace
from ....utils.memory import trim_heap
from ...state.models import GeoIngestionSubgraphState


def _emit_new_artifact_events(
    pre_paths: set[str],
    post_artifacts: list[dict[str, Any]],
    config: RunnableConfig,
    step: str,
) -> None:
    """Emit ArtifactWritten for artifacts created since pre_paths snapshot."""
    deps_init = config["configurable"].get("deps")
    if deps_init is None or getattr(deps_init, "provenance", None) is None:
        return
    provenance = deps_init.get_provenance(config["configurable"]["thread_id"])
    if provenance is None:
        return
    for a_dict in post_artifacts:
        a_path = a_dict.get("path", "")
        if a_path not in pre_paths:
            provenance.emit_artifact_written(
                artifact_kind=str(a_dict.get("kind", "")),
                artifact_path=a_path,
                artifact_sha256=str(a_dict.get("sha256", "")),
                artifact_bytes=int(a_dict.get("bytes", 0)),
                accession_code=str(a_dict.get("accession_code", "")),
                step=step,
            )


async def extract_sample_metadata(state: GeoIngestionSubgraphState, config: RunnableConfig) -> dict[str, Any]:
    mem_snap("data:extract_sample_metadata:entry")
    accession_codes = get_accession_codes(state)
    if check_step_completion("extract_data", state.datasets, accession_codes):
        return Command(update={"main_messages": [update_progress_tracker(state)], "messages": [update_progress_tracker(state)]})
    running_accession_codes = sorted(
        [accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["extract_data"].status == "running"]
    )
    accession_code = running_accession_codes[0]
    dataset_state = state.datasets[accession_code]
    metadata_artifact = next(
        (artifact for artifact in state.config.artifacts if (artifact.kind == "metadata_cache") and (artifact.accession_code == accession_code)),
        None,
    )
    with open(metadata_artifact.path, encoding="utf-8") as f:  # type: ignore
        metadata_dict = json.load(f)
    return_dict = {"config": state.config.model_dump(), "datasets": {accession_code: dataset_state.model_dump()}}
    pre_paths = {a.get("path", "") for a in return_dict["config"]["artifacts"]}
    return_dict = extract_dataset_metadata(
        accession_code,
        state.config,
        metadata_dict,
        dataset_state.metadata_extraction_result,  # type: ignore
        True,
        gpls=[dataset_state.platform_metadata.platform_id],  # type: ignore
        platform=[dataset_state.platform_metadata.title],  # type: ignore
        return_dict=return_dict,
    )
    _emit_new_artifact_events(pre_paths, return_dict["config"]["artifacts"], config, "extract_data")
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    return Command(update=return_dict)


async def generate_metadata_extraction_summary(state: GeoIngestionSubgraphState, config: RunnableConfig) -> dict[str, Any]:
    mem_snap("data:generate_metadata_extraction_summary:entry")
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted(
        [accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["extract_data"].status == "running"]
    )
    if not running_accession_codes:
        return Command(update={"main_messages": [update_progress_tracker(state)], "messages": [update_progress_tracker(state)]})
    accession_code = running_accession_codes[0]
    dataset_state = state.datasets[accession_code]
    return_dict = {"config": state.config.model_dump(), "datasets": {accession_code: dataset_state.model_dump()}}
    metadata_artifact = next(
        (artifact for artifact in state.config.artifacts if (artifact.kind == "dataset_metadata") and (artifact.accession_code == accession_code)),
        None,
    )
    metadata = pd.read_csv(metadata_artifact.path, index_col=0)  # type: ignore
    return_dict = generate_summary_data(
        metadata,
        accession_code,
        [dataset_state.platform_metadata.platform_id],  # type: ignore
        [dataset_state.platform_metadata.title],  # type: ignore
        dataset_state.refinement_history.example_errors,  # type: ignore
        return_dict,
    )
    return_dict["datasets"][accession_code]["steps"]["extract_data"] = set_step_status(
        status="completed", step=return_dict["datasets"][accession_code]["steps"]["extract_data"]
    )
    return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"] = set_step_status(
        status="running", step=return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"]
    )
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    return Command(update=return_dict)


async def format_supplementary_data(state: GeoIngestionSubgraphState, config: RunnableConfig) -> dict[str, Any]:
    mem_snap("node:format_supplementary_data:entry")
    trim_heap("node:format_supplementary_data:trimmed")
    mem_trace("format_supplementary_data_entry")
    accession_codes = get_accession_codes(state)
    if check_step_completion("supplementary_file_check", state.datasets, accession_codes):
        return Command(update={"main_messages": [update_progress_tracker(state)], "messages": [update_progress_tracker(state)]})
    running_accession_codes = sorted(
        [accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["supplementary_file_check"].status == "running"]
    )
    accession_code = running_accession_codes[0]
    dataset_state = state.datasets[accession_code]
    supplementary_file_artifacts = sorted(
        [
            artifact
            for artifact in state.config.artifacts
            if (artifact.kind == "supplementary_file_methylation_data") and (artifact.accession_code == accession_code)
        ],
        key=lambda artifact: artifact.path,
    )
    formatted_supplementary_file_artifacts = sorted(
        [
            artifact
            for artifact in state.config.artifacts
            if (artifact.kind == "supplementary_file_methylation_data_formatted") and (artifact.accession_code == accession_code)
        ],
        key=lambda artifact: artifact.path,
    )
    running_supplementary_file_artifacts = [
        artifact
        for artifact in supplementary_file_artifacts
        if f"{os.path.splitext(artifact.path)[0]}_proc"
        not in [os.path.splitext(artifact.path)[0] for artifact in formatted_supplementary_file_artifacts]
    ]
    if not running_supplementary_file_artifacts:
        return Command(update={"main_messages": [update_progress_tracker(state)], "messages": [update_progress_tracker(state)]})

    return_dict = {"config": state.config.model_dump(), "datasets": {accession_code: dataset_state.model_dump()}}
    current_artifact = running_supplementary_file_artifacts[0]
    return_dict = await format_individual_methylation_data(accession_code, return_dict, config, current_artifact, state.messages)
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    mem_snap("node:format_supplementary_data:exit")
    return Command(update=return_dict)


async def merge_supplementary_file_data(state: GeoIngestionSubgraphState, config: RunnableConfig) -> dict[str, Any]:
    mem_snap("node:merge_supplementary:entry")
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted(
        [accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["supplementary_file_check"].status == "running"]
    )
    accession_code = running_accession_codes[0]
    formatted_supplementary_file_artifacts = sorted(
        [
            artifact
            for artifact in state.config.artifacts
            if (artifact.kind == "supplementary_file_methylation_data_formatted") and (artifact.accession_code == accession_code)
        ],
        key=lambda artifact: artifact.path,
    )
    dataset_state = state.datasets[accession_code]
    methylation_dataframe_output_path = os.path.join(dataset_state.output_dir, "preqc_methylation_matrix.feather")

    return_dict = {"config": state.config.model_dump(), "datasets": {accession_code: dataset_state.model_dump()}}
    pre_paths = {a.get("path", "") for a in return_dict["config"]["artifacts"]}
    if len(formatted_supplementary_file_artifacts) == 1:
        existing = formatted_supplementary_file_artifacts[0].path
        if existing != methylation_dataframe_output_path:
            try:
                os.link(existing, methylation_dataframe_output_path)
            except OSError:
                import shutil

                shutil.copy2(existing, methylation_dataframe_output_path)
    else:
        formatted_datasets = [read_feather(artifact.path, index_name="subject_id") for artifact in formatted_supplementary_file_artifacts]
        formatted_data = pd.concat(formatted_datasets, axis=0)
        write_feather(formatted_data, methylation_dataframe_output_path, index_name="subject_id")
        del formatted_datasets, formatted_data
        gc.collect()

    methylation_artifact = ArtifactRef.model_validate(
        {
            "accession_code": accession_code,
            "path": methylation_dataframe_output_path,
            "kind": "preqc_methylation_data",
            "sha256": compute_sha256(methylation_dataframe_output_path, is_path=True),
            "bytes": os.path.getsize(methylation_dataframe_output_path),
            "created_at": datetime.now(UTC).isoformat(),
        }
    )

    return_dict["config"]["artifacts"] = consolidate_artifacts([ArtifactRef(**a) for a in return_dict["config"]["artifacts"]], [methylation_artifact])
    _emit_new_artifact_events(pre_paths, [methylation_artifact.model_dump()], config, "merge_supplementary_data")
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    mem_snap("node:merge_supplementary:exit")
    return Command(update=return_dict)


async def refine_extracted_columns(state: GeoIngestionSubgraphState, config: RunnableConfig) -> dict[str, Any]:
    mem_snap("node:refine_extracted_columns:entry")
    gc.collect()
    mem_snap("node:refine_extracted_columns:after_collect")
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted(
        [accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["supplementary_file_check"].status == "running"]
    )
    accession_code = running_accession_codes[0]
    dataset_state = state.datasets[accession_code]
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {accession_code: dataset_state.model_dump()},
        "main_messages": [update_progress_tracker(state)],
        "messages": [update_progress_tracker(state)],
    }
    pre_paths = {a.get("path", "") for a in return_dict["config"]["artifacts"]}

    methylation_artifact = get_correct_methylation_data(state.config.artifacts, accession_code)
    target_values = pd.read_feather(methylation_artifact.path, columns=["subject_id"])["subject_id"].sort_values().tolist()
    if not target_values:
        return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"] = set_step_status(
            status="completed", step=return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"]
        )
        return Command(update=return_dict)

    metadata_artifact = next(
        (artifact for artifact in state.config.artifacts if (artifact.kind == "metadata_cache") and (artifact.accession_code == accession_code)),
        None,
    )
    with open(metadata_artifact.path, encoding="utf-8") as f:  # type: ignore
        metadata_dict = json.load(f)
        sample_subject_mapping = _create_subject_id_mapping(
            accession_code,
            dataset_state.metadata_extraction_input,  # type: ignore
            metadata_dict,
            target_values,
        )
    print("\nSample to subject mapping DataFrame created:\n")
    mapper_artifact_path = os.path.join(
        os.path.dirname(metadata_artifact.path),  # type: ignore
        f"{metadata_artifact.accession_code}_subject_mapping.json",  # ty: ignore
    )
    sample_subject_mapping.to_csv(mapper_artifact_path, index=True)
    mapper_artifact = ArtifactRef.model_validate(
        {
            "accession_code": metadata_artifact.accession_code,  # type: ignore
            "path": mapper_artifact_path,
            "kind": "subject_column_mapping",
            "sha256": compute_sha256(mapper_artifact_path, is_path=True),
            "bytes": os.path.getsize(mapper_artifact_path),
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    return_dict["config"]["artifacts"] = consolidate_artifacts([ArtifactRef(**a) for a in return_dict["config"]["artifacts"]], [mapper_artifact])
    _emit_new_artifact_events(pre_paths, [mapper_artifact.model_dump()], config, "refine_extracted_columns")
    return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"] = set_step_status(
        status="completed", step=return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"]
    )
    trim_heap("node:refine_extracted_columns:trimmed")
    return Command(update=return_dict)


def summarize_geo_findings(state: GeoIngestionSubgraphState, config: RunnableConfig) -> GeoIngestionSubgraphState:
    accession_codes = set()
    payload = {
        "id": "geo-dataset-summary",
        "rowIdKey": "accession_code",
        "columns": [
            {
                "key": "accession_code",
                "label": "Accession Code",
                "priority": "primary",
                "align": "left",
            },
            {
                "key": "sample_count",
                "label": "Sample Count",
                "priority": "primary",
                "align": "right",
                "format": {
                    "kind": "number",
                },
            },
            {
                "key": "age_range",
                "label": "Age Range",
                "priority": "primary",
                "align": "right",
            },
            {
                "key": "conditions",
                "label": "Conditions",
                "priority": "secondary",
                "align": "left",
            },
            {
                "key": "sex",
                "label": "Sex",
                "priority": "secondary",
                "align": "left",
            },
            {
                "key": "platform",
                "label": "Platform",
                "priority": "secondary",
                "align": "left",
            },
        ],
        "rows": [],
    }

    for artifact in state.config.artifacts:
        if artifact.kind == "dataset_metadata":
            metadata = pd.read_csv(artifact.path, index_col=0)
            num_samples = len(metadata)
            sexes = metadata["Sex"].value_counts(normalize=True).to_dict()
            if not sexes:
                sexes = "N/A"
            else:
                female_key = next((str(k) for k in sexes.keys() if "f" in str(k).lower()), None)
                male_key = next((str(k) for k in sexes.keys() if str(k) != female_key), None)
                if any(k is None for k in [male_key, female_key]):
                    sexes = "N/A"
                else:
                    sexes = f"M ({sexes[male_key]:.2%}), F ({sexes[female_key]:.2%})"
            conditions = metadata["Disease_Status"].value_counts(normalize=True).to_dict()
            conditions = ", ".join([str(k) for k in conditions.keys()])
            min_age = metadata["age"].min()
            max_age = metadata["age"].max()
            min_age = None if pd.isna(min_age) else float(min_age)
            max_age = None if pd.isna(max_age) else float(max_age)
            age_range = "N/A" if min_age is None or max_age is None else f"{min_age:.2f}-{max_age:.2f}"
            platform_val = metadata["Platform"].iloc[0]
            platform = None if pd.isna(platform_val) else str(platform_val)
            if not any(r["accession_code"] == artifact.accession_code for r in payload["rows"]):
                payload["rows"].append(
                    {
                        "accession_code": artifact.accession_code,
                        "sample_count": num_samples,
                        "age_range": age_range,
                        "conditions": conditions,
                        "platform": platform,
                        "sex": sexes,
                    }
                )
            accession_codes.add(artifact.accession_code)
    accession_codes = sorted(list(accession_codes))
    m = hashlib.sha256()
    for accession_code in sorted(accession_codes):
        m.update(accession_code.encode("utf-8"))
    hash = m.hexdigest()

    payload["message"] = f"We have downloaded and formatted {len(accession_codes)} datasets from GEO. Here is a summary of the datasets:"
    message = ToolMessage(
        id=hash,
        content=payload["message"],
        tool_call_id=hash,
        artifact=payload,
        additional_kwargs={
            "name": "geoDatasetSummary",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )

    return_dict = {"main_messages": [message], "messages": [message]}
    mem_snap("subgraph:summarize_geo_findings:exit")
    mem_shutdown()
    return Command(update=return_dict)
