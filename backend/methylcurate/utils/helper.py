__all__ = [
    "sample_values",
    "json_extraction",
    "compute_sha256",
    "NonEmptyStr",
    "set_step_status",
    "_get_status_id",
    "retrieve_status_counts",
    "populate_todos",
    "_get_supplementary_file_id",
    "check_step_completion",
    "write_feather",
    "read_feather",
    "update_small_progress_tracker",
    "load_metadata_aligned_methylation_data",
    "benchmarking_progress",
    "qc_progress",
    "PROJECT_ROOT",
    "update_harmonization_progress_tracker",
    "get_correct_methylation_data",
]
import hashlib
import json
import random
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import pyarrow.feather as feather
from langchain_core.messages import ToolMessage
from pydantic import Field

# Seed everything
COMPLETION_LIST = {"completed", "failed", "canceled"}


def check_step_completion(step: str, datasets: dict[str, Any], accession_codes: list[str]):
    """
    Checks if a specific step is completed for all given accession codes.

    Args:
        step (str): The step to check.
        datasets (Dict[str, Any]): The datasets containing the steps.
        accession_codes (List[str]): The accession codes to check.

    Returns:
        bool: True if the step is completed for all accession codes, False otherwise.
    """
    statuses = [datasets[x].steps[step].status for x in accession_codes]
    return all(status in COMPLETION_LIST for status in statuses)


def sample_values(values: list[str], max_examples: int = 12) -> list[str]:
    """
    Samples a subset of values from a list, up to a maximum number of examples.

    Args:
        values (List[str]): The list of values to sample from.
        max_examples (int): The maximum number of examples to return.

    Returns:
        List[str]: A list of sampled values.
    """
    n = len(values)
    if n <= max_examples:
        return [str(v) for v in values]
    values_subset = random.sample(values, k=max_examples)
    return [str(v) for v in values_subset]


def json_extraction(raw: str, default_response: dict) -> dict:
    """
    Extracts a JSON object from a raw string. If extraction fails, returns a default response.

    Args:
        raw (str): The raw string containing JSON.
        default_response (Dict): The default response to return if extraction fails.

    Returns:
        dict: The extracted JSON object or the default response.
    """
    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return default_response
        parsed = json.loads(m.group(0))
    return parsed


def _compute_sha256_from_path(path: str) -> str:
    """
    Computes the SHA-256 hash of a file at the given path.

    Args:
        path (str): The path to the file.

    Returns:
        str: The SHA-256 hash of the file.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _compute_sha256_from_content(content: str) -> str:
    """
    Computes the SHA-256 hash of a string content.

    Args:
        content (str): The string content to hash.

    Returns:
        str: The SHA-256 hash of the content.
    """
    h = hashlib.sha256()
    h.update(content.encode("utf-8"))
    return h.hexdigest()


def compute_sha256(content: str, is_path: bool = False) -> str:
    """
    Computes the SHA-256 hash of a file or string content.

    Args:
        content (str): The file path or string content to hash.
        is_path (bool): If True, treat content as a file path. Otherwise, treat content as a string.

    Returns:
        str: The SHA-256 hash of the file or string content.
    """
    return _compute_sha256_from_path(content) if is_path else _compute_sha256_from_content(content)


def get_correct_methylation_data(
    artifacts: list[Any], accession_code: str, artifact_kind: str = "preqc_methylation_data"
) -> Any:
    """
    Retrieves the correct methylation data artifact for a given accession code.

    Args:
        artifacts (List[Any]): A list of artifacts to search through.
        accession_code (str): The accession code to match.
        artifact_kind (str): The kind of artifact to look for. Defaults to "preqc_methylation_data".

    Returns:
        Any: The matching artifact, or None if not found.
    """
    target_artifacts = [
        artifact
        for artifact in artifacts
        if (artifact.kind == artifact_kind) and (artifact.accession_code == accession_code)
    ]
    if len(target_artifacts) == 0:
        return None
    elif len(target_artifacts) == 1:
        return target_artifacts[0]
    else:
        return next(
            (
                artifact
                for artifact in target_artifacts
                if f"{accession_code}_preqc_methylation_matrix" not in artifact.path
            ),
            None,
        )


def get_accession_codes(state: Any) -> list[str]:
    """
    Retrieves the accession codes from the state.

    Args:
        state (Any): The state object containing configuration and dataset information.

    Returns:
        List[str]: A sorted list of accession codes.
    """
    return sorted(
        [x.upper() for x in state.config.accessions if state.datasets[x.upper()].status not in ("failed", "resolved")]
    )


def set_step_status(status="running", step=None, error=None, warnings=None) -> dict[str, Any]:
    """
    Sets the status of a step and updates its timestamps.

    Args:
        status (str): The status to set. Defaults to "running".
        step (Dict[str, Any], optional): The step dictionary to update. If None, a new step dictionary is created.
        error (str, optional): An error message, if any.
        warnings (List[str], optional): A list of warning messages, if any.

    Returns:
        Dict[str, Any]: The updated step dictionary.
    """
    if step is not None:
        step["status"] = status
        step["finished_at"] = datetime.now(UTC).isoformat() if status in ("completed", "failed", "canceled") else None
        return step
    return {
        "status": status,
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": datetime.now(UTC).isoformat() if status in ("completed", "failed", "canceled") else None,
        "error": error,
        "warnings": warnings or [],
    }


def retrieve_status_counts(accession_codes: list[str], datasets: dict[str, Any], step: str) -> dict[str, int]:
    """
    Retrieves the count of datasets in each status category for a specific step.

    Args:
        accession_codes (List[str]): A list of accession codes to check.
        datasets (Dict[str, Any]): The datasets containing the steps and their statuses.
        step (str): The step for which to retrieve status counts.

    Returns:
        Dict[str, int]: A dictionary containing the count of datasets in each status category.
    """
    return {
        "completed": len(
            [
                accession_code
                for accession_code in accession_codes
                if datasets[accession_code].steps[step].status == "completed"
            ]
        ),
        "not_started": len(
            [
                accession_code
                for accession_code in accession_codes
                if datasets[accession_code].steps[step].status == "not_started"
            ]
        ),
        "in_progress": len(
            [
                accession_code
                for accession_code in accession_codes
                if datasets[accession_code].steps[step].status in ["not_started", "running"]
            ]
        ),
        "failed": len(
            [
                accession_code
                for accession_code in accession_codes
                if datasets[accession_code].steps[step].status == "failed"
            ]
        ),
        "canceled": len(
            [
                accession_code
                for accession_code in accession_codes
                if datasets[accession_code].steps[step].status == "canceled"
            ]
        ),
    }


def _generate_todo_description(
    status: str, status_counts: dict[str, int], description_default: str | None = None
) -> str | None:
    """
    Generates a description for a todo item based on the status and status counts.

    Args:
        status (str): The status of the todo item.
        status_counts (Dict[str, int]): A dictionary containing the count of datasets in each status category.
        description_default (str, optional): The default description to use if the status is not "completed".

    Returns:
        str: The generated description.
    """
    if status == "completed":
        return f"{status_counts['completed']} succeeded, {status_counts['failed']} failed, {status_counts['canceled']} canceled."
    return description_default


def _generate_todo_id(step: str):
    """
    Generates a unique ID for a todo item based on the step.

    Args:
        step (str): The step for which to generate the todo ID.

    Returns:
        str: The generated todo ID.
    """
    if step in ["download_soft", "check_valid_dataset"]:
        return "1"
    elif step == "extract_metadata_schema":
        return "2"
    elif step == "extract_data":
        return "3"
    elif step == "refine_metadata_schema":
        return "4"
    elif step == "supplementary_file_check":
        return "5"


def _generate_step_actions(step: str):
    """
    Generates the actions for a todo item based on the step.

    Args:
        step (str): The step for which to generate the todo actions.

    Returns:
        List[str]: A list containing the actions for the todo item.
    """
    if step == "download_soft":
        return ["Downloading", "Downloaded", "Download GEO datasets..."]
    elif step == "check_valid_dataset":
        return ["Downloading", "Downloaded", "Download GEO datasets..."]
    elif step == "extract_metadata_schema":
        return ["Learning metadata schema", "Learned metadata schema", "Learning Metadata Schema Per Dataset"]
    elif step == "refine_metadata_schema":
        return [
            "Refining Learned Metadata Schema",
            "Refined Learned Metadata Schema",
            "Checking and Refining Learned Metadata Schema",
        ]
    elif step == "extract_data":
        return [
            "Formatting GEO Datasets",
            "Formatted GEO Datasets",
            "Formatting GEO Datasets",
        ]
    elif step == "supplementary_file_check":
        return [
            "Formatting GEO Supplementary File Data",
            "Formatted GEO Supplementary File Data",
            "Formatting GEO Supplementary File Data",
        ]


def _generate_description_default(step: str, status: str, counts: dict[str, int]) -> str:
    """
    Generates the default description for a todo item based on the step and status.

    Args:
        step (str): The step for which to generate the description.
        status (str): The status of the todo item.
        counts (Dict[str, int]): A dictionary containing the count of datasets in each status category.

    Returns:
        str: The generated default description.
    """
    if status != "Completed":
        if step == "download_soft":
            return "Download the user-specified GEO datasets..."
        elif step == "check_valid_dataset":
            return "Check the validity of the downloaded GEO datasets..."
        elif step == "extract_metadata_schema":
            return "Learn how to extract important metadata from each GEO dataset..."
        elif step == "refine_metadata_schema":
            return "Double check the learned metadata extraction approach for each dataset and refine as needed..."
        elif step == "extract_data":
            return "Use the learned metadata schema to extract and format the GEO datasets for downstream use..."
        elif step == "supplementary_file_check":
            return "If there are supplementary files, use the learned metadata schema to extract and format the data for downstream use..."
    else:
        return f"{counts['completed']} succeeded, {counts['failed']} failed, {counts['canceled']} canceled."
    return f"Working on {step}..."


def _generate_todo_label(
    status: str,
    past_tense: str,
    present_tense: str,
    num_datasets: int,
    counts: dict[str, int],
    default_message: str | None = None,
) -> str:
    """
    Generates the label for a todo item based on the status and counts.

    Args:
        status (str): The status of the todo item.
        past_tense (str): The past tense description of the action.
        present_tense (str): The present tense description of the action.
        num_datasets (int): The total number of datasets.
        counts (Dict[str, int]): A dictionary containing the count of datasets in each status category.
        default_message (str, optional): The default message to use if the status is not recognized.

    Returns:
        str: The generated todo label.
    """
    finished_count = counts["completed"] + counts["failed"] + counts["canceled"]
    if status == "completed":
        return f"{past_tense} {num_datasets} datasets"
    elif status == "in_progress":
        return f"{present_tense} ({min(finished_count + 1, num_datasets)} of {num_datasets} datasets)"
    else:
        return f"{default_message}"


def _generate_todo_status(counts: dict[str, int], num_datasets: int) -> str:
    """
    Generates the status for a todo item based on the counts and total number of datasets.

    Args:
        counts (Dict[str, int]): A dictionary containing the count of datasets in each status category.
        num_datasets (int): The total number of datasets.

    Returns:
        str: The generated todo status.
    """
    if counts["completed"] + counts["failed"] + counts["canceled"] == num_datasets:
        return "completed"
    elif counts["not_started"] == num_datasets:
        return "pending"
    else:
        return "in_progress"


def generate_todo_entry(
    step: str, counts: dict[str, int], num_datasets: int, other_counts: dict[str, dict[str, int]]
) -> dict[str, Any]:
    """
    Generates a todo entry for a specific step based on the counts and total number of datasets.

    Args:
        step (str): The step for which to generate the todo entry.
        counts (Dict[str, int]): A dictionary containing the count of datasets in each status category for the current step.
        num_datasets (int): The total number of datasets.
        other_counts (Dict[str, Dict[str, int]]): A dictionary containing the count of datasets in each status category for all steps.

    Returns:
        Dict[str, Any]: A dictionary representing the todo entry, including its ID, label, status, and description.
    """
    status = _generate_todo_status(counts, num_datasets)
    id_mapper = {
        "download_soft": "1",
        "check_valid_dataset": "1",
        "extract_metadata_schema": "2",
        "extract_data": "3",
        "refine_metadata_schema": "4",
        "supplementary_file_check": "5",
    }
    past_tense, present_tense, default_message = _generate_step_actions(step)
    description_default = _generate_description_default(step, status, counts)
    id = _generate_todo_id(step)
    previous_ids = range(1, int(id))
    previous_steps = [k for k, v in id_mapper.items() if int(v) in previous_ids]
    if any(
        _generate_todo_status(other_counts[previous_step], num_datasets) == "in_progress"
        for previous_step in previous_steps
    ):
        status = "pending"
    elif any(
        _generate_todo_status(other_counts[previous_step], num_datasets) == "failed" for previous_step in previous_steps
    ):
        status = "canceled"
    return {
        "id": id,
        "label": _generate_todo_label(
            status, past_tense, present_tense, num_datasets, counts, default_message=default_message
        ),
        "status": status,
        "description": _generate_todo_description(status, counts, description_default=description_default),
    }


def populate_todos(state: Any):
    """
    Populates the todo entries for the given state.

    Args:
        state (Any): The current state containing the datasets and their statuses.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the todo entries.
    """
    # Initializate variables
    accession_codes = get_accession_codes(state)
    download_soft_counts = retrieve_status_counts(get_accession_codes(state), state.datasets, "download_soft")
    check_valid_dataset_counts = retrieve_status_counts(
        get_accession_codes(state), state.datasets, "check_valid_dataset"
    )
    extract_metadata_schema_counts = retrieve_status_counts(
        get_accession_codes(state), state.datasets, "extract_metadata_schema"
    )
    refine_metadata_schema_counts = retrieve_status_counts(
        get_accession_codes(state), state.datasets, "refine_metadata_schema"
    )
    extract_data_counts = retrieve_status_counts(get_accession_codes(state), state.datasets, "extract_data")
    supplementary_file_check_counts = retrieve_status_counts(
        get_accession_codes(state), state.datasets, "supplementary_file_check"
    )
    all_count_dicts = {
        "download_soft": download_soft_counts,
        "check_valid_dataset": check_valid_dataset_counts,
        "extract_metadata_schema": extract_metadata_schema_counts,
        "refine_metadata_schema": refine_metadata_schema_counts,
        "extract_data": extract_data_counts,
        "supplementary_file_check": supplementary_file_check_counts,
    }
    first_todo = None
    if download_soft_counts["not_started"] == len(accession_codes):
        first_todo = generate_todo_entry("download_soft", download_soft_counts, len(accession_codes), all_count_dicts)
    elif check_valid_dataset_counts["not_started"] == len(accession_codes) and (
        download_soft_counts["completed"] + download_soft_counts["failed"] + download_soft_counts["canceled"]
        == len(accession_codes)
    ):
        first_todo = generate_todo_entry(
            "check_valid_dataset", check_valid_dataset_counts, len(accession_codes), all_count_dicts
        )
    else:
        first_todo = generate_todo_entry(
            "check_valid_dataset", check_valid_dataset_counts, len(accession_codes), all_count_dicts
        )
    num_datasets = len(accession_codes)
    todos = [
        first_todo,
        generate_todo_entry("extract_metadata_schema", extract_metadata_schema_counts, num_datasets, all_count_dicts),
        generate_todo_entry("extract_data", extract_data_counts, num_datasets, all_count_dicts),
        generate_todo_entry("refine_metadata_schema", refine_metadata_schema_counts, num_datasets, all_count_dicts),
        generate_todo_entry("supplementary_file_check", supplementary_file_check_counts, num_datasets, all_count_dicts),
    ]

    return todos


def _get_status_id(messages: list[Any], retrieval: str) -> str:
    """
    Generates a unique status ID based on the user query and retrieval type.
    Args:
        messages (List[Any]): The list of messages in the workflow.
        retrieval (str): The type of retrieval (e.g., "geo_retrieval", "quality_control", "benchmarking").

    Returns:
        str: A unique status ID based on the user query and retrieval type.
    """
    # Get the user query that triggered the workflow to run
    user_query = [x for x in messages if x.type == "human"][-1]
    # print(f"\n\nUser query content: {user_query}")

    # Generate deterministic hash
    m = hashlib.sha256()
    m.update(user_query.id.encode("utf-8"))
    m.update(user_query.content.encode("utf-8"))
    m.update(str(user_query.additional_kwargs["created_at"]).encode("utf-8"))
    m.update(retrieval.encode("utf-8"))
    hash = m.hexdigest()

    return hash


def update_progress_tracker(state: Any) -> ToolMessage:
    """
    Updates the progress tracker for GEO dataset retrieval based on the current state of the workflow.

    Args:
        state (Any): The current state containing the datasets and their statuses.

    Returns:
        ToolMessage: A message object representing the updated progress tracker.
    """

    hash = _get_status_id(state.messages, "geo_retrieval")

    # Hash from accession codes + user message + user message time
    message = ToolMessage(
        id=hash,
        content="",
        tool_call_id=hash,
        artifact={
            "id": hash,
            "title": "GEO Dataset Retrieval",
            "description": "Tracking the progress of retrieving and formatting GEO datasets",
            "todos": populate_todos(state),
        },
        additional_kwargs={
            "name": "geoRetrievalProgress",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return message


def update_small_progress_tracker(state: Any, retrieval: str) -> ToolMessage:
    """
    Updates the progress tracker for a specific retrieval type based on the current state of the workflow.

    Args:
        state (Any): The current state containing the datasets and their statuses.
        retrieval (str): The type of retrieval (e.g., "geo_retrieval", "quality_control", "benchmarking").

    Returns:
        ToolMessage: A message object representing the updated progress tracker.
    """
    accession_codes = sorted(state.datasets.keys())
    hash = _get_status_id(state.messages, retrieval)

    all_completed = all(
        state.datasets[x].steps["quality_control"].status in {"completed", "failed", "skipped"} for x in accession_codes
    )

    # Hash from accession codes + user message + user message time
    message = ToolMessage(
        id=hash,
        content="",
        tool_call_id=hash,
        artifact={
            "id": hash,
            "steps": [
                {
                    "id": "quality_control_steps",
                    "label": "Running quality control steps",
                    "description": "Completed quality control for {num_completed} datasets",
                    "status": "completed" if all_completed else "pending",
                }
            ],
        },
        additional_kwargs={
            "name": "qcProgress",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return message


def benchmarking_progress(state: Any) -> ToolMessage:
    """
    Updates the progress tracker for benchmarking based on the current state of the workflow.

    Args:
        state (Any): The current state containing the datasets and their statuses.

    Returns:
        ToolMessage: A message object representing the updated progress tracker.
    """
    accession_codes = sorted(state.datasets.keys())
    hash = _get_status_id(state.messages, "benchmarking")
    current_time = datetime.now(UTC).isoformat()
    previous_message = next((m for m in state.main_messages if m.tool_call_id == hash), None)
    previous_time = previous_message.additional_kwargs["created_at"] if previous_message else current_time
    num_clocks_retrieved = sum(
        1
        for clock in state.config.clock_list
        if any(
            a
            for a in state.config.artifacts
            if a.kind == "clock"
            and (a.path.endswith(f"{clock.lower()}.pt") or a.path.endswith(f"{clock.lower()}_model.pkl"))
        )
    )
    all_clocks_retrieved = num_clocks_retrieved == len(state.config.clock_list)
    num_predictions_completed = sum(
        1 for x in accession_codes if state.datasets[x].steps["make_predictions"].status == "completed"
    )
    all_predictions_completed = all(
        state.datasets[x].steps["make_predictions"].status in {"completed", "failed", "skipped"}
        for x in accession_codes
    )
    num_computations_completed = sum(
        1 for x in accession_codes if state.datasets[x].steps["make_computations"].status == "completed"
    )
    all_computations_completed = all(
        state.datasets[x].steps["make_computations"].status in {"completed", "failed", "skipped"}
        for x in accession_codes
    )

    # Hash from accession codes + user message + user message time
    message = ToolMessage(
        id=hash,
        content="",
        tool_call_id=hash,
        artifact={
            "id": hash,
            "steps": [
                {
                    "id": "retrieve_models",
                    "label": "Retrieving Aging Clock Models",
                    "description": f"Retrieved {num_clocks_retrieved} clocks",
                    "status": "completed" if all_clocks_retrieved else "in-progress",
                },
                {
                    "id": "benchmark_models",
                    "label": "Running Benchmark Models",
                    "description": f"Completed benchmarking for {num_predictions_completed} datasets",
                    "status": "completed"
                    if all_predictions_completed
                    else "in-progress"
                    if num_predictions_completed > 0
                    else "pending",
                },
                {
                    "id": "make_computations",
                    "label": "Computing Benchmark Tasks",
                    "description": f"Completed benchmarking for {num_computations_completed} datasets",
                    "status": "completed"
                    if all_computations_completed
                    else "in-progress"
                    if num_computations_completed > 0
                    else "pending",
                },
            ],
            "elapsedTime": round(
                (datetime.fromisoformat(current_time) - datetime.fromisoformat(previous_time)).total_seconds()
            ),
        },
        additional_kwargs={
            "name": "benchmarkProgress",
            "created_at": previous_time,
        },
    )
    return message


def qc_progress(state: Any) -> ToolMessage:
    """
    Updates the progress tracker for quality control based on the current state of the workflow.

    Args:
        state (Any): The current state containing the datasets and their statuses.

    Returns:
        ToolMessage: A message object representing the updated progress tracker.
    """
    hash = _get_status_id(state.messages, "quality_control")
    current_time = datetime.now(UTC).isoformat()
    previous_message = next((m for m in state.main_messages if m.tool_call_id == hash), None)
    previous_time = previous_message.additional_kwargs["created_at"] if previous_message else current_time
    num_datasets_qcd = sum(
        1
        for x in state.datasets.keys()
        if state.datasets[x].steps["quality_control"].status in {"completed", "failed", "canceled"}
    )
    all_datasets_qcd = num_datasets_qcd == len(state.datasets.keys())

    # print(f"Dataset states: {state.datasets}")
    # Hash from accession codes + user message + user message time
    message = ToolMessage(
        id=hash,
        content="",
        tool_call_id=hash,
        artifact={
            "id": hash,
            "steps": [
                {
                    "id": "quality_control",
                    "label": "Performing Quality Control on Datasets",
                    "description": f"QC'd {num_datasets_qcd} datasets",
                    "status": "completed" if all_datasets_qcd else "in-progress",
                },
            ],
            "elapsedTime": round(
                (datetime.fromisoformat(current_time) - datetime.fromisoformat(previous_time)).total_seconds()
            ),
        },
        additional_kwargs={
            "name": "benchmarkProgress",
            "created_at": previous_time,
        },
    )
    return message


def _get_supplementary_file_id(supplementary_files: list[str], accession_code: Any) -> str:
    """
    Generate a unique ID for a supplementary file based on its content and the accession code.

    Args:
        supplementary_files (List[str]): A list of supplementary file paths.
        accession_code (Any): The accession code associated with the files.

    Returns:
        str: A unique hash representing the supplementary files and accession code.
    """
    m = hashlib.sha256()
    for file in sorted(supplementary_files):
        m.update(file.encode("utf-8"))
    m.update(str(accession_code).encode("utf-8"))
    hash = m.hexdigest()
    return hash


def make_review_id(
    *,
    run_id: str,
    subgraph: str,
    entity_type: str,
    entity_id: str,
    step: str,
) -> str:
    """
    Create a unique, traceable ID for a human-in-the-loop review.

    Example:
      review:run-001:geo_ingestion:GSE40279:resolve_metadata:3f2a9c
    """
    short = uuid.uuid4().hex[:6]
    return f"review:{run_id}:{subgraph}:{entity_type}:{entity_id}:{step}:{short}"


NonEmptyStr = Annotated[str, Field(min_length=1)]


def generate_run_id() -> str:
    """
    Format: {ts}{sep}{random}
    Example: 20260204T153012-1a2b3c4d
    """
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())  # use UTC for determinism
    rnd = uuid.uuid4().hex[:8]
    return f"{ts}-{rnd}"


def consolidate_artifacts(original_artifacts: list[Any], new_artifacts: list[Any]) -> list[Any]:
    """
    Combine two lists of ArtifactRef, deduplicating by (path, kind).
    New artifacts take precedence over original ones.

    Args:
        original_artifacts (List[Any]): The original list of artifacts.
        new_artifacts (List[Any]): The new list of artifacts to merge.

    Returns:
        List[Any]: The consolidated list of artifacts.
    """
    artifact_map: dict[tuple[str, str], Any] = {}
    for artifact in original_artifacts + new_artifacts:
        key = (artifact.path, artifact.kind)  # Formerly sha256
        artifact_map[key] = artifact
    return list(artifact_map.values())


def write_feather(df: pd.DataFrame, path: str, index_name: str = "subject_id"):
    """
    Write a DataFrame to a Feather file with optional index column.

    Args:
        df (pd.DataFrame): The DataFrame to write.
        path (str): The file path to write the Feather file to.
        index_name (str): The name of the index column. Defaults to "subject_id".
    """
    df2 = df.reset_index(names=index_name)  # index -> column
    feather.write_feather(df2, path, compression="zstd")


def read_feather(path: str, index_name: str = "subject_id") -> pd.DataFrame:
    """
    Read a Feather file into a DataFrame with optional index column.

    Args:
        path (str): The file path to read the Feather file from.
        index_name (str): The name of the index column. Defaults to "subject_id".

    Returns:
        pd.DataFrame: The DataFrame read from the Feather file.
    """
    df = pd.read_feather(path)
    df.set_index(index_name, inplace=True)  # column -> index
    return df


def load_metadata_aligned_methylation_data(
    accession_code: str, artifacts: list[Any], methylation_data_type: str = "preqc_methylation_data"
) -> pd.DataFrame:
    """
    Load metadata-aligned methylation data for a given accession code.

    Args:
        accession_code (str): The accession code for the dataset.
        artifacts (List[Any]): A list of artifacts containing methylation data and subject mappings.
        methylation_data_type (str): The type of methylation data to load. Defaults to "preqc_methylation_data".

    Returns:
        pd.DataFrame: The loaded and aligned methylation data.
    """
    subject_mapper = next(
        (a for a in artifacts if a.kind == "subject_column_mapping" and a.accession_code == accession_code), None
    )
    methylation_data = get_correct_methylation_data(artifacts, accession_code, artifact_kind=methylation_data_type)
    if methylation_data_type == "postqc_methylation_data" and not methylation_data:
        return load_metadata_aligned_methylation_data(
            accession_code, artifacts, methylation_data_type="preqc_methylation_data"
        )
    methylation_df = read_feather(methylation_data.path, index_name="subject_id")
    if not subject_mapper:
        return methylation_df
    subject_mapper_df = pd.read_csv(subject_mapper.path, index_col=0)
    subject_mapper_df.set_index("Beta_Subjects", inplace=True)
    methylation_df = methylation_df.merge(subject_mapper_df, left_index=True, right_index=True, how="inner")
    methylation_df.set_index("Sample", inplace=True)
    return methylation_df


def _generate_harmonization_todo_id(step: str) -> str:
    """
    Generate a harmonization todo ID based on the given step.

    Args:
        step (str): The harmonization step.

    Returns:
        str: The harmonization todo ID.
    """
    if step == "harmonize_disease_labels":
        return "1"
    elif step == "harmonize_disease_group_labels":
        return "2"
    elif step == "harmonize_tissue_labels":
        return "3"
    elif step == "harmonize_tissue_group_labels":
        return "4"
    elif step == "harmonize_cell_type_labels":
        return "5"
    elif step == "harmonize_sex_labels":
        return "6"
    raise ValueError(f"Unknown harmonization step: {step}")


def _generate_harmonization_step_actions(step: str) -> list[str]:
    """
    Generate the actions for a harmonization step.

    Args:
        step (str): The harmonization step.

    Returns:
        List[str]: A list containing the past tense, present tense, and default message for the step.
    """
    if step == "harmonize_disease_labels":
        return ["Harmonizing", "Harmonized", "Harmonizing disease labels..."]
    elif step == "harmonize_disease_group_labels":
        return ["Harmonizing", "Harmonized", "Harmonizing disease group labels..."]
    elif step == "harmonize_tissue_labels":
        return ["Harmonizing", "Harmonized", "Harmonizing tissue labels..."]
    elif step == "harmonize_tissue_group_labels":
        return ["Harmonizing", "Harmonized", "Harmonizing tissue group labels..."]
    elif step == "harmonize_cell_type_labels":
        return ["Harmonizing", "Harmonized", "Harmonizing cell type labels..."]
    elif step == "harmonize_sex_labels":
        return ["Harmonizing", "Harmonized", "Harmonizing sex labels..."]
    raise ValueError(f"Unknown harmonization step: {step}")


def _generate_harmonization_description_default(step: str, status: str, counts: dict[str, int]) -> str:
    """
    Generate the default description for a harmonization step based on its status and counts.

    Args:
        step (str): The harmonization step.
        status (str): The status of the harmonization step.
        counts (Dict[str, int]): A dictionary containing the counts of completed, failed, and canceled datasets.

    Returns:
        str: The default description for the harmonization step.
    """
    if status != "Completed":
        if step == "harmonize_disease_labels":
            return "Harmonizing raw disease labels..."
        elif step == "harmonize_disease_group_labels":
            return "Finding common ancestors to harmonized disease labels..."
        elif step == "harmonize_tissue_labels":
            return "Harmonizing raw tissue labels..."
        elif step == "harmonize_tissue_group_labels":
            return "Finding common ancestors to harmonized tissue labels..."
        elif step == "harmonize_cell_type_labels":
            return "Harmonizing raw cell type labels..."
        elif step == "harmonize_sex_labels":
            return "Harmonizing raw sex labels..."
    else:
        return f"{counts['completed']} succeeded, {counts['failed']} failed, {counts['canceled']} canceled."
    return f"Working on {step}..."


def generate_harmonization_todo_entry(
    step: str, counts: dict[str, int], num_datasets: int, other_counts: dict[str, dict[str, int]]
) -> dict[str, Any]:
    """
    Generate a harmonization todo entry based on the given step, counts, and other counts.

    Args:
        step (str): The harmonization step.
        counts (Dict[str, int]): A dictionary containing the counts of completed, failed, and canceled datasets.
        num_datasets (int): The total number of datasets.
        other_counts (Dict[str, Dict[str, int]]): A dictionary containing the counts of other harmonization steps.

    Returns:
        Dict[str, Any]: A dictionary representing the harmonization todo entry.
    """
    status = _generate_todo_status(counts, num_datasets)
    id_mapper = {
        "harmonize_disease_labels": "1",
        "harmonize_disease_group_labels": "2",
        "harmonize_tissue_labels": "3",
        "harmonize_tissue_group_labels": "4",
        "harmonize_cell_type_labels": "5",
        "harmonize_sex_labels": "6",
    }
    past_tense, present_tense, default_message = _generate_harmonization_step_actions(step)
    description_default = _generate_harmonization_description_default(step, status, counts)
    id = _generate_harmonization_todo_id(step)
    previous_ids = range(1, int(id))
    previous_steps = [k for k, v in id_mapper.items() if int(v) in previous_ids]
    if any(
        _generate_todo_status(other_counts[previous_step], num_datasets) == "in_progress"
        for previous_step in previous_steps
    ):
        status = "pending"
    elif any(
        _generate_todo_status(other_counts[previous_step], num_datasets) == "failed" for previous_step in previous_steps
    ):
        status = "canceled"
    return {
        "id": id,
        "label": _generate_todo_label(
            status, past_tense, present_tense, num_datasets, counts, default_message=default_message
        ),
        "status": status,
        "description": _generate_todo_description(status, counts, description_default=description_default),
    }


def populate_harmonization_todos(state: Any):
    """
    Populate the harmonization todos for the given state.

    Args:
        state (Any): The current state of the application.

    Returns:
        List[Dict[str, Any]]: A list of harmonization todo entries.
    """
    # Initializate variables
    accession_codes = get_accession_codes(state)
    harmonization_disease_label_counts = retrieve_status_counts(
        get_accession_codes(state), state.datasets, "map_disease_labels_to_ontology"
    )
    harmonization_disease_group_label_counts = retrieve_status_counts(
        get_accession_codes(state), state.datasets, "group_disease_labels"
    )
    harmonize_tissue_label_counts = retrieve_status_counts(
        get_accession_codes(state), state.datasets, "map_tissue_labels_to_ontology"
    )
    harmonize_tissue_group_label_counts = retrieve_status_counts(
        get_accession_codes(state), state.datasets, "group_tissue_labels"
    )
    harmonize_cell_type_label_counts = retrieve_status_counts(
        get_accession_codes(state), state.datasets, "map_cell_type_labels_to_ontology"
    )
    harmonize_sex_label_counts = retrieve_status_counts(
        get_accession_codes(state), state.datasets, "harmonize_sex_labels"
    )

    all_count_dicts = {
        "harmonize_disease_labels": harmonization_disease_label_counts,
        "harmonize_disease_group_labels": harmonization_disease_group_label_counts,
        "harmonize_tissue_labels": harmonize_tissue_label_counts,
        "harmonize_tissue_group_labels": harmonize_tissue_group_label_counts,
        "harmonize_cell_type_labels": harmonize_cell_type_label_counts,
        "harmonize_sex_labels": harmonize_sex_label_counts,
    }

    num_datasets = len(accession_codes)
    todos = [
        generate_harmonization_todo_entry(
            "harmonize_disease_labels", harmonization_disease_label_counts, num_datasets, all_count_dicts
        ),
        generate_harmonization_todo_entry(
            "harmonize_disease_group_labels", harmonization_disease_group_label_counts, num_datasets, all_count_dicts
        ),
        generate_harmonization_todo_entry(
            "harmonize_tissue_labels", harmonize_tissue_label_counts, num_datasets, all_count_dicts
        ),
        generate_harmonization_todo_entry(
            "harmonize_tissue_group_labels", harmonize_tissue_group_label_counts, num_datasets, all_count_dicts
        ),
        generate_harmonization_todo_entry(
            "harmonize_cell_type_labels", harmonize_cell_type_label_counts, num_datasets, all_count_dicts
        ),
        generate_harmonization_todo_entry(
            "harmonize_sex_labels", harmonize_sex_label_counts, num_datasets, all_count_dicts
        ),
    ]

    return todos


def update_harmonization_progress_tracker(state: Any) -> ToolMessage:
    """
    Updates the progress tracker for dataset label harmonization based on the current state of the workflow.

    Args:
        state (Any): The current state containing the datasets and their statuses.

    Returns:
        ToolMessage: A message object representing the updated progress tracker for harmonization.
    """

    hash = _get_status_id(state.messages, "harmonization")

    # Hash from accession codes + user message + user message time
    message = ToolMessage(
        id=hash,
        content="",
        tool_call_id=hash,
        artifact={
            "id": hash,
            "title": "Harmonization Progress",
            "description": "Tracking the progress of harmonizing dataset labels",
            "todos": populate_harmonization_todos(state),
        },
        additional_kwargs={
            "name": "geoRetrievalProgress",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return message


CURRENT_FILE = Path(__file__).resolve()


def find_project_root_with_data(start: Path = CURRENT_FILE, marker_dir: str = "data") -> Path | None:
    """
    Walk up parents and return the first directory that contains `marker_dir`.
    Returns None if not found.

    Args:
        start (Path): The starting path to begin the search. Defaults to the current file's path.
        marker_dir (str): The name of the directory to look for as a marker of the project root. Defaults to "data".

    Returns:
        Optional[Path]: The path to the project root if found, otherwise None.
    """
    for ancestor in [start, *start.parents]:
        if (ancestor / marker_dir).is_dir():
            return ancestor
    return None


PROJECT_ROOT = find_project_root_with_data()
if PROJECT_ROOT is None:
    # fallback: if you know exactly how many parents to go up, set it explicitly here
    # For your layout helper.py -> utils -> package -> src -> project_root, parents[3] is project root
    PROJECT_ROOT = CURRENT_FILE.parents[3]
