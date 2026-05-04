__all__ = [
    "format_methylation_data",
    "format_individual_methylation_data",
    "merge_formatted_supplementary_data",
    "_create_subject_id_mapping",
]
import asyncio
import gzip
import json
import os
import random
import re
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from ollama._types import ResponseError
from pydantic import ValidationError
from rapidfuzz import fuzz, process

from ...agent.graphs.deps import Deps
from ...agent.state.models import GeoDatasetState, GEOIngestionConfig, GeoIngestionSubgraphState
from ...contracts.common import ArtifactRef
from ...contracts.geo import (
    ForcedSampleDataResolution,
    LexFeat,
    SampleDataResolution,
)
from ...utils.examples import (
    generate_column_interpretation_examples,
    generate_column_interpretation_examples_no_detection,
)
from ...utils.helper import compute_sha256, consolidate_artifacts, write_feather
from ...utils.prompting import (
    generate_infer_methylation_data_column_scheme_alt_prompt,
    generate_infer_methylation_data_column_scheme_prompt,
)
from .extract_sample_level_metadata import _merge_to_dataframe

CALL_TIMEOUT = 180
GLOBAL_RETRY_LIMIT = 5


def _check_for_cpg_probes(sample_data: pd.DataFrame) -> dict[str, bool]:
    """Determine whether CpG probes are in rows or columns.

    Counts how many indices and column headers contain "cg" and returns
    a dict indicating the predominant layout.

    Args:
        sample_data: A methylation data DataFrame.

    Returns:
        A dict with keys "columns" and "rows", each True if that axis
        contains the majority of CpG probe identifiers.
    """
    cols = np.asarray(sample_data.columns, dtype=str)
    rows = np.asarray(sample_data.index, dtype=str)

    n_cg_cols = (np.char.find(np.char.lower(cols), "cg") >= 0).sum()
    n_cg_rows = (np.char.find(np.char.lower(rows), "cg") >= 0).sum()

    return {
        "columns": n_cg_cols > n_cg_rows,
        "rows": n_cg_rows > n_cg_cols,
    }


def _check_for_detection_columns(sample_data: pd.DataFrame) -> bool:
    """Heuristic check for presence of detection p-value columns.

    Returns True when the number of columns is less than the number of
    rows, which suggests paired beta/detection columns rather than a
    matrix with CpG probes as rows.

    Args:
        sample_data: A methylation data DataFrame.

    Returns:
        True if detection columns are likely present.
    """
    columns = sample_data.columns.tolist()
    # detection_cols = [col for col in columns if "detect" in col.lower()] # May not have detect in the name
    columns_lt_rows = len(columns) < len(sample_data.index)  # There will almost certainly be more probes than samples
    # columns_even = len(columns) % 2 == 0 # Thoughts, doesn't necessarily have to be so
    return columns_lt_rows


async def _process_detection_columns(
    artifact: ArtifactRef, sample_data: pd.DataFrame, config: RunnableConfig, messages: list[AnyMessage]
) -> tuple[pd.DataFrame, ArtifactRef]:
    """Process methylation data with paired beta and detection p-value columns.

    Uses LLM-assisted column scheme resolution to identify beta and
    detection columns, then applies a vectorized filter to retain only
    beta values with detection p < 0.05.

    Args:
        artifact: Artifact reference for the sample data file.
        sample_data: Raw methylation DataFrame read from the artifact.
        config: LangChain runnable config providing LLM dependencies.
        messages: Prior conversation messages for context.

    Returns:
        A tuple of (filtered_methylation_df, column_scheme_artifact)
        where filtered_methylation_df has CpG probes as columns and
        samples as rows.

    Raises:
        ValueError: If the column scheme cannot be resolved.
    """
    start_time = datetime.now()
    column_scheme, artifact = await _get_column_scheme(artifact, sample_data, config)
    end_time = datetime.now()
    column_scheme_time_taken = (end_time - start_time).total_seconds()
    print(f"\nTime taken to determine column scheme: {column_scheme_time_taken} seconds\n")

    # Early Exit for Error States
    if all(s.status in ["error", "missing"] for s in [column_scheme.beta_column, column_scheme.detection_column]):
        raise ValueError(f"Unable to determine column scheme for dataset {artifact.accession_code}")

    # 1. Vectorized Numeric Conversion
    # Converting the whole dataframe once is significantly faster than per-column in a loop
    sample_data = sample_data.apply(pd.to_numeric, errors="coerce")

    # 2. Extract Patterns
    beta_pat = (
        re.compile(column_scheme.beta_column.pattern, re.IGNORECASE)
        if column_scheme.beta_column.status == "resolved"
        else None
    )
    det_pat = (
        re.compile(column_scheme.detection_column.pattern, re.IGNORECASE)
        if column_scheme.detection_column.status == "resolved"
        else None
    )

    if not beta_pat:
        return pd.DataFrame(), artifact

    # Identify Column Indices
    start_time = datetime.now()
    beta_cols = [c for c in sample_data.columns if beta_pat.search(c)]
    det_cols = [c for c in sample_data.columns if det_pat and det_pat.search(c)]
    end_time = datetime.now()
    pattern_matching_time_taken = (end_time - start_time).total_seconds()
    print(f"\nTime taken to identify columns based on patterns: {pattern_matching_time_taken} seconds\n")

    # 3. Handle Case with No Detection Columns
    if not det_pat or not det_cols:
        start_time = datetime.now()
        methylation_df = _matrix_shape_check(sample_data[beta_cols])
        end_time = datetime.now()
        no_detection_time_taken = (end_time - start_time).total_seconds()
        print(f"\nTime taken to process data without detection columns: {no_detection_time_taken} seconds\n")
        return methylation_df, artifact

    start_time = datetime.now()
    score_matrix = lexical_score_matrix(beta_cols, det_cols)
    best_j = np.argmax(score_matrix, axis=1)
    mapper = {beta_cols[i]: det_cols[j] for i, j in enumerate(best_j)}
    det_cols = [mapper[beta_col] for beta_col in beta_cols]
    end_time = datetime.now()
    mapping_time_taken = (end_time - start_time).total_seconds()
    print(f"\nTime taken to map beta columns to detection columns: {mapping_time_taken} seconds\n")

    # 4. High-Speed Vectorized Filtering
    # Slice the dataframe into two aligned matrices
    beta_matrix = sample_data[beta_cols].values
    det_matrix = sample_data[det_cols].values

    # Check for identical columns (your specific logic) and apply the 0.05 threshold
    # np.where is much faster than iterating and renaming
    mask = np.where(np.array_equal(beta_matrix, det_matrix), 0.0, det_matrix) < 0.05

    # Apply mask: Keep beta value if p < 0.05, else NaN
    # Masking is an O(n) operation in memory
    filtered_values = np.where(mask, beta_matrix, np.nan)

    # 5. Reconstruct DataFrame
    # We transpose and reconstruct to match your "Sample as Index" requirement
    start_time = datetime.now()
    methylation_df = pd.DataFrame(filtered_values.T, index=beta_cols, columns=sample_data.index)
    end_time = datetime.now()
    reconstruction_time_taken = (end_time - start_time).total_seconds()
    print(f"\nTime taken to reconstruct filtered DataFrame: {reconstruction_time_taken} seconds\n")

    # Drop columns (CpGs) that are now all NaN across all samples if necessary
    # methylation_df.dropna(axis=1, how='all', inplace=True)

    return methylation_df, artifact


async def _process_detection_columns_alt(
    artifact: ArtifactRef, sample_data: pd.DataFrame, config: RunnableConfig, messages: list[AnyMessage]
) -> tuple[pd.DataFrame, ArtifactRef]:
    """Alternative detection-column processing via per-subject pair extraction.

    Resolves column scheme with LLM, then processes each beta/detection
    column pair individually (iterative approach).  Suitable when the
    vectorized approach in _process_detection_columns is not a fit.

    Args:
        artifact: Artifact reference for the sample data file.
        sample_data: Raw methylation DataFrame.
        config: LangChain runnable config providing LLM dependencies.
        messages: Prior conversation messages for context.

    Returns:
        A tuple of (filtered_methylation_df, column_scheme_artifact).

    Raises:
        ValueError: If the column scheme cannot be resolved.
    """
    column_scheme, artifact = await _get_column_scheme(artifact, sample_data, config)
    if all(s.status in ["error", "missing"] for s in [column_scheme.beta_column, column_scheme.detection_column]):
        raise ValueError(
            f"Unable to determine column scheme for dataset {artifact.accession_code} with artifact {artifact.path}. Beta column notes: {column_scheme.beta_column.notes}, Detection column notes: {column_scheme.detection_column.notes}"
        )

    beta_pattern = (
        re.compile(column_scheme.beta_column.pattern, re.IGNORECASE)
        if column_scheme.beta_column.status == "resolved"
        else None
    )
    beta_columns = [idx for idx, col in enumerate(sample_data.columns) if beta_pattern.search(col)]

    detection_pattern = (
        re.compile(column_scheme.detection_column.pattern, re.IGNORECASE)
        if column_scheme.detection_column.status == "resolved"
        else None
    )
    detection_columns = [
        idx for idx, col in enumerate(sample_data.columns) if detection_pattern and detection_pattern.search(col)
    ]

    if not beta_pattern:
        return pd.DataFrame(), artifact
    elif not detection_pattern:
        methylation_df = _matrix_shape_check(sample_data)
        return methylation_df, artifact

    paired_columns = [
        [sample_data.columns.tolist()[beta_col], sample_data.columns.tolist()[detection_col]]
        for beta_col, detection_col in zip(beta_columns, detection_columns)
    ]
    methylation_rows = []
    methylation_columns = []
    for pair in paired_columns:
        data = sample_data[pair].copy()
        subject_name = pair[0].strip()
        data.rename(columns={pair[0]: "Value", pair[1]: "Detection_Pval"}, inplace=True)
        data["ID_REF"] = data.index
        data["Detection_Pval"] = pd.to_numeric(data["Detection_Pval"], errors="coerce")
        data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
        if data["Value"].equals(data["Detection_Pval"]):
            data["Detection_Pval"] = 0.0
        data = data[data["Detection_Pval"] < 0.05].copy()

        methylation_rows.append([subject_name] + data["Value"].tolist())
        methylation_columns.append(["Sample"] + data["ID_REF"].tolist())

    methylation_df = _merge_to_dataframe(methylation_rows, methylation_columns, index_col="Sample")
    return methylation_df, artifact


def _matrix_shape_check(sample_data: pd.DataFrame) -> pd.DataFrame:
    """Ensure CpG probes are columns and samples are rows.

    Detects whether CpG probes reside on the index or columns axis,
    then transposes if necessary to produce a sample-x-CpG layout.
    Only rows/columns starting with "cg" (case-insensitive) are kept.

    Args:
        sample_data: A methylation DataFrame.

    Returns:
        A DataFrame with samples as rows and CpG probes as columns.
    """
    data_shape = _check_for_cpg_probes(sample_data)

    if data_shape["rows"]:
        mask = sample_data.index.str.lower().str.startswith("cg", na=False)
        cpg_rows = sample_data.index[mask].sort_values()
        return sample_data.loc[cpg_rows].T

    mask = sample_data.columns.str.lower().str.startswith("cg", na=False)
    cpg_cols = sample_data.columns[mask].sort_values()
    return sample_data.loc[:, cpg_cols]


def _identify_delimiter(first_line: str) -> str:
    """Detect the field delimiter from a header line.

    Checks for tab, comma, or space in order of precedence.

    Args:
        first_line: The first non-comment, non-empty line of the file.

    Returns:
        The delimiter character ("\t", ",", or " ").

    Raises:
        ValueError: If none of the expected delimiters is found.
    """
    if "\t" in first_line:
        return "\t"
    elif "," in first_line:
        return ","
    elif " " in first_line:
        return " "
    else:
        raise ValueError("Unable to identify delimiter. Expected tab, comma, or space.")


def _read_sample_data(file_path: str) -> pd.DataFrame:
    """Read a supplementary methylation data file into a DataFrame.

    Handles gzipped files.  Auto-detects the delimiter, skips comment
    lines (starting with "#"), and attempts to set the index to the
    first column containing mostly "cg"-prefixed values.

    Args:
        file_path: Path to the data file (may be .gz).

    Returns:
        A DataFrame with cleaned, lowercased column names.

    Raises:
        ValueError: If no valid header line is found within the first
            20 lines of the file.
    """
    if file_path.endswith(".gz"):
        open_func = gzip.open
    else:
        open_func = open
    with open_func(file_path, "rt") as f:
        first_line = ""
        counter = 0
        for line in f:
            counter += 1
            if not line.startswith("#") and not line.strip() == "":
                first_line = line
                break
            if counter > 20:
                raise ValueError("Unable to find a valid header line within the first 20 lines of the file.")
    delimiter = _identify_delimiter(first_line)
    sample_data = pd.read_csv(file_path, sep=delimiter, comment="#", index_col=False)
    for col in sample_data.columns.tolist()[:5]:
        matching_cols = [x for x in sample_data[col].tolist() if str(x).startswith("cg")]
        if len(matching_cols) / sample_data.shape[0] > 0.9:
            sample_data = sample_data.set_index(col)
            sample_data = sample_data[~sample_data.index.isna()]
            break
    sample_data.columns = [x.lower().strip() for x in sample_data.columns.tolist()]
    return sample_data


def _generate_data_samples(sample_data: pd.DataFrame, seed: int = 0) -> tuple[str, pd.DataFrame]:
    """Generate a random column subset for LLM prompts.

    Samples up to 15 columns with a reproducible seed and returns both
    a markdown table (for prompts) and the sampled DataFrame.

    Args:
        sample_data: The full methylation DataFrame.
        seed: Random seed for reproducible column selection.

    Returns:
        A tuple of (markdown_string, sampled_dataframe).
    """
    random.seed(seed)
    n_cols = sample_data.shape[1]
    k = min(15, n_cols)
    idx = np.random.choice(n_cols, size=k, replace=False)
    sampled_columns = sample_data.columns[idx]
    sample_data_markdown = sample_data[sampled_columns].head(5).to_markdown(index=False)
    return sample_data_markdown, sample_data[sampled_columns].copy()


def _check_pattern_performance(pattern: str, columns: list[str]) -> tuple[set, set]:
    """Evaluate a regex pattern against a list of column names.

    Args:
        pattern: A regex pattern string (may be empty/None).
        columns: Column names to test the pattern against.

    Returns:
        A tuple of (matching_columns_set, missing_columns_set).
    """
    if not pattern:
        return set(), set(columns)
    regex = re.compile(pattern, re.IGNORECASE)
    matching_cols = [col for col in sorted(columns) if regex.search(col)]
    missing_cols = [col for col in sorted(columns) if col not in matching_cols]
    return set(matching_cols), set(missing_cols)


def _check_pattern_performance_change(
    prev_beta_pattern: str,
    prev_detection_pattern: str | None,
    current_beta_pattern: str,
    current_detection_pattern: str | None,
    columns: list[str],
) -> bool:
    """Check if pattern refinement changed what was matched.

    Compares the sets of matching/missing columns for both beta and
    detection patterns between two attempts.

    Args:
        prev_beta_pattern: Previous beta column regex pattern.
        prev_detection_pattern: Previous detection column regex pattern
            (may be None).
        current_beta_pattern: Current beta column regex pattern.
        current_detection_pattern: Current detection column regex
            pattern (may be None).
        columns: The full list of column names.

    Returns:
        True if both beta and detection matching/missing sets are
        identical between attempts.
    """
    # Beta pattern performance
    prev_beta_matching_cols, prev_beta_missing_cols = _check_pattern_performance(prev_beta_pattern, columns)
    current_beta_matching_cols, current_beta_missing_cols = _check_pattern_performance(current_beta_pattern, columns)
    # Detection pattern performance (if applicable)
    prev_detection_matching_cols, prev_detection_missing_cols = _check_pattern_performance(
        prev_detection_pattern, columns
    )
    current_detection_matching_cols, current_detection_missing_cols = _check_pattern_performance(
        current_detection_pattern, columns
    )

    return (
        (prev_beta_matching_cols == current_beta_matching_cols)
        and (prev_beta_missing_cols == current_beta_missing_cols)
        and (prev_detection_matching_cols == current_detection_matching_cols)
        and (prev_detection_missing_cols == current_detection_missing_cols)
    )


async def _get_column_scheme(
    artifact: ArtifactRef,
    sample_data: pd.DataFrame,
    config: RunnableConfig,
    messages: list[AIMessage] = [],
    count=0,
    prohibited_patterns: list[str] | None = None,
    prev_beta_pattern: str | None = None,
    prev_detection_pattern: str | None = None,
) -> tuple[SampleDataResolution, ArtifactRef]:
    """Resolve beta/detection column patterns via LLM with iterative refinement.

    Sends sampled column data to the deterministic LLM for structured
    column scheme inference.  Retries on timeout or validation errors,
    and refines patterns across up to 3 correction attempts.  The final
    scheme is persisted as a JSON artifact.

    Args:
        artifact: Artifact reference for the sample data file.
        sample_data: Raw methylation DataFrame.
        config: LangChain runnable config providing LLM dependencies.
        messages: Accumulated conversation messages (used for
            correction attempts).
        count: Current attempt number (0-indexed, max 3 refinement
            attempts).
        prohibited_patterns: Patterns to instruct the LLM to avoid.
        prev_beta_pattern: Previous attempt's beta pattern for
            convergence detection.
        prev_detection_pattern: Previous attempt's detection pattern
            for convergence detection.

    Returns:
        A tuple of (resolved_column_scheme, scheme_artifact_ref).

    Raises:
        ValueError: If the LLM fails to produce a valid scheme after
            all retries.
    """
    deps: Deps = config["configurable"]["deps"]
    deterministic_llm = deps.deterministic_llm
    default_llm = deps.default_llm
    print(
        f"\nGetting column scheme for artifact {artifact.path} with accession code {artifact.accession_code}, attempt {count + 1}"
    )
    sample_data_markdown, _ = _generate_data_samples(sample_data, seed=count)
    example_one_data_markdown, example_one_answer = generate_column_interpretation_examples()
    example_two_data_markdown, example_two_answer = generate_column_interpretation_examples(alt=True)
    example_three_data_markdown, example_three_answer = generate_column_interpretation_examples_no_detection()

    if not messages:
        message = generate_infer_methylation_data_column_scheme_prompt(
            example_one_query=example_one_data_markdown,
            example_one_response=json.dumps(example_one_answer, indent=2, ensure_ascii=False),
            example_two_query=example_two_data_markdown,
            example_two_response=json.dumps(example_two_answer, indent=2, ensure_ascii=False),
            example_three_query=example_three_data_markdown,
            example_three_response=json.dumps(example_three_answer, indent=2, ensure_ascii=False),
            prohibit_patterns=len(prohibited_patterns) > 0 if prohibited_patterns else False,
            prohibited_patterns=", ".join([f"`{p}`" for p in prohibited_patterns]) if prohibited_patterns else None,
            json_schema=ForcedSampleDataResolution.model_json_schema(),
        )

        system_message = SystemMessage(
            id=uuid.uuid4().hex,
            content=message,
            additional_kwargs={
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        human_message = HumanMessage(
            id=uuid.uuid4().hex,
            content=f"Please find the resolution for the following DNA methylation data:\n{sample_data_markdown}",
            additional_kwargs={
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        messages = [system_message, human_message]

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    while retries < retry_limit:
        try:
            resolved: Any = await asyncio.wait_for(
                deterministic_llm.acall_structured(messages, ForcedSampleDataResolution), timeout=CALL_TIMEOUT
            )
            break
        except TimeoutError:
            retries += 1
            continue
        except ResponseError:
            retries += 1
            continue
        except ValidationError as e:
            resolved: Any = SampleDataResolution(
                beta_column={"status": "error", "notes": [f"Validation error for beta column extraction: {e}"]},
                detection_column={
                    "status": "error",
                    "notes": [f"Validation error for detection column extraction: {e}"],
                },
            )
            break

    new_sample_data_markdown, new_columns = _generate_data_samples(sample_data, seed=count + 1)

    if prev_beta_pattern or prev_detection_pattern:
        if _check_pattern_performance_change(
            prev_beta_pattern,
            prev_detection_pattern,
            resolved.beta_column.pattern,
            resolved.detection_column.pattern if resolved.detection_column.status == "resolved" else None,
            new_columns.columns.tolist(),
        ):
            count = 4

    # Otherwise, if we are under the retry limit, try to fix
    if count < 3:
        agent_message = AIMessage(
            id=uuid.uuid4().hex,
            content=f"Attempt {count} resolution:\n {resolved.model_dump()}",
            additional_kwargs={
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

        beta_pattern = re.compile(resolved.beta_column.pattern, re.IGNORECASE)
        beta_columns = [col for col in sorted(new_columns.columns.tolist()) if beta_pattern.search(col)]
        missing_beta_columns = [col for col in sorted(new_columns.columns.tolist()) if col not in beta_columns]
        print(f"\n Beta pattern: {beta_pattern}")
        print(f"\nBeta columns identified with pattern {resolved.beta_column.pattern}: {beta_columns}")
        print(f"\nColumns that failed to match beta pattern {resolved.beta_column.pattern}: {missing_beta_columns}")

        detection_pattern = (
            re.compile(resolved.detection_column.pattern, re.IGNORECASE)
            if resolved.detection_column.status == "resolved"
            else None
        )
        detection_columns = [
            col for col in sorted(new_columns.columns.tolist()) if detection_pattern and detection_pattern.search(col)
        ]
        missing_detection_columns = [
            col for col in sorted(new_columns.columns.tolist()) if col not in detection_columns
        ]
        if detection_pattern:
            print(f"\n Detection pattern: {detection_pattern}")
            print(
                f"\nDetection columns identified with pattern {resolved.detection_column.pattern}: {detection_columns}"
            )
            print(
                f"\nColumns that failed to match detection pattern {resolved.detection_column.pattern}: {missing_detection_columns}"
            )

        # Cases
        complete_failure = len(beta_columns) == 0
        permissive_beta = (len(missing_beta_columns) == 0) and len(detection_columns) > 0
        double_checking_work = len(missing_beta_columns) > 0

        if complete_failure or permissive_beta or double_checking_work:
            base_prompt = generate_infer_methylation_data_column_scheme_alt_prompt(
                model_response=json.dumps(resolved.model_dump(), indent=2, ensure_ascii=False),
                sample_data=new_sample_data_markdown,
                random_column=new_columns.columns.tolist()[0],
                beta_pattern=resolved.beta_column.pattern,
                beta_columns=", ".join(beta_columns)
                if len(beta_columns) > 0
                else "N/A, because your pattern failed to match any columns.",
                not_beta_columns=", ".join(missing_beta_columns)
                if len(missing_beta_columns) > 0
                else "N/A, all columns were identified as being `beta_column`s.",
                detection_pattern=resolved.detection_column.pattern
                if resolved.detection_column.status == "resolved"
                else "N/A, because you don't believe there are any detection columns.",
                detection_column=", ".join(detection_columns)
                if len(detection_columns) > 0
                else "N/A, because you don't believe there are any detection columns.",
                not_detection_columns=", ".join(missing_detection_columns)
                if len(missing_detection_columns) > 0
                else "N/A, because you don't believe there are any detection columns.",
                json_schema=ForcedSampleDataResolution.model_json_schema(),
            )
            correction_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=base_prompt,
                additional_kwargs={"created_at": datetime.now(UTC).isoformat()},
            )
            return await _get_column_scheme(
                artifact,
                sample_data,
                config,
                messages=messages + [agent_message, correction_message],
                count=count + 1,
                prev_beta_pattern=resolved.beta_column.pattern,
                prev_detection_pattern=resolved.detection_column.pattern
                if resolved.detection_column.status == "resolved"
                else None,
            )

    column_scheme_path = os.path.splitext(artifact.path)[0] + ".json"
    with open(column_scheme_path, "w", encoding="utf-8") as f:
        json.dump(resolved.model_dump(), f, ensure_ascii=False, indent=2)
    column_scheme_artifact = ArtifactRef.model_validate(
        {
            "accession_code": artifact.accession_code,
            "path": column_scheme_path,
            "kind": "supplementary_file_methylation_data_column_scheme",
            "sha256": compute_sha256(column_scheme_path, is_path=True),
            "bytes": os.path.getsize(column_scheme_path),
            "created_at": datetime.now(UTC).isoformat(),
        }
    )

    return resolved, column_scheme_artifact


async def format_individual_methylation_data(
    accession_code: str,
    return_dict: dict[str, Any],
    config: RunnableConfig,
    artifact: ArtifactRef,
    messages: list[AnyMessage],
) -> dict[str, Any]:
    """Format a single supplementary methylation data file.

    Reads raw data, processes detection columns if present, shapes the
    matrix, and writes a Feather file.  Updates the return dict with
    the new artifact and marks the dataset as running.

    Args:
        accession_code: GEO accession identifier.
        return_dict: Mutable result dict with "config" and "datasets"
            keys.
        config: LangChain runnable config providing LLM dependencies.
        artifact: Artifact reference for the raw data file.
        messages: Prior conversation messages for context.

    Returns:
        The updated return_dict with the formatted methylation artifact
        appended and supplementary_data state marked.
    """
    methylation_dataframe_output_path = f"{os.path.splitext(artifact.path)[0]}_proc.feather"

    sample_data = _read_sample_data(artifact.path)
    if _check_for_detection_columns(sample_data):
        df, _ = await _process_detection_columns(artifact, sample_data, config, messages)
    else:
        df = _matrix_shape_check(sample_data)

    write_feather(df, methylation_dataframe_output_path, index_name="subject_id")
    methylation_artifact = ArtifactRef.model_validate(
        {
            "accession_code": accession_code,
            "path": methylation_dataframe_output_path,
            "kind": "supplementary_file_methylation_data_formatted",
            "sha256": compute_sha256(methylation_dataframe_output_path, is_path=True),
            "bytes": os.path.getsize(methylation_dataframe_output_path),
            "created_at": datetime.now(UTC).isoformat(),
        }
    )

    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]], [methylation_artifact]
    )
    return_dict["datasets"][accession_code]["supplementary_data"][artifact.sha256] = "running"
    return return_dict


def merge_formatted_supplementary_data(state: GeoIngestionSubgraphState, accession_code: str) -> pd.DataFrame:
    """Concatenate all formatted supplementary data for a GEO accession.

    Finds all formatted methylation artifacts belonging to the given
    accession and row-wise concatenates them.

    Args:
        state: The geo ingestion subgraph state containing config and
            datasets.
        accession_code: GEO accession identifier.

    Returns:
        A DataFrame of concatenated formatted methylation data.
    """
    state_config = state.config
    dataset_state = state.datasets[accession_code]
    print(f"\nDataset supplementary data state: {dataset_state.supplementary_data}")
    formatted_artifacts = [
        a
        for a in state_config.artifacts
        if (a.kind == "supplementary_file_methylation_data_formatted") and (a.accession_code == accession_code)
    ]
    formatted_datasets = [pd.read_csv(artifact.path, index_col=0) for artifact in formatted_artifacts]
    merged_data = pd.concat(formatted_datasets, axis=0)
    return merged_data


async def format_methylation_data(
    state_config: GEOIngestionConfig, state: GeoDatasetState, config: RunnableConfig, messages: list[AnyMessage]
) -> dict[str, Any]:
    """Format all supplementary methylation files for a dataset.

    Iterates over every supplementary methylation data artifact for
    the accession, processes detection columns where present, and
    concatenates the results into a single pre-QC methylation matrix
    CSV.  New artifacts are consolidated into the return dict.

    Args:
        state_config: GEO ingestion configuration with artifact list.
        state: Per-dataset state containing accession and output
            directory.
        config: LangChain runnable config providing LLM dependencies.
        messages: Prior conversation messages for context.

    Returns:
        A dict with the updated config (including new artifacts).
    """
    formatted_datasets = []
    accession_code = state.accession
    methylation_dataframe_output_path = os.path.join(state.output_dir, "preqc_methylation_matrix.csv")
    return_dict = {"config": state_config.model_dump()}

    sample_data_artifacts = [
        a
        for a in state_config.artifacts
        if (a.kind == "supplementary_file_methylation_data") and (a.accession_code == accession_code)
    ]
    sample_datasets = [_read_sample_data(artifact.path) for artifact in sample_data_artifacts]
    collected_artifacts = []

    for sample_data_artifact, sample_data in zip(sample_data_artifacts, sample_datasets):
        if _check_for_detection_columns(sample_data):
            df, artifact = await _process_detection_columns(sample_data_artifact, sample_data, config, messages)
            collected_artifacts.append(artifact)
        else:
            df = _matrix_shape_check(sample_data)
        formatted_datasets.append(df)
    formatted_data = pd.concat(formatted_datasets, axis=0)

    if not os.path.exists(methylation_dataframe_output_path):
        formatted_data.to_csv(methylation_dataframe_output_path, index=True)
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
    else:
        if next((a for a in state_config.artifacts if a.path == methylation_dataframe_output_path), None) is None:
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
    collected_artifacts.append(methylation_artifact)

    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]], collected_artifacts
    )
    return return_dict


def generate_lexical_features(name: str) -> LexFeat:
    """Tokenize and normalize a column name into lexical features.

    Splits on camelCase boundaries, separators, and alphanumeric
    transitions, then partitions tokens into numeric and word groups.

    Args:
        name: A raw column or sample identifier string.

    Returns:
        A LexFeat named tuple with raw, normalized, token, numeric, and
        word representations.
    """
    CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
    SEP_RE = re.compile(r"[_\-/]+")
    NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9\s]+")
    WS_RE = re.compile(r"\s+")

    NUM_RE = re.compile(r"^\d+$")

    ALNUM_BOUND_RE = re.compile(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)")

    def normalize_text(s: str) -> str:
        s = s.strip()
        s = CAMEL_RE.sub(" ", s)
        s = SEP_RE.sub(" ", s)
        s = ALNUM_BOUND_RE.sub(" ", s)  # <-- ADD THIS
        s = NON_ALNUM_RE.sub(" ", s)
        s = WS_RE.sub(" ", s)
        return s.lower().strip()

    def tokenize(s: str) -> list[str]:
        # Keep numbers as tokens
        t = normalize_text(s).split()
        return t

    def featurize(name: str) -> LexFeat:
        if not isinstance(name, str):
            print(f"Warning: {name} is not a string.")
        norm = normalize_text(name)
        toks = tokenize(name)
        nums = tuple([t for t in toks if NUM_RE.match(t)])
        words = tuple([t for t in toks if not NUM_RE.match(t)])
        return LexFeat(raw=name, norm=norm, tokens=tuple(toks), nums=nums, words=words)

    return featurize(name)


def build_vocab(featsA: list[LexFeat], featsB: list[LexFeat], attr: str) -> dict[str, int]:
    """Build a vocabulary index from a lexical feature attribute.

    Scans the given attribute (e.g. "words" or "nums") across both
    feature lists and assigns a unique integer index to each token.

    Args:
        featsA: Lexical features from the first string list.
        featsB: Lexical features from the second string list.
        attr: The LexFeat attribute to index (e.g. "words", "nums").

    Returns:
        A dict mapping each unique token to its integer index.
    """
    vocab: dict[str, int] = {}
    for f in featsA:
        for t in set(getattr(f, attr)):
            if t not in vocab:
                vocab[t] = len(vocab)
    for f in featsB:
        for t in set(getattr(f, attr)):
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def incidence_matrix(feats: list[LexFeat], vocab: dict[str, int], attr: str) -> np.ndarray:
    """Build a binary incidence matrix for a lexical feature attribute.

    Each row corresponds to a feature and each column to a vocabulary
    token; the entry is 1 if the token is present in that feature.

    Args:
        feats: Lexical features to encode.
        vocab: Token-to-index mapping.
        attr: The LexFeat attribute to use (e.g. "words", "nums").

    Returns:
        A (len(feats), len(vocab)) uint8 binary matrix.
    """
    X = np.zeros((len(feats), len(vocab)), dtype=np.uint8)
    for i, f in enumerate(feats):
        for t in set(getattr(f, attr)):
            X[i, vocab[t]] = 1
    return X


def jaccard_matrix(A_bin: np.ndarray, B_bin: np.ndarray) -> np.ndarray:
    """
    A_bin: (N, V) {0,1}
    B_bin: (M, V) {0,1}
    Returns: (N, M) float32 Jaccard
    """
    # Intersection counts via dot product (N,M)
    inter = (A_bin.astype(np.int16) @ B_bin.astype(np.int16).T).astype(np.float32)

    a_sum = A_bin.sum(axis=1, dtype=np.int16).astype(np.float32)[:, None]  # (N,1)
    b_sum = B_bin.sum(axis=1, dtype=np.int16).astype(np.float32)[None, :]  # (1,M)

    union = a_sum + b_sum - inter  # (N,M)
    # If both empty sets => define Jaccard as 1.0 (matches your earlier behavior)
    out = np.where(union > 0, inter / union, 1.0).astype(np.float32)
    return out


def number_compatibility_matrix(A_nums_bin: np.ndarray, B_nums_bin: np.ndarray) -> np.ndarray:
    """
    Implements:
      if neither has numbers -> 1.0
      if only one has numbers -> 0.6
      if both have numbers -> overlap/union (Jaccard on nums)
    """
    inter = (A_nums_bin.astype(np.int16) @ B_nums_bin.astype(np.int16).T).astype(np.float32)
    a_sum = A_nums_bin.sum(axis=1, dtype=np.int16).astype(np.float32)[:, None]
    b_sum = B_nums_bin.sum(axis=1, dtype=np.int16).astype(np.float32)[None, :]
    union = a_sum + b_sum - inter

    a_has = (a_sum > 0).astype(np.float32)  # (N,1)
    b_has = (b_sum > 0).astype(np.float32)  # (1,M)

    neither = (a_has == 0) & (b_has == 0)
    xor = (a_has + b_has) == 1
    both = (a_has == 1) & (b_has == 1)

    out = np.empty_like(union, dtype=np.float32)
    out[neither] = 1.0
    out[xor] = 0.6
    out[both] = np.where(union[both] > 0, inter[both] / union[both], 1.0).astype(np.float32)
    return out


def token_count_penalty_matrix(featsA, featsB, gamma: float = 2.0) -> np.ndarray:
    """
    Penalize large differences in number of tokens. Returns (N,M) in [0,1].
    gamma controls harshness.
    """
    a_n = np.array([len(f.tokens) for f in featsA], dtype=np.float32)[:, None]
    b_n = np.array([len(f.tokens) for f in featsB], dtype=np.float32)[None, :]
    max_n = np.maximum(a_n, b_n)
    rel = np.where(max_n > 0, np.abs(a_n - b_n) / max_n, 0.0)
    return np.exp(-gamma * rel).astype(np.float32)


def lexical_score_matrix(A: list[str], B: list[str], workers: int = -1, no_count_penalty: bool = False) -> np.ndarray:
    """Compute a lexical similarity matrix between two string lists.

    Combines token-sort-ratio fuzziness, word-level Jaccard, and
    numeric compatibility with optional token-count penalty.

    Args:
        A: First list of strings (e.g. beta column names).
        B: Second list of strings (e.g. detection column names).
        workers: Number of parallel workers for rapidfuzz
            (-1 = all CPUs).
        no_count_penalty: If True, skip the token-count penalty.

    Returns:
        An (N, M) float32 similarity matrix where N = len(A) and
        M = len(B).
    """
    featsA = [generate_lexical_features(x) for x in A]
    featsB = [generate_lexical_features(x) for x in B]

    # tsr: compiled all-pairs
    A_norm = [f.norm for f in featsA]
    B_norm = [f.norm for f in featsB]
    tsr = process.cdist(A_norm, B_norm, scorer=fuzz.token_sort_ratio, score_cutoff=0, workers=workers)
    tsr = np.asarray(tsr, dtype=np.float32) / 100.0  # (N,M)

    # word_j: Jaccard on non-numeric tokens
    word_vocab = build_vocab(featsA, featsB, attr="words")
    A_words_bin = incidence_matrix(featsA, word_vocab, attr="words")
    B_words_bin = incidence_matrix(featsB, word_vocab, attr="words")
    word_j = jaccard_matrix(A_words_bin, B_words_bin)  # (N,M)

    # num_score: number_compatibility
    num_vocab = build_vocab(featsA, featsB, attr="nums")
    A_nums_bin = incidence_matrix(featsA, num_vocab, attr="nums")
    B_nums_bin = incidence_matrix(featsB, num_vocab, attr="nums")
    num_score = number_compatibility_matrix(A_nums_bin, B_nums_bin)  # (N,M)

    # Combine exactly as you specified:
    S = (0.45 * tsr + 0.10 * word_j + 0.45 * num_score).astype(np.float32)
    if no_count_penalty:
        return S
    # token-count penalty (uses feats you already built)
    tok_pen = token_count_penalty_matrix(featsA, featsB, gamma=1.0)

    # (optional) character-length penalty too; this is milder if you already do tok_pen
    # A_len = np.array([len(f.norm) for f in featsA], dtype=np.float32)[:, None]
    # B_len = np.array([len(f.norm) for f in featsB], dtype=np.float32)[None, :]
    # max_len = np.maximum(A_len, B_len)
    # rel_len = np.where(max_len > 0, np.abs(A_len - B_len) / max_len, 0.0)
    # len_pen = np.exp(-1.5 * rel_len).astype(np.float32)

    S = (S * tok_pen).astype(np.float32)
    return S


def extract_subject_column_mapping(
    metadata_artifact: ArtifactRef, sample_data_df: pd.DataFrame, return_dict: dict[str, Any]
) -> Any:
    """Map sample data indices to metadata subject IDs via lexical scoring.

    Reads the metadata CSV, computes a lexical similarity matrix
    between sample data index values and metadata Subject entries, and
    saves the best mapping as a JSON artifact.

    Args:
        metadata_artifact: Artifact reference for the metadata CSV.
        sample_data_df: Methylation DataFrame whose index values are
            sample-level identifiers.
        return_dict: Mutable result dict whose "config"."artifacts"
            will be updated with the mapping artifact.

    Returns:
        The updated return_dict.
    """
    metadata_df = pd.read_csv(metadata_artifact.path, index_col=0)
    subject_ids = metadata_df["Subject"].tolist()
    print(f"\nMetadata subject IDs: {subject_ids}")
    sample_data_subjects = sample_data_df.index.tolist()
    print(f"\nSample data subjects: {sample_data_subjects}")
    score_matrix = lexical_score_matrix(sample_data_subjects, subject_ids)
    best_j = np.argmax(score_matrix, axis=1)
    mapper = {sample_data_subjects[i]: subject_ids[j] for i, j in enumerate(best_j)}
    print(f"\nGenerated subject mapping: {mapper}")
    # Save mapper
    mapper_artifact_path = os.path.join(
        os.path.dirname(metadata_artifact.path), f"{metadata_artifact.accession_code}_subject_mapping.json"
    )
    with open(mapper_artifact_path, "w", encoding="utf-8") as f:
        json.dump(mapper, f, ensure_ascii=False, indent=2)
    mapper_artifact = ArtifactRef.model_validate(
        {
            "accession_code": metadata_artifact.accession_code,
            "path": mapper_artifact_path,
            "kind": "subject_column_mapping",
            "sha256": compute_sha256(mapper_artifact_path, is_path=True),
            "bytes": os.path.getsize(mapper_artifact_path),
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    return_dict["config"]["artifacts"] = consolidate_artifacts(return_dict["config"]["artifacts"], [mapper_artifact])

    return return_dict


def _extract_best_subject_id_fields(user_input: Any, sample_data: pd.DataFrame) -> tuple[str | None, str | None]:
    """Identify which metadata field best matches sample data indices.

    Collects candidate values from all metadata fields (including
    characteristics_ch1 sub-keys), then scores each against the sample
    data index using lexical_score_matrix to find the best-matching
    field and optional sub-key.

    Args:
        user_input: A GEO sample record whose attributes are scanned
            for candidate subject-ID values.
        sample_data: Methylation DataFrame whose index holds target
            sample identifiers.

    Returns:
        A tuple of (field_name, key_name) where field_name is the best
        metadata attribute and key_name is the characteristics_ch1
        sub-key (or None).
    """
    field_name = None
    key_name = None
    candidate_dict = {}
    best_match = -1
    for attr, value in vars(user_input).items():
        if attr in ["artifact", "relation", "platform_id"]:
            continue

        if attr not in candidate_dict:
            candidate_dict[attr] = {} if attr == "characteristics_ch1" else []

        if attr == "characteristics_ch1":
            for item in getattr(user_input, attr):
                for k, v in item.items():
                    if k not in candidate_dict[attr]:
                        candidate_dict[attr][k] = [v]
                    else:
                        candidate_dict[attr][k].append(v)
        else:
            for item in getattr(user_input, attr):
                candidate_dict[attr].extend(item)

    target_values = sample_data.index.tolist()

    print(f"\nCandidate fields and values for subject extraction: {candidate_dict}")
    print(f"\nTarget values for subject extraction: {target_values}")
    for f, f_value in candidate_dict.items():
        if not f_value:
            continue
        if f == "characteristics_ch1":
            for k, k_value in f_value.items():
                score_matrix = lexical_score_matrix(k_value, target_values)
                max_score = np.max(score_matrix)
                if max_score > best_match:
                    best_match = max_score
                    field_name = f
                    key_name = k
        else:
            score_matrix = lexical_score_matrix(f_value, target_values)
            max_score = np.max(score_matrix)
            if max_score > best_match:
                best_match = max_score
                field_name = f
                key_name = None
    print(f"\nExtracting subject for field {field_name} and key {key_name})")

    return field_name, key_name


def _create_subject_id_mapping(
    accession_code: str, user_input: pd.DataFrame, metadata_dict: dict[str, Any], sample_data: pd.DataFrame
) -> pd.DataFrame:
    """Build a mapping table linking GEO samples to best subject IDs.

    Inspects all metadata fields to find the one that lexically best
    matches the sample data index, then for each GEO sample extracts
    the corresponding subject identifier.  Also assigns the closest
    beta-format subject column value for downstream use.

    Args:
        accession_code: GEO accession identifier.
        user_input: A DataFrame of GEO sample-level records.
        metadata_dict: Full metadata dict keyed by GSM names, each
            containing field-value mappings.
        sample_data: Methylation DataFrame whose index holds target
            sample identifiers.

    Returns:
        A DataFrame with columns "Sample", "Subject", and
        "Beta_Subjects".
    """

    # This is to map from the original subject id to the best subject id
    # I have the field name and key name, I need to extract that here
    def _extract_subject(
        field_name: str, target_values: list[str], metadata_dict: dict[str, Any], key_name: str | None = None
    ):
        subject = None
        candidate_subjects = []
        max_score = -1
        for f, f_value in metadata_dict.items():
            if f == field_name:
                if field_name == "characteristics_ch1":
                    for item in f_value:
                        k, k_value = item.split(":")
                        if k.strip() == key_name:
                            candidate_subjects.append(k_value)
                else:
                    for item in f_value:
                        candidate_subjects.append(item)
        candidate_subjects = list(sorted(set(candidate_subjects)))
        score_matrix = lexical_score_matrix(candidate_subjects, target_values, no_count_penalty=True)
        best_j = np.argmax(score_matrix, axis=1)
        for i, j in enumerate(best_j):
            if score_matrix[i, j] > max_score:
                max_score = score_matrix[i, j]
                subject = candidate_subjects[i]
                sample = target_values[j]
                print(
                    f"\nExtracted subject {subject} for value {sample} in field {field_name} and key {key_name} with score {max_score}"
                )
        return subject

    def _get_subject_column_values(source_values: list[str], target_values: list[str]) -> Any:
        score_matrix = lexical_score_matrix(source_values, target_values)
        best_j = np.argmax(score_matrix, axis=1)
        ordered_target_values = [target_values[j] for j in best_j]
        return ordered_target_values

    # Get metadata columns
    accession_code = accession_code
    field_name, key_name = _extract_best_subject_id_fields(user_input, sample_data)
    target_values = sorted(sample_data.index.tolist())

    rows = []
    for gsm_name, gsm in metadata_dict["sample_metadata"].items():
        # Get sample level values
        rows.append(
            {
                "Sample": gsm_name,
                "Subject": _extract_subject(field_name, target_values, gsm, key_name=key_name),
            }
        )

    df = pd.DataFrame(rows)
    df["Beta_Subjects"] = _get_subject_column_values(df["Subject"].tolist(), target_values)
    return df
