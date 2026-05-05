__all__ = [
    "get_platform_title",
    "get_platform_gpls",
    "get_all_methylation_data",
    "get_platform_metadata",
    "extract_dataset_metadata",
    "generate_summary_data",
]

import os
import re
import time
from datetime import UTC, datetime
from itertools import chain
from typing import Any, get_args

import GEOparse
import pandas as pd
import requests

from ...agent.state.models import GeoDatasetState, GEOIngestionConfig
from ...contracts.common import ArtifactRef
from ...contracts.geo import (
    Concept,
    ExtractionRule,
    FieldCoverage,
    FieldResolution,
    GEOMetadataExtractionResult,
    GeoSampleLevelMetadataBatch,
    MetadataSummary,
)
from ...utils.helper import compute_sha256, consolidate_artifacts

# Just grab all characteristics_ch1 fields?
# Allow for users to ask: "Which field contains particular phenotype data" such as sex, specific diseases, Braak stage, etc.


def get_sample_data(
    accession_code: str,
    gsm_table: pd.DataFrame | None = None,
    gsm_name: str | None = None,
    value_col: str = "VALUE",
) -> pd.DataFrame | None:
    """Filter sample data by detection P-value if available.

    If the sample table contains columns with "detection" in their name, rows
    with a detection P-value greater than 0.05 are dropped.

    Args:
        accession_code: GEO accession identifier (e.g. "GSE12345").
        gsm_table: DataFrame of methylation data for a single GSM sample.
        gsm_name: Name of the GSM sample (used for logging).
        value_col: Name of the column containing methylation beta values.

    Returns:
        The filtered DataFrame, or None if the table is missing or empty.
    """
    sample_data = gsm_table
    if sample_data is None:
        print(f"{gsm_name} from {accession_code} is missing sample data")
        return None
    if len(sample_data.columns.tolist()) < 1:
        print(f"{gsm_name} from {accession_code} has no data columns")
        return None
    detection_cols = [x for x in sample_data.columns if "detection" in x.lower()]
    if len(detection_cols) > 0:
        sample_data = sample_data[sample_data[detection_cols[0]] <= 0.05]
    return sample_data


def cpg_union(data_rows: list[list[Any]], col_rows: list[list[str]]) -> tuple[list[list[Any]], list[str]]:
    """Align heterogeneous row-column pairs to a unified superset of columns.

    Each entry in ``data_rows`` and ``col_rows`` is aligned so that all output
    rows share the same column ordering (the union of all column names). Missing
    values are filled with None.

    Args:
        data_rows: List of value lists, one per sample.
        col_rows: List of column-name lists corresponding to each value list.

    Returns:
        A tuple of ``(aligned_rows, union_columns)`` where ``aligned_rows``
        share the unified column set and ``union_columns`` preserves insertion order.
    """
    union_cols = list(dict.fromkeys(chain.from_iterable(col_rows)))
    aligned = []
    for rows, cols in zip(data_rows, col_rows, strict=True):
        col_map = {col: rows[idx] for idx, col in enumerate(cols)}
        aligned.append([col_map.get(col, None) for col in union_cols])
    return aligned, union_cols


def apply_extraction_rule(field_values: list[str], rule: ExtractionRule) -> tuple[str | None, dict[str, Any] | None]:
    """Apply an extraction rule to a list of raw field values.

    Currently supports ``"regex"`` extraction rules. When the target field is
    ``characteristics_ch1``, values are parsed as key-value pairs and matched
    against the rule's ``key_name`` before applying the regex.

    Args:
        field_values: Raw metadata values from a GSM sample.
        rule: Extraction rule specifying type, pattern, group_index, field_name,
            and optionally key_name.

    Returns:
        A tuple of ``(extracted_value, raw_source)`` where ``extracted_value``
        is the matched regex group (or None) and ``raw_source`` is the original
        value that was matched (or None).
    """
    typ = getattr(rule, "type", "none")
    if typ == "regex":
        pattern = rule.pattern  # type: ignore
        grp = rule.group_index  # type: ignore
        prog = re.compile(pattern, flags=re.IGNORECASE)
        out = None
        field_name = rule.field_name
        # if grp > prog.groups:
        #    return [None for _ in field_values]

        if field_name == "characteristics_ch1":
            field_values = {  # type: ignore
                v.split(":")[0].strip(): v.split(":")[1].strip()
                for v in field_values
                if isinstance(v, str) and ":" in v
            }
            for k, v in field_values.items():  # type: ignore
                if rule.key_name.lower() == k.lower():  # type: ignore
                    m = prog.search(v)
                    out = m.group(grp) if m else None
                    return out, v
            return out, None
        else:
            for v in field_values:
                if v is None:
                    continue
                m = prog.search(v)
                if m:
                    out = m.group(grp)

        return out, v  # type: ignore
    # add concat or other types as needed
    return None, None


def _get_platform_title_from_gsm(gpl: str, max_retries: int = 3) -> str | None:
    """Fetch the platform title for a GPL accession from NCBI GEO.

    Args:
        gpl: Platform accession identifier (e.g. "GPL13534").
        max_retries: Maximum number of HTTP request attempts with 3-second delays.

    Returns:
        The platform title string, or None if the request fails after all retries.

    Raises:
        requests.exceptions.RequestException: Propagated on the final attempt
            after all retries are exhausted.
    """
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl}&form=text&view=brief"
    for _attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url)
            response.raise_for_status()
            for line in response.text.split("\n"):
                if line.startswith("!Platform_title"):
                    return line.split("=")[1].strip()

        except Exception as e:
            print(f"Error fetching data for {gpl}: {e}")
            time.sleep(3)
    return None


def _get_platform_gpl_from_accession_code(accession_code: str, max_retries: int = 3) -> str | None:
    """Extract the platform GPL identifier from a GEO series accession page.

    Args:
        accession_code: GEO series accession (e.g. "GSE12345").
        max_retries: Maximum number of HTTP request attempts with 3-second delays.

    Returns:
        The platform GPL identifier string, or None if it cannot be determined.

    Raises:
        requests.exceptions.RequestException: Propagated on the final attempt
            after all retries are exhausted.
    """
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession_code}&form=text&view=brief"
    for _attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url)
            response.raise_for_status()
            for line in response.text.split("\n"):
                if line.startswith("!Series_platform_id"):
                    return line.split("=")[1].strip()

        except Exception as e:
            print(f"Error fetching data for {accession_code}: {e}")
            time.sleep(3)
    return None


def get_platform_gpls(gse: Any = None, accession_code: str = None) -> list[str]:  # type: ignore
    """Return the platform GPL identifiers for a GEO dataset.

    Args:
        gse: A GEOparse GSE object whose ``.name`` yields the accession.
        accession_code: Fallback accession string used when ``gse`` is None.

    Returns:
        A single-element list containing the platform GPL identifier.

    Raises:
        RuntimeError: If the platform GPL could not be resolved via NCBI.
    """
    gse_id = gse.name if gse is not None else accession_code
    platform = _get_platform_gpl_from_accession_code(gse_id)
    if platform is None:
        raise RuntimeError("Could not determine platform GPLs for GSE")
    return [platform]


def get_platform_title(gse: Any = None, accession_code: str = None) -> str | None:  # type: ignore
    """Return the human-readable platform title(s) for a GEO dataset.

    Resolves GPL identifiers via ``get_platform_gpls`` and fetches each
    platform title from NCBI.

    Args:
        gse: A GEOparse GSE object.
        accession_code: Fallback accession string when ``gse`` is None.

    Returns:
        A comma-separated string of platform titles, or None if no platforms
        could be resolved.
    """
    platforms = get_platform_gpls(gse=gse, accession_code=accession_code)
    platform_set = set()
    if len(platforms) < 1:
        return None
    for platform in platforms:
        gpl_title = _get_platform_title_from_gsm(platform)
        platform_set.add(gpl_title if gpl_title else platform)
    return ", ".join(platform_set)


def get_platform_metadata(gse: Any = None, accession_code: str = None) -> dict[str, Any]:  # type: ignore
    """Return platform metadata for a GEO dataset.

    Fetches the first platform's GPL identifier and title as a dictionary.

    Args:
        gse: A GEOparse GSE object.
        accession_code: Fallback accession string when ``gse`` is None.

    Returns:
        A dict with keys ``"platform_id"`` and ``"title"``, or None if no
        platforms could be resolved.
    """
    platforms = get_platform_gpls(gse=gse, accession_code=accession_code)
    platform_metadata = {}
    for platform in platforms:
        title = _get_platform_title_from_gsm(platform)
        platform_metadata[platform] = {"platform_id": platform, "title": title if title else platform}
    if len(platform_metadata) < 1:
        return None  # type: ignore
    platform = platforms[0]
    return platform_metadata[platform]


def _merge_to_dataframe(rows: list[Any], col_names: list[str], index_col: str | None = None) -> pd.DataFrame:
    """Merge heterogeneous rows and column-name lists into a unified DataFrame.

    Delegates to ``cpg_union`` to align columns and then constructs a
    pandas DataFrame.

    Args:
        rows: List of row-lists with heterogeneous lengths and column sets.
        col_names: List of column-name lists corresponding to each row.
        index_col: If provided and present in the unified columns, this column
            is set as the DataFrame index.

    Returns:
        A DataFrame with unified columns, optionally indexed by ``index_col``.

    Raises:
        ValueError: If any aligned row length does not match the unified
            column count.
    """
    rows, col_names = cpg_union(rows, col_names)  # type: ignore
    for row in rows:
        if len(row) != len(col_names):
            raise ValueError(f"Row length {len(row)} does not match column names length {len(col_names)}")
    df = pd.DataFrame(rows, columns=col_names)
    if index_col is not None and index_col in df.columns:
        df.set_index(index_col, inplace=True)
    return df


def get_field_value(gsm_metadata: dict, resolution: FieldResolution) -> tuple[str | None, str | None, bool]:
    """Extract a metadata field value from GSM metadata using a field resolution.

    If the resolution status is ``"missing"`` or ``"error"``, a failure tuple
    is returned. When the extraction field_name is ``"default"``, the raw
    resolution value is used directly. Otherwise the extraction rule is applied
    to the raw field values.

    Args:
        gsm_metadata: Dictionary of metadata fields for a single GSM sample.
        resolution: Field resolution object carrying the status, field name, and
            extraction rule.

    Returns:
        A tuple of ``(extracted_value, raw_source_value, is_failure)``.
        ``is_failure`` is True when extraction succeeded on the rule level but
        yielded None for both outputs; it may also be a non-bool truthy value
        in some edge cases.
    """
    if resolution.status in {"missing", "error"}:
        return None, None, False
    if resolution.extraction.field_name == "default":  # type: ignore
        return resolution.extraction.value, "default", False  # type: ignore
    field_values = gsm_metadata.get(resolution.extraction.field_name, None)  # type: ignore
    if field_values is None:
        return None, None, False
    if isinstance(field_values, str):
        field_values = [field_values]
    # if resolution.extraction is None:
    #    return None, None
    extracted_value, target_field = apply_extraction_rule(field_values, resolution.extraction)  # type: ignore
    return (  # type: ignore
        extracted_value,
        target_field,
        False
        if (extracted_value is not None) or all(x is None for x in [extracted_value, target_field])
        else (extracted_value, target_field, True),
    )


def get_field_coverage(sample_batch: GeoSampleLevelMetadataBatch, field: str) -> FieldCoverage:
    """Compute coverage statistics for a metadata field across a sample batch.

    Args:
        sample_batch: Batch of sample-level metadata objects.
        field: Name of the attribute to inspect on each sample.

    Returns:
        A ``FieldCoverage`` instance with present/missing counts, parse rate,
        unique value count, and up to 10 example values.
    """
    return FieldCoverage(
        present=sum(1 for s in sample_batch.samples if getattr(s, field) is not None),
        missing=sum(1 for s in sample_batch.samples if getattr(s, field) is None),
        parse_rate=sum(1 for s in sample_batch.samples if getattr(s, field) is not None) / len(sample_batch.samples)
        if len(sample_batch.samples) > 0
        else 0.0,
        unique_values=len(set(getattr(s, field) for s in sample_batch.samples if getattr(s, field) is not None)),
        examples=[str(getattr(s, field)) for s in sample_batch.samples if getattr(s, field) is not None][:10],
    )


def get_df_field_coverage(metadata: pd.DataFrame, field: str) -> FieldCoverage:
    """Compute coverage statistics for a metadata column in a DataFrame.

    Args:
        metadata: DataFrame containing sample metadata.
        field: Column name to inspect.

    Returns:
        A ``FieldCoverage`` instance with present/missing counts, parse rate,
        unique value count, and up to 10 example values.
    """
    return FieldCoverage(
        present=metadata[metadata[field].notna()][field].count(),
        missing=metadata[field].isna().sum(),
        parse_rate=metadata[metadata[field].notna()][field].count() / metadata.shape[0]
        if metadata.shape[0] > 0
        else 0.0,
        unique_values=metadata[metadata[field].notna()][field].nunique(),
        examples=metadata[metadata[field].notna()][field].astype(str).tolist()[:10],
    )


def extract_dataset_metadata(
    accession: str,
    state_config: GEOIngestionConfig,
    metadata_dict: Any,
    metadata_extraction_result: GEOMetadataExtractionResult,
    overwrite_artifact: bool,
    gpls: list[str] | None = None,
    platform: list[str] | None = None,
    return_dict: dict[str, Any] = None,  # type: ignore
) -> dict[str, Any]:
    """Extract sample-level metadata for a GEO dataset and write artifacts.

    Iterates over all GSM samples and applies field resolutions for disease
    status, condition, age, sex, tissue, cell type, and subject ID. Produces
    a sample metadata CSV artifact and populates ``return_dict`` with artifact
    references, raw disease statuses, and any per-concept parsing failures.

    Args:
        accession: GEO accession identifier.
        state_config: Configuration specifying output root and existing artifacts.
        metadata_dict: Dictionary containing ``"sample_metadata"`` keyed by GSM name.
        metadata_extraction_result: Resolutions mapping each ``Concept`` to its
            ``FieldResolution``.
        overwrite_artifact: If True, overwrite an existing metadata CSV on disk.
        gpls: Optional list of platform GPL identifiers.
        platform: Optional list of platform names.
        return_dict: Mutable dictionary to populate with extraction results.
            Must contain ``"config"`` with ``"artifacts"``, and ``"datasets"``
            with the accession-keyed entry.

    Returns:
        The same ``return_dict`` dictionary, updated with artifact information,
        raw disease statuses, and example parsing errors.
    """

    def get_resolution(
        metadata_extraction_result: GEOMetadataExtractionResult, concept: Concept
    ) -> FieldResolution | None:
        if hasattr(metadata_extraction_result, "resolutions"):
            return metadata_extraction_result.resolutions.get(concept, None)  # type: ignore
        else:
            return getattr(metadata_extraction_result, concept, None)

    metadata_rows = []
    metadata_col_names = []

    # Get metadata columns
    accession_code = accession
    disease_status_resolution = get_resolution(metadata_extraction_result, "disease_status")
    condition_resolution = get_resolution(metadata_extraction_result, "condition")
    age_resolution = get_resolution(metadata_extraction_result, "age")
    sex_resolution = get_resolution(metadata_extraction_result, "sex")
    tissue_resolution = get_resolution(metadata_extraction_result, "tissue")
    cell_type_resolution = get_resolution(metadata_extraction_result, "cell_type")
    subject_id_resolution = get_resolution(metadata_extraction_result, "subject_id")

    # Gather Information
    failed_parsing_info = {c: [] for c in get_args(Concept)}
    disease_statuses_raw = set()

    for gsm_name, gsm in metadata_dict["sample_metadata"].items():
        # Get sample level values
        disease_status, disease_status_info, disease_is_failure = get_field_value(gsm, disease_status_resolution)  # type: ignore
        condition, condition_status_info, condition_is_failure = get_field_value(gsm, condition_resolution)  # type: ignore
        age, age_info, age_is_failure = get_field_value(gsm, age_resolution)  # type: ignore
        sex, sex_info, sex_is_failure = get_field_value(gsm, sex_resolution)  # type: ignore
        tissue, tissue_info, tissue_is_failure = get_field_value(gsm, tissue_resolution)  # type: ignore
        cell_type, cell_type_info, cell_type_is_failure = get_field_value(gsm, cell_type_resolution)  # type: ignore
        subject, subject_info, subject_is_failure = get_field_value(gsm, subject_id_resolution)  # type: ignore

        # Track failed parsing for each concept:
        if disease_is_failure:
            failed_parsing_info["disease_status"] += [disease_status_info]
        if condition_is_failure:
            failed_parsing_info["condition"] += [condition_status_info]
        if age_is_failure:
            if any(char.isdigit() for char in age_info):  # type: ignore
                failed_parsing_info["age"] += [age_info]
        if sex_is_failure:
            failed_parsing_info["sex"] += [sex_info]
        if tissue_is_failure:
            failed_parsing_info["tissue"] += [tissue_info]
        if cell_type_is_failure:
            failed_parsing_info["cell_type"] += [cell_type_info]
        if subject_is_failure:
            failed_parsing_info["subject_id"] += [subject_info]
        if disease_status is not None:
            disease_statuses_raw.add(disease_status)

        try:
            age_f = float(age)  # type: ignore
        except (TypeError, ValueError):
            print(
                f"\n\nCould not parse age value '{age_info}' for sample {gsm_name} in {accession_code}. Metadata is {gsm.get('characteristics_ch1', {})} and resolution is {age_resolution.model_dump()}"  # type: ignore
            )
            age_f = None

        # Output files
        metadata_dataframe_output_path = os.path.join(state_config.output_root, accession_code, "sample_metadata.csv")

        metadata_rows.append(
            [
                accession_code,
                gsm_name,
                disease_status,
                condition,
                tissue,
                cell_type,
                subject,
                gsm_name,
                age_f,
                sex,
                platform,
            ]
        )
        metadata_col_names.append(
            [
                "Accession_Code",
                "GSM",
                "Disease_Status",
                "Condition",
                "Tissue",
                "Cell_Type",
                "Subject",
                "Sample",
                "age",
                "Sex",
                "Platform",
            ]
        )

    if len(metadata_rows) > 0:
        metadata_df = _merge_to_dataframe(metadata_rows, metadata_col_names, index_col="GSM")
        if not os.path.exists(metadata_dataframe_output_path) or overwrite_artifact:
            metadata_df.to_csv(metadata_dataframe_output_path, index=True)
            metadata_artifact = ArtifactRef.model_validate(
                {
                    "accession_code": accession_code,
                    "path": metadata_dataframe_output_path,
                    "kind": "dataset_metadata",
                    "sha256": compute_sha256(metadata_dataframe_output_path, is_path=True),
                    "bytes": os.path.getsize(metadata_dataframe_output_path),
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
        else:
            if next((a for a in state_config.artifacts if a.path == metadata_dataframe_output_path), None) is None:
                metadata_artifact = ArtifactRef.model_validate(
                    {
                        "accession_code": accession_code,
                        "path": metadata_dataframe_output_path,
                        "kind": "dataset_metadata",
                        "sha256": compute_sha256(metadata_dataframe_output_path, is_path=True),
                        "bytes": os.path.getsize(metadata_dataframe_output_path),
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                )
        return_dict["config"]["artifacts"] = consolidate_artifacts(
            [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]], [metadata_artifact]
        )

    failed_parsing_info = {k: list(sorted(set(v))) for k, v in failed_parsing_info.items()}
    return_dict["datasets"][accession_code]["refinement_history"]["example_errors"].append(failed_parsing_info)
    return_dict["raw_disease_statuses"] = sorted(list(disease_statuses_raw))
    return_dict["config"]["artifacts"] = [a.model_dump() for a in return_dict["config"]["artifacts"]]
    return return_dict


def generate_summary_data(
    metadata: pd.DataFrame,
    accession_code: str,
    platforms: list[str],
    gpls: list[str],
    failed_parsing_info: dict[str, list[Any]],
    return_dict: dict[str, Any],
) -> dict[str, Any]:
    """Generate a metadata coverage summary for a dataset.

    Computes per-field coverage statistics (subject ID, age, sex, tissue,
    cell type, condition, disease status) and attaches a ``MetadataSummary``
    to ``return_dict`` under the dataset's accession key.

    Args:
        metadata: DataFrame of sample-level metadata.
        accession_code: GEO accession identifier.
        platforms: List of platform names.
        gpls: List of platform GPL identifiers.
        failed_parsing_info: Dictionary of parsing failures keyed by concept.
        return_dict: Mutable dictionary to update with the summary. Must contain
            a ``"datasets"`` entry keyed by ``accession_code``.

    Returns:
        The same ``return_dict`` dictionary with the summary attached.
    """
    # Evaluation
    if metadata.empty:
        sample_summary = MetadataSummary(
            accession=accession_code,
            n_samples=metadata.shape[0],
            subject_id=0.0,  # type: ignore
            age=0.0,  # type: ignore
            sex=0.0,  # type: ignore
            tissue=0.0,  # type: ignore
            condition=0.0,  # type: ignore
            cell_type=0.0,  # type: ignore
            disease_status=0.0,  # type: ignore
            platform=[],
            gpl=[],
        )

    else:
        subject_id_coverage = get_df_field_coverage(metadata, "Subject")
        age_coverage = get_df_field_coverage(metadata, "age")
        sex_coverage = get_df_field_coverage(metadata, "Sex")
        tissue_coverage = get_df_field_coverage(metadata, "Tissue")
        cell_type_coverage = get_df_field_coverage(metadata, "Cell_Type")
        condition_coverage = get_df_field_coverage(metadata, "Condition")
        disease_status_coverage = get_df_field_coverage(metadata, "Disease_Status")

        print(f"\n GPLs: {gpls}")
        print(f"\n Platforms: {platforms}")
        sample_summary = MetadataSummary(
            accession=accession_code,
            n_samples=metadata.shape[0],
            subject_id=subject_id_coverage,
            age=age_coverage,
            sex=sex_coverage,
            tissue=tissue_coverage,
            cell_type=cell_type_coverage,
            condition=condition_coverage,
            disease_status=disease_status_coverage,
            platform=platforms,
            gpl=gpls,
        )
    return_dict["datasets"][accession_code]["metadata_summary"] = sample_summary.model_dump()
    return return_dict


def get_all_methylation_data(state_config: GEOIngestionConfig, state: GeoDatasetState) -> dict[str, Any]:
    """Extract all methylation beta values from a GEO dataset.

    Loads the GEO SOFT file from the download artifact, iterates over all GSM
    samples, filters low-quality CpG sites via ``get_sample_data``, and merges
    heterogeneous probe sets into a unified pre-QC methylation matrix CSV.

    Args:
        state_config: Configuration with output root and existing artifacts.
        state: Workflow state carrying the GEO accession and download result.

    Returns:
        A dict with an ``"artifacts"`` list containing an ``ArtifactRef`` for
        the generated pre-QC methylation matrix CSV.
    """
    methylation_rows = []
    methylation_col_names = []
    accession_code = state.accession
    return_dict = {"artifacts": []}

    gse = GEOparse.get_GEO(filepath=state.download_result.artifact.path, silent=True)  # type: ignore
    methylation_dataframe_output_path = os.path.join(
        state_config.output_root, accession_code, "preqc_methylation_matrix.csv"
    )

    for gsm_name, gsm in gse.gsms.items():
        # Remove poor quality CpG sites
        sample_data = get_sample_data(accession_code, gsm_table=gsm.table, gsm_name=gsm_name)

        if sample_data is None or len(sample_data.columns.tolist()) < 1:
            print(f"{gsm_name} from {accession_code} has no data or is missing")
            continue

        methylation_rows.append([gsm_name] + sample_data["VALUE"].tolist())
        methylation_col_names.append(["Sample"] + sample_data["ID_REF"].tolist())

    methylation_df = _merge_to_dataframe(methylation_rows, methylation_col_names, index_col="Sample")
    if not os.path.exists(methylation_dataframe_output_path):
        methylation_df.to_csv(methylation_dataframe_output_path, index=True)
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
        return_dict["artifacts"] += [methylation_artifact]
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
            return_dict["artifacts"] += [methylation_artifact]
    return return_dict
