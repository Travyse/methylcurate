__all__ = [
    "get_platform_title", "get_platform_gpls",
    "get_all_methylation_data", "get_platform_metadata",
    "extract_dataset_metadata", "generate_summary_data"]

import re
import os
import time
import requests
import GEOparse
import pandas as pd
from itertools import chain
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Any, Dict, get_args
from ...contracts.geo import Concept, GEOMetadataExtractionResult, FieldResolution, GEOSampleLevelMetadata, ExtractionRule
from ...contracts.geo import GEOSampleLevelMetadata, GeoSampleLevelMetadataBatch, FieldCoverage, MetadataSummary, GEODownloadResult
from ...contracts.common import StepStatus, ArtifactRef
from ...utils.helper import compute_sha256, consolidate_artifacts
from ...agent.state.models import GeoDatasetState, GEOIngestionConfig
# Just grab all characteristics_ch1 fields?
# Allow for users to ask: "Which field contains particular phenotype data" such as sex, specific diseases, Braak stage, etc.

def get_sample_data(
        accession_code: str, gsm_table: pd.DataFrame = None, gsm_name: str = None, value_col: str = "VALUE") -> Optional[pd.DataFrame]:
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

def cpg_union(data_rows: List[List[Any]], col_rows: List[List[str]]) -> Tuple[List[List[Any]], List[str]]:
    union_cols = list(dict.fromkeys(chain.from_iterable(col_rows)))
    aligned = []
    for rows, cols in zip(data_rows, col_rows):
        col_map = {col: rows[idx] for idx, col in enumerate(cols)}
        aligned.append([col_map.get(col, None) for col in union_cols])
    return aligned, union_cols

def apply_extraction_rule(field_values: List[str], rule: ExtractionRule) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    typ = getattr(rule, "type", "none")
    if typ == "regex":
        pattern = rule.pattern
        grp = rule.group_index
        prog = re.compile(pattern, flags=re.IGNORECASE)
        out = None
        field_name = rule.field_name
        #if grp > prog.groups:
        #    return [None for _ in field_values]
        
        if field_name == "characteristics_ch1":
            field_values = {v.split(":")[0].strip(): v.split(":")[1].strip() for v in field_values if isinstance(v, str) and ":" in v}
            for k, v in field_values.items():
                if rule.key_name.lower() == k.lower():
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

        return out, v
    # add concat or other types as needed
    return None, None

def _get_platform_title_from_gsm(gpl: str, max_retries: int = 3) -> Optional[str]:
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl}&form=text&view=brief"
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url)
            response.raise_for_status()
            for line in response.text.split('\n'):
                if line.startswith("!Platform_title"):
                    return line.split('=')[1].strip()

        except Exception as e:
            print(f"Error fetching data for {gpl}: {e}")
            time.sleep(3)
    return None

def _get_platform_gpl_from_accession_code(accession_code: str, max_retries: int = 3) -> Optional[str]:
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession_code}&form=text&view=brief"
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url)
            response.raise_for_status()
            for line in response.text.split('\n'):
                if line.startswith("!Series_platform_id"):
                    return line.split('=')[1].strip()

        except Exception as e:
            print(f"Error fetching data for {accession_code}: {e}")
            time.sleep(3)
    return None

def get_platform_gpls(gse: Any = None, accession_code: str = None) -> List[str]:
    gse_id = gse.name if gse is not None else accession_code
    platform = _get_platform_gpl_from_accession_code(gse_id)
    if platform is None:
        raise RuntimeError("Could not determine platform GPLs for GSE")
    return [platform]

def get_platform_title(gse: Any = None, accession_code: str = None) -> Optional[str]:
    platforms = get_platform_gpls(gse = gse, accession_code = accession_code)
    platform_set = set()
    if len(platforms) < 1:
        return None
    for platform in platforms:
        gpl_title = _get_platform_title_from_gsm(platform)
        platform_set.add(gpl_title if gpl_title else platform)
    return ", ".join(platform_set)

def get_platform_metadata(gse: Any = None, accession_code: str = None) -> Dict[str, Any]:
    platforms = get_platform_gpls(gse = gse, accession_code = accession_code)
    platform_metadata = {}
    for platform in platforms:
        title = _get_platform_title_from_gsm(platform)
        platform_metadata[platform] = {
            "platform_id": platform,
            "title": title if title else platform
        }
    if len(platform_metadata) < 1:
        return None
    platform = platforms[0]
    return platform_metadata[platform]

def _merge_to_dataframe(rows: List[Any], col_names: List[str], index_col: Optional[str] = None) -> pd.DataFrame:
    rows, col_names = cpg_union(rows, col_names)
    for row in rows:
        if len(row) != len(col_names):
            raise ValueError(f"Row length {len(row)} does not match column names length {len(col_names)}")
    df = pd.DataFrame(rows, columns=col_names)
    if index_col is not None and index_col in df.columns:
        df.set_index(index_col, inplace=True)
    return df

def get_field_value(gsm_metadata: dict, resolution: FieldResolution) -> Optional[str]:
    if resolution.status in {"missing", "error"}:
        return None, None, False
    if resolution.extraction.field_name == "default":
        return resolution.extraction.value, "default", False
    field_values = gsm_metadata.get(resolution.extraction.field_name, None)
    if field_values is None:
        return None, None, False
    if isinstance(field_values, str):
        field_values = [field_values]
    #if resolution.extraction is None:
    #    return None, None
    extracted_value, target_field = apply_extraction_rule(field_values, resolution.extraction)
    return extracted_value, target_field, False if (extracted_value is not None) or all(x is None for x in [extracted_value, target_field]) else (extracted_value, target_field, True)

def get_field_Coverage(sample_batch: GeoSampleLevelMetadataBatch, field: str) -> FieldCoverage:
    return FieldCoverage(
        present=sum(1 for s in sample_batch.samples if getattr(s, field) is not None),
        missing=sum(1 for s in sample_batch.samples if getattr(s, field) is None),
        parse_rate=sum(1 for s in sample_batch.samples if getattr(s, field) is not None) / len(sample_batch.samples) if len(sample_batch.samples) > 0 else 0.0,
        unique_values=len(set(getattr(s, field) for s in sample_batch.samples if getattr(s, field) is not None)),
        examples=[str(getattr(s, field)) for s in sample_batch.samples if getattr(s, field) is not None][:10]
    )

def get_df_field_Coverage(metadata: pd.DataFrame, field: str) -> FieldCoverage:
    return FieldCoverage(
        present=metadata[metadata[field].notna()][field].count(),
        missing=metadata[metadata[field].isna()][field].count(),
        parse_rate=metadata[metadata[field].notna()][field].count() / metadata.shape[0] if metadata.shape[0] > 0 else 0.0,
        unique_values=metadata[metadata[field].notna()][field].nunique(),
        examples=metadata[metadata[field].notna()][field].astype(str).tolist()[:10]
    )

def extract_dataset_metadata(
        accession: str,
        state_config: GEOIngestionConfig,
        metadata_dict: Any,

        metadata_extraction_result: GEOMetadataExtractionResult,
        overwrite_artifact: bool,
        gpls: Optional[List[str]] = None,
        platform: Optional[List[str]] = None,
        return_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    def get_resolution(metadata_extraction_result: GEOMetadataExtractionResult, concept: Concept) -> Optional[FieldResolution]:
        if hasattr(metadata_extraction_result, "resolutions"):
            return metadata_extraction_result.resolutions.get(concept, None)
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
    failed_parsing_info = {
        c: [] for c in get_args(Concept)
    }
    disease_statuses_raw = set()
    
    for gsm_name, gsm in metadata_dict["sample_metadata"].items():
        # Get sample level values
        disease_status, disease_status_info, disease_is_failure = get_field_value(gsm, disease_status_resolution)
        condition, condition_status_info, condition_is_failure = get_field_value(gsm, condition_resolution)
        age, age_info, age_is_failure = get_field_value(gsm, age_resolution)
        sex, sex_info, sex_is_failure = get_field_value(gsm, sex_resolution)
        tissue, tissue_info, tissue_is_failure = get_field_value(gsm, tissue_resolution)
        cell_type, cell_type_info, cell_type_is_failure = get_field_value(gsm, cell_type_resolution)
        subject, subject_info, subject_is_failure = get_field_value(gsm, subject_id_resolution)

        # Track failed parsing for each concept:
        if disease_is_failure:
            failed_parsing_info["disease_status"] += [disease_status_info]
        if condition_is_failure:
            failed_parsing_info["condition"] += [condition_status_info]
        if age_is_failure:
            if any(char.isdigit() for char in age_info):
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
            age_f = float(age)
        except (TypeError, ValueError):
            print(f"\n\nCould not parse age value '{age_info}' for sample {gsm_name} in {accession_code}. Metadata is {gsm.get('characteristics_ch1', {})} and resolution is {age_resolution.model_dump()}")
            age_f = None

        # Output files
        metadata_dataframe_output_path = os.path.join(
            state_config.output_root, accession_code, "sample_metadata.csv")
        
        metadata_rows.append([accession_code, gsm_name, disease_status, condition, tissue, cell_type, subject, gsm_name, age_f, sex, platform])
        metadata_col_names.append(['Accession_Code', 'GSM', 'Disease_Status', 'Condition', 'Tissue', 'Cell_Type', 'Subject', 'Sample', 'age', "Sex", "Platform"])

    if len(metadata_rows) > 0:
        metadata_df = _merge_to_dataframe(metadata_rows, metadata_col_names, index_col="GSM")
        if not os.path.exists(metadata_dataframe_output_path) or overwrite_artifact:
            metadata_df.to_csv(metadata_dataframe_output_path, index=True)
            metadata_artifact = ArtifactRef.model_validate({
                "accession_code": accession_code,
                "path": metadata_dataframe_output_path,
                "kind": "dataset_metadata",
                "sha256": compute_sha256(metadata_dataframe_output_path, is_path=True),
                "bytes": os.path.getsize(metadata_dataframe_output_path),
                "created_at": datetime.now(timezone.utc).isoformat()})
        else:
            if next((a for a in state_config.artifacts if a.path == metadata_dataframe_output_path), None) is None:
                metadata_artifact = ArtifactRef.model_validate({
                    "accession_code": accession_code,
                    "path": metadata_dataframe_output_path,
                    "kind": "dataset_metadata",
                    "sha256": compute_sha256(metadata_dataframe_output_path, is_path=True),
                    "bytes": os.path.getsize(metadata_dataframe_output_path),
                    "created_at": datetime.now(timezone.utc).isoformat()})
        return_dict["config"]["artifacts"] = consolidate_artifacts(
            [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]],
            [metadata_artifact])
    
    failed_parsing_info = {k: list(sorted(set(v))) for k, v in failed_parsing_info.items()}
    return_dict["datasets"][accession_code]["refinement_history"]["example_errors"].append(failed_parsing_info)
    return_dict["raw_disease_statuses"] = sorted(list(disease_statuses_raw))
    return_dict["config"]["artifacts"] = [a.model_dump() for a in return_dict["config"]["artifacts"]]
    return return_dict

def generate_summary_data(
        metadata: pd.DataFrame, accession_code: str, platforms: List[str], gpls: List[str], failed_parsing_info: Dict[str, List[Any]],
        return_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Evaluation
    if metadata.empty:
        sample_summary = MetadataSummary(
            accession=accession_code,
            n_samples=metadata.shape[0],
            subject_id=0.0,
            age=0.0,
            sex=0.0,
            tissue=0.0,
            condition=0.0,
            cell_type=0.0,
            disease_status=0.0,
            platform=[],
            gpl=[])
    
    else:
        subject_id_coverage = get_df_field_Coverage(metadata, "Subject")
        age_coverage = get_df_field_Coverage(metadata, "age")
        sex_coverage = get_df_field_Coverage(metadata, "Sex")
        tissue_coverage = get_df_field_Coverage(metadata, "Tissue")
        cell_type_coverage = get_df_field_Coverage(metadata, "Cell_Type")
        condition_coverage = get_df_field_Coverage(metadata, "Condition")
        disease_status_coverage = get_df_field_Coverage(metadata, "Disease_Status")

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
            gpl=gpls
        )
    return_dict["datasets"][accession_code]["metadata_summary"] = sample_summary.model_dump()
    return return_dict

def get_all_methylation_data(state_config: GEOIngestionConfig, state: GeoDatasetState) -> Dict[str, Any]:
    methylation_rows = []
    methylation_col_names = []
    accession_code = state.accession
    return_dict = {
        "artifacts": []
    } 

    gse = GEOparse.get_GEO(filepath=state.download_result.artifact.path, silent=True)
    methylation_dataframe_output_path = os.path.join(
        state_config.output_root, accession_code, "preqc_methylation_matrix.csv")
    
    for gsm_name, gsm in gse.gsms.items():

        # Remove poor quality CpG sites
        sample_data = get_sample_data(
            accession_code, gsm_table=gsm.table, gsm_name=gsm_name)

        if sample_data is None or len(sample_data.columns.tolist()) < 1:
            print(f"{gsm_name} from {accession_code} has no data or is missing")
            continue

        methylation_rows.append([gsm_name] + sample_data["VALUE"].tolist())
        methylation_col_names.append(['Sample'] + sample_data["ID_REF"].tolist())
        
    methylation_df = _merge_to_dataframe(methylation_rows, methylation_col_names, index_col="Sample")   
    if not os.path.exists(methylation_dataframe_output_path):
        methylation_df.to_csv(methylation_dataframe_output_path, index=True)
        methylation_artifact = ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": methylation_dataframe_output_path,
            "kind": "preqc_methylation_data",
            "sha256": compute_sha256(methylation_dataframe_output_path, is_path=True),
            "bytes": os.path.getsize(methylation_dataframe_output_path),
            "created_at": datetime.now(timezone.utc).isoformat()})
        return_dict["artifacts"] += [methylation_artifact]
    else:
        if next((a for a in state_config.artifacts if a.path == methylation_dataframe_output_path), None) is None:
            methylation_artifact = ArtifactRef.model_validate({
                "accession_code": accession_code,
                "path": methylation_dataframe_output_path,
                "kind": "preqc_methylation_data",
                "sha256": compute_sha256(methylation_dataframe_output_path, is_path=True),
                "bytes": os.path.getsize(methylation_dataframe_output_path),
                "created_at": datetime.now(timezone.utc).isoformat()})
            return_dict["artifacts"] += [methylation_artifact]
    return return_dict
