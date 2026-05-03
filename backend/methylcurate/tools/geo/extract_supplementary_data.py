__all__ = [
    "format_methylation_data", "format_individual_methylation_data",
    "merge_formatted_supplementary_data", "_create_subject_id_mapping"]
import os
import re
import json
import gzip
import uuid
import asyncio
import pandas as pd
import numpy as np
import random 

from rapidfuzz import fuzz, process
from datetime import datetime, timezone
from typing import List, Any, Tuple, Optional, Dict
from itertools import chain
from ollama._types import ResponseError
from pydantic import ValidationError
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from ...agent.state.models import GEOIngestionConfig, GeoDatasetState, GeoIngestionSubgraphState
from ...utils.helper import compute_sha256, consolidate_artifacts, write_feather
from ...utils.examples import generate_column_interpretation_examples, generate_column_interpretation_examples_no_detection
from ...utils.prompting import generate_infer_methylation_data_column_scheme_prompt, generate_subject_column_refinement_prompt, generate_infer_methylation_data_column_scheme_alt_prompt
from ...contracts.common import ArtifactRef
from ...contracts.geo import (
    SampleDataResolution, LexFeat, ErrorResolution, ForcedSampleDataResolution, GEOMetadataExtractionResult, ResolvedResolution)
from .extract_sample_level_metadata import get_field_value
from langchain_core.runnables import RunnableConfig
from ...agent.graphs.deps import Deps

CALL_TIMEOUT = 180
GLOBAL_RETRY_LIMIT = 5

def cpg_union(data_rows: List[List[Any]], col_rows: List[List[str]]) -> Tuple[List[List[Any]], List[str]]:
    union_cols = list(dict.fromkeys(chain.from_iterable(col_rows)))
    aligned = []
    for rows, cols in zip(data_rows, col_rows):
        col_map = {col: rows[idx] for idx, col in enumerate(cols)}
        aligned.append([col_map.get(col, None) for col in union_cols])
    return aligned, union_cols

def _merge_to_dataframe(rows: List[Any], col_names: List[str], index_col: Optional[str] = None) -> pd.DataFrame:
    rows, col_names = cpg_union(rows, col_names)
    for row in rows:
        if len(row) != len(col_names):
            raise ValueError(f"Row length {len(row)} does not match column names length {len(col_names)}")
    df = pd.DataFrame(rows, columns=col_names)
    if index_col is not None and index_col in df.columns:
        df.set_index(index_col, inplace=True)
    return df

def _check_for_cpg_probes(sample_data: pd.DataFrame) -> bool:
    cols = np.asarray(sample_data.columns, dtype=str)
    rows = np.asarray(sample_data.index, dtype=str)

    n_cg_cols = (np.char.find(np.char.lower(cols), "cg") >= 0).sum()
    n_cg_rows = (np.char.find(np.char.lower(rows), "cg") >= 0).sum()

    return {
        "columns": n_cg_cols > n_cg_rows,
        "rows": n_cg_rows > n_cg_cols,
    }

def _check_for_detection_columns(sample_data: pd.DataFrame) -> bool:
    columns = sample_data.columns.tolist()
    #detection_cols = [col for col in columns if "detect" in col.lower()] # May not have detect in the name
    columns_lt_rows = len(columns) < len(sample_data.index) # There will almost certainly be more probes than samples
    #columns_even = len(columns) % 2 == 0 # Thoughts, doesn't necessarily have to be so
    return columns_lt_rows

async def _process_detection_columns(artifact: ArtifactRef, sample_data: pd.DataFrame, config: RunnableConfig, messages: List[AnyMessage]) -> pd.DataFrame:
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
    sample_data = sample_data.apply(pd.to_numeric, errors='coerce')

    # 2. Extract Patterns
    beta_pat = re.compile(column_scheme.beta_column.pattern, re.IGNORECASE) if column_scheme.beta_column.status == "resolved" else None
    det_pat = re.compile(column_scheme.detection_column.pattern, re.IGNORECASE) if column_scheme.detection_column.status == "resolved" else None

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
    mapper = {
        beta_cols[i]: det_cols[j] for i, j in enumerate(best_j)
    }
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
    methylation_df = pd.DataFrame(
        filtered_values.T, 
        index=beta_cols, 
        columns=sample_data.index
    )
    end_time = datetime.now()
    reconstruction_time_taken = (end_time - start_time).total_seconds()
    print(f"\nTime taken to reconstruct filtered DataFrame: {reconstruction_time_taken} seconds\n")
    
    # Drop columns (CpGs) that are now all NaN across all samples if necessary
    #methylation_df.dropna(axis=1, how='all', inplace=True)

    return methylation_df, artifact

async def _process_detection_columns_alt(artifact:ArtifactRef, sample_data: pd.DataFrame, config: RunnableConfig, messages: List[AnyMessage]) -> pd.DataFrame:
    column_scheme, artifact = await _get_column_scheme(artifact, sample_data, config)
    if all(s.status in ["error", "missing"] for s in [column_scheme.beta_column, column_scheme.detection_column]):
        raise ValueError(f"Unable to determine column scheme for dataset {artifact.accession_code} with artifact {artifact.path}. Beta column notes: {column_scheme.beta_column.notes}, Detection column notes: {column_scheme.detection_column.notes}")
    
    beta_pattern = re.compile(column_scheme.beta_column.pattern, re.IGNORECASE) if column_scheme.beta_column.status == "resolved" else None
    beta_columns = [idx for idx, col in enumerate(sample_data.columns) if beta_pattern.search(col)]

    detection_pattern = re.compile(column_scheme.detection_column.pattern, re.IGNORECASE) if column_scheme.detection_column.status == "resolved" else None
    detection_columns = [idx for idx, col in enumerate(sample_data.columns) if detection_pattern and detection_pattern.search(col)]

    if not beta_pattern:
        return pd.DataFrame(), artifact
    elif not detection_pattern:
        methylation_df = _matrix_shape_check(sample_data)
        return methylation_df, artifact
    
    paired_columns = [[sample_data.columns.tolist()[beta_col], sample_data.columns.tolist()[detection_col]] for beta_col, detection_col in zip(beta_columns, detection_columns)]
    methylation_rows = []
    methylation_columns = []
    for pair in paired_columns:
        data = sample_data[pair].copy()
        subject_name = pair[0].strip()
        data.rename(columns={pair[0]: "Value", pair[1]: "Detection_Pval"}, inplace=True)
        data['ID_REF'] = data.index
        data['Detection_Pval'] = pd.to_numeric(data['Detection_Pval'], errors='coerce')
        data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
        if data['Value'].equals(data['Detection_Pval']):
            data['Detection_Pval'] = 0.0
        data = data[data['Detection_Pval'] < 0.05].copy()

        methylation_rows.append([subject_name] + data["Value"].tolist())
        methylation_columns.append(['Sample'] + data["ID_REF"].tolist())

    methylation_df = _merge_to_dataframe(methylation_rows, methylation_columns, index_col="Sample") 
    return methylation_df, artifact

def _matrix_shape_check(sample_data: pd.DataFrame) -> pd.DataFrame:
    data_shape = _check_for_cpg_probes(sample_data)

    if data_shape["rows"]:
        mask = sample_data.index.str.lower().str.startswith("cg", na=False)
        cpg_rows = sample_data.index[mask].sort_values()
        return sample_data.loc[cpg_rows].T

    mask = sample_data.columns.str.lower().str.startswith("cg", na=False)
    cpg_cols = sample_data.columns[mask].sort_values()
    return sample_data.loc[:, cpg_cols]

def _identify_delimiter(first_line: str) -> str:
    if '\t' in first_line:
        return '\t'
    elif ',' in first_line:
        return ','
    elif ' ' in first_line:
        return ' '
    else:
        raise ValueError("Unable to identify delimiter. Expected tab, comma, or space.")
    
def _read_sample_data(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".gz"):
        open_func = gzip.open
    else:
        open_func = open
    with open_func(file_path, 'rt') as f:
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

def _generate_data_samples(sample_data: pd.DataFrame, seed: int = 0) -> Dict[str, Any]:
    random.seed(seed)
    n_cols = sample_data.shape[1]
    k = min(15, n_cols)
    idx = np.random.choice(n_cols, size=k, replace=False)
    sampled_columns = sample_data.columns[idx]
    sample_data_markdown = sample_data[sampled_columns].head(5).to_markdown(index=False)
    return sample_data_markdown, sample_data[sampled_columns].copy()

def _check_pattern_performance(pattern:str, columns: List[str]) -> Any:
    if not pattern:
        return set(), set(columns)
    regex = re.compile(pattern, re.IGNORECASE)
    matching_cols = [col for col in sorted(columns) if regex.search(col)]
    missing_cols = [col for col in sorted(columns) if col not in matching_cols]
    return set(matching_cols), set(missing_cols)

def _check_pattern_performance_change(
    prev_beta_pattern: str, prev_detection_pattern: Optional[str],
    current_beta_pattern: str, current_detection_pattern: Optional[str],
    columns: List[str]) -> bool:
    # Beta pattern performance
    prev_beta_matching_cols, prev_beta_missing_cols = _check_pattern_performance(prev_beta_pattern, columns)
    current_beta_matching_cols, current_beta_missing_cols = _check_pattern_performance(current_beta_pattern, columns)
    # Detection pattern performance (if applicable)
    prev_detection_matching_cols, prev_detection_missing_cols = _check_pattern_performance(prev_detection_pattern, columns)
    current_detection_matching_cols, current_detection_missing_cols = _check_pattern_performance(current_detection_pattern, columns)

    return (
        (prev_beta_matching_cols == current_beta_matching_cols) and
        (prev_beta_missing_cols == current_beta_missing_cols) and
        (prev_detection_matching_cols == current_detection_matching_cols) and
        (prev_detection_missing_cols == current_detection_missing_cols)
    )

async def _get_column_scheme(
        artifact: ArtifactRef, sample_data: pd.DataFrame, config: RunnableConfig,
        messages: List[AIMessage] = [], count=0, prohibited_patterns: Optional[List[str]] = None,
        prev_beta_pattern: Optional[str] = None, prev_detection_pattern: Optional[str] = None) -> SampleDataResolution:
    deps: Deps = config["configurable"]["deps"]
    deterministic_llm = deps.deterministic_llm
    default_llm = deps.default_llm
    print(f"\nGetting column scheme for artifact {artifact.path} with accession code {artifact.accession_code}, attempt {count+1}")
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
            json_schema=ForcedSampleDataResolution.model_json_schema())

        system_message = SystemMessage(
            id=uuid.uuid4().hex,
            content=message,
            additional_kwargs={
                'created_at': datetime.now(timezone.utc).isoformat(),
            })
        human_message = HumanMessage(
            id=uuid.uuid4().hex,
            content=f"Please find the resolution for the following DNA methylation data:\n{sample_data_markdown}",
            additional_kwargs={
                'created_at': datetime.now(timezone.utc).isoformat(),
            })
        messages = [system_message, human_message]

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    while retries < retry_limit:
        try:
            resolved: Any = await asyncio.wait_for(deterministic_llm.acall_structured(messages, ForcedSampleDataResolution), timeout=CALL_TIMEOUT)
            break
        except asyncio.TimeoutError:
            retries += 1
            continue
        except ResponseError as e:
            retries += 1
            continue
        except ValidationError as e:
            resolved: Any = SampleDataResolution(
                beta_column={
                    "status": "error",
                    "notes": [f"Validation error for beta column extraction: {e}"]
                },
                detection_column={
                    "status": "error",
                    "notes": [f"Validation error for detection column extraction: {e}"]
                })
            break
    
    new_sample_data_markdown, new_columns = _generate_data_samples(sample_data, seed=count+1)

    if prev_beta_pattern or prev_detection_pattern:
        if _check_pattern_performance_change(
            prev_beta_pattern, prev_detection_pattern,
            resolved.beta_column.pattern, resolved.detection_column.pattern if resolved.detection_column.status == "resolved" else None,
            new_columns.columns.tolist()):
                count = 4

    # Otherwise, if we are under the retry limit, try to fix
    if count < 3:
        agent_message = AIMessage(
            id=uuid.uuid4().hex,
            content=f"Attempt {count} resolution:\n {resolved.model_dump()}",
            additional_kwargs={
                'created_at': datetime.now(timezone.utc).isoformat(),
        })
        
        beta_pattern = re.compile(resolved.beta_column.pattern, re.IGNORECASE)
        beta_columns = [col for col in sorted(new_columns.columns.tolist()) if beta_pattern.search(col)]
        missing_beta_columns = [col for col in sorted(new_columns.columns.tolist()) if col not in beta_columns]
        print(f"\n Beta pattern: {beta_pattern}")
        print(f"\nBeta columns identified with pattern {resolved.beta_column.pattern}: {beta_columns}")
        print(f"\nColumns that failed to match beta pattern {resolved.beta_column.pattern}: {missing_beta_columns}")

        detection_pattern = re.compile(resolved.detection_column.pattern, re.IGNORECASE) if resolved.detection_column.status == "resolved" else None
        detection_columns = [col for col in sorted(new_columns.columns.tolist()) if detection_pattern and detection_pattern.search(col)]
        missing_detection_columns = [col for col in sorted(new_columns.columns.tolist()) if col not in detection_columns]
        if detection_pattern:
            print(f"\n Detection pattern: {detection_pattern}")
            print(f"\nDetection columns identified with pattern {resolved.detection_column.pattern}: {detection_columns}")
            print(f"\nColumns that failed to match detection pattern {resolved.detection_column.pattern}: {missing_detection_columns}")

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
                beta_columns=", ".join(beta_columns) if len(beta_columns) > 0 else "N/A, because your pattern failed to match any columns.",
                not_beta_columns=", ".join(missing_beta_columns) if len(missing_beta_columns) > 0 else "N/A, all columns were identified as being `beta_column`s.",
                detection_pattern=resolved.detection_column.pattern if resolved.detection_column.status == "resolved" else "N/A, because you don't believe there are any detection columns.",
                detection_column=", ".join(detection_columns) if len(detection_columns) > 0 else "N/A, because you don't believe there are any detection columns.",
                not_detection_columns=", ".join(missing_detection_columns) if len(missing_detection_columns) > 0 else "N/A, because you don't believe there are any detection columns.",
                json_schema=ForcedSampleDataResolution.model_json_schema()
            )
            correction_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=base_prompt,
                additional_kwargs={
                    'created_at': datetime.now(timezone.utc).isoformat()
                })
            return await _get_column_scheme(
                artifact, sample_data, config,
                messages=messages + [agent_message, correction_message],
                count=count+1,
                prev_beta_pattern=resolved.beta_column.pattern,
                prev_detection_pattern=resolved.detection_column.pattern if resolved.detection_column.status == "resolved" else None)
            
    column_scheme_path = os.path.splitext(artifact.path)[0] + ".json"
    with open(column_scheme_path, "w", encoding="utf-8") as f:
        json.dump(
            resolved.model_dump(),
            f,
            ensure_ascii=False,
            indent=2)
    column_scheme_artifact = ArtifactRef.model_validate({
        "accession_code": artifact.accession_code,
        "path": column_scheme_path,
        "kind": "supplementary_file_methylation_data_column_scheme",
        "sha256": compute_sha256(column_scheme_path, is_path=True),
        "bytes": os.path.getsize(column_scheme_path),
        "created_at": datetime.now(timezone.utc).isoformat()
    })
            
    return resolved, column_scheme_artifact
    
async def format_individual_methylation_data(
        accession_code: str, return_dict: Dict[str, Any], config: RunnableConfig,
        artifact: ArtifactRef, messages: List[AnyMessage]) -> Dict[str, Any]:
    methylation_dataframe_output_path = f"{os.path.splitext(artifact.path)[0]}_proc.feather"

    sample_data = _read_sample_data(artifact.path)
    if _check_for_detection_columns(sample_data):
        df, _ = await _process_detection_columns(artifact, sample_data, config, messages)
    else:
        df = _matrix_shape_check(sample_data)

    write_feather(df, methylation_dataframe_output_path, index_name="subject_id")
    methylation_artifact = ArtifactRef.model_validate({
        "accession_code": accession_code,
        "path": methylation_dataframe_output_path,
        "kind": "supplementary_file_methylation_data_formatted",
        "sha256": compute_sha256(methylation_dataframe_output_path, is_path=True),
        "bytes": os.path.getsize(methylation_dataframe_output_path),
        "created_at": datetime.now(timezone.utc).isoformat()})
    
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]],
        [methylation_artifact])
    return_dict["datasets"][accession_code]["supplementary_data"][artifact.sha256] = "running"
    return return_dict

def merge_formatted_supplementary_data(state: GeoIngestionSubgraphState, accession_code: str) -> pd.DataFrame:
    state_config = state.config
    dataset_state = state.datasets[accession_code]
    print(f"\nDataset supplementary data state: {dataset_state.supplementary_data}")
    formatted_artifacts = [a for a in state_config.artifacts if (a.kind == "supplementary_file_methylation_data_formatted") and (a.accession_code == accession_code)]
    formatted_datasets = [pd.read_csv(artifact.path, index_col=0) for artifact in formatted_artifacts]
    merged_data = pd.concat(formatted_datasets, axis=0)
    return merged_data
    
async def format_methylation_data(state_config: GEOIngestionConfig, state: GeoDatasetState, config: RunnableConfig, messages: List[AnyMessage]) -> Dict[str, Any]:
    formatted_datasets = []
    accession_code = state.accession
    methylation_dataframe_output_path = os.path.join(state.output_dir, "preqc_methylation_matrix.csv")
    return_dict = {
        "config": state_config.model_dump()
    } 

    sample_data_artifacts = [a for a in state_config.artifacts if (a.kind == "supplementary_file_methylation_data") and (a.accession_code == accession_code)]
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
        methylation_artifact = ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": methylation_dataframe_output_path,
            "kind": "preqc_methylation_data",
            "sha256": compute_sha256(methylation_dataframe_output_path, is_path=True),
            "bytes": os.path.getsize(methylation_dataframe_output_path),
            "created_at": datetime.now(timezone.utc).isoformat()})
    else:
        if next((a for a in state_config.artifacts if a.path == methylation_dataframe_output_path), None) is None:
            methylation_artifact = ArtifactRef.model_validate({
                "accession_code": accession_code,
                "path": methylation_dataframe_output_path,
                "kind": "preqc_methylation_data",
                "sha256": compute_sha256(methylation_dataframe_output_path, is_path=True),
                "bytes": os.path.getsize(methylation_dataframe_output_path),
                "created_at": datetime.now(timezone.utc).isoformat()})
    collected_artifacts.append(methylation_artifact)
    
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]],
        collected_artifacts)
    return return_dict

def generate_lexical_features(name: str) -> LexFeat:
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
        s = ALNUM_BOUND_RE.sub(" ", s)   # <-- ADD THIS
        s = NON_ALNUM_RE.sub(" ", s)
        s = WS_RE.sub(" ", s)
        return s.lower().strip()

    def tokenize(s: str) -> List[str]:
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

def build_vocab(featsA: List[LexFeat], featsB: List[LexFeat], attr: str) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for f in featsA:
        for t in set(getattr(f, attr)):
            if t not in vocab:
                vocab[t] = len(vocab)
    for f in featsB:
        for t in set(getattr(f, attr)):
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab

def incidence_matrix(feats: List[LexFeat], vocab: Dict[str, int], attr: str) -> np.ndarray:
    # Use uint8 to keep memory tiny; we cast for matmul.
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



def lexical_score_matrix(A: List[str], B: List[str], workers: int = -1,  no_count_penalty: bool = False) -> np.ndarray:
    featsA = [generate_lexical_features(x) for x in A]
    featsB = [generate_lexical_features(x) for x in B]

    # tsr: compiled all-pairs
    A_norm = [f.norm for f in featsA]
    B_norm = [f.norm for f in featsB]
    tsr = process.cdist(A_norm, B_norm, scorer=fuzz.token_sort_ratio, score_cutoff=0, workers=workers)
    tsr = (np.asarray(tsr, dtype=np.float32) / 100.0)  # (N,M)

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
    #A_len = np.array([len(f.norm) for f in featsA], dtype=np.float32)[:, None]
    #B_len = np.array([len(f.norm) for f in featsB], dtype=np.float32)[None, :]
    #max_len = np.maximum(A_len, B_len)
    #rel_len = np.where(max_len > 0, np.abs(A_len - B_len) / max_len, 0.0)
    #len_pen = np.exp(-1.5 * rel_len).astype(np.float32)

    S = (S * tok_pen).astype(np.float32)
    return S

def extract_subject_column_mapping(
        metadata_artifact: ArtifactRef, sample_data_df: pd.DataFrame, return_dict: Dict[str, Any]) -> Any:
    metadata_df = pd.read_csv(metadata_artifact.path, index_col=0)
    subject_ids = metadata_df['Subject'].tolist()
    print(f"\nMetadata subject IDs: {subject_ids}")
    sample_data_subjects = sample_data_df.index.tolist()
    print(f"\nSample data subjects: {sample_data_subjects}")
    score_matrix = lexical_score_matrix(sample_data_subjects, subject_ids)
    best_j = np.argmax(score_matrix, axis=1)
    mapper = {
        sample_data_subjects[i]: subject_ids[j] for i, j in enumerate(best_j)
    }
    print(f"\nGenerated subject mapping: {mapper}")
    # Save mapper
    mapper_artifact_path = os.path.join(
        os.path.dirname(metadata_artifact.path),
        f"{metadata_artifact.accession_code}_subject_mapping.json")
    with open(mapper_artifact_path, "w", encoding="utf-8") as f:
        json.dump(mapper, f, ensure_ascii=False, indent=2)
    mapper_artifact = ArtifactRef.model_validate({
        "accession_code": metadata_artifact.accession_code,
        "path": mapper_artifact_path,
        "kind": "subject_column_mapping",
        "sha256": compute_sha256(mapper_artifact_path, is_path=True),
        "bytes": os.path.getsize(mapper_artifact_path),
        "created_at": datetime.now(timezone.utc).isoformat()
    })
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        return_dict["config"]["artifacts"],
        [mapper_artifact])

    return return_dict

def _extract_best_subject_id_fields(user_input: Any, sample_data: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
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
    accession_code: str, user_input: pd.DataFrame, metadata_dict: Dict[str, Any], sample_data: pd.DataFrame
):
    # This is to map from the original subject id to the best subject id
    # I have the field name and key name, I need to extract that here
    def _extract_subject(
            field_name: str, target_values: List[str], metadata_dict: Dict[str, Any], 
            key_name: Optional[str] = None):
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
        score_matrix = lexical_score_matrix(
            candidate_subjects, target_values, no_count_penalty = True)
        best_j = np.argmax(score_matrix, axis=1)
        for i, j in enumerate(best_j):
            if score_matrix[i, j] > max_score:
                max_score = score_matrix[i, j]
                subject = candidate_subjects[i]
                sample = target_values[j]
                print(f"\nExtracted subject {subject} for value {sample} in field {field_name} and key {key_name} with score {max_score}")
        return subject
    
    def _get_subject_column_values(source_values: List[str], target_values: List[str]) -> Any:
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
        rows.append({
            'Sample': gsm_name,
            'Subject': _extract_subject(field_name, target_values, gsm, key_name=key_name),
        })
    
    df = pd.DataFrame(rows)
    df['Beta_Subjects'] = _get_subject_column_values(df['Subject'].tolist(), target_values)
    return df
