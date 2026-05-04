__all__ = ["quality_control_plan_node", "quality_control_node"]

from dataclasses import Field
import os
import json
import hashlib
import pandas as pd
from datetime import datetime, timezone
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command
from typing import Dict, Any, List, Optional
from ...utils.logging import setup_logger
from ...utils.helper import (
    get_accession_codes, consolidate_artifacts, update_small_progress_tracker, load_metadata_aligned_methylation_data,
    read_feather, qc_progress)
from ..state.models import QualityControlSubgraphState, DatasetQualityControlState
from ...tools.qc.workflow import run_all_qc
from langchain_core.messages import ToolMessage
from ...tools.qc.data_type_conversion import detect_data_type
from ...contracts.qc import PreprocessDataInput, DNAmQCInput, CpGLevelQCInput, SampleLevelQCInput, InterarrayCorrelationQCInput, ImputationInput

def _get_data_conversion_input_or_default(accession_code: str, state: DatasetQualityControlState) -> PreprocessDataInput:
    """
    Retrieves the data conversion input for the given accession code from the state. If the data conversion input is not present in the state, it attempts to infer the necessary information from the preqc_methylation_data artifact associated with the accession code. It reads the methylation data, detects its type, and constructs a PreprocessDataInput object with the detected type as the from_type and "beta" as the to_type. If the required artifact is missing, it raises a ValueError.

    Args:
        accession_code (str): The accession code for which to retrieve the data conversion input
        state (DatasetQualityControlState): The current state of the dataset quality control process, which may contain a pre-existing data conversion input or artifacts from which to infer the input
    
    Returns:
        PreprocessDataInput: An object containing the from_type and to_type for data conversion, either retrieved from the state or inferred from the preqc_methylation_data artifact
    """
    if state.data_conversion_input is not None:
        return state.data_conversion_input
    
    relevant_artifacts = [artifact for artifact in state.config.artifacts if artifact.accession_code == accession_code and artifact.kind == "preqc_methylation_data"]
    geo_data_path = next((artifact.path for artifact in relevant_artifacts if "cache" not in artifact.path), None) if len(relevant_artifacts) > 1 else relevant_artifacts[0].path if relevant_artifacts else None
    if geo_data_path is None:
        raise ValueError(f"Missing preqc_methylation_data artifact for accession {accession_code}")
    
    data_df = read_feather(geo_data_path, index_name="subject_id")
    detected_type = detect_data_type(data_df)
    to_type = "beta"

    return PreprocessDataInput(from_type=detected_type, to_type=to_type)

def _get_dnam_qc_input_or_default(state: DatasetQualityControlState) -> DNAmQCInput:
    """
    Retrieves the DNAm QC input for the given state. If the DNAm QC input is not present in the state, it returns a default DNAmQCInput object with a dnam_cutoff of 0.96.

    Args:
        state (DatasetQualityControlState): The current state of the dataset quality control process, which may contain a pre-existing DNAm QC input
    
    Returns:
        DNAmQCInput: An object containing the DNAm QC input, either retrieved from the state or a default value
    """
    if state.dnam_qc_input is not None:
        return state.dnam_qc_input
    return DNAmQCInput(dnam_cutoff=0.96)

def _get_cpg_level_qc_input_or_default(state: DatasetQualityControlState) -> CpGLevelQCInput:
    """
    Retrieves the CpG level QC input for the given state. If the CpG level QC input is not present in the state, it returns a default CpGLevelQCInput object with a missing_cutoff of 0.2 and an imputation strategy of "whole".

    Args:
        state (DatasetQualityControlState): The current state of the dataset quality control process, which may contain a pre-existing CpG level QC input
    
    Returns:
        CpGLevelQCInput: An object containing the CpG level QC input, either retrieved from the state or a default value
    """
    if state.cpg_level_qc_input is not None:
        return state.cpg_level_qc_input
    return CpGLevelQCInput(missing_cutoff=0.2, imputation_strategy=ImputationInput(strategy="whole"))

def _get_sample_level_qc_input_or_default(state: DatasetQualityControlState) -> SampleLevelQCInput:
    """
    Retrieves the sample level QC input for the given state. If the sample level QC input is not present in the state, it returns a default SampleLevelQCInput object with a missing_cutoff of 0.1.

    Args:
        state (DatasetQualityControlState): The current state of the dataset quality control process, which may contain a pre-existing sample level QC input
    
    Returns:
        SampleLevelQCInput: An object containing the sample level QC input, either retrieved from the state or a default value
    """
    if state.sample_level_qc_input is not None:
        return state.sample_level_qc_input
    return SampleLevelQCInput(missing_cutoff=0.1)

def _get_interarray_correlation_qc_input_or_default(state: DatasetQualityControlState) -> InterarrayCorrelationQCInput:
    """
    Retrieves the interarray correlation QC input for the given state. If the interarray correlation QC input is not present in the state, it returns a default InterarrayCorrelationQCInput object with a correlation_cutoff of 0.9.

    Args:
        state (DatasetQualityControlState): The current state of the dataset quality control process, which may contain a pre-existing interarray correlation QC input
    
    Returns:
        InterarrayCorrelationQCInput: An object containing the interarray correlation QC input, either retrieved from the state or a default value
    """
    if state.interarray_correlation_qc_input is not None:
        return state.interarray_correlation_qc_input
    return InterarrayCorrelationQCInput(correlation_cutoff=0.9)

def quality_control_node(state: QualityControlSubgraphState, *, config: RunnableConfig) -> Command:
    """
    Executes the quality control process for the datasets in the given state and configuration. The function first checks if the quality control step has already been completed for all datasets in the state. If it has, it returns a Command with an update that includes messages to update the QC progress tracker. If not, it identifies the accession codes for which the quality control step is still running or not started, and retrieves the relevant artifacts for one of those accession codes. If the necessary preqc_methylation_data artifact is missing, it updates the state to mark the quality control step as failed for that accession code and returns a Command with the updated state. If the artifact is available, it calls the run_all_qc function to perform quality control on the dataset, updates the state with the results, and returns a Command with the updated state and progress messages.

    Args:
        state (QualityControlSubgraphState): The current state of the quality control subgraph, which may contain information about the datasets, their quality control status, and any relevant artifacts
        config (RunnableConfig): The configuration for the runnable, which may include parameters for the quality control process  
    
    Returns:
        Command: A Command object containing the updates to the state after performing quality control, including any
    """
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["quality_control"].status in {'running', 'not_started'}])
    if not running_accession_codes:
        return Command(update={})
    
    accession_code = running_accession_codes[0]
    logger = setup_logger(os.path.join(state.config.output_root, accession_code), f"{accession_code}_tasks", "tasks.log")
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {
            accession_code: state.datasets[accession_code].model_dump()
        }
    }
    
    dataset_state = state.datasets[accession_code]
    relevant_artifacts = [artifact for artifact in state.config.artifacts if artifact.accession_code == accession_code and artifact.kind == "preqc_methylation_data"]
    geo_data_path = next((artifact.path for artifact in relevant_artifacts if "cache" not in artifact.path), None) if len(relevant_artifacts) > 1 else relevant_artifacts[0].path if relevant_artifacts else None
    
    logger.info(f"\nQC Node: Processing accession code {accession_code} with data at {geo_data_path}")
    if geo_data_path is None:
        logger.info(f"Error: Missing preqc_methylation_data artifact for accession {accession_code}")
        state.errors.append("Missing preqc_methylation_data artifact for accession {}".format(accession_code))
        return_dict["datasets"][accession_code]["steps"]["quality_control"]["status"] = "failed"
        return Command(update=return_dict)

    qc_results = run_all_qc(
        accession_code,
        data_path=geo_data_path,
        processed_path=os.path.join(dataset_state.output_dir, f"{accession_code}_processed_data.feather"),
        data_conversion_input=_get_data_conversion_input_or_default(accession_code, state),
        sample_level_qc_input=_get_sample_level_qc_input_or_default(state),
        cpg_level_qc_input=_get_cpg_level_qc_input_or_default(state),
        dnam_qc_input=_get_dnam_qc_input_or_default(state),
        interarray_correlation_qc_input=_get_interarray_correlation_qc_input_or_default(state),
        logger=logger)
    
    dataset_state.data_conversion_result = qc_results["data_conversion_result"]
    dataset_state.sample_level_qc_result = qc_results["sample_level_qc_result"]
    dataset_state.cpg_level_qc_result = qc_results["cpg_level_qc_result"]
    dataset_state.dnam_qc_result = qc_results["dnam_qc_result"]
    dataset_state.interarray_correlation_qc_result = qc_results["interarray_correlation_qc_result"]
    dataset_state.artifacts = qc_results["artifacts"]
    return_dict["datasets"][accession_code] = dataset_state.model_dump()
    return_dict["datasets"][accession_code]["artifacts"] = consolidate_artifacts(
        dataset_state.artifacts,
        qc_results["artifacts"])
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        state.config.artifacts,
        qc_results["artifacts"])
    return_dict["datasets"][accession_code]["steps"]["quality_control"]["status"] = "completed"
    progress_message = qc_progress(state)
    return_dict["messages"] = [progress_message]
    return_dict["main_messages"] = [progress_message]

    return Command(update=return_dict)

def quality_control_summarization_node(state: QualityControlSubgraphState) -> QualityControlSubgraphState:
    """
    Summarizes the quality control findings for the datasets in the given state. The function generates a payload containing a summary of the QC results for each dataset, including the accession code, pre-QC sample count, post-QC sample count, and post-QC CpG count. It then creates a ToolMessage with the summary and updates the state with this message to be returned to the user. The function also includes progress messages to indicate the status of the QC summarization process.

    Args:
        state (QualityControlSubgraphState): The current state of the quality control subgraph, which may contain information about the datasets, their quality control results, and any relevant artifacts
    
    Returns:
        QualityControlSubgraphState: The updated state of the quality control subgraph, which includes messages summarizing the QC findings for the datasets, as well as any progress messages related to the QC summarization process
    """
    payload = {
        "id": "quality-control-summary",
        "rowIdKey": "accession_code",
        "columns": [
            {
                "key": "accession_code",
                "label": "Accession Code",
                "priority": "primary",
                "align": "left",
            },
            {
                "key": "pre_qc_n",
                "label": "(Pre-QC) Sample Count",
                "priority": "primary",
                "align": "right",
                "format": {
                    "kind": "number",
                }
            },
            {
                "key": "post_qc_n",
                "label": "(Post-QC) Sample Count",
                "priority": "primary",
                "align": "right",
                "format": {
                    "kind": "number",
                }
            },
            {
                "key": "post_qc_num_cpgs",
                "label": "(Post-QC) CpG Count",
                "priority": "primary",
                "align": "right",
                "format": {
                    "kind": "number",
                }
            }
        ],
        "rows": []
    }
    accession_codes = sorted([x for x in state.datasets.keys() if state.datasets[x].steps["quality_control"].status == "completed"])

    for accession_code in accession_codes:
        preqc_methylation_df = load_metadata_aligned_methylation_data(accession_code, state.config.artifacts)
        postqc_methylation_artifact = next((artifact for artifact in state.config.artifacts if artifact.accession_code == accession_code and artifact.kind == "postqc_methylation_data"), None)
        postqc_methylation_df = read_feather(postqc_methylation_artifact.path, index_name="subject_id")

        payload["rows"].append({
            "accession_code": accession_code,
            "pre_qc_n": preqc_methylation_df.shape[0],
            "post_qc_n": postqc_methylation_df.shape[0],
            "post_qc_num_cpgs": postqc_methylation_df.shape[1]
        })
        
    m = hashlib.sha256()
    for accession_code in sorted(accession_codes):
        m.update(accession_code.encode("utf-8"))
    m.update("Quality Control".encode("utf-8"))
    hash = m.hexdigest()

    payload["message"] = f"We performed quality control on {len(accession_codes)} datasets from GEO. Here is a summary of the datasets after the changes:"
    message = ToolMessage(
        id=hash,
        content=payload["message"],
        tool_call_id=hash,
        artifact=payload,
        additional_kwargs={
            "name": "geoDatasetSummary",
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
    )

    progress_message = qc_progress(state)
    return_dict = {
        'main_messages': [progress_message, message],
        'messages': [progress_message, message]}
    return Command(update=return_dict)
