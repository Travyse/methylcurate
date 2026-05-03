__all__ = [
    "disease_harmonization_node", "higher_level_disease_mapping_node",
    "tissue_harmonization_node", "higher_level_tissue_mapping_node",
    "cell_type_harmonization_node", "sex_harmonization_node"]

import os
import json
import pandas as pd
from datetime import datetime, timezone
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any, List, Optional

from ...contracts.common import ArtifactRef
from ...contracts.harmonize import LabelMappingSet
from ..policies.validations.harmonization import harmonization_verification
from ..state.models import HarmonizationIngestionConfig, HarmonizationSubgraphState
from ...tools.harmonize import (
    _harmonize_ontology_labels, _harmonize_ontology_group_labels, _harmonize_sex_labels,
    construct_raw_to_harmonized_label_mapping)
from ...utils.helper import get_accession_codes, check_step_completion, update_harmonization_progress_tracker, compute_sha256, consolidate_artifacts
from ...utils.examples import generate_metadata_harmonization_examples

def _get_metadata_cache(accession_code: str, artifacts: List[ArtifactRef]):
    """
    Gets the metadata cache for a given accession code from the list of artifacts. The metadata cache is expected to be a JSON file that contains cached metadata information for the dataset corresponding to the accession code. If no such artifact is found, the function returns None.

    Args:
        accession_code (str): The accession code for which to retrieve the metadata cache.
        artifacts (List[ArtifactRef]): A list of ArtifactRef objects representing the available artifacts.

    Returns:
        dict: The metadata cache for the given accession code, or None if not found.
    """
    for artifact in artifacts:
        if artifact.accession_code == accession_code and artifact.kind == "metadata_cache":
            with open(artifact.path, 'r') as f:
                return json.load(f)
    return None

def _get_sample_metadata(accession_code: str, artifacts: List[ArtifactRef]) -> Optional[pd.DataFrame]:
    """
    Gets the sample metadata for a given accession code from the list of artifacts. The sample metadata is expected to be a CSV file that contains metadata information for the dataset corresponding to the accession code. If no such artifact is found, the function returns None.

    Args:
        accession_code (str): The accession code for which to retrieve the sample metadata.
        artifacts (List[ArtifactRef]): A list of ArtifactRef objects representing the available artifacts.

    Returns:
        pd.DataFrame: The sample metadata for the given accession code, or None if not found.
    """
    for artifact in artifacts:
        if artifact.accession_code == accession_code and artifact.kind == "dataset_metadata":
            return pd.read_csv(artifact.path, index_col=0)
    return None

def _get_extraction_protocol(accession_code: str, artifacts: List[ArtifactRef]):
    """
    Gets the extraction protocol for a given accession code from the list of artifacts. The extraction protocol is expected to be a JSON file that contains the extraction protocol information for the dataset corresponding to the accession code. If no such artifact is found, the function returns None.

    Args:
        accession_code (str): The accession code for which to retrieve the extraction protocol.
        artifacts (List[ArtifactRef]): A list of ArtifactRef objects representing the available artifacts.

    Returns:
        dict: The extraction protocol for the given accession code, or None if not found.
    """
    for artifact in artifacts:
        if artifact.accession_code == accession_code and artifact.kind == "metadata_extraction_protocol":
            with open(artifact.path, 'r') as f:
                return json.load(f)
    return None

def _get_guess_results(target_type:str, accession_code: str, artifacts: List[ArtifactRef]) -> LabelMappingSet:
    """
    Gets the label guessing results for a given target type and accession code from the list of artifacts. The label guessing results are expected to be a JSON file that contains the label guessing information for the dataset corresponding to the accession code. If no such artifact is found, the function raises a ValueError.

    Args:
        target_type (str): The target type for which to retrieve the label guessing results.
        accession_code (str): The accession code for which to retrieve the label guessing results.
        artifacts (List[ArtifactRef]): A list of ArtifactRef objects representing the available artifacts.

    Returns:
        LabelMappingSet: The label guessing results for the given target type and accession code.

    Raises:
        ValueError: If no label guessing results are found for the given target type and accession code.
    """
    for artifact in artifacts:
        if artifact.accession_code == accession_code and artifact.kind == f"{target_type}_label_guessing":
            return LabelMappingSet.model_validate(json.load(open(artifact.path, 'r')))
    raise ValueError(f"No harmonization results found for {target_type} and accession code {accession_code}")

def _get_harmonization_results(target_type:str, accession_code: str, artifacts: List[ArtifactRef]) -> LabelMappingSet:
    """
    Gets the label harmonization results for a given target type and accession code from the list of artifacts. The label harmonization results are expected to be a JSON file that contains the label harmonization information for the dataset corresponding to the accession code. If no such artifact is found, the function raises a ValueError.

    Args:
        target_type (str): The target type for which to retrieve the label harmonization results.
        accession_code (str): The accession code for which to retrieve the label harmonization results.
        artifacts (List[ArtifactRef]): A list of ArtifactRef objects representing the available artifacts.

    Returns:
        LabelMappingSet: The label harmonization results for the given target type and accession code.

    Raises:
        ValueError: If no label harmonization results are found for the given target type and accession code.
    """
    for artifact in artifacts:
        if artifact.accession_code == accession_code and artifact.kind == f"{target_type}_label_harmonization":
            return LabelMappingSet.model_validate(json.load(open(artifact.path, 'r')))
    raise ValueError(f"No harmonization results found for {target_type} and accession code {accession_code}")

async def disease_harmonization_node(state: HarmonizationSubgraphState, *, config: RunnableConfig):
    """
    Performs disease harmonization for a given state and configuration.
    The function first checks if the disease harmonization step has already been completed for all datasets in the state. If it has, it returns a Command with an update that includes messages to update the harmonization progress tracker. If not, it identifies the accession codes for which the disease harmonization step is still running or not started, and selects one accession code to process.
    For the selected accession code, the function retrieves the metadata cache, sample metadata, and extraction protocol from the artifacts. If any of these are missing, or if the extraction protocol indicates that disease status is missing or should not be extracted, the function updates the state to mark the disease harmonization steps as canceled for that accession code and returns a Command with the updated state.
    If the necessary information is available, the function calls the _harmonize_ontology_labels helper function to perform disease label harmonization. The guessed and harmonized disease labels are then saved as JSON artifacts, and the state is updated to mark the disease harmonization step as completed and the disease grouping step as running for that accession code. Finally, a Command with the updated state is returned.

    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph
        config (RunnableConfig): The configuration for the runnable, which may include parameters for the harmonization process.

    Returns:
        Command: A Command object containing the updates to the state after performing disease harmonization.
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("map_disease_labels_to_ontology", state.datasets, accession_codes):
        return Command(update={
            "main_messages": [update_harmonization_progress_tracker(state)],
            "messages": [update_harmonization_progress_tracker(state)]
        })
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["map_disease_labels_to_ontology"].status in {'running', 'not_started'}])
    accession_code = running_accession_codes.pop(0)
    harmonization_dir = os.path.join(state.datasets[accession_code].output_dir, "harmonization")
    os.makedirs(harmonization_dir, exist_ok=True)
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {
            accession_code: state.datasets[accession_code].model_dump()
        }
    }
    metadata_cache = _get_metadata_cache(accession_code, state.config.artifacts)
    sample_metadata  = _get_sample_metadata(accession_code, state.config.artifacts)
    extraction_protocol = _get_extraction_protocol(accession_code, state.config.artifacts)
    if any(x is None for x in [metadata_cache, sample_metadata, extraction_protocol]):
        return_dict["datasets"][accession_code]["steps"]["map_disease_labels_to_ontology"]["status"] = "canceled"
        return_dict["datasets"][accession_code]["steps"]["group_disease_labels"]["status"] = "canceled"
        return_dict["datasets"][accession_code]["steps"]["map_tissue_labels_to_ontology"]["status"] = "canceled"
        return_dict["datasets"][accession_code]["steps"]["group_tissue_labels"]["status"] = "canceled"
        return_dict["datasets"][accession_code]["steps"]["map_cell_type_labels_to_ontology"]["status"] = "canceled"
        return_dict["datasets"][accession_code]["steps"]["harmonize_sex_labels"]["status"] = "canceled"
        return_dict["messages"] = [update_harmonization_progress_tracker(state)]
        return_dict["main_messages"] = [update_harmonization_progress_tracker(state)]
        return Command(update=return_dict)
    if extraction_protocol["disease_status"]["status"] == "missing":
        return_dict["datasets"][accession_code]["steps"]["map_disease_labels_to_ontology"]["status"] = "canceled"
        return_dict["datasets"][accession_code]["steps"]["group_disease_labels"]["status"] = "canceled"
        return Command(update=return_dict)
    elif extraction_protocol["disease_status"]["extraction"]["field_name"] == "default":
        return_dict["datasets"][accession_code]["steps"]["map_disease_labels_to_ontology"]["status"] = "canceled"
        return_dict["datasets"][accession_code]["steps"]["group_disease_labels"]["status"] = "canceled"
        return Command(update=return_dict)
    
    guessed_disease_labels, _, harmonized_disease_labels = await _harmonize_ontology_labels(
        metadata_cache, extraction_protocol, sample_metadata, config, ontology = "mondo", column_name = "Disease_Status")
    guessed_disease_labels_path = os.path.join(harmonization_dir, "guessed_disease_labels.json")
    harmonized_disease_labels_path = os.path.join(harmonization_dir, "harmonized_disease_labels.json")
    artifacts = []
    with open(guessed_disease_labels_path, "w") as f:
        json.dump(guessed_disease_labels.model_dump(), f, indent=2)
    with open(harmonized_disease_labels_path, "w") as f:
        json.dump(harmonized_disease_labels.model_dump(), f, indent=2)
    artifacts.extend([
        ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": guessed_disease_labels_path,
            "kind": "disease_label_guessing",
            "sha256": compute_sha256(guessed_disease_labels_path, is_path=True),
            "bytes": os.path.getsize(guessed_disease_labels_path),
            "created_at": datetime.now(timezone.utc).isoformat()
        }),
        ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": harmonized_disease_labels_path,
            "kind": "disease_label_harmonization",
            "sha256": compute_sha256(harmonized_disease_labels_path, is_path=True),
            "bytes": os.path.getsize(harmonized_disease_labels_path),
            "created_at": datetime.now(timezone.utc).isoformat()
        })])
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]],
        artifacts)
    return_dict["datasets"][accession_code]["steps"]["map_disease_labels_to_ontology"]["status"] = "completed"
    return_dict["datasets"][accession_code]["steps"]["group_disease_labels"]["status"] = "running"
    return_dict["messages"] = [update_harmonization_progress_tracker(state)]
    return_dict["main_messages"] = [update_harmonization_progress_tracker(state)]
    return Command(update=return_dict)

async def higher_level_disease_mapping_node(state: HarmonizationSubgraphState, *, config: RunnableConfig):
    """
    Performs higher level disease label mapping for the datasets in the given state and configuration. The function first checks if the disease grouping step has already been completed for all datasets in the state. If it has, it returns a Command with an update that includes messages to update the harmonization progress tracker. If not, it identifies the accession codes for which the disease grouping step is still running or not started, and retrieves the harmonization results for those accession codes. It then extracts the harmonized disease labels from the harmonization results and calls the _harmonize_ontology_group_labels helper function to perform higher level disease label mapping. The resulting group mapping is then used to construct a mapping from raw disease labels to harmonized group labels for each dataset, which is saved as a CSV artifact. The state is updated to mark the disease grouping step as completed and the tissue label mapping step as running for the processed accession codes, and a Command with the updated state is returned.

    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph
        config (RunnableConfig): The configuration for the runnable, which may include parameters for the harmonization process.
    
    Returns:
        Command: A Command object containing the updates to the state after performing higher level disease label mapping
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("group_disease_labels", state.datasets, accession_codes):
        return Command(update={
            "main_messages": [update_harmonization_progress_tracker(state)],
            "messages": [update_harmonization_progress_tracker(state)]
        })
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["group_disease_labels"].status in {'running', 'not_started'}])
    running_disease_mappings = [_get_harmonization_results('disease', accession_code, state.config.artifacts) for accession_code in running_accession_codes]

    harmonized_labels = sorted(set([x.target_label for disease_mapping in running_disease_mappings for x in disease_mapping.mappings if hasattr(x, 'target_label')]))
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {
            accession_code: state.datasets[accession_code].model_dump() for accession_code in running_accession_codes
        }
    }
    granular_to_coarse_disease_label_mapping, _, disease_group_harmonization_mapping = await _harmonize_ontology_group_labels(
        harmonized_labels, config, ontology = "mondo")
    
    proper_group_mapping = construct_raw_to_harmonized_label_mapping(
        granular_to_coarse_disease_label_mapping,
        disease_group_harmonization_mapping)
    return_dict["disease_group_mapping"] = disease_group_harmonization_mapping.model_dump()
    artifacts = []
    for accession_code in running_accession_codes:
        disease_mapping = _get_harmonization_results('disease', accession_code, state.config.artifacts)
        if all(not hasattr(x, "target_label") for x in disease_mapping.mappings):
            continue
        rows = []
        disease_label_guessing = _get_guess_results('disease', accession_code, state.config.artifacts)
        disease_label_mapping = _get_harmonization_results('disease', accession_code, state.config.artifacts)
        proper_mapping = construct_raw_to_harmonized_label_mapping(
            disease_label_guessing,
            disease_label_mapping)
        for mapping in proper_mapping.mappings:
            harmonized_group_label = next(group_mapping.target_label for group_mapping in proper_group_mapping.mappings if group_mapping.source_label == mapping.target_label)
            rows.append({
                "original_label": mapping.source_label,
                "harmonized_label": mapping.target_label,
                "harmonized_group_label": harmonized_group_label
            })
        harmonization_result_df = pd.DataFrame(rows)
        harmonization_result_path = os.path.join(state.datasets[accession_code].output_dir, "disease_harmonization_metadata.csv")
        harmonization_result_df.to_csv(harmonization_result_path, index=True)
        artifacts.append(ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": harmonization_result_path,
            "kind": "disease_harmonization_mapping",
            "sha256": compute_sha256(harmonization_result_path, is_path=True),
            "bytes": os.path.getsize(harmonization_result_path),
            "created_at": datetime.now(timezone.utc).isoformat()
        }))
        return_dict["datasets"][accession_code]["steps"]["group_disease_labels"]["status"] = "completed"
        return_dict["datasets"][accession_code]["steps"]["map_tissue_labels_to_ontology"]["status"] = "running"
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]],
        artifacts)
    return_dict["messages"] = [update_harmonization_progress_tracker(state)]
    return_dict["main_messages"] = [update_harmonization_progress_tracker(state)]
    return Command(update=return_dict)

async def tissue_harmonization_node(state: HarmonizationSubgraphState, *, config: RunnableConfig):
    """
    Performs tissue label harmonization for the datasets in the given state and configuration. The function first checks if the tissue label mapping step has already been completed for all datasets in the state. If it has, it returns a Command with an update that includes messages to update the harmonization progress tracker. If not, it identifies the accession codes for which the tissue label mapping step is still running or not started, and retrieves the harmonization results for those accession codes. It then extracts the harmonized tissue labels from the harmonization results and calls the _harmonize_ontology_labels helper function to perform tissue label harmonization. The resulting harmonized labels are saved as JSON artifacts. The state is updated to mark the tissue label mapping step as completed for the processed accession codes, and a Command with the updated state is returned.
    
    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph
        config (RunnableConfig): The configuration for the runnable, which may include parameters for the harmonization process.

    Returns:
        Command: A Command object containing the updates to the state after performing tissue label harmonization.
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("map_tissue_labels_to_ontology", state.datasets, accession_codes):
        return Command(update={
            "main_messages": [update_harmonization_progress_tracker(state)],
            "messages": [update_harmonization_progress_tracker(state)]
        })
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["map_tissue_labels_to_ontology"].status in {'running', 'not_started'}])
    accession_code = running_accession_codes.pop(0)
    harmonization_dir = os.path.join(state.datasets[accession_code].output_dir, "harmonization")

    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {
            accession_code: state.datasets[accession_code].model_dump()
        }
    }
    metadata_cache = _get_metadata_cache(accession_code, state.config.artifacts)
    sample_metadata  = _get_sample_metadata(accession_code, state.config.artifacts)
    extraction_protocol = _get_extraction_protocol(accession_code, state.config.artifacts)
    if extraction_protocol["tissue"]["status"] == "missing":
        return_dict["datasets"][accession_code]["steps"]["map_tissue_labels_to_ontology"]["status"] = "canceled"
        return_dict["datasets"][accession_code]["steps"]["group_tissue_labels"]["status"] = "canceled"
        return Command(update=return_dict)
    elif extraction_protocol["tissue"]["extraction"]["field_name"] == "default":
        return_dict["datasets"][accession_code]["steps"]["map_tissue_labels_to_ontology"]["status"] = "canceled"
        return_dict["datasets"][accession_code]["steps"]["group_tissue_labels"]["status"] = "canceled"
        return Command(update=return_dict)
    
    guessed_tissue_labels, _, harmonized_tissue_labels = await _harmonize_ontology_labels(
        metadata_cache, extraction_protocol, sample_metadata, config, ontology = "uberon", column_name = "Tissue")
    guessed_tissue_labels_path = os.path.join(harmonization_dir, "guessed_tissue_labels.json")
    harmonized_tissue_labels_path = os.path.join(harmonization_dir, "harmonized_tissue_labels.json")
    artifacts = []
    with open(guessed_tissue_labels_path, "w") as f:
        json.dump(guessed_tissue_labels.model_dump(), f, indent=2)
    with open(harmonized_tissue_labels_path, "w") as f:
        json.dump(harmonized_tissue_labels.model_dump(), f, indent=2)
    artifacts.extend([
        ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": guessed_tissue_labels_path,
            "kind": "tissue_label_guessing",
            "sha256": compute_sha256(guessed_tissue_labels_path, is_path=True),
            "bytes": os.path.getsize(guessed_tissue_labels_path),
            "created_at": datetime.now(timezone.utc).isoformat()
        }),
        ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": harmonized_tissue_labels_path,
            "kind": "tissue_label_harmonization",
            "sha256": compute_sha256(harmonized_tissue_labels_path, is_path=True),
            "bytes": os.path.getsize(harmonized_tissue_labels_path),
            "created_at": datetime.now(timezone.utc).isoformat()
        })])
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]],
        artifacts)
    
    return_dict["datasets"][accession_code]["steps"]["map_tissue_labels_to_ontology"]["status"] = "completed"
    return_dict["datasets"][accession_code]["steps"]["group_tissue_labels"]["status"] = "running"
    return_dict["messages"] = [update_harmonization_progress_tracker(state)]
    return_dict["main_messages"] = [update_harmonization_progress_tracker(state)]
    return Command(update=return_dict)

async def higher_level_tissue_mapping_node(state: HarmonizationSubgraphState, *, config: RunnableConfig):
    """
    Performs higher level tissue label mapping for the datasets in the given state and configuration. The function first checks if the tissue grouping step has already been completed for all datasets in the state. If it has, it returns a Command with an update that includes messages to update the harmonization progress tracker. If not, it identifies the accession codes for which the tissue grouping step is still running or not started, and retrieves the harmonization results for those accession codes. It then extracts the harmonized tissue labels from the harmonization results and calls the _harmonize_ontology_group_labels helper function to perform higher level tissue label mapping. The resulting group mapping is then used to construct a mapping from raw tissue labels to harmonized group labels for each dataset, which is saved as a CSV artifact. The state is updated to mark the tissue grouping step as completed and the cell type label mapping step as running for the processed accession codes, and a Command with the updated state is returned.

    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph
        config (RunnableConfig): The configuration for the runnable, which may include parameters for the harmonization process.
    
    Returns:
        Command: A Command object containing the updates to the state after performing higher level tissue label mapping
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("group_tissue_labels", state.datasets, accession_codes):
        return Command(update={
            "main_messages": [update_harmonization_progress_tracker(state)],
            "messages": [update_harmonization_progress_tracker(state)]
        })
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["group_tissue_labels"].status in {'running', 'not_started'}])
    running_tissue_mappings = [_get_harmonization_results('tissue', accession_code, state.config.artifacts) for accession_code in running_accession_codes]
    harmonized_labels = sorted(set([x.target_label for tissue_mapping in running_tissue_mappings for x in tissue_mapping.mappings if hasattr(x, 'target_label')]))
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {
            accession_code: state.datasets[accession_code].model_dump() for accession_code in running_accession_codes
        }
    }
    
    granular_to_coarse_tissue_label_mapping, _, tissue_group_harmonization_mapping = await _harmonize_ontology_group_labels(
        harmonized_labels, config, ontology = "uberon")
    proper_group_mapping = construct_raw_to_harmonized_label_mapping(
        granular_to_coarse_tissue_label_mapping,
        tissue_group_harmonization_mapping)
    return_dict["tissue_group_mapping"] = tissue_group_harmonization_mapping.model_dump()
    artifacts = []
    for accession_code in running_accession_codes:
        rows = []
        tissue_label_guessing = _get_guess_results('tissue', accession_code, state.config.artifacts)
        tissue_label_mapping = _get_harmonization_results('tissue', accession_code, state.config.artifacts)
        proper_mapping = construct_raw_to_harmonized_label_mapping(
            tissue_label_guessing, tissue_label_mapping)
        for mapping in proper_mapping.mappings:
            harmonized_group_label = next((group_mapping.target_label for group_mapping in proper_group_mapping.mappings if hasattr(mapping, "target_label") and group_mapping.source_label == mapping.target_label), None)
            rows.append({
                "original_label": mapping.source_label,
                "harmonized_label": mapping.target_label if hasattr(mapping, "target_label") else mapping.source_label,
                "harmonized_group_label": harmonized_group_label if harmonized_group_label is not None else ""
            })
        harmonization_result_df = pd.DataFrame(rows)
        harmonization_result_path = os.path.join(state.datasets[accession_code].output_dir, "tissue_harmonization_metadata.csv")
        harmonization_result_df.to_csv(harmonization_result_path, index=True)
        artifacts.append(ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": harmonization_result_path,
            "kind": "tissue_harmonization_mapping",
            "sha256": compute_sha256(harmonization_result_path, is_path=True),
            "bytes": os.path.getsize(harmonization_result_path),
            "created_at": datetime.now(timezone.utc).isoformat()
        }))
        return_dict["datasets"][accession_code]["steps"]["group_tissue_labels"]["status"] = "completed"
        return_dict["datasets"][accession_code]["steps"]["map_cell_type_labels_to_ontology"]["status"] = "running"
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]],
        artifacts)
    return_dict["messages"] = [update_harmonization_progress_tracker(state)]
    return_dict["main_messages"] = [update_harmonization_progress_tracker(state)]
    return Command(update=return_dict)

async def cell_type_harmonization_node(state: HarmonizationSubgraphState, *, config: RunnableConfig):
    """
    Performs cell type label harmonization for the datasets in the given state and configuration. The function first checks if the cell type label mapping step has already been completed for all datasets in the state. If it has, it returns a Command with an update that includes messages to update the harmonization progress tracker. If not, it identifies the accession codes for which the cell type label mapping step is still running or not started, and retrieves the harmonization results for those accession codes. It then extracts the harmonized cell type labels from the harmonization results and calls the _harmonize_ontology_labels helper function to perform cell type label harmonization. The resulting harmonized labels are saved as JSON artifacts. The state is updated to mark the cell type label mapping step as completed for the processed accession codes, and a Command with the updated state is returned.

    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph
        config (RunnableConfig): The configuration for the runnable, which may include parameters for the harmonization process.
    
    Returns:
        Command: A Command object containing the updates to the state after performing cell type label harmonization
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("map_cell_type_labels_to_ontology", state.datasets, accession_codes):
        return Command(update={
            "main_messages": [update_harmonization_progress_tracker(state)],
            "messages": [update_harmonization_progress_tracker(state)]
        })
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["map_cell_type_labels_to_ontology"].status in {'running', 'not_started'}])
    accession_code = running_accession_codes.pop(0)
    harmonization_dir = os.path.join(state.datasets[accession_code].output_dir, "harmonization")

    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {
            accession_code: state.datasets[accession_code].model_dump()
        }
    }
    metadata_cache = _get_metadata_cache(accession_code, state.config.artifacts)
    sample_metadata  = _get_sample_metadata(accession_code, state.config.artifacts)
    extraction_protocol = _get_extraction_protocol(accession_code, state.config.artifacts)
    if extraction_protocol["cell_type"]["status"] == "missing":
        return_dict["datasets"][accession_code]["steps"]["map_cell_type_labels_to_ontology"]["status"] = "canceled"
        return Command(update=return_dict)
    guessed_cell_type_labels, _, harmonized_cell_type_labels = await _harmonize_ontology_labels(
        metadata_cache, extraction_protocol, sample_metadata, config, ontology = "cl", column_name = "Cell_Type")
    guessed_cell_type_labels_path = os.path.join(harmonization_dir, "guessed_cell_type_labels.json")
    harmonized_cell_type_labels_path = os.path.join(harmonization_dir, "harmonized_cell_type_labels.json")
    artifacts = []
    with open(guessed_cell_type_labels_path, "w") as f:
        json.dump(guessed_cell_type_labels.model_dump(), f, indent=2)
    with open(harmonized_cell_type_labels_path, "w") as f:
        json.dump(harmonized_cell_type_labels.model_dump(), f, indent=2)
    artifacts.extend([
        ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": guessed_cell_type_labels_path,
            "kind": "cell_type_label_guessing",
            "sha256": compute_sha256(guessed_cell_type_labels_path, is_path=True),
            "bytes": os.path.getsize(guessed_cell_type_labels_path),
            "created_at": datetime.now(timezone.utc).isoformat()
        }),
        ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": harmonized_cell_type_labels_path,
            "kind": "cell_type_label_harmonization",
            "sha256": compute_sha256(harmonized_cell_type_labels_path, is_path=True),
            "bytes": os.path.getsize(harmonized_cell_type_labels_path),
            "created_at": datetime.now(timezone.utc).isoformat()
        })])
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]],
        artifacts)
    return_dict["datasets"][accession_code]["steps"]["map_cell_type_labels_to_ontology"]["status"] = "completed"
    return_dict["datasets"][accession_code]["steps"]["harmonize_sex_labels"]["status"] = "running"
    return_dict["messages"] = [update_harmonization_progress_tracker(state)]
    return_dict["main_messages"] = [update_harmonization_progress_tracker(state)]
    return Command(update=return_dict)

async def sex_harmonization_node(state: HarmonizationSubgraphState, *, config: RunnableConfig):
    """
    Performs sex label harmonization for the datasets in the given state and configuration. The function first checks if the sex harmonization step has already been completed for all datasets in the state. If it has, it returns a Command with an update that includes messages to update the harmonization progress tracker. If not, it identifies the accession codes for which the sex harmonization step is still running or not started, and retrieves the metadata cache, sample metadata, and extraction protocol for one of those accession codes. If any of these are missing, or if the extraction protocol indicates that sex is missing, the function updates the state to mark the sex harmonization step as canceled for that accession code and returns a Command with the updated state. If the necessary information is available, the function calls the _harmonize_sex_labels helper function to perform sex label harmonization. The resulting harmonized labels are saved as a JSON artifact, and the state is updated to mark the sex harmonization step as completed for the processed accession code. Finally, a Command with the updated state is returned.

    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph
        config (RunnableConfig): The configuration for the runnable, which may include parameters for the harmonization process.
    
    Returns:
        Command: A Command object containing the updates to the state after performing sex label harmonization
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("harmonize_sex_labels", state.datasets, accession_codes):
        return Command(update={
            "main_messages": [update_harmonization_progress_tracker(state)],
            "messages": [update_harmonization_progress_tracker(state)]
        })
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["harmonize_sex_labels"].status in {'running', 'not_started'}])
    accession_code = running_accession_codes.pop(0)
    harmonization_dir = os.path.join(state.datasets[accession_code].output_dir, "harmonization")

    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {
            accession_code: state.datasets[accession_code].model_dump()
        }
    }
    metadata_cache = _get_metadata_cache(accession_code, state.config.artifacts)
    sample_metadata  = _get_sample_metadata(accession_code, state.config.artifacts)
    extraction_protocol = _get_extraction_protocol(accession_code, state.config.artifacts)
    if extraction_protocol["sex"]["status"] == "missing":
        return_dict["datasets"][accession_code]["steps"]["harmonize_sex_labels"]["status"] = "canceled"
        return Command(update=return_dict)
    _, _, harmonized_sex_labels = await _harmonize_sex_labels(
        metadata_cache, extraction_protocol, sample_metadata, config)
    artifacts = []
    harmonized_sex_labels_path = os.path.join(harmonization_dir, "harmonized_sex_labels.json")
    with open(harmonized_sex_labels_path, "w") as f:
        json.dump(harmonized_sex_labels.model_dump(), f, indent=2)
    artifacts.append(
        ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": harmonized_sex_labels_path,
            "kind": "sex_label_harmonization",
            "sha256": compute_sha256(harmonized_sex_labels_path, is_path=True),
            "bytes": os.path.getsize(harmonized_sex_labels_path),
            "created_at": datetime.now(timezone.utc).isoformat()
        }))
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]],
        artifacts)
    return_dict["datasets"][accession_code]["steps"]["harmonize_sex_labels"]["status"] = "completed"
    return_dict["messages"] = [update_harmonization_progress_tracker(state)]
    return_dict["main_messages"] = [update_harmonization_progress_tracker(state)]
    return Command(update=return_dict)