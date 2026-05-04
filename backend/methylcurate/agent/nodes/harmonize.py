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
from ..state.models import HarmonizationIngestionConfig, HarmonizationSubgraphState
from ...tools.harmonize import (
    _harmonize_ontology_labels, _harmonize_ontology_group_labels, _harmonize_sex_labels,
    construct_raw_to_harmonized_label_mapping)
from ...utils.helper import get_accession_codes, check_step_completion, update_harmonization_progress_tracker, compute_sha256, consolidate_artifacts
from ...utils.examples import generate_metadata_harmonization_examples
def _find_artifact_by_kind(accession_code, kind, artifacts):
    """Retrieve the first ArtifactRef matching accession_code and kind.

    Args:
        accession_code: GEO accession code to match.
        kind: Artifact kind string to match.
        artifacts: List of ArtifactRef objects to search.

    Returns:
        The matching ArtifactRef, or None if not found.
    """
    for artifact in artifacts:
        if artifact.accession_code == accession_code and artifact.kind == kind:
            return artifact
    return None


def _save_artifact_pair(return_dict, accession_code, guessed, harmonized, kind_prefix, output_dir):
    """Write guessed and harmonized label sets as JSON and register ArtifactRefs.

    Args:
        return_dict: State update dict (mutated in-place with artifacts).
        accession_code: GEO accession code.
        guessed: LabelMappingSet with human-readable guesses.
        harmonized: LabelMappingSet with ontology selections.
        kind_prefix: Prefix for artifact kind strings (e.g. "disease").
        output_dir: Directory to write JSON files into.

    Returns:
        List of new ArtifactRef objects appended to return_dict["config"]["artifacts"].
    """
    harmonization_dir = os.path.join(output_dir, "harmonization")
    os.makedirs(harmonization_dir, exist_ok=True)
    guessed_path = os.path.join(harmonization_dir, f"guessed_{kind_prefix}_labels.json")
    harmonized_path = os.path.join(harmonization_dir, f"harmonized_{kind_prefix}_labels.json")
    with open(guessed_path, "w") as f:
        json.dump(guessed.model_dump(), f, indent=2)
    with open(harmonized_path, "w") as f:
        json.dump(harmonized.model_dump(), f, indent=2)
    artifacts = [
        ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": guessed_path,
            "kind": f"{kind_prefix}_label_guessing",
            "sha256": compute_sha256(guessed_path, is_path=True),
            "bytes": os.path.getsize(guessed_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }),
        ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": harmonized_path,
            "kind": f"{kind_prefix}_label_harmonization",
            "sha256": compute_sha256(harmonized_path, is_path=True),
            "bytes": os.path.getsize(harmonized_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }),
    ]
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]],
        artifacts,
    )
    return artifacts



def _get_metadata_cache(accession_code: str, artifacts: List[ArtifactRef]):
    """Return the metadata cache dict for an accession code, or None."""
    artifact = _find_artifact_by_kind(accession_code, "metadata_cache", artifacts)
    if artifact is not None:
        with open(artifact.path, "r") as f:
            return json.load(f)
    return None

def _get_sample_metadata(accession_code: str, artifacts: List[ArtifactRef]) -> Optional[pd.DataFrame]:
    """Return sample metadata DataFrame for an accession code, or None."""
    artifact = _find_artifact_by_kind(accession_code, "dataset_metadata", artifacts)
    if artifact is not None:
        return pd.read_csv(artifact.path, index_col=0)
    return None

def _get_extraction_protocol(accession_code: str, artifacts: List[ArtifactRef]):
    """Return the extraction protocol dict for an accession code, or None."""
    artifact = _find_artifact_by_kind(accession_code, "metadata_extraction_protocol", artifacts)
    if artifact is not None:
        with open(artifact.path, "r") as f:
            return json.load(f)
    return None

def _get_guess_results(target_type: str, accession_code: str, artifacts: List[ArtifactRef]) -> LabelMappingSet:
    """Return the label guessing LabelMappingSet, or raise ValueError if absent."""
    artifact = _find_artifact_by_kind(accession_code, f"{target_type}_label_guessing", artifacts)
    if artifact is not None:
        return LabelMappingSet.model_validate(json.load(open(artifact.path, "r")))
    raise ValueError(f"No harmonization results found for {target_type} and accession code {accession_code}")

def _get_harmonization_results(target_type: str, accession_code: str, artifacts: List[ArtifactRef]) -> LabelMappingSet:
    """Return the label harmonization LabelMappingSet, or raise ValueError if absent."""
    artifact = _find_artifact_by_kind(accession_code, f"{target_type}_label_harmonization", artifacts)
    if artifact is not None:
        return LabelMappingSet.model_validate(json.load(open(artifact.path, "r")))
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
    _save_artifact_pair(
        return_dict, accession_code, guessed_disease_labels,
        harmonized_disease_labels, "disease",
        state.datasets[accession_code].output_dir,
    )
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
    _save_artifact_pair(
        return_dict, accession_code, guessed_tissue_labels,
        harmonized_tissue_labels, "tissue",
        state.datasets[accession_code].output_dir,
    )
    
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
    _save_artifact_pair(
        return_dict, accession_code, guessed_cell_type_labels,
        harmonized_cell_type_labels, "cell_type",
        state.datasets[accession_code].output_dir,
    )
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