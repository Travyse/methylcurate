__all__ = [
    "start_geo_subgraph",
    "geo_download_node",
    "check_downloads_succeeded",
    "check_platforms_used",
    "check_data_presence",
]

import os
import pandas as pd
from datetime import datetime, timezone
from ....contracts.common import ArtifactRef
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any, List
from langchain_core.messages import ToolMessage
from ....utils.helper import (
    get_accession_codes, consolidate_artifacts, set_step_status, _get_supplementary_file_id, update_progress_tracker, check_step_completion,
    read_feather)
from ....contracts.geo import GEODownloadBatchInput
from ....contracts.geo import Concept as GeoConcept
from ...state.models import GeoIngestionSubgraphState, GeoDatasetState, GEOIngestionConfig
from ....tools.geo import (
    download_geo_datasets, parallel_downloads, get_platform_metadata)

# ----------------------------
# GEO Download Contracts
# ----------------------------

def start_geo_subgraph(state: GeoIngestionSubgraphState, *, config: RunnableConfig) -> Dict[str, Any]:
    return_dict = {}
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    return Command(update=return_dict)

def geo_download_node(state: GeoIngestionSubgraphState, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    Goal is to download GEO datasets based on accession codes in the state. Return dict should update the config and datasets.
    """
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {k: v.model_dump() for k, v in state.datasets.items()}
    }
    accession_codes = get_accession_codes(state)
    download_completion = check_step_completion("download_soft", state.datasets, accession_codes)
    valid_check_completion = check_step_completion("check_valid_dataset", state.datasets, accession_codes)
    completed = download_completion and valid_check_completion
    running_accession_codes = [x for x in accession_codes if ((not state.datasets.get(x, False)) and (state.datasets[x].steps['download_soft'].status != 'completed') and (state.datasets[x].steps['check_valid_dataset'].status != 'completed')) or (state.datasets[x].steps["download_soft"].status == "not_started")]
    if completed or len(running_accession_codes) == 0:
        return Command(update={
            "main_messages": [update_progress_tracker(state)],
            "messages": [update_progress_tracker(state)]
        })
    
    # Grab the first 5 accession codes to download in this batch (if there are more than 5, the rest will be picked up in the next iteration after we update the state with results and statuses)
    running_accession_codes = sorted(running_accession_codes)[:min(5, len(running_accession_codes))]
    
    download_inputs = [
        {
            "accession": accession_code,
            "output_root": os.path.join(state.config.output_root, accession_code)
        } for accession_code in running_accession_codes
    ]
    batch_download_input = GEODownloadBatchInput.model_validate({
        "geo_downloads": download_inputs
    })

    artifacts, supplementary_files, batch_result = download_geo_datasets(state.config, batch_download_input)
    supplementary_files_dict = {}
    for supp_file in supplementary_files:
        supplementary_files_dict.update(supp_file)

    for accession_code in running_accession_codes:
        dataset_result = next((res for res in batch_result.results if res.accession == accession_code), None)
        has_succeeded = dataset_result and dataset_result.status == "success"
        current_status = "completed" if has_succeeded else "failed"
        next_status = "running" if has_succeeded else "canceled"
        other_status = "not_started" if has_succeeded else "canceled"
        platform_metadata = get_platform_metadata(accession_code=accession_code) if has_succeeded else None
        return_dict["datasets"][accession_code] = GeoDatasetState(
            status="in_progress",
            steps={
                "download_soft": set_step_status(status=current_status) ,
                "check_valid_dataset": set_step_status(status=next_status),
                "extract_metadata_schema": set_step_status(status=other_status),
                "refine_metadata_schema": set_step_status(status=other_status),
                "extract_data": set_step_status(status=other_status),
                "supplementary_file_check": set_step_status(status=other_status)
            },
            accession=accession_code,
            output_dir=os.path.join(state.config.output_root, accession_code),
            platform_metadata=platform_metadata,
            supplementary_files=supplementary_files_dict.get(accession_code, []),
            download_result=next((res for res in batch_result.results if res.accession == accession_code), None)
        )

    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef.model_validate(x) for x in return_dict["config"]["artifacts"]],
        artifacts)
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    return Command(update=return_dict)

def _check_downloads_succeeded(return_dict: Dict[str, Any]) -> Dict[str, Any]:
    accession_code = list(return_dict["datasets"].keys())
    if len(accession_code) != 1:
        raise ValueError(f"Expected exactly one accession code in state for download check, but found {len(accession_code)}: {accession_code}")
    accession_code = accession_code[0]
    
    dataset_state = return_dict["datasets"][accession_code]
    if dataset_state["steps"]["check_valid_dataset"]["status"] == "failed":
        return return_dict
    
    download_result = dataset_state["download_result"]
    if download_result["status"] == 'failed':
        return_dict["datasets"][accession_code]["status"] = "failed"
        return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"] = set_step_status(status="failed", step=return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"])
        return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"])
        return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"])
        return_dict["datasets"][accession_code]["steps"]["extract_data"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["extract_data"])
        return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"])
        return_dict["datasets"][accession_code]["errors"].append(download_result.error) # TODO: Check to see if dataset doesn't exist vs other kinda error

    return return_dict

def check_downloads_succeeded(state: GeoIngestionSubgraphState, *, config: RunnableConfig) -> Dict[str, Any]:
    accession_codes = get_accession_codes(state)
    unstarted_accession_codes = sorted([accession_code for accession_code in accession_codes if not state.datasets.get(accession_code, False)])
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if (accession_code not in unstarted_accession_codes) and (state.datasets[accession_code].steps["check_valid_dataset"].status == "running")])
    current_accession_code = running_accession_codes.pop(0)
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {
            current_accession_code: state.datasets[current_accession_code].model_dump()
        }
    }
    return_dict = _check_downloads_succeeded(return_dict)
    return_dict["messages"] = [update_progress_tracker(state)]
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return Command(update=return_dict)

def _check_is_methylation_dataset(platform_metadata: Dict[str, any] = {}):
    if not platform_metadata:
        return False
    gpl_whitelist = [
        "GPL29753", "GPL33022", "GPL21145", "GPL18809", "GPL8490"]
    platform_titles = platform_metadata.get("title", "")
    if any("methylation" in pt.lower() or "methy" in pt.lower() for pt in platform_titles.split(",")):
        return True
    gpls = platform_metadata.get("platform_id", [])
    if any(gpl in gpl_whitelist for gpl in gpls):
        return True
    return False

def _check_platforms_used(return_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Check that each GEO dataset is a DNA methylation dataset
    for accession_code in return_dict["datasets"].keys():
        platform_metadata = return_dict["datasets"][accession_code].get("platform_metadata", {})
        is_dnam_dataset = _check_is_methylation_dataset(platform_metadata)
        if not is_dnam_dataset or return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"]["status"] == "failed":
            return_dict["datasets"][accession_code]["status"] = "failed"
            return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"] = set_step_status(status="failed", step=return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"])
            return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"])
            return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"])
            return_dict["datasets"][accession_code]["steps"]["extract_data"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["extract_data"])
            return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"])
            return_dict["datasets"][accession_code]["is_valid_dataset"] = False
            return_dict["datasets"][accession_code]["errors"].append(f"Dataset {accession_code} does not appear to be a DNA methylation dataset. As of now, we only support DNA methylation datasets from GEO.")
            
    return return_dict

def check_platforms_used(state: GeoIngestionSubgraphState, *, config: RunnableConfig) -> Dict[str, Any]:
    accession_codes = get_accession_codes(state)
    unstarted_accession_codes = sorted([accession_code for accession_code in accession_codes if not state.datasets.get(accession_code, False)])
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if (accession_code not in unstarted_accession_codes) and (state.datasets[accession_code].steps["check_valid_dataset"].status == "running")])
    current_accession_code = running_accession_codes.pop(0)
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {
            current_accession_code: state.datasets[current_accession_code].model_dump()
        }
    }
    return_dict = _check_platforms_used(return_dict)
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    return Command(update=return_dict)

def _check_if_data_present(artifacts: List[Any], return_dict: Dict[str, Any]) -> bool:
    accession_code = list(return_dict["datasets"].keys())[0]
    methylation_data_path = next((a for a in artifacts if a.accession_code == accession_code and a.kind == "preqc_methylation_data"), None)
    methylation_data = read_feather(methylation_data_path.path, index_name="subject_id")
    if methylation_data.empty:
        supplementary_files = return_dict["datasets"][accession_code].get("supplementary_files", [])
        if supplementary_files:
            hash = _get_supplementary_file_id(supplementary_files, accession_code)
            selection = interrupt(ToolMessage(
                id=hash,
                content="This GEO dataset requires you to select the datasets. Here are some files below. We suggest selecting the datasets that look like they are processed (often will have Beta or Proc in the file name, will be .txt, .csv, or .tsv files.)",
                tool_call_id=hash,
                artifact={
                    "id": hash,
                    "options": [
                        {
                            "id": os.path.basename(supplementary_file),
                            "label": os.path.basename(supplementary_file)
                        } for supplementary_file in supplementary_files
                    ],
                    "actions": [
                        {
                            "id": "skip",
                            "label": "Skip this dataset"
                        },
                        {
                            "id": "confirm",
                            "label": "Confirm selection"
                        }
                    ]
                },
                additional_kwargs={
                    "name": "geoSupplementaryFileSelection",
                    'created_at': datetime.now(timezone.utc).isoformat(),
                }
            ).model_dump())

            if selection["data"]["action"] == "skip":
                return_dict["datasets"][accession_code]["status"] = "completed"
                return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"] = set_step_status(status="completed", step=return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"])
                return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"])
                return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"])
                return_dict["datasets"][accession_code]["steps"]["extract_data"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["extract_data"])
                return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"])
                return_dict["datasets"][accession_code]["errors"].append(f"Dataset {accession_code} does not appear to have any GSM samples with data tables, and user chose to skip after being prompted with available supplementary files.")
            else:
                selections = [s for s in supplementary_files if any(y in s for y in selection["data"]["data"]["selections"])]
                artifacts = parallel_downloads(
                    accession_code, selections, return_dict["datasets"][accession_code]["output_dir"])
                return_dict["config"]["artifacts"] = consolidate_artifacts(
                    [ArtifactRef.model_validate(x) for x in return_dict["config"]["artifacts"]],
                    artifacts["artifacts"])
                return_dict["datasets"][accession_code]["supplementary_data"] = {
                    artifact.sha256: "pending" for artifact in artifacts["artifacts"]
                } if return_dict["datasets"][accession_code].get("supplementary_data", None) is None else return_dict["datasets"][accession_code]["supplementary_data"]

                return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"] = set_step_status(status="completed", step=return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"])
                return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"] = set_step_status(status="running", step=return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"])
                return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"] = set_step_status(status="not_started", step=return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"])
                return_dict["datasets"][accession_code]["steps"]["extract_data"] = set_step_status(status="not_started", step=return_dict["datasets"][accession_code]["steps"]["extract_data"])
                return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"] = set_step_status(status="not_started", step=return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"])
                return_dict["datasets"][accession_code]["is_valid_dataset"] = True

        else:
            return_dict["datasets"][accession_code]["status"] = "completed"
            return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"] = set_step_status(status="failed", step=return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"])
            return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"])
            return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"])
            return_dict["datasets"][accession_code]["steps"]["extract_data"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["extract_data"])
            return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"] = set_step_status(status="canceled", step=return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"])
            return_dict["datasets"][accession_code]["errors"].append(f"Dataset {accession_code} does not appear to have any GSM samples with data tables.")
    else:
        return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"] = set_step_status(status="completed", step=return_dict["datasets"][accession_code]["steps"]["check_valid_dataset"])
        return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"] = set_step_status(status="running", step=return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"])
        return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"] = set_step_status(status="not_started", step=return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"])
        return_dict["datasets"][accession_code]["steps"]["extract_data"] = set_step_status(status="not_started", step=return_dict["datasets"][accession_code]["steps"]["extract_data"])
        return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"] = set_step_status(status="not_started", step=return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"])
        return_dict["datasets"][accession_code]["is_valid_dataset"] = True
    return return_dict

async def check_data_presence(state: GeoIngestionSubgraphState, *, config: RunnableConfig) -> Dict[str, Any]:
    accession_codes = get_accession_codes(state)
    download_completion = check_step_completion("download_soft", state.datasets, accession_codes)
    valid_check_completion = check_step_completion("check_valid_dataset", state.datasets, accession_codes)
    completed = download_completion and valid_check_completion
    if completed:
        return Command(update={}, goto="download_soft_file")
    unstarted_accession_codes = sorted([accession_code for accession_code in accession_codes if not state.datasets.get(accession_code, False)])
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if (accession_code not in unstarted_accession_codes) and (state.datasets[accession_code].steps["check_valid_dataset"].status == "running")])
    current_accession_code = running_accession_codes.pop(0)
    artifacts = [a for a in state.config.artifacts if a.kind == "preqc_methylation_data" and a.accession_code == current_accession_code]
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {
            current_accession_code: state.datasets[current_accession_code].model_dump()
        }
    }
    return_dict = _check_if_data_present(artifacts, return_dict)
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    return Command(update=return_dict)
