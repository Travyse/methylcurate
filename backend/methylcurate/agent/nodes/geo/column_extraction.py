__all__ = [
    "extract_metadata_schema",
    "check_column_extraction_rule_formatting",
    "check_column_extraction_rule_accuracy",
    "geo_metadata_column_extraction_approval_node",
]

import os
import json
import random
import uuid
import pandas as pd
from datetime import datetime, timezone
from ....contracts.common import ArtifactRef
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from typing import Any, List, Dict, get_args
from langchain_core.messages import ToolMessage, HumanMessage
from ....utils.helper import (
    get_accession_codes,
    compute_sha256,
    set_step_status,
    update_progress_tracker,
    check_step_completion,
    consolidate_artifacts,
)
from ....utils.prompting import generate_metadata_column_user_query_alt
from ....contracts.geo import GEOMetadataExtractionInput, Concept
from ...state.models import GeoIngestionSubgraphState, GeoDatasetState
from ....tools.geo import (
    extract_metadata_columns_alt,
    _get_custom_models,
    _extract_all_columns,
    _get_extraction_resolutions,
    _check_extraction_patterns,
    _extract_column_for_concept_misformatted,
    _extract_column_for_concept_poor_parsing,
    _get_parse_rate,
    _extract_column_for_concept_disease_status,
    _extract_column_for_concept_age,
    extract_dataset_metadata,
    generate_summary_data,
)

MAX_RETRIES = 3
# ----------------------------
# GEO Column Retrieval
# ----------------------------


def _randomly_sample_from_dataset(
    metadata_dict: Dict[str, Any],
    return_dict: Dict[str, Any],
    accession_code: str,
    artifact: ArtifactRef,
    num_samples: int = 10,
) -> List[Any]:
    extraction_examples = {
        "accession_code": accession_code,
        "artifact": artifact,
        "title": [],
        "source_name_ch1": [],
        "description": [],
        "characteristics_ch1": [],
        "relation": [],
        "platform_id": [],
    }

    gsm_metadata_dict = metadata_dict["sample_metadata"]

    gsm_name_subset = random.sample(list(gsm_metadata_dict.keys()), min(num_samples, len(gsm_metadata_dict)))
    for gsm_name in gsm_name_subset:
        gsm = gsm_metadata_dict[gsm_name]
        extraction_examples["title"].append(gsm.get("title", []))
        extraction_examples["source_name_ch1"].append(gsm.get("source_name_ch1", []))
        extraction_examples["description"].append(gsm.get("description", []))
        ch1_data = {x.split(":")[0].strip(): x.split(":")[1].strip() for x in gsm.get("characteristics_ch1", [])}
        extraction_examples["characteristics_ch1"].append(ch1_data)
        if "relation" in gsm:
            extraction_examples["relation"].append(gsm.get("relation", []))
        if "platform_id" in gsm:
            extraction_examples["platform_id"].append(gsm.get("platform_id", []))

    gsm_extraction_input = GEOMetadataExtractionInput.model_validate(extraction_examples)
    return_dict["datasets"][accession_code]["metadata_extraction_input"] = gsm_extraction_input.model_dump()
    return return_dict


async def extract_metadata_schema(state: GeoIngestionSubgraphState, *, config: RunnableConfig) -> Dict[str, Any]:
    accession_codes = get_accession_codes(state)
    if check_step_completion("extract_metadata_schema", state.datasets, accession_codes):
        return Command(
            update={"main_messages": [update_progress_tracker(state)], "messages": [update_progress_tracker(state)]}
        )
    running_accession_codes = sorted(
        [
            accession_code
            for accession_code in accession_codes
            if state.datasets[accession_code].steps["extract_metadata_schema"].status == "running"
        ]
    )

    accession_code = running_accession_codes.pop(0)
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {accession_code: state.datasets[accession_code].model_dump()},
        "llm_messages": [],
    }
    artifact = next(a for a in state.config.artifacts if a.accession_code == accession_code and a.kind == "soft_file")
    metadata_artifact = next(
        a for a in state.config.artifacts if a.accession_code == accession_code and a.kind == "metadata_cache"
    )
    metadata = None
    with open(metadata_artifact.path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    dataset_dict = {
        "dataset_title": metadata["dataset_metadata"]["title"],
        "dataset_summary": metadata["dataset_metadata"]["summary"],
        "dataset_overall_design": metadata["dataset_metadata"]["overall_design"],
    }
    return_dict = _randomly_sample_from_dataset(
        metadata, return_dict, accession_code, artifact=artifact, num_samples=10
    )
    user_input = GEOMetadataExtractionInput(**return_dict["datasets"][accession_code]["metadata_extraction_input"])
    GSESpecificMetadataExtractionResult, _, __, key_names = _get_custom_models(user_input)

    key_names_str = ", ".join(sorted(list(key_names)))
    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=generate_metadata_column_user_query_alt(
            user_input=user_input.model_dump(),
            key_names=key_names_str,
            json_schema=GSESpecificMetadataExtractionResult.model_json_schema(),
            **dataset_dict,
        ),
        additional_kwargs={
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return_dict["llm_messages"].append(human_message)
    messages = [state.llm_messages[0], return_dict["llm_messages"][-1]]

    extraction_result = await _extract_all_columns(messages, config, GSESpecificMetadataExtractionResult)

    artifact_path = os.path.join(state.datasets[accession_code].output_dir, "extraction_protocol.json")
    resolutions = []
    resolutions = {key: getattr(extraction_result, key) for key in get_args(Concept) if hasattr(extraction_result, key)}

    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(
            {concept: resolution.model_dump() for concept, resolution in resolutions.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    artifact = ArtifactRef.model_validate(
        {
            "path": artifact_path,
            "kind": "metadata_extraction_protocol",
            "accession_code": accession_code,
            "sha256": compute_sha256(artifact_path, is_path=True),
            "bytes": os.path.getsize(artifact_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    extraction_result.artifact = artifact
    return_dict["datasets"][accession_code]["metadata_extraction_result"] = extraction_result.model_dump()
    return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"] = set_step_status(
        status="completed", step=return_dict["datasets"][accession_code]["steps"]["extract_metadata_schema"]
    )
    return_dict["datasets"][accession_code]["steps"]["extract_data"] = set_step_status(
        status="running", step=return_dict["datasets"][accession_code]["steps"]["extract_data"]
    )
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    return_dict["config"]["artifacts"] = consolidate_artifacts(
        [ArtifactRef(**a) for a in return_dict["config"]["artifacts"]], [artifact]
    )
    return Command(update=return_dict)


async def check_column_extraction_rule_formatting(
    state: GeoIngestionSubgraphState, *, config: RunnableConfig
) -> Dict[str, Any]:
    accession_codes = get_accession_codes(state)
    if check_step_completion("refine_metadata_schema", state.datasets, accession_codes):
        return Command(
            update={"main_messages": [update_progress_tracker(state)], "messages": [update_progress_tracker(state)]}
        )
    running_accession_codes = sorted(
        [
            accession_code
            for accession_code in accession_codes
            if state.datasets[accession_code].steps["refine_metadata_schema"].status == "running"
        ]
    )

    accession_code = running_accession_codes.pop(0)
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {accession_code: state.datasets[accession_code].model_dump()},
    }

    dataset_state = state.datasets[accession_code]
    user_input = state.datasets[accession_code].metadata_extraction_input
    extraction_result = state.datasets[accession_code].metadata_extraction_result

    _, resolution_model, resolution_model_envelope, __ = _get_custom_models(user_input)
    resolutions = _get_extraction_resolutions(extraction_result)
    flagged_concepts, flagged_concept_notes = _check_extraction_patterns(resolutions)
    for concept in flagged_concepts:
        resolutions[concept].notes.append(flagged_concept_notes[concept])
        resolutions[concept].units = None if concept != "age" else resolutions[concept].units

    if len(flagged_concepts) > 0:
        new_resolutions = await _extract_column_for_concept_misformatted(
            flagged_concepts, state.llm_messages, config, resolution_model, resolutions, resolution_model_envelope
        )
        resolutions.update(new_resolutions)

        for concept in new_resolutions.keys():
            setattr(extraction_result, concept, new_resolutions[concept])

    return_dict["datasets"][accession_code]["metadata_extraction_result"] = extraction_result.model_dump()
    return_dict["datasets"][accession_code]["refinement_history"]["formatting_history"].append(flagged_concepts)

    metadata_cache_artifact = next(
        (
            artifact
            for artifact in state.config.artifacts
            if (artifact.kind == "metadata_cache") and (artifact.accession_code == accession_code)
        ),
        None,
    )
    with open(metadata_cache_artifact.path, "r", encoding="utf-8") as f:
        metadata_dict = json.load(f)
    metadata_artifact = next(
        (
            artifact
            for artifact in state.config.artifacts
            if (artifact.kind == "dataset_metadata") and (artifact.accession_code == accession_code)
        ),
        None,
    )
    metadata = pd.read_csv(metadata_artifact.path, index_col=0)

    return_dict = extract_dataset_metadata(
        accession_code,
        state.config,
        metadata_dict,
        dataset_state.metadata_extraction_result,
        True,
        gpls=[dataset_state.platform_metadata.platform_id],
        platform=[dataset_state.platform_metadata.title],
        return_dict=return_dict,
    )
    return_dict.pop("raw_disease_statuses")
    return_dict = generate_summary_data(
        metadata,
        accession_code,
        [dataset_state.platform_metadata.platform_id],
        [dataset_state.platform_metadata.title],
        dataset_state.refinement_history.example_errors,
        return_dict,
    )

    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    return Command(update=return_dict)


async def check_column_extraction_rule_accuracy(
    state: GeoIngestionSubgraphState, *, config: RunnableConfig
) -> Dict[str, Any]:
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted(
        [
            accession_code
            for accession_code in accession_codes
            if state.datasets[accession_code].steps["refine_metadata_schema"].status == "running"
        ]
    )
    accession_code = running_accession_codes.pop(0)
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {accession_code: state.datasets[accession_code].model_dump()},
    }

    metadata_artifact = next(
        a for a in state.config.artifacts if a.accession_code == accession_code and a.kind == "metadata_cache"
    )
    metadata = None
    with open(metadata_artifact.path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    dataset_state = state.datasets[accession_code]
    user_input = dataset_state.metadata_extraction_input
    extraction_result = dataset_state.metadata_extraction_result
    _, resolution_model, resolution_model_envelope, __ = _get_custom_models(user_input)
    parse_rates = _get_parse_rate(state.datasets[accession_code].metadata_summary.model_dump())
    resolutions = _get_extraction_resolutions(extraction_result)
    failed_parsing_info = (
        return_dict["datasets"][accession_code]["refinement_history"]["example_errors"][-1]
        if len(return_dict["datasets"][accession_code]["refinement_history"]["example_errors"]) > 0
        else {}
    )
    flagged_concepts = [
        c
        for c in get_args(Concept)
        if (extraction_result.model_dump()[c]["status"] == "resolved")
        and (c in parse_rates)
        and (parse_rates[c] > 0.0 and parse_rates[c] < 1.0)
        and len(failed_parsing_info[c]) > 0
    ]

    if len(flagged_concepts) > 0:
        return_dict = extract_dataset_metadata(
            accession_code,
            state.config,
            metadata,
            extraction_result,
            overwrite_artifact=True,
            gpls=[state.datasets[accession_code].platform_metadata.platform_id],
            platform=[state.datasets[accession_code].platform_metadata.title],
            return_dict=return_dict,
        )
        parsed_disease_statuses = return_dict.pop("raw_disease_statuses", None)

        # If there are misformatted concept patterns  attempt to return these patterns
        new_resolutions = await _extract_column_for_concept_poor_parsing(
            flagged_concepts,
            state.llm_messages,
            config,
            resolution_model,
            parse_rates,
            user_input,
            resolutions,
            resolution_model_envelope,
            failed_parsing_info,
        )
        resolutions.update(new_resolutions)

        if resolutions["disease_status"].status == "resolved" and parsed_disease_statuses:
            resolutions = await _extract_column_for_concept_disease_status(resolutions, config, parsed_disease_statuses)

        if resolutions["age"].status == "missing":
            resolutions = await _extract_column_for_concept_age(
                resolutions, config, user_input, state.llm_messages, resolution_model
            )

        for concept in new_resolutions.keys():
            setattr(extraction_result, concept, new_resolutions[concept])

    return_dict["datasets"][accession_code]["metadata_extraction_result"] = extraction_result.model_dump()
    return_dict["datasets"][accession_code]["refinement_history"]["num_retries"] += 1
    return_dict["datasets"][accession_code]["refinement_history"]["parsing_history"].append(flagged_concepts)

    metadata_cache_artifact = next(
        (
            artifact
            for artifact in state.config.artifacts
            if (artifact.kind == "metadata_cache") and (artifact.accession_code == accession_code)
        ),
        None,
    )
    with open(metadata_cache_artifact.path, "r", encoding="utf-8") as f:
        metadata_dict = json.load(f)
    metadata_artifact = next(
        (
            artifact
            for artifact in state.config.artifacts
            if (artifact.kind == "dataset_metadata") and (artifact.accession_code == accession_code)
        ),
        None,
    )
    metadata = pd.read_csv(metadata_artifact.path, index_col=0)

    return_dict = extract_dataset_metadata(
        accession_code,
        state.config,
        metadata_dict,
        dataset_state.metadata_extraction_result,
        True,
        gpls=[dataset_state.platform_metadata.platform_id],
        platform=[dataset_state.platform_metadata.title],
        return_dict=return_dict,
    )
    return_dict.pop("raw_disease_statuses")
    return_dict = generate_summary_data(
        metadata,
        accession_code,
        [dataset_state.platform_metadata.platform_id],
        [dataset_state.platform_metadata.title],
        dataset_state.refinement_history.example_errors,
        return_dict,
    )

    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]
    return Command(update=return_dict)


def geo_metadata_column_extraction_approval_node(
    state: GeoIngestionSubgraphState, *, config: RunnableConfig
) -> Command:
    """
    If any of the metadata extractions have concepts that need review, route to HITL. Return updated state based on HITL decisions.
    """
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted(
        [
            accession_code
            for accession_code in accession_codes
            if state.datasets[accession_code].steps["refine_metadata_schema"].status == "running"
        ]
    )
    running_accession_codes = [
        x for x in running_accession_codes if state.datasets[x].metadata_extraction_result is not None
    ]

    accession_code = running_accession_codes[0]
    return_dict = {
        "config": state.config.model_dump(),
        "datasets": {accession_code: state.datasets[accession_code].model_dump()},
    }
    artifact_path = os.path.join(state.config.output_root, accession_code, "extraction_protocol.json")
    metadata_extraction_result = return_dict["datasets"][accession_code].get("metadata_extraction_result")
    resolutions = {
        key: metadata_extraction_result[key] for key in get_args(Concept) if key in metadata_extraction_result
    }
    return_dict["main_messages"] = [update_progress_tracker(state)]
    return_dict["messages"] = [update_progress_tracker(state)]

    exceeded_retries = state.datasets[accession_code].refinement_history.num_retries >= MAX_RETRIES
    has_formatting_issues = len(state.datasets[accession_code].refinement_history.formatting_history[-1]) > 0
    has_parsing_issues = len(state.datasets[accession_code].refinement_history.parsing_history[-1]) > 0

    if exceeded_retries or (not has_formatting_issues and not has_parsing_issues):
        if all((r["status"] in ["resolved", "missing"]) and (r["confidence"] > 0.6) for r in resolutions.values()):
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(resolutions, f, ensure_ascii=False, indent=2)
            return_dict["datasets"][accession_code]["metadata_extraction_result"]["artifact"] = (
                ArtifactRef.model_validate(
                    {
                        "path": artifact_path,
                        "kind": "metadata_extraction_protocol",
                        "accession_code": accession_code,
                        "sha256": compute_sha256(artifact_path, is_path=True),
                        "bytes": os.path.getsize(artifact_path),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                ).model_dump()
            )
        return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"] = set_step_status(
            status="completed", step=return_dict["datasets"][accession_code]["steps"]["refine_metadata_schema"]
        )

        supplementary_file_artifacts = sorted(
            [
                artifact
                for artifact in state.config.artifacts
                if (artifact.kind == "supplementary_file_methylation_data")
                and (artifact.accession_code == accession_code)
            ],
            key=lambda artifact: artifact.path,
        )
        next_status = "running" if len(supplementary_file_artifacts) > 0 else "completed"
        return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"] = set_step_status(
            status=next_status, step=return_dict["datasets"][accession_code]["steps"]["supplementary_file_check"]
        )
    return Command(update=return_dict)
