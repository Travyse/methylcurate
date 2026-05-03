import os
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, get_args, Optional
from langchain_core.messages import SystemMessage
from .models import (
    GeoIngestionSubgraphState, GEOIngestionConfig, GeoDatasetState,
    HarmonizationIngestionConfig, HarmonizationDatasetState, HarmonizationSubgraphState,
    QualityControlIngestionConfig, DatasetQualityControlState, QualityControlSubgraphState,
    BenchmarkingIngestionConfig, BenchmarkingDatasetState, BenchmarkingSubgraphState, MainState, SubgraphHandle)
from ...contracts.harmonize import Concept as HarmonizationConcepts
from ...contracts.geo import Concept as GEOConcepts
from ...contracts.geo import GEOMetadataExtractionResult
from ...utils.prompting import (
    generate_geo_system_prompt,
    generate_geo_system_concept_prompt,
    generate_router_system_prompt,
    generate_geo_metadata_harmonization_prompt)

from ...utils.examples import (
    generate_router_interpretation_examples,
    generate_router_clarification_examples,
    generate_geo_metadata_extraction_examples,
    generate_metadata_harmonization_examples,
    generate_general_geo_metadata_extraction_examples)

from langchain_core.messages import HumanMessage

def _append_user_message(state: Any, user_text: str) -> None:
    if not user_text:
        return

    # attribute-style state
    if hasattr(state, "messages") and isinstance(getattr(state, "messages"), list):
        state.messages.append(
            HumanMessage(
                id=uuid.uuid4().hex,
                content=user_text,
                additional_kwargs={
                    'created_at': datetime.now(timezone.utc).isoformat(),
                }))
        return

    # dict-style state (MessagesState / TypedDict)
    if isinstance(state, dict):
        state.setdefault("messages", []).append(HumanMessage(
            id=uuid.uuid4().hex,
            content=user_text,
            additional_kwargs={
                'created_at': datetime.now(timezone.utc).isoformat(),
            }))
        return

    # if neither works, fail loudly
    raise TypeError(f"State has no messages container: {type(state)}")

def make_geo_ingestion_state(run_id: str, params: Dict[str, Any]) -> GeoIngestionSubgraphState:
    params['output_root'] = os.path.join(params.get('output_root', 'outputs/'), "data")
    config = GEOIngestionConfig.model_validate(params)
    datasets = {}
    for accession_code in config.accessions:
        datasets[accession_code] = GeoDatasetState.model_validate({
            "accession": accession_code,
            "output_dir": os.path.join(config.output_root, accession_code)
        })

    general_system_present_examples = generate_general_geo_metadata_extraction_examples(is_missing=False)
    general_system_missing_examples = generate_general_geo_metadata_extraction_examples(is_missing=True)

    concept_examples = {
        concept: {
            "missing": generate_geo_metadata_extraction_examples(n_samples=10, concept=concept, is_missing=True),
            "present": generate_geo_metadata_extraction_examples(n_samples=10, concept=concept, is_missing=False)
        } for concept in get_args(GEOConcepts)
    }
    prompt_kwargs = {
        concept: {
            "concept": concept,
            "example_input_present": concept_examples[concept]['present'][0],
            "example_result_present": concept_examples[concept]['present'][1],
            "example_input_missing": concept_examples[concept]['missing'][0],
            "example_result_missing": concept_examples[concept]['missing'][1]
        } for concept in get_args(GEOConcepts)
    }
    general_kwargs = {
        "example_input_present": general_system_present_examples[0],
        "example_result_present": general_system_present_examples[1],
        "example_input_missing": general_system_missing_examples[0],
        "example_result_missing": general_system_missing_examples[1],
        "model_schema": json.dumps(GEOMetadataExtractionResult.model_json_schema(), indent=4)
    }

    return GeoIngestionSubgraphState(
        run_id=run_id,
        config=config,
        messages=[
            SystemMessage(content=generate_geo_system_prompt(**general_kwargs))
        ],
        llm_messages=[
            SystemMessage(content=generate_geo_system_prompt(**general_kwargs))
        ],
        concept_messages={
            concept: [
                SystemMessage(content=generate_geo_system_concept_prompt(**prompt_kwargs[concept]))
            ] for concept in get_args(GEOConcepts)
        },
        datasets=datasets)

def make_harmonization_state(run_id: str, params: Dict[str, Any]) -> HarmonizationSubgraphState:
    params['output_root'] = os.path.join(params.get('output_root', 'outputs/'), "data")
    config = HarmonizationIngestionConfig.model_validate(params)
    datasets = {}
    for accession_code in config.accessions:
        dataset_path = os.path.join(config.output_root, accession_code)
        os.makedirs(dataset_path, exist_ok=True)
        datasets[accession_code] = HarmonizationDatasetState.model_validate({
            "accession": accession_code,
            "output_dir": dataset_path
        })
        
    return HarmonizationSubgraphState(
        run_id=run_id,
        config=config,
        datasets=datasets)

def make_quality_control_state(run_id: str, params: Dict[str, Any]) -> QualityControlSubgraphState:
    params['output_root'] = os.path.join(params.get('output_root', 'outputs/'), "data")
    config = QualityControlIngestionConfig.model_validate(params)
    datasets = {}
    for accession_code in config.accessions:
        datasets[accession_code] = DatasetQualityControlState(
            accession=accession_code,
            output_dir=os.path.join(config.output_root, accession_code),
        )
    
    return QualityControlSubgraphState(
        run_id=run_id,
        config=config,
        datasets=datasets)

def make_benchmarking_state(run_id: str, params: Dict[str, Any]) -> BenchmarkingSubgraphState:
    params['output_root'] = os.path.join(params.get('output_root', 'outputs/'), "analysis")
    config = BenchmarkingIngestionConfig.model_validate(params)
    datasets = {}
    for accession_code in config.accessions:
        dataset_path = os.path.join(config.output_root, accession_code)
        os.makedirs(dataset_path, exist_ok=True)
        datasets[accession_code] = BenchmarkingDatasetState.model_validate({
            "accession": accession_code,
            "output_dir": dataset_path
        })

    return BenchmarkingSubgraphState(
        run_id=run_id,
        config=config,
        datasets=datasets)

# TODO: Later add visualizations

def register_subgraph(
    main_state: MainState,
    *,
    name: str,
    thread_id: str,
) -> SubgraphHandle:
    """
    Add a SubgraphHandle to main_state.subgraphs and return it.
    `thread_id` should be the LangGraph checkpoint thread id you will use.
    """
    handle = SubgraphHandle.model_validate({
        "name": name,
        "thread_id": thread_id,
        "status": "not_started",
        "warnings": [],
        "errors": [],
    })
    # store under the thread_id or name depending on your lookup preference.
    # Storing by thread_id avoids duplicate names across runs; storing by name
    # may be simpler to look up. Choose consistently.
    main_state.subgraphs[thread_id] = handle
    return handle

def make_subgraph_state(
    subgraph: str,
    run_id: str,
    params: Dict[str, Any],
):
    if subgraph == "geo_retrieval":
        return make_geo_ingestion_state(run_id, params)
    elif subgraph == "harmonization":
        return make_harmonization_state(run_id, params)
    elif subgraph == "quality_control":
        return make_quality_control_state(run_id, params)
    elif subgraph == "benchmarking":
        return make_benchmarking_state(run_id, params)
    else:
        raise ValueError(f"Unknown subgraph: {subgraph}")

def make_full_subgraph_state(
    subgraph: str,
    params: Dict[str, Any],
):
    if subgraph == "geo_retrieval":
        return GeoIngestionSubgraphState.model_validate(params)
    elif subgraph == "harmonization":
        return HarmonizationSubgraphState.model_validate(params)
    elif subgraph == "quality_control":
        return QualityControlSubgraphState.model_validate(params)
    elif subgraph == "benchmarking":
        return BenchmarkingSubgraphState.model_validate(params)
    else:
        raise ValueError(f"Unknown subgraph: {subgraph}")

def _build_system_directories(output_root: str) -> None:
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "data", "platforms"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "analysis"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "analysis", "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "logs"), exist_ok=True)

def make_main_state(
    *,
    run_id: str,
    default_output_root: str = "outputs/",
    user_request: Optional[str] = None,
    next_action_hint: Optional[str] = None,
) -> MainState:
    """
    Create an initial MainState for a new run.
    Keep fields explicit so checkpointing is deterministic.
    """
    output_dir = os.path.join(default_output_root, run_id)
    _build_system_directories(output_dir)
    example_input, example_output = generate_router_interpretation_examples(output_root=output_dir)
    example_initial_user_query, example_initial_agent_response, example_follow_up_user_query, example_follow_up_agent_response = generate_router_clarification_examples(output_root=output_dir)
    # Note: MainState expects NonEmptyStr for run_id; let Pydantic validate it.
    return MainState(
        run_id=run_id,
        default_output_root=output_dir,
        user_request=user_request or None,
        messages=[
            #SystemMessage(content=generate_system_prompt()),
            SystemMessage(content=generate_router_system_prompt(
                root_dir=output_dir,
                example_input=example_input,
                example_output=example_output.model_dump_json(indent=2),
                example_initial_user_query=example_initial_user_query,
                example_initial_agent_response=example_initial_agent_response.model_dump_json(indent=2),
                example_follow_up_user_query=example_follow_up_user_query,
                example_follow_up_agent_response=example_follow_up_agent_response.model_dump_json(indent=2)
            ))],
        subgraphs={},                    # no subgraphs started yet
        pending_reviews=None,              # empty list of pending review tickets
        artifacts=[],
        warnings=[],
        errors=[],
        datasets={},
        next_action_hint=next_action_hint or "Ready.",
    )


def get_dataset_for_subgraph(subgraph_name: str) -> Any:
    if subgraph_name == "geo_retrieval":
        return GeoDatasetState
    elif subgraph_name == "harmonization":
        return ConceptHarmonizationState
    elif subgraph_name == "quality_control":
        return DatasetQualityControlState
    else:
        raise ValueError(f"Unknown subgraph: {subgraph_name}")