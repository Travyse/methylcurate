__all__ = [
    "extract_metadata_columns",  "_extract_column_for_concept",
    "extract_metadata_columns_alt", "_extract_column_for_concept_with_retry", "_get_parse_rate", "_get_extraction_resolutions",
    "_get_custom_models", "_extract_all_columns", "_check_extraction_patterns", "_extract_column_for_concept_misformatted",
    "_extract_column_for_concept_poor_parsing", "_get_parse_rate", "_extract_column_for_concept_disease_status", "_extract_column_for_concept_age"]
import os
import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, get_args
from greenery import parse
from greenery.rxelems import Pattern

from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, AnyMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from ...agent.graphs.deps import Deps
from ...contracts.common import ArtifactRef
from ...contracts.geo import (
    GEOMetadataExtractionInput, GEOMetadataExtractionResult, ErrorResolution, FieldResolutionEnvelope,
    FieldResolution, Concept, build_dynamic_result_model, build_dynamic_resolution_correction_model,
    build_dynamic_control_identification_model)
from ...utils.helper import compute_sha256
from ...utils.prompting import (
    generate_metadata_column_user_query,
    generate_metadata_column_user_query_alt,
    generate_immediate_column_feedback,
    generate_immediate_single_column_feedback,
    generate_column_feedback_loop_prompt,
    generate_identify_control_value_prompt,
    generate_missing_age_check_prompt
)

CALL_TIMEOUT = 180
GLOBAL_RETRY_LIMIT = 5

async def get_all_extraction_resolutions():
    pass

def has_alternation_anywhere(regex: str) -> bool:
    root = parse(regex)

    def walk_pattern(p: Pattern) -> bool:
        # This Pattern node itself is an alternation if it has >1 branch (Conc)
        if len(p.concs) > 1:
            return True

        # Recurse into nested subpatterns inside Mults
        for conc in p.concs:
            for mult in conc.mults:
                if isinstance(mult.multiplicand, Pattern):
                    if walk_pattern(mult.multiplicand):
                        return True
        return False

    return walk_pattern(root)

def _get_custom_models(user_input: GEOMetadataExtractionInput):
    key_names = set()
    for example in user_input.characteristics_ch1:
        key_names.update(v.split(":", 1)[0].strip() for v in example)
    GEOMetadataExtractionResultDyn, FieldResolutionDyn, FieldResolutionEnvelopeDyn = build_dynamic_result_model(tuple(sorted(list(key_names))))
    return GEOMetadataExtractionResultDyn, FieldResolutionDyn, FieldResolutionEnvelopeDyn, key_names

def _get_parse_rate(metadata_summary: Dict[str, Any] = None) -> Dict[str, float]:
    parse_rates = {}
    if metadata_summary is not None:
        parse_rates = {
            concept: metadata_summary[concept]['parse_rate'] for concept in get_args(Concept) if not isinstance(metadata_summary[concept], list)
        }
    return parse_rates

def _get_extraction_resolutions(extraction_result: Any) -> Dict[Concept, Any]:
    resolutions = {}
    for concept in get_args(Concept):
        concept_presence = (concept in extraction_result) if isinstance(extraction_result, dict) else hasattr(extraction_result, concept)
        if concept_presence:
            resolution = extraction_result.get(concept, None) if isinstance(extraction_result, dict) else getattr(extraction_result, concept, None)
            resolutions[concept] = FieldResolutionEnvelope(resolution=resolution).resolution
    return resolutions

def _check_extraction_patterns(resolutions: Dict[str, Any]) -> List[Concept]:
    flagged_patterns = []
    flexible_concepts = ["sex", "disease_status", "tissue", "cell_type"]
    notes_flagged_patterns = {}
    for concept, resolution in resolutions.items():
        if resolution.status == "resolved" and resolution.extraction.field_name == "characteristics_ch1":
            key_name = resolution.extraction.key_name
            pattern = resolution.extraction.pattern
            if key_name.lower() in pattern.lower():
                flagged_patterns.append(concept)
                notes_flagged_patterns[concept] = f"Key name '{key_name}' found in pattern '{pattern}'"
            #if concept in flexible_concepts:
            #    if has_alternation_anywhere(pattern):
            #        flagged_patterns.append(concept)
            #        notes_flagged_patterns[concept] = f"The regular expression pattern '{pattern}' contains alternations, which violates the generic pattern requirement. Modify this pattern to remove all alternations and be simpler and more generic."
    return flagged_patterns, notes_flagged_patterns

async def _extract_column_for_concept_age(
    resolutions: Dict[str, Any], config: RunnableConfig, user_input: GEOMetadataExtractionInput,
    messages: List[AnyMessage], resolution_model: Any) -> Dict[str, Any]:
    deps: Deps = config["configurable"]["deps"]
    deterministic_llm = deps.deterministic_llm
    default_llm = deps.default_llm
    clarification_message = AIMessage(
        id=uuid.uuid4().hex,
        content=generate_missing_age_check_prompt(
            user_input=user_input),
        additional_kwargs={
            'created_at': datetime.now(timezone.utc).isoformat(),
    })
    prompt_messages = messages + [clarification_message]
    FieldResolutionCorrectionDyn = build_dynamic_resolution_correction_model(
        ["age"], resolution_model)
    try:
        resolved: Any = await deterministic_llm.acall_structured(prompt_messages, FieldResolutionCorrectionDyn)
        resolutions['age'] = resolved.age
    except ValidationError as e:
        print(f"\n\nValidation error for concept age: {e}. Setting resolution to error with notes.")
            
    return resolutions

async def _extract_column_for_concept_disease_status(
    resolutions: Dict[str, Any], config: RunnableConfig, parsed_disease_statuses: List[str]) -> Dict[str, Any]:
    deps: Deps = config["configurable"]["deps"]
    deterministic_llm = deps.deterministic_llm
    default_llm = deps.default_llm
    if resolutions["disease_status"].extraction.field_name == "default" or not hasattr(resolutions["disease_status"].extraction, "key_name"):
        key_name = "N/A"
    else:
        key_name = resolutions['disease_status'].extraction.key_name
    ControlIdentificationModel = build_dynamic_control_identification_model(parsed_disease_statuses)
    clarification_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=generate_identify_control_value_prompt(
            disease_statuses=parsed_disease_statuses,
            key_name=key_name,
            json_format= json.dumps(
                ControlIdentificationModel.model_json_schema(), indent=2)),
        additional_kwargs={
            'created_at': datetime.now(timezone.utc).isoformat(),
    })

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    call_messages = [clarification_message]
    while retries < retry_limit:
        try:
            resolved: Any = await asyncio.wait_for(
                deterministic_llm.acall_structured(call_messages, ControlIdentificationModel), timeout=CALL_TIMEOUT)
            resolutions['disease_status'].extraction.control_value = resolved.control_value
            break
        except asyncio.TimeoutError:
            retries += 1
            continue
        except OutputParserException as e:
            human_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=f"The previous output from the LLM failed to parse with error: {e}. Please reformat the output to match the expected format and ensure that all required fields are included.",
                additional_kwargs={
                    'created_at': datetime.now(timezone.utc).isoformat(),
                })
            call_messages += [human_message]
            continue
        except ValidationError as e:
            print(f"\n\nValidation error for concept disease_status: {e}. Setting resolution to error with notes.")
            break
        except Exception as e:
            retries += 1
            continue
            
    return resolutions


async def _extract_column_for_concept_misformatted(
        misformatted_concepts: List[Concept], messages: List[AnyMessage], config: RunnableConfig,
        resolution_model: Any, resolutions: Dict[str, Any], resolution_model_envelope: Any) -> Dict[Concept, FieldResolution]:
    print(f"\n\nMisformatted concepts: {misformatted_concepts}")
    deps: Deps = config["configurable"]["deps"]
    deterministic_llm = deps.deterministic_llm
    default_llm = deps.default_llm
    new_resolutions = resolutions
    prompt_params = {
        "misformatted_concepts": ", ".join(sorted([c for c in misformatted_concepts]))
    }
    prompt_bool_params = {
        f"is_{c}": True for c in misformatted_concepts
    }
    prompt_bool_params.update({f"is_{c}": False for c in get_args(Concept) if c not in misformatted_concepts})
    prompt_target_resolutions = {
        f"{c}_resolution": resolutions[c].model_dump() for c in misformatted_concepts
    }
    prompt_target_resolutions.update({f"{c}_resolution": resolutions[c].model_dump() for c in get_args(Concept) if c not in misformatted_concepts}) # TODO: I can just make this in the first place without needing to do this
    prompt_params.update(prompt_bool_params)
    prompt_params.update(prompt_target_resolutions)
    clarification_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=generate_immediate_single_column_feedback(**prompt_params),
        additional_kwargs={
            'created_at': datetime.now(timezone.utc).isoformat(),
    })
    new_messages = [messages[0]] + [clarification_message]
    FieldResolutionCorrectionDyn = build_dynamic_resolution_correction_model(
        misformatted_concepts, resolution_model)

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    while retries < retry_limit:
        try:
            resolved: Any = await asyncio.wait_for(
                deterministic_llm.acall_structured(new_messages, FieldResolutionCorrectionDyn), timeout=CALL_TIMEOUT)
            for concept in misformatted_concepts:
                new_resolutions[concept] = getattr(resolved, concept)
            break
        except asyncio.TimeoutError:
            retries += 1
            continue
        except OutputParserException as e:
            human_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=f"The previous output from the LLM failed to parse with error: {e}. Please reformat the output to match the expected format and ensure that all required fields are included.",
                additional_kwargs={
                    'created_at': datetime.now(timezone.utc).isoformat(),
                })
            resolved: Any = await asyncio.wait_for(
                deterministic_llm.acall_structured(new_messages + [human_message], FieldResolutionCorrectionDyn), timeout=CALL_TIMEOUT)
            for concept in misformatted_concepts:
                new_resolutions[concept] = getattr(resolved, concept)
            break
        except ValidationError as e:
            for concept in sorted(misformatted_concepts):
                error_resolution = ErrorResolution(
                    status="error",
                    confidence=0.0,
                    notes=[f"LLM output failed validation: {e}"],
                    error=f"LLM output failed validation: {e}")
                print(f"\n\nValidation error for concept {concept}: {e}. Setting resolution to error with notes.")
                new_resolutions[concept] = error_resolution
            break
        except Exception as e:
            retries += 1
            continue
    return new_resolutions

async def _extract_column_for_concept_poor_parsing(
        poorly_parsed_concepts: List[Concept], messages: Dict[str, List[AnyMessage]], config: RunnableConfig,
        resolution_model: Any, parse_rates: Dict[str, float], user_input: GEOMetadataExtractionInput,
        resolutions: Dict[str, Any], resolution_model_envelope: Any, failed_parsing_info: Dict[str, Any]) -> Dict[Concept, FieldResolution]:
    print(f"\n\nPoorly parsed concepts: {poorly_parsed_concepts} with parse rates {parse_rates}")
    deps: Deps = config["configurable"]["deps"]
    deterministic_llm = deps.deterministic_llm
    default_llm = deps.default_llm
    new_resolutions = resolutions
    prompt_params = {
        "user_input": user_input.model_dump()
    }
    prompt_bool_params = {
        f"is_{c}": True for c in poorly_parsed_concepts
    }
    prompt_bool_params.update({f"is_{c}": False for c in get_args(Concept) if c not in poorly_parsed_concepts})
    prompt_parsing_rate_params = {
        f"{c}_rate": parse_rates[c] for c in poorly_parsed_concepts
    }
    prompt_parsing_rate_params.update({f"{c}_rate": 0 for c in get_args(Concept) if c not in poorly_parsed_concepts})
    prompt_failed_parsing_info_params = {
        f"{c}_failed": failed_parsing_info[c] for c in poorly_parsed_concepts
    }
    prompt_failed_parsing_info_params.update({f"{c}_failed": [] for c in get_args(Concept) if c not in poorly_parsed_concepts})
    prompt_target_resolutions = {
        f"{c}_resolution": resolutions[c].model_dump() for c in poorly_parsed_concepts
    }
    prompt_target_resolutions.update({f"{c}_resolution": resolutions[c].model_dump() for c in get_args(Concept) if c not in poorly_parsed_concepts})
    prompt_params.update(prompt_bool_params)
    prompt_params.update(prompt_parsing_rate_params)
    prompt_params.update(prompt_failed_parsing_info_params)
    prompt_params.update(prompt_target_resolutions)

    FieldResolutionCorrectionDyn = build_dynamic_resolution_correction_model(
        poorly_parsed_concepts, resolution_model)
    prompt_params["json_format"] = json.dumps(
        FieldResolutionCorrectionDyn.model_json_schema(), indent=2)

    clarification_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=generate_column_feedback_loop_prompt(**prompt_params))
    new_messages = [messages[0]] + [clarification_message]

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    while retries < retry_limit:
        try:
            resolved: Any = await asyncio.wait_for(
                deterministic_llm.acall_structured(new_messages, FieldResolutionCorrectionDyn), timeout=CALL_TIMEOUT)
            for concept in poorly_parsed_concepts:
                new_resolutions[concept] = getattr(resolved, concept)
            break
        except asyncio.TimeoutError:
            retries += 1
            continue
        except OutputParserException as e:
            human_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=f"The previous output from the LLM failed to parse with error: {e}. Please reformat the output to match the expected format and ensure that all required fields are included.",
                additional_kwargs={
                    'created_at': datetime.now(timezone.utc).isoformat(),
                })
            resolved: Any = await asyncio.wait_for(
                deterministic_llm.acall_structured(new_messages + [human_message], FieldResolutionCorrectionDyn), timeout=CALL_TIMEOUT)
            for concept in poorly_parsed_concepts:
                new_resolutions[concept] = getattr(resolved, concept)
            break
        except ValidationError as e:
            for concept in sorted(poorly_parsed_concepts):
                error_resolution = ErrorResolution(
                    status="error",
                    confidence=0.0,
                    notes=[f"LLM output failed validation: {e}"],
                    error=f"LLM output failed validation: {e}")
                print(f"\n\nValidation error for concept {concept}: {e}. Setting resolution to error with notes.")
                new_resolutions[concept] = error_resolution
            break
        except Exception as e:
            retries += 1
            continue
    return new_resolutions

async def _extract_column_for_concept_with_retry(
        extraction_result: Any, count:int = 0, retries:int = 2, parse_rates: Dict[str, float] = None, formatting_only:bool = False,
        resolutions: Dict[str, Any] = None, config: RunnableConfig = None, messages: List[AnyMessage] = None,
        user_input: GEOMetadataExtractionInput = None, failed_parsing_info: Dict[str, Any] = None,
        parsed_disease_statuses: List[str] = None) -> Dict[Concept, Any]:
    # Construct the message here
    _, resolution_model, resolution_model_envelope, __ = _get_custom_models(user_input)
    resolutions = _get_extraction_resolutions(extraction_result) if resolutions is None else resolutions
    flagged_concepts, flagged_concept_notes = _check_extraction_patterns(resolutions)
    for concept in flagged_concepts:
        resolutions[concept].notes.append(flagged_concept_notes[concept])
        resolutions[concept].units = None if concept != "age" else resolutions[concept].units
    re_run = False
    if (count >= retries) or (len(flagged_concepts) == 0 and formatting_only):
        return resolutions, re_run
    
    if len(flagged_concepts) > 0:
        print(f"\n\nFlagged concepts for misformatted extraction patterns: {flagged_concepts}. Attempting to re-extract columns for these concepts with a formatting-focused prompt.")
        # If there are misformatted concept patterns  attempt to return these patterns
        new_resolutions = await _extract_column_for_concept_misformatted(
            flagged_concepts, messages, config, resolution_model, resolutions, resolution_model_envelope)
        resolutions.update(new_resolutions)
        new_resolutions, re_run = await _extract_column_for_concept_with_retry(
            extraction_result, count=count+1, retries=retries, parse_rates=parse_rates, formatting_only=formatting_only,
            config=config, resolutions=resolutions, messages=messages, user_input=user_input, failed_parsing_info=failed_parsing_info,
            parsed_disease_statuses=parsed_disease_statuses)
    
    if not formatting_only:
        print(f"\n\nChecking for poorly parsed concepts with parse rates {parse_rates}")
        flagged_concepts = [c for c in get_args(Concept) if (extraction_result[c]["status"] == "resolved") and (c in parse_rates) and (parse_rates[c] > 0.0 and parse_rates[c] < 1.0) and len(failed_parsing_info[c]) > 0]
        if len(flagged_concepts) > 0:
            new_resolutions = await _extract_column_for_concept_poor_parsing(
                flagged_concepts, messages, config, resolution_model, parse_rates, user_input, resolutions, resolution_model_envelope, failed_parsing_info)
            resolutions.update(new_resolutions)
            re_run = True

        if resolutions['disease_status'].status == "resolved" and parsed_disease_statuses is not None:
            resolutions = await _extract_column_for_concept_disease_status(
                resolutions, config, parsed_disease_statuses)
        
        if resolutions['age'].status == "missing":
            resolutions = await _extract_column_for_concept_age(
                resolutions, config, user_input, messages, resolution_model)

    return resolutions, re_run
    
async def _extract_all_columns(
        messages: List[AnyMessage],
        config: RunnableConfig,
        result_model: Any,
) -> GEOMetadataExtractionResult:
    deps: Deps = config["configurable"]["deps"]
    deterministic_llm = deps.deterministic_llm
    default_llm = deps.default_llm

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    call_messages = messages
    while retries < retry_limit:
        print(f"\nAttempt {retries} to extract all columns from metadata...")
        try:
            resolved: Any = result_model.model_validate(
                await asyncio.wait_for(deterministic_llm.acall_structured(call_messages, result_model), timeout=CALL_TIMEOUT))
            print(f"\n\nInitial extraction result: {resolved}\n\n")
            resolved = GEOMetadataExtractionResult(**resolved.model_dump())
            resolutions = _get_extraction_resolutions(resolved.model_dump())
            if resolutions["disease_status"].status == "resolved" and resolutions["disease_status"].extraction.field_name != "default":
                resolutions["disease_status"].extraction.pattern = "([\\s\\S]*)"
            break
        except asyncio.TimeoutError:
            retries += 1
            continue
        except OutputParserException as e:
            print(f"\n\nOutput parser exception during extraction: {e}. Retrying extraction with clarification prompt.")
            human_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=f"The previous output from the LLM failed to parse with error: {e}. Please reformat the output to match the expected format and ensure that all required fields are included.",
                additional_kwargs={
                    'created_at': datetime.now(timezone.utc).isoformat(),
                })
            call_messages += [human_message]
            retries += 1
            continue
        except ValidationError as e:
            print(f"\n\nValidation error for extraction: {e}. Setting resolution to error with notes.")
            resolved = GEOMetadataExtractionResult(
                artifact=None,
                resolutions={concept: ErrorResolution(
                    status="error",
                    confidence=0.0,
                    notes=[f"LLM output failed validation: {e}"],
                    error=f"LLM output failed validation: {e}"
                ) for concept in get_args(Concept)},
                execution_status="failed",
                error=f"LLM output failed validation: {e}"
            )
            break
        except Exception as e:
            print(f"\n\nUnexpected error during extraction: {e}. Retrying...")
            retries += 1
            continue

    return resolved

async def extract_metadata_columns_alt(
    user_input: GEOMetadataExtractionInput,
    return_dict: Dict[str, Any],
    config: RunnableConfig,
    output_root: str,
    accession_code: str,
    llm_messages: List[AnyMessage]
) -> GEOMetadataExtractionResult:
    GSESpecificMetadataExtractionResult, _, __, key_names = _get_custom_models(user_input)
    key_names_str = ", ".join(sorted(list(key_names)))
    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=generate_metadata_column_user_query_alt(user_input=user_input, key_names=key_names_str),
        additional_kwargs={
            'created_at': datetime.now(timezone.utc).isoformat(),
        })
    return_dict["llm_messages"].append(human_message)
    messages = [llm_messages[0], return_dict["llm_messages"][-1]]
    extraction_result = await _extract_all_columns(
        messages, llm_messages, config, GSESpecificMetadataExtractionResult, user_input)

    artifact = None
    artifact_path = os.path.join(output_root, "extraction_protocol.json")
    resolutions = []
    if hasattr(extraction_result, "resolutions"):
        resolutions = extraction_result.resolutions
    else:
        resolutions = {key: getattr(extraction_result, key) for key in get_args(Concept) if hasattr(extraction_result, key)}

    if all((r.status in ["resolved", "missing"]) and (r.confidence > 0.6) for r in resolutions.values()):
        try:
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump({
                    concept: resolution.model_dump() for concept, resolution in resolutions.items()
                },
                f,
                ensure_ascii=False,
                indent=2)
            artifact = ArtifactRef.model_validate({
                "path": artifact_path,
                "kind": "metadata_extraction_protocol",
                "accession_code": accession_code,
                "sha256": compute_sha256(artifact_path, is_path=True),
                "bytes": os.path.getsize(artifact_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            execution_status = "failed"
            artifact = None
    
    extraction_result.artifact = artifact
    return_dict["datasets"][accession_code]["metadata_extraction_result"] = extraction_result.model_dump()
    return return_dict

async def _extract_column_for_concept(
    messages: List[AnyMessage],
    config: RunnableConfig
) -> FieldResolution:
    deps: Deps = config["configurable"]["deps"]
    deterministic_llm = deps.deterministic_llm
    default_llm = deps.default_llm

    try:
        resolved: FieldResolution = FieldResolution.model_validate(
            await deterministic_llm.acall_structured(messages, FieldResolution))
    except ValidationError as e:
        resolved: FieldResolution = FieldResolution(
            units=None,
            extraction=None,
            confidence=0.0,
            status="error",
            notes=[],
            error=f"LLM output failed validation: {e}")

    return resolved

async def extract_metadata_columns(
    user_input: GEOMetadataExtractionInput,
    return_dict: Dict[str, Any],
    config: RunnableConfig,
    output_root: str,
    accession_code: str
) -> GEOMetadataExtractionResult:
    def _check_execution_status(resolutions: Dict[Concept, FieldResolution]) -> str:
        status = "succeeded"
        resolution_statuses = [r.status for r in resolutions.values()]
        if all(s == "error" for s in resolution_statuses):
            status = "failed"
        return status

    # On the dataset level
    resolutions = {}
    for concept in get_args(Concept):
        human_message = HumanMessage(
            id=uuid.uuid4().hex,
            content=generate_metadata_column_user_query(concept=concept, user_input=user_input),
            additional_kwargs={
                'created_at': datetime.now(timezone.utc).isoformat(),
            })
        return_dict["messages"].append(human_message)
        return_dict["concept_messages"][concept].append(human_message) 
        resolutions[concept] = await _extract_column_for_concept(
            return_dict["concept_messages"][concept], config)

    execution_status = _check_execution_status(resolutions)
    artifact = None
    artifact_path = os.path.join(output_root, "extraction_protocol.json")
    if all((r.status in ["resolved", "missing"]) and (r.confidence > 0.6) for r in resolutions.values()):
        try:
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump({
                    concept: resolution.model_dump() for concept, resolution in resolutions.items()
                },
                f,
                ensure_ascii=False,
                indent=2)
            artifact = ArtifactRef.model_validate({
                "path": artifact_path,
                "kind": "metadata_extraction_protocol",
                "accession_code": accession_code,
                "sha256": compute_sha256(artifact_path, is_path=True),
                "bytes": os.path.getsize(artifact_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            execution_status = "failed"
            artifact = None
    
    metadata_extraction_result_kwargs = {
        "artifact": artifact,
        "execution_status": execution_status,
        "error": "There was an error extracting metadata columns." if execution_status == "failed" else None
    }
    metadata_extraction_result_kwargs.update(resolutions)
    return_dict["datasets"][accession_code]["metadata_extraction_result"] = GEOMetadataExtractionResult(**metadata_extraction_result_kwargs).model_dump()
    return return_dict

    

