__all__ = [
    "extract_metadata_columns",
    "_extract_column_for_concept",
    "extract_metadata_columns_alt",
    "_extract_column_for_concept_with_retry",
    "_get_parse_rate",
    "_get_extraction_resolutions",
    "_get_custom_models",
    "_extract_all_columns",
    "_check_extraction_patterns",
    "_extract_column_for_concept_misformatted",
    "_extract_column_for_concept_poor_parsing",
    "_get_parse_rate",
    "_extract_column_for_concept_disease_status",
    "_extract_column_for_concept_age",
]
import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any, cast, get_args

from greenery import parse
from greenery.rxelems import Pattern
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from ...agent.graphs.deps import Deps
from ...contracts.common import ArtifactRef
from ...contracts.geo import (
    Concept,
    ErrorResolution,
    FieldResolution,
    FieldResolutionEnvelope,
    GEOMetadataExtractionInput,
    GEOMetadataExtractionResult,
    build_dynamic_control_identification_model,
    build_dynamic_resolution_correction_model,
    build_dynamic_result_model,
)
from ...utils.helper import compute_sha256
from ...utils.prompting import (
    generate_column_feedback_loop_prompt,
    generate_identify_control_value_prompt,
    generate_immediate_single_column_feedback,
    generate_metadata_column_user_query,
    generate_metadata_column_user_query_alt,
    generate_missing_age_check_prompt,
)

CALL_TIMEOUT = 180
GLOBAL_RETRY_LIMIT = 5


async def _invoke_llm_with_retry(
    messages,
    model,
    config,
    retry_limit=5,
    timeout=180,
    feedback_fn=None,
    error_result_factory=None,
):
    """Invoke an LLM with structured output, retrying on common failures.

    Args:
        messages: LangChain message list to send.
        model: Pydantic model class for structured output parsing.
        config: RunnableConfig with LLM dependencies.
        retry_limit: Maximum number of retry attempts.
        timeout: Timeout in seconds for each individual attempt.
        feedback_fn: Optional async callable(messages, config, attempt) that
            returns (new_messages, should_continue) for recovery attempts.
        error_result_factory: Optional callable() -> Any that creates a
            fallback result when all retries are exhausted.

    Returns:
        The parsed structured output, or the result of error_result_factory()
        if all retries are exhausted.
    """
    import asyncio

    from langchain_core.exceptions import OutputParserException
    from pydantic import ValidationError

    deps = config["configurable"]["deps"]
    llm = deps.llm

    retries = 0
    resolved = None
    while retries < retry_limit:
        try:
            resolved = await asyncio.wait_for(
                llm.acall_structured(messages, model),
                timeout=timeout,
            )
            break
        except TimeoutError:
            retries += 1
            continue
        except OutputParserException:
            if feedback_fn is not None:
                messages, should_continue = await feedback_fn(messages, config, retries)
                if not should_continue:
                    break
                retries += 1
                continue
            retries += 1
            continue
        except ValidationError:
            if error_result_factory is not None:
                return error_result_factory()
            break
        except Exception:
            retries += 1
            if retries >= retry_limit:
                if error_result_factory is not None:
                    return error_result_factory()
                break
    return resolved


async def get_all_extraction_resolutions():
    """Retrieve all extraction resolutions across known concepts.

    Returns:
        None: Not yet implemented.
    """
    pass


def has_alternation_anywhere(regex: str) -> bool:
    """Check whether a regular expression contains alternation (``|``) at any nesting level.

    Parses the regex into a ``greenery`` syntax tree and walks its subpatterns recursively.

    Args:
        regex: The regular expression string to inspect.

    Returns:
        True if any subpattern contains more than one alternative branch.
    """
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
    """Build dynamic Pydantic models and collect key names from ``characteristics_ch1``.

    Extracts the set of key names used across all sample characteristic dictionaries,
    then constructs a dynamic ``GEOMetadataExtractionResult`` model specialised for those keys.

    Args:
        user_input: The GEO metadata extraction input whose ``characteristics_ch1``
            list is scanned for key names.

    Returns:
        A 4-tuple of:
        - The dynamic ``GEOMetadataExtractionResult`` model class.
        - The dynamic ``FieldResolution`` model class.
        - The dynamic ``FieldResolutionEnvelope`` model class.
        - The set of discovered key names.
    """
    key_names = set()
    for example in user_input.characteristics_ch1:
        key_names.update(v.split(":", 1)[0].strip() for v in example)
    GEOMetadataExtractionResultDyn, FieldResolutionDyn, FieldResolutionEnvelopeDyn = build_dynamic_result_model(
        tuple(sorted(list(key_names)))
    )
    return GEOMetadataExtractionResultDyn, FieldResolutionDyn, FieldResolutionEnvelopeDyn, key_names


def _get_parse_rate(metadata_summary: dict[str, Any] = None) -> dict[str, float]:  # type: ignore
    """Extract per-concept parse rates from a metadata summary dictionary.

    For each ``Concept`` whose entry in the summary is not a list, reads the
    ``parse_rate`` value.

    Args:
        metadata_summary: An optional metadata summary dictionary keyed by concept
            name.  Each value is expected to contain a ``parse_rate`` key when it is
            not a list.  If ``None``, an empty dict is returned.

    Returns:
        A mapping from concept name to its parse rate as a float.
    """
    parse_rates = {}
    if metadata_summary is not None:
        parse_rates = {
            concept: metadata_summary[concept]["parse_rate"]
            for concept in get_args(Concept)
            if not isinstance(metadata_summary[concept], list)
        }
    return parse_rates


def _get_extraction_resolutions(extraction_result: Any) -> dict[str, Any]:
    """Extract per-concept ``FieldResolution`` objects from a raw extraction result.

    The result may be either a dictionary or an object with attribute access.
    Each present concept is wrapped through ``FieldResolutionEnvelope``.

    Args:
        extraction_result: A dict or object containing resolution data for each
            ``Concept``.

    Returns:
        A mapping from each ``Concept`` to its unwrapped ``FieldResolution``.
    """
    resolutions = {}
    for concept in get_args(Concept):
        concept_presence = (
            (concept in extraction_result)
            if isinstance(extraction_result, dict)
            else hasattr(extraction_result, concept)
        )
        if concept_presence:
            resolution = (
                extraction_result.get(concept, None)
                if isinstance(extraction_result, dict)
                else getattr(extraction_result, concept, None)
            )
            resolutions[concept] = FieldResolutionEnvelope(resolution=resolution).resolution  # type: ignore
    return resolutions


def _check_extraction_patterns(resolutions: dict[str, Any]) -> list[Concept]:
    """Flag concepts whose extraction patterns embed their own key names.

    Iterates over resolved concepts sourced from ``characteristics_ch1`` and
    checks whether the extraction pattern string contains the key name it is
    meant to match (indicating a non-generic, overfitted pattern).

    Args:
        resolutions: Per-concept ``FieldResolution`` objects keyed by concept name.

    Returns:
        A 2-tuple of:
        - A list of concept names whose patterns were flagged.
        - A dict mapping each flagged concept to a human-readable note explaining
          why it was flagged.
    """
    flagged_patterns = []
    notes_flagged_patterns = {}
    for concept, resolution in resolutions.items():
        if resolution.status == "resolved" and resolution.extraction.field_name == "characteristics_ch1":
            key_name = resolution.extraction.key_name
            pattern = resolution.extraction.pattern
            if key_name.lower() in pattern.lower():
                flagged_patterns.append(concept)
                notes_flagged_patterns[concept] = f"Key name '{key_name}' found in pattern '{pattern}'"
            # if concept in flexible_concepts:
            #    if has_alternation_anywhere(pattern):
            #        flagged_patterns.append(concept)
            #        notes_flagged_patterns[concept] = f"The regular expression pattern '{pattern}' contains alternations, which violates the generic pattern requirement. Modify this pattern to remove all alternations and be simpler and more generic."
    return flagged_patterns, notes_flagged_patterns  # type: ignore


async def _extract_column_for_concept_age(
    resolutions: dict[str, Any],
    config: RunnableConfig,
    user_input: GEOMetadataExtractionInput,
    messages: list[AnyMessage],
    resolution_model: Any,
) -> dict[str, Any]:
    """Attempt to resolve a missing ``age`` column via an LLM clarification check.

    Constructs a prompt that asks the LLM to re-examine the metadata for age
    information that may have been overlooked, then calls the deterministic LLM
    with a dynamic resolution correction model.

    Args:
        resolutions: Current per-concept resolutions, including the ``age``
            entry whose ``status`` is ``"missing"``.
        config: The runnable config providing access to ``Deps``.
        user_input: The original GEO metadata extraction input.
        messages: The message history to include in the prompt context.
        resolution_model: The dynamic ``FieldResolution`` model class used to
            build the correction model.

    Returns:
        The updated resolutions dict, with ``age`` potentially resolved or
        left unchanged on failure.
    """
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm
    clarification_message = AIMessage(
        id=uuid.uuid4().hex,
        content=generate_missing_age_check_prompt(user_input=user_input),
        additional_kwargs={
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    prompt_messages = messages + [clarification_message]
    FieldResolutionCorrectionDyn = build_dynamic_resolution_correction_model(["age"], resolution_model)  # type: ignore
    try:
        resolved: Any = await llm.acall_structured(prompt_messages, FieldResolutionCorrectionDyn)
        resolutions["age"] = resolved.age
    except ValidationError as e:
        print(f"\n\nValidation error for concept age: {e}. Setting resolution to error with notes.")

    return resolutions


async def _extract_column_for_concept_disease_status(
    resolutions: dict[str, Any], config: RunnableConfig, parsed_disease_statuses: list[str]
) -> dict[str, Any]:
    """Identify the control value among disease statuses via an LLM call.

    Builds a dynamic control identification model from the set of parsed
    disease status values and asks the deterministic LLM to pick the control.
    Retries on timeout, parse failure, and validation errors up to
    ``GLOBAL_RETRY_LIMIT``.

    Args:
        resolutions: Current per-concept resolutions, where
            ``disease_status`` must be in ``"resolved"`` state.
        config: The runnable config providing access to ``Deps``.
        parsed_disease_statuses: The list of distinct disease status strings
            parsed from the dataset.

    Returns:
        The updated resolutions dict with ``disease_status`` enriched with a
        ``control_value`` on success, or left unchanged on failure.
    """
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm
    if resolutions["disease_status"].extraction.field_name == "default" or not hasattr(
        resolutions["disease_status"].extraction, "key_name"
    ):
        key_name = "N/A"
    else:
        key_name = resolutions["disease_status"].extraction.key_name
    ControlIdentificationModel = build_dynamic_control_identification_model(parsed_disease_statuses)  # type: ignore
    clarification_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=generate_identify_control_value_prompt(
            disease_statuses=parsed_disease_statuses,
            key_name=key_name,
            json_format=json.dumps(ControlIdentificationModel.model_json_schema(), indent=2),
        ),
        additional_kwargs={
            "created_at": datetime.now(UTC).isoformat(),
        },
    )

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    call_messages = [clarification_message]
    while retries < retry_limit:
        try:
            resolved: Any = await asyncio.wait_for(
                llm.acall_structured(call_messages, ControlIdentificationModel),  # type: ignore
                timeout=CALL_TIMEOUT,
            )
            resolutions["disease_status"].extraction.control_value = resolved.control_value
            break
        except TimeoutError:
            retries += 1
            continue
        except OutputParserException as e:
            human_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=f"The previous output from the LLM failed to parse with error: {e}. Please reformat the output to match the expected format and ensure that all required fields are included.",
                additional_kwargs={
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            call_messages += [human_message]
            continue
        except ValidationError as e:
            print(f"\n\nValidation error for concept disease_status: {e}. Setting resolution to error with notes.")
            break
        except Exception:
            retries += 1
            continue

    return resolutions


async def _extract_column_for_concept_misformatted(
    misformatted_concepts: list[Concept],
    messages: list[AnyMessage],
    config: RunnableConfig,
    resolution_model: Any,
    resolutions: dict[str, Any],
    resolution_model_envelope: Any,
) -> dict[str, Any]:
    """Re-extract columns for concepts flagged with misformatted patterns.

    Generates a targeted feedback prompt that describes which concepts have
    problematic extraction patterns and asks the LLM to produce corrected
    resolutions.  Retries on timeout, parse failure, and validation errors.

    Args:
        misformatted_concepts: The list of concept names whose patterns were
            flagged as non-generic.
        messages: The message history used to build the prompt context.
        config: The runnable config providing access to ``Deps``.
        resolution_model: The dynamic ``FieldResolution`` model class for
            building the correction model.
        resolutions: Current per-concept resolutions, which will be updated
            in-place for flagged concepts.
        resolution_model_envelope: Not used in the current implementation.

    Returns:
        The updated resolutions dict with corrected entries for the
        misformatted concepts, or error resolutions on validation failure.
    """
    print(f"\n\nMisformatted concepts: {misformatted_concepts}")
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm
    new_resolutions = resolutions
    prompt_params = {"misformatted_concepts": ", ".join(sorted([c for c in misformatted_concepts]))}
    prompt_bool_params = {f"is_{c}": True for c in misformatted_concepts}
    prompt_bool_params.update({f"is_{c}": False for c in get_args(Concept) if c not in misformatted_concepts})  # type: ignore
    prompt_target_resolutions = {f"{c}_resolution": resolutions[c].model_dump() for c in misformatted_concepts}
    prompt_target_resolutions.update(  # type: ignore
        {f"{c}_resolution": resolutions[c].model_dump() for c in get_args(Concept) if c not in misformatted_concepts}
    )  # TODO: I can just make this in the first place without needing to do this
    prompt_params.update(prompt_bool_params)  # type: ignore
    prompt_params.update(prompt_target_resolutions)  # type: ignore
    clarification_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=generate_immediate_single_column_feedback(**prompt_params),
        additional_kwargs={
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    new_messages = [messages[0]] + [clarification_message]
    FieldResolutionCorrectionDyn = build_dynamic_resolution_correction_model(misformatted_concepts, resolution_model)  # type: ignore

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    while retries < retry_limit:
        try:
            resolved: Any = await asyncio.wait_for(
                llm.acall_structured(new_messages, FieldResolutionCorrectionDyn),
                timeout=CALL_TIMEOUT,
            )
            for concept in misformatted_concepts:
                new_resolutions[concept] = getattr(resolved, concept)
            break
        except TimeoutError:
            retries += 1
            continue
        except OutputParserException as e:
            human_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=f"The previous output from the LLM failed to parse with error: {e}. Please reformat the output to match the expected format and ensure that all required fields are included.",
                additional_kwargs={
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            new_messages += [human_message]
            retries += 1
            continue
        except ValidationError as e:
            for concept in sorted(misformatted_concepts):
                error_resolution = ErrorResolution(
                    status="error",
                    confidence=0.0,
                    notes=[f"LLM output failed validation: {e}"],
                    error=f"LLM output failed validation: {e}",
                )
                print(f"\n\nValidation error for concept {concept}: {e}. Setting resolution to error with notes.")
                new_resolutions[concept] = error_resolution
            break
        except Exception:
            retries += 1
            continue
    return new_resolutions


async def _extract_column_for_concept_poor_parsing(
    poorly_parsed_concepts: list[Concept],
    messages: list[AnyMessage],
    config: RunnableConfig,
    resolution_model: Any,
    parse_rates: dict[str, float],
    user_input: GEOMetadataExtractionInput,
    resolutions: dict[str, Any],
    resolution_model_envelope: Any,
    failed_parsing_info: dict[str, Any],
) -> dict[str, Any]:
    """Re-extract columns for concepts whose parse rate is below 1.0.

    Builds a feedback-loop prompt that includes parse rate metrics and
    examples of failed parsing attempts, then asks the LLM to produce
    improved resolutions.  Retries on timeout, parse failure, and validation
    errors up to ``GLOBAL_RETRY_LIMIT``.

    Args:
        poorly_parsed_concepts: The list of concept names with a parse rate
            in (0, 1).
        messages: The message history (only the first entry is reused as
            context in the new prompt).
        config: The runnable config providing access to ``Deps``.
        resolution_model: The dynamic ``FieldResolution`` model class for
            building the correction model.
        parse_rates: Per-concept parse rates as floats.
        user_input: The original GEO metadata extraction input.
        resolutions: Current per-concept resolutions, updated in-place on
            success.
        resolution_model_envelope: Not used in the current implementation.
        failed_parsing_info: Per-concept lists of samples that failed to parse
            with the current pattern, used to illustrate the problem to the
            LLM.

    Returns:
        The updated resolutions dict with improved entries for the poorly
        parsed concepts, or error resolutions on validation failure.
    """
    print(f"\n\nPoorly parsed concepts: {poorly_parsed_concepts} with parse rates {parse_rates}")
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm
    new_resolutions = resolutions
    prompt_params = {"user_input": user_input.model_dump()}
    prompt_bool_params = {f"is_{c}": True for c in poorly_parsed_concepts}
    prompt_bool_params.update({f"is_{c}": False for c in get_args(Concept) if c not in poorly_parsed_concepts})  # type: ignore
    prompt_parsing_rate_params = {f"{c}_rate": parse_rates[c] for c in poorly_parsed_concepts}
    prompt_parsing_rate_params.update({f"{c}_rate": 0 for c in get_args(Concept) if c not in poorly_parsed_concepts})  # type: ignore
    prompt_failed_parsing_info_params = {f"{c}_failed": failed_parsing_info[c] for c in poorly_parsed_concepts}
    prompt_failed_parsing_info_params.update(  # type: ignore
        {f"{c}_failed": [] for c in get_args(Concept) if c not in poorly_parsed_concepts}
    )
    prompt_target_resolutions = {f"{c}_resolution": resolutions[c].model_dump() for c in poorly_parsed_concepts}
    prompt_target_resolutions.update(  # type: ignore
        {f"{c}_resolution": resolutions[c].model_dump() for c in get_args(Concept) if c not in poorly_parsed_concepts}
    )
    prompt_params.update(prompt_bool_params)  # type: ignore
    prompt_params.update(prompt_parsing_rate_params)  # type: ignore
    prompt_params.update(prompt_failed_parsing_info_params)  # type: ignore
    prompt_params.update(prompt_target_resolutions)  # type: ignore

    FieldResolutionCorrectionDyn = build_dynamic_resolution_correction_model(poorly_parsed_concepts, resolution_model)  # type: ignore
    prompt_params["json_format"] = json.dumps(FieldResolutionCorrectionDyn.model_json_schema(), indent=2)

    clarification_message = HumanMessage(
        id=uuid.uuid4().hex, content=generate_column_feedback_loop_prompt(**prompt_params)
    )
    new_messages = [messages[0]] + [clarification_message]

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    while retries < retry_limit:
        try:
            resolved: Any = await asyncio.wait_for(
                llm.acall_structured(new_messages, FieldResolutionCorrectionDyn),
                timeout=CALL_TIMEOUT,
            )
            for concept in poorly_parsed_concepts:
                new_resolutions[concept] = getattr(resolved, concept)
            break
        except TimeoutError:
            retries += 1
            continue
        except OutputParserException as e:
            human_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=f"The previous output from the LLM failed to parse with error: {e}. Please reformat the output to match the expected format and ensure that all required fields are included.",
                additional_kwargs={
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            resolved: Any = await asyncio.wait_for(
                llm.acall_structured(new_messages + [human_message], FieldResolutionCorrectionDyn),
                timeout=CALL_TIMEOUT,
            )
            for concept in poorly_parsed_concepts:
                new_resolutions[concept] = getattr(resolved, concept)
            break
        except ValidationError as e:
            for concept in sorted(poorly_parsed_concepts):
                error_resolution = ErrorResolution(
                    status="error",
                    confidence=0.0,
                    notes=[f"LLM output failed validation: {e}"],
                    error=f"LLM output failed validation: {e}",
                )
                print(f"\n\nValidation error for concept {concept}: {e}. Setting resolution to error with notes.")
                new_resolutions[concept] = error_resolution
            break
        except Exception:
            retries += 1
            continue
    return new_resolutions


async def _extract_column_for_concept_with_retry(
    extraction_result: Any,
    count: int = 0,
    retries: int = 2,
    parse_rates: dict[str, float] | None = None,
    formatting_only: bool = False,
    resolutions: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    messages: list[AnyMessage] | None = None,
    user_input: GEOMetadataExtractionInput | None = None,
    failed_parsing_info: dict[str, Any] | None = None,
    parsed_disease_statuses: list[str] | None = None,
) -> dict[str, Any]:
    """Orchestrate iterative refinement of extraction resolutions with retries.

    This is the main feedback-loop coordinator.  On each call it:
    1. Checks for misformatted patterns and invokes
       ``_extract_column_for_concept_misformatted``.
    2. Optionally checks for poorly parsed concepts and invokes
       ``_extract_column_for_concept_poor_parsing``.
    3. Attempts to identify the control value for ``disease_status``.
    4. Attempts to resolve a missing ``age`` column.

    Recursion is bounded by ``count`` and ``retries``.

    Args:
        extraction_result: The raw extraction result from the initial LLM call.
        count: Current recursion depth (0-indexed).
        retries: Maximum recursion depth before returning.
        parse_rates: Per-concept parse rates from the metadata summary.
        formatting_only: If True, skip poor-parsing, disease-status, and age
            checks; only correct misformatted patterns.
        resolutions: Current per-concept resolutions.  Built from
            ``extraction_result`` if not provided on first call.
        config: The runnable config providing access to ``Deps``.
        messages: The message history for prompt construction.
        user_input: The original GEO metadata extraction input.
        failed_parsing_info: Per-concept lists of sample values that failed to
            parse.
        parsed_disease_statuses: Distinct disease status strings parsed from
            the dataset.

    Returns:
        A 2-tuple of (updated resolutions dict, re_run boolean).  ``re_run``
        is True if any concept was re-extracted and the caller should
        re-evaluate the workflow.
    """
    # Construct the message here
    _, resolution_model, resolution_model_envelope, __ = _get_custom_models(user_input)  # type: ignore
    resolutions = _get_extraction_resolutions(extraction_result) if resolutions is None else resolutions
    flagged_concepts, flagged_concept_notes_raw = _check_extraction_patterns(resolutions)
    flagged_concept_notes = cast(dict[str, str], flagged_concept_notes_raw)
    for concept in flagged_concepts:
        resolutions[concept].notes.append(flagged_concept_notes[concept])
        resolutions[concept].units = None if concept != "age" else resolutions[concept].units
    re_run = False
    if (count >= retries) or (len(flagged_concepts) == 0 and formatting_only):
        return resolutions, re_run  # type: ignore

    if len(flagged_concepts) > 0:
        print(
            f"\n\nFlagged concepts for misformatted extraction patterns: {flagged_concepts}. Attempting to re-extract columns for these concepts with a formatting-focused prompt."
        )
        # If there are misformatted concept patterns  attempt to return these patterns
        new_resolutions = await _extract_column_for_concept_misformatted(
            flagged_concepts,  # type: ignore
            messages,  # type: ignore
            config,  # type: ignore
            resolution_model,
            resolutions,
            resolution_model_envelope,
        )
        resolutions.update(new_resolutions)
        new_resolutions, re_run = await _extract_column_for_concept_with_retry(
            extraction_result,
            count=count + 1,
            retries=retries,
            parse_rates=parse_rates,
            formatting_only=formatting_only,
            config=config,
            resolutions=resolutions,
            messages=messages,
            user_input=user_input,
            failed_parsing_info=failed_parsing_info,
            parsed_disease_statuses=parsed_disease_statuses,
        )

    if not formatting_only:
        print(f"\n\nChecking for poorly parsed concepts with parse rates {parse_rates}")
        flagged_concepts = [
            c
            for c in get_args(Concept)
            if (extraction_result[c]["status"] == "resolved")
            and (c in parse_rates)  # type: ignore
            and (parse_rates[c] > 0.0 and parse_rates[c] < 1.0)  # type: ignore
            and len(failed_parsing_info[c]) > 0  # type: ignore
        ]
        if len(flagged_concepts) > 0:
            new_resolutions = await _extract_column_for_concept_poor_parsing(
                flagged_concepts,
                messages,  # type: ignore
                config,  # type: ignore
                resolution_model,
                parse_rates,  # type: ignore
                user_input,  # type: ignore
                resolutions,
                resolution_model_envelope,
                failed_parsing_info,  # type: ignore
            )
        resolutions.update(new_resolutions)  # ty: ignore
        re_run = True

        if resolutions["disease_status"].status == "resolved" and parsed_disease_statuses is not None:
            resolutions = await _extract_column_for_concept_disease_status(resolutions, config, parsed_disease_statuses)  # type: ignore

        if resolutions["age"].status == "missing":
            resolutions = await _extract_column_for_concept_age(
                resolutions,
                config,  # type: ignore
                user_input,  # type: ignore
                messages,  # ty: ignore
                resolution_model,
            )

    return resolutions, re_run  # type: ignore


async def _extract_all_columns(
    messages: list[AnyMessage],
    config: RunnableConfig,
    result_model: Any,
) -> dict[str, Any]:
    """Call the deterministic LLM to extract all metadata columns in one pass.

    Uses the provided dynamic result model and retries on timeout, output
    parser errors, and validation errors up to ``GLOBAL_RETRY_LIMIT``.
    On validation failure a synthetic failed result is constructed with all
    concepts set to error status.

    Args:
        messages: The prompt message history for the LLM call.
        config: The runnable config providing access to ``Deps``.
        result_model: The dynamic Pydantic model class (from
            ``_get_custom_models``) that the LLM output must conform to.

    Returns:
        A constructed ``GEOMetadataExtractionResult`` with resolved
        per-concept entries on success, or an all-error result on persistent
        failure.
    """
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    call_messages = messages
    while retries < retry_limit:
        print(f"\nAttempt {retries} to extract all columns from metadata...")
        try:
            resolved: Any = result_model.model_validate(
                await asyncio.wait_for(
                    llm.acall_structured(call_messages, result_model),
                    timeout=CALL_TIMEOUT,
                )
            )
            print(f"\n\nInitial extraction result: {resolved}\n\n")
            resolved = GEOMetadataExtractionResult(**resolved.model_dump())
            resolutions = _get_extraction_resolutions(resolved.model_dump())
            if (
                resolutions["disease_status"].status == "resolved"
                and resolutions["disease_status"].extraction.field_name != "default"
            ):
                resolutions["disease_status"].extraction.pattern = "([\\s\\S]*)"
            break
        except TimeoutError:
            retries += 1
            continue
        except OutputParserException as e:
            print(f"\n\nOutput parser exception during extraction: {e}. Retrying extraction with clarification prompt.")
            human_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=f"The previous output from the LLM failed to parse with error: {e}. Please reformat the output to match the expected format and ensure that all required fields are included.",
                additional_kwargs={
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            call_messages += [human_message]
            retries += 1
            continue
        except ValidationError as e:
            print(f"\n\nValidation error for extraction: {e}. Setting resolution to error with notes.")
            resolved = GEOMetadataExtractionResult(
                artifact=None,
                resolutions={  # type: ignore
                    concept: ErrorResolution(
                        status="error",
                        confidence=0.0,
                        notes=[f"LLM output failed validation: {e}"],
                        error=f"LLM output failed validation: {e}",
                    )
                    for concept in get_args(Concept)
                },
                execution_status="failed",  # type: ignore
                error=f"LLM output failed validation: {e}",
            )
            break
        except Exception as e:
            print(f"\n\nUnexpected error during extraction: {e}. Retrying...")
            retries += 1
            continue

    return resolved  # type: ignore


async def extract_metadata_columns_alt(
    user_input: GEOMetadataExtractionInput,
    return_dict: dict[str, Any],
    config: RunnableConfig,
    output_root: str,
    accession_code: str,
    llm_messages: list[AnyMessage],
) -> GEOMetadataExtractionResult:
    """Extract metadata columns using the alternative all-at-once LLM strategy.

    Builds custom dynamic models from the input's ``characteristics_ch1`` keys,
    constructs a single all-in-one prompt via
    ``generate_metadata_column_user_query_alt``, calls ``_extract_all_columns``,
    and saves the resulting extraction protocol as a JSON artifact when all
    resolutions are sufficiently confident.

    Args:
        user_input: The GEO metadata extraction input describing the dataset and
            field requests.
        return_dict: The shared workflow state dictionary.  The
            ``llm_messages`` list is appended to, and the extraction result is
            written into ``datasets[accession_code]``.
        config: The runnable config providing access to ``Deps``.
        output_root: Directory under which the ``extraction_protocol.json``
            artifact is written.
        accession_code: The GEO accession code identifying the current dataset.
        llm_messages: The message history used as context for the LLM call
            (only the first entry is reused).

    Returns:
        The updated ``return_dict`` after recording the extraction result.
    """
    GSESpecificMetadataExtractionResult, _, __, key_names = _get_custom_models(user_input)
    key_names_str = ", ".join(sorted(list(key_names)))
    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=generate_metadata_column_user_query_alt(user_input=user_input, key_names=key_names_str),
        additional_kwargs={
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return_dict["llm_messages"].append(human_message)
    messages = [llm_messages[0], return_dict["llm_messages"][-1]]
    extraction_result = await _extract_all_columns(  # type: ignore[call-arg,arg-type]
        messages,
        llm_messages,  # ty: ignore
        config,
        GSESpecificMetadataExtractionResult,  # ty: ignore
        user_input,
    )

    artifact = None
    artifact_path = os.path.join(output_root, "extraction_protocol.json")
    resolutions = []
    if hasattr(extraction_result, "resolutions"):
        resolutions = extraction_result.resolutions
    else:
        resolutions = {
            key: getattr(extraction_result, key) for key in get_args(Concept) if hasattr(extraction_result, key)
        }

    if all((r.status in ["resolved", "missing"]) and (r.confidence > 0.6) for r in resolutions.values()):  # type: ignore
        try:
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(
                    {concept: resolution.model_dump() for concept, resolution in resolutions.items()},  # type: ignore
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
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
        except Exception:
            artifact = None

    extraction_result.artifact = artifact  # type: ignore
    return_dict["datasets"][accession_code]["metadata_extraction_result"] = extraction_result.model_dump()  # type: ignore
    return return_dict  # type: ignore


async def _extract_column_for_concept(messages: list[AnyMessage], config: RunnableConfig) -> FieldResolution:
    """Extract a single metadata column resolution for one concept via the LLM.

    Calls the deterministic LLM with the provided message history and validates
    the output against ``FieldResolution``.  On validation error a fallback
    error ``FieldResolution`` is returned instead of raising.

    Args:
        messages: The prompt message history including the concept-specific
            extraction query.
        config: The runnable config providing access to ``Deps``.

    Returns:
        A ``FieldResolution`` object on success, or an error ``FieldResolution``
        when the LLM output fails validation.
    """
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm

    try:
        resolved: Any = FieldResolution.model_validate(
            await llm.acall_structured(messages, FieldResolution)  # type: ignore
        )
    except ValidationError as e:
        resolved = ErrorResolution(
            units=None,  # type: ignore
            extraction=None,  # type: ignore
            confidence=0.0,
            status="error",
            notes=[],
            error=f"LLM output failed validation: {e}",
        )

    return resolved


async def extract_metadata_columns(
    user_input: GEOMetadataExtractionInput,
    return_dict: dict[str, Any],
    config: RunnableConfig,
    output_root: str,
    accession_code: str,
) -> GEOMetadataExtractionResult:
    """Extract metadata columns using the per-concept LLM extraction strategy.

    Iterates over every ``Concept``, builds a concept-specific user query via
    ``generate_metadata_column_user_query``, calls ``_extract_column_for_concept``
    for each, and saves the resulting extraction protocol as a JSON artifact
    when all resolutions are sufficiently confident.

    Args:
        user_input: The GEO metadata extraction input describing the dataset and
            field requests.
        return_dict: The shared workflow state dictionary.  Messages are appended
            to ``messages`` and ``concept_messages[concept]``, and the extraction
            result is written into ``datasets[accession_code]``.
        config: The runnable config providing access to ``Deps``.
        output_root: Directory under which the ``extraction_protocol.json``
            artifact is written.
        accession_code: The GEO accession code identifying the current dataset.

    Returns:
        The updated ``return_dict`` after recording the extraction result.
    """

    def _check_execution_status(resolutions: dict[Concept, FieldResolution]) -> str:
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
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        return_dict["messages"].append(human_message)
        return_dict["concept_messages"][concept].append(human_message)
        resolutions[concept] = await _extract_column_for_concept(return_dict["concept_messages"][concept], config)

    execution_status = _check_execution_status(resolutions)
    artifact = None
    artifact_path = os.path.join(output_root, "extraction_protocol.json")
    if all((r.status in ["resolved", "missing"]) and (r.confidence > 0.6) for r in resolutions.values()):
        try:
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
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
        except Exception:
            execution_status = "failed"
            artifact = None

    metadata_extraction_result_kwargs = {
        "artifact": artifact,
        "execution_status": execution_status,
        "error": "There was an error extracting metadata columns." if execution_status == "failed" else None,
    }
    metadata_extraction_result_kwargs.update(resolutions)
    return_dict["datasets"][accession_code]["metadata_extraction_result"] = GEOMetadataExtractionResult(
        **metadata_extraction_result_kwargs  # type: ignore
    ).model_dump()
    return return_dict  # type: ignore
