__all__ = [
    "_harmonize_ontology_labels",
    "_harmonize_ontology_group_labels",
    "_harmonize_sex_labels",
    "construct_raw_to_harmonized_label_mapping",
]
import asyncio
import json
import re
import time
import unicodedata
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from ...agent.graphs.deps import Deps
from ...agent.llm.logged_client import (
    _extract_token_usage,
    _message_count,
    _message_sha256,
    _render_preview,
    _schema_name,
)
from ...contracts.harmonize import (
    BestGuessMapping,
    HumanReadableConceptInput,
    LabelMappingSet,
    create_ontology_mapping_model,
)
from ...utils.error_codes import LLM_VALIDATION_FAILURE
from ...utils.examples import (
    generate_high_level_ontology_guess_examples,
    generate_high_level_ontology_selection_examples,
    generate_ontology_guess_examples,
    generate_ontology_selection_examples,
)
from ...utils.prompting import (
    generate_high_level_ontology_label_selection_query,
    generate_ontology_group_guess_user_query,
    generate_ontology_label_query,
    generate_ontology_label_selection_query,
)

CALL_TIMEOUT = 180
GLOBAL_RETRY_LIMIT = 5

# Useful Utils

ONTOLOGY_DICT = {
    "mondo": {"ontology_name": "Mondo", "target_label": "Disease/Condition"},
    "uberon": {"ontology_name": "Uberon", "target_label": "Tissue"},
    "cl": {"ontology_name": "Cell Ontology (CL)", "target_label": "Cell Type"},
    "pato": {"ontology_name": "Phenotype And Trait Ontology (PATO)", "target_label": "sex"},
}


BEST_GUESS_NOTES = (
    "This label was not able to be mapped to the ontology, "
    "so the best guess is to keep the original label. "
    "This is likely because the label is either very noisy "
    "or represents a concept that is not well represented in the ontology."
)
CONTROL_BEST_GUESS_NOTES = "This label represents the control samples in the dataset, and is not meant to be harmonized to the ontology."


def _check_retryable(e: Exception) -> bool:
    if isinstance(e, (requests.ConnectionError, requests.Timeout, json.JSONDecodeError)):
        return True
    if isinstance(e, requests.HTTPError):
        return e.response is not None and e.response.status_code >= 500
    return False


def search_ontology_term(
    query: str,
    ontology: str = "mondo",
    k: int = 5,
    provenance: Any = None,
    max_retries: int = GLOBAL_RETRY_LIMIT,
    backoff_s: float = 1.0,
) -> list[dict[str, Any]]:
    """Search the EBI Ontology Lookup Service for terms matching a query.

    Falls back to searching DOID if no results are found in Mondo.

    Args:
        query: The search term to look up in the ontology.
        ontology: The ontology abbreviation to search (e.g. "mondo", "uberon", "cl").
        k: Maximum number of results to return.
        provenance: Optional ProvenanceLogger for event emission.
        max_retries: Maximum number of retry attempts per HTTP request.
        backoff_s: Base backoff duration in seconds (exponential backoff applied).

    Returns:
        A list of dicts, each with keys "ontology", "id" (obo_id), and "label".

    Raises:
        requests.HTTPError: If the OLS API request fails after all retries.
    """
    url = "https://www.ebi.ac.uk/ols4/api/search"

    if provenance is not None:
        provenance.emit_retrieval_query_issued(query=query, ontology=ontology, k=k, api_url=url)

    params = {"q": query, "ontology": ontology, "rows": k}

    docs = _retryable_ols_request(url, params, max_retries, backoff_s, provenance)

    if ontology == "mondo" and len(docs) == 0:
        ontology = "doid"
        params["ontology"] = "doid"
        if provenance is not None:
            provenance.emit_retrieval_query_issued(query=query, ontology=ontology, k=k, api_url=url)
        docs = _retryable_ols_request(url, params, max_retries, backoff_s, provenance)

    results = [{"ontology": ontology, "id": d.get("obo_id"), "label": d.get("label")} for d in docs]

    if provenance is not None:
        provenance.emit_retrieval_candidates_returned(
            query=query,
            ontology=ontology,
            num_candidates=len(results),
            candidate_ids=[r["id"] or "" for r in results],
            candidate_labels=[r["label"] or "" for r in results],
        )

    return results


def _retryable_ols_request(
    url: str,
    params: dict[str, Any],
    max_retries: int,
    backoff_s: float,
    provenance: Any = None,
) -> list[dict[str, Any]]:
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()["response"]["docs"]
        except Exception as e:
            if not _check_retryable(e) or attempt == total_attempts:
                if provenance is not None:
                    provenance.emit_retries_exhausted(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        total_attempts=attempt,
                        max_retries=max_retries,
                        step="search_ontology_term",
                    )
                raise
            if provenance is not None:
                provenance.emit_retry_scheduled(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    retry_count=attempt,
                    retry_limit=max_retries,
                    backoff_s=backoff_s * (2 ** (attempt - 1)),
                    step="search_ontology_term",
                )
            time.sleep(backoff_s * (2 ** (attempt - 1)))

    raise RuntimeError("Unreachable: all retry attempts should either succeed or raise")


def gather_concept_context(
    metadata_dict: dict[str, Any], extraction_protocol: dict[str, Any], unique_concept_labels: list[str]
) -> HumanReadableConceptInput:
    """Assemble dataset context for LLM-assisted ontology label guessing.

    Constructs a HumanReadableConceptInput from dataset metadata and extraction
    protocol details, providing the context needed for an LLM to suggest
    human-readable concept labels.

    Args:
        metadata_dict: Dictionary containing dataset-level metadata (title, summary,
            overall design).
        extraction_protocol: Dictionary with extraction instructions, including field
            name and optional key name.
        unique_concept_labels: Unique raw labels found in the dataset for the field.

    Returns:
        A HumanReadableConceptInput populated with dataset context and concept labels.
    """
    params = {
        "dataset_title": metadata_dict["dataset_metadata"]["title"],
        "dataset_summary": metadata_dict["dataset_metadata"]["summary"],
        "dataset_overall_design": metadata_dict["dataset_metadata"]["overall_design"],
        "metadata_field_name": extraction_protocol["extraction"]["field_name"],
        "metadata_field_key_name": extraction_protocol["extraction"].get("key_name", None),
        "concepts": unique_concept_labels,
    }
    return HumanReadableConceptInput.model_validate(params)


def slugify(value, allow_unicode=False):
    """Convert a string to a URL-safe slug.

    Adapted from Django: https://github.com/django/django/blob/stable/6.0.x/django/utils/text.py#L17

    Converts to ASCII if allow_unicode is False. Replaces spaces or repeated
    dashes with single dashes. Removes characters that are not alphanumerics,
    underscores, or hyphens. Converts to lowercase. Strips leading and
    trailing whitespace, dashes, and underscores.

    Args:
        value: The string to slugify.
        allow_unicode: If True, preserves Unicode characters.

    Returns:
        A URL-safe slug string.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


# Possibly Remove this
def create_slugified_concept_mapping(concepts: list[str]) -> dict[str, str]:
    """Create a mapping from original concept labels to their slugified forms.

    Args:
        concepts: List of concept label strings.

    Returns:
        A dict mapping each original concept label to its slugified representation.
    """
    slugified_mapping = {}
    for concept in concepts:
        slugified_mapping[concept] = slugify(concept)
    return slugified_mapping


async def call_llm_structured_with_retries(messages: list[Any], config: RunnableConfig, ResultModel: Any = None, step: str | None = None) -> Any:
    """Call an LLM with structured output, retrying on timeout or parse failure.

    Retries up to GLOBAL_RETRY_LIMIT times on timeout. On an OutputParserException,
    appends a correction message and recurses with one more retry. On a
    ValidationError, breaks and returns the current (possibly None) result.

    Args:
        messages: The message list to send to the LLM.
        config: A RunnableConfig containing a "deps" key with llm.
        ResultModel: The Pydantic model to parse the structured output into.
        step: Optional step name for provenance logging.

    Returns:
        The parsed structured output, or None if all retries or validation failed.
    """
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm
    provenance = deps.get_provenance(config["configurable"]["thread_id"]) if deps else None
    llm_cfg = getattr(llm, "config", None)
    schema_str = _schema_name(ResultModel) if ResultModel is not None else "unknown"

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    resolved: Any = None
    while retries < retry_limit:
        prompt_sha = _message_sha256(messages)
        msg_count = _message_count(messages)
        if provenance is not None:
            provenance.emit_prompt_rendered(
                template_name="call_llm_structured_with_retries",
                template_path="",
                rendered_sha256=prompt_sha,
                rendered_preview=_render_preview(messages),
                step=step,
            )
            provenance.emit_llm_request_started(
                model=getattr(llm_cfg, "model", None) if llm_cfg else None,
                provider=getattr(llm_cfg, "provider", None) if llm_cfg else None,
                temperature=getattr(llm_cfg, "temperature", None) if llm_cfg else None,
                top_p=getattr(llm_cfg, "top_p", None) if llm_cfg else None,
                top_k=getattr(llm_cfg, "top_k", None) if llm_cfg else None,
                attempt=retries + 1,
                message_count=msg_count,
                structured_output_schema=schema_str,
                prompt_sha256=prompt_sha,
                step=step,
            )
        start = time.perf_counter()
        try:
            resolved = await asyncio.wait_for(llm.acall_structured(messages, ResultModel), timeout=CALL_TIMEOUT)
            latency_ms = (time.perf_counter() - start) * 1000
            if provenance is not None:
                provenance.emit_llm_response_received(
                    latency_ms=latency_ms,
                    output_schema=schema_str,
                    parsed_success=True,
                    attempt=retries + 1,
                    step=step,
                )
                token_usage = _extract_token_usage(resolved)
                if token_usage is not None:
                    provenance.emit_llm_token_usage(
                        prompt_tokens=token_usage["prompt_tokens"],
                        completion_tokens=token_usage["completion_tokens"],
                        total_tokens=token_usage["total_tokens"],
                        step=step,
                    )
            break
        except TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            if provenance is not None:
                provenance.emit_llm_response_received(
                    latency_ms=latency_ms,
                    output_schema=schema_str,
                    parsed_success=False,
                    attempt=retries + 1,
                    step=step,
                )
                provenance.emit_retry_scheduled(
                    error_type="TimeoutError",
                    error_message="Harmonization LLM call timed out",
                    retry_count=retries + 1,
                    retry_limit=retry_limit,
                    step=step,
                )
            retries += 1
            continue
        except OutputParserException as e:
            latency_ms = (time.perf_counter() - start) * 1000
            if provenance is not None:
                provenance.emit_llm_response_received(
                    latency_ms=latency_ms,
                    output_schema=schema_str,
                    parsed_success=False,
                    attempt=retries + 1,
                    step=step,
                )
                provenance.emit_retry_scheduled(
                    error_type="OutputParserException",
                    error_message=str(e)[:500],
                    retry_count=retries + 1,
                    retry_limit=retry_limit,
                    step=step,
                )
            human_message = HumanMessage(
                id=uuid.uuid4().hex,
                content=(
                    f"The previous output from the LLM failed to parse with error: {e}. "
                    "Please reformat the output to match the expected format and ensure "
                    "that all required fields are included."
                ),
                additional_kwargs={
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            messages = messages + [human_message]
            retries += 1
            continue
        except ValidationError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            if provenance is not None:
                provenance.emit_llm_response_received(
                    latency_ms=latency_ms,
                    output_schema=schema_str,
                    parsed_success=False,
                    attempt=retries + 1,
                    step=step,
                )
                provenance.emit_error_raised(
                    error_type="ValidationError",
                    error_message=str(e)[:500],
                    error_code=LLM_VALIDATION_FAILURE,
                    step=step,
                )
            print(f"\n\nValidation error: {e}. Setting resolution to error with notes.")
            break

    if resolved is None and provenance is not None:
        provenance.emit_retries_exhausted(
            error_type="ExhaustedRetries",
            total_attempts=retry_limit,
            max_retries=retry_limit,
            step=step,
        )
    return resolved


async def _guess_human_readable_labels(
    guess_input: HumanReadableConceptInput,
    guess_result: Any,
    config: RunnableConfig,
    ontology: str = "mondo",
    ontology_literal: str = "mondo",
) -> Any:
    """Use an LLM to guess human-readable labels from raw concept labels.

    Generates example-guided system prompt and dataset-contextualized query
    to suggest human-readable labels for each raw concept, suitable for ontology
    search.

    Args:
        guess_input: Dataset context and raw concept labels.
        guess_result: Dynamic Pydantic model used for structured output parsing.
        config: RunnableConfig with LLM dependencies.
        ontology: Ontology abbreviation (e.g. "mondo", "uberon", "cl").
        ontology_literal: Ontology literal string for the dynamic model discriminator.

    Returns:
        A LabelMappingSet with the LLM's suggested human-readable mappings.
    """
    system_message = generate_ontology_guess_examples(ontology=ontology)
    template_input_labels = [f"'{label}'" for label in guess_input.concepts]
    query_prompt = generate_ontology_label_query(
        dataset_title=guess_input.dataset_title,
        dataset_summary=guess_input.dataset_summary,
        dataset_overall_design=guess_input.dataset_overall_design,
        metadata_field_name=guess_input.metadata_field_name,
        metadata_field_key_name=guess_input.metadata_field_key_name if guess_input.metadata_field_key_name is not None else "N/A",
        target_label=ONTOLOGY_DICT[ontology]["target_label"],
        ontology_name=ONTOLOGY_DICT[ontology]["ontology_name"],
        labels=", ".join(template_input_labels),
        json_schema=guess_result.model_json_schema(),
    )
    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=query_prompt,
        additional_kwargs={
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    _, LabelMappingSetDyn = create_ontology_mapping_model(
        guess_input.concepts, ontology_literal, ontology, allowed_target_labels=None, high_level=False
    )
    resolved = await call_llm_structured_with_retries([system_message, human_message], config, ResultModel=LabelMappingSetDyn)
    return resolved


async def _guess_human_readable_high_level_labels(
    input_labels: list[str],
    guess_result: Any,
    config: RunnableConfig,
    ontology: str = "mondo",
    ontology_literal: str = "mondo",
) -> Any:
    """Use an LLM to guess high-level category labels from harmonized labels.

    Similar to _guess_human_readable_labels but operates on already-harmonized
    labels to suggest broader, high-level categories for grouping purposes.

    Args:
        input_labels: List of harmonized label strings to categorize.
        guess_result: Dynamic Pydantic model used for structured output parsing.
        config: RunnableConfig with LLM dependencies.
        ontology: Ontology abbreviation (e.g. "mondo", "uberon", "cl").
        ontology_literal: Ontology literal string for the dynamic model discriminator.

    Returns:
        A LabelMappingSet with the LLM's suggested high-level categorical mappings.
    """
    system_message = generate_high_level_ontology_guess_examples(ontology=ontology)
    template_input_labels = [f"'{label}'" for label in input_labels]
    query_prompt = generate_ontology_group_guess_user_query(
        target_label=ONTOLOGY_DICT[ontology]["target_label"],
        ontology_name=ONTOLOGY_DICT[ontology]["ontology_name"],
        labels=", ".join(template_input_labels),
        json_schema=guess_result.model_json_schema(),
    )
    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=query_prompt,
        additional_kwargs={
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    _, LabelMappingSetDyn = create_ontology_mapping_model(input_labels, ontology_literal, ontology, allowed_target_labels=None, high_level=True)
    resolved = await call_llm_structured_with_retries([system_message, human_message], config, ResultModel=LabelMappingSetDyn)
    return resolved


async def _select_best_ontology_labels(
    ontology_label_dict: dict[str, Any],
    guess_input: HumanReadableConceptInput,
    guess_result: Any,
    config: RunnableConfig,
    ontology: str = "mondo",
) -> Any:
    """Use an LLM to select the best ontology label from candidate choices.

    Presents candidate labels for each suggested human-readable label to the LLM
    and asks it to choose the most appropriate ontology term based on dataset
    context.

    Args:
        ontology_label_dict: Dict mapping suggested labels to lists of candidate
            ontology labels.
        guess_input: Dataset context for informed selection.
        guess_result: Dynamic Pydantic model used for structured output parsing.
        config: RunnableConfig with LLM dependencies.
        ontology: Ontology abbreviation (e.g. "mondo", "uberon", "cl").

    Returns:
        A LabelMappingSet with the LLM's best-choice ontology label selections.
    """
    system_message = generate_ontology_selection_examples(ontology=ontology)
    table = pd.DataFrame([{"Label": key, "Candidate Mondo Labels": value} for key, value in ontology_label_dict.items()])
    query_prompt = generate_ontology_label_selection_query(
        dataset_title=guess_input.dataset_title,
        dataset_summary=guess_input.dataset_summary,
        dataset_overall_design=guess_input.dataset_overall_design,
        target_label=ONTOLOGY_DICT[ontology]["target_label"],
        ontology_name=ONTOLOGY_DICT[ontology]["ontology_name"],
        input=table.to_markdown(index=False),
        json_schema=guess_result.model_json_schema(),
    )

    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=query_prompt,
        additional_kwargs={
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    resolved = await call_llm_structured_with_retries([system_message, human_message], config, ResultModel=guess_result)
    return resolved


async def _select_best_high_level_ontology_labels(
    ontology_label_dict: dict[str, Any], guess_result: Any, config: RunnableConfig, ontology: str = "mondo"
) -> Any:
    """Use an LLM to select the best high-level ontology category from candidates.

    Presents candidate high-level labels to the LLM and asks it to choose the
    most appropriate category for each.

    Args:
        ontology_label_dict: Dict mapping labels to lists of candidate high-level
            ontology categories.
        guess_result: Dynamic Pydantic model used for structured output parsing.
        config: RunnableConfig with LLM dependencies.
        ontology: Ontology abbreviation (e.g. "mondo", "uberon", "cl").

    Returns:
        A LabelMappingSet with the LLM's best-choice high-level category selections.
    """
    system_message = generate_high_level_ontology_selection_examples(ontology=ontology)
    table = pd.DataFrame([{"Label": key, "Candidate Mondo Labels": value} for key, value in ontology_label_dict.items()])
    query_prompt = generate_high_level_ontology_label_selection_query(
        target_label=ONTOLOGY_DICT[ontology]["target_label"],
        ontology_name=ONTOLOGY_DICT[ontology]["ontology_name"],
        input=table.to_markdown(index=False),
        json_schema=guess_result.model_json_schema(),
    )

    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=query_prompt,
        additional_kwargs={
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    resolved = await call_llm_structured_with_retries([system_message, human_message], config, ResultModel=guess_result)
    return resolved


async def _harmonize_ontology_labels(
    metadata_dict: dict[str, Any],
    extraction_protocol: dict[str, Any],
    sample_metadata: pd.DataFrame,
    config: RunnableConfig,
    ontology: str = "mondo",
    ontology_literal: str = "mondo",
    column_name: str = "disease_status",
) -> tuple[LabelMappingSet, LabelMappingSet, LabelMappingSet]:
    """Harmonize raw dataset labels to ontology terms via LLM-guided mapping.

    Orchestrates a pipeline that (1) guesses human-readable labels from raw
    values, (2) searches OLS for candidate ontology terms, (3) uses an LLM to
    select the best ontology label per suggestion, and (4) handles control labels
    and un-mappable labels as best-guess fallbacks.

    Args:
        metadata_dict: Dataset-level metadata including title, summary, and design.
        extraction_protocol: Extraction protocol with field name, optional key name,
            and optional control_value for Mondo harmonization.
        sample_metadata: DataFrame containing sample-level metadata.
        config: RunnableConfig with LLM dependencies.
        ontology: Ontology abbreviation (e.g. "mondo", "uberon", "cl").
        ontology_literal: Ontology literal for the dynamic model discriminator.
        column_name: Column name in sample_metadata to harmonize.

    Returns:
        A tuple of (human_readable_ontology_labels, label_to_top_n_ontology,
        ontology_label_selection). The first is the LLM's human-readable guesses,
        the second is a dict of suggested labels to OLS search results, and the
        third is the LLM's best-choice ontology selections.
    """
    control_label = None
    unique_dataset_labels = sorted([str(x) for x in sample_metadata[column_name].unique()])
    if ontology == "mondo":
        control_label = extraction_protocol[column_name.lower()]["extraction"].get("control_value", None)
        if control_label is None:
            control_label = ""
        unique_dataset_labels = [label for label in unique_dataset_labels if label.lower() != control_label.lower()]

    deps = config["configurable"]["deps"]
    provenance = deps.get_provenance(config["configurable"]["thread_id"]) if deps else None

    if not unique_dataset_labels:
        human_readable_ontology_labels = LabelMappingSet(mappings=[])
        ontology_label_selection = LabelMappingSet(mappings=[])
        control_mapping = BestGuessMapping.model_validate(
            {
                "ontology": "best_guess",
                "source_label": control_label,
                "target_label": "Control",
                "notes": BEST_GUESS_NOTES,
            }
        )
        other_mapping = BestGuessMapping.model_validate(
            {"ontology": "best_guess", "source_label": "Control", "target_label": "Control", "notes": BEST_GUESS_NOTES}
        )
        human_readable_ontology_labels.mappings.append(control_mapping)
        ontology_label_selection.mappings.append(other_mapping)
        return human_readable_ontology_labels, {}, ontology_label_selection  # type: ignore

    dataset_context = gather_concept_context(metadata_dict, extraction_protocol[column_name.lower()], unique_dataset_labels)
    _, DatasetLabelMappingSetDyn = create_ontology_mapping_model(
        unique_dataset_labels, ontology_literal, ontology, allowed_target_labels=None, high_level=False
    )

    # Guess human readable disease labels
    human_readable_ontology_labels = await _guess_human_readable_labels(dataset_context, DatasetLabelMappingSetDyn, config, ontology=ontology)
    if provenance is not None:
        for m in human_readable_ontology_labels.mappings:
            if m.ontology != "missing":
                provenance.emit_harmonization_mapping_proposed(
                    source_label=m.source_label,
                    target_label=m.target_label,
                    ontology=ontology,
                    mapping_type="guess",
                )
    if control_label not in ["", None]:
        human_readable_ontology_labels.mappings.append(
            BestGuessMapping.model_validate(
                {
                    "ontology": "best_guess",
                    "source_label": control_label,
                    "target_label": "Control",
                    "notes": CONTROL_BEST_GUESS_NOTES,
                }
            )
        )
    # TODO: Add control back to guess
    suggested_human_readable_ontology_labels = [
        x.target_label for x in human_readable_ontology_labels.mappings if x.ontology != "missing" and x.target_label != "Control"
    ]
    print(f"\nSuggested human-readable labels: {suggested_human_readable_ontology_labels}")
    if not suggested_human_readable_ontology_labels:
        print(f"\n No labels: {json.dumps(human_readable_ontology_labels.model_dump(), indent=2)}")

    # Get best ontology label
    label_to_top_n_ontology = {
        suggested_label: search_ontology_term(suggested_label, ontology=ontology, k=5, provenance=provenance)
        for suggested_label in suggested_human_readable_ontology_labels
    }
    missing_labels = [x.source_label for x in human_readable_ontology_labels.mappings if x.ontology == "missing"]
    # Get missing
    for label, entries in label_to_top_n_ontology.items():
        if len(entries) == 0:
            missing_labels.append(label)
    print(f"\nMissing labels: {missing_labels}")

    if len(missing_labels) != len(unique_dataset_labels):
        if control_label is not None:
            missing_labels.append(control_label)
        for label in missing_labels:
            label_to_top_n_ontology.pop(label, None)
        _, OntologyGroupLabelMappingSetDyn = create_ontology_mapping_model(
            suggested_human_readable_ontology_labels,
            ontology_literal,
            ontology,
            allowed_target_labels=[entry["label"] for label, entries in label_to_top_n_ontology.items() for entry in entries],
            high_level=False,
        )
        ontology_label_dict = {label: [f"'{entry}'" for entry in entries] for label, entries in label_to_top_n_ontology.items()}
        ontology_label_selection = await _select_best_ontology_labels(
            ontology_label_dict, dataset_context, OntologyGroupLabelMappingSetDyn, config, ontology=ontology
        )
        ontology_label_selection = LabelMappingSet.model_validate(ontology_label_selection.model_dump())
        if provenance is not None:
            for m in ontology_label_selection.mappings:
                if hasattr(m, "target_label"):
                    provenance.emit_harmonization_mapping_proposed(
                        source_label=m.source_label,
                        target_label=m.target_label,  # type: ignore[union-attr]
                        ontology=ontology,
                        mapping_type="selection",
                    )
    else:
        ontology_label_selection = LabelMappingSet(mappings=[])
    # Add missing labels as best guess mappings
    for missing_label in missing_labels:
        target_label = missing_label
        if missing_label == control_label:
            target_label = "Control"
            missing_label = target_label
        ontology_label_selection.mappings.append(
            BestGuessMapping.model_validate(
                {
                    "ontology": "best_guess",
                    "source_label": missing_label,
                    "target_label": target_label,
                    "notes": BEST_GUESS_NOTES,
                }
            )
        )
        if provenance is not None:
            provenance.emit_harmonization_mapping_proposed(
                source_label=missing_label,
                target_label=target_label,
                ontology=ontology,
                mapping_type="best_guess",
            )
    return human_readable_ontology_labels, label_to_top_n_ontology, ontology_label_selection  # type: ignore


async def _harmonize_ontology_group_labels(
    harmonized_labels: list[str], config: RunnableConfig, ontology: str = "mondo", ontology_literal: str = "mondo"
) -> tuple[LabelMappingSet, LabelMappingSet, LabelMappingSet]:
    """Harmonize already-harmonized labels into high-level ontology categories.

    Takes a list of already-harmonized ontology labels and maps them to broader,
    high-level categories. Uses LLM-guided guessing for high-level labels, OLS
    search for candidate categories, and LLM selection of the best match.

    Args:
        harmonized_labels: List of previously harmonized label strings.
        config: RunnableConfig with LLM dependencies.
        ontology: Ontology abbreviation (e.g. "mondo", "uberon", "cl").
        ontology_literal: Ontology literal for the dynamic model discriminator.

    Returns:
        A tuple of (human_readable_ontology_labels, label_to_top_n_ontology,
        ontology_label_selection). The first contains the LLM's high-level
        guesses, the second maps labels to OLS search results, and the third
        contains the LLM's best-category selections.
    """
    unique_harmonized_labels = sorted(set(harmonized_labels))
    deps = config["configurable"].get("deps")
    provenance = deps.get_provenance(config["configurable"]["thread_id"]) if deps else None
    _, DatasetLabelMappingSetDyn = create_ontology_mapping_model(
        unique_harmonized_labels, ontology_literal, ontology, allowed_target_labels=None, high_level=True
    )

    # Guess human readable disease labels
    human_readable_ontology_labels = await _guess_human_readable_high_level_labels(
        unique_harmonized_labels,
        DatasetLabelMappingSetDyn,
        config,
        ontology=ontology,
        ontology_literal=ontology_literal,
    )
    if provenance is not None:
        for m in human_readable_ontology_labels.mappings:
            if m.ontology != "missing":
                provenance.emit_harmonization_mapping_proposed(
                    source_label=m.source_label,
                    target_label=m.target_label,
                    ontology=ontology,
                    mapping_type="guess",
                )

    suggested_human_readable_ontology_labels = [x.target_label for x in human_readable_ontology_labels.mappings if x.ontology != "missing"]

    # Get best ontology label
    label_to_top_n_ontology = {
        suggested_label: search_ontology_term(suggested_label, ontology=ontology, k=5, provenance=provenance)
        for suggested_label in suggested_human_readable_ontology_labels
    }
    missing_labels = [x.source_label for x in human_readable_ontology_labels.mappings if x.ontology == "missing"]
    # Get missing
    for label, entries in label_to_top_n_ontology.items():
        if len(entries) == 0:
            missing_labels.append(label)
    for label in missing_labels:
        label_to_top_n_ontology.pop(label, None)

    _, OntologyGroupLabelMappingSetDyn = create_ontology_mapping_model(
        suggested_human_readable_ontology_labels,
        ontology_literal,
        ontology,
        allowed_target_labels=[entry["label"] for label, entries in label_to_top_n_ontology.items() for entry in entries],
        high_level=True,
    )
    ontology_label_dict = {label: [entry["label"] for entry in entries] for label, entries in label_to_top_n_ontology.items()}
    ontology_label_dict = {label: [f"'{entry}'" for entry in entries] for label, entries in label_to_top_n_ontology.items()}
    ontology_label_selection = await _select_best_high_level_ontology_labels(
        ontology_label_dict, OntologyGroupLabelMappingSetDyn, config, ontology=ontology
    )
    ontology_label_selection = LabelMappingSet.model_validate(ontology_label_selection.model_dump())
    if provenance is not None:
        for m in ontology_label_selection.mappings:
            if hasattr(m, "target_label"):
                provenance.emit_harmonization_mapping_proposed(
                    source_label=m.source_label,
                    target_label=m.target_label,  # type: ignore[union-attr]
                    ontology=ontology,
                    mapping_type="selection",
                )
    # Add missing labels as best guess mappings
    for missing_label in missing_labels:
        ontology_label_selection.mappings.append(
            BestGuessMapping.model_validate(
                {
                    "ontology": "best_guess",
                    "source_label": missing_label,
                    "target_label": "Unknown/Other",
                    "notes": BEST_GUESS_NOTES,
                }
            )
        )
        if provenance is not None:
            provenance.emit_harmonization_mapping_proposed(
                source_label=missing_label,
                target_label="Unknown/Other",
                ontology=ontology,
                mapping_type="best_guess",
            )
    # TODO: Possibly set ontology of ontology_label_selection result to whatevers returned by the ols
    return human_readable_ontology_labels, label_to_top_n_ontology, ontology_label_selection  # type: ignore


async def _harmonize_sex_labels(
    metadata_dict: dict[str, Any],
    extraction_protocol: dict[str, Any],
    sample_metadata: pd.DataFrame,
    config: RunnableConfig,
) -> tuple[LabelMappingSet, LabelMappingSet, LabelMappingSet]:
    """Harmonize raw sex labels to "Male", "Female", or "Unknown/Other".

    Uses the PATO ontology context to map raw sex labels to a fixed set of
    harmonized values via LLM-guided selection.

    Args:
        metadata_dict: Dataset-level metadata including title, summary, and design.
        extraction_protocol: Extraction protocol with field details for the sex column.
        sample_metadata: DataFrame containing sample-level metadata.
        config: RunnableConfig with LLM dependencies.

    Returns:
        A tuple of (unique_dataset_labels, ontology_label_dict,
        ontology_label_selection). The first is the sorted unique raw labels,
        the second maps raw labels to the harmonized ["Male", "Female",
        "Unknown/Other"] options, and the third contains the LLM's selections.
    """
    column_name = "Sex"
    unique_dataset_labels = sorted([str(x) for x in sample_metadata[column_name].unique()])
    deps = config["configurable"].get("deps")
    provenance = deps.get_provenance(config["configurable"]["thread_id"]) if deps else None
    dataset_context = gather_concept_context(metadata_dict, extraction_protocol[column_name.lower()], unique_dataset_labels)
    harmonized_labels = ["Male", "Female", "Unknown/Other"]
    ontology_label_dict = {label: harmonized_labels for label in unique_dataset_labels}
    _, OntologyGroupLabelMappingSetDyn = create_ontology_mapping_model(
        unique_dataset_labels,
        "pato",
        "Phenotype And Trait Ontology (PATO)",
        allowed_target_labels=harmonized_labels,
        high_level=False,
    )
    ontology_label_selection = await _select_best_ontology_labels(
        ontology_label_dict, dataset_context, OntologyGroupLabelMappingSetDyn, config, ontology="pato"
    )
    if provenance is not None:
        for m in ontology_label_selection.mappings:
            provenance.emit_harmonization_mapping_proposed(
                source_label=m.source_label,
                target_label=m.target_label,
                ontology="pato",
                mapping_type="selection",
            )
    return unique_dataset_labels, ontology_label_dict, ontology_label_selection  # type: ignore


def construct_raw_to_harmonized_label_mapping(guessed_ontology_labels: LabelMappingSet, ontology_label_selection: LabelMappingSet) -> LabelMappingSet:
    """Build the final raw-to-harmonized label mapping from two LLM outputs.

    Joins the human-readable guesses with the best ontology selections to
    produce a single LabelMappingSet that maps each raw source label to its
    final harmonized target label. Labels that could not be mapped are preserved
    as best-guess fallbacks.

    Args:
        guessed_ontology_labels: Human-readable label guesses from the LLM.
        ontology_label_selection: Best ontology label selections from the LLM.

    Returns:
        A LabelMappingSet with finalized raw-to-harmonized mappings.
    """
    fixed_harmonized_label_mapping = {"mappings": []}
    for mapping in guessed_ontology_labels.mappings:
        if mapping.ontology == "missing":
            fixed_harmonized_label_mapping["mappings"].append(
                {
                    "ontology": "best_guess",
                    "source_label": mapping.source_label,
                    "target_label": mapping.source_label,
                    "notes": BEST_GUESS_NOTES,
                }
            )
            continue
        best_guess_mapping = next(
            (m for m in ontology_label_selection.mappings if m.source_label == mapping.target_label),  # ty: ignore
            None,
        )
        if best_guess_mapping is None:
            fixed_harmonized_label_mapping["mappings"].append(
                {
                    "ontology": "best_guess",
                    "source_label": mapping.source_label,
                    "target_label": mapping.source_label,
                    "notes": BEST_GUESS_NOTES,
                }
            )
        else:
            fixed_harmonized_label_mapping["mappings"].append(
                {
                    "ontology": best_guess_mapping.ontology,
                    "source_label": mapping.source_label,
                    "target_label": best_guess_mapping.target_label if hasattr(best_guess_mapping, "target_label") else mapping.source_label,
                    "notes": best_guess_mapping.notes,
                }
            )
    return LabelMappingSet.model_validate(fixed_harmonized_label_mapping)
