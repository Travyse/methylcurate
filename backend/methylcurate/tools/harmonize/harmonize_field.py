__all__ = ["_harmonize_ontology_labels", "_harmonize_ontology_group_labels", "_harmonize_sex_labels", "construct_raw_to_harmonized_label_mapping"]
import re
import json
import uuid
import asyncio
import requests
import unicodedata
import pandas as pd
from datetime import datetime, timezone
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError
from typing import Dict, Any, List, get_args
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage
from ...agent.graphs.deps import Deps
from ...contracts.harmonize import (
    HumanReadableConceptInput,
    create_ontology_mapping_model,
    LabelMappingSet,
    MissingMapping,
    BestGuessMapping
)
from ...utils.examples import (
    generate_high_level_ontology_guess_examples, 
    generate_ontology_guess_examples,
    generate_ontology_selection_examples,
    generate_high_level_ontology_selection_examples)
from ...agent.state.models import HarmonizationSubgraphState
from ...utils.prompting import (
    generate_ontology_label_query,
    generate_ontology_group_guess_user_query,
    generate_ontology_label_selection_query,
    generate_high_level_ontology_label_selection_query,
    generate_high_level_ontology_label_selection_system_prompt
)

CALL_TIMEOUT = 180
GLOBAL_RETRY_LIMIT = 5

# Useful Utils

ONTOLOGY_DICT = {
    'mondo': {
        'ontology_name': 'Mondo',
        'target_label': 'Disease/Condition'
    },
    'uberon': {
        'ontology_name': 'Uberon',
        'target_label': 'Tissue'
    },
    'cl': {
        'ontology_name': 'Cell Ontology (CL)',
        'target_label': 'Cell Type'
    },
    'pato': {
        'ontology_name': 'Phenotype And Trait Ontology (PATO)',
        'target_label': 'sex'
    }
}

def search_ols(query, ontology="mondo", k=5):
    url = "https://www.ebi.ac.uk/ols4/api/search"
    
    params = {
        "q": query,
        "ontology": ontology,
        "rows": k
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    
    docs = r.json()["response"]["docs"]

    if ontology == "mondo" and len(docs) == 0:
        # Try searching DOID as backup
        ontology = "doid"
        params["ontology"] = "doid"
        r = requests.get(url, params=params)
        r.raise_for_status()
        docs = r.json()["response"]["docs"]
    
    return [
        {
            "ontology": ontology,
            "id": d.get("obo_id"),
            "label": d.get("label")
        }
        for d in docs
    ]

def gather_concept_context(metadata_dict: Dict[str, Any], extraction_protocol: Dict[str, Any], unique_concept_labels: List[str]) -> HumanReadableConceptInput:
    # Get metadata description
    # Get unique metadata values for a given concept
    # Get field name and key name if applicable
    params = {
        "dataset_title": metadata_dict["dataset_metadata"]["title"],
        "dataset_summary": metadata_dict["dataset_metadata"]["summary"],
        "dataset_overall_design": metadata_dict["dataset_metadata"]["overall_design"],
        "metadata_field_name": extraction_protocol["extraction"]["field_name"],
        "metadata_field_key_name": extraction_protocol["extraction"].get("key_name", None),
        "concepts": unique_concept_labels
    }
    return HumanReadableConceptInput.model_validate(params)

def slugify(value, allow_unicode=False):
    """
    Taken From Django: https://github.com/django/django/blob/stable/6.0.x/django/utils/text.py#L17
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")

# Possibly Remove this
def create_slugified_concept_mapping(concepts: List[str]) -> Dict[str, str]:
    # TODO Use this when creating the models
    slugified_mapping = {}
    for concept in concepts:
        slugified_mapping[concept] = slugify(concept)
    return slugified_mapping

async def llm_conversation(
    messages: List[Any], config: RunnableConfig, ResultModel: Any = None) -> Any:
    deps: Deps = config["configurable"]["deps"]
    deterministic_llm = deps.deterministic_llm
    default_llm = deps.default_llm

    retry_limit = GLOBAL_RETRY_LIMIT
    retries = 0
    resolved = None
    while retries < retry_limit:
        try:
            resolved: Any = await asyncio.wait_for(
                deterministic_llm.acall_structured(messages, ResultModel), timeout=CALL_TIMEOUT)
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
            messages += [human_message]
            retries = 4
            return await llm_conversation(messages, config, ResultModel)
        except ValidationError as e:
            print(f"\n\nValidation error for concept disease_status: {e}. Setting resolution to error with notes.")
            break
            
    return resolved


    #generate_high_level_ontology_guess_examples, 
    #generate_ontology_guess_examples,
    #generate_ontology_selection_examples

async def _guess_human_readable_labels(
        guess_input: HumanReadableConceptInput, guess_result: Any, config: RunnableConfig,
        ontology: str = "mondo", ontology_literal: str = "mondo") -> Dict[str, str]:
    system_message = generate_ontology_guess_examples(ontology=ontology)
    template_input_labels = [f"'{label}'" for label in guess_input.concepts]    
    query_prompt = generate_ontology_label_query(
        dataset_title=guess_input.dataset_title,
        dataset_summary=guess_input.dataset_summary,
        dataset_overall_design=guess_input.dataset_overall_design,
        metadata_field_name=guess_input.metadata_field_name,
        metadata_field_key_name=guess_input.metadata_field_key_name if guess_input.metadata_field_key_name is not None else "N/A",
        target_label=ONTOLOGY_DICT[ontology]['target_label'],
        ontology_name=ONTOLOGY_DICT[ontology]['ontology_name'],
        labels=", ".join(template_input_labels),
        json_schema=guess_result.model_json_schema())
    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=query_prompt,
        additional_kwargs={
            'created_at': datetime.now(timezone.utc).isoformat(),
        })
    _, LabelMappingSetDyn = create_ontology_mapping_model(
        guess_input.concepts,
        ontology_literal,
        ontology,
        allowed_target_labels = None,
        high_level = False) 
    resolved = await llm_conversation(
        [system_message, human_message], config, ResultModel = LabelMappingSetDyn)
    return resolved

async def _guess_human_readable_high_level_labels(
        input_labels: List[str], guess_result: Any, config: RunnableConfig,
        ontology: str = "mondo", ontology_literal: str = "mondo") -> Dict[str, str]:
    system_message = generate_high_level_ontology_guess_examples(ontology=ontology)
    template_input_labels = [f"'{label}'" for label in input_labels]
    query_prompt = generate_ontology_group_guess_user_query(
        target_label=ONTOLOGY_DICT[ontology]['target_label'],
        ontology_name=ONTOLOGY_DICT[ontology]['ontology_name'],
        labels=", ".join(template_input_labels),
        json_schema=guess_result.model_json_schema())
    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=query_prompt,
        additional_kwargs={
            'created_at': datetime.now(timezone.utc).isoformat(),
        })
    _, LabelMappingSetDyn = create_ontology_mapping_model(
        input_labels,
        ontology_literal,
        ontology,
        allowed_target_labels = None,
        high_level = True) 
    resolved = await llm_conversation(
        [system_message, human_message], config, ResultModel = LabelMappingSetDyn)
    return resolved

async def _select_best_ontology_labels(
        ontology_label_dict: Dict[str, Any], guess_input: HumanReadableConceptInput, guess_result: Any, config: RunnableConfig,
        ontology: str = "mondo") -> Dict[str, str]:
    system_message = generate_ontology_selection_examples(ontology=ontology)
    table = pd.DataFrame([
        {"Label": key, "Candidate Mondo Labels": value } for key, value in ontology_label_dict.items()
    ])
    query_prompt = generate_ontology_label_selection_query(
        dataset_title=guess_input.dataset_title,
        dataset_summary=guess_input.dataset_summary,
        dataset_overall_design=guess_input.dataset_overall_design,
        target_label=ONTOLOGY_DICT[ontology]['target_label'],
        ontology_name=ONTOLOGY_DICT[ontology]['ontology_name'],
        input=table.to_markdown(index=False),
        json_schema=guess_result.model_json_schema())

    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=query_prompt,
        additional_kwargs={
            'created_at': datetime.now(timezone.utc).isoformat(),
        })
    resolved = await llm_conversation(
        [system_message, human_message], config, ResultModel = guess_result)
    return resolved

async def _select_best_high_level_ontology_labels(
        ontology_label_dict: Dict[str, Any], guess_result: Any, config: RunnableConfig,
        ontology: str = "mondo") -> Dict[str, str]:
    system_message = generate_high_level_ontology_selection_examples(ontology=ontology)
    table = pd.DataFrame([
        {"Label": key, "Candidate Mondo Labels": value } for key, value in ontology_label_dict.items()
    ])
    query_prompt = generate_high_level_ontology_label_selection_query(
        target_label=ONTOLOGY_DICT[ontology]['target_label'],
        ontology_name=ONTOLOGY_DICT[ontology]['ontology_name'],
        input=table.to_markdown(index=False),
        json_schema=guess_result.model_json_schema())

    human_message = HumanMessage(
        id=uuid.uuid4().hex,
        content=query_prompt,
        additional_kwargs={
            'created_at': datetime.now(timezone.utc).isoformat(),
        })
    resolved = await llm_conversation(
        [system_message, human_message], config, ResultModel = guess_result)
    return resolved

async def _harmonize_ontology_labels(
        metadata_dict: Dict[str, Any], extraction_protocol: Dict[str, Any], sample_metadata: pd.DataFrame, config: RunnableConfig,
        ontology: str = "mondo", ontology_literal: str = "mondo", column_name: str = "disease_status"):
    control_label = None
    unique_dataset_labels = sorted([str(x) for x in sample_metadata[column_name].unique()])
    if ontology == "mondo":
        control_label = extraction_protocol[column_name.lower()]["extraction"].get("control_value", None)
        if control_label is None:
            control_label = ""
        unique_dataset_labels = [label for label in unique_dataset_labels if label.lower() != control_label.lower()]

    if not unique_dataset_labels:
        human_readable_ontology_labels = LabelMappingSet(mappings=[])
        ontology_label_selection = LabelMappingSet(mappings=[])
        control_mapping = BestGuessMapping.model_validate({
            "ontology": "best_guess",
            "source_label": control_label,
            "target_label": "Control",
            "notes": "This label was not able to be mapped to the ontology, so the best guess is to keep the original label. This is likely because the label is either very noisy or represents a concept that is not well represented in the ontology."
        })
        other_mapping = BestGuessMapping.model_validate({
            "ontology": "best_guess",
            "source_label": "Control",
            "target_label": "Control",
            "notes": "This label was not able to be mapped to the ontology, so the best guess is to keep the original label. This is likely because the label is either very noisy or represents a concept that is not well represented in the ontology."
        })
        human_readable_ontology_labels.mappings.append(control_mapping)
        ontology_label_selection.mappings.append(other_mapping)
        return human_readable_ontology_labels, {}, ontology_label_selection

    dataset_context = gather_concept_context(metadata_dict, extraction_protocol[column_name.lower()], unique_dataset_labels)
    _, DatasetLabelMappingSetDyn = create_ontology_mapping_model(
        unique_dataset_labels,
        ontology_literal,
        ontology,
        allowed_target_labels = None,
        high_level = False)

    # Guess human readable disease labels
    human_readable_ontology_labels = await _guess_human_readable_labels(
        dataset_context, DatasetLabelMappingSetDyn, config, ontology = ontology)
    if control_label not in ["", None]:
        human_readable_ontology_labels.mappings.append(
            BestGuessMapping.model_validate({
                "ontology": "best_guess",
                "source_label": control_label,
                "target_label": "Control",
                "notes": "This label represents the control samples in the dataset, and is not meant to be harmonized to the ontology."
            })
        )
    # TODO: Add control back to guess
    suggested_human_readable_ontology_labels = [
        x.target_label for x in human_readable_ontology_labels.mappings if x.ontology != "missing" and x.target_label != "Control"]
    print(f"\nSuggested human-readable labels: {suggested_human_readable_ontology_labels}")
    if not suggested_human_readable_ontology_labels:
        print(f"\n No labels: {json.dumps(human_readable_ontology_labels.model_dump(), indent=2)}")

    # Get best ontology label
    label_to_top_n_ontology = {
        suggested_label: search_ols(suggested_label, ontology=ontology, k=5) for suggested_label in suggested_human_readable_ontology_labels
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
            allowed_target_labels = [entry['label'] for label, entries in label_to_top_n_ontology.items() for entry in entries],
            high_level = False)
        ontology_label_dict = {label: [entry['label'] for entry in entries] for label, entries in label_to_top_n_ontology.items()}
        ontology_label_dict = {label: [f"'{entry}'" for entry in entries] for label, entries in label_to_top_n_ontology.items()}
        ontology_label_selection = await _select_best_ontology_labels(
            ontology_label_dict, dataset_context, OntologyGroupLabelMappingSetDyn, config, ontology = ontology)
        ontology_label_selection = LabelMappingSet.model_validate(ontology_label_selection.model_dump())
    else:
        ontology_label_selection = LabelMappingSet(mappings=[])
    # Add missing labels as best guess mappings
    for missing_label in missing_labels:
        target_label = missing_label
        if missing_label == control_label:
            target_label = "Control"
            missing_label = target_label
        ontology_label_selection.mappings.append(
            BestGuessMapping.model_validate({
                "ontology": "best_guess",
                "source_label": missing_label,
                "target_label": target_label,
                "notes": "This label was not able to be mapped to the ontology, so the best guess is to keep the original label. This is likely because the label is either very noisy or represents a concept that is not well represented in the ontology."
            })
        )
    return human_readable_ontology_labels, label_to_top_n_ontology, ontology_label_selection

async def _harmonize_ontology_group_labels(
        harmonized_labels: List[str], config: RunnableConfig, ontology: str = "mondo", ontology_literal: str = "mondo"):
    unique_harmonized_labels = sorted(set(harmonized_labels))
    _, DatasetLabelMappingSetDyn = create_ontology_mapping_model(
        unique_harmonized_labels,
        ontology_literal,
        ontology,
        allowed_target_labels = None,
        high_level = True)

    # Guess human readable disease labels
    human_readable_ontology_labels = await _guess_human_readable_high_level_labels(
        unique_harmonized_labels, DatasetLabelMappingSetDyn, config, ontology = ontology, ontology_literal = ontology_literal)

    suggested_human_readable_ontology_labels = [
        x.target_label for x in human_readable_ontology_labels.mappings if x.ontology != "missing"]

    # Get best ontology label
    label_to_top_n_ontology = {
        suggested_label: search_ols(suggested_label, ontology=ontology, k=5) for suggested_label in suggested_human_readable_ontology_labels
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
        allowed_target_labels = [entry['label'] for label, entries in label_to_top_n_ontology.items() for entry in entries],
        high_level = True)
    ontology_label_dict = {label: [entry['label'] for entry in entries] for label, entries in label_to_top_n_ontology.items()}
    ontology_label_dict = {label: [f"'{entry}'" for entry in entries] for label, entries in label_to_top_n_ontology.items()}
    ontology_label_selection = await _select_best_high_level_ontology_labels(
        ontology_label_dict, OntologyGroupLabelMappingSetDyn, config,
        ontology = ontology)
    ontology_label_selection = LabelMappingSet.model_validate(ontology_label_selection.model_dump())
    # Add missing labels as best guess mappings
    for missing_label in missing_labels:
        ontology_label_selection.mappings.append(
            BestGuessMapping.model_validate({
                "ontology": "best_guess",
                "source_label": missing_label,
                "target_label": "Unknown/Other",
                "notes": "This label was not able to be mapped to the ontology, so the best guess is to keep the original label. This is likely because the label is either very noisy or represents a concept that is not well represented in the ontology."
            })
        )
    # TODO: Possibly set ontology of ontology_label_selection result to whatevers returned by the ols
    return human_readable_ontology_labels, label_to_top_n_ontology, ontology_label_selection

async def _harmonize_sex_labels(
        metadata_dict: Dict[str, Any], extraction_protocol: Dict[str, Any], sample_metadata: pd.DataFrame, config: RunnableConfig):
    column_name = "Sex"
    unique_dataset_labels = sorted([str(x) for x in sample_metadata[column_name].unique()])
    dataset_context = gather_concept_context(metadata_dict, extraction_protocol[column_name.lower()], unique_dataset_labels)
    harmonized_labels = ["Male", "Female", "Unknown/Other"]
    ontology_label_dict = {
        label: harmonized_labels for label in unique_dataset_labels
    }
    _, OntologyGroupLabelMappingSetDyn = create_ontology_mapping_model(
        unique_dataset_labels,
        "pato",
        "Phenotype And Trait Ontology (PATO)",
        allowed_target_labels = harmonized_labels,
        high_level = False)
    ontology_label_selection = await _select_best_ontology_labels(
        ontology_label_dict, dataset_context, OntologyGroupLabelMappingSetDyn, config, ontology = "pato")
    return unique_dataset_labels, ontology_label_dict, ontology_label_selection

def construct_raw_to_harmonized_label_mapping(guessed_ontology_labels: LabelMappingSet, ontology_label_selection: LabelMappingSet):
    fixed_harmonized_label_mapping = {
        "mappings": []
    }
    for mapping in guessed_ontology_labels.mappings:
        if mapping.ontology == "missing":
            fixed_harmonized_label_mapping["mappings"].append({
                "ontology": "best_guess",
                "source_label": mapping.source_label,
                "target_label": mapping.source_label,
                "notes": "This label was not able to be mapped to the ontology, so the best guess is to keep the original label. This is likely because the label is either very noisy or represents a concept that is not well represented in the ontology."
            })
            continue
        best_guess_mapping = next(m for m in ontology_label_selection.mappings if m.source_label == mapping.target_label)
        fixed_harmonized_label_mapping["mappings"].append({
            "ontology": best_guess_mapping.ontology,
            "source_label": mapping.source_label,
            "target_label": best_guess_mapping.target_label if hasattr(best_guess_mapping, "target_label") else mapping.source_label,
            "notes": best_guess_mapping.notes
        })
    return LabelMappingSet.model_validate(fixed_harmonized_label_mapping)