"""Prompt rendering utilities for MethylCurate agent workflows."""

from pathlib import Path
from typing import Any
from jinja2 import Environment, FileSystemLoader

BASE_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = BASE_DIR / "agent/prompts"
env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))


def render_prompt(template_path: str, **kwargs: Any) -> str:
    """Render a Jinja2 prompt template with the provided keyword arguments.

    Args:
        template_path: Path to the template file relative to the prompts directory.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    template = env.get_template(template_path)
    return template.render(**kwargs)


# Registry mapping function names to their default template paths.
_TEMPLATE_REGISTRY = {
    "generate_clarify_router_response_prompt": "router/clarify_user_intent.md",
    "generate_column_feedback_loop_prompt": "geo/column_feedback_loop_v2.md",
    "generate_geo_metadata_column_extraction_prompt": "geo/extract_columns_from_metadata.md",
    "generate_geo_metadata_harmonization_prompt": "harmonize/harmonize_metadata_field.md",
    "generate_geo_system_concept_prompt": "geo/system_concept_prompt.md",
    "generate_geo_system_prompt": "geo/system_prompt.md",
    "generate_harmonization_user_query": "harmonize/user_harmonization_query.md",
    "generate_high_level_ontology_label_selection_query": "harmonize/select_high_level_ontology_label_query.md",
    "generate_high_level_ontology_label_selection_system_prompt": "harmonize/select_high_level_ontology_label_system_prompt.md",
    "generate_identify_control_value_prompt": "geo/identify_control_value.md",
    "generate_immediate_column_feedback": "geo/immediate_column_check.md",
    "generate_immediate_single_column_feedback": "geo/immediate_single_column_check_v2.md",
    "generate_infer_methylation_data_column_scheme_alt_prompt": "geo/infer_methylation_data_column_scheme_alt.md",
    "generate_infer_methylation_data_column_scheme_prompt": "geo/infer_methylation_data_column_scheme.md",
    "generate_interpret_user_intent_prompt": "router/interpret_user_intent.md",
    "generate_metadata_column_user_query": "geo/column_extraction_user_query.md",
    "generate_metadata_column_user_query_alt": "geo/column_extraction_user_query_alt.md",
    "generate_missing_age_check_prompt": "geo/missing_age_check.md",
    "generate_ontology_group_guess_system_prompt": "harmonize/guess_high_level_ontology_label_system_prompt.md",
    "generate_ontology_group_guess_user_query": "harmonize/guess_high_level_ontology_label_query.md",
    "generate_ontology_label_query": "harmonize/guess_ontology_label_query.md",
    "generate_ontology_label_selection_query": "harmonize/select_ontology_label_query.md",
    "generate_ontology_label_selection_system_prompt": "harmonize/select_ontology_label_system_prompt.md",
    "generate_ontology_label_system_prompt": "harmonize/guess_ontology_label_system_prompt.md",
    "generate_qc_plan_prompt": "qc/qc_plan.md",
    "generate_router_clarification_prompt": "router/clarify_user_intent.md",
    "generate_router_interpretation_prompt": "router/interpret_user_intent.md",
    "generate_router_system_prompt": "router/system_prompt.md",
    "generate_subject_column_refinement_prompt": "geo/subject_column_refinement.md",
    "generate_system_prompt": "system_prompt.md",
}


def generate_clarify_router_response_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the clarify router response prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_clarify_router_response_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_column_feedback_loop_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the column feedback loop prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_column_feedback_loop_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_geo_metadata_column_extraction_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the geo metadata column extraction prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_geo_metadata_column_extraction_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_geo_metadata_harmonization_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the geo metadata harmonization prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_geo_metadata_harmonization_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_geo_system_concept_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the geo system concept prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_geo_system_concept_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_geo_system_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the geo system prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_geo_system_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_harmonization_user_query(template_path: str = None, **kwargs: Any) -> str:
    """Render the harmonization user query prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_harmonization_user_query"]
    return render_prompt(template_path, **kwargs)

def generate_high_level_ontology_label_selection_query(template_path: str = None, **kwargs: Any) -> str:
    """Render the high level ontology label selection query prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_high_level_ontology_label_selection_query"]
    return render_prompt(template_path, **kwargs)

def generate_high_level_ontology_label_selection_system_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the high level ontology label selection system prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_high_level_ontology_label_selection_system_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_identify_control_value_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the identify control value prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_identify_control_value_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_immediate_column_feedback(template_path: str = None, **kwargs: Any) -> str:
    """Render the immediate column feedback prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_immediate_column_feedback"]
    return render_prompt(template_path, **kwargs)

def generate_immediate_single_column_feedback(template_path: str = None, **kwargs: Any) -> str:
    """Render the immediate single column feedback prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_immediate_single_column_feedback"]
    return render_prompt(template_path, **kwargs)

def generate_infer_methylation_data_column_scheme_alt_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the infer methylation data column scheme alt prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_infer_methylation_data_column_scheme_alt_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_infer_methylation_data_column_scheme_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the infer methylation data column scheme prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_infer_methylation_data_column_scheme_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_interpret_user_intent_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the interpret user intent prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_interpret_user_intent_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_metadata_column_user_query(template_path: str = None, **kwargs: Any) -> str:
    """Render the metadata column user query prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_metadata_column_user_query"]
    return render_prompt(template_path, **kwargs)

def generate_metadata_column_user_query_alt(template_path: str = None, **kwargs: Any) -> str:
    """Render the metadata column user query alt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_metadata_column_user_query_alt"]
    return render_prompt(template_path, **kwargs)

def generate_missing_age_check_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the missing age check prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_missing_age_check_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_ontology_group_guess_system_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the ontology group guess system prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_group_guess_system_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_ontology_group_guess_user_query(template_path: str = None, **kwargs: Any) -> str:
    """Render the ontology group guess user query prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_group_guess_user_query"]
    return render_prompt(template_path, **kwargs)

def generate_ontology_label_query(template_path: str = None, **kwargs: Any) -> str:
    """Render the ontology label query prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_label_query"]
    return render_prompt(template_path, **kwargs)

def generate_ontology_label_selection_query(template_path: str = None, **kwargs: Any) -> str:
    """Render the ontology label selection query prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_label_selection_query"]
    return render_prompt(template_path, **kwargs)

def generate_ontology_label_selection_system_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the ontology label selection system prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_label_selection_system_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_ontology_label_system_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the ontology label system prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_label_system_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_qc_plan_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the qc plan prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_qc_plan_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_router_clarification_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the router clarification prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_router_clarification_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_router_interpretation_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the router interpretation prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_router_interpretation_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_router_system_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the router system prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_router_system_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_subject_column_refinement_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the subject column refinement prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_subject_column_refinement_prompt"]
    return render_prompt(template_path, **kwargs)

def generate_system_prompt(template_path: str = None, **kwargs: Any) -> str:
    """Render the system prompt prompt template.

    Args:
        template_path: Override path to a template file relative to prompts.
        **kwargs: Variables to substitute into the template.

    Returns:
        The rendered prompt string.
    """
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_system_prompt"]
    return render_prompt(template_path, **kwargs)
