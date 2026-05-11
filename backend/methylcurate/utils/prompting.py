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


_TEMPLATE_REGISTRY = {
    "generate_column_feedback_loop_prompt": "geo/metadata_checking/column_feedback_loop_v2.md",
    "generate_geo_system_concept_prompt": "geo/system/system_concept_prompt.md",
    "generate_geo_system_prompt": "geo/system/system_prompt.md",
    "generate_high_level_ontology_label_selection_query": "harmonize/select_label/select_high_level_ontology_label_query.md",
    "generate_high_level_ontology_label_selection_system_prompt": "harmonize/select_label/select_high_level_ontology_label_system_prompt.md",
    "generate_identify_control_value_prompt": "geo/metadata_extraction/identify_control_value.md",
    "generate_immediate_single_column_feedback": "geo/metadata_checking/immediate_single_column_check_v2.md",
    "generate_infer_methylation_data_column_scheme_alt_prompt": "geo/metadata_extraction/infer_methylation_data_column_scheme_alt.md",
    "generate_infer_methylation_data_column_scheme_prompt": "geo/metadata_extraction/infer_methylation_data_column_scheme.md",
    "generate_metadata_column_user_query": "geo/metadata_extraction/column_extraction_user_query.md",
    "generate_metadata_column_user_query_alt": "geo/metadata_extraction/column_extraction_user_query_alt.md",
    "generate_missing_age_check_prompt": "geo/metadata_checking/missing_age_check.md",
    "generate_ontology_group_guess_system_prompt": "harmonize/guess_label/guess_high_level_ontology_label_system_prompt.md",
    "generate_ontology_group_guess_user_query": "harmonize/guess_label/guess_high_level_ontology_label_query.md",
    "generate_ontology_label_query": "harmonize/guess_label/guess_ontology_label_query.md",
    "generate_ontology_label_selection_query": "harmonize/select_label/select_ontology_label_query.md",
    "generate_ontology_label_selection_system_prompt": "harmonize/select_label/select_ontology_label_system_prompt.md",
    "generate_ontology_label_system_prompt": "harmonize/guess_label/guess_ontology_label_system_prompt.md",
    "generate_router_system_prompt": "router/system_prompt.md",
}


def generate_column_feedback_loop_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_column_feedback_loop_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_geo_system_concept_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_geo_system_concept_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_geo_system_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_geo_system_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_high_level_ontology_label_selection_query(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_high_level_ontology_label_selection_query"]
    return render_prompt(template_path, **kwargs)


def generate_high_level_ontology_label_selection_system_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_high_level_ontology_label_selection_system_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_identify_control_value_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_identify_control_value_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_immediate_single_column_feedback(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_immediate_single_column_feedback"]
    return render_prompt(template_path, **kwargs)


def generate_infer_methylation_data_column_scheme_alt_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_infer_methylation_data_column_scheme_alt_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_infer_methylation_data_column_scheme_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_infer_methylation_data_column_scheme_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_metadata_column_user_query(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_metadata_column_user_query"]
    return render_prompt(template_path, **kwargs)


def generate_metadata_column_user_query_alt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_metadata_column_user_query_alt"]
    return render_prompt(template_path, **kwargs)


def generate_missing_age_check_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_missing_age_check_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_ontology_group_guess_system_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_group_guess_system_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_ontology_group_guess_user_query(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_group_guess_user_query"]
    return render_prompt(template_path, **kwargs)


def generate_ontology_label_query(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_label_query"]
    return render_prompt(template_path, **kwargs)


def generate_ontology_label_selection_query(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_label_selection_query"]
    return render_prompt(template_path, **kwargs)


def generate_ontology_label_selection_system_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_label_selection_system_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_ontology_label_system_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_ontology_label_system_prompt"]
    return render_prompt(template_path, **kwargs)


def generate_router_system_prompt(template_path: str | None = None, **kwargs: Any) -> str:
    if template_path is None:
        template_path = _TEMPLATE_REGISTRY["generate_router_system_prompt"]
    return render_prompt(template_path, **kwargs)
