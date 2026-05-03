from pathlib import Path
from typing import Dict
from jinja2 import Environment, FileSystemLoader

BASE_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = BASE_DIR / "agent/prompts"
env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))

def render_prompt(template_path: str, **kwargs) -> str:
    template = env.get_template(template_path)
    return template.render(**kwargs)

def generate_geo_metadata_column_extraction_prompt(template_path: str = "geo/extract_columns_from_metadata.md", **kwargs: Dict) -> str:
    """
    Generates a prompt for extracting columns from GEO metadata using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.
    
    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_geo_metadata_harmonization_prompt(template_path: str = "harmonize/harmonize_metadata_field.md", **kwargs: Dict) -> str:
    """
    Generates a prompt for harmonizing GEO metadata using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_qc_plan_prompt(template_path: str = "preprocess/qc_plan_prompt.md", **kwargs: Dict) -> str:
    """
    Generates a prompt for creating a QC plan using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_clarify_router_response_prompt(template_path: str = "router/clarify_user_intent.md", **kwargs: Dict) -> str:
    """
    Generates a prompt for clarifying the user's intent using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_interpret_user_intent_prompt(template_path: str ="router/interpret_user_intent.md", **kwargs: Dict) -> str:
    """
    Generates a prompt for interpreting the user's intent using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_router_interpretation_prompt(template_path: str ="router/interpret_user_intent.md", **kwargs: Dict) -> str:
    """
    Generates a prompt for interpreting the user's intent using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_router_clarification_prompt(template_path: str ="router/clarify_user_intent.md", **kwargs: Dict) -> str:
    """
    Generates a prompt for clarifying the user's intent using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_system_prompt(template_path: str = "system_prompt.md", **kwargs: Dict) -> str:
    """
    Generates a system prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_geo_system_prompt(template_path: str = "geo/system_prompt.md", **kwargs: Dict) -> str:
    """
    Generates a GEO system prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_geo_system_concept_prompt(template_path: str = "geo/system_concept_prompt.md", **kwargs: Dict) -> str:
    """
    Generates a GEO system concept prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_router_system_prompt(template_path: str = "router/system_prompt.md", **kwargs: Dict) -> str:
    """
    Generates a router system prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_metadata_column_user_query(template_path: str = "geo/column_extraction_user_query.md", **kwargs: Dict) -> str:
    """
    Generates a metadata column user query prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_metadata_column_user_query_alt(template_path: str = "geo/column_extraction_user_query_alt.md", **kwargs: Dict) -> str:
    """
    Generates an alternative metadata column user query prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_harmonization_user_query(template_path: str = "harmonize/user_harmonization_query.md", **kwargs: Dict) -> str:
    """
    Generates a harmonization user query prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_column_feedback_loop_prompt(template_path: str = "geo/column_feedback_loop_v2.md", **kwargs: Dict) -> str:
    """
    Generates a column feedback loop prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_immediate_column_feedback(template_path: str = "geo/immediate_column_check.md", **kwargs: Dict) -> str:
    """
    Generates an immediate column feedback prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_immediate_single_column_feedback(template_path: str = "geo/immediate_single_column_check_v2.md", **kwargs: Dict) -> str:
    """
    Generates an immediate single column feedback prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_identify_control_value_prompt(template_path: str = "geo/identify_control_value.md", **kwargs: Dict) -> str:
    """
    Generates an identify control value prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_missing_age_check_prompt(template_path: str = "geo/missing_age_check.md", **kwargs: Dict) -> str:
    """
    Generates a missing age check prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_infer_methylation_data_column_scheme_prompt(template_path: str = "geo/infer_methylation_data_column_scheme.md", **kwargs: Dict) -> str:
    """
    Generates an infer methylation data column scheme prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_subject_column_refinement_prompt(template_path: str = "geo/subject_column_refinement.md", **kwargs: Dict) -> str:
    """
    Generates a subject column refinement prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_infer_methylation_data_column_scheme_alt_prompt(template_path: str = "geo/infer_methylation_data_column_scheme_alt.md", **kwargs: Dict) -> str:
    """
    Generates an alternative infer methylation data column scheme prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_ontology_group_guess_user_query(template_path: str = "harmonize/guess_high_level_ontology_label_query.md", **kwargs: Dict) -> str:
    """
    Generates an ontology group guess user query prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_ontology_group_guess_system_prompt(template_path: str = "harmonize/guess_high_level_ontology_label_system_prompt.md", **kwargs: Dict) -> str:
    """
    Generates an ontology group guess system prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_ontology_label_query(template_path: str = "harmonize/guess_ontology_label_query.md", **kwargs: Dict) -> str:
    """
    Generates an ontology label query prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_ontology_label_system_prompt(template_path: str = "harmonize/guess_ontology_label_system_prompt.md", **kwargs: Dict) -> str:
    """
    Generates an ontology label system prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_ontology_label_selection_query(template_path: str = "harmonize/select_ontology_label_query.md", **kwargs: Dict) -> str:
    """
    Generates an ontology label selection query prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_ontology_label_selection_system_prompt(template_path: str = "harmonize/select_ontology_label_system_prompt.md", **kwargs: Dict) -> str:
    """
    Generates an ontology label selection system prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_high_level_ontology_label_selection_query(template_path: str = "harmonize/select_high_level_ontology_label_query.md", **kwargs: Dict) -> str:
    """
    Generates a high-level ontology label selection query prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)

def generate_high_level_ontology_label_selection_system_prompt(template_path: str = "harmonize/select_high_level_ontology_label_system_prompt.md", **kwargs: Dict) -> str:
    """
    Generates a high-level ontology label selection system prompt using a specified template and keyword arguments.

    Args:
        template_path (str): The path to the prompt template file relative to the prompts directory.
        **kwargs (Dict): A dictionary of keyword arguments to be passed to the template for rendering.

    Returns:
        str: The rendered prompt string based on the provided template and arguments.
    """
    return render_prompt(template_path, **kwargs)
