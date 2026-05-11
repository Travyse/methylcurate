__all__ = ["build_geo_retrieval_graph"]
import os

from langgraph.graph.state import END, START, StateGraph

from ...utils.helper import check_step_completion, get_accession_codes
from ..nodes.geo import (
    check_column_extraction_rule_accuracy,
    check_column_extraction_rule_formatting,
    check_data_presence,
    check_downloads_succeeded,
    check_platforms_used,
    extract_metadata_schema,
    extract_sample_metadata,
    format_supplementary_data,
    generate_metadata_extraction_summary,
    geo_download_node,
    geo_metadata_column_extraction_approval_node,
    merge_supplementary_file_data,
    refine_extracted_columns,
    start_geo_subgraph,
    summarize_geo_findings,
)
from ..state.models import GeoIngestionSubgraphState


def route_soft_file(state: GeoIngestionSubgraphState) -> str:
    """
    Determine the next node to route to after the soft file download node.

    Args:
        state (GeoIngestionSubgraphState): The current state of the GEO ingestion subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    datasets = state.datasets
    download_completion = check_step_completion("download_soft", datasets, accession_codes)
    valid_check_completion = check_step_completion("check_valid_dataset", datasets, accession_codes)
    completed = download_completion and valid_check_completion
    return "extract_metadata_column_scheme" if completed else "check_downloads_succeeded"


def route_metadata_scheme_extraction(state: GeoIngestionSubgraphState) -> str:
    """
    Determine the next node to route to after the metadata schema extraction node.

    Args:
        state (GeoIngestionSubgraphState): The current state of the GEO ingestion subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    datasets = state.datasets
    completed = check_step_completion("extract_metadata_schema", datasets, accession_codes)
    return "extract_sample_metadata" if completed else "extract_metadata_column_scheme"


def route_extract_sample_metadata(state: GeoIngestionSubgraphState) -> str:
    """
    Determine the next node to route to after the sample metadata extraction node.

    Args:
        state (GeoIngestionSubgraphState): The current state of the GEO ingestion subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    datasets = state.datasets
    completed = check_step_completion("extract_data", datasets, accession_codes)
    return "check_extraction_rule_formatting" if completed else "generate_metadata_extraction_summary"


def route_check_extraction_rule_formatting(state: GeoIngestionSubgraphState) -> str:
    """
    Determine the next node to route to after the extraction rule formatting check node.

    Args:
        state (GeoIngestionSubgraphState): The current state of the GEO ingestion subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    datasets = state.datasets
    completed = check_step_completion("refine_metadata_schema", datasets, accession_codes)
    return "format_supplementary_data" if completed else "check_extraction_rule_accuracy"


def route_format_supplementary_data(state: GeoIngestionSubgraphState) -> str:
    """
    Determine the next node to route to after the supplementary data formatting node.

    Args:
        state (GeoIngestionSubgraphState): The current state of the GEO ingestion subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    datasets = state.datasets
    completed = check_step_completion("supplementary_file_check", datasets, accession_codes)
    if completed:
        return "summarize_geo_findings"

    running_accession_codes = sorted(
        [accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["supplementary_file_check"].status == "running"]
    )
    accession_code = running_accession_codes[0]
    supplementary_file_artifacts = sorted(
        [
            artifact
            for artifact in state.config.artifacts
            if (artifact.kind == "supplementary_file_methylation_data") and (artifact.accession_code == accession_code)
        ],
        key=lambda artifact: artifact.path,
    )
    formatted_supplementary_file_artifacts = sorted(
        [
            artifact
            for artifact in state.config.artifacts
            if (artifact.kind == "supplementary_file_methylation_data_formatted") and (artifact.accession_code == accession_code)
        ],
        key=lambda artifact: artifact.path,
    )
    running_supplementary_file_artifacts = [
        artifact
        for artifact in supplementary_file_artifacts
        if f"{os.path.splitext(artifact.path)[0]}_proc"
        not in [os.path.splitext(artifact.path)[0] for artifact in formatted_supplementary_file_artifacts]
    ]

    supplementary_file_completion = len(running_supplementary_file_artifacts) == 0

    return "merge_supplementary_file_data" if supplementary_file_completion else "format_supplementary_data"


def build_geo_retrieval_graph() -> StateGraph:
    """
    Build the GEO retrieval subgraph.

    Returns:
        StateGraph: The GEO retrieval subgraph.
    """
    g = StateGraph(GeoIngestionSubgraphState)

    # Add nodes
    ## Downloads
    g.add_node("start_geo_subgraph", start_geo_subgraph)
    g.add_node("download_soft_file", geo_download_node)
    g.add_node("check_downloads_succeeded", check_downloads_succeeded)
    g.add_node("check_platforms_used", check_platforms_used)
    g.add_node("check_data_presence", check_data_presence)
    ## Metadata column extraction
    g.add_node("extract_metadata_column_scheme", extract_metadata_schema)
    ## Data extraction
    g.add_node("extract_sample_metadata", extract_sample_metadata)
    g.add_node("generate_metadata_extraction_summary", generate_metadata_extraction_summary)
    ## Schema Refinement
    g.add_node("check_extraction_rule_formatting", check_column_extraction_rule_formatting)
    g.add_node("check_extraction_rule_accuracy", check_column_extraction_rule_accuracy)
    g.add_node("geo_metadata_column_extraction_approval_node", geo_metadata_column_extraction_approval_node)
    ## Supplementary Data
    g.add_node("format_supplementary_data", format_supplementary_data)
    g.add_node("merge_supplementary_file_data", merge_supplementary_file_data)
    g.add_node("refine_extracted_columns", refine_extracted_columns)
    ## Endpoint
    g.add_node("summarize_geo_findings", summarize_geo_findings)

    # Add edges
    g.add_edge(START, "start_geo_subgraph")
    g.add_edge("start_geo_subgraph", "download_soft_file")

    # Download GEO data
    g.add_conditional_edges(
        "download_soft_file",
        route_soft_file,
        {
            "extract_metadata_column_scheme": "extract_metadata_column_scheme",
            "check_downloads_succeeded": "check_downloads_succeeded",
        },
    )
    g.add_edge("check_downloads_succeeded", "check_platforms_used")
    g.add_edge("check_platforms_used", "check_data_presence")
    g.add_edge("check_data_presence", "download_soft_file")

    # Metadata column extraction
    g.add_conditional_edges(
        "extract_metadata_column_scheme",
        route_metadata_scheme_extraction,
        {
            "extract_sample_metadata": "extract_sample_metadata",
            "extract_metadata_column_scheme": "extract_metadata_column_scheme",
        },
    )

    # Data Extraction
    g.add_conditional_edges(
        "extract_sample_metadata",
        route_extract_sample_metadata,
        {
            "check_extraction_rule_formatting": "check_extraction_rule_formatting",
            "generate_metadata_extraction_summary": "generate_metadata_extraction_summary",
        },
    )
    g.add_edge("generate_metadata_extraction_summary", "extract_sample_metadata")

    # Schema refinement

    g.add_conditional_edges(
        "check_extraction_rule_formatting",
        route_check_extraction_rule_formatting,
        {
            "check_extraction_rule_accuracy": "check_extraction_rule_accuracy",
            "format_supplementary_data": "format_supplementary_data",
        },
    )
    g.add_edge("check_extraction_rule_accuracy", "geo_metadata_column_extraction_approval_node")
    g.add_edge("geo_metadata_column_extraction_approval_node", "check_extraction_rule_formatting")

    # Supplementary data formatting and merging
    g.add_conditional_edges(
        "format_supplementary_data",
        route_format_supplementary_data,
        {
            "format_supplementary_data": "format_supplementary_data",
            "merge_supplementary_file_data": "merge_supplementary_file_data",
            "summarize_geo_findings": "summarize_geo_findings",
        },
    )
    g.add_edge("merge_supplementary_file_data", "refine_extracted_columns")
    g.add_edge("refine_extracted_columns", "format_supplementary_data")

    g.add_edge("summarize_geo_findings", END)

    return g
