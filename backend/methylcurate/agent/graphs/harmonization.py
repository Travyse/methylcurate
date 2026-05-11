__all__ = ["build_harmonization_graph"]

from langgraph.graph.state import END, START, StateGraph

from ...utils.helper import check_step_completion, get_accession_codes
from ..nodes.harmonize import (
    cell_type_harmonization_node,
    disease_harmonization_node,
    higher_level_disease_mapping_node,
    higher_level_tissue_mapping_node,
    sex_harmonization_node,
    tissue_harmonization_node,
)
from ..state.models import HarmonizationSubgraphState


def route_disease_harmonization_node(state: HarmonizationSubgraphState) -> str:
    """
    Determine the next node to route to after the disease harmonization node.

    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("map_disease_labels_to_ontology", state.datasets, accession_codes):
        return "higher_level_disease_mapping_node"
    return "disease_harmonization_node"


def route_higher_level_disease_mapping_node(state: HarmonizationSubgraphState) -> str:
    """
    Determine the next node to route to after the higher level disease mapping node.

    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("group_disease_labels", state.datasets, accession_codes):
        return "tissue_harmonization_node"
    return "higher_level_disease_mapping_node"


def route_tissue_harmonization_node(state: HarmonizationSubgraphState) -> str:
    """
    Determine the next node to route to after the tissue harmonization node.

    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("map_tissue_labels_to_ontology", state.datasets, accession_codes):
        return "higher_level_tissue_mapping_node"
    return "tissue_harmonization_node"


def route_higher_level_tissue_mapping_node(state: HarmonizationSubgraphState) -> str:
    """
    Determine the next node to route to after the higher level tissue mapping node.

    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("group_tissue_labels", state.datasets, accession_codes):
        return "cell_type_harmonization_node"
    return "higher_level_tissue_mapping_node"


def route_cell_type_harmonization_node(state: HarmonizationSubgraphState) -> str:
    """
    Determine the next node to route to after the cell type harmonization node.
    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("map_cell_type_labels_to_ontology", state.datasets, accession_codes):
        return "sex_harmonization_node"
    return "cell_type_harmonization_node"


def route_sex_harmonization_node(state: HarmonizationSubgraphState) -> str:
    """
    Determine the next node to route to after the sex harmonization node.

    Args:
        state (HarmonizationSubgraphState): The current state of the harmonization subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    if check_step_completion("harmonize_sex_labels", state.datasets, accession_codes):
        return END
    return "sex_harmonization_node"


def build_harmonization_graph() -> StateGraph:
    """
    Build the harmonization subgraph.

    Returns:
        StateGraph: The harmonization subgraph.
    """
    g = StateGraph(HarmonizationSubgraphState)

    # Add nodes
    g.add_node("disease_harmonization_node", disease_harmonization_node)
    g.add_node("higher_level_disease_mapping_node", higher_level_disease_mapping_node)
    g.add_node("tissue_harmonization_node", tissue_harmonization_node)
    g.add_node("higher_level_tissue_mapping_node", higher_level_tissue_mapping_node)
    g.add_node("cell_type_harmonization_node", cell_type_harmonization_node)
    g.add_node("sex_harmonization_node", sex_harmonization_node)

    # Add edges
    g.add_edge(START, "disease_harmonization_node")
    g.add_conditional_edges(
        "disease_harmonization_node",
        route_disease_harmonization_node,
        {
            "higher_level_disease_mapping_node": "higher_level_disease_mapping_node",
            "disease_harmonization_node": "disease_harmonization_node",
        },
    )
    g.add_conditional_edges(
        "higher_level_disease_mapping_node",
        route_higher_level_disease_mapping_node,
        {
            "tissue_harmonization_node": "tissue_harmonization_node",
            "higher_level_disease_mapping_node": "higher_level_disease_mapping_node",
        },
    )
    g.add_conditional_edges(
        "tissue_harmonization_node",
        route_tissue_harmonization_node,
        {
            "higher_level_tissue_mapping_node": "higher_level_tissue_mapping_node",
            "tissue_harmonization_node": "tissue_harmonization_node",
        },
    )
    g.add_conditional_edges(
        "higher_level_tissue_mapping_node",
        route_higher_level_tissue_mapping_node,
        {
            "cell_type_harmonization_node": "cell_type_harmonization_node",
            "higher_level_tissue_mapping_node": "higher_level_tissue_mapping_node",
        },
    )
    g.add_conditional_edges(
        "cell_type_harmonization_node",
        route_cell_type_harmonization_node,
        {
            "cell_type_harmonization_node": "cell_type_harmonization_node",
            "sex_harmonization_node": "sex_harmonization_node",
        },
    )
    g.add_conditional_edges(
        "sex_harmonization_node",
        route_sex_harmonization_node,
        {
            END: END,
            "sex_harmonization_node": "sex_harmonization_node",
        },
    )

    return g
