__all__ = ["build_quality_control_graph"]

from langgraph.graph.state import END, START, StateGraph

from ...utils.helper import get_accession_codes
from ..nodes.qc import quality_control_node, quality_control_summarization_node
from ..state.models import QualityControlSubgraphState


def route_quality_control_execution(state: QualityControlSubgraphState) -> str:
    """
    Determine the next node to route to after the quality control execution node.

    Args:
        state (QualityControlSubgraphState): The current state of the quality control subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted(
        [
            accession_code
            for accession_code in accession_codes
            if state.datasets[accession_code].steps["quality_control"].status in {"running", "not_started"}
        ]
    )
    return "quality_control_execution" if running_accession_codes else "quality_control_summarization"


def build_quality_control_graph() -> StateGraph:
    """
    Build the quality control subgraph.

    Returns:
        StateGraph: The quality control subgraph.
    """
    g = StateGraph(QualityControlSubgraphState)

    # Add nodes
    g.add_node("quality_control_execution", quality_control_node)
    g.add_node("quality_control_summarization", quality_control_summarization_node)

    # Add edges
    g.add_edge(START, "quality_control_execution")
    g.add_conditional_edges(
        "quality_control_execution",
        route_quality_control_execution,
        {
            "quality_control_execution": "quality_control_execution",
            "quality_control_summarization": "quality_control_summarization",
        },
    )
    g.add_edge("quality_control_summarization", END)

    return g
