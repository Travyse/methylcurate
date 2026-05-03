__all__ = ["build_benchmarking_graph"]
from langgraph.graph.state import StateGraph, START, END
from typing import Any
from ..state.models import BenchmarkingSubgraphState
from ...utils.helper import get_accession_codes
from ..nodes.benchmarking import clock_retrieval_node, benchmarking_node, task_computation_node, summarize_benchmarking_results

def route_clock_retrieval_node(state: BenchmarkingSubgraphState) -> str:
    """
    Determine the next node to route to after the clock retrieval node.

    Args:
        state (BenchmarkingSubgraphState): The current state of the benchmarking subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    clocks = sorted(state.config.clock_list)
    clocks_to_retrieve = [clock for clock in clocks if not any(a for a in state.config.artifacts if a.kind == "clock" and (a.path.endswith(f"{clock.lower()}.pt") or a.path.endswith(f"{clock.lower()}_model.pkl")))]
    return "clock_retrieval_node" if clocks_to_retrieve else "clock_prediction_node"

def route_clock_prediction_node(state: BenchmarkingSubgraphState) -> str:
    """
    Determine the next node to route to after the clock prediction node.

    Args:
        state (BenchmarkingSubgraphState): The current state of the benchmarking subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["make_predictions"].status in {'running', 'not_started'}])
    return "clock_prediction_node" if running_accession_codes else "task_computation_node"

def route_task_computation_node(state: BenchmarkingSubgraphState) -> str:
    """
    Determine the next node to route to after the task computation node.

    Args:
        state (BenchmarkingSubgraphState): The current state of the benchmarking subgraph.

    Returns:
        str: The key of the next node to route to.
    """
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted([accession_code for accession_code in accession_codes if state.datasets[accession_code].steps["make_computations"].status in {'running', 'not_started'}])
    return "task_computation_node" if running_accession_codes else "prediction_summarization_node"

def build_benchmarking_graph() -> StateGraph:
    """
    Build the benchmarking subgraph.

    Returns:
        StateGraph: The benchmarking subgraph.
    """
    g = StateGraph(BenchmarkingSubgraphState)

    # Add nodes
    g.add_node("clock_retrieval_node", clock_retrieval_node)
    g.add_node("clock_prediction_node", benchmarking_node)
    g.add_node("task_computation_node", task_computation_node)
    g.add_node("prediction_summarization_node", summarize_benchmarking_results)

    # Add edges
    g.add_edge(START, "clock_retrieval_node")
    g.add_conditional_edges(
        "clock_retrieval_node",
        route_clock_retrieval_node,
        {
            "clock_retrieval_node": "clock_retrieval_node",
            "clock_prediction_node": "clock_prediction_node",
        })
    g.add_conditional_edges(
        "clock_prediction_node",
        route_clock_prediction_node,
        {
            "clock_prediction_node": "clock_prediction_node",
            "task_computation_node": "task_computation_node",
        })
    g.add_conditional_edges(
        "task_computation_node",
        route_task_computation_node,
        {
            "task_computation_node": "task_computation_node",
            "prediction_summarization_node": "prediction_summarization_node",
        })
    g.add_edge("prediction_summarization_node", END)

    return g
