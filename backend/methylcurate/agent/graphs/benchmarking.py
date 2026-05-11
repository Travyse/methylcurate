__all__ = ["build_benchmarking_graph"]

from langgraph.graph.state import END, START, StateGraph

from ...utils.helper import get_accession_codes
from ..nodes.benchmarking import (
    benchmarking_node,
    clock_retrieval_node,
    summarize_benchmarking_results,
)
from ..state.models import BenchmarkingSubgraphState


def route_clock_retrieval_node(state: BenchmarkingSubgraphState) -> str:
    clocks = sorted(state.config.clock_list)
    clocks_to_retrieve = [
        clock
        for clock in clocks
        if not any(
            a
            for a in state.config.artifacts
            if a.kind == "clock" and (a.path.endswith(f"{clock.lower()}.pt") or a.path.endswith(f"{clock.lower()}_model.pkl"))
        )
    ]
    return "clock_retrieval_node" if clocks_to_retrieve else "clock_prediction_node"


def route_clock_prediction_node(state: BenchmarkingSubgraphState) -> str:
    accession_codes = get_accession_codes(state)
    running_accession_codes = sorted(
        [
            accession_code
            for accession_code in accession_codes
            if state.datasets[accession_code].steps["make_predictions"].status in {"running", "not_started"}
        ]
    )
    return "clock_prediction_node" if running_accession_codes else "prediction_summarization_node"


def build_benchmarking_graph() -> StateGraph:
    g = StateGraph(BenchmarkingSubgraphState)

    g.add_node("clock_retrieval_node", clock_retrieval_node)
    g.add_node("clock_prediction_node", benchmarking_node)
    g.add_node("prediction_summarization_node", summarize_benchmarking_results)

    g.add_edge(START, "clock_retrieval_node")
    g.add_conditional_edges(
        "clock_retrieval_node",
        route_clock_retrieval_node,
        {
            "clock_retrieval_node": "clock_retrieval_node",
            "clock_prediction_node": "clock_prediction_node",
        },
    )
    g.add_conditional_edges(
        "clock_prediction_node",
        route_clock_prediction_node,
        {
            "clock_prediction_node": "clock_prediction_node",
            "prediction_summarization_node": "prediction_summarization_node",
        },
    )
    g.add_edge("prediction_summarization_node", END)

    return g
