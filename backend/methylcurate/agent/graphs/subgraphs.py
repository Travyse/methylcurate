# agent/graphs/subgraphs.py
from langgraph.graph import StateGraph

from ..state.models import (
    GeoIngestionSubgraphState,
    HarmonizationSubgraphState,
    QualityControlSubgraphState,
    BenchmarkingSubgraphState,
)
from ..registry.nodes import GRAPH_BUILDERS

from .geo import build_geo_retrieval_graph
from .harmonization import build_harmonization_graph
from .qualitycontrol import build_quality_control_graph
from .benchmarking import build_benchmarking_graph


def build_subgraphs() -> dict[str, StateGraph]:
    """
    Build all subgraphs.

    Returns:
        dict[str, StateGraph]: A dictionary mapping subgraph names to their corresponding StateGraph instances.
    """
    graphs: dict[str, StateGraph] = {}
    graphs["geo_retrieval"] = build_geo_retrieval_graph()
    graphs["harmonization"] = build_harmonization_graph()
    graphs["quality_control"] = build_quality_control_graph()
    graphs["benchmarking"] = build_benchmarking_graph()

    return graphs


GRAPH_BUILDERS["geo_retrieval"] = build_geo_retrieval_graph
GRAPH_BUILDERS["harmonization"] = build_harmonization_graph
GRAPH_BUILDERS["quality_control"] = build_quality_control_graph
GRAPH_BUILDERS["benchmarking"] = build_benchmarking_graph
