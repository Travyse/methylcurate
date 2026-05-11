__all__ = ["build_help_graph"]

from langgraph.graph.state import END, START, StateGraph

from ..nodes.help import help_node
from ..state.models import HelpSubgraphState


def build_help_graph() -> StateGraph:
    g = StateGraph(HelpSubgraphState)

    g.add_node("help_node", help_node)

    g.add_edge(START, "help_node")
    g.add_edge("help_node", END)

    return g
