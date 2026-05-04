from langgraph.graph import END, START, StateGraph

from ..nodes.router import clarify_router_node, router_node
from ..state.models import MainState


def route_after_router(state: MainState):
    """
    Determine the next node to route to after the router node.

    Args:
        state (MainState): The current state of the main graph.

    Returns:
        str: The key of the next node to route to.
    """
    if state.pending_reviews is not None:
        return "clarify"
    return "done"


def finalize_route(state: MainState):
    """
    Finalize the routing process.

    Args:
        state (MainState): The current state of the main graph.

    Returns:
        MainState: The finalized state.
    """
    # optional: normalize fields for the runner
    # ensure routed_subgraph + routed_params are set
    return state


def build_main_graph() -> StateGraph:
    """
    Build the main graph.

    Returns:
        StateGraph: The main graph.
    """
    g = StateGraph(MainState)
    g.add_node("route", router_node)
    g.add_node("clarify", clarify_router_node)
    g.add_node("finalize", finalize_route)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", route_after_router, {"clarify": "clarify", "done": "finalize"})
    g.add_edge("clarify", "finalize")
    g.add_edge("finalize", END)
    return g
