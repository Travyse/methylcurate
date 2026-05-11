from datetime import UTC, datetime

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from ...contracts.common import HumanReviewRequest
from ...utils.helper import make_review_id
from ..nodes.router import WORKFLOW_PREREQUISITES, clarify_router_node, router_node, validate_workflow_order
from ..state.models import MainState


def route_after_router(state: MainState):
    if state.pending_reviews is not None:
        return "clarify"
    return "done"


def finalize_route(state: MainState):
    routing_history = state.routing_history
    if not routing_history:
        return state

    router_out = routing_history[-1]
    subgraph = router_out.subgraph
    prereq = WORKFLOW_PREREQUISITES.get(subgraph)
    if prereq is None:
        return state

    accessions = router_out.params.get("accessions", [])
    if not accessions:
        return state

    is_legal, missing = validate_workflow_order(state, subgraph, accessions)
    if is_legal:
        return state

    prefix = "accessions" if len(missing) > 1 else "accession"
    question = f"{', '.join(missing)} {prefix} haven't completed {prereq} yet. I'll need to run {prereq} first. Route to {prereq} instead?"
    state.pending_reviews = HumanReviewRequest(
        review_id=make_review_id(
            run_id=state.run_id,
            subgraph="router",
            entity_type="node",
            entity_id="finalize",
            step="routing_decision",
        ),
        reason="routing_clarification",
        question=question,
        payload={"router_out": router_out.model_dump(), "missing_accessions": missing},
        created_at=datetime.now(UTC).isoformat(),
    )
    state.next_action_hint = "Awaiting user clarification."
    state.messages.append(AIMessage(content=question, additional_kwargs={"created_at": datetime.now(UTC).isoformat()}))
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
