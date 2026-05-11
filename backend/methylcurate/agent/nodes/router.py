__all__ = ["router_node", "clarify_router_node", "validate_workflow_order", "WORKFLOW_PREREQUISITES"]
import asyncio
import re
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import ValidationError

from ...contracts.common import HumanReviewRequest
from ...contracts.router import RouterOutput
from ...utils.helper import make_review_id
from ..graphs.deps import Deps
from ..registry.nodes import GRAPH_BUILDERS, PARAM_SCHEMAS
from ..state.models import MainState
from ..state.utils import make_subgraph_state

MAX_RETRIES = 3
GLOBAL_TIMEOUT = 60

WORKFLOW_PREREQUISITES: dict[str, str | None] = {
    "geo_retrieval": None,
    "harmonization": "geo_retrieval",
    "quality_control": "geo_retrieval",
    "benchmarking": "geo_retrieval",
    "help": None,
}


def _extract_accessions(text: str) -> list[str]:
    return re.findall(r"GSE\d+", text, re.IGNORECASE)


def _get_latest_user_text(messages: list[AnyMessage]) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(str(p) for p in content)
    return None


def validate_workflow_order(state: MainState, subgraph: str, accessions: list[str]) -> tuple[bool, list[str]]:
    prereq = WORKFLOW_PREREQUISITES.get(subgraph)
    if prereq is None:
        return True, []

    missing = []
    for acc in accessions:
        acc_steps = state.datasets.get(acc, {})
        prereq_status = acc_steps.get(prereq)
        if prereq_status is None or prereq_status.status != "completed":
            missing.append(acc)

    return len(missing) == 0, missing


def _build_workflow_state_summary(state: MainState) -> str:
    parts = []

    parts.append("## Workflow Dependency Chain")
    parts.append("- geo_retrieval → harmonization")
    parts.append("- geo_retrieval → quality_control")
    parts.append("- geo_retrieval → benchmarking")
    parts.append("- help: no prerequisites")

    parts.append("\n## Completed Subgraphs")
    if state.subgraphs:
        for name, handle in state.subgraphs.items():
            parts.append(f"- {name}: {handle.status}")
    else:
        parts.append("- No subgraphs have been run yet.")

    parts.append("\n## Per-Dataset Status")
    if state.datasets:
        for acc, steps in state.datasets.items():
            step_str = ", ".join(f"{name}={s.status}" for name, s in steps.items())
            parts.append(f"- {acc}: {step_str}")
    else:
        parts.append("- No datasets have been processed yet.")

    parts.append("\n## Currently Legal Downstream Routes")
    known_accessions = list(state.datasets.keys()) if state.datasets else []
    if known_accessions:
        for sub_name, _prereq in WORKFLOW_PREREQUISITES.items():
            if sub_name in ("geo_retrieval", "help"):
                continue
            is_legal, _ = validate_workflow_order(state, sub_name, known_accessions)
            label = "LEGAL" if is_legal else "ILLEGAL (missing prerequisite)"
            parts.append(f"- {sub_name}: {label}")
    else:
        parts.append("- No downstream routes are legal yet. Only geo_retrieval and help are available.")

    return "\n".join(parts)


async def _get_router_decision(messages: list[AnyMessage], llm: Any) -> RouterOutput:  # type: ignore
    """
    Get a routing decision from the LLM based on the provided message history. The function attempts to call the LLM with a structured output format defined by the RouterOutput model. If the LLM call times out, it retries up to a specified limit. If the LLM output fails validation against the RouterOutput model, it raises a RuntimeError with details about the validation failure. Any other exceptions during the LLM call also result in a RuntimeError with details about the failure.'

    Args:
        messages (List[AnyMessage]): The message history to provide as context for the LLM's routing decision.
        llm (Any): The language model to call for obtaining the routing decision, which should have an acall_structured method that accepts messages and a model for structured output.

    Returns:
        RouterOutput: An instance of the RouterOutput model containing the routing decision made by the L
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            router_out: RouterOutput = RouterOutput.model_validate(
                await asyncio.wait_for(llm.acall_structured(messages, RouterOutput), timeout=GLOBAL_TIMEOUT)
            )
            return router_out
        except TimeoutError:
            retries += 1
            continue
        except ValidationError as e:
            raise RuntimeError(f"LLM output failed validation: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Some failure: {e}") from e


def _inject_workflow_context(messages: list[AnyMessage], state: MainState) -> list[AnyMessage]:
    summary = _build_workflow_state_summary(state)
    context_prompt = (
        "\n\n---\n\nThe following describes the current state of the workflow. "
        "Use this to determine which subgraphs are legal to route to.\n\n"
        + summary
        + "\n\nDo not route to a downstream subgraph for accessions that have not completed the prerequisite subgraph."
    )
    result = list(messages)
    for i, msg in enumerate(result):
        if isinstance(msg, SystemMessage):
            result[i] = SystemMessage(content=str(msg.content) + context_prompt)
            return result
    return [SystemMessage(content=context_prompt)] + result


async def router_node(state: MainState, config: RunnableConfig) -> dict[str, Any]:
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm

    return_dict: dict[str, Any] = {"messages": []}

    review_id_args = {
        "run_id": state.run_id,
        "subgraph": "router",
        "entity_type": "node",
        "entity_id": "router",
        "step": "routing_decision",
    }

    user_text = _get_latest_user_text(state.messages)
    mentioned_accessions = [a.upper() for a in _extract_accessions(user_text or "")]
    known_accessions = list(state.datasets.keys()) if state.datasets else []

    unknown = [a for a in mentioned_accessions if a not in known_accessions]
    if unknown and mentioned_accessions and not known_accessions:
        router_out = RouterOutput(
            subgraph="geo_retrieval",
            params={"accessions": mentioned_accessions, "output_root": str(state.default_output_root or "")},
            confidence=1.0,
            needs_clarification=False,
            reasons=["All mentioned accessions need to be downloaded first."],
        )
        return_dict["routing_history"] = [router_out]
        return_dict["next_action_hint"] = "Ready to run geo_retrieval."
        return return_dict

    routed_messages = _inject_workflow_context(state.messages, state)
    router_out = await _get_router_decision(routed_messages, llm)
    return_dict["routing_history"] = [router_out]

    if router_out.needs_clarification or router_out.confidence < 0.6:
        return_dict["pending_reviews"] = HumanReviewRequest(
            review_id=make_review_id(**review_id_args),
            reason="routing_clarification",
            question=router_out.clarification_question or "I need a bit more information to proceed.",
            payload=router_out.model_dump(),
            created_at=datetime.now(UTC).isoformat(),
        )
        return_dict["next_action_hint"] = "Awaiting user clarification."
        return_dict["messages"].append(
            AIMessage(
                content=router_out.clarification_question or "I need a bit more information to proceed.",
                additional_kwargs={"created_at": datetime.now(UTC).isoformat()},
            )
        )
        return return_dict

    schema = PARAM_SCHEMAS[router_out.subgraph]
    try:
        schema.model_validate(router_out.params)
    except ValidationError as e:
        return_dict["pending_reviews"] = HumanReviewRequest(
            review_id=make_review_id(**review_id_args),
            reason="invalid_route_params",
            question=router_out.clarification_question or "I need one or two details corrected before I can run this.",
            payload={"router_out": router_out.model_dump(), "validation_error": str(e)},
            created_at=datetime.now(UTC).isoformat(),
        )
        return_dict["next_action_hint"] = "Awaiting parameter correction."
        return_dict["messages"].append(
            AIMessage(
                content=router_out.clarification_question or "I need a bit more information to proceed.",
                additional_kwargs={"created_at": datetime.now(UTC).isoformat()},
            )
        )
        return return_dict

    return_dict["next_action_hint"] = f"Ready to run {return_dict['routing_history'][-1].subgraph}."
    return return_dict


async def clarify_router_node(state: MainState, config: RunnableConfig) -> dict[str, Any]:
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm
    return_dict: dict[str, Any] = {}

    if state.pending_reviews is None or state.pending_reviews.reason not in [
        "routing_clarification",
        "invalid_route_params",
    ]:
        return {}

    human_response = interrupt({"prompt": state.pending_reviews.question, "context": state.pending_reviews.payload})
    human_message = HumanMessage(content=human_response, additional_kwargs={"created_at": datetime.now(UTC).isoformat()})
    message_history = _inject_workflow_context(state.messages + [human_message], state)
    router_out = await _get_router_decision(message_history, llm)

    schema = PARAM_SCHEMAS[router_out.subgraph]
    try:
        validated_params = schema.model_validate(router_out.params)
    except Exception as e:
        p_review = state.pending_reviews
        p_review.question = f"I still need one detail to proceed: {e}"
        return_dict["pending_reviews"] = p_review
        return_dict["next_action_hint"] = "Awaiting parameter correction."
        return return_dict

    return_dict["pending_reviews"] = None
    return_dict["selected_subgraph"] = router_out.subgraph
    return_dict["selected_params"] = validated_params.model_dump()
    return_dict["router_confidence"] = router_out.confidence
    return_dict["routing_history"] = [router_out]
    return return_dict


def run_selected_subgraph(main: MainState, checkpointer) -> tuple[MainState, object]:
    """
    Run the selected subgraph with the given main state and checkpointer.

    Args:
        main (MainState): The current state of the main graph, which includes the routed subgraph and parameters.
        checkpointer: The checkpointer to use for the subgraph.

    Returns:
        tuple[MainState, object]: A tuple containing the updated main state and the output of the subgraph.
    """
    name = main.routed_subgraph  # type: ignore
    handle = main.subgraphs[name]

    handle.status = "running"

    subgraph = GRAPH_BUILDERS[name](checkpointer=checkpointer)
    sub_state = make_subgraph_state(name, run_id=main.run_id, params=main.routed_params)  # type: ignore

    # If the subgraph interrupts, your chat runtime receives the interrupt payload.
    # When resumed, invoke continues and returns final state.
    out = subgraph.invoke(sub_state, config={"configurable": {"thread_id": handle.thread_id}})

    # If it returned normally, it finished (or failed without interrupt)
    handle.status = "failed" if getattr(out, "errors", []) else "completed"
    handle.errors.extend(getattr(out, "errors", []))
    handle.warnings.extend(getattr(out, "warnings", []))
    return main, out
