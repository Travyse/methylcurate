__all__ = ["router_node", "clarify_router_node"]
import os
import asyncio
from pydantic import ValidationError
from langgraph.types import interrupt, Command
from jinja2 import Environment
from typing import Dict, Any, List
from datetime import datetime, timezone
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from ..graphs.deps import Deps
from ..state.models import MainState
from ..registry.nodes import GRAPH_BUILDERS, PARAM_SCHEMAS
from ..state.utils import make_subgraph_state
from ...contracts.common import HumanReviewRequest
from ...contracts.router import RouterOutput
from ...utils.helper import make_review_id
from ...utils.examples import generate_router_interpretation_examples, generate_router_clarification_examples

MAX_RETRIES = 3
GLOBAL_TIMEOUT = 60


async def _get_router_decision(messages: List[AnyMessage], llm: Any) -> RouterOutput:
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
        except asyncio.TimeoutError:
            retries += 1
            continue
        except ValidationError as e:
            raise RuntimeError(f"LLM output failed validation: {e}")
        except Exception as e:
            raise RuntimeError(f"Some failure: {e}")


async def router_node(state: MainState, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    Handle the routing logic for the given state using the configured LLM. This function retrieves a routing decision from the LLM, validates the decision, and prepares the appropriate response, including handling cases where human review is required. The function first calls the LLM to get a routing decision based on the message history. If the LLM indicates that clarification is needed or if the confidence in the routing decision is below a certain threshold, it creates a HumanReviewRequest for clarification and returns it along with a hint for the next action. If the routing decision is valid but the parameters fail validation against the chosen subgraph's parameter schema, it also creates a HumanReviewRequest for parameter correction. If the routing decision and parameters are valid, it prepares a response indicating that it's ready to run the selected subgraph.

    Args:
        state (MainState): The current state of the main graph, which includes the message history and other relevant information for making a routing decision.
        config (RunnableConfig): The configuration for the runnable, which includes dependencies such as the LLM to use for obtaining the routing decision.

    Returns:
        Dict[str, Any]: A dictionary containing the results of the routing logic, which may include
    """
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm

    return_dict = {"messages": []}

    router_out = await _get_router_decision(state.messages, llm)
    return_dict["routing_history"] = [router_out]

    review_id_args = {
        "run_id": state.run_id,
        "subgraph": "router",
        "entity_type": "node",
        "entity_id": "router",
        "step": "routing_decision",
    }
    # Case 1: Intent unclear OR missing required info -> HITL clarification
    if router_out.needs_clarification or router_out.confidence < 0.6:
        return_dict["pending_reviews"] = HumanReviewRequest(
            review_id=make_review_id(**review_id_args),
            reason="routing_clarification",
            question=router_out.clarification_question or "I need a bit more information to proceed.",
            payload=router_out.model_dump(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return_dict["next_action_hint"] = "Awaiting user clarification."
        return_dict["messages"].append(
            AIMessage(
                content=router_out.clarification_question or "I need a bit more information to proceed.",
                additional_kwargs={"created_at": datetime.now(timezone.utc).isoformat()},
            )
        )
        return return_dict

    # Case 2: Validate the params against the chosen subgraph param schema
    schema = PARAM_SCHEMAS[router_out.subgraph]
    try:
        validated_params = schema.model_validate(router_out.params)
    except ValidationError as e:
        # This is *also* a clarification case.
        # Prefer asking only for fields that failed validation.
        return_dict["pending_reviews"] = HumanReviewRequest(
            review_id=make_review_id(**review_id_args),
            reason="invalid_route_params",
            question=router_out.clarification_question or "I need one or two details corrected before I can run this.",
            payload={
                "router_out": router_out.model_dump(),
                "validation_error": str(e),
            },
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return_dict["next_action_hint"] = "Awaiting parameter correction."
        return_dict["messages"].append(
            AIMessage(
                content=router_out.clarification_question or "I need a bit more information to proceed.",
                additional_kwargs={"created_at": datetime.now(timezone.utc).isoformat()},
            )
        )

        return return_dict

    # Success
    return_dict["next_action_hint"] = f"Ready to run {return_dict['routing_history'][-1].subgraph}."

    return return_dict


async def clarify_router_node(state: MainState, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    Handle the clarification logic for the router node. This function is invoked when a human review is required for the routing decision, either due to low confidence or invalid parameters. It processes the human response to the clarification request, updates the message history with the human's input, and calls the LLM again to get an updated routing decision. The function then validates the new routing decision and parameters, and prepares the appropriate response based on whether the clarification resolved the issues or if further clarification is needed.

    Args:
        state (MainState): The current state of the main graph, which includes the message history, pending reviews, and other relevant information for processing the clarification.
        config (RunnableConfig): The configuration for the runnable, which includes dependencies such as the LLM to use for obtaining the updated routing decision.

    Returns:
        Dict[str, Any]: A dictionary containing the results of the clarification logic, which may include updated routing decisions, pending reviews, and hints for the next action based on the human's response to the clarification request.
    """
    deps: Deps = config["configurable"]["deps"]
    llm = deps.llm
    return_dict = {}

    if state.pending_reviews is None or state.pending_reviews.reason not in [
        "routing_clarification",
        "invalid_route_params",
    ]:
        return {}

    # Assume the human provided text in state.pending_review.response_text
    human_response = interrupt({"prompt": state.pending_reviews.question, "context": state.pending_reviews.payload})
    # Possibly have the LLM involved in this
    human_message = HumanMessage(
        content=human_response, additional_kwargs={"created_at": datetime.now(timezone.utc).isoformat()}
    )
    message_history = state.messages + [human_message]
    router_out = await _get_router_decision(message_history, llm)

    # Validate params against the chosen subgraph schema
    schema = PARAM_SCHEMAS[router_out.subgraph]
    try:
        validated_params = schema.model_validate(router_out.params)
    except Exception as e:
        # Still invalid, keep pending review
        p_review = state.pending_reviews
        p_review.question = f"I still need one detail to proceed: {e}"
        return_dict["pending_reviews"] = p_review
        return_dict["next_action_hint"] = "Awaiting parameter correction."
        return return_dict

    # Clear pending review and set selected subgraph/params
    return_dict["pending_reviews"] = None
    return_dict["selected_subgraph"] = router_out.subgraph
    return_dict["selected_params"] = validated_params.model_dump()
    return_dict["router_confidence"] = router_out.confidence
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
    name = main.routed_subgraph
    handle = main.subgraphs[name]

    handle.status = "running"

    subgraph = GRAPH_BUILDERS[name](checkpointer=checkpointer)
    sub_state = make_subgraph_state(name, run_id=main.run_id, params=main.routed_params)

    # If the subgraph interrupts, your chat runtime receives the interrupt payload.
    # When resumed, invoke continues and returns final state.
    out = subgraph.invoke(sub_state, config={"configurable": {"thread_id": handle.thread_id}})

    # If it returned normally, it finished (or failed without interrupt)
    handle.status = "failed" if getattr(out, "errors", []) else "completed"
    handle.errors.extend(getattr(out, "errors", []))
    handle.warnings.extend(getattr(out, "warnings", []))
    return main, out
