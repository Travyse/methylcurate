import asyncio
import os
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command  # <-- critical for resume with interrupt()
from pydantic import ValidationError

from ...utils.helper import consolidate_artifacts
from ..graphs.deps import Deps
from ..state.models import DatasetQualityControlState
from ..state.utils import _append_user_message, make_full_subgraph_state, make_subgraph_state

MAIN_RECURSION_LIMIT = 500
SUBGRAPH_RECURSION_LIMIT = 10000


def _get_attr_or_key(obj, key, default_value=None):
    if default_value is None:
        default_value = []
    if isinstance(obj, dict):
        return obj.get(key, default_value)
    return getattr(obj, key, None)


def _get_artifacts(obj) -> list:
    if obj is None:
        return []
    if isinstance(obj, dict):
        if "config" in obj:
            return list(_get_attr_or_key(obj["config"], "artifacts", default_value=[]))
        return list(obj.get("artifacts", []) or [])
    if hasattr(obj, "config"):
        return list(_get_attr_or_key(obj.config, "artifacts", default_value=[]))
    return list(getattr(obj, "artifacts", []) or [])


def _artifact_key(a) -> str:
    # choose ONE stable key
    if isinstance(a, dict):
        return a.get("path") or a.get("name") or str(a)
    return getattr(a, "path", None) or getattr(a, "name", None) or str(a)


def node_from_event(ev: dict) -> str | None:
    md = ev.get("metadata") or {}
    return md.get("langgraph_node") or md.get("node") or ev.get("name") or (ev.get("data") or {}).get("node")


@dataclass
class StreamEvent:
    type: str
    payload: dict[str, Any]


class StreamingRunner:
    """
    Streaming + HITL best-practice runner:
      - No in-memory session required.
      - Resume uses Command(resume=...) so interrupt() returns the answer inside the node.
      - Runtime deps passed via config["configurable"]["deps"].
      - Durable state lives in SQLite checkpointer keyed by thread_id.
    """

    def __init__(self, *, main_graph, subgraphs: dict[str, Any], checkpointer, deps: Deps):
        self.main_graph = main_graph
        self.subgraphs = subgraphs
        self.checkpointer = checkpointer
        self.deps = deps

        if not hasattr(self.main_graph, "astream_events"):
            raise RuntimeError(
                "Compiled main graph does not expose astream_events(); pin a LangGraph version that does."
            )

    async def _update_main_state_messages(self, thread_id: str, new_messages: list):
        cfg = self._cfg(thread_id)

        if hasattr(self.main_graph, "aupdate_state"):
            await self.main_graph.aupdate_state(cfg, {"messages": new_messages})
            return

        if hasattr(self.main_graph, "update_state"):
            await asyncio.to_thread(self.main_graph.update_state, cfg, {"messages": new_messages})
            return

        raise RuntimeError("Main graph does not support update_state/aupdate_state; cannot persist artifacts safely.")

    async def _persist_main_dataset_statuses(self, thread_id: str, dataset_statuses: dict[str, Any]):
        cfg = self._cfg(thread_id)

        if hasattr(self.main_graph, "aupdate_state"):
            await self.main_graph.aupdate_state(cfg, {"datasets": dataset_statuses})
            return

        if hasattr(self.main_graph, "update_state"):
            await asyncio.to_thread(self.main_graph.update_state, cfg, {"datasets": dataset_statuses})
            return

        raise RuntimeError("Main graph does not support update_state/aupdate_state; cannot persist artifacts safely.")

    async def _persist_main_artifacts(self, thread_id: str, merged: list):
        cfg = self._cfg(thread_id)

        if hasattr(self.main_graph, "aupdate_state"):
            await self.main_graph.aupdate_state(cfg, {"artifacts": merged})
            return

        if hasattr(self.main_graph, "update_state"):
            await asyncio.to_thread(self.main_graph.update_state, cfg, {"artifacts": merged})
            return

        raise RuntimeError("Main graph does not support update_state/aupdate_state; cannot persist artifacts safely.")

    # ----------------------------
    # Config helper
    # ----------------------------
    def _cfg(self, thread_id: str, *, recursion_limit: int = 100) -> dict[str, Any]:
        return {
            "recursion_limit": recursion_limit,
            "configurable": {"thread_id": thread_id, "deps": self.deps},
        }

    # ----------------------------
    # Interrupt detection helpers
    # ----------------------------
    def _extract_interrupt_old(self, ev: Any) -> Any | None:
        if not isinstance(ev, dict):
            return None

        # v2 events usually store node outputs here
        data = ev.get("data")
        if isinstance(data, dict):
            output = data.get("output")
            if isinstance(output, dict) and "__interrupt__" in output:
                intr = output["__interrupt__"]
                # docs: returned under __interrupt__; often a list
                if isinstance(intr, list) and intr:
                    return intr[0]
                return intr

        # fallback: sometimes output is at top-level
        if "__interrupt__" in ev:
            intr = ev["__interrupt__"]
            if isinstance(intr, list) and intr:
                return intr[0]
            return intr

        return None

    def _extract_interrupt(self, ev: Any) -> Any | None:
        if not isinstance(ev, dict):
            return None

        data = ev.get("data")
        if isinstance(data, dict):
            # v2: interrupt appears under data.chunk.__interrupt__
            chunk = data.get("chunk")
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                intr = chunk["__interrupt__"]
                # in your log: it's a tuple: (Interrupt(...),)
                if isinstance(intr, (list, tuple)) and intr:
                    intr = intr[0]
                # Interrupt(...) object has .value
                if hasattr(intr, "value"):
                    return intr.value
                return intr

            # older/other shapes
            output = data.get("output")
            if isinstance(output, dict) and "__interrupt__" in output:
                intr = output["__interrupt__"]
                if isinstance(intr, (list, tuple)) and intr:
                    intr = intr[0]
                if hasattr(intr, "value"):
                    return intr.value
                return intr

        # fallback
        if "__interrupt__" in ev:
            intr = ev["__interrupt__"]
            if isinstance(intr, (list, tuple)) and intr:
                intr = intr[0]
            if hasattr(intr, "value"):
                return intr.value
            return intr

        return None

    def is_interrupt_event(self, ev: Any) -> bool:
        return self._extract_interrupt(ev) is not None

    def interrupt_payload(self, ev: Any) -> dict[str, Any]:
        intr = self._extract_interrupt(ev)
        if intr is None:
            return {"raw": str(ev)}
        return intr if isinstance(intr, dict) else {"value": intr}

    async def _load_or_create_sub_state(
        self, subgraph: Any, subgraph_name: str, run_id: str, thread_id: str, params: dict[str, Any]
    ):
        # Load from checkpoint if supported
        sub_state = None
        try:
            if hasattr(subgraph, "aget_state"):
                snap = await subgraph.aget_state(self._cfg(thread_id))
                sub_state = getattr(snap, "values", snap)
            elif hasattr(subgraph, "get_state"):
                snap = await asyncio.to_thread(subgraph.get_state, self._cfg(thread_id))
                sub_state = getattr(snap, "values", snap)
            sub_state = make_full_subgraph_state(subgraph_name, sub_state)  # type: ignore
            accessions = params.get("accessions", [])
            accessions = [x for x in accessions if x not in sub_state.config.accessions]
            sub_state.config.accessions += accessions
            sub_state.config.accessions = sorted(list(set(sub_state.config.accessions)))
        except ValidationError:
            sub_state = make_subgraph_state(subgraph_name, run_id=run_id, params=params)

        artifacts = params.get("artifacts", [])
        if len(sub_state.config.artifacts) > 0:
            artifacts = consolidate_artifacts(artifacts, sub_state.config.artifacts)
        sub_state.config.artifacts = artifacts

        # Populate datasets upon loading if not present
        for dataset in params.get("datasets", []):
            if dataset not in sub_state.datasets.keys():
                sub_state.datasets[dataset] = DatasetQualityControlState.model_validate(
                    {
                        "accession": dataset,
                        "output_dir": os.path.join(params["output_root"], dataset),
                    }
                )
        return sub_state

    # ----------------------------
    # Optional: read final state from checkpoint if supported
    # ----------------------------
    async def _get_state_if_supported(self, graph: Any, thread_id: str) -> Any | None:
        cfg = self._cfg(thread_id)
        # prefer an async get_state API if the graph exposes one
        if hasattr(graph, "aget_state"):
            snap = await graph.aget_state(config=cfg)
            return getattr(snap, "values", snap)

        if hasattr(graph, "get_state"):
            # run the blocking get_state in a thread so AsyncSqliteSaver's sync methods
            # are not called from the main loop thread.
            snap = await asyncio.to_thread(graph.get_state, cfg)
            return getattr(snap, "values", snap)

        return None

    def _populate_main_datasets(self, artifacts: list[Any]) -> dict[str, Any]:
        accession_codes = sorted(set(a.accession_code for a in artifacts if hasattr(a, "accession_code")))
        step_statuses = {accession_code: {} for accession_code in accession_codes}
        all_geo_completed = [a for a in artifacts if a.kind == "preqc_methylation_data"]
        for a in all_geo_completed:
            accession_code = a.accession_code
            step_statuses[accession_code]["geo_retrieval"] = {
                "status": "completed",
                "started_at": None,
                "finished_at": None,
                "error": None,
                "warnings": [],
            }
        all_qc_completed = [a for a in artifacts if a.kind == "postqc_methylation_data"]
        for a in all_qc_completed:
            accession_code = a.accession_code
            step_statuses[accession_code]["quality_control"] = {
                "status": "completed",
                "started_at": None,
                "finished_at": None,
                "error": None,
                "warnings": [],
            }

        all_benchmarking_completed = [a for a in artifacts if a.kind == "benchmark_summary"]
        for a in all_benchmarking_completed:
            accession_code = a.accession_code
            step_statuses[accession_code]["benchmarking"] = {
                "status": "completed",
                "started_at": None,
                "finished_at": None,
                "error": None,
                "warnings": [],
            }
        return step_statuses

    # ----------------------------
    # Run main then subgraph (stream)
    # ----------------------------
    async def stream_main_then_subgraph(
        self,
        *,
        run_id: str,
        main_state: Any,
        user_text: str,
        queue: asyncio.Queue,
        session,
    ) -> None:
        """
        Runs main routing (stream), then the selected subgraph (stream).
        Emits StreamEvent into queue for SSE.
        """
        # Put user text into main state (checkpointed)
        if hasattr(main_state, "user_request"):
            main_state.user_request = user_text
        _append_user_message(main_state, user_text)

        # ---- MAIN GRAPH STREAM ----
        main_thread = f"{run_id}:main"
        checkpointed = await self._get_state_if_supported(self.main_graph, main_thread)
        if isinstance(main_state, dict):
            main_state["artifacts"] = list((checkpointed or {}).get("artifacts", []) or [])
        elif hasattr(main_state, "artifacts"):
            main_state.artifacts = list((checkpointed or {}).get("artifacts", []) or [])
        await queue.put(StreamEvent("status", {"stage": "routing"}))
        async for ev in self.main_graph.astream_events(
            main_state,
            config=self._cfg(main_thread),
            version="v2",
        ):
            await queue.put(StreamEvent("event", {"thread_id": main_thread, "raw": ev}))
            if self.is_interrupt_event(ev):
                payload = self.interrupt_payload(ev)
                # store pending interrupt for the HTTP resume endpoint / UI
                try:
                    session.pending_interrupt = {"thread_id": main_thread, "payload": payload}
                except Exception:
                    # be defensive: do not crash the loop if session isn't available
                    pass
                # Client must keep thread_id and later resume with Command(resume=...)
                await queue.put(StreamEvent("interrupt", {"thread_id": main_thread, "payload": payload}))
                return

        # Prefer state from checkpoint; fallback to invoke
        main_out = await self._get_state_if_supported(self.main_graph, main_thread)
        if main_out is None:
            main_out = await asyncio.to_thread(self.main_graph.invoke, main_state, self._cfg(main_thread))

        pending = main_out.get("pending_reviews")
        if pending is not None:
            payload = {
                "prompt": getattr(pending, "question", None)
                or pending.get("question")
                or "I need your input to continue.",
                "context": getattr(pending, "payload", None) or pending.get("payload") or {},
                "reason": getattr(pending, "reason", None) or pending.get("reason"),
                "review_id": getattr(pending, "review_id", None) or pending.get("review_id"),
            }
            try:
                session.pending_interrupt = {"thread_id": main_thread, "payload": payload}
            except Exception:
                pass
            await queue.put(StreamEvent("interrupt", {"thread_id": main_thread, "payload": payload}))
            return

        main_datasets = main_out.get("datasets", {})
        if not main_datasets:
            main_datasets = self._populate_main_datasets(main_out.get("artifacts"))
        self._persist_main_dataset_statuses(main_thread, main_datasets)  # type: ignore

        routing_history = main_out.get("routing_history") or []
        if not routing_history:
            await queue.put(StreamEvent("final", {"message": "No routing decision found."}))
            return

        router_out = routing_history[-1]
        if getattr(router_out, "needs_clarification", False):
            # should already be caught by pending_reviews, but belt+suspenders
            await queue.put(
                StreamEvent("interrupt", {"thread_id": main_thread, "payload": {"prompt": "I need one more detail."}})
            )
            return

        router_dict = router_out.model_dump()
        subgraph_name = router_dict.get("subgraph")
        params = dict(router_dict.get("params") or {})
        checkpointed = await self._get_state_if_supported(self.main_graph, main_thread)

        # hydrate in-memory state from checkpoint
        params["output_root"] = checkpointed["default_output_root"]  # type: ignore
        params["artifacts"] = list((checkpointed or {}).get("artifacts", []) or [])
        params["datasets"] = (
            checkpointed.get("datasets", {}) if checkpointed.get("datasets", {}) else main_out.get("datasets", [])  # type: ignore
        )

        if not subgraph_name:
            await queue.put(StreamEvent("final", {"message": getattr(main_out, "next_action_hint", "Ready.")}))
            return

        # ---- SUBGRAPH STREAM ----
        sub_thread = f"{run_id}:{subgraph_name}"
        subgraph = self.subgraphs[subgraph_name]
        if not hasattr(subgraph, "astream_events"):
            raise RuntimeError(f"Subgraph {subgraph_name} does not expose astream_events(); pin compatible versions.")

        await queue.put(StreamEvent("status", {"stage": "subgraph", "name": subgraph_name}))

        sub_state = await self._load_or_create_sub_state(
            subgraph=subgraph, subgraph_name=subgraph_name, run_id=run_id, thread_id=sub_thread, params=params
        )

        sub_state.messages.append(
            main_out["messages"][-1]
        )  # pass last main message to subgraph state for context; adjust as needed

        async for ev in subgraph.astream_events(
            sub_state,
            config=self._cfg(sub_thread, recursion_limit=SUBGRAPH_RECURSION_LIMIT),
            version="v2",
        ):
            await queue.put(StreamEvent("event", {"thread_id": sub_thread, "raw": ev}))
            if isinstance(ev.get("data", {}).get("output", {}), Command):
                if ev["data"]["output"] and ev["data"]["output"].update:
                    if "main_messages" in ev["data"]["output"].update:
                        node_from_event(ev)
                        msg = [x.model_dump() for x in ev["data"]["output"].update["main_messages"]]
                        # 1) persist to main thread
                        await self.main_graph.aupdate_state(self._cfg(main_thread), values={"messages": msg})

                        # 2) notify frontend (SSE)
                        await queue.put(
                            StreamEvent(
                                "messages",
                                {
                                    "messages": [
                                        {
                                            "id": msg[0]["id"],
                                            "type": "tool",
                                            "name": msg[0]["additional_kwargs"]["name"],
                                            "tool_call_id": msg[0]["tool_call_id"],
                                            "content": msg[0]["content"],
                                            "artifact": msg[0]["artifact"],
                                        }
                                    ]
                                },
                            )
                        )

            if self.is_interrupt_event(ev):
                payload = self.interrupt_payload(ev)
                try:
                    session.pending_interrupt = {"thread_id": sub_thread, "payload": payload}
                except Exception:
                    pass
                await queue.put(StreamEvent("interrupt", {"thread_id": sub_thread, "payload": payload}))
                return

        sub_out = await self._get_state_if_supported(subgraph, sub_thread)
        if sub_out is None:
            sub_out = await asyncio.to_thread(
                subgraph.invoke, sub_state, self._cfg(sub_thread, recursion_limit=SUBGRAPH_RECURSION_LIMIT)
            )

        sub_artifacts = _get_artifacts(sub_out)
        main_artifacts = _get_artifacts(main_out)

        seen = {_artifact_key(a) for a in main_artifacts}
        merged = list(main_artifacts)

        for a in sub_artifacts:
            k = _artifact_key(a)
            if k not in seen:
                merged.append(a)
                seen.add(k)

        await self._persist_main_artifacts(main_thread, merged)

        last = sub_out["messages"][-1] if sub_out.get("messages") else None
        if isinstance(last, ToolMessage):
            await self._update_main_state_messages(main_thread, [last])
            artifact = getattr(last, "artifact", None)
            name = (last.additional_kwargs or {}).get("name") or "geoDatasetSummary"

            await queue.put(
                StreamEvent(
                    "messages",
                    {
                        "messages": [
                            {
                                "id": last.id,
                                "type": "tool",
                                "name": name,
                                "tool_call_id": last.tool_call_id,
                                "content": last.content,
                                "artifact": artifact,
                            }
                        ]
                    },
                )
            )
            await queue.put(StreamEvent("final", {"message": ""}))
        else:
            await queue.put(
                StreamEvent("final", {"message": getattr(sub_out, "next_action_hint", f"{subgraph_name} completed.")})
            )

    # ----------------------------
    # Resume after interrupt() (stream)
    # ----------------------------
    async def resume_stream(
        self,
        *,
        thread_id: str,
        payload: Any,
        queue: asyncio.Queue,
        session,
    ) -> None:
        """
        Resume from an interrupt by passing Command(resume=payload).
        This makes interrupt(payload) return `payload` inside the paused node.  [oai_citation:1‡LangChain Docs](https://docs.langchain.com/oss/python/langgraph/interrupts?utm_source=chatgpt.com)
        """
        raw_id = thread_id.split(":")[0]
        # Pick graph based on thread_id
        if thread_id.endswith(":main"):
            graph = self.main_graph
        else:
            parts = thread_id.split(":", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid thread_id format: {thread_id}")
            subgraph_name = parts[1]
            if subgraph_name not in self.subgraphs:
                raise KeyError(f"Unknown subgraph in thread_id: {subgraph_name}")
            graph = self.subgraphs[subgraph_name]

        cmd = Command(resume=payload)
        async for ev in graph.astream_events(
            cmd,
            config=self._cfg(
                thread_id,
                recursion_limit=MAIN_RECURSION_LIMIT if thread_id.endswith(":main") else SUBGRAPH_RECURSION_LIMIT,
            ),
            version="v2",
        ):
            await queue.put(StreamEvent("event", {"thread_id": thread_id, "raw": ev}))
            if isinstance(ev.get("data", {}).get("output", {}), Command):
                if ev["data"]["output"] and ev["data"]["output"].update:
                    if "main_messages" in ev["data"]["output"].update:
                        node_from_event(ev)
                        msg = [x.model_dump() for x in ev["data"]["output"].update["main_messages"]]
                        # 1) persist to main thread
                        await self.main_graph.aupdate_state(
                            self._cfg(f"{thread_id.split(':', 1)[0]}:main"), values={"messages": msg}
                        )

                        # 2) notify frontend (SSE)
                        await queue.put(
                            StreamEvent(
                                "messages",
                                {
                                    "messages": [
                                        {
                                            "id": msg[0]["id"],
                                            "type": "tool",
                                            "name": msg[0]["additional_kwargs"]["name"],
                                            "tool_call_id": msg[0]["tool_call_id"],
                                            "content": msg[0]["content"],
                                            "artifact": msg[0]["artifact"],
                                        }
                                    ]
                                },
                            )
                        )

            if self.is_interrupt_event(ev):
                payload = self.interrupt_payload(ev)
                try:
                    session.pending_interrupt = {"thread_id": thread_id, "payload": payload}
                except Exception:
                    pass
                await queue.put(StreamEvent("interrupt", {"thread_id": thread_id, "payload": payload}))
                return

        main_thread = f"{raw_id}:main"
        main_out = await self._get_state_if_supported(self.main_graph, main_thread)

        sub_out = await self._get_state_if_supported(graph, thread_id)

        sub_artifacts = _get_artifacts(sub_out)
        main_artifacts = _get_artifacts(main_out)

        seen = {_artifact_key(a) for a in main_artifacts}
        merged = list(main_artifacts)

        for a in sub_artifacts:
            k = _artifact_key(a)
            if k not in seen:
                merged.append(a)
                seen.add(k)

        await self._persist_main_artifacts(main_thread, merged)

        await queue.put(StreamEvent("final", {"message": "Resumed and completed."}))
