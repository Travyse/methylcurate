# api/server.py
import asyncio
import json
import time
import os
import traceback
from datetime import datetime
from contextlib import asynccontextmanager
from uuid import uuid4
from typing import Optional, Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import sqlite3

from .session import SessionStore
from .file_parser import _append_accessions_to_text, _extract_accessions_from_files
from ..agent.state.utils import make_main_state
from ..agent.registry.services import build_services_with_checkpointer


# ----------------------------
# SSE helpers (LangGraph-SDK-ish for assistant-ui)
# ----------------------------
def sse_event(event: str, data: Any) -> bytes:
    """
    Create a Server-Sent Event (SSE) message.

    Args:
        event (str): The event type.
        data (Any): The event data.

    Returns:
        bytes: The SSE message as bytes.
    """
    # assistant-ui parseSse() reads `event:` and `data:`
    return (f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n").encode("utf-8")


def sse_keepalive() -> bytes:
    """
    Create a Server-Sent Event (SSE) keep-alive message.

    Returns:
        bytes: The SSE keep-alive message as bytes.
    """
    return b": keep-alive\n\n"


# ----------------------------
# Request models (LangGraph SDK-ish)
# ----------------------------
class ThreadsCreateResponse(BaseModel):
    """
    Response model for creating a new thread.

    Attributes:
        thread_id (str): The unique identifier for the created thread.
    """

    thread_id: str


class ThreadStateResponse(BaseModel):
    """
    Response model for retrieving the state of a thread.

    Attributes:
        values (Dict[str, Any]): The state values of the thread.
    """

    values: Dict[str, Any]


class StreamRequest(BaseModel):
    """
    Request model for streaming data to a thread.

    Attributes:
        input (Optional[Dict[str, Any]]): The input data for the stream, which may include messages and other relevant information.
        command (Optional[Dict[str, Any]]): The command data for the stream, which may include instructions for resuming or controlling the stream.
        streamMode (Optional[List[str]]): The modes for streaming, which may specify how the stream should be
    """

    # assistant-ui sends:
    # { input: { messages: [...] } | null, command: ..., streamMode: ["messages","updates"] }
    input: Optional[Dict[str, Any]] = None
    command: Optional[Dict[str, Any]] = None
    streamMode: Optional[List[str]] = None


class ThreadListItem(BaseModel):
    """
    Model representing a single thread in the thread list.

    Attributes:
        thread_id (str): The unique identifier for the thread.
        status (Literal["regular", "archived"]): The status of the thread, which can be either "regular" or "archived".
        title (Optional[str]): An optional title for the thread, which can be used for display purposes.
        updatedAt (Optional[datetime]): An optional timestamp indicating when the thread was last updated.
    """

    thread_id: str
    status: Literal["regular", "archived"] = "regular"
    title: Optional[str] = None
    updatedAt: Optional[datetime] = None  # or str/float if you prefer


class ThreadListResponse(BaseModel):
    """
    Response model for listing threads.

    Attributes:
        threads (List[ThreadListItem]): A list of threads.
    """

    threads: List[ThreadListItem]


# ----------------------------
# App + Lifespan
# ----------------------------
store = SessionStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic for the FastAPI application.
    """
    # Get path from environment variable OUTPUT_DIR
    output_dir = os.getenv("AAIS_OUTPUT_DIR", "outputs/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    checkpointer_cm = AsyncSqliteSaver.from_conn_string(os.path.join(output_dir, "checkpoints.db"))
    checkpointer = await checkpointer_cm.__aenter__()

    runner, deps = build_services_with_checkpointer(checkpointer)

    app.state.checkpointer_cm = checkpointer_cm
    app.state.checkpointer = checkpointer
    app.state.runner = runner
    app.state.deps = deps
    app.state.output_dir = output_dir

    try:
        yield
    finally:
        await checkpointer_cm.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# GET /threads  (list threads)
# ----------------------------
def list_root_threads_from_checkpoints(db_path: str) -> list[str]:
    """
    List the root threads from the checkpoints database.

    Args:
        db_path (str): The path to the checkpoints database.

    Returns:
        list[str]: A list of root thread IDs.
    """
    if not os.path.exists(db_path):
        return []
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    try:
        rows = cur.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE '%:main'").fetchall()
    except sqlite3.OperationalError as e:
        if "no such table: checkpoints" in str(e):
            return []
        raise
    finally:
        con.close()

    roots = sorted({tid.split(":", 1)[0] for (tid,) in rows})
    return roots


@app.get("/threads", response_model=ThreadListResponse)
async def list_threads(_: Request):
    """
    List all threads.

    Args:
        _: Request object (not used).

    Returns:
        ThreadListResponse: A response model containing the list of threads.
    """
    # Assistant-UI wants a thread_id. You can accept client-provided IDs too,
    # but this endpoint is for server-generated IDs.
    thread_ids = list_root_threads_from_checkpoints(os.path.join(app.state.output_dir, "checkpoints.db"))
    runner = _.app.state.runner
    thread_dict = {"threads": []}
    for thread_id in thread_ids:
        # Your runner checkpoints the "main" graph under "{run_id}:main"
        main_thread = f"{thread_id}:main"

        # Prefer reading from checkpoint via graph get_state
        cfg = {"configurable": {"thread_id": main_thread, "deps": _.app.state.deps}}

        values = None
        try:
            if hasattr(runner.main_graph, "aget_state"):
                snap = await runner.main_graph.aget_state(cfg)
                values = getattr(snap, "values", snap)
            elif hasattr(runner.main_graph, "get_state"):
                snap = await asyncio.to_thread(runner.main_graph.get_state, cfg)
                values = getattr(snap, "values", snap)
        except Exception:
            # If state doesn't exist yet, return empty
            values = None

        messages = []
        if isinstance(values, dict):
            messages = values.get("messages") or []

        thread_dict["threads"].append(
            {
                "thread_id": thread_id,
                "status": "regular",
                "title": f"{messages[-1].content[:8]}..." if messages else f"Empty Thread",
                "updatedAt": messages[-1].additional_kwargs.get("created_at") if messages else None,
            }
        )
    return thread_dict


# ----------------------------
# POST /threads  (create thread)
# ----------------------------
@app.post("/threads", response_model=ThreadsCreateResponse)
async def create_thread(_: Request):
    """
    Create a new thread.

    Args:
        _: Request object (not used).

    Returns:
        ThreadsCreateResponse: A response model containing the newly created thread ID.
    """
    # Assistant-UI wants a thread_id. You can accept client-provided IDs too,
    # but this endpoint is for server-generated IDs.
    return {"thread_id": uuid4().hex}


# ----------------------------
# DELETE /threads/{thread_id}  (delete thread)
# ----------------------------
@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str, _: Request):
    """
    Delete a thread.

    Args:
        thread_id (str): The unique identifier for the thread.
        _: Request object (not used).

    Returns:
        dict: A dictionary containing the status and the deleted thread ID.
    """
    db_path = os.path.join(_.app.state.output_dir, "checkpoints.db")

    like = f"{thread_id}:%"

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # ensure thread exists (optional but nice)
    exists = cur.execute(
        "SELECT 1 FROM checkpoints WHERE thread_id LIKE ? LIMIT 1",
        (like,),
    ).fetchone()

    if not exists:
        con.close()
        raise HTTPException(status_code=404, detail="Thread not found")

    # IMPORTANT: writes references checkpoints keys; delete writes first
    cur.execute("DELETE FROM writes WHERE thread_id LIKE ?", (like,))
    cur.execute("DELETE FROM checkpoints WHERE thread_id LIKE ?", (like,))
    con.commit()
    con.close()

    # clear in-memory session if you keep one
    try:
        store.delete(thread_id)  # if your SessionStore supports it
    except Exception:
        pass

    return {"ok": True, "thread_id": thread_id}


# ----------------------------
# GET /threads/{thread_id}/state
# ----------------------------
@app.get("/threads/{thread_id}/state", response_model=ThreadStateResponse)
async def get_thread_state(thread_id: str, request: Request):
    """
    Get the state of a thread.

    Args:
        thread_id (str): The unique identifier for the thread.
        request (Request): The FastAPI request object.

    Returns:
        ThreadStateResponse: A response model containing the thread state.
    """
    runner = request.app.state.runner

    # Your runner checkpoints the "main" graph under "{run_id}:main"
    main_thread = f"{thread_id}:main"

    # Prefer reading from checkpoint via graph get_state
    cfg = {"configurable": {"thread_id": main_thread, "deps": request.app.state.deps}}

    values = None
    try:
        if hasattr(runner.main_graph, "aget_state"):
            snap = await runner.main_graph.aget_state(cfg)
            values = getattr(snap, "values", snap)
        elif hasattr(runner.main_graph, "get_state"):
            snap = await asyncio.to_thread(runner.main_graph.get_state, cfg)
            values = getattr(snap, "values", snap)
    except Exception:
        # If state doesn't exist yet, return empty
        values = None

    messages = []
    if isinstance(values, dict):
        messages = values.get("messages") or []
        # Do not include system messages
        messages = [m for m in messages if m.type != "system"]

    # assistant-ui expects { values: { messages: [...] } }
    return {"values": {"messages": messages}}


# ----------------------------
# POST /threads/{thread_id}/stream   (SSE)
# ----------------------------
@app.post("/threads/{thread_id}/runs/stream")
async def stream_thread(thread_id: str, req: StreamRequest, request: Request):
    """
    Stream a thread.

    Args:
        thread_id (str): The unique identifier for the thread.
        req (StreamRequest): The request object containing input and command data.
        request (Request): The FastAPI request object.

    Returns:
        StreamingResponse: A streaming response for the thread.
    """
    runner = request.app.state.runner

    # We reuse your SessionStore for the streaming queue + background task.
    session = store.get(thread_id) if store.exists(thread_id) else store.create(thread_id)

    if session.main_state is None:
        from ..agent.state.models import MainState

        session.main_state = make_main_state(run_id=thread_id, default_output_root=request.app.state.output_dir)

    if session.task and not session.task.done():
        raise HTTPException(409, "Run is busy")

    # Extract last user message text from req.input.messages
    user_text = ""
    input_obj = req.input or {}
    input_messages = (input_obj.get("messages") or []) if isinstance(input_obj, dict) else []

    if input_messages:
        last = input_messages[-1]
        if isinstance(last, dict):
            user_text = last.get("content") or ""
            if isinstance(user_text, list):
                parts = []
                for p in user_text:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                user_text = "".join(parts)
        else:
            user_text = str(last)

    files_raw = (input_obj.get("files") or []) if isinstance(input_obj, dict) else []
    user_text = _append_accessions_to_text(user_text, _extract_accessions_from_files(files_raw))

    await session.queue.put(type("Ev", (), {"type": "status", "payload": {"_emit_user": user_text}}))

    async def run_bg():
        """
        Background task to handle streaming of thread events.

        This function manages the resumption of interrupted streams and the
        initiation of new streams based on the user's input and the current
        session state.

        """
        resume_payload = None
        if isinstance(req.command, dict):
            resume_payload = req.command.get("resume")

        if resume_payload is None and isinstance(req.input, dict):
            resume_payload = req.input.get("data")

        if resume_payload is None and session.pending_interrupt is not None:
            if user_text.strip():
                resume_payload = user_text

        try:
            if session.pending_interrupt is not None:
                pending = session.pending_interrupt
                session.pending_interrupt = None
                await runner.resume_stream(
                    thread_id=pending["thread_id"],
                    payload=resume_payload,
                    queue=session.queue,
                    session=session,
                )
            else:
                await runner.stream_main_then_subgraph(
                    run_id=thread_id,
                    main_state=session.main_state,
                    user_text=user_text,
                    queue=session.queue,
                    session=session,
                )
        except Exception as e:
            traceback.print_exc()
            try:
                await session.queue.put(
                    type("Ev", (), {"type": "final", "payload": {"message": f"Runner crashed: {e}"}})
                )
            except Exception:
                pass

    session.task = asyncio.create_task(run_bg())

    async def event_gen():
        """
        Generator function to yield Server-Sent Events (SSE) for the thread stream.
        """
        while True:
            if await request.is_disconnected():
                break

            try:
                ev = await asyncio.wait_for(session.queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield sse_keepalive()
                continue

            # Your runner emits: status/event/interrupt/final
            if ev.type == "messages":
                payload = ev.payload if isinstance(ev.payload, dict) else {}
                yield sse_event("messages", payload)
                continue

            if ev.type == "final":
                msg = ""
                if isinstance(ev.payload, dict):
                    msg = ev.payload.get("message") or ev.payload.get("text") or ""
                elif isinstance(ev.payload, str):
                    msg = ev.payload
                else:
                    msg = str(ev.payload)

                if msg:
                    # Emit assistant message
                    yield sse_event(
                        "messages",
                        {
                            "messages": [
                                {
                                    "type": "ai",
                                    "content": [{"type": "text", "text": msg}],
                                }
                            ]
                        },
                    )
                yield sse_event("done", {})
                break

            if ev.type == "interrupt":
                prompt = "I need your input to continue."
                if isinstance(ev.payload, dict):
                    prompt = (
                        (ev.payload.get("payload") or {}).get("prompt")
                        or ev.payload.get("payload", {}).get("content")
                        or ev.payload.get("prompt")
                        or prompt
                    )

                # Emit a message prompting for input
                if ev.payload.get("payload", {}).get("type", None) == "tool":
                    yield sse_event(
                        "messages",
                        {"messages": [ev.payload["payload"]]},
                    )
                else:
                    yield sse_event(
                        "messages",
                        {
                            "messages": [
                                {
                                    "type": "ai",
                                    "content": [{"type": "text", "text": prompt}],
                                }
                            ]
                        },
                    )

                # Optional: also emit a structured interrupt event if you want the UI to render custom HITL
                yield sse_event(
                    "updates",
                    {
                        "interrupt": {
                            "thread_id": ev.payload.get("thread_id") if isinstance(ev.payload, dict) else None,
                            "payload": ev.payload if isinstance(ev.payload, dict) else {"value": str(ev.payload)},
                        }
                    },
                )

                yield sse_event("done", {})
                break

            elif ev.type == "status" and isinstance(ev.payload, dict) and "_emit_user" in ev.payload:
                yield sse_event("messages", {"messages": [{"type": "human", "content": ev.payload["_emit_user"]}]})
                continue

            # Optional: if you start emitting incremental text deltas from the runner,
            # you can map them here as additional "messages" events.
            # For now, ignore status/raw events.

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
