import argparse
import asyncio
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore

from .agent.registry.services import build_services_with_checkpointer
from .agent.state.utils import make_main_state
from .api.session import SessionStore


# ----------------------------
# OpenAI-style chunk helpers
# ----------------------------
def openai_chunk(run_id: str, content: str) -> dict:
    return {
        "id": run_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "assistant",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }


@dataclass(frozen=True)
class CLIConfig:
    checkpoints_db: str
    default_output_root: str
    params: dict[str, Any]


def _parse_kv_params(items: list[str]) -> dict[str, str]:
    """
    Parse repeatable --param key=value flags into a dict.
    Last writer wins if the same key is repeated.
    """
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--param must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        # Keep v as-is except for trimming whitespace at the ends.
        v = v.strip()
        if not k:
            raise SystemExit(f"--param key cannot be empty, got: {item!r}")
        out[k] = v
    return out


def parse_args() -> CLIConfig:
    parser = argparse.ArgumentParser(prog="agentic_ai_aging_clock_helper")

    parser.add_argument(
        "--checkpoints-db",
        default="checkpoints.db",
        help="SQLite DB path for LangGraph checkpoints.",
    )

    parser.add_argument(
        "--default-output-root",
        default="outputs/",
        help="Default output root used to initialize MainState (string path).",
    )

    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra params as key=value. Repeatable. Example: --param output_root=outputs/qc",
    )

    ns = parser.parse_args()

    return CLIConfig(
        checkpoints_db=ns.checkpoints_db,
        default_output_root=ns.default_output_root,
        params=_parse_kv_params(ns.param),
    )


# ----------------------------
# CLI runner
# ----------------------------
async def chat_loop(cfg: CLIConfig):
    store = SessionStore()

    print(f"Using {cfg.default_output_root} as output path")
    os.makedirs(cfg.default_output_root, exist_ok=True)
    # Lifespan logic (copied from FastAPI)
    checkpointer_cm = AsyncSqliteSaver.from_conn_string(cfg.checkpoints_db)
    checkpointer = await checkpointer_cm.__aenter__()

    runner, deps = build_services_with_checkpointer(checkpointer)

    try:
        thread_id = None

        while True:
            user_text = input("\n> ").strip()
            if not user_text:
                continue
            if user_text in ("/quit", "/exit"):
                break

            run_id = thread_id or uuid4().hex
            session = store.get(run_id) if store.exists(run_id) else store.create(run_id)

            if session.main_state is None:
                session.main_state = make_main_state(run_id=run_id, default_output_root=cfg.default_output_root)

            if session.task and not session.task.done():
                print("[busy]")
                continue

            async def run(
                session: Session = session,  # type: ignore
                user_text: str = user_text,
                run_id: str = run_id,
            ):
                try:
                    if session.pending_interrupt is not None:
                        pending = session.pending_interrupt
                        session.pending_interrupt = None

                        await runner.resume_stream(  # type: ignore
                            thread_id=pending["thread_id"],
                            answer=user_text,  # type: ignore
                            queue=session.queue,
                            session=session,
                        )
                    else:
                        await runner.stream_main_then_subgraph(
                            run_id=run_id,
                            main_state=session.main_state,
                            user_text=user_text,
                            queue=session.queue,
                            session=session,
                        )
                except Exception as e:
                    traceback.print_exc()
                    await session.queue.put(
                        type(
                            "Ev",
                            (),
                            {
                                "type": "final",
                                "payload": {"message": f"Runner crashed: {e}"},
                            },
                        )
                    )

            session.task = asyncio.create_task(run())

            # Consume events exactly like /api/chat
            while True:
                ev = await session.queue.get()

                if ev.type == "final":
                    msg = ""
                    if isinstance(ev.payload, dict):
                        msg = ev.payload.get("message") or ev.payload.get("text") or ""
                    elif isinstance(ev.payload, str):
                        msg = ev.payload
                    else:
                        msg = str(ev.payload)

                    if msg:
                        chunk = openai_chunk(run_id, msg)
                        print(chunk["choices"][0]["delta"]["content"], end="", flush=True)

                    print()  # newline
                    thread_id = run_id
                    break

                if ev.type == "interrupt":
                    wrapper = ev.payload if isinstance(ev.payload, dict) else {}

                    # Your runner wraps the actual interrupt payload under "payload"
                    inner = wrapper.get("payload") if isinstance(wrapper.get("payload"), dict) else wrapper

                    prompt = inner.get("prompt") or "I need your input to continue."  # type: ignore

                    chunk = openai_chunk(run_id, prompt)
                    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                    print()
                    thread_id = run_id
                    break

                # Stream intermediate text events
                text = None
                if isinstance(ev.payload, dict):
                    text = ev.payload.get("text") or ev.payload.get("message") or ev.payload.get("content")
                if text:
                    chunk = openai_chunk(run_id, text)
                    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)

    finally:
        await checkpointer_cm.__aexit__(None, None, None)


if __name__ == "__main__":
    cfg = parse_args()
    asyncio.run(chat_loop(cfg))
    # python -m agentic_ai_aging_clock_helper.agentic_ai_aging_clock_helper --default-output-root "/Users/travyse/Documents/Research/Agentic-AI/dev-testing"
