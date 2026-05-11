# agent/registry/services.py

from __future__ import annotations

from ..graphs.deps import Deps
from ..graphs.router import build_main_graph
from ..graphs.subgraphs import build_subgraphs
from ..llm.client import LLMClient, LLMConfig
from ..runtime.chat_runner import StreamingRunner

_PROVENANCE_ENABLED = True


def _make_provenance_default(output_root: str | None = None):
    if not _PROVENANCE_ENABLED:
        return None
    try:
        from ...utils.provenance import ProvenanceRegistry

        return ProvenanceRegistry()
    except ImportError:
        return None


def build_services_with_checkpointer(checkpointer, config_path: str | None = None) -> tuple[StreamingRunner, Deps]:
    import os

    config_path = config_path or os.getenv("LLM_CONFIG_PATH")
    llm_config = LLMConfig.from_yaml(config_path) if config_path else None

    if llm_config is None:
        llm_config = LLMConfig(
            provider="ollama",
            model=os.getenv("OLLAMA_MODEL", "qwen3.5:397b-cloud"),
            temperature=0.0,
            top_k=1,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            reasoning=True,
            streaming=True,
        )

    base_llm = LLMClient(llm_config)
    provenance = _make_provenance_default()
    if provenance is not None:
        from ..llm.logged_client import LoggedLLMClient

        llm = LoggedLLMClient(base_llm)
    else:
        llm = base_llm
    deps = Deps(llm=llm, provenance=provenance)  # pyright: ignore[reportArgumentType]

    main_builder = build_main_graph()
    subgraph_builders = build_subgraphs()

    main_graph = main_builder.compile(checkpointer=checkpointer)
    subgraphs = {name: builder.compile(checkpointer=checkpointer) for name, builder in subgraph_builders.items()}

    runner = StreamingRunner(main_graph=main_graph, subgraphs=subgraphs, checkpointer=checkpointer, deps=deps)

    return runner, deps
