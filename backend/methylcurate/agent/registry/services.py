# agent/registry/services.py

from ..graphs.deps import Deps
from ..graphs.router import build_main_graph
from ..graphs.subgraphs import build_subgraphs
from ..llm.client import LLMClient, LLMConfig
from ..runtime.chat_runner import StreamingRunner


def build_services_with_checkpointer(checkpointer) -> tuple[StreamingRunner, Deps]:
    """
    Builds the services with the provided checkpointer instance. This ensures that all graphs and subgraphs share the same checkpointer, allowing for consistent state management across the entire system.

    Args:
        checkpointer: An instance of the checkpointer to be used across all graphs and subgraphs.

    Returns:
        A tuple containing the StreamingRunner instance and the Deps instance.
    """
    import os

    llm = LLMClient(
        LLMConfig(
            provider="ollama",
            model=os.getenv("OLLAMA_MODEL", "qwen3.5:397b-cloud"),
            temperature=0.0,
            top_k=1,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            reasoning=True,
            streaming=True,
        )
    )
    deps = Deps(llm=llm)

    main_builder = build_main_graph()
    subgraph_builders = build_subgraphs()

    # Compile with SAME checkpointer instance
    main_graph = main_builder.compile(checkpointer=checkpointer)
    subgraphs = {name: builder.compile(checkpointer=checkpointer) for name, builder in subgraph_builders.items()}

    runner = StreamingRunner(main_graph=main_graph, subgraphs=subgraphs, checkpointer=checkpointer, deps=deps)

    return runner, deps
