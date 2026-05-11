import pytest


class TestHelpNode:
    @pytest.mark.asyncio
    async def test_help_node_returns_rendered_template(self):
        from methylcurate.agent.nodes.help import help_node

        result = await help_node(state=None, config=None)
        assert "messages" in result
        assert len(result["messages"]) == 1
        content = result["messages"][0].content
        assert "MethylCurate" in content
        assert "GEO" in content
        assert "epigenetic aging clocks" in content
        assert "Help displayed." == result["next_action_hint"]


class TestHelpGraph:
    def test_help_graph_builds_without_error(self):
        from methylcurate.agent.graphs.help import build_help_graph

        graph = build_help_graph()
        assert graph is not None

    @pytest.mark.asyncio
    async def test_help_graph_compiles_and_runs(self):
        from methylcurate.agent.graphs.help import build_help_graph
        from methylcurate.agent.state.models import HelpSubgraphState

        graph = build_help_graph().compile()
        state = HelpSubgraphState(run_id="test-run", config={"output_root": "", "accessions": [], "artifacts": []})
        out = await graph.ainvoke(state)  # type: ignore[union-attr]
        assert len(out["messages"]) > 0
        assert "MethylCurate" in out["messages"][0].content


class TestHelpStateUtils:
    def test_make_help_state(self):
        from methylcurate.agent.state.utils import make_help_state

        state = make_help_state("run-1", {"output_root": "", "accessions": [], "artifacts": []})
        assert state.run_id == "run-1"
        assert state.subgraph == "help"

    def test_make_subgraph_state_handles_help(self):
        from methylcurate.agent.state.utils import make_subgraph_state

        state = make_subgraph_state("help", "run-1", {"output_root": "", "accessions": [], "artifacts": []})
        assert state.subgraph == "help"

    def test_make_full_subgraph_state_handles_help(self):
        from methylcurate.agent.state.utils import make_full_subgraph_state

        state = make_full_subgraph_state("help", {"run_id": "run-1", "config": {"output_root": "", "accessions": [], "artifacts": []}})
        assert state.subgraph == "help"

    def test_get_dataset_for_subgraph_returns_none_for_help(self):
        from methylcurate.agent.state.utils import get_dataset_for_subgraph

        assert get_dataset_for_subgraph("help") is None

    def test_help_registered_in_param_schemas(self):
        import methylcurate.agent.state.utils  # noqa: F401 — triggers PARAM_SCHEMAS registrations
        from methylcurate.agent.registry.nodes import PARAM_SCHEMAS
        from methylcurate.agent.state.models import HelpIngestionConfig

        assert "help" in PARAM_SCHEMAS
        assert PARAM_SCHEMAS["help"] is HelpIngestionConfig


class TestHelpRouterContracts:
    def test_help_is_valid_subgraph_name(self):
        from typing import get_args

        from methylcurate.contracts.router import SubgraphName

        valid_names = set(get_args(SubgraphName))
        assert "help" in valid_names

    def test_router_output_validates_help_subgraph(self):
        from methylcurate.contracts.router import RouterOutput

        out = RouterOutput(subgraph="help", params={}, confidence=1.0, needs_clarification=False, reasons=["user asked for help"])
        assert out.subgraph == "help"

    def test_help_registered_in_graph_builders(self):
        import methylcurate.agent.graphs.subgraphs  # noqa: F401 — triggers GRAPH_BUILDERS registrations
        from methylcurate.agent.registry.nodes import GRAPH_BUILDERS

        assert "help" in GRAPH_BUILDERS
