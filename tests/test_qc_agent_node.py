import importlib
import traceback


def test_qc_agent_node_imports_resolve_or_fail_on_known_deps():
    """Verify import failure (if any) is not from 'preprocess' module reference.

    The agent node may fail to import due to missing optional dependencies
    (pydantic, langgraph, etc.), but it must NOT fail because of a reference
    to the non-existent ``contracts.preprocess`` or ``tools.preprocess`` modules.
    """
    try:
        module = importlib.import_module("methylcurate.agent.nodes.qc")
        assert "quality_control_node" in dir(module)
    except ModuleNotFoundError:
        tb = traceback.format_exc()
        assert "preprocess" not in tb, f"Import references non-existent 'preprocess' module:\n{tb}"
