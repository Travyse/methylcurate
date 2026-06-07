"""Registry for MethylCurate agent components.

Maps available actions, tools, nodes, and workflows to their
implementations and provides the service builder for graph execution.
"""

__all__ = ["nodes", "services"]


def __dir__():
    return __all__


def __getattr__(name: str):
    _modules = {"nodes", "services"}
    if name not in _modules:
        raise AttributeError(f"module 'methylcurate.agent.registry' has no attribute {name!r}")
    return __import__(f"methylcurate.agent.registry.{name}", fromlist=[name])
