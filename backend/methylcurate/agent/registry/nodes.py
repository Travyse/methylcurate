from __future__ import annotations
__all__ = ["GRAPH_BUILDERS", "PARAM_SCHEMAS"]
from typing import Any, Callable

GRAPH_BUILDERS: dict[str, Callable[..., Any]] = {}
PARAM_SCHEMAS: dict[str, Any] = {}
