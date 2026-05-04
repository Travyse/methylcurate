from __future__ import annotations

__all__ = ["GRAPH_BUILDERS", "PARAM_SCHEMAS"]
from collections.abc import Callable
from typing import Any

GRAPH_BUILDERS: dict[str, Callable[..., Any]] = {}
PARAM_SCHEMAS: dict[str, Any] = {}
