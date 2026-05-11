from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...utils.provenance import ProvenanceLogger, ProvenanceRegistry


@dataclass
class Deps:
    llm: Any
    provenance: ProvenanceRegistry | None = None
    # later add metrics, tracing, etc.

    def get_provenance(self, thread_id: str) -> ProvenanceLogger | None:
        if self.provenance is None:
            return None
        root_id = thread_id.split(":")[0]
        return self.provenance.get(root_id)
