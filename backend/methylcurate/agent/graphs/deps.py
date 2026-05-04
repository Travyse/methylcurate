from dataclasses import dataclass
from ..llm.client import LLMClient


@dataclass
class Deps:
    llm: LLMClient
    # later add logging, metrics, tracing, etc.
