
from typing import Literal, get_args
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
from ..utils.helper import NonEmptyStr
from ..agent.registry.nodes import GRAPH_BUILDERS, PARAM_SCHEMAS

SubgraphName = Literal["geo_retrieval", "harmonization", "quality_control", "benchmarking"]

class RouterOutput(BaseModel):
    """
    Represents the output of the router.

    Attributes:
        - subgraph: The name of the selected subgraph.
        - params: Parameters for the selected subgraph.
        - confidence: Confidence score of the routing decision.
        - needs_clarification: Whether the router needs clarification from the user. Set true if confidence is <= 0.5.
        - clarification_question: Clarification question to ask the user if needs_clarification is true. The goal of the question is to improve the router's ability to route correctly.
        - reasons: List of reasons for the routing decision, useful for explainability.
    
    Validation:
        - subgraph must be one of the defined subgraph names.
        - params must conform to the schema defined for the selected subgraph.
    """
    subgraph: SubgraphName
    params: Dict = Field(default_factory=dict, description="Parameters for the selected subgraph")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0.0 to 1.0 confidence score of the routing decision. Set < 0.5 if data is missing.")
    needs_clarification: bool = Field(default=False, description="Whether the router needs clarification from the user. Set true if confidence is <= 0.5.")
    clarification_question: Optional[NonEmptyStr] = Field(default=None, description="Clarification question to ask the user if needs_clarification is true. The goal of the question is to improve your ability to route correctly.")
    reasons: List[NonEmptyStr] = Field(default_factory=list, description="List of reasons for the routing decision, useful for explainability.")

    @field_validator("subgraph", mode="after")
    def validate_subgraph(cls, v):
        """
        Validates that the subgraph name is one of the defined subgraph names. This ensures that the router can only route to valid subgraphs.
        """
        if v not in get_args(SubgraphName):
            raise ValueError(f"Invalid subgraph name: {v}")
        return v
    
    @field_validator("params", mode="after")
    def validate_params(cls, v, values):
        """
        Validates the params based on the selected subgraph. This ensures that the params conform to the schema defined for the selected subgraph.
        """
        subgraph = values.data.get("subgraph")
        if subgraph is None:
            raise ValueError("subgraph must be set before params validation")
        schema = PARAM_SCHEMAS.get(subgraph)
        try:
            schema.model_validate(v)
        except Exception as e:
            raise ValueError(f"Invalid params for subgraph {subgraph}: {e}")

        # Further param validation can be added here if needed
        return v