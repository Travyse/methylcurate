__all__ = []

import os

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import List, Dict, Optional, Literal, Annotated, Union, Any
from operator import add
from pydantic import BaseModel, Field, field_validator, model_validator
from ...contracts.harmonize import HumanReadableConceptInput, LabelMappingSet
from ...contracts.harmonize import HarmonizationConcept
from ...contracts.geo import Concept as GEOConcepts
from ...contracts.geo import GEODownloadResult, GEOMetadataExtractionInput, GEOMetadataExtractionResult, MetadataSummary, GeoSampleLevelMetadataBatch
from ...contracts.qc import PreprocessDataInput, PreprocessDataResult, SampleLevelQCInput, SampleLevelQCResult, CpGLevelQCInput, CpGLevelQCResult, DNAmQCInput, DNAmQCResult, InterarrayCorrelationQCInput, InterarrayCorrelationQCResult
from ...contracts.common import ArtifactRef, StepStatus, HumanReviewRequest, HumanReviewDecision
from ...contracts.clocks import MethylationClocks
from ...contracts.router import RouterOutput
from ...utils.helper import NonEmptyStr
from ..registry.nodes import GRAPH_BUILDERS, PARAM_SCHEMAS


def merge_dict(left: dict, right: dict) -> dict:
    return {**left, **right}
def _validate_geo_accession(v: str) -> str:
    """Ensure a value is a valid GEO Series accession (must start with 'GSE')."""
    if not v or not isinstance(v, str):
        raise ValueError("each accession must be a non-empty string")
    if not v.upper().startswith("GSE"):
        raise ValueError(f"accession {v} is not a valid GEO Series accession (must start with 'GSE')")
    return v

def _validate_accessions_list(v):
    """Ensure accessions is a non-empty list of valid GEO Series accessions."""
    if not v or not isinstance(v, list) or len(v) == 0:
        raise ValueError("accessions must be a non-empty list of strings")
    for acc in v:
        _validate_geo_accession(acc)
    return v

def _validate_output_root(v):
    """Ensure output_root is a valid existing directory path."""
    if not v or not isinstance(v, str) or not os.path.exists(v):
        raise ValueError("output_root must be a non-empty string and existing directory")
    return v



# ----------------------------
# GEO Subgraph States
# ----------------------------

class GEOIngestionConfig(BaseModel):
    output_root: NonEmptyStr
    accessions: List[NonEmptyStr]
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    
    @field_validator("accessions", mode="before")
    @classmethod
    def validate_accessions(cls, v):
        return _validate_accessions_list(v)

class GPLMetadata(BaseModel):
    platform_id: Optional[NonEmptyStr] = None
    title: Optional[NonEmptyStr] = None

class RefinementTracking(BaseModel):
    num_retries: int = 0
    formatting_history: Optional[List[List[str]]] = Field(default_factory=list) # Flagged concepts
    parsing_history: Optional[List[List[str]]] = Field(default_factory=list) # Flagged concepts
    example_errors: Optional[List[Dict[str, Any]]] = Field(default_factory=list) # Store examples of errors for review

class GeoDatasetState(BaseModel):
    accession: NonEmptyStr
    output_dir: NonEmptyStr

    steps: Dict[str, StepStatus] = Field(default_factory=lambda: {
        "download_soft": StepStatus(),
        "check_valid_dataset": StepStatus(),
        "extract_metadata_schema": StepStatus(),
        "extract_data": StepStatus(),
        "refine_metadata_schema": StepStatus(),
        "supplementary_file_check": StepStatus(),
    })

    status: Literal["not_started", "in_progress", "failed", "completed"] = Field(default="not_started")
    is_hidden: bool = Field(default=False) 
    is_valid_dataset: bool = Field(default=True)

    download_result: Optional[GEODownloadResult] = None

    metadata_extraction_input: Optional[GEOMetadataExtractionInput] = None
    metadata_extraction_result: Optional[GEOMetadataExtractionResult] = None

    metadata_summary: Optional[MetadataSummary] = None
    refinement_history: RefinementTracking = Field(default_factory=RefinementTracking)

    artifacts: List[ArtifactRef] = Field(default_factory=list)
    platform_metadata: Optional[GPLMetadata] = None
    supplementary_files: Optional[List[str]] = Field(default_factory=list)
    supplementary_data: Optional[Annotated[Dict[NonEmptyStr, Literal['pending', 'running', 'completed', 'failed']], merge_dict]] = None

    pending_review: Optional[List[HumanReviewRequest]] = None
    review_history: List[HumanReviewDecision] = Field(default_factory=list)

    warnings: List[NonEmptyStr] = Field(default_factory=list)
    errors: List[NonEmptyStr] = Field(default_factory=list)
    
    @field_validator("accession", mode="before")
    @classmethod
    def validate_accession(cls, v):
        return _validate_geo_accession(v)

class GeoIngestionSubgraphState(BaseModel):
    run_id: NonEmptyStr
    subgraph: Literal["geo_retrieval"] = "geo_retrieval"
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    main_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    llm_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    concept_messages: Dict[GEOConcepts, list[AnyMessage]] = Field(default_factory=dict)

    config: GEOIngestionConfig

    # per-accession states
    datasets: Annotated[Dict[NonEmptyStr, GeoDatasetState], merge_dict]


    # optional global artifacts/logs
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    warnings: List[NonEmptyStr] = Field(default_factory=list)
    errors: List[NonEmptyStr] = Field(default_factory=list)

    # UX/routing
    next_action_hint: Optional[NonEmptyStr] = None

    @model_validator(mode="after")
    def validate_required_datasets(self):
        if not self.datasets:
            raise ValueError("datasets must be a non-empty dictionary")
        for key in self.datasets.keys():
            _validate_geo_accession(key)
        return self

# ----------------------------
# Harmonization Subgraph States
# ----------------------------

class HarmonizationIngestionConfig(BaseModel):
    output_root: NonEmptyStr
    accessions: List[NonEmptyStr]
    artifacts: List[ArtifactRef] = Field(default_factory=list)

    @field_validator("output_root", mode="before")
    @classmethod
    def validate_output_root(cls, v):
        return _validate_output_root(v)
    
    @field_validator("accessions", mode="before")
    @classmethod
    def validate_accessions(cls, v):
        return _validate_accessions_list(v)

class HarmonizationDatasetState(BaseModel):
    accession: NonEmptyStr
    output_dir: NonEmptyStr
    status: Literal["not_started", "in_progress", "failed", "completed"] = Field(default="not_started")
    steps: Dict[NonEmptyStr, StepStatus] = Field(default_factory=lambda: {
        "map_disease_labels_to_ontology": StepStatus(),
        "group_disease_labels": StepStatus(),
        "map_tissue_labels_to_ontology": StepStatus(),
        "group_tissue_labels": StepStatus(),
        "map_cell_type_labels_to_ontology": StepStatus(),
        "harmonize_sex_labels": StepStatus(),
    })

    harmonization_input: Optional[HumanReadableConceptInput] = None

    # Disease Harmonization
    disease_label_guessing: Optional[LabelMappingSet] = None
    disease_label_mapping: Optional[LabelMappingSet] = None

    # Tissue Harmonization
    tissue_label_guessing: Optional[LabelMappingSet] = None
    tissue_label_mapping: Optional[LabelMappingSet] = None

    # Cell Type Harmonization
    cell_type_label_guessing: Optional[LabelMappingSet] = None
    cell_type_label_mapping: Optional[LabelMappingSet] = None

    sex_mapping: Optional[LabelMappingSet] = None

class HarmonizationSubgraphState(BaseModel):
    run_id: NonEmptyStr
    subgraph: Literal["harmonization"] = "harmonization"
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    main_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    config: HarmonizationIngestionConfig

    # per-concept states
    datasets: Annotated[Dict[NonEmptyStr, HarmonizationDatasetState], merge_dict] = Field(default_factory=dict)
    disease_group_mapping: Optional[LabelMappingSet] = None
    tissue_group_mapping: Optional[LabelMappingSet] = None

    # optional global artifacts/logs
    warnings: List[NonEmptyStr] = Field(default_factory=list)
    errors: List[NonEmptyStr] = Field(default_factory=list)

    # UX/routing
    next_action_hint: Optional[NonEmptyStr] = None

# ----------------------------
# Quality Control Subgraph States
# ----------------------------

class QualityControlIngestionConfig(BaseModel):
    output_root: NonEmptyStr
    accessions: List[NonEmptyStr]
    artifacts: List[ArtifactRef] = Field(default_factory=list)

    @field_validator("output_root", mode="before")
    @classmethod
    def validate_output_root(cls, v):
        return _validate_output_root(v)
    
    @field_validator("accessions", mode="before")
    @classmethod
    def validate_accessions(cls, v):
        return _validate_accessions_list(v)

class DatasetQualityControlState(BaseModel):
    accession: NonEmptyStr
    output_dir: NonEmptyStr
    status: Literal["not_started", "in_progress", "failed", "completed"] = Field(default="not_started")
    steps: Dict[NonEmptyStr, StepStatus] = Field(default_factory=lambda: {
        "quality_control": StepStatus()
    })
    
    data_conversion_result: Optional[PreprocessDataResult] = None
    sample_level_qc_result: Optional[SampleLevelQCResult] = None
    cpg_level_qc_result: Optional[CpGLevelQCResult] = None
    dnam_qc_result: Optional[DNAmQCResult] = None
    interarray_correlation_qc_result: Optional[InterarrayCorrelationQCResult] = None

    artifacts: List[ArtifactRef] = Field(default_factory=list)
    
    @field_validator("accession", mode="before")
    @classmethod
    def validate_accessions(cls, v):
        return _validate_geo_accession(v)

class QualityControlSubgraphState(BaseModel):
    run_id: NonEmptyStr
    subgraph: Literal["quality_control"] = "quality_control"
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    main_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    config: QualityControlIngestionConfig
    datasets: Annotated[Dict[NonEmptyStr, DatasetQualityControlState], merge_dict] = Field(default_factory=dict)
    
    data_conversion_input: Optional[PreprocessDataInput] = None
    sample_level_qc_input: Optional[SampleLevelQCInput] = None
    cpg_level_qc_input: Optional[CpGLevelQCInput] = None
    dnam_qc_input: Optional[DNAmQCInput] = None
    interarray_correlation_qc_input: Optional[InterarrayCorrelationQCInput] = None

    # optional global artifacts/logs
    warnings: List[NonEmptyStr] = Field(default_factory=list)
    errors: List[NonEmptyStr] = Field(default_factory=list)

    # UX/routing
    next_action_hint: Optional[NonEmptyStr] = None

    @model_validator(mode="after")
    def validate_required_datasets(self):
        if not self.datasets:
            raise ValueError("datasets must be a non-empty dictionary")
        for key in self.datasets.keys():
            _validate_geo_accession(key)
        return self

# ----------------------------
# Benchmarking Subgraph States
# ----------------------------

class BenchmarkingIngestionConfig(BaseModel):
    output_root: NonEmptyStr
    accessions: List[NonEmptyStr]
    clock_list: List[MethylationClocks]
    artifacts: List[ArtifactRef] = Field(default_factory=list)

    @field_validator("accessions", mode="before")
    @classmethod
    def validate_accessions(cls, v):
        return _validate_accessions_list(v)
    
    @field_validator("output_root", mode="before")
    @classmethod
    def validate_output_root(cls, v):
        return _validate_output_root(v)

class BenchmarkingDatasetState(BaseModel):
    accession: NonEmptyStr
    output_dir: NonEmptyStr
    status: Literal["not_started", "in_progress", "failed", "completed"] = Field(default="not_started")
    steps: Dict[NonEmptyStr, StepStatus] = Field(default_factory=lambda: {
        "retrieve_clocks": StepStatus(
            status="completed"),
        "make_predictions": StepStatus(),
        "make_computations": StepStatus()
    })
    
    benchmarking_result: Optional[Dict[MethylationClocks, Any]] = None

    artifacts: List[ArtifactRef] = Field(default_factory=list)
    
    @field_validator("accession", mode="before")
    @classmethod
    def validate_accessions(cls, v):
        return _validate_geo_accession(v)

class BenchmarkingSubgraphState(BaseModel):
    run_id: NonEmptyStr
    subgraph: Literal["benchmarking"] = "benchmarking"
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    main_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    config: BenchmarkingIngestionConfig
    datasets: Annotated[Dict[NonEmptyStr, BenchmarkingDatasetState], merge_dict] = Field(default_factory=dict)

    warnings: List[NonEmptyStr] = Field(default_factory=list)
    errors: List[NonEmptyStr] = Field(default_factory=list)

    # UX/routing
    next_action_hint: Optional[NonEmptyStr] = None

# ----------------------------
# Main State
# ----------------------------

class SubgraphHandle(BaseModel):
    name: Literal["geo_retrieval", "harmonization", "quality_control", "benchmarking"]
    status: Literal["not_started", "running", "paused_for_review", "completed", "failed"] = "not_started"
    # thread_id is the LangGraph checkpointing key you use to resume
    thread_id: NonEmptyStr
    warnings: List[NonEmptyStr] = Field(default_factory=list)
    errors: List[NonEmptyStr] = Field(default_factory=list)

class MainState(BaseModel):
    run_id: NonEmptyStr
    user_request: Optional[NonEmptyStr] = None
    default_output_root: Optional[NonEmptyStr] = None
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    routing_history: Annotated[List[RouterOutput], add] = Field(default_factory=list)
    needs_clarification: bool = False

    # registry of subgraph runs
    subgraphs: Dict[NonEmptyStr, SubgraphHandle] = Field(default_factory=dict)
    datasets: Annotated[Dict[NonEmptyStr, Dict[NonEmptyStr, StepStatus]], merge_dict] = Field(default_factory=dict) # track dataset-level status across subgraphs for UX
    # Track each step across subgraphs
    # optional: unify HITL tickets across subgraphs for UX
    pending_reviews: Optional[HumanReviewRequest] = None

    artifacts: List[ArtifactRef] = Field(default_factory=list)
    warnings: List[NonEmptyStr] = Field(default_factory=list)
    errors: List[NonEmptyStr] = Field(default_factory=list)

    next_action_hint: Optional[NonEmptyStr] = None
