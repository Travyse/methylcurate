__all__ = []

import os
from operator import add
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator

from ...contracts.clocks import MethylationClocks
from ...contracts.common import ArtifactRef, HumanReviewDecision, HumanReviewRequest, StepStatus
from ...contracts.geo import Concept as GEOConcepts
from ...contracts.geo import (
    GEODownloadResult,
    GEOMetadataExtractionInput,
    GEOMetadataExtractionResult,
    MetadataSummary,
)
from ...contracts.harmonize import HumanReadableConceptInput, LabelMappingSet
from ...contracts.qc import (
    CpGLevelQCInput,
    CpGLevelQCResult,
    DNAmQCInput,
    DNAmQCResult,
    InterarrayCorrelationQCInput,
    InterarrayCorrelationQCResult,
    PreprocessDataInput,
    PreprocessDataResult,
    SampleLevelQCInput,
    SampleLevelQCResult,
)
from ...contracts.router import RouterOutput
from ...utils.helper import NonEmptyStr


def merge_dict(left: dict, right: dict) -> dict:
    result = dict(left)
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


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
    accessions: list[NonEmptyStr]
    artifacts: list[ArtifactRef] = Field(default_factory=list)

    @field_validator("accessions", mode="before")
    @classmethod
    def validate_accessions(cls, v):
        return _validate_accessions_list(v)


class GPLMetadata(BaseModel):
    platform_id: NonEmptyStr | None = None
    title: NonEmptyStr | None = None


class RefinementTracking(BaseModel):
    num_retries: int = 0
    formatting_history: list[list[str]] | None = Field(default_factory=list)  # Flagged concepts
    parsing_history: list[list[str]] | None = Field(default_factory=list)  # Flagged concepts
    example_errors: list[dict[str, Any]] | None = Field(default_factory=list)  # Store examples of errors for review


class GeoDatasetState(BaseModel):
    accession: NonEmptyStr
    output_dir: NonEmptyStr

    steps: dict[str, StepStatus] = Field(
        default_factory=lambda: {
            "download_soft": StepStatus(),
            "check_valid_dataset": StepStatus(),
            "extract_metadata_schema": StepStatus(),
            "extract_data": StepStatus(),
            "refine_metadata_schema": StepStatus(),
            "supplementary_file_check": StepStatus(),
        }
    )

    status: Literal["not_started", "in_progress", "failed", "completed"] = Field(default="not_started")
    is_hidden: bool = Field(default=False)
    is_valid_dataset: bool = Field(default=True)

    download_result: GEODownloadResult | None = None

    metadata_extraction_input: GEOMetadataExtractionInput | None = None
    metadata_extraction_result: GEOMetadataExtractionResult | None = None

    metadata_summary: MetadataSummary | None = None
    refinement_history: RefinementTracking = Field(default_factory=RefinementTracking)

    artifacts: list[ArtifactRef] = Field(default_factory=list)
    platform_metadata: GPLMetadata | None = None
    supplementary_files: list[str] | None = Field(default_factory=list)
    supplementary_data: Annotated[dict[NonEmptyStr, Literal["pending", "running", "completed", "failed"]], merge_dict] | None = None

    pending_review: list[HumanReviewRequest] | None = None
    review_history: list[HumanReviewDecision] = Field(default_factory=list)

    warnings: list[NonEmptyStr] = Field(default_factory=list)
    errors: list[NonEmptyStr] = Field(default_factory=list)

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
    concept_messages: dict[GEOConcepts, list[AnyMessage]] = Field(default_factory=dict)

    config: GEOIngestionConfig

    # per-accession states
    datasets: Annotated[dict[NonEmptyStr, GeoDatasetState], merge_dict]

    # optional global artifacts/logs
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    errors: list[NonEmptyStr] = Field(default_factory=list)

    # UX/routing
    next_action_hint: NonEmptyStr | None = None

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
    accessions: list[NonEmptyStr]
    artifacts: list[ArtifactRef] = Field(default_factory=list)

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
    steps: dict[NonEmptyStr, StepStatus] = Field(
        default_factory=lambda: {
            "map_disease_labels_to_ontology": StepStatus(),
            "group_disease_labels": StepStatus(),
            "map_tissue_labels_to_ontology": StepStatus(),
            "group_tissue_labels": StepStatus(),
            "map_cell_type_labels_to_ontology": StepStatus(),
            "harmonize_sex_labels": StepStatus(),
        }
    )

    harmonization_input: HumanReadableConceptInput | None = None

    # Disease Harmonization
    disease_label_guessing: LabelMappingSet | None = None
    disease_label_mapping: LabelMappingSet | None = None

    # Tissue Harmonization
    tissue_label_guessing: LabelMappingSet | None = None
    tissue_label_mapping: LabelMappingSet | None = None

    # Cell Type Harmonization
    cell_type_label_guessing: LabelMappingSet | None = None
    cell_type_label_mapping: LabelMappingSet | None = None

    sex_mapping: LabelMappingSet | None = None


class HarmonizationSubgraphState(BaseModel):
    run_id: NonEmptyStr
    subgraph: Literal["harmonization"] = "harmonization"
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    main_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    config: HarmonizationIngestionConfig

    # per-concept states
    datasets: Annotated[dict[NonEmptyStr, HarmonizationDatasetState], merge_dict] = Field(default_factory=dict)
    disease_group_mapping: LabelMappingSet | None = None
    tissue_group_mapping: LabelMappingSet | None = None

    # optional global artifacts/logs
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    errors: list[NonEmptyStr] = Field(default_factory=list)

    # UX/routing
    next_action_hint: NonEmptyStr | None = None


# ----------------------------
# Quality Control Subgraph States
# ----------------------------


class QualityControlIngestionConfig(BaseModel):
    output_root: NonEmptyStr
    accessions: list[NonEmptyStr]
    artifacts: list[ArtifactRef] = Field(default_factory=list)

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
    steps: dict[NonEmptyStr, StepStatus] = Field(default_factory=lambda: {"quality_control": StepStatus()})

    data_conversion_result: PreprocessDataResult | None = None
    sample_level_qc_result: SampleLevelQCResult | None = None
    cpg_level_qc_result: CpGLevelQCResult | None = None
    dnam_qc_result: DNAmQCResult | None = None
    interarray_correlation_qc_result: InterarrayCorrelationQCResult | None = None

    artifacts: list[ArtifactRef] = Field(default_factory=list)

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
    datasets: Annotated[dict[NonEmptyStr, DatasetQualityControlState], merge_dict] = Field(default_factory=dict)

    data_conversion_input: PreprocessDataInput | None = None
    sample_level_qc_input: SampleLevelQCInput | None = None
    cpg_level_qc_input: CpGLevelQCInput | None = None
    dnam_qc_input: DNAmQCInput | None = None
    interarray_correlation_qc_input: InterarrayCorrelationQCInput | None = None

    # optional global artifacts/logs
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    errors: list[NonEmptyStr] = Field(default_factory=list)

    # UX/routing
    next_action_hint: NonEmptyStr | None = None

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
    accessions: list[NonEmptyStr]
    clock_list: list[MethylationClocks]
    artifacts: list[ArtifactRef] = Field(default_factory=list)

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
    steps: dict[NonEmptyStr, StepStatus] = Field(
        default_factory=lambda: {
            "retrieve_clocks": StepStatus(status="completed"),
            "make_predictions": StepStatus(),
        }
    )

    benchmarking_result: dict[MethylationClocks, Any] | None = None

    artifacts: list[ArtifactRef] = Field(default_factory=list)

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
    datasets: Annotated[dict[NonEmptyStr, BenchmarkingDatasetState], merge_dict] = Field(default_factory=dict)

    warnings: list[NonEmptyStr] = Field(default_factory=list)
    errors: list[NonEmptyStr] = Field(default_factory=list)

    # UX/routing
    next_action_hint: NonEmptyStr | None = None


# ----------------------------
# Help Subgraph States
# ----------------------------


class HelpIngestionConfig(BaseModel):
    output_root: str = ""
    accessions: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class HelpSubgraphState(BaseModel):
    run_id: NonEmptyStr
    subgraph: Literal["help"] = "help"
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    main_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    config: HelpIngestionConfig = Field(default_factory=HelpIngestionConfig)

    warnings: list[NonEmptyStr] = Field(default_factory=list)
    errors: list[NonEmptyStr] = Field(default_factory=list)

    next_action_hint: NonEmptyStr | None = None


# ----------------------------
# Main State
# ----------------------------


class SubgraphHandle(BaseModel):
    name: Literal["geo_retrieval", "harmonization", "quality_control", "benchmarking", "help"]
    status: Literal["not_started", "running", "paused_for_review", "completed", "failed"] = "not_started"
    # thread_id is the LangGraph checkpointing key you use to resume
    thread_id: NonEmptyStr
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    errors: list[NonEmptyStr] = Field(default_factory=list)


class MainState(BaseModel):
    run_id: NonEmptyStr
    user_request: NonEmptyStr | None = None
    default_output_root: NonEmptyStr | None = None
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    routing_history: Annotated[list[RouterOutput], add] = Field(default_factory=list)
    needs_clarification: bool = False

    # registry of subgraph runs
    subgraphs: dict[NonEmptyStr, SubgraphHandle] = Field(default_factory=dict)
    datasets: Annotated[dict[NonEmptyStr, dict[NonEmptyStr, StepStatus]], merge_dict] = Field(
        default_factory=dict
    )  # track dataset-level status across subgraphs for UX
    # Track each step across subgraphs
    # optional: unify HITL tickets across subgraphs for UX
    pending_reviews: HumanReviewRequest | None = None

    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[NonEmptyStr] = Field(default_factory=list)
    errors: list[NonEmptyStr] = Field(default_factory=list)

    next_action_hint: NonEmptyStr | None = None
