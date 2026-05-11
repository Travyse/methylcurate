import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..utils.helper import NonEmptyStr

Kind = Literal[
    "soft_file",
    "metadata_cache",
    "metadata_extraction_protocol",
    "preqc_methylation_data",
    "postqc_methylation_data",
    "dataset_metadata",
    "harmonization_json",
    "clock_predictions_csv",
    "visualization",
    "supplementary_file_methylation_data",
    "supplementary_file_methylation_data_column_scheme",
    "supplementary_file_methylation_data_formatted",
    "subject_column_mapping",
    "benchmark_summary",
    "dataset_benchmark",
    "clock",
    "disease_harmonization_mapping",
    "tissue_harmonization_mapping",
    "sex_harmonization_mapping",
    "cell_type_harmonization_mapping",
    "disease_label_guessing",
    "disease_label_harmonization",
    "tissue_label_guessing",
    "tissue_label_harmonization",
    "sex_label_harmonization",
    "cell_type_label_guessing",
    "cell_type_label_harmonization",
]


class ArtifactRef(BaseModel):
    """
    Model representing a reference to an artifact, including its path, kind, and optional metadata.

    Attributes:
        path (NonEmptyStr): The file path of the artifact.
        kind (Kind): The kind of artifact, e.g., "soft_family", "sample_metadata_csv", "beta_matrix_csv".
        accession_code (Optional[NonEmptyStr]): The accession code of the artifact, if applicable.
        sha256 (Optional[NonEmptyStr]): The SHA-256 hash of the artifact, if available.
        bytes (Optional[int]): The size of the artifact in bytes, if known.
        created_at (Optional[NonEmptyStr]): The creation timestamp of the artifact in ISO8601 format.
    """

    path: NonEmptyStr
    kind: Kind  # e.g., "soft_family", "sample_metadata_csv", "beta_matrix_csv"
    accession_code: NonEmptyStr | None = None
    sha256: NonEmptyStr | None = None
    bytes: int | None = None
    created_at: NonEmptyStr | None = None  # ISO8601

    @field_validator("path", mode="before")
    def validate_path(cls, v):
        if not v or not isinstance(v, str) or not os.path.exists(v):
            raise ValueError("path must be a non-empty string and must exist")
        return v

    @field_validator("kind", mode="before")
    def validate_kind(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("kind must be a non-empty string")
        return v


class StepStatus(BaseModel):
    """
    Model representing the status of a step in a workflow, including its current status, timestamps, and any errors or warnings.

    Attributes:
        status (Literal["not_started", "running", "completed", "failed", "paused_for_review", "canceled"]): The current status of the step.
        started_at (Optional[NonEmptyStr]): The timestamp when the step started, in ISO8601 format.
        finished_at (Optional[NonEmptyStr]): The timestamp when the step finished, in ISO8601 format.
        error (Optional[NonEmptyStr]): Any error message associated with the step, if applicable.
        warnings (List[NonEmptyStr]): A list of warning messages associated with the step, if any.
    """

    status: Literal["not_started", "running", "completed", "failed", "paused_for_review", "canceled"] = "not_started"
    started_at: NonEmptyStr | None = None
    finished_at: NonEmptyStr | None = None
    error: NonEmptyStr | None = None
    warnings: list[NonEmptyStr] = Field(default_factory=list)


class HumanReviewRequest(BaseModel):
    """
    Model representing a request for human review, including the review ID, reason for review,
    question to be answered, any relevant payload, and the creation timestamp.

    Attributes:
        review_id (NonEmptyStr): A unique identifier for the review request.
        reason (NonEmptyStr): The reason for requesting human review.
        question (NonEmptyStr): The specific question that the human reviewer is being asked to address.
        payload (Dict): A dictionary containing any relevant facts or exemplars that may assist the human reviewer in making their decision.
        created_at (NonEmptyStr): The timestamp when the review request was created, in ISO8601 format.
    """

    review_id: NonEmptyStr
    reason: NonEmptyStr
    question: NonEmptyStr
    payload: dict = Field(default_factory=dict)  # small, user-facing facts + exemplars
    created_at: NonEmptyStr


class HumanReviewDecision(BaseModel):
    """
    Model representing a decision made by a human reviewer, including the review ID, decision, any notes or edits, and the timestamp of the decision.

    Attributes:
        review_id (NonEmptyStr): A unique identifier for the review request.
        decision (Literal["approve","reject","edit"]): The decision made by the human reviewer.
            "edit" indicates that the user supplied corrected parameters or rules.
        notes (Optional[NonEmptyStr]): Any additional notes provided by the human reviewer.
        edits (Dict): A dictionary containing any edits made by the human reviewer, e.g., {"subject_id_rule": {...}}.
        decided_at (NonEmptyStr): The timestamp when the decision was made, in ISO8601 format.
    """

    review_id: NonEmptyStr
    decision: Literal["approve", "reject", "edit"]  # edit = user supplies corrected params/rules
    notes: NonEmptyStr | None = None
    edits: dict = Field(default_factory=dict)  # e.g. {"subject_id_rule": {...}}
    decided_at: NonEmptyStr
