__all__ = []

from typing import Literal, Union

from pydantic import BaseModel, Field

from ..utils.helper import NonEmptyStr

# ----------------------------
# Data-Type Conversion
# ----------------------------


class PreprocessClippingInput(BaseModel):
    """
    Represents the input for clipping during preprocessing.

    Attributes:
        - lower_bound: The lower bound for clipping (inclusive).
        - upper_bound: The upper bound for clipping (inclusive).
    """

    lower_bound: float | None = Field(ge=0.0, le=1.0, description="Lower bound for clipping (inclusive)")
    upper_bound: float | None = Field(ge=0.0, le=1.0, description="Upper bound for clipping (inclusive)")


class PreprocessDataInput(BaseModel):
    """
    Represents the input for data preprocessing.

    Attributes:
        - from_type: The current data type of the input data.
        - to_type: The desired data type to convert to.
        - clipping: Choice to apply clipping before conversion.
    """

    from_type: Literal["beta", "m"] | None = Field(None, description="The current data type of the input data")
    to_type: Literal["beta", "m"] = Field(..., description="The desired data type to convert to")
    clipping: PreprocessClippingInput | None = Field(None, description="Choice to apply clipping before conversion")


class PreprocessDataResult(BaseModel):
    """
    Represents the result of data preprocessing.

    Attributes:
        - data_type: The data type of the processed data.
    """

    data_type: Literal["beta", "m"]


# ----------------------------
# QC
# ----------------------------


class SampleLevelQCInput(BaseModel):
    """
    Represents the input for sample-level quality control (QC).

    Attributes:
        - missing_cutoff: The maximum allowed fraction of missing values per sample.
    """

    missing_cutoff: float = Field(
        0.1, ge=0.0, le=1.0, description="Maximum allowed fraction of missing values per sample"
    )


class SampleLevelQCResult(BaseModel):
    """
    Represents the result of sample-level quality control (QC).

    Attributes:
        - removed_samples: List of sample IDs removed during QC.
    """

    removed_samples: list[NonEmptyStr] = Field(default_factory=list, description="List of sample IDs removed during QC")


class SimpleImputerModelInput(BaseModel):
    """
    Represents the input for a simple imputer model.

    Attributes:
        - concept: The concept of the imputer model, in this case "simple".
        - strategy: The imputation strategy to use.
    """

    concept: Literal["simple"] = "simple"
    strategy: Literal["mean", "median", "most_frequent", "constant"] = Field(
        "mean", description="Imputation strategy to use"
    )


class KNNImputerModelInput(BaseModel):
    """
    Represents the input for a KNN imputer model.

    Attributes:
        - concept: The concept of the imputer model, in this case "knn".
        - n_neighbors: The number of neighbors to use for KNN imputation.
        - weights: The weighting strategy for KNN imputation.
    """

    concept: Literal["knn"] = "knn"
    n_neighbors: int = Field(5, ge=1, description="Number of neighbors to use for KNN imputation")
    weights: Literal["uniform", "distance"] = Field("uniform", description="Weighting strategy for KNN imputation")


ImputerModelInput = Union[
    SimpleImputerModelInput,
    KNNImputerModelInput,
]


class ImputationInput(BaseModel):
    """
    Represents the input for an imputation strategy.

    Attributes:
        - strategy: The imputation strategy to use.
        - stratify_by: The column name to stratify by if strategy is 'stratify'.
        - imputation_model: The imputer model to use for handling missing values.
    """

    strategy: Literal["stratify", "whole"] = Field("whole", description="Imputation strategy to use")
    stratify_by: NonEmptyStr | None = Field(None, description="Column name to stratify by if strategy is 'stratify'")
    imputation_model: ImputerModelInput = Field(
        default_factory=lambda: SimpleImputerModelInput(),
        description="Imputer model to use for handling missing values",
    )


class CpGLevelQCInput(BaseModel):
    """
    Represents the input for CpG-level quality control (QC).

    Attributes:
        - missing_cutoff: The maximum allowed fraction of missing values per CpG site.
        - imputation_strategy: The imputation strategy for handling missing CpG values.
    """

    missing_cutoff: float = Field(
        0.2, ge=0.0, le=1.0, description="Maximum allowed fraction of missing values per CpG site"
    )
    imputation_strategy: ImputationInput = Field(
        default_factory=lambda: ImputationInput(strategy="whole"),
        description="Imputation strategy for handling missing CpG values",
    )


class CpGLevelQCResult(BaseModel):
    """
    Represents the result of CpG-level quality control (QC).

    Attributes:
        - removed_cpgs: List of CpG IDs removed during QC.
        - missing_before_imputation: Fraction of missing values before imputation.
    """

    removed_cpgs: list[NonEmptyStr] = Field(default_factory=list, description="List of CpG IDs removed during QC")
    missing_before_imputation: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction of missing values before imputation"
    )


class DNAmQCInput(BaseModel):
    """
    Represents the input for DNA methylation (DNAm) quality control (QC).

    Attributes:
        - dnam_cutoff: The minimum allowed DNAm value per sample.
    """

    dnam_cutoff: float = Field(0.96, ge=0.0, le=1.0, description="Minimum allowed DNAm value per sample")


class DNAmQCResult(BaseModel):
    """
    Represents the result of DNA methylation (DNAm) quality control (QC).

    Attributes:
        - removed_samples: List of sample IDs removed during DNAm QC.
    """

    removed_samples: list[NonEmptyStr] = Field(
        default_factory=list, description="List of sample IDs removed during DNAm QC"
    )


class InterarrayCorrelationQCInput(BaseModel):
    """
    Represents the input for inter-array correlation quality control (QC).

    Attributes:
        - correlation_cutoff: The minimum allowed mean inter-array correlation per sample.
    """

    correlation_cutoff: float = Field(
        0.9, ge=0.0, le=1.0, description="Minimum allowed mean inter-array correlation per sample"
    )


class InterarrayCorrelationQCResult(BaseModel):
    """
    Represents the result of inter-array correlation quality control (QC).

    Attributes:
        - removed_samples: List of sample IDs removed during inter-array correlation QC.
    """

    removed_samples: list[NonEmptyStr] = Field(
        default_factory=list, description="List of sample IDs removed during inter-array correlation QC"
    )
