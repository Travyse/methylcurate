from typing import Annotated, Any, Literal, Optional, get_args

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from ..utils.helper import NonEmptyStr
from .common import ArtifactRef

# ----------------------------
# GEO Download Contracts
# ----------------------------


class GEODownloadInput(BaseModel):
    """
    Model representing the input for a GEO download, including the accession code.

    Attributes:
        accession (NonEmptyStr): The GSE accession code, e.g., GSE40279.

    Validation:
        - The accession code must start with "GSE" (case-insensitive).
    """

    accession: NonEmptyStr = Field(..., description="GSE accession, e.g. GSE40279")

    @model_validator(mode="after")
    def validate_accession(self):
        """
        Validates that the accession code starts with "GSE". This is a requirement for GEO Series accessions.
        """
        if not self.accession.upper().startswith("GSE"):
            raise ValueError("accession must start with 'GSE'")
        return self


class GEODownloadBatchInput(BaseModel):
    """
    Model representing a batch input for GEO downloads, including a list of individual GEO download inputs.

    Attributes:
        geo_downloads (List[GEODownloadInput]): A list of GEO download inputs.
    """

    geo_downloads: list[GEODownloadInput] = Field(..., description="List of GEO download inputs")


class GEODownloadResult(BaseModel):
    """
    Model representing the result of a GEO download, including the accession code, artifact reference, status, and any errors or warnings.

    Attributes:
        accession (NonEmptyStr): The GSE accession code, e.g., GSE40279.
        artifact (Optional[ArtifactRef]): Reference to the downloaded artifact, if download was successful.
        status (Literal["success", "skipped", "failed", "resolved"]): Overall status of the download attempt.
        error (Optional[NonEmptyStr]): Error message if status is failed.
        warnings (List[NonEmptyStr]): List of warning messages, if any.

    Validation:
        - If status is "failed", an error message must be provided.
        - If status is not "failed", an error message must not be provided.
        - If status is "success", an artifact reference should ideally be provided (this check can be relaxed if artifact is truly optional on success).
    """

    accession: NonEmptyStr = Field(..., description="GSE accession, e.g. GSE40279")
    artifact: ArtifactRef | None = Field(
        ..., description="Reference to the downloaded artifact, if download was successful"
    )
    status: Literal["success", "skipped", "failed", "resolved"] = Field(
        ..., description="Overall status of the download attempt"
    )
    error: NonEmptyStr | None = Field(..., description="Error message if status is failed")
    warnings: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self):
        """
        Validates the consistency of the GEO download result.

        Raises:
            ValueError: If the result is inconsistent, e.g., a failed result without an error message.
        """
        if self.status == "failed" and not self.error:
            raise ValueError("failed results must include error")
        if self.status != "failed" and self.error:
            raise ValueError("only failed results may include error")
        if self.status == "success" and self.artifact is None:
            # if artifact is truly optional on success, drop this check
            raise ValueError("success results should include artifact")
        return self


class GEODownloadBatchResult(BaseModel):
    """
    Model representing the result of a batch GEO download, including individual download results, overall batch status, and any warnings.

    Attributes:
        results (List[GEODownloadResult]): List of individual GEO download results.
        batch_status (Literal["success", "partial", "failed"]): Overall status of the batch download.
        warnings (List[NonEmptyStr]): List of warning messages, if any.
    """

    results: list[GEODownloadResult]
    batch_status: Literal["success", "partial", "failed"]
    warnings: list[NonEmptyStr] = Field(default_factory=list)


# ----------------------------
# Metadata Column Extraction Contracts
# ----------------------------

Concept = Literal["subject_id", "age", "tissue", "cell_type", "condition", "sex", "disease_status", "platform"]
ResolutionStatus = Literal["resolved", "needs_review", "missing", "error"]


class GEOMetadataExtractionInput(BaseModel):
    """
    Model representing the input for GEO metadata extraction, including references to the GEO SOFT file artifact and example metadata fields.

    Attributes:
        artifact (ArtifactRef): Reference to the GEO SOFT file artifact to extract metadata from.
        title (List[List[NonEmptyStr]]): List of example titles from GEO metadata.
        source_name_ch1 (List[List[NonEmptyStr]]): List of example source names from GEO metadata.
        description (List[List[NonEmptyStr]]): List of example descriptions from GEO metadata.
        characteristics_ch1 (List[Dict[NonEmptyStr, Any]]): List of example characteristics from GEO metadata.
        relation (Optional[List[List[NonEmptyStr]]]): List of example relations from GEO metadata.
        platform_id (Optional[List[List[NonEmptyStr]]]): List of example platform IDs from GEO metadata.

    Validation:
        - The artifact's accession code must start with "GSE" (case-insensitive).
    """

    artifact: ArtifactRef = Field(..., description="Reference to the GEO SOFT file artifact to extract metadata from")
    title: list[list[NonEmptyStr]] = Field(..., description="List of example titles from GEO metadata")
    source_name_ch1: list[list[NonEmptyStr]] = Field(..., description="List of example source names from GEO metadata")
    description: list[list[NonEmptyStr]] = Field(..., description="List of example descriptions from GEO metadata")
    characteristics_ch1: list[dict[NonEmptyStr, Any]] = Field(
        ..., description="List of example characteristics from GEO metadata"
    )
    relation: list[list[NonEmptyStr]] | None = Field(None, description="List of example relations from GEO metadata")
    platform_id: list[list[NonEmptyStr]] | None = Field(
        None, description="List of example platform IDs from GEO metadata"
    )

    @model_validator(mode="after")
    def validate_artifact(self):
        if not self.artifact.accession_code.upper().startswith("GSE"):
            raise ValueError("accession_code must start with 'GSE'")
        return self


class GEOMetadataExtractionBatchInput(BaseModel):
    """
    Model representing a batch input for GEO metadata extraction, including multiple individual dataset inputs.

    Attributes:
        datasets (List[GEOMetadataExtractionInput]): List of sample metadata inputs.
    """

    datasets: list[GEOMetadataExtractionInput] = Field(..., description="List of sample metadata inputs")


class ExtractionRuleBase(BaseModel):
    """
    Base model for extraction rules, defining common attributes and validation logic.

    Attributes:
        type (Literal["regex"]): The type of extraction rule.
        field_name (Literal["title", "source_name_ch1", "description", "characteristics_ch1", "relation", "platform_id"]): The name of the field to apply the extraction rule to.
        group_index (int): The index of the regex capture group to return.
        normalization (List[Literal["strip", "lower", "digits_only"]]): The normalization pipeline to apply to the extracted value.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["regex"] = "regex"
    field_name: Literal["title", "source_name_ch1", "description", "characteristics_ch1", "relation", "platform_id"]
    group_index: int = Field(
        ...,
        le=1,
        description=(
            "Regex capture group to return. 0 returns the full match; 1 returns the first capturing group, etc. "
            "Must be between 0 and the number of capturing groups in the pattern (inclusive). "
            "If omitted, defaults to 0."
        ),
    )
    normalization: list[Literal["strip", "lower", "digits_only"]] = Field(
        default_factory=lambda: ["strip"], description="Normalization pipeline"
    )


class DefaultValue(BaseModel):
    """
    Model representing a default value extraction rule.

    Attributes:
        field_name (Literal["default"]): The name of the field to apply the extraction rule to.
        value (NonEmptyStr): The default value to use for this concept when no extractable evidence is found in the dataset but there is evidence in the dataset metadata like title or summary. This should be a single, specific value.
    """

    field_name: Literal["default"] = "default"
    value: NonEmptyStr = Field(
        ...,
        description="The default value to use for this concept when no extractable evidence is found in the dataset but there is evidence in the dataset metadata like title or summary. This should be a single, specific value",
    )


class CharacteristicsExtractionRule(ExtractionRuleBase):
    """
    Model representing an extraction rule for characteristics_ch1 field.

    Attributes:
        field_name (Literal["characteristics_ch1"]): The name of the field to apply the extraction rule to.
        pattern (NonEmptyStr): A regular expression pattern to apply to the specified key_name value in characteristics_ch1.
        key_name (NonEmptyStr): The key inside the target key:value string in characteristics_ch1.
        control_value (Optional[NonEmptyStr]): This only set for extraction rules for disease_status. This identifies how the dataset encodes the control samples, e.g. 'healthy' or 'control'.
    """

    field_name: Literal["characteristics_ch1"]
    pattern: NonEmptyStr = Field(
        ...,
        description="A regular expression pattern to apply to the specified key_name value in characteristics_ch1. For example, if key_name is 'age' and characteristics_ch1 includes the following entry: {'age': '45 years'}, then this pattern would be applied to '45 years' and should extract the concept value of '45'.",
    )
    key_name: NonEmptyStr = Field(
        ...,
        description="The key inside the target key:value string in characteristics_ch1, e.g. 'age' in 'age: 45 years'",
    )
    control_value: NonEmptyStr | None = Field(
        None,
        description="This only set for extraction rules for disease_status. This identifies how the dataset encodes the control samples, e.g. 'healthy' or 'control'.",
    )


class OtherExtractionRule(ExtractionRuleBase):
    """
    Model representing an extraction rule for fields other than characteristics_ch1.

    Attributes:
        field_name (Literal["title", "source_name_ch1", "description", "relation", "platform_id"]): The name of the field to apply the extraction rule to.
        pattern (NonEmptyStr): A regular expression pattern to apply to the specified field. For characteristics_ch1 extractions, this pattern will be applied to each key:value pair in the list. This regex pattern should extract the concept value.
    """

    field_name: Literal["title", "source_name_ch1", "description", "relation", "platform_id"]
    pattern: NonEmptyStr = Field(
        ...,
        description="A regular expression pattern to apply to the specified field. For characteristics_ch1 extractions, this pattern will be applied to each key:value pair in the list. This regex pattern should extract the concept value.",
    )


ExtractionRule = Annotated[
    CharacteristicsExtractionRule | OtherExtractionRule | DefaultValue,
    Field(discriminator="field_name"),
]


class FieldResolutionBase(BaseModel):
    """
    Base model for field resolution.

    Attributes:
        status (ResolutionStatus): The status of the resolution.
        confidence (float): Model confidence in this extraction.
        notes (List[NonEmptyStr]): Evidence for the extraction rule and any notes about the resolution.
    """

    model_config = ConfigDict(extra="forbid")  # critical: prevents “rest of attrs should not be present”
    status: ResolutionStatus
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in this extraction")
    notes: list[NonEmptyStr] = Field(
        min_length=1, description="Evidence for the extraction rule and any notes about the resolution"
    )


class MissingResolution(FieldResolutionBase):
    """
    Model representing a missing resolution for a field.

    Attributes:
        status (Literal["missing"]): The status of the resolution.
        candidate_field_names (List[Literal["title","source_name_ch1","description","characteristics_ch1","relation","platform_id"]]): Fields checked and found no extractable evidence for this concept.
        candidate_key_names (List[str]): Key names in characteristics_ch1 checked and found no extractable evidence for this concept. Must include every possible key name.
        absence_evidence (List[NonEmptyStr]): Must be verbatim substrings copied from the input values, not analyst commentary. Never write 'no field contains…'
    """

    status: Literal["missing"]
    candidate_field_names: list[
        Literal["title", "source_name_ch1", "description", "characteristics_ch1", "relation", "platform_id"]
    ] = Field(..., min_length=6, description="Fields checked and found no extractable evidence for this concept.")
    candidate_key_names: list[str] = Field(
        ...,
        min_length=1,
        description="Key names in characteristics_ch1 checked and found no extractable evidence for this concept. Must include every possible key name.",
    )
    absence_evidence: list[NonEmptyStr] = Field(
        ...,
        min_length=1,
        description="Must be verbatim substrings copied from the input values, not analyst commentary. Never write 'no field contains…'",
    )


class ErrorResolution(FieldResolutionBase):
    """
    Model representing an error resolution for a field.

    Attributes:
        status (Literal["error"]): The status of the resolution.
        error (Optional[NonEmptyStr]): Error message if status is 'error'.
    """

    status: Literal["error"]
    error: NonEmptyStr | None = Field(..., description="Error message if status is 'error'")


class ResolvedResolution(FieldResolutionBase):
    """
    Model representing a resolved resolution for a field.

    Attributes:
        status (Literal["resolved"]): The status of the resolution.
        extraction (ExtractionRule): The rule used to extract this concept.
        units (Optional[NonEmptyStr]): Units if applicable, e.g. 'years' for age.
    """

    model_config = ConfigDict(extra="forbid")
    status: Literal["resolved"]
    extraction: ExtractionRule = Field(..., description="The rule used to extract this concept")
    units: NonEmptyStr | None = Field(None, description="Units if applicable, e.g. 'years' for age")


FieldResolution = Annotated[
    ResolvedResolution | MissingResolution | ErrorResolution,
    Field(discriminator="status"),
]


class FieldResolutionEnvelope(BaseModel):
    """
    Model representing a field resolution envelope.

    Attributes:
        resolution (FieldResolution): The resolution details for a field.
    """

    model_config = ConfigDict(extra="forbid")
    resolution: FieldResolution


class GEOMetadataExtractionResult(BaseModel):
    """
    Model representing the result of a GEO metadata extraction.

    Attributes:
        artifact (Optional[ArtifactRef]): Reference to the saved json file containing the metadata schema.
        subject_id (FieldResolution): Resolution details for subject_id concept.
        age (FieldResolution): Resolution details for age concept.
        tissue (FieldResolution): Resolution details for tissue concept.
        cell_type (FieldResolution): Resolution details for cell_type concept.
        sex (FieldResolution): Resolution details for sex concept.
        condition (FieldResolution): Resolution details for condition concept.
        disease_status (FieldResolution): Resolution details for disease_status concept.
        platform (FieldResolution): Resolution details for platform concept.
        error (Optional[NonEmptyStr]): Error message if execution_status is 'failed'.
    """

    model_config = ConfigDict(extra="forbid")
    artifact: ArtifactRef | None = Field(
        ..., description="Reference to the saved json file containing the metadata schema"
    )
    subject_id: FieldResolution = Field(..., description="Resolution details for subject_id concept")
    age: FieldResolution = Field(..., description="Resolution details for age concept")
    tissue: FieldResolution = Field(..., description="Resolution details for tissue concept")
    cell_type: FieldResolution = Field(..., description="Resolution details for cell_type concept")
    sex: FieldResolution = Field(..., description="Resolution details for sex concept")
    condition: FieldResolution = Field(..., description="Resolution details for condition concept")
    disease_status: FieldResolution = Field(..., description="Resolution details for disease_status concept")
    platform: FieldResolution = Field(..., description="Resolution details for platform concept")
    error: NonEmptyStr | None = Field(..., description="Error message if execution_status is 'failed'")


def build_dynamic_result_model(allowed_keys: tuple[str, ...]):
    """
    Returns a GEOMetadataExtractionResult-like model whose schema constrains:
    CharacteristicsExtractionRule.key_name ∈ allowed_keys

    Args:
        allowed_keys (Tuple[str, ...]): A tuple of allowed key names for characteristics_ch1 extractions. This should include every possible key name that could be used in the input data for any of the target concepts, otherwise the model will not be able to validate valid inputs that use a key name not included in this list.
    """
    allowed_keys = tuple(allowed_keys)
    KeyName = Literal[allowed_keys]

    # 1) Dynamic CharacteristicsExtractionRule
    CharacteristicsExtractionRuleDyn = create_model(
        f"CharacteristicsExtractionRule__{abs(hash(allowed_keys))}",
        __base__=ExtractionRuleBase,
        field_name=(
            Literal["characteristics_ch1"],
            Field(..., description="The name of the field in the input data that contains the concept."),
        ),
        pattern=(
            NonEmptyStr,
            Field(
                ...,
                description="A regular expression pattern to apply to the specified key_name value in characteristics_ch1. For example, if key_name is 'age' and characteristics_ch1 includes the following entry: {'age': '45 years'}, then this pattern would be applied to '45 years' and should extract the concept value of '45'.",
            ),
        ),
        key_name=(
            KeyName,
            Field(
                ...,
                description="This represents the key name of one of the key: value pairs in characteristics_ch1 that semantically encodes the concept. This is not necessarily an exact match of the concept. For example, if the concept is age and the characteristics_ch1 column includes 'age (years): 45', then key_name would be 'age'",
            ),
        ),
        control_value=(
            Optional[NonEmptyStr],
            Field(
                None,
                description="This only set for extraction rules for disease_status. This identifies how the dataset encodes the control samples, e.g. 'healthy' or 'control'.",
            ),
        ),
    )

    # 2) Dynamic ExtractionRule union
    ExtractionRuleDyn = Annotated[
        CharacteristicsExtractionRuleDyn | OtherExtractionRule | DefaultValue,
        Field(discriminator="field_name"),
    ]

    # 3) Dynamic ResolvedResolution using ExtractionRuleDyn
    ResolvedResolutionDyn = create_model(
        f"ResolvedResolution__{abs(hash(allowed_keys))}",
        __base__=FieldResolutionBase,
        __config__=ConfigDict(extra="forbid"),
        status=(Literal["resolved"], Field(..., description="The resolution status for this concept.")),
        extraction=(ExtractionRuleDyn, Field(..., description="Rule used to extract this concept")),
        units=(Optional[NonEmptyStr], Field(None, description="Units if applicable")),
    )

    MissingResolutionDyn = create_model(
        f"MissingResolution__{abs(hash(allowed_keys))}",
        __base__=FieldResolutionBase,
        __config__=ConfigDict(extra="forbid"),
        status=(Literal["missing"], Field(..., description="The resolution status for this concept.")),
        candidate_field_names=(
            list[Literal["title", "source_name_ch1", "description", "characteristics_ch1", "relation", "platform_id"]],
            Field(..., min_length=6, description="Fields checked and found no extractable evidence for this concept."),
        ),
        candidate_key_names=(
            list[KeyName],
            Field(
                ...,
                min_length=len(get_args(KeyName)),
                description="Key names in characteristics_ch1 checked and found no extractable evidence for this concept. Must include every possible key name.",
            ),
        ),
        absence_evidence=(
            list[NonEmptyStr],
            Field(
                ...,
                min_length=1,
                description="Must be verbatim substrings copied from the input values, not analyst commentary. Never write 'no field contains…'",
            ),
        ),
    )

    # 4) Dynamic FieldResolution union
    FieldResolutionDyn = Annotated[
        ResolvedResolutionDyn | MissingResolutionDyn,
        Field(discriminator="status"),
    ]

    FieldResolutionEnvelopeDyn = create_model(
        f"FieldResolutionEnvelope__{abs(hash(allowed_keys))}",
        __base__=BaseModel,
        __config__=ConfigDict(extra="forbid"),
        resolution=(FieldResolutionDyn, Field(..., description="The field resolution for this concept")),
    )

    # 5) Dynamic GEOMetadataExtractionResult using FieldResolutionDyn
    GEOMetadataExtractionResultDyn = create_model(
        f"GEOMetadataExtractionResult__{abs(hash(allowed_keys))}",
        __config__=ConfigDict(extra="forbid"),
        artifact=(Optional[ArtifactRef], Field(..., description="Reference to saved json")),
        subject_id=(FieldResolutionDyn, Field(..., description="Resolution details for subject_id concept")),
        age=(FieldResolutionDyn, Field(..., description="Resolution details for age concept")),
        tissue=(FieldResolutionDyn, Field(..., description="Resolution details for tissue concept")),
        cell_type=(FieldResolutionDyn, Field(..., description="Resolution details for cell_type concept")),
        sex=(FieldResolutionDyn, Field(..., description="Resolution details for sex concept")),
        disease_status=(FieldResolutionDyn, Field(..., description="Resolution details for disease_status concept")),
        condition=(FieldResolutionDyn, Field(..., description="Resolution details for condition concept")),
        platform=(FieldResolutionDyn, Field(..., description="Resolution details for platform concept")),
        # execution_status=(Literal["succeeded", "failed"], Field("succeeded")),
        error=(Optional[NonEmptyStr], Field(...)),
    )

    print(f"\n\nConstructed dynamic result model {CharacteristicsExtractionRuleDyn.model_json_schema()}\n\n")
    return GEOMetadataExtractionResultDyn, FieldResolutionDyn, FieldResolutionEnvelopeDyn


def build_dynamic_resolution_correction_model(allowed_concepts: tuple[str, ...], field_resolution_dyn: Any):
    """
    Build a dynamic Pydantic model for resolution correction based on allowed concepts.

    Args:
        allowed_concepts (Tuple[str, ...]): A tuple of allowed concept names for resolution correction.
        field_resolution_dyn (Any): The dynamic field resolution model to use for each concept.

    Returns:
        Type[BaseModel]: A Pydantic model class for resolution correction with fields for each allowed concept.
    """
    allowed_concepts = tuple(allowed_concepts)
    model_fields = {
        c: (field_resolution_dyn, Field(..., description=f"Resolution details for {c} concept"))
        for c in allowed_concepts
    }
    ResolutionCorrectionDyn = create_model(
        f"ResolutionCorrection__{abs(hash(allowed_concepts))}",
        __base__=BaseModel,
        __config__=ConfigDict(extra="forbid"),
        **model_fields,
    )
    return ResolutionCorrectionDyn


def build_dynamic_control_identification_model(allowed_values: tuple[str, ...]):
    """
    Build a dynamic Pydantic model for control identification based on allowed values.

    Args:
        allowed_values (Tuple[str, ...]): A tuple of allowed values for identifying control samples.

    Returns:
        Type[BaseModel]: A Pydantic model class for control identification with a field for the control value.
    """
    allowed_values = tuple(allowed_values)
    ControlIdentificationModel = create_model(
        f"ControlIdentificationModel__{abs(hash(allowed_values))}",
        __base__=BaseModel,
        __config__=ConfigDict(extra="forbid"),
        control_value=(
            Optional[Literal[allowed_values]],
            Field(
                ..., description="The value in the dataset that indicates a control sample, e.g. 'healthy' or 'control'"
            ),
        ),
    )
    return ControlIdentificationModel


def build_dynamic_constrained_resolution_model(allowed_keys: tuple[str, ...], target_concept: str):
    """
    Build a dynamic Pydantic model for constrained resolution based on allowed keys.

    Args:
        allowed_keys (Tuple[str, ...]): A tuple of allowed key names for characteristics_ch1 extractions. This should include every possible key name that could be used in the input data for any of the target concepts, otherwise the model will not be able to validate valid inputs that use a key name not included in this list.
        target_concept (str): The target concept for which the constrained resolution model is being built.

    Returns:
        Type[BaseModel]: A Pydantic model class for constrained resolution with fields for each allowed key.
    """
    allowed_keys = tuple(allowed_keys)
    KeyName = Literal[allowed_keys]
    OtherKeys = Literal["title", "source_name_ch1", "description"]

    # 1) Dynamic CharacteristicsExtractionRule
    CharacteristicsExtractionRuleDyn = create_model(
        f"CharacteristicsExtractionRule__{abs(hash(allowed_keys))}",
        __base__=BaseModel,
        field_name=(
            Literal["characteristics_ch1"],
            Field(..., description="The name of the field in the input data that contains the concept."),
        ),
        key_name=(
            KeyName,
            Field(
                ...,
                description="This represents the key name of one of the key: value pairs in characteristics_ch1 that semantically encodes the concept. This is not necessarily an exact match of the concept. For example, if the concept is age and the characteristics_ch1 column includes 'age (years): 45', then key_name would be 'age'",
            ),
        ),
    )

    OtherExtractionRuleDyn = create_model(
        f"OtherExtractionRule__{abs(hash(allowed_keys))}",
        __base__=BaseModel,
        field_name=(
            OtherKeys,
            Field(..., description="The name of the field in the input data that contains the concept."),
        ),
    )

    # 2) Dynamic ExtractionRule union
    ExtractionRuleDyn = Annotated[
        CharacteristicsExtractionRuleDyn | OtherExtractionRuleDyn | DefaultValue,
        Field(discriminator="field_name"),
    ]

    ConstrainedResolutionModel = create_model(
        f"ConstrainedResolutionModel__{abs(hash(allowed_keys))}",
        __base__=BaseModel,
        __config__=ConfigDict(extra="forbid"),
        extraction=(ExtractionRuleDyn, Field(..., description=f"The extraction rule for {target_concept}")),
    )
    return ConstrainedResolutionModel


# ----------------------------
# Sample-Level Metadata Contracts
# ----------------------------


class ResolvedColumnExtraction(BaseModel):
    """
    Represents a resolved column extraction with a regular expression pattern and evidence.

    Attributes:
    - status: A literal string indicating the resolution status, which is "resolved" in this case.
    - pattern: A non-empty string containing the regular expression pattern used to extract the target value from the column.
    - column_evidence: A list of non-empty strings representing the columns that were checked and provided evidence for the accuracy of the extraction pattern.
    - evidence: A list of non-empty strings representing additional evidence for the extraction rule and any notes about the resolution.
    """

    status: Literal["resolved"]
    pattern: NonEmptyStr = Field(..., description="A regular expression pattern to apply to extract the target value.")
    column_evidence: list[NonEmptyStr] = Field(
        min_length=1, description="Columns checked that prove the accuracy of this pattern."
    )
    evidence: list[NonEmptyStr] = Field(
        min_length=1, description="Evidence for the extraction rule and any notes about the resolution"
    )


class MissingColumnExtraction(BaseModel):
    """
    Represents a missing column extraction with candidate columns and absence evidence.

    Attributes:
    - status: A literal string indicating the resolution status, which is "missing" in this case.
    - candidate_columns: A list of non-empty strings representing the columns that were checked and found no extractable evidence for this concept.
    - absence_evidence: A list of non-empty strings representing verbatim substrings copied from the input values, not analyst commentary.
    """

    status: Literal["missing"]
    candidate_columns: list[NonEmptyStr] = Field(
        ..., min_length=1, description="Columns checked and found no extractable evidence for this concept."
    )
    absence_evidence: list[NonEmptyStr] = Field(
        ...,
        min_length=1,
        description="Must be verbatim substrings copied from the input values, not analyst commentary. Never write 'no column contains…'",
    )


class ErrorColumnExtraction(BaseModel):
    """
    Represents an error in column extraction with associated notes.

    Attributes:
    - status: A literal string indicating the resolution status, which is "error" in this case.
    - notes: A list of non-empty strings representing evidence for the extraction rule and any notes about the resolution.
    """

    status: Literal["error"]
    notes: list[NonEmptyStr] = Field(
        min_length=1, description="Evidence for the extraction rule and any notes about the resolution"
    )


ColumnExtraction = Annotated[
    ResolvedColumnExtraction | MissingColumnExtraction | ErrorColumnExtraction,
    Field(discriminator="status"),
]

ResolveOrError = Annotated[
    ResolvedColumnExtraction | ErrorColumnExtraction,
    Field(discriminator="status"),
]

MissingOrResolved = Annotated[
    ResolvedColumnExtraction | MissingColumnExtraction,
    Field(discriminator="status"),
]


class SampleDataResolution(BaseModel):
    """
    Represents the resolution of sample data extraction for beta values and detection p-values.

    Attributes:
    - beta_column: Extraction details for the beta value column.
    - detection_column: Extraction details for the detection p-value column, if applicable. This column contains p-values for each of the CpG probes.
    """

    beta_column: ResolveOrError = Field(..., description="Extraction details for the beta value column.")
    detection_column: ColumnExtraction = Field(
        ...,
        description="Extraction details for the detection p-value column, if applicable. This column contains p-values for each of the CpG probes.",
    )


class ForcedSampleDataResolution(BaseModel):
    """
    Represents the forced resolution of sample data extraction for beta values and detection p-values.

    Attributes:
    - beta_column: Extraction details for the beta value column.
    - detection_column: Extraction details for the detection p-value column, if applicable. This column contains p-values for each of the CpG probes.
    """

    beta_column: MissingOrResolved = Field(..., description="Extraction details for the beta value column.")
    detection_column: MissingOrResolved = Field(
        ...,
        description="Extraction details for the detection p-value column, if applicable. This column contains p-values for each of the CpG probes.",
    )


class LexFeat(BaseModel):
    """
    Represents lexical features extracted from a string.

    Attributes:
    - raw: The original raw string value.
    - norm: Normalized version of the string for easier comparison, e.g. lowercased, split camelCase, and stripped of special characters.
    - tokens: Individual tokens extracted from the normalized string.
    - nums: Numeric tokens extracted from the string.
    - words: Non-numeric tokens extracted from the string.
    """

    raw: str = Field(..., description="The original raw string value")
    norm: str = Field(
        ...,
        description="Normalized version of the string for easier comparison, e.g. lowercased, split camelCase, and stripped of special characters",
    )
    tokens: tuple[str, ...] = Field(..., description="Individual tokens extracted from the normalized string")
    nums: tuple[str, ...] = Field(..., description="Numeric tokens extracted from the string")
    words: tuple[str, ...] = Field(..., description="Non-numeric tokens extracted from the string")


class GEOSampleLevelMetadata(BaseModel):
    """
    Represents metadata for a single GEO sample.

    Attributes:
    - sample_name: The sample name as listed in the GEO SOFT file, e.g. GSM12345.
    - subject_id: The subject ID associated with this sample, if available.
    - age: The age of the subject in years, if available.
    - tissue: The tissue type associated with this sample, if available.
    - cell_type: The cell type associated with this sample, if available.
    - status: The disease status associated with this sample, if available.
    - sex: The sex of the subject associated with this sample, if available.
    - platform: The platform ID associated with this sample, if available.
    - gpl: The GPL IDs associated with this sample, if available.
    """

    sample_name: NonEmptyStr = Field(..., description="The sample name as listed in the GEO SOFT file, e.g. GSM12345")
    subject_id: NonEmptyStr | None = Field(
        None, description="The subject ID associated with this sample, if available"
    )
    age: float | None = Field(None, description="The age of the subject in years, if available")
    tissue: NonEmptyStr | None = Field(None, description="The tissue type associated with this sample, if available")
    cell_type: NonEmptyStr | None = Field(
        None, description="The cell type associated with this sample, if available"
    )
    status: NonEmptyStr | None = Field(
        None, description="The disease status associated with this sample, if available"
    )
    sex: NonEmptyStr | None = Field(
        None, description="The sex of the subject associated with this sample, if available"
    )
    platform: NonEmptyStr | None = Field(
        None, description="The platform ID associated with this sample, if available"
    )
    gpl: list[NonEmptyStr] | None = Field(None, description="The GPL IDs associated with this sample, if available")


class GeoSampleLevelMetadataBatch(BaseModel):
    """
    Represents a batch of sample-level metadata for a GEO Series.

    Attributes:
    - accession: The GEO Series accession, e.g. GSE12345.
    - samples: List of sample-level metadata entries for this GEO Series.
    """

    accession: NonEmptyStr = Field(..., description="The GEO Series accession, e.g. GSE12345")
    samples: list[GEOSampleLevelMetadata] = Field(
        ..., description="List of sample-level metadata entries for this GEO Series"
    )


class FieldCoverage(BaseModel):
    """
    Represents coverage details for a specific metadata field across samples.

    Attributes:
    - present: Number of samples for which this field is present.
    - missing: Number of samples for which this field is missing.
    - parse_rate: Parse rate for this field among samples where it is present.
    - unique_values: Number of unique values observed for this field.
    - examples: Example values observed for this field.
    """

    present: int = Field(..., description="Number of samples for which this field is present")
    missing: int = Field(..., description="Number of samples for which this field is missing")
    parse_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Parse rate for this field among samples where it is present"
    )
    unique_values: int = Field(..., description="Number of unique values observed for this field")
    examples: list[NonEmptyStr] = Field(default_factory=list, description="Example values observed for this field")


class MetadataSummary(BaseModel):
    """
    Represents a summary of metadata coverage for a GEO Series.

    Attributes:
    - accession: The GEO Series accession, e.g. GSE12345.
    - platform: The platform(s) associated with this GEO Series.
    - gpl: The GPL(s) associated with this GEO Series.
    - n_samples: Total number of samples in this GEO Series.
    - subject_id: Coverage details for subject_id field.
    - age: Coverage details for age field.
    - sex: Coverage details for sex field.
    - condition: Coverage details for condition field.
    - tissue: Coverage details for tissue field.
    - cell_type: Coverage details for cell_type field.
    - disease_status: Coverage details for disease_status field.
    - warnings: List of warnings related to metadata coverage.
    """

    accession: NonEmptyStr = Field(..., description="The GEO Series accession, e.g. GSE12345")
    platform: list[NonEmptyStr] = Field(..., description="The platform(s) associated with this GEO Series")
    gpl: list[NonEmptyStr] = Field(..., description="The GPL(s) associated with this GEO Series")
    n_samples: int = Field(..., description="Total number of samples in this GEO Series")
    subject_id: FieldCoverage = Field(..., description="Coverage details for subject_id field")
    age: FieldCoverage = Field(..., description="Coverage details for age field")
    sex: FieldCoverage = Field(..., description="Coverage details for sex field")
    condition: FieldCoverage = Field(..., description="Coverage details for condition field")
    tissue: FieldCoverage = Field(..., description="Coverage details for tissue field")
    cell_type: FieldCoverage = Field(..., description="Coverage details for cell_type field")
    disease_status: FieldCoverage = Field(..., description="Coverage details for disease_status field")
    warnings: list[NonEmptyStr] = Field(
        default_factory=list, description="List of warnings related to metadata coverage"
    )

    @model_validator(mode="after")
    def validate_accession(self):
        if not self.accession.upper().startswith("GSE"):
            raise ValueError("accession must start with 'GSE'")
        return self
