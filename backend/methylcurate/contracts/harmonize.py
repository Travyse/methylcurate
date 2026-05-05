__all__ = ["HumanReadableConceptInput", "create_ontology_mapping_model"]
from typing import Annotated, Literal

from pydantic import BaseModel, Field, create_model

HarmonizationConcept = Literal["tissue", "cell_type", "sex", "disease_status"]


class HumanReadableConceptInput(BaseModel):
    """
    Represents a human-readable concept extracted from a dataset's metadata.

    Attributes:
    - dataset_title: A concise title describing the dataset that is being harmonized.
    - dataset_summary: A description of the dataset being harmonized, including any relevant context that may assist in understanding the metadata.
    - dataset_overall_design: A description of the overall design of the dataset, including any relevant context that may assist in understanding the metadata.
    - metadata_field_name: The name of the metadata field that this concept was extracted from.
    - metadata_field_key_name: The key name of the metadata field that this concept was extracted from, if applicable.
    - concepts: A list of human readable concepts that were extracted from the metadata for a particular field. Each concept includes the original label and any relevant context.
    """

    dataset_title: str = Field(..., description="A concise title describing the dataset that is being harmonized")
    dataset_summary: str | None = Field(
        None,
        description="A description of the dataset being harmonized, including any relevant context that may assist in understanding the metadata.",
    )
    dataset_overall_design: str | None = Field(
        None,
        description="A description of the overall design of the dataset, including any relevant context that may assist in understanding the metadata.",
    )
    metadata_field_name: str = Field(
        ..., description="The name of the metadata field that this concept was extracted from."
    )
    metadata_field_key_name: str | None = Field(
        None, description="The key name of the metadata field that this concept was extracted from, if applicable."
    )
    concepts: list[str] = Field(
        ...,
        description="A list of human readable concepts that were extracted from the metadata for a particular field. Each concept includes the original label and any relevant context.",
    )


class BaseMapping(BaseModel):
    """
    Represents a base class for all ontology mappings.

    Attributes:
    - notes: Additional notes or context about the mapping.
    """

    notes: str | None = Field(None, description="Additional notes or context about the mapping")


class MondoMapping(BaseMapping):
    """
    Represents a mapping to the MONDO ontology.

    Attributes:
    - ontology: The ontology being used, in this case "mondo".
    - source_label: The label that is being harmonized to the MONDO ontology.
    - target_label: The ontological label that the source label is being mapped to.
    """

    ontology: Literal["mondo"]
    source_label: str = Field(..., description="The label that is being harmonized to the mondo ontology")
    target_label: str = Field(..., description="The ontological label that the source label is being mapped to")


class UberonMapping(BaseMapping):
    """
    Represents a mapping to the UBERON ontology.

    Attributes:
    - ontology: The ontology being used, in this case "uberon".
    - source_label: The label that is being harmonized to the UBERON ontology.
    - target_label: The ontological label that the source label is being mapped to.
    """

    ontology: Literal["uberon"]
    source_label: str = Field(..., description="The label that is being harmonized to the uberon ontology")
    target_label: str = Field(..., description="The ontological label that the source label is being mapped to")


class CLMapping(BaseMapping):
    """
    Represents a mapping to the CL ontology.

    Attributes:
    - ontology: The ontology being used, in this case "cl".
    - source_label: The label that is being harmonized to the CL ontology.
    - target_label: The ontological label that the source label is being mapped to.
    """

    ontology: Literal["cl"]
    source_label: str = Field(..., description="The label that is being harmonized to the cl ontology")
    target_label: str = Field(..., description="The ontological label that the source label is being mapped to")


class PATOMapping(BaseMapping):
    """
    Represents a mapping to the PATO ontology.

    Attributes:
    - ontology: The ontology being used, in this case "pato".
    - source_label: The label that is being harmonized to the PATO ontology.
    - target_label: The ontological label that the source label is being mapped to.
    """

    ontology: Literal["pato"]
    source_label: str = Field(..., description="The label that is being harmonized to the PATO ontology")
    target_label: Literal["male", "female"] = Field(
        ..., description="The ontological label that the source label is being mapped to"
    )


class BestGuessMapping(BaseMapping):
    """
    Represents a mapping where the best guess was made for the ontological label.

    Attributes:
    - ontology: The ontology being used, in this case "best_guess".
    - source_label: The label that is being harmonized but failed to map to the ontology, so a best guess was made based on the label and context. This field represents the original label.
    - target_label: The best guess ontological label that the source label is being mapped to, based on the label and context.
    """

    ontology: Literal["best_guess"]
    source_label: str = Field(
        ...,
        description="The label that is being harmonized but failed to map to the ontology, so a best guess was made based on the label and context. This field represents the original label.",
    )
    target_label: str = Field(
        ...,
        description="The best guess ontological label that the source label is being mapped to, based on the label and context.",
    )


class MissingMapping(BaseMapping):
    """
    Represents a mapping where the source label failed to map to any ontology.

    Attributes:
    - ontology: The ontology being used, in this case "missing".
    - source_label: The label that was meant to be harmonized but failed.
    """

    ontology: Literal["missing"]
    source_label: str = Field(..., description="The label that was meant to be harmonized but failed")


OntologicalMapping = Annotated[
    MondoMapping | UberonMapping | CLMapping | PATOMapping | BestGuessMapping | MissingMapping,
    Field(discriminator="ontology"),
]


class LabelMappingSet(BaseModel):
    """
    Represents a set of label mappings.

    Attributes:
    - mappings: A list of ontological mappings.
    """

    mappings: list[OntologicalMapping]


def create_ontology_mapping_model(
    allowed_source_labels: list[str],
    ontology_literal: str,
    ontology_name: str,
    allowed_target_labels: list[str] | None = None,
    high_level: bool = False,
) -> tuple[BaseModel, BaseModel]:
    """
    Creates a dynamic ontology mapping model based on the provided parameters.

    Parameters:
    - allowed_source_labels: A list of allowed source labels for the mapping.
    - ontology_literal: The literal value representing the ontology.
    - ontology_name: The name of the ontology.
    - allowed_target_labels: An optional list of allowed target labels for the mapping.
    - high_level: A boolean indicating if the mapping is high level.

    Returns:
        A tuple containing the dynamic ontology mapping model and the dynamic label mapping set model.
    """
    if ontology_name == "pato":
        LabelMappingSetDyn = create_model(
            f"LabelMappingSetModel__{ontology_name}__{abs(hash((allowed_source_labels, allowed_target_labels)))}",
            __base__=BaseModel,
            mappings=(
                list[PATOMapping],
                Field(
                    ...,
                    min=len(allowed_source_labels),
                    description=f"This represents the suggested mappings of the labels to the {ontology_name} ontology.",
                ),
            ),
        )
        return PATOMapping, LabelMappingSetDyn  # type: ignore

    allowed_source_labels = tuple(allowed_source_labels)  # type: ignore
    allowed_target_labels = tuple(allowed_target_labels) if allowed_target_labels is not None else None  # type: ignore

    if allowed_target_labels is not None:
        params = {
            "ontology": (
                Literal[ontology_literal],  # type: ignore
                Field(
                    ...,
                    description=f"This represents the ontology that the mapping is using, in this case {ontology_name}",
                ),
            ),
            "source_label": (
                Literal[allowed_source_labels],  # type: ignore
                Field(..., description=f"The input label to harmonize to the `{ontology_name}` ontology."),
            ),
            "target_label": (
                Literal[allowed_target_labels],  # type: ignore
                Field(..., description=f"The `{ontology_name}` ontology label that the source label maps to."),
            ),
            "notes": (str | None, Field(None, description="Additional notes or context about the mapping")),
        }
    else:
        target_label_description = (
            f"The high level category to match the source_label to, relevant to the {ontology_name} ontology."
            if high_level
            else f"The `{ontology_name}` ontology label, or something similar, that the source label maps to."
        )
        params = {
            "ontology": (
                Literal[ontology_literal],  # type: ignore
                Field(
                    ...,
                    description=f"This represents the ontology that the mapping is using, in this case {ontology_name}",
                ),
            ),
            "source_label": (
                Literal[allowed_source_labels],  # type: ignore
                Field(..., description=f"The input label to harmonize to the `{ontology_name}` ontology."),
            ),
            "target_label": (str, Field(..., description=target_label_description)),
            "notes": (str | None, Field(None, description="Additional notes or context about the mapping")),
        }

    OntologyMappingDyn = create_model(  # type: ignore
        f"OntologyMappingModel__{ontology_literal}__{abs(hash((allowed_source_labels, allowed_target_labels)))}",
        __base__=BaseModel,
        **params,
    )

    OntologicalMappingOrMissingDyn = Annotated[OntologyMappingDyn | MissingMapping, Field(discriminator="ontology")]

    LabelMappingSetDyn = create_model(
        f"LabelMappingSetModel__{ontology_literal}__{abs(hash((allowed_source_labels, allowed_target_labels)))}",
        __base__=BaseModel,
        mappings=(
            list[OntologicalMappingOrMissingDyn],
            Field(
                ...,
                min=len(allowed_source_labels),
                description=f"This represents the suggested mappings of the labels to the {ontology_name} ontology.",
            ),
        ),
    )
    return OntologicalMappingOrMissingDyn, LabelMappingSetDyn  # type: ignore


class BaseOntologyConcept(BaseModel):
    """
    Represents a base ontology concept.

    Attributes:
    - id: The unique identifier of the concept.
    - label: The label of the concept.
    """

    id: str
    label: str


class MondoConcept(BaseOntologyConcept):
    """
    Represents a MONDO ontology concept.

    Attributes:
    - ontology: The ontology being used, in this case "mondo".
    """

    ontology: Literal["mondo"]


class CLConcept(BaseOntologyConcept):
    """
    Represents a CL ontology concept.

    Attributes:
    - ontology: The ontology being used, in this case "cl".
    """

    ontology: Literal["cl"]


class UberonConcept(BaseOntologyConcept):
    """
    Represents an Uberon ontology concept.

    Attributes:
    - ontology: The ontology being used, in this case "uberon".
    """

    ontology: Literal["uberon"]


OntologyConcept = Annotated[
    MondoConcept | CLConcept | UberonConcept,
    Field(discriminator="ontology"),
]

TISSUE_GROUPS = {
    "Adipose": UberonConcept(ontology="uberon", id="UBERON:0001013", label="adipose tissue"),
    "Adrenal Gland": UberonConcept(ontology="uberon", id="UBERON:0002369", label="adrenal gland"),
    "Artery": UberonConcept(ontology="uberon", id="UBERON:0001637", label="artery"),
    "Brain": UberonConcept(ontology="uberon", id="UBERON:0000955", label="brain"),
    "Breast": UberonConcept(ontology="uberon", id="UBERON:0000310", label="breast"),
    "Cervix": UberonConcept(ontology="uberon", id="UBERON:0000002", label="uterine cervix"),
    "Colon": UberonConcept(ontology="uberon", id="UBERON:0001155", label="colon"),
    "Esophagus": UberonConcept(ontology="uberon", id="UBERON:0001043", label="esophagus"),
    "Fallopian Tube": UberonConcept(ontology="uberon", id="UBERON:0003889", label="fallopian tube"),
    "Heart": UberonConcept(ontology="uberon", id="UBERON:0000948", label="heart"),
    "Kidney": UberonConcept(ontology="uberon", id="UBERON:0002113", label="kidney"),
    "Liver": UberonConcept(ontology="uberon", id="UBERON:0002107", label="liver"),
    "Lung": UberonConcept(ontology="uberon", id="UBERON:0002048", label="lung"),
    "Oral Gland": UberonConcept(ontology="uberon", id="UBERON:0010047", label="oral gland"),
    "Skeletal Muscle": UberonConcept(ontology="uberon", id="UBERON:0014892", label="skeletal muscle organ, vertebrate"),
    "Nerve": UberonConcept(ontology="uberon", id="UBERON:0001021", label="nerve"),
    "Ovary": UberonConcept(ontology="uberon", id="UBERON:0000992", label="ovary"),
    "Pancreas": UberonConcept(ontology="uberon", id="UBERON:0001264", label="pancreas"),
    "Pituitary Gland": UberonConcept(ontology="uberon", id="UBERON:0000007", label="pituitary gland"),
    "Prostate Gland": UberonConcept(ontology="uberon", id="UBERON:0002367", label="prostate gland"),
    "Skin": UberonConcept(ontology="uberon", id="UBERON:0002097", label="skin of body"),
    "Small Intestine": UberonConcept(ontology="uberon", id="UBERON:0002108", label="small intestine"),
    "Spleen": UberonConcept(ontology="uberon", id="UBERON:0002106", label="spleen"),
    "Stomach": UberonConcept(ontology="uberon", id="UBERON:0000945", label="stomach"),
    "Testis": UberonConcept(ontology="uberon", id="UBERON:0000473", label="testis"),
    "Thyroid Gland": UberonConcept(ontology="uberon", id="UBERON:0002046", label="thyroid gland"),
    "Uterus": UberonConcept(ontology="uberon", id="UBERON:0006834", label="uterus"),
    "Vagina": UberonConcept(ontology="uberon", id="UBERON:0000996", label="vagina"),
    "Blood": UberonConcept(ontology="uberon", id="UBERON:0000178", label="blood"),
}
