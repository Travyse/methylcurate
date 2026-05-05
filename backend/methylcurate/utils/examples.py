__all__ = [
    "generate_router_interpretation_examples",
    "generate_router_clarification_examples",
    "generate_geo_metadata_extraction_examples",
    "generate_metadata_harmonization_examples",
    "generate_column_interpretation_examples",
    "generate_column_interpretation_examples_no_detection",
    "generate_general_geo_metadata_extraction_examples",
    "generate_high_level_ontology_guess_examples",
    "generate_ontology_guess_examples",
    "generate_ontology_selection_examples",
    "generate_high_level_ontology_selection_examples",
]
import json
import random
import re
import uuid
from typing import Any

import pandas as pd
from langchain_core.messages import SystemMessage

from ..contracts.harmonize import create_ontology_mapping_model
from ..contracts.router import RouterOutput
from .prompting import (
    generate_high_level_ontology_label_selection_query,
    generate_high_level_ontology_label_selection_system_prompt,
    generate_ontology_group_guess_system_prompt,
    generate_ontology_group_guess_user_query,
    generate_ontology_label_query,
    generate_ontology_label_selection_query,
    generate_ontology_label_selection_system_prompt,
    generate_ontology_label_system_prompt,
)


def generate_router_interpretation_examples(output_root: str = "outputs/"):
    """
    Generates examples for router interpretation.

    Args:
        output_root (str): The root directory for output files.

    Returns:
        tuple: A tuple containing the example user input and the example agent response.
    """
    example_user_input = "Run QC on the datasets you just downloaded"
    example_agent_response = RouterOutput.model_validate(
        {
            "subgraph": "quality_control",
            "params": {
                "output_root": output_root,
                "accessions": ["GSE12345", "GSE67890"],
            },
            "confidence": 0.95,
            "needs_clarification": False,
            "clarification_question": None,
            "reasons": ["The user requested quality control on datasets."],
        }
    )
    return example_user_input, example_agent_response


def generate_router_clarification_examples(output_root: str = "outputs/"):
    """
    Generates examples for router clarification.

    Args:
        output_root (str): The root directory for output files.

    Returns:
        tuple: A tuple containing the example user query, the initial agent response, the follow-up user query, and the example agent response.
    """
    example_user_query = "Can you sanity-check this real quick?"
    initial_agent_response = RouterOutput.model_validate(
        {
            "subgraph": "quality_control",
            "params": {
                "output_root": output_root,
                "accessions": ["GSE12345", "GSE67890"],
            },
            "confidence": 0.5,
            "needs_clarification": True,
            "clarification_question": "Would you like me to run quality control on the datasets you just downloaded?",
            "reasons": ["I am unsure what they are asking for"],
        }
    )
    follow_up_user_query = "Yes, please proceed with quality control."
    example_agent_response = RouterOutput.model_validate(
        {
            "subgraph": "quality_control",
            "params": {
                "subgraph": "quality_control",
                "output_root": output_root,
                "accessions": ["GSE12345", "GSE67890"],
            },
            "confidence": 0.95,
            "needs_clarification": False,
            "clarification_question": None,
            "reasons": ["The user requested quality control on datasets."],
        }
    )
    return example_user_query, initial_agent_response, follow_up_user_query, example_agent_response


def generate_geo_metadata_extraction_examples(n_samples=10, concept: str = "age", is_missing=False):
    """
    Generates examples for GEO metadata extraction.

    Args:
        n_samples (int): The number of samples to generate.
        concept (str): The concept to extract from the metadata.
        is_missing (bool): Whether the concept is missing in the metadata.

    Returns:
        tuple: A tuple containing the example input and the example resolution.
    """

    def _generate_random_characteristics_ch1(concept: str, is_missing: bool) -> dict[str, Any]:
        """
        Generates a random characteristics_ch1 dictionary for GEO metadata extraction examples.

        Args:
            concept (str): The concept to extract from the metadata.
            is_missing (bool): Whether the concept is missing in the metadata.
        Returns:
            dict: A dictionary representing the characteristics_ch1 field with random values, potentially missing the specified concept.
        """
        return {
            "tissue": random.choice(["Prefrontal Cortex", "Hippocampus", "Cerebellum"])
            if concept == "tissue" and not is_missing
            else None,
            "cell type": random.choice(["Neuron", "Astrocyte", "Microglia"])
            if concept == "cell_type" and not is_missing
            else None,
            "brain bank": random.choice(["Newcastle", "Oxford", "Cambridge"]),
            "post-mortem delay": random.randint(1, 24),
            "braak stage": random.randint(0, 6),
            "donor_id": f"donor_{random.randint(1, 100)}",
            "age (years)": random.randint(20, 100) if concept == "age" and not is_missing else None,
            "Sex": random.choice(["M", "F"]) if concept == "sex" and not is_missing else None,
            "ad diagnosis": random.choice(["Ctl", "AD"]) if concept == "disease_status" and not is_missing else None,
        }

    example_input = json.dumps(
        {
            "artifact": {
                "accession_code": "GSE12345",
                "path": "/path/to/GSE12345_family.soft.gz",
                "kind": "soft_file",
            },
            "title": [["Tissue_Sample {i}"] for i in range(n_samples)],
            "source_name_ch1": [
                [random.choice(["Prefrontal Cortex", "Hippocampus", "Cerebellum"])] for _ in range(n_samples)
            ],
            "description": [[]],
            "characteristics_ch1": [
                {k: v for k, v in _generate_random_characteristics_ch1(concept, is_missing).items() if v is not None}
                for _ in range(n_samples)
            ],
            "relation": None,
            "platform_id": [["GPL12345"] for _ in range(n_samples)],
        },
        indent=2,
        ensure_ascii=False,
    )

    resolution_dict = {
        "confidence": 0.95,
    }

    concept_mapper = {
        "subject_id": "donor_id",
        "age": "age (years)",
        "tissue": "tissue",
        "cell_type": "cell type",
        "sex": "Sex",
        "disease_status": "ad diagnosis",
        "platform": "platform",
    }
    if is_missing or concept == "condition":
        resolution_dict.update({"status": "missing", "notes": [f"{concept} is not present in the sample metadata."]})  # type: ignore
    else:
        resolution_dict.update(  # type: ignore
            {
                "extraction": {
                    "type": "regex",
                    "field_name": "characteristics_ch1",
                    "key_name": concept_mapper[concept],
                    "pattern": "\\s*(.+)$",
                    "delimiter": None,
                    "group_index": 1,
                    "normalization": ["strip"],
                },
                "status": "resolved",
                "notes": [f"Extracted {concept} using regex from characteristics_ch1 field."],
            }
        )

        if concept == "age":
            resolution_dict["units"] = "years"  # type: ignore
            resolution_dict["extraction"]["normalization"] = ["digits_only", "strip"]  # type: ignore
            resolution_dict["extraction"]["pattern"] = "(\\d+)"  # type: ignore

    resolution = json.dumps(resolution_dict, indent=2, ensure_ascii=False)

    return example_input, resolution


def generate_general_geo_metadata_extraction_examples(n_samples=10, is_missing=True):
    """
    Generates examples for general GEO metadata extraction.

    Args:
        n_samples (int): The number of samples to generate.
        is_missing (bool): Whether the concept is missing in the metadata.

    Returns:
        tuple: A tuple containing the example input and the example resolution.
    """

    def _generate_random_characteristics_ch1(is_missing: bool) -> dict[str, Any]:
        """
        Generates a random characteristics_ch1 dictionary for GEO metadata extraction examples.

        Args:
            is_missing (bool): Whether the concept is missing in the metadata.

        Returns:
            dict: A dictionary representing the characteristics_ch1 field with random values, potentially missing the specified concept.
        """
        return {
            "tissue": random.choice(["Prefrontal Cortex", "Hippocampus", "Cerebellum"]),
            "cell type": random.choice(["Neuron", "Astrocyte", "Microglia"]) if not is_missing else None,
            "brain bank": random.choice(["Newcastle", "Oxford", "Cambridge"]),
            "post-mortem delay": random.randint(1, 24),
            "braak stage": random.randint(0, 6),
            "donor_id": f"donor_{random.randint(1, 100)}",
            "age (years)": random.randint(20, 100),
            "Sex": random.choice(["M", "F"]),
            "ad diagnosis": random.choice(["Ctl", "AD"]),
        }

    example_input = json.dumps(
        {
            "artifact": {
                "accession_code": "GSE12345",
                "path": "/path/to/GSE12345_family.soft.gz",
                "kind": "soft_file",
            },
            "title": [["Tissue_Sample {i}"] for i in range(n_samples)],
            "source_name_ch1": [
                [random.choice(["Prefrontal Cortex", "Hippocampus", "Cerebellum"])] for _ in range(n_samples)
            ],
            "description": [[]],
            "characteristics_ch1": [
                {k: v for k, v in _generate_random_characteristics_ch1(is_missing).items() if v is not None}
                for _ in range(n_samples)
            ],
            "relation": None,
            "platform_id": [["GPL12345"] for _ in range(n_samples)],
        },
        indent=2,
        ensure_ascii=False,
    )

    example_response = json.dumps(
        {
            "subject_id": {
                "status": "resolved",
                "extraction": {
                    "field_name": "characteristics_ch1",
                    "key_name": "donor_id",
                    "pattern": "^.*$",
                    "group_index": 0,
                    "normalization": ["strip"],
                },
                "confidence": 0.95,
                "notes": ["donor_id appears to be a unique subject identifier"],
            },
            "age": {
                "status": "resolved",
                "extraction": {
                    "field_name": "characteristics_ch1",
                    "key_name": "age",
                    "pattern": "\\d+",
                    "group_index": 1,
                    "normalization": ["strip", "digits_only"],
                },
                "units": "years",
                "confidence": 0.95,
                "notes": ["Chronological age in years is present under this key name"],
            },
            "tissue": {
                "status": "resolved",
                "extraction": {
                    "field_name": "source_name_ch1",
                    "pattern": "^.*$",
                    "group_index": 0,
                    "normalization": ["strip"],
                },
                "confidence": 0.95,
                "notes": ["Brain tissue soruces are mentioned under 'source_name_ch1'."],
            },
            "cell_type": {
                "status": "missing",
                "candidate_field_names": [
                    "title",
                    "source_name_ch1",
                    "description",
                    "characteristics_ch1",
                    "relation",
                    "platform_id",
                ],
                "candidate_key_names": [
                    "tissue",
                    "brain bank",
                    "post-mortem delay",
                    "braak stage",
                    "donor_id",
                    "age (years)",
                    "Sex",
                    "ad diagnosis",
                ],
                "confidence": 0.95,
                "notes": ["There are no field_names or key_names that mention a cell type."],
            },
            "sex": {
                "status": "resolved",
                "extraction": {
                    "field_name": "characteristics_ch1",
                    "key_name": "Sex",
                    "pattern": "^.*$",
                    "group_index": 0,
                    "normalization": ["strip"],
                },
                "confidence": 0.95,
                "notes": ["Sex is explicitly mentioned under 'Sex'"],
            },
            "disease_status": {
                "status": "resolved",
                "extraction": {
                    "field_name": "characteristics_ch1",
                    "key_name": "ad diagnosis",
                    "pattern": "^.*$",
                    "control_value": "Ctl",
                    "group_index": 0,
                    "normalization": ["strip"],
                },
                "confidence": 0.95,
                "notes": ["Disease status is explicitly mentioned under 'ad diagnosis'"],
            },
            "platform": {
                "status": "resolved",
                "extraction": {
                    "field_name": "platform_id",
                    "pattern": "^.*$",
                    "group_index": 0,
                    "normalization": ["strip"],
                },
                "confidence": 0.95,
                "notes": ["GEO Platform GPL present"],
            },
        },
        indent=2,
        ensure_ascii=False,
    )

    return example_input, example_response


def generate_metadata_harmonization_examples() -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Generates examples for metadata harmonization.

    Returns:
        tuple: A tuple containing the example input and the example agent response.
    """
    example_input = {
        "concept": "tissue",
        "unique_values": [
            "pfc",
            "prefrontal cortex",
            "brain: prefrontal cortex",
            "pre-frontal cortex",
            "prefrontalcortex",
        ],
    }
    example_agent_response = {
        "harmonized_mapper": {
            "prefrontal cortex": [
                "pfc",
                "prefrontal cortex",
                "brain: prefrontal cortex",
                "pre-frontal cortex",
                "prefrontalcortex",
            ]
        },
        "confidence": 0.9,
        "needs_human_review": False,
        "notes": ["Mapped various representations of prefrontal cortex to 'prefrontal cortex'."],
    }
    return example_input, example_agent_response


def generate_column_interpretation_examples(alt=False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generates examples for column interpretation.

    Args:
        alt (bool): Whether to generate alternative column names.

    Returns:
        tuple: A tuple containing the example input DataFrame and the example agent response.
    """
    example_input = {
        "columns": [
            "sample_1_beta",
            "sample_1_Detection_pval",
            "sample_2_beta",
            "sample_2_Detection_pval",
            "sample_3_beta",
            "sample_3_Detection_pval",
            "sample_4_beta",
            "sample_4_Detection_pval",
            "sample_5_beta",
            "sample_5_Detection_pval",
        ],
        "rows": [[random.uniform(0, 1) for _ in range(10)] for _ in range(5)],
        "index": ["cg00000029", "cg00000108", "cg00000109", "cg00000165", "cg00000236"],
    }
    if alt:
        cols = [f"{random.randint(1, 10)}-{uuid.uuid4().hex[:4]}_{uuid.uuid4().hex[:8]}" for _ in range(5)]
        cols = [
            val
            for pair in zip([f"{c}" for c in cols], [f"{c}_Detection_pval" for c in cols], strict=True)
            for val in pair
        ]
        example_input["columns"] = cols
    df = pd.DataFrame(data=example_input["rows"], columns=example_input["columns"], index=example_input["index"])

    example_agent_response = {
        "beta_column": {
            "status": "resolved",
            "pattern": "^[a-zA-Z0-9]+_[a-zA-Z0-9]+_beta$" if not alt else r"^\d+-[a-zA-Z0-9]+_[a-zA-Z0-9]+$",
            "column_evidence": [
                df.columns.tolist()[i]
                for i in range(len(df.columns))
                if re.search(
                    "^[a-zA-Z0-9]+_[a-zA-Z0-9]+_beta$" if not alt else r"^\d+-[a-zA-Z0-9]+_[a-zA-Z0-9]+$",
                    df.columns[i],
                    re.IGNORECASE,
                )
            ],
            "evidence": [
                "Identified beta value columns based on the '_beta' suffix in column names."
                if not alt
                else "Identified beta-like values in these columns which are bound between 0 and 1, and are not detection columns."
            ],
        },
        "detection_column": {
            "status": "resolved",
            "pattern": "^[a-zA-Z0-9]+_[a-zA-Z0-9]+_Detection_pval$"
            if not alt
            else r"^\d+-[a-zA-Z0-9]+_[a-zA-Z0-9]+_Detection_pval$",
            "column_evidence": [
                df.columns.tolist()[i]
                for i in range(len(df.columns))
                if re.search(
                    "^[a-zA-Z0-9]+_[a-zA-Z0-9]+_Detection_pval$"
                    if not alt
                    else r"^\d+-[a-zA-Z0-9]+_[a-zA-Z0-9]+_Detection_pval$",
                    df.columns[i],
                    re.IGNORECASE,
                )
            ],
            "evidence": [
                "Identified detection p-value columns based on the '_Detection_pval' suffix in column names."
                if not alt
                else "Identified _Detection_pval prefix in these columns"
            ],
        },
    }

    return df.to_markdown(index=False), example_agent_response  # type: ignore


def generate_column_interpretation_examples_no_detection() -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generates examples for column interpretation without detection columns.

    Returns:
        tuple: A tuple containing the example input DataFrame and the example agent response.
    """
    example_input = {
        "rows": [[random.uniform(0, 1) for _ in range(5)] for _ in range(5)],
        "index": ["cg00000029", "cg00000108", "cg00000109", "cg00000165", "cg00000236"],
    }
    cols = [f"{random.randint(1, 10)}-{uuid.uuid4().hex[:4]}_{uuid.uuid4().hex[:8]}" for _ in range(5)]
    example_input["columns"] = cols
    df = pd.DataFrame(data=example_input["rows"], columns=example_input["columns"], index=example_input["index"])

    example_agent_response = {
        "beta_column": {
            "status": "resolved",
            "pattern": r"^\d+-[a-zA-Z0-9]+_[a-zA-Z0-9]+$",
            "column_evidence": cols,
            "evidence": [
                "Identified beta-like values in these columns which are bound between 0 and 1, and are not detection columns."
            ],
        },
        "detection_column": {
            "status": "missing",
            "candidate_columns": cols,
            "absence_evidence": [
                "No columns contain evidence of being detection p-value columns such as having 'detect' or 'pval' in the column name."
            ],
        },
    }

    return df.to_markdown(index=False), example_agent_response  # type: ignore


def _harmonization_dataset_info(ontology: str = "mondo") -> dict[str, str]:
    """
    Retrieves dataset information based on the specified ontology.

    Args:
        ontology (str): The ontology to use for dataset information.

    Returns:
        Dict[str, str]: A dictionary containing dataset information.
    """
    if ontology == "cl":
        return {
            "dataset_title": "Illumina EPIC DNA Methylation Profiling of Human Brain Cell Types Including Microglia, Astrocytes, and Neurons",
            "dataset_summary": "Genome-wide DNA methylation analysis was performed on purified human brain cell types, including microglia (MG), astrocytes (AST), and neurons (NEU), to characterize cell type-specific epigenetic signatures across the genome. Genomic DNA isolated from enriched cell populations of microglia (MG), astrocytes (AST), and neurons (NEU) was profiled using the Illumina MethylationEPIC BeadChip.",
            "dataset_overall_design": "Human brain tissue was dissociated and separated into enriched cell populations representing microglia (MG), astrocytes (AST), and neurons (NEU) using established cell isolation approaches based on cell type-specific markers. Multiple biological replicates were collected for each cell type, including microglia (MG), astrocytes (AST), and neurons (NEU), to capture reproducible methylation profiles. Genomic DNA was extracted from purified microglia (MG), astrocyte (AST), and neuron (NEU) fractions, bisulfite-converted, and randomized across Illumina MethylationEPIC arrays to reduce batch effects. Raw data were processed in R using minfi and ChAMP, including probe-level quality control, NOOB background correction, and functional normalization. Differentially methylated positions and regions distinguishing microglia (MG), astrocytes (AST), and neurons (NEU) were identified using limma and DMRcate, with adjustment for technical covariates such as array and batch. Statistical significance was defined at adjusted FDR < 0.05.",
        }
    else:
        return {
            "dataset_title": "Illumina EPIC DNA Methylation Profiling of Dementia with Lewy Bodies in the Anterior Cingulate Cortex and Substantia Nigra",
            "dataset_summary": "Genome-wide DNA methylation analysis of 312 postmortem brain samples from donors with Alzheimer disease (AD), mild cognitive impairment (MCI), dementia with Lewy bodies (DLB), and neurologically normal controls was performed in two vulnerable brain regions, the anterior cingulate cortex (ACC) and substantia nigra (SN), using the Illumina MethylationEPIC BeadChip.",
            "dataset_overall_design": "Postmortem brain tissue from the anterior cingulate cortex (ACC) and substantia nigra (SN) was obtained from 312 donors, including cases with neuropathologically and clinically characterized Alzheimer disease (AD), mild cognitive impairment (MCI), and dementia with Lewy bodies (DLB), as well as neurologically normal controls. Clinical diagnoses were established during life using standard consensus criteria for AD, MCI, and DLB and were confirmed after death by neuropathological examination, including assessment of amyloid-beta, tau, and alpha-synuclein pathology as appropriate. DNA was extracted from frozen tissue, bisulfite-converted, and randomized across Illumina MethylationEPIC arrays. Raw data were processed in R using minfi and ChAMP, with probe-level quality control, NOOB background correction, functional normalization, and adjustment for covariates including age, sex, postmortem interval, neuronal proportion, and batch effects. Differentially methylated positions and regions associated with AD, MCI, and DLB were identified using limma and DMRcate, with significance defined at adjusted FDR < 0.05.",
        }


def _harmonization_ontology_dict() -> dict[str, dict[str, str]]:
    """
    Retrieves ontology information for harmonization.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary containing ontology information.
    """
    return {
        "mondo": {"ontology_name": "Mondo", "target_label": "disease/condition"},
        "uberon": {"ontology_name": "Uberon", "target_label": "tissue"},
        "cl": {"ontology_name": "Cell Ontology (CL)", "target_label": "cell type"},
        "pato": {"ontology_name": "Phenotype And Trait Ontology (PATO)", "target_label": "sex"},
    }


def _high_level_ontology_labels(ontology: str) -> tuple[list[str], Any, Any]:
    """
    Retrieves high-level ontology labels based on the specified ontology.

    Args:
        ontology (str): The ontology to use for high-level labels.

    Returns:
        tuple: A tuple containing the list of labels, the LabelMappingSetDyn class, and the label mapping set instance.
    """
    if ontology == "mondo":
        labels = [
            "Alzheimer disease",
            "Parkinson disease",
            "Lewy body dementia",
            "schizophrenia",
            "HIV infectious disease",
            "diabetes mellitus",
        ]
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="mondo",
            ontology_name="Mondo",
            allowed_target_labels=None,
            high_level=True,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "mondo",
                        "source_label": "Alzheimer disease",
                        "target_label": "neurodegenerative disease",
                        "notes": "Alzheimer disease is a neurodegenerative disease characterized by progressive cognitive decline and neuropathological features such as amyloid plaques and neurofibrillary tangles.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "Parkinson disease",
                        "target_label": "neurodegenerative disease",
                        "notes": "Parkinson disease is a neurodegenerative disorder characterized by motor symptoms such as tremors, rigidity, and bradykinesia, as well as non-motor symptoms including cognitive impairment and mood disorders.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "Lewy body dementia",
                        "target_label": "neurodegenerative disease",
                        "notes": "Lewy body dementia is a neurodegenerative disorder characterized by the presence of Lewy bodies in the brain, leading to cognitive decline, visual hallucinations, and motor symptoms similar to Parkinson's disease.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "schizophrenia",
                        "target_label": "psychiatric disorder",
                        "notes": "Schizophrenia is a chronic psychiatric disorder characterized by symptoms such as delusions, hallucinations, disorganized thinking, and social withdrawal.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "HIV infectious disease",
                        "target_label": "infectious disease",
                        "notes": "HIV infectious disease is caused by the human immunodeficiency virus (HIV), which attacks the immune system and can lead to acquired immunodeficiency syndrome (AIDS) if not treated.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "diabetes mellitus",
                        "target_label": "disease of metabolism",
                        "notes": "Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels over a prolonged period, which can lead to various complications if not managed properly.",
                    },
                ]
            }
        )
        return labels, LabelMappingSetDyn, label_mapping_set

    elif ontology == "uberon":
        labels = ["prefrontal cortex", "hippocampus", "cerebellum", "sigmoid colon", "coronary artery", "aorta artery"]
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="uberon",
            ontology_name="Uberon",
            allowed_target_labels=None,
            high_level=True,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "uberon",
                        "source_label": "prefrontal cortex",
                        "target_label": "brain",
                        "notes": "Prefrontal cortex is a region of the brain involved in complex cognitive behavior, decision making, and moderating social behavior.",
                    },
                    {
                        "ontology": "uberon",
                        "source_label": "hippocampus",
                        "target_label": "brain",
                        "notes": "Hippocampus is a region of the brain involved in memory formation, spatial navigation, and emotional regulation.",
                    },
                    {
                        "ontology": "uberon",
                        "source_label": "cerebellum",
                        "target_label": "brain",
                        "notes": "Cerebellum is a region of the brain that plays an important role in motor control, coordination, and balance.",
                    },
                    {
                        "ontology": "uberon",
                        "source_label": "sigmoid colon",
                        "target_label": "sigmoid colon",
                        "notes": "Sigmoid colon is a part of the large intestine that connects the descending colon to the rectum.",
                    },
                    {
                        "ontology": "uberon",
                        "source_label": "coronary artery",
                        "target_label": "artery",
                        "notes": "Coronary artery is a blood vessel that supplies oxygen-rich blood to the heart muscle.",
                    },
                    {
                        "ontology": "uberon",
                        "source_label": "aorta artery",
                        "target_label": "artery",
                        "notes": "Aorta artery is the main artery that carries oxygen-rich blood from the heart to the rest of the body.",
                    },
                ]
            }
        )
        return labels, LabelMappingSetDyn, label_mapping_set

    else:
        raise ValueError(f"Unsupported ontology: {ontology}")


def generate_high_level_ontology_guess_examples(ontology: str):
    """
    Generates high-level ontology guess examples based on the specified ontology.

    Args:
        ontology (str): The ontology to use for generating examples.

    Returns:
        SystemMessage: A system message containing the generated examples.
    """
    ontology_dict = _harmonization_ontology_dict()[ontology]
    ontology_name = ontology_dict["ontology_name"]
    target_label = ontology_dict["target_label"]
    labels, LabelMappingSetDyn, label_mapping_set = _high_level_ontology_labels(ontology)
    user_query = generate_ontology_group_guess_user_query(
        ontology_name=ontology_name, target_label=target_label, labels=", ".join(labels)
    )
    agent_response = json.dumps(label_mapping_set.model_dump(), indent=2)
    example = f"User Query:\n{user_query}\n\nAgent Response:\n{agent_response}"
    system_prompt = generate_ontology_group_guess_system_prompt(
        ontology_name=ontology_name, target_label=target_label, example=example
    )
    system_message = SystemMessage(content=system_prompt)
    return system_message


def _ontology_labels(ontology: str) -> tuple[list[str], Any, Any]:
    """
    Retrieves ontology labels based on the specified ontology.

    Args:
        ontology (str): The ontology to use for labels.

    Returns:
        tuple: A tuple containing the list of labels, the LabelMappingSetDyn class, and the label mapping set instance.
    """
    if ontology == "mondo":
        labels = ["DLB", "AD", "MCI"]
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="mondo",
            ontology_name="Mondo",
            allowed_target_labels=None,
            high_level=False,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "mondo",
                        "source_label": "DLB",
                        "target_label": "dementia with Lewy bodies",
                        "notes": "This is the clearest and most direct expansion of the abbreviation. It exactly matches the dataset title, summary, and overall design, so there is no ambiguity here. It is also the standard disease wording used in clinical and ontological contexts.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "AD",
                        "target_label": "Alzheimer disease",
                        "notes": "This is the explicit expansion given in the dataset metadata. The wording “Alzheimer disease” should be preferred over “Alzheimer’s disease” because it preserves the source phrasing exactly and aligns well with common ontology naming conventions.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "MCI",
                        "target_label": "mild cognitive impairment",
                        "notes": "This is the exact expansion provided in the dataset context. While MCI is a syndrome/state rather than a single fully specific disease entity, the dataset explicitly uses it as one of the diagnostic groups, so the closest faithful label is simply “mild cognitive impairment” without adding unsupported qualifiers such as amnestic or Alzheimer-type.",
                    },
                ]
            }
        )
        return labels, LabelMappingSetDyn, label_mapping_set

    elif ontology == "uberon":
        labels = ["ACC", "SN"]
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="uberon",
            ontology_name="Uberon",
            allowed_target_labels=None,
            high_level=False,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "uberon",
                        "source_label": "ACC",
                        "target_label": "anterior cingulate cortex",
                        "notes": "Anterior cingulate cortex is a region of the brain involved in functions such as emotion regulation, decision making, and autonomic control.",
                    },
                    {
                        "ontology": "uberon",
                        "source_label": "SN",
                        "target_label": "substantia nigra",
                        "notes": "Substantia nigra is a region of the brain involved in movement control and reward processing.",
                    },
                ]
            }
        )
        return labels, LabelMappingSetDyn, label_mapping_set

    elif ontology == "cl":
        labels = ["MG", "AST", "NEU"]
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="cl",
            ontology_name="Cell Ontology",
            allowed_target_labels=None,
            high_level=False,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "cl",
                        "source_label": "MG",
                        "target_label": "microglia",
                        "notes": "Microglia are a type of glial cell located throughout the brain and spinal cord, acting as the main form of active immune defense in the central nervous system.",
                    },
                    {
                        "ontology": "cl",
                        "source_label": "AST",
                        "target_label": "astrocytes",
                        "notes": "Astrocytes are a type of glial cell in the brain and spinal cord, involved in various functions including support of neuronal metabolism, maintenance of the blood-brain barrier, and repair after injury.",
                    },
                    {
                        "ontology": "cl",
                        "source_label": "NEU",
                        "target_label": "neurons",
                        "notes": "Neurons are the primary signaling cells in the nervous system, responsible for transmitting information throughout the brain and body.",
                    },
                ]
            }
        )
        return labels, LabelMappingSetDyn, label_mapping_set

    elif ontology == "pato":
        labels = ["0", "1"]
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="pato",
            ontology_name="Phenotype And Trait Ontology (PATO)",
            allowed_target_labels=None,
            high_level=False,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "pato",
                        "source_label": "0",
                        "target_label": "male",
                        "notes": "Represents the male sex.",
                    },
                    {
                        "ontology": "pato",
                        "source_label": "1",
                        "target_label": "female",
                        "notes": "Represents the female sex.",
                    },
                ]
            }
        )
        return labels, LabelMappingSetDyn, label_mapping_set

    else:
        raise ValueError(f"Unsupported ontology: {ontology}")


def _field_names(ontology: str) -> dict[str, str]:
    """
    Retrieves the field names for the specified ontology.

    Args:
        ontology (str): The ontology to retrieve field names for.

    Returns:
        dict: A dictionary containing the metadata field name and key name.
    """
    if ontology == "mondo":
        return {"metadata_field_name": "characteristics_ch1", "metadata_key_name": "diagnosis"}
    elif ontology == "uberon":
        return {"metadata_field_name": "characteristics_ch1", "metadata_key_name": "tissue"}
    elif ontology == "cl":
        return {"metadata_field_name": "characteristics_ch1", "metadata_key_name": "cell_type"}
    elif ontology == "pato":
        return {"metadata_field_name": "characteristics_ch1", "metadata_key_name": "sex"}
    else:
        raise ValueError(f"Unsupported ontology: {ontology}")


def generate_ontology_guess_examples(ontology: str) -> SystemMessage:
    """
    Generates ontology guess examples based on the specified ontology.

    Args:
        ontology (str): The ontology to use for generating examples.

    Returns:
        SystemMessage: A system message containing the generated examples.
    """
    metadata_dict = _field_names(ontology)
    example_params = _harmonization_dataset_info(ontology=ontology)
    example_params["metadata_field_name"] = metadata_dict["metadata_field_name"]
    example_params["metadata_key_name"] = metadata_dict["metadata_key_name"]
    ontology_dict = _harmonization_ontology_dict()[ontology]
    example_params["ontology_name"] = ontology_dict["ontology_name"]
    example_params["target_label"] = ontology_dict["target_label"]
    example_labels, ExampleLabelMappingSetDyn, example_label_mapping_set = _ontology_labels(ontology)
    example_params["labels"] = ", ".join(example_labels)
    example_params["json_schema"] = ExampleLabelMappingSetDyn.model_json_schema()
    user_query = generate_ontology_label_query(**example_params)
    agent_response = json.dumps(example_label_mapping_set.model_dump(), indent=2)
    example = f"User Query:\n{user_query}\n\nAgent Response:\n{agent_response}"
    system_prompt = generate_ontology_label_system_prompt(
        ontology_name=ontology_dict["ontology_name"], target_label=ontology_dict["target_label"], example=example
    )
    system_message = SystemMessage(content=system_prompt)
    return system_message


def _suggested_ontology_labels(ontology: str) -> tuple[dict[str, list[str]], Any, Any]:
    """
    Retrieves suggested ontology labels based on the specified ontology.

    Args:
        ontology (str): The ontology to use for suggested labels.

    Returns:
        tuple: A tuple containing the list of labels, the LabelMappingSetDyn class, and the label mapping set instance.
    """
    if ontology == "mondo":
        labels = ["dementia with lewy bodies", "alzheimer disease", "mild cognitive impairment"]
        target_label_dict = {
            "dementia with lewy bodies": [
                "Lewy body dementia",
                "progressive dementia with neuroserpin inclusion bodies",
                "familial encephalopathy with neuroserpin inclusion bodies",
                "Pick disease",
                "early-onset Lafora body disease",
            ],
            "alzheimer disease": [
                "Alzheimer disease",
                "Alzheimer disease, degu",
                "Alzheimer disease, dog",
                "Alzheimer disease, sheep",
                "Alzheimer disease, domestic cat",
            ],
            "mild cognitive impairment": [
                "mild cognitive impairment",
                "adult onset demyelinating leukodystrophy",
                "spinocerebellar ataxia 19/22",
                "dystonia 22, adult-onset",
                "Smith-Magenis syndrome",
            ],
        }
        target_labels = sorted(set([label for sublist in target_label_dict.values() for label in sublist]))
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="mondo",
            ontology_name="Mondo",
            allowed_target_labels=target_labels,
            high_level=False,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "mondo",
                        "source_label": "dementia with lewy bodies",
                        "target_label": "Lewy body dementia",
                        "notes": "This is the closest match to the dataset context and suggested label.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "alzheimer disease",
                        "target_label": "Alzheimer disease",
                        "notes": "This is an exact match to the suggested label.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "mild cognitive impairment",
                        "target_label": "mild cognitive impairment",
                        "notes": "This is an exact match to the suggested label.",
                    },
                ]
            }
        )
        return target_label_dict, LabelMappingSetDyn, label_mapping_set

    elif ontology == "uberon":
        labels = ["anterior cingulate cortex", "substantia nigra"]
        target_label_dict = {
            "anterior cingulate cortex": [
                "anterior cingulate cortex",
                "rostral anterior cingulate cortex",
                "caudal anterior cingulate cortex",
                "anterior cingulate gyrus",
                "cingulate cortex",
            ],
            "substantia nigra": [
                "substantia nigra",
                "substantia nigra pars reticulata",
                "substantia nigra pars compacta",
                "substantia nigra pars lateralis",
                "midbrain tegmentum",
            ],
        }
        target_labels = sorted(set([label for sublist in target_label_dict.values() for label in sublist]))
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="uberon",
            ontology_name="Uberon",
            allowed_target_labels=target_labels,
            high_level=False,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "uberon",
                        "source_label": "anterior cingulate cortex",
                        "target_label": "anterior cingulate cortex",
                        "notes": "This is an exact match",
                    },
                    {
                        "ontology": "uberon",
                        "source_label": "substantia nigra",
                        "target_label": "substantia nigra",
                        "notes": "This is an exact match",
                    },
                ]
            }
        )
        return target_label_dict, LabelMappingSetDyn, label_mapping_set

    elif ontology == "cl":
        labels = ["microglia", "astrocytes", "neurons"]
        target_label_dict = {
            "microglia": [
                "microglial cell",
                "microglial cell (Mmus)",
                "immature microglial cell",
                "mature microglial cell",
                "astrocyte",
            ],
            "astrocytes": [
                "astrocytes",
                "astrocyte-restricted precursor",
                "astrocyte of the forebrain",
                "astrocyte of the cerebellum",
                "astrocyte of the cerebrum (Mmus)",
            ],
            "neurons": [
                "neuron",
                "neuronal receptor cell",
                "neuron associated cell",
                "neuron of the forebrain",
                "neuronal-restricted precursor",
            ],
        }
        target_labels = sorted(set([label for sublist in target_label_dict.values() for label in sublist]))
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="cl",
            ontology_name="Cell Ontology",
            allowed_target_labels=target_labels,
            high_level=False,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "cl",
                        "source_label": "microglia",
                        "target_label": "microglial cell",
                        "notes": "This is the closest match to the dataset context and suggested label.",
                    },
                    {
                        "ontology": "cl",
                        "source_label": "astrocytes",
                        "target_label": "astrocyte",
                        "notes": "This is the closest match to the dataset context and suggested label.",
                    },
                    {
                        "ontology": "cl",
                        "source_label": "neurons",
                        "target_label": "neuron",
                        "notes": "This is the closest match to the dataset context and suggested label.",
                    },
                ]
            }
        )
        return target_label_dict, LabelMappingSetDyn, label_mapping_set

    elif ontology == "pato":
        labels = ["0", "1"]
        target_label_dict = {"0": ["male", "female"], "1": ["male", "female"]}
        target_labels = sorted(set([label for sublist in target_label_dict.values() for label in sublist]))
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="pato",
            ontology_name="Phenotype And Trait Ontology (PATO)",
            allowed_target_labels=target_labels,
            high_level=False,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "pato",
                        "source_label": "0",
                        "target_label": "male",
                        "notes": "This is the closest match to the dataset context and suggested label.",
                    },
                    {
                        "ontology": "pato",
                        "source_label": "1",
                        "target_label": "female",
                        "notes": "This is the closest match to the dataset context and suggested label.",
                    },
                ]
            }
        )
        return target_label_dict, LabelMappingSetDyn, label_mapping_set

    else:
        raise ValueError(f"Unsupported ontology: {ontology}")


def generate_ontology_selection_examples(ontology: str) -> SystemMessage:
    """
    Generates ontology selection examples based on the specified ontology.

    Args:
        ontology (str): The ontology to use for generating examples.

    Returns:
        SystemMessage: A system message containing the generated examples.
    """
    example_params = _harmonization_dataset_info(ontology=ontology)
    ontology_dict = _harmonization_ontology_dict()[ontology]
    example_params["ontology_name"] = ontology_dict["ontology_name"]
    example_params["target_label"] = ontology_dict["target_label"]
    example_label_dict, ExampleLabelMappingSetDyn, example_label_mapping_set = _suggested_ontology_labels(ontology)
    example_input_dataframe = pd.DataFrame(
        {
            "Label": example_label_dict.keys(),
            f"Putative {ontology} Labels": [", ".join(candidates) for candidates in example_label_dict.values()],
        }
    )
    example_params["json_schema"] = ExampleLabelMappingSetDyn.model_json_schema()
    example_params["input"] = example_input_dataframe.to_markdown(index=False)
    user_query = generate_ontology_label_selection_query(**example_params)
    agent_response = json.dumps(example_label_mapping_set.model_dump(), indent=2)
    example = f"User Query:\n{user_query}\n\nAgent Response:\n{agent_response}"
    system_prompt = generate_ontology_label_selection_system_prompt(
        ontology_name=ontology_dict["ontology_name"], example=example
    )
    system_message = SystemMessage(content=system_prompt)
    return system_message


def _suggested_high_level_ontology_labels(ontology: str) -> tuple[dict[str, list[str]], Any, Any]:
    """
    Retrieves suggested high-level ontology labels based on the specified ontology.

    Args:
        ontology (str): The ontology to use for suggested labels.

    Returns:
        tuple: A tuple containing the list of labels, the LabelMappingSetDyn class, and the label mapping set instance.
    """
    if ontology == "mondo":
        labels = ["neurodegenerative disease", "psychiatric disorder", "infectious disease"]
        target_label_dict = {
            "neurodegenerative disease": [
                "neurodegenerative disease",
                "neurodegenerative disease, non-human animal",
                "Huntington disease-like 1",
                "inherited neurodegenerative disorder",
                "neurodegenerative vacuolar storage disease, dog",
            ],
            "psychiatric disorder": [
                "psychiatric disorder",
                "psychiatric disorder, non-human animal",
                "myoclonic epilepsy, congenital deafness, macular dystrophy, and psychiatric disorders",
                "psychogenic movement disorders",
                "Huntington disease-like 1",
            ],
            "infectious disease": [
                "infectious disease",
                "infectious disease characteristic",
                "infectious disease with sepsis",
                "infectious disease, non-human animal",
                "HHV-7 infectious disease",
            ],
        }
        target_labels = sorted(set([label for sublist in target_label_dict.values() for label in sublist]))
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="mondo",
            ontology_name="Mondo",
            allowed_target_labels=target_labels,
            high_level=False,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "mondo",
                        "source_label": "neurodegenerative disease",
                        "target_label": "neurodegenerative disease",
                        "notes": "This is an exact match to the suggested label.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "psychiatric disorder",
                        "target_label": "psychiatric disorder",
                        "notes": "This is an exact match to the suggested label.",
                    },
                    {
                        "ontology": "mondo",
                        "source_label": "infectious disease",
                        "target_label": "infectious disease",
                        "notes": "This is an exact match to the suggested label.",
                    },
                ]
            }
        )
        return target_label_dict, LabelMappingSetDyn, label_mapping_set

    elif ontology == "uberon":
        labels = ["brain", "sigmoid colon", "artery"]
        target_label_dict = {
            "brain": ["brain", "forebrain", "midbrain", "hindbrain", "brainstem"],
            "sigmoid colon": ["sigmoid colon", "colon", "rectum", "proximal-distal subdivision of colon", "hindgut"],
            "artery": [
                "artery",
                "endothelium of artery",
                "pharyngeal arch artery",
                "maxillary artery",
                "ovarian artery",
            ],
        }
        target_labels = sorted(set([label for sublist in target_label_dict.values() for label in sublist]))
        _, LabelMappingSetDyn = create_ontology_mapping_model(
            allowed_source_labels=labels,
            ontology_literal="uberon",
            ontology_name="Uberon",
            allowed_target_labels=target_labels,
            high_level=False,
        )
        label_mapping_set = LabelMappingSetDyn.model_validate(
            {
                "mappings": [
                    {
                        "ontology": "uberon",
                        "source_label": "brain",
                        "target_label": "brain",
                        "notes": "This is an exact match to the suggested label.",
                    },
                    {
                        "ontology": "uberon",
                        "source_label": "sigmoid colon",
                        "target_label": "sigmoid colon",
                        "notes": "This is an exact match to the suggested label.",
                    },
                    {
                        "ontology": "uberon",
                        "source_label": "artery",
                        "target_label": "artery",
                        "notes": "This is an exact match to the suggested label.",
                    },
                ]
            }
        )
        return target_label_dict, LabelMappingSetDyn, label_mapping_set
    raise ValueError(f"Unknown ontology: {ontology}")


def generate_high_level_ontology_selection_examples(ontology: str) -> SystemMessage:
    """
    Generates high-level ontology selection examples based on the specified ontology.

    Args:
        ontology (str): The ontology to use for generating examples.

    Returns:
        SystemMessage: A system message containing the generated examples.
    """
    example_params = _harmonization_dataset_info(ontology=ontology)
    ontology_dict = _harmonization_ontology_dict()[ontology]
    example_params["ontology_name"] = ontology_dict["ontology_name"]
    example_params["target_label"] = ontology_dict["target_label"]
    example_label_dict, ExampleLabelMappingSetDyn, example_label_mapping_set = _suggested_high_level_ontology_labels(
        ontology
    )
    example_input_dataframe = pd.DataFrame(
        {
            "Label": example_label_dict.keys(),
            f"Putative {ontology} Labels": [", ".join(candidates) for candidates in example_label_dict.values()],
        }
    )
    example_params["json_schema"] = ExampleLabelMappingSetDyn.model_json_schema()
    example_params["input"] = example_input_dataframe.to_markdown(index=False)
    user_query = generate_high_level_ontology_label_selection_query(**example_params)
    agent_response = json.dumps(example_label_mapping_set.model_dump(), indent=2)
    example = f"User Query:\n{user_query}\n\nAgent Response:\n{agent_response}"
    system_prompt = generate_high_level_ontology_label_selection_system_prompt(
        ontology_name=ontology_dict["ontology_name"], example=example
    )
    system_message = SystemMessage(content=system_prompt)
    return system_message
