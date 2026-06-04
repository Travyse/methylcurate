# Dataset Metadata Harmonization

This tutorial is a brief guide for harmonizing the metadata of any datasets that you have retrieved. 

## Harmonization Overview

MethylCurate currently supports harmonizing four tracked metadata fields: disease status, tissue, cell type, and sex. In general, the approach is to rely on an LLM to infer the human readable version of each dataset's field value, retrieve the top 5 results for this value when using it to search an ontology database, then using the LLM to select the result that best represents the original field value based on the dataset metadata and the provided field value. It's important to note that this harmonization is moreso for the user's consumption (for filtering or later analysis), as this has no impact on how MethylCurate interacts with the datasets. 

| Field          | Ontology       |
|----------------|----------------|
| Disease Status | Mondo          |
| Tissue         | Uberon         |
| Cell Type      | CL (Cell Ontology)             |
| Sex            | Phenotype and Trait Ontology (PATO) |

### Disease Status and Tissue

The disease status and tissue fields are harmonized a bit uniquely compared to the other fields. For these two fields, we first harmonize the label provided by the dataset (for example, following the approach described in [Harmonization Overview](#harmonization-overview), `AD` may be aligned to `Alzheimer disease`). This is done per-dataset to provide per-dataset mappings of original field values to ontologically-grounded values. Then, we look at all of the unique harmonized values across all datasets, and we then infer label groupings (for example, `Alzheimer disease` and `Frontotemporal Dementia` may be grouped as `Neurodegenerative Disease`. `Hippocampus` and `Frontal Lobe` may be grouped as `Brain`.)

### Tissue, Cell Type, Sex

These labels are harmonized as explained in the [Harmonization Overview](#harmonization-overview) without the additional grouping step.

### Missing Fields

It will likely be the case that datasets will be missing values one of the four supported fields. In these cases, those fields will be ignored for those datasets during the harmonization process.

### Unmappable Values

There are cases in which the human readable version of a particular field cannot be mapped to an ontological term. In these cases, the inferred human readable version of the field value is used as the "mapped" term. You are notified of how a field is mapped in the artifacts produced.

## Dataset Harmonization Request

In order for harmonization to work, you have to have successfully retrieved the relevant data already. Assuming you have done that, you can subsequently request for the harmonization of the metadata of the datasets that were successfully retrieved through some form of:
> Harmonize the aforementioned retrieved datasets.

You do not have to re-specify the datasets in each request after retrieving it successfully. 

## Outputs

### Human-Readable Label Files

For each dataset, you will be able to find a file (`<output_root>/data/<accession_code>/harmonization/guessed_<field_name>_labels.json`, where field_name is one of: tissue, disease, cell_type) that contains the mapping from the source field value to the human-readable label.

### Harmonized Label Files

For each dataset, you will be able to find a file (`<output_root>/data/<accession_code>/harmonization/harmonized_<field_name>_labels.json`, where field_name is one of: tissue, disease, cell_type, sex) that contains the mapping from the source field value to the harmonized value.

### Grouped Label Files

For each dataset, you will be able to find a file (`<output_root>/data/<accession_code>/<field_name>_harmonization_metadata.csv`, where field_name is one of: tissue, disease) that contains a table with the columns: original_label, harmonized_label, harmonized_group_label. This is a convenience file that allows you to easily map disease or tissue labels from raw field values to harmonized values to harmonized group labels.