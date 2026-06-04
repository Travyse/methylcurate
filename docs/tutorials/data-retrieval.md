# Retrieval of DNA Methylation Datasets

This tutorial explains the ways in which you can use MethylCurate to retrieve DNA methylation datasets from NCBI GEO.

## Dataset Retrieval Request

### Inline Dataset Request

You can take advantage of the dialogue-driven nature of MethylCurate to request for the retrieval of datasets through natural language. For example,
> Download GEO123, GEO456, GEO789

Will download the specified GEO datasets. This is useful when you plan to request few datasets.

### Request Datasets Via File

For larger requests, you can attach a `.csv`, `.tsv` or `.txt` file which contains the datasets of interest. To do so, the file must contain one of the following columns:
- accession_code
- accession
- accessions
- gse
- geo_accession 

Which should specify the accession codes of the datasets that you want to retrieve. 

Show picture of where to attach file.

## Human-in-the-Loop

Our tool uses `GEOparse` to retrieve the datasets. For some datasets, the DNA methylation data matrices are not included per-sample, making it necessary to retrieve this data from the supplementary data. To do so, MethylCurate will prompt you to select the appropriate supplementary file that contains the desired DNA methylation data.

Show picture here.

## Outputs

This request will produce several artifacts of interest. These will mostly live in the `<output_root>/data` path.

### Extraction Protocols

For each dataset retrieved, you will find an extraction protocol file at `<output_root>/data/<accession_code>/extraction_protocol.json`. This file contains the logic used by the LLM to extract the metadata for each sample in that given dataset. 

### Metadata Files

For each dataset retrieved, you will find a metadata file at `<output_root>/data/<accession_code>/sample_metadata.csv`, which contains the retrieved sample metadata for the following variables:
- Accession Code
- Disease Status
- Condition
- Tissue
- Cell Type
- Subject (Dataset specific subject name)
- Sample (GSM name)
- Age
- Sex
- DNA Methylation Platform

### DNA Methylation Dataset Matrices

For each dataset retrieved, you will find the DNA methylation data matrix file at `<output_root>/cache/<accession_code>_preqc_methylation_matrix.feather`. 

### Cached Downloads

For each dataset requested, the softfile and a copy of the dataset metadata is stored in `<output_root>/cache/`. Subsequent requests will retrieve the cached data.