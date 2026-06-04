# Dataset Quality Contrl 

This tutorial is a brief guide for performing quality control on the DNA methylation data that has been successfully retrieved.

## Quality Control Overview

MethylCurate performs several basic quality control steps that have been performed in epigenetic aging clock literature:
- **Datatype Conversion**: We check the retrieved DNA methylation data's datatype (beta values or M-values). We convert the data to beta values if it is not already, as this is the most common form used for epigenetic aging clocks.
- **Sample-level Missingness Filtering**: Samples that are missing more values than a user-defined cutoff will be removed from the analysis.  
- **CpG-level Missingness Filtering**: Features that are missing more values than a user-defined cutoff will be removed from the analysis.
- **Maximum DNAm Value Filtering**: Samples that have a maximum DNA methylation value (across all features) less than a user-defined cutoff will be removed from the analysis.
- **Interarray Correlation Filtering**: Samples that have an average interarray correlation less than a user-defined cutoff will be removed from the analysis.

The user can control the thresholds used for these quality control steps by modifying the qualtiy control config file (`qc_config.yml`) in the project directory. 

| Parameter                   | Possible Values |
|-----------------------------|-----------------|
| dnam\_cutoff                 | 0.0 - 1.0       |
| sample\_level\_missing\_cutoff | 0.0 - 1.0       |
| cpg\_level\_missing\_cutoff    | 0.0 - 1.0       |
| correlation\_cutoff          | 0.0 - 1.0       |

## Quality Control Request

In order for the quality control request to work, you have to have successfully retrieved the relevant data already. Assuming you have done that, you can subsequently request to perform quality control on the datasets that were successfully retrieved through some form of:
> Perform quality control on the aforementioned datasets.

## Outputs

### Processed DNA Methylation data

For each dataset on which quality control steps are performed, you will find the post-quality-control DNA methylation data matrix file at `<output_root>/data/<accession_code>/<accession_code>_processed_data.feather`. 