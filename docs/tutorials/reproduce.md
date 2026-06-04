# Reproduce Blood and Brain Dataset Evaluation

This tutorial explains how to reproduce the analysis shown in the MethylCurate paper.

## Reproduction Steps

### Dataset Retrieval

#### Prompt Used 

We requested the datasets using the following prompt:
> Download GSE109627, GSE195834, GSE66351, GSE125895, GSE134379, GSE76105, GSE61380, GSE61107, GSE74193, GSE80970, GSE144858, GSE53740, GSE190540, GSE111629, GSE72774, GSE122244

#### Human-in-the-Loop Decisions

Several of these datasets require your input to select the correct supplementary file. See Supplementary Table 2 for the decisions we made.

### Aging Clock Request

To run the aging clocks on the retrieved datasets, we used the following prompt:
> Benchmark horvath2013, hannum, skinandblood, altumage, zhangen, zhangblup, corticalage, grimage2, grimage, phenoage, zhangmortality, systemsage, systemsageblood, systemsagebrain, pchorvath, pchannum, pcphenoage, brainage on the aforementioned datasets.

## Subsequent Analysis

### Scripts for Gathering and Analyzing Predictions

See the notebook [02_methylcurate_predictions.ipynb](https://github.com/Travyse/methylcurate/blob/main/notebooks/experiments/02_methylcurate_predictions.ipynb) for convenient scripts for gathering and analyzing the results.

### Biolearn Comparison

To check Biolearn results on the same datasets, you can run the [01_biolearn.ipynb](https://github.com/Travyse/methylcurate/blob/main/notebooks/experiments/01_biolearn.ipynb) notebook.

### Plot Creation

We used [BioRender](https://www.biorender.com) to generate the heatmaps.

