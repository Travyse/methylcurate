# Pre-existing Aging Clock Usage

This tutorial is a brief guide for running aging clocks on the datasets retrieved by MethylCurate.

## Aging Clock Overview

MethylCurate provides a wrapper around [PyAging](https://github.com/lucascamillomd/pyaging) to implement common epigenetic aging clocks. 

### Supported Aging Clocks

MethylCurate officially supports the following epigenetic aging clocks through PyAging:

- altumage
- dunedinpace
- dnamic
- dnamphenoage
- grimage
- grimage2
- horvath2013
- hannum
- intrinclock
- pcgrimage
- pchannum
- pchorvath2013
- pcphenoage
- pcskinandblood
- skinandblood
- systemsage
- systemsageblood
- systemsagebrain
- systemsageheart
- systemsagehormone
- systemsageimmune
- systemsageinflammation
- systemsagekidney
- systemsageliver
- systemsagelung
- systemsagemetabolic
- systemsagemusculoskeletal
- zhangblup
- zhangen
- zhangmortality

However, the support for epigenetic aging clocks (either through PyAging or for users' who want to include their own models) can be trivially implemented in the future.

## Aging Clock Usage

### How to Request

There are several things to keep in mind before you request the use of epigenetic aging clocks:
1. You must have successfully retrieved the DNA methylation data already. 
2. Harmonization is optional and has no impact on aging clock usage.
3. Quality control is optional and will likely impact aging clock usage in terms of result aggregation (as some samples may be removed). If quality control was run, MethylCurate will use the post-quality-control processed DNA methylation data, otherwise, it will use the retrieved data directly.
4. You must know upfront what aging clocks you want to use. MethylCurate does not, by default, use all available epigenetic aging clocks.

With those details in mind, you can request the usage of user-specified epigenetic aging clocks on the retrieved datasets through some form of:
> Please use AltumAge, Horvath2013, SkinAndBlood, and ZhangEn on the aforementioned datasets.


## Outputs

### Prediction Files

For each dataset, you can find a file (`<output_root>/analysis/<accession_code>/predictions.csv`) that contains the predictions from each aging clock for each sample. in a given dataset.