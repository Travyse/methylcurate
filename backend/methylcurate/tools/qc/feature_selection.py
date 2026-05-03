__all__ = []

import GEOparse
import pandas as pd
from typing import List
from ..contracts.geo import GEOSampleLevelMetadataBatch

# Deterministic way: sample some values from each column, if I find those that satisfy regex that tests for cg##### or rs#####, I win.
# Check for percentage cg in each column, the column that is above 90% is the CpG column. Stop at the first one.

def _get_gpl_features(gpl_id: str, destdir: str) -> List[str]:
    def _percentage_cg(total_values: int, subset_count: int) -> float:
        if total_values < 1:
            return 0.0
        return subset_count / total_values
    gpl = GEOparse.get_GEO(geo=gpl_id, destdir=destdir)  # TODO: Update destination

    features = None
    gpl_columns = gpl.table.columns.tolist()
    for col in gpl_columns:
        s = gpl.table[col].dropna().astype(str)
        total_values = len(s)
        num_cg = s.str.lower().startswith("cg", na=False).sum()
        if _percentage_cg(total_values, num_cg) >= 0.9:
            features = gpl.table[col].dropna().tolist()
            break

    if features is None:
        raise RuntimeError(f"Could not find CpG probe ID column in platform {gpl_id}") 
    return features

def find_common_cpgs(dataset_metadata: List[GEOSampleLevelMetadataBatch]) -> None:
    gpls = set()
    for dataset in dataset_metadata:
        for sample in dataset.samples:
            if sample.platform is not None:
                gpls.add(sample.platform)
    gpls = list(gpls)
    cpg_intersection = set()
    for gpl in gpls:
        features = _get_gpl_features(gpl, destdir=".")
        if len(cpg_intersection) == 0:
            cpg_intersection = set(features)
        else:
            cpg_intersection = cpg_intersection.intersection(set(features))
    return cpg_intersection
