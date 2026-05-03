__all__ = []

import GEOparse
import pandas as pd
from typing import List
from ...contracts.geo import GeoSampleLevelMetadataBatch

# Deterministic way: sample some values from each column, if I find those that satisfy regex that tests for cg##### or rs#####, I win.
# Check for percentage cg in each column, the column that is above 90% is the CpG column. Stop at the first one.

def _get_gpl_features(gpl_id: str, destdir: str) -> List[str]:
    """Retrieve the list of CpG probe IDs for a GEO platform record.

    Downloads the GPL record and scans its table columns for a column
    whose content is at least 90% CpG identifiers (case-insensitive
    ``cg*`` prefix).

    Args:
        gpl_id: GEO platform accession (e.g. ``"GPL13534"``).
        destdir: Directory used for the GEOparse download.

    Returns:
        List of CpG probe identifier strings found in the platform table.

    Raises:
        RuntimeError: If no suitable CpG probe ID column is found.
    """
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

def find_common_cpgs(dataset_metadata: List[GeoSampleLevelMetadataBatch]) -> set:
    """Compute the intersection of CpG probes across all platforms in a dataset.

    Collects every unique GPL platform referenced by any sample, fetches
    the CpG probe lists for each platform, and returns the probes that
    appear on **all** platforms.

    Args:
        dataset_metadata: Per-dataset sample-level metadata batches.
            Each sample's ``platform`` field contributes a GPL accession.

    Returns:
        A ``set`` of CpG probe IDs common to every platform found.
    """
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
