__all__ = ["run_all_qc"]
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any
from .data_type_conversion import convert_data_type
from .qc import handle_cpg_level_missingness, handle_sample_level_missingness, maximum_dnam_filter, interarray_correlation
from ...contracts.preprocess import PreprocessDataInput, DNAmQCInput, CpGLevelQCInput, SampleLevelQCInput, InterarrayCorrelationQCInput
from ...contracts.common import ArtifactRef
from ...utils.helper import compute_sha256, read_feather, write_feather

def run_all_qc(
        accession_code: str,
        data_path: str = None,
        processed_path: str = None,
        data_conversion_input: PreprocessDataInput = None,
        sample_level_qc_input: SampleLevelQCInput = None,
        cpg_level_qc_input: CpGLevelQCInput = None,
        dnam_qc_input: DNAmQCInput = None,
        interarray_correlation_qc_input: InterarrayCorrelationQCInput = None,
        logger: Any = None
) -> Dict[str, Any]:
    data_df = read_feather(data_path, index_name="subject_id")  # Directly set to state
    logger.info(f"Initial data shape for accession {accession_code}: {data_df.shape}")
    data_conversion_result, data_df = convert_data_type(data_conversion_input, data_df)
    sample_level_qc_result, data_df = handle_sample_level_missingness(sample_level_qc_input, data_df)
    logger.info(f"Data shape after sample level QC for accession {accession_code}: {data_df.shape}")
    dnam_qc_result, data_df = maximum_dnam_filter(dnam_qc_input, data_df)
    logger.info(f"Data shape after DNAm QC for accession {accession_code}: {data_df.shape}")
    cpg_level_qc_result, data_df = handle_cpg_level_missingness(cpg_level_qc_input, data_df)
    logger.info(f"Data shape after CpG level QC for accession {accession_code}: {data_df.shape}")
    interarray_correlation_qc_result, data_df = interarray_correlation(interarray_correlation_qc_input, data_df)
    logger.info(f"Data shape after interarray correlation QC for accession {accession_code}: {data_df.shape}")

    write_feather(data_df, processed_path, index_name="subject_id")
    artifacts = [
        ArtifactRef.model_validate({
            "accession_code": accession_code,
            "path": processed_path,
            "kind": "postqc_methylation_data",
            "sha256": compute_sha256(processed_path, is_path=True),
            "bytes": os.path.getsize(processed_path),
            "created_at": datetime.now(timezone.utc).isoformat()})
    ]
    return {
        "data_conversion_result": data_conversion_result,
        "sample_level_qc_result": sample_level_qc_result,
        "cpg_level_qc_result": cpg_level_qc_result,
        "dnam_qc_result": dnam_qc_result,
        "interarray_correlation_qc_result": interarray_correlation_qc_result,
        "artifacts": artifacts
    }
