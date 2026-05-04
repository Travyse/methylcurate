__all__ = ["run_all_qc"]
import os
from datetime import UTC, datetime
from typing import Any

from ...contracts.common import ArtifactRef
from ...contracts.qc import (
    CpGLevelQCInput,
    DNAmQCInput,
    InterarrayCorrelationQCInput,
    PreprocessDataInput,
    SampleLevelQCInput,
)
from ...utils.helper import compute_sha256, read_feather, write_feather
from .data_type_conversion import convert_data_type
from .qc import (
    handle_cpg_level_missingness,
    handle_sample_level_missingness,
    interarray_correlation,
    maximum_dnam_filter,
)


def run_all_qc(
    accession_code: str,
    data_path: str | None = None,
    processed_path: str | None = None,
    data_conversion_input: PreprocessDataInput | None = None,
    sample_level_qc_input: SampleLevelQCInput | None = None,
    cpg_level_qc_input: CpGLevelQCInput | None = None,
    dnam_qc_input: DNAmQCInput | None = None,
    interarray_correlation_qc_input: InterarrayCorrelationQCInput | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Run the full quality-control pipeline on a single methylation dataset.

    This is the top-level orchestrator for pre-benchmarking QC.  It
    executes every step in order:

    1. **Data-type conversion** — optionally converts between
       beta-value and M-value representations.
    2. **Sample-level missingness** — drops samples with excessive
       missing CpG probes.
    3. **Maximum DNAm filter** — retains only samples whose maximum
       DNAm value exceeds the configured cutoff.
    4. **CpG-level missingness** — drops unreliable probes and
       imputes the remaining gaps with a KNN or simple imputer.
    5. **Inter-array correlation** — drops outlier samples whose
       mean correlation to the rest of the cohort is too low.

    After filtering, the cleaned matrix is written to
    *processed_path* as a Feather file (indexed by ``subject_id``).
    An artifact reference recording the file path, SHA-256 hash,
    size, and creation timestamp is included in the returned dict.

    Args:
        accession_code: GEO accession code (e.g. ``"GSE12345"``)
            used for logging and artifact identity.
        data_path: Path to the raw input Feather file.
        processed_path: Where the post-QC Feather file is written.
        data_conversion_input: Data-type conversion specification.
        sample_level_qc_input: Sample-level missingness QC input.
        cpg_level_qc_input: CpG-level QC input (missingness cutoff
            and imputation strategy/model).
        dnam_qc_input: Maximum DNAm filter QC input.
        interarray_correlation_qc_input: Inter-array correlation QC
            input.
        logger: Logger instance; ``info`` is called at each stage.

    Returns:
        A dictionary with keys:

        - ``"data_conversion_result"``
        - ``"sample_level_qc_result"``
        - ``"cpg_level_qc_result"``
        - ``"dnam_qc_result"``
        - ``"interarray_correlation_qc_result"``
        - ``"artifacts"`` — a list of ``ArtifactRef`` objects
          describing the written QC output file.
    """
    data_df = read_feather(data_path, index_name="subject_id")  # type: ignore[arg-type]
    logger.info(f"Initial data shape for accession {accession_code}: {data_df.shape}")
    data_conversion_result, data_df = convert_data_type(data_conversion_input, data_df)  # type: ignore[arg-type]
    sample_level_qc_result, data_df = handle_sample_level_missingness(sample_level_qc_input, data_df)  # type: ignore[arg-type]
    logger.info(f"Data shape after sample level QC for accession {accession_code}: {data_df.shape}")
    dnam_qc_result, data_df = maximum_dnam_filter(dnam_qc_input, data_df)  # type: ignore[arg-type]
    logger.info(f"Data shape after DNAm QC for accession {accession_code}: {data_df.shape}")
    cpg_level_qc_result, data_df = handle_cpg_level_missingness(cpg_level_qc_input, data_df)  # type: ignore[arg-type]
    logger.info(f"Data shape after CpG level QC for accession {accession_code}: {data_df.shape}")
    interarray_correlation_qc_result, data_df = interarray_correlation(interarray_correlation_qc_input, data_df)  # type: ignore[arg-type]
    logger.info(f"Data shape after interarray correlation QC for accession {accession_code}: {data_df.shape}")

    write_feather(data_df, processed_path, index_name="subject_id")  # type: ignore[arg-type]
    artifacts = [
        ArtifactRef.model_validate(
            {
                "accession_code": accession_code,
                "path": processed_path,
                "kind": "postqc_methylation_data",
                "sha256": compute_sha256(processed_path, is_path=True),  # type: ignore[arg-type]
                "bytes": os.path.getsize(processed_path),  # type: ignore[arg-type]
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
    ]
    return {
        "data_conversion_result": data_conversion_result,
        "sample_level_qc_result": sample_level_qc_result,
        "cpg_level_qc_result": cpg_level_qc_result,
        "dnam_qc_result": dnam_qc_result,
        "interarray_correlation_qc_result": interarray_correlation_qc_result,
        "artifacts": artifacts,
    }
