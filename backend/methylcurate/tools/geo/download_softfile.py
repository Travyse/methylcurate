__all__ = ["download_geo_datasets", "_family_soft_path", "_check_supplementary_files", "parallel_downloads"]
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

import GEOparse
import pandas as pd
import requests
from joblib import Parallel, delayed
from tqdm import tqdm

from ...agent.state.models import GEOIngestionConfig
from ...contracts.common import ArtifactRef
from ...contracts.geo import GEODownloadBatchInput, GEODownloadBatchResult, GEODownloadResult
from ...utils.exception_handling import classify_geo_error
from ...utils.helper import compute_sha256, write_feather
from .extract_sample_level_metadata import _merge_to_dataframe


def _family_soft_path(output_dir: str, accession: str) -> str:
    """Build the expected path for a GEO family SOFT file.

    Args:
        output_dir: Directory where the SOFT file should reside.
        accession: GEO accession code (e.g. "GSE12345").

    Returns:
        Full path to the expected family SOFT (.soft.gz) file.
    """
    return os.path.join(output_dir, f"{accession}_family.soft.gz")


def _cache_metadata(gse: Any) -> dict[str, Any]:
    """Extract and cache per-sample and dataset-level metadata from a GEOparse GSE object.

    Collects sample metadata for each GSM, top-level dataset fields (title, summary,
    overall_design), and any supplementary-file metadata keys.

    Args:
        gse: A GEOparse GSE object.

    Returns:
        A dict with keys ``sample_metadata`` and ``dataset_metadata``.
    """
    sample_metadata = {gsm_name: gsm.metadata for gsm_name, gsm in list(gse.gsms.items())}
    metadata = {
        "sample_metadata": sample_metadata,
        "dataset_metadata": {
            "title": gse.metadata.get("title", [""])[0],
            "summary": gse.metadata.get("summary", [""])[0],
            "overall_design": gse.metadata.get("overall_design", [""])[0],
        },
    }
    for k in [k for k in gse.metadata.keys() if "supplement" in k.lower()]:
        metadata["dataset_metadata"][k] = gse.metadata.get(k, None)
    return metadata


def _cache_methylation_data(gse: Any, output_dir: str) -> pd.DataFrame:
    """Extract and cache methylation data from a GEOparse GSE object.

    Iterates over each GSM sample, filters rows by detection p-value <= 0.05,
    and collects ``VALUE`` / ``ID_REF`` columns into a merged DataFrame.

    Args:
        gse: A GEOparse GSE object containing GSM samples with methylation tables.
        output_dir: Directory for caching (unused by this function but passed
            for interface compatibility).

    Returns:
        Merged methylation matrix as a DataFrame keyed by sample (``Sample``).
        Returns an empty DataFrame if no valid methylation data is found.
    """
    methylation_rows = []
    methylation_col_names = []

    for gsm_name, gsm in gse.gsms.items():
        sample_data = gsm.table
        if sample_data is None or len(sample_data.columns.tolist()) < 1:
            continue

        detection_cols = [x for x in sample_data.columns if "detection" in x.lower()]
        if len(detection_cols) > 0:
            sample_data = sample_data[sample_data[detection_cols[0]] <= 0.05]
        else:
            print(f"\nNo detection columns found: {sample_data.columns.tolist()}")

        if sample_data.empty or len(sample_data.columns.tolist()) < 1:
            continue

        methylation_rows.append([gsm_name] + sample_data["VALUE"].tolist())
        methylation_col_names.append(["Sample"] + sample_data["ID_REF"].tolist())

    if len(methylation_rows) == 0:
        return pd.DataFrame()  # Return empty DataFrame if no valid methylation data found

    return _merge_to_dataframe(methylation_rows, methylation_col_names, index_col="Sample")


def _download_geo_dataset(accession: str, output_dir: str):
    """Download a single GEO dataset and extract metadata and methylation data.

    Checks a shared cache directory first. On cache hit, returns existing
    artifacts for the SOFT file, metadata JSON, and pre-QC methylation matrix.
    On cache miss, retrieves the dataset via GEOparse, extracts per-sample
    metadata and methylation data, writes cache files, and collects
    supplementary-file URLs. Retries up to 3 times on download errors.

    Args:
        accession: GEO accession code (e.g. ``"GSE12345"``).
        output_dir: Dataset-specific output directory (used to derive the
            shared cache directory).

    Returns:
        A tuple of ``(artifacts, supplementary_files, result)`` where:

        * *artifacts*: list of :class:`~methylcurate.contracts.common.ArtifactRef`
          for the SOFT file, metadata cache, and methylation data.
        * *supplementary_files*: dict mapping accession to a list of
          supplementary file URLs (empty dict on failure).
        * *result*: :class:`~methylcurate.contracts.geo.GEODownloadResult`
          with per-accession status and any warnings.

    Raises:
        No exceptions are propagated; errors are captured in the returned
        ``GEODownloadResult`` with ``status="failed"``.
    """
    artifacts: list[ArtifactRef] = []
    os.makedirs(output_dir, exist_ok=True)

    # Get parent directory of output_dir
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(output_dir)))
    cache_output_dir = os.path.join(parent_dir, "cache")
    os.makedirs(cache_output_dir, exist_ok=True)
    cached_soft_path = os.path.join(cache_output_dir, f"{accession}_family.soft.gz")
    print(f"Checking cache for {accession} at {cached_soft_path}...")
    if os.path.exists(cached_soft_path):
        print(f"Soft file already exists for {accession} at {cached_soft_path}, skipping download.")
        # shutil.copy(cached_soft_path, soft_path) TODO: Maybe Remove
        artifacts.append(
            ArtifactRef.model_validate(
                {
                    "path": cached_soft_path,
                    "kind": "soft_file",
                    "accession_code": accession,
                    "sha256": compute_sha256(cached_soft_path, is_path=True),
                    "bytes": os.path.getsize(cached_soft_path),
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
        )
        metadata_path = os.path.join(cache_output_dir, f"{accession}_metadata.json")
        methylation_data_path = os.path.join(cache_output_dir, f"{accession}_preqc_methylation_matrix.feather")
        if any([not os.path.exists(p) for p in [metadata_path, methylation_data_path]]):
            g = GEOparse.get_GEO(filepath=cached_soft_path, silent=True)
            metadata = _cache_metadata(g)
            methylation_data = _cache_methylation_data(g, cache_output_dir)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            write_feather(methylation_data, methylation_data_path, index_name="subject_id")
            supplementary_files = {accession: _check_supplementary_files(g)}
        else:
            metadata = json.load(open(metadata_path))
            possible_supplementary_keys = [k for k in metadata["dataset_metadata"].keys() if "supplement" in k.lower()]
            supplementary_files = set()
            for supplementary_key in possible_supplementary_keys:
                if isinstance(metadata["dataset_metadata"].get(supplementary_key), list):
                    supplementary_files.update(set(metadata["dataset_metadata"][supplementary_key]))
                else:
                    supplementary_files.add(metadata["dataset_metadata"][supplementary_key])
            supplementary_files = {accession: sorted(list(supplementary_files))}

        artifacts.extend(
            [
                ArtifactRef.model_validate(
                    {
                        "path": metadata_path,
                        "kind": "metadata_cache",
                        "accession_code": accession,
                        "sha256": compute_sha256(metadata_path, is_path=True),
                        "bytes": os.path.getsize(metadata_path),
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                ),
                ArtifactRef.model_validate(
                    {
                        "path": methylation_data_path,
                        "kind": "preqc_methylation_data",
                        "accession_code": accession,
                        "sha256": compute_sha256(methylation_data_path, is_path=True),
                        "bytes": os.path.getsize(methylation_data_path),
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                ),
            ]
        )

        return (
            artifacts,
            supplementary_files,
            GEODownloadResult(
                accession=accession,
                artifact=artifacts[0],
                status="success",
                error=None,
                warnings=[f"Soft file already existed at {cached_soft_path}, skipping download."],
            ),
        )

    # Download
    for _ in range(3):
        try:
            gse = GEOparse.get_GEO(accession, destdir=cache_output_dir, silent=True)

            # Verify expected output (best-effort; GEOparse may still succeed with a different artifact set)
            if os.path.exists(cached_soft_path):
                artifacts.append(
                    ArtifactRef.model_validate(
                        {
                            "path": cached_soft_path,
                            "kind": "soft_file",
                            "accession_code": accession,
                            "sha256": compute_sha256(cached_soft_path, is_path=True),
                            "bytes": os.path.getsize(cached_soft_path),
                            "created_at": datetime.now(UTC).isoformat(),
                        }
                    )
                )
                metadata = _cache_metadata(gse)
                methylation_data = _cache_methylation_data(gse, cache_output_dir)
                metadata_path = os.path.join(cache_output_dir, f"{accession}_metadata.json")
                methylation_data_path = os.path.join(cache_output_dir, f"{accession}_preqc_methylation_matrix.feather")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)

                write_feather(methylation_data, methylation_data_path, index_name="subject_id")
                supplementary_files = {accession: _check_supplementary_files(gse)}

                artifacts.extend(
                    [
                        ArtifactRef.model_validate(
                            {
                                "path": metadata_path,
                                "kind": "metadata_cache",
                                "accession_code": accession,
                                "sha256": compute_sha256(metadata_path, is_path=True),
                                "bytes": os.path.getsize(metadata_path),
                                "created_at": datetime.now(UTC).isoformat(),
                            }
                        ),
                        ArtifactRef.model_validate(
                            {
                                "path": methylation_data_path,
                                "kind": "preqc_methylation_data",
                                "accession_code": accession,
                                "sha256": compute_sha256(methylation_data_path, is_path=True),
                                "bytes": os.path.getsize(methylation_data_path),
                                "created_at": datetime.now(UTC).isoformat(),
                            }
                        ),
                    ]
                )

                return (
                    artifacts,
                    supplementary_files,
                    GEODownloadResult(
                        accession=accession, artifact=artifacts[0], status="success", error=None, warnings=[]
                    ),
                )

        except Exception as e:
            error_msg = classify_geo_error(e)[1]
            print(f"Raw error message: {str(e)}")
            print(f"Error downloading {accession}: {error_msg}")
            continue
    else:
        error_msg = "Download failed after all retries"
    return (
        artifacts,
        {},
        GEODownloadResult(
            accession=accession,
            artifact=None,
            status="failed",
            error=error_msg,
            warnings=[],
        ),
    )


def download_geo_datasets(
    config: GEOIngestionConfig, batch: GEODownloadBatchInput
) -> tuple[list[Any], list[Any], GEODownloadBatchResult]:
    """
    Tool-friendly batch downloader:
      - does not assume a single output_dir for all accessions (each input can specify output_root)
      - returns structured per-accession results
      - continues on errors (partial success allowed)
    """
    # TODO: Restructure this to download in batches of 5 or so (I can configure this)
    results: list[Any] = Parallel(n_jobs=-1)(
        delayed(_download_geo_dataset)(item.accession, os.path.join(config.output_root, item.accession))
        for item in batch.geo_downloads
    )

    download_results = [x[2] for x in results]
    supplementary_files = [x[1] for x in results]
    artifact_results = [item for x in results for item in x[0]]

    # Batch summary
    n_total = len(results)
    succeeded = [r.accession for r in download_results if r.status == "success"]
    skipped = [r.accession for r in download_results if r.status == "skipped"]
    failed = [r.accession for r in download_results if r.status == "failed"]

    n_success = len(succeeded)
    n_skipped = len(skipped)
    n_failed = len(failed)

    if n_failed == 0 and n_success + n_skipped == n_total:
        batch_status: Literal["success", "partial", "failed"] = "success"
    elif n_failed == n_total:
        batch_status = "failed"
    else:
        batch_status = "partial"

    warnings: list[str] = []
    if batch_status == "partial":
        warnings.append("Some accessions failed; inspect per-accession results.")

    return (
        artifact_results,
        supplementary_files,
        GEODownloadBatchResult(
            results=download_results,
            batch_status=batch_status,
            warnings=warnings,
        ),
    )


def _check_supplementary_files(gse: Any):
    """Collect supplementary file URLs from a GEO dataset's metadata.

    Scans dataset-level metadata keys containing ``"supplement"`` (case-
    insensitive) and gathers all associated URLs.

    Args:
        gse: A GEOparse GSE object whose metadata may contain supplementary
            file references.

    Returns:
        A sorted list of unique supplementary file URL strings.
    """
    supplementary_files = set()
    possible_supplementary_keys = [k for k in gse.metadata.keys() if "supplement" in k.lower()]
    for supplementary_key in possible_supplementary_keys:
        if isinstance(gse.metadata.get(supplementary_key), list):
            supplementary_files.update(set(gse.metadata[supplementary_key]))
        else:
            supplementary_files.add(gse.metadata[supplementary_key])
    return sorted(list(supplementary_files))


def ftp_to_https(url: str) -> str:
    """Convert an FTP URL to its HTTPS equivalent.

    NCBI supports the same resource path over HTTPS, so FTP URLs are
    rewritten accordingly. Non-FTP URLs are returned unchanged.

    Args:
        url: The URL string to convert.

    Returns:
        The HTTPS-equivalent URL if the scheme was ``ftp``, otherwise the
        original URL.
    """
    u = urlparse(url)
    if u.scheme != "ftp":
        return url
    # NCBI supports same path over https
    return urlunparse(("https", u.netloc, u.path, "", "", ""))


def download(
    accession_code: str, url: str, destdir: str | Path = ".", chunk_size: int = 1024 * 1024, max_retries: int = 3
) -> ArtifactRef:
    """Download a single supplementary file with caching and retry logic.

    Checks a shared cache directory first, returning the existing artifact
    on hit. On miss, downloads the file via streaming HTTP, shows a progress
    bar, and retries up to ``max_retries`` times on failure.

    Args:
        accession_code: GEO accession code associated with the file.
        url: The URL to download from (FTP URLs are automatically converted
            to HTTPS).
        destdir: Destination directory (used to derive the shared cache
            directory). Defaults to ``"."``.
        chunk_size: Stream read chunk size in bytes. Defaults to
            ``1024 * 1024`` (1 MiB).
        max_retries: Maximum download attempts before raising an error.
            Defaults to ``3``.

    Returns:
        An :class:`~methylcurate.contracts.common.ArtifactRef` for the
        downloaded file with kind ``"supplementary_file_methylation_data"``.

    Raises:
        RuntimeError: If the download fails after all retry attempts.
    """
    # Add supplementary files info
    print(f"Downloading supplementary file for {accession_code} from {url}...")
    destdir = Path(destdir)
    destdir.mkdir(parents=True, exist_ok=True)

    https_url = ftp_to_https(url)
    filename = Path(urlparse(https_url).path).name

    # Get parent directory of output_dir
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(destdir)))
    cache_download_path = os.path.join(parent_dir, "cache", filename)
    print(f"Checking cache for {accession_code} at {cache_download_path}...")
    if os.path.exists(cache_download_path):
        print(f"File already exists for {accession_code} at {cache_download_path}, skipping download.")
        # shutil.copy(cache_download_path, str(cache_download_path))
        return ArtifactRef.model_validate(
            {
                "path": str(cache_download_path),
                "kind": "supplementary_file_methylation_data",
                "accession_code": accession_code,
                "sha256": compute_sha256(cache_download_path, is_path=True),
                "bytes": os.path.getsize(cache_download_path),
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

    for _attempt in range(1, max_retries + 1):
        try:
            with requests.get(https_url, stream=True, timeout=(10, 60)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)

                with (
                    open(cache_download_path, "wb") as f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=f"{accession_code}:{filename}",
                        leave=True,
                    ) as bar,
                ):
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bar.update(len(chunk))

                print(f"Download completed: {cache_download_path}")

                artifact = ArtifactRef.model_validate(
                    {
                        "path": str(cache_download_path),
                        "kind": "supplementary_file_methylation_data",
                        "accession_code": accession_code,
                        "sha256": compute_sha256(cache_download_path, is_path=True),
                        "bytes": os.path.getsize(cache_download_path),
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                )
                return artifact
        except Exception as e:
            print(f"[{accession_code}] Non-retryable error for {filename}: {type(e).__name__}: {e}")
            if cache_download_path.exists():  # type: ignore
                try:
                    cache_download_path.unlink()  # type: ignore
                except Exception:
                    pass
            time.sleep(3)

    raise RuntimeError(f"Failed to download {filename} for {accession_code} after {max_retries} attempts.")


def parallel_downloads(
    accession_code: str, urls: list[str], destdir: str, chunk_size: int = 1024 * 1024
) -> dict[str, list[ArtifactRef]]:
    """Download multiple supplementary files in parallel for a single accession.

    Dispatches each URL to :func:`download` using ``joblib.Parallel`` with
    all available CPU cores.

    Args:
        accession_code: GEO accession code associated with the files.
        urls: List of file URLs to download.
        destdir: Destination directory for downloads.
        chunk_size: Stream read chunk size in bytes. Defaults to
            ``1024 * 1024`` (1 MiB).

    Returns:
        A dict with a single key ``"artifacts"`` mapping to the list of
        :class:`~methylcurate.contracts.common.ArtifactRef` for each
        downloaded file.

    Raises:
        RuntimeError: If any individual download fails after all retries.
    """
    print(f"Starting parallel downloads for {accession_code} from URLs: {urls}")
    artifacts = Parallel(n_jobs=-1)(delayed(download)(accession_code, url, destdir, chunk_size) for url in urls)
    return {"artifacts": artifacts}
