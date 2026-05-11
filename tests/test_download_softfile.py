import inspect
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def mock_gse():
    gse = MagicMock()
    gse.gsms = {}
    gse.metadata = {"title": ["Test"], "summary": [""], "overall_design": [""]}
    return gse


class TestCacheMethylationData:
    def test_returns_dataframe(self, mock_gse):
        from methylcurate.tools.geo.download_softfile import _cache_methylation_data

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _cache_methylation_data(mock_gse, output_dir=tmpdir)
            assert isinstance(result, pd.DataFrame)


class TestDownloadReturnType:
    def test_return_type_is_artifactref_not_path(self):
        from methylcurate.tools.geo.download_softfile import download

        sig = inspect.signature(download)
        annotation_str = str(sig.return_annotation)
        assert "ArtifactRef" in annotation_str
        assert "Path" not in annotation_str


class TestParallelDownloadsReturnType:
    def test_return_type_is_dict_of_list_artifactref(self):
        from methylcurate.tools.geo.download_softfile import parallel_downloads

        sig = inspect.signature(parallel_downloads)
        annotation = str(sig.return_annotation)
        assert "ArtifactRef" in annotation
        assert "dict" in annotation
        assert "list" in annotation
        assert "Path" not in annotation


class TestDownloadGeoDatasetErrorHandling:
    @patch("methylcurate.tools.geo.download_softfile.os.makedirs")
    @patch("methylcurate.tools.geo.download_softfile.os.listdir")
    @patch("methylcurate.tools.geo.download_softfile.os.path.exists")
    @patch("methylcurate.tools.geo.download_softfile.GEOparse")
    @patch("methylcurate.tools.geo.download_softfile._cache_metadata")
    @patch("methylcurate.tools.geo.download_softfile._cache_methylation_data")
    @patch("methylcurate.tools.geo.download_softfile._check_supplementary_files")
    @patch("methylcurate.tools.geo.download_softfile.json.dump")
    @patch("methylcurate.tools.geo.download_softfile.write_feather")
    def test_error_msg_initialized_when_all_retries_fail(
        self, mock_wf, mock_jd, mock_cs, mock_cmd, mock_cm, mock_geoparse, mock_exists, mock_listdir, mock_makedirs
    ):
        from methylcurate.tools.geo.download_softfile import _download_geo_dataset

        mock_exists.side_effect = lambda p: not p.endswith("_family.soft.gz")
        mock_listdir.return_value = []
        mock_geoparse.get_GEO.side_effect = ConnectionError("network down")
        mock_cs.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts, supp, result = _download_geo_dataset("GSE12345", tmpdir)
            assert result.status == "failed"
            assert result.error is not None
            assert isinstance(result.error, str)


class TestDeadCodeRemoved:
    def test_download_function_has_no_duplicate_cache_check(self):
        from methylcurate.tools.geo.download_softfile import download

        source = inspect.getsource(download)
        lines = [line for line in source.split("\n") if "if os.path.exists(cache_download_path)" in line]
        assert len(lines) <= 1, f"Found {len(lines)} duplicate cache checks"
