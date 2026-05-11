import inspect

import numpy as np
import pandas as pd
import pytest


class TestGetFieldValueReturnsTuple:
    def test_return_type_is_three_tuple(self):
        """Verify get_field_value returns Tuple[Optional[str], Optional[str], bool]."""
        from methylcurate.tools.geo.extract_sample_level_metadata import get_field_value

        sig = inspect.signature(get_field_value)
        annotation = str(sig.return_annotation)
        assert "tuple" in annotation
        assert "str" in annotation
        assert "bool" in annotation

    def test_missing_status_returns_none_none_false(self):
        from unittest.mock import MagicMock

        from methylcurate.tools.geo.extract_sample_level_metadata import get_field_value

        resolution = MagicMock()
        resolution.status = "missing"
        resolution.extraction = MagicMock()
        resolution.extraction.field_name = "source_name_ch1"

        extracted_value, target_field, flag = get_field_value({}, resolution)
        assert extracted_value is None
        assert target_field is None
        assert flag is False


class TestFieldCoverageNaming:
    def test_get_field_coverage_name_is_pep8(self):
        from methylcurate.tools.geo import extract_sample_level_metadata as m

        assert hasattr(m, "get_field_coverage")
        assert not hasattr(m, "get_field_Coverage")

    def test_get_df_field_coverage_name_is_pep8(self):
        from methylcurate.tools.geo import extract_sample_level_metadata as m

        assert hasattr(m, "get_df_field_coverage")
        assert not hasattr(m, "get_df_field_Coverage")


class TestGetFieldCoverage:
    def test_computes_coverage_correctly(self):
        from methylcurate.contracts.geo import GEOSampleLevelMetadata, GeoSampleLevelMetadataBatch
        from methylcurate.tools.geo.extract_sample_level_metadata import get_field_coverage

        samples = [
            GEOSampleLevelMetadata(sample_name="S1", subject_id="S1", age=45.0),
            GEOSampleLevelMetadata(sample_name="S2", subject_id="S2", age=None),
            GEOSampleLevelMetadata(sample_name="S3", subject_id="S3", age=60.0),
        ]
        batch = GeoSampleLevelMetadataBatch(accession="GSE1", samples=samples)
        coverage = get_field_coverage(batch, "age")
        assert coverage.present == 2
        assert coverage.missing == 1
        assert coverage.parse_rate == pytest.approx(2.0 / 3.0)

    def test_get_df_field_coverage_works(self):
        from methylcurate.tools.geo.extract_sample_level_metadata import get_df_field_coverage

        df = pd.DataFrame({"Sex": ["Male", np.nan, "Female", "Male"]})
        coverage = get_df_field_coverage(df, "Sex")
        assert coverage.present == 3
        assert coverage.missing == 1
        assert coverage.unique_values == 2
