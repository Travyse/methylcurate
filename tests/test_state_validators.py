"""Tests for state model validators."""

import pytest


class TestGeoAccessionValidator:
    def test_valid_gse_passes(self):
        from methylcurate.agent.state.models import _validate_geo_accession
        assert _validate_geo_accession("GSE12345") == "GSE12345"
        assert _validate_geo_accession("gse12345") == "gse12345"

    def test_invalid_raises(self):
        from methylcurate.agent.state.models import _validate_geo_accession
        with pytest.raises(ValueError, match="valid GEO Series"):
            _validate_geo_accession("GSM12345")

    def test_empty_string_raises(self):
        from methylcurate.agent.state.models import _validate_geo_accession
        with pytest.raises(ValueError, match="non-empty"):
            _validate_geo_accession("")

    def test_accessions_list_validator(self):
        from methylcurate.agent.state.models import _validate_accessions_list
        result = _validate_accessions_list(["GSE1", "GSE2"])
        assert result == ["GSE1", "GSE2"]

    def test_accessions_list_rejects_invalid(self):
        from methylcurate.agent.state.models import _validate_accessions_list
        with pytest.raises(ValueError):
            _validate_accessions_list(["GSM1"])
        with pytest.raises(ValueError):
            _validate_accessions_list([])
