import inspect


class TestGetCustomModelsReturnsFourTuple:
    def test_return_type_is_a_tuple(self):
        """_get_custom_models should return a 4-tuple (the annotation may be absent,
        but callers unpack 4 values). Test that it returns exactly 4 items."""
        from unittest.mock import MagicMock, patch

        from methylcurate.contracts.geo import GEOMetadataExtractionInput
        from methylcurate.tools.geo.metadata_column_extraction import _get_custom_models

        fake_input = MagicMock(spec=GEOMetadataExtractionInput)
        fake_input.title = ["test"]
        fake_input.source_name_ch1 = ["S1"]
        fake_input.description = [""]
        fake_input.characteristics_ch1 = [{"sex": "male"}]
        fake_input.platform_id = ["GPL13534"]
        type(fake_input).accession_code = "GSE00000"
        type(fake_input).artifact = None

        with patch("methylcurate.tools.geo.metadata_column_extraction.build_dynamic_result_model") as mock_build:
            mock_build.return_value = (MagicMock(), MagicMock(), MagicMock())
            with patch("methylcurate.tools.geo.metadata_column_extraction.get_args") as mock_args:
                mock_args.return_value = ["age", "sex", "tissue", "disease_status"]
                result = _get_custom_models(fake_input)
                assert len(result) == 4


class TestExtractAllColumnsIsAsync:
    def test_extract_all_columns_is_async(self):
        from methylcurate.tools.geo.metadata_column_extraction import _extract_all_columns

        assert inspect.iscoroutinefunction(_extract_all_columns)
