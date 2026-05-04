import inspect


class TestProcessDetectionColumnsReturnType:
    def test_process_detection_columns_returns_tuple_of_df_and_artifactref(self):
        from methylcurate.tools.geo.extract_supplementary_data import (
            _process_detection_columns,
        )

        sig = inspect.signature(_process_detection_columns)
        annotation = str(sig.return_annotation)
        assert "Tuple" in annotation
        assert "DataFrame" in annotation
        assert "ArtifactRef" in annotation

    def test_process_detection_columns_alt_returns_tuple_of_df_and_artifactref(self):
        from methylcurate.tools.geo.extract_supplementary_data import (
            _process_detection_columns_alt,
        )

        sig = inspect.signature(_process_detection_columns_alt)
        annotation = str(sig.return_annotation)
        assert "Tuple" in annotation
        assert "DataFrame" in annotation
        assert "ArtifactRef" in annotation


class TestGetColumnSchemeReturnType:
    def test_get_column_scheme_returns_tuple_of_resolution_and_artifactref(self):
        from methylcurate.tools.geo.extract_supplementary_data import (
            _get_column_scheme,
        )

        sig = inspect.signature(_get_column_scheme)
        annotation = str(sig.return_annotation)
        assert "Tuple" in annotation
        assert "SampleDataResolution" in annotation
        assert "ArtifactRef" in annotation


class TestGenerateDataSamplesReturnType:
    def test_generate_data_samples_returns_tuple_of_str_and_dataframe(self):
        from methylcurate.tools.geo.extract_supplementary_data import (
            _generate_data_samples,
        )

        sig = inspect.signature(_generate_data_samples)
        annotation = str(sig.return_annotation)
        assert "Tuple" in annotation
        assert "str" in annotation
        assert "DataFrame" in annotation


class TestCheckForCpgProbesReturnType:
    def test_check_for_cpg_probes_returns_dict_of_str_to_bool(self):
        from methylcurate.tools.geo.extract_supplementary_data import (
            _check_for_cpg_probes,
        )

        sig = inspect.signature(_check_for_cpg_probes)
        annotation = str(sig.return_annotation)
        assert "Dict" in annotation
        assert "str" in annotation
        assert "bool" in annotation


class TestCreateSubjectIdMappingReturnType:
    def test_create_subject_id_mapping_returns_dataframe(self):
        from methylcurate.tools.geo.extract_supplementary_data import (
            _create_subject_id_mapping,
        )

        sig = inspect.signature(_create_subject_id_mapping)
        annotation = str(sig.return_annotation)
        assert "DataFrame" in annotation
