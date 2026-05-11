import inspect

import pandas as pd


class TestQcParamNames:
    def test_handle_sample_level_missingness_param_is_qc_input(self):
        from methylcurate.tools.qc.qc import handle_sample_level_missingness

        sig = inspect.signature(handle_sample_level_missingness)
        assert "qc_input" in sig.parameters
        assert "state" not in sig.parameters

    def test_handle_sample_level_missingness_param_type_is_not_any(self):
        from methylcurate.tools.qc.qc import handle_sample_level_missingness

        sig = inspect.signature(handle_sample_level_missingness)
        annotation = str(sig.parameters["qc_input"].annotation)
        assert "SampleLevelQCInput" in annotation
        assert "Any" not in annotation

    def test_handle_cpg_level_missingness_param_is_qc_input(self):
        from methylcurate.tools.qc.qc import handle_cpg_level_missingness

        sig = inspect.signature(handle_cpg_level_missingness)
        assert "qc_input" in sig.parameters
        assert "state" not in sig.parameters

    def test_maximum_dnam_filter_param_is_qc_input(self):
        from methylcurate.tools.qc.qc import maximum_dnam_filter

        sig = inspect.signature(maximum_dnam_filter)
        assert "qc_input" in sig.parameters
        assert "state" not in sig.parameters

    def test_interarray_correlation_param_is_qc_input(self):
        from methylcurate.tools.qc.qc import interarray_correlation

        sig = inspect.signature(interarray_correlation)
        assert "qc_input" in sig.parameters
        assert "state" not in sig.parameters


class TestEmptyDataGuards:
    def test_handle_cpg_level_empty_returns_gracefully(self):
        from methylcurate.contracts.qc import CpGLevelQCInput, ImputationInput
        from methylcurate.tools.qc.qc import handle_cpg_level_missingness

        empty_df = pd.DataFrame()
        qc_input = CpGLevelQCInput(
            missing_cutoff=0.2,
            imputation_strategy=ImputationInput(strategy="whole"),
        )
        result, filtered_df = handle_cpg_level_missingness(qc_input, empty_df)
        assert result.removed_cpgs == []
        assert filtered_df.empty

    def test_maximum_dnam_filter_empty_returns_gracefully(self):
        from methylcurate.contracts.qc import DNAmQCInput
        from methylcurate.tools.qc.qc import maximum_dnam_filter

        empty_df = pd.DataFrame()
        qc_input = DNAmQCInput(dnam_cutoff=0.96)
        result, filtered_df = maximum_dnam_filter(qc_input, empty_df)
        assert result.removed_samples == []
        assert filtered_df.empty

    def test_interarray_correlation_empty_returns_gracefully(self):
        from methylcurate.contracts.qc import InterarrayCorrelationQCInput
        from methylcurate.tools.qc.qc import interarray_correlation

        empty_df = pd.DataFrame()
        qc_input = InterarrayCorrelationQCInput(correlation_cutoff=0.9)
        result, filtered_df = interarray_correlation(qc_input, empty_df)
        assert result.removed_samples == []
        assert filtered_df.empty


class TestQcWorkflowContractImports:
    def test_run_all_qc_imports_from_contracts_qc_not_preprocess(self):
        """Verify the workflow module imports from contracts.qc (not contracts.preprocess)."""
        import methylcurate.tools.qc.workflow as mod

        source = inspect.getsource(mod)
        assert "contracts.qc" in source
        assert "contracts.preprocess" not in source
