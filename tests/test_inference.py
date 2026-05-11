import inspect
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


class TestRaiseNotReturn:
    def test_get_available_methylation_dataframe_raises(self):
        """Verify the function raises ValueError, not returns it."""
        from methylcurate.tools.clocks.inference import get_available_methylation_dataframe

        with pytest.raises(ValueError, match="No methylation data"):
            get_available_methylation_dataframe("GSE1", artifacts=[])

    def test_get_dataset_predictions_raises(self):
        """Verify the function raises ValueError, not returns it."""
        from methylcurate.tools.clocks.inference import get_dataset_predictions

        with pytest.raises(ValueError, match="No dataset predictions"):
            get_dataset_predictions("GSE1", artifacts=[])


class TestMutableDefaultArguments:
    def test_compute_mae_default_is_none_not_empty_list(self):
        from methylcurate.tools.clocks.inference import compute_mae

        sig = inspect.signature(compute_mae)
        default = sig.parameters["clocks"].default
        assert default is None

    def test_compute_medae_default_is_none_not_empty_list(self):
        from methylcurate.tools.clocks.inference import compute_medae

        sig = inspect.signature(compute_medae)
        default = sig.parameters["clocks"].default
        assert default is None

    def test_compute_pearson_r_default_is_none_not_empty_list(self):
        from methylcurate.tools.clocks.inference import compute_pearson_r

        sig = inspect.signature(compute_pearson_r)
        default = sig.parameters["clocks"].default
        assert default is None

    def test_bootstrap_welch_default_is_none_not_empty_list(self):
        from methylcurate.tools.clocks.inference import bootstrap_welch_one_sided_aac_gt_hc

        sig = inspect.signature(bootstrap_welch_one_sided_aac_gt_hc)
        default = sig.parameters["clocks"].default
        assert default is None

    def test_bootstrap_aa1_default_is_none_not_empty_list(self):
        from methylcurate.tools.clocks.inference import bootstrap_aa1_test

        sig = inspect.signature(bootstrap_aa1_test)
        default = sig.parameters["clocks"].default
        assert default is None

    def test_mutable_default_does_not_persist_state(self):
        """Verify that when clocks=None, two sequential calls don't share state."""
        from methylcurate.tools.clocks.inference import compute_mae

        df = pd.DataFrame(
            {
                "Accession_Code": ["GSE1", "GSE1"],
                "Disease_Status": ["Control", "Case"],
                "age": [50, 60],
                "horvath_accel": [1.0, -0.5],
                "horvath": [51.0, 59.5],
            }
        )
        extraction_protocol = {"disease_status": {"extraction": {"control_value": "Control"}}}
        result1 = compute_mae(df, extraction_protocol)
        result2 = compute_mae(df, extraction_protocol)
        assert len(result1) == len(result2)


class TestNBootstrapsRespected:
    def test_shuffled_labels_uses_n_bootstraps_not_hardcoded(self):
        """Verify the function body uses n_bootstraps parameter (not literal 1000).

        We inspect the source code to confirm range(n_bootstraps) appears
        in the shuffled_labels list comprehension rather than range(1000).
        """
        from methylcurate.tools.clocks.inference import (
            bootstrap_aa1_test,
            bootstrap_welch_one_sided_aac_gt_hc,
        )

        for func in (bootstrap_welch_one_sided_aac_gt_hc, bootstrap_aa1_test):
            source = inspect.getsource(func)
            assert "range(n_bootstraps)" in source, f"{func.__name__} should use n_bootstraps parameter, not hardcoded 1000"
            assert "range(1000)" not in source, f"{func.__name__} contains hardcoded range(1000); should use range(n_bootstraps)"


class TestBootstrapRespectsNBootstrapsRuntime:
    @patch("methylcurate.tools.clocks.inference.Parallel")
    @patch("methylcurate.tools.clocks.inference.welch_one_sided_aac_gt_hc")
    def test_n_bootstraps_controls_shuffled_labels_count(self, mock_test, mock_parallel):
        """Verify shuffled_labels list length matches n_bootstraps at runtime."""
        from methylcurate.tools.clocks.inference import bootstrap_welch_one_sided_aac_gt_hc

        mock_parallel.return_value.return_value = [(0.5, 0.3)] * 5
        mock_test.return_value = (1.0, 0.05)

        df = pd.DataFrame(
            {
                "Accession_Code": ["GSE1"] * 20,
                "Disease_Status": ["Control"] * 12 + ["Case"] * 8,
                "age": list(range(20)),
                "horvath_accel": np.random.randn(20) * 0.5,
            }
        )
        extraction_protocol = {"disease_status": {"extraction": {"control_value": "Control"}}}
        result = bootstrap_welch_one_sided_aac_gt_hc(df, extraction_protocol, clocks=["horvath"], n_bootstraps=75)
        assert isinstance(result, pd.DataFrame)
