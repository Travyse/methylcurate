import inspect

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
