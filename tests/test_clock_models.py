"""Tests for clock_models.py shared utilities and class behavior."""

import pandas as pd
import numpy as np
import pytest


class TestImputeClockData:
    def test_aligns_and_imputes_missing_columns(self):
        from methylcurate.tools.clocks.clock_models import _impute_clock_data

        required = ["cg001", "cg002", "cg003"]
        dnam = pd.DataFrame(
            {"cg001": [0.5, 0.6], "cg003": [0.8, 0.9]},
            index=["S1", "S2"],
        )
        default_imp = pd.DataFrame(
            {"cg001": [0.55], "cg002": [0.70], "cg003": [0.85]},
            index=["ref"],
        )
        result = _impute_clock_data(required, dnam, default_imp)
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        assert not np.isnan(result).any()
        assert result[0, 1] == pytest.approx(0.70)

    def test_raises_when_no_imputation_available(self):
        from methylcurate.tools.clocks.clock_models import _impute_clock_data

        required = ["cg001", "cg999"]
        dnam = pd.DataFrame({"cg001": [0.5]}, index=["S1"])
        with pytest.raises(ValueError, match="no imputation reference"):
            _impute_clock_data(required, dnam, None)

    def test_user_imputation_overrides_default(self):
        from methylcurate.tools.clocks.clock_models import _impute_clock_data

        required = ["cg001", "cg002"]
        dnam = pd.DataFrame({"cg001": [0.5]}, index=["S1"])
        default_imp = pd.DataFrame(
            {"cg001": [0.1], "cg002": [0.2]}, index=["ref"],
        )
        user_imp = pd.DataFrame(
            {"cg001": [0.9], "cg002": [0.8]}, index=["ref"],
        )
        result = _impute_clock_data(required, dnam, default_imp,
                                     user_imputation=user_imp)
        assert result[0, 1] == pytest.approx(0.8)
