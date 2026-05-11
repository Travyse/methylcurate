from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest


def _write_yaml(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    tmp.write(content)
    tmp.close()
    return tmp.name


class TestQcConfigFromYaml:
    def test_nonexistent_file_returns_none(self):
        from methylcurate.contracts.qc import QcConfig

        assert QcConfig.from_yaml("/nonexistent/path/qc_config.yml") is None

    def test_empty_config_returns_none_fields(self):
        from methylcurate.contracts.qc import QcConfig

        path = _write_yaml("")
        try:
            cfg = QcConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.dnam_cutoff is None
            assert cfg.sample_level_missing_cutoff is None
            assert cfg.cpg_level_missing_cutoff is None
            assert cfg.correlation_cutoff is None
        finally:
            os.unlink(path)

    def test_all_fields_set(self):
        from methylcurate.contracts.qc import QcConfig

        path = _write_yaml("dnam_cutoff: 0.95\nsample_level_missing_cutoff: 0.05\ncpg_level_missing_cutoff: 0.15\ncorrelation_cutoff: 0.85\n")
        try:
            cfg = QcConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.dnam_cutoff == 0.95
            assert cfg.sample_level_missing_cutoff == 0.05
            assert cfg.cpg_level_missing_cutoff == 0.15
            assert cfg.correlation_cutoff == 0.85
        finally:
            os.unlink(path)

    def test_partial_config_only_dnam(self):
        from methylcurate.contracts.qc import QcConfig

        path = _write_yaml("dnam_cutoff: 0.90\n")
        try:
            cfg = QcConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.dnam_cutoff == 0.90
            assert cfg.sample_level_missing_cutoff is None
            assert cfg.cpg_level_missing_cutoff is None
            assert cfg.correlation_cutoff is None
        finally:
            os.unlink(path)

    def test_partial_config_only_correlation(self):
        from methylcurate.contracts.qc import QcConfig

        path = _write_yaml("correlation_cutoff: 0.8\n")
        try:
            cfg = QcConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.dnam_cutoff is None
            assert cfg.sample_level_missing_cutoff is None
            assert cfg.cpg_level_missing_cutoff is None
            assert cfg.correlation_cutoff == 0.8
        finally:
            os.unlink(path)

    def test_unknown_fields_ignored(self):
        from methylcurate.contracts.qc import QcConfig

        path = _write_yaml("dnam_cutoff: 0.97\nunknown_setting: wrong\n")
        try:
            cfg = QcConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.dnam_cutoff == 0.97
            assert not hasattr(cfg, "unknown_setting")
        finally:
            os.unlink(path)

    def test_non_dict_yaml_raises(self):
        from methylcurate.contracts.qc import QcConfig

        path = _write_yaml("- item1\n- item2\n")
        try:
            with pytest.raises(ValueError, match="YAML mapping"):
                QcConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_env_interpolation(self):
        from methylcurate.contracts.qc import QcConfig

        os.environ["QC_TEST_VAR"] = "0.91"
        path = _write_yaml("dnam_cutoff: ${QC_TEST_VAR}\n")
        try:
            cfg = QcConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.dnam_cutoff == 0.91
        finally:
            os.unlink(path)

    def test_env_interpolation_missing_var_raises(self):
        from methylcurate.contracts.qc import QcConfig

        if "MISSING_QC_VAR" in os.environ:
            del os.environ["MISSING_QC_VAR"]
        path = _write_yaml("dnam_cutoff: ${MISSING_QC_VAR}\n")
        try:
            with pytest.raises(ValueError, match="MISSING_QC_VAR"):
                QcConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_expanduser_resolves_tilde(self):
        from pathlib import Path

        from methylcurate.contracts.qc import QcConfig

        path = _write_yaml("dnam_cutoff: 0.98\n")
        try:
            cfg = QcConfig.from_yaml(os.path.join("~", os.path.relpath(path, str(Path.home()))))
            assert cfg is not None
            assert cfg.dnam_cutoff == 0.98
        finally:
            os.unlink(path)


class TestMakeQualityControlStateWithConfig:
    def test_no_env_var_produces_none_inputs(self):
        from methylcurate.agent.state.utils import make_quality_control_state

        if "QC_CONFIG_PATH" in os.environ:
            del os.environ["QC_CONFIG_PATH"]
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            params = {"output_root": tmpdir, "accessions": ["GSE12345"]}
            state = make_quality_control_state("run-1", params)
            assert state.dnam_qc_input is None
            assert state.sample_level_qc_input is None
            assert state.cpg_level_qc_input is None
            assert state.interarray_correlation_qc_input is None
            assert state.data_conversion_input is None

    def test_config_with_all_fields_populates_inputs(self):
        from methylcurate.agent.state.utils import make_quality_control_state

        config_path = _write_yaml("dnam_cutoff: 0.90\nsample_level_missing_cutoff: 0.08\ncpg_level_missing_cutoff: 0.18\ncorrelation_cutoff: 0.82\n")
        try:
            with patch.dict(os.environ, {"QC_CONFIG_PATH": config_path}), tempfile.TemporaryDirectory() as tmpdir:
                data_dir = os.path.join(tmpdir, "data")
                os.makedirs(data_dir, exist_ok=True)
                params = {"output_root": tmpdir, "accessions": ["GSE12345"]}
                state = make_quality_control_state("run-1", params)

                assert state.dnam_qc_input is not None
                assert state.dnam_qc_input.dnam_cutoff == 0.90

                assert state.sample_level_qc_input is not None
                assert state.sample_level_qc_input.missing_cutoff == 0.08

                assert state.cpg_level_qc_input is not None
                assert state.cpg_level_qc_input.missing_cutoff == 0.18

                assert state.interarray_correlation_qc_input is not None
                assert state.interarray_correlation_qc_input.correlation_cutoff == 0.82
        finally:
            os.unlink(config_path)

    def test_config_with_partial_fields_only_sets_provided(self):
        from methylcurate.agent.state.utils import make_quality_control_state

        config_path = _write_yaml("dnam_cutoff: 0.88\ncorrelation_cutoff: 0.75\n")
        try:
            with patch.dict(os.environ, {"QC_CONFIG_PATH": config_path}), tempfile.TemporaryDirectory() as tmpdir:
                data_dir = os.path.join(tmpdir, "data")
                os.makedirs(data_dir, exist_ok=True)
                params = {"output_root": tmpdir, "accessions": ["GSE12345"]}
                state = make_quality_control_state("run-1", params)

                assert state.dnam_qc_input is not None
                assert state.dnam_qc_input.dnam_cutoff == 0.88

                assert state.interarray_correlation_qc_input is not None
                assert state.interarray_correlation_qc_input.correlation_cutoff == 0.75

                assert state.sample_level_qc_input is None
                assert state.cpg_level_qc_input is None
        finally:
            os.unlink(config_path)

    def test_config_file_not_found_still_produces_none_inputs(self):
        from methylcurate.agent.state.utils import make_quality_control_state

        with patch.dict(os.environ, {"QC_CONFIG_PATH": "/nonexistent/qc_config.yml"}), tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            params = {"output_root": tmpdir, "accessions": ["GSE12345"]}
            state = make_quality_control_state("run-1", params)
            assert state.dnam_qc_input is None
            assert state.sample_level_qc_input is None
            assert state.cpg_level_qc_input is None
            assert state.interarray_correlation_qc_input is None


class TestGetOrDefaultFallbacks:
    def test_dnam_qc_input_in_state_used(self):
        from methylcurate.agent.nodes.qc import _get_dnam_qc_input_or_default
        from methylcurate.agent.state.models import QualityControlSubgraphState

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            state = QualityControlSubgraphState.model_validate(
                {
                    "run_id": "r1",
                    "config": {"output_root": data_dir, "accessions": ["GSE12345"]},
                    "datasets": {"GSE12345": {"accession": "GSE12345", "output_dir": data_dir}},
                    "dnam_qc_input": {"dnam_cutoff": 0.87},
                }
            )
            result = _get_dnam_qc_input_or_default(state)
            assert result.dnam_cutoff == 0.87

    def test_dnam_qc_input_none_returns_default(self):
        from methylcurate.agent.nodes.qc import _get_dnam_qc_input_or_default
        from methylcurate.agent.state.models import QualityControlSubgraphState

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            state = QualityControlSubgraphState.model_validate(
                {
                    "run_id": "r1",
                    "config": {"output_root": data_dir, "accessions": ["GSE12345"]},
                    "datasets": {"GSE12345": {"accession": "GSE12345", "output_dir": data_dir}},
                }
            )
            result = _get_dnam_qc_input_or_default(state)
            assert result.dnam_cutoff == 0.96

    def test_sample_level_qc_input_none_returns_default(self):
        from methylcurate.agent.nodes.qc import _get_sample_level_qc_input_or_default
        from methylcurate.agent.state.models import QualityControlSubgraphState

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            state = QualityControlSubgraphState.model_validate(
                {
                    "run_id": "r1",
                    "config": {"output_root": data_dir, "accessions": ["GSE12345"]},
                    "datasets": {"GSE12345": {"accession": "GSE12345", "output_dir": data_dir}},
                }
            )
            result = _get_sample_level_qc_input_or_default(state)
            assert result.missing_cutoff == 0.1

    def test_cpg_level_qc_input_none_returns_default(self):
        from methylcurate.agent.nodes.qc import _get_cpg_level_qc_input_or_default
        from methylcurate.agent.state.models import QualityControlSubgraphState

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            state = QualityControlSubgraphState.model_validate(
                {
                    "run_id": "r1",
                    "config": {"output_root": data_dir, "accessions": ["GSE12345"]},
                    "datasets": {"GSE12345": {"accession": "GSE12345", "output_dir": data_dir}},
                }
            )
            result = _get_cpg_level_qc_input_or_default(state)
            assert result.missing_cutoff == 0.2
            assert result.imputation_strategy.strategy == "whole"

    def test_interarray_qc_input_none_returns_default(self):
        from methylcurate.agent.nodes.qc import _get_interarray_correlation_qc_input_or_default
        from methylcurate.agent.state.models import QualityControlSubgraphState

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            state = QualityControlSubgraphState.model_validate(
                {
                    "run_id": "r1",
                    "config": {"output_root": data_dir, "accessions": ["GSE12345"]},
                    "datasets": {"GSE12345": {"accession": "GSE12345", "output_dir": data_dir}},
                }
            )
            result = _get_interarray_correlation_qc_input_or_default(state)
            assert result.correlation_cutoff == 0.9
