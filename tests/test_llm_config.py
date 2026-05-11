from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from methylcurate.agent.llm.client import LLMConfig


def _write_yaml(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    tmp.write(content)
    tmp.close()
    return tmp.name


class TestLLMConfigFromYaml:
    def test_none_path_returns_none(self):
        assert LLMConfig.from_yaml(None) is None

    def test_empty_string_path_returns_none(self):
        assert LLMConfig.from_yaml("") is None

    def test_nonexistent_file_returns_none(self):
        assert LLMConfig.from_yaml("/nonexistent/path/llm_config.yml") is None

    def test_valid_openai_config(self):
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        path = _write_yaml("provider: openai\nmodel: gpt-4o\napi_key: ${OPENAI_API_KEY}\ntemperature: 0.5\n")
        try:
            cfg = LLMConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.provider == "openai"
            assert cfg.model == "gpt-4o"
            assert cfg.api_key == "sk-test-key"
            assert cfg.temperature == 0.5
        finally:
            os.unlink(path)

    def test_env_interpolation_missing_var_raises(self):
        if "MISSING_VAR_XYZ" in os.environ:
            del os.environ["MISSING_VAR_XYZ"]
        path = _write_yaml("provider: openai\nmodel: gpt-4o\napi_key: ${MISSING_VAR_XYZ}\n")
        try:
            with pytest.raises(ValueError, match="MISSING_VAR_XYZ"):
                LLMConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_missing_required_field_raises(self):
        path = _write_yaml("provider: openai\n")
        try:
            with pytest.raises(ValueError, match="missing required field"):
                LLMConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_non_dict_yaml_raises(self):
        path = _write_yaml("- item1\n- item2\n")
        try:
            with pytest.raises(ValueError, match="YAML mapping"):
                LLMConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_unknown_fields_ignored(self):
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        path = _write_yaml("provider: openai\nmodel: gpt-4o\napi_key: ${OPENAI_API_KEY}\nunknown_field: ignored\n")
        try:
            cfg = LLMConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.provider == "openai"
        finally:
            os.unlink(path)

    def test_non_string_values_preserved(self):
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        path = _write_yaml("provider: openai\nmodel: gpt-4o\napi_key: ${OPENAI_API_KEY}\ntemperature: 0.7\ntop_k: 50\nstreaming: false\n")
        try:
            cfg = LLMConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.temperature == 0.7
            assert cfg.top_k == 50
            assert cfg.streaming is False
        finally:
            os.unlink(path)

    def test_expanduser_resolves_tilde(self):
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        path = _write_yaml("provider: openai\nmodel: gpt-4o\napi_key: ${OPENAI_API_KEY}\n")
        try:
            cfg = LLMConfig.from_yaml(os.path.join("~", os.path.relpath(path, str(Path.home()))))
            assert cfg is not None
            assert cfg.provider == "openai"
        finally:
            os.unlink(path)

    def test_ollama_config(self):
        path = _write_yaml("provider: ollama\nmodel: llama3\nbase_url: http://localhost:11434\n")
        try:
            cfg = LLMConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.provider == "ollama"
            assert cfg.model == "llama3"
            assert cfg.base_url == "http://localhost:11434"
        finally:
            os.unlink(path)

    def test_default_values_preserved_when_not_in_yaml(self):
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        path = _write_yaml("provider: openai\nmodel: gpt-4o\napi_key: ${OPENAI_API_KEY}\n")
        try:
            cfg = LLMConfig.from_yaml(path)
            assert cfg is not None
            assert cfg.top_k == 40
            assert cfg.top_p == 0.9
            assert cfg.timeout_s == 120
            assert cfg.max_retries == 2
        finally:
            os.unlink(path)
