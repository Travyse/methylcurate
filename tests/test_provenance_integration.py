"""Integration tests for provenance event emission in LLM calls."""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from methylcurate.agent.llm.logged_client import LoggedLLMClient, _extract_token_usage, _message_sha256, _render_preview, _schema_name
from methylcurate.utils.provenance import ProvenanceLogger, set_active_provenance


@pytest.fixture
def temp_log_path():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def provenance(temp_log_path):
    return ProvenanceLogger(log_path=temp_log_path, run_id="test-run", subgraph="geo_retrieval")


def read_events(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class MockLLMConfig:
    provider = "openai"
    model = "gpt-4o"
    temperature = 0.0
    top_p = 0.9
    top_k = 40
    timeout_s = 120
    max_retries = 2
    streaming = True


class TestLoggedLLMClient:
    @pytest.mark.asyncio
    async def test_emits_request_and_response_events(self, provenance, temp_log_path):
        mock_client = MagicMock()
        mock_client.config = MockLLMConfig()
        mock_client.acall_structured = AsyncMock(return_value=MagicMock())

        client = LoggedLLMClient(mock_client)
        set_active_provenance(provenance)
        await client.acall_structured("Test prompt", str)

        events = read_events(temp_log_path)
        assert len(events) >= 3

        event_types = [e["event"] for e in events]
        assert "PromptRendered" in event_types
        assert "LLMRequestStarted" in event_types
        assert "LLMResponseReceived" in event_types

    @pytest.mark.asyncio
    async def test_emits_parsed_success_false_on_error(self, provenance, temp_log_path):
        mock_client = MagicMock()
        mock_client.config = MockLLMConfig()
        mock_client.acall_structured = AsyncMock(side_effect=RuntimeError("test error"))

        client = LoggedLLMClient(mock_client)
        set_active_provenance(provenance)
        with pytest.raises(RuntimeError):
            await client.acall_structured("Test prompt", str)

        events = read_events(temp_log_path)
        response_events = [e for e in events if e["event"] == "LLMResponseReceived"]
        assert len(response_events) == 1
        assert response_events[0]["parsed_success"] is False

    @pytest.mark.asyncio
    async def test_includes_model_and_schema_in_request(self, provenance, temp_log_path):
        mock_client = MagicMock()
        mock_client.config = MockLLMConfig()
        mock_client.acall_structured = AsyncMock(return_value=MagicMock())

        client = LoggedLLMClient(mock_client)
        set_active_provenance(provenance)
        await client.acall_structured("Test prompt", str)

        events = read_events(temp_log_path)
        request = [e for e in events if e["event"] == "LLMRequestStarted"][0]
        assert request["model"] == "gpt-4o"
        assert request["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_includes_latency_ms_in_response(self, provenance, temp_log_path):
        mock_client = MagicMock()
        mock_client.config = MockLLMConfig()
        mock_client.acall_structured = AsyncMock(return_value=MagicMock())

        client = LoggedLLMClient(mock_client)
        set_active_provenance(provenance)
        await client.acall_structured("Test prompt", str)

        events = read_events(temp_log_path)
        response = [e for e in events if e["event"] == "LLMResponseReceived"][0]
        assert "latency_ms" in response
        assert response["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_emits_token_usage_when_available(self, provenance, temp_log_path):
        mock_result = MagicMock()
        mock_result.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

        mock_client = MagicMock()
        mock_client.config = MockLLMConfig()
        mock_client.acall_structured = AsyncMock(return_value=mock_result)

        client = LoggedLLMClient(mock_client)
        set_active_provenance(provenance)
        await client.acall_structured("Test prompt", str)

        events = read_events(temp_log_path)
        token_events = [e for e in events if e["event"] == "LLMTokenUsage"]
        assert len(token_events) == 1
        assert token_events[0]["prompt_tokens"] == 100
        assert token_events[0]["completion_tokens"] == 50
        assert token_events[0]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_no_provenance_no_events(self, temp_log_path):
        mock_client = MagicMock()
        mock_client.config = MockLLMConfig()
        mock_client.acall_structured = AsyncMock(return_value=MagicMock())

        client = LoggedLLMClient(mock_client)
        set_active_provenance(None)
        await client.acall_structured("Test prompt", str)

        events = read_events(temp_log_path)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_passthrough_to_wrapped_client(self):
        mock_client = MagicMock()
        mock_client.config = MockLLMConfig()
        mock_client.acall_structured = AsyncMock(return_value="expected")
        mock_client.some_other_method = MagicMock(return_value=42)

        client = LoggedLLMClient(mock_client)
        result = await client.acall_structured("Test prompt", str)
        assert result == "expected"
        assert client.some_other_method() == 42
        assert client.config == mock_client.config


class TestHelpers:
    def test_message_sha256_str(self):
        h = _message_sha256("hello")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_message_sha256_list_of_messages(self):
        from langchain_core.messages import HumanMessage, SystemMessage

        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="hello"),
        ]
        h = _message_sha256(msgs)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_message_sha256_deterministic(self):
        from langchain_core.messages import HumanMessage

        msgs1 = [HumanMessage(content="hello")]
        msgs2 = [HumanMessage(content="hello")]
        assert _message_sha256(msgs1) == _message_sha256(msgs2)

    def test_message_sha256_different_yields_different(self):
        from langchain_core.messages import HumanMessage

        msgs1 = [HumanMessage(content="hello")]
        msgs2 = [HumanMessage(content="world")]
        assert _message_sha256(msgs1) != _message_sha256(msgs2)

    def test_render_preview_string(self):
        result = _render_preview("Short text")
        assert result == "Short text"

    def test_render_preview_long_text(self):
        long_text = "A" * 300
        result = _render_preview(long_text)
        assert len(result) <= 215
        assert result.endswith("...<truncated>")

    def test_render_preview_message_list(self):
        from langchain_core.messages import HumanMessage

        msgs = [HumanMessage(content="Hello world")]
        result = _render_preview(msgs)
        assert "Hello world" in result

    def test_schema_name_from_class(self):
        class MyModel:
            pass

        assert _schema_name(MyModel) == "MyModel"

    def test_extract_token_usage_from_usage_metadata(self):
        result = MagicMock()
        result.usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        usage = _extract_token_usage(result)
        assert usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_extract_token_usage_from_response_metadata(self):
        result = MagicMock()
        result.usage_metadata = None
        result.response_metadata = {"token_usage": {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}}
        usage = _extract_token_usage(result)
        assert usage == {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}

    def test_extract_token_usage_none(self):
        result = MagicMock()
        result.usage_metadata = None
        result.response_metadata = {}
        usage = _extract_token_usage(result)
        assert usage is None


class TestProvenanceRetryEvents:
    def test_retry_scheduled_has_correct_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            logger = ProvenanceLogger(log_path=path, run_id="test", subgraph="harmonization")
            logger.emit_retry_scheduled(
                error_type="TimeoutError",
                error_message="timed out",
                retry_count=3,
                retry_limit=5,
                backoff_s=2.0,
                accession_code="GSE40279",
                step="call_llm_structured_with_retries",
            )
            logger.close()
            events = read_events(path)
            e = events[0]
            assert e["event"] == "RetryScheduled"
            assert e["retry_count"] == 3
            assert e["retry_limit"] == 5
            assert e["backoff_s"] == 2.0
            assert e["accession_code"] == "GSE40279"
            assert e["step"] == "call_llm_structured_with_retries"
        finally:
            os.unlink(path)

    def test_error_raised_has_correct_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            logger = ProvenanceLogger(log_path=path, run_id="test", subgraph="geo_retrieval")
            logger.emit_error_raised(
                error_type="ValidationError",
                error_message="LLM output failed validation",
                error_code="LLM_VALIDATION_FAILURE",
                accession_code="GSE40279",
                step="_extract_all_columns",
            )
            logger.close()
            events = read_events(path)
            e = events[0]
            assert e["event"] == "ErrorRaised"
            assert e["error_type"] == "ValidationError"
            assert e["error_code"] == "LLM_VALIDATION_FAILURE"
        finally:
            os.unlink(path)
