"""Unit tests for ProvenanceLogger."""

import json
import os
import tempfile

import pytest
from methylcurate.utils.provenance import ProvenanceLogger, ProvenanceRegistry, _redact_sensitive, _safe_truncate


@pytest.fixture
def temp_log_path():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def logger(temp_log_path):
    return ProvenanceLogger(log_path=temp_log_path, run_id="20260505T123456-00000001", subgraph="geo_retrieval")


def read_events(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestProvenanceLogger:
    def test_emit_run_started_writes_jsonl(self, logger, temp_log_path):
        logger.emit_run_started(
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            top_p=0.9,
            top_k=40,
            timeout_s=120,
            max_retries=2,
            streaming=True,
            accessions=["GSE40279"],
            output_root="/tmp/outputs",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "RunStarted"
        assert e["run_id"] == "20260505T123456-00000001"
        assert e["subgraph"] == "geo_retrieval"
        assert e["config_snapshot"]["provider"] == "openai"
        assert e["config_snapshot"]["model"] == "gpt-4o"
        assert e["accessions"] == ["GSE40279"]

    def test_emit_artifact_written(self, logger, temp_log_path):
        logger.emit_artifact_written(
            artifact_kind="soft_file",
            artifact_path="/tmp/GSE40279_family.soft",
            artifact_sha256="abc123",
            artifact_bytes=1024,
            accession_code="GSE40279",
            step="download_soft",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "ArtifactWritten"
        assert e["artifact_kind"] == "soft_file"
        assert e["artifact_path"] == "/tmp/GSE40279_family.soft"
        assert e["artifact_sha256"] == "abc123"
        assert e["accession_code"] == "GSE40279"
        assert e["step"] == "download_soft"

    def test_emit_error_raised(self, logger, temp_log_path):
        logger.emit_error_raised(
            error_type="TimeoutError",
            error_message="LLM request timed out after 180s",
            error_code="LLM_TIMEOUT",
            accession_code="GSE40279",
            step="extract_metadata_schema",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "ErrorRaised"
        assert e["error_type"] == "TimeoutError"
        assert e["error_code"] == "LLM_TIMEOUT"

    def test_emit_run_completed_with_manifest(self, logger, temp_log_path):
        logger.emit_run_completed(
            status="completed",
            total_artifacts=3,
            artifact_manifest=[
                {"kind": "soft_file", "path": "/tmp/a.soft", "sha256": "aaa"},
                {"kind": "metadata_cache", "path": "/tmp/b.json", "sha256": "bbb"},
            ],
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "RunCompleted"
        assert e["total_artifacts"] == 3
        assert len(e["artifact_manifest"]) == 2

    def test_emit_warning_raised(self, logger, temp_log_path):
        logger.emit_warning_raised(
            warning="Parse rate below 0.8 for tissue",
            context="extraction refinement",
            accession_code="GSE40279",
            step="refine_metadata_schema",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "WarningRaised"
        assert "Parse rate" in e["warning"]

    def test_emit_retry_scheduled(self, logger, temp_log_path):
        logger.emit_retry_scheduled(
            error_type="TimeoutError",
            error_message="Request timed out",
            retry_count=2,
            retry_limit=5,
            backoff_s=3.0,
            accession_code="GSE40279",
            step="extract_metadata_schema",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "RetryScheduled"
        assert e["retry_count"] == 2
        assert e["backoff_s"] == 3.0

    def test_emit_llm_request_started(self, logger, temp_log_path):
        logger.emit_llm_request_started(
            model="gpt-4o",
            provider="openai",
            temperature=0.0,
            attempt=1,
            structured_output_schema="GEOMetadataExtractionResult",
            accession_code="GSE40279",
            step="extract_metadata_schema",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "LLMRequestStarted"
        assert e["model"] == "gpt-4o"

    def test_emit_llm_response_received(self, logger, temp_log_path):
        logger.emit_llm_response_received(
            latency_ms=1250.5,
            parsed_success=True,
            attempt=1,
            accession_code="GSE40279",
            step="extract_metadata_schema",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "LLMResponseReceived"
        assert e["latency_ms"] == 1250.5

    def test_emit_input_loaded(self, logger, temp_log_path):
        logger.emit_input_loaded(
            artifact_kind="metadata_cache",
            artifact_path="/tmp/cache.json",
            artifact_sha256="def456",
            accession_code="GSE40279",
            step="disease_harmonization_node",
            key_names=["tissue", "disease"],
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "InputLoaded"
        assert e["key_names"] == ["tissue", "disease"]

    def test_emit_prompt_rendered(self, logger, temp_log_path):
        logger.emit_prompt_rendered(
            template_name="guess_ontology_label_system_prompt",
            template_path="harmonize/guess_label/guess_ontology_label_system_prompt.md",
            rendered_sha256="abc123",
            rendered_preview="You are a biomedical ontologist...",
            accession_code="GSE40279",
            step="disease_harmonization_node",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "PromptRendered"
        assert e["template_name"] == "guess_ontology_label_system_prompt"
        assert e["rendered_sha256"] == "abc123"
        assert "biomedical" in e["rendered_preview"]

    def test_emit_retrieval_query_issued(self, logger, temp_log_path):
        logger.emit_retrieval_query_issued(
            query="Alzheimer disease",
            ontology="mondo",
            k=5,
            api_url="https://www.ebi.ac.uk/ols4/api/search",
            accession_code="GSE40279",
            step="disease_harmonization_node",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "RetrievalQueryIssued"
        assert e["ontology"] == "mondo"
        assert e["k"] == 5

    def test_emit_retrieval_candidates_returned(self, logger, temp_log_path):
        logger.emit_retrieval_candidates_returned(
            query="Alzheimer disease",
            ontology="mondo",
            num_candidates=3,
            candidate_ids=["MONDO:0004975", "MONDO:1234567", "DOID:10652"],
            candidate_labels=["Alzheimer disease", "Alzheimer's disease", "Alzheimer Disease"],
            accession_code="GSE40279",
            step="disease_harmonization_node",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "RetrievalCandidatesReturned"
        assert e["num_candidates"] == 3
        assert len(e["candidate_ids"]) == 3

    def test_emit_harmonization_mapping_proposed(self, logger, temp_log_path):
        logger.emit_harmonization_mapping_proposed(
            source_label="Alzheimer's",
            target_label="MONDO:0004975",
            ontology="mondo",
            mapping_type="selection",
            candidates=["MONDO:0004975", "MONDO:1234567"],
            accession_code="GSE40279",
            step="disease_harmonization_node",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "HarmonizationMappingProposed"
        assert e["mapping_type"] == "selection"

    def test_emit_validation_completed(self, logger, temp_log_path):
        logger.emit_validation_completed(
            concept="tissue",
            parse_rate=0.95,
            num_resolved=8,
            num_missing=0,
            num_error=0,
            accession_code="GSE40279",
            step="extract_metadata_schema",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "ValidationCompleted"
        assert e["parse_rate"] == 0.95

    def test_multiple_events_ordered(self, logger, temp_log_path):
        logger.emit_run_started(
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            top_p=0.9,
            top_k=40,
            timeout_s=120,
            max_retries=2,
            streaming=True,
            accessions=["GSE40279"],
            output_root="/tmp",
        )
        logger.emit_artifact_written(
            artifact_kind="soft_file",
            artifact_path="/tmp/a.soft",
            artifact_sha256="aaa",
            artifact_bytes=100,
        )
        logger.emit_run_completed(status="completed", total_artifacts=1)
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 3
        assert events[0]["event"] == "RunStarted"
        assert events[1]["event"] == "ArtifactWritten"
        assert events[2]["event"] == "RunCompleted"

    def test_required_base_fields_present(self, logger, temp_log_path):
        logger.emit_artifact_written(
            artifact_kind="test",
            artifact_path="/tmp/test",
        )
        logger.close()
        events = read_events(temp_log_path)
        e = events[0]
        assert "event" in e
        assert "timestamp" in e
        assert "run_id" in e
        assert "subgraph" in e

    def test_no_emit_after_close(self, logger, temp_log_path):
        logger.close()
        logger.emit_artifact_written(
            artifact_kind="test",
            artifact_path="/tmp/test",
        )
        events = read_events(temp_log_path)
        assert len(events) == 0

    def test_context_manager(self, temp_log_path):
        with ProvenanceLogger(log_path=temp_log_path, run_id="x", subgraph="y") as pl:
            pl.emit_artifact_written(artifact_kind="test", artifact_path="/tmp/test")
        events = read_events(temp_log_path)
        assert len(events) == 1

    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "subdir", "prov.jsonl")
            logger = ProvenanceLogger(log_path=log_path, run_id="x", subgraph="y")
            logger.emit_artifact_written(artifact_kind="test", artifact_path="/tmp/test")
            logger.close()
            assert os.path.exists(log_path)

    def test_thread_safe_writes(self, temp_log_path):
        import threading

        logger = ProvenanceLogger(log_path=temp_log_path, run_id="x", subgraph="y")
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    logger.emit_artifact_written(
                        artifact_kind=f"test_{n}",
                        artifact_path=f"/tmp/test_{n}_{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        logger.close()
        assert len(errors) == 0
        events = read_events(temp_log_path)
        assert len(events) == 50


class TestRedaction:
    def test_redacts_openai_api_key(self):
        text = "Authorization: Bearer sk-proj-abc123def456ghi789jkl"
        result = _redact_sensitive(text)
        assert "sk-proj-abc123def456ghi789jkl" not in result
        assert "sk-<REDACTED>" in result

    def test_redacts_anthropic_api_key(self):
        text = "x-api-key: sk-ant-api03-abc123xyz"
        result = _redact_sensitive(text)
        assert "sk-ant-api03-abc123xyz" not in result
        assert "sk-ant-<REDACTED>" in result

    def test_redacts_email(self):
        text = "Contact: user@example.com for help"
        result = _redact_sensitive(text)
        assert "user@example.com" not in result
        assert "<EMAIL_REDACTED>" in result

    def test_rendered_preview_is_redacted(self, logger, temp_log_path):
        logger.emit_prompt_rendered(
            template_name="test",
            template_path="test.md",
            rendered_sha256="abc",
            rendered_preview="Use key sk-proj-secret123 to call API. Email admin@example.com.",
        )
        logger.close()
        events = read_events(temp_log_path)
        preview = events[0]["rendered_preview"]
        assert "sk-proj-secret123" not in preview
        assert "admin@example.com" not in preview

    def test_str_without_sensitive_data_passes_through(self):
        text = "Normal text without secrets or emails"
        result = _redact_sensitive(text)
        assert result == text


class TestSafeTruncate:
    def test_truncates_long_text(self):
        long_text = "x" * 500
        result = _safe_truncate(long_text, max_chars=200)
        assert len(result) == 214  # 200 + 14 for "...<truncated>"
        assert result.endswith("...<truncated>")

    def test_preserves_short_text(self):
        short_text = "Hello world"
        result = _safe_truncate(short_text, max_chars=200)
        assert result == short_text


class TestRetriesExhausted:
    def test_emit_retries_exhausted(self, logger, temp_log_path):
        logger.emit_retries_exhausted(
            error_type="TimeoutError",
            error_message="All retries exhausted",
            total_attempts=5,
            max_retries=5,
            accession_code="GSE40279",
            step="_invoke_llm_with_retry",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "RetriesExhausted"
        assert e["error_type"] == "TimeoutError"
        assert e["total_attempts"] == 5
        assert e["max_retries"] == 5
        assert e["accession_code"] == "GSE40279"

    def test_emit_retries_exhausted_defaults(self, temp_log_path):
        logger = ProvenanceLogger(log_path=temp_log_path, run_id="test", subgraph="y")
        logger.emit_retries_exhausted(error_type="GenericError")
        logger.close()
        events = read_events(temp_log_path)
        assert events[0]["total_attempts"] == 0
        assert events[0]["max_retries"] == 0


class TestErrorCodes:
    def test_error_raised_includes_error_code(self, logger, temp_log_path):
        logger.emit_error_raised(
            error_type="ValidationError",
            error_message="test",
            error_code="LLM_VALIDATION_FAILURE",
            step="test_step",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert events[0]["error_code"] == "LLM_VALIDATION_FAILURE"

    def test_error_raised_error_code_is_none_by_default(self, temp_log_path):
        logger = ProvenanceLogger(log_path=temp_log_path, run_id="test", subgraph="x")
        logger.emit_error_raised(error_type="SomeError", error_message="test")
        logger.close()
        events = read_events(temp_log_path)
        assert events[0]["error_code"] is None


class TestProvenanceLoggerThreadId:
    def test_thread_id_in_events(self, temp_log_path):
        logger = ProvenanceLogger(
            log_path=temp_log_path,
            run_id="20260505T123456-00000001",
            subgraph="geo_retrieval",
            thread_id="abc123def456",
        )
        logger.emit_artifact_written(
            artifact_kind="soft_file",
            artifact_path="/tmp/test.soft",
            artifact_sha256="abc",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert events[0]["thread_id"] == "abc123def456"

    def test_no_thread_id_in_events_when_none(self, temp_log_path):
        logger = ProvenanceLogger(
            log_path=temp_log_path,
            run_id="20260505T123456-00000001",
            subgraph="geo_retrieval",
        )
        logger.emit_artifact_written(
            artifact_kind="soft_file",
            artifact_path="/tmp/test.soft",
            artifact_sha256="abc",
        )
        logger.close()
        events = read_events(temp_log_path)
        assert "thread_id" not in events[0]


class TestProvenanceRegistry:
    def test_get_or_create_returns_same_instance(self):
        with tempfile.TemporaryDirectory() as td:
            registry = ProvenanceRegistry()
            a = registry.get_or_create("thread-1", td)
            b = registry.get_or_create("thread-1", td)
            assert a is b

    def test_get_or_create_creates_file_in_provenance_dir(self):
        with tempfile.TemporaryDirectory() as td:
            registry = ProvenanceRegistry()
            logger = registry.get_or_create("thread-1", os.path.join(td, "thread-1"))
            logger.emit_artifact_written(
                artifact_kind="test",
                artifact_path="/tmp/test",
            )
            logger.close()
            prov_path = os.path.join(td, "thread-1", "logs", "provenance.jsonl")
            assert os.path.exists(prov_path)
            events = read_events(prov_path)
            assert len(events) == 1
            assert events[0]["thread_id"] == "thread-1"

    def test_get_returns_none_for_unknown_thread(self):
        registry = ProvenanceRegistry()
        assert registry.get("unknown") is None

    def test_close_removes_logger(self):
        with tempfile.TemporaryDirectory() as td:
            registry = ProvenanceRegistry()
            registry.get_or_create("thread-1", td)
            registry.close("thread-1")
            assert registry.get("thread-1") is None

    def test_close_all_clears_everything(self):
        with tempfile.TemporaryDirectory() as td:
            registry = ProvenanceRegistry()
            registry.get_or_create("thread-1", td)
            registry.get_or_create("thread-2", td)
            registry.close_all()
            assert registry.get("thread-1") is None
            assert registry.get("thread-2") is None

    def test_separate_threads_write_to_separate_files(self):
        with tempfile.TemporaryDirectory() as td:
            registry = ProvenanceRegistry()
            a = registry.get_or_create("thread-1", os.path.join(td, "thread-1"))
            b = registry.get_or_create("thread-2", os.path.join(td, "thread-2"))
            a.emit_artifact_written(
                artifact_kind="from_a",
                artifact_path="/tmp/a",
            )
            b.emit_artifact_written(
                artifact_kind="from_b",
                artifact_path="/tmp/b",
            )
            a.close()
            b.close()

            events_a = read_events(os.path.join(td, "thread-1", "logs", "provenance.jsonl"))
            events_b = read_events(os.path.join(td, "thread-2", "logs", "provenance.jsonl"))
            assert len(events_a) == 1
            assert events_a[0]["artifact_kind"] == "from_a"
            assert events_a[0]["thread_id"] == "thread-1"
            assert len(events_b) == 1
            assert events_b[0]["artifact_kind"] == "from_b"
            assert events_b[0]["thread_id"] == "thread-2"
