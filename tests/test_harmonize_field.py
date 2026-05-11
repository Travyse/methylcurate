import inspect
import json

import pytest
import requests


class TestRenamedFunctions:
    def test_search_ontology_term_exists(self):
        """Verify search_ols was renamed to search_ontology_term."""
        from methylcurate.tools.harmonize import harmonize_field

        assert hasattr(harmonize_field, "search_ontology_term")
        assert not hasattr(harmonize_field, "search_ols")

    def test_call_llm_structured_with_retries_exists(self):
        """Verify llm_conversation was renamed to call_llm_structured_with_retries."""
        from methylcurate.tools.harmonize import harmonize_field

        assert hasattr(harmonize_field, "call_llm_structured_with_retries")
        assert not hasattr(harmonize_field, "llm_conversation")

    def test_search_ontology_term_type_hints_correct(self):
        from methylcurate.tools.harmonize.harmonize_field import search_ontology_term

        sig = inspect.signature(search_ontology_term)
        params = sig.parameters
        assert params["query"].annotation is str
        assert params["ontology"].annotation is str
        assert params["k"].annotation is int
        assert params["max_retries"].annotation is int
        assert params["backoff_s"].annotation is float
        assert "list" in str(sig.return_annotation)


class TestHarmonizeReturnTypes:
    def test_harmonize_ontology_labels_returns_tuple_of_labelmappingsets(self):
        from methylcurate.tools.harmonize.harmonize_field import _harmonize_ontology_labels

        sig = inspect.signature(_harmonize_ontology_labels)
        annotation = str(sig.return_annotation)
        assert "tuple" in annotation
        assert "LabelMappingSet" in annotation

    def test_harmonize_ontology_group_labels_returns_tuple(self):
        from methylcurate.tools.harmonize.harmonize_field import _harmonize_ontology_group_labels

        sig = inspect.signature(_harmonize_ontology_group_labels)
        annotation = str(sig.return_annotation)
        assert "tuple" in annotation
        assert "LabelMappingSet" in annotation

    def test_harmonize_sex_labels_returns_tuple_of_labelmappingsets(self):
        from methylcurate.tools.harmonize.harmonize_field import _harmonize_sex_labels

        sig = inspect.signature(_harmonize_sex_labels)
        annotation = str(sig.return_annotation)
        assert "tuple" in annotation
        assert "LabelMappingSet" in annotation

    def test_construct_raw_mapping_returns_labelmappingset(self):
        from methylcurate.tools.harmonize.harmonize_field import construct_raw_to_harmonized_label_mapping

        sig = inspect.signature(construct_raw_to_harmonized_label_mapping)
        annotation = str(sig.return_annotation)
        assert "LabelMappingSet" in annotation


class TestConstructRawMappingHandlesMissing:
    def test_mapping_with_missing_target_does_not_stopiteration(self):
        from methylcurate.contracts.harmonize import BestGuessMapping, LabelMappingSet, MissingMapping, MondoMapping
        from methylcurate.tools.harmonize.harmonize_field import construct_raw_to_harmonized_label_mapping

        guessed = LabelMappingSet(
            mappings=[
                BestGuessMapping(
                    ontology="best_guess",
                    source_label="unknown_label",
                    target_label="unmatched_target",
                ),
                MissingMapping(
                    ontology="missing",
                    source_label="missing_label",
                ),
            ]
        )

        selection = LabelMappingSet(
            mappings=[
                MondoMapping(
                    ontology="mondo",
                    source_label="some_other",
                    target_label="DOID:1234",
                ),
            ]
        )

        result = construct_raw_to_harmonized_label_mapping(guessed, selection)
        assert isinstance(result, LabelMappingSet)
        assert len(result.mappings) == 2

    def test_empty_guess_and_selection_does_not_crash(self):
        from methylcurate.contracts.harmonize import LabelMappingSet
        from methylcurate.tools.harmonize.harmonize_field import construct_raw_to_harmonized_label_mapping

        guessed = LabelMappingSet(mappings=[])
        selection = LabelMappingSet(mappings=[])
        result = construct_raw_to_harmonized_label_mapping(guessed, selection)
        assert isinstance(result, LabelMappingSet)
        assert result.mappings == []


class TestNoDuplicateValidationErrorImport:
    def test_validationerror_not_imported_twice(self):
        """Verify pydantic.ValidationError is imported only once at module level."""
        import methylcurate.tools.harmonize.harmonize_field as mod

        source = inspect.getsource(mod)
        lines = [line for line in source.split("\n") if "from pydantic import ValidationError" in line]
        assert len(lines) == 1, "ValidationError should be imported exactly once"


class TestBestGuessNotesConstant:
    def test_best_guess_notes_is_defined(self):
        from methylcurate.tools.harmonize.harmonize_field import BEST_GUESS_NOTES

        assert isinstance(BEST_GUESS_NOTES, str)
        assert len(BEST_GUESS_NOTES) > 50
        assert "ontology" in BEST_GUESS_NOTES

    def test_control_best_guess_notes_is_defined(self):
        from methylcurate.tools.harmonize.harmonize_field import CONTROL_BEST_GUESS_NOTES

        assert isinstance(CONTROL_BEST_GUESS_NOTES, str)
        assert "control" in CONTROL_BEST_GUESS_NOTES.lower()


class TestHarmonizationConcept:
    def test_harmonization_concept_exists(self):
        import typing

        from methylcurate.contracts.harmonize import HarmonizationConcept

        assert typing.get_origin(HarmonizationConcept) is typing.Literal

    def test_concept_not_accessible_as_concept(self):
        with pytest.raises(ImportError):
            from methylcurate.contracts.harmonize import Concept  # noqa: F401  # type: ignore


class TestSearchOntologyTermRetries:
    def test_success_first_attempt(self, mocker):
        from methylcurate.tools.harmonize.harmonize_field import search_ontology_term

        mock_docs = [{"obo_id": "DOID:1234", "label": "Test Disease"}]
        mock_response = mocker.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": {"docs": mock_docs}}
        mock_get = mocker.patch("requests.get", return_value=mock_response)
        mock_provenance = mocker.Mock()

        result = search_ontology_term("test", provenance=mock_provenance)

        assert result == [{"ontology": "mondo", "id": "DOID:1234", "label": "Test Disease"}]
        assert mock_get.call_count == 1
        mock_provenance.emit_retry_scheduled.assert_not_called()
        mock_provenance.emit_retries_exhausted.assert_not_called()

    def test_connection_error_retries_then_succeeds(self, mocker):
        from methylcurate.tools.harmonize.harmonize_field import search_ontology_term

        mock_docs = [{"obo_id": "DOID:1234", "label": "Test Disease"}]
        mock_response = mocker.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": {"docs": mock_docs}}
        mock_get = mocker.patch(
            "requests.get",
            side_effect=[
                requests.ConnectionError("connection refused"),
                mock_response,
            ],
        )
        mock_provenance = mocker.Mock()

        result = search_ontology_term("test", provenance=mock_provenance)

        assert result == [{"ontology": "mondo", "id": "DOID:1234", "label": "Test Disease"}]
        assert mock_get.call_count == 2
        mock_provenance.emit_retry_scheduled.assert_called_once()
        call_kwargs = mock_provenance.emit_retry_scheduled.call_args.kwargs
        assert call_kwargs["error_type"] == "ConnectionError"
        assert call_kwargs["retry_count"] == 1
        mock_provenance.emit_retries_exhausted.assert_not_called()

    def test_json_decode_error_retries_then_succeeds(self, mocker):
        from methylcurate.tools.harmonize.harmonize_field import search_ontology_term

        mock_docs = [{"obo_id": "DOID:1234", "label": "Test Disease"}]
        mock_response = mocker.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": {"docs": mock_docs}}
        mock_get = mocker.patch(
            "requests.get",
            side_effect=[
                json.JSONDecodeError("bad json", "doc", 0),
                mock_response,
            ],
        )
        mock_provenance = mocker.Mock()

        result = search_ontology_term("test", provenance=mock_provenance)

        assert result == [{"ontology": "mondo", "id": "DOID:1234", "label": "Test Disease"}]
        assert mock_get.call_count == 2
        mock_provenance.emit_retry_scheduled.assert_called_once()
        call_kwargs = mock_provenance.emit_retry_scheduled.call_args.kwargs
        assert call_kwargs["error_type"] == "JSONDecodeError"
        mock_provenance.emit_retries_exhausted.assert_not_called()

    def test_500_retries_then_succeeds(self, mocker):
        from methylcurate.tools.harmonize.harmonize_field import search_ontology_term

        mock_docs = [{"obo_id": "DOID:1234", "label": "Test Disease"}]
        good_response = mocker.Mock()
        good_response.raise_for_status.return_value = None
        good_response.json.return_value = {"response": {"docs": mock_docs}}

        bad_response = mocker.Mock()
        bad_response.status_code = 500
        http_error = requests.HTTPError("500 Server Error", response=bad_response)

        mock_get = mocker.patch(
            "requests.get",
            side_effect=[http_error, good_response],
        )
        mock_provenance = mocker.Mock()

        result = search_ontology_term("test", provenance=mock_provenance)

        assert result == [{"ontology": "mondo", "id": "DOID:1234", "label": "Test Disease"}]
        assert mock_get.call_count == 2

    def test_400_raises_immediately_no_retry(self, mocker):
        from methylcurate.tools.harmonize.harmonize_field import search_ontology_term

        bad_response = mocker.Mock()
        bad_response.status_code = 400
        http_error = requests.HTTPError("400 Bad Request", response=bad_response)

        mock_get = mocker.patch("requests.get", side_effect=http_error)
        mock_provenance = mocker.Mock()

        with pytest.raises(requests.HTTPError):
            search_ontology_term("test", provenance=mock_provenance)

        assert mock_get.call_count == 1
        mock_provenance.emit_retries_exhausted.assert_called_once()
        call_kwargs = mock_provenance.emit_retries_exhausted.call_args.kwargs
        assert call_kwargs["error_type"] == "HTTPError"

    def test_all_retries_exhausted_raises(self, mocker):
        from methylcurate.tools.harmonize.harmonize_field import search_ontology_term

        mock_get = mocker.patch(
            "requests.get",
            side_effect=requests.ConnectionError("connection refused"),
        )
        mock_provenance = mocker.Mock()

        with pytest.raises(requests.ConnectionError):
            search_ontology_term("test", max_retries=3, provenance=mock_provenance)

        assert mock_get.call_count == 4
        assert mock_provenance.emit_retry_scheduled.call_count == 3
        mock_provenance.emit_retries_exhausted.assert_called_once()
        call_kwargs = mock_provenance.emit_retries_exhausted.call_args.kwargs
        assert call_kwargs["error_type"] == "ConnectionError"
        assert call_kwargs["total_attempts"] == 4
        assert call_kwargs["max_retries"] == 3

    def test_doid_fallback_with_retries(self, mocker):
        from methylcurate.tools.harmonize.harmonize_field import search_ontology_term

        empty_response = mocker.Mock()
        empty_response.raise_for_status.return_value = None
        empty_response.json.return_value = {"response": {"docs": []}}

        doid_fail = mocker.Mock()
        doid_fail.status_code = 503
        doid_http_error = requests.HTTPError("503", response=doid_fail)

        doid_docs = [{"obo_id": "DOID:5678", "label": "Disease X"}]
        doid_success = mocker.Mock()
        doid_success.raise_for_status.return_value = None
        doid_success.json.return_value = {"response": {"docs": doid_docs}}

        mock_get = mocker.patch(
            "requests.get",
            side_effect=[empty_response, doid_http_error, doid_success],
        )
        mock_provenance = mocker.Mock()

        result = search_ontology_term("test", provenance=mock_provenance)

        assert result == [{"ontology": "doid", "id": "DOID:5678", "label": "Disease X"}]
        assert mock_get.call_count == 3

    def test_retry_count_increases_and_backoff_multiplies(self, mocker):
        from methylcurate.tools.harmonize.harmonize_field import search_ontology_term

        mocker.patch(
            "requests.get",
            side_effect=[
                requests.Timeout("timed out"),
                requests.Timeout("timed out"),
                requests.ConnectionError("refused"),
            ],
        )
        mock_provenance = mocker.Mock()

        with pytest.raises(requests.ConnectionError):
            search_ontology_term("test", max_retries=2, provenance=mock_provenance)

        assert mock_provenance.emit_retry_scheduled.call_count == 2
        first_call = mock_provenance.emit_retry_scheduled.call_args_list[0].kwargs
        second_call = mock_provenance.emit_retry_scheduled.call_args_list[1].kwargs
        assert first_call["retry_count"] == 1
        assert second_call["retry_count"] == 2
        assert first_call["backoff_s"] == 1.0
        assert second_call["backoff_s"] == 2.0
