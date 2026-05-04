import inspect

import pytest


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
            from methylcurate.contracts.harmonize import Concept  # noqa: F401
