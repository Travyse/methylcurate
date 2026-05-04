import inspect
from unittest.mock import MagicMock, PropertyMock

import pytest


class TestStepsAttributeAccess:
    def test_update_small_progress_tracker_uses_status_attribute(self):
        """Verify update_small_progress_tracker accesses .status, not string.

        The function body must use ``.steps["quality_control"].status``
        rather than treating the StepStatus object as a plain string.
        """
        import methylcurate.utils.helper as mod

        source = inspect.getsource(mod.update_small_progress_tracker)
        assert '.steps["quality_control"].status' in source, (
            "update_small_progress_tracker must access the .status attribute, "
            "not compare the StepStatus object directly to a string"
        )

    def test_update_small_progress_tracker_no_direct_string_compare(self):
        import methylcurate.utils.helper as mod

        source = inspect.getsource(mod.update_small_progress_tracker)
        assert '.steps["quality_control"] == "completed"' not in source


class TestGetSupplementaryFileIdReturnType:
    def test_return_type_is_str_not_list(self):
        from methylcurate.utils.helper import _get_supplementary_file_id

        sig = inspect.signature(_get_supplementary_file_id)
        assert sig.return_annotation is str
