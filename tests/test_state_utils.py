import pytest


class TestHarmonizationDatasetStateResolves:
    def test_get_dataset_for_subgraph_returns_harmonizationdatasetstate(self):
        from methylcurate.agent.state.utils import get_dataset_for_subgraph
        from methylcurate.agent.state.models import HarmonizationDatasetState

        result = get_dataset_for_subgraph("harmonization")
        assert result is HarmonizationDatasetState

    def test_get_dataset_for_subgraph_covers_all_cases(self):
        from methylcurate.agent.state.utils import get_dataset_for_subgraph
        from methylcurate.agent.state.models import (
            GeoDatasetState,
            HarmonizationDatasetState,
            DatasetQualityControlState,
        )

        assert get_dataset_for_subgraph("geo_retrieval") is GeoDatasetState
        assert get_dataset_for_subgraph("harmonization") is HarmonizationDatasetState
        assert get_dataset_for_subgraph("quality_control") is DatasetQualityControlState

    def test_get_dataset_for_subgraph_raises_on_unknown(self):
        from methylcurate.agent.state.utils import get_dataset_for_subgraph

        with pytest.raises(ValueError, match="Unknown subgraph"):
            get_dataset_for_subgraph("benchmarking")
