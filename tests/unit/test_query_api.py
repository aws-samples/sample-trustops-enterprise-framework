"""Tests for ResultsQueryAPI module."""

from unittest.mock import MagicMock

import pytest

from src.data_models.storage import QueryFilter
from src.storage.query_api import PaginatedResults, ResultsQueryAPI


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def api(mock_store):
    return ResultsQueryAPI(mock_store)


class TestQuery:
    def test_basic_query(self, api, mock_store):
        mock_store.query_with_filter.return_value = [
            {"result_id": "r1"},
            {"result_id": "r2"},
        ]
        result = api.query()
        assert len(result.items) == 2
        assert result.total_count == 2
        assert result.has_more is False

    def test_query_with_filter(self, api, mock_store):
        mock_store.query_with_filter.return_value = [{"result_id": "r1"}]
        qf = QueryFilter(model_id="model-1")
        result = api.query(qf)
        assert len(result.items) == 1

    def test_pagination_offset(self, api, mock_store):
        mock_store.query_with_filter.return_value = [
            {"result_id": f"r{i}"} for i in range(5)
        ]
        result = api.query(limit=2, offset=1)
        assert len(result.items) == 2
        assert result.items[0]["result_id"] == "r1"
        assert result.offset == 1

    def test_has_more_true(self, api, mock_store):
        mock_store.query_with_filter.return_value = [
            {"result_id": f"r{i}"} for i in range(4)
        ]
        result = api.query(limit=2, offset=0)
        assert result.has_more is True

    def test_has_more_false(self, api, mock_store):
        mock_store.query_with_filter.return_value = [
            {"result_id": "r1"},
            {"result_id": "r2"},
        ]
        result = api.query(limit=10, offset=0)
        assert result.has_more is False

    def test_limit_clamped_to_min(self, api, mock_store):
        mock_store.query_with_filter.return_value = []
        result = api.query(limit=0)
        assert result.limit == 1

    def test_limit_clamped_to_max(self, api, mock_store):
        mock_store.query_with_filter.return_value = []
        result = api.query(limit=5000)
        assert result.limit == 1000

    def test_negative_offset_clamped(self, api, mock_store):
        mock_store.query_with_filter.return_value = [{"result_id": "r1"}]
        result = api.query(offset=-5)
        assert result.offset == 0


class TestGetById:
    def test_get_existing(self, api, mock_store):
        mock_store.get_metadata.return_value = {"result_id": "r1"}
        result = api.get_by_id("r1")
        assert result == {"result_id": "r1"}

    def test_get_nonexistent(self, api, mock_store):
        mock_store.get_metadata.return_value = None
        result = api.get_by_id("nonexistent")
        assert result is None


class TestConvenienceMethods:
    def test_query_by_model(self, api, mock_store):
        mock_store.query_with_filter.return_value = []
        api.query_by_model("model-1")
        call_args = mock_store.query_with_filter.call_args
        qf = call_args[1].get("query_filter") or call_args[0][0]
        assert qf.model_id == "model-1"

    def test_query_by_workflow(self, api, mock_store):
        mock_store.query_with_filter.return_value = []
        api.query_by_workflow("wf-1")
        call_args = mock_store.query_with_filter.call_args
        qf = call_args[1].get("query_filter") or call_args[0][0]
        assert qf.workflow_id == "wf-1"

    def test_query_by_type(self, api, mock_store):
        mock_store.query_with_filter.return_value = []
        api.query_by_type("baseline_evaluation")
        call_args = mock_store.query_with_filter.call_args
        qf = call_args[1].get("query_filter") or call_args[0][0]
        assert qf.result_type == "baseline_evaluation"
