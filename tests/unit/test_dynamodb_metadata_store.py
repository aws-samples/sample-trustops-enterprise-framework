"""Tests for DynamoDBMetadataStore module."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, call

import pytest

from src.data_models.storage import QueryFilter, ResultMetadata
from src.storage.dynamodb_metadata_store import DynamoDBMetadataStore


@pytest.fixture
def mock_ddb():
    return MagicMock()


@pytest.fixture
def store(mock_ddb):
    return DynamoDBMetadataStore(mock_ddb, table_name="test-results")


def _make_metadata(**overrides):
    defaults = {
        "result_id": "res-001",
        "result_type": "baseline_evaluation",
        "model_id": "model-1",
        "s3_uri": "s3://bucket/key",
        "checksum": "abc123",
        "size_bytes": 1024,
        "created_at": datetime(2024, 1, 15, tzinfo=timezone.utc),
        "version": 1,
    }
    defaults.update(overrides)
    return ResultMetadata(**defaults)


class TestPutMetadata:
    def test_put_stores_item(self, store, mock_ddb):
        meta = _make_metadata()
        store.put_metadata(meta)

        mock_ddb.put_item.assert_called_once()
        call_args = mock_ddb.put_item.call_args
        item = call_args[1]["Item"]
        assert item["result_id"] == "res-001"
        assert item["result_type"] == "baseline_evaluation"
        assert item["model_id"] == "model-1"
        assert item["version"] == 1

    def test_put_includes_optional_fields(self, store, mock_ddb):
        meta = _make_metadata(workflow_id="wf-1", dataset_id="ds-1")
        store.put_metadata(meta)

        item = mock_ddb.put_item.call_args[1]["Item"]
        assert item["workflow_id"] == "wf-1"
        assert item["dataset_id"] == "ds-1"

    def test_put_omits_none_optional_fields(self, store, mock_ddb):
        meta = _make_metadata()
        store.put_metadata(meta)

        item = mock_ddb.put_item.call_args[1]["Item"]
        assert "workflow_id" not in item
        assert "dataset_id" not in item


class TestGetMetadata:
    def test_get_returns_item(self, store, mock_ddb):
        mock_ddb.get_item.return_value = {"Item": {"result_id": "res-001"}}
        result = store.get_metadata("res-001")
        assert result == {"result_id": "res-001"}

    def test_get_returns_none_when_not_found(self, store, mock_ddb):
        mock_ddb.get_item.return_value = {}
        result = store.get_metadata("nonexistent")
        assert result is None


class TestQueryByResultType:
    def test_query_by_type(self, store, mock_ddb):
        mock_ddb.query.return_value = {"Items": [{"result_id": "r1"}]}
        results = store.query_by_result_type("baseline_evaluation")
        assert len(results) == 1
        call_args = mock_ddb.query.call_args[1]
        assert call_args["IndexName"] == "result-type-index"

    def test_query_with_date_range(self, store, mock_ddb):
        mock_ddb.query.return_value = {"Items": []}
        store.query_by_result_type("comparison", date_from="2024-01-01", date_to="2024-12-31")
        expr = mock_ddb.query.call_args[1]["KeyConditionExpression"]
        assert "BETWEEN" in expr


class TestQueryByModelId:
    def test_query_by_model(self, store, mock_ddb):
        mock_ddb.query.return_value = {"Items": [{"result_id": "r1"}]}
        results = store.query_by_model_id("model-1")
        assert len(results) == 1
        assert mock_ddb.query.call_args[1]["IndexName"] == "model-id-index"


class TestQueryByWorkflowId:
    def test_query_by_workflow(self, store, mock_ddb):
        mock_ddb.query.return_value = {"Items": [{"result_id": "r1"}]}
        results = store.query_by_workflow_id("wf-1")
        assert len(results) == 1
        assert mock_ddb.query.call_args[1]["IndexName"] == "workflow-id-index"


class TestQueryWithFilter:
    def test_filter_by_workflow_uses_gsi(self, store, mock_ddb):
        mock_ddb.query.return_value = {"Items": []}
        qf = QueryFilter(workflow_id="wf-1")
        store.query_with_filter(qf)
        assert mock_ddb.query.call_args[1]["IndexName"] == "workflow-id-index"

    def test_filter_by_model_uses_gsi(self, store, mock_ddb):
        mock_ddb.query.return_value = {"Items": []}
        qf = QueryFilter(model_id="model-1")
        store.query_with_filter(qf)
        assert mock_ddb.query.call_args[1]["IndexName"] == "model-id-index"

    def test_filter_by_tags(self, store, mock_ddb):
        mock_ddb.scan.return_value = {
            "Items": [
                {"result_id": "r1", "tags": {"env": "prod"}},
                {"result_id": "r2", "tags": {"env": "dev"}},
            ]
        }
        qf = QueryFilter(tags={"env": "prod"})
        results = store.query_with_filter(qf)
        assert len(results) == 1
        assert results[0]["result_id"] == "r1"

    def test_pagination_offset_and_limit(self, store, mock_ddb):
        mock_ddb.scan.return_value = {
            "Items": [
                {"result_id": f"r{i}"} for i in range(10)
            ]
        }
        qf = QueryFilter()
        results = store.query_with_filter(qf, limit=3, offset=2)
        assert len(results) == 3
        assert results[0]["result_id"] == "r2"


class TestDeleteMetadata:
    def test_delete(self, store, mock_ddb):
        store.delete_metadata("res-001")
        mock_ddb.delete_item.assert_called_once_with(
            TableName="test-results",
            Key={"result_id": "res-001"},
        )


class TestLogAccess:
    def test_log_access_stores_event(self, store, mock_ddb):
        store.log_access("res-001", "read", "user-1")

        call_args = mock_ddb.put_item.call_args[1]
        assert call_args["TableName"] == "trustops-access-logs"
        item = call_args["Item"]
        assert item["result_id"] == "res-001"
        assert item["action"] == "read"
        assert item["user_id"] == "user-1"
        assert "timestamp" in item
        assert "log_id" in item
