"""Tests for the unified ResultsStore class."""

import gzip
import json
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from src.data_models.storage import QueryFilter, StoragePathConfig
from src.storage.results_store import ResultsStore


@pytest.fixture
def config():
    return StoragePathConfig(bucket="test-bucket", prefix="trustops")


@pytest.fixture
def mock_s3():
    return MagicMock()


@pytest.fixture
def mock_ddb():
    return MagicMock()


@pytest.fixture
def store(mock_s3, mock_ddb, config):
    return ResultsStore(mock_s3, mock_ddb, config)


class TestStoreResult:
    def test_store_returns_metadata(self, store):
        result = store.store_result(
            {"score": 0.95},
            result_type="baseline_evaluation",
            model_id="model-1",
            workflow_id="wf-1",
        )
        assert result.result_type == "baseline_evaluation"
        assert result.model_id == "model-1"
        assert result.workflow_id == "wf-1"
        assert result.version == 1
        assert result.s3_uri.startswith("s3://test-bucket/")
        assert len(result.checksum) == 64

    def test_store_with_tags(self, store):
        result = store.store_result(
            {"x": 1},
            result_type="comparison",
            model_id="m1",
            tags={"env": "prod"},
        )
        assert result.tags == {"env": "prod"}

    def test_store_writes_to_s3(self, store, mock_s3):
        store.store_result({"x": 1}, "baseline_evaluation", "m1")
        mock_s3.put_object.assert_called_once()

    def test_store_writes_to_dynamodb(self, store, mock_ddb):
        store.store_result({"x": 1}, "baseline_evaluation", "m1")
        mock_ddb.put_item.assert_called()


class TestGetResult:
    def test_get_after_store(self, store, mock_s3):
        # Store a result
        metadata = store.store_result({"score": 0.9}, "baseline_evaluation", "m1")

        # Mock S3 get to return the compressed data
        compressed = gzip.compress(json.dumps({"score": 0.9}, sort_keys=True).encode("utf-8"))
        mock_s3.get_object.return_value = {"Body": BytesIO(compressed)}

        data, meta = store.get_result(metadata.result_id)
        assert data == {"score": 0.9}
        assert meta.result_id == metadata.result_id

    def test_get_nonexistent_raises(self, store):
        with pytest.raises(ValueError, match="Result not found"):
            store.get_result("nonexistent")

    def test_get_specific_version(self, store, mock_s3):
        # Store two versions
        meta1 = store.store_result({"v": 1}, "baseline_evaluation", "m1")

        compressed = gzip.compress(json.dumps({"v": 1}, sort_keys=True).encode("utf-8"))
        mock_s3.get_object.return_value = {"Body": BytesIO(compressed)}

        data, meta = store.get_result(meta1.result_id, version=1)
        assert data == {"v": 1}


class TestQueryResults:
    def test_query_delegates_to_api(self, store, mock_ddb):
        mock_ddb.scan.return_value = {"Items": []}
        result = store.query_results()
        assert result.items == []

    def test_query_with_filter(self, store, mock_ddb):
        mock_ddb.query.return_value = {"Items": [{"result_id": "r1", "model_id": "m1"}]}
        qf = QueryFilter(model_id="m1")
        result = store.query_results(qf)
        assert len(result.items) == 1


class TestCompareResults:
    def test_compare_two_results(self, store, mock_s3):
        # Store two results
        m1 = store.store_result({"score": 0.8}, "baseline_evaluation", "m1")
        m2 = store.store_result({"score": 0.9}, "baseline_evaluation", "m2")

        # Mock S3 gets
        def mock_get(Bucket, Key):
            if "m1" in str(m1.result_id) and m1.result_id in Key:
                data = {"score": 0.8}
            else:
                data = {"score": 0.9}
            compressed = gzip.compress(json.dumps(data, sort_keys=True).encode("utf-8"))
            return {"Body": BytesIO(compressed)}

        # Simpler approach: just return different data based on call order
        compressed_1 = gzip.compress(json.dumps({"score": 0.8}, sort_keys=True).encode("utf-8"))
        compressed_2 = gzip.compress(json.dumps({"score": 0.9}, sort_keys=True).encode("utf-8"))
        mock_s3.get_object.side_effect = [
            {"Body": BytesIO(compressed_1)},
            {"Body": BytesIO(compressed_2)},
        ]

        comparison = store.compare_results(m1.result_id, m2.result_id)
        assert comparison.result_id_1 == m1.result_id
        assert comparison.result_id_2 == m2.result_id
        assert "score" in comparison.diff


class TestExportResult:
    def test_export_json(self, store, mock_s3):
        meta = store.store_result({"score": 0.9}, "baseline_evaluation", "m1")

        compressed = gzip.compress(json.dumps({"score": 0.9}, sort_keys=True).encode("utf-8"))
        mock_s3.get_object.return_value = {"Body": BytesIO(compressed)}

        exported = store.export_result(meta.result_id, "json")
        parsed = json.loads(exported)
        assert parsed["score"] == 0.9

    def test_export_csv(self, store, mock_s3):
        meta = store.store_result({"score": 0.9}, "baseline_evaluation", "m1")

        compressed = gzip.compress(json.dumps({"score": 0.9}, sort_keys=True).encode("utf-8"))
        mock_s3.get_object.return_value = {"Body": BytesIO(compressed)}

        exported = store.export_result(meta.result_id, "csv")
        assert b"score" in exported


class TestVerifyChecksum:
    def test_valid_checksum(self, store, mock_s3):
        meta = store.store_result({"x": 1}, "baseline_evaluation", "m1")

        # Get the actual compressed data that was stored
        stored_body = mock_s3.put_object.call_args[1]["Body"]
        mock_s3.get_object.return_value = {"Body": BytesIO(stored_body)}

        assert store.verify_checksum(meta.result_id) is True

    def test_nonexistent_raises(self, store):
        with pytest.raises(ValueError, match="Result not found"):
            store.verify_checksum("nonexistent")


class TestGetVersionHistory:
    def test_empty_history(self, store):
        assert store.get_version_history("nonexistent") == []

    def test_history_after_store(self, store):
        meta = store.store_result({"x": 1}, "baseline_evaluation", "m1")
        history = store.get_version_history(meta.result_id)
        assert len(history) == 1
        assert history[0].version == 1
