"""Tests for S3ResultsStore module."""

import gzip
import json
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from src.data_models.storage import StoragePathConfig
from src.orchestration.checksum_calculator import calculate_checksum
from src.storage.s3_results_store import S3ResultsStore


@pytest.fixture
def config():
    return StoragePathConfig(bucket="test-bucket", prefix="trustops")


@pytest.fixture
def mock_s3():
    return MagicMock()


@pytest.fixture
def store(mock_s3, config):
    return S3ResultsStore(mock_s3, config)


class TestStore:
    def test_store_returns_uri_checksum_size(self, store, mock_s3):
        data = {"score": 0.95, "model": "test-model"}
        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        s3_uri, checksum, size_bytes = store.store(
            data, "wf-1", "baseline_evaluation", "res-001", ts
        )

        assert s3_uri.startswith("s3://test-bucket/trustops/wf-1/baseline_evaluation/")
        assert s3_uri.endswith("res-001.json.gz")
        assert len(checksum) == 64  # SHA-256 hex
        assert size_bytes > 0
        mock_s3.put_object.assert_called_once()

    def test_store_compresses_data(self, store, mock_s3):
        data = {"key": "value"}
        store.store(data, "wf-1", "baseline_evaluation", "res-001")

        call_args = mock_s3.put_object.call_args
        body = call_args[1]["Body"]
        # Verify it's valid gzip
        decompressed = gzip.decompress(body)
        parsed = json.loads(decompressed)
        assert parsed == data

    def test_store_sets_metadata(self, store, mock_s3):
        store.store({"x": 1}, "wf-1", "comparison", "res-002")

        call_args = mock_s3.put_object.call_args
        metadata = call_args[1]["Metadata"]
        assert metadata["result_id"] == "res-002"
        assert metadata["artifact_type"] == "comparison"
        assert metadata["workflow_id"] == "wf-1"

    def test_store_content_type_is_gzip(self, store, mock_s3):
        store.store({"x": 1}, "wf-1", "comparison", "res-002")
        call_args = mock_s3.put_object.call_args
        assert call_args[1]["ContentType"] == "application/gzip"


class TestGet:
    def test_get_decompresses_data(self, store, mock_s3):
        original = {"score": 0.85}
        compressed = gzip.compress(json.dumps(original, sort_keys=True).encode("utf-8"))

        mock_s3.get_object.return_value = {
            "Body": BytesIO(compressed),
        }

        data, checksum = store.get("s3://test-bucket/trustops/wf-1/eval/ts/res.json.gz")
        assert data == original
        assert checksum == calculate_checksum(compressed)

    def test_get_invalid_uri_raises(self, store):
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            store.get("http://invalid")


class TestGetVersion:
    def test_get_version_passes_version_id(self, store, mock_s3):
        compressed = gzip.compress(json.dumps({"v": 1}).encode("utf-8"))
        mock_s3.get_object.return_value = {"Body": BytesIO(compressed)}

        data, _ = store.get_version(
            "s3://test-bucket/trustops/wf-1/eval/ts/res.json.gz", "ver-abc"
        )
        assert data == {"v": 1}
        call_args = mock_s3.get_object.call_args
        assert call_args[1]["VersionId"] == "ver-abc"


class TestListVersions:
    def test_list_versions(self, store, mock_s3):
        mock_s3.list_object_versions.return_value = {
            "Versions": [
                {
                    "VersionId": "v1",
                    "LastModified": "2024-01-01",
                    "Size": 100,
                    "IsLatest": True,
                },
                {
                    "VersionId": "v0",
                    "LastModified": "2023-12-01",
                    "Size": 90,
                    "IsLatest": False,
                },
            ]
        }

        versions = store.list_versions("s3://test-bucket/key/path")
        assert len(versions) == 2
        assert versions[0]["version_id"] == "v1"
        assert versions[0]["is_latest"] is True


class TestDelete:
    def test_delete_calls_s3(self, store, mock_s3):
        store.delete("s3://test-bucket/some/key")
        mock_s3.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="some/key"
        )


class TestChangeStorageClass:
    def test_change_storage_class(self, store, mock_s3):
        store.change_storage_class("s3://test-bucket/some/key", "GLACIER")
        mock_s3.copy_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="some/key",
            CopySource={"Bucket": "test-bucket", "Key": "some/key"},
            StorageClass="GLACIER",
        )
