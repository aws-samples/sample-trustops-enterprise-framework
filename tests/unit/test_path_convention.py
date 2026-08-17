"""Tests for S3 path convention module."""

from datetime import datetime, timezone

import pytest

from src.data_models.storage import StoragePathConfig
from src.storage.path_convention import (
    TIMESTAMP_FORMAT,
    ParsedPath,
    build_result_path,
    build_s3_key,
    build_s3_path,
    parse_s3_path,
)


@pytest.fixture
def config():
    return StoragePathConfig(bucket="my-bucket", prefix="trustops")


class TestBuildS3Path:
    def test_basic_path(self, config):
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = build_s3_path(config, "wf-123", "baseline_evaluation", ts)
        assert result == "s3://my-bucket/trustops/wf-123/baseline_evaluation/20240115T103000Z/"

    def test_path_with_filename(self, config):
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = build_s3_path(config, "wf-123", "comparison", ts, "result.json.gz")
        assert result == "s3://my-bucket/trustops/wf-123/comparison/20240115T103000Z/result.json.gz"

    def test_default_timestamp(self, config):
        result = build_s3_path(config, "wf-123", "fine_tuning")
        assert result.startswith("s3://my-bucket/trustops/wf-123/fine_tuning/")
        assert result.endswith("/")

    def test_empty_workflow_id_raises(self, config):
        with pytest.raises(ValueError, match="workflow_id cannot be empty"):
            build_s3_path(config, "", "baseline_evaluation")

    def test_empty_artifact_type_raises(self, config):
        with pytest.raises(ValueError, match="artifact_type cannot be empty"):
            build_s3_path(config, "wf-123", "")

    def test_custom_prefix(self):
        cfg = StoragePathConfig(bucket="bucket", prefix="custom")
        ts = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = build_s3_path(cfg, "wf-1", "trust_score", ts)
        assert result == "s3://bucket/custom/wf-1/trust_score/20240601T000000Z/"


class TestBuildS3Key:
    def test_basic_key(self, config):
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = build_s3_key(config, "wf-123", "baseline_evaluation", ts)
        assert result == "trustops/wf-123/baseline_evaluation/20240115T103000Z/"

    def test_key_with_filename(self, config):
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = build_s3_key(config, "wf-123", "comparison", ts, "data.json.gz")
        assert result == "trustops/wf-123/comparison/20240115T103000Z/data.json.gz"

    def test_empty_workflow_id_raises(self, config):
        with pytest.raises(ValueError):
            build_s3_key(config, "", "baseline_evaluation")


class TestParseS3Path:
    def test_parse_full_path(self):
        uri = "s3://my-bucket/trustops/wf-123/baseline_evaluation/20240115T103000Z/"
        parsed = parse_s3_path(uri)
        assert parsed.bucket == "my-bucket"
        assert parsed.prefix == "trustops"
        assert parsed.workflow_id == "wf-123"
        assert parsed.artifact_type == "baseline_evaluation"
        assert parsed.timestamp == "20240115T103000Z"
        assert parsed.filename is None

    def test_parse_path_with_filename(self):
        uri = "s3://my-bucket/trustops/wf-123/comparison/20240115T103000Z/result.json.gz"
        parsed = parse_s3_path(uri)
        assert parsed.filename == "result.json.gz"

    def test_invalid_prefix_raises(self):
        with pytest.raises(ValueError, match="must start with 's3://'"):
            parse_s3_path("http://bucket/path")

    def test_too_short_path_raises(self):
        with pytest.raises(ValueError, match="Invalid TrustOps S3 path"):
            parse_s3_path("s3://bucket/prefix/wf")

    def test_roundtrip(self, config):
        ts = datetime(2024, 3, 20, 14, 0, 0, tzinfo=timezone.utc)
        uri = build_s3_path(config, "wf-abc", "fine_tuning", ts, "data.json.gz")
        parsed = parse_s3_path(uri)
        assert parsed.bucket == "my-bucket"
        assert parsed.workflow_id == "wf-abc"
        assert parsed.artifact_type == "fine_tuning"
        assert parsed.filename == "data.json.gz"


class TestBuildResultPath:
    def test_result_path(self, config):
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = build_result_path(config, "res-001", "wf-123", "baseline_evaluation", ts)
        assert result == "s3://my-bucket/trustops/wf-123/baseline_evaluation/20240115T103000Z/res-001.json.gz"
