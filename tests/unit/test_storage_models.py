"""
Unit tests for storage data models.

Tests Requirements: 9.1, 9.2, 9.3, 9.7
"""
import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from src.data_models.storage import (
    StoragePathConfig,
    ResultMetadata,
    QueryFilter,
    RetentionPolicy,
)


class TestStoragePathConfig:
    """Test StoragePathConfig data model."""

    def test_storage_path_config_valid_creation(self):
        """Test creating StoragePathConfig with valid data."""
        config = StoragePathConfig(
            bucket="my-trustops-bucket",
            prefix="trustops"
        )

        assert config.bucket == "my-trustops-bucket"
        assert config.prefix == "trustops"

    def test_storage_path_config_custom_prefix(self):
        """Test creating StoragePathConfig with custom prefix."""
        config = StoragePathConfig(
            bucket="my-bucket",
            prefix="custom/path"
        )

        assert config.bucket == "my-bucket"
        assert config.prefix == "custom/path"

    def test_storage_path_config_default_prefix(self):
        """Test that default prefix is 'trustops'."""
        config = StoragePathConfig(bucket="my-bucket")

        assert config.prefix == "trustops"

    def test_storage_path_config_empty_bucket_rejected(self):
        """Test that empty bucket name is rejected."""
        with pytest.raises(ValidationError):
            StoragePathConfig(bucket="")

    def test_storage_path_config_bucket_too_short_rejected(self):
        """Test that bucket name shorter than 3 chars is rejected."""
        with pytest.raises(ValidationError):
            StoragePathConfig(bucket="ab")

    def test_storage_path_config_bucket_too_long_rejected(self):
        """Test that bucket name longer than 63 chars is rejected."""
        with pytest.raises(ValidationError):
            StoragePathConfig(bucket="a" * 64)


class TestResultMetadata:
    """Test ResultMetadata data model."""

    def test_result_metadata_valid_creation(self):
        """Test creating ResultMetadata with valid data."""
        now = datetime.now()
        metadata = ResultMetadata(
            result_id="eval-001",
            result_type="baseline_evaluation",
            workflow_id="wf-123",
            model_id="model-abc",
            dataset_id="ds-456",
            s3_uri=(
                "s3://bucket/trustops/wf-123/"
                "baseline_evaluation/2024-01-01/"
            ),
            checksum="abc123def456",
            size_bytes=1024000,
            created_at=now,
            version=1,
            tags={"env": "production", "team": "ml"}
        )

        assert metadata.result_id == "eval-001"
        assert metadata.result_type == "baseline_evaluation"
        assert metadata.workflow_id == "wf-123"
        assert metadata.model_id == "model-abc"
        assert metadata.dataset_id == "ds-456"
        expected_uri = (
            "s3://bucket/trustops/wf-123/baseline_evaluation/2024-01-01/"
        )
        assert metadata.s3_uri == expected_uri
        assert metadata.checksum == "abc123def456"
        assert metadata.size_bytes == 1024000
        assert metadata.created_at == now
        assert metadata.version == 1
        assert metadata.tags == {"env": "production", "team": "ml"}

    def test_result_metadata_minimal_creation(self):
        """Test creating ResultMetadata with minimal required fields."""
        now = datetime.now()
        metadata = ResultMetadata(
            result_id="eval-002",
            result_type="comparison",
            model_id="model-xyz",
            s3_uri="s3://bucket/results/eval-002/",
            checksum="checksum123",
            size_bytes=512000,
            created_at=now,
            version=1
        )

        assert metadata.result_id == "eval-002"
        assert metadata.workflow_id is None
        assert metadata.dataset_id is None
        assert metadata.tags == {}

    def test_result_metadata_invalid_s3_uri_rejected(self):
        """Test that invalid S3 URI is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            ResultMetadata(
                result_id="eval-003",
                result_type="fine_tuning",
                model_id="model-123",
                s3_uri="http://bucket/results/",
                checksum="checksum",
                size_bytes=1000,
                created_at=now,
                version=1
            )

    def test_result_metadata_negative_size_rejected(self):
        """Test that negative size_bytes is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            ResultMetadata(
                result_id="eval-004",
                result_type="trust_score",
                model_id="model-456",
                s3_uri="s3://bucket/results/",
                checksum="checksum",
                size_bytes=-100,
                created_at=now,
                version=1
            )

    def test_result_metadata_zero_version_rejected(self):
        """Test that version less than 1 is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            ResultMetadata(
                result_id="eval-005",
                result_type="hallucination_analysis",
                model_id="model-789",
                s3_uri="s3://bucket/results/",
                checksum="checksum",
                size_bytes=1000,
                created_at=now,
                version=0
            )

    def test_result_metadata_known_result_types(self):
        """Test creating ResultMetadata with all known result types."""
        now = datetime.now()
        known_types = [
            "baseline_evaluation",
            "comparison",
            "fine_tuning",
            "trust_score",
            "hallucination_analysis",
            "workflow_manifest",
            "dataset_quality",
            "model_metadata"
        ]

        for result_type in known_types:
            metadata = ResultMetadata(
                result_id=f"eval-{result_type}",
                result_type=result_type,
                model_id="model-test",
                s3_uri="s3://bucket/results/",
                checksum="checksum",
                size_bytes=1000,
                created_at=now,
                version=1
            )
            assert metadata.result_type == result_type

    def test_result_metadata_custom_result_type(self):
        """Test that custom result types are allowed."""
        now = datetime.now()
        metadata = ResultMetadata(
            result_id="eval-custom",
            result_type="custom_analysis",
            model_id="model-test",
            s3_uri="s3://bucket/results/",
            checksum="checksum",
            size_bytes=1000,
            created_at=now,
            version=1
        )
        assert metadata.result_type == "custom_analysis"

    def test_result_metadata_serialization(self):
        """Test ResultMetadata JSON serialization."""
        now = datetime.now()
        metadata = ResultMetadata(
            result_id="eval-serialize",
            result_type="baseline_evaluation",
            model_id="model-serialize",
            s3_uri="s3://bucket/results/",
            checksum="checksum123",
            size_bytes=2048,
            created_at=now,
            version=2,
            tags={"test": "value"}
        )

        # Serialize to dict
        data = metadata.model_dump()
        assert data["result_id"] == "eval-serialize"
        assert data["result_type"] == "baseline_evaluation"
        assert data["version"] == 2
        assert data["tags"] == {"test": "value"}

        # Serialize to JSON
        json_str = metadata.model_dump_json()
        assert "eval-serialize" in json_str
        assert "baseline_evaluation" in json_str


class TestQueryFilter:
    """Test QueryFilter data model."""

    def test_query_filter_empty_creation(self):
        """Test creating QueryFilter with no filters."""
        filter = QueryFilter()

        assert filter.result_type is None
        assert filter.model_id is None
        assert filter.dataset_id is None
        assert filter.workflow_id is None
        assert filter.date_from is None
        assert filter.date_to is None
        assert filter.tags is None

    def test_query_filter_with_all_fields(self):
        """Test creating QueryFilter with all fields."""
        date_from = datetime.now() - timedelta(days=30)
        date_to = datetime.now()

        filter = QueryFilter(
            result_type="baseline_evaluation",
            model_id="model-123",
            dataset_id="ds-456",
            workflow_id="wf-789",
            date_from=date_from,
            date_to=date_to,
            tags={"env": "production"}
        )

        assert filter.result_type == "baseline_evaluation"
        assert filter.model_id == "model-123"
        assert filter.dataset_id == "ds-456"
        assert filter.workflow_id == "wf-789"
        assert filter.date_from == date_from
        assert filter.date_to == date_to
        assert filter.tags == {"env": "production"}

    def test_query_filter_partial_fields(self):
        """Test creating QueryFilter with partial fields."""
        filter = QueryFilter(
            result_type="comparison",
            model_id="model-abc"
        )

        assert filter.result_type == "comparison"
        assert filter.model_id == "model-abc"
        assert filter.dataset_id is None
        assert filter.workflow_id is None

    def test_query_filter_date_range_validation(self):
        """Test that date_from must be before date_to."""
        date_from = datetime.now()
        date_to = datetime.now() - timedelta(days=1)

        with pytest.raises(ValidationError):
            QueryFilter(
                date_from=date_from,
                date_to=date_to
            )

    def test_query_filter_future_date_rejected(self):
        """Test that future dates are rejected."""
        future_date = datetime.now() + timedelta(days=1)

        with pytest.raises(ValidationError):
            QueryFilter(date_from=future_date)

        with pytest.raises(ValidationError):
            QueryFilter(date_to=future_date)

    def test_query_filter_valid_date_range(self):
        """Test creating QueryFilter with valid date range."""
        date_from = datetime.now() - timedelta(days=7)
        date_to = datetime.now()

        filter = QueryFilter(
            date_from=date_from,
            date_to=date_to
        )

        assert filter.date_from == date_from
        assert filter.date_to == date_to

    def test_query_filter_same_date_from_and_to(self):
        """Test that date_from can equal date_to."""
        same_date = datetime.now() - timedelta(days=1)

        filter = QueryFilter(
            date_from=same_date,
            date_to=same_date
        )

        assert filter.date_from == same_date
        assert filter.date_to == same_date

    def test_query_filter_with_multiple_tags(self):
        """Test creating QueryFilter with multiple tags."""
        filter = QueryFilter(
            tags={
                "env": "production",
                "team": "ml",
                "project": "trustops"
            }
        )

        assert len(filter.tags) == 3
        assert filter.tags["env"] == "production"
        assert filter.tags["team"] == "ml"
        assert filter.tags["project"] == "trustops"


class TestRetentionPolicy:
    """Test RetentionPolicy data model."""

    def test_retention_policy_default_values(self):
        """Test RetentionPolicy with default values."""
        policy = RetentionPolicy()

        assert policy.archive_after_days == 90
        assert policy.delete_after_days == 365
        assert policy.exclude_tags == ["production", "audit"]

    def test_retention_policy_custom_values(self):
        """Test creating RetentionPolicy with custom values."""
        policy = RetentionPolicy(
            archive_after_days=30,
            delete_after_days=180,
            exclude_tags=["critical", "compliance"]
        )

        assert policy.archive_after_days == 30
        assert policy.delete_after_days == 180
        assert policy.exclude_tags == ["critical", "compliance"]

    def test_retention_policy_delete_must_be_after_archive(self):
        """Test that delete_after_days must be greater than
        archive_after_days."""
        with pytest.raises(ValidationError):
            RetentionPolicy(
                archive_after_days=100,
                delete_after_days=50
            )

    def test_retention_policy_delete_equal_to_archive_rejected(self):
        """Test that delete_after_days cannot equal
        archive_after_days."""
        with pytest.raises(ValidationError):
            RetentionPolicy(
                archive_after_days=90,
                delete_after_days=90
            )

    def test_retention_policy_valid_ordering(self):
        """Test creating RetentionPolicy with valid day ordering."""
        policy = RetentionPolicy(
            archive_after_days=60,
            delete_after_days=365
        )

        assert policy.archive_after_days == 60
        assert policy.delete_after_days == 365

    def test_retention_policy_zero_days_rejected(self):
        """Test that zero or negative days are rejected."""
        with pytest.raises(ValidationError):
            RetentionPolicy(
                archive_after_days=0,
                delete_after_days=100
            )

        with pytest.raises(ValidationError):
            RetentionPolicy(
                archive_after_days=30,
                delete_after_days=0
            )

    def test_retention_policy_negative_days_rejected(self):
        """Test that negative days are rejected."""
        with pytest.raises(ValidationError):
            RetentionPolicy(
                archive_after_days=-10,
                delete_after_days=100
            )

    def test_retention_policy_empty_exclude_tags(self):
        """Test creating RetentionPolicy with empty exclude tags."""
        policy = RetentionPolicy(
            archive_after_days=30,
            delete_after_days=90,
            exclude_tags=[]
        )

        assert policy.exclude_tags == []

    def test_retention_policy_serialization(self):
        """Test RetentionPolicy JSON serialization."""
        policy = RetentionPolicy(
            archive_after_days=45,
            delete_after_days=200,
            exclude_tags=["important", "legal"]
        )

        # Serialize to dict
        data = policy.model_dump()
        assert data["archive_after_days"] == 45
        assert data["delete_after_days"] == 200
        assert data["exclude_tags"] == ["important", "legal"]

        # Serialize to JSON
        json_str = policy.model_dump_json()
        assert "45" in json_str
        assert "200" in json_str
        assert "important" in json_str
