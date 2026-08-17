"""
Unit tests for dataset data models.

Tests Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from src.data_models.dataset import (
    DatasetFormat,
    DatasetTaskType,
    TokenStatistics,
    DatasetLineage,
    QualityIssue,
    DatasetQualityReport,
    DatasetMetadata,
)


class TestDatasetEnums:
    """Test dataset enumeration types."""

    def test_dataset_format_enum_values(self):
        """Test DatasetFormat enum has all required values."""
        assert DatasetFormat.JSONL == "jsonl"
        assert DatasetFormat.CSV == "csv"
        assert DatasetFormat.PARQUET == "parquet"
        assert DatasetFormat.HUGGINGFACE == "huggingface"

    def test_dataset_task_type_enum_values(self):
        """Test DatasetTaskType enum has all required values."""
        assert DatasetTaskType.QA == "qa"
        assert DatasetTaskType.SUMMARIZATION == "summarization"
        assert DatasetTaskType.CLASSIFICATION == "classification"
        assert DatasetTaskType.TEXT_GENERATION == "text_generation"
        assert DatasetTaskType.CHAT == "chat"
        assert DatasetTaskType.CUSTOM == "custom"


class TestTokenStatistics:
    """Test TokenStatistics data model."""

    def test_token_statistics_valid_creation(self):
        """Test creating TokenStatistics with valid data."""
        stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=500,
            avg_tokens=150.5,
            p95_tokens=450,
            prompt_tokens=6000,
            completion_tokens=4000,
        )

        assert stats.total_tokens == 10000
        assert stats.min_tokens == 10
        assert stats.max_tokens == 500
        assert stats.avg_tokens == 150.5
        assert stats.p95_tokens == 450
        assert stats.prompt_tokens == 6000
        assert stats.completion_tokens == 4000

    def test_token_statistics_negative_values_rejected(self):
        """Test that negative token counts are rejected."""
        with pytest.raises(ValidationError):
            TokenStatistics(
                total_tokens=-100,
                min_tokens=10,
                max_tokens=500,
                avg_tokens=150.5,
                p95_tokens=450,
                prompt_tokens=6000,
                completion_tokens=4000,
            )

    def test_token_statistics_negative_avg_rejected(self):
        """Test that negative average tokens are rejected."""
        with pytest.raises(ValidationError):
            TokenStatistics(
                total_tokens=10000,
                min_tokens=10,
                max_tokens=500,
                avg_tokens=-150.5,
                p95_tokens=450,
                prompt_tokens=6000,
                completion_tokens=4000,
            )


class TestDatasetLineage:
    """Test DatasetLineage data model."""

    def test_dataset_lineage_valid_creation(self):
        """Test creating DatasetLineage with valid data."""
        now = datetime.now()
        lineage = DatasetLineage(
            parent_dataset_id="parent-123",
            transformations=["split", "augment"],
            source_files=["data1.csv", "data2.csv"],
            created_by="user@example.com",
            created_at=now,
        )

        assert lineage.parent_dataset_id == "parent-123"
        assert lineage.transformations == ["split", "augment"]
        assert lineage.source_files == ["data1.csv", "data2.csv"]
        assert lineage.created_by == "user@example.com"
        assert lineage.created_at == now

    def test_dataset_lineage_no_parent(self):
        """Test creating DatasetLineage without parent dataset."""
        now = datetime.now()
        lineage = DatasetLineage(
            parent_dataset_id=None,
            transformations=[],
            source_files=["original.csv"],
            created_by="user@example.com",
            created_at=now,
        )

        assert lineage.parent_dataset_id is None
        assert lineage.transformations == []
        assert lineage.source_files == ["original.csv"]

    def test_dataset_lineage_empty_created_by_rejected(self):
        """Test that empty created_by is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            DatasetLineage(
                parent_dataset_id=None,
                transformations=[],
                source_files=["original.csv"],
                created_by="",
                created_at=now,
            )


class TestQualityIssue:
    """Test QualityIssue data model."""

    def test_quality_issue_valid_creation(self):
        """Test creating QualityIssue with valid data."""
        issue = QualityIssue(
            severity="error",
            category="missing_field",
            message="Missing required field 'prompt'",
            affected_rows=[1, 5, 10],
        )

        assert issue.severity == "error"
        assert issue.category == "missing_field"
        assert issue.message == "Missing required field 'prompt'"
        assert issue.affected_rows == [1, 5, 10]

    def test_quality_issue_severity_normalization(self):
        """Test that severity is normalized to lowercase."""
        issue = QualityIssue(
            severity="ERROR",
            category="test",
            message="Test message",
        )

        assert issue.severity == "error"

    def test_quality_issue_invalid_severity_rejected(self):
        """Test that invalid severity values are rejected."""
        with pytest.raises(ValidationError):
            QualityIssue(
                severity="critical",
                category="test",
                message="Test message",
            )

    def test_quality_issue_no_affected_rows(self):
        """Test creating QualityIssue without affected rows."""
        issue = QualityIssue(
            severity="warning",
            category="low_diversity",
            message="Dataset has low diversity",
            affected_rows=None,
        )

        assert issue.affected_rows is None


class TestDatasetQualityReport:
    """Test DatasetQualityReport data model."""

    def test_quality_report_valid_creation(self):
        """Test creating DatasetQualityReport with valid data."""
        token_stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=500,
            avg_tokens=150.5,
            p95_tokens=450,
            prompt_tokens=6000,
            completion_tokens=4000,
        )

        issue = QualityIssue(
            severity="warning",
            category="balance",
            message="Imbalanced categories",
        )

        report = DatasetQualityReport(
            completeness_score=0.95,
            diversity_score=0.80,
            balance_score=0.70,
            token_stats=token_stats,
            issues=[issue],
            recommendations=["Add more examples to minority classes"],
        )

        assert report.completeness_score == 0.95
        assert report.diversity_score == 0.80
        assert report.balance_score == 0.70
        assert report.token_stats == token_stats
        assert len(report.issues) == 1
        assert len(report.recommendations) == 1

    def test_quality_report_score_out_of_range_rejected(self):
        """Test that scores outside [0, 1] are rejected."""
        token_stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=500,
            avg_tokens=150.5,
            p95_tokens=450,
            prompt_tokens=6000,
            completion_tokens=4000,
        )

        with pytest.raises(ValidationError):
            DatasetQualityReport(
                completeness_score=1.5,
                diversity_score=0.80,
                balance_score=0.70,
                token_stats=token_stats,
                issues=[],
                recommendations=[],
            )

    def test_quality_report_empty_issues_and_recommendations(self):
        """Test creating report with no issues or recommendations."""
        token_stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=500,
            avg_tokens=150.5,
            p95_tokens=450,
            prompt_tokens=6000,
            completion_tokens=4000,
        )

        report = DatasetQualityReport(
            completeness_score=1.0,
            diversity_score=0.95,
            balance_score=0.90,
            token_stats=token_stats,
            issues=[],
            recommendations=[],
        )

        assert len(report.issues) == 0
        assert len(report.recommendations) == 0


class TestDatasetMetadata:
    """Test DatasetMetadata data model."""

    def test_dataset_metadata_valid_creation(self):
        """Test creating DatasetMetadata with valid data."""
        now = datetime.now()
        token_stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=500,
            avg_tokens=150.5,
            p95_tokens=450,
            prompt_tokens=6000,
            completion_tokens=4000,
        )

        metadata = DatasetMetadata(
            id="ds-001",
            name="Test Dataset",
            description="A test dataset for evaluation",
            format=DatasetFormat.JSONL,
            task_type=DatasetTaskType.QA,
            version="1.0.0",
            s3_uri="s3://bucket/datasets/test.jsonl",
            row_count=100,
            token_stats=token_stats,
            created_at=now,
            checksum="abc123def456",
        )

        assert metadata.id == "ds-001"
        assert metadata.name == "Test Dataset"
        assert metadata.description == "A test dataset for evaluation"
        assert metadata.format == DatasetFormat.JSONL
        assert metadata.task_type == DatasetTaskType.QA
        assert metadata.version == "1.0.0"
        assert metadata.s3_uri == "s3://bucket/datasets/test.jsonl"
        assert metadata.row_count == 100
        assert metadata.token_stats == token_stats
        assert metadata.created_at == now
        assert metadata.checksum == "abc123def456"

    def test_dataset_metadata_with_lineage_and_quality_report(self):
        """Test creating DatasetMetadata with lineage and quality report."""
        now = datetime.now()
        token_stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=500,
            avg_tokens=150.5,
            p95_tokens=450,
            prompt_tokens=6000,
            completion_tokens=4000,
        )

        lineage = DatasetLineage(
            parent_dataset_id="parent-001",
            transformations=["split"],
            source_files=["original.csv"],
            created_by="user@example.com",
            created_at=now,
        )

        quality_report = DatasetQualityReport(
            completeness_score=0.95,
            diversity_score=0.80,
            balance_score=0.70,
            token_stats=token_stats,
            issues=[],
            recommendations=[],
        )

        metadata = DatasetMetadata(
            id="ds-002",
            name="Derived Dataset",
            format=DatasetFormat.CSV,
            task_type=DatasetTaskType.CLASSIFICATION,
            version="1.0.0",
            s3_uri="s3://bucket/datasets/derived.csv",
            row_count=50,
            token_stats=token_stats,
            created_at=now,
            checksum="xyz789",
            lineage=lineage,
            quality_report=quality_report,
        )

        assert metadata.lineage == lineage
        assert metadata.quality_report == quality_report

    def test_dataset_metadata_invalid_s3_uri_rejected(self):
        """Test that invalid S3 URI is rejected."""
        now = datetime.now()
        token_stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=500,
            avg_tokens=150.5,
            p95_tokens=450,
            prompt_tokens=6000,
            completion_tokens=4000,
        )

        with pytest.raises(ValidationError):
            DatasetMetadata(
                id="ds-003",
                name="Invalid Dataset",
                format=DatasetFormat.JSONL,
                task_type=DatasetTaskType.QA,
                version="1.0.0",
                s3_uri="http://bucket/datasets/test.jsonl",
                row_count=100,
                token_stats=token_stats,
                created_at=now,
                checksum="abc123",
            )

    def test_dataset_metadata_negative_row_count_rejected(self):
        """Test that negative row count is rejected."""
        now = datetime.now()
        token_stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=500,
            avg_tokens=150.5,
            p95_tokens=450,
            prompt_tokens=6000,
            completion_tokens=4000,
        )

        with pytest.raises(ValidationError):
            DatasetMetadata(
                id="ds-004",
                name="Invalid Dataset",
                format=DatasetFormat.JSONL,
                task_type=DatasetTaskType.QA,
                version="1.0.0",
                s3_uri="s3://bucket/datasets/test.jsonl",
                row_count=-10,
                token_stats=token_stats,
                created_at=now,
                checksum="abc123",
            )

    def test_dataset_metadata_serialization(self):
        """Test DatasetMetadata JSON serialization."""
        now = datetime.now()
        token_stats = TokenStatistics(
            total_tokens=10000,
            min_tokens=10,
            max_tokens=500,
            avg_tokens=150.5,
            p95_tokens=450,
            prompt_tokens=6000,
            completion_tokens=4000,
        )

        metadata = DatasetMetadata(
            id="ds-005",
            name="Serialization Test",
            format=DatasetFormat.PARQUET,
            task_type=DatasetTaskType.SUMMARIZATION,
            version="2.0.0",
            s3_uri="s3://bucket/datasets/test.parquet",
            row_count=200,
            token_stats=token_stats,
            created_at=now,
            checksum="checksum123",
        )

        # Serialize to dict
        data = metadata.model_dump()
        assert data["id"] == "ds-005"
        assert data["name"] == "Serialization Test"
        assert data["format"] == "parquet"
        assert data["task_type"] == "summarization"

        # Serialize to JSON
        json_str = metadata.model_dump_json()
        assert "ds-005" in json_str
        assert "Serialization Test" in json_str
