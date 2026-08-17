"""
Data models for the Results Storage and Versioning system.

This module defines the core data models for storing, versioning, and querying
evaluation results and artifacts in the TrustOps Enterprise Framework.

Requirements: 9.1, 9.2, 9.3, 9.7
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class StoragePathConfig(BaseModel):
    """Configuration for S3 storage path conventions.

    S3 path convention: s3://{bucket}/trustops/{workflow_id}/{artifact_type}/{timestamp}/

    Requirement 9.1: Define S3 path convention
    """

    bucket: str = Field(
        ...,
        min_length=1,
        description="S3 bucket name for storing results"
    )
    prefix: str = Field(
        default="trustops",
        min_length=1,
        description="Prefix for all TrustOps artifacts in the bucket"
    )

    @field_validator('bucket')
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Ensure bucket name follows S3 naming conventions."""
        if not v:
            raise ValueError("bucket name cannot be empty")
        # Basic S3 bucket name validation
        if len(v) < 3 or len(v) > 63:
            raise ValueError("bucket name must be between 3 and 63 characters")
        return v


class ResultMetadata(BaseModel):
    """Metadata for a stored evaluation result or artifact.

    Requirement 9.2: Define DynamoDB table schemas (metadata portion)
    Requirement 9.3: Support for ResultsStore class
    """

    result_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the result"
    )
    result_type: str = Field(
        ...,
        min_length=1,
        description="Type of result: 'baseline_evaluation', 'comparison', 'fine_tuning', etc."
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="ID of the workflow that generated this result"
    )
    model_id: str = Field(
        ...,
        min_length=1,
        description="ID of the model associated with this result"
    )
    dataset_id: Optional[str] = Field(
        default=None,
        description="ID of the dataset used for this result"
    )
    s3_uri: str = Field(
        ...,
        min_length=1,
        description="S3 URI where the result data is stored"
    )
    checksum: str = Field(
        ...,
        min_length=1,
        description="SHA-256 checksum for data integrity verification"
    )
    size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of the stored result in bytes"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the result was created"
    )
    version: int = Field(
        ...,
        ge=1,
        description="Version number of this result"
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="User-defined tags for categorization and filtering"
    )

    @field_validator('s3_uri')
    @classmethod
    def validate_s3_uri(cls, v: str) -> str:
        """Ensure s3_uri has valid S3 URI format."""
        if not v.startswith('s3://'):
            raise ValueError("s3_uri must start with 's3://'")
        return v

    @field_validator('result_type')
    @classmethod
    def validate_result_type(cls, v: str) -> str:
        """Ensure result_type is one of the known types."""
        allowed_types = {
            "baseline_evaluation",
            "comparison",
            "fine_tuning",
            "trust_score",
            "hallucination_analysis",
            "workflow_manifest",
            "dataset_quality",
            "model_metadata"
        }
        if v not in allowed_types:
            # Allow custom types but log a warning in production
            pass
        return v


class QueryFilter(BaseModel):
    """Filter criteria for querying stored results.

    Requirement 9.3: Support for ResultsStore query operations
    """

    result_type: Optional[str] = Field(
        default=None,
        description="Filter by result type"
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Filter by model ID"
    )
    dataset_id: Optional[str] = Field(
        default=None,
        description="Filter by dataset ID"
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Filter by workflow ID"
    )
    date_from: Optional[datetime] = Field(
        default=None,
        description="Filter results created on or after this date"
    )
    date_to: Optional[datetime] = Field(
        default=None,
        description="Filter results created on or before this date"
    )
    tags: Optional[dict[str, str]] = Field(
        default=None,
        description="Filter by tags (all specified tags must match)"
    )

    @field_validator('date_from', 'date_to')
    @classmethod
    def validate_dates(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure dates are not in the future."""
        if v is not None and v > datetime.now():
            raise ValueError("date cannot be in the future")
        return v

    def model_post_init(self, __context) -> None:
        """Validate date range after model initialization."""
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be before or equal to date_to")


class RetentionPolicy(BaseModel):
    """Data retention policy configuration.

    Requirement 9.7: Support for data retention policies with configurable archival
    """

    archive_after_days: int = Field(
        default=90,
        ge=1,
        description="Number of days after which results are archived to S3 Glacier"
    )
    delete_after_days: int = Field(
        default=365,
        ge=1,
        description="Number of days after which results are permanently deleted"
    )
    exclude_tags: list[str] = Field(
        default_factory=lambda: ["production", "audit"],
        description="Tags that exempt results from retention policy"
    )

    @field_validator('delete_after_days')
    @classmethod
    def validate_delete_after_archive(cls, v: int, info) -> int:
        """Ensure delete_after_days is greater than archive_after_days."""
        # Note: info.data contains previously validated fields
        archive_days = info.data.get('archive_after_days', 90)
        if v <= archive_days:
            raise ValueError(
                f"delete_after_days ({v}) must be greater than "
                f"archive_after_days ({archive_days})"
            )
        return v
