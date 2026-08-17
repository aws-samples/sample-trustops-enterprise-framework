"""
Data models for the Dataset Manager.

This module defines the core data models for dataset preparation, validation,
quality analysis, and versioning in the TrustOps Enterprise Framework.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DatasetFormat(str, Enum):
    """Enumeration of supported dataset formats.

    Requirement 2.3: Define DatasetFormat enum
    """

    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HUGGINGFACE = "huggingface"


class DatasetTaskType(str, Enum):
    """Enumeration of supported dataset task types.

    Requirement 2.2: Define DatasetTaskType enum
    """

    QA = "qa"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    CUSTOM = "custom"


class TokenStatistics(BaseModel):
    """Token statistics for a dataset.

    Requirement 2.1: Part of DatasetMetadata schema
    """

    total_tokens: int = Field(
        ...,
        ge=0,
        description="Total number of tokens in the dataset"
    )
    min_tokens: int = Field(
        ...,
        ge=0,
        description="Minimum tokens in a single example"
    )
    max_tokens: int = Field(
        ...,
        ge=0,
        description="Maximum tokens in a single example"
    )
    avg_tokens: float = Field(
        ...,
        ge=0,
        description="Average tokens per example"
    )
    p95_tokens: int = Field(
        ...,
        ge=0,
        description="95th percentile of tokens per example"
    )
    prompt_tokens: int = Field(
        ...,
        ge=0,
        description="Total tokens in prompts"
    )
    completion_tokens: int = Field(
        ...,
        ge=0,
        description="Total tokens in completions"
    )

    @field_validator('avg_tokens')
    @classmethod
    def validate_avg_tokens(cls, v: float) -> float:
        """Ensure avg_tokens is non-negative."""
        if v < 0:
            raise ValueError("avg_tokens must be non-negative")
        return v


class DatasetLineage(BaseModel):
    """Lineage information for dataset versioning and tracking.

    Requirement 2.1: Part of DatasetMetadata schema for lineage tracking
    """

    parent_dataset_id: Optional[str] = Field(
        default=None,
        description="ID of the parent dataset if derived"
    )
    transformations: list[str] = Field(
        default_factory=list,
        description="List of transformations applied to create this dataset"
    )
    source_files: list[str] = Field(
        default_factory=list,
        description="Original source files used to create this dataset"
    )
    created_by: str = Field(
        ...,
        min_length=1,
        description="User or system that created this dataset"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when this dataset version was created"
    )


class QualityIssue(BaseModel):
    """A quality issue detected in a dataset.

    Requirement 2.4: Part of DatasetQualityReport schema
    """

    severity: str = Field(
        ...,
        description="Severity level: 'error', 'warning', or 'info'"
    )
    category: str = Field(
        ...,
        min_length=1,
        description="Category of the issue (e.g., 'missing_field')"
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of the issue"
    )
    affected_rows: Optional[list[int]] = Field(
        default=None,
        description="List of row indices affected by this issue"
    )

    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Ensure severity is one of the allowed values."""
        allowed = {"error", "warning", "info"}
        if v.lower() not in allowed:
            raise ValueError(f"severity must be one of {allowed}")
        return v.lower()


class DatasetQualityReport(BaseModel):
    """Quality analysis report for a dataset.

    Requirement 2.4: Define DatasetQualityReport schema
    """

    completeness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score indicating data completeness (0-1)"
    )
    diversity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score indicating data diversity (0-1)"
    )
    balance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score indicating category balance (0-1)"
    )
    token_stats: TokenStatistics = Field(
        ...,
        description="Token statistics for the dataset"
    )
    issues: list[QualityIssue] = Field(
        default_factory=list,
        description="List of quality issues detected"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="List of recommendations for improving dataset quality"
    )

    @field_validator('completeness_score', 'diversity_score', 'balance_score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are within valid range [0, 1]."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v


class DatasetMetadata(BaseModel):
    """Metadata for a registered dataset.

    Requirement 2.1: Define DatasetMetadata pydantic schema
    """

    id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the dataset"
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name of the dataset"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the dataset contents and purpose"
    )
    format: DatasetFormat = Field(
        ...,
        description="Format of the dataset"
    )
    task_type: DatasetTaskType = Field(
        ...,
        description="Task type this dataset is designed for"
    )
    version: str = Field(
        ...,
        min_length=1,
        description="Version identifier for the dataset"
    )
    s3_uri: str = Field(
        ...,
        min_length=1,
        description="S3 URI where the dataset is stored"
    )
    row_count: int = Field(
        ...,
        ge=0,
        description="Number of rows/examples in the dataset"
    )
    token_stats: TokenStatistics = Field(
        ...,
        description="Token statistics for the dataset"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the dataset was created"
    )
    checksum: str = Field(
        ...,
        min_length=1,
        description="SHA-256 checksum for integrity verification"
    )
    lineage: Optional[DatasetLineage] = Field(
        default=None,
        description="Lineage information for tracking dataset provenance"
    )
    quality_report: Optional[DatasetQualityReport] = Field(
        default=None,
        description="Quality analysis report for the dataset"
    )

    @field_validator('s3_uri')
    @classmethod
    def validate_s3_uri(cls, v: str) -> str:
        """Ensure s3_uri has valid S3 URI format."""
        if not v.startswith('s3://'):
            raise ValueError("s3_uri must start with 's3://'")
        return v
