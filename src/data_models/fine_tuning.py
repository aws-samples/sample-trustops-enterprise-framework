"""
Data models for the Fine-Tuning Pipeline.

This module defines the core data models for fine-tuning configuration,
job management, training metrics, and cost estimation in the TrustOps
Enterprise Framework.

Requirements: 4.1, 4.2, 4.5, 4.11
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FineTuningStatus(str, Enum):
    """Enumeration of fine-tuning job statuses.

    Requirement 4.2: Define FineTuningStatus enum
    """

    PENDING = "pending"
    VALIDATING = "validating"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class HyperparameterConfig(BaseModel):
    """Configuration for fine-tuning hyperparameters.

    Requirement 4.1: Define HyperparameterConfig schema
    """

    learning_rate: float = Field(
        default=1e-5,
        gt=0,
        description="Learning rate for training"
    )
    epochs: int = Field(
        default=3,
        gt=0,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        default=8,
        gt=0,
        description="Training batch size"
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="Number of warmup steps for learning rate scheduler"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0,
        description="Weight decay for regularization"
    )
    max_seq_length: int = Field(
        default=2048,
        gt=0,
        description="Maximum sequence length for training"
    )
    lora_rank: Optional[int] = Field(
        default=None,
        gt=0,
        description="LoRA rank for parameter-efficient fine-tuning"
    )
    lora_alpha: Optional[float] = Field(
        default=None,
        gt=0,
        description="LoRA alpha scaling parameter"
    )

    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Ensure learning rate is positive and reasonable."""
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        if v > 1.0:
            raise ValueError("learning_rate should typically be <= 1.0")
        return v


class FineTuningConfig(BaseModel):
    """Configuration for a fine-tuning job.

    Requirement 4.1: Define FineTuningConfig pydantic schema
    """

    base_model_id: str = Field(
        ...,
        min_length=1,
        description="ID of the base model to fine-tune"
    )
    training_data_id: str = Field(
        ...,
        min_length=1,
        description="ID of the training dataset"
    )
    validation_data_id: Optional[str] = Field(
        default=None,
        description="ID of the validation dataset"
    )
    hyperparameters: HyperparameterConfig = Field(
        ...,
        description="Hyperparameter configuration for training"
    )
    job_name: str = Field(
        ...,
        min_length=1,
        description="Name for the fine-tuning job"
    )
    output_model_name: str = Field(
        ...,
        min_length=1,
        description="Name for the fine-tuned model"
    )
    auto_hyperparameter_tuning: bool = Field(
        default=False,
        description="Whether to enable automated hyperparameter tuning"
    )


class TrainingMetrics(BaseModel):
    """Training metrics captured during fine-tuning.

    Requirement 4.2: Define TrainingMetrics schema
    """

    epoch: int = Field(
        ...,
        ge=0,
        description="Current training epoch"
    )
    step: int = Field(
        ...,
        ge=0,
        description="Current training step"
    )
    training_loss: float = Field(
        ...,
        ge=0,
        description="Training loss at this step"
    )
    validation_loss: Optional[float] = Field(
        default=None,
        ge=0,
        description="Validation loss at this step"
    )
    learning_rate: float = Field(
        ...,
        gt=0,
        description="Current learning rate"
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp when these metrics were recorded"
    )


class CostEstimate(BaseModel):
    """Cost estimate for a fine-tuning job.

    Requirement 4.11: Define CostEstimate schema
    """

    estimated_training_cost: float = Field(
        ...,
        ge=0,
        description="Estimated cost for training"
    )
    estimated_duration_hours: float = Field(
        ...,
        ge=0,
        description="Estimated training duration in hours"
    )
    cost_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of costs by component"
    )
    currency: str = Field(
        default="USD",
        description="Currency for cost values"
    )
    confidence: str = Field(
        ...,
        description="Confidence level: 'high', 'medium', or 'low'"
    )

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        """Ensure confidence is one of the allowed values."""
        allowed = {"high", "medium", "low"}
        if v.lower() not in allowed:
            raise ValueError(f"confidence must be one of {allowed}")
        return v.lower()


class FineTuningJob(BaseModel):
    """Fine-tuning job metadata and status.

    Requirement 4.2: Define FineTuningJob schema
    """

    job_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the fine-tuning job"
    )
    status: FineTuningStatus = Field(
        ...,
        description="Current status of the fine-tuning job"
    )
    base_model_id: str = Field(
        ...,
        min_length=1,
        description="ID of the base model being fine-tuned"
    )
    training_data_s3_uri: str = Field(
        ...,
        min_length=1,
        description="S3 URI of the training data"
    )
    validation_data_s3_uri: Optional[str] = Field(
        default=None,
        description="S3 URI of the validation data"
    )
    hyperparameters: HyperparameterConfig = Field(
        ...,
        description="Hyperparameters used for training"
    )
    training_metrics: list[TrainingMetrics] = Field(
        default_factory=list,
        description="Training metrics collected during the job"
    )
    finetuned_model_id: Optional[str] = Field(
        default=None,
        description="ID of the fine-tuned model after completion"
    )
    finetuned_model_arn: Optional[str] = Field(
        default=None,
        description="ARN of the fine-tuned model after completion"
    )
    estimated_cost: float = Field(
        ...,
        ge=0,
        description="Estimated cost for the fine-tuning job"
    )
    actual_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Actual cost after job completion"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the job was created"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when training started"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the job completed"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if the job failed"
    )
    provider_job_id: Optional[str] = Field(
        default=None,
        description="Provider-specific job ID (e.g., Bedrock job ID)"
    )

    @field_validator('training_data_s3_uri', 'validation_data_s3_uri')
    @classmethod
    def validate_s3_uri(cls, v: Optional[str]) -> Optional[str]:
        """Ensure S3 URIs have valid format."""
        if v is not None and not v.startswith('s3://'):
            raise ValueError("S3 URI must start with 's3://'")
        return v
