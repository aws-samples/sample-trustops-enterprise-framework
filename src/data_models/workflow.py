"""
Data models for workflow orchestration and management.

This module defines the core data models for workflow orchestration, including
workflow definitions, steps, events, and manifests for the TrustOps Enterprise
Framework.

Requirements: 8.1, 8.2, 8.3, 8.7
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    """Status of a workflow execution.

    Requirement 8.1: Define WorkflowStatus enum
    """

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Status of a workflow step.

    Requirement 8.2: Define StepStatus enum
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepType(str, Enum):
    """Type of workflow step.

    Requirement 8.2: Define StepType enum
    """

    DATASET_PREPARATION = "dataset_preparation"
    BASELINE_EVALUATION = "baseline_evaluation"
    FINE_TUNING = "fine_tuning"
    POST_TUNING_EVALUATION = "post_tuning_evaluation"
    COMPARISON = "comparison"
    APPROVAL_GATE = "approval_gate"
    CUSTOM = "custom"


class WorkflowStep(BaseModel):
    """Definition of a single workflow step.

    Requirement 8.2: Define WorkflowStep schema
    """

    step_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this step"
    )
    step_type: StepType = Field(
        ...,
        description="Type of workflow step"
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name for this step"
    )
    config: dict = Field(
        default_factory=dict,
        description="Configuration parameters for this step"
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="Current status of this step"
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of step IDs that must complete before this step"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when step execution started"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when step execution completed"
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Duration of step execution in seconds"
    )
    output: Optional[dict] = Field(
        default=None,
        description="Output data from step execution"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if step failed"
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this step has been retried"
    )


class WorkflowDefinition(BaseModel):
    """Definition of a workflow template.

    Requirement 8.1: Define WorkflowDefinition schema
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Name of the workflow"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of what this workflow does"
    )
    steps: list[WorkflowStep] = Field(
        ...,
        min_length=1,
        description="List of steps in this workflow"
    )
    requires_approval: bool = Field(
        default=False,
        description="Whether this workflow requires approval gates"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for failed steps"
    )
    retry_delay_seconds: int = Field(
        default=60,
        ge=0,
        description="Delay in seconds between retries"
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum time allowed for workflow execution"
    )


class WorkflowEvent(BaseModel):
    """Event in a workflow execution.

    Requirement 8.3: Define WorkflowEvent schema
    """

    event_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this event"
    )
    workflow_id: str = Field(
        ...,
        min_length=1,
        description="ID of the workflow this event belongs to"
    )
    step_id: Optional[str] = Field(
        default=None,
        description="ID of the step this event relates to, if applicable"
    )
    event_type: str = Field(
        ...,
        min_length=1,
        description=(
            "Type of event (started, completed, failed, retrying, etc.)"
        )
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp when the event occurred"
    )
    details: dict = Field(
        default_factory=dict,
        description="Additional details about the event"
    )


class ArtifactReference(BaseModel):
    """Reference to an artifact produced by a workflow.

    Requirement 8.7: Define ArtifactReference schema
    """

    artifact_type: str = Field(
        ...,
        min_length=1,
        description=(
            "Type of artifact (dataset, evaluation_report, model, etc.)"
        )
    )
    s3_uri: str = Field(
        ...,
        min_length=1,
        description="S3 URI where the artifact is stored"
    )
    checksum: str = Field(
        ...,
        min_length=1,
        description="Checksum for artifact integrity verification"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the artifact was created"
    )


class WorkflowManifest(BaseModel):
    """Complete workflow execution manifest with lineage and audit trail.

    Requirement 8.3: Define WorkflowManifest schema
    """

    workflow_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this workflow execution"
    )
    definition: WorkflowDefinition = Field(
        ...,
        description="The workflow definition being executed"
    )
    status: WorkflowStatus = Field(
        ...,
        description="Current status of the workflow"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when workflow was created"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when workflow execution started"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when workflow execution completed"
    )
    steps: list[WorkflowStep] = Field(
        ...,
        description="Current state of all workflow steps"
    )
    events: list[WorkflowEvent] = Field(
        default_factory=list,
        description="Audit trail of all workflow events"
    )
    artifacts: dict[str, ArtifactReference] = Field(
        default_factory=dict,
        description="References to artifacts produced by this workflow"
    )
    checksums: dict[str, str] = Field(
        default_factory=dict,
        description="Checksums for workflow inputs and outputs"
    )
    total_cost: float = Field(
        default=0.0,
        ge=0,
        description="Total cost of workflow execution"
    )
    created_by: str = Field(
        ...,
        min_length=1,
        description="User or system that created this workflow"
    )


# --- Legacy dataclass models for backward compatibility ---
from dataclasses import dataclass, field, asdict
import json as _json


@dataclass
class ValidationError:
    """Validation error for dataset validation."""
    error_type: str
    message: str
    line_number: Optional[int] = None
    example_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ValidationError':
        return cls(**data)

    def to_json(self) -> str:
        return _json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'ValidationError':
        return cls.from_dict(_json.loads(json_str))


@dataclass
class DataValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    total_examples: int
    valid_examples: int
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    statistics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'is_valid': self.is_valid,
            'total_examples': self.total_examples,
            'valid_examples': self.valid_examples,
            'errors': [e.to_dict() if hasattr(e, 'to_dict') else e for e in self.errors],
            'warnings': self.warnings,
            'statistics': self.statistics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DataValidationResult':
        errors = [
            ValidationError.from_dict(e) if isinstance(e, dict) else e
            for e in data.get('errors', [])
        ]
        return cls(
            is_valid=data['is_valid'],
            total_examples=data['total_examples'],
            valid_examples=data['valid_examples'],
            errors=errors,
            warnings=data.get('warnings', []),
            statistics=data.get('statistics', {}),
        )

    def to_json(self) -> str:
        return _json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'DataValidationResult':
        return cls.from_dict(_json.loads(json_str))


@dataclass
class FineTuningJob:
    """Fine-tuning job record."""
    job_id: str
    job_name: str
    base_model_id: str
    training_data_s3_uri: str
    status: str
    hyperparameters: dict = field(default_factory=dict)
    training_metrics: dict = field(default_factory=dict)
    finetuned_model_id: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'job_id': self.job_id,
            'job_name': self.job_name,
            'base_model_id': self.base_model_id,
            'training_data_s3_uri': self.training_data_s3_uri,
            'status': self.status,
            'hyperparameters': self.hyperparameters,
            'training_metrics': self.training_metrics,
            'finetuned_model_id': self.finetuned_model_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FineTuningJob':
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        completed_at = data.get('completed_at')
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)
        return cls(
            job_id=data['job_id'],
            job_name=data['job_name'],
            base_model_id=data['base_model_id'],
            training_data_s3_uri=data['training_data_s3_uri'],
            status=data['status'],
            hyperparameters=data.get('hyperparameters', {}),
            training_metrics=data.get('training_metrics', {}),
            finetuned_model_id=data.get('finetuned_model_id'),
            created_at=created_at,
            completed_at=completed_at,
            error_message=data.get('error_message'),
        )

    def to_json(self) -> str:
        return _json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'FineTuningJob':
        return cls.from_dict(_json.loads(json_str))
