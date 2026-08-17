"""
Training audit logger for fine-tuning operations.

Records all fine-tuning operations with timestamps, parameters,
and outcomes to maintain a complete audit trail.

Requirements: 4.10
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class AuditEventType(str, Enum):
    """Types of auditable fine-tuning events."""

    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_STOPPED = "job_stopped"
    JOB_RESUMED = "job_resumed"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"
    COST_ESTIMATED = "cost_estimated"
    METRICS_RECORDED = "metrics_recorded"
    EARLY_STOPPING_TRIGGERED = "early_stopping_triggered"
    MODEL_REGISTERED = "model_registered"
    HYPERPARAMETER_TUNING_STARTED = "hyperparameter_tuning_started"
    HYPERPARAMETER_TUNING_COMPLETED = "hyperparameter_tuning_completed"


@dataclass
class AuditEntry:
    """A single audit log entry.

    Attributes:
        event_type: The type of event being recorded.
        timestamp: When the event occurred.
        job_id: The fine-tuning job ID (if applicable).
        parameters: Parameters associated with the event.
        outcome: The outcome or result of the operation.
        error: Error details if the operation failed.
        user_id: The user who initiated the operation.
    """

    event_type: AuditEventType
    timestamp: datetime
    job_id: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    outcome: str = ""
    error: str = ""
    user_id: str = ""


class TrainingAuditLogger:
    """Audit logger for fine-tuning operations.

    Maintains an in-memory log of all fine-tuning operations with
    timestamps, parameters, and outcomes. Supports querying by
    job ID and event type.

    Usage::

        logger = TrainingAuditLogger()
        logger.log_job_created("job-123", {"model": "claude-v2"})
        logger.log_job_completed("job-123", {"model_arn": "arn:..."})
        entries = logger.get_entries(job_id="job-123")
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    @property
    def entries(self) -> list[AuditEntry]:
        """Return a copy of all audit entries."""
        return list(self._entries)

    def log(
        self,
        event_type: AuditEventType,
        job_id: str = "",
        parameters: dict[str, Any] | None = None,
        outcome: str = "",
        error: str = "",
        user_id: str = "",
    ) -> AuditEntry:
        """Record an audit event.

        Args:
            event_type: The type of event.
            job_id: The fine-tuning job ID.
            parameters: Parameters associated with the event.
            outcome: The outcome description.
            error: Error details if applicable.
            user_id: The user who initiated the operation.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            job_id=job_id,
            parameters=parameters or {},
            outcome=outcome,
            error=error,
            user_id=user_id,
        )
        self._entries.append(entry)
        return entry

    def log_job_created(
        self,
        job_id: str,
        parameters: dict[str, Any] | None = None,
        user_id: str = "",
    ) -> AuditEntry:
        """Log a job creation event."""
        return self.log(
            AuditEventType.JOB_CREATED,
            job_id=job_id,
            parameters=parameters or {},
            outcome="Job created successfully",
            user_id=user_id,
        )

    def log_job_started(self, job_id: str) -> AuditEntry:
        """Log a job start event."""
        return self.log(
            AuditEventType.JOB_STARTED,
            job_id=job_id,
            outcome="Training started",
        )

    def log_job_completed(
        self,
        job_id: str,
        parameters: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a job completion event."""
        return self.log(
            AuditEventType.JOB_COMPLETED,
            job_id=job_id,
            parameters=parameters or {},
            outcome="Training completed successfully",
        )

    def log_job_failed(
        self,
        job_id: str,
        error: str,
        parameters: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a job failure event."""
        return self.log(
            AuditEventType.JOB_FAILED,
            job_id=job_id,
            parameters=parameters or {},
            outcome="Training failed",
            error=error,
        )

    def log_job_stopped(self, job_id: str, reason: str = "") -> AuditEntry:
        """Log a job stop event."""
        return self.log(
            AuditEventType.JOB_STOPPED,
            job_id=job_id,
            outcome=f"Training stopped: {reason}" if reason else "Training stopped",
        )

    def log_validation(
        self,
        job_id: str,
        success: bool,
        parameters: dict[str, Any] | None = None,
        error: str = "",
    ) -> AuditEntry:
        """Log a validation event."""
        if success:
            return self.log(
                AuditEventType.VALIDATION_COMPLETED,
                job_id=job_id,
                parameters=parameters or {},
                outcome="Validation passed",
            )
        return self.log(
            AuditEventType.VALIDATION_FAILED,
            job_id=job_id,
            parameters=parameters or {},
            outcome="Validation failed",
            error=error,
        )

    def log_metrics_recorded(
        self,
        job_id: str,
        parameters: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a metrics recording event."""
        return self.log(
            AuditEventType.METRICS_RECORDED,
            job_id=job_id,
            parameters=parameters or {},
            outcome="Training metrics recorded",
        )

    def log_early_stopping(
        self,
        job_id: str,
        reason: str,
    ) -> AuditEntry:
        """Log an early stopping event."""
        return self.log(
            AuditEventType.EARLY_STOPPING_TRIGGERED,
            job_id=job_id,
            outcome=reason,
        )

    def log_model_registered(
        self,
        job_id: str,
        parameters: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a model registration event."""
        return self.log(
            AuditEventType.MODEL_REGISTERED,
            job_id=job_id,
            parameters=parameters or {},
            outcome="Fine-tuned model registered in Model Registry",
        )

    def log_cost_estimated(
        self,
        job_id: str,
        parameters: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a cost estimation event."""
        return self.log(
            AuditEventType.COST_ESTIMATED,
            job_id=job_id,
            parameters=parameters or {},
            outcome="Cost estimate calculated",
        )

    def get_entries(
        self,
        job_id: str | None = None,
        event_type: AuditEventType | None = None,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filtering.

        Args:
            job_id: Filter by job ID.
            event_type: Filter by event type.

        Returns:
            List of matching audit entries.
        """
        results = self._entries
        if job_id is not None:
            results = [e for e in results if e.job_id == job_id]
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        return results

    def get_job_timeline(self, job_id: str) -> list[AuditEntry]:
        """Get all events for a job in chronological order.

        Args:
            job_id: The job ID to get the timeline for.

        Returns:
            List of audit entries sorted by timestamp.
        """
        entries = self.get_entries(job_id=job_id)
        return sorted(entries, key=lambda e: e.timestamp)

    def clear(self) -> None:
        """Clear all audit entries."""
        self._entries = []
