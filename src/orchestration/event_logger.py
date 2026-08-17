"""
Workflow event logger for step execution tracking.

Logs step execution with timestamps, inputs, outputs, and duration.
Supports event types: started, completed, failed, retrying.

Requirements: 8.3
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class WorkflowEventType(str, Enum):
    """Types of workflow events."""

    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_RETRYING = "step_retrying"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"


@dataclass
class EventRecord:
    """A single workflow event record.

    Attributes:
        event_id: Unique event identifier.
        event_type: The type of event.
        workflow_id: The workflow this event belongs to.
        step_id: The step this event relates to (if applicable).
        timestamp: When the event occurred.
        details: Additional event details.
    """

    event_id: str
    event_type: WorkflowEventType
    workflow_id: str
    step_id: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    details: dict[str, Any] = field(default_factory=dict)


class WorkflowEventLogger:
    """Logger for workflow execution events.

    Records all workflow and step events with timestamps and details.
    Supports querying by workflow ID, step ID, and event type.

    Usage::

        logger = WorkflowEventLogger()
        logger.log_step_started("wf-1", "step-1", inputs={"data": "..."})
        logger.log_step_completed("wf-1", "step-1", outputs={"result": "..."}, duration=1.5)
    """

    def __init__(self) -> None:
        self._events: list[EventRecord] = []

    @property
    def events(self) -> list[EventRecord]:
        """Return a copy of all events."""
        return list(self._events)

    def _log(
        self,
        event_type: WorkflowEventType,
        workflow_id: str,
        step_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> EventRecord:
        event = EventRecord(
            event_id=uuid.uuid4().hex[:12],
            event_type=event_type,
            workflow_id=workflow_id,
            step_id=step_id,
            details=details or {},
        )
        self._events.append(event)
        return event

    def log_workflow_started(
        self, workflow_id: str, details: dict[str, Any] | None = None
    ) -> EventRecord:
        """Log a workflow start event."""
        return self._log(
            WorkflowEventType.WORKFLOW_STARTED, workflow_id, details=details
        )

    def log_workflow_completed(
        self, workflow_id: str, details: dict[str, Any] | None = None
    ) -> EventRecord:
        """Log a workflow completion event."""
        return self._log(
            WorkflowEventType.WORKFLOW_COMPLETED, workflow_id, details=details
        )

    def log_workflow_failed(
        self, workflow_id: str, error: str = "", details: dict[str, Any] | None = None
    ) -> EventRecord:
        """Log a workflow failure event."""
        d = dict(details or {})
        if error:
            d["error"] = error
        return self._log(
            WorkflowEventType.WORKFLOW_FAILED, workflow_id, details=d
        )

    def log_step_started(
        self,
        workflow_id: str,
        step_id: str,
        inputs: dict[str, Any] | None = None,
    ) -> EventRecord:
        """Log a step start event with inputs."""
        return self._log(
            WorkflowEventType.STEP_STARTED,
            workflow_id,
            step_id=step_id,
            details={"inputs": inputs or {}},
        )

    def log_step_completed(
        self,
        workflow_id: str,
        step_id: str,
        outputs: dict[str, Any] | None = None,
        duration_seconds: float = 0.0,
    ) -> EventRecord:
        """Log a step completion event with outputs and duration."""
        return self._log(
            WorkflowEventType.STEP_COMPLETED,
            workflow_id,
            step_id=step_id,
            details={
                "outputs": outputs or {},
                "duration_seconds": duration_seconds,
            },
        )

    def log_step_failed(
        self,
        workflow_id: str,
        step_id: str,
        error: str = "",
        duration_seconds: float = 0.0,
    ) -> EventRecord:
        """Log a step failure event."""
        return self._log(
            WorkflowEventType.STEP_FAILED,
            workflow_id,
            step_id=step_id,
            details={"error": error, "duration_seconds": duration_seconds},
        )

    def log_step_retrying(
        self,
        workflow_id: str,
        step_id: str,
        attempt: int = 0,
        error: str = "",
    ) -> EventRecord:
        """Log a step retry event."""
        return self._log(
            WorkflowEventType.STEP_RETRYING,
            workflow_id,
            step_id=step_id,
            details={"attempt": attempt, "error": error},
        )

    def get_events(
        self,
        workflow_id: str | None = None,
        step_id: str | None = None,
        event_type: WorkflowEventType | None = None,
    ) -> list[EventRecord]:
        """Query events with optional filtering.

        Args:
            workflow_id: Filter by workflow ID.
            step_id: Filter by step ID.
            event_type: Filter by event type.

        Returns:
            List of matching event records.
        """
        results = self._events
        if workflow_id is not None:
            results = [e for e in results if e.workflow_id == workflow_id]
        if step_id is not None:
            results = [e for e in results if e.step_id == step_id]
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        return results

    def get_workflow_timeline(self, workflow_id: str) -> list[EventRecord]:
        """Get all events for a workflow in chronological order."""
        events = self.get_events(workflow_id=workflow_id)
        return sorted(events, key=lambda e: e.timestamp)

    def clear(self) -> None:
        """Clear all events."""
        self._events = []
