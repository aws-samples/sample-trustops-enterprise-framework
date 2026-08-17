"""
Workflow ID generator for unique workflow identification.

Generates UUID-based workflow identifiers and logs workflow start events.

Requirements: 8.2, 8.5
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class WorkflowStartEvent:
    """Event recorded when a workflow starts.

    Attributes:
        workflow_id: The unique workflow identifier.
        timestamp: When the workflow was started.
        created_by: The user or system that created the workflow.
        metadata: Additional metadata about the workflow start.
    """

    workflow_id: str
    timestamp: datetime
    created_by: str = ""
    metadata: dict = field(default_factory=dict)


def generate_workflow_id(prefix: str = "wf") -> str:
    """Generate a unique workflow identifier.

    Creates a UUID4-based identifier with an optional prefix
    for easy identification of workflow resources.

    Args:
        prefix: Prefix for the workflow ID. Defaults to "wf".

    Returns:
        A unique workflow identifier string.
    """
    unique_id = uuid.uuid4().hex[:12]
    return f"{prefix}-{unique_id}"


def create_workflow_start_event(
    workflow_id: str,
    created_by: str = "",
    metadata: dict | None = None,
) -> WorkflowStartEvent:
    """Create and return a workflow start event.

    Args:
        workflow_id: The workflow identifier.
        created_by: The user or system that created the workflow.
        metadata: Additional metadata about the workflow.

    Returns:
        A WorkflowStartEvent with the current timestamp.
    """
    return WorkflowStartEvent(
        workflow_id=workflow_id,
        timestamp=datetime.now(timezone.utc),
        created_by=created_by,
        metadata=metadata or {},
    )
