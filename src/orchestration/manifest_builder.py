"""
Workflow manifest builder for complete workflow documentation.

Builds a complete manifest with configurations, data references, results,
and all artifact references with S3 URIs.

Requirements: 8.7
"""

from datetime import datetime, timezone
from typing import Any, Optional

from src.data_models.workflow import (
    ArtifactReference,
    WorkflowDefinition,
    WorkflowEvent,
    WorkflowManifest,
    WorkflowStatus,
    WorkflowStep,
)


class ManifestBuilder:
    """Builder for constructing WorkflowManifest objects.

    Provides a fluent interface for incrementally building a complete
    workflow manifest with all configurations, events, artifacts,
    and checksums.

    Usage::

        builder = ManifestBuilder("wf-123", definition, "user-1")
        builder.set_status(WorkflowStatus.RUNNING)
        builder.add_event(event)
        builder.add_artifact("dataset", ref)
        manifest = builder.build()
    """

    def __init__(
        self,
        workflow_id: str,
        definition: WorkflowDefinition,
        created_by: str,
    ) -> None:
        self._workflow_id = workflow_id
        self._definition = definition
        self._created_by = created_by
        self._status = WorkflowStatus.PENDING
        self._created_at = datetime.now(timezone.utc)
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
        self._steps: list[WorkflowStep] = list(definition.steps)
        self._events: list[WorkflowEvent] = []
        self._artifacts: dict[str, ArtifactReference] = {}
        self._checksums: dict[str, str] = {}
        self._total_cost: float = 0.0

    def set_status(self, status: WorkflowStatus) -> "ManifestBuilder":
        """Set the workflow status."""
        self._status = status
        if status == WorkflowStatus.RUNNING and self._started_at is None:
            self._started_at = datetime.now(timezone.utc)
        if status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        ):
            self._completed_at = datetime.now(timezone.utc)
        return self

    def set_started_at(self, started_at: datetime) -> "ManifestBuilder":
        """Set the workflow start timestamp."""
        self._started_at = started_at
        return self

    def set_completed_at(self, completed_at: datetime) -> "ManifestBuilder":
        """Set the workflow completion timestamp."""
        self._completed_at = completed_at
        return self

    def update_step(self, step: WorkflowStep) -> "ManifestBuilder":
        """Update a step in the manifest.

        Replaces the step with matching step_id.

        Args:
            step: The updated step.

        Returns:
            Self for chaining.
        """
        self._steps = [
            step if s.step_id == step.step_id else s for s in self._steps
        ]
        return self

    def add_event(self, event: WorkflowEvent) -> "ManifestBuilder":
        """Add a workflow event to the manifest."""
        self._events.append(event)
        return self

    def add_artifact(
        self, name: str, artifact: ArtifactReference
    ) -> "ManifestBuilder":
        """Add an artifact reference to the manifest.

        Args:
            name: The artifact name/key.
            artifact: The artifact reference with S3 URI and checksum.

        Returns:
            Self for chaining.
        """
        self._artifacts[name] = artifact
        return self

    def add_checksum(self, key: str, checksum: str) -> "ManifestBuilder":
        """Add a checksum for a data artifact.

        Args:
            key: The artifact key.
            checksum: The SHA-256 checksum.

        Returns:
            Self for chaining.
        """
        self._checksums[key] = checksum
        return self

    def set_total_cost(self, cost: float) -> "ManifestBuilder":
        """Set the total workflow cost."""
        self._total_cost = cost
        return self

    def build(self) -> WorkflowManifest:
        """Build and return the complete WorkflowManifest.

        Returns:
            A fully populated WorkflowManifest.
        """
        return WorkflowManifest(
            workflow_id=self._workflow_id,
            definition=self._definition,
            status=self._status,
            created_at=self._created_at,
            started_at=self._started_at,
            completed_at=self._completed_at,
            steps=list(self._steps),
            events=list(self._events),
            artifacts=dict(self._artifacts),
            checksums=dict(self._checksums),
            total_cost=self._total_cost,
            created_by=self._created_by,
        )
