"""
Workflow history query for retrieving past workflow runs.

Supports filtering by status, user, and date range. Returns workflow
runs with status, duration, and results.

Requirements: 8.8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.data_models.workflow import WorkflowManifest, WorkflowStatus


@dataclass
class WorkflowHistoryFilter:
    """Filter criteria for querying workflow history.

    Attributes:
        status: Filter by workflow status.
        created_by: Filter by the user who created the workflow.
        date_from: Filter workflows created on or after this date.
        date_to: Filter workflows created on or before this date.
    """

    status: Optional[WorkflowStatus] = None
    created_by: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


@dataclass
class WorkflowRunSummary:
    """Summary of a single workflow run.

    Attributes:
        workflow_id: Unique workflow identifier.
        name: Workflow name.
        status: Current workflow status.
        created_by: User who created the workflow.
        created_at: When the workflow was created.
        started_at: When execution started.
        completed_at: When execution completed.
        duration_seconds: Total execution duration.
        total_cost: Total cost of the workflow.
        step_count: Number of steps in the workflow.
        completed_steps: Number of completed steps.
        failed_steps: Number of failed steps.
    """

    workflow_id: str
    name: str
    status: WorkflowStatus
    created_by: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    total_cost: float = 0.0
    step_count: int = 0
    completed_steps: int = 0
    failed_steps: int = 0


def _matches_filter(
    manifest: WorkflowManifest,
    history_filter: WorkflowHistoryFilter,
) -> bool:
    """Check if a manifest matches the given filter criteria.

    Args:
        manifest: The workflow manifest to check.
        history_filter: The filter criteria.

    Returns:
        True if the manifest matches all filter criteria.
    """
    if history_filter.status is not None:
        if manifest.status != history_filter.status:
            return False

    if history_filter.created_by is not None:
        if manifest.created_by != history_filter.created_by:
            return False

    if history_filter.date_from is not None:
        created = manifest.created_at
        date_from = history_filter.date_from
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if date_from.tzinfo is None:
            date_from = date_from.replace(tzinfo=timezone.utc)
        if created < date_from:
            return False

    if history_filter.date_to is not None:
        created = manifest.created_at
        date_to = history_filter.date_to
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if date_to.tzinfo is None:
            date_to = date_to.replace(tzinfo=timezone.utc)
        if created > date_to:
            return False

    return True


def _summarize_manifest(manifest: WorkflowManifest) -> WorkflowRunSummary:
    """Create a summary from a workflow manifest.

    Args:
        manifest: The workflow manifest.

    Returns:
        A WorkflowRunSummary with key metrics.
    """
    duration = None
    if manifest.started_at and manifest.completed_at:
        delta = manifest.completed_at - manifest.started_at
        duration = delta.total_seconds()

    completed = sum(
        1 for s in manifest.steps if s.status.value == "completed"
    )
    failed = sum(
        1 for s in manifest.steps if s.status.value == "failed"
    )

    return WorkflowRunSummary(
        workflow_id=manifest.workflow_id,
        name=manifest.definition.name,
        status=manifest.status,
        created_by=manifest.created_by,
        created_at=manifest.created_at,
        started_at=manifest.started_at,
        completed_at=manifest.completed_at,
        duration_seconds=duration,
        total_cost=manifest.total_cost,
        step_count=len(manifest.steps),
        completed_steps=completed,
        failed_steps=failed,
    )


class WorkflowHistoryQuery:
    """Queries workflow history from an in-memory store.

    Supports filtering by status, user, and date range.
    Returns workflow run summaries with status, duration, and results.

    Usage::

        query = WorkflowHistoryQuery()
        query.add_manifest(manifest)
        results = query.query(WorkflowHistoryFilter(status=WorkflowStatus.COMPLETED))
    """

    def __init__(self) -> None:
        self._manifests: dict[str, WorkflowManifest] = {}

    def add_manifest(self, manifest: WorkflowManifest) -> None:
        """Add or update a workflow manifest in the store.

        Args:
            manifest: The workflow manifest to store.
        """
        self._manifests[manifest.workflow_id] = manifest

    def get_manifest(
        self, workflow_id: str
    ) -> Optional[WorkflowManifest]:
        """Get a manifest by workflow ID.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            The manifest or None if not found.
        """
        return self._manifests.get(workflow_id)

    def query(
        self,
        history_filter: Optional[WorkflowHistoryFilter] = None,
        limit: int = 100,
    ) -> list[WorkflowRunSummary]:
        """Query workflow history with optional filtering.

        Args:
            history_filter: Optional filter criteria.
            limit: Maximum number of results to return.

        Returns:
            List of WorkflowRunSummary sorted by created_at descending.
        """
        filt = history_filter or WorkflowHistoryFilter()
        matching = [
            m for m in self._manifests.values()
            if _matches_filter(m, filt)
        ]
        matching.sort(key=lambda m: m.created_at, reverse=True)
        summaries = [_summarize_manifest(m) for m in matching[:limit]]
        return summaries

    def count(
        self,
        history_filter: Optional[WorkflowHistoryFilter] = None,
    ) -> int:
        """Count workflows matching the filter.

        Args:
            history_filter: Optional filter criteria.

        Returns:
            Number of matching workflows.
        """
        filt = history_filter or WorkflowHistoryFilter()
        return sum(
            1 for m in self._manifests.values()
            if _matches_filter(m, filt)
        )
