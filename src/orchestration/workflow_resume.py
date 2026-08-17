"""
Workflow resume support for restarting from last successful step.

Persists step completion status and allows restart from the last
successful step after failures or disconnection.

Requirements: 8.15, 8.20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.data_models.workflow import StepStatus


@dataclass
class StepCheckpoint:
    """Checkpoint for a single step's completion status.

    Attributes:
        step_id: The step identifier.
        status: The step's completion status.
        output: Output data if completed.
        error: Error message if failed.
        timestamp: When the checkpoint was recorded.
    """

    step_id: str
    status: StepStatus
    output: Optional[dict] = None
    error: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class ResumePoint:
    """Information about where to resume a workflow.

    Attributes:
        workflow_id: The workflow identifier.
        last_completed_step_id: ID of the last successfully completed step.
        next_step_ids: IDs of steps to execute next.
        completed_step_ids: All completed step IDs.
        failed_step_ids: All failed step IDs.
    """

    workflow_id: str
    last_completed_step_id: Optional[str] = None
    next_step_ids: list[str] = field(default_factory=list)
    completed_step_ids: list[str] = field(default_factory=list)
    failed_step_ids: list[str] = field(default_factory=list)


class WorkflowResumeManager:
    """Manages workflow checkpoints for resume capability.

    Persists step completion status in-memory (simulating DynamoDB)
    and determines the resume point after failures.

    Usage::

        mgr = WorkflowResumeManager()
        mgr.save_checkpoint("wf-1", StepCheckpoint("s1", StepStatus.COMPLETED))
        resume = mgr.get_resume_point("wf-1", all_step_ids=["s1", "s2", "s3"])
    """

    def __init__(self) -> None:
        # workflow_id -> {step_id -> StepCheckpoint}
        self._checkpoints: dict[str, dict[str, StepCheckpoint]] = {}

    def save_checkpoint(
        self, workflow_id: str, checkpoint: StepCheckpoint
    ) -> None:
        """Save a step checkpoint.

        Args:
            workflow_id: The workflow identifier.
            checkpoint: The step checkpoint to save.
        """
        if workflow_id not in self._checkpoints:
            self._checkpoints[workflow_id] = {}
        self._checkpoints[workflow_id][checkpoint.step_id] = checkpoint

    def get_checkpoint(
        self, workflow_id: str, step_id: str
    ) -> Optional[StepCheckpoint]:
        """Get a step checkpoint.

        Args:
            workflow_id: The workflow identifier.
            step_id: The step identifier.

        Returns:
            The StepCheckpoint or None if not found.
        """
        wf_checkpoints = self._checkpoints.get(workflow_id, {})
        return wf_checkpoints.get(step_id)

    def get_all_checkpoints(
        self, workflow_id: str
    ) -> list[StepCheckpoint]:
        """Get all checkpoints for a workflow.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            List of all StepCheckpoints for the workflow.
        """
        return list(self._checkpoints.get(workflow_id, {}).values())

    def get_resume_point(
        self,
        workflow_id: str,
        all_step_ids: list[str],
        step_dependencies: Optional[dict[str, list[str]]] = None,
    ) -> ResumePoint:
        """Determine the resume point for a workflow.

        Analyzes checkpoints to find the last completed step and
        determine which steps need to be executed next.

        Args:
            workflow_id: The workflow identifier.
            all_step_ids: All step IDs in the workflow (in order).
            step_dependencies: Optional map of step_id -> dependency IDs.

        Returns:
            ResumePoint with resume information.
        """
        wf_checkpoints = self._checkpoints.get(workflow_id, {})

        completed = []
        failed = []
        last_completed = None

        for step_id in all_step_ids:
            cp = wf_checkpoints.get(step_id)
            if cp is not None:
                if cp.status == StepStatus.COMPLETED:
                    completed.append(step_id)
                    last_completed = step_id
                elif cp.status == StepStatus.FAILED:
                    failed.append(step_id)

        # Determine next steps: steps not completed and whose
        # dependencies are all completed
        deps = step_dependencies or {}
        completed_set = set(completed)
        next_steps = []
        for step_id in all_step_ids:
            if step_id in completed_set:
                continue
            step_deps = deps.get(step_id, [])
            if all(d in completed_set for d in step_deps):
                next_steps.append(step_id)

        return ResumePoint(
            workflow_id=workflow_id,
            last_completed_step_id=last_completed,
            next_step_ids=next_steps,
            completed_step_ids=completed,
            failed_step_ids=failed,
        )

    def clear_checkpoints(self, workflow_id: str) -> None:
        """Clear all checkpoints for a workflow.

        Args:
            workflow_id: The workflow identifier.
        """
        self._checkpoints.pop(workflow_id, None)

    def is_step_completed(
        self, workflow_id: str, step_id: str
    ) -> bool:
        """Check if a step has been completed.

        Args:
            workflow_id: The workflow identifier.
            step_id: The step identifier.

        Returns:
            True if the step has a COMPLETED checkpoint.
        """
        cp = self.get_checkpoint(workflow_id, step_id)
        return cp is not None and cp.status == StepStatus.COMPLETED
