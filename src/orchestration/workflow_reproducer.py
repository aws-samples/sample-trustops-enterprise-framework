"""
Workflow reproducer for re-executing workflows with identical config.

Loads a stored manifest and creates a new workflow with the same steps,
parameters, and data references for reproducibility.

Requirements: 8.9
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from src.data_models.workflow import (
    StepStatus,
    WorkflowDefinition,
    WorkflowManifest,
    WorkflowStatus,
    WorkflowStep,
)
from src.orchestration.checksum_calculator import calculate_checksum
from src.orchestration.workflow_id import generate_workflow_id

if TYPE_CHECKING:
    from src.orchestration.workflow_manager import WorkflowManager


@dataclass
class ReproductionResult:
    """Result of reproducing a workflow.

    Attributes:
        success: Whether reproduction was successful.
        new_workflow_id: The ID of the newly created workflow.
        original_workflow_id: The ID of the original workflow.
        manifest: The new workflow manifest.
        error: Error message if reproduction failed.
        audit_record: Optional audit trail record linking original
            to new workflow.
    """

    success: bool
    new_workflow_id: str = ""
    original_workflow_id: str = ""
    manifest: Optional[WorkflowManifest] = None
    error: str = ""
    audit_record: Optional[dict] = None


def _reset_steps(steps: list[WorkflowStep]) -> list[WorkflowStep]:
    """Reset all steps to pending status for re-execution.

    Args:
        steps: The original steps to reset.

    Returns:
        New list of steps with status reset to PENDING.
    """
    reset = []
    for step in steps:
        new_step = WorkflowStep(
            step_id=step.step_id,
            step_type=step.step_type,
            name=step.name,
            config=deepcopy(step.config),
            status=StepStatus.PENDING,
            depends_on=list(step.depends_on),
            retry_count=0,
        )
        reset.append(new_step)
    return reset


class WorkflowReproducer:
    """Reproduces workflows with identical configuration.

    Loads a stored manifest and creates a new workflow with the same
    steps, parameters, and data references.

    When a WorkflowManager is provided, manifests are persisted to
    DynamoDB for durable audit trails. Without one, falls back to
    in-memory storage (useful for testing and offline use).

    Usage::

        reproducer = WorkflowReproducer()
        reproducer.store_manifest(original_manifest)
        result = reproducer.reproduce("wf-original-id", created_by="user")
    """

    def __init__(
        self,
        workflow_manager: Optional[WorkflowManager] = None,
    ) -> None:
        self._workflow_manager = workflow_manager
        self._manifests: dict[str, WorkflowManifest] = {}

    def store_manifest(
        self,
        manifest: WorkflowManifest,
        dataset_checksum: str = "",
    ) -> None:
        """Store a manifest for later reproduction.

        Args:
            manifest: The workflow manifest to store.
            dataset_checksum: SHA-256 checksum of the dataset for
                integrity verification on reproduction.
        """
        if self._workflow_manager:
            config = {
                "definition": manifest.definition.__dict__
                if hasattr(manifest.definition, "__dict__")
                else {},
                "dataset_checksum": dataset_checksum,
            }
            self._workflow_manager.create_workflow(
                workflow_type=manifest.definition.name
                if hasattr(manifest.definition, "name")
                else "reproduction",
                configuration=config,
                created_by=manifest.created_by,
            )
        self._manifests[manifest.workflow_id] = manifest
        if dataset_checksum:
            if not hasattr(self, "_checksums"):
                self._checksums: dict[str, str] = {}
            self._checksums[manifest.workflow_id] = dataset_checksum

    def get_manifest(
        self, workflow_id: str
    ) -> Optional[WorkflowManifest]:
        """Get a stored manifest by workflow ID.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            The manifest or None if not found.
        """
        manifest = self._manifests.get(workflow_id)
        if manifest is None and self._workflow_manager:
            wf = self._workflow_manager.get_workflow(workflow_id)
            if wf is not None:
                return wf  # type: ignore[return-value]
        return manifest

    def reproduce(
        self,
        workflow_id: str,
        created_by: str = "system",
        dataset_content: Optional[bytes] = None,
    ) -> ReproductionResult:
        """Reproduce a workflow with identical configuration.

        Creates a new workflow with the same steps, parameters, and
        data references as the original. The new workflow gets a fresh
        ID and all steps are reset to PENDING.

        If dataset_content is provided, verifies the SHA-256 checksum
        matches the stored checksum before proceeding.

        Args:
            workflow_id: The original workflow ID to reproduce.
            created_by: The user creating the reproduction.
            dataset_content: Optional raw dataset bytes for checksum
                verification.

        Returns:
            ReproductionResult with the new manifest.
        """
        original = self._manifests.get(workflow_id)
        if original is None:
            return ReproductionResult(
                success=False,
                original_workflow_id=workflow_id,
                error=f"Workflow not found: {workflow_id}",
            )

        # Checksum verification if dataset_content provided
        if dataset_content is not None:
            checksums = getattr(self, "_checksums", {})
            stored_checksum = checksums.get(workflow_id, "")
            if stored_checksum:
                current_checksum = calculate_checksum(dataset_content)
                if current_checksum != stored_checksum:
                    return ReproductionResult(
                        success=False,
                        original_workflow_id=workflow_id,
                        error=(
                            f"Dataset integrity mismatch: "
                            f"expected {stored_checksum} "
                            f"got {current_checksum}"
                        ),
                    )

        new_id = generate_workflow_id()
        reset_steps = _reset_steps(original.definition.steps)

        new_definition = WorkflowDefinition(
            name=original.definition.name,
            description=original.definition.description,
            steps=reset_steps,
            requires_approval=original.definition.requires_approval,
            max_retries=original.definition.max_retries,
            retry_delay_seconds=original.definition.retry_delay_seconds,
            timeout_seconds=original.definition.timeout_seconds,
        )

        now = datetime.now(timezone.utc)
        new_manifest = WorkflowManifest(
            workflow_id=new_id,
            definition=new_definition,
            status=WorkflowStatus.PENDING,
            created_at=now,
            steps=reset_steps,
            created_by=created_by,
        )

        self._manifests[new_id] = new_manifest

        # Store audit trail record
        audit_record = {
            "original_workflow_id": workflow_id,
            "new_workflow_id": new_id,
            "reproduced_at": now.isoformat(),
            "reproduced_by": created_by,
        }

        if self._workflow_manager:
            self._workflow_manager.create_workflow(
                workflow_type="reproduction",
                configuration={
                    "original_workflow_id": workflow_id,
                    "new_workflow_id": new_id,
                },
                created_by=created_by,
            )

        return ReproductionResult(
            success=True,
            new_workflow_id=new_id,
            original_workflow_id=workflow_id,
            manifest=new_manifest,
            audit_record=audit_record,
        )
