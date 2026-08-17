"""
WorkflowOrchestrator integrating all orchestration components.

Provides a unified interface for creating, starting, pausing, resuming,
cancelling, and reproducing workflows with full audit trail.

Requirements: 8.1-8.15
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Optional

from src.data_models.workflow import (
    ArtifactReference,
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowEvent,
    WorkflowManifest,
    WorkflowStatus,
    WorkflowStep,
)
from src.orchestration.approval_gate import ApprovalGate, ApprovalStatus
from src.orchestration.checksum_calculator import calculate_json_checksum
from src.orchestration.event_logger import WorkflowEventLogger
from src.orchestration.failure_notification import (
    FailureNotifier,
    RecoveryOption,
)
from src.orchestration.manifest_builder import ManifestBuilder
from src.orchestration.partial_failure_handler import PartialFailureHandler
from src.orchestration.retry_handler import RetryConfig, execute_with_retry
from src.orchestration.step_executor import StepExecutor
from src.orchestration.step_runner import StepNode, run_sequential
from src.orchestration.workflow_estimator import WorkflowEstimate, WorkflowEstimator
from src.orchestration.workflow_history import (
    WorkflowHistoryFilter,
    WorkflowHistoryQuery,
    WorkflowRunSummary,
)
from src.orchestration.workflow_id import generate_workflow_id
from src.orchestration.workflow_isolation import WorkflowIsolation
from src.orchestration.workflow_reproducer import (
    ReproductionResult,
    WorkflowReproducer,
)
from src.orchestration.workflow_resume import (
    StepCheckpoint,
    WorkflowResumeManager,
)


class WorkflowOrchestrator:
    """Orchestrates multi-step evaluation and fine-tuning workflows.

    Integrates all orchestration components: step execution, retry,
    approval gates, event logging, manifest building, history,
    reproduction, estimation, isolation, partial failure handling,
    resume, and failure notification.

    Usage::

        orchestrator = WorkflowOrchestrator()
        orchestrator.register_step_handler("baseline_evaluation", eval_fn)
        manifest = orchestrator.create_workflow(definition, "user-1")
        manifest = orchestrator.start_workflow(manifest.workflow_id)
    """

    def __init__(
        self,
        step_executor: Optional[StepExecutor] = None,
        event_logger: Optional[WorkflowEventLogger] = None,
        approval_gate: Optional[ApprovalGate] = None,
        estimator: Optional[WorkflowEstimator] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        self._executor = step_executor or StepExecutor()
        self._event_logger = event_logger or WorkflowEventLogger()
        self._approval_gate = approval_gate or ApprovalGate()
        self._estimator = estimator or WorkflowEstimator()
        self._retry_config = retry_config or RetryConfig(
            max_retries=3, base_delay_seconds=1.0
        )

        self._history = WorkflowHistoryQuery()
        self._reproducer = WorkflowReproducer()
        self._isolation = WorkflowIsolation()
        self._resume_mgr = WorkflowResumeManager()
        self._notifier = FailureNotifier()

        # Active manifests keyed by workflow_id
        self._manifests: dict[str, WorkflowManifest] = {}

    # -- Registration --

    def register_step_handler(
        self, step_type: str, handler: Any
    ) -> None:
        """Register a handler for a step type.

        Args:
            step_type: The step type identifier.
            handler: Callable(step_id, config) -> dict.
        """
        self._executor.register_handler(step_type, handler)

    # -- Workflow lifecycle --

    def create_workflow(
        self,
        definition: WorkflowDefinition,
        created_by: str,
    ) -> WorkflowManifest:
        """Create a new workflow from a definition.

        Args:
            definition: The workflow definition.
            created_by: User creating the workflow.

        Returns:
            The created WorkflowManifest in PENDING status.
        """
        workflow_id = generate_workflow_id()
        now = datetime.now(timezone.utc)

        builder = ManifestBuilder(workflow_id, definition, created_by)
        builder.set_status(WorkflowStatus.PENDING)
        manifest = builder.build()

        self._manifests[workflow_id] = manifest
        self._history.add_manifest(manifest)
        self._reproducer.store_manifest(manifest)
        self._isolation.create_scope(workflow_id)

        self._event_logger.log_workflow_started(
            workflow_id,
            details={"created_by": created_by, "name": definition.name},
        )

        return manifest

    def start_workflow(self, workflow_id: str) -> WorkflowManifest:
        """Start executing a workflow.

        Runs steps sequentially in dependency order with retry
        and checkpoint support.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            Updated WorkflowManifest.

        Raises:
            KeyError: If workflow not found.
            ValueError: If workflow is not in a startable state.
        """
        manifest = self._get_manifest(workflow_id)
        if manifest.status not in (
            WorkflowStatus.PENDING,
            WorkflowStatus.PAUSED,
            WorkflowStatus.FAILED,
        ):
            raise ValueError(
                f"Cannot start workflow in status: {manifest.status.value}"
            )

        manifest = self._update_status(workflow_id, WorkflowStatus.RUNNING)

        # Build step nodes for execution
        step_map = {s.step_id: s for s in manifest.steps}
        all_step_ids = [s.step_id for s in manifest.steps]
        deps = {s.step_id: list(s.depends_on) for s in manifest.steps}

        # Get resume point to skip completed steps
        resume_point = self._resume_mgr.get_resume_point(
            workflow_id, all_step_ids, deps
        )

        for step_id in all_step_ids:
            if step_id in resume_point.completed_step_ids:
                continue

            step = step_map[step_id]

            # Check dependencies
            for dep_id in step.depends_on:
                dep_step = step_map.get(dep_id)
                if dep_step and dep_step.status != StepStatus.COMPLETED:
                    if not self._resume_mgr.is_step_completed(
                        workflow_id, dep_id
                    ):
                        # Dependency not met, fail
                        manifest = self._fail_step(
                            workflow_id, step,
                            f"Dependency {dep_id} not completed",
                        )
                        return self._update_status(
                            workflow_id, WorkflowStatus.FAILED
                        )

            # Handle approval gates
            if step.step_type == StepType.APPROVAL_GATE:
                req = self._approval_gate.request_approval(
                    workflow_id, step_id
                )
                if not self._approval_gate.is_approved(req.request_id):
                    manifest = self._update_status(
                        workflow_id, WorkflowStatus.PAUSED
                    )
                    return manifest

            # Execute step with retry
            manifest = self._execute_step_with_retry(
                workflow_id, step
            )

            # Check if step failed
            updated_step = self._find_step(manifest, step_id)
            if updated_step and updated_step.status == StepStatus.FAILED:
                self._notifier.notify_failure(
                    workflow_id,
                    step_id,
                    step.name,
                    updated_step.error or "Unknown error",
                    recovery_options=[
                        RecoveryOption.RETRY,
                        RecoveryOption.SKIP,
                        RecoveryOption.CANCEL,
                    ],
                )
                return self._update_status(
                    workflow_id, WorkflowStatus.FAILED
                )

        # All steps completed
        manifest = self._update_status(
            workflow_id, WorkflowStatus.COMPLETED
        )

        # Build final manifest with checksums
        config_checksum = calculate_json_checksum(
            manifest.definition.model_dump()
        )
        self._update_manifest_checksum(
            workflow_id, "workflow_config", config_checksum
        )

        return self._get_manifest(workflow_id)

    def pause_workflow(self, workflow_id: str) -> WorkflowManifest:
        """Pause a running workflow.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            Updated WorkflowManifest.
        """
        manifest = self._get_manifest(workflow_id)
        if manifest.status != WorkflowStatus.RUNNING:
            raise ValueError(
                f"Cannot pause workflow in status: {manifest.status.value}"
            )
        self._event_logger.log_workflow_started(
            workflow_id, details={"action": "paused"}
        )
        return self._update_status(workflow_id, WorkflowStatus.PAUSED)

    def resume_workflow(self, workflow_id: str) -> WorkflowManifest:
        """Resume a paused or failed workflow from last checkpoint.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            Updated WorkflowManifest after resuming execution.
        """
        manifest = self._get_manifest(workflow_id)
        if manifest.status not in (
            WorkflowStatus.PAUSED,
            WorkflowStatus.FAILED,
        ):
            raise ValueError(
                f"Cannot resume workflow in status: {manifest.status.value}"
            )
        return self.start_workflow(workflow_id)

    def cancel_workflow(self, workflow_id: str) -> WorkflowManifest:
        """Cancel a workflow.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            Updated WorkflowManifest.
        """
        manifest = self._get_manifest(workflow_id)
        if manifest.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.CANCELLED,
        ):
            raise ValueError(
                f"Cannot cancel workflow in status: {manifest.status.value}"
            )
        self._event_logger.log_workflow_failed(
            workflow_id, error="Cancelled by user"
        )
        return self._update_status(workflow_id, WorkflowStatus.CANCELLED)

    # -- Query methods --

    def get_status(self, workflow_id: str) -> WorkflowManifest:
        """Get current workflow status.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            The current WorkflowManifest.
        """
        return self._get_manifest(workflow_id)

    def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        created_by: Optional[str] = None,
        limit: int = 100,
    ) -> list[WorkflowRunSummary]:
        """List workflows with optional filtering.

        Args:
            status: Filter by status.
            created_by: Filter by creator.
            limit: Maximum results.

        Returns:
            List of WorkflowRunSummary.
        """
        filt = WorkflowHistoryFilter(
            status=status, created_by=created_by
        )
        return self._history.query(filt, limit=limit)

    def reproduce_workflow(
        self, workflow_id: str, created_by: str = "system"
    ) -> ReproductionResult:
        """Reproduce a workflow with identical configuration.

        Args:
            workflow_id: The original workflow ID.
            created_by: User creating the reproduction.

        Returns:
            ReproductionResult with the new manifest.
        """
        result = self._reproducer.reproduce(workflow_id, created_by)
        if result.success and result.manifest:
            self._manifests[result.new_workflow_id] = result.manifest
            self._history.add_manifest(result.manifest)
            self._isolation.create_scope(result.new_workflow_id)
        return result

    def estimate_workflow(
        self, definition: WorkflowDefinition
    ) -> WorkflowEstimate:
        """Estimate time and cost for a workflow.

        Args:
            definition: The workflow definition.

        Returns:
            WorkflowEstimate with cost and time breakdown.
        """
        return self._estimator.estimate(definition)

    def get_workflow_templates(self) -> list[WorkflowDefinition]:
        """Get pre-built workflow templates.

        Returns:
            List of available WorkflowDefinition templates.
        """
        from src.orchestration.templates import (
            get_all_templates,
        )
        return get_all_templates()

    # -- Internal helpers --

    def _get_manifest(self, workflow_id: str) -> WorkflowManifest:
        manifest = self._manifests.get(workflow_id)
        if manifest is None:
            raise KeyError(f"Workflow not found: {workflow_id}")
        return manifest

    def _update_status(
        self, workflow_id: str, status: WorkflowStatus
    ) -> WorkflowManifest:
        manifest = self._get_manifest(workflow_id)
        now = datetime.now(timezone.utc)

        # Create updated manifest
        updated = manifest.model_copy(update={"status": status})
        if status == WorkflowStatus.RUNNING and updated.started_at is None:
            updated = updated.model_copy(update={"started_at": now})
        if status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        ):
            updated = updated.model_copy(update={"completed_at": now})

        self._manifests[workflow_id] = updated
        self._history.add_manifest(updated)
        return updated

    def _execute_step_with_retry(
        self, workflow_id: str, step: WorkflowStep
    ) -> WorkflowManifest:
        """Execute a step with retry logic."""
        self._event_logger.log_step_started(
            workflow_id, step.step_id,
            inputs=step.config,
        )

        # Mark step as running
        self._update_step_status(
            workflow_id, step.step_id, StepStatus.RUNNING
        )

        def attempt():
            return self._executor.execute(
                step_id=step.step_id,
                step_type=step.step_type.value,
                config=step.config,
            )

        retry_result = execute_with_retry(
            attempt,
            config=self._retry_config,
            sleep_func=lambda _: None,  # No actual sleep in sync mode
        )

        if retry_result.success and retry_result.result:
            exec_result = retry_result.result
            if exec_result.success:
                self._event_logger.log_step_completed(
                    workflow_id, step.step_id,
                    outputs=exec_result.output,
                    duration_seconds=exec_result.duration_seconds,
                )
                self._update_step_status(
                    workflow_id, step.step_id, StepStatus.COMPLETED,
                    output=exec_result.output,
                    duration=exec_result.duration_seconds,
                )
                self._resume_mgr.save_checkpoint(
                    workflow_id,
                    StepCheckpoint(
                        step.step_id,
                        StepStatus.COMPLETED,
                        output=exec_result.output,
                    ),
                )
            else:
                return self._fail_step(
                    workflow_id, step, exec_result.error
                )
        else:
            last_error = ""
            if retry_result.attempts:
                last_error = retry_result.attempts[-1].error
            return self._fail_step(workflow_id, step, last_error)

        return self._get_manifest(workflow_id)

    def _fail_step(
        self, workflow_id: str, step: WorkflowStep, error: str
    ) -> WorkflowManifest:
        self._event_logger.log_step_failed(
            workflow_id, step.step_id, error=error
        )
        self._update_step_status(
            workflow_id, step.step_id, StepStatus.FAILED, error=error
        )
        self._resume_mgr.save_checkpoint(
            workflow_id,
            StepCheckpoint(step.step_id, StepStatus.FAILED, error=error),
        )
        return self._get_manifest(workflow_id)

    def _update_step_status(
        self,
        workflow_id: str,
        step_id: str,
        status: StepStatus,
        output: Optional[dict] = None,
        error: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> None:
        manifest = self._get_manifest(workflow_id)
        now = datetime.now(timezone.utc)
        updated_steps = []
        for s in manifest.steps:
            if s.step_id == step_id:
                updates: dict[str, Any] = {"status": status}
                if status == StepStatus.RUNNING:
                    updates["started_at"] = now
                if status in (StepStatus.COMPLETED, StepStatus.FAILED):
                    updates["completed_at"] = now
                if output is not None:
                    updates["output"] = output
                if error is not None:
                    updates["error"] = error
                if duration is not None:
                    updates["duration_seconds"] = duration
                updated_steps.append(s.model_copy(update=updates))
            else:
                updated_steps.append(s)
        updated = manifest.model_copy(update={"steps": updated_steps})
        self._manifests[workflow_id] = updated
        self._history.add_manifest(updated)

    def _update_manifest_checksum(
        self, workflow_id: str, key: str, checksum: str
    ) -> None:
        manifest = self._get_manifest(workflow_id)
        checksums = dict(manifest.checksums)
        checksums[key] = checksum
        updated = manifest.model_copy(update={"checksums": checksums})
        self._manifests[workflow_id] = updated
        self._history.add_manifest(updated)

    @staticmethod
    def _find_step(
        manifest: WorkflowManifest, step_id: str
    ) -> Optional[WorkflowStep]:
        for s in manifest.steps:
            if s.step_id == step_id:
                return s
        return None
