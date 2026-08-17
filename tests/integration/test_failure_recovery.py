"""Failure recovery integration tests.

Tests workflow resume after simulated failures, verifying progress
persistence and recovery from the last successful step.

Requirements: 8.15
"""

from typing import Callable

from src.data_models.workflow import (
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowStatus,
    WorkflowStep,
)
from src.orchestration.workflow_orchestrator import WorkflowOrchestrator
from src.orchestration.workflow_resume import WorkflowResumeManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Track how many times each step handler has been called, keyed by step_id.
_call_counts: dict[str, int] = {}


def _reset_call_counts() -> None:
    _call_counts.clear()


def _make_pipeline_definition() -> WorkflowDefinition:
    """Create a 4-step sequential pipeline for failure/recovery testing."""
    return WorkflowDefinition(
        name="failure_recovery_pipeline",
        description="Pipeline for testing failure recovery",
        steps=[
            WorkflowStep(
                step_id="step_1",
                step_type=StepType.DATASET_PREPARATION,
                name="Prepare Dataset",
                config={"dataset_path": "/tmp/data.jsonl", "name": "ds-1"},
            ),
            WorkflowStep(
                step_id="step_2",
                step_type=StepType.BASELINE_EVALUATION,
                name="Baseline Evaluation",
                config={"model_id": "anthropic.claude-v2", "dataset_id": "ds-1"},
                depends_on=["step_1"],
            ),
            WorkflowStep(
                step_id="step_3",
                step_type=StepType.FINE_TUNING,
                name="Fine-Tuning",
                config={"base_model_id": "anthropic.claude-v2", "training_data_id": "ds-1"},
                depends_on=["step_2"],
            ),
            WorkflowStep(
                step_id="step_4",
                step_type=StepType.COMPARISON,
                name="Comparison",
                config={"model_id_1": "anthropic.claude-v2", "model_id_2": "ft-model"},
                depends_on=["step_3"],
            ),
        ],
    )


def _success_handler(step_id: str, config: dict) -> dict:
    """Handler that always succeeds and records the call."""
    _call_counts[step_id] = _call_counts.get(step_id, 0) + 1
    return {
        "status": "success",
        "step_id": step_id,
        "artifact_uri": f"s3://bucket/results/{step_id}/output.json",
    }


def _make_failing_handler(fail_step_id: str, fail_until_attempt: int = 1):
    """Return a handler that fails for a specific step until N attempts.

    After ``fail_until_attempt`` calls for the target step, the handler
    succeeds.  All other steps always succeed.
    """

    def handler(step_id: str, config: dict) -> dict:
        _call_counts[step_id] = _call_counts.get(step_id, 0) + 1
        if step_id == fail_step_id and _call_counts[step_id] <= fail_until_attempt:
            raise RuntimeError(f"Simulated failure in {step_id}")
        return {
            "status": "success",
            "step_id": step_id,
            "artifact_uri": f"s3://bucket/results/{step_id}/output.json",
        }

    return handler


def _create_orchestrator_with_handler(handler: Callable) -> WorkflowOrchestrator:
    """Create an orchestrator where every step type uses the same handler."""
    orch = WorkflowOrchestrator()
    for st in StepType:
        orch.register_step_handler(st, handler)
    return orch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFailureRecovery:
    """Integration tests for workflow failure recovery and resume."""

    def setup_method(self):
        _reset_call_counts()

    # -- 1. Failure sets workflow status to FAILED --------------------------

    def test_step_failure_marks_workflow_failed(self):
        """A step raising an exception causes the workflow to be FAILED."""
        handler = _make_failing_handler("step_3", fail_until_attempt=999)
        orch = _create_orchestrator_with_handler(handler)

        defn = _make_pipeline_definition()
        manifest = orch.create_workflow(defn, created_by="test-user")
        result = orch.start_workflow(manifest.workflow_id)

        assert result.status == WorkflowStatus.FAILED

    # -- 2. Completed steps are recorded before failure ---------------------

    def test_completed_steps_persisted_before_failure(self):
        """Steps completed before the failure are recorded as COMPLETED."""
        handler = _make_failing_handler("step_3", fail_until_attempt=999)
        orch = _create_orchestrator_with_handler(handler)

        defn = _make_pipeline_definition()
        manifest = orch.create_workflow(defn, created_by="test-user")
        result = orch.start_workflow(manifest.workflow_id)

        step_map = {s.step_id: s for s in result.steps}
        assert step_map["step_1"].status == StepStatus.COMPLETED
        assert step_map["step_2"].status == StepStatus.COMPLETED
        assert step_map["step_3"].status == StepStatus.FAILED
        assert step_map["step_4"].status == StepStatus.PENDING

    # -- 3. Progress persistence via resume manager -------------------------

    def test_checkpoints_saved_for_completed_steps(self):
        """The resume manager has checkpoints for every completed step."""
        handler = _make_failing_handler("step_3", fail_until_attempt=999)
        orch = _create_orchestrator_with_handler(handler)

        defn = _make_pipeline_definition()
        manifest = orch.create_workflow(defn, created_by="test-user")
        orch.start_workflow(manifest.workflow_id)

        # Access the internal resume manager to verify persistence
        resume_mgr: WorkflowResumeManager = orch._resume_mgr
        assert resume_mgr.is_step_completed(manifest.workflow_id, "step_1")
        assert resume_mgr.is_step_completed(manifest.workflow_id, "step_2")
        assert not resume_mgr.is_step_completed(manifest.workflow_id, "step_3")

    # -- 4. Resume skips completed steps and finishes -----------------------

    def test_resume_skips_completed_steps(self):
        """Resuming a failed workflow skips already-completed steps."""
        # First run: step_3 fails on every attempt (fail_until_attempt=999)
        handler = _make_failing_handler("step_3", fail_until_attempt=999)
        orch = _create_orchestrator_with_handler(handler)

        defn = _make_pipeline_definition()
        manifest = orch.create_workflow(defn, created_by="test-user")
        orch.start_workflow(manifest.workflow_id)

        # Reset counts and swap in a handler that always succeeds
        _reset_call_counts()
        for st in StepType:
            orch.register_step_handler(st, _success_handler)

        # Resume
        result = orch.resume_workflow(manifest.workflow_id)

        assert result.status == WorkflowStatus.COMPLETED

        # step_1 and step_2 should NOT have been called again
        assert "step_1" not in _call_counts
        assert "step_2" not in _call_counts
        # step_3 and step_4 should have been executed
        assert _call_counts.get("step_3", 0) >= 1
        assert _call_counts.get("step_4", 0) >= 1

    # -- 5. Resumed workflow completes successfully -------------------------

    def test_resumed_workflow_completes_all_steps(self):
        """After resume, every step ends up COMPLETED."""
        handler = _make_failing_handler("step_3", fail_until_attempt=999)
        orch = _create_orchestrator_with_handler(handler)

        defn = _make_pipeline_definition()
        manifest = orch.create_workflow(defn, created_by="test-user")
        orch.start_workflow(manifest.workflow_id)

        # Fix the handler and resume
        for st in StepType:
            orch.register_step_handler(st, _success_handler)

        result = orch.resume_workflow(manifest.workflow_id)

        assert result.status == WorkflowStatus.COMPLETED
        for step in result.steps:
            assert step.status == StepStatus.COMPLETED

    # -- 6. Step outputs correct after recovery -----------------------------

    def test_step_outputs_correct_after_recovery(self):
        """All steps have correct output data after a successful resume."""
        handler = _make_failing_handler("step_2", fail_until_attempt=999)
        orch = _create_orchestrator_with_handler(handler)

        defn = _make_pipeline_definition()
        manifest = orch.create_workflow(defn, created_by="test-user")
        orch.start_workflow(manifest.workflow_id)

        # Fix and resume
        for st in StepType:
            orch.register_step_handler(st, _success_handler)

        result = orch.resume_workflow(manifest.workflow_id)

        for step in result.steps:
            assert step.output is not None
            assert step.output["status"] == "success"
            assert step.output["step_id"] == step.step_id

    # -- 7. Resume point correctly identifies next steps --------------------

    def test_resume_point_identifies_failed_and_next_steps(self):
        """The resume manager correctly reports completed/failed/next steps."""
        handler = _make_failing_handler("step_3", fail_until_attempt=999)
        orch = _create_orchestrator_with_handler(handler)

        defn = _make_pipeline_definition()
        manifest = orch.create_workflow(defn, created_by="test-user")
        orch.start_workflow(manifest.workflow_id)

        resume_mgr: WorkflowResumeManager = orch._resume_mgr
        all_ids = [s.step_id for s in defn.steps]
        deps = {s.step_id: list(s.depends_on) for s in defn.steps}
        resume_point = resume_mgr.get_resume_point(
            manifest.workflow_id, all_ids, deps
        )

        assert "step_1" in resume_point.completed_step_ids
        assert "step_2" in resume_point.completed_step_ids
        assert "step_3" in resume_point.failed_step_ids
        assert "step_3" in resume_point.next_step_ids

    # -- 8. Failure at first step, full recovery ----------------------------

    def test_failure_at_first_step_full_recovery(self):
        """Failure at the very first step can be recovered completely."""
        handler = _make_failing_handler("step_1", fail_until_attempt=999)
        orch = _create_orchestrator_with_handler(handler)

        defn = _make_pipeline_definition()
        manifest = orch.create_workflow(defn, created_by="test-user")
        result = orch.start_workflow(manifest.workflow_id)

        assert result.status == WorkflowStatus.FAILED
        step_map = {s.step_id: s for s in result.steps}
        assert step_map["step_1"].status == StepStatus.FAILED

        # Fix and resume
        for st in StepType:
            orch.register_step_handler(st, _success_handler)

        result = orch.resume_workflow(manifest.workflow_id)
        assert result.status == WorkflowStatus.COMPLETED
        for step in result.steps:
            assert step.status == StepStatus.COMPLETED
