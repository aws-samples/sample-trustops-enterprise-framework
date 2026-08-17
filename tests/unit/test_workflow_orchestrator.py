"""Tests for WorkflowOrchestrator class."""

import pytest

from src.data_models.workflow import (
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowManifest,
    WorkflowStatus,
    WorkflowStep,
)
from src.orchestration.workflow_orchestrator import WorkflowOrchestrator


def _make_step(step_id, step_type=StepType.BASELINE_EVALUATION, depends_on=None, config=None):
    return WorkflowStep(
        step_id=step_id,
        step_type=step_type,
        name=f"Step {step_id}",
        config=config or {},
        depends_on=depends_on or [],
    )


def _make_definition(steps=None, name="Test Workflow"):
    steps = steps or [_make_step("s1")]
    return WorkflowDefinition(name=name, steps=steps)


def _success_handler(step_id, config):
    return {"result": f"ok-{step_id}"}


def _failure_handler(step_id, config):
    raise RuntimeError(f"Step {step_id} failed")


class TestCreateWorkflow:
    def test_creates_pending_workflow(self):
        orch = WorkflowOrchestrator()
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        assert manifest.status == WorkflowStatus.PENDING
        assert manifest.created_by == "user-1"
        assert manifest.workflow_id.startswith("wf-")

    def test_workflow_in_history(self):
        orch = WorkflowOrchestrator()
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        results = orch.list_workflows()
        assert len(results) == 1
        assert results[0].workflow_id == manifest.workflow_id

    def test_get_status(self):
        orch = WorkflowOrchestrator()
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        status = orch.get_status(manifest.workflow_id)
        assert status.status == WorkflowStatus.PENDING


class TestStartWorkflow:
    def test_start_single_step_success(self):
        orch = WorkflowOrchestrator()
        orch.register_step_handler("baseline_evaluation", _success_handler)
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        result = orch.start_workflow(manifest.workflow_id)
        assert result.status == WorkflowStatus.COMPLETED

    def test_step_output_recorded(self):
        orch = WorkflowOrchestrator()
        orch.register_step_handler("baseline_evaluation", _success_handler)
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        result = orch.start_workflow(manifest.workflow_id)
        step = result.steps[0]
        assert step.status == StepStatus.COMPLETED
        assert step.output == {"result": "ok-s1"}

    def test_start_multi_step_success(self):
        orch = WorkflowOrchestrator()
        orch.register_step_handler("dataset_preparation", _success_handler)
        orch.register_step_handler("baseline_evaluation", _success_handler)
        steps = [
            _make_step("s1", StepType.DATASET_PREPARATION),
            _make_step("s2", StepType.BASELINE_EVALUATION, depends_on=["s1"]),
        ]
        defn = _make_definition(steps)
        manifest = orch.create_workflow(defn, "user-1")
        result = orch.start_workflow(manifest.workflow_id)
        assert result.status == WorkflowStatus.COMPLETED
        assert all(s.status == StepStatus.COMPLETED for s in result.steps)

    def test_start_with_failure(self):
        orch = WorkflowOrchestrator()
        orch.register_step_handler("baseline_evaluation", _failure_handler)
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        result = orch.start_workflow(manifest.workflow_id)
        assert result.status == WorkflowStatus.FAILED

    def test_cannot_start_completed_workflow(self):
        orch = WorkflowOrchestrator()
        orch.register_step_handler("baseline_evaluation", _success_handler)
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        orch.start_workflow(manifest.workflow_id)
        with pytest.raises(ValueError):
            orch.start_workflow(manifest.workflow_id)

    def test_checksums_added_on_completion(self):
        orch = WorkflowOrchestrator()
        orch.register_step_handler("baseline_evaluation", _success_handler)
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        result = orch.start_workflow(manifest.workflow_id)
        assert "workflow_config" in result.checksums


class TestPauseResumeCancel:
    def _make_running_orchestrator(self):
        """Helper to create an orchestrator with a paused workflow."""
        orch = WorkflowOrchestrator()
        # Use approval gate to pause
        steps = [
            _make_step("s1", StepType.APPROVAL_GATE),
            _make_step("s2", StepType.BASELINE_EVALUATION, depends_on=["s1"]),
        ]
        defn = _make_definition(steps)
        manifest = orch.create_workflow(defn, "user-1")
        # Starting will pause at approval gate
        result = orch.start_workflow(manifest.workflow_id)
        return orch, result

    def test_pause_pauses_at_approval_gate(self):
        orch, result = self._make_running_orchestrator()
        assert result.status == WorkflowStatus.PAUSED

    def test_cancel_workflow(self):
        orch = WorkflowOrchestrator()
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        result = orch.cancel_workflow(manifest.workflow_id)
        assert result.status == WorkflowStatus.CANCELLED

    def test_cannot_cancel_completed(self):
        orch = WorkflowOrchestrator()
        orch.register_step_handler("baseline_evaluation", _success_handler)
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        orch.start_workflow(manifest.workflow_id)
        with pytest.raises(ValueError):
            orch.cancel_workflow(manifest.workflow_id)


class TestResumeWorkflow:
    def test_resume_failed_workflow(self):
        orch = WorkflowOrchestrator()
        call_count = {"n": 0}

        def flaky_handler(step_id, config):
            call_count["n"] += 1
            if call_count["n"] <= 4:
                # First 4 calls fail (initial + 3 retries)
                raise RuntimeError("transient")
            return {"result": "ok"}

        orch.register_step_handler("baseline_evaluation", flaky_handler)
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        result = orch.start_workflow(manifest.workflow_id)
        assert result.status == WorkflowStatus.FAILED

        # Now fix the handler and resume
        orch.register_step_handler("baseline_evaluation", _success_handler)
        result = orch.resume_workflow(manifest.workflow_id)
        assert result.status == WorkflowStatus.COMPLETED

    def test_cannot_resume_pending(self):
        orch = WorkflowOrchestrator()
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        with pytest.raises(ValueError):
            orch.resume_workflow(manifest.workflow_id)


class TestListWorkflows:
    def test_list_all(self):
        orch = WorkflowOrchestrator()
        orch.create_workflow(_make_definition(), "user-1")
        orch.create_workflow(_make_definition(), "user-2")
        results = orch.list_workflows()
        assert len(results) == 2

    def test_list_by_user(self):
        orch = WorkflowOrchestrator()
        orch.create_workflow(_make_definition(), "alice")
        orch.create_workflow(_make_definition(), "bob")
        results = orch.list_workflows(created_by="alice")
        assert len(results) == 1

    def test_list_by_status(self):
        orch = WorkflowOrchestrator()
        orch.register_step_handler("baseline_evaluation", _success_handler)
        m1 = orch.create_workflow(_make_definition(), "user-1")
        orch.start_workflow(m1.workflow_id)
        orch.create_workflow(_make_definition(), "user-2")
        completed = orch.list_workflows(status=WorkflowStatus.COMPLETED)
        pending = orch.list_workflows(status=WorkflowStatus.PENDING)
        assert len(completed) == 1
        assert len(pending) == 1


class TestReproduceWorkflow:
    def test_reproduce_creates_new_workflow(self):
        orch = WorkflowOrchestrator()
        orch.register_step_handler("baseline_evaluation", _success_handler)
        defn = _make_definition()
        manifest = orch.create_workflow(defn, "user-1")
        orch.start_workflow(manifest.workflow_id)
        result = orch.reproduce_workflow(manifest.workflow_id, "user-2")
        assert result.success is True
        assert result.new_workflow_id != manifest.workflow_id
        new_manifest = orch.get_status(result.new_workflow_id)
        assert new_manifest.status == WorkflowStatus.PENDING

    def test_reproduce_not_found(self):
        orch = WorkflowOrchestrator()
        result = orch.reproduce_workflow("nonexistent")
        assert result.success is False


class TestEstimateWorkflow:
    def test_estimate(self):
        orch = WorkflowOrchestrator()
        defn = _make_definition([
            _make_step("s1", StepType.DATASET_PREPARATION),
            _make_step("s2", StepType.BASELINE_EVALUATION),
        ])
        est = orch.estimate_workflow(defn)
        assert est.estimated_duration_seconds > 0
        assert est.estimated_cost > 0


class TestGetTemplates:
    def test_returns_templates(self):
        orch = WorkflowOrchestrator()
        templates = orch.get_workflow_templates()
        assert len(templates) >= 3
        names = {t.name for t in templates}
        assert "Full Pipeline" in names


class TestWorkflowNotFound:
    def test_get_status_not_found(self):
        orch = WorkflowOrchestrator()
        with pytest.raises(KeyError):
            orch.get_status("nonexistent")

    def test_start_not_found(self):
        orch = WorkflowOrchestrator()
        with pytest.raises(KeyError):
            orch.start_workflow("nonexistent")
