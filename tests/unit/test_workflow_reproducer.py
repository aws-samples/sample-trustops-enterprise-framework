"""Tests for workflow reproducer module."""

from datetime import datetime, timezone

import pytest

from src.data_models.workflow import (
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowManifest,
    WorkflowStatus,
    WorkflowStep,
)
from src.orchestration.workflow_reproducer import (
    WorkflowReproducer,
    _reset_steps,
)


def _make_step(step_id="s1", status=StepStatus.COMPLETED):
    return WorkflowStep(
        step_id=step_id,
        step_type=StepType.BASELINE_EVALUATION,
        name=f"Step {step_id}",
        config={"model_id": "m1"},
        status=status,
        depends_on=[],
    )


def _make_manifest(workflow_id="wf-orig", steps=None):
    steps = steps or [_make_step()]
    definition = WorkflowDefinition(
        name="Test Workflow",
        description="A test workflow",
        steps=steps,
        max_retries=5,
        retry_delay_seconds=30,
    )
    now = datetime.now(timezone.utc)
    return WorkflowManifest(
        workflow_id=workflow_id,
        definition=definition,
        status=WorkflowStatus.COMPLETED,
        created_at=now,
        started_at=now,
        completed_at=now,
        steps=steps,
        total_cost=10.0,
        created_by="original-user",
    )


class TestResetSteps:
    def test_resets_status_to_pending(self):
        steps = [_make_step(status=StepStatus.COMPLETED)]
        reset = _reset_steps(steps)
        assert reset[0].status == StepStatus.PENDING

    def test_clears_runtime_fields(self):
        step = _make_step()
        step.started_at = datetime.now(timezone.utc)
        step.completed_at = datetime.now(timezone.utc)
        step.output = {"result": "ok"}
        step.error = "some error"
        step.retry_count = 3
        reset = _reset_steps([step])
        assert reset[0].started_at is None
        assert reset[0].completed_at is None
        assert reset[0].output is None
        assert reset[0].error is None
        assert reset[0].retry_count == 0

    def test_preserves_config(self):
        step = _make_step()
        step.config = {"key": "value"}
        reset = _reset_steps([step])
        assert reset[0].config == {"key": "value"}
        # Ensure deep copy
        reset[0].config["key"] = "changed"
        assert step.config["key"] == "value"


class TestWorkflowReproducer:
    def test_reproduce_not_found(self):
        r = WorkflowReproducer()
        result = r.reproduce("nonexistent")
        assert result.success is False
        assert "not found" in result.error

    def test_reproduce_creates_new_workflow(self):
        r = WorkflowReproducer()
        original = _make_manifest()
        r.store_manifest(original)
        result = r.reproduce("wf-orig", created_by="new-user")
        assert result.success is True
        assert result.new_workflow_id != "wf-orig"
        assert result.original_workflow_id == "wf-orig"
        assert result.manifest is not None

    def test_reproduced_has_pending_status(self):
        r = WorkflowReproducer()
        r.store_manifest(_make_manifest())
        result = r.reproduce("wf-orig")
        assert result.manifest.status == WorkflowStatus.PENDING

    def test_reproduced_steps_are_pending(self):
        r = WorkflowReproducer()
        r.store_manifest(_make_manifest())
        result = r.reproduce("wf-orig")
        for step in result.manifest.steps:
            assert step.status == StepStatus.PENDING

    def test_reproduced_preserves_definition(self):
        r = WorkflowReproducer()
        original = _make_manifest()
        r.store_manifest(original)
        result = r.reproduce("wf-orig")
        new_def = result.manifest.definition
        assert new_def.name == "Test Workflow"
        assert new_def.description == "A test workflow"
        assert new_def.max_retries == 5
        assert new_def.retry_delay_seconds == 30

    def test_reproduced_preserves_step_config(self):
        r = WorkflowReproducer()
        r.store_manifest(_make_manifest())
        result = r.reproduce("wf-orig")
        assert result.manifest.steps[0].config == {"model_id": "m1"}

    def test_reproduced_stored_in_reproducer(self):
        r = WorkflowReproducer()
        r.store_manifest(_make_manifest())
        result = r.reproduce("wf-orig")
        stored = r.get_manifest(result.new_workflow_id)
        assert stored is not None
        assert stored.workflow_id == result.new_workflow_id

    def test_reproduced_has_new_created_by(self):
        r = WorkflowReproducer()
        r.store_manifest(_make_manifest())
        result = r.reproduce("wf-orig", created_by="alice")
        assert result.manifest.created_by == "alice"

    def test_reproduce_multi_step_workflow(self):
        steps = [
            _make_step("s1", StepStatus.COMPLETED),
            _make_step("s2", StepStatus.FAILED),
        ]
        r = WorkflowReproducer()
        r.store_manifest(_make_manifest(steps=steps))
        result = r.reproduce("wf-orig")
        assert len(result.manifest.steps) == 2
        assert all(s.status == StepStatus.PENDING for s in result.manifest.steps)
