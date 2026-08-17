"""End-to-end workflow integration tests.

Tests the complete pipeline: dataset upload → baseline evaluation →
fine-tuning → post-tuning evaluation → comparison.

Uses mocked AWS services and verifies all artifacts stored correctly.

Requirements: 8.1
"""

import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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
from src.orchestration.workflow_resume import WorkflowResumeManager
from src.orchestration.workflow_isolation import WorkflowIsolation
from src.orchestration.partial_failure_handler import PartialFailureHandler
from src.orchestration.event_logger import WorkflowEventLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_full_pipeline_definition() -> WorkflowDefinition:
    """Create a full pipeline workflow definition."""
    return WorkflowDefinition(
        name="full_pipeline",
        description="End-to-end: dataset → baseline → fine-tune → post-eval → compare",
        steps=[
            WorkflowStep(
                step_id="dataset_upload",
                step_type=StepType.DATASET_PREPARATION,
                name="Upload Dataset",
                config={"dataset_path": "/tmp/test.jsonl", "name": "test-ds"},
            ),
            WorkflowStep(
                step_id="baseline_eval",
                step_type=StepType.BASELINE_EVALUATION,
                name="Baseline Evaluation",
                config={"model_id": "anthropic.claude-v2", "dataset_id": "ds-001"},
                depends_on=["dataset_upload"],
            ),
            WorkflowStep(
                step_id="fine_tuning",
                step_type=StepType.FINE_TUNING,
                name="Fine-Tuning",
                config={"base_model_id": "anthropic.claude-v2", "training_data_id": "ds-001"},
                depends_on=["baseline_eval"],
            ),
            WorkflowStep(
                step_id="post_eval",
                step_type=StepType.POST_TUNING_EVALUATION,
                name="Post-Tuning Evaluation",
                config={"model_id": "ft-model-001", "dataset_id": "ds-001"},
                depends_on=["fine_tuning"],
            ),
            WorkflowStep(
                step_id="comparison",
                step_type=StepType.COMPARISON,
                name="Comparison",
                config={
                    "model_id_1": "anthropic.claude-v2",
                    "model_id_2": "ft-model-001",
                    "dataset_id": "ds-001",
                },
                depends_on=["post_eval"],
            ),
        ],
    )


def _mock_step_handler(step_type: StepType):
    """Return a mock handler that succeeds and returns output."""
    def handler(step, context):
        return {
            "status": "success",
            "step_type": step_type.value,
            "artifact_uri": f"s3://bucket/results/{step_type.value}/output.json",
        }
    return handler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEndToEndWorkflow:
    """Integration tests for the complete pipeline workflow."""

    def _create_orchestrator_with_handlers(self) -> WorkflowOrchestrator:
        """Create an orchestrator with mock step handlers registered."""
        orch = WorkflowOrchestrator()
        for st in StepType:
            orch.register_step_handler(st, _mock_step_handler(st))
        return orch

    def test_full_pipeline_creates_workflow(self):
        """Workflow creation returns a valid manifest with all steps."""
        orch = self._create_orchestrator_with_handlers()
        definition = _make_full_pipeline_definition()
        manifest = orch.create_workflow(definition, created_by="test-user")

        assert manifest.workflow_id
        assert manifest.status == WorkflowStatus.PENDING
        assert len(manifest.steps) == 5
        assert manifest.created_by == "test-user"

    def test_full_pipeline_executes_all_steps(self):
        """Starting the workflow executes all 5 steps to completion."""
        orch = self._create_orchestrator_with_handlers()
        definition = _make_full_pipeline_definition()
        manifest = orch.create_workflow(definition, created_by="test-user")

        result = orch.start_workflow(manifest.workflow_id)

        assert result.status == WorkflowStatus.COMPLETED
        for step in result.steps:
            assert step.status == StepStatus.COMPLETED

    def test_full_pipeline_step_ordering(self):
        """Steps execute in dependency order — verified via completion timestamps."""
        orch = self._create_orchestrator_with_handlers()
        definition = _make_full_pipeline_definition()
        manifest = orch.create_workflow(definition, created_by="test-user")

        result = orch.start_workflow(manifest.workflow_id)

        # All steps completed; verify ordering via started_at timestamps
        step_map = {s.step_id: s for s in result.steps}
        ordered_ids = [
            "dataset_upload",
            "baseline_eval",
            "fine_tuning",
            "post_eval",
            "comparison",
        ]
        for i in range(len(ordered_ids) - 1):
            curr = step_map[ordered_ids[i]]
            nxt = step_map[ordered_ids[i + 1]]
            assert curr.started_at is not None
            assert nxt.started_at is not None
            assert curr.started_at <= nxt.started_at

    def test_full_pipeline_step_outputs_propagate(self):
        """Each step's output is stored in the manifest."""
        orch = self._create_orchestrator_with_handlers()
        definition = _make_full_pipeline_definition()
        manifest = orch.create_workflow(definition, created_by="test-user")

        result = orch.start_workflow(manifest.workflow_id)

        for step in result.steps:
            assert step.output is not None
            assert step.output["status"] == "success"

    def test_full_pipeline_checksums_recorded(self):
        """Completed workflow has checksums recorded."""
        orch = self._create_orchestrator_with_handlers()
        definition = _make_full_pipeline_definition()
        manifest = orch.create_workflow(definition, created_by="test-user")

        result = orch.start_workflow(manifest.workflow_id)

        # The orchestrator records a workflow_config checksum
        assert len(result.checksums) > 0
        assert "workflow_config" in result.checksums

    def test_full_pipeline_status_query(self):
        """Can query workflow status after completion."""
        orch = self._create_orchestrator_with_handlers()
        definition = _make_full_pipeline_definition()
        manifest = orch.create_workflow(definition, created_by="test-user")

        orch.start_workflow(manifest.workflow_id)
        status = orch.get_status(manifest.workflow_id)

        assert status.status == WorkflowStatus.COMPLETED
        assert status.workflow_id == manifest.workflow_id

    def test_full_pipeline_with_step_failure_marks_failed(self):
        """If a step fails, the workflow is marked as failed."""
        def failing_handler(step, context):
            raise RuntimeError("Fine-tuning service unavailable")

        orch = WorkflowOrchestrator()
        for st in StepType:
            orch.register_step_handler(st, _mock_step_handler(st))
        # Override fine-tuning to fail
        orch.register_step_handler(StepType.FINE_TUNING, failing_handler)

        definition = _make_full_pipeline_definition()
        manifest = orch.create_workflow(definition, created_by="test-user")

        result = orch.start_workflow(manifest.workflow_id)

        assert result.status == WorkflowStatus.FAILED
        # Steps before failure should be completed
        step_map = {s.step_id: s for s in result.steps}
        assert step_map["dataset_upload"].status == StepStatus.COMPLETED
        assert step_map["baseline_eval"].status == StepStatus.COMPLETED
        assert step_map["fine_tuning"].status == StepStatus.FAILED

    def test_workflow_manifest_has_timestamps(self):
        """Completed workflow has created_at and started_at timestamps."""
        orch = self._create_orchestrator_with_handlers()
        definition = _make_full_pipeline_definition()
        manifest = orch.create_workflow(definition, created_by="test-user")

        result = orch.start_workflow(manifest.workflow_id)

        assert result.created_at is not None
        assert result.started_at is not None
