"""Concurrent workflow integration tests.

Tests multiple workflows executing simultaneously and verifies
isolation — no cross-workflow interference.

Requirements: 8.13
"""

import concurrent.futures
import threading
from typing import Callable

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
from src.orchestration.workflow_isolation import WorkflowIsolation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_definition(name: str, model_id: str = "anthropic.claude-v2") -> WorkflowDefinition:
    """Create a simple two-step workflow definition."""
    return WorkflowDefinition(
        name=name,
        description=f"Concurrent test workflow: {name}",
        steps=[
            WorkflowStep(
                step_id="dataset_upload",
                step_type=StepType.DATASET_PREPARATION,
                name="Upload Dataset",
                config={"dataset_path": "/tmp/test.jsonl", "name": name},
            ),
            WorkflowStep(
                step_id="baseline_eval",
                step_type=StepType.BASELINE_EVALUATION,
                name="Baseline Evaluation",
                config={"model_id": model_id, "dataset_id": f"ds-{name}"},
                depends_on=["dataset_upload"],
            ),
        ],
    )


def _mock_step_handler(step_type: StepType) -> Callable:
    """Return a handler that succeeds and returns output."""
    def handler(step, context):
        return {
            "status": "success",
            "step_type": step_type.value,
            "artifact_uri": f"s3://bucket/results/{step_type.value}/output.json",
        }
    return handler


def _create_orchestrator() -> WorkflowOrchestrator:
    """Create an orchestrator with mock step handlers registered."""
    orch = WorkflowOrchestrator()
    for st in StepType:
        orch.register_step_handler(st, _mock_step_handler(st))
    return orch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcurrentWorkflows:
    """Integration tests for concurrent workflow execution and isolation."""

    def test_concurrent_workflows_get_unique_ids(self):
        """Multiple workflows created concurrently all receive unique IDs."""
        orch = _create_orchestrator()
        num_workflows = 10
        workflow_ids: list[str] = []
        lock = threading.Lock()

        def create_one(idx: int):
            defn = _make_simple_definition(f"wf-{idx}")
            manifest = orch.create_workflow(defn, created_by=f"user-{idx}")
            with lock:
                workflow_ids.append(manifest.workflow_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workflows) as pool:
            list(pool.map(create_one, range(num_workflows)))

        assert len(workflow_ids) == num_workflows
        assert len(set(workflow_ids)) == num_workflows, "All workflow IDs must be unique"

    def test_concurrent_workflows_complete_independently(self):
        """Multiple workflows started concurrently all complete successfully."""
        orch = _create_orchestrator()
        num_workflows = 5
        results: list[WorkflowManifest] = []
        lock = threading.Lock()

        definitions = [
            _make_simple_definition(f"concurrent-{i}") for i in range(num_workflows)
        ]
        manifests = [
            orch.create_workflow(d, created_by=f"user-{i}")
            for i, d in enumerate(definitions)
        ]

        def run_one(manifest: WorkflowManifest):
            result = orch.start_workflow(manifest.workflow_id)
            with lock:
                results.append(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workflows) as pool:
            list(pool.map(run_one, manifests))

        assert len(results) == num_workflows
        for result in results:
            assert result.status == WorkflowStatus.COMPLETED
            for step in result.steps:
                assert step.status == StepStatus.COMPLETED

    def test_workflow_state_isolation(self):
        """State set in one workflow scope is not visible in another."""
        isolation = WorkflowIsolation()
        isolation.create_scope("wf-a")
        isolation.create_scope("wf-b")

        isolation.set_state("wf-a", "model_id", "model-alpha")
        isolation.set_state("wf-b", "model_id", "model-beta")

        assert isolation.get_state("wf-a", "model_id") == "model-alpha"
        assert isolation.get_state("wf-b", "model_id") == "model-beta"

        # Keys in one scope don't leak to the other
        isolation.set_state("wf-a", "secret", "alpha-only")
        assert isolation.get_state("wf-b", "secret") is None

    def test_concurrent_state_writes_no_interference(self):
        """Concurrent writes to different workflow scopes don't interfere."""
        isolation = WorkflowIsolation()
        num_scopes = 20
        errors: list[str] = []
        lock = threading.Lock()

        for i in range(num_scopes):
            isolation.create_scope(f"wf-{i}")

        def write_and_verify(idx: int):
            wf_id = f"wf-{idx}"
            for k in range(50):
                isolation.set_state(wf_id, f"key-{k}", f"value-{idx}-{k}")
            # Verify all values belong to this scope
            for k in range(50):
                val = isolation.get_state(wf_id, f"key-{k}")
                expected = f"value-{idx}-{k}"
                if val != expected:
                    with lock:
                        errors.append(
                            f"{wf_id} key-{k}: expected {expected!r}, got {val!r}"
                        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_scopes) as pool:
            list(pool.map(write_and_verify, range(num_scopes)))

        assert errors == [], f"Cross-workflow interference detected: {errors}"

    def test_concurrent_workflow_execution_isolation(self):
        """Workflows executed concurrently don't corrupt each other's step outputs."""
        orch = _create_orchestrator()
        num_workflows = 5

        definitions = [
            _make_simple_definition(f"iso-{i}", model_id=f"model-{i}")
            for i in range(num_workflows)
        ]
        manifests = [
            orch.create_workflow(d, created_by=f"user-{i}")
            for i, d in enumerate(definitions)
        ]

        results: dict[str, WorkflowManifest] = {}
        lock = threading.Lock()

        def run_one(idx: int):
            m = manifests[idx]
            result = orch.start_workflow(m.workflow_id)
            with lock:
                results[m.workflow_id] = result

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workflows) as pool:
            list(pool.map(run_one, range(num_workflows)))

        # Each workflow should have its own steps and outputs
        assert len(results) == num_workflows
        seen_ids = set()
        for wf_id, result in results.items():
            assert wf_id not in seen_ids
            seen_ids.add(wf_id)
            assert result.status == WorkflowStatus.COMPLETED
            assert len(result.steps) == 2

    def test_scope_destruction_does_not_affect_other_scopes(self):
        """Destroying one workflow scope leaves other scopes intact."""
        isolation = WorkflowIsolation()
        isolation.create_scope("wf-keep")
        isolation.create_scope("wf-destroy")

        isolation.set_state("wf-keep", "data", "important")
        isolation.set_state("wf-destroy", "data", "temporary")

        isolation.destroy_scope("wf-destroy")

        assert isolation.get_state("wf-keep", "data") == "important"
        assert isolation.get_scope("wf-destroy") is None

    def test_concurrent_create_and_execute(self):
        """Workflows can be created and executed concurrently without errors."""
        orch = _create_orchestrator()
        num_workflows = 8
        results: list[WorkflowManifest] = []
        lock = threading.Lock()

        def create_and_run(idx: int):
            defn = _make_simple_definition(f"full-{idx}")
            manifest = orch.create_workflow(defn, created_by=f"user-{idx}")
            result = orch.start_workflow(manifest.workflow_id)
            with lock:
                results.append(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workflows) as pool:
            list(pool.map(create_and_run, range(num_workflows)))

        assert len(results) == num_workflows
        completed = [r for r in results if r.status == WorkflowStatus.COMPLETED]
        assert len(completed) == num_workflows
