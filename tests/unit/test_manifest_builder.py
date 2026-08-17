"""
Unit tests for the workflow manifest builder.

Requirements: 8.7
"""

from datetime import datetime, timezone

from src.data_models.workflow import (
    ArtifactReference,
    StepType,
    WorkflowDefinition,
    WorkflowEvent,
    WorkflowManifest,
    WorkflowStatus,
    WorkflowStep,
)
from src.orchestration.manifest_builder import ManifestBuilder


def _make_definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="test-workflow",
        steps=[
            WorkflowStep(
                step_id="s1",
                step_type=StepType.BASELINE_EVALUATION,
                name="Evaluate baseline",
            ),
            WorkflowStep(
                step_id="s2",
                step_type=StepType.FINE_TUNING,
                name="Fine-tune",
                depends_on=["s1"],
            ),
        ],
    )


def _make_event(workflow_id: str = "wf-1") -> WorkflowEvent:
    return WorkflowEvent(
        event_id="evt-1",
        workflow_id=workflow_id,
        event_type="started",
        timestamp=datetime.now(timezone.utc),
    )


def _make_artifact() -> ArtifactReference:
    return ArtifactReference(
        artifact_type="dataset",
        s3_uri="s3://bucket/datasets/data.jsonl",
        checksum="abc123",
        created_at=datetime.now(timezone.utc),
    )


class TestManifestBuilder:
    def test_build_basic_manifest(self):
        defn = _make_definition()
        builder = ManifestBuilder("wf-1", defn, "user-1")
        manifest = builder.build()
        assert isinstance(manifest, WorkflowManifest)
        assert manifest.workflow_id == "wf-1"
        assert manifest.created_by == "user-1"
        assert manifest.status == WorkflowStatus.PENDING
        assert len(manifest.steps) == 2

    def test_set_status_running(self):
        defn = _make_definition()
        builder = ManifestBuilder("wf-1", defn, "user-1")
        builder.set_status(WorkflowStatus.RUNNING)
        manifest = builder.build()
        assert manifest.status == WorkflowStatus.RUNNING
        assert manifest.started_at is not None

    def test_set_status_completed(self):
        defn = _make_definition()
        builder = ManifestBuilder("wf-1", defn, "user-1")
        builder.set_status(WorkflowStatus.COMPLETED)
        manifest = builder.build()
        assert manifest.status == WorkflowStatus.COMPLETED
        assert manifest.completed_at is not None

    def test_add_event(self):
        defn = _make_definition()
        builder = ManifestBuilder("wf-1", defn, "user-1")
        event = _make_event()
        builder.add_event(event)
        manifest = builder.build()
        assert len(manifest.events) == 1
        assert manifest.events[0].event_id == "evt-1"

    def test_add_artifact(self):
        defn = _make_definition()
        builder = ManifestBuilder("wf-1", defn, "user-1")
        artifact = _make_artifact()
        builder.add_artifact("training_data", artifact)
        manifest = builder.build()
        assert "training_data" in manifest.artifacts
        assert manifest.artifacts["training_data"].s3_uri == "s3://bucket/datasets/data.jsonl"

    def test_add_checksum(self):
        defn = _make_definition()
        builder = ManifestBuilder("wf-1", defn, "user-1")
        builder.add_checksum("dataset", "sha256:abc123")
        manifest = builder.build()
        assert manifest.checksums["dataset"] == "sha256:abc123"

    def test_set_total_cost(self):
        defn = _make_definition()
        builder = ManifestBuilder("wf-1", defn, "user-1")
        builder.set_total_cost(42.50)
        manifest = builder.build()
        assert manifest.total_cost == 42.50

    def test_update_step(self):
        defn = _make_definition()
        builder = ManifestBuilder("wf-1", defn, "user-1")
        updated_step = WorkflowStep(
            step_id="s1",
            step_type=StepType.BASELINE_EVALUATION,
            name="Evaluate baseline",
            output={"score": 0.85},
        )
        builder.update_step(updated_step)
        manifest = builder.build()
        s1 = next(s for s in manifest.steps if s.step_id == "s1")
        assert s1.output == {"score": 0.85}

    def test_fluent_chaining(self):
        defn = _make_definition()
        manifest = (
            ManifestBuilder("wf-1", defn, "user-1")
            .set_status(WorkflowStatus.RUNNING)
            .add_checksum("data", "sha256:xyz")
            .set_total_cost(10.0)
            .build()
        )
        assert manifest.status == WorkflowStatus.RUNNING
        assert manifest.checksums["data"] == "sha256:xyz"
        assert manifest.total_cost == 10.0

    def test_build_returns_copies(self):
        defn = _make_definition()
        builder = ManifestBuilder("wf-1", defn, "user-1")
        m1 = builder.build()
        builder.add_checksum("new", "val")
        m2 = builder.build()
        assert "new" not in m1.checksums
        assert "new" in m2.checksums
