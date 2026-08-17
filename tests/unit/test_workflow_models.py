"""
Unit tests for workflow data models.

Tests Requirements: 8.1, 8.2, 8.3, 8.7
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from src.data_models.workflow import (
    WorkflowStatus,
    StepStatus,
    StepType,
    WorkflowStep,
    WorkflowDefinition,
    WorkflowEvent,
    ArtifactReference,
    WorkflowManifest,
)


class TestWorkflowStatus:
    """Test WorkflowStatus enum."""

    def test_workflow_status_enum_values(self):
        """Test WorkflowStatus enum has all required values."""
        assert WorkflowStatus.PENDING == "pending"
        assert WorkflowStatus.RUNNING == "running"
        assert WorkflowStatus.PAUSED == "paused"
        assert WorkflowStatus.COMPLETED == "completed"
        assert WorkflowStatus.FAILED == "failed"
        assert WorkflowStatus.CANCELLED == "cancelled"


class TestStepStatus:
    """Test StepStatus enum."""

    def test_step_status_enum_values(self):
        """Test StepStatus enum has all required values."""
        assert StepStatus.PENDING == "pending"
        assert StepStatus.RUNNING == "running"
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"


class TestStepType:
    """Test StepType enum."""

    def test_step_type_enum_values(self):
        """Test StepType enum has all required values."""
        assert StepType.DATASET_PREPARATION == "dataset_preparation"
        assert StepType.BASELINE_EVALUATION == "baseline_evaluation"
        assert StepType.FINE_TUNING == "fine_tuning"
        assert StepType.POST_TUNING_EVALUATION == "post_tuning_evaluation"
        assert StepType.COMPARISON == "comparison"
        assert StepType.APPROVAL_GATE == "approval_gate"
        assert StepType.CUSTOM == "custom"


class TestWorkflowStep:
    """Test WorkflowStep data model."""

    def test_workflow_step_minimal_creation(self):
        """Test creating WorkflowStep with minimal required fields."""
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Baseline Evaluation",
        )

        assert step.step_id == "step-001"
        assert step.step_type == StepType.BASELINE_EVALUATION
        assert step.name == "Baseline Evaluation"
        assert step.config == {}
        assert step.status == StepStatus.PENDING
        assert step.depends_on == []
        assert step.started_at is None
        assert step.completed_at is None
        assert step.duration_seconds is None
        assert step.output is None
        assert step.error is None
        assert step.retry_count == 0

    def test_workflow_step_full_creation(self):
        """Test creating WorkflowStep with all fields."""
        now = datetime.now()
        step = WorkflowStep(
            step_id="step-002",
            step_type=StepType.FINE_TUNING,
            name="Fine-tune Model",
            config={"model_id": "model-123", "epochs": 3},
            status=StepStatus.COMPLETED,
            depends_on=["step-001"],
            started_at=now,
            completed_at=now,
            duration_seconds=3600.5,
            output={
                "model_arn": (
                    "arn:aws:bedrock:us-east-1:123456789012:model/"
                    "custom-model"
                )
            },
            error=None,
            retry_count=1,
        )

        assert step.step_id == "step-002"
        assert step.step_type == StepType.FINE_TUNING
        assert step.name == "Fine-tune Model"
        assert step.config == {"model_id": "model-123", "epochs": 3}
        assert step.status == StepStatus.COMPLETED
        assert step.depends_on == ["step-001"]
        assert step.started_at == now
        assert step.completed_at == now
        assert step.duration_seconds == 3600.5
        assert step.output == {
            "model_arn": (
                "arn:aws:bedrock:us-east-1:123456789012:model/custom-model"
            )
        }
        assert step.error is None
        assert step.retry_count == 1

    def test_workflow_step_with_error(self):
        """Test WorkflowStep with error status."""
        step = WorkflowStep(
            step_id="step-003",
            step_type=StepType.COMPARISON,
            name="Compare Models",
            status=StepStatus.FAILED,
            error="Model not found",
        )

        assert step.status == StepStatus.FAILED
        assert step.error == "Model not found"

    def test_workflow_step_empty_step_id_rejected(self):
        """Test that empty step_id is rejected."""
        with pytest.raises(ValidationError):
            WorkflowStep(
                step_id="",
                step_type=StepType.BASELINE_EVALUATION,
                name="Test Step",
            )

    def test_workflow_step_empty_name_rejected(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError):
            WorkflowStep(
                step_id="step-004",
                step_type=StepType.BASELINE_EVALUATION,
                name="",
            )

    def test_workflow_step_negative_duration_rejected(self):
        """Test that negative duration is rejected."""
        with pytest.raises(ValidationError):
            WorkflowStep(
                step_id="step-005",
                step_type=StepType.BASELINE_EVALUATION,
                name="Test Step",
                duration_seconds=-10.0,
            )

    def test_workflow_step_negative_retry_count_rejected(self):
        """Test that negative retry_count is rejected."""
        with pytest.raises(ValidationError):
            WorkflowStep(
                step_id="step-006",
                step_type=StepType.BASELINE_EVALUATION,
                name="Test Step",
                retry_count=-1,
            )


class TestWorkflowDefinition:
    """Test WorkflowDefinition data model."""

    def test_workflow_definition_minimal_creation(self):
        """Test creating WorkflowDefinition with minimal required fields."""
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Baseline Evaluation",
        )

        definition = WorkflowDefinition(
            name="Baseline Workflow",
            steps=[step],
        )

        assert definition.name == "Baseline Workflow"
        assert definition.description is None
        assert len(definition.steps) == 1
        assert definition.steps[0] == step
        assert definition.requires_approval is False
        assert definition.max_retries == 3
        assert definition.retry_delay_seconds == 60
        assert definition.timeout_seconds is None

    def test_workflow_definition_full_creation(self):
        """Test creating WorkflowDefinition with all fields."""
        step1 = WorkflowStep(
            step_id="step-001",
            step_type=StepType.DATASET_PREPARATION,
            name="Prepare Dataset",
        )
        step2 = WorkflowStep(
            step_id="step-002",
            step_type=StepType.BASELINE_EVALUATION,
            name="Baseline Evaluation",
            depends_on=["step-001"],
        )

        definition = WorkflowDefinition(
            name="Complete Workflow",
            description="End-to-end evaluation workflow",
            steps=[step1, step2],
            requires_approval=True,
            max_retries=5,
            retry_delay_seconds=120,
            timeout_seconds=7200,
        )

        assert definition.name == "Complete Workflow"
        assert definition.description == "End-to-end evaluation workflow"
        assert len(definition.steps) == 2
        assert definition.requires_approval is True
        assert definition.max_retries == 5
        assert definition.retry_delay_seconds == 120
        assert definition.timeout_seconds == 7200

    def test_workflow_definition_empty_name_rejected(self):
        """Test that empty name is rejected."""
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Test Step",
        )

        with pytest.raises(ValidationError):
            WorkflowDefinition(
                name="",
                steps=[step],
            )

    def test_workflow_definition_empty_steps_rejected(self):
        """Test that empty steps list is rejected."""
        with pytest.raises(ValidationError):
            WorkflowDefinition(
                name="Test Workflow",
                steps=[],
            )

    def test_workflow_definition_negative_max_retries_rejected(self):
        """Test that negative max_retries is rejected."""
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Test Step",
        )

        with pytest.raises(ValidationError):
            WorkflowDefinition(
                name="Test Workflow",
                steps=[step],
                max_retries=-1,
            )

    def test_workflow_definition_negative_timeout_rejected(self):
        """Test that negative timeout is rejected."""
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Test Step",
        )

        with pytest.raises(ValidationError):
            WorkflowDefinition(
                name="Test Workflow",
                steps=[step],
                timeout_seconds=-100,
            )


class TestWorkflowEvent:
    """Test WorkflowEvent data model."""

    def test_workflow_event_minimal_creation(self):
        """Test creating WorkflowEvent with minimal required fields."""
        now = datetime.now()
        event = WorkflowEvent(
            event_id="event-001",
            workflow_id="workflow-123",
            event_type="started",
            timestamp=now,
        )

        assert event.event_id == "event-001"
        assert event.workflow_id == "workflow-123"
        assert event.step_id is None
        assert event.event_type == "started"
        assert event.timestamp == now
        assert event.details == {}

    def test_workflow_event_full_creation(self):
        """Test creating WorkflowEvent with all fields."""
        now = datetime.now()
        event = WorkflowEvent(
            event_id="event-002",
            workflow_id="workflow-123",
            step_id="step-001",
            event_type="step_completed",
            timestamp=now,
            details={
                "duration_seconds": 120.5,
                "output": {"result": "success"}
            },
        )

        assert event.event_id == "event-002"
        assert event.workflow_id == "workflow-123"
        assert event.step_id == "step-001"
        assert event.event_type == "step_completed"
        assert event.timestamp == now
        assert event.details == {
            "duration_seconds": 120.5,
            "output": {"result": "success"}
        }

    def test_workflow_event_empty_event_id_rejected(self):
        """Test that empty event_id is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            WorkflowEvent(
                event_id="",
                workflow_id="workflow-123",
                event_type="started",
                timestamp=now,
            )

    def test_workflow_event_empty_workflow_id_rejected(self):
        """Test that empty workflow_id is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            WorkflowEvent(
                event_id="event-003",
                workflow_id="",
                event_type="started",
                timestamp=now,
            )

    def test_workflow_event_empty_event_type_rejected(self):
        """Test that empty event_type is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            WorkflowEvent(
                event_id="event-004",
                workflow_id="workflow-123",
                event_type="",
                timestamp=now,
            )


class TestArtifactReference:
    """Test ArtifactReference data model."""

    def test_artifact_reference_creation(self):
        """Test creating ArtifactReference with all fields."""
        now = datetime.now()
        artifact = ArtifactReference(
            artifact_type="evaluation_report",
            s3_uri=(
                "s3://trustops-test-artifacts-123456789012-us-east-1/workflows/workflow-123/report.json"
            ),
            checksum="sha256:abc123def456",
            created_at=now,
        )

        assert artifact.artifact_type == "evaluation_report"
        assert artifact.s3_uri == (
            "s3://trustops-test-artifacts-123456789012-us-east-1/workflows/workflow-123/report.json"
        )
        assert artifact.checksum == "sha256:abc123def456"
        assert artifact.created_at == now

    def test_artifact_reference_empty_artifact_type_rejected(self):
        """Test that empty artifact_type is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            ArtifactReference(
                artifact_type="",
                s3_uri="s3://bucket/file.json",
                checksum="sha256:abc123",
                created_at=now,
            )

    def test_artifact_reference_empty_s3_uri_rejected(self):
        """Test that empty s3_uri is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            ArtifactReference(
                artifact_type="dataset",
                s3_uri="",
                checksum="sha256:abc123",
                created_at=now,
            )

    def test_artifact_reference_empty_checksum_rejected(self):
        """Test that empty checksum is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            ArtifactReference(
                artifact_type="dataset",
                s3_uri="s3://bucket/file.json",
                checksum="",
                created_at=now,
            )


class TestWorkflowManifest:
    """Test WorkflowManifest data model."""

    def test_workflow_manifest_minimal_creation(self):
        """Test creating WorkflowManifest with minimal required fields."""
        now = datetime.now()
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Baseline Evaluation",
        )
        definition = WorkflowDefinition(
            name="Test Workflow",
            steps=[step],
        )

        manifest = WorkflowManifest(
            workflow_id="workflow-123",
            definition=definition,
            status=WorkflowStatus.PENDING,
            created_at=now,
            steps=[step],
            created_by="user@example.com",
        )

        assert manifest.workflow_id == "workflow-123"
        assert manifest.definition == definition
        assert manifest.status == WorkflowStatus.PENDING
        assert manifest.created_at == now
        assert manifest.started_at is None
        assert manifest.completed_at is None
        assert len(manifest.steps) == 1
        assert manifest.events == []
        assert manifest.artifacts == {}
        assert manifest.checksums == {}
        assert manifest.total_cost == 0.0
        assert manifest.created_by == "user@example.com"

    def test_workflow_manifest_full_creation(self):
        """Test creating WorkflowManifest with all fields."""
        now = datetime.now()
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Baseline Evaluation",
            status=StepStatus.COMPLETED,
        )
        definition = WorkflowDefinition(
            name="Complete Workflow",
            description="Full workflow test",
            steps=[step],
        )
        event = WorkflowEvent(
            event_id="event-001",
            workflow_id="workflow-123",
            event_type="started",
            timestamp=now,
        )
        artifact = ArtifactReference(
            artifact_type="evaluation_report",
            s3_uri="s3://bucket/report.json",
            checksum="sha256:abc123",
            created_at=now,
        )

        manifest = WorkflowManifest(
            workflow_id="workflow-123",
            definition=definition,
            status=WorkflowStatus.COMPLETED,
            created_at=now,
            started_at=now,
            completed_at=now,
            steps=[step],
            events=[event],
            artifacts={"report": artifact},
            checksums={"dataset": "sha256:def456"},
            total_cost=15.75,
            created_by="user@example.com",
        )

        assert manifest.workflow_id == "workflow-123"
        assert manifest.definition == definition
        assert manifest.status == WorkflowStatus.COMPLETED
        assert manifest.created_at == now
        assert manifest.started_at == now
        assert manifest.completed_at == now
        assert len(manifest.steps) == 1
        assert len(manifest.events) == 1
        assert "report" in manifest.artifacts
        assert manifest.artifacts["report"] == artifact
        assert manifest.checksums == {"dataset": "sha256:def456"}
        assert manifest.total_cost == 15.75
        assert manifest.created_by == "user@example.com"

    def test_workflow_manifest_empty_workflow_id_rejected(self):
        """Test that empty workflow_id is rejected."""
        now = datetime.now()
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Test Step",
        )
        definition = WorkflowDefinition(
            name="Test Workflow",
            steps=[step],
        )

        with pytest.raises(ValidationError):
            WorkflowManifest(
                workflow_id="",
                definition=definition,
                status=WorkflowStatus.PENDING,
                created_at=now,
                steps=[step],
                created_by="user@example.com",
            )

    def test_workflow_manifest_empty_created_by_rejected(self):
        """Test that empty created_by is rejected."""
        now = datetime.now()
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Test Step",
        )
        definition = WorkflowDefinition(
            name="Test Workflow",
            steps=[step],
        )

        with pytest.raises(ValidationError):
            WorkflowManifest(
                workflow_id="workflow-123",
                definition=definition,
                status=WorkflowStatus.PENDING,
                created_at=now,
                steps=[step],
                created_by="",
            )

    def test_workflow_manifest_negative_total_cost_rejected(self):
        """Test that negative total_cost is rejected."""
        now = datetime.now()
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Test Step",
        )
        definition = WorkflowDefinition(
            name="Test Workflow",
            steps=[step],
        )

        with pytest.raises(ValidationError):
            WorkflowManifest(
                workflow_id="workflow-123",
                definition=definition,
                status=WorkflowStatus.PENDING,
                created_at=now,
                steps=[step],
                created_by="user@example.com",
                total_cost=-10.0,
            )


class TestDataModelSerialization:
    """Test serialization of workflow data models."""

    def test_workflow_step_serialization(self):
        """Test WorkflowStep JSON serialization."""
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Baseline Evaluation",
            config={"model_id": "model-123"},
        )

        # Serialize to dict
        data = step.model_dump()
        assert data["step_id"] == "step-001"
        assert data["step_type"] == "baseline_evaluation"
        assert data["name"] == "Baseline Evaluation"
        assert data["config"] == {"model_id": "model-123"}

        # Serialize to JSON
        json_str = step.model_dump_json()
        assert "step-001" in json_str
        assert "baseline_evaluation" in json_str

    def test_workflow_definition_serialization(self):
        """Test WorkflowDefinition JSON serialization."""
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Test Step",
        )
        definition = WorkflowDefinition(
            name="Test Workflow",
            description="Test description",
            steps=[step],
        )

        # Serialize to dict
        data = definition.model_dump()
        assert data["name"] == "Test Workflow"
        assert data["description"] == "Test description"
        assert len(data["steps"]) == 1

        # Serialize to JSON
        json_str = definition.model_dump_json()
        assert "Test Workflow" in json_str
        assert "Test description" in json_str

    def test_workflow_manifest_serialization(self):
        """Test WorkflowManifest JSON serialization."""
        now = datetime.now()
        step = WorkflowStep(
            step_id="step-001",
            step_type=StepType.BASELINE_EVALUATION,
            name="Test Step",
        )
        definition = WorkflowDefinition(
            name="Test Workflow",
            steps=[step],
        )
        manifest = WorkflowManifest(
            workflow_id="workflow-123",
            definition=definition,
            status=WorkflowStatus.PENDING,
            created_at=now,
            steps=[step],
            created_by="user@example.com",
        )

        # Serialize to dict
        data = manifest.model_dump()
        assert data["workflow_id"] == "workflow-123"
        assert data["status"] == "pending"
        assert data["created_by"] == "user@example.com"

        # Serialize to JSON
        json_str = manifest.model_dump_json()
        assert "workflow-123" in json_str
        assert "pending" in json_str
        assert "user@example.com" in json_str
