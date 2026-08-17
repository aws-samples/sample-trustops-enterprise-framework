"""
Unit tests for the workflow ID generator.

Requirements: 8.2, 8.5
"""

from src.orchestration.workflow_id import (
    WorkflowStartEvent,
    create_workflow_start_event,
    generate_workflow_id,
)


class TestGenerateWorkflowId:
    def test_returns_string(self):
        wid = generate_workflow_id()
        assert isinstance(wid, str)

    def test_default_prefix(self):
        wid = generate_workflow_id()
        assert wid.startswith("wf-")

    def test_custom_prefix(self):
        wid = generate_workflow_id(prefix="eval")
        assert wid.startswith("eval-")

    def test_unique_ids(self):
        ids = {generate_workflow_id() for _ in range(100)}
        assert len(ids) == 100

    def test_id_length(self):
        wid = generate_workflow_id()
        # "wf-" + 12 hex chars = 15
        assert len(wid) == 15


class TestCreateWorkflowStartEvent:
    def test_creates_event(self):
        event = create_workflow_start_event("wf-abc123")
        assert isinstance(event, WorkflowStartEvent)
        assert event.workflow_id == "wf-abc123"
        assert event.timestamp is not None

    def test_with_created_by(self):
        event = create_workflow_start_event("wf-abc123", created_by="user-1")
        assert event.created_by == "user-1"

    def test_with_metadata(self):
        meta = {"workflow_name": "full-pipeline"}
        event = create_workflow_start_event("wf-abc123", metadata=meta)
        assert event.metadata["workflow_name"] == "full-pipeline"

    def test_default_metadata_is_empty(self):
        event = create_workflow_start_event("wf-abc123")
        assert event.metadata == {}

    def test_default_created_by_is_empty(self):
        event = create_workflow_start_event("wf-abc123")
        assert event.created_by == ""
