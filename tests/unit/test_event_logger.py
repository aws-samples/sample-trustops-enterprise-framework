"""
Unit tests for the workflow event logger.

Requirements: 8.3
"""

from src.orchestration.event_logger import (
    EventRecord,
    WorkflowEventLogger,
    WorkflowEventType,
)


class TestWorkflowEventLogger:
    def test_log_workflow_started(self):
        logger = WorkflowEventLogger()
        event = logger.log_workflow_started("wf-1")
        assert event.event_type == WorkflowEventType.WORKFLOW_STARTED
        assert event.workflow_id == "wf-1"
        assert event.event_id

    def test_log_workflow_completed(self):
        logger = WorkflowEventLogger()
        event = logger.log_workflow_completed("wf-1")
        assert event.event_type == WorkflowEventType.WORKFLOW_COMPLETED

    def test_log_workflow_failed(self):
        logger = WorkflowEventLogger()
        event = logger.log_workflow_failed("wf-1", error="timeout")
        assert event.event_type == WorkflowEventType.WORKFLOW_FAILED
        assert event.details["error"] == "timeout"

    def test_log_step_started(self):
        logger = WorkflowEventLogger()
        event = logger.log_step_started("wf-1", "s1", inputs={"data": "x"})
        assert event.event_type == WorkflowEventType.STEP_STARTED
        assert event.step_id == "s1"
        assert event.details["inputs"]["data"] == "x"

    def test_log_step_completed(self):
        logger = WorkflowEventLogger()
        event = logger.log_step_completed(
            "wf-1", "s1", outputs={"result": 42}, duration_seconds=1.5
        )
        assert event.event_type == WorkflowEventType.STEP_COMPLETED
        assert event.details["outputs"]["result"] == 42
        assert event.details["duration_seconds"] == 1.5

    def test_log_step_failed(self):
        logger = WorkflowEventLogger()
        event = logger.log_step_failed("wf-1", "s1", error="OOM", duration_seconds=0.5)
        assert event.event_type == WorkflowEventType.STEP_FAILED
        assert event.details["error"] == "OOM"

    def test_log_step_retrying(self):
        logger = WorkflowEventLogger()
        event = logger.log_step_retrying("wf-1", "s1", attempt=2, error="transient")
        assert event.event_type == WorkflowEventType.STEP_RETRYING
        assert event.details["attempt"] == 2


class TestEventQuerying:
    def test_get_all_events(self):
        logger = WorkflowEventLogger()
        logger.log_workflow_started("wf-1")
        logger.log_step_started("wf-1", "s1")
        assert len(logger.get_events()) == 2

    def test_filter_by_workflow_id(self):
        logger = WorkflowEventLogger()
        logger.log_workflow_started("wf-1")
        logger.log_workflow_started("wf-2")
        events = logger.get_events(workflow_id="wf-1")
        assert len(events) == 1

    def test_filter_by_step_id(self):
        logger = WorkflowEventLogger()
        logger.log_step_started("wf-1", "s1")
        logger.log_step_started("wf-1", "s2")
        events = logger.get_events(step_id="s1")
        assert len(events) == 1

    def test_filter_by_event_type(self):
        logger = WorkflowEventLogger()
        logger.log_step_started("wf-1", "s1")
        logger.log_step_completed("wf-1", "s1")
        events = logger.get_events(event_type=WorkflowEventType.STEP_COMPLETED)
        assert len(events) == 1

    def test_get_workflow_timeline(self):
        logger = WorkflowEventLogger()
        logger.log_workflow_started("wf-1")
        logger.log_step_started("wf-1", "s1")
        logger.log_step_completed("wf-1", "s1")
        timeline = logger.get_workflow_timeline("wf-1")
        assert len(timeline) == 3
        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp

    def test_events_property_returns_copy(self):
        logger = WorkflowEventLogger()
        logger.log_workflow_started("wf-1")
        events = logger.events
        events.clear()
        assert len(logger.events) == 1

    def test_clear(self):
        logger = WorkflowEventLogger()
        logger.log_workflow_started("wf-1")
        logger.clear()
        assert len(logger.events) == 0

    def test_unique_event_ids(self):
        logger = WorkflowEventLogger()
        e1 = logger.log_workflow_started("wf-1")
        e2 = logger.log_workflow_started("wf-1")
        assert e1.event_id != e2.event_id
