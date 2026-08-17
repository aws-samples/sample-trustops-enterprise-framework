"""Tests for workflow history query module."""

from datetime import datetime, timezone, timedelta

import pytest

from src.data_models.workflow import (
    WorkflowDefinition,
    WorkflowManifest,
    WorkflowStatus,
    WorkflowStep,
    StepStatus,
    StepType,
)
from src.orchestration.workflow_history import (
    WorkflowHistoryFilter,
    WorkflowHistoryQuery,
    WorkflowRunSummary,
    _matches_filter,
    _summarize_manifest,
)


def _make_manifest(
    workflow_id="wf-1",
    name="test-wf",
    status=WorkflowStatus.COMPLETED,
    created_by="user-1",
    created_at=None,
    started_at=None,
    completed_at=None,
    steps=None,
    total_cost=0.0,
):
    now = datetime.now(timezone.utc)
    created_at = created_at or now
    step = WorkflowStep(
        step_id="s1",
        step_type=StepType.BASELINE_EVALUATION,
        name="eval",
        status=StepStatus.COMPLETED,
    )
    definition = WorkflowDefinition(
        name=name,
        steps=steps or [step],
    )
    return WorkflowManifest(
        workflow_id=workflow_id,
        definition=definition,
        status=status,
        created_at=created_at,
        started_at=started_at,
        completed_at=completed_at,
        steps=steps or [step],
        total_cost=total_cost,
        created_by=created_by,
    )


class TestMatchesFilter:
    def test_no_filter_matches_all(self):
        m = _make_manifest()
        assert _matches_filter(m, WorkflowHistoryFilter()) is True

    def test_filter_by_status_match(self):
        m = _make_manifest(status=WorkflowStatus.COMPLETED)
        f = WorkflowHistoryFilter(status=WorkflowStatus.COMPLETED)
        assert _matches_filter(m, f) is True

    def test_filter_by_status_no_match(self):
        m = _make_manifest(status=WorkflowStatus.FAILED)
        f = WorkflowHistoryFilter(status=WorkflowStatus.COMPLETED)
        assert _matches_filter(m, f) is False

    def test_filter_by_user(self):
        m = _make_manifest(created_by="alice")
        f = WorkflowHistoryFilter(created_by="alice")
        assert _matches_filter(m, f) is True

    def test_filter_by_user_no_match(self):
        m = _make_manifest(created_by="bob")
        f = WorkflowHistoryFilter(created_by="alice")
        assert _matches_filter(m, f) is False

    def test_filter_by_date_range(self):
        now = datetime.now(timezone.utc)
        m = _make_manifest(created_at=now)
        f = WorkflowHistoryFilter(
            date_from=now - timedelta(hours=1),
            date_to=now + timedelta(hours=1),
        )
        assert _matches_filter(m, f) is True

    def test_filter_by_date_before_range(self):
        now = datetime.now(timezone.utc)
        m = _make_manifest(created_at=now - timedelta(days=2))
        f = WorkflowHistoryFilter(date_from=now - timedelta(days=1))
        assert _matches_filter(m, f) is False

    def test_filter_by_date_after_range(self):
        now = datetime.now(timezone.utc)
        m = _make_manifest(created_at=now + timedelta(days=2))
        f = WorkflowHistoryFilter(date_to=now + timedelta(days=1))
        assert _matches_filter(m, f) is False


class TestSummarizeManifest:
    def test_basic_summary(self):
        now = datetime.now(timezone.utc)
        started = now - timedelta(minutes=5)
        m = _make_manifest(
            started_at=started,
            completed_at=now,
            total_cost=1.5,
        )
        s = _summarize_manifest(m)
        assert s.workflow_id == "wf-1"
        assert s.status == WorkflowStatus.COMPLETED
        assert s.duration_seconds == pytest.approx(300.0, abs=1)
        assert s.total_cost == 1.5
        assert s.step_count == 1
        assert s.completed_steps == 1
        assert s.failed_steps == 0

    def test_no_duration_when_not_started(self):
        m = _make_manifest()
        s = _summarize_manifest(m)
        assert s.duration_seconds is None


class TestWorkflowHistoryQuery:
    def test_empty_query(self):
        q = WorkflowHistoryQuery()
        assert q.query() == []

    def test_add_and_query(self):
        q = WorkflowHistoryQuery()
        m = _make_manifest()
        q.add_manifest(m)
        results = q.query()
        assert len(results) == 1
        assert results[0].workflow_id == "wf-1"

    def test_query_with_status_filter(self):
        q = WorkflowHistoryQuery()
        q.add_manifest(_make_manifest("wf-1", status=WorkflowStatus.COMPLETED))
        q.add_manifest(_make_manifest("wf-2", status=WorkflowStatus.FAILED))
        results = q.query(WorkflowHistoryFilter(status=WorkflowStatus.COMPLETED))
        assert len(results) == 1
        assert results[0].workflow_id == "wf-1"

    def test_query_with_user_filter(self):
        q = WorkflowHistoryQuery()
        q.add_manifest(_make_manifest("wf-1", created_by="alice"))
        q.add_manifest(_make_manifest("wf-2", created_by="bob"))
        results = q.query(WorkflowHistoryFilter(created_by="bob"))
        assert len(results) == 1
        assert results[0].workflow_id == "wf-2"

    def test_query_sorted_by_created_at_desc(self):
        q = WorkflowHistoryQuery()
        now = datetime.now(timezone.utc)
        q.add_manifest(_make_manifest("wf-old", created_at=now - timedelta(hours=2)))
        q.add_manifest(_make_manifest("wf-new", created_at=now))
        results = q.query()
        assert results[0].workflow_id == "wf-new"
        assert results[1].workflow_id == "wf-old"

    def test_query_with_limit(self):
        q = WorkflowHistoryQuery()
        now = datetime.now(timezone.utc)
        for i in range(5):
            q.add_manifest(_make_manifest(
                f"wf-{i}",
                created_at=now + timedelta(seconds=i),
            ))
        results = q.query(limit=2)
        assert len(results) == 2

    def test_count(self):
        q = WorkflowHistoryQuery()
        q.add_manifest(_make_manifest("wf-1", status=WorkflowStatus.COMPLETED))
        q.add_manifest(_make_manifest("wf-2", status=WorkflowStatus.FAILED))
        assert q.count() == 2
        assert q.count(WorkflowHistoryFilter(status=WorkflowStatus.COMPLETED)) == 1

    def test_get_manifest(self):
        q = WorkflowHistoryQuery()
        m = _make_manifest("wf-1")
        q.add_manifest(m)
        assert q.get_manifest("wf-1") is m
        assert q.get_manifest("wf-999") is None
