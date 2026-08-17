"""Tests for the ProgressTracker.

Requirements: 3.13, 3.14, 3.20, 3.21
"""

import time
from unittest.mock import MagicMock

import pytest

from src.evaluation.progress_tracker import (
    EvaluationStatusValue,
    ProgressStatus,
    ProgressTracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(
    total: int = 100,
    persist_every_n: int = 10,
    dynamodb_client: object = None,
) -> ProgressTracker:
    return ProgressTracker(
        evaluation_id="eval-001",
        total=total,
        persist_every_n=persist_every_n,
        dynamodb_client=dynamodb_client,
        table_name="test_table",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestProgressTrackerInit:
    """Tests for ProgressTracker initialization."""

    def test_initial_status_is_running(self):
        tracker = _make_tracker()
        status = tracker.get_status()
        assert status.status == EvaluationStatusValue.RUNNING

    def test_initial_counts_are_zero(self):
        tracker = _make_tracker(total=50)
        status = tracker.get_status()
        assert status.completed == 0
        assert status.failed == 0
        assert status.total == 50
        assert status.progress_percent == 0.0

    def test_initial_eta_is_none(self):
        tracker = _make_tracker()
        assert tracker.get_status().eta_seconds is None

    def test_negative_total_raises(self):
        with pytest.raises(ValueError, match="total must be non-negative"):
            ProgressTracker(evaluation_id="x", total=-1)

    def test_zero_persist_every_n_raises(self):
        with pytest.raises(ValueError, match="persist_every_n must be >= 1"):
            ProgressTracker(
                evaluation_id="x", total=10, persist_every_n=0
            )


# ---------------------------------------------------------------------------
# Update and progress tracking
# ---------------------------------------------------------------------------

class TestProgressTrackerUpdate:
    """Tests for the update method and progress calculation."""

    def test_single_update_increments_completed(self):
        tracker = _make_tracker(total=10)
        status = tracker.update(completed=1, failed=0)
        assert status.completed == 1
        assert status.failed == 0
        assert status.progress_percent == 10.0

    def test_multiple_updates_accumulate(self):
        tracker = _make_tracker(total=20)
        tracker.update(completed=5, failed=1)
        status = tracker.update(completed=5, failed=2)
        assert status.completed == 10
        assert status.failed == 3
        assert status.progress_percent == 50.0

    def test_completed_capped_at_total(self):
        tracker = _make_tracker(total=5)
        status = tracker.update(completed=10, failed=0)
        assert status.completed == 5
        assert status.progress_percent == 100.0

    def test_auto_completes_when_all_done(self):
        tracker = _make_tracker(total=3)
        tracker.update(completed=2)
        status = tracker.update(completed=1)
        assert status.status == EvaluationStatusValue.COMPLETED

    def test_zero_total_gives_zero_percent(self):
        tracker = _make_tracker(total=0)
        status = tracker.get_status()
        assert status.progress_percent == 0.0


class TestProgressTrackerETA:
    """Tests for ETA calculation."""

    def test_eta_none_when_no_progress(self):
        tracker = _make_tracker(total=100)
        assert tracker.get_status().eta_seconds is None

    def test_eta_none_when_completed(self):
        tracker = _make_tracker(total=2)
        tracker.update(completed=2)
        assert tracker.get_status().eta_seconds is None

    def test_eta_is_positive_during_progress(self):
        tracker = _make_tracker(total=100)
        # Simulate some elapsed time by adjusting start
        tracker._start_time = time.monotonic() - 10.0
        tracker.update(completed=50)
        eta = tracker.get_status().eta_seconds
        assert eta is not None
        assert eta > 0

    def test_eta_decreases_as_progress_increases(self):
        tracker = _make_tracker(total=100)
        tracker._start_time = time.monotonic() - 10.0
        tracker._completed = 25
        eta_25 = tracker._calculate_eta()

        tracker._completed = 75
        eta_75 = tracker._calculate_eta()

        assert eta_25 is not None
        assert eta_75 is not None
        assert eta_75 < eta_25


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

class TestProgressTrackerStatusTransitions:
    """Tests for mark_failed and mark_completed."""

    def test_mark_failed(self):
        tracker = _make_tracker()
        status = tracker.mark_failed()
        assert status.status == EvaluationStatusValue.FAILED

    def test_mark_completed(self):
        tracker = _make_tracker()
        status = tracker.mark_completed()
        assert status.status == EvaluationStatusValue.COMPLETED


# ---------------------------------------------------------------------------
# DynamoDB persistence
# ---------------------------------------------------------------------------

class TestProgressTrackerPersistence:
    """Tests for DynamoDB checkpoint persistence."""

    def test_persists_at_threshold(self):
        mock_db = MagicMock()
        tracker = _make_tracker(
            total=100, persist_every_n=5, dynamodb_client=mock_db
        )
        # 4 updates — not yet at threshold
        for _ in range(4):
            tracker.update(completed=1)
        mock_db.put_item.assert_not_called()

        # 5th update triggers persist
        tracker.update(completed=1)
        mock_db.put_item.assert_called_once()

    def test_persists_multiple_times(self):
        mock_db = MagicMock()
        tracker = _make_tracker(
            total=100, persist_every_n=3, dynamodb_client=mock_db
        )
        for _ in range(9):
            tracker.update(completed=1)
        assert mock_db.put_item.call_count == 3

    def test_no_persist_without_client(self):
        # Should not raise even without a DynamoDB client
        tracker = _make_tracker(total=20, persist_every_n=2)
        for _ in range(10):
            tracker.update(completed=1)
        # No exception means success

    def test_persist_failure_does_not_raise(self):
        mock_db = MagicMock()
        mock_db.put_item.side_effect = Exception("DynamoDB error")
        tracker = _make_tracker(
            total=100, persist_every_n=1, dynamodb_client=mock_db
        )
        # Should not raise
        tracker.update(completed=1)

    def test_mark_failed_persists(self):
        mock_db = MagicMock()
        tracker = _make_tracker(dynamodb_client=mock_db)
        tracker.mark_failed()
        mock_db.put_item.assert_called_once()

    def test_mark_completed_persists(self):
        mock_db = MagicMock()
        tracker = _make_tracker(dynamodb_client=mock_db)
        tracker.mark_completed()
        mock_db.put_item.assert_called_once()

    def test_persisted_item_contains_correct_fields(self):
        mock_db = MagicMock()
        tracker = _make_tracker(
            total=50, persist_every_n=1, dynamodb_client=mock_db
        )
        tracker.update(completed=1, failed=1)

        call_kwargs = mock_db.put_item.call_args
        item = call_kwargs.kwargs.get("Item") or call_kwargs[1].get("Item")
        assert item["evaluation_id"]["S"] == "eval-001"
        assert item["total"]["N"] == "50"
        assert item["completed"]["N"] == "1"
        assert item["failed"]["N"] == "1"
        assert item["status"]["S"] == "running"


# ---------------------------------------------------------------------------
# ProgressStatus dataclass
# ---------------------------------------------------------------------------

class TestProgressStatus:
    """Tests for the ProgressStatus dataclass."""

    def test_fields_accessible(self):
        status = ProgressStatus(
            evaluation_id="e1",
            total=100,
            completed=50,
            failed=5,
            progress_percent=50.0,
            eta_seconds=30.0,
            status=EvaluationStatusValue.RUNNING,
        )
        assert status.evaluation_id == "e1"
        assert status.total == 100
        assert status.completed == 50
        assert status.failed == 5
        assert status.progress_percent == 50.0
        assert status.eta_seconds == 30.0
        assert status.status == EvaluationStatusValue.RUNNING
