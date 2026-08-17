"""Tests for the real-time progress display component."""

import time

import pytest

from dashboard.components.progress import estimate_completion_time


class TestEstimateCompletionTime:
    """Tests for ETA calculation."""

    def test_returns_none_for_zero_current(self):
        result = estimate_completion_time(0, 100, time.time() - 10)
        assert result is None

    def test_returns_none_for_zero_total(self):
        result = estimate_completion_time(5, 0, time.time() - 10)
        assert result is None

    def test_returns_positive_value(self):
        start = time.time() - 10  # started 10 seconds ago
        result = estimate_completion_time(50, 100, start)
        assert result is not None
        assert result >= 0

    def test_nearly_complete_has_small_eta(self):
        start = time.time() - 100
        result = estimate_completion_time(99, 100, start)
        assert result is not None
        assert result < 10  # should be very small

    def test_halfway_through(self):
        start = time.time() - 60  # 60 seconds elapsed
        result = estimate_completion_time(50, 100, start)
        assert result is not None
        # Should be approximately 60 seconds remaining
        assert 40 < result < 80

    def test_returns_none_for_future_start(self):
        # start_time in the future means elapsed <= 0
        result = estimate_completion_time(10, 100, time.time() + 100)
        assert result is None
