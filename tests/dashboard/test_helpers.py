"""Tests for dashboard helper utilities."""

import pytest
from datetime import datetime, timezone

from dashboard.utils.helpers import (
    format_timestamp,
    format_cost,
    format_duration,
    status_color,
    status_emoji,
    generate_demo_id,
    safe_get,
)


class TestFormatTimestamp:
    def test_utc_datetime(self):
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = format_timestamp(dt)
        assert "2024-01-15" in result
        assert "10:30:00" in result

    def test_naive_datetime(self):
        dt = datetime(2024, 6, 1, 12, 0, 0)
        result = format_timestamp(dt)
        assert "2024-06-01" in result


class TestFormatCost:
    def test_small_cost(self):
        assert format_cost(0.0025) == "$0.0025"

    def test_large_cost(self):
        assert format_cost(45.80) == "$45.80"

    def test_zero_cost(self):
        assert format_cost(0.0) == "$0.0000"


class TestFormatDuration:
    def test_seconds(self):
        assert format_duration(30.5) == "30.5s"

    def test_minutes(self):
        assert format_duration(150) == "2.5m"

    def test_hours(self):
        assert format_duration(7200) == "2.0h"


class TestStatusColor:
    def test_active(self):
        assert status_color("active") == "green"

    def test_failed(self):
        assert status_color("failed") == "red"

    def test_running(self):
        assert status_color("running") == "blue"

    def test_unknown(self):
        assert status_color("unknown_status") == "gray"


class TestStatusEmoji:
    def test_active(self):
        assert status_emoji("active") == "🟢"

    def test_completed(self):
        assert status_emoji("completed") == "✅"

    def test_failed(self):
        assert status_emoji("failed") == "🔴"

    def test_unknown(self):
        assert status_emoji("xyz") == "⚪"


class TestGenerateDemoId:
    def test_has_prefix(self):
        result = generate_demo_id("test")
        assert result.startswith("test-")

    def test_unique(self):
        ids = {generate_demo_id() for _ in range(100)}
        # Should generate mostly unique IDs (hash collisions extremely unlikely)
        assert len(ids) > 90


class TestSafeGet:
    def test_simple_key(self):
        data = {"a": 1}
        assert safe_get(data, "a") == 1

    def test_nested_keys(self):
        data = {"a": {"b": {"c": 42}}}
        assert safe_get(data, "a", "b", "c") == 42

    def test_missing_key(self):
        data = {"a": 1}
        assert safe_get(data, "b") is None

    def test_missing_key_with_default(self):
        data = {"a": 1}
        assert safe_get(data, "b", default="N/A") == "N/A"

    def test_non_dict_intermediate(self):
        data = {"a": "string"}
        assert safe_get(data, "a", "b") is None
