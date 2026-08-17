"""Tests for partial failure handler module."""

import pytest

from src.orchestration.partial_failure_handler import (
    ItemStatus,
    PartialFailureHandler,
)


class TestPartialFailureHandler:
    def test_record_success(self):
        h = PartialFailureHandler("step-1")
        r = h.record_success("item-1", output={"result": "ok"})
        assert r.status == ItemStatus.SUCCESS
        assert r.output == {"result": "ok"}

    def test_record_failure(self):
        h = PartialFailureHandler("step-1")
        r = h.record_failure("item-1", error="timeout")
        assert r.status == ItemStatus.FAILED
        assert r.error == "timeout"

    def test_mark_pending(self):
        h = PartialFailureHandler("step-1")
        r = h.mark_pending("item-1")
        assert r.status == ItemStatus.PENDING

    def test_get_report_empty(self):
        h = PartialFailureHandler("step-1")
        report = h.get_report()
        assert report.total_items == 0
        assert report.success_rate == 0.0

    def test_get_report_mixed(self):
        h = PartialFailureHandler("step-1")
        h.record_success("item-1")
        h.record_success("item-2")
        h.record_failure("item-3", error="err")
        report = h.get_report()
        assert report.total_items == 3
        assert report.successful_items == 2
        assert report.failed_items == 1
        assert report.success_rate == pytest.approx(2 / 3)

    def test_get_report_all_success(self):
        h = PartialFailureHandler("step-1")
        h.record_success("item-1")
        h.record_success("item-2")
        report = h.get_report()
        assert report.success_rate == 1.0
        assert report.failed_items == 0

    def test_get_failed_item_ids(self):
        h = PartialFailureHandler("step-1")
        h.record_success("item-1")
        h.record_failure("item-2")
        h.record_failure("item-3")
        assert set(h.get_failed_item_ids()) == {"item-2", "item-3"}

    def test_get_successful_item_ids(self):
        h = PartialFailureHandler("step-1")
        h.record_success("item-1")
        h.record_failure("item-2")
        assert h.get_successful_item_ids() == ["item-1"]

    def test_has_failures(self):
        h = PartialFailureHandler("step-1")
        h.record_success("item-1")
        assert h.has_failures() is False
        h.record_failure("item-2")
        assert h.has_failures() is True

    def test_reset_failed(self):
        h = PartialFailureHandler("step-1")
        h.record_success("item-1")
        h.record_failure("item-2")
        h.record_failure("item-3")
        count = h.reset_failed()
        assert count == 2
        report = h.get_report()
        assert report.failed_items == 0
        assert report.pending_items == 2
        assert report.successful_items == 1

    def test_report_includes_failure_details(self):
        h = PartialFailureHandler("step-1")
        h.record_failure("item-1", error="connection timeout")
        report = h.get_report()
        assert len(report.failures) == 1
        assert report.failures[0].error == "connection timeout"

    def test_step_id_property(self):
        h = PartialFailureHandler("my-step")
        assert h.step_id == "my-step"
