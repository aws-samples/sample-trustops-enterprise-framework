"""Tests for failure notification module."""

import pytest

from src.orchestration.failure_notification import (
    FailureNotification,
    FailureNotifier,
    RecoveryOption,
)


class TestFailureNotifier:
    def test_notify_failure(self):
        n = FailureNotifier()
        notif = n.notify_failure("wf-1", "s1", "Eval", "timeout")
        assert notif.workflow_id == "wf-1"
        assert notif.step_id == "s1"
        assert notif.error == "timeout"
        assert len(notif.recovery_options) == 3

    def test_notify_with_custom_options(self):
        n = FailureNotifier()
        notif = n.notify_failure(
            "wf-1", "s1", "Eval", "err",
            recovery_options=[RecoveryOption.RETRY, RecoveryOption.CANCEL],
        )
        assert len(notif.recovery_options) == 2

    def test_notify_with_details(self):
        n = FailureNotifier()
        notif = n.notify_failure(
            "wf-1", "s1", "Eval", "err",
            error_details={"code": 500, "model": "m-1"},
        )
        assert notif.error_details["code"] == 500

    def test_respond_valid_option(self):
        n = FailureNotifier()
        notif = n.notify_failure("wf-1", "s1", "Eval", "err")
        n.respond(notif, RecoveryOption.RETRY)
        assert n.get_response("wf-1", "s1") == RecoveryOption.RETRY

    def test_respond_invalid_option(self):
        n = FailureNotifier()
        notif = n.notify_failure(
            "wf-1", "s1", "Eval", "err",
            recovery_options=[RecoveryOption.CANCEL],
        )
        with pytest.raises(ValueError):
            n.respond(notif, RecoveryOption.RETRY)

    def test_get_response_not_found(self):
        n = FailureNotifier()
        assert n.get_response("wf-1", "s1") is None

    def test_get_notifications(self):
        n = FailureNotifier()
        n.notify_failure("wf-1", "s1", "Step 1", "err1")
        n.notify_failure("wf-2", "s1", "Step 1", "err2")
        assert len(n.get_notifications()) == 2
        assert len(n.get_notifications("wf-1")) == 1

    def test_clear_all(self):
        n = FailureNotifier()
        n.notify_failure("wf-1", "s1", "Step 1", "err")
        notif = n.get_notifications()[0]
        n.respond(notif, RecoveryOption.SKIP)
        n.clear()
        assert n.get_notifications() == []
        assert n.get_response("wf-1", "s1") is None

    def test_clear_by_workflow(self):
        n = FailureNotifier()
        n.notify_failure("wf-1", "s1", "Step 1", "err1")
        n.notify_failure("wf-2", "s1", "Step 1", "err2")
        n.clear("wf-1")
        assert len(n.get_notifications()) == 1
        assert n.get_notifications()[0].workflow_id == "wf-2"

    def test_default_recovery_options(self):
        n = FailureNotifier()
        notif = n.notify_failure("wf-1", "s1", "Eval", "err")
        assert RecoveryOption.RETRY in notif.recovery_options
        assert RecoveryOption.SKIP in notif.recovery_options
        assert RecoveryOption.CANCEL in notif.recovery_options
