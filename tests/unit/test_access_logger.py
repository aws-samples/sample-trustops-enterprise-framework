"""Tests for access logger module."""

from unittest.mock import MagicMock

import pytest

from src.storage.access_logger import AccessLogEntry, AccessLogger


class TestAccessLogger:
    def test_log_creates_entry(self):
        logger = AccessLogger()
        entry = logger.log("res-1", "read", "user-1")
        assert entry.result_id == "res-1"
        assert entry.action == "read"
        assert entry.user_id == "user-1"
        assert entry.log_id  # non-empty

    def test_log_with_details(self):
        logger = AccessLogger()
        entry = logger.log("res-1", "export", "user-1", {"format": "csv"})
        assert entry.details == {"format": "csv"}

    def test_log_persists_to_store(self):
        mock_store = MagicMock()
        logger = AccessLogger(mock_store)
        logger.log("res-1", "read", "user-1")
        mock_store.log_access.assert_called_once_with("res-1", "read", "user-1")

    def test_get_logs_all(self):
        logger = AccessLogger()
        logger.log("res-1", "read", "user-1")
        logger.log("res-2", "write", "user-2")
        logs = logger.get_logs()
        assert len(logs) == 2

    def test_get_logs_by_result_id(self):
        logger = AccessLogger()
        logger.log("res-1", "read", "user-1")
        logger.log("res-2", "write", "user-2")
        logs = logger.get_logs(result_id="res-1")
        assert len(logs) == 1
        assert logs[0].result_id == "res-1"

    def test_get_logs_by_user_id(self):
        logger = AccessLogger()
        logger.log("res-1", "read", "user-1")
        logger.log("res-2", "write", "user-2")
        logs = logger.get_logs(user_id="user-2")
        assert len(logs) == 1

    def test_get_logs_by_action(self):
        logger = AccessLogger()
        logger.log("res-1", "read", "user-1")
        logger.log("res-1", "write", "user-1")
        logs = logger.get_logs(action="read")
        assert len(logs) == 1

    def test_get_logs_with_limit(self):
        logger = AccessLogger()
        for i in range(10):
            logger.log(f"res-{i}", "read", "user-1")
        logs = logger.get_logs(limit=3)
        assert len(logs) == 3
