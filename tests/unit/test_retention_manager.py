"""Tests for retention manager module."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from src.data_models.storage import RetentionPolicy
from src.storage.retention_manager import RetentionAction, RetentionManager


@pytest.fixture
def mock_s3():
    return MagicMock()


@pytest.fixture
def mock_metadata():
    return MagicMock()


@pytest.fixture
def manager(mock_s3, mock_metadata):
    return RetentionManager(mock_s3, mock_metadata)


def _make_item(result_id="res-1", age_days=0, tags=None):
    now = datetime.now(timezone.utc)
    created = now - timedelta(days=age_days)
    return {
        "result_id": result_id,
        "created_at": created.isoformat(),
        "s3_uri": f"s3://bucket/{result_id}",
        "tags": tags or {},
    }


class TestEvaluateItem:
    def test_young_item_kept(self, manager):
        item = _make_item(age_days=10)
        policy = RetentionPolicy(archive_after_days=90, delete_after_days=365)
        action = manager.evaluate_item(item, policy)
        assert action.action == RetentionAction.KEEP

    def test_old_item_archived(self, manager):
        item = _make_item(age_days=100)
        policy = RetentionPolicy(archive_after_days=90, delete_after_days=365)
        action = manager.evaluate_item(item, policy)
        assert action.action == RetentionAction.ARCHIVE

    def test_very_old_item_deleted(self, manager):
        item = _make_item(age_days=400)
        policy = RetentionPolicy(archive_after_days=90, delete_after_days=365)
        action = manager.evaluate_item(item, policy)
        assert action.action == RetentionAction.DELETE

    def test_excluded_by_tag_key(self, manager):
        item = _make_item(age_days=400, tags={"production": "true"})
        policy = RetentionPolicy(
            archive_after_days=90,
            delete_after_days=365,
            exclude_tags=["production"],
        )
        action = manager.evaluate_item(item, policy)
        assert action.action == RetentionAction.KEEP

    def test_excluded_by_tag_value(self, manager):
        item = _make_item(age_days=400, tags={"env": "audit"})
        policy = RetentionPolicy(
            archive_after_days=90,
            delete_after_days=365,
            exclude_tags=["audit"],
        )
        action = manager.evaluate_item(item, policy)
        assert action.action == RetentionAction.KEEP

    def test_unparseable_date_kept(self, manager):
        item = {"result_id": "r1", "created_at": "not-a-date", "tags": {}}
        policy = RetentionPolicy(archive_after_days=90, delete_after_days=365)
        action = manager.evaluate_item(item, policy)
        assert action.action == RetentionAction.KEEP


class TestApplyPolicy:
    def test_apply_archives_and_deletes(self, manager, mock_s3, mock_metadata):
        items = [
            _make_item("r1", age_days=100),  # archive
            _make_item("r2", age_days=400),  # delete
            _make_item("r3", age_days=10),   # keep
        ]
        policy = RetentionPolicy(archive_after_days=90, delete_after_days=365)
        counts = manager.apply_policy(items, policy)
        assert counts["archived"] == 1
        assert counts["deleted"] == 1
        assert counts["kept"] == 1

    def test_apply_handles_errors(self, manager, mock_s3):
        mock_s3.change_storage_class.side_effect = Exception("S3 error")
        items = [_make_item("r1", age_days=100)]
        policy = RetentionPolicy(archive_after_days=90, delete_after_days=365)
        counts = manager.apply_policy(items, policy)
        assert counts["kept"] == 1
        assert counts["archived"] == 0
