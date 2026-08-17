"""Tests for version manager module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.data_models.storage import ResultMetadata
from src.storage.version_manager import VersionManager


def _make_metadata(result_id="res-1", version=1, **kwargs):
    defaults = {
        "result_id": result_id,
        "result_type": "baseline_evaluation",
        "model_id": "model-1",
        "s3_uri": f"s3://bucket/key/v{version}",
        "checksum": f"checksum-v{version}",
        "size_bytes": 100,
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "version": version,
    }
    defaults.update(kwargs)
    return ResultMetadata(**defaults)


@pytest.fixture
def vm():
    return VersionManager(MagicMock())


class TestGetNextVersion:
    def test_first_version(self, vm):
        assert vm.get_next_version("res-new") == 1

    def test_increments(self, vm):
        vm.record_version(_make_metadata("res-1", version=1))
        assert vm.get_next_version("res-1") == 2

    def test_multiple_versions(self, vm):
        vm.record_version(_make_metadata("res-1", version=1))
        vm.record_version(_make_metadata("res-1", version=2))
        vm.record_version(_make_metadata("res-1", version=3))
        assert vm.get_next_version("res-1") == 4


class TestRecordVersion:
    def test_record_and_retrieve(self, vm):
        meta = _make_metadata("res-1", version=1)
        vm.record_version(meta)
        history = vm.get_version_history("res-1")
        assert len(history) == 1
        assert history[0].version == 1


class TestGetVersionHistory:
    def test_empty_history(self, vm):
        assert vm.get_version_history("nonexistent") == []

    def test_sorted_by_version(self, vm):
        vm.record_version(_make_metadata("res-1", version=3))
        vm.record_version(_make_metadata("res-1", version=1))
        vm.record_version(_make_metadata("res-1", version=2))
        history = vm.get_version_history("res-1")
        assert [h.version for h in history] == [1, 2, 3]


class TestGetSpecificVersion:
    def test_existing_version(self, vm):
        vm.record_version(_make_metadata("res-1", version=1))
        vm.record_version(_make_metadata("res-1", version=2))
        result = vm.get_specific_version("res-1", 1)
        assert result is not None
        assert result.version == 1

    def test_nonexistent_version(self, vm):
        vm.record_version(_make_metadata("res-1", version=1))
        assert vm.get_specific_version("res-1", 99) is None

    def test_nonexistent_result(self, vm):
        assert vm.get_specific_version("nonexistent", 1) is None


class TestGetLatestVersion:
    def test_latest(self, vm):
        vm.record_version(_make_metadata("res-1", version=1))
        vm.record_version(_make_metadata("res-1", version=3))
        vm.record_version(_make_metadata("res-1", version=2))
        latest = vm.get_latest_version("res-1")
        assert latest is not None
        assert latest.version == 3

    def test_no_versions(self, vm):
        assert vm.get_latest_version("nonexistent") is None
