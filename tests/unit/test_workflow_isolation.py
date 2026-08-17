"""Tests for concurrent workflow isolation module."""

import threading

import pytest

from src.orchestration.workflow_isolation import (
    WorkflowIsolation,
    namespace_key,
    parse_namespace_key,
)


class TestNamespaceKey:
    def test_creates_namespaced_key(self):
        assert namespace_key("wf-1", "model") == "wf-1/model"

    def test_parse_round_trip(self):
        key = namespace_key("wf-1", "data/file.json")
        wf_id, k = parse_namespace_key(key)
        assert wf_id == "wf-1"
        assert k == "data/file.json"

    def test_parse_invalid_key(self):
        with pytest.raises(ValueError):
            parse_namespace_key("no-slash")


class TestWorkflowIsolation:
    def test_create_scope(self):
        iso = WorkflowIsolation()
        state = iso.create_scope("wf-1")
        assert state.workflow_id == "wf-1"

    def test_create_duplicate_scope_raises(self):
        iso = WorkflowIsolation()
        iso.create_scope("wf-1")
        with pytest.raises(ValueError):
            iso.create_scope("wf-1")

    def test_get_scope(self):
        iso = WorkflowIsolation()
        iso.create_scope("wf-1")
        assert iso.get_scope("wf-1") is not None
        assert iso.get_scope("wf-999") is None

    def test_destroy_scope(self):
        iso = WorkflowIsolation()
        iso.create_scope("wf-1")
        assert iso.destroy_scope("wf-1") is True
        assert iso.get_scope("wf-1") is None
        assert iso.destroy_scope("wf-1") is False

    def test_set_and_get_state(self):
        iso = WorkflowIsolation()
        iso.create_scope("wf-1")
        iso.set_state("wf-1", "model_id", "m-1")
        assert iso.get_state("wf-1", "model_id") == "m-1"

    def test_get_state_default(self):
        iso = WorkflowIsolation()
        iso.create_scope("wf-1")
        assert iso.get_state("wf-1", "missing", "default") == "default"

    def test_state_isolation_between_workflows(self):
        iso = WorkflowIsolation()
        iso.create_scope("wf-1")
        iso.create_scope("wf-2")
        iso.set_state("wf-1", "key", "value-1")
        iso.set_state("wf-2", "key", "value-2")
        assert iso.get_state("wf-1", "key") == "value-1"
        assert iso.get_state("wf-2", "key") == "value-2"

    def test_set_state_missing_scope_raises(self):
        iso = WorkflowIsolation()
        with pytest.raises(KeyError):
            iso.set_state("wf-missing", "key", "value")

    def test_get_state_missing_scope_raises(self):
        iso = WorkflowIsolation()
        with pytest.raises(KeyError):
            iso.get_state("wf-missing", "key")

    def test_list_scopes(self):
        iso = WorkflowIsolation()
        iso.create_scope("wf-1")
        iso.create_scope("wf-2")
        scopes = iso.list_scopes()
        assert set(scopes) == {"wf-1", "wf-2"}

    def test_concurrent_access(self):
        """Test that concurrent workflows don't interfere."""
        iso = WorkflowIsolation()
        iso.create_scope("wf-a")
        iso.create_scope("wf-b")
        errors = []

        def writer(wf_id, count):
            try:
                for i in range(count):
                    iso.set_state(wf_id, f"key-{i}", f"{wf_id}-{i}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, args=("wf-a", 50))
        t2 = threading.Thread(target=writer, args=("wf-b", 50))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
        for i in range(50):
            assert iso.get_state("wf-a", f"key-{i}") == f"wf-a-{i}"
            assert iso.get_state("wf-b", f"key-{i}") == f"wf-b-{i}"
