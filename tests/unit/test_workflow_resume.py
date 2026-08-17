"""Tests for workflow resume module."""

import pytest

from src.data_models.workflow import StepStatus
from src.orchestration.workflow_resume import (
    StepCheckpoint,
    WorkflowResumeManager,
)


class TestWorkflowResumeManager:
    def test_save_and_get_checkpoint(self):
        mgr = WorkflowResumeManager()
        cp = StepCheckpoint("s1", StepStatus.COMPLETED, output={"r": 1})
        mgr.save_checkpoint("wf-1", cp)
        result = mgr.get_checkpoint("wf-1", "s1")
        assert result is not None
        assert result.status == StepStatus.COMPLETED
        assert result.output == {"r": 1}

    def test_get_checkpoint_not_found(self):
        mgr = WorkflowResumeManager()
        assert mgr.get_checkpoint("wf-1", "s1") is None

    def test_get_all_checkpoints(self):
        mgr = WorkflowResumeManager()
        mgr.save_checkpoint("wf-1", StepCheckpoint("s1", StepStatus.COMPLETED))
        mgr.save_checkpoint("wf-1", StepCheckpoint("s2", StepStatus.FAILED))
        cps = mgr.get_all_checkpoints("wf-1")
        assert len(cps) == 2

    def test_get_all_checkpoints_empty(self):
        mgr = WorkflowResumeManager()
        assert mgr.get_all_checkpoints("wf-1") == []

    def test_resume_point_no_checkpoints(self):
        mgr = WorkflowResumeManager()
        rp = mgr.get_resume_point("wf-1", ["s1", "s2", "s3"])
        assert rp.last_completed_step_id is None
        assert rp.next_step_ids == ["s1", "s2", "s3"]
        assert rp.completed_step_ids == []

    def test_resume_point_partial_completion(self):
        mgr = WorkflowResumeManager()
        mgr.save_checkpoint("wf-1", StepCheckpoint("s1", StepStatus.COMPLETED))
        mgr.save_checkpoint("wf-1", StepCheckpoint("s2", StepStatus.FAILED))
        rp = mgr.get_resume_point("wf-1", ["s1", "s2", "s3"])
        assert rp.last_completed_step_id == "s1"
        assert "s1" not in rp.next_step_ids
        assert "s2" in rp.next_step_ids
        assert "s3" in rp.next_step_ids
        assert rp.failed_step_ids == ["s2"]

    def test_resume_point_with_dependencies(self):
        mgr = WorkflowResumeManager()
        mgr.save_checkpoint("wf-1", StepCheckpoint("s1", StepStatus.COMPLETED))
        deps = {"s1": [], "s2": ["s1"], "s3": ["s2"]}
        rp = mgr.get_resume_point("wf-1", ["s1", "s2", "s3"], deps)
        assert rp.next_step_ids == ["s2"]
        # s3 depends on s2 which isn't completed

    def test_resume_point_all_completed(self):
        mgr = WorkflowResumeManager()
        mgr.save_checkpoint("wf-1", StepCheckpoint("s1", StepStatus.COMPLETED))
        mgr.save_checkpoint("wf-1", StepCheckpoint("s2", StepStatus.COMPLETED))
        rp = mgr.get_resume_point("wf-1", ["s1", "s2"])
        assert rp.last_completed_step_id == "s2"
        assert rp.next_step_ids == []

    def test_clear_checkpoints(self):
        mgr = WorkflowResumeManager()
        mgr.save_checkpoint("wf-1", StepCheckpoint("s1", StepStatus.COMPLETED))
        mgr.clear_checkpoints("wf-1")
        assert mgr.get_all_checkpoints("wf-1") == []

    def test_is_step_completed(self):
        mgr = WorkflowResumeManager()
        mgr.save_checkpoint("wf-1", StepCheckpoint("s1", StepStatus.COMPLETED))
        mgr.save_checkpoint("wf-1", StepCheckpoint("s2", StepStatus.FAILED))
        assert mgr.is_step_completed("wf-1", "s1") is True
        assert mgr.is_step_completed("wf-1", "s2") is False
        assert mgr.is_step_completed("wf-1", "s3") is False

    def test_overwrite_checkpoint(self):
        mgr = WorkflowResumeManager()
        mgr.save_checkpoint("wf-1", StepCheckpoint("s1", StepStatus.FAILED, error="err"))
        mgr.save_checkpoint("wf-1", StepCheckpoint("s1", StepStatus.COMPLETED, output={"ok": True}))
        cp = mgr.get_checkpoint("wf-1", "s1")
        assert cp.status == StepStatus.COMPLETED
