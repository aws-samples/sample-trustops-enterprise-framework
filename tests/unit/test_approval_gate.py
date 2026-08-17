"""
Unit tests for the approval gate.

Requirements: 8.4
"""

import pytest

from src.orchestration.approval_gate import (
    ApprovalGate,
    ApprovalRequest,
    ApprovalStatus,
)


class TestApprovalGate:
    def test_request_approval(self):
        gate = ApprovalGate()
        req = gate.request_approval("wf-1", "step-1")
        assert isinstance(req, ApprovalRequest)
        assert req.status == ApprovalStatus.PENDING
        assert req.workflow_id == "wf-1"
        assert req.step_id == "step-1"

    def test_approve(self):
        gate = ApprovalGate()
        req = gate.request_approval("wf-1", "step-1")
        result = gate.approve(req.request_id, approved_by="admin")
        assert result.status == ApprovalStatus.APPROVED
        assert result.approved_by == "admin"
        assert result.resolved_at is not None

    def test_reject(self):
        gate = ApprovalGate()
        req = gate.request_approval("wf-1", "step-1")
        result = gate.reject(req.request_id, rejected_by="admin", comment="not ready")
        assert result.status == ApprovalStatus.REJECTED
        assert result.comment == "not ready"

    def test_approve_non_pending_raises(self):
        gate = ApprovalGate()
        req = gate.request_approval("wf-1", "step-1")
        gate.approve(req.request_id)
        with pytest.raises(ValueError, match="Cannot approve"):
            gate.approve(req.request_id)

    def test_reject_non_pending_raises(self):
        gate = ApprovalGate()
        req = gate.request_approval("wf-1", "step-1")
        gate.reject(req.request_id)
        with pytest.raises(ValueError, match="Cannot reject"):
            gate.reject(req.request_id)

    def test_get_request(self):
        gate = ApprovalGate()
        req = gate.request_approval("wf-1", "step-1")
        fetched = gate.get_request(req.request_id)
        assert fetched.request_id == req.request_id

    def test_get_request_not_found(self):
        gate = ApprovalGate()
        with pytest.raises(KeyError):
            gate.get_request("nonexistent")

    def test_is_approved(self):
        gate = ApprovalGate()
        req = gate.request_approval("wf-1", "step-1")
        assert gate.is_approved(req.request_id) is False
        gate.approve(req.request_id)
        assert gate.is_approved(req.request_id) is True

    def test_get_pending_requests(self):
        gate = ApprovalGate()
        gate.request_approval("wf-1", "step-1")
        gate.request_approval("wf-1", "step-2")
        req3 = gate.request_approval("wf-2", "step-1")
        gate.approve(req3.request_id)
        pending = gate.get_pending_requests()
        assert len(pending) == 2

    def test_get_pending_requests_by_workflow(self):
        gate = ApprovalGate()
        gate.request_approval("wf-1", "step-1")
        gate.request_approval("wf-2", "step-1")
        pending = gate.get_pending_requests(workflow_id="wf-1")
        assert len(pending) == 1
        assert pending[0].workflow_id == "wf-1"

    def test_unique_request_ids(self):
        gate = ApprovalGate()
        r1 = gate.request_approval("wf-1", "step-1")
        r2 = gate.request_approval("wf-1", "step-2")
        assert r1.request_id != r2.request_id
