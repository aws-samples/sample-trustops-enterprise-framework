"""
Approval gate for manual workflow approval workflows.

Supports pausing a workflow until manual approval is received,
with optional comments and timeout.

Requirements: 8.4
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


@dataclass
class ApprovalRequest:
    """A request for manual approval.

    Attributes:
        request_id: Unique identifier for this approval request.
        workflow_id: The workflow this approval belongs to.
        step_id: The step requiring approval.
        status: Current approval status.
        requested_at: When the approval was requested.
        resolved_at: When the approval was resolved.
        approved_by: Who approved or rejected.
        comment: Optional comment from the approver.
    """

    request_id: str
    workflow_id: str
    step_id: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Optional[datetime] = None
    approved_by: str = ""
    comment: str = ""


class ApprovalGate:
    """Manages manual approval workflows.

    Tracks approval requests and allows resolving them
    with approve or reject actions.

    Usage::

        gate = ApprovalGate()
        req = gate.request_approval("wf-1", "step-3")
        gate.approve(req.request_id, approved_by="admin")
    """

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}
        self._counter: int = 0

    def request_approval(
        self,
        workflow_id: str,
        step_id: str,
    ) -> ApprovalRequest:
        """Create a new approval request.

        Args:
            workflow_id: The workflow identifier.
            step_id: The step requiring approval.

        Returns:
            The created ApprovalRequest in PENDING status.
        """
        self._counter += 1
        request_id = f"apr-{self._counter}"
        request = ApprovalRequest(
            request_id=request_id,
            workflow_id=workflow_id,
            step_id=step_id,
        )
        self._requests[request_id] = request
        return request

    def approve(
        self,
        request_id: str,
        approved_by: str = "",
        comment: str = "",
    ) -> ApprovalRequest:
        """Approve a pending request.

        Args:
            request_id: The approval request ID.
            approved_by: Who is approving.
            comment: Optional comment.

        Returns:
            The updated ApprovalRequest.

        Raises:
            KeyError: If request_id not found.
            ValueError: If request is not in PENDING status.
        """
        request = self._get_request(request_id)
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Cannot approve request in status: {request.status}"
            )
        request.status = ApprovalStatus.APPROVED
        request.resolved_at = datetime.now(timezone.utc)
        request.approved_by = approved_by
        request.comment = comment
        return request

    def reject(
        self,
        request_id: str,
        rejected_by: str = "",
        comment: str = "",
    ) -> ApprovalRequest:
        """Reject a pending request.

        Args:
            request_id: The approval request ID.
            rejected_by: Who is rejecting.
            comment: Optional comment.

        Returns:
            The updated ApprovalRequest.

        Raises:
            KeyError: If request_id not found.
            ValueError: If request is not in PENDING status.
        """
        request = self._get_request(request_id)
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Cannot reject request in status: {request.status}"
            )
        request.status = ApprovalStatus.REJECTED
        request.resolved_at = datetime.now(timezone.utc)
        request.approved_by = rejected_by
        request.comment = comment
        return request

    def get_request(self, request_id: str) -> ApprovalRequest:
        """Get an approval request by ID.

        Args:
            request_id: The approval request ID.

        Returns:
            The ApprovalRequest.

        Raises:
            KeyError: If request_id not found.
        """
        return self._get_request(request_id)

    def get_pending_requests(
        self, workflow_id: str | None = None
    ) -> list[ApprovalRequest]:
        """Get all pending approval requests.

        Args:
            workflow_id: Optional filter by workflow ID.

        Returns:
            List of pending ApprovalRequests.
        """
        results = [
            r for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
        ]
        if workflow_id is not None:
            results = [r for r in results if r.workflow_id == workflow_id]
        return results

    def is_approved(self, request_id: str) -> bool:
        """Check if a request has been approved.

        Args:
            request_id: The approval request ID.

        Returns:
            True if approved, False otherwise.
        """
        request = self._get_request(request_id)
        return request.status == ApprovalStatus.APPROVED

    def _get_request(self, request_id: str) -> ApprovalRequest:
        if request_id not in self._requests:
            raise KeyError(f"Approval request not found: {request_id}")
        return self._requests[request_id]
