"""
Failure notification for workflow step failures.

Logs errors with details, updates workflow status to failed, and
notifies users with recovery options (retry, skip, cancel).

Requirements: 8.5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class RecoveryOption(str, Enum):
    """Available recovery options after a failure."""

    RETRY = "retry"
    SKIP = "skip"
    CANCEL = "cancel"


@dataclass
class FailureNotification:
    """Notification about a workflow step failure.

    Attributes:
        workflow_id: The workflow identifier.
        step_id: The failed step identifier.
        step_name: Human-readable step name.
        error: Error message.
        error_details: Additional error details.
        recovery_options: Available recovery actions.
        timestamp: When the failure occurred.
    """

    workflow_id: str
    step_id: str
    step_name: str
    error: str
    error_details: dict[str, Any] = field(default_factory=dict)
    recovery_options: list[RecoveryOption] = field(
        default_factory=lambda: [
            RecoveryOption.RETRY,
            RecoveryOption.SKIP,
            RecoveryOption.CANCEL,
        ]
    )
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class FailureNotifier:
    """Manages failure notifications for workflow steps.

    Logs errors, creates notifications, and tracks user responses.

    Usage::

        notifier = FailureNotifier()
        notification = notifier.notify_failure(
            "wf-1", "step-1", "Eval Step", "Model timeout"
        )
        notifier.respond(notification, RecoveryOption.RETRY)
    """

    def __init__(self) -> None:
        self._notifications: list[FailureNotification] = []
        self._responses: dict[str, RecoveryOption] = {}

    def notify_failure(
        self,
        workflow_id: str,
        step_id: str,
        step_name: str,
        error: str,
        error_details: Optional[dict[str, Any]] = None,
        recovery_options: Optional[list[RecoveryOption]] = None,
    ) -> FailureNotification:
        """Create and record a failure notification.

        Args:
            workflow_id: The workflow identifier.
            step_id: The failed step identifier.
            step_name: Human-readable step name.
            error: Error message.
            error_details: Additional error context.
            recovery_options: Available recovery actions.

        Returns:
            The created FailureNotification.
        """
        notification = FailureNotification(
            workflow_id=workflow_id,
            step_id=step_id,
            step_name=step_name,
            error=error,
            error_details=error_details or {},
            recovery_options=recovery_options or [
                RecoveryOption.RETRY,
                RecoveryOption.SKIP,
                RecoveryOption.CANCEL,
            ],
        )
        self._notifications.append(notification)
        return notification

    def respond(
        self,
        notification: FailureNotification,
        option: RecoveryOption,
    ) -> None:
        """Record a user's recovery choice for a notification.

        Args:
            notification: The failure notification.
            option: The chosen recovery option.

        Raises:
            ValueError: If the option is not available.
        """
        if option not in notification.recovery_options:
            raise ValueError(
                f"Option {option.value} not available. "
                f"Available: {[o.value for o in notification.recovery_options]}"
            )
        key = f"{notification.workflow_id}:{notification.step_id}"
        self._responses[key] = option

    def get_response(
        self, workflow_id: str, step_id: str
    ) -> Optional[RecoveryOption]:
        """Get the user's recovery choice for a step failure.

        Args:
            workflow_id: The workflow identifier.
            step_id: The step identifier.

        Returns:
            The chosen RecoveryOption or None if not yet responded.
        """
        key = f"{workflow_id}:{step_id}"
        return self._responses.get(key)

    def get_notifications(
        self, workflow_id: Optional[str] = None
    ) -> list[FailureNotification]:
        """Get failure notifications, optionally filtered by workflow.

        Args:
            workflow_id: Optional workflow ID filter.

        Returns:
            List of FailureNotifications.
        """
        if workflow_id is None:
            return list(self._notifications)
        return [
            n for n in self._notifications
            if n.workflow_id == workflow_id
        ]

    def clear(self, workflow_id: Optional[str] = None) -> None:
        """Clear notifications and responses.

        Args:
            workflow_id: If provided, only clear for this workflow.
        """
        if workflow_id is None:
            self._notifications = []
            self._responses = {}
        else:
            self._notifications = [
                n for n in self._notifications
                if n.workflow_id != workflow_id
            ]
            prefix = f"{workflow_id}:"
            self._responses = {
                k: v for k, v in self._responses.items()
                if not k.startswith(prefix)
            }
