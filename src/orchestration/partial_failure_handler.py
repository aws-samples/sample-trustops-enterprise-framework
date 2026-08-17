"""
Partial failure handler for workflow steps.

Tracks per-item success/failure within steps, generates partial failure
reports, and enables retry-failed-only mode.

Requirements: 8.14, 8.19
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class ItemStatus(str, Enum):
    """Status of an individual item within a step."""

    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class ItemResult:
    """Result of processing a single item.

    Attributes:
        item_id: Unique identifier for the item.
        status: Processing status.
        output: Output data if successful.
        error: Error message if failed.
        timestamp: When the item was processed.
    """

    item_id: str
    status: ItemStatus
    output: Any = None
    error: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class PartialFailureReport:
    """Report of partial failures within a step.

    Attributes:
        step_id: The step identifier.
        total_items: Total number of items processed.
        successful_items: Number of successful items.
        failed_items: Number of failed items.
        pending_items: Number of pending items.
        success_rate: Ratio of successful to total items.
        failures: Details of each failed item.
        successes: Details of each successful item.
    """

    step_id: str
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    pending_items: int = 0
    success_rate: float = 0.0
    failures: list[ItemResult] = field(default_factory=list)
    successes: list[ItemResult] = field(default_factory=list)


class PartialFailureHandler:
    """Tracks per-item success/failure within workflow steps.

    Supports generating partial failure reports and retrying
    only failed items.

    Usage::

        handler = PartialFailureHandler("step-1")
        handler.record_success("item-1", output={"result": "ok"})
        handler.record_failure("item-2", error="timeout")
        report = handler.get_report()
        failed_ids = handler.get_failed_item_ids()
    """

    def __init__(self, step_id: str) -> None:
        self._step_id = step_id
        self._items: dict[str, ItemResult] = {}

    @property
    def step_id(self) -> str:
        """The step identifier."""
        return self._step_id

    def record_success(
        self, item_id: str, output: Any = None
    ) -> ItemResult:
        """Record a successful item processing.

        Args:
            item_id: The item identifier.
            output: Output data from processing.

        Returns:
            The recorded ItemResult.
        """
        result = ItemResult(
            item_id=item_id,
            status=ItemStatus.SUCCESS,
            output=output,
        )
        self._items[item_id] = result
        return result

    def record_failure(
        self, item_id: str, error: str = ""
    ) -> ItemResult:
        """Record a failed item processing.

        Args:
            item_id: The item identifier.
            error: Error message describing the failure.

        Returns:
            The recorded ItemResult.
        """
        result = ItemResult(
            item_id=item_id,
            status=ItemStatus.FAILED,
            error=error,
        )
        self._items[item_id] = result
        return result

    def mark_pending(self, item_id: str) -> ItemResult:
        """Mark an item as pending for processing.

        Args:
            item_id: The item identifier.

        Returns:
            The recorded ItemResult.
        """
        result = ItemResult(
            item_id=item_id,
            status=ItemStatus.PENDING,
        )
        self._items[item_id] = result
        return result

    def get_report(self) -> PartialFailureReport:
        """Generate a partial failure report.

        Returns:
            PartialFailureReport with summary and details.
        """
        successes = [
            r for r in self._items.values()
            if r.status == ItemStatus.SUCCESS
        ]
        failures = [
            r for r in self._items.values()
            if r.status == ItemStatus.FAILED
        ]
        pending = [
            r for r in self._items.values()
            if r.status == ItemStatus.PENDING
        ]
        total = len(self._items)
        rate = len(successes) / total if total > 0 else 0.0

        return PartialFailureReport(
            step_id=self._step_id,
            total_items=total,
            successful_items=len(successes),
            failed_items=len(failures),
            pending_items=len(pending),
            success_rate=rate,
            failures=failures,
            successes=successes,
        )

    def get_failed_item_ids(self) -> list[str]:
        """Get IDs of all failed items for retry.

        Returns:
            List of item IDs that failed.
        """
        return [
            r.item_id for r in self._items.values()
            if r.status == ItemStatus.FAILED
        ]

    def get_successful_item_ids(self) -> list[str]:
        """Get IDs of all successful items.

        Returns:
            List of item IDs that succeeded.
        """
        return [
            r.item_id for r in self._items.values()
            if r.status == ItemStatus.SUCCESS
        ]

    def has_failures(self) -> bool:
        """Check if any items failed.

        Returns:
            True if at least one item failed.
        """
        return any(
            r.status == ItemStatus.FAILED
            for r in self._items.values()
        )

    def reset_failed(self) -> int:
        """Reset all failed items to pending for retry.

        Returns:
            Number of items reset.
        """
        count = 0
        for item_id, result in self._items.items():
            if result.status == ItemStatus.FAILED:
                self._items[item_id] = ItemResult(
                    item_id=item_id,
                    status=ItemStatus.PENDING,
                )
                count += 1
        return count
