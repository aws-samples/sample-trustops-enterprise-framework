"""
Progress Tracker for the Evaluation Engine.

Tracks evaluation progress, calculates ETA, and persists checkpoints
to DynamoDB. Supports status queries returning progress percentage,
completed count, failed count, and estimated time remaining.

Requirements: 3.13, 3.14, 3.20, 3.21
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class EvaluationStatusValue(str, Enum):
    """Status values for an evaluation run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class ProgressStatus:
    """Snapshot of evaluation progress.

    Attributes:
        evaluation_id: Unique identifier for the evaluation.
        total: Total number of examples to process.
        completed: Number of examples completed (success + failed).
        failed: Number of examples that failed.
        progress_percent: Completion percentage (0-100).
        eta_seconds: Estimated seconds remaining, or None if unknown.
        status: Current evaluation status.
    """

    evaluation_id: str
    total: int
    completed: int
    failed: int
    progress_percent: float
    eta_seconds: Optional[float]
    status: EvaluationStatusValue


class DynamoDBClient(Protocol):
    """Protocol for DynamoDB persistence operations."""

    def put_item(self, **kwargs: Any) -> Any:
        ...

    def get_item(self, **kwargs: Any) -> Any:
        ...


class ProgressTracker:
    """Tracks evaluation progress with optional DynamoDB persistence.

    Emits progress events and periodically persists checkpoints to
    DynamoDB so users can query status after disconnecting.

    Usage::

        tracker = ProgressTracker(
            evaluation_id="eval-123",
            total=100,
            persist_every_n=10,
            dynamodb_client=boto3_dynamodb_table,
            table_name="evaluation_progress",
        )
        for i, result in enumerate(results):
            tracker.update(completed=1, failed=0 if result.ok else 1)
        status = tracker.get_status()

    Requirements: 3.13, 3.14, 3.20, 3.21
    """

    def __init__(
        self,
        evaluation_id: str,
        total: int,
        persist_every_n: int = 10,
        dynamodb_client: Optional[Any] = None,
        table_name: str = "evaluation_progress",
    ):
        """
        Args:
            evaluation_id: Unique identifier for this evaluation run.
            total: Total number of examples to process.
            persist_every_n: Persist progress to DynamoDB every N
                completed examples. Default 10.
            dynamodb_client: Optional boto3 DynamoDB Table resource.
                When None, persistence is skipped.
            table_name: DynamoDB table name for persistence.
        """
        if total < 0:
            raise ValueError("total must be non-negative")
        if persist_every_n < 1:
            raise ValueError("persist_every_n must be >= 1")

        self.evaluation_id = evaluation_id
        self.total = total
        self.persist_every_n = persist_every_n
        self._dynamodb_client = dynamodb_client
        self._table_name = table_name

        self._completed = 0
        self._failed = 0
        self._start_time = time.monotonic()
        self._status = EvaluationStatusValue.RUNNING
        self._updates_since_persist = 0

    def update(self, completed: int = 1, failed: int = 0) -> ProgressStatus:
        """Record progress after processing example(s).

        Args:
            completed: Number of newly completed examples (success + fail).
            failed: Number of newly failed examples (subset of completed).

        Returns:
            Current ProgressStatus snapshot.
        """
        self._completed += completed
        self._failed += failed
        self._updates_since_persist += completed

        # Cap completed at total
        if self._completed > self.total:
            self._completed = self.total

        # Auto-complete when all done
        if self._completed >= self.total:
            self._status = EvaluationStatusValue.COMPLETED

        # Persist checkpoint if threshold reached
        if self._updates_since_persist >= self.persist_every_n:
            self._persist_checkpoint()
            self._updates_since_persist = 0

        return self.get_status()

    def get_status(self) -> ProgressStatus:
        """Return current progress snapshot.

        Returns:
            ProgressStatus with current counts, percentage, and ETA.
        """
        progress_percent = 0.0
        if self.total > 0:
            progress_percent = min(
                (self._completed / self.total) * 100.0, 100.0
            )

        eta_seconds = self._calculate_eta()

        return ProgressStatus(
            evaluation_id=self.evaluation_id,
            total=self.total,
            completed=self._completed,
            failed=self._failed,
            progress_percent=round(progress_percent, 2),
            eta_seconds=eta_seconds,
            status=self._status,
        )

    def mark_failed(self) -> ProgressStatus:
        """Mark the evaluation as failed.

        Returns:
            Updated ProgressStatus.
        """
        self._status = EvaluationStatusValue.FAILED
        self._persist_checkpoint()
        return self.get_status()

    def mark_completed(self) -> ProgressStatus:
        """Mark the evaluation as completed.

        Returns:
            Updated ProgressStatus.
        """
        self._status = EvaluationStatusValue.COMPLETED
        self._persist_checkpoint()
        return self.get_status()

    def _calculate_eta(self) -> Optional[float]:
        """Estimate remaining time based on elapsed time and rate.

        Returns:
            Estimated seconds remaining, or None if no progress yet.
        """
        if self._completed == 0 or self._completed >= self.total:
            return None

        elapsed = time.monotonic() - self._start_time
        if elapsed <= 0:
            return None

        rate = self._completed / elapsed  # examples per second
        remaining = self.total - self._completed
        return round(remaining / rate, 2)

    def _persist_checkpoint(self) -> None:
        """Persist current progress to DynamoDB.

        Silently logs warnings on failure — progress tracking should
        not block evaluation execution.
        """
        if self._dynamodb_client is None:
            return

        try:
            self._dynamodb_client.put_item(
                TableName=self._table_name,
                Item={
                    "evaluation_id": {"S": self.evaluation_id},
                    "total": {"N": str(self.total)},
                    "completed": {"N": str(self._completed)},
                    "failed": {"N": str(self._failed)},
                    "progress_percent": {
                        "N": str(
                            round(
                                (self._completed / self.total * 100.0)
                                if self.total > 0
                                else 0.0,
                                2,
                            )
                        )
                    },
                    "status": {"S": self._status.value},
                },
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist progress for %s: %s",
                self.evaluation_id,
                exc,
            )
