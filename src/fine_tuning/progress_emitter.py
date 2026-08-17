"""
Progress event emitter for fine-tuning jobs.

Emits progress events for UI consumption including loss metrics,
current epoch, and estimated time of arrival (ETA).

Requirements: 4.7
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


@dataclass
class TrainingProgressEvent:
    """A training progress event for UI consumption.

    Attributes:
        job_id: The fine-tuning job ID.
        status: Current job status string.
        epoch: Current training epoch.
        total_epochs: Total number of epochs.
        step: Current training step.
        training_loss: Current training loss.
        validation_loss: Current validation loss (if available).
        learning_rate: Current learning rate.
        progress_percent: Overall progress percentage (0-100).
        eta_seconds: Estimated seconds remaining.
        elapsed_seconds: Seconds elapsed since training started.
        timestamp: When this event was emitted.
        message: Human-readable progress message.
    """

    job_id: str
    status: str
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    training_loss: float = 0.0
    validation_loss: Optional[float] = None
    learning_rate: float = 0.0
    progress_percent: float = 0.0
    eta_seconds: Optional[float] = None
    elapsed_seconds: float = 0.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    message: str = ""


# Type alias for progress callbacks
ProgressCallback = Callable[[TrainingProgressEvent], None]


class ProgressEmitter:
    """Emits training progress events for UI consumption.

    Tracks training progress and emits events to registered callbacks.
    Calculates ETA based on elapsed time and progress.

    Usage::

        emitter = ProgressEmitter("job-123", total_epochs=3)
        emitter.on_progress(my_callback)
        emitter.start()
        emitter.emit_progress(epoch=1, step=100, loss=0.5)
        emitter.complete()
    """

    def __init__(
        self,
        job_id: str,
        total_epochs: int = 1,
    ) -> None:
        self._job_id = job_id
        self._total_epochs = total_epochs
        self._callbacks: list[ProgressCallback] = []
        self._events: list[TrainingProgressEvent] = []
        self._started_at: Optional[datetime] = None
        self._status = "pending"

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def total_epochs(self) -> int:
        return self._total_epochs

    @property
    def events(self) -> list[TrainingProgressEvent]:
        """Return a copy of all emitted events."""
        return list(self._events)

    @property
    def status(self) -> str:
        return self._status

    def on_progress(self, callback: ProgressCallback) -> None:
        """Register a callback for progress events.

        Args:
            callback: Function to call with each progress event.
        """
        self._callbacks.append(callback)

    def start(self) -> TrainingProgressEvent:
        """Emit a training start event.

        Returns:
            The emitted start event.
        """
        self._started_at = datetime.now(timezone.utc)
        self._status = "training"
        return self._emit(
            status="training",
            epoch=0,
            progress_percent=0.0,
            message=f"Training started for job {self._job_id}",
        )

    def emit_progress(
        self,
        epoch: int,
        step: int = 0,
        training_loss: float = 0.0,
        validation_loss: Optional[float] = None,
        learning_rate: float = 0.0,
    ) -> TrainingProgressEvent:
        """Emit a training progress event.

        Calculates progress percentage and ETA based on current epoch
        and elapsed time.

        Args:
            epoch: Current training epoch (1-indexed).
            step: Current training step.
            training_loss: Current training loss.
            validation_loss: Current validation loss.
            learning_rate: Current learning rate.

        Returns:
            The emitted progress event.
        """
        progress = (epoch / self._total_epochs * 100) if self._total_epochs > 0 else 0
        progress = min(progress, 100.0)

        elapsed = self._elapsed_seconds()
        eta = self._estimate_eta(progress, elapsed)

        loss_str = f"loss={training_loss:.4f}"
        if validation_loss is not None:
            loss_str += f", val_loss={validation_loss:.4f}"

        message = (
            f"Epoch {epoch}/{self._total_epochs} "
            f"(step {step}) — {loss_str}"
        )

        return self._emit(
            status="training",
            epoch=epoch,
            step=step,
            training_loss=training_loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            progress_percent=progress,
            eta_seconds=eta,
            message=message,
        )

    def complete(
        self,
        final_loss: float = 0.0,
        final_val_loss: Optional[float] = None,
    ) -> TrainingProgressEvent:
        """Emit a training completion event.

        Args:
            final_loss: Final training loss.
            final_val_loss: Final validation loss.

        Returns:
            The emitted completion event.
        """
        self._status = "completed"
        elapsed = self._elapsed_seconds()
        return self._emit(
            status="completed",
            epoch=self._total_epochs,
            training_loss=final_loss,
            validation_loss=final_val_loss,
            progress_percent=100.0,
            eta_seconds=0.0,
            message=(
                f"Training completed for job {self._job_id} "
                f"in {elapsed:.0f}s"
            ),
        )

    def fail(self, error: str = "") -> TrainingProgressEvent:
        """Emit a training failure event.

        Args:
            error: Error description.

        Returns:
            The emitted failure event.
        """
        self._status = "failed"
        return self._emit(
            status="failed",
            message=f"Training failed for job {self._job_id}: {error}",
        )

    def stop(self, reason: str = "") -> TrainingProgressEvent:
        """Emit a training stop event.

        Args:
            reason: Reason for stopping.

        Returns:
            The emitted stop event.
        """
        self._status = "stopped"
        return self._emit(
            status="stopped",
            message=(
                f"Training stopped for job {self._job_id}"
                + (f": {reason}" if reason else "")
            ),
        )

    def _emit(self, **kwargs: Any) -> TrainingProgressEvent:
        """Create and emit a progress event."""
        elapsed = self._elapsed_seconds()
        event = TrainingProgressEvent(
            job_id=self._job_id,
            total_epochs=self._total_epochs,
            elapsed_seconds=elapsed,
            **kwargs,
        )
        self._events.append(event)
        for callback in self._callbacks:
            callback(event)
        return event

    def _elapsed_seconds(self) -> float:
        """Calculate elapsed seconds since training started."""
        if self._started_at is None:
            return 0.0
        delta = datetime.now(timezone.utc) - self._started_at
        return delta.total_seconds()

    def _estimate_eta(
        self,
        progress_percent: float,
        elapsed_seconds: float,
    ) -> Optional[float]:
        """Estimate remaining time based on progress and elapsed time."""
        if progress_percent <= 0 or elapsed_seconds <= 0:
            return None
        total_estimated = elapsed_seconds / (progress_percent / 100)
        remaining = total_estimated - elapsed_seconds
        return max(0.0, remaining)
