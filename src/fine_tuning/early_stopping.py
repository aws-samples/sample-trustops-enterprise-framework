"""
Early stopping detector for fine-tuning jobs.

Monitors validation loss trend during training and triggers a stop
signal when no improvement is observed for a configurable number of
epochs (patience). Notifies the user when early stopping is triggered.

Requirements: 4.8
"""

from dataclasses import dataclass, field

from src.data_models.fine_tuning import TrainingMetrics


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping behaviour.

    Attributes:
        patience: Number of consecutive epochs with no improvement
            before stopping is triggered.
        min_delta: Minimum decrease in validation loss to count as
            an improvement.  A new loss must be at least
            ``best_loss - min_delta`` to reset the patience counter.
    """

    patience: int = 3
    min_delta: float = 0.0


@dataclass
class EarlyStoppingResult:
    """Result returned by the early stopping check.

    Attributes:
        should_stop: Whether training should be stopped.
        reason: Human-readable explanation when stopping is triggered.
        best_validation_loss: The best (lowest) validation loss seen so far.
        epochs_without_improvement: How many consecutive epochs had no
            improvement at the time of the check.
    """

    should_stop: bool
    reason: str = ""
    best_validation_loss: float | None = None
    epochs_without_improvement: int = 0


class EarlyStoppingDetector:
    """Tracks validation loss history and decides when to stop training.

    Usage::

        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=3))
        for poll_result in poll_results:
            result = detector.check(poll_result.metrics)
            if result.should_stop:
                # trigger stop and notify user
                ...
    """

    def __init__(self, config: EarlyStoppingConfig | None = None) -> None:
        self._config = config or EarlyStoppingConfig()
        if self._config.patience < 1:
            raise ValueError("patience must be >= 1")
        if self._config.min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        self._best_loss: float | None = None
        self._epochs_without_improvement: int = 0
        self._history: list[float] = []

    # -- public API --

    @property
    def config(self) -> EarlyStoppingConfig:
        return self._config

    @property
    def best_loss(self) -> float | None:
        return self._best_loss

    @property
    def epochs_without_improvement(self) -> int:
        return self._epochs_without_improvement

    @property
    def history(self) -> list[float]:
        """Return a copy of the recorded validation loss history."""
        return list(self._history)

    def check(self, metrics: list[TrainingMetrics]) -> EarlyStoppingResult:
        """Evaluate new training metrics and decide whether to stop.

        Only metrics that contain a non-``None`` ``validation_loss`` are
        considered.  Metrics are processed in the order given (assumed
        chronological).

        Args:
            metrics: A batch of ``TrainingMetrics`` from the latest poll.

        Returns:
            An ``EarlyStoppingResult`` indicating whether training should
            be stopped.
        """
        for m in metrics:
            if m.validation_loss is None:
                continue
            self._update(m.validation_loss)

        return EarlyStoppingResult(
            should_stop=self._epochs_without_improvement >= self._config.patience,
            reason=self._build_reason(),
            best_validation_loss=self._best_loss,
            epochs_without_improvement=self._epochs_without_improvement,
        )

    def reset(self) -> None:
        """Reset internal state so the detector can be reused."""
        self._best_loss = None
        self._epochs_without_improvement = 0
        self._history = []

    # -- internals --

    def _update(self, validation_loss: float) -> None:
        """Record a single validation loss observation."""
        self._history.append(validation_loss)

        if self._best_loss is None:
            self._best_loss = validation_loss
            self._epochs_without_improvement = 0
            return

        if validation_loss < self._best_loss - self._config.min_delta:
            # Improvement detected
            self._best_loss = validation_loss
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

    def _build_reason(self) -> str:
        if self._epochs_without_improvement >= self._config.patience:
            return (
                f"Early stopping triggered: no improvement in validation loss "
                f"for {self._epochs_without_improvement} consecutive epochs. "
                f"Best validation loss: {self._best_loss:.6f}."
            )
        return ""
