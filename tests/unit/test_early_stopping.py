"""
Unit tests for the early stopping detector.

Tests cover validation loss monitoring, configurable patience,
min_delta threshold, edge cases, and user notification messages.

Requirements: 4.8
"""

from datetime import datetime, timezone

import pytest

from src.data_models.fine_tuning import TrainingMetrics
from src.fine_tuning.early_stopping import (
    EarlyStoppingConfig,
    EarlyStoppingDetector,
    EarlyStoppingResult,
)


# --- Helpers ---


def _metric(
    epoch: int,
    training_loss: float,
    validation_loss: float | None = None,
) -> TrainingMetrics:
    """Create a TrainingMetrics instance for testing."""
    return TrainingMetrics(
        epoch=epoch,
        step=epoch * 100,
        training_loss=training_loss,
        validation_loss=validation_loss,
        learning_rate=1e-5,
        timestamp=datetime.now(timezone.utc),
    )


# --- EarlyStoppingConfig ---


class TestEarlyStoppingConfig:
    def test_defaults(self):
        cfg = EarlyStoppingConfig()
        assert cfg.patience == 3
        assert cfg.min_delta == 0.0

    def test_custom_values(self):
        cfg = EarlyStoppingConfig(patience=5, min_delta=0.01)
        assert cfg.patience == 5
        assert cfg.min_delta == 0.01


# --- EarlyStoppingResult ---


class TestEarlyStoppingResult:
    def test_defaults(self):
        result = EarlyStoppingResult(should_stop=False)
        assert result.should_stop is False
        assert result.reason == ""
        assert result.best_validation_loss is None
        assert result.epochs_without_improvement == 0


# --- EarlyStoppingDetector init ---


class TestEarlyStoppingDetectorInit:
    def test_default_config(self):
        detector = EarlyStoppingDetector()
        assert detector.config.patience == 3
        assert detector.best_loss is None
        assert detector.epochs_without_improvement == 0

    def test_custom_config(self):
        cfg = EarlyStoppingConfig(patience=5, min_delta=0.001)
        detector = EarlyStoppingDetector(cfg)
        assert detector.config.patience == 5
        assert detector.config.min_delta == 0.001

    def test_invalid_patience_raises(self):
        with pytest.raises(ValueError, match="patience must be >= 1"):
            EarlyStoppingDetector(EarlyStoppingConfig(patience=0))

    def test_negative_min_delta_raises(self):
        with pytest.raises(ValueError, match="min_delta must be >= 0"):
            EarlyStoppingDetector(EarlyStoppingConfig(min_delta=-0.1))


# --- Core early stopping logic ---


class TestEarlyStoppingCheck:
    def test_no_stop_when_loss_improving(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=3))
        # Steadily decreasing validation loss
        for epoch in range(5):
            result = detector.check([_metric(epoch, 0.5, 1.0 - epoch * 0.1)])
        assert result.should_stop is False
        assert detector.epochs_without_improvement == 0

    def test_triggers_stop_after_patience_exceeded(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=2))
        # Epoch 0: first observation (best = 0.5)
        result = detector.check([_metric(0, 0.5, 0.5)])
        assert result.should_stop is False

        # Epoch 1: no improvement (counter = 1)
        result = detector.check([_metric(1, 0.4, 0.6)])
        assert result.should_stop is False
        assert detector.epochs_without_improvement == 1

        # Epoch 2: still no improvement (counter = 2 == patience)
        result = detector.check([_metric(2, 0.3, 0.55)])
        assert result.should_stop is True
        assert detector.epochs_without_improvement == 2

    def test_improvement_resets_counter(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=2))
        detector.check([_metric(0, 0.5, 0.5)])
        detector.check([_metric(1, 0.4, 0.6)])  # no improvement, counter=1
        assert detector.epochs_without_improvement == 1

        # Improvement resets counter
        detector.check([_metric(2, 0.3, 0.4)])
        assert detector.epochs_without_improvement == 0
        assert detector.best_loss == 0.4

    def test_min_delta_requires_sufficient_improvement(self):
        detector = EarlyStoppingDetector(
            EarlyStoppingConfig(patience=2, min_delta=0.05)
        )
        detector.check([_metric(0, 0.5, 0.5)])

        # Tiny improvement (0.5 -> 0.48) is less than min_delta=0.05
        result = detector.check([_metric(1, 0.4, 0.48)])
        assert detector.epochs_without_improvement == 1

        # Another tiny improvement
        result = detector.check([_metric(2, 0.3, 0.46)])
        assert result.should_stop is True

    def test_min_delta_accepts_large_improvement(self):
        detector = EarlyStoppingDetector(
            EarlyStoppingConfig(patience=2, min_delta=0.05)
        )
        detector.check([_metric(0, 0.5, 0.5)])

        # Large improvement (0.5 -> 0.4) exceeds min_delta
        result = detector.check([_metric(1, 0.4, 0.4)])
        assert detector.epochs_without_improvement == 0
        assert detector.best_loss == 0.4

    def test_metrics_without_validation_loss_are_skipped(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=2))
        # Only training loss, no validation loss
        result = detector.check([_metric(0, 0.5, None)])
        assert result.should_stop is False
        assert detector.best_loss is None

    def test_mixed_metrics_only_uses_validation_loss(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=2))
        metrics = [
            _metric(0, 0.5, None),   # skipped
            _metric(1, 0.4, 0.5),    # first observation
            _metric(2, 0.3, None),   # skipped
        ]
        result = detector.check(metrics)
        assert result.should_stop is False
        assert detector.best_loss == 0.5

    def test_empty_metrics_list(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=2))
        result = detector.check([])
        assert result.should_stop is False
        assert detector.best_loss is None

    def test_multiple_metrics_in_single_check(self):
        """Multiple metrics in one call are processed sequentially."""
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=2))
        metrics = [
            _metric(0, 0.5, 0.5),   # best = 0.5
            _metric(1, 0.4, 0.6),   # no improvement, counter=1
            _metric(2, 0.3, 0.7),   # no improvement, counter=2
        ]
        result = detector.check(metrics)
        assert result.should_stop is True
        assert detector.best_loss == 0.5

    def test_patience_of_one(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=1))
        detector.check([_metric(0, 0.5, 0.5)])
        result = detector.check([_metric(1, 0.4, 0.6)])
        assert result.should_stop is True

    def test_large_patience_no_stop(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=10))
        detector.check([_metric(0, 0.5, 0.5)])
        for i in range(1, 10):
            result = detector.check([_metric(i, 0.4, 0.6)])
        # 9 epochs without improvement, patience is 10
        assert result.should_stop is False
        assert detector.epochs_without_improvement == 9


# --- Notification / reason message ---


class TestEarlyStoppingNotification:
    def test_reason_when_triggered(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=2))
        detector.check([_metric(0, 0.5, 0.5)])
        detector.check([_metric(1, 0.4, 0.6)])
        result = detector.check([_metric(2, 0.3, 0.7)])
        assert result.should_stop is True
        assert "Early stopping triggered" in result.reason
        assert "no improvement" in result.reason
        assert "0.500000" in result.reason  # best loss

    def test_no_reason_when_not_triggered(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=3))
        result = detector.check([_metric(0, 0.5, 0.5)])
        assert result.reason == ""


# --- History tracking ---


class TestEarlyStoppingHistory:
    def test_history_records_validation_losses(self):
        detector = EarlyStoppingDetector()
        detector.check([_metric(0, 0.5, 0.5)])
        detector.check([_metric(1, 0.4, 0.6)])
        assert detector.history == [0.5, 0.6]

    def test_history_skips_none_validation_loss(self):
        detector = EarlyStoppingDetector()
        detector.check([_metric(0, 0.5, None)])
        detector.check([_metric(1, 0.4, 0.6)])
        assert detector.history == [0.6]

    def test_history_is_copy(self):
        detector = EarlyStoppingDetector()
        detector.check([_metric(0, 0.5, 0.5)])
        h = detector.history
        h.append(999.0)
        assert detector.history == [0.5]


# --- Reset ---


class TestEarlyStoppingReset:
    def test_reset_clears_state(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=2))
        detector.check([_metric(0, 0.5, 0.5)])
        detector.check([_metric(1, 0.4, 0.6)])
        detector.reset()
        assert detector.best_loss is None
        assert detector.epochs_without_improvement == 0
        assert detector.history == []

    def test_reset_allows_reuse(self):
        detector = EarlyStoppingDetector(EarlyStoppingConfig(patience=1))
        detector.check([_metric(0, 0.5, 0.5)])
        detector.check([_metric(1, 0.4, 0.6)])
        assert detector.epochs_without_improvement == 1

        detector.reset()
        result = detector.check([_metric(0, 0.3, 0.3)])
        assert result.should_stop is False
        assert detector.best_loss == 0.3
