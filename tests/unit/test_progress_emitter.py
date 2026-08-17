"""
Unit tests for the progress event emitter.

Tests cover event emission, callback registration, progress calculation,
ETA estimation, and lifecycle events (start, progress, complete, fail, stop).

Requirements: 4.7
"""

from src.fine_tuning.progress_emitter import (
    ProgressEmitter,
    TrainingProgressEvent,
)


class TestProgressEmitter:
    def test_start_emits_event(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        event = emitter.start()
        assert event.job_id == "job-1"
        assert event.status == "training"
        assert event.progress_percent == 0.0

    def test_emit_progress(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        event = emitter.emit_progress(
            epoch=1, step=100, training_loss=0.5, learning_rate=1e-5
        )
        assert event.epoch == 1
        assert event.step == 100
        assert event.training_loss == 0.5
        assert event.total_epochs == 3

    def test_progress_percent_calculation(self):
        emitter = ProgressEmitter("job-1", total_epochs=4)
        emitter.start()
        event = emitter.emit_progress(epoch=2, step=50, training_loss=0.3)
        assert event.progress_percent == 50.0

    def test_progress_percent_capped_at_100(self):
        emitter = ProgressEmitter("job-1", total_epochs=2)
        emitter.start()
        event = emitter.emit_progress(epoch=3, step=50, training_loss=0.1)
        assert event.progress_percent == 100.0

    def test_complete_event(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        event = emitter.complete(final_loss=0.1, final_val_loss=0.12)
        assert event.status == "completed"
        assert event.progress_percent == 100.0
        assert event.eta_seconds == 0.0
        assert event.training_loss == 0.1
        assert event.validation_loss == 0.12

    def test_fail_event(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        event = emitter.fail("OOM error")
        assert event.status == "failed"
        assert "OOM error" in event.message

    def test_stop_event(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        event = emitter.stop("early stopping")
        assert event.status == "stopped"
        assert "early stopping" in event.message

    def test_stop_without_reason(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        event = emitter.stop()
        assert event.status == "stopped"


class TestCallbacks:
    def test_callback_receives_events(self):
        received = []
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.on_progress(lambda e: received.append(e))
        emitter.start()
        emitter.emit_progress(epoch=1, step=50, training_loss=0.5)
        assert len(received) == 2

    def test_multiple_callbacks(self):
        count_a = []
        count_b = []
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.on_progress(lambda e: count_a.append(1))
        emitter.on_progress(lambda e: count_b.append(1))
        emitter.start()
        assert len(count_a) == 1
        assert len(count_b) == 1


class TestEventHistory:
    def test_events_stored(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        emitter.emit_progress(epoch=1, step=50, training_loss=0.5)
        emitter.emit_progress(epoch=2, step=100, training_loss=0.3)
        emitter.complete(final_loss=0.1)
        assert len(emitter.events) == 4

    def test_events_returns_copy(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        events = emitter.events
        events.clear()
        assert len(emitter.events) == 1


class TestStatus:
    def test_initial_status(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        assert emitter.status == "pending"

    def test_status_after_start(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        assert emitter.status == "training"

    def test_status_after_complete(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        emitter.complete()
        assert emitter.status == "completed"

    def test_status_after_fail(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        emitter.fail("error")
        assert emitter.status == "failed"

    def test_status_after_stop(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        emitter.stop()
        assert emitter.status == "stopped"


class TestProgressMessage:
    def test_progress_message_includes_loss(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        event = emitter.emit_progress(
            epoch=1, step=50, training_loss=0.5, validation_loss=0.6
        )
        assert "loss=" in event.message
        assert "val_loss=" in event.message

    def test_progress_message_without_val_loss(self):
        emitter = ProgressEmitter("job-1", total_epochs=3)
        emitter.start()
        event = emitter.emit_progress(epoch=1, step=50, training_loss=0.5)
        assert "val_loss" not in event.message


class TestTrainingProgressEvent:
    def test_defaults(self):
        event = TrainingProgressEvent(job_id="x", status="training")
        assert event.epoch == 0
        assert event.total_epochs == 0
        assert event.validation_loss is None
        assert event.eta_seconds is None
        assert event.timestamp is not None
