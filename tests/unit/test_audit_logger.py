"""
Unit tests for the training audit logger.

Tests cover event logging, querying, filtering, timeline retrieval,
and all convenience logging methods.

Requirements: 4.10
"""

from src.fine_tuning.audit_logger import (
    AuditEntry,
    AuditEventType,
    TrainingAuditLogger,
)


class TestTrainingAuditLogger:
    def test_log_creates_entry(self):
        logger = TrainingAuditLogger()
        entry = logger.log(AuditEventType.JOB_CREATED, job_id="job-1")
        assert entry.event_type == AuditEventType.JOB_CREATED
        assert entry.job_id == "job-1"
        assert entry.timestamp is not None

    def test_entries_are_stored(self):
        logger = TrainingAuditLogger()
        logger.log(AuditEventType.JOB_CREATED, job_id="job-1")
        logger.log(AuditEventType.JOB_STARTED, job_id="job-1")
        assert len(logger.entries) == 2

    def test_entries_returns_copy(self):
        logger = TrainingAuditLogger()
        logger.log(AuditEventType.JOB_CREATED, job_id="job-1")
        entries = logger.entries
        entries.clear()
        assert len(logger.entries) == 1

    def test_log_with_parameters(self):
        logger = TrainingAuditLogger()
        params = {"model_id": "claude-v2", "epochs": 3}
        entry = logger.log(
            AuditEventType.JOB_CREATED, job_id="job-1", parameters=params
        )
        assert entry.parameters["model_id"] == "claude-v2"

    def test_log_with_error(self):
        logger = TrainingAuditLogger()
        entry = logger.log(
            AuditEventType.JOB_FAILED, job_id="job-1", error="OOM"
        )
        assert entry.error == "OOM"

    def test_log_with_user_id(self):
        logger = TrainingAuditLogger()
        entry = logger.log(
            AuditEventType.JOB_CREATED, job_id="job-1", user_id="user-42"
        )
        assert entry.user_id == "user-42"


class TestConvenienceMethods:
    def test_log_job_created(self):
        logger = TrainingAuditLogger()
        entry = logger.log_job_created("job-1", {"model": "claude"})
        assert entry.event_type == AuditEventType.JOB_CREATED
        assert "created" in entry.outcome.lower()

    def test_log_job_started(self):
        logger = TrainingAuditLogger()
        entry = logger.log_job_started("job-1")
        assert entry.event_type == AuditEventType.JOB_STARTED

    def test_log_job_completed(self):
        logger = TrainingAuditLogger()
        entry = logger.log_job_completed("job-1", {"arn": "arn:..."})
        assert entry.event_type == AuditEventType.JOB_COMPLETED

    def test_log_job_failed(self):
        logger = TrainingAuditLogger()
        entry = logger.log_job_failed("job-1", "OOM error")
        assert entry.event_type == AuditEventType.JOB_FAILED
        assert entry.error == "OOM error"

    def test_log_job_stopped(self):
        logger = TrainingAuditLogger()
        entry = logger.log_job_stopped("job-1", "early stopping")
        assert entry.event_type == AuditEventType.JOB_STOPPED
        assert "early stopping" in entry.outcome

    def test_log_validation_success(self):
        logger = TrainingAuditLogger()
        entry = logger.log_validation("job-1", success=True)
        assert entry.event_type == AuditEventType.VALIDATION_COMPLETED

    def test_log_validation_failure(self):
        logger = TrainingAuditLogger()
        entry = logger.log_validation("job-1", success=False, error="bad format")
        assert entry.event_type == AuditEventType.VALIDATION_FAILED
        assert entry.error == "bad format"

    def test_log_metrics_recorded(self):
        logger = TrainingAuditLogger()
        entry = logger.log_metrics_recorded("job-1", {"loss": 0.5})
        assert entry.event_type == AuditEventType.METRICS_RECORDED

    def test_log_early_stopping(self):
        logger = TrainingAuditLogger()
        entry = logger.log_early_stopping("job-1", "No improvement for 3 epochs")
        assert entry.event_type == AuditEventType.EARLY_STOPPING_TRIGGERED

    def test_log_model_registered(self):
        logger = TrainingAuditLogger()
        entry = logger.log_model_registered("job-1", {"model_arn": "arn:..."})
        assert entry.event_type == AuditEventType.MODEL_REGISTERED

    def test_log_cost_estimated(self):
        logger = TrainingAuditLogger()
        entry = logger.log_cost_estimated("job-1", {"cost": 50.0})
        assert entry.event_type == AuditEventType.COST_ESTIMATED


class TestQuerying:
    def test_get_entries_all(self):
        logger = TrainingAuditLogger()
        logger.log_job_created("job-1")
        logger.log_job_created("job-2")
        assert len(logger.get_entries()) == 2

    def test_filter_by_job_id(self):
        logger = TrainingAuditLogger()
        logger.log_job_created("job-1")
        logger.log_job_started("job-1")
        logger.log_job_created("job-2")
        entries = logger.get_entries(job_id="job-1")
        assert len(entries) == 2
        assert all(e.job_id == "job-1" for e in entries)

    def test_filter_by_event_type(self):
        logger = TrainingAuditLogger()
        logger.log_job_created("job-1")
        logger.log_job_started("job-1")
        logger.log_job_completed("job-1")
        entries = logger.get_entries(event_type=AuditEventType.JOB_STARTED)
        assert len(entries) == 1

    def test_filter_by_both(self):
        logger = TrainingAuditLogger()
        logger.log_job_created("job-1")
        logger.log_job_created("job-2")
        logger.log_job_started("job-1")
        entries = logger.get_entries(
            job_id="job-1", event_type=AuditEventType.JOB_CREATED
        )
        assert len(entries) == 1

    def test_get_job_timeline(self):
        logger = TrainingAuditLogger()
        logger.log_job_created("job-1")
        logger.log_job_started("job-1")
        logger.log_job_completed("job-1")
        timeline = logger.get_job_timeline("job-1")
        assert len(timeline) == 3
        # Verify chronological order
        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp

    def test_empty_timeline(self):
        logger = TrainingAuditLogger()
        assert logger.get_job_timeline("nonexistent") == []


class TestClear:
    def test_clear_removes_all(self):
        logger = TrainingAuditLogger()
        logger.log_job_created("job-1")
        logger.log_job_created("job-2")
        logger.clear()
        assert len(logger.entries) == 0
