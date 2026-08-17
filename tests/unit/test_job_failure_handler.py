"""
Unit tests for the job failure handler.

Tests cover error categorization, recovery suggestions, retryable
detection, and formatting for various error types.

Requirements: 4.12
"""

from datetime import datetime, timezone

from src.data_models.fine_tuning import (
    FineTuningJob,
    FineTuningStatus,
    HyperparameterConfig,
)
from src.fine_tuning.job_failure_handler import (
    FailureAnalysis,
    format_failure_report,
    handle_job_failure,
)


def _make_hp() -> HyperparameterConfig:
    return HyperparameterConfig(learning_rate=1e-5, epochs=3, batch_size=8)


def _make_failed_job(error_message: str = "Access denied") -> FineTuningJob:
    return FineTuningJob(
        job_id="job-fail-001",
        status=FineTuningStatus.FAILED,
        base_model_id="anthropic.claude-v2",
        training_data_s3_uri="s3://bucket/train.jsonl",
        hyperparameters=_make_hp(),
        estimated_cost=50.0,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        error_message=error_message,
    )


class TestHandleJobFailure:
    def test_permission_error(self):
        job = _make_failed_job("Access denied: not authorized to perform this action")
        result = handle_job_failure(job)
        assert result.error_category == "permission_error"
        assert result.is_retryable is False
        assert len(result.recovery_suggestions) > 0

    def test_throttling_error(self):
        job = _make_failed_job("Rate limit exceeded, too many requests")
        result = handle_job_failure(job)
        assert result.error_category == "throttling_error"
        assert result.is_retryable is True

    def test_quota_error(self):
        job = _make_failed_job("Resource limit exceeded for fine-tuning")
        result = handle_job_failure(job)
        assert result.error_category == "quota_error"
        assert result.is_retryable is False

    def test_validation_error(self):
        job = _make_failed_job("Invalid training data format: malformed JSON")
        result = handle_job_failure(job)
        assert result.error_category == "validation_error"
        assert result.is_retryable is False

    def test_timeout_error(self):
        job = _make_failed_job("Job timed out after 86400 seconds")
        result = handle_job_failure(job)
        assert result.error_category == "timeout_error"
        assert result.is_retryable is True

    def test_not_found_error(self):
        job = _make_failed_job("Model does not exist in this region")
        result = handle_job_failure(job)
        assert result.error_category == "not_found_error"

    def test_internal_error(self):
        job = _make_failed_job("Internal server error occurred")
        result = handle_job_failure(job)
        assert result.error_category == "internal_error"
        assert result.is_retryable is True

    def test_memory_error(self):
        job = _make_failed_job("Out of memory during training")
        result = handle_job_failure(job)
        assert result.error_category == "memory_error"
        assert result.is_retryable is False

    def test_training_error(self):
        job = _make_failed_job("Training diverged: NaN loss detected")
        result = handle_job_failure(job)
        assert result.error_category == "training_error"

    def test_unknown_error(self):
        job = _make_failed_job("Something completely unexpected happened")
        result = handle_job_failure(job)
        assert result.error_category == "unknown_error"
        assert len(result.recovery_suggestions) > 0

    def test_no_error_message(self):
        job = _make_failed_job("")
        job.error_message = None
        result = handle_job_failure(job)
        assert result.error_message == "No error message provided"

    def test_non_failed_job(self):
        job = _make_failed_job()
        job.status = FineTuningStatus.TRAINING
        result = handle_job_failure(job)
        assert result.error_category == "not_failed"
        assert "has not failed" in result.user_friendly_message

    def test_preserves_original_error(self):
        original = "Access denied: role arn:aws:iam::123:role/X"
        job = _make_failed_job(original)
        result = handle_job_failure(job)
        assert result.error_message == original


class TestFormatFailureReport:
    def test_returns_string(self):
        job = _make_failed_job("Access denied")
        analysis = handle_job_failure(job)
        report = format_failure_report(analysis)
        assert isinstance(report, str)

    def test_contains_job_id(self):
        job = _make_failed_job("Access denied")
        analysis = handle_job_failure(job)
        report = format_failure_report(analysis)
        assert "job-fail-001" in report

    def test_contains_suggestions(self):
        job = _make_failed_job("Access denied")
        analysis = handle_job_failure(job)
        report = format_failure_report(analysis)
        assert "Recovery suggestions" in report
        assert "1." in report


class TestFailureAnalysis:
    def test_defaults(self):
        result = FailureAnalysis(
            job_id="x", error_category="test",
            error_message="err", user_friendly_message="msg",
        )
        assert result.recovery_suggestions == []
        assert result.is_retryable is False
