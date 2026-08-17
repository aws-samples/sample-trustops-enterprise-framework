"""
Unit tests for the job completion handler.

Tests cover ARN extraction, training metadata building, Model Registry
registration, and error handling for non-completed jobs.

Requirements: 4.9
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.data_models.fine_tuning import (
    FineTuningJob,
    FineTuningStatus,
    HyperparameterConfig,
    TrainingMetrics,
)
from src.fine_tuning.job_completion_handler import (
    CompletionResult,
    extract_model_arn_from_job,
    handle_job_completion,
)


def _make_hp() -> HyperparameterConfig:
    return HyperparameterConfig(
        learning_rate=1e-5, epochs=3, batch_size=8, warmup_steps=100
    )


def _make_completed_job(
    model_arn: str = "arn:aws:bedrock:us-east-1:123:custom-model/my-ft",
    model_id: str | None = "ft-model-001",
) -> FineTuningJob:
    return FineTuningJob(
        job_id="job-123",
        status=FineTuningStatus.COMPLETED,
        base_model_id="anthropic.claude-v2",
        training_data_s3_uri="s3://bucket/train.jsonl",
        hyperparameters=_make_hp(),
        finetuned_model_id=model_id,
        finetuned_model_arn=model_arn,
        estimated_cost=50.0,
        actual_cost=48.0,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        completed_at=datetime(2024, 1, 1, 6, 0, tzinfo=timezone.utc),
        training_metrics=[
            TrainingMetrics(
                epoch=3, step=300, training_loss=0.15,
                validation_loss=0.18, learning_rate=1e-5,
                timestamp=datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc),
            )
        ],
    )


class TestExtractModelArn:
    def test_extracts_arn_from_job(self):
        job = _make_completed_job()
        assert extract_model_arn_from_job(job) == job.finetuned_model_arn

    def test_returns_none_when_no_arn(self):
        job = _make_completed_job(model_arn="")
        # finetuned_model_arn is empty string
        job.finetuned_model_arn = None
        assert extract_model_arn_from_job(job) is None


class TestHandleJobCompletion:
    def test_successful_completion(self):
        job = _make_completed_job()
        result = handle_job_completion(job)
        assert result.success is True
        assert result.finetuned_model_arn == job.finetuned_model_arn
        assert result.finetuned_model_id == "ft-model-001"

    def test_training_metadata_contains_required_fields(self):
        job = _make_completed_job()
        result = handle_job_completion(job)
        meta = result.training_metadata
        assert meta["base_model_id"] == "anthropic.claude-v2"
        assert meta["training_data_id"] == "s3://bucket/train.jsonl"
        assert "hyperparameters" in meta
        assert meta["hyperparameters"]["learning_rate"] == 1e-5
        assert meta["hyperparameters"]["epochs"] == 3
        assert meta["job_id"] == "job-123"

    def test_training_metadata_includes_final_metrics(self):
        job = _make_completed_job()
        result = handle_job_completion(job)
        meta = result.training_metadata
        assert meta["final_training_loss"] == 0.15
        assert meta["final_validation_loss"] == 0.18

    def test_no_metrics_sets_none(self):
        job = _make_completed_job()
        job.training_metrics = []
        result = handle_job_completion(job)
        assert result.training_metadata["final_training_loss"] is None
        assert result.training_metadata["final_validation_loss"] is None

    def test_generates_model_id_from_job_id(self):
        job = _make_completed_job(model_id=None)
        job.finetuned_model_id = None
        result = handle_job_completion(job)
        assert result.finetuned_model_id == "ft-job-123"

    def test_rejects_non_completed_job(self):
        job = _make_completed_job()
        job.status = FineTuningStatus.TRAINING
        result = handle_job_completion(job)
        assert result.success is False
        assert "not completed" in result.error

    def test_rejects_job_without_arn(self):
        job = _make_completed_job(model_arn="")
        job.finetuned_model_arn = None
        result = handle_job_completion(job)
        assert result.success is False
        assert "no fine-tuned model ARN" in result.error

    def test_registers_in_model_registry(self):
        registry = MagicMock()
        job = _make_completed_job()
        result = handle_job_completion(job, model_registry=registry)
        assert result.success is True
        registry.register_fine_tuned_model.assert_called_once()
        call_kwargs = registry.register_fine_tuned_model.call_args[1]
        assert call_kwargs["model_id"] == "ft-model-001"
        assert call_kwargs["base_model_id"] == "anthropic.claude-v2"

    def test_handles_registry_error(self):
        registry = MagicMock()
        registry.register_fine_tuned_model.side_effect = Exception("DB error")
        job = _make_completed_job()
        result = handle_job_completion(job, model_registry=registry)
        assert result.success is False
        assert "Failed to register" in result.error
        # Still has metadata even on failure
        assert result.finetuned_model_arn != ""

    def test_without_registry_still_succeeds(self):
        job = _make_completed_job()
        result = handle_job_completion(job, model_registry=None)
        assert result.success is True


class TestCompletionResult:
    def test_defaults(self):
        result = CompletionResult(success=True)
        assert result.finetuned_model_id == ""
        assert result.finetuned_model_arn == ""
        assert result.training_metadata == {}
        assert result.error == ""
