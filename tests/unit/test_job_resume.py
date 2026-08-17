"""
Unit tests for job resume support.

Tests cover resume eligibility checking, resume config building,
and the resume flow with mocked provider clients.

Requirements: 4.13
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.data_models.fine_tuning import (
    FineTuningJob,
    FineTuningStatus,
    HyperparameterConfig,
    TrainingMetrics,
)
from src.fine_tuning.job_resume import (
    ResumeResult,
    build_resume_config,
    can_resume_job,
    resume_training_job,
)


def _make_hp() -> HyperparameterConfig:
    return HyperparameterConfig(learning_rate=1e-5, epochs=5, batch_size=8)


def _make_stopped_job(
    model_id: str = "anthropic.claude-v2",
    status: FineTuningStatus = FineTuningStatus.STOPPED,
    provider_job_id: str = "bedrock-job-123",
    completed_epochs: int = 2,
) -> FineTuningJob:
    metrics = []
    for ep in range(1, completed_epochs + 1):
        metrics.append(
            TrainingMetrics(
                epoch=ep, step=ep * 100, training_loss=1.0 / ep,
                validation_loss=1.1 / ep, learning_rate=1e-5,
                timestamp=datetime(2024, 1, 1, ep, tzinfo=timezone.utc),
            )
        )
    return FineTuningJob(
        job_id="job-resume-001",
        status=status,
        base_model_id=model_id,
        training_data_s3_uri="s3://bucket/train.jsonl",
        hyperparameters=_make_hp(),
        estimated_cost=50.0,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        provider_job_id=provider_job_id,
        training_metrics=metrics,
    )


class TestCanResumeJob:
    def test_stopped_claude_job_can_resume(self):
        job = _make_stopped_job()
        can, reason = can_resume_job(job)
        assert can is True

    def test_failed_job_can_resume(self):
        job = _make_stopped_job(status=FineTuningStatus.FAILED)
        can, reason = can_resume_job(job)
        assert can is True

    def test_completed_job_cannot_resume(self):
        job = _make_stopped_job(status=FineTuningStatus.COMPLETED)
        can, reason = can_resume_job(job)
        assert can is False
        assert "FAILED or STOPPED" in reason

    def test_training_job_cannot_resume(self):
        job = _make_stopped_job(status=FineTuningStatus.TRAINING)
        can, reason = can_resume_job(job)
        assert can is False

    def test_titan_cannot_resume(self):
        job = _make_stopped_job(model_id="amazon.titan-text-v1")
        can, reason = can_resume_job(job)
        assert can is False
        assert "does not support" in reason

    def test_unknown_model_cannot_resume(self):
        job = _make_stopped_job(model_id="unknown-model")
        can, reason = can_resume_job(job)
        assert can is False

    def test_no_provider_job_id_cannot_resume(self):
        job = _make_stopped_job(provider_job_id="")
        job.provider_job_id = None
        can, reason = can_resume_job(job)
        assert can is False
        assert "no provider job ID" in reason

    def test_llama_can_resume(self):
        job = _make_stopped_job(model_id="meta.llama3-8b-v1")
        can, reason = can_resume_job(job)
        assert can is True


class TestBuildResumeConfig:
    def test_calculates_remaining_epochs(self):
        job = _make_stopped_job(completed_epochs=2)
        config = build_resume_config(job)
        assert config["completed_epochs"] == 2
        assert config["remaining_epochs"] == 3  # 5 total - 2 completed

    def test_minimum_one_remaining_epoch(self):
        job = _make_stopped_job(completed_epochs=5)
        config = build_resume_config(job)
        assert config["remaining_epochs"] == 1

    def test_no_warmup_on_resume(self):
        job = _make_stopped_job()
        config = build_resume_config(job)
        hp = config["hyperparameters"]
        assert hp.warmup_steps == 0

    def test_preserves_original_config(self):
        job = _make_stopped_job()
        config = build_resume_config(job)
        assert config["base_model_id"] == "anthropic.claude-v2"
        assert config["training_data_s3_uri"] == "s3://bucket/train.jsonl"

    def test_adjusted_hyperparameters(self):
        job = _make_stopped_job(completed_epochs=2)
        new_hp = HyperparameterConfig(
            learning_rate=5e-6, epochs=5, batch_size=4
        )
        config = build_resume_config(job, adjusted_hyperparameters=new_hp)
        hp = config["hyperparameters"]
        assert hp.learning_rate == 5e-6
        assert hp.batch_size == 4

    def test_no_metrics_means_zero_completed(self):
        job = _make_stopped_job(completed_epochs=0)
        config = build_resume_config(job)
        assert config["completed_epochs"] == 0
        assert config["remaining_epochs"] == 5


class TestResumeTrainingJob:
    def test_successful_resume(self):
        job = _make_stopped_job()
        client = MagicMock()
        client.create_model_customization_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123:job/resumed"
        }
        result = resume_training_job(
            job, client, role_arn="arn:aws:iam::123:role/R",
            output_s3_uri="s3://bucket/output/",
        )
        assert result.success is True
        assert "resumed" in result.new_job_arn
        assert result.resumed_from_job_id == "job-resume-001"
        client.create_model_customization_job.assert_called_once()

    def test_ineligible_job_returns_error(self):
        job = _make_stopped_job(status=FineTuningStatus.COMPLETED)
        client = MagicMock()
        result = resume_training_job(job, client)
        assert result.success is False
        assert "FAILED or STOPPED" in result.error
        client.create_model_customization_job.assert_not_called()

    def test_provider_error_handled(self):
        job = _make_stopped_job()
        client = MagicMock()
        client.create_model_customization_job.side_effect = Exception("API error")
        result = resume_training_job(
            job, client, role_arn="arn:aws:iam::123:role/R",
            output_s3_uri="s3://bucket/output/",
        )
        assert result.success is False
        assert "Failed to resume" in result.error

    def test_checkpoint_info_populated(self):
        job = _make_stopped_job(completed_epochs=2)
        client = MagicMock()
        client.create_model_customization_job.return_value = {"jobArn": "arn:new"}
        result = resume_training_job(
            job, client, role_arn="arn:role", output_s3_uri="s3://out/"
        )
        assert result.checkpoint_info["completed_epochs"] == 2
        assert result.checkpoint_info["remaining_epochs"] == 3


class TestResumeResult:
    def test_defaults(self):
        result = ResumeResult(success=True)
        assert result.new_job_arn == ""
        assert result.resumed_from_job_id == ""
        assert result.checkpoint_info == {}
        assert result.error == ""
