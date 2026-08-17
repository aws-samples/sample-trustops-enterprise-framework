"""
Unit tests for the FineTuningPipeline class.

Tests cover validation, cost estimation, job lifecycle (start, status,
stop, list), completion/failure handling, resume, and audit logging.

Requirements: 4.1-4.14
"""

import importlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.data_models.fine_tuning import (
    CostEstimate,
    FineTuningConfig,
    FineTuningJob,
    FineTuningStatus,
    HyperparameterConfig,
    TrainingMetrics,
)
from src.data_models.model import (
    ModelCapability,
    ModelMetadata,
    ModelProvider,
    ModelStatus,
)
from src.fine_tuning.audit_logger import AuditEventType
from src.fine_tuning.fine_tuning_pipeline import (
    FineTuningPipeline,
    ValidationResult,
)


def _make_hp() -> HyperparameterConfig:
    return HyperparameterConfig(
        learning_rate=1e-5, epochs=3, batch_size=8
    )


def _make_config(
    model_id: str = "anthropic.claude-v2",
) -> FineTuningConfig:
    return FineTuningConfig(
        base_model_id=model_id,
        training_data_id="dataset-001",
        hyperparameters=_make_hp(),
        job_name="test-ft-job",
        output_model_name="my-ft-model",
    )


def _make_model_metadata(
    fine_tunable: bool = True,
    status: ModelStatus = ModelStatus.ACTIVE,
) -> ModelMetadata:
    caps = [ModelCapability.TEXT_GENERATION]
    if fine_tunable:
        caps.append(ModelCapability.FINE_TUNABLE)
    return ModelMetadata(
        id="anthropic.claude-v2",
        provider=ModelProvider.BEDROCK,
        name="Claude v2",
        capabilities=caps,
        status=status,
        fine_tuning_support=fine_tunable,
        input_modalities=["text"],
        output_modalities=["text"],
        max_tokens=4096,
        region="us-east-1",
    )


def _make_training_records(n: int = 100) -> list[dict]:
    # Each record needs enough text to pass token minimums
    # Claude requires 1000+ estimated tokens total
    return [
        {
            "prompt": f"This is a detailed question number {i} "
                      f"about a complex topic that requires thought",
            "completion": f"This is a comprehensive answer number {i} "
                          f"that provides detailed information and context",
        }
        for i in range(n)
    ]


def _mock_bedrock_client(
    job_arn: str = "arn:aws:bedrock:us-east-1:123:job/ft-job",
):
    client = MagicMock()
    client.create_model_customization_job.return_value = {
        "jobArn": job_arn,
    }
    return client


# --- Validation ---


class TestValidate:
    def test_valid_request(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        metadata = _make_model_metadata()
        records = _make_training_records(100)
        result = pipeline.validate(
            config, model_metadata=metadata,
            training_records=records,
        )
        assert result.is_valid is True
        assert result.errors == []

    def test_ineligible_model(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        metadata = _make_model_metadata(fine_tunable=False)
        result = pipeline.validate(config, model_metadata=metadata)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_invalid_format(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        bad_records = [{"wrong_field": "value"}]
        result = pipeline.validate(
            config, training_records=bad_records,
        )
        assert result.is_valid is False

    def test_too_small_dataset(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        # Claude needs 32+ records
        small_records = _make_training_records(5)
        result = pipeline.validate(
            config, training_records=small_records,
        )
        assert result.is_valid is False

    def test_validation_without_metadata_or_records(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        result = pipeline.validate(config)
        assert result.is_valid is True

    def test_validation_logs_audit(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        pipeline.validate(config)
        entries = pipeline.audit_logger.get_entries(
            event_type=AuditEventType.VALIDATION_COMPLETED
        )
        assert len(entries) == 1


# --- Cost Estimation ---


class TestEstimateCost:
    def test_returns_cost_estimate(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        result = pipeline.estimate_cost(config, num_examples=1000)
        assert isinstance(result, CostEstimate)
        assert result.estimated_training_cost > 0

    def test_logs_audit(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        pipeline.estimate_cost(config, num_examples=1000)
        entries = pipeline.audit_logger.get_entries(
            event_type=AuditEventType.COST_ESTIMATED
        )
        assert len(entries) == 1


# --- Default Hyperparameters ---


class TestDefaultHyperparameters:
    def test_returns_config(self):
        pipeline = FineTuningPipeline()
        hp = pipeline.get_default_hyperparameters(
            "anthropic.claude-v2", 500
        )
        assert isinstance(hp, HyperparameterConfig)
        assert hp.learning_rate > 0


# --- Start Job ---


class TestStartJob:
    def test_creates_job_without_provider(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        job = pipeline.start(config)
        assert job.job_id.startswith("ft-")
        assert job.status == FineTuningStatus.PENDING
        assert job.base_model_id == "anthropic.claude-v2"

    def test_default_training_uri_uses_configured_bucket(self):
        """The implicit URI must come from configuration, not a fixed name."""
        uri = FineTuningPipeline._default_training_uri("ds-42")

        assert uri.startswith("s3://")
        assert "ds-42" in uri
        # Regression guard: the old fallback hardcoded a bucket named "data",
        # which is globally unique and therefore squattable. This URI is the
        # training source handed to Bedrock.
        assert not uri.startswith("s3://data/")

    def test_default_training_uri_fails_closed(self):
        """With no bucket configured, no URI may be invented."""
        cfg = importlib.import_module("config.aws_config")
        with patch.object(
            type(cfg.config),
            "datasets_bucket",
            property(
                lambda self: (_ for _ in ()).throw(
                    cfg.BucketNotConfiguredError("unset")
                )
            ),
        ):
            with pytest.raises(cfg.BucketNotConfiguredError):
                FineTuningPipeline._default_training_uri("ds-1")

    def test_creates_job_with_provider(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        client = _mock_bedrock_client()
        job = pipeline.start(
            config, provider_client=client,
            role_arn="arn:role", output_s3_uri="s3://out/",
            training_data_s3_uri="s3://bucket/train.jsonl",
        )
        assert job.status == FineTuningStatus.TRAINING
        assert job.provider_job_id is not None

    def test_failed_provider_submission(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        client = MagicMock()
        client.create_model_customization_job.side_effect = (
            Exception("API error")
        )
        job = pipeline.start(
            config, provider_client=client,
            role_arn="arn:role", output_s3_uri="s3://out/",
            training_data_s3_uri="s3://bucket/train.jsonl",
        )
        assert job.status == FineTuningStatus.FAILED
        assert "API error" in job.error_message

    def test_job_stored_in_pipeline(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        job = pipeline.start(config)
        assert job.job_id in pipeline.jobs

    def test_progress_emitter_created(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        job = pipeline.start(config)
        emitter = pipeline.get_progress_emitter(job.job_id)
        assert emitter is not None
        assert emitter.total_epochs == 3

    def test_audit_logged(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        pipeline.start(config)
        entries = pipeline.audit_logger.get_entries(
            event_type=AuditEventType.JOB_CREATED
        )
        assert len(entries) == 1


# --- Get Status ---


class TestGetStatus:
    def test_returns_job(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        job = pipeline.start(config)
        status = pipeline.get_status(job.job_id)
        assert status is not None
        assert status.job_id == job.job_id

    def test_returns_none_for_unknown(self):
        pipeline = FineTuningPipeline()
        assert pipeline.get_status("nonexistent") is None


# --- Stop Job ---


class TestStopJob:
    def test_stops_training_job(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        client = _mock_bedrock_client()
        job = pipeline.start(
            config, provider_client=client,
            role_arn="arn:role", output_s3_uri="s3://out/",
            training_data_s3_uri="s3://bucket/train.jsonl",
        )
        stopped = pipeline.stop(job.job_id, "user request")
        assert stopped.status == FineTuningStatus.STOPPED

    def test_stop_non_training_job_noop(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        job = pipeline.start(config)  # PENDING
        stopped = pipeline.stop(job.job_id)
        assert stopped.status == FineTuningStatus.PENDING

    def test_stop_unknown_job(self):
        pipeline = FineTuningPipeline()
        assert pipeline.stop("nonexistent") is None


# --- List Jobs ---


class TestListJobs:
    def test_list_all(self):
        pipeline = FineTuningPipeline()
        pipeline.start(_make_config())
        pipeline.start(_make_config())
        assert len(pipeline.list_jobs()) == 2

    def test_filter_by_status(self):
        pipeline = FineTuningPipeline()
        pipeline.start(_make_config())
        client = _mock_bedrock_client()
        pipeline.start(
            _make_config(), provider_client=client,
            role_arn="arn:role", output_s3_uri="s3://out/",
            training_data_s3_uri="s3://bucket/train.jsonl",
        )
        pending = pipeline.list_jobs(status=FineTuningStatus.PENDING)
        training = pipeline.list_jobs(
            status=FineTuningStatus.TRAINING
        )
        assert len(pending) == 1
        assert len(training) == 1

    def test_filter_by_model_id(self):
        pipeline = FineTuningPipeline()
        pipeline.start(_make_config("anthropic.claude-v2"))
        pipeline.start(_make_config("amazon.titan-text-v1"))
        claude_jobs = pipeline.list_jobs(
            model_id="anthropic.claude-v2"
        )
        assert len(claude_jobs) == 1


# --- Handle Completion ---


class TestHandleCompletion:
    def test_unknown_job(self):
        pipeline = FineTuningPipeline()
        assert pipeline.handle_completion("nonexistent") is None

    def test_completed_job(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        job = pipeline.start(config)
        # Simulate completion
        job.status = FineTuningStatus.COMPLETED
        job.finetuned_model_arn = "arn:aws:bedrock:model/ft"
        job.finetuned_model_id = "ft-model"
        job.training_metrics = [
            TrainingMetrics(
                epoch=3, step=300, training_loss=0.1,
                validation_loss=0.12, learning_rate=1e-5,
                timestamp=datetime.now(timezone.utc),
            )
        ]
        result = pipeline.handle_completion(job.job_id)
        assert result.success is True
        assert result.finetuned_model_arn == "arn:aws:bedrock:model/ft"


# --- Handle Failure ---


class TestHandleFailure:
    def test_unknown_job(self):
        pipeline = FineTuningPipeline()
        assert pipeline.handle_failure("nonexistent") is None

    def test_failed_job(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        job = pipeline.start(config)
        job.status = FineTuningStatus.FAILED
        job.error_message = "Access denied"
        analysis = pipeline.handle_failure(job.job_id)
        assert analysis.error_category == "permission_error"


# --- Resume Job ---


class TestResumeJob:
    def test_unknown_job(self):
        pipeline = FineTuningPipeline()
        assert pipeline.resume_job("nonexistent") is None

    def test_resume_without_client(self):
        pipeline = FineTuningPipeline()
        config = _make_config()
        job = pipeline.start(config)
        job.status = FineTuningStatus.STOPPED
        job.provider_job_id = "bedrock-123"
        result = pipeline.resume_job(job.job_id)
        assert result.success is False


# --- ValidationResult ---


class TestValidationResult:
    def test_defaults(self):
        result = ValidationResult(is_valid=True)
        assert result.eligibility is None
        assert result.format_validation is None
        assert result.size_validation is None
        assert result.errors == []
