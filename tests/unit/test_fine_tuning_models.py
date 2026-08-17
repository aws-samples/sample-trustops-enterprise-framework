"""
Unit tests for fine-tuning data models.

Tests Requirements: 4.1, 4.2, 4.5, 4.11
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from src.data_models.fine_tuning import (
    FineTuningStatus,
    HyperparameterConfig,
    FineTuningConfig,
    TrainingMetrics,
    CostEstimate,
    FineTuningJob,
)


class TestFineTuningStatus:
    """Test FineTuningStatus enumeration."""

    def test_fine_tuning_status_enum_values(self):
        """Test FineTuningStatus enum has all required values."""
        assert FineTuningStatus.PENDING == "pending"
        assert FineTuningStatus.VALIDATING == "validating"
        assert FineTuningStatus.TRAINING == "training"
        assert FineTuningStatus.COMPLETED == "completed"
        assert FineTuningStatus.FAILED == "failed"
        assert FineTuningStatus.STOPPED == "stopped"


class TestHyperparameterConfig:
    """Test HyperparameterConfig data model."""

    def test_hyperparameter_config_valid_creation(self):
        """Test creating HyperparameterConfig with valid data."""
        config = HyperparameterConfig(
            learning_rate=1e-5,
            epochs=3,
            batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            max_seq_length=2048,
        )

        assert config.learning_rate == 1e-5
        assert config.epochs == 3
        assert config.batch_size == 8
        assert config.warmup_steps == 100
        assert config.weight_decay == 0.01
        assert config.max_seq_length == 2048
        assert config.lora_rank is None
        assert config.lora_alpha is None

    def test_hyperparameter_config_with_lora_params(self):
        """Test creating HyperparameterConfig with LoRA parameters."""
        config = HyperparameterConfig(
            learning_rate=2e-4,
            epochs=5,
            batch_size=16,
            warmup_steps=50,
            weight_decay=0.001,
            max_seq_length=1024,
            lora_rank=8,
            lora_alpha=16.0,
        )

        assert config.lora_rank == 8
        assert config.lora_alpha == 16.0

    def test_hyperparameter_config_defaults(self):
        """Test HyperparameterConfig default values."""
        config = HyperparameterConfig()

        assert config.learning_rate == 1e-5
        assert config.epochs == 3
        assert config.batch_size == 8
        assert config.warmup_steps == 100
        assert config.weight_decay == 0.01
        assert config.max_seq_length == 2048

    def test_hyperparameter_config_negative_learning_rate_rejected(self):
        """Test that negative learning rate is rejected."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(learning_rate=-0.001)

    def test_hyperparameter_config_zero_learning_rate_rejected(self):
        """Test that zero learning rate is rejected."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(learning_rate=0.0)

    def test_hyperparameter_config_excessive_learning_rate_rejected(self):
        """Test that learning rate > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(learning_rate=1.5)

    def test_hyperparameter_config_negative_epochs_rejected(self):
        """Test that negative epochs are rejected."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(epochs=-1)

    def test_hyperparameter_config_zero_epochs_rejected(self):
        """Test that zero epochs are rejected."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(epochs=0)

    def test_hyperparameter_config_negative_batch_size_rejected(self):
        """Test that negative batch size is rejected."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(batch_size=-8)

    def test_hyperparameter_config_negative_weight_decay_rejected(self):
        """Test that negative weight decay is rejected."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(weight_decay=-0.01)

    def test_hyperparameter_config_negative_lora_rank_rejected(self):
        """Test that negative LoRA rank is rejected."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(lora_rank=-8)

    def test_hyperparameter_config_negative_lora_alpha_rejected(self):
        """Test that negative LoRA alpha is rejected."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(lora_alpha=-16.0)


class TestFineTuningConfig:
    """Test FineTuningConfig data model."""

    def test_fine_tuning_config_valid_creation(self):
        """Test creating FineTuningConfig with valid data."""
        hyperparams = HyperparameterConfig()
        config = FineTuningConfig(
            base_model_id="anthropic.claude-v2",
            training_data_id="ds-train-001",
            validation_data_id="ds-val-001",
            hyperparameters=hyperparams,
            job_name="my-fine-tuning-job",
            output_model_name="my-custom-model",
            auto_hyperparameter_tuning=False,
        )

        assert config.base_model_id == "anthropic.claude-v2"
        assert config.training_data_id == "ds-train-001"
        assert config.validation_data_id == "ds-val-001"
        assert config.hyperparameters == hyperparams
        assert config.job_name == "my-fine-tuning-job"
        assert config.output_model_name == "my-custom-model"
        assert config.auto_hyperparameter_tuning is False

    def test_fine_tuning_config_without_validation_data(self):
        """Test creating FineTuningConfig without validation data."""
        hyperparams = HyperparameterConfig()
        config = FineTuningConfig(
            base_model_id="anthropic.claude-v2",
            training_data_id="ds-train-001",
            hyperparameters=hyperparams,
            job_name="my-fine-tuning-job",
            output_model_name="my-custom-model",
        )

        assert config.validation_data_id is None

    def test_fine_tuning_config_with_auto_tuning(self):
        """Test creating FineTuningConfig with auto hyperparameter tuning."""
        hyperparams = HyperparameterConfig()
        config = FineTuningConfig(
            base_model_id="anthropic.claude-v2",
            training_data_id="ds-train-001",
            hyperparameters=hyperparams,
            job_name="my-fine-tuning-job",
            output_model_name="my-custom-model",
            auto_hyperparameter_tuning=True,
        )

        assert config.auto_hyperparameter_tuning is True

    def test_fine_tuning_config_empty_base_model_id_rejected(self):
        """Test that empty base_model_id is rejected."""
        hyperparams = HyperparameterConfig()
        with pytest.raises(ValidationError):
            FineTuningConfig(
                base_model_id="",
                training_data_id="ds-train-001",
                hyperparameters=hyperparams,
                job_name="my-fine-tuning-job",
                output_model_name="my-custom-model",
            )

    def test_fine_tuning_config_empty_training_data_id_rejected(self):
        """Test that empty training_data_id is rejected."""
        hyperparams = HyperparameterConfig()
        with pytest.raises(ValidationError):
            FineTuningConfig(
                base_model_id="anthropic.claude-v2",
                training_data_id="",
                hyperparameters=hyperparams,
                job_name="my-fine-tuning-job",
                output_model_name="my-custom-model",
            )

    def test_fine_tuning_config_empty_job_name_rejected(self):
        """Test that empty job_name is rejected."""
        hyperparams = HyperparameterConfig()
        with pytest.raises(ValidationError):
            FineTuningConfig(
                base_model_id="anthropic.claude-v2",
                training_data_id="ds-train-001",
                hyperparameters=hyperparams,
                job_name="",
                output_model_name="my-custom-model",
            )


class TestTrainingMetrics:
    """Test TrainingMetrics data model."""

    def test_training_metrics_valid_creation(self):
        """Test creating TrainingMetrics with valid data."""
        now = datetime.now()
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            training_loss=0.5,
            validation_loss=0.6,
            learning_rate=1e-5,
            timestamp=now,
        )

        assert metrics.epoch == 1
        assert metrics.step == 100
        assert metrics.training_loss == 0.5
        assert metrics.validation_loss == 0.6
        assert metrics.learning_rate == 1e-5
        assert metrics.timestamp == now

    def test_training_metrics_without_validation_loss(self):
        """Test creating TrainingMetrics without validation loss."""
        now = datetime.now()
        metrics = TrainingMetrics(
            epoch=2,
            step=200,
            training_loss=0.3,
            learning_rate=5e-6,
            timestamp=now,
        )

        assert metrics.validation_loss is None

    def test_training_metrics_negative_epoch_rejected(self):
        """Test that negative epoch is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            TrainingMetrics(
                epoch=-1,
                step=100,
                training_loss=0.5,
                learning_rate=1e-5,
                timestamp=now,
            )

    def test_training_metrics_negative_step_rejected(self):
        """Test that negative step is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            TrainingMetrics(
                epoch=1,
                step=-100,
                training_loss=0.5,
                learning_rate=1e-5,
                timestamp=now,
            )

    def test_training_metrics_negative_training_loss_rejected(self):
        """Test that negative training loss is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            TrainingMetrics(
                epoch=1,
                step=100,
                training_loss=-0.5,
                learning_rate=1e-5,
                timestamp=now,
            )

    def test_training_metrics_negative_validation_loss_rejected(self):
        """Test that negative validation loss is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            TrainingMetrics(
                epoch=1,
                step=100,
                training_loss=0.5,
                validation_loss=-0.6,
                learning_rate=1e-5,
                timestamp=now,
            )

    def test_training_metrics_zero_learning_rate_rejected(self):
        """Test that zero learning rate is rejected."""
        now = datetime.now()
        with pytest.raises(ValidationError):
            TrainingMetrics(
                epoch=1,
                step=100,
                training_loss=0.5,
                learning_rate=0.0,
                timestamp=now,
            )


class TestCostEstimate:
    """Test CostEstimate data model."""

    def test_cost_estimate_valid_creation(self):
        """Test creating CostEstimate with valid data."""
        estimate = CostEstimate(
            estimated_training_cost=150.50,
            estimated_duration_hours=2.5,
            cost_breakdown={"compute": 120.0, "storage": 30.5},
            currency="USD",
            confidence="high",
        )

        assert estimate.estimated_training_cost == 150.50
        assert estimate.estimated_duration_hours == 2.5
        assert estimate.cost_breakdown == {"compute": 120.0, "storage": 30.5}
        assert estimate.currency == "USD"
        assert estimate.confidence == "high"

    def test_cost_estimate_confidence_levels(self):
        """Test CostEstimate with different confidence levels."""
        for confidence in ["high", "medium", "low"]:
            estimate = CostEstimate(
                estimated_training_cost=100.0,
                estimated_duration_hours=1.0,
                confidence=confidence,
            )
            assert estimate.confidence == confidence

    def test_cost_estimate_confidence_case_insensitive(self):
        """Test that confidence level is normalized to lowercase."""
        estimate = CostEstimate(
            estimated_training_cost=100.0,
            estimated_duration_hours=1.0,
            confidence="HIGH",
        )
        assert estimate.confidence == "high"

    def test_cost_estimate_invalid_confidence_rejected(self):
        """Test that invalid confidence level is rejected."""
        with pytest.raises(ValidationError):
            CostEstimate(
                estimated_training_cost=100.0,
                estimated_duration_hours=1.0,
                confidence="very_high",
            )

    def test_cost_estimate_negative_cost_rejected(self):
        """Test that negative cost is rejected."""
        with pytest.raises(ValidationError):
            CostEstimate(
                estimated_training_cost=-100.0,
                estimated_duration_hours=1.0,
                confidence="high",
            )

    def test_cost_estimate_negative_duration_rejected(self):
        """Test that negative duration is rejected."""
        with pytest.raises(ValidationError):
            CostEstimate(
                estimated_training_cost=100.0,
                estimated_duration_hours=-1.0,
                confidence="high",
            )

    def test_cost_estimate_empty_cost_breakdown(self):
        """Test creating CostEstimate with empty cost breakdown."""
        estimate = CostEstimate(
            estimated_training_cost=100.0,
            estimated_duration_hours=1.0,
            cost_breakdown={},
            confidence="medium",
        )
        assert estimate.cost_breakdown == {}

    def test_cost_estimate_default_currency(self):
        """Test CostEstimate default currency is USD."""
        estimate = CostEstimate(
            estimated_training_cost=100.0,
            estimated_duration_hours=1.0,
            confidence="high",
        )
        assert estimate.currency == "USD"


class TestFineTuningJob:
    """Test FineTuningJob data model."""

    def test_fine_tuning_job_valid_creation(self):
        """Test creating FineTuningJob with valid data."""
        now = datetime.now()
        hyperparams = HyperparameterConfig()

        job = FineTuningJob(
            job_id="job-001",
            status=FineTuningStatus.TRAINING,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/training/data.jsonl",
            validation_data_s3_uri="s3://bucket/validation/data.jsonl",
            hyperparameters=hyperparams,
            training_metrics=[],
            finetuned_model_id="custom-model-001",
            finetuned_model_arn=(
                "arn:aws:bedrock:us-east-1:123456789012:model/"
                "custom-model-001"
            ),
            estimated_cost=150.0,
            actual_cost=145.50,
            created_at=now,
            started_at=now,
            completed_at=None,
            error_message=None,
            provider_job_id="bedrock-job-123",
        )

        assert job.job_id == "job-001"
        assert job.status == FineTuningStatus.TRAINING
        assert job.base_model_id == "anthropic.claude-v2"
        assert (
            job.training_data_s3_uri == "s3://bucket/training/data.jsonl"
        )
        assert (
            job.validation_data_s3_uri
            == "s3://bucket/validation/data.jsonl"
        )
        assert job.hyperparameters == hyperparams
        assert job.finetuned_model_id == "custom-model-001"
        assert job.estimated_cost == 150.0
        assert job.actual_cost == 145.50

    def test_fine_tuning_job_with_training_metrics(self):
        """Test FineTuningJob with training metrics."""
        now = datetime.now()
        hyperparams = HyperparameterConfig()

        metrics = [
            TrainingMetrics(
                epoch=1,
                step=100,
                training_loss=0.5,
                learning_rate=1e-5,
                timestamp=now,
            ),
            TrainingMetrics(
                epoch=2,
                step=200,
                training_loss=0.3,
                learning_rate=5e-6,
                timestamp=now,
            ),
        ]

        job = FineTuningJob(
            job_id="job-002",
            status=FineTuningStatus.TRAINING,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/training/data.jsonl",
            hyperparameters=hyperparams,
            training_metrics=metrics,
            estimated_cost=150.0,
            created_at=now,
        )

        assert len(job.training_metrics) == 2
        assert job.training_metrics[0].epoch == 1
        assert job.training_metrics[1].epoch == 2

    def test_fine_tuning_job_completed_status(self):
        """Test FineTuningJob with completed status."""
        now = datetime.now()
        hyperparams = HyperparameterConfig()

        job = FineTuningJob(
            job_id="job-003",
            status=FineTuningStatus.COMPLETED,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/training/data.jsonl",
            hyperparameters=hyperparams,
            training_metrics=[],
            finetuned_model_id="custom-model-003",
            finetuned_model_arn=(
                "arn:aws:bedrock:us-east-1:123456789012:model/"
                "custom-model-003"
            ),
            estimated_cost=150.0,
            actual_cost=148.0,
            created_at=now,
            started_at=now,
            completed_at=now,
        )

        assert job.status == FineTuningStatus.COMPLETED
        assert job.completed_at is not None
        assert job.finetuned_model_id is not None

    def test_fine_tuning_job_failed_status(self):
        """Test FineTuningJob with failed status."""
        now = datetime.now()
        hyperparams = HyperparameterConfig()

        job = FineTuningJob(
            job_id="job-004",
            status=FineTuningStatus.FAILED,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/training/data.jsonl",
            hyperparameters=hyperparams,
            training_metrics=[],
            estimated_cost=150.0,
            created_at=now,
            started_at=now,
            error_message="Training failed due to insufficient data",
        )

        assert job.status == FineTuningStatus.FAILED
        assert job.error_message == "Training failed due to insufficient data"
        assert job.finetuned_model_id is None

    def test_fine_tuning_job_without_validation_data(self):
        """Test FineTuningJob without validation data."""
        now = datetime.now()
        hyperparams = HyperparameterConfig()

        job = FineTuningJob(
            job_id="job-005",
            status=FineTuningStatus.PENDING,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/training/data.jsonl",
            hyperparameters=hyperparams,
            training_metrics=[],
            estimated_cost=150.0,
            created_at=now,
        )

        assert job.validation_data_s3_uri is None

    def test_fine_tuning_job_invalid_training_s3_uri_rejected(self):
        """Test that invalid training S3 URI is rejected."""
        now = datetime.now()
        hyperparams = HyperparameterConfig()

        with pytest.raises(ValidationError):
            FineTuningJob(
                job_id="job-006",
                status=FineTuningStatus.PENDING,
                base_model_id="anthropic.claude-v2",
                training_data_s3_uri="http://bucket/training/data.jsonl",
                hyperparameters=hyperparams,
                training_metrics=[],
                estimated_cost=150.0,
                created_at=now,
            )

    def test_fine_tuning_job_invalid_validation_s3_uri_rejected(self):
        """Test that invalid validation S3 URI is rejected."""
        now = datetime.now()
        hyperparams = HyperparameterConfig()

        with pytest.raises(ValidationError):
            FineTuningJob(
                job_id="job-007",
                status=FineTuningStatus.PENDING,
                base_model_id="anthropic.claude-v2",
                training_data_s3_uri="s3://bucket/training/data.jsonl",
                validation_data_s3_uri="http://bucket/validation/data.jsonl",
                hyperparameters=hyperparams,
                training_metrics=[],
                estimated_cost=150.0,
                created_at=now,
            )

    def test_fine_tuning_job_negative_estimated_cost_rejected(self):
        """Test that negative estimated cost is rejected."""
        now = datetime.now()
        hyperparams = HyperparameterConfig()

        with pytest.raises(ValidationError):
            FineTuningJob(
                job_id="job-008",
                status=FineTuningStatus.PENDING,
                base_model_id="anthropic.claude-v2",
                training_data_s3_uri="s3://bucket/training/data.jsonl",
                hyperparameters=hyperparams,
                training_metrics=[],
                estimated_cost=-150.0,
                created_at=now,
            )

    def test_fine_tuning_job_negative_actual_cost_rejected(self):
        """Test that negative actual cost is rejected."""
        now = datetime.now()
        hyperparams = HyperparameterConfig()

        with pytest.raises(ValidationError):
            FineTuningJob(
                job_id="job-009",
                status=FineTuningStatus.COMPLETED,
                base_model_id="anthropic.claude-v2",
                training_data_s3_uri="s3://bucket/training/data.jsonl",
                hyperparameters=hyperparams,
                training_metrics=[],
                estimated_cost=150.0,
                actual_cost=-145.0,
                created_at=now,
            )

    def test_fine_tuning_job_serialization(self):
        """Test FineTuningJob JSON serialization."""
        now = datetime.now()
        hyperparams = HyperparameterConfig()

        job = FineTuningJob(
            job_id="job-010",
            status=FineTuningStatus.PENDING,
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/training/data.jsonl",
            hyperparameters=hyperparams,
            training_metrics=[],
            estimated_cost=150.0,
            created_at=now,
        )

        # Serialize to dict
        data = job.model_dump()
        assert data["job_id"] == "job-010"
        assert data["status"] == "pending"
        assert data["base_model_id"] == "anthropic.claude-v2"

        # Serialize to JSON
        json_str = job.model_dump_json()
        assert "job-010" in json_str
        assert "pending" in json_str
