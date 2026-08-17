"""
Unit tests for the SageMaker training job creator.

Tests cover job creation, parameter building, data channel configuration,
validation data handling, and error handling with mocked boto3 client.

Requirements: 4.4
"""

from unittest.mock import MagicMock

from src.data_models.fine_tuning import HyperparameterConfig
from src.fine_tuning.sagemaker_trainer import (
    SageMakerJobResult,
    create_sagemaker_training_job,
)


# --- Helpers ---

def _default_hyperparams() -> HyperparameterConfig:
    return HyperparameterConfig(
        learning_rate=1e-5,
        epochs=3,
        batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        max_seq_length=2048,
    )


def _mock_sagemaker_client(
    job_arn: str = "arn:aws:sagemaker:us-east-1:123456:training-job/test-job",
):
    client = MagicMock()
    client.create_training_job.return_value = {"TrainingJobArn": job_arn}
    return client


# --- Successful job creation ---


class TestCreateSageMakerJob:
    def test_basic_job_creation(self):
        client = _mock_sagemaker_client()
        result = create_sagemaker_training_job(
            sagemaker_client=client,
            job_name="sm-ft-job",
            container_image_uri="123456.dkr.ecr.us-east-1.amazonaws.com/my-training:latest",
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/SageMakerRole",
            hyperparameters=_default_hyperparams(),
        )
        assert result.success is True
        assert "arn:aws:sagemaker" in result.job_arn
        client.create_training_job.assert_called_once()

    def test_job_arn_extracted(self):
        expected_arn = "arn:aws:sagemaker:us-east-1:123456:training-job/custom-arn"
        client = _mock_sagemaker_client(job_arn=expected_arn)
        result = create_sagemaker_training_job(
            sagemaker_client=client,
            job_name="test-job",
            container_image_uri="123456.dkr.ecr.us-east-1.amazonaws.com/img:latest",
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/SageMakerRole",
            hyperparameters=_default_hyperparams(),
        )
        assert result.job_arn == expected_arn


# --- Data channels ---


class TestDataChannels:
    def test_training_channel_only(self):
        client = _mock_sagemaker_client()
        create_sagemaker_training_job(
            sagemaker_client=client,
            job_name="test-job",
            container_image_uri="img:latest",
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/Role",
            hyperparameters=_default_hyperparams(),
        )
        call_kwargs = client.create_training_job.call_args[1]
        channels = call_kwargs["InputDataConfig"]
        assert len(channels) == 1
        assert channels[0]["ChannelName"] == "training"

    def test_with_validation_channel(self):
        client = _mock_sagemaker_client()
        create_sagemaker_training_job(
            sagemaker_client=client,
            job_name="test-job",
            container_image_uri="img:latest",
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/Role",
            hyperparameters=_default_hyperparams(),
            validation_data_s3_uri="s3://bucket/val/",
        )
        call_kwargs = client.create_training_job.call_args[1]
        channels = call_kwargs["InputDataConfig"]
        assert len(channels) == 2
        channel_names = [c["ChannelName"] for c in channels]
        assert "training" in channel_names
        assert "validation" in channel_names


# --- Hyperparameters ---


class TestHyperparameters:
    def test_hyperparameters_as_strings(self):
        client = _mock_sagemaker_client()
        hp = HyperparameterConfig(
            learning_rate=2e-5, epochs=5, batch_size=16,
            warmup_steps=200, weight_decay=0.02, max_seq_length=4096,
        )
        create_sagemaker_training_job(
            sagemaker_client=client,
            job_name="test-job",
            container_image_uri="img:latest",
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/Role",
            hyperparameters=hp,
        )
        call_kwargs = client.create_training_job.call_args[1]
        hp_dict = call_kwargs["HyperParameters"]
        assert hp_dict["epochs"] == "5"
        assert hp_dict["batch_size"] == "16"
        assert hp_dict["max_seq_length"] == "4096"
        # All values should be strings
        for v in hp_dict.values():
            assert isinstance(v, str)

    def test_lora_params_included_when_set(self):
        client = _mock_sagemaker_client()
        hp = HyperparameterConfig(lora_rank=8, lora_alpha=16.0)
        create_sagemaker_training_job(
            sagemaker_client=client,
            job_name="test-job",
            container_image_uri="img:latest",
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/Role",
            hyperparameters=hp,
        )
        call_kwargs = client.create_training_job.call_args[1]
        hp_dict = call_kwargs["HyperParameters"]
        assert hp_dict["lora_rank"] == "8"
        assert hp_dict["lora_alpha"] == "16.0"

    def test_lora_params_excluded_when_none(self):
        client = _mock_sagemaker_client()
        hp = _default_hyperparams()
        create_sagemaker_training_job(
            sagemaker_client=client,
            job_name="test-job",
            container_image_uri="img:latest",
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/Role",
            hyperparameters=hp,
        )
        call_kwargs = client.create_training_job.call_args[1]
        hp_dict = call_kwargs["HyperParameters"]
        assert "lora_rank" not in hp_dict
        assert "lora_alpha" not in hp_dict


# --- Resource config ---


class TestResourceConfig:
    def test_default_instance_config(self):
        client = _mock_sagemaker_client()
        create_sagemaker_training_job(
            sagemaker_client=client,
            job_name="test-job",
            container_image_uri="img:latest",
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/Role",
            hyperparameters=_default_hyperparams(),
        )
        call_kwargs = client.create_training_job.call_args[1]
        resource = call_kwargs["ResourceConfig"]
        assert resource["InstanceType"] == "ml.g5.2xlarge"
        assert resource["InstanceCount"] == 1
        assert resource["VolumeSizeInGB"] == 100

    def test_custom_instance_config(self):
        client = _mock_sagemaker_client()
        create_sagemaker_training_job(
            sagemaker_client=client,
            job_name="test-job",
            container_image_uri="img:latest",
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/Role",
            hyperparameters=_default_hyperparams(),
            instance_type="ml.p4d.24xlarge",
            instance_count=2,
            volume_size_gb=500,
        )
        call_kwargs = client.create_training_job.call_args[1]
        resource = call_kwargs["ResourceConfig"]
        assert resource["InstanceType"] == "ml.p4d.24xlarge"
        assert resource["InstanceCount"] == 2
        assert resource["VolumeSizeInGB"] == 500


# --- Error handling ---


class TestErrorHandling:
    def test_api_exception(self):
        client = MagicMock()
        client.create_training_job.side_effect = Exception("ResourceLimitExceeded")
        result = create_sagemaker_training_job(
            sagemaker_client=client,
            job_name="test-job",
            container_image_uri="img:latest",
            training_data_s3_uri="s3://bucket/train/",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/Role",
            hyperparameters=_default_hyperparams(),
        )
        assert result.success is False
        assert "ResourceLimitExceeded" in result.error


# --- SageMakerJobResult ---


class TestSageMakerJobResult:
    def test_defaults(self):
        result = SageMakerJobResult(success=True)
        assert result.success is True
        assert result.job_arn == ""
        assert result.error == ""
