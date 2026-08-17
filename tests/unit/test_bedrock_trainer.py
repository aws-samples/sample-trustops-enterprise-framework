"""
Unit tests for the Bedrock fine-tuning job creator.

Tests cover job creation for each model family, parameter building,
validation data handling, and error handling with mocked boto3 client.

Requirements: 4.3, 4.7
"""

from unittest.mock import MagicMock

from src.data_models.fine_tuning import HyperparameterConfig
from src.fine_tuning.bedrock_trainer import (
    BedrockJobResult,
    create_bedrock_fine_tuning_job,
)


# --- Helpers ---

def _default_hyperparams() -> HyperparameterConfig:
    return HyperparameterConfig(
        learning_rate=1e-5,
        epochs=3,
        batch_size=8,
        warmup_steps=100,
    )


def _mock_bedrock_client(job_arn: str = "arn:aws:bedrock:us-east-1:123456:job/test-job"):
    client = MagicMock()
    client.create_model_customization_job.return_value = {"jobArn": job_arn}
    return client


# --- Successful job creation ---


class TestCreateBedrockJob:
    def test_claude_job_creation(self):
        client = _mock_bedrock_client()
        result = create_bedrock_fine_tuning_job(
            bedrock_client=client,
            job_name="claude-ft-job",
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/BedrockRole",
            hyperparameters=_default_hyperparams(),
            output_model_name="my-claude-ft",
        )
        assert result.success is True
        assert "arn:aws:bedrock" in result.job_arn
        client.create_model_customization_job.assert_called_once()

    def test_titan_job_creation(self):
        client = _mock_bedrock_client()
        result = create_bedrock_fine_tuning_job(
            bedrock_client=client,
            job_name="titan-ft-job",
            base_model_id="amazon.titan-text-express-v1",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/BedrockRole",
            hyperparameters=_default_hyperparams(),
            output_model_name="my-titan-ft",
        )
        assert result.success is True

    def test_llama_job_creation(self):
        client = _mock_bedrock_client()
        result = create_bedrock_fine_tuning_job(
            bedrock_client=client,
            job_name="llama-ft-job",
            base_model_id="meta.llama3-8b-instruct-v1",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/BedrockRole",
            hyperparameters=_default_hyperparams(),
            output_model_name="my-llama-ft",
        )
        assert result.success is True

    def test_job_arn_extracted(self):
        expected_arn = "arn:aws:bedrock:us-east-1:123456:job/custom-arn"
        client = _mock_bedrock_client(job_arn=expected_arn)
        result = create_bedrock_fine_tuning_job(
            bedrock_client=client,
            job_name="test-job",
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/BedrockRole",
            hyperparameters=_default_hyperparams(),
            output_model_name="my-model",
        )
        assert result.job_arn == expected_arn


# --- Validation data ---


class TestValidationData:
    def test_with_validation_data(self):
        client = _mock_bedrock_client()
        result = create_bedrock_fine_tuning_job(
            bedrock_client=client,
            job_name="test-job",
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/BedrockRole",
            hyperparameters=_default_hyperparams(),
            output_model_name="my-model",
            validation_data_s3_uri="s3://bucket/val.jsonl",
        )
        assert result.success is True
        call_kwargs = client.create_model_customization_job.call_args[1]
        assert "validationDataConfig" in call_kwargs

    def test_without_validation_data(self):
        client = _mock_bedrock_client()
        create_bedrock_fine_tuning_job(
            bedrock_client=client,
            job_name="test-job",
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/BedrockRole",
            hyperparameters=_default_hyperparams(),
            output_model_name="my-model",
        )
        call_kwargs = client.create_model_customization_job.call_args[1]
        assert "validationDataConfig" not in call_kwargs


# --- Parameters passed correctly ---


class TestParameterPassing:
    def test_hyperparameters_as_strings(self):
        client = _mock_bedrock_client()
        hp = HyperparameterConfig(learning_rate=2e-5, epochs=5, batch_size=16, warmup_steps=200)
        create_bedrock_fine_tuning_job(
            bedrock_client=client,
            job_name="test-job",
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/BedrockRole",
            hyperparameters=hp,
            output_model_name="my-model",
        )
        call_kwargs = client.create_model_customization_job.call_args[1]
        hp_dict = call_kwargs["hyperParameters"]
        assert hp_dict["epochCount"] == "5"
        assert hp_dict["batchSize"] == "16"
        assert hp_dict["learningRate"] == str(2e-5)

    def test_customization_type_is_fine_tuning(self):
        client = _mock_bedrock_client()
        create_bedrock_fine_tuning_job(
            bedrock_client=client,
            job_name="test-job",
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/BedrockRole",
            hyperparameters=_default_hyperparams(),
            output_model_name="my-model",
        )
        call_kwargs = client.create_model_customization_job.call_args[1]
        assert call_kwargs["customizationType"] == "FINE_TUNING"


# --- Error handling ---


class TestErrorHandling:
    def test_unknown_model_family(self):
        client = MagicMock()
        result = create_bedrock_fine_tuning_job(
            bedrock_client=client,
            job_name="test-job",
            base_model_id="unknown-model-xyz",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/BedrockRole",
            hyperparameters=_default_hyperparams(),
            output_model_name="my-model",
        )
        assert result.success is False
        assert "Unsupported model family" in result.error
        client.create_model_customization_job.assert_not_called()

    def test_api_exception(self):
        client = MagicMock()
        client.create_model_customization_job.side_effect = Exception("Access denied")
        result = create_bedrock_fine_tuning_job(
            bedrock_client=client,
            job_name="test-job",
            base_model_id="anthropic.claude-v2",
            training_data_s3_uri="s3://bucket/train.jsonl",
            output_s3_uri="s3://bucket/output/",
            role_arn="arn:aws:iam::123456:role/BedrockRole",
            hyperparameters=_default_hyperparams(),
            output_model_name="my-model",
        )
        assert result.success is False
        assert "Access denied" in result.error


# --- BedrockJobResult ---


class TestBedrockJobResult:
    def test_defaults(self):
        result = BedrockJobResult(success=True)
        assert result.success is True
        assert result.job_arn == ""
        assert result.error == ""
