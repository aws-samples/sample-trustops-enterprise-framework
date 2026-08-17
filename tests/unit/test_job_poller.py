"""
Unit tests for the fine-tuning job status poller.

Tests cover Bedrock and SageMaker job polling, status mapping,
training metrics extraction, and error handling with mocked clients.

Requirements: 4.7, 4.11
"""

from unittest.mock import MagicMock

from src.data_models.fine_tuning import FineTuningStatus
from src.fine_tuning.job_poller import (
    PollResult,
    poll_bedrock_job,
    poll_sagemaker_job,
)


# --- Helpers ---


def _mock_bedrock_client(response: dict) -> MagicMock:
    client = MagicMock()
    client.get_model_customization_job.return_value = response
    return client


def _mock_sagemaker_client(response: dict) -> MagicMock:
    client = MagicMock()
    client.describe_training_job.return_value = response
    return client


# --- Bedrock polling ---


class TestPollBedrockJob:
    def test_training_in_progress(self):
        client = _mock_bedrock_client({"status": "InProgress"})
        result = poll_bedrock_job(client, "my-job")
        assert result.success is True
        assert result.status == FineTuningStatus.TRAINING

    def test_completed_with_model_arn(self):
        model_arn = (
            "arn:aws:bedrock:us-east-1:123:custom-model/my-model"
        )
        client = _mock_bedrock_client({
            "status": "Completed",
            "outputModelArn": model_arn,
        })
        result = poll_bedrock_job(client, "my-job")
        assert result.success is True
        assert result.status == FineTuningStatus.COMPLETED
        assert result.finetuned_model_arn == model_arn

    def test_failed_with_message(self):
        client = _mock_bedrock_client({
            "status": "Failed",
            "failureMessage": "Insufficient training data",
        })
        result = poll_bedrock_job(client, "my-job")
        assert result.success is True
        assert result.status == FineTuningStatus.FAILED
        assert "Insufficient training data" in result.error

    def test_stopped_status(self):
        client = _mock_bedrock_client({"status": "Stopped"})
        result = poll_bedrock_job(client, "my-job")
        assert result.success is True
        assert result.status == FineTuningStatus.STOPPED

    def test_stopping_maps_to_stopped(self):
        client = _mock_bedrock_client({"status": "Stopping"})
        result = poll_bedrock_job(client, "my-job")
        assert result.status == FineTuningStatus.STOPPED

    def test_unknown_status_defaults_to_pending(self):
        client = _mock_bedrock_client({"status": "SomeNewStatus"})
        result = poll_bedrock_job(client, "my-job")
        assert result.success is True
        assert result.status == FineTuningStatus.PENDING

    def test_extracts_training_metrics(self):
        client = _mock_bedrock_client({
            "status": "InProgress",
            "trainingMetrics": [
                {
                    "epoch": 1,
                    "step": 100,
                    "trainingLoss": 0.45,
                    "validationLoss": 0.52,
                    "learningRate": 2e-5,
                },
                {
                    "epoch": 2,
                    "step": 200,
                    "trainingLoss": 0.30,
                    "learningRate": 1e-5,
                },
            ],
        })
        result = poll_bedrock_job(client, "my-job")
        assert result.success is True
        assert len(result.metrics) == 2
        assert result.metrics[0].training_loss == 0.45
        assert result.metrics[0].validation_loss == 0.52
        assert result.metrics[0].learning_rate == 2e-5
        assert result.metrics[0].epoch == 1
        assert result.metrics[0].step == 100
        assert result.metrics[1].validation_loss is None

    def test_no_metrics_returns_empty_list(self):
        client = _mock_bedrock_client({"status": "InProgress"})
        result = poll_bedrock_job(client, "my-job")
        assert result.metrics == []

    def test_api_exception(self):
        client = MagicMock()
        client.get_model_customization_job.side_effect = (
            Exception("Access denied")
        )
        result = poll_bedrock_job(client, "my-job")
        assert result.success is False
        assert "Access denied" in result.error

    def test_calls_correct_api(self):
        client = _mock_bedrock_client({"status": "InProgress"})
        poll_bedrock_job(client, "my-job-arn")
        client.get_model_customization_job.assert_called_once_with(
            jobIdentifier="my-job-arn"
        )


# --- SageMaker polling ---


class TestPollSageMakerJob:
    def test_training_in_progress(self):
        client = _mock_sagemaker_client({"TrainingJobStatus": "InProgress"})
        result = poll_sagemaker_job(client, "my-sm-job")
        assert result.success is True
        assert result.status == FineTuningStatus.TRAINING

    def test_completed_with_model_artifacts(self):
        s3_path = "s3://bucket/output/model.tar.gz"
        client = _mock_sagemaker_client({
            "TrainingJobStatus": "Completed",
            "ModelArtifacts": {
                "S3ModelArtifacts": s3_path,
            },
        })
        result = poll_sagemaker_job(client, "my-sm-job")
        assert result.success is True
        assert result.status == FineTuningStatus.COMPLETED
        assert result.finetuned_model_arn == s3_path

    def test_failed_with_reason(self):
        client = _mock_sagemaker_client({
            "TrainingJobStatus": "Failed",
            "FailureReason": "ResourceLimitExceeded",
        })
        result = poll_sagemaker_job(client, "my-sm-job")
        assert result.success is True
        assert result.status == FineTuningStatus.FAILED
        assert "ResourceLimitExceeded" in result.error

    def test_stopped_status(self):
        client = _mock_sagemaker_client({"TrainingJobStatus": "Stopped"})
        result = poll_sagemaker_job(client, "my-sm-job")
        assert result.status == FineTuningStatus.STOPPED

    def test_unknown_status_defaults_to_pending(self):
        client = _mock_sagemaker_client({"TrainingJobStatus": "Unknown"})
        result = poll_sagemaker_job(client, "my-sm-job")
        assert result.status == FineTuningStatus.PENDING

    def test_extracts_final_metrics(self):
        client = _mock_sagemaker_client({
            "TrainingJobStatus": "Completed",
            "FinalMetricDataList": [
                {"MetricName": "train:loss", "Value": 0.35},
                {"MetricName": "validation:loss", "Value": 0.42},
                {"MetricName": "learning_rate", "Value": 5e-6},
            ],
        })
        result = poll_sagemaker_job(client, "my-sm-job")
        assert len(result.metrics) == 1
        assert result.metrics[0].training_loss == 0.35
        assert result.metrics[0].validation_loss == 0.42
        assert result.metrics[0].learning_rate == 5e-6

    def test_no_metrics_returns_empty_list(self):
        client = _mock_sagemaker_client({"TrainingJobStatus": "InProgress"})
        result = poll_sagemaker_job(client, "my-sm-job")
        assert result.metrics == []

    def test_api_exception(self):
        client = MagicMock()
        client.describe_training_job.side_effect = Exception("Throttling")
        result = poll_sagemaker_job(client, "my-sm-job")
        assert result.success is False
        assert "Throttling" in result.error

    def test_calls_correct_api(self):
        client = _mock_sagemaker_client({"TrainingJobStatus": "InProgress"})
        poll_sagemaker_job(client, "my-sm-job")
        client.describe_training_job.assert_called_once_with(
            TrainingJobName="my-sm-job"
        )


# --- PollResult defaults ---


class TestPollResult:
    def test_defaults(self):
        result = PollResult(success=True)
        assert result.success is True
        assert result.status == FineTuningStatus.PENDING
        assert result.metrics == []
        assert result.finetuned_model_arn == ""
        assert result.error == ""

    def test_failure_result(self):
        result = PollResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
