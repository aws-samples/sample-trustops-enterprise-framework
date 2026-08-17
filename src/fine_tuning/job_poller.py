"""
Job status poller for fine-tuning jobs.

Periodically checks job status via provider API (Bedrock or SageMaker)
and extracts training metrics (loss, validation_loss, learning_rate).

Requirements: 4.7, 4.11
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.data_models.fine_tuning import FineTuningStatus, TrainingMetrics


# --- Bedrock status mapping ---

_BEDROCK_STATUS_MAP: dict[str, FineTuningStatus] = {
    "InProgress": FineTuningStatus.TRAINING,
    "Completed": FineTuningStatus.COMPLETED,
    "Failed": FineTuningStatus.FAILED,
    "Stopping": FineTuningStatus.STOPPED,
    "Stopped": FineTuningStatus.STOPPED,
}

# --- SageMaker status mapping ---

_SAGEMAKER_STATUS_MAP: dict[str, FineTuningStatus] = {
    "InProgress": FineTuningStatus.TRAINING,
    "Completed": FineTuningStatus.COMPLETED,
    "Failed": FineTuningStatus.FAILED,
    "Stopping": FineTuningStatus.STOPPED,
    "Stopped": FineTuningStatus.STOPPED,
}


@dataclass
class PollResult:
    """Result of polling a fine-tuning job's status.

    Attributes:
        success: Whether the poll request succeeded.
        status: The current FineTuningStatus of the job.
        metrics: Training metrics extracted from the job, if available.
        finetuned_model_arn: ARN of the fine-tuned model on completion.
        error: Error message if the poll failed or the job failed.
    """

    success: bool
    status: FineTuningStatus = FineTuningStatus.PENDING
    metrics: list[TrainingMetrics] = field(default_factory=list)
    finetuned_model_arn: str = ""
    error: str = ""


def _parse_bedrock_metrics(response: dict[str, Any]) -> list[TrainingMetrics]:
    """Extract training metrics from a Bedrock job response.

    Bedrock returns training metrics under the 'trainingMetrics' key
    as a list of metric snapshots.
    """
    raw_metrics = response.get("trainingMetrics", [])
    if not raw_metrics:
        return []

    parsed: list[TrainingMetrics] = []
    for entry in raw_metrics:
        try:
            loss = float(entry.get("trainingLoss", 0.0))
            lr = float(entry.get("learningRate", 1e-5))
            val_loss = _safe_float(
                entry.get("validationLoss")
            )
            parsed.append(
                TrainingMetrics(
                    epoch=int(entry.get("epoch", 0)),
                    step=int(entry.get("step", 0)),
                    training_loss=loss,
                    validation_loss=val_loss,
                    learning_rate=lr,
                    timestamp=datetime.now(timezone.utc),
                )
            )
        except (ValueError, TypeError):
            continue

    return parsed


def _parse_sagemaker_metrics(
    response: dict[str, Any],
) -> list[TrainingMetrics]:
    """Extract training metrics from a SageMaker job response.

    SageMaker returns final metrics under 'FinalMetricDataList'.
    """
    raw_metrics = response.get("FinalMetricDataList", [])
    if not raw_metrics:
        return []

    # SageMaker returns flat metric entries; group into a single snapshot
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    learning_rate: float = 1e-5

    for entry in raw_metrics:
        name = entry.get("MetricName", "")
        value = entry.get("Value")
        if value is None:
            continue
        if "train" in name.lower() and "loss" in name.lower():
            training_loss = float(value)
        elif "val" in name.lower() and "loss" in name.lower():
            validation_loss = float(value)
        elif "learning_rate" in name.lower() or "lr" in name.lower():
            learning_rate = float(value)

    if training_loss is None:
        return []

    return [
        TrainingMetrics(
            epoch=0,
            step=0,
            training_loss=training_loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            timestamp=datetime.now(timezone.utc),
        )
    ]


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def poll_bedrock_job(
    bedrock_client: Any,
    job_name: str,
) -> PollResult:
    """Poll the status of a Bedrock model customization job.

    Calls get_model_customization_job() and maps the provider status
    to FineTuningStatus, extracting any available training metrics.

    Args:
        bedrock_client: A boto3 Bedrock client instance.
        job_name: The name or ARN of the customization job.

    Returns:
        PollResult with current status, metrics, and model ARN if complete.
    """
    try:
        response = bedrock_client.get_model_customization_job(
            jobIdentifier=job_name
        )
    except Exception as e:
        return PollResult(
            success=False,
            error=(
                f"Failed to poll Bedrock job '{job_name}': {e}"
            ),
        )

    raw_status = response.get("status", "")
    status = _BEDROCK_STATUS_MAP.get(raw_status, FineTuningStatus.PENDING)
    metrics = _parse_bedrock_metrics(response)
    model_arn = response.get("outputModelArn", "")
    error_msg = response.get("failureMessage", "")

    return PollResult(
        success=True,
        status=status,
        metrics=metrics,
        finetuned_model_arn=model_arn,
        error=error_msg,
    )


def poll_sagemaker_job(
    sagemaker_client: Any,
    job_name: str,
) -> PollResult:
    """Poll the status of a SageMaker training job.

    Calls describe_training_job() and maps the provider status
    to FineTuningStatus, extracting any available training metrics.

    Args:
        sagemaker_client: A boto3 SageMaker client instance.
        job_name: The name of the training job.

    Returns:
        PollResult with current status, metrics, and model ARN if complete.
    """
    try:
        response = sagemaker_client.describe_training_job(
            TrainingJobName=job_name
        )
    except Exception as e:
        return PollResult(
            success=False,
            error=(
                f"Failed to poll SageMaker job"
                f" '{job_name}': {e}"
            ),
        )

    raw_status = response.get("TrainingJobStatus", "")
    status = _SAGEMAKER_STATUS_MAP.get(raw_status, FineTuningStatus.PENDING)
    metrics = _parse_sagemaker_metrics(response)
    model_arn = response.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
    error_msg = response.get("FailureReason", "")

    return PollResult(
        success=True,
        status=status,
        metrics=metrics,
        finetuned_model_arn=model_arn,
        error=error_msg,
    )
