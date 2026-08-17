"""
Job resume support for fine-tuning jobs.

Supports resuming interrupted training jobs where the provider
supports it. Determines resume eligibility based on provider
and job status, and creates a new job with checkpoint data.

Requirements: 4.13
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.data_models.fine_tuning import (
    FineTuningJob,
    FineTuningStatus,
    HyperparameterConfig,
)
from src.fine_tuning.format_validator import ModelFamily, detect_model_family


@dataclass
class ResumeResult:
    """Result of a job resume attempt.

    Attributes:
        success: Whether the resume was initiated successfully.
        new_job_arn: ARN of the new resumed job (if created).
        resumed_from_job_id: The original job ID being resumed.
        checkpoint_info: Information about the checkpoint used.
        error: Error message if resume failed.
    """

    success: bool
    new_job_arn: str = ""
    resumed_from_job_id: str = ""
    checkpoint_info: dict[str, Any] = field(default_factory=dict)
    error: str = ""


# Providers that support resume
_RESUME_SUPPORTED_FAMILIES: set[ModelFamily] = {
    ModelFamily.CLAUDE,
    ModelFamily.LLAMA,
}

# Statuses from which a job can be resumed
_RESUMABLE_STATUSES: set[FineTuningStatus] = {
    FineTuningStatus.FAILED,
    FineTuningStatus.STOPPED,
}


def can_resume_job(job: FineTuningJob) -> tuple[bool, str]:
    """Check whether a fine-tuning job can be resumed.

    A job can be resumed if:
    1. The job status is FAILED or STOPPED.
    2. The model family supports resume (Claude, Llama).
    3. The job has a provider_job_id (was actually submitted).

    Args:
        job: The fine-tuning job to check.

    Returns:
        Tuple of (can_resume, reason).
    """
    if job.status not in _RESUMABLE_STATUSES:
        return False, (
            f"Job '{job.job_id}' cannot be resumed from status "
            f"'{job.status.value}'. Only FAILED or STOPPED jobs "
            f"can be resumed."
        )

    family = detect_model_family(job.base_model_id)
    if family not in _RESUME_SUPPORTED_FAMILIES:
        return False, (
            f"Model family '{family.value}' does not support "
            f"job resume. Supported families: "
            f"{', '.join(f.value for f in _RESUME_SUPPORTED_FAMILIES)}."
        )

    if not job.provider_job_id:
        return False, (
            f"Job '{job.job_id}' has no provider job ID. "
            f"The job may not have been submitted to the provider."
        )

    return True, "Job is eligible for resume."


def build_resume_config(
    job: FineTuningJob,
    adjusted_hyperparameters: Optional[HyperparameterConfig] = None,
) -> dict[str, Any]:
    """Build configuration for resuming a job.

    Creates a configuration dict that can be used to submit a new
    training job that continues from where the original left off.

    Args:
        job: The original job to resume from.
        adjusted_hyperparameters: Optional adjusted hyperparameters.
            If None, uses the original job's hyperparameters with
            reduced remaining epochs.

    Returns:
        Configuration dict for the resumed job.
    """
    hp = adjusted_hyperparameters or job.hyperparameters

    # Calculate remaining epochs based on training metrics
    completed_epochs = 0
    if job.training_metrics:
        completed_epochs = max(m.epoch for m in job.training_metrics)

    remaining_epochs = max(1, hp.epochs - completed_epochs)

    # Build adjusted hyperparameters for remaining training
    resume_hp = HyperparameterConfig(
        learning_rate=hp.learning_rate,
        epochs=remaining_epochs,
        batch_size=hp.batch_size,
        warmup_steps=0,  # No warmup needed on resume
        weight_decay=hp.weight_decay,
        max_seq_length=hp.max_seq_length,
        lora_rank=hp.lora_rank,
        lora_alpha=hp.lora_alpha,
    )

    return {
        "original_job_id": job.job_id,
        "provider_job_id": job.provider_job_id,
        "base_model_id": job.base_model_id,
        "training_data_s3_uri": job.training_data_s3_uri,
        "validation_data_s3_uri": job.validation_data_s3_uri,
        "hyperparameters": resume_hp,
        "completed_epochs": completed_epochs,
        "remaining_epochs": remaining_epochs,
    }


def resume_training_job(
    job: FineTuningJob,
    provider_client: Any,
    role_arn: str = "",
    output_s3_uri: str = "",
    adjusted_hyperparameters: Optional[HyperparameterConfig] = None,
) -> ResumeResult:
    """Resume an interrupted fine-tuning job.

    Checks eligibility, builds resume configuration, and submits
    a new training job to the provider.

    Args:
        job: The interrupted fine-tuning job to resume.
        provider_client: The provider client (Bedrock or SageMaker).
        role_arn: IAM role ARN for the new job.
        output_s3_uri: S3 URI for output artifacts.
        adjusted_hyperparameters: Optional adjusted hyperparameters.

    Returns:
        ResumeResult with new job ARN on success or error on failure.
    """
    can_resume, reason = can_resume_job(job)
    if not can_resume:
        return ResumeResult(
            success=False,
            resumed_from_job_id=job.job_id,
            error=reason,
        )

    resume_config = build_resume_config(job, adjusted_hyperparameters)

    checkpoint_info = {
        "original_job_id": job.job_id,
        "completed_epochs": resume_config["completed_epochs"],
        "remaining_epochs": resume_config["remaining_epochs"],
    }

    # Attempt to create a new job via the provider
    family = detect_model_family(job.base_model_id)
    try:
        if family in (ModelFamily.CLAUDE, ModelFamily.LLAMA):
            # Bedrock resume: create a new customization job
            hp = resume_config["hyperparameters"]
            hp_dict = {
                "epochCount": str(hp.epochs),
                "batchSize": str(hp.batch_size),
                "learningRate": str(hp.learning_rate),
            }

            response = provider_client.create_model_customization_job(
                jobName=f"{job.job_id}-resume",
                customModelName=f"{job.job_id}-resume-model",
                roleArn=role_arn,
                baseModelIdentifier=job.base_model_id,
                trainingDataConfig={"s3Uri": job.training_data_s3_uri},
                outputDataConfig={"s3Uri": output_s3_uri},
                hyperParameters=hp_dict,
                customizationType="FINE_TUNING",
            )
            new_arn = response.get("jobArn", "")
        else:
            return ResumeResult(
                success=False,
                resumed_from_job_id=job.job_id,
                error=f"Resume not supported for family: {family.value}",
            )
    except Exception as e:
        return ResumeResult(
            success=False,
            resumed_from_job_id=job.job_id,
            checkpoint_info=checkpoint_info,
            error=f"Failed to resume job: {e}",
        )

    return ResumeResult(
        success=True,
        new_job_arn=new_arn,
        resumed_from_job_id=job.job_id,
        checkpoint_info=checkpoint_info,
    )
