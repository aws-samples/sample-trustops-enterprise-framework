"""
Job completion handler for fine-tuning jobs.

Extracts the fine-tuned model ARN from a completed job and registers
it in the Model Registry with training metadata (base_model_id,
training_data_id, hyperparameters).

Requirements: 4.9
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.data_models.fine_tuning import (
    FineTuningJob,
    FineTuningStatus,
    HyperparameterConfig,
)


@dataclass
class CompletionResult:
    """Result of handling a completed fine-tuning job.

    Attributes:
        success: Whether the completion was handled successfully.
        finetuned_model_id: The registered model ID for the fine-tuned model.
        finetuned_model_arn: The ARN of the fine-tuned model.
        training_metadata: Metadata recorded for the training run.
        error: Error message if handling failed.
    """

    success: bool
    finetuned_model_id: str = ""
    finetuned_model_arn: str = ""
    training_metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def _build_training_metadata(
    job: FineTuningJob,
    finetuned_model_arn: str,
) -> dict[str, Any]:
    """Build training metadata dict for Model Registry registration.

    Args:
        job: The completed fine-tuning job.
        finetuned_model_arn: ARN of the fine-tuned model.

    Returns:
        Dictionary of training metadata.
    """
    hp = job.hyperparameters
    return {
        "base_model_id": job.base_model_id,
        "training_data_id": job.training_data_s3_uri,
        "validation_data_id": job.validation_data_s3_uri or "",
        "hyperparameters": {
            "learning_rate": hp.learning_rate,
            "epochs": hp.epochs,
            "batch_size": hp.batch_size,
            "warmup_steps": hp.warmup_steps,
            "weight_decay": hp.weight_decay,
            "max_seq_length": hp.max_seq_length,
            "lora_rank": hp.lora_rank,
            "lora_alpha": hp.lora_alpha,
        },
        "finetuned_model_arn": finetuned_model_arn,
        "job_id": job.job_id,
        "estimated_cost": job.estimated_cost,
        "actual_cost": job.actual_cost,
        "created_at": job.created_at.isoformat(),
        "completed_at": (
            job.completed_at.isoformat() if job.completed_at else None
        ),
        "final_training_loss": (
            job.training_metrics[-1].training_loss
            if job.training_metrics
            else None
        ),
        "final_validation_loss": (
            job.training_metrics[-1].validation_loss
            if job.training_metrics
            else None
        ),
    }


def extract_model_arn_from_job(job: FineTuningJob) -> Optional[str]:
    """Extract the fine-tuned model ARN from a completed job.

    Args:
        job: The fine-tuning job to extract the ARN from.

    Returns:
        The model ARN string, or None if not available.
    """
    if job.finetuned_model_arn:
        return job.finetuned_model_arn
    return None


def handle_job_completion(
    job: FineTuningJob,
    model_registry: Any = None,
) -> CompletionResult:
    """Handle a completed fine-tuning job.

    Extracts the fine-tuned model ARN, builds training metadata,
    and optionally registers the model in the Model Registry.

    Args:
        job: The completed fine-tuning job.
        model_registry: Optional Model Registry instance for registration.
            If None, metadata is built but registration is skipped.

    Returns:
        CompletionResult with model ID, ARN, and training metadata.
    """
    if job.status != FineTuningStatus.COMPLETED:
        return CompletionResult(
            success=False,
            error=(
                f"Job '{job.job_id}' is not completed "
                f"(status: {job.status.value})"
            ),
        )

    model_arn = extract_model_arn_from_job(job)
    if not model_arn:
        return CompletionResult(
            success=False,
            error=(
                f"Job '{job.job_id}' completed but no fine-tuned "
                f"model ARN is available"
            ),
        )

    training_metadata = _build_training_metadata(job, model_arn)

    # Generate a model ID from the job
    model_id = job.finetuned_model_id or f"ft-{job.job_id}"

    # Register in Model Registry if provided
    if model_registry is not None:
        try:
            model_registry.register_fine_tuned_model(
                model_id=model_id,
                model_arn=model_arn,
                base_model_id=job.base_model_id,
                training_metadata=training_metadata,
            )
        except Exception as e:
            return CompletionResult(
                success=False,
                finetuned_model_id=model_id,
                finetuned_model_arn=model_arn,
                training_metadata=training_metadata,
                error=f"Failed to register model in registry: {e}",
            )

    return CompletionResult(
        success=True,
        finetuned_model_id=model_id,
        finetuned_model_arn=model_arn,
        training_metadata=training_metadata,
    )
