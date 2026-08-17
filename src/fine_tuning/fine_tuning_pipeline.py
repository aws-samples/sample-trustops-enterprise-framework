"""
Fine-Tuning Pipeline for the TrustOps Enterprise Framework.

Integrates all fine-tuning components into a unified pipeline that
supports validate, estimate_cost, start, get_status, stop, and
list_jobs operations.

Requirements: 4.1-4.14
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.data_models.fine_tuning import (
    CostEstimate,
    FineTuningConfig,
    FineTuningJob,
    FineTuningStatus,
    HyperparameterConfig,
)
from src.fine_tuning.audit_logger import TrainingAuditLogger
from src.fine_tuning.cost_estimator import estimate_cost
from src.fine_tuning.dataset_size_validator import (
    DatasetSizeValidationResult,
    validate_dataset_size,
)
from src.fine_tuning.eligibility_checker import (
    EligibilityResult,
    check_fine_tuning_eligibility,
)
from src.fine_tuning.format_validator import (
    FormatValidationResult,
    validate_training_data_format,
)
from src.fine_tuning.hyperparameter_resolver import resolve_hyperparameters
from src.fine_tuning.job_completion_handler import (
    CompletionResult,
    handle_job_completion,
)
from src.fine_tuning.job_failure_handler import (
    FailureAnalysis,
    handle_job_failure,
)
from src.fine_tuning.job_resume import (
    ResumeResult,
    can_resume_job,
    resume_training_job,
)
from src.fine_tuning.progress_emitter import ProgressEmitter


@dataclass
class ValidationResult:
    """Result of validating a fine-tuning request.

    Attributes:
        is_valid: Whether all validation checks pass.
        eligibility: Model eligibility check result.
        format_validation: Training data format validation.
        size_validation: Dataset size validation result.
        errors: List of all validation errors.
    """

    is_valid: bool
    eligibility: Optional[EligibilityResult] = None
    format_validation: Optional[FormatValidationResult] = None
    size_validation: Optional[DatasetSizeValidationResult] = None
    errors: list[str] = field(default_factory=list)


class FineTuningPipeline:
    """Manages model fine-tuning across providers.

    Integrates eligibility checking, format validation, dataset
    size validation, cost estimation, job creation, status polling,
    completion/failure handling, progress events, and audit logging.

    Usage::

        pipeline = FineTuningPipeline()
        validation = pipeline.validate(config, records=records)
        if validation.is_valid:
            cost = pipeline.estimate_cost(config, num_examples=1000)
            job = pipeline.start(config, provider_client=client)
            status = pipeline.get_status(job.job_id)
    """

    def __init__(
        self,
        model_registry: Any = None,
        dataset_loader: Any = None,
        audit_logger: Optional[TrainingAuditLogger] = None,
    ) -> None:
        self._model_registry = model_registry
        self._dataset_loader = dataset_loader
        self._audit_logger = audit_logger or TrainingAuditLogger()
        self._jobs: dict[str, FineTuningJob] = {}
        self._emitters: dict[str, ProgressEmitter] = {}

    @property
    def audit_logger(self) -> TrainingAuditLogger:
        return self._audit_logger

    @property
    def jobs(self) -> dict[str, FineTuningJob]:
        """Return a copy of the jobs dict."""
        return dict(self._jobs)

    def validate(
        self,
        config: FineTuningConfig,
        model_metadata: Any = None,
        training_records: Optional[list[dict[str, Any]]] = None,
    ) -> ValidationResult:
        """Validate a fine-tuning request.

        Args:
            config: The fine-tuning configuration.
            model_metadata: Model metadata for eligibility check.
            training_records: Parsed training data records.

        Returns:
            ValidationResult with all check results.
        """
        errors: list[str] = []
        eligibility = None
        format_result = None
        size_result = None

        if model_metadata is not None:
            eligibility = check_fine_tuning_eligibility(model_metadata)
            if not eligibility.is_eligible:
                errors.extend(eligibility.reasons)

        if training_records is not None:
            format_result = validate_training_data_format(
                training_records, config.base_model_id
            )
            if not format_result.is_valid:
                errors.extend(format_result.issues)

            size_result = validate_dataset_size(
                training_records, config.base_model_id
            )
            if not size_result.is_valid:
                errors.append(size_result.reason)

        is_valid = len(errors) == 0

        self._audit_logger.log_validation(
            job_id=config.job_name,
            success=is_valid,
            parameters={
                "base_model_id": config.base_model_id,
                "training_data_id": config.training_data_id,
            },
            error="; ".join(errors) if errors else "",
        )

        return ValidationResult(
            is_valid=is_valid,
            eligibility=eligibility,
            format_validation=format_result,
            size_validation=size_result,
            errors=errors,
        )

    def estimate_cost(
        self,
        config: FineTuningConfig,
        num_examples: int,
        token_count: Optional[int] = None,
    ) -> CostEstimate:
        """Estimate fine-tuning cost before job submission.

        Args:
            config: The fine-tuning configuration.
            num_examples: Number of training examples.
            token_count: Optional total token count.

        Returns:
            CostEstimate with breakdown and confidence.
        """
        result = estimate_cost(
            model_id=config.base_model_id,
            num_examples=num_examples,
            hyperparameters=config.hyperparameters,
            token_count=token_count,
        )

        self._audit_logger.log_cost_estimated(
            job_id=config.job_name,
            parameters={
                "estimated_cost": result.estimated_training_cost,
                "num_examples": num_examples,
                "epochs": config.hyperparameters.epochs,
            },
        )

        return result

    def get_default_hyperparameters(
        self,
        model_id: str,
        dataset_size: int,
    ) -> HyperparameterConfig:
        """Get intelligent default hyperparameters.

        Args:
            model_id: The target model identifier.
            dataset_size: Number of training examples.

        Returns:
            HyperparameterConfig with resolved defaults.
        """
        return resolve_hyperparameters(model_id, dataset_size)

    @staticmethod
    def _default_training_uri(training_data_id: str) -> str:
        """Build the training-data URI from the configured datasets bucket.

        The previous fallback hardcoded a bucket literally named "data". Bucket
        names are globally unique across AWS, so that name is both maximally
        predictable and already owned by someone else: this URI is handed to
        Bedrock as the training source, meaning a squatted bucket would train
        the model on data an attacker controls. Deriving the name from
        configuration keeps the
        convenience of an implicit URI while inheriting the fail-closed
        behaviour of ``config.datasets_bucket``, which raises when unset rather
        than guessing a name.

        Args:
            training_data_id: Dataset identifier to locate under the bucket.

        Returns:
            The s3:// URI of the training data.

        Raises:
            BucketNotConfiguredError: If TRUSTOPS_DATASETS_BUCKET is unset.
        """
        from config.aws_config import config as aws_config

        return f"s3://{aws_config.datasets_bucket}/datasets/{training_data_id}"

    def start(
        self,
        config: FineTuningConfig,
        provider_client: Any = None,
        role_arn: str = "",
        output_s3_uri: str = "",
        training_data_s3_uri: str = "",
        validation_data_s3_uri: Optional[str] = None,
    ) -> FineTuningJob:
        """Start a fine-tuning job.

        Args:
            config: The fine-tuning configuration.
            provider_client: Provider client for job submission.
            role_arn: IAM role ARN for the job.
            output_s3_uri: S3 URI for output artifacts.
            training_data_s3_uri: S3 URI of training data. When omitted it is
                derived from the configured datasets bucket; see
                _default_training_uri.
            validation_data_s3_uri: Optional validation data URI.

        Returns:
            FineTuningJob with initial status.
        """
        job_id = f"ft-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        s3_uri = training_data_s3_uri or self._default_training_uri(
            config.training_data_id
        )

        job = FineTuningJob(
            job_id=job_id,
            status=FineTuningStatus.PENDING,
            base_model_id=config.base_model_id,
            training_data_s3_uri=s3_uri,
            validation_data_s3_uri=validation_data_s3_uri,
            hyperparameters=config.hyperparameters,
            estimated_cost=0.0,
            created_at=now,
        )

        self._jobs[job_id] = job

        emitter = ProgressEmitter(
            job_id=job_id,
            total_epochs=config.hyperparameters.epochs,
        )
        self._emitters[job_id] = emitter

        self._audit_logger.log_job_created(
            job_id=job_id,
            parameters={
                "base_model_id": config.base_model_id,
                "training_data_id": config.training_data_id,
                "job_name": config.job_name,
            },
        )

        if provider_client is not None:
            self._submit_to_provider(
                job, config, provider_client,
                role_arn, output_s3_uri,
                s3_uri, validation_data_s3_uri,
            )

        return job

    def _submit_to_provider(
        self,
        job: FineTuningJob,
        config: FineTuningConfig,
        provider_client: Any,
        role_arn: str,
        output_s3_uri: str,
        training_s3_uri: str,
        validation_s3_uri: Optional[str],
    ) -> None:
        """Submit a job to the provider."""
        from src.fine_tuning.bedrock_trainer import (
            create_bedrock_fine_tuning_job,
        )

        try:
            result = create_bedrock_fine_tuning_job(
                bedrock_client=provider_client,
                job_name=config.job_name,
                base_model_id=config.base_model_id,
                training_data_s3_uri=training_s3_uri,
                output_s3_uri=output_s3_uri,
                role_arn=role_arn,
                hyperparameters=config.hyperparameters,
                output_model_name=config.output_model_name,
                validation_data_s3_uri=validation_s3_uri,
            )

            if result.success:
                job.status = FineTuningStatus.TRAINING
                job.provider_job_id = result.job_arn
                job.started_at = datetime.now(timezone.utc)
                self._audit_logger.log_job_started(job.job_id)
                emitter = self._emitters.get(job.job_id)
                if emitter:
                    emitter.start()
            else:
                job.status = FineTuningStatus.FAILED
                job.error_message = result.error
                self._audit_logger.log_job_failed(
                    job.job_id, result.error
                )
        except Exception as e:
            job.status = FineTuningStatus.FAILED
            job.error_message = str(e)
            self._audit_logger.log_job_failed(
                job.job_id, str(e)
            )

    def get_status(
        self, job_id: str,
    ) -> Optional[FineTuningJob]:
        """Get current status of a fine-tuning job.

        Args:
            job_id: The job ID to query.

        Returns:
            The FineTuningJob if found, None otherwise.
        """
        return self._jobs.get(job_id)

    def stop(
        self, job_id: str, reason: str = "",
    ) -> Optional[FineTuningJob]:
        """Stop a running fine-tuning job.

        Args:
            job_id: The job ID to stop.
            reason: Reason for stopping.

        Returns:
            Updated FineTuningJob if found, None otherwise.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None

        if job.status != FineTuningStatus.TRAINING:
            return job

        job.status = FineTuningStatus.STOPPED
        job.completed_at = datetime.now(timezone.utc)

        self._audit_logger.log_job_stopped(job_id, reason)

        emitter = self._emitters.get(job_id)
        if emitter:
            emitter.stop(reason)

        return job

    def list_jobs(
        self,
        status: Optional[FineTuningStatus] = None,
        model_id: Optional[str] = None,
    ) -> list[FineTuningJob]:
        """List fine-tuning jobs with optional filtering.

        Args:
            status: Filter by job status.
            model_id: Filter by base model ID.

        Returns:
            List of matching FineTuningJob instances.
        """
        jobs = list(self._jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        if model_id is not None:
            jobs = [j for j in jobs if j.base_model_id == model_id]
        return jobs

    def handle_completion(
        self, job_id: str,
    ) -> Optional[CompletionResult]:
        """Handle a completed fine-tuning job.

        Args:
            job_id: The completed job ID.

        Returns:
            CompletionResult if job found, None otherwise.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None

        result = handle_job_completion(job, self._model_registry)

        if result.success:
            self._audit_logger.log_job_completed(
                job_id,
                parameters={
                    "finetuned_model_id": result.finetuned_model_id,
                    "finetuned_model_arn": result.finetuned_model_arn,
                },
            )
            self._audit_logger.log_model_registered(
                job_id,
                parameters=result.training_metadata,
            )

            emitter = self._emitters.get(job_id)
            if emitter:
                loss = (
                    result.training_metadata.get(
                        "final_training_loss"
                    )
                    or 0.0
                )
                val = result.training_metadata.get(
                    "final_validation_loss"
                )
                emitter.complete(loss, val)

        return result

    def handle_failure(
        self, job_id: str,
    ) -> Optional[FailureAnalysis]:
        """Handle a failed fine-tuning job.

        Args:
            job_id: The failed job ID.

        Returns:
            FailureAnalysis if job found, None otherwise.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None

        analysis = handle_job_failure(job)

        self._audit_logger.log_job_failed(
            job_id,
            error=analysis.error_message,
            parameters={"category": analysis.error_category},
        )

        emitter = self._emitters.get(job_id)
        if emitter:
            emitter.fail(analysis.error_message)

        return analysis

    def resume_job(
        self,
        job_id: str,
        provider_client: Any = None,
        role_arn: str = "",
        output_s3_uri: str = "",
    ) -> Optional[ResumeResult]:
        """Resume an interrupted fine-tuning job.

        Args:
            job_id: The job ID to resume.
            provider_client: Provider client for submission.
            role_arn: IAM role ARN.
            output_s3_uri: S3 URI for output.

        Returns:
            ResumeResult if job found, None otherwise.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None

        if provider_client is None:
            can, reason = can_resume_job(job)
            return ResumeResult(
                success=False,
                resumed_from_job_id=job_id,
                error=(
                    "No provider client provided"
                    if can
                    else reason
                ),
            )

        return resume_training_job(
            job, provider_client, role_arn, output_s3_uri
        )

    def get_progress_emitter(
        self, job_id: str,
    ) -> Optional[ProgressEmitter]:
        """Get the progress emitter for a job.

        Args:
            job_id: The job ID.

        Returns:
            ProgressEmitter if found, None otherwise.
        """
        return self._emitters.get(job_id)
