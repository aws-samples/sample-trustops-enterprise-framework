"""
Automated hyperparameter tuning for fine-tuning.

Defines a search space for key hyperparameters, launches multiple
fine-tuning jobs with different configurations, tracks results,
and selects the best configuration based on validation loss.

Requirements: 4.6
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.data_models.fine_tuning import HyperparameterConfig
from src.fine_tuning.bedrock_trainer import (
    BedrockJobResult,
    create_bedrock_fine_tuning_job,
)
from src.fine_tuning.sagemaker_trainer import (
    SageMakerJobResult,
    create_sagemaker_training_job,
)


@dataclass
class HyperparameterRange:
    """Defines a range of values for a single hyperparameter.

    Attributes:
        values: Explicit list of values to try.
    """

    values: list[Any] = field(default_factory=list)


@dataclass
class HyperparameterSearchSpace:
    """Search space for automated hyperparameter tuning.

    Each field is an optional HyperparameterRange. When set,
    the tuner will try all listed values for that parameter.
    When None, the base config value is used.

    Attributes:
        learning_rate: Range of learning rates to try.
        epochs: Range of epoch counts to try.
        batch_size: Range of batch sizes to try.
        warmup_steps: Range of warmup step counts to try.
        weight_decay: Range of weight decay values to try.
    """

    learning_rate: Optional[HyperparameterRange] = None
    epochs: Optional[HyperparameterRange] = None
    batch_size: Optional[HyperparameterRange] = None
    warmup_steps: Optional[HyperparameterRange] = None
    weight_decay: Optional[HyperparameterRange] = None


@dataclass
class TuningTrialResult:
    """Result of a single hyperparameter tuning trial.

    Attributes:
        trial_id: Unique identifier for this trial.
        hyperparameters: The hyperparameter config used.
        job_arn: ARN of the launched job.
        validation_loss: Validation loss (None if not yet available).
        success: Whether the trial job was created successfully.
        error: Error message if the trial failed.
    """

    trial_id: str
    hyperparameters: HyperparameterConfig
    job_arn: str = ""
    validation_loss: Optional[float] = None
    success: bool = True
    error: str = ""


@dataclass
class TuningResult:
    """Overall result of a hyperparameter tuning run.

    Attributes:
        best_trial: The trial with the lowest validation loss.
        trials: All trial results.
        total_trials: Total number of trials attempted.
        completed_trials: Number of trials that completed.
        best_validation_loss: The lowest validation loss found.
    """

    best_trial: Optional[TuningTrialResult] = None
    trials: list[TuningTrialResult] = field(default_factory=list)
    total_trials: int = 0
    completed_trials: int = 0
    best_validation_loss: Optional[float] = None


def generate_configurations(
    base_config: HyperparameterConfig,
    search_space: HyperparameterSearchSpace,
) -> list[HyperparameterConfig]:
    """Generate all hyperparameter configurations from a search space.

    Produces the Cartesian product of all specified ranges. Parameters
    without a range keep their base config value.

    Args:
        base_config: The base hyperparameter configuration.
        search_space: The search space defining ranges to explore.

    Returns:
        List of HyperparameterConfig instances to evaluate.
    """
    param_names = [
        "learning_rate",
        "epochs",
        "batch_size",
        "warmup_steps",
        "weight_decay",
    ]

    # Build lists of values per parameter
    param_values: list[list[Any]] = []
    active_params: list[str] = []

    for name in param_names:
        hp_range: Optional[HyperparameterRange] = getattr(
            search_space, name, None
        )
        if hp_range is not None and hp_range.values:
            param_values.append(hp_range.values)
            active_params.append(name)

    if not active_params:
        return [base_config]

    # Cartesian product via iterative expansion
    combos: list[dict[str, Any]] = [{}]
    for param_name, values in zip(active_params, param_values):
        new_combos: list[dict[str, Any]] = []
        for combo in combos:
            for val in values:
                new_combo = dict(combo)
                new_combo[param_name] = val
                new_combos.append(new_combo)
        combos = new_combos

    # Build configs from combos
    base_dict = base_config.model_dump()
    configs: list[HyperparameterConfig] = []
    for combo in combos:
        merged = dict(base_dict)
        merged.update(combo)
        configs.append(HyperparameterConfig(**merged))

    return configs


def launch_tuning_trials_bedrock(
    bedrock_client: Any,
    base_model_id: str,
    training_data_s3_uri: str,
    output_s3_uri: str,
    role_arn: str,
    base_config: HyperparameterConfig,
    search_space: HyperparameterSearchSpace,
    job_name_prefix: str = "hp-tune",
    output_model_prefix: str = "hp-tune-model",
    validation_data_s3_uri: Optional[str] = None,
) -> list[TuningTrialResult]:
    """Launch multiple Bedrock fine-tuning jobs for hyperparameter tuning.

    Generates all configurations from the search space and creates
    a Bedrock fine-tuning job for each one.

    Args:
        bedrock_client: A boto3 Bedrock client instance.
        base_model_id: The base model identifier to fine-tune.
        training_data_s3_uri: S3 URI of the training data.
        output_s3_uri: S3 URI for output artifacts.
        role_arn: IAM role ARN for the jobs.
        base_config: Base hyperparameter configuration.
        search_space: Search space defining ranges to explore.
        job_name_prefix: Prefix for job names.
        output_model_prefix: Prefix for output model names.
        validation_data_s3_uri: Optional S3 URI for validation data.

    Returns:
        List of TuningTrialResult for each launched job.
    """
    configs = generate_configurations(base_config, search_space)
    trials: list[TuningTrialResult] = []

    for idx, config in enumerate(configs):
        trial_id = f"{job_name_prefix}-trial-{idx}"
        model_name = f"{output_model_prefix}-trial-{idx}"

        result: BedrockJobResult = create_bedrock_fine_tuning_job(
            bedrock_client=bedrock_client,
            job_name=trial_id,
            base_model_id=base_model_id,
            training_data_s3_uri=training_data_s3_uri,
            output_s3_uri=output_s3_uri,
            role_arn=role_arn,
            hyperparameters=config,
            output_model_name=model_name,
            validation_data_s3_uri=validation_data_s3_uri,
        )

        trials.append(
            TuningTrialResult(
                trial_id=trial_id,
                hyperparameters=config,
                job_arn=result.job_arn,
                success=result.success,
                error=result.error,
            )
        )

    return trials


def launch_tuning_trials_sagemaker(
    sagemaker_client: Any,
    container_image_uri: str,
    training_data_s3_uri: str,
    output_s3_uri: str,
    role_arn: str,
    base_config: HyperparameterConfig,
    search_space: HyperparameterSearchSpace,
    job_name_prefix: str = "hp-tune",
    instance_type: str = "ml.g5.2xlarge",
    validation_data_s3_uri: Optional[str] = None,
) -> list[TuningTrialResult]:
    """Launch multiple SageMaker training jobs for hyperparameter tuning.

    Generates all configurations from the search space and creates
    a SageMaker training job for each one.

    Args:
        sagemaker_client: A boto3 SageMaker client instance.
        container_image_uri: ECR URI of the training container.
        training_data_s3_uri: S3 URI of the training data.
        output_s3_uri: S3 URI for output model artifacts.
        role_arn: IAM role ARN for the jobs.
        base_config: Base hyperparameter configuration.
        search_space: Search space defining ranges to explore.
        job_name_prefix: Prefix for job names.
        instance_type: SageMaker instance type.
        validation_data_s3_uri: Optional S3 URI for validation data.

    Returns:
        List of TuningTrialResult for each launched job.
    """
    configs = generate_configurations(base_config, search_space)
    trials: list[TuningTrialResult] = []

    for idx, config in enumerate(configs):
        trial_id = f"{job_name_prefix}-trial-{idx}"

        result: SageMakerJobResult = create_sagemaker_training_job(
            sagemaker_client=sagemaker_client,
            job_name=trial_id,
            container_image_uri=container_image_uri,
            training_data_s3_uri=training_data_s3_uri,
            output_s3_uri=output_s3_uri,
            role_arn=role_arn,
            hyperparameters=config,
            instance_type=instance_type,
            validation_data_s3_uri=validation_data_s3_uri,
        )

        trials.append(
            TuningTrialResult(
                trial_id=trial_id,
                hyperparameters=config,
                job_arn=result.job_arn,
                success=result.success,
                error=result.error,
            )
        )

    return trials


def select_best_trial(
    trials: list[TuningTrialResult],
) -> TuningResult:
    """Select the best trial based on validation loss.

    Filters to successful trials that have a validation_loss set,
    then picks the one with the lowest validation loss.

    Args:
        trials: List of completed tuning trial results.

    Returns:
        TuningResult with the best trial and summary statistics.
    """
    completed = [
        t for t in trials
        if t.success and t.validation_loss is not None
    ]

    best: Optional[TuningTrialResult] = None
    best_loss: Optional[float] = None

    for trial in completed:
        # `completed` already filters out trials without a validation loss.
        loss = trial.validation_loss
        if best_loss is None or loss < best_loss:
            best = trial
            best_loss = loss

    return TuningResult(
        best_trial=best,
        trials=list(trials),
        total_trials=len(trials),
        completed_trials=len(completed),
        best_validation_loss=best_loss,
    )


def default_search_space() -> HyperparameterSearchSpace:
    """Return a sensible default search space for hyperparameter tuning.

    Covers a small grid of learning rates, epoch counts, and batch sizes
    that works well across model families.

    Returns:
        HyperparameterSearchSpace with default ranges.
    """
    return HyperparameterSearchSpace(
        learning_rate=HyperparameterRange(
            values=[1e-5, 5e-6, 2e-6]
        ),
        epochs=HyperparameterRange(
            values=[2, 3, 4]
        ),
        batch_size=HyperparameterRange(
            values=[4, 8]
        ),
    )
