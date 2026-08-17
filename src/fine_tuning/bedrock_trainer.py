"""
Bedrock fine-tuning job creator.

Creates model customization jobs on AWS Bedrock for supported model
families (Claude, Titan, Llama). Handles job creation parameters
per model family and extracts the job ARN from the response.

Requirements: 4.3, 4.7
"""

from dataclasses import dataclass
from typing import Any, Optional

from src.data_models.fine_tuning import HyperparameterConfig
from src.fine_tuning.format_validator import ModelFamily, detect_model_family


@dataclass
class BedrockJobResult:
    """Result of a Bedrock fine-tuning job creation.

    Attributes:
        success: Whether the job was created successfully.
        job_arn: The ARN of the created customization job.
        error: Error message if creation failed.
    """

    success: bool
    job_arn: str = ""
    error: str = ""


def _build_hyperparameters(
    config: HyperparameterConfig,
    family: ModelFamily,
) -> dict[str, str]:
    """Build Bedrock-compatible hyperparameter dict from config.

    Bedrock expects all hyperparameter values as strings.
    """
    params: dict[str, str] = {
        "epochCount": str(config.epochs),
        "batchSize": str(config.batch_size),
        "learningRate": str(config.learning_rate),
        "warmupSteps": str(config.warmup_steps),
    }

    if family == ModelFamily.CLAUDE and config.lora_rank is not None:
        params["loraRank"] = str(config.lora_rank)

    return params


def _build_training_data_config(
    training_data_s3_uri: str,
) -> dict[str, Any]:
    """Build the training data configuration for Bedrock."""
    return {
        "s3Uri": training_data_s3_uri,
    }


def _build_output_data_config(
    output_s3_uri: str,
) -> dict[str, Any]:
    """Build the output data configuration for Bedrock."""
    return {
        "s3Uri": output_s3_uri,
    }


def create_bedrock_fine_tuning_job(
    bedrock_client: Any,
    job_name: str,
    base_model_id: str,
    training_data_s3_uri: str,
    output_s3_uri: str,
    role_arn: str,
    hyperparameters: HyperparameterConfig,
    output_model_name: str,
    validation_data_s3_uri: Optional[str] = None,
) -> BedrockJobResult:
    """Create a model customization job on AWS Bedrock.

    Calls create_model_customization_job() with parameters appropriate
    for the detected model family (Claude, Titan, Llama).

    Args:
        bedrock_client: A boto3 Bedrock client instance.
        job_name: Name for the customization job.
        base_model_id: The base model identifier to fine-tune.
        training_data_s3_uri: S3 URI of the training data.
        output_s3_uri: S3 URI for output artifacts.
        role_arn: IAM role ARN for the job.
        hyperparameters: Hyperparameter configuration.
        output_model_name: Name for the output custom model.
        validation_data_s3_uri: Optional S3 URI for validation data.

    Returns:
        BedrockJobResult with job ARN on success or error on failure.
    """
    family = detect_model_family(base_model_id)

    if family == ModelFamily.UNKNOWN:
        return BedrockJobResult(
            success=False,
            error=f"Unsupported model family for model ID: {base_model_id}",
        )

    hp_dict = _build_hyperparameters(hyperparameters, family)

    kwargs: dict[str, Any] = {
        "jobName": job_name,
        "customModelName": output_model_name,
        "roleArn": role_arn,
        "baseModelIdentifier": base_model_id,
        "trainingDataConfig": _build_training_data_config(training_data_s3_uri),
        "outputDataConfig": _build_output_data_config(output_s3_uri),
        "hyperParameters": hp_dict,
        "customizationType": "FINE_TUNING",
    }

    if validation_data_s3_uri:
        kwargs["validationDataConfig"] = {
            "validators": [
                {"s3Uri": validation_data_s3_uri}
            ]
        }

    try:
        response = bedrock_client.create_model_customization_job(**kwargs)
        job_arn = response.get("jobArn", "")
        return BedrockJobResult(success=True, job_arn=job_arn)
    except Exception as e:
        return BedrockJobResult(
            success=False,
            error=f"Failed to create Bedrock fine-tuning job: {e}",
        )
