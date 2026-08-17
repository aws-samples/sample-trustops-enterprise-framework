"""
SageMaker training job creator.

Creates training jobs on AWS SageMaker with custom containers,
hyperparameters, and data channels for models not available
through Bedrock fine-tuning.

Requirements: 4.4
"""

from dataclasses import dataclass
from typing import Any, Optional

from src.data_models.fine_tuning import HyperparameterConfig


@dataclass
class SageMakerJobResult:
    """Result of a SageMaker training job creation.

    Attributes:
        success: Whether the job was created successfully.
        job_arn: The ARN of the created training job.
        error: Error message if creation failed.
    """

    success: bool
    job_arn: str = ""
    error: str = ""


def _build_hyperparameters(config: HyperparameterConfig) -> dict[str, str]:
    """Build SageMaker-compatible hyperparameter dict.

    SageMaker expects all hyperparameter values as strings.
    """
    params: dict[str, str] = {
        "epochs": str(config.epochs),
        "batch_size": str(config.batch_size),
        "learning_rate": str(config.learning_rate),
        "warmup_steps": str(config.warmup_steps),
        "weight_decay": str(config.weight_decay),
        "max_seq_length": str(config.max_seq_length),
    }

    if config.lora_rank is not None:
        params["lora_rank"] = str(config.lora_rank)
    if config.lora_alpha is not None:
        params["lora_alpha"] = str(config.lora_alpha)

    return params


def _build_input_data_config(
    training_data_s3_uri: str,
    validation_data_s3_uri: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Build the input data channel configuration for SageMaker."""
    channels: list[dict[str, Any]] = [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": training_data_s3_uri,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "application/jsonlines",
        }
    ]

    if validation_data_s3_uri:
        channels.append(
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": validation_data_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/jsonlines",
            }
        )

    return channels


def create_sagemaker_training_job(
    sagemaker_client: Any,
    job_name: str,
    container_image_uri: str,
    training_data_s3_uri: str,
    output_s3_uri: str,
    role_arn: str,
    hyperparameters: HyperparameterConfig,
    instance_type: str = "ml.g5.2xlarge",
    instance_count: int = 1,
    volume_size_gb: int = 100,
    max_runtime_seconds: int = 86400,
    validation_data_s3_uri: Optional[str] = None,
) -> SageMakerJobResult:
    """Create a training job on AWS SageMaker.

    Args:
        sagemaker_client: A boto3 SageMaker client instance.
        job_name: Name for the training job.
        container_image_uri: ECR URI of the training container image.
        training_data_s3_uri: S3 URI of the training data.
        output_s3_uri: S3 URI for output model artifacts.
        role_arn: IAM role ARN for the job.
        hyperparameters: Hyperparameter configuration.
        instance_type: SageMaker instance type.
        instance_count: Number of training instances.
        volume_size_gb: EBS volume size in GB.
        max_runtime_seconds: Maximum training runtime in seconds.
        validation_data_s3_uri: Optional S3 URI for validation data.

    Returns:
        SageMakerJobResult with job ARN on success or error on failure.
    """
    hp_dict = _build_hyperparameters(hyperparameters)
    input_channels = _build_input_data_config(
        training_data_s3_uri, validation_data_s3_uri
    )

    try:
        response = sagemaker_client.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                "TrainingImage": container_image_uri,
                "TrainingInputMode": "File",
            },
            RoleArn=role_arn,
            InputDataConfig=input_channels,
            OutputDataConfig={
                "S3OutputPath": output_s3_uri,
            },
            ResourceConfig={
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
                "VolumeSizeInGB": volume_size_gb,
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": max_runtime_seconds,
            },
            HyperParameters=hp_dict,
        )
        job_arn = response.get("TrainingJobArn", "")
        return SageMakerJobResult(success=True, job_arn=job_arn)
    except Exception as e:
        return SageMakerJobResult(
            success=False,
            error=f"Failed to create SageMaker training job: {e}",
        )
