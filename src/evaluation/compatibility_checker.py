"""
Model-Dataset Compatibility Checker.

Validates that a model can work with a given dataset before
running evaluation. Checks task type support, format compatibility,
and returns structured validation results with specific
incompatibility reasons.

Requirements: 3.1, 3.2, 3.4
"""

from dataclasses import dataclass, field
from typing import Optional

from src.data_models.dataset import (
    DatasetFormat,
    DatasetMetadata,
    DatasetTaskType,
)
from src.data_models.model import (
    ModelCapability,
    ModelMetadata,
    ModelStatus,
)


# Mapping from dataset task types to required model capabilities.
# A model needs at least one of the listed capabilities.
TASK_TYPE_TO_CAPABILITIES: dict[
    DatasetTaskType, list[ModelCapability]
] = {
    DatasetTaskType.QA: [
        ModelCapability.TEXT_GENERATION,
        ModelCapability.CHAT,
    ],
    DatasetTaskType.SUMMARIZATION: [
        ModelCapability.TEXT_GENERATION,
        ModelCapability.CHAT,
    ],
    DatasetTaskType.CLASSIFICATION: [
        ModelCapability.TEXT_GENERATION,
        ModelCapability.CHAT,
        ModelCapability.COMPLETION,
    ],
    DatasetTaskType.TEXT_GENERATION: [
        ModelCapability.TEXT_GENERATION,
        ModelCapability.COMPLETION,
    ],
    DatasetTaskType.CHAT: [
        ModelCapability.CHAT,
    ],
    DatasetTaskType.CUSTOM: [
        ModelCapability.TEXT_GENERATION,
        ModelCapability.CHAT,
        ModelCapability.COMPLETION,
    ],
}

# Supported dataset formats for evaluation
SUPPORTED_EVALUATION_FORMATS: set[DatasetFormat] = {
    DatasetFormat.JSONL,
    DatasetFormat.CSV,
    DatasetFormat.PARQUET,
}


@dataclass
class CompatibilityResult:
    """Result of a model-dataset compatibility check.

    Attributes:
        is_compatible: Whether model and dataset are compatible.
        reasons: Specific incompatibility reasons (empty if ok).
        model_id: The model ID that was checked.
        dataset_id: The dataset ID checked (optional).
    """

    is_compatible: bool
    reasons: list[str] = field(default_factory=list)
    model_id: str = ""
    dataset_id: Optional[str] = None


def check_model_dataset_compatibility(
    model: ModelMetadata,
    dataset: DatasetMetadata,
) -> CompatibilityResult:
    """Check if a model is compatible with a dataset.

    Validates:
    - Model is active and available for inference
    - Model supports the dataset's task type
    - Dataset format is supported for evaluation

    Args:
        model: The model metadata to check.
        dataset: The dataset metadata to check against.

    Returns:
        CompatibilityResult with compatibility flag and reasons.
    """
    reasons: list[str] = []

    # Check model availability
    if model.status != ModelStatus.ACTIVE:
        reasons.append(
            f"Model '{model.id}' is not available for "
            f"inference (status: {model.status.value})"
        )

    # Check task type compatibility
    reasons.extend(
        _check_task_type_compatibility(model, dataset.task_type)
    )

    # Check format compatibility
    reasons.extend(_check_format_compatibility(dataset.format))

    return CompatibilityResult(
        is_compatible=len(reasons) == 0,
        reasons=reasons,
        model_id=model.id,
        dataset_id=dataset.id,
    )


def check_task_type_support(
    model: ModelMetadata,
    task_type: DatasetTaskType,
) -> CompatibilityResult:
    """Check if a model supports a specific task type.

    Useful for quick checks without full dataset metadata.

    Args:
        model: The model metadata to check.
        task_type: The task type to verify support for.

    Returns:
        CompatibilityResult with compatibility flag and reasons.
    """
    reasons: list[str] = []

    if model.status != ModelStatus.ACTIVE:
        reasons.append(
            f"Model '{model.id}' is not available for "
            f"inference (status: {model.status.value})"
        )

    reasons.extend(
        _check_task_type_compatibility(model, task_type)
    )

    return CompatibilityResult(
        is_compatible=len(reasons) == 0,
        reasons=reasons,
        model_id=model.id,
    )


def _check_task_type_compatibility(
    model: ModelMetadata,
    task_type: DatasetTaskType,
) -> list[str]:
    """Check if model capabilities support the task type.

    A model is compatible if it has at least one of the
    required capabilities for the task type.

    Returns:
        List of incompatibility reasons (empty if compatible).
    """
    required = TASK_TYPE_TO_CAPABILITIES.get(task_type, [])

    if not required:
        return []

    model_caps = set(model.capabilities)
    required_caps = set(required)

    if not model_caps.intersection(required_caps):
        cap_names = ", ".join(c.value for c in required)
        model_cap_names = (
            ", ".join(c.value for c in model.capabilities)
            if model.capabilities
            else "none"
        )
        return [
            f"Model '{model.id}' does not support task "
            f"type '{task_type.value}'. "
            f"Required capabilities (at least one): "
            f"[{cap_names}]. "
            f"Model capabilities: [{model_cap_names}]"
        ]

    return []


def _check_format_compatibility(
    dataset_format: DatasetFormat,
) -> list[str]:
    """Check if the dataset format is supported.

    Returns:
        List of incompatibility reasons (empty if compatible).
    """
    if dataset_format not in SUPPORTED_EVALUATION_FORMATS:
        supported = ", ".join(
            f.value for f in SUPPORTED_EVALUATION_FORMATS
        )
        return [
            f"Dataset format '{dataset_format.value}' is "
            f"not supported for evaluation. "
            f"Supported formats: [{supported}]"
        ]

    return []
