"""
Hyperparameter defaults resolver for fine-tuning.

Selects intelligent default hyperparameters based on model family
and dataset size. Provides different defaults for small (<1000),
medium (1000-10000), and large (>10000) datasets.

Requirements: 4.5, 4.9
"""

from enum import Enum

from src.data_models.fine_tuning import HyperparameterConfig
from src.fine_tuning.format_validator import ModelFamily, detect_model_family


class DatasetScale(str, Enum):
    """Dataset size categories for hyperparameter selection."""

    SMALL = "small"      # < 1000 examples
    MEDIUM = "medium"    # 1000 - 10000 examples
    LARGE = "large"      # > 10000 examples


def classify_dataset_scale(dataset_size: int) -> DatasetScale:
    """Classify dataset size into a scale category.

    Args:
        dataset_size: Number of examples in the dataset.

    Returns:
        DatasetScale enum value.
    """
    if dataset_size < 1000:
        return DatasetScale.SMALL
    elif dataset_size <= 10000:
        return DatasetScale.MEDIUM
    return DatasetScale.LARGE


# Default hyperparameters indexed by (ModelFamily, DatasetScale)
_DEFAULTS: dict[tuple[ModelFamily, DatasetScale], dict] = {
    # Claude defaults
    (ModelFamily.CLAUDE, DatasetScale.SMALL): {
        "learning_rate": 1e-5,
        "epochs": 4,
        "batch_size": 4,
        "warmup_steps": 50,
        "weight_decay": 0.01,
    },
    (ModelFamily.CLAUDE, DatasetScale.MEDIUM): {
        "learning_rate": 5e-6,
        "epochs": 3,
        "batch_size": 8,
        "warmup_steps": 100,
        "weight_decay": 0.01,
    },
    (ModelFamily.CLAUDE, DatasetScale.LARGE): {
        "learning_rate": 2e-6,
        "epochs": 2,
        "batch_size": 16,
        "warmup_steps": 200,
        "weight_decay": 0.005,
    },
    # Titan defaults
    (ModelFamily.TITAN, DatasetScale.SMALL): {
        "learning_rate": 2e-5,
        "epochs": 5,
        "batch_size": 4,
        "warmup_steps": 50,
        "weight_decay": 0.01,
    },
    (ModelFamily.TITAN, DatasetScale.MEDIUM): {
        "learning_rate": 1e-5,
        "epochs": 3,
        "batch_size": 8,
        "warmup_steps": 100,
        "weight_decay": 0.01,
    },
    (ModelFamily.TITAN, DatasetScale.LARGE): {
        "learning_rate": 5e-6,
        "epochs": 2,
        "batch_size": 16,
        "warmup_steps": 200,
        "weight_decay": 0.005,
    },
    # Llama defaults
    (ModelFamily.LLAMA, DatasetScale.SMALL): {
        "learning_rate": 2e-5,
        "epochs": 4,
        "batch_size": 4,
        "warmup_steps": 50,
        "weight_decay": 0.01,
    },
    (ModelFamily.LLAMA, DatasetScale.MEDIUM): {
        "learning_rate": 1e-5,
        "epochs": 3,
        "batch_size": 8,
        "warmup_steps": 100,
        "weight_decay": 0.01,
    },
    (ModelFamily.LLAMA, DatasetScale.LARGE): {
        "learning_rate": 5e-6,
        "epochs": 2,
        "batch_size": 16,
        "warmup_steps": 200,
        "weight_decay": 0.005,
    },
}

# Fallback defaults when model family is unknown
_FALLBACK_DEFAULTS: dict[DatasetScale, dict] = {
    DatasetScale.SMALL: {
        "learning_rate": 1e-5,
        "epochs": 4,
        "batch_size": 4,
        "warmup_steps": 50,
        "weight_decay": 0.01,
    },
    DatasetScale.MEDIUM: {
        "learning_rate": 5e-6,
        "epochs": 3,
        "batch_size": 8,
        "warmup_steps": 100,
        "weight_decay": 0.01,
    },
    DatasetScale.LARGE: {
        "learning_rate": 2e-6,
        "epochs": 2,
        "batch_size": 16,
        "warmup_steps": 200,
        "weight_decay": 0.005,
    },
}


def resolve_hyperparameters(
    model_id: str,
    dataset_size: int,
) -> HyperparameterConfig:
    """Resolve default hyperparameters based on model family and dataset size.

    Selects appropriate defaults for learning_rate, epochs, batch_size,
    warmup_steps, and weight_decay. Uses model-family-specific defaults
    when available, falling back to generic defaults for unknown families.

    Args:
        model_id: The target model identifier.
        dataset_size: Number of examples in the training dataset.

    Returns:
        HyperparameterConfig with resolved default values.
    """
    family = detect_model_family(model_id)
    scale = classify_dataset_scale(dataset_size)

    key = (family, scale)
    if key in _DEFAULTS:
        params = _DEFAULTS[key]
    else:
        params = _FALLBACK_DEFAULTS[scale]

    return HyperparameterConfig(**params)
