"""
Fine-tuning eligibility checker.

Validates whether a model is eligible for fine-tuning by checking
its capabilities, status, and fine-tuning support flag from the
Model Registry metadata.

Requirements: 4.1
"""

from dataclasses import dataclass, field

from src.data_models.model import ModelCapability, ModelMetadata, ModelStatus


@dataclass
class EligibilityResult:
    """Result of a fine-tuning eligibility check.

    Attributes:
        is_eligible: Whether the model is eligible for fine-tuning.
        reasons: List of reasons why the model is not eligible (empty if eligible).
    """

    is_eligible: bool
    reasons: list[str] = field(default_factory=list)


def check_fine_tuning_eligibility(model_metadata: ModelMetadata) -> EligibilityResult:
    """Check whether a model is eligible for fine-tuning.

    Validates three conditions:
    1. The model has the FINE_TUNABLE capability.
    2. The model status is ACTIVE.
    3. The model's fine_tuning_support flag is True.

    Args:
        model_metadata: The model metadata from the Model Registry.

    Returns:
        EligibilityResult with is_eligible=True if all checks pass,
        or is_eligible=False with reasons describing each failed check.
    """
    reasons: list[str] = []

    if ModelCapability.FINE_TUNABLE not in model_metadata.capabilities:
        reasons.append(
            f"Model '{model_metadata.id}' does not have the FINE_TUNABLE capability"
        )

    if model_metadata.status != ModelStatus.ACTIVE:
        reasons.append(
            f"Model '{model_metadata.id}' is not active (current status: {model_metadata.status.value})"
        )

    if not model_metadata.fine_tuning_support:
        reasons.append(
            f"Model '{model_metadata.id}' does not have fine-tuning support enabled"
        )

    return EligibilityResult(
        is_eligible=len(reasons) == 0,
        reasons=reasons,
    )
