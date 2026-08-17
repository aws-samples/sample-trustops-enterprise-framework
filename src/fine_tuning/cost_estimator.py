"""
Cost estimator for fine-tuning jobs.

Calculates estimated training cost based on model family, dataset size
(number of examples and token count), and number of epochs. Provides
a cost breakdown and confidence level before job submission.

Requirements: 4.11
"""

from src.data_models.fine_tuning import CostEstimate, HyperparameterConfig
from src.fine_tuning.format_validator import ModelFamily, detect_model_family


# Per-token training cost (USD per 1k tokens) by model family
_TRAINING_COST_PER_1K_TOKENS: dict[ModelFamily, float] = {
    ModelFamily.CLAUDE: 0.008,
    ModelFamily.TITAN: 0.0004,
    ModelFamily.LLAMA: 0.0008,
    ModelFamily.UNKNOWN: 0.001,
}

# Estimated tokens processed per hour by model family
_TOKENS_PER_HOUR: dict[ModelFamily, float] = {
    ModelFamily.CLAUDE: 500_000,
    ModelFamily.TITAN: 1_000_000,
    ModelFamily.LLAMA: 750_000,
    ModelFamily.UNKNOWN: 500_000,
}

# Hourly infrastructure cost by model family (compute overhead)
_INFRA_COST_PER_HOUR: dict[ModelFamily, float] = {
    ModelFamily.CLAUDE: 5.0,
    ModelFamily.TITAN: 2.0,
    ModelFamily.LLAMA: 3.5,
    ModelFamily.UNKNOWN: 3.0,
}

# Default average tokens per example when token_count is not provided
_DEFAULT_AVG_TOKENS_PER_EXAMPLE: int = 512


def estimate_cost(
    model_id: str,
    num_examples: int,
    hyperparameters: HyperparameterConfig,
    token_count: int | None = None,
) -> CostEstimate:
    """Estimate the cost of a fine-tuning job.

    Calculates training token cost, infrastructure cost, and estimated
    duration based on model family pricing, dataset size, and epochs.

    Args:
        model_id: The target model identifier.
        num_examples: Number of training examples in the dataset.
        hyperparameters: Hyperparameter configuration (uses epochs).
        token_count: Total token count in the dataset. If None, estimated
            from num_examples using a default average.

    Returns:
        CostEstimate with breakdown and confidence level.
    """
    family = detect_model_family(model_id)
    epochs = hyperparameters.epochs

    # Estimate total tokens if not provided
    if token_count is not None and token_count > 0:
        total_tokens = token_count
        confidence = "high"
    else:
        total_tokens = num_examples * _DEFAULT_AVG_TOKENS_PER_EXAMPLE
        confidence = "medium"

    # If dataset is empty, return zero cost
    if num_examples <= 0 or total_tokens <= 0:
        return CostEstimate(
            estimated_training_cost=0.0,
            estimated_duration_hours=0.0,
            cost_breakdown={
                # "token" here means LLM tokens, not a credential.
                "training_token_cost": 0.0,  # nosec B105
                "infrastructure_cost": 0.0,
            },
            currency="USD",
            confidence="high",
        )

    # Total tokens processed = dataset tokens * epochs
    total_training_tokens = total_tokens * epochs

    # Training token cost
    cost_per_1k = _TRAINING_COST_PER_1K_TOKENS.get(
        family, _TRAINING_COST_PER_1K_TOKENS[ModelFamily.UNKNOWN]
    )
    training_token_cost = (total_training_tokens / 1000) * cost_per_1k

    # Estimated duration
    tokens_per_hour = _TOKENS_PER_HOUR.get(
        family, _TOKENS_PER_HOUR[ModelFamily.UNKNOWN]
    )
    estimated_hours = total_training_tokens / tokens_per_hour

    # Infrastructure cost
    infra_per_hour = _INFRA_COST_PER_HOUR.get(
        family, _INFRA_COST_PER_HOUR[ModelFamily.UNKNOWN]
    )
    infrastructure_cost = estimated_hours * infra_per_hour

    total_cost = training_token_cost + infrastructure_cost

    # Lower confidence for unknown model families
    if family == ModelFamily.UNKNOWN:
        confidence = "low"

    return CostEstimate(
        estimated_training_cost=round(total_cost, 4),
        estimated_duration_hours=round(estimated_hours, 4),
        cost_breakdown={
            "training_token_cost": round(training_token_cost, 4),
            "infrastructure_cost": round(infrastructure_cost, 4),
        },
        currency="USD",
        confidence=confidence,
    )


def format_cost_display(estimate: CostEstimate) -> str:
    """Format a cost estimate for display before job submission.

    Args:
        estimate: The cost estimate to format.

    Returns:
        Human-readable string summarizing the estimate.
    """
    lines = [
        "Fine-Tuning Cost Estimate",
        "=" * 40,
        f"Estimated Total Cost:    "
        f"${estimate.estimated_training_cost:.2f} "
        f"{estimate.currency}",
        f"Estimated Duration:      "
        f"{estimate.estimated_duration_hours:.2f} hours",
        f"Confidence:              "
        f"{estimate.confidence}",
        "",
        "Cost Breakdown:",
    ]
    for component, cost in estimate.cost_breakdown.items():
        label = component.replace("_", " ").title()
        lines.append(f"  {label}: ${cost:.4f}")
    lines.append("=" * 40)
    return "\n".join(lines)
