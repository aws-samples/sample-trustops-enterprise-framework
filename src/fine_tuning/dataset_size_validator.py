"""
Minimum dataset size validator for fine-tuning.

Validates that training datasets meet the minimum row count and
estimated token count requirements for each model family.

Requirements: 4.14
"""

from dataclasses import dataclass, field
from typing import Any

from src.fine_tuning.format_validator import ModelFamily, detect_model_family


@dataclass
class DatasetSizeValidationResult:
    """Result of dataset size validation.

    Attributes:
        is_valid: Whether the dataset meets minimum size requirements.
        reason: Human-readable reason if validation fails.
        row_count: Number of rows in the dataset.
        estimated_token_count: Estimated total token count.
        min_rows_required: Minimum rows required for the model family.
        min_tokens_required: Minimum tokens required for the model family.
    """

    is_valid: bool
    reason: str = ""
    row_count: int = 0
    estimated_token_count: int = 0
    min_rows_required: int = 0
    min_tokens_required: int = 0


# Minimum row counts per model family
MIN_ROWS: dict[ModelFamily, int] = {
    ModelFamily.CLAUDE: 32,
    ModelFamily.TITAN: 100,
    ModelFamily.LLAMA: 64,
}

# Minimum total token counts per model family
MIN_TOKENS: dict[ModelFamily, int] = {
    ModelFamily.CLAUDE: 1000,
    ModelFamily.TITAN: 5000,
    ModelFamily.LLAMA: 2000,
}

# Approximate characters per token for estimation
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text using character-based heuristic."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _estimate_record_tokens(record: dict[str, Any], family: ModelFamily) -> int:
    """Estimate total tokens for a single record based on model family fields."""
    if family == ModelFamily.TITAN:
        input_text = record.get("inputText", "")
        output_text = record.get("outputText", "")
    else:
        # Claude and Llama use prompt/completion
        input_text = record.get("prompt", "")
        output_text = record.get("completion", "")

    return _estimate_tokens(str(input_text)) + _estimate_tokens(str(output_text))


def validate_dataset_size(
    records: list[dict[str, Any]],
    model_id: str,
) -> DatasetSizeValidationResult:
    """Validate that a dataset meets minimum size requirements for fine-tuning.

    Checks both row count and estimated token count against model-family-specific
    minimums. Different model families have different requirements:
    - Claude: 32+ examples, 1000+ tokens
    - Titan: 100+ examples, 5000+ tokens
    - Llama: 64+ examples, 2000+ tokens

    Args:
        records: List of parsed training data records.
        model_id: The target model identifier.

    Returns:
        DatasetSizeValidationResult with validation status and details.
    """
    family = detect_model_family(model_id)

    if family == ModelFamily.UNKNOWN:
        return DatasetSizeValidationResult(
            is_valid=False,
            reason=f"Unknown model family for model ID: {model_id}",
        )

    min_rows = MIN_ROWS[family]
    min_tokens = MIN_TOKENS[family]
    row_count = len(records)

    estimated_tokens = sum(
        _estimate_record_tokens(r, family) for r in records
    )

    reasons: list[str] = []

    if row_count < min_rows:
        reasons.append(
            f"Dataset has {row_count} rows but {family.value} fine-tuning "
            f"requires at least {min_rows}"
        )

    if estimated_tokens < min_tokens:
        reasons.append(
            f"Dataset has ~{estimated_tokens} estimated tokens but "
            f"{family.value} fine-tuning requires at least {min_tokens}"
        )

    return DatasetSizeValidationResult(
        is_valid=len(reasons) == 0,
        reason="; ".join(reasons),
        row_count=row_count,
        estimated_token_count=estimated_tokens,
        min_rows_required=min_rows,
        min_tokens_required=min_tokens,
    )
