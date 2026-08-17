"""
Training data format validator for fine-tuning.

Validates that training data matches model-specific format requirements
for each model family (Claude, Titan, Llama). Checks JSONL structure
and required fields per model family.

Requirements: 4.2
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelFamily(str, Enum):
    """Supported model families for fine-tuning."""

    CLAUDE = "claude"
    TITAN = "titan"
    LLAMA = "llama"
    UNKNOWN = "unknown"


@dataclass
class FormatValidationResult:
    """Result of training data format validation.

    Attributes:
        is_valid: Whether the data passes format validation.
        issues: List of specific issues found during validation.
        model_family: The detected model family.
        records_checked: Number of records validated.
    """

    is_valid: bool
    issues: list[str] = field(default_factory=list)
    model_family: str = ""
    records_checked: int = 0


# Required fields per model family
MODEL_REQUIRED_FIELDS: dict[ModelFamily, list[str]] = {
    ModelFamily.CLAUDE: ["prompt", "completion"],
    ModelFamily.TITAN: ["inputText", "outputText"],
    ModelFamily.LLAMA: ["prompt", "completion"],
}

# Optional fields per model family
MODEL_OPTIONAL_FIELDS: dict[ModelFamily, list[str]] = {
    ModelFamily.CLAUDE: ["system"],
    ModelFamily.TITAN: [],
    ModelFamily.LLAMA: [],
}


def detect_model_family(model_id: str) -> ModelFamily:
    """Detect model family from a model identifier.

    Args:
        model_id: Model identifier string.

    Returns:
        The detected ModelFamily enum value.
    """
    model_id_lower = model_id.lower()

    if "claude" in model_id_lower or "anthropic" in model_id_lower:
        return ModelFamily.CLAUDE
    elif "titan" in model_id_lower or "amazon" in model_id_lower:
        return ModelFamily.TITAN
    elif "llama" in model_id_lower or "meta" in model_id_lower:
        return ModelFamily.LLAMA
    return ModelFamily.UNKNOWN


def validate_training_data_format(
    records: list[dict[str, Any]],
    model_id: str,
) -> FormatValidationResult:
    """Validate training data format against model-specific requirements.

    Checks that each record in the dataset contains the required fields
    for the target model family, that field values are non-empty strings,
    and that the data is in valid JSONL-compatible structure.

    Args:
        records: List of parsed JSONL records (dicts).
        model_id: The target model identifier used to detect model family.

    Returns:
        FormatValidationResult with validation status and specific issues.
    """
    family = detect_model_family(model_id)

    if family == ModelFamily.UNKNOWN:
        return FormatValidationResult(
            is_valid=False,
            issues=[f"Unknown model family for model ID: {model_id}"],
            model_family=family.value,
            records_checked=0,
        )

    required_fields = MODEL_REQUIRED_FIELDS[family]
    issues: list[str] = []

    if not records:
        return FormatValidationResult(
            is_valid=False,
            issues=["Training data is empty — no records provided"],
            model_family=family.value,
            records_checked=0,
        )

    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            issues.append(
                f"Record {idx}: Expected a JSON object, got {type(record).__name__}"
            )
            continue

        # Check required fields
        for req_field in required_fields:
            if req_field not in record:
                issues.append(
                    f"Record {idx}: Missing required field '{req_field}'"
                )
            elif not isinstance(record[req_field], str):
                issues.append(
                    f"Record {idx}: Field '{req_field}' must be a string, "
                    f"got {type(record[req_field]).__name__}"
                )
            elif record[req_field].strip() == "":
                issues.append(
                    f"Record {idx}: Field '{req_field}' cannot be empty"
                )

    return FormatValidationResult(
        is_valid=len(issues) == 0,
        issues=issues,
        model_family=family.value,
        records_checked=len(records),
    )


def validate_jsonl_string(
    jsonl_content: str,
    model_id: str,
) -> FormatValidationResult:
    """Validate a raw JSONL string against model-specific requirements.

    Parses the JSONL content line-by-line and delegates to
    validate_training_data_format for field-level checks.

    Args:
        jsonl_content: Raw JSONL string (one JSON object per line).
        model_id: The target model identifier.

    Returns:
        FormatValidationResult with validation status and specific issues.
    """
    family = detect_model_family(model_id)

    if family == ModelFamily.UNKNOWN:
        return FormatValidationResult(
            is_valid=False,
            issues=[f"Unknown model family for model ID: {model_id}"],
            model_family=family.value,
            records_checked=0,
        )

    records: list[dict[str, Any]] = []
    parse_issues: list[str] = []

    lines = jsonl_content.strip().split("\n")
    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            records.append(obj)
        except json.JSONDecodeError as e:
            parse_issues.append(f"Line {idx}: Invalid JSON — {e.msg}")

    if parse_issues:
        return FormatValidationResult(
            is_valid=False,
            issues=parse_issues,
            model_family=family.value,
            records_checked=0,
        )

    result = validate_training_data_format(records, model_id)
    return result
