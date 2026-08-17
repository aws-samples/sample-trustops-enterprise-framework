"""
Base template class for domain-specific dataset templates.

Requirement 2.19: Domain-specific templates foundation
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from src.data_models.dataset import DatasetTaskType


class PIISensitivity(str, Enum):
    """PII sensitivity levels for dataset fields."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FieldDefinition(BaseModel):
    """Definition of a dataset field with validation rules."""

    name: str = Field(..., description="Field name")
    data_type: str = Field(..., description="Expected data type (str, int, float, list, dict)")
    required: bool = Field(default=True, description="Whether field is required")
    description: str = Field(..., description="Field description")
    pii_sensitivity: PIISensitivity = Field(
        default=PIISensitivity.NONE,
        description="PII sensitivity level"
    )
    min_length: Optional[int] = Field(default=None, description="Minimum string length")
    max_length: Optional[int] = Field(default=None, description="Maximum string length")
    allowed_values: Optional[list[Any]] = Field(
        default=None,
        description="List of allowed values (for enums)"
    )
    pattern: Optional[str] = Field(default=None, description="Regex pattern for validation")


class ValidationRule(BaseModel):
    """Custom validation rule for dataset."""

    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    validator: Optional[Callable[[dict], tuple[bool, Optional[str]]]] = Field(
        default=None,
        description="Validation function returning (is_valid, error_message)"
    )

    class Config:
        arbitrary_types_allowed = True


class QualityThresholds(BaseModel):
    """Quality thresholds for dataset validation."""

    min_completeness: float = Field(default=0.95, ge=0.0, le=1.0)
    min_diversity: float = Field(default=0.7, ge=0.0, le=1.0)
    min_balance: float = Field(default=0.6, ge=0.0, le=1.0)
    min_examples: int = Field(default=100, ge=1)
    max_examples: Optional[int] = Field(default=None, ge=1)


class DatasetTemplate(ABC):
    """Abstract base class for domain-specific dataset templates."""

    def __init__(self):
        """Initialize the template."""
        self._fields: list[FieldDefinition] = []
        self._validation_rules: list[ValidationRule] = []
        self._quality_thresholds = QualityThresholds()
        self._task_type: DatasetTaskType = DatasetTaskType.CUSTOM
        self._example_data: list[dict] = []
        self._initialize_template()

    @abstractmethod
    def _initialize_template(self) -> None:
        """Initialize template-specific fields and rules."""
        pass

    @property
    def name(self) -> str:
        """Get template name."""
        return self.__class__.__name__.replace("Template", "")

    @property
    def task_type(self) -> DatasetTaskType:
        """Get task type for this template."""
        return self._task_type

    @property
    def fields(self) -> list[FieldDefinition]:
        """Get field definitions."""
        return self._fields

    @property
    def validation_rules(self) -> list[ValidationRule]:
        """Get validation rules."""
        return self._validation_rules

    @property
    def quality_thresholds(self) -> QualityThresholds:
        """Get quality thresholds."""
        return self._quality_thresholds

    @property
    def example_data(self) -> list[dict]:
        """Get example data."""
        return self._example_data

    def get_schema(self) -> dict:
        """Get JSON schema representation of the template."""
        return {
            "name": self.name,
            "task_type": self.task_type.value,
            "fields": [
                {
                    "name": field.name,
                    "data_type": field.data_type,
                    "required": field.required,
                    "description": field.description,
                    "pii_sensitivity": field.pii_sensitivity.value,
                    "min_length": field.min_length,
                    "max_length": field.max_length,
                    "allowed_values": field.allowed_values,
                    "pattern": field.pattern,
                }
                for field in self._fields
            ],
            "validation_rules": [
                {"name": rule.name, "description": rule.description}
                for rule in self._validation_rules
            ],
            "quality_thresholds": self._quality_thresholds.model_dump(),
        }

    def validate_example(self, example: dict) -> tuple[bool, list[str]]:
        """
        Validate a single example against the template schema.

        Args:
            example: Example data to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check required fields
        for field in self._fields:
            if field.required and field.name not in example:
                errors.append(f"Missing required field: {field.name}")
                continue

            if field.name not in example:
                continue

            value = example[field.name]

            # Type validation
            if not self._validate_type(value, field.data_type):
                errors.append(
                    f"Field '{field.name}' has invalid type. "
                    f"Expected {field.data_type}, got {type(value).__name__}"
                )
                continue

            # String length validation
            if isinstance(value, str):
                if field.min_length and len(value) < field.min_length:
                    errors.append(
                        f"Field '{field.name}' is too short. "
                        f"Minimum length: {field.min_length}"
                    )
                if field.max_length and len(value) > field.max_length:
                    errors.append(
                        f"Field '{field.name}' is too long. "
                        f"Maximum length: {field.max_length}"
                    )

            # Allowed values validation
            if field.allowed_values and value not in field.allowed_values:
                errors.append(
                    f"Field '{field.name}' has invalid value. "
                    f"Allowed values: {field.allowed_values}"
                )

        # Apply custom validation rules
        for rule in self._validation_rules:
            if rule.validator:
                is_valid, error_msg = rule.validator(example)
                if not is_valid and error_msg:
                    errors.append(f"{rule.name}: {error_msg}")

        return len(errors) == 0, errors

    def validate_dataset(self, examples: list[dict]) -> tuple[bool, dict]:
        """
        Validate entire dataset against the template.

        Args:
            examples: List of examples to validate

        Returns:
            Tuple of (is_valid, validation report dict)
        """
        report = {
            "total_examples": len(examples),
            "valid_examples": 0,
            "invalid_examples": 0,
            "errors_by_example": {},
            "field_errors": {},
            "meets_quality_thresholds": False,
        }

        # Validate each example
        for idx, example in enumerate(examples):
            is_valid, errors = self.validate_example(example)
            if is_valid:
                report["valid_examples"] += 1
            else:
                report["invalid_examples"] += 1
                report["errors_by_example"][idx] = errors

                # Track field-level errors
                for error in errors:
                    if ":" in error:
                        field_name = error.split(":")[0].strip()
                        report["field_errors"][field_name] = (
                            report["field_errors"].get(field_name, 0) + 1
                        )

        # Check quality thresholds
        if len(examples) >= self._quality_thresholds.min_examples:
            if self._quality_thresholds.max_examples:
                meets_count = len(examples) <= self._quality_thresholds.max_examples
            else:
                meets_count = True

            completeness = report["valid_examples"] / len(examples) if examples else 0
            meets_completeness = completeness >= self._quality_thresholds.min_completeness

            report["meets_quality_thresholds"] = meets_count and meets_completeness
        else:
            report["meets_quality_thresholds"] = False

        is_valid = report["invalid_examples"] == 0 and report["meets_quality_thresholds"]
        return is_valid, report

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type."""
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)

    def get_pii_fields(self) -> list[str]:
        """Get list of fields with PII sensitivity."""
        return [
            field.name
            for field in self._fields
            if field.pii_sensitivity != PIISensitivity.NONE
        ]

    def get_required_fields(self) -> list[str]:
        """Get list of required field names."""
        return [field.name for field in self._fields if field.required]
