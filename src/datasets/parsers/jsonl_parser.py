"""
JSONL parser with validation for required fields per task type.

This module provides parsing and validation for JSONL (JSON Lines) format
datasets, ensuring required fields are present based on the task type.

Requirement 2.6: Implement JSONL parser and validator
"""

import json
from pathlib import Path
from typing import Any, Union

from src.data_models.dataset import DatasetTaskType


class JSONLParser:
    """Parser for JSONL format datasets with task-specific validation."""

    # Required fields for each task type
    REQUIRED_FIELDS = {
        DatasetTaskType.QA: ["prompt", "completion"],
        DatasetTaskType.SUMMARIZATION: ["prompt", "completion"],
        DatasetTaskType.CLASSIFICATION: ["prompt", "completion"],
        DatasetTaskType.TEXT_GENERATION: ["prompt", "completion"],
        DatasetTaskType.CHAT: ["messages"],
        DatasetTaskType.CUSTOM: ["prompt"],
    }

    # Optional but recommended fields
    OPTIONAL_FIELDS = {
        DatasetTaskType.QA: ["context", "category"],
        DatasetTaskType.SUMMARIZATION: ["category"],
        DatasetTaskType.CLASSIFICATION: ["category"],
        DatasetTaskType.TEXT_GENERATION: ["category"],
        DatasetTaskType.CHAT: ["category"],
        DatasetTaskType.CUSTOM: ["completion", "category"],
    }

    @staticmethod
    def parse(
        file_path: Union[str, Path],
        task_type: DatasetTaskType,
        validate: bool = True
    ) -> list[dict[str, Any]]:
        """
        Parse a JSONL file and optionally validate required fields.

        Args:
            file_path: Path to the JSONL file
            task_type: Task type for validation
            validate: Whether to validate required fields

        Returns:
            List of parsed JSON objects

        Raises:
            ValueError: If file doesn't exist or validation fails
            json.JSONDecodeError: If JSON parsing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        examples = []
        line_number = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_number += 1
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON at line {line_number}: {e.msg}",
                        e.doc,
                        e.pos
                    )

                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Line {line_number}: Expected JSON object, "
                        f"got {type(obj).__name__}"
                    )

                if validate:
                    JSONLParser._validate_example(
                        obj, task_type, line_number
                    )

                examples.append(obj)

        if not examples:
            raise ValueError(f"No valid examples found in {file_path}")

        return examples

    @staticmethod
    def _validate_example(
        example: dict[str, Any],
        task_type: DatasetTaskType,
        line_number: int
    ) -> None:
        """
        Validate that an example has required fields for the task type.

        Args:
            example: The example to validate
            task_type: Task type to validate against
            line_number: Line number for error reporting

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = JSONLParser.REQUIRED_FIELDS.get(
            task_type, []
        )

        missing_fields = []
        for field in required_fields:
            if field not in example:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"Line {line_number}: Missing required fields for "
                f"{task_type.value}: {', '.join(missing_fields)}"
            )

        # Validate specific field types for certain task types
        if task_type == DatasetTaskType.CHAT:
            if "messages" in example:
                if not isinstance(example["messages"], list):
                    raise ValueError(
                        f"Line {line_number}: 'messages' must be a list"
                    )
                if not example["messages"]:
                    raise ValueError(
                        f"Line {line_number}: 'messages' cannot be empty"
                    )

    @staticmethod
    def validate_file(
        file_path: Union[str, Path],
        task_type: DatasetTaskType
    ) -> dict[str, Any]:
        """
        Validate a JSONL file and return validation report.

        Args:
            file_path: Path to the JSONL file
            task_type: Task type for validation

        Returns:
            Dictionary with validation results:
            - valid: bool
            - total_examples: int
            - errors: list of error messages
            - warnings: list of warning messages
        """
        file_path = Path(file_path)
        errors = []
        warnings = []
        total_examples = 0

        try:
            examples = JSONLParser.parse(
                file_path, task_type, validate=True
            )
            total_examples = len(examples)

            # Check for optional fields
            optional_fields = JSONLParser.OPTIONAL_FIELDS.get(
                task_type, []
            )
            if optional_fields:
                missing_optional = set(optional_fields)
                for example in examples:
                    missing_optional -= set(example.keys())

                if missing_optional:
                    warnings.append(
                        f"Optional fields not found in all examples: "
                        f"{', '.join(missing_optional)}"
                    )

        except (ValueError, json.JSONDecodeError) as e:
            errors.append(str(e))

        return {
            "valid": len(errors) == 0,
            "total_examples": total_examples,
            "errors": errors,
            "warnings": warnings,
        }

    @staticmethod
    def get_required_fields(task_type: DatasetTaskType) -> list[str]:
        """
        Get required fields for a task type.

        Args:
            task_type: Task type

        Returns:
            List of required field names
        """
        return JSONLParser.REQUIRED_FIELDS.get(task_type, [])

    @staticmethod
    def get_optional_fields(task_type: DatasetTaskType) -> list[str]:
        """
        Get optional fields for a task type.

        Args:
            task_type: Task type

        Returns:
            List of optional field names
        """
        return JSONLParser.OPTIONAL_FIELDS.get(task_type, [])
