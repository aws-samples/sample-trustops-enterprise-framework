"""
CSV parser with column mapping configuration.

This module provides parsing and validation for CSV format datasets,
with configurable column mapping to standard field names.

Requirement 2.7: Implement CSV parser and validator with column mapping
"""

import csv
from pathlib import Path
from typing import Any, Optional, Union

from src.data_models.dataset import DatasetTaskType


class CSVParser:
    """Parser for CSV format datasets with column mapping."""

    # Default column mappings for common variations
    DEFAULT_COLUMN_MAPPINGS = {
        "prompt": ["prompt", "input", "question", "query", "text"],
        "completion": [
            "completion", "output", "answer", "response", "target"
        ],
        "context": ["context", "passage", "document", "source"],
        "category": ["category", "label", "class", "type"],
        "messages": ["messages", "conversation", "dialogue"],
    }

    @staticmethod
    def parse(
        file_path: Union[str, Path],
        task_type: DatasetTaskType,
        column_mapping: Optional[dict[str, str]] = None,
        validate: bool = True
    ) -> list[dict[str, Any]]:
        """
        Parse a CSV file with optional column mapping.

        Args:
            file_path: Path to the CSV file
            task_type: Task type for validation
            column_mapping: Optional mapping from CSV columns to standard
                          field names (e.g., {"question": "prompt"})
            validate: Whether to validate required fields

        Returns:
            List of parsed rows as dictionaries

        Raises:
            ValueError: If file doesn't exist or validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        examples = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise ValueError(f"No columns found in {file_path}")

            # Auto-detect column mapping if not provided
            if column_mapping is None:
                column_mapping = CSVParser._auto_detect_mapping(
                    reader.fieldnames
                )

            row_number = 1  # Header is row 0
            for row in reader:
                row_number += 1

                # Apply column mapping
                mapped_row = CSVParser._apply_mapping(
                    row, column_mapping
                )

                if validate:
                    CSVParser._validate_example(
                        mapped_row, task_type, row_number
                    )

                examples.append(mapped_row)

        if not examples:
            raise ValueError(f"No valid examples found in {file_path}")

        return examples

    @staticmethod
    def _auto_detect_mapping(
        column_names: list[str]
    ) -> dict[str, str]:
        """
        Auto-detect column mapping based on column names.

        Args:
            column_names: List of column names from CSV

        Returns:
            Dictionary mapping CSV columns to standard field names
        """
        mapping = {}
        column_names_lower = [col.lower() for col in column_names]

        for standard_field, variations in (
            CSVParser.DEFAULT_COLUMN_MAPPINGS.items()
        ):
            for variation in variations:
                if variation in column_names_lower:
                    # Find original case column name
                    idx = column_names_lower.index(variation)
                    mapping[column_names[idx]] = standard_field
                    break

        return mapping

    @staticmethod
    def _apply_mapping(
        row: dict[str, Any],
        column_mapping: dict[str, str]
    ) -> dict[str, Any]:
        """
        Apply column mapping to a row.

        Args:
            row: Original row from CSV
            column_mapping: Mapping from CSV columns to standard fields

        Returns:
            Row with mapped column names
        """
        mapped_row = {}

        # Apply mapping
        for csv_col, standard_field in column_mapping.items():
            if csv_col in row:
                mapped_row[standard_field] = row[csv_col]

        # Keep unmapped columns as-is
        for col, value in row.items():
            if col not in column_mapping:
                mapped_row[col] = value

        return mapped_row

    @staticmethod
    def _validate_example(
        example: dict[str, Any],
        task_type: DatasetTaskType,
        row_number: int
    ) -> None:
        """
        Validate that an example has required fields for the task type.

        Args:
            example: The example to validate
            task_type: Task type to validate against
            row_number: Row number for error reporting

        Raises:
            ValueError: If required fields are missing
        """
        # Use same required fields as JSONL parser
        from src.datasets.parsers.jsonl_parser import JSONLParser

        required_fields = JSONLParser.get_required_fields(task_type)

        missing_fields = []
        for field in required_fields:
            if field not in example or not example[field]:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"Row {row_number}: Missing required fields for "
                f"{task_type.value}: {', '.join(missing_fields)}"
            )

    @staticmethod
    def validate_file(
        file_path: Union[str, Path],
        task_type: DatasetTaskType,
        column_mapping: Optional[dict[str, str]] = None
    ) -> dict[str, Any]:
        """
        Validate a CSV file and return validation report.

        Args:
            file_path: Path to the CSV file
            task_type: Task type for validation
            column_mapping: Optional column mapping

        Returns:
            Dictionary with validation results:
            - valid: bool
            - total_examples: int
            - detected_mapping: dict
            - errors: list of error messages
            - warnings: list of warning messages
        """
        file_path = Path(file_path)
        errors = []
        warnings = []
        total_examples = 0
        detected_mapping = {}

        try:
            # First, detect mapping
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    detected_mapping = CSVParser._auto_detect_mapping(
                        reader.fieldnames
                    )

            # Then parse and validate
            examples = CSVParser.parse(
                file_path, task_type, column_mapping, validate=True
            )
            total_examples = len(examples)

            # Check if mapping was successful
            from src.datasets.parsers.jsonl_parser import JSONLParser
            required_fields = JSONLParser.get_required_fields(task_type)

            if column_mapping is None and detected_mapping:
                mapped_fields = set(detected_mapping.values())
                missing_mappings = set(required_fields) - mapped_fields
                if missing_mappings:
                    warnings.append(
                        f"Could not auto-detect mapping for fields: "
                        f"{', '.join(missing_mappings)}"
                    )

        except (ValueError, csv.Error) as e:
            errors.append(str(e))

        return {
            "valid": len(errors) == 0,
            "total_examples": total_examples,
            "detected_mapping": detected_mapping,
            "errors": errors,
            "warnings": warnings,
        }

    @staticmethod
    def get_column_mapping_suggestions(
        file_path: Union[str, Path]
    ) -> dict[str, list[str]]:
        """
        Get suggested column mappings for a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary mapping standard fields to possible CSV columns
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return {}

            column_names = list(reader.fieldnames)

        suggestions = {}
        column_names_lower = [col.lower() for col in column_names]

        for standard_field, variations in (
            CSVParser.DEFAULT_COLUMN_MAPPINGS.items()
        ):
            matches = []
            for variation in variations:
                if variation in column_names_lower:
                    idx = column_names_lower.index(variation)
                    matches.append(column_names[idx])

            if matches:
                suggestions[standard_field] = matches

        return suggestions
