"""
Parquet parser with schema validation.

This module provides parsing and validation for Parquet format datasets,
with schema validation to ensure data types are compatible.

Requirement 2.8: Implement Parquet reader with schema validation
"""

from pathlib import Path
from typing import Any, Union

import pyarrow.parquet as pq

from src.data_models.dataset import DatasetTaskType


class ParquetParser:
    """Parser for Parquet format datasets with schema validation."""

    # Expected data types for standard fields
    EXPECTED_TYPES = {
        "prompt": ["string", "large_string", "utf8"],
        "completion": ["string", "large_string", "utf8"],
        "context": ["string", "large_string", "utf8"],
        "category": ["string", "large_string", "utf8"],
        "messages": ["string", "large_string", "utf8", "list"],
    }

    @staticmethod
    def parse(
        file_path: Union[str, Path],
        task_type: DatasetTaskType,
        validate: bool = True
    ) -> list[dict[str, Any]]:
        """
        Parse a Parquet file and optionally validate schema.

        Args:
            file_path: Path to the Parquet file
            task_type: Task type for validation
            validate: Whether to validate schema and required fields

        Returns:
            List of parsed rows as dictionaries

        Raises:
            ValueError: If file doesn't exist or validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        try:
            table = pq.read_table(str(file_path))
        except Exception as e:
            raise ValueError(f"Failed to read Parquet file: {e}")

        if validate:
            ParquetParser._validate_schema(table.schema, task_type)

        # Convert to list of dictionaries
        examples = []
        for batch in table.to_batches():
            batch_dict = batch.to_pydict()
            num_rows = len(batch_dict[table.schema.names[0]])

            for i in range(num_rows):
                row = {
                    col: batch_dict[col][i]
                    for col in table.schema.names
                }
                examples.append(row)

        if not examples:
            raise ValueError(f"No valid examples found in {file_path}")

        return examples

    @staticmethod
    def _validate_schema(
        schema: Any,
        task_type: DatasetTaskType
    ) -> None:
        """
        Validate Parquet schema for required fields and types.

        Args:
            schema: PyArrow schema
            task_type: Task type to validate against

        Raises:
            ValueError: If schema validation fails
        """
        # Get required fields from JSONL parser
        from src.datasets.parsers.jsonl_parser import JSONLParser

        required_fields = JSONLParser.get_required_fields(task_type)

        # Check for missing required fields
        schema_fields = {field.name for field in schema}
        missing_fields = []

        for field in required_fields:
            if field not in schema_fields:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"Missing required fields in schema for "
                f"{task_type.value}: {', '.join(missing_fields)}"
            )

        # Validate data types for known fields
        for field in schema:
            if field.name in ParquetParser.EXPECTED_TYPES:
                expected_types = ParquetParser.EXPECTED_TYPES[field.name]
                field_type = str(field.type).lower()

                # Check if field type matches any expected type
                type_match = any(
                    expected in field_type
                    for expected in expected_types
                )

                if not type_match:
                    raise ValueError(
                        f"Invalid type for field '{field.name}': "
                        f"expected one of {expected_types}, "
                        f"got {field.type}"
                    )

    @staticmethod
    def validate_file(
        file_path: Union[str, Path],
        task_type: DatasetTaskType
    ) -> dict[str, Any]:
        """
        Validate a Parquet file and return validation report.

        Args:
            file_path: Path to the Parquet file
            task_type: Task type for validation

        Returns:
            Dictionary with validation results:
            - valid: bool
            - total_examples: int
            - schema: dict with field names and types
            - errors: list of error messages
            - warnings: list of warning messages
        """
        file_path = Path(file_path)
        errors = []
        warnings = []
        total_examples = 0
        schema_info = {}

        try:
            table = pq.read_table(str(file_path))
            total_examples = table.num_rows

            # Extract schema information
            schema_info = {
                field.name: str(field.type)
                for field in table.schema
            }

            # Validate schema
            ParquetParser._validate_schema(table.schema, task_type)

            # Check for optional fields
            from src.datasets.parsers.jsonl_parser import JSONLParser
            optional_fields = JSONLParser.get_optional_fields(task_type)

            if optional_fields:
                schema_fields = set(schema_info.keys())
                missing_optional = set(optional_fields) - schema_fields

                if missing_optional:
                    warnings.append(
                        f"Optional fields not found in schema: "
                        f"{', '.join(missing_optional)}"
                    )

        except (ValueError, Exception) as e:
            errors.append(str(e))

        return {
            "valid": len(errors) == 0,
            "total_examples": total_examples,
            "schema": schema_info,
            "errors": errors,
            "warnings": warnings,
        }

    @staticmethod
    def get_schema_info(
        file_path: Union[str, Path]
    ) -> dict[str, str]:
        """
        Get schema information from a Parquet file.

        Args:
            file_path: Path to the Parquet file

        Returns:
            Dictionary mapping field names to type strings

        Raises:
            ValueError: If file doesn't exist or can't be read
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        try:
            table = pq.read_table(str(file_path))
            return {
                field.name: str(field.type)
                for field in table.schema
            }
        except Exception as e:
            raise ValueError(f"Failed to read Parquet schema: {e}")

    @staticmethod
    def validate_schema_compatibility(
        file_path: Union[str, Path],
        expected_schema: dict[str, list[str]]
    ) -> dict[str, Any]:
        """
        Validate that a Parquet file's schema is compatible with expected.

        Args:
            file_path: Path to the Parquet file
            expected_schema: Dict mapping field names to list of
                           acceptable type strings

        Returns:
            Dictionary with compatibility results:
            - compatible: bool
            - incompatible_fields: dict of field -> (actual, expected)
            - missing_fields: list of missing field names
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        try:
            actual_schema = ParquetParser.get_schema_info(file_path)
        except ValueError as e:
            return {
                "compatible": False,
                "incompatible_fields": {},
                "missing_fields": [],
                "error": str(e),
            }

        incompatible_fields = {}
        missing_fields = []

        for field_name, expected_types in expected_schema.items():
            if field_name not in actual_schema:
                missing_fields.append(field_name)
            else:
                actual_type = actual_schema[field_name].lower()
                type_match = any(
                    expected.lower() in actual_type
                    for expected in expected_types
                )

                if not type_match:
                    incompatible_fields[field_name] = (
                        actual_schema[field_name],
                        expected_types
                    )

        return {
            "compatible": (
                len(incompatible_fields) == 0 and
                len(missing_fields) == 0
            ),
            "incompatible_fields": incompatible_fields,
            "missing_fields": missing_fields,
        }
