"""
PII detector for datasets.

This module provides PII (Personally Identifiable Information) detection
and masking for datasets using regex patterns to identify common PII types.

Requirements: 2.7, 2.13, 2.14
"""

import re
from collections import defaultdict
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PIIType(str, Enum):
    """Enumeration of PII types that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"


class MaskingStrategy(str, Enum):
    """Enumeration of PII masking strategies."""

    MASK = "mask"  # Replace with [PII_TYPE] placeholder
    REDACT = "redact"  # Remove the PII text (replace with empty string)
    REMOVE = "remove"  # Delete entire rows containing PII


class PIIMatch(BaseModel):
    """A single PII match found in the dataset."""

    pii_type: PIIType = Field(..., description="Type of PII detected")
    value: str = Field(..., description="The matched PII value")
    field: str = Field(..., description="Field name where PII was found")
    start_idx: int = Field(..., ge=0, description="Start index in the text")
    end_idx: int = Field(..., ge=0, description="End index in the text")


class PIIRowResult(BaseModel):
    """PII detection result for a single row."""

    row_index: int = Field(..., ge=0, description="Index of the row")
    has_pii: bool = Field(..., description="Whether PII was found in this row")
    pii_types: list[PIIType] = Field(
        default_factory=list, description="Types of PII found"
    )
    matches: list[PIIMatch] = Field(
        default_factory=list, description="Detailed PII matches"
    )


class PIIDetectionResult(BaseModel):
    """Complete PII detection result for a dataset."""

    total_rows: int = Field(..., ge=0, description="Total number of rows scanned")
    rows_with_pii: int = Field(
        ..., ge=0, description="Number of rows containing PII"
    )
    pii_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage of rows with PII"
    )
    pii_type_counts: dict[PIIType, int] = Field(
        default_factory=dict, description="Count of each PII type found"
    )
    affected_rows: list[int] = Field(
        default_factory=list, description="Indices of rows containing PII"
    )
    row_results: list[PIIRowResult] = Field(
        default_factory=list, description="Detailed results per row"
    )


class MaskingReport(BaseModel):
    """Report of PII masking operations performed on a dataset."""

    strategy: MaskingStrategy = Field(
        ..., description="Masking strategy used"
    )
    original_row_count: int = Field(
        ..., ge=0, description="Number of rows before masking"
    )
    masked_row_count: int = Field(
        ..., ge=0, description="Number of rows after masking"
    )
    rows_removed: int = Field(
        ..., ge=0, description="Number of rows removed (for REMOVE strategy)"
    )
    pii_instances_masked: int = Field(
        ..., ge=0, description="Total number of PII instances masked/redacted"
    )
    pii_types_masked: dict[PIIType, int] = Field(
        default_factory=dict,
        description="Count of each PII type masked"
    )
    selective_types: list[PIIType] | None = Field(
        None, description="PII types that were selectively masked (None = all)"
    )


class PIIDetector:
    """Detects PII in datasets using regex patterns."""

    # Regex patterns for different PII types
    PATTERNS = {
        PIIType.EMAIL: re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ),
        PIIType.PHONE: re.compile(
            r'\b(?:\+?1[-.\s]?)?'  # Optional country code
            r'(?:\(\d{3}\)|\d{3})[-.\s]?'  # Area code
            r'\d{3}[-.\s]?\d{4}\b'  # Number
        ),
        PIIType.SSN: re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b'  # Format: XXX-XX-XXXX
        ),
        PIIType.CREDIT_CARD: re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b'  # 16 digits with optional separators
        ),
        PIIType.IP_ADDRESS: re.compile(
            # IPv4
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b|'
            # IPv6 (simplified pattern)
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|'
            r'\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|'
            r'\b::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b'
        ),
    }

    def __init__(self):
        """Initialize the PII detector."""
        pass

    def detect(self, examples: list[dict[str, Any]]) -> PIIDetectionResult:
        """
        Detect PII in a list of dataset examples.

        Args:
            examples: List of dataset examples (dictionaries)

        Returns:
            PIIDetectionResult with detailed PII detection information

        Raises:
            ValueError: If examples list is empty
        """
        if not examples:
            raise ValueError("Cannot detect PII in empty dataset")

        row_results = []
        affected_rows = []
        pii_type_counts = defaultdict(int)

        for row_idx, example in enumerate(examples):
            row_result = self._detect_in_row(row_idx, example)
            row_results.append(row_result)

            if row_result.has_pii:
                affected_rows.append(row_idx)
                for pii_type in row_result.pii_types:
                    pii_type_counts[pii_type] += 1

        total_rows = len(examples)
        rows_with_pii = len(affected_rows)
        pii_percentage = (rows_with_pii / total_rows * 100) if total_rows > 0 else 0.0

        return PIIDetectionResult(
            total_rows=total_rows,
            rows_with_pii=rows_with_pii,
            pii_percentage=round(pii_percentage, 2),
            pii_type_counts=dict(pii_type_counts),
            affected_rows=affected_rows,
            row_results=row_results,
        )

    def _detect_in_row(
        self, row_idx: int, example: dict[str, Any]
    ) -> PIIRowResult:
        """
        Detect PII in a single row.

        Args:
            row_idx: Index of the row
            example: Dictionary containing the row data

        Returns:
            PIIRowResult with PII detection information for this row
        """
        matches = []
        pii_types_found = set()

        # Scan all text fields in the example
        for field_name, field_value in example.items():
            if isinstance(field_value, str):
                field_matches = self._detect_in_text(field_name, field_value)
                matches.extend(field_matches)
                for match in field_matches:
                    pii_types_found.add(match.pii_type)

        return PIIRowResult(
            row_index=row_idx,
            has_pii=len(matches) > 0,
            pii_types=sorted(list(pii_types_found)),
            matches=matches,
        )

    def _detect_in_text(self, field_name: str, text: str) -> list[PIIMatch]:
        """
        Detect PII in a text string.

        Args:
            field_name: Name of the field being scanned
            text: Text content to scan

        Returns:
            List of PIIMatch objects for all PII found in the text
        """
        matches = []

        for pii_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                # Additional validation for credit cards (Luhn algorithm)
                if pii_type == PIIType.CREDIT_CARD:
                    card_number = match.group().replace('-', '').replace(' ', '')
                    if not self._validate_luhn(card_number):
                        continue

                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        value=match.group(),
                        field=field_name,
                        start_idx=match.start(),
                        end_idx=match.end(),
                    )
                )

        return matches

    def _validate_luhn(self, card_number: str) -> bool:
        """
        Validate credit card number using Luhn algorithm.

        Args:
            card_number: Credit card number string (digits only)

        Returns:
            True if valid according to Luhn algorithm, False otherwise
        """
        if not card_number.isdigit():
            return False

        # Luhn algorithm
        digits = [int(d) for d in card_number]
        checksum = 0

        # Process from right to left
        for i in range(len(digits) - 1, -1, -1):
            digit = digits[i]

            # Double every second digit from the right
            position_from_right = len(digits) - i
            if position_from_right % 2 == 0:
                digit *= 2
                if digit > 9:
                    digit -= 9

            checksum += digit

        return checksum % 10 == 0

    def mask(
        self,
        examples: list[dict[str, Any]],
        strategy: MaskingStrategy = MaskingStrategy.MASK,
        selective_types: list[PIIType] | None = None,
    ) -> tuple[list[dict[str, Any]], MaskingReport]:
        """
        Mask or redact PII in dataset examples.

        Args:
            examples: List of dataset examples (dictionaries)
            strategy: Masking strategy to use (MASK, REDACT, or REMOVE)
            selective_types: Optional list of PII types to mask.
                If None, all PII types are masked.

        Returns:
            Tuple of (masked_examples, masking_report)

        Raises:
            ValueError: If examples list is empty
        """
        if not examples:
            raise ValueError("Cannot mask PII in empty dataset")

        # First detect all PII
        detection_result = self.detect(examples)

        # Track masking statistics
        pii_instances_masked = 0
        pii_types_masked = defaultdict(int)
        masked_examples = []

        # Apply masking strategy
        if strategy == MaskingStrategy.REMOVE:
            # Remove rows containing PII
            for idx, example in enumerate(examples):
                row_result = detection_result.row_results[idx]

                # Check if row has PII of the types we're masking
                if self._should_mask_row(row_result, selective_types):
                    # Skip this row (remove it)
                    for match in row_result.matches:
                        if selective_types is None or match.pii_type in selective_types:
                            pii_instances_masked += 1
                            pii_types_masked[match.pii_type] += 1
                else:
                    # Keep this row
                    masked_examples.append(example.copy())
        else:
            # MASK or REDACT strategies - modify text in place
            for idx, example in enumerate(examples):
                row_result = detection_result.row_results[idx]
                masked_example = self._mask_row(
                    example, row_result, strategy, selective_types
                )
                masked_examples.append(masked_example)

                # Count masked instances
                for match in row_result.matches:
                    if selective_types is None or match.pii_type in selective_types:
                        pii_instances_masked += 1
                        pii_types_masked[match.pii_type] += 1

        # Create masking report
        report = MaskingReport(
            strategy=strategy,
            original_row_count=len(examples),
            masked_row_count=len(masked_examples),
            rows_removed=len(examples) - len(masked_examples),
            pii_instances_masked=pii_instances_masked,
            pii_types_masked=dict(pii_types_masked),
            selective_types=selective_types,
        )

        return masked_examples, report

    def _should_mask_row(
        self,
        row_result: PIIRowResult,
        selective_types: list[PIIType] | None,
    ) -> bool:
        """
        Determine if a row should be masked based on selective types.

        Args:
            row_result: PII detection result for the row
            selective_types: Optional list of PII types to mask

        Returns:
            True if row contains PII that should be masked
        """
        if not row_result.has_pii:
            return False

        if selective_types is None:
            # Mask all PII
            return True

        # Check if row has any of the selective types
        for pii_type in row_result.pii_types:
            if pii_type in selective_types:
                return True

        return False

    def _mask_row(
        self,
        example: dict[str, Any],
        row_result: PIIRowResult,
        strategy: MaskingStrategy,
        selective_types: list[PIIType] | None,
    ) -> dict[str, Any]:
        """
        Mask PII in a single row.

        Args:
            example: Original example dictionary
            row_result: PII detection result for this row
            strategy: Masking strategy (MASK or REDACT)
            selective_types: Optional list of PII types to mask

        Returns:
            Masked example dictionary
        """
        masked_example = {}

        for field_name, field_value in example.items():
            if isinstance(field_value, str):
                # Get matches for this field
                field_matches = [
                    m for m in row_result.matches
                    if m.field == field_name
                    and (selective_types is None or m.pii_type in selective_types)
                ]

                if field_matches:
                    # Apply masking to this field
                    masked_value = self._mask_text(
                        field_value, field_matches, strategy
                    )
                    masked_example[field_name] = masked_value
                else:
                    # No PII in this field, keep original
                    masked_example[field_name] = field_value
            else:
                # Non-string field, keep original
                masked_example[field_name] = field_value

        return masked_example

    def _mask_text(
        self,
        text: str,
        matches: list[PIIMatch],
        strategy: MaskingStrategy,
    ) -> str:
        """
        Mask PII in a text string.

        Args:
            text: Original text
            matches: List of PII matches in this text
            strategy: Masking strategy (MASK or REDACT)

        Returns:
            Masked text
        """
        # Sort matches by start index in reverse order
        # This allows us to replace from end to start without
        # invalidating indices
        sorted_matches = sorted(matches, key=lambda m: m.start_idx, reverse=True)

        masked_text = text
        for match in sorted_matches:
            if strategy == MaskingStrategy.MASK:
                # Replace with [PII_TYPE] placeholder
                replacement = f"[{match.pii_type.value.upper()}]"
            else:  # REDACT
                # Replace with empty string
                replacement = ""

            # Replace the PII value
            masked_text = (
                masked_text[:match.start_idx]
                + replacement
                + masked_text[match.end_idx:]
            )

        return masked_text


def detect_pii_in_file(file_path: str) -> PIIDetectionResult:
    """
    Detect PII in a dataset file.

    This is a convenience function for detecting PII in JSONL files.

    Args:
        file_path: Path to the dataset file (JSONL format)

    Returns:
        PIIDetectionResult with detection information

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    import json
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num}: {e}"
                ) from e

    detector = PIIDetector()
    return detector.detect(examples)
