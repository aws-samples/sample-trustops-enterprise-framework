"""
Checksum calculator and verifier for stored results.

Calculates SHA-256 checksums on store and verifies on retrieve.

Requirements: 9.10
"""

import logging
from typing import Any

from pydantic import BaseModel

from src.orchestration.checksum_calculator import calculate_checksum

logger = logging.getLogger(__name__)


class ChecksumVerification(BaseModel):
    """Result of a checksum verification."""

    result_id: str
    expected_checksum: str
    actual_checksum: str
    is_valid: bool


def calculate_data_checksum(data: bytes) -> str:
    """Calculate SHA-256 checksum for data.

    Args:
        data: Raw bytes to checksum.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return calculate_checksum(data)


def verify_data_checksum(
    data: bytes,
    expected_checksum: str,
    result_id: str = "",
) -> ChecksumVerification:
    """Verify data integrity by comparing checksums.

    Args:
        data: Raw bytes to verify.
        expected_checksum: Expected SHA-256 hex digest.
        result_id: Optional result ID for logging.

    Returns:
        ChecksumVerification with verification status.
    """
    actual = calculate_checksum(data)
    is_valid = actual == expected_checksum

    if not is_valid:
        logger.warning(
            "Checksum mismatch for %s: expected=%s, actual=%s",
            result_id, expected_checksum, actual,
        )
    else:
        logger.debug("Checksum verified for %s", result_id)

    return ChecksumVerification(
        result_id=result_id,
        expected_checksum=expected_checksum,
        actual_checksum=actual,
        is_valid=is_valid,
    )
