"""
Checksum calculator for data artifact integrity verification.

Uses SHA-256 for computing and verifying checksums on data artifacts.

Requirements: 8.10
"""

import hashlib
import json
from typing import Any


def calculate_checksum(data: bytes) -> str:
    """Calculate SHA-256 checksum for raw bytes.

    Args:
        data: The raw bytes to checksum.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    return hashlib.sha256(data).hexdigest()


def calculate_string_checksum(text: str) -> str:
    """Calculate SHA-256 checksum for a string.

    Args:
        text: The string to checksum (encoded as UTF-8).

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    return calculate_checksum(text.encode("utf-8"))


def calculate_json_checksum(obj: Any) -> str:
    """Calculate SHA-256 checksum for a JSON-serializable object.

    Serializes with sorted keys for deterministic output.

    Args:
        obj: A JSON-serializable Python object.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    serialized = json.dumps(obj, sort_keys=True, default=str)
    return calculate_string_checksum(serialized)


def calculate_file_checksum(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate SHA-256 checksum for a file.

    Reads the file in chunks to handle large files efficiently.

    Args:
        file_path: Path to the file.
        chunk_size: Size of read chunks in bytes.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_checksum(data: bytes, expected: str) -> bool:
    """Verify that data matches an expected checksum.

    Args:
        data: The raw bytes to verify.
        expected: The expected hex-encoded SHA-256 digest.

    Returns:
        True if the checksum matches, False otherwise.
    """
    return calculate_checksum(data) == expected


def verify_file_checksum(file_path: str, expected: str) -> bool:
    """Verify that a file matches an expected checksum.

    Args:
        file_path: Path to the file.
        expected: The expected hex-encoded SHA-256 digest.

    Returns:
        True if the checksum matches, False otherwise.
    """
    return calculate_file_checksum(file_path) == expected
