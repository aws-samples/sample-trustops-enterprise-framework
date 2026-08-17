"""Tests for checksum verifier module."""

import pytest

from src.orchestration.checksum_calculator import calculate_checksum
from src.storage.checksum_verifier import (
    ChecksumVerification,
    calculate_data_checksum,
    verify_data_checksum,
)


class TestCalculateDataChecksum:
    def test_returns_hex_string(self):
        result = calculate_data_checksum(b"hello")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        a = calculate_data_checksum(b"test data")
        b = calculate_data_checksum(b"test data")
        assert a == b

    def test_different_data_different_checksum(self):
        a = calculate_data_checksum(b"data1")
        b = calculate_data_checksum(b"data2")
        assert a != b


class TestVerifyDataChecksum:
    def test_valid_checksum(self):
        data = b"test data"
        checksum = calculate_checksum(data)
        result = verify_data_checksum(data, checksum, "res-1")
        assert result.is_valid is True
        assert result.expected_checksum == checksum
        assert result.actual_checksum == checksum

    def test_invalid_checksum(self):
        data = b"test data"
        result = verify_data_checksum(data, "wrong_checksum", "res-1")
        assert result.is_valid is False
        assert result.result_id == "res-1"

    def test_empty_data(self):
        data = b""
        checksum = calculate_checksum(data)
        result = verify_data_checksum(data, checksum)
        assert result.is_valid is True
