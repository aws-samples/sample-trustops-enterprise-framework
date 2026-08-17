"""
Unit tests for the checksum calculator.

Requirements: 8.10
"""

import hashlib
import os
import tempfile

from src.orchestration.checksum_calculator import (
    calculate_checksum,
    calculate_file_checksum,
    calculate_json_checksum,
    calculate_string_checksum,
    verify_checksum,
    verify_file_checksum,
)


class TestCalculateChecksum:
    def test_returns_hex_string(self):
        result = calculate_checksum(b"hello")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest length

    def test_known_value(self):
        expected = hashlib.sha256(b"hello").hexdigest()
        assert calculate_checksum(b"hello") == expected

    def test_empty_bytes(self):
        result = calculate_checksum(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_deterministic(self):
        a = calculate_checksum(b"test data")
        b = calculate_checksum(b"test data")
        assert a == b

    def test_different_data_different_checksum(self):
        a = calculate_checksum(b"data1")
        b = calculate_checksum(b"data2")
        assert a != b


class TestCalculateStringChecksum:
    def test_string_checksum(self):
        result = calculate_string_checksum("hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert result == expected

    def test_unicode_string(self):
        result = calculate_string_checksum("héllo wörld")
        assert isinstance(result, str)
        assert len(result) == 64


class TestCalculateJsonChecksum:
    def test_dict_checksum(self):
        result = calculate_json_checksum({"key": "value"})
        assert isinstance(result, str)
        assert len(result) == 64

    def test_sorted_keys(self):
        a = calculate_json_checksum({"b": 2, "a": 1})
        b = calculate_json_checksum({"a": 1, "b": 2})
        assert a == b

    def test_list_checksum(self):
        result = calculate_json_checksum([1, 2, 3])
        assert isinstance(result, str)


class TestCalculateFileChecksum:
    def test_file_checksum(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"file content")
            f.flush()
            path = f.name
        try:
            result = calculate_file_checksum(path)
            expected = hashlib.sha256(b"file content").hexdigest()
            assert result == expected
        finally:
            os.unlink(path)

    def test_large_file(self):
        data = b"x" * 100_000
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            f.flush()
            path = f.name
        try:
            result = calculate_file_checksum(path, chunk_size=1024)
            expected = hashlib.sha256(data).hexdigest()
            assert result == expected
        finally:
            os.unlink(path)


class TestVerifyChecksum:
    def test_valid_checksum(self):
        data = b"verify me"
        checksum = calculate_checksum(data)
        assert verify_checksum(data, checksum) is True

    def test_invalid_checksum(self):
        assert verify_checksum(b"data", "wrong") is False


class TestVerifyFileChecksum:
    def test_valid_file_checksum(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"file data")
            f.flush()
            path = f.name
        try:
            checksum = calculate_file_checksum(path)
            assert verify_file_checksum(path, checksum) is True
        finally:
            os.unlink(path)

    def test_invalid_file_checksum(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"file data")
            f.flush()
            path = f.name
        try:
            assert verify_file_checksum(path, "wrong") is False
        finally:
            os.unlink(path)
