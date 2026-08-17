"""Tests for the results export component."""

import json

import pytest

from dashboard.components.export import (
    _to_csv_bytes,
    _to_json_bytes,
    _to_pdf_bytes,
    _pdf_escape,
)


class TestToCsvBytes:
    """Tests for CSV export."""

    def test_empty_data(self):
        result = _to_csv_bytes([])
        assert result == b""

    def test_single_row(self):
        data = [{"name": "test", "value": 42}]
        result = _to_csv_bytes(data)
        text = result.decode("utf-8")
        assert "name" in text
        assert "value" in text
        assert "test" in text
        assert "42" in text

    def test_multiple_rows(self):
        data = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
        result = _to_csv_bytes(data)
        text = result.decode("utf-8")
        lines = text.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_returns_bytes(self):
        data = [{"x": "hello"}]
        result = _to_csv_bytes(data)
        assert isinstance(result, bytes)


class TestToJsonBytes:
    """Tests for JSON export."""

    def test_dict_data(self):
        data = {"key": "value", "num": 42}
        result = _to_json_bytes(data)
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["num"] == 42

    def test_list_data(self):
        data = [1, 2, 3]
        result = _to_json_bytes(data)
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    def test_returns_bytes(self):
        result = _to_json_bytes({"a": 1})
        assert isinstance(result, bytes)

    def test_pretty_printed(self):
        result = _to_json_bytes({"a": 1})
        text = result.decode("utf-8")
        assert "\n" in text  # pretty-printed has newlines

    def test_handles_datetime(self):
        from datetime import datetime
        data = {"ts": datetime(2024, 1, 1)}
        result = _to_json_bytes(data)
        assert b"2024" in result


class TestToPdfBytes:
    """Tests for PDF export."""

    def test_returns_bytes(self):
        result = _to_pdf_bytes({"key": "value"})
        assert isinstance(result, bytes)

    def test_starts_with_pdf_header(self):
        result = _to_pdf_bytes({"key": "value"})
        assert result.startswith(b"%PDF")

    def test_ends_with_eof(self):
        result = _to_pdf_bytes({"key": "value"})
        assert result.rstrip().endswith(b"%%EOF")

    def test_list_data(self):
        data = [{"a": 1}, {"a": 2}]
        result = _to_pdf_bytes(data)
        assert b"%PDF" in result

    def test_string_data(self):
        result = _to_pdf_bytes("simple string")
        assert b"%PDF" in result

    def test_custom_title(self):
        result = _to_pdf_bytes({"x": 1}, title="My Report")
        assert b"My Report" in result


class TestPdfEscape:
    """Tests for PDF text escaping."""

    def test_escapes_parentheses(self):
        assert "\\(" in _pdf_escape("(test)")
        assert "\\)" in _pdf_escape("(test)")

    def test_escapes_backslash(self):
        assert "\\\\" in _pdf_escape("back\\slash")

    def test_removes_newlines(self):
        result = _pdf_escape("line1\nline2")
        assert "\n" not in result

    def test_plain_text_unchanged(self):
        assert _pdf_escape("hello world") == "hello world"
