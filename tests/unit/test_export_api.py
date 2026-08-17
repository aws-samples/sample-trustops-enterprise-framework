"""Tests for results export API."""

import csv
import io
import json

import pytest

from src.storage.export_api import (
    export_result,
    export_to_csv,
    export_to_json,
    export_to_pdf,
)


class TestExportToJson:
    def test_dict_export(self):
        data = {"score": 0.95, "model": "test"}
        result = export_to_json(data)
        parsed = json.loads(result)
        assert parsed["score"] == 0.95

    def test_list_export(self):
        data = [{"a": 1}, {"b": 2}]
        result = export_to_json(data)
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_returns_bytes(self):
        result = export_to_json({"x": 1})
        assert isinstance(result, bytes)


class TestExportToCsv:
    def test_dict_export(self):
        data = {"name": "test", "score": 0.9}
        result = export_to_csv(data)
        reader = csv.DictReader(io.StringIO(result.decode("utf-8")))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "test"

    def test_list_export(self):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = export_to_csv(data)
        reader = csv.DictReader(io.StringIO(result.decode("utf-8")))
        rows = list(reader)
        assert len(rows) == 2

    def test_nested_dict_flattened(self):
        data = {"metrics": {"score": 0.9}}
        result = export_to_csv(data)
        text = result.decode("utf-8")
        assert "metrics.score" in text

    def test_empty_list(self):
        result = export_to_csv([])
        assert result == b""


class TestExportToPdf:
    def test_dict_export(self):
        data = {"score": 0.95}
        result = export_to_pdf(data)
        text = result.decode("utf-8")
        assert "score: 0.95" in text
        assert "TrustOps Results Export" in text

    def test_list_export(self):
        data = [{"a": 1}]
        result = export_to_pdf(data)
        text = result.decode("utf-8")
        assert "Item 1" in text


class TestExportResult:
    def test_json_format(self):
        result = export_result({"x": 1}, "json")
        assert json.loads(result) == {"x": 1}

    def test_csv_format(self):
        result = export_result({"x": 1}, "csv")
        assert b"x" in result

    def test_pdf_format(self):
        result = export_result({"x": 1}, "pdf")
        assert b"TrustOps" in result

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported export format"):
            export_result({}, "xml")
