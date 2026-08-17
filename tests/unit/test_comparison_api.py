"""Tests for results comparison API."""

import pytest

from src.storage.comparison_api import ComparisonResult, compare_results, _flatten_dict


class TestFlattenDict:
    def test_flat_dict(self):
        assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = _flatten_dict({"a": {"b": 1, "c": 2}})
        assert result == {"a.b": 1, "a.c": 2}

    def test_deeply_nested(self):
        result = _flatten_dict({"a": {"b": {"c": 3}}})
        assert result == {"a.b.c": 3}

    def test_empty_dict(self):
        assert _flatten_dict({}) == {}


class TestCompareResults:
    def test_identical_results(self):
        r = {"score": 0.9, "latency": 100}
        result = compare_results(r, r, "r1", "r2")
        assert result.result_id_1 == "r1"
        assert result.result_id_2 == "r2"
        assert len(result.only_in_1) == 0
        assert len(result.only_in_2) == 0

    def test_numeric_diff(self):
        r1 = {"score": 0.8}
        r2 = {"score": 0.9}
        result = compare_results(r1, r2)
        assert "score" in result.diff
        assert result.diff["score"]["delta"] == pytest.approx(0.1)
        assert result.diff["score"]["delta_percent"] == pytest.approx(12.5)

    def test_different_keys(self):
        r1 = {"a": 1, "b": 2}
        r2 = {"b": 2, "c": 3}
        result = compare_results(r1, r2)
        assert "a" in result.only_in_1
        assert "c" in result.only_in_2
        assert "b" in result.common_keys

    def test_nested_comparison(self):
        r1 = {"metrics": {"accuracy": 0.8}}
        r2 = {"metrics": {"accuracy": 0.9}}
        result = compare_results(r1, r2)
        assert "metrics.accuracy" in result.diff

    def test_string_change(self):
        r1 = {"status": "pending"}
        r2 = {"status": "completed"}
        result = compare_results(r1, r2)
        assert result.diff["status"]["changed"] is True

    def test_zero_division_handled(self):
        r1 = {"score": 0}
        r2 = {"score": 5}
        result = compare_results(r1, r2)
        assert result.diff["score"]["delta_percent"] == 0.0
