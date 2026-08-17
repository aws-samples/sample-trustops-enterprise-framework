"""
Tests for CLI results commands.

Tests tasks 19.22-19.24:
- `trustops results get`
- `trustops results export`
- `trustops results compare`
"""

import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

import pytest
import yaml
from click.testing import CliRunner

from cli.main import cli


def _make_eval_result():
    """Create a mock evaluation result."""
    result = MagicMock()
    result.evaluation_id = "eval-abc123"
    result.model_id = "anthropic.claude-v2"
    result.dataset_id = "ds-001"
    result.status = "completed"
    result.total_examples = 100

    m = MagicMock()
    m.mean_trust_score = 0.82
    m.median_trust_score = 0.85
    m.mean_hallucination_rate = 0.15
    m.total_cost = 1.25
    m.cost_per_query = 0.0125
    m.latency_p50_ms = 250.0
    m.latency_p95_ms = 800.0
    result.aggregate_metrics = m
    result.created_at = datetime(2024, 1, 15, 10, 0, 0)
    return result


# ─── results get tests (Task 19.22) ───


class TestResultsGet:
    """Test `trustops results get` command."""

    @patch("cli.commands.results._get_results_store")
    def test_get_text_output(self, mock_get_store):
        store = MagicMock()
        store.get = AsyncMock(return_value=_make_eval_result())
        mock_get_store.return_value = store

        runner = CliRunner()
        result = runner.invoke(cli, [
            "results", "get", "eval-abc123"
        ])
        assert result.exit_code == 0
        assert "eval-abc123" in result.output

    @patch("cli.commands.results._get_results_store")
    def test_get_json_output(self, mock_get_store):
        store = MagicMock()
        store.get = AsyncMock(return_value=_make_eval_result())
        mock_get_store.return_value = store

        runner = CliRunner()
        result = runner.invoke(cli, [
            "results", "get", "eval-abc123",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["evaluation_id"] == "eval-abc123"
        assert data["aggregate_metrics"]["mean_trust_score"] == 0.82

    @patch("cli.commands.results._get_results_store")
    def test_get_yaml_output(self, mock_get_store):
        store = MagicMock()
        store.get = AsyncMock(return_value=_make_eval_result())
        mock_get_store.return_value = store

        runner = CliRunner()
        result = runner.invoke(cli, [
            "results", "get", "eval-abc123",
            "--format", "yaml",
        ])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data["evaluation_id"] == "eval-abc123"

    @patch("cli.commands.results._get_results_store")
    def test_get_not_found(self, mock_get_store):
        store = MagicMock()
        store.get = AsyncMock(return_value=None)
        mock_get_store.return_value = store

        runner = CliRunner()
        result = runner.invoke(cli, [
            "results", "get", "eval-nonexistent"
        ])
        assert result.exit_code != 0


# ─── results export tests (Task 19.23) ───


class TestResultsExport:
    """Test `trustops results export` command."""

    @patch("cli.commands.results._get_results_store")
    def test_export_text_output(self, mock_get_store):
        store = MagicMock()
        store.export = AsyncMock(
            return_value="/tmp/eval-abc123.json"
        )
        mock_get_store.return_value = store

        runner = CliRunner()
        result = runner.invoke(cli, [
            "results", "export", "eval-abc123",
            "--export-format", "json",
        ])
        assert result.exit_code == 0
        assert "exported" in result.output.lower()

    @patch("cli.commands.results._get_results_store")
    def test_export_json_output(self, mock_get_store):
        store = MagicMock()
        store.export = AsyncMock(
            return_value="/tmp/eval-abc123.csv"
        )
        mock_get_store.return_value = store

        runner = CliRunner()
        result = runner.invoke(cli, [
            "results", "export", "eval-abc123",
            "--export-format", "csv",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["evaluation_id"] == "eval-abc123"
        assert data["format"] == "csv"
        assert data["status"] == "exported"

    @patch("cli.commands.results._get_results_store")
    def test_export_with_output_path(self, mock_get_store):
        store = MagicMock()
        store.export = AsyncMock(return_value="/custom/path.pdf")
        mock_get_store.return_value = store

        runner = CliRunner()
        result = runner.invoke(cli, [
            "results", "export", "eval-abc123",
            "--export-format", "pdf",
            "--output", "/custom/path.pdf",
        ])
        assert result.exit_code == 0


# ─── results compare tests (Task 19.24) ───


class TestResultsCompare:
    """Test `trustops results compare` command."""

    @patch("cli.commands.results._get_results_store")
    def test_compare_text_output(self, mock_get_store):
        store = MagicMock()
        store.compare = AsyncMock(return_value={
            "evaluation_1": "eval-001",
            "evaluation_2": "eval-002",
            "trust_score_delta": 0.08,
            "cost_delta": 0.002,
        })
        mock_get_store.return_value = store

        runner = CliRunner()
        result = runner.invoke(cli, [
            "results", "compare", "eval-001", "eval-002"
        ])
        assert result.exit_code == 0
        assert "eval-001" in result.output
        assert "eval-002" in result.output

    @patch("cli.commands.results._get_results_store")
    def test_compare_json_output(self, mock_get_store):
        store = MagicMock()
        comparison_data = {
            "evaluation_1": "eval-001",
            "evaluation_2": "eval-002",
            "trust_score_delta": 0.08,
        }
        store.compare = AsyncMock(return_value=comparison_data)
        mock_get_store.return_value = store

        runner = CliRunner()
        result = runner.invoke(cli, [
            "results", "compare", "eval-001", "eval-002",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["trust_score_delta"] == 0.08

    @patch("cli.commands.results._get_results_store")
    def test_compare_yaml_output(self, mock_get_store):
        store = MagicMock()
        comparison_data = {
            "evaluation_1": "eval-001",
            "evaluation_2": "eval-002",
            "trust_score_delta": 0.08,
        }
        store.compare = AsyncMock(return_value=comparison_data)
        mock_get_store.return_value = store

        runner = CliRunner()
        result = runner.invoke(cli, [
            "results", "compare", "eval-001", "eval-002",
            "--format", "yaml",
        ])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data["trust_score_delta"] == 0.08
