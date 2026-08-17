"""
Tests for CLI evaluate commands.

Tests tasks 19.12-19.13:
- `trustops evaluate baseline` command
- `trustops evaluate compare` command
"""

import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

import pytest
import yaml
from click.testing import CliRunner

from cli.main import cli


def _make_aggregate_metrics(**overrides):
    """Create a mock AggregateMetrics."""
    m = MagicMock()
    m.total_examples = overrides.get("total_examples", 100)
    m.successful_examples = overrides.get("successful_examples", 95)
    m.failed_examples = overrides.get("failed_examples", 5)
    m.mean_trust_score = overrides.get("mean_trust_score", 0.82)
    m.median_trust_score = overrides.get("median_trust_score", 0.85)
    m.trust_score_std = overrides.get("trust_score_std", 0.1)
    m.mean_hallucination_rate = overrides.get(
        "mean_hallucination_rate", 0.15
    )
    m.latency_p50_ms = overrides.get("latency_p50_ms", 250.0)
    m.latency_p95_ms = overrides.get("latency_p95_ms", 800.0)
    m.latency_p99_ms = overrides.get("latency_p99_ms", 1200.0)
    m.total_input_tokens = overrides.get("total_input_tokens", 50000)
    m.total_output_tokens = overrides.get("total_output_tokens", 30000)
    m.total_cost = overrides.get("total_cost", 1.25)
    m.cost_per_query = overrides.get("cost_per_query", 0.0125)
    return m


def _make_cost_summary():
    """Create a mock CostSummary."""
    cs = MagicMock()
    cs.total_cost = 1.25
    cs.inference_cost = 1.20
    cs.storage_cost = 0.05
    cs.cost_per_example = 0.0125
    cs.currency = "USD"
    return cs


def _make_baseline_report():
    """Create a mock BaselineEvaluationReport."""
    report = MagicMock()
    report.evaluation_id = "eval-abc123"
    report.model_id = "anthropic.claude-v2"
    report.dataset_id = "ds-001"
    report.status = "completed"
    report.total_examples = 100
    report.aggregate_metrics = _make_aggregate_metrics()
    report.cost_summary = _make_cost_summary()
    report.created_at = datetime(2024, 1, 15, 10, 0, 0)
    report.completed_at = datetime(2024, 1, 15, 10, 30, 0)
    report.per_category_metrics = {}
    return report


def _make_improvement_metrics():
    """Create a mock ImprovementMetrics."""
    imp = MagicMock()
    imp.trust_score_delta = 0.08
    imp.trust_score_delta_percent = 10.5
    imp.hallucination_reduction = 0.12
    imp.hallucination_reduction_percent = 45.0
    imp.latency_delta_ms = -50.0
    imp.latency_delta_percent = -5.0
    imp.cost_delta_per_query = 0.002
    imp.cost_delta_percent = 8.0
    imp.statistical_significance = 0.98
    imp.p_value = 0.02
    return imp


def _make_comparison_report():
    """Create a mock ComparisonReport."""
    report = MagicMock()
    report.comparison_id = "cmp-xyz789"
    report.model_1_id = "anthropic.claude-v2"
    report.model_2_id = "anthropic.claude-v2-ft"
    report.dataset_id = "ds-001"
    report.recommendation = MagicMock()
    report.recommendation.value = "deploy"
    report.recommendation_justification = (
        "All thresholds met. Trust score improved by 10.5%."
    )
    report.improvement_metrics = _make_improvement_metrics()
    report.created_at = datetime(2024, 1, 15, 11, 0, 0)
    return report


# ─── Baseline evaluation tests (Task 19.12) ───


class TestEvaluateBaseline:
    """Test `trustops evaluate baseline` command."""

    @patch("cli.commands.evaluate._get_evaluation_engine")
    def test_baseline_text_output(self, mock_get_engine):
        engine = MagicMock()
        report = _make_baseline_report()
        engine.run_baseline_evaluation = AsyncMock(
            return_value=report
        )
        mock_get_engine.return_value = engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "baseline",
            "--model", "anthropic.claude-v2",
            "--dataset", "ds-001",
        ])
        assert result.exit_code == 0
        assert "eval-abc123" in result.output
        assert "0.82" in result.output  # mean trust score
        assert "Evaluation complete" in result.output

    @patch("cli.commands.evaluate._get_evaluation_engine")
    def test_baseline_json_output(self, mock_get_engine):
        engine = MagicMock()
        report = _make_baseline_report()
        engine.run_baseline_evaluation = AsyncMock(
            return_value=report
        )
        mock_get_engine.return_value = engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "baseline",
            "--model", "anthropic.claude-v2",
            "--dataset", "ds-001",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["evaluation_id"] == "eval-abc123"
        assert data["model_id"] == "anthropic.claude-v2"
        assert data["aggregate_metrics"]["mean_trust_score"] == 0.82

    @patch("cli.commands.evaluate._get_evaluation_engine")
    def test_baseline_yaml_output(self, mock_get_engine):
        engine = MagicMock()
        report = _make_baseline_report()
        engine.run_baseline_evaluation = AsyncMock(
            return_value=report
        )
        mock_get_engine.return_value = engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "baseline",
            "--model", "anthropic.claude-v2",
            "--dataset", "ds-001",
            "--format", "yaml",
        ])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data["evaluation_id"] == "eval-abc123"

    @patch("cli.commands.evaluate._get_evaluation_engine")
    def test_baseline_failure(self, mock_get_engine):
        engine = MagicMock()
        engine.run_baseline_evaluation = AsyncMock(
            side_effect=RuntimeError("Model not found")
        )
        mock_get_engine.return_value = engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "baseline",
            "--model", "bad-model",
            "--dataset", "ds-001",
        ])
        assert result.exit_code != 0

    @patch("cli.commands.evaluate._get_evaluation_engine")
    def test_baseline_with_config_file(self, mock_get_engine, tmp_path):
        engine = MagicMock()
        report = _make_baseline_report()
        engine.run_baseline_evaluation = AsyncMock(
            return_value=report
        )
        mock_get_engine.return_value = engine

        config_file = tmp_path / "eval_config.json"
        config_file.write_text(json.dumps({
            "batch_size": 20,
            "timeout_per_request": 90.0,
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "baseline",
            "--model", "anthropic.claude-v2",
            "--dataset", "ds-001",
            "--config", str(config_file),
        ])
        assert result.exit_code == 0


# ─── Comparative evaluation tests (Task 19.13) ───


class TestEvaluateCompare:
    """Test `trustops evaluate compare` command."""

    @patch("cli.commands.evaluate._get_evaluation_engine")
    def test_compare_text_output(self, mock_get_engine):
        engine = MagicMock()
        report = _make_comparison_report()
        engine.run_comparative_evaluation = AsyncMock(
            return_value=report
        )
        mock_get_engine.return_value = engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "compare",
            "--model-1", "anthropic.claude-v2",
            "--model-2", "anthropic.claude-v2-ft",
            "--dataset", "ds-001",
        ])
        assert result.exit_code == 0
        assert "cmp-xyz789" in result.output
        assert "DEPLOY" in result.output
        assert "Comparison complete" in result.output

    @patch("cli.commands.evaluate._get_evaluation_engine")
    def test_compare_json_output(self, mock_get_engine):
        engine = MagicMock()
        report = _make_comparison_report()
        engine.run_comparative_evaluation = AsyncMock(
            return_value=report
        )
        mock_get_engine.return_value = engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "compare",
            "--model-1", "anthropic.claude-v2",
            "--model-2", "anthropic.claude-v2-ft",
            "--dataset", "ds-001",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["comparison_id"] == "cmp-xyz789"
        assert data["recommendation"] == "deploy"
        assert data["improvement_metrics"]["trust_score_delta"] == 0.08

    @patch("cli.commands.evaluate._get_evaluation_engine")
    def test_compare_yaml_output(self, mock_get_engine):
        engine = MagicMock()
        report = _make_comparison_report()
        engine.run_comparative_evaluation = AsyncMock(
            return_value=report
        )
        mock_get_engine.return_value = engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "compare",
            "--model-1", "anthropic.claude-v2",
            "--model-2", "anthropic.claude-v2-ft",
            "--dataset", "ds-001",
            "--format", "yaml",
        ])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data["comparison_id"] == "cmp-xyz789"

    @patch("cli.commands.evaluate._get_evaluation_engine")
    def test_compare_failure(self, mock_get_engine):
        engine = MagicMock()
        engine.run_comparative_evaluation = AsyncMock(
            side_effect=RuntimeError("Dataset not found")
        )
        mock_get_engine.return_value = engine

        runner = CliRunner()
        result = runner.invoke(cli, [
            "evaluate", "compare",
            "--model-1", "m1",
            "--model-2", "m2",
            "--dataset", "bad-ds",
        ])
        assert result.exit_code != 0
