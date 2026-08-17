"""
Tests for CLI finetune v2 commands (start/status/stop).

Tests tasks 19.14-19.16:
- `trustops finetune start`
- `trustops finetune status`
- `trustops finetune stop`
"""

import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

import pytest
import yaml
from click.testing import CliRunner

from cli.main import cli


def _make_cost_estimate():
    """Create a mock CostEstimate."""
    est = MagicMock()
    est.estimated_training_cost = 25.50
    est.estimated_duration_hours = 2.5
    est.cost_breakdown = {"compute": 20.0, "storage": 5.50}
    est.currency = "USD"
    est.confidence = "medium"
    return est


def _make_ft_job(**overrides):
    """Create a mock FineTuningJob."""
    job = MagicMock()
    job.job_id = overrides.get("job_id", "ft-abc123")
    job.status = MagicMock()
    job.status.value = overrides.get("status", "training")
    job.base_model_id = overrides.get(
        "base_model_id", "anthropic.claude-v2"
    )
    job.finetuned_model_id = overrides.get(
        "finetuned_model_id", None
    )
    job.estimated_cost = overrides.get("estimated_cost", 25.50)
    job.actual_cost = overrides.get("actual_cost", None)
    job.created_at = overrides.get(
        "created_at", datetime(2024, 1, 15, 10, 0, 0)
    )
    job.completed_at = overrides.get("completed_at", None)
    job.error_message = overrides.get("error_message", None)

    # Training metrics
    metric = MagicMock()
    metric.epoch = 2
    metric.step = 150
    metric.training_loss = 0.45
    metric.validation_loss = 0.52
    metric.learning_rate = 0.0001
    job.training_metrics = overrides.get(
        "training_metrics", [metric]
    )

    hp = MagicMock()
    hp.learning_rate = 0.0001
    hp.epochs = 3
    hp.batch_size = 8
    job.hyperparameters = hp
    return job


# ─── finetune start tests (Task 19.14) ───


class TestFinetuneStart:
    """Test `trustops finetune start` command."""

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_start_text_output(self, mock_get_pipeline):
        pipeline = MagicMock()
        pipeline.estimate_cost = MagicMock(
            return_value=_make_cost_estimate()
        )
        pipeline.start = MagicMock(
            return_value=_make_ft_job()
        )
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "start",
            "--model", "anthropic.claude-v2",
            "--dataset", "ds-001",
        ])
        assert result.exit_code == 0
        assert "ft-abc123" in result.output
        assert "Fine-tuning job started" in result.output

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_start_json_output(self, mock_get_pipeline):
        pipeline = MagicMock()
        pipeline.estimate_cost = MagicMock(
            return_value=_make_cost_estimate()
        )
        pipeline.start = MagicMock(
            return_value=_make_ft_job()
        )
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "start",
            "--model", "anthropic.claude-v2",
            "--dataset", "ds-001",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["job_id"] == "ft-abc123"
        assert "cost_estimate" in data
        assert data["cost_estimate"]["estimated_training_cost"] == 25.50

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_start_yaml_output(self, mock_get_pipeline):
        pipeline = MagicMock()
        pipeline.estimate_cost = MagicMock(
            return_value=_make_cost_estimate()
        )
        pipeline.start = MagicMock(
            return_value=_make_ft_job()
        )
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "start",
            "--model", "anthropic.claude-v2",
            "--dataset", "ds-001",
            "--format", "yaml",
        ])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data["job_id"] == "ft-abc123"

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_start_with_config(self, mock_get_pipeline, tmp_path):
        pipeline = MagicMock()
        pipeline.estimate_cost = MagicMock(
            return_value=_make_cost_estimate()
        )
        pipeline.start = MagicMock(
            return_value=_make_ft_job()
        )
        mock_get_pipeline.return_value = pipeline

        config_file = tmp_path / "ft_config.json"
        config_file.write_text(json.dumps({
            "job_name": "my-custom-job",
            "output_model_name": "my-model-ft",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "start",
            "--model", "anthropic.claude-v2",
            "--dataset", "ds-001",
            "--config", str(config_file),
        ])
        assert result.exit_code == 0

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_start_failure(self, mock_get_pipeline):
        pipeline = MagicMock()
        pipeline.estimate_cost = MagicMock(
            side_effect=RuntimeError("Model not eligible")
        )
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "start",
            "--model", "bad-model",
            "--dataset", "ds-001",
        ])
        assert result.exit_code != 0


# ─── finetune status tests (Task 19.15) ───


class TestFinetuneStatus:
    """Test `trustops finetune status` command."""

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_status_text_output(self, mock_get_pipeline):
        pipeline = MagicMock()
        pipeline.get_status = MagicMock(
            return_value=_make_ft_job()
        )
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "status", "ft-abc123"
        ])
        assert result.exit_code == 0
        assert "ft-abc123" in result.output
        assert "training" in result.output

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_status_json_output(self, mock_get_pipeline):
        pipeline = MagicMock()
        pipeline.get_status = MagicMock(
            return_value=_make_ft_job()
        )
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "status", "ft-abc123",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["job_id"] == "ft-abc123"
        assert data["status"] == "training"
        assert "training_metrics" in data

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_status_completed_job(self, mock_get_pipeline):
        pipeline = MagicMock()
        job = _make_ft_job(
            status="completed",
            finetuned_model_id="ft-model-001",
            actual_cost=22.30,
            completed_at=datetime(2024, 1, 15, 12, 30, 0),
        )
        pipeline.get_status = MagicMock(return_value=job)
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "status", "ft-abc123"
        ])
        assert result.exit_code == 0
        assert "completed" in result.output

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_status_not_found(self, mock_get_pipeline):
        pipeline = MagicMock()
        pipeline.get_status = MagicMock(return_value=None)
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "status", "ft-nonexistent"
        ])
        assert result.exit_code != 0


# ─── finetune stop tests (Task 19.16) ───


class TestFinetuneStop:
    """Test `trustops finetune stop` command."""

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_stop_text_output(self, mock_get_pipeline):
        pipeline = MagicMock()
        job = _make_ft_job(status="stopped")
        pipeline.stop = MagicMock(return_value=job)
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "stop", "ft-abc123"
        ])
        assert result.exit_code == 0
        assert "stopped" in result.output

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_stop_json_output(self, mock_get_pipeline):
        pipeline = MagicMock()
        job = _make_ft_job(status="stopped")
        pipeline.stop = MagicMock(return_value=job)
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "stop", "ft-abc123",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["job_id"] == "ft-abc123"
        assert data["status"] == "stopped"

    @patch("cli.commands.finetune_v2._get_pipeline")
    def test_stop_failure(self, mock_get_pipeline):
        pipeline = MagicMock()
        pipeline.stop = MagicMock(
            side_effect=RuntimeError("Job not running")
        )
        mock_get_pipeline.return_value = pipeline

        runner = CliRunner()
        result = runner.invoke(cli, [
            "finetune", "stop", "ft-abc123"
        ])
        assert result.exit_code != 0
