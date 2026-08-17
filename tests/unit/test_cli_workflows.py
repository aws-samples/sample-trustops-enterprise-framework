"""
Tests for CLI workflow commands.

Tests tasks 19.17-19.21:
- `trustops workflows list`
- `trustops workflows run`
- `trustops workflows status`
- `trustops workflows resume`
- `trustops workflows reproduce`
"""

import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

import pytest
import yaml
from click.testing import CliRunner

from cli.main import cli


def _make_workflow(**overrides):
    """Create a mock workflow object."""
    wf = MagicMock()
    wf.workflow_id = overrides.get("workflow_id", "wf-abc123")
    wf.status = MagicMock()
    wf.status.value = overrides.get("status", "completed")
    wf.workflow_type = overrides.get("workflow_type", "full_pipeline")
    wf.created_at = overrides.get(
        "created_at", datetime(2024, 1, 15, 10, 0, 0)
    )
    wf.completed_at = overrides.get(
        "completed_at", datetime(2024, 1, 15, 12, 0, 0)
    )
    wf.total_cost = overrides.get("total_cost", 5.50)
    wf.estimated_cost = overrides.get("estimated_cost", 6.00)

    step1 = MagicMock()
    step1.name = "baseline_evaluation"
    step1.status = MagicMock()
    step1.status.value = "completed"

    step2 = MagicMock()
    step2.name = "fine_tuning"
    step2.status = MagicMock()
    step2.status.value = "completed"

    wf.steps = overrides.get("steps", [step1, step2])
    return wf


# ─── workflows list tests (Task 19.17) ───


class TestWorkflowsList:
    """Test `trustops workflows list` command."""

    @patch("cli.commands.workflows._get_orchestrator")
    def test_list_text_output(self, mock_get_orch):
        orch = MagicMock()
        orch.list_workflows = AsyncMock(
            return_value=[_make_workflow(), _make_workflow(
                workflow_id="wf-def456", status="running"
            )]
        )
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, ["workflows", "list"])
        assert result.exit_code == 0
        assert "wf-abc123" in result.output
        assert "2 workflow(s)" in result.output

    @patch("cli.commands.workflows._get_orchestrator")
    def test_list_json_output(self, mock_get_orch):
        orch = MagicMock()
        orch.list_workflows = AsyncMock(
            return_value=[_make_workflow()]
        )
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "list", "--format", "json"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["workflow_id"] == "wf-abc123"

    @patch("cli.commands.workflows._get_orchestrator")
    def test_list_yaml_output(self, mock_get_orch):
        orch = MagicMock()
        orch.list_workflows = AsyncMock(
            return_value=[_make_workflow()]
        )
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "list", "--format", "yaml"
        ])
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data[0]["workflow_id"] == "wf-abc123"

    @patch("cli.commands.workflows._get_orchestrator")
    def test_list_with_status_filter(self, mock_get_orch):
        orch = MagicMock()
        orch.list_workflows = AsyncMock(return_value=[])
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "list", "--status", "running"
        ])
        assert result.exit_code == 0
        orch.list_workflows.assert_called_once_with(
            status="running"
        )

    @patch("cli.commands.workflows._get_orchestrator")
    def test_list_empty(self, mock_get_orch):
        orch = MagicMock()
        orch.list_workflows = AsyncMock(return_value=[])
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, ["workflows", "list"])
        assert result.exit_code == 0
        assert "No workflows found" in result.output


# ─── workflows run tests (Task 19.18) ───


class TestWorkflowsRun:
    """Test `trustops workflows run` command."""

    @patch("cli.commands.workflows._get_orchestrator")
    def test_run_text_output(self, mock_get_orch):
        orch = MagicMock()
        wf = _make_workflow(status="running")
        orch.run_workflow = AsyncMock(return_value=wf)
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "run", "full_pipeline"
        ])
        assert result.exit_code == 0
        assert "wf-abc123" in result.output
        assert "Workflow started" in result.output

    @patch("cli.commands.workflows._get_orchestrator")
    def test_run_json_output(self, mock_get_orch):
        orch = MagicMock()
        wf = _make_workflow(status="running")
        orch.run_workflow = AsyncMock(return_value=wf)
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "run", "full_pipeline",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["workflow_id"] == "wf-abc123"
        assert data["status"] == "running"

    @patch("cli.commands.workflows._get_orchestrator")
    def test_run_with_config(self, mock_get_orch, tmp_path):
        orch = MagicMock()
        wf = _make_workflow(status="running")
        orch.run_workflow = AsyncMock(return_value=wf)
        mock_get_orch.return_value = orch

        config_file = tmp_path / "wf_config.json"
        config_file.write_text(json.dumps({
            "model_id": "anthropic.claude-v2",
            "dataset_id": "ds-001",
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "run", "full_pipeline",
            "--config", str(config_file),
        ])
        assert result.exit_code == 0


# ─── workflows status tests (Task 19.19) ───


class TestWorkflowsStatus:
    """Test `trustops workflows status` command."""

    @patch("cli.commands.workflows._get_orchestrator")
    def test_status_text_output(self, mock_get_orch):
        orch = MagicMock()
        wf = _make_workflow()
        orch.get_workflow_status = AsyncMock(return_value=wf)
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "status", "wf-abc123"
        ])
        assert result.exit_code == 0
        assert "wf-abc123" in result.output
        assert "completed" in result.output
        assert "baseline_evaluation" in result.output

    @patch("cli.commands.workflows._get_orchestrator")
    def test_status_json_output(self, mock_get_orch):
        orch = MagicMock()
        wf = _make_workflow()
        orch.get_workflow_status = AsyncMock(return_value=wf)
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "status", "wf-abc123",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["workflow_id"] == "wf-abc123"
        assert len(data["steps"]) == 2

    @patch("cli.commands.workflows._get_orchestrator")
    def test_status_not_found(self, mock_get_orch):
        orch = MagicMock()
        orch.get_workflow_status = AsyncMock(return_value=None)
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "status", "wf-nonexistent"
        ])
        assert result.exit_code != 0


# ─── workflows resume tests (Task 19.20) ───


class TestWorkflowsResume:
    """Test `trustops workflows resume` command."""

    @patch("cli.commands.workflows._get_orchestrator")
    def test_resume_text_output(self, mock_get_orch):
        orch = MagicMock()
        wf = _make_workflow(status="running")
        orch.resume_workflow = AsyncMock(return_value=wf)
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "resume", "wf-abc123"
        ])
        assert result.exit_code == 0
        assert "resumed" in result.output

    @patch("cli.commands.workflows._get_orchestrator")
    def test_resume_json_output(self, mock_get_orch):
        orch = MagicMock()
        wf = _make_workflow(status="running")
        orch.resume_workflow = AsyncMock(return_value=wf)
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "resume", "wf-abc123",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["workflow_id"] == "wf-abc123"


# ─── workflows reproduce tests (Task 19.21) ───


class TestWorkflowsReproduce:
    """Test `trustops workflows reproduce` command."""

    @patch("cli.commands.workflows._get_orchestrator")
    def test_reproduce_text_output(self, mock_get_orch):
        orch = MagicMock()
        wf = _make_workflow(
            workflow_id="wf-new123", status="running"
        )
        orch.reproduce_workflow = AsyncMock(return_value=wf)
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "reproduce", "wf-abc123"
        ])
        assert result.exit_code == 0
        assert "reproduced" in result.output.lower()
        assert "wf-new123" in result.output

    @patch("cli.commands.workflows._get_orchestrator")
    def test_reproduce_json_output(self, mock_get_orch):
        orch = MagicMock()
        wf = _make_workflow(
            workflow_id="wf-new123", status="running"
        )
        orch.reproduce_workflow = AsyncMock(return_value=wf)
        mock_get_orch.return_value = orch

        runner = CliRunner()
        result = runner.invoke(cli, [
            "workflows", "reproduce", "wf-abc123",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["workflow_id"] == "wf-new123"
