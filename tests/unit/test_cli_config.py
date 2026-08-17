"""
Tests for CLI config commands.

Tests tasks 19.25-19.26:
- `trustops config show`
- `trustops config set`
- JSON/YAML output mode across commands
"""

import json
from unittest.mock import patch, MagicMock

import pytest
import yaml
from click.testing import CliRunner

from cli.main import cli
from cli.config import load_config, save_config, get_default_config


# ─── config show tests (Task 19.25) ───


class TestConfigShow:
    """Test `trustops config show` command."""

    def test_show_text_output(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {
            "aws": {"region": "us-east-1"},
            "trust_scoring": {"threshold": 0.7},
        }
        config_file.write_text(json.dumps(config_data))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(config_file), "config", "show"],
        )
        assert result.exit_code == 0
        assert "us-east-1" in result.output
        assert "0.7" in result.output

    def test_show_json_output(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {
            "aws": {"region": "eu-west-1"},
            "trust_scoring": {"threshold": 0.8},
        }
        config_file.write_text(json.dumps(config_data))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config", str(config_file),
                "config", "show",
                "--format", "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["aws"]["region"] == "eu-west-1"
        assert data["trust_scoring"]["threshold"] == 0.8

    def test_show_yaml_output(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {
            "aws": {"region": "ap-southeast-1"},
        }
        config_file.write_text(json.dumps(config_data))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config", str(config_file),
                "config", "show",
                "--format", "yaml",
            ],
        )
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data["aws"]["region"] == "ap-southeast-1"


# ─── config set tests (Task 19.25) ───


class TestConfigSet:
    """Test `trustops config set` command."""

    def test_set_string_value(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {"aws": {"region": "us-east-1"}}
        config_file.write_text(json.dumps(config_data))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config", str(config_file),
                "config", "set",
                "aws.region", "eu-west-1",
            ],
        )
        assert result.exit_code == 0
        assert "eu-west-1" in result.output

        # Verify file was updated
        updated = json.loads(config_file.read_text())
        assert updated["aws"]["region"] == "eu-west-1"

    def test_set_numeric_value(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {"trust_scoring": {"threshold": 0.7}}
        config_file.write_text(json.dumps(config_data))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config", str(config_file),
                "config", "set",
                "trust_scoring.threshold", "0.85",
            ],
        )
        assert result.exit_code == 0

        updated = json.loads(config_file.read_text())
        assert updated["trust_scoring"]["threshold"] == 0.85

    def test_set_boolean_value(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {"features": {"debug": False}}
        config_file.write_text(json.dumps(config_data))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config", str(config_file),
                "config", "set",
                "features.debug", "true",
            ],
        )
        assert result.exit_code == 0

        updated = json.loads(config_file.read_text())
        assert updated["features"]["debug"] is True

    def test_set_creates_nested_keys(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {}
        config_file.write_text(json.dumps(config_data))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config", str(config_file),
                "config", "set",
                "new.nested.key", "value",
            ],
        )
        assert result.exit_code == 0

        updated = json.loads(config_file.read_text())
        assert updated["new"]["nested"]["key"] == "value"

    def test_set_json_output(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {"aws": {"region": "us-east-1"}}
        config_file.write_text(json.dumps(config_data))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config", str(config_file),
                "config", "set",
                "aws.region", "us-west-2",
                "--format", "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["key"] == "aws.region"
        assert data["value"] == "us-west-2"
        assert data["status"] == "updated"


# ─── JSON/YAML output mode tests (Task 19.26) ───


class TestOutputFormats:
    """Test that all new commands support --format json/yaml."""

    def test_config_show_supports_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"aws": {"region": "us-east-1"}}))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config", str(config_file),
                "config", "show",
                "--format", "json",
            ],
        )
        assert result.exit_code == 0
        # Verify it's valid JSON
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_config_show_supports_yaml(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"aws": {"region": "us-east-1"}}))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config", str(config_file),
                "config", "show",
                "--format", "yaml",
            ],
        )
        assert result.exit_code == 0
        # Verify it's valid YAML
        data = yaml.safe_load(result.output)
        assert isinstance(data, dict)

    def test_config_set_supports_yaml(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"aws": {"region": "us-east-1"}}))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config", str(config_file),
                "config", "set",
                "aws.region", "eu-west-1",
                "--format", "yaml",
            ],
        )
        assert result.exit_code == 0
        data = yaml.safe_load(result.output)
        assert data["key"] == "aws.region"
