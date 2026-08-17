"""
Tests for CLI model management commands.

Tests tasks 19.5-19.7:
- `trustops models list`
- `trustops models register`
- `trustops models info`
"""

import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from click.testing import CliRunner

from cli.main import cli
from cli.commands.models import _model_to_dict


def _make_model_metadata(**overrides):
    """Create a mock ModelMetadata object."""
    from src.data_models.model import (
        ModelMetadata,
        ModelProvider,
        ModelCapability,
        ModelStatus,
        ModelPricing,
    )

    defaults = dict(
        id="anthropic.claude-v2",
        provider=ModelProvider.BEDROCK,
        name="Claude v2",
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
        status=ModelStatus.ACTIVE,
        fine_tuning_support=True,
        max_tokens=100000,
        region="us-east-1",
        pricing=ModelPricing(
            input_price_per_1k_tokens=0.008,
            output_price_per_1k_tokens=0.024,
            fine_tuning_price_per_1k_tokens=0.012,
            currency="USD",
        ),
    )
    defaults.update(overrides)
    return ModelMetadata(**defaults)


# ─── models list tests (Task 19.5) ───


class TestModelsListCommand:
    """Test `trustops models list` CLI command."""

    def test_list_models_text_output(self):
        runner = CliRunner()
        models = [
            _make_model_metadata(),
            _make_model_metadata(
                id="amazon.titan-text-express-v1",
                name="Titan Text Express",
                fine_tuning_support=False,
            ),
        ]

        mock_registry = MagicMock()
        mock_registry.list_models = AsyncMock(return_value=models)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(cli, ["models", "list"])
            assert result.exit_code == 0
            assert "Claude v2" in result.output
            assert "Titan Text Express" in result.output
            assert "Total: 2 model(s)" in result.output

    def test_list_models_json_output(self):
        runner = CliRunner()
        models = [_make_model_metadata()]

        mock_registry = MagicMock()
        mock_registry.list_models = AsyncMock(return_value=models)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(cli, ["models", "list", "--format", "json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["id"] == "anthropic.claude-v2"
            assert data[0]["provider"] == "bedrock"

    def test_list_models_yaml_output(self):
        runner = CliRunner()
        models = [_make_model_metadata()]

        mock_registry = MagicMock()
        mock_registry.list_models = AsyncMock(return_value=models)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(cli, ["models", "list", "--format", "yaml"])
            assert result.exit_code == 0
            assert "anthropic.claude-v2" in result.output

    def test_list_models_with_provider_filter(self):
        runner = CliRunner()
        models = [_make_model_metadata()]

        mock_registry = MagicMock()
        mock_registry.list_models = AsyncMock(return_value=models)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(
                cli, ["models", "list", "--provider", "bedrock"]
            )
            assert result.exit_code == 0
            mock_registry.list_models.assert_called_once()

    def test_list_models_with_capability_filter(self):
        runner = CliRunner()
        models = [_make_model_metadata()]

        mock_registry = MagicMock()
        mock_registry.list_models = AsyncMock(return_value=models)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(
                cli, ["models", "list", "--capability", "chat"]
            )
            assert result.exit_code == 0

    def test_list_models_with_fine_tunable_flag(self):
        runner = CliRunner()
        models = [_make_model_metadata()]

        mock_registry = MagicMock()
        mock_registry.list_models = AsyncMock(return_value=models)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(
                cli, ["models", "list", "--fine-tunable"]
            )
            assert result.exit_code == 0

    def test_list_models_empty_result(self):
        runner = CliRunner()

        mock_registry = MagicMock()
        mock_registry.list_models = AsyncMock(return_value=[])

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(cli, ["models", "list"])
            assert result.exit_code == 0
            assert "No models found" in result.output

    def test_list_models_error_handling(self):
        runner = CliRunner()

        mock_registry = MagicMock()
        mock_registry.list_models = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(cli, ["models", "list"])
            assert result.exit_code != 0
            assert "Failed to list models" in result.output


# ─── models register tests (Task 19.6) ───


class TestModelsRegisterCommand:
    """Test `trustops models register` CLI command."""

    def test_register_model_text_output(self):
        runner = CliRunner()

        mock_registry = MagicMock()
        mock_registry.register_model = AsyncMock()

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(
                cli,
                [
                    "models", "register",
                    "--provider", "external_api",
                    "--model-id", "openai-gpt4",
                    "--name", "GPT-4",
                ],
            )
            assert result.exit_code == 0
            assert "registered successfully" in result.output
            mock_registry.register_model.assert_called_once()

    def test_register_model_json_output(self):
        runner = CliRunner()

        mock_registry = MagicMock()
        mock_registry.register_model = AsyncMock()

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(
                cli,
                [
                    "models", "register",
                    "--provider", "bedrock",
                    "--model-id", "test-model",
                    "--name", "Test Model",
                    "--format", "json",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["id"] == "test-model"
            assert data["provider"] == "bedrock"

    def test_register_model_with_config_file(self, tmp_path):
        runner = CliRunner()
        config_file = tmp_path / "model_config.json"
        config_file.write_text(json.dumps({
            "capabilities": ["text_generation", "chat"],
            "fine_tuning_support": True,
            "max_tokens": 8192,
            "region": "eu-west-1",
        }))

        mock_registry = MagicMock()
        mock_registry.register_model = AsyncMock()

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(
                cli,
                [
                    "models", "register",
                    "--provider", "external_api",
                    "--model-id", "custom-model",
                    "--name", "Custom Model",
                    "--config", str(config_file),
                    "--format", "json",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["fine_tuning_support"] is True

    def test_register_model_error_handling(self):
        runner = CliRunner()

        mock_registry = MagicMock()
        mock_registry.register_model = AsyncMock(
            side_effect=Exception("Registration failed")
        )

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(
                cli,
                [
                    "models", "register",
                    "--provider", "bedrock",
                    "--model-id", "bad-model",
                    "--name", "Bad Model",
                ],
            )
            assert result.exit_code != 0
            assert "Failed to register model" in result.output


# ─── models info tests (Task 19.7) ───


class TestModelsInfoCommand:
    """Test `trustops models info` CLI command."""

    def test_info_text_output(self):
        runner = CliRunner()
        model = _make_model_metadata()

        mock_registry = MagicMock()
        mock_registry.get_model = AsyncMock(return_value=model)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(cli, ["models", "info", "anthropic.claude-v2"])
            assert result.exit_code == 0
            assert "Claude v2" in result.output
            assert "bedrock" in result.output
            assert "100000" in result.output
            assert "$0.008/1k tokens" in result.output
            assert "$0.024/1k tokens" in result.output

    def test_info_json_output(self):
        runner = CliRunner()
        model = _make_model_metadata()

        mock_registry = MagicMock()
        mock_registry.get_model = AsyncMock(return_value=model)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(
                cli, ["models", "info", "anthropic.claude-v2", "--format", "json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["id"] == "anthropic.claude-v2"
            assert data["pricing"]["input_price_per_1k_tokens"] == 0.008

    def test_info_yaml_output(self):
        runner = CliRunner()
        model = _make_model_metadata()

        mock_registry = MagicMock()
        mock_registry.get_model = AsyncMock(return_value=model)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(
                cli, ["models", "info", "anthropic.claude-v2", "--format", "yaml"]
            )
            assert result.exit_code == 0
            assert "anthropic.claude-v2" in result.output

    def test_info_model_not_found(self):
        runner = CliRunner()

        mock_registry = MagicMock()
        mock_registry.get_model = AsyncMock(return_value=None)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(cli, ["models", "info", "nonexistent-model"])
            assert result.exit_code != 0
            assert "not found" in result.output

    def test_info_model_without_pricing(self):
        runner = CliRunner()
        model = _make_model_metadata(pricing=None)

        mock_registry = MagicMock()
        mock_registry.get_model = AsyncMock(return_value=model)

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(cli, ["models", "info", "anthropic.claude-v2"])
            assert result.exit_code == 0
            assert "Claude v2" in result.output
            # Should not crash without pricing
            assert "Pricing" not in result.output

    def test_info_error_handling(self):
        runner = CliRunner()

        mock_registry = MagicMock()
        mock_registry.get_model = AsyncMock(
            side_effect=Exception("Connection error")
        )

        with patch("cli.commands.models._get_registry", return_value=mock_registry):
            result = runner.invoke(cli, ["models", "info", "some-model"])
            assert result.exit_code != 0
            assert "Failed to get model info" in result.output


# ─── Helper function tests ───


class TestModelToDict:
    """Test the _model_to_dict helper."""

    def test_converts_model_with_pricing(self):
        model = _make_model_metadata()
        result = _model_to_dict(model)
        assert result["id"] == "anthropic.claude-v2"
        assert result["provider"] == "bedrock"
        assert result["pricing"]["input_price_per_1k_tokens"] == 0.008
        assert result["pricing"]["fine_tuning_price_per_1k_tokens"] == 0.012

    def test_converts_model_without_pricing(self):
        model = _make_model_metadata(pricing=None)
        result = _model_to_dict(model)
        assert "pricing" not in result

    def test_converts_capabilities(self):
        model = _make_model_metadata()
        result = _model_to_dict(model)
        assert "text_generation" in result["capabilities"]
        assert "chat" in result["capabilities"]
