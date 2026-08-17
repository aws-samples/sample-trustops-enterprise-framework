"""
Tests for CLI dataset management commands.

Tests tasks 19.8-19.11:
- `trustops datasets upload`
- `trustops datasets list`
- `trustops datasets analyze`
- `trustops datasets convert`
"""

import json
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from click.testing import CliRunner

from cli.main import cli
from cli.commands.datasets import _metadata_to_dict, _quality_report_to_dict


def _make_token_stats():
    """Create a mock TokenStatistics."""
    from src.data_models.dataset import TokenStatistics
    return TokenStatistics(
        total_tokens=50000,
        min_tokens=10,
        max_tokens=500,
        avg_tokens=125.0,
        p95_tokens=400,
        prompt_tokens=25000,
        completion_tokens=25000,
    )


def _make_dataset_metadata(**overrides):
    """Create a mock DatasetMetadata object."""
    from src.data_models.dataset import (
        DatasetMetadata,
        DatasetFormat,
        DatasetTaskType,
    )

    defaults = dict(
        id="ds-abc123",
        name="test-dataset",
        description="A test dataset",
        format=DatasetFormat.JSONL,
        task_type=DatasetTaskType.QA,
        version="1.0",
        s3_uri="s3://bucket/datasets/ds-abc123/data.jsonl",
        row_count=400,
        token_stats=_make_token_stats(),
        created_at=datetime(2024, 1, 15, 10, 30, 0),
        checksum="abc123def456",
    )
    defaults.update(overrides)
    return DatasetMetadata(**defaults)


def _make_quality_report():
    """Create a mock DatasetQualityReport."""
    from src.data_models.dataset import (
        DatasetQualityReport,
        QualityIssue,
    )

    return DatasetQualityReport(
        completeness_score=0.95,
        diversity_score=0.82,
        balance_score=0.78,
        token_stats=_make_token_stats(),
        issues=[
            QualityIssue(
                severity="warning",
                category="missing_field",
                message="5 rows missing 'context' field",
                affected_rows=[10, 25, 100, 200, 350],
            ),
            QualityIssue(
                severity="info",
                category="token_length",
                message="Some completions are very short (<10 tokens)",
            ),
        ],
        recommendations=[
            "Add context field to improve grounding evaluation",
            "Consider augmenting short completions",
        ],
    )


# ─── datasets upload tests (Task 19.8) ───


class TestDatasetsUploadCommand:
    """Test `trustops datasets upload` CLI command."""

    def test_upload_text_output(self, tmp_path):
        runner = CliRunner()
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"prompt": "test", "completion": "answer"}\n')

        metadata = _make_dataset_metadata()
        mock_manager = MagicMock()
        mock_manager.upload_dataset = AsyncMock(return_value=metadata)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli,
                ["datasets", "upload", str(data_file), "--name", "my-dataset"],
            )
            assert result.exit_code == 0
            assert "uploaded successfully" in result.output
            assert "ds-abc123" in result.output

    def test_upload_json_output(self, tmp_path):
        runner = CliRunner()
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"prompt": "test", "completion": "answer"}\n')

        metadata = _make_dataset_metadata()
        mock_manager = MagicMock()
        mock_manager.upload_dataset = AsyncMock(return_value=metadata)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli,
                [
                    "datasets", "upload", str(data_file),
                    "--name", "my-dataset",
                    "--format", "json",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["id"] == "ds-abc123"
            assert data["format"] == "jsonl"

    def test_upload_with_task_type(self, tmp_path):
        runner = CliRunner()
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"prompt": "test", "completion": "answer"}\n')

        metadata = _make_dataset_metadata()
        mock_manager = MagicMock()
        mock_manager.upload_dataset = AsyncMock(return_value=metadata)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli,
                [
                    "datasets", "upload", str(data_file),
                    "--name", "my-dataset",
                    "--task-type", "qa",
                ],
            )
            assert result.exit_code == 0
            mock_manager.upload_dataset.assert_called_once()

    def test_upload_with_description(self, tmp_path):
        runner = CliRunner()
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"prompt": "test", "completion": "answer"}\n')

        metadata = _make_dataset_metadata()
        mock_manager = MagicMock()
        mock_manager.upload_dataset = AsyncMock(return_value=metadata)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli,
                [
                    "datasets", "upload", str(data_file),
                    "--name", "my-dataset",
                    "--description", "Test dataset for QA",
                ],
            )
            assert result.exit_code == 0

    def test_upload_nonexistent_file(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["datasets", "upload", "/nonexistent/file.jsonl", "--name", "test"],
        )
        assert result.exit_code != 0

    def test_upload_error_handling(self, tmp_path):
        runner = CliRunner()
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"prompt": "test"}\n')

        mock_manager = MagicMock()
        mock_manager.upload_dataset = AsyncMock(
            side_effect=Exception("Upload failed: bucket not found")
        )

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli,
                ["datasets", "upload", str(data_file), "--name", "bad-dataset"],
            )
            assert result.exit_code != 0
            assert "Failed to upload dataset" in result.output


# ─── datasets list tests (Task 19.9) ───


class TestDatasetsListCommand:
    """Test `trustops datasets list` CLI command."""

    def test_list_text_output(self):
        runner = CliRunner()
        datasets = [
            _make_dataset_metadata(),
            _make_dataset_metadata(
                id="ds-xyz789",
                name="another-dataset",
                format="csv",
                task_type="summarization",
                row_count=1000,
            ),
        ]

        mock_manager = MagicMock()
        mock_manager.list_datasets = AsyncMock(return_value=datasets)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(cli, ["datasets", "list"])
            assert result.exit_code == 0
            assert "test-dataset" in result.output
            assert "another-dataset" in result.output
            assert "Total: 2 dataset(s)" in result.output

    def test_list_json_output(self):
        runner = CliRunner()
        datasets = [_make_dataset_metadata()]

        mock_manager = MagicMock()
        mock_manager.list_datasets = AsyncMock(return_value=datasets)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(cli, ["datasets", "list", "--format", "json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "test-dataset"

    def test_list_yaml_output(self):
        runner = CliRunner()
        datasets = [_make_dataset_metadata()]

        mock_manager = MagicMock()
        mock_manager.list_datasets = AsyncMock(return_value=datasets)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(cli, ["datasets", "list", "--format", "yaml"])
            assert result.exit_code == 0
            assert "test-dataset" in result.output

    def test_list_with_task_type_filter(self):
        runner = CliRunner()
        datasets = [_make_dataset_metadata()]

        mock_manager = MagicMock()
        mock_manager.list_datasets = AsyncMock(return_value=datasets)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli, ["datasets", "list", "--task-type", "qa"]
            )
            assert result.exit_code == 0

    def test_list_with_format_filter(self):
        runner = CliRunner()
        datasets = [_make_dataset_metadata()]

        mock_manager = MagicMock()
        mock_manager.list_datasets = AsyncMock(return_value=datasets)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli, ["datasets", "list", "--dataset-format", "jsonl"]
            )
            assert result.exit_code == 0

    def test_list_empty_result(self):
        runner = CliRunner()

        mock_manager = MagicMock()
        mock_manager.list_datasets = AsyncMock(return_value=[])

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(cli, ["datasets", "list"])
            assert result.exit_code == 0
            assert "No datasets found" in result.output

    def test_list_error_handling(self):
        runner = CliRunner()

        mock_manager = MagicMock()
        mock_manager.list_datasets = AsyncMock(
            side_effect=Exception("DynamoDB error")
        )

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(cli, ["datasets", "list"])
            assert result.exit_code != 0
            assert "Failed to list datasets" in result.output


# ─── datasets analyze tests (Task 19.10) ───


class TestDatasetsAnalyzeCommand:
    """Test `trustops datasets analyze` CLI command."""

    def test_analyze_text_output(self):
        runner = CliRunner()
        report = _make_quality_report()

        mock_manager = MagicMock()
        mock_manager.analyze_quality = AsyncMock(return_value=report)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(cli, ["datasets", "analyze", "ds-abc123"])
            assert result.exit_code == 0
            assert "Quality Scores" in result.output
            assert "0.95" in result.output  # completeness
            assert "0.82" in result.output  # diversity
            assert "0.78" in result.output  # balance
            assert "Token Statistics" in result.output
            assert "50000" in result.output  # total tokens
            assert "missing_field" in result.output
            assert "Add context field" in result.output

    def test_analyze_json_output(self):
        runner = CliRunner()
        report = _make_quality_report()

        mock_manager = MagicMock()
        mock_manager.analyze_quality = AsyncMock(return_value=report)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli, ["datasets", "analyze", "ds-abc123", "--format", "json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["completeness_score"] == 0.95
            assert data["diversity_score"] == 0.82
            assert len(data["issues"]) == 2
            assert len(data["recommendations"]) == 2

    def test_analyze_yaml_output(self):
        runner = CliRunner()
        report = _make_quality_report()

        mock_manager = MagicMock()
        mock_manager.analyze_quality = AsyncMock(return_value=report)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli, ["datasets", "analyze", "ds-abc123", "--format", "yaml"]
            )
            assert result.exit_code == 0
            assert "completeness_score" in result.output

    def test_analyze_shows_issues(self):
        runner = CliRunner()
        report = _make_quality_report()

        mock_manager = MagicMock()
        mock_manager.analyze_quality = AsyncMock(return_value=report)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(cli, ["datasets", "analyze", "ds-abc123"])
            assert result.exit_code == 0
            assert "Issues" in result.output
            assert "WARNING" in result.output
            assert "INFO" in result.output

    def test_analyze_shows_recommendations(self):
        runner = CliRunner()
        report = _make_quality_report()

        mock_manager = MagicMock()
        mock_manager.analyze_quality = AsyncMock(return_value=report)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(cli, ["datasets", "analyze", "ds-abc123"])
            assert result.exit_code == 0
            assert "Recommendations" in result.output
            assert "1." in result.output
            assert "2." in result.output

    def test_analyze_error_handling(self):
        runner = CliRunner()

        mock_manager = MagicMock()
        mock_manager.analyze_quality = AsyncMock(
            side_effect=Exception("Dataset not found")
        )

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(cli, ["datasets", "analyze", "nonexistent"])
            assert result.exit_code != 0
            assert "Failed to analyze dataset" in result.output


# ─── datasets convert tests (Task 19.11) ───


class TestDatasetsConvertCommand:
    """Test `trustops datasets convert` CLI command."""

    def test_convert_text_output(self):
        runner = CliRunner()
        from src.data_models.dataset import DatasetFormat

        converted = _make_dataset_metadata(
            id="ds-converted",
            format=DatasetFormat.CSV,
            s3_uri="s3://bucket/datasets/ds-converted/data.csv",
        )

        mock_manager = MagicMock()
        mock_manager.convert_format = AsyncMock(return_value=converted)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli,
                ["datasets", "convert", "ds-abc123", "--target-format", "csv"],
            )
            assert result.exit_code == 0
            assert "converted to csv" in result.output.lower()
            assert "ds-converted" in result.output

    def test_convert_json_output(self):
        runner = CliRunner()
        from src.data_models.dataset import DatasetFormat

        converted = _make_dataset_metadata(
            id="ds-converted",
            format=DatasetFormat.PARQUET,
        )

        mock_manager = MagicMock()
        mock_manager.convert_format = AsyncMock(return_value=converted)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli,
                [
                    "datasets", "convert", "ds-abc123",
                    "--target-format", "parquet",
                    "--format", "json",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["id"] == "ds-converted"
            assert data["format"] == "parquet"

    def test_convert_yaml_output(self):
        runner = CliRunner()
        from src.data_models.dataset import DatasetFormat

        converted = _make_dataset_metadata(
            id="ds-converted",
            format=DatasetFormat.CSV,
        )

        mock_manager = MagicMock()
        mock_manager.convert_format = AsyncMock(return_value=converted)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli,
                [
                    "datasets", "convert", "ds-abc123",
                    "--target-format", "csv",
                    "--format", "yaml",
                ],
            )
            assert result.exit_code == 0
            assert "ds-converted" in result.output

    def test_convert_to_jsonl(self):
        runner = CliRunner()
        from src.data_models.dataset import DatasetFormat

        converted = _make_dataset_metadata(
            id="ds-jsonl",
            format=DatasetFormat.JSONL,
        )

        mock_manager = MagicMock()
        mock_manager.convert_format = AsyncMock(return_value=converted)

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli,
                ["datasets", "convert", "ds-abc123", "--target-format", "jsonl"],
            )
            assert result.exit_code == 0

    def test_convert_error_handling(self):
        runner = CliRunner()

        mock_manager = MagicMock()
        mock_manager.convert_format = AsyncMock(
            side_effect=Exception("Conversion failed: unsupported format")
        )

        with patch("cli.commands.datasets._get_dataset_manager", return_value=mock_manager):
            result = runner.invoke(
                cli,
                ["datasets", "convert", "ds-abc123", "--target-format", "csv"],
            )
            assert result.exit_code != 0
            assert "Failed to convert dataset" in result.output

    def test_convert_requires_target_format(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["datasets", "convert", "ds-abc123"]
        )
        assert result.exit_code != 0


# ─── Helper function tests ───


class TestMetadataToDict:
    """Test the _metadata_to_dict helper."""

    def test_converts_metadata(self):
        meta = _make_dataset_metadata()
        result = _metadata_to_dict(meta)
        assert result["id"] == "ds-abc123"
        assert result["format"] == "jsonl"
        assert result["task_type"] == "qa"
        assert result["row_count"] == 400
        assert result["description"] == "A test dataset"
        assert result["token_stats"]["total_tokens"] == 50000

    def test_converts_metadata_without_description(self):
        meta = _make_dataset_metadata(description=None)
        result = _metadata_to_dict(meta)
        assert "description" not in result


class TestQualityReportToDict:
    """Test the _quality_report_to_dict helper."""

    def test_converts_report(self):
        report = _make_quality_report()
        result = _quality_report_to_dict(report)
        assert result["completeness_score"] == 0.95
        assert result["diversity_score"] == 0.82
        assert result["balance_score"] == 0.78
        assert len(result["issues"]) == 2
        assert result["issues"][0]["severity"] == "warning"
        assert len(result["recommendations"]) == 2
