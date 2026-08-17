"""
Unit tests for dataset parsers.

Tests JSONL, CSV, and Parquet parsers with validation for required fields
per task type.

Requirements: 2.6, 2.7, 2.8
"""

import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data_models.dataset import DatasetTaskType
from src.datasets.parsers.csv_parser import CSVParser
from src.datasets.parsers.jsonl_parser import JSONLParser
from src.datasets.parsers.parquet_parser import ParquetParser


class TestJSONLParser:
    """Tests for JSONLParser."""

    def test_parse_valid_qa_dataset(self, tmp_path):
        """Test parsing valid QA dataset."""
        file_path = tmp_path / "qa.jsonl"
        data = [
            {"prompt": "What is AI?", "completion": "Artificial Intelligence"},
            {"prompt": "What is ML?", "completion": "Machine Learning"},
        ]

        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        result = JSONLParser.parse(
            file_path, DatasetTaskType.QA, validate=True
        )

        assert len(result) == 2
        assert result[0]["prompt"] == "What is AI?"
        assert result[1]["completion"] == "Machine Learning"

    def test_parse_with_optional_fields(self, tmp_path):
        """Test parsing with optional fields."""
        file_path = tmp_path / "qa_with_context.jsonl"
        data = [
            {
                "prompt": "What is the capital?",
                "completion": "Paris",
                "context": "France is a country in Europe.",
                "category": "geography"
            }
        ]

        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        result = JSONLParser.parse(
            file_path, DatasetTaskType.QA, validate=True
        )

        assert len(result) == 1
        assert result[0]["context"] == "France is a country in Europe."
        assert result[0]["category"] == "geography"

    def test_parse_missing_required_field(self, tmp_path):
        """Test parsing fails when required field is missing."""
        file_path = tmp_path / "invalid.jsonl"
        data = [{"prompt": "What is AI?"}]  # Missing completion

        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        with pytest.raises(ValueError, match="Missing required fields"):
            JSONLParser.parse(
                file_path, DatasetTaskType.QA, validate=True
            )

    def test_parse_invalid_json(self, tmp_path):
        """Test parsing fails with invalid JSON."""
        file_path = tmp_path / "invalid.jsonl"

        with open(file_path, "w") as f:
            f.write("not valid json\n")

        with pytest.raises(json.JSONDecodeError):
            JSONLParser.parse(
                file_path, DatasetTaskType.QA, validate=True
            )

    def test_parse_chat_dataset(self, tmp_path):
        """Test parsing chat dataset with messages field."""
        file_path = tmp_path / "chat.jsonl"
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        ]

        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        result = JSONLParser.parse(
            file_path, DatasetTaskType.CHAT, validate=True
        )

        assert len(result) == 1
        assert isinstance(result[0]["messages"], list)
        assert len(result[0]["messages"]) == 2

    def test_parse_chat_invalid_messages(self, tmp_path):
        """Test parsing fails when messages is not a list."""
        file_path = tmp_path / "chat_invalid.jsonl"
        data = [{"messages": "not a list"}]

        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        with pytest.raises(ValueError, match="must be a list"):
            JSONLParser.parse(
                file_path, DatasetTaskType.CHAT, validate=True
            )

    def test_parse_empty_file(self, tmp_path):
        """Test parsing fails with empty file."""
        file_path = tmp_path / "empty.jsonl"
        file_path.touch()

        with pytest.raises(ValueError, match="No valid examples found"):
            JSONLParser.parse(
                file_path, DatasetTaskType.QA, validate=True
            )

    def test_parse_skip_empty_lines(self, tmp_path):
        """Test parsing skips empty lines."""
        file_path = tmp_path / "with_empty_lines.jsonl"

        with open(file_path, "w") as f:
            f.write('{"prompt": "Q1", "completion": "A1"}\n')
            f.write('\n')
            f.write('{"prompt": "Q2", "completion": "A2"}\n')

        result = JSONLParser.parse(
            file_path, DatasetTaskType.QA, validate=True
        )

        assert len(result) == 2

    def test_validate_file(self, tmp_path):
        """Test file validation report."""
        file_path = tmp_path / "qa.jsonl"
        data = [
            {"prompt": "What is AI?", "completion": "AI answer"},
            {"prompt": "What is ML?", "completion": "ML answer"},
        ]

        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        report = JSONLParser.validate_file(
            file_path, DatasetTaskType.QA
        )

        assert report["valid"] is True
        assert report["total_examples"] == 2
        assert len(report["errors"]) == 0

    def test_get_required_fields(self):
        """Test getting required fields for task types."""
        qa_fields = JSONLParser.get_required_fields(DatasetTaskType.QA)
        assert "prompt" in qa_fields
        assert "completion" in qa_fields

        chat_fields = JSONLParser.get_required_fields(
            DatasetTaskType.CHAT
        )
        assert "messages" in chat_fields


class TestCSVParser:
    """Tests for CSVParser."""

    def test_parse_valid_csv(self, tmp_path):
        """Test parsing valid CSV with standard column names."""
        file_path = tmp_path / "data.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt", "completion"]
            )
            writer.writeheader()
            writer.writerow(
                {"prompt": "What is AI?", "completion": "AI answer"}
            )
            writer.writerow(
                {"prompt": "What is ML?", "completion": "ML answer"}
            )

        result = CSVParser.parse(
            file_path, DatasetTaskType.QA, validate=True
        )

        assert len(result) == 2
        assert result[0]["prompt"] == "What is AI?"
        assert result[1]["completion"] == "ML answer"

    def test_parse_with_auto_mapping(self, tmp_path):
        """Test parsing with automatic column mapping."""
        file_path = tmp_path / "data.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["question", "answer"]
            )
            writer.writeheader()
            writer.writerow(
                {"question": "What is AI?", "answer": "AI answer"}
            )

        result = CSVParser.parse(
            file_path, DatasetTaskType.QA, validate=True
        )

        assert len(result) == 1
        assert result[0]["prompt"] == "What is AI?"
        assert result[0]["completion"] == "AI answer"

    def test_parse_with_custom_mapping(self, tmp_path):
        """Test parsing with custom column mapping."""
        file_path = tmp_path / "data.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["q", "a"]
            )
            writer.writeheader()
            writer.writerow({"q": "What is AI?", "a": "AI answer"})

        mapping = {"q": "prompt", "a": "completion"}
        result = CSVParser.parse(
            file_path, DatasetTaskType.QA, mapping, validate=True
        )

        assert len(result) == 1
        assert result[0]["prompt"] == "What is AI?"
        assert result[0]["completion"] == "AI answer"

    def test_parse_missing_required_field(self, tmp_path):
        """Test parsing fails when required field is missing."""
        file_path = tmp_path / "invalid.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt"])
            writer.writeheader()
            writer.writerow({"prompt": "What is AI?"})

        with pytest.raises(ValueError, match="Missing required fields"):
            CSVParser.parse(
                file_path, DatasetTaskType.QA, validate=True
            )

    def test_parse_empty_csv(self, tmp_path):
        """Test parsing fails with empty CSV."""
        file_path = tmp_path / "empty.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt", "completion"]
            )
            writer.writeheader()

        with pytest.raises(ValueError, match="No valid examples found"):
            CSVParser.parse(
                file_path, DatasetTaskType.QA, validate=True
            )

    def test_validate_file(self, tmp_path):
        """Test CSV file validation report."""
        file_path = tmp_path / "data.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt", "completion"]
            )
            writer.writeheader()
            writer.writerow(
                {"prompt": "What is AI?", "completion": "AI answer"}
            )

        report = CSVParser.validate_file(
            file_path, DatasetTaskType.QA
        )

        assert report["valid"] is True
        assert report["total_examples"] == 1
        assert len(report["errors"]) == 0

    def test_get_column_mapping_suggestions(self, tmp_path):
        """Test getting column mapping suggestions."""
        file_path = tmp_path / "data.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["question", "answer", "context"]
            )
            writer.writeheader()

        suggestions = CSVParser.get_column_mapping_suggestions(file_path)

        assert "prompt" in suggestions
        assert "question" in suggestions["prompt"]
        assert "completion" in suggestions
        assert "answer" in suggestions["completion"]


class TestParquetParser:
    """Tests for ParquetParser."""

    def test_parse_valid_parquet(self, tmp_path):
        """Test parsing valid Parquet file."""
        file_path = tmp_path / "data.parquet"

        # Create Parquet file
        data = {
            "prompt": ["What is AI?", "What is ML?"],
            "completion": ["AI answer", "ML answer"]
        }
        table = pa.table(data)
        pq.write_table(table, str(file_path))

        result = ParquetParser.parse(
            file_path, DatasetTaskType.QA, validate=True
        )

        assert len(result) == 2
        assert result[0]["prompt"] == "What is AI?"
        assert result[1]["completion"] == "ML answer"

    def test_parse_with_optional_fields(self, tmp_path):
        """Test parsing Parquet with optional fields."""
        file_path = tmp_path / "data.parquet"

        data = {
            "prompt": ["What is AI?"],
            "completion": ["AI answer"],
            "context": ["Context here"],
            "category": ["tech"]
        }
        table = pa.table(data)
        pq.write_table(table, str(file_path))

        result = ParquetParser.parse(
            file_path, DatasetTaskType.QA, validate=True
        )

        assert len(result) == 1
        assert result[0]["context"] == "Context here"
        assert result[0]["category"] == "tech"

    def test_parse_missing_required_field(self, tmp_path):
        """Test parsing fails when required field is missing."""
        file_path = tmp_path / "invalid.parquet"

        data = {"prompt": ["What is AI?"]}  # Missing completion
        table = pa.table(data)
        pq.write_table(table, str(file_path))

        with pytest.raises(ValueError, match="Missing required fields"):
            ParquetParser.parse(
                file_path, DatasetTaskType.QA, validate=True
            )

    def test_parse_invalid_type(self, tmp_path):
        """Test parsing fails with invalid field type."""
        file_path = tmp_path / "invalid_type.parquet"

        # Use integer type for prompt (should be string)
        data = {
            "prompt": pa.array([1, 2], type=pa.int64()),
            "completion": ["Answer 1", "Answer 2"]
        }
        table = pa.table(data)
        pq.write_table(table, str(file_path))

        with pytest.raises(ValueError, match="Invalid type"):
            ParquetParser.parse(
                file_path, DatasetTaskType.QA, validate=True
            )

    def test_validate_file(self, tmp_path):
        """Test Parquet file validation report."""
        file_path = tmp_path / "data.parquet"

        data = {
            "prompt": ["What is AI?", "What is ML?"],
            "completion": ["AI answer", "ML answer"]
        }
        table = pa.table(data)
        pq.write_table(table, str(file_path))

        report = ParquetParser.validate_file(
            file_path, DatasetTaskType.QA
        )

        assert report["valid"] is True
        assert report["total_examples"] == 2
        assert len(report["errors"]) == 0
        assert "prompt" in report["schema"]
        assert "completion" in report["schema"]

    def test_get_schema_info(self, tmp_path):
        """Test getting schema information."""
        file_path = tmp_path / "data.parquet"

        data = {
            "prompt": ["What is AI?"],
            "completion": ["AI answer"],
            "category": ["tech"]
        }
        table = pa.table(data)
        pq.write_table(table, str(file_path))

        schema_info = ParquetParser.get_schema_info(file_path)

        assert "prompt" in schema_info
        assert "completion" in schema_info
        assert "category" in schema_info
        assert "string" in schema_info["prompt"].lower()

    def test_validate_schema_compatibility(self, tmp_path):
        """Test schema compatibility validation."""
        file_path = tmp_path / "data.parquet"

        data = {
            "prompt": ["What is AI?"],
            "completion": ["AI answer"]
        }
        table = pa.table(data)
        pq.write_table(table, str(file_path))

        expected_schema = {
            "prompt": ["string", "utf8"],
            "completion": ["string", "utf8"]
        }

        result = ParquetParser.validate_schema_compatibility(
            file_path, expected_schema
        )

        assert result["compatible"] is True
        assert len(result["incompatible_fields"]) == 0
        assert len(result["missing_fields"]) == 0

    def test_validate_schema_incompatibility(self, tmp_path):
        """Test schema incompatibility detection."""
        file_path = tmp_path / "data.parquet"

        data = {
            "prompt": pa.array([1, 2], type=pa.int64()),
            "completion": ["Answer 1", "Answer 2"]
        }
        table = pa.table(data)
        pq.write_table(table, str(file_path))

        expected_schema = {
            "prompt": ["string", "utf8"],
            "completion": ["string", "utf8"]
        }

        result = ParquetParser.validate_schema_compatibility(
            file_path, expected_schema
        )

        assert result["compatible"] is False
        assert "prompt" in result["incompatible_fields"]


class TestParserIntegration:
    """Integration tests for all parsers."""

    def test_all_parsers_same_data(self, tmp_path):
        """Test all parsers produce same results for equivalent data."""
        # Create JSONL
        jsonl_path = tmp_path / "data.jsonl"
        data = [
            {"prompt": "What is AI?", "completion": "AI answer"},
            {"prompt": "What is ML?", "completion": "ML answer"}
        ]

        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        # Create CSV
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt", "completion"]
            )
            writer.writeheader()
            for item in data:
                writer.writerow(item)

        # Create Parquet
        parquet_path = tmp_path / "data.parquet"
        table = pa.table({
            "prompt": [item["prompt"] for item in data],
            "completion": [item["completion"] for item in data]
        })
        pq.write_table(table, str(parquet_path))

        # Parse all formats
        jsonl_result = JSONLParser.parse(
            jsonl_path, DatasetTaskType.QA, validate=True
        )
        csv_result = CSVParser.parse(
            csv_path, DatasetTaskType.QA, validate=True
        )
        parquet_result = ParquetParser.parse(
            parquet_path, DatasetTaskType.QA, validate=True
        )

        # Verify all produce same data
        assert len(jsonl_result) == len(csv_result) == len(parquet_result)
        assert jsonl_result[0]["prompt"] == csv_result[0]["prompt"]
        assert csv_result[0]["prompt"] == parquet_result[0]["prompt"]
        assert jsonl_result[1]["completion"] == csv_result[1]["completion"]
        assert (
            csv_result[1]["completion"] == parquet_result[1]["completion"]
        )
