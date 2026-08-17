"""
Unit tests for format converter.

Tests bidirectional conversion between JSONL, CSV, and Parquet formats
while ensuring data preservation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data_models.dataset import DatasetFormat, DatasetTaskType
from src.datasets.format_converter import FormatConverter


@pytest.fixture
def sample_qa_data():
    """Sample QA dataset for testing."""
    return [
        {
            "prompt": "What is Python?",
            "completion": "Python is a programming language.",
            "category": "programming"
        },
        {
            "prompt": "What is AWS?",
            "completion": "AWS is a cloud platform.",
            "category": "cloud"
        },
        {
            "prompt": "What is ML?",
            "completion": "ML is machine learning.",
            "category": "ai"
        }
    ]


@pytest.fixture
def sample_chat_data():
    """Sample chat dataset for testing."""
    return [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "category": "greeting"
        },
        {
            "messages": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"}
            ],
            "category": "greeting"
        }
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestFormatConverter:
    """Test suite for FormatConverter."""

    def test_jsonl_to_csv_conversion(self, sample_qa_data, temp_dir):
        """Test JSONL to CSV conversion."""
        # Create JSONL file
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for item in sample_qa_data:
                json.dump(item, f)
                f.write("\n")

        # Convert to CSV
        csv_path = temp_dir / "test.csv"
        result = FormatConverter.jsonl_to_csv(
            jsonl_path, csv_path, DatasetTaskType.QA
        )

        assert result["success"] is True
        assert result["rows_converted"] == len(sample_qa_data)
        assert result["source_format"] == "jsonl"
        assert result["target_format"] == "csv"
        assert csv_path.exists()

        # Verify CSV content
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == len(sample_qa_data)
            for i, row in enumerate(rows):
                assert row["prompt"] == sample_qa_data[i]["prompt"]
                assert row["completion"] == sample_qa_data[i]["completion"]
                assert row["category"] == sample_qa_data[i]["category"]

    def test_jsonl_to_parquet_conversion(self, sample_qa_data, temp_dir):
        """Test JSONL to Parquet conversion."""
        # Create JSONL file
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for item in sample_qa_data:
                json.dump(item, f)
                f.write("\n")

        # Convert to Parquet
        parquet_path = temp_dir / "test.parquet"
        result = FormatConverter.jsonl_to_parquet(
            jsonl_path, parquet_path, DatasetTaskType.QA
        )

        assert result["success"] is True
        assert result["rows_converted"] == len(sample_qa_data)
        assert result["source_format"] == "jsonl"
        assert result["target_format"] == "parquet"
        assert parquet_path.exists()

        # Verify Parquet content
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_path))
        assert table.num_rows == len(sample_qa_data)

        # Convert to dict for comparison
        data = table.to_pydict()
        for i in range(len(sample_qa_data)):
            assert data["prompt"][i] == sample_qa_data[i]["prompt"]
            assert data["completion"][i] == sample_qa_data[i]["completion"]
            assert data["category"][i] == sample_qa_data[i]["category"]

    def test_csv_to_jsonl_conversion(self, sample_qa_data, temp_dir):
        """Test CSV to JSONL conversion."""
        # Create CSV file
        csv_path = temp_dir / "test.csv"
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt", "completion", "category"]
            )
            writer.writeheader()
            writer.writerows(sample_qa_data)

        # Convert to JSONL
        jsonl_path = temp_dir / "test.jsonl"
        result = FormatConverter.csv_to_jsonl(
            csv_path, jsonl_path, DatasetTaskType.QA
        )

        assert result["success"] is True
        assert result["rows_converted"] == len(sample_qa_data)
        assert result["source_format"] == "csv"
        assert result["target_format"] == "jsonl"
        assert jsonl_path.exists()

        # Verify JSONL content
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == len(sample_qa_data)
            for i, line in enumerate(lines):
                obj = json.loads(line)
                assert obj["prompt"] == sample_qa_data[i]["prompt"]
                assert obj["completion"] == sample_qa_data[i]["completion"]
                assert obj["category"] == sample_qa_data[i]["category"]

    def test_csv_to_parquet_conversion(self, sample_qa_data, temp_dir):
        """Test CSV to Parquet conversion."""
        # Create CSV file
        csv_path = temp_dir / "test.csv"
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt", "completion", "category"]
            )
            writer.writeheader()
            writer.writerows(sample_qa_data)

        # Convert to Parquet
        parquet_path = temp_dir / "test.parquet"
        result = FormatConverter.csv_to_parquet(
            csv_path, parquet_path, DatasetTaskType.QA
        )

        assert result["success"] is True
        assert result["rows_converted"] == len(sample_qa_data)
        assert result["source_format"] == "csv"
        assert result["target_format"] == "parquet"
        assert parquet_path.exists()

    def test_parquet_to_jsonl_conversion(self, sample_qa_data, temp_dir):
        """Test Parquet to JSONL conversion."""
        # Create Parquet file
        parquet_path = temp_dir / "test.parquet"
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(sample_qa_data)
        pq.write_table(table, str(parquet_path))

        # Convert to JSONL
        jsonl_path = temp_dir / "test.jsonl"
        result = FormatConverter.parquet_to_jsonl(
            parquet_path, jsonl_path, DatasetTaskType.QA
        )

        assert result["success"] is True
        assert result["rows_converted"] == len(sample_qa_data)
        assert result["source_format"] == "parquet"
        assert result["target_format"] == "jsonl"
        assert jsonl_path.exists()

        # Verify JSONL content
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == len(sample_qa_data)
            for i, line in enumerate(lines):
                obj = json.loads(line)
                assert obj["prompt"] == sample_qa_data[i]["prompt"]
                assert obj["completion"] == sample_qa_data[i]["completion"]
                assert obj["category"] == sample_qa_data[i]["category"]

    def test_parquet_to_csv_conversion(self, sample_qa_data, temp_dir):
        """Test Parquet to CSV conversion."""
        # Create Parquet file
        parquet_path = temp_dir / "test.parquet"
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(sample_qa_data)
        pq.write_table(table, str(parquet_path))

        # Convert to CSV
        csv_path = temp_dir / "test.csv"
        result = FormatConverter.parquet_to_csv(
            parquet_path, csv_path, DatasetTaskType.QA
        )

        assert result["success"] is True
        assert result["rows_converted"] == len(sample_qa_data)
        assert result["source_format"] == "parquet"
        assert result["target_format"] == "csv"
        assert csv_path.exists()

        # Verify CSV content
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == len(sample_qa_data)

    def test_round_trip_jsonl_csv_jsonl(self, sample_qa_data, temp_dir):
        """Test round-trip conversion: JSONL -> CSV -> JSONL."""
        # Create original JSONL
        jsonl1_path = temp_dir / "original.jsonl"
        with open(jsonl1_path, "w") as f:
            for item in sample_qa_data:
                json.dump(item, f)
                f.write("\n")

        # Convert to CSV
        csv_path = temp_dir / "intermediate.csv"
        FormatConverter.jsonl_to_csv(jsonl1_path, csv_path)

        # Convert back to JSONL
        jsonl2_path = temp_dir / "final.jsonl"
        FormatConverter.csv_to_jsonl(csv_path, jsonl2_path)

        # Compare original and final
        with open(jsonl1_path, "r") as f1, open(jsonl2_path, "r") as f2:
            lines1 = [json.loads(line) for line in f1]
            lines2 = [json.loads(line) for line in f2]

            assert len(lines1) == len(lines2)
            for obj1, obj2 in zip(lines1, lines2):
                assert obj1 == obj2

    def test_round_trip_jsonl_parquet_jsonl(self, sample_qa_data, temp_dir):
        """Test round-trip conversion: JSONL -> Parquet -> JSONL."""
        # Create original JSONL
        jsonl1_path = temp_dir / "original.jsonl"
        with open(jsonl1_path, "w") as f:
            for item in sample_qa_data:
                json.dump(item, f)
                f.write("\n")

        # Convert to Parquet
        parquet_path = temp_dir / "intermediate.parquet"
        FormatConverter.jsonl_to_parquet(jsonl1_path, parquet_path)

        # Convert back to JSONL
        jsonl2_path = temp_dir / "final.jsonl"
        FormatConverter.parquet_to_jsonl(parquet_path, jsonl2_path)

        # Compare original and final
        with open(jsonl1_path, "r") as f1, open(jsonl2_path, "r") as f2:
            lines1 = [json.loads(line) for line in f1]
            lines2 = [json.loads(line) for line in f2]

            assert len(lines1) == len(lines2)
            for obj1, obj2 in zip(lines1, lines2):
                assert obj1 == obj2

    def test_round_trip_csv_parquet_csv(self, sample_qa_data, temp_dir):
        """Test round-trip conversion: CSV -> Parquet -> CSV."""
        # Create original CSV
        csv1_path = temp_dir / "original.csv"
        import csv
        with open(csv1_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt", "completion", "category"]
            )
            writer.writeheader()
            writer.writerows(sample_qa_data)

        # Convert to Parquet
        parquet_path = temp_dir / "intermediate.parquet"
        FormatConverter.csv_to_parquet(csv1_path, parquet_path)

        # Convert back to CSV
        csv2_path = temp_dir / "final.csv"
        FormatConverter.parquet_to_csv(parquet_path, csv2_path)

        # Compare original and final
        with open(csv1_path, "r") as f1, open(csv2_path, "r") as f2:
            reader1 = csv.DictReader(f1)
            reader2 = csv.DictReader(f2)
            rows1 = list(reader1)
            rows2 = list(reader2)

            assert len(rows1) == len(rows2)
            for row1, row2 in zip(rows1, rows2):
                assert row1 == row2

    def test_complex_data_preservation(self, sample_chat_data, temp_dir):
        """Test that complex data (lists, dicts) is preserved."""
        # Create JSONL with complex data
        jsonl_path = temp_dir / "complex.jsonl"
        with open(jsonl_path, "w") as f:
            for item in sample_chat_data:
                json.dump(item, f)
                f.write("\n")

        # Convert to CSV (should serialize complex types)
        csv_path = temp_dir / "complex.csv"
        FormatConverter.jsonl_to_csv(jsonl_path, csv_path)

        # Convert back to JSONL
        jsonl2_path = temp_dir / "complex2.jsonl"
        FormatConverter.csv_to_jsonl(csv_path, jsonl2_path)

        # Verify messages field is preserved (as JSON string in CSV)
        with open(jsonl2_path, "r") as f:
            lines = [json.loads(line) for line in f]
            assert len(lines) == len(sample_chat_data)
            for i, obj in enumerate(lines):
                # Messages might be a string (JSON serialized) or list
                if isinstance(obj["messages"], str):
                    messages = json.loads(obj["messages"])
                else:
                    messages = obj["messages"]
                assert messages == sample_chat_data[i]["messages"]

    def test_conversion_without_task_type(self, sample_qa_data, temp_dir):
        """Test conversion without specifying task type."""
        # Create JSONL file
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for item in sample_qa_data:
                json.dump(item, f)
                f.write("\n")

        # Convert to CSV without task type
        csv_path = temp_dir / "test.csv"
        result = FormatConverter.jsonl_to_csv(jsonl_path, csv_path)

        assert result["success"] is True
        assert result["rows_converted"] == len(sample_qa_data)
        assert csv_path.exists()

    def test_same_format_conversion_raises_error(self, temp_dir):
        """Test that converting to same format raises error."""
        jsonl_path = temp_dir / "test.jsonl"
        jsonl_path.write_text('{"prompt": "test"}\n')

        output_path = temp_dir / "output.jsonl"

        with pytest.raises(ValueError, match="same"):
            FormatConverter.convert(
                jsonl_path,
                output_path,
                DatasetFormat.JSONL,
                DatasetFormat.JSONL
            )

    def test_missing_input_file_raises_error(self, temp_dir):
        """Test that missing input file raises error."""
        input_path = temp_dir / "nonexistent.jsonl"
        output_path = temp_dir / "output.csv"

        with pytest.raises(ValueError, match="not found"):
            FormatConverter.jsonl_to_csv(input_path, output_path)

    def test_empty_data_raises_error(self, temp_dir):
        """Test that empty data raises appropriate error."""
        # Create empty JSONL file
        jsonl_path = temp_dir / "empty.jsonl"
        jsonl_path.write_text("")

        csv_path = temp_dir / "output.csv"

        with pytest.raises(ValueError, match="No data"):
            FormatConverter.jsonl_to_csv(jsonl_path, csv_path)

    def test_output_directory_creation(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        # Create JSONL file
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            json.dump({"prompt": "test", "completion": "answer"}, f)
            f.write("\n")

        # Output to nested directory that doesn't exist
        csv_path = temp_dir / "nested" / "dir" / "output.csv"

        result = FormatConverter.jsonl_to_csv(jsonl_path, csv_path)

        assert result["success"] is True
        assert csv_path.exists()
        assert csv_path.parent.exists()

    def test_field_order_preservation(self, temp_dir):
        """Test that field order is consistent in CSV output."""
        data = [
            {"z_field": "1", "a_field": "2", "m_field": "3"},
            {"z_field": "4", "a_field": "5", "m_field": "6"}
        ]

        # Create JSONL
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")

        # Convert to CSV
        csv_path = temp_dir / "test.csv"
        FormatConverter.jsonl_to_csv(jsonl_path, csv_path)

        # Check CSV header order (should be sorted)
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert fieldnames == sorted(fieldnames)

    def test_unicode_preservation(self, temp_dir):
        """Test that Unicode characters are preserved."""
        data = [
            {
                "prompt": "What is 你好?",
                "completion": "It means hello in Chinese: 你好",
                "category": "language"
            },
            {
                "prompt": "Emoji test 🚀",
                "completion": "Rocket emoji: 🚀",
                "category": "emoji"
            }
        ]

        # Create JSONL
        jsonl_path = temp_dir / "unicode.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

        # Convert to CSV and back
        csv_path = temp_dir / "unicode.csv"
        FormatConverter.jsonl_to_csv(jsonl_path, csv_path)

        jsonl2_path = temp_dir / "unicode2.jsonl"
        FormatConverter.csv_to_jsonl(csv_path, jsonl2_path)

        # Verify Unicode is preserved
        with open(jsonl2_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
            assert lines[0]["prompt"] == "What is 你好?"
            assert lines[1]["prompt"] == "Emoji test 🚀"
