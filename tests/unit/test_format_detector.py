"""
Unit tests for format auto-detection.

Tests the FormatDetector class for correctly identifying dataset formats
and task types from file content and structure.

Requirement 2.2: Test format auto-detection
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data_models.dataset import DatasetFormat, DatasetTaskType
from src.datasets.format_detector import FormatDetector


class TestFormatDetector:
    """Test suite for FormatDetector."""

    def test_detect_jsonl_format_by_extension(self, tmp_path):
        """Test JSONL format detection by file extension."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write('{"prompt": "test", "completion": "response"}\n')
            f.write('{"prompt": "test2", "completion": "response2"}\n')

        format_type = FormatDetector.detect_format(file_path)
        assert format_type == DatasetFormat.JSONL

    def test_detect_csv_format_by_extension(self, tmp_path):
        """Test CSV format detection by file extension."""
        file_path = tmp_path / "test.csv"
        with open(file_path, "w") as f:
            f.write("prompt,completion\n")
            f.write("test,response\n")

        format_type = FormatDetector.detect_format(file_path)
        assert format_type == DatasetFormat.CSV

    def test_detect_parquet_format_by_extension(self, tmp_path):
        """Test Parquet format detection by file extension."""
        pytest.importorskip("pyarrow")
        import pyarrow as pa
        import pyarrow.parquet as pq

        file_path = tmp_path / "test.parquet"
        table = pa.table({
            "prompt": ["test1", "test2"],
            "completion": ["response1", "response2"]
        })
        pq.write_table(table, str(file_path))

        format_type = FormatDetector.detect_format(file_path)
        assert format_type == DatasetFormat.PARQUET

    def test_detect_jsonl_format_by_content(self, tmp_path):
        """Test JSONL format detection by content when extension is ambiguous."""
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as f:
            f.write('{"prompt": "test", "completion": "response"}\n')
            f.write('{"prompt": "test2", "completion": "response2"}\n')

        format_type = FormatDetector.detect_format(file_path)
        assert format_type == DatasetFormat.JSONL

    def test_detect_csv_format_by_content(self, tmp_path):
        """Test CSV format detection by content when extension is ambiguous."""
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as f:
            f.write("prompt,completion\n")
            f.write("test,response\n")
            f.write("test2,response2\n")

        format_type = FormatDetector.detect_format(file_path)
        assert format_type == DatasetFormat.CSV

    def test_detect_huggingface_format(self, tmp_path):
        """Test HuggingFace format detection for single JSON object."""
        file_path = tmp_path / "test.json"
        data = {
            "train": [
                {"prompt": "test1", "completion": "response1"},
                {"prompt": "test2", "completion": "response2"}
            ]
        }
        with open(file_path, "w") as f:
            json.dump(data, f)

        format_type = FormatDetector.detect_format(file_path)
        assert format_type == DatasetFormat.HUGGINGFACE

    def test_detect_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(ValueError, match="File not found"):
            FormatDetector.detect_format("nonexistent.jsonl")

    def test_detect_qa_task_type(self, tmp_path):
        """Test QA task type detection."""
        file_path = tmp_path / "qa.jsonl"
        with open(file_path, "w") as f:
            f.write('{"question": "What is AI?", "answer": "Artificial Intelligence"}\n')

        task_type = FormatDetector.detect_task_type(file_path)
        assert task_type == DatasetTaskType.QA

    def test_detect_qa_task_type_with_context(self, tmp_path):
        """Test QA task type detection with context field."""
        file_path = tmp_path / "qa_context.jsonl"
        with open(file_path, "w") as f:
            f.write('{"prompt": "What is AI?", "context": "AI is...", "completion": "Answer"}\n')

        task_type = FormatDetector.detect_task_type(file_path)
        assert task_type == DatasetTaskType.QA

    def test_detect_chat_task_type(self, tmp_path):
        """Test chat task type detection."""
        file_path = tmp_path / "chat.jsonl"
        with open(file_path, "w") as f:
            f.write('{"messages": [{"role": "user", "content": "Hello"}]}\n')

        task_type = FormatDetector.detect_task_type(file_path)
        assert task_type == DatasetTaskType.CHAT

    def test_detect_summarization_task_type(self, tmp_path):
        """Test summarization task type detection."""
        file_path = tmp_path / "summarization.jsonl"
        with open(file_path, "w") as f:
            f.write('{"document": "Long text...", "summary": "Short summary"}\n')

        task_type = FormatDetector.detect_task_type(file_path)
        assert task_type == DatasetTaskType.SUMMARIZATION

    def test_detect_classification_task_type(self, tmp_path):
        """Test classification task type detection."""
        file_path = tmp_path / "classification.jsonl"
        with open(file_path, "w") as f:
            f.write('{"text": "This is great!", "label": "positive"}\n')

        task_type = FormatDetector.detect_task_type(file_path)
        assert task_type == DatasetTaskType.CLASSIFICATION

    def test_detect_text_generation_task_type(self, tmp_path):
        """Test text generation task type detection."""
        file_path = tmp_path / "text_gen.jsonl"
        with open(file_path, "w") as f:
            f.write('{"prompt": "Write a story", "completion": "Once upon a time..."}\n')

        task_type = FormatDetector.detect_task_type(file_path)
        assert task_type == DatasetTaskType.TEXT_GENERATION

    def test_detect_custom_task_type(self, tmp_path):
        """Test custom task type detection for unrecognized patterns."""
        file_path = tmp_path / "custom.jsonl"
        with open(file_path, "w") as f:
            f.write('{"custom_field": "value", "another_field": "data"}\n')

        task_type = FormatDetector.detect_task_type(file_path)
        assert task_type == DatasetTaskType.CUSTOM

    def test_detect_task_type_csv(self, tmp_path):
        """Test task type detection for CSV format."""
        file_path = tmp_path / "classification.csv"
        with open(file_path, "w") as f:
            f.write("text,label\n")
            f.write("This is great,positive\n")
            f.write("This is bad,negative\n")

        task_type = FormatDetector.detect_task_type(file_path)
        assert task_type == DatasetTaskType.CLASSIFICATION

    def test_detect_task_type_parquet(self, tmp_path):
        """Test task type detection for Parquet format."""
        pytest.importorskip("pyarrow")
        import pyarrow as pa
        import pyarrow.parquet as pq

        file_path = tmp_path / "qa.parquet"
        table = pa.table({
            "question": ["What is AI?", "What is ML?"],
            "answer": ["Artificial Intelligence", "Machine Learning"]
        })
        pq.write_table(table, str(file_path))

        task_type = FormatDetector.detect_task_type(file_path)
        assert task_type == DatasetTaskType.QA

    def test_detect_task_type_case_insensitive(self, tmp_path):
        """Test that task type detection is case-insensitive."""
        file_path = tmp_path / "qa_case.jsonl"
        with open(file_path, "w") as f:
            f.write('{"Question": "What is AI?", "Answer": "Artificial Intelligence"}\n')

        task_type = FormatDetector.detect_task_type(file_path)
        assert task_type == DatasetTaskType.QA

    def test_detect_task_type_empty_file(self, tmp_path):
        """Test error handling for empty file."""
        file_path = tmp_path / "empty.jsonl"
        file_path.touch()

        with pytest.raises(ValueError, match="Unable to extract field names"):
            FormatDetector.detect_task_type(file_path)

    def test_extract_field_names_jsonl(self, tmp_path):
        """Test field name extraction from JSONL."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write('{"field1": "value1", "field2": "value2"}\n')

        field_names = FormatDetector._extract_field_names(file_path, DatasetFormat.JSONL)
        assert field_names == {"field1", "field2"}

    def test_extract_field_names_csv(self, tmp_path):
        """Test field name extraction from CSV."""
        file_path = tmp_path / "test.csv"
        with open(file_path, "w") as f:
            f.write("col1,col2,col3\n")
            f.write("val1,val2,val3\n")

        field_names = FormatDetector._extract_field_names(file_path, DatasetFormat.CSV)
        assert field_names == {"col1", "col2", "col3"}

    def test_extract_field_names_parquet(self, tmp_path):
        """Test field name extraction from Parquet."""
        pytest.importorskip("pyarrow")
        import pyarrow as pa
        import pyarrow.parquet as pq

        file_path = tmp_path / "test.parquet"
        table = pa.table({
            "column_a": [1, 2, 3],
            "column_b": ["x", "y", "z"]
        })
        pq.write_table(table, str(file_path))

        field_names = FormatDetector._extract_field_names(file_path, DatasetFormat.PARQUET)
        assert field_names == {"column_a", "column_b"}

    def test_jsonlines_extension(self, tmp_path):
        """Test detection with .jsonlines extension."""
        file_path = tmp_path / "test.jsonlines"
        with open(file_path, "w") as f:
            f.write('{"prompt": "test", "completion": "response"}\n')

        format_type = FormatDetector.detect_format(file_path)
        assert format_type == DatasetFormat.JSONL

    def test_ndjson_extension(self, tmp_path):
        """Test detection with .ndjson extension."""
        file_path = tmp_path / "test.ndjson"
        with open(file_path, "w") as f:
            f.write('{"prompt": "test", "completion": "response"}\n')

        format_type = FormatDetector.detect_format(file_path)
        assert format_type == DatasetFormat.JSONL
