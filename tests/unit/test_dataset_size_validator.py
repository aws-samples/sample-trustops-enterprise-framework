"""
Unit tests for the minimum dataset size validator.

Tests cover row count validation, token count estimation,
per-model-family minimums, and edge cases.

Requirements: 4.14
"""

from src.fine_tuning.dataset_size_validator import (
    DatasetSizeValidationResult,
    MIN_ROWS,
    MIN_TOKENS,
    validate_dataset_size,
)
from src.fine_tuning.format_validator import ModelFamily


# --- Helpers ---

def _make_claude_records(n: int, text_len: int = 100) -> list[dict]:
    return [
        {"prompt": "x" * text_len, "completion": "y" * text_len}
        for _ in range(n)
    ]


def _make_titan_records(n: int, text_len: int = 100) -> list[dict]:
    return [
        {"inputText": "x" * text_len, "outputText": "y" * text_len}
        for _ in range(n)
    ]


def _make_llama_records(n: int, text_len: int = 100) -> list[dict]:
    return [
        {"prompt": "x" * text_len, "completion": "y" * text_len}
        for _ in range(n)
    ]


# --- Claude validation ---


class TestClaudeDatasetSize:
    def test_valid_claude_dataset(self):
        records = _make_claude_records(50, text_len=200)
        result = validate_dataset_size(records, "anthropic.claude-v2")
        assert result.is_valid is True
        assert result.row_count == 50
        assert result.min_rows_required == MIN_ROWS[ModelFamily.CLAUDE]

    def test_claude_too_few_rows(self):
        records = _make_claude_records(10, text_len=200)
        result = validate_dataset_size(records, "anthropic.claude-v2")
        assert result.is_valid is False
        assert "rows" in result.reason.lower()
        assert result.row_count == 10

    def test_claude_too_few_tokens(self):
        # Very short text → low token count
        records = _make_claude_records(32, text_len=2)
        result = validate_dataset_size(records, "anthropic.claude-v2")
        assert result.is_valid is False
        assert "token" in result.reason.lower()


# --- Titan validation ---


class TestTitanDatasetSize:
    def test_valid_titan_dataset(self):
        records = _make_titan_records(150, text_len=200)
        result = validate_dataset_size(records, "amazon.titan-text-v1")
        assert result.is_valid is True
        assert result.row_count == 150
        assert result.min_rows_required == MIN_ROWS[ModelFamily.TITAN]

    def test_titan_too_few_rows(self):
        records = _make_titan_records(50, text_len=200)
        result = validate_dataset_size(records, "amazon.titan-text-v1")
        assert result.is_valid is False
        assert "100" in result.reason  # Titan needs 100+

    def test_titan_too_few_tokens(self):
        records = _make_titan_records(100, text_len=2)
        result = validate_dataset_size(records, "amazon.titan-text-v1")
        assert result.is_valid is False
        assert "token" in result.reason.lower()


# --- Llama validation ---


class TestLlamaDatasetSize:
    def test_valid_llama_dataset(self):
        records = _make_llama_records(100, text_len=200)
        result = validate_dataset_size(records, "meta.llama3-8b-v1")
        assert result.is_valid is True
        assert result.min_rows_required == MIN_ROWS[ModelFamily.LLAMA]

    def test_llama_too_few_rows(self):
        records = _make_llama_records(30, text_len=200)
        result = validate_dataset_size(records, "meta.llama3-8b-v1")
        assert result.is_valid is False
        assert "64" in result.reason  # Llama needs 64+


# --- Unknown model ---


class TestUnknownModel:
    def test_unknown_model_fails(self):
        records = [{"prompt": "x", "completion": "y"}]
        result = validate_dataset_size(records, "unknown-model")
        assert result.is_valid is False
        assert "Unknown model family" in result.reason


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_dataset(self):
        result = validate_dataset_size([], "anthropic.claude-v2")
        assert result.is_valid is False
        assert result.row_count == 0

    def test_both_row_and_token_failures(self):
        records = _make_claude_records(5, text_len=2)
        result = validate_dataset_size(records, "anthropic.claude-v2")
        assert result.is_valid is False
        # Both row and token issues should be in reason
        assert "rows" in result.reason.lower() or "row" in result.reason.lower()
        assert "token" in result.reason.lower()

    def test_estimated_token_count_populated(self):
        records = _make_claude_records(50, text_len=200)
        result = validate_dataset_size(records, "anthropic.claude-v2")
        assert result.estimated_token_count > 0


# --- DatasetSizeValidationResult ---


class TestDatasetSizeValidationResult:
    def test_defaults(self):
        result = DatasetSizeValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.reason == ""
        assert result.row_count == 0
