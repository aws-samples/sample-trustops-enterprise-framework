"""
Unit tests for the training data format validator.

Tests cover model family detection, JSONL structure validation,
required field checks per model family, and edge cases.

Requirements: 4.2
"""

from src.fine_tuning.format_validator import (
    FormatValidationResult,
    ModelFamily,
    detect_model_family,
    validate_jsonl_string,
    validate_training_data_format,
)


# --- Model family detection ---


class TestDetectModelFamily:
    def test_claude_by_name(self):
        assert detect_model_family("anthropic.claude-v2") == ModelFamily.CLAUDE

    def test_claude_by_anthropic(self):
        assert detect_model_family("anthropic.claude-3-sonnet") == ModelFamily.CLAUDE

    def test_titan_by_name(self):
        assert detect_model_family("amazon.titan-text-express-v1") == ModelFamily.TITAN

    def test_titan_by_amazon(self):
        assert detect_model_family("amazon.titan-embed-text-v2") == ModelFamily.TITAN

    def test_llama_by_name(self):
        assert detect_model_family("meta.llama3-8b-instruct-v1") == ModelFamily.LLAMA

    def test_llama_by_meta(self):
        assert detect_model_family("meta.llama2-70b-chat-v1") == ModelFamily.LLAMA

    def test_unknown_model(self):
        assert detect_model_family("some-random-model") == ModelFamily.UNKNOWN

    def test_case_insensitive(self):
        assert detect_model_family("ANTHROPIC.CLAUDE-V2") == ModelFamily.CLAUDE


# --- validate_training_data_format ---


class TestValidateTrainingDataFormat:
    def test_valid_claude_records(self):
        records = [
            {"prompt": "Hello", "completion": "Hi there"},
            {"prompt": "What is AI?", "completion": "Artificial Intelligence"},
        ]
        result = validate_training_data_format(records, "anthropic.claude-v2")
        assert result.is_valid is True
        assert result.issues == []
        assert result.model_family == "claude"
        assert result.records_checked == 2

    def test_valid_titan_records(self):
        records = [
            {"inputText": "Hello", "outputText": "Hi there"},
        ]
        result = validate_training_data_format(records, "amazon.titan-text-v1")
        assert result.is_valid is True
        assert result.issues == []
        assert result.model_family == "titan"

    def test_valid_llama_records(self):
        records = [
            {"prompt": "Hello", "completion": "Hi there"},
        ]
        result = validate_training_data_format(records, "meta.llama3-8b-v1")
        assert result.is_valid is True
        assert result.model_family == "llama"

    def test_unknown_model_family(self):
        result = validate_training_data_format(
            [{"prompt": "x", "completion": "y"}], "unknown-model"
        )
        assert result.is_valid is False
        assert "Unknown model family" in result.issues[0]

    def test_empty_records(self):
        result = validate_training_data_format([], "anthropic.claude-v2")
        assert result.is_valid is False
        assert "empty" in result.issues[0].lower()

    def test_missing_required_field_claude(self):
        records = [{"prompt": "Hello"}]  # missing completion
        result = validate_training_data_format(records, "anthropic.claude-v2")
        assert result.is_valid is False
        assert any("completion" in issue for issue in result.issues)

    def test_missing_required_field_titan(self):
        records = [{"inputText": "Hello"}]  # missing outputText
        result = validate_training_data_format(records, "amazon.titan-text-v1")
        assert result.is_valid is False
        assert any("outputText" in issue for issue in result.issues)

    def test_non_string_field_value(self):
        records = [{"prompt": 123, "completion": "ok"}]
        result = validate_training_data_format(records, "anthropic.claude-v2")
        assert result.is_valid is False
        assert any("string" in issue for issue in result.issues)

    def test_empty_string_field(self):
        records = [{"prompt": "  ", "completion": "ok"}]
        result = validate_training_data_format(records, "anthropic.claude-v2")
        assert result.is_valid is False
        assert any("empty" in issue for issue in result.issues)

    def test_non_dict_record(self):
        records = ["not a dict"]
        result = validate_training_data_format(records, "anthropic.claude-v2")
        assert result.is_valid is False
        assert any("JSON object" in issue for issue in result.issues)

    def test_multiple_issues_reported(self):
        records = [
            {"prompt": "", "completion": ""},
            {"prompt": 42, "completion": None},
        ]
        result = validate_training_data_format(records, "anthropic.claude-v2")
        assert result.is_valid is False
        assert len(result.issues) >= 2

    def test_claude_with_optional_system_field(self):
        records = [
            {"prompt": "Hello", "completion": "Hi", "system": "Be helpful"},
        ]
        result = validate_training_data_format(records, "anthropic.claude-v2")
        assert result.is_valid is True


# --- validate_jsonl_string ---


class TestValidateJsonlString:
    def test_valid_jsonl(self):
        content = '{"prompt": "Hello", "completion": "Hi"}\n{"prompt": "Bye", "completion": "Goodbye"}'
        result = validate_jsonl_string(content, "anthropic.claude-v2")
        assert result.is_valid is True
        assert result.records_checked == 2

    def test_invalid_json_line(self):
        content = '{"prompt": "Hello", "completion": "Hi"}\nnot valid json'
        result = validate_jsonl_string(content, "anthropic.claude-v2")
        assert result.is_valid is False
        assert any("Invalid JSON" in issue for issue in result.issues)

    def test_empty_lines_skipped(self):
        content = '{"prompt": "Hello", "completion": "Hi"}\n\n{"prompt": "Bye", "completion": "Goodbye"}'
        result = validate_jsonl_string(content, "anthropic.claude-v2")
        assert result.is_valid is True
        assert result.records_checked == 2

    def test_unknown_model_in_jsonl(self):
        content = '{"prompt": "Hello", "completion": "Hi"}'
        result = validate_jsonl_string(content, "unknown-model")
        assert result.is_valid is False
        assert "Unknown model family" in result.issues[0]

    def test_empty_content(self):
        result = validate_jsonl_string("", "anthropic.claude-v2")
        assert result.is_valid is False


# --- FormatValidationResult ---


class TestFormatValidationResult:
    def test_defaults(self):
        result = FormatValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.issues == []
        assert result.model_family == ""
        assert result.records_checked == 0

    def test_with_issues(self):
        result = FormatValidationResult(
            is_valid=False, issues=["issue1", "issue2"], model_family="claude"
        )
        assert result.is_valid is False
        assert len(result.issues) == 2
