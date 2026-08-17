"""
Unit tests for the ResponseParser.

Tests parsing of Bedrock, SageMaker, and External API (OpenAI-compatible)
response formats into normalized ParsedResponse objects.
"""

import pytest

from src.evaluation.response_parser import (
    ProviderFormat,
    ResponseParser,
    _estimate_tokens,
)


@pytest.fixture
def parser() -> ResponseParser:
    return ResponseParser()


# --- Bedrock ---


class TestBedrockParsing:
    def test_full_bedrock_response(self, parser: ResponseParser):
        raw = {
            "response_text": "Hello world",
            "input_tokens": 5,
            "output_tokens": 2,
            "latency_ms": 120.5,
            "model_id": "anthropic.claude-v2",
            "finish_reason": "stop",
        }
        result = parser.parse(raw, ProviderFormat.BEDROCK)

        assert result.text == "Hello world"
        assert result.input_tokens == 5
        assert result.output_tokens == 2
        assert result.latency_ms == 120.5
        assert result.model_id == "anthropic.claude-v2"
        assert result.finish_reason == "stop"
        assert result.raw is raw

    def test_bedrock_missing_optional_fields(self, parser: ResponseParser):
        raw = {"response_text": "Answer"}
        result = parser.parse(raw, ProviderFormat.BEDROCK)

        assert result.text == "Answer"
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.finish_reason == "stop"

    def test_bedrock_uses_fallback_latency(self, parser: ResponseParser):
        raw = {"response_text": "Hi"}
        result = parser.parse(raw, ProviderFormat.BEDROCK, latency_ms=99.0)
        assert result.latency_ms == 99.0


# --- SageMaker ---


class TestSageMakerParsing:
    def test_generated_text_key(self, parser: ResponseParser):
        raw = {"generated_text": "SageMaker output", "model_id": "my-endpoint"}
        result = parser.parse(raw, ProviderFormat.SAGEMAKER)

        assert result.text == "SageMaker output"
        assert result.model_id == "my-endpoint"

    def test_outputs_key(self, parser: ResponseParser):
        raw = {"outputs": "output text"}
        result = parser.parse(raw, ProviderFormat.SAGEMAKER)
        assert result.text == "output text"

    def test_predictions_list(self, parser: ResponseParser):
        raw = {"predictions": ["first prediction", "second"]}
        result = parser.parse(raw, ProviderFormat.SAGEMAKER)
        assert result.text == "first prediction"

    def test_list_of_dicts(self, parser: ResponseParser):
        raw = {"generated_text": [{"generated_text": "nested"}]}
        result = parser.parse(raw, ProviderFormat.SAGEMAKER)
        assert result.text == "nested"

    def test_token_estimation_when_missing(
        self, parser: ResponseParser
    ):
        raw = {"generated_text": "Some generated text here"}
        result = parser.parse(raw, ProviderFormat.SAGEMAKER)
        assert result.output_tokens > 0

    def test_endpoint_name_fallback(
        self, parser: ResponseParser
    ):
        raw = {"generated_text": "x", "endpoint_name": "ep-1"}
        result = parser.parse(raw, ProviderFormat.SAGEMAKER)
        assert result.model_id == "ep-1"

    def test_sagemaker_uses_fallback_latency(
        self, parser: ResponseParser
    ):
        raw = {"generated_text": "x"}
        result = parser.parse(raw, ProviderFormat.SAGEMAKER, latency_ms=55.0)
        assert result.latency_ms == 55.0


# --- External API (OpenAI-compatible) ---


class TestExternalAPIParsing:
    def test_full_openai_response(self, parser: ResponseParser):
        raw = {
            "choices": [
                {
                    "message": {"content": "API response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            "model": "gpt-4",
        }
        result = parser.parse(raw, ProviderFormat.EXTERNAL_API)

        assert result.text == "API response"
        assert result.input_tokens == 10
        assert result.output_tokens == 3
        assert result.finish_reason == "stop"
        assert result.model_id == "gpt-4"

    def test_empty_choices(self, parser: ResponseParser):
        raw = {"choices": [], "usage": {}, "model": "gpt-4"}
        result = parser.parse(raw, ProviderFormat.EXTERNAL_API)
        assert result.text == ""
        assert result.finish_reason == "unknown"

    def test_missing_usage(self, parser: ResponseParser):
        raw = {
            "choices": [
                {
                    "message": {"content": "hi"},
                    "finish_reason": "length",
                }
            ],
            "model": "gpt-3.5",
        }
        result = parser.parse(raw, ProviderFormat.EXTERNAL_API)
        assert result.text == "hi"
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.finish_reason == "length"


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_dict(self, parser: ResponseParser):
        result = parser.parse({}, ProviderFormat.BEDROCK)
        assert result.text == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_none_input(self, parser: ResponseParser):
        result = parser.parse(
            None, ProviderFormat.BEDROCK  # type: ignore[arg-type]
        )
        assert result.text == ""
        assert result.raw is None

    def test_non_dict_input(self, parser: ResponseParser):
        result = parser.parse(
            "not a dict", ProviderFormat.BEDROCK  # type: ignore[arg-type]
        )
        assert result.text == ""

    def test_malformed_token_values(self, parser: ResponseParser):
        raw = {
            "response_text": "ok",
            "input_tokens": "not_a_number",
            "output_tokens": None,
        }
        result = parser.parse(raw, ProviderFormat.BEDROCK)
        assert result.text == "ok"
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_malformed_latency(self, parser: ResponseParser):
        raw = {"response_text": "ok", "latency_ms": "bad"}
        result = parser.parse(raw, ProviderFormat.BEDROCK, latency_ms=10.0)
        # Falls back to 0.0 since the raw value can't be parsed
        assert result.latency_ms == 0.0


# --- Token estimation helper ---


class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_short_string(self):
        assert _estimate_tokens("hi") >= 1

    def test_longer_string(self):
        result = _estimate_tokens("a" * 100)
        assert result == 25
