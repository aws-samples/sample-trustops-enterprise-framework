"""
Response Parser for provider-specific inference response formats.

Parses raw provider responses into a normalized ParsedResponse format,
extracting response text, token counts, latency, and finish reason
from Bedrock, SageMaker, and External API (OpenAI-compatible) formats.

Requirements: 3.3, 3.6
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProviderFormat(str, Enum):
    """Supported provider response formats."""

    BEDROCK = "bedrock"
    SAGEMAKER = "sagemaker"
    EXTERNAL_API = "external_api"


@dataclass
class ParsedResponse:
    """Normalized response parsed from a provider-specific format.

    Attributes:
        text: The generated response text.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        latency_ms: Response latency in milliseconds.
        finish_reason: Reason generation stopped (e.g. stop, length).
        model_id: Identifier of the model that produced the response.
        raw: The original raw response dict (preserved for debugging).
    """

    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "unknown"
    model_id: str = ""
    raw: Optional[dict[str, Any]] = None


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class ResponseParser:
    """Parses raw provider responses into normalized ParsedResponse objects.

    Handles Bedrock, SageMaker, and External API (OpenAI-compatible) formats
    with graceful fallbacks for missing or malformed fields.
    """

    def parse(
        self,
        raw: dict[str, Any],
        provider: ProviderFormat,
        latency_ms: float = 0.0,
    ) -> ParsedResponse:
        """Parse a raw provider response dict into a ParsedResponse.

        Args:
            raw: The raw response dictionary from the provider.
            provider: Which provider format to expect.
            latency_ms: Externally measured latency in ms (used as fallback).

        Returns:
            A normalized ParsedResponse.
        """
        if not raw or not isinstance(raw, dict):
            logger.warning("Empty or non-dict response received")
            return ParsedResponse(
                latency_ms=latency_ms,
                raw=raw if isinstance(raw, dict) else None,
            )

        if provider == ProviderFormat.BEDROCK:
            return self._parse_bedrock(raw, latency_ms)
        elif provider == ProviderFormat.SAGEMAKER:
            return self._parse_sagemaker(raw, latency_ms)
        elif provider == ProviderFormat.EXTERNAL_API:
            return self._parse_external_api(raw, latency_ms)
        else:
            logger.warning(
                "Unknown provider format: %s, attempting generic parse",
                provider,
            )
            return self._parse_generic(raw, latency_ms)

    def _parse_bedrock(
        self, raw: dict[str, Any], latency_ms: float
    ) -> ParsedResponse:
        """Parse a Bedrock-style response.

        Expected keys: response_text, input_tokens, output_tokens,
        latency_ms, model_id.
        """
        text = str(raw.get("response_text", ""))
        input_tokens = self._safe_int(raw.get("input_tokens", 0))
        output_tokens = self._safe_int(raw.get("output_tokens", 0))
        resp_latency = self._safe_float(raw.get("latency_ms", latency_ms))
        model_id = str(raw.get("model_id", ""))
        finish_reason = str(raw.get("finish_reason", "stop"))

        return ParsedResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=resp_latency,
            finish_reason=finish_reason,
            model_id=model_id,
            raw=raw,
        )

    def _parse_sagemaker(
        self, raw: dict[str, Any], latency_ms: float
    ) -> ParsedResponse:
        """Parse a SageMaker-style response.

        SageMaker endpoints return varied formats. Common keys:
        generated_text, outputs, predictions, text.
        Token counts are often absent and must be estimated.
        """
        text = self._extract_sagemaker_text(raw)
        model_id = str(raw.get("model_id", raw.get("endpoint_name", "")))

        # SageMaker rarely provides token counts; estimate if missing
        input_tokens = self._safe_int(raw.get("input_tokens", 0))
        output_tokens = self._safe_int(raw.get("output_tokens", 0))
        if output_tokens == 0 and text:
            output_tokens = _estimate_tokens(text)

        resp_latency = self._safe_float(raw.get("latency_ms", latency_ms))
        finish_reason = str(raw.get("finish_reason", "stop"))

        return ParsedResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=resp_latency,
            finish_reason=finish_reason,
            model_id=model_id,
            raw=raw,
        )

    def _extract_sagemaker_text(self, raw: dict[str, Any]) -> str:
        """Extract generated text from various SageMaker response shapes."""
        # Direct text fields
        for key in ("generated_text", "outputs", "predictions", "text"):
            value = raw.get(key)
            if value is not None:
                if isinstance(value, str):
                    return value
                if isinstance(value, list) and value:
                    first = value[0]
                    if isinstance(first, str):
                        return first
                    if isinstance(first, dict):
                        return str(
                            first.get("generated_text", first)
                        )
        return ""

    def _parse_external_api(
        self, raw: dict[str, Any], latency_ms: float
    ) -> ParsedResponse:
        """Parse an OpenAI-compatible API response.

        Expected structure::

            {
                "choices": [
                    {"message": {"content": "..."},
                     "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": N,
                    "completion_tokens": N
                },
                "model": "model-id"
            }
        """
        # Extract text from choices
        text = ""
        finish_reason = "unknown"
        choices = raw.get("choices", [])
        if choices and isinstance(choices, list):
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message", {})
            if isinstance(message, dict):
                text = str(message.get("content", ""))
            finish_reason = str(first_choice.get("finish_reason", "stop"))

        # Extract token usage
        usage = raw.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        input_tokens = self._safe_int(usage.get("prompt_tokens", 0))
        output_tokens = self._safe_int(usage.get("completion_tokens", 0))

        model_id = str(raw.get("model", ""))
        resp_latency = self._safe_float(raw.get("latency_ms", latency_ms))

        return ParsedResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=resp_latency,
            finish_reason=finish_reason,
            model_id=model_id,
            raw=raw,
        )

    def _parse_generic(
        self, raw: dict[str, Any], latency_ms: float
    ) -> ParsedResponse:
        """Best-effort parse for unknown provider formats."""
        text = str(
            raw.get(
                "text",
                raw.get("response_text", raw.get("generated_text", "")),
            )
        )
        input_tokens = self._safe_int(raw.get("input_tokens", 0))
        output_tokens = self._safe_int(raw.get("output_tokens", 0))
        resp_latency = self._safe_float(raw.get("latency_ms", latency_ms))
        model_id = str(raw.get("model_id", raw.get("model", "")))
        finish_reason = str(raw.get("finish_reason", "unknown"))

        return ParsedResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=resp_latency,
            finish_reason=finish_reason,
            model_id=model_id,
            raw=raw,
        )

    @staticmethod
    def _safe_int(value: Any) -> int:
        """Safely convert a value to int, returning 0 on failure."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Safely convert a value to float, returning 0.0 on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
