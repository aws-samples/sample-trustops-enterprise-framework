"""
Unit tests for per-model timeout configuration.

Requirements: 3.11, 3.12, 3.18, 3.19
"""

import pytest

from src.evaluation.timeout_config import (
    TimeoutConfig,
    resolve_timeout,
    _classify_model_size,
)


# ---------------------------------------------------------------------------
# TimeoutConfig defaults
# ---------------------------------------------------------------------------


class TestTimeoutConfigDefaults:
    """Verify built-in default values."""

    def test_default_small(self):
        cfg = TimeoutConfig()
        assert cfg.small == 30.0

    def test_default_medium(self):
        cfg = TimeoutConfig()
        assert cfg.medium == 60.0

    def test_default_large(self):
        cfg = TimeoutConfig()
        assert cfg.large == 120.0

    def test_custom_values(self):
        cfg = TimeoutConfig(small=10.0, medium=20.0, large=40.0)
        assert cfg.small == 10.0
        assert cfg.medium == 20.0
        assert cfg.large == 40.0


# ---------------------------------------------------------------------------
# Model size classification
# ---------------------------------------------------------------------------


class TestClassifyModelSize:
    """Keyword-based model size classification."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic.claude-3-haiku-20240307",
            "amazon.titan-lite-v1",
            "meta.llama-mini-8b",
            "some-small-model",
            "nano-gpt",
            "tiny-llm-v2",
            "claude-instant-v1",
        ],
    )
    def test_small_models(self, model_id: str):
        assert _classify_model_size(model_id) == "small"

    @pytest.mark.parametrize(
        "model_id",
        [
            "meta.llama-70b-instruct",
            "anthropic.claude-3-opus-20240229",
            "amazon.titan-large-v1",
            "some-xl-model",
            "ultra-gpt-v3",
        ],
    )
    def test_large_models(self, model_id: str):
        assert _classify_model_size(model_id) == "large"

    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic.claude-3-sonnet-20240229",
            "amazon.titan-text-express-v1",
            "meta.llama-13b-chat",
            "my-custom-model",
        ],
    )
    def test_medium_models(self, model_id: str):
        assert _classify_model_size(model_id) == "medium"


# ---------------------------------------------------------------------------
# resolve_timeout
# ---------------------------------------------------------------------------


class TestResolveTimeout:
    """Timeout resolution with overrides and size-based defaults."""

    def test_default_config_medium(self):
        timeout = resolve_timeout("anthropic.claude-3-sonnet")
        assert timeout == 60.0

    def test_default_config_small(self):
        timeout = resolve_timeout("anthropic.claude-3-haiku")
        assert timeout == 30.0

    def test_default_config_large(self):
        timeout = resolve_timeout("anthropic.claude-3-opus")
        assert timeout == 120.0

    def test_custom_config(self):
        cfg = TimeoutConfig(small=5.0, medium=10.0, large=20.0)
        assert resolve_timeout("haiku-model", cfg) == 5.0
        assert resolve_timeout("sonnet-model", cfg) == 10.0
        assert resolve_timeout("opus-model", cfg) == 20.0

    def test_model_override_takes_precedence(self):
        cfg = TimeoutConfig(
            model_overrides={"my-special-model": 999.0}
        )
        assert resolve_timeout("my-special-model", cfg) == 999.0

    def test_none_config_uses_defaults(self):
        timeout = resolve_timeout("some-model", None)
        assert timeout == 60.0
