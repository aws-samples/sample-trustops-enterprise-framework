"""
Unit tests for the fine-tuning cost estimator.

Tests cover cost estimation for each model family, token count handling,
edge cases, confidence levels, and display formatting.

Requirements: 4.11
"""

from src.data_models.fine_tuning import CostEstimate, HyperparameterConfig
from src.fine_tuning.cost_estimator import (
    estimate_cost,
    format_cost_display,
)


def _make_hp(**kwargs) -> HyperparameterConfig:
    """Helper to create HyperparameterConfig with defaults."""
    defaults = {"learning_rate": 1e-5, "epochs": 3, "batch_size": 8}
    defaults.update(kwargs)
    return HyperparameterConfig(**defaults)


# --- Basic cost estimation ---


class TestEstimateCost:
    def test_returns_cost_estimate(self):
        result = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        assert isinstance(result, CostEstimate)

    def test_cost_is_non_negative(self):
        result = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        assert result.estimated_training_cost >= 0
        assert result.estimated_duration_hours >= 0

    def test_cost_breakdown_has_components(self):
        result = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        assert "training_token_cost" in result.cost_breakdown
        assert "infrastructure_cost" in result.cost_breakdown

    def test_breakdown_sums_to_total(self):
        result = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        breakdown_sum = sum(result.cost_breakdown.values())
        assert abs(result.estimated_training_cost - breakdown_sum) < 0.01

    def test_currency_is_usd(self):
        result = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        assert result.currency == "USD"


# --- Model family pricing ---


class TestModelFamilyPricing:
    def test_claude_model(self):
        result = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        assert result.estimated_training_cost > 0

    def test_titan_model(self):
        result = estimate_cost("amazon.titan-text-v1", 1000, _make_hp())
        assert result.estimated_training_cost > 0

    def test_llama_model(self):
        result = estimate_cost("meta.llama3-8b-v1", 1000, _make_hp())
        assert result.estimated_training_cost > 0

    def test_unknown_model(self):
        result = estimate_cost("some-unknown-model", 1000, _make_hp())
        assert result.estimated_training_cost > 0

    def test_different_families_have_different_costs(self):
        hp = _make_hp()
        claude = estimate_cost("anthropic.claude-v2", 1000, hp)
        titan = estimate_cost("amazon.titan-text-v1", 1000, hp)
        # Different model families should produce different costs
        assert claude.estimated_training_cost != titan.estimated_training_cost


# --- Epochs scaling ---


class TestEpochsScaling:
    def test_more_epochs_costs_more(self):
        hp_2 = _make_hp(epochs=2)
        hp_5 = _make_hp(epochs=5)
        cost_2 = estimate_cost("anthropic.claude-v2", 1000, hp_2)
        cost_5 = estimate_cost("anthropic.claude-v2", 1000, hp_5)
        assert cost_5.estimated_training_cost > cost_2.estimated_training_cost

    def test_more_epochs_takes_longer(self):
        hp_2 = _make_hp(epochs=2)
        hp_5 = _make_hp(epochs=5)
        est_2 = estimate_cost("anthropic.claude-v2", 1000, hp_2)
        est_5 = estimate_cost("anthropic.claude-v2", 1000, hp_5)
        assert est_5.estimated_duration_hours > est_2.estimated_duration_hours

    def test_cost_scales_linearly_with_epochs(self):
        hp_1 = _make_hp(epochs=1)
        hp_3 = _make_hp(epochs=3)
        cost_1 = estimate_cost("anthropic.claude-v2", 1000, hp_1)
        cost_3 = estimate_cost("anthropic.claude-v2", 1000, hp_3)
        ratio = cost_3.estimated_training_cost / cost_1.estimated_training_cost
        assert abs(ratio - 3.0) < 0.01


# --- Dataset size scaling ---


class TestDatasetSizeScaling:
    def test_larger_dataset_costs_more(self):
        hp = _make_hp()
        small = estimate_cost("anthropic.claude-v2", 100, hp)
        large = estimate_cost("anthropic.claude-v2", 10000, hp)
        assert large.estimated_training_cost > small.estimated_training_cost

    def test_larger_dataset_takes_longer(self):
        hp = _make_hp()
        small = estimate_cost("anthropic.claude-v2", 100, hp)
        large = estimate_cost("anthropic.claude-v2", 10000, hp)
        assert large.estimated_duration_hours > small.estimated_duration_hours


# --- Token count handling ---


class TestTokenCount:
    def test_with_explicit_token_count(self):
        result = estimate_cost(
            "anthropic.claude-v2", 1000, _make_hp(), token_count=100_000
        )
        assert result.estimated_training_cost > 0
        assert result.confidence == "high"

    def test_without_token_count_uses_estimate(self):
        result = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        assert result.confidence == "medium"

    def test_explicit_tokens_vs_estimated_differ(self):
        hp = _make_hp()
        with_tokens = estimate_cost(
            "anthropic.claude-v2", 1000, hp, token_count=1_000_000
        )
        without_tokens = estimate_cost("anthropic.claude-v2", 1000, hp)
        # Explicit large token count should produce higher cost
        assert (
            with_tokens.estimated_training_cost
            != without_tokens.estimated_training_cost
        )


# --- Confidence levels ---


class TestConfidence:
    def test_high_confidence_with_token_count(self):
        result = estimate_cost(
            "anthropic.claude-v2", 1000, _make_hp(), token_count=50000
        )
        assert result.confidence == "high"

    def test_medium_confidence_without_token_count(self):
        result = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        assert result.confidence == "medium"

    def test_low_confidence_for_unknown_model(self):
        result = estimate_cost("unknown-model", 1000, _make_hp())
        assert result.confidence == "low"

    def test_unknown_model_overrides_token_confidence(self):
        # Even with token count, unknown model should be low confidence
        result = estimate_cost(
            "unknown-model", 1000, _make_hp(), token_count=50000
        )
        assert result.confidence == "low"


# --- Edge cases ---


class TestEdgeCases:
    def test_zero_examples(self):
        result = estimate_cost("anthropic.claude-v2", 0, _make_hp())
        assert result.estimated_training_cost == 0.0
        assert result.estimated_duration_hours == 0.0

    def test_negative_examples(self):
        result = estimate_cost("anthropic.claude-v2", -5, _make_hp())
        assert result.estimated_training_cost == 0.0

    def test_single_example(self):
        result = estimate_cost("anthropic.claude-v2", 1, _make_hp())
        assert result.estimated_training_cost > 0

    def test_very_large_dataset(self):
        result = estimate_cost("anthropic.claude-v2", 1_000_000, _make_hp())
        assert result.estimated_training_cost > 0
        assert result.estimated_duration_hours > 0

    def test_zero_token_count(self):
        # token_count=0 means no token data; falls back to estimation
        result = estimate_cost(
            "anthropic.claude-v2", 1000, _make_hp(), token_count=0
        )
        assert result.estimated_training_cost > 0
        assert result.confidence == "medium"


# --- Display formatting ---


class TestFormatCostDisplay:
    def test_returns_string(self):
        estimate = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        display = format_cost_display(estimate)
        assert isinstance(display, str)

    def test_contains_total_cost(self):
        estimate = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        display = format_cost_display(estimate)
        assert "$" in display
        assert "Estimated Total Cost" in display

    def test_contains_duration(self):
        estimate = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        display = format_cost_display(estimate)
        assert "hours" in display

    def test_contains_confidence(self):
        estimate = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        display = format_cost_display(estimate)
        assert "Confidence" in display

    def test_contains_breakdown(self):
        estimate = estimate_cost("anthropic.claude-v2", 1000, _make_hp())
        display = format_cost_display(estimate)
        assert "Cost Breakdown" in display
        assert "Training Token Cost" in display
        assert "Infrastructure Cost" in display
