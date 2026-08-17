"""Tests for the MetricsCalculator."""

import pytest

from src.data_models.hallucination import HallucinationResult
from src.data_models.trust_score import TrustScoreResult
from src.evaluation.batch_runner import BatchItemResult
from src.evaluation.metrics_calculator import (
    CalculatedMetrics,
    MetricsCalculator,
    PricingConfig,
    _mean,
    _median,
    _percentile,
    _std,
)
from src.evaluation.response_scorer import ScoredResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scored(
    index: int,
    trust_score: float,
    latency_ms: float = 100.0,
    input_tokens: int = 10,
    output_tokens: int = 20,
    hallucination_rate: float | None = None,
    success: bool = True,
) -> ScoredResponse:
    """Build a minimal ScoredResponse for testing."""
    batch = BatchItemResult(
        index=index,
        prompt=f"prompt-{index}",
        response_text="resp" if success else "",
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        success=success,
    )
    tsr = (
        TrustScoreResult(
            overall_score=trust_score,
            confidence_level=0.9,
        )
        if success
        else None
    )
    hr = None
    if hallucination_rate is not None and success:
        hr = HallucinationResult(
            has_hallucinations=hallucination_rate > 0,
            hallucination_rate=hallucination_rate,
            total_claims=5,
            factual_claims=5,
            supported_claims=int(5 * (1 - hallucination_rate)),
            unsupported_claims=int(5 * hallucination_rate),
            flagged_spans=[],
            overall_grounding_score=1.0 - hallucination_rate,
            claim_evidence=[],
            sensitivity_level="moderate",
        )
    return ScoredResponse(
        batch_item=batch,
        trust_score_result=tsr,
        hallucination_result=hr,
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_single(self):
        assert _mean([5.0]) == 5.0

    def test_mean_multiple(self):
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_median_empty(self):
        assert _median([]) == 0.0

    def test_median_odd(self):
        assert _median([1.0, 2.0, 3.0]) == 2.0

    def test_median_even(self):
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_std_single(self):
        assert _std([5.0], 5.0) == 0.0

    def test_std_uniform(self):
        assert _std([2.0, 2.0, 2.0], 2.0) == 0.0

    def test_std_known(self):
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        m = _mean(vals)
        assert _std(vals, m) == pytest.approx(2.0, abs=0.01)

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0

    def test_percentile_single(self):
        assert _percentile([7.0], 99) == 7.0

    def test_percentile_p50(self):
        vals = sorted([10.0, 20.0, 30.0, 40.0, 50.0])
        assert _percentile(vals, 50) == 30.0

    def test_percentile_p95_interpolation(self):
        vals = sorted(float(i) for i in range(1, 101))
        assert _percentile(vals, 95) == pytest.approx(95.05, abs=0.1)


# ---------------------------------------------------------------------------
# MetricsCalculator tests
# ---------------------------------------------------------------------------

class TestMetricsCalculator:
    def test_empty_list(self):
        calc = MetricsCalculator()
        result = calc.calculate([])
        assert result == CalculatedMetrics()
        assert result.total_examples == 0

    def test_all_failures(self):
        responses = [
            _make_scored(0, 0.0, success=False),
            _make_scored(1, 0.0, success=False),
        ]
        calc = MetricsCalculator()
        result = calc.calculate(responses)
        assert result.total_examples == 2
        assert result.successful_examples == 0
        assert result.failed_examples == 2
        assert result.mean_trust_score == 0.0

    def test_single_item(self):
        responses = [_make_scored(0, 0.8, latency_ms=150.0)]
        calc = MetricsCalculator()
        result = calc.calculate(responses)
        assert result.total_examples == 1
        assert result.successful_examples == 1
        assert result.failed_examples == 0
        assert result.mean_trust_score == pytest.approx(0.8)
        assert result.median_trust_score == pytest.approx(0.8)
        assert result.trust_score_std == 0.0  # single item
        assert result.min_trust_score == pytest.approx(0.8)
        assert result.max_trust_score == pytest.approx(0.8)
        assert result.latency_p50_ms == pytest.approx(150.0)

    def test_multiple_items_trust_stats(self):
        responses = [
            _make_scored(0, 0.6, latency_ms=100.0),
            _make_scored(1, 0.8, latency_ms=200.0),
            _make_scored(2, 0.7, latency_ms=150.0),
        ]
        calc = MetricsCalculator()
        result = calc.calculate(responses)
        assert result.total_examples == 3
        assert result.successful_examples == 3
        assert result.mean_trust_score == pytest.approx(0.7, abs=0.001)
        assert result.median_trust_score == pytest.approx(0.7)
        assert result.min_trust_score == pytest.approx(0.6)
        assert result.max_trust_score == pytest.approx(0.8)
        # std of [0.6, 0.7, 0.8] = sqrt(((0.1^2)*2)/3) ≈ 0.0816
        assert result.trust_score_std == pytest.approx(0.0816, abs=0.001)

    def test_latency_percentiles(self):
        # 10 items with latencies 10..100
        responses = [
            _make_scored(
                i, 0.5, latency_ms=float((i + 1) * 10)
            )
            for i in range(10)
        ]
        calc = MetricsCalculator()
        result = calc.calculate(responses)
        assert result.latency_p50_ms == pytest.approx(55.0)
        assert result.latency_p95_ms > result.latency_p50_ms
        assert result.latency_p99_ms >= result.latency_p95_ms

    def test_token_totals(self):
        responses = [
            _make_scored(0, 0.5, input_tokens=100, output_tokens=50),
            _make_scored(1, 0.5, input_tokens=200, output_tokens=100),
        ]
        calc = MetricsCalculator()
        result = calc.calculate(responses)
        assert result.total_input_tokens == 300
        assert result.total_output_tokens == 150

    def test_cost_calculation(self):
        pricing = PricingConfig(
            input_price_per_token=0.001,
            output_price_per_token=0.002,
        )
        responses = [
            _make_scored(0, 0.5, input_tokens=100, output_tokens=50),
            _make_scored(1, 0.5, input_tokens=200, output_tokens=100),
        ]
        calc = MetricsCalculator(pricing=pricing)
        result = calc.calculate(responses)
        # cost = 300*0.001 + 150*0.002 = 0.3 + 0.3 = 0.6
        assert result.total_cost == pytest.approx(0.6)
        assert result.cost_per_query == pytest.approx(0.3)

    def test_hallucination_rate_aggregation(self):
        responses = [
            _make_scored(0, 0.5, hallucination_rate=0.2),
            _make_scored(1, 0.5, hallucination_rate=0.4),
        ]
        calc = MetricsCalculator()
        result = calc.calculate(responses)
        assert result.mean_hallucination_rate == pytest.approx(0.3)

    def test_mixed_success_and_failure(self):
        responses = [
            _make_scored(
                0, 0.9, latency_ms=50.0,
                input_tokens=10, output_tokens=5,
            ),
            _make_scored(1, 0.0, success=False),
            _make_scored(
                2, 0.7, latency_ms=80.0,
                input_tokens=20, output_tokens=10,
            ),
        ]
        calc = MetricsCalculator()
        result = calc.calculate(responses)
        assert result.total_examples == 3
        assert result.successful_examples == 2
        assert result.failed_examples == 1
        assert result.mean_trust_score == pytest.approx(0.8)
        assert result.total_input_tokens == 30
        assert result.total_output_tokens == 15

    def test_no_hallucination_results(self):
        """When no hallucination results are present, mean rate should be 0."""
        responses = [_make_scored(0, 0.5)]
        calc = MetricsCalculator()
        result = calc.calculate(responses)
        assert result.mean_hallucination_rate == 0.0

    def test_default_pricing_zero_cost(self):
        """Default pricing (0.0) should produce zero cost."""
        responses = [
            _make_scored(
                0, 0.5,
                input_tokens=1000, output_tokens=500,
            )
        ]
        calc = MetricsCalculator()
        result = calc.calculate(responses)
        assert result.total_cost == 0.0
        assert result.cost_per_query == 0.0


# ---------------------------------------------------------------------------
# Per-category metrics (calculate_by_category) tests — Requirement 3.6
# ---------------------------------------------------------------------------

class TestCalculateByCategory:
    """Tests for MetricsCalculator.calculate_by_category."""

    @staticmethod
    def _make_scored_with_category(
        index: int,
        trust_score: float,
        category: str | None = None,
        latency_ms: float = 100.0,
        input_tokens: int = 10,
        output_tokens: int = 20,
        success: bool = True,
    ) -> ScoredResponse:
        batch = BatchItemResult(
            index=index,
            prompt=f"prompt-{index}",
            response_text="resp" if success else "",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            category=category,
        )
        tsr = (
            TrustScoreResult(overall_score=trust_score, confidence_level=0.9)
            if success
            else None
        )
        return ScoredResponse(batch_item=batch, trust_score_result=tsr)

    def test_empty_list(self):
        calc = MetricsCalculator()
        result = calc.calculate_by_category([])
        assert result == {}

    def test_single_category(self):
        responses = [
            self._make_scored_with_category(0, 0.8, category="math"),
            self._make_scored_with_category(1, 0.6, category="math"),
        ]
        calc = MetricsCalculator()
        result = calc.calculate_by_category(responses)
        assert list(result.keys()) == ["math"]
        assert result["math"].total_examples == 2
        assert result["math"].mean_trust_score == pytest.approx(0.7)

    def test_multiple_categories(self):
        responses = [
            self._make_scored_with_category(0, 0.9, category="math"),
            self._make_scored_with_category(1, 0.5, category="science"),
            self._make_scored_with_category(2, 0.7, category="math"),
        ]
        calc = MetricsCalculator()
        result = calc.calculate_by_category(responses)
        assert set(result.keys()) == {"math", "science"}
        assert result["math"].total_examples == 2
        assert result["science"].total_examples == 1
        assert result["math"].mean_trust_score == pytest.approx(0.8)
        assert result["science"].mean_trust_score == pytest.approx(0.5)

    def test_none_category_becomes_uncategorized(self):
        responses = [
            self._make_scored_with_category(0, 0.6, category=None),
            self._make_scored_with_category(1, 0.8, category="legal"),
        ]
        calc = MetricsCalculator()
        result = calc.calculate_by_category(responses)
        assert "uncategorized" in result
        assert "legal" in result
        assert result["uncategorized"].total_examples == 1
        assert result["uncategorized"].mean_trust_score == pytest.approx(0.6)

    def test_all_uncategorized(self):
        responses = [
            self._make_scored_with_category(0, 0.5, category=None),
            self._make_scored_with_category(1, 0.7, category=None),
        ]
        calc = MetricsCalculator()
        result = calc.calculate_by_category(responses)
        assert list(result.keys()) == ["uncategorized"]
        assert result["uncategorized"].total_examples == 2

    def test_failed_items_counted_in_category(self):
        responses = [
            self._make_scored_with_category(0, 0.8, category="qa"),
            self._make_scored_with_category(1, 0.0, category="qa", success=False),
        ]
        calc = MetricsCalculator()
        result = calc.calculate_by_category(responses)
        assert result["qa"].total_examples == 2
        assert result["qa"].successful_examples == 1
        assert result["qa"].failed_examples == 1
