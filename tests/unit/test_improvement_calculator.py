"""Tests for the improvement metrics calculator.

Requirements: 7.4
"""

import pytest

from src.data_models.evaluation import AggregateMetrics, ImprovementMetrics
from src.evaluation.improvement_calculator import (
    _safe_percent_change,
    calculate_improvement_metrics,
)


def _make_metrics(
    mean_trust_score: float = 0.7,
    mean_hallucination_rate: float = 0.3,
    latency_p50_ms: float = 100.0,
    cost_per_query: float = 0.01,
) -> AggregateMetrics:
    """Helper to build AggregateMetrics with sensible defaults."""
    return AggregateMetrics(
        total_examples=100,
        successful_examples=100,
        failed_examples=0,
        mean_trust_score=mean_trust_score,
        median_trust_score=mean_trust_score,
        trust_score_std=0.05,
        mean_hallucination_rate=mean_hallucination_rate,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p50_ms * 1.5,
        latency_p99_ms=latency_p50_ms * 2.0,
        total_input_tokens=10000,
        total_output_tokens=5000,
        total_cost=cost_per_query * 100,
        cost_per_query=cost_per_query,
    )


class TestSafePercentChange:
    """Tests for the _safe_percent_change helper."""

    def test_normal_increase(self):
        assert _safe_percent_change(100.0, 10.0) == pytest.approx(10.0)

    def test_normal_decrease(self):
        assert _safe_percent_change(100.0, -25.0) == pytest.approx(-25.0)

    def test_zero_baseline_returns_zero(self):
        assert _safe_percent_change(0.0, 5.0) == 0.0

    def test_zero_delta(self):
        assert _safe_percent_change(50.0, 0.0) == 0.0

    def test_both_zero(self):
        assert _safe_percent_change(0.0, 0.0) == 0.0

    def test_small_baseline(self):
        result = _safe_percent_change(0.01, 0.01)
        assert result == pytest.approx(100.0)


class TestCalculateImprovementMetrics:
    """Tests for calculate_improvement_metrics."""

    def test_returns_improvement_metrics_type(self):
        m1 = _make_metrics()
        m2 = _make_metrics()
        result = calculate_improvement_metrics(m1, m2)
        assert isinstance(result, ImprovementMetrics)

    def test_identical_models_zero_deltas(self):
        m1 = _make_metrics()
        m2 = _make_metrics()
        result = calculate_improvement_metrics(m1, m2)
        assert result.trust_score_delta == pytest.approx(0.0)
        assert result.trust_score_delta_percent == pytest.approx(0.0)
        assert result.hallucination_reduction == pytest.approx(0.0)
        assert result.hallucination_reduction_percent == pytest.approx(0.0)
        assert result.latency_delta_ms == pytest.approx(0.0)
        assert result.latency_delta_percent == pytest.approx(0.0)
        assert result.cost_delta_per_query == pytest.approx(0.0)
        assert result.cost_delta_percent == pytest.approx(0.0)

    def test_trust_score_improvement(self):
        m1 = _make_metrics(mean_trust_score=0.6)
        m2 = _make_metrics(mean_trust_score=0.8)
        result = calculate_improvement_metrics(m1, m2)
        assert result.trust_score_delta == pytest.approx(0.2)
        # 0.2 / 0.6 * 100 ≈ 33.33%
        expected_pct = 100.0 * 0.2 / 0.6
        assert result.trust_score_delta_percent == pytest.approx(
            expected_pct
        )

    def test_trust_score_degradation(self):
        m1 = _make_metrics(mean_trust_score=0.8)
        m2 = _make_metrics(mean_trust_score=0.6)
        result = calculate_improvement_metrics(m1, m2)
        assert result.trust_score_delta == pytest.approx(-0.2)
        assert result.trust_score_delta_percent < 0

    def test_hallucination_reduction_improvement(self):
        m1 = _make_metrics(mean_hallucination_rate=0.4)
        m2 = _make_metrics(mean_hallucination_rate=0.2)
        result = calculate_improvement_metrics(m1, m2)
        # reduction = 0.4 - 0.2 = 0.2 (positive = improvement)
        assert result.hallucination_reduction == pytest.approx(0.2)
        assert result.hallucination_reduction_percent == pytest.approx(
            50.0
        )

    def test_hallucination_increase(self):
        m1 = _make_metrics(mean_hallucination_rate=0.2)
        m2 = _make_metrics(mean_hallucination_rate=0.4)
        result = calculate_improvement_metrics(m1, m2)
        # reduction = 0.2 - 0.4 = -0.2 (negative = worse)
        assert result.hallucination_reduction == pytest.approx(-0.2)
        assert result.hallucination_reduction_percent == pytest.approx(
            -100.0
        )

    def test_latency_improvement(self):
        m1 = _make_metrics(latency_p50_ms=200.0)
        m2 = _make_metrics(latency_p50_ms=150.0)
        result = calculate_improvement_metrics(m1, m2)
        # delta = 150 - 200 = -50 (negative = faster)
        assert result.latency_delta_ms == pytest.approx(-50.0)
        assert result.latency_delta_percent == pytest.approx(-25.0)

    def test_latency_degradation(self):
        m1 = _make_metrics(latency_p50_ms=100.0)
        m2 = _make_metrics(latency_p50_ms=150.0)
        result = calculate_improvement_metrics(m1, m2)
        assert result.latency_delta_ms == pytest.approx(50.0)
        assert result.latency_delta_percent == pytest.approx(50.0)

    def test_cost_reduction(self):
        m1 = _make_metrics(cost_per_query=0.02)
        m2 = _make_metrics(cost_per_query=0.01)
        result = calculate_improvement_metrics(m1, m2)
        assert result.cost_delta_per_query == pytest.approx(-0.01)
        assert result.cost_delta_percent == pytest.approx(-50.0)

    def test_cost_increase(self):
        m1 = _make_metrics(cost_per_query=0.01)
        m2 = _make_metrics(cost_per_query=0.03)
        result = calculate_improvement_metrics(m1, m2)
        assert result.cost_delta_per_query == pytest.approx(0.02)
        assert result.cost_delta_percent == pytest.approx(200.0)

    def test_zero_baseline_trust_score(self):
        m1 = _make_metrics(mean_trust_score=0.0)
        m2 = _make_metrics(mean_trust_score=0.5)
        result = calculate_improvement_metrics(m1, m2)
        assert result.trust_score_delta == pytest.approx(0.5)
        # division by zero guard
        assert result.trust_score_delta_percent == 0.0

    def test_zero_baseline_hallucination_rate(self):
        m1 = _make_metrics(mean_hallucination_rate=0.0)
        m2 = _make_metrics(mean_hallucination_rate=0.0)
        result = calculate_improvement_metrics(m1, m2)
        assert result.hallucination_reduction == pytest.approx(0.0)
        assert result.hallucination_reduction_percent == 0.0

    def test_zero_baseline_latency(self):
        m1 = _make_metrics(latency_p50_ms=0.0)
        m2 = _make_metrics(latency_p50_ms=50.0)
        result = calculate_improvement_metrics(m1, m2)
        assert result.latency_delta_ms == pytest.approx(50.0)
        assert result.latency_delta_percent == 0.0

    def test_zero_baseline_cost(self):
        m1 = _make_metrics(cost_per_query=0.0)
        m2 = _make_metrics(cost_per_query=0.005)
        result = calculate_improvement_metrics(m1, m2)
        assert result.cost_delta_per_query == pytest.approx(0.005)
        assert result.cost_delta_percent == 0.0

    def test_defaults_for_statistical_fields(self):
        """stat_significance, p_value, CI are defaults."""
        m1 = _make_metrics()
        m2 = _make_metrics()
        result = calculate_improvement_metrics(m1, m2)
        assert result.statistical_significance == 0.0
        assert result.p_value == 1.0
        assert result.confidence_interval == (0.0, 0.0)

    def test_all_metrics_improve(self):
        """Model 2 is better in every dimension."""
        m1 = _make_metrics(
            mean_trust_score=0.5,
            mean_hallucination_rate=0.5,
            latency_p50_ms=200.0,
            cost_per_query=0.04,
        )
        m2 = _make_metrics(
            mean_trust_score=0.8,
            mean_hallucination_rate=0.1,
            latency_p50_ms=100.0,
            cost_per_query=0.02,
        )
        result = calculate_improvement_metrics(m1, m2)
        assert result.trust_score_delta > 0
        assert result.hallucination_reduction > 0
        assert result.latency_delta_ms < 0
        assert result.cost_delta_per_query < 0
