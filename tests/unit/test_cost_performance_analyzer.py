"""Tests for the cost-performance analyzer.

Requirements: 7.10
"""

import pytest

from src.data_models.evaluation import (
    AggregateMetrics,
    CostPerformanceAnalysis,
)
from src.evaluation.cost_performance_analyzer import (
    _cost_per_trust_point,
    _compute_break_even_volume,
    analyze_cost_performance,
)


def _make_metrics(
    mean_trust_score: float = 0.7,
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
        mean_hallucination_rate=0.2,
        latency_p50_ms=100.0,
        latency_p95_ms=150.0,
        latency_p99_ms=200.0,
        total_input_tokens=10000,
        total_output_tokens=5000,
        total_cost=cost_per_query * 100,
        cost_per_query=cost_per_query,
    )


# ── _cost_per_trust_point ──────────────────────────────────────────


class TestCostPerTrustPoint:
    def test_normal_calculation(self):
        # 0.01 / 0.5 = 0.02
        assert _cost_per_trust_point(0.01, 0.5) == pytest.approx(0.02)

    def test_zero_trust_score_returns_inf(self):
        assert _cost_per_trust_point(0.01, 0.0) == float("inf")

    def test_zero_cost_returns_zero(self):
        assert _cost_per_trust_point(0.0, 0.8) == pytest.approx(0.0)

    def test_both_zero(self):
        assert _cost_per_trust_point(0.0, 0.0) == float("inf")

    def test_high_trust_low_cost(self):
        result = _cost_per_trust_point(0.001, 0.95)
        assert result == pytest.approx(0.001 / 0.95)


# ── _compute_break_even_volume ─────────────────────────────────────


class TestComputeBreakEvenVolume:
    def test_model2_cheaper_returns_none(self):
        m1 = _make_metrics(cost_per_query=0.02, mean_trust_score=0.7)
        m2 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.8)
        assert _compute_break_even_volume(m1, m2) is None

    def test_model2_same_cost_returns_none(self):
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.7)
        m2 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.8)
        assert _compute_break_even_volume(m1, m2) is None

    def test_model2_worse_trust_returns_none(self):
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.8)
        m2 = _make_metrics(cost_per_query=0.02, mean_trust_score=0.7)
        assert _compute_break_even_volume(m1, m2) is None

    def test_model2_same_trust_returns_none(self):
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.7)
        m2 = _make_metrics(cost_per_query=0.02, mean_trust_score=0.7)
        assert _compute_break_even_volume(m1, m2) is None

    def test_model1_zero_trust_returns_none(self):
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.0)
        m2 = _make_metrics(cost_per_query=0.02, mean_trust_score=0.5)
        assert _compute_break_even_volume(m1, m2) is None

    def test_break_even_calculated(self):
        # m1: cost=0.01, trust=0.6 → cptp = 0.01/0.6
        # m2: cost=0.02, trust=0.8
        # cost_delta = 0.01, trust_delta = 0.2
        # value_per_query = 0.2 * (0.01/0.6) = 0.2/60 = 1/300
        # break_even = ceil(0.01 / (1/300)) = ceil(3) = 3
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.6)
        m2 = _make_metrics(cost_per_query=0.02, mean_trust_score=0.8)
        result = _compute_break_even_volume(m1, m2)
        assert result is not None
        assert result == 3

    def test_break_even_rounds_up(self):
        # m1: cost=0.01, trust=0.5 → cptp = 0.02
        # m2: cost=0.015, trust=0.7
        # cost_delta = 0.005, trust_delta = 0.2
        # value_per_query = 0.2 * 0.02 = 0.004
        # break_even = ceil(0.005 / 0.004) = ceil(1.25) = 2
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.5)
        m2 = _make_metrics(cost_per_query=0.015, mean_trust_score=0.7)
        result = _compute_break_even_volume(m1, m2)
        assert result is not None
        assert result == 2


# ── analyze_cost_performance ───────────────────────────────────────


class TestAnalyzeCostPerformance:
    def test_returns_cost_performance_analysis_type(self):
        m1 = _make_metrics()
        m2 = _make_metrics()
        result = analyze_cost_performance(m1, m2)
        assert isinstance(result, CostPerformanceAnalysis)

    def test_identical_models(self):
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.7)
        m2 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.7)
        result = analyze_cost_performance(m1, m2)
        assert result.model_1_cost_per_trust_point == pytest.approx(
            result.model_2_cost_per_trust_point
        )
        # Same ratio → not strictly less, and 0 trust delta < 0.05
        assert result.quality_gain_justifies_cost is False
        assert result.break_even_volume is None

    def test_model2_better_ratio(self):
        # m2 cheaper per trust point → justified
        m1 = _make_metrics(cost_per_query=0.02, mean_trust_score=0.6)
        m2 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.8)
        result = analyze_cost_performance(m1, m2)
        assert (
            result.model_2_cost_per_trust_point
            < result.model_1_cost_per_trust_point
        )
        assert result.quality_gain_justifies_cost is True
        # m2 is cheaper, so no break-even needed
        assert result.break_even_volume is None

    def test_model2_worse_ratio_but_significant_trust_gain(self):
        # m2 worse cost-per-trust-point but trust delta >= 0.05
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.6)
        m2 = _make_metrics(cost_per_query=0.03, mean_trust_score=0.7)
        result = analyze_cost_performance(m1, m2)
        assert (
            result.model_2_cost_per_trust_point
            > result.model_1_cost_per_trust_point
        )
        # trust delta = 0.1 >= 0.05 → justified
        assert result.quality_gain_justifies_cost is True
        # m2 more expensive + better trust → break-even exists
        assert result.break_even_volume is not None
        assert result.break_even_volume > 0

    def test_model2_worse_ratio_and_small_trust_gain(self):
        # m2 worse ratio and trust delta < 0.05
        m1 = _make_metrics(
            cost_per_query=0.01, mean_trust_score=0.70,
        )
        m2 = _make_metrics(
            cost_per_query=0.03, mean_trust_score=0.72,
        )
        result = analyze_cost_performance(m1, m2)
        assert result.quality_gain_justifies_cost is False

    def test_model2_worse_trust_not_justified(self):
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.8)
        m2 = _make_metrics(cost_per_query=0.02, mean_trust_score=0.6)
        result = analyze_cost_performance(m1, m2)
        assert result.quality_gain_justifies_cost is False
        assert result.break_even_volume is None

    def test_zero_trust_model1(self):
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.0)
        m2 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.5)
        result = analyze_cost_performance(m1, m2)
        assert result.model_1_cost_per_trust_point == float("inf")
        # m2 cptp is finite, so m2 < m1 → justified
        assert result.quality_gain_justifies_cost is True

    def test_zero_cost_both_models(self):
        m1 = _make_metrics(cost_per_query=0.0, mean_trust_score=0.7)
        m2 = _make_metrics(cost_per_query=0.0, mean_trust_score=0.8)
        result = analyze_cost_performance(m1, m2)
        assert result.model_1_cost_per_trust_point == pytest.approx(0.0)
        assert result.model_2_cost_per_trust_point == pytest.approx(0.0)
        # Same ratio (both 0), but trust delta = 0.1 >= 0.05 → justified
        assert result.quality_gain_justifies_cost is True

    def test_break_even_volume_present_when_applicable(self):
        m1 = _make_metrics(cost_per_query=0.01, mean_trust_score=0.6)
        m2 = _make_metrics(cost_per_query=0.02, mean_trust_score=0.8)
        result = analyze_cost_performance(m1, m2)
        assert result.break_even_volume is not None
        assert isinstance(result.break_even_volume, int)
        assert result.break_even_volume >= 1
