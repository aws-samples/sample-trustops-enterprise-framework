"""
Cost-Performance Analyzer for comparative evaluation.

Takes aggregate metrics from two models and produces a
CostPerformanceAnalysis that calculates cost per trust score
point, determines whether quality gains justify cost changes,
and computes break-even volume when applicable.

Requirements: 7.10
"""

import math
from typing import Optional

from src.data_models.evaluation import (
    AggregateMetrics,
    CostPerformanceAnalysis,
)


def _cost_per_trust_point(
    cost_per_query: float,
    mean_trust_score: float,
) -> float:
    """Calculate cost per trust score point.

    Returns ``inf`` when *mean_trust_score* is zero.

    Args:
        cost_per_query: Average cost per query.
        mean_trust_score: Mean trust score in [0, 1].

    Returns:
        Cost per trust score point.
    """
    if mean_trust_score == 0.0:
        return float("inf")
    return cost_per_query / mean_trust_score


def _compute_break_even_volume(
    model_1_metrics: AggregateMetrics,
    model_2_metrics: AggregateMetrics,
) -> Optional[int]:
    """Compute break-even query volume.

    Break-even applies when model 2 is more expensive per query
    but has a better trust score.  Each query with model 2 yields
    a trust "gain" but costs extra.  We value one full trust point
    at model 1's cost-per-trust-point, giving::

        value = trust_delta * m1_cost_per_trust_point
        break_even = ceil(cost_delta / value)

    Returns None when:
    - Model 2 is not more expensive.
    - Model 2 does not have a better trust score.
    - Model 1 has zero trust score.
    """
    cost_delta = (
        model_2_metrics.cost_per_query
        - model_1_metrics.cost_per_query
    )
    trust_delta = (
        model_2_metrics.mean_trust_score
        - model_1_metrics.mean_trust_score
    )

    # Break-even only applies when model 2 costs more AND has better trust
    if cost_delta <= 0 or trust_delta <= 0:
        return None

    # Value the trust improvement using model 1's cost-per-trust-point
    if model_1_metrics.mean_trust_score == 0.0:
        return None

    value_per_query = trust_delta * (
        model_1_metrics.cost_per_query
        / model_1_metrics.mean_trust_score
    )

    if value_per_query <= 0:
        return None

    return math.ceil(cost_delta / value_per_query)


# Threshold for "significant" trust improvement (5 percentage points)
_SIGNIFICANT_TRUST_IMPROVEMENT = 0.05


def analyze_cost_performance(
    model_1_metrics: AggregateMetrics,
    model_2_metrics: AggregateMetrics,
) -> CostPerformanceAnalysis:
    """Analyze cost-performance trade-offs between two models.

    Model 1 is the baseline; model 2 is the comparison model.

    ``quality_gain_justifies_cost`` is True when model 2 has a
    better (lower) cost-per-trust-point ratio, or when the trust
    improvement is significant (>= 5 pp).

    Args:
        model_1_metrics: Aggregate metrics for baseline model.
        model_2_metrics: Aggregate metrics for comparison model.

    Returns:
        CostPerformanceAnalysis with cost-per-trust-point,
        justification flag, and optional break-even volume.
    """
    m1_cptp = _cost_per_trust_point(
        model_1_metrics.cost_per_query, model_1_metrics.mean_trust_score
    )
    m2_cptp = _cost_per_trust_point(
        model_2_metrics.cost_per_query, model_2_metrics.mean_trust_score
    )

    trust_delta = (
        model_2_metrics.mean_trust_score
        - model_1_metrics.mean_trust_score
    )

    # Model 2 has better ratio, or trust improvement is significant enough
    quality_justified = (
        m2_cptp < m1_cptp
        or trust_delta >= _SIGNIFICANT_TRUST_IMPROVEMENT
    )

    break_even = _compute_break_even_volume(model_1_metrics, model_2_metrics)

    return CostPerformanceAnalysis(
        model_1_cost_per_trust_point=m1_cptp,
        model_2_cost_per_trust_point=m2_cptp,
        quality_gain_justifies_cost=quality_justified,
        break_even_volume=break_even,
    )
