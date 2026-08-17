"""
Improvement Metrics Calculator for comparative model evaluation.

Takes aggregate metrics from two models (model 1 = baseline, model 2 =
fine-tuned/comparison) and computes ImprovementMetrics with deltas and
percentage changes for trust score, hallucination rate, latency, and cost.

Requirements: 7.4
"""

from src.data_models.evaluation import AggregateMetrics, ImprovementMetrics


def _safe_percent_change(baseline: float, delta: float) -> float:
    """Calculate percentage change relative to baseline.

    Returns 0.0 when baseline is zero to avoid division by zero.

    Args:
        baseline: The reference value (model 1 metric).
        delta: The absolute change (model_2 - model_1 or model_1 - model_2).

    Returns:
        Percentage change as a float (e.g. 10.0 for 10%).
    """
    if baseline == 0.0:
        return 0.0
    return (delta / baseline) * 100.0


def calculate_improvement_metrics(
    model_1_metrics: AggregateMetrics,
    model_2_metrics: AggregateMetrics,
) -> ImprovementMetrics:
    """Calculate improvement metrics between two models.

    Model 1 is the baseline; model 2 is the fine-tuned or comparison model.

    Deltas:
    - trust_score_delta: model_2 - model_1 (positive = improvement)
    - hallucination_reduction: model_1 - model_2 (positive = improvement)
    - latency_delta_ms: model_2 - model_1 (negative = faster)
    - cost_delta_per_query: model_2 - model_1 (negative = cheaper)

    Percentage changes are relative to model 1. When model 1's value is
    zero, the percentage is reported as 0.0.

    statistical_significance, p_value, and confidence_interval are left
    at defaults (0.0, 1.0, (0.0, 0.0)) — they are populated by the
    statistical significance tester (task 11.15).

    Args:
        model_1_metrics: Aggregate metrics for the baseline model.
        model_2_metrics: Aggregate metrics for the comparison model.

    Returns:
        ImprovementMetrics with all delta and percentage fields populated.
    """
    # Trust score delta (positive = model 2 is better)
    trust_delta = (
        model_2_metrics.mean_trust_score
        - model_1_metrics.mean_trust_score
    )
    trust_delta_pct = _safe_percent_change(
        model_1_metrics.mean_trust_score, trust_delta
    )

    # Hallucination reduction (positive = fewer hallucinations)
    hall_reduction = (
        model_1_metrics.mean_hallucination_rate
        - model_2_metrics.mean_hallucination_rate
    )
    hall_reduction_pct = _safe_percent_change(
        model_1_metrics.mean_hallucination_rate,
        hall_reduction,
    )

    # Latency delta using p50 (positive = model 2 is slower)
    latency_delta = (
        model_2_metrics.latency_p50_ms
        - model_1_metrics.latency_p50_ms
    )
    latency_delta_pct = _safe_percent_change(
        model_1_metrics.latency_p50_ms, latency_delta
    )

    # Cost delta per query (positive = more expensive)
    cost_delta = (
        model_2_metrics.cost_per_query
        - model_1_metrics.cost_per_query
    )
    cost_delta_pct = _safe_percent_change(
        model_1_metrics.cost_per_query, cost_delta
    )

    return ImprovementMetrics(
        trust_score_delta=trust_delta,
        trust_score_delta_percent=trust_delta_pct,
        hallucination_reduction=hall_reduction,
        hallucination_reduction_percent=hall_reduction_pct,
        latency_delta_ms=latency_delta,
        latency_delta_percent=latency_delta_pct,
        cost_delta_per_query=cost_delta,
        cost_delta_percent=cost_delta_pct,
        statistical_significance=0.0,
        p_value=1.0,
        confidence_interval=(0.0, 0.0),
    )
