"""
Aggregate Metrics Calculator for the Evaluation Engine.

Computes summary statistics from a list of ScoredResponse objects,
including trust score stats, latency percentiles, token totals, and cost.

Requirements: 3.5
"""

import math
from dataclasses import dataclass
from typing import Optional

from src.evaluation.response_scorer import ScoredResponse


@dataclass
class PricingConfig:
    """Per-token pricing for cost calculation.

    Attributes:
        input_price_per_token: Cost per input token.
        output_price_per_token: Cost per output token.
    """

    input_price_per_token: float = 0.0
    output_price_per_token: float = 0.0


@dataclass
class CalculatedMetrics:
    """Aggregated metrics computed from scored responses.

    Attributes:
        total_examples: Total number of responses processed.
        successful_examples: Number with successful scoring.
        failed_examples: Number that failed scoring or inference.
        mean_trust_score: Mean of trust scores from successful items.
        median_trust_score: Median of trust scores.
        trust_score_std: Standard deviation of trust scores.
        min_trust_score: Minimum trust score.
        max_trust_score: Maximum trust score.
        mean_hallucination_rate: Mean hallucination rate.
        latency_p50_ms: 50th percentile latency.
        latency_p95_ms: 95th percentile latency.
        latency_p99_ms: 99th percentile latency.
        total_input_tokens: Sum of input tokens.
        total_output_tokens: Sum of output tokens.
        total_cost: Total cost based on pricing.
        cost_per_query: Average cost per successful query.
    """

    total_examples: int = 0
    successful_examples: int = 0
    failed_examples: int = 0
    mean_trust_score: float = 0.0
    median_trust_score: float = 0.0
    trust_score_std: float = 0.0
    min_trust_score: float = 0.0
    max_trust_score: float = 0.0
    mean_hallucination_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    cost_per_query: float = 0.0


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the *p*-th percentile using linear interpolation.

    Args:
        sorted_values: Pre-sorted list of floats (ascending).
        p: Percentile in [0, 100].

    Returns:
        Interpolated percentile value.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (p / 100.0) * (len(sorted_values) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def _mean(values: list[float]) -> float:
    """Return arithmetic mean, or 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(sorted_values: list[float]) -> float:
    """Return median of a pre-sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _std(values: list[float], mean_val: float) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


class MetricsCalculator:
    """Calculates aggregate metrics from a list of ScoredResponse objects.

    Usage::

        calc = MetricsCalculator(pricing=PricingConfig(...))
        metrics = calc.calculate(scored_responses)

    Requirements: 3.5
    """

    def __init__(self, pricing: Optional[PricingConfig] = None):
        """
        Args:
            pricing: Optional per-token pricing config for cost calculation.
        """
        self.pricing = pricing or PricingConfig()

    def calculate(self, responses: list[ScoredResponse]) -> CalculatedMetrics:
        """Compute aggregate metrics from scored responses.

        Args:
            responses: List of ScoredResponse objects from the ResponseScorer.

        Returns:
            CalculatedMetrics with all summary statistics.
        """
        if not responses:
            return CalculatedMetrics()

        total = len(responses)

        # Separate successful vs failed
        successful: list[ScoredResponse] = []
        failed = 0
        for r in responses:
            if r.batch_item.success and r.trust_score_result is not None:
                successful.append(r)
            else:
                failed += 1

        if not successful:
            return CalculatedMetrics(
                total_examples=total,
                successful_examples=0,
                failed_examples=total,
            )

        # Trust scores
        trust_scores = [
            r.trust_score_result.overall_score for r in successful
        ]
        trust_scores_sorted = sorted(trust_scores)
        mean_ts = _mean(trust_scores)
        median_ts = _median(trust_scores_sorted)
        std_ts = _std(trust_scores, mean_ts)
        min_ts = trust_scores_sorted[0]
        max_ts = trust_scores_sorted[-1]

        # Hallucination rates
        hall_rates: list[float] = []
        for r in successful:
            if r.hallucination_result is not None:
                hall_rates.append(r.hallucination_result.hallucination_rate)
        mean_hall = _mean(hall_rates)

        # Latency (from all successful batch items)
        latencies = sorted(r.batch_item.latency_ms for r in successful)
        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)

        # Tokens
        total_input = sum(r.batch_item.input_tokens for r in successful)
        total_output = sum(r.batch_item.output_tokens for r in successful)

        # Cost
        total_cost = (
            total_input * self.pricing.input_price_per_token
            + total_output * self.pricing.output_price_per_token
        )
        cost_per_query = total_cost / len(successful) if successful else 0.0

        return CalculatedMetrics(
            total_examples=total,
            successful_examples=len(successful),
            failed_examples=failed,
            mean_trust_score=mean_ts,
            median_trust_score=median_ts,
            trust_score_std=std_ts,
            min_trust_score=min_ts,
            max_trust_score=max_ts,
            mean_hallucination_rate=mean_hall,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost=total_cost,
            cost_per_query=cost_per_query,
        )

    def calculate_by_category(
        self, responses: list[ScoredResponse]
    ) -> dict[str, CalculatedMetrics]:
        """Compute per-category aggregate metrics.

        Groups responses by ``batch_item.category`` and calculates
        :class:`CalculatedMetrics` for each group.  Items whose category
        is ``None`` are placed in the ``"uncategorized"`` group.

        Args:
            responses: List of ScoredResponse objects.

        Returns:
            Dict mapping category name to CalculatedMetrics.

        Requirements: 3.6
        """
        groups: dict[str, list[ScoredResponse]] = {}
        for r in responses:
            cat = r.batch_item.category or "uncategorized"
            groups.setdefault(cat, []).append(r)
        return {cat: self.calculate(items) for cat, items in groups.items()}

