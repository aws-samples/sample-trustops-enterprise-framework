"""
Response-level comparison for paired model outputs.

Takes paired responses (from PairedResponse in parallel_invoker.py) and
computes per-pair deltas for trust score, hallucination rate, latency,
cost, and semantic similarity between the two response texts.

Requirements: 7.2, 7.3
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from src.data_models.model import ModelPricing
from src.evaluation.parallel_invoker import PairedResponse

logger = logging.getLogger(__name__)


@dataclass
class ResponseComparison:
    """Comparison results for a single paired prompt.

    Attributes:
        index: Position of the prompt in the dataset.
        prompt: The prompt sent to both models.
        model_1_id: ID of the first model.
        model_2_id: ID of the second model.
        trust_score_1: Trust score for model 1 response.
        trust_score_2: Trust score for model 2 response.
        trust_score_delta: model_2 trust score minus model_1
            trust score.
        hallucination_rate_1: Hallucination rate for model 1 response.
        hallucination_rate_2: Hallucination rate for model 2 response.
        hallucination_rate_delta: model_2 rate minus model_1 rate
            (negative = improvement).
        latency_delta_ms: model_2 latency minus model_1 latency.
        cost_1: Cost for model 1 inference.
        cost_2: Cost for model 2 inference.
        cost_delta: model_2 cost minus model_1 cost.
        semantic_similarity: Text similarity between the two
            responses [0, 1].
        category: Optional category from the dataset item.
        skipped: True if comparison was skipped (e.g. one model failed).
        skip_reason: Reason the comparison was skipped.
    """

    index: int
    prompt: str
    model_1_id: str
    model_2_id: str
    trust_score_1: float = 0.0
    trust_score_2: float = 0.0
    trust_score_delta: float = 0.0
    hallucination_rate_1: float = 0.0
    hallucination_rate_2: float = 0.0
    hallucination_rate_delta: float = 0.0
    latency_delta_ms: float = 0.0
    cost_1: float = 0.0
    cost_2: float = 0.0
    cost_delta: float = 0.0
    semantic_similarity: float = 0.0
    category: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class BatchComparisonSummary:
    """Summary of comparisons across a batch of paired responses.

    Attributes:
        total: Total number of pairs.
        compared: Number of pairs successfully compared.
        skipped: Number of pairs skipped.
        comparisons: Individual comparison results.
    """

    total: int = 0
    compared: int = 0
    skipped: int = 0
    comparisons: list[ResponseComparison] = field(default_factory=list)


def calculate_semantic_similarity(text_1: str, text_2: str) -> float:
    """Calculate semantic similarity between two response texts.

    Uses ``difflib.SequenceMatcher`` for a lightweight, dependency-free
    text similarity measure.  Returns a ratio in [0, 1] where 1 means
    the texts are identical.

    Args:
        text_1: First response text.
        text_2: Second response text.

    Returns:
        Similarity ratio in [0, 1].
    """
    if not text_1 and not text_2:
        return 1.0
    if not text_1 or not text_2:
        return 0.0
    return SequenceMatcher(None, text_1, text_2).ratio()


def _calculate_query_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: Optional[ModelPricing],
) -> float:
    """Calculate cost for a single query given pricing info.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        pricing: Model pricing (None → 0.0).

    Returns:
        Cost value.
    """
    if pricing is None:
        return 0.0
    input_price = pricing.input_price_per_1k_tokens / 1000.0
    output_price = pricing.output_price_per_1k_tokens / 1000.0
    return input_tokens * input_price + output_tokens * output_price


def compare_paired_response(
    pair: PairedResponse,
    trust_score_1: float = 0.0,
    trust_score_2: float = 0.0,
    hallucination_rate_1: float = 0.0,
    hallucination_rate_2: float = 0.0,
    pricing_1: Optional[ModelPricing] = None,
    pricing_2: Optional[ModelPricing] = None,
) -> ResponseComparison:
    """Compare a single pair of model responses.

    If either model failed (response is ``None``), the comparison is
    marked as *skipped* with a reason.

    Args:
        pair: The paired response from parallel invocation.
        trust_score_1: Trust score computed for model 1's response.
        trust_score_2: Trust score computed for model 2's response.
        hallucination_rate_1: Hallucination rate for model 1's response.
        hallucination_rate_2: Hallucination rate for model 2's response.
        pricing_1: Pricing info for model 1 (optional).
        pricing_2: Pricing info for model 2 (optional).

    Returns:
        A ``ResponseComparison`` with all deltas populated.
    """
    base = ResponseComparison(
        index=pair.index,
        prompt=pair.prompt,
        model_1_id=pair.model_1_id,
        model_2_id=pair.model_2_id,
        category=pair.category,
    )

    if not pair.both_succeeded:
        reasons = []
        if pair.error_1:
            reasons.append(f"model_1 error: {pair.error_1}")
        if pair.error_2:
            reasons.append(f"model_2 error: {pair.error_2}")
        if pair.response_1 is None and pair.error_1 is None:
            reasons.append("model_1 returned no response")
        if pair.response_2 is None and pair.error_2 is None:
            reasons.append("model_2 returned no response")
        base.skipped = True
        base.skip_reason = "; ".join(reasons) if reasons else "Unknown failure"
        return base

    # Both responses are present
    resp_1 = pair.response_1
    resp_2 = pair.response_2

    # Trust score delta (positive = model 2 is better)
    base.trust_score_1 = trust_score_1
    base.trust_score_2 = trust_score_2
    base.trust_score_delta = trust_score_2 - trust_score_1

    # Hallucination rate delta (negative = model 2 improved)
    base.hallucination_rate_1 = hallucination_rate_1
    base.hallucination_rate_2 = hallucination_rate_2
    base.hallucination_rate_delta = hallucination_rate_2 - hallucination_rate_1

    # Latency delta
    base.latency_delta_ms = pair.latency_ms_2 - pair.latency_ms_1

    # Cost delta
    base.cost_1 = _calculate_query_cost(
        resp_1.input_tokens, resp_1.output_tokens, pricing_1
    )
    base.cost_2 = _calculate_query_cost(
        resp_2.input_tokens, resp_2.output_tokens, pricing_2
    )
    base.cost_delta = base.cost_2 - base.cost_1

    # Semantic similarity between the two response texts
    base.semantic_similarity = calculate_semantic_similarity(
        resp_1.text, resp_2.text
    )

    return base


def compare_batch(
    pairs: list[PairedResponse],
    trust_scores_1: Optional[list[float]] = None,
    trust_scores_2: Optional[list[float]] = None,
    hallucination_rates_1: Optional[list[float]] = None,
    hallucination_rates_2: Optional[list[float]] = None,
    pricing_1: Optional[ModelPricing] = None,
    pricing_2: Optional[ModelPricing] = None,
) -> BatchComparisonSummary:
    """Compare a batch of paired responses.

    Each list of scores/rates must have the same length as *pairs*.
    If a list is ``None``, zeros are used for every pair.

    Args:
        pairs: List of paired responses from parallel invocation.
        trust_scores_1: Trust scores for model 1 (one per pair).
        trust_scores_2: Trust scores for model 2 (one per pair).
        hallucination_rates_1: Hallucination rates for model 1.
        hallucination_rates_2: Hallucination rates for model 2.
        pricing_1: Pricing info for model 1.
        pricing_2: Pricing info for model 2.

    Returns:
        A ``BatchComparisonSummary`` with all individual comparisons.
    """
    n = len(pairs)
    ts1 = trust_scores_1 if trust_scores_1 is not None else [0.0] * n
    ts2 = trust_scores_2 if trust_scores_2 is not None else [0.0] * n
    hr1 = (
        hallucination_rates_1
        if hallucination_rates_1 is not None
        else [0.0] * n
    )
    hr2 = (
        hallucination_rates_2
        if hallucination_rates_2 is not None
        else [0.0] * n
    )

    summary = BatchComparisonSummary(total=n)
    for i, pair in enumerate(pairs):
        comp = compare_paired_response(
            pair=pair,
            trust_score_1=ts1[i],
            trust_score_2=ts2[i],
            hallucination_rate_1=hr1[i],
            hallucination_rate_2=hr2[i],
            pricing_1=pricing_1,
            pricing_2=pricing_2,
        )
        summary.comparisons.append(comp)
        if comp.skipped:
            summary.skipped += 1
        else:
            summary.compared += 1

    return summary
