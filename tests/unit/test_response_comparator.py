"""
Unit tests for the response-level comparator.

Tests cover:
- Per-pair trust score delta calculation
- Hallucination rate delta calculation
- Latency delta calculation
- Cost delta calculation
- Semantic similarity between paired responses
- Skipped comparisons when a model fails
- Batch comparison across multiple pairs

Requirements: 7.2, 7.3
"""

import pytest

from src.data_models.model import InferenceResponse, ModelPricing
from src.evaluation.parallel_invoker import PairedResponse
from src.evaluation.response_comparator import (
    calculate_semantic_similarity,
    compare_batch,
    compare_paired_response,
)


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _resp(
    text: str = "answer",
    input_tokens: int = 10,
    output_tokens: int = 5,
    latency_ms: float = 50.0,
    model_id: str = "m1",
) -> InferenceResponse:
    return InferenceResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_id=model_id,
        finish_reason="stop",
    )


def _pair(
    text_1: str = "hello world",
    text_2: str = "hello world",
    latency_1: float = 100.0,
    latency_2: float = 120.0,
    input_tokens_1: int = 10,
    output_tokens_1: int = 20,
    input_tokens_2: int = 10,
    output_tokens_2: int = 25,
    error_1: str | None = None,
    error_2: str | None = None,
    category: str | None = None,
) -> PairedResponse:
    resp_1 = None if error_1 else _resp(
        text=text_1,
        input_tokens=input_tokens_1,
        output_tokens=output_tokens_1,
        model_id="model-a",
    )
    resp_2 = None if error_2 else _resp(
        text=text_2,
        input_tokens=input_tokens_2,
        output_tokens=output_tokens_2,
        model_id="model-b",
    )
    return PairedResponse(
        index=0,
        prompt="test prompt",
        model_1_id="model-a",
        model_2_id="model-b",
        response_1=resp_1,
        response_2=resp_2,
        error_1=error_1,
        error_2=error_2,
        latency_ms_1=latency_1,
        latency_ms_2=latency_2,
        category=category,
    )


PRICING = ModelPricing(
    input_price_per_1k_tokens=0.01,
    output_price_per_1k_tokens=0.03,
)


# -----------------------------------------------------------
# Semantic similarity
# -----------------------------------------------------------


class TestSemanticSimilarity:
    def test_identical_texts(self):
        assert calculate_semantic_similarity("abc", "abc") == 1.0

    def test_completely_different(self):
        sim = calculate_semantic_similarity("aaa", "zzz")
        assert sim < 0.5

    def test_both_empty(self):
        assert calculate_semantic_similarity("", "") == 1.0

    def test_one_empty(self):
        assert calculate_semantic_similarity("hello", "") == 0.0
        assert calculate_semantic_similarity("", "hello") == 0.0

    def test_partial_overlap(self):
        sim = calculate_semantic_similarity(
            "the quick brown fox",
            "the quick red fox",
        )
        assert 0.5 < sim < 1.0


# -----------------------------------------------------------
# Single pair comparison
# -----------------------------------------------------------


class TestComparePairedResponse:
    def test_trust_score_delta(self):
        pair = _pair()
        comp = compare_paired_response(
            pair, trust_score_1=0.7, trust_score_2=0.9
        )
        assert comp.trust_score_delta == pytest.approx(0.2)
        assert comp.trust_score_1 == pytest.approx(0.7)
        assert comp.trust_score_2 == pytest.approx(0.9)

    def test_hallucination_rate_delta(self):
        pair = _pair()
        comp = compare_paired_response(
            pair,
            hallucination_rate_1=0.4,
            hallucination_rate_2=0.1,
        )
        assert comp.hallucination_rate_delta == pytest.approx(-0.3)

    def test_latency_delta(self):
        pair = _pair(latency_1=100.0, latency_2=150.0)
        comp = compare_paired_response(pair)
        assert comp.latency_delta_ms == pytest.approx(50.0)

    def test_cost_delta_with_pricing(self):
        pair = _pair(
            input_tokens_1=100,
            output_tokens_1=50,
            input_tokens_2=100,
            output_tokens_2=80,
        )
        comp = compare_paired_response(
            pair, pricing_1=PRICING, pricing_2=PRICING
        )
        # model 1: 100*0.00001 + 50*0.00003 = 0.001 + 0.0015 = 0.0025
        # model 2: 100*0.00001 + 80*0.00003 = 0.001 + 0.0024 = 0.0034
        assert comp.cost_1 == pytest.approx(0.0025)
        assert comp.cost_2 == pytest.approx(0.0034)
        assert comp.cost_delta == pytest.approx(0.0009)

    def test_cost_zero_without_pricing(self):
        pair = _pair()
        comp = compare_paired_response(pair)
        assert comp.cost_1 == 0.0
        assert comp.cost_2 == 0.0
        assert comp.cost_delta == 0.0

    def test_semantic_similarity_identical(self):
        pair = _pair(text_1="same text", text_2="same text")
        comp = compare_paired_response(pair)
        assert comp.semantic_similarity == 1.0

    def test_semantic_similarity_different(self):
        pair = _pair(
            text_1="the cat sat on the mat",
            text_2="completely unrelated text here",
        )
        comp = compare_paired_response(pair)
        assert 0.0 <= comp.semantic_similarity < 1.0

    def test_category_preserved(self):
        pair = _pair(category="finance")
        comp = compare_paired_response(pair)
        assert comp.category == "finance"

    def test_not_skipped_when_both_succeed(self):
        pair = _pair()
        comp = compare_paired_response(pair)
        assert comp.skipped is False
        assert comp.skip_reason is None


# -----------------------------------------------------------
# Skipped comparisons
# -----------------------------------------------------------


class TestSkippedComparisons:
    def test_skipped_when_model_1_fails(self):
        pair = _pair(error_1="timeout")
        comp = compare_paired_response(pair)
        assert comp.skipped is True
        assert "model_1 error" in comp.skip_reason

    def test_skipped_when_model_2_fails(self):
        pair = _pair(error_2="rate limited")
        comp = compare_paired_response(pair)
        assert comp.skipped is True
        assert "model_2 error" in comp.skip_reason

    def test_skipped_when_both_fail(self):
        pair = _pair(error_1="err1", error_2="err2")
        comp = compare_paired_response(pair)
        assert comp.skipped is True
        assert "model_1" in comp.skip_reason
        assert "model_2" in comp.skip_reason

    def test_skipped_deltas_are_zero(self):
        pair = _pair(error_1="fail")
        comp = compare_paired_response(pair)
        assert comp.trust_score_delta == 0.0
        assert comp.hallucination_rate_delta == 0.0
        assert comp.latency_delta_ms == 0.0
        assert comp.cost_delta == 0.0
        assert comp.semantic_similarity == 0.0


# -----------------------------------------------------------
# Batch comparison
# -----------------------------------------------------------


class TestCompareBatch:
    def test_empty_batch(self):
        summary = compare_batch([])
        assert summary.total == 0
        assert summary.compared == 0
        assert summary.skipped == 0
        assert summary.comparisons == []

    def test_single_pair(self):
        pairs = [_pair()]
        summary = compare_batch(
            pairs,
            trust_scores_1=[0.6],
            trust_scores_2=[0.8],
        )
        assert summary.total == 1
        assert summary.compared == 1
        assert summary.skipped == 0
        comp = summary.comparisons[0]
        assert comp.trust_score_delta == pytest.approx(0.2)

    def test_mixed_success_and_failure(self):
        pairs = [
            _pair(),
            _pair(error_1="fail"),
            _pair(),
        ]
        summary = compare_batch(pairs)
        assert summary.total == 3
        assert summary.compared == 2
        assert summary.skipped == 1

    def test_defaults_to_zero_scores(self):
        pairs = [_pair()]
        summary = compare_batch(pairs)
        comp = summary.comparisons[0]
        assert comp.trust_score_1 == 0.0
        assert comp.trust_score_2 == 0.0
        assert comp.hallucination_rate_1 == 0.0
        assert comp.hallucination_rate_2 == 0.0

    def test_pricing_forwarded(self):
        pairs = [_pair(
            input_tokens_1=100,
            output_tokens_1=50,
            input_tokens_2=100,
            output_tokens_2=50,
        )]
        summary = compare_batch(
            pairs, pricing_1=PRICING, pricing_2=PRICING
        )
        comp = summary.comparisons[0]
        assert comp.cost_1 > 0.0
        assert comp.cost_2 > 0.0
