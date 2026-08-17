"""
Unit tests for the per-response scorer.

Tests use mocked TrustScoringEngine and HallucinationDetector to verify:
- Trust scoring is called for each successful response
- Hallucination detection runs when detector and source docs are provided
- Failed batch items are skipped gracefully
- Scoring errors are captured without crashing the batch
- score_batch processes all items even when one fails

Requirements: 3.4
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data_models.hallucination import HallucinationResult, SensitivityLevel
from src.data_models.trust_score import (
    DimensionScore,
    TrustDimension,
    TrustScoreResult,
)
from src.evaluation.batch_runner import BatchItemResult
from src.evaluation.response_scorer import ResponseScorer, ScoredResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trust_result(overall: float = 0.85) -> TrustScoreResult:
    return TrustScoreResult(
        overall_score=overall,
        dimension_scores={
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY, score=0.9,
                confidence=0.8, details={}, checks_passed=[], checks_failed=[],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY, score=0.8,
                confidence=0.7, details={}, checks_passed=[], checks_failed=[],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY, score=0.9,
                confidence=0.9, details={}, checks_passed=[], checks_failed=[],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS, score=0.8,
                confidence=0.7, details={}, checks_passed=[], checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING, score=0.85,
                confidence=0.8, details={}, checks_passed=[], checks_failed=[],
            ),
        },
        confidence_level=0.8,
        flagged_for_review=False,
        explanation="Good",
        component_details=[],
    )


def _make_hallucination_result() -> HallucinationResult:
    return HallucinationResult(
        has_hallucinations=False,
        hallucination_rate=0.0,
        total_claims=2,
        factual_claims=2,
        supported_claims=2,
        unsupported_claims=0,
        flagged_spans=[],
        overall_grounding_score=0.9,
        claim_evidence=[],
        sensitivity_level=SensitivityLevel.MODERATE,
    )


def _make_batch_item(
    index: int = 0,
    success: bool = True,
    response_text: str = "The answer is 42.",
    prompt: str = "What is the answer?",
    expected: str = "42",
    error: str | None = None,
) -> BatchItemResult:
    return BatchItemResult(
        index=index,
        prompt=prompt,
        response_text=response_text,
        success=success,
        error=error,
        expected_response=expected,
    )


def _make_trust_engine(trust_result=None) -> AsyncMock:
    engine = AsyncMock()
    engine.score_response = AsyncMock(
        return_value=trust_result or _make_trust_result()
    )
    return engine


def _make_detector(hallucination_result=None) -> MagicMock:
    detector = MagicMock()
    detector.detect = MagicMock(
        return_value=hallucination_result or _make_hallucination_result()
    )
    return detector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_score_response_with_trust_only():
    """Trust scoring runs even without a hallucination detector."""
    engine = _make_trust_engine()
    scorer = ResponseScorer(trust_engine=engine)

    item = _make_batch_item()
    result = await scorer.score_response(item)

    assert isinstance(result, ScoredResponse)
    assert result.trust_score_result is not None
    assert result.hallucination_result is None
    assert result.scoring_error is None
    engine.score_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_score_response_with_hallucination_detection():
    """Hallucination detection runs when detector and source docs provided."""
    engine = _make_trust_engine()
    detector = _make_detector()
    scorer = ResponseScorer(trust_engine=engine, hallucination_detector=detector)

    item = _make_batch_item()
    docs = ["Source document content."]
    result = await scorer.score_response(item, source_documents=docs)

    assert result.trust_score_result is not None
    assert result.hallucination_result is not None
    detector.detect.assert_called_once_with(
        response=item.response_text, source_documents=docs
    )
    # Trust engine should receive the hallucination result
    call_kwargs = engine.score_response.call_args.kwargs
    assert call_kwargs["hallucination_result"] is not None


@pytest.mark.asyncio
async def test_score_response_no_docs_skips_hallucination():
    """Hallucination detection is skipped when no source documents."""
    engine = _make_trust_engine()
    detector = _make_detector()
    scorer = ResponseScorer(trust_engine=engine, hallucination_detector=detector)

    item = _make_batch_item()
    result = await scorer.score_response(item, source_documents=None)

    assert result.hallucination_result is None
    detector.detect.assert_not_called()


@pytest.mark.asyncio
async def test_score_response_failed_item_skipped():
    """Failed batch items return a ScoredResponse with error, no scoring."""
    engine = _make_trust_engine()
    scorer = ResponseScorer(trust_engine=engine)

    item = _make_batch_item(success=False, response_text="", error="Timeout")
    result = await scorer.score_response(item)

    assert result.trust_score_result is None
    assert result.scoring_error == "Timeout"
    engine.score_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_score_response_handles_engine_exception():
    """If trust engine raises, error is captured gracefully."""
    engine = AsyncMock()
    engine.score_response = AsyncMock(side_effect=RuntimeError("boom"))
    scorer = ResponseScorer(trust_engine=engine)

    item = _make_batch_item()
    result = await scorer.score_response(item)

    assert result.trust_score_result is None
    assert result.scoring_error == "boom"


@pytest.mark.asyncio
async def test_score_batch_processes_all_items():
    """score_batch returns one ScoredResponse per input item."""
    engine = _make_trust_engine()
    scorer = ResponseScorer(trust_engine=engine)

    items = [_make_batch_item(index=i) for i in range(3)]
    results = await scorer.score_batch(items)

    assert len(results) == 3
    assert all(r.trust_score_result is not None for r in results)


@pytest.mark.asyncio
async def test_score_batch_continues_on_failure():
    """If one item fails scoring, the rest still get scored."""
    call_count = 0

    async def _side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("engine error")
        return _make_trust_result()

    engine = AsyncMock()
    engine.score_response = AsyncMock(side_effect=_side_effect)
    scorer = ResponseScorer(trust_engine=engine)

    items = [_make_batch_item(index=i) for i in range(3)]
    results = await scorer.score_batch(items)

    assert len(results) == 3
    # First and third should succeed
    assert results[0].trust_score_result is not None
    assert results[0].scoring_error is None
    # Second should have error
    assert results[1].trust_score_result is None
    assert results[1].scoring_error == "engine error"
    # Third should succeed
    assert results[2].trust_score_result is not None
    assert results[2].scoring_error is None
