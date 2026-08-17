"""
Unit tests for Hallucination-Trust Score integration.

Tests that the HallucinationDetector results are properly integrated
into the context_grounding dimension of the TrustScoringEngine.

Requirements: 6.11 - Integrate hallucination score into Trust_Score
as a weighted component of the overall score.
"""

import pytest
from unittest.mock import Mock

from src.data_models.hallucination import (
    HallucinationResult,
    SensitivityLevel,
)
from src.data_models.trust_score import (
    DimensionScore,
    TrustDimension,
    TrustScoreConfig,
    TrustScoreWeights,
)
from src.trust_scoring.scorers.context_grounding_scorer import (
    ContextGroundingScorer,
)
from src.trust_scoring.trust_scoring_engine_v2 import (
    TrustScoringEngine,
)


def _make_hallucination_result(
    hallucination_rate: float = 0.0,
    overall_grounding_score: float = 1.0,
) -> HallucinationResult:
    """Helper to build a minimal HallucinationResult."""
    return HallucinationResult(
        has_hallucinations=hallucination_rate > 0.0,
        hallucination_rate=hallucination_rate,
        total_claims=10,
        factual_claims=10,
        supported_claims=int(10 * (1 - hallucination_rate)),
        unsupported_claims=int(10 * hallucination_rate),
        flagged_spans=[],
        overall_grounding_score=overall_grounding_score,
        claim_evidence=[],
        sensitivity_level=SensitivityLevel.MODERATE,
    )


# -----------------------------------------------------------
# ContextGroundingScorer integration tests
# -----------------------------------------------------------
class TestGroundingScorerHallucinationBlend:
    """Test hallucination blending in ContextGroundingScorer."""

    @pytest.fixture
    def mock_analyzer(self):
        analyzer = Mock()
        analyzer.calculate_similarity = Mock(return_value=0.8)
        return analyzer

    @pytest.fixture
    def scorer(self, mock_analyzer):
        return ContextGroundingScorer(
            similarity_analyzer=mock_analyzer,
            hallucination_weight=0.4,
        )

    @pytest.mark.asyncio
    async def test_no_hallucination_result_unchanged(
        self, scorer, mock_analyzer
    ):
        """Without hallucination result, score equals similarity."""
        result = await scorer.calculate_context_grounding(
            response="Test response.",
            source_documents=["Source doc."],
            hallucination_result=None,
        )
        assert result.score == pytest.approx(0.8)
        assert result.details["hallucination_blended"] is False

    @pytest.mark.asyncio
    async def test_zero_hallucination_rate_boosts_score(
        self, scorer, mock_analyzer
    ):
        """Zero hallucination rate blends 1.0 grounding in."""
        hr = _make_hallucination_result(hallucination_rate=0.0)
        result = await scorer.calculate_context_grounding(
            response="Test response.",
            source_documents=["Source doc."],
            hallucination_result=hr,
        )
        # 0.8 * 0.6 + 1.0 * 0.4 = 0.48 + 0.40 = 0.88
        assert result.score == pytest.approx(0.88)
        assert result.details["hallucination_blended"] is True
        assert result.details["hallucination_grounding_score"] == 1.0

    @pytest.mark.asyncio
    async def test_high_hallucination_rate_lowers_score(
        self, scorer, mock_analyzer
    ):
        """High hallucination rate pulls grounding score down."""
        hr = _make_hallucination_result(hallucination_rate=0.8)
        result = await scorer.calculate_context_grounding(
            response="Test response.",
            source_documents=["Source doc."],
            hallucination_result=hr,
        )
        # 0.8 * 0.6 + 0.2 * 0.4 = 0.48 + 0.08 = 0.56
        assert result.score == pytest.approx(0.56)
        assert result.details["hallucination_blended"] is True

    @pytest.mark.asyncio
    async def test_full_hallucination_rate(
        self, scorer, mock_analyzer
    ):
        """100% hallucination rate gives hallucination grounding 0."""
        hr = _make_hallucination_result(hallucination_rate=1.0)
        result = await scorer.calculate_context_grounding(
            response="Test response.",
            source_documents=["Source doc."],
            hallucination_result=hr,
        )
        # 0.8 * 0.6 + 0.0 * 0.4 = 0.48
        assert result.score == pytest.approx(0.48)

    @pytest.mark.asyncio
    async def test_score_always_in_valid_range(
        self, mock_analyzer
    ):
        """Score stays in [0, 1] regardless of inputs."""
        scorer = ContextGroundingScorer(
            similarity_analyzer=mock_analyzer,
            hallucination_weight=0.9,
        )
        mock_analyzer.calculate_similarity.return_value = 0.1
        hr = _make_hallucination_result(hallucination_rate=1.0)
        result = await scorer.calculate_context_grounding(
            response="Test.",
            source_documents=["Src."],
            hallucination_result=hr,
        )
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_custom_hallucination_weight(self, mock_analyzer):
        """Custom hallucination_weight changes the blend ratio."""
        scorer = ContextGroundingScorer(
            similarity_analyzer=mock_analyzer,
            hallucination_weight=0.5,
        )
        mock_analyzer.calculate_similarity.return_value = 0.6
        hr = _make_hallucination_result(hallucination_rate=0.0)
        result = await scorer.calculate_context_grounding(
            response="Test.",
            source_documents=["Src."],
            hallucination_result=hr,
        )
        # 0.6 * 0.5 + 1.0 * 0.5 = 0.30 + 0.50 = 0.80
        assert result.score == pytest.approx(0.80)

    def test_invalid_hallucination_weight_raises(self):
        """hallucination_weight outside [0,1] raises ValueError."""
        with pytest.raises(ValueError):
            ContextGroundingScorer(hallucination_weight=1.5)
        with pytest.raises(ValueError):
            ContextGroundingScorer(hallucination_weight=-0.1)

    @pytest.mark.asyncio
    async def test_details_include_hallucination_fields(
        self, scorer, mock_analyzer
    ):
        """Details dict includes hallucination metadata."""
        hr = _make_hallucination_result(hallucination_rate=0.3)
        result = await scorer.calculate_context_grounding(
            response="Test.",
            source_documents=["Src."],
            hallucination_result=hr,
        )
        assert result.details["hallucination_blended"] is True
        assert result.details["hallucination_grounding_score"] == (
            pytest.approx(0.7)
        )
        assert result.details["hallucination_weight"] == 0.4


# -----------------------------------------------------------
# TrustScoringEngine integration tests
# -----------------------------------------------------------
class TestTrustEngineHallucinationIntegration:
    """Test TrustScoringEngine with hallucination integration."""

    @pytest.fixture
    def mock_analyzer(self):
        analyzer = Mock()
        analyzer.calculate_similarity = Mock(return_value=0.8)
        return analyzer

    @pytest.fixture
    def engine(self, mock_analyzer):
        return TrustScoringEngine(
            similarity_analyzer=mock_analyzer,
        )

    @pytest.mark.asyncio
    async def test_score_response_without_hallucination(
        self, engine
    ):
        """Engine works normally without hallucination data."""
        result = await engine.score_response(
            prompt="What is Python?",
            response="Python is a programming language.",
            source_documents=["Python is a language."],
        )
        assert 0.0 <= result.overall_score <= 1.0
        grounding = result.dimension_scores[
            TrustDimension.CONTEXT_GROUNDING
        ]
        assert grounding.details["hallucination_blended"] is False

    @pytest.mark.asyncio
    async def test_score_response_with_hallucination_result(
        self, engine
    ):
        """Passing hallucination_result enhances grounding score."""
        hr = _make_hallucination_result(hallucination_rate=0.0)
        result = await engine.score_response(
            prompt="What is Python?",
            response="Python is a programming language.",
            source_documents=["Python is a language."],
            hallucination_result=hr,
        )
        grounding = result.dimension_scores[
            TrustDimension.CONTEXT_GROUNDING
        ]
        assert grounding.details["hallucination_blended"] is True

    @pytest.mark.asyncio
    async def test_auto_detect_with_configured_detector(
        self, mock_analyzer
    ):
        """Engine auto-runs detector when configured."""
        mock_detector = Mock()
        mock_detector.detect = Mock(
            return_value=_make_hallucination_result(
                hallucination_rate=0.2
            )
        )
        engine = TrustScoringEngine(
            similarity_analyzer=mock_analyzer,
            hallucination_detector=mock_detector,
        )
        result = await engine.score_response(
            prompt="Q",
            response="A",
            source_documents=["Source"],
        )
        mock_detector.detect.assert_called_once()
        grounding = result.dimension_scores[
            TrustDimension.CONTEXT_GROUNDING
        ]
        assert grounding.details["hallucination_blended"] is True

    @pytest.mark.asyncio
    async def test_no_auto_detect_without_sources(
        self, mock_analyzer
    ):
        """Detector not called when no source_documents."""
        mock_detector = Mock()
        engine = TrustScoringEngine(
            similarity_analyzer=mock_analyzer,
            hallucination_detector=mock_detector,
        )
        await engine.score_response(
            prompt="Q",
            response="A",
        )
        mock_detector.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_result_overrides_auto_detect(
        self, mock_analyzer
    ):
        """Explicit hallucination_result skips auto-detection."""
        mock_detector = Mock()
        engine = TrustScoringEngine(
            similarity_analyzer=mock_analyzer,
            hallucination_detector=mock_detector,
        )
        hr = _make_hallucination_result(hallucination_rate=0.1)
        await engine.score_response(
            prompt="Q",
            response="A",
            source_documents=["Source"],
            hallucination_result=hr,
        )
        mock_detector.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_overall_score_affected_by_hallucination(
        self, mock_analyzer
    ):
        """High hallucination rate lowers overall trust score."""
        engine = TrustScoringEngine(
            similarity_analyzer=mock_analyzer,
        )
        # Score without hallucination
        result_clean = await engine.score_response(
            prompt="Q",
            response="A",
            source_documents=["Source"],
        )
        # Score with high hallucination
        hr_bad = _make_hallucination_result(
            hallucination_rate=0.9
        )
        result_bad = await engine.score_response(
            prompt="Q",
            response="A",
            source_documents=["Source"],
            hallucination_result=hr_bad,
        )
        # The bad hallucination should lower the overall score
        assert result_bad.overall_score < result_clean.overall_score
