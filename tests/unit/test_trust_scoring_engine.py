"""
Unit tests for TrustScoringEngine V2.
"""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine
from src.data_models.trust_score import (
    DimensionScore,
    TrustDimension,
    TrustScoreConfig,
    TrustScoreResult,
    TrustScoreWeights,
)


class TestTrustScoringEngineV2:
    """Test suite for TrustScoringEngine V2."""

    @pytest.fixture
    def mock_similarity_analyzer(self):
        """Create mock similarity analyzer."""
        analyzer = Mock()
        analyzer.calculate_similarity = Mock(return_value=0.8)
        return analyzer

    @pytest.fixture
    def engine(self, mock_similarity_analyzer):
        """Create TrustScoringEngine V2 instance with mock analyzer."""
        return TrustScoringEngine(
            similarity_analyzer=mock_similarity_analyzer,
        )

    async def test_score_response_basic(self, engine):
        """Test basic score_response call returns TrustScoreResult."""
        result = await engine.score_response(
            prompt="What is the capital of France?",
            response="The capital of France is Paris.",
            source_documents=["Paris is the capital and largest city of France."],
        )

        assert isinstance(result, TrustScoreResult)
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.confidence_level <= 1.0
        assert isinstance(result.flagged_for_review, bool)
        assert isinstance(result.explanation, str)

    async def test_score_response_dimension_scores_present(self, engine):
        """Test that all five dimension scores are returned."""
        result = await engine.score_response(
            prompt="Test prompt",
            response="Test response",
            source_documents=["Test document"],
        )

        assert TrustDimension.ACCURACY in result.dimension_scores
        assert TrustDimension.CONSISTENCY in result.dimension_scores
        assert TrustDimension.SAFETY in result.dimension_scores
        assert TrustDimension.BIAS in result.dimension_scores
        assert TrustDimension.CONTEXT_GROUNDING in result.dimension_scores

        for dim, score in result.dimension_scores.items():
            assert isinstance(score, DimensionScore)
            assert 0.0 <= score.score <= 1.0
            assert 0.0 <= score.confidence <= 1.0

    async def test_score_response_overall_in_valid_range(self, engine):
        """Test that overall trust score is always in [0, 1] range."""
        test_cases = [
            ("Short", []),
            ("A very long response with lots of detail and information.", ["doc"]),
            ("Maybe this could be correct.", ["doc"]),
        ]

        for response, docs in test_cases:
            result = await engine.score_response(
                prompt="prompt",
                response=response,
                source_documents=docs,
            )
            assert 0.0 <= result.overall_score <= 1.0

    async def test_score_response_flagged_for_review(self, engine):
        """Test flagged_for_review when score is below review threshold."""
        result = await engine.score_response(
            prompt="Test",
            response="Test response text that is fairly complete.",
            source_documents=[],
        )

        # The engine flags when overall_score < review_threshold (default 0.6)
        if result.overall_score < 0.6:
            assert result.flagged_for_review is True
        else:
            assert result.flagged_for_review is False

    async def test_score_response_explanation_not_empty(self, engine):
        """Test that explanation is generated and non-empty."""
        result = await engine.score_response(
            prompt="What is AI?",
            response="AI is artificial intelligence.",
            source_documents=["AI is a field of computer science."],
        )

        assert len(result.explanation) > 0

    async def test_score_response_component_details(self, engine):
        """Test that component_details list is populated."""
        result = await engine.score_response(
            prompt="Test",
            response="Test response.",
            source_documents=["Test doc."],
        )

        assert len(result.component_details) == 5
        for detail in result.component_details:
            assert "dimension" in detail
            assert "score" in detail
            assert "confidence" in detail

    async def test_score_response_no_source_documents(self, engine):
        """Test score_response when no source documents provided."""
        result = await engine.score_response(
            prompt="What is Python?",
            response="Python is a programming language.",
        )

        assert isinstance(result, TrustScoreResult)
        assert 0.0 <= result.overall_score <= 1.0

    async def test_score_response_with_expected_response(self, engine):
        """Test score_response with expected_response for accuracy scoring."""
        result = await engine.score_response(
            prompt="What is 2+2?",
            response="4",
            expected_response="4",
            source_documents=[],
        )

        assert isinstance(result, TrustScoreResult)
        assert 0.0 <= result.overall_score <= 1.0

    def test_combine_scores_valid_weights(self, engine):
        """Test combine_scores with valid weights."""
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY, score=0.9, confidence=0.8,
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY, score=0.7, confidence=0.6,
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY, score=0.85, confidence=0.9,
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS, score=0.8, confidence=0.7,
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING, score=0.75, confidence=0.8,
            ),
        }
        weights = TrustScoreWeights()

        overall = engine.combine_scores(dimension_scores, weights)

        assert 0.0 <= overall <= 1.0
        # Manually compute expected
        expected = (
            0.9 * weights.accuracy
            + 0.7 * weights.consistency
            + 0.85 * weights.safety
            + 0.8 * weights.bias
            + 0.75 * weights.context_grounding
        )
        assert abs(overall - expected) < 0.01

    def test_combine_scores_invalid_weights(self, engine):
        """Test combine_scores raises ValueError for invalid weights."""
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY, score=0.9, confidence=0.8,
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY, score=0.7, confidence=0.6,
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY, score=0.85, confidence=0.9,
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS, score=0.8, confidence=0.7,
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING, score=0.75, confidence=0.8,
            ),
        }
        bad_weights = TrustScoreWeights(
            accuracy=0.5, consistency=0.5, safety=0.5, bias=0.5, context_grounding=0.5,
        )

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            engine.combine_scores(dimension_scores, bad_weights)

    def test_weights_must_sum_to_one(self):
        """Test that engine constructor rejects weights that don't sum to 1.0."""
        bad_config = TrustScoreConfig(
            weights=TrustScoreWeights(
                accuracy=0.5, consistency=0.5, safety=0.5, bias=0.5,
                context_grounding=0.5,
            )
        )
        with pytest.raises(ValueError, match="Trust score weights must sum to 1.0"):
            TrustScoringEngine(config=bad_config)

    async def test_score_batch(self, engine):
        """Test batch scoring of multiple items."""
        items = [
            {"prompt": "Q1", "response": "A1", "source_documents": ["doc1"]},
            {"prompt": "Q2", "response": "A2", "source_documents": ["doc2"]},
        ]

        results = await engine.score_batch(items)

        assert len(results) == 2
        for r in results:
            assert isinstance(r, TrustScoreResult)
            assert 0.0 <= r.overall_score <= 1.0

    async def test_consistency_unavailable_without_client(self, engine):
        """Test that consistency dimension gets neutral score without inference client."""
        result = await engine.score_response(
            prompt="Test",
            response="Test response.",
        )

        consistency = result.dimension_scores[TrustDimension.CONSISTENCY]
        assert consistency.score == 0.5
        assert consistency.confidence == 0.0
