"""
Unit tests for Trust Scoring Engine V2.

Tests the multi-dimensional trust scoring system with weighted score
combination, including:
- Score combination logic with configurable weights
- Weight validation (must sum to 1.0)
- Overall score calculation in [0, 1] range
- Batch scoring functionality
- Explanation generation

Requirements: 5.1, 5.2, 5.3, 5.4
"""

import pytest

from src.data_models.trust_score import (
    DimensionScore,
    TrustDimension,
    TrustScoreConfig,
    TrustScoreWeights,
)
from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine


class TestWeightedScoreCombiner:
    """Test weighted score combination logic."""

    def test_combine_scores_with_default_weights(self):
        """Test score combination with default weights."""
        # Arrange
        engine = TrustScoringEngine()
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.8,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.7,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=1.0,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=0.9,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.6,
                confidence=0.7,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
        }
        weights = TrustScoreWeights()  # Default weights

        # Act
        overall_score = engine.combine_scores(dimension_scores, weights)

        # Assert
        # Expected: 0.8*0.25 + 0.7*0.20 + 1.0*0.20 + 0.9*0.15 + 0.6*0.20
        #         = 0.20 + 0.14 + 0.20 + 0.135 + 0.12 = 0.795
        assert 0.79 <= overall_score <= 0.80
        assert 0.0 <= overall_score <= 1.0

    def test_combine_scores_with_custom_weights(self):
        """Test score combination with custom weights."""
        # Arrange
        engine = TrustScoringEngine()
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=1.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=0.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=0.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
        }
        # Custom weights: accuracy=1.0, all others=0.0
        weights = TrustScoreWeights(
            accuracy=1.0,
            consistency=0.0,
            safety=0.0,
            bias=0.0,
            context_grounding=0.0,
        )

        # Act
        overall_score = engine.combine_scores(dimension_scores, weights)

        # Assert
        # Expected: 1.0*1.0 + 0.0*0.0 + 0.0*0.0 + 0.0*0.0 + 0.0*0.0 = 1.0
        assert overall_score == 1.0

    def test_combine_scores_all_zeros(self):
        """Test score combination when all dimension scores are 0."""
        # Arrange
        engine = TrustScoringEngine()
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=0.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=0.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
        }
        weights = TrustScoreWeights()

        # Act
        overall_score = engine.combine_scores(dimension_scores, weights)

        # Assert
        assert overall_score == 0.0

    def test_combine_scores_all_ones(self):
        """Test score combination when all dimension scores are 1."""
        # Arrange
        engine = TrustScoringEngine()
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=1.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=1.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=1.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=1.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=1.0,
                confidence=1.0,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
        }
        weights = TrustScoreWeights()

        # Act
        overall_score = engine.combine_scores(dimension_scores, weights)

        # Assert
        assert overall_score == 1.0

    def test_combine_scores_validates_weight_sum(self):
        """Test that combine_scores validates weights sum to 1.0."""
        # Arrange
        engine = TrustScoringEngine()
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.8,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.7,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=1.0,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=0.9,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.6,
                confidence=0.7,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
        }
        # Invalid weights that don't sum to 1.0
        weights = TrustScoreWeights(
            accuracy=0.5,
            consistency=0.5,
            safety=0.5,
            bias=0.5,
            context_grounding=0.5,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            engine.combine_scores(dimension_scores, weights)

    def test_score_range_invariant(self):
        """Test that overall score is always in [0, 1] range."""
        # Arrange
        engine = TrustScoringEngine()
        weights = TrustScoreWeights()

        # Test various score combinations
        test_cases = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [1.0, 0.0, 1.0, 0.0, 1.0],
        ]

        for scores in test_cases:
            dimension_scores = {
                TrustDimension.ACCURACY: DimensionScore(
                    dimension=TrustDimension.ACCURACY,
                    score=scores[0],
                    confidence=1.0,
                    details={},
                    checks_passed=[],
                    checks_failed=[],
                ),
                TrustDimension.CONSISTENCY: DimensionScore(
                    dimension=TrustDimension.CONSISTENCY,
                    score=scores[1],
                    confidence=1.0,
                    details={},
                    checks_passed=[],
                    checks_failed=[],
                ),
                TrustDimension.SAFETY: DimensionScore(
                    dimension=TrustDimension.SAFETY,
                    score=scores[2],
                    confidence=1.0,
                    details={},
                    checks_passed=[],
                    checks_failed=[],
                ),
                TrustDimension.BIAS: DimensionScore(
                    dimension=TrustDimension.BIAS,
                    score=scores[3],
                    confidence=1.0,
                    details={},
                    checks_passed=[],
                    checks_failed=[],
                ),
                TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                    dimension=TrustDimension.CONTEXT_GROUNDING,
                    score=scores[4],
                    confidence=1.0,
                    details={},
                    checks_passed=[],
                    checks_failed=[],
                ),
            }

            # Act
            overall_score = engine.combine_scores(dimension_scores, weights)

            # Assert
            assert 0.0 <= overall_score <= 1.0, (
                f"Score {overall_score} out of range for inputs {scores}"
            )


class TestTrustScoringEngineInitialization:
    """Test trust scoring engine initialization."""

    def test_initialization_with_defaults(self):
        """Test engine initializes with default configuration."""
        # Act
        engine = TrustScoringEngine()

        # Assert
        assert engine.config is not None
        assert engine.config.weights.accuracy == 0.25
        assert engine.config.weights.consistency == 0.20
        assert engine.config.weights.safety == 0.20
        assert engine.config.weights.bias == 0.15
        assert engine.config.weights.context_grounding == 0.20
        assert engine.config.review_threshold == 0.6

    def test_initialization_with_custom_config(self):
        """Test engine initializes with custom configuration."""
        # Arrange
        custom_config = TrustScoreConfig(
            weights=TrustScoreWeights(
                accuracy=0.3,
                consistency=0.2,
                safety=0.2,
                bias=0.1,
                context_grounding=0.2,
            ),
            review_threshold=0.7,
            consistency_samples=5,
        )

        # Act
        engine = TrustScoringEngine(config=custom_config)

        # Assert
        assert engine.config.weights.accuracy == 0.3
        assert engine.config.review_threshold == 0.7
        assert engine.config.consistency_samples == 5

    def test_initialization_validates_weights(self):
        """Test engine validates weights sum to 1.0 on initialization."""
        # Arrange
        invalid_config = TrustScoreConfig(
            weights=TrustScoreWeights(
                accuracy=0.5,
                consistency=0.5,
                safety=0.5,
                bias=0.5,
                context_grounding=0.5,
            )
        )

        # Act & Assert
        with pytest.raises(ValueError, match="must sum to 1.0"):
            TrustScoringEngine(config=invalid_config)


class TestExplanationGeneration:
    """Test explanation generation."""

    def test_generate_explanation_all_good(self):
        """Test explanation when all dimensions score well."""
        # Arrange
        engine = TrustScoringEngine()
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.9,
                confidence=0.9,
                details={},
                checks_passed=["exact_match"],
                checks_failed=[],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.8,
                confidence=0.8,
                details={},
                checks_passed=["high_consistency"],
                checks_failed=[],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=1.0,
                confidence=0.9,
                details={},
                checks_passed=["no_harmful_content"],
                checks_failed=[],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=0.9,
                confidence=0.8,
                details={},
                checks_passed=["no_bias"],
                checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.8,
                confidence=0.7,
                details={},
                checks_passed=["well_grounded"],
                checks_failed=[],
            ),
        }

        # Act
        explanation = engine._generate_explanation(
            dimension_scores=dimension_scores,
            overall_score=0.88,
            flagged_for_review=False,
        )

        # Assert
        assert "0.88" in explanation
        assert "Overall trust score" in explanation

    def test_generate_explanation_with_weak_dimensions(self):
        """Test explanation identifies weak dimensions."""
        # Arrange
        engine = TrustScoringEngine()
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.3,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=["exact_match", "fuzzy_match"],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.4,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=["low_consistency"],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=1.0,
                confidence=0.9,
                details={},
                checks_passed=["no_harmful_content"],
                checks_failed=[],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=0.9,
                confidence=0.8,
                details={},
                checks_passed=["no_bias"],
                checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.5,
                confidence=0.7,
                details={},
                checks_passed=[],
                checks_failed=["low_grounding"],
            ),
        }

        # Act
        explanation = engine._generate_explanation(
            dimension_scores=dimension_scores,
            overall_score=0.52,
            flagged_for_review=True,
        )

        # Assert
        assert "Weak dimensions" in explanation
        assert "accuracy" in explanation.lower()
        assert "consistency" in explanation.lower()
        assert "context_grounding" in explanation.lower()
        assert "FLAGGED FOR REVIEW" in explanation

    def test_generate_explanation_includes_failed_checks(self):
        """Test explanation includes failed checks."""
        # Arrange
        engine = TrustScoringEngine()
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.5,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=["exact_match"],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.7,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=0.0,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=["harmful_content_detected", "toxicity_detected"],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=0.9,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.8,
                confidence=0.7,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
        }

        # Act
        explanation = engine._generate_explanation(
            dimension_scores=dimension_scores,
            overall_score=0.58,
            flagged_for_review=True,
        )

        # Assert
        assert "Issues" in explanation
        assert "exact_match" in explanation
        assert "harmful_content_detected" in explanation
        assert "toxicity_detected" in explanation


class TestReviewFlagging:
    """Test review flagging logic."""

    def test_flag_for_review_below_threshold(self):
        """Test response is flagged when score below threshold."""
        # Arrange
        config = TrustScoreConfig(review_threshold=0.7)
        engine = TrustScoringEngine(config=config)

        # Create dimension scores that result in overall score < 0.7
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.5,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.5,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=0.5,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=0.5,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.5,
                confidence=0.7,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
        }

        overall_score = engine.combine_scores(dimension_scores, config.weights)

        # Act
        flagged = overall_score < config.review_threshold

        # Assert
        assert overall_score < 0.7
        assert flagged is True

    def test_no_flag_above_threshold(self):
        """Test response is not flagged when score above threshold."""
        # Arrange
        config = TrustScoreConfig(review_threshold=0.6)
        engine = TrustScoringEngine(config=config)

        # Create dimension scores that result in overall score > 0.6
        dimension_scores = {
            TrustDimension.ACCURACY: DimensionScore(
                dimension=TrustDimension.ACCURACY,
                score=0.9,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONSISTENCY: DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.8,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.SAFETY: DimensionScore(
                dimension=TrustDimension.SAFETY,
                score=1.0,
                confidence=0.9,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.BIAS: DimensionScore(
                dimension=TrustDimension.BIAS,
                score=0.9,
                confidence=0.8,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
            TrustDimension.CONTEXT_GROUNDING: DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.8,
                confidence=0.7,
                details={},
                checks_passed=[],
                checks_failed=[],
            ),
        }

        overall_score = engine.combine_scores(dimension_scores, config.weights)

        # Act
        flagged = overall_score < config.review_threshold

        # Assert
        assert overall_score > 0.6
        assert flagged is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

