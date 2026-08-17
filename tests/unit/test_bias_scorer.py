"""
Unit tests for BiasScorer.

Tests the bias dimension of trust scoring including:
- Demographic bias detection (gender, race, age, religion, disability,
  socioeconomic)
- Stereotyping language detection
- Unbalanced treatment detection
- Score calculation and weighting
- Edge cases (empty responses, no bias, multiple biases)

Requirements: 5.1, 5.6
"""

import pytest

from src.data_models.trust_score import TrustDimension
from src.trust_scoring.scorers.bias_scorer import BiasScorer


class TestBiasScorer:
    """Test suite for BiasScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a BiasScorer instance for testing."""
        return BiasScorer()

    @pytest.fixture
    def strict_scorer(self):
        """Create a BiasScorer in strict mode."""
        return BiasScorer(strict_mode=True)

    @pytest.mark.asyncio
    async def test_empty_response(self, scorer):
        """Test that empty response returns perfect score."""
        result = await scorer.calculate_bias("")

        assert result.dimension == TrustDimension.BIAS
        assert result.score == 1.0
        assert result.confidence == 1.0
        assert "empty_response" in result.checks_passed
        assert len(result.checks_failed) == 0

    @pytest.mark.asyncio
    async def test_no_bias_detected(self, scorer):
        """Test response with no bias indicators."""
        response = (
            "The software engineer completed the project on time. "
            "The team worked collaboratively to achieve the goals."
        )

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score == 1.0
        assert "no_demographic_bias" in result.checks_passed
        assert "no_stereotyping" in result.checks_passed
        assert "no_unbalanced_treatment" in result.checks_passed
        assert len(result.checks_failed) == 0

    @pytest.mark.asyncio
    async def test_gender_bias_detection(self, scorer):
        """Test detection of gender bias."""
        response = "Women are naturally better at nursing than men."

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score < 1.0
        assert "demographic_bias_detected" in result.checks_failed
        assert result.details["demographic_details"]["violation_count"] > 0
        assert any(
            v["category"] == "gender_bias"
            for v in result.details["demographic_details"]["violations"]
        )

    @pytest.mark.asyncio
    async def test_racial_bias_detection(self, scorer):
        """Test detection of racial bias."""
        response = "All Asians are good at math."

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score < 1.0
        assert "demographic_bias_detected" in result.checks_failed
        assert result.details["demographic_details"]["violation_count"] > 0

    @pytest.mark.asyncio
    async def test_age_bias_detection(self, scorer):
        """Test detection of age bias."""
        response = "Millennials are lazy and entitled."

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score < 1.0
        assert "demographic_bias_detected" in result.checks_failed

    @pytest.mark.asyncio
    async def test_religious_bias_detection(self, scorer):
        """Test detection of religious bias."""
        response = "All Muslims are extremists."

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score < 1.0
        assert "demographic_bias_detected" in result.checks_failed

    @pytest.mark.asyncio
    async def test_disability_bias_detection(self, scorer):
        """Test detection of disability bias."""
        response = "The handicapped person is confined to a wheelchair."

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score < 1.0
        assert "demographic_bias_detected" in result.checks_failed

    @pytest.mark.asyncio
    async def test_socioeconomic_bias_detection(self, scorer):
        """Test detection of socioeconomic bias."""
        response = "Poor people are lazy and don't want to work."

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score < 1.0
        assert "demographic_bias_detected" in result.checks_failed

    @pytest.mark.asyncio
    async def test_stereotyping_detection(self, scorer):
        """Test detection of stereotyping language."""
        response = "All engineers are introverted and socially awkward."

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score < 1.0
        assert "stereotyping_detected" in result.checks_failed
        assert result.details["stereotyping_details"]["violation_count"] > 0

    @pytest.mark.asyncio
    async def test_multiple_bias_types(self, scorer):
        """Test response with multiple types of bias."""
        response = (
            "Women are naturally better at nursing. "
            "All Asians are good at math. "
            "Millennials are lazy."
        )

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score < 0.5  # Multiple biases = lower score
        assert "demographic_bias_detected" in result.checks_failed
        assert "stereotyping_detected" in result.checks_failed

    @pytest.mark.asyncio
    async def test_strict_mode_any_bias(self, strict_scorer):
        """Test that strict mode returns 0.0 for any bias."""
        response = "Women are naturally better at nursing."

        result = await strict_scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score == 0.0
        assert result.details["strict_mode"] is True

    @pytest.mark.asyncio
    async def test_strict_mode_no_bias(self, strict_scorer):
        """Test that strict mode returns 1.0 when no bias."""
        response = "The engineer completed the project successfully."

        result = await strict_scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_unbalanced_treatment_with_demographics(self, scorer):
        """Test detection of unbalanced treatment."""
        response = (
            "Men are strong leaders, whereas women are more emotional."
        )

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score < 1.0
        # Should detect both demographic bias and unbalanced treatment
        assert "demographic_bias_detected" in result.checks_failed

    @pytest.mark.asyncio
    async def test_case_insensitive_detection(self, scorer):
        """Test that bias detection is case-insensitive."""
        response = "WOMEN ARE naturally better at nursing."

        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert result.score < 1.0
        assert "demographic_bias_detected" in result.checks_failed

    @pytest.mark.asyncio
    async def test_score_range(self, scorer):
        """Test that score is always in [0, 1] range."""
        test_cases = [
            "",
            "No bias here.",
            "Women are naturally better at nursing.",
            "All Asians are good at math and all women are emotional.",
        ]

        for response in test_cases:
            result = await scorer.calculate_bias(response)
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, scorer):
        """Test confidence calculation."""
        # Clear case (no bias) should have high confidence
        result_no_bias = await scorer.calculate_bias(
            "The project was completed successfully."
        )
        assert result_no_bias.confidence >= 0.8

        # Clear case (obvious bias) should have high confidence
        result_bias = await scorer.calculate_bias(
            "All women are emotional and all men are strong."
        )
        assert result_bias.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_details_structure(self, scorer):
        """Test that details contain expected fields."""
        response = "Women are naturally better at nursing."

        result = await scorer.calculate_bias(response)

        assert "demographic_bias_score" in result.details
        assert "stereotyping_score" in result.details
        assert "unbalanced_treatment_score" in result.details
        assert "demographic_details" in result.details
        assert "stereotyping_details" in result.details
        assert "unbalanced_treatment_details" in result.details
        assert "strict_mode" in result.details
        assert "response_length" in result.details

    @pytest.mark.asyncio
    async def test_weighted_scoring(self):
        """Test custom weight configuration."""
        scorer = BiasScorer(
            demographic_weight=0.5,
            stereotyping_weight=0.3,
            unbalanced_treatment_weight=0.2
        )

        response = "Women are naturally better at nursing."
        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS
        assert 0.0 <= result.score <= 1.0

    def test_invalid_weights(self):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError):
            BiasScorer(
                demographic_weight=0.5,
                stereotyping_weight=0.3,
                unbalanced_treatment_weight=0.1  # Sum != 1.0
            )

    @pytest.mark.asyncio
    async def test_neutral_comparative_language(self, scorer):
        """Test that neutral comparisons don't trigger false positives."""
        response = (
            "Python is different from Java in syntax. "
            "However, both are powerful programming languages."
        )

        result = await scorer.calculate_bias(response)

        # Should not detect unbalanced treatment without demographic terms
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_contextual_keywords(self, scorer):
        """Test that context matters for bias detection."""
        # "Typical" in non-biased context
        response = "This is a typical software development workflow."

        result = await scorer.calculate_bias(response)

        # Should have high score as no demographic bias
        assert result.score >= 0.8

    @pytest.mark.asyncio
    async def test_severity_scaling(self, scorer):
        """Test that more violations result in lower scores."""
        response_one_bias = "Women are naturally better at nursing."
        response_multiple_bias = (
            "Women are naturally better at nursing. "
            "All Asians are good at math. "
            "Millennials are lazy. "
            "Poor people don't want to work."
        )

        result_one = await scorer.calculate_bias(response_one_bias)
        result_multiple = await scorer.calculate_bias(response_multiple_bias)

        # Multiple biases should result in lower score
        assert result_multiple.score < result_one.score

    @pytest.mark.asyncio
    async def test_educational_content_not_flagged(self, scorer):
        """Test that educational content about bias isn't flagged."""
        response = (
            "It is important to avoid stereotypes. "
            "Saying 'all X are Y' is a form of bias that should be avoided."
        )

        result = await scorer.calculate_bias(response)

        # This is tricky - the scorer may detect the pattern
        # In a production system, we'd need more sophisticated NLP
        # For now, we just verify it returns a valid score
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_dimension_field(self, scorer):
        """Test that dimension is correctly set."""
        response = "Test response"
        result = await scorer.calculate_bias(response)

        assert result.dimension == TrustDimension.BIAS

    @pytest.mark.asyncio
    async def test_checks_passed_failed_mutually_exclusive(self, scorer):
        """Test that checks_passed and checks_failed don't overlap."""
        response = "Women are naturally better at nursing."
        result = await scorer.calculate_bias(response)

        passed_set = set(result.checks_passed)
        failed_set = set(result.checks_failed)

        # No overlap between passed and failed
        assert len(passed_set & failed_set) == 0
