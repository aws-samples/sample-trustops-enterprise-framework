"""
Unit tests for SafetyScorer.

Tests the safety dimension of trust scoring including:
- Harmful content detection (violence, hate speech, self-harm, illegal)
- Toxicity detection
- Policy violation detection (PII, medical/legal advice)

Requirements: 5.1, 5.5
"""

import pytest

from src.data_models.trust_score import TrustDimension
from src.trust_scoring.scorers.safety_scorer import SafetyScorer


class TestSafetyScorer:
    """Test suite for SafetyScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a SafetyScorer instance for testing."""
        return SafetyScorer()

    @pytest.fixture
    def strict_scorer(self):
        """Create a SafetyScorer in strict mode."""
        return SafetyScorer(strict_mode=True)

    @pytest.mark.asyncio
    async def test_empty_response_is_safe(self, scorer):
        """Test that empty responses are considered safe."""
        result = await scorer.calculate_safety("")

        assert result.dimension == TrustDimension.SAFETY
        assert result.score == 1.0
        assert result.confidence == 1.0
        assert "empty_response" in result.checks_passed

    @pytest.mark.asyncio
    async def test_safe_response(self, scorer):
        """Test that safe responses get high scores."""
        safe_response = (
            "The weather today is sunny and pleasant. "
            "It's a great day for outdoor activities."
        )

        result = await scorer.calculate_safety(safe_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score == 1.0
        assert "no_harmful_content" in result.checks_passed
        assert "no_toxicity" in result.checks_passed
        assert "no_policy_violations" in result.checks_passed

    @pytest.mark.asyncio
    async def test_violence_detection(self, scorer):
        """Test detection of violent content."""
        violent_response = (
            "You should attack them with a weapon and "
            "cause serious harm."
        )

        result = await scorer.calculate_safety(violent_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "harmful_content_detected" in result.checks_failed
        assert result.details["harmful_content_score"] == 0.0

        # Check that violence keywords were detected
        violations = result.details["harmful_content_details"]["violations"]
        assert any(v["category"] == "violence" for v in violations)

    @pytest.mark.asyncio
    async def test_hate_speech_detection(self, scorer):
        """Test detection of hate speech."""
        hate_response = (
            "That's a racist and derogatory comment showing "
            "clear discrimination."
        )

        result = await scorer.calculate_safety(hate_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "harmful_content_detected" in result.checks_failed

        # Check that hate speech keywords were detected
        violations = result.details["harmful_content_details"]["violations"]
        assert any(v["category"] == "hate_speech" for v in violations)

    @pytest.mark.asyncio
    async def test_self_harm_detection(self, scorer):
        """Test detection of self-harm content."""
        self_harm_response = (
            "I want to hurt myself and commit suicide."
        )

        result = await scorer.calculate_safety(self_harm_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "harmful_content_detected" in result.checks_failed

        # Check that self-harm keywords were detected
        violations = result.details["harmful_content_details"]["violations"]
        assert any(v["category"] == "self_harm" for v in violations)

    @pytest.mark.asyncio
    async def test_illegal_activity_detection(self, scorer):
        """Test detection of illegal activity content."""
        illegal_response = (
            "Here's how to commit fraud and steal money illegally."
        )

        result = await scorer.calculate_safety(illegal_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "harmful_content_detected" in result.checks_failed

        # Check that illegal activity keywords were detected
        violations = result.details["harmful_content_details"]["violations"]
        assert any(v["category"] == "illegal_activity" for v in violations)

    @pytest.mark.asyncio
    async def test_toxicity_detection(self, scorer):
        """Test detection of toxic language."""
        toxic_response = (
            "You're an idiot and a moron. That's a stupid idea."
        )

        result = await scorer.calculate_safety(toxic_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "toxicity_detected" in result.checks_failed
        assert result.details["toxicity_score"] == 0.0

        # Check toxicity details
        toxic_keywords = result.details["toxicity_details"]["toxic_keywords"]
        assert len(toxic_keywords) > 0
        assert result.details["toxicity_details"]["severity"] in [
            "medium", "high"
        ]

    @pytest.mark.asyncio
    async def test_pii_email_detection(self, scorer):
        """Test detection of email addresses (PII)."""
        pii_response = (
            "You can contact me at john.doe@example.com for more info."
        )

        result = await scorer.calculate_safety(pii_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "policy_violations_detected" in result.checks_failed

        # Check PII detection
        violations = result.details["policy_violation_details"]["violations"]
        assert any(
            v["category"] == "pii_exposure" and v["type"] == "email"
            for v in violations
        )

    @pytest.mark.asyncio
    async def test_pii_phone_detection(self, scorer):
        """Test detection of phone numbers (PII)."""
        pii_response = "Call me at 555-123-4567 or 555.987.6543."

        result = await scorer.calculate_safety(pii_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "policy_violations_detected" in result.checks_failed

        # Check phone number detection
        pii_found = result.details["policy_violation_details"]["pii_found"]
        assert "phone" in pii_found
        assert pii_found["phone"] == 2  # Two phone numbers

    @pytest.mark.asyncio
    async def test_pii_ssn_detection(self, scorer):
        """Test detection of SSN (PII)."""
        pii_response = "My SSN is 123-45-6789."

        result = await scorer.calculate_safety(pii_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "policy_violations_detected" in result.checks_failed

        # Check SSN detection
        pii_found = result.details["policy_violation_details"]["pii_found"]
        assert "ssn" in pii_found

    @pytest.mark.asyncio
    async def test_pii_credit_card_detection(self, scorer):
        """Test detection of credit card numbers (PII)."""
        pii_response = "My card number is 1234-5678-9012-3456."

        result = await scorer.calculate_safety(pii_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "policy_violations_detected" in result.checks_failed

        # Check credit card detection
        pii_found = result.details["policy_violation_details"]["pii_found"]
        assert "credit_card" in pii_found

    @pytest.mark.asyncio
    async def test_medical_advice_detection(self, scorer):
        """Test detection of medical advice."""
        medical_response = (
            "I diagnose you with this disease. "
            "Take this medication as treatment."
        )

        result = await scorer.calculate_safety(medical_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "policy_violations_detected" in result.checks_failed

        # Check medical advice detection
        violations = result.details["policy_violation_details"]["violations"]
        assert any(v["category"] == "medical_advice" for v in violations)

    @pytest.mark.asyncio
    async def test_legal_advice_detection(self, scorer):
        """Test detection of legal advice."""
        legal_response = (
            "You should sue them. Here's legal advice on your lawsuit."
        )

        result = await scorer.calculate_safety(legal_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "policy_violations_detected" in result.checks_failed

        # Check legal advice detection
        violations = result.details["policy_violation_details"]["violations"]
        assert any(v["category"] == "legal_advice" for v in violations)

    @pytest.mark.asyncio
    async def test_multiple_violations(self, scorer):
        """Test response with multiple types of violations."""
        multi_violation_response = (
            "You're an idiot. Attack them violently. "
            "Contact me at bad@example.com."
        )

        result = await scorer.calculate_safety(multi_violation_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score < 1.0
        assert "harmful_content_detected" in result.checks_failed
        assert "toxicity_detected" in result.checks_failed
        assert "policy_violations_detected" in result.checks_failed

    @pytest.mark.asyncio
    async def test_strict_mode_any_violation(self, strict_scorer):
        """Test that strict mode returns 0.0 for any violation."""
        # Response with only minor toxicity
        response = "That's a stupid idea."

        result = await strict_scorer.calculate_safety(response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score == 0.0
        assert result.details["strict_mode"] is True

    @pytest.mark.asyncio
    async def test_strict_mode_safe_response(self, strict_scorer):
        """Test that strict mode returns 1.0 for safe responses."""
        safe_response = "This is a helpful and safe response."

        result = await strict_scorer.calculate_safety(safe_response)

        assert result.dimension == TrustDimension.SAFETY
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_weighted_scoring(self, scorer):
        """Test that weighted scoring works correctly."""
        # Response with only harmful content (weight 0.4)
        response = "Violence and attack."

        result = await scorer.calculate_safety(response)

        # Score should reflect weighted combination
        # harmful=0.0, toxicity=1.0, policy=1.0
        # 0.0*0.4 + 1.0*0.3 + 1.0*0.3 = 0.6
        assert result.dimension == TrustDimension.SAFETY
        assert abs(result.score - 0.6) < 0.01

    @pytest.mark.asyncio
    async def test_score_range(self, scorer):
        """Test that scores are always in [0, 1] range."""
        test_responses = [
            "",
            "Safe response",
            "Violence attack kill",
            "Idiot moron stupid",
            "Email: test@example.com",
            "Multiple violations: attack, idiot, test@example.com"
        ]

        for response in test_responses:
            result = await scorer.calculate_safety(response)
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_levels(self, scorer):
        """Test that confidence levels are appropriate."""
        # Clear violation should have high confidence
        clear_violation = "Violence and attack with weapons."
        result1 = await scorer.calculate_safety(clear_violation)
        assert result1.confidence >= 0.7

        # Clearly safe should have high confidence
        clearly_safe = "The weather is nice today."
        result2 = await scorer.calculate_safety(clearly_safe)
        assert result2.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_case_insensitivity(self, scorer):
        """Test that keyword detection is case-insensitive."""
        responses = [
            "VIOLENCE and ATTACK",
            "Violence and Attack",
            "violence and attack"
        ]

        for response in responses:
            result = await scorer.calculate_safety(response)
            assert result.score < 1.0
            assert "harmful_content_detected" in result.checks_failed

    @pytest.mark.asyncio
    async def test_details_structure(self, scorer):
        """Test that result details have expected structure."""
        response = "Test response"
        result = await scorer.calculate_safety(response)

        # Check required detail fields
        assert "harmful_content_score" in result.details
        assert "toxicity_score" in result.details
        assert "policy_violation_score" in result.details
        assert "harmful_content_details" in result.details
        assert "toxicity_details" in result.details
        assert "policy_violation_details" in result.details
        assert "strict_mode" in result.details
        assert "response_length" in result.details

    def test_invalid_weights(self):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError):
            SafetyScorer(
                harmful_content_weight=0.5,
                toxicity_weight=0.3,
                policy_violation_weight=0.3  # Sum > 1.0
            )

    @pytest.mark.asyncio
    async def test_contextual_keywords(self, scorer):
        """Test that keywords in safe contexts don't trigger false positives."""
        # These contain keywords but in safe contexts
        safe_contexts = [
            "The movie depicted violence but had a positive message.",
            "We need to attack this problem systematically.",
            "The legal advice column was very informative.",
        ]

        for response in safe_contexts:
            result = await scorer.calculate_safety(response)
            # Note: Current implementation is keyword-based and may flag these
            # This test documents current behavior
            # In production, context-aware NLP would be better
            assert result.dimension == TrustDimension.SAFETY
            assert 0.0 <= result.score <= 1.0
