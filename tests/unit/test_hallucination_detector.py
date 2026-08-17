"""
Unit tests for HallucinationDetector.

Tests the hallucination detection pipeline including claim extraction,
classification, and rate calculation.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.8
"""

import pytest

from src.data_models.hallucination import (
    Claim,
    ClaimEvidence,
    EvidenceSource,
    HallucinationConfig,
    HallucinationResult,
    SensitivityLevel
)
from src.hallucination.hallucination_detector import HallucinationDetector


class TestHallucinationRateCalculator:
    """
    Test suite for hallucination rate calculation.

    Requirement 6.4: Calculate hallucination rate as ratio of unsupported
    to total factual claims
    """

    def test_calculate_rate_all_supported(self):
        """Test rate calculation when all factual claims are supported."""
        detector = HallucinationDetector()

        # Create claim evidence with all claims supported
        claim_evidence = [
            ClaimEvidence(
                claim=Claim(
                    text="The sky is blue",
                    start_idx=0,
                    end_idx=15,
                    is_factual=True,
                    is_opinion=False
                ),
                is_supported=True,
                similarity_score=0.9,
                evidence_sources=[]
            ),
            ClaimEvidence(
                claim=Claim(
                    text="Water is wet",
                    start_idx=16,
                    end_idx=28,
                    is_factual=True,
                    is_opinion=False
                ),
                is_supported=True,
                similarity_score=0.85,
                evidence_sources=[]
            )
        ]

        rate = detector.calculate_hallucination_rate(claim_evidence)

        # All claims supported, so rate should be 0.0
        assert rate == 0.0

    def test_calculate_rate_all_unsupported(self):
        """Test rate calculation when all factual claims are unsupported."""
        detector = HallucinationDetector()

        # Create claim evidence with all claims unsupported
        claim_evidence = [
            ClaimEvidence(
                claim=Claim(
                    text="The moon is made of cheese",
                    start_idx=0,
                    end_idx=29,
                    is_factual=True,
                    is_opinion=False
                ),
                is_supported=False,
                similarity_score=0.2,
                evidence_sources=[]
            ),
            ClaimEvidence(
                claim=Claim(
                    text="Unicorns exist",
                    start_idx=30,
                    end_idx=44,
                    is_factual=True,
                    is_opinion=False
                ),
                is_supported=False,
                similarity_score=0.1,
                evidence_sources=[]
            )
        ]

        rate = detector.calculate_hallucination_rate(claim_evidence)

        # All claims unsupported, so rate should be 1.0
        assert rate == 1.0

    def test_calculate_rate_mixed_support(self):
        """Test rate calculation with mix of supported and unsupported."""
        detector = HallucinationDetector()

        # Create claim evidence with 2 supported, 1 unsupported
        claim_evidence = [
            ClaimEvidence(
                claim=Claim(
                    text="The Earth orbits the Sun",
                    start_idx=0,
                    end_idx=24,
                    is_factual=True,
                    is_opinion=False
                ),
                is_supported=True,
                similarity_score=0.95,
                evidence_sources=[]
            ),
            ClaimEvidence(
                claim=Claim(
                    text="Mars is red",
                    start_idx=25,
                    end_idx=36,
                    is_factual=True,
                    is_opinion=False
                ),
                is_supported=True,
                similarity_score=0.88,
                evidence_sources=[]
            ),
            ClaimEvidence(
                claim=Claim(
                    text="Jupiter has 200 moons",
                    start_idx=37,
                    end_idx=58,
                    is_factual=True,
                    is_opinion=False
                ),
                is_supported=False,
                similarity_score=0.3,
                evidence_sources=[]
            )
        ]

        rate = detector.calculate_hallucination_rate(claim_evidence)

        # 1 unsupported out of 3 factual claims = 1/3 ≈ 0.333
        assert abs(rate - 0.333333) < 0.0001

    def test_calculate_rate_no_factual_claims(self):
        """Test rate calculation when there are no factual claims."""
        detector = HallucinationDetector()

        # Create claim evidence with only opinions
        claim_evidence = [
            ClaimEvidence(
                claim=Claim(
                    text="I think this is good",
                    start_idx=0,
                    end_idx=20,
                    is_factual=False,
                    is_opinion=True
                ),
                is_supported=False,
                similarity_score=0.2,
                evidence_sources=[]
            ),
            ClaimEvidence(
                claim=Claim(
                    text="It seems nice",
                    start_idx=21,
                    end_idx=34,
                    is_factual=False,
                    is_opinion=True
                ),
                is_supported=False,
                similarity_score=0.15,
                evidence_sources=[]
            )
        ]

        rate = detector.calculate_hallucination_rate(claim_evidence)

        # No factual claims, so rate should be 0.0
        assert rate == 0.0

    def test_calculate_rate_empty_list(self):
        """Test rate calculation with empty claim evidence list."""
        detector = HallucinationDetector()

        claim_evidence = []

        rate = detector.calculate_hallucination_rate(claim_evidence)

        # Empty list means no claims, so rate should be 0.0
        assert rate == 0.0

    def test_calculate_rate_mixed_factual_and_opinions(self):
        """Test that only factual claims are counted in rate calculation."""
        detector = HallucinationDetector()

        # Mix of factual and opinion claims
        claim_evidence = [
            ClaimEvidence(
                claim=Claim(
                    text="Python was created in 1991",
                    start_idx=0,
                    end_idx=26,
                    is_factual=True,
                    is_opinion=False
                ),
                is_supported=True,
                similarity_score=0.92,
                evidence_sources=[]
            ),
            ClaimEvidence(
                claim=Claim(
                    text="I think Python is great",
                    start_idx=27,
                    end_idx=50,
                    is_factual=False,
                    is_opinion=True
                ),
                is_supported=False,
                similarity_score=0.1,
                evidence_sources=[]
            ),
            ClaimEvidence(
                claim=Claim(
                    text="Python has 500 libraries",
                    start_idx=51,
                    end_idx=75,
                    is_factual=True,
                    is_opinion=False
                ),
                is_supported=False,
                similarity_score=0.4,
                evidence_sources=[]
            )
        ]

        rate = detector.calculate_hallucination_rate(claim_evidence)

        # Only 2 factual claims, 1 unsupported = 1/2 = 0.5
        # The opinion claim should not be counted
        assert rate == 0.5

    def test_calculate_rate_range_invariant(self):
        """Test that hallucination rate is always in [0, 1] range."""
        detector = HallucinationDetector()

        # Test various scenarios
        test_cases = [
            [],  # Empty
            [  # All supported
                ClaimEvidence(
                    claim=Claim(
                        text="Test",
                        start_idx=0,
                        end_idx=4,
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=True,
                    similarity_score=0.9,
                    evidence_sources=[]
                )
            ],
            [  # All unsupported
                ClaimEvidence(
                    claim=Claim(
                        text="Test",
                        start_idx=0,
                        end_idx=4,
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=False,
                    similarity_score=0.2,
                    evidence_sources=[]
                )
            ]
        ]

        for claim_evidence in test_cases:
            rate = detector.calculate_hallucination_rate(claim_evidence)
            assert 0.0 <= rate <= 1.0, (
                f"Rate {rate} is outside valid range [0, 1]"
            )


class TestHallucinationDetector:
    """Test suite for HallucinationDetector class."""

    def test_initialization_default_config(self):
        """Test detector initialization with default config."""
        detector = HallucinationDetector()

        assert detector.config is not None
        assert isinstance(detector.config, HallucinationConfig)
        assert detector.claim_extractor is not None
        assert detector.claim_classifier is not None

    def test_initialization_custom_config(self):
        """Test detector initialization with custom config."""
        config = HallucinationConfig(
            similarity_threshold=0.8,
            sensitivity_level=SensitivityLevel.STRICT,
            max_claims_per_response=100
        )

        detector = HallucinationDetector(config)

        assert detector.config == config
        assert detector.config.similarity_threshold == 0.8
        assert detector.config.sensitivity_level == SensitivityLevel.STRICT

    def test_detect_basic_response(self):
        """Test basic hallucination detection on a simple response."""
        detector = HallucinationDetector()

        response = "The sky is blue. Water is wet."

        result = detector.detect(response)

        # Verify result structure
        assert result is not None
        assert result.total_claims >= 0
        assert result.factual_claims >= 0
        assert result.supported_claims >= 0
        assert result.unsupported_claims >= 0
        assert 0.0 <= result.hallucination_rate <= 1.0
        assert 0.0 <= result.overall_grounding_score <= 1.0

    def test_detect_empty_response(self):
        """Test detection on empty response."""
        detector = HallucinationDetector()

        response = ""

        result = detector.detect(response)

        # Empty response should have no claims
        assert result.total_claims == 0
        assert result.factual_claims == 0
        assert result.hallucination_rate == 0.0

    def test_detect_opinion_only_response(self):
        """Test detection on response with only opinions."""
        detector = HallucinationDetector()

        response = "I think this is good. It seems nice to me."

        result = detector.detect(response)

        # Should extract claims but they should be opinions
        assert result.total_claims >= 0
        # Hallucination rate should be 0 if no factual claims
        if result.factual_claims == 0:
            assert result.hallucination_rate == 0.0

    class TestTextSpanHighlighter:
        """
        Test suite for text span highlighting.

        Requirement 6.5: Identify flagged text spans with low grounding scores
        """

        def test_generate_flagged_spans_unsupported_claims(self):
            """Test that unsupported factual claims produce flagged spans."""
            detector = HallucinationDetector()
            response = "The sky is blue. The moon is made of cheese."

            claim_evidence = [
                ClaimEvidence(
                    claim=Claim(
                        text="The sky is blue.",
                        start_idx=0,
                        end_idx=16,
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=True,
                    similarity_score=0.9,
                    evidence_sources=[]
                ),
                ClaimEvidence(
                    claim=Claim(
                        text="The moon is made of cheese.",
                        start_idx=17,
                        end_idx=44,
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=False,
                    similarity_score=0.25,
                    evidence_sources=[
                        EvidenceSource(
                            document_id="doc1",
                            span="The moon is rocky",
                            similarity_score=0.25
                        )
                    ]
                )
            ]

            spans = detector.generate_flagged_spans(claim_evidence, response)

            assert len(spans) == 1
            assert spans[0].text == "The moon is made of cheese."
            assert spans[0].start_idx == 17
            assert spans[0].end_idx == 44
            assert spans[0].grounding_score == 0.25
            assert "0.25" in spans[0].reason

        def test_generate_flagged_spans_no_evidence(self):
            """Test reason text when no evidence sources exist."""
            detector = HallucinationDetector()
            response = "Unicorns exist in the wild."

            claim_evidence = [
                ClaimEvidence(
                    claim=Claim(
                        text="Unicorns exist in the wild.",
                        start_idx=0,
                        end_idx=27,
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=False,
                    similarity_score=0.0,
                    evidence_sources=[]
                )
            ]

            spans = detector.generate_flagged_spans(claim_evidence, response)

            assert len(spans) == 1
            assert spans[0].reason == "No supporting evidence found"
            assert spans[0].grounding_score == 0.0

        def test_generate_flagged_spans_opinions_excluded(self):
            """Test that opinion claims are not flagged even if unsupported."""
            detector = HallucinationDetector()
            response = "I think this is great. The data is wrong."

            claim_evidence = [
                ClaimEvidence(
                    claim=Claim(
                        text="I think this is great.",
                        start_idx=0,
                        end_idx=22,
                        is_factual=False,
                        is_opinion=True
                    ),
                    is_supported=False,
                    similarity_score=0.1,
                    evidence_sources=[]
                ),
                ClaimEvidence(
                    claim=Claim(
                        text="The data is wrong.",
                        start_idx=23,
                        end_idx=41,
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=False,
                    similarity_score=0.3,
                    evidence_sources=[
                        EvidenceSource(
                            document_id="doc1",
                            span="The data is correct",
                            similarity_score=0.3
                        )
                    ]
                )
            ]

            spans = detector.generate_flagged_spans(claim_evidence, response)

            # Only the factual unsupported claim should be flagged
            assert len(spans) == 1
            assert spans[0].text == "The data is wrong."

        def test_generate_flagged_spans_all_supported(self):
            """Test that no spans are flagged when all claims are supported."""
            detector = HallucinationDetector()
            response = "The sky is blue. Water is wet."

            claim_evidence = [
                ClaimEvidence(
                    claim=Claim(
                        text="The sky is blue.",
                        start_idx=0,
                        end_idx=16,
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=True,
                    similarity_score=0.9,
                    evidence_sources=[]
                ),
                ClaimEvidence(
                    claim=Claim(
                        text="Water is wet.",
                        start_idx=17,
                        end_idx=30,
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=True,
                    similarity_score=0.85,
                    evidence_sources=[]
                )
            ]

            spans = detector.generate_flagged_spans(claim_evidence, response)

            assert len(spans) == 0

        def test_generate_flagged_spans_empty_evidence(self):
            """Test with empty claim evidence list."""
            detector = HallucinationDetector()

            spans = detector.generate_flagged_spans([], "Some response text.")

            assert len(spans) == 0

        def test_generate_flagged_spans_grounding_score_range(self):
            """Test that grounding scores are preserved in [0, 1] range."""
            detector = HallucinationDetector()
            response = "Claim one. Claim two. Claim three."

            claim_evidence = [
                ClaimEvidence(
                    claim=Claim(
                        text="Claim one.",
                        start_idx=0,
                        end_idx=10,
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=False,
                    similarity_score=0.0,
                    evidence_sources=[]
                ),
                ClaimEvidence(
                    claim=Claim(
                        text="Claim two.",
                        start_idx=11,
                        end_idx=21,
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=False,
                    similarity_score=0.45,
                    evidence_sources=[
                        EvidenceSource(
                            document_id="doc1",
                            span="some text",
                            similarity_score=0.45
                        )
                    ]
                )
            ]

            spans = detector.generate_flagged_spans(claim_evidence, response)

            for span in spans:
                assert 0.0 <= span.grounding_score <= 1.0

        def test_generate_flagged_spans_invalid_indices_skipped(self):
            """Test that claims with out-of-bounds indices are skipped."""
            detector = HallucinationDetector()
            response = "Short text."

            claim_evidence = [
                ClaimEvidence(
                    claim=Claim(
                        text="Short text.",
                        start_idx=0,
                        end_idx=100,  # Beyond response length
                        is_factual=True,
                        is_opinion=False
                    ),
                    is_supported=False,
                    similarity_score=0.2,
                    evidence_sources=[]
                )
            ]

            spans = detector.generate_flagged_spans(claim_evidence, response)

            assert len(spans) == 0


class TestSensitivityLevelConfigurator:
    """
    Test suite for sensitivity level configuration.

    Requirement 6.8: Support configurable sensitivity levels
    """

    def test_set_sensitivity_level_strict(self):
        """Test updating sensitivity level to STRICT."""
        detector = HallucinationDetector()
        assert detector.config.sensitivity_level == SensitivityLevel.MODERATE

        detector.set_sensitivity_level(SensitivityLevel.STRICT)

        assert detector.config.sensitivity_level == SensitivityLevel.STRICT
        assert detector.config.similarity_threshold == 0.8
        # Classifier should be recreated with new threshold
        assert detector.claim_classifier.get_threshold() == 0.8

    def test_set_sensitivity_level_lenient(self):
        """Test updating sensitivity level to LENIENT."""
        detector = HallucinationDetector()

        detector.set_sensitivity_level(SensitivityLevel.LENIENT)

        assert detector.config.sensitivity_level == SensitivityLevel.LENIENT
        assert detector.config.similarity_threshold == 0.4
        assert detector.claim_classifier.get_threshold() == 0.4

    def test_set_sensitivity_level_moderate(self):
        """Test updating sensitivity level to MODERATE."""
        config = HallucinationConfig(
            sensitivity_level=SensitivityLevel.STRICT,
            similarity_threshold=0.8
        )
        detector = HallucinationDetector(config)

        detector.set_sensitivity_level(SensitivityLevel.MODERATE)

        assert detector.config.sensitivity_level == SensitivityLevel.MODERATE
        assert detector.config.similarity_threshold == 0.6

    def test_get_sensitivity_config_default(self):
        """Test getting sensitivity config with defaults."""
        detector = HallucinationDetector()

        config = detector.get_sensitivity_config()

        assert config["sensitivity_level"] == SensitivityLevel.MODERATE
        assert config["threshold"] == 0.6
        assert config["all_thresholds"] == {
            "strict": 0.8,
            "moderate": 0.6,
            "lenient": 0.4,
        }

    def test_get_sensitivity_config_after_update(self):
        """Test getting sensitivity config after changing level."""
        detector = HallucinationDetector()
        detector.set_sensitivity_level(SensitivityLevel.STRICT)

        config = detector.get_sensitivity_config()

        assert config["sensitivity_level"] == SensitivityLevel.STRICT
        assert config["threshold"] == 0.8

    def test_detect_per_run_sensitivity_override(self):
        """Test per-run sensitivity override in detect()."""
        detector = HallucinationDetector()
        assert detector.config.sensitivity_level == SensitivityLevel.MODERATE

        result = detector.detect(
            "The sky is blue.",
            sensitivity_level=SensitivityLevel.STRICT
        )

        # Result should reflect the per-run override
        assert result.sensitivity_level == SensitivityLevel.STRICT
        # Detector's persistent config should be unchanged
        assert detector.config.sensitivity_level == SensitivityLevel.MODERATE

    def test_detect_without_override_uses_default(self):
        """Test detect() without override uses detector's config."""
        config = HallucinationConfig(
            sensitivity_level=SensitivityLevel.LENIENT,
            similarity_threshold=0.4
        )
        detector = HallucinationDetector(config)

        result = detector.detect("The sky is blue.")

        assert result.sensitivity_level == SensitivityLevel.LENIENT

    def test_detect_same_level_override_no_change(self):
        """Test that passing the same level as config works."""
        detector = HallucinationDetector()

        result = detector.detect(
            "The sky is blue.",
            sensitivity_level=SensitivityLevel.MODERATE
        )

        assert result.sensitivity_level == SensitivityLevel.MODERATE

    def test_sensitivity_thresholds_ordering(self):
        """Test that STRICT >= MODERATE >= LENIENT thresholds."""
        from src.hallucination.hallucination_detector import (
            SENSITIVITY_THRESHOLDS,
        )

        strict = SENSITIVITY_THRESHOLDS[SensitivityLevel.STRICT]
        moderate = SENSITIVITY_THRESHOLDS[SensitivityLevel.MODERATE]
        lenient = SENSITIVITY_THRESHOLDS[SensitivityLevel.LENIENT]

        assert strict >= moderate >= lenient
        assert strict == 0.8
        assert moderate == 0.6
        assert lenient == 0.4


class TestHallucinationComparison:
    """
    Test suite for hallucination rate comparison.

    Requirement 6.6: Compare hallucination rates between models
    """

    def _make_result(
        self,
        hallucination_rate: float,
        factual_claims: int,
        unsupported_claims: int,
    ) -> HallucinationResult:
        """Helper to create a HallucinationResult for comparison tests."""
        supported = factual_claims - unsupported_claims
        return HallucinationResult(
            has_hallucinations=hallucination_rate > 0.0,
            hallucination_rate=hallucination_rate,
            total_claims=factual_claims,
            factual_claims=factual_claims,
            supported_claims=supported,
            unsupported_claims=unsupported_claims,
            flagged_spans=[],
            overall_grounding_score=1.0 - hallucination_rate,
            claim_evidence=[],
            sensitivity_level=SensitivityLevel.MODERATE,
        )

    def test_compare_basic_improvement(self):
        """Test comparison where fine-tuned model improves."""
        detector = HallucinationDetector()
        baseline = self._make_result(0.4, 10, 4)
        finetuned = self._make_result(0.2, 10, 2)

        comparison = detector.compare_hallucination_rates(baseline, finetuned)

        assert comparison.baseline_mean_rate == 0.4
        assert comparison.comparison_mean_rate == 0.2
        assert abs(comparison.reduction - 0.2) < 1e-9
        assert abs(comparison.reduction_percent - 50.0) < 1e-9

    def test_compare_no_improvement(self):
        """Test comparison where rates are identical."""
        detector = HallucinationDetector()
        baseline = self._make_result(0.3, 10, 3)
        finetuned = self._make_result(0.3, 10, 3)

        comparison = detector.compare_hallucination_rates(baseline, finetuned)

        assert comparison.reduction == 0.0
        assert comparison.reduction_percent == 0.0

    def test_compare_regression(self):
        """Test comparison where fine-tuned model is worse."""
        detector = HallucinationDetector()
        baseline = self._make_result(0.2, 10, 2)
        finetuned = self._make_result(0.5, 10, 5)

        comparison = detector.compare_hallucination_rates(baseline, finetuned)

        assert comparison.reduction < 0.0
        assert comparison.reduction_percent < 0.0

    def test_compare_zero_baseline_rate(self):
        """Test comparison when baseline has zero hallucination rate."""
        detector = HallucinationDetector()
        baseline = self._make_result(0.0, 10, 0)
        finetuned = self._make_result(0.1, 10, 1)

        comparison = detector.compare_hallucination_rates(baseline, finetuned)

        assert comparison.baseline_mean_rate == 0.0
        assert comparison.reduction_percent == 0.0  # Cannot divide by zero

    def test_compare_both_zero_rates(self):
        """Test comparison when both models have zero hallucination rate."""
        detector = HallucinationDetector()
        baseline = self._make_result(0.0, 10, 0)
        finetuned = self._make_result(0.0, 10, 0)

        comparison = detector.compare_hallucination_rates(baseline, finetuned)

        assert comparison.reduction == 0.0
        assert comparison.reduction_percent == 0.0
        assert comparison.p_value == 1.0

    def test_compare_statistical_significance_large_difference(self):
        """Test that large differences yield high statistical significance."""
        detector = HallucinationDetector()
        # Large sample with big difference
        baseline = self._make_result(0.8, 100, 80)
        finetuned = self._make_result(0.1, 100, 10)

        comparison = detector.compare_hallucination_rates(baseline, finetuned)

        assert comparison.p_value < 0.05
        assert comparison.statistical_significance > 0.95

    def test_compare_statistical_significance_no_factual_claims(self):
        """Test significance when no factual claims exist."""
        detector = HallucinationDetector()
        baseline = self._make_result(0.0, 0, 0)
        finetuned = self._make_result(0.0, 0, 0)

        comparison = detector.compare_hallucination_rates(baseline, finetuned)

        # Cannot compute significance with no claims
        assert comparison.p_value == 1.0
        assert comparison.statistical_significance == 0.0

    def test_compare_p_value_range(self):
        """Test that p-value is always in [0, 1]."""
        detector = HallucinationDetector()

        test_cases = [
            (self._make_result(0.5, 20, 10), self._make_result(0.3, 20, 6)),
            (self._make_result(0.0, 5, 0), self._make_result(0.0, 5, 0)),
            (self._make_result(1.0, 5, 5), self._make_result(0.0, 5, 0)),
            (self._make_result(0.1, 50, 5), self._make_result(0.08, 50, 4)),
        ]

        for baseline, finetuned in test_cases:
            comparison = detector.compare_hallucination_rates(
                baseline, finetuned
            )
            assert 0.0 <= comparison.p_value <= 1.0
            assert 0.0 <= comparison.statistical_significance <= 1.0

    def test_compare_complete_elimination(self):
        """Test comparison where fine-tuned model eliminates hallucinations."""
        detector = HallucinationDetector()
        baseline = self._make_result(0.6, 10, 6)
        finetuned = self._make_result(0.0, 10, 0)

        comparison = detector.compare_hallucination_rates(baseline, finetuned)

        assert comparison.reduction == 0.6
        assert abs(comparison.reduction_percent - 100.0) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAggregateMetrics:
    """
    Test suite for aggregate hallucination metrics.

    Requirements: 6.7, 6.10
    """

    def _make_result(
        self,
        hallucination_rate: float,
        factual_claims: int = 10,
        unsupported_claims: int = 0,
        total_claims: int = 10,
        grounding_score: float = 0.8,
        num_flagged_spans: int = 0,
    ) -> HallucinationResult:
        """Helper to create a HallucinationResult for aggregation tests."""
        supported = factual_claims - unsupported_claims
        return HallucinationResult(
            has_hallucinations=hallucination_rate > 0.0,
            hallucination_rate=hallucination_rate,
            total_claims=total_claims,
            factual_claims=factual_claims,
            supported_claims=supported,
            unsupported_claims=unsupported_claims,
            flagged_spans=[],
            overall_grounding_score=grounding_score,
            claim_evidence=[],
            sensitivity_level=SensitivityLevel.MODERATE,
        )

    def test_aggregate_empty_results(self):
        """Test aggregation with empty results list."""
        detector = HallucinationDetector()

        metrics = detector.aggregate_metrics([])

        assert metrics["total_results"] == 0
        assert metrics["group_by"] == "none"
        assert metrics["groups"] == {}
        assert metrics["overall"]["count"] == 0
        assert metrics["overall"]["mean_rate"] == 0.0

    def test_aggregate_single_result_no_grouping(self):
        """Test aggregation with a single result, no grouping."""
        detector = HallucinationDetector()
        result = self._make_result(0.3, 10, 3, grounding_score=0.7)

        metrics = detector.aggregate_metrics([result])

        assert metrics["total_results"] == 1
        overall = metrics["overall"]
        assert overall["count"] == 1
        assert overall["mean_rate"] == 0.3
        assert overall["median_rate"] == 0.3
        assert overall["std_rate"] == 0.0  # single value
        assert overall["min_rate"] == 0.3
        assert overall["max_rate"] == 0.3
        assert overall["mean_grounding_score"] == 0.7

    def test_aggregate_multiple_results_no_grouping(self):
        """Test aggregation with multiple results, no grouping."""
        detector = HallucinationDetector()
        results = [
            self._make_result(0.2, 10, 2, grounding_score=0.8),
            self._make_result(0.4, 10, 4, grounding_score=0.6),
            self._make_result(0.6, 10, 6, grounding_score=0.4),
        ]

        metrics = detector.aggregate_metrics(results)

        overall = metrics["overall"]
        assert overall["count"] == 3
        assert abs(overall["mean_rate"] - 0.4) < 1e-9
        assert abs(overall["median_rate"] - 0.4) < 1e-9
        assert overall["min_rate"] == 0.2
        assert overall["max_rate"] == 0.6
        assert overall["std_rate"] > 0.0
        assert overall["total_factual_claims"] == 30
        assert overall["total_unsupported_claims"] == 12

    def test_aggregate_by_category(self):
        """Test aggregation grouped by category."""
        detector = HallucinationDetector()
        results = [
            self._make_result(0.1, 10, 1),
            self._make_result(0.2, 10, 2),
            self._make_result(0.5, 10, 5),
            self._make_result(0.6, 10, 6),
        ]
        group_keys = ["medical", "medical", "legal", "legal"]

        metrics = detector.aggregate_metrics(
            results, group_by="category", group_keys=group_keys
        )

        assert metrics["group_by"] == "category"
        assert "medical" in metrics["groups"]
        assert "legal" in metrics["groups"]

        medical = metrics["groups"]["medical"]
        assert medical["count"] == 2
        assert abs(medical["mean_rate"] - 0.15) < 1e-9

        legal = metrics["groups"]["legal"]
        assert legal["count"] == 2
        assert abs(legal["mean_rate"] - 0.55) < 1e-9

    def test_aggregate_by_model(self):
        """Test aggregation grouped by model."""
        detector = HallucinationDetector()
        results = [
            self._make_result(0.1),
            self._make_result(0.3),
            self._make_result(0.5),
        ]
        group_keys = ["model-a", "model-a", "model-b"]

        metrics = detector.aggregate_metrics(
            results, group_by="model", group_keys=group_keys
        )

        assert "model-a" in metrics["groups"]
        assert "model-b" in metrics["groups"]
        assert metrics["groups"]["model-a"]["count"] == 2
        assert metrics["groups"]["model-b"]["count"] == 1

    def test_aggregate_by_time_period(self):
        """Test aggregation grouped by time period."""
        detector = HallucinationDetector()
        results = [
            self._make_result(0.2),
            self._make_result(0.3),
            self._make_result(0.4),
        ]
        group_keys = ["2024-01", "2024-01", "2024-02"]

        metrics = detector.aggregate_metrics(
            results,
            group_by="time_period",
            group_keys=group_keys,
        )

        assert "2024-01" in metrics["groups"]
        assert "2024-02" in metrics["groups"]
        jan = metrics["groups"]["2024-01"]
        assert jan["count"] == 2
        assert abs(jan["mean_rate"] - 0.25) < 1e-9

    def test_aggregate_overall_always_present(self):
        """Test that overall stats are always computed."""
        detector = HallucinationDetector()
        results = [
            self._make_result(0.1),
            self._make_result(0.9),
        ]

        metrics = detector.aggregate_metrics(
            results,
            group_by="category",
            group_keys=["a", "b"],
        )

        overall = metrics["overall"]
        assert overall["count"] == 2
        assert abs(overall["mean_rate"] - 0.5) < 1e-9

    def test_aggregate_missing_group_keys_raises(self):
        """Test that missing group_keys raises ValueError."""
        detector = HallucinationDetector()
        results = [self._make_result(0.1)]

        with pytest.raises(ValueError, match="group_keys is required"):
            detector.aggregate_metrics(
                results, group_by="category"
            )

    def test_aggregate_mismatched_group_keys_raises(self):
        """Test that mismatched group_keys length raises ValueError."""
        detector = HallucinationDetector()
        results = [self._make_result(0.1), self._make_result(0.2)]

        with pytest.raises(ValueError, match="must match"):
            detector.aggregate_metrics(
                results,
                group_by="model",
                group_keys=["a"],
            )

    def test_aggregate_claim_totals(self):
        """Test that claim totals are summed correctly."""
        detector = HallucinationDetector()
        results = [
            self._make_result(
                0.2, factual_claims=5, unsupported_claims=1,
                total_claims=8,
            ),
            self._make_result(
                0.4, factual_claims=10, unsupported_claims=4,
                total_claims=12,
            ),
        ]

        metrics = detector.aggregate_metrics(results)
        overall = metrics["overall"]

        assert overall["total_claims"] == 20
        assert overall["total_factual_claims"] == 15
        assert overall["total_supported_claims"] == 10
        assert overall["total_unsupported_claims"] == 5

    def test_aggregate_rates_in_valid_range(self):
        """Test that all computed rates are in [0, 1]."""
        detector = HallucinationDetector()
        results = [
            self._make_result(0.0),
            self._make_result(0.5),
            self._make_result(1.0),
        ]

        metrics = detector.aggregate_metrics(results)
        overall = metrics["overall"]

        assert 0.0 <= overall["mean_rate"] <= 1.0
        assert 0.0 <= overall["median_rate"] <= 1.0
        assert 0.0 <= overall["min_rate"] <= 1.0
        assert 0.0 <= overall["max_rate"] <= 1.0


class TestAlertTrigger:
    """
    Test suite for alert trigger functionality.

    Requirement 6.12: Alert when hallucination rate exceeds threshold
    """

    def _make_result(
        self,
        hallucination_rate: float,
        factual_claims: int = 10,
    ) -> HallucinationResult:
        """Helper to create a HallucinationResult for alert tests."""
        unsupported = int(hallucination_rate * factual_claims)
        supported = factual_claims - unsupported
        return HallucinationResult(
            has_hallucinations=hallucination_rate > 0.0,
            hallucination_rate=hallucination_rate,
            total_claims=factual_claims,
            factual_claims=factual_claims,
            supported_claims=supported,
            unsupported_claims=unsupported,
            flagged_spans=[],
            overall_grounding_score=1.0 - hallucination_rate,
            claim_evidence=[],
            sensitivity_level=SensitivityLevel.MODERATE,
        )

    def test_alert_triggered_when_rate_exceeds_threshold(self):
        """Test that alert is triggered when rate exceeds threshold."""
        detector = HallucinationDetector()
        result = self._make_result(0.7)

        alert = detector.check_alert_threshold(result, alert_threshold=0.5)

        assert alert["triggered"] is True
        assert alert["hallucination_rate"] == 0.7
        assert alert["threshold"] == 0.5
        assert "ALERT" in alert["message"]
        assert "exceeds" in alert["message"]

    def test_alert_not_triggered_when_rate_within_threshold(self):
        """Test that alert is not triggered when rate is within threshold."""
        detector = HallucinationDetector()
        result = self._make_result(0.3)

        alert = detector.check_alert_threshold(result, alert_threshold=0.5)

        assert alert["triggered"] is False
        assert alert["hallucination_rate"] == 0.3
        assert alert["threshold"] == 0.5
        assert "within" in alert["message"]

    def test_alert_not_triggered_when_rate_equals_threshold(self):
        """Test that alert is not triggered when rate equals threshold."""
        detector = HallucinationDetector()
        result = self._make_result(0.5)

        alert = detector.check_alert_threshold(result, alert_threshold=0.5)

        assert alert["triggered"] is False

    def test_alert_default_threshold(self):
        """Test that default threshold is 0.5."""
        detector = HallucinationDetector()
        result = self._make_result(0.6)

        alert = detector.check_alert_threshold(result)

        assert alert["triggered"] is True
        assert alert["threshold"] == 0.5

    def test_alert_custom_threshold(self):
        """Test alert with a custom threshold."""
        detector = HallucinationDetector()
        result = self._make_result(0.3)

        alert = detector.check_alert_threshold(result, alert_threshold=0.2)

        assert alert["triggered"] is True
        assert alert["threshold"] == 0.2

    def test_alert_zero_rate(self):
        """Test alert with zero hallucination rate."""
        detector = HallucinationDetector()
        result = self._make_result(0.0)

        alert = detector.check_alert_threshold(result, alert_threshold=0.5)

        assert alert["triggered"] is False
        assert alert["hallucination_rate"] == 0.0

    def test_alert_rate_of_one(self):
        """Test alert with hallucination rate of 1.0."""
        detector = HallucinationDetector()
        result = self._make_result(1.0)

        alert = detector.check_alert_threshold(result, alert_threshold=0.5)

        assert alert["triggered"] is True
        assert alert["hallucination_rate"] == 1.0

    def test_alert_return_structure(self):
        """Test that alert dict has all required keys."""
        detector = HallucinationDetector()
        result = self._make_result(0.4)

        alert = detector.check_alert_threshold(result)

        assert "triggered" in alert
        assert "hallucination_rate" in alert
        assert "threshold" in alert
        assert "message" in alert
        assert isinstance(alert["triggered"], bool)
        assert isinstance(alert["hallucination_rate"], float)
        assert isinstance(alert["threshold"], float)
        assert isinstance(alert["message"], str)
