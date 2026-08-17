"""
Unit tests for ClaimClassifier.

Tests claim classification based on evidence similarity scores and
configurable sensitivity levels for hallucination detection.

Requirements: 6.3, 6.8
"""

import pytest

from src.data_models.hallucination import (
    Claim,
    ClaimEvidence,
    EvidenceSource,
    HallucinationConfig,
    SensitivityLevel
)
from src.hallucination.claim_classifier import ClaimClassifier


class TestClaimClassifier:
    """Test suite for ClaimClassifier class."""

    def test_classify_claim_supported(self):
        """Test classifying a claim with strong evidence as supported."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="The Earth orbits the Sun.",
            start_idx=0,
            end_idx=27,
            is_factual=True,
            is_opinion=False
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="The Earth revolves around the Sun in an elliptical orbit.",
                similarity_score=0.85
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        assert isinstance(result, ClaimEvidence)
        assert result.is_supported is True
        assert result.similarity_score == 0.85
        assert result.best_matching_document == "doc1"
        assert result.claim == claim
        assert len(result.evidence_sources) == 1

    def test_classify_claim_unsupported(self):
        """Test classifying a claim with weak evidence as unsupported."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="The Moon is made of cheese.",
            start_idx=0,
            end_idx=29,
            is_factual=True,
            is_opinion=False
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="The Moon is a rocky satellite.",
                similarity_score=0.3
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        assert result.is_supported is False
        assert result.similarity_score == 0.3
        assert result.best_matching_document == "doc1"

    def test_classify_claim_no_evidence(self):
        """Test classifying a claim with no evidence found."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="This is an unsupported claim.",
            start_idx=0,
            end_idx=30,
            is_factual=True,
            is_opinion=False
        )

        result = classifier.classify_claim(claim, [])

        assert result.is_supported is False
        assert result.similarity_score == 0.0
        assert result.best_matching_document is None
        assert result.best_matching_span is None
        assert len(result.evidence_sources) == 0

    def test_classify_claim_multiple_evidence_sources(self):
        """Test classification with multiple evidence sources."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="Python was created by Guido van Rossum.",
            start_idx=0,
            end_idx=40,
            is_factual=True,
            is_opinion=False
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="Python is a programming language.",
                similarity_score=0.4
            ),
            EvidenceSource(
                document_id="doc2",
                span="Guido van Rossum created Python in 1991.",
                similarity_score=0.9
            ),
            EvidenceSource(
                document_id="doc3",
                span="Python has many features.",
                similarity_score=0.2
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        # Should use the best evidence (highest score)
        assert result.is_supported is True
        assert result.similarity_score == 0.9
        assert result.best_matching_document == "doc2"
        assert len(result.evidence_sources) == 3

    def test_classify_claim_threshold_boundary(self):
        """Test classification at exact threshold boundary."""
        config = HallucinationConfig(similarity_threshold=0.6)
        classifier = ClaimClassifier(config)

        claim = Claim(
            text="Test claim.",
            start_idx=0,
            end_idx=11,
            is_factual=True,
            is_opinion=False
        )

        # Exactly at threshold
        evidence_at_threshold = [
            EvidenceSource(
                document_id="doc1",
                span="Test evidence.",
                similarity_score=0.6
            )
        ]

        result_at = classifier.classify_claim(claim, evidence_at_threshold)
        assert result_at.is_supported is True

        # Just below threshold
        evidence_below = [
            EvidenceSource(
                document_id="doc1",
                span="Test evidence.",
                similarity_score=0.59
            )
        ]

        result_below = classifier.classify_claim(claim, evidence_below)
        assert result_below.is_supported is False

    def test_sensitivity_level_strict(self):
        """Test strict sensitivity level (threshold=0.8)."""
        config = HallucinationConfig(
            sensitivity_level=SensitivityLevel.STRICT
        )
        classifier = ClaimClassifier(config)

        assert classifier.get_threshold() == 0.8
        assert classifier.get_sensitivity_level() == SensitivityLevel.STRICT

        claim = Claim(
            text="Test claim.",
            start_idx=0,
            end_idx=11,
            is_factual=True,
            is_opinion=False
        )

        # Score of 0.75 is unsupported in strict mode
        evidence = [
            EvidenceSource(
                document_id="doc1",
                span="Evidence.",
                similarity_score=0.75
            )
        ]

        result = classifier.classify_claim(claim, evidence)
        assert result.is_supported is False

    def test_sensitivity_level_moderate(self):
        """Test moderate sensitivity level (threshold=0.6)."""
        config = HallucinationConfig(
            sensitivity_level=SensitivityLevel.MODERATE
        )
        classifier = ClaimClassifier(config)

        assert classifier.get_threshold() == 0.6
        assert classifier.get_sensitivity_level() == SensitivityLevel.MODERATE

        claim = Claim(
            text="Test claim.",
            start_idx=0,
            end_idx=11,
            is_factual=True,
            is_opinion=False
        )

        # Score of 0.65 is supported in moderate mode
        evidence = [
            EvidenceSource(
                document_id="doc1",
                span="Evidence.",
                similarity_score=0.65
            )
        ]

        result = classifier.classify_claim(claim, evidence)
        assert result.is_supported is True

    def test_sensitivity_level_lenient(self):
        """Test lenient sensitivity level (threshold=0.4)."""
        config = HallucinationConfig(
            sensitivity_level=SensitivityLevel.LENIENT
        )
        classifier = ClaimClassifier(config)

        assert classifier.get_threshold() == 0.4
        assert classifier.get_sensitivity_level() == SensitivityLevel.LENIENT

        claim = Claim(
            text="Test claim.",
            start_idx=0,
            end_idx=11,
            is_factual=True,
            is_opinion=False
        )

        # Score of 0.45 is supported in lenient mode
        evidence = [
            EvidenceSource(
                document_id="doc1",
                span="Evidence.",
                similarity_score=0.45
            )
        ]

        result = classifier.classify_claim(claim, evidence)
        assert result.is_supported is True

    def test_custom_threshold_overrides_sensitivity(self):
        """Test that custom threshold overrides sensitivity level."""
        config = HallucinationConfig(
            sensitivity_level=SensitivityLevel.STRICT,
            similarity_threshold=0.5  # Custom threshold
        )
        classifier = ClaimClassifier(config)

        # Should use custom threshold, not strict (0.8)
        assert classifier.get_threshold() == 0.5

    def test_classify_claims_batch(self):
        """Test classifying multiple claims at once."""
        classifier = ClaimClassifier()

        claims = [
            Claim(
                text="Claim 1.",
                start_idx=0,
                end_idx=8,
                is_factual=True,
                is_opinion=False
            ),
            Claim(
                text="Claim 2.",
                start_idx=9,
                end_idx=17,
                is_factual=True,
                is_opinion=False
            ),
            Claim(
                text="Claim 3.",
                start_idx=18,
                end_idx=26,
                is_factual=True,
                is_opinion=False
            )
        ]

        evidence_map = {
            "Claim 1.": [
                EvidenceSource(
                    document_id="doc1",
                    span="Evidence for claim 1.",
                    similarity_score=0.8
                )
            ],
            "Claim 2.": [
                EvidenceSource(
                    document_id="doc2",
                    span="Evidence for claim 2.",
                    similarity_score=0.3
                )
            ],
            "Claim 3.": []  # No evidence
        }

        results = classifier.classify_claims(claims, evidence_map)

        assert len(results) == 3
        assert results[0].is_supported is True
        assert results[1].is_supported is False
        assert results[2].is_supported is False

    def test_classify_claims_empty_list(self):
        """Test classifying empty list of claims."""
        classifier = ClaimClassifier()

        results = classifier.classify_claims([], {})

        assert len(results) == 0
        assert isinstance(results, list)

    def test_classify_claims_missing_evidence(self):
        """Test classifying claims when evidence map is incomplete."""
        classifier = ClaimClassifier()

        claims = [
            Claim(
                text="Claim 1.",
                start_idx=0,
                end_idx=8,
                is_factual=True,
                is_opinion=False
            ),
            Claim(
                text="Claim 2.",
                start_idx=9,
                end_idx=17,
                is_factual=True,
                is_opinion=False
            )
        ]

        # Evidence map only has entry for Claim 1
        evidence_map = {
            "Claim 1.": [
                EvidenceSource(
                    document_id="doc1",
                    span="Evidence.",
                    similarity_score=0.7
                )
            ]
        }

        results = classifier.classify_claims(claims, evidence_map)

        assert len(results) == 2
        assert results[0].is_supported is True
        assert results[1].is_supported is False  # No evidence in map

    def test_best_matching_span_extraction(self):
        """Test that best matching span is correctly extracted."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="Test claim.",
            start_idx=0,
            end_idx=11,
            is_factual=True,
            is_opinion=False
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="First evidence span.",
                similarity_score=0.5
            ),
            EvidenceSource(
                document_id="doc2",
                span="Best evidence span.",
                similarity_score=0.9
            ),
            EvidenceSource(
                document_id="doc3",
                span="Third evidence span.",
                similarity_score=0.6
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        assert result.best_matching_span == "Best evidence span."
        assert result.best_matching_document == "doc2"

    def test_confidence_score_equals_best_similarity(self):
        """Test that similarity_score equals the best evidence score."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="Test claim.",
            start_idx=0,
            end_idx=11,
            is_factual=True,
            is_opinion=False
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="Evidence 1.",
                similarity_score=0.4
            ),
            EvidenceSource(
                document_id="doc2",
                span="Evidence 2.",
                similarity_score=0.7
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        assert result.similarity_score == 0.7

    def test_all_evidence_sources_preserved(self):
        """Test that all evidence sources are preserved in result."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="Test claim.",
            start_idx=0,
            end_idx=11,
            is_factual=True,
            is_opinion=False
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="Evidence 1.",
                similarity_score=0.5
            ),
            EvidenceSource(
                document_id="doc2",
                span="Evidence 2.",
                similarity_score=0.6
            ),
            EvidenceSource(
                document_id="doc3",
                span="Evidence 3.",
                similarity_score=0.7
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        assert len(result.evidence_sources) == 3
        assert result.evidence_sources == evidence_sources


class TestClaimClassifierEdgeCases:
    """Test edge cases and boundary conditions for ClaimClassifier."""

    def test_zero_similarity_score(self):
        """Test handling of zero similarity score."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="Test claim.",
            start_idx=0,
            end_idx=11,
            is_factual=True,
            is_opinion=False
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="Completely unrelated evidence.",
                similarity_score=0.0
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        assert result.is_supported is False
        assert result.similarity_score == 0.0

    def test_perfect_similarity_score(self):
        """Test handling of perfect similarity score (1.0)."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="Test claim.",
            start_idx=0,
            end_idx=11,
            is_factual=True,
            is_opinion=False
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="Test claim.",
                similarity_score=1.0
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        assert result.is_supported is True
        assert result.similarity_score == 1.0

    def test_opinion_claim_classification(self):
        """Test that opinion claims can still be classified."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="I think this is interesting.",
            start_idx=0,
            end_idx=29,
            is_factual=False,
            is_opinion=True
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="This is interesting.",
                similarity_score=0.7
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        # Classifier doesn't filter opinions, just classifies them
        assert result.is_supported is True
        assert result.claim.is_opinion is True

    def test_hedged_claim_classification(self):
        """Test classification of hedged claims."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="This might be correct.",
            start_idx=0,
            end_idx=22,
            is_factual=False,
            is_opinion=True,
            is_hedged=True
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="This is correct.",
                similarity_score=0.8
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        assert result.is_supported is True
        assert result.claim.is_hedged is True

    def test_very_long_claim_text(self):
        """Test classification of very long claims."""
        classifier = ClaimClassifier()

        long_text = "This is a very long claim. " * 50
        claim = Claim(
            text=long_text,
            start_idx=0,
            end_idx=len(long_text),
            is_factual=True,
            is_opinion=False
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="Evidence for long claim.",
                similarity_score=0.7
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        assert isinstance(result, ClaimEvidence)
        assert result.claim.text == long_text

    def test_unicode_in_claims_and_evidence(self):
        """Test handling of unicode characters."""
        classifier = ClaimClassifier()

        claim = Claim(
            text="The café costs €10.",
            start_idx=0,
            end_idx=19,
            is_factual=True,
            is_opinion=False
        )

        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="Le café coûte 10€.",
                similarity_score=0.75
            )
        ]

        result = classifier.classify_claim(claim, evidence_sources)

        assert result.is_supported is True
        assert "café" in result.claim.text
        assert "€" in result.claim.text


class TestSensitivityLevelOrdering:
    """Test that sensitivity levels maintain correct ordering."""

    def test_sensitivity_threshold_ordering(self):
        """Test that STRICT >= MODERATE >= LENIENT."""
        strict_config = HallucinationConfig(
            sensitivity_level=SensitivityLevel.STRICT
        )
        moderate_config = HallucinationConfig(
            sensitivity_level=SensitivityLevel.MODERATE
        )
        lenient_config = HallucinationConfig(
            sensitivity_level=SensitivityLevel.LENIENT
        )

        strict_classifier = ClaimClassifier(strict_config)
        moderate_classifier = ClaimClassifier(moderate_config)
        lenient_classifier = ClaimClassifier(lenient_config)

        strict_threshold = strict_classifier.get_threshold()
        moderate_threshold = moderate_classifier.get_threshold()
        lenient_threshold = lenient_classifier.get_threshold()

        assert strict_threshold >= moderate_threshold
        assert moderate_threshold >= lenient_threshold
        assert strict_threshold > lenient_threshold

    def test_same_evidence_different_sensitivity(self):
        """Test that same evidence produces different results by sensitivity."""
        claim = Claim(
            text="Test claim.",
            start_idx=0,
            end_idx=11,
            is_factual=True,
            is_opinion=False
        )

        # Evidence with score between lenient and strict thresholds
        evidence_sources = [
            EvidenceSource(
                document_id="doc1",
                span="Evidence.",
                similarity_score=0.5
            )
        ]

        # Lenient should support (threshold=0.4)
        lenient_classifier = ClaimClassifier(
            HallucinationConfig(sensitivity_level=SensitivityLevel.LENIENT)
        )
        lenient_result = lenient_classifier.classify_claim(
            claim, evidence_sources
        )

        # Strict should not support (threshold=0.8)
        strict_classifier = ClaimClassifier(
            HallucinationConfig(sensitivity_level=SensitivityLevel.STRICT)
        )
        strict_result = strict_classifier.classify_claim(
            claim, evidence_sources
        )

        assert lenient_result.is_supported is True
        assert strict_result.is_supported is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
