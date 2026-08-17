"""
Claim Classifier for Hallucination Detection.

This module implements claim classification, determining whether claims are
supported or unsupported based on evidence similarity scores from the
evidence searcher.

Requirements: 6.3, 6.8
"""

from typing import Optional

from src.data_models.hallucination import (
    Claim,
    ClaimEvidence,
    EvidenceSource,
    HallucinationConfig,
    SensitivityLevel
)


class ClaimClassifier:
    """
    Classify claims as supported or unsupported based on evidence.

    The claim classifier determines whether claims are supported by evidence
    based on similarity scores. It uses configurable thresholds that vary
    by sensitivity level (strict, moderate, lenient) to accommodate different
    use cases and risk tolerances.

    Requirement 6.3: Classify claims based on evidence similarity
    Requirement 6.8: Support sensitivity levels (strict, moderate, lenient)
    """

    # Threshold mappings for each sensitivity level
    SENSITIVITY_THRESHOLDS = {
        SensitivityLevel.STRICT: 0.8,
        SensitivityLevel.MODERATE: 0.6,
        SensitivityLevel.LENIENT: 0.4
    }

    def __init__(self, config: Optional[HallucinationConfig] = None):
        """
        Initialize the claim classifier.

        Args:
            config: Hallucination detection configuration (optional)
        """
        self.config = config or HallucinationConfig()

        # Determine the threshold based on sensitivity level
        self.threshold = self._get_threshold()

    def _get_threshold(self) -> float:
        """
        Get the similarity threshold based on configuration.

        Returns the configured similarity_threshold if explicitly set,
        otherwise uses the threshold mapped to the sensitivity_level.

        Returns:
            Similarity threshold value (0.0 to 1.0)
        """
        # If similarity_threshold is explicitly set and differs from default,
        # use it. Otherwise, use the sensitivity level mapping.
        default_config = HallucinationConfig()

        if self.config.similarity_threshold != default_config.similarity_threshold:
            # User explicitly set a custom threshold
            return self.config.similarity_threshold
        else:
            # Use sensitivity level mapping
            return self.SENSITIVITY_THRESHOLDS[self.config.sensitivity_level]

    def classify_claim(
        self,
        claim: Claim,
        evidence_sources: list[EvidenceSource]
    ) -> ClaimEvidence:
        """
        Classify a claim as supported or unsupported based on evidence.

        Determines whether a claim is supported by examining the similarity
        scores of evidence sources. A claim is considered supported if at
        least one evidence source has a similarity score >= threshold.

        Args:
            claim: The claim to classify
            evidence_sources: List of evidence sources with similarity scores

        Returns:
            ClaimEvidence object with classification and confidence score

        Requirement 6.3: Classify claims as supported/unsupported based on
        evidence similarity threshold
        """
        # Handle edge case: no evidence found
        if not evidence_sources:
            return ClaimEvidence(
                claim=claim,
                is_supported=False,
                similarity_score=0.0,
                best_matching_document=None,
                best_matching_span=None,
                evidence_sources=[]
            )

        # Find the best evidence (highest similarity score)
        best_evidence = max(
            evidence_sources,
            key=lambda e: e.similarity_score
        )

        # Classify based on threshold
        is_supported = best_evidence.similarity_score >= self.threshold

        # Build the ClaimEvidence result
        return ClaimEvidence(
            claim=claim,
            is_supported=is_supported,
            similarity_score=best_evidence.similarity_score,
            best_matching_document=best_evidence.document_id,
            best_matching_span=best_evidence.span,
            evidence_sources=evidence_sources
        )

    def classify_claims(
        self,
        claims: list[Claim],
        evidence_map: dict[str, list[EvidenceSource]]
    ) -> list[ClaimEvidence]:
        """
        Classify multiple claims based on their evidence.

        Args:
            claims: List of claims to classify
            evidence_map: Dictionary mapping claim text to evidence sources

        Returns:
            List of ClaimEvidence objects with classifications

        Requirement 6.3: Classify claims based on evidence
        """
        results = []

        for claim in claims:
            # Get evidence for this claim (use claim text as key)
            evidence_sources = evidence_map.get(claim.text, [])

            # Classify the claim
            claim_evidence = self.classify_claim(claim, evidence_sources)
            results.append(claim_evidence)

        return results

    def get_threshold(self) -> float:
        """
        Get the current similarity threshold being used.

        Returns:
            Current threshold value (0.0 to 1.0)
        """
        return self.threshold

    def get_sensitivity_level(self) -> SensitivityLevel:
        """
        Get the current sensitivity level.

        Returns:
            Current sensitivity level
        """
        return self.config.sensitivity_level
