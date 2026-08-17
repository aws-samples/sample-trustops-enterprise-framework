"""
Hallucination Detector for TrustOps Enterprise Framework.

This module implements the main HallucinationDetector class that orchestrates
claim extraction, evidence search, claim classification, and hallucination
rate calculation.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.8, 6.9, 6.11, 6.12, 6.13
"""

from typing import Optional

import math
import statistics

from src.data_models.hallucination import (
    ClaimEvidence,
    FlaggedSpan,
    HallucinationComparison,
    HallucinationConfig,
    HallucinationResult,
    SensitivityLevel
)
from src.hallucination.claim_classifier import ClaimClassifier
from src.hallucination.claim_extractor import ClaimExtractor
from src.hallucination.evidence_searcher import EvidenceSearcher


# Threshold mappings for each sensitivity level
SENSITIVITY_THRESHOLDS: dict[SensitivityLevel, float] = {
    SensitivityLevel.STRICT: 0.8,
    SensitivityLevel.MODERATE: 0.6,
    SensitivityLevel.LENIENT: 0.4,
}


class HallucinationDetector:
    """
    Detect and measure hallucinations in model outputs.

    The HallucinationDetector orchestrates the complete hallucination detection
    pipeline: extracting claims from responses, searching for supporting
    evidence in source documents, classifying claims as supported/unsupported,
    and calculating hallucination rates.

    Requirements:
    - 6.1: Extract claims from responses
    - 6.2: Search for supporting evidence
    - 6.3: Classify claims as supported/unsupported
    - 6.4: Calculate hallucination rate
    - 6.5: Identify flagged text spans
    - 6.8: Support configurable sensitivity levels
    """

    def __init__(self, config: Optional[HallucinationConfig] = None):
        """
        Initialize the hallucination detector.

        Args:
            config: Hallucination detection configuration (optional)
        """
        self.config = config or HallucinationConfig()
        self.claim_extractor = ClaimExtractor(self.config)
        self.claim_classifier = ClaimClassifier(self.config)
        self.evidence_searcher = EvidenceSearcher(self.config)

    def set_sensitivity_level(self, level: SensitivityLevel) -> None:
        """
        Update the sensitivity level after initialization.

        This updates the detector's configuration and recreates the
        ClaimClassifier with the new threshold derived from the level.

        Args:
            level: The new sensitivity level to use

        Requirement 6.8: Support configurable sensitivity levels
        """
        self.config.sensitivity_level = level
        self.config.similarity_threshold = SENSITIVITY_THRESHOLDS[level]
        self.claim_classifier = ClaimClassifier(self.config)

    def get_sensitivity_config(self) -> dict:
        """
        Get the current sensitivity configuration.

        Returns a dictionary with the current sensitivity level, threshold,
        and the full threshold mapping for all levels.

        Returns:
            Dictionary with sensitivity_level, threshold, and all_thresholds

        Requirement 6.8: Support configurable sensitivity levels
        """
        return {
            "sensitivity_level": self.config.sensitivity_level,
            "threshold": SENSITIVITY_THRESHOLDS[self.config.sensitivity_level],
            "all_thresholds": {
                level.value: threshold
                for level, threshold in SENSITIVITY_THRESHOLDS.items()
            },
        }

    def calculate_hallucination_rate(
        self,
        claim_evidence: list[ClaimEvidence]
    ) -> float:
        """
        Calculate the hallucination rate from claim evidence.

        The hallucination rate is defined as the ratio of unsupported factual
        claims to total factual claims. This metric quantifies how much of the
        response consists of unsupported assertions.

        Formula: hallucination_rate = unsupported_factual / total_factual

        Edge case: If there are no factual claims (all opinions/hedged
        statements), the hallucination rate is 0.0 since there are no factual
        assertions to be unsupported.

        Args:
            claim_evidence: List of ClaimEvidence objects from classification

        Returns:
            Hallucination rate in range [0.0, 1.0]

        Requirement 6.4: Calculate hallucination rate as ratio of unsupported
        to total factual claims
        """
        # Filter to only factual claims (exclude opinions)
        factual_claims = [
            ce for ce in claim_evidence
            if ce.claim.is_factual
        ]

        # Edge case: no factual claims means no hallucinations
        if len(factual_claims) == 0:
            return 0.0

        # Count unsupported factual claims
        unsupported_count = sum(
            1 for ce in factual_claims
            if not ce.is_supported
        )

        # Calculate ratio
        hallucination_rate = unsupported_count / len(factual_claims)

        return hallucination_rate

    def generate_flagged_spans(
        self,
        claim_evidence: list[ClaimEvidence],
        response: str
    ) -> list[FlaggedSpan]:
        """
        Generate FlaggedSpan objects from classified claim evidence.

        Maps unsupported factual claims to their character positions in the
        original response, including grounding score and reason for flagging.
        Only unsupported factual claims are flagged — opinions are excluded.

        Args:
            claim_evidence: List of ClaimEvidence objects from classification
            response: The original model response text

        Returns:
            List of FlaggedSpan objects for unsupported factual claims

        Requirement 6.5: Identify flagged text spans with low grounding scores
        """
        flagged_spans = []

        for ce in claim_evidence:
            # Only flag unsupported factual claims (not opinions)
            if ce.is_supported or not ce.claim.is_factual:
                continue

            # Validate that the span indices are within the response bounds
            start_idx = ce.claim.start_idx
            end_idx = ce.claim.end_idx

            if (
                start_idx < 0
                or end_idx > len(response)
                or start_idx >= end_idx
            ):
                continue

            # Determine the reason for flagging
            if ce.similarity_score == 0.0 and not ce.evidence_sources:
                reason = "No supporting evidence found"
            else:
                reason = f"Low similarity score: {ce.similarity_score:.2f}"

            flagged_span = FlaggedSpan(
                text=ce.claim.text,
                start_idx=start_idx,
                end_idx=end_idx,
                grounding_score=ce.similarity_score,
                reason=reason
            )

            flagged_spans.append(flagged_span)

        return flagged_spans

    def detect(
        self,
        response: str,
        source_documents: Optional[list[str]] = None,
        sensitivity_level: Optional[SensitivityLevel] = None
    ) -> HallucinationResult:
        """
        Detect hallucinations in a model response.

        Supports per-run sensitivity override: if a sensitivity_level is
        provided, it is used for this detection run only without changing
        the detector's persistent configuration.

        Args:
            response: The model response to analyze
            source_documents: Source documents for evidence search (optional)
            sensitivity_level: Optional per-run sensitivity level override

        Returns:
            HallucinationResult with detection analysis

        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.8
        """
        # Determine which classifier to use for this run
        if (
            sensitivity_level is not None
            and sensitivity_level != self.config.sensitivity_level
        ):
            run_config = HallucinationConfig(
                sensitivity_level=sensitivity_level,
                similarity_threshold=SENSITIVITY_THRESHOLDS[
                    sensitivity_level
                ],
                max_claims_per_response=(
                    self.config.max_claims_per_response
                ),
                hedging_keywords=self.config.hedging_keywords,
                embedding_model_id=self.config.embedding_model_id,
            )
            self._run_classifier = ClaimClassifier(run_config)
            effective_level = sensitivity_level
        else:
            self._run_classifier = self.claim_classifier
            effective_level = self.config.sensitivity_level

        # Extract claims from response
        claims = self.claim_extractor.extract_claims(response)

        # Search for evidence and classify claims
        if source_documents and claims:
            # Use evidence searcher to find supporting evidence
            evidence_map = self.evidence_searcher.search_evidence_batch(
                claims, source_documents
            )
            # Use claim classifier to classify each claim
            claim_evidence = self._run_classifier.classify_claims(
                claims, evidence_map
            )
        else:
            # No source documents or no claims — mark all factual claims
            # as unsupported with zero similarity
            claim_evidence = self._run_classifier.classify_claims(
                claims, {}
            )

        # Calculate metrics
        total_claims = len(claims)
        factual_claims = sum(1 for c in claims if c.is_factual)
        supported_claims = sum(
            1 for ce in claim_evidence
            if ce.is_supported and ce.claim.is_factual
        )
        unsupported_claims = factual_claims - supported_claims

        # Calculate hallucination rate
        hallucination_rate = self.calculate_hallucination_rate(claim_evidence)

        # Generate flagged spans for unsupported factual claims
        flagged_spans = self.generate_flagged_spans(claim_evidence, response)

        # Calculate overall grounding score
        if claim_evidence:
            overall_grounding_score = sum(
                ce.similarity_score for ce in claim_evidence
            ) / len(claim_evidence)
        else:
            overall_grounding_score = 0.0

        # Determine if hallucinations were detected
        has_hallucinations = hallucination_rate > 0.0

        return HallucinationResult(
            has_hallucinations=has_hallucinations,
            hallucination_rate=hallucination_rate,
            total_claims=total_claims,
            factual_claims=factual_claims,
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            flagged_spans=flagged_spans,
            overall_grounding_score=overall_grounding_score,
            claim_evidence=claim_evidence,
            sensitivity_level=effective_level
        )

    def detect_batch(
        self,
        items: list[dict],
        sensitivity_level: Optional[SensitivityLevel] = None,
    ) -> list[HallucinationResult]:
        """
        Detect hallucinations in multiple responses (batch mode).

        Each item in the list should be a dictionary with:
        - "response" (str): The model response to analyze
        - "source_documents" (list[str], optional): Source documents

        Args:
            items: List of dicts with "response" and optional
                "source_documents" keys.
            sensitivity_level: Optional per-run sensitivity level override
                applied to all items in the batch.

        Returns:
            List of HallucinationResult objects, one per item.

        Requirements: 6.1-6.5, 6.8
        """
        results: list[HallucinationResult] = []
        for item in items:
            response = item.get("response", "")
            source_documents = item.get("source_documents")
            result = self.detect(
                response=response,
                source_documents=source_documents,
                sensitivity_level=sensitivity_level,
            )
            results.append(result)
        return results

    def compare_hallucination_rates(
        self,
        baseline_result: HallucinationResult,
        finetuned_result: HallucinationResult,
    ) -> HallucinationComparison:
        """
        Compare hallucination rates between baseline and fine-tuned models.

        Calculates absolute reduction, percentage reduction, and statistical
        significance of the difference using a two-proportion z-test.

        Args:
            baseline_result: HallucinationResult from the baseline model
            finetuned_result: HallucinationResult from the fine-tuned model

        Returns:
            HallucinationComparison with reduction metrics and significance

        Requirement 6.6: Compare hallucination rates between models
        """
        baseline_rate = baseline_result.hallucination_rate
        finetuned_rate = finetuned_result.hallucination_rate

        # Absolute reduction (positive means improvement)
        reduction = baseline_rate - finetuned_rate

        # Percentage reduction (handle zero baseline)
        if baseline_rate == 0.0:
            reduction_percent = 0.0
        else:
            reduction_percent = (reduction / baseline_rate) * 100.0

        # Statistical significance using two-proportion z-test
        p_value = self._calculate_proportion_test_p_value(
            baseline_result, finetuned_result
        )
        statistical_significance = 1.0 - p_value

        return HallucinationComparison(
            baseline_mean_rate=baseline_rate,
            comparison_mean_rate=finetuned_rate,
            reduction=reduction,
            reduction_percent=reduction_percent,
            statistical_significance=statistical_significance,
            p_value=p_value,
        )

    def aggregate_metrics(
        self,
        results: list[HallucinationResult],
        group_by: str = "none",
        group_keys: Optional[list[str]] = None,
    ) -> dict:
        """
        Aggregate hallucination metrics across multiple results.

        Groups results by category, model_id, or time period and calculates
        aggregate statistics for each group: mean rate, median rate, standard
        deviation, min, and max.

        Args:
            results: List of HallucinationResult objects to aggregate
            group_by: Grouping strategy - "none", "category",
                "model", or "time_period". When "none", all results
                are aggregated together.
            group_keys: Parallel list of group keys corresponding to each
                result. Required when group_by is not "none". Each key
                identifies which group the corresponding result belongs to.

        Returns:
            Dictionary with aggregated metrics. Structure:
            {
                "total_results": int,
                "group_by": str,
                "groups": {
                    "<group_key>": {
                        "count": int,
                        "mean_rate": float,
                        "median_rate": float,
                        "std_rate": float,
                        "min_rate": float,
                        "max_rate": float,
                        "mean_grounding_score": float,
                        "total_claims": int,
                        "total_factual_claims": int,
                        "total_supported_claims": int,
                        "total_unsupported_claims": int,
                        "total_flagged_spans": int,
                    },
                    ...
                },
                "overall": { ... same structure as group ... }
            }

        Raises:
            ValueError: If group_by is not "none" and group_keys is None or
                has a different length than results.

        Requirements: 6.7, 6.10
        """
        if not results:
            return {
                "total_results": 0,
                "group_by": group_by,
                "groups": {},
                "overall": self._compute_group_stats([]),
            }

        # Validate group_keys when grouping is requested
        if group_by != "none":
            if group_keys is None:
                raise ValueError(
                    "group_keys is required when group_by is not 'none'"
                )
            if len(group_keys) != len(results):
                raise ValueError(
                    f"group_keys length ({len(group_keys)}) must match "
                    f"results length ({len(results)})"
                )

        # Build groups
        groups: dict[str, list[HallucinationResult]] = {}
        if group_by == "none":
            groups["all"] = results
        else:
            for result, key in zip(  # type: ignore[arg-type]
                results, group_keys
            ):
                groups.setdefault(key, []).append(result)

        # Compute stats per group
        group_stats = {
            key: self._compute_group_stats(group_results)
            for key, group_results in groups.items()
        }

        return {
            "total_results": len(results),
            "group_by": group_by,
            "groups": group_stats,
            "overall": self._compute_group_stats(results),
        }

    def check_alert_threshold(
        self,
        result: HallucinationResult,
        alert_threshold: float = 0.5,
    ) -> dict:
        """
        Check if a hallucination result exceeds the alert threshold.

        Evaluates the hallucination rate from a detection result against a
        configurable threshold and returns an alert dictionary indicating
        whether the threshold was exceeded.

        Args:
            result: A HallucinationResult from a detection run
            alert_threshold: The hallucination rate threshold that triggers
                an alert. Must be in [0.0, 1.0]. Defaults to 0.5.

        Returns:
            Dictionary with:
                - triggered (bool): True if rate exceeds threshold
                - hallucination_rate (float): The actual hallucination rate
                - threshold (float): The configured alert threshold
                - message (str): Descriptive message about the alert status

        Requirement 6.12: Alert when hallucination rate exceeds threshold
        """
        triggered = result.hallucination_rate > alert_threshold

        if triggered:
            message = (
                f"ALERT: Hallucination rate {result.hallucination_rate:.2%} "
                f"exceeds threshold {alert_threshold:.2%}"
            )
        else:
            message = (
                f"Hallucination rate {result.hallucination_rate:.2%} "
                f"is within threshold {alert_threshold:.2%}"
            )

        return {
            "triggered": triggered,
            "hallucination_rate": result.hallucination_rate,
            "threshold": alert_threshold,
            "message": message,
        }

    @staticmethod
    def _compute_group_stats(
        results: list[HallucinationResult],
    ) -> dict:
        """
        Compute aggregate statistics for a group of results.

        Args:
            results: HallucinationResult objects in this group

        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {
                "count": 0,
                "mean_rate": 0.0,
                "median_rate": 0.0,
                "std_rate": 0.0,
                "min_rate": 0.0,
                "max_rate": 0.0,
                "mean_grounding_score": 0.0,
                "total_claims": 0,
                "total_factual_claims": 0,
                "total_supported_claims": 0,
                "total_unsupported_claims": 0,
                "total_flagged_spans": 0,
            }

        rates = [r.hallucination_rate for r in results]
        grounding_scores = [r.overall_grounding_score for r in results]

        mean_rate = statistics.mean(rates)
        median_rate = statistics.median(rates)
        std_rate = statistics.stdev(rates) if len(rates) >= 2 else 0.0

        return {
            "count": len(results),
            "mean_rate": mean_rate,
            "median_rate": median_rate,
            "std_rate": std_rate,
            "min_rate": min(rates),
            "max_rate": max(rates),
            "mean_grounding_score": statistics.mean(grounding_scores),
            "total_claims": sum(r.total_claims for r in results),
            "total_factual_claims": sum(r.factual_claims for r in results),
            "total_supported_claims": sum(
                r.supported_claims for r in results
            ),
            "total_unsupported_claims": sum(
                r.unsupported_claims for r in results
            ),
            "total_flagged_spans": sum(
                len(r.flagged_spans) for r in results
            ),
        }

    def _calculate_proportion_test_p_value(
        self,
        baseline_result: HallucinationResult,
        finetuned_result: HallucinationResult,
    ) -> float:
        """
        Calculate p-value using a two-proportion z-test.

        Tests whether the difference in hallucination rates between the
        baseline and fine-tuned models is statistically significant.

        Uses the pooled proportion approach:
            p_pool = (x1 + x2) / (n1 + n2)
            SE = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z = (p1 - p2) / SE
            p_value = 2 * (1 - Phi(|z|))  (two-tailed)

        Args:
            baseline_result: HallucinationResult from baseline model
            finetuned_result: HallucinationResult from fine-tuned model

        Returns:
            p-value in [0.0, 1.0]
        """
        n1 = baseline_result.factual_claims
        n2 = finetuned_result.factual_claims

        # If either model has no factual claims, we cannot compute significance
        if n1 == 0 or n2 == 0:
            return 1.0

        x1 = baseline_result.unsupported_claims
        x2 = finetuned_result.unsupported_claims

        p1 = x1 / n1
        p2 = x2 / n2

        # Pooled proportion
        p_pool = (x1 + x2) / (n1 + n2)

        # If pooled proportion is 0 or 1, rates are identical at boundary
        if p_pool == 0.0 or p_pool == 1.0:
            return 1.0

        # Standard error
        se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))

        if se == 0.0:
            return 1.0

        # Z-statistic
        z = (p1 - p2) / se

        # Two-tailed p-value using normal CDF approximation
        p_value = 2.0 * (1.0 - self._normal_cdf(abs(z)))

        # Clamp to [0, 1]
        return max(0.0, min(1.0, p_value))

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """
        Approximate the standard normal CDF using the error function.

        Uses the relationship: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))

        Args:
            x: The z-value

        Returns:
            Cumulative probability
        """
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
