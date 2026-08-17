"""
Trust Scoring Engine V2 for TrustOps Enterprise Framework.

This module implements the multi-dimensional trust scoring system with
five dimensions: accuracy, consistency, safety, bias, and context grounding.

The engine combines individual dimension scores using configurable weights
to produce an overall trust score in the range [0, 1].

Requirements: 5.1, 5.2, 5.3, 5.4, 5.9
"""

from typing import Optional

from src.aws_clients.semantic_similarity_analyzer import SemanticSimilarityAnalyzer
from src.clients.inference_client import InferenceClient
from src.data_models.hallucination import HallucinationResult
from src.data_models.trust_score import (
    DimensionScore,
    TrustDimension,
    TrustScoreConfig,
    TrustScoreResult,
    TrustScoreWeights,
)
from src.hallucination.hallucination_detector import HallucinationDetector
from src.trust_scoring.scorers.accuracy_scorer import AccuracyScorer
from src.trust_scoring.scorers.bias_scorer import BiasScorer
from src.trust_scoring.scorers.consistency_scorer import ConsistencyScorer
from src.trust_scoring.scorers.context_grounding_scorer import ContextGroundingScorer
from src.trust_scoring.scorers.safety_scorer import SafetyScorer


class TrustScoringEngine:
    """
    Multi-dimensional trust scoring engine.

    This engine calculates trust scores across five dimensions:
    1. Accuracy: Comparison against expected answers
    2. Consistency: Response variance across multiple invocations
    3. Safety: Harmful content, toxicity, and policy violations
    4. Bias: Demographic bias and stereotyping
    5. Context Grounding: Semantic similarity to source documents

    The overall trust score is a weighted combination of all dimensions,
    with configurable weights that must sum to 1.0.

    Requirements:
    - 5.1: Calculate trust scores across five dimensions
    - 5.2: Return overall Trust_Score in [0, 1] as weighted combination
    - 5.3: Apply configurable weights to each dimension
    - 5.4: Support real-time and batch evaluation
    - 5.9: Persist all trust score calculations
    """

    def __init__(
        self,
        inference_client: Optional[InferenceClient] = None,
        similarity_analyzer: Optional[SemanticSimilarityAnalyzer] = None,
        config: Optional[TrustScoreConfig] = None,
        hallucination_detector: Optional[HallucinationDetector] = None,
    ):
        """
        Initialize the trust scoring engine.

        Args:
            inference_client: Client for model invocations (required for consistency scoring)
            similarity_analyzer: Semantic similarity analyzer for accuracy and grounding
            config: Trust score configuration with weights and thresholds
            hallucination_detector: Optional hallucination detector for enhanced
                context grounding scoring (Requirement 6.11)
        """
        self.inference_client = inference_client
        self.similarity_analyzer = similarity_analyzer or SemanticSimilarityAnalyzer()
        self.config = config or TrustScoreConfig()
        self.hallucination_detector = hallucination_detector

        # Validate weights sum to 1.0
        if not self.config.weights.validate_weights():
            weight_sum = sum([
                self.config.weights.accuracy,
                self.config.weights.consistency,
                self.config.weights.safety,
                self.config.weights.bias,
                self.config.weights.context_grounding,
            ])
            raise ValueError(
                "Trust score weights must sum to 1.0. "
                f"Current sum: {weight_sum}"
            )

        # Initialize dimension scorers
        self.accuracy_scorer = AccuracyScorer(
            similarity_analyzer=self.similarity_analyzer
        )
        self.consistency_scorer = ConsistencyScorer(
            inference_client=self.inference_client,
            similarity_analyzer=self.similarity_analyzer,
            num_samples=self.config.consistency_samples,
        ) if self.inference_client else None
        self.safety_scorer = SafetyScorer()
        self.bias_scorer = BiasScorer()
        self.context_grounding_scorer = ContextGroundingScorer(
            similarity_analyzer=self.similarity_analyzer
        )

    async def score_response(
        self,
        prompt: str,
        response: str,
        expected_response: Optional[str] = None,
        source_documents: Optional[list[str]] = None,
        model_id: Optional[str] = None,
        config: Optional[TrustScoreConfig] = None,
        hallucination_result: Optional[HallucinationResult] = None,
    ) -> TrustScoreResult:
        """
        Calculate trust score for a single response.

        This method scores the response across all five dimensions and
        combines them using configured weights to produce an overall score.

        Args:
            prompt: The original prompt/query
            response: The model's response text
            expected_response: Expected answer for accuracy scoring (optional)
            source_documents: Context documents for grounding check (optional)
            model_id: Model ID for consistency scoring (optional)
            config: Override default configuration (optional)
            hallucination_result: Pre-computed hallucination detection result
                for enhanced context grounding scoring (optional). If not
                provided but a hallucination_detector is configured, the
                detector will be called automatically when source_documents
                are available.

        Returns:
            TrustScoreResult with overall score and dimension breakdown

        Requirements:
        - 5.1: Calculate trust scores across five dimensions
        - 5.2: Return overall Trust_Score in [0, 1] as weighted combination
        - 5.3: Apply configurable weights to each dimension
        - 5.4: Support real-time evaluation (single inference)
        - 6.11: Integrate hallucination score into context_grounding dimension
        """
        # Use provided config or default
        scoring_config = config or self.config

        # Calculate dimension scores
        dimension_scores = {}

        # 1. Accuracy (if expected response provided)
        accuracy_score = await self.accuracy_scorer.calculate_accuracy(
            response=response,
            expected_response=expected_response,
        )
        dimension_scores[TrustDimension.ACCURACY] = accuracy_score

        # 2. Consistency (if model_id and inference_client provided)
        if model_id and self.consistency_scorer:
            consistency_score = await self.consistency_scorer.calculate_consistency(
                prompt=prompt,
                model_id=model_id,
                num_samples=scoring_config.consistency_samples,
            )
        else:
            # If consistency scoring not available, use neutral score
            consistency_score = DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.5,
                confidence=0.0,
                details={"message": "Consistency scoring not available"},
                checks_passed=[],
                checks_failed=["not_available"],
            )
        dimension_scores[TrustDimension.CONSISTENCY] = consistency_score

        # 3. Safety
        safety_score = await self.safety_scorer.calculate_safety(
            response=response
        )
        dimension_scores[TrustDimension.SAFETY] = safety_score

        # 4. Bias
        bias_score = await self.bias_scorer.calculate_bias(
            response=response
        )
        dimension_scores[TrustDimension.BIAS] = bias_score

        # 5. Context Grounding (if source documents provided)
        # Optionally enhanced with hallucination detection (Requirement 6.11)
        effective_hallucination_result = hallucination_result
        if (
            effective_hallucination_result is None
            and self.hallucination_detector is not None
            and source_documents
        ):
            effective_hallucination_result = self.hallucination_detector.detect(
                response=response,
                source_documents=source_documents,
            )

        grounding_score = await self.context_grounding_scorer.calculate_context_grounding(
            response=response,
            source_documents=source_documents or [],
            hallucination_result=effective_hallucination_result,
        )
        dimension_scores[TrustDimension.CONTEXT_GROUNDING] = grounding_score

        # Combine scores using configured weights
        overall_score = self.combine_scores(
            dimension_scores=dimension_scores,
            weights=scoring_config.weights,
        )

        # Calculate overall confidence as weighted average of dimension confidences
        overall_confidence = self._calculate_overall_confidence(
            dimension_scores=dimension_scores,
            weights=scoring_config.weights,
        )

        # Flag for review if below threshold
        flagged_for_review = overall_score < scoring_config.review_threshold

        # Generate explanation
        explanation = self._generate_explanation(
            dimension_scores=dimension_scores,
            overall_score=overall_score,
            flagged_for_review=flagged_for_review,
        )

        # Build component details
        component_details = [
            {
                "dimension": dim.value,
                "score": score.score,
                "confidence": score.confidence,
                "checks_passed": score.checks_passed,
                "checks_failed": score.checks_failed,
            }
            for dim, score in dimension_scores.items()
        ]

        return TrustScoreResult(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            confidence_level=overall_confidence,
            flagged_for_review=flagged_for_review,
            explanation=explanation,
            component_details=component_details,
        )

    async def score_batch(
        self,
        items: list[dict],
        config: Optional[TrustScoreConfig] = None,
    ) -> list[TrustScoreResult]:
        """
        Calculate trust scores for multiple responses.

        This method processes a batch of scoring items and returns
        trust scores for each one.

        Args:
            items: List of scoring items, each containing:
                - prompt: str
                - response: str
                - expected_response: Optional[str]
                - source_documents: Optional[list[str]]
                - model_id: Optional[str]
            config: Override default configuration (optional)

        Returns:
            List of TrustScoreResult objects

        Requirement 5.4: Support batch evaluation (full dataset)
        """
        results = []

        for item in items:
            result = await self.score_response(
                prompt=item.get("prompt", ""),
                response=item.get("response", ""),
                expected_response=item.get("expected_response"),
                source_documents=item.get("source_documents"),
                model_id=item.get("model_id"),
                config=config,
                hallucination_result=item.get("hallucination_result"),
            )
            results.append(result)

        return results

    def combine_scores(
        self,
        dimension_scores: dict[TrustDimension, DimensionScore],
        weights: TrustScoreWeights,
    ) -> float:
        """
        Combine dimension scores using configured weights.

        This method implements the weighted combination formula:
        overall_score = Σ(dimension_score × weight) for all dimensions

        The weights must sum to 1.0 to ensure the overall score is in [0, 1].

        Args:
            dimension_scores: Scores for each trust dimension
            weights: Weights for each dimension

        Returns:
            Overall trust score in [0, 1] range

        Requirements:
        - 5.2: Return overall Trust_Score in [0, 1] as weighted combination
        - 5.3: Apply configurable weights to each dimension
        """
        # Validate weights sum to 1.0
        if not weights.validate_weights():
            raise ValueError(
                "Weights must sum to 1.0 for proper score combination"
            )

        # Calculate weighted sum
        overall_score = (
            dimension_scores[TrustDimension.ACCURACY].score * weights.accuracy +
            dimension_scores[TrustDimension.CONSISTENCY].score * weights.consistency +
            dimension_scores[TrustDimension.SAFETY].score * weights.safety +
            dimension_scores[TrustDimension.BIAS].score * weights.bias +
            dimension_scores[TrustDimension.CONTEXT_GROUNDING].score * weights.context_grounding
        )

        # Ensure score is in valid range [0, 1]
        overall_score = max(0.0, min(1.0, overall_score))

        return overall_score

    def _calculate_overall_confidence(
        self,
        dimension_scores: dict[TrustDimension, DimensionScore],
        weights: TrustScoreWeights,
    ) -> float:
        """
        Calculate overall confidence as weighted average of dimension confidences.

        Args:
            dimension_scores: Scores for each trust dimension
            weights: Weights for each dimension

        Returns:
            Overall confidence in [0, 1] range
        """
        overall_confidence = (
            dimension_scores[TrustDimension.ACCURACY].confidence * weights.accuracy +
            dimension_scores[TrustDimension.CONSISTENCY].confidence * weights.consistency +
            dimension_scores[TrustDimension.SAFETY].confidence * weights.safety +
            dimension_scores[TrustDimension.BIAS].confidence * weights.bias +
            dimension_scores[TrustDimension.CONTEXT_GROUNDING].confidence * weights.context_grounding
        )

        # Ensure confidence is in valid range [0, 1]
        overall_confidence = max(0.0, min(1.0, overall_confidence))

        return overall_confidence

    def _generate_explanation(
        self,
        dimension_scores: dict[TrustDimension, DimensionScore],
        overall_score: float,
        flagged_for_review: bool,
    ) -> str:
        """
        Generate human-readable explanation of trust score.

        This method identifies weak dimensions and provides actionable
        feedback about the trust score.

        Args:
            dimension_scores: Scores for each trust dimension
            overall_score: Overall trust score
            flagged_for_review: Whether response is flagged for review

        Returns:
            Human-readable explanation string

        Requirement 5.5: Provide detailed explanations for each trust score component
        """
        # Build explanation parts
        parts = [f"Overall trust score: {overall_score:.2f}"]

        # Identify weak dimensions (score < 0.6)
        weak_dimensions = []
        for dim, score in dimension_scores.items():
            if score.score < 0.6:
                weak_dimensions.append(f"{dim.value} ({score.score:.2f})")

        if weak_dimensions:
            parts.append(f"Weak dimensions: {', '.join(weak_dimensions)}")

        # Add review flag status
        if flagged_for_review:
            parts.append("FLAGGED FOR REVIEW")

        # Add dimension-specific details
        dimension_details = []
        for dim, score in dimension_scores.items():
            if score.checks_failed:
                failed_checks = ", ".join(score.checks_failed)
                dimension_details.append(
                    f"{dim.value}: {failed_checks}"
                )

        if dimension_details:
            parts.append("Issues: " + "; ".join(dimension_details))

        return ". ".join(parts)

    async def calibrate(
        self,
        ground_truth_data: list[dict],
    ) -> dict:
        """
        Calibrate trust scores against human-labeled ground truth.

        This method compares the engine's calculated trust scores against
        human-labeled ground truth scores to validate scoring accuracy.
        It calculates correlation coefficients to measure how well the
        automated scores align with human judgment.

        Args:
            ground_truth_data: List of ground truth items, each containing:
                - prompt: str - The original prompt
                - response: str - The model response
                - human_score: float - Human-labeled trust score [0, 1]
                - expected_response: Optional[str] - Expected answer
                - source_documents: Optional[list[str]] - Context documents
                - model_id: Optional[str] - Model ID for consistency scoring

        Returns:
            CalibrationResult dict containing:
                - pearson_correlation: float - Pearson correlation coefficient
                - spearman_correlation: float - Spearman rank correlation
                - mean_absolute_error: float - Average absolute difference
                - rmse: float - Root mean squared error
                - sample_size: int - Number of samples used
                - score_pairs: list[tuple[float, float]] - (predicted, actual) pairs

        Requirement 5.8: Calibrate trust scores against human-labeled ground truth
        """
        import numpy as np
        from scipy import stats

        if not ground_truth_data:
            raise ValueError("Ground truth data cannot be empty")

        # Calculate trust scores for all ground truth items
        predicted_scores = []
        actual_scores = []

        for item in ground_truth_data:
            # Score the response
            result = await self.score_response(
                prompt=item.get("prompt", ""),
                response=item.get("response", ""),
                expected_response=item.get("expected_response"),
                source_documents=item.get("source_documents"),
                model_id=item.get("model_id"),
            )

            predicted_scores.append(result.overall_score)
            actual_scores.append(item.get("human_score", 0.0))

        # Convert to numpy arrays for calculations
        predicted = np.array(predicted_scores)
        actual = np.array(actual_scores)

        # Calculate Pearson correlation coefficient
        pearson_corr, pearson_pvalue = stats.pearsonr(predicted, actual)

        # Calculate Spearman rank correlation
        spearman_corr, spearman_pvalue = stats.spearmanr(predicted, actual)

        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(predicted - actual))

        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))

        # Create score pairs for detailed analysis
        score_pairs = list(zip(predicted_scores, actual_scores))

        # Generate calibration report
        calibration_result = {
            "pearson_correlation": float(pearson_corr),
            "pearson_pvalue": float(pearson_pvalue),
            "spearman_correlation": float(spearman_corr),
            "spearman_pvalue": float(spearman_pvalue),
            "mean_absolute_error": float(mae),
            "rmse": float(rmse),
            "sample_size": len(ground_truth_data),
            "score_pairs": score_pairs,
            "summary": self._generate_calibration_summary(
                pearson_corr=pearson_corr,
                spearman_corr=spearman_corr,
                mae=mae,
                sample_size=len(ground_truth_data),
            ),
        }

        return calibration_result

    def _generate_calibration_summary(
        self,
        pearson_corr: float,
        spearman_corr: float,
        mae: float,
        sample_size: int,
    ) -> str:
        """
        Generate human-readable calibration summary.

        Args:
            pearson_corr: Pearson correlation coefficient
            spearman_corr: Spearman rank correlation
            mae: Mean absolute error
            sample_size: Number of samples

        Returns:
            Summary string describing calibration quality
        """
        # Interpret correlation strength
        if abs(pearson_corr) >= 0.8:
            correlation_strength = "strong"
        elif abs(pearson_corr) >= 0.5:
            correlation_strength = "moderate"
        else:
            correlation_strength = "weak"

        # Interpret MAE
        if mae < 0.1:
            error_level = "excellent"
        elif mae < 0.2:
            error_level = "good"
        elif mae < 0.3:
            error_level = "acceptable"
        else:
            error_level = "poor"

        summary = (
            f"Calibration based on {sample_size} samples shows {correlation_strength} "
            f"correlation (Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}) "
            f"with {error_level} accuracy (MAE: {mae:.3f})."
        )

        # Add recommendations
        if abs(pearson_corr) < 0.5 or mae > 0.3:
            summary += " Consider adjusting dimension weights or reviewing scoring criteria."

        return summary

    def calculate_confidence_interval(
        self,
        scores: list[float],
        confidence_level: float = 0.95,
        method: str = "auto",
    ) -> dict:
        """
        Calculate confidence interval for aggregate trust scores.

        This method computes confidence intervals using either bootstrap
        resampling (for small samples <30) or normal approximation (for
        large samples >=30). The method can be automatically selected based
        on sample size or manually specified.

        Args:
            scores: List of trust scores [0, 1]
            confidence_level: Confidence level (default 0.95 for 95% CI)
            method: Method to use - "auto", "bootstrap", or "normal"

        Returns:
            Dict containing:
                - mean: float - Mean of scores
                - lower_bound: float - Lower confidence bound
                - upper_bound: float - Upper confidence bound
                - confidence_level: float - Confidence level used
                - method: str - Method used for calculation
                - sample_size: int - Number of scores

        Requirement 5.10: Calculate confidence intervals for aggregate trust scores
        """
        import numpy as np
        from scipy import stats

        if not scores:
            raise ValueError("Scores list cannot be empty")

        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")

        # Convert to numpy array
        scores_array = np.array(scores)
        sample_size = len(scores)
        mean_score = float(np.mean(scores_array))

        # Auto-select method based on sample size
        if method == "auto":
            method = "bootstrap" if sample_size < 30 else "normal"

        # Calculate confidence interval based on method
        if method == "bootstrap":
            # Bootstrap resampling for small samples
            n_bootstrap = 10000
            bootstrap_means = []

            rng = np.random.default_rng(seed=42)  # For reproducibility
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_sample = rng.choice(scores_array, size=sample_size, replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))

            # Calculate percentiles for confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = float(np.percentile(bootstrap_means, lower_percentile))
            upper_bound = float(np.percentile(bootstrap_means, upper_percentile))

        elif method == "normal":
            # Normal approximation for large samples
            std_error = stats.sem(scores_array)  # Standard error of the mean
            
            # Calculate critical value for confidence level
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha / 2)

            # Calculate confidence interval
            margin_of_error = z_critical * std_error
            lower_bound = float(mean_score - margin_of_error)
            upper_bound = float(mean_score + margin_of_error)

        else:
            raise ValueError(f"Invalid method: {method}. Must be 'auto', 'bootstrap', or 'normal'")

        # Ensure bounds are within valid score range [0, 1]
        lower_bound = max(0.0, min(1.0, lower_bound))
        upper_bound = max(0.0, min(1.0, upper_bound))

        return {
            "mean": mean_score,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": confidence_level,
            "method": method,
            "sample_size": sample_size,
        }


