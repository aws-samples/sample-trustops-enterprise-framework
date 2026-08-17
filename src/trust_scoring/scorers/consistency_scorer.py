"""
Consistency Scorer for Trust Scoring Engine.

This module implements the consistency dimension of trust scoring by
invoking a model multiple times with the same prompt and measuring
response variance using semantic similarity.

A high consistency score indicates the model produces reliable,
deterministic outputs. Low consistency may indicate randomness or
instability.

Requirements: 5.1, 5.4
"""

from typing import Optional

from src.aws_clients.semantic_similarity_analyzer import (
    SemanticSimilarityAnalyzer
)
from src.clients.inference_client import InferenceClient
from src.data_models.model import InferenceRequest
from src.data_models.trust_score import DimensionScore, TrustDimension


class ConsistencyScorer:
    """
    Calculate consistency scores by invoking model multiple times.

    The consistency scorer measures how consistent a model's responses
    are when given the same prompt multiple times. This is done by:
    1. Invoking the model N times with identical prompts
    2. Computing pairwise semantic similarity between all responses
    3. Averaging the similarities to get the consistency score

    A score of 1.0 indicates perfect consistency (identical responses).
    A score near 0.0 indicates high variance (very different responses).

    Requirement 5.4: Invoke model multiple times with same prompt,
    measure response variance
    """

    def __init__(
        self,
        inference_client: InferenceClient,
        similarity_analyzer: Optional[SemanticSimilarityAnalyzer] = None,
        num_samples: int = 3,
        high_consistency_threshold: float = 0.9,
        low_consistency_threshold: float = 0.6,
    ):
        """
        Initialize the consistency scorer.

        Args:
            inference_client: Client for invoking models
            similarity_analyzer: Semantic similarity analyzer for
                comparing responses
            num_samples: Number of times to invoke model (default: 3)
            high_consistency_threshold: Threshold for high consistency
                (default: 0.9)
            low_consistency_threshold: Threshold for low consistency
                (default: 0.6)
        """
        self.inference_client = inference_client
        self.similarity_analyzer = (
            similarity_analyzer or SemanticSimilarityAnalyzer()
        )
        self.num_samples = num_samples
        self.high_consistency_threshold = high_consistency_threshold
        self.low_consistency_threshold = low_consistency_threshold

    async def calculate_consistency(
        self,
        prompt: str,
        model_id: str,
        num_samples: Optional[int] = None,
        inference_params: Optional[dict] = None,
    ) -> DimensionScore:
        """
        Calculate consistency score by invoking model multiple times.

        This method:
        1. Invokes the model N times with the same prompt
        2. Computes pairwise semantic similarity between all responses
        3. Calculates the average pairwise similarity as consistency
        4. Returns a DimensionScore with the consistency score

        Args:
            prompt: The prompt to send to the model
            model_id: The model to evaluate for consistency
            num_samples: Number of invocations (uses default if None)
            inference_params: Optional inference parameters
                (temperature, max_tokens, etc.)

        Returns:
            DimensionScore with consistency score in [0, 1] range

        Requirement 5.4: Invoke model multiple times with same prompt,
        measure response variance using semantic similarity
        """
        # Use provided num_samples or default
        n = num_samples if num_samples is not None else self.num_samples

        # Validate num_samples
        if n < 2:
            return DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=1.0,
                confidence=0.0,
                details={
                    "error": "num_samples must be >= 2",
                    "num_samples": n,
                },
                checks_passed=[],
                checks_failed=["insufficient_samples"],
            )

        # Handle empty prompt
        if not prompt or not prompt.strip():
            return DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.0,
                confidence=1.0,
                details={
                    "error": "Empty prompt provided",
                },
                checks_passed=[],
                checks_failed=["empty_prompt"],
            )

        # Create inference request
        params = inference_params or {}
        request = InferenceRequest(
            prompt=prompt,
            max_tokens=params.get("max_tokens", 1024),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            stop_sequences=params.get("stop_sequences"),
        )

        # Invoke model multiple times
        try:
            responses = await self._invoke_multiple_times(
                model_id, request, n
            )
        except Exception as e:
            return DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=0.0,
                confidence=0.0,
                details={
                    "error": f"Failed to invoke model: {str(e)}",
                    "model_id": model_id,
                },
                checks_passed=[],
                checks_failed=["model_invocation_failed"],
            )

        # Extract response texts
        response_texts = [r.text for r in responses]

        # Handle case where all responses are empty
        if all(not text.strip() for text in response_texts):
            return DimensionScore(
                dimension=TrustDimension.CONSISTENCY,
                score=1.0,
                confidence=0.5,
                details={
                    "message": "All responses empty",
                    "num_samples": n,
                },
                checks_passed=["all_empty"],
                checks_failed=[],
            )

        # Calculate pairwise similarities
        similarities = self._calculate_pairwise_similarities(
            response_texts
        )

        # Calculate average similarity as consistency score
        if not similarities:
            consistency_score = 1.0  # Single response or all identical
        else:
            consistency_score = sum(similarities) / len(similarities)

        # Ensure score is in valid range
        consistency_score = max(0.0, min(1.0, consistency_score))

        # Determine checks passed/failed
        checks_passed = []
        checks_failed = []

        if consistency_score >= self.high_consistency_threshold:
            checks_passed.append("high_consistency")
        elif consistency_score < self.low_consistency_threshold:
            checks_failed.append("low_consistency")
        else:
            checks_passed.append("moderate_consistency")

        # Calculate confidence based on number of samples
        # More samples = higher confidence
        confidence = min(1.0, 0.5 + (n - 2) * 0.1)

        # Calculate variance for additional details
        if similarities:
            variance = sum(
                (s - consistency_score) ** 2 for s in similarities
            ) / len(similarities)
        else:
            variance = 0.0

        return DimensionScore(
            dimension=TrustDimension.CONSISTENCY,
            score=consistency_score,
            confidence=confidence,
            details={
                "num_samples": n,
                "pairwise_similarities": similarities,
                "mean_similarity": consistency_score,
                "variance": variance,
                "response_lengths": [len(text) for text in response_texts],
                "model_id": model_id,
            },
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    async def _invoke_multiple_times(
        self,
        model_id: str,
        request: InferenceRequest,
        num_samples: int,
    ) -> list:
        """
        Invoke the model multiple times with the same request.

        Args:
            model_id: The model to invoke
            request: The inference request
            num_samples: Number of times to invoke

        Returns:
            List of InferenceResponse objects
        """
        import asyncio

        # Create tasks for concurrent invocation
        tasks = [
            self.inference_client.invoke(model_id, request)
            for _ in range(num_samples)
        ]

        # Execute all invocations concurrently
        responses = await asyncio.gather(*tasks)

        return responses

    def _calculate_pairwise_similarities(
        self, texts: list[str]
    ) -> list[float]:
        """
        Calculate pairwise semantic similarities between all texts.

        This method computes the semantic similarity between every pair
        of texts and returns all similarity scores. The average of these
        scores represents the overall consistency.

        Args:
            texts: List of response texts to compare

        Returns:
            List of pairwise similarity scores in [0, 1] range
        """
        similarities = []

        # Compare each pair of texts
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                try:
                    similarity = self.similarity_analyzer.calculate_similarity(
                        texts[i], texts[j]
                    )
                    # Ensure similarity is in valid range
                    similarity = max(0.0, min(1.0, similarity))
                    similarities.append(similarity)
                except Exception:
                    # If similarity calculation fails, use neutral score
                    similarities.append(0.5)

        return similarities
