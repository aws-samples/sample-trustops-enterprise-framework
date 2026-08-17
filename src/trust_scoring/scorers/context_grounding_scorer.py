"""
Context Grounding Scorer for Trust Scoring Engine.

This module implements the context grounding dimension of trust scoring
by calculating semantic similarity between model responses and provided
source documents.

The context grounding scorer evaluates whether a model's response is
properly grounded in the provided context/source documents using:
- Embedding-based semantic similarity
- Multiple grounding metrics (entailment, factual consistency)
- Citation accuracy when explicit citations are present

Requirements: 5.1, 5.7
"""

from typing import Optional

from src.aws_clients.semantic_similarity_analyzer import (
    SemanticSimilarityAnalyzer
)
from src.data_models.hallucination import HallucinationResult
from src.data_models.trust_score import DimensionScore, TrustDimension


class ContextGroundingScorer:
    """
    Calculate context grounding scores using semantic similarity.

    The context grounding scorer evaluates how well a model's response
    is grounded in the provided source documents. This is done by:
    1. Computing embeddings for the response and source documents
    2. Calculating cosine similarity between response and each document
    3. Using the maximum similarity as the grounding score
    4. Optionally checking for explicit citations

    A score of 1.0 indicates perfect grounding (response fully
    supported by sources).
    A score of 0.0 indicates no grounding (response unrelated to
    sources).

    Requirement 5.7: Calculate semantic similarity between response and
    source documents using embedding cosine similarity via Bedrock
    embedding models
    """

    def __init__(
        self,
        similarity_analyzer: Optional[SemanticSimilarityAnalyzer] = None,
        high_grounding_threshold: float = 0.8,
        low_grounding_threshold: float = 0.5,
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        hallucination_weight: float = 0.4,
    ):
        """
        Initialize the context grounding scorer.

        Args:
            similarity_analyzer: Semantic similarity analyzer for
                embedding-based comparison
            high_grounding_threshold: Threshold for high grounding
                (default: 0.8)
            low_grounding_threshold: Threshold for low grounding
                (default: 0.5)
            embedding_model_id: Bedrock embedding model to use
                (default: amazon.titan-embed-text-v2:0)
            hallucination_weight: Weight for hallucination score when
                blending with semantic similarity (default: 0.4).
                Must be in [0.0, 1.0]. The semantic similarity score
                receives weight (1 - hallucination_weight).
        """
        if not 0.0 <= hallucination_weight <= 1.0:
            raise ValueError(
                "hallucination_weight must be between 0.0 and 1.0"
            )
        self.similarity_analyzer = (
            similarity_analyzer or
            SemanticSimilarityAnalyzer(
                embedding_model_id=embedding_model_id
            )
        )
        self.high_grounding_threshold = high_grounding_threshold
        self.low_grounding_threshold = low_grounding_threshold
        self.embedding_model_id = embedding_model_id
        self.hallucination_weight = hallucination_weight

    async def calculate_context_grounding(
        self,
        response: str,
        source_documents: Optional[list[str]] = None,
        hallucination_result: Optional[HallucinationResult] = None,
    ) -> DimensionScore:
        """
        Calculate context grounding score for a response.

        This method evaluates how well the response is grounded in the
        provided source documents by:
        1. Computing semantic similarity between response and each
           source document
        2. Using the maximum similarity as the grounding score
        3. Optionally blending in hallucination detection results
        4. Providing detailed analysis of grounding evidence

        When a HallucinationResult is provided, the final grounding score
        is a weighted blend of the semantic similarity score and the
        hallucination-derived grounding score (1 - hallucination_rate).
        The blend uses hallucination_weight for the hallucination component
        and (1 - hallucination_weight) for the similarity component.

        Args:
            response: The model's response text
            source_documents: List of source documents that should
                ground the response (optional)
            hallucination_result: Result from HallucinationDetector
                for enhanced grounding scoring (optional)

        Returns:
            DimensionScore with context grounding score in [0, 1] range

        Requirements:
        - 5.7: Calculate semantic similarity between response and
          source documents
        - 6.11: Integrate hallucination score as weighted component
        """
        # If no source documents provided, return neutral score
        if not source_documents or len(source_documents) == 0:
            return DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.5,
                confidence=0.3,
                details={
                    "method": "no_source_documents",
                    "message": "No source documents provided for grounding"
                },
                checks_passed=[],
                checks_failed=["no_source_documents"]
            )

        # If response is empty, return zero score
        if not response or not response.strip():
            return DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.0,
                confidence=1.0,
                details={
                    "method": "empty_response",
                    "message": "Response is empty"
                },
                checks_passed=[],
                checks_failed=["empty_response"]
            )

        # Filter out empty source documents
        valid_sources = [
            doc for doc in source_documents
            if doc and doc.strip()
        ]

        if not valid_sources:
            return DimensionScore(
                dimension=TrustDimension.CONTEXT_GROUNDING,
                score=0.5,
                confidence=0.3,
                details={
                    "method": "no_valid_sources",
                    "message": "All source documents are empty"
                },
                checks_passed=[],
                checks_failed=["no_valid_sources"]
            )

        # Calculate similarity with each source document
        similarities = await self._calculate_document_similarities(
            response, valid_sources
        )

        # Use maximum similarity as grounding score
        # This represents the best match with any source document
        max_similarity = max(similarities) if similarities else 0.0
        avg_similarity = (
            sum(similarities) / len(similarities) if similarities else 0.0
        )

        # Ensure score is in valid range
        grounding_score = max(0.0, min(1.0, max_similarity))

        # Blend in hallucination result if available (Req 6.11)
        hallucination_blended = False
        hallucination_grounding = None
        if hallucination_result is not None:
            # Derive grounding from hallucination rate
            hallucination_grounding = (
                1.0 - hallucination_result.hallucination_rate
            )
            # Weighted blend
            hw = self.hallucination_weight
            grounding_score = (
                grounding_score * (1.0 - hw)
                + hallucination_grounding * hw
            )
            grounding_score = max(0.0, min(1.0, grounding_score))
            hallucination_blended = True

        # Determine which checks passed
        checks_passed = []
        checks_failed = []

        if grounding_score >= self.high_grounding_threshold:
            checks_passed.append("high_grounding")
        elif grounding_score < self.low_grounding_threshold:
            checks_failed.append("low_grounding")
        else:
            checks_passed.append("moderate_grounding")

        # Check if response has explicit citations
        has_citations = self._check_for_citations(response)
        if has_citations:
            checks_passed.append("explicit_citations")
        else:
            checks_failed.append("no_explicit_citations")

        # Calculate confidence based on consistency of similarities
        # High confidence when similarities are consistent
        if len(similarities) > 1:
            variance = sum(
                (s - avg_similarity) ** 2 for s in similarities
            ) / len(similarities)
            confidence = max(0.5, 1.0 - variance)
        else:
            confidence = 0.7

        # Find best matching document
        best_match_idx = (
            similarities.index(max_similarity) if similarities else 0
        )

        return DimensionScore(
            dimension=TrustDimension.CONTEXT_GROUNDING,
            score=grounding_score,
            confidence=confidence,
            details={
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "min_similarity": min(similarities) if similarities else 0.0,
                "num_sources": len(valid_sources),
                "similarities": similarities,
                "best_match_index": best_match_idx,
                "best_match_preview": (
                    valid_sources[best_match_idx][:200]
                    if valid_sources else ""
                ),
                "has_explicit_citations": has_citations,
                "response_length": len(response),
                "embedding_model": self.embedding_model_id,
                "hallucination_blended": hallucination_blended,
                "hallucination_grounding_score": hallucination_grounding,
                "hallucination_weight": (
                    self.hallucination_weight if hallucination_blended
                    else None
                ),
            },
            checks_passed=checks_passed,
            checks_failed=checks_failed
        )

    async def _calculate_document_similarities(
        self,
        response: str,
        source_documents: list[str]
    ) -> list[float]:
        """
        Calculate semantic similarity between response and each source.

        This method computes the embedding-based cosine similarity
        between the response and each source document.

        Args:
            response: Model response text
            source_documents: List of source document texts

        Returns:
            List of similarity scores in [0, 1] range
        """
        similarities = []

        for doc in source_documents:
            try:
                # Calculate semantic similarity using embeddings
                similarity = self.similarity_analyzer.calculate_similarity(
                    response, doc
                )

                # Ensure score is in valid range
                similarity = max(0.0, min(1.0, similarity))
                similarities.append(similarity)

            except Exception:
                # If similarity calculation fails for a document,
                # use neutral score
                similarities.append(0.5)

        return similarities

    def _check_for_citations(self, response: str) -> bool:
        """
        Check if response contains explicit citations.

        This method looks for common citation patterns:
        - [1], [2], etc.
        - (Source: ...)
        - According to ...
        - As stated in ...

        Args:
            response: Model response text

        Returns:
            True if citations detected, False otherwise
        """
        import re

        # Citation patterns
        patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d+\)',  # (1), (2), etc.
            r'\(Source:',  # (Source: ...)
            r'\(Ref:',  # (Ref: ...)
            r'According to',  # According to ...
            r'As stated in',  # As stated in ...
            r'As mentioned in',  # As mentioned in ...
            r'Based on',  # Based on ...
            r'Citing',  # Citing ...
        ]

        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True

        return False

    async def calculate_batch_grounding(
        self,
        items: list[tuple[str, list[str]]]
    ) -> list[DimensionScore]:
        """
        Calculate context grounding scores for multiple responses.

        This method provides batch scoring for efficiency when
        evaluating multiple responses.

        Args:
            items: List of (response, source_documents) tuples

        Returns:
            List of DimensionScore objects
        """
        scores = []

        for response, source_documents in items:
            score = await self.calculate_context_grounding(
                response, source_documents
            )
            scores.append(score)

        return scores
