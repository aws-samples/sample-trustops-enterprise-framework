"""
Evidence Searcher for Hallucination Detection.

This module implements evidence search by computing embeddings for claims
and source documents, then using cosine similarity to find supporting
evidence. It supports a configurable embedding model and returns top-k
matches with scores.

Requirements: 6.2, 6.13
"""

import math
import hashlib
from typing import Optional

from src.data_models.hallucination import (
    Claim,
    EvidenceSource,
    HallucinationConfig,
)


class EvidenceSearcher:
    """
    Search source documents for evidence supporting claims.

    The evidence searcher computes embeddings for claims and source documents,
    then uses cosine similarity to identify the most relevant evidence spans.
    It returns top-k matches with similarity scores for each claim.

    When no external embedding provider is available, a lightweight
    deterministic bag-of-words embedding is used so the component can
    operate entirely offline (useful for testing and local evaluation).

    Requirement 6.2: Search source documents for supporting evidence using
    embedding cosine similarity.
    Requirement 6.13: Use a configurable embedding model.
    """

    def __init__(
        self,
        config: Optional[HallucinationConfig] = None,
        embedding_fn=None,
        top_k: int = 3,
    ):
        """
        Initialize the evidence searcher.

        Args:
            config: Hallucination detection configuration (optional).
            embedding_fn: Optional callable(text: str) -> list[float] that
                computes an embedding vector. When *None* a built-in
                bag-of-words fallback is used.
            top_k: Number of top matching evidence sources to return per
                claim.
        """
        self.config = config or HallucinationConfig()
        self._embedding_fn = embedding_fn
        self.top_k = max(1, top_k)
        # Cache embeddings keyed by a content hash to avoid recomputation
        self._embedding_cache: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_evidence(
        self,
        claim: Claim,
        source_documents: list[str],
    ) -> list[EvidenceSource]:
        """
        Search source documents for evidence supporting a single claim.

        Computes the embedding for the claim text, then computes cosine
        similarity against each source document embedding. Returns the
        top-k documents sorted by descending similarity score.

        Args:
            claim: The claim to find evidence for.
            source_documents: List of source document texts.

        Returns:
            List of EvidenceSource objects (up to top_k), sorted by
            descending similarity score.

        Requirement 6.2: Compute embedding cosine similarity against source
        documents and return top-k matches with scores.
        """
        if not source_documents:
            return []

        claim_embedding = self._compute_embedding(claim.text)

        scored: list[tuple[int, float]] = []
        for idx, doc in enumerate(source_documents):
            doc_embedding = self._compute_embedding(doc)
            score = self._cosine_similarity(claim_embedding, doc_embedding)
            scored.append((idx, score))

        # Sort by similarity descending
        scored.sort(key=lambda t: t[1], reverse=True)

        # Build EvidenceSource objects for top-k
        results: list[EvidenceSource] = []
        for idx, score in scored[: self.top_k]:
            doc_text = source_documents[idx]
            # Use a truncated snippet as the span (first 200 chars)
            span = doc_text[:200] if len(doc_text) > 200 else doc_text
            results.append(
                EvidenceSource(
                    document_id=f"doc_{idx}",
                    span=span,
                    similarity_score=round(max(0.0, min(1.0, score)), 6),
                )
            )

        return results

    def search_evidence_batch(
        self,
        claims: list[Claim],
        source_documents: list[str],
    ) -> dict[str, list[EvidenceSource]]:
        """
        Search evidence for multiple claims against the same source documents.

        Args:
            claims: List of claims to search evidence for.
            source_documents: List of source document texts.

        Returns:
            Dictionary mapping claim text to list of EvidenceSource objects.
        """
        evidence_map: dict[str, list[EvidenceSource]] = {}
        for claim in claims:
            evidence_map[claim.text] = self.search_evidence(
                claim, source_documents
            )
        return evidence_map

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _compute_embedding(self, text: str) -> list[float]:
        """
        Compute an embedding vector for the given text.

        Uses the injected embedding_fn if available, otherwise falls back
        to a deterministic bag-of-words approach.

        Results are cached by content hash to avoid redundant computation.
        """
        # In-memory cache key only - not a security or integrity control,
        # so a fast non-cryptographic digest is appropriate here.
        cache_key = hashlib.md5(
            text.encode("utf-8"), usedforsecurity=False
        ).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        if self._embedding_fn is not None:
            embedding = self._embedding_fn(text)
        else:
            embedding = self._bow_embedding(text)

        self._embedding_cache[cache_key] = embedding
        return embedding

    @staticmethod
    def _bow_embedding(text: str, dim: int = 128) -> list[float]:
        """
        Deterministic bag-of-words embedding fallback.

        Hashes each word into a fixed-dimension vector and normalises the
        result. This is *not* a semantic embedding but provides a
        reasonable baseline for testing and offline operation.
        """
        vec = [0.0] * dim
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            bucket = h % dim
            vec[bucket] += 1.0

        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Returns a value in [-1, 1]; callers clamp to [0, 1] as needed.
        """
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)
