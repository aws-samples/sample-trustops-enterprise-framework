"""
Unit tests for EvidenceSearcher.

Tests evidence search functionality including embedding computation,
cosine similarity, and top-k matching against source documents.

Requirements: 6.2, 6.13
"""

import pytest

from src.data_models.hallucination import (
    Claim,
    HallucinationConfig,
)
from src.hallucination.evidence_searcher import EvidenceSearcher


class TestEvidenceSearcher:
    """Test suite for EvidenceSearcher class."""

    def _make_claim(self, text: str, start_idx: int = 0) -> Claim:
        """Helper to create a Claim object."""
        return Claim(
            text=text,
            start_idx=start_idx,
            end_idx=start_idx + len(text),
            is_factual=True,
            is_opinion=False,
            is_hedged=False,
        )

    def test_initialization_default(self):
        """Test default initialization."""
        searcher = EvidenceSearcher()
        assert searcher.config is not None
        assert searcher.top_k == 3

    def test_initialization_custom_top_k(self):
        """Test initialization with custom top_k."""
        searcher = EvidenceSearcher(top_k=5)
        assert searcher.top_k == 5

    def test_initialization_top_k_minimum_one(self):
        """Test that top_k is at least 1."""
        searcher = EvidenceSearcher(top_k=0)
        assert searcher.top_k == 1

    def test_search_evidence_empty_documents(self):
        """Test search with no source documents returns empty list."""
        searcher = EvidenceSearcher()
        claim = self._make_claim("The sky is blue.")

        results = searcher.search_evidence(claim, [])

        assert results == []

    def test_search_evidence_single_document(self):
        """Test search against a single source document."""
        searcher = EvidenceSearcher()
        claim = self._make_claim("The sky is blue.")
        docs = ["The sky is blue and clear today."]

        results = searcher.search_evidence(claim, docs)

        assert len(results) == 1
        assert results[0].document_id == "doc_0"
        assert 0.0 <= results[0].similarity_score <= 1.0
        assert results[0].similarity_score > 0.0  # Should have some similarity

    def test_search_evidence_multiple_documents(self):
        """Test search against multiple source documents."""
        searcher = EvidenceSearcher(top_k=2)
        claim = self._make_claim("Python is a programming language.")
        docs = [
            "Python is a popular programming language used worldwide.",
            "Java is a programming language.",
            "The weather is sunny today.",
        ]

        results = searcher.search_evidence(claim, docs)

        assert len(results) == 2  # top_k=2
        # Results should be sorted by descending similarity
        assert results[0].similarity_score >= results[1].similarity_score

    def test_search_evidence_scores_in_valid_range(self):
        """Test that all similarity scores are in [0, 1]."""
        searcher = EvidenceSearcher(top_k=5)
        claim = self._make_claim("Machine learning is a subset of AI.")
        docs = [
            "Machine learning algorithms learn from data.",
            "Artificial intelligence encompasses machine learning.",
            "Cooking recipes are fun.",
        ]

        results = searcher.search_evidence(claim, docs)

        for result in results:
            assert 0.0 <= result.similarity_score <= 1.0

    def test_search_evidence_top_k_limits_results(self):
        """Test that results are limited to top_k."""
        searcher = EvidenceSearcher(top_k=1)
        claim = self._make_claim("Test claim.")
        docs = ["Doc one.", "Doc two.", "Doc three."]

        results = searcher.search_evidence(claim, docs)

        assert len(results) == 1

    def test_search_evidence_batch(self):
        """Test batch evidence search for multiple claims."""
        searcher = EvidenceSearcher()
        claims = [
            self._make_claim("The sky is blue.", 0),
            self._make_claim("Water is wet.", 17),
        ]
        docs = ["The sky is blue and clear.", "Water is a liquid."]

        evidence_map = searcher.search_evidence_batch(claims, docs)

        assert "The sky is blue." in evidence_map
        assert "Water is wet." in evidence_map
        assert len(evidence_map) == 2

    def test_search_evidence_identical_text_high_similarity(self):
        """Test that identical text produces high similarity."""
        searcher = EvidenceSearcher()
        claim = self._make_claim("The Earth orbits the Sun.")
        docs = ["The Earth orbits the Sun."]

        results = searcher.search_evidence(claim, docs)

        assert len(results) == 1
        # Identical text should have very high similarity
        assert results[0].similarity_score > 0.9

    def test_search_evidence_unrelated_text_low_similarity(self):
        """Test that unrelated text produces lower similarity."""
        searcher = EvidenceSearcher()
        claim = self._make_claim("Quantum physics describes subatomic particles.")
        docs = ["Chocolate cake is delicious with frosting."]

        results = searcher.search_evidence(claim, docs)

        assert len(results) == 1
        # Unrelated text should have low similarity
        assert results[0].similarity_score < 0.5

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0."""
        vec = [1.0, 2.0, 3.0]
        sim = EvidenceSearcher._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = EvidenceSearcher._cosine_similarity(a, b)
        assert abs(sim) < 1e-9

    def test_cosine_similarity_empty_vectors(self):
        """Test cosine similarity of empty vectors returns 0.0."""
        sim = EvidenceSearcher._cosine_similarity([], [])
        assert sim == 0.0

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector returns 0.0."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        sim = EvidenceSearcher._cosine_similarity(a, b)
        assert sim == 0.0

    def test_bow_embedding_deterministic(self):
        """Test that BOW embedding is deterministic."""
        text = "Hello world test"
        emb1 = EvidenceSearcher._bow_embedding(text)
        emb2 = EvidenceSearcher._bow_embedding(text)
        assert emb1 == emb2

    def test_bow_embedding_empty_text(self):
        """Test BOW embedding of empty text returns zero vector."""
        emb = EvidenceSearcher._bow_embedding("")
        assert all(v == 0.0 for v in emb)

    def test_embedding_caching(self):
        """Test that embeddings are cached for repeated text."""
        searcher = EvidenceSearcher()
        claim = self._make_claim("Cached text.")
        docs = ["Cached text."]

        # First call populates cache
        searcher.search_evidence(claim, docs)
        cache_size_after_first = len(searcher._embedding_cache)

        # Second call should use cache (no new entries)
        searcher.search_evidence(claim, docs)
        cache_size_after_second = len(searcher._embedding_cache)

        assert cache_size_after_first == cache_size_after_second

    def test_custom_embedding_fn(self):
        """Test using a custom embedding function."""
        def dummy_embedding(text: str) -> list[float]:
            # Simple hash-based embedding for testing
            return [float(len(text) % 10) / 10.0] * 4

        searcher = EvidenceSearcher(embedding_fn=dummy_embedding)
        claim = self._make_claim("Test claim.")
        docs = ["Test document."]

        results = searcher.search_evidence(claim, docs)

        assert len(results) == 1
        assert 0.0 <= results[0].similarity_score <= 1.0

    def test_evidence_source_span_truncation(self):
        """Test that long documents have their span truncated."""
        searcher = EvidenceSearcher()
        claim = self._make_claim("Short claim.")
        long_doc = "A" * 500  # 500 chars

        results = searcher.search_evidence(claim, [long_doc])

        assert len(results) == 1
        assert len(results[0].span) == 200  # Truncated to 200


class TestDetectWithSourceDocuments:
    """Test the full detect() pipeline with source documents."""

    def test_detect_with_source_documents_supported(self):
        """Test detection where claims are supported by source docs."""
        from src.hallucination.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        response = "The sky is blue."
        source_docs = ["The sky is blue and clear today."]

        result = detector.detect(response, source_documents=source_docs)

        assert result.total_claims >= 1
        assert 0.0 <= result.hallucination_rate <= 1.0
        assert 0.0 <= result.overall_grounding_score <= 1.0
        assert len(result.claim_evidence) >= 1

    def test_detect_with_source_documents_unsupported(self):
        """Test detection where claims are not supported by source docs."""
        from src.hallucination.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        response = "Unicorns fly over rainbows every Tuesday."
        source_docs = ["The weather forecast predicts rain tomorrow."]

        result = detector.detect(response, source_documents=source_docs)

        assert result.total_claims >= 1
        assert len(result.claim_evidence) >= 1

    def test_detect_without_source_documents(self):
        """Test detection without source documents (all claims unsupported)."""
        from src.hallucination.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        response = "The Earth is round."

        result = detector.detect(response)

        assert result.total_claims >= 1
        # Without source docs, factual claims should be unsupported
        if result.factual_claims > 0:
            assert result.unsupported_claims == result.factual_claims

    def test_detect_claim_evidence_populated(self):
        """Test that claim_evidence is properly populated."""
        from src.hallucination.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        response = "Python was created by Guido van Rossum."
        source_docs = ["Python was created by Guido van Rossum in 1991."]

        result = detector.detect(response, source_documents=source_docs)

        assert len(result.claim_evidence) >= 1
        for ce in result.claim_evidence:
            assert ce.claim is not None
            assert 0.0 <= ce.similarity_score <= 1.0

    def test_detect_supported_unsupported_counts_consistent(self):
        """Test that supported + unsupported = factual_claims."""
        from src.hallucination.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        response = "The sky is blue. Water boils at 100 degrees."
        source_docs = ["The sky is blue on clear days."]

        result = detector.detect(response, source_documents=source_docs)

        assert result.supported_claims + result.unsupported_claims == result.factual_claims


class TestDetectBatch:
    """Test suite for detect_batch() method."""

    def test_detect_batch_basic(self):
        """Test basic batch detection."""
        from src.hallucination.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        items = [
            {"response": "The sky is blue.", "source_documents": ["The sky is blue."]},
            {"response": "Water is wet.", "source_documents": ["Water is a liquid."]},
        ]

        results = detector.detect_batch(items)

        assert len(results) == 2
        for result in results:
            assert 0.0 <= result.hallucination_rate <= 1.0
            assert 0.0 <= result.overall_grounding_score <= 1.0

    def test_detect_batch_empty_list(self):
        """Test batch detection with empty list."""
        from src.hallucination.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()

        results = detector.detect_batch([])

        assert results == []

    def test_detect_batch_without_source_documents(self):
        """Test batch detection without source documents."""
        from src.hallucination.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        items = [
            {"response": "The sky is blue."},
            {"response": "Water is wet."},
        ]

        results = detector.detect_batch(items)

        assert len(results) == 2

    def test_detect_batch_with_sensitivity_override(self):
        """Test batch detection with sensitivity level override."""
        from src.hallucination.hallucination_detector import HallucinationDetector
        from src.data_models.hallucination import SensitivityLevel

        detector = HallucinationDetector()
        items = [
            {"response": "The sky is blue.", "source_documents": ["The sky is blue."]},
        ]

        results = detector.detect_batch(
            items, sensitivity_level=SensitivityLevel.STRICT
        )

        assert len(results) == 1
        assert results[0].sensitivity_level == SensitivityLevel.STRICT

    def test_detect_batch_mixed_items(self):
        """Test batch with mix of items with and without source docs."""
        from src.hallucination.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        items = [
            {"response": "Claim one.", "source_documents": ["Source one."]},
            {"response": "Claim two."},
            {"response": "", "source_documents": []},
        ]

        results = detector.detect_batch(items)

        assert len(results) == 3
        # Empty response should have 0 claims
        assert results[2].total_claims == 0
