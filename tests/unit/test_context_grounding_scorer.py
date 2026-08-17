"""
Unit tests for Context Grounding Scorer.

Tests the context grounding dimension of trust scoring, including:
- Semantic similarity calculation with source documents
- Handling of missing or empty source documents
- Citation detection
- Batch scoring
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock

from src.trust_scoring.scorers.context_grounding_scorer import (
    ContextGroundingScorer
)
from src.data_models.trust_score import TrustDimension


class TestContextGroundingScorer:
    """Test suite for ContextGroundingScorer."""

    @pytest.fixture
    def mock_similarity_analyzer(self):
        """Create mock similarity analyzer."""
        analyzer = Mock()
        analyzer.calculate_similarity = Mock(return_value=0.85)
        return analyzer

    @pytest.fixture
    def scorer(self, mock_similarity_analyzer):
        """Create scorer instance with mock analyzer."""
        return ContextGroundingScorer(
            similarity_analyzer=mock_similarity_analyzer
        )

    @pytest.mark.asyncio
    async def test_high_grounding_score(
        self, scorer, mock_similarity_analyzer
    ):
        """Test response with high grounding in source documents."""
        response = "The capital of France is Paris."
        sources = [
            "Paris is the capital and largest city of France.",
            "France is a country in Western Europe."
        ]

        # Mock high similarity
        mock_similarity_analyzer.calculate_similarity.return_value = 0.9

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        assert result.score >= 0.8
        assert "high_grounding" in result.checks_passed
        assert result.confidence > 0.5
        assert result.details["num_sources"] == 2
        assert result.details["max_similarity"] >= 0.8

    @pytest.mark.asyncio
    async def test_low_grounding_score(
        self, scorer, mock_similarity_analyzer
    ):
        """Test response with low grounding in source documents."""
        response = "The moon is made of cheese."
        sources = [
            "The moon is Earth's only natural satellite.",
            "The moon has no atmosphere."
        ]

        # Mock low similarity
        mock_similarity_analyzer.calculate_similarity.return_value = 0.3

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        assert result.score < 0.5
        assert "low_grounding" in result.checks_failed
        assert result.details["max_similarity"] < 0.5

    @pytest.mark.asyncio
    async def test_no_source_documents(self, scorer):
        """Test handling when no source documents provided."""
        response = "This is a response."
        sources = None

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        assert result.score == 0.5
        assert result.confidence < 0.5
        assert "no_source_documents" in result.checks_failed
        assert "no_source_documents" in result.details["method"]

    @pytest.mark.asyncio
    async def test_empty_source_documents(self, scorer):
        """Test handling when source documents list is empty."""
        response = "This is a response."
        sources = []

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        assert result.score == 0.5
        assert "no_source_documents" in result.checks_failed

    @pytest.mark.asyncio
    async def test_empty_response(self, scorer):
        """Test handling when response is empty."""
        response = ""
        sources = ["Some source document."]

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        assert result.score == 0.0
        assert result.confidence == 1.0
        assert "empty_response" in result.checks_failed

    @pytest.mark.asyncio
    async def test_whitespace_only_response(self, scorer):
        """Test handling when response is only whitespace."""
        response = "   \n\t  "
        sources = ["Some source document."]

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        assert result.score == 0.0
        assert "empty_response" in result.checks_failed

    @pytest.mark.asyncio
    async def test_all_empty_sources(self, scorer):
        """Test handling when all source documents are empty."""
        response = "This is a response."
        sources = ["", "   ", "\n\t"]

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        assert result.score == 0.5
        assert "no_valid_sources" in result.checks_failed

    @pytest.mark.asyncio
    async def test_multiple_sources_varying_similarity(
        self, scorer, mock_similarity_analyzer
    ):
        """Test with multiple sources having different similarities."""
        response = "Python is a programming language."
        sources = [
            "Python is a high-level programming language.",
            "Java is also a programming language.",
            "The sky is blue."
        ]

        # Mock varying similarities
        similarities = [0.9, 0.6, 0.2]
        mock_similarity_analyzer.calculate_similarity.side_effect = (
            similarities
        )

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        # Should use maximum similarity
        assert result.score == 0.9
        assert result.details["max_similarity"] == 0.9
        assert result.details["avg_similarity"] == pytest.approx(
            sum(similarities) / len(similarities)
        )
        assert result.details["best_match_index"] == 0

    @pytest.mark.asyncio
    async def test_citation_detection_brackets(self, scorer):
        """Test detection of bracket-style citations."""
        response = "According to research [1], this is true."
        sources = ["Research shows this is true."]

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert "explicit_citations" in result.checks_passed
        assert result.details["has_explicit_citations"] is True

    @pytest.mark.asyncio
    async def test_citation_detection_source(self, scorer):
        """Test detection of source-style citations."""
        response = "This is true (Source: Research Paper)."
        sources = ["Research shows this is true."]

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert "explicit_citations" in result.checks_passed
        assert result.details["has_explicit_citations"] is True

    @pytest.mark.asyncio
    async def test_citation_detection_according_to(self, scorer):
        """Test detection of 'According to' citations."""
        response = "According to the study, this is correct."
        sources = ["The study shows this is correct."]

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert "explicit_citations" in result.checks_passed

    @pytest.mark.asyncio
    async def test_no_citations(
        self, scorer, mock_similarity_analyzer
    ):
        """Test response without explicit citations."""
        response = "This is a simple statement."
        sources = ["This is a simple statement."]

        mock_similarity_analyzer.calculate_similarity.return_value = 0.9

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert "no_explicit_citations" in result.checks_failed
        assert result.details["has_explicit_citations"] is False

    @pytest.mark.asyncio
    async def test_similarity_calculation_error(
        self, scorer, mock_similarity_analyzer
    ):
        """Test handling of similarity calculation errors."""
        response = "This is a response."
        sources = ["Source document."]

        # Mock error in similarity calculation
        mock_similarity_analyzer.calculate_similarity.side_effect = (
            Exception("API error")
        )

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        # When calculation fails, uses neutral score 0.5 for resilience
        assert result.score == 0.5
        assert "moderate_grounding" in result.checks_passed
        # Similarity should be neutral score
        assert result.details["similarities"] == [0.5]

    @pytest.mark.asyncio
    async def test_partial_similarity_errors(
        self, scorer, mock_similarity_analyzer
    ):
        """Test handling when some similarity calculations fail."""
        response = "This is a response."
        sources = ["Source 1", "Source 2", "Source 3"]

        # Mock partial failures
        def side_effect_func(text1, text2):
            if "Source 2" in text2:
                raise Exception("API error")
            return 0.7

        mock_similarity_analyzer.calculate_similarity.side_effect = (
            side_effect_func
        )

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        # Should still return a score using successful calculations
        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        assert result.score > 0.0
        # Failed calculation should use neutral score 0.5
        assert 0.5 in result.details["similarities"]

    @pytest.mark.asyncio
    async def test_moderate_grounding(
        self, scorer, mock_similarity_analyzer
    ):
        """Test response with moderate grounding."""
        response = "This is somewhat related."
        sources = ["This is a related topic."]

        # Mock moderate similarity
        mock_similarity_analyzer.calculate_similarity.return_value = 0.65

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        assert 0.5 <= result.score < 0.8
        assert "moderate_grounding" in result.checks_passed
        assert "low_grounding" not in result.checks_failed
        assert "high_grounding" not in result.checks_passed

    @pytest.mark.asyncio
    async def test_score_range_validation(
        self, scorer, mock_similarity_analyzer
    ):
        """Test that scores are always in valid [0, 1] range."""
        response = "Test response."
        sources = ["Test source."]

        # Mock out-of-range similarity (should be clamped)
        mock_similarity_analyzer.calculate_similarity.return_value = 1.5

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_batch_grounding(
        self, scorer, mock_similarity_analyzer
    ):
        """Test batch scoring of multiple responses."""
        items = [
            ("Response 1", ["Source 1"]),
            ("Response 2", ["Source 2"]),
            ("Response 3", ["Source 3"])
        ]

        # Mock similarities
        mock_similarity_analyzer.calculate_similarity.return_value = 0.8

        results = await scorer.calculate_batch_grounding(items)

        assert len(results) == 3
        for result in results:
            assert result.dimension == TrustDimension.CONTEXT_GROUNDING
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_single_source_document(
        self, scorer, mock_similarity_analyzer
    ):
        """Test with single source document."""
        response = "This is a response."
        sources = ["This is the only source."]

        mock_similarity_analyzer.calculate_similarity.return_value = 0.75

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.dimension == TrustDimension.CONTEXT_GROUNDING
        assert result.score == 0.75
        assert result.details["num_sources"] == 1
        assert result.details["best_match_index"] == 0

    @pytest.mark.asyncio
    async def test_confidence_calculation(
        self, scorer, mock_similarity_analyzer
    ):
        """Test confidence calculation based on similarity variance."""
        response = "Test response."
        sources = ["Source 1", "Source 2", "Source 3"]

        # Mock consistent similarities (low variance)
        mock_similarity_analyzer.calculate_similarity.return_value = 0.8

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        # High consistency should lead to high confidence
        assert result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_confidence_with_variance(
        self, scorer, mock_similarity_analyzer
    ):
        """Test confidence with high variance in similarities."""
        response = "Test response."
        sources = ["Source 1", "Source 2", "Source 3"]

        # Mock varying similarities (high variance)
        similarities = [0.9, 0.5, 0.2]
        mock_similarity_analyzer.calculate_similarity.side_effect = (
            similarities
        )

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        # High variance should lead to lower confidence
        assert result.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_details_completeness(
        self, scorer, mock_similarity_analyzer
    ):
        """Test that result details contain all expected fields."""
        response = "Test response."
        sources = ["Source 1", "Source 2"]

        mock_similarity_analyzer.calculate_similarity.return_value = 0.7

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        # Check all expected detail fields
        assert "max_similarity" in result.details
        assert "avg_similarity" in result.details
        assert "min_similarity" in result.details
        assert "num_sources" in result.details
        assert "similarities" in result.details
        assert "best_match_index" in result.details
        assert "best_match_preview" in result.details
        assert "has_explicit_citations" in result.details
        assert "response_length" in result.details
        assert "embedding_model" in result.details

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, mock_similarity_analyzer):
        """Test scorer with custom grounding thresholds."""
        scorer = ContextGroundingScorer(
            similarity_analyzer=mock_similarity_analyzer,
            high_grounding_threshold=0.9,
            low_grounding_threshold=0.4
        )

        response = "Test response."
        sources = ["Test source."]

        # Test score at 0.85 (between thresholds)
        mock_similarity_analyzer.calculate_similarity.return_value = 0.85

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        # Should be moderate with custom thresholds
        assert "moderate_grounding" in result.checks_passed
        assert "high_grounding" not in result.checks_passed

    @pytest.mark.asyncio
    async def test_custom_embedding_model(self, mock_similarity_analyzer):
        """Test scorer with custom embedding model."""
        custom_model = "cohere.embed-english-v3"
        scorer = ContextGroundingScorer(
            similarity_analyzer=mock_similarity_analyzer,
            embedding_model_id=custom_model
        )

        response = "Test response."
        sources = ["Test source."]

        mock_similarity_analyzer.calculate_similarity.return_value = 0.8

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        assert result.details["embedding_model"] == custom_model

    @pytest.mark.asyncio
    async def test_best_match_preview(
        self, scorer, mock_similarity_analyzer
    ):
        """Test that best match preview is included in details."""
        response = "Test response."
        long_source = "A" * 300  # Long source document
        sources = [long_source, "Short source"]

        # Mock first source as best match
        mock_similarity_analyzer.calculate_similarity.side_effect = [
            0.9, 0.5
        ]

        result = await scorer.calculate_context_grounding(
            response, sources
        )

        # Preview should be truncated to 200 chars
        assert len(result.details["best_match_preview"]) == 200
        assert result.details["best_match_index"] == 0


class TestCitationDetection:
    """Test suite for citation detection functionality."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        return ContextGroundingScorer()

    def test_detect_numbered_brackets(self, scorer):
        """Test detection of [1], [2] style citations."""
        assert scorer._check_for_citations("This is true [1].")
        assert scorer._check_for_citations("Multiple [1] citations [2].")
        assert scorer._check_for_citations("Citation at start [1] text.")

    def test_detect_numbered_parentheses(self, scorer):
        """Test detection of (1), (2) style citations."""
        assert scorer._check_for_citations("This is true (1).")
        assert scorer._check_for_citations("Multiple (1) citations (2).")

    def test_detect_source_citations(self, scorer):
        """Test detection of (Source: ...) citations."""
        assert scorer._check_for_citations("True (Source: Paper).")
        assert scorer._check_for_citations("True (Ref: Study).")

    def test_detect_according_to(self, scorer):
        """Test detection of 'According to' citations."""
        assert scorer._check_for_citations("According to the study...")
        assert scorer._check_for_citations("according to research...")

    def test_detect_as_stated_in(self, scorer):
        """Test detection of 'As stated in' citations."""
        assert scorer._check_for_citations("As stated in the paper...")
        assert scorer._check_for_citations("as mentioned in the book...")

    def test_detect_based_on(self, scorer):
        """Test detection of 'Based on' citations."""
        assert scorer._check_for_citations("Based on the research...")
        assert scorer._check_for_citations("based on evidence...")

    def test_detect_citing(self, scorer):
        """Test detection of 'Citing' citations."""
        assert scorer._check_for_citations("Citing the study...")
        assert scorer._check_for_citations("citing previous work...")

    def test_no_citations(self, scorer):
        """Test text without citations."""
        assert not scorer._check_for_citations("Simple statement.")
        assert not scorer._check_for_citations("No references here.")
        assert not scorer._check_for_citations("")

    def test_case_insensitive(self, scorer):
        """Test that citation detection is case-insensitive."""
        assert scorer._check_for_citations("ACCORDING TO the study...")
        assert scorer._check_for_citations("As Stated In the paper...")
        assert scorer._check_for_citations("BASED ON research...")
