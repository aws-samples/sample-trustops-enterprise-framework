"""
Unit tests for ConsistencyScorer.

Tests cover:
- Multiple model invocations with same prompt
- Pairwise similarity calculation
- Consistency score calculation
- Edge cases (empty responses, failures, etc.)

Requirements: 5.4, 5.17
"""

import pytest
from unittest.mock import Mock, AsyncMock

from src.trust_scoring.scorers.consistency_scorer import ConsistencyScorer
from src.data_models.trust_score import DimensionScore, TrustDimension
from src.data_models.model import InferenceResponse


class TestConsistencyScorer:
    """Test suite for ConsistencyScorer."""

    @pytest.fixture
    def mock_inference_client(self):
        """Create mock inference client."""
        client = Mock()
        client.invoke = AsyncMock()
        return client

    @pytest.fixture
    def mock_similarity_analyzer(self):
        """Create mock similarity analyzer."""
        analyzer = Mock()
        analyzer.calculate_similarity = Mock(return_value=0.95)
        return analyzer

    @pytest.fixture
    def scorer(self, mock_inference_client, mock_similarity_analyzer):
        """Create ConsistencyScorer instance with mocks."""
        return ConsistencyScorer(
            inference_client=mock_inference_client,
            similarity_analyzer=mock_similarity_analyzer,
            num_samples=3,
            high_consistency_threshold=0.9,
            low_consistency_threshold=0.6,
        )

    # Test basic consistency calculation

    @pytest.mark.asyncio
    async def test_high_consistency_identical_responses(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test high consistency with identical responses."""
        # Mock identical responses
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Paris is the capital of France",
            input_tokens=10,
            output_tokens=7,
            latency_ms=100.0,
            model_id="test-model",
            finish_reason="stop",
        )
        mock_similarity_analyzer.calculate_similarity.return_value = 1.0

        result = await scorer.calculate_consistency(
            prompt="What is the capital of France?",
            model_id="test-model",
        )

        assert isinstance(result, DimensionScore)
        assert result.dimension == TrustDimension.CONSISTENCY
        assert result.score == 1.0
        assert "high_consistency" in result.checks_passed
        assert mock_inference_client.invoke.call_count == 3

    @pytest.mark.asyncio
    async def test_moderate_consistency(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test moderate consistency with similar responses."""
        # Mock similar but not identical responses
        responses = [
            InferenceResponse(
                text="Paris is the capital of France",
                input_tokens=10,
                output_tokens=7,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="The capital of France is Paris",
                input_tokens=10,
                output_tokens=7,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="Paris serves as France's capital",
                input_tokens=10,
                output_tokens=7,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
        ]
        mock_inference_client.invoke.side_effect = responses
        mock_similarity_analyzer.calculate_similarity.return_value = 0.75

        result = await scorer.calculate_consistency(
            prompt="What is the capital of France?",
            model_id="test-model",
        )

        assert 0.7 <= result.score <= 0.8
        assert "moderate_consistency" in result.checks_passed
        assert result.details["num_samples"] == 3

    @pytest.mark.asyncio
    async def test_low_consistency(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test low consistency with very different responses."""
        # Mock very different responses
        responses = [
            InferenceResponse(
                text="Paris is the capital of France",
                input_tokens=10,
                output_tokens=7,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="Tokyo is in Japan",
                input_tokens=10,
                output_tokens=5,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="Berlin is the capital of Germany",
                input_tokens=10,
                output_tokens=7,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
        ]
        mock_inference_client.invoke.side_effect = responses
        mock_similarity_analyzer.calculate_similarity.return_value = 0.3

        result = await scorer.calculate_consistency(
            prompt="What is the capital of France?",
            model_id="test-model",
        )

        assert result.score < 0.6
        assert "low_consistency" in result.checks_failed

    # Test configurable num_samples

    @pytest.mark.asyncio
    async def test_custom_num_samples(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test with custom number of samples."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Paris",
            input_tokens=10,
            output_tokens=1,
            latency_ms=100.0,
            model_id="test-model",
            finish_reason="stop",
        )
        mock_similarity_analyzer.calculate_similarity.return_value = 1.0

        result = await scorer.calculate_consistency(
            prompt="Capital of France?",
            model_id="test-model",
            num_samples=5,
        )

        assert result.details["num_samples"] == 5
        assert mock_inference_client.invoke.call_count == 5

    @pytest.mark.asyncio
    async def test_minimum_samples_two(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test with minimum 2 samples."""
        responses = [
            InferenceResponse(
                text="Paris",
                input_tokens=10,
                output_tokens=1,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="Paris",
                input_tokens=10,
                output_tokens=1,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
        ]
        mock_inference_client.invoke.side_effect = responses
        mock_similarity_analyzer.calculate_similarity.return_value = 1.0

        result = await scorer.calculate_consistency(
            prompt="Capital?",
            model_id="test-model",
            num_samples=2,
        )

        assert result.details["num_samples"] == 2
        assert mock_inference_client.invoke.call_count == 2
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_insufficient_samples(self, scorer):
        """Test with insufficient samples (< 2)."""
        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
            num_samples=1,
        )

        assert result.score == 1.0
        assert result.confidence == 0.0
        assert "insufficient_samples" in result.checks_failed

    # Test pairwise similarity calculation

    @pytest.mark.asyncio
    async def test_pairwise_similarities_calculated(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test that pairwise similarities are calculated correctly."""
        responses = [
            InferenceResponse(
                text="Response A",
                input_tokens=10,
                output_tokens=2,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="Response B",
                input_tokens=10,
                output_tokens=2,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="Response C",
                input_tokens=10,
                output_tokens=2,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
        ]
        mock_inference_client.invoke.side_effect = responses

        # Set different similarities for each pair
        similarities = [0.9, 0.8, 0.85]
        mock_similarity_analyzer.calculate_similarity.side_effect = (
            similarities
        )

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
        )

        # Should have 3 pairwise comparisons: (A,B), (A,C), (B,C)
        assert len(result.details["pairwise_similarities"]) == 3
        # Average should be (0.9 + 0.8 + 0.85) / 3 = 0.85
        assert abs(result.score - 0.85) < 0.01

    @pytest.mark.asyncio
    async def test_pairwise_similarities_with_five_samples(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test pairwise similarities with 5 samples."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Test",
            input_tokens=10,
            output_tokens=1,
            latency_ms=100.0,
            model_id="test-model",
            finish_reason="stop",
        )
        mock_similarity_analyzer.calculate_similarity.return_value = 0.9

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
            num_samples=5,
        )

        # 5 samples should have 10 pairwise comparisons: C(5,2) = 10
        assert len(result.details["pairwise_similarities"]) == 10

    # Test edge cases

    @pytest.mark.asyncio
    async def test_empty_prompt(self, scorer):
        """Test with empty prompt."""
        result = await scorer.calculate_consistency(
            prompt="",
            model_id="test-model",
        )

        assert result.score == 0.0
        assert result.confidence == 1.0
        assert "empty_prompt" in result.checks_failed

    @pytest.mark.asyncio
    async def test_whitespace_only_prompt(self, scorer):
        """Test with whitespace-only prompt."""
        result = await scorer.calculate_consistency(
            prompt="   ",
            model_id="test-model",
        )

        assert result.score == 0.0
        assert "empty_prompt" in result.checks_failed

    @pytest.mark.asyncio
    async def test_all_empty_responses(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test when all responses are empty."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="",
            input_tokens=10,
            output_tokens=0,
            latency_ms=100.0,
            model_id="test-model",
            finish_reason="stop",
        )

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
        )

        assert result.score == 1.0
        assert result.confidence == 0.5
        assert "all_empty" in result.checks_passed

    @pytest.mark.asyncio
    async def test_model_invocation_failure(
        self, scorer, mock_inference_client
    ):
        """Test handling of model invocation failure."""
        mock_inference_client.invoke.side_effect = Exception(
            "Model unavailable"
        )

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
        )

        assert result.score == 0.0
        assert result.confidence == 0.0
        assert "model_invocation_failed" in result.checks_failed
        assert "Failed to invoke model" in result.details["error"]

    @pytest.mark.asyncio
    async def test_similarity_calculation_failure(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test handling of similarity calculation failure."""
        responses = [
            InferenceResponse(
                text="Response A",
                input_tokens=10,
                output_tokens=2,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="Response B",
                input_tokens=10,
                output_tokens=2,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
        ]
        mock_inference_client.invoke.side_effect = responses
        mock_similarity_analyzer.calculate_similarity.side_effect = (
            Exception("API Error")
        )

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
            num_samples=2,
        )

        # Should use neutral score (0.5) for failed similarity
        assert result.score == 0.5
        assert 0.0 <= result.score <= 1.0

    # Test score range validation

    @pytest.mark.asyncio
    async def test_score_always_in_valid_range(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test that score is always in [0, 1] range."""
        test_cases = [
            (1.0, "identical"),
            (0.95, "very similar"),
            (0.75, "similar"),
            (0.5, "moderate"),
            (0.25, "different"),
            (0.0, "completely different"),
        ]

        for similarity, _ in test_cases:
            mock_inference_client.invoke.return_value = InferenceResponse(
                text="Test response",
                input_tokens=10,
                output_tokens=2,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            )
            mock_similarity_analyzer.calculate_similarity.return_value = (
                similarity
            )

            result = await scorer.calculate_consistency(
                prompt="Test",
                model_id="test-model",
            )

            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0

    # Test confidence calculation

    @pytest.mark.asyncio
    async def test_confidence_increases_with_samples(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test that confidence increases with more samples."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Test",
            input_tokens=10,
            output_tokens=1,
            latency_ms=100.0,
            model_id="test-model",
            finish_reason="stop",
        )
        mock_similarity_analyzer.calculate_similarity.return_value = 0.9

        # Test with 2 samples
        result_2 = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
            num_samples=2,
        )

        # Reset mock
        mock_inference_client.invoke.reset_mock()
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Test",
            input_tokens=10,
            output_tokens=1,
            latency_ms=100.0,
            model_id="test-model",
            finish_reason="stop",
        )

        # Test with 5 samples
        result_5 = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
            num_samples=5,
        )

        # More samples should give higher confidence
        assert result_5.confidence > result_2.confidence

    # Test details field

    @pytest.mark.asyncio
    async def test_details_contains_required_fields(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test that details field contains all required information."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Test response",
            input_tokens=10,
            output_tokens=2,
            latency_ms=100.0,
            model_id="test-model",
            finish_reason="stop",
        )
        mock_similarity_analyzer.calculate_similarity.return_value = 0.9

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
        )

        assert "num_samples" in result.details
        assert "pairwise_similarities" in result.details
        assert "mean_similarity" in result.details
        assert "variance" in result.details
        assert "response_lengths" in result.details
        assert "model_id" in result.details

    @pytest.mark.asyncio
    async def test_details_has_correct_values(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test that details field has correct values."""
        responses = [
            InferenceResponse(
                text="Short",
                input_tokens=10,
                output_tokens=1,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="Medium length",
                input_tokens=10,
                output_tokens=2,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="A much longer response",
                input_tokens=10,
                output_tokens=4,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
        ]
        mock_inference_client.invoke.side_effect = responses
        mock_similarity_analyzer.calculate_similarity.return_value = 0.8

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
        )

        assert result.details["num_samples"] == 3
        assert result.details["model_id"] == "test-model"
        assert len(result.details["response_lengths"]) == 3
        assert result.details["response_lengths"][0] == len("Short")
        assert result.details["response_lengths"][1] == len(
            "Medium length"
        )
        assert result.details["response_lengths"][2] == len(
            "A much longer response"
        )

    # Test variance calculation

    @pytest.mark.asyncio
    async def test_variance_calculation(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test variance calculation in details."""
        responses = [
            InferenceResponse(
                text="A",
                input_tokens=10,
                output_tokens=1,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="B",
                input_tokens=10,
                output_tokens=1,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="C",
                input_tokens=10,
                output_tokens=1,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
        ]
        mock_inference_client.invoke.side_effect = responses

        # Set specific similarities
        similarities = [0.9, 0.8, 0.85]
        mock_similarity_analyzer.calculate_similarity.side_effect = (
            similarities
        )

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
        )

        # Variance should be calculated
        assert "variance" in result.details
        assert result.details["variance"] >= 0.0

    # Test inference parameters

    @pytest.mark.asyncio
    async def test_custom_inference_parameters(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test with custom inference parameters."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Test",
            input_tokens=10,
            output_tokens=1,
            latency_ms=100.0,
            model_id="test-model",
            finish_reason="stop",
        )
        mock_similarity_analyzer.calculate_similarity.return_value = 0.9

        custom_params = {
            "max_tokens": 500,
            "temperature": 0.5,
            "top_p": 0.95,
            "stop_sequences": ["END"],
        }

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
            inference_params=custom_params,
        )

        # Should complete successfully with custom params
        assert result.score >= 0.0
        assert mock_inference_client.invoke.called

    # Test checks passed/failed

    @pytest.mark.asyncio
    async def test_high_consistency_check(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test high consistency check passes."""
        mock_inference_client.invoke.return_value = InferenceResponse(
            text="Test",
            input_tokens=10,
            output_tokens=1,
            latency_ms=100.0,
            model_id="test-model",
            finish_reason="stop",
        )
        mock_similarity_analyzer.calculate_similarity.return_value = 0.95

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
        )

        assert "high_consistency" in result.checks_passed
        assert len(result.checks_failed) == 0

    @pytest.mark.asyncio
    async def test_low_consistency_check(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test low consistency check fails."""
        responses = [
            InferenceResponse(
                text="A",
                input_tokens=10,
                output_tokens=1,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="B",
                input_tokens=10,
                output_tokens=1,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
        ]
        mock_inference_client.invoke.side_effect = responses
        mock_similarity_analyzer.calculate_similarity.return_value = 0.3

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
            num_samples=2,
        )

        assert "low_consistency" in result.checks_failed
        assert len(result.checks_passed) == 0

    @pytest.mark.asyncio
    async def test_moderate_consistency_check(
        self, scorer, mock_inference_client, mock_similarity_analyzer
    ):
        """Test moderate consistency check passes."""
        responses = [
            InferenceResponse(
                text="A",
                input_tokens=10,
                output_tokens=1,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
            InferenceResponse(
                text="B",
                input_tokens=10,
                output_tokens=1,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            ),
        ]
        mock_inference_client.invoke.side_effect = responses
        mock_similarity_analyzer.calculate_similarity.return_value = 0.75

        result = await scorer.calculate_consistency(
            prompt="Test",
            model_id="test-model",
            num_samples=2,
        )

        assert "moderate_consistency" in result.checks_passed
        assert "high_consistency" not in result.checks_passed
        assert "low_consistency" not in result.checks_failed
