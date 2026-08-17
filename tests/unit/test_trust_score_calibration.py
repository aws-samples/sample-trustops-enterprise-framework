"""
Unit tests for trust score calibration functionality.

Tests the calibration method that validates trust scores against
human-labeled ground truth data.

Requirements: 5.8
"""

import pytest
import numpy as np

from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine
from src.data_models.trust_score import TrustScoreConfig, TrustScoreWeights


class TestTrustScoreCalibration:
    """Test suite for trust score calibration."""

    @pytest.fixture
    def engine(self):
        """Create a trust scoring engine for testing."""
        config = TrustScoreConfig(
            weights=TrustScoreWeights(
                accuracy=0.25,
                consistency=0.20,
                safety=0.20,
                bias=0.15,
                context_grounding=0.20,
            ),
            review_threshold=0.6,
            consistency_samples=3,
        )
        return TrustScoringEngine(config=config)

    @pytest.fixture
    def perfect_correlation_data(self):
        """Ground truth data with perfect correlation."""
        return [
            {
                "prompt": "What is 2+2?",
                "response": "4",
                "expected_response": "4",
                "human_score": 0.9,
            },
            {
                "prompt": "What is the capital of France?",
                "response": "Paris",
                "expected_response": "Paris",
                "human_score": 0.9,
            },
            {
                "prompt": "What is 5+5?",
                "response": "10",
                "expected_response": "10",
                "human_score": 0.9,
            },
        ]

    @pytest.fixture
    def varied_quality_data(self):
        """Ground truth data with varied quality scores."""
        return [
            {
                "prompt": "What is 2+2?",
                "response": "4",
                "expected_response": "4",
                "human_score": 0.95,
            },
            {
                "prompt": "What is the capital of France?",
                "response": "I think it might be Paris",
                "expected_response": "Paris",
                "human_score": 0.75,
            },
            {
                "prompt": "What is 5+5?",
                "response": "Maybe 10 or 11",
                "expected_response": "10",
                "human_score": 0.50,
            },
            {
                "prompt": "What is the largest planet?",
                "response": "Jupiter is the largest planet in our solar system",
                "expected_response": "Jupiter",
                "human_score": 0.90,
            },
            {
                "prompt": "What is photosynthesis?",
                "response": "It's something plants do",
                "expected_response": "The process by which plants convert light into energy",
                "human_score": 0.40,
            },
        ]

    @pytest.mark.asyncio
    async def test_calibrate_returns_required_metrics(self, engine, perfect_correlation_data):
        """Test that calibration returns all required metrics."""
        result = await engine.calibrate(perfect_correlation_data)

        # Check all required fields are present
        assert "pearson_correlation" in result
        assert "pearson_pvalue" in result
        assert "spearman_correlation" in result
        assert "spearman_pvalue" in result
        assert "mean_absolute_error" in result
        assert "rmse" in result
        assert "sample_size" in result
        assert "score_pairs" in result
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_calibrate_sample_size(self, engine, varied_quality_data):
        """Test that sample size is correctly reported."""
        result = await engine.calibrate(varied_quality_data)

        assert result["sample_size"] == len(varied_quality_data)
        assert len(result["score_pairs"]) == len(varied_quality_data)

    @pytest.mark.asyncio
    async def test_calibrate_correlation_range(self, engine, varied_quality_data):
        """Test that correlation coefficients are in valid range [-1, 1]."""
        result = await engine.calibrate(varied_quality_data)

        assert -1.0 <= result["pearson_correlation"] <= 1.0
        assert -1.0 <= result["spearman_correlation"] <= 1.0

    @pytest.mark.asyncio
    async def test_calibrate_error_metrics_non_negative(self, engine, varied_quality_data):
        """Test that error metrics are non-negative."""
        result = await engine.calibrate(varied_quality_data)

        assert result["mean_absolute_error"] >= 0.0
        assert result["rmse"] >= 0.0

    @pytest.mark.asyncio
    async def test_calibrate_score_pairs_format(self, engine, varied_quality_data):
        """Test that score pairs are correctly formatted."""
        result = await engine.calibrate(varied_quality_data)

        score_pairs = result["score_pairs"]
        assert len(score_pairs) == len(varied_quality_data)

        for predicted, actual in score_pairs:
            # Both scores should be in [0, 1] range
            assert 0.0 <= predicted <= 1.0
            assert 0.0 <= actual <= 1.0

    @pytest.mark.asyncio
    async def test_calibrate_with_empty_data(self, engine):
        """Test that calibration raises error with empty data."""
        with pytest.raises(ValueError, match="Ground truth data cannot be empty"):
            await engine.calibrate([])

    @pytest.mark.asyncio
    async def test_calibrate_summary_generation(self, engine, varied_quality_data):
        """Test that calibration summary is generated."""
        result = await engine.calibrate(varied_quality_data)

        summary = result["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "correlation" in summary.lower()
        assert "samples" in summary.lower()

    @pytest.mark.asyncio
    async def test_calibrate_with_source_documents(self, engine):
        """Test calibration with source documents for grounding."""
        ground_truth_data = [
            {
                "prompt": "What does the document say about climate?",
                "response": "The document states that climate change is accelerating",
                "source_documents": [
                    "Climate change is accelerating due to human activities"
                ],
                "human_score": 0.85,
            },
            {
                "prompt": "What is mentioned about renewable energy?",
                "response": "Solar and wind power are growing rapidly",
                "source_documents": [
                    "Renewable energy sources like solar and wind are expanding"
                ],
                "human_score": 0.80,
            },
        ]

        result = await engine.calibrate(ground_truth_data)

        assert result["sample_size"] == 2
        assert "pearson_correlation" in result
        assert "spearman_correlation" in result

    @pytest.mark.asyncio
    async def test_calibrate_mae_calculation(self, engine):
        """Test that MAE is calculated correctly."""
        # Create data where we can predict the MAE
        ground_truth_data = [
            {
                "prompt": "Test 1",
                "response": "Response 1",
                "expected_response": "Response 1",
                "human_score": 0.8,
            },
            {
                "prompt": "Test 2",
                "response": "Response 2",
                "expected_response": "Response 2",
                "human_score": 0.6,
            },
        ]

        result = await engine.calibrate(ground_truth_data)

        # MAE should be non-negative and reasonable
        mae = result["mean_absolute_error"]
        assert mae >= 0.0
        assert mae <= 1.0  # Maximum possible error

    @pytest.mark.asyncio
    async def test_calibrate_rmse_vs_mae(self, engine, varied_quality_data):
        """Test that RMSE is >= MAE (mathematical property)."""
        result = await engine.calibrate(varied_quality_data)

        mae = result["mean_absolute_error"]
        rmse = result["rmse"]

        # RMSE should always be >= MAE
        assert rmse >= mae

    @pytest.mark.asyncio
    async def test_calibrate_pvalue_range(self, engine, varied_quality_data):
        """Test that p-values are in valid range [0, 1]."""
        result = await engine.calibrate(varied_quality_data)

        assert 0.0 <= result["pearson_pvalue"] <= 1.0
        assert 0.0 <= result["spearman_pvalue"] <= 1.0

    @pytest.mark.asyncio
    async def test_calibrate_with_minimum_samples(self, engine):
        """Test calibration with minimum number of samples."""
        # Correlation requires at least 2 samples
        ground_truth_data = [
            {
                "prompt": "Test 1",
                "response": "Response 1",
                "expected_response": "Response 1",
                "human_score": 0.8,
            },
            {
                "prompt": "Test 2",
                "response": "Response 2",
                "expected_response": "Response 2",
                "human_score": 0.6,
            },
        ]

        result = await engine.calibrate(ground_truth_data)

        assert result["sample_size"] == 2
        assert "pearson_correlation" in result
        assert "spearman_correlation" in result

    @pytest.mark.asyncio
    async def test_calibrate_summary_interpretation(self, engine):
        """Test that summary provides meaningful interpretation."""
        # Create data with known characteristics
        ground_truth_data = [
            {
                "prompt": f"Test {i}",
                "response": f"Response {i}",
                "expected_response": f"Response {i}",
                "human_score": 0.5 + (i * 0.1),
            }
            for i in range(5)
        ]

        result = await engine.calibrate(ground_truth_data)

        summary = result["summary"]

        # Summary should mention correlation strength
        assert any(
            word in summary.lower()
            for word in ["strong", "moderate", "weak"]
        )

        # Summary should mention accuracy level
        assert any(
            word in summary.lower()
            for word in ["excellent", "good", "acceptable", "poor"]
        )

    @pytest.mark.asyncio
    async def test_calibrate_consistency_across_runs(self, engine, varied_quality_data):
        """Test that calibration produces consistent results across runs."""
        result1 = await engine.calibrate(varied_quality_data)
        result2 = await engine.calibrate(varied_quality_data)

        # Correlation coefficients should be identical
        assert result1["pearson_correlation"] == result2["pearson_correlation"]
        assert result1["spearman_correlation"] == result2["spearman_correlation"]
        assert result1["mean_absolute_error"] == result2["mean_absolute_error"]
        assert result1["rmse"] == result2["rmse"]

    @pytest.mark.asyncio
    async def test_calibrate_with_perfect_scores(self, engine):
        """Test calibration when all human scores are perfect."""
        ground_truth_data = [
            {
                "prompt": f"Test {i}",
                "response": f"Response {i}",
                "expected_response": f"Response {i}",
                "human_score": 1.0,
            }
            for i in range(5)
        ]

        result = await engine.calibrate(ground_truth_data)

        # Should still produce valid metrics
        assert result["sample_size"] == 5
        assert "pearson_correlation" in result
        assert result["mean_absolute_error"] >= 0.0

    @pytest.mark.asyncio
    async def test_calibrate_with_zero_scores(self, engine):
        """Test calibration when all human scores are zero."""
        ground_truth_data = [
            {
                "prompt": f"Test {i}",
                "response": f"Bad response {i}",
                "expected_response": f"Good response {i}",
                "human_score": 0.0,
            }
            for i in range(5)
        ]

        result = await engine.calibrate(ground_truth_data)

        # Should still produce valid metrics
        assert result["sample_size"] == 5
        assert "pearson_correlation" in result
        assert result["mean_absolute_error"] >= 0.0


class TestCalibrationSummaryGeneration:
    """Test suite for calibration summary generation."""

    @pytest.fixture
    def engine(self):
        """Create a trust scoring engine for testing."""
        return TrustScoringEngine()

    def test_summary_strong_correlation(self, engine):
        """Test summary for strong correlation."""
        summary = engine._generate_calibration_summary(
            pearson_corr=0.85,
            spearman_corr=0.82,
            mae=0.08,
            sample_size=50,
        )

        assert "strong" in summary.lower()
        assert "excellent" in summary.lower()
        assert "50 samples" in summary

    def test_summary_moderate_correlation(self, engine):
        """Test summary for moderate correlation."""
        summary = engine._generate_calibration_summary(
            pearson_corr=0.65,
            spearman_corr=0.62,
            mae=0.15,
            sample_size=30,
        )

        assert "moderate" in summary.lower()
        assert "good" in summary.lower()

    def test_summary_weak_correlation(self, engine):
        """Test summary for weak correlation."""
        summary = engine._generate_calibration_summary(
            pearson_corr=0.35,
            spearman_corr=0.32,
            mae=0.35,
            sample_size=20,
        )

        assert "weak" in summary.lower()
        assert "poor" in summary.lower()
        assert "consider adjusting" in summary.lower()

    def test_summary_includes_metrics(self, engine):
        """Test that summary includes key metrics."""
        summary = engine._generate_calibration_summary(
            pearson_corr=0.75,
            spearman_corr=0.72,
            mae=0.12,
            sample_size=40,
        )

        assert "0.750" in summary  # Pearson correlation
        assert "0.720" in summary  # Spearman correlation
        assert "0.120" in summary  # MAE
        assert "40 samples" in summary
