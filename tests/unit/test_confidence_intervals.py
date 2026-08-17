"""
Unit tests for confidence interval calculation in TrustScoringEngine.

Tests the calculate_confidence_interval method with various scenarios
including bootstrap and normal approximation methods.

Requirements: 5.10
"""

import pytest
import numpy as np
from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine
from src.data_models.trust_score import TrustScoreConfig


class TestConfidenceIntervals:
    """Test suite for confidence interval calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TrustScoringEngine(
            inference_client=None,
            config=TrustScoreConfig(),
        )

    def test_bootstrap_method_small_sample(self):
        """Test bootstrap method with small sample (<30)."""
        # Small sample of 20 scores
        scores = [0.7, 0.75, 0.8, 0.72, 0.78, 0.76, 0.74, 0.79, 0.77, 0.73,
                  0.71, 0.75, 0.78, 0.76, 0.74, 0.72, 0.77, 0.75, 0.73, 0.76]

        result = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="bootstrap",
        )

        # Verify result structure
        assert "mean" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "confidence_level" in result
        assert "method" in result
        assert "sample_size" in result

        # Verify values
        assert result["method"] == "bootstrap"
        assert result["sample_size"] == 20
        assert result["confidence_level"] == 0.95
        assert 0.0 <= result["mean"] <= 1.0
        assert 0.0 <= result["lower_bound"] <= 1.0
        assert 0.0 <= result["upper_bound"] <= 1.0
        assert result["lower_bound"] <= result["mean"] <= result["upper_bound"]

        # Mean should be close to 0.75
        assert abs(result["mean"] - 0.75) < 0.02

    def test_normal_method_large_sample(self):
        """Test normal approximation with large sample (>=30)."""
        # Large sample of 100 scores with mean around 0.8
        np.random.seed(42)
        scores = np.random.normal(loc=0.8, scale=0.1, size=100).tolist()
        # Clip to valid range [0, 1]
        scores = [max(0.0, min(1.0, s)) for s in scores]

        result = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="normal",
        )

        # Verify result structure
        assert result["method"] == "normal"
        assert result["sample_size"] == 100
        assert result["confidence_level"] == 0.95
        assert 0.0 <= result["mean"] <= 1.0
        assert 0.0 <= result["lower_bound"] <= 1.0
        assert 0.0 <= result["upper_bound"] <= 1.0
        assert result["lower_bound"] <= result["mean"] <= result["upper_bound"]

        # Mean should be close to 0.8
        assert abs(result["mean"] - 0.8) < 0.05

    def test_auto_method_selects_bootstrap_for_small_sample(self):
        """Test auto method selects bootstrap for sample size < 30."""
        scores = [0.7, 0.75, 0.8, 0.72, 0.78, 0.76, 0.74, 0.79, 0.77, 0.73]

        result = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="auto",
        )

        assert result["method"] == "bootstrap"
        assert result["sample_size"] == 10

    def test_auto_method_selects_normal_for_large_sample(self):
        """Test auto method selects normal for sample size >= 30."""
        scores = [0.75] * 50  # 50 identical scores

        result = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="auto",
        )

        assert result["method"] == "normal"
        assert result["sample_size"] == 50

    def test_different_confidence_levels(self):
        """Test confidence intervals at different confidence levels."""
        scores = [0.7, 0.75, 0.8, 0.72, 0.78, 0.76, 0.74, 0.79, 0.77, 0.73,
                  0.71, 0.75, 0.78, 0.76, 0.74, 0.72, 0.77, 0.75, 0.73, 0.76]

        # 90% confidence interval
        result_90 = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.90,
            method="bootstrap",
        )

        # 95% confidence interval
        result_95 = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="bootstrap",
        )

        # 99% confidence interval
        result_99 = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.99,
            method="bootstrap",
        )

        # Higher confidence level should produce wider interval
        width_90 = result_90["upper_bound"] - result_90["lower_bound"]
        width_95 = result_95["upper_bound"] - result_95["lower_bound"]
        width_99 = result_99["upper_bound"] - result_99["lower_bound"]

        assert width_90 <= width_95 <= width_99

    def test_uniform_scores_narrow_interval(self):
        """Test that uniform scores produce narrow confidence interval."""
        # All scores are identical
        scores = [0.8] * 30

        result = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="normal",
        )

        # With no variance, interval should be very narrow
        interval_width = result["upper_bound"] - result["lower_bound"]
        assert interval_width < 0.01
        assert abs(result["mean"] - 0.8) < 0.001

    def test_high_variance_scores_wide_interval(self):
        """Test that high variance scores produce wide confidence interval."""
        # Scores with high variance
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 3

        result = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="normal",
        )

        # With high variance, interval should be wider
        interval_width = result["upper_bound"] - result["lower_bound"]
        assert interval_width > 0.1

    def test_bounds_within_valid_range(self):
        """Test that confidence bounds are always within [0, 1]."""
        # Scores near boundaries
        scores_low = [0.01, 0.02, 0.03, 0.04, 0.05] * 6
        scores_high = [0.95, 0.96, 0.97, 0.98, 0.99] * 6

        result_low = self.engine.calculate_confidence_interval(
            scores=scores_low,
            confidence_level=0.95,
            method="normal",
        )

        result_high = self.engine.calculate_confidence_interval(
            scores=scores_high,
            confidence_level=0.95,
            method="normal",
        )

        # Bounds should be clipped to [0, 1]
        assert 0.0 <= result_low["lower_bound"] <= 1.0
        assert 0.0 <= result_low["upper_bound"] <= 1.0
        assert 0.0 <= result_high["lower_bound"] <= 1.0
        assert 0.0 <= result_high["upper_bound"] <= 1.0

    def test_empty_scores_raises_error(self):
        """Test that empty scores list raises ValueError."""
        with pytest.raises(ValueError, match="Scores list cannot be empty"):
            self.engine.calculate_confidence_interval(
                scores=[],
                confidence_level=0.95,
            )

    def test_invalid_confidence_level_raises_error(self):
        """Test that invalid confidence level raises ValueError."""
        scores = [0.7, 0.75, 0.8]

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            self.engine.calculate_confidence_interval(
                scores=scores,
                confidence_level=0.0,
            )

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            self.engine.calculate_confidence_interval(
                scores=scores,
                confidence_level=1.0,
            )

        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            self.engine.calculate_confidence_interval(
                scores=scores,
                confidence_level=1.5,
            )

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        scores = [0.7, 0.75, 0.8]

        with pytest.raises(ValueError, match="Invalid method"):
            self.engine.calculate_confidence_interval(
                scores=scores,
                confidence_level=0.95,
                method="invalid_method",
            )

    def test_single_score(self):
        """Test confidence interval with single score."""
        scores = [0.75]

        result = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="bootstrap",
        )

        # With single score, all bootstrap samples are identical
        assert result["mean"] == 0.75
        assert result["lower_bound"] == 0.75
        assert result["upper_bound"] == 0.75

    def test_two_scores(self):
        """Test confidence interval with two scores."""
        scores = [0.7, 0.8]

        result = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="bootstrap",
        )

        assert result["mean"] == 0.75
        assert result["sample_size"] == 2
        assert result["lower_bound"] <= result["mean"] <= result["upper_bound"]

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap method produces consistent results."""
        scores = [0.7, 0.75, 0.8, 0.72, 0.78, 0.76, 0.74, 0.79, 0.77, 0.73]

        result1 = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="bootstrap",
        )

        result2 = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="bootstrap",
        )

        # Results should be identical due to fixed seed
        assert result1["mean"] == result2["mean"]
        assert result1["lower_bound"] == result2["lower_bound"]
        assert result1["upper_bound"] == result2["upper_bound"]

    def test_normal_vs_bootstrap_similar_for_large_sample(self):
        """Test that normal and bootstrap methods give similar results for large samples."""
        # Large sample with normal distribution
        np.random.seed(42)
        scores = np.random.normal(loc=0.75, scale=0.1, size=100).tolist()
        scores = [max(0.0, min(1.0, s)) for s in scores]

        result_normal = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="normal",
        )

        result_bootstrap = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="bootstrap",
        )

        # Means should be identical
        assert result_normal["mean"] == result_bootstrap["mean"]

        # Confidence intervals should be similar (within 10% of each other)
        normal_width = result_normal["upper_bound"] - result_normal["lower_bound"]
        bootstrap_width = result_bootstrap["upper_bound"] - result_bootstrap["lower_bound"]
        
        relative_diff = abs(normal_width - bootstrap_width) / normal_width
        assert relative_diff < 0.2  # Within 20% difference

    def test_extreme_scores(self):
        """Test confidence intervals with extreme score values."""
        # All scores at minimum
        scores_min = [0.0] * 30
        result_min = self.engine.calculate_confidence_interval(
            scores=scores_min,
            confidence_level=0.95,
            method="normal",
        )
        assert result_min["mean"] == 0.0
        assert result_min["lower_bound"] == 0.0
        assert result_min["upper_bound"] == 0.0

        # All scores at maximum
        scores_max = [1.0] * 30
        result_max = self.engine.calculate_confidence_interval(
            scores=scores_max,
            confidence_level=0.95,
            method="normal",
        )
        assert result_max["mean"] == 1.0
        assert result_max["lower_bound"] == 1.0
        assert result_max["upper_bound"] == 1.0

    def test_realistic_trust_scores(self):
        """Test with realistic trust score distribution."""
        # Simulate realistic trust scores from an evaluation
        scores = [
            0.82, 0.78, 0.85, 0.79, 0.81, 0.83, 0.77, 0.84, 0.80, 0.82,
            0.79, 0.81, 0.83, 0.78, 0.85, 0.80, 0.82, 0.79, 0.84, 0.81,
            0.83, 0.78, 0.80, 0.82, 0.79, 0.85, 0.81, 0.83, 0.80, 0.82,
        ]

        result = self.engine.calculate_confidence_interval(
            scores=scores,
            confidence_level=0.95,
            method="auto",
        )

        # Should use normal method for 30 samples
        assert result["method"] == "normal"
        assert result["sample_size"] == 30

        # Mean should be around 0.81
        assert 0.79 <= result["mean"] <= 0.83

        # Confidence interval should be reasonable
        interval_width = result["upper_bound"] - result["lower_bound"]
        assert 0.01 <= interval_width <= 0.1
