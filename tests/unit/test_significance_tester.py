"""Tests for the statistical significance tester.

Requirements: 7.5
"""

import math

import pytest

from src.evaluation.significance_tester import (
    SignificanceResult,
    calculate_significance,
)


class TestEdgeCases:
    """Edge-case handling for empty, single-element, and identical inputs."""

    def test_empty_lists(self):
        result = calculate_significance([], [])
        assert result.p_value == 1.0
        assert result.confidence_interval == (0.0, 0.0)
        assert result.statistical_significance == 0.0

    def test_single_element(self):
        result = calculate_significance([0.5], [0.8])
        assert result.p_value == 1.0
        # CI should be the single difference
        assert result.confidence_interval == pytest.approx((0.3, 0.3))
        assert result.statistical_significance == 0.0

    def test_identical_scores(self):
        scores = [0.7, 0.8, 0.6, 0.75, 0.65]
        result = calculate_significance(scores, scores)
        assert result.p_value == 1.0
        assert result.confidence_interval == (0.0, 0.0)
        assert result.statistical_significance == 0.0

    def test_constant_nonzero_difference(self):
        """All pairs differ by the same constant → perfectly significant."""
        s1 = [0.5, 0.5, 0.5, 0.5, 0.5]
        s2 = [0.7, 0.7, 0.7, 0.7, 0.7]
        result = calculate_significance(s1, s2)
        assert result.p_value == 0.0
        assert result.statistical_significance == 1.0
        assert result.confidence_interval == pytest.approx((0.2, 0.2))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            calculate_significance([0.5, 0.6], [0.7])


class TestNormalOperation:
    """Tests with realistic paired score data."""

    def test_significant_improvement(self):
        """Model 2 is clearly better → low p-value, high significance."""
        s1 = [0.4, 0.45, 0.42, 0.38, 0.41, 0.43, 0.39, 0.44, 0.40, 0.42]
        s2 = [0.7, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.74, 0.67, 0.72]
        result = calculate_significance(s1, s2)

        assert 0.0 <= result.p_value <= 1.0
        assert result.p_value < 0.05
        assert result.statistical_significance > 0.95
        # CI should be entirely positive (model 2 better)
        assert result.confidence_interval[0] > 0
        assert result.confidence_interval[1] > 0

    def test_no_significant_difference(self):
        """Scores are very similar → high p-value."""
        s1 = [0.70, 0.71, 0.69, 0.70, 0.72, 0.68, 0.71, 0.70, 0.69, 0.70]
        s2 = [0.71, 0.70, 0.70, 0.69, 0.71, 0.69, 0.70, 0.71, 0.70, 0.69]
        result = calculate_significance(s1, s2)

        assert 0.0 <= result.p_value <= 1.0
        # With such small differences, p-value should be high
        assert result.p_value > 0.05
        assert result.statistical_significance < 0.95

    def test_significant_degradation(self):
        """Model 2 is clearly worse → low p-value, negative CI."""
        s1 = [0.8, 0.82, 0.79, 0.81, 0.83, 0.80, 0.78, 0.82, 0.81, 0.80]
        s2 = [0.5, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.47, 0.51, 0.50]
        result = calculate_significance(s1, s2)

        assert result.p_value < 0.05
        assert result.statistical_significance > 0.95
        # CI should be entirely negative (model 2 worse)
        assert result.confidence_interval[1] < 0

    def test_two_elements(self):
        """Minimal case for a valid t-test (n=2)."""
        result = calculate_significance([0.5, 0.6], [0.8, 0.9])
        assert 0.0 <= result.p_value <= 1.0
        assert 0.0 <= result.statistical_significance <= 1.0
        # Mean difference is 0.3; CI should contain it (with float tolerance)
        lo, hi = result.confidence_interval
        assert lo <= 0.3 + 1e-9
        assert hi >= 0.3 - 1e-9


class TestResultProperties:
    """Verify invariants on the SignificanceResult."""

    def test_p_value_range(self):
        s1 = [0.5, 0.6, 0.55, 0.58, 0.52]
        s2 = [0.7, 0.75, 0.72, 0.68, 0.71]
        result = calculate_significance(s1, s2)
        assert 0.0 <= result.p_value <= 1.0

    def test_significance_range(self):
        s1 = [0.5, 0.6, 0.55, 0.58, 0.52]
        s2 = [0.7, 0.75, 0.72, 0.68, 0.71]
        result = calculate_significance(s1, s2)
        assert 0.0 <= result.statistical_significance <= 1.0

    def test_significance_equals_one_minus_p(self):
        s1 = [0.5, 0.6, 0.55, 0.58, 0.52]
        s2 = [0.7, 0.75, 0.72, 0.68, 0.71]
        result = calculate_significance(s1, s2)
        assert result.statistical_significance == pytest.approx(
            1.0 - result.p_value
        )

    def test_ci_lower_le_upper(self):
        s1 = [0.5, 0.6, 0.55, 0.58, 0.52]
        s2 = [0.7, 0.75, 0.72, 0.68, 0.71]
        result = calculate_significance(s1, s2)
        assert result.confidence_interval[0] <= result.confidence_interval[1]

    def test_ci_contains_mean_difference(self):
        s1 = [0.5, 0.6, 0.55, 0.58, 0.52]
        s2 = [0.7, 0.75, 0.72, 0.68, 0.71]
        mean_diff = sum(b - a for a, b in zip(s1, s2)) / len(s1)
        result = calculate_significance(s1, s2)
        lo, hi = result.confidence_interval
        assert lo <= mean_diff <= hi

    def test_result_is_frozen_dataclass(self):
        result = calculate_significance([0.5], [0.8])
        assert isinstance(result, SignificanceResult)
        with pytest.raises(AttributeError):
            result.p_value = 0.5  # type: ignore[misc]
