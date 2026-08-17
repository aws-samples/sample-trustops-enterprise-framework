"""Statistical significance testing for paired model comparisons."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy import stats


@dataclass
class StatisticalTestResult:
    p_value_ttest: float
    p_value_wilcoxon: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    test_type_used: str
    mean_difference: float
    sample_size: int
    insufficient_data: bool


class StatisticalTestEngine:

    def run_paired_tests(
        self,
        vector_a: list[float],
        vector_b: list[float],
        confidence_level: float = 0.95,
    ) -> StatisticalTestResult:
        if len(vector_a) != len(vector_b):
            raise ValueError("Vectors must have equal length")

        n = len(vector_a)

        if n < 3:
            return StatisticalTestResult(
                p_value_ttest=1.0,
                p_value_wilcoxon=1.0,
                confidence_interval_lower=0.0,
                confidence_interval_upper=0.0,
                test_type_used="none",
                mean_difference=0.0,
                sample_size=n,
                insufficient_data=True,
            )

        differences = [a - b for a, b in zip(vector_a, vector_b)]
        mean_diff = sum(differences) / n

        # Paired t-test
        _, p_ttest = stats.ttest_rel(vector_a, vector_b)

        # Wilcoxon signed-rank test
        try:
            _, p_wilcoxon = stats.wilcoxon(differences)
        except ValueError:
            p_wilcoxon = 1.0

        # Select primary test
        test_type = self._select_primary_test(n)

        # Confidence interval
        ci_lower, ci_upper = self._compute_confidence_interval(
            differences, confidence_level
        )

        return StatisticalTestResult(
            p_value_ttest=float(p_ttest),
            p_value_wilcoxon=float(p_wilcoxon),
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            test_type_used=test_type,
            mean_difference=mean_diff,
            sample_size=n,
            insufficient_data=False,
        )

    def _select_primary_test(self, n: int) -> str:
        return "wilcoxon" if n < 20 else "ttest"

    def _compute_confidence_interval(
        self,
        differences: list[float],
        confidence_level: float,
    ) -> tuple[float, float]:
        n = len(differences)
        mean_diff = sum(differences) / n
        std_diff = sqrt(
            sum((d - mean_diff) ** 2 for d in differences) / (n - 1)
        )
        se = std_diff / sqrt(n)

        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

        margin = t_critical * se
        return (mean_diff - margin, mean_diff + margin)
