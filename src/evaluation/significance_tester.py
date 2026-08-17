"""
Statistical significance tester for paired trust score comparisons.

Computes p-values and confidence intervals for the mean difference between
two paired lists of trust scores using scipy's paired t-test, with a
fallback to a simple implementation when scipy is unavailable.

Requirements: 7.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from scipy import stats as scipy_stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


@dataclass(frozen=True)
class SignificanceResult:
    """Result of a statistical significance test.

    Attributes:
        p_value: Two-sided p-value from the paired test, in [0, 1].
        confidence_interval: 95 % confidence interval for the mean
            difference (lower, upper).
        statistical_significance: 1 - p_value, clamped to [0, 1].
    """

    p_value: float
    confidence_interval: tuple[float, float]
    statistical_significance: float


def _t_critical_approx(df: int) -> float:
    """Return an approximate two-tailed t critical value for 95 % CI.

    Uses a rough lookup for small degrees of freedom and falls back to
    1.96 (normal approximation) for df >= 30.  This is only used when
    scipy is not installed.
    """
    # Selected values from the t-distribution table (two-tailed, alpha=0.05)
    table: dict[int, float] = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        25: 2.060,
        29: 2.045,
    }
    if df in table:
        return table[df]
    # For df > 29 use normal approximation
    if df >= 30:
        return 1.96
    # Linear interpolation between known values for gaps
    lower_keys = [k for k in sorted(table) if k <= df]
    upper_keys = [k for k in sorted(table) if k >= df]
    if lower_keys and upper_keys:
        lo = lower_keys[-1]
        hi = upper_keys[0]
        if lo == hi:
            return table[lo]
        frac = (df - lo) / (hi - lo)
        return table[lo] + frac * (table[hi] - table[lo])
    return 1.96  # safe fallback


def _fallback_paired_ttest(
    diffs: list[float],
) -> tuple[float, float, float]:
    """Simple paired t-test without scipy.

    Returns (t_statistic, p_value_approx, standard_error).
    The p-value is approximated as 2-tailed using the normal CDF
    approximation for |t| (accurate for n >= 30, rough otherwise).
    """
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = math.sqrt(var_d) if var_d > 0 else 0.0
    se = std_d / math.sqrt(n)

    if se == 0.0:
        # All differences are identical
        if mean_d == 0.0:
            return 0.0, 1.0, 0.0
        # Infinite t → p ≈ 0
        return float("inf"), 0.0, 0.0

    t_stat = mean_d / se

    # Approximate p-value using the complementary error function
    # P(|T| > |t|) ≈ 2 * Φ(-|t|) for large df
    abs_t = abs(t_stat)
    p_approx = math.erfc(abs_t / math.sqrt(2))  # two-tailed
    p_approx = max(0.0, min(1.0, p_approx))

    return t_stat, p_approx, se


def calculate_significance(
    scores_1: list[float],
    scores_2: list[float],
    alpha: float = 0.05,
) -> SignificanceResult:
    """Compute statistical significance for paired trust score differences.

    Args:
        scores_1: Trust scores from model 1 (baseline), one per example.
        scores_2: Trust scores from model 2 (comparison), one per example.
            Must be the same length as *scores_1*.
        alpha: Significance level for the confidence interval (default 0.05
            gives a 95 % CI).

    Returns:
        A ``SignificanceResult`` with p_value, confidence_interval, and
        statistical_significance fields.

    Edge cases:
        * Empty lists → p_value=1.0, CI=(0.0, 0.0), significance=0.0
        * Single element → p_value=1.0, CI uses the single difference as
          both bounds.
        * Identical scores → p_value=1.0, CI=(0.0, 0.0), significance=0.0
    """
    if len(scores_1) != len(scores_2):
        raise ValueError(
            f"Score lists must have equal length, got "
            f"{len(scores_1)} and {len(scores_2)}"
        )

    n = len(scores_1)

    # --- Edge case: empty lists ---
    if n == 0:
        return SignificanceResult(
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            statistical_significance=0.0,
        )

    diffs = [s2 - s1 for s1, s2 in zip(scores_1, scores_2)]

    # --- Edge case: single element ---
    if n == 1:
        d = diffs[0]
        return SignificanceResult(
            p_value=1.0,
            confidence_interval=(d, d),
            statistical_significance=0.0,
        )

    # --- Edge case: all differences identical (zero variance) ---
    mean_d = sum(diffs) / n
    if all(d == diffs[0] for d in diffs):
        if diffs[0] == 0.0:
            # Scores are identical
            return SignificanceResult(
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                statistical_significance=0.0,
            )
        # Non-zero but constant difference → perfectly significant
        return SignificanceResult(
            p_value=0.0,
            confidence_interval=(mean_d, mean_d),
            statistical_significance=1.0,
        )

    # --- Main path: paired t-test ---
    if _HAS_SCIPY:
        t_result = scipy_stats.ttest_rel(scores_2, scores_1)
        p_value: float = float(t_result.pvalue)

        # Confidence interval for the mean difference
        se = float(scipy_stats.sem(diffs))
        df = n - 1
        t_crit = float(scipy_stats.t.ppf(1 - alpha / 2, df))
        ci_lower = mean_d - t_crit * se
        ci_upper = mean_d + t_crit * se
    else:  # pragma: no cover
        _t_stat, p_value, se = _fallback_paired_ttest(diffs)
        df = n - 1
        t_crit = _t_critical_approx(df)
        ci_lower = mean_d - t_crit * se
        ci_upper = mean_d + t_crit * se

    p_value = max(0.0, min(1.0, p_value))
    significance = max(0.0, min(1.0, 1.0 - p_value))

    return SignificanceResult(
        p_value=p_value,
        confidence_interval=(ci_lower, ci_upper),
        statistical_significance=significance,
    )
