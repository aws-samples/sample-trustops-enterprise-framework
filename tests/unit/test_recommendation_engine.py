"""Tests for the deployment recommendation engine.

Requirements: 7.6
"""

from src.data_models.evaluation import (
    DeploymentRecommendation,
    DeploymentThresholds,
    ImprovementMetrics,
)
from src.evaluation.recommendation_engine import (
    generate_recommendation,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_metrics(
    trust_score_delta: float = 0.0,
    trust_score_delta_percent: float = 0.0,
    hallucination_reduction: float = 0.0,
    hallucination_reduction_percent: float = 0.0,
    cost_delta_percent: float = 0.0,
    cost_delta_per_query: float = 0.0,
    latency_delta_ms: float = 0.0,
    latency_delta_percent: float = 0.0,
    statistical_significance: float = 0.95,
    p_value: float = 0.05,
    confidence_interval: tuple[float, float] = (0.0, 0.1),
) -> ImprovementMetrics:
    return ImprovementMetrics(
        trust_score_delta=trust_score_delta,
        trust_score_delta_percent=trust_score_delta_percent,
        hallucination_reduction=hallucination_reduction,
        hallucination_reduction_percent=(
            hallucination_reduction_percent
        ),
        cost_delta_percent=cost_delta_percent,
        cost_delta_per_query=cost_delta_per_query,
        latency_delta_ms=latency_delta_ms,
        latency_delta_percent=latency_delta_percent,
        statistical_significance=statistical_significance,
        p_value=p_value,
        confidence_interval=confidence_interval,
    )


_THRESH = DeploymentThresholds()


def _recommend(metrics):
    """Shorthand for generate_recommendation with defaults."""
    return generate_recommendation(metrics, _THRESH)


# -------------------------------------------------------------------
# DEPLOY tests
# -------------------------------------------------------------------

class TestDeploy:
    """DEPLOY when all thresholds are met."""

    def test_all_thresholds_met(self):
        m = _make_metrics(
            trust_score_delta=0.10,
            hallucination_reduction=0.15,
            cost_delta_percent=5.0,
        )
        rec, justification = _recommend(m)
        assert rec == DeploymentRecommendation.DEPLOY
        assert "DEPLOY" in justification

    def test_exact_threshold_values(self):
        """Boundary: exactly at threshold should DEPLOY."""
        m = _make_metrics(
            trust_score_delta=0.05,
            hallucination_reduction=0.1,
            cost_delta_percent=20.0,
        )
        rec, _ = _recommend(m)
        assert rec == DeploymentRecommendation.DEPLOY

    def test_cost_decrease_deploys(self):
        """Negative cost (cheaper) satisfies cost threshold."""
        m = _make_metrics(
            trust_score_delta=0.06,
            hallucination_reduction=0.12,
            cost_delta_percent=-10.0,
        )
        rec, _ = _recommend(m)
        assert rec == DeploymentRecommendation.DEPLOY

    def test_custom_thresholds(self):
        thresholds = DeploymentThresholds(
            min_trust_score_improvement=0.01,
            max_cost_increase_percent=50.0,
            min_hallucination_reduction=0.02,
        )
        m = _make_metrics(
            trust_score_delta=0.02,
            hallucination_reduction=0.03,
            cost_delta_percent=30.0,
        )
        rec, _ = generate_recommendation(m, thresholds)
        assert rec == DeploymentRecommendation.DEPLOY

    def test_justification_includes_values(self):
        m = _make_metrics(
            trust_score_delta=0.10,
            hallucination_reduction=0.15,
            cost_delta_percent=5.0,
        )
        _, justification = _recommend(m)
        assert "0.1000" in justification
        assert "0.1500" in justification


# -------------------------------------------------------------------
# REJECT tests
# -------------------------------------------------------------------

class TestReject:
    """REJECT when trust decreased or hallucination increased."""

    def test_trust_decreased(self):
        m = _make_metrics(
            trust_score_delta=-0.05,
            trust_score_delta_percent=-5.0,
        )
        rec, justification = _recommend(m)
        assert rec == DeploymentRecommendation.REJECT
        assert "Trust score decreased" in justification

    def test_hallucination_increased(self):
        m = _make_metrics(
            trust_score_delta=0.02,
            hallucination_reduction=-0.08,
            hallucination_reduction_percent=-10.0,
        )
        rec, justification = _recommend(m)
        assert rec == DeploymentRecommendation.REJECT
        assert "Hallucination rate increased" in justification

    def test_both_regressed(self):
        """Both trust decreased and hallucination increased."""
        m = _make_metrics(
            trust_score_delta=-0.03,
            trust_score_delta_percent=-3.0,
            hallucination_reduction=-0.05,
            hallucination_reduction_percent=-5.0,
        )
        rec, justification = _recommend(m)
        assert rec == DeploymentRecommendation.REJECT
        assert "Trust score decreased" in justification
        txt = "Hallucination rate increased"
        assert txt in justification

    def test_reject_overrides_good_cost(self):
        """Even with cost savings, regression means REJECT."""
        m = _make_metrics(
            trust_score_delta=-0.01,
            cost_delta_percent=-50.0,
        )
        rec, _ = _recommend(m)
        assert rec == DeploymentRecommendation.REJECT


# -------------------------------------------------------------------
# ITERATE tests
# -------------------------------------------------------------------

class TestIterate:
    """ITERATE: no regression but not all thresholds met."""

    def test_trust_below_threshold(self):
        m = _make_metrics(
            trust_score_delta=0.02,
            hallucination_reduction=0.15,
            cost_delta_percent=5.0,
        )
        rec, justification = _recommend(m)
        assert rec == DeploymentRecommendation.ITERATE
        assert "trust" in justification.lower()

    def test_cost_above_threshold(self):
        m = _make_metrics(
            trust_score_delta=0.10,
            hallucination_reduction=0.15,
            cost_delta_percent=25.0,
        )
        rec, justification = _recommend(m)
        assert rec == DeploymentRecommendation.ITERATE
        assert "cost" in justification.lower()

    def test_hallucination_below_threshold(self):
        m = _make_metrics(
            trust_score_delta=0.10,
            hallucination_reduction=0.05,
            cost_delta_percent=5.0,
        )
        rec, justification = _recommend(m)
        assert rec == DeploymentRecommendation.ITERATE
        assert "hallucination" in justification.lower()

    def test_all_below_threshold(self):
        """No regression, but nothing meets threshold."""
        m = _make_metrics(
            trust_score_delta=0.01,
            hallucination_reduction=0.02,
            cost_delta_percent=25.0,
        )
        rec, justification = _recommend(m)
        assert rec == DeploymentRecommendation.ITERATE
        assert "Not met" in justification

    def test_zero_deltas(self):
        """No change → ITERATE (no regression)."""
        m = _make_metrics()
        rec, _ = _recommend(m)
        assert rec == DeploymentRecommendation.ITERATE

    def test_justification_lists_met_and_unmet(self):
        m = _make_metrics(
            trust_score_delta=0.10,
            hallucination_reduction=0.15,
            cost_delta_percent=25.0,
        )
        _, justification = _recommend(m)
        assert "Met:" in justification
        assert "Not met:" in justification


# -------------------------------------------------------------------
# Justification quality
# -------------------------------------------------------------------

class TestJustification:
    """Justification text is human-readable."""

    def test_deploy_justification_not_empty(self):
        m = _make_metrics(
            trust_score_delta=0.10,
            hallucination_reduction=0.15,
            cost_delta_percent=5.0,
        )
        _, justification = _recommend(m)
        assert len(justification) > 0

    def test_reject_justification_not_empty(self):
        m = _make_metrics(trust_score_delta=-0.05)
        _, justification = _recommend(m)
        assert len(justification) > 0

    def test_iterate_justification_not_empty(self):
        m = _make_metrics(trust_score_delta=0.02)
        _, justification = _recommend(m)
        assert len(justification) > 0

    def test_return_type(self):
        m = _make_metrics(
            trust_score_delta=0.10,
            hallucination_reduction=0.15,
            cost_delta_percent=5.0,
        )
        result = _recommend(m)
        assert isinstance(result, tuple)
        assert len(result) == 2
        rec, justification = result
        assert isinstance(rec, DeploymentRecommendation)
        assert isinstance(justification, str)
