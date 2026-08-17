"""
Deployment Recommendation Engine for comparative model evaluation.

Takes ImprovementMetrics and DeploymentThresholds and produces a
deployment recommendation (DEPLOY, ITERATE, REJECT) with a human-readable
justification explaining the decision.

Decision logic:
- REJECT: trust score decreased (trust_score_delta < 0) OR hallucination
  rate increased (hallucination_reduction < 0).
- DEPLOY: all thresholds met — trust improvement >= min, cost increase
  <= max, and hallucination reduction >= min.
- ITERATE: all other cases (partial improvement, not yet meeting all
  thresholds).

Requirements: 7.6
"""

from src.data_models.evaluation import (
    DeploymentRecommendation,
    DeploymentThresholds,
    ImprovementMetrics,
)


def generate_recommendation(
    metrics: ImprovementMetrics,
    thresholds: DeploymentThresholds,
) -> tuple[DeploymentRecommendation, str]:
    """Generate a deployment recommendation with justification.

    Args:
        metrics: Improvement metrics comparing two models.
        thresholds: Configurable thresholds for the recommendation.

    Returns:
        A tuple of (DeploymentRecommendation, justification_string).
    """
    # --- REJECT: trust decreased or hallucination rate increased ----------
    reject_reasons: list[str] = []

    if metrics.trust_score_delta < 0:
        reject_reasons.append(
            f"Trust score decreased by {abs(metrics.trust_score_delta):.4f} "
            f"({abs(metrics.trust_score_delta_percent):.1f}%)"
        )

    if metrics.hallucination_reduction < 0:
        reject_reasons.append(
            f"Hallucination rate increased by "
            f"{abs(metrics.hallucination_reduction):.4f} "
            f"({abs(metrics.hallucination_reduction_percent):.1f}%)"
        )

    if reject_reasons:
        justification = (
            "REJECT: Model performance regressed. "
            + "; ".join(reject_reasons)
            + "."
        )
        return DeploymentRecommendation.REJECT, justification

    # --- DEPLOY: all thresholds met ---------------------------------------
    trust_met = (
        metrics.trust_score_delta >= thresholds.min_trust_score_improvement
    )
    cost_met = (
        metrics.cost_delta_percent <= thresholds.max_cost_increase_percent
    )
    hallucination_met = (
        metrics.hallucination_reduction
        >= thresholds.min_hallucination_reduction
    )

    if trust_met and cost_met and hallucination_met:
        min_trust = thresholds.min_trust_score_improvement
        max_cost = thresholds.max_cost_increase_percent
        min_hall = thresholds.min_hallucination_reduction
        justification = (
            "DEPLOY: All deployment thresholds met. "
            f"Trust score improved by "
            f"{metrics.trust_score_delta:.4f} "
            f"(>= {min_trust:.4f}), "
            f"cost change is "
            f"{metrics.cost_delta_percent:+.1f}% "
            f"(<= {max_cost:.1f}%), "
            f"hallucination reduction is "
            f"{metrics.hallucination_reduction:.4f} "
            f"(>= {min_hall:.4f})."
        )
        return DeploymentRecommendation.DEPLOY, justification

    # --- ITERATE: partial improvement, not all thresholds met -------------
    unmet: list[str] = []
    met: list[str] = []

    min_trust = thresholds.min_trust_score_improvement
    max_cost = thresholds.max_cost_increase_percent
    min_hall = thresholds.min_hallucination_reduction

    if trust_met:
        met.append(
            f"trust improvement "
            f"{metrics.trust_score_delta:.4f} "
            f"(>= {min_trust:.4f})"
        )
    else:
        unmet.append(
            f"trust improvement "
            f"{metrics.trust_score_delta:.4f} "
            f"(needs >= {min_trust:.4f})"
        )

    if cost_met:
        met.append(
            f"cost change "
            f"{metrics.cost_delta_percent:+.1f}% "
            f"(<= {max_cost:.1f}%)"
        )
    else:
        unmet.append(
            f"cost increase "
            f"{metrics.cost_delta_percent:+.1f}% "
            f"(needs <= {max_cost:.1f}%)"
        )

    if hallucination_met:
        met.append(
            f"hallucination reduction "
            f"{metrics.hallucination_reduction:.4f} "
            f"(>= {min_hall:.4f})"
        )
    else:
        unmet.append(
            f"hallucination reduction "
            f"{metrics.hallucination_reduction:.4f} "
            f"(needs >= {min_hall:.4f})"
        )

    parts: list[str] = ["ITERATE: Some thresholds not yet met."]
    if met:
        parts.append("Met: " + "; ".join(met) + ".")
    if unmet:
        parts.append("Not met: " + "; ".join(unmet) + ".")

    justification = " ".join(parts)
    return DeploymentRecommendation.ITERATE, justification
