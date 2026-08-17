"""Lambda handler for generating deploy/iterate/reject recommendation."""

import json
import logging

logger = logging.getLogger(__name__)


def handler(event: dict, context) -> dict:
    for field in ("baseline_metrics", "finetuned_metrics"):
        if field not in event:
            return {
                "statusCode": 400,
                "error": "ValidationError",
                "message": f"Missing required field: {field}",
            }

    baseline = event["baseline_metrics"]
    finetuned = event["finetuned_metrics"]

    baseline_trust = baseline.get("mean_trust_score", 0.0)
    finetuned_trust = finetuned.get("mean_trust_score", 0.0)
    trust_improvement = finetuned_trust - baseline_trust

    baseline_hallucination = baseline.get("hallucination_rate", 0.0)
    finetuned_hallucination = finetuned.get("hallucination_rate", 0.0)
    hallucination_reduction = baseline_hallucination - finetuned_hallucination

    baseline_cost = baseline.get("total_cost", 0.0)
    finetuned_cost = finetuned.get("total_cost", 0.0)
    n_baseline = baseline.get("total_examples", 1)
    n_finetuned = finetuned.get("total_examples", 1)

    cost_per_query_baseline = baseline_cost / max(n_baseline, 1)
    cost_per_query_finetuned = finetuned_cost / max(n_finetuned, 1)
    cost_delta = cost_per_query_finetuned - cost_per_query_baseline
    cost_delta_pct = (
        (cost_delta / cost_per_query_baseline * 100)
        if cost_per_query_baseline > 0
        else 0.0
    )

    statistical_significance = abs(trust_improvement) >= 0.05

    # Use per-example vectors if provided for real statistical testing
    if "baseline_trust_scores" in event and "finetuned_trust_scores" in event:
        try:
            from scipy.stats import ttest_rel

            _, p_value = ttest_rel(
                event["baseline_trust_scores"],
                event["finetuned_trust_scores"],
            )
            statistical_significance = p_value < 0.05
        except Exception as e:
            # Fall back to the effect-size heuristic already computed above
            logger.warning(
                "Paired t-test failed, falling back to threshold heuristic: %s", e
            )

    recommendation, justification = _generate_recommendation(
        trust_improvement, hallucination_reduction, cost_delta_pct, statistical_significance
    )

    improvement_metrics = {
        "baseline_model_id": baseline.get("model_id", ""),
        "finetuned_model_id": finetuned.get("model_id", ""),
        "trust_score_improvement": trust_improvement,
        "hallucination_reduction": hallucination_reduction,
        "cost_delta_per_query": cost_delta,
        "cost_delta_percentage": cost_delta_pct,
        "statistical_significance": statistical_significance,
        "recommendation": recommendation,
        "justification": justification,
    }

    return {
        "improvement_metrics": improvement_metrics,
        "recommendation": recommendation,
        "justification": justification,
    }


def _generate_recommendation(
    trust_improvement: float,
    hallucination_reduction: float,
    cost_delta_pct: float,
    statistical_significance: bool,
) -> tuple[str, str]:
    if not statistical_significance:
        return (
            "iterate",
            "Trust score improvement is not statistically significant. "
            "Consider additional fine-tuning iterations.",
        )

    if trust_improvement >= 0.1 and hallucination_reduction >= 0.05:
        if cost_delta_pct <= 20:
            return (
                "deploy",
                f"Significant trust improvement ({trust_improvement:.2%}) "
                f"and hallucination reduction ({hallucination_reduction:.2%}) "
                f"with acceptable cost ({cost_delta_pct:.1f}%).",
            )
        return (
            "iterate",
            f"Good quality improvements but cost increase ({cost_delta_pct:.1f}%) "
            "is too high. Consider optimization.",
        )

    if trust_improvement >= 0.05:
        return (
            "iterate",
            f"Moderate trust improvement ({trust_improvement:.2%}). "
            "Consider additional fine-tuning.",
        )

    return (
        "reject",
        f"Insufficient improvement (trust: {trust_improvement:.2%}, "
        f"hallucination: {hallucination_reduction:.2%}). "
        "Fine-tuning did not provide meaningful benefits.",
    )
