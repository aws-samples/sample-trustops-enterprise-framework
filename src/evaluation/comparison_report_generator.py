"""
Comparison Report Generator for comparative model evaluation.

Assembles a ComparisonReport dataclass from pre-computed aggregate metrics,
per-category metrics, improvement metrics, deployment recommendation,
and cost-performance analysis.  Also generates per-category improvement
metrics breakdowns.

Requirements: 7.7
"""

import uuid
from datetime import datetime, timezone

from config.aws_config import config as aws_config
from src.data_models.evaluation import (
    AggregateMetrics,
    ComparisonReport,
    CostPerformanceAnalysis,
    DeploymentRecommendation,
    ImprovementMetrics,
)
from src.evaluation.improvement_calculator import calculate_improvement_metrics


def generate_per_category_breakdown(
    model_1_per_category: dict[str, AggregateMetrics],
    model_2_per_category: dict[str, AggregateMetrics],
) -> dict[str, ImprovementMetrics]:
    """Generate per-category improvement metrics breakdown.

    For each category present in *both* models' per-category metrics,
    calculates improvement metrics using the same logic as the overall
    improvement calculator.

    Categories present in only one model are skipped.

    Args:
        model_1_per_category: Per-category aggregate metrics for model 1.
        model_2_per_category: Per-category aggregate metrics for model 2.

    Returns:
        A dict mapping category name to ImprovementMetrics.
    """
    breakdown: dict[str, ImprovementMetrics] = {}
    common_categories = set(model_1_per_category) & set(model_2_per_category)

    for category in sorted(common_categories):
        breakdown[category] = calculate_improvement_metrics(
            model_1_per_category[category],
            model_2_per_category[category],
        )

    return breakdown


def generate_comparison_report(
    model_1_id: str,
    model_2_id: str,
    dataset_id: str,
    model_1_metrics: AggregateMetrics,
    model_2_metrics: AggregateMetrics,
    improvement_metrics: ImprovementMetrics,
    recommendation: DeploymentRecommendation,
    recommendation_justification: str,
    per_category_breakdown: dict[str, ImprovementMetrics],
    cost_performance_analysis: CostPerformanceAnalysis,
    s3_report_uri: str | None = None,
) -> ComparisonReport:
    """Assemble a ComparisonReport from pre-computed components.

    Args:
        model_1_id: ID of the baseline model.
        model_2_id: ID of the comparison model.
        dataset_id: ID of the evaluation dataset.
        model_1_metrics: Aggregate metrics for model 1.
        model_2_metrics: Aggregate metrics for model 2.
        improvement_metrics: Overall improvement metrics.
        recommendation: Deployment recommendation enum value.
        recommendation_justification: Human-readable justification.
        per_category_breakdown: Per-category improvement metrics.
        cost_performance_analysis: Cost-performance analysis.
        s3_report_uri: Optional S3 URI for the report. Auto-generated
            if not provided.

    Returns:
        A fully populated ComparisonReport.
    """
    comparison_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Derive the bucket from configuration rather than a hardcoded name; a
    # predictable literal could be squatted, and it would also record a URI
    # that does not exist in this account.
    if s3_report_uri is None:
        s3_report_uri = (
            f"s3://{aws_config.results_bucket}/comparisons/"
            f"{comparison_id}/{model_1_id}_vs_{model_2_id}/"
            f"{now.isoformat()}"
        )

    return ComparisonReport(
        comparison_id=comparison_id,
        model_1_id=model_1_id,
        model_2_id=model_2_id,
        dataset_id=dataset_id,
        model_1_metrics=model_1_metrics,
        model_2_metrics=model_2_metrics,
        improvement_metrics=improvement_metrics,
        recommendation=recommendation,
        recommendation_justification=recommendation_justification,
        per_category_breakdown=per_category_breakdown,
        cost_performance_analysis=cost_performance_analysis,
        created_at=now,
        s3_report_uri=s3_report_uri,
    )
