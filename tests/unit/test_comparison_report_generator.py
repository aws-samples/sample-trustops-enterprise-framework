"""Tests for the comparison report generator.

Requirements: 7.7
"""

import pytest

from src.data_models.evaluation import (
    AggregateMetrics,
    ComparisonReport,
    CostPerformanceAnalysis,
    DeploymentRecommendation,
    ImprovementMetrics,
)
from src.evaluation.comparison_report_generator import (
    generate_comparison_report,
    generate_per_category_breakdown,
)


def _make_metrics(
    mean_trust_score: float = 0.7,
    cost_per_query: float = 0.01,
    mean_hallucination_rate: float = 0.2,
    latency_p50_ms: float = 100.0,
) -> AggregateMetrics:
    """Helper to build AggregateMetrics with sensible defaults."""
    return AggregateMetrics(
        total_examples=100,
        successful_examples=100,
        failed_examples=0,
        mean_trust_score=mean_trust_score,
        median_trust_score=mean_trust_score,
        trust_score_std=0.05,
        mean_hallucination_rate=mean_hallucination_rate,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=150.0,
        latency_p99_ms=200.0,
        total_input_tokens=10000,
        total_output_tokens=5000,
        total_cost=cost_per_query * 100,
        cost_per_query=cost_per_query,
    )


def _make_improvement_metrics(
    trust_score_delta: float = 0.1,
) -> ImprovementMetrics:
    """Helper to build ImprovementMetrics with sensible defaults."""
    return ImprovementMetrics(
        trust_score_delta=trust_score_delta,
        trust_score_delta_percent=14.3,
        hallucination_reduction=0.05,
        hallucination_reduction_percent=25.0,
        latency_delta_ms=-10.0,
        latency_delta_percent=-10.0,
        cost_delta_per_query=0.005,
        cost_delta_percent=50.0,
        statistical_significance=0.98,
        p_value=0.02,
        confidence_interval=(0.05, 0.15),
    )


def _make_cost_performance() -> CostPerformanceAnalysis:
    """Helper to build CostPerformanceAnalysis."""
    return CostPerformanceAnalysis(
        model_1_cost_per_trust_point=0.0143,
        model_2_cost_per_trust_point=0.0188,
        quality_gain_justifies_cost=True,
        break_even_volume=5,
    )


# ── generate_comparison_report ─────────────────────────────────────


class TestGenerateComparisonReport:
    def test_returns_comparison_report_type(self):
        report = generate_comparison_report(
            model_1_id="model-a",
            model_2_id="model-b",
            dataset_id="ds-001",
            model_1_metrics=_make_metrics(mean_trust_score=0.7),
            model_2_metrics=_make_metrics(mean_trust_score=0.8),
            improvement_metrics=_make_improvement_metrics(),
            recommendation=DeploymentRecommendation.DEPLOY,
            recommendation_justification="All thresholds met.",
            per_category_breakdown={},
            cost_performance_analysis=_make_cost_performance(),
        )
        assert isinstance(report, ComparisonReport)

    def test_model_ids_set_correctly(self):
        report = generate_comparison_report(
            model_1_id="baseline-model",
            model_2_id="finetuned-model",
            dataset_id="ds-002",
            model_1_metrics=_make_metrics(),
            model_2_metrics=_make_metrics(),
            improvement_metrics=_make_improvement_metrics(),
            recommendation=DeploymentRecommendation.ITERATE,
            recommendation_justification="Partial improvement.",
            per_category_breakdown={},
            cost_performance_analysis=_make_cost_performance(),
        )
        assert report.model_1_id == "baseline-model"
        assert report.model_2_id == "finetuned-model"
        assert report.dataset_id == "ds-002"

    def test_comparison_id_is_uuid(self):
        report = generate_comparison_report(
            model_1_id="m1",
            model_2_id="m2",
            dataset_id="ds",
            model_1_metrics=_make_metrics(),
            model_2_metrics=_make_metrics(),
            improvement_metrics=_make_improvement_metrics(),
            recommendation=DeploymentRecommendation.REJECT,
            recommendation_justification="Regression.",
            per_category_breakdown={},
            cost_performance_analysis=_make_cost_performance(),
        )
        # UUID4 format: 8-4-4-4-12 hex chars
        assert len(report.comparison_id) == 36
        assert report.comparison_id.count("-") == 4

    def test_s3_report_uri_auto_generated(self):
        report = generate_comparison_report(
            model_1_id="m1",
            model_2_id="m2",
            dataset_id="ds",
            model_1_metrics=_make_metrics(),
            model_2_metrics=_make_metrics(),
            improvement_metrics=_make_improvement_metrics(),
            recommendation=DeploymentRecommendation.DEPLOY,
            recommendation_justification="OK",
            per_category_breakdown={},
            cost_performance_analysis=_make_cost_performance(),
        )
        assert report.s3_report_uri.startswith("s3://")
        assert "m1_vs_m2" in report.s3_report_uri

    def test_custom_s3_report_uri(self):
        uri = "s3://my-bucket/reports/comparison-123"
        report = generate_comparison_report(
            model_1_id="m1",
            model_2_id="m2",
            dataset_id="ds",
            model_1_metrics=_make_metrics(),
            model_2_metrics=_make_metrics(),
            improvement_metrics=_make_improvement_metrics(),
            recommendation=DeploymentRecommendation.DEPLOY,
            recommendation_justification="OK",
            per_category_breakdown={},
            cost_performance_analysis=_make_cost_performance(),
            s3_report_uri=uri,
        )
        assert report.s3_report_uri == uri

    def test_created_at_is_set(self):
        report = generate_comparison_report(
            model_1_id="m1",
            model_2_id="m2",
            dataset_id="ds",
            model_1_metrics=_make_metrics(),
            model_2_metrics=_make_metrics(),
            improvement_metrics=_make_improvement_metrics(),
            recommendation=DeploymentRecommendation.DEPLOY,
            recommendation_justification="OK",
            per_category_breakdown={},
            cost_performance_analysis=_make_cost_performance(),
        )
        assert report.created_at is not None

    def test_metrics_passed_through(self):
        m1 = _make_metrics(mean_trust_score=0.6)
        m2 = _make_metrics(mean_trust_score=0.85)
        report = generate_comparison_report(
            model_1_id="m1",
            model_2_id="m2",
            dataset_id="ds",
            model_1_metrics=m1,
            model_2_metrics=m2,
            improvement_metrics=_make_improvement_metrics(),
            recommendation=DeploymentRecommendation.DEPLOY,
            recommendation_justification="OK",
            per_category_breakdown={},
            cost_performance_analysis=_make_cost_performance(),
        )
        assert report.model_1_metrics.mean_trust_score == pytest.approx(0.6)
        assert report.model_2_metrics.mean_trust_score == pytest.approx(0.85)

    def test_recommendation_and_justification(self):
        report = generate_comparison_report(
            model_1_id="m1",
            model_2_id="m2",
            dataset_id="ds",
            model_1_metrics=_make_metrics(),
            model_2_metrics=_make_metrics(),
            improvement_metrics=_make_improvement_metrics(),
            recommendation=DeploymentRecommendation.REJECT,
            recommendation_justification="Trust score decreased.",
            per_category_breakdown={},
            cost_performance_analysis=_make_cost_performance(),
        )
        assert report.recommendation == DeploymentRecommendation.REJECT
        assert report.recommendation_justification == "Trust score decreased."

    def test_per_category_breakdown_included(self):
        cat_breakdown = {
            "legal": _make_improvement_metrics(trust_score_delta=0.15),
        }
        report = generate_comparison_report(
            model_1_id="m1",
            model_2_id="m2",
            dataset_id="ds",
            model_1_metrics=_make_metrics(),
            model_2_metrics=_make_metrics(),
            improvement_metrics=_make_improvement_metrics(),
            recommendation=DeploymentRecommendation.DEPLOY,
            recommendation_justification="OK",
            per_category_breakdown=cat_breakdown,
            cost_performance_analysis=_make_cost_performance(),
        )
        assert "legal" in report.per_category_breakdown
        assert report.per_category_breakdown["legal"].trust_score_delta == pytest.approx(0.15)

    def test_cost_performance_analysis_included(self):
        cpa = _make_cost_performance()
        report = generate_comparison_report(
            model_1_id="m1",
            model_2_id="m2",
            dataset_id="ds",
            model_1_metrics=_make_metrics(),
            model_2_metrics=_make_metrics(),
            improvement_metrics=_make_improvement_metrics(),
            recommendation=DeploymentRecommendation.DEPLOY,
            recommendation_justification="OK",
            per_category_breakdown={},
            cost_performance_analysis=cpa,
        )
        assert report.cost_performance_analysis.quality_gain_justifies_cost is True
        assert report.cost_performance_analysis.break_even_volume == 5


# ── generate_per_category_breakdown ────────────────────────────────


class TestGeneratePerCategoryBreakdown:
    def test_common_categories_computed(self):
        m1_cats = {
            "legal": _make_metrics(mean_trust_score=0.6, cost_per_query=0.01),
            "medical": _make_metrics(mean_trust_score=0.5, cost_per_query=0.02),
        }
        m2_cats = {
            "legal": _make_metrics(mean_trust_score=0.8, cost_per_query=0.015),
            "medical": _make_metrics(mean_trust_score=0.7, cost_per_query=0.025),
        }
        result = generate_per_category_breakdown(m1_cats, m2_cats)
        assert "legal" in result
        assert "medical" in result
        assert result["legal"].trust_score_delta == pytest.approx(0.2)
        assert result["medical"].trust_score_delta == pytest.approx(0.2)

    def test_disjoint_categories_skipped(self):
        m1_cats = {
            "legal": _make_metrics(mean_trust_score=0.6),
        }
        m2_cats = {
            "medical": _make_metrics(mean_trust_score=0.7),
        }
        result = generate_per_category_breakdown(m1_cats, m2_cats)
        assert result == {}

    def test_partial_overlap(self):
        m1_cats = {
            "legal": _make_metrics(mean_trust_score=0.6),
            "finance": _make_metrics(mean_trust_score=0.5),
        }
        m2_cats = {
            "legal": _make_metrics(mean_trust_score=0.75),
            "medical": _make_metrics(mean_trust_score=0.8),
        }
        result = generate_per_category_breakdown(m1_cats, m2_cats)
        assert "legal" in result
        assert "finance" not in result
        assert "medical" not in result

    def test_empty_categories(self):
        result = generate_per_category_breakdown({}, {})
        assert result == {}

    def test_hallucination_reduction_computed(self):
        m1_cats = {
            "qa": _make_metrics(mean_hallucination_rate=0.3),
        }
        m2_cats = {
            "qa": _make_metrics(mean_hallucination_rate=0.1),
        }
        result = generate_per_category_breakdown(m1_cats, m2_cats)
        # hallucination_reduction = m1 - m2 = 0.3 - 0.1 = 0.2
        assert result["qa"].hallucination_reduction == pytest.approx(0.2)

    def test_latency_delta_computed(self):
        m1_cats = {
            "chat": _make_metrics(latency_p50_ms=100.0),
        }
        m2_cats = {
            "chat": _make_metrics(latency_p50_ms=80.0),
        }
        result = generate_per_category_breakdown(m1_cats, m2_cats)
        # latency_delta = m2 - m1 = 80 - 100 = -20 (faster)
        assert result["chat"].latency_delta_ms == pytest.approx(-20.0)
