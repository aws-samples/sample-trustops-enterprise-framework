"""Tests for the ReportGenerator.

Requirements: 3.8
"""

import pytest

from src.data_models.hallucination import HallucinationResult
from src.data_models.model import ModelPricing
from src.data_models.trust_score import TrustScoreResult
from src.evaluation.batch_runner import BatchItemResult
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.response_scorer import ScoredResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scored(
    index: int,
    trust_score: float,
    category: str | None = None,
    latency_ms: float = 100.0,
    input_tokens: int = 10,
    output_tokens: int = 20,
    hallucination_rate: float | None = None,
    success: bool = True,
) -> ScoredResponse:
    """Build a minimal ScoredResponse for testing."""
    batch = BatchItemResult(
        index=index,
        prompt=f"prompt-{index}",
        response_text="resp" if success else "",
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        success=success,
        category=category,
    )
    tsr = (
        TrustScoreResult(
            overall_score=trust_score,
            confidence_level=0.9,
        )
        if success
        else None
    )
    hr = None
    if hallucination_rate is not None and success:
        hr = HallucinationResult(
            has_hallucinations=hallucination_rate > 0,
            hallucination_rate=hallucination_rate,
            total_claims=5,
            factual_claims=5,
            supported_claims=int(5 * (1 - hallucination_rate)),
            unsupported_claims=int(5 * hallucination_rate),
            flagged_spans=[],
            overall_grounding_score=1.0 - hallucination_rate,
            claim_evidence=[],
            sensitivity_level="moderate",
        )
    return ScoredResponse(
        batch_item=batch,
        trust_score_result=tsr,
        hallucination_result=hr,
    )


def _pricing(
    inp: float = 0.003, out: float = 0.015
) -> ModelPricing:
    return ModelPricing(
        input_price_per_1k_tokens=inp,
        output_price_per_1k_tokens=out,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReportGenerator:
    """Tests for ReportGenerator.generate_baseline_report."""

    def test_basic_report_structure(self):
        gen = ReportGenerator(
            model_id="test-model",
            dataset_id="ds-001",
        )
        responses = [
            _make_scored(0, 0.8),
            _make_scored(1, 0.6),
        ]
        report = gen.generate_baseline_report(responses)

        assert report.model_id == "test-model"
        assert report.dataset_id == "ds-001"
        assert report.total_examples == 2
        assert report.status == "completed"
        assert report.s3_results_uri.startswith("s3://")
        assert report.completed_at is not None
        assert report.evaluation_id  # non-empty UUID

    def test_aggregate_metrics(self):
        gen = ReportGenerator(
            model_id="m", dataset_id="d"
        )
        responses = [
            _make_scored(0, 0.6, latency_ms=100.0),
            _make_scored(1, 0.8, latency_ms=200.0),
            _make_scored(2, 0.7, latency_ms=150.0),
        ]
        report = gen.generate_baseline_report(responses)
        agg = report.aggregate_metrics

        assert agg.total_examples == 3
        assert agg.successful_examples == 3
        assert agg.failed_examples == 0
        assert agg.mean_trust_score == pytest.approx(0.7, abs=0.01)
        assert agg.median_trust_score == pytest.approx(0.7)

    def test_per_category_metrics(self):
        gen = ReportGenerator(
            model_id="m", dataset_id="d"
        )
        responses = [
            _make_scored(0, 0.9, category="math"),
            _make_scored(1, 0.5, category="science"),
            _make_scored(2, 0.7, category="math"),
        ]
        report = gen.generate_baseline_report(responses)

        assert "math" in report.per_category_metrics
        assert "science" in report.per_category_metrics
        assert report.per_category_metrics["math"].total_examples == 2
        assert report.per_category_metrics["science"].total_examples == 1
        assert report.per_category_metrics[
            "math"
        ].mean_trust_score == pytest.approx(0.8)

    def test_cost_summary_with_pricing(self):
        gen = ReportGenerator(
            model_id="m",
            dataset_id="d",
            pricing=_pricing(inp=1.0, out=2.0),
        )
        responses = [
            _make_scored(
                0, 0.5,
                input_tokens=1000, output_tokens=500,
            ),
        ]
        report = gen.generate_baseline_report(responses)
        cs = report.cost_summary

        # input: 1000 * 1.0/1000 = 1.0
        # output: 500 * 2.0/1000 = 1.0
        assert cs.inference_cost == pytest.approx(2.0)
        assert cs.total_cost == pytest.approx(2.0)
        assert cs.cost_per_example == pytest.approx(2.0)
        assert cs.currency == "USD"

    def test_cost_summary_no_pricing(self):
        """Without explicit pricing, CostCalculator falls back to
        config loader. Cost may be non-zero if model is found in
        config, or zero if not. Either way the report should be valid.
        """
        gen = ReportGenerator(
            model_id="nonexistent-model-xyz-999",
            dataset_id="d",
        )
        responses = [
            _make_scored(
                0, 0.5,
                input_tokens=1000, output_tokens=500,
            ),
        ]
        report = gen.generate_baseline_report(responses)
        assert report.cost_summary.total_cost >= 0.0
        assert report.cost_summary.currency == "USD"

    def test_empty_responses(self):
        gen = ReportGenerator(
            model_id="m", dataset_id="d"
        )
        report = gen.generate_baseline_report([])

        assert report.total_examples == 0
        assert report.aggregate_metrics.total_examples == 0
        assert report.per_category_metrics == {}
        assert report.cost_summary.total_cost == 0.0

    def test_mixed_success_failure(self):
        gen = ReportGenerator(
            model_id="m", dataset_id="d"
        )
        responses = [
            _make_scored(0, 0.9),
            _make_scored(1, 0.0, success=False),
            _make_scored(2, 0.7),
        ]
        report = gen.generate_baseline_report(responses)
        agg = report.aggregate_metrics

        assert agg.total_examples == 3
        assert agg.successful_examples == 2
        assert agg.failed_examples == 1

    def test_custom_s3_uri(self):
        gen = ReportGenerator(
            model_id="m",
            dataset_id="d",
            s3_results_uri="s3://my-bucket/results/eval-1",
        )
        report = gen.generate_baseline_report([_make_scored(0, 0.5)])
        assert report.s3_results_uri == "s3://my-bucket/results/eval-1"

    def test_hallucination_rate_in_aggregate(self):
        gen = ReportGenerator(
            model_id="m", dataset_id="d"
        )
        responses = [
            _make_scored(0, 0.5, hallucination_rate=0.2),
            _make_scored(1, 0.5, hallucination_rate=0.4),
        ]
        report = gen.generate_baseline_report(responses)
        assert report.aggregate_metrics.mean_hallucination_rate == (
            pytest.approx(0.3)
        )
