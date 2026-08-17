"""
Baseline Report Generator for the Evaluation Engine.

Compiles aggregate metrics, per-category metrics, and cost summary
from scored responses into a structured BaselineEvaluationReport.

Requirements: 3.8
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from config.aws_config import config as aws_config
from src.data_models.evaluation import (
    AggregateMetrics,
    BaselineEvaluationReport,
    CostSummary,
    EvaluationConfig,
)
from src.data_models.model import InferenceRequest, ModelPricing
from src.evaluation.cost_calculator import CostBreakdown, CostCalculator
from src.evaluation.metrics_calculator import (
    CalculatedMetrics,
    MetricsCalculator,
    PricingConfig,
)
from src.evaluation.response_scorer import ScoredResponse


def _to_aggregate_metrics(cm: CalculatedMetrics) -> AggregateMetrics:
    """Convert internal CalculatedMetrics to the AggregateMetrics model."""
    return AggregateMetrics(
        total_examples=cm.total_examples,
        successful_examples=cm.successful_examples,
        failed_examples=cm.failed_examples,
        mean_trust_score=cm.mean_trust_score,
        median_trust_score=cm.median_trust_score,
        trust_score_std=cm.trust_score_std,
        mean_hallucination_rate=cm.mean_hallucination_rate,
        latency_p50_ms=cm.latency_p50_ms,
        latency_p95_ms=cm.latency_p95_ms,
        latency_p99_ms=cm.latency_p99_ms,
        total_input_tokens=cm.total_input_tokens,
        total_output_tokens=cm.total_output_tokens,
        total_cost=cm.total_cost,
        cost_per_query=cm.cost_per_query,
    )


def _to_cost_summary(
    cb: CostBreakdown,
    total_examples: int,
    storage_cost: float = 0.0,
) -> CostSummary:
    """Convert CostBreakdown to the CostSummary model."""
    cost_per_example = (
        cb.total_cost / total_examples if total_examples > 0 else 0.0
    )
    return CostSummary(
        total_cost=cb.total_cost + storage_cost,
        inference_cost=cb.total_cost,
        storage_cost=storage_cost,
        cost_per_example=cost_per_example,
        currency=cb.currency,
    )


class ReportGenerator:
    """Generate a BaselineEvaluationReport from scored responses.

    Orchestrates MetricsCalculator and CostCalculator to compile
    aggregate metrics, per-category metrics, and cost summary into
    a structured report.

    Usage::

        gen = ReportGenerator(
            model_id="anthropic.claude-3-haiku",
            dataset_id="ds-001",
            pricing=ModelPricing(
                input_price_per_1k_tokens=0.003,
                output_price_per_1k_tokens=0.015,
            ),
        )
        report = gen.generate_baseline_report(scored_responses)

    Requirements: 3.8
    """

    def __init__(
        self,
        model_id: str,
        dataset_id: str,
        pricing: Optional[ModelPricing] = None,
        s3_results_uri: Optional[str] = None,
    ):
        """
        Args:
            model_id: ID of the evaluated model.
            dataset_id: ID of the evaluation dataset.
            pricing: Optional model pricing for cost calculation.
            s3_results_uri: Optional S3 URI for results storage.
                Defaults to a generated path.
        """
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.pricing = pricing
        self.s3_results_uri = s3_results_uri

    def _build_metrics_calculator(self) -> MetricsCalculator:
        """Create a MetricsCalculator with pricing config."""
        if self.pricing is not None:
            pc = PricingConfig(
                input_price_per_token=(
                    self.pricing.input_price_per_1k_tokens / 1000.0
                ),
                output_price_per_token=(
                    self.pricing.output_price_per_1k_tokens / 1000.0
                ),
            )
        else:
            pc = PricingConfig()
        return MetricsCalculator(pricing=pc)

    def _build_cost_calculator(self) -> CostCalculator:
        """Create a CostCalculator with explicit pricing."""
        return CostCalculator(
            model_id=self.model_id,
            pricing=self.pricing,
        )

    def generate_baseline_report(
        self,
        responses: list[ScoredResponse],
    ) -> BaselineEvaluationReport:
        """Compile scored responses into a baseline evaluation report.

        Args:
            responses: List of ScoredResponse objects from the
                ResponseScorer.

        Returns:
            A fully populated BaselineEvaluationReport.
        """
        now = datetime.now(timezone.utc)
        evaluation_id = str(uuid.uuid4())

        # Aggregate metrics
        calc = self._build_metrics_calculator()
        aggregate_cm = calc.calculate(responses)
        aggregate_metrics = _to_aggregate_metrics(aggregate_cm)

        # Per-category metrics
        category_cm = calc.calculate_by_category(responses)
        per_category_metrics = {
            cat: _to_aggregate_metrics(cm)
            for cat, cm in category_cm.items()
        }

        # Cost breakdown via CostCalculator
        cost_calc = self._build_cost_calculator()
        batch_items = [r.batch_item for r in responses]
        cost_breakdown = cost_calc.calculate_batch_cost(batch_items)
        total_examples = len(responses)
        cost_summary = _to_cost_summary(
            cost_breakdown, total_examples
        )

        # S3 URI. Derive the bucket from configuration rather than a
        # hardcoded name; a predictable literal could be squatted, and it
        # would also record a URI that does not exist in this account.
        s3_uri = self.s3_results_uri or (
            f"s3://{aws_config.results_bucket}/evaluations/"
            f"{evaluation_id}/{self.model_id}/{now.isoformat()}"
        )

        # Build minimal EvaluationConfig for the report
        config = EvaluationConfig(
            model_id=self.model_id,
            dataset_id=self.dataset_id,
            inference_params=InferenceRequest(prompt="baseline"),
        )

        return BaselineEvaluationReport(
            evaluation_id=evaluation_id,
            model_id=self.model_id,
            dataset_id=self.dataset_id,
            config=config,
            total_examples=total_examples,
            aggregate_metrics=aggregate_metrics,
            per_category_metrics=per_category_metrics,
            cost_summary=cost_summary,
            created_at=now,
            completed_at=now,
            status="completed",
            s3_results_uri=s3_uri,
        )
