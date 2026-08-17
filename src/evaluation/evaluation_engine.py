"""
Evaluation Engine — facade integrating all evaluation components.

Supports baseline evaluation (single model), comparative evaluation
(two models side-by-side), and evaluation status queries.  Delegates
to existing components for batch inference, scoring, metrics, cost,
progress tracking, comparison, and reporting.

Requirements: 3.1-3.14, 7.1-7.10
"""

import logging
import uuid
from typing import Any, Callable, Optional

from src.clients.inference_client import InferenceClient
from src.data_models.evaluation import (
    BaselineEvaluationReport,
    ComparisonConfig,
    ComparisonReport,
    EvaluationConfig,
)
from src.data_models.model import ModelPricing
from src.evaluation.batch_runner import (
    BatchRunConfig,
    BatchRunSummary,
    run_batch_inference,
)
from src.evaluation.comparison_report_generator import (
    generate_comparison_report,
    generate_per_category_breakdown,
)
from src.evaluation.compatibility_checker import (
    check_model_dataset_compatibility,
)
from src.evaluation.cost_performance_analyzer import (
    analyze_cost_performance,
)
from src.evaluation.improvement_calculator import (
    calculate_improvement_metrics,
)
from src.evaluation.metrics_calculator import (
    MetricsCalculator,
    PricingConfig,
)
from src.evaluation.parallel_invoker import (
    ParallelInvokeConfig,
    run_parallel_invocation,
)
from src.evaluation.progress_tracker import ProgressTracker
from src.evaluation.recommendation_engine import generate_recommendation
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.response_comparator import compare_batch
from src.evaluation.response_scorer import ResponseScorer, ScoredResponse
from src.evaluation.significance_tester import calculate_significance
from src.hallucination.hallucination_detector import HallucinationDetector
from src.trust_scoring.trust_scoring_engine_v2 import TrustScoringEngine

logger = logging.getLogger(__name__)

# Type alias for progress callbacks: (completed, total) -> None
ProgressCallback = Callable[[int, int], None]


class EvaluationEngine:
    """Facade that orchestrates baseline and comparative evaluations.

    Integrates:
    - CompatibilityChecker for model-dataset validation
    - BatchRunner for batch inference
    - ResponseScorer for per-response trust + hallucination scoring
    - MetricsCalculator for aggregate metrics
    - ReportGenerator for baseline reports
    - ParallelInvoker for side-by-side model invocation
    - ResponseComparator for paired response comparison
    - ImprovementCalculator for delta metrics
    - SignificanceTester for statistical significance
    - RecommendationEngine for deployment recommendations
    - CostPerformanceAnalyzer for cost-quality trade-offs
    - ComparisonReportGenerator for comparison reports
    - ProgressTracker for status tracking

    Usage::

        engine = EvaluationEngine(
            inference_client=client,
            trust_scorer=trust_engine,
            hallucination_detector=detector,
        )
        report = await engine.run_baseline_evaluation(config)

    Requirements: 3.1-3.14, 7.1-7.10
    """

    def __init__(
        self,
        inference_client: InferenceClient,
        trust_scorer: TrustScoringEngine,
        hallucination_detector: Optional[HallucinationDetector] = None,
    ):
        """
        Args:
            inference_client: Client for model invocations.
            trust_scorer: Engine for computing trust scores.
            hallucination_detector: Optional detector for
                hallucination analysis.
        """
        self.inference_client = inference_client
        self.trust_scorer = trust_scorer
        self.hallucination_detector = hallucination_detector
        self._response_scorer = ResponseScorer(
            trust_engine=trust_scorer,
            hallucination_detector=hallucination_detector,
        )
        # In-memory evaluation status store (evaluation_id -> status dict)
        self._status_store: dict[str, dict[str, Any]] = {}

    async def run_baseline_evaluation(
        self,
        config: EvaluationConfig,
        dataset_items: Optional[list[dict[str, Any]]] = None,
        model_metadata: Optional[Any] = None,
        dataset_metadata: Optional[Any] = None,
        pricing: Optional[ModelPricing] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> BaselineEvaluationReport:
        """Run baseline evaluation on a single model.

        Steps:
        1. Validate model-dataset compatibility (if metadata provided)
        2. Run batch inference via BatchRunner
        3. Score each response (trust + hallucination)
        4. Calculate aggregate and per-category metrics
        5. Generate baseline report

        Args:
            config: Evaluation configuration.
            dataset_items: List of dataset item dicts (each must have
                a ``prompt`` key; optional ``expected_response``,
                ``category``, ``source_documents``).
            model_metadata: Optional ModelMetadata for compatibility
                checking.
            dataset_metadata: Optional DatasetMetadata for compatibility
                checking.
            pricing: Optional model pricing for cost calculation.
            progress_callback: Optional callback ``(completed, total)``.

        Returns:
            BaselineEvaluationReport with all metrics.

        Raises:
            ValueError: If compatibility check fails.
        """
        evaluation_id = str(uuid.uuid4())
        items = dataset_items or []

        # Track status
        self._status_store[evaluation_id] = {
            "status": "running",
            "total": len(items),
            "completed": 0,
            "failed": 0,
        }

        # 1. Compatibility check
        if model_metadata is not None and dataset_metadata is not None:
            compat = check_model_dataset_compatibility(
                model_metadata, dataset_metadata
            )
            if not compat.is_compatible:
                self._status_store[evaluation_id]["status"] = "failed"
                raise ValueError(
                    "Model-dataset incompatible: "
                    + "; ".join(compat.reasons)
                )

        # 2. Batch inference
        tracker = ProgressTracker(
            evaluation_id=evaluation_id,
            total=len(items),
        )

        def _on_progress(completed: int, total: int) -> None:
            tracker.update(completed=1)
            self._status_store[evaluation_id]["completed"] = completed
            if progress_callback:
                progress_callback(completed, total)

        batch_config = BatchRunConfig(
            model_id=config.model_id,
            concurrency=config.concurrency,
            max_tokens=config.inference_params.max_tokens,
            temperature=config.inference_params.temperature,
            timeout_seconds=config.timeout_per_request,
        )

        batch_summary: BatchRunSummary = await run_batch_inference(
            items=items,
            client=self.inference_client,
            config=batch_config,
            progress_callback=_on_progress,
        )

        # 3. Score each response
        scored_responses: list[ScoredResponse] = []
        for result in batch_summary.results:
            source_docs = None
            if result.index < len(items):
                source_docs = items[result.index].get(
                    "source_documents"
                )
            scored = await self._response_scorer.score_response(
                item=result,
                source_documents=source_docs,
                model_id=config.model_id,
            )
            scored_responses.append(scored)

        # 4 & 5. Generate report via ReportGenerator
        report_gen = ReportGenerator(
            model_id=config.model_id,
            dataset_id=config.dataset_id,
            pricing=pricing,
        )
        report = report_gen.generate_baseline_report(scored_responses)

        # Update status
        tracker.mark_completed()
        self._status_store[evaluation_id] = {
            "status": "completed",
            "total": len(items),
            "completed": len(items),
            "failed": batch_summary.failed,
        }

        return report

    async def run_comparative_evaluation(
        self,
        config: ComparisonConfig,
        dataset_items: Optional[list[dict[str, Any]]] = None,
        pricing_1: Optional[ModelPricing] = None,
        pricing_2: Optional[ModelPricing] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ComparisonReport:
        """Run comparative evaluation between two models.

        Supports both baseline-vs-fine-tuned and any two foundation
        models.

        Steps:
        1. Invoke both models in parallel on identical prompts
        2. Score each model's responses (trust + hallucination)
        3. Compare paired responses
        4. Calculate improvement metrics
        5. Run statistical significance test
        6. Generate deployment recommendation
        7. Analyze cost-performance
        8. Generate comparison report

        Args:
            config: Comparison configuration.
            dataset_items: List of dataset item dicts.
            pricing_1: Optional pricing for model 1.
            pricing_2: Optional pricing for model 2.
            progress_callback: Optional callback ``(completed, total)``.

        Returns:
            ComparisonReport with side-by-side metrics.
        """
        evaluation_id = str(uuid.uuid4())
        items = dataset_items or []

        self._status_store[evaluation_id] = {
            "status": "running",
            "total": len(items),
            "completed": 0,
            "failed": 0,
        }

        # 1. Parallel invocation
        parallel_config = ParallelInvokeConfig(
            model_id_1=config.model_id_1,
            model_id_2=config.model_id_2,
            concurrency=getattr(config, "concurrency", 5)
            if hasattr(config, "concurrency")
            else 5,
            max_tokens=config.inference_params.max_tokens,
            temperature=config.inference_params.temperature,
            timeout_seconds=getattr(
                config, "timeout_per_request", 60.0
            )
            if hasattr(config, "timeout_per_request")
            else 60.0,
        )

        parallel_summary = await run_parallel_invocation(
            items=items,
            client=self.inference_client,
            config=parallel_config,
            progress_callback=progress_callback,
        )

        # 2. Score each model's responses
        trust_scores_1: list[float] = []
        trust_scores_2: list[float] = []
        hall_rates_1: list[float] = []
        hall_rates_2: list[float] = []
        scored_1: list[ScoredResponse] = []
        scored_2: list[ScoredResponse] = []

        for pair in parallel_summary.results:
            source_docs = None
            if pair.index < len(items):
                source_docs = items[pair.index].get(
                    "source_documents"
                )

            # Score model 1
            from src.evaluation.batch_runner import BatchItemResult

            item_1 = BatchItemResult(
                index=pair.index,
                prompt=pair.prompt,
                response_text=(
                    pair.response_1.text
                    if pair.response_1
                    else ""
                ),
                latency_ms=pair.latency_ms_1,
                input_tokens=(
                    pair.response_1.input_tokens
                    if pair.response_1
                    else 0
                ),
                output_tokens=(
                    pair.response_1.output_tokens
                    if pair.response_1
                    else 0
                ),
                success=pair.response_1 is not None,
                error=pair.error_1,
                category=pair.category,
                expected_response=pair.expected_response,
            )
            s1 = await self._response_scorer.score_response(
                item=item_1,
                source_documents=source_docs,
                model_id=config.model_id_1,
            )
            scored_1.append(s1)

            ts1 = (
                s1.trust_score_result.overall_score
                if s1.trust_score_result
                else 0.0
            )
            hr1 = (
                s1.hallucination_result.hallucination_rate
                if s1.hallucination_result
                else 0.0
            )
            trust_scores_1.append(ts1)
            hall_rates_1.append(hr1)

            # Score model 2
            item_2 = BatchItemResult(
                index=pair.index,
                prompt=pair.prompt,
                response_text=(
                    pair.response_2.text
                    if pair.response_2
                    else ""
                ),
                latency_ms=pair.latency_ms_2,
                input_tokens=(
                    pair.response_2.input_tokens
                    if pair.response_2
                    else 0
                ),
                output_tokens=(
                    pair.response_2.output_tokens
                    if pair.response_2
                    else 0
                ),
                success=pair.response_2 is not None,
                error=pair.error_2,
                category=pair.category,
                expected_response=pair.expected_response,
            )
            s2 = await self._response_scorer.score_response(
                item=item_2,
                source_documents=source_docs,
                model_id=config.model_id_2,
            )
            scored_2.append(s2)

            ts2 = (
                s2.trust_score_result.overall_score
                if s2.trust_score_result
                else 0.0
            )
            hr2 = (
                s2.hallucination_result.hallucination_rate
                if s2.hallucination_result
                else 0.0
            )
            trust_scores_2.append(ts2)
            hall_rates_2.append(hr2)

        # 3. Compare paired responses
        compare_batch(
            pairs=parallel_summary.results,
            trust_scores_1=trust_scores_1,
            trust_scores_2=trust_scores_2,
            hallucination_rates_1=hall_rates_1,
            hallucination_rates_2=hall_rates_2,
            pricing_1=pricing_1,
            pricing_2=pricing_2,
        )

        # 4. Calculate aggregate metrics for each model
        pricing_cfg_1 = PricingConfig(
            input_price_per_token=(
                pricing_1.input_price_per_1k_tokens / 1000.0
                if pricing_1
                else 0.0
            ),
            output_price_per_token=(
                pricing_1.output_price_per_1k_tokens / 1000.0
                if pricing_1
                else 0.0
            ),
        )
        pricing_cfg_2 = PricingConfig(
            input_price_per_token=(
                pricing_2.input_price_per_1k_tokens / 1000.0
                if pricing_2
                else 0.0
            ),
            output_price_per_token=(
                pricing_2.output_price_per_1k_tokens / 1000.0
                if pricing_2
                else 0.0
            ),
        )

        calc_1 = MetricsCalculator(pricing=pricing_cfg_1)
        calc_2 = MetricsCalculator(pricing=pricing_cfg_2)

        from src.evaluation.report_generator import _to_aggregate_metrics

        m1_agg = _to_aggregate_metrics(calc_1.calculate(scored_1))
        m2_agg = _to_aggregate_metrics(calc_2.calculate(scored_2))

        m1_cat = {
            k: _to_aggregate_metrics(v)
            for k, v in calc_1.calculate_by_category(scored_1).items()
        }
        m2_cat = {
            k: _to_aggregate_metrics(v)
            for k, v in calc_2.calculate_by_category(scored_2).items()
        }

        # 5. Improvement metrics
        improvement = calculate_improvement_metrics(m1_agg, m2_agg)

        # 6. Statistical significance
        sig_result = calculate_significance(
            trust_scores_1, trust_scores_2
        )
        improvement = improvement.model_copy(
            update={
                "statistical_significance": (
                    sig_result.statistical_significance
                ),
                "p_value": sig_result.p_value,
                "confidence_interval": sig_result.confidence_interval,
            }
        )

        # 7. Deployment recommendation
        thresholds = config.deploy_thresholds
        recommendation, justification = generate_recommendation(
            improvement, thresholds
        )

        # 8. Cost-performance analysis
        cost_perf = analyze_cost_performance(m1_agg, m2_agg)

        # Per-category breakdown
        per_cat_breakdown = generate_per_category_breakdown(
            m1_cat, m2_cat
        )

        # 9. Generate comparison report
        report = generate_comparison_report(
            model_1_id=config.model_id_1,
            model_2_id=config.model_id_2,
            dataset_id=config.dataset_id,
            model_1_metrics=m1_agg,
            model_2_metrics=m2_agg,
            improvement_metrics=improvement,
            recommendation=recommendation,
            recommendation_justification=justification,
            per_category_breakdown=per_cat_breakdown,
            cost_performance_analysis=cost_perf,
        )

        self._status_store[evaluation_id] = {
            "status": "completed",
            "total": len(items),
            "completed": len(items),
            "failed": parallel_summary.model_1_failures
            + parallel_summary.model_2_failures,
        }

        return report

    def get_evaluation_status(
        self, evaluation_id: str
    ) -> dict[str, Any]:
        """Get current status of an evaluation.

        Args:
            evaluation_id: The evaluation identifier.

        Returns:
            Dict with keys: status, total, completed, failed.
            Returns a ``not_found`` status if the ID is unknown.
        """
        return self._status_store.get(
            evaluation_id,
            {
                "status": "not_found",
                "total": 0,
                "completed": 0,
                "failed": 0,
            },
        )
