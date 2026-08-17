"""
Metrics aggregation for evaluation results.
"""
from typing import List, Dict, Any
import statistics

from ..data_models.results import (
    EvaluationResult,
    BaselineMetrics,
    ImprovementMetrics,
    CostPerformanceMetrics
)
from .statistical_test_engine import StatisticalTestEngine
from .break_even_calculator import BreakEvenCalculator, BreakEvenResult


class PricingConfig:
    """Model pricing configuration."""
    
    def __init__(
        self,
        input_cost_per_1k_tokens: float = 0.008,
        output_cost_per_1k_tokens: float = 0.024
    ):
        """
        Initialize pricing configuration.
        
        Args:
            input_cost_per_1k_tokens: Cost per 1000 input tokens in USD
            output_cost_per_1k_tokens: Cost per 1000 output tokens in USD
        """
        self.input_cost_per_1k_tokens = input_cost_per_1k_tokens
        self.output_cost_per_1k_tokens = output_cost_per_1k_tokens


class MetricsAggregator:
    """Aggregate and compute evaluation metrics from raw results."""
    
    def __init__(self, high_trust_threshold: float = 0.8):
        """
        Initialize metrics aggregator.
        
        Args:
            high_trust_threshold: Threshold for high-trust responses (default: 0.8)
        """
        self.high_trust_threshold = high_trust_threshold
    
    def aggregate_baseline_metrics(
        self,
        evaluation_results: List[EvaluationResult]
    ) -> BaselineMetrics:
        """
        Aggregate metrics from baseline evaluation.
        
        Args:
            evaluation_results: List of individual evaluation results
            
        Returns:
            BaselineMetrics with aggregate statistics
        """
        if not evaluation_results:
            raise ValueError("Cannot aggregate metrics from empty results list")
        
        # Extract model_id from first result
        model_id = evaluation_results[0].model_response.model_id
        
        # Extract trust scores and latencies
        trust_scores = [r.trust_score.overall_score for r in evaluation_results]
        latencies = [r.model_response.latency_ms for r in evaluation_results]
        
        # Calculate mean and median trust scores
        mean_trust_score = statistics.mean(trust_scores)
        median_trust_score = statistics.median(trust_scores)
        
        # Calculate trust score distribution
        trust_score_distribution = self._calculate_trust_distribution(trust_scores)
        
        # Calculate latency statistics
        mean_latency_ms = statistics.mean(latencies)
        p95_latency_ms = self._calculate_percentile(latencies, 95)
        
        # Calculate token and cost totals
        total_input_tokens = sum(
            r.model_response.input_tokens for r in evaluation_results
        )
        total_output_tokens = sum(
            r.model_response.output_tokens for r in evaluation_results
        )
        
        # Calculate total cost (using default pricing)
        pricing = PricingConfig()
        total_cost = (
            (total_input_tokens / 1000) * pricing.input_cost_per_1k_tokens +
            (total_output_tokens / 1000) * pricing.output_cost_per_1k_tokens
        )
        
        # Calculate hallucination rate
        hallucination_rate = statistics.mean([
            r.hallucination_analysis.hallucination_rate
            for r in evaluation_results
        ])
        
        # Calculate category breakdown
        category_breakdown = self._calculate_category_breakdown(evaluation_results)
        
        return BaselineMetrics(
            model_id=model_id,
            total_examples=len(evaluation_results),
            mean_trust_score=mean_trust_score,
            median_trust_score=median_trust_score,
            trust_score_distribution=trust_score_distribution,
            mean_latency_ms=mean_latency_ms,
            p95_latency_ms=p95_latency_ms,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost=total_cost,
            hallucination_rate=hallucination_rate,
            category_breakdown=category_breakdown
        )
    
    def compute_improvement_metrics(
        self,
        baseline_results: List[EvaluationResult],
        finetuned_results: List[EvaluationResult]
    ) -> ImprovementMetrics:
        """
        Compute improvement metrics between models.
        
        Args:
            baseline_results: Baseline model evaluation results
            finetuned_results: Fine-tuned model evaluation results
            
        Returns:
            ImprovementMetrics with deltas and percentage changes
        """
        if not baseline_results or not finetuned_results:
            raise ValueError("Both baseline and finetuned results must be non-empty")
        
        if len(baseline_results) != len(finetuned_results):
            raise ValueError(
                "Baseline and finetuned results must have the same length"
            )
        
        # Get model IDs
        baseline_model_id = baseline_results[0].model_response.model_id
        finetuned_model_id = finetuned_results[0].model_response.model_id
        
        # Calculate baseline metrics
        baseline_trust_scores = [
            r.trust_score.overall_score for r in baseline_results
        ]
        finetuned_trust_scores = [
            r.trust_score.overall_score for r in finetuned_results
        ]
        
        baseline_mean_trust = statistics.mean(baseline_trust_scores)
        finetuned_mean_trust = statistics.mean(finetuned_trust_scores)
        trust_score_improvement = finetuned_mean_trust - baseline_mean_trust
        
        # Calculate hallucination rates
        baseline_hallucination_rate = statistics.mean([
            r.hallucination_analysis.hallucination_rate for r in baseline_results
        ])
        finetuned_hallucination_rate = statistics.mean([
            r.hallucination_analysis.hallucination_rate for r in finetuned_results
        ])
        hallucination_reduction = (
            baseline_hallucination_rate - finetuned_hallucination_rate
        )
        
        # Calculate latency delta
        baseline_latencies = [
            r.model_response.latency_ms for r in baseline_results
        ]
        finetuned_latencies = [
            r.model_response.latency_ms for r in finetuned_results
        ]
        baseline_mean_latency = statistics.mean(baseline_latencies)
        finetuned_mean_latency = statistics.mean(finetuned_latencies)
        latency_delta_ms = finetuned_mean_latency - baseline_mean_latency
        
        # Calculate cost deltas
        pricing = PricingConfig()
        
        baseline_total_tokens = sum(
            r.model_response.input_tokens + r.model_response.output_tokens
            for r in baseline_results
        )
        finetuned_total_tokens = sum(
            r.model_response.input_tokens + r.model_response.output_tokens
            for r in finetuned_results
        )
        
        baseline_cost = sum([
            (r.model_response.input_tokens / 1000) * pricing.input_cost_per_1k_tokens +
            (r.model_response.output_tokens / 1000) * pricing.output_cost_per_1k_tokens
            for r in baseline_results
        ])
        finetuned_cost = sum([
            (r.model_response.input_tokens / 1000) * pricing.input_cost_per_1k_tokens +
            (r.model_response.output_tokens / 1000) * pricing.output_cost_per_1k_tokens
            for r in finetuned_results
        ])
        
        baseline_cost_per_query = baseline_cost / len(baseline_results)
        finetuned_cost_per_query = finetuned_cost / len(finetuned_results)
        cost_delta_per_query = finetuned_cost_per_query - baseline_cost_per_query
        
        cost_delta_percentage = (
            (cost_delta_per_query / baseline_cost_per_query * 100)
            if baseline_cost_per_query > 0 else 0.0
        )
        
        # Statistical significance via paired tests
        stat_engine = StatisticalTestEngine()
        trust_result = stat_engine.run_paired_tests(
            baseline_trust_scores, finetuned_trust_scores
        )

        if trust_result.insufficient_data:
            statistical_significance = abs(trust_score_improvement) >= 0.05
        else:
            primary_p = (
                trust_result.p_value_ttest
                if trust_result.test_type_used == "ttest"
                else trust_result.p_value_wilcoxon
            )
            statistical_significance = primary_p < 0.05

        # Generate recommendation
        recommendation, justification = self._generate_recommendation(
            trust_score_improvement,
            hallucination_reduction,
            cost_delta_percentage,
            statistical_significance
        )

        return ImprovementMetrics(
            baseline_model_id=baseline_model_id,
            finetuned_model_id=finetuned_model_id,
            trust_score_improvement=trust_score_improvement,
            hallucination_reduction=hallucination_reduction,
            latency_delta_ms=latency_delta_ms,
            cost_delta_per_query=cost_delta_per_query,
            cost_delta_percentage=cost_delta_percentage,
            statistical_significance=statistical_significance,
            recommendation=recommendation,
            justification=justification,
            p_value_ttest=trust_result.p_value_ttest,
            p_value_wilcoxon=trust_result.p_value_wilcoxon,
            confidence_interval_lower=trust_result.confidence_interval_lower,
            confidence_interval_upper=trust_result.confidence_interval_upper,
            test_type_used=trust_result.test_type_used,
        )
    
    def calculate_cost_performance_metrics(
        self,
        evaluation_results: List[EvaluationResult],
        pricing_config: PricingConfig
    ) -> CostPerformanceMetrics:
        """
        Calculate cost-performance metrics.
        
        Args:
            evaluation_results: Evaluation results with token counts
            pricing_config: Model pricing configuration
            
        Returns:
            CostPerformanceMetrics with cost per query and per trust score
        """
        if not evaluation_results:
            raise ValueError("Cannot calculate metrics from empty results list")
        
        # Calculate total cost
        total_cost = sum([
            (r.model_response.input_tokens / 1000) * pricing_config.input_cost_per_1k_tokens +
            (r.model_response.output_tokens / 1000) * pricing_config.output_cost_per_1k_tokens
            for r in evaluation_results
        ])
        
        # Calculate cost per query
        cost_per_query = total_cost / len(evaluation_results)
        
        # Calculate total tokens
        total_tokens = sum(
            r.model_response.input_tokens + r.model_response.output_tokens
            for r in evaluation_results
        )
        cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0.0
        
        # Calculate mean trust score
        trust_scores = [r.trust_score.overall_score for r in evaluation_results]
        mean_trust_score = statistics.mean(trust_scores)
        
        # Calculate cost per high-trust response
        high_trust_count = sum(
            1 for r in evaluation_results
            if r.trust_score.overall_score >= self.high_trust_threshold
        )
        cost_per_high_trust_response = (
            total_cost / high_trust_count if high_trust_count > 0 else 0.0
        )
        
        # Calculate cost efficiency score (trust score per dollar)
        cost_efficiency_score = (
            mean_trust_score / total_cost if total_cost > 0 else 0.0
        )
        
        # Calculate cost projections for various query volumes
        query_volumes = [1000, 10000, 100000, 1000000]
        projected_monthly_cost = {
            volume: cost_per_query * volume
            for volume in query_volumes
        }
        
        return CostPerformanceMetrics(
            total_cost=total_cost,
            cost_per_query=cost_per_query,
            cost_per_high_trust_response=cost_per_high_trust_response,
            cost_per_token=cost_per_token,
            mean_trust_score=mean_trust_score,
            cost_efficiency_score=cost_efficiency_score,
            projected_monthly_cost=projected_monthly_cost
        )
    
    def compute_break_even_analysis(
        self,
        metrics_a: CostPerformanceMetrics,
        metrics_b: CostPerformanceMetrics,
        hallucination_rate_a: float,
        hallucination_rate_b: float,
        remediation_cost: float = 50.0,
    ) -> BreakEvenResult:
        """
        Compute break-even volume analysis between two models.

        Args:
            metrics_a: Cost metrics for model A (lower-trust candidate)
            metrics_b: Cost metrics for model B (higher-trust candidate)
            hallucination_rate_a: Hallucination rate for model A
            hallucination_rate_b: Hallucination rate for model B
            remediation_cost: Cost per hallucinated response

        Returns:
            BreakEvenResult with crossover analysis
        """
        calculator = BreakEvenCalculator(
            remediation_cost_per_hallucination=remediation_cost
        )
        result = calculator.compute_break_even(
            cost_per_query_a=metrics_a.cost_per_query,
            cost_per_query_b=metrics_b.cost_per_query,
            hallucination_rate_a=hallucination_rate_a,
            hallucination_rate_b=hallucination_rate_b,
        )

        # Update metrics_b with break-even volume
        metrics_b.break_even_volume = result.break_even_volume
        metrics_b.remediation_cost_per_hallucination = remediation_cost

        return result

    def _calculate_trust_distribution(
        self,
        trust_scores: List[float]
    ) -> Dict[str, int]:
        """
        Calculate trust score distribution.
        
        Args:
            trust_scores: List of trust scores
            
        Returns:
            Dictionary with counts for "high", "medium", "low"
        """
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for score in trust_scores:
            if score >= 0.8:
                distribution["high"] += 1
            elif score >= 0.5:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def _calculate_percentile(
        self,
        values: List[float],
        percentile: int
    ) -> float:
        """
        Calculate percentile value.
        
        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        # Ensure index is within bounds
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def _calculate_category_breakdown(
        self,
        evaluation_results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """
        Calculate mean trust score by category.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Dictionary mapping category to mean trust score
        """
        category_scores: Dict[str, List[float]] = {}
        
        for result in evaluation_results:
            category = result.category
            score = result.trust_score.overall_score
            
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)
        
        return {
            category: statistics.mean(scores)
            for category, scores in category_scores.items()
        }
    
    def _generate_recommendation(
        self,
        trust_score_improvement: float,
        hallucination_reduction: float,
        cost_delta_percentage: float,
        statistical_significance: bool
    ) -> tuple[str, str]:
        """
        Generate deployment recommendation.
        
        Args:
            trust_score_improvement: Trust score improvement
            hallucination_reduction: Hallucination rate reduction
            cost_delta_percentage: Cost change percentage
            statistical_significance: Whether improvement is statistically significant
            
        Returns:
            Tuple of (recommendation, justification)
        """
        if not statistical_significance:
            return (
                "iterate",
                "Trust score improvement is not statistically significant. "
                "Consider additional fine-tuning iterations."
            )
        
        if trust_score_improvement >= 0.1 and hallucination_reduction >= 0.05:
            if cost_delta_percentage <= 20:
                return (
                    "deploy",
                    f"Significant trust score improvement ({trust_score_improvement:.2%}) "
                    f"and hallucination reduction ({hallucination_reduction:.2%}) "
                    f"with acceptable cost increase ({cost_delta_percentage:.1f}%)."
                )
            else:
                return (
                    "iterate",
                    f"Good quality improvements but cost increase ({cost_delta_percentage:.1f}%) "
                    "is too high. Consider optimization."
                )
        elif trust_score_improvement >= 0.05:
            return (
                "iterate",
                f"Moderate trust score improvement ({trust_score_improvement:.2%}). "
                "Consider additional fine-tuning for better results."
            )
        else:
            return (
                "reject",
                f"Insufficient improvement (trust: {trust_score_improvement:.2%}, "
                f"hallucination: {hallucination_reduction:.2%}). "
                "Fine-tuning did not provide meaningful benefits."
            )
