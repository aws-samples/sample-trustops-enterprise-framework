"""
Unit tests for MetricsAggregator.
"""
import pytest
from datetime import datetime

from src.orchestration.metrics_aggregator import MetricsAggregator, PricingConfig
from src.data_models.results import EvaluationResult
from src.data_models.model_response import (
    ModelResponse,
    TrustScore,
    TrustScoreComponents
)
from src.data_models.hallucination import HallucinationAnalysis


def create_evaluation_result(
    example_id: str,
    model_id: str,
    trust_score: float,
    hallucination_rate: float,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    category: str = "general"
) -> EvaluationResult:
    """Helper to create evaluation result."""
    components = TrustScoreComponents(
        context_grounding=trust_score,
        output_structure=trust_score,
        uncertainty_indicators=trust_score,
        factual_consistency=trust_score,
        response_completeness=trust_score
    )
    
    trust_score_obj = TrustScore(
        overall_score=trust_score,
        components=components,
        confidence_level="high" if trust_score >= 0.8 else "medium" if trust_score >= 0.5 else "low",
        flagged_for_review=trust_score < 0.7,
        explanation="Test trust score"
    )
    
    model_response = ModelResponse(
        response_id=f"resp-{example_id}",
        model_id=model_id,
        prompt="Test prompt",
        response_text="Test response",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        timestamp=datetime.now()
    )
    
    hallucination_analysis = HallucinationAnalysis(
        has_hallucinations=hallucination_rate > 0,
        hallucination_rate=hallucination_rate,
        flagged_spans=[],
        overall_grounding_score=1.0 - hallucination_rate
    )
    
    return EvaluationResult(
        example_id=example_id,
        model_response=model_response,
        trust_score=trust_score_obj,
        hallucination_analysis=hallucination_analysis,
        semantic_similarity=None,
        category=category,
        passed=trust_score >= 0.7
    )


class TestMetricsAggregator:
    """Test suite for MetricsAggregator."""
    
    def test_aggregate_baseline_metrics_basic(self):
        """Test basic baseline metrics aggregation."""
        aggregator = MetricsAggregator()
        
        results = [
            create_evaluation_result("1", "model-1", 0.9, 0.1, 100, 200, 150.0),
            create_evaluation_result("2", "model-1", 0.8, 0.2, 120, 180, 160.0),
            create_evaluation_result("3", "model-1", 0.7, 0.15, 110, 190, 155.0),
        ]
        
        metrics = aggregator.aggregate_baseline_metrics(results)
        
        assert metrics.model_id == "model-1"
        assert metrics.total_examples == 3
        assert metrics.mean_trust_score == pytest.approx(0.8, abs=0.01)
        assert metrics.median_trust_score == 0.8
        assert metrics.mean_latency_ms == pytest.approx(155.0, abs=0.1)
        assert metrics.total_input_tokens == 330
        assert metrics.total_output_tokens == 570
        assert metrics.hallucination_rate == pytest.approx(0.15, abs=0.01)
    
    def test_aggregate_baseline_metrics_trust_distribution(self):
        """Test trust score distribution calculation."""
        aggregator = MetricsAggregator()
        
        results = [
            create_evaluation_result("1", "model-1", 0.9, 0.1, 100, 200, 150.0),  # high
            create_evaluation_result("2", "model-1", 0.85, 0.1, 100, 200, 150.0),  # high
            create_evaluation_result("3", "model-1", 0.6, 0.2, 100, 200, 150.0),  # medium
            create_evaluation_result("4", "model-1", 0.4, 0.3, 100, 200, 150.0),  # low
        ]
        
        metrics = aggregator.aggregate_baseline_metrics(results)
        
        assert metrics.trust_score_distribution["high"] == 2
        assert metrics.trust_score_distribution["medium"] == 1
        assert metrics.trust_score_distribution["low"] == 1
    
    def test_aggregate_baseline_metrics_category_breakdown(self):
        """Test category breakdown calculation."""
        aggregator = MetricsAggregator()
        
        results = [
            create_evaluation_result("1", "model-1", 0.9, 0.1, 100, 200, 150.0, "technical"),
            create_evaluation_result("2", "model-1", 0.8, 0.1, 100, 200, 150.0, "technical"),
            create_evaluation_result("3", "model-1", 0.7, 0.2, 100, 200, 150.0, "business"),
        ]
        
        metrics = aggregator.aggregate_baseline_metrics(results)
        
        assert "technical" in metrics.category_breakdown
        assert "business" in metrics.category_breakdown
        assert metrics.category_breakdown["technical"] == pytest.approx(0.85, abs=0.01)
        assert metrics.category_breakdown["business"] == 0.7
    
    def test_aggregate_baseline_metrics_p95_latency(self):
        """Test P95 latency calculation."""
        aggregator = MetricsAggregator()
        
        # Create 20 results with varying latencies
        results = [
            create_evaluation_result(str(i), "model-1", 0.8, 0.1, 100, 200, float(100 + i * 10))
            for i in range(20)
        ]
        
        metrics = aggregator.aggregate_baseline_metrics(results)
        
        # P95 should be around the 19th value (95% of 20)
        assert metrics.p95_latency_ms >= 280.0
    
    def test_aggregate_baseline_metrics_empty_results(self):
        """Test error handling for empty results."""
        aggregator = MetricsAggregator()
        
        with pytest.raises(ValueError, match="empty results"):
            aggregator.aggregate_baseline_metrics([])
    
    def test_compute_improvement_metrics_basic(self):
        """Test basic improvement metrics computation."""
        aggregator = MetricsAggregator()
        
        baseline_results = [
            create_evaluation_result("1", "baseline", 0.7, 0.3, 100, 200, 150.0),
            create_evaluation_result("2", "baseline", 0.6, 0.4, 100, 200, 160.0),
        ]
        
        finetuned_results = [
            create_evaluation_result("1", "finetuned", 0.9, 0.1, 100, 200, 140.0),
            create_evaluation_result("2", "finetuned", 0.85, 0.15, 100, 200, 145.0),
        ]
        
        metrics = aggregator.compute_improvement_metrics(
            baseline_results,
            finetuned_results
        )
        
        assert metrics.baseline_model_id == "baseline"
        assert metrics.finetuned_model_id == "finetuned"
        assert metrics.trust_score_improvement == pytest.approx(0.225, abs=0.01)
        assert metrics.hallucination_reduction == pytest.approx(0.225, abs=0.01)
        assert metrics.latency_delta_ms < 0  # Improved latency
        assert metrics.statistical_significance is True
    
    def test_compute_improvement_metrics_recommendation_deploy(self):
        """Test recommendation generation for deployment."""
        aggregator = MetricsAggregator()
        
        baseline_results = [
            create_evaluation_result("1", "baseline", 0.6, 0.4, 100, 200, 150.0),
        ]
        
        finetuned_results = [
            create_evaluation_result("1", "finetuned", 0.8, 0.2, 100, 200, 150.0),
        ]
        
        metrics = aggregator.compute_improvement_metrics(
            baseline_results,
            finetuned_results
        )
        
        assert metrics.recommendation == "deploy"
        assert "Significant trust score improvement" in metrics.justification
    
    def test_compute_improvement_metrics_recommendation_iterate(self):
        """Test recommendation generation for iteration."""
        aggregator = MetricsAggregator()
        
        baseline_results = [
            create_evaluation_result("1", "baseline", 0.7, 0.2, 100, 200, 150.0),
        ]
        
        finetuned_results = [
            create_evaluation_result("1", "finetuned", 0.75, 0.18, 100, 200, 150.0),
        ]
        
        metrics = aggregator.compute_improvement_metrics(
            baseline_results,
            finetuned_results
        )
        
        assert metrics.recommendation == "iterate"
    
    def test_compute_improvement_metrics_recommendation_reject(self):
        """Test recommendation generation for rejection."""
        aggregator = MetricsAggregator()
        
        baseline_results = [
            create_evaluation_result("1", "baseline", 0.7, 0.2, 100, 200, 150.0),
        ]
        
        finetuned_results = [
            create_evaluation_result("1", "finetuned", 0.72, 0.19, 100, 200, 150.0),
        ]
        
        metrics = aggregator.compute_improvement_metrics(
            baseline_results,
            finetuned_results
        )
        
        # Small improvement (0.02) is not statistically significant
        assert metrics.recommendation == "iterate"
        assert "not statistically significant" in metrics.justification
    
    def test_compute_improvement_metrics_recommendation_reject_negative(self):
        """Test recommendation generation for rejection with negative improvement."""
        aggregator = MetricsAggregator()
        
        baseline_results = [
            create_evaluation_result("1", "baseline", 0.8, 0.2, 100, 200, 150.0),
        ]
        
        finetuned_results = [
            create_evaluation_result("1", "finetuned", 0.75, 0.25, 100, 200, 150.0),
        ]
        
        metrics = aggregator.compute_improvement_metrics(
            baseline_results,
            finetuned_results
        )
        
        # Negative improvement should lead to reject
        assert metrics.recommendation == "reject"
        assert "Insufficient improvement" in metrics.justification
    
    def test_compute_improvement_metrics_mismatched_lengths(self):
        """Test error handling for mismatched result lengths."""
        aggregator = MetricsAggregator()
        
        baseline_results = [
            create_evaluation_result("1", "baseline", 0.7, 0.3, 100, 200, 150.0),
        ]
        
        finetuned_results = [
            create_evaluation_result("1", "finetuned", 0.9, 0.1, 100, 200, 140.0),
            create_evaluation_result("2", "finetuned", 0.85, 0.15, 100, 200, 145.0),
        ]
        
        with pytest.raises(ValueError, match="same length"):
            aggregator.compute_improvement_metrics(baseline_results, finetuned_results)
    
    def test_compute_improvement_metrics_empty_results(self):
        """Test error handling for empty results."""
        aggregator = MetricsAggregator()
        
        with pytest.raises(ValueError, match="non-empty"):
            aggregator.compute_improvement_metrics([], [])
    
    def test_calculate_cost_performance_metrics_basic(self):
        """Test basic cost-performance metrics calculation."""
        aggregator = MetricsAggregator(high_trust_threshold=0.8)
        pricing = PricingConfig(
            input_cost_per_1k_tokens=0.008,
            output_cost_per_1k_tokens=0.024
        )
        
        results = [
            create_evaluation_result("1", "model-1", 0.9, 0.1, 1000, 2000, 150.0),
            create_evaluation_result("2", "model-1", 0.85, 0.1, 1000, 2000, 160.0),
            create_evaluation_result("3", "model-1", 0.6, 0.2, 1000, 2000, 155.0),
        ]
        
        metrics = aggregator.calculate_cost_performance_metrics(results, pricing)
        
        # Total cost = (3000 input / 1000 * 0.008) + (6000 output / 1000 * 0.024)
        # = 0.024 + 0.144 = 0.168
        assert metrics.total_cost == pytest.approx(0.168, abs=0.001)
        assert metrics.cost_per_query == pytest.approx(0.056, abs=0.001)
        assert metrics.mean_trust_score == pytest.approx(0.783, abs=0.01)
        
        # 2 high-trust responses (>= 0.8)
        assert metrics.cost_per_high_trust_response == pytest.approx(0.084, abs=0.001)
    
    def test_calculate_cost_performance_metrics_cost_efficiency(self):
        """Test cost efficiency score calculation."""
        aggregator = MetricsAggregator()
        pricing = PricingConfig(
            input_cost_per_1k_tokens=0.008,
            output_cost_per_1k_tokens=0.024
        )
        
        results = [
            create_evaluation_result("1", "model-1", 0.8, 0.1, 1000, 2000, 150.0),
        ]
        
        metrics = aggregator.calculate_cost_performance_metrics(results, pricing)
        
        # Cost efficiency = mean_trust_score / total_cost
        expected_efficiency = 0.8 / metrics.total_cost
        assert metrics.cost_efficiency_score == pytest.approx(expected_efficiency, abs=0.01)
    
    def test_calculate_cost_performance_metrics_projections(self):
        """Test cost projection calculations."""
        aggregator = MetricsAggregator()
        pricing = PricingConfig(
            input_cost_per_1k_tokens=0.008,
            output_cost_per_1k_tokens=0.024
        )
        
        results = [
            create_evaluation_result("1", "model-1", 0.8, 0.1, 1000, 2000, 150.0),
        ]
        
        metrics = aggregator.calculate_cost_performance_metrics(results, pricing)
        
        # Check projections exist for standard volumes
        assert 1000 in metrics.projected_monthly_cost
        assert 10000 in metrics.projected_monthly_cost
        assert 100000 in metrics.projected_monthly_cost
        assert 1000000 in metrics.projected_monthly_cost
        
        # Verify projection calculation
        expected_1000 = metrics.cost_per_query * 1000
        assert metrics.projected_monthly_cost[1000] == pytest.approx(expected_1000, abs=0.001)
    
    def test_calculate_cost_performance_metrics_no_high_trust(self):
        """Test handling when no high-trust responses exist."""
        aggregator = MetricsAggregator(high_trust_threshold=0.95)
        pricing = PricingConfig()
        
        results = [
            create_evaluation_result("1", "model-1", 0.7, 0.2, 1000, 2000, 150.0),
            create_evaluation_result("2", "model-1", 0.6, 0.3, 1000, 2000, 160.0),
        ]
        
        metrics = aggregator.calculate_cost_performance_metrics(results, pricing)
        
        # Should handle zero high-trust responses gracefully
        assert metrics.cost_per_high_trust_response == 0.0
    
    def test_calculate_cost_performance_metrics_empty_results(self):
        """Test error handling for empty results."""
        aggregator = MetricsAggregator()
        pricing = PricingConfig()
        
        with pytest.raises(ValueError, match="empty results"):
            aggregator.calculate_cost_performance_metrics([], pricing)
    
    def test_pricing_config_initialization(self):
        """Test PricingConfig initialization."""
        pricing = PricingConfig(
            input_cost_per_1k_tokens=0.01,
            output_cost_per_1k_tokens=0.03
        )
        
        assert pricing.input_cost_per_1k_tokens == 0.01
        assert pricing.output_cost_per_1k_tokens == 0.03
    
    def test_pricing_config_defaults(self):
        """Test PricingConfig default values."""
        pricing = PricingConfig()
        
        assert pricing.input_cost_per_1k_tokens == 0.008
        assert pricing.output_cost_per_1k_tokens == 0.024
