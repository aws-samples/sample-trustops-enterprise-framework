"""
Unit tests for evaluation data models.

Tests Requirements: 3.1, 3.2, 3.3, 3.5, 7.1, 7.2, 7.4
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from src.data_models.evaluation import (
    EvaluationConfig,
    TrustScoreWeights,
    TrustScoreResult,
    HallucinationResult,
    EvaluationResult,
    AggregateMetrics,
    CostSummary,
    BaselineEvaluationReport,
    DeploymentRecommendation,
    ComparisonConfig,
    DeploymentThresholds,
    ImprovementMetrics,
    CostPerformanceAnalysis,
    ComparisonReport,
)
from src.data_models.model import InferenceRequest


class TestTrustScoreWeights:
    """Test TrustScoreWeights data model."""

    def test_trust_score_weights_default_values(self):
        """Test TrustScoreWeights with default values."""
        weights = TrustScoreWeights()

        assert weights.accuracy == 0.25
        assert weights.consistency == 0.20
        assert weights.safety == 0.20
        assert weights.bias == 0.15
        assert weights.context_grounding == 0.20

    def test_trust_score_weights_custom_values(self):
        """Test TrustScoreWeights with custom values."""
        weights = TrustScoreWeights(
            accuracy=0.3,
            consistency=0.2,
            safety=0.2,
            bias=0.1,
            context_grounding=0.2,
        )

        assert weights.accuracy == 0.3
        assert weights.consistency == 0.2
        assert weights.safety == 0.2
        assert weights.bias == 0.1
        assert weights.context_grounding == 0.2

    def test_trust_score_weights_out_of_range_rejected(self):
        """Test that weights outside [0, 1] are rejected."""
        with pytest.raises(ValidationError):
            TrustScoreWeights(
                accuracy=1.5,
                consistency=0.2,
                safety=0.2,
                bias=0.1,
                context_grounding=0.2,
            )


class TestEvaluationConfig:
    """Test EvaluationConfig data model."""

    def test_evaluation_config_valid_creation(self):
        """Test creating EvaluationConfig with valid data."""
        inference_params = InferenceRequest(
            prompt="Test prompt",
            max_tokens=1024,
            temperature=0.7,
        )

        config = EvaluationConfig(
            model_id="model-123",
            dataset_id="dataset-456",
            inference_params=inference_params,
            batch_size=10,
            timeout_per_request=60.0,
            concurrency=5,
        )

        assert config.model_id == "model-123"
        assert config.dataset_id == "dataset-456"
        assert config.inference_params == inference_params
        assert config.batch_size == 10
        assert config.timeout_per_request == 60.0
        assert config.concurrency == 5

    def test_evaluation_config_with_trust_score_weights(self):
        """Test EvaluationConfig with custom trust score weights."""
        inference_params = InferenceRequest(prompt="Test")
        weights = TrustScoreWeights(accuracy=0.4, consistency=0.3, safety=0.3)

        config = EvaluationConfig(
            model_id="model-123",
            dataset_id="dataset-456",
            inference_params=inference_params,
            trust_score_weights=weights,
        )

        assert config.trust_score_weights == weights

    def test_evaluation_config_invalid_batch_size_rejected(self):
        """Test that invalid batch size is rejected."""
        inference_params = InferenceRequest(prompt="Test")

        with pytest.raises(ValidationError):
            EvaluationConfig(
                model_id="model-123",
                dataset_id="dataset-456",
                inference_params=inference_params,
                batch_size=0,
            )


class TestEvaluationResult:
    """Test EvaluationResult data model."""

    def test_evaluation_result_valid_creation(self):
        """Test creating EvaluationResult with valid data."""
        now = datetime.now()
        trust_score = TrustScoreResult(
            overall_score=0.85,
            dimension_scores={},
            confidence_level=0.9,
        )
        hallucination = HallucinationResult(
            has_hallucinations=False,
            hallucination_rate=0.0,
        )

        result = EvaluationResult(
            example_id="ex-001",
            prompt="What is AI?",
            expected_response="Artificial Intelligence",
            actual_response="AI is artificial intelligence",
            model_id="model-123",
            trust_score=trust_score,
            hallucination_analysis=hallucination,
            latency_ms=250.5,
            input_tokens=10,
            output_tokens=20,
            cost=0.001,
            category="qa",
            timestamp=now,
        )

        assert result.example_id == "ex-001"
        assert result.prompt == "What is AI?"
        assert result.expected_response == "Artificial Intelligence"
        assert result.actual_response == "AI is artificial intelligence"
        assert result.model_id == "model-123"
        assert result.trust_score == trust_score
        assert result.hallucination_analysis == hallucination
        assert result.latency_ms == 250.5
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.cost == 0.001
        assert result.category == "qa"
        assert result.timestamp == now
        assert result.error is None

    def test_evaluation_result_with_error(self):
        """Test EvaluationResult with error."""
        now = datetime.now()
        trust_score = TrustScoreResult(overall_score=0.0)
        hallucination = HallucinationResult()

        result = EvaluationResult(
            example_id="ex-002",
            prompt="Test",
            actual_response="",
            model_id="model-123",
            trust_score=trust_score,
            hallucination_analysis=hallucination,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            cost=0,
            timestamp=now,
            error="Timeout error",
        )

        assert result.error == "Timeout error"


class TestAggregateMetrics:
    """Test AggregateMetrics data model."""

    def test_aggregate_metrics_valid_creation(self):
        """Test creating AggregateMetrics with valid data."""
        metrics = AggregateMetrics(
            total_examples=100,
            successful_examples=95,
            failed_examples=5,
            mean_trust_score=0.82,
            median_trust_score=0.85,
            trust_score_std=0.12,
            mean_hallucination_rate=0.05,
            latency_p50_ms=200.0,
            latency_p95_ms=450.0,
            latency_p99_ms=600.0,
            total_input_tokens=5000,
            total_output_tokens=10000,
            total_cost=5.50,
            cost_per_query=0.055,
        )

        assert metrics.total_examples == 100
        assert metrics.successful_examples == 95
        assert metrics.failed_examples == 5
        assert metrics.mean_trust_score == 0.82
        assert metrics.median_trust_score == 0.85
        assert metrics.trust_score_std == 0.12
        assert metrics.mean_hallucination_rate == 0.05
        assert metrics.latency_p50_ms == 200.0
        assert metrics.latency_p95_ms == 450.0
        assert metrics.latency_p99_ms == 600.0
        assert metrics.total_input_tokens == 5000
        assert metrics.total_output_tokens == 10000
        assert metrics.total_cost == 5.50
        assert metrics.cost_per_query == 0.055

    def test_aggregate_metrics_negative_values_rejected(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            AggregateMetrics(
                total_examples=-1,
                successful_examples=95,
                failed_examples=5,
                mean_trust_score=0.82,
                median_trust_score=0.85,
                trust_score_std=0.12,
                mean_hallucination_rate=0.05,
                latency_p50_ms=200.0,
                latency_p95_ms=450.0,
                latency_p99_ms=600.0,
                total_input_tokens=5000,
                total_output_tokens=10000,
                total_cost=5.50,
                cost_per_query=0.055,
            )


class TestCostSummary:
    """Test CostSummary data model."""

    def test_cost_summary_valid_creation(self):
        """Test creating CostSummary with valid data."""
        summary = CostSummary(
            total_cost=10.50,
            inference_cost=9.00,
            storage_cost=1.50,
            cost_per_example=0.105,
            currency="USD",
        )

        assert summary.total_cost == 10.50
        assert summary.inference_cost == 9.00
        assert summary.storage_cost == 1.50
        assert summary.cost_per_example == 0.105
        assert summary.currency == "USD"

    def test_cost_summary_default_currency(self):
        """Test CostSummary with default currency."""
        summary = CostSummary(
            total_cost=5.00,
            inference_cost=4.50,
            storage_cost=0.50,
            cost_per_example=0.05,
        )

        assert summary.currency == "USD"


class TestBaselineEvaluationReport:
    """Test BaselineEvaluationReport data model."""

    def test_baseline_evaluation_report_valid_creation(self):
        """Test creating BaselineEvaluationReport with valid data."""
        now = datetime.now()
        inference_params = InferenceRequest(prompt="Test")
        config = EvaluationConfig(
            model_id="model-123",
            dataset_id="dataset-456",
            inference_params=inference_params,
        )
        metrics = AggregateMetrics(
            total_examples=100,
            successful_examples=100,
            failed_examples=0,
            mean_trust_score=0.85,
            median_trust_score=0.87,
            trust_score_std=0.10,
            mean_hallucination_rate=0.03,
            latency_p50_ms=200.0,
            latency_p95_ms=400.0,
            latency_p99_ms=500.0,
            total_input_tokens=5000,
            total_output_tokens=10000,
            total_cost=5.00,
            cost_per_query=0.05,
        )
        cost_summary = CostSummary(
            total_cost=5.00,
            inference_cost=4.50,
            storage_cost=0.50,
            cost_per_example=0.05,
        )

        report = BaselineEvaluationReport(
            evaluation_id="eval-001",
            model_id="model-123",
            dataset_id="dataset-456",
            config=config,
            total_examples=100,
            aggregate_metrics=metrics,
            per_category_metrics={},
            cost_summary=cost_summary,
            created_at=now,
            completed_at=now,
            status="completed",
            s3_results_uri="s3://bucket/results/eval-001",
        )

        assert report.evaluation_id == "eval-001"
        assert report.model_id == "model-123"
        assert report.dataset_id == "dataset-456"
        assert report.config == config
        assert report.total_examples == 100
        assert report.aggregate_metrics == metrics
        assert report.cost_summary == cost_summary
        assert report.status == "completed"

    def test_baseline_evaluation_report_invalid_s3_uri_rejected(self):
        """Test that invalid S3 URI is rejected."""
        now = datetime.now()
        inference_params = InferenceRequest(prompt="Test")
        config = EvaluationConfig(
            model_id="model-123",
            dataset_id="dataset-456",
            inference_params=inference_params,
        )
        metrics = AggregateMetrics(
            total_examples=100,
            successful_examples=100,
            failed_examples=0,
            mean_trust_score=0.85,
            median_trust_score=0.87,
            trust_score_std=0.10,
            mean_hallucination_rate=0.03,
            latency_p50_ms=200.0,
            latency_p95_ms=400.0,
            latency_p99_ms=500.0,
            total_input_tokens=5000,
            total_output_tokens=10000,
            total_cost=5.00,
            cost_per_query=0.05,
        )
        cost_summary = CostSummary(
            total_cost=5.00,
            inference_cost=4.50,
            storage_cost=0.50,
            cost_per_example=0.05,
        )

        with pytest.raises(ValidationError):
            BaselineEvaluationReport(
                evaluation_id="eval-002",
                model_id="model-123",
                dataset_id="dataset-456",
                config=config,
                total_examples=100,
                aggregate_metrics=metrics,
                cost_summary=cost_summary,
                created_at=now,
                status="completed",
                s3_results_uri="http://bucket/results/eval-002",
            )


class TestDeploymentRecommendation:
    """Test DeploymentRecommendation enum."""

    def test_deployment_recommendation_enum_values(self):
        """Test DeploymentRecommendation enum has all required values."""
        assert DeploymentRecommendation.DEPLOY == "deploy"
        assert DeploymentRecommendation.ITERATE == "iterate"
        assert DeploymentRecommendation.REJECT == "reject"


class TestDeploymentThresholds:
    """Test DeploymentThresholds data model."""

    def test_deployment_thresholds_default_values(self):
        """Test DeploymentThresholds with default values."""
        thresholds = DeploymentThresholds()

        assert thresholds.min_trust_score_improvement == 0.05
        assert thresholds.max_cost_increase_percent == 20.0
        assert thresholds.min_hallucination_reduction == 0.1
        assert thresholds.min_statistical_significance == 0.95

    def test_deployment_thresholds_custom_values(self):
        """Test DeploymentThresholds with custom values."""
        thresholds = DeploymentThresholds(
            min_trust_score_improvement=0.10,
            max_cost_increase_percent=15.0,
            min_hallucination_reduction=0.15,
            min_statistical_significance=0.99,
        )

        assert thresholds.min_trust_score_improvement == 0.10
        assert thresholds.max_cost_increase_percent == 15.0
        assert thresholds.min_hallucination_reduction == 0.15
        assert thresholds.min_statistical_significance == 0.99


class TestComparisonConfig:
    """Test ComparisonConfig data model."""

    def test_comparison_config_valid_creation(self):
        """Test creating ComparisonConfig with valid data."""
        inference_params = InferenceRequest(prompt="Test")
        thresholds = DeploymentThresholds()

        config = ComparisonConfig(
            model_id_1="baseline-model",
            model_id_2="finetuned-model",
            dataset_id="dataset-789",
            inference_params=inference_params,
            deploy_thresholds=thresholds,
        )

        assert config.model_id_1 == "baseline-model"
        assert config.model_id_2 == "finetuned-model"
        assert config.dataset_id == "dataset-789"
        assert config.inference_params == inference_params
        assert config.deploy_thresholds == thresholds


class TestImprovementMetrics:
    """Test ImprovementMetrics data model."""

    def test_improvement_metrics_valid_creation(self):
        """Test creating ImprovementMetrics with valid data."""
        metrics = ImprovementMetrics(
            trust_score_delta=0.08,
            trust_score_delta_percent=10.0,
            hallucination_reduction=0.02,
            hallucination_reduction_percent=40.0,
            latency_delta_ms=-50.0,
            latency_delta_percent=-20.0,
            cost_delta_per_query=0.01,
            cost_delta_percent=10.0,
            statistical_significance=0.98,
            p_value=0.02,
            confidence_interval=(0.05, 0.11),
        )

        assert metrics.trust_score_delta == 0.08
        assert metrics.trust_score_delta_percent == 10.0
        assert metrics.hallucination_reduction == 0.02
        assert metrics.hallucination_reduction_percent == 40.0
        assert metrics.latency_delta_ms == -50.0
        assert metrics.latency_delta_percent == -20.0
        assert metrics.cost_delta_per_query == 0.01
        assert metrics.cost_delta_percent == 10.0
        assert metrics.statistical_significance == 0.98
        assert metrics.p_value == 0.02
        assert metrics.confidence_interval == (0.05, 0.11)


class TestCostPerformanceAnalysis:
    """Test CostPerformanceAnalysis data model."""

    def test_cost_performance_analysis_valid_creation(self):
        """Test creating CostPerformanceAnalysis with valid data."""
        analysis = CostPerformanceAnalysis(
            model_1_cost_per_trust_point=0.06,
            model_2_cost_per_trust_point=0.065,
            quality_gain_justifies_cost=True,
            break_even_volume=10000,
        )

        assert analysis.model_1_cost_per_trust_point == 0.06
        assert analysis.model_2_cost_per_trust_point == 0.065
        assert analysis.quality_gain_justifies_cost is True
        assert analysis.break_even_volume == 10000

    def test_cost_performance_analysis_no_break_even(self):
        """Test CostPerformanceAnalysis without break-even volume."""
        analysis = CostPerformanceAnalysis(
            model_1_cost_per_trust_point=0.06,
            model_2_cost_per_trust_point=0.055,
            quality_gain_justifies_cost=True,
            break_even_volume=None,
        )

        assert analysis.break_even_volume is None


class TestComparisonReport:
    """Test ComparisonReport data model."""

    def test_comparison_report_valid_creation(self):
        """Test creating ComparisonReport with valid data."""
        now = datetime.now()
        metrics_1 = AggregateMetrics(
            total_examples=100,
            successful_examples=100,
            failed_examples=0,
            mean_trust_score=0.80,
            median_trust_score=0.82,
            trust_score_std=0.10,
            mean_hallucination_rate=0.05,
            latency_p50_ms=250.0,
            latency_p95_ms=500.0,
            latency_p99_ms=650.0,
            total_input_tokens=5000,
            total_output_tokens=10000,
            total_cost=5.00,
            cost_per_query=0.05,
        )
        metrics_2 = AggregateMetrics(
            total_examples=100,
            successful_examples=100,
            failed_examples=0,
            mean_trust_score=0.88,
            median_trust_score=0.90,
            trust_score_std=0.08,
            mean_hallucination_rate=0.03,
            latency_p50_ms=200.0,
            latency_p95_ms=400.0,
            latency_p99_ms=550.0,
            total_input_tokens=5000,
            total_output_tokens=10000,
            total_cost=5.50,
            cost_per_query=0.055,
        )
        improvement = ImprovementMetrics(
            trust_score_delta=0.08,
            trust_score_delta_percent=10.0,
            hallucination_reduction=0.02,
            hallucination_reduction_percent=40.0,
            latency_delta_ms=-50.0,
            latency_delta_percent=-20.0,
            cost_delta_per_query=0.005,
            cost_delta_percent=10.0,
            statistical_significance=0.98,
            p_value=0.02,
            confidence_interval=(0.05, 0.11),
        )
        cost_analysis = CostPerformanceAnalysis(
            model_1_cost_per_trust_point=0.0625,
            model_2_cost_per_trust_point=0.0625,
            quality_gain_justifies_cost=True,
            break_even_volume=5000,
        )

        report = ComparisonReport(
            comparison_id="comp-001",
            model_1_id="baseline-model",
            model_2_id="finetuned-model",
            dataset_id="dataset-789",
            model_1_metrics=metrics_1,
            model_2_metrics=metrics_2,
            improvement_metrics=improvement,
            recommendation=DeploymentRecommendation.DEPLOY,
            recommendation_justification="Significant improvement in trust score",
            per_category_breakdown={},
            cost_performance_analysis=cost_analysis,
            created_at=now,
            s3_report_uri="s3://bucket/reports/comp-001",
        )

        assert report.comparison_id == "comp-001"
        assert report.model_1_id == "baseline-model"
        assert report.model_2_id == "finetuned-model"
        assert report.dataset_id == "dataset-789"
        assert report.model_1_metrics == metrics_1
        assert report.model_2_metrics == metrics_2
        assert report.improvement_metrics == improvement
        assert report.recommendation == DeploymentRecommendation.DEPLOY
        assert report.cost_performance_analysis == cost_analysis

    def test_comparison_report_invalid_s3_uri_rejected(self):
        """Test that invalid S3 URI is rejected."""
        now = datetime.now()
        metrics = AggregateMetrics(
            total_examples=100,
            successful_examples=100,
            failed_examples=0,
            mean_trust_score=0.85,
            median_trust_score=0.87,
            trust_score_std=0.10,
            mean_hallucination_rate=0.03,
            latency_p50_ms=200.0,
            latency_p95_ms=400.0,
            latency_p99_ms=500.0,
            total_input_tokens=5000,
            total_output_tokens=10000,
            total_cost=5.00,
            cost_per_query=0.05,
        )
        improvement = ImprovementMetrics(
            trust_score_delta=0.08,
            trust_score_delta_percent=10.0,
            hallucination_reduction=0.02,
            hallucination_reduction_percent=40.0,
            latency_delta_ms=-50.0,
            latency_delta_percent=-20.0,
            cost_delta_per_query=0.005,
            cost_delta_percent=10.0,
            statistical_significance=0.98,
            p_value=0.02,
            confidence_interval=(0.05, 0.11),
        )
        cost_analysis = CostPerformanceAnalysis(
            model_1_cost_per_trust_point=0.06,
            model_2_cost_per_trust_point=0.065,
            quality_gain_justifies_cost=True,
        )

        with pytest.raises(ValidationError):
            ComparisonReport(
                comparison_id="comp-002",
                model_1_id="baseline-model",
                model_2_id="finetuned-model",
                dataset_id="dataset-789",
                model_1_metrics=metrics,
                model_2_metrics=metrics,
                improvement_metrics=improvement,
                recommendation=DeploymentRecommendation.DEPLOY,
                recommendation_justification="Test",
                cost_performance_analysis=cost_analysis,
                created_at=now,
                s3_report_uri="http://bucket/reports/comp-002",
            )


class TestDataModelSerialization:
    """Test serialization of evaluation data models."""

    def test_evaluation_config_serialization(self):
        """Test EvaluationConfig JSON serialization."""
        inference_params = InferenceRequest(prompt="Test")
        config = EvaluationConfig(
            model_id="model-123",
            dataset_id="dataset-456",
            inference_params=inference_params,
        )

        # Serialize to dict
        data = config.model_dump()
        assert data["model_id"] == "model-123"
        assert data["dataset_id"] == "dataset-456"

        # Serialize to JSON
        json_str = config.model_dump_json()
        assert "model-123" in json_str
        assert "dataset-456" in json_str

    def test_comparison_report_serialization(self):
        """Test ComparisonReport JSON serialization."""
        now = datetime.now()
        metrics = AggregateMetrics(
            total_examples=100,
            successful_examples=100,
            failed_examples=0,
            mean_trust_score=0.85,
            median_trust_score=0.87,
            trust_score_std=0.10,
            mean_hallucination_rate=0.03,
            latency_p50_ms=200.0,
            latency_p95_ms=400.0,
            latency_p99_ms=500.0,
            total_input_tokens=5000,
            total_output_tokens=10000,
            total_cost=5.00,
            cost_per_query=0.05,
        )
        improvement = ImprovementMetrics(
            trust_score_delta=0.08,
            trust_score_delta_percent=10.0,
            hallucination_reduction=0.02,
            hallucination_reduction_percent=40.0,
            latency_delta_ms=-50.0,
            latency_delta_percent=-20.0,
            cost_delta_per_query=0.005,
            cost_delta_percent=10.0,
            statistical_significance=0.98,
            p_value=0.02,
            confidence_interval=(0.05, 0.11),
        )
        cost_analysis = CostPerformanceAnalysis(
            model_1_cost_per_trust_point=0.06,
            model_2_cost_per_trust_point=0.065,
            quality_gain_justifies_cost=True,
        )

        report = ComparisonReport(
            comparison_id="comp-001",
            model_1_id="baseline-model",
            model_2_id="finetuned-model",
            dataset_id="dataset-789",
            model_1_metrics=metrics,
            model_2_metrics=metrics,
            improvement_metrics=improvement,
            recommendation=DeploymentRecommendation.DEPLOY,
            recommendation_justification="Test",
            cost_performance_analysis=cost_analysis,
            created_at=now,
            s3_report_uri="s3://bucket/reports/comp-001",
        )

        # Serialize to dict
        data = report.model_dump()
        assert data["comparison_id"] == "comp-001"
        assert data["recommendation"] == "deploy"

        # Serialize to JSON
        json_str = report.model_dump_json()
        assert "comp-001" in json_str
        assert "deploy" in json_str
