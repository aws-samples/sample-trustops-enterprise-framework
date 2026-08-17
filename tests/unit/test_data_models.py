"""
Unit tests for data model serialization and deserialization.
"""
import pytest
from datetime import datetime
from src.data_models import (
    EvaluationExample,
    EvaluationDataset,
    ModelResponse,
    TrustScore,
    TrustScoreComponents,
    HallucinationSpan,
    HallucinationAnalysis,
    EvaluationResult,
    BaselineMetrics,
    ImprovementMetrics,
    CostPerformanceMetrics,
    FineTuningJob,
    WorkflowManifest,
    DataValidationResult,
    ValidationError
)


class TestEvaluationModels:
    """Test evaluation data models."""

    def test_evaluation_example_serialization(self):
        """Test EvaluationExample JSON serialization round trip."""
        example = EvaluationExample(
            prompt="What is AI?",
            expected_response="Artificial Intelligence",
            source_documents=["doc1", "doc2"],
            category="technical",
            metadata={"key": "value"}
        )

        # Serialize and deserialize
        json_str = example.to_json()
        restored = EvaluationExample.from_json(json_str)

        assert restored.prompt == example.prompt
        assert restored.expected_response == example.expected_response
        assert restored.source_documents == example.source_documents
        assert restored.category == example.category
        assert restored.metadata == example.metadata

    def test_evaluation_dataset_serialization(self):
        """Test EvaluationDataset JSON serialization round trip."""
        now = datetime.now()
        dataset = EvaluationDataset(
            dataset_id="ds-001",
            name="Test Dataset",
            description="A test dataset",
            examples=[
                EvaluationExample(
                    prompt="Test prompt",
                    expected_response="Test response",
                    source_documents=["doc1"],
                    category="test",
                    metadata={}
                )
            ],
            created_at=now,
            version="1.0"
        )

        # Serialize and deserialize
        json_str = dataset.to_json()
        restored = EvaluationDataset.from_json(json_str)

        assert restored.dataset_id == dataset.dataset_id
        assert restored.name == dataset.name
        assert restored.description == dataset.description
        assert len(restored.examples) == 1
        assert restored.examples[0].prompt == "Test prompt"
        assert restored.version == dataset.version


class TestModelResponseModels:
    """Test model response and trust score models."""

    def test_model_response_serialization(self):
        """Test ModelResponse JSON serialization round trip."""
        now = datetime.now()
        response = ModelResponse(
            response_id="resp-001",
            model_id="claude-v1",
            prompt="Test prompt",
            response_text="Test response",
            input_tokens=10,
            output_tokens=20,
            latency_ms=150.5,
            timestamp=now,
            metadata={"key": "value"}
        )

        # Serialize and deserialize
        json_str = response.to_json()
        restored = ModelResponse.from_json(json_str)

        assert restored.response_id == response.response_id
        assert restored.model_id == response.model_id
        assert restored.input_tokens == response.input_tokens
        assert restored.output_tokens == response.output_tokens
        assert restored.latency_ms == response.latency_ms

    def test_trust_score_components_serialization(self):
        """Test TrustScoreComponents JSON serialization round trip."""
        components = TrustScoreComponents(
            context_grounding=0.8,
            output_structure=0.9,
            uncertainty_indicators=0.7,
            factual_consistency=0.85,
            response_completeness=0.95
        )

        # Serialize and deserialize
        json_str = components.to_json()
        restored = TrustScoreComponents.from_json(json_str)

        assert restored.context_grounding == components.context_grounding
        assert restored.output_structure == components.output_structure
        assert (
            restored.uncertainty_indicators ==
            components.uncertainty_indicators
        )
        assert (
            restored.factual_consistency == components.factual_consistency
        )
        assert (
            restored.response_completeness ==
            components.response_completeness
        )

    def test_trust_score_serialization(self):
        """Test TrustScore JSON serialization round trip."""
        components = TrustScoreComponents(
            context_grounding=0.8,
            output_structure=0.9,
            uncertainty_indicators=0.7,
            factual_consistency=0.85,
            response_completeness=0.95
        )
        trust_score = TrustScore(
            overall_score=0.85,
            components=components,
            confidence_level="high",
            flagged_for_review=False,
            explanation="High confidence response"
        )

        # Serialize and deserialize
        json_str = trust_score.to_json()
        restored = TrustScore.from_json(json_str)

        assert restored.overall_score == trust_score.overall_score
        assert restored.confidence_level == trust_score.confidence_level
        assert restored.flagged_for_review == trust_score.flagged_for_review
        assert restored.explanation == trust_score.explanation
        assert (
            restored.components.context_grounding ==
            components.context_grounding
        )


class TestHallucinationModels:
    """Test hallucination detection models."""

    def test_hallucination_span_serialization(self):
        """Test HallucinationSpan JSON serialization round trip."""
        span = HallucinationSpan(
            text="Unsupported claim",
            start_idx=10,
            end_idx=27,
            grounding_score=0.3,
            evidence_documents=[("doc1", 0.2), ("doc2", 0.4)]
        )

        # Serialize and deserialize
        json_str = span.to_json()
        restored = HallucinationSpan.from_json(json_str)

        assert restored.text == span.text
        assert restored.start_idx == span.start_idx
        assert restored.end_idx == span.end_idx
        assert restored.grounding_score == span.grounding_score
        assert len(restored.evidence_documents) == 2

    def test_hallucination_analysis_serialization(self):
        """Test HallucinationAnalysis JSON serialization round trip."""
        analysis = HallucinationAnalysis(
            has_hallucinations=True,
            hallucination_rate=0.25,
            flagged_spans=[
                HallucinationSpan(
                    text="Claim",
                    start_idx=0,
                    end_idx=5,
                    grounding_score=0.2,
                    evidence_documents=[]
                )
            ],
            overall_grounding_score=0.75
        )

        # Serialize and deserialize
        json_str = analysis.to_json()
        restored = HallucinationAnalysis.from_json(json_str)

        assert restored.has_hallucinations == analysis.has_hallucinations
        assert restored.hallucination_rate == analysis.hallucination_rate
        assert len(restored.flagged_spans) == 1
        assert (
            restored.overall_grounding_score ==
            analysis.overall_grounding_score
        )


class TestResultModels:
    """Test result and metrics models."""

    def test_baseline_metrics_serialization(self):
        """Test BaselineMetrics JSON serialization round trip."""
        metrics = BaselineMetrics(
            model_id="claude-v1",
            total_examples=100,
            mean_trust_score=0.85,
            median_trust_score=0.87,
            trust_score_distribution={"high": 70, "medium": 25, "low": 5},
            mean_latency_ms=150.0,
            p95_latency_ms=250.0,
            total_input_tokens=1000,
            total_output_tokens=2000,
            total_cost=0.50,
            hallucination_rate=0.05,
            category_breakdown={"technical": 0.9, "general": 0.8}
        )

        # Serialize and deserialize
        json_str = metrics.to_json()
        restored = BaselineMetrics.from_json(json_str)

        assert restored.model_id == metrics.model_id
        assert restored.total_examples == metrics.total_examples
        assert restored.mean_trust_score == metrics.mean_trust_score
        assert restored.total_cost == metrics.total_cost

    def test_improvement_metrics_serialization(self):
        """Test ImprovementMetrics JSON serialization round trip."""
        metrics = ImprovementMetrics(
            baseline_model_id="claude-v1",
            finetuned_model_id="claude-v1-ft",
            trust_score_improvement=0.05,
            hallucination_reduction=0.02,
            latency_delta_ms=-10.0,
            cost_delta_per_query=0.001,
            cost_delta_percentage=2.0,
            statistical_significance=True,
            recommendation="deploy",
            justification="Significant improvement in trust score"
        )

        # Serialize and deserialize
        json_str = metrics.to_json()
        restored = ImprovementMetrics.from_json(json_str)

        assert restored.baseline_model_id == metrics.baseline_model_id
        assert restored.finetuned_model_id == metrics.finetuned_model_id
        assert (
            restored.trust_score_improvement ==
            metrics.trust_score_improvement
        )
        assert restored.recommendation == metrics.recommendation

    def test_cost_performance_metrics_serialization(self):
        """Test CostPerformanceMetrics JSON serialization round trip."""
        metrics = CostPerformanceMetrics(
            total_cost=100.0,
            cost_per_query=0.01,
            cost_per_high_trust_response=0.012,
            cost_per_token=0.00001,
            mean_trust_score=0.85,
            cost_efficiency_score=8.5,
            projected_monthly_cost={1000: 10.0, 10000: 100.0}
        )

        # Serialize and deserialize
        json_str = metrics.to_json()
        restored = CostPerformanceMetrics.from_json(json_str)

        assert restored.total_cost == metrics.total_cost
        assert restored.cost_per_query == metrics.cost_per_query
        assert (
            restored.cost_efficiency_score == metrics.cost_efficiency_score
        )
        assert len(restored.projected_monthly_cost) == 2


class TestWorkflowModels:
    """Test workflow and fine-tuning models."""

    def test_fine_tuning_job_serialization(self):
        """Test FineTuningJob JSON serialization round trip."""
        now = datetime.now()
        job = FineTuningJob(
            job_id="job-001",
            job_name="Test Job",
            base_model_id="claude-v1",
            training_data_s3_uri="s3://bucket/data.jsonl",
            status="completed",
            hyperparameters={"epochs": 3, "batch_size": 32},
            training_metrics={"loss": 0.5},
            finetuned_model_id="claude-v1-ft",
            created_at=now,
            completed_at=now,
            error_message=None
        )

        # Serialize and deserialize
        json_str = job.to_json()
        restored = FineTuningJob.from_json(json_str)

        assert restored.job_id == job.job_id
        assert restored.job_name == job.job_name
        assert restored.status == job.status
        assert restored.finetuned_model_id == job.finetuned_model_id

    def test_workflow_manifest_serialization(self):
        """Test WorkflowManifest JSON serialization round trip."""
        now = datetime.now()
        manifest = WorkflowManifest(
            workflow_id="wf-001",
            workflow_type="baseline",
            status="completed",
            configuration={"model": "claude-v1"},
            dataset_s3_uri="s3://bucket/dataset.json",
            dataset_checksum="abc123",
            model_ids=["claude-v1"],
            results_s3_uri="s3://bucket/results.json",
            created_at=now,
            completed_at=now,
            created_by="user@example.com",
            events=[{"event": "started", "timestamp": now.isoformat()}]
        )

        # Serialize and deserialize
        json_str = manifest.to_json()
        restored = WorkflowManifest.from_json(json_str)

        assert restored.workflow_id == manifest.workflow_id
        assert restored.workflow_type == manifest.workflow_type
        assert restored.status == manifest.status
        assert len(restored.events) == 1

    def test_validation_error_serialization(self):
        """Test ValidationError JSON serialization round trip."""
        error = ValidationError(
            error_type="format_error",
            message="Invalid JSON format",
            line_number=10,
            example_id="ex-001"
        )

        # Serialize and deserialize
        json_str = error.to_json()
        restored = ValidationError.from_json(json_str)

        assert restored.error_type == error.error_type
        assert restored.message == error.message
        assert restored.line_number == error.line_number
        assert restored.example_id == error.example_id

    def test_data_validation_result_serialization(self):
        """Test DataValidationResult JSON serialization round trip."""
        result = DataValidationResult(
            is_valid=False,
            total_examples=100,
            valid_examples=95,
            errors=[
                ValidationError(
                    error_type="format_error",
                    message="Invalid format",
                    line_number=10,
                    example_id=None
                )
            ],
            warnings=["Warning message"],
            statistics={"avg_length": 100}
        )

        # Serialize and deserialize
        json_str = result.to_json()
        restored = DataValidationResult.from_json(json_str)

        assert restored.is_valid == result.is_valid
        assert restored.total_examples == result.total_examples
        assert restored.valid_examples == result.valid_examples
        assert len(restored.errors) == 1
        assert len(restored.warnings) == 1
