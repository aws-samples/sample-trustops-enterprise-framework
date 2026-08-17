"""
Property-based tests for data model serialization.

**Validates: Requirements 1.2**

Property: Serialization round trip
For any valid data model instance, serializing to JSON then deserializing
should produce an equivalent object.
"""
from datetime import datetime
from hypothesis import given, settings
import hypothesis.strategies as st

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


# Strategy generators for data models
@st.composite
def evaluation_example_strategy(draw):
    """Generate random EvaluationExample instances."""
    return EvaluationExample(
        prompt=draw(st.text(min_size=1, max_size=200)),
        expected_response=draw(st.one_of(
            st.none(),
            st.text(max_size=500)
        )),
        source_documents=draw(st.lists(
            st.text(min_size=1, max_size=100),
            min_size=0,
            max_size=5
        )),
        category=draw(st.text(min_size=1, max_size=50)),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats(allow_nan=False)),
            max_size=3
        ))
    )


@st.composite
def datetime_strategy(draw):
    """Generate random datetime instances."""
    return datetime(
        year=draw(st.integers(min_value=2020, max_value=2030)),
        month=draw(st.integers(min_value=1, max_value=12)),
        day=draw(st.integers(min_value=1, max_value=28)),
        hour=draw(st.integers(min_value=0, max_value=23)),
        minute=draw(st.integers(min_value=0, max_value=59)),
        second=draw(st.integers(min_value=0, max_value=59))
    )


@st.composite
def evaluation_dataset_strategy(draw):
    """Generate random EvaluationDataset instances."""
    return EvaluationDataset(
        dataset_id=draw(st.text(min_size=1, max_size=50)),
        name=draw(st.text(min_size=1, max_size=100)),
        description=draw(st.text(max_size=200)),
        examples=draw(st.lists(
            evaluation_example_strategy(),
            min_size=1,
            max_size=10
        )),
        created_at=draw(datetime_strategy()),
        version=draw(st.text(min_size=1, max_size=20))
    )


@st.composite
def model_response_strategy(draw):
    """Generate random ModelResponse instances."""
    return ModelResponse(
        response_id=draw(st.text(min_size=1, max_size=50)),
        model_id=draw(st.text(min_size=1, max_size=50)),
        prompt=draw(st.text(min_size=1, max_size=200)),
        response_text=draw(st.text(max_size=500)),
        input_tokens=draw(st.integers(min_value=1, max_value=10000)),
        output_tokens=draw(st.integers(min_value=1, max_value=10000)),
        latency_ms=draw(st.floats(
            min_value=0.1,
            max_value=10000.0,
            allow_nan=False,
            allow_infinity=False
        )),
        timestamp=draw(datetime_strategy()),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers()),
            max_size=3
        ))
    )


@st.composite
def trust_score_components_strategy(draw):
    """Generate random TrustScoreComponents instances."""
    return TrustScoreComponents(
        context_grounding=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        output_structure=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        uncertainty_indicators=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        factual_consistency=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        response_completeness=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        ))
    )


@st.composite
def trust_score_strategy(draw):
    """Generate random TrustScore instances."""
    return TrustScore(
        overall_score=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        components=draw(trust_score_components_strategy()),
        confidence_level=draw(st.sampled_from(["high", "medium", "low"])),
        flagged_for_review=draw(st.booleans()),
        explanation=draw(st.text(max_size=200))
    )


@st.composite
def hallucination_span_strategy(draw):
    """Generate random HallucinationSpan instances."""
    start = draw(st.integers(min_value=0, max_value=100))
    end = draw(st.integers(min_value=start, max_value=start + 50))
    return HallucinationSpan(
        text=draw(st.text(min_size=1, max_size=100)),
        start_idx=start,
        end_idx=end,
        grounding_score=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        evidence_documents=draw(st.lists(
            st.tuples(
                st.text(min_size=1, max_size=50),
                st.floats(
                    min_value=0.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False
                )
            ),
            max_size=3
        ))
    )


@st.composite
def hallucination_analysis_strategy(draw):
    """Generate random HallucinationAnalysis instances."""
    return HallucinationAnalysis(
        has_hallucinations=draw(st.booleans()),
        hallucination_rate=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        flagged_spans=draw(st.lists(
            hallucination_span_strategy(),
            max_size=5
        )),
        overall_grounding_score=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        ))
    )


@st.composite
def evaluation_result_strategy(draw):
    """Generate random EvaluationResult instances."""
    return EvaluationResult(
        example_id=draw(st.text(min_size=1, max_size=50)),
        model_response=draw(model_response_strategy()),
        trust_score=draw(trust_score_strategy()),
        hallucination_analysis=draw(hallucination_analysis_strategy()),
        semantic_similarity=draw(st.one_of(
            st.none(),
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            )
        )),
        category=draw(st.text(min_size=1, max_size=50)),
        passed=draw(st.booleans())
    )


@st.composite
def baseline_metrics_strategy(draw):
    """Generate random BaselineMetrics instances."""
    return BaselineMetrics(
        model_id=draw(st.text(min_size=1, max_size=50)),
        total_examples=draw(st.integers(min_value=1, max_value=1000)),
        mean_trust_score=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        median_trust_score=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        trust_score_distribution=draw(st.dictionaries(
            st.sampled_from(["high", "medium", "low"]),
            st.integers(min_value=0, max_value=100),
            min_size=3,
            max_size=3
        )),
        mean_latency_ms=draw(st.floats(
            min_value=0.1,
            max_value=10000.0,
            allow_nan=False,
            allow_infinity=False
        )),
        p95_latency_ms=draw(st.floats(
            min_value=0.1,
            max_value=10000.0,
            allow_nan=False,
            allow_infinity=False
        )),
        total_input_tokens=draw(st.integers(min_value=0, max_value=100000)),
        total_output_tokens=draw(st.integers(min_value=0, max_value=100000)),
        total_cost=draw(st.floats(
            min_value=0.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False
        )),
        hallucination_rate=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        category_breakdown=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False
            ),
            max_size=5
        ))
    )


@st.composite
def improvement_metrics_strategy(draw):
    """Generate random ImprovementMetrics instances."""
    return ImprovementMetrics(
        baseline_model_id=draw(st.text(min_size=1, max_size=50)),
        finetuned_model_id=draw(st.text(min_size=1, max_size=50)),
        trust_score_improvement=draw(st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        hallucination_reduction=draw(st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        latency_delta_ms=draw(st.floats(
            min_value=-10000.0,
            max_value=10000.0,
            allow_nan=False,
            allow_infinity=False
        )),
        cost_delta_per_query=draw(st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False
        )),
        cost_delta_percentage=draw(st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False
        )),
        statistical_significance=draw(st.booleans()),
        recommendation=draw(st.sampled_from(
            ["deploy", "iterate", "reject"]
        )),
        justification=draw(st.text(max_size=200))
    )


@st.composite
def cost_performance_metrics_strategy(draw):
    """Generate random CostPerformanceMetrics instances."""
    return CostPerformanceMetrics(
        total_cost=draw(st.floats(
            min_value=0.0,
            max_value=10000.0,
            allow_nan=False,
            allow_infinity=False
        )),
        cost_per_query=draw(st.floats(
            min_value=0.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False
        )),
        cost_per_high_trust_response=draw(st.floats(
            min_value=0.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False
        )),
        cost_per_token=draw(st.floats(
            min_value=0.0,
            max_value=0.01,
            allow_nan=False,
            allow_infinity=False
        )),
        mean_trust_score=draw(st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        )),
        cost_efficiency_score=draw(st.floats(
            min_value=0.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False
        )),
        projected_monthly_cost=draw(st.dictionaries(
            st.integers(min_value=100, max_value=100000),
            st.floats(
                min_value=0.0,
                max_value=100000.0,
                allow_nan=False,
                allow_infinity=False
            ),
            max_size=5
        ))
    )


@st.composite
def fine_tuning_job_strategy(draw):
    """Generate random FineTuningJob instances."""
    created = draw(datetime_strategy())
    completed = draw(st.one_of(
        st.none(),
        datetime_strategy()
    ))
    return FineTuningJob(
        job_id=draw(st.text(min_size=1, max_size=50)),
        job_name=draw(st.text(min_size=1, max_size=100)),
        base_model_id=draw(st.text(min_size=1, max_size=50)),
        training_data_s3_uri=draw(st.text(min_size=1, max_size=200)),
        status=draw(st.sampled_from(
            ["in_progress", "completed", "failed"]
        )),
        hyperparameters=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.integers(), st.floats(allow_nan=False)),
            max_size=5
        )),
        training_metrics=draw(st.one_of(
            st.none(),
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.floats(allow_nan=False, allow_infinity=False),
                max_size=5
            )
        )),
        finetuned_model_id=draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=50)
        )),
        created_at=created,
        completed_at=completed,
        error_message=draw(st.one_of(
            st.none(),
            st.text(max_size=200)
        ))
    )


@st.composite
def workflow_manifest_strategy(draw):
    """Generate random WorkflowManifest instances."""
    created = draw(datetime_strategy())
    completed = draw(st.one_of(
        st.none(),
        datetime_strategy()
    ))
    return WorkflowManifest(
        workflow_id=draw(st.text(min_size=1, max_size=50)),
        workflow_type=draw(st.sampled_from(
            ["baseline", "comparative", "fine-tuning"]
        )),
        status=draw(st.text(min_size=1, max_size=20)),
        configuration=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers()),
            max_size=5
        )),
        dataset_s3_uri=draw(st.text(min_size=1, max_size=200)),
        dataset_checksum=draw(st.text(min_size=1, max_size=64)),
        model_ids=draw(st.lists(
            st.text(min_size=1, max_size=50),
            min_size=1,
            max_size=5
        )),
        results_s3_uri=draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=200)
        )),
        created_at=created,
        completed_at=completed,
        created_by=draw(st.text(min_size=1, max_size=100)),
        events=draw(st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(st.text(), st.integers()),
                max_size=3
            ),
            max_size=10
        ))
    )


@st.composite
def validation_error_strategy(draw):
    """Generate random ValidationError instances."""
    return ValidationError(
        error_type=draw(st.text(min_size=1, max_size=50)),
        message=draw(st.text(min_size=1, max_size=200)),
        line_number=draw(st.one_of(
            st.none(),
            st.integers(min_value=1, max_value=10000)
        )),
        example_id=draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=50)
        ))
    )


@st.composite
def data_validation_result_strategy(draw):
    """Generate random DataValidationResult instances."""
    total = draw(st.integers(min_value=1, max_value=1000))
    valid = draw(st.integers(min_value=0, max_value=total))
    return DataValidationResult(
        is_valid=draw(st.booleans()),
        total_examples=total,
        valid_examples=valid,
        errors=draw(st.lists(
            validation_error_strategy(),
            max_size=10
        )),
        warnings=draw(st.lists(
            st.text(max_size=100),
            max_size=10
        )),
        statistics=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.integers(), st.floats(allow_nan=False)),
            max_size=5
        ))
    )



# Property tests for serialization round trip
@settings(max_examples=100)
@given(example=evaluation_example_strategy())
def test_evaluation_example_serialization_roundtrip(example):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid EvaluationExample instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = example.to_json()
    restored = EvaluationExample.from_json(json_str)

    assert restored.prompt == example.prompt
    assert restored.expected_response == example.expected_response
    assert restored.source_documents == example.source_documents
    assert restored.category == example.category
    assert restored.metadata == example.metadata


@settings(max_examples=100)
@given(dataset=evaluation_dataset_strategy())
def test_evaluation_dataset_serialization_roundtrip(dataset):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid EvaluationDataset instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = dataset.to_json()
    restored = EvaluationDataset.from_json(json_str)

    assert restored.dataset_id == dataset.dataset_id
    assert restored.name == dataset.name
    assert restored.description == dataset.description
    assert len(restored.examples) == len(dataset.examples)
    assert restored.created_at == dataset.created_at
    assert restored.version == dataset.version


@settings(max_examples=100)
@given(response=model_response_strategy())
def test_model_response_serialization_roundtrip(response):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid ModelResponse instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = response.to_json()
    restored = ModelResponse.from_json(json_str)

    assert restored.response_id == response.response_id
    assert restored.model_id == response.model_id
    assert restored.prompt == response.prompt
    assert restored.response_text == response.response_text
    assert restored.input_tokens == response.input_tokens
    assert restored.output_tokens == response.output_tokens
    assert restored.latency_ms == response.latency_ms
    assert restored.timestamp == response.timestamp
    assert restored.metadata == response.metadata


@settings(max_examples=100)
@given(components=trust_score_components_strategy())
def test_trust_score_components_serialization_roundtrip(components):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid TrustScoreComponents instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = components.to_json()
    restored = TrustScoreComponents.from_json(json_str)

    assert restored.context_grounding == components.context_grounding
    assert restored.output_structure == components.output_structure
    assert (
        restored.uncertainty_indicators == components.uncertainty_indicators
    )
    assert restored.factual_consistency == components.factual_consistency
    assert (
        restored.response_completeness == components.response_completeness
    )


@settings(max_examples=100)
@given(trust_score=trust_score_strategy())
def test_trust_score_serialization_roundtrip(trust_score):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid TrustScore instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = trust_score.to_json()
    restored = TrustScore.from_json(json_str)

    assert restored.overall_score == trust_score.overall_score
    assert restored.confidence_level == trust_score.confidence_level
    assert restored.flagged_for_review == trust_score.flagged_for_review
    assert restored.explanation == trust_score.explanation
    assert (
        restored.components.context_grounding ==
        trust_score.components.context_grounding
    )


@settings(max_examples=100)
@given(span=hallucination_span_strategy())
def test_hallucination_span_serialization_roundtrip(span):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid HallucinationSpan instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = span.to_json()
    restored = HallucinationSpan.from_json(json_str)

    assert restored.text == span.text
    assert restored.start_idx == span.start_idx
    assert restored.end_idx == span.end_idx
    assert restored.grounding_score == span.grounding_score
    assert restored.evidence_documents == span.evidence_documents


@settings(max_examples=100)
@given(analysis=hallucination_analysis_strategy())
def test_hallucination_analysis_serialization_roundtrip(analysis):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid HallucinationAnalysis instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = analysis.to_json()
    restored = HallucinationAnalysis.from_json(json_str)

    assert restored.has_hallucinations == analysis.has_hallucinations
    assert restored.hallucination_rate == analysis.hallucination_rate
    assert len(restored.flagged_spans) == len(analysis.flagged_spans)
    assert (
        restored.overall_grounding_score == analysis.overall_grounding_score
    )



@settings(max_examples=100)
@given(result=evaluation_result_strategy())
def test_evaluation_result_serialization_roundtrip(result):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid EvaluationResult instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = result.to_json()
    restored = EvaluationResult.from_json(json_str)

    assert restored.example_id == result.example_id
    assert restored.category == result.category
    assert restored.passed == result.passed
    assert restored.semantic_similarity == result.semantic_similarity
    assert (
        restored.model_response.response_id ==
        result.model_response.response_id
    )
    assert (
        restored.trust_score.overall_score ==
        result.trust_score.overall_score
    )


@settings(max_examples=100)
@given(metrics=baseline_metrics_strategy())
def test_baseline_metrics_serialization_roundtrip(metrics):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid BaselineMetrics instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = metrics.to_json()
    restored = BaselineMetrics.from_json(json_str)

    assert restored.model_id == metrics.model_id
    assert restored.total_examples == metrics.total_examples
    assert restored.mean_trust_score == metrics.mean_trust_score
    assert restored.median_trust_score == metrics.median_trust_score
    assert (
        restored.trust_score_distribution ==
        metrics.trust_score_distribution
    )
    assert restored.mean_latency_ms == metrics.mean_latency_ms
    assert restored.p95_latency_ms == metrics.p95_latency_ms
    assert restored.total_input_tokens == metrics.total_input_tokens
    assert restored.total_output_tokens == metrics.total_output_tokens
    assert restored.total_cost == metrics.total_cost
    assert restored.hallucination_rate == metrics.hallucination_rate
    assert restored.category_breakdown == metrics.category_breakdown


@settings(max_examples=100)
@given(metrics=improvement_metrics_strategy())
def test_improvement_metrics_serialization_roundtrip(metrics):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid ImprovementMetrics instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = metrics.to_json()
    restored = ImprovementMetrics.from_json(json_str)

    assert restored.baseline_model_id == metrics.baseline_model_id
    assert restored.finetuned_model_id == metrics.finetuned_model_id
    assert (
        restored.trust_score_improvement == metrics.trust_score_improvement
    )
    assert (
        restored.hallucination_reduction == metrics.hallucination_reduction
    )
    assert restored.latency_delta_ms == metrics.latency_delta_ms
    assert restored.cost_delta_per_query == metrics.cost_delta_per_query
    assert restored.cost_delta_percentage == metrics.cost_delta_percentage
    assert (
        restored.statistical_significance ==
        metrics.statistical_significance
    )
    assert restored.recommendation == metrics.recommendation
    assert restored.justification == metrics.justification


@settings(max_examples=100)
@given(metrics=cost_performance_metrics_strategy())
def test_cost_performance_metrics_serialization_roundtrip(metrics):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid CostPerformanceMetrics instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = metrics.to_json()
    restored = CostPerformanceMetrics.from_json(json_str)

    assert restored.total_cost == metrics.total_cost
    assert restored.cost_per_query == metrics.cost_per_query
    assert (
        restored.cost_per_high_trust_response ==
        metrics.cost_per_high_trust_response
    )
    assert restored.cost_per_token == metrics.cost_per_token
    assert restored.mean_trust_score == metrics.mean_trust_score
    assert restored.cost_efficiency_score == metrics.cost_efficiency_score
    assert (
        restored.projected_monthly_cost == metrics.projected_monthly_cost
    )


@settings(max_examples=100)
@given(job=fine_tuning_job_strategy())
def test_fine_tuning_job_serialization_roundtrip(job):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid FineTuningJob instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = job.to_json()
    restored = FineTuningJob.from_json(json_str)

    assert restored.job_id == job.job_id
    assert restored.job_name == job.job_name
    assert restored.base_model_id == job.base_model_id
    assert restored.training_data_s3_uri == job.training_data_s3_uri
    assert restored.status == job.status
    assert restored.hyperparameters == job.hyperparameters
    assert restored.training_metrics == job.training_metrics
    assert restored.finetuned_model_id == job.finetuned_model_id
    assert restored.created_at == job.created_at
    assert restored.completed_at == job.completed_at
    assert restored.error_message == job.error_message


@settings(max_examples=100)
@given(manifest=workflow_manifest_strategy())
def test_workflow_manifest_serialization_roundtrip(manifest):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid WorkflowManifest instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = manifest.to_json()
    restored = WorkflowManifest.from_json(json_str)

    assert restored.workflow_id == manifest.workflow_id
    assert restored.workflow_type == manifest.workflow_type
    assert restored.status == manifest.status
    assert restored.configuration == manifest.configuration
    assert restored.dataset_s3_uri == manifest.dataset_s3_uri
    assert restored.dataset_checksum == manifest.dataset_checksum
    assert restored.model_ids == manifest.model_ids
    assert restored.results_s3_uri == manifest.results_s3_uri
    assert restored.created_at == manifest.created_at
    assert restored.completed_at == manifest.completed_at
    assert restored.created_by == manifest.created_by
    assert restored.events == manifest.events


@settings(max_examples=100)
@given(error=validation_error_strategy())
def test_validation_error_serialization_roundtrip(error):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid ValidationError instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = error.to_json()
    restored = ValidationError.from_json(json_str)

    assert restored.error_type == error.error_type
    assert restored.message == error.message
    assert restored.line_number == error.line_number
    assert restored.example_id == error.example_id


@settings(max_examples=100)
@given(result=data_validation_result_strategy())
def test_data_validation_result_serialization_roundtrip(result):
    """
    Feature: trustops-aws-demo, Property: Serialization round trip
    **Validates: Requirements 1.2**

    For any valid DataValidationResult instance, serializing to JSON then
    deserializing should produce an equivalent object.
    """
    json_str = result.to_json()
    restored = DataValidationResult.from_json(json_str)

    assert restored.is_valid == result.is_valid
    assert restored.total_examples == result.total_examples
    assert restored.valid_examples == result.valid_examples
    assert len(restored.errors) == len(result.errors)
    assert restored.warnings == result.warnings
    assert restored.statistics == result.statistics
