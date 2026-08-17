"""Unit tests for EvaluationOrchestrator."""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock
from src.orchestration.evaluation_orchestrator import EvaluationOrchestrator
from src.data_models.model_response import (
    TrustScore,
    TrustScoreComponents
)
from src.data_models.hallucination import HallucinationAnalysis
from src.data_models.results import (
    EvaluationResult,
    BaselineMetrics,
    BaselineEvaluationResult
)


@pytest.fixture
def mock_bedrock_client():
    """Create mock Bedrock client."""
    client = Mock()
    client.invoke_model.return_value = {
        'response_text': 'This is a test response.',
        'input_tokens': 10,
        'output_tokens': 20,
        'latency_ms': 150.0,
        'cost': 0.001,
        'model_id': 'test-model'
    }
    return client


@pytest.fixture
def mock_s3_storage():
    """Create mock S3 storage manager."""
    storage = Mock()
    storage.download_dataset.return_value = {
        'content': json.dumps({
            'dataset_id': 'test-dataset',
            'name': 'Test Dataset',
            'description': 'Test description',
            'examples': [
                {
                    'prompt': 'What is AI?',
                    'expected_response': 'AI is artificial intelligence.',
                    'source_documents': ['AI is a field of computer science.'],
                    'category': 'technical',
                    'metadata': {}
                },
                {
                    'prompt': 'What is ML?',
                    'expected_response': 'ML is machine learning.',
                    'source_documents': ['ML is a subset of AI.'],
                    'category': 'technical',
                    'metadata': {}
                }
            ],
            'created_at': datetime.utcnow().isoformat(),
            'version': '1.0'
        }),
        'checksum': 'test-checksum',
        'metadata': {},
        'version_id': 'v1'
    }
    storage.store_results.return_value = {
        's3_uri': 's3://test-bucket/results.json',
        'checksum': 'result-checksum',
        'compressed': False,
        'size_bytes': 1024
    }
    return storage


@pytest.fixture
def mock_trust_scoring():
    """Create mock trust scoring engine."""
    engine = Mock()
    engine.calculate_trust_score.return_value = TrustScore(
        overall_score=0.85,
        components=TrustScoreComponents(
            context_grounding=0.9,
            output_structure=0.8,
            uncertainty_indicators=0.85,
            factual_consistency=0.9,
            response_completeness=0.8
        ),
        confidence_level='high',
        flagged_for_review=False,
        explanation='Trust score 0.85: all components within acceptable range'
    )
    engine.detect_hallucinations.return_value = HallucinationAnalysis(
        has_hallucinations=False,
        hallucination_rate=0.0,
        flagged_spans=[],
        overall_grounding_score=0.9
    )
    return engine


@pytest.fixture
def mock_workflow_manager():
    """Create mock workflow manager."""
    manager = Mock()
    return manager


@pytest.fixture
def mock_metrics_aggregator():
    """Create mock metrics aggregator."""
    from src.data_models.results import ImprovementMetrics
    
    aggregator = Mock()
    aggregator.aggregate_baseline_metrics.return_value = BaselineMetrics(
        model_id='test-model',
        total_examples=2,
        mean_trust_score=0.85,
        median_trust_score=0.85,
        trust_score_distribution={'high': 2, 'medium': 0, 'low': 0},
        mean_latency_ms=150.0,
        p95_latency_ms=150.0,
        total_input_tokens=20,
        total_output_tokens=40,
        total_cost=0.002,
        hallucination_rate=0.0,
        category_breakdown={'technical': 0.85}
    )
    aggregator.compute_improvement_metrics.return_value = ImprovementMetrics(
        baseline_model_id='baseline-model',
        finetuned_model_id='finetuned-model',
        trust_score_improvement=0.15,
        hallucination_reduction=0.10,
        latency_delta_ms=-20.0,
        cost_delta_per_query=0.0005,
        cost_delta_percentage=25.0,
        statistical_significance=True,
        recommendation='deploy',
        justification='Significant improvements with acceptable cost'
    )
    return aggregator


@pytest.fixture
def orchestrator(
    mock_bedrock_client,
    mock_s3_storage,
    mock_trust_scoring,
    mock_workflow_manager,
    mock_metrics_aggregator
):
    """Create EvaluationOrchestrator with mocked dependencies."""
    mock_semantic_similarity = Mock()
    mock_semantic_similarity.calculate_similarity.return_value = 0.75
    
    return EvaluationOrchestrator(
        bedrock_client=mock_bedrock_client,
        s3_storage=mock_s3_storage,
        trust_scoring=mock_trust_scoring,
        workflow_manager=mock_workflow_manager,
        metrics_aggregator=mock_metrics_aggregator,
        semantic_similarity=mock_semantic_similarity
    )


def test_run_baseline_evaluation_success(orchestrator, mock_workflow_manager):
    """Test successful baseline evaluation."""
    result = orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Verify result structure
    assert isinstance(result, BaselineEvaluationResult)
    assert result.workflow_id == 'test-workflow-123'
    assert result.model_id == 'test-model'
    assert len(result.evaluation_results) == 2
    assert result.metrics.mean_trust_score == 0.85
    assert result.dataset_s3_uri == 's3://test-bucket/dataset.json'
    assert result.results_s3_uri == 's3://test-bucket/results.json'

    # Verify workflow status updates
    assert mock_workflow_manager.update_workflow_status.call_count >= 3

    # Verify final status is completed
    status_calls = mock_workflow_manager.update_workflow_status
    final_call = status_calls.call_args_list[-1]
    assert final_call[0][1] == 'completed'


def test_run_baseline_evaluation_loads_dataset(
    orchestrator,
    mock_s3_storage
):
    """Test that baseline evaluation loads dataset from S3."""
    orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Verify dataset was loaded
    mock_s3_storage.download_dataset.assert_called_once_with(
        's3://test-bucket/dataset.json'
    )


def test_run_baseline_evaluation_invokes_model_for_each_example(
    orchestrator,
    mock_bedrock_client
):
    """Test that model is invoked for each example in dataset."""
    orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Should invoke model twice (2 examples in mock dataset)
    assert mock_bedrock_client.invoke_model.call_count == 2

    # Verify model_id is passed correctly
    for call in mock_bedrock_client.invoke_model.call_args_list:
        assert call[1]['model_id'] == 'test-model'


def test_run_baseline_evaluation_calculates_trust_scores(
    orchestrator,
    mock_trust_scoring
):
    """Test that trust scores are calculated for each response."""
    orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Should calculate trust score twice
    assert mock_trust_scoring.calculate_trust_score.call_count == 2


def test_run_baseline_evaluation_detects_hallucinations(
    orchestrator,
    mock_trust_scoring
):
    """Test that hallucinations are detected for each response."""
    orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Should detect hallucinations twice
    assert mock_trust_scoring.detect_hallucinations.call_count == 2


def test_run_baseline_evaluation_aggregates_metrics(
    orchestrator,
    mock_metrics_aggregator
):
    """Test that metrics are aggregated after evaluation."""
    orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Should aggregate metrics once
    mock_metrics_aggregator.aggregate_baseline_metrics.assert_called_once()

    # Verify evaluation results are passed
    call_args = mock_metrics_aggregator.aggregate_baseline_metrics.call_args
    evaluation_results = call_args[0][0]
    assert len(evaluation_results) == 2
    assert all(isinstance(r, EvaluationResult) for r in evaluation_results)


def test_run_baseline_evaluation_stores_results(
    orchestrator,
    mock_s3_storage
):
    """Test that results are stored in S3."""
    orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Should store results once
    mock_s3_storage.store_results.assert_called_once()

    # Verify workflow_id and model_id are passed
    call_args = mock_s3_storage.store_results.call_args
    assert call_args[1]['workflow_id'] == 'test-workflow-123'
    assert call_args[1]['model_id'] == 'test-model'


def test_run_baseline_evaluation_tracks_tokens_and_costs(orchestrator):
    """Test that tokens and costs are tracked for each inference."""
    result = orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Verify each evaluation result has token and cost information
    for eval_result in result.evaluation_results:
        assert eval_result.model_response.input_tokens > 0
        assert eval_result.model_response.output_tokens > 0
        assert 'cost' in eval_result.model_response.metadata
        assert eval_result.model_response.metadata['cost'] > 0


def test_run_baseline_evaluation_updates_workflow_status_on_failure(
    orchestrator,
    mock_s3_storage,
    mock_workflow_manager
):
    """Test that workflow status is updated to failed on error."""
    # Make S3 storage fail
    mock_s3_storage.download_dataset.side_effect = RuntimeError(
        "S3 error"
    )

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Baseline evaluation failed"):
        orchestrator.run_baseline_evaluation(
            model_id='test-model',
            dataset_s3_uri='s3://test-bucket/dataset.json',
            workflow_id='test-workflow-123'
        )

    # Verify workflow status was updated to failed
    status_calls = mock_workflow_manager.update_workflow_status
    failed_calls = [
        call for call in status_calls.call_args_list
        if call[0][1] == 'failed'
    ]
    assert len(failed_calls) == 1


def test_run_baseline_evaluation_continues_on_example_failure(
    orchestrator,
    mock_bedrock_client
):
    """Test that evaluation continues if individual examples fail."""
    # Make first invocation fail, second succeed
    mock_bedrock_client.invoke_model.side_effect = [
        RuntimeError("Model error"),
        {
            'response_text': 'This is a test response.',
            'input_tokens': 10,
            'output_tokens': 20,
            'latency_ms': 150.0,
            'cost': 0.001,
            'model_id': 'test-model'
        }
    ]

    result = orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Should have 1 successful result (second example)
    assert len(result.evaluation_results) == 1


def test_run_baseline_evaluation_invalid_dataset_format(
    orchestrator,
    mock_s3_storage
):
    """Test that invalid dataset format raises ValueError."""
    # Return invalid JSON
    mock_s3_storage.download_dataset.return_value = {
        'content': 'invalid json',
        'checksum': 'test-checksum',
        'metadata': {},
        'version_id': 'v1'
    }

    with pytest.raises(RuntimeError, match="Baseline evaluation failed"):
        orchestrator.run_baseline_evaluation(
            model_id='test-model',
            dataset_s3_uri='s3://test-bucket/dataset.json',
            workflow_id='test-workflow-123'
        )


def test_evaluation_result_has_timestamps(orchestrator):
    """Test that evaluation results include timestamps."""
    result = orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Verify each result has timestamp
    for eval_result in result.evaluation_results:
        assert eval_result.model_response.timestamp is not None
        assert isinstance(eval_result.model_response.timestamp, datetime)


def test_evaluation_result_includes_category(orchestrator):
    """Test that evaluation results include example category."""
    result = orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Verify each result has category
    for eval_result in result.evaluation_results:
        assert eval_result.category == 'technical'


def test_evaluation_result_passed_flag(orchestrator, mock_trust_scoring):
    """Test that evaluation results have correct passed flag."""
    # Set trust score below threshold
    mock_trust_scoring.calculate_trust_score.return_value = TrustScore(
        overall_score=0.5,  # Below 0.6 threshold
        components=TrustScoreComponents(
            context_grounding=0.5,
            output_structure=0.5,
            uncertainty_indicators=0.5,
            factual_consistency=0.5,
            response_completeness=0.5
        ),
        confidence_level='low',
        flagged_for_review=True,
        explanation='Trust score 0.5: below threshold'
    )

    result = orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Verify passed flag is False
    for eval_result in result.evaluation_results:
        assert eval_result.passed is False


def test_workflow_status_progression(orchestrator, mock_workflow_manager):
    """Test that workflow status progresses through expected stages."""
    orchestrator.run_baseline_evaluation(
        model_id='test-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )

    # Get all status update calls
    status_calls = mock_workflow_manager.update_workflow_status.call_args_list

    # Verify status progression
    statuses = [call[0][1] for call in status_calls]
    assert 'running' in statuses
    assert 'completed' in statuses

    # Verify stages are logged
    details = [call[0][2] if len(call[0]) > 2 else call[1].get('details')
               for call in status_calls]
    stages = [d.get('stage') for d in details if d and 'stage' in d]

    assert 'loading_dataset' in stages
    assert 'generating_responses' in stages
    assert 'aggregating_metrics' in stages


# Tests for comparative evaluation

@pytest.fixture
def mock_semantic_similarity():
    """Create mock semantic similarity analyzer."""
    analyzer = Mock()
    analyzer.calculate_similarity.return_value = 0.75
    return analyzer


@pytest.fixture
def orchestrator_with_similarity(
    mock_bedrock_client,
    mock_s3_storage,
    mock_trust_scoring,
    mock_workflow_manager,
    mock_metrics_aggregator,
    mock_semantic_similarity
):
    """Create EvaluationOrchestrator with semantic similarity."""
    return EvaluationOrchestrator(
        bedrock_client=mock_bedrock_client,
        s3_storage=mock_s3_storage,
        trust_scoring=mock_trust_scoring,
        workflow_manager=mock_workflow_manager,
        metrics_aggregator=mock_metrics_aggregator,
        semantic_similarity=mock_semantic_similarity
    )


def test_run_comparative_evaluation_success(
    orchestrator_with_similarity,
    mock_workflow_manager,
    mock_metrics_aggregator
):
    """Test successful comparative evaluation."""
    from src.data_models.results import (
        ComparativeEvaluationResult,
        ImprovementMetrics
    )
    
    # Mock improvement metrics
    mock_metrics_aggregator.compute_improvement_metrics.return_value = (
        ImprovementMetrics(
            baseline_model_id='baseline-model',
            finetuned_model_id='finetuned-model',
            trust_score_improvement=0.15,
            hallucination_reduction=0.10,
            latency_delta_ms=-20.0,
            cost_delta_per_query=0.0005,
            cost_delta_percentage=25.0,
            statistical_significance=True,
            recommendation='deploy',
            justification='Significant improvements with acceptable cost'
        )
    )
    
    result = orchestrator_with_similarity.run_comparative_evaluation(
        baseline_model_id='baseline-model',
        finetuned_model_id='finetuned-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-456'
    )
    
    # Verify result structure
    assert isinstance(result, ComparativeEvaluationResult)
    assert result.workflow_id == 'test-workflow-456'
    assert result.baseline_model_id == 'baseline-model'
    assert result.finetuned_model_id == 'finetuned-model'
    assert len(result.baseline_results) == 2
    assert len(result.finetuned_results) == 2
    assert result.improvement_metrics.trust_score_improvement == 0.15
    
    # Verify workflow status updates
    assert mock_workflow_manager.update_workflow_status.call_count >= 3
    
    # Verify final status is completed
    status_calls = mock_workflow_manager.update_workflow_status
    final_call = status_calls.call_args_list[-1]
    assert final_call[0][1] == 'completed'


def test_run_comparative_evaluation_identical_prompts(
    orchestrator_with_similarity,
    mock_bedrock_client
):
    """Test that both models receive identical prompts."""
    orchestrator_with_similarity.run_comparative_evaluation(
        baseline_model_id='baseline-model',
        finetuned_model_id='finetuned-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-456'
    )
    
    # Should invoke model 4 times (2 examples × 2 models)
    assert mock_bedrock_client.invoke_model.call_count == 4
    
    # Extract prompts from calls
    baseline_prompts = [
        call[1]['prompt']
        for call in mock_bedrock_client.invoke_model.call_args_list[:2]
    ]
    finetuned_prompts = [
        call[1]['prompt']
        for call in mock_bedrock_client.invoke_model.call_args_list[2:]
    ]
    
    # Verify prompts are identical
    assert baseline_prompts == finetuned_prompts


def test_run_comparative_evaluation_calculates_semantic_similarity(
    orchestrator_with_similarity,
    mock_semantic_similarity
):
    """Test that semantic similarity is calculated for response pairs."""
    result = orchestrator_with_similarity.run_comparative_evaluation(
        baseline_model_id='baseline-model',
        finetuned_model_id='finetuned-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-456'
    )
    
    # Should calculate similarity twice (2 response pairs)
    assert mock_semantic_similarity.calculate_similarity.call_count == 2
    
    # Verify semantic similarity is stored in results
    for baseline_result, finetuned_result in zip(
        result.baseline_results,
        result.finetuned_results
    ):
        assert baseline_result.semantic_similarity == 0.75
        assert finetuned_result.semantic_similarity == 0.75


def test_run_comparative_evaluation_computes_improvement_metrics(
    orchestrator_with_similarity,
    mock_metrics_aggregator
):
    """Test that improvement metrics are computed."""
    orchestrator_with_similarity.run_comparative_evaluation(
        baseline_model_id='baseline-model',
        finetuned_model_id='finetuned-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-456'
    )
    
    # Should compute improvement metrics once
    mock_metrics_aggregator.compute_improvement_metrics.assert_called_once()
    
    # Verify both baseline and finetuned results are passed
    call_args = mock_metrics_aggregator.compute_improvement_metrics.call_args
    baseline_results = call_args[0][0]
    finetuned_results = call_args[0][1]
    
    assert len(baseline_results) == 2
    assert len(finetuned_results) == 2


def test_run_comparative_evaluation_stores_comparison_report(
    orchestrator_with_similarity,
    mock_s3_storage
):
    """Test that comparison report is stored in S3."""
    orchestrator_with_similarity.run_comparative_evaluation(
        baseline_model_id='baseline-model',
        finetuned_model_id='finetuned-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-456'
    )
    
    # Should store results once
    mock_s3_storage.store_results.assert_called_once()
    
    # Verify comparison report is included
    call_args = mock_s3_storage.store_results.call_args
    # call_args is a tuple of (args, kwargs)
    results_dict = call_args[1]['results']
    
    assert 'comparison_report' in results_dict
    assert 'baseline_model_id' in results_dict['comparison_report']
    assert 'finetuned_model_id' in results_dict['comparison_report']
    assert 'trust_score_improvement' in results_dict['comparison_report']
    assert 'statistical_significance' in results_dict['comparison_report']
    assert 'recommendation' in results_dict['comparison_report']


def test_run_comparative_evaluation_handles_mismatched_results(
    orchestrator_with_similarity,
    mock_bedrock_client
):
    """Test that evaluation handles mismatched result counts."""
    # Make one baseline call fail
    call_count = 0
    
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Model error")
        return {
            'response_text': 'This is a test response.',
            'input_tokens': 10,
            'output_tokens': 20,
            'latency_ms': 150.0,
            'cost': 0.001,
            'model_id': kwargs.get('model_id', 'test-model')
        }
    
    mock_bedrock_client.invoke_model.side_effect = side_effect
    
    result = orchestrator_with_similarity.run_comparative_evaluation(
        baseline_model_id='baseline-model',
        finetuned_model_id='finetuned-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-456'
    )
    
    # Should have matching counts (minimum of both)
    assert len(result.baseline_results) == len(result.finetuned_results)


def test_run_comparative_evaluation_continues_on_similarity_failure(
    orchestrator_with_similarity,
    mock_semantic_similarity
):
    """Test that evaluation continues if similarity calculation fails."""
    # Make first similarity calculation fail
    mock_semantic_similarity.calculate_similarity.side_effect = [
        RuntimeError("Similarity error"),
        0.75
    ]
    
    result = orchestrator_with_similarity.run_comparative_evaluation(
        baseline_model_id='baseline-model',
        finetuned_model_id='finetuned-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-456'
    )
    
    # Should have results with None for failed similarity
    assert result.baseline_results[0].semantic_similarity is None
    assert result.finetuned_results[0].semantic_similarity is None
    
    # Second pair should have similarity
    assert result.baseline_results[1].semantic_similarity == 0.75
    assert result.finetuned_results[1].semantic_similarity == 0.75


def test_run_comparative_evaluation_updates_workflow_status_on_failure(
    orchestrator_with_similarity,
    mock_s3_storage,
    mock_workflow_manager
):
    """Test that workflow status is updated to failed on error."""
    # Make S3 storage fail
    mock_s3_storage.download_dataset.side_effect = RuntimeError("S3 error")
    
    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Comparative evaluation failed"):
        orchestrator_with_similarity.run_comparative_evaluation(
            baseline_model_id='baseline-model',
            finetuned_model_id='finetuned-model',
            dataset_s3_uri='s3://test-bucket/dataset.json',
            workflow_id='test-workflow-456'
        )
    
    # Verify workflow status was updated to failed
    status_calls = mock_workflow_manager.update_workflow_status
    failed_calls = [
        call for call in status_calls.call_args_list
        if call[0][1] == 'failed'
    ]
    assert len(failed_calls) == 1


def test_run_comparative_evaluation_aggregates_both_models(
    orchestrator_with_similarity,
    mock_metrics_aggregator
):
    """Test that metrics are aggregated for both models."""
    orchestrator_with_similarity.run_comparative_evaluation(
        baseline_model_id='baseline-model',
        finetuned_model_id='finetuned-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-456'
    )
    
    # Should aggregate baseline metrics twice (once for each model)
    assert mock_metrics_aggregator.aggregate_baseline_metrics.call_count == 2


def test_comparative_evaluation_workflow_status_progression(
    orchestrator_with_similarity,
    mock_workflow_manager
):
    """Test that workflow status progresses through expected stages."""
    orchestrator_with_similarity.run_comparative_evaluation(
        baseline_model_id='baseline-model',
        finetuned_model_id='finetuned-model',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-456'
    )
    
    # Get all status update calls
    status_calls = mock_workflow_manager.update_workflow_status.call_args_list
    
    # Verify status progression
    statuses = [call[0][1] for call in status_calls]
    assert 'running' in statuses
    assert 'completed' in statuses
    
    # Verify stages are logged
    details = [
        call[0][2] if len(call[0]) > 2 else call[1].get('details')
        for call in status_calls
    ]
    stages = [d.get('stage') for d in details if d and 'stage' in d]
    
    assert 'loading_dataset' in stages
    assert 'generating_baseline_responses' in stages
    assert 'generating_finetuned_responses' in stages
    assert 'calculating_semantic_similarity' in stages
    assert 'aggregating_metrics' in stages
