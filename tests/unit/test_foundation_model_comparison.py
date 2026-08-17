"""Unit tests for foundation model comparison orchestrator."""

import pytest
import json
import hashlib
from datetime import datetime
from unittest.mock import Mock, patch
from src.orchestration.evaluation_orchestrator import EvaluationOrchestrator
from src.data_models.model_response import (
    TrustScore,
    TrustScoreComponents
)
from src.data_models.hallucination import HallucinationAnalysis
from src.data_models.results import (
    EvaluationResult,
    BaselineMetrics,
    ComparativeEvaluationResult,
    ImprovementMetrics
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
        baseline_model_id='model-1',
        finetuned_model_id='model-2',
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
def mock_semantic_similarity():
    """Create mock semantic similarity analyzer."""
    analyzer = Mock()
    analyzer.calculate_similarity.return_value = 0.75
    return analyzer


@pytest.fixture
def orchestrator(
    mock_bedrock_client,
    mock_s3_storage,
    mock_trust_scoring,
    mock_workflow_manager,
    mock_metrics_aggregator,
    mock_semantic_similarity
):
    """Create EvaluationOrchestrator with mocked dependencies."""
    return EvaluationOrchestrator(
        bedrock_client=mock_bedrock_client,
        s3_storage=mock_s3_storage,
        trust_scoring=mock_trust_scoring,
        workflow_manager=mock_workflow_manager,
        metrics_aggregator=mock_metrics_aggregator,
        semantic_similarity=mock_semantic_similarity
    )


# Test 1: Test orchestrator with valid model IDs and dataset
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_success(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_workflow_manager
):
    """Test successful foundation model comparison with valid inputs."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    result = orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Verify result structure
    assert isinstance(result, ComparativeEvaluationResult)
    assert result.workflow_id == 'test-workflow-123'
    assert result.baseline_model_id == 'anthropic.claude-3-haiku-20240307-v1:0'
    assert result.finetuned_model_id == 'anthropic.claude-3-sonnet-20240229-v1:0'
    assert len(result.baseline_results) == 2
    assert len(result.finetuned_results) == 2
    
    # Verify workflow status updates
    assert mock_workflow_manager.update_workflow_status.call_count >= 3
    
    # Verify final status is completed
    status_calls = mock_workflow_manager.update_workflow_status.call_args_list
    final_call = status_calls[-1]
    assert final_call[0][1] == 'completed'


# Test 2: Test model ID validation (identical models rejected)
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_identical_models_rejected(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_workflow_manager
):
    """Test that identical model IDs are rejected."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_validate_different.side_effect = ValueError(
        "Cannot compare identical models. Please provide two different model IDs."
    )
    
    with pytest.raises(ValueError, match="Cannot compare identical models"):
        orchestrator.run_foundation_model_comparison(
            model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
            model_id_2='anthropic.claude-3-haiku-20240307-v1:0',
            dataset_s3_uri='s3://test-bucket/dataset.json',
            workflow_id='test-workflow-123'
        )
    
    # Verify workflow status was updated to failed
    status_calls = mock_workflow_manager.update_workflow_status.call_args_list
    failed_calls = [
        call for call in status_calls
        if call[0][1] == 'failed'
    ]
    assert len(failed_calls) == 1


# Test 3: Test dataset loading with valid S3 URI
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_loads_dataset(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_s3_storage
):
    """Test that foundation model comparison loads dataset from S3."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Verify dataset was loaded
    mock_s3_storage.download_dataset.assert_called_once_with(
        's3://test-bucket/dataset.json'
    )


# Test 4: Test dataset loading with invalid S3 URI (error handling)
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_invalid_dataset_uri(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_s3_storage,
    mock_workflow_manager
):
    """Test error handling for invalid S3 URI."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    mock_s3_storage.download_dataset.side_effect = RuntimeError(
        "Failed to load dataset from s3://invalid/dataset.json: Bucket not found"
    )
    
    with pytest.raises(RuntimeError, match="Foundation model comparison failed"):
        orchestrator.run_foundation_model_comparison(
            model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
            model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
            dataset_s3_uri='s3://invalid/dataset.json',
            workflow_id='test-workflow-123'
        )
    
    # Verify workflow status was updated to failed
    status_calls = mock_workflow_manager.update_workflow_status.call_args_list
    failed_calls = [
        call for call in status_calls
        if call[0][1] == 'failed'
    ]
    assert len(failed_calls) == 1


# Test 5: Test dataset checksum calculation
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_calculates_checksum(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_workflow_manager
):
    """Test that dataset checksum is calculated for reproducibility."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Verify checksum was included in workflow status
    status_calls = mock_workflow_manager.update_workflow_status.call_args_list
    checksum_calls = [
        call for call in status_calls
        if len(call[0]) > 2 and call[0][2].get('dataset_checksum')
    ]
    assert len(checksum_calls) >= 1
    
    # Verify checksum is a valid SHA-256 hash (64 hex characters)
    checksum = checksum_calls[0][0][2]['dataset_checksum']
    assert len(checksum) == 64
    assert all(c in '0123456789abcdef' for c in checksum)


# Test 6: Test model evaluation with mock Bedrock responses
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_invokes_both_models(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_bedrock_client
):
    """Test that both models are invoked for each example."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Should invoke model 4 times (2 examples × 2 models)
    assert mock_bedrock_client.invoke_model.call_count == 4
    
    # Verify correct model IDs are used
    calls = mock_bedrock_client.invoke_model.call_args_list
    model_1_calls = [c for c in calls[:2]]
    model_2_calls = [c for c in calls[2:]]
    
    for call in model_1_calls:
        assert call[1]['model_id'] == 'anthropic.claude-3-haiku-20240307-v1:0'
    
    for call in model_2_calls:
        assert call[1]['model_id'] == 'anthropic.claude-3-sonnet-20240229-v1:0'


# Test 7: Test identical inference parameters used for both models
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_identical_prompts(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_bedrock_client
):
    """Test that both models receive identical prompts."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Extract prompts from calls
    calls = mock_bedrock_client.invoke_model.call_args_list
    model_1_prompts = [calls[0][1]['prompt'], calls[1][1]['prompt']]
    model_2_prompts = [calls[2][1]['prompt'], calls[3][1]['prompt']]
    
    # Verify prompts are identical
    assert model_1_prompts == model_2_prompts


# Test 8: Test error handling for individual example failures
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_continues_on_example_failure(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_bedrock_client
):
    """Test that evaluation continues if individual examples fail."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    # Make first model 1 invocation fail, others succeed
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
    
    result = orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Should have 1 result for model 1 (second example succeeded)
    assert len(result.baseline_results) == 1
    # Should have matching count for model 2
    assert len(result.finetuned_results) == 1


# Test 9: Test partial results storage when some examples fail
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_stores_partial_results(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_bedrock_client,
    mock_s3_storage
):
    """Test that partial results are stored when some examples fail."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    # Make first invocation fail
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
    
    orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Verify results were stored despite failures
    mock_s3_storage.store_results.assert_called_once()


# Test 10: Test workflow state transitions (running → completed)
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_workflow_status_progression(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_workflow_manager
):
    """Test that workflow status progresses through expected stages."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
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
    details = [
        call[0][2] if len(call[0]) > 2 else {}
        for call in status_calls
    ]
    stages = [d.get('stage') for d in details if 'stage' in d]
    
    assert 'loading_dataset' in stages
    assert 'evaluating_model_1' in stages
    assert 'evaluating_model_2' in stages
    assert 'aggregating_metrics' in stages


# Test 11: Test workflow state transitions (running → failed)
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_workflow_failed_on_error(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_s3_storage,
    mock_workflow_manager
):
    """Test that workflow status is updated to failed on error."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    mock_s3_storage.download_dataset.side_effect = RuntimeError("S3 error")
    
    with pytest.raises(RuntimeError, match="Foundation model comparison failed"):
        orchestrator.run_foundation_model_comparison(
            model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
            model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
            dataset_s3_uri='s3://test-bucket/dataset.json',
            workflow_id='test-workflow-123'
        )
    
    # Verify workflow status was updated to failed
    status_calls = mock_workflow_manager.update_workflow_status.call_args_list
    failed_calls = [
        call for call in status_calls
        if call[0][1] == 'failed'
    ]
    assert len(failed_calls) == 1
    
    # Verify error details are included
    failed_call = failed_calls[0]
    assert 'error' in failed_call[0][2]


# Test 12: Test results storage in S3
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_stores_results(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_s3_storage
):
    """Test that comparison results are stored in S3."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    result = orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Verify results were stored
    mock_s3_storage.store_results.assert_called_once()
    
    # Verify S3 URI is included in result
    assert result.results_s3_uri == 's3://test-bucket/results.json'


# Test 13: Test recommendation generation for significant improvement
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_recommendation_significant(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_metrics_aggregator
):
    """Test recommendation generation for significant improvement."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    # Mock significant improvement
    mock_metrics_aggregator.compute_improvement_metrics.return_value = (
        ImprovementMetrics(
            baseline_model_id='model-1',
            finetuned_model_id='model-2',
            trust_score_improvement=0.20,
            hallucination_reduction=0.15,
            latency_delta_ms=-30.0,
            cost_delta_per_query=0.0005,
            cost_delta_percentage=25.0,
            statistical_significance=True,
            recommendation='deploy',
            justification='Significant trust score improvement with acceptable cost increase'
        )
    )
    
    result = orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Verify recommendation
    assert result.improvement_metrics.recommendation == 'deploy'
    assert result.improvement_metrics.statistical_significance is True
    assert result.improvement_metrics.trust_score_improvement == 0.20


# Test 14: Test recommendation generation for non-significant improvement
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_recommendation_non_significant(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_metrics_aggregator
):
    """Test recommendation generation for non-significant improvement."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    # Mock non-significant improvement
    mock_metrics_aggregator.compute_improvement_metrics.return_value = (
        ImprovementMetrics(
            baseline_model_id='model-1',
            finetuned_model_id='model-2',
            trust_score_improvement=0.02,
            hallucination_reduction=0.01,
            latency_delta_ms=-5.0,
            cost_delta_per_query=0.0005,
            cost_delta_percentage=25.0,
            statistical_significance=False,
            recommendation='iterate',
            justification='Trust score improvement not statistically significant'
        )
    )
    
    result = orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Verify recommendation
    assert result.improvement_metrics.recommendation == 'iterate'
    assert result.improvement_metrics.statistical_significance is False


# Test 15: Test integration with existing MetricsAggregator
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_uses_metrics_aggregator(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_metrics_aggregator
):
    """Test integration with MetricsAggregator."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Verify metrics aggregator was called for both models
    assert mock_metrics_aggregator.aggregate_baseline_metrics.call_count == 2
    
    # Verify improvement metrics were computed
    mock_metrics_aggregator.compute_improvement_metrics.assert_called_once()
    
    # Verify evaluation results were passed
    call_args = mock_metrics_aggregator.compute_improvement_metrics.call_args
    model_1_results = call_args[0][0]
    model_2_results = call_args[0][1]
    
    assert len(model_1_results) == 2
    assert len(model_2_results) == 2
    assert all(isinstance(r, EvaluationResult) for r in model_1_results)
    assert all(isinstance(r, EvaluationResult) for r in model_2_results)


# Test 16: Test integration with existing TrustScoringEngine
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_uses_trust_scoring(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_trust_scoring
):
    """Test integration with TrustScoringEngine."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Should calculate trust score 4 times (2 examples × 2 models)
    assert mock_trust_scoring.calculate_trust_score.call_count == 4
    
    # Should detect hallucinations 4 times
    assert mock_trust_scoring.detect_hallucinations.call_count == 4


# Test 17: Test integration with existing WorkflowManager
@patch('src.orchestration.evaluation_orchestrator.validate_bedrock_model_id')
@patch('src.orchestration.evaluation_orchestrator.validate_models_different')
@patch('src.orchestration.evaluation_orchestrator.get_model_metadata')
def test_run_foundation_model_comparison_uses_workflow_manager(
    mock_get_metadata,
    mock_validate_different,
    mock_validate_id,
    orchestrator,
    mock_workflow_manager
):
    """Test integration with WorkflowManager."""
    # Setup mocks
    mock_validate_id.return_value = True
    mock_get_metadata.return_value = {
        'model_family': 'Claude',
        'pricing': {'input': 0.00001, 'output': 0.00003}
    }
    
    orchestrator.run_foundation_model_comparison(
        model_id_1='anthropic.claude-3-haiku-20240307-v1:0',
        model_id_2='anthropic.claude-3-sonnet-20240229-v1:0',
        dataset_s3_uri='s3://test-bucket/dataset.json',
        workflow_id='test-workflow-123'
    )
    
    # Verify workflow manager was called multiple times
    assert mock_workflow_manager.update_workflow_status.call_count >= 5
    
    # Verify workflow_id is passed correctly
    for call in mock_workflow_manager.update_workflow_status.call_args_list:
        assert call[0][0] == 'test-workflow-123'
    
    # Verify final status includes all required metadata
    final_call = mock_workflow_manager.update_workflow_status.call_args_list[-1]
    assert final_call[0][1] == 'completed'
    details = final_call[0][2]
    
    assert 'results_s3_uri' in details
    assert 'total_examples' in details
    assert 'model_1_mean_trust_score' in details
    assert 'model_2_mean_trust_score' in details
    assert 'trust_score_improvement' in details
    assert 'recommendation' in details
    assert 'dataset_checksum' in details
