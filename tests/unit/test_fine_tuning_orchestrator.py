"""Unit tests for FineTuningOrchestrator."""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch
from src.orchestration.fine_tuning_orchestrator import FineTuningOrchestrator
from src.data_models.workflow import (
    FineTuningJob,
    DataValidationResult,
    ValidationError
)


@pytest.fixture
def mock_bedrock_client():
    """Create mock Bedrock client."""
    client = Mock()
    client.create_fine_tuning_job.return_value = {
        'job_id': 'arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job-123',
        'job_arn': 'arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job-123',
        'status': 'InProgress',
        'created_at': datetime.utcnow().isoformat()
    }
    client.get_fine_tuning_job_status.return_value = {
        'job_id': 'arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job-123',
        'status': 'Completed',
        'custom_model_arn': 'arn:aws:bedrock:us-east-1:123456789012:custom-model/test-model',
        'failure_message': None,
        'training_metrics': {
            'trainingLoss': 0.5,
            'trainingTimeInHours': 2.5
        }
    }
    return client


@pytest.fixture
def mock_s3_storage():
    """Create mock S3 storage manager."""
    storage = Mock()
    storage.download_dataset.return_value = {
        'content': json.dumps([
            {
                'prompt': 'What is AI?',
                'completion': 'AI is artificial intelligence.'
            },
            {
                'prompt': 'What is ML?',
                'completion': 'ML is machine learning.'
            },
            {
                'prompt': 'What is DL?',
                'completion': 'DL is deep learning.'
            },
            {
                'prompt': 'What is NLP?',
                'completion': 'NLP is natural language processing.'
            },
            {
                'prompt': 'What is CV?',
                'completion': 'CV is computer vision.'
            },
            {
                'prompt': 'What is RL?',
                'completion': 'RL is reinforcement learning.'
            },
            {
                'prompt': 'What is supervised learning?',
                'completion': 'Supervised learning uses labeled data.'
            },
            {
                'prompt': 'What is unsupervised learning?',
                'completion': 'Unsupervised learning finds patterns in unlabeled data.'
            },
            {
                'prompt': 'What is a neural network?',
                'completion': 'A neural network is a computational model inspired by the brain.'
            },
            {
                'prompt': 'What is a transformer?',
                'completion': 'A transformer is an architecture for sequence modeling.'
            }
        ]),
        'checksum': 'test-checksum',
        'metadata': {},
        'version_id': 'v1'
    }
    return storage


@pytest.fixture
def mock_data_validator():
    """Create mock data validator."""
    validator = Mock()
    validator.validate_training_data.return_value = DataValidationResult(
        is_valid=True,
        total_examples=10,
        valid_examples=10,
        errors=[],
        warnings=[],
        statistics={
            'avg_prompt_length': 20,
            'avg_completion_length': 40
        }
    )
    return validator


@pytest.fixture
def mock_workflow_manager():
    """Create mock workflow manager."""
    manager = Mock()
    return manager


@pytest.fixture
def orchestrator(
    mock_bedrock_client,
    mock_s3_storage,
    mock_data_validator,
    mock_workflow_manager
):
    """Create FineTuningOrchestrator with mocked dependencies."""
    return FineTuningOrchestrator(
        bedrock_client=mock_bedrock_client,
        s3_storage=mock_s3_storage,
        data_validator=mock_data_validator,
        workflow_manager=mock_workflow_manager
    )


def test_validate_training_data_success(orchestrator, mock_s3_storage, mock_data_validator):
    """Test successful training data validation."""
    result = orchestrator.validate_training_data(
        training_data_s3_uri='s3://test-bucket/training.jsonl'
    )

    # Verify result
    assert isinstance(result, DataValidationResult)
    assert result.is_valid is True
    assert result.total_examples == 10
    assert result.valid_examples == 10

    # Verify S3 download was called
    mock_s3_storage.download_dataset.assert_called_once_with(
        's3://test-bucket/training.jsonl'
    )

    # Verify validator was called
    mock_data_validator.validate_training_data.assert_called_once()


def test_validate_training_data_with_errors(
    orchestrator,
    mock_data_validator
):
    """Test validation with errors."""
    # Set up validator to return errors
    mock_data_validator.validate_training_data.return_value = DataValidationResult(
        is_valid=False,
        total_examples=10,
        valid_examples=8,
        errors=[
            ValidationError(
                error_type='missing_fields',
                message='Missing required field: completion',
                line_number=5,
                example_id='example_4'
            ),
            ValidationError(
                error_type='invalid_prompt_length',
                message='Prompt is too short',
                line_number=7,
                example_id='example_6'
            )
        ],
        warnings=['Example 3: Completion is very long'],
        statistics={}
    )

    result = orchestrator.validate_training_data(
        training_data_s3_uri='s3://test-bucket/training.jsonl'
    )

    # Verify validation failed
    assert result.is_valid is False
    assert len(result.errors) == 2
    assert result.valid_examples == 8


def test_validate_training_data_invalid_json(
    orchestrator,
    mock_s3_storage
):
    """Test validation with invalid JSON format."""
    # Return invalid JSON
    mock_s3_storage.download_dataset.return_value = {
        'content': 'invalid json content',
        'checksum': 'test-checksum',
        'metadata': {},
        'version_id': 'v1'
    }

    with pytest.raises(ValueError, match="Invalid training data format"):
        orchestrator.validate_training_data(
            training_data_s3_uri='s3://test-bucket/training.jsonl'
        )


def test_start_fine_tuning_job_success(
    orchestrator,
    mock_bedrock_client,
    mock_workflow_manager
):
    """Test successful fine-tuning job creation."""
    job = orchestrator.start_fine_tuning_job(
        base_model_id='anthropic.claude-v2',
        training_data_s3_uri='s3://test-bucket/training.jsonl',
        job_name='test-job',
        hyperparameters={'epochCount': '3'},
        workflow_id='test-workflow-123'
    )

    # Verify job object
    assert isinstance(job, FineTuningJob)
    assert job.job_name == 'test-job'
    assert job.base_model_id == 'anthropic.claude-v2'
    assert job.status == 'in_progress'
    assert job.hyperparameters == {'epochCount': '3'}

    # Verify Bedrock API was called
    mock_bedrock_client.create_fine_tuning_job.assert_called_once()
    call_args = mock_bedrock_client.create_fine_tuning_job.call_args
    assert call_args[1]['base_model_id'] == 'anthropic.claude-v2'
    assert call_args[1]['job_name'] == 'test-job'

    # Verify workflow status was updated
    mock_workflow_manager.update_workflow_status.assert_called_once()
    status_call = mock_workflow_manager.update_workflow_status.call_args
    assert status_call[0][0] == 'test-workflow-123'
    assert status_call[0][1] == 'running'


def test_start_fine_tuning_job_validates_data_first(
    orchestrator,
    mock_data_validator
):
    """Test that training data is validated before job creation."""
    orchestrator.start_fine_tuning_job(
        base_model_id='anthropic.claude-v2',
        training_data_s3_uri='s3://test-bucket/training.jsonl',
        job_name='test-job'
    )

    # Verify validation was called
    mock_data_validator.validate_training_data.assert_called_once()


def test_start_fine_tuning_job_fails_on_invalid_data(
    orchestrator,
    mock_data_validator,
    mock_bedrock_client,
    mock_workflow_manager
):
    """Test that job creation fails if validation fails."""
    # Set up validator to return errors
    mock_data_validator.validate_training_data.return_value = DataValidationResult(
        is_valid=False,
        total_examples=10,
        valid_examples=5,
        errors=[
            ValidationError(
                error_type='insufficient_samples',
                message='Dataset has 5 valid samples, minimum required is 10',
                line_number=None,
                example_id=None
            )
        ],
        warnings=[],
        statistics={}
    )

    with pytest.raises(ValueError, match="Training data validation failed"):
        orchestrator.start_fine_tuning_job(
            base_model_id='anthropic.claude-v2',
            training_data_s3_uri='s3://test-bucket/training.jsonl',
            job_name='test-job',
            workflow_id='test-workflow-123'
        )

    # Verify Bedrock API was NOT called
    mock_bedrock_client.create_fine_tuning_job.assert_not_called()

    # Verify workflow status was updated to failed
    mock_workflow_manager.update_workflow_status.assert_called_once()
    status_call = mock_workflow_manager.update_workflow_status.call_args
    assert status_call[0][1] == 'failed'


def test_start_fine_tuning_job_without_workflow_id(
    orchestrator,
    mock_workflow_manager
):
    """Test job creation without workflow tracking."""
    job = orchestrator.start_fine_tuning_job(
        base_model_id='anthropic.claude-v2',
        training_data_s3_uri='s3://test-bucket/training.jsonl',
        job_name='test-job'
    )

    # Verify job was created
    assert isinstance(job, FineTuningJob)

    # Verify workflow manager was NOT called
    mock_workflow_manager.update_workflow_status.assert_not_called()


def test_poll_job_status_completed(
    orchestrator,
    mock_bedrock_client,
    mock_workflow_manager
):
    """Test polling a completed job."""
    job = orchestrator.poll_job_status(
        job_id='arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job-123',
        workflow_id='test-workflow-123'
    )

    # Verify job status
    assert isinstance(job, FineTuningJob)
    assert job.status == 'completed'
    assert job.finetuned_model_id == 'arn:aws:bedrock:us-east-1:123456789012:custom-model/test-model'
    assert job.training_metrics is not None
    assert job.error_message is None

    # Verify Bedrock API was called
    mock_bedrock_client.get_fine_tuning_job_status.assert_called_once()

    # Verify workflow status was updated to completed
    status_calls = mock_workflow_manager.update_workflow_status.call_args_list
    final_call = status_calls[-1]
    assert final_call[0][1] == 'completed'
    assert 'finetuned_model_id' in final_call[0][2]
    assert 'total_cost' in final_call[0][2]


def test_poll_job_status_in_progress_with_backoff(
    orchestrator,
    mock_bedrock_client
):
    """Test polling with exponential backoff for in-progress job."""
    # Set up status responses: in-progress, in-progress, completed
    mock_bedrock_client.get_fine_tuning_job_status.side_effect = [
        {
            'job_id': 'test-job-123',
            'status': 'InProgress',
            'custom_model_arn': None,
            'failure_message': None,
            'training_metrics': {}
        },
        {
            'job_id': 'test-job-123',
            'status': 'InProgress',
            'custom_model_arn': None,
            'failure_message': None,
            'training_metrics': {}
        },
        {
            'job_id': 'test-job-123',
            'status': 'Completed',
            'custom_model_arn': 'arn:aws:bedrock:us-east-1:123456789012:custom-model/test-model',
            'failure_message': None,
            'training_metrics': {'trainingTimeInHours': 2.0}
        }
    ]

    with patch('time.sleep') as mock_sleep:
        job = orchestrator.poll_job_status(
            job_id='test-job-123'
        )

        # Verify job completed
        assert job.status == 'completed'

        # Verify exponential backoff was used
        assert mock_sleep.call_count == 2
        # First delay: 30 seconds
        assert mock_sleep.call_args_list[0][0][0] == 30
        # Second delay: 30 * 1.5 = 45 seconds
        assert mock_sleep.call_args_list[1][0][0] == 45


def test_poll_job_status_failed(
    orchestrator,
    mock_bedrock_client,
    mock_workflow_manager
):
    """Test polling a failed job."""
    mock_bedrock_client.get_fine_tuning_job_status.return_value = {
        'job_id': 'test-job-123',
        'status': 'Failed',
        'custom_model_arn': None,
        'failure_message': 'Training data format error',
        'training_metrics': None
    }

    with pytest.raises(RuntimeError, match="Fine-tuning job failed"):
        orchestrator.poll_job_status(
            job_id='test-job-123',
            workflow_id='test-workflow-123'
        )

    # Verify workflow status was updated to failed
    mock_workflow_manager.update_workflow_status.assert_called()
    status_call = mock_workflow_manager.update_workflow_status.call_args
    assert status_call[0][1] == 'failed'
    assert 'error' in status_call[0][2]


def test_poll_job_status_stopped(
    orchestrator,
    mock_bedrock_client
):
    """Test polling a stopped job."""
    mock_bedrock_client.get_fine_tuning_job_status.return_value = {
        'job_id': 'test-job-123',
        'status': 'Stopped',
        'custom_model_arn': None,
        'failure_message': None,
        'training_metrics': None
    }

    with pytest.raises(RuntimeError, match="Fine-tuning job was stopped"):
        orchestrator.poll_job_status(job_id='test-job-123')


def test_poll_job_status_timeout(
    orchestrator,
    mock_bedrock_client
):
    """Test polling timeout after max attempts."""
    # Always return in-progress status
    mock_bedrock_client.get_fine_tuning_job_status.return_value = {
        'job_id': 'test-job-123',
        'status': 'InProgress',
        'custom_model_arn': None,
        'failure_message': None,
        'training_metrics': {}
    }

    # Set low max attempts for testing
    orchestrator.MAX_POLL_ATTEMPTS = 3

    with patch('time.sleep'):
        with pytest.raises(TimeoutError, match="polling timeout"):
            orchestrator.poll_job_status(job_id='test-job-123')


def test_poll_job_status_without_workflow_id(
    orchestrator,
    mock_workflow_manager
):
    """Test polling without workflow tracking."""
    job = orchestrator.poll_job_status(
        job_id='test-job-123'
    )

    # Verify job was returned
    assert isinstance(job, FineTuningJob)

    # Verify workflow manager was NOT called
    mock_workflow_manager.update_workflow_status.assert_not_called()


def test_poll_job_status_logs_progress(
    orchestrator,
    mock_bedrock_client,
    mock_workflow_manager
):
    """Test that polling logs progress to workflow manager."""
    # Set up multiple status checks
    mock_bedrock_client.get_fine_tuning_job_status.side_effect = [
        {
            'job_id': 'test-job-123',
            'status': 'InProgress',
            'custom_model_arn': None,
            'failure_message': None,
            'training_metrics': {}
        },
        {
            'job_id': 'test-job-123',
            'status': 'Completed',
            'custom_model_arn': 'test-model',
            'failure_message': None,
            'training_metrics': {'trainingTimeInHours': 2.0}
        }
    ]

    with patch('time.sleep'):
        orchestrator.poll_job_status(
            job_id='test-job-123',
            workflow_id='test-workflow-123'
        )

    # Verify workflow status was updated multiple times
    assert mock_workflow_manager.update_workflow_status.call_count >= 2

    # Verify progress updates include job status
    status_calls = mock_workflow_manager.update_workflow_status.call_args_list
    progress_calls = [call for call in status_calls if call[0][1] == 'running']
    assert len(progress_calls) >= 1
    assert 'job_status' in progress_calls[0][0][2]


def test_calculate_training_cost(orchestrator):
    """Test training cost calculation."""
    cost = orchestrator._calculate_training_cost(
        {'trainingTimeInHours': 3.0}
    )

    # Verify cost calculation
    expected_cost = 3.0 * orchestrator.TRAINING_COST_PER_HOUR
    assert cost == expected_cost


def test_calculate_training_cost_default(orchestrator):
    """Test training cost calculation with default hours."""
    cost = orchestrator._calculate_training_cost({})

    # Should use default 2.0 hours
    expected_cost = 2.0 * orchestrator.TRAINING_COST_PER_HOUR
    assert cost == expected_cost


def test_calculate_storage_cost(orchestrator):
    """Test storage cost calculation."""
    cost = orchestrator._calculate_storage_cost()

    # Verify cost calculation
    expected_cost = 1.0 * orchestrator.STORAGE_COST_PER_GB_MONTH
    assert cost == expected_cost


def test_fine_tuning_job_includes_costs_on_completion(
    orchestrator,
    mock_workflow_manager
):
    """Test that completed job includes cost information."""
    orchestrator.poll_job_status(
        job_id='test-job-123',
        workflow_id='test-workflow-123'
    )

    # Get final workflow status update
    status_calls = mock_workflow_manager.update_workflow_status.call_args_list
    final_call = status_calls[-1]

    # Verify costs are included
    details = final_call[0][2]
    assert 'training_cost' in details
    assert 'storage_cost' in details
    assert 'total_cost' in details
    assert details['total_cost'] > 0


def test_validate_training_data_jsonl_format(
    orchestrator,
    mock_s3_storage,
    mock_data_validator
):
    """Test validation with JSONL format."""
    # Set up JSONL content
    mock_s3_storage.download_dataset.return_value = {
        'content': '{"prompt": "Q1", "completion": "A1"}\n{"prompt": "Q2", "completion": "A2"}',
        'checksum': 'test-checksum',
        'metadata': {},
        'version_id': 'v1'
    }

    orchestrator.validate_training_data(
        training_data_s3_uri='s3://test-bucket/training.jsonl'
    )

    # Verify validator was called with correct format
    call_args = mock_data_validator.validate_training_data.call_args
    assert call_args[1]['data_format'] == 'jsonl'


def test_validate_training_data_json_format(
    orchestrator,
    mock_s3_storage,
    mock_data_validator
):
    """Test validation with JSON format."""
    orchestrator.validate_training_data(
        training_data_s3_uri='s3://test-bucket/training.json'
    )

    # Verify validator was called with correct format
    call_args = mock_data_validator.validate_training_data.call_args
    assert call_args[1]['data_format'] == 'json'


def test_start_fine_tuning_job_default_hyperparameters(
    orchestrator,
    mock_bedrock_client
):
    """Test that default hyperparameters are used when not provided."""
    orchestrator.start_fine_tuning_job(
        base_model_id='anthropic.claude-v2',
        training_data_s3_uri='s3://test-bucket/training.jsonl',
        job_name='test-job'
    )

    # Verify Bedrock was called with hyperparameters
    call_args = mock_bedrock_client.create_fine_tuning_job.call_args
    assert 'hyperparameters' in call_args[1]


def test_poll_job_status_max_delay_cap(orchestrator, mock_bedrock_client):
    """Test that poll delay is capped at MAX_POLL_DELAY."""
    # Set up many in-progress responses
    responses = [
        {
            'job_id': 'test-job-123',
            'status': 'InProgress',
            'custom_model_arn': None,
            'failure_message': None,
            'training_metrics': {}
        }
    ] * 10 + [
        {
            'job_id': 'test-job-123',
            'status': 'Completed',
            'custom_model_arn': 'test-model',
            'failure_message': None,
            'training_metrics': {'trainingTimeInHours': 2.0}
        }
    ]
    mock_bedrock_client.get_fine_tuning_job_status.side_effect = responses

    with patch('time.sleep') as mock_sleep:
        orchestrator.poll_job_status(job_id='test-job-123')

        # Verify delays don't exceed MAX_POLL_DELAY
        for call in mock_sleep.call_args_list:
            delay = call[0][0]
            assert delay <= orchestrator.MAX_POLL_DELAY
