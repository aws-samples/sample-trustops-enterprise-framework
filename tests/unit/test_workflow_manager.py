"""Unit tests for WorkflowManager."""

import pytest
from datetime import datetime
from moto import mock_aws
import boto3

from src.orchestration.workflow_manager import WorkflowManager


@pytest.fixture
def mock_aws_services():
    """Create mocked AWS services."""
    with mock_aws():
        # Create DynamoDB table
        dynamodb = boto3.client('dynamodb', region_name='us-east-1')
        dynamodb.create_table(
            TableName='trustops-workflows',
            KeySchema=[
                {'AttributeName': 'workflow_id', 'KeyType': 'HASH'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'workflow_id', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        
        # Create S3 bucket
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket='trustops-test-artifacts-123456789012-us-east-1')
        
        # Create CloudWatch Logs log group
        logs = boto3.client('logs', region_name='us-east-1')
        logs.create_log_group(logGroupName='/aws/trustops')
        
        yield {
            'dynamodb': dynamodb,
            's3': s3,
            'logs': logs
        }


@pytest.fixture
def workflow_manager(mock_aws_services):
    """Create WorkflowManager instance with mocked AWS services."""
    return WorkflowManager(
        dynamodb_client=mock_aws_services['dynamodb'],
        s3_client=mock_aws_services['s3'],
        logs_client=mock_aws_services['logs']
    )


def test_create_workflow_generates_unique_id(workflow_manager):
    """Test that create_workflow generates a unique workflow ID."""
    workflow_id_1 = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'test-model'}
    )
    
    workflow_id_2 = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'test-model'}
    )
    
    assert workflow_id_1 != workflow_id_2
    assert workflow_id_1.startswith('baseline-')
    assert workflow_id_2.startswith('baseline-')


def test_create_workflow_stores_manifest(workflow_manager):
    """Test that create_workflow stores manifest in DynamoDB."""
    workflow_id = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={
            'model_id': 'test-model',
            'dataset_s3_uri': 's3://test-bucket/dataset.json'
        },
        created_by='test-user'
    )
    
    # Retrieve manifest
    manifest = workflow_manager.get_workflow(workflow_id)
    
    assert manifest is not None
    assert manifest.workflow_id == workflow_id
    assert manifest.workflow_type == 'baseline'
    assert manifest.status == 'created'
    assert manifest.configuration['model_id'] == 'test-model'
    assert manifest.created_by == 'test-user'
    assert manifest.model_ids == ['test-model']


def test_update_workflow_status(workflow_manager):
    """Test updating workflow status."""
    workflow_id = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'test-model'}
    )
    
    # Update status to running
    workflow_manager.update_workflow_status(
        workflow_id,
        'running',
        {'step': 'loading_dataset'}
    )
    
    manifest = workflow_manager.get_workflow(workflow_id)
    assert manifest.status == 'running'
    assert len(manifest.events) == 1
    assert manifest.events[0]['event_type'] == 'status_update'
    assert manifest.events[0]['status'] == 'running'
    
    # Update status to completed
    workflow_manager.update_workflow_status(
        workflow_id,
        'completed',
        {'results_s3_uri': 's3://test-bucket/results.json'}
    )
    
    manifest = workflow_manager.get_workflow(workflow_id)
    assert manifest.status == 'completed'
    assert manifest.completed_at is not None
    assert manifest.results_s3_uri == 's3://test-bucket/results.json'
    assert len(manifest.events) == 2


def test_update_workflow_status_nonexistent_workflow(workflow_manager):
    """Test updating status of nonexistent workflow raises error."""
    with pytest.raises(ValueError, match="Workflow .* not found"):
        workflow_manager.update_workflow_status(
            'nonexistent-workflow',
            'running'
        )


def test_get_workflow_history_no_filters(workflow_manager):
    """Test retrieving workflow history without filters."""
    # Create multiple workflows
    workflow_id_1 = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'model-1'}
    )
    
    workflow_id_2 = workflow_manager.create_workflow(
        workflow_type='comparative',
        configuration={'model_id': 'model-2'}
    )
    
    # Get history
    history = workflow_manager.get_workflow_history()
    
    assert len(history) == 2
    workflow_ids = [w['workflow_id'] for w in history]
    assert workflow_id_1 in workflow_ids
    assert workflow_id_2 in workflow_ids


def test_get_workflow_history_with_type_filter(workflow_manager):
    """Test retrieving workflow history with type filter."""
    # Create workflows of different types
    baseline_id = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'model-1'}
    )
    
    workflow_manager.create_workflow(
        workflow_type='comparative',
        configuration={'model_id': 'model-2'}
    )
    
    # Filter by type
    history = workflow_manager.get_workflow_history(
        filters={'workflow_type': 'baseline'}
    )
    
    assert len(history) == 1
    assert history[0]['workflow_id'] == baseline_id
    assert history[0]['workflow_type'] == 'baseline'


def test_get_workflow_history_with_status_filter(workflow_manager):
    """Test retrieving workflow history with status filter."""
    # Create workflows with different statuses
    workflow_id_1 = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'model-1'}
    )
    
    workflow_id_2 = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'model-2'}
    )
    
    # Update one to completed
    workflow_manager.update_workflow_status(workflow_id_1, 'completed')
    
    # Filter by status
    history = workflow_manager.get_workflow_history(
        filters={'status': 'completed'}
    )
    
    assert len(history) == 1
    assert history[0]['workflow_id'] == workflow_id_1
    assert history[0]['status'] == 'completed'


def test_reproduce_workflow(workflow_manager):
    """Test reproducing a workflow with identical configuration."""
    # Create original workflow
    original_id = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={
            'model_id': 'test-model',
            'dataset_s3_uri': 's3://test-bucket/dataset.json',
            'hyperparameters': {'temperature': 0.7}
        },
        created_by='original-user'
    )
    
    # Reproduce workflow
    reproduced_id = workflow_manager.reproduce_workflow(
        original_id,
        created_by='reproduction-user'
    )
    
    # Verify new workflow has different ID
    assert reproduced_id != original_id
    
    # Verify configuration is identical
    original_manifest = workflow_manager.get_workflow(original_id)
    reproduced_manifest = workflow_manager.get_workflow(reproduced_id)
    
    assert reproduced_manifest.workflow_type == original_manifest.workflow_type
    assert reproduced_manifest.configuration == original_manifest.configuration
    assert reproduced_manifest.created_by == 'reproduction-user'
    assert reproduced_manifest.status == 'created'


def test_reproduce_nonexistent_workflow(workflow_manager):
    """Test reproducing nonexistent workflow raises error."""
    with pytest.raises(ValueError, match="Workflow .* not found"):
        workflow_manager.reproduce_workflow('nonexistent-workflow')


def test_extract_model_ids_from_configuration(workflow_manager):
    """Test extracting model IDs from various configuration formats."""
    # Test with single model_id
    workflow_id_1 = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'model-1'}
    )
    manifest_1 = workflow_manager.get_workflow(workflow_id_1)
    assert manifest_1.model_ids == ['model-1']
    
    # Test with baseline and finetuned model IDs
    workflow_id_2 = workflow_manager.create_workflow(
        workflow_type='comparative',
        configuration={
            'baseline_model_id': 'model-1',
            'finetuned_model_id': 'model-2'
        }
    )
    manifest_2 = workflow_manager.get_workflow(workflow_id_2)
    assert set(manifest_2.model_ids) == {'model-1', 'model-2'}
    
    # Test with base_model_id
    workflow_id_3 = workflow_manager.create_workflow(
        workflow_type='fine-tuning',
        configuration={'base_model_id': 'model-3'}
    )
    manifest_3 = workflow_manager.get_workflow(workflow_id_3)
    assert manifest_3.model_ids == ['model-3']


def test_workflow_events_logged(workflow_manager):
    """Test that workflow events are added to manifest."""
    workflow_id = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'test-model'}
    )
    
    # Update status multiple times
    workflow_manager.update_workflow_status(workflow_id, 'running')
    workflow_manager.update_workflow_status(workflow_id, 'completed')
    
    manifest = workflow_manager.get_workflow(workflow_id)
    
    # Should have 2 events (status updates)
    assert len(manifest.events) == 2
    assert manifest.events[0]['event_type'] == 'status_update'
    assert manifest.events[0]['status'] == 'running'
    assert manifest.events[1]['event_type'] == 'status_update'
    assert manifest.events[1]['status'] == 'completed'
    
    # All events should have timestamps
    for event in manifest.events:
        assert 'timestamp' in event
        assert event['timestamp']


def test_workflow_manifest_stored_in_s3(workflow_manager, mock_aws_services):
    """Test that workflow manifest is stored in S3."""
    workflow_id = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'test-model'}
    )
    
    # Check S3 for manifest
    s3 = mock_aws_services['s3']
    key = f"workflows/{workflow_id}/manifest.json"
    
    response = s3.get_object(Bucket='trustops-test-artifacts-123456789012-us-east-1', Key=key)
    content = response['Body'].read().decode('utf-8')
    
    assert workflow_id in content
    assert 'baseline' in content


def test_completed_workflow_has_completion_timestamp(workflow_manager):
    """Test that completed workflows have completion timestamp."""
    workflow_id = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'test-model'}
    )
    
    # Initially no completion timestamp
    manifest = workflow_manager.get_workflow(workflow_id)
    assert manifest.completed_at is None
    
    # Complete workflow
    workflow_manager.update_workflow_status(workflow_id, 'completed')
    
    # Should have completion timestamp
    manifest = workflow_manager.get_workflow(workflow_id)
    assert manifest.completed_at is not None
    assert isinstance(manifest.completed_at, datetime)


def test_failed_workflow_has_completion_timestamp(workflow_manager):
    """Test that failed workflows have completion timestamp."""
    workflow_id = workflow_manager.create_workflow(
        workflow_type='baseline',
        configuration={'model_id': 'test-model'}
    )
    
    # Fail workflow
    workflow_manager.update_workflow_status(
        workflow_id,
        'failed',
        {'error': 'Test error'}
    )
    
    # Should have completion timestamp
    manifest = workflow_manager.get_workflow(workflow_id)
    assert manifest.completed_at is not None
    assert manifest.status == 'failed'
