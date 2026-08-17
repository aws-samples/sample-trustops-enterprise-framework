"""
Unit tests for BedrockClient.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError
from datetime import datetime

from src.aws_clients.bedrock_client import BedrockClient


@pytest.fixture
def bedrock_client():
    """Create a BedrockClient instance with mocked boto3 clients."""
    with patch('src.aws_clients.bedrock_client.boto3.Session') as mock_session:
        mock_bedrock_runtime = Mock()
        mock_bedrock = Mock()
        
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = lambda service: (
            mock_bedrock_runtime if service == 'bedrock-runtime' else mock_bedrock
        )
        mock_session.return_value = mock_session_instance
        
        client = BedrockClient()
        client.bedrock_runtime = mock_bedrock_runtime
        client.bedrock = mock_bedrock
        
        return client


class TestGetModelConfig:
    """Tests for get_model_config method."""
    
    def test_get_model_config_valid_model(self, bedrock_client):
        """Test retrieving configuration for a valid model."""
        model_id = 'anthropic.claude-v2'
        
        bedrock_client.bedrock.get_foundation_model.return_value = {
            'modelDetails': {
                'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2',
                'modelName': 'Claude v2',
                'providerName': 'Anthropic',
                'inputModalities': ['TEXT'],
                'outputModalities': ['TEXT'],
                'responseStreamingSupported': True,
                'customizationsSupported': ['FINE_TUNING'],
                'inferenceTypesSupported': ['ON_DEMAND']
            }
        }
        
        config = bedrock_client.get_model_config(model_id)
        
        assert config['model_id'] == model_id
        assert config['model_name'] == 'Claude v2'
        assert config['provider_name'] == 'Anthropic'
        assert 'TEXT' in config['input_modalities']
        assert config['response_streaming_supported'] is True
        assert 'pricing' in config
        assert config['pricing']['input'] == 0.008
        
        bedrock_client.bedrock.get_foundation_model.assert_called_once_with(
            modelIdentifier=model_id
        )
    
    def test_get_model_config_invalid_model(self, bedrock_client):
        """Test retrieving configuration for an invalid model ID."""
        model_id = 'invalid.model'
        
        error_response = {
            'Error': {
                'Code': 'ResourceNotFoundException',
                'Message': 'Model not found'
            }
        }
        bedrock_client.bedrock.get_foundation_model.side_effect = ClientError(
            error_response, 'GetFoundationModel'
        )
        
        with pytest.raises(ValueError, match="Invalid model ID"):
            bedrock_client.get_model_config(model_id)
    
    def test_get_model_config_api_error(self, bedrock_client):
        """Test handling of AWS API errors."""
        model_id = 'anthropic.claude-v2'
        
        error_response = {
            'Error': {
                'Code': 'InternalServerError',
                'Message': 'Internal error'
            }
        }
        bedrock_client.bedrock.get_foundation_model.side_effect = ClientError(
            error_response, 'GetFoundationModel'
        )
        
        with pytest.raises(ClientError):
            bedrock_client.get_model_config(model_id)


class TestInvokeModel:
    """Tests for invoke_model method."""
    
    def test_invoke_claude_model_success(self, bedrock_client):
        """Test successful inference with Claude model."""
        model_id = 'anthropic.claude-v2'
        prompt = 'What is the capital of France?'
        
        response_body = {
            'completion': 'The capital of France is Paris.',
            'stop_reason': 'end_turn'
        }
        
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps(response_body).encode()
        
        bedrock_client.bedrock_runtime.invoke_model.return_value = mock_response
        
        result = bedrock_client.invoke_model(model_id, prompt, max_tokens=100)
        
        assert result['response_text'] == 'The capital of France is Paris.'
        assert result['model_id'] == model_id
        assert result['input_tokens'] > 0
        assert result['output_tokens'] > 0
        assert result['latency_ms'] > 0
        assert result['cost'] > 0
        
        # Verify the API was called with correct parameters
        call_args = bedrock_client.bedrock_runtime.invoke_model.call_args
        assert call_args[1]['modelId'] == model_id
        body = json.loads(call_args[1]['body'])
        assert 'Human:' in body['prompt']
        assert body['max_tokens_to_sample'] == 100
    
    def test_invoke_titan_model_success(self, bedrock_client):
        """Test successful inference with Titan model."""
        model_id = 'amazon.titan-text-express-v1'
        prompt = 'Explain quantum computing.'
        
        response_body = {
            'results': [{
                'outputText': 'Quantum computing uses quantum mechanics...',
                'tokenCount': 50
            }],
            'inputTextTokenCount': 10
        }
        
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps(response_body).encode()
        
        bedrock_client.bedrock_runtime.invoke_model.return_value = mock_response
        
        result = bedrock_client.invoke_model(model_id, prompt)
        
        assert 'Quantum computing' in result['response_text']
        assert result['input_tokens'] == 10
        assert result['output_tokens'] == 50
        assert result['cost'] > 0
    
    def test_invoke_llama_model_success(self, bedrock_client):
        """Test successful inference with Llama model."""
        model_id = 'meta.llama2-13b-chat-v1'
        prompt = 'Write a haiku about coding.'
        
        response_body = {
            'generation': 'Code flows like water\nBugs hide in shadows deep\nDebug brings the light',
            'prompt_token_count': 15,
            'generation_token_count': 25
        }
        
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps(response_body).encode()
        
        bedrock_client.bedrock_runtime.invoke_model.return_value = mock_response
        
        result = bedrock_client.invoke_model(model_id, prompt)
        
        assert 'Code flows' in result['response_text']
        assert result['input_tokens'] == 15
        assert result['output_tokens'] == 25
    
    def test_invoke_model_unsupported_provider(self, bedrock_client):
        """Test error handling for unsupported model provider."""
        model_id = 'unsupported.model-v1'
        prompt = 'Test prompt'
        
        with pytest.raises(ValueError, match="Unsupported model provider"):
            bedrock_client.invoke_model(model_id, prompt)
    
    def test_invoke_model_validation_error(self, bedrock_client):
        """Test handling of validation errors."""
        model_id = 'anthropic.claude-v2'
        prompt = 'Test'
        
        error_response = {
            'Error': {
                'Code': 'ValidationException',
                'Message': 'Invalid parameters'
            }
        }
        bedrock_client.bedrock_runtime.invoke_model.side_effect = ClientError(
            error_response, 'InvokeModel'
        )
        
        with pytest.raises(ValueError, match="Invalid request parameters"):
            bedrock_client.invoke_model(model_id, prompt)
    
    def test_invoke_model_throttling_error(self, bedrock_client):
        """Test handling of rate limit errors."""
        model_id = 'anthropic.claude-v2'
        prompt = 'Test'
        
        error_response = {
            'Error': {
                'Code': 'ThrottlingException',
                'Message': 'Rate exceeded'
            }
        }
        bedrock_client.bedrock_runtime.invoke_model.side_effect = ClientError(
            error_response, 'InvokeModel'
        )
        
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            bedrock_client.invoke_model(model_id, prompt)
    
    def test_invoke_model_not_found(self, bedrock_client):
        """Test handling of model not found error."""
        model_id = 'anthropic.claude-nonexistent'
        prompt = 'Test'
        
        error_response = {
            'Error': {
                'Code': 'ResourceNotFoundException',
                'Message': 'Model not found'
            }
        }
        # 'anthropic.claude-nonexistent' routes through the Converse API,
        # so the error surfaces from converse() rather than invoke_model().
        bedrock_client.bedrock_runtime.converse.side_effect = ClientError(
            error_response, 'Converse'
        )

        with pytest.raises(ValueError, match="Model not found"):
            bedrock_client.invoke_model(model_id, prompt)


class TestCreateFineTuningJob:
    """Tests for create_fine_tuning_job method."""
    
    def test_create_fine_tuning_job_success(self, bedrock_client):
        """Test successful fine-tuning job creation."""
        base_model_id = 'anthropic.claude-v2'
        training_data_s3_uri = 's3://bucket/training-data.jsonl'
        job_name = 'test-fine-tuning-job'
        output_data_s3_uri = 's3://bucket/output/'
        role_arn = 'arn:aws:iam::123456789012:role/BedrockRole'
        
        bedrock_client.bedrock.create_model_customization_job.return_value = {
            'jobArn': 'arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job'
        }
        
        result = bedrock_client.create_fine_tuning_job(
            base_model_id=base_model_id,
            training_data_s3_uri=training_data_s3_uri,
            job_name=job_name,
            output_data_s3_uri=output_data_s3_uri,
            role_arn=role_arn
        )
        
        assert 'job_id' in result
        assert 'job_arn' in result
        assert result['status'] == 'InProgress'
        assert 'created_at' in result
        
        # Verify API call
        call_args = bedrock_client.bedrock.create_model_customization_job.call_args
        assert call_args[1]['jobName'] == job_name
        assert call_args[1]['baseModelIdentifier'] == base_model_id
        assert call_args[1]['roleArn'] == role_arn
    
    def test_create_fine_tuning_job_with_hyperparameters(self, bedrock_client):
        """Test fine-tuning job creation with custom hyperparameters."""
        hyperparameters = {
            'epochCount': '5',
            'batchSize': '2',
            'learningRate': '0.00002'
        }
        
        bedrock_client.bedrock.create_model_customization_job.return_value = {
            'jobArn': 'arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job'
        }
        
        result = bedrock_client.create_fine_tuning_job(
            base_model_id='anthropic.claude-v2',
            training_data_s3_uri='s3://bucket/data.jsonl',
            job_name='test-job',
            output_data_s3_uri='s3://bucket/output/',
            role_arn='arn:aws:iam::123456789012:role/Role',
            hyperparameters=hyperparameters
        )
        
        assert result['status'] == 'InProgress'
        
        call_args = bedrock_client.bedrock.create_model_customization_job.call_args
        assert call_args[1]['hyperParameters'] == hyperparameters
    
    def test_create_fine_tuning_job_validation_error(self, bedrock_client):
        """Test handling of validation errors."""
        error_response = {
            'Error': {
                'Code': 'ValidationException',
                'Message': 'Invalid training data format'
            }
        }
        bedrock_client.bedrock.create_model_customization_job.side_effect = ClientError(
            error_response, 'CreateModelCustomizationJob'
        )
        
        with pytest.raises(ValueError, match="Invalid fine-tuning parameters"):
            bedrock_client.create_fine_tuning_job(
                base_model_id='anthropic.claude-v2',
                training_data_s3_uri='s3://bucket/data.jsonl',
                job_name='test-job',
                output_data_s3_uri='s3://bucket/output/',
                role_arn='arn:aws:iam::123456789012:role/Role'
            )
    
    def test_create_fine_tuning_job_access_denied(self, bedrock_client):
        """Test handling of access denied errors."""
        error_response = {
            'Error': {
                'Code': 'AccessDeniedException',
                'Message': 'Insufficient permissions'
            }
        }
        bedrock_client.bedrock.create_model_customization_job.side_effect = ClientError(
            error_response, 'CreateModelCustomizationJob'
        )
        
        with pytest.raises(RuntimeError, match="Access denied"):
            bedrock_client.create_fine_tuning_job(
                base_model_id='anthropic.claude-v2',
                training_data_s3_uri='s3://bucket/data.jsonl',
                job_name='test-job',
                output_data_s3_uri='s3://bucket/output/',
                role_arn='arn:aws:iam::123456789012:role/Role'
            )


class TestGetFineTuningJobStatus:
    """Tests for get_fine_tuning_job_status method."""
    
    def test_get_job_status_in_progress(self, bedrock_client):
        """Test getting status of in-progress job."""
        job_id = 'arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job'
        
        bedrock_client.bedrock.get_model_customization_job.return_value = {
            'status': 'InProgress',
            'trainingMetrics': {
                'trainingLoss': 0.5
            }
        }
        
        result = bedrock_client.get_fine_tuning_job_status(job_id)
        
        assert result['job_id'] == job_id
        assert result['status'] == 'InProgress'
        assert result['training_metrics']['trainingLoss'] == 0.5
        assert result['custom_model_arn'] is None
    
    def test_get_job_status_completed(self, bedrock_client):
        """Test getting status of completed job."""
        job_id = 'arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job'
        model_arn = 'arn:aws:bedrock:us-east-1:123456789012:custom-model/test-model'
        
        bedrock_client.bedrock.get_model_customization_job.return_value = {
            'status': 'Completed',
            'outputModelArn': model_arn,
            'trainingMetrics': {
                'trainingLoss': 0.2
            }
        }
        
        result = bedrock_client.get_fine_tuning_job_status(job_id)
        
        assert result['status'] == 'Completed'
        assert result['custom_model_arn'] == model_arn
        assert result['failure_message'] is None
    
    def test_get_job_status_failed(self, bedrock_client):
        """Test getting status of failed job."""
        job_id = 'arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job'
        
        bedrock_client.bedrock.get_model_customization_job.return_value = {
            'status': 'Failed',
            'failureMessage': 'Training data validation failed'
        }
        
        result = bedrock_client.get_fine_tuning_job_status(job_id)
        
        assert result['status'] == 'Failed'
        assert 'validation failed' in result['failure_message']
    
    def test_get_job_status_not_found(self, bedrock_client):
        """Test handling of job not found error."""
        job_id = 'nonexistent-job'
        
        error_response = {
            'Error': {
                'Code': 'ResourceNotFoundException',
                'Message': 'Job not found'
            }
        }
        bedrock_client.bedrock.get_model_customization_job.side_effect = ClientError(
            error_response, 'GetModelCustomizationJob'
        )
        
        with pytest.raises(ValueError, match="Fine-tuning job not found"):
            bedrock_client.get_fine_tuning_job_status(job_id)


class TestCalculateCost:
    """Tests for calculate_cost method."""
    
    def test_calculate_cost_claude(self, bedrock_client):
        """Test cost calculation for Claude model."""
        model_id = 'anthropic.claude-v2'
        input_tokens = 1000
        output_tokens = 500
        
        cost = bedrock_client.calculate_cost(model_id, input_tokens, output_tokens)
        
        # Expected: (1000/1000 * 0.008) + (500/1000 * 0.024) = 0.008 + 0.012 = 0.020
        assert abs(cost - 0.020) < 0.0001
    
    def test_calculate_cost_titan(self, bedrock_client):
        """Test cost calculation for Titan model."""
        model_id = 'amazon.titan-text-express-v1'
        input_tokens = 2000
        output_tokens = 1000
        
        cost = bedrock_client.calculate_cost(model_id, input_tokens, output_tokens)
        
        # Expected: (2000/1000 * 0.0002) + (1000/1000 * 0.0006) = 0.0004 + 0.0006 = 0.001
        assert abs(cost - 0.001) < 0.0001
    
    def test_calculate_cost_unknown_model(self, bedrock_client):
        """Test cost calculation for unknown model (should return 0)."""
        model_id = 'unknown.model'
        input_tokens = 1000
        output_tokens = 500
        
        cost = bedrock_client.calculate_cost(model_id, input_tokens, output_tokens)
        
        assert cost == 0.0
    
    def test_calculate_cost_zero_tokens(self, bedrock_client):
        """Test cost calculation with zero tokens."""
        model_id = 'anthropic.claude-v2'
        
        cost = bedrock_client.calculate_cost(model_id, 0, 0)
        
        assert cost == 0.0


class TestTokenCounting:
    """Tests for _count_tokens method."""
    
    def test_count_tokens_simple_text(self, bedrock_client):
        """Test token counting for simple text."""
        text = "Hello, world!"
        
        token_count = bedrock_client._count_tokens(text)
        
        # Approximately 13 characters / 4 = 3.25, rounded to 3
        assert token_count >= 1
        assert token_count <= 10
    
    def test_count_tokens_empty_string(self, bedrock_client):
        """Test token counting for empty string."""
        text = ""
        
        token_count = bedrock_client._count_tokens(text)
        
        # Should return at least 1
        assert token_count == 1
    
    def test_count_tokens_long_text(self, bedrock_client):
        """Test token counting for longer text."""
        text = "This is a longer piece of text that should result in more tokens."
        
        token_count = bedrock_client._count_tokens(text)
        
        # Should be proportional to length
        assert token_count > 10
