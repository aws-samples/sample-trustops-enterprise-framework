"""Unit tests for SageMakerAdapter.

Tests the SageMaker adapter implementation including endpoint validation,
invocation, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from botocore.exceptions import ClientError
import json

from src.adapters.sagemaker_adapter import SageMakerAdapter
from src.adapters.base_adapter import InferenceRequest, InferenceResponse
from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
)

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_boto3_session():
    """Create a mock boto3 session."""
    with patch('src.adapters.sagemaker_adapter.boto3.Session') as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock SageMaker clients
        mock_sagemaker_runtime = Mock()
        mock_sagemaker = Mock()
        
        def client_factory(service_name):
            if service_name == 'sagemaker-runtime':
                return mock_sagemaker_runtime
            elif service_name == 'sagemaker':
                return mock_sagemaker
            return Mock()
        
        mock_session.client.side_effect = client_factory
        
        yield {
            'session': mock_session,
            'sagemaker_runtime': mock_sagemaker_runtime,
            'sagemaker': mock_sagemaker
        }


@pytest.fixture
def sagemaker_adapter(mock_boto3_session):
    """Create a SageMakerAdapter instance with mocked clients."""
    return SageMakerAdapter(
        endpoint_name='test-endpoint',
        region='us-east-1'
    )


class TestSageMakerAdapterInit:
    """Test SageMakerAdapter initialization."""

    def test_init_with_region(self, mock_boto3_session):
        """Test initialization with explicit region."""
        adapter = SageMakerAdapter(endpoint_name='test-endpoint', region='us-west-2')
        assert adapter.endpoint_name == 'test-endpoint'
        assert adapter.region == 'us-west-2'

    def test_init_without_region(self, mock_boto3_session):
        """Test initialization without explicit region uses default."""
        with patch('src.adapters.sagemaker_adapter.config') as mock_config:
            mock_config.region = 'us-east-1'
            mock_config.get_boto3_session_kwargs.return_value = {}
            
            adapter = SageMakerAdapter(endpoint_name='test-endpoint')
            assert adapter.endpoint_name == 'test-endpoint'
            assert adapter.region == 'us-east-1'


class TestSageMakerAdapterInvoke:
    """Test model invocation."""

    @pytest.mark.asyncio
    async def test_invoke_success_dict_response(self, sagemaker_adapter, mock_boto3_session):
        """Test successful invocation with dictionary response."""
        # Mock the invoke_endpoint response
        mock_response = {
            'Body': MagicMock()
        }
        response_body = {
            'generated_text': 'This is a test response from SageMaker'
        }
        mock_response['Body'].read.return_value.decode.return_value = json.dumps(response_body)
        
        mock_boto3_session['sagemaker_runtime'].invoke_endpoint.return_value = mock_response

        request = InferenceRequest(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )

        response = await sagemaker_adapter.invoke(request)

        assert isinstance(response, InferenceResponse)
        assert response.text == 'This is a test response from SageMaker'
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.latency_ms > 0
        assert response.model_id == 'test-endpoint'
        assert response.finish_reason == 'stop'

        # Verify the client was called correctly
        mock_boto3_session['sagemaker_runtime'].invoke_endpoint.assert_called_once()
        call_kwargs = mock_boto3_session['sagemaker_runtime'].invoke_endpoint.call_args[1]
        assert call_kwargs['EndpointName'] == 'test-endpoint'
        assert call_kwargs['ContentType'] == 'application/json'
        assert call_kwargs['Accept'] == 'application/json'
        
        # Verify request body
        body = json.loads(call_kwargs['Body'])
        assert body['inputs'] == 'Test prompt'
        assert body['parameters']['max_new_tokens'] == 100
        assert body['parameters']['temperature'] == 0.7
        assert body['parameters']['top_p'] == 0.9

    @pytest.mark.asyncio
    async def test_invoke_with_outputs_field(self, sagemaker_adapter, mock_boto3_session):
        """Test invocation with 'outputs' field in response."""
        mock_response = {
            'Body': MagicMock()
        }
        response_body = {
            'outputs': 'Response using outputs field'
        }
        mock_response['Body'].read.return_value.decode.return_value = json.dumps(response_body)
        
        mock_boto3_session['sagemaker_runtime'].invoke_endpoint.return_value = mock_response

        request = InferenceRequest(prompt="Test")
        response = await sagemaker_adapter.invoke(request)

        assert response.text == 'Response using outputs field'

    @pytest.mark.asyncio
    async def test_invoke_with_list_response(self, sagemaker_adapter, mock_boto3_session):
        """Test invocation with list response format."""
        mock_response = {
            'Body': MagicMock()
        }
        response_body = {
            'generated_text': [
                {'generated_text': 'First response in list'}
            ]
        }
        mock_response['Body'].read.return_value.decode.return_value = json.dumps(response_body)
        
        mock_boto3_session['sagemaker_runtime'].invoke_endpoint.return_value = mock_response

        request = InferenceRequest(prompt="Test")
        response = await sagemaker_adapter.invoke(request)

        assert response.text == 'First response in list'

    @pytest.mark.asyncio
    async def test_invoke_with_stop_sequences(self, sagemaker_adapter, mock_boto3_session):
        """Test invocation with stop sequences."""
        mock_response = {
            'Body': MagicMock()
        }
        response_body = {'generated_text': 'Response'}
        mock_response['Body'].read.return_value.decode.return_value = json.dumps(response_body)
        
        mock_boto3_session['sagemaker_runtime'].invoke_endpoint.return_value = mock_response

        request = InferenceRequest(
            prompt="Test",
            stop_sequences=["STOP", "END"]
        )

        await sagemaker_adapter.invoke(request)

        call_kwargs = mock_boto3_session['sagemaker_runtime'].invoke_endpoint.call_args[1]
        body = json.loads(call_kwargs['Body'])
        assert 'stop' in body['parameters']
        assert body['parameters']['stop'] == ["STOP", "END"]

    @pytest.mark.asyncio
    async def test_invoke_validation_error(self, sagemaker_adapter, mock_boto3_session):
        """Test invocation with validation error."""
        error_response = {'Error': {'Code': 'ValidationError', 'Message': 'Invalid input'}}
        mock_boto3_session['sagemaker_runtime'].invoke_endpoint.side_effect = ClientError(
            error_response, 'InvokeEndpoint'
        )

        request = InferenceRequest(prompt="Test")

        with pytest.raises(ValueError, match="Invalid request parameters"):
            await sagemaker_adapter.invoke(request)

    @pytest.mark.asyncio
    async def test_invoke_model_error(self, sagemaker_adapter, mock_boto3_session):
        """Test invocation with model error."""
        error_response = {'Error': {'Code': 'ModelError', 'Message': 'Model failed'}}
        mock_boto3_session['sagemaker_runtime'].invoke_endpoint.side_effect = ClientError(
            error_response, 'InvokeEndpoint'
        )

        request = InferenceRequest(prompt="Test")

        with pytest.raises(RuntimeError, match="Model invocation failed"):
            await sagemaker_adapter.invoke(request)

    @pytest.mark.asyncio
    async def test_invoke_service_unavailable(self, sagemaker_adapter, mock_boto3_session):
        """Test invocation when service is unavailable."""
        error_response = {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service down'}}
        mock_boto3_session['sagemaker_runtime'].invoke_endpoint.side_effect = ClientError(
            error_response, 'InvokeEndpoint'
        )

        request = InferenceRequest(prompt="Test")

        with pytest.raises(RuntimeError, match="SageMaker endpoint unavailable"):
            await sagemaker_adapter.invoke(request)

    @pytest.mark.asyncio
    async def test_invoke_json_decode_error(self, sagemaker_adapter, mock_boto3_session):
        """Test invocation with invalid JSON response."""
        mock_response = {
            'Body': MagicMock()
        }
        mock_response['Body'].read.return_value.decode.return_value = 'invalid json'
        
        mock_boto3_session['sagemaker_runtime'].invoke_endpoint.return_value = mock_response

        request = InferenceRequest(prompt="Test")

        with pytest.raises(RuntimeError, match="Failed to parse SageMaker response"):
            await sagemaker_adapter.invoke(request)


class TestSageMakerAdapterInvokeStream:
    """Test streaming invocation."""

    @pytest.mark.asyncio
    async def test_invoke_stream(self, sagemaker_adapter, mock_boto3_session):
        """Test streaming invocation (currently falls back to non-streaming)."""
        mock_response = {
            'Body': MagicMock()
        }
        response_body = {'generated_text': 'Streamed response'}
        mock_response['Body'].read.return_value.decode.return_value = json.dumps(response_body)
        
        mock_boto3_session['sagemaker_runtime'].invoke_endpoint.return_value = mock_response

        request = InferenceRequest(prompt="Test streaming")

        chunks = []
        async for chunk in sagemaker_adapter.invoke_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == 'Streamed response'


class TestSageMakerAdapterListModels:
    """Test endpoint discovery."""

    @pytest.mark.asyncio
    async def test_list_models_success(self, sagemaker_adapter, mock_boto3_session):
        """Test successful endpoint listing."""
        # Mock list_endpoints response
        mock_boto3_session['sagemaker'].list_endpoints.return_value = {
            'Endpoints': [
                {
                    'EndpointName': 'endpoint-1',
                    'EndpointArn': 'arn:aws:sagemaker:us-east-1:123456789012:endpoint/endpoint-1',
                    'EndpointStatus': 'InService',
                    'CreationTime': datetime(2024, 1, 1, 12, 0, 0),
                    'LastModifiedTime': datetime(2024, 1, 2, 12, 0, 0)
                },
                {
                    'EndpointName': 'endpoint-2',
                    'EndpointArn': 'arn:aws:sagemaker:us-east-1:123456789012:endpoint/endpoint-2',
                    'EndpointStatus': 'InService',
                    'CreationTime': datetime(2024, 1, 3, 12, 0, 0),
                    'LastModifiedTime': datetime(2024, 1, 4, 12, 0, 0)
                }
            ]
        }

        # Mock describe_endpoint responses
        def mock_describe_endpoint(EndpointName):
            return {
                'EndpointName': EndpointName,
                'EndpointArn': f'arn:aws:sagemaker:us-east-1:123456789012:endpoint/{EndpointName}',
                'EndpointStatus': 'InService',
                'EndpointConfigName': f'{EndpointName}-config',
                'CreationTime': datetime(2024, 1, 1, 12, 0, 0),
                'LastModifiedTime': datetime(2024, 1, 2, 12, 0, 0)
            }

        mock_boto3_session['sagemaker'].describe_endpoint.side_effect = mock_describe_endpoint

        # Mock describe_endpoint_config responses
        def mock_describe_config(EndpointConfigName):
            return {
                'ProductionVariants': [
                    {'InstanceType': 'ml.m5.xlarge'}
                ]
            }

        mock_boto3_session['sagemaker'].describe_endpoint_config.side_effect = mock_describe_config

        models = await sagemaker_adapter.list_models()

        assert len(models) == 2
        
        # Check first endpoint
        endpoint1 = models[0]
        assert isinstance(endpoint1, ModelMetadata)
        assert endpoint1.id == 'endpoint-1'
        assert endpoint1.provider == ModelProvider.SAGEMAKER
        assert endpoint1.name == 'endpoint-1'
        assert ModelCapability.TEXT_GENERATION in endpoint1.capabilities
        assert endpoint1.status == ModelStatus.ACTIVE
        assert endpoint1.fine_tuning_support is False
        assert endpoint1.metadata['instance_type'] == 'ml.m5.xlarge'

        # Check second endpoint
        endpoint2 = models[1]
        assert endpoint2.id == 'endpoint-2'
        assert endpoint2.name == 'endpoint-2'

    @pytest.mark.asyncio
    async def test_list_models_empty(self, sagemaker_adapter, mock_boto3_session):
        """Test listing endpoints when none are available."""
        mock_boto3_session['sagemaker'].list_endpoints.return_value = {
            'Endpoints': []
        }

        models = await sagemaker_adapter.list_models()

        assert len(models) == 0

    @pytest.mark.asyncio
    async def test_list_models_client_error(self, sagemaker_adapter, mock_boto3_session):
        """Test listing endpoints with client error."""
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}
        mock_boto3_session['sagemaker'].list_endpoints.side_effect = ClientError(
            error_response, 'ListEndpoints'
        )

        with pytest.raises(RuntimeError, match="Failed to list SageMaker endpoints"):
            await sagemaker_adapter.list_models()

    @pytest.mark.asyncio
    async def test_list_models_skips_failed_endpoints(self, sagemaker_adapter, mock_boto3_session):
        """Test that list_models continues when individual endpoint info fails."""
        mock_boto3_session['sagemaker'].list_endpoints.return_value = {
            'Endpoints': [
                {'EndpointName': 'endpoint-1', 'EndpointStatus': 'InService'},
                {'EndpointName': 'endpoint-2', 'EndpointStatus': 'InService'},
            ]
        }

        # First endpoint fails, second succeeds
        def mock_describe_endpoint(EndpointName):
            if EndpointName == 'endpoint-1':
                raise ValueError("Endpoint error")
            return {
                'EndpointName': EndpointName,
                'EndpointArn': f'arn:test:{EndpointName}',
                'EndpointStatus': 'InService',
                'EndpointConfigName': f'{EndpointName}-config',
                'CreationTime': datetime(2024, 1, 1),
                'LastModifiedTime': datetime(2024, 1, 2)
            }

        mock_boto3_session['sagemaker'].describe_endpoint.side_effect = mock_describe_endpoint
        mock_boto3_session['sagemaker'].describe_endpoint_config.return_value = {
            'ProductionVariants': [{'InstanceType': 'ml.m5.large'}]
        }

        models = await sagemaker_adapter.list_models()

        # Should only have the successful endpoint
        assert len(models) == 1
        assert models[0].id == 'endpoint-2'


class TestSageMakerAdapterGetModelInfo:
    """Test getting individual endpoint info."""

    @pytest.mark.asyncio
    async def test_get_model_info_success(self, sagemaker_adapter, mock_boto3_session):
        """Test successful endpoint info retrieval."""
        mock_boto3_session['sagemaker'].describe_endpoint.return_value = {
            'EndpointName': 'test-endpoint',
            'EndpointArn': 'arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint',
            'EndpointStatus': 'InService',
            'EndpointConfigName': 'test-config',
            'CreationTime': datetime(2024, 1, 1, 12, 0, 0),
            'LastModifiedTime': datetime(2024, 1, 2, 12, 0, 0)
        }

        mock_boto3_session['sagemaker'].describe_endpoint_config.return_value = {
            'ProductionVariants': [
                {'InstanceType': 'ml.g4dn.xlarge'}
            ]
        }

        metadata = await sagemaker_adapter.get_model_info('test-endpoint')

        assert isinstance(metadata, ModelMetadata)
        assert metadata.id == 'test-endpoint'
        assert metadata.provider == ModelProvider.SAGEMAKER
        assert metadata.name == 'test-endpoint'
        assert metadata.status == ModelStatus.ACTIVE
        assert metadata.fine_tuning_support is False
        assert metadata.metadata['instance_type'] == 'ml.g4dn.xlarge'

    @pytest.mark.asyncio
    async def test_get_model_info_not_found(self, sagemaker_adapter, mock_boto3_session):
        """Test getting info for non-existent endpoint."""
        error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Not found'}}
        mock_boto3_session['sagemaker'].describe_endpoint.side_effect = ClientError(
            error_response, 'DescribeEndpoint'
        )

        with pytest.raises(ValueError, match="Endpoint not found"):
            await sagemaker_adapter.get_model_info('invalid-endpoint')

    @pytest.mark.asyncio
    async def test_get_model_info_without_config(self, sagemaker_adapter, mock_boto3_session):
        """Test getting endpoint info when config retrieval fails."""
        mock_boto3_session['sagemaker'].describe_endpoint.return_value = {
            'EndpointName': 'test-endpoint',
            'EndpointArn': 'arn:test',
            'EndpointStatus': 'InService',
            'EndpointConfigName': 'test-config',
            'CreationTime': datetime(2024, 1, 1),
            'LastModifiedTime': datetime(2024, 1, 2)
        }

        # Config retrieval fails
        mock_boto3_session['sagemaker'].describe_endpoint_config.side_effect = Exception("Config error")

        metadata = await sagemaker_adapter.get_model_info('test-endpoint')

        # Should still succeed but without instance type
        assert metadata.id == 'test-endpoint'
        assert metadata.metadata['instance_type'] is None


class TestSageMakerAdapterValidateConnection:
    """Test connection validation."""

    @pytest.mark.asyncio
    async def test_validate_connection_success(self, sagemaker_adapter, mock_boto3_session):
        """Test successful connection validation."""
        mock_boto3_session['sagemaker'].describe_endpoint.return_value = {
            'EndpointStatus': 'InService'
        }

        result = await sagemaker_adapter.validate_connection()

        assert result is True
        mock_boto3_session['sagemaker'].describe_endpoint.assert_called_once_with(
            EndpointName='test-endpoint'
        )

    @pytest.mark.asyncio
    async def test_validate_connection_not_in_service(self, sagemaker_adapter, mock_boto3_session):
        """Test connection validation when endpoint is not in service."""
        mock_boto3_session['sagemaker'].describe_endpoint.return_value = {
            'EndpointStatus': 'Creating'
        }

        result = await sagemaker_adapter.validate_connection()

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_connection_failure(self, sagemaker_adapter, mock_boto3_session):
        """Test connection validation failure."""
        mock_boto3_session['sagemaker'].describe_endpoint.side_effect = Exception("Connection error")

        result = await sagemaker_adapter.validate_connection()

        assert result is False


class TestSageMakerAdapterCapabilities:
    """Test capability checking methods."""

    def test_supports_streaming(self, sagemaker_adapter):
        """Test streaming support check."""
        # Currently returns False as streaming is not fully implemented
        assert sagemaker_adapter.supports_streaming() is False

    def test_supports_fine_tuning(self, sagemaker_adapter):
        """Test fine-tuning support check."""
        # SageMaker endpoints don't support fine-tuning (they're already deployed)
        assert sagemaker_adapter.supports_fine_tuning('test-endpoint') is False


class TestSageMakerAdapterStatusMapping:
    """Test endpoint status mapping."""

    def test_map_endpoint_status_in_service(self, sagemaker_adapter):
        """Test mapping InService status."""
        assert sagemaker_adapter._map_endpoint_status('InService') == ModelStatus.ACTIVE

    def test_map_endpoint_status_creating(self, sagemaker_adapter):
        """Test mapping Creating status."""
        assert sagemaker_adapter._map_endpoint_status('Creating') == ModelStatus.PROVISIONING

    def test_map_endpoint_status_updating(self, sagemaker_adapter):
        """Test mapping Updating status."""
        assert sagemaker_adapter._map_endpoint_status('Updating') == ModelStatus.PROVISIONING

    def test_map_endpoint_status_failed(self, sagemaker_adapter):
        """Test mapping Failed status."""
        assert sagemaker_adapter._map_endpoint_status('Failed') == ModelStatus.FAILED

    def test_map_endpoint_status_out_of_service(self, sagemaker_adapter):
        """Test mapping OutOfService status."""
        assert sagemaker_adapter._map_endpoint_status('OutOfService') == ModelStatus.INACTIVE

    def test_map_endpoint_status_unknown(self, sagemaker_adapter):
        """Test mapping unknown status."""
        assert sagemaker_adapter._map_endpoint_status('UnknownStatus') == ModelStatus.INACTIVE


class TestSageMakerAdapterTokenEstimation:
    """Test token estimation."""

    def test_estimate_tokens(self, sagemaker_adapter):
        """Test token estimation."""
        text = "This is a test string with some words"
        tokens = sagemaker_adapter._estimate_tokens(text)
        
        # Should be roughly len(text) / 4
        expected = max(1, len(text) // 4)
        assert tokens == expected

    def test_estimate_tokens_empty(self, sagemaker_adapter):
        """Test token estimation with empty string."""
        tokens = sagemaker_adapter._estimate_tokens("")
        assert tokens == 1  # Minimum of 1 token
