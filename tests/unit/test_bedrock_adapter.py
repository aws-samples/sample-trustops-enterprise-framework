"""Unit tests for BedrockAdapter.

Tests the Bedrock adapter implementation including model discovery,
invocation, and error handling.
"""

import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError

from src.adapters.bedrock_adapter import BedrockAdapter
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
def mock_bedrock_client():
    """Create a mock BedrockClient."""
    mock_path = 'src.adapters.bedrock_adapter.BedrockClient'
    with patch(mock_path) as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.region = 'us-east-1'

        # Mock bedrock service client
        mock_client.bedrock = Mock()
        mock_client.bedrock_runtime = Mock()

        yield mock_client


@pytest.fixture
def bedrock_adapter(mock_bedrock_client):
    """Create a BedrockAdapter instance with mocked client."""
    return BedrockAdapter(
        model_id='anthropic.claude-v2',
        region='us-east-1'
    )


class TestBedrockAdapterInit:
    """Test BedrockAdapter initialization."""

    def test_init_with_region(self, mock_bedrock_client):
        """Test initialization with explicit region."""
        adapter = BedrockAdapter(model_id='test-model', region='us-west-2')
        assert adapter.model_id == 'test-model'
        assert adapter.region == 'us-west-2'

    def test_init_without_region(self, mock_bedrock_client):
        """Test initialization without explicit region uses default."""
        adapter = BedrockAdapter(model_id='test-model')
        assert adapter.model_id == 'test-model'
        assert adapter.region == 'us-east-1'  # From mock


class TestBedrockAdapterInvoke:
    """Test model invocation."""

    @pytest.mark.asyncio
    async def test_invoke_success(self, bedrock_adapter, mock_bedrock_client):
        """Test successful model invocation."""
        # Mock the invoke_model response
        mock_bedrock_client.invoke_model.return_value = {
            'response_text': 'This is a test response',
            'input_tokens': 10,
            'output_tokens': 5,
            'latency_ms': 123.45,
            'model_id': 'anthropic.claude-v2',
            'cost': 0.001
        }

        request = InferenceRequest(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )

        response = await bedrock_adapter.invoke(request)

        assert isinstance(response, InferenceResponse)
        assert response.text == 'This is a test response'
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.latency_ms == 123.45
        assert response.model_id == 'anthropic.claude-v2'
        assert response.finish_reason == 'stop'

        # Verify the client was called correctly
        mock_bedrock_client.invoke_model.assert_called_once()
        call_kwargs = mock_bedrock_client.invoke_model.call_args[1]
        assert call_kwargs['model_id'] == 'anthropic.claude-v2'
        assert call_kwargs['prompt'] == 'Test prompt'
        assert call_kwargs['max_tokens'] == 100
        assert call_kwargs['temperature'] == 0.7
        assert call_kwargs['top_p'] == 0.9

    @pytest.mark.asyncio
    async def test_invoke_with_stop_sequences(self, bedrock_adapter, mock_bedrock_client):
        """Test invocation with stop sequences."""
        mock_bedrock_client.invoke_model.return_value = {
            'response_text': 'Response',
            'input_tokens': 5,
            'output_tokens': 3,
            'latency_ms': 100.0,
            'model_id': 'anthropic.claude-v2',
            'cost': 0.0005
        }

        request = InferenceRequest(
            prompt="Test",
            stop_sequences=["STOP", "END"]
        )

        await bedrock_adapter.invoke(request)

        call_kwargs = mock_bedrock_client.invoke_model.call_args[1]
        assert 'stop' in call_kwargs
        assert call_kwargs['stop'] == ["STOP", "END"]

    @pytest.mark.asyncio
    async def test_invoke_value_error(self, bedrock_adapter, mock_bedrock_client):
        """Test invocation with invalid parameters."""
        mock_bedrock_client.invoke_model.side_effect = ValueError("Invalid parameters")

        request = InferenceRequest(prompt="Test")

        with pytest.raises(ValueError, match="Invalid request parameters"):
            await bedrock_adapter.invoke(request)

    @pytest.mark.asyncio
    async def test_invoke_runtime_error(self, bedrock_adapter, mock_bedrock_client):
        """Test invocation with runtime error."""
        mock_bedrock_client.invoke_model.side_effect = RuntimeError("API error")

        request = InferenceRequest(prompt="Test")

        with pytest.raises(RuntimeError, match="Model invocation failed"):
            await bedrock_adapter.invoke(request)


class TestBedrockAdapterInvokeStream:
    """Test streaming invocation."""

    @pytest.mark.asyncio
    async def test_invoke_stream(self, bedrock_adapter, mock_bedrock_client):
        """Test streaming invocation (currently falls back to non-streaming)."""
        mock_bedrock_client.invoke_model.return_value = {
            'response_text': 'Streamed response',
            'input_tokens': 5,
            'output_tokens': 2,
            'latency_ms': 50.0,
            'model_id': 'anthropic.claude-v2',
            'cost': 0.0003
        }

        request = InferenceRequest(prompt="Test streaming")

        chunks = []
        async for chunk in bedrock_adapter.invoke_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == 'Streamed response'


class TestBedrockAdapterListModels:
    """Test model discovery."""

    @pytest.mark.asyncio
    async def test_list_models_success(self, bedrock_adapter, mock_bedrock_client):
        """Test successful model listing."""
        # Mock list_foundation_models response
        mock_bedrock_client.bedrock.list_foundation_models.return_value = {
            'modelSummaries': [
                {
                    'modelId': 'anthropic.claude-v2',
                    'inferenceTypesSupported': ['CHAT', 'TEXT_GENERATION'],
                    'customizationsSupported': ['FINE_TUNING']
                },
                {
                    'modelId': 'amazon.titan-text-express-v1',
                    'inferenceTypesSupported': ['TEXT_GENERATION'],
                    'customizationsSupported': []
                }
            ]
        }

        # Mock get_model_config responses
        def mock_get_config(model_id):
            if model_id == 'anthropic.claude-v2':
                return {
                    'model_name': 'Claude v2',
                    'provider_name': 'Anthropic',
                    'model_arn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2',
                    'input_modalities': ['text'],
                    'output_modalities': ['text'],
                    'response_streaming_supported': True,
                    'customizations_supported': ['FINE_TUNING'],
                    'inference_types_supported': ['CHAT', 'TEXT_GENERATION'],
                    'pricing': {'input': 0.008, 'output': 0.024}
                }
            elif model_id == 'amazon.titan-text-express-v1':
                return {
                    'model_name': 'Titan Text Express',
                    'provider_name': 'Amazon',
                    'model_arn': 'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-express-v1',
                    'input_modalities': ['text'],
                    'output_modalities': ['text'],
                    'response_streaming_supported': False,
                    'customizations_supported': [],
                    'inference_types_supported': ['TEXT_GENERATION'],
                    'pricing': {'input': 0.0002, 'output': 0.0006}
                }

        mock_bedrock_client.get_model_config.side_effect = mock_get_config

        models = await bedrock_adapter.list_models()

        assert len(models) == 2
        
        # Check first model (Claude)
        claude = models[0]
        assert isinstance(claude, ModelMetadata)
        assert claude.id == 'anthropic.claude-v2'
        assert claude.provider == ModelProvider.BEDROCK
        assert claude.name == 'Claude v2'
        assert ModelCapability.TEXT_GENERATION in claude.capabilities
        assert ModelCapability.CHAT in claude.capabilities
        assert ModelCapability.FINE_TUNABLE in claude.capabilities
        assert claude.fine_tuning_support is True
        assert claude.status == ModelStatus.ACTIVE
        assert claude.pricing.input_price_per_1k_tokens == 0.008
        assert claude.pricing.output_price_per_1k_tokens == 0.024

        # Check second model (Titan)
        titan = models[1]
        assert titan.id == 'amazon.titan-text-express-v1'
        assert titan.name == 'Titan Text Express'
        assert ModelCapability.TEXT_GENERATION in titan.capabilities
        assert ModelCapability.FINE_TUNABLE not in titan.capabilities
        assert titan.fine_tuning_support is False

    @pytest.mark.asyncio
    async def test_list_models_empty(self, bedrock_adapter, mock_bedrock_client):
        """Test listing models when none are available."""
        mock_bedrock_client.bedrock.list_foundation_models.return_value = {
            'modelSummaries': []
        }

        models = await bedrock_adapter.list_models()

        assert len(models) == 0

    @pytest.mark.asyncio
    async def test_list_models_client_error(self, bedrock_adapter, mock_bedrock_client):
        """Test listing models with client error."""
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}
        mock_bedrock_client.bedrock.list_foundation_models.side_effect = ClientError(
            error_response, 'ListFoundationModels'
        )

        with pytest.raises(RuntimeError, match="Failed to list Bedrock models"):
            await bedrock_adapter.list_models()

    @pytest.mark.asyncio
    async def test_list_models_skips_failed_configs(self, bedrock_adapter, mock_bedrock_client):
        """Test that list_models continues when individual model config fails."""
        mock_bedrock_client.bedrock.list_foundation_models.return_value = {
            'modelSummaries': [
                {'modelId': 'model-1'},
                {'modelId': 'model-2'},
            ]
        }

        # First model fails, second succeeds
        def mock_get_config(model_id):
            if model_id == 'model-1':
                raise ValueError("Config error")
            return {
                'model_name': 'Model 2',
                'provider_name': 'Test',
                'model_arn': 'arn:test',
                'input_modalities': ['text'],
                'output_modalities': ['text'],
                'pricing': {'input': 0.001, 'output': 0.002}
            }

        mock_bedrock_client.get_model_config.side_effect = mock_get_config

        models = await bedrock_adapter.list_models()

        # Should only have the successful model
        assert len(models) == 1
        assert models[0].id == 'model-2'


class TestBedrockAdapterGetModelInfo:
    """Test getting individual model info."""

    @pytest.mark.asyncio
    async def test_get_model_info_success(self, bedrock_adapter, mock_bedrock_client):
        """Test successful model info retrieval."""
        mock_bedrock_client.get_model_config.return_value = {
            'model_name': 'Claude v2',
            'provider_name': 'Anthropic',
            'model_arn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2',
            'input_modalities': ['text'],
            'output_modalities': ['text'],
            'response_streaming_supported': True,
            'customizations_supported': ['FINE_TUNING'],
            'inference_types_supported': ['CHAT'],
            'pricing': {'input': 0.008, 'output': 0.024}
        }

        metadata = await bedrock_adapter.get_model_info('anthropic.claude-v2')

        assert isinstance(metadata, ModelMetadata)
        assert metadata.id == 'anthropic.claude-v2'
        assert metadata.provider == ModelProvider.BEDROCK
        assert metadata.name == 'Claude v2'
        assert metadata.fine_tuning_support is True
        assert metadata.pricing.input_price_per_1k_tokens == 0.008

    @pytest.mark.asyncio
    async def test_get_model_info_not_found(self, bedrock_adapter, mock_bedrock_client):
        """Test getting info for non-existent model."""
        mock_bedrock_client.get_model_config.side_effect = ValueError("Invalid model ID")

        with pytest.raises(ValueError, match="Model not found"):
            await bedrock_adapter.get_model_info('invalid-model')

    @pytest.mark.asyncio
    async def test_get_model_info_client_error(self, bedrock_adapter, mock_bedrock_client):
        """Test getting model info with client error."""
        error_response = {'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Not found'}}
        mock_bedrock_client.get_model_config.side_effect = ClientError(
            error_response, 'GetFoundationModel'
        )

        with pytest.raises(ValueError, match="Model not found"):
            await bedrock_adapter.get_model_info('test-model')


class TestBedrockAdapterValidateConnection:
    """Test connection validation."""

    @pytest.mark.asyncio
    async def test_validate_connection_success(self, bedrock_adapter, mock_bedrock_client):
        """Test successful connection validation."""
        mock_bedrock_client.bedrock.list_foundation_models.return_value = {
            'modelSummaries': []
        }

        result = await bedrock_adapter.validate_connection()

        assert result is True
        mock_bedrock_client.bedrock.list_foundation_models.assert_called_once_with(maxResults=1)

    @pytest.mark.asyncio
    async def test_validate_connection_failure(self, bedrock_adapter, mock_bedrock_client):
        """Test connection validation failure."""
        mock_bedrock_client.bedrock.list_foundation_models.side_effect = Exception("Connection error")

        result = await bedrock_adapter.validate_connection()

        assert result is False


class TestBedrockAdapterCapabilities:
    """Test capability checking methods."""

    def test_supports_streaming(self, bedrock_adapter):
        """Test streaming support check."""
        # Currently returns False as streaming is not fully implemented
        assert bedrock_adapter.supports_streaming() is False

    def test_supports_fine_tuning_true(self, bedrock_adapter, mock_bedrock_client):
        """Test fine-tuning support check for supported model."""
        mock_bedrock_client.get_model_config.return_value = {
            'customizations_supported': ['FINE_TUNING']
        }

        result = bedrock_adapter.supports_fine_tuning('anthropic.claude-v2')

        assert result is True

    def test_supports_fine_tuning_false(self, bedrock_adapter, mock_bedrock_client):
        """Test fine-tuning support check for unsupported model."""
        mock_bedrock_client.get_model_config.return_value = {
            'customizations_supported': []
        }

        result = bedrock_adapter.supports_fine_tuning('amazon.titan-text-express-v1')

        assert result is False

    def test_supports_fine_tuning_error(self, bedrock_adapter, mock_bedrock_client):
        """Test fine-tuning support check with error."""
        mock_bedrock_client.get_model_config.side_effect = Exception("Error")

        result = bedrock_adapter.supports_fine_tuning('invalid-model')

        assert result is False
