"""Unit tests for ExternalAPIAdapter.

Tests the External API adapter implementation including model discovery,
invocation, streaming, rate limiting, and error handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp
import json

from src.adapters.external_api_adapter import ExternalAPIAdapter
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
def external_api_adapter():
    """Create an ExternalAPIAdapter instance."""
    return ExternalAPIAdapter(
        model_id='gpt-4',
        base_url='https://api.openai.com/v1',
        api_key='test-api-key',
        organization_id='test-org',
        timeout_seconds=30.0,
        max_retries=3
    )


def create_mock_response(status=200, json_data=None, headers=None):
    """Helper to create a mock aiohttp response."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.headers = headers or {}
    if json_data is not None:
        mock_response.json = AsyncMock(return_value=json_data)
    mock_response.text = AsyncMock(return_value=json.dumps(json_data) if json_data else "")
    return mock_response


def create_mock_session(response):
    """Helper to create a mock aiohttp session with proper async context managers."""
    mock_session = MagicMock()
    
    # Create async context manager for the response
    mock_response_cm = MagicMock()
    mock_response_cm.__aenter__ = AsyncMock(return_value=response)
    mock_response_cm.__aexit__ = AsyncMock(return_value=None)
    
    # Set up post and get methods
    mock_session.post = MagicMock(return_value=mock_response_cm)
    mock_session.get = MagicMock(return_value=mock_response_cm)
    
    # Create async context manager for the session itself
    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)
    
    return mock_session_cm


class TestExternalAPIAdapterInit:
    """Test ExternalAPIAdapter initialization."""

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        adapter = ExternalAPIAdapter(
            model_id='gpt-4',
            base_url='https://api.openai.com/v1',
            api_key='test-key',
            organization_id='test-org',
            timeout_seconds=60.0,
            max_retries=5
        )
        
        assert adapter.model_id == 'gpt-4'
        assert adapter.base_url == 'https://api.openai.com/v1'
        assert adapter.api_key == 'test-key'
        assert adapter.organization_id == 'test-org'
        assert adapter.timeout_seconds == 60.0
        assert adapter.max_retries == 5

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from base URL."""
        adapter = ExternalAPIAdapter(
            model_id='gpt-4',
            base_url='https://api.openai.com/v1/',
            api_key='test-key'
        )
        
        assert adapter.base_url == 'https://api.openai.com/v1'


class TestExternalAPIAdapterGetHeaders:
    """Test header generation."""

    def test_get_headers_with_org(self, external_api_adapter):
        """Test headers include organization ID when provided."""
        headers = external_api_adapter._get_headers()
        
        assert headers['Authorization'] == 'Bearer test-api-key'
        assert headers['Content-Type'] == 'application/json'
        assert headers['OpenAI-Organization'] == 'test-org'

    def test_get_headers_without_org(self):
        """Test headers without organization ID."""
        adapter = ExternalAPIAdapter(
            model_id='gpt-4',
            base_url='https://api.openai.com/v1',
            api_key='test-key'
        )
        
        headers = adapter._get_headers()
        
        assert headers['Authorization'] == 'Bearer test-key'
        assert headers['Content-Type'] == 'application/json'
        assert 'OpenAI-Organization' not in headers


class TestExternalAPIAdapterInvoke:
    """Test model invocation."""

    async def test_invoke_success(self, external_api_adapter):
        """Test successful model invocation."""
        mock_response_data = {
            'choices': [
                {
                    'message': {'content': 'This is a test response'},
                    'finish_reason': 'stop'
                }
            ],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 5
            }
        }
        
        mock_response = create_mock_response(200, mock_response_data)
        mock_session_cm = create_mock_session(mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            request = InferenceRequest(
                prompt="Test prompt",
                max_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
            
            response = await external_api_adapter.invoke(request)
            
            assert isinstance(response, InferenceResponse)
            assert response.text == 'This is a test response'
            assert response.input_tokens == 10
            assert response.output_tokens == 5
            assert response.model_id == 'gpt-4'
            assert response.finish_reason == 'stop'

    async def test_invoke_400_error(self, external_api_adapter):
        """Test handling of 400 Bad Request error."""
        error_data = {'error': {'message': 'Invalid parameters'}}
        mock_response = create_mock_response(400, error_data)
        mock_session_cm = create_mock_session(mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            request = InferenceRequest(prompt="Test")
            
            with pytest.raises(ValueError, match="Invalid request"):
                await external_api_adapter.invoke(request)

    async def test_invoke_401_error(self, external_api_adapter):
        """Test handling of 401 Unauthorized error."""
        error_data = {'error': {'message': 'Invalid API key'}}
        mock_response = create_mock_response(401, error_data)
        mock_session_cm = create_mock_session(mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            request = InferenceRequest(prompt="Test")
            
            with pytest.raises(RuntimeError, match="Authentication failed"):
                await external_api_adapter.invoke(request)

    async def test_invoke_404_error(self, external_api_adapter):
        """Test handling of 404 Not Found error."""
        error_data = {'error': {'message': 'Model not found'}}
        mock_response = create_mock_response(404, error_data)
        mock_session_cm = create_mock_session(mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            request = InferenceRequest(prompt="Test")
            
            with pytest.raises(ValueError, match="Model not found"):
                await external_api_adapter.invoke(request)


class TestExternalAPIAdapterListModels:
    """Test model discovery."""

    async def test_list_models_success(self, external_api_adapter):
        """Test successful model listing."""
        mock_response_data = {
            'data': [
                {
                    'id': 'gpt-4',
                    'created': 1687882411,
                    'owned_by': 'openai'
                },
                {
                    'id': 'gpt-3.5-turbo',
                    'created': 1677610602,
                    'owned_by': 'openai'
                }
            ]
        }
        
        mock_response = create_mock_response(200, mock_response_data)
        mock_session_cm = create_mock_session(mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            models = await external_api_adapter.list_models()
            
            assert len(models) == 2
            assert models[0].id == 'gpt-4'
            assert models[0].provider == ModelProvider.EXTERNAL_API
            assert ModelCapability.TEXT_GENERATION in models[0].capabilities

    async def test_list_models_empty(self, external_api_adapter):
        """Test listing models when none are available."""
        mock_response_data = {'data': []}
        mock_response = create_mock_response(200, mock_response_data)
        mock_session_cm = create_mock_session(mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            models = await external_api_adapter.list_models()
            assert len(models) == 0


class TestExternalAPIAdapterGetModelInfo:
    """Test getting individual model info."""

    async def test_get_model_info_success(self, external_api_adapter):
        """Test successful model info retrieval."""
        mock_response_data = {
            'id': 'gpt-4',
            'created': 1687882411,
            'owned_by': 'openai'
        }
        
        mock_response = create_mock_response(200, mock_response_data)
        mock_session_cm = create_mock_session(mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            metadata = await external_api_adapter.get_model_info('gpt-4')
            
            assert isinstance(metadata, ModelMetadata)
            assert metadata.id == 'gpt-4'
            assert metadata.provider == ModelProvider.EXTERNAL_API

    async def test_get_model_info_not_found(self, external_api_adapter):
        """Test getting info for non-existent model."""
        mock_response = create_mock_response(404)
        mock_session_cm = create_mock_session(mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            with pytest.raises(ValueError, match="Model not found"):
                await external_api_adapter.get_model_info('invalid-model')


class TestExternalAPIAdapterValidateConnection:
    """Test connection validation."""

    async def test_validate_connection_success(self, external_api_adapter):
        """Test successful connection validation."""
        mock_response = create_mock_response(200, {'data': []})
        mock_session_cm = create_mock_session(mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            result = await external_api_adapter.validate_connection()
            assert result is True

    async def test_validate_connection_failure(self, external_api_adapter):
        """Test connection validation failure."""
        with patch('aiohttp.ClientSession', side_effect=Exception("Connection error")):
            result = await external_api_adapter.validate_connection()
            assert result is False


class TestExternalAPIAdapterCapabilities:
    """Test capability checking methods."""

    def test_supports_streaming(self, external_api_adapter):
        """Test streaming support check."""
        assert external_api_adapter.supports_streaming() is True

    def test_supports_fine_tuning(self, external_api_adapter):
        """Test fine-tuning support check."""
        result = external_api_adapter.supports_fine_tuning('gpt-4')
        assert result is False


class TestExternalAPIAdapterRateLimiting:
    """Test rate limiting functionality."""

    def test_update_rate_limit_info(self, external_api_adapter):
        """Test rate limit info extraction from headers."""
        headers = {
            'x-ratelimit-remaining-requests': '100',
            'x-ratelimit-reset-requests': '5m0s'
        }
        
        external_api_adapter._update_rate_limit_info(headers)
        
        assert external_api_adapter.rate_limit_remaining == 100
        assert external_api_adapter.rate_limit_reset == '5m0s'

    def test_get_retry_after_from_header(self, external_api_adapter):
        """Test retry-after extraction from Retry-After header."""
        headers = {'retry-after': '30'}
        retry_after = external_api_adapter._get_retry_after(headers)
        assert retry_after == 30.0

    def test_get_retry_after_default(self, external_api_adapter):
        """Test default retry-after when no headers present."""
        headers = {}
        retry_after = external_api_adapter._get_retry_after(headers)
        assert retry_after == 60.0
