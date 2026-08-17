"""External API adapter for OpenAI-compatible API providers.

This module implements the BaseModelAdapter interface for external API
providers that support OpenAI-compatible API format (OpenAI, Anthropic direct,
etc.), providing configurable base URL and authentication.

Requirements: 1.3, 1.6
"""

import asyncio
import json
import time
from typing import AsyncIterator, Optional
import aiohttp
from aiohttp import ClientError as AiohttpClientError, ClientTimeout

from src.adapters.base_adapter import (
    BaseModelAdapter,
    InferenceRequest,
    InferenceResponse,
)
from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
    ModelPricing,
)


class ExternalAPIAdapter(BaseModelAdapter):
    """Adapter for external API providers with OpenAI-compatible format.

    This adapter provides integration with external API providers that support
    OpenAI-compatible API format, including configurable base URL and
    authentication headers. It handles rate limiting and error responses.

    Requirements:
        - 1.3: External API credentials validation and registration
        - 1.6: Unified inference interface for external APIs
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        organization_id: Optional[str] = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize the External API adapter.

        Args:
            model_id: The model identifier to use for invocations
            base_url: Base URL for the API (e.g., 'https://api.openai.com/v1')
            api_key: API key for authentication
            organization_id: Optional organization ID for multi-tenant APIs
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.model_id = model_id
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.organization_id = organization_id
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Rate limiting state
        self._rate_limit_remaining = None
        self._rate_limit_reset = None

    def _get_headers(self) -> dict:
        """Get authentication headers for API requests.

        Returns:
            Dictionary of HTTP headers including authentication
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        if self.organization_id:
            headers['OpenAI-Organization'] = self.organization_id
        
        return headers

    async def invoke(self, request: InferenceRequest) -> InferenceResponse:
        """Invoke an external API model with the given request.

        Args:
            request: The inference request containing prompt and parameters

        Returns:
            InferenceResponse containing the generated text and metadata

        Raises:
            ValueError: If the request parameters are invalid
            RuntimeError: If the model invocation fails
        """
        # Format request body in OpenAI-compatible format
        body = {
            'model': self.model_id,
            'messages': [
                {'role': 'user', 'content': request.prompt}
            ],
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
        }
        
        if request.stop_sequences:
            body['stop'] = request.stop_sequences
        
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                timeout = ClientTimeout(total=self.timeout_seconds)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f'{self.base_url}/chat/completions',
                        headers=self._get_headers(),
                        json=body
                    ) as response:
                        # Update rate limit info from headers
                        self._update_rate_limit_info(response.headers)
                        
                        # Handle rate limiting
                        if response.status == 429:
                            retry_after = self._get_retry_after(response.headers)
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(retry_after)
                                continue
                            raise RuntimeError(
                                f"Rate limit exceeded. Retry after {retry_after} seconds."
                            )
                        
                        # Handle other error status codes
                        if response.status >= 400:
                            error_body = await response.text()
                            try:
                                error_data = json.loads(error_body)
                                error_message = error_data.get('error', {}).get('message', error_body)
                            except json.JSONDecodeError:
                                error_message = error_body
                            
                            if response.status == 400:
                                raise ValueError(f"Invalid request: {error_message}")
                            elif response.status == 401:
                                raise RuntimeError(f"Authentication failed: {error_message}")
                            elif response.status == 404:
                                raise ValueError(f"Model not found: {self.model_id}")
                            else:
                                raise RuntimeError(
                                    f"API error (status {response.status}): {error_message}"
                                )
                        
                        # Parse successful response
                        response_data = await response.json()
                        
                        # Extract response text
                        choices = response_data.get('choices', [])
                        if not choices:
                            raise RuntimeError("No response choices returned from API")
                        
                        message = choices[0].get('message', {})
                        response_text = message.get('content', '')
                        finish_reason = choices[0].get('finish_reason', 'stop')
                        
                        # Extract token usage
                        usage = response_data.get('usage', {})
                        input_tokens = usage.get('prompt_tokens', 0)
                        output_tokens = usage.get('completion_tokens', 0)
                        
                        # Calculate latency
                        latency_ms = (time.time() - start_time) * 1000
                        
                        return InferenceResponse(
                            text=response_text,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            latency_ms=latency_ms,
                            model_id=self.model_id,
                            finish_reason=finish_reason
                        )
            
            except aiohttp.ClientError as e:
                last_exception = RuntimeError(f"Network error: {str(e)}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    continue
            except asyncio.TimeoutError as e:
                last_exception = RuntimeError(
                    f"Request timeout after {self.timeout_seconds} seconds"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except (ValueError, RuntimeError):
                # Don't retry on validation or auth errors
                raise
            except Exception as e:
                last_exception = RuntimeError(f"Unexpected error: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
        
        # If we exhausted all retries, raise the last exception
        if last_exception:
            raise last_exception
        
        raise RuntimeError("Failed to invoke model after all retry attempts")

    async def invoke_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[str]:
        """Stream model response tokens as they are generated.

        Args:
            request: The inference request containing prompt and parameters

        Yields:
            str: Individual tokens or chunks of the generated response

        Raises:
            ValueError: If the request parameters are invalid
            RuntimeError: If the model invocation fails
        """
        # Format request body in OpenAI-compatible format with streaming
        body = {
            'model': self.model_id,
            'messages': [
                {'role': 'user', 'content': request.prompt}
            ],
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'stream': True,
        }
        
        if request.stop_sequences:
            body['stop'] = request.stop_sequences
        
        try:
            timeout = ClientTimeout(total=self.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f'{self.base_url}/chat/completions',
                    headers=self._get_headers(),
                    json=body
                ) as response:
                    # Update rate limit info
                    self._update_rate_limit_info(response.headers)
                    
                    # Handle errors
                    if response.status >= 400:
                        error_body = await response.text()
                        try:
                            error_data = json.loads(error_body)
                            error_message = error_data.get('error', {}).get('message', error_body)
                        except json.JSONDecodeError:
                            error_message = error_body
                        
                        if response.status == 400:
                            raise ValueError(f"Invalid request: {error_message}")
                        else:
                            raise RuntimeError(f"API error: {error_message}")
                    
                    # Stream response chunks
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        # Skip empty lines
                        if not line:
                            continue
                        
                        # Parse SSE format
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            
                            # Check for stream end
                            if data == '[DONE]':
                                break
                            
                            try:
                                chunk_data = json.loads(data)
                                choices = chunk_data.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                # Skip malformed chunks
                                continue
        
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error during streaming: {str(e)}") from e
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"Streaming timeout after {self.timeout_seconds} seconds"
            ) from e
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error during streaming: {str(e)}") from e

    async def list_models(self) -> list[ModelMetadata]:
        """List all available models from the external API.

        Returns:
            List of ModelMetadata objects describing available models

        Raises:
            RuntimeError: If model discovery fails
        """
        try:
            timeout = ClientTimeout(total=self.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f'{self.base_url}/models',
                    headers=self._get_headers()
                ) as response:
                    if response.status >= 400:
                        error_body = await response.text()
                        raise RuntimeError(
                            f"Failed to list models (status {response.status}): {error_body}"
                        )
                    
                    response_data = await response.json()
                    models_data = response_data.get('data', [])
                    
                    models = []
                    for model_data in models_data:
                        model_id = model_data.get('id')
                        if not model_id:
                            continue
                        
                        # Create metadata with available information
                        metadata = ModelMetadata(
                            id=model_id,
                            provider=ModelProvider.EXTERNAL_API,
                            name=model_id,
                            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
                            status=ModelStatus.ACTIVE,
                            fine_tuning_support=False,  # Most external APIs don't expose fine-tuning
                            input_modalities=['text'],
                            output_modalities=['text'],
                            max_tokens=4096,  # Default, varies by model
                            region='external',
                            pricing=ModelPricing(
                                input_price_per_1k_tokens=0.0,  # Pricing varies by provider
                                output_price_per_1k_tokens=0.0,
                                currency="USD"
                            ),
                            metadata={
                                'base_url': self.base_url,
                                'created': model_data.get('created'),
                                'owned_by': model_data.get('owned_by'),
                            }
                        )
                        
                        models.append(metadata)
                    
                    return models
        
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error during model discovery: {str(e)}") from e
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"Model discovery timeout: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during model discovery: {str(e)}") from e

    async def get_model_info(self, model_id: str) -> ModelMetadata:
        """Get metadata for a specific model.

        Args:
            model_id: The unique identifier of the model

        Returns:
            ModelMetadata object with model details

        Raises:
            ValueError: If the model_id is not found
            RuntimeError: If metadata retrieval fails
        """
        try:
            timeout = ClientTimeout(total=self.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f'{self.base_url}/models/{model_id}',
                    headers=self._get_headers()
                ) as response:
                    if response.status == 404:
                        raise ValueError(f"Model not found: {model_id}")
                    
                    if response.status >= 400:
                        error_body = await response.text()
                        raise RuntimeError(
                            f"Failed to get model info (status {response.status}): {error_body}"
                        )
                    
                    model_data = await response.json()
                    
                    # Create metadata
                    metadata = ModelMetadata(
                        id=model_id,
                        provider=ModelProvider.EXTERNAL_API,
                        name=model_id,
                        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
                        status=ModelStatus.ACTIVE,
                        fine_tuning_support=False,
                        input_modalities=['text'],
                        output_modalities=['text'],
                        max_tokens=4096,
                        region='external',
                        pricing=ModelPricing(
                            input_price_per_1k_tokens=0.0,
                            output_price_per_1k_tokens=0.0,
                            currency="USD"
                        ),
                        metadata={
                            'base_url': self.base_url,
                            'created': model_data.get('created'),
                            'owned_by': model_data.get('owned_by'),
                        }
                    )
                    
                    return metadata
        
        except ValueError:
            raise
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error: {str(e)}") from e
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"Request timeout: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {str(e)}") from e

    async def validate_connection(self) -> bool:
        """Validate connectivity to the external API.

        Performs a lightweight check by attempting to list models.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            timeout = ClientTimeout(total=10.0)  # Short timeout for validation
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f'{self.base_url}/models',
                    headers=self._get_headers()
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    def supports_streaming(self) -> bool:
        """Check if this adapter supports streaming responses.

        Returns:
            True - external APIs typically support streaming
        """
        return True

    def supports_fine_tuning(self, model_id: str) -> bool:
        """Check if the specified model supports fine-tuning.

        Args:
            model_id: The unique identifier of the model

        Returns:
            False - fine-tuning is typically not exposed via external APIs
        """
        # Most external APIs don't expose fine-tuning through the API
        # Fine-tuning is usually done through separate interfaces
        return False

    def _update_rate_limit_info(self, headers: dict) -> None:
        """Update rate limit information from response headers.

        Args:
            headers: HTTP response headers
        """
        # Extract rate limit headers (OpenAI format)
        if 'x-ratelimit-remaining-requests' in headers:
            try:
                self._rate_limit_remaining = int(
                    headers['x-ratelimit-remaining-requests']
                )
            except (ValueError, TypeError):
                pass
        
        if 'x-ratelimit-reset-requests' in headers:
            self._rate_limit_reset = headers['x-ratelimit-reset-requests']

    def _get_retry_after(self, headers: dict) -> float:
        """Get retry-after duration from response headers.

        Args:
            headers: HTTP response headers

        Returns:
            Number of seconds to wait before retrying
        """
        # Check for Retry-After header
        if 'retry-after' in headers:
            try:
                return float(headers['retry-after'])
            except (ValueError, TypeError):
                pass
        
        # Check for rate limit reset time
        if 'x-ratelimit-reset-requests' in headers:
            try:
                reset_time = headers['x-ratelimit-reset-requests']
                # Parse duration format (e.g., "6m0s")
                if 'm' in reset_time:
                    minutes = int(reset_time.split('m')[0])
                    return minutes * 60
                elif 's' in reset_time:
                    return float(reset_time.rstrip('s'))
            except (ValueError, TypeError):
                pass
        
        # Default backoff
        return 60.0

    @property
    def rate_limit_remaining(self) -> Optional[int]:
        """Get the number of remaining requests before rate limit.

        Returns:
            Number of remaining requests, or None if unknown
        """
        return self._rate_limit_remaining

    @property
    def rate_limit_reset(self) -> Optional[str]:
        """Get the rate limit reset time.

        Returns:
            Reset time string, or None if unknown
        """
        return self._rate_limit_reset
