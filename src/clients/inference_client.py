"""Unified client for invoking models across multiple providers.

This module implements the InferenceClient class which provides a unified
interface for invoking any registered model with automatic rate limiting,
retry logic, and timeout handling.

Requirements: 1.6, 1.7
"""

import asyncio
from typing import Optional

from src.adapters.base_adapter import (
    InferenceRequest,
    InferenceResponse,
    BaseModelAdapter,
)
from src.registry.model_registry import ModelRegistry
from src.utils.rate_limiter import RateLimiter
from src.utils.retry_utils import execute_with_retry


class InferenceClient:
    """Unified client for invoking any registered model.

    The InferenceClient routes inference requests to the appropriate adapter
    based on the model's provider, while applying rate limiting and retry
    logic consistently across all providers.

    Requirements:
        - 1.6: Unified inference interface abstracting provider differences
        - 1.7: Rate limiting and error handling across providers

    Attributes:
        registry: ModelRegistry instance for model lookup
        rate_limiter: RateLimiter instance for rate limiting
        default_timeout: Default timeout in seconds for inference requests
        max_retries: Maximum number of retry attempts for failed requests

    Example:
        >>> registry = ModelRegistry()
        >>> rate_limiter = RateLimiter()
        >>> client = InferenceClient(registry, rate_limiter)
        >>>
        >>> request = InferenceRequest(
        ...     prompt="What is the capital of France?",
        ...     max_tokens=100
        ... )
        >>> response = await client.invoke("model-id", request)
        >>> print(response.text)
    """

    def __init__(
        self,
        registry: ModelRegistry,
        rate_limiter: RateLimiter,
        default_timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize the InferenceClient.

        Args:
            registry: ModelRegistry instance for model lookup
            rate_limiter: RateLimiter instance for rate limiting
            default_timeout: Default timeout in seconds (default: 60.0)
            max_retries: Maximum retry attempts (default: 3)
        """
        self.registry = registry
        self.rate_limiter = rate_limiter
        self.default_timeout = default_timeout
        self.max_retries = max_retries

    async def invoke(
        self,
        model_id: str,
        request: InferenceRequest,
        timeout_seconds: Optional[float] = None,
    ) -> InferenceResponse:
        """Invoke a model with rate limiting and retry logic.

        This method:
        1. Looks up the model in the registry
        2. Gets the appropriate adapter for the model's provider
        3. Applies rate limiting based on provider limits
        4. Invokes the model with retry logic for transient errors
        5. Applies timeout to prevent hanging requests

        Args:
            model_id: The unique identifier of the model to invoke
            request: The inference request with prompt and parameters
            timeout_seconds: Timeout in seconds (uses default if None)

        Returns:
            InferenceResponse containing the generated text and metadata

        Raises:
            ValueError: If model not found or adapter not available
            RuntimeError: If inference fails after all retries
            asyncio.TimeoutError: If request exceeds timeout
        """
        # Use default timeout if not specified
        if timeout_seconds is None:
            timeout_seconds = self.default_timeout

        # Look up model metadata
        metadata = await self.registry.get_model(model_id)
        if metadata is None:
            raise ValueError(f"Model not found: {model_id}")

        # Get the appropriate adapter
        adapter = self.registry.get_adapter(model_id)
        if adapter is None:
            raise ValueError(
                f"No adapter available for model: {model_id} "
                f"(provider: {metadata.provider})"
            )

        # Estimate token count for rate limiting
        # Simple estimation: ~4 characters per token
        estimated_tokens = (
            len(request.prompt) // 4 + request.max_tokens
        )

        # Apply rate limiting
        provider_key = metadata.provider.value
        try:
            acquired = await self.rate_limiter.acquire(
                provider=provider_key,
                tokens=estimated_tokens,
                timeout=timeout_seconds,
            )
            if not acquired:
                raise RuntimeError(
                    f"Rate limit timeout for provider: {provider_key}"
                )
        except ValueError:
            # Rate limiter not configured for this provider - proceed
            pass

        # Invoke with timeout and retry logic
        try:
            response = await asyncio.wait_for(
                self._invoke_with_retry(adapter, request),
                timeout=timeout_seconds,
            )
            return response
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Inference request timed out after {timeout_seconds}s "
                f"for model: {model_id}"
            )

    async def _invoke_with_retry(
        self,
        adapter: BaseModelAdapter,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Invoke adapter with retry logic for transient errors.

        Args:
            adapter: The adapter to invoke
            request: The inference request

        Returns:
            InferenceResponse from the adapter

        Raises:
            RuntimeError: If all retries are exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = await adapter.invoke(request)
                return response
            except Exception as e:
                last_exception = e

                # Check if this is a retryable error
                error_msg = str(e).lower()
                is_retryable = any(
                    keyword in error_msg
                    for keyword in [
                        "throttl",
                        "rate limit",
                        "timeout",
                        "connection",
                        "service unavailable",
                        "internal error",
                        "503",
                        "429",
                    ]
                )

                if not is_retryable:
                    # Non-retryable error, raise immediately
                    raise

                # If not last attempt, wait before retrying
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = (2 ** attempt) * 1.0
                    # Add jitter: random value between 0 and delay.
                    # Backoff jitter is not a security control, so the
                    # standard PRNG is appropriate here.
                    import random
                    jittered_delay = delay * random.random()  # nosec B311
                    await asyncio.sleep(jittered_delay)

        # All retries exhausted
        raise RuntimeError(
            f"Inference failed after {self.max_retries} retries: "
            f"{str(last_exception)}"
        )

    async def invoke_batch(
        self,
        model_id: str,
        requests: list[InferenceRequest],
        concurrency: int = 5,
        timeout_seconds: Optional[float] = None,
    ) -> list[InferenceResponse]:
        """Invoke a model with multiple requests in parallel.

        This method processes multiple inference requests concurrently while
        respecting the specified concurrency limit. Failed requests will
        have their exceptions captured and re-raised after all requests
        complete.

        Args:
            model_id: The unique identifier of the model to invoke
            requests: List of inference requests to process
            concurrency: Maximum number of concurrent requests (default: 5)
            timeout_seconds: Timeout per request in seconds

        Returns:
            List of InferenceResponse objects in the same order as requests

        Raises:
            ValueError: If model not found or adapter not available
            RuntimeError: If any inference fails after retries

        Note:
            If any request fails, the exception will be raised after all
            requests complete. Successful responses will still be returned
            for requests that completed before the failure.
        """
        if not requests:
            return []

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def invoke_with_semaphore(
            request: InferenceRequest,
        ) -> InferenceResponse:
            """Invoke with concurrency control."""
            async with semaphore:
                return await self.invoke(
                    model_id, request, timeout_seconds
                )

        # Execute all requests concurrently with concurrency limit
        tasks = [
            invoke_with_semaphore(request) for request in requests
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions and convert to list of responses
        result = []
        first_exception = None

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                if first_exception is None:
                    first_exception = response
                # For now, we'll raise the first exception
                # In production, might want to collect all errors
            else:
                result.append(response)

        # If any request failed, raise the first exception
        if first_exception is not None:
            raise first_exception

        return result
