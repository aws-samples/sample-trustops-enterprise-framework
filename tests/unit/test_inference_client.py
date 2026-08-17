"""Unit tests for InferenceClient.

This module tests the unified inference client including rate limiting,
retry logic, timeout handling, and batch inference.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from src.clients.inference_client import InferenceClient
from src.adapters.base_adapter import (
    InferenceRequest,
    InferenceResponse,
    BaseModelAdapter,
)
from src.registry.model_registry import ModelRegistry
from src.utils.rate_limiter import RateLimiter, RateLimitConfig
from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
)


# Test fixtures


@pytest.fixture
def mock_adapter():
    """Create a mock adapter for testing."""
    adapter = AsyncMock(spec=BaseModelAdapter)
    adapter.invoke = AsyncMock(
        return_value=InferenceResponse(
            text="Test response",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100.0,
            model_id="test-model",
            finish_reason="stop",
        )
    )
    return adapter


@pytest.fixture
def mock_registry(mock_adapter):
    """Create a mock registry for testing."""
    registry = AsyncMock(spec=ModelRegistry)

    # Mock model metadata
    metadata = ModelMetadata(
        id="test-model",
        provider=ModelProvider.BEDROCK,
        name="Test Model",
        capabilities=[ModelCapability.TEXT_GENERATION],
        status=ModelStatus.ACTIVE,
        fine_tuning_support=False,
        max_tokens=4096,
        region="us-east-1",
    )

    registry.get_model = AsyncMock(return_value=metadata)
    registry.get_adapter = Mock(return_value=mock_adapter)

    return registry


@pytest.fixture
def rate_limiter():
    """Create a rate limiter for testing."""
    limiter = RateLimiter()
    # Configure with generous limits for testing
    limiter.set_limit(
        "bedrock",
        RateLimitConfig(
            requests_per_minute=1000, tokens_per_minute=100000
        ),
    )
    return limiter


@pytest.fixture
def inference_client(mock_registry, rate_limiter):
    """Create an InferenceClient for testing."""
    return InferenceClient(
        registry=mock_registry,
        rate_limiter=rate_limiter,
        default_timeout=10.0,
        max_retries=3,
    )


@pytest.fixture
def sample_request():
    """Create a sample inference request."""
    return InferenceRequest(
        prompt="What is the capital of France?",
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
    )


# Test cases


class TestInferenceClientBasic:
    """Test basic InferenceClient functionality."""

    @pytest.mark.asyncio
    async def test_invoke_success(
        self, inference_client, sample_request
    ):
        """Test successful model invocation."""
        response = await inference_client.invoke(
            "test-model", sample_request
        )

        assert response.text == "Test response"
        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.model_id == "test-model"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_invoke_model_not_found(
        self, inference_client, sample_request
    ):
        """Test invocation with non-existent model."""
        # Mock registry to return None
        inference_client.registry.get_model = AsyncMock(
            return_value=None
        )

        with pytest.raises(ValueError, match="Model not found"):
            await inference_client.invoke(
                "nonexistent-model", sample_request
            )

    @pytest.mark.asyncio
    async def test_invoke_no_adapter(
        self, inference_client, sample_request
    ):
        """Test invocation when no adapter is available."""
        # Mock registry to return None for adapter
        inference_client.registry.get_adapter = Mock(return_value=None)

        with pytest.raises(ValueError, match="No adapter available"):
            await inference_client.invoke("test-model", sample_request)

    @pytest.mark.asyncio
    async def test_invoke_with_custom_timeout(
        self, inference_client, sample_request
    ):
        """Test invocation with custom timeout."""
        response = await inference_client.invoke(
            "test-model", sample_request, timeout_seconds=5.0
        )

        assert response.text == "Test response"

    @pytest.mark.asyncio
    async def test_invoke_uses_default_timeout(
        self, inference_client, sample_request
    ):
        """Test that default timeout is used when not specified."""
        # Should use default_timeout=10.0 from fixture
        response = await inference_client.invoke(
            "test-model", sample_request
        )

        assert response.text == "Test response"


class TestInferenceClientRetry:
    """Test retry logic in InferenceClient."""

    @pytest.mark.asyncio
    async def test_retry_on_throttling_error(
        self, inference_client, sample_request, mock_adapter
    ):
        """Test retry on throttling error."""
        # First two calls fail with throttling, third succeeds
        mock_adapter.invoke = AsyncMock(
            side_effect=[
                RuntimeError("Rate limit exceeded"),
                RuntimeError("Throttling error"),
                InferenceResponse(
                    text="Success after retry",
                    input_tokens=10,
                    output_tokens=20,
                    latency_ms=100.0,
                    model_id="test-model",
                    finish_reason="stop",
                ),
            ]
        )

        response = await inference_client.invoke(
            "test-model", sample_request
        )

        assert response.text == "Success after retry"
        assert mock_adapter.invoke.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(
        self, inference_client, sample_request, mock_adapter
    ):
        """Test retry on connection error."""
        # First call fails with connection error, second succeeds
        mock_adapter.invoke = AsyncMock(
            side_effect=[
                RuntimeError("Connection timeout"),
                InferenceResponse(
                    text="Success after retry",
                    input_tokens=10,
                    output_tokens=20,
                    latency_ms=100.0,
                    model_id="test-model",
                    finish_reason="stop",
                ),
            ]
        )

        response = await inference_client.invoke(
            "test-model", sample_request
        )

        assert response.text == "Success after retry"
        assert mock_adapter.invoke.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_validation_error(
        self, inference_client, sample_request, mock_adapter
    ):
        """Test that validation errors are not retried."""
        # Non-retryable error should fail immediately
        mock_adapter.invoke = AsyncMock(
            side_effect=ValueError("Invalid parameter")
        )

        with pytest.raises(ValueError, match="Invalid parameter"):
            await inference_client.invoke("test-model", sample_request)

        # Should only be called once (no retries)
        assert mock_adapter.invoke.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhaustion(
        self, inference_client, sample_request, mock_adapter
    ):
        """Test that retries are exhausted after max attempts."""
        # All calls fail with retryable error
        mock_adapter.invoke = AsyncMock(
            side_effect=RuntimeError("Service unavailable")
        )

        with pytest.raises(
            RuntimeError, match="failed after 3 retries"
        ):
            await inference_client.invoke("test-model", sample_request)

        # Should be called max_retries times
        assert mock_adapter.invoke.call_count == 3


class TestInferenceClientTimeout:
    """Test timeout handling in InferenceClient."""

    @pytest.mark.asyncio
    async def test_timeout_exceeded(
        self, inference_client, sample_request, mock_adapter
    ):
        """Test that timeout is enforced."""

        async def slow_invoke(request):
            """Simulate slow response."""
            await asyncio.sleep(2.0)
            return InferenceResponse(
                text="Too slow",
                input_tokens=10,
                output_tokens=20,
                latency_ms=2000.0,
                model_id="test-model",
                finish_reason="stop",
            )

        mock_adapter.invoke = slow_invoke

        with pytest.raises(asyncio.TimeoutError, match="timed out"):
            await inference_client.invoke(
                "test-model", sample_request, timeout_seconds=0.5
            )

    @pytest.mark.asyncio
    async def test_timeout_not_exceeded(
        self, inference_client, sample_request, mock_adapter
    ):
        """Test successful completion within timeout."""

        async def fast_invoke(request):
            """Simulate fast response."""
            await asyncio.sleep(0.1)
            return InferenceResponse(
                text="Fast response",
                input_tokens=10,
                output_tokens=20,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            )

        mock_adapter.invoke = fast_invoke

        response = await inference_client.invoke(
            "test-model", sample_request, timeout_seconds=1.0
        )

        assert response.text == "Fast response"


class TestInferenceClientRateLimiting:
    """Test rate limiting in InferenceClient."""

    @pytest.mark.asyncio
    async def test_rate_limiting_applied(
        self, inference_client, sample_request
    ):
        """Test that rate limiting is applied."""
        # Rate limiter should allow the request
        response = await inference_client.invoke(
            "test-model", sample_request
        )

        assert response.text == "Test response"

    @pytest.mark.asyncio
    async def test_rate_limiting_not_configured(
        self, mock_registry, sample_request, mock_adapter
    ):
        """Test behavior when rate limiter not configured."""
        # Create client with empty rate limiter
        limiter = RateLimiter()
        client = InferenceClient(
            registry=mock_registry,
            rate_limiter=limiter,
            default_timeout=10.0,
        )

        # Should proceed without rate limiting
        response = await client.invoke("test-model", sample_request)

        assert response.text == "Test response"

    @pytest.mark.asyncio
    async def test_rate_limiter_integration(
        self, mock_registry, sample_request, mock_adapter
    ):
        """Test that rate limiter is properly integrated."""
        # Create rate limiter with reasonable limits
        limiter = RateLimiter()
        limiter.set_limit(
            "bedrock",
            RateLimitConfig(
                requests_per_minute=100,
                tokens_per_minute=10000
            ),
        )

        client = InferenceClient(
            registry=mock_registry,
            rate_limiter=limiter,
            default_timeout=10.0,
        )

        # Make several requests - all should succeed
        for _ in range(5):
            response = await client.invoke("test-model", sample_request)
            assert response.text == "Test response"

        # Verify rate limiter has tracked the requests
        capacity = limiter.get_available_capacity("bedrock")
        assert "requests" in capacity
        # Some capacity should have been consumed
        assert capacity["requests"] < 100


class TestInferenceClientBatch:
    """Test batch inference functionality."""

    @pytest.mark.asyncio
    async def test_invoke_batch_success(
        self, inference_client, mock_adapter
    ):
        """Test successful batch invocation."""
        requests = [
            InferenceRequest(prompt=f"Prompt {i}", max_tokens=100)
            for i in range(5)
        ]

        # Mock adapter to return different responses
        mock_adapter.invoke = AsyncMock(
            side_effect=[
                InferenceResponse(
                    text=f"Response {i}",
                    input_tokens=10,
                    output_tokens=20,
                    latency_ms=100.0,
                    model_id="test-model",
                    finish_reason="stop",
                )
                for i in range(5)
            ]
        )

        responses = await inference_client.invoke_batch(
            "test-model", requests, concurrency=2
        )

        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.text == f"Response {i}"

    @pytest.mark.asyncio
    async def test_invoke_batch_empty(self, inference_client):
        """Test batch invocation with empty list."""
        responses = await inference_client.invoke_batch(
            "test-model", []
        )

        assert responses == []

    @pytest.mark.asyncio
    async def test_invoke_batch_with_failure(
        self, inference_client, mock_adapter
    ):
        """Test batch invocation with one failure."""
        requests = [
            InferenceRequest(prompt=f"Prompt {i}", max_tokens=100)
            for i in range(3)
        ]

        # Second request fails
        mock_adapter.invoke = AsyncMock(
            side_effect=[
                InferenceResponse(
                    text="Response 0",
                    input_tokens=10,
                    output_tokens=20,
                    latency_ms=100.0,
                    model_id="test-model",
                    finish_reason="stop",
                ),
                RuntimeError("Inference failed"),
                InferenceResponse(
                    text="Response 2",
                    input_tokens=10,
                    output_tokens=20,
                    latency_ms=100.0,
                    model_id="test-model",
                    finish_reason="stop",
                ),
            ]
        )

        # Should raise the exception
        with pytest.raises(RuntimeError, match="Inference failed"):
            await inference_client.invoke_batch("test-model", requests)

    @pytest.mark.asyncio
    async def test_invoke_batch_concurrency_limit(
        self, inference_client, mock_adapter
    ):
        """Test that concurrency limit is respected."""
        requests = [
            InferenceRequest(prompt=f"Prompt {i}", max_tokens=100)
            for i in range(10)
        ]

        # Track concurrent calls
        concurrent_calls = 0
        max_concurrent = 0

        async def track_invoke(request):
            """Track concurrent invocations."""
            nonlocal concurrent_calls, max_concurrent
            concurrent_calls += 1
            max_concurrent = max(max_concurrent, concurrent_calls)
            await asyncio.sleep(0.1)
            concurrent_calls -= 1
            return InferenceResponse(
                text="Response",
                input_tokens=10,
                output_tokens=20,
                latency_ms=100.0,
                model_id="test-model",
                finish_reason="stop",
            )

        mock_adapter.invoke = track_invoke

        responses = await inference_client.invoke_batch(
            "test-model", requests, concurrency=3
        )

        assert len(responses) == 10
        # Max concurrent should not exceed the limit
        assert max_concurrent <= 3


class TestInferenceClientIntegration:
    """Integration tests for InferenceClient."""

    @pytest.mark.asyncio
    async def test_end_to_end_flow(
        self, inference_client, sample_request
    ):
        """Test complete end-to-end inference flow."""
        # This test verifies the complete flow:
        # 1. Model lookup
        # 2. Adapter retrieval
        # 3. Rate limiting
        # 4. Invocation
        # 5. Response return

        response = await inference_client.invoke(
            "test-model", sample_request
        )

        # Verify registry was called
        inference_client.registry.get_model.assert_called_once_with(
            "test-model"
        )
        inference_client.registry.get_adapter.assert_called_once_with(
            "test-model"
        )

        # Verify response
        assert response.text == "Test response"
        assert response.model_id == "test-model"

    @pytest.mark.asyncio
    async def test_multiple_sequential_invocations(
        self, inference_client, sample_request
    ):
        """Test multiple sequential invocations."""
        for i in range(5):
            response = await inference_client.invoke(
                "test-model", sample_request
            )
            assert response.text == "Test response"

    @pytest.mark.asyncio
    async def test_different_models(self, mock_registry, rate_limiter):
        """Test invoking different models."""
        # Create adapters for different models
        adapter1 = AsyncMock(spec=BaseModelAdapter)
        adapter1.invoke = AsyncMock(
            return_value=InferenceResponse(
                text="Response from model 1",
                input_tokens=10,
                output_tokens=20,
                latency_ms=100.0,
                model_id="model-1",
                finish_reason="stop",
            )
        )

        adapter2 = AsyncMock(spec=BaseModelAdapter)
        adapter2.invoke = AsyncMock(
            return_value=InferenceResponse(
                text="Response from model 2",
                input_tokens=15,
                output_tokens=25,
                latency_ms=150.0,
                model_id="model-2",
                finish_reason="stop",
            )
        )

        # Mock registry to return different adapters
        def get_adapter(model_id):
            if model_id == "model-1":
                return adapter1
            elif model_id == "model-2":
                return adapter2
            return None

        mock_registry.get_adapter = Mock(side_effect=get_adapter)

        # Mock get_model to return appropriate metadata
        async def get_model(model_id):
            return ModelMetadata(
                id=model_id,
                provider=ModelProvider.BEDROCK,
                name=f"Test {model_id}",
                capabilities=[ModelCapability.TEXT_GENERATION],
                status=ModelStatus.ACTIVE,
                fine_tuning_support=False,
                max_tokens=4096,
                region="us-east-1",
            )

        mock_registry.get_model = AsyncMock(side_effect=get_model)

        client = InferenceClient(mock_registry, rate_limiter)

        # Invoke both models
        request = InferenceRequest(prompt="Test", max_tokens=100)

        response1 = await client.invoke("model-1", request)
        assert response1.text == "Response from model 1"
        assert response1.model_id == "model-1"

        response2 = await client.invoke("model-2", request)
        assert response2.text == "Response from model 2"
        assert response2.model_id == "model-2"
