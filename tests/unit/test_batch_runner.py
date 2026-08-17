"""
Unit tests for the batch inference runner.

Tests use a mocked InferenceClient to verify:
- Sequential and concurrent processing
- Timing information collection
- Graceful handling of individual failures
- Progress callback invocation
- Empty/edge-case inputs

Requirements: 3.3, 3.15
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.data_models.model import InferenceResponse
from src.evaluation.batch_runner import (
    BatchRunConfig,
    run_batch_inference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(
    text: str = "answer",
    input_tokens: int = 10,
    output_tokens: int = 5,
    latency_ms: float = 50.0,
    model_id: str = "test-model",
) -> InferenceResponse:
    return InferenceResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_id=model_id,
        finish_reason="stop",
    )


def _make_client(
    side_effect=None, return_value=None
) -> AsyncMock:
    """Create a mocked InferenceClient."""
    client = AsyncMock()
    if side_effect is not None:
        client.invoke = AsyncMock(side_effect=side_effect)
    elif return_value is not None:
        client.invoke = AsyncMock(return_value=return_value)
    else:
        client.invoke = AsyncMock(
            return_value=_make_response()
        )
    return client


DEFAULT_CONFIG = BatchRunConfig(model_id="test-model")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchRunnerEmptyInput:
    """Edge case: empty dataset."""

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty_summary(self):
        client = _make_client()
        summary = await run_batch_inference(
            items=[], client=client, config=DEFAULT_CONFIG
        )
        assert summary.total == 0
        assert summary.succeeded == 0
        assert summary.failed == 0
        assert summary.results == []
        client.invoke.assert_not_called()


class TestBatchRunnerSingleItem:
    """Processing a single dataset item."""

    @pytest.mark.asyncio
    async def test_single_item_success(self):
        client = _make_client()
        items = [{"prompt": "Hello?"}]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        assert summary.total == 1
        assert summary.succeeded == 1
        assert summary.failed == 0
        result = summary.results[0]
        assert result.success is True
        assert result.response_text == "answer"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.latency_ms > 0
        assert result.prompt == "Hello?"

    @pytest.mark.asyncio
    async def test_single_item_preserves_category(self):
        client = _make_client()
        items = [
            {
                "prompt": "Q",
                "category": "science",
                "expected_response": "A",
            }
        ]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        result = summary.results[0]
        assert result.category == "science"
        assert result.expected_response == "A"


class TestBatchRunnerMultipleItems:
    """Processing multiple items."""

    @pytest.mark.asyncio
    async def test_multiple_items_all_succeed(self):
        client = _make_client()
        items = [
            {"prompt": f"Q{i}"} for i in range(5)
        ]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        assert summary.total == 5
        assert summary.succeeded == 5
        assert summary.failed == 0
        assert len(summary.results) == 5
        # Results should be in order
        for i, r in enumerate(summary.results):
            assert r.index == i

    @pytest.mark.asyncio
    async def test_results_preserve_order(self):
        """Results are returned in the same order as input items."""
        responses = [
            _make_response(text=f"A{i}") for i in range(3)
        ]
        client = _make_client(side_effect=responses)
        items = [{"prompt": f"Q{i}"} for i in range(3)]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        for i, r in enumerate(summary.results):
            assert r.index == i
            assert r.response_text == f"A{i}"


class TestBatchRunnerFailureHandling:
    """Graceful handling of individual item failures."""

    @pytest.mark.asyncio
    async def test_failed_item_continues_processing(self):
        """One failure should not stop the rest."""
        side_effects = [
            _make_response(text="ok1"),
            RuntimeError("model error"),
            _make_response(text="ok3"),
        ]
        client = _make_client(side_effect=side_effects)
        items = [{"prompt": f"Q{i}"} for i in range(3)]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        assert summary.total == 3
        assert summary.succeeded == 2
        assert summary.failed == 1
        # Failed item
        failed = summary.results[1]
        assert failed.success is False
        assert "model error" in failed.error
        assert failed.response_text == ""
        # Successful items
        assert summary.results[0].success is True
        assert summary.results[2].success is True

    @pytest.mark.asyncio
    async def test_empty_prompt_marked_as_failure(self):
        client = _make_client()
        items = [{"prompt": ""}]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        assert summary.failed == 1
        assert summary.results[0].success is False
        assert "Empty prompt" in summary.results[0].error
        client.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_items_fail(self):
        client = _make_client(
            side_effect=RuntimeError("boom")
        )
        items = [{"prompt": f"Q{i}"} for i in range(3)]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        assert summary.total == 3
        assert summary.succeeded == 0
        assert summary.failed == 3


class TestBatchRunnerConcurrency:
    """Configurable concurrency support."""

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """With concurrency=2, at most 2 items run at once."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def _slow_invoke(*args, **kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return _make_response()

        client = AsyncMock()
        client.invoke = AsyncMock(side_effect=_slow_invoke)

        config = BatchRunConfig(
            model_id="test-model", concurrency=2
        )
        items = [{"prompt": f"Q{i}"} for i in range(6)]
        await run_batch_inference(
            items=items, client=client, config=config
        )
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_sequential_by_default(self):
        """Default concurrency=1 means sequential execution."""
        config = BatchRunConfig(model_id="test-model")
        assert config.concurrency == 1


class TestBatchRunnerProgressCallback:
    """Progress callback invocation."""

    @pytest.mark.asyncio
    async def test_callback_called_for_each_item(self):
        client = _make_client()
        calls = []

        def on_progress(completed: int, total: int):
            calls.append((completed, total))

        items = [{"prompt": f"Q{i}"} for i in range(3)]
        await run_batch_inference(
            items=items,
            client=client,
            config=DEFAULT_CONFIG,
            progress_callback=on_progress,
        )
        # Should have been called 3 times
        assert len(calls) == 3
        # All calls should have total=3
        assert all(t == 3 for _, t in calls)
        # Final call should show all completed
        completed_values = sorted(c for c, _ in calls)
        assert completed_values == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_no_callback_is_fine(self):
        """No error when progress_callback is None."""
        client = _make_client()
        items = [{"prompt": "Q"}]
        summary = await run_batch_inference(
            items=items,
            client=client,
            config=DEFAULT_CONFIG,
            progress_callback=None,
        )
        assert summary.succeeded == 1


class TestBatchRunnerConfig:
    """Config is forwarded to InferenceClient."""

    @pytest.mark.asyncio
    async def test_config_params_forwarded(self):
        client = _make_client()
        config = BatchRunConfig(
            model_id="my-model",
            max_tokens=256,
            temperature=0.5,
            timeout_seconds=30.0,
        )
        items = [{"prompt": "Q"}]
        await run_batch_inference(
            items=items, client=client, config=config
        )
        call_kwargs = client.invoke.call_args
        assert call_kwargs.kwargs["model_id"] == "my-model"
        assert call_kwargs.kwargs["timeout_seconds"] == 30.0
        req = call_kwargs.kwargs["request"]
        assert req.max_tokens == 256
        assert req.temperature == 0.5


class TestBatchRunnerTimingInfo:
    """Timing information is collected."""

    @pytest.mark.asyncio
    async def test_latency_is_positive(self):
        client = _make_client()
        items = [{"prompt": "Q"}]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        assert summary.results[0].latency_ms > 0

    @pytest.mark.asyncio
    async def test_failed_item_still_has_latency(self):
        client = _make_client(
            side_effect=RuntimeError("fail")
        )
        items = [{"prompt": "Q"}]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        # Even failed items record elapsed time
        assert summary.results[0].latency_ms >= 0


# ---------------------------------------------------------------------------
# Timeout handling (Requirements: 3.11, 3.12, 3.18, 3.19)
# ---------------------------------------------------------------------------

from src.evaluation.timeout_config import TimeoutConfig


class TestBatchRunnerTimeoutHandling:
    """Timeout enforcement via asyncio.wait_for."""

    @pytest.mark.asyncio
    async def test_timeout_marks_item_as_failed(self):
        """When client.invoke hangs, the item should be marked failed
        with a clear 'Timeout' error message."""

        async def _hang(*args, **kwargs):
            await asyncio.sleep(10)  # will be cancelled by timeout
            return _make_response()

        client = AsyncMock()
        client.invoke = AsyncMock(side_effect=_hang)

        config = BatchRunConfig(
            model_id="test-model",
            timeout_seconds=0.05,  # 50ms timeout
        )
        items = [{"prompt": "Q"}]
        summary = await run_batch_inference(
            items=items, client=client, config=config
        )
        assert summary.failed == 1
        result = summary.results[0]
        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_timeout_continues_with_remaining_items(self):
        """A timed-out item should not prevent other items from processing."""
        call_count = 0

        async def _mixed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                await asyncio.sleep(10)  # will timeout
            return _make_response()

        client = AsyncMock()
        client.invoke = AsyncMock(side_effect=_mixed)

        config = BatchRunConfig(
            model_id="test-model",
            timeout_seconds=0.05,
        )
        items = [{"prompt": f"Q{i}"} for i in range(3)]
        summary = await run_batch_inference(
            items=items, client=client, config=config
        )
        assert summary.total == 3
        assert summary.succeeded == 2
        assert summary.failed == 1
        assert summary.results[1].success is False
        assert "Timeout" in summary.results[1].error

    @pytest.mark.asyncio
    async def test_timeout_config_resolves_for_small_model(self):
        """TimeoutConfig should resolve a shorter timeout for small models."""
        client = _make_client()
        tc = TimeoutConfig(small=15.0, medium=30.0, large=60.0)
        config = BatchRunConfig(
            model_id="claude-haiku-model",
            timeout_config=tc,
        )
        items = [{"prompt": "Q"}]
        summary = await run_batch_inference(
            items=items, client=client, config=config
        )
        assert summary.succeeded == 1
        # Verify the resolved timeout was passed to client.invoke
        call_kwargs = client.invoke.call_args
        assert call_kwargs.kwargs["timeout_seconds"] == 15.0

    @pytest.mark.asyncio
    async def test_timeout_config_resolves_for_large_model(self):
        """TimeoutConfig should resolve a longer timeout for large models."""
        client = _make_client()
        tc = TimeoutConfig(small=15.0, medium=30.0, large=60.0)
        config = BatchRunConfig(
            model_id="claude-opus-model",
            timeout_config=tc,
        )
        items = [{"prompt": "Q"}]
        summary = await run_batch_inference(
            items=items, client=client, config=config
        )
        assert summary.succeeded == 1
        call_kwargs = client.invoke.call_args
        assert call_kwargs.kwargs["timeout_seconds"] == 60.0


# ---------------------------------------------------------------------------
# Error handling for individual prompt failures (Requirement: 3.14)
# ---------------------------------------------------------------------------


class TestBatchRunnerErrorDetails:
    """Verify error details are logged and partial results reported."""

    @pytest.mark.asyncio
    async def test_error_message_preserved_in_result(self):
        """The specific error message should be captured in the result."""
        client = _make_client(
            side_effect=ValueError("Invalid model response format")
        )
        items = [{"prompt": "Q"}]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        assert summary.failed == 1
        assert "Invalid model response format" in summary.results[0].error

    @pytest.mark.asyncio
    async def test_partial_results_with_mixed_failures(self):
        """Summary should accurately report succeeded and failed counts."""
        side_effects = [
            _make_response(text="ok1"),
            RuntimeError("network error"),
            _make_response(text="ok3"),
            ConnectionError("connection reset"),
            _make_response(text="ok5"),
        ]
        client = _make_client(side_effect=side_effects)
        items = [{"prompt": f"Q{i}"} for i in range(5)]
        summary = await run_batch_inference(
            items=items, client=client, config=DEFAULT_CONFIG
        )
        assert summary.total == 5
        assert summary.succeeded == 3
        assert summary.failed == 2
        # Successful items have response text
        assert summary.results[0].response_text == "ok1"
        assert summary.results[2].response_text == "ok3"
        assert summary.results[4].response_text == "ok5"
        # Failed items have error details
        assert "network error" in summary.results[1].error
        assert "connection reset" in summary.results[3].error
