"""
Unit tests for the parallel model invoker.

Tests use a mocked InferenceClient to verify:
- Both models are invoked concurrently for each prompt
- Paired responses are collected correctly
- Graceful error handling when one or both models fail
- Concurrency control via semaphore
- Progress callback invocation
- Empty/edge-case inputs

Requirements: 7.1, 7.5
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.data_models.model import InferenceResponse
from src.evaluation.parallel_invoker import (
    PairedResponse,
    ParallelInvokeConfig,
    run_parallel_invocation,
)


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _resp(
    text: str = "answer",
    input_tokens: int = 10,
    output_tokens: int = 5,
    latency_ms: float = 50.0,
    model_id: str = "m1",
) -> InferenceResponse:
    return InferenceResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_id=model_id,
        finish_reason="stop",
    )


def _make_client(side_effect=None) -> AsyncMock:
    """Create a mocked InferenceClient.

    If *side_effect* is provided it is set on ``client.invoke``.
    Otherwise invoke returns a default response whose model_id
    matches the *model_id* kwarg passed at call time.
    """
    client = AsyncMock()
    if side_effect is not None:
        client.invoke = AsyncMock(side_effect=side_effect)
    else:
        async def _default_invoke(
            model_id, request, timeout_seconds=60.0
        ):
            return _resp(
                text=f"resp-{model_id}",
                model_id=model_id,
            )
        client.invoke = AsyncMock(
            side_effect=_default_invoke
        )
    return client


DEFAULT_CFG = ParallelInvokeConfig(
    model_id_1="model-a",
    model_id_2="model-b",
)


# -----------------------------------------------------------
# Tests
# -----------------------------------------------------------


class TestParallelInvokerEmptyInput:
    """Edge case: empty dataset."""

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty_summary(self):
        client = _make_client()
        summary = await run_parallel_invocation(
            items=[], client=client, config=DEFAULT_CFG
        )
        assert summary.total == 0
        assert summary.results == []
        assert summary.both_succeeded == 0

    @pytest.mark.asyncio
    async def test_empty_items_does_not_invoke(self):
        client = _make_client()
        await run_parallel_invocation(
            items=[], client=client, config=DEFAULT_CFG
        )
        client.invoke.assert_not_called()


class TestParallelInvokerSinglePrompt:
    """Single prompt sent to both models."""

    @pytest.mark.asyncio
    async def test_both_models_invoked(self):
        client = _make_client()
        items = [{"prompt": "Hello"}]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        assert client.invoke.call_count == 2
        assert summary.total == 1
        assert summary.both_succeeded == 1

    @pytest.mark.asyncio
    async def test_paired_response_fields(self):
        client = _make_client()
        items = [{"prompt": "Hi", "category": "greet"}]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        pair = summary.results[0]
        assert pair.prompt == "Hi"
        assert pair.model_1_id == "model-a"
        assert pair.model_2_id == "model-b"
        assert pair.response_1 is not None
        assert pair.response_2 is not None
        assert pair.error_1 is None
        assert pair.error_2 is None
        assert pair.category == "greet"
        assert pair.both_succeeded is True
        assert pair.any_failed is False

    @pytest.mark.asyncio
    async def test_response_model_ids_match(self):
        client = _make_client()
        items = [{"prompt": "Q"}]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        pair = summary.results[0]
        assert pair.response_1.model_id == "model-a"
        assert pair.response_2.model_id == "model-b"


class TestParallelInvokerMultiplePrompts:
    """Multiple prompts processed concurrently."""

    @pytest.mark.asyncio
    async def test_all_prompts_processed(self):
        client = _make_client()
        items = [
            {"prompt": "Q1"},
            {"prompt": "Q2"},
            {"prompt": "Q3"},
        ]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        assert summary.total == 3
        assert summary.both_succeeded == 3
        # 2 invocations per prompt
        assert client.invoke.call_count == 6

    @pytest.mark.asyncio
    async def test_results_preserve_order(self):
        client = _make_client()
        items = [
            {"prompt": "A"},
            {"prompt": "B"},
            {"prompt": "C"},
        ]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        prompts = [r.prompt for r in summary.results]
        assert prompts == ["A", "B", "C"]


class TestParallelInvokerFailureHandling:
    """Graceful handling when one or both models fail."""

    @pytest.mark.asyncio
    async def test_model_1_failure_recorded(self):
        call_count = 0

        async def _side_effect(
            model_id, request, timeout_seconds=60.0
        ):
            nonlocal call_count
            call_count += 1
            if model_id == "model-a":
                raise RuntimeError("model-a down")
            return _resp(model_id=model_id)

        client = _make_client(side_effect=_side_effect)
        items = [{"prompt": "Q"}]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        pair = summary.results[0]
        assert pair.response_1 is None
        assert pair.error_1 == "model-a down"
        assert pair.response_2 is not None
        assert pair.error_2 is None
        assert pair.both_succeeded is False
        assert pair.any_failed is True
        assert summary.model_1_failures == 1
        assert summary.model_2_failures == 0

    @pytest.mark.asyncio
    async def test_model_2_failure_recorded(self):
        async def _side_effect(
            model_id, request, timeout_seconds=60.0
        ):
            if model_id == "model-b":
                raise RuntimeError("model-b down")
            return _resp(model_id=model_id)

        client = _make_client(side_effect=_side_effect)
        items = [{"prompt": "Q"}]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        pair = summary.results[0]
        assert pair.response_1 is not None
        assert pair.response_2 is None
        assert pair.error_2 == "model-b down"
        assert summary.model_1_failures == 0
        assert summary.model_2_failures == 1

    @pytest.mark.asyncio
    async def test_both_models_fail(self):
        async def _side_effect(
            model_id, request, timeout_seconds=60.0
        ):
            raise RuntimeError(f"{model_id} down")

        client = _make_client(side_effect=_side_effect)
        items = [{"prompt": "Q"}]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        pair = summary.results[0]
        assert pair.response_1 is None
        assert pair.response_2 is None
        assert pair.both_succeeded is False
        assert summary.model_1_failures == 1
        assert summary.model_2_failures == 1

    @pytest.mark.asyncio
    async def test_empty_prompt_marked_as_failure(self):
        client = _make_client()
        items = [{"prompt": ""}]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        pair = summary.results[0]
        assert pair.error_1 == "Empty prompt"
        assert pair.error_2 == "Empty prompt"
        assert pair.both_succeeded is False
        client.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_does_not_block_others(self):
        """One failing prompt should not prevent others."""
        call_idx = 0

        async def _side_effect(
            model_id, request, timeout_seconds=60.0
        ):
            nonlocal call_idx
            call_idx += 1
            if request.prompt == "bad":
                raise RuntimeError("bad prompt")
            return _resp(model_id=model_id)

        client = _make_client(side_effect=_side_effect)
        items = [
            {"prompt": "good1"},
            {"prompt": "bad"},
            {"prompt": "good2"},
        ]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        assert summary.total == 3
        assert summary.both_succeeded == 2
        assert summary.results[1].any_failed is True


class TestParallelInvokerConcurrency:
    """Concurrency control via semaphore."""

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def _side_effect(
            model_id, request, timeout_seconds=60.0
        ):
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            await asyncio.sleep(0.01)
            async with lock:
                current -= 1
            return _resp(model_id=model_id)

        client = _make_client(side_effect=_side_effect)
        cfg = ParallelInvokeConfig(
            model_id_1="model-a",
            model_id_2="model-b",
            concurrency=2,
        )
        items = [{"prompt": f"Q{i}"} for i in range(6)]
        await run_parallel_invocation(
            items=items, client=client, config=cfg
        )
        # With concurrency=2, at most 2 prompts in-flight,
        # each spawning 2 calls = max 4 concurrent invocations
        assert max_concurrent <= 4


class TestParallelInvokerProgressCallback:
    """Progress callback invocation."""

    @pytest.mark.asyncio
    async def test_callback_called_for_each_prompt(self):
        client = _make_client()
        calls = []

        def cb(done, total):
            calls.append((done, total))

        items = [{"prompt": "A"}, {"prompt": "B"}]
        await run_parallel_invocation(
            items=items,
            client=client,
            config=DEFAULT_CFG,
            progress_callback=cb,
        )
        assert len(calls) == 2
        totals = [c[1] for c in calls]
        assert all(t == 2 for t in totals)
        dones = sorted(c[0] for c in calls)
        assert dones == [1, 2]

    @pytest.mark.asyncio
    async def test_no_callback_is_fine(self):
        client = _make_client()
        items = [{"prompt": "X"}]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        assert summary.total == 1


class TestParallelInvokerLatency:
    """Latency tracking for both models."""

    @pytest.mark.asyncio
    async def test_latency_is_positive(self):
        client = _make_client()
        items = [{"prompt": "Q"}]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        pair = summary.results[0]
        assert pair.latency_ms_1 >= 0
        assert pair.latency_ms_2 >= 0

    @pytest.mark.asyncio
    async def test_failed_model_still_has_latency(self):
        async def _side_effect(
            model_id, request, timeout_seconds=60.0
        ):
            if model_id == "model-a":
                raise RuntimeError("fail")
            return _resp(model_id=model_id)

        client = _make_client(side_effect=_side_effect)
        items = [{"prompt": "Q"}]
        summary = await run_parallel_invocation(
            items=items, client=client, config=DEFAULT_CFG
        )
        pair = summary.results[0]
        assert pair.latency_ms_1 >= 0
        assert pair.latency_ms_2 >= 0


class TestParallelInvokerConfig:
    """Config parameters forwarded correctly."""

    @pytest.mark.asyncio
    async def test_custom_max_tokens_and_temperature(self):
        client = _make_client()
        cfg = ParallelInvokeConfig(
            model_id_1="model-a",
            model_id_2="model-b",
            max_tokens=256,
            temperature=0.1,
        )
        items = [{"prompt": "Q"}]
        await run_parallel_invocation(
            items=items, client=client, config=cfg
        )
        for call_args in client.invoke.call_args_list:
            _, kwargs = call_args
            req = kwargs["request"]
            assert req.max_tokens == 256
            assert req.temperature == 0.1


class TestPairedResponseProperties:
    """Test PairedResponse dataclass properties."""

    def test_both_succeeded_true(self):
        pr = PairedResponse(
            index=0,
            prompt="Q",
            model_1_id="a",
            model_2_id="b",
            response_1=_resp(),
            response_2=_resp(),
        )
        assert pr.both_succeeded is True
        assert pr.any_failed is False

    def test_both_succeeded_false_when_one_fails(self):
        pr = PairedResponse(
            index=0,
            prompt="Q",
            model_1_id="a",
            model_2_id="b",
            response_1=_resp(),
            error_2="fail",
        )
        assert pr.both_succeeded is False
        assert pr.any_failed is True

    def test_any_failed_both_errors(self):
        pr = PairedResponse(
            index=0,
            prompt="Q",
            model_1_id="a",
            model_2_id="b",
            error_1="e1",
            error_2="e2",
        )
        assert pr.both_succeeded is False
        assert pr.any_failed is True
