"""
Batch Inference Runner.

Iterates through a dataset, invokes a model via InferenceClient,
and collects responses with timing information. Supports configurable
concurrency to respect provider rate limits.

Requirements: 3.3, 3.11, 3.12, 3.14, 3.15, 3.18, 3.19
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.clients.inference_client import InferenceClient
from src.data_models.model import InferenceRequest
from src.evaluation.timeout_config import TimeoutConfig, resolve_timeout

logger = logging.getLogger(__name__)


@dataclass
class BatchItemResult:
    """Result for a single dataset item processed by the batch runner.

    Attributes:
        index: Position of the item in the dataset.
        prompt: The prompt sent to the model.
        response_text: Generated text (empty string on failure).
        latency_ms: Inference latency in milliseconds.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        success: Whether inference succeeded.
        error: Error message if inference failed.
        category: Optional category from the dataset item.
        expected_response: Optional expected response from dataset.
    """

    index: int
    prompt: str
    response_text: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    category: Optional[str] = None
    expected_response: Optional[str] = None


@dataclass
class BatchRunConfig:
    """Configuration for a batch inference run.

    Attributes:
        model_id: ID of the model to invoke.
        concurrency: Max concurrent requests (default 1 = sequential).
        max_tokens: Max tokens per request.
        temperature: Sampling temperature.
        timeout_seconds: Timeout per request in seconds.
            When set, this value takes precedence over timeout_config.
        timeout_config: Per-model-size timeout configuration.
            Used to resolve timeout when timeout_seconds is not explicitly
            changed from the default.
    """

    model_id: str
    concurrency: int = 1
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout_seconds: float = 60.0
    timeout_config: Optional[TimeoutConfig] = None


@dataclass
class BatchRunSummary:
    """Summary of a completed batch run.

    Attributes:
        total: Total items processed.
        succeeded: Number of successful inferences.
        failed: Number of failed inferences.
        results: Individual item results.
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[BatchItemResult] = field(default_factory=list)


# Type alias for progress callback: (completed, total) -> None
ProgressCallback = Callable[[int, int], None]


async def run_batch_inference(
    items: list[dict[str, Any]],
    client: InferenceClient,
    config: BatchRunConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> BatchRunSummary:
    """Run batch inference over a list of dataset items.

    Each item dict should contain at minimum a ``prompt`` key.
    Optional keys: ``expected_response``, ``category``.

    Args:
        items: Dataset items to process.
        client: InferenceClient for model invocations.
        config: Batch run configuration.
        progress_callback: Optional callback invoked as
            ``callback(completed_count, total_count)`` after each item.

    Returns:
        BatchRunSummary with all individual results.
    """
    if not items:
        return BatchRunSummary()

    # Resolve effective timeout: use timeout_config if available,
    # otherwise fall back to timeout_seconds.
    if config.timeout_config is not None:
        effective_timeout = resolve_timeout(
            config.model_id, config.timeout_config
        )
    else:
        effective_timeout = config.timeout_seconds

    summary = BatchRunSummary(total=len(items))
    semaphore = asyncio.Semaphore(config.concurrency)
    completed = 0
    lock = asyncio.Lock()

    async def _process_item(
        index: int, item: dict[str, Any]
    ) -> BatchItemResult:
        nonlocal completed

        prompt = item.get("prompt", "")
        category = item.get("category")
        expected = item.get("expected_response")

        if not prompt:
            result = BatchItemResult(
                index=index,
                prompt=prompt,
                success=False,
                error="Empty prompt",
                category=category,
                expected_response=expected,
            )
            async with lock:
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(items))
            return result

        request = InferenceRequest(
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        async with semaphore:
            start = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    client.invoke(
                        model_id=config.model_id,
                        request=request,
                        timeout_seconds=effective_timeout,
                    ),
                    timeout=effective_timeout,
                )
                elapsed_ms = (time.monotonic() - start) * 1000.0

                result = BatchItemResult(
                    index=index,
                    prompt=prompt,
                    response_text=response.text,
                    latency_ms=elapsed_ms,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    success=True,
                    category=category,
                    expected_response=expected,
                )
            except asyncio.TimeoutError:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                logger.warning(
                    "Batch item %d timed out after %.1fs",
                    index,
                    effective_timeout,
                )
                result = BatchItemResult(
                    index=index,
                    prompt=prompt,
                    latency_ms=elapsed_ms,
                    success=False,
                    error=f"Timeout after {effective_timeout:.0f}s",
                    category=category,
                    expected_response=expected,
                )
            except Exception as exc:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                logger.warning(
                    "Batch item %d failed: %s", index, exc
                )
                result = BatchItemResult(
                    index=index,
                    prompt=prompt,
                    latency_ms=elapsed_ms,
                    success=False,
                    error=str(exc),
                    category=category,
                    expected_response=expected,
                )

        async with lock:
            completed += 1
            if progress_callback:
                progress_callback(completed, len(items))

        return result

    tasks = [
        _process_item(i, item) for i, item in enumerate(items)
    ]
    results = await asyncio.gather(*tasks)

    # Sort by original index to preserve order
    sorted_results = sorted(results, key=lambda r: r.index)
    summary.results = sorted_results
    summary.succeeded = sum(1 for r in sorted_results if r.success)
    summary.failed = sum(1 for r in sorted_results if not r.success)

    return summary
