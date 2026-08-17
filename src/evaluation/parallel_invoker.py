"""
Parallel Model Invoker for comparative evaluation.

Sends identical prompts to two models concurrently and collects paired
responses for side-by-side comparison. Supports configurable concurrency
and graceful error handling per prompt.

Requirements: 7.1, 7.5
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.clients.inference_client import InferenceClient
from src.data_models.model import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)


@dataclass
class PairedResponse:
    """Paired responses from two models for a single prompt.

    Attributes:
        index: Position of the prompt in the dataset.
        prompt: The prompt sent to both models.
        model_1_id: ID of the first model.
        model_2_id: ID of the second model.
        response_1: Response from model 1 (None on failure).
        response_2: Response from model 2 (None on failure).
        error_1: Error message if model 1 failed.
        error_2: Error message if model 2 failed.
        latency_ms_1: Latency for model 1 in milliseconds.
        latency_ms_2: Latency for model 2 in milliseconds.
        category: Optional category from the dataset item.
        expected_response: Optional expected response from dataset.
    """

    index: int
    prompt: str
    model_1_id: str
    model_2_id: str
    response_1: Optional[InferenceResponse] = None
    response_2: Optional[InferenceResponse] = None
    error_1: Optional[str] = None
    error_2: Optional[str] = None
    latency_ms_1: float = 0.0
    latency_ms_2: float = 0.0
    category: Optional[str] = None
    expected_response: Optional[str] = None

    @property
    def both_succeeded(self) -> bool:
        """Return True if both models responded successfully."""
        return self.response_1 is not None and self.response_2 is not None

    @property
    def any_failed(self) -> bool:
        """Return True if either model failed."""
        return self.error_1 is not None or self.error_2 is not None


@dataclass
class ParallelInvokeConfig:
    """Configuration for parallel model invocation.

    Attributes:
        model_id_1: ID of the first model (e.g. baseline).
        model_id_2: ID of the second model (e.g. fine-tuned).
        concurrency: Max number of prompts processed concurrently.
            Each prompt invokes both models, so actual concurrent
            requests can be up to 2 * concurrency.
        max_tokens: Max tokens per request.
        temperature: Sampling temperature.
        timeout_seconds: Timeout per individual model invocation.
    """

    model_id_1: str
    model_id_2: str
    concurrency: int = 5
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout_seconds: float = 60.0


@dataclass
class ParallelInvokeSummary:
    """Summary of a parallel invocation run.

    Attributes:
        total: Total number of prompts processed.
        both_succeeded: Number of prompts where both models succeeded.
        model_1_failures: Number of prompts where model 1 failed.
        model_2_failures: Number of prompts where model 2 failed.
        results: Individual paired results.
    """

    total: int = 0
    both_succeeded: int = 0
    model_1_failures: int = 0
    model_2_failures: int = 0
    results: list[PairedResponse] = field(default_factory=list)


# Type alias for progress callback: (completed, total) -> None
ProgressCallback = Callable[[int, int], None]


async def _invoke_model(
    client: InferenceClient,
    model_id: str,
    request: InferenceRequest,
    timeout_seconds: float,
) -> tuple[Optional[InferenceResponse], Optional[str], float]:
    """Invoke a single model, returning (response, error, latency_ms).

    Args:
        client: InferenceClient for model invocation.
        model_id: ID of the model to invoke.
        request: The inference request.
        timeout_seconds: Timeout for this invocation.

    Returns:
        Tuple of (response_or_None, error_or_None, latency_ms).
    """
    start = time.monotonic()
    try:
        response = await asyncio.wait_for(
            client.invoke(
                model_id=model_id,
                request=request,
                timeout_seconds=timeout_seconds,
            ),
            timeout=timeout_seconds,
        )
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return response, None, elapsed_ms
    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        error = f"Timeout after {timeout_seconds:.0f}s"
        logger.warning(
            "Model %s timed out for prompt (%.1fs)",
            model_id,
            timeout_seconds,
        )
        return None, error, elapsed_ms
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        error = str(exc)
        logger.warning("Model %s failed: %s", model_id, error)
        return None, error, elapsed_ms


async def run_parallel_invocation(
    items: list[dict[str, Any]],
    client: InferenceClient,
    config: ParallelInvokeConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> ParallelInvokeSummary:
    """Send identical prompts to two models concurrently.

    Collects paired responses for side-by-side comparison.
    For each prompt in *items*, both models are invoked
    simultaneously. A semaphore limits how many prompts are
    in-flight at once (each prompt spawns two concurrent
    model calls).

    Args:
        items: Dataset items; each dict must contain a ``prompt`` key.
            Optional keys: ``expected_response``, ``category``.
        client: InferenceClient for model invocations.
        config: Parallel invocation configuration.
        progress_callback: Optional callback invoked as
            ``callback(completed_count, total_count)`` after each prompt pair.

    Returns:
        ParallelInvokeSummary with all paired results.
    """
    if not items:
        return ParallelInvokeSummary()

    summary = ParallelInvokeSummary(total=len(items))
    semaphore = asyncio.Semaphore(config.concurrency)
    completed = 0
    lock = asyncio.Lock()

    async def _process_prompt(
        index: int, item: dict[str, Any]
    ) -> PairedResponse:
        nonlocal completed

        prompt = item.get("prompt", "")
        category = item.get("category")
        expected = item.get("expected_response")

        if not prompt:
            result = PairedResponse(
                index=index,
                prompt=prompt,
                model_1_id=config.model_id_1,
                model_2_id=config.model_id_2,
                error_1="Empty prompt",
                error_2="Empty prompt",
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
            # Invoke both models concurrently for this prompt
            resp_1_task = _invoke_model(
                client, config.model_id_1, request, config.timeout_seconds
            )
            resp_2_task = _invoke_model(
                client, config.model_id_2, request, config.timeout_seconds
            )
            results_pair = await asyncio.gather(
                resp_1_task, resp_2_task
            )
            (resp_1, err_1, lat_1) = results_pair[0]
            (resp_2, err_2, lat_2) = results_pair[1]

        result = PairedResponse(
            index=index,
            prompt=prompt,
            model_1_id=config.model_id_1,
            model_2_id=config.model_id_2,
            response_1=resp_1,
            response_2=resp_2,
            error_1=err_1,
            error_2=err_2,
            latency_ms_1=lat_1,
            latency_ms_2=lat_2,
            category=category,
            expected_response=expected,
        )

        async with lock:
            completed += 1
            if progress_callback:
                progress_callback(completed, len(items))

        return result

    tasks = [_process_prompt(i, item) for i, item in enumerate(items)]
    results = await asyncio.gather(*tasks)

    # Sort by original index to preserve order
    sorted_results = sorted(results, key=lambda r: r.index)
    summary.results = sorted_results
    summary.both_succeeded = sum(1 for r in sorted_results if r.both_succeeded)
    summary.model_1_failures = sum(
        1 for r in sorted_results if r.error_1 is not None
    )
    summary.model_2_failures = sum(
        1 for r in sorted_results if r.error_2 is not None
    )

    return summary
