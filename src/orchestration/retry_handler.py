"""
Retry handler with exponential backoff for transient failures.

Supports configurable max retries, base delay, and backoff multiplier.

Requirements: 8.6
"""

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay_seconds: Initial delay before first retry.
        backoff_multiplier: Multiplier applied to delay after each retry.
        max_delay_seconds: Maximum delay between retries.
        jitter: Whether to add random jitter to delays.
    """

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 60.0
    jitter: bool = True


@dataclass
class RetryAttempt:
    """Record of a single retry attempt.

    Attributes:
        attempt_number: The attempt number (0 = first try).
        timestamp: When the attempt was made.
        success: Whether the attempt succeeded.
        error: Error message if the attempt failed.
        delay_seconds: Delay before this attempt (0 for first).
    """

    attempt_number: int
    timestamp: datetime
    success: bool
    error: str = ""
    delay_seconds: float = 0.0


@dataclass
class RetryResult:
    """Result of a retry operation.

    Attributes:
        success: Whether the operation eventually succeeded.
        result: The return value if successful.
        attempts: List of all retry attempts.
        total_retries: Number of retries performed (excludes initial attempt).
    """

    success: bool
    result: object = None
    attempts: list[RetryAttempt] = field(default_factory=list)
    total_retries: int = 0


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate the delay before the next retry attempt.

    Uses exponential backoff with optional jitter.

    Args:
        attempt: The current attempt number (0-based).
        config: Retry configuration.

    Returns:
        Delay in seconds.
    """
    delay = config.base_delay_seconds * (config.backoff_multiplier ** attempt)
    delay = min(delay, config.max_delay_seconds)
    if config.jitter:
        # Retry backoff jitter - not a security control, so the standard
        # PRNG is appropriate here.
        delay = delay * (0.5 + random.random() * 0.5)  # nosec B311
    return delay


def execute_with_retry(
    func: Callable[[], T],
    config: RetryConfig | None = None,
    sleep_func: Callable[[float], None] | None = None,
) -> RetryResult:
    """Execute a function with retry logic.

    Retries the function on failure using exponential backoff.

    Args:
        func: The function to execute (no arguments).
        config: Retry configuration. Uses defaults if not provided.
        sleep_func: Optional sleep function for testing. Defaults to time.sleep.

    Returns:
        RetryResult with success status and all attempt records.
    """
    config = config or RetryConfig()
    sleep_fn = sleep_func or time.sleep
    attempts: list[RetryAttempt] = []

    for attempt_num in range(config.max_retries + 1):
        delay = 0.0
        if attempt_num > 0:
            delay = calculate_delay(attempt_num - 1, config)
            sleep_fn(delay)

        try:
            result = func()
            attempts.append(
                RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.now(timezone.utc),
                    success=True,
                    delay_seconds=delay,
                )
            )
            return RetryResult(
                success=True,
                result=result,
                attempts=attempts,
                total_retries=attempt_num,
            )
        except Exception as exc:
            attempts.append(
                RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=datetime.now(timezone.utc),
                    success=False,
                    error=str(exc),
                    delay_seconds=delay,
                )
            )

    return RetryResult(
        success=False,
        attempts=attempts,
        total_retries=config.max_retries,
    )
