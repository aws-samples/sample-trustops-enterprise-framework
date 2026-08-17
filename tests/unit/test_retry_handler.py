"""
Unit tests for the retry handler.

Requirements: 8.6
"""

from src.orchestration.retry_handler import (
    RetryConfig,
    RetryResult,
    calculate_delay,
    execute_with_retry,
)


class TestCalculateDelay:
    def test_first_attempt_base_delay(self):
        config = RetryConfig(base_delay_seconds=1.0, jitter=False)
        delay = calculate_delay(0, config)
        assert delay == 1.0

    def test_exponential_backoff(self):
        config = RetryConfig(
            base_delay_seconds=1.0, backoff_multiplier=2.0, jitter=False
        )
        assert calculate_delay(0, config) == 1.0
        assert calculate_delay(1, config) == 2.0
        assert calculate_delay(2, config) == 4.0

    def test_max_delay_cap(self):
        config = RetryConfig(
            base_delay_seconds=10.0,
            backoff_multiplier=10.0,
            max_delay_seconds=30.0,
            jitter=False,
        )
        delay = calculate_delay(5, config)
        assert delay == 30.0

    def test_jitter_reduces_delay(self):
        config = RetryConfig(base_delay_seconds=10.0, jitter=True)
        delays = [calculate_delay(0, config) for _ in range(50)]
        # With jitter, delays should vary and be <= base
        assert min(delays) < max(delays)
        assert all(d <= 10.0 for d in delays)


class TestExecuteWithRetry:
    def test_success_on_first_try(self):
        result = execute_with_retry(lambda: 42)
        assert result.success is True
        assert result.result == 42
        assert result.total_retries == 0
        assert len(result.attempts) == 1

    def test_success_after_retries(self):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "ok"

        config = RetryConfig(max_retries=3, base_delay_seconds=0.01, jitter=False)
        result = execute_with_retry(flaky, config, sleep_func=lambda _: None)
        assert result.success is True
        assert result.result == "ok"
        assert result.total_retries == 2
        assert len(result.attempts) == 3

    def test_failure_after_max_retries(self):
        config = RetryConfig(max_retries=2, base_delay_seconds=0.01, jitter=False)
        result = execute_with_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            config,
            sleep_func=lambda _: None,
        )
        assert result.success is False
        assert result.total_retries == 2
        assert len(result.attempts) == 3  # initial + 2 retries

    def test_records_error_messages(self):
        config = RetryConfig(max_retries=1, base_delay_seconds=0.01, jitter=False)

        def fail():
            raise ValueError("bad input")

        result = execute_with_retry(fail, config, sleep_func=lambda _: None)
        assert result.success is False
        assert all("bad input" in a.error for a in result.attempts)

    def test_sleep_func_called(self):
        delays = []

        def record_sleep(d):
            delays.append(d)

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail")
            return "ok"

        config = RetryConfig(max_retries=3, base_delay_seconds=1.0, jitter=False)
        execute_with_retry(flaky, config, sleep_func=record_sleep)
        assert len(delays) == 1
        assert delays[0] == 1.0

    def test_zero_retries(self):
        config = RetryConfig(max_retries=0)

        def fail():
            raise RuntimeError("fail")

        result = execute_with_retry(fail, config, sleep_func=lambda _: None)
        assert result.success is False
        assert result.total_retries == 0
        assert len(result.attempts) == 1

    def test_default_config(self):
        result = execute_with_retry(lambda: "ok")
        assert result.success is True
