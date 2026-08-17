"""
Unit tests for retry utilities with exponential backoff.
"""
import pytest
import time
from unittest.mock import Mock
from botocore.exceptions import ClientError

from src.utils.retry_utils import (
    retry_with_exponential_backoff,
    execute_with_retry,
    DEFAULT_MAX_RETRIES,
    DEFAULT_INITIAL_DELAY,
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_JITTER
)


class TestRetryWithExponentialBackoff:
    """Test retry decorator with exponential backoff."""
    
    def test_successful_execution_no_retry(self):
        """Test that successful execution doesn't trigger retries."""
        mock_func = Mock(return_value="success")
        
        @retry_with_exponential_backoff()
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_on_client_error(self):
        """Test that ClientError triggers retry."""
        mock_func = Mock(side_effect=[
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            "success"
        ])
        
        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_exponential_backoff_delays(self):
        """Test that delays follow exponential backoff pattern."""
        mock_func = Mock(side_effect=[
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            "success"
        ])
        
        @retry_with_exponential_backoff(
            max_retries=3,
            initial_delay=0.1,
            backoff_multiplier=2.0,
            jitter=False  # Disable jitter for predictable timing
        )
        def test_func():
            return mock_func()
        
        start_time = time.time()
        result = test_func()
        elapsed_time = time.time() - start_time
        
        # Expected delays: 0.1s + 0.2s = 0.3s (with some tolerance)
        assert result == "success"
        assert elapsed_time >= 0.3
        assert elapsed_time < 0.5  # Allow some overhead
    
    def test_max_retries_exhausted(self):
        """Test that exception is raised when max retries are exhausted."""
        error = ClientError(
            {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
            'PutObject'
        )
        mock_func = Mock(side_effect=error)
        
        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with pytest.raises(ClientError):
            test_func()
        
        assert mock_func.call_count == 3
    
    def test_non_retryable_error_no_retry(self):
        """Test that non-retryable errors don't trigger retry."""
        error = ClientError(
            {'Error': {'Code': 'NoSuchKey', 'Message': 'Key not found'}},
            'GetObject'
        )
        mock_func = Mock(side_effect=error)
        
        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with pytest.raises(ClientError):
            test_func()
        
        # Should fail immediately without retry
        assert mock_func.call_count == 1
    
    def test_no_such_bucket_no_retry(self):
        """Test that NoSuchBucket error doesn't trigger retry."""
        error = ClientError(
            {'Error': {'Code': 'NoSuchBucket', 'Message': 'Bucket not found'}},
            'GetObject'
        )
        mock_func = Mock(side_effect=error)
        
        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with pytest.raises(ClientError):
            test_func()
        
        # Should fail immediately without retry
        assert mock_func.call_count == 1
    
    def test_validation_exception_no_retry(self):
        """Test that ValidationException doesn't trigger retry."""
        error = ClientError(
            {'Error': {'Code': 'ValidationException', 'Message': 'Invalid parameter'}},
            'PutItem'
        )
        mock_func = Mock(side_effect=error)
        
        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with pytest.raises(ClientError):
            test_func()
        
        # Should fail immediately without retry
        assert mock_func.call_count == 1
    
    def test_custom_retryable_exceptions(self):
        """Test retry with custom exception types."""
        mock_func = Mock(side_effect=[
            ValueError("Temporary error"),
            ValueError("Temporary error"),
            "success"
        ])
        
        @retry_with_exponential_backoff(
            max_retries=3,
            initial_delay=0.01,
            jitter=False,
            retryable_exceptions=(ValueError,)
        )
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_jitter_enabled(self):
        """Test that jitter adds randomness to delays."""
        mock_func = Mock(side_effect=[
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            "success"
        ])
        
        @retry_with_exponential_backoff(
            max_retries=3,
            initial_delay=0.1,
            backoff_multiplier=2.0,
            jitter=True
        )
        def test_func():
            return mock_func()
        
        start_time = time.time()
        result = test_func()
        elapsed_time = time.time() - start_time
        
        # With jitter, delays are random between 0 and max_delay
        # Expected max delays: 0.1s + 0.2s = 0.3s
        # With jitter, actual delays should be less than max
        assert result == "success"
        assert elapsed_time < 0.3  # Should be less than non-jittered time
        assert elapsed_time > 0.0  # But still some delay
    
    def test_jitter_disabled(self):
        """Test that jitter=False produces consistent delays."""
        mock_func = Mock(side_effect=[
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            "success"
        ])
        
        @retry_with_exponential_backoff(
            max_retries=2,
            initial_delay=0.1,
            backoff_multiplier=2.0,
            jitter=False
        )
        def test_func():
            return mock_func()
        
        start_time = time.time()
        result = test_func()
        elapsed_time = time.time() - start_time
        
        # Without jitter, delay should be exactly 0.1s (with small tolerance)
        assert result == "success"
        assert elapsed_time >= 0.1
        assert elapsed_time < 0.15  # Small tolerance for overhead


class TestExecuteWithRetry:
    """Test execute_with_retry function."""
    
    def test_successful_execution_no_retry(self):
        """Test that successful execution doesn't trigger retries."""
        mock_func = Mock(return_value="success")
        
        result = execute_with_retry(mock_func)
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_on_client_error(self):
        """Test that ClientError triggers retry."""
        mock_func = Mock(side_effect=[
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            "success"
        ])
        
        result = execute_with_retry(
            mock_func,
            max_retries=3,
            initial_delay=0.01
        )
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_with_function_arguments(self):
        """Test execute_with_retry with function arguments."""
        mock_func = Mock(return_value="success")
        
        result = execute_with_retry(
            mock_func,
            max_retries=3,
            initial_delay=0.01,
            arg1="value1",
            arg2="value2"
        )
        
        assert result == "success"
        mock_func.assert_called_once_with(arg1="value1", arg2="value2")
    
    def test_max_retries_exhausted(self):
        """Test that exception is raised when max retries are exhausted."""
        error = ClientError(
            {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
            'PutObject'
        )
        mock_func = Mock(side_effect=error)
        
        with pytest.raises(ClientError):
            execute_with_retry(
                mock_func,
                max_retries=3,
                initial_delay=0.01,
                jitter=False
            )
        
        assert mock_func.call_count == 3
    
    def test_jitter_enabled_execute(self):
        """Test that jitter works with execute_with_retry."""
        mock_func = Mock(side_effect=[
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            "success"
        ])
        
        start_time = time.time()
        result = execute_with_retry(
            mock_func,
            max_retries=2,
            initial_delay=0.1,
            backoff_multiplier=2.0,
            jitter=True
        )
        elapsed_time = time.time() - start_time
        
        # With jitter, delay should be random between 0 and 0.1s
        assert result == "success"
        assert elapsed_time < 0.15  # Should be less than max delay + overhead
        assert elapsed_time > 0.0  # But still some delay
    
    def test_jitter_disabled_execute(self):
        """Test that jitter=False works with execute_with_retry."""
        mock_func = Mock(side_effect=[
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            "success"
        ])
        
        start_time = time.time()
        result = execute_with_retry(
            mock_func,
            max_retries=2,
            initial_delay=0.1,
            backoff_multiplier=2.0,
            jitter=False
        )
        elapsed_time = time.time() - start_time
        
        # Without jitter, delay should be exactly 0.1s
        assert result == "success"
        assert elapsed_time >= 0.1
        assert elapsed_time < 0.15  # Small tolerance for overhead


class TestRetryConfiguration:
    """Test retry configuration parameters."""
    
    def test_default_max_retries(self):
        """Test that default max retries is 3."""
        assert DEFAULT_MAX_RETRIES == 3
    
    def test_default_initial_delay(self):
        """Test that default initial delay is 1.0 seconds."""
        assert DEFAULT_INITIAL_DELAY == 1.0
    
    def test_default_backoff_multiplier(self):
        """Test that default backoff multiplier is 2.0."""
        assert DEFAULT_BACKOFF_MULTIPLIER == 2.0
    
    def test_default_jitter(self):
        """Test that default jitter is enabled."""
        assert DEFAULT_JITTER is True
    
    def test_custom_max_retries(self):
        """Test retry with custom max retries."""
        error = ClientError(
            {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
            'PutObject'
        )
        mock_func = Mock(side_effect=error)
        
        @retry_with_exponential_backoff(max_retries=5, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with pytest.raises(ClientError):
            test_func()
        
        assert mock_func.call_count == 5
    
    def test_exponential_backoff_pattern(self):
        """Test that backoff follows exponential pattern (1s, 2s, 4s)."""
        mock_func = Mock(side_effect=[
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            ClientError(
                {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                'PutObject'
            ),
            "success"
        ])
        
        @retry_with_exponential_backoff(
            max_retries=4,
            initial_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False  # Disable jitter for predictable timing
        )
        def test_func():
            return mock_func()
        
        start_time = time.time()
        result = test_func()
        elapsed_time = time.time() - start_time
        
        # Expected delays: 1s + 2s + 4s = 7s (with some tolerance)
        assert result == "success"
        assert elapsed_time >= 7.0
        assert elapsed_time < 8.0  # Allow some overhead
