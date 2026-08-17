"""
Retry utilities with exponential backoff for AWS operations.
"""
import time
import random
import functools
from typing import Callable, Any, Tuple, Type
from botocore.exceptions import ClientError


# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_BACKOFF_MULTIPLIER = 2.0  # exponential backoff: 1s, 2s, 4s
DEFAULT_JITTER = True  # Add random jitter to prevent thundering herd


def retry_with_exponential_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    jitter: bool = DEFAULT_JITTER,
    retryable_exceptions: Tuple[Type[Exception], ...] = (ClientError,)
):
    """
    Decorator for retrying functions with exponential backoff and optional jitter.
    
    Jitter adds randomness to retry delays to prevent thundering herd problem
    when multiple clients retry simultaneously.
    
    Args:
        max_retries: Maximum number of retry attempts (default 3)
        initial_delay: Initial delay in seconds (default 1.0)
        backoff_multiplier: Multiplier for exponential backoff (default 2.0)
        jitter: Whether to add random jitter to delays (default True)
        retryable_exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_exponential_backoff(max_retries=3, jitter=True)
        def upload_to_s3(bucket, key, data):
            s3_client.put_object(Bucket=bucket, Key=key, Body=data)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # Check if this is a retryable error
                    if isinstance(e, ClientError):
                        error_code = e.response.get('Error', {}).get('Code', '')
                        
                        # Don't retry on non-retryable errors
                        if error_code in [
                            'NoSuchKey',
                            'NoSuchBucket',
                            'InvalidParameterValue',
                            'ValidationException'
                        ]:
                            raise
                    
                    # If not last attempt, wait before retrying
                    if attempt < max_retries - 1:
                        # Apply jitter if enabled: random value between 0 and delay.
                        # Backoff jitter is not a security control, so the
                        # standard PRNG is appropriate here.
                        actual_delay = delay * random.random() if jitter else delay  # nosec B311
                        time.sleep(actual_delay)
                        delay *= backoff_multiplier
            
            # All retries exhausted, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


def execute_with_retry(
    func: Callable,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    jitter: bool = DEFAULT_JITTER,
    retryable_exceptions: Tuple[Type[Exception], ...] = (ClientError,),
    *args,
    **kwargs
) -> Any:
    """
    Execute a function with retry logic and optional jitter.
    
    This is a functional alternative to the decorator for cases where
    you want to apply retry logic without decorating the function.
    
    Jitter adds randomness to retry delays to prevent thundering herd problem
    when multiple clients retry simultaneously.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts (default 3)
        initial_delay: Initial delay in seconds (default 1.0)
        backoff_multiplier: Multiplier for exponential backoff (default 2.0)
        jitter: Whether to add random jitter to delays (default True)
        retryable_exceptions: Tuple of exception types to retry on
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Result of func execution
        
    Raises:
        Exception: If all retries are exhausted
        
    Example:
        result = execute_with_retry(
            s3_client.put_object,
            max_retries=3,
            jitter=True,
            Bucket='my-bucket',
            Key='my-key',
            Body=data
        )
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
            
        except retryable_exceptions as e:
            last_exception = e
            
            # Check if this is a retryable error
            if isinstance(e, ClientError):
                error_code = e.response.get('Error', {}).get('Code', '')
                
                # Don't retry on non-retryable errors
                if error_code in [
                    'NoSuchKey',
                    'NoSuchBucket',
                    'InvalidParameterValue',
                    'ValidationException'
                ]:
                    raise
            
            # If not last attempt, wait before retrying
            if attempt < max_retries - 1:
                # Apply jitter if enabled: random value between 0 and delay.
                # Backoff jitter is not a security control, so the
                # standard PRNG is appropriate here.
                actual_delay = delay * random.random() if jitter else delay  # nosec B311
                time.sleep(actual_delay)
                delay *= backoff_multiplier
    
    # All retries exhausted, raise the last exception
    raise last_exception
