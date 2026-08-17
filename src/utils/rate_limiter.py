"""
Rate limiter with token bucket algorithm for API request throttling.

This module provides per-provider rate limiting to ensure compliance with
API rate limits and prevent throttling errors. It supports both request-based
and token-based rate limiting.
"""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.
    
    Attributes:
        requests_per_minute: Maximum number of requests allowed per minute
        tokens_per_minute: Maximum number of tokens allowed per minute
        burst_size: Maximum burst size for token bucket (defaults to requests_per_minute)
    """
    requests_per_minute: int
    tokens_per_minute: Optional[int] = None
    burst_size: Optional[int] = None
    
    def __post_init__(self):
        """Set default burst size if not provided."""
        if self.burst_size is None:
            self.burst_size = self.requests_per_minute


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.
    
    Implements the token bucket algorithm for smooth rate limiting with burst support.
    
    Attributes:
        capacity: Maximum number of tokens the bucket can hold
        refill_rate: Number of tokens added per second
        tokens: Current number of tokens in the bucket
        last_refill: Timestamp of last refill operation
    """
    capacity: float
    refill_rate: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize bucket to full capacity."""
        self.tokens = self.capacity
        self.last_refill = time.time()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time and refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: float = 1.0) -> bool:
        """Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_time(self, tokens: float = 1.0) -> float:
        """Calculate wait time until tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Wait time in seconds (0 if tokens are available now)
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0.0
            
            # Calculate how long until we have enough tokens
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.refill_rate


class RateLimiter:
    """Per-provider rate limiter using token bucket algorithm.
    
    This class manages rate limits for multiple providers, ensuring that
    API calls stay within configured limits for both requests per minute
    and tokens per minute.
    
    Example:
        >>> config = RateLimitConfig(requests_per_minute=60, tokens_per_minute=100000)
        >>> limiter = RateLimiter()
        >>> limiter.set_limit("bedrock", config)
        >>> 
        >>> # Acquire permission before making request
        >>> await limiter.acquire("bedrock", tokens=500)
        >>> # Make API call...
    """
    
    def __init__(self):
        """Initialize rate limiter with empty provider configurations."""
        self._request_buckets: Dict[str, TokenBucket] = {}
        self._token_buckets: Dict[str, TokenBucket] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        self._lock = Lock()
    
    def set_limit(self, provider: str, config: RateLimitConfig) -> None:
        """Set rate limit configuration for a provider.
        
        Args:
            provider: Provider identifier (e.g., "bedrock", "sagemaker")
            config: Rate limit configuration
        """
        with self._lock:
            self._configs[provider] = config
            
            # Create request bucket
            # Convert requests/minute to requests/second for refill rate
            request_refill_rate = config.requests_per_minute / 60.0
            self._request_buckets[provider] = TokenBucket(
                capacity=float(config.burst_size),
                refill_rate=request_refill_rate
            )
            
            # Create token bucket if token limit is specified
            if config.tokens_per_minute is not None:
                token_refill_rate = config.tokens_per_minute / 60.0
                # Use tokens_per_minute as burst capacity for tokens
                token_burst = float(config.tokens_per_minute)
                self._token_buckets[provider] = TokenBucket(
                    capacity=token_burst,
                    refill_rate=token_refill_rate
                )
    
    def get_config(self, provider: str) -> Optional[RateLimitConfig]:
        """Get rate limit configuration for a provider.
        
        Args:
            provider: Provider identifier
            
        Returns:
            Rate limit configuration or None if not configured
        """
        return self._configs.get(provider)
    
    def remove_limit(self, provider: str) -> None:
        """Remove rate limit configuration for a provider.
        
        Args:
            provider: Provider identifier
        """
        with self._lock:
            self._configs.pop(provider, None)
            self._request_buckets.pop(provider, None)
            self._token_buckets.pop(provider, None)
    
    async def acquire(
        self,
        provider: str,
        tokens: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> bool:
        """Acquire permission to make a request.
        
        This method will block until rate limit allows the request, or until
        timeout is reached.
        
        Args:
            provider: Provider identifier
            tokens: Number of tokens for this request (for token-based limiting)
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Returns:
            True if permission acquired, False if timeout reached
            
        Raises:
            ValueError: If provider is not configured
        """
        if provider not in self._configs:
            raise ValueError(f"Rate limit not configured for provider: {provider}")
        
        start_time = time.time()
        
        while True:
            # Check request bucket
            request_bucket = self._request_buckets[provider]
            request_wait = request_bucket.wait_time(1.0)
            
            # Check token bucket if applicable
            token_wait = 0.0
            if tokens is not None and provider in self._token_buckets:
                token_bucket = self._token_buckets[provider]
                token_wait = token_bucket.wait_time(float(tokens))
            
            # Calculate total wait time
            wait_time = max(request_wait, token_wait)
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)
            
            # If no wait needed, try to consume tokens
            if wait_time == 0:
                request_consumed = request_bucket.consume(1.0)
                token_consumed = True
                
                if tokens is not None and provider in self._token_buckets:
                    token_bucket = self._token_buckets[provider]
                    token_consumed = token_bucket.consume(float(tokens))
                
                if request_consumed and token_consumed:
                    return True
                
                # If consumption failed, wait a bit and retry
                await asyncio.sleep(0.01)
            else:
                # Wait for tokens to be available
                await asyncio.sleep(wait_time)
    
    def try_acquire(
        self,
        provider: str,
        tokens: Optional[int] = None
    ) -> bool:
        """Try to acquire permission without blocking.
        
        Args:
            provider: Provider identifier
            tokens: Number of tokens for this request
            
        Returns:
            True if permission acquired, False otherwise
            
        Raises:
            ValueError: If provider is not configured
        """
        if provider not in self._configs:
            raise ValueError(f"Rate limit not configured for provider: {provider}")
        
        # Try to consume from request bucket
        request_bucket = self._request_buckets[provider]
        if not request_bucket.consume(1.0):
            return False
        
        # Try to consume from token bucket if applicable
        if tokens is not None and provider in self._token_buckets:
            token_bucket = self._token_buckets[provider]
            if not token_bucket.consume(float(tokens)):
                # Refund the request token since we couldn't get token quota
                request_bucket.tokens = min(
                    request_bucket.capacity,
                    request_bucket.tokens + 1.0
                )
                return False
        
        return True
    
    def get_available_capacity(self, provider: str) -> Dict[str, float]:
        """Get current available capacity for a provider.
        
        Args:
            provider: Provider identifier
            
        Returns:
            Dictionary with 'requests' and optionally 'tokens' capacity
            
        Raises:
            ValueError: If provider is not configured
        """
        if provider not in self._configs:
            raise ValueError(f"Rate limit not configured for provider: {provider}")
        
        result = {}
        
        # Get request capacity
        request_bucket = self._request_buckets[provider]
        with request_bucket._lock:
            request_bucket._refill()
            result['requests'] = request_bucket.tokens
        
        # Get token capacity if applicable
        if provider in self._token_buckets:
            token_bucket = self._token_buckets[provider]
            with token_bucket._lock:
                token_bucket._refill()
                result['tokens'] = token_bucket.tokens
        
        return result
