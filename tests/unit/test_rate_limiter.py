"""
Unit tests for RateLimiter class.
"""

import pytest
import asyncio
import time
from src.utils.rate_limiter import RateLimiter, RateLimitConfig, TokenBucket


class TestTokenBucket:
    """Tests for TokenBucket class."""
    
    def test_initialization(self):
        """Test token bucket initializes with full capacity."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        assert bucket.tokens == 10.0
        assert bucket.capacity == 10.0
        assert bucket.refill_rate == 1.0
    
    def test_consume_success(self):
        """Test successful token consumption."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        assert bucket.consume(5.0) is True
        assert bucket.tokens == 5.0
    
    def test_consume_failure(self):
        """Test failed token consumption when insufficient tokens."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        bucket.consume(8.0)
        assert bucket.consume(5.0) is False
        # Tokens not consumed on failure (allow small refill from timing)
        assert bucket.tokens >= 2.0
        assert bucket.tokens < 2.1
    
    def test_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10.0, refill_rate=2.0)  # 2 tokens/second
        bucket.consume(8.0)  # 2 tokens left
        
        time.sleep(1.1)  # Wait for refill
        bucket._refill()
        
        # Should have refilled ~2.2 tokens, so ~4.2 total
        assert bucket.tokens >= 4.0
        assert bucket.tokens <= 5.0
    
    def test_refill_cap(self):
        """Test refill doesn't exceed capacity."""
        bucket = TokenBucket(capacity=10.0, refill_rate=5.0)
        bucket.consume(5.0)
        
        time.sleep(3.0)  # Would refill 15 tokens if uncapped
        bucket._refill()
        
        assert bucket.tokens == 10.0  # Capped at capacity
    
    def test_wait_time_available(self):
        """Test wait time is zero when tokens are available."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        assert bucket.wait_time(5.0) == 0.0
    
    def test_wait_time_unavailable(self):
        """Test wait time calculation when tokens unavailable."""
        bucket = TokenBucket(capacity=10.0, refill_rate=2.0)
        bucket.consume(9.0)  # 1 token left
        
        wait = bucket.wait_time(5.0)  # Need 4 more tokens
        expected_wait = 4.0 / 2.0  # 2 seconds at 2 tokens/sec
        assert abs(wait - expected_wait) < 0.1


class TestRateLimitConfig:
    """Tests for RateLimitConfig class."""
    
    def test_default_burst_size(self):
        """Test burst size defaults to requests_per_minute."""
        config = RateLimitConfig(requests_per_minute=60)
        assert config.burst_size == 60
    
    def test_custom_burst_size(self):
        """Test custom burst size is preserved."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=100)
        assert config.burst_size == 100
    
    def test_with_token_limit(self):
        """Test configuration with token limit."""
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=100000
        )
        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 100000


class TestRateLimiter:
    """Tests for RateLimiter class."""
    
    def test_initialization(self):
        """Test rate limiter initializes empty."""
        limiter = RateLimiter()
        assert len(limiter._configs) == 0
        assert len(limiter._request_buckets) == 0
        assert len(limiter._token_buckets) == 0
    
    def test_set_limit(self):
        """Test setting rate limit for a provider."""
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_minute=60)
        
        limiter.set_limit("test_provider", config)
        
        assert "test_provider" in limiter._configs
        assert "test_provider" in limiter._request_buckets
        assert limiter._configs["test_provider"] == config
    
    def test_set_limit_with_tokens(self):
        """Test setting rate limit with token limit."""
        limiter = RateLimiter()
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=100000
        )
        
        limiter.set_limit("test_provider", config)
        
        assert "test_provider" in limiter._token_buckets
    
    def test_get_config(self):
        """Test retrieving rate limit configuration."""
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_minute=60)
        limiter.set_limit("test_provider", config)
        
        retrieved = limiter.get_config("test_provider")
        assert retrieved == config
    
    def test_get_config_not_found(self):
        """Test retrieving config for non-existent provider."""
        limiter = RateLimiter()
        assert limiter.get_config("nonexistent") is None
    
    def test_remove_limit(self):
        """Test removing rate limit configuration."""
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_minute=60)
        limiter.set_limit("test_provider", config)
        
        limiter.remove_limit("test_provider")
        
        assert "test_provider" not in limiter._configs
        assert "test_provider" not in limiter._request_buckets
    
    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Test successful request acquisition."""
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_minute=60)
        limiter.set_limit("test_provider", config)
        
        result = await limiter.acquire("test_provider")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_acquire_with_tokens(self):
        """Test acquisition with token count."""
        limiter = RateLimiter()
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=1000
        )
        limiter.set_limit("test_provider", config)
        
        result = await limiter.acquire("test_provider", tokens=100)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_acquire_unconfigured_provider(self):
        """Test acquisition fails for unconfigured provider."""
        limiter = RateLimiter()
        
        with pytest.raises(ValueError, match="not configured"):
            await limiter.acquire("unconfigured")
    
    @pytest.mark.asyncio
    async def test_acquire_rate_limiting(self):
        """Test that rate limiting actually limits requests."""
        limiter = RateLimiter()
        # Very low limit: 6 requests per minute = 0.1 requests/second
        config = RateLimitConfig(requests_per_minute=6, burst_size=2)
        limiter.set_limit("test_provider", config)
        
        # First 2 should succeed immediately (burst)
        result1 = await limiter.acquire("test_provider")
        result2 = await limiter.acquire("test_provider")
        assert result1 is True
        assert result2 is True
        
        # Third should require waiting
        start = time.time()
        result3 = await limiter.acquire("test_provider", timeout=15.0)
        elapsed = time.time() - start
        
        assert result3 is True
        assert elapsed > 0.5  # Should have waited
    
    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test acquisition timeout."""
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_minute=6, burst_size=1)
        limiter.set_limit("test_provider", config)
        
        # Consume the burst
        await limiter.acquire("test_provider")
        
        # Try to acquire with short timeout
        result = await limiter.acquire("test_provider", timeout=0.1)
        assert result is False
    
    def test_try_acquire_success(self):
        """Test non-blocking acquisition success."""
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_minute=60)
        limiter.set_limit("test_provider", config)
        
        result = limiter.try_acquire("test_provider")
        assert result is True
    
    def test_try_acquire_failure(self):
        """Test non-blocking acquisition failure."""
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_minute=60, burst_size=2)
        limiter.set_limit("test_provider", config)
        
        # Consume all burst tokens
        limiter.try_acquire("test_provider")
        limiter.try_acquire("test_provider")
        
        # Should fail without waiting
        result = limiter.try_acquire("test_provider")
        assert result is False
    
    def test_try_acquire_with_tokens(self):
        """Test non-blocking acquisition with token limit."""
        limiter = RateLimiter()
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=1000
        )
        limiter.set_limit("test_provider", config)
        
        result = limiter.try_acquire("test_provider", tokens=100)
        assert result is True
    
    def test_try_acquire_insufficient_tokens(self):
        """Test non-blocking acquisition fails with insufficient tokens."""
        limiter = RateLimiter()
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=100
        )
        limiter.set_limit("test_provider", config)
        
        # Consume most tokens (leave only 5)
        limiter.try_acquire("test_provider", tokens=95)
        
        # Should fail due to insufficient tokens
        result = limiter.try_acquire("test_provider", tokens=50)
        assert result is False
    
    def test_try_acquire_refunds_on_token_failure(self):
        """Test request token is refunded if token limit fails."""
        limiter = RateLimiter()
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=100
        )
        limiter.set_limit("test_provider", config)
        
        # Consume most tokens (leave only 5)
        limiter.try_acquire("test_provider", tokens=95)
        
        # Get initial request capacity
        capacity_before = limiter.get_available_capacity("test_provider")
        
        # Try to acquire with insufficient tokens
        result = limiter.try_acquire("test_provider", tokens=50)
        assert result is False
        
        # Request capacity should be refunded (within small tolerance for timing)
        capacity_after = limiter.get_available_capacity("test_provider")
        assert abs(capacity_after['requests'] - capacity_before['requests']) < 0.1
    
    def test_get_available_capacity(self):
        """Test getting available capacity."""
        limiter = RateLimiter()
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=1000
        )
        limiter.set_limit("test_provider", config)
        
        capacity = limiter.get_available_capacity("test_provider")
        
        assert 'requests' in capacity
        assert 'tokens' in capacity
        assert capacity['requests'] > 0
        assert capacity['tokens'] > 0
    
    def test_get_available_capacity_unconfigured(self):
        """Test getting capacity for unconfigured provider."""
        limiter = RateLimiter()
        
        with pytest.raises(ValueError, match="not configured"):
            limiter.get_available_capacity("unconfigured")
    
    @pytest.mark.asyncio
    async def test_multiple_providers(self):
        """Test rate limiting with multiple providers."""
        limiter = RateLimiter()
        
        config1 = RateLimitConfig(requests_per_minute=60)
        config2 = RateLimitConfig(requests_per_minute=120)
        
        limiter.set_limit("provider1", config1)
        limiter.set_limit("provider2", config2)
        
        result1 = await limiter.acquire("provider1")
        result2 = await limiter.acquire("provider2")
        
        assert result1 is True
        assert result2 is True
    
    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self):
        """Test concurrent acquisitions are properly rate limited."""
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_minute=120, burst_size=10)
        limiter.set_limit("test_provider", config)
        
        # Launch 10 concurrent acquisitions
        tasks = [
            limiter.acquire("test_provider", timeout=10.0)
            for _ in range(10)
        ]
        
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        # All should succeed
        assert all(results)
        
        # Should complete quickly since burst allows all 10
        assert elapsed < 5.0


class TestRateLimiterIntegration:
    """Integration tests for RateLimiter."""
    
    @pytest.mark.asyncio
    async def test_realistic_bedrock_scenario(self):
        """Test realistic Bedrock rate limiting scenario."""
        limiter = RateLimiter()
        
        # Bedrock typical limits: 100 requests/minute, 200k tokens/minute
        config = RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=200000
        )
        limiter.set_limit("bedrock", config)
        
        # Simulate 10 requests with varying token counts
        token_counts = [1000, 2000, 1500, 3000, 2500, 1000, 1500, 2000, 1000, 1500]
        
        for tokens in token_counts:
            result = await limiter.acquire("bedrock", tokens=tokens, timeout=5.0)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_realistic_openai_scenario(self):
        """Test realistic OpenAI rate limiting scenario."""
        limiter = RateLimiter()
        
        # OpenAI typical limits: 3500 requests/minute, 90k tokens/minute
        config = RateLimitConfig(
            requests_per_minute=3500,
            tokens_per_minute=90000
        )
        limiter.set_limit("openai", config)
        
        # Simulate rapid requests
        tasks = [
            limiter.acquire("openai", tokens=500, timeout=2.0)
            for _ in range(20)
        ]
        
        results = await asyncio.gather(*tasks)
        assert all(results)
