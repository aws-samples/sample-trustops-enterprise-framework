# Discovery Scheduler

The `DiscoveryScheduler` provides periodic model discovery to keep the model registry up-to-date with available models from all registered providers.

## Overview

The Discovery Scheduler automatically refreshes the model registry at configurable intervals, ensuring that:
- New models are discovered and registered
- Existing model metadata is updated
- The registry reflects the current state of all providers

**Requirements:** 1.8

## Features

- **Periodic Discovery**: Automatically discovers models at configurable intervals
- **Manual Triggering**: Supports on-demand discovery outside the schedule
- **Notification Callbacks**: Optional callbacks for discovery completion
- **Error Resilience**: Continues operating even if individual discovery cycles fail
- **Statistics Tracking**: Monitors discovery cycles, failures, and model counts
- **Graceful Lifecycle**: Clean start/stop with proper resource cleanup

## Usage

### Basic Usage

```python
from src.registry.model_registry import ModelRegistry
from src.registry.discovery_scheduler import DiscoveryScheduler
from src.adapters.bedrock_adapter import BedrockAdapter

# Initialize registry and register adapters
registry = ModelRegistry()
bedrock_adapter = BedrockAdapter(region="us-east-1")
registry.register_adapter(bedrock_adapter)

# Create scheduler with 1-hour refresh interval (default)
scheduler = DiscoveryScheduler(registry)

# Start periodic discovery
await scheduler.start()

# ... scheduler runs in background ...

# Stop when done
await scheduler.stop()
```

### Custom Refresh Interval

```python
# Refresh every 30 minutes
scheduler = DiscoveryScheduler(
    registry=registry,
    refresh_interval_seconds=1800
)
```

### With Notification Callback

```python
async def on_discovery_complete(models: list[ModelMetadata]):
    """Called after each discovery cycle."""
    print(f"Discovered {len(models)} models")
    # Send notification, update dashboard, etc.

scheduler = DiscoveryScheduler(
    registry=registry,
    refresh_interval_seconds=3600,
    notification_callback=on_discovery_complete
)
```

### Manual Discovery

```python
# Trigger discovery immediately
models = await scheduler.discover_now()
print(f"Found {len(models)} models")
```

### Monitoring Statistics

```python
stats = scheduler.get_stats()
print(f"Running: {stats['running']}")
print(f"Discoveries: {stats['discovery_count']}")
print(f"Failures: {stats['failure_count']}")
print(f"Last discovery: {stats['last_discovery_time']}")
print(f"Models found: {stats['last_discovered_count']}")
```

## Configuration

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `registry` | `ModelRegistry` | Required | Registry instance to refresh |
| `refresh_interval_seconds` | `int` | `3600` | Interval between discovery runs (1 hour) |
| `notification_callback` | `Callable` | `None` | Optional async callback for discovery completion |

### Recommended Intervals

- **Development**: 300 seconds (5 minutes) - Frequent updates for testing
- **Production**: 3600 seconds (1 hour) - Balance between freshness and API costs
- **Low-frequency**: 86400 seconds (24 hours) - For stable environments

## Architecture

### Discovery Flow

```
┌─────────────────────────────────────────────────────────┐
│                  Discovery Scheduler                     │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │         Periodic Discovery Loop                 │    │
│  │                                                  │    │
│  │  1. Wait for refresh_interval_seconds          │    │
│  │  2. Call registry.discover_models()            │    │
│  │  3. Update statistics                          │    │
│  │  4. Trigger notification callback (if set)     │    │
│  │  5. Repeat                                     │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  Manual Discovery:                                       │
│  - discover_now() bypasses schedule                     │
│  - Useful for immediate updates                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Model Registry                         │
│                                                          │
│  discover_models():                                      │
│  - Queries all registered adapters                      │
│  - Applies pricing information                          │
│  - Updates cache                                        │
│  - Returns discovered models                            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Provider Adapters                           │
│                                                          │
│  - BedrockAdapter.list_models()                         │
│  - SageMakerAdapter.list_models()                       │
│  - ExternalAPIAdapter.list_models()                     │
└─────────────────────────────────────────────────────────┘
```

### Error Handling

The scheduler is designed to be resilient:

1. **Discovery Errors**: Logged and tracked in `failure_count`, but scheduler continues
2. **Callback Errors**: Logged but don't interrupt the discovery cycle
3. **Graceful Shutdown**: Properly cancels background task on stop

```python
# Scheduler continues even if discovery fails
await scheduler.start()

# If registry.discover_models() fails:
# - Error is logged
# - failure_count increments
# - Scheduler waits for next interval
# - Next cycle attempts discovery again
```

## Integration Examples

### With Health Checker

```python
from src.registry.health_checker import HealthChecker
from src.registry.discovery_scheduler import DiscoveryScheduler

# Create both schedulers
health_checker = HealthChecker(registry, check_interval_seconds=300)
discovery_scheduler = DiscoveryScheduler(registry, refresh_interval_seconds=3600)

# Start both
await health_checker.start()
await discovery_scheduler.start()

# Both run independently in background
# Health checker: monitors existing models
# Discovery scheduler: finds new models

# Stop both
await health_checker.stop()
await discovery_scheduler.stop()
```

### With Notification System

```python
import boto3

sns_client = boto3.client('sns')
TOPIC_ARN = "arn:aws:sns:us-east-1:123456789012:model-updates"

async def notify_admins(models: list[ModelMetadata]):
    """Send SNS notification when new models are discovered."""
    message = f"Model discovery completed: {len(models)} models available"
    
    # Group by provider
    by_provider = {}
    for model in models:
        provider = model.provider.value
        by_provider[provider] = by_provider.get(provider, 0) + 1
    
    details = "\n".join([f"- {p}: {c} models" for p, c in by_provider.items()])
    
    sns_client.publish(
        TopicArn=TOPIC_ARN,
        Subject="Model Registry Update",
        Message=f"{message}\n\n{details}"
    )

scheduler = DiscoveryScheduler(
    registry=registry,
    refresh_interval_seconds=3600,
    notification_callback=notify_admins
)
```

### In a Web Application

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage scheduler lifecycle with FastAPI."""
    # Startup
    scheduler = DiscoveryScheduler(registry, refresh_interval_seconds=3600)
    await scheduler.start()
    
    yield
    
    # Shutdown
    await scheduler.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/models/refresh")
async def trigger_discovery():
    """Endpoint to manually trigger discovery."""
    models = await scheduler.discover_now()
    return {"models_found": len(models)}

@app.get("/scheduler/stats")
async def get_scheduler_stats():
    """Endpoint to get scheduler statistics."""
    return scheduler.get_stats()
```

## Best Practices

### 1. Choose Appropriate Intervals

```python
# Too frequent (may hit API rate limits)
scheduler = DiscoveryScheduler(registry, refresh_interval_seconds=60)  # ❌

# Reasonable for production
scheduler = DiscoveryScheduler(registry, refresh_interval_seconds=3600)  # ✅

# For stable environments
scheduler = DiscoveryScheduler(registry, refresh_interval_seconds=86400)  # ✅
```

### 2. Always Stop Scheduler on Shutdown

```python
try:
    await scheduler.start()
    # ... application runs ...
finally:
    await scheduler.stop()  # Ensure cleanup
```

### 3. Monitor Statistics

```python
# Periodically check scheduler health
stats = scheduler.get_stats()

if stats['failure_count'] > 10:
    logger.warning("High failure rate in discovery scheduler")
    # Alert administrators

if stats['last_discovered_count'] == 0:
    logger.warning("No models discovered in last cycle")
    # Check provider connectivity
```

### 4. Use Callbacks for Side Effects

```python
async def on_discovery(models: list[ModelMetadata]):
    """Handle discovery completion."""
    # Update cache
    await cache.set("model_count", len(models))
    
    # Update metrics
    metrics.gauge("models.total", len(models))
    
    # Log to audit trail
    await audit_log.record("model_discovery", {"count": len(models)})

scheduler = DiscoveryScheduler(
    registry=registry,
    notification_callback=on_discovery
)
```

## Testing

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_discovery_scheduler():
    # Create mock registry
    mock_registry = AsyncMock()
    mock_registry.discover_models.return_value = [model1, model2]
    
    # Create scheduler with short interval
    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=0.1
    )
    
    # Start and wait for discovery
    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()
    
    # Verify discovery was called
    assert mock_registry.discover_models.call_count >= 1
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_discovery_with_real_registry():
    # Create real registry with test adapter
    registry = ModelRegistry()
    adapter = BedrockAdapter(region="us-east-1")
    registry.register_adapter(adapter)
    
    # Create scheduler
    scheduler = DiscoveryScheduler(registry, refresh_interval_seconds=1)
    
    # Trigger manual discovery
    models = await scheduler.discover_now()
    
    # Verify models were discovered
    assert len(models) > 0
    assert all(isinstance(m, ModelMetadata) for m in models)
```

## Troubleshooting

### No Models Discovered

**Symptoms**: `last_discovered_count` is 0

**Possible Causes**:
1. No adapters registered in registry
2. Provider API credentials not configured
3. Network connectivity issues
4. Provider API rate limiting

**Solutions**:
```python
# Check registered adapters
stats = scheduler.get_stats()
print(f"Models found: {stats['last_discovered_count']}")

# Try manual discovery to see error
try:
    await scheduler.discover_now()
except Exception as e:
    print(f"Discovery error: {e}")

# Verify adapters are registered
adapters = registry._provider_adapters
print(f"Registered adapters: {adapters}")
```

### High Failure Rate

**Symptoms**: `failure_count` increasing rapidly

**Possible Causes**:
1. Provider API outages
2. Invalid credentials
3. Network issues
4. Rate limiting

**Solutions**:
```python
# Increase interval to reduce API calls
scheduler = DiscoveryScheduler(
    registry,
    refresh_interval_seconds=7200  # 2 hours
)

# Add retry logic in notification callback
async def resilient_callback(models):
    try:
        await process_models(models)
    except Exception as e:
        logger.error(f"Callback error: {e}")
        # Don't raise - let scheduler continue
```

### Memory Leaks

**Symptoms**: Memory usage grows over time

**Possible Causes**:
1. Scheduler not stopped properly
2. Callback holding references

**Solutions**:
```python
# Always stop scheduler
try:
    await scheduler.start()
    # ... run application ...
finally:
    await scheduler.stop()  # Critical!

# Use weak references in callbacks if needed
import weakref

class CallbackHandler:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
```

## See Also

- [Model Registry](model_registry.md) - Core registry functionality
- [Health Checker](health_checker.md) - Model availability monitoring
- [Adapters](adapters.md) - Provider-specific implementations
