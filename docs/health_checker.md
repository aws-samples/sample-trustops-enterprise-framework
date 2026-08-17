# Model Health Checker

The Model Health Checker provides periodic monitoring of model availability and automatic status updates when models become unavailable.

## Overview

The `HealthChecker` class implements requirements 1.5 and 1.13 from the TrustOps Enterprise Framework:

- **Requirement 1.5**: Mark models as inactive when they become unavailable and notify administrators
- **Requirement 1.13**: Periodic ping to verify model availability

## Features

- **Periodic Health Checks**: Automatically checks all registered models at configurable intervals
- **Status Updates**: Updates model status to `INACTIVE` when health checks fail
- **Administrator Notifications**: Calls a configurable callback when model status changes
- **Manual Checks**: Supports immediate health checks on specific models
- **Error Handling**: Gracefully handles errors and continues monitoring
- **Statistics**: Tracks check counts, failures, and timing

## Usage

### Basic Setup

```python
from src.registry.model_registry import ModelRegistry
from src.registry.health_checker import HealthChecker

# Create a ModelRegistry
registry = ModelRegistry()

# Create a HealthChecker
health_checker = HealthChecker(
    registry=registry,
    check_interval_seconds=300  # Check every 5 minutes
)

# Start monitoring
await health_checker.start()

# ... application runs ...

# Stop monitoring
await health_checker.stop()
```

### With Notification Callback

```python
from src.data_models.model import ModelStatus

async def notify_admin(
    model_id: str,
    old_status: ModelStatus,
    new_status: ModelStatus
) -> None:
    """Send notification when model status changes."""
    print(f"Alert: {model_id} changed from {old_status} to {new_status}")
    # Send email, Slack message, PagerDuty alert, etc.

health_checker = HealthChecker(
    registry=registry,
    check_interval_seconds=300,
    notification_callback=notify_admin
)

await health_checker.start()
```

### Manual Health Check

```python
# Check a specific model immediately
is_healthy = await health_checker.check_model_now("anthropic.claude-v2")

if is_healthy:
    print("Model is healthy")
else:
    print("Model is unhealthy")
```

### Get Statistics

```python
stats = health_checker.get_stats()

print(f"Running: {stats['running']}")
print(f"Check Interval: {stats['check_interval_seconds']}s")
print(f"Total Checks: {stats['check_count']}")
print(f"Failed Checks: {stats['failure_count']}")
print(f"Last Check: {stats['last_check_time']}")
```

## Configuration

### Constructor Parameters

- **registry** (ModelRegistry): The ModelRegistry instance to monitor (required)
- **check_interval_seconds** (int): Interval between health checks in seconds (default: 300)
- **notification_callback** (callable): Optional async callback for status changes

### Notification Callback Signature

```python
async def callback(
    model_id: str,
    old_status: ModelStatus,
    new_status: ModelStatus
) -> None:
    """
    Args:
        model_id: The model that changed status
        old_status: Previous status
        new_status: New status
    """
    pass
```

## Health Check Process

1. **List Models**: Retrieves all registered models from the registry
2. **Check Each Model**: For each model:
   - Calls `registry.health_check(model_id)`
   - The registry validates connectivity to the model's provider
   - Updates the model's `last_health_check` timestamp
3. **Update Status**: If health check fails:
   - Updates model status to `INACTIVE`
   - Calls notification callback if configured
4. **Repeat**: Waits for the configured interval and repeats

## Status Transitions

The health checker manages the following status transitions:

- `ACTIVE` → `INACTIVE`: When a health check fails
- `INACTIVE` → `ACTIVE`: When a health check succeeds after previous failures

## Error Handling

The health checker is designed to be resilient:

- **Individual Model Errors**: If checking one model fails, it continues with other models
- **Callback Errors**: If the notification callback raises an error, it's logged but doesn't stop monitoring
- **Loop Errors**: If the entire check cycle fails, it's logged and the next cycle continues

## Integration with ModelRegistry

The health checker relies on the `ModelRegistry.health_check()` method, which:

1. Retrieves model metadata
2. Finds the appropriate adapter for the model's provider
3. Calls `adapter.validate_connection()`
4. Updates the model's status based on the result

## Production Considerations

### Notification Callbacks

In production, implement notification callbacks that:

- Send email alerts to administrators
- Post to Slack/Teams channels
- Create PagerDuty incidents for critical models
- Log to centralized monitoring systems (CloudWatch, Datadog, etc.)

Example:

```python
async def production_notification(
    model_id: str,
    old_status: ModelStatus,
    new_status: ModelStatus
) -> None:
    # Send email
    await send_email(
        to="ml-ops@company.com",
        subject=f"Model Status Change: {model_id}",
        body=f"{model_id} changed from {old_status} to {new_status}"
    )
    
    # Post to Slack
    await slack_client.post_message(
        channel="#ml-alerts",
        text=f"🚨 Model {model_id} is now {new_status}"
    )
    
    # Log to CloudWatch
    cloudwatch.put_metric_data(
        Namespace="TrustOps/Models",
        MetricData=[{
            "MetricName": "ModelStatusChange",
            "Value": 1,
            "Dimensions": [
                {"Name": "ModelId", "Value": model_id},
                {"Name": "NewStatus", "Value": new_status.value}
            ]
        }]
    )
```

### Check Interval

Choose an appropriate check interval based on:

- **Model criticality**: More frequent checks for critical models
- **Provider rate limits**: Avoid overwhelming provider APIs
- **Cost**: Some providers charge for health check requests
- **Latency requirements**: How quickly you need to detect failures

Recommended intervals:
- **Production critical models**: 60-300 seconds (1-5 minutes)
- **Development models**: 600-1800 seconds (10-30 minutes)
- **Archived models**: 3600+ seconds (1+ hour)

### Running as a Background Service

For production deployments, run the health checker as a background service:

```python
# In your application startup
async def startup():
    registry = ModelRegistry()
    health_checker = HealthChecker(
        registry=registry,
        check_interval_seconds=300,
        notification_callback=production_notification
    )
    await health_checker.start()
    
    # Store reference for shutdown
    app.state.health_checker = health_checker

# In your application shutdown
async def shutdown():
    await app.state.health_checker.stop()
```

## Testing

The health checker includes comprehensive unit tests covering:

- Initialization and configuration
- Starting and stopping
- Health check execution
- Status change detection
- Notification callbacks
- Error handling
- Statistics tracking

Run tests with:

```bash
pytest tests/unit/test_health_checker.py -v
```

## Demo

A complete demo is available in `demo/health_checker_demo.py`:

```bash
python demo/health_checker_demo.py
```

## Related Components

- **ModelRegistry**: Manages model metadata and provides the `health_check()` method
- **BaseModelAdapter**: Provides the `validate_connection()` method for each provider
- **ModelStatus**: Enum defining model status values (ACTIVE, INACTIVE, etc.)
