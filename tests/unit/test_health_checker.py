"""Unit tests for the HealthChecker class.

Tests periodic health checking, status updates, and administrator notifications.

Requirements: 1.5, 1.13
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.registry.health_checker import HealthChecker
from src.registry.model_registry import ModelRegistry
from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
)


@pytest.fixture
def mock_registry():
    """Create a mock ModelRegistry for testing."""
    registry = AsyncMock(spec=ModelRegistry)
    return registry


@pytest.fixture
def sample_model():
    """Create a sample model metadata for testing."""
    return ModelMetadata(
        id="test-model-1",
        provider=ModelProvider.BEDROCK,
        name="Test Model",
        capabilities=[ModelCapability.TEXT_GENERATION],
        status=ModelStatus.ACTIVE,
        fine_tuning_support=False,
        input_modalities=["text"],
        output_modalities=["text"],
        max_tokens=4096,
        region="us-east-1",
        last_health_check=datetime.now(),
    )


@pytest.fixture
def health_checker(mock_registry):
    """Create a HealthChecker instance for testing."""
    return HealthChecker(
        registry=mock_registry,
        check_interval_seconds=1  # Short interval for testing
    )


@pytest.mark.asyncio
async def test_health_checker_initialization(mock_registry):
    """Test HealthChecker initialization with default parameters."""
    checker = HealthChecker(registry=mock_registry)

    assert checker.registry == mock_registry
    assert checker.check_interval_seconds == 300  # Default 5 minutes
    assert checker.notification_callback is None
    assert not checker._running
    assert checker._task is None


@pytest.mark.asyncio
async def test_health_checker_custom_interval(mock_registry):
    """Test HealthChecker initialization with custom interval."""
    checker = HealthChecker(
        registry=mock_registry,
        check_interval_seconds=60
    )

    assert checker.check_interval_seconds == 60


@pytest.mark.asyncio
async def test_health_checker_with_notification_callback(mock_registry):
    """Test HealthChecker initialization with notification callback."""
    callback = AsyncMock()
    checker = HealthChecker(
        registry=mock_registry,
        notification_callback=callback
    )

    assert checker.notification_callback == callback


@pytest.mark.asyncio
async def test_start_health_checker(health_checker):
    """Test starting the health checker."""
    await health_checker.start()

    assert health_checker._running
    assert health_checker._task is not None

    # Clean up
    await health_checker.stop()


@pytest.mark.asyncio
async def test_start_already_running(health_checker):
    """Test starting health checker when already running."""
    await health_checker.start()
    
    # Try to start again
    await health_checker.start()
    
    assert health_checker._running
    
    # Clean up
    await health_checker.stop()


@pytest.mark.asyncio
async def test_stop_health_checker(health_checker):
    """Test stopping the health checker."""
    await health_checker.start()
    assert health_checker._running

    await health_checker.stop()
    assert not health_checker._running
    assert health_checker._task is None


@pytest.mark.asyncio
async def test_stop_not_running(health_checker):
    """Test stopping health checker when not running."""
    # Should not raise an error
    await health_checker.stop()
    assert not health_checker._running


@pytest.mark.asyncio
async def test_check_model_healthy(mock_registry, sample_model):
    """Test checking a healthy model."""
    mock_registry.get_model.return_value = sample_model
    mock_registry.health_check.return_value = True

    checker = HealthChecker(registry=mock_registry)
    await checker._check_model(sample_model.id)

    mock_registry.health_check.assert_called_once_with(sample_model.id)


@pytest.mark.asyncio
async def test_check_model_unhealthy_status_change(mock_registry, sample_model):
    """Test checking an unhealthy model that changes status."""
    # Initial model is active
    active_model = sample_model.model_copy()
    active_model.status = ModelStatus.ACTIVE

    # After health check, model becomes inactive
    inactive_model = sample_model.model_copy()
    inactive_model.status = ModelStatus.INACTIVE

    mock_registry.get_model.side_effect = [active_model, inactive_model]
    mock_registry.health_check.return_value = False

    notification_callback = AsyncMock()
    checker = HealthChecker(
        registry=mock_registry,
        notification_callback=notification_callback
    )

    await checker._check_model(sample_model.id)

    # Verify notification was called
    notification_callback.assert_called_once_with(
        sample_model.id,
        ModelStatus.ACTIVE,
        ModelStatus.INACTIVE
    )


@pytest.mark.asyncio
async def test_check_model_no_status_change(mock_registry, sample_model):
    """Test checking a model where status doesn't change."""
    mock_registry.get_model.return_value = sample_model
    mock_registry.health_check.return_value = True

    notification_callback = AsyncMock()
    checker = HealthChecker(
        registry=mock_registry,
        notification_callback=notification_callback
    )

    await checker._check_model(sample_model.id)

    # Notification should not be called
    notification_callback.assert_not_called()


@pytest.mark.asyncio
async def test_check_model_not_found(mock_registry):
    """Test checking a model that doesn't exist."""
    mock_registry.get_model.return_value = None

    checker = HealthChecker(registry=mock_registry)
    
    # Should not raise an error
    await checker._check_model("nonexistent-model")
    
    mock_registry.health_check.assert_not_called()


@pytest.mark.asyncio
async def test_check_all_models(mock_registry, sample_model):
    """Test checking all registered models."""
    model1 = sample_model.model_copy()
    model1.id = "model-1"
    
    model2 = sample_model.model_copy()
    model2.id = "model-2"

    mock_registry.list_models.return_value = [model1, model2]
    mock_registry.get_model.return_value = sample_model
    mock_registry.health_check.return_value = True

    checker = HealthChecker(registry=mock_registry)
    await checker._check_all_models()

    # Verify health check was called for both models
    assert mock_registry.health_check.call_count == 2


@pytest.mark.asyncio
async def test_check_all_models_with_error(mock_registry, sample_model):
    """Test checking all models when one check fails."""
    model1 = sample_model.model_copy()
    model1.id = "model-1"
    
    model2 = sample_model.model_copy()
    model2.id = "model-2"

    mock_registry.list_models.return_value = [model1, model2]
    
    # First model check fails, second succeeds
    mock_registry.get_model.side_effect = [
        None,  # First model not found
        model2,  # Second model found
        model2   # Second model after health check
    ]
    mock_registry.health_check.return_value = True

    checker = HealthChecker(registry=mock_registry)
    
    # Should not raise an error
    await checker._check_all_models()
    
    # Only one health check should succeed
    mock_registry.health_check.assert_called_once()


@pytest.mark.asyncio
async def test_check_model_now(mock_registry, sample_model):
    """Test immediate health check on a specific model."""
    active_model = sample_model.model_copy()
    active_model.status = ModelStatus.ACTIVE

    mock_registry.get_model.return_value = active_model
    mock_registry.health_check.return_value = True

    checker = HealthChecker(registry=mock_registry)
    result = await checker.check_model_now(sample_model.id)

    assert result is True
    mock_registry.health_check.assert_called_once_with(sample_model.id)


@pytest.mark.asyncio
async def test_check_model_now_unhealthy(mock_registry, sample_model):
    """Test immediate health check on an unhealthy model."""
    inactive_model = sample_model.model_copy()
    inactive_model.status = ModelStatus.INACTIVE

    mock_registry.get_model.side_effect = [
        sample_model,  # Before check in _check_model
        inactive_model,  # After check in _check_model
        inactive_model  # Final check in check_model_now
    ]
    mock_registry.health_check.return_value = False

    checker = HealthChecker(registry=mock_registry)
    result = await checker.check_model_now(sample_model.id)

    assert result is False


@pytest.mark.asyncio
async def test_notification_callback_error(mock_registry, sample_model):
    """Test that notification callback errors are handled gracefully."""
    active_model = sample_model.model_copy()
    active_model.status = ModelStatus.ACTIVE

    inactive_model = sample_model.model_copy()
    inactive_model.status = ModelStatus.INACTIVE

    mock_registry.get_model.side_effect = [active_model, inactive_model]
    mock_registry.health_check.return_value = False

    # Callback that raises an error
    notification_callback = AsyncMock(side_effect=Exception("Callback error"))
    
    checker = HealthChecker(
        registry=mock_registry,
        notification_callback=notification_callback
    )

    # Should not raise an error
    await checker._check_model(sample_model.id)
    
    notification_callback.assert_called_once()


@pytest.mark.asyncio
async def test_periodic_health_check_loop(mock_registry, sample_model):
    """Test the periodic health check loop."""
    mock_registry.list_models.return_value = [sample_model]
    mock_registry.get_model.return_value = sample_model
    mock_registry.health_check.return_value = True

    checker = HealthChecker(
        registry=mock_registry,
        check_interval_seconds=0.1  # Very short interval for testing
    )

    await checker.start()
    
    # Wait for at least 2 checks
    await asyncio.sleep(0.3)
    
    await checker.stop()

    # Verify multiple checks occurred
    assert checker._check_count >= 2
    assert mock_registry.list_models.call_count >= 2


@pytest.mark.asyncio
async def test_get_stats(health_checker):
    """Test getting health checker statistics."""
    stats = health_checker.get_stats()

    assert stats["running"] is False
    assert stats["check_interval_seconds"] == 1
    assert stats["last_check_time"] is None
    assert stats["check_count"] == 0
    assert stats["failure_count"] == 0


@pytest.mark.asyncio
async def test_get_stats_after_checks(mock_registry, sample_model):
    """Test statistics after performing health checks."""
    mock_registry.list_models.return_value = [sample_model]
    mock_registry.get_model.return_value = sample_model
    mock_registry.health_check.return_value = True

    checker = HealthChecker(
        registry=mock_registry,
        check_interval_seconds=0.1
    )

    await checker.start()
    await asyncio.sleep(0.3)
    await checker.stop()

    stats = checker.get_stats()

    assert stats["running"] is False
    assert stats["check_count"] >= 2
    assert stats["last_check_time"] is not None


@pytest.mark.asyncio
async def test_health_check_loop_error_handling(mock_registry):
    """Test that errors in health check loop are handled gracefully."""
    # Make list_models raise an error
    mock_registry.list_models.side_effect = Exception("Test error")

    checker = HealthChecker(
        registry=mock_registry,
        check_interval_seconds=0.1
    )

    await checker.start()
    await asyncio.sleep(0.3)
    await checker.stop()

    # Checker should still be running despite errors
    assert checker._failure_count > 0
