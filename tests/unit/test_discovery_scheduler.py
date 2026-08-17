"""Unit tests for the DiscoveryScheduler class.

Tests periodic model discovery, notification callbacks, and manual discovery.

Requirements: 1.8
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.registry.discovery_scheduler import DiscoveryScheduler
from src.data_models.model import (
    ModelMetadata,
    ModelProvider,
    ModelCapability,
    ModelStatus,
)


@pytest.fixture
def sample_models():
    """Create sample model metadata for testing."""
    return [
        ModelMetadata(
            id="bedrock-model-1",
            provider=ModelProvider.BEDROCK,
            name="Claude 3 Sonnet",
            capabilities=[ModelCapability.TEXT_GENERATION],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=True,
            input_modalities=["text"],
            output_modalities=["text"],
            max_tokens=4096,
            region="us-east-1",
        ),
        ModelMetadata(
            id="bedrock-model-2",
            provider=ModelProvider.BEDROCK,
            name="Titan Text G1",
            capabilities=[ModelCapability.TEXT_GENERATION],
            status=ModelStatus.ACTIVE,
            fine_tuning_support=False,
            input_modalities=["text"],
            output_modalities=["text"],
            max_tokens=8192,
            region="us-east-1",
        ),
    ]


@pytest.fixture
def mock_registry(sample_models):
    """Create a mock ModelRegistry."""
    registry = MagicMock()
    registry.discover_models = AsyncMock(return_value=sample_models)
    registry.list_models = AsyncMock(return_value=sample_models)
    return registry


@pytest.mark.asyncio
async def test_discovery_scheduler_initialization(mock_registry):
    """Test DiscoveryScheduler initialization with default parameters."""
    scheduler = DiscoveryScheduler(mock_registry)

    assert scheduler.registry == mock_registry
    assert scheduler.refresh_interval_seconds == 3600  # Default 1 hour
    assert scheduler.notification_callback is None
    assert scheduler._running is False
    assert scheduler._task is None
    assert scheduler._last_discovery_time is None
    assert scheduler._discovery_count == 0
    assert scheduler._failure_count == 0


@pytest.mark.asyncio
async def test_discovery_scheduler_custom_interval(mock_registry):
    """Test DiscoveryScheduler with custom refresh interval."""
    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=600  # 10 minutes
    )

    assert scheduler.refresh_interval_seconds == 600


@pytest.mark.asyncio
async def test_discovery_scheduler_with_callback(mock_registry, sample_models):
    """Test DiscoveryScheduler with notification callback."""
    callback = AsyncMock()
    scheduler = DiscoveryScheduler(
        mock_registry,
        notification_callback=callback
    )

    assert scheduler.notification_callback == callback


@pytest.mark.asyncio
async def test_start_discovery_scheduler(mock_registry):
    """Test starting the discovery scheduler."""
    scheduler = DiscoveryScheduler(mock_registry, refresh_interval_seconds=1)

    await scheduler.start()

    assert scheduler._running is True
    assert scheduler._task is not None
    assert isinstance(scheduler._task, asyncio.Task)

    # Clean up
    await scheduler.stop()


@pytest.mark.asyncio
async def test_start_already_running(mock_registry, caplog):
    """Test starting scheduler when already running logs warning."""
    scheduler = DiscoveryScheduler(mock_registry, refresh_interval_seconds=1)

    await scheduler.start()
    await scheduler.start()  # Try to start again

    assert "already running" in caplog.text

    # Clean up
    await scheduler.stop()


@pytest.mark.asyncio
async def test_stop_discovery_scheduler(mock_registry):
    """Test stopping the discovery scheduler."""
    scheduler = DiscoveryScheduler(mock_registry, refresh_interval_seconds=1)

    await scheduler.start()
    assert scheduler._running is True

    await scheduler.stop()

    assert scheduler._running is False
    assert scheduler._task is None


@pytest.mark.asyncio
async def test_stop_not_running(mock_registry, caplog):
    """Test stopping scheduler when not running logs warning."""
    scheduler = DiscoveryScheduler(mock_registry)

    await scheduler.stop()

    assert "not running" in caplog.text


@pytest.mark.asyncio
async def test_periodic_discovery_loop(mock_registry, sample_models):
    """Test the periodic discovery loop."""
    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=0.1  # Very short interval for testing
    )

    await scheduler.start()

    # Wait for at least 2 discovery cycles
    await asyncio.sleep(0.3)

    await scheduler.stop()

    # Verify discover_models was called multiple times
    assert mock_registry.discover_models.call_count >= 2
    assert scheduler._discovery_count >= 2
    assert scheduler._last_discovery_time is not None
    assert isinstance(scheduler._last_discovery_time, datetime)


@pytest.mark.asyncio
async def test_discovery_updates_stats(mock_registry, sample_models):
    """Test that discovery updates statistics correctly."""
    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=0.1
    )

    await scheduler.start()
    await asyncio.sleep(0.15)  # Wait for one discovery cycle
    await scheduler.stop()

    assert scheduler._discovery_count >= 1
    assert scheduler._last_discovered_count == len(sample_models)
    assert scheduler._last_discovery_time is not None


@pytest.mark.asyncio
async def test_discovery_with_notification_callback(mock_registry, sample_models):
    """Test discovery triggers notification callback."""
    callback = AsyncMock()
    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=0.1,
        notification_callback=callback
    )

    await scheduler.start()
    await asyncio.sleep(0.15)  # Wait for one discovery cycle
    await scheduler.stop()

    # Verify callback was called with discovered models
    assert callback.call_count >= 1
    callback.assert_called_with(sample_models)


@pytest.mark.asyncio
async def test_discovery_callback_error_handling(mock_registry, sample_models, caplog):
    """Test that callback errors don't crash the scheduler."""
    callback = AsyncMock(side_effect=Exception("Callback error"))
    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=0.1,
        notification_callback=callback
    )

    await scheduler.start()
    await asyncio.sleep(0.15)  # Wait for one discovery cycle
    await scheduler.stop()

    # Scheduler should continue running despite callback error
    assert scheduler._discovery_count >= 1
    assert "Error in discovery notification callback" in caplog.text


@pytest.mark.asyncio
async def test_discovery_error_handling(mock_registry, caplog):
    """Test that discovery errors are logged and don't crash scheduler."""
    mock_registry.discover_models = AsyncMock(
        side_effect=Exception("Discovery failed")
    )

    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=0.1
    )

    await scheduler.start()
    await asyncio.sleep(0.15)  # Wait for one discovery cycle
    await scheduler.stop()

    # Scheduler should continue running despite errors
    assert scheduler._failure_count >= 1
    assert "Error in discovery loop" in caplog.text


@pytest.mark.asyncio
async def test_discover_now(mock_registry, sample_models):
    """Test manual discovery trigger."""
    scheduler = DiscoveryScheduler(mock_registry)

    result = await scheduler.discover_now()

    # Verify discover_models was called
    mock_registry.discover_models.assert_called_once()
    mock_registry.list_models.assert_called_once()

    # Verify result matches sample models
    assert result == sample_models
    assert scheduler._last_discovered_count == len(sample_models)


@pytest.mark.asyncio
async def test_discover_now_with_error(mock_registry):
    """Test manual discovery with error."""
    mock_registry.discover_models = AsyncMock(
        side_effect=RuntimeError("Discovery failed")
    )

    scheduler = DiscoveryScheduler(mock_registry)

    with pytest.raises(RuntimeError, match="Discovery failed"):
        await scheduler.discover_now()


@pytest.mark.asyncio
async def test_get_stats_initial(mock_registry):
    """Test get_stats returns correct initial values."""
    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=600
    )

    stats = scheduler.get_stats()

    assert stats["running"] is False
    assert stats["refresh_interval_seconds"] == 600
    assert stats["last_discovery_time"] is None
    assert stats["discovery_count"] == 0
    assert stats["failure_count"] == 0
    assert stats["last_discovered_count"] == 0


@pytest.mark.asyncio
async def test_get_stats_after_discovery(mock_registry, sample_models):
    """Test get_stats returns correct values after discovery."""
    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=0.1
    )

    await scheduler.start()
    await asyncio.sleep(0.15)  # Wait for one discovery cycle
    await scheduler.stop()

    stats = scheduler.get_stats()

    assert stats["running"] is False
    assert stats["refresh_interval_seconds"] == 0.1
    assert stats["last_discovery_time"] is not None
    assert stats["discovery_count"] >= 1
    assert stats["failure_count"] == 0
    assert stats["last_discovered_count"] == len(sample_models)


@pytest.mark.asyncio
async def test_get_stats_with_failures(mock_registry):
    """Test get_stats tracks failures correctly."""
    mock_registry.discover_models = AsyncMock(
        side_effect=Exception("Discovery failed")
    )

    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=0.1
    )

    await scheduler.start()
    await asyncio.sleep(0.15)  # Wait for one discovery cycle
    await scheduler.stop()

    stats = scheduler.get_stats()

    assert stats["failure_count"] >= 1


@pytest.mark.asyncio
async def test_multiple_start_stop_cycles(mock_registry):
    """Test multiple start/stop cycles work correctly."""
    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=0.1
    )

    # First cycle
    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()
    first_count = scheduler._discovery_count

    # Second cycle
    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()
    second_count = scheduler._discovery_count

    # Discovery count should increase across cycles
    assert second_count > first_count


@pytest.mark.asyncio
async def test_concurrent_discover_now_calls(mock_registry, sample_models):
    """Test multiple concurrent discover_now calls."""
    scheduler = DiscoveryScheduler(mock_registry)

    # Run multiple discover_now calls concurrently
    results = await asyncio.gather(
        scheduler.discover_now(),
        scheduler.discover_now(),
        scheduler.discover_now()
    )

    # All should succeed and return the same models
    assert len(results) == 3
    for result in results:
        assert result == sample_models


@pytest.mark.asyncio
async def test_discovery_scheduler_logs_info(mock_registry, caplog):
    """Test that discovery scheduler logs appropriate info messages."""
    import logging
    caplog.set_level(logging.INFO)
    
    scheduler = DiscoveryScheduler(
        mock_registry,
        refresh_interval_seconds=0.1
    )

    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()

    # Check for expected log messages
    assert "Discovery scheduler started" in caplog.text
    assert "Starting model discovery" in caplog.text
    assert "Discovery completed" in caplog.text
    assert "Discovery scheduler stopped" in caplog.text
