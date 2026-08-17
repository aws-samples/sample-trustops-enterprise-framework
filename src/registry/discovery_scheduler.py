"""Model discovery scheduler for periodic model refresh.

This module implements the DiscoveryScheduler class which provides periodic
discovery of available models from all registered providers.

Requirements: 1.8
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Awaitable

from src.data_models.model import ModelMetadata


logger = logging.getLogger(__name__)


class DiscoveryScheduler:
    """Periodic scheduler for model discovery.

    The DiscoveryScheduler runs periodic discovery operations to refresh
    the model registry with the latest available models from all providers.
    This ensures the registry stays up-to-date with new models and changes
    to existing models.

    Requirements:
        - 1.8: Periodic refresh of available models from each provider

    Attributes:
        registry: ModelRegistry instance to refresh
        refresh_interval_seconds: Interval between discovery runs
        notification_callback: Optional callback for discovery completion
        _running: Flag indicating if scheduler is running
        _task: Background task handle
    """

    def __init__(
        self,
        registry: "ModelRegistry",
        refresh_interval_seconds: int = 3600,
        notification_callback: Optional[
            Callable[[list[ModelMetadata]], Awaitable[None]]
        ] = None
    ):
        """Initialize the DiscoveryScheduler.

        Args:
            registry: ModelRegistry instance to refresh
            refresh_interval_seconds: Interval between discovery runs (default: 1 hour)
            notification_callback: Optional async callback for discovery completion.
                Signature: async def callback(discovered_models: list[ModelMetadata])
        """
        self.registry = registry
        self.refresh_interval_seconds = refresh_interval_seconds
        self.notification_callback = notification_callback
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_discovery_time: Optional[datetime] = None
        self._discovery_count = 0
        self._failure_count = 0
        self._last_discovered_count = 0

    async def start(self) -> None:
        """Start the periodic discovery scheduler.

        This method starts a background task that periodically discovers
        available models from all registered providers.
        """
        if self._running:
            logger.warning("Discovery scheduler is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._discovery_loop())
        logger.info(
            f"Discovery scheduler started with interval: "
            f"{self.refresh_interval_seconds}s"
        )

    async def stop(self) -> None:
        """Stop the periodic discovery scheduler.

        This method stops the background discovery task gracefully.
        """
        if not self._running:
            logger.warning("Discovery scheduler is not running")
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Discovery scheduler stopped")

    async def _discovery_loop(self) -> None:
        """Background loop that performs periodic model discovery."""
        while self._running:
            try:
                await self._discover_models()
                self._last_discovery_time = datetime.now()
                self._discovery_count += 1
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                self._failure_count += 1

            # Wait for next discovery interval
            await asyncio.sleep(self.refresh_interval_seconds)

    async def _discover_models(self) -> None:
        """Discover models from all registered providers."""
        logger.info("Starting model discovery from all providers")

        try:
            discovered_models = await self.registry.discover_models()
            self._last_discovered_count = len(discovered_models)

            logger.info(
                f"Discovery completed: {len(discovered_models)} models found"
            )

            # Notify callback if configured
            if self.notification_callback:
                try:
                    await self.notification_callback(discovered_models)
                except Exception as e:
                    logger.error(
                        f"Error in discovery notification callback: {e}"
                    )

        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            raise

    async def discover_now(self) -> list[ModelMetadata]:
        """Perform an immediate model discovery.

        This method allows manual triggering of model discovery outside
        of the periodic schedule.

        Returns:
            List of discovered ModelMetadata objects

        Raises:
            RuntimeError: If model discovery fails
        """
        logger.info("Manual model discovery triggered")
        await self._discover_models()
        return await self.registry.list_models()

    def get_stats(self) -> dict:
        """Get discovery scheduler statistics.

        Returns:
            Dictionary with statistics including:
            - running: Whether scheduler is running
            - refresh_interval_seconds: Configured refresh interval
            - last_discovery_time: Timestamp of last discovery
            - discovery_count: Total number of discoveries performed
            - failure_count: Number of failed discovery cycles
            - last_discovered_count: Number of models found in last discovery
        """
        return {
            "running": self._running,
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "last_discovery_time": (
                self._last_discovery_time.isoformat()
                if self._last_discovery_time
                else None
            ),
            "discovery_count": self._discovery_count,
            "failure_count": self._failure_count,
            "last_discovered_count": self._last_discovered_count,
        }
