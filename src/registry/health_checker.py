"""Model health checker for periodic availability monitoring.

This module implements the HealthChecker class which provides periodic
health checking for registered models and automatic status updates.

Requirements: 1.5, 1.13
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Awaitable

from src.data_models.model import ModelStatus


logger = logging.getLogger(__name__)


class HealthChecker:
    """Periodic health checker for model availability.

    The HealthChecker runs periodic health checks on registered models,
    updates their status when they become unavailable, and notifies
    administrators of status changes.

    Requirements:
        - 1.5: Mark models as inactive when unavailable
        - 1.13: Periodic ping to verify model availability

    Attributes:
        registry: ModelRegistry instance to check models
        check_interval_seconds: Interval between health checks
        notification_callback: Optional callback for status change notifications
        _running: Flag indicating if health checker is running
        _task: Background task handle
    """

    def __init__(
        self,
        registry: "ModelRegistry",
        check_interval_seconds: int = 300,
        notification_callback: Optional[
            Callable[[str, ModelStatus, ModelStatus], Awaitable[None]]
        ] = None
    ):
        """Initialize the HealthChecker.

        Args:
            registry: ModelRegistry instance to monitor
            check_interval_seconds: Interval between checks (default: 5 minutes)
            notification_callback: Optional async callback for status changes.
                Signature: async def callback(model_id, old_status, new_status)
        """
        self.registry = registry
        self.check_interval_seconds = check_interval_seconds
        self.notification_callback = notification_callback
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_check_time: Optional[datetime] = None
        self._check_count = 0
        self._failure_count = 0

    async def start(self) -> None:
        """Start the periodic health checker.

        This method starts a background task that periodically checks
        all registered models for availability.
        """
        if self._running:
            logger.warning("Health checker is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())
        logger.info(
            f"Health checker started with interval: "
            f"{self.check_interval_seconds}s"
        )

    async def stop(self) -> None:
        """Stop the periodic health checker.

        This method stops the background health check task gracefully.
        """
        if not self._running:
            logger.warning("Health checker is not running")
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Health checker stopped")

    async def _health_check_loop(self) -> None:
        """Background loop that performs periodic health checks."""
        while self._running:
            try:
                await self._check_all_models()
                self._last_check_time = datetime.now()
                self._check_count += 1
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                self._failure_count += 1

            # Wait for next check interval
            await asyncio.sleep(self.check_interval_seconds)

    async def _check_all_models(self) -> None:
        """Check health of all registered models."""
        models = await self.registry.list_models()

        logger.info(f"Starting health check for {len(models)} models")

        for model in models:
            try:
                await self._check_model(model.id)
            except Exception as e:
                logger.error(
                    f"Error checking model {model.id}: {e}"
                )

    async def _check_model(self, model_id: str) -> None:
        """Check health of a single model and update status if needed.

        Args:
            model_id: The model ID to check
        """
        # Get current model metadata
        metadata = await self.registry.get_model(model_id)
        if not metadata:
            logger.warning(f"Model not found: {model_id}")
            return

        old_status = metadata.status

        # Perform health check
        is_healthy = await self.registry.health_check(model_id)

        # Get updated metadata to check if status changed
        updated_metadata = await self.registry.get_model(model_id)
        if not updated_metadata:
            return

        new_status = updated_metadata.status

        # Log status change
        if old_status != new_status:
            logger.info(
                f"Model {model_id} status changed: "
                f"{old_status.value} -> {new_status.value}"
            )

            # Notify administrators if callback is configured
            if self.notification_callback:
                try:
                    await self.notification_callback(
                        model_id, old_status, new_status
                    )
                except Exception as e:
                    logger.error(
                        f"Error in notification callback for {model_id}: {e}"
                    )
        else:
            # Log health check result even if status didn't change
            status_str = "healthy" if is_healthy else "unhealthy"
            logger.debug(f"Model {model_id} is {status_str}")

    async def check_model_now(self, model_id: str) -> bool:
        """Perform an immediate health check on a specific model.

        This method allows manual triggering of a health check outside
        of the periodic schedule.

        Args:
            model_id: The model ID to check

        Returns:
            True if model is healthy, False otherwise
        """
        await self._check_model(model_id)
        metadata = await self.registry.get_model(model_id)
        return metadata.status == ModelStatus.ACTIVE if metadata else False

    def get_stats(self) -> dict:
        """Get health checker statistics.

        Returns:
            Dictionary with statistics including:
            - running: Whether health checker is running
            - check_interval_seconds: Configured check interval
            - last_check_time: Timestamp of last check
            - check_count: Total number of checks performed
            - failure_count: Number of failed check cycles
        """
        return {
            "running": self._running,
            "check_interval_seconds": self.check_interval_seconds,
            "last_check_time": (
                self._last_check_time.isoformat()
                if self._last_check_time
                else None
            ),
            "check_count": self._check_count,
            "failure_count": self._failure_count,
        }
