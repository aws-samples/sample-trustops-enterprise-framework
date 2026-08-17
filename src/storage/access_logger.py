"""
Access logging for audit compliance.

Logs all data access events including user ID, action, timestamp, and result_id.

Requirements: 9.9
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AccessLogEntry(BaseModel):
    """A single access log entry."""

    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    result_id: str
    action: str
    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict = Field(default_factory=dict)


class AccessLogger:
    """Logs data access events for audit compliance."""

    def __init__(self, metadata_store: Any = None):
        """Initialize AccessLogger.

        Args:
            metadata_store: Optional DynamoDBMetadataStore for persistent logging.
        """
        self._store = metadata_store
        self._log: list[AccessLogEntry] = []

    def log(
        self,
        result_id: str,
        action: str,
        user_id: str,
        details: dict | None = None,
    ) -> AccessLogEntry:
        """Log a data access event.

        Args:
            result_id: ID of the result accessed.
            action: Action performed (e.g., 'read', 'write', 'delete', 'export').
            user_id: ID of the user performing the action.
            details: Optional additional details.

        Returns:
            The created AccessLogEntry.
        """
        entry = AccessLogEntry(
            result_id=result_id,
            action=action,
            user_id=user_id,
            details=details or {},
        )

        self._log.append(entry)

        if self._store:
            self._store.log_access(result_id, action, user_id)

        logger.info(
            "Access: user=%s action=%s result=%s",
            user_id, action, result_id,
        )

        return entry

    def get_logs(
        self,
        result_id: str | None = None,
        user_id: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[AccessLogEntry]:
        """Get access log entries with optional filtering.

        Args:
            result_id: Filter by result ID.
            user_id: Filter by user ID.
            action: Filter by action.
            limit: Maximum entries to return.

        Returns:
            List of matching AccessLogEntry objects.
        """
        entries = self._log
        if result_id:
            entries = [e for e in entries if e.result_id == result_id]
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        if action:
            entries = [e for e in entries if e.action == action]
        return entries[:limit]
