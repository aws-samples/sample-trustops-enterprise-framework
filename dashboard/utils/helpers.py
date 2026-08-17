"""Dashboard helper utilities."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any, Optional


def format_timestamp(dt: datetime) -> str:
    """Format a datetime for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_cost(cost: float) -> str:
    """Format a cost value as a dollar string."""
    if cost == 0.0:
        return "$0.0000"
    # Use 4 decimal places for small costs, 2 for larger ones
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


_STATUS_COLORS = {
    "active": "green",
    "completed": "green",
    "running": "blue",
    "in_progress": "blue",
    "pending": "orange",
    "queued": "orange",
    "failed": "red",
    "error": "red",
    "inactive": "gray",
    "cancelled": "gray",
}


def status_color(status: str) -> str:
    """Return a colour name for a given status string."""
    return _STATUS_COLORS.get(status.lower(), "gray")


_STATUS_EMOJIS = {
    "active": "🟢",
    "completed": "✅",
    "running": "🔵",
    "in_progress": "🔵",
    "pending": "🟡",
    "queued": "🟡",
    "failed": "🔴",
    "error": "🔴",
    "inactive": "⚪",
    "cancelled": "⚪",
}


def status_emoji(status: str) -> str:
    """Return an emoji for a given status string."""
    return _STATUS_EMOJIS.get(status.lower(), "⚪")


def generate_demo_id(prefix: str = "demo") -> str:
    """Generate a short unique ID for demo data."""
    import uuid
    h = uuid.uuid4().hex[:8]
    return f"{prefix}-{h}"


def safe_get(data: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current
