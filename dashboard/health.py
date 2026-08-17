"""Health check endpoint for the dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone


def get_health_status() -> dict:
    """Return the current health status of the dashboard."""
    checks: dict[str, str] = {}

    # Check streamlit
    try:
        import streamlit  # noqa: F401
        checks["streamlit"] = "ok"
    except ImportError:
        checks["streamlit"] = "missing"

    # Check plotly
    try:
        import plotly  # noqa: F401
        checks["plotly"] = "ok"
    except ImportError:
        checks["plotly"] = "missing"

    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "healthy" if all_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "checks": checks,
    }


def health_check_json() -> str:
    """Return health status as a JSON string."""
    return json.dumps(get_health_status(), indent=2)
