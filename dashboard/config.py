"""Dashboard configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DashboardConfig:
    """Central configuration for the TrustOps dashboard."""

    page_title: str = "TrustOps Enterprise"
    page_icon: str = "🛡️"
    layout: str = "wide"
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    # No default bucket name: S3 names are globally unique, so a predictable
    # default could be squatted by another account. Falls back to the shared
    # TRUSTOPS_RESULTS_BUCKET setting, then to empty, which the dashboard
    # surfaces as a configuration error rather than reading a foreign bucket.
    s3_bucket: str = field(
        default_factory=lambda: os.getenv("TRUSTOPS_S3_BUCKET")
        or os.getenv("TRUSTOPS_RESULTS_BUCKET", "")
    )
    cache_ttl: int = 300
    pages: dict[str, str] = field(default_factory=lambda: {
        "Home": "🏠",
        "Models": "🤖",
        "Datasets": "📊",
        "Evaluation": "📈",
        "Comparison": "⚖️",
        "Fine-Tuning": "🔧",
        "Workflows": "🔄",
    })


_config: Optional[DashboardConfig] = None


def get_config() -> DashboardConfig:
    """Return singleton dashboard config."""
    global _config
    if _config is None:
        _config = DashboardConfig()
    return _config
