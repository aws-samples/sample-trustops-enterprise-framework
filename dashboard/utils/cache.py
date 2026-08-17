"""Dashboard caching and performance utilities."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional


def paginate_data(data: list, page: int = 1, page_size: int = 25) -> tuple[list, int]:
    """Return a page slice and total page count."""
    total_pages = max(1, math.ceil(len(data) / page_size))
    start = (page - 1) * page_size
    end = min(start + page_size, len(data))
    return data[start:end], total_pages


def cached_load_models() -> list[dict]:
    """Load models with caching (placeholder)."""
    return []


def cached_load_datasets() -> list[dict]:
    """Load datasets with caching (placeholder)."""
    return []


def cached_load_evaluations() -> list[dict]:
    """Load evaluations with caching (placeholder)."""
    return []


def cached_load_workflows() -> list[dict]:
    """Load workflows with caching (placeholder)."""
    return []


def clear_all_caches() -> None:
    """Clear all cached data."""
    pass


def lazy_load_data(loader: Callable, *args: Any, **kwargs: Any) -> Any:
    """Lazy-load data using the provided loader function."""
    return loader(*args, **kwargs)
