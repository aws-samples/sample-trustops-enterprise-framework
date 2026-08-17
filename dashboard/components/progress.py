"""Real-time progress display component."""

from __future__ import annotations

import time
from typing import Optional


def estimate_completion_time(
    current: int,
    total: int,
    start_time: float,
) -> Optional[float]:
    """Estimate remaining seconds based on progress so far.

    Returns ``None`` when an estimate cannot be computed.
    """
    if current <= 0 or total <= 0:
        return None

    elapsed = time.time() - start_time
    if elapsed <= 0:
        return None

    rate = current / elapsed  # items per second
    remaining = total - current
    return remaining / rate
