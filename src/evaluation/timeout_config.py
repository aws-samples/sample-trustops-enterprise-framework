"""
Per-model timeout configuration with size-based defaults.

Provides a TimeoutConfig dataclass with default timeouts for small, medium,
and large models, and a resolve_timeout() function that determines the
appropriate timeout for a given model ID.

Requirements: 3.11, 3.12, 3.18, 3.19
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Keywords used to classify model size from model ID strings.
_SMALL_KEYWORDS: frozenset[str] = frozenset(
    {"lite", "mini", "small", "nano", "tiny", "haiku", "instant"}
)
_LARGE_KEYWORDS: frozenset[str] = frozenset(
    {"large", "xl", "xxl", "ultra", "opus", "70b", "65b", "180b"}
)


@dataclass(frozen=True)
class TimeoutConfig:
    """Per-model-size timeout defaults in seconds.

    Attributes:
        small: Timeout for small/lite models (default 30s).
        medium: Timeout for medium models (default 60s).
        large: Timeout for large/xl models (default 120s).
        model_overrides: Optional per-model-id timeout overrides.
    """

    small: float = 30.0
    medium: float = 60.0
    large: float = 120.0
    model_overrides: dict[str, float] = field(default_factory=dict)


def _classify_model_size(model_id: str) -> str:
    """Classify a model as small, medium, or large based on its ID.

    The classification uses keyword matching against the lowercased model ID.
    If no keywords match, defaults to ``"medium"``.

    Args:
        model_id: The model identifier string.

    Returns:
        One of ``"small"``, ``"medium"``, or ``"large"``.
    """
    lower_id = model_id.lower()
    for keyword in _SMALL_KEYWORDS:
        if keyword in lower_id:
            return "small"
    for keyword in _LARGE_KEYWORDS:
        if keyword in lower_id:
            return "large"
    return "medium"


def resolve_timeout(
    model_id: str,
    config: TimeoutConfig | None = None,
) -> float:
    """Return the appropriate timeout in seconds for *model_id*.

    Resolution order:
    1. Explicit per-model override in ``config.model_overrides``.
    2. Size-based default from ``config`` (small / medium / large).
    3. If *config* is ``None``, uses the built-in ``TimeoutConfig`` defaults.

    Args:
        model_id: The model identifier string.
        config: Optional timeout configuration. Uses defaults when ``None``.

    Returns:
        Timeout value in seconds.
    """
    if config is None:
        config = TimeoutConfig()

    # 1. Check explicit override
    if model_id in config.model_overrides:
        timeout = config.model_overrides[model_id]
        logger.debug(
            "Using override timeout %.1fs for model %s",
            timeout,
            model_id,
        )
        return timeout

    # 2. Size-based default
    size = _classify_model_size(model_id)
    timeout = getattr(config, size)
    logger.debug(
        "Resolved timeout %.1fs for model %s (size=%s)",
        timeout,
        model_id,
        size,
    )
    return timeout
