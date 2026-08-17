"""
Results comparison API for TrustOps Enterprise Framework.

Compares two results side-by-side and returns diff of metrics.

Requirements: 9.8
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ComparisonResult(BaseModel):
    """Side-by-side comparison of two results."""

    result_id_1: str
    result_id_2: str
    metrics_1: dict = Field(default_factory=dict)
    metrics_2: dict = Field(default_factory=dict)
    diff: dict = Field(default_factory=dict)
    common_keys: list[str] = Field(default_factory=list)
    only_in_1: list[str] = Field(default_factory=list)
    only_in_2: list[str] = Field(default_factory=list)


def compare_results(
    result_1: dict,
    result_2: dict,
    result_id_1: str = "result_1",
    result_id_2: str = "result_2",
) -> ComparisonResult:
    """Compare two results side-by-side.

    Computes the diff of numeric metrics between two result dicts.

    Args:
        result_1: First result data.
        result_2: Second result data.
        result_id_1: ID of first result.
        result_id_2: ID of second result.

    Returns:
        ComparisonResult with metrics diff.
    """
    flat_1 = _flatten_dict(result_1)
    flat_2 = _flatten_dict(result_2)

    keys_1 = set(flat_1.keys())
    keys_2 = set(flat_2.keys())
    common = keys_1 & keys_2
    only_1 = keys_1 - keys_2
    only_2 = keys_2 - keys_1

    diff = {}
    for key in sorted(common):
        v1 = flat_1[key]
        v2 = flat_2[key]
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            delta = v2 - v1
            pct = (delta / v1 * 100) if v1 != 0 else 0.0
            diff[key] = {
                "value_1": v1,
                "value_2": v2,
                "delta": round(delta, 6),
                "delta_percent": round(pct, 2),
            }
        elif v1 != v2:
            diff[key] = {
                "value_1": v1,
                "value_2": v2,
                "changed": True,
            }

    return ComparisonResult(
        result_id_1=result_id_1,
        result_id_2=result_id_2,
        metrics_1=flat_1,
        metrics_2=flat_2,
        diff=diff,
        common_keys=sorted(common),
        only_in_1=sorted(only_1),
        only_in_2=sorted(only_2),
    )


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict into dot-separated keys.

    Args:
        d: Dict to flatten.
        parent_key: Prefix for keys.
        sep: Separator between levels.

    Returns:
        Flattened dict.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
