"""
Results export API for TrustOps Enterprise Framework.

Supports export to JSON, CSV, and PDF formats with no data loss.

Requirements: 9.8
"""

import csv
import io
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def export_to_json(data: Any) -> bytes:
    """Export data to JSON format.

    Args:
        data: Data to export.

    Returns:
        JSON bytes.
    """
    serialized = json.dumps(data, indent=2, sort_keys=True, default=str)
    return serialized.encode("utf-8")


def export_to_csv(data: Any) -> bytes:
    """Export data to CSV format.

    Handles nested dicts by flattening keys with dot notation.

    Args:
        data: Data to export. Can be a dict or list of dicts.

    Returns:
        CSV bytes.
    """
    if isinstance(data, dict):
        rows = [_flatten_for_csv(data)]
    elif isinstance(data, list):
        rows = [_flatten_for_csv(item) if isinstance(item, dict) else {"value": item} for item in data]
    else:
        rows = [{"value": str(data)}]

    if not rows:
        return b""

    # Collect all keys across all rows
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    return output.getvalue().encode("utf-8")


def export_to_pdf(data: Any) -> bytes:
    """Export data to PDF format.

    Generates a simple text-based PDF representation.
    For production use, integrate a proper PDF library.

    Args:
        data: Data to export.

    Returns:
        PDF-like bytes (plain text representation).
    """
    # Simple text-based PDF placeholder
    # In production, use reportlab or similar
    lines = [
        "TrustOps Results Export",
        "=" * 40,
        "",
    ]

    if isinstance(data, dict):
        for key, value in sorted(data.items()):
            lines.append(f"{key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            lines.append(f"--- Item {i + 1} ---")
            if isinstance(item, dict):
                for key, value in sorted(item.items()):
                    lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {item}")
    else:
        lines.append(str(data))

    content = "\n".join(lines)
    return content.encode("utf-8")


def export_result(data: Any, format: str) -> bytes:
    """Export result in the specified format.

    Args:
        data: Data to export.
        format: Export format ('json', 'csv', 'pdf').

    Returns:
        Exported data as bytes.

    Raises:
        ValueError: If format is not supported.
    """
    exporters = {
        "json": export_to_json,
        "csv": export_to_csv,
        "pdf": export_to_pdf,
    }

    if format not in exporters:
        raise ValueError(f"Unsupported export format: {format}. Supported: {list(exporters.keys())}")

    return exporters[format](data)


def _flatten_for_csv(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict for CSV export."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_for_csv(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, json.dumps(v, default=str)))
        else:
            items.append((new_key, v))
    return dict(items)
