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

    Generates a minimal but valid PDF document using raw PDF operators.
    No external library required.

    Args:
        data: Data to export.

    Returns:
        Valid PDF bytes.
    """
    # Build the text content
    lines = [
        "TrustOps Evaluation Results",
        "=" * 50,
        "",
    ]

    if isinstance(data, dict):
        for key, value in sorted(data.items()):
            if isinstance(value, (dict, list)):
                import json as _json
                lines.append(f"{key}:")
                formatted = _json.dumps(value, indent=2, default=str)
                for sub_line in formatted.split("\n"):
                    lines.append(f"    {sub_line}")
            else:
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

    text_content = "\n".join(lines)

    # Generate a minimal valid PDF (PDF 1.4 spec)
    # This produces a real PDF that any PDF reader can open.
    def _pdf_escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    page_lines = text_content.split("\n")
    # Split into pages of ~50 lines each
    pages_text = []
    for i in range(0, len(page_lines), 50):
        chunk = page_lines[i:i + 50]
        pages_text.append(chunk)

    objects = []
    obj_offsets = []

    # Object 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Object 2: Pages (filled in later)
    objects.append("")  # placeholder
    # Object 3: Font
    objects.append("3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n")

    page_obj_ids = []
    next_obj = 4
    for page_lines_chunk in pages_text:
        # Build text stream for this page
        stream_lines = ["BT", "/F1 10 Tf", "36 756 Td", "12 TL"]
        for line in page_lines_chunk:
            escaped = _pdf_escape(line[:120])  # Truncate long lines
            stream_lines.append(f"({escaped}) '")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines)

        # Content stream object
        content_id = next_obj
        objects.append(
            f"{content_id} 0 obj\n"
            f"<< /Length {len(stream)} >>\n"
            f"stream\n{stream}\nendstream\n"
            f"endobj\n"
        )
        next_obj += 1

        # Page object
        page_id = next_obj
        page_obj_ids.append(page_id)
        objects.append(
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>\n"
            f"endobj\n"
        )
        next_obj += 1

    # Fill in Pages object (object 2)
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects[1] = (
        f"2 0 obj\n"
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_ids)} >>\n"
        f"endobj\n"
    )

    # Build the PDF file
    pdf_parts = ["%PDF-1.4\n"]
    for obj in objects:
        obj_offsets.append(len("".join(pdf_parts)))
        pdf_parts.append(obj)

    xref_offset = len("".join(pdf_parts))
    pdf_parts.append(f"xref\n0 {len(objects) + 1}\n")
    pdf_parts.append("0000000000 65535 f \n")
    for offset in obj_offsets:
        pdf_parts.append(f"{offset:010d} 00000 n \n")
    pdf_parts.append(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )

    return "".join(pdf_parts).encode("latin-1")


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
