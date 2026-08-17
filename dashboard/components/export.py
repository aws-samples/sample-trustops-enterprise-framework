"""Results export component."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any


def _to_csv_bytes(data: list[dict]) -> bytes:
    """Convert a list of dicts to CSV bytes."""
    if not data:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    return buf.getvalue().encode("utf-8")


class _JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def _to_json_bytes(data: Any) -> bytes:
    """Convert data to pretty-printed JSON bytes."""
    return json.dumps(data, indent=2, cls=_JSONEncoder).encode("utf-8")


def _pdf_escape(text: str) -> str:
    """Escape special characters for PDF text streams."""
    text = text.replace("\\", "\\\\")
    text = text.replace("(", "\\(")
    text = text.replace(")", "\\)")
    text = text.replace("\n", " ").replace("\r", " ")
    return text


def _to_pdf_bytes(data: Any, title: str = "TrustOps Report") -> bytes:
    """Generate a minimal valid PDF containing the data as text."""
    if isinstance(data, (dict, list)):
        body = json.dumps(data, indent=2, cls=_JSONEncoder)
    else:
        body = str(data)

    escaped_title = _pdf_escape(title)
    escaped_body = _pdf_escape(body)

    # Minimal PDF structure
    objects: list[str] = []
    # obj 1 – catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")
    # obj 2 – pages
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj")
    # obj 3 – page
    objects.append(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj"
    )
    # obj 4 – content stream
    stream_content = f"BT /F1 16 Tf 50 750 Td ({escaped_title}) Tj ET\n"
    # Split body into lines for the PDF
    lines = escaped_body.split(" ")
    y = 720
    line_buf = ""
    for word in lines:
        if len(line_buf) + len(word) > 80:
            stream_content += f"BT /F1 10 Tf 50 {y} Td ({_pdf_escape(line_buf.strip())}) Tj ET\n"
            y -= 14
            line_buf = word + " "
            if y < 50:
                break
        else:
            line_buf += word + " "
    if line_buf.strip():
        stream_content += f"BT /F1 10 Tf 50 {y} Td ({_pdf_escape(line_buf.strip())}) Tj ET\n"

    objects.append(
        f"4 0 obj\n<< /Length {len(stream_content)} >>\nstream\n{stream_content}endstream\nendobj"
    )
    # obj 5 – font
    objects.append(
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj"
    )

    # Build PDF bytes
    pdf = "%PDF-1.4\n"
    offsets: list[int] = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj + "\n"

    xref_offset = len(pdf)
    pdf += "xref\n"
    pdf += f"0 {len(objects) + 1}\n"
    pdf += "0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n"
    pdf += "trailer\n"
    pdf += f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    pdf += "startxref\n"
    pdf += f"{xref_offset}\n"
    pdf += "%%EOF\n"

    return pdf.encode("latin-1")
