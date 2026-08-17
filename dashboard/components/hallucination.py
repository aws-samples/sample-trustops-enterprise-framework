"""Hallucination visualization component."""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go


def render_grounding_score_gauge(
    score: float,
    title: str = "Grounding Score",
    height: int = 250,
) -> go.Figure:
    """Render a gauge chart for a grounding score (0-1)."""
    pct = score * 100.0
    if score >= 0.7:
        color = "#2ecc71"
    elif score >= 0.4:
        color = "#f39c12"
    else:
        color = "#e74c3c"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
            },
        )
    )
    fig.update_layout(height=height)
    return fig


def highlight_unsupported_claims(text: str, spans: list[dict]) -> str:
    """Return HTML with unsupported claim spans highlighted."""
    if not spans:
        return text

    # Filter and sort valid spans
    valid = [
        s for s in spans
        if s.get("start_idx", 0) < s.get("end_idx", 0)
        and s.get("start_idx", len(text)) < len(text)
        and s.get("end_idx", 0) <= len(text)
    ]
    valid.sort(key=lambda s: s["start_idx"])

    if not valid:
        return text

    parts: list[str] = []
    prev_end = 0
    for span in valid:
        start = span["start_idx"]
        end = span["end_idx"]
        score = span.get("grounding_score", 0)
        reason = span.get("reason", "")
        parts.append(text[prev_end:start])
        parts.append(
            f'<span style="background-color: rgba(231,76,60,0.25);" '
            f'title="Score: {score:.2f} — {reason}">'
            f"{text[start:end]}</span>"
        )
        prev_end = end
    parts.append(text[prev_end:])
    return "".join(parts)
