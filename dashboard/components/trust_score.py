"""Trust score drill-down component."""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go

DIMENSION_INFO: dict[str, dict[str, str]] = {
    "accuracy": {
        "label": "Accuracy",
        "icon": "🎯",
        "description": "Measures how closely model responses match expected answers using exact, fuzzy, and semantic matching.",
    },
    "consistency": {
        "label": "Consistency",
        "icon": "🔄",
        "description": "Evaluates response stability by invoking the model multiple times and measuring variance.",
    },
    "safety": {
        "label": "Safety",
        "icon": "🛡️",
        "description": "Checks for harmful content, toxicity, and policy violations in model outputs.",
    },
    "bias": {
        "label": "Bias",
        "icon": "⚖️",
        "description": "Detects demographic bias indicators, stereotyping language, and unbalanced treatment across groups.",
    },
    "context_grounding": {
        "label": "Context Grounding",
        "icon": "📎",
        "description": "Calculates semantic similarity between responses and source documents to measure factual grounding.",
    },
}


def render_trust_radar_chart(
    scores: dict[str, float],
    title: str = "Trust Score Dimensions",
    height: int = 400,
) -> go.Figure:
    """Render a radar chart for trust score dimensions."""
    labels = []
    values = []
    for dim in DIMENSION_INFO:
        info = DIMENSION_INFO[dim]
        labels.append(info["label"])
        values.append(scores.get(dim, 0.0))

    # Close the polygon
    if labels:
        labels.append(labels[0])
        values.append(values[0])

    fig = go.Figure(
        data=go.Scatterpolar(r=values, theta=labels, fill="toself", name="Trust Score"),
    )
    fig.update_layout(
        title=dict(text=title),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=height,
        showlegend=False,
    )
    return fig


def render_trust_distribution_histogram(
    scores: list[float],
    title: str = "Trust Score Distribution",
    height: int = 350,
) -> go.Figure:
    """Render a histogram of trust scores."""
    fig = go.Figure(data=go.Histogram(x=scores, nbinsx=20, marker_color="#1a73e8"))
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title="Trust Score", range=[0, 1]),
        yaxis=dict(title="Count"),
        height=height,
    )
    return fig
