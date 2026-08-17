"""Tests for the hallucination visualization component."""

import pytest

from dashboard.components.hallucination import (
    render_grounding_score_gauge,
    highlight_unsupported_claims,
)


class TestRenderGroundingScoreGauge:
    """Tests for the grounding score gauge."""

    def test_returns_figure(self):
        fig = render_grounding_score_gauge(0.75)
        assert fig is not None
        assert hasattr(fig, "data")

    def test_high_score_color(self):
        fig = render_grounding_score_gauge(0.85)
        bar_color = fig.data[0].gauge.bar.color
        assert bar_color == "#2ecc71"  # green

    def test_medium_score_color(self):
        fig = render_grounding_score_gauge(0.55)
        bar_color = fig.data[0].gauge.bar.color
        assert bar_color == "#f39c12"  # orange

    def test_low_score_color(self):
        fig = render_grounding_score_gauge(0.2)
        bar_color = fig.data[0].gauge.bar.color
        assert bar_color == "#e74c3c"  # red

    def test_custom_title(self):
        fig = render_grounding_score_gauge(0.5, title="Custom Gauge")
        assert fig.data[0].title.text == "Custom Gauge"

    def test_custom_height(self):
        fig = render_grounding_score_gauge(0.5, height=300)
        assert fig.layout.height == 300

    def test_value_is_percentage(self):
        fig = render_grounding_score_gauge(0.75)
        assert fig.data[0].value == 75.0

    def test_zero_score(self):
        fig = render_grounding_score_gauge(0.0)
        assert fig.data[0].value == 0.0

    def test_perfect_score(self):
        fig = render_grounding_score_gauge(1.0)
        assert fig.data[0].value == 100.0


class TestHighlightUnsupportedClaims:
    """Tests for text highlighting."""

    def test_no_spans_returns_original(self):
        text = "This is a normal response."
        result = highlight_unsupported_claims(text, [])
        assert result == text

    def test_empty_spans_returns_original(self):
        text = "Hello world"
        result = highlight_unsupported_claims(text, [])
        assert result == text

    def test_single_span_highlighted(self):
        text = "The sky is green and the grass is blue."
        spans = [
            {"start_idx": 0, "end_idx": 16, "grounding_score": 0.2, "reason": "Incorrect color"},
        ]
        result = highlight_unsupported_claims(text, spans)
        assert "<span" in result
        assert "The sky is green" in result
        assert "background-color" in result

    def test_multiple_spans(self):
        text = "Claim one is here. Claim two is here."
        spans = [
            {"start_idx": 0, "end_idx": 18, "grounding_score": 0.3, "reason": "Unsupported"},
            {"start_idx": 19, "end_idx": 37, "grounding_score": 0.2, "reason": "Unsupported"},
        ]
        result = highlight_unsupported_claims(text, spans)
        assert result.count("<span") == 2

    def test_span_includes_score_in_title(self):
        text = "The earth is flat."
        spans = [
            {"start_idx": 0, "end_idx": 18, "grounding_score": 0.1, "reason": "False claim"},
        ]
        result = highlight_unsupported_claims(text, spans)
        assert "0.10" in result
        assert "False claim" in result

    def test_invalid_span_indices_ignored(self):
        text = "Short text"
        spans = [
            {"start_idx": 50, "end_idx": 100, "grounding_score": 0.1, "reason": "Out of range"},
        ]
        result = highlight_unsupported_claims(text, spans)
        assert result == text

    def test_span_with_start_after_end_ignored(self):
        text = "Some text here"
        spans = [
            {"start_idx": 10, "end_idx": 5, "grounding_score": 0.1, "reason": "Bad range"},
        ]
        result = highlight_unsupported_claims(text, spans)
        assert result == text
