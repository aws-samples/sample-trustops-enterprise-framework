"""Tests for the trust score drill-down component."""

import pytest

from dashboard.components.trust_score import (
    DIMENSION_INFO,
    render_trust_radar_chart,
    render_trust_distribution_histogram,
)


class TestDimensionInfo:
    """Tests for dimension metadata."""

    def test_has_five_dimensions(self):
        expected = {"accuracy", "consistency", "safety", "bias", "context_grounding"}
        assert set(DIMENSION_INFO.keys()) == expected

    def test_each_dimension_has_label(self):
        for key, info in DIMENSION_INFO.items():
            assert "label" in info
            assert isinstance(info["label"], str)
            assert len(info["label"]) > 0

    def test_each_dimension_has_icon(self):
        for key, info in DIMENSION_INFO.items():
            assert "icon" in info

    def test_each_dimension_has_description(self):
        for key, info in DIMENSION_INFO.items():
            assert "description" in info
            assert len(info["description"]) > 10


class TestRenderTrustRadarChart:
    """Tests for the radar chart component."""

    def test_returns_figure(self):
        scores = {
            "accuracy": 0.8,
            "consistency": 0.7,
            "safety": 0.9,
            "bias": 0.6,
            "context_grounding": 0.75,
        }
        fig = render_trust_radar_chart(scores)
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) > 0

    def test_custom_title(self):
        scores = {"accuracy": 0.5, "consistency": 0.5, "safety": 0.5, "bias": 0.5, "context_grounding": 0.5}
        fig = render_trust_radar_chart(scores, title="Custom Title")
        assert fig.layout.title.text == "Custom Title"

    def test_custom_height(self):
        scores = {"accuracy": 0.5}
        fig = render_trust_radar_chart(scores, height=500)
        assert fig.layout.height == 500

    def test_handles_missing_dimensions(self):
        scores = {"accuracy": 0.8}
        fig = render_trust_radar_chart(scores)
        assert fig is not None

    def test_handles_empty_scores(self):
        fig = render_trust_radar_chart({})
        assert fig is not None

    def test_radar_data_is_closed_polygon(self):
        scores = {
            "accuracy": 0.8,
            "consistency": 0.7,
            "safety": 0.9,
            "bias": 0.6,
            "context_grounding": 0.75,
        }
        fig = render_trust_radar_chart(scores)
        r_values = fig.data[0].r
        theta_values = fig.data[0].theta
        # Closed polygon: first and last values should match
        assert r_values[0] == r_values[-1]
        assert theta_values[0] == theta_values[-1]


class TestRenderTrustDistributionHistogram:
    """Tests for the distribution histogram."""

    def test_returns_figure(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        fig = render_trust_distribution_histogram(scores)
        assert fig is not None
        assert hasattr(fig, "data")

    def test_empty_scores(self):
        fig = render_trust_distribution_histogram([])
        assert fig is not None

    def test_custom_title(self):
        fig = render_trust_distribution_histogram([0.5], title="My Histogram")
        assert fig.layout.title.text == "My Histogram"

    def test_custom_height(self):
        fig = render_trust_distribution_histogram([0.5], height=400)
        assert fig.layout.height == 400

    def test_x_axis_range(self):
        fig = render_trust_distribution_histogram([0.5])
        assert tuple(fig.layout.xaxis.range) == (0, 1)
