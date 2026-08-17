"""Tests for the dashboard comparison page.

Validates the helper functions, data loading, and rendering logic
for the comparison page (task 21.6, Requirement 10.4).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestComparisonDataLoading:
    """Tests for comparison page data loading."""

    def test_load_comparison_data_falls_back_to_demo(self):
        """When all backends return None, demo data is used."""
        from dashboard.pages.comparison import _load_comparison_data

        with patch("dashboard.utils.backend.get_evaluation_engine", return_value=None), \
             patch("dashboard.utils.backend.get_model_registry", return_value=None):
            data = _load_comparison_data()
            assert data["demo_mode"] is True
            assert len(data["comparisons"]) > 0
            assert len(data["models"]) > 0

    def test_demo_comparisons_have_expected_fields(self):
        """Demo comparisons should have all fields the page uses."""
        from dashboard.pages.comparison import _build_demo_comparisons

        comparisons = _build_demo_comparisons()
        assert len(comparisons) >= 1
        for cmp in comparisons:
            assert "id" in cmp
            assert "model_1_id" in cmp
            assert "model_2_id" in cmp
            assert "model_1_metrics" in cmp
            assert "model_2_metrics" in cmp
            assert "improvement" in cmp
            assert "recommendation" in cmp
            assert "justification" in cmp

    def test_demo_comparisons_have_improvement_fields(self):
        """Improvement metrics should have all required fields."""
        from dashboard.pages.comparison import _build_demo_comparisons

        comparisons = _build_demo_comparisons()
        for cmp in comparisons:
            imp = cmp["improvement"]
            assert "trust_score_delta" in imp
            assert "trust_score_delta_percent" in imp
            assert "hallucination_reduction" in imp
            assert "p_value" in imp
            assert "statistical_significance" in imp
            assert "confidence_interval" in imp

    def test_demo_comparisons_have_model_metrics(self):
        """Model metrics should have trust score, cost, latency."""
        from dashboard.pages.comparison import _build_demo_comparisons

        comparisons = _build_demo_comparisons()
        for cmp in comparisons:
            for key in ("model_1_metrics", "model_2_metrics"):
                m = cmp[key]
                assert "mean_trust_score" in m
                assert "hallucination_rate" in m
                assert "cost_per_query" in m
                assert "total_cost" in m


class TestComparisonRendering:
    """Tests for comparison page render functions."""

    def test_render_comparison_page_callable(self):
        from dashboard.pages.comparison import render_comparison_page
        assert callable(render_comparison_page)

    def test_render_comparison_summary_callable(self):
        from dashboard.pages.comparison import _render_comparison_summary
        assert callable(_render_comparison_summary)

    def test_render_comparison_table_callable(self):
        from dashboard.pages.comparison import _render_comparison_table
        assert callable(_render_comparison_table)

    def test_render_side_by_side_metrics_callable(self):
        from dashboard.pages.comparison import _render_side_by_side_metrics
        assert callable(_render_side_by_side_metrics)

    def test_render_improvement_metrics_callable(self):
        from dashboard.pages.comparison import _render_improvement_metrics
        assert callable(_render_improvement_metrics)

    def test_render_statistical_significance_callable(self):
        from dashboard.pages.comparison import _render_statistical_significance
        assert callable(_render_statistical_significance)

    def test_render_deployment_recommendation_callable(self):
        from dashboard.pages.comparison import _render_deployment_recommendation
        assert callable(_render_deployment_recommendation)

    def test_render_cost_performance_chart_callable(self):
        from dashboard.pages.comparison import _render_cost_performance_chart
        assert callable(_render_cost_performance_chart)


class TestComparisonSummary:
    """Tests for the summary KPI rendering."""

    def test_summary_renders_three_metrics(self):
        from dashboard.pages.comparison import _render_comparison_summary

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(3)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        comparisons = [
            {
                "recommendation": "deploy",
                "improvement": {"trust_score_delta": 0.10},
            },
            {
                "recommendation": "iterate",
                "improvement": {"trust_score_delta": 0.03},
            },
        ]
        _render_comparison_summary(mock_st, comparisons)
        mock_st.columns.assert_called_once_with(3)
        assert mock_st.metric.call_count == 3

    def test_summary_empty_comparisons(self):
        from dashboard.pages.comparison import _render_comparison_summary

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(3)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        _render_comparison_summary(mock_st, [])
        assert mock_st.metric.call_count == 3


class TestComparisonTable:
    """Tests for the comparison listing table."""

    def test_table_empty_shows_info(self):
        from dashboard.pages.comparison import _render_comparison_table

        mock_st = MagicMock()
        result = _render_comparison_table(mock_st, [])
        assert result is None
        mock_st.info.assert_called_once()

    def test_table_returns_none_when_no_selection(self):
        from dashboard.pages.comparison import _render_comparison_table

        mock_st = MagicMock()
        mock_st.columns.return_value = [MagicMock() for _ in range(6)]
        # button returns False (no click)
        for col in mock_st.columns.return_value:
            col.button.return_value = False

        comparisons = [
            {
                "id": "cmp-001",
                "model_1_id": "m1",
                "model_2_id": "m2",
                "improvement": {"trust_score_delta": 0.1},
                "recommendation": "deploy",
                "status": "completed",
            },
        ]
        result = _render_comparison_table(mock_st, comparisons)
        assert result is None


class TestSideBySideMetrics:
    """Tests for side-by-side metrics rendering."""

    def test_side_by_side_renders_rows(self):
        from dashboard.pages.comparison import _render_side_by_side_metrics

        mock_st = MagicMock()
        mock_st.columns.return_value = [MagicMock() for _ in range(3)]

        comparison = {
            "model_1_id": "m1",
            "model_2_id": "m2",
            "model_1_metrics": {
                "mean_trust_score": 0.72,
                "hallucination_rate": 0.18,
                "latency_p50_ms": 320.0,
                "latency_p95_ms": 890.0,
                "cost_per_query": 0.050,
                "total_cost": 12.50,
            },
            "model_2_metrics": {
                "mean_trust_score": 0.85,
                "hallucination_rate": 0.10,
                "latency_p50_ms": 370.0,
                "latency_p95_ms": 940.0,
                "cost_per_query": 0.052,
                "total_cost": 13.00,
            },
        }
        _render_side_by_side_metrics(mock_st, comparison)
        mock_st.subheader.assert_called_once()
        # 1 header + 6 metric rows = 7 calls to st.columns
        assert mock_st.columns.call_count == 7


class TestImprovementMetrics:
    """Tests for improvement metrics rendering."""

    def test_improvement_renders_four_metrics(self):
        from dashboard.pages.comparison import _render_improvement_metrics

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(4)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        comparison = {
            "improvement": {
                "trust_score_delta": 0.13,
                "trust_score_delta_percent": 18.1,
                "hallucination_reduction": 0.08,
                "hallucination_reduction_percent": 44.4,
                "latency_delta_ms": 50,
                "latency_delta_percent": 15.6,
                "cost_delta_per_query": 0.002,
                "cost_delta_percent": 4.0,
            },
        }
        _render_improvement_metrics(mock_st, comparison)
        assert mock_st.metric.call_count == 4

    def test_improvement_empty_shows_info(self):
        from dashboard.pages.comparison import _render_improvement_metrics

        mock_st = MagicMock()
        _render_improvement_metrics(mock_st, {"improvement": {}})
        mock_st.info.assert_called_once()


class TestStatisticalSignificance:
    """Tests for statistical significance rendering."""

    def test_significance_renders_three_metrics(self):
        from dashboard.pages.comparison import _render_statistical_significance

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(4)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        comparison = {
            "improvement": {
                "p_value": 0.003,
                "statistical_significance": 0.997,
                "confidence_interval": (-0.17, -0.09),
            },
        }
        _render_statistical_significance(mock_st, comparison)
        # p-value, test used, 95% CI, secondary test
        assert mock_st.metric.call_count == 4

    def test_significance_significant_label(self):
        from dashboard.pages.comparison import _render_statistical_significance

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(4)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        comparison = {
            "improvement": {
                "p_value": 0.003,
                "statistical_significance": 0.997,
                "confidence_interval": (-0.17, -0.09),
            },
        }
        _render_statistical_significance(mock_st, comparison)
        # Check that caption was called with "Significant"
        mock_st.caption.assert_called_once()
        caption_text = mock_st.caption.call_args[0][0]
        assert "Significant" in caption_text


class TestDeploymentRecommendation:
    """Tests for deployment recommendation rendering."""

    def test_deploy_recommendation(self):
        from dashboard.pages.comparison import _render_deployment_recommendation

        mock_st = MagicMock()
        comparison = {
            "recommendation": "deploy",
            "justification": "All thresholds met.",
        }
        _render_deployment_recommendation(mock_st, comparison)
        mock_st.subheader.assert_called_once()
        mock_st.markdown.assert_called_once()
        assert "DEPLOY" in mock_st.markdown.call_args[0][0]
        mock_st.info.assert_called_once_with("All thresholds met.")

    def test_iterate_recommendation(self):
        from dashboard.pages.comparison import _render_deployment_recommendation

        mock_st = MagicMock()
        comparison = {
            "recommendation": "iterate",
            "justification": "Needs more tuning.",
        }
        _render_deployment_recommendation(mock_st, comparison)
        assert "ITERATE" in mock_st.markdown.call_args[0][0]

    def test_reject_recommendation(self):
        from dashboard.pages.comparison import _render_deployment_recommendation

        mock_st = MagicMock()
        comparison = {
            "recommendation": "reject",
            "justification": "Trust score decreased.",
        }
        _render_deployment_recommendation(mock_st, comparison)
        assert "REJECT" in mock_st.markdown.call_args[0][0]


class TestCostPerformanceChart:
    """Tests for cost-performance chart rendering."""

    def test_chart_renders_plotly_figure(self):
        from dashboard.pages.comparison import _render_cost_performance_chart
        import plotly.graph_objects as go

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(2)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        comparison = {
            "model_1_id": "m1",
            "model_2_id": "m2",
            "model_1_metrics": {
                "mean_trust_score": 0.72,
                "cost_per_query": 0.050,
                "hallucination_rate": 0.18,
            },
            "model_2_metrics": {
                "mean_trust_score": 0.85,
                "cost_per_query": 0.052,
                "hallucination_rate": 0.10,
            },
        }
        _render_cost_performance_chart(mock_st, comparison)
        mock_st.plotly_chart.assert_called_once()
        fig = mock_st.plotly_chart.call_args[0][0]
        assert isinstance(fig, go.Figure)

    def test_chart_insufficient_data_shows_info(self):
        from dashboard.pages.comparison import _render_cost_performance_chart

        mock_st = MagicMock()
        comparison = {
            "model_1_metrics": {},
            "model_2_metrics": {},
        }
        _render_cost_performance_chart(mock_st, comparison)
        mock_st.info.assert_called_once()
        mock_st.plotly_chart.assert_not_called()
