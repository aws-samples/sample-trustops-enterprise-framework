"""Tests for the dashboard home page.

Validates the helper functions, data loading, and rendering logic
for the home page (task 21.2, Requirement 10.2).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestHomeHelpers:
    """Tests for home page helper functions."""

    def test_count_by_status_default_key(self):
        from dashboard.pages.home import _count_by_status
        items = [
            {"status": "active"},
            {"status": "active"},
            {"status": "inactive"},
        ]
        result = _count_by_status(items)
        assert result == {"active": 2, "inactive": 1}

    def test_count_by_status_custom_key(self):
        from dashboard.pages.home import _count_by_status
        items = [
            {"provider": "bedrock"},
            {"provider": "bedrock"},
            {"provider": "sagemaker"},
        ]
        result = _count_by_status(items, key="provider")
        assert result == {"bedrock": 2, "sagemaker": 1}

    def test_count_by_status_missing_key(self):
        from dashboard.pages.home import _count_by_status
        items = [{"name": "foo"}, {"status": "active"}]
        result = _count_by_status(items)
        assert result == {"unknown": 1, "active": 1}

    def test_count_by_status_empty(self):
        from dashboard.pages.home import _count_by_status
        assert _count_by_status([]) == {}

    def test_total_cost(self):
        from dashboard.pages.home import _total_cost
        evals = [
            {"total_cost": 10.0},
            {"total_cost": 5.5},
            {"total_cost": 2.0},
        ]
        assert _total_cost(evals) == pytest.approx(17.5)

    def test_total_cost_empty(self):
        from dashboard.pages.home import _total_cost
        assert _total_cost([]) == 0.0

    def test_total_cost_missing_field(self):
        from dashboard.pages.home import _total_cost
        evals = [{"id": "e1"}, {"total_cost": 3.0}]
        assert _total_cost(evals) == pytest.approx(3.0)

    def test_avg_trust_score(self):
        from dashboard.pages.home import _avg_trust_score
        evals = [
            {"mean_trust_score": 0.8},
            {"mean_trust_score": 0.6},
        ]
        assert _avg_trust_score(evals) == pytest.approx(0.7)

    def test_avg_trust_score_empty(self):
        from dashboard.pages.home import _avg_trust_score
        assert _avg_trust_score([]) is None

    def test_avg_trust_score_no_scores(self):
        from dashboard.pages.home import _avg_trust_score
        evals = [{"id": "e1"}, {"id": "e2"}]
        assert _avg_trust_score(evals) is None


class TestHomeDataLoading:
    """Tests for the home page data loading."""

    def test_load_home_data_falls_back_to_demo(self):
        """When all backends return None, demo data is used."""
        from dashboard.pages.home import _load_home_data

        with patch("dashboard.utils.backend.get_model_registry", return_value=None), \
             patch("dashboard.utils.backend.get_dataset_manager", return_value=None), \
             patch("dashboard.utils.backend.get_evaluation_engine", return_value=None), \
             patch("dashboard.utils.backend.get_workflow_orchestrator", return_value=None):
            data = _load_home_data()
            assert data["demo_mode"] is True
            assert len(data["models"]) > 0
            assert len(data["datasets"]) > 0
            assert len(data["evaluations"]) > 0
            assert len(data["workflows"]) > 0

    def test_load_home_data_models_have_expected_fields(self):
        """Demo models should have the fields the home page uses."""
        from dashboard.pages.home import _load_home_data
        data = _load_home_data()
        for model in data["models"]:
            assert "id" in model
            assert "provider" in model
            assert "status" in model

    def test_load_home_data_evaluations_have_expected_fields(self):
        """Demo evaluations should have the fields the home page uses."""
        from dashboard.pages.home import _load_home_data
        data = _load_home_data()
        for ev in data["evaluations"]:
            assert "id" in ev
            assert "model_id" in ev
            assert "mean_trust_score" in ev


class TestHomeRendering:
    """Tests for the home page render function (without Streamlit context)."""

    def test_render_home_page_callable(self):
        from dashboard.pages.home import render_home_page
        assert callable(render_home_page)

    def test_render_kpi_row_callable(self):
        from dashboard.pages.home import _render_kpi_row
        assert callable(_render_kpi_row)

    def test_render_system_health_callable(self):
        from dashboard.pages.home import _render_system_health
        assert callable(_render_system_health)

    def test_render_cost_summary_callable(self):
        from dashboard.pages.home import _render_cost_summary
        assert callable(_render_cost_summary)

    def test_render_recent_evaluations_callable(self):
        from dashboard.pages.home import _render_recent_evaluations
        assert callable(_render_recent_evaluations)

    def test_render_recent_workflows_callable(self):
        from dashboard.pages.home import _render_recent_workflows
        assert callable(_render_recent_workflows)

    def test_render_model_provider_chart_callable(self):
        from dashboard.pages.home import _render_model_provider_chart
        assert callable(_render_model_provider_chart)

    def test_render_trust_score_overview_callable(self):
        from dashboard.pages.home import _render_trust_score_overview
        assert callable(_render_trust_score_overview)


class TestHomeCharts:
    """Tests for chart generation functions."""

    def test_model_provider_chart_with_data(self):
        """Provider chart should produce a plotly figure."""
        from dashboard.pages.home import _render_model_provider_chart
        import plotly.graph_objects as go

        # Use a mock st that captures plotly_chart calls
        mock_st = MagicMock()
        models = [
            {"provider": "bedrock"},
            {"provider": "bedrock"},
            {"provider": "sagemaker"},
        ]
        _render_model_provider_chart(mock_st, models)
        mock_st.plotly_chart.assert_called_once()
        fig = mock_st.plotly_chart.call_args[0][0]
        assert isinstance(fig, go.Figure)

    def test_model_provider_chart_empty(self):
        """Provider chart should not render with empty data."""
        from dashboard.pages.home import _render_model_provider_chart
        mock_st = MagicMock()
        _render_model_provider_chart(mock_st, [])
        mock_st.plotly_chart.assert_not_called()

    def test_trust_score_overview_with_data(self):
        """Trust score chart should produce a plotly figure."""
        from dashboard.pages.home import _render_trust_score_overview
        import plotly.graph_objects as go

        mock_st = MagicMock()
        evals = [
            {"id": "e1", "mean_trust_score": 0.85},
            {"id": "e2", "mean_trust_score": 0.65},
        ]
        _render_trust_score_overview(mock_st, evals)
        mock_st.plotly_chart.assert_called_once()
        fig = mock_st.plotly_chart.call_args[0][0]
        assert isinstance(fig, go.Figure)

    def test_trust_score_overview_empty(self):
        """Trust score chart should not render with no scored evals."""
        from dashboard.pages.home import _render_trust_score_overview
        mock_st = MagicMock()
        _render_trust_score_overview(mock_st, [])
        mock_st.plotly_chart.assert_not_called()


class TestHomeKPIRow:
    """Tests for the KPI row rendering."""

    def test_kpi_row_renders_four_metrics(self):
        from dashboard.pages.home import _render_kpi_row

        mock_st = MagicMock()
        # st.columns returns context managers
        col_mocks = [MagicMock() for _ in range(4)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        models = [{"status": "active"}, {"status": "inactive"}]
        datasets = [{"row_count": 100}]
        evals = [{"mean_trust_score": 0.8, "total_cost": 5.0}]
        workflows = [{"status": "running"}]

        _render_kpi_row(mock_st, models, datasets, evals, workflows)

        # Verify columns were requested
        mock_st.columns.assert_called_once_with(4)
        # st.metric is called 4 times (once per KPI)
        assert mock_st.metric.call_count == 4
