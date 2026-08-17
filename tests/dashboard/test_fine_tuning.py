"""Tests for the dashboard fine-tuning page.

Validates helper functions, data loading, cost estimation, and
rendering logic for the fine-tuning page (task 21.7, Requirement 10.5).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestFineTuningHelpers:
    """Tests for fine-tuning page helper functions."""

    def test_running_jobs_filters_running(self):
        from dashboard.pages.fine_tuning import _running_jobs
        jobs = [
            {"id": "j1", "status": "running"},
            {"id": "j2", "status": "completed"},
            {"id": "j3", "status": "training"},
        ]
        result = _running_jobs(jobs)
        assert len(result) == 2
        ids = {j["id"] for j in result}
        assert ids == {"j1", "j3"}

    def test_running_jobs_empty(self):
        from dashboard.pages.fine_tuning import _running_jobs
        assert _running_jobs([]) == []

    def test_running_jobs_none_running(self):
        from dashboard.pages.fine_tuning import _running_jobs
        jobs = [
            {"id": "j1", "status": "completed"},
            {"id": "j2", "status": "failed"},
        ]
        assert _running_jobs(jobs) == []

    def test_estimate_cost_known_model(self):
        from dashboard.pages.fine_tuning import _estimate_cost
        result = _estimate_cost("bedrock-claude-3", 1000, 3)
        assert result["estimated_training_cost"] > 0
        assert result["confidence"] == "high"
        assert result["currency"] == "USD"
        assert "compute" in result["cost_breakdown"]

    def test_estimate_cost_unknown_model(self):
        from dashboard.pages.fine_tuning import _estimate_cost
        result = _estimate_cost("unknown-model", 2000, 2)
        assert result["estimated_training_cost"] > 0
        assert result["confidence"] == "medium"

    def test_estimate_cost_scales_with_epochs(self):
        from dashboard.pages.fine_tuning import _estimate_cost
        r1 = _estimate_cost("bedrock-claude-3", 1000, 1)
        r3 = _estimate_cost("bedrock-claude-3", 1000, 3)
        assert r3["estimated_training_cost"] > r1["estimated_training_cost"]


class TestFineTuningDataLoading:
    """Tests for the fine-tuning page data loading."""

    def test_load_data_falls_back_to_demo(self):
        from dashboard.pages.fine_tuning import _load_fine_tuning_data

        with (
            patch(
                "dashboard.utils.backend.get_fine_tuning_pipeline",
                return_value=None,
            ),
            patch(
                "dashboard.utils.backend.get_model_registry",
                return_value=None,
            ),
            patch(
                "dashboard.utils.backend.get_dataset_manager",
                return_value=None,
            ),
        ):
            data = _load_fine_tuning_data()
            assert data["demo_mode"] is True
            assert len(data["jobs"]) > 0
            assert len(data["models"]) > 0
            assert len(data["datasets"]) > 0

    def test_demo_jobs_have_expected_fields(self):
        from dashboard.pages.fine_tuning import _build_demo_jobs
        jobs = _build_demo_jobs()
        for job in jobs:
            assert "id" in job
            assert "status" in job
            assert "training_metrics" in job

    def test_demo_jobs_have_enriched_fields(self):
        from dashboard.pages.fine_tuning import _build_demo_jobs
        jobs = _build_demo_jobs()
        completed = [
            j for j in jobs
            if j.get("status") == "completed"
        ]
        assert len(completed) > 0
        for job in completed:
            assert "hyperparameters" in job
            assert "estimated_cost" in job


class TestFineTuningRendering:
    """Tests for rendering functions (without Streamlit context)."""

    def test_render_fine_tuning_page_callable(self):
        from dashboard.pages.fine_tuning import render_fine_tuning_page
        assert callable(render_fine_tuning_page)

    def test_render_job_summary_callable(self):
        from dashboard.pages.fine_tuning import _render_job_summary
        assert callable(_render_job_summary)

    def test_render_job_table_callable(self):
        from dashboard.pages.fine_tuning import _render_job_table
        assert callable(_render_job_table)

    def test_render_wizard_callable(self):
        from dashboard.pages.fine_tuning import _render_wizard
        assert callable(_render_wizard)

    def test_render_cost_estimate_callable(self):
        from dashboard.pages.fine_tuning import _render_cost_estimate
        assert callable(_render_cost_estimate)

    def test_render_training_progress_callable(self):
        from dashboard.pages.fine_tuning import _render_training_progress
        assert callable(_render_training_progress)

    def test_render_loss_chart_callable(self):
        from dashboard.pages.fine_tuning import _render_loss_chart
        assert callable(_render_loss_chart)

    def test_render_job_detail_callable(self):
        from dashboard.pages.fine_tuning import _render_job_detail
        assert callable(_render_job_detail)


class TestFineTuningCharts:
    """Tests for chart generation functions."""

    def test_loss_chart_with_data(self):
        from dashboard.pages.fine_tuning import _render_loss_chart
        import plotly.graph_objects as go

        mock_st = MagicMock()
        metrics = [
            {"epoch": 1, "loss": 2.5, "validation_loss": 2.8},
            {"epoch": 2, "loss": 1.8, "validation_loss": 2.1},
            {"epoch": 3, "loss": 1.2, "validation_loss": 1.5},
        ]
        _render_loss_chart(mock_st, metrics, "ft-001")
        mock_st.plotly_chart.assert_called_once()
        fig = mock_st.plotly_chart.call_args[0][0]
        assert isinstance(fig, go.Figure)
        # Should have 2 traces (training + validation)
        assert len(fig.data) == 2

    def test_loss_chart_no_validation(self):
        from dashboard.pages.fine_tuning import _render_loss_chart
        import plotly.graph_objects as go

        mock_st = MagicMock()
        metrics = [
            {"epoch": 1, "loss": 2.5},
            {"epoch": 2, "loss": 1.8},
        ]
        _render_loss_chart(mock_st, metrics)
        mock_st.plotly_chart.assert_called_once()
        fig = mock_st.plotly_chart.call_args[0][0]
        assert isinstance(fig, go.Figure)
        # Only training loss trace
        assert len(fig.data) == 1


class TestFineTuningJobSummary:
    """Tests for the job summary KPI rendering."""

    def test_job_summary_renders_four_metrics(self):
        from dashboard.pages.fine_tuning import _render_job_summary

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(4)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        jobs = [
            {"status": "completed", "actual_cost": 42.50},
            {"status": "running", "actual_cost": None},
        ]
        _render_job_summary(mock_st, jobs)
        mock_st.columns.assert_called_once_with(4)
        assert mock_st.metric.call_count == 4


class TestFineTuningJobTable:
    """Tests for the job table rendering."""

    def test_job_table_empty_shows_info(self):
        from dashboard.pages.fine_tuning import _render_job_table

        mock_st = MagicMock()
        result = _render_job_table(mock_st, [])
        assert result is None
        mock_st.info.assert_called_once()

    def test_job_table_renders_header(self):
        from dashboard.pages.fine_tuning import _render_job_table

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(6)]
        mock_st.columns.return_value = col_mocks

        jobs = [
            {
                "id": "ft-001",
                "model_id": "m1",
                "status": "completed",
                "training_metrics": [{"loss": 1.2}],
            },
        ]
        _render_job_table(mock_st, jobs)
        # Header + 1 data row = 2 calls to columns
        assert mock_st.columns.call_count == 2
