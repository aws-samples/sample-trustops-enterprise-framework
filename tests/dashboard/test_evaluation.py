"""Tests for the dashboard evaluation page.

Validates helper functions, data loading, and rendering logic
for the evaluation page (task 21.5, Requirements 10.3, 10.6).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestEvaluationHelpers:
    """Tests for evaluation page helper functions."""

    def test_avg_trust_score(self):
        from dashboard.pages.evaluation import _avg_trust_score
        evals = [
            {"mean_trust_score": 0.8},
            {"mean_trust_score": 0.6},
        ]
        assert _avg_trust_score(evals) == pytest.approx(0.7)

    def test_avg_trust_score_empty(self):
        from dashboard.pages.evaluation import _avg_trust_score
        assert _avg_trust_score([]) is None

    def test_avg_trust_score_no_scores(self):
        from dashboard.pages.evaluation import _avg_trust_score
        evals = [{"id": "e1"}, {"id": "e2"}]
        assert _avg_trust_score(evals) is None

    def test_total_cost(self):
        from dashboard.pages.evaluation import _total_cost
        evals = [
            {"total_cost": 10.0},
            {"total_cost": 5.5},
        ]
        assert _total_cost(evals) == pytest.approx(15.5)

    def test_total_cost_empty(self):
        from dashboard.pages.evaluation import _total_cost
        assert _total_cost([]) == 0.0

    def test_total_cost_missing_field(self):
        from dashboard.pages.evaluation import _total_cost
        evals = [{"id": "e1"}, {"total_cost": 3.0}]
        assert _total_cost(evals) == pytest.approx(3.0)

    def test_running_evaluations(self):
        from dashboard.pages.evaluation import _running_evaluations
        evals = [
            {"id": "e1", "status": "running"},
            {"id": "e2", "status": "completed"},
            {"id": "e3", "status": "in_progress"},
        ]
        result = _running_evaluations(evals)
        assert len(result) == 2
        ids = {e["id"] for e in result}
        assert ids == {"e1", "e3"}

    def test_running_evaluations_empty(self):
        from dashboard.pages.evaluation import _running_evaluations
        assert _running_evaluations([]) == []

    def test_running_evaluations_none_status(self):
        from dashboard.pages.evaluation import _running_evaluations
        evals = [{"id": "e1"}]
        assert _running_evaluations(evals) == []


class TestEnrichDemoEvaluations:
    """Tests for the demo data enrichment."""

    def test_enriched_evals_have_extra_fields(self):
        from dashboard.pages.evaluation import (
            _enrich_demo_evaluations,
        )
        evals = _enrich_demo_evaluations()
        assert len(evals) > 0
        for ev in evals:
            assert "status" in ev
            assert "total_examples" in ev
            assert "latency_p50_ms" in ev
            assert "per_category" in ev

    def test_enriched_evals_preserve_original_fields(self):
        from dashboard.pages.evaluation import (
            _enrich_demo_evaluations,
        )
        evals = _enrich_demo_evaluations()
        for ev in evals:
            assert "id" in ev
            assert "model_id" in ev
            assert "mean_trust_score" in ev

    def test_running_eval_has_progress(self):
        from dashboard.pages.evaluation import (
            _enrich_demo_evaluations,
        )
        evals = _enrich_demo_evaluations()
        running = [
            e for e in evals if e.get("status") == "running"
        ]
        for ev in running:
            assert "progress_percent" in ev
            assert "completed_examples" in ev


class TestEvaluationDataLoading:
    """Tests for evaluation page data loading."""

    def test_load_data_falls_back_to_demo(self):
        from dashboard.pages.evaluation import (
            _load_evaluation_data,
        )
        with (
            patch(
                "dashboard.utils.backend.get_evaluation_engine",
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
            data = _load_evaluation_data()
            assert data["demo_mode"] is True
            assert len(data["evaluations"]) > 0
            assert len(data["models"]) > 0
            assert len(data["datasets"]) > 0

    def test_load_data_evaluations_have_fields(self):
        from dashboard.pages.evaluation import (
            _load_evaluation_data,
        )
        data = _load_evaluation_data()
        for ev in data["evaluations"]:
            assert "id" in ev
            assert "model_id" in ev
            assert "mean_trust_score" in ev


class TestEvaluationRendering:
    """Tests for evaluation page render functions."""

    def test_render_evaluation_page_callable(self):
        from dashboard.pages.evaluation import (
            render_evaluation_page,
        )
        assert callable(render_evaluation_page)

    def test_render_evaluation_summary_callable(self):
        from dashboard.pages.evaluation import (
            _render_evaluation_summary,
        )
        assert callable(_render_evaluation_summary)

    def test_render_evaluation_table_callable(self):
        from dashboard.pages.evaluation import (
            _render_evaluation_table,
        )
        assert callable(_render_evaluation_table)

    def test_render_evaluation_wizard_callable(self):
        from dashboard.pages.evaluation import (
            _render_evaluation_wizard,
        )
        assert callable(_render_evaluation_wizard)

    def test_render_progress_display_callable(self):
        from dashboard.pages.evaluation import (
            _render_progress_display,
        )
        assert callable(_render_progress_display)

    def test_render_aggregate_metrics_callable(self):
        from dashboard.pages.evaluation import (
            _render_aggregate_metrics,
        )
        assert callable(_render_aggregate_metrics)

    def test_render_category_breakdown_callable(self):
        from dashboard.pages.evaluation import (
            _render_category_breakdown,
        )
        assert callable(_render_category_breakdown)

    def test_render_evaluation_detail_callable(self):
        from dashboard.pages.evaluation import (
            _render_evaluation_detail,
        )
        assert callable(_render_evaluation_detail)


class TestEvaluationSummary:
    """Tests for the summary KPI rendering."""

    def test_summary_renders_four_metrics(self):
        from dashboard.pages.evaluation import (
            _render_evaluation_summary,
        )

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(4)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        evals = [
            {
                "id": "e1",
                "mean_trust_score": 0.8,
                "total_cost": 5.0,
                "status": "completed",
            },
        ]
        _render_evaluation_summary(mock_st, evals)
        mock_st.columns.assert_called_once_with(4)
        assert mock_st.metric.call_count == 4


class TestCategoryBreakdownChart:
    """Tests for per-category breakdown chart."""

    def test_category_chart_with_data(self):
        from dashboard.pages.evaluation import (
            _render_category_breakdown,
        )
        import plotly.graph_objects as go

        mock_st = MagicMock()
        evaluation = {
            "per_category": {
                "factual": {
                    "mean_trust_score": 0.85,
                    "count": 100,
                },
                "reasoning": {
                    "mean_trust_score": 0.72,
                    "count": 50,
                },
            },
        }
        _render_category_breakdown(mock_st, evaluation)
        mock_st.plotly_chart.assert_called_once()
        fig = mock_st.plotly_chart.call_args[0][0]
        assert isinstance(fig, go.Figure)

    def test_category_chart_empty(self):
        from dashboard.pages.evaluation import (
            _render_category_breakdown,
        )
        mock_st = MagicMock()
        _render_category_breakdown(mock_st, {})
        mock_st.plotly_chart.assert_not_called()
        mock_st.info.assert_called_once()


class TestProgressDisplay:
    """Tests for the progress display rendering."""

    def test_progress_display_with_running(self):
        from dashboard.pages.evaluation import (
            _render_progress_display,
        )
        mock_st = MagicMock()
        evals = [
            {
                "id": "e1",
                "status": "running",
                "model_id": "m1",
                "progress_percent": 50.0,
                "completed_examples": 25,
                "total_examples": 50,
            },
        ]
        _render_progress_display(mock_st, evals)
        mock_st.subheader.assert_called_once()
        mock_st.progress.assert_called_once_with(0.5)

    def test_progress_display_no_running(self):
        from dashboard.pages.evaluation import (
            _render_progress_display,
        )
        mock_st = MagicMock()
        evals = [{"id": "e1", "status": "completed"}]
        _render_progress_display(mock_st, evals)
        mock_st.subheader.assert_not_called()


class TestEvaluationTable:
    """Tests for the evaluation table rendering."""

    def test_table_empty_shows_info(self):
        from dashboard.pages.evaluation import (
            _render_evaluation_table,
        )
        mock_st = MagicMock()
        result = _render_evaluation_table(mock_st, [])
        assert result is None
        mock_st.info.assert_called_once()
