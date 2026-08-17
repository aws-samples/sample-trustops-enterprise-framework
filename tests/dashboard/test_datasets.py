"""Tests for the dashboard datasets page.

Validates helper functions, data loading, and rendering logic
for the datasets page (task 21.4, Requirement 10.14).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestDatasetHelpers:
    """Tests for dataset page helper functions."""

    def test_get_quality_scores_from_quality_dict(self):
        from dashboard.pages.datasets import _get_quality_scores
        ds = {"quality": {"completeness": 0.9, "diversity": 0.8, "balance": 0.7}}
        result = _get_quality_scores(ds)
        assert result == {"completeness": 0.9, "diversity": 0.8, "balance": 0.7}

    def test_get_quality_scores_from_quality_report(self):
        from dashboard.pages.datasets import _get_quality_scores
        ds = {"quality_report": {
            "completeness_score": 0.95,
            "diversity_score": 0.85,
            "balance_score": 0.75,
        }}
        result = _get_quality_scores(ds)
        assert result == {"completeness": 0.95, "diversity": 0.85, "balance": 0.75}

    def test_get_quality_scores_missing(self):
        from dashboard.pages.datasets import _get_quality_scores
        result = _get_quality_scores({})
        assert result == {"completeness": 0.0, "diversity": 0.0, "balance": 0.0}

    def test_avg_quality(self):
        from dashboard.pages.datasets import _avg_quality
        datasets = [
            {"quality": {"completeness": 0.8, "diversity": 0.6, "balance": 0.9}},
            {"quality": {"completeness": 1.0, "diversity": 0.8, "balance": 0.7}},
        ]
        assert _avg_quality(datasets, "completeness") == pytest.approx(0.9)
        assert _avg_quality(datasets, "diversity") == pytest.approx(0.7)

    def test_avg_quality_empty(self):
        from dashboard.pages.datasets import _avg_quality
        assert _avg_quality([], "completeness") is None

    def test_demo_pii_results_not_empty(self):
        from dashboard.pages.datasets import _demo_pii_results
        results = _demo_pii_results()
        assert len(results) > 0
        for r in results:
            assert "row" in r
            assert "pii_type" in r

    def test_demo_version_history_ordered(self):
        from dashboard.pages.datasets import _demo_version_history
        versions = _demo_version_history()
        assert len(versions) > 0
        assert versions[0]["version"] == "v3"

    def test_demo_lineage_has_steps(self):
        from dashboard.pages.datasets import _demo_lineage
        lineage = _demo_lineage()
        assert len(lineage) > 0
        for step in lineage:
            assert "step" in step
            assert "source" in step


class TestDatasetDataLoading:
    """Tests for dataset page data loading."""

    def test_load_datasets_data_falls_back_to_demo(self):
        from dashboard.pages.datasets import _load_datasets_data
        with patch("dashboard.utils.backend.get_dataset_manager", return_value=None):
            data = _load_datasets_data()
            assert data["demo_mode"] is True
            assert len(data["datasets"]) > 0

    def test_load_datasets_data_has_expected_fields(self):
        from dashboard.pages.datasets import _load_datasets_data
        data = _load_datasets_data()
        for ds in data["datasets"]:
            assert "id" in ds
            assert "name" in ds
            assert "row_count" in ds


class TestDatasetRendering:
    """Tests for dataset page render functions."""

    def test_render_datasets_page_callable(self):
        from dashboard.pages.datasets import render_datasets_page
        assert callable(render_datasets_page)

    def test_render_dataset_summary_callable(self):
        from dashboard.pages.datasets import _render_dataset_summary
        assert callable(_render_dataset_summary)

    def test_render_dataset_table_callable(self):
        from dashboard.pages.datasets import _render_dataset_table
        assert callable(_render_dataset_table)

    def test_render_upload_form_callable(self):
        from dashboard.pages.datasets import _render_upload_form
        assert callable(_render_upload_form)

    def test_render_quality_report_callable(self):
        from dashboard.pages.datasets import _render_quality_report
        assert callable(_render_quality_report)

    def test_render_pii_results_callable(self):
        from dashboard.pages.datasets import _render_pii_results
        assert callable(_render_pii_results)

    def test_render_version_history_callable(self):
        from dashboard.pages.datasets import _render_version_history
        assert callable(_render_version_history)

    def test_render_lineage_callable(self):
        from dashboard.pages.datasets import _render_lineage
        assert callable(_render_lineage)


class TestDatasetSummary:
    """Tests for the summary KPI rendering."""

    def test_summary_renders_four_metrics(self):
        from dashboard.pages.datasets import _render_dataset_summary

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(4)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        datasets = [
            {"id": "ds-1", "row_count": 100,
             "format": "jsonl",
             "quality": {"completeness": 0.9, "diversity": 0.8, "balance": 0.7}},
        ]
        _render_dataset_summary(mock_st, datasets)
        mock_st.columns.assert_called_once_with(4)
        assert mock_st.metric.call_count == 4


class TestQualityChart:
    """Tests for quality report chart generation."""

    def test_quality_report_produces_chart(self):
        from dashboard.pages.datasets import _render_quality_report
        import plotly.graph_objects as go

        mock_st = MagicMock()
        dataset = {
            "name": "Test DS",
            "quality": {"completeness": 0.95, "diversity": 0.82, "balance": 0.78},
        }
        _render_quality_report(mock_st, dataset)
        mock_st.plotly_chart.assert_called_once()
        fig = mock_st.plotly_chart.call_args[0][0]
        assert isinstance(fig, go.Figure)


class TestPIIResults:
    """Tests for PII detection results rendering."""

    def test_pii_results_with_data(self):
        from dashboard.pages.datasets import _render_pii_results
        import plotly.graph_objects as go

        mock_st = MagicMock()
        dataset = {"pii_results": [
            {"row": 1, "field": "prompt", "pii_type": "email", "value": "[EMAIL]"},
        ]}
        _render_pii_results(mock_st, dataset)
        mock_st.warning.assert_called_once()
        mock_st.plotly_chart.assert_called_once()
