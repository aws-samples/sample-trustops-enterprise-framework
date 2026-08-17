"""Tests for the dashboard workflows page.

Validates helper functions, data loading, and rendering logic
for the workflows page (task 21.8, Requirement 10.13).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestWorkflowHelpers:
    """Tests for workflow page helper functions."""

    def test_step_progress_all_completed(self):
        from dashboard.pages.workflows import _step_progress
        steps = [
            {"name": "A", "status": "completed"},
            {"name": "B", "status": "completed"},
        ]
        done, total = _step_progress(steps)
        assert done == 2
        assert total == 2

    def test_step_progress_partial(self):
        from dashboard.pages.workflows import _step_progress
        steps = [
            {"name": "A", "status": "completed"},
            {"name": "B", "status": "running"},
            {"name": "C", "status": "pending"},
        ]
        done, total = _step_progress(steps)
        assert done == 1
        assert total == 3

    def test_step_progress_empty(self):
        from dashboard.pages.workflows import _step_progress
        assert _step_progress([]) == (0, 0)

    def test_running_workflows_filters(self):
        from dashboard.pages.workflows import (
            _running_workflows,
        )
        wfs = [
            {"id": "w1", "status": "running"},
            {"id": "w2", "status": "completed"},
            {"id": "w3", "status": "in_progress"},
        ]
        result = _running_workflows(wfs)
        assert len(result) == 2
        ids = {w["id"] for w in result}
        assert ids == {"w1", "w3"}

    def test_running_workflows_empty(self):
        from dashboard.pages.workflows import (
            _running_workflows,
        )
        assert _running_workflows([]) == []

    def test_running_workflows_none_running(self):
        from dashboard.pages.workflows import (
            _running_workflows,
        )
        wfs = [
            {"id": "w1", "status": "completed"},
            {"id": "w2", "status": "failed"},
        ]
        assert _running_workflows(wfs) == []


class TestWorkflowTemplates:
    """Tests for workflow template definitions."""

    def test_templates_exist(self):
        from dashboard.pages.workflows import (
            WORKFLOW_TEMPLATES,
        )
        assert len(WORKFLOW_TEMPLATES) == 3

    def test_full_pipeline_template(self):
        from dashboard.pages.workflows import (
            WORKFLOW_TEMPLATES,
        )
        tmpl = WORKFLOW_TEMPLATES["Full Pipeline"]
        assert tmpl["id"] == "full_pipeline"
        assert len(tmpl["steps"]) == 5

    def test_evaluation_only_template(self):
        from dashboard.pages.workflows import (
            WORKFLOW_TEMPLATES,
        )
        tmpl = WORKFLOW_TEMPLATES["Evaluation Only"]
        assert tmpl["id"] == "evaluation_only"
        assert len(tmpl["steps"]) == 2

    def test_comparison_only_template(self):
        from dashboard.pages.workflows import (
            WORKFLOW_TEMPLATES,
        )
        tmpl = WORKFLOW_TEMPLATES["Comparison Only"]
        assert tmpl["id"] == "comparison_only"
        assert len(tmpl["steps"]) == 1

    def test_all_templates_have_required_keys(self):
        from dashboard.pages.workflows import (
            WORKFLOW_TEMPLATES,
        )
        for name, tmpl in WORKFLOW_TEMPLATES.items():
            assert "id" in tmpl, f"{name} missing id"
            assert "description" in tmpl
            assert "steps" in tmpl
            assert len(tmpl["steps"]) > 0


class TestWorkflowDataLoading:
    """Tests for the workflows page data loading."""

    def test_load_data_falls_back_to_demo(self):
        from dashboard.pages.workflows import (
            _load_workflows_data,
        )

        with patch(
            "dashboard.utils.backend"
            ".get_workflow_orchestrator",
            return_value=None,
        ):
            data = _load_workflows_data()
            assert data["demo_mode"] is True
            assert len(data["workflows"]) > 0

    def test_demo_workflows_have_expected_fields(self):
        from dashboard.pages.workflows import (
            _build_demo_workflows,
        )
        workflows = _build_demo_workflows()
        for wf in workflows:
            assert "id" in wf
            assert "status" in wf
            assert "steps" in wf

    def test_demo_workflows_have_enriched_fields(self):
        from dashboard.pages.workflows import (
            _build_demo_workflows,
        )
        workflows = _build_demo_workflows()
        completed = [
            w for w in workflows
            if w.get("status") == "completed"
        ]
        assert len(completed) > 0
        for wf in completed:
            assert "template" in wf
            assert "audit_trail" in wf
            assert len(wf["audit_trail"]) > 0

    def test_demo_workflows_audit_trail_structure(self):
        from dashboard.pages.workflows import (
            _build_demo_workflows,
        )
        workflows = _build_demo_workflows()
        for wf in workflows:
            trail = wf.get("audit_trail", [])
            for event in trail:
                assert "timestamp" in event
                assert "step" in event
                assert "event" in event
                assert "detail" in event

    def test_load_data_uses_backend_when_available(self):
        from dashboard.pages.workflows import (
            _load_workflows_data,
        )

        mock_orch = MagicMock()
        mock_orch.list_workflows.return_value = [
            {
                "id": "wf-live",
                "status": "completed",
                "steps": [],
            },
        ]

        with patch(
            "dashboard.utils.backend"
            ".get_workflow_orchestrator",
            return_value=mock_orch,
        ):
            data = _load_workflows_data()
            assert data["demo_mode"] is False
            assert len(data["workflows"]) == 1
            assert data["workflows"][0]["id"] == "wf-live"

    def test_load_data_falls_back_on_exception(self):
        from dashboard.pages.workflows import (
            _load_workflows_data,
        )

        mock_orch = MagicMock()
        mock_orch.list_workflows.side_effect = RuntimeError(
            "connection failed"
        )

        with patch(
            "dashboard.utils.backend"
            ".get_workflow_orchestrator",
            return_value=mock_orch,
        ):
            data = _load_workflows_data()
            assert data["demo_mode"] is True
            assert len(data["workflows"]) > 0


class TestWorkflowRendering:
    """Tests for rendering functions (callable checks)."""

    def test_render_workflows_page_callable(self):
        from dashboard.pages.workflows import (
            render_workflows_page,
        )
        assert callable(render_workflows_page)

    def test_render_workflow_summary_callable(self):
        from dashboard.pages.workflows import (
            _render_workflow_summary,
        )
        assert callable(_render_workflow_summary)

    def test_render_workflow_table_callable(self):
        from dashboard.pages.workflows import (
            _render_workflow_table,
        )
        assert callable(_render_workflow_table)

    def test_render_step_visualization_callable(self):
        from dashboard.pages.workflows import (
            _render_step_visualization,
        )
        assert callable(_render_step_visualization)

    def test_render_execution_progress_callable(self):
        from dashboard.pages.workflows import (
            _render_execution_progress,
        )
        assert callable(_render_execution_progress)

    def test_render_template_selector_callable(self):
        from dashboard.pages.workflows import (
            _render_template_selector,
        )
        assert callable(_render_template_selector)

    def test_render_audit_trail_callable(self):
        from dashboard.pages.workflows import (
            _render_audit_trail,
        )
        assert callable(_render_audit_trail)

    def test_render_workflow_detail_callable(self):
        from dashboard.pages.workflows import (
            _render_workflow_detail,
        )
        assert callable(_render_workflow_detail)


class TestWorkflowSummaryKPIs:
    """Tests for the workflow summary KPI rendering."""

    def test_summary_renders_four_metrics(self):
        from dashboard.pages.workflows import (
            _render_workflow_summary,
        )

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(4)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        workflows = [
            {
                "status": "completed",
                "total_cost": 57.50,
            },
            {
                "status": "running",
                "total_cost": 4.30,
            },
        ]
        _render_workflow_summary(mock_st, workflows)
        mock_st.columns.assert_called_once_with(4)
        assert mock_st.metric.call_count == 4

    def test_summary_empty_workflows(self):
        from dashboard.pages.workflows import (
            _render_workflow_summary,
        )

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(4)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        _render_workflow_summary(mock_st, [])
        assert mock_st.metric.call_count == 4


class TestWorkflowTable:
    """Tests for the workflow table rendering."""

    def test_table_empty_shows_info(self):
        from dashboard.pages.workflows import (
            _render_workflow_table,
        )

        mock_st = MagicMock()
        result = _render_workflow_table(
            mock_st, [], "All",
        )
        assert result is None
        mock_st.info.assert_called_once()

    def test_table_renders_header_and_rows(self):
        from dashboard.pages.workflows import (
            _render_workflow_table,
        )

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(5)]
        mock_st.columns.return_value = col_mocks

        workflows = [
            {
                "id": "wf-001",
                "name": "Test",
                "status": "completed",
                "steps": [
                    {"status": "completed"},
                ],
            },
        ]
        _render_workflow_table(mock_st, workflows, "All")
        # Header + 1 data row = 2 calls to columns
        assert mock_st.columns.call_count == 2

    def test_table_filters_by_status(self):
        from dashboard.pages.workflows import (
            _render_workflow_table,
        )

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(5)]
        mock_st.columns.return_value = col_mocks

        workflows = [
            {
                "id": "wf-001",
                "status": "completed",
                "steps": [],
            },
            {
                "id": "wf-002",
                "status": "failed",
                "steps": [],
            },
        ]
        _render_workflow_table(
            mock_st, workflows, "Failed",
        )
        # Header + 1 matching row = 2 calls
        assert mock_st.columns.call_count == 2

    def test_table_filter_no_match(self):
        from dashboard.pages.workflows import (
            _render_workflow_table,
        )

        mock_st = MagicMock()
        workflows = [
            {
                "id": "wf-001",
                "status": "completed",
                "steps": [],
            },
        ]
        result = _render_workflow_table(
            mock_st, workflows, "Failed",
        )
        assert result is None
        mock_st.info.assert_called_once()


class TestAuditTrail:
    """Tests for the audit trail rendering."""

    def test_audit_trail_empty_shows_info(self):
        from dashboard.pages.workflows import (
            _render_audit_trail,
        )

        mock_st = MagicMock()
        _render_audit_trail(mock_st, [])
        mock_st.info.assert_called_once()

    def test_audit_trail_renders_events(self):
        from dashboard.pages.workflows import (
            _render_audit_trail,
        )

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(4)]
        mock_st.columns.return_value = col_mocks

        trail = [
            {
                "timestamp": "2024-01-12T08:00:00Z",
                "step": "Dataset Upload",
                "event": "started",
                "detail": "Uploading data",
            },
            {
                "timestamp": "2024-01-12T08:02:00Z",
                "step": "Dataset Upload",
                "event": "completed",
                "detail": "Done",
            },
        ]
        _render_audit_trail(mock_st, trail)
        # Header + 2 event rows = 3 calls to columns
        assert mock_st.columns.call_count == 3


class TestStepVisualization:
    """Tests for step-by-step visualization."""

    def test_step_viz_empty_shows_info(self):
        from dashboard.pages.workflows import (
            _render_step_visualization,
        )

        mock_st = MagicMock()
        _render_step_visualization(mock_st, [])
        mock_st.info.assert_called_once()

    def test_step_viz_creates_columns(self):
        from dashboard.pages.workflows import (
            _render_step_visualization,
        )

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(3)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        steps = [
            {"name": "A", "status": "completed"},
            {"name": "B", "status": "running"},
            {"name": "C", "status": "pending"},
        ]
        _render_step_visualization(mock_st, steps)
        mock_st.columns.assert_called_once_with(3)
