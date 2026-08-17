"""Tests for dashboard page modules.

These tests verify that page modules import correctly and that
their render functions are callable. Full Streamlit rendering
tests require a running Streamlit context, so we focus on
structural validation and helper logic.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestPageImports:
    """Verify all page modules import without errors."""

    def test_import_home(self):
        from dashboard.pages.home import render_home_page
        assert callable(render_home_page)

    def test_import_models(self):
        from dashboard.pages.models import render_models_page
        assert callable(render_models_page)

    def test_import_datasets(self):
        from dashboard.pages.datasets import render_datasets_page
        assert callable(render_datasets_page)

    def test_import_evaluation(self):
        from dashboard.pages.evaluation import render_evaluation_page
        assert callable(render_evaluation_page)

    def test_import_comparison(self):
        from dashboard.pages.comparison import render_comparison_page
        assert callable(render_comparison_page)

    def test_import_fine_tuning(self):
        from dashboard.pages.fine_tuning import render_fine_tuning_page
        assert callable(render_fine_tuning_page)

    def test_import_workflows(self):
        from dashboard.pages.workflows import render_workflows_page
        assert callable(render_workflows_page)


class TestAppModule:
    """Tests for the main app module."""

    def test_import_app(self):
        from dashboard.app import main, PAGE_RENDERERS
        assert callable(main)
        assert isinstance(PAGE_RENDERERS, dict)

    def test_page_renderers_complete(self):
        from dashboard.app import PAGE_RENDERERS
        expected_pages = {"Home", "Models", "Datasets", "Evaluation", "Comparison", "Fine-Tuning", "Workflows"}
        assert set(PAGE_RENDERERS.keys()) == expected_pages

    def test_all_renderers_callable(self):
        from dashboard.app import PAGE_RENDERERS
        for name, renderer in PAGE_RENDERERS.items():
            assert callable(renderer), f"Renderer for '{name}' is not callable"


class TestBackendModule:
    """Tests for the backend connector module."""

    def test_import_backend(self):
        from dashboard.utils.backend import (
            get_model_registry,
            get_dataset_manager,
            get_evaluation_engine,
            get_fine_tuning_pipeline,
            get_workflow_orchestrator,
            get_results_store,
        )
        # All should be callable
        assert callable(get_model_registry)
        assert callable(get_dataset_manager)
        assert callable(get_evaluation_engine)
        assert callable(get_fine_tuning_pipeline)
        assert callable(get_workflow_orchestrator)
        assert callable(get_results_store)

    def test_backend_graceful_failure(self):
        """Backend connectors should not raise exceptions."""
        from dashboard.utils.backend import (
            get_model_registry,
            get_dataset_manager,
            get_evaluation_engine,
            get_fine_tuning_pipeline,
            get_workflow_orchestrator,
            get_results_store,
        )
        # These should either return an instance or None,
        # but never raise an exception
        try:
            result = get_model_registry()
            assert result is None or result is not None  # either is fine
        except Exception:
            pytest.fail("get_model_registry raised an exception")

        try:
            result = get_dataset_manager()
            assert result is None or result is not None
        except Exception:
            pytest.fail("get_dataset_manager raised an exception")

        try:
            result = get_evaluation_engine()
            assert result is None or result is not None
        except Exception:
            pytest.fail("get_evaluation_engine raised an exception")

        try:
            result = get_fine_tuning_pipeline()
            assert result is None or result is not None
        except Exception:
            pytest.fail("get_fine_tuning_pipeline raised an exception")

        try:
            result = get_workflow_orchestrator()
            assert result is None or result is not None
        except Exception:
            pytest.fail("get_workflow_orchestrator raised an exception")

        try:
            result = get_results_store()
            assert result is None or result is not None
        except Exception:
            pytest.fail("get_results_store raised an exception")


class TestWorkflowTemplates:
    """Tests for workflow template definitions."""

    def test_templates_defined(self):
        from dashboard.pages.workflows import WORKFLOW_TEMPLATES
        assert len(WORKFLOW_TEMPLATES) >= 3

    def test_template_structure(self):
        from dashboard.pages.workflows import WORKFLOW_TEMPLATES
        for name, template in WORKFLOW_TEMPLATES.items():
            assert "id" in template
            assert "description" in template
            assert "steps" in template
            assert isinstance(template["steps"], list)
            assert len(template["steps"]) > 0

    def test_full_pipeline_template(self):
        from dashboard.pages.workflows import WORKFLOW_TEMPLATES
        full = WORKFLOW_TEMPLATES["Full Pipeline"]
        assert full["id"] == "full_pipeline"
        assert len(full["steps"]) == 5
        assert "Fine-Tuning" in full["steps"]

    def test_evaluation_only_template(self):
        from dashboard.pages.workflows import WORKFLOW_TEMPLATES
        eval_only = WORKFLOW_TEMPLATES["Evaluation Only"]
        assert eval_only["id"] == "evaluation_only"
        assert "Fine-Tuning" not in eval_only["steps"]
