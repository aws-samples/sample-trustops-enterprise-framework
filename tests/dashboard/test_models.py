"""Tests for the dashboard models page.

Validates helper functions, data loading, filtering, and rendering
logic for the models page (task 21.3, Requirement 10.3).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestModelsHelpers:
    """Tests for models page helper functions."""

    def test_unique_values_simple(self):
        from dashboard.pages.models import _unique_values
        models = [
            {"provider": "bedrock"},
            {"provider": "sagemaker"},
            {"provider": "bedrock"},
        ]
        result = _unique_values(models, "provider")
        assert result == ["bedrock", "sagemaker"]

    def test_unique_values_list_field(self):
        from dashboard.pages.models import _unique_values
        models = [
            {"capabilities": ["chat", "embedding"]},
            {"capabilities": ["chat", "text_generation"]},
        ]
        result = _unique_values(models, "capabilities")
        assert result == ["chat", "embedding", "text_generation"]

    def test_unique_values_missing_key(self):
        from dashboard.pages.models import _unique_values
        models = [{"name": "foo"}, {"provider": "bedrock"}]
        result = _unique_values(models, "provider")
        assert result == ["bedrock"]

    def test_unique_values_empty(self):
        from dashboard.pages.models import _unique_values
        assert _unique_values([], "provider") == []

    def test_filter_models_by_provider(self):
        from dashboard.pages.models import _filter_models
        models = [
            {"provider": "bedrock", "status": "active"},
            {"provider": "sagemaker", "status": "active"},
        ]
        result = _filter_models(models, provider="bedrock")
        assert len(result) == 1
        assert result[0]["provider"] == "bedrock"

    def test_filter_models_by_capability(self):
        from dashboard.pages.models import _filter_models
        models = [
            {"capabilities": ["chat"], "provider": "bedrock"},
            {"capabilities": ["embedding"], "provider": "bedrock"},
        ]
        result = _filter_models(models, capability="chat")
        assert len(result) == 1

    def test_filter_models_by_status(self):
        from dashboard.pages.models import _filter_models
        models = [
            {"status": "active"},
            {"status": "inactive"},
            {"status": "active"},
        ]
        result = _filter_models(models, status="active")
        assert len(result) == 2

    def test_filter_models_combined(self):
        from dashboard.pages.models import _filter_models
        models = [
            {"provider": "bedrock", "status": "active",
             "capabilities": ["chat"]},
            {"provider": "bedrock", "status": "inactive",
             "capabilities": ["chat"]},
            {"provider": "sagemaker", "status": "active",
             "capabilities": ["chat"]},
        ]
        result = _filter_models(
            models, provider="bedrock", status="active",
        )
        assert len(result) == 1

    def test_filter_models_no_filters(self):
        from dashboard.pages.models import _filter_models
        models = [{"provider": "a"}, {"provider": "b"}]
        assert _filter_models(models) == models

    def test_filter_models_case_insensitive_provider(self):
        from dashboard.pages.models import _filter_models
        models = [{"provider": "Bedrock"}]
        result = _filter_models(models, provider="bedrock")
        assert len(result) == 1



class TestModelsDataLoading:
    """Tests for the models page data loading."""

    def test_load_models_data_falls_back_to_demo(self):
        from dashboard.pages.models import _load_models_data

        with patch(
            "dashboard.utils.backend.get_model_registry",
            return_value=None,
        ):
            data = _load_models_data()
            assert data["demo_mode"] is True
            assert len(data["models"]) > 0

    def test_load_models_data_models_have_expected_fields(self):
        from dashboard.pages.models import _load_models_data

        with patch(
            "dashboard.utils.backend.get_model_registry",
            return_value=None,
        ):
            data = _load_models_data()
            for model in data["models"]:
                assert "id" in model
                assert "provider" in model
                assert "status" in model

    def test_load_models_data_uses_backend_when_available(self):
        from dashboard.pages.models import _load_models_data

        mock_registry = MagicMock()
        mock_registry.list_models.return_value = [
            {"id": "m1", "provider": "bedrock", "status": "active"},
        ]
        with patch(
            "dashboard.utils.backend.get_model_registry",
            return_value=mock_registry,
        ):
            data = _load_models_data()
            assert data["demo_mode"] is False
            assert len(data["models"]) == 1

    def test_load_models_data_backend_exception_falls_back(self):
        from dashboard.pages.models import _load_models_data

        mock_registry = MagicMock()
        mock_registry.list_models.side_effect = RuntimeError("boom")
        with patch(
            "dashboard.utils.backend.get_model_registry",
            return_value=mock_registry,
        ):
            data = _load_models_data()
            assert data["demo_mode"] is True
            assert len(data["models"]) > 0


class TestModelsRendering:
    """Tests for the models page render functions."""

    def test_render_models_page_callable(self):
        from dashboard.pages.models import render_models_page
        assert callable(render_models_page)

    def test_render_model_table_callable(self):
        from dashboard.pages.models import _render_model_table
        assert callable(_render_model_table)

    def test_render_model_summary_callable(self):
        from dashboard.pages.models import _render_model_summary
        assert callable(_render_model_summary)

    def test_render_registration_form_callable(self):
        from dashboard.pages.models import _render_registration_form
        assert callable(_render_registration_form)

    def test_render_model_table_empty(self):
        from dashboard.pages.models import _render_model_table
        mock_st = MagicMock()
        _render_model_table(mock_st, [])
        mock_st.info.assert_called_once()

    def test_render_model_table_with_data(self):
        from dashboard.pages.models import _render_model_table
        mock_st = MagicMock()
        models = [
            {
                "id": "m1",
                "name": "Claude 3",
                "provider": "bedrock",
                "capabilities": ["chat"],
                "status": "active",
                "max_tokens": 4096,
                "fine_tuning_support": True,
            },
        ]
        _render_model_table(mock_st, models)
        # Header + 1 data row = 2 calls to st.columns
        assert mock_st.columns.call_count == 2

    def test_render_model_summary_metrics(self):
        from dashboard.pages.models import _render_model_summary

        mock_st = MagicMock()
        col_mocks = [MagicMock() for _ in range(4)]
        for cm in col_mocks:
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = col_mocks

        models = [
            {"status": "active", "fine_tuning_support": True},
            {"status": "inactive", "fine_tuning_support": False},
        ]
        _render_model_summary(mock_st, models, models)
        assert mock_st.metric.call_count == 4
