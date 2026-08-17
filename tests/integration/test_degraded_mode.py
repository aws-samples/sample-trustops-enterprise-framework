"""Degraded mode integration tests.

Tests system behavior when AWS services are partially unavailable.
Verifies graceful degradation and clear error messages.

Requirements: 10.17
"""

from unittest.mock import patch, MagicMock

import pytest

from dashboard.utils.error_handler import (
    ServiceStatus,
    SERVICE_FEATURE_MAP,
    check_service_availability,
    get_unavailable_features,
    is_feature_available,
    render_status_banner,
    with_degraded_fallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_available_statuses() -> dict[str, ServiceStatus]:
    """Return statuses where every service is available."""
    return {
        name: ServiceStatus(name=name, available=True, affected_features=features)
        for name, features in SERVICE_FEATURE_MAP.items()
    }


def _statuses_with_down(*service_names: str) -> dict[str, ServiceStatus]:
    """Return statuses where the given services are down, rest are up."""
    statuses: dict[str, ServiceStatus] = {}
    for name, features in SERVICE_FEATURE_MAP.items():
        if name in service_names:
            statuses[name] = ServiceStatus(
                name=name,
                available=False,
                error=f"{name} connection refused",
                affected_features=features,
            )
        else:
            statuses[name] = ServiceStatus(
                name=name, available=True, affected_features=features
            )
    return statuses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPartialServiceUnavailability:
    """Test system behavior when individual AWS services are down."""

    def test_s3_down_disables_storage_dependent_features(self):
        """When results_store (S3-backed) is down, Evaluation/Comparison/Workflows are affected."""
        statuses = _statuses_with_down("results_store")
        unavailable = get_unavailable_features(statuses)

        assert "Evaluation" in unavailable
        assert "Comparison" in unavailable
        assert "Workflows" in unavailable
        # Models and Datasets should still work
        assert is_feature_available("Models", statuses)
        assert is_feature_available("Datasets", statuses)

    def test_model_registry_down_disables_model_features(self):
        """When model_registry is down, Models/Evaluation/Comparison/Fine-Tuning are affected."""
        statuses = _statuses_with_down("model_registry")
        unavailable = get_unavailable_features(statuses)

        assert "Models" in unavailable
        assert "Evaluation" in unavailable
        assert "Comparison" in unavailable
        assert "Fine-Tuning" in unavailable
        # Datasets and Workflows should still work
        assert is_feature_available("Datasets", statuses)
        assert is_feature_available("Workflows", statuses)

    def test_evaluation_engine_down_only_affects_eval_features(self):
        """When evaluation_engine is down, only Evaluation and Comparison are affected."""
        statuses = _statuses_with_down("evaluation_engine")
        unavailable = get_unavailable_features(statuses)

        assert "Evaluation" in unavailable
        assert "Comparison" in unavailable
        assert is_feature_available("Models", statuses)
        assert is_feature_available("Datasets", statuses)
        assert is_feature_available("Fine-Tuning", statuses)
        assert is_feature_available("Workflows", statuses)

    def test_fine_tuning_down_only_affects_fine_tuning(self):
        """When fine_tuning_pipeline is down, only Fine-Tuning is affected."""
        statuses = _statuses_with_down("fine_tuning_pipeline")
        unavailable = get_unavailable_features(statuses)

        assert unavailable == {"Fine-Tuning"}
        # Everything else works
        for feature in ["Models", "Datasets", "Evaluation", "Comparison", "Workflows"]:
            assert is_feature_available(feature, statuses)

    def test_workflow_orchestrator_down_only_affects_workflows(self):
        """When workflow_orchestrator is down, only Workflows is affected."""
        statuses = _statuses_with_down("workflow_orchestrator")
        unavailable = get_unavailable_features(statuses)

        assert unavailable == {"Workflows"}

    def test_dataset_manager_down_disables_dataset_dependent_features(self):
        """When dataset_manager is down, Datasets/Evaluation/Fine-Tuning are affected."""
        statuses = _statuses_with_down("dataset_manager")
        unavailable = get_unavailable_features(statuses)

        assert "Datasets" in unavailable
        assert "Evaluation" in unavailable
        assert "Fine-Tuning" in unavailable
        assert is_feature_available("Models", statuses)
        assert is_feature_available("Workflows", statuses)


class TestMultipleServicesDown:
    """Test behavior when multiple services are simultaneously unavailable."""

    def test_model_registry_and_results_store_down(self):
        """Multiple services down produces union of affected features."""
        statuses = _statuses_with_down("model_registry", "results_store")
        unavailable = get_unavailable_features(statuses)

        # model_registry affects: Models, Evaluation, Comparison, Fine-Tuning
        # results_store affects: Evaluation, Comparison, Workflows
        assert "Models" in unavailable
        assert "Evaluation" in unavailable
        assert "Comparison" in unavailable
        assert "Fine-Tuning" in unavailable
        assert "Workflows" in unavailable
        # Only Datasets should still work
        assert is_feature_available("Datasets", statuses)

    def test_all_services_down(self):
        """When all services are down, all features are unavailable."""
        statuses = _statuses_with_down(*SERVICE_FEATURE_MAP.keys())
        unavailable = get_unavailable_features(statuses)

        all_features = set()
        for features in SERVICE_FEATURE_MAP.values():
            all_features.update(features)

        assert unavailable == all_features

    def test_all_services_up_means_no_unavailable_features(self):
        """When all services are up, no features are unavailable."""
        statuses = _all_available_statuses()
        unavailable = get_unavailable_features(statuses)
        assert len(unavailable) == 0


class TestClearErrorMessages:
    """Test that clear error messages are returned for unavailable features."""

    def test_service_status_carries_error_message(self):
        """ServiceStatus stores a human-readable error when unavailable."""
        status = ServiceStatus(
            name="results_store",
            available=False,
            error="EndpointConnectionError: Could not connect to S3",
            affected_features=["Evaluation"],
        )
        assert "EndpointConnectionError" in status.error
        assert "S3" in status.error

    def test_service_status_no_error_when_available(self):
        """Available services have no error message."""
        status = ServiceStatus(name="model_registry", available=True)
        assert status.error is None

    @patch("dashboard.utils.error_handler.st")
    def test_with_degraded_fallback_shows_message_when_unavailable(self, mock_st):
        """with_degraded_fallback returns False and shows info message for unavailable features."""
        statuses = _statuses_with_down("evaluation_engine")
        result = with_degraded_fallback("Evaluation", statuses)

        assert result is False
        mock_st.info.assert_called_once()
        call_msg = mock_st.info.call_args[0][0]
        assert "Evaluation" in call_msg
        assert "unavailable" in call_msg

    @patch("dashboard.utils.error_handler.st")
    def test_with_degraded_fallback_returns_true_when_available(self, mock_st):
        """with_degraded_fallback returns True and shows nothing for available features."""
        statuses = _all_available_statuses()
        result = with_degraded_fallback("Models", statuses)

        assert result is True
        mock_st.info.assert_not_called()

    @patch("dashboard.utils.error_handler.st")
    def test_custom_fallback_message(self, mock_st):
        """Custom fallback message is displayed when provided."""
        statuses = _statuses_with_down("fine_tuning_pipeline")
        custom_msg = "Fine-tuning is temporarily offline. Please try again later."
        with_degraded_fallback("Fine-Tuning", statuses, fallback_message=custom_msg)

        call_msg = mock_st.info.call_args[0][0]
        assert custom_msg in call_msg


class TestStatusBanner:
    """Test that status banners correctly indicate which features are unavailable."""

    @patch("dashboard.utils.error_handler.st")
    def test_banner_shown_when_services_down(self, mock_st):
        """Status banner is displayed when services are unavailable."""
        statuses = _statuses_with_down("model_registry")
        render_status_banner(statuses)

        mock_st.warning.assert_called_once()
        warning_msg = mock_st.warning.call_args[0][0]
        assert "Degraded Mode" in warning_msg
        assert "model_registry" in warning_msg

    @patch("dashboard.utils.error_handler.st")
    def test_banner_lists_affected_features(self, mock_st):
        """Status banner lists all affected features."""
        statuses = _statuses_with_down("model_registry")
        render_status_banner(statuses)

        warning_msg = mock_st.warning.call_args[0][0]
        # model_registry affects Models, Evaluation, Comparison, Fine-Tuning
        assert "Models" in warning_msg

    @patch("dashboard.utils.error_handler.st")
    def test_no_banner_when_all_services_up(self, mock_st):
        """No banner is shown when all services are available."""
        statuses = _all_available_statuses()
        render_status_banner(statuses)

        mock_st.warning.assert_not_called()

    @patch("dashboard.utils.error_handler.st")
    def test_banner_lists_multiple_down_services(self, mock_st):
        """Banner lists all unavailable services when multiple are down."""
        statuses = _statuses_with_down("model_registry", "results_store")
        render_status_banner(statuses)

        warning_msg = mock_st.warning.call_args[0][0]
        assert "model_registry" in warning_msg
        assert "results_store" in warning_msg


class TestServiceAvailabilityDetection:
    """Test the error handler's service availability detection via check_service_availability."""

    @patch("dashboard.utils.backend.get_results_store")
    @patch("dashboard.utils.backend.get_workflow_orchestrator")
    @patch("dashboard.utils.backend.get_fine_tuning_pipeline")
    @patch("dashboard.utils.backend.get_evaluation_engine")
    @patch("dashboard.utils.backend.get_dataset_manager")
    @patch("dashboard.utils.backend.get_model_registry")
    def test_detects_all_services_available(
        self, mock_mr, mock_dm, mock_ee, mock_ftp, mock_wo, mock_rs
    ):
        """All services returning valid instances are marked available."""
        mock_mr.return_value = MagicMock()
        mock_dm.return_value = MagicMock()
        mock_ee.return_value = MagicMock()
        mock_ftp.return_value = MagicMock()
        mock_wo.return_value = MagicMock()
        mock_rs.return_value = MagicMock()

        statuses = check_service_availability()

        for name, status in statuses.items():
            assert status.available is True, f"{name} should be available"
            assert status.error is None

    @patch("dashboard.utils.backend.get_results_store")
    @patch("dashboard.utils.backend.get_workflow_orchestrator")
    @patch("dashboard.utils.backend.get_fine_tuning_pipeline")
    @patch("dashboard.utils.backend.get_evaluation_engine")
    @patch("dashboard.utils.backend.get_dataset_manager")
    @patch("dashboard.utils.backend.get_model_registry")
    def test_detects_service_returning_none(
        self, mock_mr, mock_dm, mock_ee, mock_ftp, mock_wo, mock_rs
    ):
        """A service getter returning None is marked unavailable."""
        mock_mr.return_value = None
        mock_dm.return_value = MagicMock()
        mock_ee.return_value = MagicMock()
        mock_ftp.return_value = MagicMock()
        mock_wo.return_value = MagicMock()
        mock_rs.return_value = MagicMock()

        statuses = check_service_availability()

        assert statuses["model_registry"].available is False
        assert statuses["model_registry"].error == "Service returned None"
        # Others should be available
        assert statuses["dataset_manager"].available is True

    @patch("dashboard.utils.backend.get_results_store")
    @patch("dashboard.utils.backend.get_workflow_orchestrator")
    @patch("dashboard.utils.backend.get_fine_tuning_pipeline")
    @patch("dashboard.utils.backend.get_evaluation_engine")
    @patch("dashboard.utils.backend.get_dataset_manager")
    @patch("dashboard.utils.backend.get_model_registry")
    def test_detects_service_raising_connection_error(
        self, mock_mr, mock_dm, mock_ee, mock_ftp, mock_wo, mock_rs
    ):
        """A service getter raising an exception is marked unavailable with error message."""
        mock_mr.return_value = MagicMock()
        mock_dm.return_value = MagicMock()
        mock_ee.side_effect = ConnectionError("Could not connect to DynamoDB endpoint")
        mock_ftp.return_value = MagicMock()
        mock_wo.return_value = MagicMock()
        mock_rs.return_value = MagicMock()

        statuses = check_service_availability()

        assert statuses["evaluation_engine"].available is False
        assert "DynamoDB" in statuses["evaluation_engine"].error
        assert statuses["model_registry"].available is True

    @patch("dashboard.utils.backend.get_results_store")
    @patch("dashboard.utils.backend.get_workflow_orchestrator")
    @patch("dashboard.utils.backend.get_fine_tuning_pipeline")
    @patch("dashboard.utils.backend.get_evaluation_engine")
    @patch("dashboard.utils.backend.get_dataset_manager")
    @patch("dashboard.utils.backend.get_model_registry")
    def test_detects_multiple_services_down(
        self, mock_mr, mock_dm, mock_ee, mock_ftp, mock_wo, mock_rs
    ):
        """Multiple services failing are all detected as unavailable."""
        mock_mr.side_effect = RuntimeError("Bedrock unavailable")
        mock_dm.return_value = MagicMock()
        mock_ee.return_value = MagicMock()
        mock_ftp.return_value = MagicMock()
        mock_wo.return_value = MagicMock()
        mock_rs.side_effect = OSError("S3 endpoint unreachable")

        statuses = check_service_availability()

        assert statuses["model_registry"].available is False
        assert "Bedrock" in statuses["model_registry"].error
        assert statuses["results_store"].available is False
        assert "S3" in statuses["results_store"].error
        # Remaining services should be available
        assert statuses["dataset_manager"].available is True
        assert statuses["evaluation_engine"].available is True

    @patch("dashboard.utils.backend.get_results_store")
    @patch("dashboard.utils.backend.get_workflow_orchestrator")
    @patch("dashboard.utils.backend.get_fine_tuning_pipeline")
    @patch("dashboard.utils.backend.get_evaluation_engine")
    @patch("dashboard.utils.backend.get_dataset_manager")
    @patch("dashboard.utils.backend.get_model_registry")
    def test_affected_features_populated_from_map(
        self, mock_mr, mock_dm, mock_ee, mock_ftp, mock_wo, mock_rs
    ):
        """Each status has affected_features matching SERVICE_FEATURE_MAP."""
        mock_mr.return_value = MagicMock()
        mock_dm.return_value = MagicMock()
        mock_ee.return_value = MagicMock()
        mock_ftp.return_value = MagicMock()
        mock_wo.return_value = MagicMock()
        mock_rs.return_value = MagicMock()

        statuses = check_service_availability()

        for name, status in statuses.items():
            assert status.affected_features == SERVICE_FEATURE_MAP[name]
