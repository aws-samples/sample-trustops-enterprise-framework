"""Tests for the degraded mode error handler."""

import pytest

from dashboard.utils.error_handler import (
    ServiceStatus,
    SERVICE_FEATURE_MAP,
    get_unavailable_features,
    is_feature_available,
)


class TestServiceStatus:
    """Tests for the ServiceStatus dataclass."""

    def test_create_available_service(self):
        status = ServiceStatus(name="test_service", available=True)
        assert status.name == "test_service"
        assert status.available is True
        assert status.error is None
        assert status.affected_features == []

    def test_create_unavailable_service(self):
        status = ServiceStatus(
            name="broken",
            available=False,
            error="Connection refused",
            affected_features=["Feature1"],
        )
        assert status.available is False
        assert status.error == "Connection refused"
        assert "Feature1" in status.affected_features


class TestServiceFeatureMap:
    """Tests for the service-to-feature mapping."""

    def test_has_all_services(self):
        expected_services = {
            "model_registry",
            "dataset_manager",
            "evaluation_engine",
            "fine_tuning_pipeline",
            "workflow_orchestrator",
            "results_store",
        }
        assert set(SERVICE_FEATURE_MAP.keys()) == expected_services

    def test_each_service_has_features(self):
        for service, features in SERVICE_FEATURE_MAP.items():
            assert isinstance(features, list)
            assert len(features) > 0

    def test_model_registry_affects_models(self):
        assert "Models" in SERVICE_FEATURE_MAP["model_registry"]

    def test_evaluation_engine_affects_evaluation(self):
        assert "Evaluation" in SERVICE_FEATURE_MAP["evaluation_engine"]


class TestGetUnavailableFeatures:
    """Tests for determining unavailable features."""

    def test_all_available(self):
        statuses = {
            "model_registry": ServiceStatus("model_registry", True, affected_features=["Models"]),
            "evaluation_engine": ServiceStatus("evaluation_engine", True, affected_features=["Evaluation"]),
        }
        result = get_unavailable_features(statuses)
        assert len(result) == 0

    def test_one_service_down(self):
        statuses = {
            "model_registry": ServiceStatus(
                "model_registry", False,
                affected_features=["Models", "Evaluation"],
            ),
            "evaluation_engine": ServiceStatus(
                "evaluation_engine", True,
                affected_features=["Evaluation"],
            ),
        }
        result = get_unavailable_features(statuses)
        assert "Models" in result
        assert "Evaluation" in result

    def test_empty_statuses(self):
        result = get_unavailable_features({})
        assert len(result) == 0


class TestIsFeatureAvailable:
    """Tests for checking individual feature availability."""

    def test_feature_available(self):
        statuses = {
            "model_registry": ServiceStatus(
                "model_registry", True,
                affected_features=["Models"],
            ),
        }
        assert is_feature_available("Models", statuses) is True

    def test_feature_unavailable(self):
        statuses = {
            "model_registry": ServiceStatus(
                "model_registry", False,
                affected_features=["Models"],
            ),
        }
        assert is_feature_available("Models", statuses) is False

    def test_unrelated_feature_available(self):
        statuses = {
            "model_registry": ServiceStatus(
                "model_registry", False,
                affected_features=["Models"],
            ),
        }
        # Workflows is not affected by model_registry
        assert is_feature_available("Workflows", statuses) is True

    def test_empty_statuses_means_available(self):
        assert is_feature_available("Anything", {}) is True
