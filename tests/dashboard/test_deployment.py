"""Tests for dashboard deployment configuration."""

import json
import os

import pytest

from dashboard.health import get_health_status, health_check_json


class TestHealthCheck:
    """Tests for the health check endpoint."""

    def test_returns_dict(self):
        status = get_health_status()
        assert isinstance(status, dict)

    def test_has_status_field(self):
        status = get_health_status()
        assert "status" in status
        assert status["status"] in ("healthy", "degraded")

    def test_has_timestamp(self):
        status = get_health_status()
        assert "timestamp" in status

    def test_has_checks(self):
        status = get_health_status()
        assert "checks" in status
        assert isinstance(status["checks"], dict)

    def test_has_version(self):
        status = get_health_status()
        assert "version" in status
        assert status["version"] == "1.0.0"

    def test_streamlit_check(self):
        status = get_health_status()
        assert "streamlit" in status["checks"]
        assert status["checks"]["streamlit"] == "ok"

    def test_plotly_check(self):
        status = get_health_status()
        assert "plotly" in status["checks"]
        assert status["checks"]["plotly"] == "ok"

    def test_healthy_when_all_ok(self):
        status = get_health_status()
        # Both streamlit and plotly should be installed in test env
        assert status["status"] == "healthy"


class TestHealthCheckJson:
    """Tests for JSON health check output."""

    def test_returns_valid_json(self):
        result = health_check_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_json_has_status(self):
        result = health_check_json()
        parsed = json.loads(result)
        assert "status" in parsed


class TestDockerfile:
    """Tests for Dockerfile existence and content."""

    def test_dockerfile_exists(self):
        assert os.path.exists("dashboard/Dockerfile")

    def test_dockerfile_has_python_base(self):
        with open("dashboard/Dockerfile") as f:
            content = f.read()
        assert "python:3.11" in content

    def test_dockerfile_exposes_port(self):
        with open("dashboard/Dockerfile") as f:
            content = f.read()
        assert "EXPOSE 8501" in content

    def test_dockerfile_has_healthcheck(self):
        with open("dashboard/Dockerfile") as f:
            content = f.read()
        assert "HEALTHCHECK" in content

    def test_dockerfile_has_entrypoint(self):
        with open("dashboard/Dockerfile") as f:
            content = f.read()
        assert "ENTRYPOINT" in content or "CMD" in content


class TestDockerCompose:
    """Tests for docker-compose.yml existence and content."""

    def test_docker_compose_exists(self):
        assert os.path.exists("dashboard/docker-compose.yml")

    def test_docker_compose_has_dashboard_service(self):
        with open("dashboard/docker-compose.yml") as f:
            content = f.read()
        assert "dashboard" in content

    def test_docker_compose_maps_port(self):
        with open("dashboard/docker-compose.yml") as f:
            content = f.read()
        assert "8501:8501" in content


class TestStreamlitConfig:
    """Tests for Streamlit configuration file."""

    def test_config_exists(self):
        assert os.path.exists(".streamlit/config.toml")

    def test_config_has_server_section(self):
        with open(".streamlit/config.toml") as f:
            content = f.read()
        assert "[server]" in content

    def test_config_has_theme_section(self):
        with open(".streamlit/config.toml") as f:
            content = f.read()
        assert "[theme]" in content

    def test_config_disables_usage_stats(self):
        with open(".streamlit/config.toml") as f:
            content = f.read()
        assert "gatherUsageStats = false" in content
