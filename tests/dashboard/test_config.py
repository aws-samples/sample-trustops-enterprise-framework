"""Tests for dashboard configuration."""

import pytest

from dashboard.config import DashboardConfig, get_config


class TestDashboardConfig:
    """Tests for DashboardConfig."""

    def test_default_config(self):
        config = DashboardConfig()
        assert config.page_title == "TrustOps Enterprise"
        assert config.page_icon == "🛡️"
        assert config.layout == "wide"

    def test_default_pages(self):
        config = DashboardConfig()
        assert "Home" in config.pages
        assert "Models" in config.pages
        assert "Datasets" in config.pages
        assert "Evaluation" in config.pages
        assert "Comparison" in config.pages
        assert "Fine-Tuning" in config.pages
        assert "Workflows" in config.pages
        assert len(config.pages) == 7

    def test_get_config_singleton(self):
        # Reset singleton
        import dashboard.config as cfg_mod
        cfg_mod._config = None

        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_custom_config(self):
        config = DashboardConfig(
            page_title="Custom Title",
            aws_region="eu-west-1",
            cache_ttl=600,
        )
        assert config.page_title == "Custom Title"
        assert config.aws_region == "eu-west-1"
        assert config.cache_ttl == 600
