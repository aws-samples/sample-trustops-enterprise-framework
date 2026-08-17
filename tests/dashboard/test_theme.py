"""Tests for dashboard theming and branding."""

import pytest

from dashboard.styles.theme import (
    LIGHT_THEME,
    DARK_THEME,
    get_theme_colors,
    _build_css,
)


class TestThemeColors:
    """Tests for theme colour palettes."""

    def test_light_theme_has_required_keys(self):
        required = {"primary", "secondary", "success", "warning", "danger",
                     "background", "surface", "text", "text_secondary", "border"}
        assert required.issubset(set(LIGHT_THEME.keys()))

    def test_dark_theme_has_required_keys(self):
        required = {"primary", "secondary", "success", "warning", "danger",
                     "background", "surface", "text", "text_secondary", "border"}
        assert required.issubset(set(DARK_THEME.keys()))

    def test_light_theme_values_are_strings(self):
        for key, value in LIGHT_THEME.items():
            assert isinstance(value, str)

    def test_dark_theme_values_are_strings(self):
        for key, value in DARK_THEME.items():
            assert isinstance(value, str)

    def test_themes_are_different(self):
        assert LIGHT_THEME["background"] != DARK_THEME["background"]
        assert LIGHT_THEME["text"] != DARK_THEME["text"]


class TestGetThemeColors:
    """Tests for get_theme_colors function."""

    def test_default_returns_light(self):
        colors = get_theme_colors()
        assert colors == LIGHT_THEME

    def test_light_mode(self):
        colors = get_theme_colors(dark_mode=False)
        assert colors == LIGHT_THEME

    def test_dark_mode(self):
        colors = get_theme_colors(dark_mode=True)
        assert colors == DARK_THEME


class TestBuildCss:
    """Tests for CSS generation."""

    def test_returns_string(self):
        css = _build_css(LIGHT_THEME)
        assert isinstance(css, str)

    def test_contains_style_tag(self):
        css = _build_css(LIGHT_THEME)
        assert "<style>" in css
        assert "</style>" in css

    def test_contains_responsive_media_query(self):
        css = _build_css(LIGHT_THEME)
        assert "@media" in css
        assert "768px" in css

    def test_contains_status_badge_classes(self):
        css = _build_css(LIGHT_THEME)
        assert ".status-badge" in css
        assert ".status-badge.success" in css
        assert ".status-badge.warning" in css
        assert ".status-badge.danger" in css

    def test_uses_theme_colors(self):
        css = _build_css(LIGHT_THEME)
        assert LIGHT_THEME["primary"] in css
        assert LIGHT_THEME["surface"] in css

    def test_dark_theme_css_uses_dark_colors(self):
        css = _build_css(DARK_THEME)
        assert DARK_THEME["primary"] in css
        assert DARK_THEME["surface"] in css
