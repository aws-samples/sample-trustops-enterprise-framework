"""Dashboard theming and branding."""

from __future__ import annotations

LIGHT_THEME: dict[str, str] = {
    "primary": "#1a73e8",
    "secondary": "#5f6368",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "background": "#ffffff",
    "surface": "#f8f9fa",
    "text": "#202124",
    "text_secondary": "#5f6368",
    "border": "#dadce0",
}

DARK_THEME: dict[str, str] = {
    "primary": "#8ab4f8",
    "secondary": "#9aa0a6",
    "success": "#81c995",
    "warning": "#fdd663",
    "danger": "#f28b82",
    "background": "#202124",
    "surface": "#292a2d",
    "text": "#e8eaed",
    "text_secondary": "#9aa0a6",
    "border": "#3c4043",
}


def get_theme_colors(dark_mode: bool = False) -> dict[str, str]:
    """Return the colour palette for the requested mode."""
    return DARK_THEME if dark_mode else LIGHT_THEME


def _build_css(colors: dict[str, str]) -> str:
    """Build a CSS stylesheet string from a colour palette."""
    return f"""<style>
:root {{
    --primary: {colors['primary']};
    --surface: {colors['surface']};
    --text: {colors['text']};
    --border: {colors['border']};
}}
.stApp {{
    background-color: {colors['background']};
    color: {colors['text']};
}}
.status-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 500;
}}
.status-badge.success {{
    background-color:
 {colors['success']}20;
    color: {colors['success']};
}}
.status-badge.warning {{
    background-color: {colors['warning']}20;
    color: {colors['warning']};
}}
.status-badge.danger {{
    background-color: {colors['danger']}20;
    color: {colors['danger']};
}}
@media (max-width: 768px) {{
    .stApp {{
        padding: 0.5rem;
    }}
}}
</style>"""
