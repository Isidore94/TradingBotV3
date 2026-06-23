from __future__ import annotations

from pathlib import Path
from typing import Mapping


THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "bg_app": "#0F1216",
        "bg_panel": "#171B21",
        "bg_elevated": "#1E242C",
        "bg_hover": "#222936",
        "border": "#2A313B",
        "text_primary": "#E6EAF0",
        "text_secondary": "#9AA4B2",
        "text_muted": "#6B7480",
        "accent": "#4C8DFF",
        "accent_soft": "#20365F",
        "long": "#2ECC71",
        "short": "#FF5C5C",
        "caution": "#F5A623",
        "info": "#4C8DFF",
        "neutral": "#6B7480",
        "favorite": "#E9B949",
        "near": "#6EA8FF",
        "study": "#9B7CFF",
        "input_bg": "#12161C",
        "selection": "#244B86",
    },
    "light": {
        "bg_app": "#F4F6F9",
        "bg_panel": "#FFFFFF",
        "bg_elevated": "#EEF2F7",
        "bg_hover": "#E5EBF4",
        "border": "#D2D9E4",
        "text_primary": "#111827",
        "text_secondary": "#475569",
        "text_muted": "#7A8697",
        "accent": "#2563EB",
        "accent_soft": "#DCEAFE",
        "long": "#168A46",
        "short": "#D93B3B",
        "caution": "#B87408",
        "info": "#2563EB",
        "neutral": "#64748B",
        "favorite": "#B7791F",
        "near": "#2563EB",
        "study": "#7C3AED",
        "input_bg": "#F8FAFC",
        "selection": "#BFD7FF",
    },
}


_ACTIVE_THEME = "dark"


def tokens(theme_name: str) -> dict[str, str]:
    return dict(THEMES.get(theme_name, THEMES["dark"]))


def active_theme() -> str:
    return _ACTIVE_THEME


def color(name: str, theme_name: str | None = None) -> str:
    """Return a semantic token color for the active (or given) theme.

    Widgets and table models should read colors through this accessor instead of
    hard-coding hex so the light/dark toggle restyles every surface.
    """
    values = THEMES.get(theme_name or _ACTIVE_THEME, THEMES["dark"])
    return values.get(name, values["neutral"])


def with_alpha(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` to an ``rgba(...)`` string (alpha in 0..1)."""
    raw = hex_color.lstrip("#")
    if len(raw) != 6:
        return hex_color
    try:
        red, green, blue = (int(raw[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return hex_color
    return f"rgba({red}, {green}, {blue}, {max(0.0, min(1.0, alpha)):.3f})"


def build_stylesheet(theme_name: str = "dark", compact: bool = False) -> str:
    values = tokens(theme_name)
    values["row_height"] = "24px" if compact else "32px"
    values["padding_small"] = "4px" if compact else "6px"
    values["padding_medium"] = "8px" if compact else "12px"
    template = (Path(__file__).with_name("theme.qss")).read_text(encoding="utf-8")
    return _replace_tokens(template, values)


def _replace_tokens(template: str, values: Mapping[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"@{key}@", value)
    return rendered


def apply_theme(app, theme_name: str = "dark", compact: bool = False) -> None:
    global _ACTIVE_THEME
    _ACTIVE_THEME = theme_name if theme_name in THEMES else "dark"
    app.setStyleSheet(build_stylesheet(_ACTIVE_THEME, compact))
