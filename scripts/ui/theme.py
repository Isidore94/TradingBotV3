from __future__ import annotations

import sys
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
        # Chart-line palette (user-specified 2026-07-29): the D1 overlays are
        # read by color first, label second, so these are fixed assignments -
        # SMA200 purple / SMA100 pink / SMA50 light blue, EMA15 pink /
        # EMA21 yellow / EMA8 grey, AVWAPE white (prev yellow), bands
        # blue / green / light blue for 1/2/3σ.
        "chart_purple": "#A855F7",
        "chart_pink": "#F472B6",
        "chart_light_blue": "#7DD3FC",
        "chart_yellow": "#FDE047",
        "chart_grey": "#9AA4B2",
        "chart_white": "#F5F7FA",
        "chart_blue": "#3B82F6",
        "chart_green": "#34D399",
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
        # Same assignments, darkened where the dark-theme value would wash
        # out on a white panel (white -> near-black keeps AVWAPE readable).
        "chart_purple": "#7E22CE",
        "chart_pink": "#DB2777",
        "chart_light_blue": "#0284C7",
        "chart_yellow": "#CA8A04",
        "chart_grey": "#64748B",
        "chart_white": "#1F2937",
        "chart_blue": "#1D4ED8",
        "chart_green": "#059669",
    },
}


_ACTIVE_THEME = "dark"


def tokens(theme_name: str) -> dict[str, str]:
    values = dict(THEMES.get(theme_name, THEMES["dark"]))
    values.update(_derived_tokens(values))
    return values


def _derived_tokens(values: Mapping[str, str]) -> dict[str, str]:
    """Pre-mixed rgba tokens the stylesheet needs but qss cannot compute.

    These existed as f-strings inside widget constructors, which meant Qt
    parsed a fresh stylesheet for every alert row and every focus chip - up to
    250 rows at a time on a feed rebuild. Mixing them here instead lets the
    same colours live in theme.qss and be parsed once, at startup.

    Named for the surface rather than the colour, so a theme change moves them
    together with everything else.
    """
    short = values.get("short", "#ff5555")
    favorite = values.get("favorite", "#ffc857")
    return {
        "alert_watch_border": with_alpha(short, 0.90),
        "alert_watch_bg": with_alpha(short, 0.12),
        "alert_focus_border": with_alpha(favorite, 0.85),
        "alert_focus_bg": with_alpha(favorite, 0.14),
        "alert_star_dim": with_alpha(favorite, 0.75),
        "alert_dislike": with_alpha(short, 0.65),
    }


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


def ui_font_family() -> str:
    """Native UI font per platform.

    Windows keeps Segoe UI. Naming a family the platform does not ship makes Qt
    populate every font alias looking for it (~60 ms at startup) before falling
    back, so macOS asks for the system font by its real family name.
    """
    if sys.platform == "darwin":
        return '".AppleSystemUIFont"'
    if sys.platform.startswith("win"):
        return '"Segoe UI"'
    return "sans-serif"


# ----------------------------------------------------------------------
# UI scale
#
# Every size in the shell is expressed at scale 1.0 = the 4K desktop's
# appearance, then multiplied through. Fonts are emitted in px rather than pt
# on purpose: Qt reports 96 logical DPI on Windows but 72 on macOS, so the same
# "10.5pt" rule rendered 14px of text on the desk and 10.5px on the MacBook -
# tiny type inside chrome that stayed the same size. px is device-independent
# on both, so one number means one appearance and the scale factor is the only
# thing that changes it.
# ----------------------------------------------------------------------

VALID_SCALES = ("auto", "0.80", "0.85", "0.90", "0.95", "1.00", "1.10", "1.25")
MIN_SCALE, MAX_SCALE = 0.7, 1.5

# px at scale 1.0.
BASE_METRICS: dict[str, float] = {
    "font_body": 14,
    "font_title": 23,
    "font_section": 17,
    "font_setup": 16,
    "row_height": 32,
    "row_height_compact": 24,
    "pad_small": 6,
    "pad_small_compact": 4,
    "pad_medium": 12,
    "pad_medium_compact": 8,
    "cell_pad": 6,
    "radius": 8,
    "radius_panel": 10,
    "header_pad_v": 7,
    "header_pad_h": 8,
    "nav_pad_v": 9,
    "nav_pad_h": 12,
    "tab_pad_v": 8,
    "tab_pad_h": 14,
    "scrollbar": 12,
}

_ACTIVE_SCALE = 1.0


def auto_scale_for(width: int, height: int) -> float:
    """Pick a scale from the screen's available *logical* size.

    Logical, not physical: the 4K desk reports ~2560x1440 of workspace after
    Windows display scaling, while the MacBook reports 1680x954 no matter how
    dense the panel is. The desk's three-column workspace needs roughly
    1900 logical px to lay out without squeezing control rows, so anything
    narrower gets proportionally smaller chrome instead of clipped widgets.
    """
    width = int(width or 0)
    height = int(height or 0)
    if width >= 2400 and height >= 1300:
        return 1.00
    if width >= 1900 and height >= 1050:
        return 0.95
    if width >= 1560:
        return 0.85
    return 0.80


def resolve_scale(setting: str, screen_size: tuple[int, int] | None = None) -> float:
    """Turn a stored ui_scale setting into a usable multiplier."""
    raw = str(setting or "auto").strip().lower()
    if raw in ("", "auto"):
        if screen_size is None:
            return 1.0
        return auto_scale_for(*screen_size)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    # Tolerate "90" as well as "0.90" so a percentage typed into settings works.
    # The threshold is 10, not MAX_SCALE: reading "9.0" as 9% would clamp a
    # too-large value to the SMALLEST shell, which is the opposite of asked.
    if value >= 10:
        value = value / 100.0
    return max(MIN_SCALE, min(MAX_SCALE, value))


def active_scale() -> float:
    return _ACTIVE_SCALE


def px(value: float) -> int:
    """Scale a design pixel for use in Python-side sizing (min widths, icons).

    Widgets that hard-code a pixel budget must route it through here, or the
    stylesheet shrinks while the widget's own floor does not - which is how a
    1680px-wide desk ends up with truncated button rows.
    """
    return max(1, round(float(value) * _ACTIVE_SCALE))


def metrics(compact: bool = False) -> dict[str, int]:
    """Every scaled shell metric, as whole pixels."""
    values = {name: px(base) for name, base in BASE_METRICS.items()}
    if compact:
        values["row_height"] = values["row_height_compact"]
        values["pad_small"] = values["pad_small_compact"]
        values["pad_medium"] = values["pad_medium_compact"]
    return values


def build_stylesheet(
    theme_name: str = "dark", compact: bool = False, scale: float | None = None
) -> str:
    global _ACTIVE_SCALE
    if scale is not None:
        _ACTIVE_SCALE = max(MIN_SCALE, min(MAX_SCALE, float(scale)))
    values = tokens(theme_name)
    values["ui_font"] = ui_font_family()
    for name, size in metrics(compact).items():
        values[name] = f"{size}px"
    # Kept for any stylesheet still using the pre-scale token names.
    values["padding_small"] = values["pad_small"]
    values["padding_medium"] = values["pad_medium"]
    template = (Path(__file__).with_name("theme.qss")).read_text(encoding="utf-8")
    return _replace_tokens(template, values)


def _replace_tokens(template: str, values: Mapping[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"@{key}@", value)
    return rendered


def apply_theme(
    app, theme_name: str = "dark", compact: bool = False, scale: float | None = None
) -> None:
    global _ACTIVE_THEME
    _ACTIVE_THEME = theme_name if theme_name in THEMES else "dark"
    app.setStyleSheet(build_stylesheet(_ACTIVE_THEME, compact, scale))
