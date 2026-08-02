from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from project_paths import get_local_setting, save_local_setting


VALID_WORKSPACE_MODES = {"workspace", "tabs"}
VALID_THEMES = {"dark", "light"}
# "auto" reads the screen; the rest are explicit multipliers. Per-machine like
# every other qt_* setting, so the 4K desk and the MacBook keep their own value.
VALID_UI_SCALES = {"auto", "0.80", "0.85", "0.90", "0.95", "1.00", "1.10", "1.25"}


@dataclass
class UiState:
    workspace_mode: str = "workspace"
    theme_name: str = "dark"
    explain_mode: bool = True
    compact_density: bool = False
    nav_collapsed: bool = False
    ui_scale: str = "auto"

    @classmethod
    def load(cls) -> "UiState":
        workspace_mode = _choice("qt_workspace_mode", "workspace", VALID_WORKSPACE_MODES)
        theme_name = _choice("qt_theme", "dark", VALID_THEMES)
        return cls(
            workspace_mode=workspace_mode,
            theme_name=theme_name,
            explain_mode=bool(get_local_setting("qt_explain_mode", True)),
            compact_density=bool(get_local_setting("qt_compact_density", False)),
            nav_collapsed=bool(get_local_setting("qt_nav_collapsed", False)),
            ui_scale=_choice("qt_ui_scale", "auto", VALID_UI_SCALES),
        )

    def save(self) -> None:
        save_local_setting("qt_workspace_mode", self.workspace_mode)
        save_local_setting("qt_theme", self.theme_name)
        save_local_setting("qt_explain_mode", bool(self.explain_mode))
        save_local_setting("qt_compact_density", bool(self.compact_density))
        save_local_setting("qt_nav_collapsed", bool(self.nav_collapsed))
        save_local_setting("qt_ui_scale", self.ui_scale)


def _choice(key: str, default: str, valid: set[str]) -> str:
    value: Any = get_local_setting(key, default)
    normalized = str(value or "").strip().lower()
    return normalized if normalized in valid else default
