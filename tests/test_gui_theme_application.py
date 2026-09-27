"""State changes only repolish Qt when the rendered theme actually changes."""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ui import theme  # noqa: E402


class StyledTarget:
    def __init__(self):
        self.sheet = ""
        self.writes = []

    def styleSheet(self):  # noqa: N802
        return self.sheet

    def setStyleSheet(self, sheet):  # noqa: N802
        self.sheet = sheet
        self.writes.append(sheet)


def test_repeated_theme_skips_repolish_but_updates_active_state(monkeypatch):
    monkeypatch.setattr(theme, "_ACTIVE_THEME", "dark")
    monkeypatch.setattr(theme, "_ACTIVE_SCALE", 1.0)
    target = StyledTarget()
    theme.apply_theme(target, "dark", scale=1.0)
    theme.apply_theme(target, "dark", scale=1.0)
    assert len(target.writes) == 1
    theme._ACTIVE_THEME, theme._ACTIVE_SCALE = "light", .8
    theme.apply_theme(target, "dark", scale=1.0)
    assert len(target.writes) == 1
    assert theme.active_theme() == "dark"
    assert theme.active_scale() == 1.0


def test_theme_density_scale_and_external_style_changes_still_apply(monkeypatch):
    monkeypatch.setattr(theme, "_ACTIVE_THEME", "dark")
    monkeypatch.setattr(theme, "_ACTIVE_SCALE", 1.0)
    target = StyledTarget()
    for name, compact, scale in (
        ("dark", False, 1.0), ("light", False, 1.0),
        ("light", True, 1.0), ("light", True, .8),
    ):
        previous = len(target.writes)
        theme.apply_theme(target, name, compact, scale)
        assert len(target.writes) == previous + 1
    target.sheet = "externally replaced"
    theme.apply_theme(target, "light", True, .8)
    assert len(target.writes) == 5
    assert target.sheet == target.writes[-1]
