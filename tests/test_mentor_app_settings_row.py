"""P1 follow-up: with `mentor_app_enabled` on, the desk Settings page shows one line instead of
"Pause today" + "Next prompt" (the app owns the schedule); off = today's row."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the desk is Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture()
def panel():
    QApplication.instance() or QApplication([])
    from ui.panels.settings_panel import SettingsPanel
    from ui.state import UiState

    widget = SettingsPanel(UiState())
    widget.show()
    yield widget
    widget.close()
    widget.deleteLater()


def test_flag_off_keeps_pause_today_and_next_prompt(panel):
    panel.set_mentor_app_mode(False)
    assert not panel.trade_mentor_pause.isHidden()
    assert not panel.trade_mentor_next.isHidden()
    assert panel.trade_mentor_app_line.isHidden()


def test_flag_on_shows_one_line_pointing_at_the_app(panel):
    panel.set_mentor_app_mode(True)
    assert panel.trade_mentor_pause.isHidden()
    assert panel.trade_mentor_next.isHidden()
    assert not panel.trade_mentor_app_line.isHidden()
    assert panel.trade_mentor_app_line.text() == "Trade Mentor runs in its own app (`/pause` there)"


def test_the_default_is_todays_row(panel):
    assert not panel.trade_mentor_pause.isHidden() and panel.trade_mentor_app_line.isHidden()


def test_the_desk_window_hands_the_flag_to_the_settings_page():
    source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
    assert "self.settings_panel.set_mentor_app_mode(self.mentor_app_enabled)" in source
