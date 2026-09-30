"""Pause AI on the desk: the Settings > General row and the status-bar pill."""

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

import ai_pause  # noqa: E402
import project_paths  # noqa: E402


@pytest.fixture(autouse=True)
def scratch_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_FILE", tmp_path / "local_settings.json")
    project_paths.invalidate_local_settings_cache()
    QApplication.instance() or QApplication([])
    yield
    project_paths.invalidate_local_settings_cache()


@pytest.fixture()
def panel():
    from ui.panels.settings_panel import SettingsPanel
    from ui.state import UiState

    widget = SettingsPanel(UiState())
    widget.show()
    yield widget
    widget.close()
    widget.deleteLater()


def test_settings_general_has_the_pause_ai_row(panel):
    row = panel.ai_pause_row
    assert row.button.text() == "Pause AI"
    assert row.label.text() == "AI is on"
    assert [a.text() for a in row.button.menu().actions() if a.text()] == [
        "For 2 hours", "For 4 hours", "Until 06:00", "Until I resume", "Resume AI"]
    assert not row.button.resume_action.isEnabled()


def test_the_row_pauses_and_resumes_through_the_one_setting(panel):
    row = panel.ai_pause_row
    row.button.pause_actions["2h"].trigger()
    assert ai_pause.is_paused()
    assert row.label.text().startswith("AI paused until ")
    assert row.button.text() == "AI paused" and row.button.resume_action.isEnabled()
    row.button.resume_action.trigger()
    assert not ai_pause.is_paused()
    assert row.label.text() == "AI is on"


def test_until_i_resume_reads_so(panel):
    panel.ai_pause_row.button.pause_actions["until_resumed"].trigger()
    assert panel.ai_pause_row.label.text() == "AI paused until you resume"


def test_the_status_pill_shows_only_while_paused():
    from ui.widgets.ai_pause_control import REFRESH_MS, AiPausedPill

    pill = AiPausedPill()
    assert pill.isHidden()
    ai_pause.pause_for("4h")
    pill.refresh()
    assert not pill.isHidden() and pill.text() == "AI paused"
    assert pill.toolTip().startswith("AI paused until ")
    assert pill._timer.isActive() and pill._timer.interval() == REFRESH_MS == 5000
    ai_pause.resume()
    pill.refresh()
    assert pill.isHidden()
    pill.deleteLater()


def test_the_desk_status_bar_carries_the_pill():
    source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
    assert "self.ai_paused_pill = AiPausedPill(self)" in source
    assert "status.addPermanentWidget(self.ai_paused_pill)" in source
