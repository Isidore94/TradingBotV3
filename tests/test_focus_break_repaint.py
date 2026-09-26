"""B13: the Focus break-state signal fires only when a state actually changed.

The 60-second D1 poll used to emit `focusBreakStatesChanged` every tick, and
each emit repaints the whole Focus board on the Qt thread.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from ui.panels import alert_center_panel as panel_mod  # noqa: E402
from ui.panels.alert_center_panel import AlertCenterPanel  # noqa: E402


class _FocusService:
    def is_focus(self, symbol, side=None, category=None):
        return symbol == "NVDA"

    def focus_category(self, symbol):
        return "m5" if symbol == "NVDA" else None

    def focus_side(self, symbol, category=None):
        return "long" if symbol == "NVDA" else None

    def all_focus(self, category=None):
        return {"long": ["NVDA"], "short": []}


# Yesterday's high is 100.00.
_D1 = [
    {"dt": datetime(2026, 8, day, 0, 0), "high": 100.0 if day == 4 else 99.0,
     "low": 95.0, "close": 98.0}
    for day in (3, 4)
]


def _panel(tmp_path, monkeypatch, m5_bars):
    monkeypatch.setattr(
        panel_mod, "evaluate_d1_event_watch", lambda *args, **kwargs: None
    )
    panel = AlertCenterPanel(
        parked_symbols_path=tmp_path / "parked.json",
        focus_d1_flags_path=tmp_path / "focus_flags.json",
    )
    panel.focus_service = _FocusService()
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: list(_D1))
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol: list(m5_bars))
    monkeypatch.setattr(panel, "_m5_unknown", lambda symbol: False)
    emits: list[int] = []
    panel.focusBreakStatesChanged.connect(lambda: emits.append(1))
    return panel, emits


def test_identical_polls_emit_once_and_a_changed_state_emits_again(tmp_path, monkeypatch):
    m5_bars = [
        {"dt": datetime(2026, 8, 5, 9, 35), "high": 99.5, "low": 98.0, "close": 99.0},
    ]
    panel, emits = _panel(tmp_path, monkeypatch, m5_bars)

    # First poll always emits; a second poll with the same answer does not.
    panel._poll_focus_d1_interest(now=datetime(2026, 8, 5, 9, 45))
    panel._poll_focus_d1_interest(now=datetime(2026, 8, 5, 9, 46))
    assert panel.focus_break_state("NVDA", "long") == "closed"
    assert len(emits) == 1

    # The name closes above yesterday's high: the state changed, so it emits.
    m5_bars.append(
        {"dt": datetime(2026, 8, 5, 11, 0), "high": 101.5, "low": 99.5, "close": 101.2}
    )
    panel._poll_focus_d1_interest(now=datetime(2026, 8, 5, 11, 6))
    assert panel.focus_break_state("NVDA", "long") == "open"
    assert len(emits) == 2

    panel._poll_focus_d1_interest(now=datetime(2026, 8, 5, 11, 7))
    assert len(emits) == 2


def test_an_empty_first_poll_still_emits(tmp_path, monkeypatch):
    panel, emits = _panel(tmp_path, monkeypatch, [])
    panel.focus_service.all_focus = lambda category=None: {"long": [], "short": []}
    panel._poll_focus_d1_interest(now=datetime(2026, 8, 5, 9, 45))
    assert len(emits) == 1
