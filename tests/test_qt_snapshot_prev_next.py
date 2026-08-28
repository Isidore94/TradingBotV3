"""The setups-table chart popup walks the list with Prev / Next (trader, 2026-08-27).

"This is the chart I get when I double click a chart in Master AVWAP setups.
I'd like a next or previous button so I can continue cycling down the list."
Space on the table already advanced; the buttons put the same walk on the
popup. Both only MOVE - no dislike, no favorite, no evidence row.
"""

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

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture
def setups(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    import chart_snapshot
    from ui.models.setup import SetupRow
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _s: [])
    panel = MasterAvwapPanel(None, review_events_path=tmp_path / "events.jsonl")
    panel.set_rows(
        [
            SetupRow(symbol="NVDA", side="LONG", score=90.0),
            SetupRow(symbol="TSLA", side="SHORT", score=80.0),
            SetupRow(symbol="AMD", side="LONG", score=70.0),
        ]
    )
    first = panel.proxy.index(0, 2)
    panel.table.setCurrentIndex(first)
    panel._open_symbol_snapshot(first)
    yield panel, panel._symbol_snapshot_dialog
    panel.close()


def test_the_buttons_show_only_in_a_review_walk(setups):
    _panel, dialog = setups
    assert dialog.next_button.isVisibleTo(dialog)
    assert dialog.previous_button.isVisibleTo(dialog)
    # A plain typed lookup has no list to walk.
    dialog.show_symbol("AAPL")
    assert not dialog.next_button.isVisibleTo(dialog)
    assert not dialog.previous_button.isVisibleTo(dialog)


def test_next_walks_down_and_wraps(setups):
    panel, dialog = setups
    assert dialog._symbol == "NVDA"
    dialog.next_button.click()
    assert dialog._symbol == "TSLA" and panel.table.currentIndex().row() == 1
    dialog.next_button.click()
    assert dialog._symbol == "AMD"
    dialog.next_button.click()
    assert dialog._symbol == "NVDA", "wraps rather than dead-ending"


def test_previous_walks_up_and_wraps(setups):
    panel, dialog = setups
    dialog.previous_button.click()
    assert dialog._symbol == "AMD" and panel.table.currentIndex().row() == 2
    dialog.previous_button.click()
    assert dialog._symbol == "TSLA"
    dialog.next_button.click()
    assert dialog._symbol == "AMD"


def test_the_walk_records_nothing(setups, tmp_path):
    from review_events import load_review_events

    _panel, dialog = setups
    dialog.next_button.click()
    dialog.previous_button.click()
    assert load_review_events(tmp_path / "events.jsonl") == []


def test_the_side_travels_with_the_chart(setups):
    _panel, dialog = setups
    dialog.next_button.click()
    assert dialog._side == "SHORT"
    assert "TSLA (SHORT)" in dialog.windowTitle()
