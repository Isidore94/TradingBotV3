"""EVENING never fills the chart-review queue (trader, 2026-09-23).

On the flip out of EVENING the queue is EMPTY and fills normally from then
on. The backing lists, History and evidence are written as before; the
diverted alerts are held (references only) for the catch-up card.
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

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.models.bounce import BounceAlert  # noqa: E402


def _alert(symbol, side="LONG", *, trigger="[S-TIER] VWAP reclaim", d1=False):
    alert = BounceAlert(
        time_text="11:30:00",
        symbol=symbol,
        side=side,
        trigger=trigger,
        timeframe="5m",
        raw_text=f"[S-TIER] {symbol}: {trigger}",
    )
    alert.is_d1 = d1
    return alert


def _panel(monkeypatch, mode):
    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: mode)
    panel = AlertCenterPanel()
    panel._auto_mode_cached = None
    monkeypatch.setattr(panel, "mover_state", lambda symbol, side="": "open")
    return panel


def test_a_d1_chart_does_not_queue_in_evening(monkeypatch):
    panel = _panel(monkeypatch, "EVENING")
    panel._enqueue_review_alert(_alert("AAA", trigger="MASTER_AVWAP_D1_ZONE: zone1", d1=True))
    assert panel._review_queue == []
    assert [alert.symbol for alert in panel.evening_catchup_alerts()] == ["AAA"]


def test_the_flip_out_of_evening_empties_the_queue_and_desk_fills_it_again(monkeypatch):
    panel = _panel(monkeypatch, "EVENING")
    stale = _alert("OLD", trigger="MASTER_AVWAP_D1_ZONE: zone1", d1=True)
    panel._review_queue = [stale]

    monkeypatch.setattr("autopilot_core.read_auto_pilot_mode", lambda *_a, **_k: "DESK")
    panel.on_auto_mode_changed("EVENING", "DESK")
    assert panel._review_queue == []

    monkeypatch.setattr(
        panel, "_is_ordinary_d1_scan_review", lambda alert: False
    )
    panel._current_review_alert = _alert("ONSCREEN")
    panel._enqueue_review_alert(_alert("NEW", trigger="MASTER_AVWAP_D1_ZONE: zone1", d1=True))
    queued = [alert.symbol for alert in panel._review_queue]
    assert queued == ["NEW"]


def test_the_flip_takes_effect_before_the_mode_cache_expires(monkeypatch):
    panel = _panel(monkeypatch, "DESK")
    assert panel._auto_mode_now() == "DESK"
    panel.on_auto_mode_changed("DESK", "EVENING")
    assert panel._auto_mode_now() == "EVENING"


def test_entering_evening_starts_a_fresh_catchup_list(monkeypatch):
    panel = _panel(monkeypatch, "EVENING")
    panel._enqueue_review_alert(_alert("AAA"))
    panel.on_auto_mode_changed("DESK", "EVENING")
    assert panel.evening_catchup_alerts() == []
    assert panel.evening_started_at() is not None
