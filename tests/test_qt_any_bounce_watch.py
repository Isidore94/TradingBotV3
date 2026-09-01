"""R5 section 4 in the desk: arming, persisting and firing the any-bounce watch.

The evaluation rules are proven in ``test_any_bounce_watch.py``. What matters
here is ownership and lifecycle: the Alert Center is the ONE component that
writes this store, a second click disarms, the fired watch retires itself and
lands in the feed as a chart-watch alert naming the level that held.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

OPEN = datetime(2026, 8, 17, 6, 30)


def _panel(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - headless without Qt
        pytest.skip("PySide6 not installed")
    QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    return AlertCenterPanel()


def _bars():
    """Two completed bars that tag 100.0 and reclaim it."""
    rows = [(101.0, 99.98, 100.05), (101.5, 100.2, 101.2)]
    return [
        {
            "dt": OPEN + timedelta(minutes=5 * index),
            "open": row[2],
            "high": row[0],
            "low": row[1],
            "close": row[2],
        }
        for index, row in enumerate(rows)
    ]


def test_arming_is_a_toggle_and_the_panel_owns_the_store(monkeypatch, tmp_path):
    panel = _panel(monkeypatch)
    panel._any_bounce_watches_path = tmp_path / "any_bounce_watches.json"

    assert panel.arm_any_bounce_watch("aapl", "long") is True
    assert panel.any_bounce_armed_for("AAPL") is True
    # Arming twice is not two watches.
    assert panel.arm_any_bounce_watch("AAPL", "long") is False
    assert len(panel._any_bounce_watches) == 1

    from chart_watch import load_any_bounce_watches

    stored = load_any_bounce_watches(panel._any_bounce_watches_path)
    assert [watch.symbol for watch in stored] == ["AAPL"]

    assert panel.disarm_any_bounce_watch("AAPL") is True
    assert panel.any_bounce_armed_for("AAPL") is False
    assert load_any_bounce_watches(panel._any_bounce_watches_path) == []


def test_a_bounce_fires_once_names_the_level_and_retires(monkeypatch, tmp_path):
    panel = _panel(monkeypatch)
    panel._any_bounce_watches_path = tmp_path / "any_bounce_watches.json"
    panel.arm_any_bounce_watch("AAPL", "long")

    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol: _bars())
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: [])
    monkeypatch.setattr(
        panel,
        "_any_bounce_levels_for",
        lambda symbol, moment, **kwargs: {"d1_ema15": 100.0},
    )

    before = len(panel._alerts)
    panel._poll_any_bounce_watches(now=OPEN + timedelta(minutes=10))

    assert panel.any_bounce_armed_for("AAPL") is False
    assert len(panel._alerts) == before + 1
    fired = panel._alerts[0]
    assert fired.symbol == "AAPL"
    assert "D1 15 EMA" in fired.trigger
    # One shot: a second poll on the same bars cannot fire again.
    panel._poll_any_bounce_watches(now=OPEN + timedelta(minutes=10))
    assert len(panel._alerts) == before + 1


def test_nothing_fires_without_a_bounce(monkeypatch, tmp_path):
    panel = _panel(monkeypatch)
    panel._any_bounce_watches_path = tmp_path / "any_bounce_watches.json"
    panel.arm_any_bounce_watch("AAPL", "long")

    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol: _bars())
    monkeypatch.setattr(
        panel,
        "_any_bounce_levels_for",
        lambda symbol, moment: {"d1_ema15": 90.0},
    )
    panel._poll_any_bounce_watches(now=OPEN + timedelta(minutes=10))

    assert panel.any_bounce_armed_for("AAPL") is True


def test_a_symbol_with_no_levels_keeps_waiting(monkeypatch, tmp_path):
    """No zone-arms entry and no bars is uncertainty, not a disarm."""
    panel = _panel(monkeypatch)
    panel._any_bounce_watches_path = tmp_path / "any_bounce_watches.json"
    panel.arm_any_bounce_watch("AAPL", "long")

    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol: [])
    monkeypatch.setattr(panel, "_any_bounce_levels_for", lambda symbol, moment: {})
    panel._poll_any_bounce_watches(now=OPEN + timedelta(minutes=10))

    assert panel.any_bounce_armed_for("AAPL") is True


def test_the_level_set_reads_the_scans_prior_anchor_avwap(monkeypatch):
    """R5 section 8.3's new key is what makes the prior AVWAP watchable."""
    panel = _panel(monkeypatch)
    entry = {"avwape": 101.0, "prev_avwape": 97.5, "trigger_levels": []}

    class _Bot:
        d1_zone_arms = {"AAPL": entry}

    monkeypatch.setattr(panel, "_current_bot", lambda: _Bot())
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol: [])
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: [])

    levels = panel._any_bounce_levels_for("AAPL", OPEN)
    assert levels["prev_avwape"] == 97.5
    assert levels["avwape"] == 101.0
