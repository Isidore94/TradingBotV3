"""P1-6 6d: entry-timing chips on D1 rows from the chart-watch stores. Read only.

Seen to fail before the change: no `entry_timing`, no `timing` column.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import entry_timing as et  # noqa: E402

TODAY = date(2026, 9, 24)


def _watch(symbol, side="LONG", kind="pullback", fired=None, declined=False):
    return {"symbol": symbol, "kind": kind, "side": side, "armed_at": "2026-09-22T08:00:00",
            "fired": dict(fired or {}), "declined": declined}


def _fire(symbol, side, trigger, timeframe, bar_dt):
    return {"action": "watch_fired", "symbol": symbol, "side": side, "ts": bar_dt,
            "detail": {"kind": "pullback", "trigger": trigger, "timeframe": timeframe,
                       "confirm_bar_dt": bar_dt}}


def test_an_h1_fire_reads_as_the_trader_says_it():
    """The H1 leg retires its watch, so its fire is only in the review-event rows."""
    mapping = et.build_timing([], [_fire("NVDA", "LONG", "h1_ema15_bounce", "H1", "2026-09-24T10:30:00")],
                              today=TODAY)
    assert mapping[("NVDA", "LONG")]["text"] == "timing: H1 15-EMA held 10:30"
    assert mapping[("NVDA", "LONG")]["state"] == et.FIRED


def test_an_armed_watch_with_an_sma_leg_fire_shows_the_fire():
    watches = [_watch("ALM", "SHORT", fired={"reclaim_then_lrsi@M30": "2026-09-23T10:15:00"})]
    mapping = et.build_timing(watches, [], today=TODAY)
    assert mapping[("ALM", "SHORT")]["text"] == "timing: reclaim then LRSI M30 09-23 10:15"


def test_armed_and_nothing_fired():
    mapping = et.build_timing([_watch("AMD", kind="h1_ema_bounce")], [], today=TODAY)
    assert mapping[("AMD", "LONG")] == {"state": et.ARMED, "at": None, "text": "timing: pullback armed"}


def test_the_newest_fire_wins_and_old_fires_are_dropped():
    watches = [_watch("NVDA", fired={"sma_retest@M15": "2026-09-24T09:45:00"})]
    fires = [
        _fire("NVDA", "LONG", "h1_ema15_bounce", "H1", "2026-09-24T10:30:00"),
        _fire("TSLA", "LONG", "h1_ema15_bounce", "H1", "2026-09-01T10:30:00"),  # too old
    ]
    mapping = et.build_timing(watches, fires, today=TODAY)
    assert mapping[("NVDA", "LONG")]["text"] == "timing: H1 15-EMA held 10:30"
    assert ("TSLA", "LONG") not in mapping


def test_declined_other_kinds_and_other_actions_are_ignored():
    watches = [_watch("X", declined=True), _watch("Y", kind="new_hod")]
    fires = [{"action": "veto", "symbol": "Z", "detail": {"kind": "pullback"}},
             {"action": "watch_fired", "symbol": "W", "detail": {"kind": "new_hod"}}]
    assert et.build_timing(watches, fires, today=TODAY) == {}


def test_a_watch_side_covers_its_side_and_a_sideless_one_covers_both():
    mapping = et.build_timing([_watch("NVDA", side="WATCH")], [], today=TODAY)
    assert et.timing_for(mapping, "NVDA", "SHORT")["text"] == "timing: pullback armed"
    mapping = et.build_timing([_watch("NVDA", side="LONG")], [], today=TODAY)
    assert et.timing_for(mapping, "NVDA", "SHORT") is None
    assert et.timing_for(mapping, "nvda", "LONG")["state"] == et.ARMED


def test_the_readers_are_read_only_and_tail_the_event_file(tmp_path):
    watches = tmp_path / "alert_chart_watches.json"
    watches.write_text(json.dumps({"market_date": "2026-09-24", "watches": [_watch("NVDA")]}), encoding="utf-8")
    before = watches.read_bytes()
    assert et.read_watch_rows(watches)[0]["symbol"] == "NVDA"
    assert watches.read_bytes() == before
    events = tmp_path / "review-events-x.jsonl"
    filler = json.dumps({"action": "like", "symbol": "AAA", "pad": "x" * 200})
    fire = json.dumps(_fire("NVDA", "LONG", "h1_ema15_bounce", "H1", "2026-09-24T10:30:00"))
    events.write_text("\n".join([filler] * 50 + [fire]) + "\n", encoding="utf-8")
    rows = et.read_fire_rows([events], tail_bytes=1_000)
    assert [row["symbol"] for row in rows] == ["NVDA"]
    assert et.read_watch_rows(tmp_path / "missing.json") == []
    assert et.read_fire_rows([tmp_path / "missing.jsonl"]) == []


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _rows():
    from ui.models.setup import SetupRow

    return [SetupRow(symbol="NVDA", side="LONG", bucket="favorite_setup", key_level="$100"),
            SetupRow(symbol="AMD", side="SHORT", bucket="favorite_setup", key_level="$50")]


def test_the_row_shows_the_chip_and_the_key_level_tooltip_carries_it(app):
    from PySide6.QtCore import Qt
    from ui.models.setup_table_model import SetupTableModel

    model = SetupTableModel(_rows())
    columns = [key for key, _label in model.COLUMNS]
    mapping = et.build_timing([], [_fire("NVDA", "LONG", "h1_ema15_bounce", "H1", "2026-09-24T10:30:00")],
                              today=TODAY)
    assert model.set_entry_timing(mapping) is True
    assert model.set_entry_timing(mapping) is False
    cell = model.data(model.index(0, columns.index("timing")), Qt.ItemDataRole.DisplayRole)
    assert cell == "H1 15-EMA held 10:30"
    assert model.data(model.index(1, columns.index("timing")), Qt.ItemDataRole.DisplayRole) == ""
    tooltip = model.data(model.index(0, columns.index("key_level")), Qt.ItemDataRole.ToolTipRole)
    assert tooltip.endswith("timing: H1 15-EMA held 10:30")
    assert model.has_timing()


def test_compact_hides_the_timing_column_full_shows_it(app):
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    panel = MasterAvwapPanel()
    try:
        panel.set_rows(_rows())
        column = [key for key, _label in panel.model.COLUMNS].index("timing")
        panel._on_entry_timing_ready(
            et.build_timing([_watch("NVDA")], [], today=datetime.now().date())
        )
        panel.set_column_profile("compact")
        assert panel.table.isColumnHidden(column)
        panel.set_column_profile("full")
        assert not panel.table.isColumnHidden(column)
        panel._on_entry_timing_ready({})
        assert panel.table.isColumnHidden(column)
    finally:
        panel.deleteLater()
