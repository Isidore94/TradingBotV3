"""The Alert Center's wall leg (trader, 2026-09-23).

"Don't show me stocks right at SMAs (within 1 ATR). Instead auto-set an alert
... Same with trendline breaks." `test_wall_gate.py` proves the rule; this
proves the panel hides only what it could follow up on, arms existing watch
kinds without duplicating or re-arming a declined one, caps the count, and
logs every hide.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
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

from ui.models.bounce import BounceAlert  # noqa: E402


def _daily(segments, today):
    closes = []
    for count, first, last in segments:
        for i in range(count):
            step = 0.0 if count == 1 else (last - first) * i / (count - 1)
            closes.append(first + step)
    start = today.replace(hour=0, minute=0) - timedelta(days=len(closes))
    return [
        {"dt": start + timedelta(days=i), "open": c, "high": c + 0.5, "low": c - 0.5,
         "close": c, "volume": 1000.0}
        for i, c in enumerate(closes)
    ]


def _m5(closes, today):
    return [
        {"dt": today - timedelta(minutes=10 * (len(closes) - i)), "open": c,
         "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 100.0}
        for i, c in enumerate(closes)
    ]


# SMA50 = 20, SMA100 = 21, SMA200 = 19.5, ATR20 ~ 1: a D1 long at 20.8 is above
# its SMA200 (the trend leg passes) and 0.2 ATR under its SMA100 - a wall.
STEP_DOWN = [(150, 18.0, 18.0), (50, 22.0, 22.0), (50, 20.0, 20.0)]
# SMA200 = 19, SMA50 = 20: a long at 20.8 has only support under it.
STEP_UP = [(150, 18.0, 18.0), (100, 20.0, 20.0)]
AT_WALL = [20.6, 20.7, 20.8]


def _d1_alert(symbol, side="LONG"):
    word = side.lower()
    return BounceAlert(
        time_text="08:25:00",
        symbol=symbol,
        side=side,
        trigger=f"({word}) zone1 reject at AVWAPE",
        timeframe="D1",
        tag=f"d1_flag_{word}",
        raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} ({word}) zone1 reject at AVWAPE",
        is_d1=True,
    )


class _Desk:
    """A bare panel at 11:00 whose bars, trendlines and log are in memory."""

    def __init__(self, monkeypatch, tmp_path, *, persist=False):
        QApplication.instance() or QApplication([])
        from ui.panels import alert_center_panel
        from ui.panels.alert_center_panel import AlertCenterPanel
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

        monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
        self.today = datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)
        today = self.today

        class _At11(datetime):
            @classmethod
            def now(cls, tz=None):  # noqa: D102 - stdlib signature
                return today if tz is None else today.astimezone(tz)

        from alert_center_support import patch_alert_center_global

        patch_alert_center_global(monkeypatch, "datetime", _At11)
        kwargs = {}
        if persist:
            kwargs = {
                "wall_gate_arms_path": tmp_path / "wall_gate_arms.json",
                "d1_event_watches_path": tmp_path / "d1_event_watches.json",
                "chart_watches_path": tmp_path / "alert_chart_watches.json",
            }
        self.kwargs = kwargs
        self.module = alert_center_panel
        self.panel = AlertCenterPanel(**kwargs)
        self.daily: dict[str, list] = {}
        self.intraday: dict[str, list] = {}
        self.lines: dict[str, list] = {}
        self.events: list[tuple[str, dict]] = []
        self._wire(self.panel, monkeypatch)
        # Trend-leg tests run after the optional Show-all view; so do these.
        self.panel._show_all_d1_scan_reviews = True
        self.panel._refresh_d1_scan_review_view()

    def _wire(self, panel, monkeypatch):
        monkeypatch.setattr(panel, "_d1_bars_for", lambda s: self.daily.get(s, []))
        monkeypatch.setattr(panel, "_m5_bars_for", lambda s, **kw: self.intraday.get(s, []))
        monkeypatch.setattr(panel, "_wall_trendlines_for", lambda s: self.lines.get(s, []))
        monkeypatch.setattr(
            panel,
            "_record_review_event",
            lambda action, **kw: self.events.append((action, kw)),
        )
        monkeypatch.setattr(
            panel,
            "_record_review_events",
            lambda rows: self.events.extend((row["action"], row) for row in rows),
        )

    def reopen(self, monkeypatch):
        from ui.panels.alert_center_panel import AlertCenterPanel

        self.panel = AlertCenterPanel(**self.kwargs)
        self._wire(self.panel, monkeypatch)
        self.panel._show_all_d1_scan_reviews = True
        self.panel._refresh_d1_scan_review_view()
        return self.panel

    def name(self, symbol, segments, m5=AT_WALL):
        self.daily[symbol] = _daily(segments, self.today)
        self.intraday[symbol] = _m5(m5, self.today) if m5 else []

    def charted(self):
        current = self.panel._current_review_alert
        return ([current.symbol] if current is not None else []) + [
            queued.symbol for queued in self.panel._review_queue
        ]

    def actions(self, action):
        return [kw for name, kw in self.events if name == action]


@pytest.fixture()
def desk(monkeypatch, tmp_path):
    return _Desk(monkeypatch, tmp_path)


class TestSmaWall:
    def test_a_long_right_under_its_sma200_is_hidden_and_followed_up(self, desk):
        desk.name("HHH", STEP_DOWN)
        desk.panel.add_alert(_d1_alert("HHH"))
        assert desk.charted() == []
        assert desk.panel.hidden_inside_range_count() == 1
        assert desk.panel.armed_d1_event_kinds("HHH") == {"sma_break", "ema15_reject"}
        hidden = desk.actions("wall_hidden")
        assert len(hidden) == 1
        detail = hidden[0]["detail"]
        assert detail["wall"] == "SMA100" and 0.1 < detail["distance_atr"] < 0.3
        assert detail["reason"].startswith("SMA100 wall")
        armed = desk.actions("auto_arm_watch")
        assert {row["detail"]["kind"] for row in armed} == {"sma_break", "ema15_reject"}
        assert all(row["detail"]["source_text"].startswith("auto: SMA100 wall") for row in armed)
        # A machine arm is never a trader take.
        assert desk.actions("arm_d1_event") == [] and desk.actions("arm_watch") == []

    def test_a_long_with_its_sma_as_support_shows(self, desk):
        desk.name("JJJ", STEP_UP)
        desk.panel.add_alert(_d1_alert("JJJ"))
        assert desk.charted() == ["JJJ"]
        assert desk.panel.armed_d1_event_kinds("JJJ") == set()
        assert desk.panel.chart_review.mover_badge.text() == "MOVING"

    def test_once_through_the_sma_the_chart_shows(self, desk):
        desk.name("HHH", STEP_DOWN)
        assert desk.panel._review_chart_state(_d1_alert("HHH")) == "closed"
        # The SMA break fires: price closes above the SMA100 at 21.
        desk.intraday["HHH"] = _m5([20.8, 20.9, 21.1, 21.3], desk.today)
        desk.panel._review_queue.clear()
        desk.panel.add_alert(_d1_alert("HHH"))
        assert desk.charted() == ["HHH"]

    def test_an_existing_watch_is_not_duplicated(self, desk):
        desk.name("HHH", STEP_DOWN)
        assert desk.panel.arm_d1_event_watch("HHH", "sma_break", side="LONG")
        desk.panel.add_alert(_d1_alert("HHH"))
        kinds = [w.kind for w in desk.panel._d1_event_watches if w.symbol == "HHH"]
        assert sorted(kinds) == ["ema15_reject", "sma_break"]
        assert desk.charted() == []

    def test_a_declined_watch_is_never_re_armed_and_the_chart_then_shows(self, desk):
        desk.name("HHH", STEP_DOWN)
        desk.panel.add_alert(_d1_alert("HHH"))
        desk.panel.disarm_d1_event_watch("HHH", "sma_break")
        desk.panel.disarm_d1_event_watch("HHH", "ema15_reject")
        desk.panel._hidden_inside_range.clear()
        desk.panel.add_alert(_d1_alert("HHH"))
        assert desk.panel.armed_d1_event_kinds("HHH") == set()
        assert desk.charted() == ["HHH"]
        assert desk.panel.chart_review.mover_badge.text() == "at wall"
        shown = desk.actions("wall_shown_uncovered")
        assert shown and shown[0]["detail"]["why_shown"] == "follow-up declined"

    def test_a_failed_arm_shows_the_chart(self, desk, monkeypatch, tmp_path):
        desk.panel._d1_event_watches_path = tmp_path / "d1.json"

        def _boom(*a, **k):
            raise OSError("disk full")

        from alert_center_support import patch_alert_center_global

        patch_alert_center_global(monkeypatch, "save_d1_event_watches", _boom)
        desk.name("HHH", STEP_DOWN)
        desk.panel.add_alert(_d1_alert("HHH"))
        assert desk.charted() == ["HHH"]
        assert desk.panel.armed_d1_event_kinds("HHH") == set()
        assert desk.panel.chart_review.mover_badge.text() == "at wall"

    def test_a_failed_log_write_loses_the_event_never_the_alert(self, desk, monkeypatch):
        def _boom(*a, **k):
            raise OSError("log locked")

        monkeypatch.setattr(desk.panel, "_review_events_path", Path("unused.jsonl"))
        monkeypatch.setattr(desk.module, "record_review_event", _boom)
        desk.name("HHH", STEP_DOWN)
        from ui.panels.alert_center_panel import AlertCenterPanel

        # Use the real writer (which swallows) instead of the test recorder.
        monkeypatch.setattr(
            desk.panel,
            "_record_review_event",
            AlertCenterPanel._record_review_event.__get__(desk.panel),
        )
        alert = _d1_alert("HHH")
        desk.panel.add_alert(alert)
        assert desk.panel._hidden_inside_range["HHH"] is alert
        assert desk.panel.armed_d1_event_kinds("HHH") == {"sma_break", "ema15_reject"}


class TestCap:
    def test_at_the_cap_the_next_wall_name_shows_tagged(self, desk):
        cap = desk.panel.WALL_AUTO_ARM_CAP
        assert cap == 20
        for i in range(cap):
            symbol = f"W{i:02d}"
            desk.name(symbol, STEP_DOWN)
            assert desk.panel._review_chart_state(_d1_alert(symbol)) == "closed"
        assert len(desk.panel._live_wall_symbols()) == cap
        desk.name("OVER", STEP_DOWN)
        desk.panel.add_alert(_d1_alert("OVER"))
        assert "OVER" in desk.charted()
        assert desk.panel.armed_d1_event_kinds("OVER") == set()
        assert desk.panel._wall_uncovered["OVER"] == "cap of 20 reached"
        # A name already followed up does not count twice.
        assert desk.panel._review_chart_state(_d1_alert("W00")) == "closed"

    def test_a_fired_watch_frees_its_place(self, desk):
        for i in range(desk.panel.WALL_AUTO_ARM_CAP):
            symbol = f"W{i:02d}"
            desk.name(symbol, STEP_DOWN)
            desk.panel._review_chart_state(_d1_alert(symbol))
        # Both W00 watches fired (one-shot): gone from the store.
        desk.panel._d1_event_watches = [
            w for w in desk.panel._d1_event_watches if w.symbol != "W00"
        ]
        desk.name("NEXT", STEP_DOWN)
        assert desk.panel._review_chart_state(_d1_alert("NEXT")) == "closed"


class TestTrendlineWall:
    def _line(self, desk, symbol, value, **extra):
        record = {
            "line_id": f"d1_trendline:H-:{symbol}",
            "current_line_price": value,
            "slope_log_per_bar": 0.0,
            "lookback_end": (desk.today - timedelta(days=1)).date().isoformat(),
        }
        record.update(extra)
        desk.lines[symbol] = [record]

    def test_a_name_under_its_trendline_arms_an_m15_m30_pullback(self, desk):
        desk.name("TTT", [(250, 30.0, 30.0)], m5=[30.6, 30.8, 31.0])
        self._line(desk, "TTT", 31.5)
        desk.panel.add_alert(_d1_alert("TTT"))
        assert desk.charted() == []
        watches = [w for w in desk.panel._chart_watches if w.symbol == "TTT"]
        assert len(watches) == 1
        watch = watches[0]
        assert watch.kind == "pullback" and watch.side == "LONG"
        assert watch.timeframes == ("M15", "M30")
        assert watch.source_text == "auto: trendline wall 0.5 ATR"
        assert desk.actions("wall_hidden")[0]["detail"]["wall"] == "trendline"

    def test_the_break_day_is_hidden_and_the_next_day_shows(self, desk):
        desk.name("BRK", [(250, 30.0, 30.0)], m5=[32.6, 32.8, 33.0])
        self._line(desk, "BRK", 28.0, break_date=desk.today.date().isoformat())
        assert desk.panel._review_chart_state(_d1_alert("BRK")) == "closed"
        # A break two sessions back, the line still 5 ATR below: shows.
        earlier = (desk.today - timedelta(days=2)).date().isoformat()
        self._line(desk, "BRK", 28.0, break_date=earlier)
        desk.panel.add_alert(_d1_alert("BRK"))
        assert desk.charted() == ["BRK"]

    def test_no_cached_m5_bars_means_it_shows_tagged(self, desk):
        """A pullback alert judges M15/M30 built from cached M5 bars; with none
        it could never fire, so the chart is not hidden."""
        desk.name("NOM5", [(250, 29.0, 30.0)], m5=[])
        self._line(desk, "NOM5", 30.4)
        desk.panel.add_alert(_d1_alert("NOM5"))
        assert desk.charted() == ["NOM5"]
        assert desk.panel._chart_watches == []
        assert desk.panel._wall_uncovered["NOM5"] == "no cached M5 bars"
        assert desk.panel.chart_review.mover_badge.text() == "at wall"

    def test_a_sweep_owned_pullback_is_not_cover(self, desk):
        """The claim/Focus sweep may retire its own pullback later, so it never
        covers a hidden wall name; the chart shows tagged instead."""
        from chart_watch import arm_chart_watch

        desk.name("SWP", [(250, 30.0, 30.0)], m5=[30.6, 30.8, 31.0])
        self._line(desk, "SWP", 31.5)
        swept = arm_chart_watch(
            "pullback", "SWP", "LONG", (), now=desk.today,
            source_text=desk.panel.PULLBACK_AUTO_SOURCES["claim"],
        )
        desk.panel._chart_watches = [swept]
        desk.panel.add_alert(_d1_alert("SWP"))
        assert desk.charted() == ["SWP"]
        assert desk.panel._wall_uncovered["SWP"] == "pullback owned by the claim/Focus sweep"

    def test_a_declined_wall_pullback_is_kept_declined_and_not_re_armed(self, desk):
        desk.name("TTT", [(250, 30.0, 30.0)], m5=[30.6, 30.8, 31.0])
        self._line(desk, "TTT", 31.5)
        desk.panel.add_alert(_d1_alert("TTT"))
        assert desk.panel.disarm_chart_watch_for("TTT", "pullback")
        rows = [w for w in desk.panel._chart_watches if w.symbol == "TTT"]
        assert len(rows) == 1 and rows[0].declined
        desk.panel._hidden_inside_range.clear()
        desk.panel.add_alert(_d1_alert("TTT"))
        assert desk.charted() == ["TTT"]
        assert len([w for w in desk.panel._chart_watches if w.symbol == "TTT"]) == 1


class TestLedgerPersists:
    def test_a_decline_survives_a_restart(self, monkeypatch, tmp_path):
        desk = _Desk(monkeypatch, tmp_path, persist=True)
        desk.name("HHH", STEP_DOWN)
        desk.panel.add_alert(_d1_alert("HHH"))
        payload = json.loads((tmp_path / "wall_gate_arms.json").read_text(encoding="utf-8"))
        assert set(payload["entries"]["HHH"]["kinds"]) == {"sma_break", "ema15_reject"}
        desk.panel.disarm_d1_event_watch("HHH", "sma_break")
        desk.panel.disarm_d1_event_watch("HHH", "ema15_reject")
        panel = desk.reopen(monkeypatch)
        assert panel._review_chart_state(_d1_alert("HHH")) == "unknown"
        assert panel.armed_d1_event_kinds("HHH") == set()


def test_the_kill_switch_turns_the_leg_off(desk, monkeypatch):
    import wall_gate

    monkeypatch.setattr(wall_gate, "WALL_GATE_ENABLED", False)
    desk.name("HHH", STEP_DOWN)
    desk.panel.add_alert(_d1_alert("HHH"))
    assert desk.charted() == ["HHH"]
    assert desk.panel.armed_d1_event_kinds("HHH") == set()
