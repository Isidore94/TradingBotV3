"""WS-10C, the builder's own tests: the seams the packet's file does not reach.

Added, never a replacement - `tests/test_ws_10c_h1_retester.py` is the packet's
contract and nothing here weakens it. What is proved here:

* the arm-bar button emits the kind once a symbol is charted (the packet's own
  version of this clicks a bar with no symbol, where `ArmBar` disables every
  watch toggle by design - `set_enabled_for_symbol` - so that assertion is
  left red and answered here instead);
* the 30 s M5 chart-watch poll no longer RETIRES an H1 retester on the date
  roll. The packet pins the store's load path; without the same exemption in
  `watch_is_stale` the watch would survive the restart and then be deleted by
  the next poll, which is the same bug with a longer fuse;
* an invalidation disarms and is NOT a phone event;
* too little history is a WAIT, not a retire, and not a fire;
* the fired row carries every measured reason.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_ws_10c_h1_retester import (  # noqa: E402
    ARM_REASON_LONG,
    BUTTON_LABEL,
    GOLDEN_ATR,
    GOLDEN_CONFIRM_DT,
    WATCH_KIND,
    _bucket_dt,
    _close_for_ema_offset,
    _mirror,
    golden_long_h1_bars,
    golden_m5_series,
)


def _qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - headless without Qt
        pytest.skip("PySide6 not installed")
    return QApplication.instance() or QApplication([])


def _panel(monkeypatch, tmp_path):
    _qt_app()
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    panel = AlertCenterPanel()
    panel._chart_watches_path = tmp_path / "chart_watches.json"
    return panel


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def notify_armed_watch(self, *, watch_id, title, message):
        self.calls.append({"watch_id": watch_id, "title": title, "message": message})
        return {"ok": True}


def test_the_arm_bar_button_emits_the_kind_once_a_symbol_is_charted():
    _qt_app()
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    # Every watch toggle is disabled until a symbol is charted - there is
    # nothing to arm AT otherwise. This is how the shipped arm-bar tests drive
    # it (tests/test_qt_arm_dock.py).
    bar.set_enabled_for_symbol(True)
    button = bar.watch_buttons[WATCH_KIND]
    assert BUTTON_LABEL in button.text()

    emitted: list[str] = []
    bar.watchToggled.connect(emitted.append)
    button.click()

    assert emitted == [WATCH_KIND]


def test_the_h1_button_says_it_is_not_a_session_watch():
    _qt_app()
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    tooltip = bar.watch_buttons[WATCH_KIND].toolTip()
    assert "ten trading days" in tooltip
    assert "completed H1 bar" in tooltip


def test_the_m5_poll_retires_the_session_watch_and_keeps_the_h1_one(
    monkeypatch, tmp_path
):
    """The date roll, from the other side.

    `load_chart_watches` keeps the H1 row across the roll; the 30 s poll would
    then delete it within a minute unless `watch_is_stale` says the same thing.
    """
    from chart_watch import ChartWatch

    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: [])
    yesterday = datetime(2026, 8, 26, 7, 15)
    panel._chart_watches = [
        ChartWatch(
            symbol="AAPL",
            kind=WATCH_KIND,
            armed_at=yesterday,
            side="LONG",
            reason=ARM_REASON_LONG,
            watch_id="ws10c-roll",
        ),
        ChartWatch(symbol="AAPL", kind="new_hod", armed_at=yesterday, side="LONG"),
    ]

    panel._poll_chart_watches(now=datetime(2026, 8, 27, 7, 15))

    kinds = {watch.kind for watch in panel._chart_watches}
    assert kinds == {WATCH_KIND}


def test_an_invalidated_retest_disarms_and_never_reaches_the_phone(
    monkeypatch, tmp_path
):
    from indicators.h1_ema_bounce import closed_h1_bars

    bars, ema = golden_long_h1_bars()
    broken = list(bars[:54])
    close = _close_for_ema_offset(ema[53], -1.5)
    broken.append(
        {
            "dt": _bucket_dt(54),
            "open": close + 0.30,
            "high": close + 0.40,
            "low": close - 0.10,
            "close": close,
        }
    )
    m5 = golden_m5_series(broken)
    assert len(closed_h1_bars(m5)) == len(broken)

    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: list(m5))
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol, **kw: [])
    recorder = _Recorder()
    panel.price_alert_service = recorder
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    before = len(panel._alerts)

    panel._poll_d1_event_watches(now=_bucket_dt(54) + timedelta(hours=1))

    assert not [w for w in panel._chart_watches if w.kind == WATCH_KIND]
    assert len(panel._alerts) == before  # nothing to act on: no alert
    assert recorder.calls == []  # and nothing to wake the trader for


def test_too_little_history_waits_and_is_neither_fired_nor_retired(
    monkeypatch, tmp_path
):
    """Two sessions of cached M5 bars cannot answer a 45-bar warm-up. Missing
    data is uncertainty (plan.md sec 5): the watch stays armed."""
    bars, _ = golden_long_h1_bars()
    m5 = golden_m5_series(bars[:14])  # two sessions

    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: list(m5))
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol, **kw: [])
    recorder = _Recorder()
    panel.price_alert_service = recorder
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    before = len(panel._alerts)

    panel._poll_d1_event_watches(now=_bucket_dt(13) + timedelta(hours=1))

    assert [w.kind for w in panel._chart_watches] == [WATCH_KIND]
    assert len(panel._alerts) == before
    assert recorder.calls == []


def test_the_fired_row_records_every_measured_reason(monkeypatch, tmp_path):
    panel = _panel(monkeypatch, tmp_path)
    bars, _ = golden_long_h1_bars()
    monkeypatch.setattr(
        panel, "_m5_bars_for", lambda symbol, **kw: golden_m5_series(bars)
    )
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol, **kw: [])
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)

    panel._poll_d1_event_watches(now=GOLDEN_CONFIRM_DT + timedelta(hours=1))

    payload = dict(panel._alerts[0].payload or {})
    reasons = list(payload.get("reasons") or ())
    assert len(reasons) >= 3
    assert any("tagged the 15-EMA" in reason for reason in reasons)
    assert any("back through the line" in reason for reason in reasons)
    assert payload["rule_version"] == "h1_ema_bounce_v1"
    assert payload["reason"] == ARM_REASON_LONG
    # The alert names the setup in words the trader armed it with.
    assert "H1 15-EMA retest confirmed" in panel._alerts[0].trigger


def test_a_persistent_watch_is_not_stale_the_next_day_in_the_armed_inventory():
    from ui.widgets.armed_watch_list import HEALTH_OK, HEALTH_STALE, watch_health

    armed_at = datetime(2026, 8, 26, 7, 15)
    tomorrow = datetime(2026, 8, 27, 7, 15)
    assert watch_health(WATCH_KIND, True, armed_at, tomorrow) == HEALTH_OK
    assert watch_health("new_hod", True, armed_at, tomorrow) == HEALTH_STALE


def test_the_short_side_arms_with_its_own_reason():
    from chart_watch import arm_chart_watch

    watch = arm_chart_watch(
        WATCH_KIND, "msft", "SHORT", [], now=datetime(2026, 8, 26, 7, 15)
    )
    assert watch.reason == "waiting for an H1 15-EMA bounce (SHORT)"
    assert watch.watch_id
    # A chart with no side of its own still arms, and says so.
    either = arm_chart_watch(
        WATCH_KIND, "msft", "WATCH", [], now=datetime(2026, 8, 26, 7, 15)
    )
    assert either.reason == "waiting for an H1 15-EMA bounce (EITHER SIDE)"
    assert either.watch_id != watch.watch_id


def test_a_side_less_watch_is_evaluated_BOTH_ways():
    """A chart with no side of its own still gets an answer: the rule is run
    long and short and the first confirmation wins. Without that, arming from
    a WATCH-side chart would be a button that can never fire."""
    from chart_watch import ChartWatch, evaluate_h1_bounce_watch

    bars, _ = golden_long_h1_bars()
    watch = ChartWatch(
        symbol="AAPL",
        kind=WATCH_KIND,
        armed_at=datetime(2026, 8, 26, 7, 15),
        side="WATCH",
        watch_id="ws10c-either",
    )

    long_hit = evaluate_h1_bounce_watch(watch, golden_m5_series(bars))
    assert long_hit is not None and long_hit.fired is True
    assert long_hit.side == "long"

    short_hit = evaluate_h1_bounce_watch(watch, golden_m5_series(_mirror(bars)))
    assert short_hit is not None and short_hit.fired is True
    assert short_hit.side == "short"
    assert short_hit.touch_bar_dt == long_hit.touch_bar_dt


def test_the_rule_is_not_measured_rather_than_false_without_an_atr():
    from indicators.h1_ema_bounce import evaluate

    bars, _ = golden_long_h1_bars()
    assert evaluate(bars, "long", atr=None) is None
    assert evaluate(bars, "long", atr=0.0) is None
    assert evaluate(bars, "sideways", atr=GOLDEN_ATR) is None


def test_the_expiry_row_names_the_store_and_the_ten_trading_days():
    import armed_alert_expiry
    from chart_watch import ChartWatch

    watch = ChartWatch(
        symbol="AAPL",
        kind=WATCH_KIND,
        armed_at=datetime(2026, 8, 17, 7, 15),
        side="LONG",
        watch_id="ws10c-expiry",
    )
    kept, rows = armed_alert_expiry.partition(
        [watch], store="chart_watches", today=date(2026, 8, 31)
    )
    assert kept == []
    assert len(rows) == 1
    assert rows[0]["store"] == "chart_watches"
    assert rows[0]["kind"] == WATCH_KIND
    assert rows[0]["trading_days"] == armed_alert_expiry.DEFAULT_EXPIRY_TRADING_DAYS


def test_the_armed_inventory_says_not_measured_with_the_bar_count(
    monkeypatch, tmp_path
):
    """Lead ruling 2026-09-13: if the desk's cached M5 window cannot reach the
    45-bar warm-up, the watch stays armed and the REASON is visible - never a
    row reporting `ok` beside a watch that cannot evaluate at all."""
    from indicators.h1_ema_bounce import WARMUP_BARS

    bars, _ = golden_long_h1_bars()
    thin = golden_m5_series(bars[:14])  # two sessions -> 14 completed H1 bars

    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: list(thin))
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    watch = next(w for w in panel._chart_watches if w.kind == WATCH_KIND)

    assert panel._armed_watch_note(watch) == f"not measured (14 of {WARMUP_BARS} H1 bars)"

    # It reaches the table the trader reads, in the health column.
    panel._refresh_armed_list()
    row = next(row for row in panel.armed_list._rows if row[0] == "AAPL")
    assert row[5] == f"not measured (14 of {WARMUP_BARS} H1 bars)"

    # With enough history the note is silent and the ordinary health shows.
    monkeypatch.setattr(
        panel, "_m5_bars_for", lambda symbol, **kw: golden_m5_series(bars)
    )
    assert panel._armed_watch_note(watch) == ""


def test_a_session_watch_never_gets_the_h1_note(monkeypatch, tmp_path):
    from chart_watch import ChartWatch

    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(panel, "_m5_bars_for", lambda symbol, **kw: [])
    session_watch = ChartWatch(
        symbol="AAPL",
        kind="new_hod",
        armed_at=datetime(2026, 8, 26, 7, 15),
        side="LONG",
    )
    assert panel._armed_watch_note(session_watch) == ""
