"""First-30 chart hold (trader 2026-10-02).

While "Hide first 30 min" is on, a D1 scan / Focus D1 / chart-watch chart
received 09:30-10:00 ET does not reach Visual Chart Review. At 10:00 ET each
is shown only if the last completed M5 bar at/before 10:00 (the 09:55 bar)
closed past its alert level on its side; the rest are counted ("N failed by
10:00") and one click shows them. No bars / level / side = failed. Price
alerts never wait, and an M5 row the switch hides never repaints the chart.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for _path in (SCRIPTS_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_p8_p9_alert_filter import _grades_payload, _make_panel, env  # noqa: E402,F401

ET = ZoneInfo("America/New_York")
DAY = (2026, 10, 2)


def _at(hh, mm, ss=0):
    return datetime(*DAY, hh, mm, ss, tzinfo=ET)


def _bar(hh, mm, close, tz=ET):
    """A naive market-local M5 bar starting at hh:mm ET."""
    stamp = _at(hh, mm).astimezone(tz).replace(tzinfo=None)
    return {"dt": stamp, "open": close, "high": close, "low": close, "close": close}


def _focus_d1(symbol, when, *, side="LONG", level=10.0):
    from ui.models.bounce import FOCUS_D1_EVENT_TAG, BounceAlert

    return BounceAlert(
        time_text=when.strftime("%H:%M:%S"),
        symbol=symbol,
        side=side,
        trigger="Focus D1 · 15EMA reject",
        timeframe="D1",
        tag=FOCUS_D1_EVENT_TAG,
        raw_text=f"FOCUS D1 {symbol} ({side}): 15EMA reject",
        is_d1=True,
        payload={"focus_d1_kind": "ema15_reject", "alert_level": level},
        received_at=when,
    )


def _chart_watch(symbol, when, *, side="LONG", level=10.0):
    from ui.models.bounce import CHART_WATCH_TAG, BounceAlert

    return BounceAlert(
        time_text=when.strftime("%H:%M:%S"),
        symbol=symbol,
        side=side,
        trigger="Range breakout",
        timeframe="D1",
        tag=CHART_WATCH_TAG,
        raw_text=f"CHART WATCH {symbol} ({side}): Range breakout",
        payload={"chart_watch_kind": "range_breakout", "alert_level": level},
        received_at=when,
    )


def _price_alert(symbol, when):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text=when.strftime("%H:%M:%S"),
        symbol=symbol,
        side="WATCH",
        trigger="crossed 10.00",
        timeframe="",
        tag="research",
        raw_text=f"PRICE ALERT {symbol} crossed above 10.00",
        received_at=when,
    )


def _m5(symbol, when, *, side="LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text=when.strftime("%H:%M:%S"),
        symbol=symbol,
        side=side,
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text=f"[S-TIER] {symbol}: Bounce confirmed",
        payload={"feedback": {"bounce_types": "beetype"}},
        received_at=when,
    )


@pytest.fixture
def panel(env, tmp_path, monkeypatch):  # noqa: F811
    made = _make_panel(tmp_path, monkeypatch)
    made.set_setup_grades(_grades_payload())
    clock = {"now": _at(9, 36)}
    bars: dict[str, list] = {}
    monkeypatch.setattr(made, "_first30_now", lambda: clock["now"], raising=False)
    monkeypatch.setattr(made, "_first30_local_tz", lambda: ET, raising=False)
    monkeypatch.setattr(
        made, "_m5_bars_for", lambda symbol, sessions=1: list(bars.get(symbol, []))
    )
    made.test_clock = clock
    made.test_bars = bars
    yield made
    made.deleteLater()


def _review_symbols(panel) -> set[str]:
    current = panel._current_review_alert
    return {a.symbol for a in panel._review_queue} | ({current.symbol} if current else set())


def _release(panel, hh=10, mm=0, ss=1):
    panel.test_clock["now"] = _at(hh, mm, ss)
    panel._release_first30_holds()


# --------------------------------------------------------------------------- pure
def test_level_parse_from_payload_and_d1_raw_text():
    import alert_show_filter as f
    from ui.models.bounce import BounceAlert

    d1 = BounceAlert(
        time_text="09:40:00",
        symbol="HBM",
        side="LONG",
        tag="d1_flag_long",
        raw_text="MASTER_AVWAP_D1_FLAG: HBM (long) Favorite upgrade: retest AVWAPE@12.34 [price=12.50]",
        is_d1=True,
    )
    assert f.alert_level(d1) == pytest.approx(12.34)
    assert f.alert_level(_focus_d1("X", _at(9, 40), level=7.5)) == pytest.approx(7.5)
    d1.raw_text = "MASTER_AVWAP_D1_FLAG: HBM (long) 15EMA break"
    assert f.alert_level(d1) is None
    # A non-D1 row is never parsed from text.
    assert f.alert_level(_m5("X", _at(9, 40))) is None


def test_direction_prefers_payload_side_and_unknown_is_blank():
    import alert_show_filter as f

    alert = _focus_d1("X", _at(9, 40), side="LONG")
    assert f.alert_direction(alert) == "LONG"
    alert.payload["alert_side"] = "short"
    assert f.alert_direction(alert) == "SHORT"
    watch = _chart_watch("X", _at(9, 40), side="WATCH")
    assert f.alert_direction(watch) == ""


def test_check_bar_is_the_last_completed_bar_at_or_before_10():
    import alert_show_filter as f

    release = f.first30_release_at(_at(9, 40))
    assert release == _at(10, 0)
    bars = [_bar(9, 25, 1.0), _bar(9, 50, 2.0), _bar(9, 55, 3.0), _bar(10, 0, 4.0)]
    bar = f.first30_check_bar(bars, release, ET)
    assert bar["close"] == 3.0
    early = f.first30_check_bar(bars[:2], release, ET)
    assert early["close"] == 2.0
    # Only pre-open bars, or yesterday's: nothing to check.
    assert f.first30_check_bar([_bar(9, 25, 1.0)], release, ET) is None
    yesterday = {"dt": datetime(2026, 10, 1, 9, 55), "close": 9.0}
    assert f.first30_check_bar([yesterday], release, ET) is None


def test_check_bar_reads_bars_in_the_desk_zone():
    import alert_show_filter as f

    pacific = ZoneInfo("America/Los_Angeles")
    release = f.first30_release_at(_at(9, 40))
    bars = [_bar(9, 55, 3.0, tz=pacific), _bar(10, 0, 3.1, tz=pacific)]
    assert f.first30_check_bar(bars, release, pacific)["close"] == 3.0
    assert f.first30_bars_fresh(bars, release, pacific) is True


def test_only_a_bar_from_10_00_on_proves_the_0955_bar_finished():
    import alert_show_filter as f

    release = f.first30_release_at(_at(9, 40))
    assert f.first30_bars_fresh([_bar(9, 50, 1.0), _bar(9, 55, 2.0)], release, ET) is False
    assert f.first30_bars_fresh([_bar(9, 55, 2.0), _bar(10, 0, 2.1)], release, ET) is True
    assert f.first30_bars_fresh([], release, ET) is False
    tomorrow = {"dt": datetime(2026, 10, 3, 9, 30), "close": 1.0}
    assert f.first30_bars_fresh([tomorrow], release, ET) is False


def test_verdicts():
    import alert_show_filter as f

    bar = {"close": 10.5}
    assert f.first30_verdict("LONG", 10.0, bar) == f.HOLD_HELD
    assert f.first30_verdict("LONG", 10.5, bar) == f.HOLD_FAILED
    assert f.first30_verdict("SHORT", 11.0, bar) == f.HOLD_HELD
    assert f.first30_verdict("SHORT", 10.0, bar) == f.HOLD_FAILED
    assert f.first30_verdict("LONG", 10.0, None) == f.HOLD_NO_DATA
    assert f.first30_verdict("LONG", None, bar) == f.HOLD_NO_LEVEL
    assert f.first30_verdict("WATCH", 10.0, bar) == f.HOLD_NO_SIDE


# --------------------------------------------------------------------------- panel
def test_a_first30_focus_d1_chart_waits_and_shows_at_10_when_it_held(panel):
    panel.add_alert(_focus_d1("HBM", _at(9, 36), level=10.0))
    assert "HBM" not in _review_symbols(panel)
    assert panel._first30_timer.isActive()
    assert "1 wait for 10:00" in panel.chart_review.first30_button.text()
    panel.test_bars["HBM"] = [_bar(9, 50, 9.0), _bar(9, 55, 10.4), _bar(10, 0, 9.0)]
    _release(panel)
    assert "HBM" in _review_symbols(panel)
    assert not panel._first30_held and not panel._first30_failed
    assert not panel.chart_review.first30_button.isVisibleTo(panel.chart_review)


def test_a_chart_watch_that_failed_is_counted_and_one_click_shows_it(panel):
    panel.add_alert(_chart_watch("GTLB", _at(9, 38), level=50.0))
    assert "GTLB" not in _review_symbols(panel)
    panel.test_bars["GTLB"] = [_bar(9, 55, 49.0), _bar(10, 0, 49.0)]
    _release(panel)
    assert "GTLB" not in _review_symbols(panel)
    assert panel.chart_review.first30_button.text() == "1 failed by 10:00 - show"
    panel.chart_review.first30_button.click()
    assert "GTLB" in _review_symbols(panel)
    assert not panel._first30_failed


def test_a_short_holds_below_its_level(panel):
    panel.add_alert(_focus_d1("SHO", _at(9, 45), side="SHORT", level=20.0))
    panel.test_bars["SHO"] = [_bar(9, 55, 19.5), _bar(10, 0, 19.5)]
    _release(panel)
    assert "SHO" in _review_symbols(panel)


def test_no_data_no_level_or_no_side_is_hidden_after_the_grace(panel):
    panel.add_alert(_focus_d1("NOD", _at(9, 40)))
    panel.add_alert(_chart_watch("NOL", _at(9, 41), level=None))
    panel.add_alert(_chart_watch("NOS", _at(9, 42), side="WATCH"))
    panel.test_bars["NOL"] = [_bar(9, 55, 99.0), _bar(10, 0, 99.0)]
    panel.test_bars["NOS"] = [_bar(9, 55, 99.0), _bar(10, 0, 99.0)]
    # Just after 10:00 the 09:55 bar may not be cached yet: NOD keeps waiting.
    _release(panel)
    assert set(panel._first30_held) == {("NOD", "LONG")}
    assert panel._first30_timer.isActive()
    _release(panel, 10, 4)
    assert not panel._first30_held
    assert {key[0] for key in panel._first30_failed} == {"NOD", "NOL", "NOS"}
    assert not ({"NOD", "NOL", "NOS"} & _review_symbols(panel))
    assert "no 5-minute bar" in panel.chart_review.first30_button.toolTip()


def test_price_alerts_never_wait(panel):
    panel.add_alert(_price_alert("PRC", _at(9, 40)))
    assert "PRC" in _review_symbols(panel)
    assert not panel._first30_held


def test_after_10_and_switch_off_nothing_waits(panel):
    panel.test_clock["now"] = _at(10, 1)
    panel.add_alert(_focus_d1("LAT", _at(10, 1)))
    assert "LAT" in _review_symbols(panel)
    panel.test_clock["now"] = _at(9, 40)
    panel.first30_input.setChecked(False)
    panel.add_alert(_focus_d1("OFF", _at(9, 40)))
    assert "OFF" in _review_symbols(panel)
    assert not panel._first30_held


def test_turning_the_switch_off_releases_every_waiting_chart(panel):
    panel.add_alert(_focus_d1("ONE", _at(9, 40)))
    panel.add_alert(_chart_watch("TWO", _at(9, 41)))
    assert not ({"ONE", "TWO"} & _review_symbols(panel))
    panel.first30_input.setChecked(False)
    assert {"ONE", "TWO"} <= _review_symbols(panel)
    assert not panel._first30_held
    assert not panel._first30_timer.isActive()


def test_same_symbol_twice_keeps_the_latest(panel):
    panel.add_alert(_focus_d1("DUP", _at(9, 40), level=10.0))
    panel.add_alert(_focus_d1("DUP", _at(9, 50), level=12.0))
    assert list(panel._first30_held) == [("DUP", "LONG")]
    panel.test_bars["DUP"] = [_bar(9, 55, 11.0), _bar(10, 0, 11.0)]
    _release(panel)
    # Judged against the latest level (12.0): 11.0 did not hold.
    assert ("DUP", "LONG") in panel._first30_failed


def test_a_new_day_clears_the_holds(panel):
    panel.add_alert(_focus_d1("OLD", _at(9, 40)))
    panel._first30_failed[("GONE", "LONG")] = (_focus_d1("GONE", _at(9, 41)), "failed")
    panel._ignored_market_date = "2000-01-01"
    panel._refresh_ignored_market_date()
    assert not panel._first30_held and not panel._first30_failed
    assert not panel._first30_timer.isActive()


def test_a_held_chart_does_not_repaint_the_current_review(panel):
    panel.test_clock["now"] = _at(9, 20)
    first = _focus_d1("HBM", _at(9, 20))
    panel.add_alert(first)
    assert panel._current_review_alert is first
    panel.test_clock["now"] = _at(9, 40)
    panel.add_alert(_chart_watch("HBM", _at(9, 40)))
    assert panel._current_review_alert is first


def test_a_first30_hidden_m5_row_does_not_replace_the_charted_review(panel):
    import alert_show_filter

    panel.show_filter_input.setCurrentIndex(
        panel.show_filter_input.findData(alert_show_filter.ALL)
    )
    panel.test_clock["now"] = _at(9, 20)
    first = _focus_d1("BEE", _at(9, 20))
    panel.add_alert(first)
    assert panel._current_review_alert is first
    m5 = _m5("BEE", _at(9, 40))
    assert panel.show_filter_reason(m5) == alert_show_filter.REASON_FIRST30
    panel.add_alert(m5)
    assert panel._current_review_alert is first
    # The M5 bar still gets the row.
    assert m5 in panel.test_posted


def test_chart_watch_alert_carries_level_and_time(panel):
    from chart_watch import ChartWatchTrigger, D1LevelWatch

    watch = D1LevelWatch(symbol="LVL", direction="above", level=12.5, armed_at=datetime(*DAY, 8))
    hit = ChartWatchTrigger(
        watch=watch, price=12.9, bar_dt=datetime(*DAY, 9, 40), message="D1 level break",
        resolved_side="long", level=12.5,
    )
    moment = datetime(*DAY, 9, 45)
    alert = panel._chart_watch_alert(hit, moment)
    assert alert.payload["alert_level"] == 12.5
    assert alert.side == "LONG"
    assert alert.received_at is not None and alert.received_at.utcoffset() is not None


def test_a_hit_without_a_reference_level_is_no_level_never_the_trigger_price(panel):
    from chart_watch import ChartWatch, ChartWatchTrigger

    watch = ChartWatch(symbol="RB", kind="range_breakout", armed_at=datetime(*DAY, 8), side="LONG")
    hit = ChartWatchTrigger(
        watch=watch, price=33.3, bar_dt=datetime(*DAY, 9, 40), message="Range breakout"
    )
    alert = panel._chart_watch_alert(hit, datetime(*DAY, 9, 45))
    assert alert.payload["alert_level"] is None
    import alert_show_filter

    assert alert_show_filter.alert_level(alert) is None


def _d1_fixture():
    daily = []
    day = datetime(2026, 8, 20)
    while len(daily) < 30:
        day += timedelta(days=1)
        if day.weekday() < 5:
            daily.append({"dt": day, "open": 97.0, "high": 100.0, "low": 95.0, "close": 98.0})
    m5 = [
        {"dt": datetime(*DAY, 9, 30), "open": 99.0, "high": 99.5, "low": 98.5, "close": 99.2},
        {"dt": datetime(*DAY, 9, 35), "open": 99.2, "high": 103.0, "low": 99.0, "close": 102.0},
    ]
    return daily, m5


def test_a_new_20d_high_is_judged_against_the_prior_high_not_the_bar_high(panel):
    """Reviewer's case: "New 20-day high: 103.00 > 100.00", 09:55 close 101.5 holds."""
    from chart_watch import D1EventWatch, evaluate_d1_event_watch

    daily, m5 = _d1_fixture()
    watch = D1EventWatch(symbol="NEW", kind="new_20d_high", armed_at=datetime(*DAY, 8), side="LONG")
    hit = evaluate_d1_event_watch(watch, m5, daily, now=datetime(*DAY, 9, 41))
    assert hit is not None and "103.00 > 100.00" in hit.message
    alert = panel._chart_watch_alert(hit, datetime(*DAY, 9, 41))
    alert.received_at = _at(9, 41)
    assert alert.payload["alert_level"] == 100.0
    panel.test_clock["now"] = _at(9, 41)
    panel.add_alert(alert)
    assert "NEW" not in _review_symbols(panel)
    panel.test_bars["NEW"] = [_bar(9, 55, 101.5), _bar(10, 0, 101.6)]
    _release(panel)
    assert "NEW" in _review_symbols(panel)


def test_focus_d1_flag_alert_carries_the_reference_level(panel, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ui.panels import alert_center_panel as panel_mod

    captured = []
    monkeypatch.setattr(panel, "add_alert", captured.append)
    panel._focus_d1_flags_path = tmp_path / "flags.json"
    panel.focus_service = SimpleNamespace(all_focus=lambda: {"long": ["FOC"], "short": []})
    monkeypatch.setattr(panel, "_m5_unknown", lambda symbol, sessions=1: False)
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: [{"dt": datetime(*DAY)}])
    monkeypatch.setattr(
        panel, "_update_focus_break_state", lambda *a, **k: datetime(*DAY, 9, 30)
    )
    monkeypatch.setattr(panel, "_note_focus_activity", lambda *a, **k: None)
    hit = SimpleNamespace(price=21.9, level=21.5, resolved_side="long", message="15EMA reject")
    monkeypatch.setattr(panel_mod, "evaluate_d1_event_watch", lambda *a, **k: hit)
    panel._poll_focus_d1_interest(now=datetime(*DAY, 9, 40))
    assert captured
    alert = captured[0]
    assert alert.payload["alert_level"] == 21.5
    assert alert.payload["alert_side"] == "LONG"
    assert alert.received_at is not None and alert.received_at.utcoffset() is not None


def test_a_0955_bar_without_a_later_bar_waits_then_counts_as_no_data(panel):
    """Advisory 2: a 09:55 bar fetched mid-print is not proof; wait for a 10:00 bar."""
    panel.add_alert(_focus_d1("PAR", _at(9, 40), level=10.0))
    panel.add_alert(_focus_d1("LAT", _at(9, 41), level=10.0))
    panel.test_bars["PAR"] = [_bar(9, 55, 10.5)]
    panel.test_bars["LAT"] = [_bar(9, 55, 10.5)]
    _release(panel)
    assert set(panel._first30_held) == {("PAR", "LONG"), ("LAT", "LONG")}
    # LAT's cache refreshes after 10:00 within the grace: judged on its 09:55 bar.
    panel.test_bars["LAT"] = [_bar(9, 55, 10.5), _bar(10, 0, 9.0)]
    _release(panel, 10, 1)
    assert "LAT" in _review_symbols(panel)
    _release(panel, 10, 3, 30)
    assert "PAR" not in _review_symbols(panel)
    assert panel._first30_failed[("PAR", "LONG")][1] == "no_data"


def _scan(symbol, when, *, level):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text=when.strftime("%H:%M:%S"),
        symbol=symbol,
        side="LONG",
        trigger="D1 flag",
        timeframe="D1",
        tag="d1_flag_long",
        raw_text=f"MASTER_AVWAP_D1_FLAG: {symbol} (long) Favorite upgrade: AVWAPE@{level:.2f}",
        is_d1=True,
        received_at=when,
    )


def test_a_waiting_scan_shown_by_show_all_before_10_joins_the_hold(panel):
    panel.add_alert(_scan("SCN", _at(9, 40), level=20.0))
    assert "SCN" in panel._held_d1_scan_reviews
    panel._toggle_d1_scan_review_view()
    assert "SCN" not in _review_symbols(panel)
    assert ("SCN", "LONG") in panel._first30_held


def test_a_waiting_scan_shown_by_show_all_after_10_faces_the_verdict(panel):
    panel.add_alert(_scan("FAL", _at(9, 40), level=20.0))
    panel.add_alert(_scan("HLD", _at(9, 45), level=20.0))
    assert not panel._first30_held
    panel.test_bars["FAL"] = [_bar(9, 55, 19.0), _bar(10, 0, 19.0)]
    panel.test_bars["HLD"] = [_bar(9, 55, 21.0), _bar(10, 0, 21.0)]
    panel.test_clock["now"] = _at(10, 20)
    panel._toggle_d1_scan_review_view()
    assert not ({"FAL", "HLD"} & _review_symbols(panel))
    panel._release_first30_holds()
    assert "HLD" in _review_symbols(panel)
    assert "FAL" not in _review_symbols(panel)
    assert ("FAL", "LONG") in panel._first30_failed
