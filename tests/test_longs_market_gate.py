"""Longs off in a bad market (the trader 2026-09-26, "Ok"). Presentation only.

"Only take longs when the market is with you ... the scan should show longs only in a
good market (your regime or SPY's 20-day), and say 'longs off' the rest of the time."

A switch, default on, hides LONG M5 rows, LONG swing rows and LONG phone picks while
`setup_permutations.long_regime_working` says no. Focus, typed names, chart-watch hits,
open positions and the leader_pullback / post_earnings_drift rows always show. Unknown
shows. Every hidden alert is still recorded with `hidden_by_show` reason `longs_off`.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for _path in (SCRIPTS_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_p8_p9_alert_filter import _grades_payload, _m5, _make_panel, env  # noqa: E402,F401

TODAY = date(2026, 9, 28)  # a Monday; the reference session is Friday 2026-09-25


def _sessions(end: date, count: int) -> list[date]:
    import market_calendar

    days = [end]
    while len(days) < count:
        days.append(market_calendar.previous_session(days[-1]))
    return days[::-1]


def _spy(values_by_offset) -> dict[str, float]:
    days = _sessions(date(2026, 9, 25), len(values_by_offset))
    return {day.isoformat(): float(value) for day, value in zip(days, values_by_offset, strict=True)}


RISING = [100.0 + i for i in range(40)]  # above a rising 20-day
# 30 rising days, then 10 days well under the 20-day.
FALLING = [100.0 + i for i in range(30)] + [110.0 - i for i in range(10)]


def _off(reason="SPY is under its 20-day", since="2026-09-14", opens=()):
    import longs_market_gate as g

    return g.Verdict(day=date.today().isoformat(), verdict="no", rule="spy_above_rising_sma20",
                     reason=reason, since=since, open_symbols=frozenset(opens))


# --------------------------------------------------------------------------- pure
def test_spy_above_a_rising_20_day_keeps_longs_on():
    import longs_market_gate as g

    verdict = g.compute(today=TODAY, regime_rows=[], spy_closes=_spy(RISING))
    assert (verdict.verdict, verdict.longs_off) == ("yes", False)
    assert g.banner_text(verdict) == ""


def test_spy_under_its_20_day_turns_longs_off_with_since():
    import longs_market_gate as g

    closes = _spy(FALLING)
    verdict = g.compute(today=TODAY, regime_rows=[], spy_closes=closes)
    assert verdict.longs_off is True
    assert verdict.rule == "spy_above_rising_sma20"
    assert verdict.reason == "SPY is under its 20-day"
    days = sorted(closes)
    # The run of "no" days starts on or after the first falling close.
    assert days[30] <= verdict.since <= days[-1]
    assert g.banner_text(verdict) == f"Longs off: SPY is under its 20-day (since {verdict.since})"


def test_the_trader_regime_decides_first():
    import longs_market_gate as g

    rows = [
        {"segment_id": 1, "regime": "bull_run", "start_date": "2026-08-01"},
        {"segment_id": 2, "regime": "range", "start_date": "2026-09-08"},
        {"segment_id": 3, "regime": "capitulation", "start_date": "2026-09-21"},
    ]
    verdict = g.compute(today=TODAY, regime_rows=rows, spy_closes=_spy(RISING))
    assert (verdict.verdict, verdict.rule) == ("no", "trader")
    assert verdict.since == "2026-09-08"
    assert g.banner_text(verdict) == "Longs off: your regime is capitulation (since 2026-09-08)"
    good = g.compute(today=TODAY, regime_rows=rows[:1], spy_closes=_spy(FALLING))
    assert (good.verdict, good.longs_off) == ("yes", False)


def test_unknown_market_never_hides():
    import longs_market_gate as g

    stale = {k: v for k, v in _spy(FALLING).items() if k < "2026-09-25"}
    for closes in ({}, _spy(FALLING[-10:]), stale):
        verdict = g.compute(today=TODAY, regime_rows=[], spy_closes=closes)
        assert verdict.verdict == "unknown" and verdict.longs_off is False
        assert g.hides_long(verdict, "LONG", "ABC") is False
    assert g.hides_long(None, "LONG", "ABC") is False


def test_today_partial_bar_is_not_used():
    import longs_market_gate as g

    closes = _spy(RISING)
    closes[TODAY.isoformat()] = 1.0  # an in-progress bar far under the 20-day
    assert g.compute(today=TODAY, regime_rows=[], spy_closes=closes).verdict == "yes"


def test_open_positions_and_shorts_never_hide():
    import longs_market_gate as g

    verdict = _off(opens=("HELD",))
    assert g.hides_long(verdict, "LONG", "ABC") is True
    assert g.hides_long(verdict, "LONG", "held") is False
    assert g.hides_long(verdict, "SHORT", "ABC") is False
    assert g.hides_long(verdict, "LONG", "ABC", exempt=True) is False


def test_leader_pullback_and_post_earnings_drift_rows_are_exempt():
    import longs_market_gate as g

    assert g.row_is_exempt({"setup_family": "leader_pullback"}) is True
    assert g.row_is_exempt({"setup_family": "post_earnings_drift"}) is True
    assert g.row_is_exempt({"leader_pullback": "leader_pullback_long"}) is True
    assert g.row_is_exempt({"setup_family": "avwap_band_bounce"}) is False
    assert g.row_is_exempt(None) is False


def test_the_switch_defaults_on_and_persists(env):  # noqa: F811
    import longs_market_gate as g

    assert g.enabled() is True
    g.set_enabled(False)
    assert env[g.SETTING] is False and g.enabled() is False


def test_the_worker_cache_never_reads_on_snapshot(monkeypatch):
    import longs_market_gate as g

    g.clear_cache()
    calls = []
    monkeypatch.setattr(g, "load", lambda today=None: calls.append(1) or _off())
    assert g.snapshot() is None and calls == []
    assert g.current().longs_off is True and calls == [1]
    assert g.snapshot() is not None
    g.current()
    assert calls == [1]  # cached for the day
    g.clear_cache()


# --------------------------------------------------------------------------- Alert Center
@pytest.fixture
def panel(env, tmp_path, monkeypatch):  # noqa: F811
    import alert_show_filter
    import longs_market_gate

    longs_market_gate.clear_cache()
    made = _make_panel(tmp_path, monkeypatch)
    made.set_setup_grades(_grades_payload())
    # Show: All and first-30 off, so only longs off can hide.
    made.show_filter_input.setCurrentIndex(made.show_filter_input.findData(alert_show_filter.ALL))
    made.first30_input.setChecked(False)
    yield made
    made.deleteLater()
    longs_market_gate.clear_cache()


def test_the_switch_is_on_by_default_in_the_alert_center(panel):
    assert panel.longs_off_input.isChecked() is True
    assert panel.longs_off_input in panel._control_widgets


def test_bad_market_hides_longs_with_a_banner(panel):
    alert = _m5("BEE", "provtype")  # even a top grade hides
    short = _m5("BEE", "provtype", side="SHORT")
    assert panel.show_filter_hides(alert) is False
    panel.set_longs_gate(_off())
    assert panel.show_filter_hides(alert) is True
    assert panel.show_filter_reason(alert) == "longs_off"
    assert panel.show_filter_hides(short) is False
    assert panel.longs_off_banner_text() == "Longs off: SPY is under its 20-day (since 2026-09-14)"
    assert panel.longs_off_banner.text() == panel.longs_off_banner_text()


def test_unknown_market_shows_longs(panel):
    import longs_market_gate as g

    panel.set_longs_gate(g.Verdict(day=date.today().isoformat()))
    assert panel.show_filter_hides(_m5("BEE", "beetype")) is False
    assert panel.longs_off_banner_text() == ""


def test_switch_off_shows_all_longs(panel, env):  # noqa: F811
    import longs_market_gate as g

    panel.set_longs_gate(_off())
    alert = _m5("BEE", "beetype")
    assert panel.show_filter_hides(alert) is True
    panel.longs_off_input.setChecked(False)
    assert env[g.SETTING] is False
    assert panel.show_filter_hides(alert) is False
    assert panel.longs_off_banner_text() == ""


def test_focus_typed_watch_and_held_names_still_show(panel):
    from ui.models.bounce import CHART_WATCH_TAG

    panel.set_longs_gate(_off(opens=("HELD",)))
    watch = _m5("WAT", "beetype")
    watch.tag = CHART_WATCH_TAG
    for alert in (_m5("FOC", "deetype"), _m5("TYPED", "ceetype"), watch, _m5("HELD", "beetype")):
        assert panel.show_filter_hides(alert) is False, alert.symbol


def test_hidden_longs_are_still_recorded_with_the_reason(panel, tmp_path):
    import review_events

    panel.set_longs_gate(_off())
    for symbol in ("BEE", "FOC"):
        panel.add_alert(_m5(symbol, "beetype"))
    panel.add_alert(_m5("SHO", "beetype", side="SHORT"))
    assert {a.symbol for a in panel._alerts} == {"BEE", "FOC", "SHO"}
    assert {key[0] for key in panel._feed_row_registry()} == {"FOC", "SHO"}
    panel.flush_show_hidden_writes()
    rows = [
        row
        for row in review_events.load_review_events(tmp_path / "alert_review_events.jsonl")
        if row.get("action") == "hidden_by_show"
    ]
    assert [(row["symbol"], row["detail"]["reason"]) for row in rows] == [("BEE", "longs_off")]


# --------------------------------------------------------------------------- swing table
def _swing_rows():
    from ui.models.setup import SetupRow

    return [
        SetupRow(symbol="AAA", side="LONG", score=90.0),
        SetupRow(symbol="BBB", side="SHORT", score=80.0),
        SetupRow(symbol="FOC", side="LONG", score=70.0),  # a Focus name
        SetupRow(symbol="TYPED", side="LONG", score=60.0),  # in longs.txt
        SetupRow(symbol="HELD", side="LONG", score=50.0),  # an open position
        SetupRow(symbol="LPB", side="LONG", score=40.0, raw={"setup_family": "leader_pullback"}),
        SetupRow(symbol="PED", side="LONG", score=30.0, raw={"setup_family": "post_earnings_drift"}),
    ]


def test_the_swing_proxy_hides_only_unexempt_longs():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    import longs_market_gate as g
    from ui.models.setup_table_model import SetupFilterProxyModel, SetupTableModel

    model = SetupTableModel()
    model.set_rows(_swing_rows())
    proxy = SetupFilterProxyModel()
    proxy.setSourceModel(model)

    def visible():
        return [model.row_at(proxy.mapToSource(proxy.index(r, 0)).row()).symbol for r in range(proxy.rowCount())]

    everything = ["AAA", "BBB", "FOC", "TYPED", "HELD", "LPB", "PED"]
    assert visible() == everything, "the bare proxy hides nothing"
    proxy.set_filters(longs_gate=_off(opens=("HELD",)),
                      longs_exempt=lambda row: row.symbol in {"FOC", "TYPED"})
    assert visible() == ["BBB", "FOC", "TYPED", "HELD", "LPB", "PED"]
    assert proxy.hidden_longs() == 1
    proxy.set_filters(min_score=0.0)  # a partial call keeps the gate
    assert "AAA" not in visible()
    proxy.set_filters(longs_gate=g.Verdict(day=date.today().isoformat()))  # unknown shows
    assert visible() == everything
    proxy.set_filters(longs_gate=None)  # switch off
    assert visible() == everything
    assert len(model.rows()) == 7, "hidden, never deleted"


def test_the_swing_panel_switch_and_banner(env, tmp_path, monkeypatch):  # noqa: F811
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    import chart_snapshot
    import longs_market_gate as g
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    g.clear_cache()
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _s: [])
    panel = MasterAvwapPanel(None, review_events_path=tmp_path / "events.jsonl")
    try:
        panel.set_rows(_swing_rows())
        assert panel.longs_off_toggle.isChecked() is True
        assert panel.proxy.rowCount() == 7, "no verdict yet = unknown = shows"
        panel.set_longs_gate(_off(opens=("HELD",)))
        visible = {panel._row_at_proxy(r).symbol for r in range(panel.proxy.rowCount())}
        assert visible == {"BBB", "TYPED", "HELD", "LPB", "PED"}  # no Focus service here
        assert panel.longs_off_banner_text() == "Longs off: SPY is under its 20-day (since 2026-09-14)"
        assert "(2)" in panel.longs_off_toggle.text()
        panel.longs_off_toggle.setChecked(False)
        assert env[g.SETTING] is False
        assert panel.proxy.rowCount() == 7
        assert panel.longs_off_banner_text() == ""
    finally:
        panel.close()
        g.clear_cache()


# --------------------------------------------------------------------------- phone report
def _report_payload():
    return {
        "generated_at": "2026-09-28 10:00:00",
        "swing_picks": [
            {"symbol": "AAA", "side": "LONG", "bucket": "favorite_setup", "raw": {}},
            {"symbol": "BBB", "side": "SHORT", "bucket": "favorite_setup", "raw": {}},
            {"symbol": "TYPED", "side": "LONG", "bucket": "favorite_setup", "raw": {}},
            {"symbol": "HELD", "side": "LONG", "bucket": "favorite_setup", "raw": {}},
            {"symbol": "LPB", "side": "LONG", "bucket": "favorite_setup",
             "raw": {"setup_family": "leader_pullback"}},
        ],
        "swing_data_current": True,
        "longs": ["TYPED"],
        "shorts": [],
        "bucket_roster": {"favorite_setup": {"LONG": ["AAA", "TYPED", "HELD"], "SHORT": ["BBB"]}},
    }


def test_the_report_drops_hidden_longs_for_one_longs_off_line():
    import autopilot_core as core

    out = core.hide_longs_off(_report_payload(), verdict=_off(opens=("HELD",)), exempt_symbols=["TYPED"])
    assert [p["symbol"] for p in out["swing_picks"]] == ["BBB", "TYPED", "HELD", "LPB"]
    assert out["bucket_roster"] == {"favorite_setup": {"LONG": ["TYPED", "HELD"], "SHORT": ["BBB"]}}
    assert out["longs_off_hidden_count"] == 1
    line = "Longs off: SPY is under its 20-day (since 2026-09-14) - 1 long name(s) hidden"
    assert out["longs_off_line"] == line
    text = core.render_away_report(out)
    assert text.count("Longs off:") == 1
    assert "AAA (LONG)" not in text
    title, message = core.build_swing_push(out)
    assert message.splitlines()[0] == line and "AAA" not in message


def test_the_report_is_untouched_in_a_good_or_unknown_market():
    import autopilot_core as core
    import longs_market_gate as g

    for verdict in (None, g.Verdict(day="2026-09-28"), g.Verdict(day="2026-09-28", verdict="yes")):
        out = core.hide_longs_off(_report_payload(), verdict=verdict)
        assert out == _report_payload()
        assert "Longs off" not in core.render_away_report(out)


def test_the_report_filter_reads_the_switch(env, monkeypatch):  # noqa: F811
    import longs_market_gate as g
    from ui.services.autopilot_service import AutopilotService

    monkeypatch.setattr(g, "current", lambda: _off())
    out = AutopilotService._hide_longs_off(_report_payload(), ["TYPED"])
    assert "AAA" not in [p["symbol"] for p in out["swing_picks"]]
    g.set_enabled(False)
    assert AutopilotService._hide_longs_off(_report_payload(), ["TYPED"]) == _report_payload()


def test_the_phone_push_says_longs_off_once_per_day(monkeypatch):
    from datetime import datetime

    import autopilot_core as core
    import push_notify
    from test_away_push_gating import _Sent, _service

    monkeypatch.setattr(push_notify, "push_configured", lambda: True)
    monkeypatch.setattr(core, "swing_push_due", lambda *_a, **_k: True)
    sent = _Sent()
    monkeypatch.setattr(push_notify, "send_push", sent)
    service = _service()
    payload = core.hide_longs_off(_report_payload(), verdict=_off(opens=("HELD",)), exempt_symbols=["TYPED"])
    for hour in (10, 11):
        service._push_swing_picks(payload, now=datetime(2026, 9, 28, hour, 5))
    service._push_swing_picks(payload, now=datetime(2026, 9, 29, 10, 5))
    firsts = [message.splitlines()[0].startswith("Longs off:") for _title, message in sent.calls]
    assert firsts == [True, False, True]
