"""S8 retest-entry study: synthetic alerts for fill, no-fill, stop-first, and the loaders."""

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_warehouse import retest_entry as rt  # noqa: E402

NY = ZoneInfo("America/New_York")
LA = ZoneInfo("America/Los_Angeles")
SESSION = date(2026, 9, 24)
T0 = datetime(2026, 9, 24, 10, 0, tzinfo=NY)


def _bars(rows):
    """rows: (open, high, low, close) per 5-minute bar from T0."""
    return [
        {"dt": T0 + timedelta(minutes=5 * i), "open": o, "high": h, "low": low, "close": c}
        for i, (o, h, low, c) in enumerate(rows)
    ]


# Ten flat bars of range 1.0 give an ATR of 1.0 at bar 9 (the alert bar).
PRE = [(100.0, 100.5, 99.5, 100.0)] * 9 + [(100.0, 100.5, 99.5, 100.2)]
ALERT_BAR = T0 + timedelta(minutes=45)


def _alert(**extra):
    base = {"family": "vwap", "side": "long", "symbol": "AAA", "alert_bar": ALERT_BAR,
            "entry": 100.2, "level": 99.0}
    base.update(extra)
    return base


def test_retest_fills_at_the_limit_and_wins():
    # limit = 99.0 + 0.25 * 1.0 = 99.25; stop 98.5 -> retest risk 0.75, target 100.0
    after = [(100.1, 100.3, 99.6, 99.8), (99.6, 99.7, 99.2, 99.4), (99.4, 100.1, 99.3, 100.0)]
    out = rt.simulate(_alert(), _bars(PRE + after))
    assert out["status"] == "ok" and out["atr"] == pytest.approx(1.0)
    assert out["retest_filled"] and out["retest_bar"] == 2
    assert out["retest_fill"] == pytest.approx(99.25)
    assert out["retest_r"] == 1.0 and out["retest_exit"] == "target"
    # Flag entry 100.2, risk 1.7: no target, no stop -> close R
    assert out["flag_exit"] == "close"
    assert out["flag_r"] == pytest.approx((100.0 - 100.2) / 1.7)
    assert out["stop"] == pytest.approx(98.5)  # level - 0.5 ATR


def test_no_fill_when_price_never_returns_within_six_bars():
    after = [(100.3, 100.9, 99.3, 100.8)] + [(100.8, 101.5, 100.5, 101.2)] * 5 + [
        (101.0, 101.1, 98.0, 98.2)  # 7th bar crashes through the limit: too late
    ]
    out = rt.simulate(_alert(), _bars(PRE + after))
    assert out["retest_filled"] is False and out["retest_r"] is None
    # Flag target 101.9 is never reached; the 7th bar takes the 98.5 stop.
    assert out["flag_exit"] == "stop" and out["flag_r"] == -1.0


def test_stop_first_when_one_bar_touches_both():
    # Flag entry 100.2, stop 98.5, target 101.9: bar 1 spans both -> stop first.
    after = [(100.2, 102.0, 98.4, 101.0)]
    out = rt.simulate(_alert(), _bars(PRE + after))
    assert out["flag_r"] == -1.0 and out["flag_exit"] == "stop"
    # The same bar also fills the retest and hits its stop: a loss, not a win.
    assert out["retest_filled"] and out["retest_r"] == -1.0 and out["retest_exit"] == "stop"


def test_only_bars_after_the_alert_count():
    # A bar BEFORE the alert trades through the limit; nothing after does.
    pre = list(PRE)
    pre[8] = (100.0, 100.5, 99.0, 100.0)
    after = [(100.3, 100.8, 99.6, 100.7)] * 6
    out = rt.simulate(_alert(), _bars(pre + after))
    assert out["retest_filled"] is False


def test_a_logged_stop_is_ignored_both_entries_share_level_minus_half_atr():
    out = rt.simulate(_alert(stop=99.6), _bars(PRE + [(100.2, 100.4, 100.0, 100.3)]))
    assert out["stop"] == pytest.approx(99.0 - rt.STOP_ATR * 1.0)


def test_short_side_mirrors():
    # Short at 100.2 with level 101.0: limit 100.75, stop 101.5
    after = [(100.2, 100.8, 100.0, 100.6), (100.6, 100.6, 99.9, 100.0)]
    out = rt.simulate(_alert(side="short", level=101.0), _bars(PRE + after))
    assert out["retest_fill"] == pytest.approx(100.75) and out["retest_bar"] == 1
    assert out["retest_r"] == 1.0  # target 100.0 on bar 2
    assert out["flag_r"] == pytest.approx((100.2 - 100.0) / 1.3)


def test_movers_level_is_the_names_own_low_up_to_the_flag():
    pre = list(PRE)
    pre[3] = (100.0, 100.5, 99.2, 100.0)  # episode low
    after = [(100.2, 100.3, 99.4, 99.6), (99.6, 100.4, 99.5, 100.3)]
    alert = {"family": "movers_pullback", "side": "long", "symbol": "AAA",
             "alert_bar": ALERT_BAR, "entry": None, "level": None,
             "episode_start": T0 + timedelta(minutes=10)}
    out = rt.simulate(alert, _bars(pre + after))
    assert out["level"] == pytest.approx(99.2)
    assert out["flag_entry"] == pytest.approx(100.2)  # the flag bar's close
    atr = out["atr"]  # 13 bars of range 1.0 and one of 1.3
    assert atr == pytest.approx(10.3 / 10)
    assert out["stop"] == pytest.approx(99.2 - 0.5 * atr)
    assert out["retest_fill"] == pytest.approx(99.2 + 0.25 * atr)


def test_missing_alert_bar_is_unknown_not_a_result():
    out = rt.simulate(_alert(alert_bar=T0 + timedelta(hours=3)), _bars(PRE))
    assert out["status"] == "no_alert_bar" and "flag_r" not in out


def test_run_study_reports_both_evs_n_and_no_fill_share():
    fill = PRE + [(100.1, 100.3, 99.6, 99.8), (99.6, 99.7, 99.2, 99.4), (99.4, 100.1, 99.3, 100.0)]
    nofill = PRE + [(100.3, 102.0, 100.1, 101.95)]
    series = {"AAA": _bars(fill), "BBB": _bars(nofill)}
    alerts = [
        {"source": "m5_alert", "session": SESSION, **_alert(symbol="AAA")},
        {"source": "m5_alert", "session": SESSION, **_alert(symbol="BBB")},
        {"source": "m5_alert", "session": SESSION, **_alert(symbol="CCC")},  # no bars
    ]
    report = rt.run_study(alerts, lambda session, symbols: series)
    assert report["schema"] == rt.SCHEMA and report["skipped"] == {"no_bars": 1}
    (row,) = report["families"]
    assert row["n"] == 2 and row["retest_n_filled"] == 1
    assert row["retest_no_fill_share"] == pytest.approx(0.5)
    assert row["retest_ev_r_per_fill"] == pytest.approx(1.0)
    assert row["retest_ev_r_per_alert"] == pytest.approx(0.5)
    assert row["flag_ev_r"] == pytest.approx(((100.0 - 100.2) / 1.7 + 1.0) / 2)


def test_load_m5_alerts_reads_confirmed_rows_once(tmp_path, monkeypatch):
    import pandas as pd

    calls = []
    real = pd.read_csv

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    candle = json.dumps({"bounce": {"time": "20260924  07:45:00"}})
    rows = [
        {"event_id": "A1", "event_type": "confirmed", "trade_date": "2026-09-24", "symbol": "aaa",
         "direction": "long", "bounce_types": "vwap;ema", "entry_price": "100.2",
         "stop_price": "98.5", "levels_json": json.dumps({"vwap": 99.0, "ema": 98.0}),
         "candle_json": candle, "extra": "x"},
        {"event_id": "A1", "event_type": "confirmed", "trade_date": "2026-09-24", "symbol": "aaa",
         "direction": "long", "bounce_types": "vwap", "entry_price": "1", "stop_price": "0",
         "levels_json": "{}", "candle_json": candle, "extra": "x"},
        {"event_id": "B1", "event_type": "near_miss", "trade_date": "2026-09-24", "symbol": "bbb",
         "direction": "long", "bounce_types": "vwap", "entry_price": "1", "stop_price": "0",
         "levels_json": "{}", "candle_json": candle, "extra": "x"},
        {"event_id": "C1", "event_type": "confirmed", "trade_date": "2026-07-01", "symbol": "ccc",
         "direction": "long", "bounce_types": "vwap", "entry_price": "1", "stop_price": "0",
         "levels_json": "{}", "candle_json": candle, "extra": "x"},
    ]
    path = tmp_path / "candidates.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    alerts = rt.load_m5_alerts(path, since=date(2026, 9, 1), local_zone=LA)
    assert calls and calls[0]["chunksize"] == 250_000 and "extra" not in calls[0]["usecols"]
    (alert,) = alerts
    assert alert["family"] == "vwap" and alert["level"] == 99.0 and alert["symbol"] == "AAA"
    assert alert["alert_bar"] == datetime(2026, 9, 24, 10, 45, tzinfo=NY)


def test_load_movers_dip_reads_flag_rows_only(tmp_path):
    path = tmp_path / "dip.jsonl"
    rows = [
        {"kind": "flag", "session": "2026-09-24", "side": "long", "symbol": "tech",
         "episode": "2026-09-24T12:55:00-04:00", "flagged_bar": "2026-09-24T13:40:00-04:00"},
        {"kind": "outcome", "session": "2026-09-24", "side": "long", "symbol": "TECH"},
        {"kind": "flag", "session": "2026-09-24", "side": "short", "state": "rally", "symbol": "X",
         "episode": "2026-09-24T12:55:00-04:00", "flagged_bar": "2026-09-24T13:40:00-04:00"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{broken", encoding="utf-8")
    alerts = rt.load_movers_dip(path, since=date(2026, 9, 1))
    assert [(a["family"], a["side"], a["symbol"]) for a in alerts] == [
        ("movers_pullback", "long", "TECH"), ("movers_rally", "short", "X")]
    assert alerts[0]["alert_bar"] == datetime(2026, 9, 24, 13, 40, tzinfo=NY)


def test_lake_bars_keep_completed_rth_bars_once_per_interval():
    from datetime import timezone

    utc = timezone.utc
    t = datetime(2026, 9, 24, 14, 0, tzinfo=utc)
    bar = {"symbol": "AAA", "interval_start": t, "session_phase": "RTH", "is_complete": True,
           "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}
    rows = [
        bar,
        {**bar, "close": 1.6},  # a second revision of the same bar
        {**bar, "interval_start": t + timedelta(minutes=5), "is_complete": False},
        {**bar, "interval_start": t - timedelta(hours=5), "session_phase": "PRE"},
    ]
    seen = {}

    class Store:
        def read_rows(self, dataset, partition, **kwargs):
            seen.update(dataset=dataset, partition=partition, **kwargs)
            return rows

    out = rt.lake_bars_for(Store())(SESSION, ["AAA", "AAA"])
    assert seen["dataset"] == "bar_m5" and seen["partition"] == "month=2026-09"
    assert seen["symbols"] == ["AAA"]
    assert seen["interval_start_range"][0] == datetime(2026, 9, 24, 4, 0, tzinfo=utc)
    assert out == {"AAA": [{"dt": t, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.6}]}
