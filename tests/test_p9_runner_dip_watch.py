"""p9 runner dip watch: strong names dip under the earnings AVWAP and squeeze on M5.

The trader, 2026-09-27: "When these stocks are running they should be flagged and go into a mini
watchlist that starts to fire when SPY is set up for it and these stocks dip below AVWAPE and start
to compress on lower time frames." Members once per D1 scan, armed when the market works and the
close is within 1 sigma of the earnings AVWAP (15 max by RS), fire on today's completed M5 bars
under the AVWAP inside a 12-bar squeeze (the `compression_break_events` box).
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import runner_dip_watch as rdw  # noqa: E402

ET = ZoneInfo("America/New_York")


# --- daily bars

def _days(count: int) -> list[str]:
    out, day = [], date(2025, 9, 1)
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


DAYS = _days(260)
AS_OF = DAYS[-1]


def _daily(drift: float, *, gap_at: int | None = None, gap: float = 0.0, volume: float = 2_000_000.0):
    bars, level = [], 50.0
    for index, day in enumerate(DAYS):
        level += drift + (gap if index == gap_at else 0.0)
        bars.append({"date": day, "open": level - 0.2 - (gap if index == gap_at else 0.0) * 0.0,
                     "high": level + 0.5, "low": level - 0.5, "close": level, "volume": volume})
    if gap_at is not None:
        bars[gap_at]["open"] = bars[gap_at - 1]["close"] + gap
    return bars


def _universe():
    """25 names: RS ranks follow the drift; the top three (0.9+ percentile) are runners."""
    bars = {f"N{index:02d}": _daily(0.02 * index) for index in range(22)}
    gap_at = len(DAYS) - 30
    for symbol, drift in (("RUNA", 0.5), ("RUNB", 0.55), ("RUNC", 0.6)):
        bars[symbol] = _daily(drift, gap_at=gap_at, gap=3.0)
    dates = {symbol: [DAYS[gap_at]] for symbol in ("RUNA", "RUNB", "RUNC")}
    return bars, dates


def _build(bars, dates, *, caps=None, working="yes"):
    return rdw.build_members(
        bars_by_symbol=bars, spy_bars=_daily(0.01),
        feature_rows=[{"symbol": "N00", "perm_regime_working": working, "perm_regime_working_rule": "trader"}],
        market_cap_by_symbol=caps if caps is not None else {symbol: 5_000.0 for symbol in bars},
        earnings_dates_by_symbol=dates, as_of=AS_OF)


def test_members_are_the_top_rs_strong_names_with_a_known_earnings_avwap():
    bars, dates = _universe()
    payload = _build(bars, dates)
    assert payload["as_of"] == AS_OF and payload["market_working"] == "yes"
    assert sorted(row["symbol"] for row in payload["members"]) == ["RUNA", "RUNB", "RUNC"]
    member = next(row for row in payload["members"] if row["symbol"] == "RUNC")
    assert set(member) >= {"symbol", "as_of", "close", "atr20", "avwape", "sigma", "avwape_z", "anchor_date",
                           "rs_percentile", "strength_sma50_atr", "sector", "armed"}
    assert member["rs_percentile"] >= 0.9 and member["strength_sma50_atr"] >= 2.0
    assert member["anchor_date"] == DAYS[len(DAYS) - 31]  # the session before the reaction
    # A steady runner closes far over its earnings AVWAP: listed, not armed.
    assert member["avwape_z"] > 1.0 and member["armed"] is False and payload["armed"] == []


def test_no_earnings_avwap_or_under_the_liquidity_floor_is_no_member():
    bars, dates = _universe()
    dates.pop("RUNA")
    payload = _build(bars, dates, caps={**{symbol: 5_000.0 for symbol in bars}, "RUNB": 500.0})
    assert [row["symbol"] for row in payload["members"]] == ["RUNC"]


def test_a_stale_name_and_spy_are_never_members():
    bars, dates = _universe()
    bars["RUNA"] = bars["RUNA"][:-1]
    payload = _build(bars, dates)
    assert "RUNA" not in {row["symbol"] for row in payload["members"]}
    assert "SPY" not in {row["symbol"] for row in _build({**bars, "SPY": bars["RUNC"]}, dates)["members"]}
    assert rdw.build_members(bars_by_symbol=bars, spy_bars=[], feature_rows=[], as_of="")["members"] == []


def test_each_member_rule_refuses_on_its_own():
    bars = _daily(0.6, gap_at=230, gap=3.0)
    dates = [DAYS[230]]
    assert rdw._member("X", bars, atr=None, rs=0.95, sector="", earnings_dates=dates, gap_date=None)
    assert rdw._member("X", bars, atr=None, rs=0.89, sector="", earnings_dates=dates, gap_date=None) is None
    # 2 ATR over the SMA50 is the floor: a huge scan ATR makes the same name "not strong".
    assert rdw._member("X", bars, atr=50.0, rs=0.95, sector="", earnings_dates=dates, gap_date=None) is None
    falling = _daily(0.6)[:200] + _daily(-0.6)[200:]
    assert rdw._member("X", falling, atr=None, rs=0.95, sector="", earnings_dates=[DAYS[230]],
                       gap_date=None) is None  # under the 100/200-day
    assert rdw._member("X", bars, atr=None, rs=0.95, sector="", earnings_dates=[DAYS[100]],
                       gap_date=None) is None  # reaction more than 120 sessions back


# --- armed

def _m(symbol, rs, z, *, sigma=1.0, avwape=100.0):
    return {"symbol": symbol, "rs_percentile": rs, "avwape": avwape, "sigma": sigma, "close": avwape + z * sigma}


def test_armed_is_a_working_market_and_within_one_sigma_of_the_avwape():
    members = [_m("UNDER", 0.95, -0.5), _m("EDGE", 0.93, 1.0), _m("FAR", 0.99, 1.01)]
    assert rdw.arm(members, "yes") == ["UNDER", "EDGE"]
    assert [row["armed"] for row in members] == [True, True, False]
    assert rdw.arm(members, "no") == [] and not any(row["armed"] for row in members)
    assert rdw.arm(members, "unknown") == []


def test_at_most_fifteen_armed_by_rs():
    members = [_m(f"S{index:02d}", 0.9 + index / 1000, 0.0) for index in range(20)]
    armed = rdw.arm(members, "yes")
    assert len(armed) == rdw.ARMED_MAX == 15
    assert armed[0] == "S19" and "S04" not in armed and "S05" in armed


def test_under_the_avwape_names_take_the_cap_first_then_rs():
    """Lead, 2026-09-27: Friday armed 15 by RS and left CMBT and FRO (under the AVWAPE) out."""
    members = [_m(f"S{index:02d}", 0.99 - index / 1000, 0.5) for index in range(20)]
    members += [_m("UNDERLOW", 0.901, -0.3), _m("UNDERHI", 0.95, -1.2)]
    armed = rdw.arm(members, "yes")
    assert len(armed) == 15 and armed[:2] == ["UNDERHI", "UNDERLOW"]
    assert armed[2:] == [f"S{index:02d}" for index in range(13)]


def test_the_armed_list_serves_the_next_sessions_only():
    payload = {"as_of": "2026-09-25", "market_working": "yes",
               "members": [{"symbol": "gtlb", "armed": True}, {"symbol": "NVDA", "armed": False}]}
    assert rdw.armed_symbols(payload, today="2026-09-28") == ["GTLB"]
    assert rdw.armed_symbols(payload, today="2026-09-25") == []  # the scan's own session
    assert rdw.armed_symbols(payload, today="2026-09-30") == []  # stale
    assert rdw.armed_symbols({**payload, "market_working": "no"}, today="2026-09-28") == []
    assert rdw.armed_symbols(None, today="2026-09-28") == []
    assert rdw.status_line(payload, today="2026-09-28") == "Runner dips: 1 armed (GTLB)."
    assert rdw.status_line(payload, today="2026-09-30") == ""
    assert "none armed" in rdw.status_line({**payload, "market_working": "no"}, today="2026-09-28")


# --- the M5 fire

SESSION = date(2026, 9, 28)


def _m5(count: int, *, start=None, center=99.0, width=0.1, day=SESSION, prior: int = 0):
    """``prior`` bars late on the previous session, then ``count`` bars from 09:30 ET."""
    bars = []
    previous = day - timedelta(days=3 if day.weekday() == 0 else 1)
    for index in range(prior):
        stamp = datetime.combine(previous, datetime.min.time()).replace(hour=15, minute=55) \
            - timedelta(minutes=5 * (prior - 1 - index))
        bars.append({"date": stamp, "open": center, "high": center + width, "low": center - width,
                     "close": center, "volume": 1000.0})
    for index in range(count):
        stamp = datetime.combine(day, datetime.min.time()).replace(hour=9, minute=30) + timedelta(minutes=5 * index)
        bars.append({"date": stamp, "open": center, "high": center + width, "low": center - width,
                     "close": center, "volume": 1000.0})
    return bars


def _now(bars, minutes_after_last_start=5):
    return bars[-1]["date"] + timedelta(minutes=minutes_after_last_start)


MEMBER = {"symbol": "gtlb", "avwape": 100.0, "as_of": "2026-09-25", "rs_percentile": 0.95,
          "strength_sma50_atr": 2.4}


def test_a_close_under_the_avwape_in_a_squeeze_fires_with_the_box():
    bars = _m5(24)
    bars[-3]["high"], bars[-5]["low"] = 99.3, 98.8
    hit = rdw.evaluate(MEMBER, bars, now=_now(bars), tz=ET)
    assert hit is not None and hit.symbol == "GTLB"
    assert (hit.box_high, hit.box_low, hit.close) == (99.3, 98.8, 99.0)
    assert hit.bar_time.tzinfo is not None and hit.session == "2026-09-28"
    assert hit.line == "GTLB runner dip: strong name under the earnings VWAP (100.00), squeezing on M5 - box 98.80-99.30"
    detail = rdw.fire_detail(hit, spy_price=600.0)
    assert detail["ts"].endswith("-04:00") and detail["price"] == 99.0 and detail["spy_price"] == 600.0
    assert {"avwape", "box_high", "box_low", "as_of", "rs_percentile", "strength_sma50_atr"} <= set(detail)


def test_no_fire_at_or_over_the_avwape_or_without_a_squeeze():
    bars = _m5(24, center=100.0)
    assert rdw.evaluate(MEMBER, bars, now=_now(bars), tz=ET) is None  # at the AVWAP is not under it
    bars = _m5(24)
    bars[-2]["high"] = 99.0 + 3.0  # one wide bar: box 3.1 vs ATR ~0.35
    assert rdw.evaluate(MEMBER, bars, now=_now(bars), tz=ET) is None


def test_only_completed_bars_of_todays_regular_session_count():
    bars = _m5(24)
    # A forming bar is not read: the fire bar is the last COMPLETED one.
    forming = rdw.evaluate(MEMBER, bars, now=_now(bars, 4), tz=ET)
    assert forming is not None and forming.bar_time == bars[-2]["date"].replace(tzinfo=ET)
    # Yesterday's bars never fire today.
    assert rdw.evaluate(MEMBER, bars, now=_now(bars) + timedelta(days=1), tz=ET) is None
    # Too few bars for the ATR is unknown: no fire.
    few = _m5(20)
    assert rdw.evaluate(MEMBER, few, now=_now(few), tz=ET) is None
    # The box may not reach into yesterday; the ATR may.
    early = _m5(11, prior=15)
    assert rdw.evaluate(MEMBER, early, now=_now(early), tz=ET) is None
    early = _m5(12, prior=15)
    assert rdw.evaluate(MEMBER, early, now=_now(early), tz=ET) is not None


def test_missing_data_is_no_fire():
    bars = _m5(24)
    bars[5]["close"] = None
    assert rdw.evaluate(MEMBER, bars, now=_now(bars), tz=ET) is None
    assert rdw.evaluate({**MEMBER, "avwape": None}, _m5(24), now=_now(_m5(24)), tz=ET) is None
    assert rdw.evaluate(MEMBER, [], now=datetime(2026, 9, 28, 12), tz=ET) is None


def test_the_box_matches_compression_break_events():
    """Parity: where the shadow engine breaks a squeeze at bar k, the runner box on bars[:k] is its box."""
    import m5_signal_engines as engines

    assert (rdw.SQUEEZE_BOX_BARS, rdw.SQUEEZE_ATR_BARS, rdw.SQUEEZE_RANGE_ATR) == (
        engines.SQUEEZE_BOX_BARS, engines.SQUEEZE_ATR_BARS, engines.SQUEEZE_RANGE_ATR)
    bars = _m5(30, prior=20, width=0.4)
    for index, bar in enumerate(bars):
        wiggle = ((index * 7) % 5 - 2) * 0.05
        bar["high"] += wiggle if wiggle > 0 else 0.0
        bar["low"] += wiggle if wiggle < 0 else 0.0
        bar["close"] = bar["open"] + wiggle
    bars[-1]["close"] = bars[-1]["high"] = 101.0
    events = engines.compression_break_events(bars, symbol="GTLB", side="LONG", now=_now(bars), tz=ET)
    assert events, "the fixture must break a squeeze"
    event = events[-1]
    k = next(index for index, bar in enumerate(bars) if bar["date"].replace(tzinfo=ET) == event.bar_time)
    head = bars[:k]
    hit = rdw.evaluate({**MEMBER, "avwape": 1000.0}, head, now=_now(head), tz=ET)
    assert hit is not None
    details = dict(event.details)
    assert hit.box_high == pytest.approx(event.level) and hit.box_low == pytest.approx(event.stop)
    assert hit.range_atr == pytest.approx(details["range_atr"])


# --- grading (shadow)

def _event(symbol, session, price, spy_price=None, action=rdw.FIRED_ACTION):
    return {"action": action, "symbol": symbol,
            "detail": {"session": session, "price": price, "spy_price": spy_price}}


def test_fires_are_graded_once_per_name_and_session_and_pending_is_never_zero():
    fires = rdw.fires_from_events([
        _event("GTLB", "2026-09-28", 50.0, 600.0),
        _event("GTLB", "2026-09-28", 49.0, 600.0),  # a restart re-fire: the first wins
        _event("NVDA", "2026-09-28", 100.0),        # no SPY price: vs SPY unknown
        _event("AMD", "2026-09-28", 10.0, action="watch_fired"),
    ])
    assert [(row["symbol"], row["price"]) for row in fires] == [("GTLB", 50.0), ("NVDA", 100.0)]
    spy_days = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05"]
    spy = {day: 600.0 + index for index, day in enumerate(spy_days)}
    closes = {"GTLB": {"2026-09-29": 51.0, "2026-10-05": 55.0}, "NVDA": {"2026-09-29": 99.0}}
    graded = rdw.grade_fires(fires, closes, spy)
    gtlb, nvda = graded
    assert gtlb["return_1"] == pytest.approx(2.0) and gtlb["spy_1"] == pytest.approx(1 / 6, abs=1e-3)
    assert gtlb["return_5"] == pytest.approx(10.0) and gtlb["return_10"] is None
    assert nvda["return_1"] == pytest.approx(-1.0) and nvda["spy_1"] is None
    line = rdw.grade_line(graded)
    assert line.startswith("Runner dips (shadow, M5 trigger untested): 2 fired")
    assert "1d avg +0.5%, beat SPY 1/1" in line and "10d pending 2" in line
    assert rdw.grade_line([]) == "Runner dips (shadow, M5 trigger untested): no fire yet."


# --- the one file owner

def test_the_scan_publishes_the_runner_file_beside_the_long_setups(tmp_path, monkeypatch):
    import long_setups
    import long_setups_store

    bars, dates = _universe()
    monkeypatch.setattr(long_setups_store, "read_earnings_dates", lambda path=None: dates)
    kwargs = dict(bars_by_symbol=bars, spy_bars=_daily(0.01),
                  feature_rows=[{"symbol": "N00", "perm_regime_working": "yes", "perm_regime_working_rule": "trader"}],
                  market_cap_by_symbol={symbol: 5_000.0 for symbol in bars}, as_of=AS_OF,
                  now=datetime(2026, 9, 25, 16, 30))
    runner = tmp_path / "runner_dip_watch.json"
    payload = long_setups_store.publish_long_setups(**kwargs, path=tmp_path / "a.json",
                                                    history_path=tmp_path / "b.json", runner_path=runner)
    expected = long_setups.build_rows(**{k: v for k, v in kwargs.items() if k != "now"},
                                      earnings_dates_by_symbol=dates)
    assert payload["rows"] == expected["rows"]  # the long setups are untouched by the runner file
    written = json.loads(runner.read_text(encoding="utf-8"))
    assert written["as_of"] == AS_OF and written["schema_version"] == rdw.SCHEMA_VERSION
    assert sorted(row["symbol"] for row in written["members"]) == ["RUNA", "RUNB", "RUNC"]
    assert long_setups_store.read_runner_dip_watch(runner) == written


def test_a_failed_runner_build_keeps_the_long_setups_and_the_last_good_file(tmp_path, monkeypatch):
    import long_setups_store

    runner = tmp_path / "runner_dip_watch.json"
    runner.write_text(json.dumps({"as_of": "2026-09-24", "members": [{"symbol": "OLD"}]}), encoding="utf-8")
    bars, dates = _universe()
    monkeypatch.setattr(long_setups_store, "read_earnings_dates", lambda path=None: dates)

    def _boom(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(long_setups_store.runner_dip_watch, "build_members", _boom)
    payload = long_setups_store.publish_long_setups(
        bars_by_symbol=bars, spy_bars=_daily(0.01), feature_rows=[], as_of=AS_OF,
        market_cap_by_symbol={symbol: 5_000.0 for symbol in bars},
        path=tmp_path / "a.json", history_path=tmp_path / "b.json", runner_path=runner)
    assert payload["as_of"] == AS_OF and (tmp_path / "a.json").is_file()
    assert json.loads(runner.read_text(encoding="utf-8"))["members"] == [{"symbol": "OLD"}]


def test_a_scan_with_no_current_bars_keeps_the_last_good_runner_file(tmp_path):
    import long_setups_store

    runner = tmp_path / "runner_dip_watch.json"
    runner.write_text(json.dumps({"as_of": "2026-09-24", "members": [{"symbol": "OLD"}]}), encoding="utf-8")
    kept = long_setups_store.publish_runner_dip_watch(
        {"bars_by_symbol": {"AAA": _daily(0.1)[:-1]}, "spy_bars": [], "feature_rows": [], "as_of": AS_OF},
        path=runner)
    assert kept["members"] == [{"symbol": "OLD"}]
    assert json.loads(runner.read_text(encoding="utf-8"))["as_of"] == "2026-09-24"
    assert long_setups_store.publish_runner_dip_watch({"bars_by_symbol": {}, "spy_bars": [], "feature_rows": [],
                                                       "as_of": ""}, path=runner) is None


# --- the auto-longs feed (the bounce bot only has M5 bars for its scan set)

def _runner_file(tmp_path, *, as_of="2026-09-25", working="yes", name="runner_dip_watch.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"as_of": as_of, "market_working": working, "members": [
        {"symbol": "GTLB", "armed": True}, {"symbol": "NVDA", "armed": False}]}), encoding="utf-8")
    return path


def test_armed_names_are_appended_to_autolongs_through_the_autopilot(tmp_path):
    import autopilot_core as core

    runner, auto = _runner_file(tmp_path), tmp_path / "autolongs.txt"
    auto.write_text("AAPL\n", encoding="utf-8")
    assert core.sync_runner_dip_auto_longs(today=date(2026, 9, 28), path=runner, auto_longs_path=auto) == ["GTLB"]
    assert auto.read_text(encoding="utf-8").split() == ["AAPL", "GTLB"]
    assert core.sync_runner_dip_auto_longs(today=date(2026, 9, 28), path=runner, auto_longs_path=auto) == []
    assert auto.read_text(encoding="utf-8").split() == ["AAPL", "GTLB"]


def test_a_stale_or_not_working_runner_file_adds_nothing(tmp_path):
    import autopilot_core as core

    auto = tmp_path / "autolongs.txt"
    auto.write_text("AAPL\n", encoding="utf-8")
    for runner, today in ((_runner_file(tmp_path), date(2026, 9, 25)),
                          (_runner_file(tmp_path, working="no", name="off.json"), date(2026, 9, 28)),
                          (tmp_path / "missing.json", date(2026, 9, 28))):
        assert core.sync_runner_dip_auto_longs(today=today, path=runner, auto_longs_path=auto) == []
    assert auto.read_text(encoding="utf-8").split() == ["AAPL"]


def _service(*, enabled=True, shadow=True):
    from ui.services.autopilot_service import AutopilotService

    service = AutopilotService.__new__(AutopilotService)
    service._enabled = enabled
    service._logged = []
    service._log = service._logged.append
    service._shadow_research_allowed = lambda: shadow
    return service


def test_the_autopilot_tick_syncs_once_per_change_and_never_in_strict_off(tmp_path, monkeypatch):
    import autopilot_core as core
    import project_paths
    from ui.services import autopilot_service

    calls = []
    monkeypatch.setattr(core, "sync_runner_dip_auto_longs", lambda **kw: calls.append(kw) or ["GTLB"])
    monkeypatch.setattr(project_paths, "RUNNER_DIP_WATCH_FILE", _runner_file(tmp_path))
    monkeypatch.setattr(autopilot_service, "AUTO_LONGS_FILE", tmp_path / "autolongs.txt")
    now = datetime(2026, 9, 28, 6, 45)
    _service(enabled=False, shadow=False)._maybe_sync_runner_dip_names(now)
    assert calls == []
    service = _service(enabled=False, shadow=True)
    service._maybe_sync_runner_dip_names(now)
    service._maybe_sync_runner_dip_names(now)
    assert calls == [{"today": date(2026, 9, 28)}]
    assert "GTLB" in service._logged[-1]
    (tmp_path / "autolongs.txt").write_text("AAPL\n", encoding="utf-8")  # the open scan rewrote it
    service._maybe_sync_runner_dip_names(now)
    assert len(calls) == 2


def test_the_tick_runs_the_sync_after_the_day_roll_clear():
    import inspect

    from ui.services.autopilot_service import AutopilotService

    source = inspect.getsource(AutopilotService._tick)
    assert source.index("_maybe_clear_stale_auto_lists(now)") < source.index("_maybe_sync_runner_dip_names(now)")


# --- the Setup Tracker line (worker side)

def test_the_long_leaders_section_carries_the_runner_dip_grade(tmp_path, monkeypatch):
    import pandas as pd

    import project_paths
    import review_events
    from diagnostics.artifact_io import atomic_write_json
    from ui.services import working_lately_service as service

    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(json.dumps(row) for row in (
        {"action": "runner_dip_fired", "symbol": "GTLB", "ts": "2026-09-21T10:00:00",
         "detail": {"session": "2026-09-21", "price": 50.0, "spy_price": 600.0}},
        {"action": "watch_fired", "symbol": "AMD", "ts": "2026-09-21T10:00:00", "detail": {}},
    )) + "\n", encoding="utf-8")
    monkeypatch.setattr(review_events, "review_event_sources", lambda *a, **k: [events])
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    days = pd.to_datetime(["2026-09-21", "2026-09-22", "2026-09-23"])
    pd.DataFrame({"datetime": days, "close": [600.0, 606.0, 700.0]}).to_parquet(bars_dir / "SPY.parquet")
    pd.DataFrame({"datetime": days, "close": [50.0, 52.0, 99.0]}).to_parquet(bars_dir / "GTLB.parquet")
    monkeypatch.setattr(project_paths, "MASTER_AVWAP_DAILY_BARS_DIR", bars_dir)
    # 09-23's bar is still forming: completed bars only.
    monkeypatch.setattr(service, "_last_completed_session", lambda: date(2026, 9, 22))
    current, history = tmp_path / "long_setups.json", tmp_path / "long_setups_history.json"
    atomic_write_json(current, {"as_of": "2026-09-25", "market_working": "yes", "rows": []})
    atomic_write_json(history, {"rows": []})
    monkeypatch.setattr(project_paths, "LONG_SETUPS_FILE", current)
    monkeypatch.setattr(project_paths, "LONG_SETUPS_HISTORY_FILE", history)
    service._LOOKING_BACK_CACHE.clear()
    try:
        lines = service.read_long_leader_lines()
    finally:
        service._LOOKING_BACK_CACHE.clear()
    assert lines[-1].startswith("strong + under AVWAPE (shadow)")
    assert lines[-2] == ("Runner dips (shadow, M5 trigger untested): 1 fired · 1d avg +4.0%, beat SPY 1/1"
                         " · 5d pending 1 · 10d pending 1")
