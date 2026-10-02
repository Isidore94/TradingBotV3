"""AVWAPE quick test (`avwape_quick_test`): a Setup Tracker setup, testing only, both sides.

The trader, 2026-10-02: "a stock tests the LOWER_1 stdev level for longs (invert for shorts, so
UPPER_1) then hammers through AVWAPE ... a quick test, not a prolonged period on the LOWER_1
level". A wick touch of the band with at most ONE close beyond it, the reclaim close through
the earnings AVWAP within 3 sessions of the touch. Rows only: no points, alert, Focus or phone.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import avwape_quick_test as aqt  # noqa: E402
import long_setups as ls  # noqa: E402


def _days(count: int) -> list[str]:
    out, day = [], date(2026, 1, 5)
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    return out


DAYS = _days(200)
#: The earnings reaction is bar 5 (a gap), so the AVWAPE anchors at bar 4.
EARNINGS = [DAYS[5]]
VOLUME = 2_000_000


def _flat(level: float) -> dict:
    return {"open": level, "high": level + 0.5, "low": level - 0.5, "close": level, "volume": VOLUME}


def _base(calm: int = 10) -> list[dict]:
    """Five flat bars, an earnings gap, a 40-bar wave (so sigma is real), then calm bars at AVWAPE."""
    bars = [_flat(100.0) for _ in range(5)]
    bars.append({"open": 106.0, "high": 106.5, "low": 105.0, "close": 106.0, "volume": VOLUME})
    bars += [_flat(103.0 + 4.0 * math.sin(k / 3.0)) for k in range(40)]
    for index, bar in enumerate(bars):
        bar["date"] = DAYS[index]
    return _extend(bars, [(-0.3, 0.1, 0.3)] * calm)


def _extend(bars: list[dict], shape, *, short: bool = False) -> list[dict]:
    """Append bars given as (low, close, high) in sigmas from the prefix's AVWAPE (mirrored for a short)."""
    out = [dict(bar) for bar in bars]
    for low_z, close_z, high_z in shape:
        if short:
            low_z, close_z, high_z = -high_z, -close_z, -low_z
        anchor = ls.earnings_anchor_index(out, earnings_dates=EARNINGS)
        level, sigma = ls.avwap_bands(out, anchor)
        close = level + close_z * sigma
        out.append({"date": DAYS[len(out)], "open": close, "high": level + high_z * sigma,
                    "low": level + low_z * sigma, "close": close, "volume": VOLUME})
    return out


#: A quiet bar just under AVWAPE, a wick to LOWER_1 that closes back over it, then the reclaim.
QUIET = (-0.6, -0.3, -0.1)
TOUCH = (-1.6, -0.5, -0.2)
RECLAIM = (-0.4, 0.6, 0.9)


def _detect(bars, side="LONG", **kwargs):
    kwargs.setdefault("earnings_dates", EARNINGS)
    kwargs.setdefault("atr", 2.0)
    return aqt.detect(bars, side=side, **kwargs)


# --- the rule

def test_the_rules_numbers():
    assert (aqt.SETUP, aqt.LABEL, aqt.STATUS_TESTING) == ("avwape_quick_test", "AVWAPE quick test", "testing")
    assert (aqt.TEST_WINDOW_SESSIONS, aqt.LOOKBACK_SESSIONS, aqt.MAX_CLOSES_BEYOND_BAND) == (3, 10, 1)


def test_fires_long_with_entry_stop_and_exit():
    bars = _extend(_base(), [QUIET, TOUCH, RECLAIM])
    row = _detect(bars)
    assert row is not None
    assert (row["setup"], row["side"], row["status"]) == ("avwape_quick_test", "LONG", "testing")
    assert row["as_of"] == bars[-1]["date"] and row["test_date"] == bars[-2]["date"]
    assert row["sessions_since_test"] == 1 and row["closes_beyond"] == 0
    assert row["test_extreme"] == round(bars[-2]["low"], 4)
    assert row["close"] > row["avwape"] > row["band"] and row["band"] == pytest.approx(row["avwape"] - row["sigma"], abs=1e-3)
    assert row["avwape_z"] > 0
    assert row["entry_limit"] == round(bars[-1]["close"] - ls.ENTRY_ATR_BELOW * 2.0, 4)
    stop = min(bar["low"] for bar in bars[-3:])
    assert row["stop"] == round(stop, 4)
    assert row["exit"] == f"hold up to 10 sessions, stop under {stop:.2f}"
    assert row["reasons"] == [f"tested LOWER_1 on {bars[-2]['date']} (0 closes under)", "closed back above AVWAPE"]
    assert _detect(bars, side="SHORT") is None


def test_fires_short_as_the_mirror():
    bars = _extend(_base(), [QUIET, TOUCH, RECLAIM], short=True)
    row = _detect(bars, side="SHORT")
    assert row is not None and row["side"] == "SHORT"
    assert row["close"] < row["avwape"] < row["band"]
    assert row["band"] == pytest.approx(row["avwape"] + row["sigma"], abs=1e-3)
    assert row["test_extreme"] == round(bars[-2]["high"], 4)
    assert row["entry_limit"] == round(bars[-1]["close"] + ls.ENTRY_ATR_BELOW * 2.0, 4)
    stop = max(bar["high"] for bar in bars[-3:])
    assert row["stop"] == round(stop, 4)
    assert row["exit"] == f"hold up to 10 sessions, stop over {stop:.2f}"
    assert row["reasons"][0].startswith("tested UPPER_1 on ") and row["reasons"][1] == "closed back below AVWAPE"
    assert _detect(bars, side="LONG") is None


@pytest.mark.parametrize("short", [False, True])
def test_the_reclaim_must_come_within_three_sessions_of_the_touch(short):
    side = "SHORT" if short else "LONG"
    third = _extend(_base(), [TOUCH, QUIET, RECLAIM], short=short)
    row = _detect(third, side=side)
    assert row is not None and row["sessions_since_test"] == 2 and row["test_date"] == third[-3]["date"]
    fourth = _extend(_base(), [TOUCH, QUIET, QUIET, RECLAIM], short=short)
    assert _detect(fourth, side=side) is None


@pytest.mark.parametrize("short", [False, True])
def test_two_closes_beyond_the_band_in_the_lookback_is_no_setup(short):
    side = "SHORT" if short else "LONG"
    under = (-1.8, -1.3, -0.9)
    one = _extend(_base(), [QUIET, under, RECLAIM], short=short)
    row = _detect(one, side=side)
    assert row is not None and row["closes_beyond"] == 1
    assert f"(1 close {'over' if short else 'under'})" in row["reasons"][0]
    two = _extend(_base(), [under, QUIET, QUIET, QUIET, QUIET, QUIET, under, RECLAIM], short=short)
    assert _detect(two, side=side) is None


def test_an_old_close_under_the_band_out_of_the_lookback_does_not_count():
    under = (-1.8, -1.3, -0.9)
    bars = _extend(_base(), [under, *[QUIET] * 9, under, RECLAIM])
    row = _detect(bars)
    assert row is not None and row["closes_beyond"] == 1


@pytest.mark.parametrize("short", [False, True])
def test_the_touch_and_the_reclaim_in_one_bar_fires(short):
    bars = _extend(_base(), [QUIET, QUIET, (-1.5, 0.5, 0.8)], short=short)
    row = _detect(bars, side="SHORT" if short else "LONG")
    assert row is not None and row["sessions_since_test"] == 0 and row["test_date"] == bars[-1]["date"]


@pytest.mark.parametrize("short", [False, True])
def test_the_session_after_a_fire_does_not_fire_again(short):
    side = "SHORT" if short else "LONG"
    fired = _extend(_base(), [QUIET, TOUCH, RECLAIM], short=short)
    assert _detect(fired, side=side) is not None
    assert _detect(_extend(fired, [(-0.2, 0.7, 1.0)], short=short), side=side) is None


def test_no_touch_or_no_reclaim_is_no_setup():
    assert _detect(_extend(_base(), [QUIET, QUIET, RECLAIM])) is None
    assert _detect(_extend(_base(), [QUIET, TOUCH, QUIET])) is None


@pytest.mark.parametrize("dates", [None, [], ["2020-01-02"]])
def test_no_earnings_anchor_is_no_row(dates):
    bars = _extend(_base(), [QUIET, TOUCH, RECLAIM])
    assert _detect(bars, earnings_dates=dates) is None


def test_missing_data_is_no_setup():
    bars = _extend(_base(), [QUIET, TOUCH, RECLAIM])
    holed = [dict(bar) for bar in bars]
    holed[-5]["close"] = None
    assert _detect(holed) is None
    assert _detect(bars, side="sideways") is None


def test_is_point_in_time():
    bars = _extend(_base(), [QUIET, TOUCH, RECLAIM])
    later = _extend(bars, [(-3.0, -2.5, 0.0)] * 3)
    assert _detect(later[:len(bars)]) == _detect(bars)


def test_band_series_equals_long_setups_avwap_bands_on_every_prefix():
    bars = _extend(_base(), [QUIET, TOUCH, RECLAIM])
    bars[20]["volume"] = 0  # a zero-volume bar is skipped by both
    bars = ls._clean_bars(bars)
    for start in (0, 4, 30):
        series = aqt.band_series(bars, start)
        assert len(series) == len(bars)
        assert all(value is None for value in series[:start])
        for k in range(start, len(bars)):
            assert series[k] == ls.avwap_bands(bars[:k + 1], start)
    assert aqt.band_series(bars, len(bars)) == [None] * len(bars)


# --- one scan

def _scan(bars=None, *, cap=5000.0, caps=None, as_of=None, **kwargs):
    bars = bars or _extend(_base(), [QUIET, TOUCH, RECLAIM])
    return aqt.build_rows(
        bars_by_symbol={"QT": bars, "SPY": bars, **kwargs.pop("extra", {})},
        spy_bars=bars, earnings_dates_by_symbol={"QT": EARNINGS, "SPY": EARNINGS, "OLD": EARNINGS},
        atr_by_symbol={"QT": 2.0}, market_cap_by_symbol=caps,
        feature_rows=[{"symbol": "QT", "perm_market_cap_m": cap}], as_of=as_of or bars[-1]["date"])


def test_build_rows_fires_both_sides_sorted_and_skips_spy():
    long_bars = _extend(_base(), [QUIET, TOUCH, RECLAIM])
    short_bars = _extend(_base(), [QUIET, TOUCH, RECLAIM], short=True)
    payload = aqt.build_rows(
        bars_by_symbol={"SPY": long_bars, "ZED": long_bars, "ABC": short_bars, "AAA": long_bars},
        spy_bars=long_bars, earnings_dates_by_symbol={"ZED": EARNINGS, "ABC": EARNINGS, "AAA": EARNINGS},
        market_cap_by_symbol={"ZED": 5000.0, "ABC": 5000.0, "AAA": 5000.0}, as_of=long_bars[-1]["date"])
    assert payload["as_of"] == long_bars[-1]["date"]
    assert [(row["side"], row["symbol"]) for row in payload["rows"]] == [("LONG", "AAA"), ("LONG", "ZED"),
                                                                          ("SHORT", "ABC")]
    assert all("promoted" not in row and "points" not in row for row in payload["rows"])


def test_a_stale_name_is_skipped():
    bars = _extend(_base(), [QUIET, TOUCH, RECLAIM])
    assert _scan(bars)["rows"]
    assert _scan(bars[:-1], as_of=bars[-1]["date"])["rows"] == []
    assert aqt.build_rows(bars_by_symbol={"QT": bars}, spy_bars=bars, as_of=None) == {"as_of": "", "rows": []}


@pytest.mark.parametrize("cap, rows", [(5000.0, 1), (1000.0, 1), (999.0, 0), (None, 0)])
def test_the_liquidity_floor(cap, rows):
    assert len(_scan(cap=cap)["rows"]) == rows


def test_the_scan_cap_cache_wins_over_the_row_and_volume_counts():
    assert _scan(cap=None, caps={"QT": 2000.0})["rows"]
    assert _scan(cap=5000.0, caps={"QT": 500.0})["rows"] == []
    thin = [{**bar, "volume": 500_000} for bar in _extend(_base(), [QUIET, TOUCH, RECLAIM])]
    assert _scan(thin)["rows"] == []


# --- grading history

def _graded_bars(closes, *, spread=0.5):
    days = _days(len(closes))
    return [{"date": d, "open": c, "high": c + spread, "low": c - spread, "close": c, "volume": VOLUME}
            for d, c in zip(days, closes, strict=True)]


def test_settle_signs_for_both_sides_and_no_fill():
    bars = _graded_bars([100.0] * 3 + [99.9] + [100.0] * 8 + [110.0] + [100.0] * 2)
    spy = _graded_bars([400.0] * 12 + [404.0] + [400.0] * 2)
    day = bars[2]["date"]
    history = [
        {"symbol": "X", "as_of": day, "side": "LONG", "entry_limit": 99.8},
        {"symbol": "X", "as_of": day, "side": "SHORT", "entry_limit": 100.2},
        {"symbol": "X", "as_of": day, "side": "SHORT", "entry_limit": 120.0},
        {"symbol": "X", "as_of": day, "side": "LONG", "entry_limit": 90.0},
        {"symbol": "X", "as_of": day, "side": "LONG", "entry_limit": 99.8, "outcome": "filled", "return_pct": 7.0},
    ]
    long_row, short_row, short_miss, long_miss, done = aqt.settle(history, {"X": bars}, spy)
    assert long_row["outcome"] == "filled" and long_row["fill"] == 99.8
    assert long_row["target_session"] == bars[12]["date"]
    assert long_row["return_pct"] == pytest.approx((110.0 / 99.8 - 1) * 100, abs=1e-3)
    # A short in the trade's favour is positive; here the close ran 10% against it.
    assert short_row["outcome"] == "filled" and short_row["fill"] == 100.2
    assert short_row["return_pct"] < 0
    assert short_row["return_pct"] == pytest.approx((100.2 / 110.0 - 1) * 100, abs=1e-3)
    assert long_row["spy_return_pct"] == short_row["spy_return_pct"] == pytest.approx(1.0)
    assert short_miss["outcome"] == "no_fill" and long_miss["outcome"] == "no_fill"
    assert "return_pct" not in short_miss
    assert done["return_pct"] == 7.0
    unsettled = aqt.settle(history[:1], {"X": bars[:12]}, spy)
    assert "outcome" not in unsettled[0]


def test_settle_a_short_that_worked_is_positive():
    bars = _graded_bars([100.0] * 12 + [90.0])
    (row,) = aqt.settle([{"symbol": "X", "as_of": bars[2]["date"], "side": "SHORT", "entry_limit": 100.2}],
                        {"X": bars}, bars)
    assert row["return_pct"] > 0
    assert row["return_pct"] == pytest.approx((100.2 / 90.0 - 1) * 100, abs=1e-3)


def test_upsert_first_write_wins_per_session_symbol_and_side():
    old = [{"symbol": "B", "as_of": "2026-09-25", "side": "LONG", "setup": aqt.SETUP, "entry_limit": 10.0,
            "outcome": "filled", "return_pct": 2.0}]
    new = [{"symbol": "B", "as_of": "2026-09-25", "side": "LONG", "setup": aqt.SETUP, "entry_limit": 11.0},
           {"symbol": "B", "as_of": "2026-09-25", "side": "SHORT", "setup": aqt.SETUP, "entry_limit": 12.0,
            "reasons": ["x"], "exit": "y", "test_date": "2026-09-24"}]
    merged = aqt.upsert_history(old, new)
    assert [(row["symbol"], row["side"]) for row in merged] == [("B", "LONG"), ("B", "SHORT")]
    assert merged[0]["entry_limit"] == 10.0 and merged[0]["return_pct"] == 2.0
    assert set(merged[1]) == set(aqt.HISTORY_KEYS) and merged[1]["test_date"] == "2026-09-24"
    assert set(aqt.HISTORY_KEYS) == {"symbol", "as_of", "side", "setup", "close", "avwape", "sigma", "test_date",
                                     "closes_beyond", "atr", "entry_limit", "stop"}


# --- the grade cells (Setup Tracker), both sides

def _hist(side, day, outcome, ret=None, spy=None):
    return {"setup": aqt.SETUP, "side": side, "as_of": day, "outcome": outcome, "return_pct": ret,
            "spy_return_pct": spy}


def test_grade_cells_both_sides_and_the_short_tape_rule():
    import setup_grades

    rows = [
        _hist("LONG", "2026-09-01", "filled", 3.0, 2.0),    # raw win (SPY up 2%), beats SPY
        _hist("LONG", "2026-09-02", "filled", 1.0, 0.0),    # flat SPY: tape only, beats SPY
        _hist("LONG", "2026-09-03", "no_fill"),             # never a loss
        _hist("SHORT", "2026-09-01", "filled", 1.0, -2.0),  # SPY down 2%: raw win; 1.0 < 2.0 = loses the tape
        _hist("SHORT", "2026-09-02", "filled", 1.0, 2.0),   # SPY up: tape only; 1.0 > -2.0 beats it
        _hist("SHORT", "2026-09-03", "filled", -1.0, -3.0),  # raw loss; -1.0 < 3.0 loses the tape
        _hist("SHORT", "2026-09-04", ""),                   # unsettled: left out
        _hist("SHORT", "2026-09-05", "filled", 1.0, None),  # SPY unknown: left out
    ]
    long_cell, short_cell = setup_grades.avwape_quick_test_cells(rows)
    assert (long_cell["family"], long_cell["side"]) == ("avwape_quick_test", "LONG")
    assert (short_cell["family"], short_cell["side"]) == ("avwape_quick_test", "SHORT")
    assert (long_cell["raw"]["n"], long_cell["raw"]["wins"]) == (1, 1)
    assert (long_cell["tape"]["n"], long_cell["tape"]["wins"]) == (2, 2)
    assert long_cell["no_fill"] == 1
    assert (short_cell["raw"]["n"], short_cell["raw"]["wins"]) == (2, 1)
    assert (short_cell["tape"]["n"], short_cell["tape"]["wins"]) == (3, 1)
    assert short_cell["first"] == "2026-09-01" and short_cell["last"] == "2026-09-05"
    long_line = setup_grades.avwape_quick_test_line(long_cell)
    short_line = setup_grades.avwape_quick_test_line(short_cell)
    assert long_line.startswith("avwape_quick_test LONG: raw in SPY-up (>1%) windows")
    assert short_line.startswith("avwape_quick_test SHORT: raw in SPY-down (<-1%) windows")
    assert "vs SPY" in short_line and short_line.endswith("limit not filled 0")
    empty = setup_grades.avwape_quick_test_cells([])
    assert [setup_grades.avwape_quick_test_line(cell) for cell in empty] == [
        "avwape_quick_test LONG: no settled filled rows yet.", "avwape_quick_test SHORT: no settled filled rows yet."]


# --- the words

ROW = {"symbol": "NVDA", "side": "LONG", "entry_limit": 120.1, "exit": "hold up to 10 sessions, stop under 115.20",
       "test_date": "2026-09-30", "closes_beyond": 1}


def test_tracker_lines_for_none_empty_and_rows():
    assert aqt.tracker_lines(None) == ["AVWAPE quick test (testing, both sides): no scan yet."]
    assert aqt.tracker_lines({"as_of": "2026-10-01", "rows": []}) == [
        "AVWAPE quick test (testing, both sides): 0 long, 0 short", "none this scan"]
    short = {**ROW, "symbol": "TSLA", "side": "SHORT", "entry_limit": 250.5, "closes_beyond": 0,
             "exit": "hold up to 10 sessions, stop over 260.00"}
    lines = aqt.tracker_lines({"as_of": "2026-10-01", "rows": [ROW, short]})
    assert lines == [
        "AVWAPE quick test (testing, both sides): 1 long, 1 short",
        "NVDA LONG | buy limit 120.10 | hold up to 10 sessions, stop under 115.20 | tested LOWER_1 2026-09-30 "
        "(1 close under), reclaimed AVWAPE",
        "TSLA SHORT | sell limit 250.50 | hold up to 10 sessions, stop over 260.00 | tested UPPER_1 2026-09-30 "
        "(0 closes over), lost AVWAPE",
    ]
    many = aqt.tracker_lines({"rows": [ROW] * 14}, limit=12)
    assert len(many) == 14 and many[-1] == "(+2 more)"


# --- the store

def _publish(store, tmp_path, bars_by_symbol, as_of, dates_path):
    return store.publish_avwape_quick_test(
        bars_by_symbol=bars_by_symbol, spy_bars=next(iter(bars_by_symbol.values())),
        feature_rows=[{"symbol": "QT", "perm_market_cap_m": 5000.0}], atr_by_symbol={"QT": 2.0},
        as_of=as_of, path=tmp_path / "aqt.json", history_path=tmp_path / "hist.json",
        earnings_dates_path=dates_path)


def test_the_store_publishes_and_keeps_the_last_good_file(tmp_path):
    import avwape_quick_test_store as store

    dates = tmp_path / "earnings_dates.json"
    dates.write_text(json.dumps({"symbols": {"QT": {"dates": EARNINGS}}}), encoding="utf-8")
    bars = _extend(_base(), [QUIET, TOUCH, RECLAIM])
    got = _publish(store, tmp_path, {"QT": bars}, bars[-1]["date"], dates)
    target = tmp_path / "aqt.json"
    good = target.read_bytes()
    assert json.loads(good)["schema_version"] == store.SCHEMA_VERSION == 1
    assert [row["symbol"] for row in got["rows"]] == ["QT"]
    assert store.read_avwape_quick_test(target)["rows"] == got["rows"]
    history = store.read_history(tmp_path / "hist.json")
    assert [(row["symbol"], row["side"]) for row in history] == [("QT", "LONG")]
    # A cutoff no name has a bar for never wipes the last good file.
    kept = _publish(store, tmp_path, {"QT": bars}, "2099-01-05", dates)
    assert target.read_bytes() == good and kept["rows"]
    # A real empty session (the name is current, no setup) still publishes.
    calm = _base()
    _publish(store, tmp_path, {"QT": calm}, calm[-1]["date"], dates)
    assert json.loads(target.read_text(encoding="utf-8"))["rows"] == []
    # No session writes nothing.
    assert _publish(store, tmp_path, {"QT": calm}, None, dates) == {"as_of": "", "rows": []}


def test_the_readers_tolerate_missing_and_bad_files(tmp_path):
    import avwape_quick_test_store as store

    assert store.read_avwape_quick_test(tmp_path / "none.json") is None
    assert store.read_history(tmp_path / "none.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert store.read_avwape_quick_test(bad) is None and store.read_history(bad) == []


def test_the_files_live_next_to_the_long_setups():
    import project_paths

    assert project_paths.AVWAPE_QUICK_TEST_FILE.parent == project_paths.LONG_SETUPS_FILE.parent
    assert project_paths.AVWAPE_QUICK_TEST_FILE.name == "avwape_quick_test.json"
    assert project_paths.AVWAPE_QUICK_TEST_HISTORY_FILE.name == "avwape_quick_test_history.json"


# --- the Setup Tracker section (display only)

def test_the_worker_reader_builds_the_section_from_the_files(tmp_path, monkeypatch):
    import project_paths
    from diagnostics.artifact_io import atomic_write_json
    from ui.services import working_lately_service as service

    current, history = tmp_path / "aqt.json", tmp_path / "aqt_history.json"
    atomic_write_json(current, {"schema_version": 1, "as_of": "2026-10-01", "rows": [ROW]})
    atomic_write_json(history, {"rows": [_hist("SHORT", "2026-09-01", "filled", 1.0, -2.0)]})
    monkeypatch.setattr(project_paths, "AVWAPE_QUICK_TEST_FILE", current)
    monkeypatch.setattr(project_paths, "AVWAPE_QUICK_TEST_HISTORY_FILE", history)
    service._LOOKING_BACK_CACHE.clear()
    try:
        lines = service.read_avwape_quick_test_lines()
    finally:
        service._LOOKING_BACK_CACHE.clear()
    assert lines[0] == "AVWAPE quick test (testing, both sides): 1 long, 0 short"
    assert lines[1].startswith("NVDA LONG | buy limit 120.10")
    assert lines[2] == "avwape_quick_test LONG: no settled filled rows yet."
    assert lines[3].startswith("avwape_quick_test SHORT: raw in SPY-down")


def test_the_tracker_worker_carries_the_section_and_survives_a_failure(monkeypatch):
    from ui.panels import setup_tracker_panel as module
    from ui.services import working_lately_service

    monkeypatch.setattr(working_lately_service, "read_avwape_quick_test_lines", lambda: ["a", "b"])
    assert module._read_tracker_exports(1)["avwape_quick_test_lines"] == ["a", "b"]

    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(working_lately_service, "read_avwape_quick_test_lines", _boom)
    exports = module._read_tracker_exports(1)
    assert exports["avwape_quick_test_lines"] == ["AVWAPE quick test: unreadable right now."]
    assert exports["long_leader_lines"]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.qt
def test_the_section_sits_right_under_the_long_leaders(qapp):
    from ui.panels import setup_tracker_panel as module

    panel = module.SetupTrackerPanel()
    try:
        panel._on_exports_loaded({"signatures": {}, "ranked": {}, "raw": {}, "long_leader_lines": ["x"],
                                  "avwape_quick_test_lines": ["q", "r"]})
        assert panel.avwape_quick_test_label.text() == "q\nr"
        layout = panel.layout()
        widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
        assert widgets.index(panel.avwape_quick_test_label) == widgets.index(panel.long_leaders_label) + 1
    finally:
        panel.shutdown()
        panel.deleteLater()


# --- never read by a detector, a score or the review policy

def test_nothing_live_reads_the_quick_test():
    root = SCRIPTS_DIR
    allowed = {"avwape_quick_test.py", "avwape_quick_test_store.py", "setup_grades.py",
               "working_lately_service.py", "runner.py", "project_paths.py"}
    readers = sorted(path.name for path in root.rglob("*.py")
                     if "avwape_quick_test" in path.read_text(encoding="utf-8", errors="replace")
                     and path.name not in allowed)
    assert readers == ["setup_tracker_panel.py"]
    policy = root.parent / "config" / "review_policy.json"
    if policy.is_file():
        assert "avwape_quick_test" not in policy.read_text(encoding="utf-8")


# --- the scan: the real `runner._run_master_impl` in a child process with a scratch home

_RUNNER_HOOK = r'''
# The AVWAPE quick test publish raises (it records the call first) when asked to.
if os.environ.get("AVWAPE_QUICK_TEST_HOOK") == "raise":
    def _raise_quick_test(*args, **kwargs):
        (SCRATCH / "quick_test_called").write_text("yes", encoding="utf-8")
        raise RuntimeError("quick test boom")
    runner.publish_avwape_quick_test = _raise_quick_test
'''

_RUNNER_DUMP = r'''
quick_path = getattr(project_paths, "AVWAPE_QUICK_TEST_FILE", None)
print("QUICK-RESULT::" + json.dumps({
    "quick_test": json.loads(_read(quick_path)) if quick_path and _read(quick_path) else None,
    "called": (SCRATCH / "quick_test_called").is_file(),
}, default=str))
'''


def _scan_run(tmp_path: Path, mode: str) -> dict:
    import os
    import subprocess

    import test_long_setups_scan_parity as parity

    scratch = tmp_path / f"quick-{mode}"
    for name in ("home", "localappdata", "diag"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    source = parity._child_source()
    anchor = "\n\ndef _sessions(count):"
    assert anchor in source
    child = scratch / "child.py"
    child.write_text(source.replace(anchor, "\n" + _RUNNER_HOOK + anchor, 1) + _RUNNER_DUMP, encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["LONG_SETUPS_HOOK"] = "on"
    environment["AVWAPE_QUICK_TEST_HOOK"] = mode
    completed = subprocess.run(
        [sys.executable, str(child), str(scratch), str(SCRIPTS_DIR), "on", "300", "", "leader_pullback"],
        capture_output=True, text=True, timeout=900, env=environment, cwd=str(SCRIPTS_DIR),
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    out: dict = {}
    for marker in ("LONG-RESULT::", "QUICK-RESULT::"):
        line = next((text[len(marker):] for text in reversed(completed.stdout.splitlines())
                     if text.startswith(marker)), "")
        assert line, completed.stdout[-4000:]
        out.update(json.loads(line))
    return out


@pytest.fixture(scope="module")
def scan_runs(tmp_path_factory):
    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    folder = tmp_path_factory.mktemp("avwape-quick-test-scan")
    return _scan_run(folder, "on"), _scan_run(folder, "raise")


def test_the_scan_publishes_the_quick_test_file(scan_runs):
    on, _boom = scan_runs
    assert on["quick_test"] is not None, "the scan did not publish the AVWAPE quick test file"
    assert on["quick_test"]["schema_version"] == 1 and on["quick_test"]["as_of"]
    assert on["long_setups"] is not None


def test_a_failing_quick_test_publish_never_fails_the_scan(scan_runs):
    on, boom = scan_runs
    assert boom["called"] is True, "the scan never called the quick test publish"
    assert boom["quick_test"] is None
    # The scan finished, and the long setups were still published, with the same rows.
    assert boom["long_setups"] is not None and boom["priority_report"]
    assert [row["symbol"] for row in boom["long_setups"]["rows"]] == \
        [row["symbol"] for row in on["long_setups"]["rows"]]
