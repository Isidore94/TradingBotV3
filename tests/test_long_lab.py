"""Long lab: point-in-time long-rule replay on synthetic daily bars (shadow research)."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_warehouse import long_lab as lab  # noqa: E402

START = date(2024, 1, 1)


def _days(count: int, start: date = START) -> list[date]:
    out, day = [], start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _rows(closes, *, days=None, gaps=None, volume=1_000_000.0, volumes=None, unit="shares", spread=0.01):
    """Bars from a close path: open = prior close (or prior close + a gap), 1% range."""
    days = days or _days(len(closes))
    gaps = gaps or {}
    rows, prev = [], float(closes[0])
    for i, close in enumerate(closes):
        close = float(close)
        open_ = prev + gaps.get(i, 0.0)
        high = max(open_, close) * (1 + spread)
        low = min(open_, close) * (1 - spread)
        vol = volumes[i] if volumes is not None else volume
        rows.append({"date": days[i], "open": open_, "high": high, "low": low, "close": close,
                     "volume": vol, "volume_unit": unit})
        prev = close
    return rows


def _series(symbol, closes, **kw) -> lab.Series:
    return lab.Series.from_rows(symbol, _rows(closes, **kw))


# ---------------------------------------------------------------- point in time
def _universe(n_bars=460, seed=7):
    rng = np.random.default_rng(seed)
    days = _days(n_bars)
    series, earnings = {}, {}
    spy = 400 * np.cumprod(1 + rng.normal(0.0004, 0.009, n_bars))
    series["SPY"] = lab.Series.from_rows("SPY", _rows(spy, days=days))
    for k in range(14):
        drift = rng.uniform(-0.0005, 0.0025)
        path = 50 * np.cumprod(1 + rng.normal(drift, 0.02, n_bars))
        gaps, vols, events = {}, [1e6] * n_bars, []
        for g in range(260 + k * 3, n_bars - 10, 63):
            gaps[g] = path[g - 1] * 0.08  # a ~4 ATR gap up on 3x volume
            path[g:] *= 1.08
            vols[g] = 3e6
            events.append(days[g])
        closes = path
        symbol = f"S{k:02d}"
        series[symbol] = lab.Series.from_rows(symbol, _rows(closes, days=days, gaps=gaps, volumes=vols))
        earnings[symbol] = events
    return series, earnings


def _keys(candidates, cut):
    return sorted(
        (c["rule"], c["symbol"], c["date"],
         tuple(sorted((k, round(v, 9) if isinstance(v, float) else v) for k, v in c["features"].items())))
        for c in candidates if c["date"] <= cut
    )


def _truncate(series, cut, *, wild=False):
    """Bars up to ``cut`` unchanged; later bars dropped, or (``wild``) crashed 70% on 10x volume."""
    out = {}
    for sym, s in series.items():
        rows = []
        for i, d in enumerate(s.dates):
            later = d > cut
            if later and not wild:
                continue
            factor = 0.3 if later else 1.0
            rows.append({"date": d, "open": s.open[i] * factor, "high": s.high[i] * factor,
                         "low": s.low[i] * factor, "close": s.close[i] * factor,
                         "volume": s.volume[i] * (10 if later else 1), "volume_unit": s.volume_unit[i]})
        out[sym] = lab.Series.from_rows(sym, rows)
    return out


def pit_mismatches(series, earnings, sessions, full, cuts, rules=lab.DEFAULT_RULES):
    """The cuts at which the flags up to the cut change once later bars are gone or rewritten."""
    bad = []
    for cut in cuts:
        past = _keys(full, cut)
        alone = lab.find_candidates(_truncate(series, cut), [d for d in sessions if d <= cut],
                                    rules=rules, earnings=earnings)
        wild = lab.find_candidates(_truncate(series, cut, wild=True), sessions, rules=rules, earnings=earnings)
        if _keys(alone, cut) != past or _keys(wild, cut) != past:
            bad.append(cut)
    return bad


def test_a_future_bar_never_changes_a_past_flag():
    series, earnings = _universe()
    sessions = series["SPY"].dates[-200:]
    full = lab.find_candidates(series, sessions, earnings=earnings)
    rules_seen = {c["rule"] for c in full if c["date"] <= sessions[120]}
    assert {"leader_pullback_loose", "leader_pullback", "rising_20_50_baseline", "post_earnings_drift",
            "gap_volume_drift_proxy", "favourite_zone_long"} <= rules_seen, rules_seen
    # Many flag days are cuts: a leak into a flag shows at that flag's own cut.
    cuts = sorted({c["date"] for c in full})[::3]
    assert pit_mismatches(series, earnings, sessions, full, cuts) == []


# ---------------------------------------------------------------- rules
def _leader_path(run_to: float):
    flat = [50.0 + 0.2 * np.sin(i / 3) for i in range(260)]
    up = list(np.linspace(50.0, run_to, 31)[1:])  # 30 sessions up
    down = list(np.linspace(run_to, run_to * 0.84, 15)[1:])  # back 16% over 14 sessions
    return flat + up + down


def test_leader_pullback_flags_a_leader_back_at_its_ema_and_not_a_laggard():
    s = _series("LEAD", _leader_path(70.0))  # a 40% run in 30 sessions
    flags = [(i, lab.rule_leader_pullback(lab.Day(s, i, None, {}))) for i in range(len(s.dates))]
    hits = [(i, f) for i, f in flags if f]
    assert hits, "the textbook leader pullback never flagged"
    assert all(i >= 290 for i, _f in hits)  # never in the flat base or the run
    assert all(0.08 <= f["depth"] <= 0.25 and f["run"] >= 0.30 for _i, f in hits)

    lag = _series("LAG", _leader_path(56.0))  # a 12% run: not a leader
    assert not any(lab.rule_leader_pullback(lab.Day(lag, i, None, {})) for i in range(len(lag.dates)))


def test_post_earnings_drift_flags_once_on_the_fourth_session_and_needs_the_gap_open():
    closes = [50.0] * 60 + [54.0] * 20
    gaps = {60: 3.5}  # open 53.5 vs 50 close, far over 1 ATR
    s = _series("PED", closes, gaps=gaps)
    earn = {"PED": [s.dates[60]]}
    hits = [i for i in range(len(s.dates)) if lab.rule_post_earnings_drift(lab.Day(s, i, None, earn))]
    assert hits == [64]
    feats = lab.rule_post_earnings_drift(lab.Day(s, 64, None, earn))
    assert feats["sessions_since_gap"] == 4 and feats["source"] == "earnings_dates"

    filled = _series("PEF", [50.0] * 60 + [54.0] * 3 + [49.0] * 17, gaps=gaps)
    assert not any(lab.rule_post_earnings_drift(lab.Day(filled, i, None, {"PEF": [filled.dates[60]]}))
                   for i in range(len(filled.dates)))


def test_gap_volume_proxy_needs_two_times_volume_in_the_same_unit():
    closes = [50.0] * 60 + [54.0] * 20
    gaps = {60: 3.5}
    vols = [1e6] * 80
    vols[60] = 2.5e6
    s = _series("PX", closes, gaps=gaps, volumes=vols)
    hits = [i for i in range(len(s.dates)) if lab.rule_gap_volume_drift_proxy(lab.Day(s, i, None, {}))]
    assert hits == [64]

    quiet = list(vols)
    quiet[60] = 1.5e6
    s2 = _series("PQ", closes, gaps=gaps, volumes=quiet)
    assert not any(lab.rule_gap_volume_drift_proxy(lab.Day(s2, i, None, {})) for i in range(80))

    # Lots before the gap, shares on it: never compared, so unknown -> no flag.
    rows = _rows(closes, gaps=gaps, volumes=vols)
    for row in rows[:60]:
        row["volume_unit"] = "unknown"
    s3 = lab.Series.from_rows("PU", rows)
    assert not any(lab.rule_gap_volume_drift_proxy(lab.Day(s3, i, None, {})) for i in range(80))


def test_baseline_needs_a_rising_20_and_50():
    up = _series("UP", list(np.linspace(40, 60, 80)))
    down = _series("DN", list(np.linspace(60, 40, 80)))
    assert lab.rule_rising_20_50_baseline(lab.Day(up, 79, None, {})) == {}
    assert lab.rule_rising_20_50_baseline(lab.Day(down, 79, None, {})) is None
    assert lab.rule_rising_20_50_baseline(lab.Day(up, 40, None, {})) is None  # no 50-day yet: unknown


def test_anchored_vwap_matches_the_champion_formula():
    from master_avwap_lib.legacy import calc_anchored_vwap_bands
    import pandas as pd

    rng = np.random.default_rng(3)
    closes = 30 * np.cumprod(1 + rng.normal(0, 0.02, 60))
    vols = list(rng.integers(0, 5, 60) * 1e5)  # some zero-volume bars too
    s = _series("AV", closes, volumes=vols)
    frame = pd.DataFrame({"open": s.open, "high": s.high, "low": s.low, "close": s.close, "volume": s.volume})
    for anchor in (0, 5, 17):
        for end in (anchor + 3, 40, 59):
            vwap, sigma, _bands = calc_anchored_vwap_bands(frame.iloc[: end + 1].reset_index(drop=True), anchor)
            got = lab.anchored_vwap_bands(s, anchor, end)
            if np.isnan(vwap):
                assert got is None
                continue
            assert got[0] == pytest.approx(vwap, abs=1e-9) and got[1] == pytest.approx(sigma, abs=1e-9)


# ---------------------------------------------------------------- outcomes
def _flat_bars(n, price=100.0):
    return [{"date": d, "open": price, "high": price + 1, "low": price - 1, "close": price,
             "volume": 1e6, "volume_unit": "shares"} for d in _days(n)]


def test_measure_entry_returns_vs_spy_excursions_limit_and_exits():
    rows = _flat_bars(45)  # ATR(14) = 2.0 on the flag bar
    i = 20
    rows[i + 1].update(open=100.0, high=101.0, low=99.4, close=100.0)  # limit 99.5 fills at 99.5
    rows[i + 2].update(open=100.0, high=102.5, low=99.5, close=102.0)  # +1 ATR take (102) hit
    for j in range(i + 3, i + 6):
        rows[j].update(open=102.0, high=103.0, low=101.5, close=102.0)
    rows[i + 6].update(open=102.0, high=102.0, low=99.0, close=100.0)
    s = lab.Series.from_rows("M", rows)
    spy = lab.Series.from_rows("SPY", _flat_bars(45, price=400.0))
    assert s.atr[i] == pytest.approx(2.0)
    out = lab.measure(s, i, spy)
    assert out["entry"] == 100.0
    h5 = out["h5"]
    assert h5["raw"] == pytest.approx(0.02)
    assert h5["vs_spy"] == pytest.approx(0.02)
    assert h5["mfe_atr"] == pytest.approx(1.5)  # high 103 vs 100 over ATR 2
    assert h5["mae_atr"] == pytest.approx(0.3)  # low 99.4
    assert out["limit"]["filled"] is True and out["limit"]["fill"] == pytest.approx(99.5)
    assert out["exits"]["take_1atr"] == pytest.approx(1.0)
    assert out["exits"]["time_10"] == pytest.approx(0.0)  # back to 100 by session 10
    # Trail: stop rises to 103 - 2 = 101 after bar i+3; bar i+6 (open 102, low 99) hits it.
    assert out["exits"]["trail_1atr"] == pytest.approx(0.5)


def test_an_unfinished_horizon_is_pending_never_zero():
    s = lab.Series.from_rows("P", _flat_bars(28))
    spy = lab.Series.from_rows("SPY", _flat_bars(28, price=400.0))
    out = lab.measure(s, 20, spy)
    assert "h5" in out and "h10" not in out and "h20" not in out
    assert out["exits"] == {}


def test_a_gap_open_fills_the_limit_at_the_open():
    rows = _flat_bars(30)
    rows[21].update(open=98.0, high=99.0, low=97.5, close=98.5)
    s = lab.Series.from_rows("G", rows)
    out = lab.measure(s, 20, lab.Series.from_rows("SPY", _flat_bars(30, 400.0)))
    assert out["limit"]["fill"] == pytest.approx(98.0)


# ---------------------------------------------------------------- report
def test_report_is_per_regime_never_pooled_and_carries_the_sweep():
    series, earnings = _universe()
    sessions = series["SPY"].dates[-200:]
    candidates = lab.find_candidates(series, sessions, earnings=earnings)
    split = sessions[100]
    segments = [{"start_date": split.isoformat(), "regime": "bull_trend"}]
    report = lab.build_report(candidates, series, sessions, segments=segments)
    json.dumps(report, default=str)
    assert report["schema"] == lab.SCHEMA
    regimes = {c["regime"] for c in report["cells"] if c["axis"] == "structural"}
    assert regimes == {"unknown", "bull_trend"}
    assert not any(c["regime"] in ("all", "pooled", "all regimes") for c in report["cells"])
    # Each regime cell is its own rows: the per-regime counts add up to the rule's total.
    for rule in {c["rule"] for c in report["cells"]}:
        total = sum(1 for c in candidates if c["rule"] == rule and "h5" in c["outcome"])
        for axis in lab.AXES:
            cells = [c for c in report["cells"] if c["rule"] == rule and c["axis"] == axis and c["horizon"] == 5]
            assert sum(c["n"] for c in cells) == total
    assert any("Survivorship" in text for text in report["caveats"])
    knobs = {c["knob"] for c in report["sweep"]}
    assert {"pullback_depth", "run_size", "rs_decile", "made_52w_high"} <= knobs
    for cell in report["cells"]:
        if cell["thin"]:
            assert cell["n"] < lab.MIN_CELL_N
    for row in report["exits"]:
        if row["best"]:
            assert row["n"] >= lab.MIN_CELL_N


def test_spy_trend_labels():
    spy = lab.Series.from_rows("SPY", _rows(list(np.linspace(100, 130, 40)) + list(np.linspace(130, 110, 20))))
    labels = lab.spy_trend_labels(spy)
    assert labels[spy.dates[10]] == "unknown"
    assert labels[spy.dates[35]] == "above_rising_20d"
    assert labels[spy.dates[59]] == "below_20d"


def test_inputs_read_from_files_and_the_report_round_trips(tmp_path):
    bars = tmp_path / "bars"
    bars.mkdir()
    (bars / "abc.csv").write_text(
        "datetime,open,high,low,close,volume,source,volume_unit\n"
        "2026-01-02,10,11,9,10.5,100,yahoo,shares\n"
        "garbage,row\n"
        "2026-01-05,10.5,12,10,11,200,yahoo,shares\n", encoding="utf-8")
    loaded = lab.load_bars_dir(bars)
    assert list(loaded) == ["ABC"] and len(loaded["ABC"].dates) == 2
    earn = tmp_path / "e.json"
    earn.write_text(json.dumps({"symbols": {"abc": {"dates": ["2026-01-05", "bad"]}}}), encoding="utf-8")
    assert lab.load_earnings_dates(earn) == {"ABC": [date(2026, 1, 5)]}
    assert lab.load_earnings_dates(tmp_path / "missing.json") == {}
    out = tmp_path / "r.json"
    lab.write_report({"schema": lab.SCHEMA, "cells": []}, out)
    assert lab.read_report(out) == {"schema": lab.SCHEMA, "cells": []}
    assert lab.read_report(tmp_path / "none.json") is None


def test_no_spy_is_an_error_report_not_a_crash():
    report = lab.run_lab({"X": lab.Series.from_rows("X", _flat_bars(30))})
    assert report["error"] == "no SPY bars" and report["cells"] == []
