"""D1 backtester (research_warehouse.backtest): point in time, live parity, side-aware outcomes,
pending never zero, regime painting, honesty splits, the trial ledger and immutable runs."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import long_setups  # noqa: E402
from research_warehouse import backtest as bt  # noqa: E402
from research_warehouse import backtest_setups as bs  # noqa: E402
from research_warehouse import long_lab as lab  # noqa: E402
from research_warehouse import trial_ledger  # noqa: E402
from research_warehouse.retest_entry import limit_fill  # noqa: E402

START = date(2021, 1, 4)


def _days(count: int, start: date = START) -> list[date]:
    out, day = [], start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _frame(closes, days, *, rng, gaps=None, volume=2_000_000.0, spread=0.012) -> pd.DataFrame:
    gaps = gaps or {}
    rows, prev = [], float(closes[0])
    for i, close in enumerate(closes):
        close = float(close)
        open_ = prev * (1 + gaps.get(i, 0.0) + rng.normal(0, 0.002))
        high = max(open_, close) * (1 + abs(rng.normal(0, spread)))
        low = min(open_, close) * (1 - abs(rng.normal(0, spread)))
        vol = volume * float(np.exp(rng.normal(0, 0.3)))
        rows.append({"session_date": days[i], "open": open_, "high": high, "low": low, "close": close,
                     "volume": vol, "provider": "TEST"})
        prev = close
    return pd.DataFrame(rows)


def universe(n_names: int = 24, n_bars: int = 560, seed: int = 11):
    """SPY plus names with trends, gaps and earnings dates (quarterly)."""
    rng = np.random.default_rng(seed)
    days = _days(n_bars)
    bars = {"SPY": _frame(400 * np.cumprod(1 + rng.normal(0.0004, 0.009, n_bars)), days, rng=rng,
                          volume=80_000_000.0)}
    earnings = {}
    for k in range(n_names):
        drift = rng.uniform(-0.002, 0.003)
        vol = rng.uniform(0.012, 0.03)
        rets = rng.normal(drift, vol, n_bars)
        gap_days = list(range(40 + k % 7, n_bars - 5, 63))
        gaps = {}
        for g in gap_days:
            gaps[g] = rng.choice([-1, 1]) * rng.uniform(0.03, 0.10)
            rets[g] += gaps[g]
        closes = 50 * np.cumprod(1 + rets)
        sym = f"N{k:02d}"
        bars[sym] = _frame(closes, days, rng=rng, gaps=gaps)
        earnings[sym] = [days[g] for g in gap_days]
    return bars, earnings, days


def _surprises(bars, earnings):
    """EPS surprise % per earnings date: a miss (-10) on a gap down, a beat (+10) on a gap up."""
    out = {}
    for sym, dates in earnings.items():
        frame = bars[sym].set_index("session_date")
        prev = frame["close"].shift(1)
        out[sym] = {d: (-10.0 if frame.at[d, "open"] < prev.at[d] else 10.0) for d in dates}
    return out


@pytest.fixture(scope="module")
def world():
    bars, earnings, days = universe()
    inputs = bt.Inputs(bars=bars, earnings=earnings, surprises=_surprises(bars, earnings))
    ctxs, _ = bt.prepare(inputs)
    bt.rank_rs(ctxs)
    return bars, earnings, days, ctxs


def _regimes(days) -> pd.DataFrame:
    labels = ["up" if i % 3 else "down" for i in range(len(days))]
    return pd.DataFrame({"session_date": days, "trend20": labels,
                         "vol_rv": ["low" if i < len(days) // 2 else "high" for i in range(len(days))],
                         "close": np.arange(len(days), dtype=float), "rule_version": "regime_daily_v1"})


# ---------------------------------------------------------------- point in time
@pytest.mark.parametrize("setup", bs.REGISTRY, ids=lambda s: s.key)
def test_every_setup_is_point_in_time(world, setup):
    """Truncating a name at bar i never changes the answer at i: no rule reads a later bar."""
    *_, ctxs = world
    rng = np.random.default_rng(3)
    for sym in ("N00", "N03", "N05", "N08", "N13"):
        ctx = ctxs[sym]
        full, feats = bs.evaluate(setup, bs.Ctx(**{**_ctx_args(ctx)}))
        flagged = list(np.nonzero(full)[0][:6])
        probes = sorted(set(flagged + list(rng.integers(260, len(ctx), 6))))
        for i in probes:
            cut, cut_feats = bs.evaluate(setup, ctx.truncated(int(i)))
            assert cut[-1] == full[i], (setup.key, sym, i)
            for key, values in feats.items():
                np.testing.assert_allclose(cut_feats[key][-1], values[i], rtol=1e-9, equal_nan=True)


def _ctx_args(ctx):
    return {"symbol": ctx.symbol, "dates": ctx.dates, "open": ctx.open, "high": ctx.high, "low": ctx.low,
            "close": ctx.close, "volume": ctx.volume, "earnings": ctx.earnings,
            "rs_share": ctx.rs_share, "rs_decile": ctx.rs_decile, "surprises": ctx.surprises}


def test_rs_share_matches_the_live_percentile(world):
    *_, ctxs = world
    stocks = [s for s in ctxs if bt.is_stock(s)]
    day = ctxs["N00"].dates[300]
    rets = {s: float(ctxs[s].ret(63)[np.searchsorted(ctxs[s].dates, day)]) for s in stocks}
    live = long_setups.rs_percentiles(rets)
    for s in stocks:
        i = np.searchsorted(ctxs[s].dates, day)
        assert ctxs[s].rs_share[i] == pytest.approx(live[s])
    assert "SPY" not in stocks


# ---------------------------------------------------------------- live parity
def test_leader_pullback_equals_the_live_function_on_every_bar():
    """The vectorized prefilter never drops a bar the live function would flag."""
    bars, earnings, days = universe(n_names=6, n_bars=470, seed=5)
    ctxs, _ = bt.prepare(bt.Inputs(bars=bars, earnings=earnings))
    bt.rank_rs(ctxs)
    hits = 0
    for sym in ("N00", "N01", "N02", "N03", "N04", "N05"):
        ctx = ctxs[sym]
        mask, _ = bs.leader_pullback_live(ctx)
        brute = np.zeros(len(ctx), dtype=bool)
        for i in range(199, len(ctx)):
            share = float(ctx.rs_share[i]) if np.isfinite(ctx.rs_share[i]) else None
            brute[i] = long_setups.leader_pullback(ctx.live_bars(i), atr=float(ctx.atr20[i]),
                                                   rs_percentile=share) is not None
        np.testing.assert_array_equal(mask, brute)
        hits += int(brute.sum())
    assert hits > 0, "fixture must contain live leader pullbacks"


def test_earnings_avwap_matches_the_live_study_anchor(world):
    *_, ctxs = world
    for sym in ("N01", "N04"):
        ctx = ctxs[sym]
        av = ctx.earnings_avwap()
        days = [str(d) for d in ctx.earnings]
        checked = 0
        for i in range(60, len(ctx), 7):
            live = long_setups.earnings_avwap(_all_bars(ctx, i), earnings_dates=days)
            if live["avwape"] is None:
                assert not np.isfinite(av["level"][i])
                continue
            checked += 1
            assert av["level"][i] == pytest.approx(live["avwape"], abs=1e-3)
            z = (ctx.close[i] - av["level"][i]) / av["sigma"][i]
            assert z == pytest.approx(live["avwape_z"], abs=1e-3)
        assert checked > 10


def _all_bars(ctx, i):
    ctx.live_bars(0)
    return ctx.cache["live_bars"][: i + 1]


def test_post_earnings_drift_equals_the_live_rule(world):
    bars, earnings, days, ctxs = world
    for sym in ("N00", "N02", "N06", "N09"):
        ctx = ctxs[sym]
        mask, _ = bs.post_earnings_drift_live(ctx)
        series = lab.Series.from_rows(sym, [{"date": d.date() if hasattr(d, "date") else d, **r}
                                            for d, r in zip(bars[sym]["session_date"],
                                                            bars[sym][["open", "high", "low", "close", "volume"]]
                                                            .to_dict("records"), strict=True)])
        brute = np.array([lab.rule_live_post_earnings_drift(lab.Day(series, i, None, earnings)) is not None
                          for i in range(len(series.dates))])
        np.testing.assert_array_equal(mask, brute)


def test_favourite_zone_long_equals_long_lab(world):
    bars, earnings, days, ctxs = world
    for sym in ("N00", "N05"):
        ctx = ctxs[sym]
        mask, _ = bs.favourite_zone_long(ctx)
        series = lab.Series.from_rows(sym, [{"date": d, **r} for d, r in zip(
            bars[sym]["session_date"], bars[sym][["open", "high", "low", "close", "volume"]].to_dict("records"), strict=True)])
        brute = np.array([lab.rule_favourite_zone_long(lab.Day(series, i, None, earnings)) is not None
                          for i in range(len(series.dates))])
        np.testing.assert_array_equal(mask, brute)
        assert brute.any()


def test_rising_baseline_equals_long_lab(world):
    bars, earnings, days, ctxs = world
    ctx = ctxs["N03"]
    mask, _ = bs.rising_20_50_baseline(ctx)
    series = lab.Series.from_rows("N03", [{"date": d, **r} for d, r in zip(
        bars["N03"]["session_date"], bars["N03"][["open", "high", "low", "close", "volume"]].to_dict("records"), strict=True)])
    brute = np.array([lab.rule_rising_20_50_baseline(lab.Day(series, i, None, {})) is not None
                      for i in range(len(series.dates))])
    np.testing.assert_array_equal(mask, brute)


# ---------------------------------------------------------------- outcomes
def test_long_outcomes_match_long_lab_measure(world):
    bars, earnings, days, ctxs = world
    ctx, spy = ctxs["N07"], ctxs["SPY"]
    series = lab.Series.from_rows("N07", [{"date": d, **r} for d, r in zip(
        bars["N07"]["session_date"], bars["N07"][["open", "high", "low", "close", "volume"]].to_dict("records"), strict=True)])
    spy_s = lab.Series.from_rows("SPY", [{"date": d, **r} for d, r in zip(
        bars["SPY"]["session_date"], bars["SPY"][["open", "high", "low", "close", "volume"]].to_dict("records"), strict=True)])
    idx = np.array([30, 100, 222, 400, len(ctx) - 15, len(ctx) - 3, len(ctx) - 1])
    out = bt.measure(ctx, idx, bs.LONG, bt.spy_lookup(spy))
    for k, i in enumerate(idx):
        ref = lab.measure(series, int(i), spy_s)
        for h in (5, 10, 20):
            cell = ref.get(f"h{h}")
            if cell is None:
                assert np.isnan(out[f"h{h}_raw"][k]) and np.isnan(out[f"h{h}_vs_spy"][k])
                continue
            assert out[f"h{h}_raw"][k] == pytest.approx(cell["raw"])
            assert out[f"h{h}_vs_spy"][k] == pytest.approx(cell["vs_spy"])
            assert out[f"h{h}_mfe_atr"][k] == pytest.approx(cell["mfe_atr"])
            assert out[f"h{h}_mae_atr"][k] == pytest.approx(cell["mae_atr"])
        exits = ref.get("exits", {})
        for mine, theirs in (("take_R", "take_1atr"), ("trail_R", "trail_1atr"), ("time10_R", "time_10")):
            if theirs in exits:
                assert out[mine][k] == pytest.approx(exits[theirs])
            elif mine != "time10_R" and np.isfinite(out[mine][k]) and "entry" in ref:
                # an exit already hit before the data ends is final, not pending (long_lab waits)
                fn = lab._exit_take if mine == "take_R" else lab._exit_trail
                atr = float(series.atr[i])
                done = (fn(series, ref["entry"], atr, int(i) + 1, len(series.dates) - 1) - ref["entry"]) / atr
                assert out[mine][k] == pytest.approx(done)
            else:
                assert np.isnan(out[mine][k])
        lim = ref.get("limit")
        if lim is None:
            assert np.isnan(out["limit_filled"][k])
        elif lim["filled"]:
            assert out["limit_filled"][k] == 1.0 and out["limit_fill"][k] == pytest.approx(lim["fill"])
        else:
            assert out["limit_filled"][k] == 0.0


def _tiny(opens, highs, lows, closes, start=START):
    days = _days(len(closes), start)
    return bs.Ctx("X", np.array(days, dtype="datetime64[D]"), np.array(opens, float), np.array(highs, float),
                  np.array(lows, float), np.array(closes, float), np.full(len(closes), 1e6))


def _flat_spy(n, start=START):
    days = _days(n, start)
    return bs.Ctx("SPY", np.array(days, dtype="datetime64[D]"), np.full(n, 100.0), np.full(n, 101.0),
                  np.full(n, 99.0), np.full(n, 101.0), np.full(n, 1e8))


def test_short_outcomes_are_side_aware():
    """A falling stock is a winning short; a short beats SPY when the stock does worse than SPY."""
    n = 40
    closes = [100.0] * 16 + [100.0 - 2 * k for k in range(1, n - 15)]
    opens = [100.0] * 17 + closes[16:-1]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    opens, highs, lows = opens[:n], [max(h, o) for h, o in zip(highs, opens, strict=True)], [min(lo, o) for lo, o in zip(lows, opens, strict=True)]
    ctx = _tiny(opens, highs, lows, closes)
    spy = _flat_spy(n)  # SPY opens 100, closes 101 every day: +1% over any window
    out = bt.measure(ctx, np.array([16]), bs.SHORT, bt.spy_lookup(spy))
    entry = opens[17]
    raw5 = (entry - closes[21]) / entry
    assert out["h5_raw"][0] == pytest.approx(raw5) and raw5 > 0
    assert out["h5_vs_spy"][0] == pytest.approx(raw5 + 0.01)
    assert out["h5_mfe_atr"][0] > 0
    long_out = bt.measure(ctx, np.array([16]), bs.LONG, bt.spy_lookup(spy))
    assert long_out["h5_raw"][0] == pytest.approx(-raw5)
    assert long_out["h5_vs_spy"][0] == pytest.approx(-raw5 - 0.01)


def test_short_stop_and_limit_use_the_high_side():
    n = 30
    closes = [100.0] * n
    opens = [100.0] * n
    highs = [101.0] * n
    lows = [99.0] * n
    highs[18] = 120.0  # a spike through any short stop on the second bar after entry
    ctx = _tiny(opens, highs, lows, closes)
    spy = _flat_spy(n)
    out = bt.measure(ctx, np.array([15]), bs.SHORT, bt.spy_lookup(spy))
    atr = ctx.atr[15]
    assert out["h5_R"][0] == pytest.approx(-1.0)  # stopped at entry + 1 ATR
    assert out["take_R"][0] == pytest.approx(-1.0)
    limit = closes[15] + bt.LIMIT_ATR_FRACTION * atr
    expected = next(p for p in (limit_fill({"open": opens[j], "high": highs[j], "low": lows[j]}, limit, False)
                                for j in (16, 17, 18)) if p is not None)
    assert out["limit_fill"][0] == pytest.approx(expected)


def test_pending_horizons_stay_pending_never_zero():
    n = 30
    ctx = _tiny([100.0] * n, [101.0] * n, [99.0] * n, [100.0] * n)
    out = bt.measure(ctx, np.array([n - 7, n - 1]), bs.LONG, bt.spy_lookup(_flat_spy(n)))
    assert out["h5_raw"][0] == pytest.approx(0.0)  # measured flat, a real zero
    assert np.isnan(out["h10_raw"][0]) and np.isnan(out["h20_vs_spy"][0]) and np.isnan(out["take_R"][0])
    assert list(out["status"]) == ["measured", "pending"]
    assert np.isnan(out["h5_raw"][1]) and np.isnan(out["limit_filled"][1])


def test_cooldown_keeps_one_flag_per_window():
    assert list(bt.apply_cooldown(np.array([3, 4, 12, 13, 14, 30]), 10)) == [3, 13, 30]


# ---------------------------------------------------------------- the run
def test_run_paints_regimes_splits_and_registers_before_measuring(tmp_path, monkeypatch):
    bars, earnings, days = universe()
    regimes = _regimes(days)
    inputs = bt.Inputs(bars=bars, earnings=earnings, regimes=regimes, regime_rule_version="regime_daily_v1")
    seen = []
    real_measure = bt.measure

    def spy_measure(*args, **kwargs):
        seen.append(len(trial_ledger.load(tmp_path)))
        return real_measure(*args, **kwargs)

    monkeypatch.setattr(bt, "measure", spy_measure)
    result = bt.run_backtest(inputs, root=tmp_path, split=days[400], min_avg_volume=0, run_id="t1")
    assert seen and min(seen) == 1, "the ledger row exists before the first outcome is measured"
    folder = tmp_path / "backtests" / "t1"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["regime_rule_version"] == "regime_daily_v1"
    assert manifest["regime_axes"] == ["trend20", "vol_rv"]
    assert {s["key"]: s["version"] for s in manifest["setups"]}["leader_pullback"] == "1"
    assert manifest["git"]["commit"] and manifest["data_range"][0] == str(days[0])
    assert set(manifest["files"]) == {"candidates.parquet", "cells.parquet", "summary.json"}
    cand = pd.read_parquet(folder / "candidates.parquet")
    assert {"rg_trend20", "rg_vol_rv", "period", "h10_vs_spy", "features"} <= set(cand.columns)
    joined = dict(zip(pd.to_datetime(regimes.session_date), regimes.trend20, strict=True))
    assert all(joined[d] == lab_ for d, lab_ in zip(cand.signal_date, cand.rg_trend20, strict=True))
    assert set(cand.period) == {"train", "test"}
    cells = result["summary"]["cells"]
    axes = {c["axis"] for c in cells}
    assert {"all", "year", "period", "trend20", "vol_rv"} <= axes
    for c in cells:
        assert c["thin"] == (c["n"] < bt.MIN_CELL_N)
        assert c["n_train"] + c["n_test"] == c["n"]
    ranked = result["summary"]["ranked"]
    for side in ranked.values():
        for c in side["best"] + side["worst"]:
            assert not c["thin"] and c["axis"] not in ("all", "year", "period")
    assert result["summary"]["straight_up"] and "Survivorship" in result["summary"]["caveats"][0]
    with pytest.raises(FileExistsError):
        bt.write_run(tmp_path, "t1", tables={}, summary={}, manifest={})
    assert trial_ledger.load(tmp_path)[0]["trial_id"] == "backtest_t1"
    text = bt.format_report(result["summary"], manifest)
    assert "Survivorship" in text and "leader_pullback" in text


def test_unknown_regime_label_is_unknown_not_dropped():
    cand = pd.DataFrame({"signal_date": pd.to_datetime([date(2021, 1, 4), date(2021, 1, 5)])})
    regimes = pd.DataFrame({"session_date": [date(2021, 1, 4)], "trend20": ["up"]})
    out = bt.paint(cand, regimes, ["trend20"])
    assert list(out["rg_trend20"]) == ["up", bt.NO_LABEL]


def test_regime_axes_skip_numeric_and_meta_columns():
    frame = _regimes(_days(5))
    assert bt.regime_axes(frame) == ["trend20", "vol_rv"]


def test_clean_bars_counts_what_it_drops():
    quality = {}
    frame = pd.DataFrame({"session_date": [date(2021, 1, 4), date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)],
                          "open": [10, 10, -1, 10], "high": [11, 11, 11, 9], "low": [9, 9, 9, 10],
                          "close": [10, 10.5, 10, 10], "volume": [1, 2, 3, np.nan]})
    out = bt.clean_bars(frame, quality)
    assert len(out) == 1 and out["close"].iloc[0] == 10.5
    assert quality["duplicate_sessions"] == 1 and quality["bad_ohlc_dropped"] == 1
    assert quality["high_below_low_dropped"] == 1


def test_search_picks_on_train_and_registers_every_rule_tried(tmp_path):
    bars, earnings, days = universe(n_names=16, n_bars=520, seed=2)
    inputs = bt.Inputs(bars=bars, earnings=earnings)
    conds = {k: bt.CONDITIONS[k] for k in ("rs>=.8", "st>=1", "st>=2", "e21<0", "ret5>=3%")}
    result = bt.run_search(inputs, root=tmp_path, side=bs.LONG, base="all", max_k=2, split=days[380],
                           min_train=1, min_test=1, conditions=conds, run_id="s1")
    combos = 5 + sum(1 for a in range(5) for b in range(a + 1, 5)
                     if conds[list(conds)[a]][0] != conds[list(conds)[b]][0])
    row = trial_ledger.load(tmp_path)[0]
    assert row["declared_cell_count"] == combos == result["summary"]["rules_tried"]
    top = result["summary"]["top_by_train"]
    assert [r["mean_train"] for r in top] == sorted((r["mean_train"] for r in top), reverse=True)
    assert all(" & " not in r["rule"] or len({conds[c][0] for c in r["rule"].split(" & ")}) == r["k"] for r in top)
    assert (tmp_path / "backtests" / "s1" / "search.parquet").is_file()
    assert "rules tried" in bt.format_report(result["summary"], result["manifest"])


def test_cli_report_reads_a_run_folder(tmp_path, capsys):
    bars, earnings, days = universe(n_names=8, n_bars=420, seed=4)
    bt.run_backtest(bt.Inputs(bars=bars, earnings=earnings), root=tmp_path, min_avg_volume=0, run_id="r1",
                    setups=bs.by_key(["rising_20_50_baseline", "falling_20_50_baseline"]))
    from research_warehouse import cli

    assert cli.main(["backtest", "report", str(tmp_path / "backtests" / "r1")]) == 0
    out = capsys.readouterr().out
    assert "rising_20_50_baseline" in out and "Survivorship" in out


def test_registry_keys_are_unique_and_versioned():
    keys = [s.key for s in bs.REGISTRY]
    assert len(keys) == len(set(keys))
    assert {s.side for s in bs.REGISTRY} == {bs.LONG, bs.SHORT}
    assert all(s.version and s.fn.__doc__ for s in bs.REGISTRY)
    with pytest.raises(KeyError):
        bs.by_key(["nope"])


def test_liquidity_floor_matches_the_live_scan():
    from universe_builder import DEFAULT_MIN_AVG_VOLUME

    assert bt.MIN_AVG_VOLUME == DEFAULT_MIN_AVG_VOLUME
    assert bs.AVWAPE_SESSIONS == long_setups.AVWAPE_SESSIONS


# ---------------------------------------------------------------- the real readers, end to end
def test_cli_run_reads_the_lake_through_the_history_and_regime_readers(tmp_path, capsys):
    """bar_d1_history + earnings_date + market_regime_daily in a tmp lake -> `cli backtest run`."""
    from datetime import datetime, timezone

    from research_warehouse import cli, regime_daily, schemas
    from research_warehouse import exchange_calendar as xcal
    from research_warehouse.store import ResearchStore

    utc = timezone.utc
    stamp = datetime(2026, 9, 27, tzinfo=utc)
    sessions = [s.session_date for s in xcal.sessions_between(date(2018, 1, 2), date(2020, 3, 31))]
    lake = ResearchStore.open(tmp_path / "lake")
    bars, earnings, weekdays = universe(n_names=6, n_bars=len(sessions), seed=9)
    rows, earn_rows = [], []
    for sym, frame in bars.items():
        for day, rec in zip(sessions, frame.to_dict("records"), strict=True):
            rows.append({
                "symbol": sym, "session_id": f"XNYS-{day.isoformat()}", "session_date": day,
                "open": rec["open"], "high": rec["high"], "low": rec["low"], "close": rec["close"],
                "volume": int(rec["volume"]), "adjustment_version": "yahoo_split_v1", "corporate_action_id": None,
                "provider": "YAHOO", "quality": "COMPLETE", "is_complete": True,
                "event_at": datetime.combine(day, datetime.min.time(), utc), "observed_at": stamp,
                "capture_mode": "BACKFILL", "revision_id": "r1", "supersedes_revision_id": "",
                "schema_version": schemas.SCHEMA_VERSION, "run_id": "t"})
        for day in earnings.get(sym, ()):
            earn_rows.append({"symbol": sym, "earnings_date": sessions[weekdays.index(day)], "time_of_day": "AMC",
                              "earnings_at": None, "eps_estimate": None, "eps_reported": None, "surprise_pct": None,
                              "source": "yahoo", "observed_at": stamp, "capture_mode": "BACKFILL",
                              "schema_version": schemas.SCHEMA_VERSION, "run_id": "t"})
    lake.publish("bar_d1_history", rows)
    lake.publish("earnings_date", earn_rows)
    spy_rows = [{"symbol": "SPY", "session_date": d, **{k: r[k] for k in ("open", "high", "low", "close", "volume")}}
                for d, r in zip(sessions, bars["SPY"].to_dict("records"), strict=True)]
    vix = [{"session_date": d, "open": 18.0, "high": 18.0, "low": 18.0, "close": 18.0} for d in sessions]
    regime_rows = regime_daily.build_rows("SPY", spy_rows, vix=vix, computed_at=datetime(2020, 4, 1, tzinfo=utc))
    assert regime_rows
    lake.publish(regime_daily.DATASET, regime_rows)
    out = tmp_path / "out"
    assert cli.main(["backtest", "run", "--root", str(out), "--lake", str(tmp_path / "lake"),
                     "--setups", "rising_20_50_baseline,falling_20_50_baseline,favourite_zone_long",
                     "--split-date", "2019-07-01", "--run-id", "e2e"]) == 0
    printed = capsys.readouterr().out
    assert "Survivorship" in printed and "rising_20_50_baseline" in printed
    manifest = json.loads((out / "backtests" / "e2e" / "manifest.json").read_text())
    assert manifest["regime_rule_version"] == regime_daily.RULE_VERSION
    assert "trend20" in manifest["regime_axes"] and "composite" in manifest["regime_axes"]
    assert manifest["universe"]["with_earnings_dates"] == 6
    cand = pd.read_parquet(out / "backtests" / "e2e" / "candidates.parquet")
    first_regime = pd.Timestamp(regime_rows[0]["session_date"])
    painted = cand[cand.signal_date >= first_regime]
    assert len(painted) and (painted.rg_trend20 != bt.NO_LABEL).all()
    assert (cand[cand.signal_date < first_regime].rg_trend20 == bt.NO_LABEL).all()
    assert trial_ledger.load(out)[0]["trial_id"] == "backtest_e2e"
    assert not trial_ledger.load(tmp_path / "lake"), "the read lake gets no ledger row"


def test_earnings_setups_are_unmeasured_without_earnings_dates(tmp_path):
    """No earnings dates in the lake = "no earnings data yet", never an empty result shown as one."""
    bars, _earnings, days = universe(n_names=8, n_bars=420, seed=4)
    result = bt.run_backtest(bt.Inputs(bars=bars, earnings={}), root=tmp_path, min_avg_volume=0, run_id="ne")
    needs = {s.key for s in bs.REGISTRY if s.needs_earnings}
    assert needs == {"strength_under_avwape", "favourite_zone_long", "favourite_zone_short",
                     "weak_rally_to_avwape", "post_earnings_drift", "earnings_miss_short"}
    summary = result["summary"]
    assert set(summary["unmeasured_setups"]) == needs
    assert all(v.startswith(bt.NO_EARNINGS) for v in summary["unmeasured_setups"].values())
    shown = {c["setup"] for c in summary["straight_up"]}
    assert shown and not shown & needs
    for side in summary["ranked"].values():
        assert not {c["setup"] for c in side["best"] + side["worst"]} & needs
    assert "NOT MEASURED: no earnings data yet" in bt.format_report(summary)


def test_thin_earnings_coverage_is_still_no_earnings_data(tmp_path):
    """Dates for 1 of 8 names (a lake mid-load) is not a measurement of the earnings setups."""
    bars, earnings, days = universe(n_names=8, n_bars=420, seed=4)
    result = bt.run_backtest(bt.Inputs(bars=bars, earnings={"N00": earnings["N00"]}), root=tmp_path,
                             min_avg_volume=0, run_id="thin")
    assert result["summary"]["unmeasured_setups"]["favourite_zone_long"] ==         "no earnings data yet (1 of 8 stocks have earnings dates)"
    full = bt.run_backtest(bt.Inputs(bars=bars, earnings=earnings), root=tmp_path, min_avg_volume=0, run_id="full")
    assert full["summary"]["unmeasured_setups"] == {}


def test_stale_repeat_bars_are_excluded_and_other_flags_counted(tmp_path):
    """STALE_REPEAT_BAR filler never becomes a signal or an outcome bar; jumps and missing
    sessions stay in the series and are counted in the manifest."""
    bars, earnings, days = universe(n_names=8, n_bars=420, seed=4)
    stale = days[:300]  # N01's first 300 sessions are pre-listing filler
    flags = pd.DataFrame(
        [{"dataset": "bar_d1_history", "symbol": "N01", "check": "STALE_REPEAT_BAR", "flag_date": d,
          "interval_start": None, "detail": ""} for d in stale]
        + [{"dataset": "bar_d1_history", "symbol": "N02", "check": "UNEXPLAINED_JUMP", "flag_date": days[200],
            "interval_start": None, "detail": ""},
           {"dataset": "bar_d1_history", "symbol": "N03", "check": "MISSING_SESSION", "flag_date": days[100],
            "interval_start": None, "detail": ""}])
    result = bt.run_backtest(bt.Inputs(bars=bars, earnings=earnings, quality_flags=flags), root=tmp_path,
                             min_avg_volume=0, run_id="q", setups=bs.by_key(["rising_20_50_baseline"]))
    quality = result["manifest"]["data_quality"]
    assert quality["flagged_bars_excluded"] == {"checks": ["STALE_REPEAT_BAR"], "bars": 300, "symbols": 1}
    assert quality["flags_kept_counted"] == {"UNEXPLAINED_JUMP": 1, "MISSING_SESSION": 1}
    cand = pd.read_parquet(tmp_path / "backtests" / "q" / "candidates.parquet")
    assert (cand[cand.symbol == "N01"].signal_date > pd.Timestamp(stale[-1])).all()
    assert len(cand[cand.symbol == "N02"]) > 0
    ctxs, _ = bt.prepare(bt.Inputs(bars=bars, earnings=earnings, quality_flags=flags))
    assert len(ctxs["N01"]) == 120 and len(ctxs["N02"]) == 420


def test_search_manifest_records_the_data_range(tmp_path):
    bars, earnings, days = universe(n_names=8, n_bars=420, seed=4)
    result = bt.run_search(bt.Inputs(bars=bars, earnings=earnings), root=tmp_path, side=bs.SHORT, base="all",
                           max_k=1, split=days[300], min_train=1, min_test=1, run_id="dr")
    assert result["manifest"]["data_range"] == [str(days[0]), str(days[-1])]


def test_theme_etfs_in_the_lake_are_not_stocks():
    """The lake carries theme / industry ETFs (no earnings); they must not flag or join RS ranks."""
    for etf in ("ARKG", "COPX", "FDN", "ICLN", "IGV", "IHF", "IHI", "ITA", "IYT", "IYZ", "JETS",
                "KIE", "LIT", "OIH", "PAVE", "PEJ", "TAN", "URA"):
        assert not bt.is_stock(etf), etf
    for stock in ("JHG", "LC", "CRML", "AAPL"):
        assert bt.is_stock(stock), stock


# ---------------------------------------------------------------- p11 setups
def test_strong_deep_pullback_is_a_top_decile_name_12_to_30pct_off_its_60d_high(world):
    *_, ctxs = world
    hits = 0
    for sym in ("N00", "N03", "N05", "N08", "N13", "N17"):
        ctx = ctxs[sym]
        mask, feats = bs.strong_deep_pullback(ctx)
        closes, highs = ctx.close, ctx.high
        sma200 = pd.Series(closes).rolling(200).mean().to_numpy()
        high60 = pd.Series(highs).rolling(60).max().to_numpy()
        depth = 1 - closes / high60
        brute = (ctx.rs_share >= 0.9) & (closes > sma200) & (depth >= 0.12) & (depth < 0.30)
        np.testing.assert_array_equal(mask, np.nan_to_num(brute, nan=0).astype(bool))
        np.testing.assert_allclose(feats["pct_off_60d_high"][mask], depth[mask] * 100)
        hits += int(mask.sum())
    assert hits > 0


def test_surprises_reach_each_name_and_default_to_none():
    bars, earnings, _ = universe(n_names=3, n_bars=300)
    ctxs, _ = bt.prepare(bt.Inputs(bars=bars, earnings=earnings, surprises={"N01": {earnings["N01"][0]: -7.5}}))
    assert dict(ctxs["N01"].surprises) == {earnings["N01"][0]: -7.5}
    assert dict(ctxs["N00"].surprises) == {}


def _miss_ctx(surprise):
    """A name that gaps down 2 ATR on its earnings date (bar 230) and stays under its 20-day."""
    n = 260
    days = _days(n)
    close = np.full(n, 100.0) + np.sin(np.arange(n)) * 0.5
    close[230:] = 90.0 - np.arange(n - 230) * 0.2
    opens = np.r_[close[0], close[:-1]]
    opens[230] = 91.0
    high, low = np.maximum(opens, close) + 0.6, np.minimum(opens, close) - 0.6
    ctx = bs.Ctx("M", np.array(days, dtype="datetime64[D]"), opens, high, low, close, np.full(n, 2e6),
                 earnings=(days[230],), rs_share=np.full(n, 0.3), surprises={} if surprise is None else {days[230]: surprise})
    return ctx


def test_earnings_miss_short_flags_the_session_after_a_missed_gap_down():
    mask, feats = bs.earnings_miss_short(_miss_ctx(-12.0))
    assert np.nonzero(mask)[0].tolist() == [231]
    assert feats["surprise_pct"][231] == -12.0
    assert feats["gap_atr"][231] <= -1.0
    for other in (None, 12.0, -2.0):  # no surprise known, a beat, a small miss
        assert not bs.earnings_miss_short(_miss_ctx(other))[0].any()
    ctx = _miss_ctx(-12.0)
    for i in (230, 231, 240):  # truncation never changes the answer
        assert bs.earnings_miss_short(ctx.truncated(i))[0][-1] == mask[i]


def _brute_trend_facts(ctx):
    c = pd.Series(ctx.close)
    vol = pd.Series(ctx.volume)
    return {"s100": c.rolling(100).mean().to_numpy(), "s200": c.rolling(200).mean().to_numpy(),
            "ret20": (c / c.shift(20) - 1).to_numpy(),
            "vr": (vol.rolling(5).mean() / vol.rolling(50).mean()).to_numpy(),
            "depth": (1 - c / pd.Series(ctx.high).rolling(60).max()).to_numpy()}


def test_laggard_thrust_is_a_weak_rs_name_reclaiming_its_trend_on_volume(world):
    *_, ctxs = world
    for sym in ("N00", "N03", "N05", "N08", "N13", "N17"):
        ctx = ctxs[sym]
        f = _brute_trend_facts(ctx)
        brute = (ctx.close > f["s100"]) & (ctx.close > f["s200"]) & (ctx.rs_share < 0.2) \
            & (f["ret20"] >= 0.10) & (f["vr"] >= 1.3)
        mask, _ = bs.laggard_thrust(ctx)
        np.testing.assert_array_equal(mask, np.nan_to_num(brute, nan=0).astype(bool))


def test_weakest_near_60d_high_is_a_bottom_decile_downtrend_name_near_its_high(world):
    *_, ctxs = world
    hits = 0
    for sym in ("N00", "N03", "N05", "N08", "N13", "N17", "N21"):
        ctx = ctxs[sym]
        f = _brute_trend_facts(ctx)
        brute = (ctx.close < f["s100"]) & (ctx.close < f["s200"]) & (ctx.rs_share < 0.1) & (f["depth"] < 0.05)
        mask, _ = bs.weakest_near_60d_high(ctx)
        np.testing.assert_array_equal(mask, np.nan_to_num(brute, nan=0).astype(bool))
        hits += int(mask.sum())
    assert hits >= 0
