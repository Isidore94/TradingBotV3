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


@pytest.fixture(scope="module")
def world():
    bars, earnings, days = universe()
    inputs = bt.Inputs(bars=bars, earnings=earnings)
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
            "rs_share": ctx.rs_share, "rs_decile": ctx.rs_decile}


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
