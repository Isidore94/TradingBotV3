"""setup_study: costs, the week-block bootstrap, concentration, the verdict and the placebo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from research_warehouse import setup_study as ss  # noqa: E402

SPLIT = pd.Timestamp("2024-01-01")


def _trades(rows):
    return pd.DataFrame(rows, columns=["symbol", "signal_date", "out"]).assign(
        signal_date=lambda f: pd.to_datetime(f["signal_date"]))


def test_costs_charge_a_round_trip_and_borrow_on_shorts_only():
    np.testing.assert_allclose(ss.net_of_costs([0.01], "long", 10), [0.009])
    np.testing.assert_allclose(ss.net_of_costs([0.01], "short", 10), [0.01 - 0.001 - 0.005])


def test_bootstrap_resamples_weeks_not_trades():
    # one week of 50 identical winners and 49 weeks of one small loser: a trade bootstrap would
    # be sure the mean is positive; resampling weeks shows the interval spans zero.
    rows = [("A", "2024-01-02", 0.05)] * 50 + [("B", str(pd.Timestamp("2024-01-08") + pd.Timedelta(weeks=k))[:10], -0.01)
                                               for k in range(49)]
    f = _trades(rows)
    lo, hi = ss.block_bootstrap(f["out"].to_numpy(), ss.week_of(f["signal_date"]).to_numpy())
    assert lo < 0 < hi
    assert f["out"].mean() > 0


def test_bootstrap_needs_two_blocks():
    lo, hi = ss.block_bootstrap(np.array([0.1, 0.2]), np.array(["w", "w"]))
    assert np.isnan(lo) and np.isnan(hi)


def test_concentration_and_verdict_catch_a_one_name_result():
    rows = [("HERO", f"2024-0{m}-10", 0.50) for m in range(1, 7)]
    rows += [(f"N{k}", f"20{y}-03-1{k % 9}", -0.01) for k in range(6) for y in (19, 20, 21, 22, 23, 24, 25)]
    f = _trades(rows)
    s = ss.summarize(f, "out", split=SPLIT)
    assert s["all"]["mean"] > 0
    assert s["concentration"]["mean_ex_top_symbols"] < 0
    v = ss.verdict(s)
    assert not v["works"] and not v["checks"]["not_concentrated"]
    assert not v["checks"]["test_n"]  # 6 + 12 test trades < 30


def test_verdict_passes_a_broad_steady_edge():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2019-01-01", "2025-12-31")
    rows = [(f"S{k % 40}", d, 0.01 + rng.normal(0, 0.005)) for k, d in enumerate(days[::3])]
    s = ss.summarize(_trades(rows), "out", split=SPLIT)
    v = ss.verdict(s)
    assert v["works"], v
    assert s["test"]["weeks"] <= s["test"]["n"]
    assert s["test"]["ci"][0] > 0


def test_placebo_keeps_symbol_and_year_and_scores_timing():
    days = pd.bdate_range("2024-01-01", "2024-12-31")
    pool = pd.DataFrame({"symbol": "A", "signal_date": days, "out": np.where(np.arange(len(days)) % 10 == 0, 0.05, 0.0)})
    good = pool[pool["out"] > 0].head(20)
    res = ss.date_placebo(good, pool, "out", samples=300)
    assert abs(res["real"] - 0.05) < 1e-12
    assert res["p_value"] < 0.05
    assert abs(res["placebo_mean"] - 0.005) < 0.003
    # a symbol-year missing from the pool is not measured
    other = good.assign(symbol="Z")
    assert ss.date_placebo(other, pool, "out")["samples"] == 0


def test_name_placebo_holds_the_day_and_swaps_the_name():
    days = pd.bdate_range("2024-01-01", periods=60)
    rows = [(f"N{k}", d, (0.04 if k == 0 else 0.0) + (0.01 if j % 2 else -0.01))
            for j, d in enumerate(days) for k in range(10)]
    pool = pd.DataFrame(rows, columns=["symbol", "signal_date", "out"])
    picks = pool[pool["symbol"] == "N0"]
    res = ss.name_placebo(picks, pool, "out", samples=300)
    assert res["p_value"] < 0.05
    # the day's market move is kept: the placebo mean is the day-average, about +0.4%
    assert abs(res["placebo_mean"] - 0.004) < 0.002
    # random picks are not special
    rand = pool.groupby("signal_date").sample(1, random_state=1)
    assert ss.name_placebo(rand, pool, "out", samples=300)["p_value"] > 0.05
