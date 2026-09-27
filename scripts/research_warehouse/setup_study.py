"""Honest statistics for a setup's trades: costs, week-block bootstrap, concentration, the
"works" verdict and a shuffled-date placebo. Pure pandas/numpy; shadow research only.

A trade frame has one row per flag with ``symbol``, ``signal_date`` and an outcome column
(a return, NaN = pending). Nothing here reads the lake or feeds a live score, alert or list.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

ROUND_TRIP_COST = 0.0010  # 10 bps per round trip, both sides
BORROW_PER_SESSION = 0.0005  # shorts: 5 bps per session held
MIN_TEST_TRADES = 30
TOP_SYMBOLS = 5
BOOT_SAMPLES = 2000
YEAR_MIN_TRADES = 10  # a year counts toward "positive in most years" only with this many trades


def net_of_costs(returns: pd.Series | np.ndarray, side: str, sessions: int) -> np.ndarray:
    """A return after the round-trip cost, and for a short the borrow over ``sessions``."""
    values = np.asarray(returns, dtype=float)
    cost = ROUND_TRIP_COST + (BORROW_PER_SESSION * sessions if side == "short" else 0.0)
    return values - cost


def week_of(dates: pd.Series) -> pd.Series:
    """The Monday of each signal date's week: trades in one week are one block."""
    d = pd.to_datetime(dates).dt.normalize()
    return d - pd.to_timedelta(d.dt.dayofweek, unit="D")


def block_bootstrap(values: np.ndarray, blocks: np.ndarray, *, samples: int = BOOT_SAMPLES,
                    seed: int = 7, level: float = 0.90) -> tuple[float, float]:
    """(low, high) interval of the mean when whole blocks (weeks) are resampled with
    replacement; NaN values are dropped first. (nan, nan) with fewer than 2 blocks."""
    v = np.asarray(values, dtype=float)
    b = np.asarray(blocks)
    keep = np.isfinite(v)
    v, b = v[keep], b[keep]
    codes, uniq = pd.factorize(b)
    k = len(uniq)
    if k < 2:
        return float("nan"), float("nan")
    sums = np.bincount(codes, weights=v, minlength=k)
    counts = np.bincount(codes, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, k, size=(samples, k))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def concentration(frame: pd.DataFrame, col: str, *, top: int = TOP_SYMBOLS) -> dict[str, Any]:
    """How much the result leans on a few names or one month: the mean without the ``top``
    best-contributing symbols and without the best month, and their share of the total."""
    f = frame[np.isfinite(frame[col].to_numpy(dtype=float))]
    if f.empty:
        return {"mean_ex_top_symbols": None, "top_symbols_share": None, "mean_ex_best_month": None,
                "best_month": None, "top_symbols": []}
    total = float(f[col].sum())
    by_sym = f.groupby("symbol")[col].sum().sort_values(ascending=False)
    top_syms = list(by_sym.index[:top])
    rest = f[~f["symbol"].isin(top_syms)]
    month = pd.to_datetime(f["signal_date"]).dt.to_period("M").astype(str)
    by_month = f.groupby(month)[col].sum().sort_values(ascending=False)
    best = by_month.index[0]
    ex_month = f[month != best]
    return {
        "mean_ex_top_symbols": float(rest[col].mean()) if len(rest) else None,
        "top_symbols_share": float(by_sym.iloc[:top].sum() / total) if total else None,
        "mean_ex_best_month": float(ex_month[col].mean()) if len(ex_month) else None,
        "best_month": str(best), "top_symbols": top_syms,
    }


def summarize(frame: pd.DataFrame, col: str, *, split: pd.Timestamp, seed: int = 7) -> dict[str, Any]:
    """Straight-up, train/test, by-year, distinct weeks, week-bootstrap CI and concentration."""
    f = frame[np.isfinite(frame[col].to_numpy(dtype=float))].copy()
    dates = pd.to_datetime(f["signal_date"])
    f["_week"] = week_of(dates)
    test = dates >= split

    def part(sel) -> dict[str, Any]:
        g = f[sel]
        v = g[col].to_numpy(dtype=float)
        if not len(v):
            return {"n": 0, "weeks": 0, "mean": None, "median": None, "win": None, "ci": [None, None]}
        lo, hi = block_bootstrap(v, g["_week"].to_numpy(), seed=seed)
        return {"n": int(len(v)), "weeks": int(g["_week"].nunique()), "symbols": int(g["symbol"].nunique()),
                "mean": float(v.mean()), "median": float(np.median(v)), "win": float((v > 0).mean()),
                "ci": [lo, hi]}

    years = {}
    for year, g in f.groupby(dates.dt.year):
        v = g[col].to_numpy(dtype=float)
        years[int(year)] = {"n": int(len(v)), "mean": float(v.mean()), "win": float((v > 0).mean())}
    return {"all": part(np.ones(len(f), dtype=bool)), "train": part(~test.to_numpy()),
            "test": part(test.to_numpy()), "years": years, "concentration": concentration(f, col)}


def verdict(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The four "works" checks: test n >= 30, test mean > 0 (after costs), positive in most
    years it trades (years with >= `YEAR_MIN_TRADES`), and not carried by the top symbols or
    one month (the mean stays > 0 without them)."""
    test = summary["test"]
    years = [y for y in summary["years"].values() if y["n"] >= YEAR_MIN_TRADES]
    pos_years = sum(1 for y in years if y["mean"] > 0)
    conc = summary["concentration"]
    checks = {
        "test_n": test["n"] >= MIN_TEST_TRADES,
        "test_mean_positive": bool(test["mean"] is not None and test["mean"] > 0),
        "most_years_positive": bool(years) and pos_years * 2 > len(years),
        "not_concentrated": bool((conc["mean_ex_top_symbols"] or 0) > 0 and (conc["mean_ex_best_month"] or 0) > 0),
    }
    return {"works": all(checks.values()), "checks": checks, "years_positive": f"{pos_years}/{len(years)}"}


def date_placebo(trades: pd.DataFrame, pool: pd.DataFrame, col: str, *, samples: int = 200,
                 seed: int = 11) -> dict[str, Any]:
    """Keep each trade's symbol and year, draw a random eligible day of that symbol-year from
    ``pool`` instead of the signal day, and re-average. The share of shuffles whose mean reaches
    the real mean says how much of the result the timing rule adds over just owning those names."""
    t = trades[np.isfinite(trades[col].to_numpy(dtype=float))]
    p = pool[np.isfinite(pool[col].to_numpy(dtype=float))]
    if t.empty or p.empty:
        return {"real": None, "placebo_mean": None, "p_value": None, "samples": 0}
    key_t = t["symbol"].astype(str) + "|" + pd.to_datetime(t["signal_date"]).dt.year.astype(str)
    key_p = p["symbol"].astype(str) + "|" + pd.to_datetime(p["signal_date"]).dt.year.astype(str)
    groups = {k: g.to_numpy(dtype=float) for k, g in p[col].groupby(key_p.to_numpy())}
    wanted = [k for k in key_t if k in groups]
    if not wanted:
        return {"real": float(t[col].mean()), "placebo_mean": None, "p_value": None, "samples": 0}
    rng = np.random.default_rng(seed)
    pools = [groups[k] for k in wanted]
    sizes = np.array([len(g) for g in pools])
    flat = np.concatenate(pools)
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    draws = offsets[None, :] + (rng.random((samples, len(pools))) * sizes[None, :]).astype(int)
    means = flat[draws].mean(axis=1)
    real = float(t[col].mean())
    return {"real": real, "placebo_mean": float(means.mean()), "placebo_p95": float(np.quantile(means, 0.95)),
            "p_value": float((means >= real).mean()), "samples": samples}


def name_placebo(trades: pd.DataFrame, pool: pd.DataFrame, col: str, *, samples: int = 200,
                 seed: int = 13) -> dict[str, Any]:
    """Keep each trade's signal DAY, swap its name for a random eligible name that day. The share
    of shuffles reaching the real mean says whether the rule picks better names than chance on
    the same days (the market and the week are held fixed; no future facts pick the names)."""
    t = trades[np.isfinite(trades[col].to_numpy(dtype=float))]
    p = pool[np.isfinite(pool[col].to_numpy(dtype=float))]
    if t.empty or p.empty:
        return {"real": None, "placebo_mean": None, "p_value": None, "samples": 0}
    days_t = pd.to_datetime(t["signal_date"]).dt.normalize().to_numpy()
    p = p.assign(_d=pd.to_datetime(p["signal_date"]).dt.normalize()).sort_values("_d", kind="stable")
    days_p = p["_d"].to_numpy()
    vals = p[col].to_numpy(dtype=float)
    lo = np.searchsorted(days_p, days_t, side="left")
    hi = np.searchsorted(days_p, days_t, side="right")
    ok = hi > lo
    if not ok.any():
        return {"real": float(t[col].mean()), "placebo_mean": None, "p_value": None, "samples": 0}
    lo, size = lo[ok], (hi - lo)[ok]
    rng = np.random.default_rng(seed)
    draws = lo[None, :] + (rng.random((samples, len(lo))) * size[None, :]).astype(int)
    means = vals[draws].mean(axis=1)
    real = float(t[col].to_numpy(dtype=float)[ok].mean())
    return {"real": real, "placebo_mean": float(means.mean()), "placebo_p95": float(np.quantile(means, 0.95)),
            "p_value": float((means >= real).mean()), "samples": samples}


__all__ = ["block_bootstrap", "concentration", "date_placebo", "name_placebo", "net_of_costs", "summarize",
           "verdict", "week_of"]
