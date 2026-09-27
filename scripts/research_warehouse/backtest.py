"""D1 backtester: replay registered long and short setups over the lake's daily history,
paint every flag into the SPY regime of its signal session, and write an immutable run.
Shadow research only: nothing here feeds a detector, score, alert, tier, gate or the desk.

    python -m research_warehouse.cli backtest run [--setups a,b] [--split-date 2024-01-01]
    python -m research_warehouse.cli backtest search --side long [--base trend] [--max-k 3]
    python -m research_warehouse.cli backtest report <run_id | run folder>

Generalizes `long_lab` (point-in-time rules, RS ranks, exit models, the limit retest fill,
Wilson bounds, `MIN_CELL_N`) to both sides:

* Replay: a setup flags a name at most once per `COOLDOWN_SESSIONS` of that name's bars;
  entry at the next open (and a limit variant); outcomes at 5/10/20 sessions - raw, vs
  SPY over the same open-to-close window (a short beats SPY when the stock does worse),
  R with a 1 ATR stop, MFE / MAE in ATR - plus 1 ATR take/stop, 1 ATR trail and a
  10-session time exit. A horizon whose bars do not exist yet is pending (NaN), never 0.
* Paint: every flag carries its signal session's regime labels (B2's `read_regimes`);
  cells per setup x axis x label, plus the unconditioned straight-up result, by year and
  by a fixed train / test date split. Cells under `MIN_CELL_N` are shown, never ranked.
* Honesty: every run and search is registered in `trial_ledger` BEFORE its outcomes are
  measured, so the number of things tried is on record.

The universe is the lake's names today (Yahoo): delisted names are missing, which
flatters every long result and hides short winners - printed with every summary.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from research_warehouse import backtest_setups as bs
from research_warehouse import trial_ledger

SCHEMA = "backtest_run_v1"
SEARCH_SCHEMA = "backtest_search_v1"
HORIZONS = (5, 10, 20)
PRIMARY_HORIZON = 10
COOLDOWN_SESSIONS = 10
STOP_ATR = 1.0
TAKE_ATR = 1.0
TRAIL_ATR = 1.0
EXIT_CAP_SESSIONS = 20
TIME_EXIT_SESSIONS = 10
LIMIT_ATR_FRACTION = 0.25  # long_lab / retest_entry: 0.25 ATR through the flag close
LIMIT_LIVE_SESSIONS = 3
MIN_PRICE = 5.0
#: The live scan's liquidity floor (`universe_builder.DEFAULT_MIN_AVG_VOLUME`), 20-session mean.
MIN_AVG_VOLUME = 1_000_000
LIQUIDITY_SESSIONS = 20
MIN_CELL_N = 30  # evidence_stats.MIN_REPORTABLE_N: thinner cells are shown, never ranked
RS_SESSIONS = 63
RS_MIN_NAMES = 20
DEFAULT_SPLIT = date(2024, 1, 1)
BENCHMARK = "SPY"
#: Index / ETF symbols in the lake that are not tradeable stock setups (plus any "^" symbol).
NON_STOCKS = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "MDY", "RSP", "VTI", "VOO", "TLT", "IEF", "SHY", "HYG", "LQD",
    "GLD", "SLV", "USO", "UNG", "UUP", "EEM", "EFA", "FXI", "SMH", "SOXX", "XBI", "IBB", "KRE",
    "KBE", "XHB", "ITB", "XRT", "ARKK", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
    "XLU", "XLV", "XLY", "XME", "XOP", "GDX", "GDXJ", "VNQ", "TQQQ", "SQQQ", "UVXY", "VXX",
    # theme / industry ETFs found in the lake (no earnings dates), 2026-09-27
    "ARKG", "COPX", "FDN", "ICLN", "IGV", "IHF", "IHI", "ITA", "IYT", "IYZ", "JETS", "KIE",
    "LIT", "OIH", "PAVE", "PEJ", "TAN", "URA",
})
NO_LABEL = "unknown"
REGIME_META_COLUMNS = frozenset({"session_date", "symbol", "rule_version", "computed_at",
                                 "generated_at", "source", "provider", "as_of"})
LEDGER_FAMILY = "BACKTEST_D1"
AUTHORIZATION = ('trader 2026-09-27: "The goal is to be able to make determinations about what '
                 'setups work and be able to back test setups as well ... paint different setups '
                 'into [regimes] to see what works best"')
SURVIVORSHIP = ("Survivorship bias: the universe is the lake's names today (Yahoo); delisted and "
                "dropped names are missing. That flatters every long result and hides short "
                "winners - read every number as an upper bound for longs.")
CAVEATS = (
    SURVIVORSHIP,
    "Point in time: a flag reads bars up to its close only; entry is the next open. A horizon "
    "whose bars do not exist yet is pending, never zero.",
    f"Cells under {MIN_CELL_N} flags are shown but never ranked.",
    "Regime labels are as of the signal session's close (known before the next-open entry).",
    "'approx' setups reproduce a live scanner rule from bars only (no sector / top-pattern facts).",
    "Every run and search is registered in the trial ledger before outcomes are measured; "
    "read a good cell against how many were tried.",
    "Shadow research: nothing here changes an alert, score, grade, gate or list.",
)


# ---------------------------------------------------------------- inputs
@dataclass
class Inputs:
    """What a run reads: D1 bars per symbol, earnings dates, SPY regime labels."""

    bars: Mapping[str, pd.DataFrame]
    earnings: Mapping[str, Sequence[date]] = field(default_factory=dict)
    regimes: pd.DataFrame | None = None
    regime_rule_version: str | None = None
    regime_axes: Sequence[str] | None = None
    #: history_reader.read_quality_flags rows (dataset, symbol, check, flag_date, ...)
    quality_flags: pd.DataFrame | None = None
    source: dict[str, Any] = field(default_factory=dict)


def _to_day(value: Any) -> date:
    return pd.Timestamp(value).date()


def clean_bars(frame: pd.DataFrame, quality: dict[str, int]) -> pd.DataFrame:
    """Sorted, one row per session, finite positive OHLC with high >= low. Every dropped or
    suspicious row is counted in ``quality`` (the run's manifest carries it)."""
    df = frame.copy()
    if "session_date" not in df.columns:
        df = df.rename(columns={"date": "session_date", "datetime": "session_date"})
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col] if col in df.columns else np.nan, errors="coerce")
    quality["rows_in"] = quality.get("rows_in", 0) + len(df)
    before = len(df)
    df = df.sort_values("session_date").drop_duplicates("session_date", keep="last")
    quality["duplicate_sessions"] = quality.get("duplicate_sessions", 0) + before - len(df)
    ohlc = df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    bad = ~np.isfinite(ohlc).all(axis=1) | (ohlc <= 0).any(axis=1)
    quality["bad_ohlc_dropped"] = quality.get("bad_ohlc_dropped", 0) + int(bad.sum())
    df = df[~bad]
    inverted = (df["high"] < df["low"]).to_numpy()
    quality["high_below_low_dropped"] = quality.get("high_below_low_dropped", 0) + int(inverted.sum())
    df = df[~inverted]
    outside = ((df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1)))
    quality["open_close_outside_range_kept"] = quality.get("open_close_outside_range_kept", 0) + int(outside.sum())
    vol = df["volume"].to_numpy(dtype=float)
    quality["missing_volume"] = quality.get("missing_volume", 0) + int((~np.isfinite(vol)).sum())
    quality["zero_volume"] = quality.get("zero_volume", 0) + int((vol == 0).sum())
    quality["rows_kept"] = quality.get("rows_kept", 0) + len(df)
    return df.reset_index(drop=True)


def make_ctx(symbol: str, frame: pd.DataFrame, earnings: Sequence[date] = ()) -> bs.Ctx:
    return bs.Ctx(
        symbol=symbol,
        dates=frame["session_date"].to_numpy(dtype="datetime64[D]"),
        open=frame["open"].to_numpy(dtype=float), high=frame["high"].to_numpy(dtype=float),
        low=frame["low"].to_numpy(dtype=float), close=frame["close"].to_numpy(dtype=float),
        volume=frame["volume"].to_numpy(dtype=float), earnings=tuple(earnings or ()))


def is_stock(symbol: str) -> bool:
    return not symbol.startswith("^") and symbol.upper() not in NON_STOCKS


def rank_rs(ctxs: Mapping[str, bs.Ctx]) -> np.ndarray:
    """Per session, each stock's `RS_SESSIONS` return as the share of that day's names it beats
    (strictly; `long_setups.rs_percentiles`) and its decile 1-10; unknown under `RS_MIN_NAMES`
    names. Ranking the name's own return equals ranking it vs SPY. Writes onto each Ctx."""
    names = [s for s in ctxs if is_stock(s)]
    if not names:
        return np.zeros(0, dtype="datetime64[D]")
    sessions = np.unique(np.concatenate([ctxs[s].dates for s in names]))
    matrix = np.full((len(sessions), len(names)), np.nan)
    positions = {}
    for j, sym in enumerate(names):
        pos = np.searchsorted(sessions, ctxs[sym].dates)
        positions[sym] = pos
        matrix[pos, j] = ctxs[sym].ret(RS_SESSIONS)
    frame = pd.DataFrame(matrix)
    below = frame.rank(axis=1, method="min").to_numpy() - 1.0
    count = np.isfinite(matrix).sum(axis=1)[:, None].astype(float)
    with np.errstate(all="ignore"):
        share = np.where(count >= RS_MIN_NAMES, below / np.maximum(count - 1.0, 1.0), np.nan)
        decile = np.where(count >= RS_MIN_NAMES, np.minimum(10.0, np.floor(below * 10.0 / count) + 1.0), np.nan)
    share[~np.isfinite(matrix)] = np.nan
    decile[~np.isfinite(matrix)] = np.nan
    for j, sym in enumerate(names):
        ctxs[sym].rs_share = share[positions[sym], j]
        ctxs[sym].rs_decile = decile[positions[sym], j]
    return sessions


def eligible_mask(ctx: bs.Ctx, start: date | None, end: date | None, *, min_price: float = MIN_PRICE,
                  min_avg_volume: float = MIN_AVG_VOLUME) -> np.ndarray:
    """A bar may flag: close >= ``min_price``, 20-session mean volume >= the floor (all known),
    signal date inside [start, end]. Every input is at or before the bar."""
    vol_mean = bs.rolling(ctx.volume, LIQUIDITY_SESSIONS, np.mean)
    ok = (ctx.close >= min_price) & np.isfinite(vol_mean) & (vol_mean >= min_avg_volume)
    if start is not None:
        ok &= ctx.dates >= np.datetime64(start, "D")
    if end is not None:
        ok &= ctx.dates <= np.datetime64(end, "D")
    return ok


def apply_cooldown(indices: np.ndarray, cooldown: int = COOLDOWN_SESSIONS) -> np.ndarray:
    """Keep a flag only when ``cooldown`` or more of the name's bars passed since the last kept one."""
    kept, last = [], -10**9
    for i in indices.tolist():
        if i - last >= cooldown:
            kept.append(i)
            last = i
    return np.asarray(kept, dtype=np.int64)


# ---------------------------------------------------------------- outcomes (vectorized)
def _side_prices(ctx: bs.Ctx, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(open, high, low, close) in the side's favour: a short is a long on the negated prices,
    so every exit rule is written once and a price difference keeps its meaning in ATR."""
    if side == bs.LONG:
        return ctx.open, ctx.high, ctx.low, ctx.close
    return -ctx.open, -ctx.low, -ctx.high, -ctx.close


def _windows(values: np.ndarray, first: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """``values[first + k]`` for k < width as a 2-D array, and the mask of bars that exist."""
    cols = first[:, None] + np.arange(width)[None, :]
    exists = cols < len(values)
    return values[np.clip(cols, 0, len(values) - 1)], exists


def _stop_time_exit(o, h, lo, c, entry, stop, first, last) -> np.ndarray:
    """Exit price of a stop + time exit: from bar ``first`` to ``last`` inclusive, a later bar
    opening through the stop exits at its open, a bar touching the stop exits at the stop, else
    the close of ``last``. Rows with ``last`` past the data are NaN (pending)."""
    width = int(np.max(last - first + 1)) if len(first) else 0
    out = np.full(len(first), np.nan)
    if width <= 0:
        return out
    ow, exists = _windows(o, first, width)
    lw, _ = _windows(lo, first, width)
    k = np.arange(width)[None, :]
    inside = exists & (k <= (last - first)[:, None])
    gap = inside & (k > 0) & (ow <= stop[:, None])
    touch = inside & (lw <= stop[:, None])
    event = gap | touch
    has = event.any(axis=1)
    j = np.argmax(event, axis=1)
    rows = np.arange(len(first))
    price = np.where(gap[rows, j], ow[rows, j], stop)
    complete = last < len(c)
    out = np.where(has, price, np.where(complete, c[np.clip(last, 0, len(c) - 1)], np.nan))
    # a stop hit before the data ends is final even when ``last`` is still in the future
    return np.where(complete | has, out, np.nan)


def _take_exit(o, h, lo, c, entry, atr, first, last) -> np.ndarray:
    """long_lab._exit_take, vectorized: +TAKE / -STOP ATR, a later open through either exits at
    the open, a bar touching both is a stop, else the close of ``last``."""
    width = EXIT_CAP_SESSIONS
    target, stop = entry + TAKE_ATR * atr, entry - STOP_ATR * atr
    ow, exists = _windows(o, first, width)
    hw, _ = _windows(h, first, width)
    lw, _ = _windows(lo, first, width)
    k = np.arange(width)[None, :]
    inside = exists & (k <= (last - first)[:, None])
    later = inside & (k > 0)
    gap_stop = later & (ow <= stop[:, None])
    gap_take = later & (ow >= target[:, None])
    touch_stop = inside & (lw <= stop[:, None])
    touch_take = inside & (hw >= target[:, None])
    event = gap_stop | gap_take | touch_stop | touch_take
    has = event.any(axis=1)
    j = np.argmax(event, axis=1)
    rows = np.arange(len(first))
    price = np.where(gap_stop[rows, j] | gap_take[rows, j], ow[rows, j],
                     np.where(touch_stop[rows, j], stop, target))
    complete = last < len(c)
    out = np.where(has, price, c[np.clip(last, 0, len(c) - 1)])
    return np.where(complete | has, out, np.nan)


def _trail_exit(o, h, lo, c, entry, atr, first, last) -> np.ndarray:
    """long_lab._exit_trail, vectorized: stop starts TRAIL ATR under entry and is raised to the
    highest completed high - TRAIL ATR only after each bar completes."""
    width = EXIT_CAP_SESSIONS
    ow, exists = _windows(o, first, width)
    hw, _ = _windows(h, first, width)
    lw, _ = _windows(lo, first, width)
    k = np.arange(width)[None, :]
    inside = exists & (k <= (last - first)[:, None])
    hw_in = np.where(inside, hw, -np.inf)
    prior_high = np.concatenate([np.full((len(first), 1), -np.inf),
                                 np.maximum.accumulate(hw_in, axis=1)[:, :-1]], axis=1)
    stop = np.maximum((entry - TRAIL_ATR * atr)[:, None], prior_high - TRAIL_ATR * atr[:, None])
    gap = inside & (k > 0) & (ow <= stop)
    touch = inside & (lw <= stop)
    event = gap | touch
    has = event.any(axis=1)
    j = np.argmax(event, axis=1)
    rows = np.arange(len(first))
    price = np.where(gap[rows, j], ow[rows, j], stop[rows, j])
    complete = last < len(c)
    out = np.where(has, price, c[np.clip(last, 0, len(c) - 1)])
    return np.where(complete | has, out, np.nan)


def spy_lookup(spy: bs.Ctx) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """dates -> (SPY open, SPY close) on those sessions; NaN when SPY has no bar that day."""
    def lookup(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = np.searchsorted(spy.dates, days)
        pos_c = np.clip(pos, 0, len(spy.dates) - 1)
        hit = (pos < len(spy.dates)) & (spy.dates[pos_c] == days)
        return np.where(hit, spy.open[pos_c], np.nan), np.where(hit, spy.close[pos_c], np.nan)
    return lookup


def measure(ctx: bs.Ctx, idx: np.ndarray, side: str, spy_at: Callable,
            horizons: Sequence[int] = HORIZONS) -> dict[str, np.ndarray]:
    """Outcomes of flags at the closes of bars ``idx`` (entry at the next open), side-aware.

    Returns arrays aligned with ``idx``. Pending (bars not there yet) is NaN, never zero;
    ``status`` says measured / pending / no_atr."""
    n, m = len(ctx), len(idx)
    sign = 1.0 if side == bs.LONG else -1.0
    o, h, lo, c = _side_prices(ctx, side)
    atr = ctx.atr[idx]
    has_entry = idx + 1 < n
    e_idx = np.clip(idx + 1, 0, n - 1)
    entry = np.where(has_entry, o[e_idx], np.nan)
    base = np.abs(entry)
    ok_atr = np.isfinite(atr) & (atr > 0)
    status = np.where(~ok_atr, "no_atr", np.where(has_entry, "measured", "pending")).astype(object)
    out: dict[str, np.ndarray] = {"entry": np.abs(entry), "atr": atr, "status": status}
    entry_day = np.where(has_entry, ctx.dates[e_idx], np.datetime64("NaT"))
    out["entry_date"] = entry_day
    spy_open, _ = spy_at(ctx.dates[e_idx])
    spy_open = np.where(has_entry, spy_open, np.nan)
    live = has_entry & ok_atr
    atr_safe = np.where(ok_atr, atr, np.nan)
    first = idx + 1
    for hz in horizons:
        end = idx + hz
        done = live & (end < n)
        end_c = np.clip(end, 0, n - 1)
        raw = np.where(done, (c[end_c] - entry) / base, np.nan)
        _, spy_close = spy_at(ctx.dates[end_c])
        spy_ret = np.where(done, spy_close / spy_open - 1.0, np.nan)
        hi_w, ex = _windows(h, first, hz)
        lo_w, _ = _windows(lo, first, hz)
        mfe = (np.max(np.where(ex, hi_w, -np.inf), axis=1) - entry) / atr_safe
        mae = (entry - np.min(np.where(ex, lo_w, np.inf), axis=1)) / atr_safe
        stop = entry - STOP_ATR * atr_safe
        exit_px = _stop_time_exit(o, h, lo, c, entry, stop, first, end) if m else np.zeros(0)
        out[f"h{hz}_raw"] = raw
        out[f"h{hz}_spy"] = spy_ret
        out[f"h{hz}_vs_spy"] = raw - sign * spy_ret
        out[f"h{hz}_R"] = np.where(live, (exit_px - entry) / (STOP_ATR * atr_safe), np.nan)
        out[f"h{hz}_mfe_atr"] = np.where(done, mfe, np.nan)
        out[f"h{hz}_mae_atr"] = np.where(done, mae, np.nan)
    last = idx + EXIT_CAP_SESSIONS
    if m:
        out["take_R"] = np.where(live, (_take_exit(o, h, lo, c, entry, atr_safe, first, last) - entry) / atr_safe, np.nan)
        out["trail_R"] = np.where(live, (_trail_exit(o, h, lo, c, entry, atr_safe, first, last) - entry) / atr_safe, np.nan)
    else:
        out["take_R"] = out["trail_R"] = np.zeros(0)
    t_end = idx + TIME_EXIT_SESSIONS
    out["time10_R"] = np.where(live & (t_end < n), (c[np.clip(t_end, 0, n - 1)] - entry) / atr_safe, np.nan)
    # limit variant: LIMIT_ATR_FRACTION ATR through the flag close, live LIMIT_LIVE_SESSIONS bars
    limit = c[idx] - LIMIT_ATR_FRACTION * atr_safe
    ow, ex = _windows(o, first, LIMIT_LIVE_SESSIONS)
    lw, _ = _windows(lo, first, LIMIT_LIVE_SESSIONS)
    touched = ex & (lw <= limit[:, None])
    filled = touched.any(axis=1) & ok_atr
    j = np.argmax(touched, axis=1)
    rows = np.arange(m)
    fill = np.where(filled, np.minimum(ow[rows, j], limit), np.nan)
    window_done = idx + LIMIT_LIVE_SESSIONS < n
    out["limit_filled"] = np.where(filled, 1.0, np.where(window_done & ok_atr, 0.0, np.nan))
    out["limit_fill"] = np.abs(fill)
    fill_bar = idx + 1 + j
    for hz in horizons:
        end = idx + hz
        ok = filled & (end < n) & (fill_bar <= end)
        out[f"limit_h{hz}_raw"] = np.where(ok, (c[np.clip(end, 0, n - 1)] - fill) / np.abs(fill), np.nan)
    return out


# ---------------------------------------------------------------- regimes
def regime_axes(regimes: pd.DataFrame | None, axes: Sequence[str] | None = None) -> list[str]:
    """The label columns of B2's regime frame (every non-meta, non-numeric column)."""
    if regimes is None or regimes.empty:
        return []
    if axes:
        return [a for a in axes if a in regimes.columns]
    out = []
    for col in regimes.columns:
        if col in REGIME_META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(regimes[col]) and not pd.api.types.is_bool_dtype(regimes[col]):
            continue
        out.append(col)
    return out


def spy_trend_fallback(spy: bs.Ctx) -> pd.DataFrame:
    """Used only when no regime frame is given: SPY vs its 20-day SMA (long_lab.spy_trend_labels)."""
    s20 = spy.sma(20)
    p20 = bs.shift(s20, 5)
    label = np.where(~bs.finite(s20, p20), NO_LABEL,
                     np.where(spy.close <= s20, "below_20d",
                              np.where(s20 > p20, "above_rising_20d", "above_falling_20d")))
    return pd.DataFrame({"session_date": pd.to_datetime(spy.dates), "spy_trend20_fallback": label})


def paint(candidates: pd.DataFrame, regimes: pd.DataFrame, axes: Sequence[str]) -> pd.DataFrame:
    """Join each flag's signal session to its regime labels (``rg_<axis>``); missing = unknown."""
    if candidates.empty or not axes:
        return candidates
    frame = regimes[["session_date", *axes]].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame = frame.drop_duplicates("session_date", keep="last")
    frame = frame.rename(columns={a: f"rg_{a}" for a in axes})
    out = candidates.merge(frame, how="left", left_on="signal_date", right_on="session_date")
    out = out.drop(columns=["session_date"])
    for a in axes:
        col = f"rg_{a}"
        out[col] = out[col].astype(object).where(out[col].notna(), NO_LABEL).astype(str)
    return out


# ---------------------------------------------------------------- the replay
def find_and_measure(ctxs: Mapping[str, bs.Ctx], setups: Sequence[bs.Setup], spy: bs.Ctx, *,
                     start: date | None = None, end: date | None = None,
                     min_price: float = MIN_PRICE, min_avg_volume: float = MIN_AVG_VOLUME,
                     horizons: Sequence[int] = HORIZONS, cooldown: int = COOLDOWN_SESSIONS,
                     timings: dict[str, float] | None = None) -> pd.DataFrame:
    """Every flag of every setup over every stock, measured. One row per flag."""
    spy_at = spy_lookup(spy)
    parts: list[pd.DataFrame] = []
    timings = timings if timings is not None else {}
    for sym in sorted(ctxs):
        if not is_stock(sym):
            continue
        ctx = ctxs[sym]
        ok = eligible_mask(ctx, start, end, min_price=min_price, min_avg_volume=min_avg_volume)
        if not ok.any():
            continue
        ctx.cache["eligible"] = ok
        for setup in setups:
            t0 = time.perf_counter()
            mask, feats = bs.evaluate(setup, ctx)
            idx = apply_cooldown(np.nonzero(mask & ok)[0], cooldown)
            timings[setup.key] = timings.get(setup.key, 0.0) + time.perf_counter() - t0
            if not len(idx):
                continue
            res = measure(ctx, idx, setup.side, spy_at, horizons)
            frame = pd.DataFrame({
                "setup": setup.key, "side": setup.side, "family": setup.family,
                "setup_version": setup.version, "symbol": sym,
                "signal_date": pd.to_datetime(ctx.dates[idx]),
                "signal_close": ctx.close[idx], "rs_share": ctx.rs_share[idx],
                "rs_decile": ctx.rs_decile[idx], **res})
            frame["entry_date"] = pd.to_datetime(frame["entry_date"])
            frame["features"] = [json.dumps({k: _json_num(v[i]) for k, v in feats.items()}, sort_keys=True)
                                 for i in idx.tolist()]
            parts.append(frame)
        ctx.cache.clear()  # the next name needs the memory more than this one needs the cache
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _json_num(value: Any) -> float | None:
    value = float(value)
    return round(value, 6) if math.isfinite(value) else None


# ---------------------------------------------------------------- aggregation
def _wilson(wins: int, n: int) -> float | None:
    from swing_headline import wilson_lower_bound

    return wilson_lower_bound(wins, n)


def _period(dates: pd.Series, split: date) -> pd.Series:
    return np.where(dates < pd.Timestamp(split), "train", "test")


def _cell_stats(frame: pd.DataFrame, hz: int) -> dict[str, Any]:
    vs = frame[f"h{hz}_vs_spy"].to_numpy(dtype=float)
    raw = frame[f"h{hz}_raw"].to_numpy(dtype=float)
    r = frame[f"h{hz}_R"].to_numpy(dtype=float)
    period = frame["period"].to_numpy()
    known = np.isfinite(vs)
    n = int(known.sum())
    wins = int((vs[known] > 0).sum())
    out: dict[str, Any] = {
        "n": n, "pending": int((frame["status"] == "pending").sum() + (~known & (frame["status"] == "measured")).sum()),
        "symbols": int(frame.loc[known, "symbol"].nunique()),
        "win_vs_spy": wins / n if n else None, "wilson_lb": _wilson(wins, n),
        "mean_vs_spy": float(np.mean(vs[known])) if n else None,
        "median_vs_spy": float(np.median(vs[known])) if n else None,
        "win_raw": float(np.mean(raw[np.isfinite(raw)] > 0)) if np.isfinite(raw).any() else None,
        "mean_raw": float(np.nanmean(raw)) if np.isfinite(raw).any() else None,
        "mean_R": float(np.nanmean(r)) if np.isfinite(r).any() else None,
        "mfe_atr": _nanmedian(frame[f"h{hz}_mfe_atr"]), "mae_atr": _nanmedian(frame[f"h{hz}_mae_atr"]),
        "thin": n < MIN_CELL_N,
    }
    for name in ("train", "test"):
        sel = known & (period == name)
        k = int(sel.sum())
        out[f"n_{name}"] = k
        out[f"win_{name}"] = float(np.mean(vs[sel] > 0)) if k else None
        out[f"mean_{name}"] = float(np.mean(vs[sel])) if k else None
    for model in ("take_R", "trail_R", "time10_R"):
        out[model] = _nanmean(frame[model])
    lf = frame["limit_filled"].to_numpy(dtype=float)
    out["limit_fill_rate"] = float(np.nanmean(lf)) if np.isfinite(lf).any() else None
    out["limit_mean_raw"] = _nanmean(frame[f"limit_h{hz}_raw"])
    return out


def _nanmean(series: pd.Series) -> float | None:
    arr = series.to_numpy(dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else None


def _nanmedian(series: pd.Series) -> float | None:
    arr = series.to_numpy(dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else None


def build_cells(candidates: pd.DataFrame, axes: Sequence[str], horizons: Sequence[int] = HORIZONS
                ) -> list[dict[str, Any]]:
    """Cells per setup x axis x label x horizon. Axis ``all`` is the straight-up result, ``year``
    and ``period`` (train / test) the honesty splits, the rest the regime axes."""
    if candidates.empty:
        return []
    cells: list[dict[str, Any]] = []
    frame = candidates.copy()
    frame["all"] = "all"
    frame["year"] = frame["signal_date"].dt.year.astype(str)
    columns = [("all", "all"), ("year", "year"), ("period", "period")] + [(a, f"rg_{a}") for a in axes]
    for (setup, side), sub in frame.groupby(["setup", "side"], sort=True):
        for axis, col in columns:
            for label, group in sub.groupby(col, sort=True):
                for hz in horizons:
                    cells.append({"setup": setup, "side": side, "axis": axis, "label": str(label),
                                  "horizon": hz, **_cell_stats(group, hz)})
    return cells


def rank_cells(cells: Sequence[Mapping[str, Any]], *, horizon: int = PRIMARY_HORIZON, top: int = 3
               ) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Best and worst regime cells per side at ``horizon`` by mean vs SPY - only cells with
    n >= `MIN_CELL_N` on a regime axis (never all / year / period)."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for side in (bs.LONG, bs.SHORT):
        pool = [c for c in cells if c["side"] == side and c["horizon"] == horizon and not c["thin"]
                and c["axis"] not in ("all", "year", "period") and c["label"] != NO_LABEL
                and c["mean_vs_spy"] is not None]
        pool.sort(key=lambda c: (c["mean_vs_spy"], c["wilson_lb"] or 0.0), reverse=True)
        out[side] = {"best": pool[:top], "worst": list(reversed(pool[-top:])) if pool else []}
    return out


def straight_up(cells: Sequence[Mapping[str, Any]], horizon: int = PRIMARY_HORIZON) -> list[dict[str, Any]]:
    return [c for c in cells if c["axis"] == "all" and c["horizon"] == horizon]


# ---------------------------------------------------------------- provenance and the run folder
def git_state(repo: Path | None = None) -> dict[str, Any]:
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "scripts"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        branch = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None, "branch": "unknown"}
    return {"commit": head or "unknown", "dirty": bool(dirty), "branch": branch or "unknown"}


def new_run_id(kind: str, params: Mapping[str, Any], now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{stamp}_{kind}_{digest}"


def backtests_dir(root: Path) -> Path:
    return Path(root) / "backtests"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run(root: Path, run_id: str, *, tables: Mapping[str, pd.DataFrame],
              summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> Path:
    """Write one immutable run folder: built beside it, then renamed into place. An existing
    run id is refused - a run is never rewritten."""
    final = backtests_dir(root) / run_id
    if final.exists():
        raise FileExistsError(f"backtest run {run_id} already exists; runs are immutable")
    partial = backtests_dir(root) / f".{run_id}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    files = {}
    for name, table in tables.items():
        path = partial / f"{name}.parquet"
        table.to_parquet(path, index=False)
        files[path.name] = _sha256(path)
    (partial / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    files["summary.json"] = _sha256(partial / "summary.json")
    full = {**manifest, "files": files}
    (partial / "manifest.json").write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
    os.replace(partial, final)
    return final


def resolve_run(root: Path | None, ref: str) -> Path:
    path = Path(ref)
    if path.is_dir():
        return path
    if root is None:
        raise FileNotFoundError(f"no research store configured and {ref} is not a folder")
    path = backtests_dir(root) / ref
    if not path.is_dir():
        raise FileNotFoundError(f"no backtest run {ref} under {backtests_dir(root)}")
    return path


def family_trials(root: Path) -> int:
    return sum(1 for row in trial_ledger.load(root) if row.get("family") == LEDGER_FAMILY)


# ---------------------------------------------------------------- one run
#: Quality flags whose bars are not real trading (flat zero-volume pre-listing filler): dropped.
EXCLUDED_FLAGS = frozenset({"STALE_REPEAT_BAR"})
#: Flags kept in the series but counted in the manifest.
COUNTED_FLAGS = ("UNEXPLAINED_JUMP", "MISSING_SESSION")


def _flag_days(flags: pd.DataFrame | None) -> tuple[dict[str, set], dict[str, int]]:
    """``{symbol: {excluded session dates}}`` and per-check counts of every flag given."""
    if flags is None or flags.empty:
        return {}, {}
    counts = {str(k): int(v) for k, v in flags["check"].value_counts().items()}
    drop = flags[flags["check"].isin(EXCLUDED_FLAGS)]
    days: dict[str, set] = {}
    for sym, group in drop.groupby("symbol"):
        days[str(sym).upper()] = set(pd.to_datetime(group["flag_date"]).dt.normalize())
    return days, counts


def prepare(inputs: Inputs) -> tuple[dict[str, bs.Ctx], dict[str, Any]]:
    """Clean bars per symbol; bars flagged `EXCLUDED_FLAGS` are removed before any signal or
    outcome reads them. The manifest's data_quality carries every count."""
    quality: dict[str, Any] = {}
    ctxs: dict[str, bs.Ctx] = {}
    drop_days, flag_counts = _flag_days(inputs.quality_flags)
    excluded, excluded_symbols = 0, 0
    for sym, frame in inputs.bars.items():
        if frame is None or len(frame) == 0:
            continue
        clean = clean_bars(frame, quality)
        key = str(sym).upper()
        if key in drop_days and len(clean):
            flagged = clean["session_date"].isin(drop_days[key]).to_numpy()
            if flagged.any():
                excluded += int(flagged.sum())
                excluded_symbols += 1
                clean = clean[~flagged].reset_index(drop=True)
        if len(clean):
            ctxs[key] = make_ctx(key, clean, inputs.earnings.get(sym, ()))
    quality["symbols"] = len(ctxs)
    quality["flagged_bars_excluded"] = {"checks": sorted(EXCLUDED_FLAGS), "bars": excluded,
                                        "symbols": excluded_symbols}
    quality["flags_by_check"] = flag_counts
    quality["flags_kept_counted"] = {c: flag_counts.get(c, 0) for c in COUNTED_FLAGS}
    return ctxs, quality


def data_range(ctxs: Mapping[str, bs.Ctx]) -> list[str | None]:
    first = min((c.dates[0] for c in ctxs.values() if len(c)), default=None)
    last = max((c.dates[-1] for c in ctxs.values() if len(c)), default=None)
    return [str(first) if first is not None else None, str(last) if last is not None else None]


def run_backtest(inputs: Inputs, *, root: Path, setups: Sequence[bs.Setup] | None = None,
                 start: date | None = None, end: date | None = None, split: date = DEFAULT_SPLIT,
                 horizons: Sequence[int] = HORIZONS, axes: Sequence[str] | None = None,
                 min_price: float = MIN_PRICE, min_avg_volume: float = MIN_AVG_VOLUME,
                 run_id: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Replay, register, measure, paint, aggregate and write one immutable run. Returns
    ``{run_id, path, summary}``."""
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    setups = list(setups or bs.REGISTRY)
    ctxs, quality = prepare(inputs)
    if BENCHMARK not in ctxs:
        raise ValueError("no SPY bars: the benchmark is required")
    spy = ctxs[BENCHMARK]
    rank_rs(ctxs)
    timings["load_and_rank_s"] = time.perf_counter() - t_start
    regimes = inputs.regimes
    default_axes = inputs.regime_axes
    if regimes is None or regimes.empty:
        regimes = spy_trend_fallback(spy)
        regime_version, default_axes = "fallback:spy_trend20", None
    else:
        regime_version = inputs.regime_rule_version or _rule_version(regimes)
    axis_list = regime_axes(regimes, axes or default_axes)
    params = {"setups": [s.key for s in setups], "start": start, "end": end, "split": split,
              "horizons": list(horizons), "axes": axis_list, "min_price": min_price,
              "min_avg_volume": min_avg_volume, "cooldown_sessions": COOLDOWN_SESSIONS,
              "entry": "next session open", "stop_atr": STOP_ATR, "take_atr": TAKE_ATR,
              "trail_atr": TRAIL_ATR, "exit_cap_sessions": EXIT_CAP_SESSIONS,
              "time_exit_sessions": TIME_EXIT_SESSIONS,
              "limit": f"{LIMIT_ATR_FRACTION} ATR through the flag close, live {LIMIT_LIVE_SESSIONS} sessions",
              "atr": "ATR(14) mean true range", "rs": f"{RS_SESSIONS}-session return share, >= {RS_MIN_NAMES} names",
              "regime_rule_version": regime_version}
    run_id = run_id or new_run_id("run", params, now)
    # registered BEFORE any outcome is measured: the ledger records the look, not the result
    labels = sum(max(1, regimes[a].nunique()) for a in axis_list)
    trial = {
        "trial_id": f"backtest_{run_id}", "family": LEDGER_FAMILY,
        "question": "Straight-up and regime-painted forward results of the registered D1 setups.",
        "failure_mode": ("A regime cell looks good by chance: many setups x axes x labels are read, "
                         "so a cell is evidence only when it also holds in the test period."),
        "declared_cells": {"setup": [f"{s.key}@v{s.version}" for s in setups], "horizon": list(horizons),
                           "regime_axes": axis_list},
        "declared_cell_count": len(setups) * (1 + labels) * len(horizons),
        "declared_floors": {"min_cell_n": MIN_CELL_N},
        "declared_window": {"kind": "historical_replay", "split": str(split)},
        "authorization": AUTHORIZATION, "analysis_unit": "flag", "status": trial_ledger.STATUS_REGISTERED,
        "outcome": "", "registered_by": "research_warehouse.backtest run",
    }
    trial_ledger.register(root, trial)
    t0 = time.perf_counter()
    candidates = find_and_measure(ctxs, setups, spy, start=start, end=end, min_price=min_price,
                                  min_avg_volume=min_avg_volume, horizons=horizons, timings=timings)
    timings["replay_s"] = time.perf_counter() - t0
    if not candidates.empty:
        candidates["period"] = _period(candidates["signal_date"], split)
        candidates = paint(candidates, regimes, axis_list)
    t0 = time.perf_counter()
    cells = build_cells(candidates, axis_list, horizons)
    timings["aggregate_s"] = time.perf_counter() - t0
    data_first = min((c.dates[0] for c in ctxs.values() if len(c)), default=None)
    data_last = max((c.dates[-1] for c in ctxs.values() if len(c)), default=None)
    signal_range = ([str(candidates["signal_date"].min().date()), str(candidates["signal_date"].max().date())]
                    if not candidates.empty else [None, None])
    counts = candidates.groupby("setup").size().to_dict() if not candidates.empty else {}
    unmeasured = unmeasured_setups(setups, ctxs)
    shown = [c for c in cells if c["setup"] not in unmeasured]
    summary = {
        "schema": SCHEMA, "run_id": run_id, "caveats": list(CAVEATS),
        "unmeasured_setups": unmeasured,
        "straight_up": straight_up(shown), "ranked": rank_cells(shown),
        "cells": cells, "flags_per_setup": counts, "family_trials_to_date": family_trials(root),
    }
    timings["total_s"] = time.perf_counter() - t_start
    manifest = {
        "schema": SCHEMA, "run_id": run_id, "kind": "run",
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "git": git_state(), "setups": [s.meta() for s in setups], "params": params,
        "regime_rule_version": regime_version, "regime_axes": axis_list,
        "data_range": [str(data_first) if data_first is not None else None,
                       str(data_last) if data_last is not None else None],
        "signal_range": signal_range, "data_quality": quality, "source": inputs.source,
        "universe": {"symbols": len(ctxs), "stocks": sum(1 for s in ctxs if is_stock(s)),
                     "with_earnings_dates": sum(1 for s, c in ctxs.items() if c.earnings)},
        "trial_id": trial["trial_id"], "timings": {k: round(v, 3) for k, v in timings.items()},
        "survivorship": SURVIVORSHIP,
    }
    cells_frame = pd.DataFrame(cells)
    path = write_run(root, run_id, tables={"candidates": candidates, "cells": cells_frame},
                     summary=summary, manifest=manifest)
    return {"run_id": run_id, "path": path, "summary": summary, "manifest": manifest}


NO_EARNINGS = "no earnings data yet"
#: Earnings setups are reported only when at least this share of the stocks has earnings dates.
EARNINGS_MIN_COVERAGE = 0.5


def unmeasured_setups(setups: Sequence[bs.Setup], ctxs: Mapping[str, bs.Ctx]) -> dict[str, str]:
    """Setups that read earnings dates when too few stocks have any: their result is missing
    input, not evidence, so they are listed here and kept out of the straight-up and ranked views."""
    stocks = [c for s, c in ctxs.items() if is_stock(s)]
    covered = sum(1 for c in stocks if c.earnings)
    if stocks and covered / len(stocks) >= EARNINGS_MIN_COVERAGE:
        return {}
    why = f"{NO_EARNINGS} ({covered} of {len(stocks)} stocks have earnings dates)"
    return {s.key: why for s in setups if s.needs_earnings}


def _rule_version(regimes: pd.DataFrame) -> str:
    if "rule_version" in regimes.columns:
        values = sorted({str(v) for v in regimes["rule_version"].dropna().unique()})
        return ",".join(values) or "unknown"
    return "unknown"


# ---------------------------------------------------------------- search
#: name -> (feature, low, high): the condition holds when low <= feature < high (None = open).
CONDITIONS: dict[str, tuple[str, float | None, float | None]] = {
    "rs>=.9": ("rs", 0.9, None), "rs>=.8": ("rs", 0.8, None),
    "rs<.2": ("rs", None, 0.2), "rs<.1": ("rs", None, 0.1),
    "st>=1": ("st", 1.0, None), "st>=2": ("st", 2.0, None), "st>=3": ("st", 3.0, None),
    "st<-1": ("st", None, -1.0), "st<-2": ("st", None, -2.0), "-1<=st<1": ("st", -1.0, 1.0),
    "avz<-1": ("avz", None, -1.0), "-1<=avz<0": ("avz", -1.0, 0.0), "0<=avz<1": ("avz", 0.0, 1.0),
    "avz>=1": ("avz", 1.0, None),
    "since<=20": ("since", None, 20.5), "since21-60": ("since", 20.5, 60.5), "since>60": ("since", 60.5, None),
    "depth<5%": ("depth", None, 0.05), "depth5-12%": ("depth", 0.05, 0.12), "depth>=12%": ("depth", 0.12, None),
    "e21<0": ("e21", None, 0.0), "e21>=1": ("e21", 1.0, None),
    "hi52>=.95": ("hi52", 0.95, None), "hi52<.85": ("hi52", None, 0.85),
    "lo52<1.05": ("lo52", None, 1.05),
    "ret5<-3%": ("ret5", None, -0.03), "ret5>=3%": ("ret5", 0.03, None),
    "ret20<-10%": ("ret20", None, -0.10), "ret20>=10%": ("ret20", 0.10, None),
    "vr<0.8": ("vr", None, 0.8), "vr>=1.3": ("vr", 1.3, None),
}


def search_features(ctx: bs.Ctx) -> dict[str, np.ndarray]:
    """The search's per-bar features, each from bars <= i."""
    c = ctx.close
    atr20 = ctx.atr20
    av = ctx.earnings_avwap(bs.AVWAPE_SESSIONS)
    vol = np.where(np.isfinite(ctx.volume), ctx.volume, np.nan)
    with np.errstate(all="ignore"):
        return {
            "rs": ctx.rs_share,
            "st": (c - ctx.sma(50)) / atr20,
            "avz": (c - av["level"]) / av["sigma"],
            "since": np.where(av["since"] >= 0, av["since"], np.nan).astype(float),
            "depth": 1.0 - c / ctx.max_high(60),
            "e21": (c - ctx.ema(21)) / atr20,
            "hi52": c / ctx.max_high(bs.HIGH_52W_BARS),
            "lo52": c / ctx.min_low(bs.HIGH_52W_BARS),
            "ret5": ctx.ret(5), "ret20": ctx.ret(20),
            "vr": bs.rolling(vol, 5, np.mean) / bs.rolling(vol, 50, np.mean),
        }


BASES = ("trend", "all")


def search_population(ctxs: Mapping[str, bs.Ctx], side: str, spy: bs.Ctx, *, base: str = "trend",
                      horizon: int = PRIMARY_HORIZON, start: date | None = None, end: date | None = None,
                      setups: Sequence[bs.Setup] = bs.REGISTRY, min_price: float = MIN_PRICE,
                      min_avg_volume: float = MIN_AVG_VOLUME) -> pd.DataFrame:
    """One row per eligible name-day in the base population with features and the outcome vs
    SPY at ``horizon`` (side-aware; NaN = pending). ``base``: ``trend`` (long: above the 100 and
    200-day; short: under both), ``all``, or a setup key (that setup's raw flags)."""
    spy_at = spy_lookup(spy)
    known = {s.key: s for s in setups}
    sign = 1.0 if side == bs.LONG else -1.0
    parts = []
    for sid, sym in enumerate(sorted(ctxs)):
        if not is_stock(sym):
            continue
        ctx = ctxs[sym]
        ok = eligible_mask(ctx, start, end, min_price=min_price, min_avg_volume=min_avg_volume)
        ctx.cache["eligible"] = ok
        if base == "trend":
            s100, s200 = ctx.sma(100), ctx.sma(200)
            ok = ok & (((ctx.close > s100) & (ctx.close > s200)) if side == bs.LONG
                       else ((ctx.close < s100) & (ctx.close < s200)))
        elif base in known:
            ok = ok & bs.evaluate(known[base], ctx)[0]
        elif base != "all":
            raise KeyError(f"unknown base {base!r}: use {', '.join(BASES)} or a setup key")
        idx = np.nonzero(ok)[0]
        if len(idx):
            n = len(ctx)
            e = np.clip(idx + 1, 0, n - 1)
            end_i = np.clip(idx + horizon, 0, n - 1)
            done = (idx + horizon < n)
            entry = ctx.open[e]
            spy_o, _ = spy_at(ctx.dates[e])
            _, spy_c = spy_at(ctx.dates[end_i])
            stock = ctx.close[end_i] / entry - 1.0
            out = np.where(done, sign * (stock - (spy_c / spy_o - 1.0)), np.nan)
            feats = search_features(ctx)
            frame = pd.DataFrame({"sid": sid, "symbol": sym, "i": idx,
                                  "signal_date": pd.to_datetime(ctx.dates[idx]),
                                  **{k: v[idx].astype(np.float32) for k, v in feats.items()},
                                  "out": out})
            parts.append(frame)
        ctx.cache.clear()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _combo_score(mask: np.ndarray, key: np.ndarray, out: np.ndarray, train: np.ndarray
                 ) -> tuple[tuple[int, float, float], tuple[int, float, float], tuple[int, float, float]]:
    """(n, win, mean) on train / test / all after one flag per name per cooldown block."""
    rows = np.nonzero(mask)[0]
    if len(rows):
        k = key[rows]
        rows = rows[np.concatenate(([True], k[1:] != k[:-1]))]
    o, tr = out[rows], train[rows]
    fin = np.isfinite(o)

    def stats(sel):
        vals = o[sel & fin]
        return (len(vals), float(np.mean(vals > 0)) if len(vals) else float("nan"),
                float(np.mean(vals)) if len(vals) else float("nan"))
    return stats(tr), stats(~tr), stats(np.ones(len(rows), dtype=bool))


def run_search(inputs: Inputs, *, root: Path, side: str, base: str = "trend", horizon: int = PRIMARY_HORIZON,
               max_k: int = 3, split: date = DEFAULT_SPLIT, min_train: int = 40, min_test: int = 20,
               top: int = 25, start: date | None = None, end: date | None = None,
               regime: tuple[str, str] | None = None, conditions: Mapping[str, tuple] | None = None,
               run_id: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Grid-search feature conditions (1..``max_k`` of them, never two on one feature) on a base
    population; pick on the TRAIN period, judge on the TEST period. Registered in the trial ledger
    with the number of rules tried before any is scored."""
    t_start = time.perf_counter()
    conditions = dict(conditions or CONDITIONS)
    ctxs, quality = prepare(inputs)
    spy = ctxs[BENCHMARK]
    rank_rs(ctxs)
    regimes = inputs.regimes if inputs.regimes is not None and not inputs.regimes.empty else spy_trend_fallback(spy)
    names = list(conditions)
    combos = [combo for k in range(1, max_k + 1) for combo in itertools.combinations(names, k)
              if len({conditions[c][0] for c in combo}) == k]
    params = {"side": side, "base": base, "horizon": horizon, "max_k": max_k, "split": split,
              "min_train": min_train, "min_test": min_test, "start": start, "end": end,
              "regime": list(regime) if regime else None, "conditions": conditions,
              "cooldown_sessions": COOLDOWN_SESSIONS}
    run_id = run_id or new_run_id(f"search-{side}", params, now)
    trial = {
        "trial_id": f"backtest_{run_id}", "family": LEDGER_FAMILY,
        "question": f"Which feature conditions on the {side} '{base}' population beat SPY at {horizon} sessions?",
        "failure_mode": "The best train rules are noise: judge only on the test period and against the count tried.",
        "declared_cells": {"conditions": names, "max_k": max_k, "regime": list(regime) if regime else None},
        "declared_cell_count": len(combos), "declared_floors": {"min_train": min_train, "min_test": min_test},
        "declared_window": {"kind": "historical_replay", "split": str(split)},
        "authorization": AUTHORIZATION, "analysis_unit": "flag", "status": trial_ledger.STATUS_REGISTERED,
        "outcome": "", "registered_by": "research_warehouse.backtest search",
    }
    trial_ledger.register(root, trial)
    pop = search_population(ctxs, side, spy, base=base, horizon=horizon, start=start, end=end)
    if pop.empty:
        raise ValueError("empty search population")
    if regime:
        axis, label = regime
        painted = paint(pop[["signal_date"]].copy(), regimes, [axis])
        pop = pop[(painted[f"rg_{axis}"] == label).to_numpy()].reset_index(drop=True)
    pop = pop.sort_values(["sid", "i"], kind="stable").reset_index(drop=True)
    key = pop["sid"].to_numpy(np.int64) * 1_000_000 + pop["i"].to_numpy(np.int64) // COOLDOWN_SESSIONS
    out = pop["out"].to_numpy(dtype=float)
    train = (pop["signal_date"] < pd.Timestamp(split)).to_numpy()
    masks = {}
    for name, (feat, low, high) in conditions.items():
        values = pop[feat].to_numpy(dtype=float)
        m = np.isfinite(values)
        if low is not None:
            m &= values >= low
        if high is not None:
            m &= values < high
        masks[name] = m
    rows = []
    for combo in combos:
        mask = masks[combo[0]].copy()
        for c in combo[1:]:
            mask &= masks[c]
        tr, te, al = _combo_score(mask, key, out, train)
        rows.append({"rule": " & ".join(combo), "k": len(combo), "n_train": tr[0], "win_train": tr[1],
                     "mean_train": tr[2], "n_test": te[0], "win_test": te[1], "mean_test": te[2],
                     "n": al[0], "win": al[1], "mean": al[2]})
    table = pd.DataFrame(rows)
    base_tr, base_te, base_all = _combo_score(np.ones(len(pop), dtype=bool), key, out, train)
    eligible = table[(table.n_train >= min_train) & (table.n_test >= min_test)]
    picked = eligible.sort_values("mean_train", ascending=False).head(top)
    summary = {
        "schema": SEARCH_SCHEMA, "run_id": run_id, "side": side, "base": base, "horizon": horizon,
        "caveats": list(CAVEATS), "rules_tried": len(combos), "rules_scored": int(len(eligible)),
        "population": {"train": base_tr, "test": base_te, "all": base_all},
        "top_by_train": picked.to_dict("records"),
        "top_beat_spy_on_test": int((picked.mean_test > 0).sum()),
        "top_median_test": float(picked.mean_test.median()) if len(picked) else None,
        "family_trials_to_date": family_trials(root),
    }
    manifest = {
        "schema": SEARCH_SCHEMA, "run_id": run_id, "kind": "search",
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "git": git_state(), "params": params, "data_quality": quality, "source": inputs.source,
        "data_range": data_range(ctxs),
        "trial_id": trial["trial_id"], "survivorship": SURVIVORSHIP,
        "timings": {"total_s": round(time.perf_counter() - t_start, 3)},
    }
    path = write_run(root, run_id, tables={"search": table}, summary=summary, manifest=manifest)
    return {"run_id": run_id, "path": path, "summary": summary, "manifest": manifest}


# ---------------------------------------------------------------- report
def _pct(value: Any) -> str:
    return "-" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value * 100:+.1f}%"


def _rate(value: Any) -> str:
    return "-" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value * 100:.0f}%"


def format_report(summary: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    lines = []
    if summary.get("schema") == SEARCH_SCHEMA:
        pop = summary["population"]
        lines.append(f"SEARCH {summary['run_id']}: {summary['side']} base={summary['base']} "
                     f"h={summary['horizon']} rules tried {summary['rules_tried']} (scored {summary['rules_scored']})")
        lines.append(f"population train n {pop['train'][0]} win {_rate(pop['train'][1])} avg {_pct(pop['train'][2])}"
                     f" | test n {pop['test'][0]} win {_rate(pop['test'][1])} avg {_pct(pop['test'][2])}")
        lines.append(f"{'rule':52} {'n_tr':>6} {'win_tr':>6} {'avg_tr':>7} {'n_te':>6} {'win_te':>6} {'avg_te':>7}")
        for row in summary["top_by_train"]:
            lines.append(f"{row['rule'][:52]:52} {row['n_train']:>6} {_rate(row['win_train']):>6} "
                         f"{_pct(row['mean_train']):>7} {row['n_test']:>6} {_rate(row['win_test']):>6} "
                         f"{_pct(row['mean_test']):>7}")
        lines.append(f"of the top {len(summary['top_by_train'])} on train, {summary['top_beat_spy_on_test']} beat SPY"
                     f" on test; median test {_pct(summary['top_median_test'])}")
    else:
        lines.append(f"RUN {summary['run_id']}  (straight up, {PRIMARY_HORIZON} sessions, vs SPY)")
        lines.append(f"{'setup':24} {'side':5} {'n':>6} {'win':>5} {'LB':>5} {'avg':>7} {'med':>7} {'R':>6} "
                     f"{'n_test':>6} {'win_te':>6} {'avg_te':>7}")
        for c in summary["straight_up"]:
            lines.append(f"{c['setup'][:24]:24} {c['side']:5} {c['n']:>6} {_rate(c['win_vs_spy']):>5} "
                         f"{_rate(c['wilson_lb']):>5} {_pct(c['mean_vs_spy']):>7} {_pct(c['median_vs_spy']):>7} "
                         f"{'-' if c['mean_R'] is None else format(c['mean_R'], '+.2f'):>6} {c['n_test']:>6} "
                         f"{_rate(c['win_test']):>6} {_pct(c['mean_test']):>7}")
        for key, why in (summary.get("unmeasured_setups") or {}).items():
            lines.append(f"{key[:24]:24} NOT MEASURED: {why}")
        for side, ranked in summary["ranked"].items():
            for which in ("best", "worst"):
                lines.append(f"{side} {which} regime cells (n >= {MIN_CELL_N}):")
                for c in ranked[which]:
                    lines.append(f"  {c['setup']:24} {c['axis']}={c['label']}: n {c['n']} win {_rate(c['win_vs_spy'])} "
                                 f"avg {_pct(c['mean_vs_spy'])} | test n {c['n_test']} avg {_pct(c['mean_test'])}")
    lines.append(f"trials in family {LEDGER_FAMILY} to date: {summary.get('family_trials_to_date')}")
    lines.append("CAVEAT: " + SURVIVORSHIP)
    if manifest:
        t = manifest.get("timings", {})
        lines.append(f"git {manifest.get('git', {}).get('commit', '?')[:10]} data {manifest.get('data_range')} "
                     f"total {t.get('total_s')}s")
    return "\n".join(lines)


# ---------------------------------------------------------------- lake inputs
def load_lake_inputs(start: date | None = None, end: date | None = None, symbols: Sequence[str] | None = None,
                     rule_version: str | None = None, *, store: Any = None) -> Inputs:
    """B1's D1 history + earnings dates and B2's SPY regimes from the lake (read-only). Bars
    load from the lake's first session (indicator warm-up); ``start`` bounds the signals only."""
    from research_warehouse import history_reader, regime_daily

    bars = history_reader.read_d1(list(symbols) if symbols else None, None, end, store=store)
    earnings = history_reader.read_earnings_dates(list(bars), store=store)
    regimes = regime_daily.read_regimes(BENCHMARK, rule_version, store=store, as_of="close")
    flags = history_reader.read_quality_flags("bar_d1_history", list(bars), store=store)
    version = rule_version or (_rule_version(regimes) if regimes is not None and not regimes.empty else None)
    providers: dict[str, int] = {}
    for frame in bars.values():
        if "provider" in frame.columns and len(frame):
            key = str(frame["provider"].iloc[-1])
            providers[key] = providers.get(key, 0) + 1
    return Inputs(bars=bars, earnings=earnings, regimes=regimes, regime_rule_version=version,
                  regime_axes=tuple(regime_daily.AXES), quality_flags=flags,
                  source={"bars": "history_reader.read_d1", "earnings": "history_reader.read_earnings_dates",
                          "regimes": f"regime_daily.read_regimes({BENCHMARK}, as_of=close)",
                          "symbols_by_latest_provider": providers,
                          "symbols_with_earnings_dates": sum(1 for v in earnings.values() if v)})


# ---------------------------------------------------------------- CLI
def _date(text: str | None) -> date | None:
    return date.fromisoformat(text) if text else None


def _root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    from research_warehouse import config

    root = config.get_research_store_dir()
    if root is None:
        raise SystemExit("research_store_dir is unset; pass --root")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research_warehouse backtest", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="replay the registered setups and write an immutable run")
    search = sub.add_parser("search", help="grid-search feature conditions: pick on train, judge on test")
    report = sub.add_parser("report", help="print a run's or search's summary")
    for p in (run, search):
        p.add_argument("--root", default="", help="where the run and its ledger row go (default: the configured lake)")
        p.add_argument("--lake", default="", help="lake to READ bars and regimes from (default: the configured lake)")
        p.add_argument("--start", default="", help="first signal date YYYY-MM-DD")
        p.add_argument("--end", default="", help="last signal date YYYY-MM-DD")
        p.add_argument("--split-date", default=DEFAULT_SPLIT.isoformat(), help="first TEST session")
        p.add_argument("--symbols", default="", help="comma list (default: every lake name)")
        p.add_argument("--rule-version", default="", help="regime rule_version (default: latest)")
        p.add_argument("--run-id", default="")
    run.add_argument("--setups", default="", help="comma list of setup keys (default: all)")
    run.add_argument("--axes", default="", help="comma list of regime axes (default: all label columns)")
    search.add_argument("--side", choices=(bs.LONG, bs.SHORT), required=True)
    search.add_argument("--base", default="trend", help="trend | all | <setup key>")
    search.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    search.add_argument("--max-k", type=int, default=3)
    search.add_argument("--top", type=int, default=25)
    search.add_argument("--regime", default="", help="axis=label: search inside one regime")
    report.add_argument("run", help="run id or run folder")
    report.add_argument("--root", default="")
    args = parser.parse_args(argv)
    if args.command == "report":
        folder = resolve_run(_root(args.root) if args.root or not Path(args.run).is_dir() else None, args.run)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        print(format_report(summary, manifest))
        return 0
    root = _root(args.root)
    from research_warehouse.store import ResearchStore

    lake = ResearchStore(_root(args.lake))  # a plain read handle: no layout creation, no lock
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    if symbols and BENCHMARK not in symbols:
        symbols.append(BENCHMARK)
    t0 = time.perf_counter()
    inputs = load_lake_inputs(_date(args.start), _date(args.end), symbols, args.rule_version or None, store=lake)
    inputs.source["read_s"] = round(time.perf_counter() - t0, 1)
    inputs.source["lake"] = str(lake.root)
    common = {"root": root, "start": _date(args.start), "end": _date(args.end),
              "split": date.fromisoformat(args.split_date), "run_id": args.run_id or None}
    if args.command == "run":
        keys = [k.strip() for k in args.setups.split(",") if k.strip()]
        axes = [a.strip() for a in args.axes.split(",") if a.strip()] or None
        result = run_backtest(inputs, setups=bs.by_key(keys), axes=axes, **common)
    else:
        regime = tuple(args.regime.split("=", 1)) if args.regime else None
        result = run_search(inputs, side=args.side, base=args.base, horizon=args.horizon, max_k=args.max_k,
                            top=args.top, regime=regime, **common)
    print(format_report(result["summary"], result["manifest"]))
    print(f"wrote {result['path']}")
    return 0


__all__ = ["Inputs", "load_lake_inputs", "main", "measure", "run_backtest", "run_search"]

