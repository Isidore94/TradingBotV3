"""Long lab: a point-in-time replay of long-candidate rules on cached daily bars. Shadow only.

For every session in SPY's cached daily history it flags long candidates under
pluggable rule functions, each seeing only bars completed by that session's
close, then measures what happened next:

* raw return and return vs SPY at 5/10/20 sessions, entry at the next open;
* MFE / MAE over the same sessions, in the flag day's ATR(14);
* a limit 0.25 ATR under the flag close, live 3 sessions (`retest_entry.limit_fill`);
* three exit models from the next open: +1 ATR take (1 ATR stop, 20-session cap),
  a 10-session time stop, and a 1 ATR trail off the highest high (20-session cap).

Every candidate is tagged with the market on its day - SPY vs its 20-day SMA
(and whether that SMA is rising), the trader's structural regime
(`regime_join`), and the calendar month - and every number is reported per
regime, never pooled across regimes.

The universe is whatever names have cached bars today: names that were delisted
or dropped from the cache are missing (survivorship bias, named in the report).
Nothing here feeds a live score, alert, tier, gate or Focus list.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from research_warehouse.retest_entry import limit_fill

SCHEMA = "long_lab_v1"
HORIZONS = (5, 10, 20)
SWEEP_HORIZON = 10
ATR_BARS = 14
LIMIT_ATR_FRACTION = 0.25
LIMIT_LIVE_SESSIONS = 3
EXIT_CAP_SESSIONS = 20
TIME_STOP_SESSIONS = 10
TAKE_ATR = 1.0
TRAIL_ATR = 1.0
COOLDOWN_SESSIONS = 10
MIN_PRICE = 5.0
MIN_CELL_N = 30  # evidence_stats.MIN_REPORTABLE_N: thinner cells are shown, never ranked
BENCHMARK = "SPY"
NO_REGIME = "unknown"

AXES = ("spy_trend", "structural", "month")


# ---------------------------------------------------------------- series and indicators
def _rolling(values: np.ndarray, window: int, how: Callable) -> np.ndarray:
    """``how`` over each full trailing window (bars <= i); NaN until ``window`` bars exist."""
    out = np.full(len(values), np.nan)
    if len(values) >= window:
        view = np.lib.stride_tricks.sliding_window_view(values, window)
        with np.errstate(all="ignore"):
            out[window - 1:] = how(view, axis=1)
    return out


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """EMA seeded by the SMA of the first ``span`` bars; NaN before that. Causal."""
    out = np.full(len(values), np.nan)
    if len(values) < span:
        return out
    alpha = 2.0 / (span + 1.0)
    level = float(np.mean(values[:span]))
    out[span - 1] = level
    for i in range(span, len(values)):
        level = alpha * float(values[i]) + (1.0 - alpha) * level
        out[i] = level
    return out


@dataclass
class Series:
    """One name's daily bars (oldest first) plus causal indicator arrays."""

    symbol: str
    dates: list[date]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    volume_unit: list[str]
    index: dict[date, int] = field(init=False)
    cache: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {day: i for i, day in enumerate(self.dates)}
        close, high, low = self.close, self.high, self.low
        prev = np.concatenate(([np.nan], close[:-1]))
        tr = np.where(np.isnan(prev), high - low,
                      np.maximum(high, prev) - np.minimum(low, prev))
        self.atr = _rolling(tr, ATR_BARS, np.mean)
        self.sma20 = _rolling(close, 20, np.mean)
        self.sma50 = _rolling(close, 50, np.mean)
        self.sma200 = _rolling(close, 200, np.mean)
        self.ema21 = _ema(close, 21)
        self.ema50 = _ema(close, 50)
        hh252 = _rolling(high, 252, np.max)
        self.new_52w = np.where(np.isnan(hh252), 0.0, (high >= hh252).astype(float))
        self.run40 = high / _rolling(low, 40, np.min) - 1.0
        self.peak120 = _rolling(high, 120, np.max)
        self.peak120_age = np.full(len(high), np.nan)
        if len(high) >= 120:
            view = np.lib.stride_tricks.sliding_window_view(high, 120)
            self.peak120_age[119:] = 119 - np.argmax(view, axis=1)
        self.any_52w_120 = _rolling(self.new_52w, 120, np.max)
        self.max_run_120 = _rolling(np.nan_to_num(self.run40, nan=-1.0), 120, np.max)
        back = np.concatenate((np.full(63, np.nan), close[:-63])) if len(close) > 63 else np.full(len(close), np.nan)
        self.ret63 = close / back - 1.0

    @classmethod
    def from_rows(cls, symbol: str, rows: Sequence[Mapping[str, Any]]) -> "Series":
        ordered = sorted(rows, key=lambda row: row["date"])
        return cls(
            symbol=symbol,
            dates=[row["date"] for row in ordered],
            open=np.array([float(row["open"]) for row in ordered]),
            high=np.array([float(row["high"]) for row in ordered]),
            low=np.array([float(row["low"]) for row in ordered]),
            close=np.array([float(row["close"]) for row in ordered]),
            volume=np.array([float(row.get("volume") or 0.0) for row in ordered]),
            volume_unit=[str(row.get("volume_unit") or "unknown") for row in ordered],
        )


def _ok(*values: float) -> bool:
    return all(value is not None and math.isfinite(value) for value in values)


# ---------------------------------------------------------------- earnings events
def reaction_day(s: Series, earnings_day: date) -> int | None:
    """The reaction session of one earnings date: of the first session on/after the date
    and the one after it, the one with the larger |open - prior close| (release time is
    not in the dates store). None when either bar or the prior ATR is missing."""
    first = next((i for i, day in enumerate(s.dates) if day >= earnings_day), None)
    if first is None or first < 1 or first + 1 >= len(s.dates):
        return None
    best, best_gap = None, -1.0
    for k in (first, first + 1):
        gap = abs(s.open[k] - s.close[k - 1])
        if gap > best_gap:
            best, best_gap = k, gap
    return best


def gap_atr(s: Series, k: int) -> float | None:
    """(open - prior close) / prior ATR at bar ``k``."""
    if k < 1 or not _ok(float(s.atr[k - 1])) or s.atr[k - 1] <= 0:
        return None
    return float((s.open[k] - s.close[k - 1]) / s.atr[k - 1])


def _earnings_reactions(s: Series, earnings: Mapping[str, Sequence[date]]) -> list[int]:
    if "earn" not in s.cache:
        days = earnings.get(s.symbol) or ()
        found = {k for k in (reaction_day(s, day) for day in days) if k is not None}
        s.cache["earn"] = sorted(found)
    return s.cache["earn"]


def _volume_ratio(s: Series, k: int, lookback: int = 20, min_bars: int = 10) -> float | None:
    """Bar ``k``'s volume over the mean of the prior ``lookback`` bars in the SAME unit
    (the cache mixes share counts and IB lots); None when too few comparable bars."""
    unit = s.volume_unit[k]
    prior = [s.volume[j] for j in range(max(0, k - lookback), k)
             if s.volume_unit[j] == unit and s.volume[j] > 0]
    if len(prior) < min_bars or s.volume[k] <= 0:
        return None
    return float(s.volume[k] / (sum(prior) / len(prior)))


# ---------------------------------------------------------------- rules
@dataclass
class Day:
    """What a rule may read for (series, i): bars <= i, and the day's RS decile."""

    s: Series
    i: int
    rs_decile: int | None
    earnings: Mapping[str, Sequence[date]]
    memo: dict[str, Any] = field(default_factory=dict)


def leader_pullback_features(day: Day) -> dict[str, Any] | None:
    """The loose leader-pullback superset (the sweep's population); None when not one."""
    if "lp" in day.memo:
        return day.memo["lp"]
    s, i = day.s, day.i
    out = None
    close, low = float(s.close[i]), float(s.low[i])
    peak, age = float(s.peak120[i]), float(s.peak120_age[i])
    sma200, ema21, ema50 = float(s.sma200[i]), float(s.ema21[i]), float(s.ema50[i])
    if _ok(peak, age, sma200, ema21, ema50) and age > 0 and close > sma200:
        depth = 1.0 - close / peak
        run = float(s.max_run_120[i])
        made_52w = bool(s.any_52w_120[i] >= 1.0)
        at_ema = low <= ema21 * 1.01 and close >= ema50 * 0.98
        if at_ema and 0.03 <= depth <= 0.40 and (made_52w or run >= 0.15):
            out = {"depth": depth, "run": max(run, 0.0), "made_52w": made_52w,
                   "rs_decile": day.rs_decile}
    day.memo["lp"] = out
    return out


def rule_leader_pullback(day: Day) -> dict[str, Any] | None:
    """52w high or a >=30% run in <=40 sessions within the last 120, now 8-25% off the
    120-session high, above the 200-day, low back to the 21 EMA with the close over the 50."""
    feats = leader_pullback_features(day)
    if feats and 0.08 <= feats["depth"] <= 0.25 and (feats["made_52w"] or feats["run"] >= 0.30):
        return feats
    return None


def _drift(day: Day, gap_days: Iterable[int], label: str) -> dict[str, Any] | None:
    s, i = day.s, day.i
    for g in gap_days:
        since = i - g
        if not 4 <= since <= 7:
            continue
        size = gap_atr(s, g)
        if size is None or size < 1.0 or not s.close[i] > s.close[g - 1]:
            continue
        # First session in the 4-7 window with the gap still open; later ones repeat it.
        if any(4 <= j - g and s.close[j] > s.close[g - 1] for j in range(g + 4, i)):
            continue
        return {"gap_atr": size, "sessions_since_gap": since, "source": label}
    return None


def rule_post_earnings_drift(day: Day) -> dict[str, Any] | None:
    """A >=1 ATR gap up on an earnings reaction day (dates store), flagged on the first of
    sessions 4-7 after it with the close still above the pre-gap close."""
    return _drift(day, _earnings_reactions(day.s, day.earnings), "earnings_dates")


def rule_gap_volume_drift_proxy(day: Day) -> dict[str, Any] | None:
    """PROXY for post-earnings drift: any >=1 ATR gap up on >=2x its 20-session volume,
    same 4-7 session flag. Catches non-earnings gaps too - labelled a proxy everywhere."""
    s, i = day.s, day.i
    if "proxy_gaps" not in s.cache:  # each bar's test reads only that bar and earlier ones
        flags = set()
        for g in range(1, len(s.dates)):
            size = gap_atr(s, g)
            if size is None or size < 1.0:
                continue
            ratio = _volume_ratio(s, g)
            if ratio is not None and ratio >= 2.0:
                flags.add(g)
        s.cache["proxy_gaps"] = flags
    gaps = [g for g in range(max(1, i - 7), i - 3) if g in s.cache["proxy_gaps"]]
    return _drift(day, gaps, "gap_volume_proxy")


def anchored_vwap_bands(s: Series, anchor: int, end: int) -> tuple[float, float] | None:
    """(AVWAP, sigma) from ``anchor`` to ``end`` inclusive - the same running-deviation
    sigma as `master_avwap_lib.legacy.calc_anchored_vwap_bands` (parity-tested)."""
    key = f"avwap:{anchor}"
    if key not in s.cache:
        tp = (s.open[anchor:] + s.high[anchor:] + s.low[anchor:] + s.close[anchor:]) / 4.0
        vol = np.where(s.volume[anchor:] > 0, s.volume[anchor:], 0.0)
        cum_v = np.cumsum(vol)
        cum_vp = np.cumsum(tp * vol)
        with np.errstate(all="ignore"):
            running = np.where(cum_v > 0, cum_vp / np.where(cum_v > 0, cum_v, 1.0), np.nan)
        dev = np.where(vol > 0, tp - running, 0.0)
        cum_sd = np.cumsum(np.nan_to_num(dev * dev * vol))
        s.cache[key] = (cum_v, cum_vp, cum_sd)
    cum_v, cum_vp, cum_sd = s.cache[key]
    k = end - anchor
    if k < 0 or cum_v[k] <= 0:
        return None
    return float(cum_vp[k] / cum_v[k]), float((cum_sd[k] / cum_v[k]) ** 0.5)


def rule_favourite_zone_long(day: Day) -> dict[str, Any] | None:
    """The old favourite-zone long, for contrast: close between the AVWAP anchored the
    session before the latest earnings reaction and its +1 sigma band, 2-63 sessions on."""
    s, i = day.s, day.i
    past = [g for g in _earnings_reactions(s, day.earnings) if g + 1 <= i]
    if not past:
        return None
    g = past[-1]
    if not 2 <= i - g <= 63 or g < 1:
        return None
    bands = anchored_vwap_bands(s, g - 1, i)
    if bands is None:
        return None
    avwap, sigma = bands
    if avwap <= s.close[i] <= avwap + sigma:
        return {"sessions_since_gap": i - g, "avwap": avwap, "sigma": sigma}
    return None


def rule_rising_20_50_baseline(day: Day) -> dict[str, Any] | None:
    """Baseline: close above a rising 20-day and a rising 50-day SMA (rising = above its
    value 5 sessions earlier)."""
    s, i = day.s, day.i
    if i < 5:
        return None
    values = (float(s.sma20[i]), float(s.sma20[i - 5]), float(s.sma50[i]), float(s.sma50[i - 5]))
    if not _ok(*values):
        return None
    c = float(s.close[i])
    if c > values[0] and c > values[2] and values[0] > values[1] and values[2] > values[3]:
        return {}
    return None


@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    fn: Callable[[Day], dict[str, Any] | None]
    note: str = ""


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("leader_pullback", "(a) Leader pullback", rule_leader_pullback),
    Rule("post_earnings_drift", "(b) Post-earnings drift", rule_post_earnings_drift,
         "earnings dates from the dates store; reaction day = the larger gap of the date and the next session"),
    Rule("gap_volume_drift_proxy", "(b') Gap + 2x volume drift (PROXY)", rule_gap_volume_drift_proxy,
         "proxy: any >=1 ATR gap on >=2x volume, not only earnings"),
    Rule("favourite_zone_long", "(c) Favourite zone long (old)", rule_favourite_zone_long,
         "AVWAP anchored before the latest earnings reaction; approximates the scanner's anchor"),
    Rule("rising_20_50_baseline", "(d) Above rising 20 and 50 (baseline)", rule_rising_20_50_baseline),
)


# ---------------------------------------------------------------- market state
def spy_trend_labels(spy: Series) -> dict[date, str]:
    """SPY close vs its 20-day SMA on each day; the SMA is rising when above its value
    5 sessions earlier. Missing history is ``unknown``."""
    out: dict[date, str] = {}
    for i, day in enumerate(spy.dates):
        sma, prior = float(spy.sma20[i]), float(spy.sma20[i - 5]) if i >= 5 else float("nan")
        if not _ok(sma, prior):
            out[day] = NO_REGIME
        elif spy.close[i] <= sma:
            out[day] = "below_20d"
        else:
            out[day] = "above_rising_20d" if sma > prior else "above_falling_20d"
    return out


def rs_deciles(series: Mapping[str, Series], sessions: Iterable[date]) -> dict[tuple[str, date], int]:
    """Per session, each name's 63-session return decile (10 = strongest) among names with
    a bar that day. Ranking a name's own return equals ranking its return vs SPY."""
    wanted = set(sessions)
    by_day: dict[date, list[tuple[float, str]]] = {}
    for symbol, s in series.items():
        for i, day in enumerate(s.dates):
            if day in wanted and math.isfinite(s.ret63[i]):
                by_day.setdefault(day, []).append((float(s.ret63[i]), symbol))
    out: dict[tuple[str, date], int] = {}
    for day, values in by_day.items():
        values.sort()
        count = len(values)
        for rank, (_value, symbol) in enumerate(values):
            out[(symbol, day)] = min(10, int(rank * 10 / count) + 1)
    return out


# ---------------------------------------------------------------- outcomes
def _exit_take(s: Series, entry: float, atr: float, first: int, last: int) -> float:
    target, stop = entry + TAKE_ATR * atr, entry - TAKE_ATR * atr
    for j in range(first, last + 1):
        if j > first and s.open[j] <= stop:
            return float(s.open[j])
        if j > first and s.open[j] >= target:
            return float(s.open[j])
        if s.low[j] <= stop:  # a bar touching both is a stop
            return stop
        if s.high[j] >= target:
            return target
    return float(s.close[last])


def _exit_trail(s: Series, entry: float, atr: float, first: int, last: int) -> float:
    stop = entry - TRAIL_ATR * atr
    for j in range(first, last + 1):
        if j > first and s.open[j] <= stop:
            return float(s.open[j])
        if s.low[j] <= stop:
            return stop
        stop = max(stop, float(s.high[j]) - TRAIL_ATR * atr)  # raised only after the bar completes
    return float(s.close[last])


def measure(s: Series, i: int, spy: Series) -> dict[str, Any]:
    """Outcomes of a flag at the close of bar ``i``. A horizon whose bars do not exist yet
    is pending (absent), never zero."""
    out: dict[str, Any] = {}
    atr = float(s.atr[i])
    if i + 1 >= len(s.dates) or not _ok(atr) or atr <= 0:
        return out
    entry = float(s.open[i + 1])
    out["entry"] = entry
    spy_open = spy.index.get(s.dates[i + 1])
    for h in HORIZONS:
        end = i + h
        if end >= len(s.dates):
            continue
        raw = float(s.close[end]) / entry - 1.0
        cell = {
            "raw": raw,
            "mfe_atr": (float(np.max(s.high[i + 1:end + 1])) - entry) / atr,
            "mae_atr": (entry - float(np.min(s.low[i + 1:end + 1]))) / atr,
        }
        spy_end = spy.index.get(s.dates[end])
        if spy_open is not None and spy_end is not None:
            cell["vs_spy"] = raw - (float(spy.close[spy_end]) / float(spy.open[spy_open]) - 1.0)
        out[f"h{h}"] = cell
    limit = float(s.close[i]) - LIMIT_ATR_FRACTION * atr
    fill_at = None
    for j in range(i + 1, min(i + 1 + LIMIT_LIVE_SESSIONS, len(s.dates))):
        price = limit_fill({"open": s.open[j], "high": s.high[j], "low": s.low[j]}, limit, True)
        if price is not None:
            fill_at = (j, price)
            break
    if fill_at is None and i + LIMIT_LIVE_SESSIONS < len(s.dates):
        out["limit"] = {"filled": False}
    elif fill_at is not None:
        j, price = fill_at
        out["limit"] = {"filled": True, "fill": price,
                        **{f"h{h}": float(s.close[i + h]) / price - 1.0
                           for h in HORIZONS if j <= i + h < len(s.dates)}}
    exits: dict[str, float] = {}
    if i + TIME_STOP_SESSIONS < len(s.dates):
        exits["time_10"] = (float(s.close[i + TIME_STOP_SESSIONS]) - entry) / atr
    if i + EXIT_CAP_SESSIONS < len(s.dates):
        last = i + EXIT_CAP_SESSIONS
        exits["take_1atr"] = (_exit_take(s, entry, atr, i + 1, last) - entry) / atr
        exits["trail_1atr"] = (_exit_trail(s, entry, atr, i + 1, last) - entry) / atr
    out["exits"] = exits
    return out


# ---------------------------------------------------------------- the replay
def find_candidates(
    series: Mapping[str, Series],
    sessions: Sequence[date],
    *,
    rules: Sequence[Rule] = DEFAULT_RULES,
    earnings: Mapping[str, Sequence[date]] | None = None,
    extra_loose: bool = True,
) -> list[dict[str, Any]]:
    """Every flag of every rule, point in time. A name flags a rule at most once per
    ``COOLDOWN_SESSIONS`` of its own bars. Also yields the loose leader-pullback superset
    (rule ``leader_pullback_loose``) for the threshold sweep."""
    earnings = earnings or {}
    wanted = set(sessions)
    deciles = rs_deciles(series, sessions)
    out: list[dict[str, Any]] = []
    for symbol in sorted(series):
        if symbol == BENCHMARK:
            continue
        s = series[symbol]
        last_flag: dict[str, int] = {}
        for i, day in enumerate(s.dates):
            if day not in wanted or s.close[i] < MIN_PRICE:
                continue
            ctx = Day(s, i, deciles.get((symbol, day)), earnings)
            checks = [(rule.key, rule.fn) for rule in rules]
            if extra_loose:
                checks.append(("leader_pullback_loose", leader_pullback_features))
            for key, fn in checks:
                if i - last_flag.get(key, -10**9) < COOLDOWN_SESSIONS:
                    continue
                feats = fn(ctx)
                if feats is None:
                    continue
                last_flag[key] = i
                out.append({"rule": key, "symbol": symbol, "date": day, "i": i,
                            "features": dict(feats)})
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _wilson(wins: int, n: int) -> float | None:
    from swing_headline import wilson_lower_bound

    return wilson_lower_bound(wins, n)


def _horizon_cell(rows: list[dict[str, Any]], h: int) -> dict[str, Any]:
    key = f"h{h}"
    measured = [row["outcome"][key] for row in rows if key in row["outcome"]]
    raw = [cell["raw"] for cell in measured]
    vs = [cell["vs_spy"] for cell in measured if "vs_spy" in cell]
    wins = sum(1 for value in raw if value > 0)
    limits = [row["outcome"]["limit"] for row in rows if "limit" in row["outcome"]]
    filled = [lim for lim in limits if lim.get("filled")]
    limit_raw = [lim[key] for lim in filled if key in lim]
    return {
        "n": len(measured),
        "pending": len(rows) - len(measured),
        "symbols": len({row["symbol"] for row in rows if key in row["outcome"]}),
        "win_raw": wins / len(raw) if raw else None,
        "wilson_lb": _wilson(wins, len(raw)),
        "n_vs_spy": len(vs),
        "win_vs_spy": sum(1 for value in vs if value > 0) / len(vs) if vs else None,
        "mean_raw": _mean(raw), "median_raw": _median(raw),
        "mean_vs_spy": _mean(vs), "median_vs_spy": _median(vs),
        "mfe_atr": _median([cell["mfe_atr"] for cell in measured]),
        "mae_atr": _median([cell["mae_atr"] for cell in measured]),
        "limit_fill_rate": len(filled) / len(limits) if limits else None,
        "limit_mean_raw": _mean(limit_raw),
        "thin": len(measured) < MIN_CELL_N,
    }


EXIT_MODELS = ("take_1atr", "time_10", "trail_1atr")


def _exit_cells(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    cells = []
    for model in EXIT_MODELS:
        values = [row["outcome"]["exits"][model] for row in rows
                  if model in row["outcome"].get("exits", {})]
        cells.append({"model": model, "n": len(values), "mean_atr": _mean(values),
                      "median_atr": _median(values),
                      "win": sum(1 for v in values if v > 0) / len(values) if values else None})
    ranked = [cell for cell in cells if cell["mean_atr"] is not None and cell["n"] >= MIN_CELL_N]
    best = max(ranked, key=lambda cell: cell["mean_atr"])["model"] if ranked else None
    return cells, best


DEPTH_BUCKETS = ((0.03, 0.08), (0.08, 0.12), (0.12, 0.16), (0.16, 0.20), (0.20, 0.25), (0.25, 0.40))
RUN_BUCKETS = ((0.0, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.80), (0.80, float("inf")))
RS_BUCKETS = ((1, 3), (4, 6), (7, 8), (9, 10))


def _bucket(value, buckets) -> str | None:
    if value is None:
        return None
    for low, high in buckets:
        if isinstance(low, int) and isinstance(high, int):
            if low <= value <= high:
                return f"{low}-{high}"
        elif low <= value < high or (high == buckets[-1][1] and value == high):
            top = "+" if math.isinf(high) else f"{high * 100:.0f}%"
            return f"{low * 100:.0f}%-{top}"
    return None


def _sweep_keys(feats: Mapping[str, Any]) -> list[tuple[str, str]]:
    keys = [("pullback_depth", _bucket(feats.get("depth"), DEPTH_BUCKETS)),
            ("run_size", _bucket(feats.get("run"), RUN_BUCKETS)),
            ("made_52w_high", "yes" if feats.get("made_52w") else "no"),
            ("rs_decile", _bucket(feats.get("rs_decile"), RS_BUCKETS))]
    return [(knob, bucket) for knob, bucket in keys if bucket is not None]


def build_report(
    candidates: list[dict[str, Any]],
    series: Mapping[str, Series],
    sessions: Sequence[date],
    *,
    segments: Sequence[Mapping[str, Any]] | None = None,
    rules: Sequence[Rule] = DEFAULT_RULES,
    now: datetime | None = None,
    universe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure every candidate and aggregate per rule x axis x regime (x horizon)."""
    import regime_join

    spy = series[BENCHMARK]
    trend = spy_trend_labels(spy)
    joiner = regime_join.Joiner(segments or ())
    for row in candidates:
        row["outcome"] = measure(series[row["symbol"]], row["i"], spy)
        row["regimes"] = {
            "spy_trend": trend.get(row["date"], NO_REGIME),
            "structural": joiner.label(row["date"]) if joiner else NO_REGIME,
            "month": row["date"].strftime("%Y-%m"),
        }
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        for axis in AXES:
            groups.setdefault((row["rule"], axis, row["regimes"][axis]), []).append(row)
    cells, exits = [], []
    for (rule, axis, regime), rows in sorted(groups.items()):
        for h in HORIZONS:
            cells.append({"rule": rule, "axis": axis, "regime": regime, "horizon": h,
                          **_horizon_cell(rows, h)})
        models, best = _exit_cells(rows)
        for model in models:
            exits.append({"rule": rule, "axis": axis, "regime": regime, **model,
                          "best": model["model"] == best})
        for cell in cells[-len(HORIZONS):]:
            cell["best_exit"] = best
    sweep_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        if row["rule"] != "leader_pullback_loose":
            continue
        for knob, bucket in _sweep_keys(row["features"]):
            for axis in ("spy_trend", "structural"):
                sweep_groups.setdefault((knob, bucket, axis, row["regimes"][axis]), []).append(row)
    sweep = []
    for (knob, bucket, axis, regime), rows in sorted(sweep_groups.items()):
        cell = _horizon_cell(rows, SWEEP_HORIZON)
        sweep.append({"knob": knob, "bucket": bucket, "axis": axis, "regime": regime,
                      "horizon": SWEEP_HORIZON, "n": cell["n"], "win_raw": cell["win_raw"],
                      "win_vs_spy": cell["win_vs_spy"], "mean_raw": cell["mean_raw"],
                      "mean_vs_spy": cell["mean_vs_spy"], "thin": cell["thin"]})
    spreads: dict[tuple[str, str, str], list[float]] = {}
    for cell in sweep:
        if not cell["thin"] and cell["mean_vs_spy"] is not None:
            spreads.setdefault((cell["knob"], cell["axis"], cell["regime"]), []).append(cell["mean_vs_spy"])
    knob_spread = [
        {"knob": knob, "axis": axis, "regime": regime, "buckets": len(values),
         "spread_vs_spy": max(values) - min(values)}
        for (knob, axis, regime), values in sorted(spreads.items()) if len(values) >= 2
    ]
    counts: dict[str, int] = {}
    for row in candidates:
        counts[row["rule"]] = counts.get(row["rule"], 0) + 1
    rule_meta = [{"key": rule.key, "label": rule.label, "doc": (rule.fn.__doc__ or "").strip(),
                  "note": rule.note, "candidates": counts.get(rule.key, 0)} for rule in rules]
    rule_meta.append({"key": "leader_pullback_loose", "label": "(a) loose superset, sweep only",
                      "doc": (leader_pullback_features.__doc__ or "").strip(),
                      "note": "depth 3-40%, run >=15% or 52w high; the sweep buckets this set",
                      "candidates": counts.get("leader_pullback_loose", 0)})
    return {
        "schema": SCHEMA,
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "first_session": sessions[0].isoformat() if sessions else None,
        "last_session": sessions[-1].isoformat() if sessions else None,
        "sessions": len(sessions),
        "universe": dict(universe or {}),
        "rules": rule_meta,
        "cells": cells,
        "exits": exits,
        "sweep": sweep,
        "knob_spread": knob_spread,
        "params": {
            "horizons": list(HORIZONS), "entry": "next session open", "atr_bars": ATR_BARS,
            "limit": f"{LIMIT_ATR_FRACTION} ATR under the flag close, live {LIMIT_LIVE_SESSIONS} sessions",
            "exits": {"take_1atr": "+1 ATR target / -1 ATR stop, else close of session 20",
                      "time_10": "close of session 10",
                      "trail_1atr": "stop 1 ATR under the highest high (start entry - 1 ATR), else close of session 20"},
            "cooldown_sessions": COOLDOWN_SESSIONS, "min_price": MIN_PRICE,
            "min_cell_n": MIN_CELL_N, "sweep_horizon": SWEEP_HORIZON,
            "spy_trend": "SPY close vs its 20-day SMA; rising = above its value 5 sessions earlier",
        },
        "caveats": [
            "Survivorship bias: the universe is the names with cached bars today; "
            "delisted or dropped names are missing, which flatters every rule.",
            "Per regime only: no number pools two regimes. Cells under "
            f"{MIN_CELL_N} are marked thin and never pick a best exit.",
            "Point in time: a flag uses bars up to its close only; outcomes use later bars. "
            "A horizon not yet complete is pending, never zero.",
            "(b') is a proxy (gap + volume), not earnings. (c) approximates the scanner's "
            "earnings anchor from the dates store.",
            "Shadow research: nothing here changes an alert, score, grade or list.",
        ],
    }


# ---------------------------------------------------------------- inputs (IO: worker / CLI only)
def load_bars_dir(directory: Path, *, min_date: date | None = None) -> dict[str, Series]:
    """Every ``<SYMBOL>.csv`` (datetime,open,high,low,close,volume,...,volume_unit).
    A malformed row is skipped; an unreadable file skips the name."""
    out: dict[str, Series] = {}
    for path in sorted(Path(directory).glob("*.csv")):
        rows = []
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    try:
                        day = date.fromisoformat(str(raw.get("datetime") or raw.get("date"))[:10])
                        row = {"date": day, "open": float(raw["open"]), "high": float(raw["high"]),
                               "low": float(raw["low"]), "close": float(raw["close"]),
                               "volume": float(raw.get("volume") or 0.0),
                               "volume_unit": raw.get("volume_unit") or "unknown"}
                    except (TypeError, ValueError, KeyError):
                        continue
                    if all(math.isfinite(row[k]) and row[k] > 0 for k in ("open", "high", "low", "close")):
                        rows.append(row)
        except OSError:
            continue
        dedup = {row["date"]: row for row in rows}
        if dedup:
            out[path.stem.upper()] = Series.from_rows(path.stem.upper(), list(dedup.values()))
    return out


def load_earnings_dates(path: Path | None) -> dict[str, list[date]]:
    """``{symbol: [dates]}`` from the earnings-dates cache; ``{}`` when absent."""
    if path is None or not Path(path).is_file():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, list[date]] = {}
    for symbol, entry in (payload.get("symbols") or {}).items():
        days = []
        for text in (entry or {}).get("dates") or ():
            try:
                days.append(date.fromisoformat(str(text)[:10]))
            except ValueError:
                continue
        if days:
            out[str(symbol).upper()] = sorted(set(days))
    return out


def run_lab(
    series: Mapping[str, Series],
    *,
    earnings: Mapping[str, Sequence[date]] | None = None,
    segments: Sequence[Mapping[str, Any]] | None = None,
    rules: Sequence[Rule] = DEFAULT_RULES,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Replay every SPY session in ``series``; the report dict."""
    if BENCHMARK not in series:
        return {"schema": SCHEMA, "error": "no SPY bars", "cells": [], "exits": [], "sweep": []}
    sessions = list(series[BENCHMARK].dates)
    earnings = earnings or {}
    candidates = find_candidates(series, sessions, rules=rules, earnings=earnings)
    in_window = sum(1 for sym, s in series.items() if sym != BENCHMARK and s.dates
                    and s.dates[-1] >= sessions[0])
    universe = {"symbols_loaded": len(series), "symbols_in_window": in_window,
                "symbols_with_earnings_dates": sum(1 for sym in series if earnings.get(sym))}
    return build_report(candidates, series, sessions, segments=segments, rules=rules,
                        now=now, universe=universe)


def lab_from_live_inputs(*, bars_dir: Path | None = None, earnings_path: Path | None = None,
                         journal_path: Path | None = None) -> dict[str, Any]:
    """Reads the cached daily bars, the earnings-dates cache and the structural regime
    (all read-only) and runs the lab. IO: never on the Qt thread."""
    import project_paths
    import regime_join

    series = load_bars_dir(bars_dir or project_paths.DAILY_BARS_CACHE_DIR)
    earnings = load_earnings_dates(earnings_path or project_paths.EARNINGS_DATES_CACHE_FILE)
    segments = regime_join.read_segments(journal_path)
    return run_lab(series, earnings=earnings, segments=segments)


def write_report(report: Mapping[str, Any], path: Path | None = None) -> Path:
    """The one writer of the long-lab report (atomic; a failure keeps the last good file)."""
    import project_paths
    from diagnostics.artifact_io import atomic_write_json

    return atomic_write_json(path or project_paths.LONG_LAB_REPORT_FILE, report, indent=1)


def read_report(path: Path | None = None) -> dict[str, Any] | None:
    """The last written report, or None. IO: never on the Qt thread."""
    import project_paths

    target = Path(path or project_paths.LONG_LAB_REPORT_FILE)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bars-dir", type=Path, default=None)
    parser.add_argument("--earnings", type=Path, default=None)
    parser.add_argument("--journal", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="report path (default: the live report file)")
    args = parser.parse_args(argv)
    report = lab_from_live_inputs(bars_dir=args.bars_dir, earnings_path=args.earnings,
                                  journal_path=args.journal)
    path = write_report(report, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
