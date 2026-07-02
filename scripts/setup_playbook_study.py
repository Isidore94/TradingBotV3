#!/usr/bin/env python3
"""Setup playbook study: hypothesize setup families and backfill-measure them.

The live setup tracker can only *confirm* setups the scan already promotes.
This module goes the other way: it defines a playbook of candidate setup
families (AVWAP band behavior, moving-average pullbacks/reclaims, structure
breaks, volume events, earnings-anchored patterns), detects them over the
durable daily-bar store for the last N sessions, and measures every episode
with the tracker's outcome discipline so families are comparable in R:

- entry at the NEXT session's open (no same-bar fills),
- a representative stop under the signal bar (0.1 ATR beyond its extreme),
- intrabar stop-first (a bar that touches the stop is a stop-out, gaps fill
  at the open beyond the stop),
- time stop at ``TRACKER_MAX_HOLD_DAYS`` sessions,
- net of the tracker's round-trip cost model,
- one open episode per (symbol, family, side): overlapping signals are
  ignored until the prior episode resolves (episode de-dup).

Shorts are detected by mirroring the price frame (negate prices, swap
high/low), so every family is written once in long form and measured on the
original prices with the side inverted. A ``baseline_every5`` control family
(unconditional entry on every 5th session per symbol) anchors the results:
a family is only interesting if it beats that.

Detection at signal date T uses only data through T, with AVWAP bands from
the earnings anchor as it would have been chosen ON T
(``pick_current_earnings_anchor_for_reference_date``), so backfilled signals
match what the live scan would have seen. Caveat: universe membership is
today's (survivorship), and band families need the symbol in the earnings
cache (~90% coverage).

Run:
    .venv/Scripts/python.exe scripts/setup_playbook_study.py --days 60
    .venv/Scripts/python.exe scripts/setup_playbook_study.py --days 60 --symbols NVDA,AMD
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import OUTPUT_DIR  # noqa: E402
from master_avwap_lib.legacy import (  # noqa: E402
    MASTER_AVWAP_DAILY_BARS_DIR,
    TRACKER_COST_COMMISSION_PER_SHARE,
    TRACKER_MAX_HOLD_DAYS,
    TRACKER_SLIPPAGE_FRACTION_PER_SIDE,
    _WINDOWS_RESERVED_FILENAME_STEMS,
    calc_anchored_vwap_band_history,
    compute_indicator_frame,
    load_earnings_date_cache,
    pick_current_earnings_anchor_for_reference_date,
)

PLAYBOOK_EPISODES_CSV = OUTPUT_DIR / "reports" / "setup_playbook_episodes.csv"
PLAYBOOK_LEADERBOARD_CSV = OUTPUT_DIR / "reports" / "setup_playbook_leaderboard.csv"
PLAYBOOK_CONTEXT_CSV = OUTPUT_DIR / "reports" / "setup_playbook_context.csv"
PLAYBOOK_REPORT_TXT = OUTPUT_DIR / "reports" / "setup_playbook_report.txt"
PLAYBOOK_AI_DIGEST_JSON = OUTPUT_DIR / "reports" / "setup_playbook_ai_digest.json"

STOP_ATR_BUFFER = 0.10
TAG_ATR_TOLERANCE = 0.15
HORIZON_MARK_SESSIONS = (5, 10)
MIN_BARS_REQUIRED = 80
MIN_PRICE = 3.0
LEADERBOARD_MIN_CLOSED = 8

# Weekly-chart context: signed streak of COMPLETED weeks closing above (+) or
# below (-) the weekly 8EMA, evaluated with no lookahead (the week containing
# the signal day is excluded). |streak| >= 5 marks the regime the user trades
# around ("super strong above the 8EMA for 5-10 weeks", or the weak mirror).
WEEKLY_STREAK_STRONG_WEEKS = 5
WEEKLY_STREAK_MIN_HISTORY_WEEKS = 10
# Minimum closed episodes before the AI digest treats a combo as evidence.
DIGEST_MIN_CLOSED = 30
DIGEST_ACTIONABLE_MIN_EDGE = 0.15


# ---------------------------------------------------------------------------
# Per-symbol context (one side view)
# ---------------------------------------------------------------------------
@dataclass
class SymbolContext:
    """Everything a detector may look at, arrays aligned to the bar index.

    Band arrays hold the *as-of* values: ``vwap[i]`` is the anchored VWAP a
    live scan on day i would have used (anchor chosen for day i, cumulated
    through day i). ``band_asof(i, j)`` exposes the day-j value under day-i's
    anchor for streak conditions, matching how the live scan replays band
    history under the current anchor.
    """

    symbol: str
    dates: list[date]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    ema8: np.ndarray
    ema15: np.ndarray
    ema21: np.ndarray
    sma50: np.ndarray
    sma200: np.ndarray
    atr: np.ndarray
    vwap: np.ndarray
    upper1: np.ndarray
    upper2: np.ndarray
    anchor_idx: np.ndarray  # bar index of day-i's anchor start (-1 if none)
    mirrored: bool = False
    earnings_session_idx: list[int] = field(default_factory=list)  # bar idx at/after each earnings date
    _anchor_key: list[str] = field(default_factory=list)
    _anchor_hist: dict[str, dict[str, dict]] = field(default_factory=dict)

    def band_asof(self, i: int, j: int, label: str) -> float:
        """Day-j band value under day-i's anchor, in this context's (possibly
        mirrored) coordinates: ask for UPPER_1 and a short context transparently
        serves the negated LOWER_1. NaN when unavailable."""
        key = self._anchor_key[i] if 0 <= i < len(self._anchor_key) else ""
        if not key or j < 0 or j >= len(self.dates):
            return float("nan")
        entry = self._anchor_hist.get(key, {}).get(self.dates[j].isoformat())
        if not entry:
            return float("nan")
        sign = -1.0 if self.mirrored else 1.0
        if self.mirrored and label != "VWAP":
            label = label.replace("UPPER", "LOWER") if "UPPER" in label else label.replace("LOWER", "UPPER")
        raw = entry.get("vwap") if label == "VWAP" else entry.get("bands", {}).get(label)
        return sign * float(raw) if raw is not None else float("nan")

    def sessions_since_last_earnings(self, i: int) -> int | None:
        last = None
        for idx in self.earnings_session_idx:
            if idx <= i:
                last = idx
            else:
                break
        return (i - last) if last is not None else None

    def last_earnings_session(self, i: int) -> int | None:
        since = self.sessions_since_last_earnings(i)
        return (i - since) if since is not None else None


def compute_weekly_streak_series(df: pd.DataFrame) -> np.ndarray:
    """Per daily bar: signed completed-week streak vs the weekly 8EMA.

    +N  = the last N completed weekly candles all closed at/above the weekly
          8EMA; -N = all closed below. The week containing each daily bar is
          in progress and therefore excluded (no lookahead). 0 while the EMA
          is still warming up (< WEEKLY_STREAK_MIN_HISTORY_WEEKS completed).
    """
    n = len(df)
    out = np.zeros(n, dtype=int)
    if n == 0:
        return out
    when = pd.to_datetime(df["datetime"])
    weeks = when.dt.to_period("W-FRI")
    weekly_close = pd.Series(df["close"].to_numpy(dtype=float), index=weeks.values).groupby(level=0).last()
    ema8 = weekly_close.ewm(span=8, adjust=False).mean()

    streak_by_week: dict = {}
    streak = 0
    for week, wclose, wema in zip(weekly_close.index, weekly_close.values, ema8.values):
        if wclose >= wema:
            streak = streak + 1 if streak >= 0 else 1
        else:
            streak = streak - 1 if streak <= 0 else -1
        streak_by_week[week] = streak

    week_list = list(weekly_close.index)
    week_position = {week: idx for idx, week in enumerate(week_list)}
    for i in range(n):
        pos = week_position.get(weeks.iloc[i])
        if pos is None or pos - 1 < WEEKLY_STREAK_MIN_HISTORY_WEEKS:
            continue
        out[i] = streak_by_week[week_list[pos - 1]]
    return out


def classify_weekly_context(streak: int) -> str:
    if streak >= WEEKLY_STREAK_STRONG_WEEKS:
        return "weekly_strong"
    if streak <= -WEEKLY_STREAK_STRONG_WEEKS:
        return "weekly_weak"
    return "weekly_mixed"


def _mirror_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Negate prices and swap high/low so short setups read as long setups.

    Anchored VWAP/stdev math commutes with this transform (mirrored UPPER_k
    equals the negated original LOWER_k), as do SMAs/EMAs, so one long-form
    detector serves both sides.
    """
    out = df.copy()
    out["open"] = -df["open"]
    out["close"] = -df["close"]
    out["high"] = -df["low"]
    out["low"] = -df["high"]
    return out


def _first_session_on_or_after(dates: list[date], target: date) -> int | None:
    for idx, value in enumerate(dates):
        if value >= target:
            return idx
    return None


def build_symbol_context(
    symbol: str,
    df: pd.DataFrame,
    earnings_dates: list[str],
    *,
    scan_start_idx: int,
    mirrored: bool = False,
) -> SymbolContext:
    frame = _mirror_frame(df) if mirrored else df
    ind = compute_indicator_frame(frame)
    dates = [d.date() if hasattr(d, "date") else d for d in pd.to_datetime(frame["datetime"])]
    n = len(frame)

    ctx = SymbolContext(
        symbol=symbol,
        dates=dates,
        open=frame["open"].to_numpy(dtype=float),
        high=frame["high"].to_numpy(dtype=float),
        low=frame["low"].to_numpy(dtype=float),
        close=frame["close"].to_numpy(dtype=float),
        volume=frame["volume"].to_numpy(dtype=float),
        ema8=ind["ema_8"].to_numpy(dtype=float),
        ema15=ind["ema_15"].to_numpy(dtype=float),
        ema21=ind["ema_21"].to_numpy(dtype=float),
        sma50=ind["sma_50"].to_numpy(dtype=float),
        sma200=ind["sma_200"].to_numpy(dtype=float),
        atr=ind["atr_20"].to_numpy(dtype=float),
        vwap=np.full(n, np.nan),
        upper1=np.full(n, np.nan),
        upper2=np.full(n, np.nan),
        anchor_idx=np.full(n, -1, dtype=int),
        mirrored=mirrored,
        _anchor_key=[""] * n,
    )

    for iso in sorted({str(v) for v in earnings_dates or []}):
        parsed = datetime.fromisoformat(iso).date() if iso else None
        idx = _first_session_on_or_after(dates, parsed) if parsed else None
        if idx is not None:
            ctx.earnings_session_idx.append(idx)
    ctx.earnings_session_idx.sort()

    # Resolve the as-of anchor per scan day; one band history per distinct anchor.
    label = "LOWER" if mirrored else "UPPER"
    for i in range(max(scan_start_idx, 0), n):
        anchor = pick_current_earnings_anchor_for_reference_date(earnings_dates or [], dates[i])
        if anchor is None:
            continue
        anchor_session = _first_session_on_or_after(dates, anchor)
        if anchor_session is None or anchor_session >= i:
            continue
        key = dates[anchor_session].isoformat()
        if key not in ctx._anchor_hist:
            ctx._anchor_hist[key] = calc_anchored_vwap_band_history(df.reset_index(drop=True), key)
        entry = ctx._anchor_hist[key].get(dates[i].isoformat())
        ctx._anchor_key[i] = key
        ctx.anchor_idx[i] = anchor_session
        if entry:
            sign = -1.0 if mirrored else 1.0
            ctx.vwap[i] = sign * float(entry["vwap"])
            ctx.upper1[i] = sign * float(entry["bands"][f"{label}_1"])
            ctx.upper2[i] = sign * float(entry["bands"][f"{label}_2"])
    return ctx


# ---------------------------------------------------------------------------
# Detectors (long form; i is the signal bar; data through i only)
# ---------------------------------------------------------------------------
def _tag_and_hold(ctx: SymbolContext, i: int, level: float) -> bool:
    """Low tags/pierces the level, close recovers above it."""
    if not np.isfinite(level):
        return False
    return ctx.low[i] <= level + TAG_ATR_TOLERANCE * ctx.atr[i] and ctx.close[i] > level


def _crossed_up(ctx: SymbolContext, i: int, level_now: float, level_prev: float) -> bool:
    if not (np.isfinite(level_now) and np.isfinite(level_prev)):
        return False
    return ctx.close[i] > level_now and ctx.close[i - 1] <= level_prev


def detect_vwap_reclaim(ctx: SymbolContext, i: int) -> bool:
    return _crossed_up(ctx, i, ctx.vwap[i], ctx.band_asof(i, i - 1, "VWAP"))


def detect_vwap_bounce(ctx: SymbolContext, i: int) -> bool:
    prev_vwap = ctx.band_asof(i, i - 1, "VWAP")
    if not np.isfinite(prev_vwap) or ctx.close[i - 1] <= prev_vwap:
        return False
    return _tag_and_hold(ctx, i, ctx.vwap[i]) and ctx.low[i] > ctx.vwap[i] - 0.75 * ctx.atr[i]


def detect_first_dev_bounce(ctx: SymbolContext, i: int) -> bool:
    prev_u1 = ctx.band_asof(i, i - 1, "UPPER_1")
    if not np.isfinite(prev_u1) or ctx.close[i - 1] < prev_u1:
        return False
    return _tag_and_hold(ctx, i, ctx.upper1[i])


def detect_first_dev_breakout(ctx: SymbolContext, i: int) -> bool:
    return _crossed_up(ctx, i, ctx.upper1[i], ctx.band_asof(i, i - 1, "UPPER_1"))


def detect_second_dev_breakout(ctx: SymbolContext, i: int) -> bool:
    return _crossed_up(ctx, i, ctx.upper2[i], ctx.band_asof(i, i - 1, "UPPER_2"))


def detect_second_dev_power_hold(ctx: SymbolContext, i: int) -> bool:
    if i < 12 or not np.isfinite(ctx.upper2[i]) or ctx.close[i] <= ctx.upper2[i]:
        return False
    held = sum(
        1
        for j in range(i - 11, i + 1)
        if np.isfinite(u2 := ctx.band_asof(i, j, "UPPER_2")) and ctx.close[j] > u2
    )
    return held >= 10


def detect_power_trend_pullback(ctx: SymbolContext, i: int) -> bool:
    if i < 16:
        return False
    above = sum(
        1
        for j in range(i - 15, i)
        if np.isfinite(u1 := ctx.band_asof(i, j, "UPPER_1")) and ctx.close[j] > u1
    )
    if above < 10:
        return False
    level = max(ctx.ema8[i], ctx.upper1[i]) if np.isfinite(ctx.upper1[i]) else ctx.ema8[i]
    return _tag_and_hold(ctx, i, level)


def detect_band_test_rebound(ctx: SymbolContext, i: int) -> bool:
    if i < 11:
        return False
    tested = any(
        np.isfinite(u2 := ctx.band_asof(i, k, "UPPER_2"))
        and ctx.high[k] >= u2 - TAG_ATR_TOLERANCE * ctx.atr[i]
        for k in range(i - 10, i - 4)
    )
    if not tested:
        return False
    held = all(
        np.isfinite(u1 := ctx.band_asof(i, j, "UPPER_1")) and ctx.close[j] > u1
        for j in range(i - 4, i)
    )
    return held and ctx.close[i] > ctx.high[i - 1]


def detect_ema8_pullback_uptrend(ctx: SymbolContext, i: int) -> bool:
    if i < 10:
        return False
    above = sum(1 for j in range(i - 10, i) if ctx.close[j] > ctx.ema8[j])
    return above >= 8 and _tag_and_hold(ctx, i, ctx.ema8[i])


def detect_ema21_pullback_uptrend(ctx: SymbolContext, i: int) -> bool:
    if not (np.isfinite(ctx.sma50[i]) and np.isfinite(ctx.sma200[i])):
        return False
    if ctx.close[i] <= ctx.sma50[i] or ctx.close[i] <= ctx.sma200[i]:
        return False
    return _tag_and_hold(ctx, i, ctx.ema21[i])


def detect_sma50_reclaim(ctx: SymbolContext, i: int) -> bool:
    if i < 6 or not np.isfinite(ctx.sma50[i]):
        return False
    below = all(np.isfinite(ctx.sma50[j]) and ctx.close[j] < ctx.sma50[j] for j in range(i - 5, i))
    return below and ctx.close[i] > ctx.sma50[i]


def detect_sma200_reclaim(ctx: SymbolContext, i: int) -> bool:
    if i < 6 or not np.isfinite(ctx.sma200[i]):
        return False
    below = all(np.isfinite(ctx.sma200[j]) and ctx.close[j] < ctx.sma200[j] for j in range(i - 5, i))
    return below and ctx.close[i] > ctx.sma200[i]


def detect_golden_pullback_sma50(ctx: SymbolContext, i: int) -> bool:
    if i < 16 or not (np.isfinite(ctx.sma50[i]) and np.isfinite(ctx.sma50[i - 10])):
        return False
    if ctx.sma50[i] <= ctx.sma50[i - 10]:
        return False
    above = all(ctx.close[j] > ctx.sma50[j] for j in range(i - 15, i) if np.isfinite(ctx.sma50[j]))
    return above and _tag_and_hold(ctx, i, ctx.sma50[i])


def detect_high_252_breakout(ctx: SymbolContext, i: int) -> bool:
    if i < 60:
        return False
    lookback = min(i, 252)
    prior_high = float(np.max(ctx.high[i - lookback : i]))
    prev_prior_high = float(np.max(ctx.high[max(i - 1 - lookback, 0) : i - 1]))
    return ctx.close[i] > prior_high and ctx.close[i - 1] <= prev_prior_high


def detect_breakout_retest_252(ctx: SymbolContext, i: int) -> bool:
    if i < 70:
        return False
    for b in range(i - 10, i - 2):
        lookback = min(b, 252)
        if lookback < 60:
            continue
        prior_high = float(np.max(ctx.high[b - lookback : b]))
        if ctx.close[b] > prior_high and ctx.close[b - 1] <= prior_high:
            return _tag_and_hold(ctx, i, prior_high)
    return False


def detect_range5d_break_above_vwap(ctx: SymbolContext, i: int) -> bool:
    if i < 6 or not np.isfinite(ctx.vwap[i]):
        return False
    return ctx.close[i] > float(np.max(ctx.high[i - 5 : i])) and ctx.close[i] > ctx.vwap[i]


def detect_gap_up_hold(ctx: SymbolContext, i: int) -> bool:
    if not np.isfinite(ctx.vwap[i]):
        return False
    gap = ctx.open[i] >= ctx.close[i - 1] + 0.015 * abs(ctx.close[i - 1])
    return gap and ctx.close[i] >= ctx.open[i] and ctx.close[i] > ctx.vwap[i]


def detect_inside_day_break(ctx: SymbolContext, i: int) -> bool:
    if i < 2 or not np.isfinite(ctx.vwap[i]):
        return False
    inside = ctx.high[i - 1] <= ctx.high[i - 2] and ctx.low[i - 1] >= ctx.low[i - 2]
    return inside and ctx.close[i] > ctx.high[i - 1] and ctx.close[i] > ctx.vwap[i]


def detect_volume_thrust(ctx: SymbolContext, i: int) -> bool:
    if i < 21 or not np.isfinite(ctx.vwap[i]):
        return False
    avg_vol = float(np.mean(ctx.volume[i - 20 : i]))
    if avg_vol <= 0:
        return False
    up_move = ctx.close[i] >= ctx.close[i - 1] + 0.015 * abs(ctx.close[i - 1])
    return up_move and ctx.volume[i] >= 2.0 * avg_vol and ctx.close[i] > ctx.vwap[i]


def detect_quiet_pullback_resume(ctx: SymbolContext, i: int) -> bool:
    if i < 24 or not np.isfinite(ctx.sma50[i]) or ctx.close[i] <= ctx.sma50[i]:
        return False
    avg_vol = float(np.mean(ctx.volume[i - 20 : i]))
    if avg_vol <= 0:
        return False
    quiet = all(ctx.close[j] <= ctx.close[j - 1] and ctx.volume[j] < avg_vol for j in range(i - 3, i))
    return quiet and ctx.close[i] > ctx.close[i - 1] and ctx.volume[i] > ctx.volume[i - 1]


def detect_post_earnings_gap_hold3(ctx: SymbolContext, i: int) -> bool:
    g = ctx.last_earnings_session(i)
    if g is None or g < 1 or i != g + 2:
        return False
    gapped = ctx.open[g] >= ctx.close[g - 1] + 0.03 * abs(ctx.close[g - 1])
    held = all(ctx.close[j] > ctx.close[g - 1] for j in range(g, i + 1))
    return gapped and held and ctx.close[i] > ctx.open[g]


def detect_post_earnings_avwape_first_tag(ctx: SymbolContext, i: int) -> bool:
    since = ctx.sessions_since_last_earnings(i)
    anchor_session = int(ctx.anchor_idx[i])
    if since is None or not (3 <= since <= 40) or anchor_session < 0 or not np.isfinite(ctx.vwap[i]):
        return False
    if not _tag_and_hold(ctx, i, ctx.vwap[i]):
        return False
    for j in range(anchor_session + 1, i):
        vwap_j = ctx.band_asof(i, j, "VWAP")
        if np.isfinite(vwap_j) and ctx.low[j] <= vwap_j + TAG_ATR_TOLERANCE * ctx.atr[i]:
            return False  # not the first tag
    return True


def detect_baseline_every5(ctx: SymbolContext, i: int) -> bool:
    return (i + (hash(ctx.symbol) % 5)) % 5 == 0


PLAYBOOK: dict[str, dict] = {
    "vwap_reclaim": {"fn": detect_vwap_reclaim, "group": "avwap", "needs_bands": True},
    "vwap_bounce": {"fn": detect_vwap_bounce, "group": "avwap", "needs_bands": True},
    "first_dev_bounce": {"fn": detect_first_dev_bounce, "group": "avwap", "needs_bands": True},
    "first_dev_breakout": {"fn": detect_first_dev_breakout, "group": "avwap", "needs_bands": True},
    "second_dev_breakout": {"fn": detect_second_dev_breakout, "group": "avwap", "needs_bands": True},
    "second_dev_power_hold": {"fn": detect_second_dev_power_hold, "group": "avwap", "needs_bands": True},
    "power_trend_pullback": {"fn": detect_power_trend_pullback, "group": "avwap", "needs_bands": True},
    "band_test_rebound": {"fn": detect_band_test_rebound, "group": "avwap", "needs_bands": True},
    "ema8_pullback_uptrend": {"fn": detect_ema8_pullback_uptrend, "group": "ma"},
    "ema21_pullback_uptrend": {"fn": detect_ema21_pullback_uptrend, "group": "ma"},
    "sma50_reclaim": {"fn": detect_sma50_reclaim, "group": "ma"},
    "sma200_reclaim": {"fn": detect_sma200_reclaim, "group": "ma"},
    "golden_pullback_sma50": {"fn": detect_golden_pullback_sma50, "group": "ma"},
    "high_252_breakout": {"fn": detect_high_252_breakout, "group": "structure"},
    "breakout_retest_252": {"fn": detect_breakout_retest_252, "group": "structure"},
    "range5d_break_above_vwap": {"fn": detect_range5d_break_above_vwap, "group": "structure", "needs_bands": True},
    "gap_up_hold": {"fn": detect_gap_up_hold, "group": "structure", "needs_bands": True},
    "inside_day_break": {"fn": detect_inside_day_break, "group": "structure", "needs_bands": True},
    "volume_thrust": {"fn": detect_volume_thrust, "group": "volume", "needs_bands": True},
    "quiet_pullback_resume": {"fn": detect_quiet_pullback_resume, "group": "volume"},
    "post_earnings_gap_hold3": {"fn": detect_post_earnings_gap_hold3, "group": "earnings"},
    "post_earnings_avwape_first_tag": {"fn": detect_post_earnings_avwape_first_tag, "group": "earnings", "needs_bands": True},
    "baseline_every5": {"fn": detect_baseline_every5, "group": "baseline"},
}


# ---------------------------------------------------------------------------
# Outcome measurement (tracker-consistent, on the ORIGINAL frame)
# ---------------------------------------------------------------------------
def _round_trip_cost_per_share(entry: float, exit_price: float) -> float:
    return 2.0 * float(TRACKER_COST_COMMISSION_PER_SHARE) + float(TRACKER_SLIPPAGE_FRACTION_PER_SIDE) * (
        abs(entry) + abs(exit_price)
    )


def measure_episode(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    signal_idx: int,
    side: str,
    *,
    max_hold: int = TRACKER_MAX_HOLD_DAYS,
) -> dict | None:
    """Measure one episode. Returns None when the signal can't be traded."""
    i = signal_idx
    n = len(closes)
    if i + 1 >= n:
        return None  # no next bar yet -> nothing to enter on
    direction = 1.0 if side == "LONG" else -1.0
    entry_idx = i + 1
    entry = float(opens[entry_idx])
    atr_value = float(atr[i]) if np.isfinite(atr[i]) else 0.0
    if not np.isfinite(entry) or entry <= 0 or atr_value <= 0:
        return None
    stop = float(lows[i]) - STOP_ATR_BUFFER * atr_value if side == "LONG" else float(highs[i]) + STOP_ATR_BUFFER * atr_value
    risk = (entry - stop) * direction
    if risk <= 0:
        return None  # gapped through the stop before entry; untradeable as designed

    status = "OPEN"
    exit_idx = None
    exit_price = None
    horizon_r: dict[int, float | None] = {h: None for h in HORIZON_MARK_SESSIONS}

    for j in range(entry_idx, n):
        held = j - entry_idx
        stopped = (float(lows[j]) <= stop) if side == "LONG" else (float(highs[j]) >= stop)
        if stopped:
            fill = float(opens[j]) if (float(opens[j]) - stop) * direction < 0 else stop
            status, exit_idx, exit_price = "STOPPED", j, fill
        elif held >= max_hold:
            status, exit_idx, exit_price = "TIME_STOP", j, float(closes[j])

        for h in HORIZON_MARK_SESSIONS:
            if horizon_r[h] is None and (held == h or exit_idx is not None):
                mark = exit_price if exit_idx is not None else float(closes[j])
                pnl = (mark - entry) * direction - _round_trip_cost_per_share(entry, mark)
                horizon_r[h] = pnl / risk
        if exit_idx is not None:
            break

    if exit_idx is None:
        # Still open at data end: mark at the last close but report as OPEN.
        exit_price = float(closes[-1])
        exit_idx = n - 1

    pnl = (float(exit_price) - entry) * direction - _round_trip_cost_per_share(entry, float(exit_price))
    return {
        "entry_idx": entry_idx,
        "exit_idx": int(exit_idx),
        "status": status,
        "entry": entry,
        "stop": stop,
        "exit": float(exit_price),
        "risk": risk,
        "net_r": pnl / risk,
        "hold_sessions": int(exit_idx - entry_idx),
        **{f"r_{h}": horizon_r[h] for h in HORIZON_MARK_SESSIONS},
    }


# ---------------------------------------------------------------------------
# Backfill runner
# ---------------------------------------------------------------------------
def _durable_symbols() -> list[str]:
    symbols = []
    for path in sorted(Path(MASTER_AVWAP_DAILY_BARS_DIR).glob("*.parquet")):
        stem = path.stem
        if stem.endswith("_") and stem[:-1] in _WINDOWS_RESERVED_FILENAME_STEMS:
            stem = stem[:-1]
        symbols.append(stem)
    return symbols


def _load_daily_frame(symbol: str) -> pd.DataFrame | None:
    stem = f"{symbol}_" if symbol in _WINDOWS_RESERVED_FILENAME_STEMS else symbol
    path = Path(MASTER_AVWAP_DAILY_BARS_DIR) / f"{stem}.parquet"
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        logging.warning("%s: failed reading durable bars (%s)", symbol, exc)
        return None
    frame = frame.dropna(subset=["close"]).reset_index(drop=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame if len(frame) >= MIN_BARS_REQUIRED else None


def run_symbol(
    symbol: str,
    df: pd.DataFrame,
    earnings_dates: list[str],
    *,
    days: int,
) -> list[dict]:
    n = len(df)
    scan_start = max(n - days, 30)
    if scan_start >= n:
        return []
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    # Weekly context is computed once in original orientation; a +7 streak
    # means "7 completed weeks above the weekly 8EMA" for longs AND shorts.
    weekly_streaks = compute_weekly_streak_series(df)

    episodes: list[dict] = []
    for side, mirrored in (("LONG", False), ("SHORT", True)):
        ctx = build_symbol_context(symbol, df, earnings_dates, scan_start_idx=scan_start, mirrored=mirrored)
        atr = ctx.atr  # ATR is mirror-invariant
        last_exit: dict[str, int] = {}
        for i in range(scan_start, n):
            if closes[i] < MIN_PRICE:
                continue
            for family, spec in PLAYBOOK.items():
                if spec.get("needs_bands") and not np.isfinite(ctx.vwap[i]):
                    continue
                if i <= last_exit.get(family, -1):
                    continue
                try:
                    if not spec["fn"](ctx, i):
                        continue
                except (IndexError, ValueError):
                    continue
                outcome = measure_episode(opens, highs, lows, closes, atr, i, side)
                if outcome is None:
                    continue
                last_exit[family] = outcome["exit_idx"]
                episodes.append(
                    {
                        "symbol": symbol,
                        "family": family,
                        "group": spec.get("group", ""),
                        "side": side,
                        "signal_date": ctx.dates[i].isoformat(),
                        "entry_date": ctx.dates[outcome["entry_idx"]].isoformat(),
                        "weekly_streak": int(weekly_streaks[i]),
                        "weekly_ctx": classify_weekly_context(int(weekly_streaks[i])),
                        **{k: v for k, v in outcome.items() if k not in {"entry_idx", "exit_idx"}},
                    }
                )
    return episodes


def _aggregate_rows(frame: pd.DataFrame, keys: list[str]) -> list[dict]:
    rows: list[dict] = []
    if frame.empty:
        return rows
    closed = frame[frame["status"].isin(["STOPPED", "TIME_STOP"])]
    for key_values, group in frame.groupby(keys):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        mask = pd.Series(True, index=closed.index)
        for key, value in zip(keys, key_values):
            mask &= closed[key] == value
        closed_group = closed[mask]
        net_r = closed_group["net_r"]
        r5 = pd.to_numeric(group["r_5"], errors="coerce").dropna()
        r10 = pd.to_numeric(group["r_10"], errors="coerce").dropna()
        std = float(net_r.std(ddof=1)) if len(net_r) > 1 else float("nan")
        rows.append(
            {
                **dict(zip(keys, key_values)),
                "group": group["group"].iloc[0],
                "episodes": int(len(group)),
                "closed": int(len(closed_group)),
                "symbols": int(group["symbol"].nunique()),
                "win_rate": float((net_r > 0).mean()) if len(net_r) else None,
                "avg_r": float(net_r.mean()) if len(net_r) else None,
                "median_r": float(net_r.median()) if len(net_r) else None,
                "avg_r5": float(r5.mean()) if len(r5) else None,
                "avg_r10": float(r10.mean()) if len(r10) else None,
                "stop_rate": float((closed_group["status"] == "STOPPED").mean()) if len(closed_group) else None,
                "avg_hold": float(closed_group["hold_sessions"].mean()) if len(closed_group) else None,
                "t_stat": (float(net_r.mean()) / (std / math.sqrt(len(net_r)))) if len(net_r) > 1 and std and std > 0 else None,
            }
        )
    rows.sort(key=lambda row: (row["avg_r"] is None, -(row["avg_r"] if row["avg_r"] is not None else -999)))
    return rows


def aggregate_leaderboard(episodes: list[dict]) -> list[dict]:
    return _aggregate_rows(pd.DataFrame(episodes), ["family", "side"])


def aggregate_context_leaderboard(episodes: list[dict]) -> list[dict]:
    """Family performance split by the weekly-8EMA regime the signal fired in."""
    frame = pd.DataFrame(episodes)
    if frame.empty or "weekly_ctx" not in frame.columns:
        return []
    return _aggregate_rows(frame, ["family", "side", "weekly_ctx"])


def build_ai_digest(episodes: list[dict], *, days: int) -> dict:
    """Machine-readable study digest: baselines, per-combo edges, verdicts.

    Written for an AI (or the trader) to read in one pass: every
    family/side/weekly-context combo is compared to the SAME side+context
    baseline, so 'edge_r' is always apples-to-apples, and combos with enough
    evidence get an explicit actionable/avoid verdict plus a plain-language
    action line.
    """
    context_rows = aggregate_context_leaderboard(episodes)
    overall_rows = aggregate_leaderboard(episodes)
    for row in overall_rows:
        row["weekly_ctx"] = "all"
    all_rows = context_rows + overall_rows

    baselines: dict[tuple[str, str], dict] = {}
    for row in all_rows:
        if row["family"] == "baseline_every5" and row["avg_r"] is not None:
            baselines[(row["side"], row["weekly_ctx"])] = {
                "avg_r": row["avg_r"],
                "closed": row["closed"],
                "win_rate": row["win_rate"],
            }

    ctx_label = {
        "weekly_strong": f"weekly-strong ({WEEKLY_STREAK_STRONG_WEEKS}+ completed weeks above the weekly 8EMA)",
        "weekly_weak": f"weekly-weak ({WEEKLY_STREAK_STRONG_WEEKS}+ completed weeks below the weekly 8EMA)",
        "weekly_mixed": "weekly-mixed (no sustained weekly 8EMA streak)",
        "all": "all weekly contexts",
    }

    combos = []
    for row in all_rows:
        if row["family"] == "baseline_every5" or row["avg_r"] is None:
            continue
        base = baselines.get((row["side"], row["weekly_ctx"]))
        edge = (row["avg_r"] - base["avg_r"]) if base else None
        enough = row["closed"] >= DIGEST_MIN_CLOSED and bool(base) and base["closed"] >= DIGEST_MIN_CLOSED
        if edge is None or not enough:
            verdict = "insufficient_data"
        elif edge >= DIGEST_ACTIONABLE_MIN_EDGE:
            verdict = "actionable"
        elif edge <= -DIGEST_ACTIONABLE_MIN_EDGE:
            verdict = "avoid"
        else:
            verdict = "neutral"
        action = ""
        if verdict in {"actionable", "avoid"}:
            direction = "Take" if verdict == "actionable" else "Skip"
            action = (
                f"{direction} {row['side']} {row['family']} signals in {ctx_label[row['weekly_ctx']]}: "
                f"avg {row['avg_r']:+.2f}R vs baseline {base['avg_r']:+.2f}R "
                f"(edge {edge:+.2f}R, n={row['closed']}, win {row['win_rate'] * 100:.0f}%, "
                f"stop-out {row['stop_rate'] * 100:.0f}%, avg hold {row['avg_hold']:.1f} sessions)."
            )
        combos.append(
            {
                "family": row["family"],
                "side": row["side"],
                "weekly_ctx": row["weekly_ctx"],
                "closed": row["closed"],
                "win_rate": row["win_rate"],
                "avg_r": row["avg_r"],
                "avg_r5": row["avg_r5"],
                "avg_r10": row["avg_r10"],
                "stop_rate": row["stop_rate"],
                "edge_r_vs_baseline": edge,
                "verdict": verdict,
                "action": action,
            }
        )
    combos.sort(
        key=lambda c: (c["edge_r_vs_baseline"] is None, -(c["edge_r_vs_baseline"] if c["edge_r_vs_baseline"] is not None else -999))
    )

    frame = pd.DataFrame(episodes)
    return {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "window_sessions": days,
        "window_start": str(frame["signal_date"].min()) if not frame.empty else None,
        "window_end": str(frame["signal_date"].max()) if not frame.empty else None,
        "episode_count": int(len(frame)),
        "symbol_count": int(frame["symbol"].nunique()) if not frame.empty else 0,
        "methodology": {
            "entry": "next session open after the signal bar",
            "stop": f"{STOP_ATR_BUFFER} ATR beyond the signal bar extreme, intrabar stop-first, gaps fill at the open",
            "time_stop_sessions": int(TRACKER_MAX_HOLD_DAYS),
            "costs": "tracker round-trip cost model (commission + slippage fraction)",
            "dedup": "one open episode per symbol/family/side",
            "weekly_context": "signed streak of completed weeks closing above/below the weekly 8EMA, no lookahead",
        },
        "caveats": [
            "Survivorship: today's symbol universe replayed backward.",
            "Tight under-bar stops penalize bounce/pullback families; their unstopped 5-10d returns are strong, so treat their 'avoid' verdicts as a stop-model finding, not a signal-quality finding.",
            "Edges are window-specific; require agreement across windows (or with the live tracker leaderboard) before promoting to scoring.",
        ],
        "baselines": {f"{side}|{ctx}": stats for (side, ctx), stats in baselines.items()},
        "actionable": [c for c in combos if c["verdict"] == "actionable"],
        "avoid": [c for c in combos if c["verdict"] == "avoid"],
        "all_combos": combos,
    }


def render_report(leaderboard: list[dict], episodes: list[dict], *, days: int, digest: dict | None = None) -> str:
    frame = pd.DataFrame(episodes)
    lines = [
        f"SETUP PLAYBOOK STUDY  (generated {datetime.now().isoformat(timespec='minutes')})",
        f"Backfill window: last {days} sessions | entry next open, stop 0.1 ATR under signal bar,",
        f"intrabar stop-first, time stop {TRACKER_MAX_HOLD_DAYS} sessions, net of tracker costs.",
        f"Episodes: {len(frame)} across {frame['symbol'].nunique() if not frame.empty else 0} symbols.",
        "Compare every family to baseline_every5 (unconditional entries) before believing it.",
        "",
        f"{'family':<32}{'side':<7}{'n':>5}{'clsd':>6}{'win%':>7}{'avgR':>7}{'medR':>7}{'R@5':>7}{'R@10':>7}{'stop%':>7}{'hold':>6}{'t':>6}",
    ]

    def _fmt(value, pct=False, digits=2):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "-"
        return f"{value * 100:.0f}" if pct else f"{value:.{digits}f}"

    for row in leaderboard:
        if row["closed"] < LEADERBOARD_MIN_CLOSED and row["family"] != "baseline_every5":
            continue
        lines.append(
            f"{row['family']:<32}{row['side']:<7}{row['episodes']:>5}{row['closed']:>6}"
            f"{_fmt(row['win_rate'], pct=True):>7}{_fmt(row['avg_r']):>7}{_fmt(row['median_r']):>7}"
            f"{_fmt(row['avg_r5']):>7}{_fmt(row['avg_r10']):>7}{_fmt(row['stop_rate'], pct=True):>7}"
            f"{_fmt(row['avg_hold'], digits=1):>6}{_fmt(row['t_stat'], digits=1):>6}"
        )
    small = [r for r in leaderboard if r["closed"] < LEADERBOARD_MIN_CLOSED and r["family"] != "baseline_every5"]
    if small:
        lines += ["", f"(hidden: {len(small)} family/side rows with < {LEADERBOARD_MIN_CLOSED} closed episodes)"]

    digest = digest if digest is not None else build_ai_digest(episodes, days=days)
    lines += [
        "",
        "== WEEKLY 8EMA CONTEXT (edge vs same side+context baseline) ==",
        f"weekly_strong/weak = {WEEKLY_STREAK_STRONG_WEEKS}+ completed weeks above/below the weekly 8EMA.",
    ]
    for verdict, title in (("actionable", "-- actionable --"), ("avoid", "-- avoid --")):
        lines.append(title)
        picked = [c for c in digest[verdict] if c["weekly_ctx"] != "all"][:12]
        if not picked:
            lines.append("  (none with enough evidence)")
        for combo in picked:
            lines.append(f"  {combo['action']}")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_playbook_study(
    *,
    days: int = 60,
    symbols: list[str] | None = None,
    max_symbols: int | None = None,
    write_outputs: bool = True,
) -> dict:
    earnings_cache = load_earnings_date_cache()
    universe = symbols or _durable_symbols()
    if max_symbols:
        universe = universe[: int(max_symbols)]

    episodes: list[dict] = []
    processed = 0
    for symbol in universe:
        df = _load_daily_frame(symbol)
        if df is None:
            continue
        dates = (earnings_cache["symbols"].get(symbol) or {}).get("dates") or []
        episodes.extend(run_symbol(symbol, df, dates, days=days))
        processed += 1
        if processed % 200 == 0:
            logging.info("...%s symbols processed, %s episodes so far", processed, len(episodes))

    leaderboard = aggregate_leaderboard(episodes)
    context_rows = aggregate_context_leaderboard(episodes)
    digest = build_ai_digest(episodes, days=days)
    report = render_report(leaderboard, episodes, days=days, digest=digest)
    if write_outputs:
        _write_csv(PLAYBOOK_EPISODES_CSV, episodes)
        _write_csv(PLAYBOOK_LEADERBOARD_CSV, leaderboard)
        _write_csv(PLAYBOOK_CONTEXT_CSV, context_rows)
        PLAYBOOK_REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
        PLAYBOOK_REPORT_TXT.write_text(report, encoding="utf-8")
        PLAYBOOK_AI_DIGEST_JSON.write_text(json.dumps(digest, indent=2), encoding="utf-8")
        logging.info(
            "Playbook study wrote %s episodes / %s leaderboard rows / %s actionable combos",
            len(episodes),
            len(leaderboard),
            len(digest["actionable"]),
        )
    return {
        "episodes": episodes,
        "leaderboard": leaderboard,
        "context_rows": context_rows,
        "digest": digest,
        "report": report,
        "symbols_processed": processed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill-measure the setup playbook")
    parser.add_argument("--days", type=int, default=60, help="sessions to backfill (default 60)")
    parser.add_argument("--symbols", default="", help="comma-separated symbol subset")
    parser.add_argument("--max-symbols", type=int, default=0, help="cap symbol count (debug)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    subset = [s.strip().upper() for s in args.symbols.replace(",", " ").split() if s.strip()] or None
    result = run_playbook_study(days=args.days, symbols=subset, max_symbols=args.max_symbols or None)
    print(result["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
