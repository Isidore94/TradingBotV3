"""Two live LONG setups (p9): the leader pullback and the post-earnings drift. Pure, no I/O.

The trader, 2026-09-26: "let's build a scan to find these strong names on pullbacks. Give
them their own setup. Basically anything super strong at one point should be a candidate
for it. But especially leaders. The bot should really promote these." and, of the gap-up
names, "Ya that's the post earnings play".

* ``leader_pullback`` - a name that was strong at one point in the last
  `STRONG_LOOKBACK_SESSIONS` (a 52-week high, a `RUN_MIN_PCT` run inside
  `RUN_MAX_SESSIONS`, or 63-day RS vs SPY in the universe's top decile), now pulling
  back: `UNDER_AVWAP_PCT` under the VWAP anchored at the swing high OR back at the 21/50
  EMA, `OFF_HIGH_PCT` off that high, above its 200-day, on lighter volume than the run.
  A sector in the top RS third or a ``top_pattern_tracking`` row is a leader (a bonus).
* ``post_earnings_drift`` - an earnings gap up of `PED_GAP_MIN_ATR`+ ATR that closed in
  the upper half of the gap day, now `PED_SESSIONS` sessions later and holding above the
  gap-day low and the VWAP anchored at the gap day.

Each row carries an entry (a LIMIT `ENTRY_ATR_BELOW` ATR under the scan close - the S8 /
S15.9 fill rule, `research_warehouse.retest_entry`), a stop, the F16 / S13 exit ("take +1
ATR or 10 sessions"), its strength reasons in plain words and the market gate
(`setup_permutations.long_regime_working`, the trader's regime first): when the market is
not working the row still exists, reads "waiting for the market", and is not promoted.

Completed daily bars only, oldest first, as ``{date, open, high, low, close, volume}``.
Point in time: nothing after the last bar is read. Missing data is no setup, never a
guess. This never touches a detector, a score, the priority points or the buckets.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable, Mapping, Sequence

from indicators.atr import wilder_atr
from research_warehouse.retest_entry import RETEST_ATR_FRACTION, limit_fill
from setup_permutations import UNKNOWN, _sector_third, long_regime_working
from universe_builder import DEFAULT_MIN_AVG_VOLUME as MIN_AVG_VOLUME
from universe_builder import DEFAULT_MIN_MARKET_CAP_M as MIN_MARKET_CAP_M

LEADER_PULLBACK = "leader_pullback"
POST_EARNINGS_DRIFT = "post_earnings_drift"
#: Every long setup, in display order.
SETUPS = (LEADER_PULLBACK, POST_EARNINGS_DRIFT)
SETUP_LABELS = {LEADER_PULLBACK: "leader pullback", POST_EARNINGS_DRIFT: "post-earnings drift"}

# --- thresholds, in ONE place. First cut from TODO F10 / F16-F23 (2026-09-26); the long-lab
# builder calibrates them here and nowhere else.

#: "Strong at one point" is looked for in this many sessions back from the scan session.
STRONG_LOOKBACK_SESSIONS = 120
#: A 52-week high = a high at least as high as every high of the 252 sessions ending there.
HIGH_52W_SESSIONS = 252
#: A run = the close up this % from the lowest close of the prior `RUN_MAX_SESSIONS`.
RUN_MIN_PCT = 30.0
RUN_MAX_SESSIONS = 40
#: RS vs SPY over this many sessions, top this fraction of the scanned universe, with
#: at least `RS_MIN_NAMES` names measured (fewer = unknown).
RS_SESSIONS = 63
RS_TOP_FRACTION = 0.10
RS_MIN_NAMES = 20
#: The pullback's swing high is the highest high of this many sessions.
SWING_HIGH_LOOKBACK = 60
#: Pulling back = this % under the VWAP anchored at the swing high ...
UNDER_AVWAP_PCT = (3.0, 12.0)
#: ... or the close within `EMA_NEAR_ATR` ATR of one of these EMAs.
EMA_LENGTHS = (21, 50)
EMA_NEAR_ATR = 0.5
#: And this % off the swing high, above this SMA.
OFF_HIGH_PCT = (8.0, 25.0)
TREND_SMA = 200
#: The run's volume = the mean of this many sessions ending at the swing high.
RUN_VOLUME_SESSIONS = 20
#: ATR length when the scan row has no ATR of its own (the scan's `atr20`).
ATR_LENGTH = 20
#: Post-earnings drift: gap size floor (ATR), the gap day's close location floor
#: (0 = its low, 1 = its high), and the sessions-after window.
PED_GAP_MIN_ATR = 1.0
PED_CLOSE_LOCATION_MIN = 0.5
PED_SESSIONS = (4, 7)
#: The entry is a limit this many ATR under the scan close (S8 / S15.9).
ENTRY_ATR_BELOW = RETEST_ATR_FRACTION
#: The stop sits under the pullback / gap-day low when that is within this many ATR of
#: the entry; otherwise it is this many ATR under the entry.
STOP_MAX_ATR = 1.5
#: The exit (F16 / S13): take +1 ATR, or sell after 10 sessions.
TARGET_ATR = 1.0
TIME_EXIT_SESSIONS = 10
#: Grading: the limit rests through session 1; the return is read at this session's close.
GRADE_SESSIONS = 5
#: Grading: a long's raw win counts only in a window where SPY rose more than this %.
GRADE_SPY_UP_MIN_PCT = 1.0
#: Focus: a promoted row is a Focus candidate for this many calendar days after its scan
#: session, with this score base (the auto-populate ADR scores run ~1.25-3).
FOCUS_MAX_AGE_DAYS = 4
FOCUS_SCORE_BASE = 3.0

#: The trader's floor (2026-09-26: "We want 1B market cap and a avg20 daily volume of shares
#: traded to be 1m. That's my minimum."): market cap >= `MIN_MARKET_CAP_M` ($M) and the mean
#: share volume of the last `LIQUIDITY_VOLUME_SESSIONS` completed bars >= `MIN_AVG_VOLUME`,
#: both the universe builder's own constants. An unknown cap or volume is no row.
LIQUIDITY_VOLUME_SESSIONS = 20

STATUS_READY = "ready"
STATUS_WAITING = "waiting for the market"


# --- small helpers

def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _clean_bars(bars: Any) -> list[dict[str, Any]] | None:
    """The bars with numeric OHLC and a date; None when any bar lacks them (missing = no setup)."""
    out = []
    for bar in bars or ():
        if not isinstance(bar, Mapping):
            return None
        values = [_num(bar.get(key)) for key in ("open", "high", "low", "close")]
        day = _text(bar.get("date"))[:10]
        if not day or any(value is None or value <= 0 for value in values):
            return None
        out.append({"date": day, "open": values[0], "high": values[1], "low": values[2],
                    "close": values[3], "volume": _num(bar.get("volume"))})
    return out


def _sliding(values: Sequence[float], window: int, *, largest: bool) -> list[float | None]:
    """The max (or min) of the `window` values ending at each index; None before a full window."""
    out: list[float | None] = []
    queue: deque[int] = deque()
    for index, value in enumerate(values):
        while queue and ((values[queue[-1]] <= value) if largest else (values[queue[-1]] >= value)):
            queue.pop()
        queue.append(index)
        if queue[0] <= index - window:
            queue.popleft()
        out.append(values[queue[0]] if index >= window - 1 else None)
    return out


def ema(values: Sequence[float], length: int) -> float | None:
    """The last EMA value (seeded with the first `length` values' mean); None when too short."""
    if length <= 0 or len(values) < length:
        return None
    level = sum(values[:length]) / length
    alpha = 2.0 / (length + 1.0)
    for value in values[length:]:
        level += alpha * (value - level)
    return level


def anchored_vwap(bars: Sequence[Mapping[str, Any]], start: int) -> float | None:
    """VWAP of the typical price from bar `start` through the last bar; None without volume."""
    if not 0 <= start < len(bars):
        return None
    weighted = volume_total = 0.0
    for bar in bars[start:]:
        volume = _num(bar.get("volume"))
        if volume is None or volume < 0:
            return None
        weighted += (bar["high"] + bar["low"] + bar["close"]) / 3.0 * volume
        volume_total += volume
    return weighted / volume_total if volume_total > 0 else None


def _atr(bars: Sequence[Mapping[str, Any]], atr: Any) -> float | None:
    value = _num(atr)
    if value is not None and value > 0:
        return value
    value = wilder_atr(list(bars), ATR_LENGTH)
    return value if value is not None and value > 0 else None


def meets_liquidity_floor(bars: Any, market_cap_m: Any) -> bool:
    """True only for a known cap >= $1B and a known 20-session mean share volume >= 1M."""
    cap = _num(market_cap_m)
    if cap is None or cap < MIN_MARKET_CAP_M:
        return False
    recent = list(bars or ())[-LIQUIDITY_VOLUME_SESSIONS:]
    volumes = [_num(bar.get("volume")) if isinstance(bar, Mapping) else None for bar in recent]
    if len(volumes) < LIQUIDITY_VOLUME_SESSIONS or any(volume is None for volume in volumes):
        return False
    return sum(volumes) / len(volumes) >= MIN_AVG_VOLUME


# --- "strong at one point"

def made_52w_high(bars: Sequence[Mapping[str, Any]]) -> bool | None:
    """A 52-week high inside the lookback; None when no bar there has 252 sessions behind it."""
    highs = [bar["high"] for bar in bars]
    rolling = _sliding(highs, HIGH_52W_SESSIONS, largest=True)
    window = range(max(0, len(bars) - STRONG_LOOKBACK_SESSIONS), len(bars))
    known = [index for index in window if rolling[index] is not None]
    if any(highs[index] >= rolling[index] for index in known):
        return True
    return False if len(known) == len(window) and known else None


def best_run_pct(bars: Sequence[Mapping[str, Any]]) -> float | None:
    """The biggest close-over-lowest-prior-close run (%) inside the lookback; None when too short."""
    closes = [bar["close"] for bar in bars]
    lows = _sliding(closes, RUN_MAX_SESSIONS, largest=False)
    best = None
    for index in range(max(1, len(bars) - STRONG_LOOKBACK_SESSIONS), len(bars)):
        low = lows[index - 1]
        if low is None or low <= 0:
            continue
        run = (closes[index] / low - 1.0) * 100.0
        best = run if best is None else max(best, run)
    return best


def rs_vs_spy(bars: Sequence[Mapping[str, Any]], spy_closes: Mapping[str, float]) -> float | None:
    """(1 + the name's `RS_SESSIONS` return) / (1 + SPY's) - 1, both on the same two dates."""
    if len(bars) <= RS_SESSIONS:
        return None
    start, end = bars[-1 - RS_SESSIONS], bars[-1]
    spy_start, spy_end = _num(spy_closes.get(start["date"])), _num(spy_closes.get(end["date"]))
    if not spy_start or not spy_end or spy_start <= 0:
        return None
    return (end["close"] / start["close"]) / (spy_end / spy_start) - 1.0


def rs_percentiles(rs_by_symbol: Mapping[str, float | None]) -> dict[str, float]:
    """``{symbol: share of measured names it beats (0-1)}``; empty under `RS_MIN_NAMES` names."""
    known = {symbol: value for symbol, value in rs_by_symbol.items() if value is not None}
    if len(known) < RS_MIN_NAMES:
        return {}
    values = sorted(known.values())
    count = len(values)
    return {symbol: sum(1 for other in values if other < value) / (count - 1) if count > 1 else 0.0
            for symbol, value in known.items()}


# --- the two setups

def _leader_reasons(sector_top_third: bool | None, top_pattern: bool | None) -> list[str]:
    reasons = []
    if sector_top_third:
        reasons.append("leader: sector in the top third by RS")
    if top_pattern:
        reasons.append("leader: top-pattern tracking name")
    return reasons


def _plan(close: float, atr: float, structural_low: float | None, low_name: str) -> dict[str, Any]:
    """Entry, stop and exit for one row (all rounded to cents)."""
    entry = close - ENTRY_ATR_BELOW * atr
    if structural_low is not None and structural_low < entry and entry - structural_low <= STOP_MAX_ATR * atr:
        stop, basis = structural_low - 0.01, f"under the {low_name}"
    else:
        stop, basis = entry - STOP_MAX_ATR * atr, f"{STOP_MAX_ATR:g} ATR under the entry"
    target = entry + TARGET_ATR * atr
    return {
        "entry_limit": round(entry, 2),
        "stop": round(stop, 2),
        "stop_basis": basis,
        "target": round(target, 2),
        "time_exit_sessions": TIME_EXIT_SESSIONS,
        "exit": f"take +{TARGET_ATR:g} ATR at {target:.2f} or sell after {TIME_EXIT_SESSIONS} sessions",
    }


def leader_pullback(
    bars: Any,
    *,
    atr: Any = None,
    rs_percentile: float | None = None,
    sector_top_third: bool | None = None,
    top_pattern: bool | None = None,
) -> dict[str, Any] | None:
    """The leader-pullback row for one name's completed bars, or None (not a setup / unknown)."""
    bars = _clean_bars(bars)
    if not bars or len(bars) < TREND_SMA:
        return None
    atr_value = _atr(bars, atr)
    if atr_value is None:
        return None
    closes = [bar["close"] for bar in bars]
    close = closes[-1]
    sma = sum(closes[-TREND_SMA:]) / TREND_SMA
    if close <= sma:
        return None
    recent = range(len(bars) - SWING_HIGH_LOOKBACK, len(bars))
    high_index = max(recent, key=lambda index: (bars[index]["high"], index))
    if high_index >= len(bars) - 1:
        return None
    swing_high = bars[high_index]["high"]
    off_high = (swing_high - close) / swing_high * 100.0
    if not OFF_HIGH_PCT[0] <= off_high <= OFF_HIGH_PCT[1]:
        return None
    vwap = anchored_vwap(bars, high_index)
    under_vwap = (vwap - close) / vwap * 100.0 if vwap else None
    at_vwap = under_vwap is not None and UNDER_AVWAP_PCT[0] <= under_vwap <= UNDER_AVWAP_PCT[1]
    emas = {length: ema(closes, length) for length in EMA_LENGTHS}
    near_emas = [length for length, level in emas.items()
                 if level is not None and abs(close - level) <= EMA_NEAR_ATR * atr_value]
    if not at_vwap and not near_emas:
        return None
    run_volumes = [bar["volume"] for bar in bars[max(0, high_index - RUN_VOLUME_SESSIONS + 1):high_index + 1]]
    pullback_volumes = [bar["volume"] for bar in bars[high_index + 1:]]
    if any(volume is None for volume in (*run_volumes, *pullback_volumes)) or not run_volumes:
        return None
    run_volume = sum(run_volumes) / len(run_volumes)
    pullback_volume = sum(pullback_volumes) / len(pullback_volumes)
    if run_volume <= 0 or pullback_volume >= run_volume:
        return None
    strong = []
    if made_52w_high(bars):
        strong.append(f"made a 52-week high in the last {STRONG_LOOKBACK_SESSIONS} sessions")
    run = best_run_pct(bars)
    if run is not None and run >= RUN_MIN_PCT:
        strong.append(f"ran {run:.0f}% inside {RUN_MAX_SESSIONS} sessions")
    if rs_percentile is not None and rs_percentile >= 1.0 - RS_TOP_FRACTION:
        strong.append(f"{RS_SESSIONS}-day strength vs SPY in the top {RS_TOP_FRACTION:.0%} of the scan")
    if not strong:
        return None
    leaders = _leader_reasons(sector_top_third, top_pattern)
    where = []
    if at_vwap:
        where.append(f"{under_vwap:.1f}% under the VWAP from the high")
    where.extend(f"at the {length} EMA" for length in near_emas)
    reasons = [*strong, *leaders,
               f"pulling back {off_high:.1f}% off the high, " + " and ".join(where),
               "above the 200-day", "lighter volume on the pullback"]
    strength = len(strong) + len(leaders) + (rs_percentile or 0.0)
    return {
        "setup": LEADER_PULLBACK,
        "close": round(close, 2),
        "atr": round(atr_value, 4),
        "swing_high": round(swing_high, 2),
        "swing_high_date": bars[high_index]["date"],
        "pct_off_high": round(off_high, 2),
        "pct_under_avwap": None if under_vwap is None else round(under_vwap, 2),
        "leader": bool(leaders),
        "strength": round(strength, 3),
        "reasons": reasons,
        **_plan(close, atr_value, min(bar["low"] for bar in bars[high_index + 1:]), "pullback low"),
    }


def post_earnings_drift(
    bars: Any,
    *,
    gap_date: Any,
    gap_is_up: Any,
    gap_atr_multiple: Any,
    atr: Any = None,
    sector_top_third: bool | None = None,
    top_pattern: bool | None = None,
) -> dict[str, Any] | None:
    """The post-earnings-drift row for one name's completed bars, or None (not a setup / unknown)."""
    bars = _clean_bars(bars)
    gap_day = _text(gap_date)[:10]
    gap_size = _num(gap_atr_multiple)
    if not bars or not gap_day or gap_is_up is not True or gap_size is None or gap_size < PED_GAP_MIN_ATR:
        return None
    index = next((i for i, bar in enumerate(bars) if bar["date"] == gap_day), None)
    if index is None:
        return None
    after = len(bars) - 1 - index
    if not PED_SESSIONS[0] <= after <= PED_SESSIONS[1]:
        return None
    gap_bar = bars[index]
    day_range = gap_bar["high"] - gap_bar["low"]
    if day_range <= 0 or (gap_bar["close"] - gap_bar["low"]) / day_range < PED_CLOSE_LOCATION_MIN:
        return None
    if any(bar["close"] <= gap_bar["low"] for bar in bars[index + 1:]):
        return None
    vwap = anchored_vwap(bars, index)
    close = bars[-1]["close"]
    if vwap is None or close <= vwap:
        return None
    atr_value = _atr(bars, atr)
    if atr_value is None:
        return None
    leaders = _leader_reasons(sector_top_third, top_pattern)
    reasons = [f"earnings gap up {gap_size:.1f} ATR on {gap_day}, closed in the top half of the day",
               *leaders,
               f"{after} sessions later, holding above the gap-day low and the earnings-day VWAP"]
    strength = 1.0 + min(gap_size, 4.0) / 4.0 + len(leaders)
    return {
        "setup": POST_EARNINGS_DRIFT,
        "close": round(close, 2),
        "atr": round(atr_value, 4),
        "gap_date": gap_day,
        "gap_atr": round(gap_size, 3),
        "sessions_after_gap": after,
        "leader": bool(leaders),
        "strength": round(strength, 3),
        "reasons": reasons,
        **_plan(close, atr_value, gap_bar["low"], "gap-day low"),
    }


# --- one scan

def market_gate(feature_rows: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
    """``(working, rule)`` from the scan's regime columns; recomputed when only the inputs are there."""
    for row in feature_rows or ():
        verdict = _text(row.get("perm_regime_working"))
        if verdict in ("yes", "no"):
            return verdict, _text(row.get("perm_regime_working_rule")) or UNKNOWN
        if "perm_spy_vs_sma20_pct" in row or "perm_regime_trader" in row:
            return long_regime_working(row.get("perm_regime_trader"), row.get("perm_spy_vs_sma20_pct"),
                                       row.get("perm_spy_sma20_slope_pct"))
    return UNKNOWN, UNKNOWN


def apply_market_gate(rows: list[dict[str, Any]], working: str, rule: str) -> list[dict[str, Any]]:
    """Every row keeps its place; only a working market (``yes``) promotes it."""
    for row in rows:
        row["market_working"] = working
        row["market_rule"] = rule
        row["promoted"] = working == "yes"
        row["status"] = STATUS_READY if row["promoted"] else STATUS_WAITING
    return rows


def rank(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strongest first; ties by setup order then symbol."""
    return sorted(rows, key=lambda row: (-(row.get("strength") or 0.0), SETUPS.index(row["setup"]), row["symbol"]))


def build_rows(
    *,
    bars_by_symbol: Mapping[str, Any],
    spy_bars: Any,
    feature_rows: Iterable[Mapping[str, Any]],
    earnings_by_symbol: Mapping[str, Mapping[str, Any]] | None = None,
    atr_by_symbol: Mapping[str, Any] | None = None,
    sector_by_symbol: Mapping[str, Any] | None = None,
    market_cap_by_symbol: Mapping[str, Any] | None = None,
    as_of: Any = None,
) -> dict[str, Any]:
    """Every long-setup row of one scan, gated and ranked: ``{as_of, market_working, market_rule, rows}``.

    ``bars_by_symbol`` are each name's completed bars; a name whose last bar is not
    ``as_of`` is stale and skipped. ``feature_rows`` give the sector RS rank, the family
    and the market gate; they are read, never changed. A name under the trader's liquidity
    floor (`meets_liquidity_floor`; cap from ``market_cap_by_symbol``, else the row's
    ``perm_market_cap_m``) gives no row.
    """
    as_of_text = _text(as_of)[:10]
    feature_rows = list(feature_rows or ())
    working, rule = market_gate(feature_rows)
    if not as_of_text:
        return {"as_of": "", "market_working": working, "market_rule": rule, "rows": []}
    spy = _clean_bars(spy_bars) or []
    spy_closes = {bar["date"]: bar["close"] for bar in spy}
    rows_by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in feature_rows:
        rows_by_symbol.setdefault(_text(row.get("symbol")).upper(), []).append(row)
    current: dict[str, list[dict[str, Any]]] = {}
    for symbol, raw in (bars_by_symbol or {}).items():
        symbol = _text(symbol).upper()
        bars = _clean_bars(raw)
        if symbol and symbol != "SPY" and bars and bars[-1]["date"] == as_of_text:
            current[symbol] = bars
    percentiles = rs_percentiles({symbol: rs_vs_spy(bars, spy_closes) for symbol, bars in current.items()})
    out = []
    for symbol, bars in current.items():
        facts = rows_by_symbol.get(symbol, [])
        cap = _num((market_cap_by_symbol or {}).get(symbol))
        if cap is None:
            cap = next((_num(row.get("perm_market_cap_m")) for row in facts
                        if _num(row.get("perm_market_cap_m")) is not None), None)
        if not meets_liquidity_floor(bars, cap):
            continue
        thirds = {_sector_third(row, "perm_sector_rs_rank_20d") for row in facts} - {None}
        sector_top = True if "top" in thirds else (False if thirds else None)
        top_pattern = any(_text(row.get("setup_family")) == "top_pattern_tracking" for row in facts)
        sector = _text((sector_by_symbol or {}).get(symbol)) or next(
            (_text(row.get("sector")) for row in facts if _text(row.get("sector"))), "")
        atr = (atr_by_symbol or {}).get(symbol)
        found = [leader_pullback(bars, atr=atr, rs_percentile=percentiles.get(symbol),
                                 sector_top_third=sector_top, top_pattern=top_pattern)]
        earnings = (earnings_by_symbol or {}).get(symbol) or {}
        if earnings:
            found.append(post_earnings_drift(
                bars, gap_date=earnings.get("gap_date"), gap_is_up=earnings.get("gap_is_up"),
                gap_atr_multiple=earnings.get("gap_atr_multiple"), atr=atr,
                sector_top_third=sector_top, top_pattern=top_pattern))
        for row in found:
            if row is not None:
                out.append({"symbol": symbol, "as_of": as_of_text, "sector": sector, **row})
    return {"as_of": as_of_text, "market_working": working, "market_rule": rule,
            "rows": rank(apply_market_gate(out, working, rule))}


# --- grading (the scan settles; the Setup Tracker grades)

def settle(history: Iterable[Mapping[str, Any]], bars_by_symbol: Mapping[str, Any],
           spy_bars: Any) -> list[dict[str, Any]]:
    """Fill each unsettled history row's outcome once its `GRADE_SESSIONS` session is complete.

    The limit rests through session 1 (`retest_entry.limit_fill`; a gap fills at the open);
    no touch is ``no_fill`` - no trade, never a zero. A filled row gets ``return_pct`` (fill
    to the session-N close) and ``spy_return_pct`` (SPY's close on the scan session to the
    same close). Too few bars leaves the row unsettled (unknown).
    """
    spy_closes = {bar["date"]: bar["close"] for bar in (_clean_bars(spy_bars) or [])}
    cleaned: dict[str, list[dict[str, Any]] | None] = {}
    out = []
    for raw in history or ():
        row = dict(raw)
        out.append(row)
        if _text(row.get("outcome")):
            continue
        symbol = _text(row.get("symbol")).upper()
        if symbol not in cleaned:
            cleaned[symbol] = _clean_bars((bars_by_symbol or {}).get(symbol))
        bars = cleaned[symbol] or []
        index = next((i for i, bar in enumerate(bars) if bar["date"] == _text(row.get("as_of"))[:10]), None)
        entry = _num(row.get("entry_limit"))
        if index is None or entry is None or index + GRADE_SESSIONS >= len(bars):
            continue
        target = bars[index + GRADE_SESSIONS]
        row["target_session"] = target["date"]
        fill = limit_fill(bars[index + 1], entry, True)
        if fill is None:
            row["outcome"] = "no_fill"
            continue
        spy_start, spy_end = spy_closes.get(bars[index]["date"]), spy_closes.get(target["date"])
        row["outcome"] = "filled"
        row["fill"] = round(fill, 4)
        row["return_pct"] = round((target["close"] / fill - 1.0) * 100.0, 4)
        row["spy_return_pct"] = (round((spy_end / spy_start - 1.0) * 100.0, 4)
                                 if spy_start and spy_end else None)
    return out


def upsert_history(history: Iterable[Mapping[str, Any]], rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The history with this scan's rows; a later scan of the same session replaces that session's rows."""
    rows = [dict(row) for row in rows]
    days = {_text(row.get("as_of")) for row in rows}
    kept = [dict(row) for row in history or () if _text(row.get("as_of")) not in days]
    keep_keys = ("symbol", "as_of", "setup", "close", "atr", "entry_limit", "stop", "target",
                 "strength", "leader", "promoted", "market_working")
    return kept + [{key: row.get(key) for key in keep_keys} for row in rows]


# --- the words on the desk and the phone

def row_line(row: Mapping[str, Any], number: int | None = None) -> str:
    """``1. NVDA leader pullback | buy limit 120.10 | stop 115.20 (...) | take +1 ATR ... | ready``."""
    head = f"{number}. " if number is not None else ""
    return (f"{head}{row.get('symbol')} {SETUP_LABELS.get(row.get('setup'), row.get('setup'))}"
            f" | buy limit {_num(row.get('entry_limit')) or 0:.2f}"
            f" | stop {_num(row.get('stop')) or 0:.2f} ({row.get('stop_basis') or ''})"
            f" | {row.get('exit') or ''} | {row.get('status') or STATUS_WAITING}"
            f" | {'; '.join(row.get('reasons') or ())}")


def gate_line(payload: Mapping[str, Any] | None) -> str:
    payload = payload or {}
    working = _text(payload.get("market_working")) or UNKNOWN
    rule = _text(payload.get("market_rule")) or UNKNOWN
    words = {"yes": "the market is working for longs", "no": "the market is not working for longs"}
    return f"{words.get(working, 'the market state is unknown')} ({rule})"


def tracker_lines(payload: Mapping[str, Any] | None, *, limit: int = 12) -> list[str]:
    """The Setup Tracker's Long leaders section: a head line, then the rows by strength."""
    if not payload:
        return ["Long leaders: no scan has published long setups yet."]
    rows = list(payload.get("rows") or ())
    ready = sum(1 for row in rows if row.get("promoted"))
    head = (f"Long leaders (scan session {payload.get('as_of') or 'unknown'}): {len(rows)} setups, "
            f"{ready} ready - {gate_line(payload)}.")
    lines = [head, *(row_line(row, index) for index, row in enumerate(rows[:limit], start=1))]
    if len(rows) > limit:
        lines.append(f"(+{len(rows) - limit} more)")
    return lines


def phone_line(payload: Mapping[str, Any] | None, *, limit: int = 6) -> str:
    """One line for the phone report when a promoted row exists; "" otherwise."""
    rows = [row for row in (payload or {}).get("rows") or () if row.get("promoted")]
    if not rows:
        return ""
    parts = [f"{row.get('symbol')} ({SETUP_LABELS.get(row.get('setup'), row.get('setup'))}, "
             f"limit {_num(row.get('entry_limit')) or 0:.2f}, stop {_num(row.get('stop')) or 0:.2f})"
             for row in rows[:limit]]
    more = f" +{len(rows) - limit} more" if len(rows) > limit else ""
    return f"Long leaders {payload.get('as_of')}: " + ", ".join(parts) + more


def focus_candidates(payload: Mapping[str, Any] | None, *, today: Any) -> dict[str, list[dict[str, Any]]]:
    """Promoted rows as auto-populate LONG candidates (the Focus gate still judges each one).

    A payload older than `FOCUS_MAX_AGE_DAYS` calendar days, or dated after ``today``, gives none.
    """
    from datetime import date

    out: dict[str, list[dict[str, Any]]] = {"longs": [], "shorts": []}
    try:
        as_of = date.fromisoformat(_text((payload or {}).get("as_of"))[:10])
        day = today if isinstance(today, date) else date.fromisoformat(_text(today)[:10])
    except ValueError:
        return out
    if not 0 <= (day - as_of).days <= FOCUS_MAX_AGE_DAYS:
        return out
    for row in (payload or {}).get("rows") or ():
        if not row.get("promoted") or not _text(row.get("symbol")):
            continue
        out["longs"].append({
            "symbol": _text(row.get("symbol")).upper(),
            "score": FOCUS_SCORE_BASE + (_num(row.get("strength")) or 0.0),
            "reason": f"Long leaders: {SETUP_LABELS.get(row.get('setup'), row.get('setup'))}",
        })
    return out
