"""The M5 strength score and its filters (plan.md Phase 0.5, packet R2 Part B).

The trader's TC2000 scan, restated. Per symbol, on M5:

    strength = ( SUM over the last 12 completed bars of ((C/O) - 1) * 100 ) / 12
               * ( (C + C50) / 2 ) / ATR50

where `C50` is **TC2000 displacement syntax: the close FIFTY BARS AGO**, and
`ATR50` is the 50-bar M5 average true range. Rank descending and keep the top
25%; mirror for shorts.

`C50` is a single historical price, not an average. The first build read it as
a 50-bar SMA because the spec restated it that way; the trader corrected it on
2026-08-15 and TC2000 parity is the intent (see that plan's §B.1). The
difference is real: an SMA smooths away the very displacement the price factor
is asking about.

The shape is "average per-bar body move, scaled by price level and divided by
volatility": the first factor says how hard the last hour pushed, the price
factor keeps a $400 name comparable to a $20 one by anchoring on where it was
an hour and a half back, and ATR50 normalises so a quiet name that moves 1%
ranks above a jumpy one that moves the same.

Pure arithmetic - no Qt, no network, no project imports. It deliberately does
NOT touch `real_relative_strength`: that is a load-bearing, fenced engine
computing an ATR-normalised SPY/sector/industry excess, which is a different
question with a different answer. The existing RS/RW board keeps working
unchanged beside this one.

Completed bars only, everywhere (plan.md sec 5). A forming bar is a preview,
and a board that ranked on one would reshuffle every few seconds against moves
that had not happened yet.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "STRENGTH_ATR_PERIOD",
    "STRENGTH_BODY_BARS",
    "STRENGTH_EMA_SPAN",
    "STRENGTH_FETCH_PERIOD",
    "STRENGTH_TOP_FRACTION",
    "atr",
    "build_strength_board",
    "displaced_close",
    "ema",
    "percentile_cut",
    "score_symbol",
    "sma",
    "strength_score",
    "true_ranges",
]

#: Bars in the body-move sum (the trader's "last 12 completed 5-minute bars" -
#: one hour of tape).
STRENGTH_BODY_BARS = 12
#: Lookback shared by ATR50 and the C50 displacement. Both need 51 bars - ATR
#: because its first bar contributes no true range, C50 because the close fifty
#: bars back is the fifty-first value - so one history check covers both.
STRENGTH_ATR_PERIOD = 50
#: The M5 EMA the trader filters on.
STRENGTH_EMA_SPAN = 15
#: Keep the strongest / weakest quarter of what was measurable.
STRENGTH_TOP_FRACTION = 0.25
#: Five days of 5m bars, NOT one. The formula needs 50 completed bars for
#: ATR50 and C50, and a 1d window holds about 78 bars for a FULL session - so at
#: 07:00 PT it holds six, and every symbol would be unmeasurable for the first
#: four hours of the session the trader is actually trading. Measured
#: 2026-08-15: 5d gives every symbol >= 50 bars (median 390) and costs 27.6 s
#: over the whole 1,506-symbol universe. Spanning sessions is also correct
#: rather than merely convenient - TC2000's M5 displacement and ATR span them too.
STRENGTH_FETCH_PERIOD = "5d"


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sma(values: Sequence[float], period: int) -> float | None:
    """Simple average of the last ``period`` values, or None if there are not
    that many. Not "as many as we have" - a 12-value average presented as a
    50-value one would silently rank a name that just listed against one with
    real history. Used by `atr`; the score's price factor uses
    `displaced_close`, not this."""
    period = int(period)
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    if any(value is None or not math.isfinite(value) for value in window):
        return None
    return sum(window) / float(period)


def displaced_close(values: Sequence[float], bars_ago: int) -> float | None:
    """The close ``bars_ago`` bars back - TC2000's `C50` displacement syntax.

    `C` is the current close, `C1` the close one bar ago, `C50` the close fifty
    bars ago. So this is a single historical price, NOT an average of the last
    fifty: the formula's price factor compares where the name is now with where
    it was an hour and a half back, and averaging would smear exactly the
    displacement it is asking about.

    Needs ``bars_ago + 1`` values, which is the same 51 bars ATR50 needs (its
    first bar contributes no true range), so the two refusals coincide rather
    than one silently masking the other.
    """
    bars_ago = int(bars_ago)
    if bars_ago < 0 or len(values) < bars_ago + 1:
        return None
    value = values[-(bars_ago + 1)]
    return value if value is not None and math.isfinite(value) else None


def true_ranges(bars: Sequence[Mapping[str, Any]]) -> list[float] | None:
    """Wilder true range per bar, from the second bar on.

    TR = max(high - low, |high - prev_close|, |low - prev_close|). The first bar
    has no previous close, so it contributes none - which is why ATR50 needs 51
    bars, not 50.
    """
    if len(bars) < 2:
        return None
    ranges: list[float] = []
    previous_close = _finite(bars[0].get("close"))
    if previous_close is None:
        return None
    for bar in bars[1:]:
        high = _finite(bar.get("high"))
        low = _finite(bar.get("low"))
        close = _finite(bar.get("close"))
        if high is None or low is None or close is None or low > high:
            return None
        ranges.append(
            max(high - low, abs(high - previous_close), abs(low - previous_close))
        )
        previous_close = close
    return ranges


def atr(bars: Sequence[Mapping[str, Any]], period: int = STRENGTH_ATR_PERIOD) -> float | None:
    """Simple ATR over ``period`` completed bars, or None without enough of them."""
    ranges = true_ranges(bars)
    if ranges is None:
        return None
    return sma(ranges, period)


def strength_score(
    bars: Sequence[Mapping[str, Any]],
    *,
    body_bars: int = STRENGTH_BODY_BARS,
    atr_period: int = STRENGTH_ATR_PERIOD,
) -> float | None:
    """The trader's M5 strength score, or None when it cannot be computed.

    ``bars`` must already be completed bars in ascending time order. Returns
    None rather than a partial answer on short history, a non-finite input, or a
    zero ATR - a board row that cannot be measured is not a weak row, and
    presenting it as one would rank a data problem against real setups.
    """
    body_bars = int(body_bars)
    if body_bars <= 0 or len(bars) < body_bars:
        return None

    body_sum = 0.0
    for bar in bars[-body_bars:]:
        open_price = _finite(bar.get("open"))
        close = _finite(bar.get("close"))
        if open_price is None or close is None or open_price <= 0:
            return None
        body_sum += ((close / open_price) - 1.0) * 100.0
    average_body = body_sum / float(body_bars)

    closes = [_finite(bar.get("close")) for bar in bars]
    if any(close is None for close in closes):
        return None
    displaced = displaced_close(closes, atr_period)  # type: ignore[arg-type]
    if displaced is None:
        return None
    last_close = closes[-1]
    price_factor = (last_close + displaced) / 2.0  # type: ignore[operator]

    volatility = atr(bars, atr_period)
    if volatility is None or volatility <= 0:
        return None

    score = average_body * price_factor / volatility
    return score if math.isfinite(score) else None


def percentile_cut(
    scored: Iterable[tuple[str, float]],
    *,
    fraction: float = 0.25,
    side: str = "long",
) -> list[tuple[str, float]]:
    """The strongest (or weakest) ``fraction`` of a scored population.

    Longs take the top slice by signed score, shorts the bottom, and the shorts
    come back weakest-first so both sides read "best row at the top".

    The cut is a proportion of what was actually measured, so a session where
    half the universe is unmeasurable narrows the board instead of promoting
    noise into it. At least one row survives a non-empty population - a 25% cut
    of three names is not zero names.
    """
    rows = [(symbol, score) for symbol, score in scored if _finite(score) is not None]
    if not rows:
        return []
    fraction = min(max(float(fraction), 0.0), 1.0)
    keep = max(1, int(round(len(rows) * fraction)))
    ordered = sorted(rows, key=lambda row: row[1], reverse=True)
    if str(side or "").strip().lower().startswith("short"):
        return list(reversed(ordered[-keep:]))
    return ordered[:keep]


def ema(values: Sequence[float], span: int) -> float | None:
    """Final EMA value, seeded on the first sample.

    Same seeding as `chart_snapshot.ema_series` (which the desk's EMA overlays
    use), so a board row and the chart the trader opens from it agree about
    where the 15EMA is.
    """
    span = max(1, int(span))
    if not values:
        return None
    alpha = 2.0 / (span + 1.0)
    result = _finite(values[0])
    if result is None:
        return None
    for value in values[1:]:
        current = _finite(value)
        if current is None:
            return None
        result = alpha * current + (1.0 - alpha) * result
    return result


def _session_groups(bars: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Split ascending bars into per-date groups."""
    groups: list[list[Mapping[str, Any]]] = []
    current_date = None
    for bar in bars:
        stamp = bar.get("dt")
        bar_date = stamp.date() if hasattr(stamp, "date") else None
        if bar_date != current_date or not groups:
            groups.append([])
            current_date = bar_date
        groups[-1].append(bar)
    return groups


def score_symbol(
    symbol: str,
    bars: Sequence[Mapping[str, Any]],
    *,
    body_bars: int = STRENGTH_BODY_BARS,
    atr_period: int = STRENGTH_ATR_PERIOD,
    ema_span: int = STRENGTH_EMA_SPAN,
) -> dict[str, Any] | None:
    """One board row's raw measurements, or None if it cannot be measured.

    Every field is derived from COMPLETED bars the caller has already trimmed.
    The filters are evaluated per side by `build_strength_board`; this only
    measures, so a row's numbers do not depend on which side asked for them.
    """
    score = strength_score(bars, body_bars=body_bars, atr_period=atr_period)
    if score is None:
        return None
    sessions = _session_groups(bars)
    if len(sessions) < 2:
        return None  # no prior session, so no yesterday's high/low to compare
    today = sessions[-1]
    previous = sessions[-2]
    closes = [_finite(bar.get("close")) for bar in bars]
    if any(close is None for close in closes):
        return None
    last_close = closes[-1]

    prev_highs = [_finite(bar.get("high")) for bar in previous]
    prev_lows = [_finite(bar.get("low")) for bar in previous]
    if any(value is None for value in prev_highs) or any(value is None for value in prev_lows):
        return None

    # Session VWAP over today's completed bars only. Same accumulation as
    # `chart_snapshot.session_vwap_series` - the running-deviation variant every
    # band consumer is calibrated to - restricted to the current date.
    cum_volume = cum_price_volume = 0.0
    for bar in today:
        volume = _finite(bar.get("volume")) or 0.0
        if volume <= 0:
            continue
        typical = (
            (_finite(bar.get("open")) or 0.0)
            + (_finite(bar.get("high")) or 0.0)
            + (_finite(bar.get("low")) or 0.0)
            + (_finite(bar.get("close")) or 0.0)
        ) / 4.0
        cum_volume += volume
        cum_price_volume += typical * volume
    session_vwap = cum_price_volume / cum_volume if cum_volume > 0 else None

    session_open = _finite(today[0].get("open")) if today else None
    day_pct = (
        (last_close - session_open) / session_open * 100.0
        if session_open not in (None, 0)
        else None
    )

    return {
        "symbol": str(symbol or "").strip().upper(),
        "strength": score,
        "last": last_close,
        "session_vwap": session_vwap,
        "vwap_distance_pct": (
            (last_close - session_vwap) / session_vwap * 100.0
            if session_vwap not in (None, 0)
            else None
        ),
        "prev_high": max(prev_highs),  # type: ignore[type-var]
        "prev_low": min(prev_lows),  # type: ignore[type-var]
        "ema15": ema(closes, ema_span),  # type: ignore[arg-type]
        "day_pct": day_pct,
        "bars": len(bars),
    }


def _passes_filters(row: Mapping[str, Any], side: str) -> tuple[bool, str]:
    """The trader's board filters for one side.

    Price / 20-day volume / market cap / optionable are NOT re-checked here:
    membership in `universe_all.txt` already means they passed at universe-build
    time, and re-fetching them per refresh would cost more than the whole scan.

    Missing data fails, as everywhere else - an unmeasurable row is not a
    qualifying one.
    """
    short = str(side or "").strip().lower().startswith("short")
    last = _finite(row.get("last"))
    if last is None:
        return False, "no completed close"
    for label, level in (
        ("session VWAP", _finite(row.get("session_vwap"))),
        ("the 15EMA", _finite(row.get("ema15"))),
        (
            "yesterday's low" if short else "yesterday's high",
            _finite(row.get("prev_low") if short else row.get("prev_high")),
        ),
    ):
        if level is None:
            return False, f"cannot measure {label}"
        if short and last >= level:
            return False, f"not below {label}"
        if not short and last <= level:
            return False, f"not above {label}"
    return True, "above all three" if not short else "below all three"


def build_strength_board(
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    fraction: float = STRENGTH_TOP_FRACTION,
    body_bars: int = STRENGTH_BODY_BARS,
    atr_period: int = STRENGTH_ATR_PERIOD,
    ema_span: int = STRENGTH_EMA_SPAN,
) -> dict[str, Any]:
    """Score everything, cut to the top/bottom fraction, then filter.

    Order matters and follows the spec: the percentile cut is taken over the
    whole measurable population FIRST, so "top 25%" means what it says, and only
    the survivors are filtered. Cutting after filtering would make the fraction
    a proportion of an already-filtered set and quietly change the scan.

    Returns both sides plus the accounting a trader needs to trust a short
    board: how many symbols were offered, how many could be measured, and how
    many the filters removed. Honest zero rows beat a filled panel.
    """
    scored: list[tuple[str, float]] = []
    rows: dict[str, dict[str, Any]] = {}
    for symbol, bars in (bars_by_symbol or {}).items():
        row = score_symbol(
            symbol, list(bars or []), body_bars=body_bars, atr_period=atr_period, ema_span=ema_span
        )
        if row is None:
            continue
        rows[row["symbol"]] = row
        scored.append((row["symbol"], row["strength"]))

    board: dict[str, Any] = {
        "offered": len(bars_by_symbol or {}),
        "measured": len(scored),
    }
    for side in ("long", "short"):
        kept: list[dict[str, Any]] = []
        filtered = 0
        for symbol, _score in percentile_cut(scored, fraction=fraction, side=side):
            row = dict(rows[symbol])
            passes, reason = _passes_filters(row, side)
            if not passes:
                filtered += 1
                continue
            row["side"] = side
            row["filter_reason"] = reason
            kept.append(row)
        board[side] = kept
        board[f"{side}_filtered_out"] = filtered
    return board
