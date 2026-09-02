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
    "D1_SMA_PERIODS",
    "MIN_PRICE",
    "RVOL_BARS",
    "RVOL_PRIOR_SESSIONS",
    "RVOL_TOP_FRACTION",
    "SESSION_M5_BARS",
    "SESSION_VOLUME_TOP_FRACTION",
    "floor_checks",
    "relative_volume",
    "session_volume",
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
STRENGTH_FETCH_PERIOD = "1mo"
#: THE PERIOD GREW FROM 5d TO 1mo, and it had to (V1, decision 0016 answer 9).
#: The trader's relative volume compares each of the last 12 bars with the SAME
#: bar offset over the **prior 15 sessions**, so the series needs sixteen
#: sessions of M5 bars; `5d` holds five and every RVOL would have been blank.
#: yfinance serves 5m bars for 60 days, so `1mo` is inside the provider's limit.
#: The cost is real and is stated rather than hidden: the 5d fetch measured
#: 27.6 s over 1,506 symbols on 2026-08-15, and this moves about six times the
#: rows. The board's refresh cadence is unchanged.
STRENGTH_FETCH_PERIOD_BEFORE_RVOL = "5d"

#: Completed M5 bars in one regular session: 6.5 h / 5 min.
SESSION_M5_BARS = 78
#: The trader's TC2000 relative volume, restated: `AVG(V / mean(V78, V156, ...
#: V1170), 12)` - each of the last 12 bars against the same bar offset over the
#: prior 15 sessions, averaged.
RVOL_BARS = 12
RVOL_PRIOR_SESSIONS = 15
#: Keep the busier half, on both the per-bar RVOL and today's session volume.
RVOL_TOP_FRACTION = 0.50
SESSION_VOLUME_TOP_FRACTION = 0.50

#: The trader's price floor. A dollar figure, not a percentile.
MIN_PRICE = 5.0

#: The daily simple moving averages the price must be above (long) or below
#: (short). **ASSUMPTION, stated so one line can correct it:** the trader wrote
#: "above the 200 and 100 SMA" without naming a timeframe, and D1 is the reading
#: that makes them structural rather than another intraday filter. The 15 EMA is
#: assumed M5 for the mirror reason - it is the intraday trigger line. Decision
#: 0016 answer 9 records both as open.
D1_SMA_PERIODS = (100, 200)


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


def relative_volume(
    bars: Sequence[Mapping[str, Any]],
    *,
    bar_count: int = RVOL_BARS,
    session_bars: int = SESSION_M5_BARS,
    prior_sessions: int = RVOL_PRIOR_SESSIONS,
) -> float | None:
    """The trader's TC2000 relative volume, or None when it cannot be measured.

    `AVG(V / mean(V78, V156, ... V1170), 12)`: each of the last 12 completed
    bars divided by the mean volume at the same bar offset over the prior 15
    sessions, and those twelve ratios averaged.

    **POSITIONAL, exactly as TC2000 is.** `V78` means "the volume 78 bars ago",
    not "the volume at this time of day yesterday". On a clean series of regular
    sessions the two are the same thing. On a half day - 3.25 hours, 39 bars -
    they are not, and every offset past that half day is shifted by 39 bars, so
    a 10:00 bar is compared with a 13:00 bar. That is a real divergence from what
    a reader assumes the number means, and it is TC2000's divergence too: parity
    with the trader's scan is the requirement, so it is documented here rather
    than silently corrected into a different number.

    **None, never zero, when there is not enough history.** A blank says "not
    measured"; a zero says "no relative volume", which would rank the symbol at
    the bottom of a filter it was never eligible for. The same rule covers a
    prior window whose mean volume is zero - a halted or untraded name divides by
    nothing, and its ratio is unmeasurable rather than infinite.
    """
    if bar_count <= 0 or session_bars <= 0 or prior_sessions <= 0:
        return None
    needed = bar_count + session_bars * prior_sessions
    if len(bars) < needed:
        return None
    volumes = [_finite(bar.get("volume")) for bar in bars]
    if any(volume is None for volume in volumes[-needed:]):
        return None
    ratios: list[float] = []
    for step in range(bar_count):
        index = len(volumes) - 1 - step
        prior = [
            volumes[index - session_bars * back]
            for back in range(1, prior_sessions + 1)
        ]
        average = sum(prior) / float(prior_sessions)  # type: ignore[arg-type]
        if average <= 0:
            return None
        ratios.append(float(volumes[index]) / average)  # type: ignore[arg-type]
    return sum(ratios) / float(len(ratios))


def session_volume(bars: Sequence[Mapping[str, Any]]) -> float | None:
    """Today's completed-bar volume, for the second half of the RVOL filter.

    The trader's rule has two parts and they are different questions: the RVOL
    asks whether each bar is busier than its own history, and this asks whether
    the SESSION is busy at all. A name can clear the first on twelve quiet bars
    that are merely less quiet than usual.
    """
    sessions = _session_groups(bars)
    if not sessions:
        return None
    total = 0.0
    for bar in sessions[-1]:
        volume = _finite(bar.get("volume"))
        if volume is None:
            return None
        total += volume
    return total


def floor_checks(
    row: Mapping[str, Any],
    side: str,
    *,
    min_price: float = MIN_PRICE,
) -> dict[str, Any]:
    """The trader's floors, each as a NAMED boolean plus what failed.

    Named rather than folded into one pass/fail because the board shows a failing
    row GREYED with its reason instead of hiding it (decision 0010: a display
    filter is not a suppression). A single boolean could not say which line the
    trader should look at.

    Missing data FAILS and says so - `cannot measure the 200 SMA` is a different
    sentence from `below the 200 SMA`, and only the second is a fact about the
    stock.
    """
    short = str(side or "").strip().lower().startswith("short")
    last = _finite(row.get("last"))
    checks: dict[str, bool] = {}
    failed: list[str] = []

    if last is None:
        return {
            "floors": {"price": False, "sma200": False, "sma100": False, "ema15": False},
            "failed_floors": ["no completed close"],
            "passes_floors": False,
        }

    checks["price"] = last > float(min_price)
    if not checks["price"]:
        failed.append(f"price ${last:.2f} is not over ${min_price:.0f}")

    for key, label, value in (
        ("sma200", "the D1 200 SMA", _finite(row.get("sma200_d1"))),
        ("sma100", "the D1 100 SMA", _finite(row.get("sma100_d1"))),
        ("ema15", "the M5 15 EMA", _finite(row.get("ema15"))),
    ):
        if value is None:
            checks[key] = False
            failed.append(f"cannot measure {label}")
            continue
        ok = last < value if short else last > value
        checks[key] = ok
        if not ok:
            failed.append(f"{'not below' if short else 'not above'} {label}")

    return {
        "floors": checks,
        "failed_floors": failed,
        "passes_floors": not failed,
    }


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
    daily_closes: Sequence[Any] | None = None,
) -> dict[str, Any] | None:
    """One board row's raw measurements, or None if it cannot be measured.

    Every field is derived from COMPLETED bars the caller has already trimmed.
    The filters are evaluated per side by `build_strength_board`; this only
    measures, so a row's numbers do not depend on which side asked for them.
    """
    score = strength_score(bars, body_bars=body_bars, atr_period=atr_period)
    if score is None:
        return None
    _daily = [value for value in (_finite(item) for item in (daily_closes or ())) if value is not None]
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
        # V1: the trader's own relative volume, and the two daily levels the
        # floors need. All three are BLANK rather than zero when they cannot be
        # measured - a blank says "not measured" and a zero says "measured, and
        # it is nothing", which is a different claim about the stock.
        "rvol": relative_volume(bars),
        "session_volume": session_volume(bars),
        "sma100_d1": sma(_daily, 100) if _daily else None,
        "sma200_d1": sma(_daily, 200) if _daily else None,
        "daily_bars": len(_daily),
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
    daily_closes_by_symbol: Mapping[str, Sequence[Any]] | None = None,
    rvol_fraction: float = RVOL_TOP_FRACTION,
    session_volume_fraction: float = SESSION_VOLUME_TOP_FRACTION,
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
            symbol,
            list(bars or []),
            body_bars=body_bars,
            atr_period=atr_period,
            ema_span=ema_span,
            daily_closes=(daily_closes_by_symbol or {}).get(str(symbol or "").strip().upper()),
        )
        if row is None:
            continue
        rows[row["symbol"]] = row
        scored.append((row["symbol"], row["strength"]))

    # V1: the two volume cuts, taken over the SAME measurable population the
    # strength cut is taken over, and before any filter - "top 50%" has to mean
    # top 50% of what was measured, exactly as the 25% strength cut does.
    #
    # A symbol whose RVOL is BLANK is not in the cut and not against it: it is
    # unmeasured. Treating a blank as a zero would put every name with under
    # sixteen sessions of history at the bottom of a ranking it was never in.
    rvol_scored = [
        (symbol, row["rvol"]) for symbol, row in rows.items() if row.get("rvol") is not None
    ]
    volume_scored = [
        (symbol, row["session_volume"])
        for symbol, row in rows.items()
        if row.get("session_volume") is not None
    ]
    top_rvol = {
        symbol for symbol, _value in percentile_cut(rvol_scored, fraction=rvol_fraction, side="long")
    }
    top_volume = {
        symbol
        for symbol, _value in percentile_cut(
            volume_scored, fraction=session_volume_fraction, side="long"
        )
    }

    board: dict[str, Any] = {
        "offered": len(bars_by_symbol or {}),
        "measured": len(scored),
        "rvol_measured": len(rvol_scored),
    }
    for side in ("long", "short"):
        kept: list[dict[str, Any]] = []
        filtered = 0
        for symbol, _score in percentile_cut(scored, fraction=fraction, side=side):
            row = dict(rows[symbol])
            passes, reason = _passes_filters(row, side)
            row.update(floor_checks(row, side))
            # The two volume cuts join the floors: they are the trader's own
            # filters, and a row that misses one is shown greyed with the reason
            # rather than dropped. **This is a display filter, never a
            # suppression** (decision 0010) - the row is in the board, carrying
            # why it is not a pick.
            row["in_top_rvol"] = symbol in top_rvol
            row["in_top_session_volume"] = symbol in top_volume
            if row.get("rvol") is None:
                row["failed_floors"] = [*row["failed_floors"], "relative volume not measurable"]
            elif not row["in_top_rvol"]:
                row["failed_floors"] = [*row["failed_floors"], "not in the busier half by RVOL"]
            if row.get("session_volume") is None:
                row["failed_floors"] = [*row["failed_floors"], "session volume not measurable"]
            elif not row["in_top_session_volume"]:
                row["failed_floors"] = [*row["failed_floors"], "not in the busier half today"]
            if not passes:
                row["failed_floors"] = [*row["failed_floors"], reason]
            row["passes_floors"] = not row["failed_floors"]
            row["side"] = side
            row["filter_reason"] = reason
            if not row["passes_floors"]:
                filtered += 1
            kept.append(row)
        board[side] = kept
        # The COUNT of greyed rows keeps its old name and its old meaning: how
        # many of the top-fraction rows are not picks. What changed is that they
        # are still in the list, so the trader can see what nearly qualified.
        board[f"{side}_filtered_out"] = filtered
        board[f"{side}_picks"] = sum(1 for row in kept if row["passes_floors"])
    return board
