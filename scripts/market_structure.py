"""Machine structure labels from completed daily bars (S16 item 2). Pure.

Four facts, each shown beside the trader's own regime label and never over it:

* weekly higher-high / higher-low count over the last `WEEKLY_LOOKBACK` weeks;
* the daily channel from pivots (the scan's `_find_trendline_pivots`): `lh_ll`,
  `hh_hl`, `mixed` or `unknown`;
* ATR-percentile compression (ATR% of close against its own last year);
* distance to the 20-day SMA and the SMA's 5-session slope.

Input rows are dicts with `session_date` (date) and `open/high/low/close`.
`structure_facts` keeps only sessions before `as_of`; a week counts only once
it is over. Too few bars is `unknown`, never a guess.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable, Mapping, Sequence

UNKNOWN = "unknown"
WEEKLY_LOOKBACK = 13
CHANNEL_LOOKBACK = 60
ATR_PERIOD = 14
ATR_HISTORY = 252
ATR_MIN_HISTORY = 60
COMPRESSED_PERCENTILE = 20.0
SMA_PERIOD = 20
SMA_SLOPE_SESSIONS = 5


def _day(row: Mapping[str, Any]) -> date | None:
    value = row.get("session_date")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _num(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _clean(rows: Iterable[Mapping[str, Any]], as_of: date | None = None) -> list[dict]:
    """Rows with a date and OHLC (volume 0 when missing), before `as_of`, oldest first, one per date."""
    by_day: dict[date, dict] = {}
    for row in rows or ():
        day = _day(row)
        values = {key: _num(row.get(key)) for key in ("open", "high", "low", "close")}
        if day is None or any(value is None for value in values.values()):
            continue
        if as_of is not None and day >= as_of:
            continue
        by_day[day] = {"session_date": day, **values, "volume": _num(row.get("volume")) or 0.0}
    return [by_day[day] for day in sorted(by_day)]


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


# ------------------------------------------------------------- weekly --


def completed_weekly_bars(d1_rows: Iterable[Mapping[str, Any]], as_of: date) -> list[dict]:
    """Monday-Sunday weekly bars from sessions before `as_of`, finished weeks only."""
    weeks: dict[date, list[dict]] = {}
    for row in _clean(d1_rows, as_of):
        monday = row["session_date"] - timedelta(days=row["session_date"].weekday())
        if monday + timedelta(days=7) > as_of:
            continue  # the week is still running on `as_of`
        weeks.setdefault(monday, []).append(row)
    bars = []
    for monday in sorted(weeks):
        members = weeks[monday]
        bars.append(
            {
                "week_start": monday,
                "session_date": members[-1]["session_date"],
                "open": members[0]["open"],
                "high": max(row["high"] for row in members),
                "low": min(row["low"] for row in members),
                "close": members[-1]["close"],
                "volume": sum(row["volume"] for row in members),
            }
        )
    return bars


def weekly_structure(weekly: Sequence[Mapping[str, Any]], lookback: int = WEEKLY_LOOKBACK) -> dict:
    """Week-over-week higher/lower highs and lows over the last `lookback` weeks."""
    tail = list(weekly)[-(lookback + 1):]
    if len(tail) < 2:
        return {
            "status": UNKNOWN, "weeks": 0, "higher_highs": None, "higher_lows": None,
            "lower_highs": None, "lower_lows": None,
        }
    pairs = list(zip(tail, tail[1:], strict=False))
    return {
        "status": "ok",
        "weeks": len(pairs),
        "higher_highs": sum(1 for prev, cur in pairs if cur["high"] > prev["high"]),
        "higher_lows": sum(1 for prev, cur in pairs if cur["low"] > prev["low"]),
        "lower_highs": sum(1 for prev, cur in pairs if cur["high"] < prev["high"]),
        "lower_lows": sum(1 for prev, cur in pairs if cur["low"] < prev["low"]),
        "last_week": tail[-1]["week_start"].isoformat(),
    }


# ------------------------------------------------------------ channel --


def pivot_finder() -> Callable:
    """The scan's own pivot code, imported (never copied) from the swing scan."""
    from master_avwap_lib.legacy import _find_trendline_pivots

    return _find_trendline_pivots


def _pivot(entry: Mapping[str, Any] | None) -> dict | None:
    if not entry:
        return None
    return {"date": str(entry.get("date")), "value": _r(float(entry["value"]))}


def daily_channel(d1_rows: Iterable[Mapping[str, Any]], lookback: int = CHANNEL_LOOKBACK) -> dict:
    """Lower-high / lower-low channel (or the opposite) from the last two pivots of each kind."""
    rows = _clean(d1_rows)[-lookback:]
    empty = {"label": UNKNOWN, "last_pivot_high": None, "prev_pivot_high": None,
             "last_pivot_low": None, "prev_pivot_low": None, "sessions": len(rows)}
    if not rows:
        return empty
    import pandas as pd

    frame = pd.DataFrame(
        {
            "datetime": [pd.Timestamp(row["session_date"]) for row in rows],
            "high": [row["high"] for row in rows],
            "low": [row["low"] for row in rows],
        }
    )
    find = pivot_finder()
    highs = find(frame, "high", "high")
    lows = find(frame, "low", "low")
    result = {
        **empty,
        "last_pivot_high": _pivot(highs[-1] if highs else None),
        "prev_pivot_high": _pivot(highs[-2] if len(highs) > 1 else None),
        "last_pivot_low": _pivot(lows[-1] if lows else None),
        "prev_pivot_low": _pivot(lows[-2] if len(lows) > 1 else None),
    }
    if len(highs) < 2 or len(lows) < 2:
        return result
    high_now, high_before = highs[-1]["value"], highs[-2]["value"]
    low_now, low_before = lows[-1]["value"], lows[-2]["value"]
    if high_now < high_before and low_now < low_before:
        result["label"] = "lh_ll"
    elif high_now > high_before and low_now > low_before:
        result["label"] = "hh_hl"
    else:
        result["label"] = "mixed"
    return result


# -------------------------------------------------------- compression --


def _atr_pct_series(rows: Sequence[Mapping[str, Any]], period: int) -> list[float]:
    """ATR (simple mean of true range) as a percent of close, one per session from `period` on."""
    ranges = []
    for index, row in enumerate(rows):
        prev_close = rows[index - 1]["close"] if index else row["close"]
        ranges.append(max(row["high"] - row["low"], abs(row["high"] - prev_close), abs(row["low"] - prev_close)))
    series = []
    for end in range(period, len(rows) + 1):
        close = rows[end - 1]["close"]
        if close <= 0:
            continue
        series.append(sum(ranges[end - period:end]) / period / close * 100.0)
    return series


def atr_compression(d1_rows: Iterable[Mapping[str, Any]]) -> dict:
    """Where today's ATR% sits in its last year: percentile 0-100, compressed at or under 20."""
    rows = _clean(d1_rows)
    series = _atr_pct_series(rows, ATR_PERIOD)[-ATR_HISTORY:]
    if len(series) < ATR_MIN_HISTORY:
        return {"status": UNKNOWN, "atr_pct": None, "percentile": None, "compressed": None, "history": len(series)}
    current = series[-1]
    percentile = 100.0 * sum(1 for value in series if value <= current) / len(series)
    return {
        "status": "ok",
        "atr_pct": _r(current),
        "percentile": round(percentile, 1),
        "compressed": percentile <= COMPRESSED_PERCENTILE,
        "history": len(series),
    }


# -------------------------------------------------------------- sma20 --


def sma20_facts(d1_rows: Iterable[Mapping[str, Any]]) -> dict:
    """Last close vs the 20-day SMA (percent) and the SMA's change over 5 sessions (percent)."""
    closes = [row["close"] for row in _clean(d1_rows)]
    if len(closes) < SMA_PERIOD + SMA_SLOPE_SESSIONS:
        return {"status": UNKNOWN, "sma20": None, "distance_pct": None, "slope_pct": None, "side": UNKNOWN}
    sma = sum(closes[-SMA_PERIOD:]) / SMA_PERIOD
    earlier = closes[-SMA_PERIOD - SMA_SLOPE_SESSIONS:-SMA_SLOPE_SESSIONS]
    sma_before = sum(earlier) / SMA_PERIOD
    if sma <= 0 or sma_before <= 0:
        return {"status": UNKNOWN, "sma20": None, "distance_pct": None, "slope_pct": None, "side": UNKNOWN}
    distance = (closes[-1] / sma - 1) * 100
    return {
        "status": "ok",
        "sma20": _r(sma),
        "distance_pct": _r(distance),
        "slope_pct": _r((sma / sma_before - 1) * 100),
        "side": "above" if distance > 0 else "below" if distance < 0 else "at",
    }


def structure_facts(d1_rows: Iterable[Mapping[str, Any]], as_of: date) -> dict:
    """All four structure facts from sessions completed before `as_of`."""
    rows = _clean(d1_rows, as_of)
    return {
        "weekly": weekly_structure(completed_weekly_bars(rows, as_of)),
        "daily_channel": daily_channel(rows),
        "atr": atr_compression(rows),
        "sma20": sma20_facts(rows),
    }


__all__ = [
    "COMPRESSED_PERCENTILE",
    "WEEKLY_LOOKBACK",
    "atr_compression",
    "completed_weekly_bars",
    "daily_channel",
    "pivot_finder",
    "sma20_facts",
    "structure_facts",
    "weekly_structure",
]
