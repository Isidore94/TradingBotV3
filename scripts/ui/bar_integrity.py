"""Whether a candle can be drawn honestly, and what to do when it cannot.

Every OHLC bar carries an invariant the chart never checked::

    low <= min(open, close) <= max(open, close) <= high

The chart's y-range comes from the lows and the highs, but the *body* is drawn
from the open and the close. Those are the same numbers only while the
invariant holds. When it does not - a zero open, a close from another scale, a
NaN - the body is drawn from a coordinate the range never saw, so it paints a
solid column straight through the viewport while the axis stays perfectly
normal. That is the "massive green candle" that hides a whole session
(trader, 2026-08-21).

The rule here is the repo's rule: missing or broken data is uncertainty, never
confirmation. A malformed bar is therefore neither trusted nor silently
dropped - it is named, drawn in a way that cannot be mistaken for a real
candle, and kept out of the price range so the bars around it stay readable.

Pure functions, no Qt, so both the renderer and the data service can use the
same judgement and a test can exercise it without a window.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

#: A price field was absent, or not a number at all.
DEFECT_NOT_NUMERIC = "not_numeric"
#: A price is NaN or an infinity. NaN fails every compare, so it would slip
#: through an ordering check and reach the painter as a coordinate.
DEFECT_NOT_FINITE = "not_finite"
#: high < low. The bar's own range is impossible; nothing about it is usable.
DEFECT_RANGE_INVERTED = "range_inverted"
#: The open sits outside [low, high] - the body escapes the range.
DEFECT_OPEN_OUTSIDE = "open_outside_range"
#: The close sits outside [low, high] - same, and it also decides the color,
#: so a bogus close paints a *green* wall on a down day.
DEFECT_CLOSE_OUTSIDE = "close_outside_range"

#: NOT a defect: a well-formed bar whose range dwarfs its neighbours. It is
#: what an aggregate row (a daily or hourly bar that reached an M5 series)
#: looks like from the outside, and it is indistinguishable from a genuine
#: violent bar without knowing the source - so it is OBSERVED and never
#: redrawn. ``scan_bars`` does not return these; only the diagnostic does.
DEFECT_RANGE_OUTLIER = "range_outlier"

_PRICE_FIELDS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class BarDefect:
    """One bar the chart may not draw at face value."""

    index: int
    defect: str
    bar: dict

    @property
    def drawable(self) -> bool:
        """Whether low/high survive, so a clamped candle still says something.

        A bad open or close leaves the bar's *range* intact and worth seeing;
        a non-finite price or an inverted range leaves nothing to stand on.
        """
        return self.defect in (DEFECT_OPEN_OUTSIDE, DEFECT_CLOSE_OUTSIDE)


def _price(bar: Mapping[str, Any], field: str) -> float | None:
    try:
        value = float(bar[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value


def bar_defect(bar: Mapping[str, Any]) -> str | None:
    """The defect name for one bar, or ``None`` when it is well formed."""
    values: dict[str, float] = {}
    for field in _PRICE_FIELDS:
        value = _price(bar, field)
        if value is None:
            return DEFECT_NOT_NUMERIC
        if not isfinite(value):
            return DEFECT_NOT_FINITE
        values[field] = value
    if values["high"] < values["low"]:
        return DEFECT_RANGE_INVERTED
    # Order matters only for reporting: check the open first so a bar that is
    # wrong at both ends reports the earlier field, which is the one a
    # provider bug usually explains.
    if not values["low"] <= values["open"] <= values["high"]:
        return DEFECT_OPEN_OUTSIDE
    if not values["low"] <= values["close"] <= values["high"]:
        return DEFECT_CLOSE_OUTSIDE
    return None


def scan_bars(bars: Sequence[Mapping[str, Any]]) -> list[BarDefect]:
    """Every malformed bar in a series, in the order they appear."""
    found: list[BarDefect] = []
    for index, bar in enumerate(bars or ()):
        defect = bar_defect(bar)
        if defect is not None:
            found.append(BarDefect(index=index, defect=defect, bar=dict(bar)))
    return found


def range_outliers(
    bars: Sequence[Mapping[str, Any]],
    *,
    factor: float = 6.0,
    share: float = 0.5,
    minimum_bars: int = 12,
) -> list[BarDefect]:
    """Well-formed bars whose range is a wild outlier for their own series.

    Two conditions, both required, because either alone fires on ordinary
    days: the bar's range is ``factor`` times the median range, AND it covers
    at least ``share`` of the whole series' range. A gap day clears the first
    easily; only a row that is effectively a summary of everything clears the
    second.

    Advisory. Nothing here changes what is drawn - see DEFECT_RANGE_OUTLIER.
    """
    measured: list[tuple[int, float, dict]] = []
    for index, bar in enumerate(bars or ()):
        if bar_defect(bar) is not None:
            continue
        low, high = _price(bar, "low"), _price(bar, "high")
        if low is None or high is None:
            continue
        measured.append((index, high - low, dict(bar)))
    if len(measured) < max(3, int(minimum_bars)):
        return []
    ranges = sorted(span for _index, span, _bar in measured)
    middle = len(ranges) // 2
    median_range = (
        ranges[middle]
        if len(ranges) % 2
        else (ranges[middle - 1] + ranges[middle]) / 2.0
    )
    span = price_range(bars)
    if not median_range or span is None:
        return []
    series_range = span[1] - span[0]
    if series_range <= 0:
        return []
    return [
        BarDefect(index=index, defect=DEFECT_RANGE_OUTLIER, bar=bar)
        for index, bar_span, bar in measured
        if bar_span >= median_range * factor and bar_span >= series_range * share
    ]


def price_range(bars: Sequence[Mapping[str, Any]]) -> tuple[float, float] | None:
    """(low, high) across the bars a chart may take its scale from.

    Well-formed bars decide the range. If EVERY bar is malformed the range
    falls back to the ones whose low/high still hold, because a chart with no
    range at all is less useful than a chart drawn from what survived; if not
    even that is available the caller gets ``None`` and should leave the view
    alone rather than invent one.
    """
    trusted: list[tuple[float, float]] = []
    salvaged: list[tuple[float, float]] = []
    for bar in bars or ():
        low = _price(bar, "low")
        high = _price(bar, "high")
        if low is None or high is None or not isfinite(low) or not isfinite(high):
            continue
        if high < low:
            continue
        (trusted if bar_defect(bar) is None else salvaged).append((low, high))
    pool = trusted or salvaged
    if not pool:
        return None
    return (min(low for low, _ in pool), max(high for _, high in pool))


def clamped_body(bar: Mapping[str, Any]) -> tuple[float, float] | None:
    """(bottom, top) of a body forced inside the bar's own low/high.

    Used for a drawable defect: the candle still shows where the range was and
    which way it closed, and it cannot paint outside the price scale.
    """
    low = _price(bar, "low")
    high = _price(bar, "high")
    open_ = _price(bar, "open")
    close = _price(bar, "close")
    if low is None or high is None or not isfinite(low) or not isfinite(high) or high < low:
        return None
    candidates = [value for value in (open_, close) if value is not None and isfinite(value)]
    if not candidates:
        return (low, high)
    bounded = [min(max(value, low), high) for value in candidates]
    return (min(bounded), max(bounded))


def defect_record(
    symbol: str, timeframe: str, defect: BarDefect, *, source: str = ""
) -> dict[str, Any]:
    """One diagnostics row naming a malformed bar and where it came from."""
    bar = defect.bar
    stamp = bar.get("dt")
    return {
        "symbol": str(symbol or "").upper(),
        "timeframe": str(timeframe or ""),
        "defect": defect.defect,
        "index": defect.index,
        "dt": stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp),
        "open": bar.get("open"),
        "high": bar.get("high"),
        "low": bar.get("low"),
        "close": bar.get("close"),
        "volume": bar.get("volume"),
        "source": str(source or ""),
    }


# -- diagnostics ---------------------------------------------------------
#
# Naming the bad bar is the point: the renderer can stop a corrupt row from
# ruining a chart, but only a record of WHICH symbol, WHICH timestamp and
# WHICH cache handed it over can get the source fixed. Bounded and guarded -
# a diagnostic that breaks a session is worse than no diagnostic.

LOG_NAME = "bad_bars.jsonl"
#: Per-process ceiling. A chart re-renders on every poll; without a cap a
#: single persistently bad row would write until the disk complained.
MAX_RECORDS_PER_SESSION = 200

_seen: set[tuple] = set()
_records = 0


def log_path():
    from pathlib import Path

    from project_paths import get_diagnostics_dir

    return Path(get_diagnostics_dir()) / LOG_NAME


def log_defects(
    symbol: str,
    timeframe: str,
    bars: Sequence[Mapping[str, Any]],
    *,
    source: str = "",
    path=None,
) -> int:
    """Record any malformed bars once each. Returns how many were written.

    Deduplicated on (symbol, timeframe, timestamp, defect) for the life of the
    process, because the same cached series is rebuilt on every refresh and
    the second thousand copies of one row teach nothing the first did not.
    """
    global _records
    import json
    from datetime import datetime

    defects = scan_bars(bars) + range_outliers(bars)
    if not defects:
        return 0
    target = path if path is not None else log_path()
    written = 0
    for defect in defects:
        if _records >= MAX_RECORDS_PER_SESSION:
            break
        record = defect_record(symbol, timeframe, defect, source=source)
        key = (record["symbol"], record["timeframe"], record["dt"], record["defect"])
        if key in _seen:
            continue
        _seen.add(key)
        record["ts"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + chr(10))
        except OSError:
            return written  # diagnostics must never break the session
        _records += 1
        written += 1
    return written
