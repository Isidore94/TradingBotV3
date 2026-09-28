"""Daily rolling RRS between two daily-bar series matched on their dates.

``indicators.rolling_rrs`` wants two aligned lists; daily stores are lists of
dicts (``date`` or ``datetime``) or pandas frames that can miss a day. This
module pairs the two series on the dates both have, up to an optional
``through`` date, and answers None (unknown) when the symbol's last day is
missing from the reference or the history is too short.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from indicators.rolling_rrs import (
    RollingRrs,
    RollingRrsConfig,
    daily_rrs_series,
    rolling_from_series,
)


def _day(bar: Any) -> str | None:
    """The bar's session date as ISO text, from ``date`` or ``datetime``."""
    raw = None
    if isinstance(bar, Mapping):
        raw = bar.get("date") or bar.get("datetime")
    else:
        raw = getattr(bar, "date", None) or getattr(bar, "datetime", None)
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    if hasattr(raw, "date") and callable(raw.date):  # pandas Timestamp
        try:
            return raw.date().isoformat()
        except (TypeError, ValueError):
            return None
    text = str(raw).strip()
    return text[:10] if len(text) >= 10 else None


def frame_to_bars(frame: Any) -> list[dict]:
    """A daily OHLC pandas frame (``datetime`` column) as a list of dicts."""
    if frame is None or getattr(frame, "empty", True):
        return []
    columns = [c for c in ("datetime", "date", "open", "high", "low", "close") if c in frame.columns]
    return [dict(record) for record in frame[columns].to_dict("records")]


def align_daily(
    symbol_bars: Iterable[Any],
    reference_bars: Iterable[Any],
    *,
    through: date | str | None = None,
) -> tuple[list[Any], list[Any]]:
    """Both series cut to the dates they share (oldest first), none after ``through``.

    Empty lists when the symbol's last day (through ``through``) is not in the
    reference: a read that silently used an older day would not be as-of.
    """
    limit = through.isoformat() if isinstance(through, date) else (str(through)[:10] if through else None)
    symbol_by_day: dict[str, Any] = {}
    for bar in symbol_bars or ():
        day = _day(bar)
        if day and (limit is None or day <= limit):
            symbol_by_day[day] = bar
    reference_by_day: dict[str, Any] = {}
    for bar in reference_bars or ():
        day = _day(bar)
        if day and (limit is None or day <= limit):
            reference_by_day[day] = bar
    if not symbol_by_day or max(symbol_by_day) not in reference_by_day:
        return [], []
    days = sorted(set(symbol_by_day) & set(reference_by_day))
    return [symbol_by_day[d] for d in days], [reference_by_day[d] for d in days]


def daily_point_series(
    symbol_bars: Sequence[Any],
    reference_bars: Sequence[Any],
    length: int,
    config: RollingRrsConfig,
    *,
    through: date | str | None = None,
) -> list[tuple[float | None, float | None]]:
    """(rrs, power) per shared day for an N-day move, with ``config``'s daily ATR."""
    sym, ref = align_daily(symbol_bars, reference_bars, through=through)
    return daily_rrs_series(
        sym,
        ref,
        length,
        atr_days=config.daily_atr_days,
        min_atr_days=config.min_daily_atr_days,
    )


def daily_rolling_rrs(
    symbol_bars: Sequence[Any],
    reference_bars: Sequence[Any],
    config: RollingRrsConfig,
    *,
    through: date | str | None = None,
) -> RollingRrs | None:
    """Rolling daily RRS of ``symbol_bars`` vs ``reference_bars`` at the last shared day."""
    series = daily_point_series(symbol_bars, reference_bars, config.length, config, through=through)
    return rolling_from_series(series, config)


def latest_point(
    symbol_bars: Sequence[Any],
    reference_bars: Sequence[Any],
    length: int,
    config: RollingRrsConfig,
    *,
    through: date | str | None = None,
) -> float | None:
    """The newest single N-day point read (not rolled), or None."""
    series = daily_point_series(symbol_bars, reference_bars, length, config, through=through)
    return series[-1][0] if series else None
