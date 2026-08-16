"""Weekend strength boards on H1, D1 and Monthly (R8 §4/§5).

The TC2000 formula is unchanged on every timeframe and is **imported**, never
copied: `scripts/strength_scan.py` is fenced by the spec's §2 and §8 and is not
edited by this packet. What is reimplemented here is the ~20 lines of board
orchestration around it — the completed-bar rule and the filters, both of which
are genuinely per-timeframe.

WHAT "COMPLETED" MEANS, PER TIMEFRAME

The invariant is one sentence — completed bars only, a forming bar is preview —
and it needs three different tests to enforce:

* **H1**: the bar has closed when `bar_start + 60min <= now`. Clock arithmetic,
  because an hourly bar's identity *is* its duration.
* **D1**: the bar's date is at or before `market_calendar.last_completed_session`.
  Not "yesterday": a Monday holiday means Friday is the last completed session,
  and counting back a day would score a day the market never traded.
* **Monthly**: drop any bar whose `(year, month)` equals the current month —
  **identity, never duration**. A monthly bar is stamped on the 1st and grows
  all month; on the 1st of a month that bar is minutes old and, measured by
  duration, would look like a complete previous month. The build-start probe
  confirmed the live frame's last row is the in-progress month.

FILTERS (spec §5, trader-approved as proposed 2026-08-15)

Session VWAP is dropped above M5 rather than imitated — there is no session
inside an H1, D1 or monthly bar, and a look-alike would produce a number that
reads like the M5 gate's and means something else. Each leg is its own named
function so one timeframe can be amended without touching the others, and a leg
that cannot be measured **fails with a reason** rather than passing by default.

ORDER

Percentile cut first, filters second — identical to the R2 M5 board. The cut
describes the measurable population; filtering first would take the top 25% of
an already-filtered set and quietly mean something else.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable

import market_calendar
from strength_scan import (
    STRENGTH_ATR_PERIOD,
    STRENGTH_BODY_BARS,
    STRENGTH_EMA_SPAN,
    atr,
    displaced_close,
    ema,
    percentile_cut,
    strength_score,
)

#: Top/bottom fraction of the measurable population. Same as the M5 board.
PERCENTILE_FRACTION = 0.25

#: 51 bars: ATR50 needs 50 true ranges, and a true range needs a previous close.
MIN_BARS = STRENGTH_ATR_PERIOD + 1


@dataclass(frozen=True)
class StrengthTimeframe:
    key: str
    label: str
    yf_interval: str
    yf_period: str
    bar_kind: str
    bar_minutes: int
    body_bars: int = STRENGTH_BODY_BARS
    atr_period: int = STRENGTH_ATR_PERIOD
    ema_span: int = STRENGTH_EMA_SPAN


H1 = StrengthTimeframe("h1", "Hourly", "1h", "3mo", "intraday", 60)
D1 = StrengthTimeframe("d1", "Daily", "1d", "1y", "session", 1440)
# 6y, not period="max": 51 completed months is ~4.35 years, so 6 clears the
# minimum with margin while keeping memory bounded. A short-history name returns
# None and is reported missing rather than scored.
M1 = StrengthTimeframe("m1", "Monthly", "1mo", "6y", "month", 0)

TIMEFRAMES: tuple[StrengthTimeframe, ...] = (H1, D1, M1)
TIMEFRAMES_BY_KEY = {timeframe.key: timeframe for timeframe in TIMEFRAMES}


def _finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _bar_time(bar: Mapping[str, Any]) -> datetime | None:
    """A bar's start time, whatever the producer called the field.

    ``dt`` is first because it is what ``autopilot_core._frame_rows`` actually
    emits, and that is the real fetch path. Omitting it meant every bar from a
    live download had "no readable timestamp" and was silently dropped - a board
    that would have measured nothing while every hand-built unit test passed.
    Found by the one test that went through the downloader instead of around it.
    """
    value = bar.get("dt") or bar.get("timestamp") or bar.get("time") or bar.get("date")
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.fromisoformat(text[:10])
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Completed bars, three ways
# ---------------------------------------------------------------------------


def completed_bars(
    timeframe: StrengthTimeframe, bars: Sequence[Mapping[str, Any]], *, now: datetime
) -> list[Mapping[str, Any]]:
    """Drop whatever is still forming, by this timeframe's own rule."""
    if timeframe.bar_kind == "month":
        return _completed_months(bars, now=now)
    if timeframe.bar_kind == "session":
        return _completed_sessions(bars, now=now)
    return _completed_intraday(timeframe, bars, now=now)


def _completed_months(bars: Sequence[Mapping[str, Any]], *, now: datetime) -> list[Mapping[str, Any]]:
    """Month identity, never duration.

    On the 1st of a month the in-progress bar is minutes old; a duration test
    ("has 30 days passed?") would call the *previous* month incomplete and this
    one complete, which is exactly backwards. Comparing (year, month) is right
    on every day of the month including the first.
    """
    current = (now.year, now.month)
    kept: list[Mapping[str, Any]] = []
    for bar in bars:
        stamp = _bar_time(bar)
        if stamp is None:
            continue
        if (stamp.year, stamp.month) == current:
            continue
        kept.append(bar)
    return kept


def _completed_sessions(bars: Sequence[Mapping[str, Any]], *, now: datetime) -> list[Mapping[str, Any]]:
    """Date <= the last completed NYSE session.

    Not "yesterday": after a Monday holiday the last completed session is
    Friday, and counting back a calendar day would score a day the market never
    opened. Fails open to a plain yesterday if the calendar cannot answer, since
    the alternative is scoring nothing at all.
    """
    try:
        cutoff = market_calendar.last_completed_session(now)
    except Exception:
        cutoff = (now - timedelta(days=1)).date()
    kept: list[Mapping[str, Any]] = []
    for bar in bars:
        stamp = _bar_time(bar)
        if stamp is None:
            continue
        if stamp.date() <= cutoff:
            kept.append(bar)
    return kept


def _completed_intraday(
    timeframe: StrengthTimeframe, bars: Sequence[Mapping[str, Any]], *, now: datetime
) -> list[Mapping[str, Any]]:
    """bar_start + bar_minutes <= now."""
    span = timedelta(minutes=max(1, int(timeframe.bar_minutes)))
    kept: list[Mapping[str, Any]] = []
    for bar in bars:
        stamp = _bar_time(bar)
        if stamp is None:
            continue
        if stamp.tzinfo is not None and now.tzinfo is None:
            stamp = stamp.astimezone(_local_timezone()).replace(tzinfo=None)
        elif stamp.tzinfo is None and now.tzinfo is not None:
            stamp = stamp.replace(tzinfo=now.tzinfo)
        if stamp + span <= now:
            kept.append(bar)
    return kept


def _local_timezone():
    return datetime.now().astimezone().tzinfo


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_symbol(
    timeframe: StrengthTimeframe, symbol: str, bars: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """One row's raw numbers, or None when the history cannot support them.

    Measures only. Which side asked does not change a single number here, which
    is what lets the same measurement feed both the long and the short board.
    """
    if len(bars) < MIN_BARS:
        return None
    score = strength_score(bars, body_bars=timeframe.body_bars, atr_period=timeframe.atr_period)
    if score is None:
        return None
    closes = [_finite(bar.get("close")) for bar in bars]
    if any(close is None for close in closes):
        return None
    return {
        "symbol": str(symbol).upper(),
        "timeframe": timeframe.key,
        "score": score,
        "last_close": closes[-1],
        "ema": ema(closes, timeframe.ema_span),  # type: ignore[arg-type]
        "atr": atr(bars, timeframe.atr_period),
        "displaced_close": displaced_close(closes, timeframe.atr_period),  # type: ignore[arg-type]
        "bars": list(bars),
        "bar_count": len(bars),
    }


# ---------------------------------------------------------------------------
# Filters (spec §5). One named function per leg, per timeframe.
# ---------------------------------------------------------------------------


def _above_or_below(value: float | None, reference: float | None, side: str) -> bool | None:
    if value is None or reference is None:
        return None
    return value > reference if side == "long" else value < reference


def leg_trend_vs_ema(row: Mapping[str, Any], side: str) -> tuple[bool, str]:
    """Last close on the right side of EMA15 of this timeframe's closes."""
    verdict = _above_or_below(row.get("last_close"), row.get("ema"), side)
    if verdict is None:
        return False, "EMA15 could not be measured"
    return verdict, "" if verdict else ("not above EMA15" if side == "long" else "not below EMA15")


def _group_extreme(bars: Sequence[Mapping[str, Any]], key: Callable[[datetime], Any], side: str):
    """The prior completed group's high (long) or low (short), and its label."""
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    order: list[Any] = []
    for bar in bars:
        stamp = _bar_time(bar)
        if stamp is None:
            continue
        bucket = key(stamp)
        if bucket not in groups:
            groups[bucket] = []
            order.append(bucket)
        groups[bucket].append(bar)
    if len(order) < 2:
        return None, "no prior completed period to compare against"
    previous = groups[order[-2]]
    field_name = "high" if side == "long" else "low"
    values = [_finite(bar.get(field_name)) for bar in previous]
    values = [value for value in values if value is not None]
    if not values:
        return None, f"prior period has no usable {field_name}"
    return (max(values) if side == "long" else min(values)), ""


def leg_prior_session_extreme(row: Mapping[str, Any], side: str) -> tuple[bool, str]:
    """H1: beyond the prior completed session's high (long) / low (short)."""
    extreme, reason = _group_extreme(row.get("bars") or [], lambda s: s.date(), side)
    if extreme is None:
        return False, reason
    verdict = _above_or_below(row.get("last_close"), extreme, side)
    if verdict is None:
        return False, "last close could not be measured"
    return verdict, "" if verdict else f"not beyond prior session {'high' if side == 'long' else 'low'}"


def leg_prior_week_extreme(row: Mapping[str, Any], side: str) -> tuple[bool, str]:
    """D1: beyond the prior completed ISO week's high (long) / low (short)."""
    extreme, reason = _group_extreme(row.get("bars") or [], lambda s: s.isocalendar()[:2], side)
    if extreme is None:
        return False, reason
    verdict = _above_or_below(row.get("last_close"), extreme, side)
    if verdict is None:
        return False, "last close could not be measured"
    return verdict, "" if verdict else f"not beyond prior week {'high' if side == 'long' else 'low'}"


def leg_prior_month_extreme(row: Mapping[str, Any], side: str) -> tuple[bool, str]:
    """M1: last completed month's close beyond the previous month's extreme.

    The monthly leg is deliberately only this one. An EMA15 of monthly closes is
    fifteen *months* of history on top of the 51 the score already needs, which
    would silently exclude names the board should be showing.
    """
    extreme, reason = _group_extreme(row.get("bars") or [], lambda s: (s.year, s.month), side)
    if extreme is None:
        return False, reason
    verdict = _above_or_below(row.get("last_close"), extreme, side)
    if verdict is None:
        return False, "last close could not be measured"
    return verdict, "" if verdict else f"not beyond prior month {'high' if side == 'long' else 'low'}"


FILTERS: dict[str, tuple[Callable[[Mapping[str, Any], str], tuple[bool, str]], ...]] = {
    "h1": (leg_trend_vs_ema, leg_prior_session_extreme),
    "d1": (leg_trend_vs_ema, leg_prior_week_extreme),
    "m1": (leg_prior_month_extreme,),
}


def passes_filters(timeframe: StrengthTimeframe, row: Mapping[str, Any], side: str) -> tuple[bool, str]:
    """Every leg, in order. The first refusal is the reason reported."""
    for leg in FILTERS[timeframe.key]:
        ok, reason = leg(row, side)
        if not ok:
            return False, reason or f"{leg.__name__} failed"
    return True, ""


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------


@dataclass
class WeekendBoard:
    timeframe: str
    side: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    offered: int = 0
    measured: int = 0
    in_percentile: int = 0
    filtered_out: int = 0
    as_of: str = ""

    @property
    def accounting(self) -> str:
        """The honest one-liner the panel shows under every board."""
        missing = self.offered - self.measured
        text = (
            f"{self.offered} offered, {self.measured} measurable, "
            f"{self.in_percentile} in the top/bottom {int(PERCENTILE_FRACTION * 100)}%, "
            f"{len(self.rows)} after filters"
        )
        if missing:
            text += f" ({missing} had too little history to score)"
        return text


def build_board(
    timeframe: StrengthTimeframe,
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    side: str = "long",
    now: datetime,
    fraction: float = PERCENTILE_FRACTION,
) -> WeekendBoard:
    """Measure, cut, then filter — in that order, and the order is the point.

    Cutting first means the percentile describes the *measurable population*.
    Filtering first would take the top 25% of an already-filtered set, which is
    a different statement about a different set, and would silently disagree
    with the M5 board the trader already reads.
    """
    board = WeekendBoard(timeframe=timeframe.key, side=side, as_of=now.isoformat(timespec="seconds"))
    board.offered = len(bars_by_symbol)

    measured: dict[str, dict[str, Any]] = {}
    for symbol, raw_bars in bars_by_symbol.items():
        trimmed = completed_bars(timeframe, raw_bars, now=now)
        row = measure_symbol(timeframe, symbol, trimmed)
        if row is not None:
            measured[row["symbol"]] = row
    board.measured = len(measured)

    cut = percentile_cut(
        [(symbol, row["score"]) for symbol, row in measured.items()], fraction=fraction, side=side
    )
    board.in_percentile = len(cut)

    rows: list[dict[str, Any]] = []
    for symbol, score in cut:
        row = measured[symbol]
        ok, reason = passes_filters(timeframe, row, side)
        if not ok:
            board.filtered_out += 1
            continue
        rows.append(
            {
                "symbol": symbol,
                "side": side,
                "timeframe": timeframe.key,
                "score": score,
                "last_close": row["last_close"],
                "ema": row["ema"],
                "atr": row["atr"],
                "bar_count": row["bar_count"],
                "reason": reason,
            }
        )
    board.rows = rows
    return board
