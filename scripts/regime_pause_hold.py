"""Is this name still holding its high (or pressing its low), right now?

The regime-pause watch tells the trader a symbol is "holding highs". On
2026-08-21 it said that about MRK while MRK's high of day was **75 minutes
old** and price was fading off it. Two things were wrong, and this module is
the measurement that answers both:

1. **Distance was never measured at all.** The detector's third qualifying
   branch is "fell less than SPY", which a name in free-fall can satisfy. Any
   distance test has to be in **ATR, not percent** (trader, same day): "a stock
   like MRK moves slower than say MU, we can't use the 1% rule." Measured on
   that day's own batch, M5 ATR ran from 0.084% of price to 1.160% - a 14x
   spread inside one alert - so a single percentage is both far too loose and
   far too tight at the same time. Tolerance is **1.0 ATR** (trader's call).

2. **Nothing re-measured.** One alert per symbol per day, fired within a few
   candles of a SPY pause, then displayed unchanged for hours. So a claim that
   was true at 08:30 was still on screen at 09:40. The rule (trader): the alert
   is good for **15 minutes**, and it is deleted after that **unless it
   continues to make a new extreme** - a new high refreshes the clock, because
   a name printing new highs is exactly what the alert claims.

Shorts are the mirror throughout: low of day, pressing lows, SPY bouncing.

Pure and offline. Completed bars in (via `completed_bars`), a verdict out; no
clock of its own, no I/O, no detector imports, and it decides nothing on its
own - callers use it to display and to expire a queue row. Deleting a row
NEVER deletes evidence: the alert list, the review-event stream and the
tracker's outcome rows are written before any display decision and are not
consulted by one (trader's call, 2026-08-21).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from completed_bars import bar_time, completed_m5_bars
from indicators.atr import DEFAULT_LENGTH, wilder_atr

#: How far off the session extreme a name may sit and still be "holding".
#: Trader's call, 2026-08-21, from that day's measured batch: 1.0 ATR kept
#: about half the names on distance alone and dropped MRK (1.78 ATR) clear.
HOLD_TOLERANCE_ATR = 1.0

#: How long a "holding" claim is good for without a new extreme.
HOLD_FRESHNESS_MINUTES = 15

#: hold_state reasons.
AT_EXTREME = "at_extreme"
WITHIN_TOLERANCE = "within_tolerance"
TOO_FAR = "too_far"
UNMEASURABLE = "unmeasurable"

#: queue_verdict reasons.
FRESH_ALERT = "fresh_alert"
NEW_EXTREME = "new_extreme"
EXPIRED_STALE = "expired_stale"

_HIGH_KEYS = ("high", "High", "h")
_LOW_KEYS = ("low", "Low", "l")
_CLOSE_KEYS = ("close", "Close", "c")


def _is_short(side: Any) -> bool:
    return str(side or "").strip().upper().startswith("SHORT")


def _price(bar: Any, keys: Sequence[str]) -> float | None:
    if isinstance(bar, Mapping):
        for key in keys:
            if key in bar:
                raw = bar[key]
                break
        else:
            return None
    else:
        for key in keys:
            raw = getattr(bar, key, None)
            if raw is not None:
                break
        else:
            return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    # NaN fails this compare, which is the intent.
    return value if value == value else None


def _align(first: datetime, second: datetime) -> tuple[datetime, datetime]:
    """Make two stamps comparable by ATTACHING a zone to the naive one.

    Never by stripping the aware one. `_gate_moment` learned that the hard way
    on 2026-08-19: ``replace(tzinfo=None)`` discards an offset instead of
    converting through it, which ends the crash and keeps the outage.
    """
    if (first.tzinfo is None) == (second.tzinfo is None):
        return first, second
    if first.tzinfo is None:
        return first.replace(tzinfo=second.tzinfo), second
    return first, second.replace(tzinfo=first.tzinfo)


@dataclass(frozen=True)
class HoldState:
    """Where price sits against its own session extreme, in ATR."""

    holding: bool
    reason: str
    side: str
    extreme: float | None = None
    extreme_at: datetime | None = None
    close: float | None = None
    atr: float | None = None
    distance_atr: float | None = None
    bars_since_extreme: int | None = None

    def describe(self) -> str:
        """One short phrase for a chart header or an alert row."""
        if self.reason == UNMEASURABLE:
            return "hold unmeasurable"
        word = "LOD" if _is_short(self.side) else "HOD"
        if self.reason == AT_EXTREME:
            return f"new {word}"
        if self.distance_atr is None:
            return f"{word} unmeasured"
        age = ""
        if self.bars_since_extreme is not None:
            age = f", {word} {self.bars_since_extreme * 5} min old"
        return f"{self.distance_atr:.1f} ATR off {word}{age}"


@dataclass(frozen=True)
class QueueVerdict:
    """Whether a queued regime-pause row may still be shown."""

    keep: bool
    reason: str
    hold: HoldState
    expires_at: datetime | None = None


def session_extreme(
    bars: Sequence[Any], side: Any
) -> tuple[float | None, datetime | None, int | None]:
    """(extreme, when it was set, how many bars ago) for one side."""
    short = _is_short(side)
    keys = _LOW_KEYS if short else _HIGH_KEYS
    rows = list(bars or ())
    best: float | None = None
    best_at: datetime | None = None
    best_index: int | None = None
    for index, bar in enumerate(rows):
        value = _price(bar, keys)
        if value is None:
            continue
        # Strict compare, so a level that is merely EQUALLED later does not
        # refresh the clock. Equalling a high is not making a new one, and the
        # freshness rule exists to catch exactly the name that stopped.
        if best is None or (value < best if short else value > best):
            best = value
            best_at = bar_time(bar)
            best_index = index
    if best is None:
        return None, None, None
    ago = len(rows) - 1 - best_index if best_index is not None else None
    return best, best_at, ago


def _last_session(bars: Sequence[Any]) -> list[Any]:
    """The tail of ``bars`` sharing the last bar's calendar date.

    Bars whose timestamp cannot be read are kept with the tail rather than
    dropped: an unreadable stamp is uncertainty, and discarding a bar would
    silently narrow the session the extreme is taken from.
    """
    rows = list(bars or ())
    if not rows:
        return rows
    last = bar_time(rows[-1])
    if last is None:
        return rows
    session: list[Any] = []
    for bar in reversed(rows):
        stamp = bar_time(bar)
        if stamp is not None and stamp.date() != last.date():
            break
        session.append(bar)
    session.reverse()
    return session


def hold_state(
    bars: Sequence[Any],
    side: Any,
    *,
    now: datetime,
    tolerance_atr: float = HOLD_TOLERANCE_ATR,
    atr_length: int = DEFAULT_LENGTH,
) -> HoldState:
    """Measure "holding" on the completed bars only.

    ``bars`` should be the symbol's cached M5 series. Pass MORE than today's
    session when you have it: an ATR(14) needs 15 bars, and 42 minutes after
    the open there are only nine - which is precisely when this detector fires
    most. The extreme is taken from whatever slice the caller passes, so hand
    in the session you mean.
    """
    completed = completed_m5_bars(bars or (), now=now)
    side_text = "SHORT" if _is_short(side) else "LONG"
    if not completed:
        return HoldState(holding=False, reason=UNMEASURABLE, side=side_text)
    # ATR over everything supplied; the EXTREME over the last bar's session
    # only. Hand this two sessions and both are right: the ATR is measurable
    # from the first minute of the day, and "high of day" still means today.
    # Doing it here rather than asking callers to pass two lists removes the
    # footgun where a two-session series quietly makes yesterday's high the
    # one a name is judged against.
    session = _last_session(completed)
    extreme, extreme_at, bars_since = session_extreme(session, side_text)
    close = _price(completed[-1], _CLOSE_KEYS)
    atr = wilder_atr(completed, atr_length)
    if extreme is not None and close is not None and not atr and bars_since == 0:
        # No ATR - too few bars, or a series with no range at all - but the
        # extreme was set on the last completed bar. Being AT the high is a
        # fact that needs no tolerance to state, and refusing to state it would
        # silently switch the whole rule off early in a session: an ATR(14)
        # needs fifteen bars, and the sweep fires while there are nine.
        return HoldState(
            holding=True,
            reason=AT_EXTREME,
            side=side_text,
            extreme=extreme,
            extreme_at=extreme_at,
            close=close,
            atr=atr,
            bars_since_extreme=bars_since,
        )
    if extreme is None or close is None or not atr:
        # Off the extreme with no ATR: the DISTANCE is what cannot be judged,
        # and inventing a tolerance to judge it with is the one thing not to do.
        return HoldState(
            holding=False,
            reason=UNMEASURABLE,
            side=side_text,
            extreme=extreme,
            extreme_at=extreme_at,
            close=close,
            atr=atr,
            bars_since_extreme=bars_since,
        )
    gap = (close - extreme) if _is_short(side_text) else (extreme - close)
    distance = max(0.0, gap) / atr
    if bars_since == 0:
        reason, holding = AT_EXTREME, True
    elif distance <= float(tolerance_atr):
        reason, holding = WITHIN_TOLERANCE, True
    else:
        reason, holding = TOO_FAR, False
    return HoldState(
        holding=holding,
        reason=reason,
        side=side_text,
        extreme=extreme,
        extreme_at=extreme_at,
        close=close,
        atr=atr,
        distance_atr=distance,
        bars_since_extreme=bars_since,
    )


def queue_verdict(
    bars: Sequence[Any],
    side: Any,
    *,
    alert_time: datetime | None,
    now: datetime,
    minutes: float = HOLD_FRESHNESS_MINUTES,
    tolerance_atr: float = HOLD_TOLERANCE_ATR,
) -> QueueVerdict:
    """Whether a queued "holding highs" row has earned its place on screen.

    Kept while ``now`` is within ``minutes`` of the LATER of the alert and the
    last new extreme - the trader's rule stated once: good for fifteen minutes,
    unless it keeps making new highs.

    **Uncertainty never deletes.** No bars, no readable timestamp, no ATR: the
    row is kept and labelled, because a symbol whose cache has not warmed yet
    is not a symbol that stopped making highs. The trader can still retire it
    by hand, and that is a decision rather than a guess.
    """
    state = hold_state(bars, side, now=now, tolerance_atr=tolerance_atr)
    if state.reason == UNMEASURABLE:
        return QueueVerdict(keep=True, reason=UNMEASURABLE, hold=state)
    anchor = alert_time
    extreme_at = state.extreme_at
    if anchor is not None and extreme_at is not None:
        anchor, extreme_at = _align(anchor, extreme_at)
        anchor = max(anchor, extreme_at)
    elif anchor is None:
        anchor = extreme_at
    if anchor is None:
        return QueueVerdict(keep=True, reason=UNMEASURABLE, hold=state)
    anchor, moment = _align(anchor, now)
    expires_at = anchor + timedelta(minutes=float(minutes))
    if moment <= expires_at:
        reason = NEW_EXTREME if state.reason == AT_EXTREME else FRESH_ALERT
        return QueueVerdict(keep=True, reason=reason, hold=state, expires_at=expires_at)
    return QueueVerdict(
        keep=False, reason=EXPIRED_STALE, hold=state, expires_at=expires_at
    )
