"""One D1 market environment label per session (`d1_environment_v1`).

WISHLIST item 7, as the lead ruled it for packet WS-ENV: the trader wants to
read every readout CUT BY the kind of day the market was having, and the only
honest way to do that is to decide the kind of day ONCE, from completed daily
bars, under a named and versioned rule - not per surface, per reader, or per
memory of what February felt like.

**This module only NAMES the day.** It reaches no detector, no score, no alert
and no Focus list (plan.md sec 5); the label LABELS a readout and nothing else.

Pure, in the `scripts/indicators/` contract: completed bars in, an immutable
tuple out, `None` for anything unmeasurable. No clock, no I/O, no provider, no
engine import - the RUNNER fetches the bars (through the pinned daily fetch)
and calls this.

The rule, stated once, in this order::

    len(bars) < WARMUP_SESSIONS (34)     -> unknown, reason "warmup"
    atr14 unmeasurable                   -> unknown, reason "unmeasurable"
    range_atr <= COMPRESSION_RANGE_ATR   -> compressed
    slope_atr > +TREND_SLOPE_ATR and close > sma20 -> trending_up
    slope_atr < -TREND_SLOPE_ATR and close < sma20 -> trending_down
    otherwise                            -> mixed

where

* ``atr14`` is Wilder's ATR at the LAST supplied bar, computed over the WHOLE
  supplied series - **including** the 10-session range window. The packet left
  that choice open ("over the bars before the window is fine - state it"); it
  is stated here, it is what the golden fixture was hand-computed under, and it
  is the recurrence `indicators.atr.wilder_atr` already owns rather than a
  fourth copy of Wilder's smoothing;
* ``range_atr`` = (max high - min low over the last ``LOOKBACK_SESSIONS``) /
  ``atr14`` - how much ground the last two weeks covered, measured in units of
  how far this market moves in a day;
* ``slope_atr`` = (SMA20 at the last bar - SMA20 ten sessions earlier) /
  ``atr14`` - the drift over the same window, in the same units.

**Compression is decided FIRST, deliberately.** A quiet market grinding gently
higher inside a three-ATR box is a compressed market; calling it a trend is how
a reader talks themself into size on a day that never went anywhere. On the
recorded SPY series 2026-04-29 is exactly that day: ``slope_atr`` 3.84 with the
close above its SMA20, and ``range_atr`` 2.19 - labelled `compressed`.

The warm-up is 34 bars because SMA20 ten sessions back needs 30 bars and ATR14
needs 15; 34 is the first bar where every term of the rule is measurable. Fewer
is `unknown` with `reason = "warmup"` - a shorter window is not a quieter
market, it is an unread one, and missing data is uncertainty (plan.md sec 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from .atr import wilder_atr

#: The rule's name. Every stored row carries it, and a changed threshold below
#: is a NEW version beside this one, never a silent re-reading of history.
RULE_VERSION = "d1_environment_v1"

#: How far back the range and the slope look, in completed sessions.
LOOKBACK_SESSIONS = 10
#: Wilder's ATR length - the unit both measures are expressed in.
ATR_SESSIONS = 14
#: The trend reference the close is compared against.
SMA_SESSIONS = 20
#: At or under this many ATRs of ground covered in the window, the market is
#: compressed. Inclusive at the boundary.
COMPRESSION_RANGE_ATR = 3.0
#: The drift, in ATRs over the window, a market must carry to be called trending.
TREND_SLOPE_ATR = 0.5
#: The first bar at which every term above is measurable.
WARMUP_SESSIONS = 34

LABEL_UNKNOWN = "unknown"
LABEL_COMPRESSED = "compressed"
LABEL_TRENDING_UP = "trending_up"
LABEL_TRENDING_DOWN = "trending_down"
LABEL_MIXED = "mixed"

#: Every label this rule can produce. A reader that pools `unknown` into one of
#: the others is claiming a reading that was never taken.
LABELS = (
    LABEL_COMPRESSED,
    LABEL_TRENDING_UP,
    LABEL_TRENDING_DOWN,
    LABEL_MIXED,
    LABEL_UNKNOWN,
)

REASON_WARMUP = "warmup"
REASON_UNMEASURABLE = "unmeasurable"

_TIME_KEYS = ("dt", "datetime", "timestamp", "time", "date", "session")
_HIGH_KEYS = ("high", "High", "h")
_LOW_KEYS = ("low", "Low", "l")
_CLOSE_KEYS = ("close", "Close", "c")


@dataclass(frozen=True)
class D1Environment:
    """What one benchmark's tape was doing as of one completed session.

    Frozen: a label that can be edited after the fact is a label nobody can
    join on. Every measured field is `None` rather than 0.0 when it could not
    be measured, because a zero ATR reads as "this market does not move".
    """

    label: str
    rule_version: str
    as_of_session: str
    range_atr: float | None
    slope_atr: float | None
    sma20: float | None
    atr14: float | None
    bars_used: int
    reason: str = ""

    @property
    def is_known(self) -> bool:
        return self.label != LABEL_UNKNOWN


def _field(bar: Any, keys: Sequence[str]) -> float | None:
    """One OHLC field from a dict-like bar or an attribute-style bar."""
    raw: Any = None
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
    # NaN fails this compare, which is the intent: an unreadable price is
    # unmeasurable, never a number.
    return value if value == value else None


def session_of(bar: Any) -> str:
    """The bar's session as `YYYY-MM-DD`, whatever the producer called it."""
    value: Any = None
    if isinstance(bar, Mapping):
        for key in _TIME_KEYS:
            if key in bar and bar[key] is not None:
                value = bar[key]
                break
    else:
        for key in _TIME_KEYS:
            value = getattr(bar, key, None)
            if value is not None:
                break
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    return text[:10]


def _unknown(bars: Sequence[Any], reason: str) -> D1Environment:
    last = bars[-1] if bars else None
    return D1Environment(
        label=LABEL_UNKNOWN,
        rule_version=RULE_VERSION,
        as_of_session=session_of(last) if last is not None else "",
        range_atr=None,
        slope_atr=None,
        sma20=None,
        atr14=None,
        bars_used=len(bars),
        reason=reason,
    )


def _sma(closes: Sequence[float | None], end_index: int, length: int) -> float | None:
    """The simple mean of `length` closes ending at `end_index`, or None."""
    start = end_index - length + 1
    if start < 0:
        return None
    window = closes[start : end_index + 1]
    if len(window) < length or any(value is None for value in window):
        return None
    return sum(float(value) for value in window) / float(length)  # type: ignore[arg-type]


def classify_environment(bars: Sequence[Any]) -> D1Environment:
    """Label one benchmark's session from its completed daily bars, oldest first.

    Point-in-time by construction: the caller passes the bars it had, and the
    answer is about the LAST of them. Nothing here looks forward, and nothing
    here reads a clock - a bar that is still forming must be dropped by the
    caller (`scripts/completed_bars.py`) before it arrives.
    """
    series = list(bars or ())
    if len(series) < WARMUP_SESSIONS:
        return _unknown(series, REASON_WARMUP)

    atr14 = wilder_atr(series, ATR_SESSIONS)
    if atr14 is None or atr14 <= 0:
        return _unknown(series, REASON_UNMEASURABLE)

    highs = [_field(bar, _HIGH_KEYS) for bar in series]
    lows = [_field(bar, _LOW_KEYS) for bar in series]
    closes = [_field(bar, _CLOSE_KEYS) for bar in series]

    last = len(series) - 1
    window_highs = [value for value in highs[-LOOKBACK_SESSIONS:] if value is not None]
    window_lows = [value for value in lows[-LOOKBACK_SESSIONS:] if value is not None]
    sma20 = _sma(closes, last, SMA_SESSIONS)
    prior_sma20 = _sma(closes, last - LOOKBACK_SESSIONS, SMA_SESSIONS)
    close = closes[last]

    if not window_highs or not window_lows or sma20 is None or prior_sma20 is None or close is None:
        return _unknown(series, REASON_UNMEASURABLE)

    range_atr = (max(window_highs) - min(window_lows)) / atr14
    slope_atr = (sma20 - prior_sma20) / atr14

    if range_atr <= COMPRESSION_RANGE_ATR:
        label = LABEL_COMPRESSED
    elif slope_atr > TREND_SLOPE_ATR and close > sma20:
        label = LABEL_TRENDING_UP
    elif slope_atr < -TREND_SLOPE_ATR and close < sma20:
        label = LABEL_TRENDING_DOWN
    else:
        label = LABEL_MIXED

    return D1Environment(
        label=label,
        rule_version=RULE_VERSION,
        as_of_session=session_of(series[last]),
        range_atr=range_atr,
        slope_atr=slope_atr,
        sma20=sma20,
        atr14=atr14,
        bars_used=len(series),
        reason="",
    )
