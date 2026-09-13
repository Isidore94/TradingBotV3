"""The ``h1_ema_bounce_v1`` rule sheet: an H1 15-EMA retest, frozen (WISHLIST 10C).

The trader's brief: on a name whose weekly pattern they already like, they do
not want to chase - they want to be told when the hourly chart comes back to
its 15-EMA and holds it. That is **entry timing, never a claim**: this module
answers "did the retest print", and nothing here says the trade is good.

What "the bounce" means, stated once so it can be argued with:

* a **touch** - a completed H1 bar whose low (long) or high (short) comes
  within ``TOUCH_TOLERANCE_ATR`` of the 15-EMA, measured in ATR so a slow name
  and a fast one are held to the same standard (the desk's ATR rule, trader
  2026-08-21);
* a **reclaim** - the LAST completed bar closes back past the EMA in the
  trade's direction by at least ``REJECTION_CLOSE_ATR``;
* within ``MAX_TOUCH_AGE_BARS`` bars of each other, with the EMA itself moving
  the trade's way over ``SLOPE_LOOKBACK_BARS`` bars.

Deliberately NOT the same question as the shipped ``assess_h1_riding_ema15``
in the intraday bounce engine, which asks whether price is *riding* the line -
several closes beyond it. Riding is a trend description; this is a pullback
that got bought. The packet checked before reusing it, and it is a different
rule.

Three refusals, each of which is the useful answer rather than a smaller one:

* fewer than ``WARMUP_BARS`` bars -> ``None``. An EMA seeded a dozen bars ago
  is mostly seed, and "not measured" is not "no".
* bars that stopped arriving (``now`` more than ``STALE_AFTER`` past the last
  completed bar) -> ``None``. A cached series from Friday answers nothing
  about Monday.
* a touch and a reclaim on the SAME bar -> ``ambiguous``, never fired. Inside
  one hourly candle there is no order of events, so "it tagged the line and
  then closed back above it" is a story the bar cannot support.

Pure: completed bar dicts in, one frozen result out. No clock of its own, no
store, no Qt, no network - and nothing from the intraday bounce engine's
package: importing it drags ``ibapi`` and ~1,050 modules (measured 2026-09-13,
2.55 s), so the H1 aggregation below is a faithful COPY of that package's
``_closed_h1_bars`` reading the dict bars ``BounceBot.m5_chart_bars`` returns,
rather than its ``IbBar`` objects. ``tests/test_ws_10c_h1_retester.py`` pins
the copy bar-for-bar against the shipped original and runs the import in a
subprocess to prove the engine stays out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

RULE_VERSION = "h1_ema_bounce_v1"

#: The line the trader watches on the hourly chart.
EMA_LENGTH = 15
#: Bars of history before the EMA is a level rather than a seed (~8 sessions).
WARMUP_BARS = 45
#: How close the wick has to come to count as a touch, in ATR.
TOUCH_TOLERANCE_ATR = 0.25
#: How far past the line the confirming close has to finish, in ATR.
REJECTION_CLOSE_ATR = 0.10
#: How many bars a touch stays live for (0 = the confirm bar itself).
MAX_TOUCH_AGE_BARS = 3
#: A close this far the WRONG side of the line ends the episode.
INVALIDATION_CLOSE_ATR = 1.0
#: The EMA has to be moving the trade's way over this many bars.
SLOPE_LOOKBACK_BARS = 5
#: Past this gap between the last completed bar and ``now``, nothing is
#: measured. An ordinary overnight gap is at most ~19 hours (the 12:30 bucket
#: to the next session's first completed hour), so this never blanks a normal
#: morning; a weekend or a dead feed does, which is the point.
STALE_AFTER = timedelta(hours=24)

H1_SPAN = timedelta(minutes=60)
M5_SPAN = timedelta(minutes=5)

REASON_CONFIRMED = "bounce_confirmed"
REASON_AMBIGUOUS = "ambiguous"
REASON_NO_TOUCH = "no_touch"
REASON_INVALIDATED = "invalidated"
REASON_AWAITING = "awaiting_reclaim"
REASON_SLOPE_AGAINST = "slope_against"

_LONG = "long"
_SHORT = "short"


@dataclass(frozen=True)
class H1Bounce:
    """One evaluation of the rule against one series.

    ``reason`` is the verdict; ``reasons`` is every measured observation behind
    it, because the packet asks for ONE event per watch recording all of them -
    a fired row that says only "confirmed" cannot be argued with later.
    """

    fired: bool
    reason: str
    side: str
    rule_version: str = RULE_VERSION
    touch_bar_dt: datetime | None = None
    confirm_bar_dt: datetime | None = None
    ema: float | None = None
    atr: float | None = None
    distance_atr: float | None = None
    skipped_bars: int = 0
    reasons: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Bars in
# ---------------------------------------------------------------------------
def _naive(moment: datetime) -> datetime:
    # The chart-watch store's convention: IB serves this desk's bars on the
    # local clock, and arm times come from the same clock, so comparisons drop
    # tzinfo rather than convert across zones.
    return moment.replace(tzinfo=None) if moment.tzinfo is not None else moment


def _bar(raw: Mapping[str, Any] | Any) -> dict[str, Any] | None:
    """One readable bar dict, or ``None`` when it is not a candle.

    plan.md sec 5: ``low <= open, close <= high``. A bar that breaks the
    invariant is not a price - averaging it into the EMA would move the very
    level being measured - so it is dropped and COUNTED, never repaired.
    """
    if isinstance(raw, Mapping):
        get = raw.get
    else:  # pragma: no cover - attribute-style bars are not on this path today
        def get(key, default=None):
            return getattr(raw, key, default)
    stamp = get("dt")
    if not isinstance(stamp, datetime):
        return None
    try:
        values = {name: float(get(name)) for name in ("open", "high", "low", "close")}
    except (TypeError, ValueError):
        return None
    if any(value != value for value in values.values()):  # NaN
        return None
    low, high = values["low"], values["high"]
    if low > high:
        return None
    if low > values["open"] or low > values["close"]:
        return None
    if values["open"] > high or values["close"] > high:
        return None
    values["dt"] = _naive(stamp)
    return values


def _readable(bars: Iterable[Any] | None) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    skipped = 0
    for raw in bars or ():
        bar = _bar(raw)
        if bar is None:
            skipped += 1
        else:
            kept.append(bar)
    kept.sort(key=lambda bar: bar["dt"])
    return kept, skipped


def _session_open(reference: datetime) -> datetime | None:
    try:
        from market_session import get_market_session_open_naive

        return get_market_session_open_naive(reference=reference)
    except Exception:
        return None


def _session_close(reference: datetime) -> datetime | None:
    try:
        from market_session import get_market_session_close_naive

        return get_market_session_close_naive(reference=reference)
    except Exception:
        return None


def closed_h1_bars(m5_bars: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Session-aligned H1 bars from M5 bars, with the forming bucket dropped.

    A faithful copy of the intraday engine's ``_closed_h1_bars`` (and the
    bucketing half of its ``_aggregate_bars_timeframe``) over dict bars. Two
    properties are load-bearing and are why this is not a clock-aligned
    ``dt.hour`` group-by:

    * buckets start at the SESSION open (06:30 local -> 06:30, 07:30 ... 12:30),
      so the last bucket of the day is the short 30-minute hour;
    * a bucket is closed once the M5 data itself reaches its end - bucket start
      + 60 minutes, or the session close, whichever is earlier. Completeness
      comes from the tape, not from a ``now`` argument, which is what lets the
      short final candle count the moment the bell rings.

    Takes no ``now``: a forming bucket is a preview and simply is not returned.
    """
    bars, _ = _readable(m5_bars)
    if not bars:
        return []

    buckets: dict[datetime, dict[str, Any]] = {}
    order: list[datetime] = []
    for bar in bars:
        stamp = bar["dt"]
        market_open = _session_open(stamp)
        if market_open is None:
            minutes_since = -1
        else:
            minutes_since = int((stamp - market_open).total_seconds() // 60)
        if minutes_since < 0 or market_open is None:
            bucket_start = stamp.replace(minute=0, second=0, microsecond=0)
        else:
            bucket_start = market_open + timedelta(minutes=(minutes_since // 60) * 60)
        existing = buckets.get(bucket_start)
        if existing is None:
            buckets[bucket_start] = {
                "dt": bucket_start,
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
            }
            order.append(bucket_start)
        else:
            existing["high"] = max(existing["high"], bar["high"])
            existing["low"] = min(existing["low"], bar["low"])
            existing["close"] = bar["close"]

    if not order:
        return []
    last = order[-1]
    bucket_end = last + H1_SPAN
    session_close = _session_close(last)
    if session_close is not None and session_close > last:
        bucket_end = min(bucket_end, session_close)
    if bars[-1]["dt"] + M5_SPAN < bucket_end:
        order = order[:-1]
    return [buckets[start] for start in order]


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------
def ema_series(closes: Sequence[float], length: int = EMA_LENGTH) -> list[float]:
    """Close-based EMA seeded with the FIRST close.

    The convention both existing copies in this repo already use (the intraday
    engine's ``_ema_series`` and ``chart_watch._ema_last``). Seeding
    with an SMA instead would move every level this rule measures against, so
    the golden pins the resulting floats.
    """
    if not closes:
        return []
    alpha = 2.0 / (float(max(1, int(length))) + 1.0)
    value = float(closes[0])
    out = [value]
    for close in closes[1:]:
        value = alpha * float(close) + (1.0 - alpha) * value
        out.append(value)
    return out


def _side(raw: object) -> str | None:
    text = str(raw or "").strip().lower()
    if text in ("long", "buy"):
        return _LONG
    if text in ("short", "sell"):
        return _SHORT
    return None


def evaluate(
    h1_bars: Iterable[Any] | None,
    side: str,
    *,
    atr: float | None,
    now: datetime | None = None,
) -> H1Bounce | None:
    """The rule, anchored at the LAST completed bar. ``None`` = not measured.

    ``atr`` is the H1 ATR14 the caller measured (``indicators.atr.wilder_atr``);
    without it nothing here is comparable across names, so it is not optional
    and a missing one is "not measured" rather than a distance of zero.
    """
    direction = _side(side)
    if direction is None:
        return None
    try:
        atr_value = float(atr) if atr is not None else None
    except (TypeError, ValueError):
        atr_value = None
    if atr_value is None or not atr_value > 0:
        return None

    bars, skipped = _readable(h1_bars)
    if len(bars) < WARMUP_BARS:
        return None
    if now is not None and _naive(now) - bars[-1]["dt"] > STALE_AFTER:
        return None

    ema = ema_series([bar["close"] for bar in bars])
    last = len(bars) - 1
    confirm = bars[last]
    confirm_ema = ema[last]
    window_start = max(0, last - MAX_TOUCH_AGE_BARS)
    reasons: list[str] = []
    long_side = direction == _LONG

    def signed(value: float) -> float:
        """Positive = the trade's way, negative = against it."""
        return value if long_side else -value

    # 1. Did the level fail? A close a full ATR the wrong side of the line
    #    inside the window ends the episode, whatever else printed.
    for index in range(window_start, last + 1):
        against = -signed(bars[index]["close"] - ema[index])
        if against >= INVALIDATION_CLOSE_ATR * atr_value:
            reasons.append(f"close {against / atr_value:.2f} ATR through the 15-EMA")
            return H1Bounce(
                fired=False,
                reason=REASON_INVALIDATED,
                side=direction,
                confirm_bar_dt=bars[index]["dt"],
                ema=ema[index],
                atr=atr_value,
                skipped_bars=skipped,
                reasons=tuple(reasons),
            )

    # 2. The most recent touch inside the age window.
    touch_index: int | None = None
    for index in range(last, window_start - 1, -1):
        extreme = bars[index]["low"] if long_side else bars[index]["high"]
        distance = abs(extreme - ema[index])
        if distance <= TOUCH_TOLERANCE_ATR * atr_value:
            touch_index = index
            break
    if touch_index is None:
        return H1Bounce(
            fired=False,
            reason=REASON_NO_TOUCH,
            side=direction,
            confirm_bar_dt=confirm["dt"],
            ema=confirm_ema,
            atr=atr_value,
            skipped_bars=skipped,
            reasons=("no bar tagged the 15-EMA inside the age window",),
        )

    touch_bar = bars[touch_index]
    touch_extreme = touch_bar["low"] if long_side else touch_bar["high"]
    distance_atr = abs(touch_extreme - ema[touch_index]) / atr_value
    age = last - touch_index
    reasons.append(
        f"tagged the 15-EMA {distance_atr:.2f} ATR away at "
        f"{touch_bar['dt'].strftime('%H:%M')}"
    )

    reclaim = signed(confirm["close"] - confirm_ema)
    reclaimed = reclaim >= REJECTION_CLOSE_ATR * atr_value
    base = {
        "side": direction,
        "touch_bar_dt": touch_bar["dt"],
        "confirm_bar_dt": confirm["dt"],
        "ema": confirm_ema,
        "atr": atr_value,
        "distance_atr": distance_atr,
        "skipped_bars": skipped,
    }

    # 3. One candle cannot order its own events.
    if touch_index == last and reclaimed:
        reasons.append("the touch and the reclaim are the same candle")
        return H1Bounce(fired=False, reason=REASON_AMBIGUOUS, reasons=tuple(reasons), **base)

    if not reclaimed:
        reasons.append(
            f"last close {reclaim / atr_value:+.2f} ATR from the line - no reclaim yet"
        )
        return H1Bounce(fired=False, reason=REASON_AWAITING, reasons=tuple(reasons), **base)

    # 4. The line itself has to be going the trade's way.
    slope_from = max(0, last - SLOPE_LOOKBACK_BARS)
    slope = signed(confirm_ema - ema[slope_from])
    if not slope > 0:
        reasons.append(f"the 15-EMA is not rising the trade's way over {SLOPE_LOOKBACK_BARS} bars")
        return H1Bounce(fired=False, reason=REASON_SLOPE_AGAINST, reasons=tuple(reasons), **base)

    reasons.append(
        f"closed {reclaim / atr_value:.2f} ATR back through the line at "
        f"{confirm['dt'].strftime('%H:%M')}"
    )
    reasons.append(f"confirmed {age} bar(s) after the touch")
    reasons.append(f"the 15-EMA is rising over the last {SLOPE_LOOKBACK_BARS} bars")
    return H1Bounce(fired=True, reason=REASON_CONFIRMED, reasons=tuple(reasons), **base)
