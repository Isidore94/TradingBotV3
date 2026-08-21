"""Wilder's Average True Range (R5 pure-indicator contract).

Written once, here, because the repo already carries the rule twice and the
copies disagree: `bounce_bot_lib.legacy._wilder_atr_last` is Wilder-smoothed
and private to a detector, while `market_state._m5_atr` is a plain mean of the
last N true ranges under the same name. Neither is importable as a shared
rule, so a third caller either picks a private name out of a legacy module or
writes a fourth copy. This is the shared one; the older two migrate
opportunistically, never as a silent change to a shipped detector.

Wilder's smoothing, stated once::

    TR_i  = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR_n = mean(TR_1 .. TR_n)                        # the seed
    ATR_i = (ATR_(i-1) * (n - 1) + TR_i) / n          # thereafter

The first true range needs a previous close, so ``length`` bars of ATR need
``length + 1`` bars of input. Fewer than that is **unmeasurable, not zero**:
this returns ``None``, and a caller that treats None as 0 turns "I do not know
how fast this moves" into "it does not move", which is the exact failure a
distance-in-ATR test would then wave through.

Pure and offline: completed bars in, a float or None out. No clock, no I/O,
no provider, no detector imports.

**Why an intraday ATR at all** (trader, 2026-08-21): "holding highs should be
a measure of its ATR not its % because a stock like MRK moves slower than say
MU, we can't use the 1% rule." Measured on one real batch that day, M5 ATR
ranged from 0.084% of price (HMC) to 1.160% (CIFR) - a 14x spread inside a
single alert - so any fixed percentage is simultaneously far too loose for one
name and far too tight for another.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

DEFAULT_LENGTH = 14

_HIGH_KEYS = ("high", "High", "h")
_LOW_KEYS = ("low", "Low", "l")
_CLOSE_KEYS = ("close", "Close", "c")


def _value(bar: Any, keys: Sequence[str]) -> float | None:
    """One OHLC field from a dict-like bar or an attribute-style bar."""
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
    # unmeasurable, and a NaN true range would poison every later ATR through
    # the smoothing recurrence.
    return value if value == value else None


def true_ranges(bars: Sequence[Any]) -> list[float]:
    """TR for every bar after the first. Empty when nothing is measurable."""
    ranges: list[float] = []
    previous_close: float | None = None
    for bar in bars or ():
        high = _value(bar, _HIGH_KEYS)
        low = _value(bar, _LOW_KEYS)
        close = _value(bar, _CLOSE_KEYS)
        if high is None or low is None or close is None:
            # A bar we cannot read breaks the chain rather than being skipped
            # past: pairing bar i-2's close with bar i's high would invent a
            # true range that no two adjacent bars ever produced.
            previous_close = None
            continue
        if previous_close is not None:
            ranges.append(
                max(high - low, abs(high - previous_close), abs(low - previous_close))
            )
        previous_close = close
    return ranges


def wilder_atr(bars: Sequence[Any], length: int = DEFAULT_LENGTH) -> float | None:
    """Wilder ATR at the last bar, or ``None`` when it cannot be measured."""
    span = max(1, int(length))
    ranges = true_ranges(bars)
    if len(ranges) < span:
        return None
    atr = sum(ranges[:span]) / float(span)
    for value in ranges[span:]:
        atr = ((atr * (span - 1)) + value) / float(span)
    return atr if atr > 0 else None
