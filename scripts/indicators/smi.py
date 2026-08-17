"""Stochastic Momentum Index, TC2000 parity (R5 section 2.1).

The trader's TC2000 formula, recorded verbatim in WISHLIST history (commit
`994f575`)::

    XUP(
      XAVG(XAVG(C - (MAXH5 + MINL5)/2, 5), 20)
      / XAVG(XAVG(MAXH5 - MINL5, 5), 20),
      XAVG(..., 6)
    )

Read literally, that is:

- ``(MAXH5 + MINL5) / 2`` - the midpoint of the 5-bar high/low range;
- ``C - midpoint`` - how far the close sits from that midpoint;
- both that distance and the raw range ``MAXH5 - MINL5`` are smoothed by a
  5-period EMA and then a 20-period EMA (TC2000's ``XAVG`` is an EMA);
- the ratio of the two is the SMI line, ``sm1``;
- a 6-period EMA of ``sm1`` is the signal line, ``sm2``.

**The parity detail that matters**: the double smoothing is applied to the
NUMERATOR AND DENOMINATOR SEPARATELY, and only then divided. Smoothing the
ratio instead gives a visibly different curve, and it is the mistake this
formula invites - so `test_smi.py` carries a fixture that fails under it.

TC2000 does not halve the denominator, so neither does this. The conventional
Blau SMI uses ``range / 2`` and lands on a -100..100 scale; the trader reads
the TC2000 curve, so parity with TC2000 wins over convention. The output here
is a bare ratio (roughly -1..1 in normal conditions), not a percentage.

Signal of interest, per the spec: ``sm1 < sm2`` with **both below zero**, then
``sm1`` crossing above ``sm2``. That is a momentum turn from a washed-out
state, not merely any crossover.

Pure and offline: completed bars in, immutable tuples out. No provider calls,
no clock, no ledger. ``None`` marks a bar the indicator cannot answer for
rather than a fabricated zero - a warm-up bar is unmeasurable, not neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

FEATURE_VERSION = "smi_tc2000_v1"


@dataclass(frozen=True)
class SmiConfig:
    range_length: int = 5
    first_smoothing: int = 5
    second_smoothing: int = 20
    signal_smoothing: int = 6

    def __post_init__(self) -> None:
        for name in (
            "range_length",
            "first_smoothing",
            "second_smoothing",
            "signal_smoothing",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass(frozen=True)
class SmiResult:
    feature_version: str
    sm1: tuple[float | None, ...]
    sm2: tuple[float | None, ...]

    def bullish_cross_indices(self) -> tuple[int, ...]:
        """Bars where sm1 crosses ABOVE sm2 with both lines below zero.

        Both-below-zero is part of the signal, not a filter bolted on after:
        the trader's interest is a turn out of a washed-out state. A crossover
        happening above zero is a different event and is deliberately not
        reported here.
        """
        out: list[int] = []
        for index in range(1, len(self.sm1)):
            previous_1, previous_2 = self.sm1[index - 1], self.sm2[index - 1]
            current_1, current_2 = self.sm1[index], self.sm2[index]
            if None in (previous_1, previous_2, current_1, current_2):
                continue
            if previous_1 >= previous_2:
                continue
            if not current_1 > current_2:
                continue
            # "Both below zero" is measured at the moment of the cross.
            if current_1 < 0 and current_2 < 0:
                out.append(index)
        return tuple(out)


def _ema(values: Sequence[float | None], length: int) -> list[float | None]:
    """EMA that starts at the first non-None value and holds gaps.

    A None in the middle of the series is carried forward as None rather than
    treated as zero: the alternative silently pulls the average toward zero and
    prints a confident number for a bar whose input was missing.
    """
    multiplier = 2.0 / (float(length) + 1.0)
    out: list[float | None] = []
    running: float | None = None
    for value in values:
        if value is None:
            out.append(None)
            continue
        value = float(value)
        running = value if running is None else (value - running) * multiplier + running
        out.append(running)
    return out


def compute_smi(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    config: SmiConfig | None = None,
) -> SmiResult:
    """SMI over completed bars. Series align 1:1 with the inputs."""
    config = config or SmiConfig()
    length = min(len(highs), len(lows), len(closes))
    if length == 0:
        return SmiResult(FEATURE_VERSION, (), ())

    window = int(config.range_length)
    distances: list[float | None] = []
    ranges: list[float | None] = []
    for index in range(length):
        if index + 1 < window:
            # Fewer bars than the range needs. Unmeasurable, not zero.
            distances.append(None)
            ranges.append(None)
            continue
        highest = max(float(value) for value in highs[index + 1 - window : index + 1])
        lowest = min(float(value) for value in lows[index + 1 - window : index + 1])
        distances.append(float(closes[index]) - (highest + lowest) / 2.0)
        ranges.append(highest - lowest)

    # Numerator and denominator are smoothed SEPARATELY and divided last.
    smoothed_distance = _ema(
        _ema(distances, config.first_smoothing), config.second_smoothing
    )
    smoothed_range = _ema(_ema(ranges, config.first_smoothing), config.second_smoothing)

    sm1: list[float | None] = []
    for numerator, denominator in zip(smoothed_distance, smoothed_range):
        if numerator is None or denominator is None or denominator == 0:
            # A motionless window has no range to normalize by. That is
            # unmeasurable, not neutral - reporting 0.0 would say "at the
            # midpoint", which is a claim the data does not support.
            sm1.append(None)
            continue
        sm1.append(numerator / denominator)

    sm2 = _ema(sm1, config.signal_smoothing)
    return SmiResult(FEATURE_VERSION, tuple(sm1), tuple(sm2))
