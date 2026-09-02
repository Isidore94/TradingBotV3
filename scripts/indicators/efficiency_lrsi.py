"""TC2000-style "LRSI" efficiency oscillator (R5 section 2.2).

**This is NOT `indicators/laguerre_rsi.py`.** That module is Ehlers' Laguerre
RSI with fractal-energy modulation - a different algorithm entirely, with zero
importers, deliberately excluded from the frozen bundle. The trader calls the
oscillator below "LRSI" because TC2000 does, and the spec is explicit that it
must carry a distinct name so the two can never be confused in a call site, a
fixture, or an alert label.

The trader's TC2000 source::

    SUM(ABS(C >= EMA9.prev) * (EMA9 - EMA9.prev), 4)
    / SUM(ABS(EMA9 - EMA9.prev), 4) * 100

That is an efficiency ratio over EMA9 CHANGES: net directional movement of the
EMA over the last 4 bars, divided by total absolute movement, as a percentage.

Two readings of the source are possible and they differ:

- 0..100, where the numerator is the signed net change and negative net
  movement clamps at 0; or
- -100..100, signed.

The spec pins it: **"range 0-100"**, and the crossing states are "up through
20" and "up through 50". Both of those only make sense on a 0..100 scale, so a
downward-efficient window reads as a LOW value, not a negative one. The
numerator is therefore the signed net EMA change and the result is clamped at
zero on the downside.

100 means every one of the last 4 EMA steps went the same way - a perfectly
efficient move. 50 means half the motion was retraced. Near 0 means the EMA
churned or fell.

Pure and offline: completed bars in, immutable tuples out. ``None`` marks a bar
the indicator cannot answer for - a warm-up bar, or a window where the EMA did
not move at all and efficiency is genuinely undefined rather than zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

FEATURE_VERSION = "efficiency_lrsi_tc2000_v1"

#: The crossing levels the trader reads. Up through 20 is the strongest tell
#: (a name coming out of pure churn); up through 50 is the ordinary one.
#: **Live M5 levels - do not change these.** The M5 signal engines are
#: calibrated to them.
CROSS_LEVELS: tuple[float, ...] = (20.0, 50.0)

#: Levels the SHADOW research lane reads (Phase 0.12 B3), additive to the live
#: pair. 80 is the "this move is nearly perfectly efficient" line; the research
#: grid uses cross-up through 20/50 for its long legs and cross-DOWN through
#: 50/80 for its short legs.
#:
#: **Why the short legs are unmirrored, and what that costs.** The formula
#: clamps the numerator at zero, so this series is not symmetric: a perfectly
#: efficient DOWN move and a motionless one both read 0. There are therefore
#: two genuinely different short features available, and they are not
#: interchangeable:
#:
#: * the MIRRORED-close idiom the live M5 engines use - negate the closes and
#:   read cross-UP - which measures how efficient the DOWN move is; and
#: * this series read with cross-DOWN, which measures the UP move's efficiency
#:   collapsing.
#:
#: The research grid takes the second. Three reasons, in order of weight:
#:
#: 1. **The four legs have to be comparable.** A grid whose long legs read one
#:    series and whose short legs read another is not asking one question about
#:    one line; it is two studies sharing a table, and "does this line pay on
#:    the short side?" stops being answerable from it.
#: 2. **The mirrored idiom is already the live engines'.** Re-running it here
#:    would produce a shadow result that reads like evidence about a champion
#:    detector when it is not, which is exactly the confusion plan.md sec 7
#:    exists to prevent.
#: 3. **It matches how the trader reads the oscillator** - "the efficiency is
#:    going" - in the same units as the long legs, on the same series.
#:
#: The cost is real and stated: this measures EXHAUSTION, not down-momentum,
#: and it fires earlier. `tests/fixtures/efficiency_lrsi_research_v1.json`
#: pins the gap on one series - the unmirrored down-cross at bar 27, the
#: mirrored up-cross at bar 29. If the study wants down-momentum later, that
#: is a SECOND registered feature, never a reinterpretation of this one.
RESEARCH_CROSS_LEVELS: tuple[float, ...] = (20.0, 50.0, 80.0)


@dataclass(frozen=True)
class EfficiencyLrsiConfig:
    ema_length: int = 9
    sum_length: int = 4

    def __post_init__(self) -> None:
        if int(self.ema_length) < 1:
            raise ValueError("ema_length must be at least 1")
        if int(self.sum_length) < 1:
            raise ValueError("sum_length must be at least 1")


@dataclass(frozen=True)
class EfficiencyLrsiResult:
    feature_version: str
    values: tuple[float | None, ...]
    ema: tuple[float | None, ...]

    def cross_up_indices(self, level: float) -> tuple[int, ...]:
        """Bars where the oscillator crossed UP through ``level``.

        Strictly a crossing: the previous bar must be at or below the level and
        the current bar strictly above it. A series that is already above the
        level does not keep re-reporting - that would turn one event into an
        alert every five minutes, which is the spam R4 section 6.3 exists to
        undo.
        """
        out: list[int] = []
        for index in range(1, len(self.values)):
            previous, current = self.values[index - 1], self.values[index]
            if previous is None or current is None:
                continue
            if previous <= level < current:
                out.append(index)
        return tuple(out)

    def cross_down_indices(self, level: float) -> tuple[int, ...]:
        """The short-side mirror, and the ORB flow's pullback trigger."""
        out: list[int] = []
        for index in range(1, len(self.values)):
            previous, current = self.values[index - 1], self.values[index]
            if previous is None or current is None:
                continue
            if previous >= level > current:
                out.append(index)
        return tuple(out)


def _ema(values: Sequence[float], length: int) -> list[float | None]:
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


def compute_efficiency_lrsi(
    closes: Sequence[float],
    config: EfficiencyLrsiConfig | None = None,
) -> EfficiencyLrsiResult:
    """The oscillator over completed bars. Aligns 1:1 with ``closes``."""
    config = config or EfficiencyLrsiConfig()
    if not closes:
        return EfficiencyLrsiResult(FEATURE_VERSION, (), ())

    ema = _ema(closes, config.ema_length)
    steps: list[float | None] = [None]
    for index in range(1, len(ema)):
        previous, current = ema[index - 1], ema[index]
        steps.append(None if previous is None or current is None else current - previous)

    window = int(config.sum_length)
    values: list[float | None] = []
    for index in range(len(steps)):
        chunk = steps[max(0, index + 1 - window) : index + 1]
        if len(chunk) < window or any(step is None for step in chunk):
            values.append(None)
            continue
        gross = sum(abs(step) for step in chunk)
        if gross == 0:
            # The EMA did not move at all across the window. Efficiency is
            # undefined, not zero: a motionless name is unmeasurable, not
            # maximally inefficient. (A flat series is legitimate here
            # precisely because flatness IS the property under test.)
            values.append(None)
            continue
        net = sum(chunk)
        values.append(max(0.0, net / gross * 100.0))

    return EfficiencyLrsiResult(FEATURE_VERSION, tuple(values), tuple(ema))
