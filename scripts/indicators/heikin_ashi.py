"""Heikin-Ashi transform and reversal classifier (R5 section 2.3).

Standard HA::

    HA_close = (O + H + L + C) / 4
    HA_open  = (HA_open.prev + HA_close.prev) / 2      # seeded (O + C) / 2
    HA_high  = max(H, HA_open, HA_close)
    HA_low   = min(L, HA_open, HA_close)

The seeding rule is the part worth stating: the first bar has no previous HA
candle, so its open is seeded from the raw ``(O + C) / 2``. Different platforms
seed differently and the first few candles diverge accordingly; this one is
documented so a future comparison against TC2000 has something to compare to
rather than a silent assumption.

A REVERSAL here means the first HA candle whose colour opposes the run it ends
- not merely any colour change, because a single alternating candle inside
chop would otherwise fire on every bar. A run of length 1 still counts as a
run, so alternating candles do each report; callers that want conviction pair
this with the SMI/LRSI confluence window (R5 section 3.2), which is exactly why
the correlator exists.

Doji candles (HA open == HA close) are their own colour, ``FLAT``, and never
end or start a run. Calling a doji green or red would invent a direction the
candle does not have.

Pure and offline: completed bars in, immutable tuples out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

FEATURE_VERSION = "heikin_ashi_v1"

GREEN = "GREEN"
RED = "RED"
FLAT = "FLAT"


@dataclass(frozen=True)
class HeikinAshiBar:
    open: float
    high: float
    low: float
    close: float

    @property
    def color(self) -> str:
        if self.close > self.open:
            return GREEN
        if self.close < self.open:
            return RED
        return FLAT


@dataclass(frozen=True)
class HeikinAshiResult:
    feature_version: str
    bars: tuple[HeikinAshiBar, ...]

    @property
    def colors(self) -> tuple[str, ...]:
        return tuple(bar.color for bar in self.bars)

    def reversal_indices(self) -> tuple[int, ...]:
        """Bars that flip the prevailing direction.

        A flat (doji) candle neither ends a run nor starts one: the run in
        progress simply continues past it, so GREEN, FLAT, RED reports the
        reversal at the RED - which is where the direction actually changed.
        """
        out: list[int] = []
        prevailing = ""
        for index, color in enumerate(self.colors):
            if color == FLAT:
                continue
            if prevailing and color != prevailing:
                out.append(index)
            prevailing = color
        return tuple(out)

    def bullish_reversal_indices(self) -> tuple[int, ...]:
        return tuple(i for i in self.reversal_indices() if self.colors[i] == GREEN)

    def bearish_reversal_indices(self) -> tuple[int, ...]:
        return tuple(i for i in self.reversal_indices() if self.colors[i] == RED)


def compute_heikin_ashi(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> HeikinAshiResult:
    """The HA series over completed bars. Aligns 1:1 with the inputs."""
    length = min(len(opens), len(highs), len(lows), len(closes))
    bars: list[HeikinAshiBar] = []
    previous: HeikinAshiBar | None = None
    for index in range(length):
        raw_open = float(opens[index])
        raw_high = float(highs[index])
        raw_low = float(lows[index])
        raw_close = float(closes[index])
        ha_close = (raw_open + raw_high + raw_low + raw_close) / 4.0
        if previous is None:
            ha_open = (raw_open + raw_close) / 2.0
        else:
            ha_open = (previous.open + previous.close) / 2.0
        bar = HeikinAshiBar(
            open=ha_open,
            high=max(raw_high, ha_open, ha_close),
            low=min(raw_low, ha_open, ha_close),
            close=ha_close,
        )
        bars.append(bar)
        previous = bar
    return HeikinAshiResult(FEATURE_VERSION, tuple(bars))
