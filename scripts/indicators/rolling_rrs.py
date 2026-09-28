"""Rolling Real Relative Strength vs SPY (H.S., r/RealDayTrading).

One point read is the desk's existing RRS (``group_rrs.real_relative_strength``,
parity-tested): SPY's move over ``length`` bars in its own ATR is the power
index, and RRS is how far the stock moved beyond ``power * stock_atr``, in the
stock's ATR. The rolling read is the mean of the last ``roll`` point reads, so
one burst bar decays instead of flipping the read, and steady strength scores
higher than a single spike.

Two side reads split the same point reads by SPY's direction:
``hold_on_dips`` is the mean RRS while SPY's power index was negative (a long
leader resists), ``drive_on_rips`` the mean while it was positive (a long
leader outruns). Either is None when SPY never moved that way in the window.

Pure: aligned bars in, floats out. Callers align the two series first
(``group_rrs.align_bars``). Too few bars is None, never zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

FEATURE_VERSION = "rolling_rrs_v1"


@dataclass(frozen=True)
class RollingRrsConfig:
    length: int = 12  # bars in each point read (legacy RRS_LENGTH)
    roll: int = 12  # point reads averaged into the rolling read
    min_reads: int | None = None  # None = all ``roll`` reads required

    def required_reads(self) -> int:
        wanted = self.roll if self.min_reads is None else self.min_reads
        return max(1, min(self.roll, wanted))


@dataclass(frozen=True)
class RollingRrs:
    rolling: float  # mean of the point reads used
    point: float  # latest point read (= today's desk RRS)
    power: float  # latest SPY power index
    hold_on_dips: float | None  # mean point read while SPY power < 0
    drive_on_rips: float | None  # mean point read while SPY power > 0
    reads: tuple[float, ...]  # point reads used, oldest first
    powers: tuple[float, ...]  # SPY power per read, oldest first
    version: str = FEATURE_VERSION


def _value(bar: Any, key: str) -> float | None:
    raw = bar.get(key) if isinstance(bar, Mapping) else getattr(bar, key, None)
    try:
        number = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _wilder_atr_prefixes(bars: Sequence[Any], length: int) -> list[float | None]:
    """ATR of each prefix ``bars[:k+1]``, as ``group_rrs.wilder_atr_last`` computes it."""
    out: list[float | None] = [None] * len(bars)
    atr: float | None = None
    seed: list[float] = []
    for index in range(1, len(bars)):
        high = _value(bars[index], "high")
        low = _value(bars[index], "low")
        prev_close = _value(bars[index - 1], "close")
        if high is None or low is None or prev_close is None:
            # wilder_atr_last answers None for any prefix holding a bad bar.
            return out
        true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if atr is None:
            seed.append(true_range)
            if len(seed) == length:
                atr = sum(seed) / float(length)
        else:
            atr = ((atr * (length - 1)) + true_range) / float(length)
        if atr is not None:
            out[index] = atr if atr > 0 else None
    return out


def rrs_series(
    symbol_bars: Sequence[Any], spy_bars: Sequence[Any], length: int = 12
) -> list[tuple[float | None, float | None]]:
    """(rrs, power) ending at every bar; (None, None) where it cannot be read.

    Entry ``i`` equals ``real_relative_strength(symbol_bars[:i+1], spy_bars[:i+1], length)``.
    """
    symbol_bars = list(symbol_bars or ())
    spy_bars = list(spy_bars or ())
    count = min(len(symbol_bars), len(spy_bars))
    out: list[tuple[float | None, float | None]] = [(None, None)] * count
    if count != len(symbol_bars) or count != len(spy_bars) or length < 1:
        return out  # unaligned input: unknown, not a guess
    sym_atr = _wilder_atr_prefixes(symbol_bars, length)
    spy_atr = _wilder_atr_prefixes(spy_bars, length)
    for end in range(length + 1, count):
        s_atr, m_atr = sym_atr[end - 1], spy_atr[end - 1]
        sym_last = _value(symbol_bars[end], "close")
        sym_prior = _value(symbol_bars[end - length], "close")
        spy_last = _value(spy_bars[end], "close")
        spy_prior = _value(spy_bars[end - length], "close")
        if not s_atr or not m_atr or None in (sym_last, sym_prior, spy_last, spy_prior):
            continue
        power = (spy_last - spy_prior) / m_atr
        out[end] = ((sym_last - sym_prior) - power * s_atr) / s_atr, power
    return out


def rolling_rrs(
    symbol_bars: Sequence[Any],
    spy_bars: Sequence[Any],
    config: RollingRrsConfig | None = None,
) -> RollingRrs | None:
    """Rolling RRS at the last bar, or None when too few point reads exist."""
    config = config or RollingRrsConfig()
    series = rrs_series(symbol_bars, spy_bars, config.length)
    window = series[-config.roll :] if config.roll > 0 else []
    if not window or window[-1][0] is None:
        return None
    pairs = [(rrs, power) for rrs, power in window if rrs is not None and power is not None]
    if len(pairs) < config.required_reads():
        return None
    reads = tuple(rrs for rrs, _ in pairs)
    powers = tuple(power for _, power in pairs)
    dips = [rrs for rrs, power in pairs if power < 0]
    rips = [rrs for rrs, power in pairs if power > 0]
    return RollingRrs(
        rolling=sum(reads) / len(reads),
        point=reads[-1],
        power=powers[-1],
        hold_on_dips=sum(dips) / len(dips) if dips else None,
        drive_on_rips=sum(rips) / len(rips) if rips else None,
        reads=reads,
        powers=powers,
    )
