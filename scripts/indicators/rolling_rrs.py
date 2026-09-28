"""Rolling Real Relative Strength vs SPY (H.S., r/RealDayTrading).

A point read is "stock power minus SPY power": each one's move over the last
hour divided by its own average hourly range. If SPY moved 4x its hourly ATR,
the stock is expected to move 4x its own; RRS is the excess, in the stock's
hourly ATR. The move must sit inside one session (no gap), and the hourly ATR
is the mean high-low range of the last ``atr_hours`` in-session hour blocks
that finished before the move began (gaps excluded, H.S.'s 50 hours).

``atr_mode="bar"`` instead reproduces the desk's existing RRS exactly
(``group_rrs.real_relative_strength``: a Wilder ATR of single bars). It reads
about 4-5x larger and its ratio to the hourly read varies by stock, so the two
scales are not interchangeable.

The rolling read is the mean of the last ``roll`` point reads, so one burst bar
decays instead of flipping RS/RW. ``hold_on_dips`` is the mean read while SPY's
power was negative (a long leader resists), ``drive_on_rips`` while it was
positive (a long leader outruns); either is None if SPY never moved that way.

Pure: aligned bars in, floats out. Callers align the two series first
(``group_rrs.align_bars``). Too few bars is None, never zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from completed_bars import bar_time

FEATURE_VERSION = "rolling_rrs_v2"


@dataclass(frozen=True)
class RollingRrsConfig:
    length: int = 12  # bars in each move (12 M5 bars = one hour)
    roll: int = 12  # point reads averaged into the rolling read
    min_reads: int | None = None  # None = all ``roll`` reads required
    atr_mode: str = "hourly"  # "hourly" (H.S.) or "bar" (desk RRS parity)
    atr_hours: int = 50  # hour blocks averaged into the hourly ATR
    min_atr_hours: int = 20  # fewer finished hour blocks = unknown

    def required_reads(self) -> int:
        wanted = self.roll if self.min_reads is None else self.min_reads
        return max(1, min(self.roll, wanted))


@dataclass(frozen=True)
class RollingRrs:
    rolling: float  # mean of the point reads used
    point: float  # latest point read
    power: float  # latest SPY power index
    hold_on_dips: float | None  # mean point read while SPY power < 0
    drive_on_rips: float | None  # mean point read while SPY power > 0
    reads: tuple[float, ...]  # point reads used, oldest first
    powers: tuple[float, ...]  # SPY power per read, oldest first
    atr_mode: str = "hourly"
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
    """Desk-scale (rrs, power) ending at every bar; (None, None) where it cannot be read.

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


def _session_keys(bars: Sequence[Any]) -> list[Any]:
    keys = []
    for bar in bars:
        stamp = bar_time(bar)
        keys.append(stamp.date() if stamp is not None else None)
    return keys


def _hour_blocks(bars: Sequence[Any], keys: list[Any], length: int) -> list[tuple[int, float | None]]:
    """(last index, high-low range) of each full ``length``-bar block, counted from each session's first bar."""
    blocks: list[tuple[int, float | None]] = []
    start = 0
    while start < len(bars):
        end = start
        while end + 1 < len(bars) and keys[end + 1] is not None and keys[end + 1] == keys[start]:
            end += 1
        if keys[start] is not None:
            for first in range(start, end - length + 2, length):
                chunk = bars[first : first + length]
                highs = [_value(bar, "high") for bar in chunk]
                lows = [_value(bar, "low") for bar in chunk]
                if None in highs or None in lows:
                    blocks.append((first + length - 1, None))
                else:
                    blocks.append((first + length - 1, max(highs) - min(lows)))
        start = end + 1
    return blocks


def _hourly_atrs(
    bars: Sequence[Any], keys: list[Any], length: int, atr_hours: int, min_atr_hours: int
) -> list[float | None]:
    """Hourly ATR usable by a move starting at each index: blocks that ended at or before it."""
    blocks = _hour_blocks(bars, keys, length)
    out: list[float | None] = [None] * len(bars)
    done: list[float | None] = []
    cursor = 0
    for index in range(len(bars)):
        while cursor < len(blocks) and blocks[cursor][0] <= index:
            done.append(blocks[cursor][1])
            cursor += 1
        recent = done[-atr_hours:]
        if len(recent) >= max(1, min_atr_hours) and None not in recent:
            atr = sum(recent) / len(recent)  # type: ignore[arg-type]
            out[index] = atr if atr > 0 else None
    return out


def hourly_rrs_series(
    symbol_bars: Sequence[Any],
    spy_bars: Sequence[Any],
    length: int = 12,
    *,
    atr_hours: int = 50,
    min_atr_hours: int = 20,
) -> list[tuple[float | None, float | None]]:
    """H.S.'s (rrs, power) ending at every bar; (None, None) where it cannot be read."""
    symbol_bars = list(symbol_bars or ())
    spy_bars = list(spy_bars or ())
    count = min(len(symbol_bars), len(spy_bars))
    out: list[tuple[float | None, float | None]] = [(None, None)] * count
    if count != len(symbol_bars) or count != len(spy_bars) or length < 1:
        return out  # unaligned input: unknown, not a guess
    keys = _session_keys(symbol_bars)
    if keys != _session_keys(spy_bars):
        return out
    sym_atr = _hourly_atrs(symbol_bars, keys, length, atr_hours, min_atr_hours)
    spy_atr = _hourly_atrs(spy_bars, keys, length, atr_hours, min_atr_hours)
    for end in range(length, count):
        begin = end - length
        if keys[begin] is None or keys[begin] != keys[end]:
            continue  # the move would span the overnight gap
        s_atr, m_atr = sym_atr[begin], spy_atr[begin]
        sym_last = _value(symbol_bars[end], "close")
        sym_prior = _value(symbol_bars[begin], "close")
        spy_last = _value(spy_bars[end], "close")
        spy_prior = _value(spy_bars[begin], "close")
        if not s_atr or not m_atr or None in (sym_last, sym_prior, spy_last, spy_prior):
            continue
        power = (spy_last - spy_prior) / m_atr
        out[end] = (sym_last - sym_prior) / s_atr - power, power
    return out


def rolling_rrs(
    symbol_bars: Sequence[Any],
    spy_bars: Sequence[Any],
    config: RollingRrsConfig | None = None,
) -> RollingRrs | None:
    """Rolling RRS at the last bar, or None when too few point reads exist."""
    config = config or RollingRrsConfig()
    if config.atr_mode == "bar":
        series = rrs_series(symbol_bars, spy_bars, config.length)
    elif config.atr_mode == "hourly":
        series = hourly_rrs_series(
            symbol_bars,
            spy_bars,
            config.length,
            atr_hours=config.atr_hours,
            min_atr_hours=config.min_atr_hours,
        )
    else:
        raise ValueError(f"unknown atr_mode: {config.atr_mode!r}")
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
        atr_mode=config.atr_mode,
    )
