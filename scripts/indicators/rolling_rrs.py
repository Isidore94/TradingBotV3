"""Rolling Real Relative Strength vs SPY (H.S., r/RealDayTrading).

A point read is "stock power minus SPY power": each one's move over the last
hour divided by its own average hourly range. If SPY moved 4x its hourly ATR,
the stock is expected to move 4x its own; RRS is the excess, in the stock's
hourly ATR. The move must sit inside one session (no gap), and the hourly ATR
is the mean high-low range of the last ``atr_hours`` one-hour candles that
finished before the move began (H.S.'s ATR50(H), gaps excluded). Pass real
hourly candles for both symbols when the M5 series is too short for 50 hours;
otherwise hours are cut from the M5 bars, from each session's first bar.

``atr_mode="bar"`` instead reproduces the desk's existing RRS exactly
(``group_rrs.real_relative_strength``: a Wilder ATR of single bars). It reads
about 4-5x larger and its ratio to the hourly read varies by stock, so the two
scales are not interchangeable.

``atr_mode="daily"`` is the same idea on daily candles: the move over
``length`` days against each symbol's mean daily true range over the last
``daily_atr_days`` days before the move (gaps count on the daily).

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

from datetime import timedelta

from completed_bars import align_to, bar_time

FEATURE_VERSION = "rolling_rrs_v2"


@dataclass(frozen=True)
class RollingRrsConfig:
    length: int = 12  # bars in each move (12 M5 bars = one hour)
    roll: int = 12  # point reads averaged into the rolling read
    min_reads: int | None = None  # None = all ``roll`` reads required
    atr_mode: str = "hourly"  # "hourly" (H.S.), "daily", or "bar" (desk RRS parity)
    atr_hours: int = 50  # hour blocks averaged into the hourly ATR
    min_atr_hours: int = 20  # fewer finished hour blocks = unknown
    daily_atr_days: int = 50  # daily candles averaged into the daily ATR
    min_daily_atr_days: int = 20  # fewer = unknown
    sample_every: int = 1  # average every Nth read (3 = 15m candles from M5 reads)

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


def _candle_atrs(
    bars: Sequence[Any],
    hour_bars: Sequence[Any],
    atr_hours: int,
    min_atr_hours: int,
    bar_minutes: int,
) -> list[float | None]:
    """Hourly ATR from one-hour candles that had closed by the end of each M5 bar."""
    candles = []
    for candle in hour_bars or ():
        stamp = bar_time(candle)
        high, low = _value(candle, "high"), _value(candle, "low")
        if stamp is not None:
            candles.append((stamp, None if high is None or low is None else high - low))
    candles.sort(key=lambda item: item[0])
    out: list[float | None] = [None] * len(bars)
    done: list[float | None] = []
    cursor = 0
    for index, bar in enumerate(bars):
        stamp = bar_time(bar)
        if stamp is None:
            continue
        bar_end = stamp + timedelta(minutes=bar_minutes)
        while cursor < len(candles) and align_to(candles[cursor][0], bar_end) + timedelta(hours=1) <= bar_end:
            done.append(candles[cursor][1])
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
    symbol_hour_bars: Sequence[Any] | None = None,
    spy_hour_bars: Sequence[Any] | None = None,
    bar_minutes: int = 5,
) -> list[tuple[float | None, float | None]]:
    """H.S.'s (rrs, power) ending at every bar; (None, None) where it cannot be read.

    Hourly candles are used only when given for BOTH symbols, so the two ATRs
    are always measured the same way.
    """
    symbol_bars = list(symbol_bars or ())
    spy_bars = list(spy_bars or ())
    count = min(len(symbol_bars), len(spy_bars))
    out: list[tuple[float | None, float | None]] = [(None, None)] * count
    if count != len(symbol_bars) or count != len(spy_bars) or length < 1:
        return out  # unaligned input: unknown, not a guess
    keys = _session_keys(symbol_bars)
    if keys != _session_keys(spy_bars):
        return out
    if symbol_hour_bars and spy_hour_bars:
        sym_atr = _candle_atrs(symbol_bars, symbol_hour_bars, atr_hours, min_atr_hours, bar_minutes)
        spy_atr = _candle_atrs(spy_bars, spy_hour_bars, atr_hours, min_atr_hours, bar_minutes)
    else:
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


def daily_rrs_series(
    symbol_bars: Sequence[Any],
    spy_bars: Sequence[Any],
    length: int = 5,
    *,
    atr_days: int = 50,
    min_atr_days: int = 20,
) -> list[tuple[float | None, float | None]]:
    """(rrs, power) ending at every daily candle; (None, None) where it cannot be read."""
    symbol_bars = list(symbol_bars or ())
    spy_bars = list(spy_bars or ())
    count = min(len(symbol_bars), len(spy_bars))
    out: list[tuple[float | None, float | None]] = [(None, None)] * count
    if count != len(symbol_bars) or count != len(spy_bars) or length < 1:
        return out
    sym_tr = _true_ranges(symbol_bars)
    spy_tr = _true_ranges(spy_bars)
    for end in range(length, count):
        begin = end - length
        s_atr = _mean_recent(sym_tr, begin, atr_days, min_atr_days)
        m_atr = _mean_recent(spy_tr, begin, atr_days, min_atr_days)
        sym_last = _value(symbol_bars[end], "close")
        sym_prior = _value(symbol_bars[begin], "close")
        spy_last = _value(spy_bars[end], "close")
        spy_prior = _value(spy_bars[begin], "close")
        if not s_atr or not m_atr or None in (sym_last, sym_prior, spy_last, spy_prior):
            continue
        power = (spy_last - spy_prior) / m_atr
        out[end] = (sym_last - sym_prior) / s_atr - power, power
    return out


def _true_ranges(bars: Sequence[Any]) -> list[float | None]:
    """True range of each candle (None for the first and for bad candles)."""
    out: list[float | None] = [None] * len(bars)
    for index in range(1, len(bars)):
        high = _value(bars[index], "high")
        low = _value(bars[index], "low")
        prev_close = _value(bars[index - 1], "close")
        if None not in (high, low, prev_close):
            out[index] = max(high - low, abs(high - prev_close), abs(low - prev_close))  # type: ignore[operator]
    return out


def _mean_recent(values: list[float | None], last: int, count: int, minimum: int) -> float | None:
    """Mean of up to ``count`` values ending at index ``last``; None if too few or any bad."""
    recent = values[max(1, last - count + 1) : last + 1]
    if len(recent) < max(1, minimum) or None in recent:
        return None
    mean = sum(recent) / len(recent)  # type: ignore[arg-type]
    return mean if mean > 0 else None


def rrs_point_series(
    symbol_bars: Sequence[Any],
    spy_bars: Sequence[Any],
    config: RollingRrsConfig | None = None,
    *,
    symbol_hour_bars: Sequence[Any] | None = None,
    spy_hour_bars: Sequence[Any] | None = None,
) -> list[tuple[float | None, float | None]]:
    """The point (rrs, power) series for ``config.atr_mode``, one entry per bar."""
    config = config or RollingRrsConfig()
    if config.atr_mode == "bar":
        return rrs_series(symbol_bars, spy_bars, config.length)
    if config.atr_mode == "hourly":
        return hourly_rrs_series(
            symbol_bars,
            spy_bars,
            config.length,
            atr_hours=config.atr_hours,
            min_atr_hours=config.min_atr_hours,
            symbol_hour_bars=symbol_hour_bars,
            spy_hour_bars=spy_hour_bars,
        )
    if config.atr_mode == "daily":
        return daily_rrs_series(
            symbol_bars,
            spy_bars,
            config.length,
            atr_days=config.daily_atr_days,
            min_atr_days=config.min_daily_atr_days,
        )
    raise ValueError(f"unknown atr_mode: {config.atr_mode!r}")


def rolling_from_series(
    series: Sequence[tuple[float | None, float | None]],
    config: RollingRrsConfig | None = None,
) -> RollingRrs | None:
    """Roll an existing point series (one computed series can feed several candle sizes)."""
    config = config or RollingRrsConfig()
    step = max(1, config.sample_every)
    sampled = list(series)[::-1][::step][: max(0, config.roll)][::-1]
    if not sampled or sampled[-1][0] is None:
        return None
    pairs = [(rrs, power) for rrs, power in sampled if rrs is not None and power is not None]
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


def rolling_rrs(
    symbol_bars: Sequence[Any],
    spy_bars: Sequence[Any],
    config: RollingRrsConfig | None = None,
    *,
    symbol_hour_bars: Sequence[Any] | None = None,
    spy_hour_bars: Sequence[Any] | None = None,
) -> RollingRrs | None:
    """Rolling RRS at the last bar, or None when too few point reads exist.

    Hour bars feed the hourly ATR (hourly mode only); see ``hourly_rrs_series``.
    """
    config = config or RollingRrsConfig()
    series = rrs_point_series(
        symbol_bars,
        spy_bars,
        config,
        symbol_hour_bars=symbol_hour_bars,
        spy_hour_bars=spy_hour_bars,
    )
    return rolling_from_series(series, config)
