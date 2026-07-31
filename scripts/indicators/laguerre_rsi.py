"""Ehlers Laguerre RSI with optional Fractal Energy modulation.

The implementation is deliberately independent of the live application:

* inputs are completed OHLCV arrays;
* outputs are immutable tuples;
* there are no provider calls, clocks, ledgers, or Technical Integrity imports;
* the multi-timeframe helper drops an incomplete trailing aggregate.

Fractal Energy is the normalized logarithmic path-length/range ratio commonly
used by the Mobius-style Laguerre RSI.  A directional window has energy near
zero; a window whose true-range path repeatedly traverses a small price span
has energy near one.  In adaptive mode that value is the Laguerre ``gamma``:
higher energy therefore produces more smoothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


FEATURE_VERSION = "laguerre_rsi_fe_v1"


class LaguerreState(str, Enum):
    """Four post-threshold states of the oscillator."""

    TREND_UP = "TREND_UP"
    DECAY_UP = "DECAY_UP"
    TREND_DOWN = "TREND_DOWN"
    DECAY_DOWN = "DECAY_DOWN"


@dataclass(frozen=True)
class LaguerreRsiConfig:
    price_source: str = "close"
    fractal_energy_lookback: int = 13
    upper_threshold: float = 0.8
    lower_threshold: float = 0.2
    fixed_gamma: float | None = None

    def __post_init__(self) -> None:
        allowed_sources = {"open", "high", "low", "close", "hl2", "hlc3", "ohlc4"}
        if self.price_source not in allowed_sources:
            raise ValueError(f"price_source must be one of {sorted(allowed_sources)}")
        if int(self.fractal_energy_lookback) < 2:
            raise ValueError("fractal_energy_lookback must be at least 2")
        if not 0.0 <= float(self.lower_threshold) < float(self.upper_threshold) <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= lower < upper <= 1")
        if self.fixed_gamma is not None and not 0.0 <= float(self.fixed_gamma) < 1.0:
            raise ValueError("fixed_gamma must satisfy 0 <= gamma < 1")


@dataclass(frozen=True)
class LaguerreRsiResult:
    feature_version: str
    oscillator: tuple[float, ...]
    fractal_energy: tuple[float | None, ...]
    gamma: tuple[float, ...]
    states: tuple[LaguerreState | None, ...]


@dataclass(frozen=True)
class MultiTimeframeLaguerreResult:
    feature_version: str
    base: LaguerreRsiResult
    higher: LaguerreRsiResult
    higher_factor: int
    higher_completed_base_indices: tuple[int, ...]
    higher_state_on_base: tuple[LaguerreState | None, ...]
    higher_oscillator_on_base: tuple[float | None, ...]


def _float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    converted: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} contains a non-finite value")
        converted.append(number)
    return tuple(converted)


def _validated_ohlc(
    open_values: Sequence[float],
    high_values: Sequence[float],
    low_values: Sequence[float],
    close_values: Sequence[float],
    volume_values: Sequence[float] | None,
) -> tuple[tuple[float, ...], ...]:
    open_rows = _float_tuple(open_values, "open")
    high_rows = _float_tuple(high_values, "high")
    low_rows = _float_tuple(low_values, "low")
    close_rows = _float_tuple(close_values, "close")
    lengths = {len(open_rows), len(high_rows), len(low_rows), len(close_rows)}
    if volume_values is not None:
        volume_rows = _float_tuple(volume_values, "volume")
        lengths.add(len(volume_rows))
    else:
        volume_rows = tuple(0.0 for _ in close_rows)
    if len(lengths) != 1:
        raise ValueError("OHLCV arrays must have equal lengths")
    for index, (open_value, high_value, low_value, close_value) in enumerate(
        zip(open_rows, high_rows, low_rows, close_rows)
    ):
        if low_value > high_value:
            raise ValueError(f"bar {index} has low above high")
        if not low_value <= min(open_value, close_value) <= high_value:
            raise ValueError(f"bar {index} open/close falls outside its high-low range")
        if not low_value <= max(open_value, close_value) <= high_value:
            raise ValueError(f"bar {index} open/close falls outside its high-low range")
    return open_rows, high_rows, low_rows, close_rows, volume_rows


def _price_series(
    open_values: tuple[float, ...],
    high_values: tuple[float, ...],
    low_values: tuple[float, ...],
    close_values: tuple[float, ...],
    source: str,
) -> tuple[float, ...]:
    if source == "open":
        return open_values
    if source == "high":
        return high_values
    if source == "low":
        return low_values
    if source == "close":
        return close_values
    if source == "hl2":
        return tuple((high + low) / 2.0 for high, low in zip(high_values, low_values))
    if source == "hlc3":
        return tuple(
            (high + low + close) / 3.0
            for high, low, close in zip(high_values, low_values, close_values)
        )
    return tuple(
        (open_value + high + low + close) / 4.0
        for open_value, high, low, close in zip(
            open_values, high_values, low_values, close_values
        )
    )


def compute_fractal_energy(
    high_values: Sequence[float],
    low_values: Sequence[float],
    close_values: Sequence[float],
    *,
    lookback: int = 13,
) -> tuple[float | None, ...]:
    """Return normalized Mobius-style Fractal Energy in the range ``0..1``."""

    high_rows = _float_tuple(high_values, "high")
    low_rows = _float_tuple(low_values, "low")
    close_rows = _float_tuple(close_values, "close")
    if len({len(high_rows), len(low_rows), len(close_rows)}) != 1:
        raise ValueError("high, low, and close arrays must have equal lengths")
    window = int(lookback)
    if window < 2:
        raise ValueError("lookback must be at least 2")
    for index, (high, low) in enumerate(zip(high_rows, low_rows)):
        if low > high:
            raise ValueError(f"bar {index} has low above high")

    true_ranges: list[float] = []
    for index, (high, low) in enumerate(zip(high_rows, low_rows)):
        if index == 0:
            true_ranges.append(high - low)
        else:
            previous_close = close_rows[index - 1]
            true_ranges.append(
                max(high - low, abs(high - previous_close), abs(low - previous_close))
            )

    denominator = math.log(float(window))
    result: list[float | None] = []
    for index in range(len(close_rows)):
        start = index - window + 1
        if start < 0:
            result.append(None)
            continue
        price_span = max(high_rows[start : index + 1]) - min(low_rows[start : index + 1])
        path_length = sum(true_ranges[start : index + 1])
        if price_span <= 0.0 or path_length <= 0.0:
            result.append(1.0)
            continue
        ratio = max(1.0, path_length / price_span)
        energy = math.log(ratio) / denominator
        result.append(min(1.0, max(0.0, energy)))
    return tuple(result)


def classify_laguerre_states(
    oscillator: Sequence[float],
    *,
    upper_threshold: float = 0.8,
    lower_threshold: float = 0.2,
) -> tuple[LaguerreState | None, ...]:
    """Classify completed oscillator values without looking at a later bar.

    Values before the first threshold encounter are ``None``: neither decay
    state is truthful until a trend state has first been established.
    """

    upper = float(upper_threshold)
    lower = float(lower_threshold)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= lower < upper <= 1")
    state: LaguerreState | None = None
    output: list[LaguerreState | None] = []
    for raw_value in oscillator:
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("oscillator values must be finite and bounded 0..1")
        if state is None:
            if value >= upper:
                state = LaguerreState.TREND_UP
            elif value <= lower:
                state = LaguerreState.TREND_DOWN
        elif state == LaguerreState.TREND_UP:
            if value <= lower:
                state = LaguerreState.TREND_DOWN
            elif value < upper:
                state = LaguerreState.DECAY_UP
        elif state == LaguerreState.DECAY_UP:
            if value <= lower:
                state = LaguerreState.TREND_DOWN
            elif value >= upper:
                state = LaguerreState.TREND_UP
        elif state == LaguerreState.TREND_DOWN:
            if value >= upper:
                state = LaguerreState.TREND_UP
            elif value > lower:
                state = LaguerreState.DECAY_DOWN
        elif state == LaguerreState.DECAY_DOWN:
            if value >= upper:
                state = LaguerreState.TREND_UP
            elif value <= lower:
                state = LaguerreState.TREND_DOWN
        output.append(state)
    return tuple(output)


def compute_laguerre_rsi(
    open_values: Sequence[float],
    high_values: Sequence[float],
    low_values: Sequence[float],
    close_values: Sequence[float],
    volume_values: Sequence[float] | None = None,
    *,
    config: LaguerreRsiConfig | None = None,
) -> LaguerreRsiResult:
    """Compute Laguerre RSI for arrays containing completed bars."""

    active = config or LaguerreRsiConfig()
    open_rows, high_rows, low_rows, close_rows, _volume_rows = _validated_ohlc(
        open_values,
        high_values,
        low_values,
        close_values,
        volume_values,
    )
    prices = _price_series(
        open_rows,
        high_rows,
        low_rows,
        close_rows,
        active.price_source,
    )
    energy = compute_fractal_energy(
        high_rows,
        low_rows,
        close_rows,
        lookback=active.fractal_energy_lookback,
    )
    if not prices:
        return LaguerreRsiResult(
            feature_version=FEATURE_VERSION,
            oscillator=(),
            fractal_energy=(),
            gamma=(),
            states=(),
        )

    l0 = l1 = l2 = l3 = prices[0]
    previous_lrsi = 0.5
    oscillator: list[float] = []
    gamma_values: list[float] = []
    for index, price in enumerate(prices):
        if active.fixed_gamma is not None:
            gamma = float(active.fixed_gamma)
        else:
            # FE needs a complete lookback. A neutral 0.5 warmup avoids
            # importing future window information into the first values.
            gamma = 0.5 if energy[index] is None else float(energy[index])
            gamma = min(0.999999, max(0.0, gamma))
        previous_l0, previous_l1, previous_l2, previous_l3 = l0, l1, l2, l3
        l0 = (1.0 - gamma) * price + gamma * previous_l0
        l1 = -gamma * l0 + previous_l0 + gamma * previous_l1
        l2 = -gamma * l1 + previous_l1 + gamma * previous_l2
        l3 = -gamma * l2 + previous_l2 + gamma * previous_l3

        cumulative_up = 0.0
        cumulative_down = 0.0
        for faster, slower in ((l0, l1), (l1, l2), (l2, l3)):
            if faster >= slower:
                cumulative_up += faster - slower
            else:
                cumulative_down += slower - faster
        denominator = cumulative_up + cumulative_down
        lrsi = cumulative_up / denominator if denominator > 0.0 else previous_lrsi
        lrsi = min(1.0, max(0.0, lrsi))
        oscillator.append(lrsi)
        gamma_values.append(gamma)
        previous_lrsi = lrsi

    states = classify_laguerre_states(
        oscillator,
        upper_threshold=active.upper_threshold,
        lower_threshold=active.lower_threshold,
    )
    return LaguerreRsiResult(
        feature_version=FEATURE_VERSION,
        oscillator=tuple(oscillator),
        fractal_energy=energy,
        gamma=tuple(gamma_values),
        states=states,
    )


def _aggregate_completed_groups(
    open_values: tuple[float, ...],
    high_values: tuple[float, ...],
    low_values: tuple[float, ...],
    close_values: tuple[float, ...],
    volume_values: tuple[float, ...],
    factor: int,
) -> tuple[tuple[float, ...], ...]:
    complete_count = len(close_values) // factor
    aggregate_open: list[float] = []
    aggregate_high: list[float] = []
    aggregate_low: list[float] = []
    aggregate_close: list[float] = []
    aggregate_volume: list[float] = []
    completion_indices: list[int] = []
    for group_index in range(complete_count):
        start = group_index * factor
        end = start + factor
        aggregate_open.append(open_values[start])
        aggregate_high.append(max(high_values[start:end]))
        aggregate_low.append(min(low_values[start:end]))
        aggregate_close.append(close_values[end - 1])
        aggregate_volume.append(sum(volume_values[start:end]))
        completion_indices.append(end - 1)
    return (
        tuple(aggregate_open),
        tuple(aggregate_high),
        tuple(aggregate_low),
        tuple(aggregate_close),
        tuple(aggregate_volume),
        tuple(completion_indices),
    )


def compute_multitimeframe_laguerre_rsi(
    open_values: Sequence[float],
    high_values: Sequence[float],
    low_values: Sequence[float],
    close_values: Sequence[float],
    volume_values: Sequence[float] | None = None,
    *,
    config: LaguerreRsiConfig | None = None,
    higher_factor: int = 3,
) -> MultiTimeframeLaguerreResult:
    """Compute base and N× signals without exposing a partial higher bar.

    The first input bar must begin on a higher-timeframe boundary. Completed
    groups are emitted only after all ``higher_factor`` base bars exist; any
    trailing partial group remains ``None`` in the base-aligned projection.
    """

    factor = int(higher_factor)
    if factor < 2:
        raise ValueError("higher_factor must be at least 2")
    validated = _validated_ohlc(
        open_values,
        high_values,
        low_values,
        close_values,
        volume_values,
    )
    open_rows, high_rows, low_rows, close_rows, volume_rows = validated
    active = config or LaguerreRsiConfig()
    base = compute_laguerre_rsi(
        open_rows,
        high_rows,
        low_rows,
        close_rows,
        volume_rows,
        config=active,
    )
    (
        higher_open,
        higher_high,
        higher_low,
        higher_close,
        higher_volume,
        completion_indices,
    ) = _aggregate_completed_groups(
        open_rows,
        high_rows,
        low_rows,
        close_rows,
        volume_rows,
        factor,
    )
    higher = compute_laguerre_rsi(
        higher_open,
        higher_high,
        higher_low,
        higher_close,
        higher_volume,
        config=active,
    )

    projected_states: list[LaguerreState | None] = [None] * len(close_rows)
    projected_oscillator: list[float | None] = [None] * len(close_rows)
    last_state: LaguerreState | None = None
    last_oscillator: float | None = None
    completion_map = {
        base_index: higher_index
        for higher_index, base_index in enumerate(completion_indices)
    }
    for base_index in range(len(close_rows)):
        higher_index = completion_map.get(base_index)
        if higher_index is not None:
            last_state = higher.states[higher_index]
            last_oscillator = higher.oscillator[higher_index]
        projected_states[base_index] = last_state
        projected_oscillator[base_index] = last_oscillator

    return MultiTimeframeLaguerreResult(
        feature_version=FEATURE_VERSION,
        base=base,
        higher=higher,
        higher_factor=factor,
        higher_completed_base_indices=completion_indices,
        higher_state_on_base=tuple(projected_states),
        higher_oscillator_on_base=tuple(projected_oscillator),
    )
