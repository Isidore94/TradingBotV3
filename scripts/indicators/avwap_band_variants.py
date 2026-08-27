"""The OneOption / Option Stalker Pro anchored-VWAP band, replicated (Phase 0.10 B-0).

Governing spec: ``docs/AVWAP_BAND_VARIANT_STUDY.md`` section 2b, where the
formula was pinned on 2026-08-26 by three hover readings the trader took off
OneOption's own OKTA chart (anchor 2026-05-29).

**The formula, verbatim**::

    centre_t  = anchored VWAP of HLC/3, volume-weighted, from the anchor bar to t
    sigma_t   = population standard deviation of the last 20 CLOSES ending at t
                (the window ignores the anchor and reaches back before it)
    band_k,t  = centre_t +/- k * sigma_t          k = 1, 2, 3

That is a textbook Bollinger width laid on an anchored HLC/3 centre. The two
halves know nothing about each other: the centre remembers the anchor, the
width does not.

Vendor: OneOption / Option Stalker Pro ("AVWAP E/Q Standard Deviation lines",
release notes Feb + Aug 2024; no formula published). Replicated 2026-08-26.

**The tempting wrong forms this is NOT**

* *The champion* (``master_avwap_lib.legacy.calc_anchored_vwap_bands``): OHLC/4
  typical price, sigma accumulated as each bar's deviation from the RUNNING
  AVWAP at that bar. It is zero on a one-bar anchor by construction; OneOption
  read 10.28 there. The champion is frozen (decision 0008, plan.md section 5)
  and this module never touches it - it does not import it and it is not
  imported by it.
* *The distribution sigma* (TradingView's built-in): deviation from the CURRENT
  AVWAP rather than the running one. Also zero on a one-bar anchor, so also
  dead.
* *Sample stdev of every O/H/L/C print since the anchor around the running
  AVWAP* (n-1 denominator). This one survived the anchor bar - it predicts
  128.51 against a reading of 128.47 - and was killed by the second hover: it
  predicts an upper band of **138.09** on 2026-06-02 where the trader read
  **144.60**. It is reproduced in ``tests/test_avwap_band_variants.py`` as the
  discriminator, so a future edit that quietly drifts toward it fails.
* Any range form ((H-L)/2 = 9.15, (H-L)/sqrt(12) = 5.28), any ATR multiple
  (1x = 5.4, 2x = 10.8) and any percentage: all miss the 10.28 anchor-bar
  reading, per the study's section 2b kill round.

**Shadow only.** Nothing here may reach a detector, score, rank, tier, alert,
zone arm, Focus list or the review queue. It is a challenger under the
plan.md section 7 ladder, computed beside the champion and graded against it.

Pure and offline in the ``scripts/indicators/`` shape: completed bars in,
immutable aligned tuples out, ``None`` for anything unmeasurable, no I/O, no
clock, no provider call. Fewer than ``lookback`` closes before a bar means the
sigma is ``None`` and the band is absent - never padded, never zero, never a
shorter window.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

FEATURE_VERSION = "avwap_bands_oneoption_bb20_v1"

#: The band multiples this module publishes, and the key suffix each uses.
BAND_MULTIPLES = (1, 2, 3)

#: Series keys in the mapping ``oneoption_avwap_band_series`` returns.
SERIES_KEYS = (
    "centre",
    "sigma",
    "upper_1",
    "upper_2",
    "upper_3",
    "lower_1",
    "lower_2",
    "lower_3",
)

#: The champion's band-dict keys, so a caller can hold the two side by side.
BAND_KEYS = ("UPPER_1", "LOWER_1", "UPPER_2", "LOWER_2", "UPPER_3", "LOWER_3")

_PRICE_FIELDS = ("high", "low", "close", "volume")


def _finite(value: Any) -> float | None:
    """``float(value)`` when it is a real, finite number; otherwise ``None``.

    A NaN volume must not be allowed to poison the running sums, and a NaN
    close must not be allowed to poison a standard deviation - both are
    missing data, which is uncertainty rather than zero.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _field(bar: Any, name: str) -> float | None:
    """Read one named price field from a bar, mapping-first then attribute.

    Mapping access covers dicts and pandas ``Series`` rows; attribute access
    covers dataclasses and namedtuples. Nothing here imports pandas: the module
    must stay importable in a headless, dependency-light context.
    """
    try:
        return _finite(bar[name])
    except (TypeError, KeyError, IndexError):
        pass
    return _finite(getattr(bar, name, None))


def _columns(bars: Any) -> list[dict[str, float | None]] | None:
    """Column-wise read of a DataFrame-shaped input, or ``None`` if not one.

    Duck-typed on ``.columns`` so a pandas frame is read once per column
    instead of once per row - ``df.iloc[i]`` builds a Series per bar, which is
    the expensive way to walk a few hundred sessions.
    """
    columns = getattr(bars, "columns", None)
    if columns is None:
        return None
    try:
        names = {str(name).lower(): name for name in columns}
    except TypeError:
        return None
    if not all(field in names for field in _PRICE_FIELDS):
        return None
    series: dict[str, list[float | None]] = {}
    for field in _PRICE_FIELDS:
        try:
            values = list(bars[names[field]])
        except Exception:  # pragma: no cover - a frame that cannot be read
            return None
        series[field] = [_finite(value) for value in values]
    length = min(len(values) for values in series.values())
    return [{field: series[field][index] for field in _PRICE_FIELDS} for index in range(length)]


def _rows(bars: Any) -> list[dict[str, float | None]]:
    """Normalize any accepted bar container into plain per-bar dicts."""
    columnar = _columns(bars)
    if columnar is not None:
        return columnar
    out: list[dict[str, float | None]] = []
    for bar in bars:
        out.append({field: _field(bar, field) for field in _PRICE_FIELDS})
    return out


def _population_stdev(values: Sequence[float], ddof: int) -> float | None:
    count = len(values)
    if count - ddof <= 0:
        return None
    mean = sum(values) / count
    total = sum((value - mean) ** 2 for value in values)
    return math.sqrt(total / (count - ddof))


def oneoption_avwap_band_series(
    bars: Any,
    anchor_index: int,
    *,
    lookback: int = 20,
    ddof: int = 0,
) -> dict[str, tuple[float | None, ...]]:
    """Per-bar centre, sigma and +/-1/2/3 bands, aligned 1:1 with ``bars``.

    ``bars`` may be a pandas DataFrame with ``high``/``low``/``close``/
    ``volume`` columns, or any sequence of mapping- or attribute-accessible
    bars with those fields.

    Bars before ``anchor_index`` are ``None`` in every series: the centre is
    anchored and does not exist before its anchor. The sigma window, by
    contrast, deliberately reaches back BEFORE the anchor - that is what the
    replication says OneOption does, and it is why the band is already wide on
    the anchor bar where the champion's is exactly zero.

    A zero, negative, missing or NaN volume bar is skipped in the centre
    exactly as the champion skips it, but its close still counts in the sigma:
    the sigma is not volume-weighted.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    if ddof < 0:
        raise ValueError("ddof must be zero or positive")

    rows = _rows(bars)
    length = len(rows)
    empty: dict[str, tuple[float | None, ...]] = {key: () for key in SERIES_KEYS}
    if length == 0:
        return empty

    anchor = int(anchor_index)
    if anchor < 0 or anchor >= length:
        raise IndexError(f"anchor_index {anchor_index} outside 0..{length - 1}")

    centres: list[float | None] = [None] * length
    sigmas: list[float | None] = [None] * length
    bands: dict[int, list[float | None]] = {}
    for multiple in BAND_MULTIPLES:
        bands[multiple] = [None] * length
        bands[-multiple] = [None] * length

    cumulative_volume = 0.0
    cumulative_price_volume = 0.0
    for index in range(anchor, length):
        row = rows[index]
        volume = row["volume"]
        high, low, close = row["high"], row["low"], row["close"]
        if volume is not None and volume > 0 and None not in (high, low, close):
            typical = (high + low + close) / 3.0
            cumulative_volume += volume
            cumulative_price_volume += typical * volume

        centre = cumulative_price_volume / cumulative_volume if cumulative_volume > 0 else None
        centres[index] = centre

        window_start = index + 1 - lookback
        sigma: float | None = None
        if window_start >= 0:
            window = [rows[position]["close"] for position in range(window_start, index + 1)]
            # One missing close makes the whole window unmeasurable. Dropping it
            # instead would quietly compute a shorter-window sigma and print a
            # confident number for a window the data does not support.
            if None not in window:
                sigma = _population_stdev([value for value in window if value is not None], ddof)
        sigmas[index] = sigma

        if centre is None or sigma is None:
            continue
        for multiple in BAND_MULTIPLES:
            bands[multiple][index] = centre + multiple * sigma
            bands[-multiple][index] = centre - multiple * sigma

    result: dict[str, tuple[float | None, ...]] = {
        "centre": tuple(centres),
        "sigma": tuple(sigmas),
    }
    for multiple in BAND_MULTIPLES:
        result[f"upper_{multiple}"] = tuple(bands[multiple])
        result[f"lower_{multiple}"] = tuple(bands[-multiple])
    return result


def oneoption_avwap_bands(
    bars: Any,
    anchor_index: int,
    **kwargs: Any,
) -> tuple[float | None, float | None, dict[str, float]]:
    """Final-bar ``(vwap, stdev, bands)``, the champion's return shape.

    Deliberately the same three-tuple as
    ``master_avwap_lib.legacy.calc_anchored_vwap_bands`` so a caller can hold
    the two answers side by side without adapting either.

    One difference, and it is the ``indicators/`` contract rather than an
    oversight: an unmeasurable value is ``None`` here where the champion
    returns ``float('nan')``. A ``None`` sigma yields ``(centre, None, {})`` -
    the bands are absent, never padded and never zero.
    """
    series = oneoption_avwap_band_series(bars, anchor_index, **kwargs)
    if not series["centre"]:
        return None, None, {}
    centre = series["centre"][-1]
    sigma = series["sigma"][-1]
    if centre is None or sigma is None:
        return centre, sigma, {}
    bands = {
        f"{'UPPER' if sign > 0 else 'LOWER'}_{multiple}": centre + sign * multiple * sigma
        for multiple in BAND_MULTIPLES
        for sign in (1, -1)
    }
    return centre, sigma, bands
