"""One place for how the desk measures relative strength vs SPY.

Every RS/RW calculation reads its engine, cutoff and windows from here, so the
whole desk moves together. ``RRS_ENGINE = "desk"`` (or the environment
variable ``TRADINGBOTV3_RRS_ENGINE=desk``) puts every site back on the old
formulas without a code change.
"""

from __future__ import annotations

import os

from indicators.rolling_rrs import RollingRrsConfig

ENGINE_ROLLING = "rolling_hourly"  # H.S. rolling RRS (indicators/rolling_rrs.py)
ENGINE_DESK = "desk"  # the pre-2026-09-28 formulas
RRS_ENGINE = ENGINE_ROLLING

# Cutoffs: |RRS| at or above this is RS/RW. The trader set 1.0 on 2026-09-27.
ROLLING_RRS_CUTOFF = 1.0
DESK_RRS_CUTOFF = 2.0

# Any other RRS constant written on the old desk scale (bonuses, margins,
# hard gates) is multiplied by this, so it keeps its place relative to the cutoff.
DESK_TO_ROLLING = ROLLING_RRS_CUTOFF / DESK_RRS_CUTOFF

# Intraday: one-hour move over the hourly ATR, mean of the last 12 reads.
# At least 3 reads, so the first read comes 70 minutes after the open (H.S.:
# "at least 1 hour of price action").
INTRADAY = RollingRrsConfig(length=12, roll=12, min_reads=3)

# Scan candle size (minutes) -> reads between candle closes on the M5 series.
SAMPLE_EVERY = {5: 1, 15: 3, 30: 6, 60: 12}

# Daily: move over 5 (or 20) days vs the daily ATR, mean of the last 5 reads.
DAILY_5D = RollingRrsConfig(atr_mode="daily", length=5, roll=5)
DAILY_20D = RollingRrsConfig(atr_mode="daily", length=20, roll=5)


def engine() -> str:
    """The active engine; the environment variable wins over the constant."""
    value = (os.environ.get("TRADINGBOTV3_RRS_ENGINE") or RRS_ENGINE).strip().lower()
    return ENGINE_DESK if value == ENGINE_DESK else ENGINE_ROLLING


def use_rolling() -> bool:
    return engine() == ENGINE_ROLLING


def cutoff() -> float:
    """The RS/RW cutoff for the active engine."""
    return ROLLING_RRS_CUTOFF if use_rolling() else DESK_RRS_CUTOFF


def intraday_config(candle_minutes: int = 5) -> RollingRrsConfig:
    """INTRADAY rolled every Nth read for a scan candle size."""
    step = SAMPLE_EVERY.get(int(candle_minutes), max(1, int(candle_minutes) // 5))
    if step == 1:
        return INTRADAY
    return RollingRrsConfig(length=12, roll=12, min_reads=3, sample_every=step)
