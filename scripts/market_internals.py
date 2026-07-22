"""Market internals: volatility, credit, duration, breadth and concentration.

The bot's regime read (``get_market_environment``) is derived from a single
SPY 5-minute series, so a ``bearish_strong`` on a calm tape and one on a
volatility spike are indistinguishable to it. This module adds the missing
axes as *recorded context*:

  vol           VXX   - volatility bid (risk-off when rising)
  credit        HYG   - high-yield credit (risk-on when rising)
  duration      TLT   - long treasuries (bid in flight-to-quality)
  breadth       RSP   - equal-weight S&P vs SPY: is the average stock
                        participating, or is the index being carried?
  concentration MAGS  - mega-cap leaders vs SPY: how narrow is the tape?

``breadth`` and ``concentration`` are *ratios* against SPY, which is what
makes them readable: RSP-minus-SPY is positive exactly when the median stock
is outperforming the cap-weighted index. That is the number that tells you
whether a long watchlist has the tape behind it.

Decision-support only, and deliberately inert: this module computes and
labels, it never gates an alert or moves a score. Promotion to a live input
requires measured evidence per plan.md sec 6-7 - the readings are stamped on
every alert first so the tracker can condition on them.

Pure: no IB, no network, no I/O, no Qt. The caller supplies bar series.
"""

from __future__ import annotations

import math
from typing import Any

INTERNALS_SCHEMA_VERSION = 1

# Instruments and what a RISE in each one means for risk appetite. Tradeable
# ETFs only - IB stock contracts cannot request index symbols like ^VIX, and
# VXX tracks the same volatility bid without a contract-type special case.
INTERNALS_INSTRUMENTS = {
    "VXX": {"axis": "vol", "rise_means": "risk_off", "label": "volatility"},
    "HYG": {"axis": "credit", "rise_means": "risk_on", "label": "credit"},
    "TLT": {"axis": "duration", "rise_means": "risk_off", "label": "duration"},
}
# Ratio instruments: measured as their session move MINUS SPY's session move.
INTERNALS_RATIOS = {
    "RSP": {"axis": "breadth", "label": "equal-weight breadth"},
    "MAGS": {"axis": "concentration", "label": "mega-cap concentration"},
}
INTERNALS_BENCHMARK = "SPY"

INTERNALS_SYMBOLS = tuple(list(INTERNALS_INSTRUMENTS) + list(INTERNALS_RATIOS) + [INTERNALS_BENCHMARK])

# A move smaller than this is noise, not a signal, on any of these series.
INTERNALS_FLAT_PCT = 0.15
# Ratio spreads are smaller than outright moves; breadth needs its own floor.
INTERNALS_RATIO_FLAT_PCT = 0.10


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def session_change_pct(bars: Any) -> float | None:
    """Percent change from the prior session's close to the latest bar close.

    ``bars`` is a chronological series of objects (or mappings) carrying
    ``dt`` and ``close``. The prior close is the last bar of the most recent
    earlier calendar date, which makes the reading a true session move rather
    than a rolling window. Returns ``None`` when a full prior session is not
    available - never a fabricated zero.
    """
    cleaned: list[tuple[Any, float]] = []
    for bar in bars or []:
        if isinstance(bar, dict):
            dt, close = bar.get("dt"), _finite(bar.get("close"))
        else:
            dt, close = getattr(bar, "dt", None), _finite(getattr(bar, "close", None))
        if dt is None or close is None or close <= 0:
            continue
        cleaned.append((dt, close))
    if len(cleaned) < 2:
        return None

    latest_date = getattr(cleaned[-1][0], "date", lambda: None)()
    if latest_date is None:
        return None
    prior_close = None
    for dt, close in reversed(cleaned[:-1]):
        bar_date = getattr(dt, "date", lambda: None)()
        if bar_date is not None and bar_date != latest_date:
            prior_close = close
            break
    if prior_close is None or prior_close <= 0:
        return None
    return (cleaned[-1][1] - prior_close) / prior_close * 100.0


def _state_for(change_pct: float | None, rise_means: str, flat_pct: float) -> str:
    if change_pct is None:
        return "unknown"
    if abs(change_pct) < flat_pct:
        return "flat"
    rising = change_pct > 0
    if rise_means == "risk_on":
        return "risk_on" if rising else "risk_off"
    return "risk_off" if rising else "risk_on"


def build_internals_snapshot(bars_by_symbol: Any, *, as_of: str = "") -> dict:
    """Readings for every internals instrument plus a plain-language summary.

    ``bars_by_symbol`` maps symbol -> bar series. Missing symbols degrade to
    ``unknown`` rather than removing the whole snapshot: a session with no
    MAGS data still gets a valid volatility and credit read.
    """
    bars_by_symbol = bars_by_symbol if isinstance(bars_by_symbol, dict) else {}
    readings: dict[str, dict] = {}

    for symbol, spec in INTERNALS_INSTRUMENTS.items():
        change = session_change_pct(bars_by_symbol.get(symbol))
        readings[spec["axis"]] = {
            "symbol": symbol,
            "label": spec["label"],
            "change_pct": round(change, 3) if change is not None else None,
            "state": _state_for(change, spec["rise_means"], INTERNALS_FLAT_PCT),
        }

    benchmark_change = session_change_pct(bars_by_symbol.get(INTERNALS_BENCHMARK))
    for symbol, spec in INTERNALS_RATIOS.items():
        change = session_change_pct(bars_by_symbol.get(symbol))
        spread = (
            round(change - benchmark_change, 3)
            if change is not None and benchmark_change is not None
            else None
        )
        # A ratio is risk_on when the breadth/concentration proxy leads SPY.
        readings[spec["axis"]] = {
            "symbol": symbol,
            "label": spec["label"],
            "change_pct": round(change, 3) if change is not None else None,
            "spread_vs_spy": spread,
            "state": _state_for(spread, "risk_on", INTERNALS_RATIO_FLAT_PCT),
        }

    risk_on = sum(1 for r in readings.values() if r["state"] == "risk_on")
    risk_off = sum(1 for r in readings.values() if r["state"] == "risk_off")
    known = risk_on + risk_off + sum(1 for r in readings.values() if r["state"] == "flat")
    if known == 0:
        tape = "unknown"
    elif risk_on and not risk_off:
        tape = "risk_on"
    elif risk_off and not risk_on:
        tape = "risk_off"
    elif risk_on == risk_off:
        tape = "mixed"
    else:
        tape = "risk_on_tilt" if risk_on > risk_off else "risk_off_tilt"

    return {
        "schema_version": INTERNALS_SCHEMA_VERSION,
        "as_of": str(as_of or ""),
        "benchmark_change_pct": round(benchmark_change, 3) if benchmark_change is not None else None,
        "tape": tape,
        "risk_on_count": risk_on,
        "risk_off_count": risk_off,
        "readings": readings,
        "advisory_only": True,
    }


def internals_context_fields(snapshot: Any) -> dict:
    """Flat per-alert columns for the learning rows.

    Kept deliberately small - one label plus the two numbers that carry the
    most information the SPY-only regime cannot see (is volatility bid, and
    is the average stock keeping up).
    """
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    readings = snapshot.get("readings") or {}

    def _num(axis, key="change_pct"):
        value = (readings.get(axis) or {}).get(key)
        return value if value is not None else ""

    return {
        "internals_tape": str(snapshot.get("tape") or ""),
        "internals_vol_pct": _num("vol"),
        "internals_breadth_spread": _num("breadth", "spread_vs_spy"),
    }


def format_internals_line(snapshot: Any) -> str:
    """One-line tape summary for reports and the status feed."""
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    readings = snapshot.get("readings") or {}
    if not readings or snapshot.get("tape") in (None, "", "unknown"):
        return "Internals: unavailable"
    parts = []
    for axis in ("vol", "credit", "duration", "breadth", "concentration"):
        reading = readings.get(axis)
        if not reading or reading.get("state") == "unknown":
            continue
        value = reading.get("spread_vs_spy") if "spread_vs_spy" in reading else reading.get("change_pct")
        if value is None:
            continue
        suffix = " vs SPY" if "spread_vs_spy" in reading else ""
        parts.append(f"{reading['symbol']} {value:+.2f}%{suffix}")
    tape = str(snapshot.get("tape") or "unknown").replace("_", " ")
    detail = " | ".join(parts)
    return f"Internals: {tape}" + (f" [{detail}]" if detail else "")
