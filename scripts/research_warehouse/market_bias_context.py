"""Versioned five-timeframe Auto Market Bias context for setup research.

The live Auto Market Bias formula remains the champion. This module calls the
same complete pure decision (VWAP regime plus the early-session day-percent
fallback) over completed SPY bars at M5, M30, H1, H4 and D1. It records context
only; it cannot reach a detector, score or alert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any

try:  # package import
    from . import aggregate, exchange_calendar as xcal
    from .manifest import utc_now
    from .outcomes import _entry_bar_after_d1_close
    from .schemas import SCHEMA_VERSION
except ImportError:  # pragma: no cover
    import aggregate  # type: ignore
    import exchange_calendar as xcal  # type: ignore
    from manifest import utc_now  # type: ignore
    from outcomes import _entry_bar_after_d1_close  # type: ignore
    from schemas import SCHEMA_VERSION  # type: ignore

BIAS_DEFINITION_ID = "auto_market_bias_multiframe_v1"
TIMEFRAMES = ("M5", "M30", "H1", "H4", "D1")
ROLLING_BARS = 20
UNKNOWN = "unknown"


@dataclass
class ContextReport:
    dataset: str = "setup_market_context"
    status: str = "OK"
    rows: int = 0
    occurrences: int = 0
    unknown: dict[str, int] = field(default_factory=dict)


def _number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def _bar_object(row: dict) -> SimpleNamespace:
    return SimpleNamespace(
        open=float(row.get("open") or row.get("close") or 0),
        high=float(row.get("high") or row.get("close") or 0),
        low=float(row.get("low") or row.get("close") or 0),
        close=float(row.get("close") or 0),
        volume=float(row.get("volume") or 0),
    )


def _worst_capture_mode(rows) -> str:
    rank = {"LIVE": 0, "DELAYED": 1, "BACKFILL": 2, "RECONSTRUCTED": 3}
    modes = [str(row.get("capture_mode") or row.get("input_capture_mode_worst") or "") for row in rows]
    return max(modes, key=lambda mode: rank.get(mode, 4), default="")


def _champion_read(rows: list[dict], reference_close: float | None) -> dict[str, Any]:
    if reference_close is None or not rows:
        return {"env_key": UNKNOWN, "source": "insufficient_completed_bars"}
    try:
        from bounce_bot_lib.legacy import _auto_market_regime_stats

        reading = _auto_market_regime_stats([_bar_object(row) for row in rows], reference_close)
    except Exception:
        reading = None
    if reading is None:
        return {"env_key": UNKNOWN, "source": "insufficient_completed_bars"}
    stats = reading.get("vwap_stats") or {}
    return {
        "env_key": str(reading.get("env_key") or UNKNOWN),
        "source": f"champion_{reading.get('source') or 'unknown'}_regime_window",
        "last_close": _number(reading.get("last_close")),
        "reference_close": reference_close,
        "vwap": _number(stats.get("vwap")),
        "stdev": _number(stats.get("stdev")),
        "above_band_frac": _number(stats.get("above_band_frac")),
        "below_band_frac": _number(stats.get("below_band_frac")),
    }


def _completed_d1(spy_d1: list[dict], entry_at: datetime) -> list[dict]:
    entry_session = xcal.session_for(entry_at)
    if entry_session is None:
        return []
    return sorted(
        [row for row in spy_d1 if row.get("session_date") and row["session_date"] < entry_session.session_date],
        key=lambda row: row["session_date"],
    )


def _derived(spy_m5: list[dict], timeframe: str, entry_at: datetime) -> list[dict]:
    by_day: dict[date, list[dict]] = {}
    for row in spy_m5:
        start = row.get("interval_start")
        if not isinstance(start, datetime) or start >= entry_at:
            continue
        session = xcal.session_for(start)
        if session is None:
            continue
        by_day.setdefault(session.session_date, []).append(row)
    result: list[dict] = []
    for day in sorted(by_day):
        session = xcal.trading_session(day)
        if session is None:
            continue
        result.extend(
            aggregate.derive_session_bars(
                by_day[day], session, timeframe, as_of=entry_at, computed_at=entry_at
            )
        )
    return sorted(result, key=lambda row: row["interval_end"])


def context_at(entry_at: datetime, *, spy_m5: list[dict], spy_d1: list[dict]) -> dict[str, dict]:
    """Five independent reads, all using bars completed by ``entry_at``."""
    d1 = _completed_d1(spy_d1, entry_at)
    entry_session = xcal.session_for(entry_at)
    current_m5 = []
    if entry_session is not None:
        current_m5 = sorted(
            [
                row for row in spy_m5
                if isinstance(row.get("interval_end"), datetime)
                and entry_session.rth_open_at < row["interval_end"] <= entry_at
                and row.get("is_complete", True)
            ],
            key=lambda row: row["interval_end"],
        )
    previous_d1_close = _number(d1[-1].get("close")) if d1 else None
    result = {"M5": {**_champion_read(current_m5, previous_d1_close), "bars": current_m5}}
    for timeframe in ("M30", "H1", "H4"):
        series = _derived(spy_m5, timeframe, entry_at)
        window = series[-ROLLING_BARS:]
        reference = _number(series[-ROLLING_BARS - 1].get("close")) if len(series) > ROLLING_BARS else None
        result[timeframe] = {**_champion_read(window, reference), "bars": window}
    d1_window = d1[-ROLLING_BARS:]
    d1_reference = _number(d1[-ROLLING_BARS - 1].get("close")) if len(d1) > ROLLING_BARS else None
    result["D1"] = {**_champion_read(d1_window, d1_reference), "bars": d1_window}
    return result


def build_context_rows(occurrences, *, spy_m5: list[dict], spy_d1: list[dict], now: datetime | None = None, run_id: str = "") -> list[dict]:
    stamp = now or utc_now()
    cache: dict[datetime, dict[str, dict]] = {}
    rows: list[dict] = []
    for occurrence in occurrences or []:
        entry_bar, _session = _entry_bar_after_d1_close(occurrence, spy_m5)
        if entry_bar is None:
            continue
        entry_at = entry_bar["interval_end"]
        readings = cache.setdefault(entry_at, context_at(entry_at, spy_m5=spy_m5, spy_d1=spy_d1))
        for timeframe in TIMEFRAMES:
            reading = readings[timeframe]
            bars = list(reading.get("bars") or [])
            rows.append(
                {
                    "occurrence_id": occurrence.get("occurrence_id"),
                    "symbol": occurrence.get("symbol"),
                    "entry_at": entry_at,
                    "timeframe": timeframe,
                    "bias_definition_id": BIAS_DEFINITION_ID,
                    "env_key": str(reading.get("env_key") or UNKNOWN),
                    "source": str(reading.get("source") or ""),
                    "last_close": reading.get("last_close"),
                    "reference_close": reading.get("reference_close"),
                    "vwap": reading.get("vwap"),
                    "stdev": reading.get("stdev"),
                    "above_band_frac": reading.get("above_band_frac"),
                    "below_band_frac": reading.get("below_band_frac"),
                    "bar_count": len(bars),
                    "computed_at": stamp,
                    "input_capture_mode_worst": _worst_capture_mode(bars),
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                }
            )
    return rows


def record_context(store, occurrences, *, spy_m5: list[dict], spy_d1: list[dict], now: datetime | None = None, run_id: str = "") -> ContextReport:
    occurrence_list = list(occurrences or [])
    report = ContextReport(occurrences=len(occurrence_list))
    if store is None:
        report.status = "DISABLED"
        return report
    candidates = build_context_rows(occurrence_list, spy_m5=spy_m5, spy_d1=spy_d1, now=now, run_id=run_id)
    occurrence_ids = [str(row.get("occurrence_id") or "") for row in occurrence_list]
    known = {
        (str(row.get("occurrence_id")), str(row.get("timeframe")), str(row.get("bias_definition_id")))
        for row in store.read_rows(
            "setup_market_context",
            columns=["occurrence_id", "timeframe", "bias_definition_id"],
            occurrence_ids=occurrence_ids,
        )
    }
    rows = [
        row for row in candidates
        if (str(row["occurrence_id"]), row["timeframe"], row["bias_definition_id"]) not in known
    ]
    for row in rows:
        if row["env_key"] == UNKNOWN:
            report.unknown[row["timeframe"]] = report.unknown.get(row["timeframe"], 0) + 1
    if not rows:
        report.status = "NOTHING_TO_RECORD"
        return report
    report.rows = store.publish("setup_market_context", rows, job_id="setup_market_context").rows_published
    return report


__all__ = [
    "BIAS_DEFINITION_ID",
    "TIMEFRAMES",
    "ContextReport",
    "build_context_rows",
    "context_at",
    "record_context",
]
