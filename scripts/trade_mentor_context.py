"""Small, point-in-time market facts for one Trade Mentor read.

This module is deliberately pure.  It accepts bars already read by the context
service, excludes incomplete or bad observations, and returns only the small
facts a later journal reader may need.  It never fetches, scores, or advises.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from typing import Any, Mapping, Sequence

from market_calendar import MARKET_TZ, is_session, previous_session
from market_early_close import session_close

SYMBOLS = (
    "VXX", "RSP", "USO", "TLT", "IWM", "QQQ", "SPY", "XLB", "XLC",
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
)

_M5_KEYS = (
    "symbol", "m5_status", "m5_reason", "m5_as_of", "m5_change_30m_pct",
    "m5_direction", "m5_vs_session_vwap", "d1_status", "d1_reason", "d1_as_of",
    "d1_change_5d_pct", "d1_vs_sma20",
)
_OPEN = time(9, 30)


def _blank(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "m5_status": "unavailable", "m5_reason": "no usable M5 bars",
        "m5_as_of": None, "m5_change_30m_pct": None, "m5_direction": None,
        "m5_vs_session_vwap": None, "d1_status": "unavailable",
        "d1_reason": "no usable D1 bars", "d1_as_of": None,
        "d1_change_5d_pct": None, "d1_vs_sma20": None,
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _field(bar: Any, name: str) -> Any:
    return bar.get(name) if isinstance(bar, Mapping) else getattr(bar, name, None)


def _stamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _market_now(now: datetime) -> datetime:
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(MARKET_TZ)


def _last_m5_start(now: datetime) -> datetime:
    """The latest completed regular-hours M5 bar start at ``now``."""
    moment = _market_now(now)
    day = moment.date()
    try:
        if not is_session(day) or moment < datetime.combine(day, _OPEN, MARKET_TZ):
            day = previous_session(day)
            return session_close(day) - timedelta(minutes=5)
        close = session_close(day)
    except Exception:
        # A calendar refusal is uncertainty.  The caller will report stale/no
        # data rather than extrapolating a regular session.
        return datetime.min.replace(tzinfo=MARKET_TZ)
    if moment >= close:
        return close - timedelta(minutes=5)
    elapsed = int((moment - datetime.combine(day, _OPEN, MARKET_TZ)).total_seconds() // 60)
    return datetime.combine(day, _OPEN, MARKET_TZ) + timedelta(minutes=(elapsed // 5 - 1) * 5)


def _last_completed_exchange_session(now: datetime) -> date:
    """Latest closed exchange day, including a real early close."""
    moment = _market_now(now)
    day = moment.date()
    if is_session(day) and session_close(day) <= moment:
        return day
    return previous_session(day)


def _valid_m5(bars: Sequence[Any], now: datetime) -> tuple[list[dict[str, Any]], str]:
    """Return regular completed bars or one concise reason they cannot speak."""
    moment = _market_now(now)
    seen: set[datetime] = set()
    out: list[dict[str, Any]] = []
    for bar in bars or ():
        stamp = _stamp(_field(bar, "dt") or _field(bar, "timestamp") or _field(bar, "time"))
        if stamp is None or stamp.tzinfo is None:
            return [], "M5 timestamps must be timezone-aware"
        market_stamp = stamp.astimezone(MARKET_TZ)
        if market_stamp in seen:
            return [], "duplicate M5 bar timestamp"
        seen.add(market_stamp)
        values = {name: _number(_field(bar, name)) for name in ("open", "high", "low", "close", "volume")}
        if any(values[name] is None or values[name] <= 0 for name in ("open", "high", "low", "close")):
            continue
        if values["low"] > values["open"] or values["low"] > values["close"] or values["high"] < values["open"] or values["high"] < values["close"]:
            continue
        if values["volume"] is None or values["volume"] < 0:
            continue
        try:
            close = session_close(market_stamp.date())
            regular = is_session(market_stamp.date()) and datetime.combine(market_stamp.date(), _OPEN, MARKET_TZ) <= market_stamp and market_stamp + timedelta(minutes=5) <= close
        except Exception:
            regular = False
        if regular and market_stamp + timedelta(minutes=5) <= moment:
            out.append({"dt": stamp, "market_dt": market_stamp, **values})
    out.sort(key=lambda row: row["market_dt"])
    return out, ""


def _m5_reading(row: dict[str, Any], bars: Sequence[Any], now: datetime) -> None:
    valid, reason = _valid_m5(bars, now)
    if reason:
        row["m5_reason"] = reason
        return
    if not valid:
        row["m5_reason"] = "no completed regular-session M5 bars"
        return
    expected = _last_m5_start(now)
    last = valid[-1]
    if expected == datetime.min.replace(tzinfo=MARKET_TZ) or expected - last["market_dt"] > timedelta(minutes=15):
        row.update(m5_status="stale", m5_reason="M5 bars are stale", m5_as_of=last["dt"].isoformat())
        return
    by_stamp = {bar["market_dt"]: bar for bar in valid}
    stamps = [last["market_dt"] - timedelta(minutes=5 * offset) for offset in range(6, -1, -1)]
    if any(stamp not in by_stamp for stamp in stamps):
        row.update(m5_reason="M5 bars have a gap in the 30-minute window", m5_as_of=last["dt"].isoformat())
        return
    then = by_stamp[stamps[0]]["close"]
    if then is None or then == 0:
        row.update(m5_reason="M5 30-minute base close is unusable", m5_as_of=last["dt"].isoformat())
        return
    change = (last["close"] - then) / then * 100.0
    row.update(
        m5_status="measured", m5_reason="", m5_as_of=last["dt"].isoformat(),
        m5_change_30m_pct=round(change, 6),
        m5_direction="up" if change > 0 else "down" if change < 0 else "flat",
    )
    # A session VWAP means every completed regular M5 bar since this session's
    # open was present and carried valid volume.  A tail is never called VWAP.
    start = datetime.combine(last["market_dt"].date(), _OPEN, MARKET_TZ)
    full_stamps = []
    cursor = start
    while cursor <= last["dt"]:
        full_stamps.append(cursor)
        cursor += timedelta(minutes=5)
    if all(stamp in by_stamp for stamp in full_stamps):
        numerator = sum(((bar["high"] + bar["low"] + bar["close"]) / 3.0) * bar["volume"] for bar in (by_stamp[stamp] for stamp in full_stamps))
        denominator = sum(by_stamp[stamp]["volume"] for stamp in full_stamps)
        if denominator > 0:
            vwap = numerator / denominator
            row["m5_vs_session_vwap"] = "above" if last["close"] > vwap else "below" if last["close"] < vwap else "at"


def _valid_d1(bars: Sequence[Any], now: datetime) -> tuple[list[dict[str, Any]], str]:
    try:
        cutoff = _last_completed_exchange_session(now)
    except Exception:
        return [], "latest completed D1 session is unavailable"
    seen: set[date] = set()
    out: list[dict[str, Any]] = []
    for bar in bars or ():
        stamp = _stamp(_field(bar, "dt") or _field(bar, "date") or _field(bar, "timestamp"))
        if stamp is None:
            continue
        day = stamp.astimezone(MARKET_TZ).date() if stamp.tzinfo else stamp.date()
        if day in seen:
            return [], "duplicate D1 bar date"
        seen.add(day)
        values = {name: _number(_field(bar, name)) for name in ("open", "high", "low", "close")}
        if any(value is None or value <= 0 for value in values.values()):
            continue
        if values["low"] > values["open"] or values["low"] > values["close"] or values["high"] < values["open"] or values["high"] < values["close"]:
            continue
        try:
            regular = is_session(day)
        except Exception:
            regular = False
        if regular and day <= cutoff:
            out.append({"day": day, **values})
    out.sort(key=lambda row: row["day"])
    return out, ""


def _d1_reading(row: dict[str, Any], bars: Sequence[Any], now: datetime) -> None:
    valid, reason = _valid_d1(bars, now)
    if reason:
        row["d1_reason"] = reason
        return
    if not valid:
        row["d1_reason"] = "no completed regular-session D1 bars"
        return
    try:
        expected = _last_completed_exchange_session(now)
    except Exception:
        row["d1_reason"] = "latest completed D1 session is unavailable"
        return
    last = valid[-1]
    if last["day"] != expected:
        row.update(d1_status="stale", d1_reason="D1 bars are stale", d1_as_of=last["day"].isoformat())
        return
    if len(valid) < 20:
        row.update(d1_reason="not enough completed D1 bars for SMA20 and five-session change", d1_as_of=last["day"].isoformat())
        return
    expected_days = [expected]
    for _ in range(19):
        expected_days.append(previous_session(expected_days[-1]))
    if [item["day"] for item in valid[-20:]] != list(reversed(expected_days)):
        row.update(d1_reason="D1 bars have a gap in the completed-session history", d1_as_of=last["day"].isoformat())
        return
    base = valid[-6]["close"]
    if base == 0:
        row.update(d1_reason="D1 five-session base close is unusable", d1_as_of=last["day"].isoformat())
        return
    change = (last["close"] - base) / base * 100.0
    sma20 = sum(item["close"] for item in valid[-20:]) / 20.0
    row.update(
        d1_status="measured", d1_reason="", d1_as_of=last["day"].isoformat(),
        d1_change_5d_pct=round(change, 6),
        d1_vs_sma20="above" if last["close"] > sma20 else "below" if last["close"] < sma20 else "at",
    )


def build_context(*, now: datetime, m5_bars: Mapping[str, Sequence[Any]], d1_bars: Mapping[str, Sequence[Any]], sources: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the bounded, JSON-safe snapshot attached to one journal read."""
    moment = _market_now(now)
    readings: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        reading = _blank(symbol)
        _m5_reading(reading, (m5_bars or {}).get(symbol, ()), moment)
        _d1_reading(reading, (d1_bars or {}).get(symbol, ()), moment)
        readings.append(reading)
    return {
        "schema": "trade_mentor_context_v1",
        "captured_at": now.isoformat(),
        "availability": "available",
        "reason": "",
        "rules": {"m5": "completed regular-session 30-minute change", "d1": "completed-session five-day change and SMA20"},
        "sources": dict(sources or {"m5": "unknown", "d1": "unknown"}),
        "readings": readings,
    }


def unavailable_context(*, now: datetime, reason: str) -> dict[str, Any]:
    """An explicit all-symbol absence, used when a note wins a slow fetch."""
    return {
        "schema": "trade_mentor_context_v1", "captured_at": now.isoformat(),
        "availability": "unavailable", "reason": str(reason),
        "rules": {"m5": "completed regular-session 30-minute change", "d1": "completed-session five-day change and SMA20"},
        "sources": {"m5": "unavailable", "d1": "unavailable"},
        "readings": [_blank(symbol) for symbol in SYMBOLS],
    }


def compact_for_ai(context: Any) -> Any:
    """Replace repeated journal-reading keys with a compact scalar table.

    Stored journal rows keep their ordinary, independently readable shape.  This
    projection is only for the existing bounded AI evidence package.
    """
    if not isinstance(context, Mapping) or context.get("schema") != "trade_mentor_context_v1":
        return context
    readings = context.get("readings")
    if not isinstance(readings, list) or not all(isinstance(row, Mapping) for row in readings):
        return context
    # The evidence source keeps this under ``mentor.context_compact``.  The
    # durable journal's full ``mentor.context`` remains untouched.
    return {
        "captured_at": context.get("captured_at"),
        "columns": list(_M5_KEYS),
        "rows": [[row.get(key) for key in _M5_KEYS] for row in readings],
    }
