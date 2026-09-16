"""Pure, exit-independent forward movement measures for research.

This module deliberately does not import a detector, a score, a store, or an
exit simulator.  It measures the gross path that *completed* supplied bars
offered after a named feasible entry.  A hypothetical stop can therefore never
truncate MFE here.  The older :mod:`outcome_path` payload remains the frozen
exit-policy record; this is a separately versioned research view.

M5 bar times are completed-bar end times.  M5 windows are elapsed NYSE regular
session time from the feasible entry, never a count of arrived bars, and never
cross a session.  Daily windows use the Nth completed exchange session strictly
after the entry session close.  Missing timestamps are uncertainty, not zero.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from market_calendar import is_session


ENTRY_QUALITY_SCHEMA = "entry_quality_forward_v1"
M5_WINDOW_MINUTES: tuple[int, ...] = (5, 15, 30, 60, 120, 180)
SWING_WINDOW_SESSIONS: tuple[int, ...] = (1, 2, 3, 5, 10)
M5_BAR_MINUTES = 5
MARKET_TZ = ZoneInfo("America/New_York")
REGULAR_SESSION_CLOSE = time(16, 0)
EARLY_SESSION_CLOSE = time(13, 0)
ENTRY_SESSION_CONVENTION = "next_completed_exchange_session_close_v1"


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=((weekday - first.weekday()) % 7) + 7 * (occurrence - 1))


def _early_close(day: date) -> time | None:
    """Return the scheduled NYSE early close covered by this research clock.

    The normal calendar intentionally models full closures only.  This narrow
    helper is local to forward-window measurement because an early close
    changes a window endpoint.  It covers the recurring 13:00 ET sessions:
    the Friday after Thanksgiving, Christmas Eve when open, and the regular
    session immediately before the observed Independence-Day closure.  An
    unscheduled exchange closure is still unknowable and must be supplied as a
    data gap by the caller rather than guessed here.
    """
    if not is_session(day):
        return None
    thanksgiving = _nth_weekday(day.year, 11, 3, 4)
    if day == thanksgiving + timedelta(days=1):
        return EARLY_SESSION_CLOSE
    if (day.month, day.day) == (12, 24):
        return EARLY_SESSION_CLOSE

    independence = date(day.year, 7, 4)
    if independence.weekday() == 5:  # Saturday: Friday is a full holiday.
        observed = independence - timedelta(days=1)
    elif independence.weekday() == 6:
        observed = independence + timedelta(days=1)
    else:
        observed = independence
    if day == observed - timedelta(days=1) and is_session(day):
        return EARLY_SESSION_CLOSE
    return None


def session_endpoint(day: date) -> datetime:
    """Scheduled completed-session endpoint in America/New_York.

    It intentionally exposes an aware timestamp so daylight-saving changes are
    represented by the exchange timezone rather than a frozen UTC offset.
    """
    if not is_session(day):
        raise ValueError(f"{day.isoformat()} is not an exchange session")
    return datetime.combine(day, _early_close(day) or REGULAR_SESSION_CLOSE, tzinfo=MARKET_TZ)


def exchange_session_endpoints(start: date, horizons: Iterable[int]) -> dict[int, date]:
    """Map each positive horizon to the Nth exchange session strictly after start."""
    requested = tuple(sorted({int(value) for value in horizons if int(value) > 0}))
    if not requested:
        return {}
    maximum = requested[-1]
    endpoints: dict[int, date] = {}
    found = 0
    cursor = start + timedelta(days=1)
    while found < maximum:
        if is_session(cursor):
            found += 1
            if found in requested:
                endpoints[found] = cursor
        cursor += timedelta(days=1)
    return endpoints


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _moment(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(MARKET_TZ)


def _day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.astimezone(MARKET_TZ).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _entry_moments(entry: Mapping[str, Any]) -> tuple[datetime | None, datetime | None]:
    return _moment(entry.get("trigger_knowledge_time")), _moment(entry.get("feasible_entry_time"))


def _base_payload(entry: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    source = str(entry.get("source_knowledge_basis") or "unknown")
    anchor = str(entry.get("anchor_knowledge_basis") or "unknown")
    knowledge_label = "reconstructed" if "reconstructed" in (source, anchor) else (
        "observed" if source == anchor == "observed" else "mixed_or_unknown"
    )
    prospective = source == anchor == "observed"
    return {
        "schema": ENTRY_QUALITY_SCHEMA,
        "measure_kind": "gross_excursion_no_exit",
        "timeframe": kind,
        "opportunity_id": entry.get("opportunity_id"),
        "attempt_id": entry.get("attempt_id"),
        "symbol": entry.get("symbol"),
        "side": str(entry.get("side") or "LONG").upper(),
        "entry_rule": entry.get("entry_rule"),
        "entry_rule_version": entry.get("entry_rule_version"),
        "trigger_knowledge_time": entry.get("trigger_knowledge_time"),
        "feasible_entry_time": entry.get("feasible_entry_time"),
        "feasible_entry_price": entry.get("feasible_entry_price"),
        "source_knowledge_basis": source,
        "anchor_knowledge_basis": anchor,
        "knowledge_label": knowledge_label,
        "prospective_eligible": prospective,
    }


def _coverage(expected: int, observed: int) -> dict[str, int]:
    return {"expected_bars": expected, "observed_bars": observed, "missing_bars": max(0, expected - observed)}


def _m5_bars(rows: Iterable[Mapping[str, Any]]) -> tuple[list[tuple[datetime, float, float, float]], int]:
    """Coerce and de-duplicate completed M5 bars by their end timestamp."""
    by_time: dict[datetime, tuple[datetime, float, float, float]] = {}
    rejected = 0
    for row in rows or ():
        when = _moment(row.get("time") or row.get("datetime") or row.get("end_time"))
        high, low, close = _float(row.get("high")), _float(row.get("low")), _float(row.get("close"))
        if when is None or high is None or low is None or close is None or low > high:
            rejected += 1
            continue
        by_time.setdefault(when, (when, high, low, close))
    return sorted(by_time.values(), key=lambda value: value[0]), rejected


def _risk(entry: float, risk_price: Any, side: str) -> tuple[float | None, str | None]:
    price = _float(risk_price)
    if price is None:
        return None, "invalid_entry_risk_reference"
    distance = (entry - price) if side == "LONG" else (price - entry)
    if distance <= 0:
        return None, "invalid_entry_risk_reference"
    return distance, None


def _moves(
    bars: list[tuple[datetime, float, float, float]],
    *,
    entry_price: float,
    entry_time: datetime,
    side: str,
    atr: float | None,
    risk: float | None,
    favorable_threshold: float | None,
    adverse_threshold: float | None,
    daily_order_ambiguous: bool = False,
) -> dict[str, Any]:
    if side == "SHORT":
        favorable = lambda high, low: entry_price - low
        adverse = lambda high, low: entry_price - high
        closing = lambda close: entry_price - close
    else:
        favorable = lambda high, low: high - entry_price
        adverse = lambda high, low: low - entry_price
        closing = lambda close: close - entry_price

    best_move: float | None = None
    worst_move: float | None = None
    best_time: datetime | None = None
    first_favorable: datetime | None = None
    first_adverse: datetime | None = None
    for when, high, low, _close in bars:
        best, worst = favorable(high, low), adverse(high, low)
        if best_move is None or best > best_move:
            best_move, best_time = best, when
        if worst_move is None or worst < worst_move:
            worst_move = worst
        if favorable_threshold is not None and first_favorable is None and best >= favorable_threshold:
            first_favorable = when
        if adverse_threshold is not None and first_adverse is None and -worst >= adverse_threshold:
            first_adverse = when

    assert best_move is not None and worst_move is not None and best_time is not None
    if first_favorable is None or first_adverse is None:
        order = "favorable_only" if first_favorable else ("adverse_only" if first_adverse else "neither_threshold_touched")
    elif first_favorable == first_adverse:
        order = "ambiguous_same_daily_bar" if daily_order_ambiguous else "ambiguous_same_m5_bar"
    elif first_favorable < first_adverse:
        order = "favorable_first"
    else:
        order = "adverse_first"

    close_move = closing(bars[-1][3])
    result: dict[str, Any] = {
        "mfe_price_move": best_move,
        "mae_price_move": worst_move,
        "close_price_move": close_move,
        "mfe_pct": best_move / entry_price * 100.0,
        "mae_pct": worst_move / entry_price * 100.0,
        "close_pct": close_move / entry_price * 100.0,
        "time_to_mfe_minutes": (best_time - entry_time).total_seconds() / 60.0,
        "mfe_time": best_time.isoformat(),
        "first_favorable_time": first_favorable.isoformat() if first_favorable else None,
        "first_adverse_time": first_adverse.isoformat() if first_adverse else None,
        "first_touch_order": order,
    }
    if atr is not None and atr > 0:
        result.update({"mfe_atr": best_move / atr, "mae_atr": worst_move / atr, "close_atr": close_move / atr})
    else:
        result.update({"mfe_atr": None, "mae_atr": None, "close_atr": None})
    if risk is not None:
        result.update({"mfe_r": best_move / risk, "mae_r": worst_move / risk, "close_r": close_move / risk})
    else:
        result.update({"mfe_r": None, "mae_r": None, "close_r": None})
    return result


def _empty_window(*, state: str, reason: str, expected: int, observed: int = 0, endpoint: datetime | None = None) -> dict[str, Any]:
    return {
        "state": state,
        "reason": reason,
        "endpoint_time": endpoint.isoformat() if endpoint else None,
        "coverage": _coverage(expected, observed),
        "mfe_pct": None,
        "mae_pct": None,
        "close_pct": None,
        "mfe_atr": None,
        "mae_atr": None,
        "close_atr": None,
        "mfe_r": None,
        "mae_r": None,
        "close_r": None,
        "mfe_time": None,
        "time_to_mfe_minutes": None,
        "first_favorable_time": None,
        "first_adverse_time": None,
        "first_touch_order": "not_measured",
        "risk_reason": None,
        "confirmation_valid": False,
        "confirmation_reason": reason,
    }


def measure_m5_forward(entry: Mapping[str, Any], bars: Iterable[Mapping[str, Any]], *, as_of: datetime) -> dict[str, Any]:
    """Measure fixed M5 windows from supplied completed bars without any exit rule."""
    result = _base_payload(entry, kind="M5")
    result["window_minutes"] = M5_WINDOW_MINUTES
    knowledge_time, feasible_time = _entry_moments(entry)
    entry_price = _float(entry.get("feasible_entry_price"))
    side = result["side"]
    as_of_market = _moment(as_of)
    if side not in {"LONG", "SHORT"}:
        side = "LONG"
        result["side"] = side
    if knowledge_time is None or feasible_time is None or entry_price is None or entry_price <= 0 or as_of_market is None:
        result["windows"] = {
            f"{minutes}m": _empty_window(state="invalid_entry", reason="invalid_or_timezone_missing_entry", expected=0)
            for minutes in M5_WINDOW_MINUTES
        }
        result["windows"]["session_close"] = _empty_window(state="invalid_entry", reason="invalid_or_timezone_missing_entry", expected=0)
        return result

    measurement_start = max(knowledge_time, feasible_time)
    day = measurement_start.date()
    if not is_session(day):
        result["windows"] = {
            f"{minutes}m": _empty_window(state="invalid_entry", reason="entry_not_on_exchange_session", expected=0)
            for minutes in M5_WINDOW_MINUTES
        }
        result["windows"]["session_close"] = _empty_window(state="invalid_entry", reason="entry_not_on_exchange_session", expected=0)
        return result

    close_at = session_endpoint(day)
    all_bars, rejected = _m5_bars(bars)
    eligible = [bar for bar in all_bars if measurement_start < bar[0] <= close_at]
    excluded_before_knowledge = sum(1 for bar in all_bars if bar[0] <= knowledge_time)
    risk, risk_reason = _risk(entry_price, entry.get("risk_price"), side)
    atr = _float(entry.get("entry_atr"))
    favorable_threshold = _float(entry.get("favorable_threshold_pct"))
    adverse_threshold = _float(entry.get("adverse_threshold_pct"))
    if favorable_threshold is not None:
        favorable_threshold = entry_price * favorable_threshold / 100.0
    if adverse_threshold is not None:
        adverse_threshold = entry_price * adverse_threshold / 100.0
    reconstructed = not bool(result["prospective_eligible"])
    base_state = str(entry.get("state") or "complete")

    def one_window(endpoint: datetime, *, unavailable: bool = False) -> dict[str, Any]:
        expected = max(0, int((endpoint - measurement_start).total_seconds() // (M5_BAR_MINUTES * 60)))
        in_window = [bar for bar in eligible if bar[0] <= endpoint]
        if unavailable:
            return _empty_window(
                state="unavailable", reason="window_ends_after_session_close", expected=expected, endpoint=endpoint
            )
        if base_state == "no_trigger":
            return _empty_window(state="no_trigger", reason="entry_variant_did_not_trigger", expected=expected, endpoint=endpoint)
        if as_of_market < endpoint:
            return _empty_window(state="pending", reason="window_endpoint_not_completed", expected=expected, observed=len(in_window), endpoint=endpoint)
        if not in_window:
            return _empty_window(state="missing_data", reason="no_bar_at_or_before_window_endpoint", expected=expected, endpoint=endpoint)
        # A stale last observation cannot silently stretch a fixed elapsed window.
        if endpoint - in_window[-1][0] > timedelta(minutes=15):
            return _empty_window(
                state="missing_data",
                reason="no_bar_at_or_before_window_endpoint",
                expected=expected,
                observed=len(in_window),
                endpoint=endpoint,
            )
        state = "complete" if len(in_window) >= expected else "partial"
        reason = "complete_coverage" if state == "complete" else "completed_window_has_missing_bars"
        row = _moves(
            in_window,
            entry_price=entry_price,
            entry_time=measurement_start,
            side=side,
            atr=atr,
            risk=risk,
            favorable_threshold=favorable_threshold,
            adverse_threshold=adverse_threshold,
        )
        row.update(
            {
                "state": state,
                "reason": reason,
                "endpoint_time": endpoint.isoformat(),
                "coverage": _coverage(expected, len(in_window)),
                "risk_reason": risk_reason,
                "bars_excluded_before_knowledge": excluded_before_knowledge,
                "bars_rejected_invalid": rejected,
                "confirmation_valid": not reconstructed,
                "confirmation_reason": (
                    "observed_knowledge" if not reconstructed else "reconstructed_knowledge_cannot_confirm_prospective_claim"
                ),
            }
        )
        return row

    windows: dict[str, Any] = {}
    for minutes in M5_WINDOW_MINUTES:
        endpoint = measurement_start + timedelta(minutes=minutes)
        windows[f"{minutes}m"] = one_window(endpoint, unavailable=endpoint > close_at)
    windows["session_close"] = one_window(close_at)
    result["windows"] = windows
    return result


def measure_swing_forward(
    entry: Mapping[str, Any], daily_bars: Iterable[Mapping[str, Any]], *, last_completed_session: date
) -> dict[str, Any]:
    """Measure fixed daily-session endpoints; daily same-bar touch order is unknown."""
    result = _base_payload(entry, kind="D1")
    result["window_sessions"] = SWING_WINDOW_SESSIONS
    result["entry_session_convention"] = ENTRY_SESSION_CONVENTION
    knowledge_time, feasible_time = _entry_moments(entry)
    entry_price = _float(entry.get("feasible_entry_price"))
    side = result["side"]
    if side not in {"LONG", "SHORT"}:
        side = "LONG"
        result["side"] = side
    if knowledge_time is None or feasible_time is None or entry_price is None or entry_price <= 0:
        result["windows"] = {
            f"{sessions}_session": _swing_empty("invalid_entry", "invalid_or_timezone_missing_entry", 0, None)
            for sessions in SWING_WINDOW_SESSIONS
        }
        return result
    endpoints = exchange_session_endpoints(feasible_time.date(), SWING_WINDOW_SESSIONS)
    parsed: dict[date, tuple[date, float, float, float]] = {}
    for row in daily_bars or ():
        day, high, low, close = _day(row.get("date") or row.get("time")), _float(row.get("high")), _float(row.get("low")), _float(row.get("close"))
        if day is not None and high is not None and low is not None and close is not None and low <= high:
            parsed.setdefault(day, (day, high, low, close))
    risk, risk_reason = _risk(entry_price, entry.get("risk_price"), side)
    atr = _float(entry.get("entry_atr"))
    favourable = _float(entry.get("favorable_threshold_pct"))
    adverse = _float(entry.get("adverse_threshold_pct"))
    favourable = entry_price * favourable / 100.0 if favourable is not None else None
    adverse = entry_price * adverse / 100.0 if adverse is not None else None
    reconstructed = not bool(result["prospective_eligible"])
    base_state = str(entry.get("state") or "complete")
    windows: dict[str, Any] = {}
    for sessions in SWING_WINDOW_SESSIONS:
        endpoint = endpoints[sessions]
        key = f"{sessions}_session"
        coverage = {"expected_sessions": sessions, "observed_sessions": 0, "missing_sessions": sessions}
        if base_state == "no_trigger":
            windows[key] = _swing_empty("no_trigger", "entry_variant_did_not_trigger", sessions, endpoint)
            continue
        if endpoint > last_completed_session:
            windows[key] = _swing_empty("pending", "window_endpoint_not_completed", sessions, endpoint)
            continue
        in_window = [row for day, row in parsed.items() if feasible_time.date() < day <= endpoint]
        in_window.sort(key=lambda row: row[0])
        coverage["observed_sessions"] = len(in_window)
        coverage["missing_sessions"] = max(0, sessions - len(in_window))
        if not in_window:
            windows[key] = _swing_empty("missing_data", "no_daily_bar_at_or_before_window_endpoint", sessions, endpoint)
            continue
        state = "complete" if len(in_window) >= sessions and endpoint in parsed else "partial"
        row = _moves(
            [(datetime.combine(day, REGULAR_SESSION_CLOSE, tzinfo=MARKET_TZ), high, low, close) for day, high, low, close in in_window],
            entry_price=entry_price,
            entry_time=feasible_time,
            side=side,
            atr=atr,
            risk=risk,
            favorable_threshold=favourable,
            adverse_threshold=adverse,
            daily_order_ambiguous=True,
        )
        row.update(
            {
                "state": state,
                "reason": "complete_coverage" if state == "complete" else "completed_window_has_missing_daily_bars",
                "endpoint_session": endpoint.isoformat(),
                "coverage": coverage,
                "risk_reason": risk_reason,
                "confirmation_valid": not reconstructed,
                "confirmation_reason": (
                    "observed_knowledge" if not reconstructed else "reconstructed_knowledge_cannot_confirm_prospective_claim"
                ),
            }
        )
        windows[key] = row
    result["windows"] = windows
    return result


def _swing_empty(state: str, reason: str, expected: int, endpoint: date | None) -> dict[str, Any]:
    return {
        "state": state,
        "reason": reason,
        "endpoint_session": endpoint.isoformat() if endpoint else None,
        "coverage": {"expected_sessions": expected, "observed_sessions": 0, "missing_sessions": expected},
        "mfe_pct": None,
        "mae_pct": None,
        "close_pct": None,
        "mfe_atr": None,
        "mae_atr": None,
        "close_atr": None,
        "mfe_r": None,
        "mae_r": None,
        "close_r": None,
        "mfe_time": None,
        "time_to_mfe_minutes": None,
        "first_favorable_time": None,
        "first_adverse_time": None,
        "first_touch_order": "not_measured",
        "risk_reason": None,
        "confirmation_valid": False,
        "confirmation_reason": reason,
    }
