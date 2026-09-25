"""How today's M5 alerts have done since they fired, in R - "Working now".

Trader, 2026-09-22 (change #3): instant feedback on which trades are working
right now. The desk's M5 column shows one line built from this module.

**Display only.** Nothing here reads or changes a detector, a score, an alert
or a store. It takes the entry/stop the alert already carried
(`payload["feedback"]`), the bot's CACHED M5 bars (never an IB fetch), and
does arithmetic.

Rules it keeps:

* Completed bars only: a bar counts once `dt + 5 min <= now`.
* Only bars from the M5 bucket that holds the alert's arrival onward.
* The first bar that touches the stop ends the trade at -1R, or worse when
  that bar OPENS beyond the stop (a gap fills at the open, not the stop).
* Missing prices or no usable bars is UNKNOWN - never a result, never averaged.
* One row per symbol+side per day: the FIRST alert. Repeats add nothing.
* Timestamps carry zones. `received_at` is stamped America/New_York; a naive
  bar stamp is market-local (IB `formatDate=1`), so the desk's zone is
  ATTACHED to it, and an aware one is converted as the instant it is.

No Qt here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import setup_grades

NY = ZoneInfo("America/New_York")
BAR_SPAN = timedelta(minutes=5)

OK = "ok"
OPEN = "open"
STOPPED = "stopped"
UNKNOWN = "unknown"

_SIDES = {"LONG": "LONG", "SHORT": "SHORT", "BUY": "LONG", "SELL": "SHORT"}
_MINUS = "−"


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None  # NaN is unmeasured


def desk_zone() -> tzinfo:
    """The zone a naive bar stamp is written in (the desk's market-local zone)."""
    try:
        from market_session import get_market_local_timezone

        zone, _name = get_market_local_timezone()
        return zone
    except Exception:  # noqa: BLE001 - no settings or tzdata: the market's own zone
        return NY


def to_ny(moment: datetime, naive_zone: tzinfo | None = None) -> datetime:
    """An aware New York datetime. Naive means `naive_zone` (ATTACH, never strip)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=naive_zone or desk_zone())
    return moment.astimezone(NY)


def _feedback(alert: Any) -> Mapping[str, Any]:
    payload = getattr(alert, "payload", None)
    feedback = payload.get("feedback") if isinstance(payload, dict) else None
    return feedback if isinstance(feedback, dict) else {}


def entry_from_alert(alert: Any, received_at: datetime) -> dict[str, Any] | None:
    """The trade an alert describes, or None when it is not a LONG/SHORT alert.

    A trade alert whose entry or stop is missing, or whose risk is not
    positive, comes back with `status` UNKNOWN - still one row, never a result.
    """
    feedback = _feedback(alert)
    symbol = str(feedback.get("symbol") or getattr(alert, "symbol", "") or "").strip().upper()
    side = _SIDES.get(str(getattr(alert, "side", "") or "").strip().upper())
    if side is None:
        side = _SIDES.get(str(feedback.get("direction") or "").strip().upper())
    if not symbol or side is None:
        return None
    entry = _float(feedback.get("entry_price"))
    stop = _float(feedback.get("stop_price"))
    risk = _float(feedback.get("risk_per_share"))
    if risk is None and entry is not None and stop is not None:
        risk = abs(entry - stop)
    status, reason = OK, ""
    if entry is None or stop is None:
        status, reason = UNKNOWN, "no entry/stop"
    elif risk is None or risk <= 0:
        status, reason = UNKNOWN, "no risk"
    return {
        "symbol": symbol,
        "side": side,
        "bounce_types": str(feedback.get("bounce_types") or getattr(alert, "trigger", "") or ""),
        "entry": entry,
        "stop": stop,
        "risk": risk,
        "received_at": to_ny(received_at, NY),
        "status": status,
        "reason": reason,
    }


def _bucket(moment: datetime) -> datetime:
    return moment.replace(minute=moment.minute - moment.minute % 5, second=0, microsecond=0)


def _r(value: float) -> float:
    return round(value, 2)


def result_for(
    entry: Mapping[str, Any],
    bars: Iterable[Mapping[str, Any]] | None,
    now: datetime,
    *,
    naive_zone: tzinfo | None = None,
) -> dict[str, Any]:
    """How `entry` has done over the completed bars since it fired."""
    result = {
        **dict(entry),
        "status": UNKNOWN,
        "r": None,
        "best_r": None,
        "worst_r": None,
        "bars": 0,
    }
    if entry.get("status") != OK:
        result["reason"] = entry.get("reason") or "no entry/stop"
        return result
    result["reason"] = "no bars"
    zone = naive_zone or desk_zone()
    now_ny = to_ny(now, zone)
    start = _bucket(entry["received_at"])
    usable: list[tuple[datetime, float, float, float, float]] = []
    for bar in bars or ():
        try:
            begins = to_ny(bar["dt"], zone)
            row = (begins, float(bar["open"]), float(bar["high"]), float(bar["low"]),
                   float(bar["close"]))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if begins < start or begins.date() != start.date() or begins + BAR_SPAN > now_ny:
            continue
        usable.append(row)
    if not usable:
        return result
    usable.sort(key=lambda row: row[0])

    price, stop, risk = float(entry["entry"]), float(entry["stop"]), float(entry["risk"])
    long_side = entry["side"] == "LONG"
    sign = 1.0 if long_side else -1.0

    def r_of(value: float) -> float:
        return sign * (value - price) / risk

    best = worst = None
    for _begins, open_, high, low, _close in usable:
        result["bars"] += 1
        touched = low <= stop if long_side else high >= stop
        if touched:
            gapped = open_ <= stop if long_side else open_ >= stop
            stopped_r = min(r_of(open_), -1.0) if gapped else -1.0
            result.update(
                status=STOPPED,
                r=_r(stopped_r),
                best_r=_r(best if best is not None else stopped_r),
                worst_r=_r(min(worst, stopped_r) if worst is not None else stopped_r),
                reason="",
            )
            return result
        favourable, adverse = (r_of(high), r_of(low)) if long_side else (r_of(low), r_of(high))
        best = favourable if best is None else max(best, favourable)
        worst = adverse if worst is None else min(worst, adverse)
    result.update(
        status=OPEN,
        r=_r(r_of(usable[-1][4])),
        best_r=_r(best),
        worst_r=_r(worst),
        reason="",
    )
    return result


class AlertBook:
    """Today's trade alerts, the FIRST per symbol+side per day, in arrival order."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, Any], dict[str, Any]] = {}

    def add(self, alert: Any, received_at: datetime) -> bool:
        entry = entry_from_alert(alert, received_at)
        if entry is None:
            return False
        key = (entry["symbol"], entry["side"], entry["received_at"].date())
        if key in self._entries:
            return False
        self._entries[key] = entry
        return True

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def summarize(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per grade: count, average R, how many open / stopped. Unknowns apart."""
    groups: dict[str, dict[str, Any]] = {}
    unknown = 0
    for row in results:
        if row.get("status") not in (OPEN, STOPPED) or row.get("r") is None:
            unknown += 1
            continue
        grade = str(row.get("grade") or setup_grades.NEW)
        group = groups.setdefault(
            grade, {"grade": grade, "count": 0, "open": 0, "stopped": 0, "_sum": 0.0}
        )
        group["count"] += 1
        group["open" if row["status"] == OPEN else "stopped"] += 1
        group["_sum"] += float(row["r"])
    ordered = []
    for grade in sorted(groups, key=setup_grades.sort_rank):
        group = groups[grade]
        total = group.pop("_sum")
        group["avg_r"] = _r(total / group["count"])
        ordered.append(group)
    return {"grades": ordered, "unknown": unknown}


def format_r(value: float | None) -> str:
    if value is None:
        return "?"
    value = round(float(value), 1)
    if value < 0:
        return f"{_MINUS}{abs(value):.1f}R"
    return f"+{abs(value):.1f}R"


def strip_text(summary: Mapping[str, Any]) -> str:
    parts = ["Working now"]
    for group in summary.get("grades") or ():
        label = setup_grades.badge(group["grade"])
        parts.append(f"{label} {group['count']} ({format_r(group['avg_r'])})")
    unknown = int(summary.get("unknown") or 0)
    if unknown:
        parts.append(f"{unknown} no data")
    if len(parts) == 1:
        parts.append("no M5 alerts yet today")
    return " · ".join(parts)


def tooltip_line(row: Mapping[str, Any]) -> str:
    received = row.get("received_at")
    clock = received.strftime("%H:%M") if isinstance(received, datetime) else "--:--"
    head = f"{row.get('symbol', '')} {row.get('side', '')} {clock} [{setup_grades.badge(row.get('grade'))}]"
    status = row.get("status")
    if status == OPEN:
        return (
            f"{head} {format_r(row.get('r'))} now · best {format_r(row.get('best_r'))}"
            f" · worst {format_r(row.get('worst_r'))}"
        )
    if status == STOPPED:
        return f"{head} STOPPED {format_r(row.get('r'))} · best {format_r(row.get('best_r'))}"
    reason = str(row.get("reason") or "")
    return f"{head} no data ({reason})" if reason else f"{head} no data"
