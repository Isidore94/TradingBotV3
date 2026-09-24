"""The EVENING catch-up card's content: the best of the morning, per section.

Pure and Qt-free, so it runs on a worker. It ranks nothing new: it reuses the
evening strength checks' held verdicts, the Movers board's own order, the
Alert Center tiers, `evening_mode.select_best_d1_rows` and the price-alert
trigger log. Presentation only; nothing here writes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping

import evening_mode

#: Rows per section.
SECTION_LIMIT = 5

_TIER_RANK = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}


def _side(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("short"):
        return "short"
    if text.startswith("long"):
        return "long"
    return ""


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return ""


def _strength_rows(
    side: str,
    persistence: Mapping[str, Mapping[str, Any]],
    movers_board: Mapping[str, Any] | None,
    limit: int,
) -> list[dict[str, str]]:
    """Held staged picks first (by score), then the Movers board's Pop order."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    held = [
        (symbol, info)
        for symbol, info in (persistence or {}).items()
        if info.get("verdict") == "held" and _side(info.get("side")) == side
    ]
    held.sort(key=lambda item: -float(item[1].get("score") or 0.0))
    for symbol, info in held:
        symbol = str(symbol or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        rows.append(
            {"symbol": symbol, "side": side, "text": f"held strong: {info.get('detail', '')}".strip()}
        )
    pop = ((movers_board or {}).get("pop") or {}).get(side) or []
    for row in pop:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        bits = []
        day = _fmt_pct(row.get("day_pct"))
        if day:
            bits.append(f"day {day}")
        move = _fmt_pct(row.get("move15_pct"))
        if move:
            bits.append(f"15m {move}")
        rows.append({"symbol": symbol, "side": side, "text": "mover " + ", ".join(bits)})
    return rows[:limit]


def _ranked_alerts(alerts: Iterable[Mapping[str, Any]], *, d1: bool) -> list[dict[str, Any]]:
    """One row per (symbol, side): best tier, then most fires, then earliest."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for alert in alerts or ():
        if bool(alert.get("is_d1")) != d1:
            continue
        symbol = str(alert.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        side = _side(alert.get("side"))
        key = (symbol, side)
        tier = str(alert.get("tier") or "").strip().upper()
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = {
                "symbol": symbol,
                "side": side,
                "tier": tier,
                "count": 1,
                "first": str(alert.get("time_text") or ""),
                "label": str(alert.get("cell") or alert.get("trigger") or "").strip(),
            }
            continue
        entry["count"] += 1
        if _TIER_RANK.get(tier, -1) > _TIER_RANK.get(entry["tier"], -1):
            entry["tier"] = tier
            entry["label"] = str(alert.get("cell") or alert.get("trigger") or "").strip()
    ranked = list(grouped.values())
    ranked.sort(
        key=lambda item: (
            -_TIER_RANK.get(item["tier"], -1),
            -item["count"],
            item["first"],
        )
    )
    return ranked


def _alert_text(item: Mapping[str, Any]) -> str:
    bits = []
    if item.get("tier"):
        bits.append(f"{item['tier']}-tier")
    if item.get("label"):
        bits.append(str(item["label"]))
    if int(item.get("count") or 0) > 1:
        bits.append(f"x{item['count']}")
    if item.get("first"):
        bits.append(f"first {item['first']}")
    return " ".join(bits)


def _swing_rows(
    alerts: Iterable[Mapping[str, Any]], swing_rows: Iterable[Any], limit: int
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in _ranked_alerts(alerts, d1=True):
        key = (item["symbol"], item["side"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {"symbol": item["symbol"], "side": item["side"], "text": "D1 alert " + _alert_text(item)}
        )
    best = evening_mode.select_best_d1_rows(swing_rows or [], per_side=limit)
    for side in ("long", "short"):
        for row in best.get(side) or []:
            key = (str(row.get("symbol") or ""), side)
            if not key[0] or key in seen:
                continue
            seen.add(key)
            bits = [str(row.get("bucket") or "").strip()]
            if row.get("expected_r") is not None:
                bits.append(f"{float(row['expected_r']):.2f}R")
            rows.append(
                {"symbol": key[0], "side": side, "text": "scan " + " ".join(b for b in bits if b)}
            )
    return rows[:limit]


def _price_alert_rows(
    triggers: Iterable[Mapping[str, Any]], since: datetime | None, limit: int
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for trigger in triggers or ():
        if since is not None:
            stamp = f"{trigger.get('date', '')} {trigger.get('at', '')}".strip()
            try:
                fired = datetime.fromisoformat(stamp)
            except ValueError:
                fired = None
            # An unreadable time is kept: missing data never hides an alert.
            if fired is not None and fired < since:
                continue
        symbol = str(trigger.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        direction = "above" if str(trigger.get("side")) == "above" else "below"
        rows.append(
            {
                "symbol": symbol,
                "side": "",
                "text": (
                    f"{trigger.get('at', '')} crossed {direction} {trigger.get('level', '')} "
                    f"(last {trigger.get('last', '')})"
                ).strip(),
            }
        )
    return rows[-limit:] if limit else rows


def build_catchup(
    *,
    alerts: Iterable[Mapping[str, Any]] = (),
    movers_board: Mapping[str, Any] | None = None,
    persistence: Mapping[str, Mapping[str, Any]] | None = None,
    swing_rows: Iterable[Any] = (),
    price_triggers: Iterable[Mapping[str, Any]] = (),
    since: datetime | None = None,
    now: datetime | None = None,
    limit: int = SECTION_LIMIT,
) -> dict[str, Any]:
    """The card: a title line and five sections of clickable rows."""
    alerts = list(alerts or ())
    persistence = dict(persistence or {})
    pullbacks = [
        {"symbol": item["symbol"], "side": item["side"], "text": _alert_text(item)}
        for item in _ranked_alerts(alerts, d1=False)
    ][:limit]
    sections = [
        {"key": "longs", "title": "Strongest longs", "rows": _strength_rows("long", persistence, movers_board, limit)},
        {"key": "shorts", "title": "Weakest shorts", "rows": _strength_rows("short", persistence, movers_board, limit)},
        {"key": "pullbacks", "title": "Strong on pullbacks (M5 alerts)", "rows": pullbacks},
        {"key": "swing", "title": "Best swing setups (D1)", "rows": _swing_rows(alerts, swing_rows, limit)},
        {"key": "price", "title": "Price alerts that fired", "rows": _price_alert_rows(price_triggers, since, limit)},
    ]
    moment = now or datetime.now()
    since_text = since.strftime("%H:%M") if since else "the open"
    return {
        "title": f"While you slept (since {since_text}, built {moment.strftime('%H:%M')})",
        "sections": sections,
        "alert_count": len(alerts),
    }
