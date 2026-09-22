"""Read-only, selected completed-session learning projections (TJ-17B)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Mapping

import market_calendar

SCHEMA = "session_learning_window_v1"
WINDOW_CHOICES = (5, 10, 20)
# market_read_grades._hour_of stores the exchange-local hour.
HOUR_TIMEZONE = str(market_calendar.MARKET_TZ)


def _empty(*, requested: int = 5) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "window": {"requested": requested, "sessions": [], "start": "", "end": ""},
        "reads": {"horizons": {}},
        "by_hour": [],
        "by_environment": [],
        "trade_groups": [],
        "trade_rows": [],
        "coverage": {"reads": {"unknown_context": 0}, "trades": {"unknown_context": 0}},
        "error": "",
    }


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value or "")[:10])


def _window_sessions(end_session: Any, count: int, now: datetime | None) -> list[str]:
    moment = now or datetime.now()
    last = market_calendar.last_completed_session(moment)
    requested = _as_date(end_session) if end_session else last
    end = min(requested, last)
    while not market_calendar.is_session(end):
        end = market_calendar.previous_session(end)
    sessions = [end]
    while len(sessions) < count:
        sessions.append(market_calendar.previous_session(sessions[-1]))
    return [item.isoformat() for item in reversed(sessions)]


def _context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("context")
    return value if isinstance(value, Mapping) else {}


def _group_rows(rows: Iterable[Mapping[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    import prediction_ledger

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        horizon = prediction_ledger.horizon_of(row)
        context = _context(row)
        if kind == "hour":
            value = context.get("hour")
            try:
                key = f"{int(value):02d}:00 {HOUR_TIMEZONE}"
            except (TypeError, ValueError):
                key = f"unknown hour {HOUR_TIMEZONE}"
        else:
            key = str(context.get("d1_environment") or "unknown")
        grouped.setdefault((horizon, key), []).append(row)

    output: list[dict[str, Any]] = []
    for (horizon, key), members in sorted(grouped.items()):
        readout = prediction_ledger.build_readout(members)
        cell = dict((readout.get("horizons") or {}).get(horizon) or {})
        output.append(
            {
                "horizon": horizon,
                "key": key,
                "accuracy": dict(cell.get("accuracy") or {}),
                "baselines": dict(cell.get("baselines") or {}),
                "session_count": len({str(row.get("session") or "")[:10] for row in members}),
                "read_ids": sorted({str(row.get("read_id") or "") for row in members if row.get("read_id")}),
                "sessions": sorted({str(row.get("session") or "")[:10] for row in members if row.get("session")}),
            }
        )
    return output


def _read_trades() -> list[Any]:
    from ui.services.journal_feed import load_trades

    return list(load_trades())


def _read_labels() -> dict[str, str]:
    from d1_environment_store import labels_by_session

    return dict(labels_by_session())


def _raw(trade: Any) -> dict[str, Any]:
    raw = getattr(trade, "raw", None)
    return dict(raw) if isinstance(raw, Mapping) else (dict(trade) if isinstance(trade, Mapping) else {})


def _value(trade: Any, name: str) -> Any:
    return getattr(trade, name, _raw(trade).get(name))


def _trade_rows(trades: Iterable[Any], start: str, end: str, labels: Mapping[str, str]) -> tuple[list[dict[str, Any]], int]:
    import context_join
    import research_results

    rows: list[dict[str, Any]] = []
    unknown = 0
    for trade in trades:
        if not research_results.in_window(trade, start, end):
            continue
        reference = context_join.build_ref(
            _value(trade, "opened_at"), when=context_join.WHEN_ENTRY, labels_by_session=labels
        )
        environment = str(reference.label or "unknown")
        unknown += int(environment == "unknown")
        raw = _raw(trade)
        rows.append(
            {
                "trade_id": str(_value(trade, "trade_id") or ""),
                "opened_at": str(_value(trade, "opened_at") or ""),
                "closed_at": str(_value(trade, "closed_at") or ""),
                "closed_session": str(_value(trade, "closed_at") or "")[:10],
                "opened_session": str(_value(trade, "opened_at") or "")[:10],
                "environment": environment,
                "context_certainty": reference.certainty,
                "currency": str(_value(trade, "currency") or raw.get("currency") or ""),
                "net_pnl": _value(trade, "net_pnl"),
                "fees": _value(trade, "fees"),
                "tag_status": str(raw.get("tag_status") or ""),
                "holding": research_results.holding_bucket(trade),
            }
        )
    return rows, unknown


def _trade_groups(trades: list[Any], start: str, end: str, labels: Mapping[str, str]) -> list[dict[str, Any]]:
    """Use Results' owner stats; never sum currencies or invent a trade rate."""
    import context_join
    import evidence_stats
    import research_results

    labels_seen = {"unknown"}
    for trade in trades:
        ref = context_join.build_ref(
            _value(trade, "opened_at"), when=context_join.WHEN_ENTRY, labels_by_session=labels
        )
        labels_seen.add(str(ref.label or "unknown"))
    out: list[dict[str, Any]] = []
    for horizon in ("day", "swing"):
        for environment in (research_results.ENVIRONMENT_ALL, *sorted(labels_seen)):
            view = research_results.build_results_view(
                population="mine", horizon=horizon, window=(start, end),
                journal_trades=trades, environment_filter=environment,
                environment_labels=labels,
            )
            section = view.sections[0] if view.sections else None
            stats = dict(section.stats) if section else {}
            owner_rate = dict((section.analytics or {}).get("overall") or {}).get("win_rate") if section else None
            out.append({
                "horizon": horizon,
                "environment": environment,
                "scope": "all_contexts" if environment == research_results.ENVIRONMENT_ALL else "entry_environment",
                "stats": stats,
                "win_rate": owner_rate,
                "meets_floor": int(stats.get("closed") or 0) >= evidence_stats.MIN_REPORTABLE_N,
                "trade_ids": list(stats.get("trade_ids") or ()),
                "reason": str(section.sentence if section else "unreadable"),
            })
    return out


def read_learning_window(
    *, end_session: Any, sessions: int = 5, root: Any = None,
    trades: Iterable[Any] | None = None, environment_labels: Mapping[str, str] | None = None,
    now: datetime | None = None, start_session: Any = "",
) -> dict[str, Any]:
    """One worker-only projection over 5, 10, or 20 completed NYSE sessions."""
    if sessions not in WINDOW_CHOICES:
        raise ValueError(f"sessions must be one of {WINDOW_CHOICES}")
    payload = _empty(requested=sessions)
    try:
        chosen = _window_sessions(end_session, sessions, now)
        if start_session:
            first = _as_date(start_session)
            chosen = [day for day in chosen if _as_date(day) >= first]
        if not chosen:
            return payload
        payload["window"] = {"requested": sessions, "sessions": chosen, "start": chosen[0], "end": chosen[-1]}
    except Exception as exc:  # calendar failures must be visible, not a blank page
        payload["error"] = f"window unavailable: {exc}"
        return payload

    try:
        import market_read_grades as grades
        import prediction_ledger

        clicked = prediction_ledger.read_ledger(chosen, root=root, source=grades.SOURCE_CLICK)
        payload["reads"] = prediction_ledger.build_readout(clicked)
        payload["by_hour"] = _group_rows(clicked, kind="hour")
        payload["by_environment"] = _group_rows(clicked, kind="environment")
        payload["coverage"]["horizons"] = {
            name: {
                "sessions": sorted({str(row.get("session") or "")[:10] for row in clicked
                                    if prediction_ledger.horizon_of(row) == name and row.get("session")}),
                "read_ids": sorted({str(row.get("read_id") or "") for row in clicked
                                    if prediction_ledger.horizon_of(row) == name and row.get("read_id")}),
            }
            for name in prediction_ledger.HORIZONS
        }
        payload["coverage"]["reads"] = {
            "current": len(clicked),
            "unknown_context": sum(
                str(_context(row).get("d1_environment") or "unknown").lower()
                in {"", "unknown", "unmeasured", "unavailable"}
                for row in clicked
            ),
        }
    except Exception as exc:  # one reader should not blank prior verified sections
        payload["error"] = f"reads unavailable: {exc}"

    try:
        listed = list(trades) if trades is not None else _read_trades()
        labels = dict(environment_labels) if environment_labels is not None else _read_labels()
        rows, unknown = _trade_rows(listed, payload["window"]["start"], payload["window"]["end"], labels)
        payload["trade_rows"] = rows
        payload["trade_groups"] = _trade_groups(listed, payload["window"]["start"], payload["window"]["end"], labels)
        payload["coverage"]["trades"] = {"closed_in_window": len(rows), "unknown_context": unknown}
    except Exception as exc:
        payload["error"] = "; ".join(filter(None, (payload["error"], f"trades unavailable: {exc}")))
    return payload


__all__ = ["HOUR_TIMEZONE", "SCHEMA", "WINDOW_CHOICES", "read_learning_window"]
