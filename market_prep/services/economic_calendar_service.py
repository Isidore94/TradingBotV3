from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from market_prep.config_loader import CONFIG_DIR, load_market_prep_config
from market_prep.models import MarketPrepConfig
from market_prep.services.forexfactory_calendar_service import load_forexfactory_events


ECONOMIC_EVENTS_PRIORITY_FILE = CONFIG_DIR / "economic_events_priority.json"
MANUAL_ECONOMIC_CALENDAR_FILE = CONFIG_DIR / "manual_economic_calendar.json"
NO_CONFIGURED_EVENTS_MESSAGE = "No configured economic events found."
PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

DEFAULT_PRIORITY_PAYLOAD = {
    "schema_version": 1,
    "high_priority": [
        "FOMC Rate Decision",
        "FOMC Minutes",
        "Fed Chair Speech",
        "CPI",
        "Core CPI",
        "PPI",
        "Core PPI",
        "PCE",
        "Core PCE",
        "Nonfarm Payrolls",
        "Unemployment Rate",
        "Average Hourly Earnings",
        "Jobless Claims",
        "ADP Employment",
        "ISM Manufacturing",
        "ISM Services",
        "Retail Sales",
        "GDP",
        "JOLTS",
        "Consumer Sentiment",
        "10Y Treasury Auction",
        "30Y Treasury Auction",
    ],
    "medium_priority": [],
    "low_priority": [],
}

MANUAL_CALENDAR_TEMPLATE = [
    {
        "date": "2026-04-29",
        "time_et": "08:30",
        "event": "GDP",
        "priority": "HIGH",
        "notes": "",
    }
]


def get_upcoming_economic_events(
    *,
    start_date: date | None = None,
    days: int = 7,
    manual_calendar_path: Path = MANUAL_ECONOMIC_CALENDAR_FILE,
    priority_path: Path = ECONOMIC_EVENTS_PRIORITY_FILE,
    config: MarketPrepConfig | None = None,
    include_forexfactory: bool = True,
    refresh_forexfactory: bool = False,
) -> dict[str, Any]:
    start = start_date or datetime.now().date()
    end = start + timedelta(days=max(0, int(days)))
    active_config = config or load_market_prep_config()
    priority_map = load_event_priorities(priority_path)
    manual_events = load_manual_economic_calendar(manual_calendar_path)
    filtered_events = [
        _normalize_manual_event(event, priority_map)
        for event in manual_events
        if _is_event_in_window(event, start, end)
    ]
    filtered_events = [event for event in filtered_events if event is not None]
    forexfactory_payload = (
        _load_forexfactory_payload(
            active_config,
            start=start,
            days=days,
            refresh=refresh_forexfactory,
        )
        if include_forexfactory
        else _forexfactory_disabled_payload()
    )
    merged_events = _merge_economic_events(filtered_events, forexfactory_payload.get("events", []))
    merged_events.sort(key=_event_sort_key)
    status = _forexfactory_status(forexfactory_payload)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "manual+ForexFactory" if status.get("status") not in {"disabled", ""} else "manual",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "events": merged_events,
        "forexfactory_status": status,
        "warnings": status.get("warnings", []),
        "message": "" if merged_events else NO_CONFIGURED_EVENTS_MESSAGE,
    }


def get_economic_events_for_date(
    target_date: date | None = None,
    *,
    manual_calendar_path: Path = MANUAL_ECONOMIC_CALENDAR_FILE,
    priority_path: Path = ECONOMIC_EVENTS_PRIORITY_FILE,
    config: MarketPrepConfig | None = None,
    include_forexfactory: bool = True,
    refresh_forexfactory: bool = False,
) -> dict[str, Any]:
    target = target_date or datetime.now().date()
    return get_upcoming_economic_events(
        start_date=target,
        days=0,
        manual_calendar_path=manual_calendar_path,
        priority_path=priority_path,
        config=config,
        include_forexfactory=include_forexfactory,
        refresh_forexfactory=refresh_forexfactory,
    )


def get_current_week_economic_events(
    target_date: date | None = None,
    *,
    manual_calendar_path: Path = MANUAL_ECONOMIC_CALENDAR_FILE,
    priority_path: Path = ECONOMIC_EVENTS_PRIORITY_FILE,
    config: MarketPrepConfig | None = None,
    include_forexfactory: bool = True,
    refresh_forexfactory: bool = False,
) -> dict[str, Any]:
    start = _current_week_start(target_date or datetime.now().date())
    return get_upcoming_economic_events(
        start_date=start,
        days=6,
        manual_calendar_path=manual_calendar_path,
        priority_path=priority_path,
        config=config,
        include_forexfactory=include_forexfactory,
        refresh_forexfactory=refresh_forexfactory,
    )


def load_event_priorities(path: Path = ECONOMIC_EVENTS_PRIORITY_FILE) -> dict[str, str]:
    ensure_economic_events_priority(path)
    payload = _read_json(path, default={})
    priority_map: dict[str, str] = {}
    _add_priority_events(priority_map, "HIGH", DEFAULT_PRIORITY_PAYLOAD["high_priority"])
    _add_priority_events(priority_map, "MEDIUM", DEFAULT_PRIORITY_PAYLOAD["medium_priority"])
    _add_priority_events(priority_map, "LOW", DEFAULT_PRIORITY_PAYLOAD["low_priority"])

    if isinstance(payload, dict):
        priorities = payload.get("priorities")
        if isinstance(priorities, dict):
            for priority, events in priorities.items():
                _add_priority_events(priority_map, priority, events)

        for priority_key, priority_label in (
            ("high_priority", "HIGH"),
            ("medium_priority", "MEDIUM"),
            ("low_priority", "LOW"),
        ):
            _add_priority_events(priority_map, priority_label, payload.get(priority_key))

        events = payload.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_name = str(event.get("event") or event.get("name") or "").strip()
                priority = str(event.get("priority") or "").strip().upper()
                if event_name and priority:
                    priority_map[_priority_key(event_name)] = priority

    return priority_map


def ensure_economic_events_priority(path: Path = ECONOMIC_EVENTS_PRIORITY_FILE) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return False
    _write_json_atomic(target, DEFAULT_PRIORITY_PAYLOAD)
    return True


def load_manual_economic_calendar(path: Path = MANUAL_ECONOMIC_CALENDAR_FILE) -> list[dict[str, Any]]:
    ensure_manual_economic_calendar(path)
    payload = _read_json(path, default=[])
    if not isinstance(payload, list):
        logging.warning("Manual economic calendar at %s is not a list.", path)
        return []
    return [event for event in payload if isinstance(event, dict)]


def ensure_manual_economic_calendar(path: Path = MANUAL_ECONOMIC_CALENDAR_FILE) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return False
    _write_json_atomic(target, MANUAL_CALENDAR_TEMPLATE)
    return True


def _add_priority_events(priority_map: dict[str, str], priority: str, events: Any) -> None:
    if not isinstance(events, list):
        return
    normalized_priority = str(priority or "").strip().upper()
    if not normalized_priority:
        return
    for event_name in events:
        cleaned = str(event_name or "").strip()
        if cleaned:
            priority_map[_priority_key(cleaned)] = normalized_priority


def _normalize_manual_event(event: dict[str, Any], priority_map: dict[str, str]) -> dict[str, str] | None:
    event_date = _parse_event_date(event.get("date"))
    name = str(event.get("event") or "").strip()
    if event_date is None or not name:
        return None
    priority = str(event.get("priority") or "").strip().upper()
    if not priority:
        priority = priority_map.get(_priority_key(name), "LOW")
    return {
        "date": event_date.isoformat(),
        "time_et": str(event.get("time_et") or "").strip(),
        "event": name,
        "currency": str(event.get("currency") or "").strip().upper(),
        "priority": priority,
        "impact": str(event.get("impact") or "").strip().lower(),
        "previous": str(event.get("previous") or "").strip(),
        "forecast": str(event.get("forecast") or "").strip(),
        "actual": str(event.get("actual") or "").strip(),
        "source": str(event.get("source") or "manual").strip(),
        "notes": str(event.get("notes") or "").strip(),
    }


def _is_event_in_window(event: dict[str, Any], start: date, end: date) -> bool:
    event_date = _parse_event_date(event.get("date"))
    return event_date is not None and start <= event_date <= end


def _parse_event_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        text = str(value or "").strip()
        return datetime.fromisoformat(text).date() if text else None
    except ValueError:
        return None


def _current_week_start(value: date) -> date:
    if value.weekday() == 6:
        return value
    return value - timedelta(days=value.weekday())


def _event_sort_key(event: dict[str, Any]) -> tuple[int, str, str, str]:
    priority = str(event.get("priority") or "LOW").upper()
    return (
        PRIORITY_ORDER.get(priority, 3),
        str(event.get("date") or ""),
        str(event.get("time_et") or ""),
        str(event.get("event") or ""),
    )


def _load_forexfactory_payload(
    config: MarketPrepConfig,
    *,
    start: date,
    days: int,
    refresh: bool,
) -> dict[str, Any]:
    try:
        return load_forexfactory_events(
            config,
            start_date=start,
            days_ahead=days,
            force_refresh=refresh,
        )
    except Exception as exc:
        logging.getLogger("market_prep").exception("ForexFactory calendar enrichment failed.")
        return {
            "source": "ForexFactory",
            "events": [],
            "status": "failed",
            "status_label": "Failed, using manual fallback",
            "message": "ForexFactory Failed, using manual fallback",
            "warnings": [str(exc)],
        }


def _forexfactory_disabled_payload() -> dict[str, Any]:
    return {
        "source": "ForexFactory",
        "events": [],
        "status": "disabled",
        "status_label": "Disabled",
        "message": "ForexFactory Disabled",
        "warnings": [],
    }


def _forexfactory_status(payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status") or "disabled")
    return {
        "status": status,
        "status_label": str(payload.get("status_label") or status),
        "message": str(payload.get("message") or ""),
        "cache_path": str(payload.get("cache_path") or ""),
        "warnings": list(payload.get("warnings") or []),
    }


def _merge_economic_events(manual_events: list[dict[str, Any]], forexfactory_events: Any) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for event in manual_events:
        merged[_event_key(event)] = dict(event)
    forex_events = forexfactory_events if isinstance(forexfactory_events, list) else []
    for event in forex_events:
        if not isinstance(event, dict):
            continue
        key = _event_key(event)
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(event)
            continue
        combined = dict(event)
        for field, value in existing.items():
            if value or field in {"event", "priority", "notes"}:
                combined[field] = value
        for field in ("currency", "impact", "previous", "forecast", "actual"):
            if event.get(field) and not combined.get(field):
                combined[field] = event.get(field)
        combined["source"] = "manual+ForexFactory"
        merged[key] = combined
    return list(merged.values())


def _event_key(event: dict[str, Any]) -> tuple[str, str, str, str]:
    currency = str(event.get("currency") or "USD").strip().upper()
    return (
        str(event.get("date") or "").strip(),
        str(event.get("time_et") or "").strip(),
        _priority_key(str(event.get("event") or "")),
        currency,
    )


def _priority_key(event_name: str) -> str:
    return " ".join(str(event_name or "").strip().lower().split())


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Failed reading economic calendar JSON at %s: %s", path, exc)
        return default


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
