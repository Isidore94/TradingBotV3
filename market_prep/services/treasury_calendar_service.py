from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from market_prep.cache import read_json_cache, write_json_cache
from market_prep.models import MarketPrepConfig


TREASURY_CALENDAR_CACHE_FILE_NAME = "treasury_calendar_cache.json"
TREASURY_UPCOMING_AUCTIONS_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/upcoming_auctions"
)
TREASURY_DATASET_URL = "https://fiscaldata.treasury.gov/datasets/upcoming-auctions/"
DEFAULT_SETTINGS = {
    "cache_ttl_hours": 24,
    "days_ahead": 14,
    "important_auctions": ["2-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"],
    "rates_market_driver": True,
    "request_timeout_seconds": 15,
}


def get_treasury_calendar_events(
    config: MarketPrepConfig,
    *,
    start_date: date | None = None,
    days_ahead: int | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    start = start_date or datetime.now().date()
    settings = get_treasury_calendar_settings(config)
    days = _safe_int(days_ahead, _safe_int(settings.get("days_ahead"), 14))
    end = start + timedelta(days=max(0, days))
    cache_path = _cache_path(config)
    generated_at = datetime.now().isoformat(timespec="seconds")

    if not is_treasury_calendar_enabled(config):
        return _payload(
            generated_at,
            start,
            end,
            [],
            status="disabled",
            status_label="Disabled",
            message="Treasury calendar disabled.",
            cache_path=cache_path,
        )

    cache_payload = read_json_cache(cache_path, default={})
    ttl = timedelta(hours=max(0.0, _safe_float(settings.get("cache_ttl_hours"), 24.0)))
    if not force_refresh and _is_cache_fresh(cache_payload, ttl):
        events = _events_in_window(cache_payload.get("events"), start, end)
        return _payload(
            generated_at,
            start,
            end,
            events,
            status="cache",
            status_label="Loaded from cache",
            message="" if events else "No Treasury auction events found.",
            cache_path=cache_path,
        )

    try:
        rows = _fetch_upcoming_auctions(start, end, settings)
        events = [_normalize_auction_row(row, settings) for row in rows if isinstance(row, dict)]
        events = [event for event in events if event is not None]
        events.sort(key=_event_sort_key)
        write_json_cache(
            cache_path,
            {
                "fetched_at": generated_at,
                "source": "U.S. Treasury FiscalData",
                "events": events,
            },
        )
        window_events = _events_in_window(events, start, end)
        return _payload(
            generated_at,
            start,
            end,
            window_events,
            status="refreshed",
            status_label="Refreshed",
            message="" if window_events else "No Treasury auction events found.",
            cache_path=cache_path,
        )
    except Exception as exc:
        logging.getLogger("market_prep").exception("Treasury calendar enrichment failed.")
        cached_events = _events_in_window(cache_payload.get("events"), start, end)
        if cached_events:
            return _payload(
                generated_at,
                start,
                end,
                cached_events,
                status="cache_fallback",
                status_label="Failed, using cache",
                message="Treasury calendar failed; using cached events.",
                warnings=[str(exc)],
                cache_path=cache_path,
            )
        return _payload(
            generated_at,
            start,
            end,
            [],
            status="failed",
            status_label="Failed",
            message=f"Treasury calendar unavailable: {exc}",
            warnings=[str(exc)],
            cache_path=cache_path,
        )


def is_treasury_calendar_enabled(config: MarketPrepConfig | None) -> bool:
    if config is None:
        return False
    return bool(config.features.get("treasury_calendar", False))


def get_treasury_calendar_settings(config: MarketPrepConfig | None) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if config is not None and isinstance(config.treasury_calendar, dict):
        settings.update(config.treasury_calendar)
    return settings


def _fetch_upcoming_auctions(start: date, end: date, settings: dict[str, Any]) -> list[dict[str, Any]]:
    params = {
        "filter": f"auction_date:gte:{start.isoformat()},auction_date:lte:{end.isoformat()}",
        "sort": "auction_date,security_type,security_term",
        "page[size]": "500",
    }
    url = TREASURY_UPCOMING_AUCTIONS_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TradingBotV3 market prep",
            "Accept": "application/json",
        },
    )
    timeout = max(1, _safe_int(settings.get("request_timeout_seconds"), 15))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    rows = payload.get("data") if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _normalize_auction_row(row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any] | None:
    auction_date = _parse_date(row.get("auction_date"))
    security_type = str(row.get("security_type") or "").strip()
    security_term = str(row.get("security_term") or "").strip()
    if auction_date is None or not security_type or not security_term:
        return None
    event_name = f"{security_term} Treasury {security_type} Auction"
    offering_amount = _format_dollars(row.get("offering_amt"))
    notes = []
    if offering_amount:
        notes.append(f"Offering {offering_amount}")
    reopening = str(row.get("reopening") or "").strip()
    if reopening:
        notes.append(f"Reopening: {reopening}")
    issue_date = str(row.get("issue_date") or "").strip()
    if issue_date:
        notes.append(f"Issue date {issue_date}")
    cusip = str(row.get("cusip") or "").strip()
    if cusip:
        notes.append(f"CUSIP {cusip}")
    return {
        "date": auction_date.isoformat(),
        "time_et": "",
        "event": event_name,
        "priority": _auction_priority(security_term, security_type, settings),
        "source": "U.S. Treasury FiscalData",
        "url": TREASURY_DATASET_URL,
        "notes": "; ".join(notes),
        "security_type": security_type,
        "security_term": security_term,
        "offering_amount": _safe_int_or_none(row.get("offering_amt")),
        "cusip": cusip,
    }


def _auction_priority(security_term: str, security_type: str, settings: dict[str, Any]) -> str:
    term = _normalize_term(security_term)
    security = str(security_type or "").strip().lower()
    important = {_normalize_term(item) for item in settings.get("important_auctions", []) if str(item).strip()}
    rates_driver = bool(settings.get("rates_market_driver", True))
    if term in {"10-year", "20-year", "30-year"}:
        return "HIGH" if rates_driver else "MEDIUM"
    if term in {"2-year", "5-year", "7-year"}:
        return "MEDIUM"
    if term in important:
        return "MEDIUM"
    if "bill" in security:
        return "LOW"
    return "LOW"


def _normalize_term(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "-")


def _payload(
    generated_at: str,
    start: date,
    end: date,
    events: list[dict[str, Any]],
    *,
    status: str,
    status_label: str,
    message: str,
    warnings: list[str] | None = None,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "source": "U.S. Treasury FiscalData",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "events": events,
        "status": status,
        "status_label": status_label,
        "message": message,
        "warnings": warnings or [],
        "cache_path": str(cache_path or ""),
    }


def _cache_path(config: MarketPrepConfig) -> Path:
    cache_dir = config.resolved_paths().get("cache_dir") or Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / TREASURY_CALENDAR_CACHE_FILE_NAME


def _events_in_window(rows: Any, start: date, end: date) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return events
    for row in rows:
        if not isinstance(row, dict):
            continue
        event_date = _parse_date(row.get("date"))
        if event_date is not None and start <= event_date <= end:
            events.append(dict(row))
    events.sort(key=_event_sort_key)
    return events


def _event_sort_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    priority = str(row.get("priority") or "LOW").upper()
    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(priority, 9)
    return (
        priority_rank,
        str(row.get("date") or ""),
        str(row.get("time_et") or ""),
        str(row.get("event") or ""),
    )


def _is_cache_fresh(payload: Any, ttl: timedelta) -> bool:
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("events"), list):
        return False
    fetched = _parse_datetime(payload.get("fetched_at"))
    if fetched is None:
        return False
    return datetime.now() - fetched <= ttl


def _parse_datetime(value: Any) -> datetime | None:
    try:
        text = str(value or "").strip()
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def _parse_date(value: Any) -> date | None:
    try:
        text = str(value or "").strip()
        return datetime.fromisoformat(text).date() if text else None
    except ValueError:
        return None


def _format_dollars(value: Any) -> str:
    numeric = _safe_int_or_none(value)
    if numeric is None:
        return ""
    amount = float(numeric)
    for suffix, divisor in (("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000)):
        if abs(amount) >= divisor:
            return f"${amount / divisor:.2f}{suffix}"
    return f"${amount:,.0f}"


def _safe_int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
