from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import requests

from market_prep.cache import get_default_cache_dir, read_json_cache, write_json_cache
from market_prep.config_loader import CONFIG_DIR
from market_prep.models import MarketPrepConfig
from market_prep.services.yfinance_service import (
    classify_market_cap_importance,
    enrich_events_with_metadata,
    format_market_cap,
    get_yfinance_earnings_dates,
    get_yfinance_settings,
    is_yfinance_earnings_fallback_enabled,
    is_yfinance_metadata_enabled,
)

try:
    from scripts.earnings_history import (
        merge_events as merge_shared_earnings_events,
        normalize_manual_event,
        normalize_nasdaq_event,
        normalize_release_session,
        record_yfinance_rows,
        shared_event_to_market_prep_row,
    )
except ImportError:  # pragma: no cover - used when Market Prep is launched from scripts/
    from earnings_history import (
        merge_events as merge_shared_earnings_events,
        normalize_manual_event,
        normalize_nasdaq_event,
        normalize_release_session,
        record_yfinance_rows,
        shared_event_to_market_prep_row,
    )


MANUAL_EARNINGS_CALENDAR_FILE = CONFIG_DIR / "manual_earnings_calendar.json"
NO_CONFIGURED_EARNINGS_MESSAGE = "No configured earnings found."
EARNINGS_PROVIDER_NAMES = ("manual", "nasdaq", "finnhub", "fmp")
IMPORTANCE_ORDER = {"MEGA": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
NASDAQ_EARNINGS_API_URL = "https://api.nasdaq.com/api/calendar/earnings?date={date}"
NASDAQ_EARNINGS_CACHE_FILE_NAME = "nasdaq_earnings_calendar_cache.json"
NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
}
DEFAULT_EARNINGS_SETTINGS = {
    "provider": "nasdaq",
    "include_manual": True,
    "nasdaq_cache_ttl_hours": 6,
    "request_delay_seconds": 0.15,
    "filter_by_market_cap_and_volume": False,
    "min_market_cap": 1_000_000_000,
    "min_average_volume": 1_000_000,
    "exclude_unknown_market_cap": True,
    "exclude_unknown_average_volume": True,
}


class EarningsProvider(Protocol):
    name: str

    def get_earnings(self, *, start_date: date, end_date: date) -> list[dict[str, Any]]:
        ...


class ManualEarningsProvider:
    name = "manual"

    def __init__(self, path: Path = MANUAL_EARNINGS_CALENDAR_FILE):
        self.path = Path(path)

    def get_earnings(self, *, start_date: date, end_date: date) -> list[dict[str, Any]]:
        rows = load_manual_earnings_calendar(self.path)
        normalized_events = [
            normalize_manual_event(row)
            for row in rows
            if _is_earnings_in_window(row, start_date, end_date)
        ]
        merged_events = merge_shared_earnings_events([row for row in normalized_events if row is not None])
        return [_shared_event_to_earnings_row(event) for event in merged_events]


class NasdaqEarningsProvider:
    name = "nasdaq"

    def __init__(self, *, config: MarketPrepConfig | None = None, include_manual: bool | None = None):
        self.config = config
        settings = get_earnings_settings(config)
        self.include_manual = bool(settings.get("include_manual", True) if include_manual is None else include_manual)

    def get_earnings(self, *, start_date: date, end_date: date) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.include_manual:
            manual_rows = load_manual_earnings_calendar()
            events.extend(
                event
                for event in (normalize_manual_event(row) for row in manual_rows if _is_earnings_in_window(row, start_date, end_date))
                if event is not None
            )

        settings = get_earnings_settings(self.config)
        delay_seconds = max(0.0, _safe_float(settings.get("request_delay_seconds"), 0.15))
        current = start_date
        while current <= end_date:
            raw_rows = fetch_nasdaq_earnings_for_date(current.isoformat(), config=self.config)
            for raw_row in raw_rows:
                event = normalize_nasdaq_event(raw_row, current)
                if event is not None:
                    events.append(event)
            current += timedelta(days=1)
            if delay_seconds and current <= end_date:
                time.sleep(delay_seconds)

        merged_events = merge_shared_earnings_events(events)
        return _dedupe_earnings_rows([_shared_event_to_earnings_row(event) for event in merged_events])


class ProviderStub:
    def __init__(self, name: str):
        self.name = name

    def get_earnings(self, *, start_date: date, end_date: date) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{self.name} earnings provider is not implemented yet.")


def get_upcoming_earnings(
    *,
    start_date: date | None = None,
    days: int = 7,
    provider: EarningsProvider | None = None,
    config: MarketPrepConfig | None = None,
) -> dict[str, Any]:
    start = start_date or datetime.now().date()
    end = start + timedelta(days=max(0, int(days)))
    active_provider = provider or get_configured_earnings_provider(config)
    rows = active_provider.get_earnings(start_date=start, end_date=end)
    rows = _dedupe_earnings_rows(rows)
    rows.sort(key=_earnings_sort_key)
    settings = get_earnings_settings(config)
    filter_enabled = _earnings_filter_enabled(config, settings)
    filter_stats: dict[str, Any] = {}
    if filter_enabled:
        rows, prefilter_stats = _prefilter_earnings_rows_by_market_cap(rows, settings)
        filter_stats.update(prefilter_stats)
    metadata_payload = _empty_yfinance_status(config)
    if config is not None and is_yfinance_metadata_enabled(config):
        rows, metadata_payload = enrich_events_with_metadata(rows, config=config)
        if rows:
            merge_shared_earnings_events(rows)
    if filter_enabled:
        rows, postfilter_stats = _filter_earnings_rows_by_market_cap_and_volume(rows, settings)
        filter_stats = _combine_filter_stats(filter_stats, postfilter_stats)
    rows.sort(key=_earnings_sort_key)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": active_provider.name,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "earnings": rows,
        "filters": filter_stats if filter_enabled else {},
        "yfinance_status": _status_from_metadata_payload(metadata_payload),
        "message": "" if rows else NO_CONFIGURED_EARNINGS_MESSAGE,
    }


def get_earnings_for_today_and_tomorrow(
    target_date: date | None = None,
    *,
    config: MarketPrepConfig | None = None,
) -> dict[str, Any]:
    start = target_date or datetime.now().date()
    return get_upcoming_earnings(start_date=start, days=1, config=config)


def get_earnings_for_date(
    target_date: date | None = None,
    *,
    config: MarketPrepConfig | None = None,
) -> dict[str, Any]:
    target = target_date or datetime.now().date()
    return get_upcoming_earnings(start_date=target, days=0, config=config)


def get_tomorrows_earnings(
    target_date: date | None = None,
    *,
    config: MarketPrepConfig | None = None,
) -> dict[str, Any]:
    start = (target_date or datetime.now().date()) + timedelta(days=1)
    return get_upcoming_earnings(start_date=start, days=0, config=config)


def get_current_week_earnings(
    target_date: date | None = None,
    *,
    config: MarketPrepConfig | None = None,
) -> dict[str, Any]:
    start = _current_week_start(target_date or datetime.now().date())
    return get_upcoming_earnings(start_date=start, days=6, config=config)


def get_watchlist_earnings(
    tickers,
    days_ahead: int = 14,
    *,
    start_date: date | None = None,
    provider: EarningsProvider | None = None,
    config: MarketPrepConfig | None = None,
) -> dict[str, Any]:
    symbols = _normalize_ticker_list(tickers)
    start = start_date or datetime.now().date()
    upcoming = get_upcoming_earnings(start_date=start, days=days_ahead, provider=provider, config=config)
    upcoming_rows = list(upcoming.get("earnings", []))
    fallback_rows, fallback_status = _load_yfinance_earnings_fallback(symbols, start, days_ahead, config)
    if fallback_rows:
        upcoming_rows = _dedupe_earnings_rows(upcoming_rows + fallback_rows)
        if config is not None and is_yfinance_metadata_enabled(config):
            upcoming_rows, metadata_payload = enrich_events_with_metadata(upcoming_rows, config=config)
            upcoming["yfinance_status"] = _combine_yfinance_status(
                upcoming.get("yfinance_status"),
                _status_from_metadata_payload(metadata_payload),
                fallback_status,
            )
        else:
            upcoming["yfinance_status"] = _combine_yfinance_status(upcoming.get("yfinance_status"), fallback_status)
    elif fallback_status:
        upcoming["yfinance_status"] = _combine_yfinance_status(upcoming.get("yfinance_status"), fallback_status)
    symbol_set = set(symbols)
    matching = [
        row for row in upcoming_rows
        if str(row.get("ticker") or "").strip().upper() in symbol_set
    ]
    matching.sort(key=_earnings_sort_key)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": upcoming.get("source") or "manual",
        "start_date": upcoming.get("start_date") or start.isoformat(),
        "end_date": upcoming.get("end_date") or (start + timedelta(days=max(0, int(days_ahead)))).isoformat(),
        "tickers": symbols,
        "earnings": matching,
        "yfinance_status": upcoming.get("yfinance_status") or _empty_yfinance_status(config),
        "message": "" if matching else NO_CONFIGURED_EARNINGS_MESSAGE,
    }


def get_watchlist_earnings_from_config(
    config: MarketPrepConfig,
    *,
    days_ahead: int = 14,
    start_date: date | None = None,
) -> dict[str, Any]:
    tickers = load_watchlist_tickers(config)
    return get_watchlist_earnings(tickers, days_ahead=days_ahead, start_date=start_date, config=config)


def get_earnings_settings(config: MarketPrepConfig | None) -> dict[str, Any]:
    settings = dict(DEFAULT_EARNINGS_SETTINGS)
    if config is not None and isinstance(config.earnings, dict):
        settings.update(config.earnings)
    return settings


def get_configured_earnings_provider(config: MarketPrepConfig | None = None) -> EarningsProvider:
    provider_name = str(get_earnings_settings(config).get("provider") or "nasdaq").strip().lower()
    if provider_name == "manual":
        return ManualEarningsProvider()
    if provider_name == "nasdaq":
        return NasdaqEarningsProvider(config=config)
    return get_provider_stub(provider_name)


def fetch_nasdaq_earnings_for_date(date_str: str, *, config: MarketPrepConfig | None = None) -> list[dict[str, Any]]:
    event_date = _parse_date(date_str)
    if event_date is None:
        return []
    normalized_date = event_date.isoformat()
    settings = get_earnings_settings(config)
    cache_path = get_default_cache_dir() / NASDAQ_EARNINGS_CACHE_FILE_NAME
    cache_payload = read_json_cache(cache_path, default={})
    cache_payload = cache_payload if isinstance(cache_payload, dict) else {}
    cache_dates = cache_payload.get("dates") if isinstance(cache_payload.get("dates"), dict) else {}
    cached_entry = cache_dates.get(normalized_date)
    cached_rows = cached_entry.get("rows", []) if isinstance(cached_entry, dict) else []
    cached_rows = cached_rows if isinstance(cached_rows, list) else []
    if _is_nasdaq_cache_entry_fresh(cached_entry, event_date, settings):
        return cached_rows

    try:
        response = requests.get(
            NASDAQ_EARNINGS_API_URL.format(date=normalized_date),
            headers=NASDAQ_HEADERS,
            timeout=10,
        )
        response.raise_for_status()
        rows = response.json().get("data", {}).get("rows", []) or []
        rows = rows if isinstance(rows, list) else []
        cache_dates[normalized_date] = {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "rows": rows,
        }
        write_json_cache(
            cache_path,
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "nasdaq",
                "dates": cache_dates,
            },
        )
        return rows
    except Exception as exc:
        logging.getLogger("market_prep").warning("Failed fetching Nasdaq earnings for %s: %s", normalized_date, exc)
        return cached_rows


def load_watchlist_tickers(config: MarketPrepConfig) -> list[str]:
    resolved_paths = config.resolved_paths()
    paths = [
        resolved_paths.get("longs_file"),
        resolved_paths.get("shorts_file"),
    ]
    symbols: list[str] = []
    seen = set()
    for path in paths:
        for symbol in _read_watchlist_file(path):
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    return symbols


def load_manual_earnings_calendar(path: Path = MANUAL_EARNINGS_CALENDAR_FILE) -> list[dict[str, Any]]:
    ensure_manual_earnings_calendar(path)
    payload = _read_json(path, default=[])
    if not isinstance(payload, list):
        logging.warning("Manual earnings calendar at %s is not a list.", path)
        return []
    return [row for row in payload if isinstance(row, dict)]


def ensure_manual_earnings_calendar(path: Path = MANUAL_EARNINGS_CALENDAR_FILE) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return False
    _write_json_atomic(target, [])
    return True


def get_provider_stub(name: str) -> ProviderStub:
    normalized = str(name or "").strip().lower()
    if normalized not in EARNINGS_PROVIDER_NAMES:
        raise ValueError(f"Unknown earnings provider: {name}")
    if normalized == "manual":
        raise ValueError("The manual earnings provider is implemented; use ManualEarningsProvider.")
    return ProviderStub(normalized)


def _normalize_earnings_row(row: dict[str, Any]) -> dict[str, Any] | None:
    earnings_date = _parse_date(row.get("date"))
    ticker = str(row.get("ticker") or "").strip().upper()
    if earnings_date is None or not ticker:
        return None
    normalized = {
        "date": earnings_date.isoformat(),
        "time": str(row.get("time") or "").strip().upper(),
        "ticker": ticker,
        "company": str(row.get("company") or "").strip(),
        "importance": str(row.get("importance") or "").strip().upper() or "LOW",
        "notes": str(row.get("notes") or "").strip(),
        "source": str(row.get("source") or "manual").strip() or "manual",
    }
    for key in (
        "market_cap",
        "market_cap_fmt",
        "market_cap_source",
        "average_volume",
        "average_volume_10d",
        "regular_market_volume",
        "sector",
        "industry",
        "metadata_source",
        "market_impact",
        "eps_forecast",
        "estimate_count",
        "fiscal_quarter_ending",
        "last_year_report_date",
        "last_year_eps",
    ):
        if key in row:
            normalized[key] = row.get(key)
    return normalized


def _normalize_nasdaq_earnings_row(row: dict[str, Any], event_date: date) -> dict[str, Any] | None:
    event = normalize_nasdaq_event(row, event_date)
    return _shared_event_to_earnings_row(event) if event is not None else None


def _normalize_nasdaq_release_time(value: Any) -> str:
    return normalize_release_session(value)


def _shared_event_to_earnings_row(event: dict[str, Any]) -> dict[str, Any]:
    row = shared_event_to_market_prep_row(event)
    market_cap = _safe_market_cap(row.get("market_cap"))
    market_impact = classify_market_cap_importance(market_cap)
    row.update(
        {
            "importance": market_impact if market_impact != "UNKNOWN" else "LOW",
            "notes": _earnings_notes_from_normalized_row(row),
            "market_cap": market_cap,
            "market_cap_fmt": format_market_cap(market_cap),
            "market_cap_source": row.get("source") if market_cap is not None else "",
            "metadata_source": row.get("source") if market_cap is not None else "",
            "market_impact": market_impact,
        }
    )
    return row


def _earnings_notes_from_normalized_row(row: dict[str, Any]) -> str:
    parts = []
    eps = str(row.get("eps_forecast") or "").strip()
    estimates = str(row.get("estimate_count") or "").strip()
    quarter = str(row.get("fiscal_quarter_ending") or "").strip()
    if eps:
        parts.append(f"EPS est {eps}")
    if estimates:
        parts.append(f"{estimates} ests")
    if quarter:
        parts.append(f"quarter {quarter}")
    return "; ".join(parts)


def _nasdaq_earnings_notes(row: dict[str, Any]) -> str:
    parts = []
    eps = str(row.get("epsForecast") or "").strip()
    estimates = str(row.get("noOfEsts") or "").strip()
    quarter = str(row.get("fiscalQuarterEnding") or "").strip()
    if eps:
        parts.append(f"EPS est {eps}")
    if estimates:
        parts.append(f"{estimates} ests")
    if quarter:
        parts.append(f"quarter {quarter}")
    return "; ".join(parts)


def _parse_market_cap(value: Any) -> int | None:
    direct = _safe_market_cap(value)
    if direct is not None:
        return direct
    text = str(value or "").strip()
    if not text or text.upper() in {"N/A", "NA", "--", "NONE"}:
        return None
    cleaned = text.replace("$", "").replace(",", "").strip().upper()
    multiplier = 1.0
    if cleaned.endswith("T"):
        multiplier = 1_000_000_000_000.0
        cleaned = cleaned[:-1]
    elif cleaned.endswith("B"):
        multiplier = 1_000_000_000.0
        cleaned = cleaned[:-1]
    elif cleaned.endswith("M"):
        multiplier = 1_000_000.0
        cleaned = cleaned[:-1]
    try:
        numeric = int(float(cleaned) * multiplier)
    except ValueError:
        return None
    return numeric if numeric >= 0 else None


def _is_nasdaq_cache_entry_fresh(entry: Any, event_date: date, settings: dict[str, Any]) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("rows"), list):
        return False
    try:
        fetched_at = datetime.fromisoformat(str(entry.get("fetched_at") or ""))
    except ValueError:
        return False
    today = datetime.now().date()
    if event_date < today:
        return True
    ttl = timedelta(hours=max(0.0, _safe_float(settings.get("nasdaq_cache_ttl_hours"), 6.0)))
    if ttl.total_seconds() <= 0:
        return False
    return datetime.now() - fetched_at <= ttl


def _is_earnings_in_window(row: dict[str, Any], start: date, end: date) -> bool:
    earnings_date = _parse_date(row.get("date"))
    return earnings_date is not None and start <= earnings_date <= end


def _read_watchlist_file(path: Path | None) -> list[str]:
    if path is None or not Path(path).exists():
        return []
    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    return _normalize_ticker_list(raw_text.replace(",", "\n").splitlines())


def _normalize_ticker_list(tickers) -> list[str]:
    symbols: list[str] = []
    seen = set()
    if isinstance(tickers, str):
        values = tickers.replace(",", "\n").splitlines()
    else:
        values = list(tickers or [])
    for raw_value in values:
        symbol = str(raw_value or "").split("#", 1)[0].strip().upper()
        if not symbol or symbol.startswith("SYMBOLS FROM TC2000") or symbol in seen:
            continue
        for part in symbol.replace(",", "\n").splitlines():
            for candidate in part.split():
                cleaned = candidate.strip().upper()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                symbols.append(cleaned)
    return symbols


def _earnings_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str, str, str]:
    market_cap = _safe_market_cap(row.get("market_cap"))
    importance = str(row.get("importance") or "UNKNOWN").upper()
    return (
        1 if market_cap is None else 0,
        -(market_cap or 0),
        IMPORTANCE_ORDER.get(importance, 9),
        str(row.get("date") or ""),
        _time_sort_key(row.get("time")),
        str(row.get("ticker") or ""),
    )


def _safe_market_cap(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        numeric = int(float(value))
        return numeric if numeric >= 0 else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_int_optional(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        numeric = int(float(value))
        return numeric if numeric >= 0 else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _empty_yfinance_status(config: MarketPrepConfig | None) -> dict[str, Any]:
    if config is not None and not is_yfinance_metadata_enabled(config):
        return {"status": "disabled", "status_label": "Disabled", "message": "Disabled"}
    return {"status": "enabled", "status_label": "Enabled", "message": "Enabled"}


def _status_from_metadata_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "empty", "status_label": "No metadata requested", "message": "No metadata requested"}
    return {
        "status": payload.get("status") or "empty",
        "status_label": payload.get("status_label") or payload.get("message") or "No metadata requested",
        "message": payload.get("message") or payload.get("status_label") or "",
        "warnings": payload.get("warnings") if isinstance(payload.get("warnings"), list) else [],
        "skipped_count": payload.get("skipped_count") or 0,
        "cache_path": payload.get("cache_path") or "",
    }


def _combine_yfinance_status(*statuses: dict[str, Any] | None) -> dict[str, Any]:
    valid = [status for status in statuses if isinstance(status, dict)]
    if not valid:
        return {"status": "empty", "status_label": "No metadata requested", "message": "No metadata requested"}
    warnings: list[str] = []
    labels: list[str] = []
    for status in valid:
        label = str(status.get("status_label") or status.get("message") or status.get("status") or "").strip()
        if label and label not in labels:
            labels.append(label)
        status_warnings = status.get("warnings")
        if isinstance(status_warnings, list):
            warnings.extend(str(item) for item in status_warnings if str(item).strip())
    primary = valid[0]
    return {
        "status": primary.get("status") or "empty",
        "status_label": "; ".join(labels) if labels else primary.get("status_label") or "No metadata requested",
        "message": "; ".join(labels) if labels else primary.get("message") or "",
        "warnings": warnings,
        "skipped_count": sum(int(status.get("skipped_count") or 0) for status in valid),
        "cache_path": primary.get("cache_path") or "",
    }


def _load_yfinance_earnings_fallback(
    symbols: list[str],
    start: date,
    days_ahead: int,
    config: MarketPrepConfig | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if config is None or not is_yfinance_earnings_fallback_enabled(config) or not symbols:
        return [], None
    settings = get_yfinance_settings(config)
    max_tickers = max(0, _safe_int(settings.get("max_tickers_per_run"), 250))
    delay_seconds = max(0.0, _safe_float(settings.get("request_delay_seconds"), 0.25))
    end = start + timedelta(days=max(0, int(days_ahead)))
    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for symbol in symbols[:max_tickers]:
        payload = get_yfinance_earnings_dates(symbol, config=config)
        statuses.append(_status_from_metadata_payload(payload))
        for row in payload.get("earnings", []) if isinstance(payload, dict) else []:
            if not isinstance(row, dict):
                continue
            earnings_date = _parse_date(row.get("date"))
            if earnings_date is not None and start <= earnings_date <= end:
                normalized = _normalize_earnings_row(row)
                if normalized is not None:
                    rows.append(normalized)
        if delay_seconds:
            time.sleep(delay_seconds)
    if len(symbols) > max_tickers:
        statuses.append(
            {
                "status": "skipped",
                "status_label": "Skipped yfinance earnings fallback due to max_tickers_per_run",
                "message": "Skipped yfinance earnings fallback due to max_tickers_per_run",
                "skipped_count": len(symbols) - max_tickers,
            }
        )
    if rows:
        record_yfinance_rows(rows)
    return rows, _combine_yfinance_status(*statuses)


def _earnings_filter_enabled(config: MarketPrepConfig | None, settings: dict[str, Any]) -> bool:
    return config is not None and bool(settings.get("filter_by_market_cap_and_volume", False))


def _prefilter_earnings_rows_by_market_cap(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_market_cap = max(0, _safe_int(settings.get("min_market_cap"), 0))
    if min_market_cap <= 0:
        return rows, _base_earnings_filter_stats(settings, prefilter_removed=0, removed=0)
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        market_cap = _safe_market_cap(row.get("market_cap"))
        if market_cap is not None and market_cap < min_market_cap:
            removed += 1
            continue
        kept.append(row)
    return kept, _base_earnings_filter_stats(settings, prefilter_removed=removed, removed=0)


def _filter_earnings_rows_by_market_cap_and_volume(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_market_cap = max(0, _safe_int(settings.get("min_market_cap"), 1_000_000_000))
    min_average_volume = max(0, _safe_int(settings.get("min_average_volume"), 1_000_000))
    exclude_unknown_market_cap = bool(settings.get("exclude_unknown_market_cap", True))
    exclude_unknown_average_volume = bool(settings.get("exclude_unknown_average_volume", True))
    kept: list[dict[str, Any]] = []
    removed = 0
    removed_unknown = 0
    for row in rows:
        market_cap = _safe_market_cap(row.get("market_cap"))
        average_volume = _average_volume_from_row(row)
        missing_market_cap = market_cap is None
        missing_volume = average_volume is None
        if (
            (missing_market_cap and exclude_unknown_market_cap)
            or (missing_volume and exclude_unknown_average_volume)
        ):
            removed += 1
            removed_unknown += 1
            continue
        if market_cap is not None and market_cap < min_market_cap:
            removed += 1
            continue
        if average_volume is not None and average_volume < min_average_volume:
            removed += 1
            continue
        kept.append(row)
    return kept, _base_earnings_filter_stats(
        settings,
        prefilter_removed=0,
        removed=removed,
        removed_unknown=removed_unknown,
    )


def _base_earnings_filter_stats(
    settings: dict[str, Any],
    *,
    prefilter_removed: int,
    removed: int,
    removed_unknown: int = 0,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "min_market_cap": max(0, _safe_int(settings.get("min_market_cap"), 1_000_000_000)),
        "min_average_volume": max(0, _safe_int(settings.get("min_average_volume"), 1_000_000)),
        "prefilter_removed_count": int(prefilter_removed),
        "removed_count": int(removed),
        "removed_unknown_count": int(removed_unknown),
    }


def _combine_filter_stats(*stats_items: dict[str, Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for stats in stats_items:
        if not isinstance(stats, dict):
            continue
        combined.update({key: value for key, value in stats.items() if key not in {"prefilter_removed_count", "removed_count", "removed_unknown_count"}})
        for key in ("prefilter_removed_count", "removed_count", "removed_unknown_count"):
            combined[key] = int(combined.get(key) or 0) + int(stats.get(key) or 0)
    return combined


def _average_volume_from_row(row: dict[str, Any]) -> int | None:
    for key in ("average_volume", "avg_daily_volume", "average_volume_10d", "regular_market_volume"):
        value = _safe_int_optional(row.get(key))
        if value is not None:
            return value
    return None


def _dedupe_earnings_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("date") or "").strip(),
            str(row.get("time") or "").strip().upper(),
            str(row.get("ticker") or "").strip().upper(),
        )
        if key not in by_key:
            by_key[key] = row
            continue
        existing = by_key[key]
        merged = dict(row)
        merged.update({field: value for field, value in existing.items() if value not in (None, "")})
        by_key[key] = merged
    return list(by_key.values())


def _time_sort_key(value: Any) -> str:
    text = str(value or "").strip().upper()
    order = {"BMO": "00", "PRE": "00", "AMC": "99", "POST": "99"}
    return order.get(text, text)


def _parse_date(value: Any) -> date | None:
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


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Failed reading earnings JSON at %s: %s", path, exc)
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
