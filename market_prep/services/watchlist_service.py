from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from market_prep.config_loader import CONFIG_DIR
from market_prep.models import MarketPrepConfig
from market_prep.services.earnings_service import get_upcoming_earnings, get_watchlist_earnings
from market_prep.services.yfinance_service import (
    format_market_cap,
    get_many_ticker_metadata,
    is_yfinance_earnings_fallback_enabled,
)


NO_WATCHLIST_TICKERS_MESSAGE = "No watchlist tickers found."
SYMPATHY_MAP_FILE = CONFIG_DIR / "sympathy_map.json"
SEVERITY_ORDER = {
    "Earnings Today/Tomorrow": 0,
    "Earnings Within 14 Days": 1,
    "Sympathy Risk": 2,
    "Corporate Event Risk": 3,
    "Macro Event Risk Today": 4,
    "Clean": 5,
}


def scan_watchlist_risk(
    config: MarketPrepConfig,
    *,
    todays_events: dict | None = None,
    today_tomorrow_earnings: dict | None = None,
    upcoming_earnings: dict | None = None,
    start_date: date | None = None,
    days_ahead: int = 14,
) -> dict[str, Any]:
    today = start_date or datetime.now().date()
    entries = load_watchlist_entries(config)
    tickers = [entry["ticker"] for entry in entries]
    today_tomorrow = today_tomorrow_earnings or get_upcoming_earnings(start_date=today, days=1, config=config)
    upcoming = upcoming_earnings or get_upcoming_earnings(start_date=today, days=days_ahead, config=config)
    if tickers and is_yfinance_earnings_fallback_enabled(config):
        watchlist_earnings = get_watchlist_earnings(
            tickers,
            days_ahead=days_ahead,
            start_date=today,
            config=config,
        )
        upcoming = _merge_earnings_payloads(upcoming, watchlist_earnings)
    metadata_payload = get_many_ticker_metadata(tickers, config=config)
    metadata_by_ticker = metadata_payload.get("metadata") if isinstance(metadata_payload, dict) else {}
    today_tomorrow_by_ticker = _earnings_by_ticker(today_tomorrow)
    upcoming_by_ticker = _earnings_by_ticker(upcoming)
    sympathy_by_ticker = _sympathy_risk_by_ticker(upcoming, tickers, today=today, days_ahead=7)
    high_macro_events = _high_priority_events(todays_events or {})
    rows = [
        _classify_watchlist_entry(
            entry,
            today_tomorrow_by_ticker=today_tomorrow_by_ticker,
            upcoming_by_ticker=upcoming_by_ticker,
            sympathy_by_ticker=sympathy_by_ticker,
            high_macro_events=high_macro_events,
            metadata_by_ticker=metadata_by_ticker,
        )
        for entry in entries
    ]
    rows.sort(key=_watchlist_sort_key)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": upcoming.get("source") if isinstance(upcoming, dict) else "manual",
        "start_date": today.isoformat(),
        "end_date": (today + timedelta(days=max(0, int(days_ahead)))).isoformat(),
        "tickers": tickers,
        "risks": rows,
        "sympathy_risks": [row for row in rows if "Sympathy Risk" in str(row.get("classification") or "")],
        "yfinance_status": _combine_yfinance_status(
            _status_from_metadata_payload(metadata_payload),
            upcoming.get("yfinance_status") if isinstance(upcoming, dict) else None,
        ),
        "missing_files": entries.missing_files if isinstance(entries, WatchlistEntries) else [],
        "message": "" if rows else NO_WATCHLIST_TICKERS_MESSAGE,
    }


class WatchlistEntries(list):
    def __init__(self, rows: list[dict[str, Any]], missing_files: list[str]):
        super().__init__(rows)
        self.missing_files = missing_files


def load_watchlist_entries(config: MarketPrepConfig) -> WatchlistEntries:
    resolved_paths = config.resolved_paths()
    source_paths = [
        ("longs", resolved_paths.get("longs_file")),
        ("shorts", resolved_paths.get("shorts_file")),
    ]
    by_ticker: dict[str, dict[str, Any]] = {}
    missing_files: list[str] = []
    for source_name, path in source_paths:
        if path is None:
            missing_files.append(f"{source_name}: <unresolved>")
            continue
        symbols = _read_watchlist_symbols(path)
        if not Path(path).exists():
            missing_files.append(f"{source_name}: {path}")
        for symbol in symbols:
            row = by_ticker.setdefault(symbol, {"ticker": symbol, "source_lists": []})
            if source_name not in row["source_lists"]:
                row["source_lists"].append(source_name)

    rows = sorted(by_ticker.values(), key=lambda row: row["ticker"])
    return WatchlistEntries(rows, missing_files)


def _read_watchlist_symbols(path: Path) -> list[str]:
    target = Path(path)
    if not target.exists():
        return []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    symbols: list[str] = []
    seen = set()
    for raw_line in lines:
        cleaned_line = str(raw_line or "").split("#", 1)[0].replace(",", " ").strip().upper()
        if not cleaned_line or cleaned_line.startswith("SYMBOLS FROM TC2000"):
            continue
        for candidate in cleaned_line.split():
            symbol = candidate.strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def _classify_watchlist_entry(
    entry: dict[str, Any],
    *,
    today_tomorrow_by_ticker: dict[str, list[dict[str, Any]]],
    upcoming_by_ticker: dict[str, list[dict[str, Any]]],
    sympathy_by_ticker: dict[str, list[dict[str, Any]]],
    high_macro_events: list[dict[str, Any]],
    metadata_by_ticker: dict[str, Any],
) -> dict[str, Any]:
    ticker = str(entry.get("ticker") or "").upper()
    source_lists = list(entry.get("source_lists") or [])
    today_tomorrow_rows = today_tomorrow_by_ticker.get(ticker, [])
    upcoming_rows = upcoming_by_ticker.get(ticker, [])
    sympathy_rows = sympathy_by_ticker.get(ticker, [])
    metadata = metadata_by_ticker.get(ticker) if isinstance(metadata_by_ticker, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}

    if today_tomorrow_rows:
        row = today_tomorrow_rows[0]
        classification = "❌ Earnings Today/Tomorrow"
        reason = _earnings_reason(row)
    elif upcoming_rows:
        row = upcoming_rows[0]
        classification = "⚠️ Earnings Within 14 Days"
        reason = _earnings_reason(row)
    elif sympathy_rows:
        row = sympathy_rows[0]
        classification = "Sympathy Risk"
        reason = _sympathy_reason(row)
    elif high_macro_events:
        classification = "⚠️ Macro Event Risk Today"
        reason = "High-priority macro event today: " + ", ".join(
            str(event.get("event") or "").strip()
            for event in high_macro_events
            if str(event.get("event") or "").strip()
        )
    else:
        classification = "✅ Clean"
        reason = "No configured earnings or high-priority macro event risk found."

    company = str(metadata.get("company_name") or "").strip()
    sector = str(metadata.get("sector") or "").strip()
    industry = str(metadata.get("industry") or "").strip()
    market_cap = _safe_market_cap(metadata.get("market_cap"))
    return {
        "ticker": ticker,
        "company": company,
        "sector": sector,
        "industry": industry,
        "market_cap": market_cap,
        "market_cap_fmt": format_market_cap(market_cap),
        "metadata_source": "yfinance" if company or market_cap is not None else "",
        "source_lists": source_lists,
        "source_list": ", ".join(source_lists),
        "classification": classification,
        "reason": reason,
    }


def load_sympathy_map(path: Path = SYMPATHY_MAP_FILE) -> dict[str, list[str]]:
    if not Path(path).exists():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logging.getLogger("market_prep").warning("Failed reading sympathy map at %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for major_ticker, related in payload.items():
        major = str(major_ticker or "").strip().upper()
        if not major or not isinstance(related, list):
            continue
        related_tickers: list[str] = []
        seen = set()
        for value in related:
            ticker = str(value or "").strip().upper()
            if ticker and ticker not in seen:
                seen.add(ticker)
                related_tickers.append(ticker)
        if related_tickers:
            normalized[major] = related_tickers
    return normalized


def _sympathy_risk_by_ticker(
    upcoming_earnings: dict,
    watchlist_tickers: list[str],
    *,
    today: date,
    days_ahead: int,
) -> dict[str, list[dict[str, Any]]]:
    watchlist_set = {str(ticker or "").strip().upper() for ticker in watchlist_tickers if str(ticker or "").strip()}
    if not watchlist_set:
        return {}
    sympathy_map = load_sympathy_map()
    if not sympathy_map:
        return {}
    end = today + timedelta(days=max(0, int(days_ahead)))
    earnings_by_ticker = _earnings_by_ticker(upcoming_earnings)
    risks: dict[str, list[dict[str, Any]]] = {}
    for major_ticker, related_tickers in sympathy_map.items():
        earnings_rows = [
            row for row in earnings_by_ticker.get(major_ticker, [])
            if _is_earnings_row_in_window(row, today, end)
        ]
        if not earnings_rows:
            continue
        source_row = earnings_rows[0]
        for related_ticker in related_tickers:
            if related_ticker not in watchlist_set:
                continue
            risks.setdefault(related_ticker, []).append(
                {
                    "related_ticker": related_ticker,
                    "major_ticker": major_ticker,
                    "major_earnings": source_row,
                }
            )
    for rows in risks.values():
        rows.sort(key=lambda row: _earnings_sort_key(row.get("major_earnings") or {}))
    return risks


def _is_earnings_row_in_window(row: dict[str, Any], start: date, end: date) -> bool:
    event_date = _parse_date(row.get("date"))
    return event_date is not None and start <= event_date <= end


def _sympathy_reason(row: dict[str, Any]) -> str:
    major = str(row.get("major_ticker") or "").strip().upper()
    earnings_row = row.get("major_earnings") if isinstance(row.get("major_earnings"), dict) else {}
    earnings_date = str(earnings_row.get("date") or "").strip()
    time_value = str(earnings_row.get("time") or "").strip()
    company = str(earnings_row.get("company_yfinance") or earnings_row.get("company") or "").strip()
    parts = [major, "earnings", earnings_date, time_value, company]
    return "Related to " + " ".join(part for part in parts if part).strip()


def _earnings_by_ticker(payload: dict) -> dict[str, list[dict[str, Any]]]:
    rows = payload.get("earnings") if isinstance(payload, dict) else []
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(rows, list):
        return by_ticker
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        by_ticker.setdefault(ticker, []).append(row)
    return by_ticker


def _merge_earnings_payloads(base_payload: dict, extra_payload: dict) -> dict[str, Any]:
    merged = dict(base_payload) if isinstance(base_payload, dict) else {}
    rows = _dedupe_earnings_rows(
        list(merged.get("earnings") or []) + list(extra_payload.get("earnings") or [])
        if isinstance(extra_payload, dict)
        else list(merged.get("earnings") or [])
    )
    rows.sort(key=_earnings_sort_key)
    merged["earnings"] = rows
    extra_status = extra_payload.get("yfinance_status") if isinstance(extra_payload, dict) else None
    if extra_status:
        merged["yfinance_status"] = extra_status
    return merged


def _dedupe_earnings_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("date") or "").strip(),
            str(row.get("time") or "").strip().upper(),
            str(row.get("ticker") or "").strip().upper(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _earnings_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    market_cap = _safe_market_cap(row.get("market_cap"))
    return (
        1 if market_cap is None else 0,
        -(market_cap or 0),
        str(row.get("ticker") or ""),
    )


def _high_priority_events(payload: dict) -> list[dict[str, Any]]:
    events = payload.get("events") if isinstance(payload, dict) else []
    if not isinstance(events, list):
        return []
    return [
        event for event in events
        if isinstance(event, dict) and str(event.get("priority") or "").upper() == "HIGH"
    ]


def _earnings_reason(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("ticker") or "").strip().upper(),
        "earnings",
        str(row.get("date") or "").strip(),
        str(row.get("time") or "").strip(),
    ]
    company = str(row.get("company") or "").strip()
    notes = str(row.get("notes") or "").strip()
    detail = " ".join(part for part in parts if part).strip()
    if company:
        detail += f" - {company}"
    if notes:
        detail += f" ({notes})"
    return detail or "Configured earnings risk found."


def _watchlist_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    classification = str(row.get("classification") or "")
    severity = 9
    for marker, rank in SEVERITY_ORDER.items():
        if marker in classification:
            severity = rank
            break
    market_cap = _safe_market_cap(row.get("market_cap"))
    return (
        severity,
        -(market_cap or 0) if market_cap is not None else 0,
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
    labels: list[str] = []
    warnings: list[str] = []
    for status in valid:
        label = str(status.get("status_label") or status.get("message") or status.get("status") or "").strip()
        if label and label not in labels and label != "No metadata requested":
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
        "skipped_count": sum(_safe_int(status.get("skipped_count"), 0) for status in valid),
        "cache_path": primary.get("cache_path") or "",
    }


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
