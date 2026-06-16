from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from market_prep.cache import get_default_cache_dir, read_json_cache, write_json_cache
from market_prep.models import MarketPrepConfig


FUTURE_ROADMAP_CACHE_FILE_NAME = "future_roadmap_memory.json"
DEFAULT_FUTURE_ROADMAP_SETTINGS = {
    "days_ahead": 5,
    "recap_lookback_days": 5,
    "max_items_per_day": 8,
    "persist": True,
    "memory_retention_days": 45,
}
PRIORITY_RANK = {"HIGH": 0, "MEGA": 0, "MEDIUM": 1, "LOW": 2}
MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
MAJOR_MACRO_TERMS = (
    "CPI",
    "CORE CPI",
    "PCE",
    "CORE PCE",
    "PPI",
    "NONFARM",
    "NFP",
    "PAYROLL",
    "UNEMPLOYMENT",
    "JOBLESS CLAIMS",
    "ADP",
    "FOMC",
    "RATE DECISION",
    "RETAIL SALES",
    "GDP",
    "ISM",
    "JOLTS",
    "CONSUMER SENTIMENT",
)
MAJOR_NEWS_TERMS = (
    "ipo",
    "initial public offering",
    "starts trading",
    "begin trading",
    "begins trading",
    "listing",
    "listed on",
    "direct listing",
    "debut",
    "tariff deadline",
    "deadline",
    "vote",
    "court ruling",
    "supreme court",
    "opec meeting",
    "export controls",
    "sanctions",
    "rate decision",
    "fomc",
    "cpi",
    "pce",
    "payrolls",
)
LOW_SIGNAL_NEWS_PHRASES = (
    "should you buy",
    "how to buy",
    "what to know",
    "what every investor needs to know",
    "watch before",
    "stock to watch",
    "stocks to watch",
    "opinion",
    "prediction",
    "price target",
)


def get_future_roadmap_settings(config: MarketPrepConfig | None) -> dict[str, Any]:
    settings = dict(DEFAULT_FUTURE_ROADMAP_SETTINGS)
    if config is not None and isinstance(getattr(config, "future_roadmap", None), dict):
        settings.update(config.future_roadmap)
    return settings


def build_future_roadmap(
    *,
    report_date: str | date | datetime | None = None,
    next_7_events: dict[str, Any] | None = None,
    fed_calendar: dict[str, Any] | None = None,
    treasury_calendar: dict[str, Any] | None = None,
    next_7_earnings: dict[str, Any] | None = None,
    watchlist_risk: dict[str, Any] | None = None,
    rss_headlines: dict[str, Any] | None = None,
    recent_economic_events: dict[str, Any] | None = None,
    config: MarketPrepConfig | None = None,
    persist: bool | None = None,
) -> dict[str, Any]:
    settings = get_future_roadmap_settings(config)
    start = _parse_date(report_date) or datetime.now().date()
    days_ahead = max(1, _safe_int(settings.get("days_ahead"), 5))
    end = start + timedelta(days=days_ahead - 1)
    max_items_per_day = max(1, _safe_int(settings.get("max_items_per_day"), 8))
    recap_lookback_days = max(1, _safe_int(settings.get("recap_lookback_days"), 5))
    should_persist = bool(settings.get("persist", True)) if persist is None else bool(persist)
    generated_at = datetime.now().isoformat(timespec="seconds")

    memory = _load_memory(config) if should_persist else _empty_memory()
    items = _collect_upcoming_items(
        start=start,
        end=end,
        next_7_events=next_7_events or {},
        fed_calendar=fed_calendar or {},
        treasury_calendar=treasury_calendar or {},
        next_7_earnings=next_7_earnings or {},
        watchlist_risk=watchlist_risk or {},
        rss_headlines=rss_headlines or {},
    )
    recaps = _build_recaps(
        memory,
        recent_economic_events=recent_economic_events or {},
        report_date=start,
        lookback_days=recap_lookback_days,
    )
    grouped_days = _group_roadmap_days(items, start=start, days_ahead=days_ahead, max_items_per_day=max_items_per_day)

    if should_persist:
        memory = _updated_memory(
            memory,
            current_items=items,
            recent_economic_events=recent_economic_events or {},
            report_date=start,
            generated_at=generated_at,
            retention_days=max(7, _safe_int(settings.get("memory_retention_days"), 45)),
        )
        write_json_cache(_memory_path(config), memory)

    return {
        "generated_at": generated_at,
        "source": "calendar+earnings+major_news+memory",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days_ahead": days_ahead,
        "items": items,
        "days": grouped_days,
        "recaps": recaps,
        "memory_status": "updated" if should_persist else "disabled",
        "cache_path": str(_memory_path(config)) if should_persist else "",
        "message": "" if items or recaps else "No major upcoming roadmap items found.",
    }


def _collect_upcoming_items(
    *,
    start: date,
    end: date,
    next_7_events: dict[str, Any],
    fed_calendar: dict[str, Any],
    treasury_calendar: dict[str, Any],
    next_7_earnings: dict[str, Any],
    watchlist_risk: dict[str, Any],
    rss_headlines: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    _extend_calendar_items(items, next_7_events, start=start, end=end, bucket="Economic")
    _extend_calendar_items(items, fed_calendar, start=start, end=end, bucket="Fed")
    _extend_calendar_items(items, treasury_calendar, start=start, end=end, bucket="Treasury")
    _extend_earnings_items(items, next_7_earnings, watchlist_risk, start=start, end=end)
    _extend_news_items(items, rss_headlines, start=start, end=end)
    return _dedupe_items(items)


def _extend_calendar_items(
    items: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    start: date,
    end: date,
    bucket: str,
) -> None:
    for event in _payload_rows(payload, "events"):
        event_date = _parse_date(event.get("date"))
        if event_date is None or not (start <= event_date <= end):
            continue
        priority = str(event.get("priority") or "").strip().upper() or _priority_from_impact(event)
        if priority not in {"HIGH", "MEDIUM"} and not _is_major_macro_event(event):
            continue
        title = _calendar_title(event)
        item = _roadmap_item(
            date_value=event_date,
            time_value=str(event.get("time_et") or ""),
            bucket=bucket,
            priority=priority or "MEDIUM",
            title=title,
            detail=_event_stats_text(event),
            source=str(event.get("source") or payload.get("source") or bucket),
            url=str(event.get("url") or ""),
            event_type="calendar",
            key=_calendar_key(bucket, event),
            raw=event,
        )
        items.append(item)


def _extend_earnings_items(
    items: list[dict[str, Any]],
    payload: dict[str, Any],
    watchlist_risk: dict[str, Any],
    *,
    start: date,
    end: date,
) -> None:
    watchlist_tickers = {
        str(row.get("ticker") or "").strip().upper()
        for row in _payload_rows(watchlist_risk, "risks")
        if "Earnings" in str(row.get("classification") or "")
    }
    for row in _payload_rows(payload, "earnings"):
        event_date = _parse_date(row.get("date"))
        if event_date is None or not (start <= event_date <= end):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        importance = str(row.get("importance") or "").strip().upper()
        if importance not in {"MEGA", "HIGH", "MEDIUM"} and ticker not in watchlist_tickers:
            continue
        parts = [
            ticker,
            str(row.get("company_yfinance") or row.get("company") or "").strip(),
            str(row.get("time") or "").strip().upper(),
            str(row.get("market_cap_fmt") or "").strip(),
        ]
        items.append(
            _roadmap_item(
                date_value=event_date,
                time_value=_earnings_time(row),
                bucket="Earnings",
                priority=importance or "MEDIUM",
                title=" | ".join(part for part in parts if part),
                detail=str(row.get("notes") or "").strip(),
                source=str(payload.get("source") or row.get("source") or "earnings"),
                url="",
                event_type="earnings",
                key=f"earnings|{event_date.isoformat()}|{ticker}",
                raw=row,
            )
        )


def _extend_news_items(
    items: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    start: date,
    end: date,
) -> None:
    for headline in _payload_rows(payload, "headlines"):
        text = _headline_text(headline)
        if not _is_major_future_news(text):
            continue
        event_date = _headline_event_date(headline, start=start, end=end)
        if event_date is None or not (start <= event_date <= end):
            continue
        tags = _news_event_tags(text)
        title = str(headline.get("title") or "").strip()
        if not title:
            continue
        items.append(
            _roadmap_item(
                date_value=event_date,
                time_value="",
                bucket="Major News",
                priority="HIGH" if "IPO/listing" in tags or "Policy/geopolitics" in tags else "MEDIUM",
                title=title,
                detail=", ".join(tags),
                source=str(headline.get("source") or payload.get("source") or "news"),
                url=str(headline.get("url") or ""),
                event_type="major_news",
                key=f"news|{event_date.isoformat()}|{_normalize_key(title)}",
                raw=headline,
            )
        )


def _build_recaps(
    memory: dict[str, Any],
    *,
    recent_economic_events: dict[str, Any],
    report_date: date,
    lookback_days: int,
) -> list[dict[str, Any]]:
    cutoff = report_date - timedelta(days=lookback_days)
    recent_by_key = {
        _calendar_key("Economic", event): event
        for event in _payload_rows(recent_economic_events, "events")
        if _parse_date(event.get("date")) is not None
    }
    recaps: list[dict[str, Any]] = []
    for stored in _memory_events(memory):
        event_date = _parse_date(stored.get("date"))
        if event_date is None or not (cutoff <= event_date < report_date):
            continue
        key = str(stored.get("event_key") or "")
        recent = recent_by_key.get(key)
        merged = dict(stored)
        if recent:
            merged.update({k: v for k, v in recent.items() if v not in (None, "")})
        actual = str(merged.get("actual") or "").strip()
        if actual:
            recaps.append(_recap_item(stored, merged, status="filled"))
        elif str(stored.get("event_type") or "") == "major_news":
            recaps.append(_recap_item(stored, merged, status="follow_up"))
        elif str(stored.get("priority") or "").upper() in {"HIGH", "MEGA"}:
            recaps.append(_recap_item(stored, merged, status="pending"))
    recaps.sort(key=lambda row: (str(row.get("date") or ""), str(row.get("title") or "")))
    return recaps[:12]


def _recap_item(stored: dict[str, Any], merged: dict[str, Any], *, status: str) -> dict[str, Any]:
    title = str(stored.get("title") or merged.get("event") or "").strip()
    actual = str(merged.get("actual") or "").strip()
    forecast = str(merged.get("forecast") or "").strip()
    previous = str(merged.get("previous") or "").strip()
    if actual:
        detail_parts = [f"Actual {actual}"]
        if forecast:
            detail_parts.append(f"Forecast {forecast}")
        if previous:
            detail_parts.append(f"Previous {previous}")
        detail = "; ".join(detail_parts)
    elif status == "follow_up":
        detail = "Event passed; review post-event market reaction."
    else:
        detail = "Event passed; actual data not available in configured sources yet."
    return {
        "date": str(stored.get("date") or ""),
        "bucket": str(stored.get("bucket") or "Event"),
        "priority": str(stored.get("priority") or ""),
        "title": title,
        "detail": detail,
        "status": status,
        "source": str(merged.get("source") or stored.get("source") or ""),
        "url": str(merged.get("url") or stored.get("url") or ""),
    }


def _updated_memory(
    memory: dict[str, Any],
    *,
    current_items: list[dict[str, Any]],
    recent_economic_events: dict[str, Any],
    report_date: date,
    generated_at: str,
    retention_days: int,
) -> dict[str, Any]:
    by_key = {str(event.get("event_key") or ""): dict(event) for event in _memory_events(memory) if event.get("event_key")}
    for item in current_items:
        key = str(item.get("event_key") or "")
        if not key:
            continue
        existing = by_key.get(key, {})
        merged = {**existing, **item}
        merged["first_seen"] = existing.get("first_seen") or generated_at
        merged["last_seen"] = generated_at
        by_key[key] = merged
    for event in _payload_rows(recent_economic_events, "events"):
        event_date = _parse_date(event.get("date"))
        if event_date is None:
            continue
        key = _calendar_key("Economic", event)
        existing = by_key.get(key)
        if existing is None:
            existing = _roadmap_item(
                date_value=event_date,
                time_value=str(event.get("time_et") or ""),
                bucket="Economic",
                priority=str(event.get("priority") or _priority_from_impact(event) or "MEDIUM"),
                title=_calendar_title(event),
                detail=_event_stats_text(event),
                source=str(event.get("source") or "Economic"),
                url=str(event.get("url") or ""),
                event_type="calendar",
                key=key,
                raw=event,
            )
            existing["first_seen"] = generated_at
        for field in ("actual", "forecast", "previous", "notes", "source", "url"):
            value = event.get(field)
            if value not in (None, ""):
                existing[field] = value
        existing["last_seen"] = generated_at
        by_key[key] = existing
    cutoff = report_date - timedelta(days=retention_days)
    retained = [
        event
        for event in by_key.values()
        if (_parse_date(event.get("date")) or report_date) >= cutoff
    ]
    retained.sort(key=lambda row: (str(row.get("date") or ""), str(row.get("bucket") or ""), str(row.get("title") or "")))
    return {
        "schema_version": 1,
        "updated_at": generated_at,
        "events": retained,
    }


def _group_roadmap_days(
    items: list[dict[str, Any]],
    *,
    start: date,
    days_ahead: int,
    max_items_per_day: int,
) -> list[dict[str, Any]]:
    grouped = []
    for offset in range(days_ahead):
        day = start + timedelta(days=offset)
        day_items = [item for item in items if _parse_date(item.get("date")) == day]
        grouped.append(
            {
                "date": day.isoformat(),
                "label": _day_label(day, start),
                "items": day_items[:max_items_per_day],
                "hidden_count": max(0, len(day_items) - max_items_per_day),
            }
        )
    return grouped


def _roadmap_item(
    *,
    date_value: date,
    time_value: str,
    bucket: str,
    priority: str,
    title: str,
    detail: str,
    source: str,
    url: str,
    event_type: str,
    key: str,
    raw: dict[str, Any],
) -> dict[str, Any]:
    item = {
        "date": date_value.isoformat(),
        "time_et": str(time_value or "").strip(),
        "bucket": str(bucket or "").strip(),
        "priority": str(priority or "").strip().upper(),
        "title": str(title or "").strip(),
        "detail": str(detail or "").strip(),
        "source": str(source or "").strip(),
        "url": str(url or "").strip(),
        "event_type": str(event_type or "").strip(),
        "event_key": str(key or "").strip(),
        "status": "upcoming",
    }
    for field in ("actual", "forecast", "previous", "notes", "currency", "event"):
        value = raw.get(field) if isinstance(raw, dict) else None
        if value not in (None, ""):
            item[field] = value
    return item


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for item in sorted(items, key=_item_sort_key):
        key = str(item.get("event_key") or "").strip() or (
            str(item.get("date") or ""),
            str(item.get("bucket") or ""),
            _normalize_key(item.get("title")),
        )
        if key in seen or not item.get("date") or not item.get("title"):
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _item_sort_key(item: dict[str, Any]) -> tuple[str, int, int, str, str]:
    return (
        str(item.get("date") or "9999-99-99"),
        _time_sort_value(item.get("time_et")),
        PRIORITY_RANK.get(str(item.get("priority") or "LOW").upper(), 9),
        str(item.get("bucket") or ""),
        str(item.get("title") or ""),
    )


def _calendar_key(bucket: str, event: dict[str, Any]) -> str:
    return "|".join(
        [
            "calendar",
            _normalize_key(bucket),
            str(event.get("date") or "").strip(),
            str(event.get("time_et") or "").strip(),
            str(event.get("currency") or "").strip().upper(),
            _normalize_key(event.get("event")),
        ]
    )


def _calendar_title(event: dict[str, Any]) -> str:
    currency = str(event.get("currency") or "").strip().upper()
    name = str(event.get("event") or "").strip()
    return " ".join(part for part in (currency, name) if part)


def _event_stats_text(event: dict[str, Any]) -> str:
    parts = []
    for label, key in (("Actual", "actual"), ("Forecast", "forecast"), ("Previous", "previous")):
        value = str(event.get(key) or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    notes = str(event.get("notes") or "").strip()
    if notes:
        parts.append(notes)
    return "; ".join(parts)


def _headline_text(headline: dict[str, Any]) -> str:
    return " ".join(
        str(value or "")
        for value in (
            headline.get("title"),
            headline.get("summary"),
            headline.get("query"),
            " ".join(str(tag) for tag in (headline.get("tags") or []) if str(tag).strip()),
        )
    ).lower()


def _is_major_future_news(text: str) -> bool:
    if not any(term in text for term in MAJOR_NEWS_TERMS):
        return False
    if any(phrase in text for phrase in LOW_SIGNAL_NEWS_PHRASES) and "ipo" not in text:
        return False
    if "ipo" in text and any(phrase in text for phrase in ("should you buy", "how to buy")):
        return False
    return True


def _news_event_tags(text: str) -> list[str]:
    tags = []
    if "ipo" in text or "initial public offering" in text or "starts trading" in text or "begin trading" in text:
        tags.append("IPO/listing")
    if any(term.lower() in text for term in MAJOR_MACRO_TERMS):
        tags.append("Macro release")
    if any(term in text for term in ("tariff", "deadline", "vote", "court ruling", "supreme court", "export controls", "sanctions")):
        tags.append("Policy/geopolitics")
    if "opec" in text or "oil" in text:
        tags.append("Oil/energy")
    return tags or ["Major news"]


def _headline_event_date(headline: dict[str, Any], *, start: date, end: date) -> date | None:
    text = _headline_text(headline)
    explicit = _explicit_date_in_text(text, start.year)
    if explicit and start <= explicit <= end:
        return explicit
    if "tomorrow" in text:
        return start + timedelta(days=1)
    if "today" in text:
        return start
    weekday = _weekday_date_in_text(text, start=start, end=end)
    if weekday:
        return weekday
    return None


def _explicit_date_in_text(text: str, default_year: int) -> date | None:
    month_names = "|".join(sorted(MONTHS.keys(), key=len, reverse=True))
    pattern = re.compile(
        rf"\b(?P<month>{month_names})\.?\s+(?P<day>\d{{1,2}})(?:,\s*(?P<year>\d{{4}}))?\b",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        month = MONTHS.get(match.group("month").lower().rstrip("."))
        day = int(match.group("day"))
        year = int(match.group("year") or default_year)
        try:
            return date(year, month, day)
        except (TypeError, ValueError):
            continue
    iso_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if iso_match:
        try:
            return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
        except ValueError:
            return None
    return None


def _weekday_date_in_text(text: str, *, start: date, end: date) -> date | None:
    for name, weekday in WEEKDAYS.items():
        if not re.search(rf"\b{name}\b", text):
            continue
        delta = (weekday - start.weekday()) % 7
        candidate = start + timedelta(days=delta)
        if start <= candidate <= end:
            return candidate
    return None


def _day_label(day: date, start: date) -> str:
    if day == start:
        prefix = "Today"
    elif day == start + timedelta(days=1):
        prefix = "Tomorrow"
    else:
        prefix = day.strftime("%A")
    return f"{prefix}, {day.isoformat()}"


def _is_major_macro_event(event: dict[str, Any]) -> bool:
    text = str(event.get("event") or "").upper()
    return any(term in text for term in MAJOR_MACRO_TERMS)


def _priority_from_impact(event: dict[str, Any]) -> str:
    impact = str(event.get("impact") or "").strip().lower()
    if impact == "high":
        return "HIGH"
    if impact == "medium":
        return "MEDIUM"
    return ""


def _earnings_time(row: dict[str, Any]) -> str:
    label = str(row.get("time") or "").strip().upper()
    if label == "BMO":
        return "08:00"
    if label == "AMC":
        return "16:05"
    if re.fullmatch(r"\d{2}:\d{2}", label):
        return label
    return ""


def _payload_rows(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = payload.get(key) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError, IndexError, AttributeError):
        return None


def _time_sort_value(value: Any) -> int:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{2}:\d{2}", text):
        return 24 * 60 + 59
    hour, minute = text.split(":", 1)
    return int(hour) * 60 + int(minute)


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")


def _memory_path(config: MarketPrepConfig | None) -> Path:
    if config is not None:
        cache_dir = config.resolved_paths().get("cache_dir")
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir / FUTURE_ROADMAP_CACHE_FILE_NAME
    return get_default_cache_dir() / FUTURE_ROADMAP_CACHE_FILE_NAME


def _load_memory(config: MarketPrepConfig | None) -> dict[str, Any]:
    payload = read_json_cache(_memory_path(config), default={})
    return payload if isinstance(payload, dict) else _empty_memory()


def _empty_memory() -> dict[str, Any]:
    return {"schema_version": 1, "updated_at": "", "events": []}


def _memory_events(memory: dict[str, Any]) -> list[dict[str, Any]]:
    rows = memory.get("events") if isinstance(memory, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
