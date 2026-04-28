from __future__ import annotations

import html
import logging
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from market_prep.cache import read_json_cache, write_json_cache
from market_prep.models import MarketPrepConfig


FED_CALENDAR_CACHE_FILE_NAME = "fed_calendar_cache.json"
FED_BASE_URL = "https://www.federalreserve.gov"
FED_MONTH_CALENDAR_URL = FED_BASE_URL + "/newsevents/{year:04d}-{month:02d}.htm"
FED_RSS_FEEDS = (
    ("Federal Reserve Monetary Policy RSS", FED_BASE_URL + "/feeds/press_monetary.xml"),
    ("Federal Reserve Speeches RSS", FED_BASE_URL + "/feeds/speeches.xml"),
    ("Federal Reserve Testimony RSS", FED_BASE_URL + "/feeds/testimony.xml"),
)
DEFAULT_SETTINGS = {
    "cache_ttl_hours": 12,
    "days_ahead": 14,
    "request_timeout_seconds": 15,
}
HIGH_TERMS = ("FOMC", "BEIGE BOOK", "MONETARY POLICY REPORT")
MEDIUM_TERMS = ("SPEECH", "TESTIMONY", "DISCUSSION", "CHAIR", "GOVERNOR", "VICE CHAIR")
EASTERN = ZoneInfo("America/New_York")


def get_fed_calendar_events(
    config: MarketPrepConfig,
    *,
    start_date: date | None = None,
    days_ahead: int | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    start = start_date or datetime.now().date()
    settings = get_fed_calendar_settings(config)
    days = _safe_int(days_ahead, _safe_int(settings.get("days_ahead"), 14))
    end = start + timedelta(days=max(0, days))
    cache_path = _cache_path(config)
    generated_at = datetime.now().isoformat(timespec="seconds")

    if not is_fed_calendar_enabled(config):
        return _payload(
            generated_at,
            start,
            end,
            [],
            status="disabled",
            status_label="Disabled",
            message="Fed calendar disabled.",
            cache_path=cache_path,
        )

    cache_payload = read_json_cache(cache_path, default={})
    ttl = timedelta(hours=max(0.0, _safe_float(settings.get("cache_ttl_hours"), 12.0)))
    if not force_refresh and _is_cache_fresh(cache_payload, ttl):
        events = _events_in_window(cache_payload.get("events"), start, end)
        return _payload(
            generated_at,
            start,
            end,
            events,
            status="cache",
            status_label="Loaded from cache",
            message="" if events else "No Fed calendar events found.",
            cache_path=cache_path,
        )

    try:
        raw_events = _fetch_fed_events(start, end, settings)
        raw_events = _dedupe_events(raw_events)
        raw_events.sort(key=_event_sort_key)
        write_json_cache(
            cache_path,
            {
                "fetched_at": generated_at,
                "source": "Federal Reserve",
                "events": raw_events,
            },
        )
        events = _events_in_window(raw_events, start, end)
        return _payload(
            generated_at,
            start,
            end,
            events,
            status="refreshed",
            status_label="Refreshed",
            message="" if events else "No Fed calendar events found.",
            cache_path=cache_path,
        )
    except Exception as exc:
        logging.getLogger("market_prep").exception("Fed calendar enrichment failed.")
        cached_events = _events_in_window(cache_payload.get("events"), start, end)
        if cached_events:
            return _payload(
                generated_at,
                start,
                end,
                cached_events,
                status="cache_fallback",
                status_label="Failed, using cache",
                message="Fed calendar failed; using cached events.",
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
            message=f"Fed calendar unavailable: {exc}",
            warnings=[str(exc)],
            cache_path=cache_path,
        )


def is_fed_calendar_enabled(config: MarketPrepConfig | None) -> bool:
    if config is None:
        return False
    return bool(config.features.get("fed_calendar", False))


def get_fed_calendar_settings(config: MarketPrepConfig | None) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if config is not None and isinstance(config.fed_calendar, dict):
        settings.update(config.fed_calendar)
    return settings


def _fetch_fed_events(start: date, end: date, settings: dict[str, Any]) -> list[dict[str, Any]]:
    timeout = max(1, _safe_int(settings.get("request_timeout_seconds"), 15))
    events: list[dict[str, Any]] = []
    for year, month in _months_in_window(start, end):
        try:
            events.extend(_parse_month_calendar(year, month, timeout=timeout))
        except Exception as exc:
            logging.getLogger("market_prep").warning(
                "Fed monthly calendar fetch failed for %04d-%02d: %s",
                year,
                month,
                exc,
            )
    for feed_name, feed_url in FED_RSS_FEEDS:
        try:
            events.extend(_parse_rss_feed(feed_name, feed_url, timeout=timeout))
        except Exception as exc:
            logging.getLogger("market_prep").warning("Fed RSS fetch failed for %s: %s", feed_url, exc)
    return events


def _parse_month_calendar(year: int, month: int, *, timeout: int) -> list[dict[str, Any]]:
    url = FED_MONTH_CALENDAR_URL.format(year=year, month=month)
    text = _fetch_text(url, timeout=timeout)
    events: list[dict[str, Any]] = []
    section_pattern = re.compile(
        r'<div class="row cal-nojs__rowTitle">\s*<h4 class="col-md-12">(?P<section>.*?)</h4>\s*</div>(?P<body>.*?)(?=<div class="row cal-nojs__rowTitle">|</main>|</div>\s*</div>\s*</div>\s*<footer)',
        re.IGNORECASE | re.DOTALL,
    )
    for section_match in section_pattern.finditer(text):
        section = _clean_html(section_match.group("section"))
        if not _is_relevant_section(section):
            continue
        body = section_match.group("body")
        panel_pattern = re.compile(
            r'<div class="panel[^"]*.*?<div class="col-xs-2">\s*<p>(?P<time>.*?)</p>\s*</div>\s*'
            r'<div class="col-xs-7"[^>]*>(?P<detail>.*?)</div>\s*'
            r'<div class="col-xs-3">\s*<p>(?P<day>.*?)</p>',
            re.IGNORECASE | re.DOTALL,
        )
        for panel in panel_pattern.finditer(body):
            day_values = _extract_day_values(panel.group("day"))
            if not day_values:
                continue
            detail_html = panel.group("detail")
            title = _first_paragraph_text(detail_html)
            if not title:
                continue
            detail_title = _calendar_detail_title(detail_html)
            event_name = _normalize_calendar_event_name(title, detail_title)
            notes = _calendar_notes(detail_html)
            event_url = _first_href(detail_html)
            for day_value in day_values:
                try:
                    event_date = date(year, month, day_value)
                except ValueError:
                    continue
                events.append(
                    {
                        "date": event_date.isoformat(),
                        "time_et": _normalize_fed_time(panel.group("time")),
                        "event": event_name,
                        "priority": _fed_priority(event_name, notes),
                        "source": "Federal Reserve",
                        "url": event_url,
                        "notes": notes,
                    }
                )
    return events


def _parse_rss_feed(feed_name: str, url: str, *, timeout: int) -> list[dict[str, Any]]:
    text = _fetch_text(url, timeout=timeout)
    root = ET.fromstring(text.lstrip("\ufeff"))
    events: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = _xml_text(item, "title")
        if not _is_relevant_rss_title(title):
            continue
        pub_date = _parse_rss_datetime(_xml_text(item, "pubDate"))
        if pub_date is None:
            continue
        link = _xml_text(item, "link")
        description = _xml_text(item, "description")
        events.append(
            {
                "date": pub_date.date().isoformat(),
                "time_et": pub_date.strftime("%H:%M"),
                "event": _normalize_rss_event_name(title),
                "priority": _fed_priority(title, description),
                "source": "Federal Reserve",
                "url": link,
                "notes": f"{feed_name}: {description}".strip(),
            }
        )
    return events


def _fetch_text(url: str, *, timeout: int) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TradingBotV3 market prep",
            "Accept": "text/html,application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


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
        "source": "Federal Reserve",
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
    return cache_dir / FED_CALENDAR_CACHE_FILE_NAME


def _months_in_window(start: date, end: date) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    current = date(start.year, start.month, 1)
    final = date(end.year, end.month, 1)
    while current <= final:
        months.append((current.year, current.month))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


def _is_relevant_section(section: str) -> bool:
    lowered = section.lower()
    return any(term in lowered for term in ("fomc", "speech", "testimony", "beige book", "board meetings"))


def _is_relevant_rss_title(title: str) -> bool:
    upper = title.upper()
    return any(term in upper for term in HIGH_TERMS + MEDIUM_TERMS + ("MINUTES", "STATEMENT"))


def _normalize_calendar_event_name(title: str, detail_title: str) -> str:
    title = " ".join(title.split())
    detail_title = " ".join(detail_title.split())
    if "FOMC Minutes" in title:
        return "FOMC Minutes"
    if "FOMC Press Conference" in title:
        return "FOMC Press Conference"
    if "FOMC Meeting" in title:
        return "FOMC Meeting"
    if "Beige Book" in title:
        return "Beige Book"
    if detail_title:
        return f"{title}: {detail_title}"
    return title


def _normalize_rss_event_name(title: str) -> str:
    clean = " ".join(str(title or "").split())
    upper = clean.upper()
    if "MINUTES OF THE FEDERAL OPEN MARKET COMMITTEE" in upper:
        return "FOMC Minutes"
    if "FOMC STATEMENT" in upper:
        return "FOMC Statement"
    if "ECONOMIC PROJECTIONS" in upper:
        return "FOMC Economic Projections"
    return clean


def _fed_priority(event_name: str, notes: str = "") -> str:
    upper = f"{event_name} {notes}".upper()
    if any(term in upper for term in ("FOMC", "BEIGE BOOK", "MONETARY POLICY REPORT", "POWELL", "CHAIR ")):
        return "HIGH"
    if any(term in upper for term in MEDIUM_TERMS):
        return "MEDIUM"
    return "LOW"


def _first_paragraph_text(fragment: str) -> str:
    match = re.search(r"<p[^>]*>(.*?)</p>", fragment, re.IGNORECASE | re.DOTALL)
    return _clean_html(match.group(1)) if match else ""


def _calendar_detail_title(fragment: str) -> str:
    match = re.search(r"<p[^>]*class=['\"]calendar__title['\"][^>]*>(.*?)</p>", fragment, re.IGNORECASE | re.DOTALL)
    return _clean_html(match.group(1)) if match else ""


def _calendar_notes(fragment: str) -> str:
    paragraphs = [_clean_html(match.group(1)) for match in re.finditer(r"<p[^>]*>(.*?)</p>", fragment, re.IGNORECASE | re.DOTALL)]
    clean = [item for item in paragraphs[1:] if item and item.lower() != "watch live"]
    return "; ".join(clean[:3])


def _first_href(fragment: str) -> str:
    match = re.search(r'href=["\']([^"\']+)["\']', fragment, re.IGNORECASE)
    if not match:
        return ""
    href = html.unescape(match.group(1)).strip()
    if href.startswith("/"):
        return FED_BASE_URL + href
    return href


def _extract_day_values(value: str) -> list[int]:
    text = _clean_html(value)
    values = []
    for match in re.finditer(r"\b\d{1,2}\b", text):
        try:
            day = int(match.group(0))
        except ValueError:
            continue
        if 1 <= day <= 31 and day not in values:
            values.append(day)
    return values


def _normalize_fed_time(value: str) -> str:
    text = _clean_html(value).lower().replace(".", "")
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*([ap]m)", text)
    if not match:
        return ""
    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    period = match.group(3)
    if period == "pm" and hour != 12:
        hour += 12
    if period == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def _clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    return " ".join(text.split())


def _xml_text(item: ET.Element, name: str) -> str:
    node = item.find(name)
    return html.unescape((node.text or "").strip()) if node is not None else ""


def _parse_rss_datetime(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(str(value or "").strip())
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed.astimezone(EASTERN)


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


def _dedupe_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("date") or "").strip(),
            str(row.get("time_et") or "").strip(),
            str(row.get("event") or "").strip().upper(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


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
