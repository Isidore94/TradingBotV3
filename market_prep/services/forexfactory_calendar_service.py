from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from market_prep.cache import get_default_cache_dir, read_json_cache, write_json_cache
from market_prep.models import MarketPrepConfig


FOREXFACTORY_CACHE_FILE_NAME = "forexfactory_calendar_cache.json"
SELENIUM_REQUIREMENT_MESSAGE = "ForexFactory scraper requires Selenium/browser driver or compatible parser."
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DEFAULT_FOREXFACTORY_SETTINGS = {
    "enabled": False,
    "base_url": "https://www.forexfactory.com/calendar",
    "currencies": ["USD"],
    "impacts": ["high", "medium"],
    "days_ahead": 14,
    "timezone": "America/New_York",
    "use_selenium": True,
    "request_timeout_seconds": 20,
    "cache_ttl_hours": 6,
}


class ForexFactoryScrapeError(RuntimeError):
    pass


def load_forexfactory_events(
    config: MarketPrepConfig,
    *,
    start_date: date | None = None,
    days_ahead: int | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    settings = get_forexfactory_settings(config)
    cache_path = get_default_cache_dir() / FOREXFACTORY_CACHE_FILE_NAME
    if not is_forexfactory_enabled(config):
        return _status_payload(
            state="disabled",
            message="ForexFactory Disabled",
            cache_path=cache_path,
        )

    window_start = start_date or datetime.now().date()
    window_days = settings["days_ahead"] if days_ahead is None else days_ahead
    window_end = window_start + timedelta(days=max(0, int(window_days)))
    cached_payload = read_json_cache(cache_path, default=None)
    if (
        not force_refresh
        and _is_cache_fresh(cached_payload, settings["cache_ttl_hours"])
        and _cache_covers_window(cached_payload, window_start, window_end)
    ):
        return _filtered_cache_payload(cached_payload, window_start, window_end, cache_path)

    warnings: list[str] = []
    try:
        raw_events, scrape_warnings = _refresh_events(settings)
        warnings.extend(scrape_warnings)
        events = _normalize_events(raw_events, settings, window_start, window_end)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "ForexFactory",
            "start_date": window_start.isoformat(),
            "end_date": window_end.isoformat(),
            "events": events,
            "status": "refreshed",
            "status_label": "Refreshed successfully",
            "message": "ForexFactory Refreshed successfully",
            "warnings": warnings,
            "cache_path": str(cache_path),
        }
        write_json_cache(cache_path, payload)
        return payload
    except Exception as exc:
        logging.getLogger("market_prep").exception("ForexFactory scrape failed.")
        if cached_payload and isinstance(cached_payload, dict):
            fallback = _filtered_cache_payload(cached_payload, window_start, window_end, cache_path)
            fallback["status"] = "cache"
            fallback["status_label"] = "Loaded from cache"
            fallback["message"] = "ForexFactory failed; loaded from cache."
            fallback.setdefault("warnings", []).append(f"ForexFactory failed; using cache. {exc}")
            return fallback
        return _status_payload(
            state="failed",
            message="ForexFactory Failed, using manual fallback",
            cache_path=cache_path,
            warning=f"{SELENIUM_REQUIREMENT_MESSAGE} {exc}",
        )


def is_forexfactory_enabled(config: MarketPrepConfig) -> bool:
    feature_enabled = bool(config.features.get("forexfactory_calendar", False))
    settings_enabled = bool(get_forexfactory_settings(config).get("enabled", False))
    return feature_enabled and settings_enabled


def get_forexfactory_settings(config: MarketPrepConfig) -> dict[str, Any]:
    settings = dict(DEFAULT_FOREXFACTORY_SETTINGS)
    if isinstance(config.forexfactory, dict):
        settings.update(config.forexfactory)
    settings["currencies"] = _normalize_string_list(settings.get("currencies"), default=["USD"], upper=True)
    settings["impacts"] = _normalize_string_list(settings.get("impacts"), default=["high", "medium"], upper=False)
    settings["days_ahead"] = _safe_int(settings.get("days_ahead"), 14)
    settings["request_timeout_seconds"] = _safe_int(settings.get("request_timeout_seconds"), 20)
    settings["cache_ttl_hours"] = _safe_float(settings.get("cache_ttl_hours"), 6.0)
    settings["base_url"] = str(settings.get("base_url") or DEFAULT_FOREXFACTORY_SETTINGS["base_url"])
    settings["timezone"] = str(settings.get("timezone") or "America/New_York")
    settings["use_selenium"] = bool(settings.get("use_selenium", True))
    return settings


def _refresh_events(settings: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    if settings.get("use_selenium"):
        try:
            html = _fetch_calendar_html_with_selenium(settings)
            events = _fetch_events_from_calendar_html(html, settings)
            if events:
                return events, warnings
        except Exception as exc:
            warning = f"{SELENIUM_REQUIREMENT_MESSAGE} Falling back to compatible parser. {exc}"
            warnings.append(warning)
            errors.append(str(exc))

    try:
        events = _fetch_events_with_export_parser(settings)
        if events:
            return events, warnings
    except Exception as exc:
        errors.append(str(exc))

    detail = "; ".join(errors) if errors else "No parser returned events."
    raise ForexFactoryScrapeError(f"{SELENIUM_REQUIREMENT_MESSAGE} {detail}")


def _fetch_calendar_html_with_selenium(settings: dict[str, Any]) -> str:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as exc:
        raise ForexFactoryScrapeError("Selenium is not installed.") from exc

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(f"--user-agent={USER_AGENT}")
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(settings["request_timeout_seconds"])
        driver.get(settings["base_url"])
        driver.implicitly_wait(5)
        return str(driver.page_source or "")
    except Exception as exc:
        raise ForexFactoryScrapeError("Selenium/browser driver could not load ForexFactory.") from exc
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


def _fetch_events_from_calendar_html(html: str, settings: dict[str, Any]) -> list[dict[str, Any]]:
    export_urls = _extract_export_json_urls(html, settings["base_url"])
    events: list[dict[str, Any]] = []
    for url in export_urls:
        events.extend(_fetch_export_json(url, settings["request_timeout_seconds"]))
    return events


def _fetch_events_with_export_parser(settings: dict[str, Any]) -> list[dict[str, Any]]:
    html = _http_get_text(settings["base_url"], settings["request_timeout_seconds"])
    events = _fetch_events_from_calendar_html(html, settings)
    if events:
        return events
    return _fetch_export_json("https://nfs.faireconomy.media/ff_calendar_thisweek.json", settings["request_timeout_seconds"])


def _extract_export_json_urls(html: str, base_url: str) -> list[str]:
    urls: list[str] = []
    seen = set()
    for match in re.findall(r"""href=["']([^"']*ff_calendar_[^"']+?\.json[^"']*)["']""", html, flags=re.I):
        url = urljoin(base_url, match.replace("&amp;", "&"))
        if url not in seen:
            seen.add(url)
            urls.append(url)
    for match in re.findall(r"""https?://[^"' ]*ff_calendar_[^"' ]+?\.json(?:\?[^"' ]*)?""", html, flags=re.I):
        url = match.replace("&amp;", "&")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _fetch_export_json(url: str, timeout_seconds: int) -> list[dict[str, Any]]:
    text = _http_get_text(url, timeout_seconds)
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ForexFactoryScrapeError(f"ForexFactory export was not a list: {url}")
    return [row for row in payload if isinstance(row, dict)]


def _http_get_text(url: str, timeout_seconds: int) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise ForexFactoryScrapeError(f"Request failed for {url}: {exc}") from exc


def _normalize_events(
    raw_events: list[dict[str, Any]],
    settings: dict[str, Any],
    start_date: date,
    end_date: date,
) -> list[dict[str, str]]:
    currencies = set(settings["currencies"])
    impacts = {impact.lower() for impact in settings["impacts"]}
    events = []
    for raw_event in raw_events:
        normalized = _normalize_event(raw_event, settings)
        if normalized is None:
            continue
        event_date = datetime.fromisoformat(normalized["date"]).date()
        impact = normalized["impact"].lower()
        currency = normalized["currency"].upper()
        is_usd_holiday = currency == "USD" and "holiday" in normalized["event"].lower()
        include = (
            start_date <= event_date <= end_date
            and currency in currencies
            and (
                impact in impacts
                or _is_fed_fomc_event(normalized["event"])
                or is_usd_holiday
            )
        )
        if include:
            events.append(normalized)
    events.sort(key=lambda event: (event["date"], event["time_et"], event["event"]))
    return events


def _normalize_event(raw_event: dict[str, Any], settings: dict[str, Any]) -> dict[str, str] | None:
    event_name = str(raw_event.get("title") or raw_event.get("event") or "").strip()
    currency = str(raw_event.get("country") or raw_event.get("currency") or "").strip().upper()
    raw_date = str(raw_event.get("date") or "").strip()
    if not event_name or not currency or not raw_date:
        return None
    try:
        event_dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except ValueError:
        return None
    target_zone = ZoneInfo(str(settings.get("timezone") or "America/New_York"))
    event_dt = event_dt.astimezone(target_zone)
    impact = str(raw_event.get("impact") or "").strip().lower()
    priority = _impact_to_priority(impact)
    return {
        "date": event_dt.date().isoformat(),
        "time_et": event_dt.strftime("%H:%M"),
        "event": event_name,
        "currency": currency,
        "priority": priority,
        "impact": impact or "low",
        "previous": str(raw_event.get("previous") or "").strip(),
        "forecast": str(raw_event.get("forecast") or "").strip(),
        "actual": str(raw_event.get("actual") or "").strip(),
        "source": "ForexFactory",
        "notes": "",
    }


def _impact_to_priority(impact: str) -> str:
    normalized = str(impact or "").strip().lower()
    if "high" in normalized or "red" in normalized:
        return "HIGH"
    if "medium" in normalized or "med" in normalized or "orange" in normalized:
        return "MEDIUM"
    return "LOW"


def _is_fed_fomc_event(event_name: str) -> bool:
    upper = str(event_name or "").upper()
    return "FOMC" in upper or "FED" in upper or "FEDERAL FUNDS" in upper


def _is_cache_fresh(payload: Any, ttl_hours: float) -> bool:
    if not isinstance(payload, dict):
        return False
    generated_at = str(payload.get("generated_at") or "")
    try:
        generated_dt = datetime.fromisoformat(generated_at)
    except ValueError:
        return False
    return datetime.now() - generated_dt <= timedelta(hours=max(0.0, float(ttl_hours)))


def _cache_covers_window(payload: Any, start: date, end: date) -> bool:
    if not isinstance(payload, dict):
        return False
    try:
        cache_start = datetime.fromisoformat(str(payload.get("start_date") or "")).date()
        cache_end = datetime.fromisoformat(str(payload.get("end_date") or "")).date()
    except ValueError:
        return False
    return cache_start <= start and cache_end >= end


def _filtered_cache_payload(payload: dict[str, Any], start: date, end: date, cache_path: Path) -> dict[str, Any]:
    events = []
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        try:
            event_date = datetime.fromisoformat(str(event.get("date") or "")).date()
        except ValueError:
            continue
        if start <= event_date <= end:
            events.append(dict(event))
    return {
        "generated_at": payload.get("generated_at") or datetime.now().isoformat(timespec="seconds"),
        "source": "ForexFactory",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "events": events,
        "status": "cache",
        "status_label": "Loaded from cache",
        "message": "ForexFactory Loaded from cache",
        "warnings": list(payload.get("warnings") or []),
        "cache_path": str(cache_path),
    }


def _status_payload(
    *,
    state: str,
    message: str,
    cache_path: Path,
    warning: str | None = None,
) -> dict[str, Any]:
    warnings = [warning] if warning else []
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "ForexFactory",
        "events": [],
        "status": state,
        "status_label": _status_label(state),
        "message": message,
        "warnings": warnings,
        "cache_path": str(cache_path),
    }


def _status_label(state: str) -> str:
    labels = {
        "disabled": "Disabled",
        "cache": "Loaded from cache",
        "refreshed": "Refreshed successfully",
        "failed": "Failed, using manual fallback",
    }
    return labels.get(state, state)


def _normalize_string_list(value: Any, *, default: list[str], upper: bool) -> list[str]:
    values = value if isinstance(value, list) else default
    normalized: list[str] = []
    seen = set()
    for raw_value in values:
        text = str(raw_value or "").strip()
        text = text.upper() if upper else text.lower()
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized or default


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
