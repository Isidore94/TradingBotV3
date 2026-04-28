from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from market_prep.cache import read_json_cache, write_json_cache
from market_prep.models import MarketPrepConfig
from market_prep.services.watchlist_service import load_watchlist_entries


SEC_FILINGS_CACHE_FILE_NAME = "sec_filings_cache.json"
SEC_TICKER_CACHE_FILE_NAME = "sec_ticker_cik_cache.json"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"
DEFAULT_FORMS = ["8-K", "10-Q", "10-K", "S-1", "424B", "13D", "13G", "4"]
DEFAULT_DANGER_KEYWORDS = [
    "offering",
    "shelf",
    "at-the-market",
    "ATM",
    "resignation",
    "going concern",
    "material weakness",
    "investigation",
    "subpoena",
    "restatement",
    "impairment",
    "termination",
    "default",
    "bankruptcy",
    "merger",
    "acquisition",
    "strategic alternatives",
]
DEFAULT_SETTINGS = {
    "cache_ttl_hours": 6,
    "days_back": 7,
    "max_watchlist_tickers": 30,
    "max_filings_per_ticker": 8,
    "max_document_chars": 200000,
    "request_delay_seconds": 0.15,
    "request_timeout_seconds": 15,
    "user_agent": "TradingBotV3 market prep contact@example.com",
    "forms": DEFAULT_FORMS,
    "danger_keywords": DEFAULT_DANGER_KEYWORDS,
}
HIGH_KEYWORDS = {
    "offering",
    "shelf",
    "at-the-market",
    "ATM",
    "going concern",
    "material weakness",
    "investigation",
    "subpoena",
    "restatement",
    "default",
    "bankruptcy",
    "strategic alternatives",
}


def get_sec_filing_risk(
    config: MarketPrepConfig,
    *,
    tickers: list[str] | None = None,
    start_date: date | None = None,
    days_back: int | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    end = start_date or datetime.now().date()
    settings = get_sec_filings_settings(config)
    window_days = _safe_int(days_back, _safe_int(settings.get("days_back"), 7))
    start = end - timedelta(days=max(0, window_days))
    cache_path = _cache_path(config)
    generated_at = datetime.now().isoformat(timespec="seconds")

    if not is_sec_filings_enabled(config):
        return _payload(
            generated_at,
            start,
            end,
            [],
            tickers=[],
            status="disabled",
            status_label="Disabled",
            message="SEC filings disabled.",
            cache_path=cache_path,
        )

    requested_tickers = _normalize_tickers(tickers if tickers is not None else _watchlist_tickers(config))
    max_tickers = max(0, _safe_int(settings.get("max_watchlist_tickers"), 30))
    skipped_count = max(0, len(requested_tickers) - max_tickers)
    requested_tickers = requested_tickers[:max_tickers]
    if not requested_tickers:
        return _payload(
            generated_at,
            start,
            end,
            [],
            tickers=[],
            status="empty",
            status_label="No watchlist tickers",
            message="No watchlist tickers found for SEC filing scan.",
            cache_path=cache_path,
        )

    cache_payload = read_json_cache(cache_path, default={})
    ttl = timedelta(hours=max(0.0, _safe_float(settings.get("cache_ttl_hours"), 6.0)))
    if not force_refresh and _is_cache_fresh(cache_payload, ttl) and _same_tickers(cache_payload, requested_tickers):
        filings = _filings_in_window(cache_payload.get("filings"), start, end)
        return _payload(
            generated_at,
            start,
            end,
            filings,
            tickers=requested_tickers,
            status="cache",
            status_label="Loaded from cache",
            message="" if filings else "No SEC filing risk found.",
            skipped_count=skipped_count,
            cache_path=cache_path,
        )

    try:
        cik_map = _load_ticker_cik_map(config, settings)
        filings = _fetch_recent_filings_for_tickers(requested_tickers, cik_map, start, end, settings)
        filings.sort(key=_filing_sort_key)
        write_json_cache(
            cache_path,
            {
                "fetched_at": generated_at,
                "tickers": requested_tickers,
                "filings": filings,
            },
        )
        return _payload(
            generated_at,
            start,
            end,
            filings,
            tickers=requested_tickers,
            status="refreshed",
            status_label="Refreshed",
            message="" if filings else "No SEC filing risk found.",
            skipped_count=skipped_count,
            cache_path=cache_path,
        )
    except Exception as exc:
        logging.getLogger("market_prep").exception("SEC filing enrichment failed.")
        cached_filings = _filings_in_window(cache_payload.get("filings"), start, end)
        if cached_filings:
            return _payload(
                generated_at,
                start,
                end,
                cached_filings,
                tickers=requested_tickers,
                status="cache_fallback",
                status_label="Failed, using cache",
                message="SEC filing scan failed; using cached filings.",
                warnings=[str(exc)],
                skipped_count=skipped_count,
                cache_path=cache_path,
            )
        return _payload(
            generated_at,
            start,
            end,
            [],
            tickers=requested_tickers,
            status="failed",
            status_label="Failed",
            message=f"SEC filing scan unavailable: {exc}",
            warnings=[str(exc)],
            skipped_count=skipped_count,
            cache_path=cache_path,
        )


def is_sec_filings_enabled(config: MarketPrepConfig | None) -> bool:
    if config is None:
        return False
    return bool(config.features.get("sec_filings", False))


def get_sec_filings_settings(config: MarketPrepConfig | None) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if config is not None and isinstance(config.sec_filings, dict):
        settings.update(config.sec_filings)
    forms = settings.get("forms")
    settings["forms"] = [str(item).strip().upper() for item in forms] if isinstance(forms, list) else DEFAULT_FORMS
    keywords = settings.get("danger_keywords")
    settings["danger_keywords"] = (
        [str(item).strip() for item in keywords if str(item).strip()]
        if isinstance(keywords, list)
        else DEFAULT_DANGER_KEYWORDS
    )
    return settings


def _watchlist_tickers(config: MarketPrepConfig) -> list[str]:
    return [str(entry.get("ticker") or "").strip().upper() for entry in load_watchlist_entries(config)]


def _fetch_recent_filings_for_tickers(
    tickers: list[str],
    cik_map: dict[str, dict[str, str]],
    start: date,
    end: date,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    filings: list[dict[str, Any]] = []
    missing: list[str] = []
    delay = max(0.0, _safe_float(settings.get("request_delay_seconds"), 0.15))
    for ticker in tickers:
        cik_entry = cik_map.get(ticker)
        if not cik_entry:
            missing.append(ticker)
            continue
        cik = cik_entry["cik"]
        company = cik_entry.get("company") or ""
        submission = _fetch_json(
            SEC_SUBMISSIONS_URL.format(cik=cik),
            settings=settings,
            sec_data_host=True,
        )
        filings.extend(_normalize_submission_filings(ticker, company, cik, submission, start, end, settings))
        if delay:
            time.sleep(delay)
    if missing:
        logging.getLogger("market_prep").warning("No SEC CIK mapping for tickers: %s", ", ".join(missing[:20]))
    return filings


def _normalize_submission_filings(
    ticker: str,
    company: str,
    cik: str,
    submission: dict[str, Any],
    start: date,
    end: date,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    recent = submission.get("filings", {}).get("recent") if isinstance(submission, dict) else {}
    if not isinstance(recent, dict):
        return []
    forms = recent.get("form") if isinstance(recent.get("form"), list) else []
    filing_dates = recent.get("filingDate") if isinstance(recent.get("filingDate"), list) else []
    accession_numbers = recent.get("accessionNumber") if isinstance(recent.get("accessionNumber"), list) else []
    primary_documents = recent.get("primaryDocument") if isinstance(recent.get("primaryDocument"), list) else []
    items_values = recent.get("items") if isinstance(recent.get("items"), list) else []
    max_rows = max(1, _safe_int(settings.get("max_filings_per_ticker"), 8))
    rows: list[dict[str, Any]] = []
    for index, raw_form in enumerate(forms):
        if len(rows) >= max_rows:
            break
        form = str(raw_form or "").strip().upper()
        filing_date = _parse_date(_list_get(filing_dates, index))
        if filing_date is None or filing_date < start:
            if filing_date is not None and filing_date < start:
                break
            continue
        if filing_date > end or not _form_matches(form, settings.get("forms", DEFAULT_FORMS)):
            continue
        accession = str(_list_get(accession_numbers, index) or "").strip()
        primary_document = str(_list_get(primary_documents, index) or "").strip()
        url = _filing_url(cik, accession, primary_document)
        text_source = " ".join(
            part
            for part in (
                form,
                str(_list_get(items_values, index) or ""),
                accession,
                primary_document,
            )
            if part
        )
        matched_keywords = _matched_keywords(text_source, settings.get("danger_keywords", DEFAULT_DANGER_KEYWORDS))
        document_warning = ""
        if url:
            try:
                document_text = _fetch_text_limited(url, settings=settings)
                matched_keywords = _merge_keywords(
                    matched_keywords,
                    _matched_keywords(document_text, settings.get("danger_keywords", DEFAULT_DANGER_KEYWORDS)),
                )
            except Exception as exc:
                document_warning = f"document scan failed: {exc}"
                logging.getLogger("market_prep").warning(
                    "SEC document scan failed for %s %s %s: %s",
                    ticker,
                    form,
                    accession,
                    exc,
                )
        risk = _risk_classification(form, matched_keywords)
        rows.append(
            {
                "ticker": ticker,
                "company": company,
                "form": form,
                "filing_date": filing_date.isoformat(),
                "matched_keywords": matched_keywords,
                "url": url,
                "risk_classification": risk,
                "source": "SEC EDGAR",
                "accession_number": accession,
                "notes": document_warning,
            }
        )
    return rows


def _load_ticker_cik_map(config: MarketPrepConfig, settings: dict[str, Any]) -> dict[str, dict[str, str]]:
    cache_path = _ticker_cache_path(config)
    cache_payload = read_json_cache(cache_path, default={})
    if _is_cache_fresh(cache_payload, timedelta(days=7)) and isinstance(cache_payload.get("ticker_map"), dict):
        return cache_payload["ticker_map"]
    payload = _fetch_json(SEC_COMPANY_TICKERS_URL, settings=settings, sec_data_host=False)
    ticker_map: dict[str, dict[str, str]] = {}
    rows = payload.values() if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        cik_value = row.get("cik_str")
        if not ticker or cik_value in (None, ""):
            continue
        cik = str(cik_value).strip().zfill(10)
        ticker_map[ticker] = {"cik": cik, "company": str(row.get("title") or "").strip()}
    write_json_cache(
        cache_path,
        {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "ticker_map": ticker_map,
        },
    )
    return ticker_map


def _fetch_json(url: str, *, settings: dict[str, Any], sec_data_host: bool) -> Any:
    request = urllib.request.Request(url, headers=_sec_headers(settings, sec_data_host=sec_data_host))
    timeout = max(1, _safe_int(settings.get("request_timeout_seconds"), 15))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _fetch_text_limited(url: str, *, settings: dict[str, Any]) -> str:
    request = urllib.request.Request(url, headers=_sec_headers(settings, sec_data_host=False))
    timeout = max(1, _safe_int(settings.get("request_timeout_seconds"), 15))
    limit = max(0, _safe_int(settings.get("max_document_chars"), 200000))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if limit <= 0:
            return ""
        return response.read(limit).decode("utf-8", errors="replace")


def _sec_headers(settings: dict[str, Any], *, sec_data_host: bool) -> dict[str, str]:
    user_agent = str(settings.get("user_agent") or "TradingBotV3 market prep contact@example.com").strip()
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "identity",
        "Accept": "application/json" if sec_data_host else "text/html,application/json,*/*",
    }


def _form_matches(form: str, configured_forms: Any) -> bool:
    forms = configured_forms if isinstance(configured_forms, list) else DEFAULT_FORMS
    normalized = str(form or "").strip().upper()
    for raw_value in forms:
        value = str(raw_value or "").strip().upper()
        if not value:
            continue
        if value == "424B" and normalized.startswith("424B"):
            return True
        if value in {"13D", "13G"} and value in normalized:
            return True
        if value == "4" and normalized == "4":
            return True
        if normalized == value or normalized.startswith(value + "/") or normalized.startswith(value + "-"):
            return True
    return False


def _matched_keywords(text: str, keywords: list[str]) -> list[str]:
    lowered = str(text or "").lower()
    matches = []
    for keyword in keywords:
        value = str(keyword or "").strip()
        if not value:
            continue
        if value.lower() in lowered and value not in matches:
            matches.append(value)
    return matches


def _merge_keywords(left: list[str], right: list[str]) -> list[str]:
    merged = list(left)
    for keyword in right:
        if keyword not in merged:
            merged.append(keyword)
    return merged


def _risk_classification(form: str, matched_keywords: list[str]) -> str:
    keyword_set = {keyword for keyword in matched_keywords}
    if any(keyword in keyword_set for keyword in HIGH_KEYWORDS):
        return "HIGH"
    normalized_form = str(form or "").upper()
    if matched_keywords:
        return "MEDIUM"
    if normalized_form in {"8-K", "10-Q", "10-K", "S-1"} or normalized_form.startswith("424B"):
        return "MEDIUM"
    return "LOW"


def _payload(
    generated_at: str,
    start: date,
    end: date,
    filings: list[dict[str, Any]],
    *,
    tickers: list[str],
    status: str,
    status_label: str,
    message: str,
    warnings: list[str] | None = None,
    skipped_count: int = 0,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "source": "SEC EDGAR",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "tickers": tickers,
        "filings": filings,
        "status": status,
        "status_label": status_label,
        "message": message,
        "warnings": warnings or [],
        "skipped_count": skipped_count,
        "cache_path": str(cache_path or ""),
    }


def _cache_path(config: MarketPrepConfig) -> Path:
    cache_dir = config.resolved_paths().get("cache_dir") or Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / SEC_FILINGS_CACHE_FILE_NAME


def _ticker_cache_path(config: MarketPrepConfig) -> Path:
    cache_dir = config.resolved_paths().get("cache_dir") or Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / SEC_TICKER_CACHE_FILE_NAME


def _filings_in_window(rows: Any, start: date, end: date) -> list[dict[str, Any]]:
    filings: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return filings
    for row in rows:
        if not isinstance(row, dict):
            continue
        filing_date = _parse_date(row.get("filing_date"))
        if filing_date is not None and start <= filing_date <= end:
            filings.append(dict(row))
    filings.sort(key=_filing_sort_key)
    return filings


def _filing_sort_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    priority = str(row.get("risk_classification") or "LOW").upper()
    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(priority, 9)
    return (
        priority_rank,
        str(row.get("filing_date") or ""),
        str(row.get("ticker") or ""),
        str(row.get("form") or ""),
    )


def _filing_url(cik: str, accession: str, primary_document: str) -> str:
    if not cik or not accession or not primary_document:
        return ""
    try:
        cik_int = int(str(cik))
    except ValueError:
        return ""
    accession_clean = accession.replace("-", "")
    return SEC_ARCHIVES_URL.format(cik_int=cik_int, accession=accession_clean, document=primary_document)


def _same_tickers(payload: Any, tickers: list[str]) -> bool:
    if not isinstance(payload, dict):
        return False
    cached = _normalize_tickers(payload.get("tickers") if isinstance(payload.get("tickers"), list) else [])
    return cached == tickers


def _normalize_tickers(values: Any) -> list[str]:
    tickers: list[str] = []
    seen = set()
    rows = values if isinstance(values, list) else []
    for value in rows:
        ticker = str(value or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def _is_cache_fresh(payload: Any, ttl: timedelta) -> bool:
    if not isinstance(payload, dict):
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


def _list_get(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else ""


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
