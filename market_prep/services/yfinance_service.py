from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from market_prep.cache import get_default_cache_dir, read_json_cache, write_json_cache
from market_prep.models import MarketPrepConfig

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - depends on optional local environment
    yf = None


YFINANCE_MISSING_MESSAGE = "yfinance not installed. Run: pip install yfinance"
METADATA_CACHE_FILE_NAME = "yfinance_metadata_cache.json"
EARNINGS_CACHE_FILE_NAME = "yfinance_earnings_cache.json"
DEFAULT_YFINANCE_SETTINGS = {
    "enabled": True,
    "metadata_cache_ttl_hours": 24,
    "earnings_cache_ttl_hours": 12,
    "request_delay_seconds": 0.25,
    "max_tickers_per_run": 250,
}
IMPORTANCE_RANK = {"MEGA": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}


def get_ticker_metadata(ticker: str, config: MarketPrepConfig | None = None) -> dict[str, Any]:
    payload = get_many_ticker_metadata([ticker], config=config)
    return payload.get("metadata", {}).get(_normalize_ticker(ticker), _empty_metadata(ticker))


def get_many_ticker_metadata(tickers, config: MarketPrepConfig | None = None) -> dict[str, Any]:
    symbols = _normalize_tickers(tickers)
    settings = get_yfinance_settings(config)
    cache_path = get_default_cache_dir() / METADATA_CACHE_FILE_NAME
    cache_payload = read_json_cache(cache_path, default={})
    cache_payload = cache_payload if isinstance(cache_payload, dict) else {}
    cache_entries = cache_payload.get("metadata") if isinstance(cache_payload.get("metadata"), dict) else {}

    if not is_yfinance_metadata_enabled(config):
        return _metadata_payload(symbols, {}, "disabled", "Disabled", cache_path)
    if yf is None:
        logging.getLogger("market_prep").warning(YFINANCE_MISSING_MESSAGE)
        return _metadata_payload(symbols, {}, "missing", YFINANCE_MISSING_MESSAGE, cache_path)

    now = datetime.now()
    ttl = timedelta(hours=max(0.0, _safe_float(settings.get("metadata_cache_ttl_hours"), 24.0)))
    max_tickers = max(0, _safe_int(settings.get("max_tickers_per_run"), 250))
    request_delay = max(0.0, _safe_float(settings.get("request_delay_seconds"), 0.25))
    metadata: dict[str, dict[str, Any]] = {}
    to_fetch: list[str] = []

    for symbol in symbols:
        cached = cache_entries.get(symbol)
        if _is_metadata_entry_fresh(cached, now, ttl):
            metadata[symbol] = dict(cached.get("metadata") or _empty_metadata(symbol))
        else:
            to_fetch.append(symbol)

    limited_fetch = to_fetch[:max_tickers]
    skipped_count = max(0, len(to_fetch) - len(limited_fetch))
    failures = []
    refreshed_count = 0
    for symbol in limited_fetch:
        try:
            row = _fetch_ticker_metadata(symbol)
            metadata[symbol] = row
            cache_entries[symbol] = {
                "fetched_at": now.isoformat(timespec="seconds"),
                "metadata": row,
            }
            refreshed_count += 1
        except Exception as exc:
            logging.getLogger("market_prep").exception("Failed fetching yfinance metadata for %s.", symbol)
            failures.append(f"{symbol}: {exc}")
            metadata[symbol] = _empty_metadata(symbol)
        if request_delay:
            time.sleep(request_delay)

    for symbol in symbols:
        metadata.setdefault(symbol, _empty_metadata(symbol))

    write_json_cache(
        cache_path,
        {
            "generated_at": now.isoformat(timespec="seconds"),
            "source": "yfinance",
            "metadata": cache_entries,
        },
    )
    status = _metadata_status(metadata, refreshed_count, failures, skipped_count)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "source": "yfinance",
        "status": status["status"],
        "status_label": status["status_label"],
        "message": status["message"],
        "metadata": metadata,
        "cache_path": str(cache_path),
        "warnings": failures,
        "skipped_count": skipped_count,
    }


def get_market_cap(ticker: str, config: MarketPrepConfig | None = None) -> int | None:
    metadata = get_ticker_metadata(ticker, config=config)
    return _safe_market_cap(metadata.get("market_cap"))


def format_market_cap(value) -> str:
    market_cap = _safe_market_cap(value)
    if market_cap is None:
        return "Market cap unknown"
    amount = float(market_cap)
    for suffix, divisor in (("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000)):
        if abs(amount) >= divisor:
            return f"${amount / divisor:.2f}{suffix}"
    return f"${amount:,.0f}"


def sort_events_by_market_cap(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=_market_cap_sort_key)


def enrich_events_with_metadata(events: list[dict[str, Any]], config: MarketPrepConfig | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tickers = [event.get("ticker") for event in events if isinstance(event, dict)]
    metadata_payload = get_many_ticker_metadata(tickers, config=config)
    metadata = metadata_payload.get("metadata") if isinstance(metadata_payload, dict) else {}
    enriched = [enrich_event_with_metadata(event, metadata) for event in events if isinstance(event, dict)]
    return sort_events_by_market_cap(enriched), metadata_payload


def enrich_event_with_metadata(event: dict[str, Any], metadata_by_ticker: dict[str, Any]) -> dict[str, Any]:
    ticker = _normalize_ticker(event.get("ticker"))
    metadata = metadata_by_ticker.get(ticker) if isinstance(metadata_by_ticker, dict) else None
    metadata = metadata if isinstance(metadata, dict) else _empty_metadata(ticker)
    enriched = dict(event)
    original_market_cap = _safe_market_cap(enriched.get("market_cap"))
    company_name = str(metadata.get("company_name") or "").strip()
    if company_name and not str(enriched.get("company") or "").strip():
        enriched["company"] = company_name
    elif company_name:
        enriched["company_yfinance"] = company_name
    yfinance_market_cap = _safe_market_cap(metadata.get("market_cap"))
    enriched["market_cap"] = yfinance_market_cap if yfinance_market_cap is not None else original_market_cap
    enriched["market_cap_fmt"] = format_market_cap(enriched.get("market_cap"))
    enriched["sector"] = metadata.get("sector") or ""
    enriched["industry"] = metadata.get("industry") or ""
    enriched["exchange"] = metadata.get("exchange") or ""
    enriched["currency"] = metadata.get("currency") or ""
    if metadata.get("source") == "yfinance" and (yfinance_market_cap is not None or company_name):
        enriched["metadata_source"] = "yfinance"
    else:
        enriched["metadata_source"] = str(enriched.get("metadata_source") or "").strip()
    enriched["market_impact"] = classify_market_cap_importance(enriched.get("market_cap"))
    enriched["importance"] = upgrade_importance(enriched.get("importance"), enriched.get("market_cap"))
    return enriched


def classify_market_cap_importance(value) -> str:
    market_cap = _safe_market_cap(value)
    if market_cap is None:
        return "UNKNOWN"
    if market_cap >= 1_000_000_000_000:
        return "MEGA"
    if market_cap >= 200_000_000_000:
        return "HIGH"
    if market_cap >= 25_000_000_000:
        return "MEDIUM"
    return "LOW"


def upgrade_importance(current: Any, market_cap: Any) -> str:
    current_text = str(current or "LOW").strip().upper()
    if current_text not in IMPORTANCE_RANK:
        current_text = "LOW"
    market_impact = classify_market_cap_importance(market_cap)
    return current_text if IMPORTANCE_RANK[current_text] <= IMPORTANCE_RANK[market_impact] else market_impact


def get_yfinance_earnings_dates(ticker: str, config: MarketPrepConfig | None = None) -> dict[str, Any]:
    settings = get_yfinance_settings(config)
    cache_path = get_default_cache_dir() / EARNINGS_CACHE_FILE_NAME
    symbol = _normalize_ticker(ticker)
    if not is_yfinance_earnings_fallback_enabled(config):
        return _earnings_payload(symbol, [], "disabled", "Disabled", cache_path)
    if yf is None:
        logging.getLogger("market_prep").warning(YFINANCE_MISSING_MESSAGE)
        return _earnings_payload(symbol, [], "missing", YFINANCE_MISSING_MESSAGE, cache_path)

    cache_payload = read_json_cache(cache_path, default={})
    cache_payload = cache_payload if isinstance(cache_payload, dict) else {}
    cache_entries = cache_payload.get("earnings") if isinstance(cache_payload.get("earnings"), dict) else {}
    cached = cache_entries.get(symbol)
    ttl = timedelta(hours=max(0.0, _safe_float(settings.get("earnings_cache_ttl_hours"), 12.0)))
    if _is_earnings_entry_fresh(cached, datetime.now(), ttl):
        return _earnings_payload(symbol, list(cached.get("earnings") or []), "cache", "Loaded earnings from cache", cache_path)

    try:
        ticker_obj = yf.Ticker(symbol)
        raw = getattr(ticker_obj, "get_earnings_dates", None)
        if callable(raw):
            data = raw(limit=12)
        else:
            data = getattr(ticker_obj, "earnings_dates", None)
        rows = _normalize_earnings_dates(symbol, data)
        cache_entries[symbol] = {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "earnings": rows,
        }
        write_json_cache(
            cache_path,
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "yfinance",
                "earnings": cache_entries,
            },
        )
        return _earnings_payload(symbol, rows, "refreshed", "Refreshed earnings fallback", cache_path)
    except Exception as exc:
        logging.getLogger("market_prep").exception("Failed fetching yfinance earnings dates for %s.", symbol)
        return _earnings_payload(symbol, [], "failed", f"yfinance earnings fallback failed: {exc}", cache_path)


def is_yfinance_metadata_enabled(config: MarketPrepConfig | None) -> bool:
    if config is None:
        return True
    return bool(config.features.get("yfinance_metadata", True)) and bool(get_yfinance_settings(config).get("enabled", True))


def is_yfinance_earnings_fallback_enabled(config: MarketPrepConfig | None) -> bool:
    if config is None:
        return False
    return bool(config.features.get("yfinance_earnings_fallback", False)) and bool(get_yfinance_settings(config).get("enabled", True))


def get_yfinance_settings(config: MarketPrepConfig | None) -> dict[str, Any]:
    settings = dict(DEFAULT_YFINANCE_SETTINGS)
    if config is not None and isinstance(config.yfinance, dict):
        settings.update(config.yfinance)
    return settings


def _fetch_ticker_metadata(symbol: str) -> dict[str, Any]:
    ticker_obj = yf.Ticker(_yfinance_lookup_symbol(symbol))
    info = {}
    try:
        info = ticker_obj.get_info()
    except Exception:
        info = getattr(ticker_obj, "info", {}) or {}
    if not isinstance(info, dict):
        info = {}
    company_name = str(info.get("shortName") or info.get("longName") or "").strip()
    return {
        "ticker": symbol,
        "company_name": company_name,
        "sector": str(info.get("sector") or "").strip(),
        "industry": str(info.get("industry") or "").strip(),
        "market_cap": _safe_market_cap(info.get("marketCap")),
        "market_cap_fmt": format_market_cap(info.get("marketCap")),
        "exchange": str(info.get("exchange") or "").strip(),
        "currency": str(info.get("currency") or "").strip(),
        "quoteType": str(info.get("quoteType") or "").strip(),
        "source": "yfinance",
    }


def _yfinance_lookup_symbol(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if "." in text and not text.endswith(".TO"):
        return text.replace(".", "-")
    return text


def _empty_metadata(ticker: Any) -> dict[str, Any]:
    symbol = _normalize_ticker(ticker)
    return {
        "ticker": symbol,
        "company_name": "",
        "sector": "",
        "industry": "",
        "market_cap": None,
        "market_cap_fmt": "Market cap unknown",
        "exchange": "",
        "currency": "",
        "quoteType": "",
        "source": "yfinance",
    }


def _metadata_payload(
    symbols: list[str],
    metadata: dict[str, dict[str, Any]],
    status: str,
    message: str,
    cache_path,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "yfinance",
        "status": status,
        "status_label": message,
        "message": message,
        "metadata": {symbol: metadata.get(symbol, _empty_metadata(symbol)) for symbol in symbols},
        "cache_path": str(cache_path),
        "warnings": [message] if status == "missing" else [],
        "skipped_count": 0,
    }


def _metadata_status(metadata: dict[str, dict[str, Any]], refreshed_count: int, failures: list[str], skipped_count: int) -> dict[str, str]:
    has_known = any(row.get("market_cap") is not None or row.get("company_name") for row in metadata.values())
    if failures and has_known:
        return {
            "status": "partial_failure",
            "status_label": "Partial metadata failure",
            "message": "Partial metadata failure",
        }
    if failures:
        return {
            "status": "failed",
            "status_label": "Partial metadata failure",
            "message": "Partial metadata failure",
        }
    if refreshed_count:
        return {
            "status": "refreshed",
            "status_label": "Refreshed metadata",
            "message": "Refreshed metadata",
        }
    if metadata:
        return {
            "status": "cache",
            "status_label": "Loaded metadata from cache",
            "message": "Loaded metadata from cache",
        }
    if skipped_count:
        return {
            "status": "skipped",
            "status_label": "Skipped metadata due to max_tickers_per_run",
            "message": "Skipped metadata due to max_tickers_per_run",
        }
    return {
        "status": "enabled",
        "status_label": "Enabled",
        "message": "Enabled",
    }


def _market_cap_sort_key(event: dict[str, Any]) -> tuple[int, int, int, str]:
    market_cap = _safe_market_cap(event.get("market_cap"))
    unknown = 1 if market_cap is None else 0
    importance = str(event.get("importance") or "UNKNOWN").upper()
    return (
        unknown,
        -(market_cap or 0),
        IMPORTANCE_RANK.get(importance, 9),
        str(event.get("ticker") or ""),
    )


def _normalize_earnings_dates(symbol: str, data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    rows: list[dict[str, Any]] = []
    try:
        records = data.reset_index().to_dict("records")
    except Exception:
        return []
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_date = record.get("Earnings Date") or record.get("index") or record.get("Date")
        try:
            earnings_date = raw_date.date().isoformat() if hasattr(raw_date, "date") else datetime.fromisoformat(str(raw_date)).date().isoformat()
        except Exception:
            continue
        rows.append(
            {
                "date": earnings_date,
                "time": "",
                "ticker": symbol,
                "company": "",
                "importance": "UNKNOWN",
                "notes": "yfinance earnings fallback",
                "source": "yfinance",
            }
        )
    return rows


def _earnings_payload(symbol: str, rows: list[dict[str, Any]], status: str, message: str, cache_path) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "yfinance",
        "ticker": symbol,
        "earnings": rows,
        "status": status,
        "status_label": message,
        "message": message,
        "cache_path": str(cache_path),
    }


def _is_metadata_entry_fresh(entry: Any, now: datetime, ttl: timedelta) -> bool:
    if not isinstance(entry, dict):
        return False
    try:
        fetched_at = datetime.fromisoformat(str(entry.get("fetched_at") or ""))
    except ValueError:
        return False
    return now - fetched_at <= ttl and isinstance(entry.get("metadata"), dict)


def _is_earnings_entry_fresh(entry: Any, now: datetime, ttl: timedelta) -> bool:
    if not isinstance(entry, dict):
        return False
    try:
        fetched_at = datetime.fromisoformat(str(entry.get("fetched_at") or ""))
    except ValueError:
        return False
    return now - fetched_at <= ttl and isinstance(entry.get("earnings"), list)


def _normalize_tickers(tickers) -> list[str]:
    symbols: list[str] = []
    seen = set()
    for raw_ticker in list(tickers or []):
        symbol = _normalize_ticker(raw_ticker)
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def _normalize_ticker(ticker: Any) -> str:
    return str(ticker or "").strip().upper()


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


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
