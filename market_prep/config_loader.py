from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import MarketPrepConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "market_prep_config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "timezone": "America/Vancouver",
    "market_timezone": "America/New_York",
    "api_keys": {
        "fred": "",
        "finnhub": "",
        "newsapi": "",
        "youtube": "",
        "openai": "",
        "polygon": "",
    },
    "features": {
        "earnings": True,
        "economic_calendar": True,
        "news": False,
        "youtube": False,
        "forexfactory_calendar": False,
        "yfinance_metadata": True,
        "yfinance_earnings_fallback": False,
        "fed_calendar": False,
        "treasury_calendar": False,
        "social": False,
        "options_flow": False,
        "sec_filings": False,
        "llm_summary": True,
    },
    "earnings": {
        "provider": "nasdaq",
        "include_manual": True,
        "nasdaq_cache_ttl_hours": 6,
        "nasdaq_today_cache_ttl_minutes": 30,
        "nasdaq_future_cache_ttl_hours": 2,
        "request_delay_seconds": 0.15,
        "request_timeout_seconds": 10,
        "request_retries": 2,
        "request_retry_backoff_seconds": 0.5,
    },
    "forexfactory": {
        "enabled": False,
        "base_url": "https://www.forexfactory.com/calendar",
        "currencies": ["USD"],
        "impacts": ["high", "medium"],
        "days_ahead": 14,
        "timezone": "America/New_York",
        "use_selenium": True,
        "request_timeout_seconds": 20,
        "cache_ttl_hours": 6,
    },
    "yfinance": {
        "enabled": True,
        "metadata_cache_ttl_hours": 24,
        "earnings_cache_ttl_hours": 12,
        "request_delay_seconds": 0.25,
        "max_tickers_per_run": 250,
    },
    "fed_calendar": {
        "cache_ttl_hours": 12,
        "days_ahead": 14,
        "request_timeout_seconds": 15,
    },
    "treasury_calendar": {
        "cache_ttl_hours": 24,
        "days_ahead": 14,
        "important_auctions": ["2-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"],
        "rates_market_driver": True,
        "request_timeout_seconds": 15,
    },
    "sec_filings": {
        "cache_ttl_hours": 6,
        "days_back": 7,
        "max_watchlist_tickers": 30,
        "max_filings_per_ticker": 8,
        "max_document_chars": 200000,
        "request_delay_seconds": 0.15,
        "request_timeout_seconds": 15,
        "user_agent": "TradingBotV3 market prep contact@example.com",
        "forms": ["8-K", "10-Q", "10-K", "S-1", "424B", "13D", "13G", "4"],
        "danger_keywords": [
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
        ],
    },
    "google_news_rss": {
        "enabled": False,
        "cache_ttl_hours": 3,
        "max_watchlist_tickers": 30,
        "max_feeds": 80,
        "request_timeout_seconds": 10,
        "queries": [
            "{ticker} earnings",
            "{ticker} guidance",
            "{ticker} downgrade",
            "{ticker} lawsuit",
            "{ticker} acquisition",
            "Federal Reserve inflation",
            "Treasury auction yields",
            "semiconductor export controls",
            "oil prices Middle East",
        ],
    },
    "llm_summary": {
        "model": "gpt-5-mini",
        "max_output_tokens": 800,
        "headline_limit": 20,
        "article_limit": 4,
        "article_char_limit": 2000,
        "request_timeout_seconds": 45,
        "article_timeout_seconds": 8,
        "reasoning_effort": "low",
        "text_verbosity": "low",
        "user_context": "",
    },
    "ticker_lookup": {
        "days_ahead": 10,
        "news_limit": 40,
        "max_peer_tickers": 8,
        "include_sec_filings": True,
        "include_peer_earnings": True,
        "include_industry_news": True,
        "queries": [
            "{ticker} earnings",
            "{ticker} guidance",
            "{ticker} analyst rating",
            "{ticker} price target",
            "{ticker} upgrade downgrade",
            "{ticker} investor day",
            "{ticker} conference",
            "{ticker} presentation",
            "{ticker} catalyst",
            "{ticker} announces",
            "{ticker} partnership",
            "{ticker} product launch",
            "{ticker} acquisition",
            "{ticker} contract",
            "{ticker} shipment",
            "{ticker} demand",
            "{ticker} regulation",
            "{ticker} lawsuit",
            "{ticker} offering",
            "{ticker} insider selling",
        ],
    },
    "market_prep_ai": {
        "enabled": True,
        "model": "gpt-5.2",
        "timeout_seconds": 30,
        "max_context_chars": 12000,
    },
    "paths": {
        "longs_file": "longs.txt",
        "shorts_file": "shorts.txt",
        "output_dir": "output",
        "cache_dir": "data/cache",
        "log_dir": "logs",
    },
}


def load_market_prep_config(config_path: Path | None = None) -> MarketPrepConfig:
    path = Path(config_path) if config_path else CONFIG_FILE
    created = ensure_default_config(path)
    payload = _read_json(path, default=copy.deepcopy(DEFAULT_CONFIG))
    if not isinstance(payload, dict):
        logging.warning("Market Prep config at %s is not an object; using defaults.", path)
        payload = copy.deepcopy(DEFAULT_CONFIG)
    payload = _merge_defaults(payload)
    config = MarketPrepConfig.from_mapping(payload, config_path=path, repo_root=REPO_ROOT)
    ensure_market_prep_folders(config)
    if created:
        logging.info("Created default Market Prep config at %s", path)
    return config


def set_forexfactory_enabled(enabled: bool, config_path: Path | None = None) -> MarketPrepConfig:
    path = Path(config_path) if config_path else CONFIG_FILE
    ensure_default_config(path)
    payload = _read_json(path, default=copy.deepcopy(DEFAULT_CONFIG))
    if not isinstance(payload, dict):
        payload = copy.deepcopy(DEFAULT_CONFIG)
    payload = _merge_defaults(payload)
    features = payload.setdefault("features", {})
    if not isinstance(features, dict):
        features = {}
        payload["features"] = features
    forexfactory = payload.setdefault("forexfactory", {})
    if not isinstance(forexfactory, dict):
        forexfactory = {}
        payload["forexfactory"] = forexfactory
    features["forexfactory_calendar"] = bool(enabled)
    forexfactory["enabled"] = bool(enabled)
    _write_json_atomic(path, payload)
    return load_market_prep_config(path)


def save_llm_summary_settings(settings: dict[str, Any], config_path: Path | None = None) -> MarketPrepConfig:
    path = Path(config_path) if config_path else CONFIG_FILE
    ensure_default_config(path)
    payload = _read_json(path, default=copy.deepcopy(DEFAULT_CONFIG))
    if not isinstance(payload, dict):
        payload = copy.deepcopy(DEFAULT_CONFIG)
    payload = _merge_defaults(payload)
    llm_summary = payload.setdefault("llm_summary", {})
    if not isinstance(llm_summary, dict):
        llm_summary = {}
        payload["llm_summary"] = llm_summary
    for key, value in settings.items():
        if key in DEFAULT_CONFIG["llm_summary"]:
            llm_summary[str(key)] = value
    features = payload.setdefault("features", {})
    if isinstance(features, dict):
        features["llm_summary"] = True
    _write_json_atomic(path, payload)
    return load_market_prep_config(path)


def get_market_prep_openai_api_key(config: MarketPrepConfig | None = None) -> str:
    env_value = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_value:
        return env_value
    local_value = str(_load_local_secret("openai_api_key") or "").strip()
    if local_value:
        return local_value
    if config is not None:
        return str(config.api_keys.get("openai") or "").strip()
    return ""


def get_market_prep_openai_key_source(config: MarketPrepConfig | None = None) -> str:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "environment"
    if str(_load_local_secret("openai_api_key") or "").strip():
        return "local_secret"
    if config is not None and str(config.api_keys.get("openai") or "").strip():
        return "config"
    return "missing"


def save_market_prep_openai_api_key(api_key: str) -> Path:
    return _save_local_secret("openai_api_key", str(api_key or "").strip())


def ensure_default_config(config_path: Path = CONFIG_FILE) -> bool:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    _write_json_atomic(path, DEFAULT_CONFIG)
    return True


def ensure_market_prep_folders(config: MarketPrepConfig) -> None:
    for key in ("output_dir", "cache_dir", "log_dir"):
        folder = config.resolved_paths().get(key)
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)


def _merge_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.warning("Failed parsing Market Prep config JSON at %s: %s", path, exc)
        return default
    except OSError as exc:
        logging.warning("Failed reading Market Prep config at %s: %s", path, exc)
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


def _local_market_prep_secret_file() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "TradingBotV3" / "market_prep_secrets.json"
    return Path.home() / ".local" / "share" / "TradingBotV3" / "market_prep_secrets.json"


def _load_local_secrets() -> dict[str, Any]:
    path = _local_market_prep_secret_file()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_local_secret(name: str) -> str:
    return str(_load_local_secrets().get(name) or "")


def _save_local_secret(name: str, value: str) -> Path:
    path = _local_market_prep_secret_file()
    payload = _load_local_secrets()
    if value:
        payload[name] = value
    else:
        payload.pop(name, None)
    _write_json_atomic(path, payload)
    return path
