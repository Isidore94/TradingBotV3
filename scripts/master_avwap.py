#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import json
import re
import math
import importlib.util
import logging
import threading
import argparse
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, date
from pathlib import Path
from statistics import mean, median
from dataclasses import dataclass, asdict

try:
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
except Exception:
    tk = None
    ttk = None
    filedialog = None
    messagebox = None

import requests
import pandas as pd
import yfinance as yf
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

from project_paths import (
    LOCAL_SETTINGS_FILE,
    DATA_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    ROOT_DIR,
    LONGS_FILE,
    SHORTS_FILE,
    EARNINGS_CACHE_FILE,
    PREV_EARNINGS_CACHE_FILE,
    EARNINGS_DATES_CACHE_FILE,
    EARNINGS_CALENDAR_CACHE_FILE,
    YAHOO_SYMBOL_META_CACHE_FILE,
    DAILY_BARS_CACHE_DIR,
    MASTER_AVWAP_HISTORY_FILE,
    MASTER_AVWAP_AI_STATE_FILE,
    D1_FEATURES_FILE,
    AVWAP_SIGNALS_FILE,
    MASTER_AVWAP_EVENT_TICKERS_FILE,
    MASTER_AVWAP_PRIORITY_SETUPS_FILE,
    MASTER_POSITIONS_FILE,
    MASTER_AVWAP_REPORT_FILE,
    MASTER_AVWAP_STDEV_REPORT_FILE,
    MASTER_AVWAP_TRADINGVIEW_REPORT_FILE,
    MASTER_AVWAP_FOCUS_FILE,
    MASTER_AVWAP_SETUP_TRACKER_FILE,
    MASTER_AVWAP_SETUP_SCENARIOS_FILE,
    MASTER_AVWAP_SETUP_DAILY_FILE,
    MASTER_AVWAP_SETUP_STATS_FILE,
    EARNINGS_ANCHORS_FILE,
    EARNINGS_ANCHOR_CANDIDATES_FILE,
    EARNINGS_ANCHOR_CANDIDATES_REPORT_FILE,
    PREVIOUS_GAP_UPS_FILE,
    MASTER_ANCHOR_AVWAP_REPORT_FILE,
    ANCHOR_AVWAP_SIGNALS_FILE,
    MASTER_AVWAP_LOG_FILE,
    APP_LOG_BACKUP_COUNT,
    get_shared_watchlist_paths,
    get_tracker_storage_details,
    open_path_in_file_manager,
    save_tracker_storage_dir,
)

# ============================================================================
# PATHS / CONFIG
# ============================================================================

EVENT_TICKERS_FILE = MASTER_AVWAP_EVENT_TICKERS_FILE
PRIORITY_SETUPS_FILE = MASTER_AVWAP_PRIORITY_SETUPS_FILE
AVWAP_CSV_COLUMNS = [
    "run_date",
    "symbol",
    "trade_date",
    "side",
    "anchor_type",
    "anchor_date",
    "signal_type",
    "avwap_price",
    "band_price",
    "stdev",
    "priority_bucket",
    "is_favorite_setup",
    "is_near_favorite_zone",
    "favorite_zone",
    "favorite_signals",
    "favorite_context_signals",
]

EVENT_LEVEL_SORT_ORDER = [
    "UPPER_1",
    "UPPER_2",
    "UPPER_3",
    "VWAP",
    "LOWER_1",
    "LOWER_2",
    "LOWER_3",
]

FAVORITE_CURRENT_SIGNALS = {
    "LONG": {
        "BOUNCE_VWAP": 120,
        "CROSS_UP_UPPER_1": 110,
    },
    "SHORT": {
        "BOUNCE_VWAP": 120,
        "CROSS_DOWN_LOWER_1": 110,
    },
}

FAVORITE_CONTEXT_SIGNALS = {
    "LONG": {
        "PREV_BOUNCE_VWAP": 40,
        "PREV_CROSS_UPPER_1": 30,
    },
    "SHORT": {
        "PREV_BOUNCE_VWAP": 40,
        "PREV_CROSS_LOWER_1": 30,
    },
}

CURRENT_CACHE_FILE = EARNINGS_CACHE_FILE
PREV_CACHE_FILE = PREV_EARNINGS_CACHE_FILE
HISTORY_FILE = MASTER_AVWAP_HISTORY_FILE
AI_STATE_FILE = MASTER_AVWAP_AI_STATE_FILE
OUTPUT_FILE = MASTER_AVWAP_REPORT_FILE
STDEV_RANGE_FILE = MASTER_AVWAP_STDEV_REPORT_FILE
TRADINGVIEW_REPORT_FILE = MASTER_AVWAP_TRADINGVIEW_REPORT_FILE
SETUP_TRACKER_FILE = MASTER_AVWAP_SETUP_TRACKER_FILE
SETUP_SCENARIOS_FILE = MASTER_AVWAP_SETUP_SCENARIOS_FILE
SETUP_DAILY_FILE = MASTER_AVWAP_SETUP_DAILY_FILE
SETUP_STATS_FILE = MASTER_AVWAP_SETUP_STATS_FILE

API_URL = "https://api.nasdaq.com/api/calendar/earnings?date={date}"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*"
}

MAX_LOOKBACK_DAYS = 130       # Nasdaq earnings scan window
EARNINGS_RECENT_DISCOVERY_WINDOW_DAYS = 21
EARNINGS_FORCE_REFRESH_AFTER_DAYS = 100
EARNINGS_DEEP_LOOKBACK_DAYS = 220
EARNINGS_HISTORY_LIMIT = 8
EARNINGS_YF_RETRY_AFTER_DAYS = 30
MASTER_AVWAP_FOCUS_LIMIT_PER_BUCKET = 25
RECENT_DAYS = 10              # if earnings < RECENT_DAYS, use prior one for "current"
ATR_LENGTH = 20
ATR_MULT = 0.05               # eps / push = 0.05 * ATR(20)
BOUNCE_ATR_TOL_PCT = 0.001    # 0.1% of ATR(20) distance allowance for bounces
HISTORY_DAYS_TO_KEEP = 20     # multi-day context window
MIN_AVG_VOLUME_20D = 1_000_000
MIN_PRICE = 5.0
MIN_MARKET_CAP = 1_000_000_000
MIN_GAP_ATR_MULTIPLE = 1.0
RECENT_EARNINGS_SESSION_BLOCK = 12  # skip current AVWAPE events if last earnings is this recent
STDEV_RECENT_EARNINGS_BLOCK = 7      # skip stdev 2-3 scan when earnings is too recent
BOUNCE_LEVEL_ATR_TOL_PCT = 0.12      # 12% ATR proximity threshold for level touch/bounce
PRIORITY_SMA_LOOKBACK_DAYS = 320
PRIORITY_SMA_PERIODS = (20, 50, 100, 200)
PRIORITY_SMA_DQ_ATR = 2.0
PRIORITY_SMA_WARN_ATR = 3.0
PRIORITY_PREV_AVWAP_WARN_ATR = 3.0
PRIORITY_TRENDLINE_LOOKBACK_BARS = 200
PRIORITY_TRENDLINE_MIN_ANGLE_DEG = 30.0
PRIORITY_TRENDLINE_MAX_ANGLE_DEG = 60.0
PRIORITY_TRENDLINE_TOUCH_TOL_ATR = 0.35
PRIORITY_TRENDLINE_ALERT_ATR = 1.0
PRIORITY_TRENDLINE_PIVOT_WINDOW = 3
PRIORITY_TRENDLINE_MIN_SEPARATION_BARS = 10
PRIORITY_TRENDLINE_BREAK_RECENT_BARS = 3
PRIORITY_TRENDLINE_BREAK_MAX_ATR = 1.5
PRIORITY_TRENDLINE_BREAK_SCORE_BONUS = 18
PRIORITY_RETEST_LOOKBACK_BARS = 5
PRIORITY_RETEST_TOUCH_TOL_ATR = 0.25
PRIORITY_RETEST_CONFIRM_PUSH_ATR = 0.10
PRIORITY_FAVORITE_ZONE_SCORE_BONUS = 18
PRIORITY_SECOND_BAND_TEST_LOOKBACK_DAYS = 8
PRIORITY_SECOND_BAND_FIRST_TEST_SCORE_PENALTY = 10
PRIORITY_SECOND_BAND_REPEAT_TEST_SCORE_PENALTY = 4
PRIORITY_SECOND_BAND_MAX_SCORE_PENALTY = 22
PRIORITY_RETEST_FOLLOWTHROUGH_SCORE_BONUS = 30
PRIORITY_RETEST_LEVEL_SCORE_BONUS = {
    "AVWAPE": 12,
    "UPPER_1": 16,
    "LOWER_1": 16,
}
PRIORITY_RETEST_ZONE_CONFLUENCE_SCORE_BONUS = 10
PRIORITY_RETEST_TREND_ALIGNMENT_SCORE_BONUS = 8
PRIORITY_RETEST_CLEAR_PATH_SCORE_BONUS = 14
SETUP_TRACKER_SCHEMA_VERSION = 1
STANDARDIZED_RISK_USD = 500.0
TRACKER_STOP_FAILURE_CLOSES = 2
TRACKER_PREV_AVWAPE_NEAR_ATR = 0.5
TRACKER_EMA_CONSOLIDATION_ATR = 0.35
TRACKER_EMA_CONSOLIDATION_RANGE_ATR = 1.0
PRIORITY_COMPRESSION_STDEV_ATR_MAX = 0.75
PRIORITY_COMPRESSION_RANGE_ATR_MAX = 4.0
PRIORITY_COMPRESSION_CLOSE_RANGE_ATR_MAX = 2.5
PRIORITY_COMPRESSION_SCORE_PENALTY = 16
SETUP_EXIT_TEMPLATES = [
    {
        "id": "full_band2",
        "label": "Full at band2",
        "partial_target_label": None,
        "final_target_label": "BAND2",
        "trail_after_partial_label": None,
    },
    {
        "id": "half_band2_trail_band1",
        "label": "50% at band2, trail band1",
        "partial_target_label": "BAND2",
        "final_target_label": None,
        "trail_after_partial_label": "BAND1",
    },
    {
        "id": "full_band3",
        "label": "Full at band3",
        "partial_target_label": None,
        "final_target_label": "BAND3",
        "trail_after_partial_label": None,
    },
    {
        "id": "half_band2_band3_trail_band1",
        "label": "50% at band2, rest at band3 with band1 trail",
        "partial_target_label": "BAND2",
        "final_target_label": "BAND3",
        "trail_after_partial_label": "BAND1",
    },
]
POSITION_LEVELS = [
    "VWAP",
    "UPPER_1",
    "UPPER_2",
    "UPPER_3",
    "LOWER_1",
    "LOWER_2",
    "LOWER_3",
]

GUI_DARK_BG = "#2E2E2E"
GUI_DARK_PANEL = "#3A3A3A"
GUI_DARK_INPUT = "#252525"
# Softer light-gray text improves readability on platforms where pure white
# foreground can appear blown out against themed widget backgrounds.
GUI_DARK_TEXT = "#C7CDD4"
WATCHLIST_SYMBOL_RE = re.compile(r"[A-Z0-9.\-]+")
WATCHLIST_SKIP_TOKENS = {"LONG", "SHORT", "NONE"}
YF_EARNINGS_LOGGER_NAMES = (
    "yfinance",
    "yfinance.base",
    "yfinance.scrapers",
    "yfinance.scrapers.calendar",
)
YF_EARNINGS_NO_DATA_MARKERS = (
    "No earnings dates found",
    "symbol may be delisted",
    "'NoneType' object has no attribute 'index'",
)
YF_EARNINGS_LOOKUP_DISABLED_REASON = (
    "lxml is not installed"
    if importlib.util.find_spec("lxml") is None
    else None
)
YF_EARNINGS_LOOKUP_DISABLED_LOGGED = False
DAILY_BAR_CACHE_MAX_AGE_MINUTES = 30
DAILY_BAR_CACHE_HISTORY_BUFFER_DAYS = 10
DAILY_BAR_CACHE_RECENT_REFRESH_DAYS = 20
SYMBOL_METADATA_CACHE_MAX_AGE_DAYS = 1
EARNINGS_CALENDAR_TODAY_CACHE_MAX_AGE_MINUTES = 30
DAILY_BAR_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
APP_LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s]: %(message)s"

# ============================================================================
# LOGGING
# ============================================================================

def configure_logging():
    logger = logging.getLogger()
    if logger.handlers:
        return  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(APP_LOG_FORMAT)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(ch)
    try:
        fh = RotatingFileHandler(
            MASTER_AVWAP_LOG_FILE,
            maxBytes=2_000_000,
            backupCount=APP_LOG_BACKUP_COUNT,
        )
    except OSError as exc:
        logger.warning(f"File logging disabled for {MASTER_AVWAP_LOG_FILE}: {exc}")
        return

    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

configure_logging()

# ============================================================================
# UTILITY: WATCHLISTS
# ============================================================================

def load_tickers(path: Path):
    if not path.exists():
        logging.warning(f"Ticker file not found: {path}")
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if not v or v.upper().startswith("SYMBOLS FROM TC2000"):
                continue
            out.append(v.upper())
    return out


def connect_daily_data_client(client_id: int, startup_wait: float = 1.0) -> IBApi | None:
    ib = IBApi()
    try:
        ib.connect("127.0.0.1", 7496, clientId=client_id)
        threading.Thread(target=ib.run, daemon=True).start()
        time.sleep(startup_wait)
        if ib.isConnected():
            logging.info("Connected to IBKR for daily bar requests.")
            return ib
        logging.warning("IBKR is unavailable; falling back to Yahoo Finance daily bars for this scan.")
    except Exception as exc:
        logging.warning(f"IBKR daily-bar connection failed; using Yahoo Finance fallback ({exc}).")
    try:
        ib.disconnect()
    except Exception:
        pass
    return None


def disconnect_daily_data_client(ib: IBApi | None) -> None:
    if ib is None:
        return
    try:
        ib.disconnect()
    except Exception:
        pass


def resolve_scan_watchlist_paths(
    longs_path: Path | None = None,
    shorts_path: Path | None = None,
    use_shared_watchlists: bool = False,
) -> tuple[Path, Path, str]:
    if longs_path is not None and shorts_path is not None:
        return Path(longs_path), Path(shorts_path), "custom watchlists"
    if use_shared_watchlists:
        shared_longs_path, shared_shorts_path = get_shared_watchlist_paths()
        return shared_longs_path, shared_shorts_path, "home folder watchlists"
    return LONGS_FILE, SHORTS_FILE, "home folder watchlists"


def run_master_with_shared_watchlists():
    return run_master(use_shared_watchlists=True)

# ============================================================================
# CACHE HELPERS
# ============================================================================

def load_json(path: Path, default):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"JSON corrupt at {path}, starting fresh.")
    return default

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


_EARNINGS_CALENDAR_ROWS_CACHE: dict | None = None
_SYMBOL_METADATA_CACHE: dict | None = None
_DAILY_BAR_FRAME_CACHE: dict[str, pd.DataFrame] = {}
_DAILY_BAR_CACHE_TOUCHED_AT: dict[str, datetime] = {}


def _is_timestamp_within_minutes(value, minutes: int) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return (datetime.now() - parsed) <= timedelta(minutes=max(1, int(minutes)))


def _load_earnings_calendar_rows_cache() -> dict:
    global _EARNINGS_CALENDAR_ROWS_CACHE
    if _EARNINGS_CALENDAR_ROWS_CACHE is None:
        raw = load_json(EARNINGS_CALENDAR_CACHE_FILE, default={})
        _EARNINGS_CALENDAR_ROWS_CACHE = raw if isinstance(raw, dict) else {}
    return _EARNINGS_CALENDAR_ROWS_CACHE


def _save_earnings_calendar_rows_cache() -> None:
    if _EARNINGS_CALENDAR_ROWS_CACHE is None:
        return
    save_json(EARNINGS_CALENDAR_CACHE_FILE, _EARNINGS_CALENDAR_ROWS_CACHE)


def _load_symbol_metadata_cache() -> dict:
    global _SYMBOL_METADATA_CACHE
    if _SYMBOL_METADATA_CACHE is None:
        raw = load_json(YAHOO_SYMBOL_META_CACHE_FILE, default={})
        _SYMBOL_METADATA_CACHE = raw if isinstance(raw, dict) else {}
    return _SYMBOL_METADATA_CACHE


def _save_symbol_metadata_cache() -> None:
    if _SYMBOL_METADATA_CACHE is None:
        return
    save_json(YAHOO_SYMBOL_META_CACHE_FILE, _SYMBOL_METADATA_CACHE)


def _symbol_metadata_is_fresh(entry: dict | None) -> bool:
    if not isinstance(entry, dict):
        return False
    updated_on = _parse_iso_date_or_none(entry.get("updated_on"))
    if updated_on is None:
        return False
    return (datetime.now().date() - updated_on).days < max(1, SYMBOL_METADATA_CACHE_MAX_AGE_DAYS)


def _sanitize_symbol_for_filename(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    return re.sub(r"[^A-Z0-9._-]+", "_", text) or "UNKNOWN"


def _daily_bar_cache_file(symbol: str) -> Path:
    return DAILY_BARS_CACHE_DIR / f"{_sanitize_symbol_for_filename(symbol)}.csv"


def _empty_daily_bar_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_BAR_COLUMNS)


def _normalize_daily_bar_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_daily_bar_frame()

    normalized = df.copy()
    if "datetime" not in normalized.columns:
        if "Date" in normalized.columns:
            normalized = normalized.rename(columns={"Date": "datetime"})
        elif "date" in normalized.columns:
            normalized = normalized.rename(columns={"date": "datetime"})
        else:
            normalized = normalized.rename(columns={normalized.columns[0]: "datetime"})

    normalized = normalized.rename(columns={str(col): str(col).lower() for col in normalized.columns})
    if "datetime" not in normalized.columns:
        return _empty_daily_bar_frame()

    normalized["datetime"] = pd.to_datetime(normalized["datetime"], errors="coerce")
    if getattr(normalized["datetime"].dt, "tz", None) is not None:
        normalized["datetime"] = normalized["datetime"].dt.tz_localize(None)

    for column in DAILY_BAR_COLUMNS[1:]:
        if column not in normalized.columns:
            return _empty_daily_bar_frame()
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=DAILY_BAR_COLUMNS)
    if normalized.empty:
        return _empty_daily_bar_frame()

    normalized = (
        normalized[DAILY_BAR_COLUMNS]
        .drop_duplicates(subset=["datetime"], keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    return normalized


def _load_cached_daily_bar_frame(symbol: str) -> pd.DataFrame:
    symbol = str(symbol or "").strip().upper()
    cached = _DAILY_BAR_FRAME_CACHE.get(symbol)
    if cached is not None:
        return cached.copy()

    cache_path = _daily_bar_cache_file(symbol)
    if not cache_path.exists():
        return _empty_daily_bar_frame()

    try:
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
    except Exception as exc:
        logging.warning(f"{symbol}: failed reading cached daily bars ({exc})")
        return _empty_daily_bar_frame()

    normalized = _normalize_daily_bar_frame(df)
    _DAILY_BAR_FRAME_CACHE[symbol] = normalized
    _DAILY_BAR_CACHE_TOUCHED_AT[symbol] = datetime.fromtimestamp(cache_path.stat().st_mtime)
    return normalized.copy()


def _write_cached_daily_bar_frame(symbol: str, df: pd.DataFrame) -> None:
    symbol = str(symbol or "").strip().upper()
    normalized = _normalize_daily_bar_frame(df)
    cache_path = _daily_bar_cache_file(symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(cache_path, index=False)
    _DAILY_BAR_FRAME_CACHE[symbol] = normalized
    _DAILY_BAR_CACHE_TOUCHED_AT[symbol] = datetime.now()


def _merge_daily_bar_frames(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return _normalize_daily_bar_frame(fresh)
    if fresh is None or fresh.empty:
        return _normalize_daily_bar_frame(existing)
    combined = pd.concat([existing, fresh], ignore_index=True)
    return _normalize_daily_bar_frame(combined)


def _daily_bar_cache_covers_history(df: pd.DataFrame, days: int) -> bool:
    if df is None or df.empty:
        return False
    required_days = max(int(days), ATR_LENGTH + 5) + DAILY_BAR_CACHE_HISTORY_BUFFER_DAYS
    required_start = datetime.now().date() - timedelta(days=required_days)
    try:
        oldest_date = pd.to_datetime(df["datetime"].iloc[0]).date()
    except Exception:
        return False
    return oldest_date <= required_start


def _daily_bar_cache_is_recent(symbol: str) -> bool:
    symbol = str(symbol or "").strip().upper()
    touched_at = _DAILY_BAR_CACHE_TOUCHED_AT.get(symbol)
    if touched_at is None:
        cache_path = _daily_bar_cache_file(symbol)
        if not cache_path.exists():
            return False
        touched_at = datetime.fromtimestamp(cache_path.stat().st_mtime)
    return (datetime.now() - touched_at) <= timedelta(minutes=DAILY_BAR_CACHE_MAX_AGE_MINUTES)


def _get_cached_symbol_metadata(symbol: str, ticker_obj=None) -> dict:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return {}

    cache = _load_symbol_metadata_cache()
    cached = cache.get(normalized_symbol)
    if _symbol_metadata_is_fresh(cached):
        return cached

    if ticker_obj is None:
        ticker_obj = yf.Ticker(normalized_symbol)

    info = _get_info_with_fallbacks(ticker_obj)
    try:
        options = sorted(
            {
                pd.Timestamp(value).date().isoformat()
                for value in (ticker_obj.options or [])
            }
        )
    except Exception:
        options = []

    payload = {
        "updated_on": datetime.now().date().isoformat(),
        "market_cap": _coerce_int(
            info.get("marketCap")
            or info.get("market_cap")
            or getattr(getattr(ticker_obj, "fast_info", {}), "get", lambda *_: None)("marketCap")
        ),
        "options_expirations": options,
    }
    cache[normalized_symbol] = payload
    _save_symbol_metadata_cache()
    return payload


def load_earnings_date_cache():
    raw_cache = load_json(EARNINGS_DATES_CACHE_FILE, default={})
    normalized = {
        "schema_version": 2,
        "last_recent_refresh_on": None,
        "symbols": {},
    }
    if not isinstance(raw_cache, dict):
        return normalized

    normalized["last_recent_refresh_on"] = (
        raw_cache.get("last_recent_refresh_on") or raw_cache.get("refreshed_on")
    )
    raw_symbols = raw_cache.get("symbols")
    if not isinstance(raw_symbols, dict):
        raw_symbols = raw_cache.get("data", {})

    for raw_symbol, raw_entry in raw_symbols.items():
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol:
            continue
        entry = _normalize_earnings_cache_entry(
            raw_entry,
            fallback_refresh_on=normalized["last_recent_refresh_on"],
        )
        if entry["dates"]:
            normalized["symbols"][symbol] = entry

    return normalized


def save_earnings_date_cache(cache_obj):
    symbols_payload = {}
    raw_symbols = cache_obj.get("symbols", {}) if isinstance(cache_obj, dict) else {}
    for raw_symbol, raw_entry in raw_symbols.items():
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol:
            continue
        entry = _normalize_earnings_cache_entry(raw_entry)
        if entry["dates"]:
            symbols_payload[symbol] = entry

    normalized = {
        "schema_version": 2,
        "last_recent_refresh_on": (
            cache_obj.get("last_recent_refresh_on") if isinstance(cache_obj, dict) else None
        ),
        "symbols": symbols_payload,
    }
    save_json(EARNINGS_DATES_CACHE_FILE, normalized)


def _parse_iso_date_or_none(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except (TypeError, ValueError):
        return None


def _normalize_earnings_dates(dates, today=None, limit=EARNINGS_HISTORY_LIMIT):
    today = today or datetime.now().date()
    normalized = []
    seen = set()
    for value in dates or []:
        parsed = _parse_iso_date_or_none(value)
        if parsed is None or parsed > today:
            continue
        iso = parsed.isoformat()
        if iso in seen:
            continue
        seen.add(iso)
        normalized.append(iso)
    normalized.sort(reverse=True)
    return normalized[:limit]


def _normalize_earnings_cache_entry(raw_entry, fallback_refresh_on=None):
    if isinstance(raw_entry, dict):
        dates = raw_entry.get("dates", [])
        last_deep_refresh_on = (
            raw_entry.get("last_deep_refresh_on")
            or raw_entry.get("last_full_refresh_on")
            or raw_entry.get("refreshed_on")
            or fallback_refresh_on
        )
        last_yf_refresh_on = raw_entry.get("last_yf_refresh_on")
    else:
        dates = raw_entry
        last_deep_refresh_on = fallback_refresh_on
        last_yf_refresh_on = None

    return {
        "dates": _normalize_earnings_dates(dates),
        "last_deep_refresh_on": last_deep_refresh_on,
        "last_yf_refresh_on": last_yf_refresh_on,
    }


def _merge_earnings_dates(*date_lists):
    merged = []
    for values in date_lists:
        merged.extend(values or [])
    return _normalize_earnings_dates(merged)


def _earnings_symbol_needs_deep_refresh(symbol_entry, today=None):
    today = today or datetime.now().date()
    dates = _normalize_earnings_dates(symbol_entry.get("dates", []), today=today)
    if len(dates) < 2:
        return True
    latest_date = _parse_iso_date_or_none(dates[0])
    if latest_date is None:
        return True
    return (today - latest_date).days >= EARNINGS_FORCE_REFRESH_AFTER_DAYS


def _earnings_symbol_needs_yf_refresh(symbol_entry, today=None):
    today = today or datetime.now().date()
    last_yf_refresh = _parse_iso_date_or_none(symbol_entry.get("last_yf_refresh_on"))
    if last_yf_refresh is None:
        return True
    return (today - last_yf_refresh).days >= EARNINGS_YF_RETRY_AFTER_DAYS


def _can_attempt_yf_earnings_lookup(log_once: bool = False) -> bool:
    global YF_EARNINGS_LOOKUP_DISABLED_LOGGED

    if YF_EARNINGS_LOOKUP_DISABLED_REASON:
        if log_once and not YF_EARNINGS_LOOKUP_DISABLED_LOGGED:
            logging.info(
                f"Skipping yfinance earnings supplement: {YF_EARNINGS_LOOKUP_DISABLED_REASON}."
            )
            YF_EARNINGS_LOOKUP_DISABLED_LOGGED = True
        return False
    return True


def compute_atr_from_ohlc(daily_rows, last_trade_date, length=ATR_LENGTH):
    target = last_trade_date.isoformat()
    relevant = [row for row in daily_rows if row["date"] <= target]
    if not relevant:
        return None

    relevant = relevant[-length:]
    trs = []
    prev_close = None
    for row in relevant:
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = close

    if not trs:
        return None
    return sum(trs) / len(trs)


def compute_trend_label_20d(daily_rows, last_trade_date):
    target = last_trade_date.isoformat()
    closes = [float(row["close"]) for row in daily_rows if row["date"] <= target]
    relevant_rows = [row for row in daily_rows if row["date"] <= target]
    if len(closes) < 10 or not relevant_rows:
        return "SIDEWAYS"

    short = closes[-5:]
    long = closes[-20:] if len(closes) >= 20 else closes

    short_avg = mean(short)
    long_avg = mean(long)
    if long_avg == 0:
        return "SIDEWAYS"

    diff_pct = (short_avg - long_avg) / long_avg * 100
    recent_closes = closes[-10:]
    recent_rows = relevant_rows[-5:]

    n = len(recent_closes)
    x_mean = (n - 1) / 2
    y_mean = mean(recent_closes)
    denom = sum((idx - x_mean) ** 2 for idx in range(n))
    slope = (
        sum((idx - x_mean) * (value - y_mean) for idx, value in enumerate(recent_closes)) / denom
        if denom else 0.0
    )
    slope_pct = (slope / y_mean * 100) if y_mean else 0.0
    swing_base = recent_closes[0]
    swing_pct = ((recent_closes[-1] - swing_base) / swing_base * 100) if swing_base else 0.0

    recent_highs = [float(row["high"]) for row in recent_rows]
    recent_lows = [float(row["low"]) for row in recent_rows]
    up_structure = recent_highs[-1] > recent_highs[0] and recent_lows[-1] > recent_lows[0]
    down_structure = recent_highs[-1] < recent_highs[0] and recent_lows[-1] < recent_lows[0]

    if (
        (diff_pct >= 0.40 and slope_pct >= 0.08)
        or (swing_pct >= 1.00 and up_structure)
    ):
        return "UP"
    if (
        (diff_pct <= -0.40 and slope_pct <= -0.08)
        or (swing_pct <= -1.00 and down_structure)
    ):
        return "DOWN"
    return "SIDEWAYS"


def compute_five_day_breakout_flags(daily_rows, last_trade_date):
    target = last_trade_date.isoformat()
    relevant = [row for row in daily_rows if row["date"] <= target]
    if len(relevant) < 6:
        return False, False

    current = relevant[-1]
    prior_five = relevant[-6:-1]
    current_close = float(current["close"])
    prior_high = max(float(row["high"]) for row in prior_five)
    prior_low = min(float(row["low"]) for row in prior_five)
    return current_close > prior_high, current_close < prior_low


def count_recent_band_extension_days(daily_rows, last_trade_date, band_level, side, lookback=5):
    if band_level is None:
        return 0

    target = last_trade_date.isoformat()
    relevant = [row for row in daily_rows if row["date"] <= target]
    if len(relevant) < 2:
        return 0

    prior_rows = relevant[max(0, len(relevant) - 1 - lookback):-1]
    count = 0
    for row in reversed(prior_rows):
        if side == "LONG" and float(row["low"]) < band_level:
            count += 1
            continue
        if side == "SHORT" and float(row["high"]) > band_level:
            count += 1
            continue
        break
    return count


def count_recent_band_test_days(
    daily_rows,
    last_trade_date,
    band_level,
    side,
    lookback=PRIORITY_SECOND_BAND_TEST_LOOKBACK_DAYS,
):
    if band_level is None:
        return 0

    target = last_trade_date.isoformat()
    relevant = [row for row in daily_rows if row["date"] <= target]
    if not relevant:
        return 0

    recent_rows = relevant[-max(1, int(lookback)):]
    count = 0
    for row in recent_rows:
        try:
            row_high = float(row["high"])
            row_low = float(row["low"])
        except (KeyError, TypeError, ValueError):
            continue
        if side == "LONG" and row_high >= float(band_level):
            count += 1
        elif side == "SHORT" and row_low <= float(band_level):
            count += 1
    return count


def compute_recent_second_band_penalty(recent_second_band_test_days: int) -> int:
    if recent_second_band_test_days <= 0:
        return 0
    penalty = PRIORITY_SECOND_BAND_FIRST_TEST_SCORE_PENALTY
    penalty += max(0, int(recent_second_band_test_days) - 1) * PRIORITY_SECOND_BAND_REPEAT_TEST_SCORE_PENALTY
    return min(PRIORITY_SECOND_BAND_MAX_SCORE_PENALTY, penalty)


def analyze_avwap_retest_behavior(
    daily_rows,
    last_trade_date,
    current_vwap,
    side,
    current_upper_1=None,
    current_lower_1=None,
    atr20=None,
):
    result = {
        "retest_followthrough": False,
        "retest_reference_level": "",
        "retest_note": "",
    }
    if current_vwap is None or atr20 is None or atr20 <= 0:
        return result

    target = last_trade_date.isoformat()
    relevant = [row for row in daily_rows if row["date"] <= target]
    if len(relevant) < PRIORITY_RETEST_LOOKBACK_BARS:
        return result

    recent = relevant[-PRIORITY_RETEST_LOOKBACK_BARS:]
    prior = recent[:-1]
    current = recent[-1]
    prior_closes = [float(row["close"]) for row in prior]
    prior_lows = [float(row["low"]) for row in prior]
    prior_highs = [float(row["high"]) for row in prior]
    current_close = float(current["close"])
    current_low = float(current["low"])
    current_high = float(current["high"])
    touch_tol = max(BOUNCE_LEVEL_ATR_TOL_PCT * float(atr20), PRIORITY_RETEST_TOUCH_TOL_ATR * float(atr20))
    confirm_push = max(ATR_MULT * float(atr20), PRIORITY_RETEST_CONFIRM_PUSH_ATR * float(atr20))
    retest_window = prior[-3:]

    if side == "SHORT":
        candidate_levels = [
            ("UPPER_1", current_upper_1),
            ("AVWAPE", current_vwap),
        ]
        directional_drive = prior_closes[-1] < prior_closes[0]
        continuation = (
            current_close < prior_closes[-1]
            and current_low < min(prior_lows[-2:])
            and current_close < current_vwap - confirm_push
        )
        if not (directional_drive and continuation):
            return result

        for label, level in candidate_levels:
            if level is None:
                continue
            touched = any(float(row["high"]) >= float(level) - touch_tol for row in retest_window)
            rejected = any(
                float(row["close"]) <= float(level) + touch_tol
                for row in retest_window
                if float(row["high"]) >= float(level) - touch_tol
            )
            if not (touched and rejected):
                continue
            result["retest_followthrough"] = True
            result["retest_reference_level"] = label
            result["retest_note"] = f"Retested {label} and continued lower"
            return result

        return result

    candidate_levels = [
        ("LOWER_1", current_lower_1),
        ("AVWAPE", current_vwap),
    ]
    directional_drive = prior_closes[-1] > prior_closes[0]
    continuation = (
        current_close > prior_closes[-1]
        and current_high > max(prior_highs[-2:])
        and current_close > current_vwap + confirm_push
    )
    if not (directional_drive and continuation):
        return result

    for label, level in candidate_levels:
        if level is None:
            continue
        touched = any(float(row["low"]) <= float(level) + touch_tol for row in retest_window)
        reclaimed = any(
            float(row["close"]) >= float(level) - touch_tol
            for row in retest_window
            if float(row["low"]) <= float(level) + touch_tol
        )
        if not (touched and reclaimed):
            continue
        result["retest_followthrough"] = True
        result["retest_reference_level"] = label
        result["retest_note"] = f"Retested {label} and continued higher"
        return result

    return result


def compute_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy().reset_index(drop=True)
    work["close_num"] = pd.to_numeric(work["close"], errors="coerce")
    work["high_num"] = pd.to_numeric(work["high"], errors="coerce")
    work["low_num"] = pd.to_numeric(work["low"], errors="coerce")

    for period in (10, 20, 50, 100, 200):
        work[f"sma_{period}"] = work["close_num"].rolling(period).mean()
        work[f"sma_{period}_prev"] = work[f"sma_{period}"].shift(1)

    for span in (15, 21):
        work[f"ema_{span}"] = work["close_num"].ewm(span=span, adjust=False).mean()

    prev_close = work["close_num"].shift(1)
    tr_components = pd.concat(
        [
            work["high_num"] - work["low_num"],
            (work["high_num"] - prev_close).abs(),
            (work["low_num"] - prev_close).abs(),
        ],
        axis=1,
    )
    work["atr_20"] = tr_components.max(axis=1).rolling(ATR_LENGTH, min_periods=1).mean()
    work["recent_high_3"] = work["high_num"].rolling(3).max()
    work["recent_low_3"] = work["low_num"].rolling(3).min()
    work["recent_close_std_3"] = work["close_num"].rolling(3).std(ddof=0)
    work["trade_date"] = pd.to_datetime(work["datetime"]).dt.date.astype(str)
    return work


def calc_anchored_vwap_band_history(df: pd.DataFrame, anchor_date_iso: str) -> dict[str, dict]:
    try:
        anchor_date = datetime.fromisoformat(str(anchor_date_iso)).date()
    except ValueError:
        return {}

    idxs = df.index[df["datetime"].dt.date == anchor_date]
    if idxs.empty:
        return {}

    anchor_idx = int(idxs[0])
    cum_vol = 0.0
    cum_vp = 0.0
    cum_sd = 0.0
    history = {}

    for i in range(anchor_idx, len(df)):
        row = df.iloc[i]
        volume = float(row["volume"])
        if volume <= 0:
            continue
        tp = (float(row["open"]) + float(row["high"]) + float(row["low"]) + float(row["close"])) / 4.0
        cum_vol += volume
        cum_vp += tp * volume
        vwap = cum_vp / cum_vol
        dev = tp - vwap
        cum_sd += dev * dev * volume
        stdev = (cum_sd / cum_vol) ** 0.5 if cum_vol > 0 else float("nan")
        history[row["datetime"].date().isoformat()] = {
            "vwap": float(vwap),
            "stdev": float(stdev),
            "bands": {
                "UPPER_1": float(vwap + stdev),
                "LOWER_1": float(vwap - stdev),
                "UPPER_2": float(vwap + 2 * stdev),
                "LOWER_2": float(vwap - 2 * stdev),
                "UPPER_3": float(vwap + 3 * stdev),
                "LOWER_3": float(vwap - 3 * stdev),
            },
        }

    return history


def _anchor_level_value(anchor_levels: dict | None, label: str):
    if not anchor_levels or not label:
        return None
    if label in {"AVWAPE", "VWAP"}:
        return _coerce_float(anchor_levels.get("vwap"))
    bands = anchor_levels.get("bands", {}) if isinstance(anchor_levels.get("bands"), dict) else {}
    return _coerce_float(bands.get(label))


def _indicator_level_value(indicator_row: pd.Series | dict | None, label: str):
    if indicator_row is None or not label:
        return None
    field_map = {
        "SMA_10": "sma_10",
        "SMA_20": "sma_20",
        "SMA_50": "sma_50",
        "SMA_100": "sma_100",
        "SMA_200": "sma_200",
        "EMA_15": "ema_15",
        "EMA_21": "ema_21",
    }
    field = field_map.get(label)
    if not field:
        return None
    if isinstance(indicator_row, pd.Series):
        return _coerce_float(indicator_row.get(field))
    return _coerce_float(indicator_row.get(field))


def _resolve_dynamic_level(label: str, current_anchor_levels: dict | None, indicator_row: pd.Series | dict | None):
    if label in {"AVWAPE", "VWAP", "UPPER_1", "LOWER_1", "UPPER_2", "LOWER_2", "UPPER_3", "LOWER_3"}:
        return _anchor_level_value(current_anchor_levels, label)
    return _indicator_level_value(indicator_row, label)


def _favorable_band_label(side: str, band_number: int) -> str:
    side = normalize_side(side)
    if band_number == 1:
        return "UPPER_1" if side == "LONG" else "LOWER_1"
    if band_number == 2:
        return "UPPER_2" if side == "LONG" else "LOWER_2"
    return "UPPER_3" if side == "LONG" else "LOWER_3"


def _protective_band_label(side: str) -> str:
    return "LOWER_1" if normalize_side(side) == "LONG" else "UPPER_1"


def summarize_anchor_compression(price_slice: pd.DataFrame, anchor_stdev: float, atr20: float) -> dict:
    result = {
        "is_compressed": False,
        "compression_score": 0,
        "compression_penalty": 0,
        "compression_note": "",
        "compression_stdev_atr_ratio": None,
        "compression_range_atr_ratio": None,
        "compression_close_range_atr_ratio": None,
    }
    if price_slice is None or price_slice.empty or anchor_stdev is None or atr20 is None or atr20 <= 0:
        return result

    highs = pd.to_numeric(price_slice["high"], errors="coerce").dropna()
    lows = pd.to_numeric(price_slice["low"], errors="coerce").dropna()
    closes = pd.to_numeric(price_slice["close"], errors="coerce").dropna()
    if highs.empty or lows.empty or closes.empty:
        return result

    stdev_atr_ratio = float(anchor_stdev) / float(atr20)
    range_atr_ratio = (float(highs.max()) - float(lows.min())) / float(atr20)
    close_range_atr_ratio = (float(closes.max()) - float(closes.min())) / float(atr20)
    compression_score = 0
    if stdev_atr_ratio <= PRIORITY_COMPRESSION_STDEV_ATR_MAX:
        compression_score += 1
    if range_atr_ratio <= PRIORITY_COMPRESSION_RANGE_ATR_MAX:
        compression_score += 1
    if close_range_atr_ratio <= PRIORITY_COMPRESSION_CLOSE_RANGE_ATR_MAX:
        compression_score += 1

    result["compression_score"] = compression_score
    result["compression_stdev_atr_ratio"] = float(stdev_atr_ratio)
    result["compression_range_atr_ratio"] = float(range_atr_ratio)
    result["compression_close_range_atr_ratio"] = float(close_range_atr_ratio)
    if compression_score >= 2:
        result["is_compressed"] = True
        penalty = PRIORITY_COMPRESSION_SCORE_PENALTY + (4 if compression_score >= 3 else 0)
        result["compression_penalty"] = penalty
        result["compression_note"] = (
            f"Compressed post-earnings structure "
            f"(stdev={stdev_atr_ratio:.2f} ATR, range={range_atr_ratio:.2f} ATR, close_range={close_range_atr_ratio:.2f} ATR)"
        )
    return result


def evaluate_anchor_compression(
    df: pd.DataFrame,
    anchor_date_iso: str | None,
    anchor_stdev: float | None,
    atr20: float | None,
    last_trade_date: date,
) -> dict:
    if not anchor_date_iso:
        return summarize_anchor_compression(pd.DataFrame(), anchor_stdev, atr20)
    try:
        anchor_date = datetime.fromisoformat(str(anchor_date_iso)).date()
    except ValueError:
        return summarize_anchor_compression(pd.DataFrame(), anchor_stdev, atr20)

    price_slice = df[
        (df["datetime"].dt.date >= anchor_date)
        & (df["datetime"].dt.date <= last_trade_date)
    ].copy()
    return summarize_anchor_compression(price_slice, anchor_stdev, atr20)


def build_tracker_feature_snapshot(
    side: str,
    bar_row: pd.Series,
    indicator_row: pd.Series | None,
    current_anchor_levels: dict | None,
    previous_anchor_levels: dict | None,
    compression_summary: dict | None,
) -> dict:
    atr20 = _coerce_float(indicator_row.get("atr_20")) if isinstance(indicator_row, pd.Series) else None
    close_value = _coerce_float(bar_row.get("close"))
    prev_avwape = _anchor_level_value(previous_anchor_levels, "AVWAPE")
    prev_distance_atr = None
    prev_near_flag = False
    if close_value is not None and prev_avwape is not None and atr20 and atr20 > 0:
        prev_distance_atr = (close_value - prev_avwape) / atr20
        prev_near_flag = abs(prev_distance_atr) <= TRACKER_PREV_AVWAPE_NEAR_ATR

    ema15 = _indicator_level_value(indicator_row, "EMA_15")
    ema21 = _indicator_level_value(indicator_row, "EMA_21")
    sma10 = _indicator_level_value(indicator_row, "SMA_10")
    sma20 = _indicator_level_value(indicator_row, "SMA_20")
    sma50 = _indicator_level_value(indicator_row, "SMA_50")
    sma100 = _indicator_level_value(indicator_row, "SMA_100")
    sma200 = _indicator_level_value(indicator_row, "SMA_200")

    ema15_distance_atr = None
    ema21_distance_atr = None
    if close_value is not None and atr20 and atr20 > 0:
        if ema15 is not None:
            ema15_distance_atr = (close_value - ema15) / atr20
        if ema21 is not None:
            ema21_distance_atr = (close_value - ema21) / atr20

    prior_sma10 = _coerce_float(indicator_row.get("sma_10_prev")) if isinstance(indicator_row, pd.Series) else None
    prior_sma20 = _coerce_float(indicator_row.get("sma_20_prev")) if isinstance(indicator_row, pd.Series) else None
    sma_cross = ""
    if sma10 is not None and sma20 is not None and prior_sma10 is not None and prior_sma20 is not None:
        if prior_sma10 <= prior_sma20 and sma10 > sma20:
            sma_cross = "UP"
        elif prior_sma10 >= prior_sma20 and sma10 < sma20:
            sma_cross = "DOWN"

    ema21_consolidation = False
    consolidation_span_atr = None
    if isinstance(indicator_row, pd.Series):
        recent_high = _coerce_float(indicator_row.get("recent_high_3"))
        recent_low = _coerce_float(indicator_row.get("recent_low_3"))
        recent_close_std = _coerce_float(indicator_row.get("recent_close_std_3"))
        if atr20 and atr20 > 0 and recent_high is not None and recent_low is not None and ema21 is not None:
            consolidation_span_atr = (recent_high - recent_low) / atr20
            ema21_consolidation = (
                consolidation_span_atr <= TRACKER_EMA_CONSOLIDATION_RANGE_ATR
                and ema21_distance_atr is not None
                and abs(ema21_distance_atr) <= TRACKER_EMA_CONSOLIDATION_ATR
                and recent_close_std is not None
                and (recent_close_std / atr20) <= 0.45
            )

    current_vwap = _anchor_level_value(current_anchor_levels, "AVWAPE")
    current_upper_1 = _anchor_level_value(current_anchor_levels, "UPPER_1")
    current_lower_1 = _anchor_level_value(current_anchor_levels, "LOWER_1")

    snapshot = {
        "atr20": atr20,
        "current_close": close_value,
        "current_avwape": current_vwap,
        "current_upper_1": current_upper_1,
        "current_lower_1": current_lower_1,
        "previous_avwape": prev_avwape,
        "previous_avwape_distance_atr": prev_distance_atr,
        "previous_avwape_near_0_5atr": prev_near_flag,
        "ema15": ema15,
        "ema21": ema21,
        "ema15_distance_atr": ema15_distance_atr,
        "ema21_distance_atr": ema21_distance_atr,
        "ema21_consolidation": ema21_consolidation,
        "ema21_consolidation_span_atr": consolidation_span_atr,
        "sma10": sma10,
        "sma20": sma20,
        "sma50": sma50,
        "sma100": sma100,
        "sma200": sma200,
        "sma10_20_cross": sma_cross,
        "compression_flag": bool((compression_summary or {}).get("is_compressed")),
        "compression_penalty": int((compression_summary or {}).get("compression_penalty", 0) or 0),
        "compression_note": (compression_summary or {}).get("compression_note", ""),
    }
    return snapshot


def _default_setup_tracker_payload() -> dict:
    return {
        "schema_version": SETUP_TRACKER_SCHEMA_VERSION,
        "updated_at": None,
        "daily_watchlists": {},
        "setups": {},
        "stats": [],
    }


def load_setup_tracker_payload() -> dict:
    payload = load_json(SETUP_TRACKER_FILE, default={})
    if not isinstance(payload, dict):
        return _default_setup_tracker_payload()

    tracker = _default_setup_tracker_payload()
    tracker["schema_version"] = int(payload.get("schema_version", SETUP_TRACKER_SCHEMA_VERSION) or SETUP_TRACKER_SCHEMA_VERSION)
    tracker["updated_at"] = payload.get("updated_at")
    tracker["daily_watchlists"] = payload.get("daily_watchlists", {}) if isinstance(payload.get("daily_watchlists"), dict) else {}
    tracker["setups"] = payload.get("setups", {}) if isinstance(payload.get("setups"), dict) else {}
    tracker["stats"] = payload.get("stats", []) if isinstance(payload.get("stats"), list) else []
    return tracker


def save_setup_tracker_payload(payload: dict) -> None:
    payload["schema_version"] = SETUP_TRACKER_SCHEMA_VERSION
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(SETUP_TRACKER_FILE, payload)


def _setup_id_for_row(row: dict, symbol_entry: dict) -> str:
    scan_date = str(symbol_entry.get("last_trade_date") or datetime.now().date().isoformat())
    anchor_date = ((symbol_entry.get("current_anchor") or {}).get("date")) or scan_date
    bucket = str(row.get("priority_bucket") or "tracked")
    return f"{scan_date}:{row.get('symbol', '')}:{normalize_side(row.get('side'))}:{anchor_date}:{bucket}"


def _find_tracker_stop_candidates(row: dict, symbol_entry: dict) -> list[dict]:
    side = normalize_side(row.get("side") or symbol_entry.get("side") or "LONG")
    current_anchor = symbol_entry.get("current_anchor") or {}
    atr20 = _coerce_float(symbol_entry.get("atr20"))
    primary_stop_label = _protective_band_label(side)
    primary_stop_level = _anchor_level_value(current_anchor, primary_stop_label)
    priority_bucket = str(row.get("priority_bucket") or "")
    candidates = []

    def _add(label: str, level: float | None, source_type: str) -> None:
        if level is None:
            return
        if any(existing["label"] == label for existing in candidates):
            return
        candidates.append(
            {
                "label": label,
                "level": float(level),
                "source_type": source_type,
            }
        )

    if priority_bucket == "favorite_setup":
        _add("AVWAPE", _anchor_level_value(current_anchor, "AVWAPE"), "current_anchor")
    _add(primary_stop_label, primary_stop_level, "current_anchor")

    sma_levels = row.get("sma_levels") or symbol_entry.get("priority_sma_levels") or {}
    if primary_stop_level is not None and atr20 and atr20 > 0:
        for label, level in sorted(sma_levels.items()):
            if level is None:
                continue
            if abs(float(level) - float(primary_stop_level)) <= float(atr20):
                _add(str(label), float(level), "sma")

    return candidates


def _build_tracker_scenarios(entry_price: float | None, stop_candidates: list[dict], side: str) -> dict[str, dict]:
    scenarios = {}
    direction = 1.0 if normalize_side(side) == "LONG" else -1.0

    for stop in stop_candidates:
        stop_level = _coerce_float(stop.get("level"))
        risk_per_share = abs(float(entry_price) - float(stop_level)) if entry_price is not None and stop_level is not None else None
        shares = int(STANDARDIZED_RISK_USD // risk_per_share) if risk_per_share and risk_per_share > 0 else 0
        for template in SETUP_EXIT_TEMPLATES:
            scenario_id = f"{str(stop['label']).lower()}__{template['id']}"
            scenarios[scenario_id] = {
                "scenario_id": scenario_id,
                "stop_reference_label": stop["label"],
                "stop_reference_level": stop_level,
                "stop_source_type": stop["source_type"],
                "exit_template_id": template["id"],
                "exit_template_label": template["label"],
                "partial_target_label": _favorable_band_label(side, 2) if template.get("partial_target_label") == "BAND2" else None,
                "final_target_label": (
                    _favorable_band_label(side, 2) if template.get("final_target_label") == "BAND2"
                    else _favorable_band_label(side, 3) if template.get("final_target_label") == "BAND3"
                    else None
                ),
                "trail_after_partial_label": _favorable_band_label(side, 1) if template.get("trail_after_partial_label") == "BAND1" else None,
                "shares": int(shares),
                "initial_risk_per_share": risk_per_share,
                "initial_risk_usd": float(shares * risk_per_share) if shares and risk_per_share else 0.0,
                "direction": direction,
                "tradeable": bool(shares > 0 and risk_per_share and risk_per_share > 0),
                "status": "UNTRADEABLE" if not (shares > 0 and risk_per_share and risk_per_share > 0) else "OPEN",
                "events": [],
                "remaining_shares": int(shares),
                "partial_taken": False,
                "partial_shares": 0,
                "realized_pnl": 0.0,
                "realized_r": 0.0,
                "unrealized_pnl": 0.0,
                "unrealized_r": 0.0,
                "total_pnl": 0.0,
                "total_r": 0.0,
                "close_failure_count": 0,
                "active_stop_label": stop["label"],
                "active_stop_level": stop_level,
                "last_action": "Awaiting update" if shares > 0 else "Risk too tight for $500 standard risk",
                "days_held": 0,
                "max_favorable_r": 0.0,
                "max_adverse_r": 0.0,
            }
    return scenarios


def build_tracker_setup_record(
    row: dict,
    symbol_entry: dict,
    feature_row: dict | None,
    generated_at: str,
    indicator_row: pd.Series | None,
) -> dict | None:
    entry_price = _coerce_float(symbol_entry.get("last_close"))
    if entry_price is None:
        return None

    current_anchor = symbol_entry.get("current_anchor") or {}
    previous_anchor = symbol_entry.get("previous_anchor") or {}
    compression_summary = {
        "is_compressed": bool(row.get("compression_flag")),
        "compression_penalty": int(row.get("compression_penalty", 0) or 0),
        "compression_note": row.get("compression_note", ""),
    }
    entry_snapshot = build_tracker_feature_snapshot(
        normalize_side(row.get("side")),
        pd.Series({"open": entry_price, "high": entry_price, "low": entry_price, "close": entry_price}),
        indicator_row,
        current_anchor,
        previous_anchor,
        compression_summary,
    )

    stop_candidates = _find_tracker_stop_candidates(row, symbol_entry)
    scenarios = _build_tracker_scenarios(entry_price, stop_candidates, normalize_side(row.get("side")))
    setup_id = _setup_id_for_row(row, symbol_entry)
    scan_date = str(symbol_entry.get("last_trade_date") or datetime.now().date().isoformat())

    return {
        "setup_id": setup_id,
        "symbol": str(row.get("symbol", "")).strip().upper(),
        "side": normalize_side(row.get("side")),
        "scan_date": scan_date,
        "scan_timestamp": generated_at,
        "entry_trade_date": scan_date,
        "entry_price": float(entry_price),
        "anchor_date": str(current_anchor.get("date") or scan_date),
        "previous_anchor_date": str(previous_anchor.get("date") or ""),
        "priority_bucket": str(row.get("priority_bucket") or ""),
        "priority_score": float(row.get("score", 0) or 0),
        "favorite_zone": row.get("favorite_zone") or "",
        "favorite_signals": list(row.get("favorite_signals") or []),
        "context_signals": list(row.get("context_signals") or []),
        "recent_second_band_test_days": int(row.get("recent_second_band_test_days", 0) or 0),
        "second_band_penalty": int(row.get("second_band_penalty", 0) or 0),
        "retest_followthrough": bool(row.get("retest_followthrough")),
        "retest_reference_level": row.get("retest_reference_level") or "",
        "retest_note": row.get("retest_note") or "",
        "extension_note": row.get("extension_note") or "",
        "ranking_note": row.get("ranking_note") or "",
        "score_bonus_note": row.get("score_bonus_note") or "",
        "trendline_break_recent": bool(row.get("trendline_break_recent")),
        "trendline_break_note": row.get("trendline_break_note") or "",
        "compression_flag": bool(row.get("compression_flag")),
        "compression_penalty": int(row.get("compression_penalty", 0) or 0),
        "compression_note": row.get("compression_note") or "",
        "current_anchor_entry": current_anchor,
        "previous_anchor_entry": previous_anchor,
        "sma_levels_entry": row.get("sma_levels") or {},
        "entry_feature_snapshot": entry_snapshot,
        "scenarios": scenarios,
        "daily_marks": [],
        "latest_snapshot": {},
        "setup_status": "OPEN" if any(item.get("tradeable") for item in scenarios.values()) else "UNTRADEABLE",
        "open_scenario_count": 0,
        "closed_scenario_count": 0,
        "feature_row": feature_row or {},
    }


def _scenario_is_open(status: str) -> bool:
    return str(status).upper() in {"OPEN", "PARTIAL", "ACTIVE"}


def _bar_hits_target(side: str, bar_row: pd.Series, target_level: float | None) -> bool:
    if target_level is None:
        return False
    side = normalize_side(side)
    if side == "LONG":
        return _coerce_float(bar_row.get("high")) is not None and float(bar_row["high"]) >= float(target_level)
    return _coerce_float(bar_row.get("low")) is not None and float(bar_row["low"]) <= float(target_level)


def _apply_scenario_exit_event(scenario: dict, qty: int, exit_price: float, trade_date: str, reason: str) -> dict:
    qty = int(max(0, qty))
    if qty <= 0:
        return {}
    direction = float(scenario.get("direction", 1.0) or 1.0)
    entry_price = float(scenario.get("entry_price"))
    pnl = (float(exit_price) - entry_price) * qty * direction
    scenario["realized_pnl"] = float(scenario.get("realized_pnl", 0.0) + pnl)
    scenario["remaining_shares"] = int(max(0, int(scenario.get("remaining_shares", 0)) - qty))
    risk_usd = float(scenario.get("initial_risk_usd", 0.0) or 0.0)
    scenario["realized_r"] = (scenario["realized_pnl"] / risk_usd) if risk_usd > 0 else 0.0
    event = {
        "trade_date": trade_date,
        "reason": reason,
        "price": float(exit_price),
        "shares": qty,
        "pnl": float(pnl),
    }
    scenario.setdefault("events", []).append(event)
    scenario["last_action"] = f"{reason} @ {exit_price:.2f} on {trade_date}"
    return event


def _evaluate_tracker_scenario_bar(
    scenario: dict,
    side: str,
    trade_date: str,
    bar_row: pd.Series,
    current_anchor_levels: dict | None,
    indicator_row: pd.Series | None,
    is_entry_day: bool,
) -> list[dict]:
    if not scenario.get("tradeable"):
        scenario["status"] = "UNTRADEABLE"
        return []

    events = []
    entry_price = float(scenario.get("entry_price"))
    initial_risk_per_share = float(scenario.get("initial_risk_per_share", 0.0) or 0.0)
    direction = float(scenario.get("direction", 1.0) or 1.0)
    close_value = _coerce_float(bar_row.get("close"))
    high_value = _coerce_float(bar_row.get("high"))
    low_value = _coerce_float(bar_row.get("low"))
    if close_value is None or high_value is None or low_value is None:
        return events

    favorable_move = (high_value - entry_price) if normalize_side(side) == "LONG" else (entry_price - low_value)
    adverse_move = (entry_price - low_value) if normalize_side(side) == "LONG" else (high_value - entry_price)
    if initial_risk_per_share > 0:
        scenario["max_favorable_r"] = max(float(scenario.get("max_favorable_r", 0.0) or 0.0), favorable_move / initial_risk_per_share)
        scenario["max_adverse_r"] = max(float(scenario.get("max_adverse_r", 0.0) or 0.0), adverse_move / initial_risk_per_share)

    if is_entry_day:
        scenario["unrealized_pnl"] = (close_value - entry_price) * int(scenario.get("remaining_shares", 0)) * direction
        risk_usd = float(scenario.get("initial_risk_usd", 0.0) or 0.0)
        scenario["unrealized_r"] = (scenario["unrealized_pnl"] / risk_usd) if risk_usd > 0 else 0.0
        scenario["total_pnl"] = float(scenario.get("realized_pnl", 0.0) + scenario.get("unrealized_pnl", 0.0))
        scenario["total_r"] = float(scenario.get("realized_r", 0.0) + scenario.get("unrealized_r", 0.0))
        return events

    if not _scenario_is_open(scenario.get("status", "OPEN")):
        scenario["unrealized_pnl"] = 0.0
        scenario["unrealized_r"] = 0.0
        scenario["total_pnl"] = float(scenario.get("realized_pnl", 0.0))
        scenario["total_r"] = float(scenario.get("realized_r", 0.0))
        return events

    partial_target_level = _resolve_dynamic_level(str(scenario.get("partial_target_label") or ""), current_anchor_levels, indicator_row)
    final_target_level = _resolve_dynamic_level(str(scenario.get("final_target_label") or ""), current_anchor_levels, indicator_row)
    active_stop_label = str(scenario.get("active_stop_label") or scenario.get("stop_reference_label") or "")
    active_stop_level = _resolve_dynamic_level(active_stop_label, current_anchor_levels, indicator_row)
    scenario["active_stop_level"] = active_stop_level

    if not scenario.get("partial_taken") and partial_target_level is not None and _bar_hits_target(side, bar_row, partial_target_level):
        qty = max(1, int(scenario.get("remaining_shares", 0)) // 2)
        qty = min(qty, int(scenario.get("remaining_shares", 0)))
        event = _apply_scenario_exit_event(scenario, qty, float(partial_target_level), trade_date, "PARTIAL_TARGET")
        if event:
            events.append(event)
        scenario["partial_taken"] = True
        scenario["partial_shares"] = int(qty)
        if scenario.get("trail_after_partial_label"):
            scenario["active_stop_label"] = str(scenario.get("trail_after_partial_label"))
            scenario["close_failure_count"] = 0
        scenario["status"] = "PARTIAL" if int(scenario.get("remaining_shares", 0)) > 0 else "TARGET_HIT"

    final_target_level = _resolve_dynamic_level(str(scenario.get("final_target_label") or ""), current_anchor_levels, indicator_row)
    if int(scenario.get("remaining_shares", 0)) > 0 and final_target_level is not None and _bar_hits_target(side, bar_row, final_target_level):
        event = _apply_scenario_exit_event(
            scenario,
            int(scenario.get("remaining_shares", 0)),
            float(final_target_level),
            trade_date,
            "FINAL_TARGET",
        )
        if event:
            events.append(event)
        scenario["status"] = "TARGET_HIT"
        scenario["unrealized_pnl"] = 0.0
        scenario["unrealized_r"] = 0.0
        scenario["total_pnl"] = float(scenario.get("realized_pnl", 0.0))
        scenario["total_r"] = float(scenario.get("realized_r", 0.0))
        return events

    active_stop_label = str(scenario.get("active_stop_label") or scenario.get("stop_reference_label") or "")
    active_stop_level = _resolve_dynamic_level(active_stop_label, current_anchor_levels, indicator_row)
    scenario["active_stop_level"] = active_stop_level
    if active_stop_level is not None:
        failed = close_value <= active_stop_level if normalize_side(side) == "LONG" else close_value >= active_stop_level
        scenario["close_failure_count"] = int(scenario.get("close_failure_count", 0) + 1) if failed else 0
        if scenario["close_failure_count"] >= TRACKER_STOP_FAILURE_CLOSES and int(scenario.get("remaining_shares", 0)) > 0:
            reason = "TRAIL_STOP" if scenario.get("partial_taken") and active_stop_label != scenario.get("stop_reference_label") else "STOP_FAIL"
            event = _apply_scenario_exit_event(
                scenario,
                int(scenario.get("remaining_shares", 0)),
                float(close_value),
                trade_date,
                reason,
            )
            if event:
                events.append(event)
            scenario["status"] = "STOPPED"
            scenario["unrealized_pnl"] = 0.0
            scenario["unrealized_r"] = 0.0
            scenario["total_pnl"] = float(scenario.get("realized_pnl", 0.0))
            scenario["total_r"] = float(scenario.get("realized_r", 0.0))
            return events

    remaining = int(scenario.get("remaining_shares", 0))
    unrealized_pnl = (close_value - entry_price) * remaining * direction
    risk_usd = float(scenario.get("initial_risk_usd", 0.0) or 0.0)
    scenario["unrealized_pnl"] = float(unrealized_pnl)
    scenario["unrealized_r"] = (unrealized_pnl / risk_usd) if risk_usd > 0 else 0.0
    scenario["total_pnl"] = float(scenario.get("realized_pnl", 0.0) + scenario.get("unrealized_pnl", 0.0))
    scenario["total_r"] = float(scenario.get("realized_r", 0.0) + scenario.get("unrealized_r", 0.0))
    scenario["status"] = "PARTIAL" if scenario.get("partial_taken") else "OPEN"
    return events


def recompute_tracker_setup_record(setup: dict, df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return setup

    indicator_frame = compute_indicator_frame(df)
    current_history = calc_anchored_vwap_band_history(df, str(setup.get("anchor_date") or ""))
    previous_history = calc_anchored_vwap_band_history(df, str(setup.get("previous_anchor_date") or "")) if setup.get("previous_anchor_date") else {}
    entry_trade_date = str(setup.get("entry_trade_date") or setup.get("scan_date") or "")
    if not entry_trade_date:
        return setup

    working_scenarios = {}
    for scenario_id, original in (setup.get("scenarios") or {}).items():
        if not isinstance(original, dict):
            continue
        scenario = dict(original)
        scenario["entry_price"] = _coerce_float(setup.get("entry_price"))
        scenario["events"] = []
        scenario["remaining_shares"] = int(scenario.get("shares", 0) or 0)
        scenario["partial_taken"] = False
        scenario["partial_shares"] = 0
        scenario["realized_pnl"] = 0.0
        scenario["realized_r"] = 0.0
        scenario["unrealized_pnl"] = 0.0
        scenario["unrealized_r"] = 0.0
        scenario["total_pnl"] = 0.0
        scenario["total_r"] = 0.0
        scenario["close_failure_count"] = 0
        scenario["active_stop_label"] = scenario.get("stop_reference_label")
        scenario["active_stop_level"] = scenario.get("stop_reference_level")
        scenario["last_action"] = "Awaiting update" if scenario.get("tradeable") else "Risk too tight for $500 standard risk"
        scenario["days_held"] = 0
        scenario["max_favorable_r"] = 0.0
        scenario["max_adverse_r"] = 0.0
        scenario["status"] = "UNTRADEABLE" if not scenario.get("tradeable") else "OPEN"
        working_scenarios[scenario_id] = scenario

    trade_df = df[df["datetime"].dt.date.astype(str) >= entry_trade_date].copy().reset_index(drop=True)
    indicator_trade = indicator_frame[indicator_frame["trade_date"] >= entry_trade_date].copy().reset_index(drop=True)
    daily_marks = []

    try:
        anchor_start_date = datetime.fromisoformat(str(setup.get("anchor_date") or entry_trade_date)).date()
    except ValueError:
        anchor_start_date = datetime.fromisoformat(entry_trade_date).date()

    for idx in range(len(trade_df)):
        bar_row = trade_df.iloc[idx]
        indicator_row = indicator_trade.iloc[idx] if idx < len(indicator_trade) else None
        trade_date = bar_row["datetime"].date().isoformat()
        current_levels = current_history.get(trade_date)
        previous_levels = previous_history.get(trade_date)
        slice_df = df[
            (df["datetime"].dt.date >= anchor_start_date)
            & (df["datetime"].dt.date <= bar_row["datetime"].date())
        ]
        compression_summary = summarize_anchor_compression(
            slice_df,
            _coerce_float((current_levels or {}).get("stdev")),
            _coerce_float(indicator_row.get("atr_20")) if isinstance(indicator_row, pd.Series) else None,
        )
        feature_snapshot = build_tracker_feature_snapshot(
            normalize_side(setup.get("side")),
            bar_row,
            indicator_row,
            current_levels,
            previous_levels,
            compression_summary,
        )

        scenario_events = []
        for scenario in working_scenarios.values():
            scenario_events.extend(
                _evaluate_tracker_scenario_bar(
                    scenario,
                    normalize_side(setup.get("side")),
                    trade_date,
                    bar_row,
                    current_levels,
                    indicator_row,
                    is_entry_day=(trade_date == entry_trade_date),
                )
            )
            scenario["days_held"] = max(0, idx)

        daily_marks.append(
            {
                "trade_date": trade_date,
                "is_entry_day": bool(trade_date == entry_trade_date),
                "close": _coerce_float(bar_row.get("close")),
                "high": _coerce_float(bar_row.get("high")),
                "low": _coerce_float(bar_row.get("low")),
                "current_levels": current_levels or {},
                "previous_levels": previous_levels or {},
                "feature_snapshot": feature_snapshot,
                "scenario_events": scenario_events,
            }
        )

    setup["scenarios"] = working_scenarios
    setup["daily_marks"] = daily_marks
    setup["latest_snapshot"] = daily_marks[-1] if daily_marks else {}
    setup["open_scenario_count"] = sum(1 for scenario in working_scenarios.values() if _scenario_is_open(scenario.get("status")))
    setup["closed_scenario_count"] = sum(
        1
        for scenario in working_scenarios.values()
        if str(scenario.get("status", "")).upper() in {"STOPPED", "TARGET_HIT"}
    )
    if setup["open_scenario_count"] > 0:
        setup["setup_status"] = "OPEN"
    elif any(str(scenario.get("status", "")).upper() == "UNTRADEABLE" for scenario in working_scenarios.values()):
        setup["setup_status"] = "UNTRADEABLE"
    else:
        setup["setup_status"] = "CLOSED"
    return setup


def _flatten_tracker_scenarios(setups: dict[str, dict]) -> list[dict]:
    rows = []
    for setup in setups.values():
        for scenario in (setup.get("scenarios") or {}).values():
            rows.append(
                {
                    "setup_id": setup.get("setup_id"),
                    "scan_date": setup.get("scan_date"),
                    "symbol": setup.get("symbol"),
                    "side": setup.get("side"),
                    "priority_bucket": setup.get("priority_bucket"),
                    "entry_price": setup.get("entry_price"),
                    "anchor_date": setup.get("anchor_date"),
                    "retest_reference_level": setup.get("retest_reference_level"),
                    "compression_flag": bool(setup.get("compression_flag")),
                    "stop_reference_label": scenario.get("stop_reference_label"),
                    "stop_source_type": scenario.get("stop_source_type"),
                    "exit_template_id": scenario.get("exit_template_id"),
                    "exit_template_label": scenario.get("exit_template_label"),
                    "shares": int(scenario.get("shares", 0) or 0),
                    "initial_risk_per_share": _coerce_float(scenario.get("initial_risk_per_share")),
                    "initial_risk_usd": _coerce_float(scenario.get("initial_risk_usd")),
                    "status": scenario.get("status"),
                    "remaining_shares": int(scenario.get("remaining_shares", 0) or 0),
                    "realized_pnl": _coerce_float(scenario.get("realized_pnl")),
                    "realized_r": _coerce_float(scenario.get("realized_r")),
                    "unrealized_pnl": _coerce_float(scenario.get("unrealized_pnl")),
                    "unrealized_r": _coerce_float(scenario.get("unrealized_r")),
                    "total_pnl": _coerce_float(scenario.get("total_pnl")),
                    "total_r": _coerce_float(scenario.get("total_r")),
                    "max_favorable_r": _coerce_float(scenario.get("max_favorable_r")),
                    "max_adverse_r": _coerce_float(scenario.get("max_adverse_r")),
                    "days_held": int(scenario.get("days_held", 0) or 0),
                    "last_action": scenario.get("last_action", ""),
                }
            )
    return rows


def _flatten_tracker_daily_marks(setups: dict[str, dict]) -> list[dict]:
    rows = []
    for setup in setups.values():
        for mark in setup.get("daily_marks") or []:
            feature_snapshot = mark.get("feature_snapshot") or {}
            current_levels = mark.get("current_levels") or {}
            previous_levels = mark.get("previous_levels") or {}
            rows.append(
                {
                    "setup_id": setup.get("setup_id"),
                    "scan_date": setup.get("scan_date"),
                    "symbol": setup.get("symbol"),
                    "side": setup.get("side"),
                    "trade_date": mark.get("trade_date"),
                    "is_entry_day": bool(mark.get("is_entry_day")),
                    "close": _coerce_float(mark.get("close")),
                    "high": _coerce_float(mark.get("high")),
                    "low": _coerce_float(mark.get("low")),
                    "current_avwape": _anchor_level_value(current_levels, "AVWAPE"),
                    "current_upper_1": _anchor_level_value(current_levels, "UPPER_1"),
                    "current_lower_1": _anchor_level_value(current_levels, "LOWER_1"),
                    "current_upper_2": _anchor_level_value(current_levels, "UPPER_2"),
                    "current_lower_2": _anchor_level_value(current_levels, "LOWER_2"),
                    "current_upper_3": _anchor_level_value(current_levels, "UPPER_3"),
                    "current_lower_3": _anchor_level_value(current_levels, "LOWER_3"),
                    "previous_avwape": _anchor_level_value(previous_levels, "AVWAPE"),
                    "previous_upper_1": _anchor_level_value(previous_levels, "UPPER_1"),
                    "previous_lower_1": _anchor_level_value(previous_levels, "LOWER_1"),
                    "atr20": _coerce_float(feature_snapshot.get("atr20")),
                    "ema15": _coerce_float(feature_snapshot.get("ema15")),
                    "ema21": _coerce_float(feature_snapshot.get("ema21")),
                    "ema21_consolidation": bool(feature_snapshot.get("ema21_consolidation")),
                    "sma10": _coerce_float(feature_snapshot.get("sma10")),
                    "sma20": _coerce_float(feature_snapshot.get("sma20")),
                    "sma50": _coerce_float(feature_snapshot.get("sma50")),
                    "sma100": _coerce_float(feature_snapshot.get("sma100")),
                    "sma200": _coerce_float(feature_snapshot.get("sma200")),
                    "sma10_20_cross": feature_snapshot.get("sma10_20_cross", ""),
                    "previous_avwape_near_0_5atr": bool(feature_snapshot.get("previous_avwape_near_0_5atr")),
                    "compression_flag": bool(feature_snapshot.get("compression_flag")),
                    "compression_penalty": int(feature_snapshot.get("compression_penalty", 0) or 0),
                    "scenario_events": "; ".join(
                        f"{item.get('reason')}:{item.get('shares')}@{item.get('price')}"
                        for item in mark.get("scenario_events") or []
                    ),
                }
            )
    return rows


def build_tracker_stats_rows(scenario_rows: list[dict]) -> list[dict]:
    if not scenario_rows:
        return []

    grouped = {}
    for row in scenario_rows:
        key = f"{row.get('stop_reference_label')}__{row.get('exit_template_id')}"
        grouped.setdefault(key, []).append(row)

    stats_rows = []
    for key, rows in grouped.items():
        tradeable = [row for row in rows if int(row.get("shares", 0) or 0) > 0]
        closed = [row for row in tradeable if str(row.get("status", "")).upper() in {"STOPPED", "TARGET_HIT"}]
        open_rows = [row for row in tradeable if _scenario_is_open(row.get("status", ""))]
        closed_rs = [float(row.get("total_r", 0.0) or 0.0) for row in closed]
        total_rs = [float(row.get("total_r", 0.0) or 0.0) for row in tradeable]
        pnl_values = [float(row.get("total_pnl", 0.0) or 0.0) for row in closed]
        wins = [value for value in closed_rs if value > 0]
        stats_rows.append(
            {
                "scenario_group": key,
                "stop_reference_label": rows[0].get("stop_reference_label"),
                "exit_template_id": rows[0].get("exit_template_id"),
                "exit_template_label": rows[0].get("exit_template_label"),
                "tracked_setups": len(rows),
                "tradeable_setups": len(tradeable),
                "closed_setups": len(closed),
                "open_setups": len(open_rows),
                "win_rate_closed": (len(wins) / len(closed)) if closed else None,
                "avg_closed_r": mean(closed_rs) if closed_rs else None,
                "median_closed_r": median(closed_rs) if closed_rs else None,
                "avg_total_r": mean(total_rs) if total_rs else None,
                "avg_closed_pnl": mean(pnl_values) if pnl_values else None,
            }
        )

    stats_rows.sort(key=lambda item: (item["avg_total_r"] is None, -(item["avg_total_r"] or -9999), item["scenario_group"]))
    return stats_rows


def export_setup_tracker_views(payload: dict) -> None:
    setups = payload.get("setups", {}) if isinstance(payload, dict) else {}
    scenario_rows = _flatten_tracker_scenarios(setups)
    daily_rows = _flatten_tracker_daily_marks(setups)
    stats_rows = build_tracker_stats_rows(scenario_rows)
    payload["stats"] = stats_rows

    SETUP_SCENARIOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scenario_rows).to_csv(SETUP_SCENARIOS_FILE, index=False)
    pd.DataFrame(daily_rows).to_csv(SETUP_DAILY_FILE, index=False)
    pd.DataFrame(stats_rows).to_csv(SETUP_STATS_FILE, index=False)


def update_setup_tracker_from_scan(
    tracked_rows: list[dict],
    ai_state: dict,
    feature_rows_by_symbol: dict[str, dict],
    daily_frames_by_symbol: dict[str, pd.DataFrame],
    ib: IBApi | None,
    scan_date: str | None = None,
) -> None:
    tracker = load_setup_tracker_payload()
    symbol_map = ai_state.get("symbols", {}) if isinstance(ai_state, dict) else {}
    now_iso = datetime.now().isoformat(timespec="seconds")
    target_scan_date = str(scan_date or datetime.now().date().isoformat())

    existing_today_ids = [
        setup_id
        for setup_id, setup in list((tracker.get("setups") or {}).items())
        if isinstance(setup, dict) and str(setup.get("scan_date")) == target_scan_date
    ]
    for setup_id in existing_today_ids:
        tracker["setups"].pop(setup_id, None)

    current_setup_ids = []
    for row in tracked_rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        symbol_entry = symbol_map.get(symbol)
        if not isinstance(symbol_entry, dict):
            continue
        df = daily_frames_by_symbol.get(symbol)
        indicator_row = None
        if isinstance(df, pd.DataFrame) and not df.empty:
            indicator_frame = compute_indicator_frame(df)
            if not indicator_frame.empty:
                indicator_row = indicator_frame.iloc[-1]
        setup = build_tracker_setup_record(
            row,
            symbol_entry,
            feature_rows_by_symbol.get(symbol),
            now_iso,
            indicator_row,
        )
        if not setup:
            continue
        tracker["setups"][setup["setup_id"]] = setup
        current_setup_ids.append(setup["setup_id"])

    tracker["daily_watchlists"][target_scan_date] = {
        "updated_at": now_iso,
        "setup_ids": current_setup_ids,
        "symbols": sorted(
            {
                setup.get("symbol")
                for setup in tracker["setups"].values()
                if setup.get("setup_id") in current_setup_ids
            }
        ),
    }

    recompute_cache = {}
    for setup_id, setup in list((tracker.get("setups") or {}).items()):
        if not isinstance(setup, dict):
            continue
        symbol = str(setup.get("symbol", "")).strip().upper()
        if symbol in recompute_cache:
            df = recompute_cache[symbol]
        else:
            df = daily_frames_by_symbol.get(symbol)
            if df is None or df.empty:
                anchor_date = str(setup.get("anchor_date") or setup.get("entry_trade_date") or target_scan_date)
                days_needed = ATR_LENGTH + 220
                try:
                    anchor_date_obj = datetime.fromisoformat(anchor_date).date()
                    days_needed = max(days_needed, (datetime.now().date() - anchor_date_obj).days + 20)
                except ValueError:
                    pass
                df = fetch_daily_bars(ib, symbol, days_needed)
            recompute_cache[symbol] = df
        if df is None or df.empty:
            continue
        tracker["setups"][setup_id] = recompute_tracker_setup_record(setup, df)

    export_setup_tracker_views(tracker)
    save_setup_tracker_payload(tracker)
def get_last_daily_row_for_date(daily_rows, last_trade_date):
    target = last_trade_date.isoformat()
    for row in reversed(daily_rows):
        if row["date"] == target:
            return row
    return None


def sessions_since_date(df: pd.DataFrame, event_date: date) -> int | None:
    """
    Count trading-session candles from event_date (exclusive) through the latest
    available bar in df. Returns None if event_date is not present in df.
    """
    if df.empty or "datetime" not in df:
        return None

    session_dates = df["datetime"].dt.date
    if event_date not in set(session_dates.tolist()):
        return None

    return int((session_dates > event_date).sum())

# ============================================================================
# EARNINGS FETCH (NASDAQ + YFINANCE FALLBACK)
# ============================================================================

def fetch_earnings_for_date(date_str: str):
    cache = _load_earnings_calendar_rows_cache()
    cached_entry = cache.get(date_str) if isinstance(cache, dict) else None
    cached_rows = cached_entry.get("rows", []) if isinstance(cached_entry, dict) else []
    today_iso = datetime.now().date().isoformat()
    if isinstance(cached_entry, dict):
        if date_str < today_iso:
            return cached_rows if isinstance(cached_rows, list) else []
        if date_str == today_iso and _is_timestamp_within_minutes(
            cached_entry.get("fetched_at"),
            EARNINGS_CALENDAR_TODAY_CACHE_MAX_AGE_MINUTES,
        ):
            return cached_rows if isinstance(cached_rows, list) else []

    try:
        resp = requests.get(API_URL.format(date=date_str),
                            headers=HEADERS, timeout=10)
        resp.raise_for_status()
        rows = resp.json().get("data", {}).get("rows", []) or []
        cache[date_str] = {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "rows": rows if isinstance(rows, list) else [],
        }
        _save_earnings_calendar_rows_cache()
        return rows if isinstance(rows, list) else []
    except Exception as e:
        logging.warning(f"Failed fetch earnings for {date_str}: {e}")
        if isinstance(cached_rows, list) and cached_rows:
            logging.info(f"Using cached earnings calendar rows for {date_str} after fetch failure.")
            return cached_rows
        time.sleep(0.3)
        return []

def collect_earnings_dates(
    symbols,
    fetch_fn=fetch_earnings_for_date,
    base_sleep=0.5,
    lookback_days=MAX_LOOKBACK_DAYS,
    stop_when_complete=False,
    min_dates_per_symbol=1,
):
    """
    Return dict: sym -> sorted list of past earnings dates (YYYY-MM-DD), most recent first.

    Default behavior walks the entire lookback window so quarterly anchor history
    stays correct. Optional early-stop behavior is kept for targeted refreshes and
    dry-run validation.
    """
    normalized_symbols = sorted(
        {str(sym or "").strip().upper() for sym in symbols if str(sym or "").strip()}
    )
    symbol_dates = {sym: [] for sym in normalized_symbols}
    if not symbol_dates:
        return {}

    today = datetime.now().date()
    pending = set(normalized_symbols) if stop_when_complete else set()
    minimum_dates = max(1, int(min_dates_per_symbol))

    for delta in range(lookback_days):
        day = today - timedelta(days=delta)
        rows = fetch_fn(day.isoformat())
        if not isinstance(rows, list):
            rows = []
        for row in rows:
            sym = str(row.get("symbol", "")).strip().upper()
            if sym not in symbol_dates:
                continue
            ds = day.isoformat()
            if ds not in symbol_dates[sym]:
                symbol_dates[sym].append(ds)
            if stop_when_complete and len(symbol_dates[sym]) >= minimum_dates:
                pending.discard(sym)

        if stop_when_complete and not pending:
            logging.info("Collected required earnings history for all requested symbols; stopping early.")
            break

        if base_sleep:
            time.sleep(base_sleep)

    for sym, dates in symbol_dates.items():
        symbol_dates[sym] = _normalize_earnings_dates(dates, today=today)

    return symbol_dates


def dry_run_collect_earnings_dates_short_circuit():
    """
    Dry-run helper to validate optional short-circuit behavior.
    """
    symbols = ["AAA", "BBB", "CCC"]
    today = datetime.now().date()

    schedule = {
        today.isoformat(): [{"symbol": "AAA"}],
        (today - timedelta(days=1)).isoformat(): [{"symbol": "BBB"}],
        (today - timedelta(days=2)).isoformat(): [{"symbol": "CCC"}],
        (today - timedelta(days=3)).isoformat(): [{"symbol": "AAA"}],
    }

    calls = []

    def mock_fetch(date_str: str):
        calls.append(date_str)
        return schedule.get(date_str, [])

    results = collect_earnings_dates(
        symbols,
        fetch_fn=mock_fetch,
        base_sleep=0,
        lookback_days=10,
        stop_when_complete=True,
        min_dates_per_symbol=1,
    )

    assert len(calls) == 3, f"Expected 3 fetches before short-circuiting, got {len(calls)}"
    assert all(results[sym] for sym in symbols), f"Missing earnings data for: {results}"

    return {"calls": calls, "results": results}

def pick_current_earnings_anchor(dates):
    """
    dates: ISO strings desc (most recent first).
    Current anchor:
      - If most recent within RECENT_DAYS, use SECOND most recent as anchor
        (to avoid 'fresh earnings gap' ambiguity)
      - Else, use most recent.
    """
    if not dates:
        return None
    today = datetime.now().date()
    first_date = datetime.fromisoformat(dates[0]).date()
    if (today - first_date).days <= RECENT_DAYS and len(dates) > 1:
        return datetime.fromisoformat(dates[1]).date()
    return first_date

def pick_previous_earnings_anchor(dates):
    """
    dates: ISO strings desc (most recent first).
    Previous anchor: second most recent past earnings.
    """
    if len(dates) < 2:
        return None
    return datetime.fromisoformat(dates[1]).date()


def _earnings_dates_as_of(dates, reference_date: date) -> list[str]:
    filtered = []
    for value in dates or []:
        parsed = _parse_iso_date_or_none(value)
        if parsed is None or parsed > reference_date:
            continue
        filtered.append(parsed.isoformat())
    return _normalize_earnings_dates(filtered, today=reference_date, limit=EARNINGS_HISTORY_LIMIT)


def pick_current_earnings_anchor_for_reference_date(dates, reference_date: date):
    eligible_dates = _earnings_dates_as_of(dates, reference_date)
    if not eligible_dates:
        return None
    first_date = datetime.fromisoformat(eligible_dates[0]).date()
    if (reference_date - first_date).days <= RECENT_DAYS and len(eligible_dates) > 1:
        return datetime.fromisoformat(eligible_dates[1]).date()
    return first_date


def pick_previous_earnings_anchor_for_reference_date(dates, reference_date: date):
    eligible_dates = _earnings_dates_as_of(dates, reference_date)
    if len(eligible_dates) < 2:
        return None
    return datetime.fromisoformat(eligible_dates[1]).date()

def yf_earnings_dates(symbol: str):
    """
    Fallback using yfinance earnings_dates.
    """
    if not _can_attempt_yf_earnings_lookup(log_once=True):
        return []

    logger_state = []
    for logger_name in YF_EARNINGS_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        logger_state.append((logger, logger.level, logger.propagate))
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

    try:
        t = yf.Ticker(symbol)
        ed = t.get_earnings_dates(limit=8)
        if ed is None or getattr(ed, "empty", True):
            return []
        if not hasattr(ed, "index") or ed.index is None:
            return []
        if getattr(ed.index, "tz", None) is not None:
            ed.index = ed.index.tz_localize(None)
        past = ed[ed.index < pd.Timestamp.today().tz_localize(None)]
        sorted_past = sorted(past.index, reverse=True)
        return [d.date().isoformat() for d in sorted_past]
    except Exception as e:
        message = str(e)
        if "Import lxml" in message:
            global YF_EARNINGS_LOOKUP_DISABLED_REASON
            YF_EARNINGS_LOOKUP_DISABLED_REASON = "lxml is not installed"
            _can_attempt_yf_earnings_lookup(log_once=True)
            return []
        if any(marker in message for marker in YF_EARNINGS_NO_DATA_MARKERS):
            return []
        logging.warning(f"{symbol}: yfinance earnings lookup failed: {e}")
        return []
    finally:
        for logger, level, propagate in logger_state:
            logger.setLevel(level)
            logger.propagate = propagate


def load_or_refresh_earnings(symbols):
    """
    Maintain a rolling earnings cache that can discover new quarterly releases.

    The cache does a daily recent-window refresh across the universe to catch new
    earnings quickly, then forces a deeper rebuild for symbols whose latest known
    earnings is stale or whose history is incomplete.
    """
    cache = load_earnings_date_cache()
    symbol_cache = cache.setdefault("symbols", {})
    today = datetime.now().date()
    today_iso = today.isoformat()
    normalized_symbols = sorted(
        {str(sym or "").strip().upper() for sym in symbols if str(sym or "").strip()}
    )

    for sym in normalized_symbols:
        symbol_cache.setdefault(sym, _normalize_earnings_cache_entry({}))

    if cache.get("last_recent_refresh_on") != today_iso:
        logging.info(
            f"Refreshing recent earnings window across {len(normalized_symbols)} symbols "
            f"({EARNINGS_RECENT_DISCOVERY_WINDOW_DAYS} days)."
        )
        recent_updates = collect_earnings_dates(
            normalized_symbols,
            lookback_days=EARNINGS_RECENT_DISCOVERY_WINDOW_DAYS,
            base_sleep=0.15,
        )
        for sym in normalized_symbols:
            entry = symbol_cache.setdefault(sym, _normalize_earnings_cache_entry({}))
            entry["dates"] = _merge_earnings_dates(entry.get("dates", []), recent_updates.get(sym, []))
        cache["last_recent_refresh_on"] = today_iso

    deep_refresh_symbols = [
        sym for sym in normalized_symbols
        if _earnings_symbol_needs_deep_refresh(symbol_cache.get(sym, {}), today=today)
    ]
    if deep_refresh_symbols:
        logging.info(
            f"Refreshing deep earnings history for {len(deep_refresh_symbols)} symbol(s) "
            f"({EARNINGS_DEEP_LOOKBACK_DAYS} days)."
        )
        deep_updates = collect_earnings_dates(
            deep_refresh_symbols,
            lookback_days=EARNINGS_DEEP_LOOKBACK_DAYS,
            base_sleep=0.15,
        )
        for sym in deep_refresh_symbols:
            entry = symbol_cache.setdefault(sym, _normalize_earnings_cache_entry({}))
            entry["dates"] = _merge_earnings_dates(entry.get("dates", []), deep_updates.get(sym, []))
            entry["last_deep_refresh_on"] = today_iso

        yf_refresh_symbols = [
            sym for sym in deep_refresh_symbols
            if _earnings_symbol_needs_yf_refresh(symbol_cache.get(sym, {}), today=today)
        ]
        if yf_refresh_symbols and _can_attempt_yf_earnings_lookup(log_once=True):
            logging.info(
                f"Supplementing stale/incomplete earnings history with yfinance for {len(yf_refresh_symbols)} symbol(s)."
            )
            for sym in yf_refresh_symbols:
                entry = symbol_cache.setdefault(sym, _normalize_earnings_cache_entry({}))
                yf_dates = yf_earnings_dates(sym)
                if yf_dates:
                    entry["dates"] = _merge_earnings_dates(entry.get("dates", []), yf_dates)
                entry["last_yf_refresh_on"] = today_iso

    save_earnings_date_cache(cache)
    return {
        sym: _normalize_earnings_dates(symbol_cache.get(sym, {}).get("dates", []), today=today)
        for sym in normalized_symbols
    }

# ============================================================================
# EARNINGS GAP-ANCHOR SCANNER (D1 FOUNDATION)
# ============================================================================

EARNINGS_ANCHOR_COLUMNS = [
    "ticker",
    "side",
    "anchor_date",
    "gap_date",
    "earnings_date",
    "release_session",
    "gap_atr_multiple",
    "price",
    "avg_volume20",
    "market_cap",
    "notes",
    "source",
    "created_at",
]

ANCHOR_AVWAP_OUTPUT_FILE = MASTER_ANCHOR_AVWAP_REPORT_FILE
EARNINGS_ANCHOR_CANDIDATES_OUTPUT_FILE = EARNINGS_ANCHOR_CANDIDATES_REPORT_FILE
ANCHOR_AVWAP_SIGNAL_COLUMNS = [
    "run_date",
    "trade_date",
    "ticker",
    "list_name",
    "side",
    "anchor_date",
    "gap_date",
    "event",
    "avwap",
    "stdev",
    "level_name",
    "level_price",
]


@dataclass
class EarningsGapAnchorCandidate:
    ticker: str
    anchor_date: str
    gap_date: str
    earnings_date: str
    release_session: str
    gap_atr_multiple: float
    price: float
    avg_volume20: int
    market_cap: int
    side: str = "LONG"
    notes: str = ""
    source: str = "program"



def ensure_anchor_file(path: Path = EARNINGS_ANCHORS_FILE):
    if not path.exists():
        pd.DataFrame(columns=EARNINGS_ANCHOR_COLUMNS).to_csv(path, index=False)
        return

    df = pd.read_csv(path)
    updated = False
    for col in EARNINGS_ANCHOR_COLUMNS:
        if col not in df.columns:
            df[col] = ""
            updated = True

    if "side" in df.columns:
        normalized = df["side"].fillna("").astype(str).str.upper().replace({"": "LONG"})
        if not normalized.equals(df["side"]):
            df["side"] = normalized
            updated = True

    if updated:
        df[EARNINGS_ANCHOR_COLUMNS].to_csv(path, index=False)


def ensure_anchor_candidate_file(path: Path = EARNINGS_ANCHOR_CANDIDATES_FILE):
    if not path.exists():
        pd.DataFrame(columns=EARNINGS_ANCHOR_COLUMNS).to_csv(path, index=False)
        return

    df = pd.read_csv(path)
    updated = False
    for col in EARNINGS_ANCHOR_COLUMNS:
        if col not in df.columns:
            df[col] = ""
            updated = True

    if "side" in df.columns:
        normalized = df["side"].fillna("").astype(str).str.upper().replace({"": "LONG"})
        if not normalized.equals(df["side"]):
            df["side"] = normalized
            updated = True

    if updated:
        df[EARNINGS_ANCHOR_COLUMNS].to_csv(path, index=False)


def ensure_previous_gap_file(path: Path = PREVIOUS_GAP_UPS_FILE):
    columns = EARNINGS_ANCHOR_COLUMNS + ["archived_at"]
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return

    df = pd.read_csv(path)
    updated = False
    for col in columns:
        if col not in df.columns:
            df[col] = ""
            updated = True

    if "side" in df.columns:
        normalized = df["side"].fillna("").astype(str).str.upper().replace({"": "LONG"})
        if not normalized.equals(df["side"]):
            df["side"] = normalized
            updated = True

    if updated:
        df[columns].to_csv(path, index=False)



def infer_release_session(row: dict) -> str:
    blob = " ".join(str(v) for v in row.values() if v is not None).lower()
    if any(token in blob for token in ["after market close", "after close", "amc", "post-market"]):
        return "amc"
    if any(token in blob for token in ["before market open", "before open", "bmo", "pre-market"]):
        return "bmo"
    return "unknown"



def get_last_market_session_date() -> date | None:
    benchmark = fetch_daily_bars(None, "SPY", 10)
    if benchmark is None or benchmark.empty:
        return None
    return pd.to_datetime(benchmark["datetime"].iloc[-1]).date()


def get_recent_market_session_dates(lookback_days: int = 1):
    lookback_days = max(1, int(lookback_days))
    period_days = max(10, lookback_days * 4)
    benchmark = fetch_daily_bars(None, "SPY", period_days)
    if benchmark is None or benchmark.empty:
        return []

    unique_dates = []
    seen = set()
    for ts in reversed(pd.to_datetime(benchmark["datetime"]).tolist()):
        session_date = ts.date()
        if session_date in seen:
            continue
        seen.add(session_date)
        unique_dates.append(session_date)
        if len(unique_dates) >= lookback_days:
            break
    return unique_dates



def fetch_earnings_calendar_rows(date_obj):
    rows = fetch_earnings_for_date(date_obj.isoformat())
    cleaned = []
    for row in rows:
        symbol = str(row.get("symbol", "")).upper().strip()
        if not symbol or "^" in symbol or "/" in symbol:
            continue
        cleaned.append(row)
    return cleaned



def _get_info_with_fallbacks(ticker_obj):
    info = {}
    try:
        info = ticker_obj.get_info() or {}
    except Exception:
        try:
            info = ticker_obj.info or {}
        except Exception:
            info = {}
    return info



def has_weekly_options(symbol: str, ticker_obj=None, metadata: dict | None = None) -> bool:
    metadata = metadata if isinstance(metadata, dict) else _get_cached_symbol_metadata(symbol, ticker_obj=ticker_obj)
    expirations = []
    for value in metadata.get("options_expirations", []):
        try:
            expirations.append(pd.Timestamp(value).date())
        except Exception:
            continue

    expirations = sorted(set(expirations))
    if len(expirations) < 2:
        return False

    for prev, curr in zip(expirations, expirations[1:]):
        if (curr - prev).days <= 8:
            return True
    return False



def _coerce_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None



def _coerce_int(value):
    v = _coerce_float(value)
    if v is None:
        return None
    return int(v)



def load_latest_earnings_release_map(symbols) -> dict[str, dict | None]:
    normalized_symbols = _ordered_unique_symbols(symbols)
    if not normalized_symbols:
        return {}

    earnings_lookup = load_or_refresh_earnings(normalized_symbols)
    unique_dates = sorted({dates[0] for dates in earnings_lookup.values() if dates}, reverse=True)
    rows_by_date = {}
    for earnings_date_iso in unique_dates:
        rows = fetch_earnings_for_date(earnings_date_iso)
        rows_by_date[earnings_date_iso] = rows if isinstance(rows, list) else []

    release_map = {}
    for symbol in normalized_symbols:
        dates = earnings_lookup.get(symbol, [])
        if not dates:
            release_map[symbol] = None
            continue

        earnings_date_iso = dates[0]
        release_session = "unknown"
        for row in rows_by_date.get(earnings_date_iso, []):
            row_symbol = str(row.get("symbol", "")).strip().upper()
            if row_symbol != symbol:
                continue
            release_session = infer_release_session(row)
            break

        release_map[symbol] = {
            "earnings_date": earnings_date_iso,
            "release_session": release_session,
        }

    return release_map


def _load_symbol_daily_history_frame(symbol: str):
    ticker_obj = yf.Ticker(symbol)
    df = fetch_daily_bars(None, symbol, 130)
    if df.empty:
        logging.warning(f"{symbol}: failed history download for earnings anchor resolution.")
        return ticker_obj, pd.DataFrame()

    df = df.sort_values("datetime").reset_index(drop=True)
    df["trade_date"] = df["datetime"].dt.date
    return ticker_obj, df


def _infer_gap_index_from_history(df: pd.DataFrame, earnings_idx: int):
    candidates = []
    if earnings_idx > 0:
        same_day_gap = abs(float(df.iloc[earnings_idx]["open"]) - float(df.iloc[earnings_idx - 1]["close"]))
        candidates.append((same_day_gap, earnings_idx, "bmo_inferred"))
    if earnings_idx + 1 < len(df):
        next_day_gap = abs(float(df.iloc[earnings_idx + 1]["open"]) - float(df.iloc[earnings_idx]["close"]))
        candidates.append((next_day_gap, earnings_idx + 1, "amc_inferred"))
    if not candidates:
        return None, "unknown"
    _, gap_idx, resolved_session = max(candidates, key=lambda item: item[0])
    return gap_idx, resolved_session


def _resolve_gap_window_from_earnings_event(
    df: pd.DataFrame,
    earnings_date,
    release_session: str,
):
    earnings_idx = df.index[df["trade_date"] == earnings_date]
    if earnings_idx.empty:
        return None

    earnings_idx = int(earnings_idx[0])
    normalized_session = (release_session or "").strip().lower()
    if normalized_session == "amc":
        gap_idx = earnings_idx + 1
        resolved_session = "amc"
    elif normalized_session == "bmo":
        gap_idx = earnings_idx
        resolved_session = "bmo"
    else:
        gap_idx, resolved_session = _infer_gap_index_from_history(df, earnings_idx)

    if gap_idx is None or gap_idx <= 0 or gap_idx >= len(df):
        return None

    return {
        "earnings_idx": earnings_idx,
        "anchor_idx": gap_idx - 1,
        "gap_idx": gap_idx,
        "release_session": resolved_session,
    }


def build_earnings_anchor_candidate(
    symbol: str,
    earnings_date,
    release_session: str,
    side: str = "LONG",
    require_filters: bool = True,
    source: str = "program",
    notes: str = "",
):
    ticker_obj, df = _load_symbol_daily_history_frame(symbol)
    if df.empty or len(df) < 2:
        return None

    gap_window = _resolve_gap_window_from_earnings_event(df, earnings_date, release_session)
    if not gap_window:
        return None

    gap_idx = gap_window["gap_idx"]
    anchor_idx = gap_window["anchor_idx"]
    gap_row = df.iloc[gap_idx]
    anchor_row = df.iloc[anchor_idx]
    pre_gap = df.iloc[:gap_idx + 1]

    atr20 = get_atr20(pre_gap[["open", "high", "low", "close", "volume"]], ATR_LENGTH)
    if atr20 is None or atr20 <= 0:
        return None

    prev_close = float(anchor_row["close"])
    gap_open = float(gap_row["open"])
    gap_size = abs(gap_open - prev_close)
    gap_atr_multiple = gap_size / atr20

    avg_vol_20 = int(pre_gap["volume"].tail(20).mean()) if len(pre_gap) >= 20 else 0
    last_price = float(gap_row["close"])
    market_cap = 0

    if require_filters:
        metadata = _get_cached_symbol_metadata(symbol, ticker_obj=ticker_obj)
        market_cap = _coerce_int(metadata.get("market_cap"))

        if last_price < MIN_PRICE:
            return None
        if avg_vol_20 < MIN_AVG_VOLUME_20D:
            return None
        if market_cap is None or market_cap < MIN_MARKET_CAP:
            return None
        if not has_weekly_options(symbol, ticker_obj=ticker_obj, metadata=metadata):
            return None
        if gap_atr_multiple < MIN_GAP_ATR_MULTIPLE:
            return None

        market_cap = int(market_cap or 0)

    return EarningsGapAnchorCandidate(
        ticker=symbol,
        anchor_date=anchor_row["trade_date"].isoformat(),
        side=normalize_side(side),
        gap_date=gap_row["trade_date"].isoformat(),
        earnings_date=earnings_date.isoformat(),
        release_session=gap_window["release_session"],
        gap_atr_multiple=round(gap_atr_multiple, 3),
        price=round(last_price, 3),
        avg_volume20=avg_vol_20,
        market_cap=market_cap,
        notes=str(notes or "").strip(),
        source=str(source or "").strip() or "program",
    )


def evaluate_earnings_gap_candidate(symbol: str, earnings_date, release_session: str):
    return build_earnings_anchor_candidate(
        symbol,
        earnings_date,
        release_session,
        side="LONG",
        require_filters=True,
        source="earnings_scan",
    )



def append_anchor_candidates(candidates, path: Path = EARNINGS_ANCHORS_FILE):
    ensure_anchor_file(path)
    existing = pd.read_csv(path)
    if existing.empty:
        existing = pd.DataFrame(columns=EARNINGS_ANCHOR_COLUMNS)

    for col in EARNINGS_ANCHOR_COLUMNS:
        if col not in existing.columns:
            existing[col] = ""

    existing_keys = {
        (str(r["ticker"]).upper(), str(r["anchor_date"]))
        for _, r in existing.iterrows()
        if str(r.get("ticker", "")).strip() and str(r.get("anchor_date", "")).strip()
    }

    new_rows = []
    now_iso = datetime.now().isoformat(timespec="seconds")
    for candidate in candidates:
        key = (candidate.ticker.upper(), candidate.anchor_date)
        if key in existing_keys:
            continue
        row = asdict(candidate)
        row["side"] = normalize_side(row.get("side", "LONG"))
        row["created_at"] = now_iso
        new_rows.append(row)

    if not new_rows:
        return 0

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined["side"] = combined["side"].apply(normalize_side)
    combined = combined[EARNINGS_ANCHOR_COLUMNS]
    combined.to_csv(path, index=False)
    return len(new_rows)



def _candidate_to_row(candidate: EarningsGapAnchorCandidate, created_at: str) -> dict:
    row = asdict(candidate)
    row["ticker"] = str(row.get("ticker", "")).strip().upper()
    row["side"] = normalize_side(row.get("side", "LONG"))
    row["notes"] = str(row.get("notes", "")).strip()
    row["source"] = str(row.get("source", "")).strip() or "program"
    row["created_at"] = created_at
    return {col: row.get(col, "") for col in EARNINGS_ANCHOR_COLUMNS}


def _candidate_from_row(row: dict) -> EarningsGapAnchorCandidate:
    return EarningsGapAnchorCandidate(
        ticker=str(row.get("ticker", "")).strip().upper(),
        anchor_date=str(row.get("anchor_date", "")).strip(),
        gap_date=str(row.get("gap_date", "")).strip(),
        earnings_date=str(row.get("earnings_date", "")).strip(),
        release_session=str(row.get("release_session", "")).strip(),
        gap_atr_multiple=float(_coerce_float(row.get("gap_atr_multiple")) or 0.0),
        price=float(_coerce_float(row.get("price")) or 0.0),
        avg_volume20=int(_coerce_int(row.get("avg_volume20")) or 0),
        market_cap=int(_coerce_int(row.get("market_cap")) or 0),
        side=normalize_side(row.get("side", "LONG")),
        notes=str(row.get("notes", "")).strip(),
        source=str(row.get("source", "")).strip() or "program",
    )


def _format_symbols_for_tc2000(symbols) -> str:
    cleaned = set()
    for symbol in symbols or []:
        text = str(symbol).strip().upper()
        if not text:
            continue
        cleaned.add(text.split()[0])
    cleaned = sorted(cleaned)
    return ", ".join(cleaned) if cleaned else "None"


def _ordered_unique_symbols(symbols) -> list[str]:
    ordered = []
    seen = set()
    for symbol in symbols or []:
        text = str(symbol).strip().upper()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return ordered


def _format_symbol_group(symbols) -> str:
    ordered = _ordered_unique_symbols(symbols)
    return ", ".join(ordered) if ordered else "None"


def _extract_symbols_from_text(text: str) -> list[str]:
    matches = WATCHLIST_SYMBOL_RE.findall(str(text or "").upper())
    return [
        symbol
        for symbol in _ordered_unique_symbols(matches)
        if symbol not in WATCHLIST_SKIP_TOKENS
    ]


def load_latest_anchor_candidate_rows(
    path: Path = EARNINGS_ANCHOR_CANDIDATES_FILE,
) -> dict[str, dict]:
    ensure_anchor_candidate_file(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logging.warning(f"Failed to load earnings anchor candidates from {path}: {exc}")
        return {}

    if df.empty:
        return {}

    for col in EARNINGS_ANCHOR_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if "created_at" not in df.columns:
        df["created_at"] = ""

    df = df.fillna("")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["anchor_date"] = df["anchor_date"].astype(str).str.strip()
    df["created_at"] = df["created_at"].astype(str).str.strip()
    df = df.sort_values(by=["created_at", "anchor_date", "ticker"], ascending=[False, False, True], kind="stable")

    latest_rows = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or ticker in latest_rows:
            continue
        latest_rows[ticker] = {col: row.get(col, "") for col in EARNINGS_ANCHOR_COLUMNS}
    return latest_rows


def resolve_bulk_anchor_candidates(
    symbols,
    default_side: str = "LONG",
    focus_side_map: dict[str, str] | None = None,
) -> dict:
    ordered_symbols = _ordered_unique_symbols(symbols)
    if not ordered_symbols:
        return {
            "candidates": [],
            "unresolved": [],
            "candidate_matches": 0,
            "earnings_matches": 0,
        }

    normalized_side_map = {
        str(symbol).strip().upper(): normalize_side(side)
        for symbol, side in (focus_side_map or {}).items()
        if str(symbol).strip()
    }
    default_side = normalize_side(default_side)

    candidate_rows = load_latest_anchor_candidate_rows()
    resolved_candidates = []
    unresolved_symbols = []
    candidate_matches = 0
    earnings_matches = 0

    latest_release_map = load_latest_earnings_release_map(ordered_symbols)
    for symbol in ordered_symbols:
        resolved_side = normalized_side_map.get(symbol, default_side)
        release_info = latest_release_map.get(symbol)
        if release_info:
            earnings_date_iso = release_info.get("earnings_date")
            release_session = release_info.get("release_session", "unknown")
            try:
                earnings_date = datetime.fromisoformat(str(earnings_date_iso)).date()
            except ValueError:
                earnings_date = None

            if earnings_date is not None:
                candidate = build_earnings_anchor_candidate(
                    symbol,
                    earnings_date,
                    release_session,
                    side=resolved_side,
                    require_filters=False,
                    source="bulk_import",
                    notes="Bulk import from ticker group",
                )
                if candidate and candidate.anchor_date:
                    resolved_candidates.append(candidate)
                    earnings_matches += 1
                    continue

        candidate_row = candidate_rows.get(symbol)
        if candidate_row:
            candidate = _candidate_from_row(candidate_row)
            if candidate.anchor_date:
                candidate.side = resolved_side
                if not candidate.source:
                    candidate.source = "earnings_scan"
                resolved_candidates.append(candidate)
                candidate_matches += 1
                continue

        unresolved_symbols.append(symbol)

    return {
        "candidates": resolved_candidates,
        "unresolved": unresolved_symbols,
        "candidate_matches": candidate_matches,
        "earnings_matches": earnings_matches,
    }


def write_anchor_candidates_output(
    candidates,
    sessions,
    lookback_days: int,
    csv_path: Path = EARNINGS_ANCHOR_CANDIDATES_FILE,
    report_path: Path = EARNINGS_ANCHOR_CANDIDATES_OUTPUT_FILE,
):
    ensure_anchor_candidate_file(csv_path)

    created_at = datetime.now().isoformat(timespec="seconds")
    rows = [
        _candidate_to_row(candidate, created_at)
        for candidate in sorted(candidates, key=lambda item: (normalize_side(item.side), item.ticker, item.anchor_date))
    ]
    df_candidates = pd.DataFrame(rows, columns=EARNINGS_ANCHOR_COLUMNS)
    df_candidates.to_csv(csv_path, index=False)

    long_symbols = [row["ticker"] for row in rows if row.get("side") == "LONG"]
    short_symbols = [row["ticker"] for row in rows if row.get("side") == "SHORT"]

    report_lines = [
        "Earnings gap anchor candidates",
        f"Generated at {created_at}",
        f"Sessions scanned: {len(sessions)}",
        f"Lookback sessions requested: {max(1, int(lookback_days))}",
        f"Qualified candidates: {len(rows)}",
        "",
        "TC2000 copy/paste",
        f"LONG: {_format_symbols_for_tc2000(long_symbols)}",
        f"SHORT: {_format_symbols_for_tc2000(short_symbols)}",
        "",
        "Candidate rows",
    ]

    if not rows:
        report_lines.append("None")
    else:
        for row in rows:
            report_lines.append(
                f"{row['ticker']:<6} {row['side']:<5} anchor={row['anchor_date']} gap={row['gap_date']} "
                f"earnings={row['earnings_date']} gap_atr={row['gap_atr_multiple']} "
                f"price={row['price']} avg_vol20={row['avg_volume20']} release={row['release_session']} "
                f"source={row['source']}"
            )
            notes = str(row.get("notes", "")).strip()
            if notes:
                report_lines.append(f"  notes={notes}")

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return rows


def scan_last_session_earnings_for_anchors(lookback_days: int = 1):
    sessions = get_recent_market_session_dates(lookback_days)
    if not sessions:
        logging.error("Could not determine market session dates. Earnings anchor scan aborted.")
        return []

    candidates = []
    for session in sessions:
        rows = fetch_earnings_calendar_rows(session)
        logging.info(f"Found {len(rows)} raw earnings rows for {session.isoformat()}.")

        for row in rows:
            symbol = str(row.get("symbol", "")).upper().strip()
            if not symbol:
                continue

            release_session = infer_release_session(row)
            candidate = evaluate_earnings_gap_candidate(symbol, session, release_session)
            if candidate:
                candidates.append(candidate)
                logging.info(
                    f"{symbol}: qualified (anchor={candidate.anchor_date}, gap={candidate.gap_date}, "
                    f"gap_atr={candidate.gap_atr_multiple})."
                )

    write_anchor_candidates_output(candidates, sessions, lookback_days)
    logging.info(
        f"Earnings gap anchor scan complete. Sessions={len(sessions)}, qualified={len(candidates)}, "
        f"candidate_file={EARNINGS_ANCHOR_CANDIDATES_FILE}, "
        f"report={EARNINGS_ANCHOR_CANDIDATES_OUTPUT_FILE}"
    )
    return candidates


# ============================================================================
# IBKR API
# ============================================================================

class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.ready = {}

    def historicalData(self, reqId, bar):
        self.data.setdefault(reqId, []).append({
            "time":   bar.date,
            "open":   bar.open,
            "high":   bar.high,
            "low":    bar.low,
            "close":  bar.close,
            "volume": bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        self.ready[reqId] = True

    def error(self, reqId, code, msg):
        if code not in (2104, 2106, 2158, 2176):
            logging.error(f"IB Error {code}[{reqId}]: {msg}")

def create_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c

def fetch_daily_bars_from_yahoo(symbol: str, days: int) -> pd.DataFrame:
    """Fetch daily OHLCV bars from Yahoo Finance."""

    period = f"{max(days, ATR_LENGTH + 5)}d"
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    except Exception as e:
        logging.error(f"{symbol}: failed to download daily bars from Yahoo: {e}")
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    if df is None or df.empty:
        logging.warning(f"{symbol}: no daily data returned from Yahoo.")
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df = df.reset_index()

    # Handle potential MultiIndex columns (e.g., when yfinance returns columns like
    # ('Open', 'SPGI')) by flattening to the last element of the tuple and then
    # normalising to lowercase for easier downstream handling.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]

    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df.rename(columns={date_col: "datetime"}, inplace=True)
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        logging.error(f"{symbol}: missing expected columns from Yahoo response: {sorted(missing)}")
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df = df.dropna(subset=list(required))
    df = df.sort_values("datetime")
    return _normalize_daily_bar_frame(df[["datetime", "open", "high", "low", "close", "volume"]])


def _fetch_live_daily_bars(ib: IBApi | None, symbol: str, days: int) -> pd.DataFrame:
    if ib is None:
        return fetch_daily_bars_from_yahoo(symbol, days)

    # Try IBKR first
    try:
        reqId = int(time.time() * 1000) % (2**31 - 1)
        ib.data[reqId] = []
        ib.ready[reqId] = False

        if days > 365:
            dur = f"{max(1, days // 365)} Y"
        else:
            dur = f"{max(2, days)} D"

        ib.reqHistoricalData(
            reqId,
            create_contract(symbol),
            "",
            dur,
            "1 day",
            "TRADES",
            1,
            1,
            False,
            []
        )

        for _ in range(60):
            if ib.ready.get(reqId):
                break
            time.sleep(0.5)

        bars = ib.data.pop(reqId, [])
        ib.ready.pop(reqId, None)

        df = pd.DataFrame(bars)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d", errors="coerce")
            df = df.sort_values("datetime").reset_index(drop=True)
            return _normalize_daily_bar_frame(df)
        logging.warning(f"{symbol}: no daily bars returned from IBKR, falling back to Yahoo.")
    except Exception as e:
        logging.error(f"{symbol}: IBKR daily fetch failed ({e}), falling back to Yahoo.")

    return fetch_daily_bars_from_yahoo(symbol, days)


def fetch_daily_bars(ib: IBApi | None, symbol: str, days: int) -> pd.DataFrame:
    normalized_symbol = str(symbol or "").strip().upper()
    requested_days = max(int(days), ATR_LENGTH + 5)
    cached = _load_cached_daily_bar_frame(normalized_symbol)
    cache_has_history = _daily_bar_cache_covers_history(cached, requested_days)

    if cache_has_history and _daily_bar_cache_is_recent(normalized_symbol):
        return cached.copy()

    refresh_days = (
        requested_days
        if not cache_has_history
        else min(requested_days, max(ATR_LENGTH + 5, DAILY_BAR_CACHE_RECENT_REFRESH_DAYS))
    )
    fresh = _fetch_live_daily_bars(ib, normalized_symbol, refresh_days)
    if fresh is not None and not fresh.empty:
        merged = _merge_daily_bar_frames(cached, fresh)
        _write_cached_daily_bar_frame(normalized_symbol, merged)
        return merged.copy()

    if not cached.empty:
        logging.info(f"{normalized_symbol}: using cached daily bars because live refresh was unavailable.")
        return cached.copy()

    return _empty_daily_bar_frame()

# ============================================================================
# AVWAP CALCULATION
# ============================================================================

def calc_anchored_vwap_bands(df: pd.DataFrame, anchor_idx: int):
    """
    Anchored VWAP + 1/2/3σ bands from anchor_idx → end.
    TradingView-style volume-weighted stdev.
    """
    cumVol = 0.0
    cumVP = 0.0
    cumSD = 0.0

    for i in range(anchor_idx, len(df)):
        row = df.iloc[i]
        v = float(row["volume"])
        if v <= 0:
            continue
        tp = (row["open"] + row["high"] + row["low"] + row["close"]) / 4.0
        cumVol += v
        cumVP += tp * v
        vw = cumVP / cumVol
        dev = tp - vw
        cumSD += dev * dev * v

    if cumVol == 0:
        return float("nan"), float("nan"), {}

    final_vwap = cumVP / cumVol
    final_stdev = (cumSD / cumVol) ** 0.5

    bands = {
        "UPPER_1": final_vwap + final_stdev,
        "LOWER_1": final_vwap - final_stdev,
        "UPPER_2": final_vwap + 2 * final_stdev,
        "LOWER_2": final_vwap - 2 * final_stdev,
        "UPPER_3": final_vwap + 3 * final_stdev,
        "LOWER_3": final_vwap - 3 * final_stdev,
    }
    return final_vwap, final_stdev, bands


def _to_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def normalize_side(value: str) -> str:
    raw = str(value or "").strip().upper()
    return "SHORT" if raw == "SHORT" else "LONG"


def _build_ordered_level_points(anchor_meta: dict) -> list[tuple[str, float]]:
    if not anchor_meta:
        return []

    bands = anchor_meta.get("bands", {}) if anchor_meta else {}
    level_points = []
    for name, level in [
        ("LOWER_3", bands.get("LOWER_3")),
        ("LOWER_2", bands.get("LOWER_2")),
        ("LOWER_1", bands.get("LOWER_1")),
        ("VWAP", anchor_meta.get("vwap")),
        ("UPPER_1", bands.get("UPPER_1")),
        ("UPPER_2", bands.get("UPPER_2")),
        ("UPPER_3", bands.get("UPPER_3")),
    ]:
        if level is None:
            continue
        try:
            if pd.isna(level):
                continue
        except TypeError:
            pass
        level_points.append((name, float(level)))

    level_points.sort(key=lambda item: item[1])
    return level_points


def classify_position_by_band(last_close: float, anchor_meta: dict, side: str = "LONG"):
    """Return the active break level for the current side."""
    if last_close is None:
        return None

    level_points = _build_ordered_level_points(anchor_meta)
    if not level_points:
        return None

    side = normalize_side(side)
    if side == "SHORT":
        for name, level in level_points:
            if last_close <= level:
                return name
        return level_points[-1][0]

    category = level_points[0][0]
    for name, level in level_points:
        if last_close >= level:
            category = name
        else:
            break
    return category


def get_band_context(last_close: float, anchor_meta: dict, side: str = "LONG") -> dict:
    level_points = _build_ordered_level_points(anchor_meta)
    if last_close is None or not level_points:
        return {
            "active_level": None,
            "nearby_levels": [],
            "zone": "",
        }

    nearby_levels = []
    for name, level in level_points:
        if abs(last_close - level) <= 1e-9:
            nearby_levels = [name]
            break

    if not nearby_levels:
        if last_close <= level_points[0][1]:
            nearby_levels = [level_points[0][0]]
        elif last_close >= level_points[-1][1]:
            nearby_levels = [level_points[-1][0]]
        else:
            for idx in range(len(level_points) - 1):
                lower_name, lower_level = level_points[idx]
                upper_name, upper_level = level_points[idx + 1]
                if lower_level <= last_close <= upper_level:
                    nearby_levels = [lower_name, upper_name]
                    break

    side = normalize_side(side)
    if side == "SHORT" and len(nearby_levels) == 2:
        nearby_levels = list(reversed(nearby_levels))

    return {
        "active_level": classify_position_by_band(last_close, anchor_meta, side=side),
        "nearby_levels": nearby_levels,
        "zone": " to ".join(nearby_levels),
    }


def select_primary_cross_signal(
    df: pd.DataFrame,
    side: str,
    prefix: str,
    vwap_value: float,
    bands: dict,
):
    side = normalize_side(side)
    candidates = []

    if side == "LONG":
        cross_tests = [
            ("VWAP", vwap_value, cross_up_through_level),
            ("UPPER_1", bands.get("UPPER_1"), cross_up_through_level),
            ("UPPER_2", bands.get("UPPER_2"), cross_up_through_level),
            ("UPPER_3", bands.get("UPPER_3"), cross_up_through_level),
        ]
        direction = "UP"
    else:
        cross_tests = [
            ("VWAP", vwap_value, cross_down_through_level),
            ("LOWER_1", bands.get("LOWER_1"), cross_down_through_level),
            ("LOWER_2", bands.get("LOWER_2"), cross_down_through_level),
            ("LOWER_3", bands.get("LOWER_3"), cross_down_through_level),
        ]
        direction = "DOWN"

    for level_name, level_price, test_fn in cross_tests:
        if level_price is None:
            continue
        if test_fn(df, level_price):
            candidates.append((f"{prefix}CROSS_{direction}_{level_name}", level_price))

    if not candidates:
        return None
    return candidates[-1]

# ============================================================================
# ATR + BOUNCE / CROSS LOGIC
# ============================================================================

def get_atr20(df: pd.DataFrame, length: int = ATR_LENGTH):
    if df is None or df.empty or len(df) < length + 1:
        return None

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    trs = []
    prev_close = closes[0]
    for i in range(1, len(df)):
        h = highs[i]
        l = lows[i]
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = closes[i]

    if len(trs) < length:
        return None

    atr_series = pd.Series(trs).rolling(length).mean()
    atr = atr_series.iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return None
    return float(atr)

def bounce_up_at_level(df: pd.DataFrame, level: float) -> bool:
    if level is None or pd.isna(level) or len(df) < ATR_LENGTH + 3:
        return False
    atr = get_atr20(df)
    if atr is None:
        return False
    eps = max(BOUNCE_ATR_TOL_PCT * atr, BOUNCE_LEVEL_ATR_TOL_PCT * atr)
    push = ATR_MULT * atr
    B, C = df.iloc[-2], df.iloc[-1]

    # Same-day touch-and-reclaim (bounce happens on latest bar)
    touched_today = abs(C.low - level) <= eps
    confirm_today = C.close >= level
    confirm_push_today = C.close >= level + push
    if touched_today and (confirm_today or confirm_push_today):
        return True

    # Two-day pattern: yesterday touched/reclaimed, today confirms follow-through
    touched = abs(B.low - level) <= eps
    reclaimed = B.close >= level
    confirm = C.close > B.close and (C.close >= level or C.close >= level + push)
    return bool(touched and reclaimed and confirm)

def bounce_down_at_level(df: pd.DataFrame, level: float) -> bool:
    if level is None or pd.isna(level) or len(df) < ATR_LENGTH + 3:
        return False
    atr = get_atr20(df)
    if atr is None:
        return False
    eps = max(BOUNCE_ATR_TOL_PCT * atr, BOUNCE_LEVEL_ATR_TOL_PCT * atr)
    push = ATR_MULT * atr
    B, C = df.iloc[-2], df.iloc[-1]

    # Same-day touch-and-reject (bounce happens on latest bar)
    touched_today = abs(C.high - level) <= eps
    confirm_today = C.close <= level
    confirm_push_today = C.close <= level - push
    if touched_today and (confirm_today or confirm_push_today):
        return True

    # Two-day pattern: yesterday touched/rejected, today confirms follow-through
    touched = abs(B.high - level) <= eps
    rejected = B.close <= level
    confirm = C.close < B.close and (C.close <= level or C.close <= level - push)
    return bool(touched and rejected and confirm)

def cross_up_through_level(df: pd.DataFrame, level: float) -> bool:
    if level is None or pd.isna(level) or len(df) < ATR_LENGTH + 3:
        return False
    atr = get_atr20(df)
    if atr is None:
        return False
    eps = ATR_MULT * atr
    push = ATR_MULT * atr
    B, C = df.iloc[-2], df.iloc[-1]
    below_then_above = (
        B["close"] <= level - eps and
        C["close"] >= level + push and
        C["close"] > B["close"]
    )
    return bool(below_then_above)

def cross_down_through_level(df: pd.DataFrame, level: float) -> bool:
    if level is None or pd.isna(level) or len(df) < ATR_LENGTH + 3:
        return False
    atr = get_atr20(df)
    if atr is None:
        return False
    eps = ATR_MULT * atr
    push = ATR_MULT * atr
    B, C = df.iloc[-2], df.iloc[-1]
    above_then_below = (
        B["close"] >= level + eps and
        C["close"] <= level - push and
        C["close"] < B["close"]
    )
    return bool(above_then_below)

# ============================================================================
# MULTI-DAY PATTERNS
# ============================================================================

def load_history():
    return load_json(HISTORY_FILE, default={})

def save_history(history):
    save_json(HISTORY_FILE, history)

def trim_history(history):
    today = datetime.now().date()
    cutoff = today - timedelta(days=HISTORY_DAYS_TO_KEEP)
    for sym, entries in list(history.items()):
        new_entries = [e for e in entries
                       if datetime.fromisoformat(e["date"]).date() >= cutoff]
        history[sym] = new_entries

def compute_multi_day_patterns(symbol, side, today_events, prev_events):
    patterns = []
    prev_set = set(prev_events or [])
    today_set = set(today_events or [])

    for e_prev in prev_set:
        # focus on bounce-type prior events
        if not (e_prev.startswith("PREV_BOUNCE_") or e_prev.startswith("BOUNCE_")):
            continue
        for e_today in today_set:
            if side == "LONG" and e_today.startswith("CROSS_UP_"):
                patterns.append(f"MD_{e_prev}_TO_{e_today}")
            elif side == "SHORT" and e_today.startswith("CROSS_DOWN_"):
                patterns.append(f"MD_{e_prev}_TO_{e_today}")
    # dedupe
    return sorted(set(patterns))


def sort_events_for_output(events):
    """Sort events so similar types are grouped in the output file."""
    def _sort_key(event):
        label_key = event_label_sort_key(event[2])
        return (
            label_key[0],
            label_key[1],
            event[0],
            event[2],
            event[1],
        )

    return sorted(events, key=_sort_key)


def event_label_sort_key(label: str):
    def group_rank(lbl: str) -> int:
        if lbl.startswith("PREV_"):
            return 0
        if lbl.startswith("MD_"):
            return 2
        return 1

    def level_rank(lbl: str) -> int:
        for idx, token in enumerate(EVENT_LEVEL_SORT_ORDER):
            if token in lbl:
                return idx
        return len(EVENT_LEVEL_SORT_ORDER)

    return (group_rank(label), level_rank(label), label)


def format_signal_label(signal: str) -> str:
    return signal.replace("VWAP", "AVWAPE").replace("_", " ")


def build_priority_setup_summary(
    symbol: str,
    side: str,
    events_today: list[str],
    all_events: list[str],
    trend_label: str,
    favorite_zone: str | None,
    recent_band_extension_days: int = 0,
    recent_second_band_test_days: int = 0,
    second_band_penalty: int = 0,
    breakout_5d: bool = False,
    retest_followthrough: bool = False,
    retest_reference_level: str = "",
    retest_note: str = "",
    extension_note: str = "",
) -> dict:
    current_weights = FAVORITE_CURRENT_SIGNALS.get(side, {})
    context_weights = FAVORITE_CONTEXT_SIGNALS.get(side, {})

    favorite_signals = sorted(evt for evt in events_today if evt in current_weights)
    context_signals = sorted(evt for evt in events_today if evt in context_weights)
    score = sum(current_weights.get(evt, 0) for evt in favorite_signals)
    score += sum(context_weights.get(evt, 0) for evt in context_signals)
    trend_is_aligned = (
        (side == "LONG" and trend_label == "UP")
        or (side == "SHORT" and trend_label == "DOWN")
    )

    if favorite_zone:
        score += PRIORITY_FAVORITE_ZONE_SCORE_BONUS

    if retest_followthrough:
        score += PRIORITY_RETEST_FOLLOWTHROUGH_SCORE_BONUS
        score += PRIORITY_RETEST_LEVEL_SCORE_BONUS.get(retest_reference_level or "", 0)
        if favorite_zone:
            score += PRIORITY_RETEST_ZONE_CONFLUENCE_SCORE_BONUS
        if trend_is_aligned:
            score += PRIORITY_RETEST_TREND_ALIGNMENT_SCORE_BONUS

    if breakout_5d:
        score += 14

    if recent_band_extension_days <= 2:
        score += 8
    else:
        score -= min(18, (recent_band_extension_days - 2) * 6)

    score -= max(0, int(second_band_penalty or 0))

    if trend_is_aligned:
        score += 8

    if favorite_signals and any(evt.startswith("MD_") for evt in all_events):
        score += 5

    return {
        "symbol": symbol,
        "side": side,
        "score": score,
        "favorite_signals": favorite_signals,
        "context_signals": context_signals,
        "favorite_zone": favorite_zone,
        "trend_20d": trend_label,
        "has_favorite_signal": bool(favorite_signals),
        "recent_band_extension_days": recent_band_extension_days,
        "recent_second_band_test_days": int(recent_second_band_test_days or 0),
        "second_band_penalty": int(second_band_penalty or 0),
        "breakout_5d": breakout_5d,
        "retest_followthrough": retest_followthrough,
        "retest_reference_level": retest_reference_level,
        "retest_note": retest_note,
        "extension_note": extension_note,
    }


def compute_major_sma_levels(df: pd.DataFrame) -> dict[str, float]:
    if df is None or df.empty or "close" not in df.columns:
        return {}

    closes = pd.to_numeric(df["close"], errors="coerce").dropna().reset_index(drop=True)
    if closes.empty:
        return {}

    sma_levels = {}
    for period in PRIORITY_SMA_PERIODS:
        if len(closes) < period:
            continue
        value = closes.rolling(period).mean().iloc[-1]
        if pd.notna(value):
            sma_levels[f"SMA_{period}"] = float(value)
    return sma_levels


def _directional_atr_distance(last_close: float, level: float, atr20: float, side: str):
    if last_close is None or level is None or atr20 is None or atr20 <= 0:
        return None

    side = normalize_side(side)
    if side == "SHORT":
        delta = float(last_close) - float(level)
    else:
        delta = float(level) - float(last_close)

    if delta < 0:
        return None
    return float(delta) / float(atr20)


def _format_obstacle_labels(obstacles: list[dict], limit: int = 4) -> str:
    if not obstacles:
        return "None"

    parts = []
    for entry in obstacles[:limit]:
        label = entry.get("label", "?")
        atr_distance = entry.get("atr_distance")
        if atr_distance is None:
            parts.append(label)
        else:
            parts.append(f"{label}@{atr_distance:.1f} ATR")
    if len(obstacles) > limit:
        parts.append(f"+{len(obstacles) - limit} more")
    return ", ".join(parts)


def _find_trendline_pivots(df: pd.DataFrame, price_col: str, mode: str, window: int = PRIORITY_TRENDLINE_PIVOT_WINDOW):
    if df is None or df.empty or price_col not in df.columns or len(df) < (window * 2 + 1):
        return []

    values = pd.to_numeric(df[price_col], errors="coerce").tolist()
    pivots = []
    for idx in range(window, len(values) - window):
        center = values[idx]
        if center is None or pd.isna(center):
            continue
        local = values[idx - window: idx + window + 1]
        local_clean = [value for value in local if value is not None and not pd.isna(value)]
        if len(local_clean) != len(local):
            continue

        if mode == "high":
            if center != max(local_clean):
                continue
            if sum(1 for value in local_clean if value == center) > 1:
                continue
        else:
            if center != min(local_clean):
                continue
            if sum(1 for value in local_clean if value == center) > 1:
                continue

        pivots.append(
            {
                "idx": idx,
                "value": float(center),
                "datetime": df.iloc[idx]["datetime"],
                "date": df.iloc[idx]["datetime"].date().isoformat(),
            }
        )
    return pivots


def _compute_trendline_angle_deg(x1: int, y1: float, x2: int, y2: float, bar_count: int, log_range: float):
    if bar_count <= 1 or log_range <= 0 or x2 <= x1:
        return None

    dx_norm = float(x2 - x1) / float(bar_count - 1)
    dy_norm = abs(float(y2 - y1)) / float(log_range)
    if dx_norm <= 0:
        return None
    return float(math.degrees(math.atan2(dy_norm, dx_norm)))


def _describe_trendline_candidate(candidate: dict | None) -> str:
    if not candidate:
        return ""
    break_suffix = ""
    bars_since_break = candidate.get("bars_since_break")
    if bars_since_break is not None:
        bar_label = "bar" if int(bars_since_break) == 1 else "bars"
        break_suffix = f" break={int(bars_since_break)} {bar_label} ago"
    return (
        f"{candidate.get('type', '?')} {candidate.get('start_date', '?')} -> {candidate.get('end_date', '?')} "
        f"angle={candidate.get('angle_deg', 0.0):.1f}deg touches={candidate.get('touch_count', 0)} "
        f"line={candidate.get('current_line_price', 0.0):.2f} dist={candidate.get('atr_distance', 0.0):.2f} ATR"
        f"{break_suffix}"
    )


def find_directional_trendline_candidate(
    df: pd.DataFrame,
    side: str,
    last_close: float,
    atr20: float,
) -> dict:
    side = normalize_side(side)
    if df is None or df.empty or last_close is None or atr20 is None or atr20 <= 0:
        return {
            "trendline_candidate": None,
            "trendline_note": "",
            "trendline_within_alert_range": False,
            "trendline_break_candidate": None,
            "trendline_break_note": "",
            "trendline_break_recent": False,
        }

    work_df = df.tail(PRIORITY_TRENDLINE_LOOKBACK_BARS).copy().reset_index(drop=True)
    if work_df.empty or len(work_df) < (PRIORITY_TRENDLINE_PIVOT_WINDOW * 2 + 3):
        return {
            "trendline_candidate": None,
            "trendline_note": "",
            "trendline_within_alert_range": False,
            "trendline_break_candidate": None,
            "trendline_break_note": "",
            "trendline_break_recent": False,
        }

    price_col = "high" if side == "LONG" else "low"
    mode = "high" if side == "LONG" else "low"
    pivots = _find_trendline_pivots(work_df, price_col, mode)
    if len(pivots) < 2:
        return {
            "trendline_candidate": None,
            "trendline_note": "",
            "trendline_within_alert_range": False,
            "trendline_break_candidate": None,
            "trendline_break_note": "",
            "trendline_break_recent": False,
        }

    highs = pd.to_numeric(work_df["high"], errors="coerce").tolist()
    lows = pd.to_numeric(work_df["low"], errors="coerce").tolist()
    closes = pd.to_numeric(work_df["close"], errors="coerce").tolist()
    datetimes = pd.to_datetime(work_df["datetime"])
    log_high = [math.log(value) for value in highs if value and value > 0]
    log_low = [math.log(value) for value in lows if value and value > 0]
    if not log_high or not log_low:
        return {
            "trendline_candidate": None,
            "trendline_note": "",
            "trendline_within_alert_range": False,
            "trendline_break_candidate": None,
            "trendline_break_note": "",
            "trendline_break_recent": False,
        }
    log_range = max(log_high) - min(log_low)
    if log_range <= 0:
        return {
            "trendline_candidate": None,
            "trendline_note": "",
            "trendline_within_alert_range": False,
            "trendline_break_candidate": None,
            "trendline_break_note": "",
            "trendline_break_recent": False,
        }

    best_candidate = None
    best_break_candidate = None
    line_touch_tol = float(atr20) * PRIORITY_TRENDLINE_TOUCH_TOL_ATR
    last_idx = len(work_df) - 1

    for left_idx in range(len(pivots) - 1):
        pivot_a = pivots[left_idx]
        for right_idx in range(left_idx + 1, len(pivots)):
            pivot_b = pivots[right_idx]
            x1 = int(pivot_a["idx"])
            x2 = int(pivot_b["idx"])
            if x2 - x1 < PRIORITY_TRENDLINE_MIN_SEPARATION_BARS:
                continue

            price_a = float(pivot_a["value"])
            price_b = float(pivot_b["value"])
            if side == "LONG" and not (price_b < price_a):
                continue
            if side == "SHORT" and not (price_b > price_a):
                continue

            y1 = math.log(price_a)
            y2 = math.log(price_b)
            angle_deg = _compute_trendline_angle_deg(x1, y1, x2, y2, len(work_df), log_range)
            if angle_deg is None:
                continue
            if not (PRIORITY_TRENDLINE_MIN_ANGLE_DEG <= angle_deg <= PRIORITY_TRENDLINE_MAX_ANGLE_DEG):
                continue

            slope = (y2 - y1) / float(x2 - x1)
            touch_count = 0
            invalidated = False
            break_idx = None
            break_distance = 0.0

            for idx in range(x1, len(work_df)):
                line_log = y1 + slope * (idx - x1)
                line_price = math.exp(line_log)
                close_value = closes[idx]
                if side == "LONG":
                    candle_extreme = float(highs[idx])
                    overshoot = candle_extreme - line_price
                    close_delta = (
                        float(close_value) - line_price
                        if close_value is not None and not pd.isna(close_value)
                        else None
                    )
                else:
                    candle_extreme = float(lows[idx])
                    overshoot = line_price - candle_extreme
                    close_delta = (
                        line_price - float(close_value)
                        if close_value is not None and not pd.isna(close_value)
                        else None
                    )

                if abs(candle_extreme - line_price) <= line_touch_tol:
                    touch_count += 1

                if idx <= x2:
                    continue

                if close_delta is not None and close_delta > line_touch_tol:
                    break_idx = idx
                    break_distance = close_delta
                    break
                if overshoot > line_touch_tol:
                    invalidated = True
                    break

            if invalidated or touch_count < 2:
                continue

            current_line_price = math.exp(y1 + slope * (last_idx - x1))
            if break_idx is not None:
                bars_since_break = last_idx - break_idx
                if bars_since_break > PRIORITY_TRENDLINE_BREAK_RECENT_BARS:
                    continue

                if side == "LONG":
                    if current_line_price >= float(last_close):
                        continue
                    atr_distance = (float(last_close) - current_line_price) / float(atr20)
                else:
                    if current_line_price <= float(last_close):
                        continue
                    atr_distance = (current_line_price - float(last_close)) / float(atr20)

                if atr_distance > PRIORITY_TRENDLINE_BREAK_MAX_ATR:
                    continue

                candidate = {
                    "type": "H-break" if side == "LONG" else "L-break",
                    "start_date": pivot_a["date"],
                    "end_date": pivot_b["date"],
                    "break_date": datetimes.iloc[break_idx].date().isoformat(),
                    "bars_since_break": int(bars_since_break),
                    "start_idx": x1,
                    "end_idx": x2,
                    "break_idx": int(break_idx),
                    "current_line_price": float(current_line_price),
                    "atr_distance": float(atr_distance),
                    "break_atr_distance": float(break_distance / float(atr20)),
                    "angle_deg": float(angle_deg),
                    "touch_count": int(touch_count),
                    "bar_count": int(len(work_df)),
                    "lookback_start": datetimes.iloc[0].date().isoformat(),
                    "lookback_end": datetimes.iloc[-1].date().isoformat(),
                    "slope_log_per_bar": float(slope),
                }

                if best_break_candidate is None:
                    best_break_candidate = candidate
                    continue

                current_key = (
                    candidate["bars_since_break"],
                    candidate["atr_distance"],
                    -candidate["touch_count"],
                    abs(candidate["angle_deg"] - 45.0),
                )
                best_key = (
                    best_break_candidate["bars_since_break"],
                    best_break_candidate["atr_distance"],
                    -best_break_candidate["touch_count"],
                    abs(best_break_candidate["angle_deg"] - 45.0),
                )
                if current_key < best_key:
                    best_break_candidate = candidate
                continue

            if side == "LONG":
                if current_line_price <= float(last_close):
                    continue
                atr_distance = (current_line_price - float(last_close)) / float(atr20)
            else:
                if current_line_price >= float(last_close):
                    continue
                atr_distance = (float(last_close) - current_line_price) / float(atr20)

            candidate = {
                "type": "H-" if side == "LONG" else "L+",
                "start_date": pivot_a["date"],
                "end_date": pivot_b["date"],
                "start_idx": x1,
                "end_idx": x2,
                "current_line_price": float(current_line_price),
                "atr_distance": float(atr_distance),
                "angle_deg": float(angle_deg),
                "touch_count": int(touch_count),
                "bar_count": int(len(work_df)),
                "lookback_start": datetimes.iloc[0].date().isoformat(),
                "lookback_end": datetimes.iloc[-1].date().isoformat(),
                "slope_log_per_bar": float(slope),
                "max_overshoot": 0.0,
            }

            if best_candidate is None:
                best_candidate = candidate
                continue

            current_key = (candidate["atr_distance"], -candidate["touch_count"], abs(candidate["angle_deg"] - 45.0))
            best_key = (best_candidate["atr_distance"], -best_candidate["touch_count"], abs(best_candidate["angle_deg"] - 45.0))
            if current_key < best_key:
                best_candidate = candidate

    return {
        "trendline_candidate": best_candidate,
        "trendline_note": _describe_trendline_candidate(best_candidate),
        "trendline_within_alert_range": bool(
            best_candidate and best_candidate.get("atr_distance") is not None
            and best_candidate["atr_distance"] <= PRIORITY_TRENDLINE_ALERT_ATR
        ),
        "trendline_break_candidate": best_break_candidate,
        "trendline_break_note": _describe_trendline_candidate(best_break_candidate),
        "trendline_break_recent": bool(best_break_candidate),
    }


def assess_priority_directional_obstacles(
    side: str,
    last_close: float,
    atr20: float,
    sma_levels: dict[str, float],
    previous_anchor_meta: dict | None,
) -> dict:
    sma_obstacles = []
    sma_blockers = []
    sma_penalty = 0

    for label, level in sorted(sma_levels.items(), key=lambda item: int(item[0].split("_")[-1])):
        atr_distance = _directional_atr_distance(last_close, level, atr20, side)
        if atr_distance is None or atr_distance > PRIORITY_SMA_WARN_ATR:
            continue
        obstacle = {
            "label": label,
            "level": float(level),
            "atr_distance": float(atr_distance),
        }
        sma_obstacles.append(obstacle)
        if atr_distance <= PRIORITY_SMA_DQ_ATR:
            sma_blockers.append(obstacle)
        else:
            sma_penalty += 10 if atr_distance <= 2.5 else 6

    prev_anchor_obstacles = []
    prev_avwap_obstacles = []
    prev_band_obstacles = []
    prev_anchor_penalty = 0
    if previous_anchor_meta:
        prev_levels = [("PREV_AVWAPE", previous_anchor_meta.get("vwap"))]
        prev_bands = previous_anchor_meta.get("bands", {}) or {}
        prev_levels.extend(
            [
                ("PREV_UPPER_1", prev_bands.get("UPPER_1")),
                ("PREV_LOWER_1", prev_bands.get("LOWER_1")),
            ]
        )

        for label, level in prev_levels:
            atr_distance = _directional_atr_distance(last_close, level, atr20, side)
            if atr_distance is None or atr_distance > PRIORITY_PREV_AVWAP_WARN_ATR:
                continue
            obstacle = {
                "label": label,
                "level": float(level),
                "atr_distance": float(atr_distance),
            }
            prev_anchor_obstacles.append(obstacle)
            if label == "PREV_AVWAPE":
                prev_avwap_obstacles.append(obstacle)
            else:
                prev_band_obstacles.append(obstacle)
            if atr_distance <= 1.0:
                prev_anchor_penalty += 10
            elif atr_distance <= 2.0:
                prev_anchor_penalty += 6
            else:
                prev_anchor_penalty += 3

    ranking_blocked = bool(sma_blockers)
    notes = []
    if ranking_blocked:
        notes.append(f"SMA DQ: {_format_obstacle_labels(sma_blockers)}")
    elif sma_obstacles:
        notes.append(f"SMAs ahead: {_format_obstacle_labels(sma_obstacles)}")
    if prev_avwap_obstacles:
        notes.append(f"Prev AVWAPE ahead: {_format_obstacle_labels(prev_avwap_obstacles)}")
    if prev_band_obstacles:
        notes.append(f"Prev stdev bands ahead: {_format_obstacle_labels(prev_band_obstacles)}")

    return {
        "ranking_blocked": ranking_blocked,
        "ranking_block_reason": _format_obstacle_labels(sma_blockers) if ranking_blocked else "",
        "sma_obstacles": sma_obstacles,
        "sma_blockers": sma_blockers,
        "previous_anchor_obstacles": prev_anchor_obstacles,
        "previous_anchor_path_clear": bool(previous_anchor_meta) and not prev_anchor_obstacles,
        "score_penalty": sma_penalty + prev_anchor_penalty,
        "score_penalty_sma": sma_penalty,
        "score_penalty_previous_anchor": prev_anchor_penalty,
        "ranking_note": " | ".join(notes),
    }


def refine_priority_rows_with_directional_filters(
    priority_rows: list[dict],
    ai_state: dict,
    ib: IBApi,
) -> None:
    symbol_map = ai_state.setdefault("symbols", {})
    priority_candidates = [
        row for row in priority_rows
        if row.get("has_favorite_signal") or row.get("favorite_zone") or row.get("retest_followthrough")
    ]

    for row in priority_candidates:
        symbol = row.get("symbol")
        symbol_entry = symbol_map.get(symbol, {})
        last_close = symbol_entry.get("last_close")
        atr20 = symbol_entry.get("atr20")
        side = row.get("side") or symbol_entry.get("side") or "LONG"

        sma_df = fetch_daily_bars(ib, symbol, PRIORITY_SMA_LOOKBACK_DAYS)
        sma_levels = compute_major_sma_levels(sma_df)
        trendline_summary = find_directional_trendline_candidate(
            sma_df,
            side=side,
            last_close=last_close,
            atr20=atr20,
        )
        obstacle_summary = assess_priority_directional_obstacles(
            side=side,
            last_close=last_close,
            atr20=atr20,
            sma_levels=sma_levels,
            previous_anchor_meta=symbol_entry.get("previous_anchor"),
        )
        trendline_score_bonus = (
            PRIORITY_TRENDLINE_BREAK_SCORE_BONUS
            if trendline_summary.get("trendline_break_recent")
            else 0
        )
        clean_path_score_bonus = (
            PRIORITY_RETEST_CLEAR_PATH_SCORE_BONUS
            if row.get("retest_followthrough") and obstacle_summary.get("previous_anchor_path_clear")
            else 0
        )
        bonus_notes = []
        if trendline_score_bonus:
            bonus_notes.append(f"Trendline break +{trendline_score_bonus}")
        if clean_path_score_bonus:
            bonus_notes.append(f"Clean prev anchor path +{clean_path_score_bonus}")

        row["base_score"] = row["score"]
        row["score"] = (
            row["score"]
            - obstacle_summary["score_penalty"]
            + trendline_score_bonus
            + clean_path_score_bonus
        )
        row.update(obstacle_summary)
        row.update(trendline_summary)
        row["trendline_score_bonus"] = trendline_score_bonus
        row["clean_path_score_bonus"] = clean_path_score_bonus
        row["score_bonus_note"] = " | ".join(bonus_notes)
        row["sma_levels"] = sma_levels

        if row.get("ranking_blocked"):
            logging.info(
                f"{symbol}: excluded from priority rankings due to nearby SMAs "
                f"({row.get('ranking_block_reason')})."
            )
        elif row.get("ranking_note"):
            logging.info(f"{symbol}: priority ranking adjusted by filters ({row['ranking_note']}).")
        if row.get("trendline_within_alert_range"):
            logging.info(f"{symbol}: nearby trendline candidate found ({row.get('trendline_note')}).")
        if row.get("trendline_break_recent"):
            logging.info(
                f"{symbol}: recent trendline break candidate found "
                f"({row.get('trendline_break_note')}); added +{trendline_score_bonus} ranking bonus."
            )
        if clean_path_score_bonus:
            logging.info(
                f"{symbol}: retest setup has clear previous-anchor path; "
                f"added +{clean_path_score_bonus} ranking bonus."
            )

        symbol_entry["priority_score"] = row["score"]
        symbol_entry["priority_base_score"] = row["base_score"]
        symbol_entry["priority_ranking_blocked"] = bool(row.get("ranking_blocked"))
        symbol_entry["priority_ranking_note"] = row.get("ranking_note", "")
        symbol_entry["priority_sma_levels"] = sma_levels
        symbol_entry["priority_sma_obstacles"] = list(row.get("sma_obstacles") or [])
        symbol_entry["priority_previous_anchor_obstacles"] = list(row.get("previous_anchor_obstacles") or [])
        symbol_entry["priority_previous_anchor_path_clear"] = bool(row.get("previous_anchor_path_clear"))
        symbol_entry["priority_trendline_candidate"] = row.get("trendline_candidate")
        symbol_entry["priority_trendline_note"] = row.get("trendline_note", "")
        symbol_entry["priority_trendline_within_alert_range"] = bool(row.get("trendline_within_alert_range"))
        symbol_entry["priority_trendline_break_candidate"] = row.get("trendline_break_candidate")
        symbol_entry["priority_trendline_break_note"] = row.get("trendline_break_note", "")
        symbol_entry["priority_trendline_break_recent"] = bool(row.get("trendline_break_recent"))
        symbol_entry["priority_trendline_score_bonus"] = int(row.get("trendline_score_bonus", 0) or 0)
        symbol_entry["priority_clean_path_score_bonus"] = int(row.get("clean_path_score_bonus", 0) or 0)
        symbol_entry["priority_score_bonus_note"] = row.get("score_bonus_note", "")


def apply_final_priority_buckets(
    priority_rows: list[dict],
    ai_state: dict,
    csv_rows: list[dict],
    feature_rows_by_symbol: dict[str, dict],
) -> None:
    priority_map = {row["symbol"]: row for row in priority_rows}
    symbol_map = ai_state.setdefault("symbols", {})

    for symbol, symbol_entry in symbol_map.items():
        row = priority_map.get(symbol)
        priority_bucket = ""
        is_favorite_setup = False
        is_near_favorite_zone = False
        if row and not row.get("ranking_blocked"):
            if row.get("has_favorite_signal") or (
                row.get("retest_followthrough") and row.get("previous_anchor_path_clear")
            ):
                priority_bucket = "favorite_setup"
                is_favorite_setup = True
            elif row.get("favorite_zone") or row.get("retest_followthrough"):
                priority_bucket = "near_favorite_zone"
                is_near_favorite_zone = True

        symbol_entry["priority_bucket"] = priority_bucket
        symbol_entry["is_favorite_setup"] = is_favorite_setup
        symbol_entry["is_near_favorite_zone"] = is_near_favorite_zone
        if row:
            row["priority_bucket"] = priority_bucket
            row["is_favorite_setup"] = is_favorite_setup
            row["is_near_favorite_zone"] = is_near_favorite_zone

        feature_row = feature_rows_by_symbol.get(symbol)
        if row and feature_row is not None:
            feature_row["priority_score"] = row.get("score", feature_row.get("priority_score"))

    for record in csv_rows:
        row = priority_map.get(record.get("symbol"))
        if not row or row.get("ranking_blocked"):
            record["priority_bucket"] = ""
            record["is_favorite_setup"] = False
            record["is_near_favorite_zone"] = False
            continue

        if row.get("has_favorite_signal") or (
            row.get("retest_followthrough") and row.get("previous_anchor_path_clear")
        ):
            record["priority_bucket"] = "favorite_setup"
            record["is_favorite_setup"] = True
            record["is_near_favorite_zone"] = False
        elif row.get("favorite_zone") or row.get("retest_followthrough"):
            record["priority_bucket"] = "near_favorite_zone"
            record["is_favorite_setup"] = False
            record["is_near_favorite_zone"] = True
        else:
            record["priority_bucket"] = ""
            record["is_favorite_setup"] = False
            record["is_near_favorite_zone"] = False

def write_priority_setup_report(path: Path, priority_rows: list[dict]) -> None:
    favorites = sorted(
        [
            row for row in priority_rows
            if (
                row["has_favorite_signal"]
                or (row.get("retest_followthrough") and row.get("previous_anchor_path_clear"))
            )
            and not row.get("ranking_blocked")
        ],
        key=lambda row: (-row["score"], row["symbol"]),
    )
    watchlist = sorted(
        [
            row for row in priority_rows
            if (
                not row["has_favorite_signal"]
                and not (row.get("retest_followthrough") and row.get("previous_anchor_path_clear"))
                and (row["favorite_zone"] or row.get("retest_followthrough"))
                and not row.get("ranking_blocked")
            )
        ],
        key=lambda row: (-row["score"], row["symbol"]),
    )

    def _write_rows(handle, title: str, rows: list[dict]) -> None:
        handle.write(f"{title}\n")
        handle.write("-" * len(title) + "\n")
        if not rows:
            handle.write("None\n\n")
            return
        for row in rows:
            signals = ", ".join(format_signal_label(evt) for evt in row["favorite_signals"]) or "None"
            context = ", ".join(format_signal_label(evt) for evt in row["context_signals"]) or "None"
            zone = row["favorite_zone"] or "None"
            extension_days = row.get("recent_band_extension_days", 0)
            second_band_tests = row.get("recent_second_band_test_days", 0)
            breakout_text = "Y" if row.get("breakout_5d") else "N"
            retest_text = "Y" if row.get("retest_followthrough") else "N"
            ranking_note = row.get("ranking_note", "")
            score_bonus_note = row.get("score_bonus_note", "")
            retest_note = row.get("retest_note", "")
            extension_note = row.get("extension_note", "")
            compression_note = row.get("compression_note", "")
            trendline_note = row.get("trendline_note", "")
            trendline_break_note = row.get("trendline_break_note", "")
            handle.write(
                f"{row['symbol']:<6} {row['side']:<5} score={row['score']:<3} "
                f"signals={signals} | context={context} | zone={zone} | trend={row['trend_20d']} "
                f"| ext_days={extension_days} | 2nd_tests={second_band_tests} "
                f"| 5d_breakout={breakout_text} | retest={retest_text}\n"
            )
            if ranking_note:
                handle.write(f"  filters={ranking_note}\n")
            if score_bonus_note:
                handle.write(f"  bonus={score_bonus_note}\n")
            if retest_note:
                handle.write(f"  retest={retest_note}\n")
            if extension_note:
                handle.write(f"  extension={extension_note}\n")
            if compression_note:
                handle.write(f"  compression={compression_note}\n")
            if trendline_note:
                prefix = "trendline_alert" if row.get("trendline_within_alert_range") else "trendline"
                handle.write(f"  {prefix}={trendline_note}\n")
            if trendline_break_note:
                handle.write(f"  trendline_break={trendline_break_note}\n")
        handle.write("\n")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Master AVWAP priority setups\n")
        handle.write(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        handle.write("Focus: AVWAPE bounces, directional retest continuations, and UPPER_1 / LOWER_1 crosses\n\n")
        _write_rows(handle, "Best current favorite setups", favorites)
        _write_rows(handle, "Near favorite zones", watchlist)


def write_master_avwap_focus_feed(path: Path, priority_rows: list[dict], ai_state: dict) -> None:
    ranked_rows = sorted(priority_rows, key=lambda row: (-row["score"], row["symbol"]))
    favorite_rows = [
        row for row in ranked_rows
        if (
            row["has_favorite_signal"]
            or (row.get("retest_followthrough") and row.get("previous_anchor_path_clear"))
        )
        and not row.get("ranking_blocked")
    ]
    near_rows = [
        row for row in ranked_rows
        if (
            not row["has_favorite_signal"]
            and not (row.get("retest_followthrough") and row.get("previous_anchor_path_clear"))
            and (row["favorite_zone"] or row.get("retest_followthrough"))
            and not row.get("ranking_blocked")
        )
    ]

    def _build_entry(row: dict, bucket: str, rank: int) -> dict:
        symbol = row["symbol"]
        symbol_state = ai_state.get("symbols", {}).get(symbol, {})
        return {
            "symbol": symbol,
            "side": row["side"],
            "priority_bucket": bucket,
            "priority_rank": rank,
            "priority_score": row["score"],
            "favorite_zone": row.get("favorite_zone") or "",
            "trend_20d": row.get("trend_20d") or "SIDEWAYS",
            "favorite_signals": list(row.get("favorite_signals") or []),
            "favorite_context_signals": list(row.get("context_signals") or []),
            "recent_band_extension_days": int(row.get("recent_band_extension_days", 0) or 0),
            "recent_second_band_test_days": int(row.get("recent_second_band_test_days", 0) or 0),
            "second_band_penalty": int(row.get("second_band_penalty", 0) or 0),
            "breakout_5d": bool(row.get("breakout_5d")),
            "retest_followthrough": bool(row.get("retest_followthrough")),
            "retest_reference_level": row.get("retest_reference_level") or "",
            "retest_note": row.get("retest_note") or "",
            "extension_note": row.get("extension_note") or "",
            "compression_flag": bool(row.get("compression_flag")),
            "compression_penalty": int(row.get("compression_penalty", 0) or 0),
            "compression_note": row.get("compression_note") or "",
            "trendline_note": row.get("trendline_note") or "",
            "trendline_break_note": row.get("trendline_break_note") or "",
            "trendline_break_recent": bool(row.get("trendline_break_recent")),
            "trendline_score_bonus": int(row.get("trendline_score_bonus", 0) or 0),
            "has_bounce_event_today": bool(symbol_state.get("has_bounce_event_today")),
            "last_trade_date": symbol_state.get("last_trade_date"),
        }

    favorites = [_build_entry(row, "favorite_setup", idx + 1) for idx, row in enumerate(favorite_rows)]
    near_favorites = [_build_entry(row, "near_favorite_zone", idx + 1) for idx, row in enumerate(near_rows)]
    symbol_map = {
        entry["symbol"]: entry
        for entry in favorites + near_favorites
    }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_date": datetime.now().date().isoformat(),
        "favorites": favorites,
        "near_favorite_zones": near_favorites,
        "symbols": symbol_map,
    }
    save_json(path, payload)


def closes_between_bands(
    df: pd.DataFrame,
    lower: float,
    upper: float,
    min_days: int = 4,
) -> bool:
    if df is None or df.empty or lower is None or upper is None or min_days <= 0:
        return False
    if len(df) < min_days:
        return False
    low, high = sorted([lower, upper])
    recent = df.tail(min_days)
    closes = recent["close"].tolist()
    if any(pd.isna(value) for value in closes):
        return False
    return all(low <= float(value) <= high for value in closes)


def write_stdev_range_report(path: Path, range_hits: dict, cross_hits: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("AVWAP 2nd-3rd stdev multi-day range + 2nd stdev crosses/bounces\n")
        f.write(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        long_range = ", ".join(sorted(set(range_hits.get("long", [])))) or "None"
        short_range = ", ".join(sorted(set(range_hits.get("short", [])))) or "None"
        f.write("Traded between 2nd and 3rd stdev for >= 4 days (current anchors)\n")
        f.write(f"Longs (UPPER_2 to UPPER_3): {long_range}\n")
        f.write(f"Shorts (LOWER_3 to LOWER_2): {short_range}\n\n")

        long_cross = ", ".join(sorted(set(cross_hits.get("long", [])))) or "None"
        short_cross = ", ".join(sorted(set(cross_hits.get("short", [])))) or "None"
        f.write("Crosses/bounces through 2nd stdev (current anchors)\n")
        f.write(f"Longs crossing/bouncing at UPPER_2: {long_cross}\n")
        f.write(f"Shorts crossing/bouncing at LOWER_2: {short_cross}\n")


def write_tradingview_report(
    path: Path,
    priority_rows: list[dict],
    event_buckets: dict,
    range_buckets: dict,
    stdev_range_hits: dict,
    stdev_cross_hits: dict,
) -> None:
    favorites = sorted(
        [
            row for row in priority_rows
            if (
                row["has_favorite_signal"]
                or (row.get("retest_followthrough") and row.get("previous_anchor_path_clear"))
            )
            and not row.get("ranking_blocked")
        ],
        key=lambda row: (-row["score"], row["symbol"]),
    )
    near_favorites = sorted(
        [
            row for row in priority_rows
            if (
                not row["has_favorite_signal"]
                and not (row.get("retest_followthrough") and row.get("previous_anchor_path_clear"))
                and (row["favorite_zone"] or row.get("retest_followthrough"))
                and not row.get("ranking_blocked")
            )
        ],
        key=lambda row: (-row["score"], row["symbol"]),
    )

    def _symbols(rows, side: str) -> str:
        values = [row["symbol"] for row in rows if row.get("side") == side]
        return _format_symbols_for_tc2000(values)

    def _write_block(handle, title: str, lines: list[tuple[str, str]]) -> None:
        handle.write(f"{title}\n")
        handle.write("-" * len(title) + "\n")
        if not lines:
            handle.write("None\n\n")
            return
        for label, values in lines:
            handle.write(f"{label}: {values}\n")
        handle.write("\n")

    event_lines = []
    for lbl in sorted(event_buckets.keys(), key=event_label_sort_key):
        display_label = format_signal_label(lbl)
        for side in ("LONG", "SHORT"):
            values = _format_symbols_for_tc2000(event_buckets[lbl].get(side, []))
            if values == "None":
                continue
            event_lines.append((f"{display_label} {side}", values))

    range_lines = [
        ("Longs between AVWAP and UPPER_1", _format_symbols_for_tc2000(range_buckets.get("long_avwap_to_upper_1", []))),
        ("Longs between UPPER_1 and UPPER_2", _format_symbols_for_tc2000(range_buckets.get("long_upper_1_to_upper_2", []))),
        ("Shorts between AVWAP and LOWER_1", _format_symbols_for_tc2000(range_buckets.get("short_avwap_to_lower_1", []))),
        ("Shorts between LOWER_1 and LOWER_2", _format_symbols_for_tc2000(range_buckets.get("short_lower_1_to_lower_2", []))),
    ]

    stdev_lines = [
        ("Longs between UPPER_2 and UPPER_3", _format_symbols_for_tc2000(stdev_range_hits.get("long", []))),
        ("Shorts between LOWER_2 and LOWER_3", _format_symbols_for_tc2000(stdev_range_hits.get("short", []))),
        ("Longs crossing or bouncing UPPER_2", _format_symbols_for_tc2000(stdev_cross_hits.get("long", []))),
        ("Shorts crossing or bouncing LOWER_2", _format_symbols_for_tc2000(stdev_cross_hits.get("short", []))),
    ]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Master AVWAP TradingView lists\n")
        handle.write(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        _write_block(
            handle,
            "Best current favorite setups",
            [("LONG", _symbols(favorites, "LONG")), ("SHORT", _symbols(favorites, "SHORT"))],
        )
        _write_block(
            handle,
            "Near favorite zones",
            [("LONG", _symbols(near_favorites, "LONG")), ("SHORT", _symbols(near_favorites, "SHORT"))],
        )
        _write_block(handle, "Event tickers", event_lines)
        _write_block(handle, "Price ranges (current anchors)", range_lines)
        _write_block(handle, "Stdev 2-3 groups", stdev_lines)



def _archive_expired_anchor_rows(df_current: pd.DataFrame, today_run: date) -> pd.DataFrame:
    ensure_previous_gap_file()
    if df_current.empty:
        return df_current

    keep_rows = []
    moved_rows = []
    for _, row in df_current.iterrows():
        gap_date_str = str(row.get("gap_date", "")).strip()
        if not gap_date_str:
            keep_rows.append(row)
            continue
        try:
            gap_date = datetime.fromisoformat(gap_date_str).date()
        except ValueError:
            keep_rows.append(row)
            continue

        age_days = (today_run - gap_date).days
        if age_days > 15:
            row_dict = row.to_dict()
            row_dict["side"] = normalize_side(row_dict.get("side", "LONG"))
            row_dict["archived_at"] = datetime.now().isoformat(timespec="seconds")
            moved_rows.append(row_dict)
        else:
            keep_rows.append(row)

    if moved_rows:
        prev_df = pd.read_csv(PREVIOUS_GAP_UPS_FILE)
        if prev_df.empty:
            prev_df = pd.DataFrame(columns=EARNINGS_ANCHOR_COLUMNS + ["archived_at"])
        prev_df = pd.concat([prev_df, pd.DataFrame(moved_rows)], ignore_index=True)
        prev_df["side"] = prev_df["side"].apply(normalize_side)
        prev_df = prev_df[EARNINGS_ANCHOR_COLUMNS + ["archived_at"]]
        prev_df.to_csv(PREVIOUS_GAP_UPS_FILE, index=False)

    if keep_rows:
        kept_df = pd.DataFrame(keep_rows)
    else:
        kept_df = pd.DataFrame(columns=df_current.columns)
    return kept_df


def _build_anchor_levels(df: pd.DataFrame, anchor_date_iso: str):
    try:
        anchor_date = datetime.fromisoformat(anchor_date_iso).date()
    except ValueError:
        return None

    idxs = df.index[df["datetime"].dt.date == anchor_date]
    if idxs.empty:
        return None
    anchor_idx = int(idxs[0])
    avwap, stdev, bands = calc_anchored_vwap_bands(df, anchor_idx)
    if pd.isna(avwap) or not bands:
        return None
    return float(avwap), float(stdev), {k: float(v) for k, v in bands.items()}


def run_anchor_watchlist_scan(archive_expired: bool = False) -> list[dict]:
    ensure_anchor_file()
    ensure_previous_gap_file()

    today_run = datetime.now().date()
    current_df = pd.read_csv(EARNINGS_ANCHORS_FILE)
    previous_df = pd.read_csv(PREVIOUS_GAP_UPS_FILE)

    if current_df.empty and previous_df.empty:
        ANCHOR_AVWAP_OUTPUT_FILE.write_text("No anchor watchlist rows to process.\n", encoding="utf-8")
        return []

    for frame in (current_df, previous_df):
        for col in EARNINGS_ANCHOR_COLUMNS:
            if col not in frame.columns:
                frame[col] = ""
        frame["side"] = frame["side"].apply(normalize_side)

    if archive_expired:
        current_df = _archive_expired_anchor_rows(current_df, today_run)
        current_df = current_df[EARNINGS_ANCHOR_COLUMNS] if not current_df.empty else pd.DataFrame(columns=EARNINGS_ANCHOR_COLUMNS)
        current_df.to_csv(EARNINGS_ANCHORS_FILE, index=False)
        previous_df = pd.read_csv(PREVIOUS_GAP_UPS_FILE)

    all_rows = []
    if not current_df.empty:
        temp = current_df.copy()
        temp["list_name"] = "active_gap_ups"
        all_rows.append(temp)
    if not previous_df.empty:
        temp = previous_df.copy()
        temp["list_name"] = "previous_gap_ups"
        all_rows.append(temp)

    merged = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if merged.empty:
        ANCHOR_AVWAP_OUTPUT_FILE.write_text("No anchor watchlist rows to process.\n", encoding="utf-8")
        return []

    ib = connect_daily_data_client(client_id=1004, startup_wait=1.0)

    events = []
    signal_rows = []

    for _, row in merged.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        side = normalize_side(row.get("side", "LONG"))
        anchor_date_iso = str(row.get("anchor_date", "")).strip()
        if not ticker or not anchor_date_iso:
            continue

        days_needed = ATR_LENGTH + 30
        try:
            anchor_date = datetime.fromisoformat(anchor_date_iso).date()
            days_needed = max(days_needed, (today_run - anchor_date).days + 10)
        except ValueError:
            pass

        df = fetch_daily_bars(ib, ticker, days_needed)
        if df.empty:
            continue

        levels = _build_anchor_levels(df, anchor_date_iso)
        if not levels:
            continue
        avwap, stdev, _ = levels
        last_trade_date = df["datetime"].iloc[-1].date().isoformat()

        named_levels = {
            "AVWAP": avwap,
        }

        def add_anchor_event(name: str, level_name: str):
            level_price = named_levels.get(level_name)
            events.append({
                "ticker": ticker,
                "list_name": row.get("list_name", ""),
                "side": side,
                "anchor_date": anchor_date_iso,
                "gap_date": str(row.get("gap_date", "")),
                "event": name,
                "trade_date": last_trade_date,
                "level_name": level_name,
                "level_price": _to_float(level_price),
                "avwap": _to_float(avwap),
                "stdev": _to_float(stdev),
            })

        if side == "LONG":
            if bounce_up_at_level(df, avwap):
                add_anchor_event("BOUNCE_UP_AVWAP", "AVWAP")
        else:
            if bounce_down_at_level(df, avwap):
                add_anchor_event("BOUNCE_DOWN_AVWAP", "AVWAP")

    disconnect_daily_data_client(ib)

    now_iso = datetime.now().isoformat(timespec="seconds")
    if events:
        lines = [f"Anchor AVWAP bounce events generated at {now_iso}", "=" * 80]
        for item in sorted(events, key=lambda x: (x["ticker"], x["event"])):
            lines.append(
                f"{item['ticker']} [{item['list_name']}] {item['side']} {item['event']} "
                f"anchor={item['anchor_date']} gap={item['gap_date']} trade_date={item['trade_date']}"
            )
            signal_rows.append({
                "run_date": today_run.isoformat(),
                "trade_date": item["trade_date"],
                "ticker": item["ticker"],
                "list_name": item["list_name"],
                "side": item["side"],
                "anchor_date": item["anchor_date"],
                "gap_date": item["gap_date"],
                "event": item["event"],
                "avwap": item["avwap"],
                "stdev": item["stdev"],
                "level_name": item["level_name"],
                "level_price": item["level_price"],
            })
        ANCHOR_AVWAP_OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        ANCHOR_AVWAP_OUTPUT_FILE.write_text(
            f"Anchor AVWAP scan completed at {now_iso}. No AVWAP bounce events.\n",
            encoding="utf-8",
        )

    if signal_rows:
        df_signals = pd.DataFrame(signal_rows, columns=ANCHOR_AVWAP_SIGNAL_COLUMNS)
        write_header = (not ANCHOR_AVWAP_SIGNALS_FILE.exists()) or ANCHOR_AVWAP_SIGNALS_FILE.stat().st_size == 0
        df_signals.to_csv(ANCHOR_AVWAP_SIGNALS_FILE, mode="a", index=False, header=write_header)

    return events

# ============================================================================
# MAIN MASTER RUN
# ============================================================================

def _run_anchor_watchlist_scan_safe():
    try:
        anchor_events = run_anchor_watchlist_scan(archive_expired=False)
        logging.info(f"Anchor watchlist AVWAP scan complete. Events: {len(anchor_events)}")
    except Exception as exc:
        logging.exception(f"Anchor watchlist scan failed: {exc}")


def _evaluate_priority_snapshot_for_date(
    symbol: str,
    side: str,
    df_full: pd.DataFrame,
    evaluation_date: date,
    current_anchor_iso: str | None,
    previous_anchor_iso: str | None,
    recent_earnings_dates: list[str],
    history_state: dict[str, list[dict]],
) -> dict | None:
    df = df_full[df_full["datetime"].dt.date <= evaluation_date].copy()
    if df.empty:
        return None

    last_trade_date = df["datetime"].iloc[-1].date()
    if last_trade_date != evaluation_date:
        return None

    symbol_events_today = []
    symbol_multi_day = []
    current_anchor_meta = None
    prev_anchor_meta = None
    skip_current_events = False

    if recent_earnings_dates:
        try:
            last_earnings_date = datetime.fromisoformat(recent_earnings_dates[0]).date()
            sessions_since_last_earnings = sessions_since_date(df, last_earnings_date)
            if (
                sessions_since_last_earnings is not None
                and sessions_since_last_earnings <= RECENT_EARNINGS_SESSION_BLOCK
            ):
                skip_current_events = True
        except ValueError:
            skip_current_events = False

    def add_signal(event_name, _anchor_type, _anchor_date, _avwap_value, _stdev_value, _band_value):
        if event_name not in symbol_events_today:
            symbol_events_today.append(event_name)

    if current_anchor_iso:
        current_anchor_date = datetime.fromisoformat(current_anchor_iso).date()
        idxs = df.index[df["datetime"].dt.date == current_anchor_date]
        if not idxs.empty:
            anchor_idx = int(idxs[0])
            vwap_c, sd_c, bands_c = calc_anchored_vwap_bands(df, anchor_idx)
            if pd.notna(vwap_c) and bands_c:
                current_anchor_meta = {
                    "date": current_anchor_iso,
                    "vwap": float(vwap_c),
                    "stdev": float(sd_c),
                    "bands": {k: float(v) for k, v in bands_c.items()},
                }
                if not skip_current_events:
                    primary_cross = select_primary_cross_signal(df, side, "", vwap_c, bands_c)
                    if primary_cross:
                        lbl, lvl = primary_cross
                        add_signal(lbl, "CURRENT", current_anchor_iso, vwap_c, sd_c, lvl)

                    if side == "LONG":
                        bounce_tests = [
                            ("BOUNCE_VWAP", vwap_c),
                            ("BOUNCE_LOWER_1", bands_c["LOWER_1"]),
                            ("BOUNCE_LOWER_2", bands_c["LOWER_2"]),
                            ("BOUNCE_LOWER_3", bands_c["LOWER_3"]),
                            ("BOUNCE_UPPER_1", bands_c["UPPER_1"]),
                            ("BOUNCE_UPPER_2", bands_c["UPPER_2"]),
                            ("BOUNCE_UPPER_3", bands_c["UPPER_3"]),
                        ]
                        for lbl, lvl in bounce_tests:
                            if bounce_up_at_level(df, lvl):
                                add_signal(lbl, "CURRENT", current_anchor_iso, vwap_c, sd_c, lvl)
                    else:
                        bounce_tests = [
                            ("BOUNCE_VWAP", vwap_c),
                            ("BOUNCE_UPPER_1", bands_c["UPPER_1"]),
                            ("BOUNCE_UPPER_2", bands_c["UPPER_2"]),
                            ("BOUNCE_UPPER_3", bands_c["UPPER_3"]),
                            ("BOUNCE_LOWER_1", bands_c["LOWER_1"]),
                            ("BOUNCE_LOWER_2", bands_c["LOWER_2"]),
                            ("BOUNCE_LOWER_3", bands_c["LOWER_3"]),
                        ]
                        for lbl, lvl in bounce_tests:
                            if bounce_down_at_level(df, lvl):
                                add_signal(lbl, "CURRENT", current_anchor_iso, vwap_c, sd_c, lvl)

    if previous_anchor_iso:
        previous_anchor_date = datetime.fromisoformat(previous_anchor_iso).date()
        idxs = df.index[df["datetime"].dt.date == previous_anchor_date]
        if not idxs.empty:
            anchor_idx = int(idxs[0])
            vwap_p, sd_p, bands_p = calc_anchored_vwap_bands(df, anchor_idx)
            if pd.notna(vwap_p) and bands_p:
                prev_anchor_meta = {
                    "date": previous_anchor_iso,
                    "vwap": float(vwap_p),
                    "stdev": float(sd_p),
                    "bands": {k: float(v) for k, v in bands_p.items()},
                }

                if side == "LONG":
                    prev_bounce_tests = [
                        ("PREV_BOUNCE_VWAP", vwap_p),
                        ("PREV_BOUNCE_LOWER_1", bands_p.get("LOWER_1")),
                        ("PREV_BOUNCE_LOWER_2", bands_p.get("LOWER_2")),
                        ("PREV_BOUNCE_LOWER_3", bands_p.get("LOWER_3")),
                        ("PREV_BOUNCE_UPPER_1", bands_p.get("UPPER_1")),
                        ("PREV_BOUNCE_UPPER_2", bands_p.get("UPPER_2")),
                        ("PREV_BOUNCE_UPPER_3", bands_p.get("UPPER_3")),
                    ]
                    for lbl, lvl in prev_bounce_tests:
                        if bounce_up_at_level(df, lvl):
                            add_signal(lbl, "PREVIOUS", previous_anchor_iso, vwap_p, sd_p, lvl)
                else:
                    prev_bounce_tests = [
                        ("PREV_BOUNCE_VWAP", vwap_p),
                        ("PREV_BOUNCE_UPPER_1", bands_p.get("UPPER_1")),
                        ("PREV_BOUNCE_UPPER_2", bands_p.get("UPPER_2")),
                        ("PREV_BOUNCE_UPPER_3", bands_p.get("UPPER_3")),
                        ("PREV_BOUNCE_LOWER_1", bands_p.get("LOWER_1")),
                        ("PREV_BOUNCE_LOWER_2", bands_p.get("LOWER_2")),
                        ("PREV_BOUNCE_LOWER_3", bands_p.get("LOWER_3")),
                    ]
                    for lbl, lvl in prev_bounce_tests:
                        if bounce_down_at_level(df, lvl):
                            add_signal(lbl, "PREVIOUS", previous_anchor_iso, vwap_p, sd_p, lvl)

                primary_prev_cross = select_primary_cross_signal(df, side, "PREV_", vwap_p, bands_p)
                if primary_prev_cross:
                    lbl, lvl = primary_prev_cross
                    add_signal(lbl, "PREVIOUS", previous_anchor_iso, vwap_p, sd_p, lvl)

    symbol_events_today = sorted(set(symbol_events_today))
    previous_entries = history_state.get(symbol, [])
    previous_events = previous_entries[-1]["events"] if previous_entries else []
    symbol_multi_day = compute_multi_day_patterns(symbol, side, symbol_events_today, previous_events)
    full_event_list = symbol_events_today + symbol_multi_day
    history_state.setdefault(symbol, []).append(
        {
            "date": evaluation_date.isoformat(),
            "side": side,
            "events": full_event_list,
        }
    )

    df_recent = df.tail(60).copy()
    daily_ohlc = []
    for _, row in df_recent.iterrows():
        if pd.isna(row["datetime"]):
            continue
        daily_ohlc.append(
            {
                "date": row["datetime"].date().isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    last_row = get_last_daily_row_for_date(daily_ohlc, last_trade_date)
    last_close = float(last_row["close"]) if last_row else None
    last_volume = float(last_row["volume"]) if last_row else None
    atr20 = compute_atr_from_ohlc(daily_ohlc, last_trade_date)
    trend_label = compute_trend_label_20d(daily_ohlc, last_trade_date)
    has_bounce_event_today = bool(symbol_events_today)

    current_vwap = current_anchor_meta.get("vwap") if current_anchor_meta else None
    current_upper_1 = current_anchor_meta.get("bands", {}).get("UPPER_1") if current_anchor_meta else None
    current_upper_2 = current_anchor_meta.get("bands", {}).get("UPPER_2") if current_anchor_meta else None
    current_lower_1 = current_anchor_meta.get("bands", {}).get("LOWER_1") if current_anchor_meta else None
    current_lower_2 = current_anchor_meta.get("bands", {}).get("LOWER_2") if current_anchor_meta else None
    current_lower_3 = current_anchor_meta.get("bands", {}).get("LOWER_3") if current_anchor_meta else None
    current_upper_3 = current_anchor_meta.get("bands", {}).get("UPPER_3") if current_anchor_meta else None

    current_band_context = get_band_context(last_close, current_anchor_meta, side)
    previous_band_context = get_band_context(last_close, prev_anchor_meta, side)

    def _distance(level):
        if last_close is None or level is None:
            return None
        return last_close - level

    def _pct(level):
        if last_close is None or level is None or level == 0:
            return None
        return (last_close - level) / level * 100

    def _between(level_a, level_b):
        if last_close is None or level_a is None or level_b is None:
            return False
        low, high = sorted([level_a, level_b])
        return low <= last_close <= high

    favorite_zone = None
    if side == "LONG" and _between(current_vwap, current_upper_1):
        favorite_zone = "AVWAPE to UPPER_1"
    elif side == "SHORT" and _between(current_lower_1, current_vwap):
        favorite_zone = "LOWER_1 to AVWAPE"

    breakout_long_5d, breakout_short_5d = compute_five_day_breakout_flags(daily_ohlc, last_trade_date)
    recent_band_extension_days = count_recent_band_extension_days(
        daily_ohlc,
        last_trade_date,
        current_lower_1 if side == "LONG" else current_upper_1,
        side,
    )
    recent_second_band_test_days = count_recent_band_test_days(
        daily_ohlc,
        last_trade_date,
        current_upper_2 if side == "LONG" else current_lower_2,
        side,
    )
    second_band_penalty = compute_recent_second_band_penalty(recent_second_band_test_days)
    extension_note = ""
    if recent_second_band_test_days > 0:
        second_band_label = "UPPER_2" if side == "LONG" else "LOWER_2"
        extension_note = (
            f"Recent {second_band_label} tests={recent_second_band_test_days} "
            f"(-{second_band_penalty})"
        )
    breakout_5d = breakout_long_5d if side == "LONG" else breakout_short_5d
    retest_summary = analyze_avwap_retest_behavior(
        daily_ohlc,
        last_trade_date,
        current_vwap,
        side,
        current_upper_1=current_upper_1,
        current_lower_1=current_lower_1,
        atr20=atr20,
    )
    retest_followthrough = bool(retest_summary["retest_followthrough"])
    compression_summary = evaluate_anchor_compression(
        df,
        current_anchor_meta.get("date") if current_anchor_meta else None,
        current_anchor_meta.get("stdev") if current_anchor_meta else None,
        atr20,
        last_trade_date,
    )

    symbol_entry = {
        "side": side,
        "last_trade_date": last_trade_date.isoformat(),
        "current_anchor": current_anchor_meta,
        "previous_anchor": prev_anchor_meta,
        "events_today": symbol_events_today,
        "multi_day_patterns": symbol_multi_day,
        "events_all_for_day": full_event_list,
        "daily_ohlc": daily_ohlc,
        "last_close": last_close,
        "last_volume": last_volume,
        "atr20": atr20,
        "current_active_level": current_band_context["active_level"],
        "current_nearby_bands": list(current_band_context["nearby_levels"]),
        "current_band_zone": current_band_context["zone"],
        "previous_active_level": previous_band_context["active_level"],
        "previous_nearby_bands": list(previous_band_context["nearby_levels"]),
        "previous_band_zone": previous_band_context["zone"],
        "distance_from_current_vwap": _distance(current_vwap),
        "pct_from_current_vwap": _pct(current_vwap),
        "distance_from_current_upper_1": _distance(current_upper_1),
        "pct_from_current_upper_1": _pct(current_upper_1),
        "distance_from_current_lower_1": _distance(current_lower_1),
        "pct_from_current_lower_1": _pct(current_lower_1),
        "trend_20d": trend_label,
        "has_bounce_event_today": has_bounce_event_today,
        "favorite_zone": favorite_zone,
        "recent_band_extension_days": recent_band_extension_days,
        "recent_second_band_test_days": recent_second_band_test_days,
        "second_band_penalty": second_band_penalty,
        "breakout_5d": breakout_5d,
        "retest_followthrough": retest_followthrough,
        "retest_reference_level": retest_summary["retest_reference_level"],
        "retest_note": retest_summary["retest_note"],
        "extension_note": extension_note,
        "compression_flag": bool(compression_summary.get("is_compressed")),
        "compression_penalty": int(compression_summary.get("compression_penalty", 0) or 0),
        "compression_note": compression_summary.get("compression_note", ""),
    }

    priority_summary = build_priority_setup_summary(
        symbol=symbol,
        side=side,
        events_today=symbol_events_today,
        all_events=full_event_list,
        trend_label=trend_label,
        favorite_zone=favorite_zone,
        recent_band_extension_days=recent_band_extension_days,
        recent_second_band_test_days=recent_second_band_test_days,
        second_band_penalty=second_band_penalty,
        breakout_5d=breakout_5d,
        retest_followthrough=retest_followthrough,
        retest_reference_level=retest_summary["retest_reference_level"],
        retest_note=retest_summary["retest_note"],
        extension_note=extension_note,
    )
    priority_summary["score"] = float(
        priority_summary["score"] - int(compression_summary.get("compression_penalty", 0) or 0)
    )
    priority_summary["compression_flag"] = bool(compression_summary.get("is_compressed"))
    priority_summary["compression_penalty"] = int(compression_summary.get("compression_penalty", 0) or 0)
    priority_summary["compression_note"] = compression_summary.get("compression_note", "")

    feature_row = {
        "symbol": symbol,
        "side": side,
        "last_trade_date": last_trade_date.isoformat(),
        "last_close": last_close,
        "last_volume": last_volume,
        "atr20": atr20,
        "current_active_level": current_band_context["active_level"],
        "current_nearby_bands": ";".join(current_band_context["nearby_levels"]),
        "current_band_zone": current_band_context["zone"],
        "previous_active_level": previous_band_context["active_level"],
        "previous_nearby_bands": ";".join(previous_band_context["nearby_levels"]),
        "previous_band_zone": previous_band_context["zone"],
        "current_anchor_date": current_anchor_meta.get("date") if current_anchor_meta else None,
        "current_anchor_vwap": current_vwap,
        "current_anchor_stdev": current_anchor_meta.get("stdev") if current_anchor_meta else None,
        "distance_from_current_vwap": _distance(current_vwap),
        "pct_from_current_vwap": _pct(current_vwap),
        "distance_from_current_upper_1": _distance(current_upper_1),
        "pct_from_current_upper_1": _pct(current_upper_1),
        "distance_from_current_lower_1": _distance(current_lower_1),
        "pct_from_current_lower_1": _pct(current_lower_1),
        "trend_20d": trend_label,
        "has_bounce_event_today": has_bounce_event_today,
        "favorite_zone": favorite_zone,
        "recent_band_extension_days": recent_band_extension_days,
        "recent_second_band_test_days": recent_second_band_test_days,
        "second_band_penalty": second_band_penalty,
        "breakout_5d": breakout_5d,
        "retest_followthrough": retest_followthrough,
        "retest_reference_level": retest_summary["retest_reference_level"],
        "retest_note": retest_summary["retest_note"],
        "extension_note": extension_note,
        "compression_flag": bool(compression_summary.get("is_compressed")),
        "compression_penalty": int(compression_summary.get("compression_penalty", 0) or 0),
        "compression_note": compression_summary.get("compression_note", ""),
        "priority_score": priority_summary["score"],
        "favorite_signals": ";".join(priority_summary["favorite_signals"]),
        "favorite_context_signals": ";".join(priority_summary["context_signals"]),
        "events_today": ";".join(symbol_events_today),
    }

    return {
        "priority_row": priority_summary,
        "symbol_entry": symbol_entry,
        "feature_row": feature_row,
    }


def backfill_setup_tracker_from_recent_sessions(
    lookback_sessions: int = 5,
    longs_path: Path | None = None,
    shorts_path: Path | None = None,
    use_shared_watchlists: bool = False,
) -> dict:
    lookback_sessions = max(1, int(lookback_sessions))
    longs_path, shorts_path, watchlist_label = resolve_scan_watchlist_paths(
        longs_path=longs_path,
        shorts_path=shorts_path,
        use_shared_watchlists=use_shared_watchlists,
    )
    longs = load_tickers(longs_path)
    shorts = load_tickers(shorts_path)
    symbols = sorted(set(longs + shorts))
    if not symbols:
        logging.warning(f"No symbols found in {watchlist_label}; historical tracker backfill skipped.")
        return {"dates": [], "watchlists": {}}

    evaluation_dates = list(reversed(get_recent_market_session_dates(lookback_sessions)))
    if not evaluation_dates:
        logging.warning("Could not determine recent market sessions for tracker backfill.")
        return {"dates": [], "watchlists": {}}

    earnings_data = load_or_refresh_earnings(symbols)
    history_state: dict[str, list[dict]] = {}
    daily_frames_by_symbol: dict[str, pd.DataFrame] = {}
    earliest_eval_date = min(evaluation_dates)

    ib = connect_daily_data_client(client_id=1004, startup_wait=1.5)
    try:
        for symbol in symbols:
            symbol_side = "LONG" if symbol in longs else "SHORT"
            symbol_earnings = [
                _parse_iso_date_or_none(value)
                for value in earnings_data.get(symbol, [])
            ]
            symbol_earnings = [value for value in symbol_earnings if value is not None]

            days_needed = max(ATR_LENGTH + 220, (datetime.now().date() - earliest_eval_date).days + ATR_LENGTH + 20)
            if symbol_earnings:
                oldest_relevant_anchor = min(symbol_earnings)
                days_needed = max(days_needed, (datetime.now().date() - oldest_relevant_anchor).days + ATR_LENGTH + 20)

            df = fetch_daily_bars(ib, symbol, days_needed)
            if df.empty:
                logging.warning(f"{symbol}: no daily bars returned for tracker backfill.")
                continue
            daily_frames_by_symbol[symbol] = df

        watchlists_by_date = {}
        for evaluation_date in evaluation_dates:
            ai_state = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "run_date": evaluation_date.isoformat(),
                "symbols": {},
            }
            priority_rows = []
            feature_rows_by_symbol = {}

            for symbol in symbols:
                df = daily_frames_by_symbol.get(symbol)
                if df is None or df.empty:
                    continue
                side = "LONG" if symbol in longs else "SHORT"
                recent_earnings_dates = _earnings_dates_as_of(earnings_data.get(symbol, []), evaluation_date)
                current_anchor = pick_current_earnings_anchor_for_reference_date(
                    earnings_data.get(symbol, []),
                    evaluation_date,
                )
                previous_anchor = pick_previous_earnings_anchor_for_reference_date(
                    earnings_data.get(symbol, []),
                    evaluation_date,
                )
                snapshot = _evaluate_priority_snapshot_for_date(
                    symbol=symbol,
                    side=side,
                    df_full=df,
                    evaluation_date=evaluation_date,
                    current_anchor_iso=current_anchor.isoformat() if current_anchor else None,
                    previous_anchor_iso=previous_anchor.isoformat() if previous_anchor else None,
                    recent_earnings_dates=recent_earnings_dates,
                    history_state=history_state,
                )
                if not snapshot:
                    continue

                ai_state["symbols"][symbol] = snapshot["symbol_entry"]
                feature_rows_by_symbol[symbol] = snapshot["feature_row"]
                priority_rows.append(snapshot["priority_row"])

            if not priority_rows:
                update_setup_tracker_from_scan(
                    [],
                    ai_state,
                    feature_rows_by_symbol,
                    daily_frames_by_symbol,
                    ib,
                    scan_date=evaluation_date.isoformat(),
                )
                watchlists_by_date[evaluation_date.isoformat()] = []
                continue

            refine_priority_rows_with_directional_filters(priority_rows, ai_state, ib)
            apply_final_priority_buckets(priority_rows, ai_state, [], feature_rows_by_symbol)
            tracked_rows = [
                row
                for row in priority_rows
                if row.get("priority_bucket") in {"favorite_setup", "near_favorite_zone"}
                and not row.get("ranking_blocked")
            ]
            tracked_symbols = sorted({str(row.get("symbol", "")).strip().upper() for row in tracked_rows if str(row.get("symbol", "")).strip()})
            update_setup_tracker_from_scan(
                tracked_rows,
                ai_state,
                feature_rows_by_symbol,
                daily_frames_by_symbol,
                ib,
                scan_date=evaluation_date.isoformat(),
            )
            watchlists_by_date[evaluation_date.isoformat()] = tracked_symbols
            logging.info(
                f"Tracker backfill {evaluation_date.isoformat()}: tracked {len(tracked_rows)} setup(s) "
                f"across {len(tracked_symbols)} symbol(s)."
            )

        return {
            "dates": [value.isoformat() for value in evaluation_dates],
            "watchlists": watchlists_by_date,
        }
    finally:
        disconnect_daily_data_client(ib)


def run_master(
    longs_path: Path | None = None,
    shorts_path: Path | None = None,
    use_shared_watchlists: bool = False,
):
    longs_path, shorts_path, watchlist_label = resolve_scan_watchlist_paths(
        longs_path=longs_path,
        shorts_path=shorts_path,
        use_shared_watchlists=use_shared_watchlists,
    )
    logging.info(f"Running Master AVWAP scan using {watchlist_label}: {longs_path} | {shorts_path}")

    longs = load_tickers(longs_path)
    shorts = load_tickers(shorts_path)
    symbols = sorted(set(longs + shorts))

    if not symbols:
        logging.warning(
            f"No symbols found in {watchlist_label}. Running anchor-watchlist scan only."
        )
        _run_anchor_watchlist_scan_safe()
        return

    curr_cache = load_json(CURRENT_CACHE_FILE, default={})
    prev_cache = load_json(PREV_CACHE_FILE, default={})
    history = load_history()

    earnings_data = load_or_refresh_earnings(symbols)
    today_iso = datetime.now().date().isoformat()
    earnings_cache_updated = False

    logging.info(f"Refreshing earnings anchors for {len(symbols)} symbols…")
    refreshed_curr = 0
    refreshed_prev = 0
    missing_anchors = []

    for sym in symbols:
        dates = earnings_data.get(sym, [])

        if not dates:
            missing_anchors.append(sym)
            continue

        current_anchor = pick_current_earnings_anchor(dates)
        if current_anchor:
            curr_iso = current_anchor.isoformat()
            if curr_cache.get(sym) != curr_iso:
                curr_cache[sym] = curr_iso
                refreshed_curr += 1
                logging.info(f"{sym}: current anchor -> {current_anchor}")

        previous_anchor = pick_previous_earnings_anchor(dates)
        if previous_anchor:
            prev_iso = previous_anchor.isoformat()
            if prev_cache.get(sym) != prev_iso:
                prev_cache[sym] = prev_iso
                refreshed_prev += 1
                logging.info(f"{sym}: previous anchor -> {previous_anchor}")

    if earnings_cache_updated:
        earnings_cache = load_earnings_date_cache()
        symbol_cache = earnings_cache.setdefault("symbols", {})
        for sym, dates in earnings_data.items():
            if not dates:
                continue
            entry = symbol_cache.setdefault(sym, _normalize_earnings_cache_entry({}))
            entry["dates"] = _merge_earnings_dates(entry.get("dates", []), dates)
            entry["last_yf_refresh_on"] = today_iso
        save_earnings_date_cache(earnings_cache)

    if missing_anchors:
        logging.warning(
            "No earnings data found for: " + ", ".join(sorted(missing_anchors))
        )
    logging.info(
        f"Earnings anchors refreshed (current: {refreshed_curr}, previous: {refreshed_prev})."
    )

    ib = connect_daily_data_client(client_id=1003, startup_wait=1.5)

    today_run = datetime.now().date()
    events_for_output = []
    csv_rows = []
    feature_rows = []
    feature_rows_by_symbol = {}
    daily_frames_by_symbol = {}
    priority_rows = []
    positions = {
        "current": {lvl: [] for lvl in POSITION_LEVELS},
        "previous": {lvl: [] for lvl in POSITION_LEVELS},
    }
    range_buckets = {
        "long_avwap_to_upper_1": [],
        "long_upper_1_to_upper_2": [],
        "short_avwap_to_lower_1": [],
        "short_lower_1_to_lower_2": [],
    }
    stdev_range_hits = {"long": [], "short": []}
    stdev_cross_hits = {"long": [], "short": []}
    ai_state = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_date": today_run.isoformat(),
        "symbols": {}
    }

    for sym in symbols:
        side = "LONG" if sym in longs else "SHORT"
        curr_iso = curr_cache.get(sym)
        prev_iso = prev_cache.get(sym)

        if not curr_iso and not prev_iso:
            logging.warning(f"{sym}: no earnings anchors available.")
            continue

        # Determine days needed for a single daily fetch
        days_needed = ATR_LENGTH + 5
        anchor_dates = []
        if curr_iso:
            anchor_dates.append(datetime.fromisoformat(curr_iso).date())
        if prev_iso:
            anchor_dates.append(datetime.fromisoformat(prev_iso).date())

        if anchor_dates:
            max_span = max((today_run - d).days for d in anchor_dates)
            days_needed = max(days_needed, max_span + 5)

        df = fetch_daily_bars(ib, sym, days_needed)
        if df.empty:
            logging.warning(f"{sym}: no daily bars returned.")
            continue
        daily_frames_by_symbol[sym] = df.copy()

        last_trade_date = df["datetime"].iloc[-1].date()
        dstr = df["datetime"].iloc[-1].strftime("%m/%d")

        logging.info(f"-> Processing {sym} ({side}) with {len(df)} daily bars; last date {last_trade_date}")

        symbol_events_today = []
        symbol_multi_day = []
        current_anchor_meta = None
        prev_anchor_meta = None
        symbol_signal_info = {}
        skip_current_events = False

        recent_earnings_dates = earnings_data.get(sym, [])
        if recent_earnings_dates:
            try:
                last_earnings_date = datetime.fromisoformat(recent_earnings_dates[0]).date()
                sessions_since_last_earnings = sessions_since_date(df, last_earnings_date)
                if (
                    sessions_since_last_earnings is not None
                    and sessions_since_last_earnings <= RECENT_EARNINGS_SESSION_BLOCK
                ):
                    skip_current_events = True
                    logging.info(
                        f"{sym}: skipping CURRENT AVWAPE events; last earnings {last_earnings_date} "
                        f"was {sessions_since_last_earnings} session(s) ago (<= {RECENT_EARNINGS_SESSION_BLOCK})."
                    )
            except ValueError:
                logging.warning(f"{sym}: invalid recent earnings date format: {recent_earnings_dates[0]}")

        def add_signal(event_name, anchor_type, anchor_date, avwap_value, stdev_value, band_value):
            symbol_events_today.append(event_name)
            if event_name not in symbol_signal_info:
                symbol_signal_info[event_name] = {
                    "run_date": today_run.isoformat(),
                    "symbol": sym,
                    "trade_date": last_trade_date.isoformat(),
                    "side": side,
                    "anchor_type": anchor_type,
                    "anchor_date": anchor_date,
                    "signal_type": event_name,
                    "avwap_price": _to_float(avwap_value),
                    "band_price": _to_float(band_value),
                    "stdev": _to_float(stdev_value),
                    "priority_bucket": "",
                    "is_favorite_setup": False,
                    "is_near_favorite_zone": False,
                    "favorite_zone": "",
                    "favorite_signals": "",
                    "favorite_context_signals": "",
                }

        # Current earnings AVWAP
        if curr_iso:
            curr_date = datetime.fromisoformat(curr_iso).date()
            idxs = df.index[df["datetime"].dt.date == curr_date]
            if not idxs.empty:
                anchor_idx = int(idxs[0])
                vwap_c, sd_c, bands_c = calc_anchored_vwap_bands(df, anchor_idx)
                if pd.notna(vwap_c) and bands_c:
                    current_anchor_meta = {
                        "date": curr_iso,
                        "vwap": float(vwap_c),
                        "stdev": float(sd_c),
                        "bands": {k: float(v) for k, v in bands_c.items()}
                    }
                    if not skip_current_events:
                        primary_cross = select_primary_cross_signal(
                            df,
                            side,
                            "",
                            vwap_c,
                            bands_c,
                        )
                        if primary_cross:
                            lbl, lvl = primary_cross
                            add_signal(lbl, "CURRENT", curr_iso, vwap_c, sd_c, lvl)

                        # bounces (current)
                        if side == "LONG":
                            bounce_tests = [
                                ("BOUNCE_VWAP", vwap_c),
                                ("BOUNCE_LOWER_1", bands_c["LOWER_1"]),
                                ("BOUNCE_LOWER_2", bands_c["LOWER_2"]),
                                ("BOUNCE_LOWER_3", bands_c["LOWER_3"]),
                                ("BOUNCE_UPPER_1", bands_c["UPPER_1"]),
                                ("BOUNCE_UPPER_2", bands_c["UPPER_2"]),
                                ("BOUNCE_UPPER_3", bands_c["UPPER_3"]),
                            ]
                            for lbl, lvl in bounce_tests:
                                if bounce_up_at_level(df, lvl):
                                    add_signal(lbl, "CURRENT", curr_iso, vwap_c, sd_c, lvl)
                        else:
                            bounce_tests = [
                                ("BOUNCE_VWAP", vwap_c),
                                ("BOUNCE_UPPER_1", bands_c["UPPER_1"]),
                                ("BOUNCE_UPPER_2", bands_c["UPPER_2"]),
                                ("BOUNCE_UPPER_3", bands_c["UPPER_3"]),
                                ("BOUNCE_LOWER_1", bands_c["LOWER_1"]),
                                ("BOUNCE_LOWER_2", bands_c["LOWER_2"]),
                                ("BOUNCE_LOWER_3", bands_c["LOWER_3"]),
                            ]
                            for lbl, lvl in bounce_tests:
                                if bounce_down_at_level(df, lvl):
                                    add_signal(lbl, "CURRENT", curr_iso, vwap_c, sd_c, lvl)

                else:
                    logging.warning(f"{sym}: invalid current AVWAP / bands.")
            else:
                logging.warning(f"{sym}: no candle on current earnings date {curr_date}.")

        # Previous earnings AVWAP
        if prev_iso:
            prev_date = datetime.fromisoformat(prev_iso).date()
            idxs = df.index[df["datetime"].dt.date == prev_date]
            if not idxs.empty:
                anchor_idx = int(idxs[0])
                vwap_p, sd_p, bands_p = calc_anchored_vwap_bands(df, anchor_idx)
                if pd.notna(vwap_p) and bands_p:
                    prev_anchor_meta = {
                        "date": prev_iso,
                        "vwap": float(vwap_p),
                        "stdev": float(sd_p),
                        "bands": {k: float(v) for k, v in bands_p.items()}
                    }

                    # previous bounces
                    if side == "LONG":
                        prev_bounce_tests = [
                            ("PREV_BOUNCE_VWAP", vwap_p),
                            ("PREV_BOUNCE_LOWER_1", bands_p.get("LOWER_1")),
                            ("PREV_BOUNCE_LOWER_2", bands_p.get("LOWER_2")),
                            ("PREV_BOUNCE_LOWER_3", bands_p.get("LOWER_3")),
                            ("PREV_BOUNCE_UPPER_1", bands_p.get("UPPER_1")),
                            ("PREV_BOUNCE_UPPER_2", bands_p.get("UPPER_2")),
                            ("PREV_BOUNCE_UPPER_3", bands_p.get("UPPER_3")),
                        ]
                        for lbl, lvl in prev_bounce_tests:
                            if bounce_up_at_level(df, lvl):
                                add_signal(lbl, "PREVIOUS", prev_iso, vwap_p, sd_p, lvl)
                    else:
                        prev_bounce_tests = [
                            ("PREV_BOUNCE_VWAP", vwap_p),
                            ("PREV_BOUNCE_UPPER_1", bands_p.get("UPPER_1")),
                            ("PREV_BOUNCE_UPPER_2", bands_p.get("UPPER_2")),
                            ("PREV_BOUNCE_UPPER_3", bands_p.get("UPPER_3")),
                            ("PREV_BOUNCE_LOWER_1", bands_p.get("LOWER_1")),
                            ("PREV_BOUNCE_LOWER_2", bands_p.get("LOWER_2")),
                            ("PREV_BOUNCE_LOWER_3", bands_p.get("LOWER_3")),
                        ]
                        for lbl, lvl in prev_bounce_tests:
                            if bounce_down_at_level(df, lvl):
                                add_signal(lbl, "PREVIOUS", prev_iso, vwap_p, sd_p, lvl)

                    # previous crosses
                    primary_prev_cross = select_primary_cross_signal(
                        df,
                        side,
                        "PREV_",
                        vwap_p,
                        bands_p,
                    )
                    if primary_prev_cross:
                        lbl, lvl = primary_prev_cross
                        add_signal(lbl, "PREVIOUS", prev_iso, vwap_p, sd_p, lvl)
                else:
                    logging.warning(f"{sym}: invalid previous AVWAP / bands.")
            else:
                logging.warning(f"{sym}: no candle on previous earnings date {prev_date}.")

        # dedupe and sort events for consistency
        symbol_events_today = sorted(set(symbol_events_today))

        # multi-day pattern detection
        prev_entries = history.get(sym, [])
        prev_events = prev_entries[-1]["events"] if prev_entries else []
        md_patterns = compute_multi_day_patterns(sym, side,
                                                 symbol_events_today,
                                                 prev_events)
        symbol_multi_day = md_patterns

        # include multi-day patterns as events as well
        full_event_list = symbol_events_today + symbol_multi_day

        # record in history
        entry = {
            "date": today_run.isoformat(),
            "side": side,
            "events": full_event_list
        }
        history.setdefault(sym, []).append(entry)

        # append to output lines
        for lbl in full_event_list:
            events_for_output.append((sym, dstr, lbl, side))

        # prepare daily OHLC slice for AI (recent window only)
        # use last ~60 days of df
        df_recent = df.tail(60).copy()
        daily_ohlc = []
        for _, row in df_recent.iterrows():
            if pd.isna(row["datetime"]):
                continue
            daily_ohlc.append({
                "date": row["datetime"].date().isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

        last_row = get_last_daily_row_for_date(daily_ohlc, last_trade_date)
        last_close = float(last_row["close"]) if last_row else None
        last_volume = float(last_row["volume"]) if last_row else None

        atr20 = compute_atr_from_ohlc(daily_ohlc, last_trade_date)
        trend_label = compute_trend_label_20d(daily_ohlc, last_trade_date)
        has_bounce_event_today = bool(symbol_events_today)

        current_vwap = current_anchor_meta.get("vwap") if current_anchor_meta else None
        current_upper_1 = (
            current_anchor_meta.get("bands", {}).get("UPPER_1")
            if current_anchor_meta else None
        )
        current_upper_2 = (
            current_anchor_meta.get("bands", {}).get("UPPER_2")
            if current_anchor_meta else None
        )
        current_upper_3 = (
            current_anchor_meta.get("bands", {}).get("UPPER_3")
            if current_anchor_meta else None
        )
        current_lower_1 = (
            current_anchor_meta.get("bands", {}).get("LOWER_1")
            if current_anchor_meta else None
        )
        current_lower_2 = (
            current_anchor_meta.get("bands", {}).get("LOWER_2")
            if current_anchor_meta else None
        )
        current_lower_3 = (
            current_anchor_meta.get("bands", {}).get("LOWER_3")
            if current_anchor_meta else None
        )
        current_band_context = get_band_context(last_close, current_anchor_meta, side)
        previous_band_context = get_band_context(last_close, prev_anchor_meta, side)

        def _distance(level):
            if last_close is None or level is None:
                return None
            return last_close - level

        def _pct(level):
            if last_close is None or level is None or level == 0:
                return None
            return (last_close - level) / level * 100

        dist_vwap = _distance(current_vwap)
        pct_vwap = _pct(current_vwap)
        dist_upper_1 = _distance(current_upper_1)
        pct_upper_1 = _pct(current_upper_1)
        dist_lower_1 = _distance(current_lower_1)
        pct_lower_1 = _pct(current_lower_1)

        def _between(level_a, level_b):
            if last_close is None or level_a is None or level_b is None:
                return False
            low, high = sorted([level_a, level_b])
            return low <= last_close <= high

        if side == "LONG":
            if _between(current_vwap, current_upper_1):
                range_buckets["long_avwap_to_upper_1"].append(sym)
            if _between(current_upper_1, current_upper_2):
                range_buckets["long_upper_1_to_upper_2"].append(sym)
        else:
            if _between(current_lower_1, current_vwap):
                range_buckets["short_avwap_to_lower_1"].append(sym)
            if _between(current_lower_2, current_lower_1):
                range_buckets["short_lower_1_to_lower_2"].append(sym)

        favorite_zone = None
        if side == "LONG" and _between(current_vwap, current_upper_1):
            favorite_zone = "AVWAPE to UPPER_1"
        elif side == "SHORT" and _between(current_lower_1, current_vwap):
            favorite_zone = "LOWER_1 to AVWAPE"

        breakout_long_5d, breakout_short_5d = compute_five_day_breakout_flags(daily_ohlc, last_trade_date)
        recent_band_extension_days = count_recent_band_extension_days(
            daily_ohlc,
            last_trade_date,
            current_lower_1 if side == "LONG" else current_upper_1,
            side,
        )
        recent_second_band_test_days = count_recent_band_test_days(
            daily_ohlc,
            last_trade_date,
            current_upper_2 if side == "LONG" else current_lower_2,
            side,
        )
        second_band_penalty = compute_recent_second_band_penalty(recent_second_band_test_days)
        extension_note = ""
        if recent_second_band_test_days > 0:
            second_band_label = "UPPER_2" if side == "LONG" else "LOWER_2"
            extension_note = (
                f"Recent {second_band_label} tests={recent_second_band_test_days} "
                f"(-{second_band_penalty})"
            )
        breakout_5d = breakout_long_5d if side == "LONG" else breakout_short_5d
        retest_summary = analyze_avwap_retest_behavior(
            daily_ohlc,
            last_trade_date,
            current_vwap,
            side,
            current_upper_1=current_upper_1,
            current_lower_1=current_lower_1,
            atr20=atr20,
        )
        retest_followthrough = bool(retest_summary["retest_followthrough"])
        compression_summary = evaluate_anchor_compression(
            df,
            current_anchor_meta.get("date") if current_anchor_meta else None,
            current_anchor_meta.get("stdev") if current_anchor_meta else None,
            atr20,
            last_trade_date,
        )

        if current_anchor_meta:
            stdev_blocked_by_recent_earnings = False
            if recent_earnings_dates:
                try:
                    stdev_last_earnings = datetime.fromisoformat(recent_earnings_dates[0]).date()
                    stdev_sessions_since = sessions_since_date(df, stdev_last_earnings)
                    stdev_blocked_by_recent_earnings = (
                        stdev_sessions_since is not None
                        and stdev_sessions_since <= STDEV_RECENT_EARNINGS_BLOCK
                    )
                except ValueError:
                    stdev_blocked_by_recent_earnings = False

            if not stdev_blocked_by_recent_earnings:
                if side == "LONG":
                    if (
                        current_upper_2 is not None
                        and current_upper_3 is not None
                        and last_close is not None
                        and last_close >= current_upper_2
                        and closes_between_bands(df, current_upper_2, current_upper_3, 4)
                    ):
                        stdev_range_hits["long"].append(sym)
                    if current_upper_2 is not None and cross_up_through_level(df, current_upper_2):
                        stdev_cross_hits["long"].append(sym)
                    if current_upper_2 is not None and bounce_up_at_level(df, current_upper_2):
                        stdev_cross_hits["long"].append(f"{sym} (bounce)")
                else:
                    if (
                        current_lower_2 is not None
                        and current_lower_3 is not None
                        and last_close is not None
                        and last_close <= current_lower_2
                        and closes_between_bands(df, current_lower_3, current_lower_2, 4)
                    ):
                        stdev_range_hits["short"].append(sym)
                    if current_lower_2 is not None and cross_down_through_level(df, current_lower_2):
                        stdev_cross_hits["short"].append(sym)
                    if current_lower_2 is not None and bounce_down_at_level(df, current_lower_2):
                        stdev_cross_hits["short"].append(f"{sym} (bounce)")

        current_position = current_band_context["active_level"]
        if current_position:
            positions["current"].setdefault(current_position, []).append(sym)

        previous_position = previous_band_context["active_level"]
        if previous_position:
            positions["previous"].setdefault(previous_position, []).append(sym)

        symbol_entry = {
            "side": side,
            "last_trade_date": last_trade_date.isoformat(),
            "current_anchor": current_anchor_meta,
            "previous_anchor": prev_anchor_meta,
            "events_today": symbol_events_today,
            "multi_day_patterns": symbol_multi_day,
            "events_all_for_day": full_event_list,
            "daily_ohlc": daily_ohlc,
            "last_close": last_close,
            "last_volume": last_volume,
            "atr20": atr20,
            "current_active_level": current_band_context["active_level"],
            "current_nearby_bands": list(current_band_context["nearby_levels"]),
            "current_band_zone": current_band_context["zone"],
            "previous_active_level": previous_band_context["active_level"],
            "previous_nearby_bands": list(previous_band_context["nearby_levels"]),
            "previous_band_zone": previous_band_context["zone"],
            "distance_from_current_vwap": dist_vwap,
            "pct_from_current_vwap": pct_vwap,
            "distance_from_current_upper_1": dist_upper_1,
            "pct_from_current_upper_1": pct_upper_1,
            "distance_from_current_lower_1": dist_lower_1,
            "pct_from_current_lower_1": pct_lower_1,
            "trend_20d": trend_label,
            "has_bounce_event_today": has_bounce_event_today,
            "favorite_zone": favorite_zone,
            "recent_band_extension_days": recent_band_extension_days,
            "recent_second_band_test_days": recent_second_band_test_days,
            "second_band_penalty": second_band_penalty,
            "breakout_5d": breakout_5d,
            "retest_followthrough": retest_followthrough,
            "retest_reference_level": retest_summary["retest_reference_level"],
            "retest_note": retest_summary["retest_note"],
            "extension_note": extension_note,
            "compression_flag": bool(compression_summary.get("is_compressed")),
            "compression_penalty": int(compression_summary.get("compression_penalty", 0) or 0),
            "compression_note": compression_summary.get("compression_note", ""),
        }

        priority_summary = build_priority_setup_summary(
            symbol=sym,
            side=side,
            events_today=symbol_events_today,
            all_events=full_event_list,
            trend_label=trend_label,
            favorite_zone=favorite_zone,
            recent_band_extension_days=recent_band_extension_days,
            recent_second_band_test_days=recent_second_band_test_days,
            second_band_penalty=second_band_penalty,
            breakout_5d=breakout_5d,
            retest_followthrough=retest_followthrough,
            retest_reference_level=retest_summary["retest_reference_level"],
            retest_note=retest_summary["retest_note"],
            extension_note=extension_note,
        )
        priority_summary["score"] = float(priority_summary["score"] - int(compression_summary.get("compression_penalty", 0) or 0))
        priority_summary["compression_flag"] = bool(compression_summary.get("is_compressed"))
        priority_summary["compression_penalty"] = int(compression_summary.get("compression_penalty", 0) or 0)
        priority_summary["compression_note"] = compression_summary.get("compression_note", "")
        symbol_entry["priority_score"] = priority_summary["score"]
        symbol_entry["favorite_signals"] = priority_summary["favorite_signals"]
        symbol_entry["favorite_context_signals"] = priority_summary["context_signals"]
        priority_rows.append(priority_summary)

        priority_bucket = ""
        if priority_summary["has_favorite_signal"]:
            priority_bucket = "favorite_setup"
        elif favorite_zone:
            priority_bucket = "near_favorite_zone"

        for record in symbol_signal_info.values():
            record["priority_bucket"] = priority_bucket
            record["is_favorite_setup"] = bool(priority_summary["has_favorite_signal"])
            record["is_near_favorite_zone"] = bool(favorite_zone)
            record["favorite_zone"] = favorite_zone or ""
            record["favorite_signals"] = ";".join(priority_summary["favorite_signals"])
            record["favorite_context_signals"] = ";".join(priority_summary["context_signals"])

        for lbl in symbol_events_today:
            record = symbol_signal_info.get(lbl)
            if record:
                csv_rows.append(record)

        ai_state["symbols"][sym] = symbol_entry

        feature_row = {
            "symbol": sym,
            "side": side,
            "last_trade_date": last_trade_date.isoformat(),
            "last_close": last_close,
            "last_volume": last_volume,
            "atr20": atr20,
            "current_active_level": current_band_context["active_level"],
            "current_nearby_bands": ";".join(current_band_context["nearby_levels"]),
            "current_band_zone": current_band_context["zone"],
            "previous_active_level": previous_band_context["active_level"],
            "previous_nearby_bands": ";".join(previous_band_context["nearby_levels"]),
            "previous_band_zone": previous_band_context["zone"],
            "current_anchor_date": current_anchor_meta.get("date") if current_anchor_meta else None,
            "current_anchor_vwap": current_vwap,
            "current_anchor_stdev": current_anchor_meta.get("stdev") if current_anchor_meta else None,
            "distance_from_current_vwap": dist_vwap,
            "pct_from_current_vwap": pct_vwap,
            "distance_from_current_upper_1": dist_upper_1,
            "pct_from_current_upper_1": pct_upper_1,
            "distance_from_current_lower_1": dist_lower_1,
            "pct_from_current_lower_1": pct_lower_1,
            "trend_20d": trend_label,
            "has_bounce_event_today": has_bounce_event_today,
            "favorite_zone": favorite_zone,
            "recent_band_extension_days": recent_band_extension_days,
            "recent_second_band_test_days": recent_second_band_test_days,
            "second_band_penalty": second_band_penalty,
            "breakout_5d": breakout_5d,
            "retest_followthrough": retest_followthrough,
            "retest_reference_level": retest_summary["retest_reference_level"],
            "retest_note": retest_summary["retest_note"],
            "extension_note": extension_note,
            "compression_flag": bool(compression_summary.get("is_compressed")),
            "compression_penalty": int(compression_summary.get("compression_penalty", 0) or 0),
            "compression_note": compression_summary.get("compression_note", ""),
            "priority_score": priority_summary["score"],
            "favorite_signals": ";".join(priority_summary["favorite_signals"]),
            "favorite_context_signals": ";".join(priority_summary["context_signals"]),
            "events_today": ";".join(symbol_events_today),
        }
        feature_rows.append(feature_row)
        feature_rows_by_symbol[sym] = feature_row

        logging.info(
            f"{sym}: events_today={symbol_events_today}, "
            f"multi_day={symbol_multi_day}"
        )

    refine_priority_rows_with_directional_filters(priority_rows, ai_state, ib)
    apply_final_priority_buckets(priority_rows, ai_state, csv_rows, feature_rows_by_symbol)
    tracked_rows = [
        row
        for row in priority_rows
        if row.get("priority_bucket") in {"favorite_setup", "near_favorite_zone"}
        and not row.get("ranking_blocked")
    ]
    update_setup_tracker_from_scan(tracked_rows, ai_state, feature_rows_by_symbol, daily_frames_by_symbol, ib)

    disconnect_daily_data_client(ib)

    if csv_rows:
        df_signals = pd.DataFrame(csv_rows)
        df_signals = df_signals.reindex(columns=AVWAP_CSV_COLUMNS)
        df_signals.sort_values(["run_date", "trade_date", "symbol", "signal_type"], inplace=True)

        if AVWAP_SIGNALS_FILE.exists() and AVWAP_SIGNALS_FILE.stat().st_size > 0:
            existing_signals = pd.read_csv(AVWAP_SIGNALS_FILE)
            existing_signals = existing_signals.reindex(columns=AVWAP_CSV_COLUMNS)
            df_signals = pd.concat([existing_signals, df_signals], ignore_index=True)
            df_signals.sort_values(["run_date", "trade_date", "symbol", "signal_type"], inplace=True)

        df_signals.to_csv(
            AVWAP_SIGNALS_FILE,
            index=False,
        )
        logging.info(
            f"Appended {len(df_signals)} AVWAP signals to {AVWAP_SIGNALS_FILE}"
        )
    else:
        logging.info(
            f"No AVWAP signals generated for {today_run.isoformat()}; nothing appended."
        )

    # trim history to last N days
    trim_history(history)
    write_priority_setup_report(PRIORITY_SETUPS_FILE, priority_rows)
    write_master_avwap_focus_feed(MASTER_AVWAP_FOCUS_FILE, priority_rows, ai_state)

    # write human-readable events file (grouped for easier scanning)
    sorted_events = sort_events_for_output(events_for_output)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        priority_text = PRIORITY_SETUPS_FILE.read_text(encoding="utf-8").strip()
        if priority_text:
            f.write(priority_text)
            f.write("\n\n")
        for s, d, lbl, side in sorted_events:
            f.write(f"{s},{d},{lbl},{side}\n")

        def _write_range_line(label, tickers):
            items = ", ".join(sorted(set(tickers))) if tickers else "None"
            f.write(f"{label}: {items}\n")

        f.write("\nPrice ranges (current anchors):\n")
        _write_range_line(
            "Longs between AVWAP and UPPER_1",
            range_buckets["long_avwap_to_upper_1"],
        )
        _write_range_line(
            "Longs between UPPER_1 and UPPER_2",
            range_buckets["long_upper_1_to_upper_2"],
        )
        _write_range_line(
            "Shorts between AVWAP and LOWER_1",
            range_buckets["short_avwap_to_lower_1"],
        )
        _write_range_line(
            "Shorts between LOWER_1 and LOWER_2",
            range_buckets["short_lower_1_to_lower_2"],
        )
        f.write(f"\nRun completed at {datetime.now().strftime('%H:%M:%S')}\n")

    # write grouped ticker lists for easy copy/paste into TradingView/TC2000
    event_buckets = {}
    for sym, _, lbl, side in sorted_events:
        event_buckets.setdefault(lbl, {"LONG": [], "SHORT": []})[side].append(sym)

    def _fmt_items(values):
        return ", ".join(sorted(set(values))) if values else "None"

    with open(EVENT_TICKERS_FILE, "w", encoding="utf-8") as f:
        f.write("AVWAP crosses and bounces by event type\n")
        f.write(f"Priority setups report: {PRIORITY_SETUPS_FILE.name}\n\n")
        for lbl in sorted(event_buckets.keys(), key=event_label_sort_key):
            for side in ("LONG", "SHORT"):
                tickers = sorted(set(event_buckets[lbl][side]))
                if not tickers:
                    continue
                display_label = format_signal_label(lbl)
                f.write(f"{display_label}, {side.capitalize()}: {', '.join(tickers)}\n")

        f.write("\nPrice ranges (current anchors)\n")
        range_labels = [
            ("Longs between AVWAP and UPPER_1", "long_avwap_to_upper_1"),
            ("Longs between UPPER_1 and UPPER_2", "long_upper_1_to_upper_2"),
            ("Shorts between AVWAP and LOWER_1", "short_avwap_to_lower_1"),
            ("Shorts between LOWER_1 and LOWER_2", "short_lower_1_to_lower_2"),
        ]
        for label, key in range_labels:
            f.write(f"{label}: {_fmt_items(range_buckets[key])}\n")

    write_stdev_range_report(STDEV_RANGE_FILE, stdev_range_hits, stdev_cross_hits)
    write_tradingview_report(
        TRADINGVIEW_REPORT_FILE,
        priority_rows,
        event_buckets,
        range_buckets,
        stdev_range_hits,
        stdev_cross_hits,
    )

    feature_columns = [
        "symbol",
        "side",
        "last_trade_date",
        "last_close",
        "last_volume",
        "atr20",
        "current_active_level",
        "current_nearby_bands",
        "current_band_zone",
        "previous_active_level",
        "previous_nearby_bands",
        "previous_band_zone",
        "current_anchor_date",
        "current_anchor_vwap",
        "current_anchor_stdev",
        "distance_from_current_vwap",
        "pct_from_current_vwap",
        "distance_from_current_upper_1",
        "pct_from_current_upper_1",
        "distance_from_current_lower_1",
        "pct_from_current_lower_1",
        "trend_20d",
        "has_bounce_event_today",
        "favorite_zone",
        "recent_band_extension_days",
        "recent_second_band_test_days",
        "second_band_penalty",
        "breakout_5d",
        "retest_followthrough",
        "retest_reference_level",
        "retest_note",
        "extension_note",
        "compression_flag",
        "compression_penalty",
        "compression_note",
        "priority_score",
        "favorite_signals",
        "favorite_context_signals",
        "events_today",
    ]

    df_features = pd.DataFrame(feature_rows, columns=feature_columns)
    df_features.to_csv(D1_FEATURES_FILE, index=False)

    positions_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "levels": POSITION_LEVELS,
        "current": {
            lvl: sorted(set(positions["current"].get(lvl, [])))
            for lvl in POSITION_LEVELS
        },
        "previous": {
            lvl: sorted(set(positions["previous"].get(lvl, [])))
            for lvl in POSITION_LEVELS
        },
    }

    with open(MASTER_POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(positions_payload, f, indent=2)

    save_json(CURRENT_CACHE_FILE, curr_cache)
    save_json(PREV_CACHE_FILE, prev_cache)
    save_history(history)
    save_json(AI_STATE_FILE, ai_state)

    _run_anchor_watchlist_scan_safe()

    logging.info(
        f"Master AVWAP run complete. "
        f"Events: {OUTPUT_FILE}, AI state: {AI_STATE_FILE}, history: {HISTORY_FILE}"
    )

# ============================================================================
# GUI
# ============================================================================

class MasterAvwapGUI:
    def __init__(self, root, standalone=True):
        self.root = root
        self.standalone = standalone
        if standalone:
            self.root.title("Master AVWAP Manager")
            self.root.geometry("1200x760")

        # Embedded mode may pass a ttk container (e.g. ttk.Frame), which
        # does not expose a `bg`/`background` configure option.
        try:
            self.root.configure(bg=GUI_DARK_BG)
        except tk.TclError:
            try:
                self.root.configure(background=GUI_DARK_BG)
            except tk.TclError:
                pass
        self._configure_dark_theme()

        self.status_var = tk.StringVar(value="Ready")
        self.tracker_storage_var = tk.StringVar(value="")
        self.ticker_var = tk.StringVar()
        self.anchor_var = tk.StringVar()
        self.side_var = tk.StringVar(value="LONG")
        self.notes_var = tk.StringVar()
        self.bulk_side_var = tk.StringVar(value="LONG")
        self.earnings_lookback_var = tk.IntVar(value=1)
        self.tracker_backfill_sessions_var = tk.IntVar(value=5)
        self.focus_side_map = {}
        self.setup_tracker_row_map = {}

        self._build_layout()
        self.refresh_table()
        self.refresh_avwap_output_view()
        self.refresh_anchor_output_view()
        self.refresh_setup_tracker_view()

    def _configure_dark_theme(self):
        style = ttk.Style(self.root)

        try:
            current_theme = style.theme_use()
            if current_theme == "default":
                style.theme_use("clam")
        except Exception:
            pass

        self.root.option_add("*Background", GUI_DARK_BG)
        self.root.option_add("*Foreground", GUI_DARK_TEXT)
        self.root.option_add("*Entry*Background", GUI_DARK_INPUT)
        self.root.option_add("*Entry*Foreground", GUI_DARK_TEXT)
        self.root.option_add("*Text*Background", GUI_DARK_INPUT)
        self.root.option_add("*Text*Foreground", GUI_DARK_TEXT)

        style.configure(".", background=GUI_DARK_BG, foreground=GUI_DARK_TEXT)
        style.configure("TFrame", background=GUI_DARK_BG)
        style.configure("TLabel", background=GUI_DARK_BG, foreground=GUI_DARK_TEXT)

        style.configure("TButton", background=GUI_DARK_PANEL, foreground=GUI_DARK_TEXT)
        style.map(
            "TButton",
            background=[("active", "#4A4A4A"), ("disabled", GUI_DARK_PANEL)],
            foreground=[("disabled", "#9AA0A6")],
        )

        style.configure("TEntry", fieldbackground=GUI_DARK_INPUT, foreground=GUI_DARK_TEXT)
        style.map(
            "TEntry",
            fieldbackground=[("readonly", GUI_DARK_INPUT), ("disabled", GUI_DARK_PANEL)],
            foreground=[("readonly", GUI_DARK_TEXT), ("disabled", "#9AA0A6")],
        )

        style.configure("TSpinbox", fieldbackground=GUI_DARK_INPUT, foreground=GUI_DARK_TEXT)
        style.map(
            "TSpinbox",
            fieldbackground=[("readonly", GUI_DARK_INPUT), ("disabled", GUI_DARK_PANEL)],
            foreground=[("readonly", GUI_DARK_TEXT), ("disabled", "#9AA0A6")],
        )

        style.configure("TCombobox", fieldbackground=GUI_DARK_INPUT, foreground=GUI_DARK_TEXT)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", GUI_DARK_INPUT), ("disabled", GUI_DARK_PANEL)],
            foreground=[("readonly", GUI_DARK_TEXT), ("disabled", "#9AA0A6")],
        )

        style.configure("TNotebook", background=GUI_DARK_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=GUI_DARK_PANEL, foreground=GUI_DARK_TEXT)
        style.map("TNotebook.Tab", background=[("selected", "#4A4A4A")])
        style.configure("TLabelframe", background=GUI_DARK_BG, foreground=GUI_DARK_TEXT)
        style.configure("TLabelframe.Label", background=GUI_DARK_BG, foreground=GUI_DARK_TEXT)

        # Use explicit style names so Treeview + heading colors are stable on
        # Windows where the default native theme can ignore generic style keys.
        style.configure(
            "Dark.Treeview",
            background=GUI_DARK_INPUT,
            fieldbackground=GUI_DARK_INPUT,
            foreground=GUI_DARK_TEXT,
            bordercolor=GUI_DARK_BG,
            rowheight=24,
        )
        style.map(
            "Dark.Treeview",
            background=[("selected", "#4A4A4A")],
            foreground=[("selected", GUI_DARK_TEXT)],
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=GUI_DARK_PANEL,
            foreground=GUI_DARK_TEXT,
            relief="flat",
        )
        style.map("Dark.Treeview.Heading", background=[("active", "#4A4A4A")])

        style.configure("Vertical.TScrollbar", background=GUI_DARK_PANEL, troughcolor=GUI_DARK_BG)

    def _apply_dark_theme_to_text_widgets(self):
        for widget in (
            self.avwap_text,
            self.favorite_symbols_text,
            self.near_favorite_symbols_text,
            self.bulk_anchor_text,
            self.anchor_scan_text,
            self.setup_tracker_stats_text,
        ):
            widget.configure(
                bg=GUI_DARK_INPUT,
                fg=GUI_DARK_TEXT,
                insertbackground=GUI_DARK_TEXT,
                selectbackground="#4A4A4A",
                selectforeground=GUI_DARK_TEXT,
                highlightbackground=GUI_DARK_BG,
                highlightcolor=GUI_DARK_PANEL,
            )
        self.avwap_text.tag_configure("trendline_bold", font=("Courier New", 10, "bold"))

    def _build_layout(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=10, pady=8)

        ttk.Label(toolbar, text="Earnings gap lookback:").pack(side="left", padx=(0, 4))
        lookback_spin = ttk.Spinbox(
            toolbar,
            from_=1,
            to=20,
            width=4,
            textvariable=self.earnings_lookback_var,
        )
        lookback_spin.pack(side="left", padx=(0, 10))

        ttk.Button(toolbar, text="Run Earnings Gap Scan", command=self.run_earnings_scan).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Run Master Scan", command=self.run_master_once).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Run Home Folder Scan", command=self.run_shared_watchlist_scan_once).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Run Anchor Watchlist Scan", command=self.run_anchor_scan_once).pack(side="left", padx=4)

        ttk.Label(
            self.root,
            text="Views refresh automatically when scans finish or when you open a tab.",
            justify="left",
        ).pack(fill="x", padx=10, pady=(0, 8))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=8)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        tracker_tab = ttk.Frame(self.notebook)
        self.tracker_tab = tracker_tab
        self.notebook.add(tracker_tab, text="Setup Tracker")

        tracker_toolbar = ttk.Frame(tracker_tab)
        tracker_toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(tracker_toolbar, text="Copy Active Symbols", command=self.copy_setup_tracker_symbols).pack(side="left", padx=(8, 0))
        ttk.Label(tracker_toolbar, text="Backfill sessions:").pack(side="left", padx=(16, 4))
        tracker_backfill_spin = ttk.Spinbox(
            tracker_toolbar,
            from_=1,
            to=30,
            width=4,
            textvariable=self.tracker_backfill_sessions_var,
        )
        tracker_backfill_spin.pack(side="left")
        ttk.Button(
            tracker_toolbar,
            text="Backfill Tracker",
            command=self.backfill_setup_tracker_history,
        ).pack(side="left", padx=(8, 0))
        ttk.Frame(tracker_toolbar).pack(side="left", fill="x", expand=True)
        ttk.Button(tracker_toolbar, text="Change Home Folder", command=self.choose_tracker_storage_dir).pack(side="left", padx=(16, 0))
        ttk.Button(tracker_toolbar, text="Open Home Folder", command=self.open_tracker_storage_dir).pack(side="left", padx=(8, 0))
        ttk.Button(tracker_toolbar, text="Open Settings File", command=self.open_tracker_settings_file).pack(side="left", padx=(8, 0))

        tracker_storage_label = ttk.Label(
            tracker_tab,
            textvariable=self.tracker_storage_var,
            justify="left",
            wraplength=1100,
        )
        tracker_storage_label.pack(fill="x", pady=(0, 8))

        tracker_body = ttk.Frame(tracker_tab)
        tracker_body.pack(fill="both", expand=True)

        tracker_left = ttk.Frame(tracker_body)
        tracker_left.pack(side="left", fill="both", expand=True)

        setup_frame = ttk.LabelFrame(tracker_left, text="Tracked Setups")
        setup_frame.pack(fill="both", expand=True, pady=(0, 8))
        setup_columns = (
            "scan_date",
            "symbol",
            "side",
            "bucket",
            "entry_price",
            "status",
            "open_scenarios",
            "priority_score",
            "retest",
            "compression",
        )
        self.setup_tracker_table = ttk.Treeview(setup_frame, columns=setup_columns, show="headings", style="Dark.Treeview", height=12)
        tracker_col_widths = {
            "scan_date": 96,
            "symbol": 80,
            "side": 60,
            "bucket": 120,
            "entry_price": 86,
            "status": 86,
            "open_scenarios": 96,
            "priority_score": 90,
            "retest": 110,
            "compression": 100,
        }
        for col in setup_columns:
            self.setup_tracker_table.heading(col, text=col)
            self.setup_tracker_table.column(col, width=tracker_col_widths.get(col, 100), anchor="w")
        setup_scroll = ttk.Scrollbar(setup_frame, orient="vertical", command=self.setup_tracker_table.yview)
        self.setup_tracker_table.configure(yscrollcommand=setup_scroll.set)
        self.setup_tracker_table.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        setup_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.setup_tracker_table.bind("<<TreeviewSelect>>", self._on_setup_tracker_selected)

        scenario_frame = ttk.LabelFrame(tracker_left, text="Scenario Outcomes")
        scenario_frame.pack(fill="both", expand=True)
        scenario_columns = (
            "stop",
            "exit",
            "shares",
            "status",
            "total_r",
            "total_pnl",
            "last_action",
        )
        self.setup_tracker_scenario_table = ttk.Treeview(
            scenario_frame,
            columns=scenario_columns,
            show="headings",
            style="Dark.Treeview",
            height=10,
        )
        scenario_col_widths = {
            "stop": 90,
            "exit": 180,
            "shares": 70,
            "status": 86,
            "total_r": 80,
            "total_pnl": 90,
            "last_action": 260,
        }
        for col in scenario_columns:
            self.setup_tracker_scenario_table.heading(col, text=col)
            self.setup_tracker_scenario_table.column(col, width=scenario_col_widths.get(col, 110), anchor="w")
        scenario_scroll = ttk.Scrollbar(scenario_frame, orient="vertical", command=self.setup_tracker_scenario_table.yview)
        self.setup_tracker_scenario_table.configure(yscrollcommand=scenario_scroll.set)
        self.setup_tracker_scenario_table.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scenario_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)

        tracker_right = ttk.Frame(tracker_body, width=360)
        tracker_right.pack(side="right", fill="y", padx=(12, 0))
        tracker_right.pack_propagate(False)
        tracker_stats_frame = ttk.LabelFrame(tracker_right, text="Tracker Stats / Details")
        tracker_stats_frame.pack(fill="both", expand=True)
        self.setup_tracker_stats_text = tk.Text(tracker_stats_frame, wrap="word", font=("Courier New", 10))
        self.setup_tracker_stats_text.pack(fill="both", expand=True, padx=8, pady=8)

        anchors_tab = ttk.Frame(self.notebook)
        self.anchors_tab = anchors_tab
        self.notebook.add(anchors_tab, text="Earnings Anchors")

        ttk.Label(
            anchors_tab,
            text="Anchor rows refresh automatically after scans, imports, edits, and when you reopen this tab.",
            justify="left",
            wraplength=1100,
        ).pack(fill="x", pady=(0, 8))

        form = ttk.LabelFrame(anchors_tab, text="Manual Anchor Entry")
        form.pack(fill="x", padx=0, pady=(0, 8))

        ttk.Label(form, text="Ticker").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Entry(form, textvariable=self.ticker_var, width=12).grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Label(form, text="Anchor Date (YYYY-MM-DD)").grid(row=0, column=2, padx=6, pady=6, sticky="w")
        ttk.Entry(form, textvariable=self.anchor_var, width=16).grid(row=0, column=3, padx=6, pady=6, sticky="w")
        ttk.Label(form, text="Side").grid(row=0, column=4, padx=6, pady=6, sticky="w")
        ttk.Combobox(form, textvariable=self.side_var, values=("LONG", "SHORT"), width=8, state="readonly").grid(row=0, column=5, padx=6, pady=6, sticky="w")
        ttk.Label(form, text="Notes").grid(row=0, column=6, padx=6, pady=6, sticky="w")
        ttk.Entry(form, textvariable=self.notes_var, width=32).grid(row=0, column=7, padx=6, pady=6, sticky="we")
        ttk.Button(form, text="Add Manual Entry", command=self.add_manual_entry).grid(row=0, column=8, padx=8, pady=6)
        ttk.Button(form, text="Set Side on Selected", command=self.set_selected_anchor_side).grid(row=0, column=9, padx=8, pady=6)
        ttk.Button(form, text="Delete Selected", command=self.delete_selected_anchor).grid(row=0, column=10, padx=8, pady=6)
        form.columnconfigure(7, weight=1)

        anchors_body = ttk.Frame(anchors_tab)
        anchors_body.pack(fill="both", expand=True, padx=0, pady=0)

        table_frame = ttk.Frame(anchors_body)
        table_frame.pack(side="left", fill="both", expand=True)

        self.table = ttk.Treeview(table_frame, columns=EARNINGS_ANCHOR_COLUMNS, show="headings", style="Dark.Treeview")
        for col in EARNINGS_ANCHOR_COLUMNS:
            self.table.heading(col, text=col)
            width = 120 if col not in {"notes"} else 260
            self.table.column(col, width=width, anchor="w")

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=yscroll.set)

        self.table.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        anchors_side = ttk.Frame(anchors_body, width=340)
        anchors_side.pack(side="right", fill="y", padx=(12, 0))
        anchors_side.pack_propagate(False)

        bulk_frame = ttk.LabelFrame(anchors_side, text="Bulk Watchlist Import")
        bulk_frame.pack(fill="both", expand=True)
        ttk.Label(
            bulk_frame,
            text=(
                "Paste comma or newline separated tickers here. Bulk imports resolve the latest earnings event "
                "and anchor the pre-gap day automatically: morning releases anchor the prior session, and "
                "after-close releases anchor the earnings session."
            ),
            justify="left",
            wraplength=300,
        ).pack(fill="x", padx=8, pady=(8, 8))

        bulk_side_row = ttk.Frame(bulk_frame)
        bulk_side_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(bulk_side_row, text="Fallback side:").pack(side="left")
        ttk.Combobox(
            bulk_side_row,
            textvariable=self.bulk_side_var,
            values=("LONG", "SHORT"),
            width=8,
            state="readonly",
        ).pack(side="left", padx=(8, 0))

        self.bulk_anchor_text = tk.Text(bulk_frame, wrap="word", height=12, font=("Courier New", 10))
        self.bulk_anchor_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        bulk_actions = ttk.Frame(bulk_frame)
        bulk_actions.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bulk_actions, text="Add Pasted Tickers", command=self.add_pasted_tickers_to_anchors).pack(side="left")
        ttk.Button(bulk_actions, text="Clear", command=self.clear_bulk_anchor_text).pack(side="left", padx=(8, 0))

        avwap_tab = ttk.Frame(self.notebook)
        self.avwap_tab = avwap_tab
        self.notebook.add(avwap_tab, text="AVWAP Results")

        ttk.Label(
            avwap_tab,
            text="Priority setups, event tickers, and stdev groups refresh automatically after each scan and when you open this tab.",
            justify="left",
            wraplength=1100,
        ).pack(fill="x", pady=(0, 8))

        avwap_body = ttk.Frame(avwap_tab)
        avwap_body.pack(fill="both", expand=True)

        avwap_main = ttk.Frame(avwap_body)
        avwap_main.pack(side="left", fill="both", expand=True)
        self.avwap_text = tk.Text(avwap_main, wrap="word", font=("Courier New", 10))
        self.avwap_text.pack(side="left", fill="both", expand=True)
        output_scroll = ttk.Scrollbar(avwap_main, orient="vertical", command=self.avwap_text.yview)
        self.avwap_text.configure(yscrollcommand=output_scroll.set)
        output_scroll.pack(side="right", fill="y")

        avwap_side = ttk.Frame(avwap_body, width=340)
        avwap_side.pack(side="right", fill="y", padx=(12, 0))
        avwap_side.pack_propagate(False)
        ttk.Label(
            avwap_side,
            text="Comma-separated ticker groups for TradingView paste and bulk watchlist imports.",
            justify="left",
            wraplength=300,
        ).pack(fill="x", pady=(0, 8))

        favorite_frame = ttk.LabelFrame(avwap_side, text="Favorite Setups")
        favorite_frame.pack(fill="x", pady=(0, 10))
        self.favorite_symbols_text = tk.Text(favorite_frame, wrap="word", height=8, font=("Courier New", 10))
        self.favorite_symbols_text.pack(fill="both", expand=True, padx=8, pady=(8, 8))
        ttk.Button(favorite_frame, text="Copy Favorites", command=self.copy_favorite_symbols).pack(anchor="w", padx=8, pady=(0, 8))

        near_favorite_frame = ttk.LabelFrame(avwap_side, text="Near Favorite Zones")
        near_favorite_frame.pack(fill="x")
        self.near_favorite_symbols_text = tk.Text(near_favorite_frame, wrap="word", height=8, font=("Courier New", 10))
        self.near_favorite_symbols_text.pack(fill="both", expand=True, padx=8, pady=(8, 8))
        ttk.Button(
            near_favorite_frame,
            text="Copy Near Favorites",
            command=self.copy_near_favorite_symbols,
        ).pack(anchor="w", padx=8, pady=(0, 8))

        anchor_scan_tab = ttk.Frame(self.notebook)
        self.anchor_scan_tab = anchor_scan_tab
        self.notebook.add(anchor_scan_tab, text="Anchor Results")

        ttk.Label(
            anchor_scan_tab,
            text="Anchor watchlist output refreshes automatically after scans and when you open this tab.",
            justify="left",
            wraplength=1100,
        ).pack(fill="x", pady=(0, 8))
        self.anchor_scan_text = tk.Text(anchor_scan_tab, wrap="word", font=("Courier New", 10))
        self.anchor_scan_text.pack(side="left", fill="both", expand=True)
        anchor_output_scroll = ttk.Scrollbar(anchor_scan_tab, orient="vertical", command=self.anchor_scan_text.yview)
        self.anchor_scan_text.configure(yscrollcommand=anchor_output_scroll.set)
        anchor_output_scroll.pack(side="right", fill="y")

        self._apply_dark_theme_to_text_widgets()

        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill="x", padx=10, pady=(0, 10))
        self.refresh_tracker_storage_summary()

    def refresh_tracker_storage_summary(self):
        details = get_tracker_storage_details()
        shared_longs_path, shared_shorts_path = get_shared_watchlist_paths()
        self.tracker_storage_dir = Path(details["data_dir"])
        self.tracker_storage_runtime_dir = Path(details["runtime_dir"])
        self.tracker_storage_settings_file = Path(details["settings_file"])
        self.tracker_storage_var.set(
            f"Home folder: {details['data_dir']}\n"
            f"Mutable data: {details['mutable_data_dir']}\n"
            f"Logs: {details['logs_dir']}\n"
            f"Reports: {details['output_dir']}\n"
            f"Runtime tracker data: {details['runtime_dir']}\n"
            f"Home-folder longs.txt: {'OK' if shared_longs_path.exists() else 'missing'}\n"
            f"Home-folder shorts.txt: {'OK' if shared_shorts_path.exists() else 'missing'}\n"
            f"Source: {details['source_label']}"
        )

    def _open_folder_in_explorer(self, path: Path):
        try:
            path.mkdir(parents=True, exist_ok=True)
            open_path_in_file_manager(path)
        except Exception as exc:
            if messagebox:
                messagebox.showerror("Open Folder", f"Could not open folder:\n{path}\n\n{exc}")
            else:
                raise

    def choose_tracker_storage_dir(self):
        if filedialog is None:
            self.status_var.set("Folder picker is not available in this environment.")
            return
        initial_dir = self.tracker_storage_dir if getattr(self, "tracker_storage_dir", None) else Path.home()
        selected = filedialog.askdirectory(
            title="Choose home folder",
            initialdir=str(initial_dir if initial_dir.exists() else Path.home()),
            mustexist=False,
        )
        if not selected:
            return
        target = save_tracker_storage_dir(selected)
        self.refresh_tracker_storage_summary()
        self.status_var.set("Saved home folder. Restart the GUI to use the new location.")
        if messagebox:
            messagebox.showinfo(
                "Home Folder Saved",
                "Saved this computer's home folder.\n\n"
                f"Folder: {target}\n"
                f"Settings file: {LOCAL_SETTINGS_FILE}\n\n"
                "Place longs.txt and shorts.txt in that folder root to share watchlists across devices.\n\n"
                "Restart the GUI to start using the new home folder.",
            )

    def open_tracker_storage_dir(self):
        self.refresh_tracker_storage_summary()
        self._open_folder_in_explorer(self.tracker_storage_dir)

    def open_tracker_settings_file(self):
        self.refresh_tracker_storage_summary()
        self._open_folder_in_explorer(self.tracker_storage_settings_file.parent)

    def _refresh_active_tab(self):
        selected_tab = self.notebook.select()
        if selected_tab == str(self.tracker_tab):
            self.refresh_setup_tracker_view()
        elif selected_tab == str(self.anchors_tab):
            self.refresh_table()
        elif selected_tab == str(self.avwap_tab):
            self.refresh_avwap_output_view()
        elif selected_tab == str(self.anchor_scan_tab):
            self.refresh_anchor_output_view()

    def _on_notebook_tab_changed(self, _event=None):
        self._refresh_active_tab()

    def _read_text_file(self, path: Path):
        if not path.exists():
            return f"[Missing file] {path.name}"
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            return f"[Error reading {path.name}] {exc}"

    def _set_text_widget_contents(self, widget, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="normal")

    def _load_ai_state_payload(self) -> dict:
        payload = load_json(AI_STATE_FILE, default={})
        return payload if isinstance(payload, dict) else {}

    def _load_setup_tracker_payload(self) -> dict:
        payload = load_setup_tracker_payload()
        return payload if isinstance(payload, dict) else _default_setup_tracker_payload()

    def copy_setup_tracker_symbols(self):
        symbols = []
        for item_id in self.setup_tracker_table.get_children():
            values = self.setup_tracker_table.item(item_id, "values")
            if len(values) >= 2 and values[1]:
                symbols.append(str(values[1]).strip().upper())
        symbols = sorted(set(symbols))
        if not symbols:
            self.status_var.set("No setup tracker symbols to copy.")
            return
        self._copy_to_clipboard(", ".join(symbols))
        self.status_var.set("Copied setup tracker symbols to clipboard.")

    def _render_setup_tracker_stats_text(self, payload: dict, selected_setup: dict | None = None):
        stats_rows = payload.get("stats", []) if isinstance(payload.get("stats"), list) else []
        daily_watchlists = payload.get("daily_watchlists", {}) if isinstance(payload.get("daily_watchlists"), dict) else {}
        lines = []
        lines.append("Tracker summary")
        lines.append("-" * 80)
        lines.append(f"Updated: {payload.get('updated_at') or 'n/a'}")
        lines.append(f"Tracked setups: {len(payload.get('setups', {}) or {})}")
        if daily_watchlists:
            latest_date = max(daily_watchlists.keys())
            latest_watchlist = daily_watchlists.get(latest_date, {})
            lines.append(
                f"Latest final scan day: {latest_date} | symbols={len(latest_watchlist.get('symbols', []) or [])}"
            )
            lines.append("")
            lines.append("Recent daily watchlists")
            lines.append("-" * 80)
            for watchlist_date in sorted(daily_watchlists.keys(), reverse=True)[:8]:
                watchlist = daily_watchlists.get(watchlist_date, {})
                symbols = [str(symbol).strip().upper() for symbol in watchlist.get("symbols", []) if str(symbol).strip()]
                preview = ", ".join(symbols[:10]) if symbols else "None"
                if len(symbols) > 10:
                    preview += f" (+{len(symbols) - 10} more)"
                lines.append(f"{watchlist_date}: {preview}")
        lines.append("")
        lines.append("Top scenario stats")
        lines.append("-" * 80)
        if not stats_rows:
            lines.append("No scenario stats yet.")
        else:
            for row in stats_rows[:8]:
                avg_r = row.get("avg_total_r")
                avg_r_text = f"{avg_r:.2f}R" if avg_r is not None else "n/a"
                win_rate = row.get("win_rate_closed")
                win_rate_text = f"{win_rate * 100:.0f}%" if win_rate is not None else "n/a"
                lines.append(
                    f"{row.get('stop_reference_label')} / {row.get('exit_template_id')}: "
                    f"avg_total={avg_r_text} closed={row.get('closed_setups', 0)} win={win_rate_text}"
                )

        if selected_setup:
            lines.append("")
            lines.append("Selected setup")
            lines.append("-" * 80)
            lines.append(
                f"{selected_setup.get('symbol')} {selected_setup.get('side')} "
                f"{selected_setup.get('priority_bucket', '')} scan={selected_setup.get('scan_date')}"
            )
            lines.append(
                f"entry={_coerce_float(selected_setup.get('entry_price')) or 0:.2f} "
                f"anchor={selected_setup.get('anchor_date')} "
                f"score={_coerce_float(selected_setup.get('priority_score')) or 0:.1f}"
            )
            if selected_setup.get("favorite_zone"):
                lines.append(f"zone={selected_setup.get('favorite_zone')}")
            if selected_setup.get("retest_note"):
                lines.append(f"retest={selected_setup.get('retest_note')}")
            if selected_setup.get("extension_note"):
                lines.append(f"extension={selected_setup.get('extension_note')}")
            if selected_setup.get("score_bonus_note"):
                lines.append(f"bonus={selected_setup.get('score_bonus_note')}")
            if selected_setup.get("ranking_note"):
                lines.append(f"filters={selected_setup.get('ranking_note')}")
            if selected_setup.get("compression_note"):
                lines.append(f"compression={selected_setup.get('compression_note')}")

        lines.append("")
        lines.append("Exports")
        lines.append("-" * 80)
        lines.append(f"Tracker JSON: {SETUP_TRACKER_FILE}")
        lines.append(f"Scenario CSV: {SETUP_SCENARIOS_FILE}")
        lines.append(f"Daily CSV: {SETUP_DAILY_FILE}")
        lines.append(f"Stats CSV: {SETUP_STATS_FILE}")
        self._set_text_widget_contents(self.setup_tracker_stats_text, "\n".join(lines))

    def _populate_setup_tracker_scenarios(self, setup: dict | None):
        for item in self.setup_tracker_scenario_table.get_children():
            self.setup_tracker_scenario_table.delete(item)
        if not setup:
            return

        scenarios = list((setup.get("scenarios") or {}).values())
        scenarios.sort(key=lambda item: (not _scenario_is_open(item.get("status", "")), -(float(item.get("total_r", 0.0) or 0.0)), item.get("scenario_id", "")))
        for scenario in scenarios:
            total_r = _coerce_float(scenario.get("total_r"))
            total_pnl = _coerce_float(scenario.get("total_pnl"))
            self.setup_tracker_scenario_table.insert(
                "",
                "end",
                values=(
                    scenario.get("stop_reference_label", ""),
                    scenario.get("exit_template_label", ""),
                    int(scenario.get("shares", 0) or 0),
                    scenario.get("status", ""),
                    "" if total_r is None else f"{total_r:.2f}",
                    "" if total_pnl is None else f"{total_pnl:.2f}",
                    scenario.get("last_action", ""),
                ),
            )

    def _on_setup_tracker_selected(self, _event=None):
        selected = self.setup_tracker_table.selection()
        setup = self.setup_tracker_row_map.get(selected[0]) if selected else None
        self._populate_setup_tracker_scenarios(setup)
        self._render_setup_tracker_stats_text(self.setup_tracker_payload, setup)

    def refresh_setup_tracker_view(self):
        self.refresh_tracker_storage_summary()
        payload = self._load_setup_tracker_payload()
        self.setup_tracker_payload = payload
        self.setup_tracker_row_map = {}

        for item in self.setup_tracker_table.get_children():
            self.setup_tracker_table.delete(item)

        setups = list((payload.get("setups") or {}).values())
        setups.sort(
            key=lambda setup: (
                setup.get("setup_status") != "OPEN",
                str(setup.get("scan_date") or ""),
                str(setup.get("symbol") or ""),
            ),
            reverse=False,
        )
        setups = sorted(
            setups,
            key=lambda setup: (
                setup.get("setup_status") != "OPEN",
                str(setup.get("scan_date") or ""),
                str(setup.get("symbol") or ""),
            ),
        )
        setups = sorted(
            setups,
            key=lambda setup: (
                setup.get("setup_status") != "OPEN",
                -(float(setup.get("priority_score", 0) or 0.0)),
                str(setup.get("symbol") or ""),
            ),
        )

        for setup in setups:
            item_id = self.setup_tracker_table.insert(
                "",
                "end",
                values=(
                    setup.get("scan_date", ""),
                    setup.get("symbol", ""),
                    setup.get("side", ""),
                    setup.get("priority_bucket", ""),
                    "" if _coerce_float(setup.get("entry_price")) is None else f"{float(setup.get('entry_price')):.2f}",
                    setup.get("setup_status", ""),
                    int(setup.get("open_scenario_count", 0) or 0),
                    "" if _coerce_float(setup.get("priority_score")) is None else f"{float(setup.get('priority_score')):.1f}",
                    setup.get("retest_reference_level", ""),
                    "Y" if setup.get("compression_flag") else "",
                ),
            )
            self.setup_tracker_row_map[item_id] = setup

        first_item = self.setup_tracker_table.get_children()
        if first_item:
            self.setup_tracker_table.selection_set(first_item[0])
            self.setup_tracker_table.focus(first_item[0])
            self._on_setup_tracker_selected()
        else:
            self._populate_setup_tracker_scenarios(None)
            self._render_setup_tracker_stats_text(payload, None)

    def _highlight_trendline_candidates_in_avwap_output(self):
        self.avwap_text.tag_remove("trendline_bold", "1.0", tk.END)
        payload = self._load_ai_state_payload()
        raw_symbols = payload.get("symbols", {}) if isinstance(payload, dict) else {}
        highlighted = {
            str(symbol).strip().upper()
            for symbol, entry in (raw_symbols.items() if isinstance(raw_symbols, dict) else [])
            if isinstance(entry, dict) and (
                entry.get("priority_trendline_within_alert_range")
                or entry.get("priority_trendline_break_recent")
            )
        }
        if not highlighted:
            return

        section_end = self.avwap_text.search("MASTER AVWAP EVENT TICKERS", "1.0", stopindex=tk.END)
        if section_end:
            total_lines = max(0, int(section_end.split(".")[0]) - 1)
        else:
            total_lines = int(self.avwap_text.index("end-1c").split(".")[0])

        for line_no in range(1, total_lines + 1):
            line_start = f"{line_no}.0"
            line_end = f"{line_no}.end"
            line_text = self.avwap_text.get(line_start, line_end).strip()
            if not line_text or "score=" not in line_text:
                continue
            parts = line_text.split()
            if len(parts) < 2 or parts[1].upper() not in {"LONG", "SHORT"}:
                continue
            symbol = parts[0].strip(",").upper()
            if symbol in highlighted:
                self.avwap_text.tag_add("trendline_bold", line_start, line_end)

    def _load_tradingview_groups(self) -> dict | None:
        text = self._read_text_file(TRADINGVIEW_REPORT_FILE)
        if not text or text.startswith("[Missing file]") or text.startswith("[Error reading"):
            return None

        section_lookup = {
            "Best current favorite setups": "favorites",
            "Near favorite zones": "near_favorite_zones",
        }
        groups = {
            "favorites": {"LONG": [], "SHORT": []},
            "near_favorite_zones": {"LONG": [], "SHORT": []},
        }

        current_section = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                current_section = None
                continue
            if line in section_lookup:
                current_section = section_lookup[line]
                continue
            if line.startswith("-"):
                continue
            if current_section not in groups or ":" not in line:
                continue

            side_label, values = line.split(":", 1)
            side = side_label.strip().upper()
            if side not in ("LONG", "SHORT"):
                continue
            groups[current_section][side] = _extract_symbols_from_text(values)

        return groups

    def _load_focus_payload(self) -> dict:
        payload = load_json(MASTER_AVWAP_FOCUS_FILE, default={})
        return payload if isinstance(payload, dict) else {}

    def refresh_focus_group_boxes(self):
        self.focus_side_map = {}
        payload = self._load_focus_payload()
        favorites = payload.get("favorites", [])
        near_favorites = payload.get("near_favorite_zones", [])
        symbol_map = payload.get("symbols", {})

        if isinstance(symbol_map, dict):
            for symbol, row in symbol_map.items():
                ticker = str(symbol).strip().upper()
                if not ticker or not isinstance(row, dict):
                    continue
                self.focus_side_map[ticker] = normalize_side(row.get("side", "LONG"))

        favorite_symbols = [
            str(entry.get("symbol", "")).strip().upper()
            for entry in favorites
            if isinstance(entry, dict) and str(entry.get("symbol", "")).strip()
        ]
        near_favorite_symbols = [
            str(entry.get("symbol", "")).strip().upper()
            for entry in near_favorites
            if isinstance(entry, dict) and str(entry.get("symbol", "")).strip()
        ]

        if favorite_symbols or near_favorite_symbols:
            favorite_text = _format_symbol_group(favorite_symbols)
            near_favorite_text = _format_symbol_group(near_favorite_symbols)
        else:
            tradingview_groups = self._load_tradingview_groups()
            if tradingview_groups:
                favorite_longs = tradingview_groups["favorites"]["LONG"]
                favorite_shorts = tradingview_groups["favorites"]["SHORT"]
                near_longs = tradingview_groups["near_favorite_zones"]["LONG"]
                near_shorts = tradingview_groups["near_favorite_zones"]["SHORT"]

                for symbol in favorite_longs + near_longs:
                    self.focus_side_map[symbol] = "LONG"
                for symbol in favorite_shorts + near_shorts:
                    self.focus_side_map[symbol] = "SHORT"

                favorite_text = _format_symbol_group(favorite_longs + favorite_shorts)
                near_favorite_text = _format_symbol_group(near_longs + near_shorts)
            else:
                favorite_text = "None"
                near_favorite_text = "None"

        self._set_text_widget_contents(self.favorite_symbols_text, favorite_text)
        self._set_text_widget_contents(self.near_favorite_symbols_text, near_favorite_text)

    def _copy_text_widget_contents(self, widget, empty_message: str, success_message: str):
        text = widget.get("1.0", tk.END).strip()
        if not text or text == "None":
            self.status_var.set(empty_message)
            return
        self._copy_to_clipboard(text)
        self.status_var.set(success_message)

    def copy_favorite_symbols(self):
        self._copy_text_widget_contents(
            self.favorite_symbols_text,
            "No favorite setup symbols to copy.",
            "Copied favorite setup symbols to clipboard.",
        )

    def copy_near_favorite_symbols(self):
        self._copy_text_widget_contents(
            self.near_favorite_symbols_text,
            "No near favorite zone symbols to copy.",
            "Copied near favorite zone symbols to clipboard.",
        )

    def clear_bulk_anchor_text(self):
        self.bulk_anchor_text.delete("1.0", tk.END)
        self.status_var.set("Cleared pasted ticker input.")

    def refresh_avwap_output_view(self):
        priority_section = self._read_text_file(PRIORITY_SETUPS_FILE)
        event_section = self._read_text_file(EVENT_TICKERS_FILE)
        stdev_section = self._read_text_file(STDEV_RANGE_FILE)

        combined = (
            "MASTER AVWAP PRIORITY SETUPS\n"
            + "=" * 80
            + "\n"
            + (priority_section or "No priority setup output yet.")
            + "\n\n"
            "MASTER AVWAP EVENT TICKERS\n"
            + "=" * 80
            + "\n"
            + (event_section or "No event tickers output yet.")
            + "\n\n"
            + "MASTER AVWAP STDEV 2-3 OUTPUT\n"
            + "=" * 80
            + "\n"
            + (stdev_section or "No stdev output yet.")
            + "\n"
        )

        self._set_text_widget_contents(self.avwap_text, combined)
        self._highlight_trendline_candidates_in_avwap_output()
        self.refresh_focus_group_boxes()

    def refresh_anchor_output_view(self):
        text = self._read_text_file(ANCHOR_AVWAP_OUTPUT_FILE)
        self._set_text_widget_contents(self.anchor_scan_text, text or "No anchor AVWAP output yet.")

    def _copy_to_clipboard(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()

    def _run_background(self, target, running_msg, done_msg, done_callback=None):
        self.status_var.set(running_msg)

        def _task():
            try:
                target()
                self.root.after(0, lambda: self.status_var.set(done_msg))
                self.root.after(0, self.refresh_table)
                if done_callback:
                    self.root.after(0, done_callback)
            except Exception as exc:
                logging.exception("GUI background task failed")
                error_message = f"Error: {exc}"
                self.root.after(0, lambda msg=error_message: self.status_var.set(msg))

        threading.Thread(target=_task, daemon=True).start()

    def run_earnings_scan(self):
        try:
            lookback_days = max(1, int(self.earnings_lookback_var.get()))
        except Exception:
            lookback_days = 1
            self.earnings_lookback_var.set(1)

        self.notebook.select(self.anchors_tab)
        self._run_background(
            lambda: scan_last_session_earnings_for_anchors(lookback_days=lookback_days),
            f"Running earnings gap anchor scan (last {lookback_days} session(s))...",
            "Earnings gap candidate scan complete.",
            done_callback=lambda: self.notebook.select(self.anchors_tab),
        )

    def run_master_once(self):
        self.notebook.select(self.avwap_tab)
        self._run_background(
            run_master,
            "Running full Master AVWAP scan...",
            "Master AVWAP scan complete.",
            done_callback=lambda: (
                self.refresh_avwap_output_view(),
                self.refresh_anchor_output_view(),
                self.refresh_setup_tracker_view(),
            ),
        )

    def backfill_setup_tracker_history(self):
        try:
            lookback_sessions = max(1, int(self.tracker_backfill_sessions_var.get()))
        except Exception:
            lookback_sessions = 5
            self.tracker_backfill_sessions_var.set(lookback_sessions)

        self.notebook.select(self.tracker_tab)
        self._run_background(
            lambda: backfill_setup_tracker_from_recent_sessions(lookback_sessions=lookback_sessions),
            f"Backfilling tracker from last {lookback_sessions} session(s)...",
            f"Setup tracker backfill complete for last {lookback_sessions} session(s).",
            done_callback=self.refresh_setup_tracker_view,
        )

    def run_shared_watchlist_scan_once(self):
        self.notebook.select(self.avwap_tab)
        self._run_background(
            run_master_with_shared_watchlists,
            "Running Master AVWAP scan from home-folder longs.txt / shorts.txt...",
            "Home-folder Master AVWAP scan complete.",
            done_callback=lambda: (
                self.refresh_avwap_output_view(),
                self.refresh_anchor_output_view(),
                self.refresh_setup_tracker_view(),
            ),
        )

    def run_anchor_scan_once(self):
        self.notebook.select(self.anchor_scan_tab)
        self._run_background(
            run_anchor_watchlist_scan,
            "Running anchor watchlist AVWAP scan...",
            "Anchor watchlist AVWAP scan complete.",
            done_callback=self.refresh_anchor_output_view,
        )

    def refresh_table(self):
        ensure_anchor_file()
        df = pd.read_csv(EARNINGS_ANCHORS_FILE)

        for item in self.table.get_children():
            self.table.delete(item)

        if df.empty:
            return

        df = df.fillna("")
        for _, row in df.iterrows():
            self.table.insert("", "end", values=[row.get(col, "") for col in EARNINGS_ANCHOR_COLUMNS])

    def add_pasted_tickers_to_anchors(self):
        symbols = _extract_symbols_from_text(self.bulk_anchor_text.get("1.0", tk.END))
        if not symbols:
            self.status_var.set("No tickers found in the pasted input.")
            return

        fallback_side = normalize_side(self.bulk_side_var.get())
        self.status_var.set(f"Resolving anchors for {len(symbols)} pasted ticker(s)...")

        def _task():
            try:
                result = resolve_bulk_anchor_candidates(
                    symbols,
                    default_side=fallback_side,
                    focus_side_map=self.focus_side_map,
                )
                candidates = result.get("candidates", [])
                unresolved = result.get("unresolved", [])
                added = append_anchor_candidates(candidates)
                duplicates = len(candidates) - added

                def _finish():
                    self.refresh_table()
                    self.notebook.select(self.anchors_tab)
                    summary = (
                        f"Added {added} anchor row(s)"
                        f" ({result.get('candidate_matches', 0)} from earnings scan, "
                        f"{result.get('earnings_matches', 0)} from direct earnings resolution)."
                    )
                    if duplicates > 0:
                        summary += f" {duplicates} already existed."
                    if unresolved:
                        preview = ", ".join(unresolved[:6])
                        more = "" if len(unresolved) <= 6 else f" (+{len(unresolved) - 6} more)"
                        summary += f" Unresolved: {preview}{more}."
                    self.status_var.set(summary)

                self.root.after(0, _finish)
            except Exception as exc:
                logging.exception("Bulk anchor import failed")
                error_message = f"Bulk import failed: {exc}"
                self.root.after(0, lambda msg=error_message: self.status_var.set(msg))

        threading.Thread(target=_task, daemon=True).start()

    def set_selected_anchor_side(self):
        selected = self.table.selection()
        if not selected:
            self.status_var.set("No row selected to update side.")
            return

        values = self.table.item(selected[0], "values")
        if not values:
            return

        ticker = str(values[0]).upper().strip()
        anchor_date = str(values[2]).strip()
        side = normalize_side(self.side_var.get())

        ensure_anchor_file()
        df = pd.read_csv(EARNINGS_ANCHORS_FILE)
        mask = (df["ticker"].astype(str).str.upper() == ticker) & (df["anchor_date"].astype(str) == anchor_date)
        if not mask.any():
            self.status_var.set(f"Entry not found: {ticker} {anchor_date}")
            return

        df.loc[mask, "side"] = side
        df.to_csv(EARNINGS_ANCHORS_FILE, index=False)
        self.status_var.set(f"Updated side for {ticker} {anchor_date} -> {side}")
        self.refresh_table()

    def delete_selected_anchor(self):
        selected = self.table.selection()
        if not selected:
            self.status_var.set("No row selected to delete.")
            return

        values = self.table.item(selected[0], "values")
        if not values:
            return

        ticker = str(values[0]).upper().strip()
        anchor_date = str(values[2]).strip()

        ensure_anchor_file()
        df = pd.read_csv(EARNINGS_ANCHORS_FILE)
        before = len(df)
        if before == 0:
            return

        mask = ~((df["ticker"].astype(str).str.upper() == ticker) & (df["anchor_date"].astype(str) == anchor_date))
        df = df[mask]
        if len(df) == before:
            self.status_var.set(f"Entry not found: {ticker} {anchor_date}")
            return

        df.to_csv(EARNINGS_ANCHORS_FILE, index=False)
        self.status_var.set(f"Deleted anchor entry: {ticker} {anchor_date}")
        self.refresh_table()

    def add_manual_entry(self):
        ticker = self.ticker_var.get().strip().upper()
        anchor_date = self.anchor_var.get().strip()
        side = normalize_side(self.side_var.get())
        notes = self.notes_var.get().strip()

        if not ticker or not anchor_date:
            if messagebox:
                messagebox.showerror("Missing data", "Ticker and anchor date are required.")
            return

        try:
            datetime.fromisoformat(anchor_date)
        except ValueError:
            if messagebox:
                messagebox.showerror("Invalid date", "Anchor date must be in YYYY-MM-DD format.")
            return

        manual_candidate = EarningsGapAnchorCandidate(
            ticker=ticker,
            anchor_date=anchor_date,
            side=side,
            gap_date="",
            earnings_date="",
            release_session="manual",
            gap_atr_multiple=0.0,
            price=0.0,
            avg_volume20=0,
            market_cap=0,
            notes=notes,
            source="manual",
        )

        added = append_anchor_candidates([manual_candidate])
        if added == 0:
            self.status_var.set(f"Entry already exists: {ticker} {anchor_date}")
        else:
            self.status_var.set(f"Manual entry added: {ticker} {anchor_date}")
        self.refresh_table()

def launch_gui():
    if tk is None or ttk is None:
        logging.error("tkinter is unavailable in this Python environment; cannot launch GUI.")
        return

    ensure_anchor_file()
    ensure_anchor_candidate_file()
    root = tk.Tk()
    MasterAvwapGUI(root, standalone=True)
    root.mainloop()


# ============================================================================
# ENTRYPOINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run Master AVWAP scanner")
    parser.add_argument("--once", action="store_true", help="Run a single AVWAP scan and exit.")
    parser.add_argument("--loop", action="store_true", help="Run AVWAP scan in hourly loop.")
    parser.add_argument(
        "--scan-earnings",
        action="store_true",
        help="Scan the last market session for earnings gap anchor candidates and write the review list.",
    )
    parser.add_argument("--gui", action="store_true", help="Launch the Master AVWAP management GUI.")
    parser.add_argument("--anchor-scan", action="store_true", help="Run only the anchor watchlist AVWAP scan.")
    args = parser.parse_args()

    if args.scan_earnings:
        scan_last_session_earnings_for_anchors()

    if args.anchor_scan:
        run_anchor_watchlist_scan(archive_expired=True)

    if args.once:
        run_master()
        return

    if args.loop:
        logging.info("Starting hourly Master AVWAP loop (once per hour)…")
        while True:
            start = time.time()
            run_master()
            elapsed = time.time() - start
            sleep_seconds = max(0, 3600 - elapsed)
            logging.info(f"Sleeping {int(sleep_seconds)} seconds until next run…")
            time.sleep(sleep_seconds)

    if args.gui or (not args.once and not args.loop and not args.scan_earnings and not args.anchor_scan):
        launch_gui()


if __name__ == "__main__":
    main()
