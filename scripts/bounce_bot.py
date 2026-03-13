# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import re
from pathlib import Path
# Suppress Tk deprecation warnings on macOS
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import math
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import zoneinfo
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import tkinter.font as tkFont

import pandas as pd
import yfinance as yf

# IB API imports
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# For colored console output (still used for logging)
from colorama import init, Fore, Style
init(autoreset=True)

from project_paths import (
    DATA_DIR,
    LOG_DIR,
    ROOT_DIR,
    LONGS_FILE,
    SHORTS_FILE,
    BOUNCE_LOG_FILE,
    TRADING_BOT_LOG_FILE,
    INTRADAY_BOUNCES_FILE,
    AVWAP_SIGNALS_FILE,
    RRS_STRENGTH_LOG_FILE,
    RRS_GROUP_STRENGTH_LOG_FILE,
    SECTOR_ETF_MAP_FILE,
    INDUSTRY_ETF_MAP_FILE,
    SYMBOL_CLASSIFICATION_CACHE_FILE,
    MASTER_AVWAP_FOCUS_FILE,
    APP_LOG_BACKUP_COUNT,
    get_tracker_storage_details,
    get_local_setting,
    save_local_setting,
    save_tracker_storage_dir,
)

##########################################
# Adjustable Parameters
##########################################
LONGS_FILENAME = LONGS_FILE
SHORTS_FILENAME = SHORTS_FILE
BOUNCE_LOG_FILENAME = BOUNCE_LOG_FILE
TRADING_BOT_LOG_FILENAME = TRADING_BOT_LOG_FILE
INTRADAY_BOUNCES_CSV = INTRADAY_BOUNCES_FILE
MASTER_AVWAP_SIGNALS_FILENAME = AVWAP_SIGNALS_FILE
MASTER_AVWAP_FOCUS_FILENAME = MASTER_AVWAP_FOCUS_FILE
STRENGTH_SCAN_LOG_FILENAME = RRS_STRENGTH_LOG_FILE
GROUP_STRENGTH_SCAN_LOG_FILENAME = RRS_GROUP_STRENGTH_LOG_FILE
SECTOR_ETF_MAP_FILENAME = SECTOR_ETF_MAP_FILE
INDUSTRY_ETF_MAP_FILENAME = INDUSTRY_ETF_MAP_FILE
SYMBOL_CLASSIFICATION_CACHE_FILENAME = SYMBOL_CLASSIFICATION_CACHE_FILE
ATR_PERIOD = 20
THRESHOLD_MULTIPLIER = 0.02
CONSECUTIVE_CANDLES = 6  # Number of candles price must respect level before bounce
CHECK_CONSECUTIVE_CANDLES = True  # Parameter to enable/disable this check
CHECK_BOUNCE_VVWAP = True
CHECK_BOUNCE_DYNAMIC_VVWAP = True
CHECK_BOUNCE_EOD_VWAP = True 
CHECK_BOUNCE_VWAP_EOD_CONFLUENCE = True
CHECK_BOUNCE_IMPULSE_RETEST_VWAP_EOD = True
CHECK_BOUNCE_8_EMA = True
CHECK_BOUNCE_15_EMA = True
CHECK_BOUNCE_21_EMA = False
CHECK_BOUNCE_10_CANDLE = False
CHECK_BOUNCE_PREV_DAY_HIGH = True
CHECK_BOUNCE_PREV_DAY_LOW = True
CHECK_BOUNCE_VWAP_UPPER_BAND = False 
CHECK_BOUNCE_VWAP_LOWER_BAND = False  
CHECK_BOUNCE_DYNAMIC_VWAP_UPPER_BAND = False 
CHECK_BOUNCE_DYNAMIC_VWAP_LOWER_BAND = False  
CHECK_BOUNCE_EOD_VWAP_UPPER_BAND = False
CHECK_BOUNCE_EOD_VWAP_LOWER_BAND = False 
LOGGING_MODE = True
SCAN_OUTSIDE_MARKET_HOURS = True
LOG_PRICE_APPROACHING = True
USE_GUI = True  # New parameter to toggle GUI on/off
RECLAIM_LOOKBACK_CANDLES = 3
CONFLUENCE_MAX_SPREAD_ATR = 0.25
VWAP_BOUNCE_NEAR_ATR = 0.08
VWAP_BOUNCE_PIERCE_ATR = 0.08
VWAP_BOUNCE_CLOSE_BUFFER_ATR = 0.03
IMPULSE_LOOKBACK_BARS = 12
IMPULSE_MAX_BARS = 6
IMPULSE_MIN_ATR = 0.8
IMPULSE_INTRADAY_ATR_PERIOD = 14
IMPULSE_RETRACE_MIN_FRAC = 0.05
IMPULSE_RETRACE_MAX_FRAC = 0.90
IMPULSE_RETEST_MAX_BARS = 8
IMPULSE_RETEST_PIERCE_ATR = 0.40
IMPULSE_RETEST_CLOSE_BUFFER_ATR = 0.02
IMPULSE_CHOP_LOOKBACK_BARS = 8
IMPULSE_MAX_NEAR_BARS = 6
IMPULSE_LEVEL_BAND_ATR = 0.14
IMPULSE_MAX_CROSSES = 2

BOUNCE_TYPE_DEFAULTS = {
    "10_candle": CHECK_BOUNCE_10_CANDLE,
    "vwap": CHECK_BOUNCE_VVWAP,
    "dynamic_vwap": CHECK_BOUNCE_DYNAMIC_VVWAP,
    "eod_vwap": CHECK_BOUNCE_EOD_VWAP,
    "vwap_eod_confluence": CHECK_BOUNCE_VWAP_EOD_CONFLUENCE,
    "impulse_retest_vwap_eod": CHECK_BOUNCE_IMPULSE_RETEST_VWAP_EOD,
    "ema_8": CHECK_BOUNCE_8_EMA,
    "ema_15": CHECK_BOUNCE_15_EMA,
    "ema_21": CHECK_BOUNCE_21_EMA,
    "vwap_upper_band": CHECK_BOUNCE_VWAP_UPPER_BAND,
    "vwap_lower_band": CHECK_BOUNCE_VWAP_LOWER_BAND,
    "dynamic_vwap_upper_band": CHECK_BOUNCE_DYNAMIC_VWAP_UPPER_BAND,
    "dynamic_vwap_lower_band": CHECK_BOUNCE_DYNAMIC_VWAP_LOWER_BAND,
    "eod_vwap_upper_band": CHECK_BOUNCE_EOD_VWAP_UPPER_BAND,
    "eod_vwap_lower_band": CHECK_BOUNCE_EOD_VWAP_LOWER_BAND,
    "prev_day_high": CHECK_BOUNCE_PREV_DAY_HIGH,
    "prev_day_low": CHECK_BOUNCE_PREV_DAY_LOW,
}

BOUNCE_TYPE_LABELS = {
    "10_candle": "10-Candle",
    "vwap": "Std VWAP",
    "dynamic_vwap": "Dynamic VWAP",
    "eod_vwap": "EOD VWAP",
    "vwap_eod_confluence": "VWAP+EOD Confluence",
    "impulse_retest_vwap_eod": "Impulse Retest VWAP/EOD",
    "ema_8": "8 EMA",
    "ema_15": "15 EMA",
    "ema_21": "21 EMA",
    "vwap_upper_band": "VWAP 1SD Upper",
    "vwap_lower_band": "VWAP 1SD Lower",
    "dynamic_vwap_upper_band": "Dynamic VWAP 1SD Upper",
    "dynamic_vwap_lower_band": "Dynamic VWAP 1SD Lower",
    "eod_vwap_upper_band": "EOD VWAP 1SD Upper",
    "eod_vwap_lower_band": "EOD VWAP 1SD Lower",
    "prev_day_high": "Previous Day High",
    "prev_day_low": "Previous Day Low",
}

# Connection & Request settings
MAX_CONCURRENT_REQUESTS = 1
REQUEST_DELAY = 0.1  # seconds between IB historical data requests

# RRS (Real Relative Strength) settings
RRS_DEFAULT_THRESHOLD = 2.0
RRS_LENGTH = 12
RRS_TIMEOUT = 3.0
RRS_TIMEFRAMES = {
    "5m": {"label": "5 min", "bar_size": "5 mins", "duration": "5 D", "minutes": 5},
    "15m": {"label": "15 min", "bar_size": "15 mins", "duration": "5 D", "minutes": 15},
    "30m": {"label": "30 min", "bar_size": "30 mins", "duration": "5 D", "minutes": 30},
    "1h": {"label": "1 hour", "bar_size": "1 hour", "duration": "5 D", "minutes": 60},
}
SCAN_EXTREME_COUNT = 5
GROUP_STRENGTH_TIMEFRAMES = {
    "D1": {"bar_size": "1 day", "duration": "6 M"},
    "H1": {"bar_size": "1 hour", "duration": "5 D"},
    "M5": {"bar_size": "5 mins", "duration": "5 D"},
}
ENVIRONMENT_FOCUS_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]+$")
DEFAULT_SECTOR_ETF_MAP = {
    "communication-services": "XLC",
    "consumer-cyclical": "XLY",
    "consumer-defensive": "XLP",
    "energy": "XLE",
    "financial-services": "XLF",
    "healthcare": "XLV",
    "industrials": "XLI",
    "basic-materials": "XLB",
    "real-estate": "XLRE",
    "technology": "XLK",
    "utilities": "XLU",
}

MARKET_ENVIRONMENTS = {
    "bearish_strong": {"label": "Bearish Strong"},
    "bearish_weak": {"label": "Bearish Weak"},
    "bullish_strong": {"label": "Bullish Strong"},
    "bullish_weak": {"label": "Bullish Weak"},
}
MIN_MOVE_RATIO_FOR_SIGNAL = 0.25
MIN_EXCESS_MOVE_RATIO_FOR_SIGNAL = 0.15
MASTER_AVWAP_FOCUS_MIN_ABS_RRS = 2.75
MASTER_AVWAP_FOCUS_MIN_MOVE_RATIO = 0.45
MASTER_AVWAP_FOCUS_MIN_EXCESS_MOVE_RATIO = 0.25
EMIT_MASTER_AVWAP_FOCUS_RRS_ALERTS = False
ENVIRONMENT_HIGHLIGHT_LIMIT = 6
ENVIRONMENT_SCAN_LIMIT = 25
SPY_COMPRESSION_THRESHOLD = 0.35
SPY_UP_THRESHOLD = 0.20
SPY_PULLBACK_DELTA_THRESHOLD = 0.25
IMPULSE_RRS_PROFILE_LENGTH = 6
IMPULSE_RRS_RECENT_PHASE_BARS = 3
IMPULSE_RRS_FAVORED_TREND_RATIO = 0.60
IMPULSE_RRS_FAVORED_EXCESS_RATIO = 0.20
IMPULSE_RRS_UNFAVORED_TREND_RATIO = 0.90
IMPULSE_RRS_UNFAVORED_EXCESS_RATIO = 0.45
IMPULSE_RRS_COUNTER_RRS = 0.35
IMPULSE_RRS_COUNTER_MOVE_RATIO = 0.20
IMPULSE_RRS_COUNTER_BARS_REQUIRED = 2
BOUNCE_LEVEL_LOOKBACK_BARS = 8
BOUNCE_LEVEL_BAND_ATR = 0.12
BOUNCE_MAX_NEAR_BARS = 4
BOUNCE_MAX_CROSSES = 2
BOUNCE_MIN_SEPARATION_ATR = 0.45
BOUNCE_MIN_LEVEL_SLOPE_ATR = 0.04
BOUNCE_LEVEL_SLOPE_LOOKBACK = 3
BOUNCE_ONE_SIDED_BARS = 4
BOUNCE_ONE_SIDED_RATIO = 0.75
BOUNCE_REACTION_LOOKBACK_BARS = 6
BOUNCE_MIN_BODY_TO_MEDIAN = 1.10
BOUNCE_MIN_RANGE_TO_MEDIAN = 1.05
BOUNCE_VOLUME_LOOKBACK_BARS = 6
BOUNCE_MIN_RELATIVE_VOLUME = 1.10
BOUNCE_CONFIRMATION_MAX_CANDLES = 3
VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES = 4
APP_LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s]: %(message)s"


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify_key(value):
    if value is None:
        return ""
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return text.strip("-")


def load_sector_etf_map():
    SECTOR_ETF_MAP_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    if not SECTOR_ETF_MAP_FILENAME.exists():
        with open(SECTOR_ETF_MAP_FILENAME, "w") as fh:
            json.dump(DEFAULT_SECTOR_ETF_MAP, fh, indent=2, sort_keys=True)
    try:
        with open(SECTOR_ETF_MAP_FILENAME, "r") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            normalized = {}
            for k, v in data.items():
                if not v:
                    continue
                key = slugify_key(k)
                if key:
                    normalized[key] = str(v).strip().upper()
            if normalized:
                return normalized
    except Exception as exc:
        logging.warning(f"Failed loading sector ETF map: {exc}")
    return dict(DEFAULT_SECTOR_ETF_MAP)


def _load_industry_etf_map_file():
    INDUSTRY_ETF_MAP_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    default_map = {"version": 1, "updated_utc": utc_now_iso(), "yahoo_industryKey_to_ref": {}}
    if not INDUSTRY_ETF_MAP_FILENAME.exists():
        with open(INDUSTRY_ETF_MAP_FILENAME, "w") as fh:
            json.dump(default_map, fh, indent=2, sort_keys=True)
        return default_map
    try:
        with open(INDUSTRY_ETF_MAP_FILENAME, "r") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    data.setdefault("version", 1)
    data.setdefault("updated_utc", utc_now_iso())
    data.setdefault("yahoo_industryKey_to_ref", {})
    return data


def load_and_update_industry_etf_map(industryKey, sectorKey, industry_name, sector_name):
    data = _load_industry_etf_map_file()
    refs = data.setdefault("yahoo_industryKey_to_ref", {})
    now = utc_now_iso()
    key = (industryKey or "").strip()
    if key:
        entry = refs.get(key)
        if not isinstance(entry, dict):
            entry = {
                "sectorKey": sectorKey or "",
                "industry": industry_name or "",
                "sector": sector_name or "",
                "etf": None,
                "first_seen_utc": now,
                "last_seen_utc": now,
                "seen_count": 1,
            }
        else:
            entry["sectorKey"] = entry.get("sectorKey") or (sectorKey or "")
            entry["industry"] = entry.get("industry") or (industry_name or "")
            entry["sector"] = entry.get("sector") or (sector_name or "")
            entry["etf"] = entry.get("etf") if entry.get("etf") else None
            entry["first_seen_utc"] = entry.get("first_seen_utc") or now
            entry["last_seen_utc"] = now
            entry["seen_count"] = int(entry.get("seen_count", 0) or 0) + 1
        refs[key] = entry
        data["updated_utc"] = now
        with open(INDUSTRY_ETF_MAP_FILENAME, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
    return data


def resolve_sector_etf(sectorKey, sector_map=None):
    key = (sectorKey or "").strip()
    mapping = sector_map if isinstance(sector_map, dict) else load_sector_etf_map()
    if not key:
        return "SPY"
    candidates = [
        key,
        key.lower(),
        key.replace("_", "-").lower(),
        slugify_key(key),
    ]
    for candidate in candidates:
        if candidate and candidate in mapping:
            return mapping[candidate]
    return "SPY"


def resolve_industry_ref_etf(industryKey, sectorKey):
    if industryKey:
        data = _load_industry_etf_map_file()
        entry = data.get("yahoo_industryKey_to_ref", {}).get(industryKey, {})
        etf = (entry.get("etf") or "").strip().upper() if isinstance(entry, dict) else ""
        if etf:
            return etf
    return resolve_sector_etf(sectorKey)

##########################################
# Logging Filter
##########################################
class HistoricalDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Code=2176" in msg:
            return False
        if ("REQUEST reqHistoricalData" in msg or
            "SENDING reqHistoricalData" in msg or
            "REQUEST startApi" in msg or
            "SENDING startApi" in msg or
            "HistoricalDataEnd" in msg or
            "Building ATR cache for new symbols" in msg or
            "Calculated shares" in msg or
            "Not enough" in msg):
            return False
        return True

##########################################
# Utility Functions
##########################################
def wait_for_candle_close():
    now = datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    sec_since_5 = (now.minute % 5) * 60 + now.second
    sec_to_go = 300 - sec_since_5
    logging.info(f"Waiting for candle to close: {sec_to_go} seconds remaining.")
    while sec_to_go > 0:
        time.sleep(1)
        sec_to_go -= 1
    logging.info("Candle has closed.")

def read_tickers(file_path):
    if not os.path.exists(file_path):
        logging.warning(f"{file_path} does not exist.")
        return []
    with open(file_path, "r") as f:
        tickers = [line.strip().upper() for line in f 
                  if line.strip() and "Symbols from TC2000" not in line]
    logging.debug(f"Loaded tickers from {file_path}: {tickers}")
    return tickers


def _parse_iso_date_safe(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None

def reset_log_files():
    TRADING_BOT_LOG_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    Path(TRADING_BOT_LOG_FILENAME).touch(exist_ok=True)

    try:
        if os.path.exists(BOUNCE_LOG_FILENAME):
            os.remove(BOUNCE_LOG_FILENAME)
            print(f"Previous session bounce list deleted: {BOUNCE_LOG_FILENAME}")
        with open(BOUNCE_LOG_FILENAME, "w", encoding="utf-8"):
            pass
        print(f"Created fresh bounce list: {BOUNCE_LOG_FILENAME}")
    except Exception as e:
        print(f"Error resetting bounce list {BOUNCE_LOG_FILENAME}: {e}")


def configure_app_logging():
    if getattr(configure_app_logging, "_configured", False):
        return

    TRADING_BOT_LOG_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(APP_LOG_FORMAT))

    file_handler = RotatingFileHandler(
        TRADING_BOT_LOG_FILENAME,
        maxBytes=1_000_000,
        backupCount=APP_LOG_BACKUP_COUNT,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(APP_LOG_FORMAT))

    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler], force=True)
    logger = logging.getLogger()
    logger.addFilter(HistoricalDataFilter())
    logging.getLogger("ibapi").addFilter(HistoricalDataFilter())
    logging.getLogger("ibapi.client").addFilter(HistoricalDataFilter())
    configure_app_logging._configured = True


@dataclass(frozen=True)
class IbBar:
    dt: datetime
    open: float
    high: float
    low: float
    close: float


def _parse_ib_bar_datetime(value):
    text = str(value).strip()
    for fmt in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _bars_to_ib(bars):
    ib_bars = []
    for bar in bars:
        dt = _parse_ib_bar_datetime(bar.get("time"))
        if dt is None:
            continue
        ib_bars.append(
            IbBar(
                dt=dt,
                open=float(bar.get("open", 0.0)),
                high=float(bar.get("high", 0.0)),
                low=float(bar.get("low", 0.0)),
                close=float(bar.get("close", 0.0)),
            )
        )
    return ib_bars


def _dedupe_bars(bars):
    by_dt = {}
    for bar in bars:
        by_dt[bar.dt] = bar
    return sorted(by_dt.values(), key=lambda b: b.dt)


def _align_bars_with_map(symbol_bars, spy_by_dt):
    if not symbol_bars or not spy_by_dt:
        return [], []
    sym_by_dt = {bar.dt: bar for bar in symbol_bars}
    common = sorted(sym_by_dt.keys() & spy_by_dt.keys())
    if not common:
        return [], []
    return [sym_by_dt[dt] for dt in common], [spy_by_dt[dt] for dt in common]


def _aggregate_bars_timeframe(bars, timeframe_minutes):
    if timeframe_minutes <= 5:
        return list(bars)
    expected = max(1, timeframe_minutes // 5)
    buckets = {}
    order = []
    counts = {}
    for bar in bars:
        dt = bar.dt
        market_open = dt.replace(hour=6, minute=30, second=0, microsecond=0)
        minutes_since = int((dt - market_open).total_seconds() // 60)
        if minutes_since < 0:
            bucket_minute = (dt.minute // timeframe_minutes) * timeframe_minutes
            bucket_start = dt.replace(minute=bucket_minute, second=0, microsecond=0)
        else:
            bucket_index = minutes_since // timeframe_minutes
            bucket_start = market_open + timedelta(minutes=bucket_index * timeframe_minutes)
        if bucket_start not in buckets:
            buckets[bucket_start] = IbBar(
                dt=bucket_start,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
            )
            counts[bucket_start] = 1
            order.append(bucket_start)
        else:
            agg = buckets[bucket_start]
            buckets[bucket_start] = IbBar(
                dt=agg.dt,
                open=agg.open,
                high=max(agg.high, bar.high),
                low=min(agg.low, bar.low),
                close=bar.close,
            )
            counts[bucket_start] += 1

    if not order:
        return []

    result = []
    last_bucket = order[-1]
    for bucket in order:
        if bucket == last_bucket and counts.get(bucket, 0) < expected:
            continue
        result.append(buckets[bucket])
    return result


def _wilder_atr_last(bars, length):
    if len(bars) < length + 1:
        return None
    true_ranges = []
    for idx in range(1, len(bars)):
        high = bars[idx].high
        low = bars[idx].low
        prev_close = bars[idx - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if len(true_ranges) < length:
        return None
    atr = sum(true_ranges[:length]) / float(length)
    for tr in true_ranges[length:]:
        atr = ((atr * (length - 1)) + tr) / float(length)
    return atr if atr > 0 else None


def real_relative_strength(symbol_bars, spy_bars, length=RRS_LENGTH):
    if not symbol_bars or not spy_bars:
        return None, None
    min_bars = length + 2
    if len(symbol_bars) < min_bars or len(spy_bars) < min_bars:
        return None, None
    sym_move = symbol_bars[-1].close - symbol_bars[-1 - length].close
    spy_move = spy_bars[-1].close - spy_bars[-1 - length].close
    sym_atr = _wilder_atr_last(symbol_bars[:-1], length)
    spy_atr = _wilder_atr_last(spy_bars[:-1], length)
    if sym_atr is None or spy_atr is None or sym_atr == 0 or spy_atr == 0:
        return None, None
    power_index = spy_move / spy_atr
    rrs = (sym_move - (power_index * sym_atr)) / sym_atr
    return rrs, power_index


##########################################
# Request Queue Class
##########################################
class RequestQueue:
    def __init__(self):
        self.queue = []
        self.active_requests = 0
        self.lock = threading.Lock()

    def add_request(self, func, *args):
        self.queue.append((func, args))
        self.process_queue()

    def process_queue(self):
        with self.lock:
            while self.queue and self.active_requests < MAX_CONCURRENT_REQUESTS:
                func, args = self.queue.pop(0)
                self.active_requests += 1
                threading.Thread(target=self._execute_request, args=(func, args)).start()

    def _execute_request(self, func, args):
        try:
            func(*args)
            time.sleep(REQUEST_DELAY)
        finally:
            with self.lock:
                self.active_requests -= 1
                self.process_queue()

##########################################
# BounceBot Class with GUI callback
##########################################
class BounceBot(EWrapper, EClient):
    def __init__(self, gui_callback=None):
        EClient.__init__(self, self)
        self.connection_status = False
        self.reqIdCount = 1000
        self.data_lock = threading.Lock()
        self.reqId_lock = threading.Lock()
        self.connect_lock = threading.Lock()
        self.ib_host = None
        self.ib_port = None
        self.ib_client_id_base = None
        self.ib_client_id = None
        self.ib_client_id_offset = 0
        self.client_id_conflict = False
        self.client_id_namespace = max(1, os.getpid() % 1000) * 1000
        self.api_thread = None

        self.data = {}
        self.data_ready_events = {}
        self.reqid_to_symbol = {}
        self.invalid_security_symbols = set()

        self.longs = read_tickers(LONGS_FILENAME)
        self.shorts = read_tickers(SHORTS_FILENAME)
        self.atr_cache = {}
        self.symbol_metrics = {}  # Store precomputed VWAP and level metrics

        self.alerted_symbols = set()
        self.bounce_candidates = {}  # Track candidate bounces
        self.request_queue = RequestQueue()
        self.gui_callback = gui_callback  # Callback to update the GUI
        
        # Add this to track which symbols we've already warned about
        self.warned_symbols = set()

        # RRS settings (thread-safe for GUI updates)
        self.rrs_lock = threading.Lock()
        self.rrs_threshold = RRS_DEFAULT_THRESHOLD
        self.rrs_length = RRS_LENGTH
        self.rrs_timeframe_key = "5m"
        self.rrs_bar_size = RRS_TIMEFRAMES[self.rrs_timeframe_key]["bar_size"]
        self.rrs_duration = RRS_TIMEFRAMES[self.rrs_timeframe_key]["duration"]

        # Cache latest 5-minute bars per symbol for reuse (RRS)
        self.latest_bars = {}
        self.latest_scan_extremes = {}
        self.latest_group_extremes = {}
        self.market_environment = "bullish_strong"
        self.market_environment_lock = threading.Lock()
        self.latest_rrs_payload = None

        self.bounce_type_toggles = dict(BOUNCE_TYPE_DEFAULTS)
        self.scanning_enabled = True
        self.scanning_lock = threading.Lock()

        self.master_avwap_events = {}
        self.master_avwap_last_scan_date = None
        self.emitted_master_avwap_events = set()
        self.master_avwap_focus_map = {}
        self.emitted_master_avwap_focus_alerts = set()

        self.sector_etf_map = load_sector_etf_map()
        self.industry_map_data = _load_industry_etf_map_file()
        self.symbol_classification_cache = {}
        self._load_symbol_classification_cache()


    def getReqId(self):
        with self.reqId_lock:
            reqId = self.reqIdCount
            self.reqIdCount += 1
            return reqId

    def create_stock_contract(self, symbol):
        logging.debug(f"Creating contract for symbol: {symbol}")
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def set_connection_info(self, host, port, client_id):
        self.ib_host = host
        self.ib_port = port
        self.ib_client_id_base = int(client_id) if client_id is not None else 125
        if self.ib_client_id is None:
            self.ib_client_id = self._allocate_client_id()

    def _allocate_client_id(self):
        base = int(self.ib_client_id_base or self.ib_client_id or 125)
        client_id = (base * 10000) + self.client_id_namespace + self.ib_client_id_offset
        self.ib_client_id_offset += 1
        self.ib_client_id = client_id
        self.client_id_conflict = False
        return client_id

    def _ensure_api_thread(self):
        if self.api_thread is None or not self.api_thread.is_alive():
            self.api_thread = threading.Thread(target=self.run, daemon=True)
            self.api_thread.start()

    def ensure_connected(self, timeout=10):
        if self.connection_status:
            return True
        if self.ib_host is None or self.ib_port is None or self.ib_client_id is None:
            logging.error("IB connection info not set.")
            return False
        deadline = time.time() + timeout
        with self.connect_lock:
            if self.connection_status:
                return True
            while time.time() < deadline:
                if self.client_id_conflict or self.ib_client_id is None:
                    old_client_id = self.ib_client_id
                    new_client_id = self._allocate_client_id()
                    if old_client_id is not None:
                        logging.warning(
                            f"IB client id {old_client_id} is already in use; retrying with {new_client_id}."
                        )
                try:
                    try:
                        self.disconnect()
                    except Exception:
                        pass
                    current_client_id = self.ib_client_id
                    self.connect(self.ib_host, self.ib_port, clientId=current_client_id)
                    self._ensure_api_thread()
                    attempt_deadline = min(deadline, time.time() + 2.5)
                    while not self.connection_status and time.time() < attempt_deadline:
                        if self.client_id_conflict:
                            break
                        time.sleep(0.2)
                    if self.connection_status:
                        logging.info(f"Reconnected to IB with clientId={self.ib_client_id}.")
                        return True
                    if self.client_id_conflict:
                        continue
                except Exception as e:
                    logging.exception(f"Reconnect error: {e}")
                    return False
            logging.error("Failed to reconnect to IB within timeout.")
            return False

    def set_bounce_type_enabled(self, bounce_type, enabled):
        if bounce_type in self.bounce_type_toggles:
            self.bounce_type_toggles[bounce_type] = bool(enabled)

    def is_bounce_type_enabled(self, bounce_type):
        return self.bounce_type_toggles.get(bounce_type, False)

    def set_scanning_enabled(self, enabled):
        with self.scanning_lock:
            self.scanning_enabled = bool(enabled)
        state = "enabled" if self.scanning_enabled else "paused"
        logging.info(f"Scanning {state}.")
        if self.gui_callback:
            self.gui_callback(f"Scanning {state}.", "blue")

    def is_scanning_enabled(self):
        with self.scanning_lock:
            return self.scanning_enabled

    def _normalize_master_avwap_event_row(self, row):
        signal_type = (row.get("signal_type") or "").strip().upper()
        if not signal_type:
            return None
        if "BETWEEN" in signal_type:
            return None
        if not (signal_type.startswith("CROSS") or signal_type.startswith("BOUNCE")):
            return None

        symbol = (row.get("symbol") or "").strip().upper()
        if not symbol:
            return None

        trade_date = _parse_iso_date_safe(row.get("trade_date"))
        if trade_date is None:
            return None

        level = signal_type
        for prefix in ("CROSS_UP_", "CROSS_DOWN_", "BOUNCE_"):
            if signal_type.startswith(prefix) and len(signal_type) > len(prefix):
                level = signal_type[len(prefix):]
                break

        priority_bucket = (row.get("priority_bucket") or "").strip().lower()
        favorite_zone = (row.get("favorite_zone") or "").strip()
        favorite_signals = [
            value.strip().upper()
            for value in str(row.get("favorite_signals") or "").split(";")
            if value.strip()
        ]
        is_favorite_setup = str(row.get("is_favorite_setup") or "").strip().lower() in ("1", "true", "yes")
        is_near_favorite_zone = str(row.get("is_near_favorite_zone") or "").strip().lower() in ("1", "true", "yes")

        return {
            "symbol": symbol,
            "trade_date": trade_date,
            "signal_type": signal_type,
            "anchor_type": (row.get("anchor_type") or "").strip().upper() or "UNKNOWN",
            "anchor_date": (row.get("anchor_date") or "").strip(),
            "side": (row.get("side") or "").strip().upper(),
            "level": level,
            "priority_bucket": priority_bucket,
            "favorite_zone": favorite_zone,
            "favorite_signals": favorite_signals,
            "is_favorite_setup": is_favorite_setup,
            "is_near_favorite_zone": is_near_favorite_zone,
        }

    def load_master_avwap_events_today(self):
        """
        Load today's NEW cross/bounce events from master_avwap signal output.
        Keeps parsing generic so new signal/level names continue to work.
        """
        if not MASTER_AVWAP_SIGNALS_FILENAME.exists():
            self.master_avwap_events = {}
            self.master_avwap_last_scan_date = datetime.now().date()
            return

        try:
            with open(MASTER_AVWAP_SIGNALS_FILENAME, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        except Exception as exc:
            logging.warning(f"Failed reading master AVWAP signals file: {exc}")
            self.master_avwap_events = {}
            return

        today = datetime.now().date()
        events_map = {}
        for row in rows:
            normalized = self._normalize_master_avwap_event_row(row)
            if not normalized:
                continue
            if normalized["trade_date"] != today:
                continue

            symbol = normalized["symbol"]
            events_map.setdefault(symbol, []).append(normalized)

        self.master_avwap_events = events_map
        self.master_avwap_last_scan_date = today

    def _build_master_avwap_active_level_map(self):
        active_levels = {}
        for symbol, events in self.master_avwap_events.items():
            levels = set()
            for event in events:
                signal = event.get("signal_type", "")
                if signal.startswith("CROSS") or signal.startswith("BOUNCE"):
                    levels.add(event.get("level") or signal)
            if levels:
                active_levels[symbol] = sorted(levels)
        return active_levels
    def find_active_master_avwap_bounces(self):
        self.load_master_avwap_events_today()
        active = {}
        for symbol, events in self.master_avwap_events.items():
            bounce_levels = sorted(
                {
                    event.get("level") or event.get("signal_type")
                    for event in events
                    if str(event.get("signal_type", "")).startswith("BOUNCE")
                }
            )
            if bounce_levels:
                active[symbol] = bounce_levels
        return active

    def _master_avwap_event_key(self, event):
        return (
            event.get("symbol"),
            event.get("trade_date"),
            event.get("signal_type"),
            event.get("anchor_type"),
            event.get("anchor_date"),
        )

    def load_master_avwap_focus(self):
        if not MASTER_AVWAP_FOCUS_FILENAME.exists():
            self.master_avwap_focus_map = {}
            return

        try:
            with open(MASTER_AVWAP_FOCUS_FILENAME, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            logging.warning(f"Failed reading master AVWAP focus file: {exc}")
            self.master_avwap_focus_map = {}
            return

        raw_symbols = payload.get("symbols", {}) if isinstance(payload, dict) else {}
        focus_map = {}
        if isinstance(raw_symbols, dict):
            for raw_symbol, raw_entry in raw_symbols.items():
                entry = raw_entry if isinstance(raw_entry, dict) else {}
                symbol = (entry.get("symbol") or raw_symbol or "").strip().upper()
                if not symbol:
                    continue
                focus_map[symbol] = {
                    "symbol": symbol,
                    "side": (entry.get("side") or "").strip().upper(),
                    "priority_bucket": (entry.get("priority_bucket") or "").strip().lower(),
                    "priority_rank": entry.get("priority_rank"),
                    "priority_score": entry.get("priority_score"),
                    "favorite_zone": (entry.get("favorite_zone") or "").strip(),
                    "favorite_signals": [
                        str(value).strip().upper()
                        for value in (entry.get("favorite_signals") or [])
                        if str(value).strip()
                    ],
                    "favorite_context_signals": [
                        str(value).strip().upper()
                        for value in (entry.get("favorite_context_signals") or [])
                        if str(value).strip()
                    ],
                    "breakout_5d": bool(entry.get("breakout_5d")),
                    "retest_followthrough": bool(entry.get("retest_followthrough")),
                }

        self.master_avwap_focus_map = focus_map

    def _describe_master_avwap_focus(self, focus_entry):
        bucket = (focus_entry or {}).get("priority_bucket", "")
        if bucket == "favorite_setup":
            return "best current favorite setup"
        if bucket == "near_favorite_zone":
            return "near favorite zone"
        return "master avwap focus"

    def _emit_master_avwap_focus_bounce_alert(self, symbol, direction, levels_list):
        if not self.gui_callback:
            return
        focus_entry = self.master_avwap_focus_map.get(symbol)
        if not focus_entry:
            return
        focus_side = focus_entry.get("side")
        if direction == "long" and focus_side != "LONG":
            return
        if direction == "short" and focus_side != "SHORT":
            return

        normalized_levels = tuple(sorted(str(level) for level in (levels_list or [])))
        alert_key = (datetime.now().date().isoformat(), "bounce", symbol, direction, normalized_levels)
        if alert_key in self.emitted_master_avwap_focus_alerts:
            return
        self.emitted_master_avwap_focus_alerts.add(alert_key)

        reason = self._describe_master_avwap_focus(focus_entry)
        level_text = ", ".join(normalized_levels) if normalized_levels else "level"
        message = f"MASTER_AVWAP_FOCUS_BOUNCE: {symbol} ({direction}) {level_text} [{reason}]"
        gui_tag = "master_avwap_focus_long" if direction == "long" else "master_avwap_focus_short"
        self.gui_callback(message, gui_tag)
        self.log_symbol(symbol, message)

    def _focus_rrs_is_significant(self, entry, threshold):
        rrs_value = entry.get("rrs")
        move_ratio = entry.get("move_ratio")
        excess_move_ratio = entry.get("excess_move_ratio")
        if rrs_value is None or move_ratio is None or excess_move_ratio is None:
            return False
        if not self._is_move_significant(entry):
            return False
        return (
            abs(rrs_value) >= max(MASTER_AVWAP_FOCUS_MIN_ABS_RRS, float(threshold) + 0.75)
            and abs(move_ratio) >= MASTER_AVWAP_FOCUS_MIN_MOVE_RATIO
            and abs(excess_move_ratio) >= MASTER_AVWAP_FOCUS_MIN_EXCESS_MOVE_RATIO
        )

    def _emit_master_avwap_focus_rrs_alerts(self, symbol_context, threshold, timeframe_key):
        if not EMIT_MASTER_AVWAP_FOCUS_RRS_ALERTS:
            return
        if not self.gui_callback or not self.master_avwap_focus_map:
            return

        timeframe_label = RRS_TIMEFRAMES.get(timeframe_key, {}).get("label", timeframe_key)
        today_iso = datetime.now().date().isoformat()
        for entry in symbol_context:
            symbol = entry.get("symbol")
            if not symbol:
                continue
            focus_entry = self.master_avwap_focus_map.get(symbol)
            if not focus_entry:
                continue

            focus_side = focus_entry.get("side")
            signal = entry.get("signal")
            if focus_side == "LONG" and signal != "RS":
                continue
            if focus_side == "SHORT" and signal != "RW":
                continue
            if not self._focus_rrs_is_significant(entry, threshold):
                continue

            alert_key = (today_iso, timeframe_key, symbol, signal)
            if alert_key in self.emitted_master_avwap_focus_alerts:
                continue
            self.emitted_master_avwap_focus_alerts.add(alert_key)

            reason = self._describe_master_avwap_focus(focus_entry)
            move_ratio = entry.get("move_ratio")
            excess_move_ratio = entry.get("excess_move_ratio")
            message = (
                f"MASTER_AVWAP_FOCUS_RRS: {symbol} ({focus_side.lower()}) {signal} "
                f"{entry['rrs']:+.2f} ER={excess_move_ratio:+.2f} MR={move_ratio:+.2f} "
                f"[{reason}, {timeframe_label}]"
            )
            gui_tag = "master_avwap_focus_long" if focus_side == "LONG" else "master_avwap_focus_short"
            self.gui_callback(message, gui_tag)
            self.log_symbol(symbol, message)


    def update_watchlists_from_master_avwap(self):
        self.load_master_avwap_events_today()
        active_level_map = self._build_master_avwap_active_level_map()
        event_symbols = set(active_level_map)
        if not event_symbols:
            logging.info("Master AVWAP: no new cross/bounce events for today.")
            return

        current_longs = set(self.longs)
        current_shorts = set(self.shorts)
        monitored = current_longs | current_shorts
        matched_symbols = sorted(event_symbols & monitored)
        if not matched_symbols:
            logging.info(
                "Master AVWAP: today's event symbols do not intersect bounce bot watchlists."
            )
            return

        for symbol in matched_symbols:
            levels = active_level_map.get(symbol, [])
            side = "LONG" if symbol in current_longs else "SHORT"
            symbol_events = self.master_avwap_events.get(symbol, [])
            newly_emitted = 0
            for event in symbol_events:
                event_key = self._master_avwap_event_key(event)
                if event_key in self.emitted_master_avwap_events:
                    continue
                self.emitted_master_avwap_events.add(event_key)
                newly_emitted += 1
                signal_type = event.get("signal_type", "")
                msg = f"MASTER_AVWAP_EVENT: {symbol} ({side}) {signal_type}"
                self.log_symbol(symbol, msg)

            if newly_emitted == 0:
                continue

            summary_msg = (
                f"MASTER_AVWAP_ACTIVE_EVENT: {symbol} ({side}) levels={levels} new_events={newly_emitted}"
            )
            logging.info(summary_msg)

    def set_rrs_threshold(self, value):
        with self.rrs_lock:
            self.rrs_threshold = float(value)

    def set_rrs_timeframe(self, key):
        if key not in RRS_TIMEFRAMES:
            return
        with self.rrs_lock:
            self.rrs_timeframe_key = key
            self.rrs_bar_size = RRS_TIMEFRAMES[key]["bar_size"]
            self.rrs_duration = RRS_TIMEFRAMES[key]["duration"]

    def get_rrs_settings(self):
        with self.rrs_lock:
            return (
                self.rrs_threshold,
                self.rrs_bar_size,
                self.rrs_duration,
                self.rrs_length,
                self.rrs_timeframe_key,
            )

    def set_market_environment(self, env_key):
        if env_key not in MARKET_ENVIRONMENTS:
            return
        with self.market_environment_lock:
            self.market_environment = env_key
        status_msg = f"Market environment set to {MARKET_ENVIRONMENTS[env_key]['label']}"
        self._refresh_rrs_gui(status_msg=status_msg)

    def get_market_environment(self):
        with self.market_environment_lock:
            return self.market_environment

    def _refresh_rrs_gui(self, status_msg=None):
        if not self.latest_rrs_payload or not self.gui_callback:
            return
        snapshot = self._decorate_snapshot(self.latest_rrs_payload)
        self.gui_callback(snapshot, "rrs_snapshot")
        if status_msg:
            self.gui_callback(status_msg, "rrs_status")

    def _decorate_snapshot(self, base_snapshot):
        decorated = dict(base_snapshot)
        env_key = self.get_market_environment()
        decorated["market_environment"] = env_key
        decorated["market_environment_label"] = MARKET_ENVIRONMENTS.get(env_key, {}).get("label", env_key)
        decorated["environment_highlights"] = self._build_environment_highlights(
            base_snapshot.get("symbol_context", []),
            base_snapshot.get("spy_move_ratio"),
            base_snapshot.get("environment_scan"),
        )
        return decorated

    def _build_environment_highlights(self, symbol_context, spy_ratio, environment_scan=None):
        if environment_scan is not None:
            sections = self._build_intraday_environment_sections(environment_scan)
            if sections:
                return sections
            return [
                {
                    "title": "Environment focus",
                    "rows": [{"text": "No intraday RS/RW candidates matched the current SPY context windows.", "tag": "rrs_hdr"}],
                    "tag": "rrs_hdr",
                }
            ]

        scenario = self.get_market_environment()
        strong_entries = sorted(
            [entry for entry in symbol_context if entry.get("signal") == "RS"],
            key=lambda row: (-(row.get("excess_move_ratio") or 0.0), -row["rrs"]),
        )
        weak_entries = sorted(
            [entry for entry in symbol_context if entry.get("signal") == "RW"],
            key=lambda row: ((row.get("excess_move_ratio") or 0.0), row["rrs"]),
        )
        strong_long_entries = [entry for entry in strong_entries if entry.get("watchlist_bias") == "long"]
        weak_short_entries = [entry for entry in weak_entries if entry.get("watchlist_bias") == "short"]
        sections = []

        def append_section(title, entries, tag):
            section = self._build_section(title, entries, tag)
            if section:
                sections.append(section)

        if scenario == "bearish_strong":
            append_section("Weak Stocks (Bearish Strong focus)", weak_entries, "rrs_rw")
        elif scenario == "bearish_weak":
            append_section("Weak Stocks (Bearish Weak focus)", weak_entries, "rrs_rw")
            append_section("Strong Stocks (Bearish Weak focus)", strong_long_entries, "rrs_rs")
        elif scenario == "bullish_strong":
            append_section("Strong Stocks (Bullish Strong focus)", strong_entries, "rrs_rs")
        elif scenario == "bullish_weak":
            append_section("Strong Stocks (Bullish Weak focus)", strong_entries, "rrs_rs")
            append_section("Weak Stocks (Bullish Weak focus)", weak_short_entries, "rrs_rw")
        else:
            append_section("Strong Stocks", strong_entries, "rrs_rs")
            append_section("Weak Stocks", weak_entries, "rrs_rw")

        spy_section = self._build_spy_section(spy_ratio, weak_entries)
        if spy_section:
            sections.append(spy_section)

        if not sections:
            sections.append(
                {
                    "title": "Environment focus",
                    "rows": [{"text": "No RS/RW candidates available", "tag": "rrs_hdr"}],
                    "tag": "rrs_hdr",
                }
            )
        return sections

    def _build_spy_context_windows(self, spy_bars, length):
        if not spy_bars or len(spy_bars) < length + 2:
            return []

        current_date = spy_bars[-1].dt.date()
        windows = []
        for idx in range(length + 1, len(spy_bars)):
            bar = spy_bars[idx]
            if bar.dt.date() != current_date:
                continue
            move_ratio = self._calc_move_ratio(spy_bars[:idx + 1], length)
            if move_ratio is None:
                continue
            prev_move_ratio = windows[-1]["move_ratio"] if windows else None
            delta = move_ratio - prev_move_ratio if prev_move_ratio is not None else 0.0
            compression = abs(move_ratio) < SPY_COMPRESSION_THRESHOLD
            rising = move_ratio >= SPY_UP_THRESHOLD or delta >= SPY_PULLBACK_DELTA_THRESHOLD
            falling = move_ratio <= -SPY_UP_THRESHOLD or delta <= -SPY_PULLBACK_DELTA_THRESHOLD
            windows.append(
                {
                    "dt": bar.dt,
                    "move_ratio": move_ratio,
                    "delta": delta,
                    "compression": compression,
                    "spy_weak": falling,
                    "spy_strong": rising,
                    "long_eval": compression or falling,
                    "short_eval": compression or rising,
                }
            )
        return windows

    def _summarize_environment_scan(self, intraday_profiles, spy_windows, threshold):
        if not intraday_profiles or not spy_windows:
            return None

        window_map = {item["dt"]: item for item in spy_windows}
        long_candidates = []
        short_candidates = []
        weak_spy_window_count = sum(1 for item in spy_windows if item.get("spy_weak"))
        strong_spy_window_count = sum(1 for item in spy_windows if item.get("spy_strong"))
        compression_window_count = sum(1 for item in spy_windows if item.get("compression"))

        def summarize_bucket(profile, direction, flag_key):
            samples = []
            hits = []
            for item in profile:
                window = window_map.get(item.get("dt"))
                if not window or not window.get(flag_key):
                    continue
                if item.get("rrs") is None:
                    continue
                samples.append(item)
                if direction == "long":
                    if item["rrs"] >= threshold:
                        hits.append(item)
                else:
                    if item["rrs"] <= -threshold:
                        hits.append(item)

            sample_rrs = [item["rrs"] for item in samples if item.get("rrs") is not None]
            sample_excess = [item["excess_move_ratio"] for item in samples if item.get("excess_move_ratio") is not None]
            avg_rrs = (sum(sample_rrs) / len(sample_rrs)) if sample_rrs else None
            best_rrs = None
            if sample_rrs:
                best_rrs = max(sample_rrs) if direction == "long" else min(sample_rrs)
            avg_excess = (sum(sample_excess) / len(sample_excess)) if sample_excess else None
            return {
                "windows": len(samples),
                "hits": len(hits),
                "hit_rate": (len(hits) / len(samples)) if samples else 0.0,
                "avg_rrs": avg_rrs,
                "best_rrs": best_rrs,
                "avg_excess": avg_excess,
            }

        def finalize_candidate(symbol, direction, context_summary, compression_summary):
            if context_summary["hits"] <= 0 and compression_summary["hits"] <= 0:
                return None
            score = (
                context_summary["hits"] * 160.0
                + context_summary["hit_rate"] * 110.0
                + (abs(context_summary["avg_rrs"]) * 18.0 if context_summary["avg_rrs"] is not None else 0.0)
                + (abs(context_summary["best_rrs"]) * 8.0 if context_summary["best_rrs"] is not None else 0.0)
                + (abs(context_summary["avg_excess"]) * 10.0 if context_summary["avg_excess"] is not None else 0.0)
                + compression_summary["hits"] * 55.0
                + compression_summary["hit_rate"] * 35.0
                + (abs(compression_summary["avg_rrs"]) * 6.0 if compression_summary["avg_rrs"] is not None else 0.0)
                + (abs(compression_summary["best_rrs"]) * 3.0 if compression_summary["best_rrs"] is not None else 0.0)
                + (abs(compression_summary["avg_excess"]) * 4.0 if compression_summary["avg_excess"] is not None else 0.0)
            )
            return {
                "symbol": symbol,
                "direction": direction,
                "context_hits": context_summary["hits"],
                "context_windows": context_summary["windows"],
                "context_hit_rate": context_summary["hit_rate"],
                "context_avg_rrs": context_summary["avg_rrs"],
                "context_best_rrs": context_summary["best_rrs"],
                "context_avg_excess": context_summary["avg_excess"],
                "compression_hits": compression_summary["hits"],
                "compression_windows": compression_summary["windows"],
                "compression_hit_rate": compression_summary["hit_rate"],
                "compression_avg_rrs": compression_summary["avg_rrs"],
                "compression_best_rrs": compression_summary["best_rrs"],
                "compression_avg_excess": compression_summary["avg_excess"],
                "score": score,
            }

        for symbol, profile in intraday_profiles.items():
            if not profile:
                continue

            if symbol in self.longs:
                context_summary = summarize_bucket(profile, "long", "spy_weak")
                compression_summary = summarize_bucket(profile, "long", "compression")
                candidate = finalize_candidate(symbol, "long", context_summary, compression_summary)
                if candidate:
                    long_candidates.append(candidate)

            if symbol in self.shorts:
                context_summary = summarize_bucket(profile, "short", "spy_strong")
                compression_summary = summarize_bucket(profile, "short", "compression")
                candidate = finalize_candidate(symbol, "short", context_summary, compression_summary)
                if candidate:
                    short_candidates.append(candidate)

        long_candidates.sort(
            key=lambda row: (
                -row["score"],
                -row["context_hits"],
                -(row["context_avg_rrs"] if row["context_avg_rrs"] is not None else float("-inf")),
                row["symbol"],
            )
        )
        short_candidates.sort(
            key=lambda row: (
                -row["score"],
                -row["context_hits"],
                (row["context_avg_rrs"] if row["context_avg_rrs"] is not None else float("inf")),
                row["symbol"],
            )
        )
        return {
            "long_candidates": long_candidates,
            "short_candidates": short_candidates,
            "weak_spy_window_count": weak_spy_window_count,
            "strong_spy_window_count": strong_spy_window_count,
            "compression_window_count": compression_window_count,
            "spy_windows": len(spy_windows),
        }

    def _build_intraday_environment_sections(self, environment_scan):
        scenario = self.get_market_environment()
        long_candidates = environment_scan.get("long_candidates", [])
        short_candidates = environment_scan.get("short_candidates", [])
        weak_spy_windows = int(environment_scan.get("weak_spy_window_count", 0) or 0)
        strong_spy_windows = int(environment_scan.get("strong_spy_window_count", 0) or 0)
        compression_windows = int(environment_scan.get("compression_window_count", 0) or 0)
        sections = []

        def candidate_sort_key(entry):
            return (
                -entry.get("score", 0.0),
                -entry.get("context_hits", 0),
                entry.get("symbol", ""),
            )

        long_context_candidates = sorted(
            [entry for entry in long_candidates if entry.get("context_hits", 0) > 0],
            key=candidate_sort_key,
        )
        short_context_candidates = sorted(
            [entry for entry in short_candidates if entry.get("context_hits", 0) > 0],
            key=candidate_sort_key,
        )
        long_compression_candidates = sorted(
            [entry for entry in long_candidates if entry.get("compression_hits", 0) > 0],
            key=candidate_sort_key,
        )
        short_compression_candidates = sorted(
            [entry for entry in short_candidates if entry.get("compression_hits", 0) > 0],
            key=candidate_sort_key,
        )

        def add_candidate_section(title, candidates, tag, focus_mode):
            if not candidates:
                return
            rows = []
            for entry in candidates[:ENVIRONMENT_SCAN_LIMIT]:
                rows.append({
                    "text": self._format_environment_scan_entry(entry, focus_mode=focus_mode),
                    "tag": tag,
                    "symbol": entry.get("symbol"),
                })
            sections.append({"title": title, "rows": rows, "tag": tag})

        prefer_shorts_first = scenario.startswith("bearish")
        primary_sections = []

        if weak_spy_windows > 0 and long_context_candidates:
            primary_sections.append((
                f"Strong Longs During SPY Weakness ({weak_spy_windows} weak windows)",
                long_context_candidates,
                "rrs_rs",
                "context",
            ))
        if strong_spy_windows > 0 and short_context_candidates:
            primary_sections.append((
                f"Weak Shorts During SPY Strength ({strong_spy_windows} strong windows)",
                short_context_candidates,
                "rrs_rw",
                "context",
            ))

        if prefer_shorts_first:
            primary_sections.sort(key=lambda item: 0 if "Shorts" in item[0] else 1)

        for title, candidates, tag, focus_mode in primary_sections:
            add_candidate_section(title, candidates, tag, focus_mode)

        if not sections:
            if compression_windows > 0 and long_compression_candidates:
                add_candidate_section(
                    f"Strong Longs During SPY Compression ({compression_windows} compression windows)",
                    long_compression_candidates,
                    "rrs_rs",
                    "compression",
                )
            if compression_windows > 0 and short_compression_candidates:
                add_candidate_section(
                    f"Weak Shorts During SPY Compression ({compression_windows} compression windows)",
                    short_compression_candidates,
                    "rrs_rw",
                    "compression",
                )

        if sections:
            return sections
        return None

    def _format_environment_scan_entry(self, entry, focus_mode="context"):
        direction = entry.get("direction")
        if focus_mode == "compression":
            focus_label = "comp"
            focus_hits = entry.get("compression_hits", 0)
            focus_windows = entry.get("compression_windows", 0)
            avg_rrs = entry.get("compression_avg_rrs")
            best_rrs = entry.get("compression_best_rrs")
            avg_excess = entry.get("compression_avg_excess")
            secondary_label = "weakSPY" if direction == "long" else "strongSPY"
            secondary_hits = entry.get("context_hits", 0)
            secondary_windows = entry.get("context_windows", 0)
        else:
            focus_label = "weakSPY" if direction == "long" else "strongSPY"
            focus_hits = entry.get("context_hits", 0)
            focus_windows = entry.get("context_windows", 0)
            avg_rrs = entry.get("context_avg_rrs")
            best_rrs = entry.get("context_best_rrs")
            avg_excess = entry.get("context_avg_excess")
            secondary_label = "comp"
            secondary_hits = entry.get("compression_hits", 0)
            secondary_windows = entry.get("compression_windows", 0)

        avg_rrs_text = f"{avg_rrs:+.2f}" if avg_rrs is not None else "n/a"
        best_rrs_text = f"{best_rrs:+.2f}" if best_rrs is not None else "n/a"
        line = (
            f"{entry['symbol']} {focus_label}={focus_hits}/{focus_windows} "
            f"avgRRS={avg_rrs_text} best={best_rrs_text}"
        )
        if avg_excess is not None:
            line += f" avgER={avg_excess:+.2f}"
        if secondary_windows:
            line += f" {secondary_label}={secondary_hits}/{secondary_windows}"
        return line

    def _build_section(self, title, entries, tag):
        if not entries:
            return None
        significant = [entry for entry in entries if self._is_move_significant(entry)]
        source = significant if significant else entries
        rows = []
        for entry in source[:ENVIRONMENT_HIGHLIGHT_LIMIT]:
            rows.append({
                "text": self._format_environment_entry(entry),
                "tag": tag,
                "symbol": entry.get("symbol"),
            })
        if not rows:
            return None
        return {"title": title, "rows": rows, "tag": tag}

    def _build_spy_section(self, spy_ratio, weak_entries):
        if spy_ratio is None or not weak_entries:
            return None
        if spy_ratio >= SPY_UP_THRESHOLD:
            title = "SPY rising / flat -> weak stocks to watch"
        elif abs(spy_ratio) < SPY_COMPRESSION_THRESHOLD:
            title = "SPY compressing -> weak names to monitor"
        else:
            return None
        return self._build_section(title, weak_entries, "rrs_rw")

    def _format_environment_entry(self, entry):
        move_ratio = entry.get("move_ratio")
        excess_move_ratio = entry.get("excess_move_ratio")
        move_text = f"{move_ratio:+.2f}" if move_ratio is not None else "n/a"
        excess_text = f"{excess_move_ratio:+.2f}" if excess_move_ratio is not None else "n/a"
        signal = entry.get("signal", "")
        marker = "UP" if signal == "RS" else "DN"
        return f"{entry['symbol']} {marker} {signal} {entry['rrs']:+.2f} ER={excess_text} MR={move_text}"

    def _is_move_significant(self, entry):
        move_ratio = entry.get("move_ratio")
        excess_move_ratio = entry.get("excess_move_ratio")
        if move_ratio is None or excess_move_ratio is None:
            return False
        return (
            abs(move_ratio) >= MIN_MOVE_RATIO_FOR_SIGNAL
            and abs(excess_move_ratio) >= MIN_EXCESS_MOVE_RATIO_FOR_SIGNAL
        )

    def _calc_move_ratio(self, bars, length):
        if not bars or len(bars) < length + 1:
            return None
        move = bars[-1].close - bars[-(length + 1)].close
        atr = _wilder_atr_last(bars[:-1], length)
        if atr is None or atr == 0:
            return None
        return move / atr

    def _build_intraday_rrs_profile(self, symbol_bars, spy_bars, length=IMPULSE_RRS_PROFILE_LENGTH):
        if not symbol_bars or not spy_bars:
            return []
        aligned_sym, aligned_spy = _align_bars_with_map(symbol_bars, {bar.dt: bar for bar in spy_bars})
        if len(aligned_sym) < length + 2 or len(aligned_spy) < length + 2:
            return []

        profile = []
        for idx in range(length + 1, len(aligned_sym)):
            sym_slice = aligned_sym[:idx + 1]
            spy_slice = aligned_spy[:idx + 1]
            rrs_value, power_index = real_relative_strength(sym_slice, spy_slice, length=length)
            if rrs_value is None:
                continue
            sym_move_ratio = self._calc_move_ratio(sym_slice, length)
            spy_move_ratio = self._calc_move_ratio(spy_slice, length)
            excess_move_ratio = None
            if sym_move_ratio is not None and spy_move_ratio is not None:
                excess_move_ratio = sym_move_ratio - spy_move_ratio
            profile.append(
                {
                    "dt": sym_slice[-1].dt,
                    "rrs": rrs_value,
                    "power_index": power_index,
                    "move_ratio": sym_move_ratio,
                    "spy_move_ratio": spy_move_ratio,
                    "excess_move_ratio": excess_move_ratio,
                }
            )
        return profile

    def _impulse_regime_transition_ok(self, symbol, direction, profile):
        if len(profile) < (IMPULSE_RRS_RECENT_PHASE_BARS + 2):
            logging.debug(f"{symbol}: Insufficient RS/RW profile points for impulse regime filter.")
            return False

        env = self.get_market_environment()
        favored_env = (
            (direction == "long" and env.startswith("bullish"))
            or (direction == "short" and env.startswith("bearish"))
        )
        weak_env = env.endswith("weak")

        pre_phase = profile[:-IMPULSE_RRS_RECENT_PHASE_BARS]
        recent_phase = profile[-IMPULSE_RRS_RECENT_PHASE_BARS:]
        if not pre_phase or not recent_phase:
            return False

        trend_ratio_req = (
            IMPULSE_RRS_FAVORED_TREND_RATIO if favored_env else IMPULSE_RRS_UNFAVORED_TREND_RATIO
        )
        excess_ratio_req = (
            IMPULSE_RRS_FAVORED_EXCESS_RATIO if (favored_env and weak_env) else 0.0
        )
        if not favored_env:
            excess_ratio_req = IMPULSE_RRS_UNFAVORED_EXCESS_RATIO

        if direction == "long":
            best_trend_ratio = max((item.get("move_ratio") for item in pre_phase if item.get("move_ratio") is not None), default=None)
            best_excess_ratio = max((item.get("excess_move_ratio") for item in pre_phase if item.get("excess_move_ratio") is not None), default=None)
            counter_rrs_count = sum(
                1 for item in recent_phase
                if (item.get("rrs") is not None and item.get("rrs") <= -IMPULSE_RRS_COUNTER_RRS)
            )
            counter_move_count = sum(
                1 for item in recent_phase
                if (item.get("move_ratio") is not None and item.get("move_ratio") <= -IMPULSE_RRS_COUNTER_MOVE_RATIO)
            )
            trend_ok = best_trend_ratio is not None and best_trend_ratio >= trend_ratio_req
            excess_ok = excess_ratio_req <= 0.0 or (best_excess_ratio is not None and best_excess_ratio >= excess_ratio_req)
        else:
            best_trend_ratio = min((item.get("move_ratio") for item in pre_phase if item.get("move_ratio") is not None), default=None)
            best_excess_ratio = min((item.get("excess_move_ratio") for item in pre_phase if item.get("excess_move_ratio") is not None), default=None)
            counter_rrs_count = sum(
                1 for item in recent_phase
                if (item.get("rrs") is not None and item.get("rrs") >= IMPULSE_RRS_COUNTER_RRS)
            )
            counter_move_count = sum(
                1 for item in recent_phase
                if (item.get("move_ratio") is not None and item.get("move_ratio") >= IMPULSE_RRS_COUNTER_MOVE_RATIO)
            )
            trend_ok = best_trend_ratio is not None and best_trend_ratio <= -trend_ratio_req
            excess_ok = excess_ratio_req <= 0.0 or (best_excess_ratio is not None and best_excess_ratio <= -excess_ratio_req)

        counter_ok = (
            counter_rrs_count >= IMPULSE_RRS_COUNTER_BARS_REQUIRED
            and counter_move_count >= IMPULSE_RRS_COUNTER_BARS_REQUIRED
        )

        if trend_ok and excess_ok and counter_ok:
            logging.debug(
                f"{symbol}: Impulse RS/RW regime accepted. Env={env}, Direction={direction}, "
                f"TrendRatio={best_trend_ratio}, ExcessRatio={best_excess_ratio}, "
                f"CounterRRSBars={counter_rrs_count}, CounterMoveBars={counter_move_count}"
            )
            return True

        logging.debug(
            f"{symbol}: Rejecting impulse retest on RS/RW regime filter. Env={env}, Direction={direction}, "
            f"TrendOK={trend_ok}, ExcessOK={excess_ok}, CounterOK={counter_ok}, "
            f"TrendRatio={best_trend_ratio}, ExcessRatio={best_excess_ratio}, "
            f"CounterRRSBars={counter_rrs_count}, CounterMoveBars={counter_move_count}"
        )
        return False

    def _build_intraday_level_series_maps(self, df, current_date):
        if "datetime" in df.columns and "typical_price" in df.columns:
            prepared = df.copy()
        else:
            prepared = self._prepare_vwap_frame(df)
        if prepared.empty:
            return {}

        today_prepared = prepared[prepared["datetime"].dt.date == current_date].copy()
        if today_prepared.empty:
            return {}

        series_maps = {}

        std_work = today_prepared.copy()
        std_work["cum_vol"] = std_work["volume"].cumsum()
        std_work["cum_vol_price"] = (std_work["typical_price"] * std_work["volume"]).cumsum()
        std_work["level_value"] = std_work["cum_vol_price"] / std_work["cum_vol"]
        series_maps["vwap"] = {
            row["datetime"]: float(row["level_value"])
            for _, row in std_work.iterrows()
        }

        previous_sessions = sorted(d for d in prepared["datetime"].dt.date.unique() if d < current_date)
        previous_session_df = (
            prepared[prepared["datetime"].dt.date == previous_sessions[-1]].copy()
            if previous_sessions
            else pd.DataFrame()
        )

        dynamic_source = (
            pd.concat([previous_session_df, today_prepared], ignore_index=True)
            if not previous_session_df.empty
            else today_prepared.copy()
        )
        dynamic_skip_rows = len(previous_session_df)
        dynamic_source["cum_vol"] = dynamic_source["volume"].cumsum()
        dynamic_source["cum_vol_price"] = (dynamic_source["typical_price"] * dynamic_source["volume"]).cumsum()
        dynamic_source["level_value"] = dynamic_source["cum_vol_price"] / dynamic_source["cum_vol"]
        series_maps["dynamic_vwap"] = {
            row["datetime"]: float(row["level_value"])
            for _, row in dynamic_source.iloc[dynamic_skip_rows:].iterrows()
        }

        if previous_sessions:
            prev_tail = previous_session_df.tail(1).copy()
            eod_source = pd.concat([prev_tail, today_prepared], ignore_index=True)
            skip_rows = len(prev_tail)
        else:
            eod_source = today_prepared.copy()
            skip_rows = 0

        eod_source["cum_vol"] = eod_source["volume"].cumsum()
        eod_source["cum_vol_price"] = (eod_source["typical_price"] * eod_source["volume"]).cumsum()
        eod_source["level_value"] = eod_source["cum_vol_price"] / eod_source["cum_vol"]
        series_maps["eod_vwap"] = {
            row["datetime"]: float(row["level_value"])
            for _, row in eod_source.iloc[skip_rows:].iterrows()
        }

        confluence_map = {}
        dt_keys = sorted(set(series_maps["vwap"]) | set(series_maps["eod_vwap"]))
        for dt in dt_keys:
            std_val = series_maps["vwap"].get(dt)
            eod_val = series_maps["eod_vwap"].get(dt)
            if std_val is None and eod_val is None:
                continue
            if std_val is None:
                confluence_map[dt] = (eod_val, eod_val)
            elif eod_val is None:
                confluence_map[dt] = (std_val, std_val)
            else:
                confluence_map[dt] = (min(std_val, eod_val), max(std_val, eod_val))
        series_maps["vwap_eod_confluence"] = confluence_map
        return series_maps

    def _build_vwap_invalidation_snapshot(self, df, direction):
        prepared = self._prepare_vwap_frame(df)
        if prepared.empty:
            return None

        current_date = prepared["datetime"].iloc[-1].date()
        today_prepared = prepared[prepared["datetime"].dt.date == current_date].copy()
        if today_prepared.empty:
            return None

        level_series_maps = self._build_intraday_level_series_maps(prepared, current_date)
        if not level_series_maps:
            return None

        consecutive_bad_closes = 0
        recent_rows = []
        for _, row in today_prepared.sort_values("datetime", ascending=False).iterrows():
            bar_dt = row["datetime"]
            close_value = float(row["close"])
            levels = {
                "vwap": level_series_maps.get("vwap", {}).get(bar_dt),
                "dynamic_vwap": level_series_maps.get("dynamic_vwap", {}).get(bar_dt),
                "eod_vwap": level_series_maps.get("eod_vwap", {}).get(bar_dt),
            }
            if any(level is None for level in levels.values()):
                break

            if direction == "long":
                violates_levels = all(close_value < level for level in levels.values())
            else:
                violates_levels = all(close_value > level for level in levels.values())

            recent_rows.append(
                {
                    "datetime": bar_dt.isoformat(),
                    "close": close_value,
                    "vwap": float(levels["vwap"]),
                    "dynamic_vwap": float(levels["dynamic_vwap"]),
                    "eod_vwap": float(levels["eod_vwap"]),
                    "violates_levels": violates_levels,
                }
            )

            if not violates_levels:
                break
            consecutive_bad_closes += 1

        recent_rows.reverse()
        last_row = recent_rows[-1] if recent_rows else None
        return {
            "direction": direction,
            "required_consecutive_closes": VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES,
            "consecutive_bad_closes": consecutive_bad_closes,
            "bars_reviewed": len(recent_rows),
            "last_row": last_row,
            "recent_rows": recent_rows,
        }

    def _load_symbol_classification_cache(self):
        if not SYMBOL_CLASSIFICATION_CACHE_FILENAME.exists():
            return
        try:
            with open(SYMBOL_CLASSIFICATION_CACHE_FILENAME, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    symbol = (row.get("symbol") or "").strip().upper()
                    if symbol:
                        sector_key = slugify_key(row.get("sectorKey") or row.get("sector") or "")
                        industry_key = slugify_key(row.get("industryKey") or row.get("industry") or "")
                        self.symbol_classification_cache[symbol] = {
                            "symbol": symbol,
                            "sectorKey": sector_key,
                            "industryKey": industry_key,
                            "sector": row.get("sector", ""),
                            "industry": row.get("industry", ""),
                            "updated_utc": row.get("updated_utc", ""),
                        }
        except Exception as exc:
            logging.warning(f"Failed loading symbol classification cache: {exc}")

    def _write_symbol_classification_cache(self):
        SYMBOL_CLASSIFICATION_CACHE_FILENAME.parent.mkdir(parents=True, exist_ok=True)
        with open(SYMBOL_CLASSIFICATION_CACHE_FILENAME, "w", newline="") as fh:
            fieldnames = ["symbol", "sectorKey", "industryKey", "sector", "industry", "updated_utc"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for symbol in sorted(self.symbol_classification_cache):
                writer.writerow(self.symbol_classification_cache[symbol])

    def _fetch_symbol_classification(self, symbol):
        sector_name = ""
        industry_name = ""
        sector_key = ""
        industry_key = ""
        try:
            info = yf.Ticker(symbol).get_info()
            sector_name = (info.get("sectorDisp") or info.get("sector") or "").strip()
            industry_name = (info.get("industryDisp") or info.get("industry") or "").strip()
            sector_key = (info.get("sectorKey") or "").strip()
            industry_key = (info.get("industryKey") or "").strip()
        except Exception as exc:
            logging.warning(f"Yahoo classification failed for {symbol}: {exc}")
        sector_key = slugify_key(sector_key or sector_name)
        industry_key = slugify_key(industry_key or industry_name)
        return {
            "symbol": symbol,
            "sectorKey": sector_key,
            "industryKey": industry_key,
            "sector": sector_name,
            "industry": industry_name,
            "updated_utc": utc_now_iso(),
        }

    def get_symbol_classification(self, symbol):
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return None
        cached = self.symbol_classification_cache.get(symbol)
        if cached is not None:
            return cached
        row = self._fetch_symbol_classification(symbol)
        self.symbol_classification_cache[symbol] = row
        self._write_symbol_classification_cache()
        if row.get("industryKey"):
            self.industry_map_data = load_and_update_industry_etf_map(
                row.get("industryKey"),
                row.get("sectorKey"),
                row.get("industry"),
                row.get("sector"),
            )
        return row

    def _get_cached_bars(self, symbol, duration, bar_size):
        key = f"{symbol}|{duration}|{bar_size}"
        bars = self.latest_bars.get(key)
        if bars:
            return bars
        bars_raw = self.request_historical_bars(symbol, duration, bar_size, timeout=RRS_TIMEOUT)
        if not bars_raw:
            return []
        bars_ib = _dedupe_bars(_bars_to_ib(bars_raw))
        self.latest_bars[key] = bars_ib
        self.latest_bars.setdefault(symbol, bars_ib)
        return bars_ib

    def request_historical_bars(self, symbol, duration, bar_size, timeout=RRS_TIMEOUT):
        if symbol in self.invalid_security_symbols:
            logging.debug(f"{symbol}: Skipping historical request due to prior IB security-definition failure.")
            return None
        if not self.ensure_connected():
            logging.warning("Not connected to IB; skipping historical request.")
            return None
        reqId = self.getReqId()
        with self.data_lock:
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
            self.reqid_to_symbol[reqId] = symbol
        contract = self.create_stock_contract(symbol)
        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        if not self.data_ready_events[reqId].wait(timeout=timeout):
            with self.data_lock:
                del self.data_ready_events[reqId]
                self.data.pop(reqId, None)
                self.reqid_to_symbol.pop(reqId, None)
            logging.warning(f"{symbol}: Timeout waiting for RRS data.")
            return None
        with self.data_lock:
            bars = self.data.get(reqId, [])
            del self.data_ready_events[reqId]
            self.data.pop(reqId, None)
            self.reqid_to_symbol.pop(reqId, None)
        return bars

    def get_cached_5m_bars(self, symbol):
        return self._get_cached_bars(symbol, "5 D", "5 mins")

    def run_rrs_scan(self, timeframe_key_override=None, emit_gui=True):
        self.load_master_avwap_focus()
        threshold, bar_size, duration, length, timeframe_key = self.get_rrs_settings()
        if timeframe_key_override in RRS_TIMEFRAMES:
            timeframe_key = timeframe_key_override
            bar_size = RRS_TIMEFRAMES[timeframe_key]["bar_size"]
            duration = RRS_TIMEFRAMES[timeframe_key]["duration"]
        timeframe_minutes = RRS_TIMEFRAMES[timeframe_key]["minutes"]
        if emit_gui and self.gui_callback:
            self.gui_callback(f"RRS scan running ({RRS_TIMEFRAMES[timeframe_key]['label']})...", "rrs_status")

        spy_5m = self.get_cached_5m_bars("SPY")
        if not spy_5m:
            if emit_gui and self.gui_callback:
                self.gui_callback("RRS scan: SPY data unavailable.", "rrs_status")
            return

        spy_bars = spy_5m if timeframe_minutes == 5 else _aggregate_bars_timeframe(spy_5m, timeframe_minutes)
        spy_by_dt = {bar.dt: bar for bar in spy_bars}
        spy_move_ratio = self._calc_move_ratio(spy_bars, length)

        results = []
        sector_results = []
        industry_results = []
        symbol_context = []
        all_scores = []
        intraday_profiles = {}
        all_symbols = sorted(set(self.longs + self.shorts) - {"SPY"})
        spy_context_windows = self._build_spy_context_windows(spy_5m, length)
        for symbol in all_symbols:
            sym_5m = self.get_cached_5m_bars(symbol)
            if not sym_5m:
                continue
            intraday_profile = self._build_intraday_rrs_profile(sym_5m, spy_5m, length=length)
            if intraday_profile:
                current_date = spy_5m[-1].dt.date()
                today_profile = [item for item in intraday_profile if item.get("dt") and item["dt"].date() == current_date]
                if today_profile:
                    intraday_profiles[symbol] = today_profile
            sym_bars = sym_5m if timeframe_minutes == 5 else _aggregate_bars_timeframe(sym_5m, timeframe_minutes)
            aligned_sym, aligned_spy = _align_bars_with_map(sym_bars, spy_by_dt)
            rrs_value, power_index = real_relative_strength(aligned_sym, aligned_spy, length=length)
            if rrs_value is None:
                continue
            symbol_move_ratio = self._calc_move_ratio(sym_bars, length)
            excess_move_ratio = None
            if symbol_move_ratio is not None and spy_move_ratio is not None:
                excess_move_ratio = symbol_move_ratio - spy_move_ratio
            all_scores.append((symbol, rrs_value, power_index))
            environment_signal = None
            if rrs_value >= threshold:
                environment_signal = "RS"
            elif rrs_value <= -threshold:
                environment_signal = "RW"

            if environment_signal:
                symbol_context.append(
                    {
                        "symbol": symbol,
                        "signal": environment_signal,
                        "rrs": rrs_value,
                        "move_ratio": symbol_move_ratio,
                        "excess_move_ratio": excess_move_ratio,
                        "power_index": power_index,
                        "watchlist_bias": (
                            "long" if symbol in self.longs else "short" if symbol in self.shorts else ""
                        ),
                    }
                )

            if symbol in self.longs and rrs_value >= threshold:
                results.append(("RS", symbol, rrs_value, power_index))
            elif symbol in self.shorts and rrs_value <= -threshold:
                results.append(("RW", symbol, rrs_value, power_index))

            classification = self.get_symbol_classification(symbol)
            if not classification:
                continue
            sector_key = classification.get("sectorKey", "")
            industry_key = classification.get("industryKey", "")
            if industry_key:
                self.industry_map_data = load_and_update_industry_etf_map(
                    industry_key,
                    sector_key,
                    classification.get("industry", ""),
                    classification.get("sector", ""),
                )

            sector_etf = resolve_sector_etf(sector_key, self.sector_etf_map)
            sec_5m = self.get_cached_5m_bars(sector_etf)
            if sec_5m:
                sec_bars = sec_5m if timeframe_minutes == 5 else _aggregate_bars_timeframe(sec_5m, timeframe_minutes)
                aligned_sym_sec, aligned_sec = _align_bars_with_map(sym_bars, {bar.dt: bar for bar in sec_bars})
                sec_rrs, sec_power = real_relative_strength(aligned_sym_sec, aligned_sec, length=length)
                if sec_rrs is not None:
                    if symbol in self.longs and sec_rrs >= threshold:
                        sector_results.append(("RS", symbol, sec_rrs, sec_power))
                    elif symbol in self.shorts and sec_rrs <= -threshold:
                        sector_results.append(("RW", symbol, sec_rrs, sec_power))

            industry_ref = resolve_industry_ref_etf(industry_key, sector_key)
            ind_5m = self.get_cached_5m_bars(industry_ref)
            if ind_5m:
                ind_bars = ind_5m if timeframe_minutes == 5 else _aggregate_bars_timeframe(ind_5m, timeframe_minutes)
                aligned_sym_ind, aligned_ind = _align_bars_with_map(sym_bars, {bar.dt: bar for bar in ind_bars})
                ind_rrs, ind_power = real_relative_strength(aligned_sym_ind, aligned_ind, length=length)
                if ind_rrs is not None:
                    if symbol in self.longs and ind_rrs >= threshold:
                        industry_results.append(("RS", symbol, ind_rrs, ind_power))
                    elif symbol in self.shorts and ind_rrs <= -threshold:
                        industry_results.append(("RW", symbol, ind_rrs, ind_power))

        strongest = sorted(all_scores, key=lambda row: row[1], reverse=True)[:SCAN_EXTREME_COUNT]
        weakest = sorted(all_scores, key=lambda row: row[1])[:SCAN_EXTREME_COUNT]
        self.latest_scan_extremes[timeframe_key] = strongest + weakest
        if strongest or weakest:
            self._log_scan_extremes(timeframe_key, strongest, weakest)

        rs_results = sorted([r for r in results if r[0] == "RS"], key=lambda r: -r[2])
        rw_results = sorted([r for r in results if r[0] == "RW"], key=lambda r: r[2])
        ordered_results = rs_results + rw_results

        def _ordered(rows):
            rs = sorted([r for r in rows if r[0] == "RS"], key=lambda r: -r[2])
            rw = sorted([r for r in rows if r[0] == "RW"], key=lambda r: r[2])
            return rs + rw

        sector_payload = _ordered(sector_results)
        industry_payload = _ordered(industry_results)
        environment_scan = self._summarize_environment_scan(intraday_profiles, spy_context_windows, threshold)
        snapshot_payload = {
            "timestamp": datetime.now(),
            "threshold": threshold,
            "timeframe_key": timeframe_key,
            "results": ordered_results,
            "results_sector": sector_payload,
            "results_industry": industry_payload,
            "group_strength": self.compute_group_strengths(),
            "symbol_context": symbol_context,
            "spy_move_ratio": spy_move_ratio,
            "environment_scan": environment_scan,
        }
        if emit_gui:
            self._emit_master_avwap_focus_rrs_alerts(symbol_context, threshold, timeframe_key)
        self.latest_rrs_payload = snapshot_payload
        if emit_gui and self.gui_callback:
            decorated_snapshot = self._decorate_snapshot(snapshot_payload)
            self.gui_callback(decorated_snapshot, "rrs_snapshot")
            self.gui_callback(
                f"RRS scan complete ({len(ordered_results)} SPY, {len(sector_payload)} sector, {len(industry_payload)} industry refs)",
                "rrs_status",
            )

    def compute_group_strengths(self):
        results = {}
        industry_refs = self.industry_map_data.get("yahoo_industryKey_to_ref", {})
        industry_etfs = sorted({(v.get("etf") or "").strip().upper() for v in industry_refs.values() if isinstance(v, dict) and (v.get("etf") or "").strip()})
        for tf_key, tf in GROUP_STRENGTH_TIMEFRAMES.items():
            spy = self._get_cached_bars("SPY", tf["duration"], tf["bar_size"])
            if not spy:
                continue
            spy_by_dt = {bar.dt: bar for bar in spy}
            sectors = []
            for sector_key, etf in sorted(self.sector_etf_map.items()):
                bars = self._get_cached_bars(etf, tf["duration"], tf["bar_size"])
                if not bars:
                    continue
                aligned_etf, aligned_spy = _align_bars_with_map(bars, spy_by_dt)
                rrs, power = real_relative_strength(aligned_etf, aligned_spy, length=self.rrs_length)
                if rrs is not None:
                    sectors.append({"group_key": sector_key, "etf": etf, "rrs": rrs, "power_index": power})
            industries = []
            for etf in industry_etfs:
                bars = self._get_cached_bars(etf, tf["duration"], tf["bar_size"])
                if not bars:
                    continue
                aligned_etf, aligned_spy = _align_bars_with_map(bars, spy_by_dt)
                rrs, power = real_relative_strength(aligned_etf, aligned_spy, length=self.rrs_length)
                if rrs is not None:
                    industries.append({"group_key": etf, "etf": etf, "rrs": rrs, "power_index": power})
            results[tf_key] = {
                "sectors": sorted(sectors, key=lambda x: x["rrs"], reverse=True),
                "industries": sorted(industries, key=lambda x: x["rrs"], reverse=True),
            }
        self.latest_group_extremes = results
        self._log_group_strength_extremes(results)
        return results

    def _log_group_strength_extremes(self, grouped):
        timestamp_local = datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
        GROUP_STRENGTH_SCAN_LOG_FILENAME.parent.mkdir(parents=True, exist_ok=True)
        write_header = not GROUP_STRENGTH_SCAN_LOG_FILENAME.exists()
        with open(GROUP_STRENGTH_SCAN_LOG_FILENAME, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["timestamp_local", "timeframe", "group_type", "group_key", "etf", "rrs", "power_index"],
            )
            if write_header:
                writer.writeheader()
            for timeframe, payload in grouped.items():
                for group_type, items in (("sector", payload.get("sectors", [])), ("industry", payload.get("industries", []))):
                    top = items[:SCAN_EXTREME_COUNT]
                    bottom = list(reversed(items[-SCAN_EXTREME_COUNT:])) if len(items) > SCAN_EXTREME_COUNT else []
                    for item in top + bottom:
                        writer.writerow({
                            "timestamp_local": timestamp_local,
                            "timeframe": timeframe,
                            "group_type": group_type,
                            "group_key": item.get("group_key", ""),
                            "etf": item.get("etf", ""),
                            "rrs": f"{item.get('rrs', 0.0):.4f}",
                            "power_index": f"{item.get('power_index', 0.0):.4f}",
                        })

    def _log_scan_extremes(self, timeframe_key, strongest, weakest):
        timestamp_local = datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
        STRENGTH_SCAN_LOG_FILENAME.parent.mkdir(parents=True, exist_ok=True)
        write_header = not STRENGTH_SCAN_LOG_FILENAME.exists()
        with open(STRENGTH_SCAN_LOG_FILENAME, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["timestamp_local", "timeframe", "bucket", "symbol", "rrs", "power_index"],
            )
            if write_header:
                writer.writeheader()
            for bucket, items in (("strongest", strongest), ("weakest", weakest)):
                for symbol, rrs_value, power_index in items:
                    writer.writerow(
                        {
                            "timestamp_local": timestamp_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
                            "timeframe": timeframe_key,
                            "bucket": bucket,
                            "symbol": symbol,
                            "rrs": f"{rrs_value:.4f}",
                            "power_index": (
                                f"{power_index:.4f}" if power_index is not None else ""
                            ),
                        }
                    )

    def get_monitored_extreme_symbols(self):
        symbols = set()
        for entries in self.latest_scan_extremes.values():
            symbols.update(item[0] for item in entries)
        return symbols

    def has_minimum_candles_completed(self, required=CONSECUTIVE_CANDLES):
        now = datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
        market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)
        if now < market_open:
            return False
        elapsed = (now - market_open).total_seconds()
        return elapsed >= required * 300

    def nextValidId(self, orderId):
        self.connection_status = True
        self.client_id_conflict = False
        logging.info(f"Connected to IB API. NextValidId={orderId}")

    def error(self, reqId, errorCode, errorString):
        logging.error(f"IB Error. ReqId={reqId}, Code={errorCode}, Msg={errorString}")
        if errorCode == 200:
            with self.data_lock:
                symbol = self.reqid_to_symbol.get(reqId)
                if symbol:
                    self.invalid_security_symbols.add(symbol)
                if reqId in self.data_ready_events:
                    self.data_ready_events[reqId].set()
        if errorCode == 326:
            self.connection_status = False
            self.client_id_conflict = True
        if errorCode in (504, 1100) or "not connected" in errorString.lower():
            self.connection_status = False
        elif errorCode in (1101, 1102):
            self.connection_status = True

    def connectionClosed(self):
        self.connection_status = False
        logging.warning("IB connection closed.")

    def historicalData(self, reqId, bar):
        with self.data_lock:
            if reqId not in self.data:
                self.data[reqId] = []
            self.data[reqId].append({
                "time": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            })
            logging.debug(f"ReqId={reqId} | Received bar: {bar.date} O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close}")

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        logging.debug(f"Historical data end: ReqId={reqId}, Start={start}, End={end}. Total bars: {len(self.data.get(reqId, []))}")
        with self.data_lock:
            if reqId in self.data_ready_events:
                self.data_ready_events[reqId].set()

    def calculate_atr(self, df_daily, period=ATR_PERIOD):
        # Calculate True Range
        df_daily["prev_close"] = df_daily["close"].shift(1)
        df_daily["h_pc"] = abs(df_daily["high"] - df_daily["prev_close"])
        df_daily["l_pc"] = abs(df_daily["low"] - df_daily["prev_close"])
        df_daily["h_l"] = abs(df_daily["high"] - df_daily["low"])
        df_daily["tr"] = df_daily[["h_pc", "l_pc", "h_l"]].max(axis=1)
        
        # Initialize ATR
        df_daily["atr"] = float('nan')
        
        # First ATR value is the simple average of TR
        first_atr = df_daily["tr"].iloc[:period].mean()
        df_daily.loc[df_daily.index[period-1], "atr"] = first_atr
        
        # Apply Wilder's smoothing formula
        multiplier = 1.0 / period
        for i in range(period, len(df_daily)):
            prev_atr = df_daily["atr"].iloc[i-1]
            current_tr = df_daily["tr"].iloc[i]
            df_daily.loc[df_daily.index[i], "atr"] = (prev_atr * (period - 1) + current_tr) * multiplier
        
        return df_daily["atr"].iloc[-1]
    
    


    def calculate_vwap(self, df):
        if df.empty or ('volume' not in df) or (df['volume'].sum() == 0):
            return None
        
        df = df.copy()
        
        # TC2000 style typical price calculation
        df["typical"] = (df["high"] + df["low"] + df["open"] + df["close"]) / 4
        
        # Calculate VWAP
        vwap = (df["typical"] * df["volume"]).sum() / df["volume"].sum()
        
        return vwap
    
    def calculate_vwap_with_stdev_bands(self, df, band_mult=1.0):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None, None, None

        current_date = prepared_df["datetime"].iloc[-1].date()
        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        vwap, upper_band, lower_band = self._calculate_vwap_bands(today_df, band_mult=band_mult)

        if LOGGING_MODE and vwap is not None:
            logging.debug(f"Standard VWAP calculation: VWAP={vwap:.4f}")
            logging.debug(f"Standard VWAP bands: Upper({band_mult}x)={upper_band:.4f}, Lower({band_mult}x)={lower_band:.4f}")

        return vwap, upper_band, lower_band




    def _prepare_vwap_frame(self, df):
        if df.empty:
            return pd.DataFrame()

        df_copy = df.copy()
        df_copy["datetime"] = pd.to_datetime(
            df_copy["time"],
            format="%Y%m%d  %H:%M:%S",
            errors="coerce",
        )
        if df_copy["datetime"].isna().all():
            df_copy["datetime"] = pd.to_datetime(df_copy["time"], errors="coerce")
        if df_copy["datetime"].isna().all():
            logging.error("Failed to convert time to datetime for VWAP calculation")
            return pd.DataFrame()

        df_copy = df_copy.sort_values("datetime").reset_index(drop=True)
        df_copy["typical_price"] = (
            df_copy["high"] + df_copy["low"] + df_copy["open"] + df_copy["close"]
        ) / 4.0
        return df_copy

    def _calculate_vwap_value(self, df):
        if df.empty or df["volume"].sum() == 0:
            return None

        vol_price = df["typical_price"] * df["volume"]
        return vol_price.sum() / df["volume"].sum()

    def _calculate_vwap_bands(self, df, band_mult=1.0):
        if df.empty or df["volume"].sum() == 0:
            return None, None, None

        df = df.copy()
        df["vol_times_price"] = df["typical_price"] * df["volume"]
        cum_vol = df["volume"].cumsum()
        cum_vol_price = df["vol_times_price"].cumsum()
        df["vwap"] = cum_vol_price / cum_vol

        df["sq_dev"] = ((df["typical_price"] - df["vwap"]) ** 2) * df["volume"]
        cum_sq_dev = df["sq_dev"].cumsum()
        df["stdev"] = (cum_sq_dev / cum_vol).apply(lambda x: math.sqrt(x) if x > 0 else 0)

        vwap = df["vwap"].iloc[-1]
        stdev = df["stdev"].iloc[-1]
        upper_band = vwap + stdev * band_mult
        lower_band = vwap - stdev * band_mult
        return vwap, upper_band, lower_band

    def calculate_standard_vwap(self, df):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None

        current_date = prepared_df["datetime"].iloc[-1].date()
        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        if today_df.empty:
            logging.warning("No today's data available for standard VWAP calculation")
            return None

        logging.debug(f"Standard VWAP using data from {current_date} only")
        logging.debug(f"Today's data count: {len(today_df)}")

        return self._calculate_vwap_value(today_df)

    def calculate_dynamic_vwap(self, df):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None

        unique_dates = sorted(prepared_df["datetime"].dt.date.unique())
        current_date = unique_dates[-1]
        previous_date = unique_dates[-2] if len(unique_dates) > 1 else None

        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        previous_df = (
            prepared_df[prepared_df["datetime"].dt.date == previous_date]
            if previous_date
            else pd.DataFrame()
        )

        logging.debug(
            f"Dynamic VWAP using data from {previous_date} and {current_date}"
        )
        logging.debug(
            f"Previous day data count: {len(previous_df)}, Today's data count: {len(today_df)}"
        )

        combined_df = (
            pd.concat([previous_df, today_df])
            if not previous_df.empty
            else today_df
        )
        return self._calculate_vwap_value(combined_df)

    def calculate_eod_vwap(self, df):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None

        unique_dates = sorted(prepared_df["datetime"].dt.date.unique())
        current_date = unique_dates[-1]
        previous_date = unique_dates[-2] if len(unique_dates) > 1 else None

        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        previous_df = (
            prepared_df[prepared_df["datetime"].dt.date == previous_date]
            if previous_date
            else pd.DataFrame()
        )

        logging.debug(
            f"EOD VWAP using data from {current_date} and last candle from {previous_date}"
        )
        logging.debug(
            f"Previous day data count: {len(previous_df)}, Today's data count: {len(today_df)}"
        )

        if previous_df.empty:
            return self._calculate_vwap_value(today_df)

        eod_df = pd.concat([previous_df.tail(1), today_df])
        return self._calculate_vwap_value(eod_df)
    
    def calculate_dynamic_vwap_with_stdev_bands(self, df, band_mult=1.0):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None, None, None

        unique_dates = sorted(prepared_df["datetime"].dt.date.unique())
        current_date = unique_dates[-1]
        previous_date = unique_dates[-2] if len(unique_dates) > 1 else None

        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        prev_day_df = (
            prepared_df[prepared_df["datetime"].dt.date == previous_date]
            if previous_date
            else pd.DataFrame()
        )

        if prev_day_df.empty and previous_date:
            symbol_key = f"dynamic_vwap_{current_date}"
            if symbol_key not in self.warned_symbols:
                logging.warning(
                    f"No previous trading day data for dynamic VWAP bands (looking for {previous_date})"
                )
                self.warned_symbols.add(symbol_key)

        combined_df = (
            pd.concat([prev_day_df, today_df])
            if not prev_day_df.empty
            else today_df
        )
        dynamic_vwap, upper_band, lower_band = self._calculate_vwap_bands(
            combined_df,
            band_mult=band_mult,
        )

        if LOGGING_MODE and dynamic_vwap is not None:
            logging.debug(f"Dynamic VWAP calculation: VWAP={dynamic_vwap:.4f}")
            logging.debug(
                f"Dynamic VWAP bands: Upper({band_mult}x)={upper_band:.4f}, Lower({band_mult}x)={lower_band:.4f}"
            )

        return dynamic_vwap, upper_band, lower_band



    def calculate_eod_vwap_with_stdev_bands(self, df, band_mult=1.0):
        prepared_df = self._prepare_vwap_frame(df)
        if prepared_df.empty:
            return None, None, None

        unique_dates = sorted(prepared_df["datetime"].dt.date.unique())
        current_date = unique_dates[-1]
        previous_date = unique_dates[-2] if len(unique_dates) > 1 else None

        today_df = prepared_df[prepared_df["datetime"].dt.date == current_date]
        prev_day_df = (
            prepared_df[prepared_df["datetime"].dt.date == previous_date]
            if previous_date
            else pd.DataFrame()
        )

        if prev_day_df.empty and previous_date:
            symbol_key = f"eod_vwap_{current_date}"
            if symbol_key not in self.warned_symbols:
                logging.warning(
                    f"No previous trading day data for EOD VWAP bands (looking for {previous_date})"
                )
                self.warned_symbols.add(symbol_key)

        eod_df = (
            pd.concat([prev_day_df.tail(1), today_df])
            if not prev_day_df.empty
            else today_df
        )
        eod_vwap, upper_band, lower_band = self._calculate_vwap_bands(
            eod_df,
            band_mult=band_mult,
        )

        if LOGGING_MODE and eod_vwap is not None:
            logging.debug(f"EOD VWAP calculation: VWAP={eod_vwap:.4f}")
            logging.debug(
                f"EOD VWAP bands: Upper({band_mult}x)={upper_band:.4f}, Lower({band_mult}x)={lower_band:.4f}"
            )

        return eod_vwap, upper_band, lower_band



 

    def calculate_dynamic_vwap2(self, df):
        try:
            df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
        except Exception as e:
            logging.warning(f"Datetime conversion error in dynamic VWAP2: {e}")
            return None
        
        if df.empty:
            return None
        
        today_date = df["datetime"].iloc[-1].date()
        two_days_ago = today_date - timedelta(days=2)
        
        today_df = df[df["datetime"].dt.date == today_date].copy()
        two_days_ago_df = df[df["datetime"].dt.date == two_days_ago].copy()
        
        dynamic_df = pd.concat([two_days_ago_df, today_df]) if not two_days_ago_df.empty else today_df
        
        return self.calculate_vwap(dynamic_df)

    def get_previous_day_extremes(self, df):
        try:
            df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
        except Exception as e:
            logging.warning(f"Error converting datetime: {e}")
            return None, None
        current_date = df["datetimde"].iloc[-1].date()
        prev_session = df[df["datetime"].dt.date < current_date]
        if prev_session.empty:
            return None, None
        prev_high = prev_session["high"].max()
        prev_low = prev_session["low"].min()
        return prev_high, prev_low
    


    def build_atr_cache(self):
        all_symbols = set(self.longs + self.shorts)
        to_fetch = [s for s in all_symbols if s not in self.atr_cache]
        if not to_fetch:
            logging.debug("No new symbols for ATR update.")
            return
        for sym in to_fetch:
            if not sym.isalnum():
                logging.error(f"Invalid symbol: {sym}. Skipping ATR calculation.")
                self.atr_cache[sym] = None
                continue
            reqId = self.getReqId()
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
            contract = self.create_stock_contract(sym)
            logging.debug(f"Requesting 20-day ATR data for {sym} with reqId {reqId}")
            self.reqHistoricalData(
                reqId=reqId,
                contract=contract,
                endDateTime="",
                durationStr="20 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            if self.data_ready_events[reqId].wait(timeout=15):
                bars = self.data.get(reqId, [])
                if len(bars) < ATR_PERIOD:
                    self.atr_cache[sym] = None
                    logging.warning(f"{sym}: Insufficient data for ATR calculation.")
                else:
                    df_daily = pd.DataFrame(bars)
                    df_daily["datetime"] = pd.to_datetime(df_daily["time"], errors="coerce")
                    atr_val = self.calculate_atr(df_daily, period=ATR_PERIOD)
                    self.atr_cache[sym] = atr_val
                    logging.info(f"{sym}: 20-day ATR = {atr_val:.2f}")
            else:
                self.atr_cache[sym] = None
                logging.warning(f"{sym}: ATR data request timed out.")
            del self.data_ready_events[reqId]

    # Helper function for color coded logging (non-ATR messages)
    # Option 1: Update the method definition to accept a tag parameter with a default value
    def log_symbol(self, symbol, msg, tag=None):
        if "ATR" in msg:
            logging.info(msg)
        else:
            if symbol in self.longs:
                colored_msg = Fore.GREEN + msg + Style.RESET_ALL
            elif symbol in self.shorts:
                colored_msg = Fore.RED + msg + Style.RESET_ALL
            else:
                colored_msg = msg
            logging.info(colored_msg)


    def log_symbol_metrics(self, symbol, df_5m):
        # Get the current price from the last candle
        current_price = df_5m["close"].iloc[-1] if not df_5m.empty else None
        
        # Calculate the different VWAPs directly
        standard_vwap = self.calculate_standard_vwap(df_5m)
        dynamic_vwap = self.calculate_dynamic_vwap(df_5m)
        eod_vwap = self.calculate_eod_vwap(df_5m)
        
        # Get previous day extremes
        prev_high, prev_low = self.get_previous_day_extremes(df_5m)
        
        # Get ATR value
        atr_val = self.atr_cache.get(symbol, float('nan'))
        
        # Build the log message
        msg = f"{symbol} Metrics -> "
        msg += f"Price: {current_price:.2f}, " if current_price is not None else "Price: N/A, "
        msg += f"Std VWAP: {standard_vwap:.2f}, " if standard_vwap is not None else "Std VWAP: N/A, "
        msg += f"Dynamic VWAP: {dynamic_vwap:.2f}, " if dynamic_vwap is not None else "Dynamic VWAP: N/A, "
        msg += f"EOD VWAP: {eod_vwap:.2f}, " if eod_vwap is not None else "EOD VWAP: N/A, "
        
        if symbol in self.longs:
            msg += f"Prev Day High: {prev_high:.2f}, " if prev_high is not None else "Prev Day High: N/A, "
        elif symbol in self.shorts:
            msg += f"Prev Day Low: {prev_low:.2f}, " if prev_low is not None else "Prev Day Low: N/A, "
        
        msg += f"ATR: {atr_val:.2f}"
        self.log_symbol(symbol, msg)



    def evaluate_bounce_candidate(self, symbol, df, allowed_bounce_types=None):
        if len(df) < 10:
            return None

        # Get the current candle
        current_candle = df.iloc[-1].copy()
        
        # Ensure we're working with scalar values, not Series objects
        def get_scalar(value):
            if isinstance(value, pd.Series):
                return value.iloc[0] if len(value) > 0 else value
            return value

        # Normalize current candle data
        current_candle_data = {
            "open": get_scalar(current_candle["open"]),
            "high": get_scalar(current_candle["high"]),
            "low": get_scalar(current_candle["low"]),
            "close": get_scalar(current_candle["close"]),
            "volume": get_scalar(current_candle["volume"]),
            "time": current_candle["time"]
        }
        
        # Get ATR for this symbol
        atr = self.atr_cache.get(symbol, None)
        if atr is None:
            logging.debug(f"{symbol}: No ATR available, skipping bounce evaluation")
            return None

        # Set threshold for proximity to levels
        threshold = THRESHOLD_MULTIPLIER * atr
        direction = "long" if symbol in self.longs else "short"
        
        # Get the metrics from the cache
        metrics = self.symbol_metrics.get(symbol, {})
        if not metrics:
            logging.debug(f"{symbol}: No metrics available, skipping bounce evaluation")
            return None
            
        # Add datetime column if it doesn't exist
        if "datetime" not in df.columns:
            try:
                df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
            except Exception as e:
                logging.error(f"{symbol}: Error in datetime conversion: {e}")
                return None
        
        # Get today's candles
        current_date = df["datetime"].iloc[-1].date()
        today_df = df[df["datetime"].dt.date == current_date].copy()
        level_series_maps = self._build_intraday_level_series_maps(df, current_date)

        # Skip if we don't have enough candles for today
        if len(today_df) < CONSECUTIVE_CANDLES + 1:  # +1 for current candle
            logging.debug(f"{symbol}: Not enough candles for today ({len(today_df)}) to check consecutive condition")
            # We can still continue without the consecutive check if it's disabled
            if CHECK_CONSECUTIVE_CANDLES:
                return None

        # Estimate intraday ATR from today's 5-minute candles for impulse sizing.
        intraday_impulse_atr = None
        try:
            if len(today_df) >= 3:
                intraday_df = today_df[["high", "low", "close"]].copy()
                intraday_df["prev_close"] = intraday_df["close"].shift(1)
                tr_components = pd.concat(
                    [
                        (intraday_df["high"] - intraday_df["low"]).abs(),
                        (intraday_df["high"] - intraday_df["prev_close"]).abs(),
                        (intraday_df["low"] - intraday_df["prev_close"]).abs(),
                    ],
                    axis=1,
                )
                intraday_df["tr"] = tr_components.max(axis=1)
                lookback = min(IMPULSE_INTRADAY_ATR_PERIOD, len(intraday_df))
                intraday_impulse_atr = float(intraday_df["tr"].tail(lookback).mean())
                if intraday_impulse_atr <= 0 or math.isnan(intraday_impulse_atr):
                    intraday_impulse_atr = None
        except Exception:
            intraday_impulse_atr = None

        intraday_rrs_profile = []
        try:
            spy_5m = self.latest_bars.get("SPY") or self.get_cached_5m_bars("SPY")
            if spy_5m:
                today_symbol_bars = _dedupe_bars(_bars_to_ib(today_df.to_dict("records")))
                today_spy_bars = [bar for bar in spy_5m if bar.dt.date() == current_date]
                intraday_rrs_profile = self._build_intraday_rrs_profile(
                    today_symbol_bars,
                    today_spy_bars,
                    length=IMPULSE_RRS_PROFILE_LENGTH,
                )
        except Exception as exc:
            logging.debug(f"{symbol}: Failed building intraday RS/RW profile for impulse filter: {exc}")
            intraday_rrs_profile = []
        
        # Log the metrics being used for evaluation
        if LOGGING_MODE:
            std_vwap_str = f"{metrics.get('std_vwap'):.4f}" if metrics.get('std_vwap') is not None else "None"
            dynamic_vwap_str = f"{metrics.get('dynamic_vwap'):.4f}" if metrics.get('dynamic_vwap') is not None else "None"
            eod_vwap_str = f"{metrics.get('eod_vwap'):.4f}" if metrics.get('eod_vwap') is not None else "None"
            ema_8_str = f"{metrics.get('ema_8'):.4f}" if metrics.get('ema_8') is not None else "None"
            ema_15_str = f"{metrics.get('ema_15'):.4f}" if metrics.get('ema_15') is not None else "None"
            ema_21_str = f"{metrics.get('ema_21'):.4f}" if metrics.get('ema_21') is not None else "None"
            upper_band_str = f"{metrics.get('vwap_1stdev_upper'):.4f}" if metrics.get('vwap_1stdev_upper') is not None else "None"
            lower_band_str = f"{metrics.get('vwap_1stdev_lower'):.4f}" if metrics.get('vwap_1stdev_lower') is not None else "None"
            dynamic_upper_str = f"{metrics.get('dynamic_vwap_1stdev_upper'):.4f}" if metrics.get('dynamic_vwap_1stdev_upper') is not None else "None"
            dynamic_lower_str = f"{metrics.get('dynamic_vwap_1stdev_lower'):.4f}" if metrics.get('dynamic_vwap_1stdev_lower') is not None else "None"
            eod_upper_str = f"{metrics.get('eod_vwap_1stdev_upper'):.4f}" if metrics.get('eod_vwap_1stdev_upper') is not None else "None"
            eod_lower_str = f"{metrics.get('eod_vwap_1stdev_lower'):.4f}" if metrics.get('eod_vwap_1stdev_lower') is not None else "None"
            
            logging.debug(f"{symbol} evaluation using - Std VWAP: {std_vwap_str}, Dynamic VWAP: {dynamic_vwap_str}, EOD VWAP: {eod_vwap_str}, EMA8: {ema_8_str}, EMA15: {ema_15_str}, EMA21: {ema_21_str}")
            logging.debug(f"{symbol} bands - Std 1SD Upper: {upper_band_str}, Std 1SD Lower: {lower_band_str}")
            logging.debug(f"{symbol} dyn bands - Dynamic 1SD Upper: {dynamic_upper_str}, Dynamic 1SD Lower: {dynamic_lower_str}")
            logging.debug(f"{symbol} eod bands - EOD 1SD Upper: {eod_upper_str}, EOD 1SD Lower: {eod_lower_str}")

        # Initialize dictionary for reference levels that triggered bounce condition
        ref_levels = {}
        triggered_levels = []
        allowed_types = set(allowed_bounce_types) if allowed_bounce_types is not None else None

        def bounce_type_allowed(bounce_type):
            return allowed_types is None or bounce_type in allowed_types

        def mark_trigger(level_name):
            if level_name not in triggered_levels:
                triggered_levels.append(level_name)

        def is_touch_reject(level_value):
            if level_value is None:
                return False
            interacted = (
                current_candle_data["high"] >= (level_value - threshold)
                and current_candle_data["low"] <= (level_value + threshold)
            )
            if direction == "long":
                return (
                    interacted
                    and current_candle_data["close"] >= (level_value - threshold)
                    and current_candle_data["close"] > current_candle_data["open"]
                )
            return (
                interacted
                and current_candle_data["close"] <= (level_value + threshold)
                and current_candle_data["close"] < current_candle_data["open"]
            )

        def is_recent_reclaim(level_value, level_name):
            if level_value is None or len(today_df) < 2:
                return False
            lookback = min(RECLAIM_LOOKBACK_CANDLES, len(today_df) - 1)
            previous_candles = today_df.iloc[-(lookback + 1):-1]
            if previous_candles.empty:
                return False

            if direction == "long":
                was_below = (previous_candles["close"] <= (level_value + threshold)).any()
                reclaimed_now = (
                    current_candle_data["close"] > level_value
                    and current_candle_data["high"] >= (level_value - threshold)
                    and current_candle_data["low"] <= (level_value + threshold)
                    and current_candle_data["close"] > current_candle_data["open"]
                )
            else:
                was_above = (previous_candles["close"] >= (level_value - threshold)).any()
                reclaimed_now = (
                    current_candle_data["close"] < level_value
                    and current_candle_data["high"] >= (level_value - threshold)
                    and current_candle_data["low"] <= (level_value + threshold)
                    and current_candle_data["close"] < current_candle_data["open"]
                )
                was_below = was_above

            reclaimed = bool(was_below and reclaimed_now)
            if reclaimed:
                logging.debug(
                    f"{symbol}: {level_name} reclaim detected within last {lookback} candles"
                )
            return reclaimed

        def is_vwap_zone_reaction(level_low, level_high, level_name):
            if level_low is None or level_high is None:
                return False

            near_buffer = max(threshold, VWAP_BOUNCE_NEAR_ATR * atr)
            pierce_buffer = max(threshold, VWAP_BOUNCE_PIERCE_ATR * atr)
            close_buffer = max(threshold, VWAP_BOUNCE_CLOSE_BUFFER_ATR * atr)

            if direction == "long":
                approached_zone = current_candle_data["low"] <= (level_high + near_buffer)
                limited_undershoot = current_candle_data["low"] >= (level_low - pierce_buffer)
                recovered = (
                    current_candle_data["close"] >= (level_high - close_buffer)
                    and current_candle_data["close"] > current_candle_data["open"]
                )
                distance = level_high - current_candle_data["low"]
            else:
                approached_zone = current_candle_data["high"] >= (level_low - near_buffer)
                limited_overshoot = current_candle_data["high"] <= (level_high + pierce_buffer)
                recovered = (
                    current_candle_data["close"] <= (level_low + close_buffer)
                    and current_candle_data["close"] < current_candle_data["open"]
                )
                limited_undershoot = limited_overshoot
                distance = current_candle_data["high"] - level_low

            reacted = approached_zone and limited_undershoot and recovered
            if reacted:
                logging.debug(
                    f"{symbol}: {level_name} zone reaction accepted. "
                    f"ZoneLow={level_low:.4f}, ZoneHigh={level_high:.4f}, "
                    f"Distance={distance:.4f}, NearBuffer={near_buffer:.4f}, "
                    f"PierceBuffer={pierce_buffer:.4f}, CloseBuffer={close_buffer:.4f}"
                )
            return reacted

        def has_impulse_retest_structure():
            prior_candles = today_df.iloc[:-1].copy().reset_index(drop=True)
            if len(prior_candles) < 4:
                return False
            window = prior_candles.tail(IMPULSE_LOOKBACK_BARS).reset_index(drop=True)
            if len(window) < 4:
                return False

            if direction == "long":
                start_pos = int(window["low"].idxmin())
                if start_pos >= len(window) - 1:
                    return False
                tail = window.iloc[start_pos:]
                end_pos = int(tail["high"].idxmax())
                if end_pos < start_pos or end_pos >= len(window):
                    return False
                impulse_low = float(window.iloc[start_pos]["low"])
                impulse_high = float(window.iloc[end_pos]["high"])
                retrace = impulse_high - float(current_candle_data["low"])
            else:
                start_pos = int(window["high"].idxmax())
                if start_pos >= len(window) - 1:
                    return False
                tail = window.iloc[start_pos:]
                end_pos = int(tail["low"].idxmin())
                if end_pos < start_pos or end_pos >= len(window):
                    return False
                impulse_high = float(window.iloc[start_pos]["high"])
                impulse_low = float(window.iloc[end_pos]["low"])
                retrace = float(current_candle_data["high"]) - impulse_low

            impulse_size = impulse_high - impulse_low
            if impulse_size <= 0:
                return False
            impulse_atr_reference = intraday_impulse_atr if intraday_impulse_atr is not None else atr
            if impulse_size < (IMPULSE_MIN_ATR * impulse_atr_reference):
                return False

            bars_in_leg = end_pos - start_pos + 1
            if bars_in_leg > IMPULSE_MAX_BARS:
                return False

            window_start_in_prior = len(prior_candles) - len(window)
            end_pos_global = window_start_in_prior + end_pos
            bars_since_end = len(prior_candles) - 1 - end_pos_global
            if bars_since_end > IMPULSE_RETEST_MAX_BARS:
                return False

            retrace_frac = retrace / impulse_size
            if retrace_frac < IMPULSE_RETRACE_MIN_FRAC or retrace_frac > IMPULSE_RETRACE_MAX_FRAC:
                return False

            logging.debug(
                f"{symbol}: Impulse/retest structure accepted. "
                f"Impulse={impulse_size:.4f}, BarsInLeg={bars_in_leg}, "
                f"BarsSinceImpulse={bars_since_end}, RetraceFrac={retrace_frac:.3f}"
            )
            return True

        def is_compressed_around_levels(level_low, level_high):
            if level_low is None or level_high is None:
                return False
            prior = today_df.iloc[-(IMPULSE_CHOP_LOOKBACK_BARS + 1):-1].copy()
            if prior.empty:
                return False

            band = IMPULSE_LEVEL_BAND_ATR * atr
            near_mask = prior["close"].between(level_low - band, level_high + band)
            near_count = int(near_mask.sum())
            if near_count > IMPULSE_MAX_NEAR_BARS:
                logging.debug(
                    f"{symbol}: Rejecting impulse retest due to compression near VWAP/EOD. "
                    f"NearBars={near_count}, MaxNearBars={IMPULSE_MAX_NEAR_BARS}"
                )
                return True

            states = []
            upper = level_high + band
            lower = level_low - band
            for close_val in prior["close"]:
                if close_val > upper:
                    states.append(1)
                elif close_val < lower:
                    states.append(-1)
                else:
                    states.append(0)

            crosses = 0
            prev_state = 0
            for state in states:
                if state == 0:
                    continue
                if prev_state != 0 and state != prev_state:
                    crosses += 1
                prev_state = state

            if crosses > IMPULSE_MAX_CROSSES:
                logging.debug(
                    f"{symbol}: Rejecting impulse retest due to repeated level crossing. "
                    f"Crosses={crosses}, MaxCrosses={IMPULSE_MAX_CROSSES}"
                )
                return True
            return False

        def build_zone_bounds(zone_key, bars, fallback_low, fallback_high):
            if bars.empty:
                return []
            zone_map = level_series_maps.get(zone_key, {})
            bounds = []
            for _, row in bars.iterrows():
                dt = row.get("datetime")
                if zone_key == "vwap_eod_confluence":
                    low_high = zone_map.get(dt)
                    if low_high is None:
                        low_val, high_val = fallback_low, fallback_high
                    else:
                        low_val, high_val = low_high
                else:
                    value = zone_map.get(dt)
                    if value is None:
                        low_val = fallback_low
                        high_val = fallback_high
                    else:
                        low_val = value
                        high_val = value
                bounds.append((row, low_val, high_val))
            return bounds

        def passes_level_quality_filter(zone_key, level_low, level_high, label):
            min_required_bars = max(BOUNCE_ONE_SIDED_BARS, 3)
            if len(today_df) <= min_required_bars:
                return True

            prior = today_df.iloc[-(BOUNCE_LEVEL_LOOKBACK_BARS + 1):-1].copy()
            if len(prior) < min_required_bars:
                return True

            zone_rows = build_zone_bounds(zone_key, prior, level_low, level_high)
            if not zone_rows:
                return True

            band = BOUNCE_LEVEL_BAND_ATR * atr
            near_count = 0
            crosses = 0
            prev_state = 0
            max_separation = 0.0
            centers = []
            directional_accepts = 0
            acceptance_rows = zone_rows[-BOUNCE_ONE_SIDED_BARS:]

            for idx, (row, row_low, row_high) in enumerate(zone_rows):
                close_val = float(row["close"])
                high_val = float(row["high"])
                low_val = float(row["low"])

                if (row_low - band) <= close_val <= (row_high + band):
                    near_count += 1
                    state = 0
                elif close_val > (row_high + band):
                    state = 1
                else:
                    state = -1

                if prev_state != 0 and state != 0 and state != prev_state:
                    crosses += 1
                if state != 0:
                    prev_state = state

                centers.append((row_low + row_high) / 2.0)

                if direction == "long":
                    max_separation = max(max_separation, (high_val - row_high) / atr)
                else:
                    max_separation = max(max_separation, (row_low - low_val) / atr)

                if idx >= len(zone_rows) - len(acceptance_rows):
                    if direction == "long" and close_val >= (row_high - band):
                        directional_accepts += 1
                    elif direction == "short" and close_val <= (row_low + band):
                        directional_accepts += 1

            one_sided_ratio = directional_accepts / max(1, len(acceptance_rows))
            one_sided_ok = one_sided_ratio >= BOUNCE_ONE_SIDED_RATIO
            separation_ok = max_separation >= BOUNCE_MIN_SEPARATION_ATR

            slope_ok = True
            if len(centers) > BOUNCE_LEVEL_SLOPE_LOOKBACK:
                slope_delta = (centers[-1] - centers[-1 - BOUNCE_LEVEL_SLOPE_LOOKBACK]) / atr
                if direction == "long":
                    slope_ok = slope_delta >= BOUNCE_MIN_LEVEL_SLOPE_ATR
                else:
                    slope_ok = slope_delta <= -BOUNCE_MIN_LEVEL_SLOPE_ATR

            reaction_prior = today_df.iloc[-(BOUNCE_REACTION_LOOKBACK_BARS + 1):-1].copy()
            body_ok = True
            range_ok = True
            if not reaction_prior.empty:
                prior_bodies = (reaction_prior["close"] - reaction_prior["open"]).abs()
                prior_ranges = reaction_prior["high"] - reaction_prior["low"]
                median_body = float(prior_bodies.median()) if not prior_bodies.empty else 0.0
                median_range = float(prior_ranges.median()) if not prior_ranges.empty else 0.0
                current_body = abs(current_candle_data["close"] - current_candle_data["open"])
                current_range = current_candle_data["high"] - current_candle_data["low"]
                if median_body > 0:
                    body_ok = current_body >= (median_body * BOUNCE_MIN_BODY_TO_MEDIAN)
                if median_range > 0:
                    range_ok = current_range >= (median_range * BOUNCE_MIN_RANGE_TO_MEDIAN)

            volume_ok = True
            volume_prior = today_df.iloc[-(BOUNCE_VOLUME_LOOKBACK_BARS + 1):-1].copy()
            if not volume_prior.empty:
                avg_volume = float(volume_prior["volume"].mean())
                if avg_volume > 0:
                    volume_ok = current_candle_data["volume"] >= (avg_volume * BOUNCE_MIN_RELATIVE_VOLUME)

            passes = (
                near_count <= BOUNCE_MAX_NEAR_BARS
                and crosses <= BOUNCE_MAX_CROSSES
                and one_sided_ok
                and separation_ok
                and slope_ok
                and body_ok
                and range_ok
                and volume_ok
            )

            if not passes:
                logging.debug(
                    f"{symbol}: Rejecting {label} bounce due to chop/quality filter. "
                    f"NearBars={near_count}, Crosses={crosses}, Separation={max_separation:.2f}, "
                    f"OneSidedRatio={one_sided_ratio:.2f}, SlopeOK={slope_ok}, "
                    f"BodyOK={body_ok}, RangeOK={range_ok}, VolumeOK={volume_ok}"
                )
            return passes

        # Function to check if a candle respects a level based on direction
        def respects_level(candle, level_value, direction, threshold):
            if direction == "long":
                # For longs, low should be above level (with threshold leeway)
                return candle["low"] >= (level_value - threshold)
            else:
                # For shorts, high should be below level (with threshold leeway)
                return candle["high"] <= (level_value + threshold)
        
        # Function to check consecutive candles respect a level
        def check_consecutive_respect(level_value, level_name):
            if not CHECK_CONSECUTIVE_CANDLES or level_value is None:
                return True  # Skip check if disabled or level is None
                
            if len(today_df) <= CONSECUTIVE_CANDLES:
                return False  # Not enough candles to check
                
            # Get N candles prior to current candle
            prev_n_candles = today_df.iloc[-(CONSECUTIVE_CANDLES+1):-1]
            
            # Check each candle respects the level
            respect_count = sum(1 for _, candle in prev_n_candles.iterrows() 
                            if respects_level(candle, level_value, direction, threshold))
            
            level_respected = respect_count >= CONSECUTIVE_CANDLES
            
            if level_respected:
                logging.debug(f"{symbol}: {level_name} respected for {respect_count}/{CONSECUTIVE_CANDLES} candles")
            else:
                logging.debug(f"{symbol}: {level_name} only respected for {respect_count}/{CONSECUTIVE_CANDLES} candles, skipping")
                
            return level_respected

        # Check for 10-candle bounce if enabled
        if bounce_type_allowed("10_candle") and self.is_bounce_type_enabled("10_candle") and len(df) >= 11:
            if direction == "long":
                # For longs, check if current candle creates a new lowest low
                last_10_candles = df.iloc[-11:-1].copy()  # Exclude current candle
                lowest_low_prev = last_10_candles["low"].min()
                
                # Bounce condition: current candle creates a new low and closes above the open
                if current_candle_data["low"] < lowest_low_prev and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["10_candle_low"] = current_candle_data["low"]
                    mark_trigger("10_candle_low")
                    logging.debug(f"{symbol}: 10-candle LONG bounce candidate found. New low: {current_candle_data['low']:.2f}, Previous lowest: {lowest_low_prev:.2f}")
            else:
                # For shorts, check if current candle creates a new highest high
                last_10_candles = df.iloc[-11:-1].copy()
                highest_high_prev = last_10_candles["high"].max()
                
                # Bounce condition: current candle creates a new high and closes below the open
                if current_candle_data["high"] > highest_high_prev and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["10_candle_high"] = current_candle_data["high"]
                    mark_trigger("10_candle_high")
                    logging.debug(f"{symbol}: 10-candle SHORT bounce candidate found. New high: {current_candle_data['high']:.2f}, Previous highest: {highest_high_prev:.2f}")

        # Check for standard VWAP bounces if enabled
        if bounce_type_allowed("vwap") and self.is_bounce_type_enabled("vwap") and metrics.get("std_vwap") is not None:
            std_vwap = metrics.get("std_vwap")
            # Check if price respected standard VWAP for consecutive candles
            respected = check_consecutive_respect(std_vwap, "Standard VWAP")
            reclaim = is_recent_reclaim(std_vwap, "Standard VWAP")
            zone_reaction = is_vwap_zone_reaction(std_vwap, std_vwap, "Standard VWAP")
            quality_ok = passes_level_quality_filter("vwap", std_vwap, std_vwap, "Standard VWAP")
            touch_reject = is_touch_reject(std_vwap)
            if quality_ok and ((respected and (touch_reject or zone_reaction)) or reclaim):
                ref_levels["vwap"] = std_vwap
                mark_trigger("vwap")
                logging.debug(
                    f"{symbol}: Standard VWAP bounce candidate found ({'reclaim' if reclaim else 'zone reaction' if zone_reaction and not touch_reject else 'touch/reject'}). "
                    f"VWAP: {std_vwap:.2f}, Close: {current_candle_data['close']:.2f}"
                )

        # Check for Dynamic VWAP bounces if enabled
        if bounce_type_allowed("dynamic_vwap") and self.is_bounce_type_enabled("dynamic_vwap") and metrics.get("dynamic_vwap") is not None:
            dynamic_vwap = metrics.get("dynamic_vwap")
            # Check if price respected dynamic VWAP for consecutive candles
            respected = check_consecutive_respect(dynamic_vwap, "Dynamic VWAP")
            reclaim = is_recent_reclaim(dynamic_vwap, "Dynamic VWAP")
            if (respected and is_touch_reject(dynamic_vwap)) or reclaim:
                ref_levels["dynamic_vwap"] = dynamic_vwap
                mark_trigger("dynamic_vwap")
                logging.debug(
                    f"{symbol}: Dynamic VWAP bounce candidate found ({'reclaim' if reclaim else 'touch/reject'}). "
                    f"DVWAP: {dynamic_vwap:.2f}, Close: {current_candle_data['close']:.2f}"
                )

        # Check for EOD VWAP bounces if enabled
        if bounce_type_allowed("eod_vwap") and self.is_bounce_type_enabled("eod_vwap") and metrics.get("eod_vwap") is not None:
            eod_vwap = metrics.get("eod_vwap")
            # Check if price respected EOD VWAP for consecutive candles
            respected = check_consecutive_respect(eod_vwap, "EOD VWAP")
            reclaim = is_recent_reclaim(eod_vwap, "EOD VWAP")
            zone_reaction = is_vwap_zone_reaction(eod_vwap, eod_vwap, "EOD VWAP")
            quality_ok = passes_level_quality_filter("eod_vwap", eod_vwap, eod_vwap, "EOD VWAP")
            touch_reject = is_touch_reject(eod_vwap)
            if quality_ok and ((respected and (touch_reject or zone_reaction)) or reclaim):
                ref_levels["eod_vwap"] = eod_vwap
                mark_trigger("eod_vwap")
                logging.debug(
                    f"{symbol}: EOD VWAP bounce candidate found ({'reclaim' if reclaim else 'zone reaction' if zone_reaction and not touch_reject else 'touch/reject'}). "
                    f"EOD VWAP: {eod_vwap:.2f}, Close: {current_candle_data['close']:.2f}"
                )

        if (
            bounce_type_allowed("vwap_eod_confluence")
            and self.is_bounce_type_enabled("vwap_eod_confluence")
        ):
            std_vwap = metrics.get("std_vwap")
            eod_vwap = metrics.get("eod_vwap")
            if std_vwap is not None and eod_vwap is not None:
                level_spread = abs(std_vwap - eod_vwap)
                max_allowed_spread = CONFLUENCE_MAX_SPREAD_ATR * atr
                if level_spread <= max_allowed_spread:
                    confluence_level = max(std_vwap, eod_vwap) if direction == "long" else min(std_vwap, eod_vwap)
                    touch_reject = is_touch_reject(confluence_level)
                    zone_reaction = is_vwap_zone_reaction(
                        min(std_vwap, eod_vwap),
                        max(std_vwap, eod_vwap),
                        "VWAP+EOD Confluence",
                    )
                    reclaim = is_recent_reclaim(confluence_level, "VWAP+EOD Confluence")
                    quality_ok = passes_level_quality_filter(
                        "vwap_eod_confluence",
                        min(std_vwap, eod_vwap),
                        max(std_vwap, eod_vwap),
                        "VWAP+EOD Confluence",
                    )
                    if quality_ok and (touch_reject or zone_reaction or reclaim):
                        ref_levels["vwap_eod_confluence"] = confluence_level
                        mark_trigger("vwap_eod_confluence")
                        ref_levels.setdefault("vwap", std_vwap)
                        ref_levels.setdefault("eod_vwap", eod_vwap)
                        logging.debug(
                            f"{symbol}: VWAP+EOD confluence bounce candidate found "
                            f"({'reclaim' if reclaim else 'zone reaction' if zone_reaction and not touch_reject else 'touch/reject'}). "
                            f"Spread: {level_spread:.4f} (max {max_allowed_spread:.4f})"
                        )

        if (
            bounce_type_allowed("impulse_retest_vwap_eod")
            and self.is_bounce_type_enabled("impulse_retest_vwap_eod")
        ):
            std_vwap = metrics.get("std_vwap")
            eod_vwap = metrics.get("eod_vwap")
            ema_21 = metrics.get("ema_21")
            if std_vwap is not None and eod_vwap is not None:
                level_low = min(std_vwap, eod_vwap)
                level_high = max(std_vwap, eod_vwap)
                grace = IMPULSE_RETEST_PIERCE_ATR * atr
                close_buffer = IMPULSE_RETEST_CLOSE_BUFFER_ATR * atr
                compression_low = min(level_low, ema_21) if ema_21 is not None else level_low
                compression_high = max(level_high, ema_21) if ema_21 is not None else level_high

                interacted_zone = (
                    current_candle_data["high"] >= (level_low - grace)
                    and current_candle_data["low"] <= (level_high + grace)
                )
                if direction == "long":
                    recovered_side = (
                        current_candle_data["close"] >= (level_high - close_buffer)
                        and current_candle_data["close"] > current_candle_data["open"]
                    )
                else:
                    recovered_side = (
                        current_candle_data["close"] <= (level_low + close_buffer)
                        and current_candle_data["close"] < current_candle_data["open"]
                    )

                if (
                    interacted_zone
                    and recovered_side
                    and has_impulse_retest_structure()
                    and self._impulse_regime_transition_ok(symbol, direction, intraday_rrs_profile)
                ):
                    if not is_compressed_around_levels(compression_low, compression_high):
                        ref_levels["impulse_retest_vwap_eod"] = (level_low + level_high) / 2.0
                        mark_trigger("impulse_retest_vwap_eod")
                        ref_levels.setdefault("vwap", std_vwap)
                        ref_levels.setdefault("eod_vwap", eod_vwap)
                        if ema_21 is not None:
                            ref_levels.setdefault("ema_21", ema_21)
                        logging.debug(
                            f"{symbol}: Impulse retest VWAP/EOD candidate found. "
                            f"ZoneLow={level_low:.4f}, ZoneHigh={level_high:.4f}, Grace={grace:.4f}"
                        )

        # Check EMA bounces (must also be on the correct side of standard VWAP)
        for ema_key, ema_label in (("ema_8", "8 EMA"), ("ema_15", "15 EMA"), ("ema_21", "21 EMA")):
            if not bounce_type_allowed(ema_key) or not self.is_bounce_type_enabled(ema_key):
                continue
            ema_value = metrics.get(ema_key)
            std_vwap = metrics.get("std_vwap")
            if ema_value is None or std_vwap is None:
                continue
            if not check_consecutive_respect(ema_value, ema_label):
                continue

            if direction == "long":
                is_above_vwap = current_candle_data["close"] > std_vwap
                clean_bounce = (
                    abs(current_candle_data["low"] - ema_value) <= threshold
                    and current_candle_data["close"] > current_candle_data["open"]
                    and current_candle_data["close"] > ema_value
                )
                if is_above_vwap and clean_bounce:
                    ref_levels[ema_key] = ema_value
                    mark_trigger(ema_key)
                    logging.debug(
                        f"{symbol}: {ema_label} LONG bounce candidate found. {ema_label}: {ema_value:.2f}, Std VWAP: {std_vwap:.2f}"
                    )
            elif direction == "short":
                is_below_vwap = current_candle_data["close"] < std_vwap
                clean_bounce = (
                    abs(current_candle_data["high"] - ema_value) <= threshold
                    and current_candle_data["close"] < current_candle_data["open"]
                    and current_candle_data["close"] < ema_value
                )
                if is_below_vwap and clean_bounce:
                    ref_levels[ema_key] = ema_value
                    mark_trigger(ema_key)
                    logging.debug(
                        f"{symbol}: {ema_label} SHORT bounce candidate found. {ema_label}: {ema_value:.2f}, Std VWAP: {std_vwap:.2f}"
                    )

        # Check for VWAP upper band bounces for longs
        if bounce_type_allowed("vwap_upper_band") and self.is_bounce_type_enabled("vwap_upper_band") and direction == "long" and metrics.get("vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("vwap_1stdev_upper"), "VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["vwap_upper_band"] = metrics.get("vwap_1stdev_upper")
                    mark_trigger("vwap_upper_band")
                    logging.debug(f"{symbol}: VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for VWAP lower band bounces for shorts
        if bounce_type_allowed("vwap_lower_band") and self.is_bounce_type_enabled("vwap_lower_band") and direction == "short" and metrics.get("vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("vwap_1stdev_lower"), "VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["vwap_lower_band"] = metrics.get("vwap_1stdev_lower")
                    mark_trigger("vwap_lower_band")
                    logging.debug(f"{symbol}: VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for Dynamic VWAP upper band bounces for longs
        if bounce_type_allowed("dynamic_vwap_upper_band") and self.is_bounce_type_enabled("dynamic_vwap_upper_band") and direction == "long" and metrics.get("dynamic_vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("dynamic_vwap_1stdev_upper"), "Dynamic VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("dynamic_vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["dynamic_vwap_upper_band"] = metrics.get("dynamic_vwap_1stdev_upper")
                    mark_trigger("dynamic_vwap_upper_band")
                    logging.debug(f"{symbol}: Dynamic VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('dynamic_vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for Dynamic VWAP lower band bounces for shorts
        if bounce_type_allowed("dynamic_vwap_lower_band") and self.is_bounce_type_enabled("dynamic_vwap_lower_band") and direction == "short" and metrics.get("dynamic_vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("dynamic_vwap_1stdev_lower"), "Dynamic VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("dynamic_vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["dynamic_vwap_lower_band"] = metrics.get("dynamic_vwap_1stdev_lower")
                    mark_trigger("dynamic_vwap_lower_band")
                    logging.debug(f"{symbol}: Dynamic VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('dynamic_vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for EOD VWAP upper band bounces for longs
        if bounce_type_allowed("eod_vwap_upper_band") and self.is_bounce_type_enabled("eod_vwap_upper_band") and direction == "long" and metrics.get("eod_vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("eod_vwap_1stdev_upper"), "EOD VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("eod_vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["eod_vwap_upper_band"] = metrics.get("eod_vwap_1stdev_upper")
                    mark_trigger("eod_vwap_upper_band")
                    logging.debug(f"{symbol}: EOD VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('eod_vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for EOD VWAP lower band bounces for shorts
        if bounce_type_allowed("eod_vwap_lower_band") and self.is_bounce_type_enabled("eod_vwap_lower_band") and direction == "short" and metrics.get("eod_vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("eod_vwap_1stdev_lower"), "EOD VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("eod_vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["eod_vwap_lower_band"] = metrics.get("eod_vwap_1stdev_lower")
                    mark_trigger("eod_vwap_lower_band")
                    logging.debug(f"{symbol}: EOD VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('eod_vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for previous day high/low bounces if enabled
        if direction == "long" and bounce_type_allowed("prev_day_high") and self.is_bounce_type_enabled("prev_day_high") and metrics.get("prev_high") is not None:
            # Check if price respected previous day high for consecutive candles
            if check_consecutive_respect(metrics.get("prev_high"), "Previous Day High"):
                # Only consider bounce if price respected the level all day
                if abs(current_candle_data["low"] - metrics.get("prev_high")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["prev_day_high"] = metrics.get("prev_high")
                    mark_trigger("prev_day_high")
                    logging.debug(f"{symbol}: Previous Day High LONG bounce candidate found. Prev High: {metrics.get('prev_high'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        elif direction == "short" and bounce_type_allowed("prev_day_low") and self.is_bounce_type_enabled("prev_day_low") and metrics.get("prev_low") is not None:
            # Check if price respected previous day low for consecutive candles
            if check_consecutive_respect(metrics.get("prev_low"), "Previous Day Low"):
                # Only consider bounce if price respected the level all day
                if abs(current_candle_data["high"] - metrics.get("prev_low")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["prev_day_low"] = metrics.get("prev_low")
                    mark_trigger("prev_day_low")
                    logging.debug(f"{symbol}: Previous Day Low SHORT bounce candidate found. Prev Low: {metrics.get('prev_low'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Return None if no reference levels were found, otherwise return the details
        return {
            "levels": ref_levels,
            "triggered_levels": triggered_levels,
            "candle": current_candle_data
        } if ref_levels else None



    def request_and_detect_bounce(self, symbol, allowed_bounce_types=None):
        # Only scan within market hours (if enabled)
        if not SCAN_OUTSIDE_MARKET_HOURS:
            current_time = datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
            market_open = current_time.replace(hour=6, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=13, minute=0, second=0, microsecond=0)
            if not (market_open <= current_time <= market_close):
                logging.debug(f"{symbol}: Not within market hours for bounce detection.")
                return

        # Request 5 days of data to ensure we get enough market days
        five_day_reqId = self.getReqId()
        self.data[five_day_reqId] = []
        self.data_ready_events[five_day_reqId] = threading.Event()
        contract = self.create_stock_contract(symbol)
        
        # Request historical data from IB
        self.reqHistoricalData(
            reqId=five_day_reqId,
            contract=contract,
            endDateTime="",  # up to now
            durationStr="5 D",  # Increased to 5 days to account for weekends/holidays
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        
        # Wait for data with timeout
        if not self.data_ready_events[five_day_reqId].wait(timeout=15):
            logging.warning(f"{symbol}: Timeout waiting for historical data.")
            del self.data_ready_events[five_day_reqId]
            return
        
        all_bars = self.data.get(five_day_reqId, [])
        if len(all_bars) < 10:
            logging.warning(f"{symbol}: Insufficient historical data, only {len(all_bars)} bars received")
            del self.data_ready_events[five_day_reqId]
            return
        
        # Clean up
        del self.data_ready_events[five_day_reqId]

        # Cache 5-minute bars for RRS reuse
        self.latest_bars[symbol] = _dedupe_bars(_bars_to_ib(all_bars))
        
        # Convert to DataFrame
        df = pd.DataFrame(all_bars)
        
        # Add datetime column for date filtering
        try:
            df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
            if df["datetime"].isna().all():
                logging.error(f"{symbol}: Failed to convert any timestamps to datetime objects")
                return
        except Exception as e:
            logging.error(f"{symbol}: Error in datetime conversion: {e}")
            return
        
        # Get unique dates in the data, sorted in ascending order
        unique_dates = sorted(df["datetime"].dt.date.unique())
        logging.debug(f"{symbol}: Found data for these dates: {unique_dates}")
        
        if len(unique_dates) < 2:
            logging.warning(f"{symbol}: Need at least 2 days of data, but only found {len(unique_dates)} days")
        
        # The last date is today, the second-to-last date is the previous trading day
        current_date = unique_dates[-1] if unique_dates else None
        previous_date = unique_dates[-2] if len(unique_dates) >= 2 else None
        
        logging.debug(f"{symbol}: Current date = {current_date}, Previous trading date = {previous_date}")
        
        # Create separate dataframes for different time periods
        today_df = df[df["datetime"].dt.date == current_date].copy() if current_date else pd.DataFrame()
        prev_day_df = df[df["datetime"].dt.date == previous_date].copy() if previous_date else pd.DataFrame()
        
        logging.debug(f"{symbol}: Found {len(today_df)} bars for today and {len(prev_day_df)} bars for previous trading day")
        
        if today_df.empty:
            logging.warning(f"{symbol}: No data for today")
            return
        
        # 1. Calculate Standard VWAP (today only)
        standard_vwap = self.calculate_standard_vwap(df)
        if standard_vwap is not None:
            logging.debug(
                f"{symbol}: Standard VWAP calculated from {len(today_df)} today's bars = {standard_vwap:.4f}"
            )

        # 2. Calculate Dynamic VWAP (previous trading day + today)
        dynamic_vwap = self.calculate_dynamic_vwap(df)
        if dynamic_vwap is not None and not prev_day_df.empty:
            logging.debug(
                f"{symbol}: Dynamic VWAP calculated from {len(prev_day_df) + len(today_df)} bars (previous day + today) = {dynamic_vwap:.4f}"
            )
        elif dynamic_vwap is not None:
            logging.debug(
                f"{symbol}: Dynamic VWAP calculated from today's bars only = {dynamic_vwap:.4f}"
            )

        # 3. Calculate EOD VWAP (previous trading day's last candle + today)
        eod_vwap = self.calculate_eod_vwap(df)
        if eod_vwap is not None and not prev_day_df.empty:
            logging.debug(
                f"{symbol}: EOD VWAP calculated from {len(today_df) + 1} bars (prev day's last + today) = {eod_vwap:.4f}"
            )
        elif eod_vwap is not None:
            logging.debug(
                f"{symbol}: EOD VWAP calculated from today's bars only = {eod_vwap:.4f}"
            )
        
        # 4. Get previous day extremes
        prev_high = prev_day_df["high"].max() if not prev_day_df.empty else None
        prev_low = prev_day_df["low"].min() if not prev_day_df.empty else None
        
        logging.debug(f"{symbol}: Previous day high = {prev_high}, low = {prev_low}")

        # 5. Calculate short EMAs (today only)
        ema_8 = None
        ema_15 = None
        ema_21 = None
        if not today_df.empty:
            if len(today_df) >= 8:
                ema_8 = today_df["close"].ewm(span=8, adjust=False).mean().iloc[-1]
            if len(today_df) >= 15:
                ema_15 = today_df["close"].ewm(span=15, adjust=False).mean().iloc[-1]
            if len(today_df) >= 21:
                ema_21 = today_df["close"].ewm(span=21, adjust=False).mean().iloc[-1]

        # 6. Get current price
        current_price = today_df["close"].iloc[-1] if not today_df.empty else None
        direction = "long" if symbol in self.longs else "short"
        
        # Calculate standard VWAP with bands
        vwap_value, vwap_upper_band, vwap_lower_band = self.calculate_vwap_with_stdev_bands(today_df)
        logging.debug(f"{symbol}: VWAP: {vwap_value:.4f}, Upper 1SD: {vwap_upper_band:.4f}, Lower 1SD: {vwap_lower_band:.4f}")

        # Calculate dynamic VWAP with bands
        dynamic_vwap, dynamic_upper_band, dynamic_lower_band = self.calculate_dynamic_vwap_with_stdev_bands(df)
        
        # Fix the logging to handle None values
        dynamic_vwap_str = f"{dynamic_vwap:.4f}" if dynamic_vwap is not None else "None"
        dynamic_upper_str = f"{dynamic_upper_band:.4f}" if dynamic_upper_band is not None else "None"
        dynamic_lower_str = f"{dynamic_lower_band:.4f}" if dynamic_lower_band is not None else "None"
        logging.debug(f"{symbol}: Dynamic VWAP: {dynamic_vwap_str}, Upper 1SD: {dynamic_upper_str}, Lower 1SD: {dynamic_lower_str}")

        # Calculate EOD VWAP with bands
        eod_vwap, eod_upper_band, eod_lower_band = self.calculate_eod_vwap_with_stdev_bands(df)
        
        # Fix the logging to handle None values
        eod_vwap_str = f"{eod_vwap:.4f}" if eod_vwap is not None else "None"
        eod_upper_str = f"{eod_upper_band:.4f}" if eod_upper_band is not None else "None"
        eod_lower_str = f"{eod_lower_band:.4f}" if eod_lower_band is not None else "None"
        logging.debug(f"{symbol}: EOD VWAP: {eod_vwap_str}, Upper 1SD: {eod_upper_str}, Lower 1SD: {eod_lower_str}")

        vwap_invalidation = self._build_vwap_invalidation_snapshot(df, direction)

        # Store all metrics in one comprehensive dictionary
        self.symbol_metrics[symbol] = {
            "std_vwap": standard_vwap,
            "dynamic_vwap": dynamic_vwap,
            "eod_vwap": eod_vwap,
            "prev_high": prev_high,
            "prev_low": prev_low,
            "price": current_price,
            "vwap_1stdev_upper": vwap_upper_band,
            "vwap_1stdev_lower": vwap_lower_band,
            "dynamic_vwap_1stdev_upper": dynamic_upper_band,
            "dynamic_vwap_1stdev_lower": dynamic_lower_band,
            "eod_vwap_1stdev_upper": eod_upper_band,
            "eod_vwap_1stdev_lower": eod_lower_band,
            "ema_8": ema_8,
            "ema_15": ema_15,
            "ema_21": ema_21,
            "vwap_invalidation": vwap_invalidation,
        }

        # Then continue with detailed logging if LOGGING_MODE is enabled
        if LOGGING_MODE:
            msg = f"{symbol} Metrics -> "
            msg += f"Price: {current_price:.2f}, " if current_price is not None else "Price: N/A, "
            msg += f"Std VWAP: {standard_vwap:.4f}, " if standard_vwap is not None else "Std VWAP: N/A, "
            msg += f"Dynamic VWAP: {dynamic_vwap:.4f}, " if dynamic_vwap is not None else "Dynamic VWAP: N/A, "
            msg += f"EOD VWAP: {eod_vwap:.4f}, " if eod_vwap is not None else "EOD VWAP: N/A, "
            msg += f"VWAP 1SD Upper: {vwap_upper_band:.4f}, " if vwap_upper_band is not None else "VWAP 1SD Upper: N/A, "
            msg += f"VWAP 1SD Lower: {vwap_lower_band:.4f}, " if vwap_lower_band is not None else "VWAP 1SD Lower: N/A, "
            msg += f"Dynamic VWAP 1SD Upper: {dynamic_upper_band:.4f}, " if dynamic_upper_band is not None else "Dynamic VWAP 1SD Upper: N/A, "
            msg += f"Dynamic VWAP 1SD Lower: {dynamic_lower_band:.4f}, " if dynamic_lower_band is not None else "Dynamic VWAP 1SD Lower: N/A, "
            msg += f"EOD VWAP 1SD Upper: {eod_upper_band:.4f}, " if eod_upper_band is not None else "EOD VWAP 1SD Upper: N/A, "
            msg += f"EOD VWAP 1SD Lower: {eod_lower_band:.4f}, " if eod_lower_band is not None else "EOD VWAP 1SD Lower: N/A, "
            msg += f"21 EMA: {ema_21:.4f}, " if ema_21 is not None else "21 EMA: N/A, "
            
            if symbol in self.longs:
                msg += f"Prev Day High: {prev_high:.4f}, " if prev_high is not None else "Prev Day High: N/A, "
            elif symbol in self.shorts:
                msg += f"Prev Day Low: {prev_low:.4f}, " if prev_low is not None else "Prev Day Low: N/A, "
            
            atr = self.atr_cache.get(symbol, None)
            msg += f"ATR: {atr:.4f}" if atr is not None else "ATR: N/A"
            self.log_symbol(symbol, msg)


        # Continue with evaluating bounce candidates
        candidate_info = self.evaluate_bounce_candidate(symbol, df, allowed_bounce_types=allowed_bounce_types)

        
        # STEP 1: First check if we have an existing bounce candidate to confirm
        if symbol in self.bounce_candidates:
            bounce_data = self.bounce_candidates[symbol]
            bounce_candle = bounce_data["bounce_candle"]
        
            # Convert time strings to datetime objects for comparison
            try:
                bounce_time = pd.to_datetime(bounce_candle["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
                current_time = pd.to_datetime(today_df.iloc[-1]["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
            
                # Check if this is a new candle (different from the bounce candidate candle)
                if current_time > bounce_time:
                    logging.debug(f"{symbol}: Checking confirmation - Bounce time: {bounce_time}, Current time: {current_time}")
                    current_candle = today_df.iloc[-1]
                    bounce_data["candles_waited"] += 1
                    candles_waited = bounce_data["candles_waited"]

                    if direction == "long" and current_candle["close"] < bounce_candle["low"]:
                        logging.debug(
                            f"{symbol}: Invalidating long bounce candidate - close {current_candle['close']:.2f} below bounce low {bounce_candle['low']:.2f}"
                        )
                        self.bounce_candidates.pop(symbol)
                        return
                    if direction == "short" and current_candle["close"] > bounce_candle["high"]:
                        logging.debug(
                            f"{symbol}: Invalidating short bounce candidate - close {current_candle['close']:.2f} above bounce high {bounce_candle['high']:.2f}"
                        )
                        self.bounce_candidates.pop(symbol)
                        return
                    
                    # Confirm only after a later candle closes through the bounce candle extreme.
                    if direction == "long" and current_candle["close"] > bounce_candle["high"]:
                        levels = bounce_data["levels"]
                        levels_list = bounce_data.get("triggered_levels") or list(levels.keys())
                        bounce_msg = f"{symbol}: Bounce confirmed (long) from {levels_list}"
                        if self.gui_callback:
                            self.gui_callback(bounce_msg, "green")
                        self._emit_master_avwap_focus_bounce_alert(symbol, "long", levels_list)
                        self.log_symbol(symbol, f"ALERT: {bounce_msg}")
                        self.log_bounce_to_file(
                            symbol=symbol,
                            direction="long",
                            levels=levels,
                            bounce_candle=bounce_candle,
                            current_candle=current_candle,
                            threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0)
                        )
                        self.alerted_symbols.add(symbol)
                        self.bounce_candidates.pop(symbol)
                        return  # Exit after confirming a bounce
                
                    elif direction == "short" and current_candle["close"] < bounce_candle["low"]:
                        levels = bounce_data["levels"]
                        levels_list = bounce_data.get("triggered_levels") or list(levels.keys())
                        bounce_msg = f"{symbol}: Bounce confirmed (short) from {levels_list}"
                        if self.gui_callback:
                            self.gui_callback(bounce_msg, "red")
                        self._emit_master_avwap_focus_bounce_alert(symbol, "short", levels_list)
                        self.log_symbol(symbol, f"ALERT: {bounce_msg}")
                        self.log_bounce_to_file(
                            symbol=symbol,
                            direction="short",
                            levels=levels,
                            bounce_candle=bounce_candle,
                            current_candle=current_candle,
                            threshold=THRESHOLD_MULTIPLIER * self.atr_cache.get(symbol, 0)
                        )
                        self.alerted_symbols.add(symbol)
                        self.bounce_candidates.pop(symbol)
                        return  # Exit after confirming a bounce
                    else:
                        if candles_waited >= BOUNCE_CONFIRMATION_MAX_CANDLES:
                            logging.debug(
                                f"{symbol}: Removing bounce candidate after {candles_waited} candles without confirmation"
                            )
                            self.bounce_candidates.pop(symbol)
                            return
                        # Remove stale candidates after a certain time period (e.g., 4 hours)
                        detection_time = bounce_data["detection_time"]
                        if (datetime.now() - detection_time).total_seconds() > 14400:  # 4 hours in seconds
                            logging.debug(f"{symbol}: Removing stale bounce candidate detected at {detection_time}")
                            self.bounce_candidates.pop(symbol)
                else:
                    logging.debug(f"{symbol}: Current candle is the same as bounce candle, waiting for next candle")
            except Exception as e:
                logging.error(f"{symbol}: Error during bounce confirmation: {e}")
                self.bounce_candidates.pop(symbol, None)  # Remove problematic candidate

        # STEP 2: Check if current candle is a new bounce candidate - FIXED INDENTATION
        if candidate_info and symbol not in self.bounce_candidates:
            self.bounce_candidates[symbol] = {
                "levels": candidate_info["levels"],
                "triggered_levels": candidate_info.get("triggered_levels", []),
                "bounce_candle": candidate_info["candle"],
                "detection_time": datetime.now(),
                "candles_waited": 0
            }

                        # In the evaluate_bounce_candidate function, where price approaching is logged:
            if LOG_PRICE_APPROACHING:
                # Filter out 10-candle levels for approaching alerts
                display_level_names = set(candidate_info.get("triggered_levels", candidate_info["levels"].keys()))
                approaching_levels = {k: v for k, v in candidate_info["levels"].items()
                                    if k in display_level_names
                                    if "10_candle" not in k}
                
                # Only show approaching alert if there are non-10-candle levels
                if approaching_levels:
                    level_details = ", ".join(f"{lvl}: {val:.2f}" for lvl, val in approaching_levels.items())
                    approaching_msg = f"{symbol}: Price approaching levels ({direction}) - {level_details}"
                    
                    # Use self.log_symbol to ensure proper symbol coloring
                    self.log_symbol(symbol, approaching_msg, "approaching")
                    
                    if self.gui_callback:
                        direction_tag = "approaching_green" if direction == "long" else "approaching_red"
                        self.gui_callback(approaching_msg, direction_tag)


    def check_removal_conditions(self):
        if not self.has_minimum_candles_completed(required=VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES):
            logging.info(
                f"Skipping removal conditions until {VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES} completed 5-minute candles."
            )
            return
        for watchlist_path, direction in [(LONGS_FILENAME, "long"), (SHORTS_FILENAME, "short")]:
            tickers = read_tickers(watchlist_path)
            for symbol in tickers:
                if symbol not in self.symbol_metrics:
                    continue

                metrics = self.symbol_metrics[symbol]
                invalidation = metrics.get("vwap_invalidation") or {}
                consecutive_bad_closes = int(invalidation.get("consecutive_bad_closes") or 0)
                if consecutive_bad_closes < VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES:
                    continue

                last_row = invalidation.get("last_row") or {}
                close_value = last_row.get("close")
                vwap_value = last_row.get("vwap")
                dynamic_vwap = last_row.get("dynamic_vwap")
                eod_vwap = last_row.get("eod_vwap")
                level_side = "below" if direction == "long" else "above"

                self.remove_from_watchlist(symbol, direction)
                removal_msg = (
                    f"{symbol} removed from {direction}s watchlist - "
                    f"{consecutive_bad_closes} completed M5 closes {level_side} VWAP/Dynamic VWAP/EOD VWAP "
                    f"(close={close_value:.2f}, vwap={vwap_value:.2f}, dvwap={dynamic_vwap:.2f}, eod={eod_vwap:.2f})"
                )
                logging.info(removal_msg)
                if self.gui_callback:
                    self.gui_callback(removal_msg, "blue")

    def remove_from_watchlist(self, symbol, direction):
        filename = LONGS_FILENAME if direction == "long" else SHORTS_FILENAME
        try:
            with open(filename, 'r') as f:
                symbols = f.read().splitlines()
            symbols = [s for s in symbols if s.strip() != symbol]
            with open(filename, 'w') as f:
                f.write('\n'.join(symbols))
            if direction == "long":
                self.longs = symbols
            else:
                self.shorts = symbols
            self.bounce_candidates.pop(symbol, None)
            self.symbol_metrics.pop(symbol, None)
            self.latest_bars.pop(symbol, None)
            stale_bar_keys = [key for key in self.latest_bars if str(key).startswith(f"{symbol}|")]
            for key in stale_bar_keys:
                self.latest_bars.pop(key, None)
            logging.info(f"{symbol} removed from {filename} due to removal condition.")
        except Exception as e:
            logging.error(f"Error removing {symbol} from {filename}: {e}")

    def run_strategy(self):
        last_warning_reset = datetime.now().date()
        
        while True:
            try:
                if not self.is_scanning_enabled():
                    time.sleep(0.5)
                    continue

                if not self.ensure_connected():
                    logging.warning("IB not connected; retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                # Reset warning cache daily
                current_date = datetime.now().date()
                if current_date != last_warning_reset:
                    self.warned_symbols.clear()
                    self.emitted_master_avwap_events.clear()
                    self.emitted_master_avwap_focus_alerts.clear()
                    last_warning_reset = current_date
                    logging.info("Daily warning cache reset completed")
                
                self.longs = read_tickers(LONGS_FILENAME)
                self.shorts = read_tickers(SHORTS_FILENAME)
                self.load_master_avwap_focus()
                self.update_watchlists_from_master_avwap()
                self.alerted_symbols.clear()
                self.symbol_metrics = {}
                self.latest_bars = {}
                self.build_atr_cache()

                # Log strongest/weakest names for key intraday timeframes each cycle.
                for timeframe_key in ("5m", "15m", "1h"):
                    self.run_rrs_scan(timeframe_key_override=timeframe_key, emit_gui=False)
                # Keep the GUI view synced with user-selected RRS timeframe.
                self.run_rrs_scan()

                monitored_symbols = self.get_monitored_extreme_symbols()
                logging.info(f"Monitoring {len(monitored_symbols)} strongest/weakest symbols for EMA bounces.")
                all_symbols = set(self.longs + self.shorts)
                processed_symbols = set()
                enabled_bounce_types = {
                    bounce_type for bounce_type, enabled in self.bounce_type_toggles.items() if enabled
                }
                non_ema_extreme_bounce_types = enabled_bounce_types - {"ema_8", "ema_15"}

                # 1) Prioritize strongest/weakest names first.
                for sym in sorted(monitored_symbols):
                    if not self.is_scanning_enabled():
                        break
                    if sym not in all_symbols or self.atr_cache.get(sym) is None:
                        continue
                    self.request_and_detect_bounce(sym, allowed_bounce_types=enabled_bounce_types)
                    processed_symbols.add(sym)

                # 2) Then scan all remaining symbols for non-EMA-8/15 bounce types.
                for sym in sorted(all_symbols - processed_symbols):
                    if not self.is_scanning_enabled():
                        break
                    if self.atr_cache.get(sym) is None:
                        continue
                    self.request_and_detect_bounce(sym, allowed_bounce_types=non_ema_extreme_bounce_types)

                if not self.is_scanning_enabled():
                    continue

                self.check_removal_conditions()
                wait_for_candle_close()
                if self.gui_callback:
                    self.gui_callback("Candle has closed", "candle_line")
            except Exception as e:
                logging.exception(f"Error in strategy loop: {e}")
                time.sleep(5)


    def check_dynamic_vwap_touches(self):
        results = []
        all_symbols = set(self.longs + self.shorts)
        for symbol in all_symbols:
            reqId = self.getReqId()
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
            contract = self.create_stock_contract(symbol)
            self.reqHistoricalData(
                reqId=reqId,
                contract=contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            if self.data_ready_events[reqId].wait(timeout=15):
                bars = self.data.get(reqId, [])
                df = pd.DataFrame(bars)
                if not df.empty:
                    dvwap = self.calculate_dynamic_vwap(df)
                    if dvwap is not None:
                        touched = any(row['low'] <= dvwap <= row['high'] for _, row in df.iterrows())
                        if touched:
                            results.append(f"{symbol} touched DVWAP")
            del self.data_ready_events[reqId]
        return results    

    def check_dynamic_vwap2_touches(self):
        results = []
        all_symbols = set(self.longs + self.shorts)
        for symbol in all_symbols:
            reqId = self.getReqId()
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
            contract = self.create_stock_contract(symbol)
            self.reqHistoricalData(
                reqId=reqId,
                contract=contract,
                endDateTime="",
                durationStr="3 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            if self.data_ready_events[reqId].wait(timeout=15):
                bars = self.data.get(reqId, [])
                df = pd.DataFrame(bars)
                if not df.empty:
                    dvwap2 = self.calculate_dynamic_vwap2(df)
                    if dvwap2 is not None:
                        df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
                        today_data = df[df["datetime"].dt.date == df["datetime"].iloc[-1].date()]
                        touched = any(row['low'] <= dvwap2 <= row['high'] for _, row in today_data.iterrows())
                        if touched:
                            results.append(f"{symbol} touched DVWAP2")
            del self.data_ready_events[reqId]
        return results
    
    def check_eod_vwap_touches(self):
        results = []
        all_symbols = set(self.longs + self.shorts)
        for symbol in all_symbols:
            reqId = self.getReqId()
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
            contract = self.create_stock_contract(symbol)
            self.reqHistoricalData(
                reqId=reqId,
                contract=contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
            if self.data_ready_events[reqId].wait(timeout=15):
                bars = self.data.get(reqId, [])
                df = pd.DataFrame(bars)
                if not df.empty:
                    eod_vwap = self.calculate_eod_vwap(df)
                    if eod_vwap is not None:
                        df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d  %H:%M:%S", errors="coerce")
                        today_data = df[df["datetime"].dt.date == df["datetime"].iloc[-1].date()]
                        touched = any(row['low'] <= eod_vwap <= row['high'] for _, row in today_data.iterrows())
                        if touched:
                            results.append(f"{symbol} touched EOD VWAP")
            del self.data_ready_events[reqId]
        return results
    
    def log_bounce_to_file(self, symbol, direction, levels, bounce_candle, current_candle, threshold):
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            bounce_types_list = list(levels.keys())
            bounce_types_str = ", ".join(bounce_types_list)

            candle_time = str(current_candle.get("time", "")).strip() if current_candle is not None else ""
            trade_dt = None
            if candle_time:
                for fmt in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S"):
                    try:
                        trade_dt = datetime.strptime(candle_time, fmt)
                        break
                    except ValueError:
                        continue
            if trade_dt is None:
                trade_dt = datetime.now()
            trade_date_str = trade_dt.strftime("%Y-%m-%d")

            BOUNCE_LOG_FILENAME.parent.mkdir(parents=True, exist_ok=True)
            with open(BOUNCE_LOG_FILENAME, "a") as f:
                f.write(f"{timestamp} | {symbol} | {bounce_types_str} | {direction}\n")

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            file_exists = INTRADAY_BOUNCES_CSV.exists()
            with INTRADAY_BOUNCES_CSV.open("a", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=["time_local", "trade_date", "symbol", "direction", "bounce_types"],
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        "time_local": timestamp,
                        "trade_date": trade_date_str,
                        "symbol": symbol,
                        "direction": direction,
                        "bounce_types": ", ".join(bounce_types_list),
                    }
                )

            logging.info(
                f"Simplified bounce details for {symbol} logged to {BOUNCE_LOG_FILENAME}"
            )
        except Exception as e:
            logging.error(f"Error logging bounce to file: {e}")



##########################################
# Run Bot with GUI Integration
##########################################
def run_bot_with_gui(gui_callback):
    configure_app_logging()

    bot = BounceBot(gui_callback=gui_callback)
    bot.set_connection_info("127.0.0.1", 7496, 125)
    try:
        bot.connect("127.0.0.1", 7496, clientId=bot.ib_client_id)
    except Exception as exc:
        logging.warning(f"Initial IB connection attempt failed: {exc}")
    api_thread = threading.Thread(target=bot.run, daemon=True)
    bot.api_thread = api_thread
    api_thread.start()
    logging.info("Starting strategy loop (connection will auto-retry until IB is available)...")
    strategy_thread = threading.Thread(target=bot.run_strategy, daemon=True)
    strategy_thread.start()
    return bot

##########################################
# GUI Code using Tkinter
##########################################
# Find and replace the light_grey variable definition with dark theme colors
# Around line 677 in the start_gui() function

def build_environment_focus_copy_text(snapshot):
    sections = snapshot.get("environment_highlights", []) if isinstance(snapshot, dict) else []
    label = "Environment Focus Lists"
    if isinstance(snapshot, dict):
        env_label = snapshot.get("market_environment_label", "Environment")
        label = f"{env_label} Focus Lists"

    lines = [label]
    if not sections:
        lines.extend(["", "None"])
        return "\n".join(lines)

    for section in sections:
        title = str(section.get("title", "Section")).strip() or "Section"
        seen = set()
        symbols = []
        for row in section.get("rows", []):
            symbol = str(row.get("symbol") or "").strip().upper()
            if not symbol:
                text = str(row.get("text") or "").strip()
                if text:
                    first_token = text.split()[0].strip(",")
                    if ENVIRONMENT_FOCUS_SYMBOL_RE.fullmatch(first_token):
                        symbol = first_token
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)

        lines.append("")
        lines.append(title)
        lines.append(", ".join(symbols) if symbols else "None")

    return "\n".join(lines).strip()


def copy_text_to_clipboard(widget: tk.Misc, text: str) -> None:
    widget.clipboard_clear()
    widget.clipboard_append(text)
    widget.update_idletasks()


def create_rrs_confirmed_panel(parent, bot_instance, dark_grey="#2E2E2E", text_color="#E0E0E0"):
    """Create the BounceBot RS/RW confirmed screen (industry + sector) inside `parent`."""
    rrs_container = tk.Frame(parent, bg=dark_grey)

    rrs_controls = tk.Frame(rrs_container, padx=10, pady=10, bg=dark_grey)
    rrs_controls.pack(fill=tk.X)

    rrs_status_var = tk.StringVar(value="RRS ready")
    tk.Label(rrs_controls, textvariable=rrs_status_var, fg=text_color, bg=dark_grey).pack(side=tk.LEFT, padx=(0, 10))

    rrs_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)

    def on_rrs_threshold_change(*_):
        bot_instance.set_rrs_threshold(rrs_threshold_var.get())

    rrs_threshold_var.trace_add("write", on_rrs_threshold_change)

    tk.Scale(
        rrs_controls,
        from_=0.0,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        label="RRS Sensitivity",
        variable=rrs_threshold_var,
        length=220,
        bg=dark_grey,
        fg=text_color,
        highlightthickness=0,
    ).pack(side=tk.LEFT, padx=(0, 10))

    timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)

    def on_timeframe_change() -> None:
        bot_instance.set_rrs_timeframe(timeframe_var.get())

    for key in ("5m", "15m", "30m", "1h"):
        label = RRS_TIMEFRAMES[key]["label"]
        tk.Radiobutton(
            rrs_controls,
            text=label,
            variable=timeframe_var,
            value=key,
            indicatoron=0,
            command=on_timeframe_change,
            padx=6,
            pady=2,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        ).pack(side=tk.LEFT, padx=2)

    env_selection_var = tk.StringVar(value=bot_instance.get_market_environment())
    env_label_var = tk.StringVar(
        value=f"Environment: {MARKET_ENVIRONMENTS.get(env_selection_var.get(), {}).get('label', env_selection_var.get())}"
    )

    def on_environment_change():
        selected = env_selection_var.get()
        env_label_var.set(
            f"Environment: {MARKET_ENVIRONMENTS.get(selected, {}).get('label', selected)}"
        )
        bot_instance.set_market_environment(selected)

    env_mode_frame = tk.Frame(rrs_container, bg=dark_grey, pady=4)
    env_mode_frame.pack(fill=tk.X, padx=10)
    tk.Label(env_mode_frame, textvariable=env_label_var, fg=text_color, bg=dark_grey).pack(side=tk.LEFT)
    env_button_frame = tk.Frame(env_mode_frame, bg=dark_grey)
    env_button_frame.pack(side=tk.RIGHT)
    for key, info in MARKET_ENVIRONMENTS.items():
        tk.Radiobutton(
            env_button_frame,
            text=info["label"],
            variable=env_selection_var,
            value=key,
            indicatoron=0,
            command=on_environment_change,
            padx=6,
            pady=2,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        ).pack(side=tk.LEFT, padx=2)

    rrs_frame = tk.Frame(rrs_container, padx=10, pady=10, bg=dark_grey)
    rrs_frame.pack(fill=tk.BOTH, expand=True)

    vertical_pane = tk.PanedWindow(
        rrs_frame,
        orient=tk.VERTICAL,
        sashrelief=tk.RAISED,
        sashwidth=10,
        showhandle=True,
        bg=dark_grey,
    )
    vertical_pane.pack(fill=tk.BOTH, expand=True)

    env_focus_pane = tk.PanedWindow(
        vertical_pane,
        orient=tk.HORIZONTAL,
        sashrelief=tk.RAISED,
        sashwidth=10,
        showhandle=True,
        bg=dark_grey,
    )
    vertical_pane.add(env_focus_pane, minsize=150)

    env_focus_frame = tk.LabelFrame(env_focus_pane, text="Environment Focus", bg=dark_grey, fg=text_color)
    env_focus_text = scrolledtext.ScrolledText(
        env_focus_frame,
        wrap=tk.NONE,
        width=80,
        height=10,
        font=('Courier', 10),
        state='disabled',
        bg=dark_grey,
        fg=text_color,
    )
    env_focus_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    env_focus_text.tag_config("rrs_hdr", foreground="#BD93F9", font=('Courier', 11, 'bold'))
    env_focus_text.tag_config("rrs_rs", foreground="#50FA7B", font=('Courier', 11, 'bold'))
    env_focus_text.tag_config("rrs_rw", foreground="#FF5555", font=('Courier', 11, 'bold'))
    env_focus_pane.add(env_focus_frame, stretch="always")

    env_copy_frame = tk.LabelFrame(env_focus_pane, text="Environment Focus Lists", bg=dark_grey, fg=text_color)
    env_copy_toolbar = tk.Frame(env_copy_frame, bg=dark_grey)
    env_copy_toolbar.pack(fill=tk.X, padx=4, pady=(4, 0))

    env_copy_text = scrolledtext.ScrolledText(
        env_copy_frame,
        wrap=tk.WORD,
        width=52,
        height=10,
        font=('Courier', 10),
        state='disabled',
        bg=dark_grey,
        fg=text_color,
    )

    def copy_env_focus_lists():
        text = env_copy_text.get("1.0", tk.END).strip()
        if not text:
            rrs_status_var.set("Environment focus lists: nothing to copy.")
            return
        copy_text_to_clipboard(env_copy_text, text)
        rrs_status_var.set("Copied environment focus lists to clipboard.")

    tk.Button(
        env_copy_toolbar,
        text="Copy",
        command=copy_env_focus_lists,
        relief=tk.RAISED,
        padx=10,
        bg=dark_grey,
        fg=text_color,
    ).pack(side=tk.LEFT)

    env_copy_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    env_focus_pane.add(env_copy_frame, stretch="always")

    notebook_host = tk.Frame(vertical_pane, bg=dark_grey)
    vertical_pane.add(notebook_host, stretch="always")

    notebook = ttk.Notebook(notebook_host)
    notebook.pack(fill=tk.BOTH, expand=True)

    def _make_text(parent_widget, width, height):
        widget = scrolledtext.ScrolledText(
            parent_widget,
            wrap=tk.NONE,
            width=width,
            height=height,
            font=("Courier", 10),
            state="disabled",
            bg=dark_grey,
            fg=text_color,
        )
        widget.tag_config("rrs_hdr", foreground="#BD93F9", font=("Courier", 11, "bold"))
        widget.tag_config("rrs_rs", foreground="#50FA7B", font=("Courier", 11, "bold"))
        widget.tag_config("rrs_rw", foreground="#FF5555", font=("Courier", 11, "bold"))
        return widget

    def _create_split_signal_tab(parent_widget, title, threshold_var=None, timeframe_var=None):
        tab = ttk.Frame(parent_widget)
        parent_widget.add(tab, text=title)

        if threshold_var is not None or timeframe_var is not None:
            controls = tk.Frame(tab, bg=dark_grey, padx=6, pady=6)
            controls.pack(fill=tk.X)
            if threshold_var is not None:
                tk.Scale(
                    controls,
                    from_=0.0,
                    to=5.0,
                    resolution=0.1,
                    orient=tk.HORIZONTAL,
                    label="Sensitivity",
                    variable=threshold_var,
                    length=180,
                    bg=dark_grey,
                    fg=text_color,
                    highlightthickness=0,
                ).pack(side=tk.LEFT, padx=(0, 10))
            if timeframe_var is not None:
                tf_frame = tk.Frame(controls, bg=dark_grey)
                tf_frame.pack(side=tk.LEFT)
                for key in ("5m", "15m", "30m", "1h"):
                    tk.Radiobutton(
                        tf_frame,
                        text=key,
                        variable=timeframe_var,
                        value=key,
                        indicatoron=0,
                        padx=4,
                        pady=1,
                        bg=dark_grey,
                        fg=text_color,
                        selectcolor="#444444",
                        activebackground="#444444",
                        activeforeground=text_color,
                    ).pack(side=tk.LEFT, padx=1)

        split = tk.PanedWindow(
            tab,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=10,
            showhandle=True,
            bg=dark_grey,
        )
        split.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        strong_frame = tk.LabelFrame(split, text="Strong", bg=dark_grey, fg=text_color)
        weak_frame = tk.LabelFrame(split, text="Weak", bg=dark_grey, fg=text_color)
        strong_text = _make_text(strong_frame, width=40, height=18)
        weak_text = _make_text(weak_frame, width=40, height=18)
        strong_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        weak_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        split.add(strong_frame, stretch="always")
        split.add(weak_frame, stretch="always")
        return {
            "tab": tab,
            "strong_text": strong_text,
            "weak_text": weak_text,
        }

    industry_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)
    sector_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)

    industry_timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)
    sector_timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)

    spy_view = _create_split_signal_tab(notebook, "VS SPY")
    industry_view = _create_split_signal_tab(notebook, "Industry Ref", industry_threshold_var, industry_timeframe_var)
    sector_view = _create_split_signal_tab(notebook, "Sector", sector_threshold_var, sector_timeframe_var)

    groups_tab = ttk.Frame(notebook)
    notebook.add(groups_tab, text="Top Industries/Sectors")
    groups_split = tk.PanedWindow(
        groups_tab,
        orient=tk.HORIZONTAL,
        sashrelief=tk.RAISED,
        sashwidth=10,
        showhandle=True,
        bg=dark_grey,
    )
    groups_split.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
    sectors_frame = tk.LabelFrame(groups_split, text="Sectors", bg=dark_grey, fg=text_color)
    industries_frame = tk.LabelFrame(groups_split, text="Industries", bg=dark_grey, fg=text_color)
    sectors_text = _make_text(sectors_frame, width=44, height=18)
    industries_text = _make_text(industries_frame, width=44, height=18)
    sectors_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    industries_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    groups_split.add(sectors_frame, stretch="always")
    groups_split.add(industries_frame, stretch="always")

    def render_rrs_snapshot(snapshot):
        threshold = snapshot.get("threshold", RRS_DEFAULT_THRESHOLD)
        timeframe_key = snapshot.get("timeframe_key", "5m")
        timeframe_label = RRS_TIMEFRAMES.get(timeframe_key, {}).get("label", timeframe_key)
        results = snapshot.get("results", [])
        sector_results = snapshot.get("results_sector", [])
        industry_results = snapshot.get("results_industry", [])
        group_strength = snapshot.get("group_strength", {})
        timestamp = snapshot.get("timestamp", datetime.now())
        env_label_var.set(f"Environment: {snapshot.get('market_environment_label', 'Environment')}")

        def render_split_table(view, title, rows, local_threshold, selected_tf):
            filtered = [r for r in rows if abs(r[2]) >= local_threshold]
            strong_rows = [r for r in filtered if r[0] == "RS"]
            weak_rows = [r for r in filtered if r[0] == "RW"]
            for widget, heading, subset, tag in (
                (view["strong_text"], "Strong", strong_rows, "rrs_rs"),
                (view["weak_text"], "Weak", weak_rows, "rrs_rw"),
            ):
                widget.config(state="normal")
                widget.delete("1.0", tk.END)
                widget.insert(
                    tk.END,
                    f"{title} {heading}  TF:{selected_tf}  Threshold:{local_threshold:.2f}\n",
                    "rrs_hdr",
                )
                widget.insert(tk.END, "SYMBOL  SIDE  RRS    POWER\n")
                widget.insert(tk.END, "--------------------------\n")
                if not subset:
                    widget.insert(tk.END, "No symbols flagged.\n")
                for signal, symbol, rrs_value, power in subset:
                    line = f"{symbol:<6}  {signal:<4}  {rrs_value:+.2f}  {power if power is not None else 0:>6.2f}\n"
                    widget.insert(tk.END, line, tag)
                widget.config(state="disabled")
                widget.see("1.0")

        render_split_table(spy_view, "RS/RW vs SPY", results, threshold, timeframe_key)
        render_split_table(
            industry_view,
            "RS/RW vs Industry Ref",
            industry_results,
            industry_threshold_var.get(),
            industry_timeframe_var.get(),
        )
        render_split_table(
            sector_view,
            "RS/RW vs Sector",
            sector_results,
            sector_threshold_var.get(),
            sector_timeframe_var.get(),
        )

        def render_group_column(widget, label, key_name):
            widget.config(state="normal")
            widget.delete("1.0", tk.END)
            widget.insert(
                tk.END,
                f"{label}  Last scan: {timestamp.strftime('%H:%M:%S')}  Base TF:{timeframe_label}\n",
                "rrs_hdr",
            )
            for tf in ("M5", "H1", "D1"):
                payload = group_strength.get(tf, {})
                items = payload.get(key_name, [])
                widget.insert(tk.END, f"\n[{tf}] Strongest\n", "rrs_hdr")
                if not items:
                    widget.insert(tk.END, "  No data\n")
                for item in items[:SCAN_EXTREME_COUNT]:
                    widget.insert(
                        tk.END,
                        f"  + {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n",
                        "rrs_rs",
                    )
                widget.insert(tk.END, f"[{tf}] Weakest\n", "rrs_hdr")
                if not items:
                    widget.insert(tk.END, "  No data\n")
                for item in list(reversed(items[-SCAN_EXTREME_COUNT:])):
                    widget.insert(
                        tk.END,
                        f"  - {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n",
                        "rrs_rw",
                    )
            widget.config(state="disabled")
            widget.see("1.0")

        render_group_column(sectors_text, "Sectors", "sectors")
        render_group_column(industries_text, "Industries", "industries")

        env_focus_text.config(state='normal')
        env_focus_text.delete("1.0", tk.END)
        env_highlights = snapshot.get("environment_highlights", [])
        env_focus_text.insert(tk.END, f"{snapshot.get('market_environment_label', 'Environment')} Focus\n", "rrs_hdr")
        for section in env_highlights:
            env_focus_text.insert(tk.END, f"\n{section.get('title', 'Section')}\n", "rrs_hdr")
            rows = section.get("rows", [])
            if not rows:
                env_focus_text.insert(tk.END, "  None\n")
            for row in rows:
                env_focus_text.insert(tk.END, f"  {row.get('text', '')}\n", row.get("tag", "rrs_rs"))
        env_focus_text.config(state='disabled')
        env_focus_text.see("1.0")

        env_copy_text.config(state='normal')
        env_copy_text.delete("1.0", tk.END)
        env_copy_text.insert(tk.END, build_environment_focus_copy_text(snapshot))
        env_copy_text.config(state='disabled')
        env_copy_text.see("1.0")

    return {
        "container": rrs_container,
        "rrs_status_var": rrs_status_var,
        "render_rrs_snapshot": render_rrs_snapshot,
    }


def choose_gui_mode():
    preferred_mode = str(get_local_setting("bounce_bot_gui_mode", "full") or "full").strip().lower()
    if preferred_mode not in {"full", "lightweight"}:
        preferred_mode = "full"
    selection = {"mode": preferred_mode}
    picker = tk.Tk()
    picker.title("BounceBot Mode")
    picker.geometry("360x160")
    picker.configure(bg="#2E2E2E")
    picker.resizable(False, False)

    tk.Label(
        picker,
        text="Choose BounceBot startup mode",
        bg="#2E2E2E",
        fg="#E0E0E0",
        font=("Arial", 12, "bold"),
    ).pack(pady=(18, 10))

    tk.Label(
        picker,
        text=(
            "Full mode keeps the RS/RW panels.\n"
            "Lightweight mode keeps alerts and core bounce controls only.\n"
            f"Default on this computer: {preferred_mode.title()}"
        ),
        bg="#2E2E2E",
        fg="#E0E0E0",
        justify=tk.CENTER,
    ).pack(pady=(0, 14))

    button_row = tk.Frame(picker, bg="#2E2E2E")
    button_row.pack()

    def select_mode(mode):
        selection["mode"] = mode
        save_local_setting("bounce_bot_gui_mode", mode)
        picker.destroy()

    tk.Button(button_row, text="Full", width=12, command=lambda: select_mode("full")).pack(side=tk.LEFT, padx=8)
    tk.Button(button_row, text="Lightweight", width=12, command=lambda: select_mode("lightweight")).pack(side=tk.LEFT, padx=8)

    picker.protocol("WM_DELETE_WINDOW", lambda: select_mode("full"))
    picker.mainloop()
    return selection["mode"]


def prompt_change_home_folder(root, cleanup_callback=None):
    details = get_tracker_storage_details()
    current_dir = Path(details["data_dir"])
    selected = filedialog.askdirectory(
        title="Choose home folder",
        initialdir=str(current_dir if current_dir.exists() else Path.home()),
        mustexist=False,
    )
    if not selected:
        return

    target = save_tracker_storage_dir(selected)
    restart_now = messagebox.askyesno(
        "Home Folder Saved",
        "Saved this computer's home folder.\n\n"
        f"Folder: {target}\n\n"
        "Restart BounceBot now so it starts using the new home folder?",
    )
    if restart_now:
        if cleanup_callback is not None:
            cleanup_callback()
        root.destroy()
        os.execv(sys.executable, [sys.executable] + sys.argv)


def append_alert_message(text_area, msg, tag, timestamp):
    if msg.startswith("MASTER_AVWAP_FAVORITE_BOUNCE:"):
        text_area.insert(tk.END, f"{timestamp} - {msg}\n", tag)
    elif "Bounce confirmed" in msg:
        parts = msg.split(":", 1)
        if len(parts) == 2:
            symbol = parts[0].strip()
            rest = ":" + parts[1]
            if "(long)" in rest:
                text_area.insert(tk.END, f"{timestamp} - ", tag)
                text_area.insert(tk.END, symbol, "pink_symbol")
                text_area.insert(tk.END, rest + "\n", "green")
            elif "(short)" in rest:
                text_area.insert(tk.END, f"{timestamp} - ", tag)
                text_area.insert(tk.END, symbol, "orange_symbol")
                text_area.insert(tk.END, rest + "\n", "red")
            else:
                text_area.insert(tk.END, f"{timestamp} - {msg}\n", tag)
        else:
            text_area.insert(tk.END, f"{timestamp} - {msg}\n", tag)
    elif "Price approaching levels" in msg:
        parts = msg.split(":", 1)
        if len(parts) == 2:
            symbol = parts[0].strip()
            rest = ":" + parts[1]
            if "(long)" in rest:
                text_area.insert(tk.END, f"{timestamp} - ", tag)
                text_area.insert(tk.END, symbol, "pink_symbol")
                text_area.insert(tk.END, rest + "\n", "approaching_green")
            elif "(short)" in rest:
                text_area.insert(tk.END, f"{timestamp} - ", tag)
                text_area.insert(tk.END, symbol, "orange_symbol")
                text_area.insert(tk.END, rest + "\n", "approaching_red")
            else:
                text_area.insert(tk.END, f"{timestamp} - {msg}\n", tag)
        else:
            text_area.insert(tk.END, f"{timestamp} - {msg}\n", tag)
    else:
        text_area.insert(tk.END, f"{timestamp} - {msg}\n", tag)


def configure_alert_tags(text_area, font_size=12):
    text_area.tag_config("green", foreground="#50FA7B", font=("Courier", font_size))
    text_area.tag_config("red", foreground="#FF5555", font=("Courier", font_size))
    text_area.tag_config("pink_symbol", foreground="#FF79C6", font=("Courier", font_size, "bold"))
    text_area.tag_config("orange_symbol", foreground="#FFB86C", font=("Courier", font_size, "bold"))
    text_area.tag_config("blue", foreground="#8BE9FD", font=("Courier", font_size))
    text_area.tag_config("master_avwap_favorite_long", foreground="#00E5FF", font=("Courier", font_size, "bold"))
    text_area.tag_config("master_avwap_favorite_short", foreground="#FFD166", font=("Courier", font_size, "bold"))
    text_area.tag_config("master_avwap_focus_long", foreground="#7DF9FF", font=("Courier", font_size, "bold"))
    text_area.tag_config("master_avwap_focus_short", foreground="#FFB000", font=("Courier", font_size, "bold"))
    text_area.tag_config("candle_line", foreground="#BD93F9", overstrike=1)
    text_area.tag_config("approaching", foreground="#FF79C6", font=("Courier", font_size))
    text_area.tag_config("approaching_green", foreground="#50FA7B", font=("Courier", font_size))
    text_area.tag_config("approaching_red", foreground="#FF5555", font=("Courier", font_size))


def start_lightweight_gui():
    bounce_queue = queue.Queue()
    dark_grey = "#2E2E2E"
    text_color = "#E0E0E0"
    input_grey = "#252525"
    panel_grey = "#3A3A3A"

    def gui_callback(message, tag):
        if tag.startswith("rrs"):
            return
        if tag == "approaching" or tag.startswith("approaching_"):
            return
        if tag == "blue" and "removed from" in str(message):
            return
        bounce_queue.put((message, tag))

    bot_instance = run_bot_with_gui(gui_callback)

    root = tk.Tk()
    root.title("BounceBot Lightweight")
    root.geometry("980x680")
    root.configure(background=dark_grey)

    container = tk.Frame(root, padx=10, pady=10, bg=dark_grey)
    container.pack(fill=tk.BOTH, expand=True)

    header = tk.Frame(container, bg=dark_grey)
    header.pack(fill=tk.X, pady=(0, 8))

    status_var = tk.StringVar(value="listening for alerts")
    connection_var = tk.StringVar(value="IB: connected")
    tk.Label(header, text="BounceBot Lightweight", bg=dark_grey, fg=text_color, font=("Arial", 11, "bold")).pack(side=tk.LEFT)
    tk.Label(header, textvariable=connection_var, bg=dark_grey, fg=text_color).pack(side=tk.LEFT, padx=(12, 0))
    tk.Label(header, textvariable=status_var, bg=dark_grey, fg=text_color).pack(side=tk.LEFT, padx=(12, 0))

    def disconnect_bot():
        nonlocal bot_instance
        try:
            bot_instance.disconnect()
        except Exception:
            pass
        bot_instance = None
        connection_var.set("IB: disconnected")
        status_var.set("alerts paused")

    def restart_bot():
        nonlocal bot_instance
        disconnect_bot()
        connection_var.set("IB: reconnecting")
        status_var.set("starting...")
        bot_instance = run_bot_with_gui(gui_callback)
        rrs_threshold_var.set(bot_instance.rrs_threshold)
        timeframe_var.set(bot_instance.rrs_timeframe_key)
        env_selection_var.set(bot_instance.get_market_environment())
        connection_var.set("IB: connected")
        status_var.set("listening for alerts")

    def switch_mode(new_mode):
        save_local_setting("bounce_bot_gui_mode", new_mode)
        disconnect_bot()
        root.destroy()
        start_gui(mode=new_mode)

    tk.Button(
        header,
        text="Change Home Folder",
        command=lambda: prompt_change_home_folder(root, cleanup_callback=disconnect_bot),
        relief=tk.RAISED,
        padx=10,
        bg=panel_grey,
        fg=text_color,
    ).pack(side=tk.RIGHT)
    tk.Button(header, text="Switch to Full", command=lambda: switch_mode("full"), relief=tk.RAISED, padx=10, bg=panel_grey, fg=text_color).pack(side=tk.RIGHT)
    tk.Button(header, text="Reconnect", command=restart_bot, relief=tk.RAISED, padx=10, bg=panel_grey, fg=text_color).pack(side=tk.RIGHT)
    tk.Button(header, text="Disconnect", command=disconnect_bot, relief=tk.RAISED, padx=10, bg=panel_grey, fg=text_color).pack(side=tk.RIGHT, padx=(0, 8))

    def clear_alerts():
        text_area.config(state="normal")
        text_area.delete("1.0", tk.END)
        text_area.config(state="disabled")

    tk.Button(header, text="Clear", command=clear_alerts, relief=tk.RAISED, padx=10, bg=panel_grey, fg=text_color).pack(side=tk.RIGHT, padx=(0, 8))

    controls = tk.Frame(container, bg=dark_grey)
    controls.pack(fill=tk.X, pady=(0, 8))

    tk.Label(controls, text="RRS Sensitivity", bg=dark_grey, fg=text_color).pack(side=tk.LEFT)
    rrs_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)

    def on_rrs_threshold_change(*_):
        bot_instance.set_rrs_threshold(rrs_threshold_var.get())

    rrs_threshold_var.trace_add("write", on_rrs_threshold_change)

    tk.Scale(
        controls,
        from_=0.0,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        variable=rrs_threshold_var,
        length=180,
        bg=dark_grey,
        fg=text_color,
        highlightthickness=0,
    ).pack(side=tk.LEFT, padx=(8, 14))

    tk.Label(controls, text="Timeframe", bg=dark_grey, fg=text_color).pack(side=tk.LEFT)
    timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)

    def on_timeframe_change():
        selected = timeframe_var.get()
        bot_instance.set_rrs_timeframe(selected)
        status_var.set(f"RRS timeframe set to {RRS_TIMEFRAMES[selected]['label']}")

    for key in ("5m", "15m", "30m", "1h"):
        tk.Radiobutton(
            controls,
            text=RRS_TIMEFRAMES[key]["label"],
            variable=timeframe_var,
            value=key,
            indicatoron=0,
            command=on_timeframe_change,
            padx=6,
            pady=2,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        ).pack(side=tk.LEFT, padx=2)

    env_frame = tk.Frame(container, bg=dark_grey)
    env_frame.pack(fill=tk.X, pady=(0, 8))
    tk.Label(env_frame, text="Market Environment", bg=dark_grey, fg=text_color).pack(side=tk.LEFT, padx=(0, 8))
    env_selection_var = tk.StringVar(value=bot_instance.get_market_environment())

    def on_environment_change():
        selected = env_selection_var.get()
        bot_instance.set_market_environment(selected)
        status_var.set(f"Environment: {MARKET_ENVIRONMENTS[selected]['label']}")

    for key, info in MARKET_ENVIRONMENTS.items():
        tk.Radiobutton(
            env_frame,
            text=info["label"],
            variable=env_selection_var,
            value=key,
            indicatoron=0,
            command=on_environment_change,
            padx=8,
            pady=3,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        ).pack(side=tk.LEFT, padx=2)

    alerts_frame = tk.Frame(container, bg=dark_grey)
    alerts_frame.pack(fill=tk.BOTH, expand=True)
    text_area = scrolledtext.ScrolledText(
        alerts_frame,
        wrap=tk.WORD,
        width=80,
        height=30,
        font=("Courier", 11),
        state="disabled",
        bg=input_grey,
        fg=text_color,
        insertbackground=text_color,
    )
    text_area.pack(fill=tk.BOTH, expand=True)
    configure_alert_tags(text_area, font_size=11)

    def process_bounce_queue():
        while True:
            try:
                msg, tag = bounce_queue.get_nowait()
            except queue.Empty:
                break
            text_area.config(state="normal")
            append_alert_message(text_area, str(msg), str(tag), datetime.now().strftime("%H:%M:%S"))
            text_area.config(state="disabled")
            text_area.see(tk.END)
            root.update()
        root.after(150, process_bounce_queue)

    def on_closing():
        disconnect_bot()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    process_bounce_queue()
    root.mainloop()


def start_gui(mode="prompt"):
    if mode == "prompt":
        mode = str(get_local_setting("bounce_bot_gui_mode", "full") or "full").strip().lower()
        if mode not in {"full", "lightweight"}:
            mode = "full"
    if mode in {"full", "lightweight"}:
        save_local_setting("bounce_bot_gui_mode", mode)
    if mode == "lightweight":
        start_lightweight_gui()
        return

    bounce_queue = queue.Queue()
    rrs_queue = queue.Queue()
    # Replace light_grey with dark_grey
    dark_grey = "#2E2E2E"  # Dark grey color code
    text_color = "#E0E0E0"  # Light text color for dark background

    def gui_callback(message, tag):
        if tag.startswith("rrs"):
            rrs_queue.put((message, tag))
        elif tag == "candle_line":
            bounce_queue.put((message, tag))
        elif tag == "blue" and "removed from" in message:
            pass
        else:
            bounce_queue.put((message, tag))

    bot_instance = run_bot_with_gui(gui_callback)

    # Main bounce alerts window
    root = tk.Tk()
    root.title("BounceBot Alerts")
    root.geometry("800x600")
    root.configure(background=dark_grey)

    frame = tk.Frame(root, padx=10, pady=10, bg=dark_grey)
    frame.pack(fill=tk.BOTH, expand=True)

    header = tk.Frame(frame, bg=dark_grey)
    header.pack(fill=tk.X, pady=(0, 8))
    tk.Label(header, text="BounceBot Full", bg=dark_grey, fg=text_color, font=("Arial", 11, "bold")).pack(side=tk.LEFT)

    def switch_mode(new_mode):
        save_local_setting("bounce_bot_gui_mode", new_mode)
        try:
            bot_instance.disconnect()
        except Exception:
            pass
        root.destroy()
        start_gui(mode=new_mode)

    tk.Button(
        header,
        text="Switch to Lightweight",
        command=lambda: switch_mode("lightweight"),
        relief=tk.RAISED,
        padx=10,
        bg="#3A3A3A",
        fg=text_color,
    ).pack(side=tk.RIGHT)
    tk.Button(
        header,
        text="Change Home Folder",
        command=lambda: prompt_change_home_folder(root, cleanup_callback=lambda: bot_instance.disconnect()),
        relief=tk.RAISED,
        padx=10,
        bg="#3A3A3A",
        fg=text_color,
    ).pack(side=tk.RIGHT, padx=(0, 8))

    content_pane = tk.PanedWindow(frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg=dark_grey)
    content_pane.pack(fill=tk.BOTH, expand=True)

    alerts_frame = tk.Frame(content_pane, bg=dark_grey)
    text_area = scrolledtext.ScrolledText(
        alerts_frame,
        wrap=tk.WORD,
        width=80,
        height=30,
        font=('Courier', 12),
        state='disabled',
        bg=dark_grey,
        fg=text_color  # Add text color
    )
    text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    content_pane.add(alerts_frame, stretch="always")

    configure_alert_tags(text_area, font_size=12)

    # Create RRS panel inside main window
    rrs_container = tk.Frame(content_pane, bg=dark_grey)
    content_pane.add(rrs_container, stretch="always")

    rrs_controls = tk.Frame(rrs_container, padx=10, pady=10, bg=dark_grey)
    rrs_controls.pack(fill=tk.X)

    rrs_status_var = tk.StringVar(value="RRS ready")
    rrs_status_label = tk.Label(rrs_controls, textvariable=rrs_status_var, fg=text_color, bg=dark_grey)
    rrs_status_label.pack(side=tk.LEFT, padx=(0, 10))

    rrs_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)

    def on_rrs_threshold_change(*_):
        bot_instance.set_rrs_threshold(rrs_threshold_var.get())

    rrs_threshold_var.trace_add("write", on_rrs_threshold_change)

    rrs_scale = tk.Scale(
        rrs_controls,
        from_=0.0,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        label="RRS Sensitivity",
        variable=rrs_threshold_var,
        length=220,
        bg=dark_grey,
        fg=text_color,
        highlightthickness=0,
    )
    rrs_scale.pack(side=tk.LEFT, padx=(0, 10))

    timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)

    def on_timeframe_change() -> None:
        bot_instance.set_rrs_timeframe(timeframe_var.get())

    for key in ("5m", "15m", "30m", "1h"):
        label = RRS_TIMEFRAMES[key]["label"]
        btn = tk.Radiobutton(
            rrs_controls,
            text=label,
            variable=timeframe_var,
            value=key,
            indicatoron=0,
            command=on_timeframe_change,
            padx=6,
            pady=2,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        )
        btn.pack(side=tk.LEFT, padx=2)

    env_selection_var = tk.StringVar(value=bot_instance.get_market_environment())
    env_label_var = tk.StringVar(
        value=f"Environment: {MARKET_ENVIRONMENTS.get(env_selection_var.get(), {}).get('label', env_selection_var.get())}"
    )

    def on_environment_change():
        selected = env_selection_var.get()
        env_label_var.set(
            f"Environment: {MARKET_ENVIRONMENTS.get(selected, {}).get('label', selected)}"
        )
        bot_instance.set_market_environment(selected)

    env_mode_frame = tk.Frame(rrs_container, bg=dark_grey, pady=4)
    env_mode_frame.pack(fill=tk.X, padx=10)
    tk.Label(env_mode_frame, textvariable=env_label_var, fg=text_color, bg=dark_grey).pack(side=tk.LEFT)
    env_button_frame = tk.Frame(env_mode_frame, bg=dark_grey)
    env_button_frame.pack(side=tk.RIGHT)
    for key, info in MARKET_ENVIRONMENTS.items():
        tk.Radiobutton(
            env_button_frame,
            text=info["label"],
            variable=env_selection_var,
            value=key,
            indicatoron=0,
            command=on_environment_change,
            padx=6,
            pady=2,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        ).pack(side=tk.LEFT, padx=2)

    rrs_frame = tk.Frame(rrs_container, padx=10, pady=10, bg=dark_grey)
    rrs_frame.pack(fill=tk.BOTH, expand=True)

    rrs_main_text = scrolledtext.ScrolledText(
        rrs_frame,
        wrap=tk.NONE,
        width=80,
        height=12,
        font=('Courier', 11),
        state='disabled',
        bg=dark_grey,
        fg=text_color
    )
    rrs_main_text.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

    env_focus_row = tk.PanedWindow(
        rrs_frame,
        orient=tk.HORIZONTAL,
        sashrelief=tk.RAISED,
        sashwidth=10,
        showhandle=True,
        bg=dark_grey,
    )
    env_focus_row.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

    env_focus_frame = tk.LabelFrame(env_focus_row, text="Environment Focus", bg=dark_grey, fg=text_color)
    env_focus_text = scrolledtext.ScrolledText(
        env_focus_frame,
        wrap=tk.NONE,
        width=80,
        height=8,
        font=('Courier', 10),
        state='disabled',
        bg=dark_grey,
        fg=text_color,
    )
    env_focus_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    env_focus_text.tag_config("rrs_hdr", foreground="#BD93F9", font=('Courier', 11, 'bold'))
    env_focus_text.tag_config("rrs_rs", foreground="#50FA7B", font=('Courier', 11, 'bold'))
    env_focus_text.tag_config("rrs_rw", foreground="#FF5555", font=('Courier', 11, 'bold'))
    env_focus_row.add(env_focus_frame, stretch="always")

    env_copy_frame = tk.LabelFrame(env_focus_row, text="Environment Focus Lists", bg=dark_grey, fg=text_color)
    env_copy_toolbar = tk.Frame(env_copy_frame, bg=dark_grey)
    env_copy_toolbar.pack(fill=tk.X, padx=4, pady=(4, 0))
    env_copy_text = scrolledtext.ScrolledText(
        env_copy_frame,
        wrap=tk.WORD,
        width=52,
        height=8,
        font=('Courier', 10),
        state='disabled',
        bg=dark_grey,
        fg=text_color,
    )

    def copy_env_focus_lists():
        text = env_copy_text.get("1.0", tk.END).strip()
        if not text:
            rrs_status_var.set("Environment focus lists: nothing to copy.")
            return
        copy_text_to_clipboard(env_copy_text, text)
        rrs_status_var.set("Copied environment focus lists to clipboard.")

    tk.Button(
        env_copy_toolbar,
        text="Copy",
        command=copy_env_focus_lists,
        relief=tk.RAISED,
        padx=10,
        bg=dark_grey,
        fg=text_color,
    ).pack(side=tk.LEFT)
    env_copy_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    env_focus_row.add(env_copy_frame, stretch="always")

    rrs_compare_frame = tk.Frame(rrs_frame, bg=dark_grey)
    rrs_compare_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    industry_col = tk.LabelFrame(rrs_compare_frame, text="RS/RW vs Industry Ref", bg=dark_grey, fg=text_color)
    industry_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
    sector_col = tk.LabelFrame(rrs_compare_frame, text="RS/RW vs Sector", bg=dark_grey, fg=text_color)
    sector_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

    industry_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)
    sector_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)

    tk.Scale(industry_col, from_=0.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, label="Sensitivity",
             variable=industry_threshold_var, length=180, bg=dark_grey, fg=text_color, highlightthickness=0).pack(fill=tk.X, padx=4)
    tk.Scale(sector_col, from_=0.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, label="Sensitivity",
             variable=sector_threshold_var, length=180, bg=dark_grey, fg=text_color, highlightthickness=0).pack(fill=tk.X, padx=4)

    industry_timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)
    sector_timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)

    industry_tf = tk.Frame(industry_col, bg=dark_grey)
    industry_tf.pack(fill=tk.X)
    sector_tf = tk.Frame(sector_col, bg=dark_grey)
    sector_tf.pack(fill=tk.X)
    for key in ("5m", "15m", "30m", "1h"):
        for parent, var in ((industry_tf, industry_timeframe_var), (sector_tf, sector_timeframe_var)):
            tk.Radiobutton(parent, text=key, variable=var, value=key, indicatoron=0, padx=4, pady=1,
                           bg=dark_grey, fg=text_color, selectcolor="#444444", activebackground="#444444",
                           activeforeground=text_color).pack(side=tk.LEFT, padx=1)

    industry_text = scrolledtext.ScrolledText(industry_col, wrap=tk.NONE, width=45, height=14, font=('Courier', 10), state='disabled', bg=dark_grey, fg=text_color)
    industry_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    sector_text = scrolledtext.ScrolledText(sector_col, wrap=tk.NONE, width=45, height=14, font=('Courier', 10), state='disabled', bg=dark_grey, fg=text_color)
    sector_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    group_frame = tk.LabelFrame(rrs_frame, text="Top Industries/Sectors", bg=dark_grey, fg=text_color)
    group_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    group_text = scrolledtext.ScrolledText(group_frame, wrap=tk.NONE, width=90, height=14, font=('Courier', 10), state='disabled', bg=dark_grey, fg=text_color)
    group_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    for widget in (rrs_main_text, industry_text, sector_text, group_text):
        widget.tag_config("rrs_hdr", foreground="#BD93F9", font=('Courier', 11, 'bold'))
        widget.tag_config("rrs_rs", foreground="#50FA7B", font=('Courier', 11, 'bold'))
        widget.tag_config("rrs_rw", foreground="#FF5555", font=('Courier', 11, 'bold'))


    button_frame = tk.Frame(frame, bg=dark_grey)  # Add background color to button frame
    button_frame.pack(fill=tk.X, pady=10)

    bounce_toggle_frame = tk.LabelFrame(
        frame,
        text="Bounce Filters",
        bg=dark_grey,
        fg=text_color,
        padx=8,
        pady=6,
        highlightbackground="#444444",
        highlightcolor="#444444",
    )
    bounce_toggle_frame.pack(fill=tk.X, pady=(0, 8))

    bounce_toggle_vars = {}

    def on_toggle_bounce(bounce_key, var):
        bot_instance.set_bounce_type_enabled(bounce_key, bool(var.get()))

    toggle_order = [
        "10_candle",
        "vwap",
        "dynamic_vwap",
        "eod_vwap",
        "vwap_eod_confluence",
        "impulse_retest_vwap_eod",
        "ema_8",
        "ema_15",
        "ema_21",
        "vwap_upper_band",
        "vwap_lower_band",
        "dynamic_vwap_upper_band",
        "dynamic_vwap_lower_band",
        "eod_vwap_upper_band",
        "eod_vwap_lower_band",
        "prev_day_high",
        "prev_day_low",
    ]

    for idx, bounce_key in enumerate(toggle_order):
        var = tk.BooleanVar(value=bot_instance.is_bounce_type_enabled(bounce_key))
        bounce_toggle_vars[bounce_key] = var
        chk = tk.Checkbutton(
            bounce_toggle_frame,
            text=BOUNCE_TYPE_LABELS.get(bounce_key, bounce_key),
            variable=var,
            command=lambda k=bounce_key, v=var: on_toggle_bounce(k, v),
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        )
        row = idx // 4
        col = idx % 4
        chk.grid(row=row, column=col, sticky="w", padx=6, pady=2)


    def check_dvwap_touches():
        results = bot_instance.check_dynamic_vwap_touches()
        text_area.config(state='normal')
        text_area.insert(tk.END, "\n=== DVWAP Touch Check Results ===\n", "blue")
        for result in results:
            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {result}\n", "green")
        text_area.config(state='disabled')
        text_area.see(tk.END)
        root.update()

    def check_dvwap2_touches():
        results = bot_instance.check_dynamic_vwap2_touches()
        text_area.config(state='normal')
        text_area.insert(tk.END, "\n=== DVWAP2 Touch Check Results ===\n", "blue")
        for result in results:
            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {result}\n", "green")
        text_area.config(state='disabled')
        text_area.see(tk.END)
        root.update()

    dvwap_button = tk.Button(
        button_frame, 
        text="Check DVWAP Touches",
        command=check_dvwap_touches,
        relief=tk.RAISED,
        padx=10
    )
    dvwap_button.pack(side=tk.LEFT, padx=5)

    dvwap2_button = tk.Button(
        button_frame,
        text="Check DVWAP2 Touches",
        command=check_dvwap2_touches,
        relief=tk.RAISED,
        padx=10
    )
    dvwap2_button.pack(side=tk.LEFT, padx=5)

    scanning_button_text = tk.StringVar(
        value="Stop Scanning" if bot_instance.is_scanning_enabled() else "Start Scanning"
    )

    def toggle_scanning():
        new_state = not bot_instance.is_scanning_enabled()
        bot_instance.set_scanning_enabled(new_state)
        scanning_button_text.set("Stop Scanning" if new_state else "Start Scanning")

    scanning_button = tk.Button(
        button_frame,
        textvariable=scanning_button_text,
        command=toggle_scanning,
        relief=tk.RAISED,
        padx=10,
    )
    scanning_button.pack(side=tk.LEFT, padx=5)

    def process_bounce_queue():
        while True:
            try:
                msg, tag = bounce_queue.get_nowait()
                text_area.config(state='normal')
                append_alert_message(text_area, str(msg), str(tag), datetime.now().strftime('%H:%M:%S'))
                text_area.config(state='disabled')
                text_area.see(tk.END)
                root.update()
            except queue.Empty:
                break
        root.after(100, process_bounce_queue)

    def render_rrs_snapshot(snapshot):
        threshold = snapshot.get("threshold", RRS_DEFAULT_THRESHOLD)
        timeframe_key = snapshot.get("timeframe_key", "5m")
        timeframe_label = RRS_TIMEFRAMES.get(timeframe_key, {}).get("label", timeframe_key)
        results = snapshot.get("results", [])
        sector_results = snapshot.get("results_sector", [])
        industry_results = snapshot.get("results_industry", [])
        group_strength = snapshot.get("group_strength", {})
        timestamp = snapshot.get("timestamp", datetime.now())

        def render_table(widget, title, rows, local_threshold):
            selected_tf = timeframe_key
            if title.startswith("RS/RW vs Industry"):
                selected_tf = industry_timeframe_var.get()
            elif title.startswith("RS/RW vs Sector"):
                selected_tf = sector_timeframe_var.get()
            widget.config(state='normal')
            widget.delete("1.0", tk.END)
            widget.insert(tk.END, f"{title}  TF:{selected_tf}  Threshold:{local_threshold:.2f}\n", "rrs_hdr")
            widget.insert(tk.END, "SYMBOL  SIDE  RRS    POWER\n")
            widget.insert(tk.END, "--------------------------\n")
            filtered = [r for r in rows if abs(r[2]) >= local_threshold]
            if not filtered:
                widget.insert(tk.END, "No symbols flagged.\n")
            for signal, symbol, rrs_value, power in filtered:
                line = f"{symbol:<6}  {signal:<4}  {rrs_value:+.2f}  {power if power is not None else 0:>6.2f}\n"
                widget.insert(tk.END, line, "rrs_rs" if signal == "RS" else "rrs_rw")
            widget.config(state='disabled')
            widget.see("1.0")

        render_table(rrs_main_text, "RS/RW vs SPY", results, threshold)
        render_table(industry_text, "RS/RW vs Industry Ref", industry_results, industry_threshold_var.get())
        render_table(sector_text, "RS/RW vs Sector", sector_results, sector_threshold_var.get())

        group_text.config(state='normal')
        group_text.delete("1.0", tk.END)
        group_text.insert(tk.END, f"Last scan: {timestamp.strftime('%H:%M:%S')}   Timeframe: {timeframe_label}\n", "rrs_hdr")
        for tf in ("D1", "H1", "M5"):
            payload = group_strength.get(tf, {})
            sectors = payload.get("sectors", [])
            industries = payload.get("industries", [])
            group_text.insert(tk.END, f"\n[{tf}] Sectors\n", "rrs_hdr")
            for item in sectors[:SCAN_EXTREME_COUNT]:
                group_text.insert(tk.END, f"  + {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n", "rrs_rs")
            for item in list(reversed(sectors[-SCAN_EXTREME_COUNT:])):
                group_text.insert(tk.END, f"  - {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n", "rrs_rw")
            group_text.insert(tk.END, f"[{tf}] Industries\n", "rrs_hdr")
            for item in industries[:SCAN_EXTREME_COUNT]:
                group_text.insert(tk.END, f"  + {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n", "rrs_rs")
            for item in list(reversed(industries[-SCAN_EXTREME_COUNT:])):
                group_text.insert(tk.END, f"  - {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n", "rrs_rw")
        group_text.config(state='disabled')
        group_text.see("1.0")

        env_focus_text.config(state='normal')
        env_focus_text.delete("1.0", tk.END)
        env_highlights = snapshot.get("environment_highlights", [])
        env_focus_text.insert(tk.END, f"{snapshot.get('market_environment_label', 'Environment')} Focus\n", "rrs_hdr")
        for section in env_highlights:
            env_focus_text.insert(tk.END, f"\n{section.get('title', 'Section')}\n", "rrs_hdr")
            rows = section.get("rows", [])
            if not rows:
                env_focus_text.insert(tk.END, "  None\n")
            for row in rows:
                env_focus_text.insert(tk.END, f"  {row.get('text', '')}\n", row.get("tag", "rrs_rs"))
        env_focus_text.config(state='disabled')
        env_focus_text.see("1.0")

        env_copy_text.config(state='normal')
        env_copy_text.delete("1.0", tk.END)
        env_copy_text.insert(tk.END, build_environment_focus_copy_text(snapshot))
        env_copy_text.config(state='disabled')
        env_copy_text.see("1.0")

    def process_rrs_queue():
        while True:
            try:
                msg, tag = rrs_queue.get_nowait()
            except queue.Empty:
                break

            if tag == "rrs_status":
                rrs_status_var.set(str(msg))
            elif tag == "rrs_snapshot":
                render_rrs_snapshot(msg)
        root.after(150, process_rrs_queue)


    def on_closing():
        bot_instance.disconnect()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    process_bounce_queue()
    process_rrs_queue()
    root.mainloop()


##########################################
# Main
##########################################
if __name__ == "__main__":
    print("Starting script...")
    reset_log_files()  # Use the new function instead of reset_log_file()
    print("Runtime files prepared.")
    
    # Rest of the code remains the same...

    
    try:
        import argparse
        import sys
        print("Imports successful.")

        parser = argparse.ArgumentParser()
        parser.add_argument("--use_gui", action="store_true", help="Use the Tkinter GUI")
        parser.add_argument(
            "--gui_mode",
            choices=("prompt", "full", "lightweight"),
            default="prompt",
            help="Choose GUI startup mode when launching with the GUI.",
        )
        print("Parser created.")
        
        args = parser.parse_args()
        print(f"Arguments parsed: {args}")
        
        # Determine whether to use GUI based on args or global setting
        use_gui = args.use_gui if args.use_gui else USE_GUI
        print(f"Using GUI: {use_gui}")
        
        if use_gui:
            print("Initializing GUI mode...")
            start_gui(mode=args.gui_mode)
        else:
            print("Initializing console mode...")
            configure_app_logging()
            print("Logging configured.")
            print("Filters applied.")
            
            # Initialize and run the bot without GUI
            print("Creating BounceBot instance...")
            bot = BounceBot()
            bot.set_connection_info("127.0.0.1", 7496, 125)
            print("Connecting to IB API...")
            bot.connect("127.0.0.1", 7496, clientId=bot.ib_client_id)
            
            # Wait for connection
            print("Waiting for connection...")
            connection_timeout = 30  # seconds
            connection_start = time.time()
            while not bot.connection_status and time.time() - connection_start < connection_timeout:
                time.sleep(1)
                print(".", end="", flush=True)
                
            if not bot.connection_status:
                print("\nConnection failed!")
                logging.error("Failed to connect to IB API within timeout period.")
                sys.exit(1)
                
            print("\nConnection successful!")
            logging.info("BounceBot is connected. Starting strategy loop...")
            
            # Run the API in a separate thread
            print("Starting API thread...")
            api_thread = threading.Thread(target=bot.run, daemon=True)
            api_thread.start()
            
            # Run the strategy
            print("Starting strategy loop...")
            bot.run_strategy()  # This will run in the main thread
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
