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
from dataclasses import dataclass
from datetime import datetime, timedelta
import zoneinfo
import queue
import tkinter as tk
from tkinter import scrolledtext
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

##########################################
# Adjustable Parameters
##########################################
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
LONGS_FILENAME = ROOT_DIR / "longs.txt"
SHORTS_FILENAME = ROOT_DIR / "shorts.txt"
BOUNCE_LOG_FILENAME = LOG_DIR / "bouncers.txt"
INTRADAY_BOUNCES_CSV = DATA_DIR / "intraday_bounces.csv"
STRENGTH_SCAN_LOG_FILENAME = LOG_DIR / "rrs_strength_extremes.csv"
GROUP_STRENGTH_SCAN_LOG_FILENAME = LOG_DIR / "rrs_group_strength_extremes.csv"
SECTOR_ETF_MAP_FILENAME = DATA_DIR / "sector_etf_map.json"
INDUSTRY_ETF_MAP_FILENAME = DATA_DIR / "industry_etf_map.json"
SYMBOL_CLASSIFICATION_CACHE_FILENAME = DATA_DIR / "symbol_classification.csv"
ATR_PERIOD = 20
THRESHOLD_MULTIPLIER = 0.02
CONSECUTIVE_CANDLES = 6  # Number of candles price must respect level before bounce
CHECK_CONSECUTIVE_CANDLES = True  # Parameter to enable/disable this check
CHECK_BOUNCE_VVWAP = True
CHECK_BOUNCE_DYNAMIC_VVWAP = True
CHECK_BOUNCE_EOD_VWAP = True 
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

BOUNCE_TYPE_DEFAULTS = {
    "10_candle": CHECK_BOUNCE_10_CANDLE,
    "vwap": CHECK_BOUNCE_VVWAP,
    "dynamic_vwap": CHECK_BOUNCE_DYNAMIC_VVWAP,
    "eod_vwap": CHECK_BOUNCE_EOD_VWAP,
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


def utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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
            return {str(k): str(v).upper() for k, v in data.items() if v}
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
    return mapping.get(key, "SPY")


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

def reset_log_files():
    files_to_reset = ["trading_bot.log", BOUNCE_LOG_FILENAME]

    for log_file_path in files_to_reset:
        try:
            # Check if file exists before attempting to remove it
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
                print(f"Previous log file deleted: {log_file_path}")
            
            # Create an empty file
            with open(log_file_path, 'w') as f:
                pass
            
            print(f"Created fresh log file: {log_file_path}")
        except Exception as e:
            print(f"Error resetting log file {log_file_path}: {e}")


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
        self.ib_client_id = None
        self.api_thread = None

        self.data = {}
        self.data_ready_events = {}

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

        self.bounce_type_toggles = dict(BOUNCE_TYPE_DEFAULTS)

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
        self.ib_client_id = client_id

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
        with self.connect_lock:
            if self.connection_status:
                return True
            try:
                try:
                    self.disconnect()
                except Exception:
                    pass
                self.connect(self.ib_host, self.ib_port, clientId=self.ib_client_id)
                self._ensure_api_thread()
                start = time.time()
                while not self.connection_status and time.time() - start < timeout:
                    time.sleep(0.2)
                if not self.connection_status:
                    logging.error("Failed to reconnect to IB within timeout.")
                    return False
                logging.info("Reconnected to IB.")
                return True
            except Exception as e:
                logging.exception(f"Reconnect error: {e}")
                return False

    def set_bounce_type_enabled(self, bounce_type, enabled):
        if bounce_type in self.bounce_type_toggles:
            self.bounce_type_toggles[bounce_type] = bool(enabled)

    def is_bounce_type_enabled(self, bounce_type):
        return self.bounce_type_toggles.get(bounce_type, False)

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

    def _load_symbol_classification_cache(self):
        if not SYMBOL_CLASSIFICATION_CACHE_FILENAME.exists():
            return
        try:
            with open(SYMBOL_CLASSIFICATION_CACHE_FILENAME, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    symbol = (row.get("symbol") or "").strip().upper()
                    if symbol:
                        self.symbol_classification_cache[symbol] = {
                            "symbol": symbol,
                            "sectorKey": row.get("sectorKey", ""),
                            "industryKey": row.get("industryKey", ""),
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
        if not sector_key:
            sector_key = slugify_key(sector_name)
        if not industry_key:
            industry_key = slugify_key(industry_name)
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
        if not self.ensure_connected():
            logging.warning("Not connected to IB; skipping historical request.")
            return None
        reqId = self.getReqId()
        with self.data_lock:
            self.data[reqId] = []
            self.data_ready_events[reqId] = threading.Event()
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
            logging.warning(f"{symbol}: Timeout waiting for RRS data.")
            return None
        with self.data_lock:
            bars = self.data.get(reqId, [])
            del self.data_ready_events[reqId]
            self.data.pop(reqId, None)
        return bars

    def get_cached_5m_bars(self, symbol):
        return self._get_cached_bars(symbol, "5 D", "5 mins")

    def run_rrs_scan(self, timeframe_key_override=None, emit_gui=True):
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

        results = []
        sector_results = []
        industry_results = []
        all_scores = []
        all_symbols = sorted(set(self.longs + self.shorts) - {"SPY"})
        for symbol in all_symbols:
            sym_5m = self.get_cached_5m_bars(symbol)
            if not sym_5m:
                continue
            sym_bars = sym_5m if timeframe_minutes == 5 else _aggregate_bars_timeframe(sym_5m, timeframe_minutes)
            aligned_sym, aligned_spy = _align_bars_with_map(sym_bars, spy_by_dt)
            rrs_value, power_index = real_relative_strength(aligned_sym, aligned_spy, length=length)
            if rrs_value is None:
                continue
            all_scores.append((symbol, rrs_value, power_index))
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

        snapshot = {
            "timestamp": datetime.now(),
            "threshold": threshold,
            "timeframe_key": timeframe_key,
            "results": ordered_results,
            "results_sector": _ordered(sector_results),
            "results_industry": _ordered(industry_results),
            "group_strength": self.compute_group_strengths(),
        }
        if emit_gui and self.gui_callback:
            self.gui_callback(snapshot, "rrs_snapshot")
            self.gui_callback(
                f"RRS scan complete ({len(ordered_results)} SPY, {len(snapshot['results_sector'])} sector, {len(snapshot['results_industry'])} industry refs)",
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
        logging.info(f"Connected to IB API. NextValidId={orderId}")

    def error(self, reqId, errorCode, errorString):
        logging.error(f"IB Error. ReqId={reqId}, Code={errorCode}, Msg={errorString}")
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
        
        # Skip if we don't have enough candles for today
        if len(today_df) < CONSECUTIVE_CANDLES + 1:  # +1 for current candle
            logging.debug(f"{symbol}: Not enough candles for today ({len(today_df)}) to check consecutive condition")
            # We can still continue without the consecutive check if it's disabled
            if CHECK_CONSECUTIVE_CANDLES:
                return None
        
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
        allowed_types = set(allowed_bounce_types) if allowed_bounce_types is not None else None

        def bounce_type_allowed(bounce_type):
            return allowed_types is None or bounce_type in allowed_types

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
                    logging.debug(f"{symbol}: 10-candle LONG bounce candidate found. New low: {current_candle_data['low']:.2f}, Previous lowest: {lowest_low_prev:.2f}")
            else:
                # For shorts, check if current candle creates a new highest high
                last_10_candles = df.iloc[-11:-1].copy()
                highest_high_prev = last_10_candles["high"].max()
                
                # Bounce condition: current candle creates a new high and closes below the open
                if current_candle_data["high"] > highest_high_prev and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["10_candle_high"] = current_candle_data["high"]
                    logging.debug(f"{symbol}: 10-candle SHORT bounce candidate found. New high: {current_candle_data['high']:.2f}, Previous highest: {highest_high_prev:.2f}")

        # Check for standard VWAP bounces if enabled
        if bounce_type_allowed("vwap") and self.is_bounce_type_enabled("vwap") and metrics.get("std_vwap") is not None:
            # Check if price respected standard VWAP for consecutive candles
            if check_consecutive_respect(metrics.get("std_vwap"), "Standard VWAP"):
                if direction == "long" and abs(current_candle_data["low"] - metrics.get("std_vwap")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["vwap"] = metrics.get("std_vwap")
                    logging.debug(f"{symbol}: Standard VWAP LONG bounce candidate found. VWAP: {metrics.get('std_vwap'):.2f}, Current Low: {current_candle_data['low']:.2f}")
                elif direction == "short" and abs(current_candle_data["high"] - metrics.get("std_vwap")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["vwap"] = metrics.get("std_vwap")
                    logging.debug(f"{symbol}: Standard VWAP SHORT bounce candidate found. VWAP: {metrics.get('std_vwap'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for Dynamic VWAP bounces if enabled
        if bounce_type_allowed("dynamic_vwap") and self.is_bounce_type_enabled("dynamic_vwap") and metrics.get("dynamic_vwap") is not None:
            # Check if price respected dynamic VWAP for consecutive candles
            if check_consecutive_respect(metrics.get("dynamic_vwap"), "Dynamic VWAP"):
                if direction == "long" and abs(current_candle_data["low"] - metrics.get("dynamic_vwap")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["dynamic_vwap"] = metrics.get("dynamic_vwap")
                    logging.debug(f"{symbol}: Dynamic VWAP LONG bounce candidate found. DVWAP: {metrics.get('dynamic_vwap'):.2f}, Current Low: {current_candle_data['low']:.2f}")
                elif direction == "short" and abs(current_candle_data["high"] - metrics.get("dynamic_vwap")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["dynamic_vwap"] = metrics.get("dynamic_vwap")
                    logging.debug(f"{symbol}: Dynamic VWAP SHORT bounce candidate found. DVWAP: {metrics.get('dynamic_vwap'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for EOD VWAP bounces if enabled
        if bounce_type_allowed("eod_vwap") and self.is_bounce_type_enabled("eod_vwap") and metrics.get("eod_vwap") is not None:
            # Check if price respected EOD VWAP for consecutive candles
            if check_consecutive_respect(metrics.get("eod_vwap"), "EOD VWAP"):
                if direction == "long" and abs(current_candle_data["low"] - metrics.get("eod_vwap")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["eod_vwap"] = metrics.get("eod_vwap")
                    logging.debug(f"{symbol}: EOD VWAP LONG bounce candidate found. EOD VWAP: {metrics.get('eod_vwap'):.2f}, Current Low: {current_candle_data['low']:.2f}")
                elif direction == "short" and abs(current_candle_data["high"] - metrics.get("eod_vwap")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["eod_vwap"] = metrics.get("eod_vwap")
                    logging.debug(f"{symbol}: EOD VWAP SHORT bounce candidate found. EOD VWAP: {metrics.get('eod_vwap'):.2f}, Current High: {current_candle_data['high']:.2f}")

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
                    logging.debug(
                        f"{symbol}: {ema_label} SHORT bounce candidate found. {ema_label}: {ema_value:.2f}, Std VWAP: {std_vwap:.2f}"
                    )

        # Check for VWAP upper band bounces for longs
        if bounce_type_allowed("vwap_upper_band") and self.is_bounce_type_enabled("vwap_upper_band") and direction == "long" and metrics.get("vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("vwap_1stdev_upper"), "VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["vwap_upper_band"] = metrics.get("vwap_1stdev_upper")
                    logging.debug(f"{symbol}: VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for VWAP lower band bounces for shorts
        if bounce_type_allowed("vwap_lower_band") and self.is_bounce_type_enabled("vwap_lower_band") and direction == "short" and metrics.get("vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("vwap_1stdev_lower"), "VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["vwap_lower_band"] = metrics.get("vwap_1stdev_lower")
                    logging.debug(f"{symbol}: VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for Dynamic VWAP upper band bounces for longs
        if bounce_type_allowed("dynamic_vwap_upper_band") and self.is_bounce_type_enabled("dynamic_vwap_upper_band") and direction == "long" and metrics.get("dynamic_vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("dynamic_vwap_1stdev_upper"), "Dynamic VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("dynamic_vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["dynamic_vwap_upper_band"] = metrics.get("dynamic_vwap_1stdev_upper")
                    logging.debug(f"{symbol}: Dynamic VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('dynamic_vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for Dynamic VWAP lower band bounces for shorts
        if bounce_type_allowed("dynamic_vwap_lower_band") and self.is_bounce_type_enabled("dynamic_vwap_lower_band") and direction == "short" and metrics.get("dynamic_vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("dynamic_vwap_1stdev_lower"), "Dynamic VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("dynamic_vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["dynamic_vwap_lower_band"] = metrics.get("dynamic_vwap_1stdev_lower")
                    logging.debug(f"{symbol}: Dynamic VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('dynamic_vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for EOD VWAP upper band bounces for longs
        if bounce_type_allowed("eod_vwap_upper_band") and self.is_bounce_type_enabled("eod_vwap_upper_band") and direction == "long" and metrics.get("eod_vwap_1stdev_upper") is not None:
            # Check if price respected upper band for consecutive candles
            if check_consecutive_respect(metrics.get("eod_vwap_1stdev_upper"), "EOD VWAP 1SD Upper Band"):
                if abs(current_candle_data["low"] - metrics.get("eod_vwap_1stdev_upper")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["eod_vwap_upper_band"] = metrics.get("eod_vwap_1stdev_upper")
                    logging.debug(f"{symbol}: EOD VWAP 1SD Upper Band LONG bounce candidate found. Upper Band: {metrics.get('eod_vwap_1stdev_upper'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        # Check for EOD VWAP lower band bounces for shorts
        if bounce_type_allowed("eod_vwap_lower_band") and self.is_bounce_type_enabled("eod_vwap_lower_band") and direction == "short" and metrics.get("eod_vwap_1stdev_lower") is not None:
            # Check if price respected lower band for consecutive candles
            if check_consecutive_respect(metrics.get("eod_vwap_1stdev_lower"), "EOD VWAP 1SD Lower Band"):
                if abs(current_candle_data["high"] - metrics.get("eod_vwap_1stdev_lower")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["eod_vwap_lower_band"] = metrics.get("eod_vwap_1stdev_lower")
                    logging.debug(f"{symbol}: EOD VWAP 1SD Lower Band SHORT bounce candidate found. Lower Band: {metrics.get('eod_vwap_1stdev_lower'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Check for previous day high/low bounces if enabled
        if direction == "long" and bounce_type_allowed("prev_day_high") and self.is_bounce_type_enabled("prev_day_high") and metrics.get("prev_high") is not None:
            # Check if price respected previous day high for consecutive candles
            if check_consecutive_respect(metrics.get("prev_high"), "Previous Day High"):
                # Only consider bounce if price respected the level all day
                if abs(current_candle_data["low"] - metrics.get("prev_high")) <= threshold and current_candle_data["close"] > current_candle_data["open"]:
                    ref_levels["prev_day_high"] = metrics.get("prev_high")
                    logging.debug(f"{symbol}: Previous Day High LONG bounce candidate found. Prev High: {metrics.get('prev_high'):.2f}, Current Low: {current_candle_data['low']:.2f}")

        elif direction == "short" and bounce_type_allowed("prev_day_low") and self.is_bounce_type_enabled("prev_day_low") and metrics.get("prev_low") is not None:
            # Check if price respected previous day low for consecutive candles
            if check_consecutive_respect(metrics.get("prev_low"), "Previous Day Low"):
                # Only consider bounce if price respected the level all day
                if abs(current_candle_data["high"] - metrics.get("prev_low")) <= threshold and current_candle_data["close"] < current_candle_data["open"]:
                    ref_levels["prev_day_low"] = metrics.get("prev_low")
                    logging.debug(f"{symbol}: Previous Day Low SHORT bounce candidate found. Prev Low: {metrics.get('prev_low'):.2f}, Current High: {current_candle_data['high']:.2f}")

        # Return None if no reference levels were found, otherwise return the details
        return {"levels": ref_levels, "candle": current_candle_data} if ref_levels else None



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
            "ema_21": ema_21
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
        direction = "long" if symbol in self.longs else "short"
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
                    
                    # For longs: confirm if the current candle's high is greater than the bounce candle's high
                    if direction == "long" and current_candle["high"] > bounce_candle["high"]:
                        levels = bounce_data["levels"]
                        levels_list = list(levels.keys())
                        bounce_msg = f"{symbol}: Bounce confirmed (long) from {levels_list}"
                        if self.gui_callback:
                            self.gui_callback(bounce_msg, "green")
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
                
                    # For shorts: confirm if the current candle's low is less than the bounce candle's low
                    elif direction == "short" and current_candle["low"] < bounce_candle["low"]:
                        levels = bounce_data["levels"]
                        levels_list = list(levels.keys())
                        bounce_msg = f"{symbol}: Bounce confirmed (short) from {levels_list}"
                        if self.gui_callback:
                            self.gui_callback(bounce_msg, "red")
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
                        if candles_waited >= 3:
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
                "bounce_candle": candidate_info["candle"],
                "detection_time": datetime.now(),
                "candles_waited": 0
            }

                        # In the evaluate_bounce_candidate function, where price approaching is logged:
            if LOG_PRICE_APPROACHING:
                # Filter out 10-candle levels for approaching alerts
                approaching_levels = {k: v for k, v in candidate_info["levels"].items() 
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

                    
                    self.log_symbol(symbol, f"Price approaching levels - {level_details}", "approaching")


    def check_removal_conditions(self):
        if not self.has_minimum_candles_completed():
            logging.info("Skipping removal conditions until 6 completed 5-minute candles.")
            return
        for file_name, direction in [(LONGS_FILENAME, "long"), (SHORTS_FILENAME, "short")]:
            tickers = read_tickers(file_name)
            for symbol in tickers:
                if symbol not in self.symbol_metrics:
                    continue
                    
                metrics = self.symbol_metrics[symbol]
                current_price = metrics.get("price")
                eod_vwap = metrics.get("eod_vwap")
                dynamic_vwap = metrics.get("dynamic_vwap")
                prev_high = metrics.get("prev_high")
                prev_low = metrics.get("prev_low")
                
                if current_price is None:
                    continue
                    
                if (eod_vwap is None or dynamic_vwap is None or 
                    (direction == "long" and prev_high is None) or 
                    (direction == "short" and prev_low is None)):
                    continue
                
                if direction == "long":
                    if (current_price < eod_vwap and 
                        current_price < dynamic_vwap and 
                        current_price < prev_high):
                        self.remove_from_watchlist(symbol, direction)
                        if self.gui_callback:
                            removal_msg = f"{symbol} removed from {direction}s watchlist - price below all key levels"
                            self.gui_callback(removal_msg, "blue")
                else:
                    if (current_price > eod_vwap and 
                        current_price > dynamic_vwap and 
                        current_price > prev_low):
                        self.remove_from_watchlist(symbol, direction)
                        if self.gui_callback:
                            removal_msg = f"{symbol} removed from {direction}s watchlist - price above all key levels"
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
            logging.info(f"{symbol} removed from {filename} due to removal condition.")
        except Exception as e:
            logging.error(f"Error removing {symbol} from {filename}: {e}")

    def run_strategy(self):
        last_warning_reset = datetime.now().date()
        
        while True:
            try:
                if not self.ensure_connected():
                    logging.warning("IB not connected; retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                # Reset warning cache daily
                current_date = datetime.now().date()
                if current_date != last_warning_reset:
                    self.warned_symbols.clear()
                    last_warning_reset = current_date
                    logging.info("Daily warning cache reset completed")
                
                self.longs = read_tickers(LONGS_FILENAME)
                self.shorts = read_tickers(SHORTS_FILENAME)
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
                    if sym not in all_symbols or self.atr_cache.get(sym) is None:
                        continue
                    self.request_and_detect_bounce(sym, allowed_bounce_types=enabled_bounce_types)
                    processed_symbols.add(sym)

                # 2) Then scan all remaining symbols for non-EMA-8/15 bounce types.
                for sym in sorted(all_symbols - processed_symbols):
                    if self.atr_cache.get(sym) is None:
                        continue
                    self.request_and_detect_bounce(sym, allowed_bounce_types=non_ema_extreme_bounce_types)
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
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler("trading_bot.log", mode="a")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])
    logger = logging.getLogger()
    logger.addFilter(HistoricalDataFilter())
    logging.getLogger("ibapi").addFilter(HistoricalDataFilter())
    logging.getLogger("ibapi.client").addFilter(HistoricalDataFilter())

    bot = BounceBot(gui_callback=gui_callback)
    bot.set_connection_info("127.0.0.1", 7496, 125)
    bot.connect("127.0.0.1", 7496, clientId=125)
    api_thread = threading.Thread(target=bot.run, daemon=True)
    bot.api_thread = api_thread
    api_thread.start()
    while not bot.connection_status:
        time.sleep(1)
    logging.info("BounceBot is connected. Starting strategy loop...")
    strategy_thread = threading.Thread(target=bot.run_strategy, daemon=True)
    strategy_thread.start()
    return bot

##########################################
# GUI Code using Tkinter
##########################################
# Find and replace the light_grey variable definition with dark theme colors
# Around line 677 in the start_gui() function

def start_gui():
    bounce_queue = queue.Queue()
    approaching_queue = queue.Queue()
    rrs_queue = queue.Queue()
    # Replace light_grey with dark_grey
    dark_grey = "#2E2E2E"  # Dark grey color code
    text_color = "#E0E0E0"  # Light text color for dark background

    def gui_callback(message, tag):
        if tag.startswith("rrs"):
            rrs_queue.put((message, tag))
        elif tag == "approaching" or tag.startswith("approaching_"):
            approaching_queue.put((message, tag))
        elif tag == "candle_line":
            bounce_queue.put((message, tag))
            approaching_queue.put((message, tag))
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

    # Configure tags with new color scheme
    text_area.tag_config("green", foreground="#50FA7B", font=('Courier', 12))  # Green for long message text
    text_area.tag_config("red", foreground="#FF5555", font=('Courier', 12))    # Red for short message text
    text_area.tag_config("pink_symbol", foreground="#FF79C6", font=('Courier', 12, 'bold'))  # Pink for long symbols
    text_area.tag_config("orange_symbol", foreground="#FFB86C", font=('Courier', 12, 'bold'))  # Orange for short symbols
    text_area.tag_config("blue", foreground="#8BE9FD", font=('Courier', 12))           # Light blue
    text_area.tag_config("candle_line", foreground="#BD93F9", overstrike=1)            # Purple

    # Create approaching prices window
    approaching_window = tk.Toplevel(root)
    approaching_window.title("Price Approaching Levels")
    approaching_window.geometry("800x600")
    approaching_window.configure(background=dark_grey)
    
    approaching_frame = tk.Frame(approaching_window, padx=10, pady=10, bg=dark_grey)
    approaching_frame.pack(fill=tk.BOTH, expand=True)
    
    approaching_text = scrolledtext.ScrolledText(
        approaching_frame,
        wrap=tk.WORD,
        width=80,
        height=30,
        font=('Courier', 12),
        state='disabled',
        bg=dark_grey,
        fg=text_color  # Add text color
    )
    approaching_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Configure tags for approaching window
    approaching_text.tag_config("approaching", foreground="#FF79C6", font=('Courier', 12))  # Pink
    approaching_text.tag_config("green", foreground="#50FA7B", font=('Courier', 12))        # Green for long message text
    approaching_text.tag_config("red", foreground="#FF5555", font=('Courier', 12))          # Red for short message text
    approaching_text.tag_config("pink_symbol", foreground="#FF79C6", font=('Courier', 12, 'bold'))  # Pink for long symbols
    approaching_text.tag_config("orange_symbol", foreground="#FFB86C", font=('Courier', 12, 'bold'))  # Orange for short symbols
    approaching_text.tag_config("blue", foreground="#8BE9FD", font=('Courier', 12))         # Light blue
    approaching_text.tag_config("approaching_green", foreground="#50FA7B", font=('Courier', 12))
    approaching_text.tag_config("approaching_red", foreground="#FF5555", font=('Courier', 12))
    approaching_text.tag_config("candle_line", foreground="#BD93F9", overstrike=1)          # Purple

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

    def process_bounce_queue():
        while True:
            try:
                msg, tag = bounce_queue.get_nowait()
                text_area.config(state='normal')
                
                # Special handling for bounce confirmations to color the symbol differently
                if "Bounce confirmed" in msg:
                    parts = msg.split(":", 1)  # Split at first colon to separate symbol from rest
                    if len(parts) == 2:
                        symbol = parts[0].strip()
                        rest = ":" + parts[1]
                        
                        # Determine symbol color based on direction
                        if "(long)" in rest:
                            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - ", tag)
                            text_area.insert(tk.END, symbol, "pink_symbol")  # Pink symbol for longs
                            text_area.insert(tk.END, rest + "\n", "green")   # Green text for rest of long message
                        elif "(short)" in rest:
                            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - ", tag)
                            text_area.insert(tk.END, symbol, "orange_symbol")  # Orange symbol for shorts
                            text_area.insert(tk.END, rest + "\n", "red")       # Red text for rest of short message
                        else:
                            # Fallback if direction can't be determined
                            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                    else:
                        text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                else:
                    # Standard handling for other messages
                    text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                    
                text_area.config(state='disabled')
                text_area.see(tk.END)
                root.update()
            except queue.Empty:
                break
        root.after(100, process_bounce_queue)

    def process_approaching_queue():
        while True:
            try:
                msg, tag = approaching_queue.get_nowait()
                approaching_text.config(state='normal')
                
                # Special handling for approaching messages to color the symbol differently
                if "Price approaching levels" in msg:
                    parts = msg.split(":", 1)  # Split at first colon to separate symbol from rest
                    if len(parts) == 2:
                        symbol = parts[0].strip()
                        rest = ":" + parts[1]
                        
                        # Determine symbol color based on direction
                        if "(long)" in rest:
                            approaching_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - ", tag)
                            approaching_text.insert(tk.END, symbol, "pink_symbol")  # Pink symbol for longs
                            approaching_text.insert(tk.END, rest + "\n", "approaching_green")  # Green text for rest
                        elif "(short)" in rest:
                            approaching_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - ", tag)
                            approaching_text.insert(tk.END, symbol, "orange_symbol")  # Orange symbol for shorts
                            approaching_text.insert(tk.END, rest + "\n", "approaching_red")  # Red text for rest
                        else:
                            # Fallback if direction can't be determined
                            approaching_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                    else:
                        approaching_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                else:
                    # Standard handling for other messages
                    approaching_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n", tag)
                    
                approaching_text.config(state='disabled')
                approaching_text.see(tk.END)
                approaching_window.update()
            except queue.Empty:
                break
        root.after(100, process_approaching_queue)

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
    approaching_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing just the approaching window
    
    process_bounce_queue()
    process_approaching_queue()
    process_rrs_queue()
    root.mainloop()


##########################################
# Main
##########################################
if __name__ == "__main__":
    print("Starting script...")
    reset_log_files()  # Use the new function instead of reset_log_file()
    print("Log files reset complete.")
    
    # Rest of the code remains the same...

    
    try:
        import argparse
        import sys
        print("Imports successful.")

        parser = argparse.ArgumentParser()
        parser.add_argument("--use_gui", action="store_true", help="Use the Tkinter GUI")
        print("Parser created.")
        
        args = parser.parse_args()
        print(f"Arguments parsed: {args}")
        
        # Determine whether to use GUI based on args or global setting
        use_gui = args.use_gui if args.use_gui else USE_GUI
        print(f"Using GUI: {use_gui}")
        
        if use_gui:
            print("Initializing GUI mode...")
            start_gui()
        else:
            print("Initializing console mode...")
            # Set up logging for console mode
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(console_formatter)

            file_handler = logging.FileHandler("trading_bot.log", mode="a")
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)

            logging.basicConfig(level=logging.DEBUG, handlers=[console_handler, file_handler])
            print("Logging configured.")
            
            logger = logging.getLogger()
            logger.addFilter(HistoricalDataFilter())
            logging.getLogger("ibapi").addFilter(HistoricalDataFilter())
            logging.getLogger("ibapi.client").addFilter(HistoricalDataFilter())
            print("Filters applied.")
            
            # Initialize and run the bot without GUI
            print("Creating BounceBot instance...")
            bot = BounceBot()
            print("Connecting to IB API...")
            bot.connect("127.0.0.1", 7496, clientId=125)
            
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
