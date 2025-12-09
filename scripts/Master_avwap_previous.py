diff --git a/scripts/Master_avwap_previous.py b/scripts/Master_avwap_previous.py
new file mode 100644
index 0000000000000000000000000000000000000000..4f1e0ac5f27ed48c4f2f7d1d524b6de2211f192a
--- /dev/null
+++ b/scripts/Master_avwap_previous.py
@@ -0,0 +1,1047 @@
#!/usr/bin/env python3
import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

import requests
import pandas as pd
import yfinance as yf
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# ============================================================================
# PATHS / CONFIG
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = ROOT_DIR / "logs"
AVWAP_SIGNALS_FILE = DATA_DIR / "avwap_signals.csv"
EVENT_TICKERS_FILE = OUTPUT_DIR / "master_avwap_event_tickers.txt"
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

for d in (DATA_DIR, OUTPUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LONGS_FILE = ROOT_DIR / "longs.txt"
SHORTS_FILE = ROOT_DIR / "shorts.txt"

CURRENT_CACHE_FILE = DATA_DIR / "earnings_cache.json"
PREV_CACHE_FILE = DATA_DIR / "prev_earnings_cache.json"
HISTORY_FILE = DATA_DIR / "master_avwap_history.json"
AI_STATE_FILE = DATA_DIR / "master_avwap_ai_state.json"
D1_FEATURES_FILE = DATA_DIR / "d1_features.csv"
OUTPUT_FILE = OUTPUT_DIR / "master_avwap_events.txt"

API_URL = "https://api.nasdaq.com/api/calendar/earnings?date={date}"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*"
}

MAX_LOOKBACK_DAYS = 130       # Nasdaq earnings scan window
RECENT_DAYS = 10              # if earnings < RECENT_DAYS, use prior one for "current"
ATR_LENGTH = 20
ATR_MULT = 0.05               # eps / push = 0.05 * ATR(20)
HISTORY_DAYS_TO_KEEP = 20     # multi-day context window

# ============================================================================
# LOGGING
# ============================================================================

def configure_logging():
    logger = logging.getLogger()
    if logger.handlers:
        return  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(LOG_DIR / "master_avwap_previous.log", mode="a")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    logger.addHandler(ch)
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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
    if len(closes) < 20:
        return "SIDEWAYS"

    short = closes[-5:]
    long = closes[-20:-5]
    if not long:
        return "SIDEWAYS"

    short_avg = mean(short)
    long_avg = mean(long)
    if long_avg == 0:
        return "SIDEWAYS"

    diff_pct = (short_avg - long_avg) / long_avg * 100
    threshold = 1.0
    if diff_pct > threshold:
        return "UP"
    if diff_pct < -threshold:
        return "DOWN"
    return "SIDEWAYS"


def get_last_daily_row_for_date(daily_rows, last_trade_date):
    target = last_trade_date.isoformat()
    for row in reversed(daily_rows):
        if row["date"] == target:
            return row
    return None

# ============================================================================
# EARNINGS FETCH (NASDAQ + YFINANCE FALLBACK)
# ============================================================================

def fetch_earnings_for_date(date_str: str):
    try:
        resp = requests.get(API_URL.format(date=date_str),
                            headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("rows", []) or []
    except Exception as e:
        logging.warning(f"Failed fetch earnings for {date_str}: {e}")
        time.sleep(0.3)
        return []

def collect_earnings_dates(symbols):
    """
    Return dict: sym -> sorted list of past earnings dates (YYYY-MM-DD), most recent first.
    """
    symbol_dates = {sym: [] for sym in symbols}
    today = datetime.now().date()

    for delta in range(MAX_LOOKBACK_DAYS):
        date = today - timedelta(days=delta)
        rows = fetch_earnings_for_date(date.isoformat())
        time.sleep(0.5)  # throttle
        for row in rows:
            sym = row.get("symbol", "").upper()
            if sym in symbol_dates:
                ds = date.isoformat()
                if ds not in symbol_dates[sym]:
                    symbol_dates[sym].append(ds)

    for sym, dates in symbol_dates.items():
        past = [d for d in dates
                if datetime.fromisoformat(d).date() <= today]
        past.sort(reverse=True)
        symbol_dates[sym] = past

    return symbol_dates

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

def yf_earnings_dates(symbol: str):
    """
    Fallback using yfinance earnings_dates.
    """
    try:
        t = yf.Ticker(symbol)
        ed = t.get_earnings_dates(limit=8)
        ed.index = ed.index.tz_localize(None)
        past = ed[ed.index < pd.Timestamp.today().tz_localize(None)]
        sorted_past = sorted(past.index, reverse=True)
        return [d.date().isoformat() for d in sorted_past]
    except Exception as e:
        logging.warning(f"{symbol}: yfinance earnings lookup failed: {e}")
        return []

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

def fetch_daily_bars(ib: IBApi, symbol: str, days: int) -> pd.DataFrame:
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
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d", errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

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
    eps = ATR_MULT * atr
    push = ATR_MULT * atr
    B, C = df.iloc[-2], df.iloc[-1]
    touched = B.low <= level + eps
    reclaimed = B.close >= level
    confirm = C.close > B.close and C.close >= level + push
    return bool(touched and reclaimed and confirm)

def bounce_down_at_level(df: pd.DataFrame, level: float) -> bool:
    if level is None or pd.isna(level) or len(df) < ATR_LENGTH + 3:
        return False
    atr = get_attr20(df)
    if atr is None:
        return False
    eps = ATR_MULT * atr
    push = ATR_MULT * atr
    B, C = df.iloc[-2], df.iloc[-1]
    touched = B.high >= level - eps
    rejected = B.close <= level
    confirm = C.close < B.close and C.close <= level - push
    return bool(touched and rejected and confirm)

def cross_up_through_level(df: pd.DataFrame, level: float) -> bool:
    if level is None or pd.isna(level) or len(df) < ATR_LENGTH_
