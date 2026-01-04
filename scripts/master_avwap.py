#!/usr/bin/env python3
import os
import time
import json
import logging
import threading
import argparse
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
MASTER_POSITIONS_FILE = OUTPUT_DIR / "master_positions.json"
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
EARNINGS_DATES_CACHE_FILE = DATA_DIR / "earnings_dates_cache.json"
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
BOUNCE_ATR_TOL_PCT = 0.001    # 0.1% of ATR(20) distance allowance for bounces
HISTORY_DAYS_TO_KEEP = 20     # multi-day context window
POSITION_LEVELS = [
    "VWAP",
    "UPPER_1",
    "UPPER_2",
    "UPPER_3",
    "LOWER_1",
    "LOWER_2",
    "LOWER_3",
]

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

    fh = logging.FileHandler(LOG_DIR / "master_avwap.log", mode="a")
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


def load_earnings_date_cache():
    return load_json(EARNINGS_DATES_CACHE_FILE, default={"refreshed_on": None, "data": {}})


def save_earnings_date_cache(cache_obj):
    save_json(EARNINGS_DATES_CACHE_FILE, cache_obj)


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


def load_or_refresh_earnings(symbols):
    """
    Refresh earnings dates at most once per day. Subsequent runs reuse the
    cache unless new symbols are added.
    """
    cache = load_earnings_date_cache()
    cached_date = cache.get("refreshed_on")
    cached_data = cache.get("data", {})

    today_iso = datetime.now().date().isoformat()
    missing_symbols = [s for s in symbols if s not in cached_data]
    cache_stale = cached_date != today_iso

    if cache_stale or missing_symbols:
        logging.info("Refreshing earnings date cache from Nasdaq (once per day)…")
        earnings_data = collect_earnings_dates(symbols)
        cache = {"refreshed_on": today_iso, "data": earnings_data}
        save_earnings_date_cache(cache)
        return earnings_data

    return {sym: cached_data.get(sym, []) for sym in symbols}

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
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df.rename(
        columns={
            date_col: "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        },
        inplace=True,
    )

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.sort_values("datetime")
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def fetch_daily_bars(ib: IBApi, symbol: str, days: int) -> pd.DataFrame:
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
            return df
        logging.warning(f"{symbol}: no daily bars returned from IBKR, falling back to Yahoo.")
    except Exception as e:
        logging.error(f"{symbol}: IBKR daily fetch failed ({e}), falling back to Yahoo.")

    return fetch_daily_bars_from_yahoo(symbol, days)

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


def classify_position_by_band(last_close: float, anchor_meta: dict):
    """Return the band label the last_close sits above (or at)."""
    if last_close is None or not anchor_meta:
        return None

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

    if not level_points:
        return None

    level_points.sort(key=lambda x: x[1])
    category = level_points[0][0]
    for name, level in level_points:
        if last_close >= level:
            category = name
        else:
            break
    return category

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
    eps = BOUNCE_ATR_TOL_PCT * atr
    push = ATR_MULT * atr
    B, C = df.iloc[-2], df.iloc[-1]

    # Same-day touch-and-reclaim (bounce happens on latest bar)
    touched_today = abs(C.low - level) <= eps
    confirm_today = C.close >= level + push
    if touched_today and confirm_today:
        return True

    # Two-day pattern: yesterday touched/reclaimed, today confirms follow-through
    touched = abs(B.low - level) <= eps
    reclaimed = B.close >= level
    confirm = C.close > B.close and C.close >= level + push
    return bool(touched and reclaimed and confirm)

def bounce_down_at_level(df: pd.DataFrame, level: float) -> bool:
    if level is None or pd.isna(level) or len(df) < ATR_LENGTH + 3:
        return False
    atr = get_atr20(df)
    if atr is None:
        return False
    eps = BOUNCE_ATR_TOL_PCT * atr
    push = ATR_MULT * atr
    B, C = df.iloc[-2], df.iloc[-1]

    # Same-day touch-and-reject (bounce happens on latest bar)
    touched_today = abs(C.high - level) <= eps
    confirm_today = C.close <= level - push
    if touched_today and confirm_today:
        return True

    # Two-day pattern: yesterday touched/rejected, today confirms follow-through
    touched = abs(B.high - level) <= eps
    rejected = B.close <= level
    confirm = C.close < B.close and C.close <= level - push
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

    def group_rank(label: str) -> int:
        if label.startswith("PREV_"):
            return 0  # previous AVWAP crosses/bounces first
        if label.startswith("MD_"):
            return 2  # multi-day patterns last
        return 1

    def level_rank(label: str) -> int:
        for idx, token in enumerate(EVENT_LEVEL_SORT_ORDER):
            if token in label:
                return idx
        return len(EVENT_LEVEL_SORT_ORDER)

    return sorted(
        events,
        key=lambda e: (
            group_rank(e[2]),      # prev vs current vs multi-day
            level_rank(e[2]),      # e.g. all UPPER_1 before UPPER_2
            e[0],                  # alphabetical within a level by symbol
            e[2],                  # stable ordering for identical symbols
            e[1],
        ),
    )

# ============================================================================
# MAIN MASTER RUN
# ============================================================================

def run_master():
    longs = load_tickers(LONGS_FILE)
    shorts = load_tickers(SHORTS_FILE)
    symbols = sorted(set(longs + shorts))

    if not symbols:
        logging.warning("No symbols found in longs/shorts lists.")
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
            dates = yf_earnings_dates(sym)
            if dates:
                earnings_data[sym] = dates
                earnings_cache_updated = True

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
        save_earnings_date_cache({"refreshed_on": today_iso, "data": earnings_data})

    if missing_anchors:
        logging.warning(
            "No earnings data found for: " + ", ".join(sorted(missing_anchors))
        )
    logging.info(
        f"Earnings anchors refreshed (current: {refreshed_curr}, previous: {refreshed_prev})."
    )

    ib = IBApi()
    ib.connect("127.0.0.1", 7496, clientId=1003)
    threading.Thread(target=ib.run, daemon=True).start()
    time.sleep(1.5)

    today_run = datetime.now().date()
    events_for_output = []
    csv_rows = []
    feature_rows = []
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

        last_trade_date = df["datetime"].iloc[-1].date()
        dstr = df["datetime"].iloc[-1].strftime("%m/%d")

        logging.info(f"→ Processing {sym} ({side}) with {len(df)} daily bars; last date {last_trade_date}")

        symbol_events_today = []
        symbol_multi_day = []
        current_anchor_meta = None
        prev_anchor_meta = None
        symbol_signal_info = {}

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
                    # crosses (current)
                    if side == "LONG" and cross_up_through_level(df, vwap_c):
                        add_signal("CROSS_UP_VWAP", "CURRENT", curr_iso, vwap_c, sd_c, vwap_c)
                    if side == "SHORT" and cross_down_through_level(df, vwap_c):
                        add_signal("CROSS_DOWN_VWAP", "CURRENT", curr_iso, vwap_c, sd_c, vwap_c)

                    for k in (1, 2, 3):
                        if side == "LONG":
                            lvl = bands_c.get(f"UPPER_{k}")
                            if lvl is None:
                                continue
                            if cross_up_through_level(df, lvl):
                                lbl = f"CROSS_UP_UPPER_{k}"
                                add_signal(lbl, "CURRENT", curr_iso, vwap_c, sd_c, lvl)
                        else:
                            lvl = bands_c.get(f"LOWER_{k}")
                            if lvl is None:
                                continue
                            if cross_down_through_level(df, lvl):
                                lbl = f"CROSS_DOWN_LOWER_{k}"
                                add_signal(lbl, "CURRENT", curr_iso, vwap_c, sd_c, lvl)

                    # bounces (current)
                    if side == "LONG":
                        bounce_tests = [
                            ("BOUNCE_LOWER_1", bands_c["LOWER_1"]),
                            ("BOUNCE_VWAP", vwap_c),
                            ("BOUNCE_UPPER_1", bands_c["UPPER_1"]),
                        ]
                        for lbl, lvl in bounce_tests:
                            if bounce_up_at_level(df, lvl):
                                add_signal(lbl, "CURRENT", curr_iso, vwap_c, sd_c, lvl)
                    else:
                        bounce_tests = [
                            ("BOUNCE_UPPER_1", bands_c["UPPER_1"]),
                            ("BOUNCE_VWAP", vwap_c),
                            ("BOUNCE_LOWER_1", bands_c["LOWER_1"]),
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
                        if bounce_up_at_level(df, bands_p.get("LOWER_1")):
                            add_signal("PREV_BOUNCE_LOWER_1", "PREVIOUS", prev_iso, vwap_p, sd_p, bands_p.get("LOWER_1"))
                        if bounce_up_at_level(df, bands_p.get("UPPER_1")):
                            add_signal("PREV_BOUNCE_UPPER_1", "PREVIOUS", prev_iso, vwap_p, sd_p, bands_p.get("UPPER_1"))
                        if bounce_up_at_level(df, vwap_p):
                            add_signal("PREV_BOUNCE_VWAP", "PREVIOUS", prev_iso, vwap_p, sd_p, vwap_p)
                    else:
                        if bounce_down_at_level(df, bands_p.get("UPPER_1")):
                            add_signal("PREV_BOUNCE_UPPER_1", "PREVIOUS", prev_iso, vwap_p, sd_p, bands_p.get("UPPER_1"))
                        if bounce_down_at_level(df, bands_p.get("LOWER_1")):
                            add_signal("PREV_BOUNCE_LOWER_1", "PREVIOUS", prev_iso, vwap_p, sd_p, bands_p.get("LOWER_1"))
                        if bounce_down_at_level(df, vwap_p):
                            add_signal("PREV_BOUNCE_VWAP", "PREVIOUS", prev_iso, vwap_p, sd_p, vwap_p)

                    # previous crosses
                    if side == "LONG" and cross_up_through_level(df, vwap_p):
                        add_signal("PREV_CROSS_UP_VWAP", "PREVIOUS", prev_iso, vwap_p, sd_p, vwap_p)
                    if side == "SHORT" and cross_down_through_level(df, vwap_p):
                        add_signal("PREV_CROSS_DOWN_VWAP", "PREVIOUS", prev_iso, vwap_p, sd_p, vwap_p)

                    if side == "LONG":
                        for k in (1, 2, 3):
                            lvl = bands_p.get(f"UPPER_{k}")
                            if lvl is None:
                                continue
                            if cross_up_through_level(df, lvl):
                                add_signal(f"PREV_CROSS_UPPER_{k}", "PREVIOUS", prev_iso, vwap_p, sd_p, lvl)
                    else:
                        for k in (1, 2, 3):
                            lvl = bands_p.get(f"LOWER_{k}")
                            if lvl is None:
                                continue
                            if cross_down_through_level(df, lvl):
                                add_signal(f"PREV_CROSS_LOWER_{k}", "PREVIOUS", prev_iso, vwap_p, sd_p, lvl)
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

        for lbl in symbol_events_today:
            record = symbol_signal_info.get(lbl)
            if record:
                csv_rows.append(record)

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
        current_lower_1 = (
            current_anchor_meta.get("bands", {}).get("LOWER_1")
            if current_anchor_meta else None
        )
        current_lower_2 = (
            current_anchor_meta.get("bands", {}).get("LOWER_2")
            if current_anchor_meta else None
        )

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

        current_position = classify_position_by_band(last_close, current_anchor_meta)
        if current_position:
            positions["current"].setdefault(current_position, []).append(sym)

        previous_position = classify_position_by_band(last_close, prev_anchor_meta)
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
            "distance_from_current_vwap": dist_vwap,
            "pct_from_current_vwap": pct_vwap,
            "distance_from_current_upper_1": dist_upper_1,
            "pct_from_current_upper_1": pct_upper_1,
            "distance_from_current_lower_1": dist_lower_1,
            "pct_from_current_lower_1": pct_lower_1,
            "trend_20d": trend_label,
            "has_bounce_event_today": has_bounce_event_today,
        }

        ai_state["symbols"][sym] = symbol_entry

        feature_rows.append({
            "symbol": sym,
            "side": side,
            "last_trade_date": last_trade_date.isoformat(),
            "last_close": last_close,
            "last_volume": last_volume,
            "atr20": atr20,
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
            "events_today": ";".join(symbol_events_today),
        })

        logging.info(
            f"{sym}: events_today={symbol_events_today}, "
            f"multi_day={symbol_multi_day}"
        )

    ib.disconnect()

    if csv_rows:
        df_signals = pd.DataFrame(csv_rows)
        df_signals = df_signals.reindex(columns=AVWAP_CSV_COLUMNS)
        df_signals.sort_values(["run_date", "trade_date", "symbol", "signal_type"], inplace=True)

        write_header = (
            not AVWAP_SIGNALS_FILE.exists()
            or AVWAP_SIGNALS_FILE.stat().st_size == 0
        )
        df_signals.to_csv(
            AVWAP_SIGNALS_FILE,
            mode="a",
            index=False,
            header=write_header,
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

    # write human-readable events file (grouped for easier scanning)
    sorted_events = sort_events_for_output(events_for_output)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
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

    def _event_label_sort_key(label: str):
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

    def _fmt_items(values):
        return ", ".join(sorted(set(values))) if values else "None"

    with open(EVENT_TICKERS_FILE, "w", encoding="utf-8") as f:
        f.write("AVWAP crosses and bounces by event type\n")
        for lbl in sorted(event_buckets.keys(), key=_event_label_sort_key):
            for side in ("LONG", "SHORT"):
                tickers = sorted(set(event_buckets[lbl][side]))
                if not tickers:
                    continue
                display_label = lbl.capitalize()
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

    feature_columns = [
        "symbol",
        "side",
        "last_trade_date",
        "last_close",
        "last_volume",
        "atr20",
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

    logging.info(
        f"Master AVWAP run complete. "
        f"Events: {OUTPUT_FILE}, AI state: {AI_STATE_FILE}, history: {HISTORY_FILE}"
    )

# ============================================================================
# ENTRYPOINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run Master AVWAP scanner")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan and exit (no hourly loop).",
    )
    args = parser.parse_args()

    if args.once:
        run_master()
        return

    logging.info("Starting hourly Master AVWAP loop (once per hour)…")
    while True:
        start = time.time()
        run_master()
        elapsed = time.time() - start
        sleep_seconds = max(0, 3600 - elapsed)
        logging.info(f"Sleeping {int(sleep_seconds)} seconds until next run…")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
