#!/usr/bin/env python3
import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

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

for d in (DATA_DIR, OUTPUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LONGS_FILE = ROOT_DIR / "longs.txt"
SHORTS_FILE = ROOT_DIR / "shorts.txt"

CURRENT_CACHE_FILE = DATA_DIR / "earnings_cache.json"
PREV_CACHE_FILE = DATA_DIR / "prev_earnings_cache.json"
HISTORY_FILE = DATA_DIR / "master_avwap_history.json"
AI_STATE_FILE = DATA_DIR / "master_avwap_ai_state.json"
OUTPUT_FILE = OUTPUT_DIR / "master_avwap_events.txt"

API_URL = "https://api.nasdaq.com/api/calendar/earnings?date={date}"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*"
}

MAX_LOOKBACK_DAYS = 250       # Nasdaq earnings scan window
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
    atr = get_atr20(df)
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

    need_curr = [s for s in symbols if s not in curr_cache]
    need_prev = [s for s in symbols if s not in prev_cache]

    if need_curr or need_prev:
        logging.info(f"Fetching earnings history for {len(set(need_curr+need_prev))} symbols (Nasdaq)…")
        all_dates = collect_earnings_dates(list(set(need_curr + need_prev)))
    else:
        all_dates = {}

    # fill current cache
    for sym in need_curr:
        dates = all_dates.get(sym, [])
        if not dates:
            # try yfinance fallback
            dates = yf_earnings_dates(sym)
        if not dates:
            continue
        anchor = pick_current_earnings_anchor(dates)
        if anchor:
            curr_cache[sym] = anchor.isoformat()
            logging.info(f"{sym}: current anchor -> {anchor}")

    # fill previous cache
    for sym in need_prev:
        dates = all_dates.get(sym, [])
        if not dates:
            dates = yf_earnings_dates(sym)
        if not dates:
            continue
        anchor = pick_previous_earnings_anchor(dates)
        if anchor:
            prev_cache[sym] = anchor.isoformat()
            logging.info(f"{sym}: previous anchor -> {anchor}")

    ib = IBApi()
    ib.connect("127.0.0.1", 7496, clientId=1003)
    threading.Thread(target=ib.run, daemon=True).start()
    time.sleep(1.5)

    today_run = datetime.now().date()
    events_for_output = []
    csv_rows = []
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
                    close = float(df["close"].iloc[-1])

                    # position relative to bands
                    if side == "LONG":
                        if close > bands_c["UPPER_3"]:
                            add_signal("UPPER_3", "CURRENT", curr_iso, vwap_c, sd_c, bands_c["UPPER_3"])
                        elif close > bands_c["UPPER_2"]:
                            add_signal("UPPER_2", "CURRENT", curr_iso, vwap_c, sd_c, bands_c["UPPER_2"])
                        elif close > bands_c["UPPER_1"]:
                            add_signal("UPPER_1", "CURRENT", curr_iso, vwap_c, sd_c, bands_c["UPPER_1"])
                    else:
                        if close < bands_c["LOWER_3"]:
                            add_signal("LOWER_3", "CURRENT", curr_iso, vwap_c, sd_c, bands_c["LOWER_3"])
                        elif close < bands_c["LOWER_2"]:
                            add_signal("LOWER_2", "CURRENT", curr_iso, vwap_c, sd_c, bands_c["LOWER_2"])
                        elif close < bands_c["LOWER_1"]:
                            add_signal("LOWER_1", "CURRENT", curr_iso, vwap_c, sd_c, bands_c["LOWER_1"])

                    # crosses (current)
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
                            ("BOUNCE_LOWER_2", bands_c["LOWER_2"]),
                            ("BOUNCE_LOWER_1", bands_c["LOWER_1"]),
                            ("BOUNCE_VWAP", vwap_c),
                            ("BOUNCE_UPPER_1", bands_c["UPPER_1"]),
                        ]
                        for lbl, lvl in bounce_tests:
                            if bounce_up_at_level(df, lvl):
                                add_signal(lbl, "CURRENT", curr_iso, vwap_c, sd_c, lvl)
                    else:
                        bounce_tests = [
                            ("BOUNCE_UPPER_2", bands_c["UPPER_2"]),
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
                        if bounce_up_at_level(df, bands_p.get("UPPER_1")):
                            add_signal("PREV_BOUNCE_UPPER_1", "PREVIOUS", prev_iso, vwap_p, sd_p, bands_p.get("UPPER_1"))
                    else:
                        if bounce_down_at_level(df, bands_p.get("LOWER_1")):
                            add_signal("PREV_BOUNCE_LOWER_1", "PREVIOUS", prev_iso, vwap_p, sd_p, bands_p.get("LOWER_1"))

                    # previous crosses
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

        ai_state["symbols"][sym] = {
            "side": side,
            "last_trade_date": last_trade_date.isoformat(),
            "current_anchor": current_anchor_meta,
            "previous_anchor": prev_anchor_meta,
            "events_today": symbol_events_today,
            "multi_day_patterns": symbol_multi_day,
            "events_all_for_day": full_event_list,
            "daily_ohlc": daily_ohlc,
        }

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

    # write human-readable events file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s, d, lbl, side in events_for_output:
            f.write(f"{s},{d},{lbl},{side}\n")
        f.write(f"\nRun completed at {datetime.now().strftime('%H:%M:%S')}\n")

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

if __name__ == "__main__":
    run_master()
