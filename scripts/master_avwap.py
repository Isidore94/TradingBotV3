#!/usr/bin/env python3
import os
import time
import json
import logging
import threading
import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
from statistics import mean
from dataclasses import dataclass, asdict

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    tk = None
    ttk = None
    messagebox = None

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
STDEV_RANGE_FILE = OUTPUT_DIR / "master_avwap_stdev2_3.txt"
EARNINGS_ANCHORS_FILE = DATA_DIR / "earnings_avwap_anchors.csv"

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
MIN_AVG_VOLUME_20D = 1_000_000
MIN_PRICE = 5.0
MIN_MARKET_CAP = 1_000_000_000
MIN_GAP_ATR_MULTIPLE = 1.0
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

def collect_earnings_dates(symbols, fetch_fn=fetch_earnings_for_date, base_sleep=0.5):
    """
    Return dict: sym -> sorted list of past earnings dates (YYYY-MM-DD), most recent first.

    Short-circuits once every symbol has at least one earnings date, and reduces
    throttle after the first discovery to speed up remaining lookups.
    """
    symbol_dates = {sym: [] for sym in symbols}
    today = datetime.now().date()
    pending = set(symbols)

    for delta in range(MAX_LOOKBACK_DAYS):
        date = today - timedelta(days=delta)
        rows = fetch_fn(date.isoformat())
        for row in rows:
            sym = row.get("symbol", "").upper()
            if sym in symbol_dates:
                ds = date.isoformat()
                if ds not in symbol_dates[sym]:
                    symbol_dates[sym].append(ds)
                    pending.discard(sym)

        if not pending:
            logging.info("Collected earnings dates for all symbols; stopping early.")
            break

        sleep_duration = base_sleep if len(pending) == len(symbols) else min(base_sleep, 0.2)
        if sleep_duration:
            time.sleep(sleep_duration)

    for sym, dates in symbol_dates.items():
        past = [d for d in dates
                if datetime.fromisoformat(d).date() <= today]
        past.sort(reverse=True)
        symbol_dates[sym] = past

    return symbol_dates


def dry_run_collect_earnings_dates_short_circuit():
    """
    Dry-run helper to validate collect_earnings_dates short-circuits once all
    symbols have an earnings date. Uses a mocked fetch to avoid network calls.
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

    results = collect_earnings_dates(symbols, fetch_fn=mock_fetch, base_sleep=0)

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

PREVIOUS_GAP_UPS_FILE = OUTPUT_DIR / "previous_gap_ups.csv"
ANCHOR_AVWAP_OUTPUT_FILE = OUTPUT_DIR / "master_anchor_avwap_events.txt"
ANCHOR_AVWAP_SIGNALS_FILE = OUTPUT_DIR / "master_anchor_avwap_signals.csv"
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
    try:
        benchmark = yf.download("SPY", period="10d", interval="1d", progress=False, auto_adjust=False)
    except Exception as exc:
        logging.error(f"Failed to determine last market session from Yahoo: {exc}")
        return None

    if benchmark is None or benchmark.empty:
        return None
    idx = benchmark.index.tz_localize(None) if getattr(benchmark.index, "tz", None) is not None else benchmark.index
    return idx[-1].date()


def get_recent_market_session_dates(lookback_days: int = 1):
    lookback_days = max(1, int(lookback_days))
    period_days = max(10, lookback_days * 4)
    try:
        benchmark = yf.download(
            "SPY",
            period=f"{period_days}d",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        logging.error(f"Failed to determine recent market sessions from Yahoo: {exc}")
        return []

    if benchmark is None or benchmark.empty:
        return []

    idx = benchmark.index.tz_localize(None) if getattr(benchmark.index, "tz", None) is not None else benchmark.index
    unique_dates = []
    seen = set()
    for ts in reversed(idx):
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



def has_weekly_options(symbol: str) -> bool:
    try:
        tkr = yf.Ticker(symbol)
        expirations = [pd.Timestamp(d).date() for d in (tkr.options or [])]
    except Exception:
        return False

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



def evaluate_earnings_gap_candidate(symbol: str, earnings_date, release_session: str):
    tkr = yf.Ticker(symbol)
    try:
        hist = tkr.history(period="4mo", interval="1d", auto_adjust=False)
    except Exception as exc:
        logging.warning(f"{symbol}: failed history download for earnings scan ({exc})")
        return None

    if hist is None or hist.empty or len(hist) < ATR_LENGTH + 5:
        return None

    hist = hist.copy()
    if getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_localize(None)

    hist = hist.rename(columns={c: c.lower() for c in hist.columns})
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in hist.columns:
            return None

    df = hist.reset_index().rename(columns={"Date": "datetime", "date": "datetime"})
    if "datetime" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["trade_date"] = df["datetime"].dt.date

    earnings_idx = df.index[df["trade_date"] == earnings_date]
    if earnings_idx.empty:
        return None

    release_session = (release_session or "unknown").lower()
    if release_session == "amc":
        gap_idx = int(earnings_idx[0]) + 1
    else:
        gap_idx = int(earnings_idx[0])

    if gap_idx <= 0 or gap_idx >= len(df):
        return None

    gap_row = df.iloc[gap_idx]
    anchor_row = df.iloc[gap_idx - 1]
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

    info = _get_info_with_fallbacks(tkr)
    market_cap = _coerce_int(
        info.get("marketCap")
        or info.get("market_cap")
        or getattr(getattr(tkr, "fast_info", {}), "get", lambda *_: None)("marketCap")
    )

    if last_price < MIN_PRICE:
        return None
    if avg_vol_20 < MIN_AVG_VOLUME_20D:
        return None
    if market_cap is None or market_cap < MIN_MARKET_CAP:
        return None
    if not has_weekly_options(symbol):
        return None
    if gap_atr_multiple < MIN_GAP_ATR_MULTIPLE:
        return None

    return EarningsGapAnchorCandidate(
        ticker=symbol,
        anchor_date=anchor_row["trade_date"].isoformat(),
        side="LONG",
        gap_date=gap_row["trade_date"].isoformat(),
        earnings_date=earnings_date.isoformat(),
        release_session=release_session,
        gap_atr_multiple=round(gap_atr_multiple, 3),
        price=round(last_price, 3),
        avg_volume20=avg_vol_20,
        market_cap=market_cap,
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

    added = append_anchor_candidates(candidates)
    logging.info(
        f"Earnings gap anchor scan complete. Sessions={len(sessions)}, qualified={len(candidates)}, "
        f"newly_added={added}, "
        f"file={EARNINGS_ANCHORS_FILE}"
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


def normalize_side(value: str) -> str:
    raw = str(value or "").strip().upper()
    return "SHORT" if raw == "SHORT" else "LONG"


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
    eps = BOUNCE_ATR_TOL_PCT * atr
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
        f.write("AVWAP 2nd-3rd stdev multi-day range + 2nd stdev crosses\n")
        f.write(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        long_range = ", ".join(sorted(set(range_hits.get("long", [])))) or "None"
        short_range = ", ".join(sorted(set(range_hits.get("short", [])))) or "None"
        f.write("Traded between 2nd and 3rd stdev for >= 4 days (current anchors)\n")
        f.write(f"Longs (UPPER_2 to UPPER_3): {long_range}\n")
        f.write(f"Shorts (LOWER_3 to LOWER_2): {short_range}\n\n")

        long_cross = ", ".join(sorted(set(cross_hits.get("long", [])))) or "None"
        short_cross = ", ".join(sorted(set(cross_hits.get("short", [])))) or "None"
        f.write("Crosses through 2nd stdev (current anchors)\n")
        f.write(f"Longs crossing up UPPER_2: {long_cross}\n")
        f.write(f"Shorts crossing down LOWER_2: {short_cross}\n")



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

    ib = IBApi()
    ib.connect("127.0.0.1", 7496, clientId=1004)
    threading.Thread(target=ib.run, daemon=True).start()
    time.sleep(1.0)

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
        avwap, stdev, bands = levels
        last_trade_date = df["datetime"].iloc[-1].date().isoformat()

        named_levels = {
            "AVWAP": avwap,
            "UPPER_1": bands.get("UPPER_1"),
            "LOWER_1": bands.get("LOWER_1"),
            "UPPER_2": bands.get("UPPER_2"),
            "LOWER_2": bands.get("LOWER_2"),
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
            if bounce_up_at_level(df, bands.get("LOWER_1")):
                add_anchor_event("BOUNCE_UP_LOWER_1", "LOWER_1")
            if bounce_up_at_level(df, bands.get("UPPER_1")):
                add_anchor_event("BOUNCE_UP_UPPER_1", "UPPER_1")
            if cross_up_through_level(df, bands.get("UPPER_1")):
                add_anchor_event("CROSS_UP_UPPER_1", "UPPER_1")
            if cross_down_through_level(df, bands.get("LOWER_1")):
                add_anchor_event("CROSS_DOWN_LOWER_1", "LOWER_1")
        else:
            if bounce_down_at_level(df, avwap):
                add_anchor_event("BOUNCE_DOWN_AVWAP", "AVWAP")
            if bounce_down_at_level(df, bands.get("UPPER_1")):
                add_anchor_event("BOUNCE_DOWN_UPPER_1", "UPPER_1")
            if bounce_down_at_level(df, bands.get("LOWER_1")):
                add_anchor_event("BOUNCE_DOWN_LOWER_1", "LOWER_1")
            if cross_down_through_level(df, bands.get("LOWER_1")):
                add_anchor_event("CROSS_DOWN_LOWER_1", "LOWER_1")
            if cross_up_through_level(df, bands.get("UPPER_1")):
                add_anchor_event("CROSS_UP_UPPER_1", "UPPER_1")

    ib.disconnect()

    now_iso = datetime.now().isoformat(timespec="seconds")
    if events:
        lines = [f"Anchor AVWAP events generated at {now_iso}", "=" * 80]
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
        ANCHOR_AVWAP_OUTPUT_FILE.write_text(f"Anchor AVWAP scan completed at {now_iso}. No events.\n", encoding="utf-8")

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


def run_master():
    longs = load_tickers(LONGS_FILE)
    shorts = load_tickers(SHORTS_FILE)
    symbols = sorted(set(longs + shorts))

    if not symbols:
        logging.warning("No symbols found in longs/shorts lists. Running anchor-watchlist scan only.")
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

        last_trade_date = df["datetime"].iloc[-1].date()
        dstr = df["datetime"].iloc[-1].strftime("%m/%d")

        logging.info(f"-> Processing {sym} ({side}) with {len(df)} daily bars; last date {last_trade_date}")

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

        if current_anchor_meta:
            if side == "LONG":
                if closes_between_bands(df, current_upper_2, current_upper_3, 4):
                    stdev_range_hits["long"].append(sym)
                if cross_up_through_level(df, current_upper_2):
                    stdev_cross_hits["long"].append(sym)
            else:
                if closes_between_bands(df, current_lower_3, current_lower_2, 4):
                    stdev_range_hits["short"].append(sym)
                if cross_down_through_level(df, current_lower_2):
                    stdev_cross_hits["short"].append(sym)

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

    write_stdev_range_report(STDEV_RANGE_FILE, stdev_range_hits, stdev_cross_hits)

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
        self.ticker_var = tk.StringVar()
        self.anchor_var = tk.StringVar()
        self.side_var = tk.StringVar(value="LONG")
        self.notes_var = tk.StringVar()
        self.earnings_lookback_var = tk.IntVar(value=1)

        self._build_layout()
        self.refresh_table()
        self.refresh_avwap_output_view()
        self.refresh_anchor_output_view()

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
        for widget in (self.avwap_text, self.anchor_scan_text):
            widget.configure(
                bg=GUI_DARK_INPUT,
                fg=GUI_DARK_TEXT,
                insertbackground=GUI_DARK_TEXT,
                selectbackground="#4A4A4A",
                selectforeground=GUI_DARK_TEXT,
                highlightbackground=GUI_DARK_BG,
                highlightcolor=GUI_DARK_PANEL,
            )

    def _build_layout(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=10, pady=8)

        ttk.Label(toolbar, text="Earnings lookback days:").pack(side="left", padx=(0, 4))
        lookback_spin = ttk.Spinbox(
            toolbar,
            from_=1,
            to=20,
            width=4,
            textvariable=self.earnings_lookback_var,
        )
        lookback_spin.pack(side="left", padx=(0, 10))

        ttk.Button(toolbar, text="Run Earnings Gap Scan", command=self.run_earnings_scan).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Run AVWAP Scan Once", command=self.run_master_once).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Refresh Table", command=self.refresh_table).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Refresh AVWAP Output", command=self.refresh_avwap_output_view).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Run Anchor Watchlist Scan", command=self.run_anchor_scan_once).pack(side="left", padx=4)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=8)

        anchors_tab = ttk.Frame(self.notebook)
        self.notebook.add(anchors_tab, text="Earnings Anchors")

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

        table_frame = ttk.Frame(anchors_tab)
        table_frame.pack(fill="both", expand=True, padx=0, pady=0)

        self.table = ttk.Treeview(table_frame, columns=EARNINGS_ANCHOR_COLUMNS, show="headings", style="Dark.Treeview")
        for col in EARNINGS_ANCHOR_COLUMNS:
            self.table.heading(col, text=col)
            width = 120 if col not in {"notes"} else 260
            self.table.column(col, width=width, anchor="w")

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=yscroll.set)

        self.table.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        avwap_tab = ttk.Frame(self.notebook)
        self.notebook.add(avwap_tab, text="AVWAP Scan Output")

        self.avwap_text = tk.Text(avwap_tab, wrap="word", font=("Courier New", 10))
        self.avwap_text.pack(side="left", fill="both", expand=True)
        output_scroll = ttk.Scrollbar(avwap_tab, orient="vertical", command=self.avwap_text.yview)
        self.avwap_text.configure(yscrollcommand=output_scroll.set)
        output_scroll.pack(side="right", fill="y")

        anchor_scan_tab = ttk.Frame(self.notebook)
        self.notebook.add(anchor_scan_tab, text="Anchor AVWAP Output")
        self.anchor_scan_text = tk.Text(anchor_scan_tab, wrap="word", font=("Courier New", 10))
        self.anchor_scan_text.pack(side="left", fill="both", expand=True)
        anchor_output_scroll = ttk.Scrollbar(anchor_scan_tab, orient="vertical", command=self.anchor_scan_text.yview)
        self.anchor_scan_text.configure(yscrollcommand=anchor_output_scroll.set)
        anchor_output_scroll.pack(side="right", fill="y")

        self._apply_dark_theme_to_text_widgets()

        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill="x", padx=10, pady=(0, 10))

    def _read_text_file(self, path: Path):
        if not path.exists():
            return f"[Missing file] {path.name}"
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            return f"[Error reading {path.name}] {exc}"

    def refresh_avwap_output_view(self):
        top_section = self._read_text_file(EVENT_TICKERS_FILE)
        bottom_section = self._read_text_file(STDEV_RANGE_FILE)

        combined = (
            "MASTER AVWAP EVENT TICKERS\n"
            + "=" * 80
            + "\n"
            + (top_section or "No event tickers output yet.")
            + "\n\n"
            + "MASTER AVWAP STDEV 2-3 OUTPUT\n"
            + "=" * 80
            + "\n"
            + (bottom_section or "No stdev output yet.")
            + "\n"
        )

        self.avwap_text.configure(state="normal")
        self.avwap_text.delete("1.0", tk.END)
        self.avwap_text.insert("1.0", combined)
        self.avwap_text.configure(state="normal")

    def refresh_anchor_output_view(self):
        text = self._read_text_file(ANCHOR_AVWAP_OUTPUT_FILE)
        self.anchor_scan_text.configure(state="normal")
        self.anchor_scan_text.delete("1.0", tk.END)
        self.anchor_scan_text.insert("1.0", text or "No anchor AVWAP output yet.")
        self.anchor_scan_text.configure(state="normal")

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
                self.root.after(0, lambda: self.status_var.set(f"Error: {exc}"))

        threading.Thread(target=_task, daemon=True).start()

    def run_earnings_scan(self):
        try:
            lookback_days = max(1, int(self.earnings_lookback_var.get()))
        except Exception:
            lookback_days = 1
            self.earnings_lookback_var.set(1)

        self.notebook.select(0)
        self._run_background(
            lambda: scan_last_session_earnings_for_anchors(lookback_days=lookback_days),
            f"Running earnings gap anchor scan (last {lookback_days} session(s))...",
            "Earnings gap anchor scan complete.",
            done_callback=lambda: self.notebook.select(0),
        )

    def run_master_once(self):
        self.notebook.select(1)
        self._run_background(
            run_master,
            "Running full Master AVWAP scan...",
            "Master AVWAP scan complete.",
            done_callback=lambda: (self.refresh_avwap_output_view(), self.refresh_anchor_output_view()),
        )

    def run_anchor_scan_once(self):
        self.notebook.select(2)
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
        help="Scan the last market session for earnings gap anchors and update data file.",
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
