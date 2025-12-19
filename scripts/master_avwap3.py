#!/usr/bin/env python3
import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta

import requests
import pandas as pd
import yfinance as yf
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# ── Configuration ────────────────────────────────────────────────
LONGS_FILE          = "longs.txt"
SHORTS_FILE         = "shorts.txt"
EARNINGS_CACHE_FILE = "earnings_cache.json"
LOG_FILE            = "combined_avwap.txt"

API_URL = "https://api.nasdaq.com/api/calendar/earnings?date={date}"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*"
}

MAX_LOOKBACK_DAYS   = 150       # Nasdaq earnings scan window
RECENT_DAYS         = 10        # If most recent earnings < RECENT_DAYS, use previous
FETCH_INTERVAL      = 45 * 60   # seconds between runs

# ── Output Filters ──────────────────────────────────────────────
OUTPUT_TIER1                = True
OUTPUT_TIER2                = True
OUTPUT_TIER3                = False
OUTPUT_VWAP                 = True
OUTPUT_CROSS_UPS_LONG       = True
OUTPUT_CROSS_DOWNS_SHORT    = True
OUTPUT_BOUNCES              = True

# ── Bounce Sensitivity (ATR-based) ──────────────────────────────
ATR_LENGTH        = 20
ATR_MULT          = 0.05    # eps/push = 0.05 * ATR(20)

# ── Logging Setup ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# ── Utility: Load tickers ───────────────────────────────────────
def load_tickers_from_file(path: str):
    if not os.path.exists(path):
        logging.warning(f"Ticker file not found: {path}")
        return []
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            val = line.strip()
            if not val or val.upper().startswith("SYMBOLS FROM TC2000"):
                continue
            tickers.append(val.upper())
    return tickers

# ── Earnings Date Cache ─────────────────────────────────────────
def load_earnings_cache():
    def normalize_entry(entry):
        """Normalize cache entry to a dict with sorted date list."""
        if isinstance(entry, str):
            dates = [entry]
            return {"dates": sorted(set(dates), reverse=True)}

        if isinstance(entry, dict):
            dates = entry.get("dates") or []
            if isinstance(dates, str):
                dates = [dates]
            dates = [d for d in dates if isinstance(d, str)]
            dates = sorted(set(dates), reverse=True)
            out = {"dates": dates}
            if entry.get("current"):
                out["current"] = entry.get("current")
            if entry.get("previous"):
                out["previous"] = entry.get("previous")
            return out

        return {"dates": []}

    if os.path.exists(EARNINGS_CACHE_FILE):
        with open(EARNINGS_CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                raw = json.load(f)
                return {sym: normalize_entry(val) for sym, val in raw.items()}
            except json.JSONDecodeError:
                logging.warning("Earnings cache file is corrupt; starting fresh.")
                return {}
    return {}

def save_earnings_cache(cache: dict):
    with open(EARNINGS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

# ── Nasdaq Earnings Lookup ──────────────────────────────────────
def fetch_earnings_for_date(date_str: str):
    try:
        resp = requests.get(API_URL.format(date=date_str),
                            headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("rows", []) or []
    except Exception as e:
        logging.warning(f"Failed fetch earnings for {date_str}: {e}")
        time.sleep(0.5)
        return []

def collect_earnings_dates(symbols):
    symbol_dates = {sym: [] for sym in symbols}
    today = datetime.now().date()

    for delta in range(MAX_LOOKBACK_DAYS):
        date = today - timedelta(days=delta)
        rows = fetch_earnings_for_date(date.isoformat())
        time.sleep(1.0)  # throttle
        for row in rows:
            sym = row.get("symbol", "").upper()
            if sym in symbol_dates:
                ds = date.isoformat()
                if ds not in symbol_dates[sym]:
                    symbol_dates[sym].append(ds)

        if all(symbol_dates[s] for s in symbols):
            break

    # sort dates most recent first
    for sym in symbol_dates:
        symbol_dates[sym].sort(reverse=True)
    return symbol_dates

# ── Choose best earnings date ───────────────────────────────────
def select_best_date(dates):
    if not dates:
        return None
    today = datetime.now().date()
    most = datetime.fromisoformat(dates[0]).date()
    if (today - most).days <= RECENT_DAYS and len(dates) > 1:
        return datetime.fromisoformat(dates[1]).date()
    return most


def pick_current_earnings_anchor(dates):
    """
    Dates: ISO strings desc (most recent first).
    If the most recent earnings date is within RECENT_DAYS, use the second
    most recent to avoid fresh-gap ambiguity; otherwise use the most recent.
    """
    if not dates:
        return None

    today = datetime.now().date()
    first_date = datetime.fromisoformat(dates[0]).date()
    if (today - first_date).days <= RECENT_DAYS and len(dates) > 1:
        return datetime.fromisoformat(dates[1]).date()
    return first_date


def pick_previous_earnings_anchor(dates):
    """Second most recent past earnings date, if available."""
    if len(dates) < 2:
        return None
    return datetime.fromisoformat(dates[1]).date()

# ── yfinance fallback ───────────────────────────────────────────
def get_anchor_date(symbol: str, cache: dict):
    try:
        t = yf.Ticker(symbol)
        ed = t.get_earnings_dates(limit=8)
        ed.index = ed.index.tz_localize(None)
        past = ed[ed.index < pd.Timestamp.today().tz_localize(None)]
        if not past.empty:
            chosen = past.index.max().date()
            cache[symbol] = chosen.isoformat()
            logging.info(f"Cached via yfinance {symbol} -> {chosen}")
            return chosen
    except Exception as e:
        logging.warning(f"yfinance lookup failed for {symbol}: {e}")
    return None


def get_anchor_dates(symbol: str):
    """Return list of past earnings dates (ISO strings) via yfinance fallback."""
    try:
        t = yf.Ticker(symbol)
        ed = t.get_earnings_dates(limit=8)
        ed.index = ed.index.tz_localize(None)
        past = ed[ed.index < pd.Timestamp.today().tz_localize(None)]
        if not past.empty:
            sorted_past = sorted(past.index, reverse=True)
            return [d.date().isoformat() for d in sorted_past]
    except Exception as e:
        logging.warning(f"yfinance lookup failed for {symbol}: {e}")
    return []

# ── IBKR API Wrapper ────────────────────────────────────────────
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

# ── Contract Helper ─────────────────────────────────────────────
def create_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol   = symbol
    c.secType  = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c

# ── Fetch Daily Bars ────────────────────────────────────────────
def fetch_daily_bars(ib: IBApi, symbol: str, days: int) -> pd.DataFrame:
    reqId = int(time.time() * 1000) % (2**31 - 1)
    ib.data[reqId] = []
    ib.ready[reqId] = False

    if days > 365:
        dur = f"{max(1, days // 365)} Y"
    else:
        dur = f"{max(2, days)} D"

    ib.reqHistoricalData(
        reqId=reqId,
        contract=create_contract(symbol),
        endDateTime="",
        durationStr=dur,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
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

# ── AVWAP + Bands ───────────────────────────────────────────────
def calc_anchored_vwap_bands(df: pd.DataFrame, anchor_idx: int):
    """
    Compute anchored VWAP + volume-weighted stdev + 1/2/3σ bands from anchor_idx → end.
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

# ── ATR(20) Helper ─────────────────────────────────────────────
def get_atr20(df: pd.DataFrame, length: int = ATR_LENGTH):
    """
    Standard True Range ATR(20).
    Needs at least length+1 rows.
    """
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

# ── Bounce Helpers (ATR-based) ──────────────────────────────────
def bounce_up_at_level(df: pd.DataFrame, level: float, atr: float) -> bool:
    """
    Long bounce:
      eps  = 0.05 * ATR20
      push = 0.05 * ATR20
      B.low <= level + eps
      B.close >= level
      C.close > B.close and C.close >= level + push
    """
    if atr is None or level is None or pd.isna(level) or len(df) < ATR_LENGTH + 3:
        return False

    eps = ATR_MULT * atr
    push = ATR_MULT * atr

    A, B, C = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    touched = B.low <= level + eps
    reclaimed = B.close >= level
    confirm = C.close > B.close and C.close >= level + push

    return bool(touched and reclaimed and confirm)

def bounce_down_at_level(df: pd.DataFrame, level: float, atr: float) -> bool:
    """
    Short bounce (rejection):
      eps  = 0.05 * ATR20
      push = 0.05 * ATR20
      B.high >= level - eps
      B.close <= level
      C.close < B.close and C.close <= level - push
    """
    if atr is None or level is None or pd.isna(level) or len(df) < ATR_LENGTH + 3:
        return False

    eps = ATR_MULT * atr
    push = ATR_MULT * atr

    A, B, C = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    touched = B.high >= level - eps
    rejected = B.close <= level
    confirm = C.close < B.close and C.close <= level - push

    return bool(touched and rejected and confirm)

# ── Bounce Detection Wrapper ────────────────────────────────────
def detect_bounces_for_symbol(sym: str,
                              df: pd.DataFrame,
                              vwap: float,
                              bands: dict,
                              is_long: bool,
                              is_short: bool,
                              prefix: str = "",
                              atr: float | None = None):
    """
    Longs:
      - BOUNCE_LOWER_2, BOUNCE_LOWER_1, BOUNCE_VWAP, BOUNCE_UPPER_1
    Shorts:
      - BOUNCE_UPPER_2, BOUNCE_UPPER_1, BOUNCE_VWAP, BOUNCE_LOWER_1
    All using ATR(20)-based touch + push.
    """
    results = []
    if df is None or df.empty or len(df) < ATR_LENGTH + 3:
        return results

    label_prefix = f"{prefix}_" if prefix else ""

    atr = atr if atr is not None else get_atr20(df)
    if atr is None:
        return results

    last_date = df.iloc[-1]["datetime"].date()
    dstr = last_date.strftime("%m/%d")

    u1 = bands.get("UPPER_1")
    u2 = bands.get("UPPER_2")
    l1 = bands.get("LOWER_1")
    l2 = bands.get("LOWER_2")

    # Longs
    if is_long:
        if l2 is not None and bounce_up_at_level(df, l2, atr):
            results.append((sym, dstr, f"{label_prefix}BOUNCE_LOWER_2", "LONG"))
        if l1 is not None and bounce_up_at_level(df, l1, atr):
            results.append((sym, dstr, f"{label_prefix}BOUNCE_LOWER_1", "LONG"))
        if vwap is not None and not pd.isna(vwap) and bounce_up_at_level(df, vwap, atr):
            results.append((sym, dstr, f"{label_prefix}BOUNCE_VWAP", "LONG"))
        if u1 is not None and bounce_up_at_level(df, u1, atr):
            results.append((sym, dstr, f"{label_prefix}BOUNCE_UPPER_1", "LONG"))

    # Shorts
    if is_short:
        if u2 is not None and bounce_down_at_level(df, u2, atr):
            results.append((sym, dstr, f"{label_prefix}BOUNCE_UPPER_2", "SHORT"))
        if u1 is not None and bounce_down_at_level(df, u1, atr):
            results.append((sym, dstr, f"{label_prefix}BOUNCE_UPPER_1", "SHORT"))
        if vwap is not None and not pd.isna(vwap) and bounce_down_at_level(df, vwap, atr):
            results.append((sym, dstr, f"{label_prefix}BOUNCE_VWAP", "SHORT"))
        # LOWER_1 as resistance after breakdown
        if l1 is not None and bounce_down_at_level(df, l1, atr):
            results.append((sym, dstr, f"{label_prefix}BOUNCE_LOWER_1", "SHORT"))

    return results

# ── Single Run ──────────────────────────────────────────────────
def run_once():
    longs  = load_tickers_from_file(LONGS_FILE)
    shorts = load_tickers_from_file(SHORTS_FILE)
    symbols = sorted(set(longs + shorts))

    if not symbols:
        logging.warning("No symbols found.")
        return

    cache = load_earnings_cache()

    # Fill cache via Nasdaq for uncached
    uncached = [s for s in symbols if s not in cache or not cache[s].get("dates")]
    if uncached:
        logging.info(f"Fetching earnings for {len(uncached)} uncached symbols…")
        all_dates = collect_earnings_dates(uncached)
        for s, dates in all_dates.items():
            past_dates = sorted(dates, reverse=True)
            if past_dates:
                cache[s] = {"dates": past_dates}

    # IB connection
    ib = IBApi()
    ib.connect("127.0.0.1", 7496, clientId=999)
    threading.Thread(target=ib.run, daemon=True).start()
    time.sleep(1.5)

    today = datetime.now().date()

    # Buckets
    tier3 = []
    tier2 = []
    tier1 = []
    vwap_crosses = []
    cross_ups_long = []
    cross_downs_short = []
    bounces = []
    previous_crosses = []
    previous_bounces = []

    for sym in symbols:
        is_long = sym in longs
        is_short = sym in shorts
        logging.info(f"→ Processing {sym} ({'LONG' if is_long else 'SHORT' if is_short else 'NA'})")

        # Anchor dates from cache or yfinance
        cache_entry = cache.get(sym, {})
        dates = cache_entry.get("dates", [])
        if not dates:
            dates = get_anchor_dates(sym)
            if dates:
                cache[sym] = {"dates": dates}

        if not dates:
            logging.warning(f"No earnings date for {sym}")
            continue

        dates = sorted(set(dates), reverse=True)
        current_anchor = pick_current_earnings_anchor(dates)
        previous_anchor = pick_previous_earnings_anchor(dates)

        cache[sym] = {"dates": dates}
        if current_anchor:
            cache[sym]["current"] = current_anchor.isoformat()
        if previous_anchor:
            cache[sym]["previous"] = previous_anchor.isoformat()

        anchors = [d for d in (current_anchor, previous_anchor) if d]
        if not anchors:
            logging.warning(f"No valid anchors for {sym}")
            continue

        days = max(ATR_LENGTH + 3, max((today - d).days for d in anchors) + 3)
        df = fetch_daily_bars(ib, sym, days)
        if df.empty:
            logging.warning(f"No price data for {sym}")
            continue

        idxs = df.index[df["datetime"].dt.date == current_anchor]
        if idxs.empty:
            logging.warning(f"No candle on earnings date {current_anchor} for {sym}")
            continue
        anchor_idx = int(idxs[0])

        if len(df) - anchor_idx < 3:
            logging.warning(f"Not enough bars after anchor for {sym}")
            continue

        vwap, sd, bands = calc_anchored_vwap_bands(df, anchor_idx)
        if pd.isna(vwap) or pd.isna(sd) or not bands:
            logging.warning(f"NaN bands for {sym}, skipping.")
            continue

        last_row  = df.iloc[-1]
        last_date = last_row["datetime"].date()
        close     = last_row["close"]
        dstr      = last_date.strftime("%m/%d")
        atr       = get_atr20(df)

        # ── Tier classification ────────────────────────────────
        if is_long:
            if close > bands["UPPER_3"]:
                tier3.append((sym, dstr, "UPPER_3", "LONG"))
            elif close > bands["UPPER_2"]:
                tier2.append((sym, dstr, "UPPER_2", "LONG"))
            elif close > bands["UPPER_1"]:
                tier1.append((sym, dstr, "UPPER_1", "LONG"))

        if is_short:
            if close < bands["LOWER_3"]:
                tier3.append((sym, dstr, "LOWER_3", "SHORT"))
            elif close < bands["LOWER_2"]:
                tier2.append((sym, dstr, "LOWER_2", "SHORT"))
            elif close < bands["LOWER_1"]:
                tier1.append((sym, dstr, "LOWER_1", "SHORT"))

        # ── VWAP crosses (last 2 days) ─────────────────────────
        recent_dates = sorted(df["datetime"].dt.date.unique())[-2:]
        hits = {d: set() for d in recent_dates}
        levels = {"VWAP": vwap, **bands}

        recent_df = df[df["datetime"].dt.date.isin(recent_dates)]
        for _, row in recent_df.iterrows():
            d = row["datetime"].date()
            for lvl_name, lvl_val in levels.items():
                if pd.notna(lvl_val) and row["low"] <= lvl_val <= row["high"]:
                    hits[d].add(lvl_name)

        for d, touched in hits.items():
            if {"VWAP", "UPPER_1", "LOWER_1"}.issubset(touched):
                continue
            if "VWAP" in touched:
                side = "LONG" if is_long else "SHORT" if is_short else None
                if side:
                    vwap_crosses.append((sym, d.strftime("%m/%d"), "VWAP", side))

        # ── Directional crosses ────────────────────────────────
        if len(df) >= 2:
            prev_close = df.iloc[-2]["close"]
            curr_close = df.iloc[-1]["close"]

            if is_long:
                for k in (1, 2, 3):
                    lvl = bands.get(f"UPPER_{k}")
                    if pd.notna(lvl) and prev_close <= lvl < curr_close:
                        cross_ups_long.append((sym, dstr, f"CROSS_UP_UPPER_{k}", "LONG"))

            if is_short:
                for k in (1, 2, 3):
                    lvl = bands.get(f"LOWER_{k}")
                    if pd.notna(lvl) and prev_close >= lvl > curr_close:
                        cross_downs_short.append((sym, dstr, f"CROSS_DOWN_LOWER_{k}", "SHORT"))

        # ── ATR-based Bounce Detection ─────────────────────────
        if OUTPUT_BOUNCES and (is_long or is_short):
            sym_bounces = detect_bounces_for_symbol(sym, df, vwap, bands, is_long, is_short, atr=atr)
            bounces.extend(sym_bounces)

        # ── Previous anchor AVWAP / bands ─────────────────────
        if previous_anchor:
            prev_idxs = df.index[df["datetime"].dt.date == previous_anchor]
            if prev_idxs.empty:
                logging.warning(f"No candle on previous earnings date {previous_anchor} for {sym}")
            else:
                prev_vwap, prev_sd, prev_bands = calc_anchored_vwap_bands(df, int(prev_idxs[0]))
                if pd.isna(prev_vwap) or pd.isna(prev_sd) or not prev_bands:
                    logging.warning(f"NaN previous bands for {sym}, skipping previous checks.")
                else:
                    if len(df) >= 2:
                        prev_close_val = df.iloc[-2]["close"]
                        curr_close_val = df.iloc[-1]["close"]

                        if is_long:
                            if pd.notna(prev_vwap) and prev_close_val <= prev_vwap < curr_close_val:
                                previous_crosses.append((sym, dstr, "PREV_CROSS_UP_VWAP", "LONG"))
                            for k in (1, 2, 3):
                                lvl = prev_bands.get(f"UPPER_{k}")
                                if pd.notna(lvl) and prev_close_val <= lvl < curr_close_val:
                                    previous_crosses.append((sym, dstr, f"PREV_CROSS_UPPER_{k}", "LONG"))

                        if is_short:
                            if pd.notna(prev_vwap) and prev_close_val >= prev_vwap > curr_close_val:
                                previous_crosses.append((sym, dstr, "PREV_CROSS_DOWN_VWAP", "SHORT"))
                            for k in (1, 2, 3):
                                lvl = prev_bands.get(f"LOWER_{k}")
                                if pd.notna(lvl) and prev_close_val >= lvl > curr_close_val:
                                    previous_crosses.append((sym, dstr, f"PREV_CROSS_LOWER_{k}", "SHORT"))

                    if OUTPUT_BOUNCES and (is_long or is_short):
                        sym_prev_bounces = detect_bounces_for_symbol(
                            sym,
                            df,
                            prev_vwap,
                            prev_bands,
                            is_long,
                            is_short,
                            prefix="PREV",
                            atr=atr,
                        )
                        previous_bounces.extend(sym_prev_bounces)

    # ── Writer: LONGS first then SHORTS ─────────────────────────
    def write_category(f, items):
        if not items:
            return
        for s, d, lvl, side in items:
            if side == "LONG":
                f.write(f"{s},{d},{lvl},{side}\n")
        for s, d, lvl, side in items:
            if side == "SHORT":
                f.write(f"{s},{d},{lvl},{side}\n")

    # ── Write LOG_FILE ─────────────────────────────────────────
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        if OUTPUT_TIER3:
            write_category(f, tier3)
            f.write("\n")
        if OUTPUT_TIER2:
            write_category(f, tier2)
            f.write("\n")
        if OUTPUT_TIER1:
            write_category(f, tier1)
            f.write("\n")
        if OUTPUT_VWAP:
            write_category(f, vwap_crosses)
            f.write("\n")
        if OUTPUT_CROSS_UPS_LONG:
            write_category(f, cross_ups_long)
            f.write("\n")
        if OUTPUT_CROSS_DOWNS_SHORT:
            write_category(f, cross_downs_short)
            f.write("\n")
        if OUTPUT_CROSS_UPS_LONG or OUTPUT_CROSS_DOWNS_SHORT:
            write_category(f, previous_crosses)
            f.write("\n")
        if OUTPUT_BOUNCES:
            write_category(f, bounces)
            f.write("\n")
            write_category(f, previous_bounces)
            f.write("\n")
        f.write(f"Run completed at {datetime.now().strftime('%H:%M:%S')}\n")

    ib.disconnect()
    save_earnings_cache(cache)
    logging.info(f"Run complete. Log: {LOG_FILE}, Cache: {EARNINGS_CACHE_FILE}")

# ── Main Loop ───────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        run_once()
        logging.info(f"Sleeping {FETCH_INTERVAL/60:.0f}m…")
        time.sleep(FETCH_INTERVAL)
