#!/usr/bin/env python3
"""Self-sufficient scan universe builder (no more hand-fed ticker lists).

Recreates the TC2000 "weekly options + quality" watchlist locally:

1. Pull the full US listing directory from NASDAQ Trader (nasdaqlisted +
   otherlisted, no API key), dropping ETFs and test issues.
2. Optionally intersect with the CBOE available-weeklys list so only names with
   weekly options survive (falls back gracefully if CBOE is unreachable).
3. Batch-download daily history from yfinance (chunked, cached locally with a
   max age so re-runs are cheap and Google Drive stays untouched) and apply the
   quality screen: price > $5, 20-day average volume > 1M shares, market cap >
   $1B, plus the trend cut of price above/below both the 100- and 200-day SMAs.
4. Write the results to the home folder as plain watchlist files the scanners
   already know how to read:
       universe_all.txt     - passed the base quality screen
       universe_longs.txt   - base + above SMA100 and SMA200
       universe_shorts.txt  - base + below SMA100 and SMA200
   plus data/universe_metadata.csv with the per-symbol metrics.

API-limit scaling: ``--max-symbols`` caps how many names are priced per run
(ranked by dollar volume, so the most liquid survive first), history and market
caps are cached per machine, and everything comes from yfinance -- zero IBKR
pacing budget is spent building the universe.

Run:
    .venv/Scripts/python.exe scripts/universe_builder.py
    .venv/Scripts/python.exe scripts/universe_builder.py --max-symbols 1500 --no-weeklies
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import CACHE_DIR, DATA_DIR, PERSISTENT_DATA_DIR  # noqa: E402

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
CBOE_WEEKLYS_URL = "https://www.cboe.com/available_weeklys/get_csv_download/"

UNIVERSE_CACHE_DIR = CACHE_DIR / "universe"
SYMBOL_DIRECTORY_CACHE = UNIVERSE_CACHE_DIR / "symbol_directory.json"
WEEKLYS_CACHE = UNIVERSE_CACHE_DIR / "cboe_weeklys.json"
MARKET_CAP_CACHE = UNIVERSE_CACHE_DIR / "market_caps.json"
PRICE_HISTORY_CACHE = UNIVERSE_CACHE_DIR / "price_history.parquet"

UNIVERSE_ALL_FILE = PERSISTENT_DATA_DIR / "universe_all.txt"
UNIVERSE_LONGS_FILE = PERSISTENT_DATA_DIR / "universe_longs.txt"
UNIVERSE_SHORTS_FILE = PERSISTENT_DATA_DIR / "universe_shorts.txt"
UNIVERSE_METADATA_FILE = DATA_DIR / "universe_metadata.csv"

SYMBOL_DIRECTORY_MAX_AGE_DAYS = 7
WEEKLYS_MAX_AGE_DAYS = 7
MARKET_CAP_MAX_AGE_DAYS = 7
PRICE_HISTORY_MAX_AGE_HOURS = 20

DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_AVG_VOLUME = 1_000_000
DEFAULT_MIN_MARKET_CAP_M = 1000.0
DEFAULT_MAX_SYMBOLS = 2000
YF_CHUNK_SIZE = 200
HISTORY_PERIOD = "1y"


# ---------------------------------------------------------------------------
# Symbol directory (NASDAQ Trader) + CBOE weeklys
# ---------------------------------------------------------------------------
def parse_symbol_directory(nasdaq_text: str, other_text: str) -> list[str]:
    """Common-stock symbols from the pipe-delimited NASDAQ Trader files.

    Drops ETFs (ETF flag = Y), test issues, and structured-product tickers with
    $ / ^ / ~ in them. Dots (class shares like BRK.B) are kept in Yahoo format
    (BRK-B) so yfinance accepts them directly.
    """
    symbols: set[str] = set()

    def _rows(text: str) -> list[dict]:
        lines = [line for line in (text or "").splitlines() if line and "|" in line and not line.startswith("File Creation")]
        if not lines:
            return []
        reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="|")
        return list(reader)

    for row in _rows(nasdaq_text):
        symbol = str(row.get("Symbol") or "").strip().upper()
        if not symbol or row.get("Test Issue") == "Y" or row.get("ETF") == "Y":
            continue
        symbols.add(symbol)
    for row in _rows(other_text):
        symbol = str(row.get("ACT Symbol") or row.get("NASDAQ Symbol") or "").strip().upper()
        if not symbol or row.get("Test Issue") == "Y" or row.get("ETF") == "Y":
            continue
        symbols.add(symbol)

    cleaned = []
    for symbol in symbols:
        if any(ch in symbol for ch in ("$", "^", "~", "=")):
            continue
        cleaned.append(symbol.replace(".", "-"))
    return sorted(set(cleaned))


def _load_json_cache(path: Path, max_age_days: float) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(str(payload.get("fetched_at")))
        if datetime.now() - fetched <= timedelta(days=max_age_days):
            return payload
    except Exception:
        return None
    return None


def _save_json_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload), encoding="utf-8")


def fetch_all_listed_symbols(*, refresh: bool = False) -> list[str]:
    if not refresh:
        cached = _load_json_cache(SYMBOL_DIRECTORY_CACHE, SYMBOL_DIRECTORY_MAX_AGE_DAYS)
        if cached and cached.get("symbols"):
            return list(cached["symbols"])
    nasdaq_text = requests.get(NASDAQ_LISTED_URL, timeout=30).text
    other_text = requests.get(OTHER_LISTED_URL, timeout=30).text
    symbols = parse_symbol_directory(nasdaq_text, other_text)
    if symbols:
        _save_json_cache(SYMBOL_DIRECTORY_CACHE, {"symbols": symbols})
    return symbols


def parse_weeklys_csv(text: str) -> list[str]:
    """Ticker column from the CBOE available-weeklys CSV (stock/ETF rows)."""
    symbols: set[str] = set()
    for line in (text or "").splitlines():
        cells = [cell.strip().strip('"') for cell in line.split(",")]
        if not cells or not cells[0]:
            continue
        candidate = cells[0].upper()
        # Header/section lines contain spaces or lowercase words; tickers don't.
        if candidate.isalpha() and 1 <= len(candidate) <= 5 and candidate == cells[0]:
            symbols.add(candidate.replace(".", "-"))
    return sorted(symbols)


def fetch_weekly_option_symbols(*, refresh: bool = False) -> list[str]:
    """CBOE weeklys list; empty list means 'unavailable' (callers skip the filter)."""
    if not refresh:
        cached = _load_json_cache(WEEKLYS_CACHE, WEEKLYS_MAX_AGE_DAYS)
        if cached and cached.get("symbols"):
            return list(cached["symbols"])
    try:
        response = requests.get(CBOE_WEEKLYS_URL, timeout=30)
        response.raise_for_status()
        symbols = parse_weeklys_csv(response.text)
    except Exception as exc:
        logging.warning("CBOE weeklys list unavailable (%s); weekly-options filter skipped.", exc)
        return []
    if symbols:
        _save_json_cache(WEEKLYS_CACHE, {"symbols": symbols})
    return symbols


# ---------------------------------------------------------------------------
# Price history + metrics (yfinance, chunked + cached)
# ---------------------------------------------------------------------------
def fetch_price_history(symbols: list[str], *, refresh: bool = False) -> pd.DataFrame:
    """Long-form frame [symbol, datetime, close, volume] for every fetchable symbol."""
    tickers = sorted({str(s or "").strip().upper() for s in symbols if str(s or "").strip()})
    if not tickers:
        return pd.DataFrame(columns=["symbol", "datetime", "close", "volume"])

    if not refresh and PRICE_HISTORY_CACHE.exists():
        age_hours = (time.time() - PRICE_HISTORY_CACHE.stat().st_mtime) / 3600.0
        if age_hours <= PRICE_HISTORY_MAX_AGE_HOURS:
            try:
                cached = pd.read_parquet(PRICE_HISTORY_CACHE)
                if set(tickers) <= set(cached["symbol"].unique()):
                    return cached[cached["symbol"].isin(tickers)]
            except Exception:
                pass

    import yfinance as yf

    parts: list[pd.DataFrame] = []
    for start in range(0, len(tickers), YF_CHUNK_SIZE):
        chunk = tickers[start : start + YF_CHUNK_SIZE]
        logging.info("Universe price fetch %s-%s of %s...", start + 1, start + len(chunk), len(tickers))
        try:
            raw = yf.download(
                tickers=" ".join(chunk),
                period=HISTORY_PERIOD,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as exc:
            logging.warning("Chunk download failed (%s); continuing.", exc)
            continue
        if raw is None or raw.empty:
            continue
        for symbol in chunk:
            try:
                sub = raw[symbol] if isinstance(raw.columns, pd.MultiIndex) else raw
            except (KeyError, TypeError):
                continue
            if sub is None or sub.empty:
                continue
            frame = sub.reset_index().rename(columns={"Date": "datetime", "Close": "close", "Volume": "volume"})
            frame = frame.dropna(subset=["close"])
            if frame.empty:
                continue
            frame["symbol"] = symbol
            parts.append(frame[["symbol", "datetime", "close", "volume"]])
    history = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["symbol", "datetime", "close", "volume"])
    if not history.empty:
        try:
            PRICE_HISTORY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            history.to_parquet(PRICE_HISTORY_CACHE, index=False)
        except Exception as exc:
            logging.debug("Could not cache price history (%s).", exc)
    return history


def compute_universe_metrics(history: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol screen metrics from long-form history: last price, 20d average
    volume, dollar volume, and position vs the 100/200-day SMAs."""
    rows = []
    if history is None or history.empty:
        return pd.DataFrame(
            columns=["symbol", "last_price", "avg_volume_20d", "dollar_volume_20d", "sma_100", "sma_200", "above_sma_100", "above_sma_200"]
        )
    for symbol, group in history.groupby("symbol"):
        work = group.sort_values("datetime")
        closes = pd.to_numeric(work["close"], errors="coerce").dropna()
        volumes = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
        if len(closes) < 20:
            continue
        last_price = float(closes.iloc[-1])
        avg_volume = float(volumes.tail(20).mean())
        sma_100 = float(closes.tail(100).mean()) if len(closes) >= 100 else None
        sma_200 = float(closes.tail(200).mean()) if len(closes) >= 200 else None
        rows.append(
            {
                "symbol": symbol,
                "last_price": last_price,
                "avg_volume_20d": avg_volume,
                "dollar_volume_20d": last_price * avg_volume,
                "sma_100": sma_100,
                "sma_200": sma_200,
                "above_sma_100": bool(sma_100 is not None and last_price >= sma_100),
                "above_sma_200": bool(sma_200 is not None and last_price >= sma_200),
            }
        )
    return pd.DataFrame(rows)


def fetch_market_caps(symbols: list[str], *, refresh: bool = False) -> dict[str, float]:
    """Market caps (millions USD) via yfinance fast_info, cached for a week.

    Only called for screen survivors, so the per-symbol calls stay bounded."""
    cached_payload = None if refresh else _load_json_cache(MARKET_CAP_CACHE, MARKET_CAP_MAX_AGE_DAYS)
    caps: dict[str, float] = dict((cached_payload or {}).get("caps") or {})
    missing = [s for s in symbols if s not in caps]
    if missing:
        import yfinance as yf

        logging.info("Fetching market caps for %s symbol(s)...", len(missing))
        for symbol in missing:
            try:
                cap = yf.Ticker(symbol).fast_info.get("marketCap")
                caps[symbol] = float(cap) / 1e6 if cap else 0.0
            except Exception:
                caps[symbol] = 0.0
        _save_json_cache(MARKET_CAP_CACHE, {"caps": caps})
    return {s: caps.get(s, 0.0) for s in symbols}


# ---------------------------------------------------------------------------
# Screen + outputs
# ---------------------------------------------------------------------------
def apply_universe_screen(
    metrics: pd.DataFrame,
    *,
    min_price: float = DEFAULT_MIN_PRICE,
    min_avg_volume: float = DEFAULT_MIN_AVG_VOLUME,
    market_caps_m: dict[str, float] | None = None,
    min_market_cap_m: float = DEFAULT_MIN_MARKET_CAP_M,
) -> pd.DataFrame:
    """Base quality screen (price / volume / cap); cap skipped when caps missing."""
    if metrics is None or metrics.empty:
        return metrics
    work = metrics.copy()
    work = work[(work["last_price"] > float(min_price)) & (work["avg_volume_20d"] > float(min_avg_volume))]
    if market_caps_m:
        work["market_cap_m"] = work["symbol"].map(lambda s: float(market_caps_m.get(s, 0.0)))
        # Cap of 0 means "unknown" (fetch failed); keep those rather than silently
        # dropping real large-caps on a flaky lookup.
        work = work[(work["market_cap_m"] <= 0.0) | (work["market_cap_m"] > float(min_market_cap_m))]
    return work.reset_index(drop=True)


def _write_watchlist(path: Path, symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")


def build_universe(
    *,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    min_price: float = DEFAULT_MIN_PRICE,
    min_avg_volume: float = DEFAULT_MIN_AVG_VOLUME,
    min_market_cap_m: float = DEFAULT_MIN_MARKET_CAP_M,
    use_weeklies: bool = True,
    refresh: bool = False,
    write_outputs: bool = True,
) -> dict:
    listed = fetch_all_listed_symbols(refresh=refresh)
    logging.info("Listing directory: %s symbols.", len(listed))

    weeklys = fetch_weekly_option_symbols(refresh=refresh) if use_weeklies else []
    if weeklys:
        listed = [s for s in listed if s in set(weeklys)]
        logging.info("Weekly-options filter: %s symbols remain.", len(listed))

    history = fetch_price_history(listed, refresh=refresh)
    metrics = compute_universe_metrics(history)
    logging.info("Priced %s symbols.", len(metrics))

    # Scale to the API budget: most-liquid first, cap the priced universe.
    metrics = metrics.sort_values("dollar_volume_20d", ascending=False)
    if max_symbols and len(metrics) > int(max_symbols):
        metrics = metrics.head(int(max_symbols))

    pre_cap = apply_universe_screen(
        metrics,
        min_price=min_price,
        min_avg_volume=min_avg_volume,
    )
    caps = fetch_market_caps(sorted(pre_cap["symbol"]), refresh=refresh) if not pre_cap.empty else {}
    screened = apply_universe_screen(
        pre_cap,
        min_price=min_price,
        min_avg_volume=min_avg_volume,
        market_caps_m=caps,
        min_market_cap_m=min_market_cap_m,
    )

    all_symbols = sorted(screened["symbol"])
    longs = sorted(screened[screened["above_sma_100"] & screened["above_sma_200"]]["symbol"])
    shorts = sorted(screened[~screened["above_sma_100"] & ~screened["above_sma_200"]]["symbol"])

    if write_outputs:
        _write_watchlist(UNIVERSE_ALL_FILE, all_symbols)
        _write_watchlist(UNIVERSE_LONGS_FILE, longs)
        _write_watchlist(UNIVERSE_SHORTS_FILE, shorts)
        UNIVERSE_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        screened.to_csv(UNIVERSE_METADATA_FILE, index=False)
        logging.info(
            "Universe written: %s total / %s longs / %s shorts -> %s",
            len(all_symbols),
            len(longs),
            len(shorts),
            UNIVERSE_ALL_FILE.parent,
        )
    return {
        "all": all_symbols,
        "longs": longs,
        "shorts": shorts,
        "metrics": screened,
        "weeklys_applied": bool(weeklys),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the self-sufficient scan universe")
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_MAX_SYMBOLS)
    parser.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    parser.add_argument("--min-avg-volume", type=float, default=DEFAULT_MIN_AVG_VOLUME)
    parser.add_argument("--min-market-cap-m", type=float, default=DEFAULT_MIN_MARKET_CAP_M)
    parser.add_argument("--no-weeklies", action="store_true", help="skip the CBOE weekly-options filter")
    parser.add_argument("--refresh", action="store_true", help="ignore local caches")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = build_universe(
        max_symbols=args.max_symbols,
        min_price=args.min_price,
        min_avg_volume=args.min_avg_volume,
        min_market_cap_m=args.min_market_cap_m,
        use_weeklies=not args.no_weeklies,
        refresh=args.refresh,
    )
    print(
        f"Universe: {len(result['all'])} symbols "
        f"({len(result['longs'])} longs / {len(result['shorts'])} shorts); "
        f"weekly-options filter {'applied' if result['weeklys_applied'] else 'skipped'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
