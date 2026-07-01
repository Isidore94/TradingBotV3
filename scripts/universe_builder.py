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

from project_paths import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    PERSISTENT_DATA_DIR,
    UNIVERSE_ALL_FILE,
    UNIVERSE_LONGS_FILE,
    UNIVERSE_SHORTS_FILE,
)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
CBOE_WEEKLYS_URL = "https://www.cboe.com/available_weeklys/get_csv_download/"
# Full CBOE equity/index options directory (~5,300 rows): effectively every
# optionable US stock, since CBOE trades essentially all multi-listed equity
# options. This is the "Optionable Stocks Is True" filter from TC2000.
CBOE_SYMBOL_DIRECTORY_URL = "https://www.cboe.com/us/options/symboldir/?download=csv"

OPTIONS_FILTER_CHOICES = ("optionable", "weeklies", "none")
DEFAULT_OPTIONS_FILTER = "optionable"

UNIVERSE_CACHE_DIR = CACHE_DIR / "universe"
SYMBOL_DIRECTORY_CACHE = UNIVERSE_CACHE_DIR / "symbol_directory.json"
WEEKLYS_CACHE = UNIVERSE_CACHE_DIR / "cboe_weeklys.json"
OPTIONABLE_CACHE = UNIVERSE_CACHE_DIR / "cboe_optionable.json"
MARKET_CAP_CACHE = UNIVERSE_CACHE_DIR / "market_caps.json"
PRICE_HISTORY_CACHE = UNIVERSE_CACHE_DIR / "price_history.parquet"

UNIVERSE_METADATA_FILE = DATA_DIR / "universe_metadata.csv"
# Manual include lists: names the free sources miss (e.g. NYSE tickers absent
# from the NASDAQ Trader directory, or optionable names CBOE's file omits).
# Merged-in symbols persist here so every rebuild re-applies them.
UNIVERSE_LIST_FILES = {
    "all": UNIVERSE_ALL_FILE,
    "longs": UNIVERSE_LONGS_FILE,
    "shorts": UNIVERSE_SHORTS_FILE,
}
UNIVERSE_INCLUDE_FILES = {
    "all": PERSISTENT_DATA_DIR / "universe_include_all.txt",
    "longs": PERSISTENT_DATA_DIR / "universe_include_longs.txt",
    "shorts": PERSISTENT_DATA_DIR / "universe_include_shorts.txt",
}

SYMBOL_DIRECTORY_MAX_AGE_DAYS = 7
WEEKLYS_MAX_AGE_DAYS = 7
OPTIONABLE_MAX_AGE_DAYS = 7
MARKET_CAP_MAX_AGE_DAYS = 7
PRICE_HISTORY_MAX_AGE_HOURS = 20

DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_AVG_VOLUME = 1_000_000
DEFAULT_MIN_MARKET_CAP_M = 1000.0
# Safety valve on how many names get priced/capped per run, ranked most-liquid
# first. The optionable pool intersects to ~4k names, so the default sits above
# that; tighten it only if yfinance starts throttling.
DEFAULT_MAX_SYMBOLS = 5000
YF_CHUNK_SIZE = 200
HISTORY_PERIOD = "1y"
# Politeness pacing so Yahoo never sees a burst worth throttling: a short pause
# between 200-ticker batch downloads, one retry after a longer cool-off if a
# chunk still fails, and a small delay between the per-symbol market-cap calls.
YF_CHUNK_PAUSE_SECONDS = 1.0
YF_CHUNK_RETRY_PAUSE_SECONDS = 20.0
CAP_FETCH_PAUSE_SECONDS = 0.12
# Flush the cap cache periodically so an interrupted first run keeps progress.
CAP_CACHE_FLUSH_EVERY = 100


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


def parse_cboe_symbol_directory(text: str) -> list[str]:
    """'Stock Symbol' column from the CBOE equity/index options directory CSV."""
    symbols: set[str] = set()
    reader = csv.reader(io.StringIO(text or ""))
    header: list[str] | None = None
    symbol_idx = 1
    for cells in reader:
        if not cells:
            continue
        if header is None:
            header = [cell.strip().lower() for cell in cells]
            for idx, name in enumerate(header):
                if "stock symbol" in name:
                    symbol_idx = idx
                    break
            continue
        if symbol_idx >= len(cells):
            continue
        candidate = cells[symbol_idx].strip().upper()
        if candidate and all(ch.isalnum() or ch in ".-" for ch in candidate):
            symbols.add(candidate.replace(".", "-"))
    return sorted(symbols)


def fetch_optionable_symbols(*, refresh: bool = False) -> list[str]:
    """Every optionable stock per the CBOE symbol directory ("Optionable Stocks
    Is True"); empty list means 'unavailable' (callers skip the filter)."""
    if not refresh:
        cached = _load_json_cache(OPTIONABLE_CACHE, OPTIONABLE_MAX_AGE_DAYS)
        if cached and cached.get("symbols"):
            return list(cached["symbols"])
    try:
        response = requests.get(CBOE_SYMBOL_DIRECTORY_URL, timeout=30)
        response.raise_for_status()
        symbols = parse_cboe_symbol_directory(response.text)
    except Exception as exc:
        logging.warning("CBOE symbol directory unavailable (%s); optionable filter skipped.", exc)
        return []
    if symbols:
        _save_json_cache(OPTIONABLE_CACHE, {"symbols": symbols})
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

    def _download_chunk(chunk: list[str]):
        return yf.download(
            tickers=" ".join(chunk),
            period=HISTORY_PERIOD,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )

    parts: list[pd.DataFrame] = []
    for start in range(0, len(tickers), YF_CHUNK_SIZE):
        chunk = tickers[start : start + YF_CHUNK_SIZE]
        logging.info("Universe price fetch %s-%s of %s...", start + 1, start + len(chunk), len(tickers))
        if start:
            time.sleep(YF_CHUNK_PAUSE_SECONDS)
        try:
            raw = _download_chunk(chunk)
        except Exception as exc:
            logging.warning(
                "Chunk download failed (%s); retrying once after %ss cool-off.",
                exc,
                YF_CHUNK_RETRY_PAUSE_SECONDS,
            )
            time.sleep(YF_CHUNK_RETRY_PAUSE_SECONDS)
            try:
                raw = _download_chunk(chunk)
            except Exception as retry_exc:
                logging.warning("Chunk retry failed (%s); continuing without it.", retry_exc)
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
    columns = [
        "symbol", "last_price", "avg_volume_20d", "dollar_volume_20d",
        "sma_50", "sma_100", "sma_200",
        "above_sma_50", "above_sma_100", "above_sma_200",
        "below_sma_50", "below_sma_100", "below_sma_200",
    ]
    if history is None or history.empty:
        return pd.DataFrame(columns=columns)
    for symbol, group in history.groupby("symbol"):
        work = group.sort_values("datetime")
        closes = pd.to_numeric(work["close"], errors="coerce").dropna()
        volumes = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
        if len(closes) < 20:
            continue
        last_price = float(closes.iloc[-1])
        avg_volume = float(volumes.tail(20).mean())
        row = {
            "symbol": symbol,
            "last_price": last_price,
            "avg_volume_20d": avg_volume,
            "dollar_volume_20d": last_price * avg_volume,
        }
        # Both above_* and below_* require the SMA to exist, so a young listing
        # with no 200-day average can never sneak into either trend list.
        for period in (50, 100, 200):
            sma = float(closes.tail(period).mean()) if len(closes) >= period else None
            row[f"sma_{period}"] = sma
            row[f"above_sma_{period}"] = bool(sma is not None and last_price >= sma)
            row[f"below_sma_{period}"] = bool(sma is not None and last_price < sma)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns if not rows else None)


def fetch_market_caps(symbols: list[str], *, refresh: bool = False) -> dict[str, float]:
    """Market caps (millions USD) via yfinance fast_info, cached for a week.

    Only called for screen survivors, so the per-symbol calls stay bounded."""
    cached_payload = None if refresh else _load_json_cache(MARKET_CAP_CACHE, MARKET_CAP_MAX_AGE_DAYS)
    caps: dict[str, float] = dict((cached_payload or {}).get("caps") or {})
    missing = [s for s in symbols if s not in caps]
    if missing:
        import yfinance as yf

        logging.info("Fetching market caps for %s symbol(s)...", len(missing))
        for count, symbol in enumerate(missing, start=1):
            try:
                cap = yf.Ticker(symbol).fast_info.get("marketCap")
                caps[symbol] = float(cap) / 1e6 if cap else 0.0
            except Exception:
                caps[symbol] = 0.0
            # Sequential + paced (~8/sec ceiling) so this never looks like a
            # burst; flush progress so an interrupted first run resumes cheaply.
            time.sleep(CAP_FETCH_PAUSE_SECONDS)
            if count % CAP_CACHE_FLUSH_EVERY == 0:
                _save_json_cache(MARKET_CAP_CACHE, {"caps": caps})
                logging.info("Market caps %s/%s (cache flushed).", count, len(missing))
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


def compare_symbol_lists(ours: list[str], theirs: list[str]) -> dict:
    """Diff our universe against an external list (e.g. pasted from TC2000).

    Symbols are normalized to Yahoo form (BRK.B -> BRK-B) so the same name in
    both conventions counts as a match. ``overlap_pct`` is measured against the
    external list: "how much of YOUR list did we reproduce"."""

    def _norm(value: str) -> str:
        return str(value or "").strip().upper().replace(".", "-")

    ours_set = {_norm(s) for s in ours if _norm(s)}
    theirs_set = {_norm(s) for s in theirs if _norm(s)}
    # TC2000 writes class shares with no separator at all (BRKB); Yahoo uses a
    # dash (BRK-B). Remap an unmatched external symbol onto our dashed form when
    # that variant exists on our side, so the same company never lands on both
    # sides of the diff.
    for symbol in list(theirs_set - ours_set):
        if len(symbol) >= 3:
            dashed = symbol[:-1] + "-" + symbol[-1]
            if dashed in ours_set:
                theirs_set.discard(symbol)
                theirs_set.add(dashed)
    matched = sorted(ours_set & theirs_set)
    return {
        "matched": matched,
        "only_ours": sorted(ours_set - theirs_set),
        "only_theirs": sorted(theirs_set - ours_set),
        "ours_count": len(ours_set),
        "theirs_count": len(theirs_set),
        "matched_count": len(matched),
        "overlap_pct": round(100.0 * len(matched) / len(theirs_set), 1) if theirs_set else 0.0,
    }


def _write_watchlist(path: Path, symbols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")


def _read_watchlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return sorted({token.strip().upper() for token in raw.replace(",", "\n").split() if token.strip()})


def load_manual_includes(list_name: str) -> list[str]:
    path = UNIVERSE_INCLUDE_FILES.get(str(list_name or "").strip().lower())
    return _read_watchlist(path) if path else []


def merge_external_into_universe(list_name: str, external_symbols: list[str]) -> dict:
    """Amalgamate an external watchlist (e.g. pasted from TC2000) into ours.

    The union is written to the universe file immediately, and the names we were
    missing are appended to the matching include file so every future rebuild
    re-applies them — this is how gaps in the free sources get patched for good.
    """
    key = str(list_name or "").strip().lower()
    list_path = UNIVERSE_LIST_FILES.get(key)
    include_path = UNIVERSE_INCLUDE_FILES.get(key)
    if list_path is None or include_path is None:
        raise ValueError(f"Unknown universe list '{list_name}' (expected one of {sorted(UNIVERSE_LIST_FILES)}).")
    ours = _read_watchlist(list_path)
    diff = compare_symbol_lists(ours, external_symbols)
    added = list(diff["only_theirs"])
    if added:
        _write_watchlist(include_path, sorted(set(load_manual_includes(key)) | set(added)))
        _write_watchlist(list_path, sorted(set(ours) | set(added)))
    return {
        "list_name": key,
        "added": added,
        "added_count": len(added),
        "total": len(set(ours) | set(added)),
        "include_file": str(include_path),
    }


def build_universe(
    *,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    min_price: float = DEFAULT_MIN_PRICE,
    min_avg_volume: float = DEFAULT_MIN_AVG_VOLUME,
    min_market_cap_m: float = DEFAULT_MIN_MARKET_CAP_M,
    options_filter: str = DEFAULT_OPTIONS_FILTER,
    refresh: bool = False,
    write_outputs: bool = True,
) -> dict:
    """Longs mirror the TC2000 long screen (quality + above SMA100 and SMA200);
    shorts mirror the TC2000 short screen (quality + below SMA50, SMA100 AND
    SMA200). ``options_filter``: "optionable" (any listed options, the TC2000
    'Optionable Stocks Is True' box), "weeklies" (weekly options only) or "none"."""
    listed = fetch_all_listed_symbols(refresh=refresh)
    logging.info("Listing directory: %s symbols.", len(listed))

    options_filter = str(options_filter or "none").strip().lower()
    option_symbols: list[str] = []
    if options_filter == "weeklies":
        option_symbols = fetch_weekly_option_symbols(refresh=refresh)
    elif options_filter == "optionable":
        option_symbols = fetch_optionable_symbols(refresh=refresh)
    if option_symbols:
        listed = [s for s in listed if s in set(option_symbols)]
        logging.info("%s options filter: %s symbols remain.", options_filter, len(listed))

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

    longs = sorted(screened[screened["above_sma_100"] & screened["above_sma_200"]]["symbol"])
    shorts = sorted(
        screened[
            screened["below_sma_50"] & screened["below_sma_100"] & screened["below_sma_200"]
        ]["symbol"]
    )
    # Re-apply manual includes so amalgamated names (source-gap patches merged in
    # via the Universe tab) survive every rebuild.
    include_longs = load_manual_includes("longs")
    include_shorts = load_manual_includes("shorts")
    include_all = load_manual_includes("all")
    longs = sorted(set(longs) | set(include_longs))
    shorts = sorted(set(shorts) | set(include_shorts))
    all_symbols = sorted(
        set(screened["symbol"]) | set(include_all) | set(include_longs) | set(include_shorts)
    )

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
        "options_filter": options_filter,
        "options_filter_applied": bool(option_symbols),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the self-sufficient scan universe")
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_MAX_SYMBOLS)
    parser.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    parser.add_argument("--min-avg-volume", type=float, default=DEFAULT_MIN_AVG_VOLUME)
    parser.add_argument("--min-market-cap-m", type=float, default=DEFAULT_MIN_MARKET_CAP_M)
    parser.add_argument(
        "--options-filter",
        choices=OPTIONS_FILTER_CHOICES,
        default=DEFAULT_OPTIONS_FILTER,
        help="optionable = any listed options (TC2000 'Optionable Stocks'); weeklies = weekly options only",
    )
    parser.add_argument("--refresh", action="store_true", help="ignore local caches")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = build_universe(
        max_symbols=args.max_symbols,
        min_price=args.min_price,
        min_avg_volume=args.min_avg_volume,
        min_market_cap_m=args.min_market_cap_m,
        options_filter=args.options_filter,
        refresh=args.refresh,
    )
    applied = "applied" if result["options_filter_applied"] else "SKIPPED (source unreachable)"
    print(
        f"Universe: {len(result['all'])} symbols "
        f"({len(result['longs'])} longs / {len(result['shorts'])} shorts); "
        f"{result['options_filter']} options filter {applied}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
