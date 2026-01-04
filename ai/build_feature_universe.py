#!/usr/bin/env python3
"""Build the daily feature universe for the trading assistant."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

AVWAP_SIGNALS_FILE = DATA_DIR / "avwap_signals.csv"
INTRADAY_BOUNCES_FILE = DATA_DIR / "intraday_bounces.csv"
UNIVERSE_OUTPUT_FILE = OUTPUT_DIR / "today_universe.csv"

EMA_WINDOWS = (8, 15, 21)
SMA_WINDOWS = (20, 50, 100, 200)
LOOKBACK_DAYS = 400

PREV_LOWER_BOUNCE_SIGNALS = {"PREV_BOUNCE_LOWER_1", "PREV_BOUNCE_LOWER_2"}
PREV_LOWER_CROSS_SIGNALS = {
    "PREV_CROSS_LOWER_1",
    "PREV_CROSS_LOWER_2",
    "PREV_CROSS_LOWER_3",
}

UNIVERSE_COLUMNS = [
    "symbol",
    "trade_date",
    "side",
    "anchor_type",
    "anchor_date",
    "signal_type",
    "avwap_price",
    "band_price",
    "stdev",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ema_8",
    "ema_15",
    "ema_21",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    "has_intraday_bounce_today",
    "prev_lower_bounce_last_5d",
    "prev_lower_cross_last_5d",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def to_float(value):
    """Best-effort conversion helper that tolerates pandas containers."""

    if value is None:
        return None

    # yfinance occasionally returns pandas Series objects even when we expect a
    # scalar (for example when the index contains duplicate timestamps).  In
    # that scenario ``Series.get`` yields a Series and calling ``pd.isna`` on it
    # raises ``ValueError`` because the truthiness of a Series is ambiguous.
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[-1]

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_inputs() -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not AVWAP_SIGNALS_FILE.exists():
        logging.warning("AVWAP signals file not found at %s; exiting.", AVWAP_SIGNALS_FILE)
        return None, None

    avwap_df = pd.read_csv(AVWAP_SIGNALS_FILE)

    if INTRADAY_BOUNCES_FILE.exists():
        bounce_df = pd.read_csv(INTRADAY_BOUNCES_FILE)
    else:
        logging.info(
            "Intraday bounces file not found at %s; continuing without bounce context.",
            INTRADAY_BOUNCES_FILE,
        )
        bounce_df = pd.DataFrame(
            columns=["symbol", "trade_date", "direction", "bounce_types", "time_local"]
        )

    return avwap_df, bounce_df


def normalise_symbol_frame(df: pd.DataFrame, symbol_column: str) -> pd.DataFrame:
    if symbol_column not in df:
        df[symbol_column] = ""
    df[symbol_column] = df[symbol_column].astype(str).str.upper().str.strip()
    return df


def fetch_daily_history(symbol: str) -> pd.DataFrame:
    start_date = datetime.today().date() - timedelta(days=LOOKBACK_DAYS)
    try:
        data = yf.download(
            symbol,
            start=start_date.isoformat(),
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.warning("%s: failed to download daily history (%s)", symbol, exc)
        return pd.DataFrame()

    if data.empty:
        logging.warning("%s: no daily data returned from yfinance", symbol)
        return data

    data.index = pd.to_datetime(data.index).tz_localize(None)
    lower = data.rename(columns=str.lower)
    for col in ["adj close"]:
        if col in lower.columns:
            lower = lower.drop(columns=[col])
    lower = lower[["open", "high", "low", "close", "volume"]].copy()

    for span in EMA_WINDOWS:
        lower[f"ema_{span}"] = lower["close"].ewm(span=span, adjust=False).mean()
    for window in SMA_WINDOWS:
        lower[f"sma_{window}"] = lower["close"].rolling(window).mean()

    return lower


def build_universe(avwap_df: pd.DataFrame, bounce_df: pd.DataFrame) -> None:
    avwap_df = normalise_symbol_frame(avwap_df, "symbol")
    bounce_df = normalise_symbol_frame(bounce_df, "symbol")

    avwap_df["trade_date"] = pd.to_datetime(avwap_df.get("trade_date"), errors="coerce")
    bounce_df["trade_date"] = pd.to_datetime(bounce_df.get("trade_date"), errors="coerce")

    invalid_avwap = avwap_df[avwap_df["trade_date"].isna()]
    if not invalid_avwap.empty:
        logging.warning(
            "Dropping %d AVWAP signal rows with invalid trade dates.", len(invalid_avwap)
        )
        avwap_df = avwap_df[avwap_df["trade_date"].notna()]

    bounce_df = bounce_df[bounce_df["trade_date"].notna()]

    if avwap_df.empty:
        logging.info("No AVWAP signals found. Writing empty universe file.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=UNIVERSE_COLUMNS).to_csv(UNIVERSE_OUTPUT_FILE, index=False)
        return

    symbols = sorted(
        set(avwap_df["symbol"]) | set(bounce_df["symbol"])
    )

    logging.info("Fetching daily history for %d symbols via yfinance…", len(symbols))
    history_cache: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        history_cache[symbol] = fetch_daily_history(symbol)

    bounce_lookup = {
        (row.symbol, row.trade_date.normalize())
        for row in bounce_df.itertuples()
        if isinstance(row.symbol, str) and isinstance(row.trade_date, pd.Timestamp)
    }

    output_rows: List[Dict[str, object]] = []

    for symbol, symbol_signals in avwap_df.groupby("symbol"):
        symbol_history = history_cache.get(symbol)
        if symbol_history is None or symbol_history.empty:
            logging.warning("%s: skipping %d signals due to missing daily history.", symbol, len(symbol_signals))
            continue

        symbol_signals = symbol_signals.sort_values("trade_date").copy()
        symbol_signals["trade_date_norm"] = symbol_signals["trade_date"].dt.normalize()
        history_index = symbol_history.index.sort_values()

        for _, signal_row in symbol_signals.iterrows():
            trade_ts = signal_row["trade_date_norm"]
            if pd.isna(trade_ts):
                continue

            if trade_ts not in symbol_history.index:
                logging.warning(
                    "%s: missing OHLC data for %s; skipping signal %s.",
                    symbol,
                    trade_ts.date().isoformat(),
                    signal_row.get("signal_type"),
                )
                continue

            price_row = symbol_history.loc[trade_ts]
            if isinstance(price_row, pd.DataFrame):
                price_row = price_row.iloc[-1]

            prior_idx = history_index[history_index < trade_ts]
            recent_prior = prior_idx[-5:]
            prior_dates_set = set(recent_prior)

            prior_signals = symbol_signals[
                symbol_signals["trade_date_norm"].isin(prior_dates_set)
            ]

            has_intraday_bounce = (symbol, trade_ts) in bounce_lookup
            prev_lower_bounce = prior_signals["signal_type"].isin(
                PREV_LOWER_BOUNCE_SIGNALS
            ).any()
            prev_lower_cross = prior_signals["signal_type"].isin(
                PREV_LOWER_CROSS_SIGNALS
            ).any()

            anchor_date_value = signal_row.get("anchor_date")
            if pd.isna(anchor_date_value):
                anchor_date_str = ""
            else:
                anchor_date_str = str(anchor_date_value)

            output_rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_ts.date().isoformat(),
                    "side": signal_row.get("side"),
                    "anchor_type": signal_row.get("anchor_type"),
                    "anchor_date": anchor_date_str,
                    "signal_type": signal_row.get("signal_type"),
                    "avwap_price": to_float(signal_row.get("avwap_price")),
                    "band_price": to_float(signal_row.get("band_price")),
                    "stdev": to_float(signal_row.get("stdev")),
                    "open": to_float(price_row.get("open")),
                    "high": to_float(price_row.get("high")),
                    "low": to_float(price_row.get("low")),
                    "close": to_float(price_row.get("close")),
                    "volume": to_float(price_row.get("volume")),
                    "ema_8": to_float(price_row.get("ema_8")),
                    "ema_15": to_float(price_row.get("ema_15")),
                    "ema_21": to_float(price_row.get("ema_21")),
                    "sma_20": to_float(price_row.get("sma_20")),
                    "sma_50": to_float(price_row.get("sma_50")),
                    "sma_100": to_float(price_row.get("sma_100")),
                    "sma_200": to_float(price_row.get("sma_200")),
                    "has_intraday_bounce_today": bool(has_intraday_bounce),
                    "prev_lower_bounce_last_5d": bool(prev_lower_bounce),
                    "prev_lower_cross_last_5d": bool(prev_lower_cross),
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if output_rows:
        universe_df = pd.DataFrame(output_rows)
        universe_df = universe_df.reindex(columns=UNIVERSE_COLUMNS)
        universe_df.sort_values(["trade_date", "symbol", "signal_type"], inplace=True)
    else:
        logging.info("No enriched signals available; writing empty universe file.")
        universe_df = pd.DataFrame(columns=UNIVERSE_COLUMNS)

    universe_df.to_csv(UNIVERSE_OUTPUT_FILE, index=False)
    logging.info("Wrote %d rows to %s", len(universe_df), UNIVERSE_OUTPUT_FILE)


def main() -> None:
    setup_logging()
    avwap_df, bounce_df = load_inputs()
    if avwap_df is None or bounce_df is None:
        return
    build_universe(avwap_df, bounce_df)


if __name__ == "__main__":
    main()
