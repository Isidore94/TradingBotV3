#!/usr/bin/env python3
"""Consolidate AVWAP signals and intraday bounce logs into a single daily snapshot.

This helper script reads the AVWAP signal CSV produced by ``master_avwap.py`` and the
intraday bounce table (``data/intraday_bounces.csv``) if available.  When the structured
CSV is missing it falls back to the legacy ``logs/bouncers.txt`` output.  The data are
filtered to the current date, combined on ``symbol`` + ``side`` and written to
``data/ai_snapshot_YYYYMMDD.csv``.

Running this periodically means the AI coach only needs to ingest one file per day.
"""
from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
AVWAP_SIGNALS_FILE = DATA_DIR / "avwap_signals.csv"
INTRADAY_BOUNCES_FILE = DATA_DIR / "intraday_bounces.csv"
BOUNCE_LOG_FILE = LOG_DIR / "bouncers.txt"
EMA_WINDOWS = (8, 15, 21)
SMA_WINDOWS = (20, 50, 100, 200)
LOOKBACK_DAYS = 400
INDICATOR_COLUMNS = [
    *(f"ema_{span}" for span in EMA_WINDOWS),
    *(f"sma_{window}" for window in SMA_WINDOWS),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a consolidated AI snapshot.")
    parser.add_argument(
        "--date",
        dest="target_date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date.today(),
        help="Date (YYYY-MM-DD) to consolidate. Defaults to today.",
    )
    return parser.parse_args()


def _format_float(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _clean_str(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _normalise_symbol(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().upper()
    return cleaned or None


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_avwap_signals(target_date: date) -> pd.DataFrame:
    if not AVWAP_SIGNALS_FILE.exists():
        return pd.DataFrame(columns=[
            "symbol",
            "trade_date",
            "side",
            "anchor_type",
            "anchor_date",
            "signal_type",
            "avwap_price",
            "band_price",
            "stdev",
        ])

    df = pd.read_csv(AVWAP_SIGNALS_FILE, dtype=str)

    if "trade_date" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    # The CSV stores numeric fields; convert them after filtering to keep strings intact.
    df["trade_date"] = df["trade_date"].astype(str)
    df = df[df["trade_date"] == target_date.isoformat()].copy()

    if df.empty:
        return df

    # Normalise casing and convert numeric columns.
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    df["side"] = df["side"].astype(str).str.upper()

    for col in ("avwap_price", "band_price", "stdev"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def summarise_signals(df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "symbol",
            "side",
            "trade_date",
            "avwap_signal_count",
            "avwap_signals",
            "avwap_anchor_types",
            "avwap_details",
        ])

    records: List[Dict[str, str]] = []
    for (symbol, side), group in df.groupby(["symbol", "side"], sort=True):
        signal_values = []
        for value in group["signal_type"]:
            cleaned = _clean_str(value)
            if cleaned:
                signal_values.append(cleaned)
        signal_names = list(OrderedDict.fromkeys(signal_values))

        anchor_values = []
        for value in group["anchor_type"]:
            cleaned = _clean_str(value)
            if cleaned:
                anchor_values.append(cleaned)
        anchor_types = list(OrderedDict.fromkeys(anchor_values))

        detail_bits: List[str] = []
        for _, row in group.iterrows():
            signal = _clean_str(row.get("signal_type")) or "?"
            anchor = _clean_str(row.get("anchor_type"))
            anchor_date = _clean_str(row.get("anchor_date"))
            detail = (
                f"{signal}"
                f" [{anchor}:{anchor_date},"
                f" avwap={_format_float(row.get('avwap_price'))},"
                f" band={_format_float(row.get('band_price'))},"
                f" stdev={_format_float(row.get('stdev'))}]"
            )
            detail_bits.append(detail)

        records.append(
            {
                "symbol": symbol,
                "side": side,
                "trade_date": target_date.isoformat(),
                "avwap_signal_count": len(group),
                "avwap_signals": "; ".join(signal_names),
                "avwap_anchor_types": "; ".join(anchor_types),
                "avwap_details": "; ".join(detail_bits),
            }
        )

    return pd.DataFrame.from_records(records)


BOUNCE_LINE_REGEX = re.compile(
    r"^\s*(?P<ts>[^|]+?)\s*\|\s*(?P<symbol>[^|]+?)\s*\|\s*(?P<types>[^|]*)\s*\|\s*(?P<direction>[^|]+?)\s*$"
)


def _parse_time_fragment(fragment: str, target_date: date) -> Optional[datetime]:
    fragment = fragment.strip()
    if not fragment:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(fragment, fmt)
        except ValueError:
            continue

    try:
        parsed_time = datetime.strptime(fragment, "%H:%M:%S").time()
        return datetime.combine(target_date, parsed_time)
    except ValueError:
        return None


def _parse_csv_bounce_time(row: pd.Series, target_date: date) -> Optional[datetime]:
    time_str = _clean_str(row.get("time_local"))
    if time_str:
        parsed = _parse_time_fragment(time_str, target_date)
        if parsed:
            return parsed
    # Fall back to full timestamp parsing if available.
    raw_timestamp = _clean_str(row.get("timestamp"))
    if raw_timestamp:
        parsed = _parse_time_fragment(raw_timestamp, target_date)
        if parsed:
            return parsed
    # As a last resort, just return midnight of the trade date so the row is kept.
    return datetime.combine(target_date, datetime.min.time())


def load_bounce_events_from_csv(target_date: date) -> List[Dict[str, object]]:
    if not INTRADAY_BOUNCES_FILE.exists():
        return []

    df = pd.read_csv(INTRADAY_BOUNCES_FILE, dtype=str)
    if df.empty:
        return []

    df["trade_date"] = pd.to_datetime(df.get("trade_date"), errors="coerce").dt.date
    df = df[df["trade_date"] == target_date]
    if df.empty:
        return []

    events: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        symbol = _normalise_symbol(row.get("symbol"))
        if not symbol:
            continue

        direction = _clean_str(row.get("direction")).upper()
        if direction not in {"LONG", "SHORT"}:
            continue

        bounce_types_raw = _clean_str(row.get("bounce_types"))
        bounce_types = [bt.strip() for bt in bounce_types_raw.split(",") if bt.strip()]

        timestamp = _parse_csv_bounce_time(row, target_date)
        if timestamp is None:
            continue

        events.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": direction,
                "bounce_types_raw": bounce_types_raw,
                "bounce_types": bounce_types,
            }
        )

    return events


def load_bounce_events_from_log(target_date: date) -> List[Dict[str, object]]:
    if not BOUNCE_LOG_FILE.exists():
        return []

    events: List[Dict[str, object]] = []
    with open(BOUNCE_LOG_FILE, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            match = BOUNCE_LINE_REGEX.match(line)
            if not match:
                continue

            timestamp_raw = match.group("ts")
            when = _parse_time_fragment(timestamp_raw, target_date)
            if when is None:
                continue

            if when.date() != target_date:
                continue

            symbol = match.group("symbol").strip().upper()
            direction = match.group("direction").strip().upper()
            if direction not in {"LONG", "SHORT"}:
                direction = direction.capitalize()
                if direction.upper() not in {"LONG", "SHORT"}:
                    continue
                direction = direction.upper()

            bounce_types_raw = match.group("types").strip()
            bounce_types = [bt.strip() for bt in bounce_types_raw.split(",") if bt.strip()]

            events.append(
                {
                    "timestamp": when,
                    "symbol": symbol,
                    "side": direction,
                    "bounce_types_raw": bounce_types_raw,
                    "bounce_types": bounce_types,
                }
            )

    return events


def load_bounce_events(target_date: date) -> List[Dict[str, object]]:
    events = load_bounce_events_from_csv(target_date)
    if events:
        return events
    return load_bounce_events_from_log(target_date)


def summarise_bounces(events: Iterable[Dict[str, object]], target_date: date) -> pd.DataFrame:
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}

    for event in events:
        key = (event["symbol"], event["side"])
        bucket = grouped.setdefault(
            key,
            {
                "symbol": event["symbol"],
                "side": event["side"],
                "trade_date": target_date.isoformat(),
                "bounce_count": 0,
                "bounce_types": OrderedDict(),
                "bounce_events": [],
                "latest_bounce_time": None,
            },
        )

        bucket["bounce_count"] += 1
        if event["bounce_types"]:
            for bt in event["bounce_types"]:
                bucket["bounce_types"].setdefault(bt, None)

        ts: datetime = event["timestamp"]
        bucket["bounce_events"].append(
            f"{ts.strftime('%H:%M:%S')} [{event['bounce_types_raw'] or 'n/a'}]"
        )
        if not bucket["latest_bounce_time"] or ts > bucket["latest_bounce_time"]:
            bucket["latest_bounce_time"] = ts

    if not grouped:
        return pd.DataFrame(columns=[
            "symbol",
            "side",
            "trade_date",
            "bounce_count",
            "bounce_types",
            "latest_bounce_time",
            "bounce_events",
        ])

    records: List[Dict[str, object]] = []
    for bucket in grouped.values():
        records.append(
            {
                "symbol": bucket["symbol"],
                "side": bucket["side"],
                "trade_date": bucket["trade_date"],
                "bounce_count": bucket["bounce_count"],
                "bounce_types": "; ".join(bucket["bounce_types"].keys()),
                "latest_bounce_time": (
                    bucket["latest_bounce_time"].strftime("%H:%M:%S")
                    if bucket["latest_bounce_time"]
                    else ""
                ),
                "bounce_events": "; ".join(bucket["bounce_events"]),
            }
        )

    return pd.DataFrame.from_records(records)


def load_indicator_snapshot(
    symbols: Iterable[object], target_date: date
) -> Dict[str, Dict[str, Optional[float]]]:
    cleaned_symbols = sorted(
        {
            sym
            for sym in (_normalise_symbol(symbol) for symbol in symbols)
            if sym is not None
        }
    )

    if not cleaned_symbols:
        return {}

    start_date = target_date - timedelta(days=LOOKBACK_DAYS)
    end_date = target_date + timedelta(days=1)

    indicator_rows: Dict[str, Dict[str, Optional[float]]] = {}

    for symbol in cleaned_symbols:
        try:
            history = yf.download(
                symbol,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Warning: failed to download daily history for {symbol}: {exc}")
            continue

        if history.empty:
            print(f"Warning: no daily history returned for {symbol}")
            continue

        history.index = pd.to_datetime(history.index).tz_localize(None)
        lower = history.rename(columns=str.lower)

        if "close" not in lower.columns:
            print(f"Warning: close prices missing for {symbol}")
            continue

        close_prices = lower["close"]
        for span in EMA_WINDOWS:
            lower[f"ema_{span}"] = close_prices.ewm(span=span, adjust=False).mean()
        for window in SMA_WINDOWS:
            lower[f"sma_{window}"] = close_prices.rolling(window).mean()

        target_ts = pd.Timestamp(target_date)
        if target_ts not in lower.index:
            print(
                f"Warning: {symbol} missing {target_date.isoformat()} candle;"
                " indicators unavailable"
            )
            continue

        price_row = lower.loc[target_ts]
        if isinstance(price_row, pd.DataFrame):
            price_row = price_row.iloc[-1]

        indicator_rows[symbol] = {
            column: _safe_float(price_row.get(column)) for column in INDICATOR_COLUMNS
        }

    return indicator_rows


def main() -> None:
    args = _parse_args()
    target_date: date = args.target_date

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    signals_df = load_avwap_signals(target_date)
    signals_summary = summarise_signals(signals_df, target_date)

    bounce_events = load_bounce_events(target_date)
    bounces_summary = summarise_bounces(bounce_events, target_date)

    merged = pd.merge(
        signals_summary,
        bounces_summary,
        on=["symbol", "side", "trade_date"],
        how="outer",
        sort=True,
        suffixes=("_avwap", "_bounce"),
    )

    for column in INDICATOR_COLUMNS:
        merged[column] = pd.Series([None] * len(merged), dtype=float)

    if not merged.empty:
        indicator_lookup = load_indicator_snapshot(merged["symbol"], target_date)

        def _indicator_value(symbol: object, column: str) -> Optional[float]:
            cleaned = _normalise_symbol(symbol)
            if cleaned is None:
                return None
            return indicator_lookup.get(cleaned, {}).get(column)

        for column in INDICATOR_COLUMNS:
            merged[column] = merged["symbol"].map(lambda sym, col=column: _indicator_value(sym, col))

    merged.insert(0, "snapshot_date", target_date.isoformat())
    merged.sort_values(["symbol", "side"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    output_path = DATA_DIR / f"ai_snapshot_{target_date.strftime('%Y%m%d')}.csv"
    merged.to_csv(output_path, index=False)

    print(f"Wrote {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main()
