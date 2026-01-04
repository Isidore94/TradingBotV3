"""Generate label-ready CSV based on today's universe data."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LABEL_DIR = DATA_DIR / "labeling"
OUTPUT_DIR = ROOT_DIR / "output"
FEATURE_UNIVERSE_FILE = OUTPUT_DIR / "build_feature_universe.csv"

LABEL_COLUMNS: List[str] = [
    "taken",
    "conviction_score",
    "plan_entry",
    "plan_stop",
    "plan_target_1",
    "plan_target_2",
    "notes",
]


def ensure_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with label columns inserted at the start."""
    existing_columns = list(df.columns)

    # Add missing label columns so downstream labeling never fails.
    for column in LABEL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    # Re-order columns: label columns first (in defined order) followed by
    # whatever existed before, preserving their original order.
    ordered_columns = LABEL_COLUMNS + [
        column for column in existing_columns if column not in LABEL_COLUMNS
    ]
    df = df[ordered_columns]
    return df


def _normalise_trade_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _load_reference_universe() -> Tuple[Optional[date], Set[str]]:
    if not FEATURE_UNIVERSE_FILE.exists():
        return None, set()

    ref_df = pd.read_csv(FEATURE_UNIVERSE_FILE, dtype=str)
    if "trade_date" not in ref_df.columns:
        return None, set()

    ref_df["trade_date_norm"] = _normalise_trade_dates(ref_df["trade_date"])
    ref_df = ref_df.dropna(subset=["trade_date_norm"])
    if ref_df.empty:
        return None, set()

    latest_date = ref_df["trade_date_norm"].max()
    symbols_series = (
        ref_df.loc[ref_df["trade_date_norm"] == latest_date, "symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    valid_symbols = {symbol for symbol in symbols_series if symbol}
    return latest_date, valid_symbols


def filter_today_universe(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if "trade_date" not in df.columns:
        raise KeyError("'trade_date' column not found in today_universe.csv")

    df = df.copy()
    df["trade_date_norm"] = _normalise_trade_dates(df["trade_date"])
    df = df.dropna(subset=["trade_date_norm"])
    if df.empty:
        raise ValueError("No valid trade_date values found in today_universe.csv")

    reference_date, reference_symbols = _load_reference_universe()
    if reference_date is not None:
        target_date = reference_date
    else:
        target_date = df["trade_date_norm"].max()

    if "symbol" in df.columns:
        df["symbol_norm"] = df["symbol"].astype(str).str.upper().str.strip()
    else:
        df["symbol_norm"] = ""
    filtered_df = df[df["trade_date_norm"] == target_date]

    if reference_symbols:
        refined_df = filtered_df[filtered_df["symbol_norm"].isin(reference_symbols)]
        if not refined_df.empty:
            filtered_df = refined_df

    if filtered_df.empty:
        raise ValueError(
            f"No rows remain for trade_date {target_date}. "
            "Check that build_feature_universe.csv is up to date."
        )

    trade_date_str = target_date.isoformat()
    filtered_df = filtered_df.drop(columns=["trade_date_norm", "symbol_norm"], errors="ignore")
    filtered_df["trade_date"] = trade_date_str
    return filtered_df, trade_date_str


def main() -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    source_file = OUTPUT_DIR / "today_universe.csv"
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    print(f"Reading source file: {source_file}")
    df = pd.read_csv(source_file, dtype=str)

    df, trade_date = filter_today_universe(df)
    output_filename = f"to_label_{trade_date.replace('-', '')}.csv"
    destination_file = LABEL_DIR / output_filename

    df = ensure_label_columns(df)

    df.to_csv(destination_file, index=False)
    print(f"Wrote label file: {destination_file}")


if __name__ == "__main__":
    main()
