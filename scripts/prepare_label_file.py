"""Generate label-ready CSV based on today's universe data."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LABEL_DIR = DATA_DIR / "labeling"
OUTPUT_DIR = ROOT_DIR / "output"

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
    """Return DataFrame with label columns appended, preserving existing data."""
    existing_columns = list(df.columns)
    missing_columns: List[str] = []

    for column in LABEL_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)
            df[column] = ""

    # Ensure label columns are at end and existing order preserved
    ordered_columns = existing_columns + [col for col in LABEL_COLUMNS if col in missing_columns]
    df = df[ordered_columns]
    return df


def main() -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    source_file = OUTPUT_DIR / "today_universe.csv"
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    print(f"Reading source file: {source_file}")
    df = pd.read_csv(source_file, dtype=str)

    if "trade_date" not in df.columns:
        raise KeyError("'trade_date' column not found in today_universe.csv")

    unique_trade_dates = df["trade_date"].dropna().unique()
    unique_trade_dates = [td for td in unique_trade_dates if td != ""]

    if len(unique_trade_dates) != 1:
        raise ValueError(
            "Expected exactly one unique trade_date, "
            f"found {len(unique_trade_dates)} values: {unique_trade_dates}"
        )

    trade_date = unique_trade_dates[0]
    output_filename = f"to_label_{trade_date.replace('-', '')}.csv"
    destination_file = LABEL_DIR / output_filename

    df = ensure_label_columns(df)

    df.to_csv(destination_file, index=False)
    print(f"Wrote label file: {destination_file}")


if __name__ == "__main__":
    main()
