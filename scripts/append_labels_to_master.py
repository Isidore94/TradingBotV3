"""Append manually labeled setups into the master dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LABEL_DIR = DATA_DIR / "labeling"
MASTER_PATH = DATA_DIR / "master_setups.parquet"

CORE_COLUMNS: List[str] = [
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

LABEL_COLUMNS: List[str] = [
    "taken",
    "conviction_score",
    "plan_entry",
    "plan_stop",
    "plan_target_1",
    "plan_target_2",
    "notes",
]

OUTCOME_COLUMNS: List[str] = [
    "realized_rr",
    "hit_target",
    "stopped_out",
    "days_held",
    "outcome_label",
]

REQUIRED_ORDER: List[str] = [
    "example_id",
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
    "taken",
    "conviction_score",
    "plan_entry",
    "plan_stop",
    "plan_target_1",
    "plan_target_2",
    "notes",
    "realized_rr",
    "hit_target",
    "stopped_out",
    "days_held",
    "outcome_label",
]

ALL_COLUMNS: List[str] = ["example_id"] + CORE_COLUMNS + LABEL_COLUMNS + OUTCOME_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append manually labeled CSV rows into the master setups parquet file."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the labeled CSV created from data/labeling/to_label_YYYYMMDD.csv",
    )
    return parser.parse_args()


def normalize_trade_date(value: str) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if not value:
        return ""
    try:
        parsed = pd.to_datetime(value)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return value


def ensure_columns(df: pd.DataFrame, columns: List[str], fill_value: object = "") -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = fill_value
    return df


def build_example_id(row: pd.Series) -> str:
    return (
        f"{row.get('symbol', '').strip()}_"
        f"{row.get('trade_date', '').strip()}_"
        f"{row.get('anchor_type', '').strip()}_"
        f"{row.get('signal_type', '').strip()}_"
        f"{row.get('side', '').strip()}_"
        f"{row.name}"
    )


def load_master_dataframe() -> pd.DataFrame:
    if MASTER_PATH.exists():
        master_df = pd.read_parquet(MASTER_PATH)
    else:
        master_df = pd.DataFrame(columns=ALL_COLUMNS)

    master_df = ensure_columns(master_df, ALL_COLUMNS, pd.NA)
    if "example_id" not in master_df.columns:
        master_df["example_id"] = ""
    return master_df


def main() -> None:
    args = parse_args()
    csv_path = Path(args.path).expanduser()

    if not csv_path.exists():
        print(f"Labeled CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading labeled CSV: {csv_path}")
    labeled_df = pd.read_csv(csv_path, dtype=str)
    original_row_count = len(labeled_df)

    if original_row_count == 0:
        print("No rows found in labeled CSV. Nothing to append.")
        sys.exit(0)

    labeled_df = labeled_df.reset_index(drop=True)

    labeled_df = ensure_columns(labeled_df, CORE_COLUMNS)
    labeled_df = ensure_columns(labeled_df, LABEL_COLUMNS)
    labeled_df = ensure_columns(labeled_df, OUTCOME_COLUMNS, pd.NA)

    labeled_df["symbol"] = (
        labeled_df["symbol"].fillna("").astype(str).str.strip().str.upper()
    )
    labeled_df["trade_date"] = labeled_df["trade_date"].apply(normalize_trade_date)

    if "example_id" not in labeled_df.columns:
        labeled_df["example_id"] = ""

    labeled_df["example_id"] = labeled_df.apply(
        lambda row: row["example_id"] if str(row["example_id"]).strip() else build_example_id(row),
        axis=1,
    )

    print(f"Loaded {original_row_count} labeled rows.")

    master_df = load_master_dataframe()
    existing_ids = set(master_df["example_id"].dropna().astype(str))

    new_df = labeled_df[~labeled_df["example_id"].isin(existing_ids)].copy()
    skipped = len(labeled_df) - len(new_df)
    if skipped:
        print(f"Skipped {skipped} duplicate rows based on example_id.")

    if new_df.empty:
        print("No new rows to append after removing duplicates.")
        sys.exit(0)

    combined_df = pd.concat([master_df, new_df], ignore_index=True)
    combined_df = ensure_columns(combined_df, ALL_COLUMNS, pd.NA)

    ordered_columns = REQUIRED_ORDER + [
        col for col in combined_df.columns if col not in REQUIRED_ORDER
    ]
    combined_df = combined_df[ordered_columns]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(MASTER_PATH, index=False)

    print(f"Appended {len(new_df)} new rows to master dataset.")
    print(f"Master dataset now contains {len(combined_df)} rows.")


if __name__ == "__main__":
    main()
