"""Merge realized trade outcomes into the master setups dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MASTER_PATH = DATA_DIR / "master_setups.parquet"
TRADES_PATH = DATA_DIR / "trades_history.csv"


OUTCOME_COLUMNS = ["realized_rr", "hit_target", "stopped_out", "days_held", "outcome_label"]


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def get_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([pd.NA] * len(df), index=df.index)


def normalize_string(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.date


def is_taken(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) == 1.0
    return str(value).strip() == "1"


def is_empty(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def determine_outcome(realized_rr: Any) -> Dict[str, Any]:
    if pd.isna(realized_rr):
        return {
            "realized_rr": pd.NA,
            "hit_target": pd.NA,
            "stopped_out": pd.NA,
            "days_held": pd.NA,
            "outcome_label": "",
        }

    outcome_label = ""
    if realized_rr >= 2.0:
        outcome_label = "big_win"
    elif realized_rr <= -1.0:
        outcome_label = "full_loss"
    elif -1.0 < realized_rr < -0.25:
        outcome_label = "small_loss"
    elif -0.25 <= realized_rr <= 0.25:
        outcome_label = "flat"
    elif 0.0 < realized_rr < 2.0:
        outcome_label = "small_win"

    return {
        "realized_rr": realized_rr,
        "hit_target": realized_rr >= 1.0,
        "stopped_out": realized_rr <= -1.0,
        "outcome_label": outcome_label,
    }


def main() -> None:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master setups file not found: {MASTER_PATH}")
    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"Trades history file not found: {TRADES_PATH}")

    master_df = pd.read_parquet(MASTER_PATH)
    master_df = ensure_columns(master_df, OUTCOME_COLUMNS)

    trades_df = pd.read_csv(
        TRADES_PATH,
        dtype={
            "symbol": str,
            "side": str,
            "entry_price": float,
            "exit_price": float,
            "stop_price": float,
            "quantity": float,
        },
        parse_dates=["entry_date", "exit_date"],
    )

    master_df["symbol_norm"] = normalize_string(get_column(master_df, "symbol"))
    master_df["side_norm"] = normalize_string(get_column(master_df, "side"))
    master_df["trade_date_norm"] = parse_dates(get_column(master_df, "trade_date"))

    trades_df["symbol_norm"] = normalize_string(trades_df["symbol"])
    trades_df["side_norm"] = normalize_string(trades_df["side"])
    trades_df["entry_date_norm"] = trades_df["entry_date"].dt.date
    trades_df["exit_date_norm"] = trades_df["exit_date"].dt.date

    trades_df = trades_df.sort_values(["entry_date_norm", "exit_date_norm"]).reset_index(drop=True)

    taken_mask = get_column(master_df, "taken").apply(is_taken)
    rr_empty_mask = master_df["realized_rr"].apply(is_empty)
    candidate_mask = taken_mask & rr_empty_mask

    candidate_indices = master_df.index[candidate_mask].tolist()
    matched_count = 0
    used_trade_indices: set[int] = set()

    for idx in candidate_indices:
        symbol = master_df.at[idx, "symbol_norm"]
        side = master_df.at[idx, "side_norm"]
        trade_date = master_df.at[idx, "trade_date_norm"]

        if pd.isna(trade_date):
            continue

        matches = trades_df[
            (trades_df["symbol_norm"] == symbol)
            & (trades_df["side_norm"] == side)
            & (trades_df["entry_date_norm"] == trade_date)
        ]
        matches = matches[~matches.index.isin(used_trade_indices)]

        if matches.empty:
            continue

        if len(matches) > 1:
            print(
                "Warning: Multiple trades matched setup for symbol"
                f" {symbol} on {trade_date}. Using the first match."
            )

        trade_row = matches.iloc[0]
        used_trade_indices.add(trade_row.name)

        entry_date = trade_row["entry_date_norm"]
        exit_date = trade_row["exit_date_norm"]

        if pd.isna(entry_date) or pd.isna(exit_date):
            continue

        if trade_row["side_norm"] not in {"LONG", "SHORT"}:
            continue

        if trade_row["side_norm"] == "LONG":
            risk_per_share = trade_row["entry_price"] - trade_row["stop_price"]
            pnl_per_share = trade_row["exit_price"] - trade_row["entry_price"]
        else:
            risk_per_share = trade_row["stop_price"] - trade_row["entry_price"]
            pnl_per_share = trade_row["entry_price"] - trade_row["exit_price"]

        if risk_per_share is None or risk_per_share <= 0:
            print(
                "Warning: Non-positive risk for symbol"
                f" {symbol} on {trade_date}. Skipping outcome calculation."
            )
            continue

        realized_rr = pnl_per_share / risk_per_share
        days_held = (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days

        outcome = determine_outcome(realized_rr)
        outcome["days_held"] = days_held

        master_df.at[idx, "realized_rr"] = outcome["realized_rr"]
        master_df.at[idx, "hit_target"] = outcome["hit_target"]
        master_df.at[idx, "stopped_out"] = outcome["stopped_out"]
        master_df.at[idx, "days_held"] = outcome["days_held"]
        master_df.at[idx, "outcome_label"] = outcome["outcome_label"]

        matched_count += 1

    master_df = master_df.drop(columns=["symbol_norm", "side_norm", "trade_date_norm"], errors="ignore")

    master_df.to_parquet(MASTER_PATH, index=False)

    total_candidates = len(candidate_indices)
    unmatched = total_candidates - matched_count

    print(f"Candidate setups: {total_candidates}")
    print(f"Matched setups:   {matched_count}")
    print(f"Unmatched setups: {unmatched}")


if __name__ == "__main__":
    main()
