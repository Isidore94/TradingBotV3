"""Merge realized trade outcomes into the master setups dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
from pandas.errors import EmptyDataError


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MASTER_PATH = DATA_DIR / "master_setups.parquet"
TRADES_PATH = DATA_DIR / "trades_history.csv"


OUTCOME_COLUMNS = ["realized_rr", "hit_target", "stopped_out", "days_held", "outcome_label"]
TRADE_LOG_COLUMNS = [
    "example_id",
    "symbol",
    "side",
    "trade_date",
    "entry_price",
    "stop_price",
    "target_1",
    "conviction_score",
    "exit_price",
    "exit_date",
    "exit_notes",
]


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


def ensure_trade_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in TRADE_LOG_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


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

    trades_df = pd.read_csv(TRADES_PATH, dtype=str)
    trades_df = ensure_trade_log_columns(trades_df)

    trades_df["example_id"] = trades_df["example_id"].fillna("").astype(str).str.strip()
    trades_df["symbol_norm"] = normalize_string(trades_df["symbol"])
    trades_df["side_norm"] = normalize_string(trades_df["side"])
    trades_df["entry_price_val"] = to_numeric(trades_df["entry_price"])
    trades_df["stop_price_val"] = to_numeric(trades_df["stop_price"])
    trades_df["exit_price_val"] = to_numeric(trades_df["exit_price"])

    entry_date_source = trades_df["trade_date"].where(
        trades_df["trade_date"].astype(str).str.strip() != "",
        trades_df.get("entry_date", trades_df["trade_date"]),
    )
    trades_df["entry_date_norm"] = parse_dates(entry_date_source)
    trades_df["exit_date_norm"] = parse_dates(trades_df["exit_date"])

    completed_trades = trades_df[
        (trades_df["example_id"] != "")
        & trades_df["exit_date_norm"].notna()
        & trades_df["exit_price_val"].notna()
    ].copy()

    completed_trades = completed_trades.sort_values(
        ["exit_date_norm", "entry_date_norm"]
    ).reset_index(drop=True)

    taken_mask = get_column(master_df, "taken").apply(is_taken)
    rr_empty_mask = master_df["realized_rr"].apply(is_empty)
    candidate_indices = master_df.index[(taken_mask & rr_empty_mask)].tolist()

    matched_count = 0
    trade_lookup: dict[str, pd.Series] = {
        row["example_id"]: row for _, row in completed_trades.iterrows()
    }

    for idx in candidate_indices:
        example_id = str(master_df.at[idx, "example_id"]).strip()
        if not example_id or example_id not in trade_lookup:
            continue

        trade_row = trade_lookup.pop(example_id)

        entry_date = trade_row["entry_date_norm"]
        exit_date = trade_row["exit_date_norm"]

        if pd.isna(entry_date) or pd.isna(exit_date):
            continue

        side_series = normalize_string(pd.Series([trade_row["side"]]))
        fallback_side = normalize_string(pd.Series([master_df.at[idx, "side"]]))
        side = (side_series.iloc[0] or fallback_side.iloc[0]).upper()

        if side not in {"LONG", "SHORT"}:
            continue

        entry_price = trade_row["entry_price_val"]
        stop_price = trade_row["stop_price_val"]
        exit_price = trade_row["exit_price_val"]

        if any(pd.isna(x) for x in [entry_price, stop_price, exit_price]):
            continue

        if side == "LONG":
            risk_per_share = entry_price - stop_price
            pnl_per_share = exit_price - entry_price
        else:
            risk_per_share = stop_price - entry_price
            pnl_per_share = entry_price - exit_price

        if risk_per_share is None or risk_per_share <= 0:
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

    master_df.to_parquet(MASTER_PATH, index=False)

    total_candidates = len(candidate_indices)
    unmatched = total_candidates - matched_count

    print(f"Candidate setups: {total_candidates}")
    print(f"Matched setups:   {matched_count}")
    print(f"Unmatched setups: {unmatched}")


if __name__ == "__main__":
    main()
