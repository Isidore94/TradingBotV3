# -*- coding: utf-8 -*-
"""Trial script for TC2000-style VWAP calculations on 5-minute bars."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Tc2000VwapResult:
    current_vwap: float | None
    previous_vwap: float | None
    eod_vwap: float | None
    current_vwap_series: pd.Series
    eod_vwap_series: pd.Series


def _parse_bars(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["time"],
        errors="coerce",
        format="%Y%m%d  %H:%M:%S",
    )
    if df["datetime"].isna().all():
        df["datetime"] = pd.to_datetime(df["time"], errors="coerce")

    if df["datetime"].isna().any():
        bad_rows = df[df["datetime"].isna()]["time"].head(5).tolist()
        raise ValueError(
            "Failed to parse some time values. Example rows: "
            f"{bad_rows}"
        )

    df = df.sort_values("datetime").reset_index(drop=True)
    df["typical_price"] = (
        df["open"] + df["high"] + df["low"] + df["close"]
    ) / 4.0
    return df


def _warn_if_not_five_minute_bars(df: pd.DataFrame) -> None:
    if len(df) < 2:
        return
    deltas = df["datetime"].diff().dropna()
    median_minutes = deltas.median().total_seconds() / 60
    if abs(median_minutes - 5) > 0.1:
        print(
            "Warning: median bar interval is "
            f"{median_minutes:.2f} minutes; TC2000 AVWAP uses 5-minute bars."
        )


def _calculate_vwap_series(df: pd.DataFrame) -> pd.Series:
    if df.empty or df["volume"].sum() == 0:
        return pd.Series([], dtype=float)

    vol_price = df["typical_price"] * df["volume"]
    cum_vol = df["volume"].cumsum()
    cum_vol_price = vol_price.cumsum()
    return cum_vol_price / cum_vol


def calculate_tc2000_vwaps(df: pd.DataFrame) -> Tc2000VwapResult:
    if df.empty:
        return Tc2000VwapResult(None, None, None, pd.Series([], dtype=float), pd.Series([], dtype=float))

    unique_dates = sorted(df["datetime"].dt.date.unique())
    current_date = unique_dates[-1]
    previous_date = unique_dates[-2] if len(unique_dates) > 1 else None

    today_df = df[df["datetime"].dt.date == current_date]
    previous_df = df[df["datetime"].dt.date == previous_date] if previous_date else pd.DataFrame()

    current_vwap_series = _calculate_vwap_series(today_df)
    current_vwap = current_vwap_series.iloc[-1] if not current_vwap_series.empty else None

    previous_vwap_series = _calculate_vwap_series(previous_df)
    previous_vwap = previous_vwap_series.iloc[-1] if not previous_vwap_series.empty else None

    if previous_df.empty:
        eod_vwap_series = pd.Series([], dtype=float)
        eod_vwap = None
    else:
        last_prev_bar = previous_df.iloc[[-1]]
        eod_df = pd.concat([last_prev_bar, today_df])
        eod_vwap_series = _calculate_vwap_series(eod_df)
        eod_vwap = eod_vwap_series.iloc[-1] if not eod_vwap_series.empty else None

    return Tc2000VwapResult(
        current_vwap=current_vwap,
        previous_vwap=previous_vwap,
        eod_vwap=eod_vwap,
        current_vwap_series=current_vwap_series,
        eod_vwap_series=eod_vwap_series,
    )


def _attach_series(
    df: pd.DataFrame,
    today_df: pd.DataFrame,
    current_vwap_series: pd.Series,
    eod_vwap_series: pd.Series,
    previous_vwap: float | None,
) -> pd.DataFrame:
    output = today_df.copy()
    output["tc2000_vwap"] = current_vwap_series.to_list()
    if not eod_vwap_series.empty:
        output["tc2000_eod_avwap"] = eod_vwap_series.iloc[1:].to_list()
    else:
        output["tc2000_eod_avwap"] = None
    output["tc2000_previous_avwap"] = previous_vwap
    output = output.drop(columns=["typical_price"]) if "typical_price" in output.columns else output
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trial TC2000-style VWAP calculations using 5-minute bars."
    )
    parser.add_argument("csv", type=Path, help="CSV file with time/open/high/low/close/volume columns")
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output CSV for today's bars with TC2000 VWAP columns",
    )
    args = parser.parse_args()

    df = _parse_bars(args.csv)
    _warn_if_not_five_minute_bars(df)
    result = calculate_tc2000_vwaps(df)

    print("TC2000 VWAP snapshot")
    print(f"Current VWAP: {result.current_vwap}")
    print(f"Previous Day VWAP: {result.previous_vwap}")
    print(f"EOD AVWAP (last prev bar + today): {result.eod_vwap}")

    if args.out:
        current_date = df["datetime"].dt.date.iloc[-1]
        today_df = df[df["datetime"].dt.date == current_date]
        output = _attach_series(
            df,
            today_df,
            result.current_vwap_series,
            result.eod_vwap_series,
            result.previous_vwap,
        )
        output.to_csv(args.out, index=False)
        print(f"Wrote {len(output)} rows to {args.out}")


if __name__ == "__main__":
    main()
