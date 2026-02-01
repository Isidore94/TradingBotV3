# -*- coding: utf-8 -*-
"""Trial script for TC2000-style VWAP calculations on 5-minute bars."""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


@dataclass
class Tc2000VwapResult:
    current_vwap: float | None
    previous_vwap: float | None
    eod_vwap: float | None
    current_vwap_series: pd.Series
    eod_vwap_series: pd.Series


class IbHistoricalClient(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self._data: list[dict[str, float | int | str]] = []
        self._data_ready = threading.Event()
        self._connected = threading.Event()
        self._errors: list[str] = []

    def nextValidId(self, orderId: int) -> None:
        self._connected.set()

    def error(self, reqId: int, errorCode: int, errorString: str) -> None:
        self._errors.append(f"ReqId={reqId} Code={errorCode} Msg={errorString}")

    def historicalData(self, reqId: int, bar) -> None:
        self._data.append(
            {
                "time": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        self._data_ready.set()

    def wait_for_connection(self, timeout: float) -> bool:
        return self._connected.wait(timeout=timeout)

    def wait_for_data(self, timeout: float) -> bool:
        return self._data_ready.wait(timeout=timeout)

    def consume_data(self) -> list[dict[str, float | int | str]]:
        return list(self._data)

    def consume_errors(self) -> list[str]:
        return list(self._errors)


def _bars_to_dataframe(bars: list[dict[str, float | int | str]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
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


def _create_stock_contract(symbol: str) -> Contract:
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def _fetch_ib_bars(
    symbol: str,
    host: str,
    port: int,
    client_id: int,
    duration: str,
    bar_size: str,
    use_rth: int,
    timeout: float,
) -> list[dict[str, float | int | str]]:
    client = IbHistoricalClient()
    client.connect(host, port, clientId=client_id)
    api_thread = threading.Thread(target=client.run, daemon=True)
    api_thread.start()

    if not client.wait_for_connection(timeout):
        client.disconnect()
        raise RuntimeError("Timed out waiting for IB API connection.")

    req_id = 1001
    contract = _create_stock_contract(symbol)
    client.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )

    if not client.wait_for_data(timeout):
        client.disconnect()
        raise RuntimeError("Timed out waiting for IB historical data.")

    client.disconnect()
    errors = client.consume_errors()
    if errors:
        print("IB API warnings/errors:")
        for message in errors:
            print(f" - {message}")

    return client.consume_data()


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
    parser.add_argument("symbol", help="Ticker symbol to request from IB.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="IB Gateway/TWS host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7496,
        help="IB Gateway/TWS port (default: 7496).",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=125,
        help="IB API client id (default: 125).",
    )
    parser.add_argument(
        "--duration",
        default="5 D",
        help="IB duration string for historical data (default: 5 D).",
    )
    parser.add_argument(
        "--bar-size",
        default="5 mins",
        help="IB bar size setting (default: 5 mins).",
    )
    parser.add_argument(
        "--use-rth",
        type=int,
        default=1,
        help="Use regular trading hours (1=yes, 0=no). Default 1.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for IB connection/data (default: 20).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output CSV for today's bars with TC2000 VWAP columns",
    )
    args = parser.parse_args()

    bars = _fetch_ib_bars(
        symbol=args.symbol,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        duration=args.duration,
        bar_size=args.bar_size,
        use_rth=args.use_rth,
        timeout=args.timeout,
    )
    df = _bars_to_dataframe(bars)
    if df.empty:
        raise RuntimeError("No historical bars returned from IB.")
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
