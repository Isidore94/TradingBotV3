from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - project currently depends on pandas
    pd = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only when yfinance is absent
    yf = None


MARKET_SNAPSHOT_TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "^VIX",
    "TLT",
    "HYG",
    "XLK",
    "XLF",
    "XLE",
    "XLU",
    "XLV",
    "XLY",
    "XLP",
    "XLI",
    "XLC",
    "XLB",
    "SMH",
    "ARKK",
    "GLD",
    "USO",
    "BTC-USD",
]


def fetch_market_snapshot(tickers: list[str] | None = None) -> dict[str, Any]:
    if yf is None:
        return _error_payload("yfinance is not installed. Install it to enable Market Prep price snapshots.")
    if pd is None:
        return _error_payload("pandas is not installed. Install it to enable Market Prep price snapshots.")

    ticker_list = tickers or MARKET_SNAPSHOT_TICKERS
    generated_at = datetime.now().isoformat(timespec="seconds")
    try:
        data = yf.download(
            tickers=" ".join(ticker_list),
            period="90d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        return _error_payload(f"Failed downloading market snapshot data: {exc}", generated_at=generated_at)

    rows = []
    for ticker in ticker_list:
        frame = _extract_ticker_frame(data, ticker, multiple=len(ticker_list) > 1)
        rows.append(_build_ticker_snapshot(ticker, frame))

    classification = classify_market_snapshot(rows)
    return {
        "generated_at": generated_at,
        "source": "yfinance",
        "tickers": ticker_list,
        "classification": classification,
        "rows": rows,
        "errors": [
            row["error"]
            for row in rows
            if row.get("error")
        ],
    }


def classify_market_snapshot(rows: list[dict[str, Any]]) -> dict[str, str]:
    row_map = {str(row.get("ticker") or ""): row for row in rows}
    spy = row_map.get("SPY") or {}
    qqq = row_map.get("QQQ") or {}
    spy_above_21 = spy.get("above_21_sma")
    qqq_above_21 = qqq.get("above_21_sma")
    spy_return_5d = _safe_float(spy.get("return_5d_pct"))
    qqq_return_5d = _safe_float(qqq.get("return_5d_pct"))

    if spy_above_21 is None or qqq_above_21 is None or spy_return_5d is None or qqq_return_5d is None:
        return {
            "label": "❌ Noisy",
            "reason": "Insufficient SPY/QQQ data for the basic regime check.",
        }
    if spy_above_21 and qqq_above_21 and qqq_return_5d > spy_return_5d:
        return {
            "label": "✅ Clean",
            "reason": "SPY and QQQ are above 21 SMA, and QQQ is outperforming SPY over 5 days.",
        }
    if not spy_above_21 and not qqq_above_21:
        return {
            "label": "❌ Noisy",
            "reason": "SPY and QQQ are both below 21 SMA.",
        }
    return {
        "label": "⚠️ Watch",
        "reason": "SPY/QQQ trend and relative strength signals are mixed.",
    }


def _build_ticker_snapshot(ticker: str, frame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return _empty_row(ticker, "No price history returned.")

    price_series = _get_price_series(frame)
    if price_series is None or price_series.dropna().empty:
        return _empty_row(ticker, "No close price history returned.")

    price_series = price_series.dropna()
    last_close = _last_value(price_series)
    sma_21 = _last_value(price_series.rolling(21).mean())
    sma_50 = _last_value(price_series.rolling(50).mean())
    volume_series = frame["Volume"].dropna() if "Volume" in frame.columns else None
    last_volume = _last_value(volume_series) if volume_series is not None else None
    volume_avg_20 = _last_value(volume_series.rolling(20).mean()) if volume_series is not None else None

    return {
        "ticker": ticker,
        "last_close": _round_float(last_close),
        "return_1d_pct": _round_float(_return_pct(price_series, 1)),
        "return_5d_pct": _round_float(_return_pct(price_series, 5)),
        "return_20d_pct": _round_float(_return_pct(price_series, 20)),
        "sma_21": _round_float(sma_21),
        "sma_50": _round_float(sma_50),
        "above_21_sma": _above_sma(last_close, sma_21),
        "above_50_sma": _above_sma(last_close, sma_50),
        "last_volume": _round_float(last_volume),
        "volume_avg_20": _round_float(volume_avg_20),
        "volume_vs_20d_avg_pct": _round_float(_volume_vs_average_pct(last_volume, volume_avg_20)),
        "error": "",
    }


def _extract_ticker_frame(data, ticker: str, *, multiple: bool):
    if data is None or data.empty:
        return None
    try:
        if multiple and isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.get_level_values(0):
                return None
            frame = data[ticker].copy()
        else:
            frame = data.copy()
    except Exception:
        return None
    return frame.dropna(how="all")


def _get_price_series(frame):
    for column in ("Adj Close", "Close"):
        if column in frame.columns:
            return frame[column]
    return None


def _return_pct(series, periods: int) -> float | None:
    if series is None or len(series) <= periods:
        return None
    current = _last_value(series)
    previous = _last_value(series.iloc[:-periods])
    if current is None or previous in (None, 0):
        return None
    return ((current - previous) / previous) * 100


def _volume_vs_average_pct(last_volume: float | None, average_volume: float | None) -> float | None:
    if last_volume is None or average_volume in (None, 0):
        return None
    return ((last_volume - average_volume) / average_volume) * 100


def _above_sma(last_close: float | None, sma_value: float | None) -> bool | None:
    if last_close is None or sma_value is None:
        return None
    return last_close >= sma_value


def _last_value(series) -> float | None:
    if series is None:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    return _safe_float(clean.iloc[-1])


def _safe_float(value: Any) -> float | None:
    try:
        if pd is not None and pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_float(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _empty_row(ticker: str, error: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "last_close": None,
        "return_1d_pct": None,
        "return_5d_pct": None,
        "return_20d_pct": None,
        "sma_21": None,
        "sma_50": None,
        "above_21_sma": None,
        "above_50_sma": None,
        "last_volume": None,
        "volume_avg_20": None,
        "volume_vs_20d_avg_pct": None,
        "error": error,
    }


def _error_payload(message: str, *, generated_at: str | None = None) -> dict[str, Any]:
    return {
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "source": "yfinance",
        "tickers": MARKET_SNAPSHOT_TICKERS,
        "classification": {
            "label": "❌ Noisy",
            "reason": message,
        },
        "rows": [],
        "errors": [message],
    }
