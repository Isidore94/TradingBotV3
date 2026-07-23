from __future__ import annotations

"""Pure D1/M5 snapshot-chart data for the symbol quick-look popup.

Turns the data the app already holds (the master-AVWAP daily parquet store
and BounceBot's cached M5 bars) into plot-ready payloads: candle bars plus
aligned overlay series. No Qt, no network, no IB - everything here is local
reads and arithmetic, so the GUI can call it synchronously on click.

Overlay contract: each overlay is {"label", "values", "color", "width",
"dash"} where ``values`` aligns 1:1 with the bar list (None = undefined at
that bar; the chart breaks the line there). ``color`` is a ui.theme role
name resolved by the widget, keeping this module theme-agnostic.

The M5 VWAP/sigma series mirrors ``calc_anchored_vwap_bands`` exactly
(typical price = OHLC/4, sigma accumulated from each bar's deviation from
the RUNNING vwap, volume-weighted) - the running-deviation variant every
band consumer is calibrated to. Do not "fix" it toward a distribution
stdev; see plan.md section 5.
"""

from typing import Any, Callable, Iterable, Mapping

D1_DEFAULT_SESSIONS = 90

D1_OVERLAY_SPECS = (
    ("sma", 50, "SMA50", "info", 1.6, False),
    ("sma", 100, "SMA100", "caution", 1.6, False),
    ("sma", 200, "SMA200", "short", 1.6, False),
    ("ema", 8, "EMA8", "favorite", 1.1, True),
    ("ema", 15, "EMA15", "near", 1.1, True),
    ("ema", 21, "EMA21", "study", 1.1, True),
)

M5_EMA_SPECS = (
    ("ema", 15, "EMA15", "near", 1.1, True),
    ("ema", 21, "EMA21", "study", 1.1, True),
)


def sma_series(values: Iterable[float], period: int) -> list[float | None]:
    """Simple moving average; None until a full window exists."""
    values = [float(value) for value in values]
    period = max(1, int(period))
    result: list[float | None] = [None] * len(values)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index >= period - 1:
            result[index] = running / period
    return result


def ema_series(values: Iterable[float], span: int) -> list[float | None]:
    """Exponential moving average, pandas ``ewm(span, adjust=False)`` recursion.

    Matches how the rest of the codebase computes EMAs (e.g. the daily-trend
    gate), so the popup's EMA8/15/21 agree with the values the bot acts on.
    """
    values = [float(value) for value in values]
    if not values:
        return []
    span = max(1, int(span))
    alpha = 2.0 / (span + 1.0)
    result: list[float | None] = [values[0]]
    for value in values[1:]:
        previous = result[-1]
        result.append(alpha * value + (1.0 - alpha) * previous)
    return result


def session_vwap_series(bars: list[Mapping[str, Any]]) -> dict[str, list[float | None]]:
    """Per-session running VWAP and +/-1 sigma bands, aligned to ``bars``.

    Accumulation restarts on every date change (session VWAP). Bar math is
    the calc_anchored_vwap_bands running-deviation variant: at each bar the
    plotted value is exactly what that function would return anchored at the
    session open and ended at that bar. Zero-volume bars carry the previous
    values forward so the drawn line has no artificial breaks.
    """
    vwap: list[float | None] = []
    upper: list[float | None] = []
    lower: list[float | None] = []
    cum_vol = cum_vp = cum_sd = 0.0
    current_date = None
    for bar in bars or []:
        stamp = bar.get("dt")
        bar_date = stamp.date() if hasattr(stamp, "date") else None
        if bar_date != current_date:
            current_date = bar_date
            cum_vol = cum_vp = cum_sd = 0.0
        try:
            volume = float(bar.get("volume") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        if volume > 0:
            tp = (
                float(bar["open"]) + float(bar["high"]) + float(bar["low"]) + float(bar["close"])
            ) / 4.0
            cum_vol += volume
            cum_vp += tp * volume
            running = cum_vp / cum_vol
            cum_sd += (tp - running) * (tp - running) * volume
        if cum_vol > 0:
            value = cum_vp / cum_vol
            stdev = (cum_sd / cum_vol) ** 0.5
            vwap.append(value)
            upper.append(value + stdev)
            lower.append(value - stdev)
        else:
            vwap.append(None)
            upper.append(None)
            lower.append(None)
    return {"vwap": vwap, "upper_1": upper, "lower_1": lower}


def _overlay(label: str, values: list[float | None], color: str, width: float, dash: bool) -> dict:
    return {"label": label, "values": values, "color": color, "width": width, "dash": dash}


def _tail(values: list, count: int) -> list:
    return values[-count:] if count and len(values) > count else values


def load_d1_bars(symbol: str) -> list[dict[str, Any]]:
    """Full daily history from the durable parquet store as chart bars."""
    from setup_playbook_study import _load_daily_frame

    frame = _load_daily_frame(str(symbol or "").strip().upper())
    if frame is None:
        return []
    bars: list[dict[str, Any]] = []
    has_volume = "volume" in frame.columns
    for row in frame.itertuples(index=False):
        bars.append(
            {
                "dt": row.datetime.to_pydatetime() if hasattr(row.datetime, "to_pydatetime") else row.datetime,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(getattr(row, "volume", 0.0) or 0.0) if has_volume else 0.0,
            }
        )
    return bars


def build_d1_snapshot(
    symbol: str,
    *,
    sessions: int = D1_DEFAULT_SESSIONS,
    loader: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Daily candles + SMA50/100/200 + EMA8/15/21, indicators computed on the
    full history so the displayed tail carries correct long-lookback values."""
    bars = (loader or load_d1_bars)(symbol)
    if not bars:
        return {"symbol": symbol, "timeframe": "D1", "bars": [], "overlays": [], "note": "no daily store"}
    closes = [bar["close"] for bar in bars]
    overlays = []
    for kind, period, label, color, width, dash in D1_OVERLAY_SPECS:
        series = sma_series(closes, period) if kind == "sma" else ema_series(closes, period)
        overlays.append(_overlay(label, _tail(series, sessions), color, width, dash))
    return {
        "symbol": symbol,
        "timeframe": "D1",
        "bars": _tail(bars, sessions),
        "overlays": overlays,
        "note": "",
    }


def build_m5_snapshot(symbol: str, bars: list[Mapping[str, Any]]) -> dict[str, Any]:
    """M5 candles + session VWAP with +/-1 sigma bands + EMA15/21."""
    bars = [dict(bar) for bar in bars or []]
    if not bars:
        return {"symbol": symbol, "timeframe": "M5", "bars": [], "overlays": [], "note": "no cached M5 bars"}
    closes = [bar["close"] for bar in bars]
    vwap = session_vwap_series(bars)
    overlays = [
        _overlay("VWAP", vwap["vwap"], "accent", 1.6, False),
        _overlay("+1σ", vwap["upper_1"], "accent", 1.0, True),
        _overlay("-1σ", vwap["lower_1"], "accent", 1.0, True),
    ]
    for kind, span, label, color, width, dash in M5_EMA_SPECS:
        overlays.append(_overlay(label, ema_series(closes, span), color, width, dash))
    return {"symbol": symbol, "timeframe": "M5", "bars": bars, "overlays": overlays, "note": ""}
