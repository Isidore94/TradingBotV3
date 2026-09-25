"""P1-6 6c: how far a journal trade went for and against the trader (MFE / MAE).

Read only, and never on the Qt thread (the Trades tab asks from a worker):

* a DAY trade (opened and closed on one date) reads the Day Review's durable
  cached M5 bars for that session (`day_review_bars.read_session_bars`) - every
  COMPLETED bar from the one holding the entry to the one holding the exit;
* a SWING (held overnight, or still open) reads the durable daily store
  (`MASTER_AVWAP_DAILY_BARS_DIR/<SYMBOL>.parquet`) - the daily bars AFTER the
  entry date through the exit date (the entry day's range may predate the fill).

MFE/MAE are in price from the actual average entry, and in R when the trader's
planned entry and stop give a 1R. Missing bars, an option, or an entry at
midnight (a broker-file fill with no time) is UNKNOWN - never zero. Nothing is
written anywhere.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, tzinfo
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
M5_SPAN = timedelta(minutes=5)

MEASURED = "measured"
UNKNOWN = "unknown"

#: Security types whose own symbol has the bars this reads.
_BAR_TYPES = {"STK", "ETF", "UNKNOWN"}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _moment(value: Any, zone: tzinfo) -> datetime | None:
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return moment.replace(tzinfo=zone) if moment.tzinfo is None else moment


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def trade_kind(trade: Mapping[str, Any]) -> str:
    """`day` when opened and closed on one date, else `swing` (overnight or still open)."""
    opened, closed = _day(trade.get("opened_at")), _day(trade.get("closed_at"))
    return "day" if opened is not None and closed is not None and opened == closed else "swing"


def _unknown(kind: str, reason: str) -> dict[str, Any]:
    return {"state": UNKNOWN, "kind": kind, "reason": reason, "mfe": None, "mae": None,
            "mfe_r": None, "mae_r": None, "bars": 0}


def _one_r(trade: Mapping[str, Any]) -> float | None:
    planned, stop = _number(trade.get("planned_entry")), _number(trade.get("planned_stop"))
    if not planned or not stop:
        return None
    gap = abs(planned - stop)
    return gap if gap > 0 else None


def excursion(
    trade: Mapping[str, Any],
    bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime,
    zone: tzinfo = NY,
) -> dict[str, Any]:
    """MFE/MAE for one trade over the bars given (M5 for a day trade, D1 for a swing)."""
    kind = trade_kind(trade)
    from journal_identity import normalize_security_type

    if normalize_security_type(trade.get("security_type")) not in _BAR_TYPES:
        return _unknown(kind, "not a stock: its own bars are not the underlying's")
    entry = _number(trade.get("average_entry_price"))
    side = str(trade.get("direction") or "").strip().upper()
    if not entry or side not in {"LONG", "SHORT"}:
        return _unknown(kind, "no entry price or side")
    opened = _moment(trade.get("opened_at"), zone)
    closed = _moment(trade.get("closed_at"), zone)
    if opened is None:
        return _unknown(kind, "no entry time")
    now_aware = now if now.tzinfo is not None else now.replace(tzinfo=zone)
    rows: list[tuple[float, float]] = []
    if kind == "day":
        if (opened.hour, opened.minute, opened.second) == (0, 0, 0):
            return _unknown(kind, "the entry has no time of day")
        start = opened.replace(minute=opened.minute - opened.minute % 5, second=0, microsecond=0)
        end = closed if closed is not None else now_aware
        for bar in bars or ():
            begins = _moment(bar.get("dt"), zone)
            high, low = _number(bar.get("high")), _number(bar.get("low"))
            if begins is None or high is None or low is None:
                continue
            if begins < start or begins >= end or begins + M5_SPAN > now_aware:
                continue
            rows.append((high, low))
    else:
        first = opened.date()
        last = closed.date() if closed is not None else None
        today = now_aware.astimezone(NY).date()
        for bar in bars or ():
            day = _day(bar.get("date") or bar.get("datetime") or bar.get("dt"))
            high, low = _number(bar.get("high")), _number(bar.get("low"))
            if day is None or high is None or low is None:
                continue
            # A daily bar is complete once its session is over: never today's.
            if day <= first or (last is not None and day > last) or day >= today:
                continue
            rows.append((high, low))
    if not rows:
        return _unknown(kind, "no cached bars for the holding window")
    highest = max(high for high, _low in rows)
    lowest = min(low for _high, low in rows)
    if side == "LONG":
        mfe, mae = highest - entry, entry - lowest
    else:
        mfe, mae = entry - lowest, highest - entry
    mfe, mae = max(0.0, mfe), max(0.0, mae)
    one_r = _one_r(trade)
    return {
        "state": MEASURED,
        "kind": kind,
        "reason": "",
        "mfe": round(mfe, 4),
        "mae": round(mae, 4),
        "mfe_r": round(mfe / one_r, 2) if one_r else None,
        "mae_r": round(mae / one_r, 2) if one_r else None,
        "bars": len(rows),
    }


def excursion_text(result: Mapping[str, Any] | None) -> str:
    """One line for the Trades detail: `MFE +1.20 (+2.4R) · MAE -0.30 (-0.6R) · 14 M5 bars`."""
    if not result:
        return "MFE / MAE: measuring..."
    if result.get("state") != MEASURED:
        return f"MFE / MAE: unknown ({result.get('reason') or 'no data'})"

    def part(label: str, value, r_value, sign: str) -> str:
        text = f"{label} {sign}{value:.2f}"
        return text + (f" ({sign}{r_value:.1f}R)" if r_value is not None else "")

    unit = "M5 bars" if result.get("kind") == "day" else "daily bars"
    return " · ".join([
        part("MFE", result["mfe"], result.get("mfe_r"), "+"),
        part("MAE", result["mae"], result.get("mae_r"), "-"),
        f"{result.get('bars', 0)} {unit}",
    ])


# ------------------------------------------------------------------ loaders
def load_session_m5(symbol: str, session: str) -> list[dict[str, Any]]:
    """The Day Review's durable M5 bars for one symbol and session, or []."""
    try:
        import day_review_bars

        grouped = day_review_bars.read_session_bars(session) or {}
    except Exception:  # noqa: BLE001 - unreadable is unknown
        return []
    return list(grouped.get(str(symbol or "").strip().upper()) or [])


def load_daily(symbol: str) -> list[dict[str, Any]]:
    """The durable daily store's rows for one symbol, as dicts, or []."""
    try:
        from human_focus_tracking import _load_durable_daily_frame

        frame = _load_durable_daily_frame(str(symbol or "").strip().upper())
    except Exception:  # noqa: BLE001 - unreadable is unknown
        return []
    if frame is None or getattr(frame, "empty", True):
        return []
    work = frame.rename(columns={column: str(column).strip().lower() for column in frame.columns})
    stamp = next((name for name in ("datetime", "date", "timestamp") if name in work.columns), None)
    if stamp is None or not {"high", "low"} <= set(work.columns):
        return []
    work = work.rename(columns={stamp: "date"})
    return [
        {"date": str(row["date"])[:10], "high": row["high"], "low": row["low"]}
        for row in work[["date", "high", "low"]].to_dict("records")
    ]


def measure_trade(
    trade: Mapping[str, Any],
    *,
    now: datetime | None = None,
    m5_loader: Callable[[str, str], list] = load_session_m5,
    daily_loader: Callable[[str], list] = load_daily,
    zone: tzinfo | None = None,
) -> dict[str, Any]:
    """Load the right cached bars for this trade and measure it. Worker threads only."""
    if zone is None:
        try:
            import live_alert_results

            zone = live_alert_results.desk_zone()
        except Exception:  # noqa: BLE001
            zone = NY
    now = now or datetime.now(tz=NY)
    symbol = str(trade.get("symbol") or "")
    if trade_kind(trade) == "day":
        session = str(trade.get("opened_at") or "")[:10]
        bars = m5_loader(symbol, session)
    else:
        bars = daily_loader(symbol)
    return excursion(trade, bars, now=now, zone=zone)
