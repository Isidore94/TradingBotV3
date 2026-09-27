"""Runner dip watch (p9): strong names that dip under the earnings AVWAP and squeeze on M5.

The trader, 2026-09-27: "When these stocks are running they should be flagged and go into a
mini watchlist that starts to fire when SPY is set up for it and these stocks dip below AVWAPE
and start to compress on lower time frames." Long lab (SPY above a rising 20-day, a leader
pullback + strength + a close under the earnings AVWAP): n 112, beat SPY 58%, +5.4% avg over
10 sessions. The M5 squeeze trigger itself is UNTESTED: every fire is logged for grading.

* Members ("runners"), once per D1 scan (`build_members`, written by `long_setups_store`):
  63-day RS percentile >= `RS_MIN_PERCENTILE`, close above the 100- and 200-day SMA, (close -
  SMA50) / ATR20 >= `STRENGTH_MIN_SMA50_ATR`, the long-setups liquidity floor, and a known
  earnings AVWAP (the study's anchor, `long_setups.earnings_anchor_index`).
* Armed: the market is working for longs AND the close is at most `ARM_MAX_SIGMA` sigma over
  the earnings AVWAP. At most `ARMED_MAX` names, by RS. Unarmed members are listed, never fire.
* Fire (`evaluate`): on today's completed regular-session M5 bars, the last close is under the
  earnings AVWAP AND the last `SQUEEZE_BOX_BARS` bars span at most `SQUEEZE_RANGE_ATR` x the
  `SQUEEZE_ATR_BARS`-bar M5 ATR (the S6 compression-break box in `m5_signal_engines`, copied
  and pinned by a parity test). Missing bars = unknown = no fire. Once per name per session
  (the caller holds that set).
* Grading (shadow, `grade_fires`): each fire vs SPY at 1 / 5 / 10 sessions on completed daily
  bars; pending is pending, never zero.

Pure: no I/O, no clock of its own. Decision support only; it never changes a detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

import long_setups
from completed_bars import bar_time, completed_m5_bars
from setup_permutations import STRENGTH_MIN_SMA50_ATR

SCHEMA_VERSION = 1
#: The review-event action every fire is recorded under (the grading reads it back).
FIRED_ACTION = "runner_dip_fired"

# --- the member / armed rules (lead's design, 2026-09-27; the trader can overrule)
#: 63-day RS vs SPY percentile over the scanned universe (`long_setups.rs_percentiles`).
RS_MIN_PERCENTILE = 0.9
#: Close above every one of these SMAs (`long_setups.TREND_SMAS`).
TREND_SMAS = long_setups.TREND_SMAS
#: Armed when the close is at most this many sigma over the earnings AVWAP.
ARM_MAX_SIGMA = 1.0
#: At most this many armed names, highest RS first.
ARMED_MAX = 15
#: The armed list serves the sessions after its scan session, for this many calendar days.
MAX_AGE_DAYS = long_setups.FOCUS_MAX_AGE_DAYS

# --- the M5 squeeze: the same numbers as `m5_signal_engines` SQUEEZE_BOX_BARS /
# SQUEEZE_ATR_BARS / SQUEEZE_RANGE_ATR (a parity test pins them and the box math).
SQUEEZE_BOX_BARS = 12
SQUEEZE_ATR_BARS = 20
SQUEEZE_RANGE_ATR = 2.5
EXCHANGE_TIMEZONE = "America/New_York"
_REGULAR_OPEN = 9 * 60 + 30
_REGULAR_CLOSE = 16 * 60
_M5_MINUTES = 5

#: Grading horizons (sessions after the fire session).
GRADE_SESSIONS = (1, 5, 10)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


# --- members, once per D1 scan

def _member(symbol: str, bars: list[dict[str, Any]], *, atr: Any, rs: float | None, sector: str,
            earnings_dates: Iterable[Any] | None, gap_date: Any) -> dict[str, Any] | None:
    """One runner's facts at the last completed close, or None (not a runner / unknown)."""
    if rs is None or rs < RS_MIN_PERCENTILE or len(bars) < max(TREND_SMAS):
        return None
    closes = [bar["close"] for bar in bars]
    close = closes[-1]
    if any(close <= sum(closes[-length:]) / length for length in TREND_SMAS):
        return None
    atr_value = long_setups._atr(bars, atr)
    if atr_value is None:
        return None
    strength = long_setups.strength_shadow(bars, atr_value, None, None)["strength_sma50_atr"]
    if strength is None or strength < STRENGTH_MIN_SMA50_ATR:
        return None
    anchor = long_setups.earnings_anchor_index(bars, earnings_dates=earnings_dates, gap_date=gap_date)
    bands = long_setups.avwap_bands(bars, anchor) if anchor is not None else None
    if bands is None or not bands[1] > 0:
        return None
    level, sigma = bands
    return {
        "symbol": symbol,
        "as_of": bars[-1]["date"],
        "close": round(close, 4),
        "atr20": round(atr_value, 4),
        "avwape": round(level, 4),
        "sigma": round(sigma, 4),
        "avwape_z": round((close - level) / sigma, 4),
        "anchor_date": bars[anchor]["date"],
        "rs_percentile": round(rs, 4),
        "strength_sma50_atr": strength,
        "sector": sector,
    }


def near_avwape(member: Mapping[str, Any]) -> bool:
    """The close is at most `ARM_MAX_SIGMA` sigma over the earnings AVWAP (under it counts)."""
    close, level, sigma = (_num(member.get(key)) for key in ("close", "avwape", "sigma"))
    return None not in (close, level, sigma) and close <= level + ARM_MAX_SIGMA * sigma


def arm(members: list[dict[str, Any]], working: str) -> list[str]:
    """Mark each member ``armed`` (working market + near the AVWAPE). The `ARMED_MAX` cap takes the
    names already under the AVWAPE first, then RS (lead, 2026-09-27)."""

    def order(row):
        close, level = _num(row.get("close")), _num(row.get("avwape"))
        under = close is not None and level is not None and close < level
        return (not under, -(row.get("rs_percentile") or 0.0), row["symbol"])

    armed: list[str] = []
    for member in sorted(members, key=order):
        member["armed"] = working == "yes" and len(armed) < ARMED_MAX and near_avwape(member)
        if member["armed"]:
            armed.append(member["symbol"])
    return armed


def build_members(
    *,
    bars_by_symbol: Mapping[str, Any],
    spy_bars: Any,
    feature_rows: Iterable[Mapping[str, Any]],
    earnings_by_symbol: Mapping[str, Mapping[str, Any]] | None = None,
    atr_by_symbol: Mapping[str, Any] | None = None,
    sector_by_symbol: Mapping[str, Any] | None = None,
    market_cap_by_symbol: Mapping[str, Any] | None = None,
    earnings_dates_by_symbol: Mapping[str, Iterable[Any]] | None = None,
    as_of: Any = None,
) -> dict[str, Any]:
    """``{as_of, market_working, market_rule, armed, members}`` for one scan (the same inputs,
    universe, liquidity floor and market gate as `long_setups.build_rows`)."""
    as_of_text = _text(as_of)[:10]
    feature_rows = list(feature_rows or ())
    working, rule = long_setups.market_gate(feature_rows)
    out = {"as_of": as_of_text, "market_working": working, "market_rule": rule, "armed": [], "members": []}
    if not as_of_text:
        return out
    spy_closes = {bar["date"]: bar["close"] for bar in long_setups._clean_bars(spy_bars) or []}
    facts_by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in feature_rows:
        facts_by_symbol.setdefault(_text(row.get("symbol")).upper(), []).append(row)
    current: dict[str, list[dict[str, Any]]] = {}
    for symbol, raw in (bars_by_symbol or {}).items():
        symbol = _text(symbol).upper()
        bars = long_setups._clean_bars(raw)
        if symbol and symbol != "SPY" and bars and bars[-1]["date"] == as_of_text:
            current[symbol] = bars
    percentiles = long_setups.rs_percentiles(
        {symbol: long_setups.rs_vs_spy(bars, spy_closes) for symbol, bars in current.items()})
    members = []
    for symbol, bars in current.items():
        facts = facts_by_symbol.get(symbol, [])
        cap = _num((market_cap_by_symbol or {}).get(symbol))
        if cap is None:
            cap = next((_num(row.get("perm_market_cap_m")) for row in facts
                        if _num(row.get("perm_market_cap_m")) is not None), None)
        if not long_setups.meets_liquidity_floor(bars, cap):
            continue
        sector = _text((sector_by_symbol or {}).get(symbol)) or next(
            (_text(row.get("sector")) for row in facts if _text(row.get("sector"))), "")
        member = _member(symbol, bars, atr=(atr_by_symbol or {}).get(symbol), rs=percentiles.get(symbol),
                         sector=sector, earnings_dates=(earnings_dates_by_symbol or {}).get(symbol),
                         gap_date=((earnings_by_symbol or {}).get(symbol) or {}).get("gap_date"))
        if member is not None:
            members.append(member)
    out["armed"] = arm(members, working)
    out["members"] = sorted(members, key=lambda row: (not row["armed"], -row["rs_percentile"], row["symbol"]))
    return out


def _day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(_text(value)[:10])
    except ValueError:
        return None


def payload_is_current(payload: Mapping[str, Any] | None, *, today: Any) -> bool:
    """True when the payload's scan session is before ``today`` and at most `MAX_AGE_DAYS` old."""
    as_of, day = _day((payload or {}).get("as_of")), _day(today)
    return as_of is not None and day is not None and 0 < (day - as_of).days <= MAX_AGE_DAYS


def armed_members(payload: Mapping[str, Any] | None, *, today: Any) -> list[dict[str, Any]]:
    """Today's armed members; none from a stale payload or a market that is not working."""
    if not payload_is_current(payload, today=today) or _text((payload or {}).get("market_working")) != "yes":
        return []
    return [dict(row) for row in (payload or {}).get("members") or ()
            if isinstance(row, Mapping) and row.get("armed") and _text(row.get("symbol"))]


def armed_symbols(payload: Mapping[str, Any] | None, *, today: Any) -> list[str]:
    return [_text(row["symbol"]).upper() for row in armed_members(payload, today=today)]


def status_line(payload: Mapping[str, Any] | None, *, today: Any) -> str:
    """``Runner dips: 3 armed (GTLB, NVDA, ...)`` for the Alert Center; "" with no current scan."""
    if not payload_is_current(payload, today=today):
        return ""
    names = armed_symbols(payload, today=today)
    members = len((payload or {}).get("members") or ())
    if _text((payload or {}).get("market_working")) != "yes":
        return f"Runner dips: {members} runner(s), none armed (the market is not working for longs)."
    if not names:
        return f"Runner dips: {members} runner(s), none near the earnings VWAP."
    shown = ", ".join(names[:8]) + (f" +{len(names) - 8}" if len(names) > 8 else "")
    return f"Runner dips: {len(names)} armed ({shown})."


# --- the fire, on today's completed regular-session M5 bars

@dataclass(frozen=True)
class RunnerDipHit:
    symbol: str
    bar_time: datetime  # the fire bar's START, zone-aware exchange time
    close: float
    avwape: float
    box_high: float
    box_low: float
    atr: float
    range_atr: float
    as_of: str
    rs_percentile: float | None
    strength_sma50_atr: float | None

    @property
    def bar_close(self) -> datetime:
        return self.bar_time + timedelta(minutes=_M5_MINUTES)

    @property
    def session(self) -> str:
        return self.bar_time.date().isoformat()

    @property
    def line(self) -> str:
        return (f"{self.symbol} runner dip: strong name under the earnings VWAP ({self.avwape:.2f}), "
                f"squeezing on M5 - box {self.box_low:.2f}-{self.box_high:.2f}")


def _exchange_tz():
    from zoneinfo import ZoneInfo

    return ZoneInfo(EXCHANGE_TIMEZONE)


def _default_local_tz():
    from market_session import get_market_local_timezone

    return get_market_local_timezone()[0]


def _ohlc(bar: Any) -> tuple[float, float, float, float] | None:
    values = []
    for key in ("open", "high", "low", "close"):
        raw = bar.get(key) if isinstance(bar, Mapping) else getattr(bar, key, None)
        value = _num(raw)
        if value is None or value <= 0:
            return None
        values.append(value)
    return values[0], values[1], values[2], values[3]


def _regular(bars: Sequence[Any], *, now: datetime, tz) -> list[tuple[datetime, tuple]] | None:
    """Completed regular-session bars as (exchange start, ohlc); None when a bar is unreadable."""
    local = tz if tz is not None else _default_local_tz()
    exchange = _exchange_tz()
    out = []
    for bar in completed_m5_bars(bars, now=now):
        stamp = bar_time(bar)
        if stamp is None:
            continue
        ohlc = _ohlc(bar)
        if ohlc is None:
            return None
        start = (stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=local)).astimezone(exchange)
        if _REGULAR_OPEN <= start.hour * 60 + start.minute < _REGULAR_CLOSE:
            out.append((start, ohlc))
    return out


def squeeze_box(regular: Sequence[tuple[datetime, tuple]]) -> tuple[float, float, float] | None:
    """(box high, box low, M5 ATR) over the last `SQUEEZE_BOX_BARS` bars when they squeeze, else None.

    The S6 compression-break box for a bar right after these: the box bars all on the last
    bar's day, the ATR the mean true range of the last `SQUEEZE_ATR_BARS` bars (21 bars needed).
    """
    count = len(regular)
    if count < max(SQUEEZE_ATR_BARS + 1, SQUEEZE_BOX_BARS):
        return None
    box = regular[count - SQUEEZE_BOX_BARS:]
    day = regular[-1][0].date()
    if any(start.date() != day for start, _ohlc_values in box):
        return None
    true_ranges = []
    for index in range(count - SQUEEZE_ATR_BARS, count):
        _o, high, low, _c = regular[index][1]
        prior_close = regular[index - 1][1][3]
        true_ranges.append(max(high - low, abs(high - prior_close), abs(low - prior_close)))
    atr = sum(true_ranges) / SQUEEZE_ATR_BARS
    box_high = max(ohlc[1] for _start, ohlc in box)
    box_low = min(ohlc[2] for _start, ohlc in box)
    if atr <= 0 or (box_high - box_low) / atr > SQUEEZE_RANGE_ATR:
        return None
    return box_high, box_low, atr


def evaluate(member: Mapping[str, Any], bars: Sequence[Any], *, now: datetime, tz=None) -> RunnerDipHit | None:
    """The fire on the last completed M5 bar of TODAY's regular session, or None.

    Today = ``now``'s exchange date. Needs the last bar on today, its close under the member's
    earnings AVWAP and a squeeze (`squeeze_box`). Anything missing is no fire.
    """
    avwape = _num(member.get("avwape"))
    symbol = _text(member.get("symbol")).upper()
    if avwape is None or not symbol or not bars:
        return None
    regular = _regular(bars, now=now, tz=tz)
    if not regular:
        return None
    local = tz if tz is not None else _default_local_tz()
    today = (now if now.tzinfo is not None else now.replace(tzinfo=local)).astimezone(_exchange_tz()).date()
    start, ohlc = regular[-1]
    if start.date() != today or not ohlc[3] < avwape:
        return None
    box = squeeze_box(regular)
    if box is None:
        return None
    box_high, box_low, atr = box
    return RunnerDipHit(
        symbol=symbol, bar_time=start, close=ohlc[3], avwape=avwape,
        box_high=round(box_high, 4), box_low=round(box_low, 4), atr=round(atr, 6),
        range_atr=round((box_high - box_low) / atr, 4), as_of=_text(member.get("as_of")),
        rs_percentile=_num(member.get("rs_percentile")),
        strength_sma50_atr=_num(member.get("strength_sma50_atr")),
    )


def fire_detail(hit: RunnerDipHit, *, spy_price: Any = None) -> dict[str, Any]:
    """The review-event detail: everything the outcome grading reads back."""
    return {
        "kind": "runner_dip",
        "session": hit.session,
        "bar_time": hit.bar_time.isoformat(),
        "ts": hit.bar_close.isoformat(),
        "price": round(hit.close, 4),
        "avwape": round(hit.avwape, 4),
        "box_high": hit.box_high,
        "box_low": hit.box_low,
        "m5_atr": hit.atr,
        "range_atr": hit.range_atr,
        "as_of": hit.as_of,
        "rs_percentile": hit.rs_percentile,
        "strength_sma50_atr": hit.strength_sma50_atr,
        "spy_price": _num(spy_price),
        "message": hit.line,
    }


# --- grading (shadow)

def fires_from_events(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The first fire per (symbol, session) from review-event rows (a restart may repeat one)."""
    seen: set[tuple[str, str]] = set()
    out = []
    for row in events or ():
        if not isinstance(row, Mapping) or _text(row.get("action")) != FIRED_ACTION:
            continue
        detail = row.get("detail") if isinstance(row.get("detail"), Mapping) else {}
        symbol, session = _text(row.get("symbol")).upper(), _text(detail.get("session"))[:10]
        price = _num(detail.get("price"))
        if not symbol or not session or price is None or (symbol, session) in seen:
            continue
        seen.add((symbol, session))
        out.append({"symbol": symbol, "session": session, "price": price,
                    "spy_price": _num(detail.get("spy_price"))})
    return out


def grade_fires(fires: Iterable[Mapping[str, Any]], closes_by_symbol: Mapping[str, Mapping[str, float]],
                spy_closes: Mapping[str, float]) -> list[dict[str, Any]]:
    """Each fire's return (fire price -> the close N sessions after the fire session) and SPY's
    (its price at the fire -> the same close), per `GRADE_SESSIONS`. A horizon whose close is not
    in the completed daily bars yet is ``None`` (pending); SPY without a fire price is unknown."""
    spy_days = sorted(spy_closes or {})
    out = []
    for fire in fires or ():
        closes = (closes_by_symbol or {}).get(_text(fire.get("symbol")).upper()) or {}
        price, spy_price = _num(fire.get("price")), _num(fire.get("spy_price"))
        session = _text(fire.get("session"))[:10]
        row = {"symbol": fire.get("symbol"), "session": session}
        later = [day for day in spy_days if day > session]
        for sessions in GRADE_SESSIONS:
            target = later[sessions - 1] if len(later) >= sessions else None
            close = _num(closes.get(target)) if target else None
            spy_close = _num((spy_closes or {}).get(target)) if target else None
            ret = (close / price - 1.0) * 100.0 if close and price else None
            spy_ret = (spy_close / spy_price - 1.0) * 100.0 if spy_close and spy_price else None
            row[f"return_{sessions}"] = None if ret is None else round(ret, 4)
            row[f"spy_{sessions}"] = None if spy_ret is None else round(spy_ret, 4)
        out.append(row)
    return out


def grade_line(graded: Sequence[Mapping[str, Any]]) -> str:
    """``Runner dips (shadow, M5 trigger untested): n fired · 1d beat SPY 2/3, avg +0.8% ...``."""
    if not graded:
        return "Runner dips (shadow, M5 trigger untested): no fire yet."
    parts = []
    for sessions in GRADE_SESSIONS:
        done = [row for row in graded if row.get(f"return_{sessions}") is not None]
        pending = len(graded) - len(done)
        if not done:
            parts.append(f"{sessions}d pending {pending}")
            continue
        avg = sum(row[f"return_{sessions}"] for row in done) / len(done)
        versus = [row for row in done if row.get(f"spy_{sessions}") is not None]
        beat = sum(1 for row in versus if row[f"return_{sessions}"] > row[f"spy_{sessions}"])
        text = f"{sessions}d avg {avg:+.1f}%"
        if versus:
            text += f", beat SPY {beat}/{len(versus)}"
        if pending:
            text += f" ({pending} pending)"
        parts.append(text)
    return f"Runner dips (shadow, M5 trigger untested): {len(graded)} fired · " + " · ".join(parts)

