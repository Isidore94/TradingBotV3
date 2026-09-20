"""Small, point-in-time market facts for one Trade Mentor read.

This module is deliberately pure.  It accepts bars already read by the context
service, excludes incomplete or bad observations, and returns only the small
facts a later journal reader may need.  It never fetches, scores, or advises.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from typing import Any, Mapping, Sequence

from market_calendar import MARKET_TZ, is_session, previous_session
from market_early_close import session_close

#: TJ-14A item 4. `XLRE` joined the list because the desk's own sector map
#: (`group_rrs.SECTOR_ETFS`) has always carried eleven SPDRs while this context
#: carried ten - a trader reading the strip saw a hole where real estate should
#: be. The order stays alphabetical inside the sector block, so `SPY` keeps its
#: place and every existing positional reader is untouched.
SYMBOLS = (
    "VXX", "RSP", "USO", "TLT", "IWM", "QQQ", "SPY", "XLB", "XLC",
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
)

#: The eleven SPDR sector funds, in the order the derived lines rank them by
#: name when two of them are exactly tied.
SECTORS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
#: Offense against defense - the read the trader used to type by hand.
OFFENSE = ("XLK", "XLY", "XLC")
DEFENSE = ("XLP", "XLU", "XLV")

SCHEMA_V1 = "trade_mentor_context_v1"
SCHEMA = "trade_mentor_context_v2"

#: How many names a leader / laggard list carries.
_RANK_DEPTH = 3

_M5_KEYS = (
    "symbol", "m5_status", "m5_reason", "m5_as_of", "m5_change_30m_pct",
    "m5_direction", "m5_vs_session_vwap", "d1_status", "d1_reason", "d1_as_of",
    "d1_change_5d_pct", "d1_vs_sma20",
)
#: v2 adds four facts per symbol. They are appended, never interleaved, so a
#: reader that walks `_M5_KEYS` positionally keeps reading the same columns.
_DAY_KEYS = ("day_change_pct", "day_range_place", "vs_prior_high", "vs_prior_low")
_V2_KEYS = _M5_KEYS + _DAY_KEYS

#: Every derived line, in the order the card prints them.
DERIVED_LINES = (
    "breadth", "fear", "rates", "oil", "sector_leaders", "sector_laggards",
    "offense_vs_defense", "sectors_above_vwap",
)

_OPEN = time(9, 30)


def _blank(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "m5_status": "unavailable", "m5_reason": "no usable M5 bars",
        "m5_as_of": None, "m5_change_30m_pct": None, "m5_direction": None,
        "m5_vs_session_vwap": None, "d1_status": "unavailable",
        "d1_reason": "no usable D1 bars", "d1_as_of": None,
        "d1_change_5d_pct": None, "d1_vs_sma20": None,
        # v2. Present and empty when they cannot be measured, never absent:
        # "not measured" and "the key did not exist yet" are two different
        # absences and a later reader must be able to tell them apart.
        "day_change_pct": None, "day_range_place": None,
        "vs_prior_high": None, "vs_prior_low": None,
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _field(bar: Any, name: str) -> Any:
    return bar.get(name) if isinstance(bar, Mapping) else getattr(bar, name, None)


def _stamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _market_now(now: datetime) -> datetime:
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(MARKET_TZ)


def _last_m5_start(now: datetime) -> datetime:
    """The latest completed regular-hours M5 bar start at ``now``."""
    moment = _market_now(now)
    day = moment.date()
    try:
        if not is_session(day) or moment < datetime.combine(day, _OPEN, MARKET_TZ):
            day = previous_session(day)
            return session_close(day) - timedelta(minutes=5)
        close = session_close(day)
    except Exception:
        # A calendar refusal is uncertainty.  The caller will report stale/no
        # data rather than extrapolating a regular session.
        return datetime.min.replace(tzinfo=MARKET_TZ)
    if moment >= close:
        return close - timedelta(minutes=5)
    elapsed = int((moment - datetime.combine(day, _OPEN, MARKET_TZ)).total_seconds() // 60)
    return datetime.combine(day, _OPEN, MARKET_TZ) + timedelta(minutes=(elapsed // 5 - 1) * 5)


def _last_completed_exchange_session(now: datetime) -> date:
    """Latest closed exchange day, including a real early close."""
    moment = _market_now(now)
    day = moment.date()
    if is_session(day) and session_close(day) <= moment:
        return day
    return previous_session(day)


def _valid_m5(bars: Sequence[Any], now: datetime) -> tuple[list[dict[str, Any]], str]:
    """Return regular completed bars or one concise reason they cannot speak."""
    moment = _market_now(now)
    seen: set[datetime] = set()
    out: list[dict[str, Any]] = []
    for bar in bars or ():
        stamp = _stamp(_field(bar, "dt") or _field(bar, "timestamp") or _field(bar, "time"))
        if stamp is None or stamp.tzinfo is None:
            return [], "M5 timestamps must be timezone-aware"
        market_stamp = stamp.astimezone(MARKET_TZ)
        if market_stamp in seen:
            return [], "duplicate M5 bar timestamp"
        seen.add(market_stamp)
        values = {name: _number(_field(bar, name)) for name in ("open", "high", "low", "close", "volume")}
        if any(values[name] is None or values[name] <= 0 for name in ("open", "high", "low", "close")):
            continue
        if values["low"] > values["open"] or values["low"] > values["close"] or values["high"] < values["open"] or values["high"] < values["close"]:
            continue
        if values["volume"] is None or values["volume"] < 0:
            continue
        try:
            close = session_close(market_stamp.date())
            regular = is_session(market_stamp.date()) and datetime.combine(market_stamp.date(), _OPEN, MARKET_TZ) <= market_stamp and market_stamp + timedelta(minutes=5) <= close
        except Exception:
            regular = False
        if regular and market_stamp + timedelta(minutes=5) <= moment:
            out.append({"dt": stamp, "market_dt": market_stamp, **values})
    out.sort(key=lambda row: row["market_dt"])
    return out, ""


def _m5_reading(row: dict[str, Any], bars: Sequence[Any], now: datetime) -> None:
    valid, reason = _valid_m5(bars, now)
    if reason:
        row["m5_reason"] = reason
        return
    if not valid:
        row["m5_reason"] = "no completed regular-session M5 bars"
        return
    expected = _last_m5_start(now)
    last = valid[-1]
    if expected == datetime.min.replace(tzinfo=MARKET_TZ) or expected - last["market_dt"] > timedelta(minutes=15):
        row.update(m5_status="stale", m5_reason="M5 bars are stale", m5_as_of=last["dt"].isoformat())
        return
    by_stamp = {bar["market_dt"]: bar for bar in valid}
    stamps = [last["market_dt"] - timedelta(minutes=5 * offset) for offset in range(6, -1, -1)]
    if any(stamp not in by_stamp for stamp in stamps):
        row.update(m5_reason="M5 bars have a gap in the 30-minute window", m5_as_of=last["dt"].isoformat())
        return
    then = by_stamp[stamps[0]]["close"]
    if then is None or then == 0:
        row.update(m5_reason="M5 30-minute base close is unusable", m5_as_of=last["dt"].isoformat())
        return
    change = (last["close"] - then) / then * 100.0
    row.update(
        m5_status="measured", m5_reason="", m5_as_of=last["dt"].isoformat(),
        m5_change_30m_pct=round(change, 6),
        m5_direction="up" if change > 0 else "down" if change < 0 else "flat",
    )
    # A session VWAP means every completed regular M5 bar since this session's
    # open was present and carried valid volume.  A tail is never called VWAP.
    start = datetime.combine(last["market_dt"].date(), _OPEN, MARKET_TZ)
    full_stamps = []
    cursor = start
    while cursor <= last["dt"]:
        full_stamps.append(cursor)
        cursor += timedelta(minutes=5)
    if all(stamp in by_stamp for stamp in full_stamps):
        numerator = sum(((bar["high"] + bar["low"] + bar["close"]) / 3.0) * bar["volume"] for bar in (by_stamp[stamp] for stamp in full_stamps))
        denominator = sum(by_stamp[stamp]["volume"] for stamp in full_stamps)
        if denominator > 0:
            vwap = numerator / denominator
            row["m5_vs_session_vwap"] = "above" if last["close"] > vwap else "below" if last["close"] < vwap else "at"


def _valid_d1(bars: Sequence[Any], now: datetime) -> tuple[list[dict[str, Any]], str]:
    try:
        cutoff = _last_completed_exchange_session(now)
    except Exception:
        return [], "latest completed D1 session is unavailable"
    seen: set[date] = set()
    out: list[dict[str, Any]] = []
    for bar in bars or ():
        stamp = _stamp(_field(bar, "dt") or _field(bar, "date") or _field(bar, "timestamp"))
        if stamp is None:
            continue
        day = stamp.astimezone(MARKET_TZ).date() if stamp.tzinfo else stamp.date()
        if day in seen:
            return [], "duplicate D1 bar date"
        seen.add(day)
        values = {name: _number(_field(bar, name)) for name in ("open", "high", "low", "close")}
        if any(value is None or value <= 0 for value in values.values()):
            continue
        if values["low"] > values["open"] or values["low"] > values["close"] or values["high"] < values["open"] or values["high"] < values["close"]:
            continue
        try:
            regular = is_session(day)
        except Exception:
            regular = False
        if regular and day <= cutoff:
            out.append({"day": day, **values})
    out.sort(key=lambda row: row["day"])
    return out, ""


def _d1_reading(row: dict[str, Any], bars: Sequence[Any], now: datetime) -> None:
    valid, reason = _valid_d1(bars, now)
    if reason:
        row["d1_reason"] = reason
        return
    if not valid:
        row["d1_reason"] = "no completed regular-session D1 bars"
        return
    try:
        expected = _last_completed_exchange_session(now)
    except Exception:
        row["d1_reason"] = "latest completed D1 session is unavailable"
        return
    last = valid[-1]
    if last["day"] != expected:
        row.update(d1_status="stale", d1_reason="D1 bars are stale", d1_as_of=last["day"].isoformat())
        return
    if len(valid) < 20:
        row.update(d1_reason="not enough completed D1 bars for SMA20 and five-session change", d1_as_of=last["day"].isoformat())
        return
    expected_days = [expected]
    for _ in range(19):
        expected_days.append(previous_session(expected_days[-1]))
    if [item["day"] for item in valid[-20:]] != list(reversed(expected_days)):
        row.update(d1_reason="D1 bars have a gap in the completed-session history", d1_as_of=last["day"].isoformat())
        return
    base = valid[-6]["close"]
    if base == 0:
        row.update(d1_reason="D1 five-session base close is unusable", d1_as_of=last["day"].isoformat())
        return
    change = (last["close"] - base) / base * 100.0
    sma20 = sum(item["close"] for item in valid[-20:]) / 20.0
    row.update(
        d1_status="measured", d1_reason="", d1_as_of=last["day"].isoformat(),
        d1_change_5d_pct=round(change, 6),
        d1_vs_sma20="above" if last["close"] > sma20 else "below" if last["close"] < sma20 else "at",
    )


def _day_reading(row: dict[str, Any], m5_bars: Sequence[Any], d1_bars: Sequence[Any], now: datetime) -> None:
    """TJ-14A item 4: the day's change, the place in its range, the two sides.

    Measured from COMPLETED bars only, exactly like the two readings above it.
    The day's change is read against the PRIOR SESSION'S CLOSE rather than this
    session's open, because "up on the day" is what a trader means by it and an
    opening gap is part of the move; the prior high and low come from the same
    completed daily bar, so all three facts rest on one session.
    """
    valid, reason = _valid_m5(m5_bars, now)
    if reason or not valid:
        return
    last = valid[-1]
    session_day = last["market_dt"].date()
    today = [bar for bar in valid if bar["market_dt"].date() == session_day]
    high = max(bar["high"] for bar in today)
    low = min(bar["low"] for bar in today)
    if high > low:
        # A fraction of ONE, so it is rounded far finer than the percentages
        # above it: ten places keep the stored row small without rounding a
        # place in the range into a different place in the range.
        row["day_range_place"] = round(
            min(1.0, max(0.0, (last["close"] - low) / (high - low))), 10
        )
    daily, d1_reason = _valid_d1(d1_bars, now)
    if d1_reason:
        return
    prior = [bar for bar in daily if bar["day"] < session_day]
    if not prior:
        return
    previous = prior[-1]
    if previous["close"] > 0:
        row["day_change_pct"] = round(
            (last["close"] - previous["close"]) / previous["close"] * 100.0, 6
        )
    row["vs_prior_high"] = _side(last["close"], previous["high"])
    row["vs_prior_low"] = _side(last["close"], previous["low"])


def _side(value: float, level: float) -> str:
    return "above" if value > level else "below" if value < level else "at"


def _direction(value: float) -> str:
    return "up" if value > 0 else "down" if value < 0 else "flat"


def _line(inputs: Sequence[str], missing: Sequence[str], *, needs: str = "day reading", **extra: Any) -> dict[str, Any]:
    """One derived line, which always NAMES the readings it rests on.

    A missing input makes THIS line `unmeasured` and says which reading was
    absent; it never poisons a line that rests on other readings, and it is
    never a zero (plan.md §5: missing data is uncertainty). `needs` names the
    reading that is actually missing - `sectors_above_vwap` rests on the
    session VWAP, not on the day's change, and saying "day reading" there sent
    a reader looking for the wrong hole.
    """
    row: dict[str, Any] = {
        "status": "unmeasured" if missing else "measured",
        "inputs": [str(name) for name in inputs],
        "reason": (
            f"no completed {needs} for " + ", ".join(sorted(set(missing)))
            if missing
            else ""
        ),
        "value": None,
    }
    if missing:
        for key in extra:
            row[key] = None
        return row
    row.update(extra)
    return row


def _derived_block(readings: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The reads the trader used to type out by hand, measured once.

    Nothing here is a score, a signal or a threshold: every line is a stated
    number with the readings it came from, printed on the card and stored on
    the row. A line is `measured` only when every reading it names could speak.
    """
    by_symbol = {str(row.get("symbol") or ""): row for row in readings}

    def day(symbol: str) -> float | None:
        row = by_symbol.get(symbol)
        return _number(row.get("day_change_pct")) if row else None

    def m30(symbol: str) -> float | None:
        row = by_symbol.get(symbol)
        return _number(row.get("m5_change_30m_pct")) if row else None

    def missing_day(names: Sequence[str]) -> list[str]:
        return [name for name in names if day(name) is None]

    block: dict[str, dict[str, Any]] = {}

    pair = ("RSP", "SPY")
    absent = missing_day(pair)
    block["breadth"] = _line(pair, absent)
    if not absent:
        block["breadth"]["value"] = round(day("RSP") - day("SPY"), 6)

    fear_inputs = ("VXX", "SPY")
    absent = missing_day(fear_inputs)
    block["fear"] = _line(
        fear_inputs, absent, vxx_direction=None, spy_direction=None, divergence=None
    )
    if not absent:
        vxx, spy = day("VXX"), day("SPY")
        block["fear"].update(
            vxx_direction=_direction(vxx),
            spy_direction=_direction(spy),
            # Both up or both down is the day the trader would have written out:
            # the hedge and the index are saying the same thing, which they
            # normally do not.
            divergence=bool((vxx > 0 and spy > 0) or (vxx < 0 and spy < 0)),
        )

    for name, symbol in (("rates", "TLT"), ("oil", "USO")):
        absent = missing_day((symbol,))
        block[name] = _line((symbol,), absent)
        if not absent:
            block[name]["value"] = day(symbol)

    absent = sorted(set(missing_day(SECTORS)) | {name for name in SECTORS if m30(name) is None})
    leaders = _line(SECTORS, absent, day=None, m30=None)
    laggards = _line(SECTORS, absent, day=None, m30=None)
    if not absent:
        # TWO rankings, never one list read twice: a sector can lead the day
        # while it is falling over the last half hour, and that disagreement is
        # the whole point of printing both.
        by_day = sorted(SECTORS, key=lambda name: (-day(name), name))
        by_m30 = sorted(SECTORS, key=lambda name: (-m30(name), name))
        leaders.update(day=by_day[:_RANK_DEPTH], m30=by_m30[:_RANK_DEPTH])
        # Worst first, so a reader can stop after one name. It is the REVERSE
        # of the leader order, so a tie breaks the same way at both ends.
        laggards.update(
            day=list(reversed(by_day))[:_RANK_DEPTH],
            m30=list(reversed(by_m30))[:_RANK_DEPTH],
        )
    block["sector_leaders"] = leaders
    block["sector_laggards"] = laggards

    battle = OFFENSE + DEFENSE
    absent = missing_day(battle)
    block["offense_vs_defense"] = _line(battle, absent)
    if not absent:
        offense = sum(day(name) for name in OFFENSE) / len(OFFENSE)
        defense = sum(day(name) for name in DEFENSE) / len(DEFENSE)
        block["offense_vs_defense"]["value"] = round(offense - defense, 6)

    vwap_absent = [
        name
        for name in SECTORS
        if str((by_symbol.get(name) or {}).get("m5_vs_session_vwap") or "") not in ("above", "below", "at")
    ]
    above = _line(
        SECTORS, vwap_absent, needs="session VWAP reading", count=None, denominator=None
    )
    if not vwap_absent:
        # A COUNT WITH ITS DENOMINATOR, never a bare number: "nine" means
        # nothing without "of eleven".
        above.update(
            count=sum(
                1
                for name in SECTORS
                if str(by_symbol[name].get("m5_vs_session_vwap") or "") == "above"
            ),
            denominator=len(SECTORS),
        )
    block["sectors_above_vwap"] = above
    return block


_RULES = {
    "m5": "completed regular-session 30-minute change",
    "d1": "completed-session five-day change and SMA20",
    "day": "completed regular-session change from the prior session's close, place in the day's range, and the two prior-session sides",
}


def build_context(*, now: datetime, m5_bars: Mapping[str, Sequence[Any]], d1_bars: Mapping[str, Sequence[Any]], sources: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the bounded, JSON-safe snapshot attached to one journal read.

    THE one builder: the live card and :func:`internals_at`'s rebuild from the
    durable tape both come through here, because two implementations of the
    same reading drift on the first rounding decision.
    """
    moment = _market_now(now)
    readings: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        reading = _blank(symbol)
        m5 = (m5_bars or {}).get(symbol, ())
        d1 = (d1_bars or {}).get(symbol, ())
        _m5_reading(reading, m5, moment)
        _d1_reading(reading, d1, moment)
        _day_reading(reading, m5, d1, moment)
        readings.append(reading)
    return {
        "schema": SCHEMA,
        "captured_at": now.isoformat(),
        "availability": "available",
        "reason": "",
        "rules": dict(_RULES),
        "sources": dict(sources or {"m5": "unknown", "d1": "unknown"}),
        "readings": readings,
        "derived": _derived_block(readings),
    }


def unavailable_context(*, now: datetime, reason: str) -> dict[str, Any]:
    """An explicit all-symbol absence, used when a note wins a slow fetch."""
    readings = [_blank(symbol) for symbol in SYMBOLS]
    return {
        "schema": SCHEMA, "captured_at": now.isoformat(),
        "availability": "unavailable", "reason": str(reason),
        "rules": dict(_RULES),
        "sources": {"m5": "unavailable", "d1": "unavailable"},
        "readings": readings,
        "derived": _derived_block(readings),
    }


def internals_at(session: Any, stamp: datetime, bars: Mapping[str, Any]) -> dict[str, Any]:
    """The same block, for any moment, from bars the caller supplies.

    PURE. It takes its bars as an argument (`{"m5": ..., "d1": ...}`) so a
    rebuild of a skipped hour, a note typed on the desk tab and a prediction's
    own context snapshot all read ONE function - and so this module still
    fetches nothing. :func:`internals_bars_at` is the thin loader beside it.
    """
    payload = bars or {}
    context = build_context(
        now=stamp,
        m5_bars=payload.get("m5") or {},
        d1_bars=payload.get("d1") or {},
        sources=payload.get("sources") or {"m5": "rebuilt", "d1": "rebuilt"},
    )
    # Says what it is. The readings and the derived block are byte-identical to
    # what the live card would have shown on the same bars; this key only
    # records which session was asked for.
    context["rebuilt_for_session"] = str(session or "")[:10]
    return context


#: How many completed sessions before the rebuilt one the loader will read a
#: tape for when the daily cache cannot answer. One is enough for the day facts
#: (they rest on the PRIOR session alone); the extra two cost one parquet read
#: each and cover a symbol whose previous session was never downloaded.
_TAPE_D1_LOOKBACK = 3


def _session_bar_from_tape(rows: Sequence[Any], day: date) -> dict[str, Any] | None:
    """One completed session's OHLC, built from its own completed M5 bars.

    RSP, USO and TLT have no file in the machine's daily-bar cache - the scan
    universe never fetches them, and the live card only ever escaped through
    Yahoo. Without this the rebuild could never measure breadth, rates or oil,
    which are three of the eight derived lines. The bar is not a download and
    not a guess: it is the session's own tape, which `day_review_bars` already
    holds now that it carries the internals symbols.
    """
    prices: list[tuple[datetime, dict[str, float]]] = []
    for row in rows or ():
        stamp = _stamp(_field(row, "dt") or _field(row, "timestamp") or _field(row, "time"))
        if stamp is None or stamp.tzinfo is None:
            continue
        market_stamp = stamp.astimezone(MARKET_TZ)
        if market_stamp.date() != day:
            continue
        values = {name: _number(_field(row, name)) for name in ("open", "high", "low", "close")}
        if any(value is None or value <= 0 for value in values.values()):
            continue
        prices.append((market_stamp, values))
    if not prices:
        return None
    prices.sort(key=lambda item: item[0])
    return {
        "dt": day.isoformat(),
        "open": prices[0][1]["open"],
        "high": max(item[1]["high"] for item in prices),
        "low": min(item[1]["low"] for item in prices),
        "close": prices[-1][1]["close"],
    }


def internals_bars_at(session: Any, stamp: datetime) -> dict[str, Any]:
    """Read the bars :func:`internals_at` needs, cut point-in-time.

    The thin loader, and the only part of this module that touches a store. M5
    comes from TJ-2A's durable session tape (which is why `day_review_bars`
    downloads these symbols); D1 comes from the SAME daily cache the live
    context service already reads, and - for a symbol that cache has never
    heard of - from the PRIOR sessions' tapes, one daily bar each. Both are cut
    to completed observations at or before `stamp` by the builder itself, so no
    caller can widen the cut. A symbol with nothing in either store has its
    facts `unmeasured`, never guessed; a symbol answered from the tape has its
    DAY facts measured while its five-session and SMA20 facts stay `unmeasured`
    and say why (too few completed daily bars).
    """
    day = str(session or "")[:10]

    def tape_for(target: str) -> dict[str, Any]:
        try:
            from day_review_bars import read_session_bars

            return read_session_bars(target) or {}
        except Exception:  # noqa: BLE001 - a missing tape is unmeasured, never an error
            return {}

    tape = tape_for(day)
    m5 = {symbol: tape.get(symbol) or [] for symbol in SYMBOLS if tape.get(symbol)}
    d1: dict[str, Any] = {}
    try:
        from d1_environment_store import _cached_daily_bars

        for symbol in SYMBOLS:
            rows = _cached_daily_bars(symbol)
            if rows:
                d1[symbol] = rows
    except Exception:  # noqa: BLE001
        d1 = {}

    missing = [symbol for symbol in SYMBOLS if not d1.get(symbol)]
    from_tape = False
    if missing:
        try:
            cursor = date.fromisoformat(day)
        except ValueError:
            cursor = None
        for _ in range(_TAPE_D1_LOOKBACK if cursor else 0):
            try:
                cursor = previous_session(cursor)
            except Exception:  # noqa: BLE001 - a calendar refusal ends the walk
                break
            earlier = tape_for(cursor.isoformat())
            if not earlier:
                continue
            for symbol in missing:
                bar = _session_bar_from_tape(earlier.get(symbol) or (), cursor)
                if bar is not None:
                    d1.setdefault(symbol, [])
                    d1[symbol].append(bar)
                    from_tape = True
    for symbol in missing:
        # Oldest first, the shape every D1 reader here expects.
        if d1.get(symbol):
            d1[symbol].sort(key=lambda row: str(row.get("dt") or ""))
    return {
        "m5": m5,
        "d1": d1,
        "sources": {
            "m5": "day_review_tape",
            "d1": "daily_cache+day_review_tape" if from_tape else "daily_cache",
        },
    }


def compact_for_ai(context: Any) -> Any:
    """Replace repeated journal-reading keys with a compact scalar table.

    Stored journal rows keep their ordinary, independently readable shape.  This
    projection is only for the existing bounded AI evidence package.
    """
    schema = str((context or {}).get("schema") or "") if isinstance(context, Mapping) else ""
    if schema not in (SCHEMA_V1, SCHEMA):
        return context
    readings = context.get("readings")
    if not isinstance(readings, list) or not all(isinstance(row, Mapping) for row in readings):
        return context
    keys = _V2_KEYS if schema == SCHEMA else _M5_KEYS
    # The evidence source keeps this under ``mentor.context_compact``.  The
    # durable journal's full ``mentor.context`` remains untouched.  Values
    # shared by every symbol move into ``common`` so the journal source can
    # keep a complete Mentor read within its existing evidence allowance.
    common = {
        key: context.get(key)
        for key in ("schema", "captured_at", "availability", "reason", "rules", "sources")
    }
    if schema == SCHEMA:
        # The derived lines are identical for every symbol - they ARE the
        # whole-market read - so they belong in `common` exactly once. They are
        # the reads the trader used to type by hand and they must reach the AI.
        common["derived"] = context.get("derived")
        # ...and they must SURVIVE the evidence package's depth cut. Measured
        # 2026-09-19: `ai_summary._bounded` stops six levels down, which is
        # exactly where a derived line's `inputs` and leader lists sit, so the
        # model saw "[nested content omitted]" where the sector names should
        # be. This one SCALAR sits a level higher and says the same thing in
        # the card's own words.
        common["internals"] = "\n".join(internals_lines(context))
    columns: list[str] = []
    for key in keys:
        values = [row.get(key) for row in readings]
        # These two headings keep the compact table directly readable by
        # older evidence consumers.  All remaining uniform values belong in
        # ``common`` and are restored by combining it with each row.
        if key not in {"symbol", "m5_direction"} and values and all(value == values[0] for value in values[1:]):
            common[key] = values[0]
        else:
            columns.append(key)
    return {
        # Kept at the legacy location for existing bounded-source readers;
        # ``common`` is the lossless representation used by new consumers.
        "captured_at": context.get("captured_at"),
        "common": common,
        "columns": columns,
        "rows": [[row.get(key) for key in columns] for row in readings],
    }


# ---------------------------------------------------------------------------
# what the card prints
# ---------------------------------------------------------------------------
def _pct(value: Any) -> str:
    number = _number(value)
    return "unmeasured" if number is None else f"{number:+.2f}%"


def internals_lines(context: Any) -> tuple[str, ...]:
    """The strip's text, computed OUTSIDE Qt and with no widget in sight.

    Pure, so the card does no work beyond ``setText`` and the wording is
    testable without a screen. An ``unmeasured`` line is PRINTED as unmeasured:
    a blank where a reading should be reads as calm, which is the one thing it
    is not.
    """
    if not isinstance(context, Mapping):
        return ("Internals: nothing was read.",)
    derived = context.get("derived")
    availability = str(context.get("availability") or "")
    if not isinstance(derived, Mapping):
        return (f"Internals: {availability or 'unavailable'}.",)

    def line(name: str) -> Mapping[str, Any]:
        row = derived.get(name)
        return row if isinstance(row, Mapping) else {}

    def measured(name: str) -> bool:
        return str(line(name).get("status") or "") == "measured"

    def value(name: str) -> str:
        return _pct(line(name).get("value")) if measured(name) else "unmeasured"

    def names(name: str, key: str) -> str:
        values = line(name).get(key)
        return ", ".join(str(item) for item in values) if values else "unmeasured"

    fear = line("fear")
    fear_text = (
        (
            f"VXX {fear.get('vxx_direction')} / SPY {fear.get('spy_direction')}"
            + ("  (both the same way)" if fear.get("divergence") else "")
        )
        if measured("fear")
        else "unmeasured"
    )
    above = line("sectors_above_vwap")
    above_text = (
        f"{above.get('count')} of {above.get('denominator')}"
        if measured("sectors_above_vwap")
        else "unmeasured"
    )
    rows = [
        "  ·  ".join(
            (
                f"Breadth (RSP-SPY) {value('breadth')}",
                f"Fear {fear_text}",
                f"Rates (TLT) {value('rates')}",
                f"Oil (USO) {value('oil')}",
            )
        ),
        "  ·  ".join(
            (
                f"Leaders {names('sector_leaders', 'day')}",
                f"Laggards {names('sector_laggards', 'day')}",
                f"Offense-defense {value('offense_vs_defense')}",
                f"Sectors above VWAP {above_text}",
            )
        ),
    ]
    if availability and availability != "available":
        rows.insert(
            0, f"Internals {availability}: {context.get('reason') or 'no reason given'}"
        )
    return tuple(rows)


__all__ = [
    "DEFENSE", "DERIVED_LINES", "OFFENSE", "SCHEMA", "SCHEMA_V1", "SECTORS",
    "SYMBOLS", "build_context", "compact_for_ai", "internals_at",
    "internals_bars_at", "internals_lines", "unavailable_context",
]
