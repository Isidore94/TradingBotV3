"""The Movers board's pure model: pops, SPY pullback state, dip-strong names.

Display-only ranking (trader, 2026-09-23: "what's the strongest thing moving
right now" and "what's strong during a SPY pullback"). No Qt, no network, no
project I/O; the service hands bars in and gets plain dicts back.

Rules kept here:
- Completed M5 bars only (`completed_bars.completed_m5_bars`); a forming bar
  never ranks.
- Naive bar stamps are market-local wall time (`local_tz`) and are converted to
  New York time before any session maths; aware stamps are converted, never
  stripped.
- Missing data is UNKNOWN: an unmeasurable RVOL is None (neutral weight, shown
  "—"), an unmeasurable ATR drops the row from the ranked lists, missing SPY
  bars make the market state "unknown" and light nothing.

The RVOL baseline helpers (`build_rvol_baseline`, `recent_rvol`) are pure and
importable on their own so other tools can share them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import strength_scan
from completed_bars import bar_time, completed_m5_bars

NY_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
BAR_MINUTES = 5

# ---------------------------------------------------------------- tunables
#: Bars in the "pop" move (15 minutes) and the longer read (30 minutes).
POP_BARS = 3
POP_LONG_BARS = 6
#: M5 ATR period used to normalise every move.
MOVERS_ATR_PERIOD = 14
#: A pop must move at least this many ATRs over POP_BARS to be listed.
POP_MIN_ATR_MOVE = 0.5
#: RVOL weight: score x sqrt(clamp(rvol)). None -> 1.0 (no bonus, no penalty).
RVOL_WEIGHT_MIN = 0.5
RVOL_WEIGHT_MAX = 4.0
#: Floors. The price floor is the Strength Board's; volume is completed-bar
#: shares so far today.
MIN_PRICE = strength_scan.MIN_PRICE
MIN_SESSION_VOLUME = 50_000.0
#: SPY pullback: last completed close this far below the session high (0.30%).
PULLBACK_MIN_PCT = 0.30
#: The session high must be on bar index >= this (i.e. after the first 2 bars).
PULLBACK_MIN_HIGH_INDEX = 2
#: A series whose last completed bar starts before
#: floor5(now) - 5 min x (STALE_BARS_ALLOWED + 1) is stale: not ranked, and a
#: stale SPY makes the market state unknown.
STALE_BARS_ALLOWED = 1
#: RVOL baseline: prior sessions averaged, and the fewest that still count.
RVOL_BASELINE_SESSIONS = 20
RVOL_BASELINE_MIN_SESSIONS = 5
#: Rows kept per list.
MOVERS_TOP_N = 15
#: "ext": more than this many ATRs from session VWAP (information only).
EXT_ATR = 2.0
#: Group tag: this many names of one industry inside a list's top N.
GROUP_MIN_COUNT = 3
GROUP_TOP_N = 15
GROUP_LABEL_CHARS = 10
GROUP_ABBREVIATIONS = {
    "semiconductor": "Semis",
    "software": "Software",
    "biotechnology": "Biotech",
    "banks": "Banks",
    "oil & gas": "Oil&Gas",
    "internet": "Internet",
    "drug manufacturers": "Pharma",
    "solar": "Solar",
    "gold": "Gold",
    "auto": "Autos",
}


# ---------------------------------------------------------------- time
def market_time(stamp: Any, local_tz: tzinfo | None = None) -> datetime | None:
    """A bar stamp as an aware New York datetime. Naive = market-local wall time."""
    if isinstance(stamp, datetime):
        value = stamp
    elif isinstance(stamp, date):
        value = datetime(stamp.year, stamp.month, stamp.day)
    else:
        value = bar_time({"dt": stamp})
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=local_tz or NY_TZ)
    return value.astimezone(NY_TZ)


def session_offset(moment: datetime) -> int | None:
    """Bar index inside the regular session (09:30 NY = 0), None outside it."""
    clock = moment.time()
    if clock < SESSION_OPEN or clock >= SESSION_CLOSE:
        return None
    minutes = (moment.hour * 60 + moment.minute) - (SESSION_OPEN.hour * 60 + SESSION_OPEN.minute)
    return minutes // BAR_MINUTES


def normalize_bars(
    bars: Iterable[Any], *, now: datetime, local_tz: tzinfo | None = None
) -> list[dict[str, Any]]:
    """Completed regular-session bars as dicts with an aware NY `dt`, ascending."""
    moment = now if now.tzinfo is not None else now.replace(tzinfo=local_tz or NY_TZ)
    out: list[dict[str, Any]] = []
    for bar in bars or ():
        stamp = market_time(bar_time(bar), local_tz)
        if stamp is None or session_offset(stamp) is None:
            continue
        values = {}
        for key in ("open", "high", "low", "close", "volume"):
            raw = bar.get(key) if isinstance(bar, Mapping) else getattr(bar, key, None)
            values[key] = _finite(raw)
        if None in (values["open"], values["high"], values["low"], values["close"]):
            continue
        values["dt"] = stamp
        out.append(values)
    out.sort(key=lambda row: row["dt"])
    return completed_m5_bars(out, now=moment.astimezone(NY_TZ))


def split_today(
    bars: Sequence[Mapping[str, Any]], today: date | None = None
) -> tuple[list, list]:
    """(prior-session bars, today's bars). `today` defaults to the last bar's NY date;
    a series whose last bar is older than `today` has no bars today."""
    if not bars:
        return [], []
    today = today or bars[-1]["dt"].date()
    return (
        [bar for bar in bars if bar["dt"].date() < today],
        [bar for bar in bars if bar["dt"].date() == today],
    )


# ---------------------------------------------------------------- RVOL
def build_rvol_baseline(
    bars: Iterable[Any],
    *,
    before: date,
    local_tz: tzinfo | None = None,
    sessions: int = RVOL_BASELINE_SESSIONS,
    min_sessions: int = RVOL_BASELINE_MIN_SESSIONS,
) -> dict[int, float] | None:
    """Mean volume per session bar offset over the last `sessions` sessions before `before`.

    A session that never reached an offset contributes nothing there (missing,
    not zero). None when fewer than `min_sessions` sessions are available.
    """
    by_session: dict[date, dict[int, float]] = {}
    for bar in bars or ():
        stamp = market_time(bar_time(bar), local_tz)
        if stamp is None or stamp.date() >= before:
            continue
        offset = session_offset(stamp)
        if offset is None:
            continue
        raw = bar.get("volume") if isinstance(bar, Mapping) else getattr(bar, "volume", None)
        volume = _finite(raw)
        if volume is None or volume < 0:
            continue
        by_session.setdefault(stamp.date(), {})[offset] = volume
    days = sorted(by_session)[-max(1, int(sessions)):]
    if len(days) < max(1, int(min_sessions)):
        return None
    sums: dict[int, list[float]] = {}
    for day in days:
        for offset, volume in by_session[day].items():
            sums.setdefault(offset, []).append(volume)
    return {offset: sum(values) / len(values) for offset, values in sums.items()}


def recent_rvol(
    today_bars: Sequence[Mapping[str, Any]],
    baseline: Mapping[int, float] | None,
    *,
    bars: int = POP_BARS,
) -> float | None:
    """Mean of volume / baseline-at-same-offset over the last `bars` bars. None if unmeasurable."""
    if not baseline or bars <= 0 or len(today_bars) < bars:
        return None
    ratios: list[float] = []
    for bar in today_bars[-bars:]:
        offset = session_offset(bar["dt"])
        volume = _finite(bar.get("volume"))
        mean = baseline.get(offset) if offset is not None else None
        if volume is None or mean is None or mean <= 0:
            return None
        ratios.append(volume / mean)
    return sum(ratios) / len(ratios)


def rvol_weight(rvol: float | None) -> float:
    """Score multiplier. Unknown RVOL is neutral, never a bonus."""
    if rvol is None:
        return 1.0
    return math.sqrt(min(RVOL_WEIGHT_MAX, max(RVOL_WEIGHT_MIN, float(rvol))))


# ---------------------------------------------------------------- market state
@dataclass(frozen=True)
class MarketState:
    state: str  # "unknown" | "up_day" | "down_day" | "flat"
    pullback: bool = False  # up day, SPY off its high
    bounce: bool = False  # down day, SPY off its low
    extreme_time: str = ""  # HH:MM NY of the pullback/bounce start bar
    extreme_price: float | None = None
    start_dt: datetime | None = None
    spy_from_extreme_pct: float | None = None  # last close vs the high/low
    spy_day_pct: float | None = None
    spy_last: float | None = None
    spy_vwap: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_dt"] = self.start_dt.isoformat() if self.start_dt else ""
        return data


def freshness_cutoff(now: datetime) -> datetime:
    """Oldest bar START still fresh: floor5(now) - 5 min x (STALE_BARS_ALLOWED + 1)."""
    ny = now.astimezone(NY_TZ)
    floor = ny.replace(minute=ny.minute - ny.minute % BAR_MINUTES, second=0, microsecond=0)
    return floor - timedelta(minutes=BAR_MINUTES * (STALE_BARS_ALLOWED + 1))


def market_state(
    spy_bars: Sequence[Mapping[str, Any]],
    today_date: date | None = None,
    *,
    fresh_after: datetime | None = None,
) -> MarketState:
    """SPY up/down day and pullback/bounce from normalised completed bars.

    Pullback: SPY above its session open, its session high made while above
    the running session VWAP, and now >= PULLBACK_MIN_PCT off that high (it may
    be below VWAP now). Bounce mirrors it on a down day. SPY bars older than
    `fresh_after` make the state unknown.
    """
    prior, today = split_today(spy_bars, today_date)
    if not today:
        return MarketState("unknown", reason="no SPY bars")
    if fresh_after is not None and today[-1]["dt"] < fresh_after:
        return MarketState("unknown", spy_last=today[-1]["close"], reason="SPY bars stale")
    try:
        from chart_snapshot import session_vwap_series

        vwaps = session_vwap_series(today)["vwap"]
    except Exception:
        vwaps = [None] * len(today)
    vwap = vwaps[-1] if vwaps else None
    last = today[-1]["close"]
    session_open = today[0]["open"]
    reference = prior[-1]["close"] if prior else session_open
    day_pct = _pct(last, reference)
    if vwap is None:
        return MarketState(
            "unknown", spy_last=last, spy_day_pct=day_pct, reason="SPY VWAP unknown"
        )
    common = {"spy_last": last, "spy_vwap": vwap, "spy_day_pct": day_pct}
    last_index = len(today) - 1

    def turn_on(index: int, off: float | None, beyond: bool) -> bool:
        return (
            PULLBACK_MIN_HIGH_INDEX <= index < last_index
            and off is not None
            and beyond
        )

    if last > session_open:
        index = max(range(len(today)), key=lambda i: (today[i]["high"], -i))
        high = today[index]["high"]
        off = _pct(last, high)
        at_high = vwaps[index]
        made_above = at_high is not None and today[index]["close"] > at_high
        on = made_above and turn_on(index, off, off is not None and off <= -PULLBACK_MIN_PCT)
        if on or last > vwap:
            return MarketState(
                "up_day", pullback=on, extreme_time=today[index]["dt"].strftime("%H:%M"),
                extreme_price=high, start_dt=today[index]["dt"] if on else None,
                spy_from_extreme_pct=off, **common,
            )
    if last < session_open:
        index = min(range(len(today)), key=lambda i: (today[i]["low"], i))
        low = today[index]["low"]
        off = _pct(last, low)
        at_low = vwaps[index]
        made_below = at_low is not None and today[index]["close"] < at_low
        on = made_below and turn_on(index, off, off is not None and off >= PULLBACK_MIN_PCT)
        if on or last < vwap:
            return MarketState(
                "down_day", bounce=on, extreme_time=today[index]["dt"].strftime("%H:%M"),
                extreme_price=low, start_dt=today[index]["dt"] if on else None,
                spy_from_extreme_pct=off, **common,
            )
    return MarketState("flat", **common)


# ---------------------------------------------------------------- rows
@dataclass(frozen=True)
class MoverRow:
    symbol: str
    last: float | None = None
    move15_pct: float | None = None
    move30_pct: float | None = None
    day_pct: float | None = None
    rvol: float | None = None
    vs_spy15_pct: float | None = None
    atr: float | None = None
    pop_score: float | None = None  # signed, ATRs x RVOL weight
    since_start_pct: float | None = None  # since the pullback/bounce start bar
    dip_score: float | None = None  # signed excess vs SPY since start, ATRs x weight
    session_volume: float | None = None
    passes_floors: bool = False
    stale: bool = False
    note: str = ""
    focus_side: str = ""
    # Stretch / level (information only; never part of a score).
    hod: float | None = None
    lod: float | None = None
    session_vwap: float | None = None
    prev_high: float | None = None
    prev_low: float | None = None
    from_hod_atr: float | None = None  # <= 0: ATRs below the session high
    from_lod_atr: float | None = None  # >= 0: ATRs above the session low
    from_vwap_atr: float | None = None
    hod_break: bool = False  # last completed bar made a new session high
    lod_break: bool = False
    ext_up: bool = False  # more than EXT_ATR above VWAP
    ext_down: bool = False
    er: bool = False  # reports today or reported after the last close
    group: str = ""  # short industry label when 3+ share a list's top 15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _window_move_pct(bars: Sequence[Mapping[str, Any]], start: datetime, end: datetime):
    """% from the open of the first bar at/after `start` to the close of the last bar at/before `end`."""
    inside = [bar for bar in bars if start <= bar["dt"] <= end]
    if not inside:
        return None
    return _pct(inside[-1]["close"], inside[0]["open"])


def _close_at(bars: Sequence[Mapping[str, Any]], stamp: datetime) -> float | None:
    for bar in bars:
        if bar["dt"] == stamp:
            return bar["close"]
    return None


def measure_symbol(
    symbol: str,
    bars: Sequence[Mapping[str, Any]],
    *,
    baseline: Mapping[int, float] | None,
    spy_bars: Sequence[Mapping[str, Any]],
    state: MarketState,
    reference_end: datetime | None,
    today_date: date | None = None,
) -> MoverRow:
    """Every Movers number for one symbol from normalised completed bars."""
    prior, today = split_today(bars, today_date)
    spy_today = split_today(spy_bars, today_date)[1] if spy_bars else []
    if not today:
        return MoverRow(symbol, note="no bars today")
    last_bar = today[-1]
    last = last_bar["close"]
    atr = strength_scan.atr(list(bars), period=MOVERS_ATR_PERIOD)
    reference = prior[-1]["close"] if prior else today[0]["open"]
    day_pct = _pct(last, reference)
    volumes = [bar.get("volume") for bar in today]
    session_vol = None if any(v is None for v in volumes) else float(sum(volumes))
    stale = bool(reference_end is not None and last_bar["dt"] < reference_end)
    passes = (
        last > MIN_PRICE and session_vol is not None and session_vol >= MIN_SESSION_VOLUME
    )
    move15 = _pct(last, today[-POP_BARS]["open"]) if len(today) >= POP_BARS else None
    move30 = _pct(last, today[-POP_LONG_BARS]["open"]) if len(today) >= POP_LONG_BARS else None
    rvol = recent_rvol(today, baseline, bars=POP_BARS)
    vs_spy = None
    if move15 is not None and spy_today:
        spy_move = _window_move_pct(spy_today, today[-POP_BARS]["dt"], last_bar["dt"])
        if spy_move is not None:
            vs_spy = move15 - spy_move
    pop_score = None
    if move15 is not None and atr and atr > 0:
        move_price = last - today[-POP_BARS]["open"]
        pop_score = (move_price / atr) * rvol_weight(rvol)

    since = dip = None
    note = ""
    if state.start_dt is not None:
        start_close = _close_at(today, state.start_dt)
        spy_start = _close_at(spy_today, state.start_dt)
        spy_last = spy_today[-1]["close"] if spy_today else None
        if start_close is None or spy_start is None or spy_last is None:
            note = "no bar at the pullback start"
        else:
            since = _pct(last, start_close)
            spy_since = _pct(spy_last, spy_start)
            span = [bar for bar in today if bar["dt"] > state.start_dt]
            if since is not None and spy_since is not None and atr and atr > 0 and span:
                excess_price = start_close * (since - spy_since) / 100.0
                span_rvol = recent_rvol(today, baseline, bars=len(span))
                dip = (excess_price / atr) * rvol_weight(span_rvol)
    if atr is None:
        note = note or "ATR unmeasurable"
    levels = _levels(prior, today, atr)
    return MoverRow(
        symbol=symbol, last=last, move15_pct=move15, move30_pct=move30, day_pct=day_pct,
        rvol=rvol, vs_spy15_pct=vs_spy, atr=atr, pop_score=pop_score,
        since_start_pct=since, dip_score=dip, session_volume=session_vol,
        passes_floors=passes, stale=stale, note=note, **levels,
    )


def _levels(prior: Sequence[Mapping[str, Any]], today: Sequence[Mapping[str, Any]],
            atr: float | None) -> dict[str, Any]:
    """HOD/LOD/VWAP distances in ATRs and the break/extension tags."""
    last = today[-1]["close"]
    hod = max(bar["high"] for bar in today)
    lod = min(bar["low"] for bar in today)
    earlier = today[:-1]
    hod_break = bool(earlier) and today[-1]["high"] > max(bar["high"] for bar in earlier)
    lod_break = bool(earlier) and today[-1]["low"] < min(bar["low"] for bar in earlier)
    prev_session = [bar for bar in prior if prior and bar["dt"].date() == prior[-1]["dt"].date()]
    try:
        from chart_snapshot import session_vwap_series

        vwap = session_vwap_series(list(today))["vwap"][-1]
    except Exception:
        vwap = None
    out: dict[str, Any] = {
        "hod": hod, "lod": lod, "session_vwap": vwap,
        "prev_high": max(bar["high"] for bar in prev_session) if prev_session else None,
        "prev_low": min(bar["low"] for bar in prev_session) if prev_session else None,
        "hod_break": hod_break, "lod_break": lod_break,
    }
    if atr and atr > 0:
        out["from_hod_atr"] = (last - hod) / atr
        out["from_lod_atr"] = (last - lod) / atr
        if vwap is not None:
            out["from_vwap_atr"] = (last - vwap) / atr
            out["ext_up"] = out["from_vwap_atr"] > EXT_ATR
            out["ext_down"] = out["from_vwap_atr"] < -EXT_ATR
    return out


# ---------------------------------------------------------------- tags
def earnings_symbols(events: Iterable[Mapping[str, Any]], *, today: date, previous: date) -> set[str]:
    """Names reporting today (any session) or after the previous session's close (AMC)."""
    out: set[str] = set()
    for event in events or ():
        symbol = str(event.get("ticker") or event.get("symbol") or "").strip().upper()
        when = str(event.get("earnings_date") or "")[:10]
        session = str(event.get("release_session") or "").strip().upper()
        if not symbol:
            continue
        if when == today.isoformat() or (when == previous.isoformat() and session == "AMC"):
            out.add(symbol)
    return out


def short_group(industry: str) -> str:
    """A compact industry label for a narrow cell."""
    text = str(industry or "").strip()
    for long_name, short in GROUP_ABBREVIATIONS.items():
        if text.lower().startswith(long_name):
            return short
    return text[:GROUP_LABEL_CHARS]


def apply_group_tags(board: dict[str, Any], industry_by_symbol: Mapping[str, str]) -> dict[str, Any]:
    """Tag rows whose industry has GROUP_MIN_COUNT+ names in a list's top GROUP_TOP_N."""
    groups: dict[str, dict[str, list]] = {}
    for mode in ("pop", "dip"):
        groups[mode] = {}
        for side in ("long", "short"):
            rows = ((board.get(mode) or {}).get(side)) or []
            counts: dict[str, int] = {}
            for row in rows[:GROUP_TOP_N]:
                industry = industry_by_symbol.get(str(row.get("symbol") or "").upper())
                if industry:
                    counts[industry] = counts.get(industry, 0) + 1
            hot = {name for name, count in counts.items() if count >= GROUP_MIN_COUNT}
            for row in rows:
                industry = industry_by_symbol.get(str(row.get("symbol") or "").upper())
                row["group"] = short_group(industry) if industry in hot else ""
            groups[mode][side] = [
                [short_group(name), counts[name]]
                for name in sorted(hot, key=lambda n: (-counts[n], n))
            ]
    board["groups"] = groups
    return board


def apply_persistence(board: dict[str, Any], memory: dict[str, Any], *, session: date) -> dict[str, Any]:
    """Stamp each listed row with `streak` (consecutive ticks on that list) and
    `rank_change` (+ = up, None = new). Returns the memory for the next tick."""
    if memory.get("session") != session:
        memory = {"session": session, "lists": {}}
    lists = memory["lists"]
    fresh: dict[str, dict[str, tuple[int, int]]] = {}
    for mode in ("pop", "dip"):
        for side in ("long", "short"):
            key = f"{mode}:{side}"
            before = lists.get(key, {})
            now_list: dict[str, tuple[int, int]] = {}
            for rank, row in enumerate(((board.get(mode) or {}).get(side)) or [], start=1):
                symbol = str(row.get("symbol") or "").upper()
                previous = before.get(symbol)
                row["streak"] = previous[1] + 1 if previous else 1
                row["rank_change"] = previous[0] - rank if previous else None
                now_list[symbol] = (rank, row["streak"])
            fresh[key] = now_list
    memory["lists"] = fresh
    return memory


# ---------------------------------------------------------------- board
def build_movers_board(
    bars_by_symbol: Mapping[str, Sequence[Any]],
    spy_bars: Sequence[Any] | None,
    *,
    now: datetime,
    baselines: Mapping[str, Mapping[int, float] | None] | None = None,
    focus_by_side: Mapping[str, Iterable[str]] | None = None,
    local_tz: tzinfo | None = None,
    top_n: int = MOVERS_TOP_N,
    earnings: Iterable[str] | None = None,
) -> dict[str, Any]:
    """The whole board as plain dicts (safe to emit across threads).

    `focus_by_side` is {"long": [...], "short": [...]} of the trader's Focus
    names; `earnings` the names to tag ER. Lists: pop/dip/mine, each
    {"long": rows, "short": rows}.
    """
    baselines = baselines or {}
    er_names = {str(s or "").strip().upper() for s in earnings or ()}
    spy = normalize_bars(spy_bars or (), now=now, local_tz=local_tz)
    moment = now if now.tzinfo is not None else now.replace(tzinfo=local_tz or NY_TZ)
    today_date = moment.astimezone(NY_TZ).date()
    cutoff = freshness_cutoff(moment)
    state = market_state(spy, today_date, fresh_after=cutoff)
    normalised = {
        str(sym).strip().upper(): normalize_bars(bars, now=now, local_tz=local_tz)
        for sym, bars in (bars_by_symbol or {}).items()
        if str(sym or "").strip()
    }
    normalised.pop("SPY", None)
    reference_end = cutoff

    rows: dict[str, MoverRow] = {}
    for symbol, bars in normalised.items():
        rows[symbol] = measure_symbol(
            symbol, bars, baseline=baselines.get(symbol), spy_bars=spy,
            state=state, reference_end=reference_end, today_date=today_date,
        )
        if symbol in er_names:
            rows[symbol] = replace(rows[symbol], er=True)

    def rankable(row: MoverRow) -> bool:
        return row.passes_floors and not row.stale and row.atr is not None

    pop_long = sorted(
        (r for r in rows.values() if rankable(r) and r.pop_score is not None
         and r.pop_score >= POP_MIN_ATR_MOVE),
        key=lambda r: (-r.pop_score, r.symbol),
    )
    pop_short = sorted(
        (r for r in rows.values() if rankable(r) and r.pop_score is not None
         and r.pop_score <= -POP_MIN_ATR_MOVE),
        key=lambda r: (r.pop_score, r.symbol),
    )
    dip_long: list[MoverRow] = []
    dip_short: list[MoverRow] = []
    if state.pullback:
        dip_long = sorted(
            (r for r in rows.values() if rankable(r) and r.dip_score is not None
             and r.dip_score >= 0),
            key=lambda r: (-r.dip_score, r.symbol),
        )
    if state.bounce:
        dip_short = sorted(
            (r for r in rows.values() if rankable(r) and r.dip_score is not None
             and r.dip_score <= 0),
            key=lambda r: (r.dip_score, r.symbol),
        )

    mine: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    for side in ("long", "short"):
        for raw in (focus_by_side or {}).get(side, ()) or ():
            symbol = str(raw or "").strip().upper()
            if not symbol or any(item["symbol"] == symbol for item in mine[side]):
                continue
            row = rows.get(symbol) or MoverRow(symbol, note="no bars")
            data = row.to_dict()
            data["focus_side"] = side
            mine[side].append(data)

    spy_today = split_today(spy, today_date)[1]
    spy_last_bar = spy_today[-1]["dt"] if spy_today else None
    return {
        "as_of": spy_last_bar.isoformat(timespec="seconds") if spy_last_bar else "",
        "as_of_stale": spy_last_bar is None or spy_last_bar < cutoff,
        "tick_at": now.isoformat(timespec="seconds"),
        "fresh": sum(1 for r in rows.values() if r.last is not None and not r.stale),
        "state": state.to_dict(),
        "pop": {"long": [r.to_dict() for r in pop_long[:top_n]],
                "short": [r.to_dict() for r in pop_short[:top_n]]},
        "dip": {"long": [r.to_dict() for r in dip_long[:top_n]],
                "short": [r.to_dict() for r in dip_short[:top_n]]},
        "mine": mine,
        "measured": sum(1 for r in rows.values() if r.pop_score is not None),
        "offered": len(normalised),
    }


def sort_mine(rows: Sequence[Mapping[str, Any]], mode: str, side: str) -> list[dict[str, Any]]:
    """My names sorted by the active mode's score; unmeasured rows last."""
    key = "dip_score" if mode == "dip" else "pop_score"
    sign = -1.0 if side == "long" else 1.0

    def order(row):
        value = row.get(key)
        return (value is None, sign * value if value is not None else 0.0, row.get("symbol", ""))

    return [dict(row) for row in sorted(rows, key=order)]


# ---------------------------------------------------------------- helpers
def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pct(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return (value / base - 1.0) * 100.0
