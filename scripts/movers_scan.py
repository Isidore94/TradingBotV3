"""The Movers board's pure model: pops, SPY pullback/bounce/rally state, and the
strong/weak lists for each turn (Dip-*, Bounce-*, Rip-*).

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
- Dip boxes (trader, 2026-09-29): Dip-strong measures every name against SPY
  from SPY's lowest low since its last major M5 dip began, Dip-weak from its
  highest high since the last major rip began (else the low / high of day so
  far). A major move is a run of SWING_HA_RUN same-colour Heikin-Ashi candles
  on SPY's completed M5 bars, so the anchors shift as new runs form. An anchor
  under SWING_MIN_AGE_MIN old keeps last tick's (`previous_anchors`), else the
  low / high of day if old enough, else the open. Entry (trader, 2026-09-29):
  Dip-strong needs last above session VWAP, the previous day's high and the
  daily 100 and 200 SMA; Dip-weak below the previous day's low, VWAP and the
  daily 50 SMA (50 only). Unknown VWAP, level or SMA keeps a name off. A name once on
  a box today stays on it while it still qualifies (`held_by_side`). These
  `swing` lists feed the boxes only; the P8 notices, outcome logs and M5 watch
  feed still read the pullback/bounce/rally `dip`/`rip` lists.
- Quality floor (trader, 2026-09-28: "we get a lot of riff raff"): a ranked
  row needs market cap >= $1B and a 20-session mean share volume >= 1M (the
  universe builder's numbers). Unknown stays; My names is never filtered.
- D1 trend gate (trader, 2026-09-28): a long pop/dip/rip row sits above the
  daily 100 and 200 SMA, a short row below the daily 50 and 100. Too little
  daily history is UNKNOWN: the row stays, tagged, never dropped. My names is
  the trader's own list: tagged, never filtered.

The RVOL baseline helpers (`build_rvol_baseline`, `recent_rvol`) are pure and
importable on their own so other tools can share them.
They are a named variant of `rvol.session_rvol`, not a copy: 20 sessions, mean of
per-bar ratios, keyed by time of day, so their numbers differ.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import strength_scan
from completed_bars import bar_time, completed_m5_bars
from indicators.heikin_ashi import GREEN, RED, compute_heikin_ashi

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
#: A rally may start from a low in the first 2 bars once this many completed
#: bars have passed since that low (30 minutes; lead 2026-09-25).
RALLY_OPEN_LOW_MIN_BARS = 6
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
#: D1 trend gate (trader, 2026-09-28: "a lot of these charts are shit"): a long
#: row must sit above these daily SMAs, a short row below them. Daily closes
#: are completed sessions only; too little history is unknown, never a fail.
TREND_SMA_LONG = (100, 200)
TREND_SMA_SHORT = (50, 100)
DIP_WEAK_SMA = 50  # Dip-weak box: below this daily SMA only (trader 2026-09-29)
#: Dip boxes: a completed run of at least this many same-colour Heikin-Ashi
#: candles on SPY's M5 is a major move (trader: "more than 5 in a row").
SWING_HA_RUN = 6
#: Dip boxes: an anchor bar must be this many minutes old (trader, 2026-09-29).
SWING_MIN_AGE_MIN = 30
#: Quality floor for every ranked list (trader, 2026-09-28).
MIN_MARKET_CAP_M = 1000.0
MIN_AVG_VOLUME_20D = 1_000_000.0
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
    rally: bool = False  # either day, SPY up off its last swing low
    extreme_time: str = ""  # HH:MM NY of the pullback/bounce/rally start bar
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

    Pullback: SPY's session high made while above the running session VWAP,
    and now >= PULLBACK_MIN_PCT off that high (it may be below VWAP or the
    open now). Bounce mirrors it off the session low; when both qualify, the
    later turn wins. Rally: SPY now >= PULLBACK_MIN_PCT above its last swing
    low (`last_swing_low`), on either day type (a low in the first 2 bars
    counts after RALLY_OPEN_LOW_MIN_BARS bars); it beats a pullback or bounce
    only when its low is the later turn (a bounce off the same low stays a
    bounce). Up/down day is SPY vs its session open. SPY bars older than
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

    high_index = max(range(len(today)), key=lambda i: (today[i]["high"], -i))
    high = today[high_index]["high"]
    off_high = _pct(last, high)
    at_high = vwaps[high_index]
    pull = (at_high is not None and today[high_index]["close"] > at_high
            and turn_on(high_index, off_high,
                        off_high is not None and off_high <= -PULLBACK_MIN_PCT))
    low_index = min(range(len(today)), key=lambda i: (today[i]["low"], i))
    low = today[low_index]["low"]
    off_low = _pct(last, low)
    at_low = vwaps[low_index]
    bounce = (at_low is not None and today[low_index]["close"] < at_low
              and turn_on(low_index, off_low,
                          off_low is not None and off_low >= PULLBACK_MIN_PCT))
    if pull and bounce:
        # Both turns qualify (chop): the later one is the live one.
        pull, bounce = high_index > low_index, low_index > high_index
    swing_index = last_swing_low(today)
    swing_low = today[swing_index]["low"]
    off_swing = _pct(last, swing_low)
    rally_beyond = off_swing is not None and off_swing >= PULLBACK_MIN_PCT
    rally = turn_on(swing_index, off_swing, rally_beyond) or (
        swing_index < PULLBACK_MIN_HIGH_INDEX and rally_beyond
        and last_index - swing_index >= RALLY_OPEN_LOW_MIN_BARS
    )
    if rally and ((pull and high_index >= swing_index) or (bounce and low_index >= swing_index)):
        rally = False  # the pullback/bounce turn is as late or later
    if rally:
        pull = bounce = False
    live = pull or bounce or rally
    if last > session_open and (live or last > vwap):
        day = "up_day"
    elif last < session_open and (live or last < vwap):
        day = "down_day"
    elif live:
        day = "flat"
    else:
        return MarketState("flat", **common)
    use_high = pull or (not bounce and not rally and day == "up_day")
    index, extreme, off = (high_index, high, off_high) if use_high else (low_index, low, off_low)
    if rally:
        index, extreme, off = swing_index, swing_low, off_swing
    return MarketState(
        day, pullback=pull, bounce=bounce, rally=rally,
        extreme_time=today[index]["dt"].strftime("%H:%M"),
        extreme_price=extreme, start_dt=today[index]["dt"] if live else None,
        spy_from_extreme_pct=off, **common,
    )


def last_swing_low(today: Sequence[Mapping[str, Any]]) -> int:
    """Index of SPY's last swing low: the session low, or a later low that nothing
    after it undercut and that a down leg of >= PULLBACK_MIN_PCT led into."""
    if not today:
        return 0
    best = min(range(len(today)), key=lambda i: (today[i]["low"], i))
    for j in range(best + 1, len(today)):
        low = today[j]["low"]
        if any(today[k]["low"] < low for k in range(j + 1, len(today))):
            continue
        if j - best < 2:
            continue
        peak = max(today[k]["high"] for k in range(best + 1, j))
        drop = _pct(low, peak)
        if drop is not None and drop <= -PULLBACK_MIN_PCT:
            best = j
    return best


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
    since_start_pct: float | None = None  # since the pullback/bounce/rally start bar
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
    prev_session: str = ""  # NY date (ISO) of the M5 bars prev_high/prev_low come from
    from_hod_atr: float | None = None  # <= 0: ATRs below the session high
    from_lod_atr: float | None = None  # >= 0: ATRs above the session low
    from_vwap_atr: float | None = None
    hod_break: bool = False  # last completed bar made a new session high
    lod_break: bool = False
    ext_up: bool = False  # more than EXT_ATR above VWAP
    ext_down: bool = False
    er: bool = False  # reports today or reported after the last close
    group: str = ""  # short industry label when 3+ share a list's top 15
    # Quality floor: None = cap or 20-day volume unknown.
    market_cap_m: float | None = None
    avg_volume_20d: float | None = None
    quality_ok: bool | None = None
    # D1 trend gate against TREND_SMA_*: None = not enough daily history.
    trend_long: bool | None = None
    trend_short: bool | None = None
    daily_bars: int = 0

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
        since, dip, found = excess_since(
            today, spy_today, state.start_dt, atr=atr, baseline=baseline
        )
        if not found:
            note = "no bar at the pullback start"
    if atr is None:
        note = note or "ATR unmeasurable"
    levels = _levels(prior, today, atr)
    return MoverRow(
        symbol=symbol, last=last, move15_pct=move15, move30_pct=move30, day_pct=day_pct,
        rvol=rvol, vs_spy15_pct=vs_spy, atr=atr, pop_score=pop_score,
        since_start_pct=since, dip_score=dip, session_volume=session_vol,
        passes_floors=passes, stale=stale, note=note, **levels,
    )


def swing_anchors(
    spy_bars: Sequence[Mapping[str, Any]],
    today_date: date | None = None,
    *,
    now: datetime | None = None,
    previous_anchors: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Where each Dip box measures from, off SPY's completed M5 bars (normalised).

    A major move is a run of SWING_HA_RUN+ same-colour Heikin-Ashi candles today
    (it counts while still running). Longs measure from the lowest low since the
    last major dip began; shorts from the highest high since the last major rip
    began. No major dip yet: longs use the low of day so far (`kind` "lod"); no
    major rip yet: shorts use the high of day ("hod"). Ties take the later bar.

    An anchor bar must be SWING_MIN_AGE_MIN old at `now` (default: the last bar's
    end). A younger one yields to `previous_anchors[side]` (last tick's, today
    only; `held_from_previous` True), else the low / high of day if old enough,
    else the first bar today (`kind` "open"). None only when SPY has no bars today."""
    empty: dict[str, dict[str, Any] | None] = {"long": None, "short": None}
    if not spy_bars:
        return empty
    bars = list(spy_bars)
    colors = compute_heikin_ashi(
        [b["open"] for b in bars], [b["high"] for b in bars],
        [b["low"] for b in bars], [b["close"] for b in bars],
    ).colors
    day = today_date or bars[-1]["dt"].date()
    first = next((i for i, b in enumerate(bars) if b["dt"].date() == day), None)
    if first is None:
        return empty
    moment = now or (bars[-1]["dt"] + timedelta(minutes=BAR_MINUTES))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=NY_TZ)
    min_age = timedelta(minutes=SWING_MIN_AGE_MIN)
    runs: list[tuple[str, int, int]] = []  # (colour, start, end) of major runs today
    start = first
    for index in range(first + 1, len(bars) + 1):
        if index == len(bars) or colors[index] != colors[start]:
            if colors[start] in (GREEN, RED) and index - start >= SWING_HA_RUN:
                runs.append((colors[start], start, index - 1))
            start = index

    def extreme(side: str, window: range) -> int:
        if side == "long":
            return min(window, key=lambda i: (bars[i]["low"], -i))
        return max(window, key=lambda i: (bars[i]["high"], i))

    def pack(at: int, kind: str, price: float | None = None) -> dict[str, Any]:
        stamp = bars[at]["dt"]
        if price is None:
            price = bars[at]["low"] if kind in ("swing_long", "lod") else bars[at]["high"]
        return {"dt": stamp.isoformat(timespec="seconds"), "time": stamp.strftime("%H:%M"),
                "price": price, "kind": "swing" if kind.startswith("swing") else kind,
                "held_from_previous": False, "_dt": stamp}

    def old_enough(at: int) -> bool:
        return moment - bars[at]["dt"] >= min_age

    def previous(side: str) -> dict[str, Any] | None:
        # Last tick's anchor for this side, only if it names a bar we hold today.
        prior = (previous_anchors or {}).get(side)
        if not isinstance(prior, Mapping):
            return None
        try:
            stamp = datetime.fromisoformat(str(prior.get("dt") or ""))
        except ValueError:
            return None
        if stamp.tzinfo is None:
            return None
        at = next((i for i in range(first, len(bars)) if bars[i]["dt"] == stamp), None)
        if at is None:
            return None
        kind = str(prior.get("kind") or "swing")
        price = _finite(prior.get("price"))
        if price is None and kind == "open":
            price = bars[at]["open"]
        out = pack(at, "swing_" + side if kind == "swing" else kind, price)
        out["held_from_previous"] = True
        return out

    def anchor(side: str) -> dict[str, Any]:
        move = RED if side == "long" else GREEN
        day_kind = "lod" if side == "long" else "hod"
        last = next((r for r in reversed(runs) if r[0] == move), None)
        if last is None:
            at, kind = extreme(side, range(first, len(bars))), day_kind
        else:
            at, kind = extreme(side, range(last[1], len(bars))), "swing_" + side
        if old_enough(at):
            return pack(at, kind)
        held = previous(side)
        if held is not None:
            return held
        at = extreme(side, range(first, len(bars)))
        if old_enough(at):
            return pack(at, day_kind)
        return pack(first, "open", bars[first]["open"])

    return {"long": anchor("long"), "short": anchor("short")}


def excess_since(
    today: Sequence[Mapping[str, Any]],
    spy_today: Sequence[Mapping[str, Any]],
    start_dt: datetime,
    *,
    atr: float | None,
    baseline: Mapping[int, float] | None,
) -> tuple[float | None, float | None, bool]:
    """(% since the start bar's close, excess vs SPY in ATRs x RVOL weight, found).
    `found` is False when the name or SPY has no bar at the start."""
    start_close = _close_at(today, start_dt)
    spy_start = _close_at(spy_today, start_dt)
    spy_last = spy_today[-1]["close"] if spy_today else None
    if start_close is None or spy_start is None or spy_last is None or not today:
        return None, None, False
    since = _pct(today[-1]["close"], start_close)
    spy_since = _pct(spy_last, spy_start)
    span = [bar for bar in today if bar["dt"] > start_dt]
    score = None
    if since is not None and spy_since is not None and atr and atr > 0 and span:
        excess_price = start_close * (since - spy_since) / 100.0
        span_rvol = recent_rvol(today, baseline, bars=len(span))
        score = (excess_price / atr) * rvol_weight(span_rvol)
    return since, score, True


def update_held(held: Mapping[str, Any], board: Mapping[str, Any], *, session: date) -> dict[str, Any]:
    """The names each Dip box has listed this session (order kept), for `held_by_side`."""
    if held.get("session") != session:
        held = {"session": session, "long": [], "short": []}
    out = {"session": session, "long": list(held.get("long") or []),
           "short": list(held.get("short") or [])}
    for side in ("long", "short"):
        for row in ((board.get("swing") or {}).get(side)) or []:
            symbol = str(row.get("symbol") or "").strip().upper()
            if symbol and symbol not in out[side]:
                out[side].append(symbol)
    return out


def quality_ok(market_cap_m: float | None, avg_volume_20d: float | None) -> bool | None:
    """The quality floor: False on a known miss, True when both pass, else None."""
    cap, volume = _finite(market_cap_m), _finite(avg_volume_20d)
    if (cap is not None and cap < MIN_MARKET_CAP_M) or (
        volume is not None and volume < MIN_AVG_VOLUME_20D
    ):
        return False
    return True if cap is not None and volume is not None else None


def trend_flags(
    last: float | None, daily_closes: Sequence[Any] | None
) -> tuple[bool | None, bool | None, int]:
    """(long ok, short ok, daily closes used). Long: `last` above every
    TREND_SMA_LONG daily SMA; short: below every TREND_SMA_SHORT one. A known
    miss is False even when a longer SMA is unmeasurable; otherwise an
    unmeasurable SMA makes the side None (unknown)."""
    closes = [c for c in (_finite(x) for x in daily_closes or ()) if c is not None]
    price = _finite(last)
    if price is None or not closes:
        return None, None, len(closes)

    def check(periods: Sequence[int], above: bool) -> bool | None:
        verdict: bool | None = True
        for period in periods:
            level = strength_scan.sma(closes, period)
            if level is None:
                verdict = None
            elif (price <= level) if above else (price >= level):
                return False
        return verdict

    return check(TREND_SMA_LONG, True), check(TREND_SMA_SHORT, False), len(closes)


def below_sma(last: float | None, daily_closes: Sequence[Any] | None, period: int) -> bool | None:
    """`last` below the daily `period` SMA; None when either is unmeasurable."""
    closes = [c for c in (_finite(x) for x in daily_closes or ()) if c is not None]
    price, level = _finite(last), strength_scan.sma(closes, period)
    return None if price is None or level is None else price < level


def dip_box_ok(row: MoverRow, side: str, below_weak_sma: bool | None) -> bool:
    """Dip-box entry: long above session VWAP, the previous day's high and the
    100/200 SMA; short below the previous day's low, VWAP and the 50 SMA only.
    Unknown VWAP, level or SMA never qualifies."""
    last, vwap = _finite(row.last), _finite(row.session_vwap)
    if last is None or vwap is None:
        return False
    if side == "long":
        high = _finite(row.prev_high)
        return high is not None and last > vwap and last > high and row.trend_long is True
    low = _finite(row.prev_low)
    return low is not None and last < vwap and last < low and below_weak_sma is True


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
        "prev_session": prev_session[0]["dt"].date().isoformat() if prev_session else "",
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
    for mode in ("pop", "dip", "rip", "swing"):
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
    for mode in ("pop", "dip", "rip", "swing"):
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
    daily_closes: Mapping[str, Sequence[Any]] | None = None,
    held_by_side: Mapping[str, Iterable[str]] | None = None,
    fundamentals: Mapping[str, Mapping[str, Any]] | None = None,
    previous_anchors: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """The whole board as plain dicts (safe to emit across threads).

    `focus_by_side` is {"long": [...], "short": [...]} of the trader's Focus
    names; `earnings` the names to tag ER; `daily_closes` completed daily
    closes per symbol for the D1 trend gate (a name without them is unknown).
    Lists: pop/dip/rip/mine, each {"long": rows, "short": rows}; dip is lit by
    a pullback or bounce, rip by a rally (long = Rip-strong, short = Rip-weak).
    The ranked lists drop a row on the wrong side of its D1 SMAs; mine never does.
    `swing` holds the Dip boxes (see `swing_anchors`); `held_by_side` keeps a name
    listed earlier today on its box while it still qualifies. `fundamentals` is
    symbol -> {"market_cap_m", "avg_volume_20d"} for the quality floor.
    `previous_anchors` is last tick's `swing_anchor`, kept while a new anchor is young.
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
        long_ok, short_ok, count = trend_flags(
            rows[symbol].last, (daily_closes or {}).get(symbol)
        )
        rows[symbol] = replace(
            rows[symbol], trend_long=long_ok, trend_short=short_ok, daily_bars=count
        )
        facts = (fundamentals or {}).get(symbol) or {}
        cap, volume = _finite(facts.get("market_cap_m")), _finite(facts.get("avg_volume_20d"))
        rows[symbol] = replace(
            rows[symbol], market_cap_m=cap, avg_volume_20d=volume,
            quality_ok=quality_ok(cap, volume),
        )

    def rankable(row: MoverRow) -> bool:
        # The quality floor: a known miss never ranks; unknown does.
        return (row.passes_floors and not row.stale and row.atr is not None
                and row.quality_ok is not False)

    # The D1 trend gate: a known miss leaves the ranked lists; unknown stays.
    pop_long = sorted(
        (r for r in rows.values() if rankable(r) and r.pop_score is not None
         and r.pop_score >= POP_MIN_ATR_MOVE and r.trend_long is not False),
        key=lambda r: (-r.pop_score, r.symbol),
    )
    pop_short = sorted(
        (r for r in rows.values() if rankable(r) and r.pop_score is not None
         and r.pop_score <= -POP_MIN_ATR_MOVE and r.trend_short is not False),
        key=lambda r: (r.pop_score, r.symbol),
    )
    # Dip lists, lit by a pullback or a bounce: long = beating SPY since the
    # turn (strong), short = lagging it (weak).
    turn_long = sorted(
        (r for r in rows.values() if rankable(r) and r.dip_score is not None
         and r.dip_score >= 0 and r.trend_long is not False),
        key=lambda r: (-r.dip_score, r.symbol),
    )
    turn_short = sorted(
        (r for r in rows.values() if rankable(r) and r.dip_score is not None
         and r.dip_score < 0 and r.trend_short is not False),
        key=lambda r: (r.dip_score, r.symbol),
    )
    dip_on = state.pullback or state.bounce
    dip_long, dip_short = (turn_long, turn_short) if dip_on else ([], [])
    # Rip lists, lit by a rally: same score since the rally start bar.
    rip_long, rip_short = (turn_long, turn_short) if state.rally else ([], [])

    # Dip boxes: each side against SPY from its own swing anchor.
    anchors = swing_anchors(spy, today_date, now=moment, previous_anchors=previous_anchors)
    spy_today = split_today(spy, today_date)[1]
    swing: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    for side, anchor in anchors.items():
        if anchor is None:
            continue
        keep = {str(s or "").strip().upper() for s in (held_by_side or {}).get(side, ()) or ()}
        scored = []
        for symbol, row in rows.items():
            if not rankable(row):
                continue
            # Entry gate; an unmeasured SMA/VWAP/level can't be claimed, so it drops.
            weak_sma = None if side == "long" else below_sma(
                row.last, (daily_closes or {}).get(symbol), DIP_WEAK_SMA)
            if not dip_box_ok(row, side, weak_sma):
                continue
            today_bars = split_today(normalised[symbol], today_date)[1]
            since, score, _found = excess_since(
                today_bars, spy_today, anchor["_dt"], atr=row.atr,
                baseline=baselines.get(symbol),
            )
            if score is None or (score < 0 if side == "long" else score >= 0):
                continue
            scored.append((symbol, row, since, score))
        scored.sort(key=lambda t: ((-t[3] if side == "long" else t[3]), t[0]))
        top = scored[:top_n]
        extra = [t for t in scored[top_n:] if t[0] in keep]
        for symbol, row, since, score in top + extra:
            data = row.to_dict()
            data.update(since_start_pct=since, dip_score=score, held=symbol not in
                        {t[0] for t in top})
            swing[side].append(data)

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
        "rip": {"long": [r.to_dict() for r in rip_long[:top_n]],
                "short": [r.to_dict() for r in rip_short[:top_n]]},
        "swing": swing,
        "swing_anchor": {side: ({k: v for k, v in anchor.items() if k != "_dt"}
                                if anchor else None) for side, anchor in anchors.items()},
        "mine": mine,
        "measured": sum(1 for r in rows.values() if r.pop_score is not None),
        "daily_measured": sum(1 for r in rows.values() if r.daily_bars),
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
