"""The Movers board's M30 and Daily tabs: the same three boxes as the M5 board
(Movers, Dip-strong, Dip-weak), measured on 30-minute or daily bars.

Trader, 2026-09-29: "It does the EXACT same thing as movers tab does on an M5
basis but on an M30 and a D1 basis. The D1 scan only occurs after the close.
the M30 scan happens once per day at around 0900." Display and evidence only:
no alert, no watchlist or Focus write, no champion input. No Qt, no network,
no project I/O; the service hands bars in and gets plain dicts back.

Rules kept here:
- Completed bars only: an M30 bar counts once its 30 minutes are over, a daily
  bar once its session has closed (16:00 New York); a forming bar never ranks.
- Naive intraday stamps are market-local wall time (`local_tz`) and become New
  York time (`movers_scan.market_time`); a daily bar is keyed by its session
  date. Missing data is unknown: unknown RVOL is neutral, unknown ATR drops the
  row, an unknown level or SMA keeps a name off the Dip boxes.
- The M5 helpers are reused unchanged (`movers_scan`): RVOL weight, trend gate,
  quality floor, Dip-box entry gate, excess vs SPY, group tags.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any, Iterable, Mapping, Sequence

import movers_scan
import strength_scan
from completed_bars import bar_time
from indicators.heikin_ashi import GREEN, RED, compute_heikin_ashi

NY_TZ = movers_scan.NY_TZ
TF_M30 = "m30"
TF_D1 = "d1"
TIMEFRAMES = (TF_M30, TF_D1)
TF_LABELS = {TF_M30: "M30", TF_D1: "Daily"}
M30_MINUTES = 30
SESSION_CLOSE = movers_scan.SESSION_CLOSE

# ---------------------------------------------------------------- tunables
#: When each board is measured, New York time. The M30 board always reads the
#: bars completed by 12:00 New York (09:00 on the trader's Pacific clock), so a
#: late catch-up run shows the same board (lead decision 2026-09-29, trader can overrule).
M30_SCAN_TIME = time(12, 0)
#: The Daily board runs 15 minutes after the close (lead decision 2026-09-29, trader can overrule).
D1_SCAN_TIME = time(16, 15)
#: The pop move and the longer read, in bars of the timeframe (same as M5).
POP_BARS = movers_scan.POP_BARS
POP_LONG_BARS = movers_scan.POP_LONG_BARS
ATR_PERIOD = movers_scan.MOVERS_ATR_PERIOD
POP_MIN_ATR_MOVE = movers_scan.POP_MIN_ATR_MOVE
#: RVOL: M30 = the last 3 bars vs the same time-of-day slots over the prior 20
#: sessions; D1 = the last 3 days' mean volume vs the 20 days before them
#: (lead decision 2026-09-29, trader can overrule).
RVOL_SESSIONS = movers_scan.RVOL_BASELINE_SESSIONS
RVOL_MIN_SESSIONS = movers_scan.RVOL_BASELINE_MIN_SESSIONS
#: Dip-box anchor search: the last SWING_HA_RUN same-colour Heikin-Ashi run on
#: SPY within this lookback; none -> the extreme of the fallback window. M30 looks
#: back 5 sessions (fallback: prior + current session); D1 60 sessions (fallback:
#: last 20) (lead decision 2026-09-29, trader can overrule).
M30_LOOKBACK_SESSIONS = 5
M30_FALLBACK_SESSIONS = 2
D1_LOOKBACK_BARS = 60
D1_FALLBACK_BARS = 20
#: An anchor bar needs at least this many completed bars after it, else the
#: fallback window's extreme among old-enough bars (lead decision 2026-09-29, trader can overrule).
ANCHOR_MIN_AGE_BARS = 2
#: A series whose last bar is more than one bar behind the expected last bar is stale.
STALE_BARS_ALLOWED = 1
#: Column headers for the 3-bar and 6-bar moves.
MOVE_LABELS = {TF_M30: ("90m", "3h"), TF_D1: ("3d", "6d")}
#: Rows kept per list (same as M5).
TOP_N = movers_scan.MOVERS_TOP_N


# ---------------------------------------------------------------- bars
def session_date(stamp: Any) -> date | None:
    """A daily bar's session date: aware -> its New York date, naive -> its own date."""
    if isinstance(stamp, datetime):
        return stamp.astimezone(NY_TZ).date() if stamp.tzinfo is not None else stamp.date()
    if isinstance(stamp, date):
        return stamp
    parsed = bar_time({"dt": stamp})
    return session_date(parsed) if parsed is not None else None


def _values(bar: Any) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("open", "high", "low", "close", "volume"):
        raw = bar.get(key) if isinstance(bar, Mapping) else getattr(bar, key, None)
        out[key] = movers_scan._finite(raw)
    return out


def _ny(moment: datetime, local_tz: tzinfo | None = None) -> datetime:
    value = moment if moment.tzinfo is not None else moment.replace(tzinfo=local_tz or NY_TZ)
    return value.astimezone(NY_TZ)


def daily_completed(day: date, now: datetime) -> bool:
    """A session's daily bar is complete once its 16:00 New York close has passed."""
    ny = _ny(now)
    return day < ny.date() or (day == ny.date() and ny.time() >= SESSION_CLOSE)


def normalize_tf_bars(
    tf: str, bars: Iterable[Any], *, now: datetime, local_tz: tzinfo | None = None
) -> list[dict[str, Any]]:
    """Completed bars as dicts with an aware New York `dt`, ascending, one per stamp.

    M30: regular-session bars whose 30 minutes are over at `now`. D1: one bar
    per session date (dt = that date's midnight New York) once it has closed."""
    moment = _ny(now, local_tz)
    by_stamp: dict[datetime, dict[str, Any]] = {}
    for bar in bars or ():
        raw = bar_time(bar)
        if raw is None:
            continue
        if tf == TF_D1:
            day = session_date(raw)
            if day is None or not daily_completed(day, moment):
                continue
            stamp = datetime(day.year, day.month, day.day, tzinfo=NY_TZ)
        else:
            stamp = movers_scan.market_time(raw, local_tz)
            if stamp is None or movers_scan.session_offset(stamp) is None:
                continue
            if stamp + timedelta(minutes=M30_MINUTES) > moment:
                continue  # still forming
        values = _values(bar)
        if None in (values["open"], values["high"], values["low"], values["close"]):
            continue
        values["dt"] = stamp
        by_stamp[stamp] = values
    return [by_stamp[key] for key in sorted(by_stamp)]


def _expected_last_m30(moment: datetime) -> datetime:
    """Start of the last M30 bar that should be complete at `moment` (NY)."""
    close = moment.replace(hour=SESSION_CLOSE.hour, minute=SESSION_CLOSE.minute,
                           second=0, microsecond=0)
    clock = min(moment, close)
    floor = clock.replace(minute=clock.minute - clock.minute % M30_MINUTES,
                          second=0, microsecond=0)
    return floor - timedelta(minutes=M30_MINUTES)


# ---------------------------------------------------------------- RVOL
def d1_rvol(bars: Sequence[Mapping[str, Any]]) -> float | None:
    """Mean volume of the last POP_BARS days over the mean of the RVOL_SESSIONS
    days before them. None when any recent volume or too few prior days are known."""
    if len(bars) < POP_BARS + 1:
        return None
    recent = [movers_scan._finite(b.get("volume")) for b in bars[-POP_BARS:]]
    prior = [movers_scan._finite(b.get("volume"))
             for b in bars[-(POP_BARS + RVOL_SESSIONS):-POP_BARS]]
    prior = [v for v in prior if v is not None and v >= 0]
    if any(v is None for v in recent) or len(prior) < RVOL_MIN_SESSIONS:
        return None
    base = sum(prior) / len(prior)
    if base <= 0:
        return None
    return (sum(recent) / len(recent)) / base


def avg_volume_20d(daily: Sequence[Mapping[str, Any]]) -> float | None:
    """Mean share volume of the last 20 completed sessions, when all 20 are known."""
    recent = [movers_scan._finite(b.get("volume")) for b in daily[-20:]]
    if len(recent) < 20 or any(v is None for v in recent):
        return None
    return sum(recent) / 20.0


# ---------------------------------------------------------------- rows
def _move(bars: Sequence[Mapping[str, Any]], count: int) -> float | None:
    if len(bars) < count:
        return None
    return movers_scan._pct(bars[-1]["close"], bars[-count]["open"])


def m30_baseline(bars: Sequence[Mapping[str, Any]], today: date,
                 local_tz: tzinfo | None = None) -> dict[int, float] | None:
    """Mean volume per time-of-day slot over the prior RVOL_SESSIONS sessions (M5 helper)."""
    return movers_scan.build_rvol_baseline(
        bars, before=today, local_tz=local_tz, sessions=RVOL_SESSIONS,
        min_sessions=RVOL_MIN_SESSIONS,
    )


def measure_m30(
    symbol: str,
    bars: Sequence[Mapping[str, Any]],
    *,
    spy: Sequence[Mapping[str, Any]],
    today: date,
    reference_end: datetime | None,
    baseline: Mapping[int, float] | None,
) -> movers_scan.MoverRow:
    """The M5 measurements on completed M30 bars; the 6-bar move may span the prior session."""
    row = movers_scan.measure_symbol(
        symbol, bars, baseline=baseline, spy_bars=spy,
        state=movers_scan.MarketState("unknown"), reference_end=reference_end,
        today_date=today,
    )
    if row.last is not None:
        row = replace(row, move30_pct=_move(bars, POP_LONG_BARS))
    return row


def measure_d1(
    symbol: str,
    bars: Sequence[Mapping[str, Any]],
    *,
    spy: Sequence[Mapping[str, Any]],
    reference_end: datetime | None,
) -> movers_scan.MoverRow:
    """Pop numbers on completed daily bars: 3- and 6-day moves, day %, RVOL, vs SPY."""
    if not bars:
        return movers_scan.MoverRow(symbol, note="no daily bars")
    last_bar = bars[-1]
    last = last_bar["close"]
    atr = strength_scan.atr(list(bars), period=ATR_PERIOD)
    move3 = _move(bars, POP_BARS)
    rvol = d1_rvol(bars)
    vs_spy = None
    if move3 is not None and spy:
        spy_move = movers_scan._window_move_pct(spy, bars[-POP_BARS]["dt"], last_bar["dt"])
        if spy_move is not None:
            vs_spy = move3 - spy_move
    pop_score = None
    if move3 is not None and atr and atr > 0:
        pop_score = ((last - bars[-POP_BARS]["open"]) / atr) * movers_scan.rvol_weight(rvol)
    volume = last_bar.get("volume")
    return movers_scan.MoverRow(
        symbol=symbol, last=last, move15_pct=move3, move30_pct=_move(bars, POP_LONG_BARS),
        day_pct=movers_scan._pct(last, bars[-2]["close"]) if len(bars) > 1 else None,
        rvol=rvol, vs_spy15_pct=vs_spy, atr=atr, pop_score=pop_score,
        session_volume=volume,
        passes_floors=last > movers_scan.MIN_PRICE and volume is not None,
        stale=bool(reference_end is not None and last_bar["dt"] < reference_end),
        note="" if atr is not None else "ATR unmeasurable",
        hod=last_bar["high"], lod=last_bar["low"],
    )


# ---------------------------------------------------------------- anchors
def tf_anchors(tf: str, spy: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any] | None]:
    """Where each Dip box measures from, off SPY's completed bars on this timeframe.

    A major dip / rip is a run of SWING_HA_RUN red / green Heikin-Ashi candles
    inside the lookback. M30 longs: the highest high from the start of the last
    rip before the last dip (else the lookback start) through that dip's end;
    M30 shorts: the mirror low. D1 longs: the lowest low since the last dip
    began; D1 shorts: the highest high since the last rip. No such run, or an extreme under
    ANCHOR_MIN_AGE_BARS bars old: the extreme of the fallback window among bars
    old enough (`kind` "window"). None when SPY has too few bars."""
    empty: dict[str, dict[str, Any] | None] = {"long": None, "short": None}
    bars = list(spy or ())
    if not bars:
        return empty
    last = len(bars) - 1
    colors = compute_heikin_ashi(
        [b["open"] for b in bars], [b["high"] for b in bars],
        [b["low"] for b in bars], [b["close"] for b in bars],
    ).colors
    if tf == TF_M30:
        sessions = sorted({b["dt"].date() for b in bars})
        look_day = sessions[-M30_LOOKBACK_SESSIONS:][0]
        fall_day = sessions[-M30_FALLBACK_SESSIONS:][0]
        look = next(i for i, b in enumerate(bars) if b["dt"].date() >= look_day)
        fall = next(i for i, b in enumerate(bars) if b["dt"].date() >= fall_day)
    else:
        look = max(0, len(bars) - D1_LOOKBACK_BARS)
        fall = max(0, len(bars) - D1_FALLBACK_BARS)
    runs: list[tuple[str, int, int]] = []  # (colour, start, end) of major runs in the lookback
    start = look
    for index in range(look + 1, len(bars) + 1):
        if index == len(bars) or colors[index] != colors[start]:
            if colors[start] in (GREEN, RED) and index - start >= movers_scan.SWING_HA_RUN:
                runs.append((colors[start], start, index - 1))
            start = index
    # M30 longs read the top SPY's last big dip fell from, shorts the bottom its last
    # big rip rose from (trader 2026-09-30); D1 keeps the low/high since the last run.
    top_for_long = tf == TF_M30

    def old_enough(at: int) -> bool:
        return last - at >= ANCHOR_MIN_AGE_BARS

    def high_side(side: str) -> bool:
        return (side == "long") == top_for_long

    def extreme(side: str, window: Sequence[int]) -> int:
        if high_side(side):
            return max(window, key=lambda i: (bars[i]["high"], i))
        return min(window, key=lambda i: (bars[i]["low"], -i))

    def pack(at: int, side: str, kind: str) -> dict[str, Any]:
        stamp = bars[at]["dt"]
        return {"dt": stamp.isoformat(timespec="seconds"), "date": stamp.date().isoformat(),
                "time": stamp.strftime("%H:%M") if tf == TF_M30 else "",
                "price": bars[at]["high"] if high_side(side) else bars[at]["low"],
                "kind": kind, "_dt": stamp}

    def span(side: str) -> range | None:
        move, before = (RED, GREEN) if side == "long" else (GREEN, RED)
        at = next((i for i in range(len(runs) - 1, -1, -1) if runs[i][0] == move), None)
        if at is None:
            return None
        if not top_for_long:
            return range(runs[at][1], len(bars))
        prior = next((r for r in reversed(runs[:at]) if r[0] == before), None)
        return range(prior[1] if prior else look, runs[at][2] + 1)

    def anchor(side: str) -> dict[str, Any] | None:
        window = span(side)
        if window is not None:
            at = extreme(side, window)
            if old_enough(at):
                return pack(at, side, "swing")
        window = [i for i in range(fall, len(bars)) if old_enough(i)]
        return pack(extreme(side, window), side, "window") if window else None

    return {"long": anchor("long"), "short": anchor("short")}


def anchored_vwap(bars: Sequence[Mapping[str, Any]], anchor_dt: datetime) -> float | None:
    """The name's VWAP anchored at `anchor_dt` through its last bar (None if no bar there)."""
    at = next((i for i, b in enumerate(bars) if b["dt"] == anchor_dt), None)
    if at is None:
        return None
    try:
        from chart_snapshot import anchored_vwap_band_series

        return anchored_vwap_band_series(list(bars), at)["avwap"][-1]
    except Exception:
        return None


def d1_dip_ok(last: float | None, avwap: float | None, side: str, trend_long: bool | None,
              below_weak_sma: bool | None) -> bool:
    """Daily Dip-box entry: long above the SPY-anchored VWAP and the 100/200 SMA;
    short below that VWAP and the 50 SMA. Unknown never qualifies
    (lead decision 2026-09-29, trader can overrule)."""
    price, level = movers_scan._finite(last), movers_scan._finite(avwap)
    if price is None or level is None:
        return False
    if side == "long":
        return price > level and trend_long is True
    return price < level and below_weak_sma is True


# ---------------------------------------------------------------- board
def build_timeframe_board(
    tf: str,
    bars_by_symbol: Mapping[str, Sequence[Any]],
    spy_bars: Sequence[Any] | None,
    *,
    now: datetime,
    daily_bars: Mapping[str, Sequence[Any]] | None = None,
    fundamentals: Mapping[str, Mapping[str, Any]] | None = None,
    industry: Mapping[str, str] | None = None,
    earnings: Iterable[str] | None = None,
    top_n: int = TOP_N,
    local_tz: tzinfo | None = None,
) -> dict[str, Any]:
    """The M30 or Daily board as plain dicts (safe to emit across threads).

    `bars_by_symbol`/`spy_bars` are that timeframe's bars; `daily_bars` the daily
    bars for the SMA trend gate and 20-day volume (D1 reads its own bars when
    omitted); `fundamentals` symbol -> {"market_cap_m", "avg_volume_20d"}.
    Lists: `pop` and `swing` (the Dip boxes), each {"long": rows, "short": rows}."""
    if tf not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {tf!r}")
    moment = _ny(now, local_tz)
    today = moment.date()
    er_names = {str(s or "").strip().upper() for s in earnings or ()}
    spy = normalize_tf_bars(tf, spy_bars or (), now=moment, local_tz=local_tz)
    normalised = {
        str(sym).strip().upper(): normalize_tf_bars(tf, bars, now=moment, local_tz=local_tz)
        for sym, bars in (bars_by_symbol or {}).items() if str(sym or "").strip()
    }
    normalised.pop("SPY", None)
    if daily_bars is None and tf == TF_D1:
        daily = normalised
    else:
        daily = {str(sym).strip().upper(): normalize_tf_bars(TF_D1, bars, now=moment)
                 for sym, bars in (daily_bars or {}).items() if str(sym or "").strip()}

    if tf == TF_M30:
        expected = _expected_last_m30(moment)
        reference = expected - timedelta(minutes=M30_MINUTES * STALE_BARS_ALLOWED)
    else:
        reference = spy[-1]["dt"] if spy else None

    baselines: dict[str, dict[int, float] | None] = {}
    rows: dict[str, movers_scan.MoverRow] = {}
    for symbol, bars in normalised.items():
        if tf == TF_M30:
            baselines[symbol] = m30_baseline(bars, today, local_tz)
            row = measure_m30(symbol, bars, spy=spy, today=today, reference_end=reference,
                              baseline=baselines[symbol])
        else:
            row = measure_d1(symbol, bars, spy=spy, reference_end=reference)
        closes = [b["close"] for b in daily.get(symbol) or ()]
        long_ok, short_ok, count = movers_scan.trend_flags(row.last, closes)
        facts = (fundamentals or {}).get(symbol) or {}
        cap = movers_scan._finite(facts.get("market_cap_m"))
        volume = movers_scan._finite(facts.get("avg_volume_20d"))
        if volume is None:
            volume = avg_volume_20d(daily.get(symbol) or ())
        rows[symbol] = replace(
            row, er=symbol in er_names, trend_long=long_ok, trend_short=short_ok,
            daily_bars=count, market_cap_m=cap, avg_volume_20d=volume,
            quality_ok=movers_scan.quality_ok(cap, volume),
        )

    def rankable(row: movers_scan.MoverRow) -> bool:
        return (row.passes_floors and not row.stale and row.atr is not None
                and row.quality_ok is not False)

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

    anchors = tf_anchors(tf, spy)
    swing: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    for side, anchor in anchors.items():
        if anchor is None:
            continue
        scored = []
        for symbol, row in rows.items():
            if not rankable(row):
                continue
            closes = [b["close"] for b in daily.get(symbol) or ()]
            weak_sma = None if side == "long" else movers_scan.below_sma(
                row.last, closes, movers_scan.DIP_WEAK_SMA)
            bars = normalised[symbol]
            avwap = None
            if tf == TF_M30:
                # Exactly the M5 entry gate, on M30 levels (lead decision 2026-09-29, trader can overrule).
                if not movers_scan.dip_box_ok(row, side, weak_sma):
                    continue
            else:
                avwap = anchored_vwap(bars, anchor["_dt"])
                if not d1_dip_ok(row.last, avwap, side, row.trend_long, weak_sma):
                    continue
            since, score, _found = movers_scan.excess_since(
                bars, spy, anchor["_dt"], atr=row.atr, baseline=baselines.get(symbol))
            if score is None or (score < 0 if side == "long" else score >= 0):
                continue
            scored.append((symbol, row, since, score, avwap))
        scored.sort(key=lambda t: ((-t[3] if side == "long" else t[3]), t[0]))
        for _symbol, row, since, score, avwap in scored[:top_n]:
            data = row.to_dict()
            data.update(since_start_pct=since, dip_score=score)
            if tf == TF_D1:
                data["avwap"] = avwap
            swing[side].append(data)

    spy_last = spy[-1] if spy else None
    spy_day = None
    if spy_last is not None:
        if tf == TF_M30:
            prior = [b for b in spy if b["dt"].date() < spy_last["dt"].date()]
            spy_day = movers_scan._pct(spy_last["close"], prior[-1]["close"]) if prior else None
        elif len(spy) > 1:
            spy_day = movers_scan._pct(spy_last["close"], spy[-2]["close"])
    if spy_last is None:
        state = {"state": "unknown", "reason": "no SPY bars"}
    else:
        kind = "flat" if not spy_day else ("up_day" if spy_day > 0 else "down_day")
        state = {"state": kind, "spy_day_pct": spy_day, "spy_last": spy_last["close"],
                 "reason": ""}
    board: dict[str, Any] = {
        "tf": tf,
        "scanned_at": now.isoformat(timespec="seconds"),
        "as_of": spy_last["dt"].isoformat(timespec="seconds") if spy_last else "",
        "session": spy_last["dt"].date().isoformat() if spy_last else "",
        "state": state,
        "pop": {"long": [r.to_dict() for r in pop_long[:top_n]],
                "short": [r.to_dict() for r in pop_short[:top_n]]},
        "swing": swing,
        "swing_anchor": {side: ({k: v for k, v in anchor.items() if k != "_dt"}
                                if anchor else None) for side, anchor in anchors.items()},
        "measured": sum(1 for r in rows.values() if r.pop_score is not None),
        "fresh": sum(1 for r in rows.values() if r.last is not None and not r.stale),
        "daily_measured": sum(1 for r in rows.values() if r.daily_bars),
        "offered": len(normalised),
    }
    movers_scan.apply_group_tags(board, industry or {})
    return board


def listed_symbols(board: Mapping[str, Any]) -> list[str]:
    """Every symbol on the board's lists, in list order, once."""
    seen: dict[str, None] = {}
    for key in ("pop", "swing"):
        for side in ("long", "short"):
            for row in ((board.get(key) or {}).get(side)) or []:
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol:
                    seen.setdefault(symbol, None)
    return list(seen)

