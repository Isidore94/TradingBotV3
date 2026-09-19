"""Pure, durable-data walk-away rows for the Day Review page (TJ-2B, TJ-11).

TJ-2B measured one thing: the best excursion after a decision, on that session's
five-minute tape. TJ-11 is the trader's second look at it (2026-09-19):

> *"if they said no to a bunch of stocks that went on to have great moves that
> day or the next day, then I want to know about it. … I want what I missed to
> be very apparent."*

Measured on the live stores that week: **~95% of the day's decisions are made on
D1 charts** (09-17: 146 D1 calls against 6 M5 ones), and TJ-2B was grading all
of them on five-minute bars that end at the close. So this module now carries
two rulers and never mixes them:

* an **M5 decision** is measured on the session's own tape, from the open of the
  first completed bar after the stamp (TJ-2B's rule, unchanged);
* a **D1 decision** is measured on DAILY bars from the CLOSE of the session it
  was made in, over 1, 3 and 5 exchange sessions, and reads ``pending <date>``
  until that horizon closes - never zero.

Everything it adds is REPORTED: `REAL_MISS_V1`'s verdict, the base rate of the
three populations of the same scan, one deterministic sentence per table. None
of it reaches a detector, score, alert, watchlist, Focus, the review queue or
`review_policy.json`, and no row is ever rewritten.

Four rules worth keeping in mind while reading:

* **Missing is `unmeasured`, never zero.** A missing ATR leaves the ATR columns
  empty and the percent columns alone; an open horizon is `pending <date>`.
* **The unmeasured are SHOWN, never assumed** (Q1). Every skill cell carries
  ``n`` (the population) and ``measured`` (the Wilson denominator) separately.
* **Size and name order the view, never a result** (gate #43). No R statistic is
  computed here or used to choose what is shown.
* **A stamp outside a session belongs to the NEXT session**
  (`market_calendar.decision_session`). Existing rows are never rewritten; this
  reader maps them forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence

import market_calendar
import real_miss
from evidence_stats import LATELY_SESSIONS, MIN_REPORTABLE_N
# Import only: `scripts/indicators/` is an ask-first tree and nothing here edits it.
from indicators.atr import DEFAULT_LENGTH as ATR_LENGTH, wilder_atr
from swing_headline import WILSON_Z, wilson_lower_bound


REJECTS = frozenset({"veto", "pass", "not_today", "dislike", "m5_click_away"})
LIKES = frozenset({"like", "swing_favorite"})

#: The D1 ruler's horizons, in exchange sessions. The last one is the window the
#: three moves are measured over, and the one `pending` counts down to.
HORIZONS: tuple[int, ...] = (1, 3, 5)

#: How many previous sessions the "Earlier calls, now" table reaches back over.
#: Exchange sessions, walked on the calendar - never five calendar days.
EARLIER_SESSION_COUNT = 5

#: A position held longer than this is not judged by one day's tape (TJ-11 item
#: 7). Measured 2026-09-19: 15 of 37 closed trades were held past five sessions.
LONG_HOLD_SESSIONS = 5

#: The three populations of the SAME scan the skill line compares. They
#: PARTITION the scan's distinct (symbol, side) pairs: every name is in exactly
#: one of them.
POPULATIONS: tuple[str, ...] = ("liked_or_claimed", "rejected", "untouched")

#: What each population is called in a sentence.
POPULATION_WORDS = {
    "liked_or_claimed": "likes",
    "rejected": "vetoes",
    "untouched": "untouched",
}

#: The five tables, in the order the page reads them.
TABLES: tuple[str, ...] = (
    "liked_not_traded",
    "rejected",
    "traded_left_early",
    "claimed_d1",
    "earlier_calls",
)


@dataclass(frozen=True)
class WalkawayRow:
    decision_id: tuple[str, str, str, str, str, str, str]
    time: datetime | None
    symbol: str
    side: str
    category: str
    what_you_did: str
    ran_after_pct: float | None = None
    held_at_close_pct: float | None = None
    traded: str = "no"
    you_made: float | None = None
    left_on_table_pct: float | None = None
    state: str = "unmeasured"
    # -- TJ-11 --------------------------------------------------------------
    #: Worst adverse excursion BEFORE the best favourable one, signed negative.
    against_first_pct: float | None = None
    #: Side-adjusted move to the measured close.
    at_close_pct: float | None = None
    ran_after_atr: float | None = None
    against_first_atr: float | None = None
    at_close_atr: float | None = None
    #: D1 only: ``((1, pct|None), (3, ...), (5, ...))``.
    horizon_moves: tuple[tuple[int, float | None], ...] = ()
    #: `real_miss.verdict`'s answer for this row. "" means nobody asked.
    real_miss: str = ""
    #: The veto vocabulary code (or the pass codes) as the store recorded them.
    reason: str = ""
    instrument: str = ""
    #: Closed trades only: EXCHANGE sessions from open to close.
    sessions_held: int | None = None
    #: Option rows: what the legs say, or `unmeasured` when they say nothing.
    assignment: str = ""
    #: Why a money figure was withheld. Never a wrong number instead.
    not_judged_reason: str = ""


@dataclass(frozen=True)
class WalkawayDay:
    liked_not_traded: tuple[WalkawayRow, ...] = ()
    rejected: tuple[WalkawayRow, ...] = ()
    traded_left_early: tuple[WalkawayRow, ...] = ()
    claimed_d1: tuple[WalkawayRow, ...] = ()
    #: TJ-11: the D1 calls of the previous five sessions, most-ran first.
    earlier_calls: tuple[WalkawayRow, ...] = ()
    #: ``{"session": WINDOW, "lately": WINDOW}``; see :func:`_skill_window`.
    skill: dict[str, Any] | None = None
    #: One deterministic line per table, keyed by population name.
    sentences: dict[str, str] = field(default_factory=dict)
    #: ``{"n": int, "net": float|None, "line": str}`` - every money line carries
    #: its `n` and says "too few to call" under `MIN_REPORTABLE_N`.
    money: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(("liked_not_traded", "rejected", "traded_left_early", "claimed_d1"))

    def __contains__(self, key: object) -> bool:
        return key in tuple(self)


# ---------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------


def _moment(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _day_of(bar: Mapping[str, Any]) -> date | None:
    stamp = _moment(bar.get("dt") if isinstance(bar, Mapping) else None)
    return stamp.date() if stamp is not None else None


#: One entry per distinct date TEXT, not per row: the horizon-outcomes store
#: holds hundreds of thousands of rows carrying a few dozen dates between them,
#: and the calendar answer for a date never changes.
_session_text_cache: dict[str, str] = {}


def _session_text(value: object) -> str:
    """The exchange session a stamp belongs to, as text, or ``""``.

    One seam for the whole module: a Saturday `session_date` and a Friday
    21:04 Pacific `created_at` both answer Monday.
    """
    # Plain dates only: a full timestamp is unique per decision and caching one
    # would grow a dict for the life of the desk to answer it once.
    if isinstance(value, str) and len(value) <= 10:
        cached = _session_text_cache.get(value)
        if cached is not None:
            return cached
        answer = market_calendar.decision_session(value)
        text = answer.isoformat() if answer is not None else ""
        _session_text_cache[value] = text
        return text
    answer = market_calendar.decision_session(value)
    return answer.isoformat() if answer is not None else ""


def earlier_sessions(session: str, *, count: int = EARLIER_SESSION_COUNT) -> tuple[str, ...]:
    """The ``count`` exchange sessions strictly BEFORE ``session``, oldest first.

    Walked on the calendar, so Labor Day is skipped rather than counted and a
    Monday reaches back into the week before instead of over a weekend.
    """
    try:
        cursor = date.fromisoformat(str(session)[:10])
    except ValueError:
        return ()
    out: list[str] = []
    try:
        for _ in range(max(0, int(count))):
            cursor = market_calendar.previous_session(cursor)
            out.append(cursor.isoformat())
    except market_calendar.SessionCalendarError:
        pass
    return tuple(reversed(out))


def _nth_session_after(day: date, count: int) -> date | None:
    cursor = day
    try:
        for _ in range(max(0, int(count))):
            cursor = market_calendar.next_session(cursor)
    except market_calendar.SessionCalendarError:
        return None
    return cursor


def _last_completed(now: datetime) -> date | None:
    try:
        return market_calendar.last_completed_session(now)
    except Exception:  # noqa: BLE001 - an unanswerable calendar measures nothing
        return None


# ---------------------------------------------------------------------------
# the two rulers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Moves:
    """Three measured moves and the price they were measured from."""

    ran: float | None = None
    against: float | None = None
    at_close: float | None = None
    reference: float | None = None


def _side_pct(price: float | None, reference: float, side: str) -> float | None:
    if price is None or not reference:
        return None
    if str(side).upper() == "SHORT":
        return (reference - price) / reference * 100
    return (price - reference) / reference * 100


def _atr_units(pct: float | None, reference: float | None, atr: float | None) -> float | None:
    """A percent move in ATR units. Missing ATR is `unmeasured`, never zero."""
    if pct is None or not reference or not atr or atr <= 0:
        return None
    one_atr_pct = atr / reference * 100
    if one_atr_pct <= 0:
        return None
    return pct / one_atr_pct


def _excursions(rows: Sequence[Mapping[str, Any]], reference: float, side: str) -> _Moves:
    """Ran-after, against-you-first and at-the-close, from one reference price.

    "Against you first" is the worst adverse extreme up to AND INCLUDING the bar
    that produced the best favourable one - what you were up against before the
    move happened, not the give-back afterwards. It is never positive: a name
    that never traded against you was never against you.
    """
    short = str(side).upper() == "SHORT"
    best_index, best_price = None, None
    for index, bar in enumerate(rows):
        price = _number(bar.get("low")) if short else _number(bar.get("high"))
        if price is None:
            continue
        if best_price is None or (price < best_price if short else price > best_price):
            best_index, best_price = index, price
    if best_price is None:
        return _Moves(reference=reference)
    adverse = None
    for bar in rows[: (best_index or 0) + 1]:
        price = _number(bar.get("high")) if short else _number(bar.get("low"))
        if price is None:
            continue
        if adverse is None or (price > adverse if short else price < adverse):
            adverse = price
    closes = [_number(bar.get("close")) for bar in rows]
    closes = [value for value in closes if value is not None]
    against = _side_pct(adverse, reference, side)
    return _Moves(
        ran=_side_pct(best_price, reference, side),
        against=min(0.0, against) if against is not None else None,
        at_close=_side_pct(closes[-1], reference, side) if closes else None,
        reference=reference,
    )


def _bars_for(
    bars: Mapping[str, Any], symbol: str, session: str, *, allow_direct: bool = True
) -> Sequence[Mapping[str, Any]]:
    direct = bars.get(symbol)
    if allow_direct and isinstance(direct, Sequence):
        return direct
    session_rows = bars.get(session)
    if isinstance(session_rows, Sequence):
        return session_rows
    if isinstance(session_rows, Mapping):
        rows = session_rows.get(symbol)
        return rows if isinstance(rows, Sequence) else ()
    # Tests and the durable reader both use a session map; tolerate a single-symbol
    # tape too, which is the old TJ-2A handoff shape.
    return ()


def _after_stamp(rows: Sequence[Mapping[str, Any]], stamp: datetime | None) -> list[Mapping[str, Any]]:
    """The bars that START strictly after the stamp, in order."""
    eligible: list[Mapping[str, Any]] = []
    for bar in rows or ():
        moment = _moment(bar.get("dt"))
        if moment is None:
            continue
        if stamp is not None:
            here = moment
            if here.tzinfo is None and stamp.tzinfo is not None:
                here = here.replace(tzinfo=stamp.tzinfo)
            elif here.tzinfo is not None and stamp.tzinfo is None:
                here = here.astimezone().replace(tzinfo=None)
            if here <= stamp:
                continue
        eligible.append(bar)
    return eligible


def _after_move(rows: Sequence[Mapping[str, Any]], stamp: datetime | None, side: str):
    """TJ-2A's one number, kept: best excursion after the stamp, in percent."""
    eligible = _after_stamp(rows, stamp)
    if not eligible:
        return None
    start = _number(eligible[0].get("open"))
    if not start:
        return None
    return _excursions(eligible, start, side).ran


def _completed_daily(rows: Sequence[Mapping[str, Any]], last_session: date | None) -> list[Mapping[str, Any]]:
    """Daily bars up to the last COMPLETED session, oldest first.

    `chart_snapshot.load_d1_bars` hands back today's forming bar during the
    session; a forming bar is preview, never a measurement (plan.md sec 5).
    """
    out: list[tuple[date, Mapping[str, Any]]] = []
    for bar in rows or ():
        if not isinstance(bar, Mapping):
            continue
        day = _day_of(bar)
        if day is None:
            continue
        if last_session is not None and day > last_session:
            continue
        out.append((day, bar))
    out.sort(key=lambda pair: pair[0])
    return [bar for _day, bar in out]


def _daily_atr(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Wilder ATR(14) on completed daily bars, or ``None`` when unmeasurable."""
    if len(rows) <= ATR_LENGTH:
        return None
    return wilder_atr(rows, ATR_LENGTH)


@dataclass(frozen=True)
class _D1Reading:
    moves: _Moves = _Moves()
    horizons: tuple[tuple[int, float | None], ...] = ()
    verdict: str = ""
    state: str = "unmeasured no_bars"
    complete: bool = False


def _d1_reading(
    daily: Sequence[Mapping[str, Any]],
    decision_day: date,
    side: str,
    *,
    atr: float | None,
    last_session: date | None,
    horizons: tuple[int, ...] = HORIZONS,
    window_end: date | None = None,
) -> _D1Reading:
    """The D1 ruler for one swing call: three moves, three horizons, one state.

    ``window_end`` overrides the measured window's last session - the
    Earlier-calls table measures to the SELECTED session's close rather than to
    a five-session horizon.
    """
    rows = _completed_daily(daily, last_session)
    before = [bar for bar in rows if (_day_of(bar) or decision_day) <= decision_day]
    reference = _number(before[-1].get("close")) if before else None
    horizon_end = window_end or _nth_session_after(decision_day, max(horizons or (0,)))
    if reference is None or horizon_end is None:
        return _D1Reading(state="unmeasured no_bars", verdict=real_miss.UNMEASURED_NO_BARS)
    after = [
        bar
        for bar in rows
        if (day := _day_of(bar)) is not None and decision_day < day <= horizon_end
    ]
    closes = {_day_of(bar): _number(bar.get("close")) for bar in after}
    horizon_moves: list[tuple[int, float | None]] = []
    for step in horizons:
        target = _nth_session_after(decision_day, step)
        measured = (
            target is not None
            and last_session is not None
            and target <= last_session
        )
        horizon_moves.append(
            (step, _side_pct(closes.get(target), reference, side) if measured else None)
        )
    complete = last_session is not None and horizon_end <= last_session
    verdict = real_miss.verdict(after, stamp=None, side=side, atr=atr, reference=reference)
    if verdict != real_miss.RUN and not complete:
        # A "no_run" over an unfinished window is not a finding; it is a clock
        # that has not run out. A run, once reached, cannot be taken back.
        verdict = "unmeasured:horizon_open"
    if not complete:
        return _D1Reading(
            moves=_Moves(reference=reference),
            horizons=tuple(horizon_moves),
            verdict=verdict,
            state=f"pending {horizon_end.isoformat()}",
            complete=False,
        )
    moves = _excursions(after, reference, side) if after else _Moves(reference=reference)
    return _D1Reading(
        moves=moves,
        horizons=tuple(horizon_moves),
        verdict=verdict,
        state="measured" if moves.ran is not None else "unmeasured no_bars",
        complete=True,
    )


# ---------------------------------------------------------------------------
# identity, preference and claims (TJ-2B, unchanged rules)
# ---------------------------------------------------------------------------


def _identity(session: str, row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        session,
        str(row.get("symbol") or "").upper(),
        str(row.get("side") or "").upper(),
        str(row.get("category") or "pick"),
        str(row.get("verdict") or ""),
        str(row.get("timeframe") or "M5").upper(),
        str(row.get("stamp") or row.get("created_at") or ""),
    )


def _preference_row(
    rows: Sequence[Mapping[str, Any]], session: str, symbol: str, side: str, source: str, verdict: str
) -> Mapping[str, Any] | None:
    channels = {
        ("annotations", "like"): "annotation:like_claim",
        ("annotations", "pass"): "annotation:pass",
        ("annotations", "veto"): "annotation:veto",
        ("pick_feedback", "like"): "pick_feedback:like",
        ("pick_feedback", "dislike"): "pick_feedback:dislike",
        ("pick_feedback", "not_today"): "pick_feedback:not_today",
        ("swing_favorites", "swing_favorite"): "swing_favorite",
        ("review_events", "m5_click_away"): "review_event:m5_click_away",
    }
    channel = channels.get((source, verdict), "")
    for row in rows:
        row_session = str(row.get("session_date") or "")[:10]
        row_channel = str(row.get("channel") or "")
        if (
            (row_session == session or not row_session)
            and str(row.get("symbol") or "").upper() == symbol
            and str(row.get("side") or row.get("direction") or "").upper() == side
            and (row_channel == channel or not row_channel)
        ):
            return row
    return None


def _claim_events(claims: Sequence[Mapping[str, Any]]):
    events: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in claims:
        key = (
            str(row.get("symbol") or "").upper(),
            str(row.get("side") or "").upper(),
            str(row.get("claimed_setup_id") or ""),
        )
        events.setdefault(key, []).append(row)
    return events


# ---------------------------------------------------------------------------
# instrument-aware trade rows (TJ-11 item 6)
# ---------------------------------------------------------------------------


def _assignment_of(trade: Mapping[str, Any]) -> str:
    """What the LEGS say about assignment, or ``unmeasured``.

    `journal_exposure` has no assignment field and this packet may not invent
    one: an assigned option reaches the journal as a real fill whose broker
    payload says so ("Buy 100 … (Assignment)"). Legs that say nothing leave the
    row `unmeasured` - never "not assigned", which would be a claim.
    """
    legs = trade.get("legs") or ()
    if not legs:
        return "unmeasured"
    for leg in legs:
        if not isinstance(leg, Mapping):
            continue
        haystack = " ".join(
            str(leg.get(key) or "")
            for key in ("role", "side", "type", "description", "raw_json")
        ).upper()
        if "ASSIGN" in haystack:
            return "assigned"
    return "not assigned"


def _trade_judgement(trade: Mapping[str, Any], exit_stamp: datetime | None) -> tuple[str, int | None, str, str]:
    """``(instrument, sessions_held, assignment, not_judged_reason)``.

    An option's premium and the underlying's tape are different rulers
    (`journal_exposure`'s rule: a long option is never a bullish setup), and a
    position held past five sessions was never a bet on today. Both keep their
    money figure and lose "left on the table".
    """
    instrument = str(trade.get("security_type") or "").strip().upper()
    opened = _moment(trade.get("opened_at"))
    sessions_held: int | None = None
    if opened is not None and exit_stamp is not None:
        try:
            sessions_held = market_calendar.trading_days_between(opened.date(), exit_stamp.date())
        except Exception:  # noqa: BLE001 - an unanswerable calendar counts nothing
            sessions_held = None
    assignment = _assignment_of(trade) if instrument == "OPT" else ""
    reason = ""
    if instrument == "OPT":
        reason = (
            "not judged here: an option's premium is not the underlying's move"
        )
    elif sessions_held is not None and sessions_held > LONG_HOLD_SESSIONS:
        reason = (
            f"not judged here: held {sessions_held} sessions, longer than "
            f"{LONG_HOLD_SESSIONS}, so today's tape is the wrong ruler"
        )
    return instrument, sessions_held, assignment, reason


# ---------------------------------------------------------------------------
# the skill line (TJ-11 item 4 / plan item 6)
# ---------------------------------------------------------------------------


def _wilson(runs: int, measured: int) -> tuple[float | None, float | None]:
    """The ONE Wilson interval (`swing_headline`'s z), both ends."""
    if measured <= 0:
        return None, None
    low = wilson_lower_bound(runs, measured)
    hits = max(0, min(int(runs), int(measured)))
    phat = hits / measured
    denominator = 1.0 + (WILSON_Z * WILSON_Z) / measured
    centre = phat + (WILSON_Z * WILSON_Z) / (2.0 * measured)
    margin = WILSON_Z * (
        (phat * (1.0 - phat) + (WILSON_Z * WILSON_Z) / (4.0 * measured)) / measured
    ) ** 0.5
    return low, min(1.0, (centre + margin) / denominator)


def _scan_names(
    scan_rows: Sequence[Mapping[str, Any]], sessions: set[str]
) -> dict[tuple[str, str, str], str]:
    """``{(session, symbol, side): setup_family}`` for the scan rows in scope.

    The POPULATION grain is the distinct name, never the row count: the live
    horizon-outcomes file holds four rows per name, one per horizon.
    """
    names: dict[tuple[str, str, str], str] = {}
    for row in scan_rows or ():
        session = _session_text(row.get("scan_date") or row.get("session_date"))
        if not session or session not in sessions:
            continue
        key = (
            session,
            str(row.get("symbol") or "").upper(),
            str(row.get("side") or "").upper(),
        )
        if not key[1]:
            continue
        family = str(row.get("setup_family") or "").strip()
        if key not in names or (family and not names[key]):
            names[key] = family
    return names


def _skill_cells(
    names: Mapping[tuple[str, str, str], str],
    decided: Mapping[tuple[str, str, str], str],
    verdicts: Mapping[tuple[str, str, str], str],
) -> tuple[dict[str, Any], ...]:
    """One cell per (population, side, family cut). Sizes only - no ranking."""
    members: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    for key, family in names.items():
        population = decided.get(key, "untouched")
        members.setdefault((population, key[2], ""), []).append(key)
        if family:
            members.setdefault((population, key[2], family), []).append(key)
    sides = sorted({key[2] for key in names})
    families = sorted({family for family in names.values() if family})
    cuts: list[tuple[str, str]] = [(side, "") for side in sides]
    for side in sides:
        for family in families:
            measured = 0
            for population in POPULATIONS:
                for key in members.get((population, side, family), ()):
                    if verdicts.get(key, "") in (real_miss.RUN, real_miss.NO_RUN):
                        measured += 1
            # A family cut is shown only where `n` allows it; the floor is the
            # same MIN_REPORTABLE_N every other trader-facing surface uses.
            if measured >= MIN_REPORTABLE_N:
                cuts.append((side, family))
    cells: list[dict[str, Any]] = []
    for side, family in cuts:
        for population in POPULATIONS:
            keys = members.get((population, side, family), [])
            answers = [verdicts.get(key, "") for key in keys]
            measured = [answer for answer in answers if answer in (real_miss.RUN, real_miss.NO_RUN)]
            runs = sum(1 for answer in measured if answer == real_miss.RUN)
            low, high = _wilson(runs, len(measured))
            cells.append(
                {
                    "population": population,
                    "side": side,
                    "setup_family": family,
                    "n": len(keys),
                    "measured": len(measured),
                    "unmeasured": len(keys) - len(measured),
                    "runs": runs,
                    "rate": (runs / len(measured)) if measured else None,
                    "low": low,
                    "high": high,
                    "reportable": len(measured) >= MIN_REPORTABLE_N,
                }
            )
    return tuple(cells)


def _overlaps(cells: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    """Population pairs whose intervals touch, within one side and family cut."""
    pairs: list[tuple[str, str]] = []
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        if cell["reportable"] and cell["low"] is not None:
            groups.setdefault((cell["side"], cell["setup_family"]), []).append(cell)
    for group in groups.values():
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                if first["low"] <= second["high"] and second["low"] <= first["high"]:
                    pairs.append((first["population"], second["population"]))
    return tuple(sorted(set(pairs)))


def _cell_words(cell: Mapping[str, Any]) -> str:
    word = POPULATION_WORDS.get(cell["population"], cell["population"])
    if not cell["reportable"] or cell["rate"] is None:
        return f"{word} too few to call (n {cell['n']}, measured {cell['measured']})"
    body = f"{word} {round(cell['rate'] * 100)}% (n {cell['n']}"
    if cell["measured"] != cell["n"]:
        body += f", measured {cell['measured']}"
    return body + ")"


def _skill_sentence(cells: Sequence[Mapping[str, Any]], overlapping: Sequence[tuple[str, str]], label: str) -> str:
    """Deterministic, and it never names a leader: these are base rates."""
    headline = [cell for cell in cells if cell["setup_family"] == ""]
    if not headline:
        return f"{label}: no scan rows to compare against."
    order = {name: index for index, name in enumerate(POPULATIONS)}
    headline.sort(key=lambda cell: (cell["side"], order.get(cell["population"], 9)))
    parts = ", ".join(_cell_words(cell) for cell in headline)
    text = f"{label}: {parts}."
    if overlapping:
        joined = "; ".join(
            f"{POPULATION_WORDS.get(a, a)} and {POPULATION_WORDS.get(b, b)}"
            for a, b in overlapping
        )
        text += f" Intervals overlap: {joined}."
    else:
        text += " No two intervals overlap."
    return text


def _skill_window(
    names: Mapping[tuple[str, str, str], str],
    decided: Mapping[tuple[str, str, str], str],
    verdicts: Mapping[tuple[str, str, str], str],
    *,
    window_sessions: int,
    label: str,
) -> dict[str, Any]:
    cells = _skill_cells(names, decided, verdicts)
    overlapping = _overlaps(cells)
    return {
        "window_sessions": window_sessions,
        "cells": cells,
        "overlapping": overlapping,
        "sentence": _skill_sentence(cells, overlapping, label),
    }


# ---------------------------------------------------------------------------
# the sentences (TJ-11 item 5)
# ---------------------------------------------------------------------------


def _real_miss_clause(rows: Sequence[WalkawayRow]) -> str:
    runs = sum(1 for row in rows if row.real_miss == real_miss.RUN)
    unmeasured = sum(1 for row in rows if not row.real_miss.startswith(("run", "no_run")))
    body = f"{runs} was a real miss" if runs == 1 else f"{runs} were real misses"
    if unmeasured:
        body += f" ({unmeasured} unmeasured)"
    return body


def _reason_clause(rows: Sequence[WalkawayRow]) -> str:
    """The one code the most rows share. Overlapping codes are never summed."""
    counts: dict[str, int] = {}
    for row in rows:
        for code in str(row.reason or "").split(","):
            code = code.strip()
            if code:
                counts[code] = counts.get(code, 0) + 1
    if not counts:
        return ""
    code, count = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0]
    if count < 2:
        return ""
    return f"; {count} share the reason {code}"


def _sentences(day: Mapping[str, Sequence[WalkawayRow]]) -> dict[str, str]:
    liked = day["liked_not_traded"]
    rejected = day["rejected"]
    traded = day["traded_left_early"]
    claimed = day["claimed_d1"]
    earlier = day["earlier_calls"]
    not_judged = sum(1 for row in traded if row.not_judged_reason)
    pending = sum(1 for row in claimed if row.state.startswith("pending"))
    return {
        "liked_not_traded": (
            f"You liked {len(liked)} you did not trade. {_real_miss_clause(liked)}."
        ),
        "rejected": (
            f"You vetoed {len(rejected)}. {_real_miss_clause(rejected)}"
            f"{_reason_clause(rejected)}."
        ),
        "traded_left_early": (
            f"You closed {len(traded)} position(s) on this day's tape; "
            f"{not_judged} not judged here."
        ),
        "claimed_d1": (
            f"You claimed {len(claimed)} D1 pick(s); {pending} still pending."
        ),
        "earlier_calls": (
            f"{len(earlier)} D1 call(s) from the previous {EARLIER_SESSION_COUNT} "
            f"sessions. {_real_miss_clause(earlier)}."
        ),
    }


def _money(rows: Sequence[WalkawayRow]) -> dict[str, Any]:
    """Every money line carries its `n` (TJ-11 item 8)."""
    amounts = [row.you_made for row in rows if row.you_made is not None]
    n = len(amounts)
    net = sum(amounts) if amounts else None
    body = f"{n} closed trade(s) counted, net {net:+.2f}" if net is not None else f"{n} closed trade(s) counted"
    if n < MIN_REPORTABLE_N:
        body += f" - too few to call (under {MIN_REPORTABLE_N})"
    return {"n": n, "net": net, "line": body + "."}


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------


def build(
    session: str,
    sources: Mapping[str, Any],
    bars: Mapping[str, Any],
    *,
    trades=(),
    claims=(),
    now: datetime | None = None,
    daily_bars: Mapping[str, Any] | None = None,
) -> WalkawayDay:
    """Build the five populations without touching a store, clock, or network.

    ``daily_bars`` is ``{symbol: [daily bar, ...]}`` from the durable daily store
    (`chart_snapshot.load_d1_bars`, read on the Day Review worker). A symbol that
    is absent is `unmeasured` - its ATR columns stay empty and its D1 ruler says
    so. Nothing here fetches anything.
    """
    session = str(session)[:10]
    moment = now or datetime.now()
    last_session = _last_completed(moment)
    daily_bars = daily_bars or {}
    try:
        session_day: date | None = date.fromisoformat(session)
    except ValueError:
        session_day = None

    atr_cache: dict[tuple[str, date | None], float | None] = {}
    completed_cache: dict[str, list[Mapping[str, Any]]] = {}

    def _daily(symbol: str) -> list[Mapping[str, Any]]:
        if symbol not in completed_cache:
            completed_cache[symbol] = _completed_daily(daily_bars.get(symbol) or (), last_session)
        return completed_cache[symbol]

    def _atr(symbol: str, as_of: date | None = None) -> float | None:
        """ATR(14) as it stood AT THE DECISION - never hindsight's ATR.

        plan.md sec 5: point-in-time evidence uses only what was available at the
        simulated decision time. An ATR that includes the move being measured
        would shrink the very excursion it is the yardstick for.
        """
        key = (symbol, as_of)
        if key not in atr_cache:
            rows = _daily(symbol)
            if as_of is not None:
                rows = [bar for bar in rows if (_day_of(bar) or as_of) <= as_of]
            atr_cache[key] = _daily_atr(rows)
        return atr_cache[key]

    decisions = [
        row for row in (sources.get("decisions") or ())
        if _session_text(row.get("session_date") or row.get("stamp")) == session
    ]
    unique: dict[tuple[str, str, str, str, str, str, str], Mapping[str, Any]] = {}
    for row in decisions:
        key = _identity(session, row)
        unique.setdefault(key, row)  # source duplicates are one decision; times are not.
    claim_events = _claim_events(tuple(claims))
    claimed_refs = {str(row.get("annotation_ref") or "") for row in claims if str(row.get("annotation_ref") or "")}
    preference = tuple(sources.get("preference") or ())
    liked: list[WalkawayRow] = []
    rejected: list[WalkawayRow] = []
    early: list[WalkawayRow] = []
    early_trade_ids: set[str] = set()
    decided: dict[tuple[str, str, str], str] = {}

    def _measure(symbol: str, side: str, stamp: datetime | None, timeframe: str, decision_day: date | None):
        """The right ruler for one decision: daily for D1, the tape for M5."""
        atr = _atr(symbol, decision_day)
        if timeframe == "D1" and decision_day is not None:
            reading = _d1_reading(
                _daily(symbol), decision_day, side, atr=atr, last_session=last_session
            )
            return reading.moves, reading.horizons, reading.verdict, reading.state, atr
        eligible = _after_stamp(_bars_for(bars, symbol, session), stamp)
        start = _number(eligible[0].get("open")) if eligible else None
        if not start:
            return _Moves(), (), real_miss.UNMEASURED_NO_BARS, "unmeasured no_bars", atr
        moves = _excursions(eligible, start, side)
        verdict = real_miss.verdict(eligible, stamp=None, side=side, atr=atr, reference=start)
        return moves, (), verdict, ("measured" if moves.ran is not None else "unmeasured no_bars"), atr

    def _row(ident, row, moves, horizons, verdict, state, atr, **extra) -> WalkawayRow:
        return WalkawayRow(
            ident,
            _moment(ident[-1]),
            ident[1],
            ident[2],
            ident[3],
            extra.pop("what_you_did", ident[4].replace("_", " ")),
            ran_after_pct=moves.ran,
            against_first_pct=moves.against,
            at_close_pct=moves.at_close,
            ran_after_atr=_atr_units(moves.ran, moves.reference, atr),
            against_first_atr=_atr_units(moves.against, moves.reference, atr),
            at_close_atr=_atr_units(moves.at_close, moves.reference, atr),
            horizon_moves=horizons,
            real_miss=verdict,
            reason=str((row or {}).get("reason") or ""),
            state=state,
            **extra,
        )

    for ident, row in unique.items():
        symbol, side, verdict_name = ident[1], ident[2], ident[4]
        timeframe = ident[5]
        stamp = _moment(ident[-1])
        moves, horizons, verdict, state, atr = _measure(symbol, side, stamp, timeframe, session_day)
        capture = str(row.get("capture_id") or row.get("event_id") or "")
        pref_row = _preference_row(preference, session, symbol, side, str(row.get("source") or ""), verdict_name)
        pref = str((pref_row or {}).get("match_state") or "")
        wanted_trade_id = str((pref_row or {}).get("trade_id") or "")
        # Match only the exact durable trade identity. A blank id is unknown,
        # not permission to choose another same-symbol position.
        matches = [
            trade
            for trade in trades
            if wanted_trade_id
            and str(trade.get("trade_id") or "") == wanted_trade_id
            and (_moment(trade.get("opened_at")) or datetime.min) > (stamp or datetime.min)
        ]
        claimed = capture and capture in claimed_refs
        if verdict_name in REJECTS:
            decided.setdefault((session, symbol, side), "rejected")
            rejected.append(_row(ident, row, moves, horizons, verdict, state, atr))
        elif verdict_name in LIKES and not claimed:
            decided[(session, symbol, side)] = "liked_or_claimed"
            if matches and pref == "matched":
                trade = matches[0]
                exit_stamp = _moment(trade.get("last_closing_leg_at") or trade.get("closed_at"))
                entered = str(trade.get("opened_at") or "")[:10][5:]
                instrument, held, assignment, not_judged = _trade_judgement(trade, exit_stamp)
                if str(trade.get("status") or "").lower() != "closed":
                    early.append(
                        _row(
                            ident, row, _Moves(), (), verdict, "pending trade open", atr,
                            what_you_did=f"liked {session[5:]}, entered {entered}",
                            traded="yes",
                            you_made=_number(trade.get("net_pnl")),
                            instrument=instrument,
                            sessions_held=held,
                            assignment=assignment,
                            not_judged_reason=not_judged,
                        )
                    )
                else:
                    exit_day = str((trade.get("last_closing_leg_at") or trade.get("closed_at") or ""))[:10]
                    left = (
                        None
                        if not_judged
                        else _after_move(_bars_for(bars, symbol, exit_day, allow_direct=False), exit_stamp, side)
                    )
                    if not_judged:
                        exit_state = "measured elsewhere"
                    else:
                        exit_state = "measured" if left is not None else f"unmeasured no_bars (exit {exit_day})"
                    early.append(
                        _row(
                            ident, row, _Moves(), (), verdict, exit_state, atr,
                            what_you_did=f"liked {session[5:]}, entered {entered}",
                            traded="yes",
                            you_made=_number(trade.get("net_pnl")),
                            left_on_table_pct=left,
                            instrument=instrument,
                            sessions_held=held,
                            assignment=assignment,
                            not_judged_reason=not_judged,
                        )
                    )
                    early_trade_ids.add(str(trade.get("trade_id") or ""))
            else:
                liked.append(
                    _row(
                        ident, row, moves, horizons, verdict, state, atr,
                        traded=("window" if pref == "window_open" else "no"),
                    )
                )

    # Every position closed on the selected day belongs in C, even if no
    # earlier like was linked to it. A linked trade remains one row.
    for trade in trades:
        exit_stamp = _moment(trade.get("last_closing_leg_at") or trade.get("closed_at"))
        if str(trade.get("status") or "").lower() != "closed" or not exit_stamp or exit_stamp.date().isoformat() != session:
            continue
        trade_id = str(trade.get("trade_id") or "")
        if trade_id in early_trade_ids:
            continue
        symbol = str(trade.get("symbol") or "").upper()
        side = str(trade.get("direction") or trade.get("side") or "").upper()
        instrument, held, assignment, not_judged = _trade_judgement(trade, exit_stamp)
        left = (
            None
            if not_judged
            else _after_move(_bars_for(bars, symbol, session, allow_direct=False), exit_stamp, side)
        )
        if not_judged:
            state = "measured elsewhere"
        else:
            state = "measured" if left is not None else f"unmeasured no_bars (exit {session})"
        early.append(
            WalkawayRow(
                (session, symbol, side, "trade", "trade_close", "M5", exit_stamp.isoformat()),
                exit_stamp,
                symbol,
                side,
                "trade",
                "closed trade",
                traded="yes",
                you_made=_number(trade.get("net_pnl")),
                left_on_table_pct=left,
                state=state,
                instrument=instrument,
                sessions_held=held,
                assignment=assignment,
                not_judged_reason=not_judged,
            )
        )

    claimed_rows: list[WalkawayRow] = []
    for claim in claims:
        if str(claim.get("action") or "").lower() != "claim" or _session_text(claim.get("session_date")) != session:
            continue
        symbol, side, setup = (
            str(claim.get("symbol") or "").upper(),
            str(claim.get("side") or "").upper(),
            str(claim.get("claimed_setup_id") or ""),
        )
        decided[(session, symbol, side)] = "liked_or_claimed"
        horizon = str(claim.get("horizon") or "").lower()
        events = claim_events.get((symbol, side, setup), ())
        start = list(events).index(claim)
        drop = next((row for row in events[start + 1 :] if str(row.get("action") or "").lower() in {"drop", "expire"}), None)
        decision = next(
            (ident for ident, row in unique.items() if str(row.get("capture_id") or "") == str(claim.get("annotation_ref") or "")),
            (session, symbol, side, setup or "claim", "claim", "D1", ""),
        )
        horizon_sessions = {"d1": 5, "m5": 1}.get(horizon)
        outcome = next(
            (
                row
                for row in sources.get("outcomes") or ()
                if str(row.get("scan_date") or row.get("session_date") or "")[:10] == session
                and str(row.get("symbol") or "").upper() == symbol
                and str(row.get("side") or "").upper() == side
                and (horizon_sessions is None or int(row.get("horizon_sessions") or horizon_sessions) == horizon_sessions)
            ),
            None,
        )
        held = _number((outcome or {}).get("eod_move_pct") if isinstance(outcome, Mapping) else None)
        if not horizon:
            state = "unmeasured no_horizon"
        elif outcome and (horizon != "d1" or bool(outcome.get("measured"))):
            state = "measured"
        else:
            # The live horizon file has no `maturity_date` column, so the date a
            # claim is pending UNTIL is computed from the exchange calendar.
            maturity = (
                _nth_session_after(session_day, horizon_sessions)
                if session_day is not None and horizon_sessions
                else None
            )
            state = f"pending {maturity.isoformat()}" if maturity else "unmeasured no_outcome"
        if drop:
            action = str(drop.get("action") or "dropped").lower()
            action = "dropped" if action == "drop" else action
            state = f"{action} {str(drop.get('session_date') or '')[:10]}; {state}"
        atr = _atr(symbol, session_day)
        claimed_rows.append(
            WalkawayRow(
                decision,
                _moment(decision[-1]),
                symbol,
                side,
                setup or "claim",
                "claimed D1 pick",
                held_at_close_pct=held,
                state=state,
                instrument="STK",
                real_miss=(
                    _d1_reading(_daily(symbol), session_day, side, atr=atr, last_session=last_session).verdict
                    if session_day is not None
                    else ""
                ),
            )
        )

    # -- the fifth population: earlier calls, now ---------------------------
    window = set(earlier_sessions(session))
    earlier_rows: list[WalkawayRow] = []
    for row in sources.get("earlier_decisions") or ():
        day_text = _session_text(row.get("session_date") or row.get("stamp"))
        if day_text not in window:
            continue
        if str(row.get("timeframe") or "M5").upper() != "D1":
            continue
        verdict_name = str(row.get("verdict") or "")
        if verdict_name not in REJECTS and verdict_name not in LIKES and verdict_name != "claim":
            continue
        ident = _identity(day_text, row)
        symbol, side = ident[1], ident[2]
        decision_day = date.fromisoformat(day_text)
        atr = _atr(symbol, decision_day)
        reading = _d1_reading(
            _daily(symbol),
            decision_day,
            side,
            atr=atr,
            last_session=last_session,
            window_end=session_day,
        )
        at_close = None
        if session_day is not None:
            closes = {
                _day_of(bar): _number(bar.get("close")) for bar in _daily(symbol)
            }
            reference = reading.moves.reference
            if reference:
                at_close = _side_pct(closes.get(session_day), reference, side)
        moves = _Moves(
            ran=reading.moves.ran,
            against=reading.moves.against,
            at_close=at_close,
            reference=reading.moves.reference,
        )
        earlier_rows.append(
            _row(
                ident,
                row,
                moves,
                reading.horizons,
                reading.verdict,
                reading.state,
                atr,
                what_you_did=f"{verdict_name.replace('_', ' ')} {day_text[5:]}",
            )
        )

    sort = lambda row: (row.ran_after_pct is None, -(row.ran_after_pct or 0), row.symbol)
    populations = {
        "liked_not_traded": tuple(sorted(liked, key=sort)),
        "rejected": tuple(sorted(rejected, key=sort)),
        "traded_left_early": tuple(early),
        "claimed_d1": tuple(claimed_rows),
        "earlier_calls": tuple(sorted(earlier_rows, key=sort)),
    }

    # -- the skill line ----------------------------------------------------
    scan_rows = tuple(sources.get("scan_rows") or ())
    lately = set(earlier_sessions(session, count=max(0, LATELY_SESSIONS - 1))) | {session}
    names = _scan_names(scan_rows, lately)
    for row in sources.get("earlier_decisions") or ():
        day_text = _session_text(row.get("session_date") or row.get("stamp"))
        verdict_name = str(row.get("verdict") or "")
        key = (day_text, str(row.get("symbol") or "").upper(), str(row.get("side") or "").upper())
        if verdict_name in LIKES or verdict_name == "claim":
            decided[key] = "liked_or_claimed"
        elif verdict_name in REJECTS:
            decided.setdefault(key, "rejected")
    verdicts: dict[tuple[str, str, str], str] = {}
    for key in names:
        day_text, symbol, side = key
        try:
            scan_day = date.fromisoformat(day_text)
        except ValueError:
            continue
        verdicts[key] = _d1_reading(
            _daily(symbol), scan_day, side, atr=_atr(symbol, scan_day), last_session=last_session
        ).verdict
    session_names = {key: family for key, family in names.items() if key[0] == session}
    skill = {
        "session": _skill_window(
            session_names, decided, verdicts,
            window_sessions=1, label="Real runs this session",
        ),
        "lately": _skill_window(
            names, decided, verdicts,
            window_sessions=LATELY_SESSIONS,
            label=f"Real runs over {LATELY_SESSIONS} sessions",
        ),
    }

    return WalkawayDay(
        populations["liked_not_traded"],
        populations["rejected"],
        populations["traded_left_early"],
        populations["claimed_d1"],
        earlier_calls=populations["earlier_calls"],
        skill=skill,
        sentences=_sentences(populations),
        money=_money(populations["traded_left_early"]),
    )


__all__ = [
    "EARLIER_SESSION_COUNT",
    "HORIZONS",
    "LONG_HOLD_SESSIONS",
    "POPULATIONS",
    "TABLES",
    "WalkawayDay",
    "WalkawayRow",
    "build",
    "earlier_sessions",
]
