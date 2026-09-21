"""Were you right? — the read grader, the prediction ledger and the congruence
lines (TJ-10).

Decision 0021 answer 14: *"'Were you right' is a MEASURED row, never a model's
opinion."* Nothing in this module calls a model, opens a network, touches a
detector, a score, an alert, a watchlist, Focus, the review queue or
`review_policy.json`. It is arithmetic over bars that were handed to it.

Five rules hold it to evidence rather than interpretation.

1. **A CLICK is the read, and the words are never consulted for a row that has
   one.** `market_journal.prediction_of` is the ONE accessor for a click.
   Extraction (`market_thesis.extract_thesis`) only ever speaks for an entry
   with no click, and its rows are LABELLED `extracted`. The two are never
   pooled in one statistic (:class:`PoolingError`) - an inferred stance and a
   stated one are different evidence about different things (answer 29).
2. **A complete answer that cannot be graded is still recorded.** `No view` is
   an answer; an `unstated` note is a note. Both produce a row, and neither is
   ever right or wrong.
3. **Completed bars only, and nothing at or before the stamp.** The
   rest-of-day anchor is the OPEN of the first completed M5 bar that STARTS
   after the read, and the day ends at the exchange's own close for that date
   (`market_early_close.session_close`) - 13:00 Eastern on a scheduled half
   day, never the last row a provider happened to hand over. The five-session
   anchor is the DECISION session's own daily close
   (`market_calendar.decision_session`), which is the last close the trader
   could have known, for an after-close call as much as an in-session one.
4. **An open horizon is `pending <date>` and a missing bar is
   `unmeasured:<reason>`.** Never zero, never `flat` by default: missing data
   is uncertainty, never confirmation (plan.md sec 5).
5. **Append-only, and a matured grade is a NEW row naming the old one.** The
   first answer stays on disk exactly as it was written, because the
   interesting question later is what the desk knew WHEN.

The flat band is ONE constant with its reason written beside it
(:data:`FLAT_BAND_ATR`, :data:`FLAT_BAND_REASON`) and every grade row is
stamped with the rule's NAME (:data:`FLAT_BAND_RULE`), so changing the number
later supersedes cleanly instead of silently re-grading history.

PURE: :func:`grade_read` opens no store, starts no thread, sleeps for nothing
and has no clock of its own - `now` is an argument. The store functions at the
bottom are the only part of this file that touches a disk, and the Day Review
worker is the only thing that calls them on the desk.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

_log = logging.getLogger(__name__)

#: Schema NAMES (ground rule 5). They travel on the rows.
SCHEMA_READ = "market_read_v1"
SCHEMA_GRADE = "market_read_grade_v1"

#: Where a read came from. A click is what the trader STATED; an extraction is
#: what the desk INFERRED from their prose. Never pooled (answer 29).
SOURCE_CLICK = "click"
SOURCE_EXTRACTED = "extracted"
SOURCES = (SOURCE_CLICK, SOURCE_EXTRACTED)

#: The benchmark a read is about when the trader did not name another.
DEFAULT_BENCHMARK = "SPY"

VERDICT_RIGHT = "right"
VERDICT_WRONG = "wrong"
VERDICT_FLAT = "flat"
#: A congruence PICKS line (`picks_side_mix` / `m5_picks_side_mix`) whose
#: counted `n` sits under `evidence_stats.MIN_REPORTABLE_N` never reports
#: `agrees` / `disagrees` - live 2026-09-17 printed `disagrees` on a 2-of-2 M5
#: side mix, and a chip or a later pooling packet keys on `verdict`, not on the
#: text beside it (reviewer, 2026-09-20). The counts and the floor note in the
#: text are unchanged; only the verdict a reader could act on moves to this.
VERDICT_TOO_FEW = "too_few"
#: `pending <date>` while the horizon is open; `unmeasured:<reason>` when a bar
#: or the ATR is missing. Both are prefixes, so a reader tells them apart
#: without parsing English.
PENDING_PREFIX = "pending"
UNMEASURED_PREFIX = "unmeasured"
#: What a field the desk could not measure says. Never 0, never None, never a
#: guess - the same word the context block uses for an unmeasurable reading.
UNMEASURED = "unmeasured"

#: **The flat band, in ATR, and the lead's number - not the trader's.**
#: A quarter of the benchmark's point-in-time daily ATR(14), edge INCLUSIVE,
#: the same constant for both horizons in v1. It exists because a directional
#: call on a day that went nowhere is not a wrong call, and counting it wrong
#: would make a trader who reads chop correctly look like a coin flip. The
#: number is a starting point chosen to be small enough that a real trend day
#: always clears it and large enough that a drifting session does not; it is
#: the trader's to change, and never from one session's result.
FLAT_BAND_ATR = 0.25
#: The rule's NAME, stamped on every grade row so a later band supersedes
#: cleanly instead of silently re-grading rows measured under this one.
FLAT_BAND_RULE = "atr_0.25_v1"
FLAT_BAND_REASON = (
    "a move inside a quarter of the benchmark's point-in-time daily ATR(14) is "
    "the market doing nothing, not the trader being wrong; the edge is inside "
    "the band and the rule is named on every row so a later band supersedes "
    "instead of re-grading"
)

#: A five-session call is REPORTED at 1 and 3 sessions and JUDGED at 5.
D1_CHECKPOINT_SESSIONS = (1, 3, 5)

#: The naive rules a read is measured against (TJ-12 / TJ-16 read these).
BASELINES = ("always_up", "same_as_the_last_hour", "with_the_d1_environment")

#: The three congruence lines, in the order they are printed. This is the D1
#: spine and it is always exactly these three, in this order.
CONGRUENCE_KINDS = ("desk_d1_label", "picks_side_mix", "fills_bias")
#: The M5 half of lead decision 6, APPENDED when the session has an M5 read.
#: TJ-15 measured 1,026 M5 reviewed decisions against 1,013 D1 over 20 sessions,
#: so a rest-of-day read compared with D1 picks would be answering about the
#: wrong crowd. It is a separate kind rather than a second `picks_side_mix` row
#: because every reader of these lines keys on `kind`.
CONGRUENCE_M5_KIND = "m5_picks_side_mix"
_M5_TIMEFRAME = "M5"

#: What a read may be graded ON. `no_view`, `cautious` and `unstated` are
#: complete answers and are never in it.
GRADABLE_DIRECTIONS = ("up", "down", "chop", "range")
#: The two calls that say "no direction" - one per horizon's vocabulary.
FLAT_DIRECTIONS = ("chop", "range")

#: `market_thesis` stance -> the direction it means. `cautious` and `unstated`
#: are deliberately absent: neither is a call (lead decision, 2026-09-19).
_STANCE_DIRECTION = {
    "bullish": "up",
    "bearish": "down",
}

#: The desk's own D1 environment labels that carry a direction. `compressed`,
#: `mixed` and `unknown` say the tape has none, and a label with no direction
#: is never agreement (nor disagreement) with a directional read.
LABEL_DIRECTION = {"trending_up": "up", "trending_down": "down"}

_M5_MINUTES = 5

#: What a grade row's context says when nobody took a snapshot at the stamp.
#: A NAMED absence rather than a blank field: `{}` reads as "this row has no
#: context" and as "nobody has filled it in yet", and a store cannot tell those
#: two apart. :func:`append_grades` refuses a blank; this is what a grader with
#: no entry to read writes instead, and it is never mistaken for a measurement.
#: The trader-facing guarantee - a REAL snapshot beside every graded click - is
#: kept where the entry is in hand: `DayReviewService.build_reads_for` calls
#: :func:`context_for` for every read it grades.
CONTEXT_UNMEASURED = {
    "availability": UNMEASURED,
    "reason": "no context snapshot was taken at this read's stamp",
}


class PoolingError(ValueError):
    """A statistic that mixed a clicked read with an extracted one."""


class ContextMissingError(ValueError):
    """A gradable grade offered to the store with no context snapshot."""


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def _market_tz():
    from market_calendar import MARKET_TZ

    return MARKET_TZ


def _aware(moment: datetime | None) -> datetime | None:
    """`moment` with a zone attached, never stripped of one.

    A naive stamp is the desk's own wall clock (that is what `datetime.now()`
    writes), so it is ATTACHED to the local zone rather than assumed to be UTC.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.astimezone()
    return moment


def _as_datetime(value: Any) -> datetime | None:
    """Any stored stamp shape -> an aware datetime, or None."""
    if isinstance(value, datetime):
        return _aware(value)
    if isinstance(value, date):
        return None
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _aware(datetime.fromisoformat(text))
    except ValueError:
        return None


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _digest(*parts: Any) -> str:
    return hashlib.sha1(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()[:12]


def _entry_stamp(entry: Mapping[str, Any]) -> datetime | None:
    """When the trader ANSWERED - `mentor.responded_at` else `created_at`.

    The rule `day_review_markers._entry_stamp` already keeps: a card scheduled
    at 07:00 and answered at 07:22 is a 07:22 read, and an unanswered prompt is
    not a thing the trader said.
    """
    mentor = entry.get("mentor")
    if isinstance(mentor, Mapping):
        answered = _as_datetime(mentor.get("responded_at"))
        if answered is not None:
            return answered
    return _as_datetime(entry.get("created_at"))


def _decision_session_of(entry: Mapping[str, Any], stamp: datetime) -> str:
    """The session this read JUDGED (TJ-11F, trader 2026-09-19).

    Friday evening's D1 call is stamped with New York's Saturday date and
    belongs to FRIDAY: Monday's scan is new information the trader did not have.
    """
    try:
        import market_calendar

        judged = market_calendar.decision_session(stamp)
        if judged is not None:
            return judged.isoformat()
    except Exception:  # noqa: BLE001 - an unanswerable calendar falls back
        _log.debug("The decision session could not be read.", exc_info=True)
    return str(entry.get("session_date") or "")[:10]


def _benchmark_for(entry: Mapping[str, Any], because: str) -> str:
    """The benchmark a read is about. `DEFAULT_BENCHMARK` unless another is named.

    SPY wins whenever it is named at all, because the desk's default question is
    about SPY and a note that mentions both is not a call on the other one.
    """
    import re

    from market_story import BENCHMARKS

    haystack = " ".join(
        [str(entry.get("text") or ""), str(because or "")]
        + [str(item) for item in (entry.get("symbols") or ())]
    ).upper()
    found: list[str] = []
    for symbol in BENCHMARKS:
        match = re.search(r"(?<![A-Z0-9])" + symbol + r"(?![A-Z0-9])", haystack)
        if match:
            found.append(symbol)
    if not found or DEFAULT_BENCHMARK in found:
        return DEFAULT_BENCHMARK
    return found[0]


# ---------------------------------------------------------------------------
# item 1 - one read row per prediction
# ---------------------------------------------------------------------------
def read_rows(entries: Iterable[Mapping[str, Any]], *, session: str) -> list[dict[str, Any]]:
    """One row per read the session's own entries carry, OLDEST FIRST.

    A machine row is never a read (`market_journal.is_machine_entry` is the ONE
    filter, TJ-1), and neither is a pasted forecast: those are somebody else's
    words, and this ledger is about the trader's.
    """
    import market_journal

    target = str(session or "")[:10]
    made: list[tuple[datetime, dict[str, Any]]] = []
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("event_type") or "entry") != "entry":
            continue
        if market_journal.is_machine_entry(entry):
            continue
        if str(entry.get("origin") or "") == market_journal.ORIGIN_EXTERNAL_FORECAST:
            continue
        stamp = _entry_stamp(entry)
        if stamp is None:
            continue
        judged = _decision_session_of(entry, stamp)
        if target and judged != target:
            continue
        made.append((stamp, _read_row(entry, stamp=stamp, session=judged or target)))
    made.sort(key=lambda item: item[0])
    return [row for _stamp, row in made]


def _read_row(entry: Mapping[str, Any], *, stamp: datetime, session: str) -> dict[str, Any]:
    import market_journal

    click = market_journal.prediction_of(entry)
    timeframe = str(entry.get("timeframe") or "").strip().upper()
    if click is not None:
        source = SOURCE_CLICK
        direction = click.direction
        horizon = click.horizon
        confidence = click.confidence
        because = click.because
        # A click is not a quotation, so it carries no span.
        span: tuple[int, int] | tuple[()] = ()
    else:
        import market_thesis

        draft = market_thesis.extract_thesis(entry)
        source = SOURCE_EXTRACTED
        horizon = market_journal.HORIZON_FOR_TIMEFRAME.get(
            timeframe, market_journal.HORIZON_REST_OF_DAY
        )
        direction = _direction_for_stance(draft.stance, horizon)
        confidence = ""
        because = ""
        raw_span = draft.spans.get("stance")
        span = tuple(raw_span) if raw_span else ()
    if not timeframe:
        timeframe = (
            market_journal.TIMEFRAME_D1
            if horizon == market_journal.HORIZON_NEXT_5_SESSIONS
            else market_journal.TIMEFRAME_M5
        )
    benchmark = _benchmark_for(entry, because)
    entry_id = str(entry.get("entry_id") or "")
    return {
        "schema": SCHEMA_READ,
        "read_id": "rd-" + _digest(entry_id, source, horizon, benchmark, stamp.isoformat()),
        "entry_id": entry_id,
        "session": str(session or "")[:10],
        "stamp": stamp.isoformat(),
        "benchmark": benchmark,
        "source": source,
        "direction": direction,
        "horizon": horizon,
        "confidence": confidence,
        "because": because,
        "span": span,
        "timeframe": timeframe,
    }


def _direction_for_stance(stance: str, horizon: str) -> str:
    """An extracted stance as a direction (lead decision 7, 2026-09-19).

    bullish -> up, bearish -> down, neutral -> the horizon's own no-direction
    word. `cautious` and `unstated` are NOT calls and stay as they were read, so
    :func:`is_gradable` refuses them.
    """
    import market_journal

    mapped = _STANCE_DIRECTION.get(str(stance or ""))
    if mapped:
        return mapped
    if str(stance or "") == "neutral":
        return (
            "range"
            if horizon == market_journal.HORIZON_NEXT_5_SESSIONS
            else "chop"
        )
    return str(stance or "")


def is_gradable(row: Mapping[str, Any]) -> bool:
    """Can this read be right or wrong at all?

    `No view` is a COMPLETE answer and an `unstated` note is a note. Both are
    recorded; neither is ever graded (packet items 1 and 2).
    """
    return str((row or {}).get("direction") or "") in GRADABLE_DIRECTIONS


def pooled_accuracy(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """One accuracy cell over rows that share a SOURCE.

    Raises :class:`PoolingError` on a mix: decision 0021 answer 29 keeps a
    stated call and an inferred stance apart, and a rate that averaged the two
    would describe neither.
    """
    listed = [dict(row) for row in rows or ()]
    sources = {str(row.get("source") or "") for row in listed if row.get("source")}
    if len(sources) > 1:
        raise PoolingError(
            "a clicked read and an extracted one are different evidence and are "
            f"never pooled in one statistic: {sorted(sources)}"
        )
    cell = accuracy(listed)
    cell["source"] = next(iter(sources), "")
    return cell


def accuracy(grades: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Integer counts, and the ONE Wilson on the closed horizons.

    `n` is the CLOSED horizons - right + wrong + flat (lead decision 3: a flat
    reading sits IN the denominator). A pending horizon is printed and is in
    neither half, and an unmeasured one is never assumed into a rate.
    """
    import evidence_stats
    import swing_headline

    right = wrong = flat = pending = unmeasured = 0
    for row in grades or ():
        verdict = str((row or {}).get("verdict") or "")
        if verdict == VERDICT_RIGHT:
            right += 1
        elif verdict == VERDICT_WRONG:
            wrong += 1
        elif verdict == VERDICT_FLAT:
            flat += 1
        elif verdict.startswith(PENDING_PREFIX):
            pending += 1
        else:
            unmeasured += 1
    total = right + wrong + flat
    return {
        "right": right,
        "wrong": wrong,
        "flat": flat,
        "pending": pending,
        "unmeasured": unmeasured,
        "n": total,
        "rate": (right / total) if total else None,
        "rate_lb": swing_headline.wilson_lower_bound(right, total),
        "meets_floor": total >= evidence_stats.MIN_REPORTABLE_N,
    }


# ---------------------------------------------------------------------------
# item 4 - the naive baselines, on the trader's own stamps
# ---------------------------------------------------------------------------
def baseline_reads(rows: Iterable[Mapping[str, Any]], name: str) -> list[dict[str, Any]]:
    """The same stamps, answered by a naive rule instead of by the trader.

    The rule reads each row's OWN context snapshot and no bars at all, so a
    baseline is point-in-time by construction and cannot see a bar the trader
    could not. A question the trader declined is never answered by a baseline.

    **Feed it GRADE rows, not bare read rows.** The `context` a baseline reads
    (`last_hour_spy`, `d1_environment`) is attached when a read is GRADED, so a
    caller that hands over the output of :func:`read_rows` gets `unmeasured` on
    every rule that needs one. TJ-12 and TJ-16 read the stored ledger
    (:func:`read_grades`) and pass each row's `read` merged with its `context`;
    the rows come back in the same order with the same stamps, so the trader's
    score and the baseline's are measured on identical questions.
    """
    rule = str(name or "")
    if rule not in BASELINES:
        raise ValueError(f"no such baseline: {name!r}")
    out: list[dict[str, Any]] = []
    for row in rows or ():
        made = dict(row)
        made["baseline"] = rule
        made["source"] = str(row.get("source") or "")
        if is_gradable(row):
            made["direction"] = _baseline_direction(rule, row)
        made["read_id"] = "bl-" + _digest(rule, str(row.get("read_id") or ""))
        made["confidence"] = ""
        made["because"] = f"baseline: {rule}"
        made["span"] = ()
        out.append(made)
    return out


def _baseline_direction(rule: str, row: Mapping[str, Any]) -> str:
    context = row.get("context")
    context = context if isinstance(context, Mapping) else {}
    if rule == "always_up":
        return "up"
    if rule == "same_as_the_last_hour":
        moved = str(context.get("last_hour_spy") or "")
        return moved if moved in ("up", "down") else UNMEASURED
    label = str(context.get("d1_environment") or "")
    return LABEL_DIRECTION.get(label, UNMEASURED)


# ---------------------------------------------------------------------------
# item 2 - the grade
# ---------------------------------------------------------------------------
def grade_read(
    row: Mapping[str, Any],
    *,
    m5_bars: Sequence[Any] = (),
    daily_bars: Sequence[Any] = (),
    atr: float | None = None,
    now: datetime,
    supersedes: str = "",
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One measured verdict for one read. PURE.

    Every bar it may see is handed in; it opens no store, has no clock of its
    own and calls no model. `atr` is the benchmark's POINT-IN-TIME daily
    ATR(14) - nothing after the decision session - and a missing one is
    `unmeasured`, never a band of zero.
    """
    import market_journal

    moment = _aware(now)
    read = dict(row or {})
    horizon = str(read.get("horizon") or "")
    result: dict[str, Any]
    if not is_gradable(read):
        result = _unmeasured_result("not_a_call")
    elif horizon == market_journal.HORIZON_NEXT_5_SESSIONS:
        result = _grade_five_sessions(read, daily_bars, atr, moment)
    else:
        result = _grade_rest_of_day(read, m5_bars, atr, moment)
    verdict = str(result.get("verdict") or "")
    # `gap`: an explicit reason a builder attached to the result (e.g. a
    # `pending` row whose anchor session has not closed yet), else the
    # `unmeasured:<reason>` suffix, so both a data gap and an open-anchor gap
    # reach the same named field (TJ-14B's `grader_gap` Mentor kind reads it).
    gap = str(result.get("gap") or "")
    if not gap and verdict.startswith(f"{UNMEASURED_PREFIX}:"):
        reason = verdict.split(":", 1)[1]
        if reason != "not_a_call":
            gap = reason
    graded_at = moment.isoformat()
    return {
        "schema": SCHEMA_GRADE,
        "grade_id": "gr-" + _digest(read.get("read_id"), graded_at, verdict, supersedes),
        "read_id": str(read.get("read_id") or ""),
        "entry_id": str(read.get("entry_id") or ""),
        "session": str(read.get("session") or ""),
        "horizon": horizon,
        "benchmark": str(read.get("benchmark") or DEFAULT_BENCHMARK),
        "source": str(read.get("source") or ""),
        "direction": str(read.get("direction") or ""),
        "verdict": verdict,
        "anchor_price": result.get("anchor_price"),
        "anchor_at": result.get("anchor_at"),
        "final_price": result.get("final_price"),
        "final_at": result.get("final_at"),
        "move": result.get("move"),
        "move_atr": result.get("move_atr"),
        "checkpoints": result.get("checkpoints", ()),
        "atr": atr,
        "flat_band_atr": FLAT_BAND_ATR,
        "flat_band_rule": FLAT_BAND_RULE,
        # The read itself travels with its grade so the nightly hook can
        # re-measure a matured horizon without a second join, and so a stored
        # row can be read years later without the ledger beside it.
        "read": read,
        "context": dict(context or read.get("context") or CONTEXT_UNMEASURED),
        # A plain field, never a signal: what the desk needed and did not have.
        # TJ-14B's dormant `grader_gap` Mentor kind reads THIS.
        "grader_gap": gap,
        "supersedes": str(supersedes or ""),
        "graded_at": graded_at,
    }


def _unmeasured_result(reason: str) -> dict[str, Any]:
    return {
        "verdict": f"{UNMEASURED_PREFIX}:{reason}",
        "anchor_price": None, "anchor_at": None,
        "final_price": None, "final_at": None,
        "move": None, "move_atr": None, "checkpoints": (),
    }


def _pending_result(day: str, **fields: Any) -> dict[str, Any]:
    out = {
        "verdict": f"{PENDING_PREFIX} {day}",
        "anchor_price": None, "anchor_at": None,
        "final_price": None, "final_at": None,
        "move": None, "move_atr": None, "checkpoints": (),
    }
    out.update(fields)
    return out


def _verdict_for(direction: str, move_atr: float) -> str:
    """The band decides, and its EDGE is inside it.

    A `Chop` or `Range` call is a call like any other: right inside the band and
    wrong outside it. A directional call inside the band is `flat` - the trader
    was not wrong, the market did nothing.
    """
    inside = abs(move_atr) <= FLAT_BAND_ATR
    if direction in FLAT_DIRECTIONS:
        return VERDICT_RIGHT if inside else VERDICT_WRONG
    if inside:
        return VERDICT_FLAT
    if direction == "up":
        return VERDICT_RIGHT if move_atr > 0 else VERDICT_WRONG
    return VERDICT_RIGHT if move_atr < 0 else VERDICT_WRONG


def _grade_rest_of_day(
    read: Mapping[str, Any], m5_bars: Sequence[Any], atr: float | None, now: datetime
) -> dict[str, Any]:
    import market_early_close
    from completed_bars import bar_time, completed_m5_bars

    session = _as_date(read.get("session"))
    if session is None:
        return _unmeasured_result("no_session_date")
    close_at = market_early_close.session_close(session)
    day = session.isoformat()
    if now < close_at:
        # The bell has not rung. The last bar is forming and the read is open.
        return _pending_result(day)
    stamp = _as_datetime(read.get("stamp"))
    if stamp is None:
        return _unmeasured_result("no_stamp")
    completed = completed_m5_bars(list(m5_bars or ()), now=now)
    anchor_bar = None
    final_bar = None
    for bar in completed:
        start = bar_time(bar)
        if start is None:
            continue
        start = _aware(start)
        if start > stamp and anchor_bar is None:
            anchor_bar = (start, bar)
        # The session's own close, from the calendar - never the last row a
        # provider handed over. Yahoo has served post-bell rows on a half day.
        if start + timedelta(minutes=_M5_MINUTES) <= close_at:
            final_bar = (start, bar)
    if anchor_bar is None:
        return _unmeasured_result("no_completed_bar_after_the_stamp")
    if final_bar is None or final_bar[0] < anchor_bar[0]:
        return _unmeasured_result("no_completed_bar_at_the_close")
    anchor_price = _number(_field(anchor_bar[1], "open"))
    final_price = _number(_field(final_bar[1], "close"))
    if anchor_price is None or final_price is None:
        return _unmeasured_result("unreadable_bar")
    measured = {
        "anchor_price": anchor_price,
        "anchor_at": anchor_bar[0].isoformat(),
        "final_price": final_price,
        "final_at": final_bar[0].isoformat(),
        "move": final_price - anchor_price,
        "move_atr": None,
        "checkpoints": (),
    }
    band = _number(atr)
    if band is None or band <= 0:
        measured["verdict"] = f"{UNMEASURED_PREFIX}:no_atr"
        return measured
    measured["move_atr"] = measured["move"] / band
    measured["verdict"] = _verdict_for(
        str(read.get("direction") or ""), measured["move_atr"]
    )
    return measured


def _grade_five_sessions(
    read: Mapping[str, Any], daily_bars: Sequence[Any], atr: float | None, now: datetime
) -> dict[str, Any]:
    import market_calendar

    session = _as_date(read.get("session"))
    if session is None:
        return _unmeasured_result("no_session_date")
    if not _session_has_closed(session, now):
        # The decision session is still trading, so its own daily bar is
        # FORMING, not a close (reviewer, 2026-09-20: reproduced an anchor of
        # 130.0 read off a still-open bar). The row waits like any other open
        # horizon rather than anchor on a price that is not final yet.
        return _pending_result(
            session.isoformat(),
            gap="the decision session has not closed yet",
        )
    closes = _daily_closes(daily_bars)
    anchor_price = closes.get(session)
    if anchor_price is None:
        return _unmeasured_result("no_anchor_close")
    band = _number(atr)
    checkpoints: list[dict[str, Any]] = []
    cursor = session
    wanted = max(D1_CHECKPOINT_SESSIONS)
    horizon_days: dict[int, date] = {}
    try:
        for step in range(1, wanted + 1):
            cursor = market_calendar.next_session(cursor)
            if step in D1_CHECKPOINT_SESSIONS:
                horizon_days[step] = cursor
    except Exception:  # noqa: BLE001 - an unreachable calendar is unmeasured
        return _unmeasured_result("no_exchange_calendar")
    final: dict[str, Any] | None = None
    for step in D1_CHECKPOINT_SESSIONS:
        day = horizon_days[step]
        cell: dict[str, Any] = {
            "sessions": step, "session": day.isoformat(),
            "close": None, "move": None, "move_atr": None, "status": "",
        }
        if not _session_has_closed(day, now):
            cell["status"] = PENDING_PREFIX
        else:
            close = closes.get(day)
            if close is None:
                cell["status"] = UNMEASURED
            else:
                cell["close"] = close
                cell["move"] = close - anchor_price
                if band and band > 0:
                    cell["move_atr"] = cell["move"] / band
                cell["status"] = "measured"
        checkpoints.append(cell)
        if step == max(D1_CHECKPOINT_SESSIONS):
            final = cell
    verdict_day = horizon_days[max(D1_CHECKPOINT_SESSIONS)].isoformat()
    base = {
        "anchor_price": anchor_price,
        "anchor_at": session.isoformat(),
        "final_price": None, "final_at": None,
        "move": None, "move_atr": None,
        "checkpoints": tuple(checkpoints),
    }
    if final is None or final["status"] == PENDING_PREFIX:
        return _pending_result(verdict_day, **{**base, "checkpoints": tuple(checkpoints)})
    if final["status"] != "measured":
        return {**base, "verdict": f"{UNMEASURED_PREFIX}:no_daily_close_at_the_horizon"}
    base["final_price"] = final["close"]
    base["final_at"] = final["session"]
    base["move"] = final["move"]
    if band is None or band <= 0:
        return {**base, "verdict": f"{UNMEASURED_PREFIX}:no_atr"}
    base["move_atr"] = final["move_atr"]
    base["verdict"] = _verdict_for(str(read.get("direction") or ""), base["move_atr"])
    return base


def _session_has_closed(day: date, now: datetime) -> bool:
    """Has `day`'s own close already happened? A forming daily bar is not one.

    `chart_snapshot.load_d1_bars` returns every row in the parquet file and does
    NOT strip today's forming bar (measured 2026-09-19), so the cut is made here.
    """
    import market_early_close

    return now >= market_early_close.session_close(day)


def _daily_closes(daily_bars: Sequence[Any]) -> dict[date, float]:
    out: dict[date, float] = {}
    for bar in daily_bars or ():
        day = _as_date(_field(bar, "dt") or _field(bar, "date") or _field(bar, "timestamp"))
        close = _number(_field(bar, "close"))
        if day is None or close is None:
            continue
        out[day] = close
    return out


def _field(bar: Any, name: str) -> Any:
    if isinstance(bar, Mapping):
        for key in (name, name.capitalize(), name[:1]):
            if key in bar:
                return bar[key]
        return None
    return getattr(bar, name, None)


def daily_atr(daily_bars: Sequence[Any], *, through: Any, length: int = 14) -> float | None:
    """The benchmark's daily ATR(14) as of `through`, POINT-IN-TIME.

    Nothing after the decision session is in it - an ATR that included the move
    being graded would size the band with the answer.
    """
    from indicators.atr import wilder_atr

    edge = _as_date(through)
    if edge is None:
        return None
    kept = [
        bar for bar in daily_bars or ()
        if (day := _as_date(_field(bar, "dt") or _field(bar, "date"))) is not None
        and day <= edge
    ]
    return wilder_atr(kept, length)


# ---------------------------------------------------------------------------
# item 3 / TJ-16 item 1 - the point-in-time context snapshot
# ---------------------------------------------------------------------------
def context_for(
    entry: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    bars: Mapping[str, Any] | None = None,
    spy_m5_bars: Sequence[Any] = (),
    prior_daily_bar: Mapping[str, Any] | None = None,
    d1_labels: Mapping[str, Any] | None = None,
    prior_grades: Iterable[Mapping[str, Any]] = (),
    latest_d1_click: Mapping[str, Any] | None = None,
    mood_entries: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """What the desk looked like AT THE STAMP, and nothing after it.

    The market half is `trade_mentor_context_v2`: the entry's own stored block
    when it has one, else a rebuild through
    `trade_mentor_context.internals_at` - there is never a second builder. A
    field the desk cannot measure reads :data:`UNMEASURED`.

    `mood_entries` is TJ-7's one addition and it is POINT-IN-TIME like every
    other field here: the score of the latest mood already recorded at or
    before the stamp, else :data:`UNMEASURED`. The mood the trader clicks is
    usually on the session's LAST card, after the trade and often after the
    close, so a later one filled in here would be hindsight dressed as a
    measurement - and a contrast walks every scalar in this block.
    """
    import trade_mentor_context

    stamp = _as_datetime(row.get("stamp")) or _entry_stamp(entry)
    session = str(row.get("session") or entry.get("session_date") or "")[:10]
    stored = None
    mentor = entry.get("mentor") if isinstance(entry, Mapping) else None
    if isinstance(mentor, Mapping):
        block = mentor.get("context")
        if (
            isinstance(block, Mapping)
            and str(block.get("schema") or "") == trade_mentor_context.SCHEMA
        ):
            stored = dict(block)
    if stored is None:
        try:
            stored = trade_mentor_context.internals_at(session, stamp, bars or {})
        except Exception:  # noqa: BLE001 - an unbuildable block is unmeasured
            _log.debug("The internals block could not be rebuilt.", exc_info=True)
            stored = {}

    completed = _bars_through(spy_m5_bars, stamp)
    return {
        "internals": stored,
        "hour": _hour_of(stamp),
        "spy_vs_session_vwap": _vs_session_vwap(completed),
        "spy_vs_prior_range": _vs_prior_range(completed, prior_daily_bar),
        "gap_pct": _gap_pct(completed, prior_daily_bar),
        "d1_environment": _prior_label(session, d1_labels),
        "last_hour_spy": _last_hour(completed),
        "agrees_with_own_d1": _agrees_with_d1(row, latest_d1_click),
        "previous_call_verdict": _previous_verdict(prior_grades, stamp),
        "confidence": str(row.get("confidence") or "") or UNMEASURED,
        "direction": str(row.get("direction") or "") or UNMEASURED,
        "mood": _mood_at(mood_entries, stamp),
    }


def _mood_at(entries: Iterable[Mapping[str, Any]], stamp: datetime | None) -> Any:
    """The mood score the read was made KNOWING, or :data:`UNMEASURED`.

    Never a zero and never a later mood. `market_journal.mood_at` compares with
    `astimezone` and is the ONE reader; a block with no face clicked is a
    process answer, not a score, and reads `unmeasured` here.
    """
    if stamp is None:
        return UNMEASURED
    try:
        import market_journal

        recorded = market_journal.mood_at(entries or (), stamp)
    except Exception:  # noqa: BLE001 - an unreadable mood is unmeasured, never a zero
        _log.debug("A mood could not be read for a context snapshot.", exc_info=True)
        return UNMEASURED
    score = getattr(recorded, "score", None)
    if isinstance(score, bool) or not isinstance(score, int):
        return UNMEASURED
    return score


def _bars_through(bars: Sequence[Any], stamp: datetime | None) -> list[Any]:
    """Every bar that had FINISHED at `stamp`. The rest is the future."""
    if stamp is None:
        return []
    from completed_bars import completed_m5_bars

    return completed_m5_bars(list(bars or ()), now=stamp)


def _hour_of(stamp: datetime | None) -> Any:
    if stamp is None:
        return UNMEASURED
    return stamp.astimezone(_market_tz()).hour


def _vs_session_vwap(bars: Sequence[Any]) -> str:
    if not bars:
        return UNMEASURED
    volume_sum = 0.0
    value_sum = 0.0
    for bar in bars:
        high = _number(_field(bar, "high"))
        low = _number(_field(bar, "low"))
        close = _number(_field(bar, "close"))
        volume = _number(_field(bar, "volume"))
        if None in (high, low, close):
            continue
        typical = (high + low + close) / 3.0
        weight = volume if volume and volume > 0 else 1.0
        value_sum += typical * weight
        volume_sum += weight
    last = _number(_field(bars[-1], "close"))
    if not volume_sum or last is None:
        return UNMEASURED
    vwap = value_sum / volume_sum
    if last > vwap:
        return "above"
    if last < vwap:
        return "below"
    return "at"


def _vs_prior_range(bars: Sequence[Any], prior: Mapping[str, Any] | None) -> str:
    if not bars or not prior:
        return UNMEASURED
    last = _number(_field(bars[-1], "close"))
    high = _number(_field(prior, "high"))
    low = _number(_field(prior, "low"))
    if last is None or high is None or low is None:
        return UNMEASURED
    if last > high:
        return "above"
    if last < low:
        return "below"
    return "inside"


def _gap_pct(bars: Sequence[Any], prior: Mapping[str, Any] | None) -> Any:
    if not bars or not prior:
        return UNMEASURED
    first = _number(_field(bars[0], "open"))
    close = _number(_field(prior, "close"))
    if first is None or close is None or close <= 0:
        return UNMEASURED
    return (first - close) / close * 100.0


def _prior_label(session: str, labels: Mapping[str, Any] | None) -> str:
    """The desk's D1 label for the PRIOR session - the newest one that existed.

    The desk labels a session from its own closed bars, so at 08:02 on the 18th
    the newest label the trader could have seen is the 17th's. Measured
    2026-09-19: `d1_environment.jsonl` holds 15 rows over five sessions and none
    for the 18th, so this honestly reads `unmeasured` most days.
    """
    if not labels:
        return UNMEASURED
    day = _as_date(session)
    if day is None:
        return UNMEASURED
    try:
        import market_calendar

        prior = market_calendar.previous_session(day)
    except Exception:  # noqa: BLE001
        return UNMEASURED
    label = str(labels.get(prior.isoformat()) or "")
    return label or UNMEASURED


def _last_hour(bars: Sequence[Any]) -> str:
    """Which way SPY ran in the hour BEFORE the stamp."""
    from completed_bars import bar_time

    if not bars:
        return UNMEASURED
    end = bar_time(bars[-1])
    if end is None:
        return UNMEASURED
    window_start = _aware(end) + timedelta(minutes=_M5_MINUTES) - timedelta(minutes=60)
    inside = [
        bar for bar in bars
        if (start := bar_time(bar)) is not None and _aware(start) >= window_start
    ]
    if not inside:
        return UNMEASURED
    opened = _number(_field(inside[0], "open"))
    closed = _number(_field(inside[-1], "close"))
    if opened is None or closed is None:
        return UNMEASURED
    if closed > opened:
        return "up"
    if closed < opened:
        return "down"
    return "flat"


def _agrees_with_d1(row: Mapping[str, Any], d1: Mapping[str, Any] | None) -> Any:
    if not d1:
        return UNMEASURED
    mine = str(row.get("direction") or "")
    theirs = str(d1.get("direction") or "")
    if mine not in GRADABLE_DIRECTIONS or theirs not in GRADABLE_DIRECTIONS:
        return UNMEASURED
    return mine == theirs


def _previous_verdict(prior_grades: Iterable[Mapping[str, Any]], stamp: datetime | None) -> str:
    """The last verdict the trader KNEW at the stamp - never one that matured later.

    A call that was still open when they clicked is not a previous verdict; that
    difference is the whole after-a-miss question.
    """
    if stamp is None:
        return "none"
    known: list[tuple[datetime, str]] = []
    for grade in prior_grades or ():
        verdict = str(grade.get("verdict") or "")
        if verdict not in (VERDICT_RIGHT, VERDICT_WRONG, VERDICT_FLAT):
            continue
        made = _as_datetime(grade.get("stamp"))
        graded = _as_datetime(grade.get("graded_at"))
        if made is None or graded is None:
            continue
        if made >= stamp or graded > stamp:
            continue
        known.append((made, verdict))
    if not known:
        return "none"
    known.sort(key=lambda item: item[0])
    return known[-1][1]


# ---------------------------------------------------------------------------
# item 6 - the congruence lines. PRINTED, never pushed.
# ---------------------------------------------------------------------------
def select_read(
    rows: Iterable[Mapping[str, Any]], *, timeframe: str
) -> tuple[dict[str, Any] | None, str]:
    """The ONE read of `timeframe` a congruence line may be compared with.

    `(read, note)`. Two rules, both from the fix round 2026-09-20:

    * **A CLICKED read always wins over any extracted one** for that timeframe -
      a stated call outranks an inferred stance, always (decision 0021 answer 29).
    * An EXTRACTED stance is used only when the session's extracted stances of
      that timeframe do NOT contradict each other. On 2026-09-17 one note read
      `up` and another on the same session read `down`; comparing either one
      with the desk's label would be picking a view the trader never stated. The
      note says so and the line is `unmeasured`.
    """
    mine = [
        dict(row) for row in rows or ()
        if _timeframe_of(row) == str(timeframe or "").strip().upper()
        and is_gradable(row)
    ]
    if not mine:
        return None, ""
    clicked = [row for row in mine if str(row.get("source") or "") == SOURCE_CLICK]
    if clicked:
        return clicked[-1], ""
    directional = [row for row in mine if _read_direction(row)]
    ups = [row for row in directional if _read_direction(row) == "up"]
    downs = [row for row in directional if _read_direction(row) == "down"]
    if ups and downs:
        return None, (
            f"your notes read both ways ({len(ups)} up, {len(downs)} down) - "
            "no single read to compare"
        )
    if directional:
        return directional[-1], ""
    return mine[-1], ""


def read_phrase(read: Mapping[str, Any] | None) -> str:
    """How a read is NAMED on a trader-facing line, source and all.

    Live clicks are 0 and every live read row is an extraction, so a surface
    that said "your D1 read is up" would be presenting an inferred stance as the
    trader's own stated call (reviewer, 2026-09-20). A click says it was
    clicked, and an extraction says the desk read it out of a note.
    """
    if not read:
        return "no read"
    direction = str(read.get("direction") or "") or UNMEASURED
    if str(read.get("source") or "") == SOURCE_CLICK:
        stamp = _as_datetime(read.get("stamp"))
        when = f" (clicked {stamp.strftime('%H:%M')})" if stamp else " (clicked)"
        return f"your call: {direction}{when}"
    return f"we read your note as {direction}"


def congruence_lines(
    *,
    session: str,
    d1_read: Mapping[str, Any] | None = None,
    d1_label: str = "",
    decisions: Iterable[Mapping[str, Any]] = (),
    claims: Iterable[Mapping[str, Any]] = (),
    trades: Iterable[Mapping[str, Any]] = (),
    d1_note: str = "",
    m5_read: Mapping[str, Any] | None = None,
    m5_note: str = "",
) -> tuple[dict[str, Any], ...]:
    """Your view against the desk's, against your picks' and against your fills'.

    Trader, 2026-09-19: *"if my thoughts about the market are potentially
    incongruent with my overall D1 picture, I want to know about it."* Every
    line names its TIMEFRAME, its SOURCE (a click or an extraction, in plain
    words), its `n` and its source ids; a missing side is named and is never
    read as agreement; a count under `evidence_stats.MIN_REPORTABLE_N` says
    `too few to call` beside its counts. Nothing here carries a threshold, a
    priority or an alert, and it reaches no notifier: decision 0021 answer 15 -
    printed, never pushed, never acted on.

    :data:`CONGRUENCE_KINDS` is the D1 spine and is always all three lines in
    that order. An M5 line is APPENDED beside the D1 picks line when an
    ``m5_read`` is given (lead decision 6: half the trader's reviewed decisions
    are M5, and a rest-of-day read belongs with M5 picks, not D1 ones).
    """
    lines = [
        _line_desk_label(d1_read, d1_label, d1_note),
        _line_picks(d1_read, decisions, claims, session, note=d1_note),
        _line_fills(d1_read, trades, d1_note),
    ]
    if m5_read is not None or m5_note:
        lines.insert(
            2,
            _line_picks(
                m5_read, decisions, claims, session,
                note=m5_note, kind=CONGRUENCE_M5_KIND,
                timeframe=_M5_TIMEFRAME,
            ),
        )
    return tuple(lines)


def _floor_note(n: int) -> str:
    import evidence_stats

    if n and n < evidence_stats.MIN_REPORTABLE_N:
        return f" · too few to call (n={n}, under {evidence_stats.MIN_REPORTABLE_N})"
    return ""


def _congruence_verdict(mine: str, majority: str, n: int) -> str:
    """`agrees` / `disagrees`, or `VERDICT_TOO_FEW` under the reporting floor.

    Only a PICKS line calls this (`_line_picks`, D1 and M5 alike): the desk
    label and fills lines are not in the packet that named this floor.
    """
    import evidence_stats

    if n < evidence_stats.MIN_REPORTABLE_N:
        return VERDICT_TOO_FEW
    return "agrees" if mine == majority else "disagrees"


def _read_direction(read: Mapping[str, Any] | None) -> str:
    if not read:
        return ""
    direction = str(read.get("direction") or "")
    return direction if direction in ("up", "down") else ""


def _timeframe_of(read: Mapping[str, Any] | None) -> str:
    """Which population this read may be compared with (lead decision 6).

    TJ-15 measured 1,026 M5 decisions against 1,013 D1 ones over 20 sessions, so
    a line that pooled the two would compare a D1 view with a mostly-M5 crowd.
    """
    import market_journal

    if not read:
        return ""
    timeframe = str(read.get("timeframe") or "").strip().upper()
    if timeframe:
        return timeframe
    horizon = str(read.get("horizon") or "")
    return (
        market_journal.TIMEFRAME_D1
        if horizon == market_journal.HORIZON_NEXT_5_SESSIONS
        else market_journal.TIMEFRAME_M5
    )


def _line(kind: str, **fields: Any) -> dict[str, Any]:
    row = {
        "kind": kind,
        "text": "",
        "verdict": UNMEASURED,
        "counts": {},
        "source_ids": [],
        "timeframe": "",
        "missing": "",
    }
    row.update(fields)
    return row


def _line_desk_label(
    read: Mapping[str, Any] | None, label: str, note: str = ""
) -> dict[str, Any]:
    kind = CONGRUENCE_KINDS[0]
    timeframe = _timeframe_of(read) or "D1"
    source_ids = [str(read.get("read_id") or "")] if read else []
    name = str(label or "").strip()
    if not read:
        # Own content FIRST: the desk's own label, then the reason there is
        # nothing to compare it with - a contradiction note, or plainly none.
        desk_said = (
            f"the desk reads {name}" if name
            else "the desk has no D1 label for this session"
        )
        reason = note or "there is no D1 read today to compare it with"
        return _line(
            kind,
            text=f"{desk_said}; {reason}",
            missing="your D1 read", timeframe=timeframe, source_ids=source_ids,
        )
    mine = _read_direction(read)
    said = read_phrase(read)
    if not name:
        return _line(
            kind,
            text=f"{said}; the desk has no D1 label for this session",
            missing="the desk's D1 label", timeframe=timeframe,
            source_ids=source_ids,
        )
    theirs = LABEL_DIRECTION.get(name, "")
    if not mine or not theirs:
        return _line(
            kind,
            text=(
                f"{said}; the desk reads {name}, which carries no direction"
                if not theirs
                else f"{said} - not a direction; the desk reads {name}"
            ),
            missing="" if theirs else f"a direction in {name}",
            timeframe=timeframe, source_ids=source_ids,
        )
    return _line(
        kind,
        text=f"{said}; the desk reads {name}",
        verdict="agrees" if mine == theirs else "disagrees",
        timeframe=timeframe, source_ids=source_ids,
    )


#: A rejection of a side, counted beside the likes on an M5 line but never
#: folded into the side mix: "not today" says what the trader did NOT take, and
#: reading it as a pick of the other side would be a claim they never made.
_NOT_TODAY_MARKERS = ("not_today", "m5_not_today")


def _line_picks(
    read: Mapping[str, Any] | None,
    decisions: Iterable[Mapping[str, Any]],
    claims: Iterable[Mapping[str, Any]],
    session: str,
    *,
    note: str = "",
    kind: str = "",
    timeframe: str = "",
) -> dict[str, Any]:
    import market_journal

    kind = kind or CONGRUENCE_KINDS[1]
    timeframe = (timeframe or _timeframe_of(read) or "D1").strip().upper()
    longs: list[str] = []
    shorts: list[str] = []
    not_today = 0
    for row in decisions or ():
        if str(row.get("timeframe") or "").strip().upper() != timeframe:
            continue
        verdict = str(row.get("verdict") or "")
        source = str(row.get("capture_id") or "") or f"decision:{row.get('symbol')}"
        if any(marker in verdict for marker in _NOT_TODAY_MARKERS):
            not_today += 1
            continue
        if verdict != "like":
            continue
        side = str(row.get("side") or "").strip().upper()
        if side == "LONG":
            longs.append(source)
        elif side == "SHORT":
            shorts.append(source)
    # A claimed pick is a D1 claim by construction (`claimed_picks.HORIZON_D1`),
    # so it counts beside a D1 like and never beside an M5 one.
    if timeframe == market_journal.TIMEFRAME_D1:
        for claim in claims or ():
            if str(claim.get("session_date") or "")[:10] != str(session or "")[:10]:
                continue
            side = str(claim.get("side") or "").strip().upper()
            source = f"claim:{claim.get('symbol')}:{claim.get('claim_at')}"
            if side == "LONG":
                longs.append(source)
            elif side == "SHORT":
                shorts.append(source)
    counts = {"long": len(longs), "short": len(shorts)}
    if timeframe != market_journal.TIMEFRAME_D1:
        counts["not_today"] = not_today
    source_ids = longs + shorts
    total = len(source_ids)
    what = "likes and claims" if timeframe == market_journal.TIMEFRAME_D1 else "likes"
    majority = "up" if counts["long"] > counts["short"] else (
        "down" if counts["short"] > counts["long"] else ""
    )
    tail = f", {not_today} not today" if counts.get("not_today") else ""
    if not total:
        mine_mix = f"no {timeframe} {what} this session"
    elif majority:
        mine_mix = (
            f"{max(counts['long'], counts['short'])} of {total} {timeframe} {what} "
            f"were {'LONG' if majority == 'up' else 'SHORT'} "
            f"(long {counts['long']}, short {counts['short']})"
        )
    else:
        # A TIE names no side: "half and half" is not a lean, and printing one
        # would invent a crowd the session did not have.
        mine_mix = (
            f"{counts['long']} long, {counts['short']} short - no lean in "
            f"{total} {timeframe} {what}"
        )
    mine_mix = f"{mine_mix}{tail}"
    if not read:
        # Own content FIRST: the mix this line counted, then the reason there
        # is nothing to compare it with - a contradiction note, or plainly
        # none. The counts are never dropped just because the read is absent.
        reason = note or "no read to compare them with"
        return _line(
            kind,
            text=f"{mine_mix}{_floor_note(total)}; {reason}",
            counts=counts, source_ids=source_ids,
            timeframe=timeframe, missing=f"your {timeframe} read",
        )
    said = read_phrase(read)
    if not total:
        return _line(
            kind, text=f"{said}; no {timeframe} {what} this session",
            counts=counts, source_ids=source_ids, timeframe=timeframe,
            missing=f"your {timeframe} {what}",
        )
    mine = _read_direction(read)
    text = f"{said}; {mine_mix}" + _floor_note(total)
    if not mine or not majority:
        return _line(
            kind, text=text, counts=counts, source_ids=source_ids,
            timeframe=timeframe,
            missing="" if mine else "a directional read",
        )
    return _line(
        kind, text=text, counts=counts, source_ids=source_ids, timeframe=timeframe,
        verdict=_congruence_verdict(mine, majority, total),
    )


def _line_fills(
    read: Mapping[str, Any] | None,
    trades: Iterable[Mapping[str, Any]],
    note: str = "",
) -> dict[str, Any]:
    kind = CONGRUENCE_KINDS[2]
    listed = [dict(trade) for trade in trades or ()]
    counts = {"bullish": 0, "bearish": 0, "unknown": 0}
    source_ids: list[str] = []
    if listed:
        import journal_exposure

        classified = journal_exposure.classify_all(listed)
        for trade in listed:
            trade_id = str(trade.get("trade_id") or "")
            source_ids.append(trade_id)
            exposure = classified.get(trade_id)
            bias = str(getattr(exposure, "market_bias", "") or "")
            if bias.startswith(journal_exposure.BIAS_BULLISH):
                counts["bullish"] += 1
            elif bias.startswith(journal_exposure.BIAS_BEARISH):
                counts["bearish"] += 1
            else:
                counts["unknown"] += 1
    total = counts["bullish"] + counts["bearish"]
    # A fill carries no timeframe of its own, so the line names the timeframe of
    # the READ it is compared with and says the fills are all of them. Every
    # line names a timeframe; a blank one reads as "nobody decided".
    timeframe = _timeframe_of(read) or "D1"
    if not listed:
        return _line(
            kind, text="no fills today", missing="your fills", timeframe=timeframe,
        )
    majority = "up" if counts["bullish"] > counts["bearish"] else (
        "down" if counts["bearish"] > counts["bullish"] else ""
    )
    mix = (
        f"{len(listed)} fills (all timeframes): {counts['bullish']} bullish, "
        f"{counts['bearish']} bearish, {counts['unknown']} the legs could not call"
    )
    if not majority and total:
        mix += " - no lean"
    if not read:
        # Own content FIRST: the fills' own mix, then the reason there is
        # nothing to compare it with.
        reason = note or "no read to compare them with"
        return _line(
            kind,
            text=f"{mix}{_floor_note(total)}; {reason}",
            counts=counts, source_ids=source_ids, missing=f"your {timeframe} read",
            timeframe=timeframe,
        )
    mine = _read_direction(read)
    said = read_phrase(read)
    text = f"{said}; {mix}" + _floor_note(total)
    if not mine or not majority:
        return _line(
            kind, text=text, counts=counts, source_ids=source_ids,
            timeframe=timeframe,
            missing="" if mine else "a directional read",
        )
    return _line(
        kind, text=text, counts=counts, source_ids=source_ids, timeframe=timeframe,
        verdict="agrees" if mine == majority else "disagrees",
    )


# ---------------------------------------------------------------------------
# the store: append-only JSONL under the durable Day Review folder
# ---------------------------------------------------------------------------
def _default_root() -> Path:
    from project_paths import DAY_REVIEW_READS_DIR

    # The constant is `<DAY_REVIEW_DIR>/reads`; `reads_path` appends `reads` to
    # whatever root it is given, so the root is that folder's parent.
    return Path(DAY_REVIEW_READS_DIR).parent


def reads_path(session: str, *, root: Any = None) -> Path:
    """`<root>/reads/<session>.jsonl` - one file per session, append-only.

    With no root it is `project_paths.DAY_REVIEW_READS_DIR`, the named durable
    constant this ledger owns.
    """
    base = Path(root) if root is not None else _default_root()
    return base / "reads" / f"{str(session or '')[:10]}.jsonl"


def _is_gradable_grade(grade: Mapping[str, Any]) -> bool:
    verdict = str(grade.get("verdict") or "")
    return (
        verdict in (VERDICT_RIGHT, VERDICT_WRONG, VERDICT_FLAT)
        or verdict.startswith(PENDING_PREFIX)
    )


def _has_real_context(grade: Mapping[str, Any]) -> bool:
    """Is this row's context a MEASUREMENT rather than a named absence?

    Keyed on `availability` rather than on the exact dict, so a caller cannot
    slip past the check by adding a key to the named absence.
    """
    context = grade.get("context") or {}
    if not context:
        return False
    return str(context.get("availability") or "") != UNMEASURED


def append_grades(session: str, rows: Iterable[Mapping[str, Any]], *, root: Any = None) -> int:
    """Append grade rows, refusing any gradable one with no context.

    TJ-16 item 1: the context snapshot ships WITH this packet so no graded
    click is ever stored without one - a row written blank can never be given
    one afterwards, because the market has moved on.

    **The store is STRICTER for a CLICK** (lead decision, fix round
    2026-09-20). A blank `context` is refused for every row; a CLICKED gradable
    grade is refused for the NAMED ABSENCE too, because a stated call is the
    evidence this whole ledger exists for and one stored without its snapshot is
    permanently unanswerable. An EXTRACTED row, and a RE-GRADE of an earlier row
    (`supersedes` set, whose context was taken when the read was first seen),
    may carry the named absence.

    The check runs over the WHOLE batch before anything is written, so a refusal
    leaves no half-written file behind.
    """
    listed = [dict(row) for row in rows or ()]
    for row in listed:
        if not _is_gradable_grade(row):
            continue
        if not (row.get("context") or {}):
            raise ContextMissingError(
                "a gradable grade may not be stored without its point-in-time "
                f"context: {row.get('grade_id') or row.get('read_id')}"
            )
        clicked = str(row.get("source") or "") == SOURCE_CLICK
        if clicked and not row.get("supersedes") and not _has_real_context(row):
            raise ContextMissingError(
                "a gradable CLICKED grade may not be stored with a named-absence "
                "context: a stated call stored without the snapshot it was made "
                f"in can never be given one: {row.get('grade_id') or row.get('read_id')}"
            )
    if not listed:
        return 0
    target = reads_path(session, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for row in listed:
            handle.write(
                json.dumps(row, default=str, sort_keys=True, separators=(",", ":")) + "\n"
            )
    return len(listed)


def read_grades(session: str, *, root: Any = None) -> list[dict[str, Any]]:
    """Every grade row for one session, in the order it was written."""
    target = reads_path(session, root=root)
    rows: list[dict[str, Any]] = []
    try:
        handle = target.open("r", encoding="utf-8")
    except OSError:
        return rows
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                _log.debug("market_read_grades: unreadable row skipped.")
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def current_grades(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The current view: a superseded row is hidden, never removed."""
    listed = [dict(row) for row in rows or ()]
    replaced = {str(row.get("supersedes") or "") for row in listed if row.get("supersedes")}
    return [row for row in listed if str(row.get("grade_id") or "") not in replaced]


#: What a verdict is WORTH, so a re-grade can only ever move a row UP.
#:
#: The defect this fixes (reviewer, 2026-09-20): `regrade_matured` appended
#: whenever the verdict CHANGED, and on a desk whose durable daily store is
#: empty a correct `pending 2026-09-25` was superseded by
#: `unmeasured:no_anchor_close` - after which the row was never looked at again,
#: because only `pending*` rows were revisited. The true answer was `right`.
#: **Missing data is uncertainty, never confirmation** (plan.md sec 5): an
#: `unmeasured` result never supersedes a `pending` row, and a row that is
#: already unmeasured for a DATA reason is revisited on every later run.
_VERDICT_RANK_MEASURED = 3
_VERDICT_RANK_PENDING = 2
_VERDICT_RANK_UNMEASURED_DATA = 1
_VERDICT_RANK_FINAL = 0
#: The one `unmeasured` reason that is FINAL: the trader answered `No view` (or
#: wrote no stance), and no bar arriving later can turn that into a call.
FINAL_UNMEASURED_REASONS = ("not_a_call", "no_session_date", "no_stamp")


def verdict_rank(verdict: str) -> int:
    """How far along a verdict is. A re-grade may only ever raise it."""
    text = str(verdict or "")
    if text in (VERDICT_RIGHT, VERDICT_WRONG, VERDICT_FLAT):
        return _VERDICT_RANK_MEASURED
    if text.startswith(PENDING_PREFIX):
        return _VERDICT_RANK_PENDING
    if text.startswith(f"{UNMEASURED_PREFIX}:"):
        reason = text.split(":", 1)[1]
        return (
            _VERDICT_RANK_FINAL
            if reason in FINAL_UNMEASURED_REASONS
            else _VERDICT_RANK_UNMEASURED_DATA
        )
    return _VERDICT_RANK_FINAL


def regrade_matured(
    now: datetime,
    *,
    root: Any = None,
    daily_bars_for: Callable[[str], Sequence[Any]] | None = None,
    m5_bars_for: Callable[[str, str], Sequence[Any]] | None = None,
    atr_for: Callable[[str, str], float | None] | None = None,
) -> list[dict[str, Any]]:
    """The nightly hook: re-measure the horizons that have matured. ONE function.

    Deterministic and modelless. Two rules, both from the fix round:

    * It revisits every row that is still OPEN - `pending`, and also one left
      `unmeasured` for a DATA reason (no bars, no ATR, no anchor close), because
      the store that was missing last night may be there tonight. Only
      `unmeasured:not_a_call` is final.
    * **A new row is appended only when the verdict moves UP**
      (:func:`verdict_rank`), so an absent store can never turn a correct
      `pending` into an `unmeasured` row and retire it. A closed horizon is
      closed and a second night writes nothing.

    The bars come from the SAME loaders the Day Review page uses
    (:func:`daily_bars_for`, :func:`session_bars_for`, :func:`atr_for`) unless
    the caller injects its own; a night that reads a different store from the
    page is the defect this signature exists to prevent.
    """
    moment = _aware(now)
    base = Path(root) if root is not None else _default_root()
    folder = base / "reads"
    written: list[dict[str, Any]] = []
    try:
        files = sorted(folder.glob("*.jsonl"))
    except OSError:
        return written
    load_daily = daily_bars_for or daily_bars_for_symbol
    load_m5 = m5_bars_for or session_bars_for
    load_atr = atr_for or atr_for_session
    for path in files:
        session = path.stem
        stored = read_grades(session, root=root)
        for grade in current_grades(stored):
            was = str(grade.get("verdict") or "")
            rank = verdict_rank(was)
            if rank not in (_VERDICT_RANK_PENDING, _VERDICT_RANK_UNMEASURED_DATA):
                continue
            read = grade.get("read")
            if not isinstance(read, Mapping):
                continue
            symbol = str(read.get("benchmark") or DEFAULT_BENCHMARK)
            daily: Sequence[Any] = ()
            m5: Sequence[Any] = ()
            try:
                if str(read.get("horizon") or "").endswith("sessions"):
                    daily = load_daily(symbol) or ()
                else:
                    m5 = load_m5(symbol, session) or ()
                band = load_atr(symbol, session)
            except Exception:  # noqa: BLE001 - one unreadable name costs one row
                _log.debug("A matured read could not be re-measured.", exc_info=True)
                continue
            fresh = grade_read(
                read, m5_bars=m5, daily_bars=daily, atr=band, now=moment,
                supersedes=str(grade.get("grade_id") or ""),
                context=grade.get("context") or {},
            )
            if verdict_rank(str(fresh.get("verdict") or "")) <= rank:
                # Nothing was learned, or less than before. The row stays as it
                # is and tomorrow's run looks at it again.
                continue
            try:
                append_grades(session, [fresh], root=root)
            except ContextMissingError:
                _log.debug("A matured grade had no context and was not stored.")
                continue
            written.append(fresh)
    return written


# ---------------------------------------------------------------------------
# the shared loaders - ONE daily store, ONE tape, ONE ATR
# ---------------------------------------------------------------------------
def daily_bars_for_symbol(symbol: str) -> list[Any]:
    """One benchmark's daily history: the durable store, else the desk's cache.

    THE one loader. The Day Review page and the nightly re-grade both call it,
    because a night that read a different store from the page graded a different
    market (reviewer, 2026-09-20: the page had a machine-cache fallback and the
    night did not, so the night turned a correct `pending` into `unmeasured`).

    `chart_snapshot.load_d1_bars` is the durable parquet store - off the Qt
    thread, mtime-cached, no network, no IB. Measured on the desk 2026-09-20 it
    holds NO file for SPY, QQQ, IWM or VXX: the home folder has no `daily_bars/`
    at all. The fallback is the machine-local daily cache the desk's own D1
    environment labels are built from (1,993 symbols), wrapped HERE so nothing
    else reaches into `d1_environment_store`'s private reader. Neither is a
    fetch; a symbol in neither is `unmeasured` and says so.
    """
    name = str(symbol or "").strip().upper()
    if not name:
        return []
    bars: list[Any] = []
    try:
        import chart_snapshot

        bars = list(chart_snapshot.load_d1_bars(name) or [])
    except Exception:  # noqa: BLE001 - a missing daily store is unmeasured
        _log.debug("The durable daily store was unreadable.", exc_info=True)
    if bars:
        return bars
    try:
        import d1_environment_store

        # The ONE place this private reader is called from. It is the desk's own
        # cached daily bars - a file, not a provider - and it is private only
        # because nobody outside that module needed it before.
        return list(d1_environment_store._cached_daily_bars(name) or [])
    except Exception:  # noqa: BLE001
        _log.debug("The machine-local daily cache was unreadable.", exc_info=True)
        return []


def session_bars_for(symbol: str, session: str) -> list[Any]:
    """One benchmark's durable M5 tape for one session (TJ-2A). THE one loader.

    Measured on the desk 2026-09-20: `day_review/bars/` holds no session at all
    yet, so this answers empty and every rest-of-day read is honestly
    `unmeasured` until the post-close tick writes a tape.
    """
    try:
        import day_review_bars

        tape = day_review_bars.read_session_bars(str(session or "")[:10]) or {}
        return list(tape.get(str(symbol or "").strip().upper()) or [])
    except Exception:  # noqa: BLE001 - a missing tape is unmeasured
        _log.debug("The durable session tape was unreadable.", exc_info=True)
        return []


def atr_for_session(symbol: str, session: str) -> float | None:
    """The benchmark's point-in-time daily ATR(14) at `session`. THE one default."""
    return daily_atr(daily_bars_for_symbol(symbol), through=session)


__all__ = [
    "BASELINES",
    "CONGRUENCE_KINDS",
    "CONGRUENCE_M5_KIND",
    "CONTEXT_UNMEASURED",
    "ContextMissingError",
    "FINAL_UNMEASURED_REASONS",
    "atr_for_session",
    "daily_bars_for_symbol",
    "read_phrase",
    "select_read",
    "session_bars_for",
    "verdict_rank",
    "D1_CHECKPOINT_SESSIONS",
    "DEFAULT_BENCHMARK",
    "FLAT_BAND_ATR",
    "FLAT_BAND_REASON",
    "FLAT_BAND_RULE",
    "GRADABLE_DIRECTIONS",
    "LABEL_DIRECTION",
    "PENDING_PREFIX",
    "PoolingError",
    "SCHEMA_GRADE",
    "SCHEMA_READ",
    "SOURCE_CLICK",
    "SOURCE_EXTRACTED",
    "UNMEASURED",
    "UNMEASURED_PREFIX",
    "VERDICT_FLAT",
    "VERDICT_RIGHT",
    "VERDICT_TOO_FEW",
    "VERDICT_WRONG",
    "accuracy",
    "append_grades",
    "baseline_reads",
    "congruence_lines",
    "context_for",
    "current_grades",
    "daily_atr",
    "grade_read",
    "is_gradable",
    "pooled_accuracy",
    "read_grades",
    "read_rows",
    "reads_path",
    "regrade_matured",
]
