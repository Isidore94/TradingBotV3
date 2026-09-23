"""The six-line report card that heads Day Review - TJ-12.

Trader, 2026-09-19: *"I want what I missed to be very apparent. I want what I
did well with to also be very apparent."*

Six deterministic lines, each with its own ``n``, each clickable to the table it
came from. **This module computes NO new statistic.** Every number on the card
is read from the function that owns it:

======================  ===================================================
line                    its owner
======================  ===================================================
``did_well``            `walkaway_day`'s liked rows, its `REAL_MISS_V1`
                        verdicts, its skill cells and their own `low`
``missed``              `walkaway_day`'s rejected rows and its own
                        `_reason_clause` sentence
``your_reads``          `prediction_ledger.your_reads`
``congruence``          `market_read_grades.congruence_lines`
``process``             `trade_origin.planned_state` / `label_provenance`
``how_fresh``           the facts the worker read, plus `ai_jobs.ledger`
======================  ===================================================

Four rules it keeps, all of them plan.md sec 5's:

* **PURE.** :func:`build` opens no store, reads no clock, calls no model and
  touches no Qt. Everything it needs arrives already read by the Day Review
  worker inside the ONE payload (TJ-1). The single exception is
  :func:`how_fresh`, which reads the AI-job ledger's TAIL through an explicit
  path - on that same worker, never on the Qt thread.
* **Missing is said, never zeroed.** A line whose input the desk did not have
  says so; ``n - measured`` never enters a rate and is never printed as a zero.
* **ONE Wilson.** A DAY line carries no interval of its own: `walkaway_day`
  already put `swing_headline`'s bound on every cell as ``low``, and a second
  interval over the same numbers is a second opinion. Only :func:`week` computes
  one, from POOLED counts - never the mean of two days' rates.
* **Nothing here is acted on.** No detector, score, gate, alert, watchlist,
  Focus list, review queue or ``review_policy.json`` can reach this module's
  output. It is read by a page and by TJ-4's pack, and that is all.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import evidence_stats
import market_read_grades as grades
import prediction_ledger
import real_miss
import trade_origin
import walkaway_day

_log = logging.getLogger(__name__)

#: The six lines, in the order the packet names them. `how_fresh` is the sixth
#: and smaller one the trader added on the second look (AMENDED 2026-09-19).
LINE_KEYS: tuple[str, ...] = (
    "did_well",
    "missed",
    "your_reads",
    "congruence",
    "process",
    "how_fresh",
)

#: What a click on each line opens. `did_well` and `missed` name
#: `walkaway_day.TABLES` entries, because the table under the line is the one
#: the count came from; the other four name sections of the Day Review page.
LINE_TARGETS: dict[str, str] = {
    "did_well": "liked_not_traded",
    "missed": "rejected",
    "your_reads": "said",
    "congruence": "congruence",
    "process": "trades",
    "how_fresh": "status",
}

#: The lines that go into TJ-4's day pack, and therefore into what the night's
#: story is allowed to cite. `how_fresh` is DELIBERATELY absent: it describes
#: the MACHINE's night rather than the trader's day, and its text moves every
#: time the job ledger gains a row - inside the pack's hashed `body` that would
#: move `inputs_hash` and buy a model call to re-narrate the same session night
#: after night. The other five are facts OF THE DAY, and when one of them really
#: moves (a D1 horizon matures) the hash SHOULD move.
PACK_LINE_KEYS: tuple[str, ...] = tuple(key for key in LINE_KEYS if key != "how_fresh")

#: How many ledger rows the freshness line may look at. The live ledger measured
#: 1,256,082 bytes on 2026-09-20, and a night writes a few dozen rows: a tail is
#: all this line can use, and a whole-file read on the Day Review worker is a
#: cost the trader pays for a sentence.
LEDGER_TAIL_ROWS = 500

#: Past this size the tail is read by SEEKING from the end of the file rather
#: than through `ai_jobs.ledger.recent_rows`, which reads the whole file and
#: then slices it. Its behaviour is unchanged for its other callers; this is the
#: bound that keeps ONE sentence from reading a megabyte on a worker.
LEDGER_TAIL_BYTES = 256 * 1024

#: The four lanes `trade_origin.planned_state` reads, in its own argument order.
#: Named here because two lane builders - the desk's Mentor card and the Day
#: Review worker - have to DECLARE which of them they opened, and two literals
#: in two files is how they come to disagree.
ORIGIN_LANES: tuple[str, ...] = ("decisions", "claims", "focus_adds", "armed")

#: What the desk can actually read TODAY. `focus_picks` has no public reader
#: that hands back a row carrying a stamp key `trade_origin` understands
#: (`_episode_started_at` is private and membership-derived) and the armed-alert
#: rows are built per caller, so both are unread until **TJ-12F**.
#:
#: This is not a detail: `planned_state` answers ``unplanned`` whenever no lane
#: row precedes the first fill, so an UNREAD lane is indistinguishable from
#: "nothing was said" - missing data read as confirmation, which plan.md sec 5
#: forbids. Measured on the live journal 2026-09-20: 30 of the trader's 33
#: trades since 2026-08-20 read ``unplanned`` for exactly this reason. So the
#: card never prints a bare "unplanned" while a lane is unread; it says what it
#: DID look at, and names what it did not.
DESK_ORIGIN_LANES_READ: tuple[str, ...] = ("decisions", "claims")

#: How each lane reads in a sentence.
_LANE_WORDS = {
    "decisions": "likes and vetoes",
    "claims": "claimed picks",
    "focus_adds": "Focus adds",
    "armed": "armed alerts",
}

#: The instrument words a trade may name itself with. An option's premium is not
#: the underlying's move (`journal_exposure`'s rule), so it is NOT judged here.
_OPTION_WORDS = frozenset({"OPT", "OPTION", "OPTIONS"})

#: What `real_miss` calls a measured answer. Anything else is uncertainty.
_MEASURED_VERDICTS = frozenset({real_miss.RUN, real_miss.NO_RUN})

_NO_STATEMENT = "the desk had nothing to read for this line"


@dataclass(frozen=True)
class ReportCard:
    """One card: six lines, in :data:`LINE_KEYS` order."""

    session: str = ""
    lines: tuple[dict[str, Any], ...] = ()
    #: The WEEK re-cut names what it pooled. A day card pools nothing.
    sessions: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# small readers - none of them measures anything
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    return str(value or "").strip()


def _line(key: str, text: str, n: int, measured: int, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": key,
        "text": text.strip(),
        "n": int(n),
        "measured": int(measured),
        "target": LINE_TARGETS[key],
    }
    row.update(extra)
    return row


def _rows_of(walkaway: Any, name: str) -> tuple[Any, ...]:
    return tuple(getattr(walkaway, name, ()) or ()) if walkaway is not None else ()


def _verdict_counts(rows: Sequence[Any]) -> tuple[int, int, int]:
    """``(n, measured, runs)`` over walk-away rows. Counting, not measuring."""
    measured = 0
    runs = 0
    for row in rows:
        verdict = _text(getattr(row, "real_miss", ""))
        if verdict in _MEASURED_VERDICTS:
            measured += 1
        if verdict == real_miss.RUN:
            runs += 1
    return len(rows), measured, runs


def _skill_window(walkaway: Any) -> Mapping[str, Any] | None:
    """TJ-11's own window for this session, or the lately one behind it."""
    skill = getattr(walkaway, "skill", None) if walkaway is not None else None
    if not isinstance(skill, Mapping):
        return None
    for name in ("session", "lately"):
        window = skill.get(name)
        if isinstance(window, Mapping):
            return window
    return None


def _skill_sentence(walkaway: Any) -> str:
    window = _skill_window(walkaway)
    return _text(window.get("sentence")) if window else ""


def _sentence(walkaway: Any, name: str) -> str:
    sentences = getattr(walkaway, "sentences", None) if walkaway is not None else None
    if isinstance(sentences, Mapping):
        return _text(sentences.get(name))
    return ""


def _family_cells(window: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """The reportable family cuts of the trader's OWN population.

    `walkaway_day` already applied `MIN_REPORTABLE_N` when it decided which
    family cuts exist at all; `reportable` is its own answer for the cell, and
    this reads it rather than re-deciding it.
    """
    if not window:
        return []
    out: list[Mapping[str, Any]] = []
    for cell in window.get("cells") or ():
        if not isinstance(cell, Mapping):
            continue
        if _text(cell.get("setup_family")) == "":
            continue
        if cell.get("population") != "liked_or_claimed":
            continue
        if not cell.get("reportable") or cell.get("low") is None:
            continue
        out.append(cell)
    return out


def _best_family(window: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The family with the best Wilson LOWER BOUND - never the best rate.

    The bound is the cell's own ``low``, computed once by `walkaway_day` from
    `swing_headline`'s single z. Nothing is recomputed here.
    """
    cells = _family_cells(window)
    if not cells:
        return None
    best = sorted(cells, key=lambda cell: (-float(cell["low"]), _text(cell["setup_family"])))[0]
    return {
        "setup_family": _text(best.get("setup_family")),
        "runs": int(best.get("runs") or 0),
        "measured": int(best.get("measured") or 0),
        "low": best.get("low"),
    }


# ---------------------------------------------------------------------------
# 1. Did well
# ---------------------------------------------------------------------------
def did_well_line(walkaway: Any) -> dict[str, Any]:
    """The likes that really ran, beside TJ-11's base rate.

    ``n`` is the ``liked_not_traded`` table and ONLY that table, because that is
    the table the line's click opens: gate #157 asks that the numbers on the
    card match the tables under them, and a count that pooled two tables while
    pointing at one could not (reviewer, 2026-09-20). Claimed D1 picks are said
    separately, with their own count and their own table.
    """
    rows = _rows_of(walkaway, "liked_not_traded")
    claimed = _rows_of(walkaway, "claimed_d1")
    n, measured, runs = _verdict_counts(rows)
    claimed_n, claimed_measured, claimed_runs = _verdict_counts(claimed)
    if walkaway is None:
        return _line(
            "did_well",
            f"Did well: {_NO_STATEMENT} - no walk-away tables were read for this session.",
            0,
            0,
            runs=0,
            claimed=0,
            claimed_measured=0,
            claimed_runs=0,
            best_family=None,
        )
    window = _skill_window(walkaway)
    best = _best_family(window)
    parts = [_sentence(walkaway, "liked_not_traded") or f"You liked {n}."]
    if claimed_n:
        # Its own count and its own table: the click on this line opens
        # `liked_not_traded`, so the headline number has to be that table's.
        parts.append(
            f"Plus {claimed_n} claimed D1 pick(s) in their own table - "
            f"{claimed_runs} real run(s) of {claimed_measured} measured."
        )
    sentence = _skill_sentence(walkaway)
    if sentence:
        parts.append(sentence)
    if best:
        parts.append(
            f"Your best family by the Wilson lower bound: {best['setup_family']}, "
            f"{best['runs']} of {best['measured']} measured (bound "
            f"{best['low']:.2f}, n {best['measured']})."
        )
    else:
        parts.append(
            "No setup family has enough measured names to name one - too few to call "
            f"(under {evidence_stats.MIN_REPORTABLE_N})."
        )
    return _line(
        "did_well",
        "Did well: " + " ".join(parts),
        n,
        measured,
        runs=runs,
        claimed=claimed_n,
        claimed_measured=claimed_measured,
        claimed_runs=claimed_runs,
        best_family=best,
    )


# ---------------------------------------------------------------------------
# 2. Missed
# ---------------------------------------------------------------------------
def missed_line(walkaway: Any) -> dict[str, Any]:
    """The rejections that really ran, and the reason they share."""
    rows = _rows_of(walkaway, "rejected")
    n, measured, runs = _verdict_counts(rows)
    if walkaway is None:
        return _line(
            "missed",
            f"Missed: {_NO_STATEMENT} - no walk-away tables were read for this session.",
            0,
            0,
            runs=0,
        )
    parts = [_sentence(walkaway, "rejected") or f"You vetoed {n}."]
    sentence = _skill_sentence(walkaway)
    if sentence:
        parts.append(sentence)
    unmeasured = n - measured
    if unmeasured:
        parts.append(
            f"{unmeasured} of the {n} could not be measured and is in neither half."
        )
    return _line("missed", "Missed: " + " ".join(parts), n, measured, runs=runs)


# ---------------------------------------------------------------------------
# 3. Your reads
# ---------------------------------------------------------------------------
def your_reads_line(tally: Mapping[str, Any] | None) -> dict[str, Any]:
    """`prediction_ledger.your_reads`' inventory and horizon cells, unchanged."""
    horizons = tally.get("horizons") if isinstance(tally, Mapping) else None
    if not isinstance(tally, Mapping) or tally.get("empty"):
        return _line(
            "your_reads",
            "Your reads: no graded reads yet for this session.",
            0,
            0,
            right=0,
            wrong=0,
            flat=0,
            pending=0,
            unmeasured=0,
            horizons=horizons if isinstance(horizons, Mapping) else {},
        )
    n = int(tally.get("n") or 0)
    text = _text(tally.get("text"))
    return _line(
        "your_reads",
        f"Your reads: {text}" if not text.lower().startswith("your reads") else text,
        n,
        n,
        right=int(tally.get("right") or 0),
        wrong=int(tally.get("wrong") or 0),
        flat=int(tally.get("flat") or 0),
        pending=int(tally.get("pending") or 0),
        unmeasured=int(tally.get("unmeasured") or 0),
        horizons=horizons if isinstance(horizons, Mapping) else {},
    )


# ---------------------------------------------------------------------------
# 4. Congruence
# ---------------------------------------------------------------------------
def congruence_line(lines: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """TJ-10's congruence lines, counted and quoted - never re-judged.

    A line the desk could not measure is NAMED as unmeasured; reading a missing
    side as agreement is the one thing `market_read_grades` refuses to do, and
    a card that summed it away would undo that refusal.
    """
    rows = [line for line in (lines or ()) if isinstance(line, Mapping)]
    if not rows:
        return _line(
            "congruence",
            f"Congruence: {_NO_STATEMENT} - no read was graded for this session.",
            0,
            0,
        )
    measured = [row for row in rows if _text(row.get("verdict")) != "unmeasured"]
    unmeasured = [row for row in rows if _text(row.get("verdict")) == "unmeasured"]
    parts = [
        f"Congruence: {len(measured)} of {len(rows)} checks measured."
    ]
    for row in measured:
        parts.append(f"{_text(row.get('kind'))}: {_text(row.get('text'))}")
    for row in unmeasured:
        missing = _text(row.get("missing"))
        parts.append(
            f"{_text(row.get('kind'))} unmeasured"
            + (f" - missing: {missing}" if missing else "")
            + f" ({_text(row.get('text'))})"
        )
    return _line("congruence", " · ".join(parts), len(rows), len(measured))


# ---------------------------------------------------------------------------
# 5. Process - and the two Mentor answers it is the named reader for
# ---------------------------------------------------------------------------
def _instrument_of(trade: Mapping[str, Any]) -> str:
    for key in ("security_type", "instrument", "asset_class"):
        text = _text(trade.get(key)).upper()
        if text:
            return text
    return ""


def _is_option(trade: Mapping[str, Any]) -> bool:
    return _instrument_of(trade) in _OPTION_WORDS


def _sessions_held(trade: Mapping[str, Any]) -> int | None:
    value = trade.get("sessions_held")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _not_judged(trade: Mapping[str, Any]) -> str:
    """Why this trade's own tape is the wrong ruler, or ``""``.

    The rule is `walkaway_day`'s (TJ-11 item 6): an option's premium is not the
    underlying's move, and a position held past `LONG_HOLD_SESSIONS` was never a
    bet on one day.
    """
    if _is_option(trade):
        return "an option's premium is not the underlying's move"
    held = _sessions_held(trade)
    if held is not None and held > walkaway_day.LONG_HOLD_SESSIONS:
        return (
            f"held {held} sessions, longer than {walkaway_day.LONG_HOLD_SESSIONS}, "
            "so one day's tape is the wrong ruler"
        )
    return ""


def _has_an_exit(trade: Mapping[str, Any]) -> bool:
    """Did this trade close any of its position? TJ-9E's denominator.

    Read off the assembled row's own `quantity_closed` rather than by opening
    the leg table: the card is built from rows a caller already has, and a
    per-trade query here would turn a six-trade session into six more reads on
    a worker. An OPEN position that nobody has touched has nothing to explain.
    """
    try:
        return float(trade.get("quantity_closed") or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def exit_note_counts(
    rows: Sequence[Mapping[str, Any]], exit_notes: Mapping[str, Any] | None
) -> tuple[int | None, int, int | None, str]:
    """``(K, N, confirmed, the clause)`` - INTEGERS ONLY, and never a zero for
    an unread lane.

    ``K`` is how many of the session's exits the trader wrote a note about and
    ``confirmed`` how many of THOSE they have since signed off, said
    separately because a draft nobody clicked is not the trader's. There is no
    rate at any count: a fraction of five exits is not a statistic about a
    trader, and printing one would be exactly the thing ground rule 10's floor
    exists to stop.
    """
    total = sum(1 for trade in rows if _has_an_exit(trade))
    if exit_notes is None:
        return (
            None,
            total,
            None,
            f"Exits explained: unmeasured - nobody opened the exit notes ({total} exit(s)).",
        )
    notes = dict(exit_notes)
    explained = [
        trade for trade in rows
        if _has_an_exit(trade)
        # WORDS. "I do not remember" is a complete ANSWER and stops the
        # question being asked, but it is not an explanation and nothing here
        # counts it as one.
        and _text((notes.get(_text(trade.get("trade_id"))) or {}).get("raw_text"))
    ]
    confirmed = sum(
        1
        for trade in explained
        if (notes.get(_text(trade.get("trade_id"))) or {}).get("exit_fields")
    )
    return (
        len(explained),
        total,
        confirmed,
        f"Exits explained {len(explained)} of {total}, "
        f"{confirmed} of those confirmed by you.",
    )


def process_line(
    trades: Sequence[Mapping[str, Any]] | None,
    *,
    origin_lanes: Mapping[str, Any] | None = None,
    lanes_read: Sequence[str] | None = None,
    mentor_answers: Sequence[Mapping[str, Any]] | None = (),
    walkaway: Any = None,
    open_positions: Sequence[Mapping[str, Any]] | None = (),
    exit_notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What the trader DID: planned or not, labelled or not, judged or not.

    Planned-vs-unplanned is `trade_origin.planned_state` and nothing else: a
    date-only broker fill is ``unmeasured``, never ``unplanned`` (a broker file
    is authoritative for money and blind to time). The label ages come from the
    row's own `label_provenance`, written by TJ-9; a row with the key PRESENT
    and EMPTY is an OLD row and counts as unlabelled, never as a third age.

    ``lanes_read`` is what the CALLER opened, declared rather than guessed - an
    empty lane and an unread one look identical from here, and the difference is
    the whole meaning of the number. While any of :data:`ORIGIN_LANES` is
    unread, the line says ``no claim or like before the fill`` and NAMES what it
    could not look at; with all four declared read it says ``unplanned`` plainly
    again. The COUNTS keep their names either way, so TJ-9's own readers see
    exactly what they always saw.

    This is also the reader `mentor_questions.REGISTRY` names for the
    ``trade_origin`` question: where the desk could not tell where a trade came
    from and ASKED, the trader's own answer is filed under that key and is read
    back here. A row with no answer is not an answer.

    ``exit_notes`` is TJ-9E's lane and follows the SAME declared-rather-than-
    guessed rule as ``lanes_read``: ``None`` means the caller did not open the
    notes, which reads ``unmeasured``, never "0 of 5". A mapping - the answer
    of `trade_mentor_trade_check.exit_notes_for_session` - gives
    ``exits explained K of N`` in INTEGERS ONLY. No rate is printed at any
    count, so no reporting floor arises: K is the exits the trader explained
    and N the exits there were, and a fraction of five is not a statistic about
    anybody. N counts trades that CLOSED something, because a position nobody
    closed has no exit to explain and counting it would say the trader is worse
    at explaining than they are.
    """
    rows = [trade for trade in (trades or ()) if isinstance(trade, Mapping)]
    answers: dict[str, str] = {}
    for row in mentor_answers or ():
        if not isinstance(row, Mapping):
            continue
        subject = _text(row.get("subject_id")) or _text(row.get("trade_id"))
        said = _text(row.get("trade_origin"))
        if subject and said:
            answers[subject] = said
    read = tuple(
        name for name in ORIGIN_LANES
        if name in (DESK_ORIGIN_LANES_READ if lanes_read is None else tuple(lanes_read))
    )
    unread = tuple(name for name in ORIGIN_LANES if name not in read)
    holds = long_hold_lines(open_positions, mentor_answers=mentor_answers)
    exits_explained, exits_n, exits_confirmed, exits_clause = exit_note_counts(
        rows, exit_notes
    )
    if not rows:
        return _line(
            "process",
            "Process: no trades on this session - nothing to judge. " + exits_clause,
            0,
            0,
            exits_explained=exits_explained,
            exits_n=exits_n,
            exits_confirmed=exits_confirmed,
            planned=0,
            unplanned=0,
            unmeasured=0,
            unlabelled=0,
            label_provenance={name: 0 for name in trade_origin.LABEL_PROVENANCES},
            not_judged=0,
            told_us=0,
            origin_answers={},
            lanes_read=read,
            lanes_unread=unread,
            long_holds=holds,
        )
    lanes = dict(origin_lanes or {})
    decisions = lanes.get("decisions") or ()
    claims = lanes.get("claims") or ()
    focus_adds = lanes.get("focus_adds") or ()
    armed = lanes.get("armed") or ()

    states = {name: 0 for name in trade_origin.PLANNED_STATES}
    provenance = {name: 0 for name in trade_origin.LABEL_PROVENANCES}
    unlabelled = 0
    not_judged: list[str] = []
    told_us = 0
    said: dict[str, int] = {}
    for trade in rows:
        state = trade_origin.planned_state(trade, decisions, claims, focus_adds, armed)
        states[state] = states.get(state, 0) + 1
        answer = answers.get(_text(trade.get("trade_id")))
        if answer:
            told_us += 1
            said[answer] = said.get(answer, 0) + 1
        age = _text(trade.get("label_provenance"))
        confirmed = _text(trade.get("tag_status")).lower() == "confirmed"
        if confirmed and age in provenance:
            provenance[age] += 1
        else:
            unlabelled += 1
        reason = _not_judged(trade)
        if reason:
            not_judged.append(f"{_text(trade.get('symbol')) or '?'}: {reason}")

    n = len(rows)
    unmeasured = states.get(trade_origin.UNMEASURED, 0)
    without = states.get(trade_origin.UNPLANNED, 0)
    if unread:
        # Never a bare "unplanned" while a lane is unread: the desk can only say
        # what it LOOKED at, and it says which doors it could not open.
        parts = [
            f"Process: {n} trade(s) - {states.get(trade_origin.PLANNED, 0)} planned "
            f"(a claim or like before the first fill), {without} with no claim or "
            f"like before the fill, {unmeasured} unmeasured (a date-only fill has "
            "no time to plan against).",
            "Read for this: "
            + " and ".join(_LANE_WORDS.get(name, name) for name in read)
            + ". "
            + " and ".join(_LANE_WORDS.get(name, name) for name in unread)
            + " are not read yet (TJ-12F), so a trade with nothing said before "
            "it is not proof there was no plan.",
        ]
    else:
        parts = [
            f"Process: {n} trade(s) - {states.get(trade_origin.PLANNED, 0)} planned, "
            f"{without} unplanned, "
            f"{unmeasured} unmeasured (a date-only fill has no time to plan against)."
        ]
    labelled = n - unlabelled
    parts.append(
        f"{labelled} labelled ("
        + ", ".join(f"{name} {count}" for name, count in provenance.items())
        + f"), {unlabelled} unlabelled."
    )
    if told_us:
        parts.append(
            f"{told_us} you told the desk where it came from ("
            + ", ".join(
                f"{name.replace('_', ' ')} {count}"
                for name, count in sorted(said.items())
            )
            + ")."
        )
    if not_judged:
        parts.append(f"{len(not_judged)} not judged here - " + "; ".join(not_judged) + ".")
    left = [
        row for row in _rows_of(walkaway, "traded_left_early")
        if getattr(row, "left_on_table_pct", None) is not None
    ]
    if left:
        parts.append(f"Left on the table was measured on {len(left)} closed position(s).")
    if holds:
        parts.append(f"{len(holds)} open position(s) held past {walkaway_day.LONG_HOLD_SESSIONS} sessions.")
    parts.append(exits_clause)
    return _line(
        "process",
        " ".join(parts),
        n,
        n - unmeasured,
        exits_explained=exits_explained,
        exits_n=exits_n,
        exits_confirmed=exits_confirmed,
        planned=states.get(trade_origin.PLANNED, 0),
        unplanned=states.get(trade_origin.UNPLANNED, 0),
        unmeasured=unmeasured,
        unlabelled=unlabelled,
        label_provenance=provenance,
        not_judged=len(not_judged),
        told_us=told_us,
        origin_answers=said,
        lanes_read=read,
        lanes_unread=unread,
        long_holds=holds,
    )


def long_hold_lines(
    open_positions: Sequence[Mapping[str, Any]] | None,
    *,
    mentor_answers: Sequence[Mapping[str, Any]] | None = (),
) -> tuple[dict[str, Any], ...]:
    """One row per OPEN position the trader was asked about, with their answer.

    The Mentor's ``open_position_check`` files the trader's answer under the key
    the registry names, and this is the reader it names. An unanswered position
    is reported as unanswered - never as "the thesis is intact", which is a
    claim nobody made.
    """
    answered: dict[str, str] = {}
    for row in mentor_answers or ():
        if not isinstance(row, Mapping):
            continue
        subject = _text(row.get("subject_id")) or _text(row.get("trade_id"))
        state = _text(row.get("open_position_state"))
        if subject and state:
            answered[subject] = state
    out: list[dict[str, Any]] = []
    for row in open_positions or ():
        if not isinstance(row, Mapping):
            continue
        if _text(row.get("status")).upper() == "CLOSED":
            continue
        held = _sessions_held(row)
        if held is None or held <= walkaway_day.LONG_HOLD_SESSIONS:
            continue
        trade_id = _text(row.get("trade_id"))
        state = answered.get(trade_id, "")
        out.append(
            {
                "trade_id": trade_id,
                "symbol": _text(row.get("symbol")),
                "sessions_held": held,
                "state": state or "unanswered",
                "answered": bool(state),
            }
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# 6. How fresh
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _LedgerTail:
    """What one bounded read of the ledger saw, AND what it could not see.

    A tail that only handed back rows let the card say "no overnight slots ran"
    about a night that simply fell out of the window: on the live 1.2 MB ledger
    the 256 KB window holds 173 of 483 rows and reaches back to 2026-09-11, so 9
    of the 15 sessions the picker offers read "none reported trouble" over 10-16
    real slots (reviewer, 2026-09-20). A reader that cannot say where its own
    sight ends cannot tell silence from absence, and plan.md sec 5 is that
    missing data is uncertainty, never confirmation.
    """

    rows: tuple[Mapping[str, Any], ...] = ()
    #: Was there more file than this read looked at?
    truncated: bool = False
    #: The oldest ``session_date`` in :attr:`rows`, or ``""``. When the read
    #: TRUNCATED, this is the edge of the card's sight - and the night it names
    #: may itself have been cut in half, so it is inside the blind spot, not
    #: outside it.
    oldest_session: str = ""


def _tail_rows(path: Any, limit: int) -> _LedgerTail:
    """The LAST ``limit`` ledger rows, without reading a megabyte for a sentence.

    Small files go through `ai_jobs.ledger.recent_rows`, the owner's own bounded
    reader, with an EXPLICIT path - its default `ledger_path()` CREATES the AI
    store, and a reader that wants to say "unknown" about a missing store may
    not make one. A big file is SEEKED from the end here rather than read whole:
    `recent_rows` reads the whole file and then slices, which is right for the
    runner and wrong for a line on a page. `ai_jobs/ledger.py` is unchanged.

    The file is opened ONCE either way. Answering "I cannot see that night" by
    opening it again would have paid the whole 1.2 MB to say so.
    """
    import ai_jobs.ledger as ledger

    from pathlib import Path

    target = Path(path)
    wanted = max(1, int(limit))
    try:
        size = target.stat().st_size
    except OSError:
        return _LedgerTail()
    if size <= LEDGER_TAIL_BYTES:
        rows = list(ledger.recent_rows(wanted, path=target))
        # `recent_rows` slices to the last `wanted` and never says how many it
        # dropped, so a full window is read as "there may be more" - the
        # conservative half, which is the honest half here.
        return _tail_of(rows, truncated=len(rows) >= wanted)
    rows = []
    try:
        with open(target, "rb") as handle:
            handle.seek(size - LEDGER_TAIL_BYTES)
            # The first line of the window is almost certainly cut in half, so
            # it is dropped rather than half-parsed.
            handle.readline()
            for raw in handle.read().splitlines():
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return _LedgerTail()
    # Bytes were left behind by definition, and the row slice may drop more.
    return _tail_of(rows[-wanted:], truncated=True)


def _tail_of(rows: Sequence[Mapping[str, Any]], *, truncated: bool) -> _LedgerTail:
    oldest = ""
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        session = _text(row.get("session_date"))[:10]
        if len(session) == 10 and (not oldest or session < oldest):
            oldest = session
    return _LedgerTail(tuple(rows), truncated, oldest)


def _slot_verdicts(rows: Sequence[Mapping[str, Any]], session: str) -> dict[str, str]:
    """``{job: the status that DECIDED it}`` for one night. The owner's words.

    `ai_jobs/ledger.py` owns this vocabulary and this reads its constants rather
    than spelling any of them:

    * ``STATUS_OK`` is the only completion (``CANONICAL_COMPLETION_STATUSES``);
    * ``ATTEMPT_STATUSES`` - failed and degraded - is the owner's own "something
      went wrong" set, and the two are reported SEPARATELY: a degraded run
      published a real document with no narrative, and calling that "nothing
      ran" is a different fact;
    * ``STATUS_SKIPPED``, ``STATUS_MANUAL`` and ``STATUS_CORRECTION`` DECIDE
      NOTHING. A skip is what the runner writes when the window or the
      already-done check says there is nothing to do, and it is written every
      half hour for the rest of the night. Reading one as a failure named 22 of
      2026-09-18's slots broken when 20 of them had finished `ok` hours earlier
      (reviewer, 2026-09-20, on a copy of the live 1.2 MB ledger).

    The LAST deciding row wins: a slot that failed and then recovered is fine,
    and one that ran and then failed is not - a later skip cannot undo either.
    """
    import ai_jobs.ledger as ledger

    deciding = frozenset({ledger.STATUS_OK}) | frozenset(ledger.ATTEMPT_STATUSES)
    verdicts: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("noncanonical"):
            # A correction is commentary about a run, not a run.
            continue
        if session and _text(row.get("session_date"))[:10] != session:
            continue
        job = _text(row.get("job"))
        if not job:
            continue
        status = _text(row.get("status"))
        # Every slot the night TOUCHED is counted, even when nothing it wrote
        # decides anything: `n` is what was looked at, not what went wrong.
        verdicts.setdefault(job, "")
        if status in deciding:
            verdicts[job] = status
    return verdicts


def how_fresh(freshness: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the card rests on: when the story was written, how far the fills
    reach, which session the reads were graded through, and any overnight slot
    the ledger says went wrong - by name, and by WHICH way it went wrong.

    A missing AI store is ``night status unknown`` and CREATES NOTHING. Unknown
    is not "fine": a night nobody can see is a night nobody checked.
    """
    import ai_jobs.ledger as ledger

    facts = dict(freshness or {})
    session = _text(facts.get("session"))
    story = _text(facts.get("story_written_at"))
    fills = _text(facts.get("fills_current_to"))
    graded = _text(facts.get("reads_graded_through"))

    parts = [
        f"How fresh: story written {story}" if story else "How fresh: no story written yet",
        f"fills verified to {fills}" if fills else "fills: no verified coverage yet",
        f"reads graded through {graded}" if graded else "reads: nothing graded yet",
    ]

    path = facts.get("ledger_path")
    verdicts: dict[str, str] = {}
    night_status = "unknown"
    beyond_the_tail = False
    if path is not None:
        from pathlib import Path

        target = Path(path)
        # `exists` never creates; `ledger_path()` would have made the folder.
        if target.exists():
            tail = _tail_rows(target, LEDGER_TAIL_ROWS)
            # A session the window could not reach is UNKNOWN, never a quiet
            # night. The oldest session the tail returned is itself inside the
            # blind spot - the window may have cut that night in half, which is
            # exactly what 2026-09-11 did (5 slots shown, 16 real).
            beyond_the_tail = tail.truncated and (
                not tail.oldest_session or (bool(session) and session <= tail.oldest_session)
            )
            if not beyond_the_tail:
                verdicts = _slot_verdicts(tail.rows, session)
                night_status = "read" if verdicts else "no_rows"
    ok = tuple(sorted(job for job, status in verdicts.items() if status == ledger.STATUS_OK))
    failed = tuple(
        sorted(job for job, status in verdicts.items() if status == ledger.STATUS_FAILED)
    )
    degraded = tuple(
        sorted(job for job, status in verdicts.items() if status == ledger.STATUS_DEGRADED)
    )
    named = tuple(sorted(failed + degraded))
    if beyond_the_tail:
        # No counts at all: a number here would be a measurement of the window,
        # not of the night.
        parts.append(
            "night status unknown for this session (older than the ledger tail)"
        )
    elif night_status == "unknown":
        parts.append("night status unknown - the desk has no AI job ledger to read")
    elif night_status == "no_rows":
        # Covered by the window and genuinely empty. Saying "0 read, none
        # reported trouble" would read as a clean night, which is a claim.
        parts.append("no overnight rows for this session")
    elif named:
        trouble = []
        if failed:
            trouble.append("failed: " + ", ".join(failed))
        if degraded:
            trouble.append("degraded: " + ", ".join(degraded))
        parts.append(
            f"{len(verdicts)} overnight slot(s) read, {len(ok)} finished ok; "
            + "; ".join(trouble)
        )
    else:
        parts.append(
            f"{len(verdicts)} overnight slot(s) read, {len(ok)} finished ok, "
            "none reported trouble"
        )
    return _line(
        "how_fresh",
        "; ".join(parts) + ".",
        len(verdicts),
        len(verdicts),
        failed_slots=named,
        slots_failed=failed,
        slots_degraded=degraded,
        slots_ok=len(ok),
        night_status=night_status,
    )


# ---------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------
def _unreadable_line(key: str, exc: BaseException) -> dict[str, Any]:
    """One line that could not be built, SAYING so - and costing only itself.

    A guard around the WHOLE card loses six sentences and leaves the page
    showing six placeholders with nothing saying anything failed (reviewer,
    2026-09-20): the quiet-lie shape `How fresh` exists to prevent. ``measured``
    stays an integer count, as it is on the other five lines, and
    ``measured_ok`` is what says this line measured nothing at all.
    """
    reason = f"{type(exc).__name__}: {exc}"
    return _line(
        key,
        f"{key.replace('_', ' ').capitalize()}: could not be read: {reason[:160]}",
        0,
        0,
        measured_ok=False,
        unreadable=reason,
    )


def _guarded(key: str, builder, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return builder(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - one owner never costs five lines
        _log.debug("The %s report-card line could not be built.", key, exc_info=True)
        return _unreadable_line(key, exc)


def build(day_inputs: Mapping[str, Any]) -> ReportCard:
    """The six lines for ONE session, from what the worker already read.

    PURE apart from :func:`how_fresh`'s bounded ledger tail: every other input
    is a value the Day Review worker opened once, inside the ONE payload.

    Every line is guarded on ITS OWN: an owner that raises costs its own
    sentence and says so, and the other five still say what they measured.
    """
    inputs = dict(day_inputs or {})
    walkaway = inputs.get("walkaway")
    lines = (
        _guarded("did_well", did_well_line, walkaway),
        _guarded("missed", missed_line, walkaway),
        _guarded("your_reads", your_reads_line, inputs.get("your_reads")),
        _guarded("congruence", congruence_line, inputs.get("congruence")),
        _guarded(
            "process",
            process_line,
            inputs.get("trades") or (),
            origin_lanes=inputs.get("origin_lanes"),
            lanes_read=inputs.get("origin_lanes_read"),
            mentor_answers=inputs.get("mentor_answers") or (),
            walkaway=walkaway,
            open_positions=inputs.get("open_positions") or (),
            # TJ-9E. `None` - the default - is "nobody opened the exit notes",
            # which the line SAYS; a mapping is the read the Day Review worker
            # already made. Never `or {}`: an empty mapping is "the desk looked
            # and the trader has explained none of them", and the two are
            # different facts (review 1 blocker 3).
            exit_notes=inputs.get("exit_notes"),
        ),
        _guarded("how_fresh", how_fresh, inputs.get("freshness")),
    )
    return ReportCard(session=_text(inputs.get("session"))[:10], lines=lines)


def pack_card(card: Any) -> ReportCard | None:
    """The card as TJ-4's day pack may hold it, or ``None`` when there is none.

    The ONE seam for :data:`PACK_LINE_KEYS`. `day_review_pack.build_pack` is
    deliberately NOT where this lives: it carries whatever it is handed, so a
    caller with a whole card still gets a whole card, and the decision about
    what the NIGHT may cite is made here, once, by the module that owns what a
    line is.

    Takes a :class:`ReportCard` or the plain mapping the Day Review payload
    carries one as. A line the desk could not build goes in AS WHAT IT IS - a
    night that narrated around a hole would be telling the trader about a day
    the desk never measured.
    """
    lines = getattr(card, "lines", None)
    session = _text(getattr(card, "session", ""))
    if lines is None and isinstance(card, Mapping):
        lines = card.get("lines")
        session = _text(card.get("session"))
    kept = tuple(
        dict(line)
        for line in (lines or ())
        if isinstance(line, Mapping) and _text(line.get("key")) in PACK_LINE_KEYS
    )
    if not kept:
        return None
    return ReportCard(session=session, lines=kept)


# ---------------------------------------------------------------------------
# the week re-cut (TJ-5's strip reads this)
# ---------------------------------------------------------------------------
def _wilson(hits: int, total: int) -> float | None:
    """The ONE Wilson lower bound (`swing_headline`, z 1.96), pooled.

    Imported HERE and not at module scope so the DAY path can be proven never to
    reach it: a Wilson over one session would be a statistic the card invented.
    """
    import swing_headline

    return swing_headline.wilson_lower_bound(hits, total) if total > 0 else None


def _card_lines(card: Any) -> tuple[Mapping[str, Any], ...]:
    """One card's lines, whether it is a `ReportCard` or the mapping one stores as.

    TJ-5's week strip pools the lines a day PACK stored (`day_review_pack`'s
    `report_card` section), which are plain dicts by the time they come off
    disk. A pooling function that only understood the dataclass would need the
    caller to rebuild one, and a rebuilt card is a second opinion about what a
    line said.
    """
    lines = getattr(card, "lines", None)
    if lines is None and isinstance(card, Mapping):
        lines = card.get("lines")
    return tuple(line for line in (lines or ()) if isinstance(line, Mapping))


def _card_session(card: Any) -> str:
    session = getattr(card, "session", None)
    if session is None and isinstance(card, Mapping):
        session = card.get("session")
    return _text(session)[:10]


def _pooled(cards: Sequence[Any], key: str) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for card in cards:
        for line in _card_lines(card):
            if line.get("key") == key:
                out.append(line)
    return out


def _rate_keys(hits: int, total: int) -> dict[str, Any]:
    """``rate``/``rate_lb``/``meets_floor`` from POOLED counts, never averaged."""
    return {
        "rate": (hits / total) if total else None,
        "rate_lb": _wilson(hits, total),
        "meets_floor": total >= evidence_stats.MIN_REPORTABLE_N,
    }


def _read_horizons(line: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The new per-horizon read shape, or ``None`` for a stored old card."""
    horizons = line.get("horizons")
    return horizons if isinstance(horizons, Mapping) else None


def _horizon_accuracy(cell: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Accept a Day cell or a pooled cell without deriving a new outcome."""
    if not isinstance(cell, Mapping):
        return {}
    accuracy = cell.get("accuracy")
    return accuracy if isinstance(accuracy, Mapping) else cell


def _pooled_horizon_cell(lines: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    """Pool one named horizon only; scalar coverage never reaches this path."""
    cells = [
        _horizon_accuracy((_read_horizons(line) or {}).get(name))
        for line in lines
        if isinstance((_read_horizons(line) or {}).get(name), Mapping)
    ]

    def total(field: str) -> int:
        return sum(int(cell.get(field) or 0) for cell in cells)

    accuracy = {
        "right": total("right"),
        "wrong": total("wrong"),
        "flat": total("flat"),
        "pending": total("pending"),
        "unmeasured": total("unmeasured"),
        "n": total("n"),
    }
    accuracy.update(_rate_keys(accuracy["right"], accuracy["n"]))
    baselines: dict[str, dict[str, Any]] = {}
    for rule in grades.BASELINES:
        baseline_cells = [
            (dict(((_read_horizons(line) or {}).get(name) or {}).get("baselines") or {}).get(rule) or {})
            for line in lines
            if isinstance((_read_horizons(line) or {}).get(name), Mapping)
        ]
        baseline = {
            "right": sum(int(cell.get("right") or 0) for cell in baseline_cells),
            "wrong": sum(int(cell.get("wrong") or 0) for cell in baseline_cells),
            "flat": sum(int(cell.get("flat") or 0) for cell in baseline_cells),
            "pending": sum(int(cell.get("pending") or 0) for cell in baseline_cells),
            "unmeasured": sum(int(cell.get("unmeasured") or 0) for cell in baseline_cells),
            "n": sum(int(cell.get("n") or 0) for cell in baseline_cells),
            "baseline": rule,
            "label": prediction_ledger.BASELINE_LABELS.get(rule, rule),
        }
        baseline.update(_rate_keys(baseline["right"], baseline["n"]))
        baselines[rule] = baseline
    return {
        "horizon": name,
        "label": prediction_ledger.HORIZON_LABELS.get(name, name),
        **accuracy,
        "accuracy": dict(accuracy),
        "baselines": baselines,
    }


def _pooled_read_horizons(lines: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every known horizon stays a separate evidence population."""
    return {
        name: _pooled_horizon_cell(lines, name)
        for name in prediction_ledger.HORIZONS
    }


def _unseparated_read_sessions(cards: Sequence[Any]) -> tuple[str, ...]:
    """Name old scalar cards instead of inventing a horizon for their counts."""
    sessions: list[str] = []
    for card in cards:
        session = _card_session(card)
        for line in _card_lines(card):
            if (
                line.get("key") == "your_reads"
                and line.get("measured_ok") is not False
                and _read_horizons(line) is None
                and session
                and session not in sessions
            ):
                sessions.append(session)
    return tuple(sessions)


def _horizon_coverage_text(horizons: Mapping[str, Mapping[str, Any]]) -> str:
    """A count-only summary. Rates, where useful, live in each named cell."""
    return " · ".join(
        f"{cell['label']}: {cell['n']} finished, {cell['right']} right, "
        f"{cell['wrong']} wrong, {cell['flat']} flat, {cell['pending']} waiting, "
        f"{cell['unmeasured']} unmeasured"
        for cell in (horizons.get(name) or {} for name in prediction_ledger.HORIZONS)
        if cell
    )


def _floor_clause(total: int) -> str:
    if total >= evidence_stats.MIN_REPORTABLE_N:
        return ""
    return f" - too few to call (n {total}, under {evidence_stats.MIN_REPORTABLE_N})"


def _pooled_families(days: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The best family over the WHOLE window, from summed runs and measured."""
    totals: dict[str, list[int]] = {}
    for day in days:
        window = _skill_window(day.get("walkaway"))
        for cell in _family_cells(window):
            name = _text(cell.get("setup_family"))
            bucket = totals.setdefault(name, [0, 0])
            bucket[0] += int(cell.get("runs") or 0)
            bucket[1] += int(cell.get("measured") or 0)
    candidates = []
    for name, (runs, measured) in totals.items():
        if measured < evidence_stats.MIN_REPORTABLE_N:
            continue
        low = _wilson(runs, measured)
        if low is None:
            continue
        candidates.append({"setup_family": name, "runs": runs, "measured": measured, "low": low})
    if not candidates:
        return None
    return sorted(candidates, key=lambda cell: (-cell["low"], cell["setup_family"]))[0]


#: What a week pooled from STORED cards says where `week()` names a best family.
#:
#: A stored line carries that DAY's best family and its own two integers; five
#: day-winners are five different questions answered once each, and picking the
#: best of them is a ranking of days, not of families (lead decision, TJ-5,
#: 2026-09-20). The per-family cells live on TJ-11's walk-away object, which a
#: pack does not carry, so a week built from cards says what it could not do.
NO_FAMILY_FROM_CARDS = (
    " No best setup family over this window: pooling families needs the walk-away "
    "skill cells, and a stored card carries only that day's own winner - a best of "
    "five day-winners would be a ranking of days, not of families."
)


def week(sessions: Sequence[Mapping[str, Any]]) -> ReportCard:
    """The same six lines over a LIST of sessions - TJ-5's week / month strip.

    Same functions, longer window, ``n`` everywhere. A week POOLS counts and
    computes the ONE Wilson from the pooled pair; it never averages two days'
    rates, which on days of unequal length is a different number.

    This is the path for a caller that still has TJ-11's `WalkawayDay` objects,
    so it can pool the family cells too. A caller holding only the BUILT lines -
    TJ-5's week strip, reading them back out of the day packs - uses
    :func:`week_from_cards`, which is the same pooling without that half.
    """
    # ONE row per session: a window handed the same day twice must not pool it
    # twice, or a week's `n` is a fact about the caller rather than the trader
    # (reviewer, 2026-09-20). First occurrence wins, order kept.
    days: list[dict[str, Any]] = []
    seen: set[str] = set()
    for day in sessions or ():
        if not isinstance(day, Mapping):
            continue
        name = _text(day.get("session"))[:10]
        if name and name in seen:
            continue
        if name:
            seen.add(name)
        days.append(dict(day))
    return _pool_cards(
        [build(day) for day in days],
        sessions=tuple(_text(day.get("session"))[:10] for day in days),
        best_family=_pooled_families(days),
        no_family_clause=(
            " No setup family has enough measured names over this window - too few "
            "to call."
        ),
    )


def week_from_cards(cards: Sequence[Any]) -> ReportCard:
    """The same pooling, over cards that were BUILT once and stored (TJ-5).

    TJ-5's week strip reads TJ-12's lines back out of the day packs: the pack
    carries the built line with all of its integers, and rebuilding a card from
    a pack is impossible anyway (`build` needs TJ-11's `WalkawayDay` object,
    which a pack does not hold). The arithmetic lives HERE rather than in the
    page's service so there is ONE place that knows what pooling a line means -
    the module that owns what a line is (lead decision, 2026-09-20).

    Takes `ReportCard`s or the mappings a pack stores them as. Duplicate
    sessions are pooled once, first occurrence winning, exactly as :func:`week`
    does. It names NO best family and says why: see :data:`NO_FAMILY_FROM_CARDS`.
    """
    kept: list[Any] = []
    names: list[str] = []
    seen: set[str] = set()
    for card in cards or ():
        name = _card_session(card)
        if name and name in seen:
            continue
        if name:
            seen.add(name)
        kept.append(card)
        names.append(name)
    return _pool_cards(
        kept,
        sessions=tuple(names),
        best_family=None,
        no_family_clause=NO_FAMILY_FROM_CARDS,
    )


def _pool_cards(
    cards: Sequence[Any],
    *,
    sessions: Sequence[str],
    best_family: Mapping[str, Any] | None,
    no_family_clause: str,
) -> ReportCard:
    """Pool the six lines of several cards into one. Counting and ONE Wilson.

    Every number here is a SUM of integers the day lines already carried, and
    the only statistic is `_rate_keys`' pooled Wilson. Nothing is re-measured
    and nothing is ranked.

    A day whose line the desk could NOT build contributes nothing to any count -
    it never did - but it is NAMED on the pooled line (``unreadable_sessions``,
    and a clause in the text). Dropping the marker made an unreadable day
    indistinguishable from a quiet one, which is the quiet-lie shape
    `_unreadable_line` exists to prevent (reviewer advisory 2, 2026-09-20).
    """
    names = tuple(sessions)
    count = len(names)
    label = f"over {count} session(s)"
    unreadable = _unreadable_sessions(cards)

    def _sum(key: str, field_name: str) -> int:
        return sum(int(line.get(field_name) or 0) for line in _pooled(cards, key))

    lines: list[dict[str, Any]] = []

    # -- Did well ----------------------------------------------------------
    n = _sum("did_well", "n")
    measured = _sum("did_well", "measured")
    runs = _sum("did_well", "runs")
    best = dict(best_family) if best_family else None
    text = (
        f"Did well {label}: {runs} real run(s) of {measured} measured, {n} considered"
        + _floor_clause(measured)
        + "."
    )
    if best:
        text += (
            f" Best family by the Wilson lower bound: {best['setup_family']}, "
            f"{best['runs']} of {best['measured']} measured (bound {best['low']:.2f})."
        )
    else:
        text += no_family_clause
    lines.append(
        _line("did_well", text, n, measured, runs=runs, best_family=best, **_rate_keys(runs, measured))
    )

    # -- Missed ------------------------------------------------------------
    n = _sum("missed", "n")
    measured = _sum("missed", "measured")
    runs = _sum("missed", "runs")
    lines.append(
        _line(
            "missed",
            f"Missed {label}: {runs} real miss(es) of {measured} measured, {n} rejected"
            + _floor_clause(measured)
            + ".",
            n,
            measured,
            runs=runs,
            **_rate_keys(runs, measured),
        )
    )

    # -- Your reads --------------------------------------------------------
    n = _sum("your_reads", "n")
    measured = _sum("your_reads", "measured")
    right = _sum("your_reads", "right")
    wrong = _sum("your_reads", "wrong")
    flat = _sum("your_reads", "flat")
    pending = _sum("your_reads", "pending")
    unmeasured = _sum("your_reads", "unmeasured")
    read_lines = _pooled(cards, "your_reads")
    horizons = _pooled_read_horizons(read_lines)
    unseparated = _unseparated_read_sessions(cards)
    if n or pending or unmeasured:
        text = (
            f"Your reads {label}: coverage {n} finished, {right} right, "
            f"{wrong} wrong, {flat} flat, {pending} waiting, {unmeasured} unmeasured. "
            + _horizon_coverage_text(horizons)
        )
    else:
        text = f"Your reads {label}: no graded reads yet. " + _horizon_coverage_text(horizons)
    if unseparated:
        text += (
            f" {len(unseparated)} historical session(s) have unseparated read horizons: "
            + ", ".join(unseparated)
            + "."
        )
    lines.append(
        _line(
            "your_reads",
            text,
            n,
            measured,
            right=right,
            wrong=wrong,
            flat=flat,
            pending=pending,
            unmeasured=unmeasured,
            horizons=horizons,
            unseparated_sessions=unseparated,
        )
    )

    # -- Congruence --------------------------------------------------------
    n = _sum("congruence", "n")
    measured = _sum("congruence", "measured")
    lines.append(
        _line(
            "congruence",
            f"Congruence {label}: {measured} of {n} checks measured."
            if n
            else f"Congruence {label}: no read was graded.",
            n,
            measured,
        )
    )

    # -- Process -----------------------------------------------------------
    n = _sum("process", "n")
    planned = _sum("process", "planned")
    unplanned = _sum("process", "unplanned")
    unmeasured = _sum("process", "unmeasured")
    unlabelled = _sum("process", "unlabelled")
    provenance = {name: 0 for name in trade_origin.LABEL_PROVENANCES}
    for line in _pooled(cards, "process"):
        for name, value in (line.get("label_provenance") or {}).items():
            if name in provenance:
                provenance[name] += int(value or 0)
    text = (
        f"Process {label}: {n} trade(s) - {planned} planned, {unplanned} unplanned, "
        f"{unmeasured} unmeasured; {unlabelled} unlabelled."
        if n
        else f"Process {label}: no trades."
    )
    lines.append(
        _line(
            "process",
            text,
            n,
            n - unmeasured,
            planned=planned,
            unplanned=unplanned,
            unmeasured=unmeasured,
            unlabelled=unlabelled,
            label_provenance=provenance,
            **_rate_keys(planned, n - unmeasured),
        )
    )

    # -- How fresh ---------------------------------------------------------
    fresh = _pooled(cards, "how_fresh")
    failed: list[str] = []
    for line in fresh:
        for job in line.get("failed_slots") or ():
            if job not in failed:
                failed.append(job)
    slots = _sum("how_fresh", "n")
    night_status = "read" if any(line.get("night_status") == "read" for line in fresh) else "unknown"
    if night_status == "unknown":
        text = f"How fresh {label}: night status unknown - no AI job ledger was read."
    elif failed:
        text = (
            f"How fresh {label}: {slots} overnight slot run(s) read; these did not "
            "finish ok: " + ", ".join(sorted(failed)) + "."
        )
    else:
        text = f"How fresh {label}: {slots} overnight slot run(s) read, every one finished ok."
    lines.append(
        _line(
            "how_fresh",
            text,
            slots,
            slots,
            failed_slots=tuple(sorted(failed)),
            night_status=night_status,
        )
    )

    for line in lines:
        named = unreadable.get(line["key"], ())
        # PRESENT and EMPTY when every day was readable: a reader has one shape
        # either way, and an absent key would be a third state to handle.
        line["unreadable_sessions"] = named
        if named:
            line["text"] = (
                line["text"].rstrip()
                + f" {len(named)} day(s) could not be read: "
                + ", ".join(named)
                + "."
            )

    return ReportCard(
        session=names[-1] if names else "",
        lines=tuple(lines),
        sessions=names,
    )


def _unreadable_sessions(cards: Sequence[Any]) -> dict[str, tuple[str, ...]]:
    """``{line key: the sessions whose line the desk could not build}``.

    `_unreadable_line` marks its line ``measured_ok: False``; every other line
    leaves the key absent. A card with no session of its own cannot be named,
    so it is counted into the marker only when it has one - and the names are
    de-duplicated and ordered, because two readers of the same window must read
    the same sentence.
    """
    found: dict[str, list[str]] = {}
    for card in cards or ():
        session = _card_session(card)
        for line in _card_lines(card):
            if line.get("measured_ok") is not False:
                continue
            key = _text(line.get("key"))
            if not key or not session:
                continue
            bucket = found.setdefault(key, [])
            if session not in bucket:
                bucket.append(session)
    return {key: tuple(sorted(names)) for key, names in found.items()}


# ---------------------------------------------------------------------------
# display helpers for the Day Review page (Day Recap step A)
# ---------------------------------------------------------------------------

#: Internal names the trader should never read, and the words that replace them.
_PLAIN_NAMES: tuple[tuple[str, str], ...] = (
    (grades.CONGRUENCE_M5_KIND, "your M5 picks' sides"),
    ("picks_side_mix", "your D1 picks' sides"),
    ("desk_d1_label", "the desk's D1 label"),
    ("fills_bias", "your fills"),
    ("claimed_before_entry", "claimed before entry"),
    ("same_session", "same session"),
    ("recalled_after", "recalled after"),
    ("liked_or_claimed", "liked or claimed"),
)
#: Packet names like "(TJ-12F)" or "TJ-2 brings ...", with any brackets.
_PACKET = re.compile(r"\s*\(?\bTJ-\d+[A-Z]*\b\)?")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def _all_zero(part: str) -> bool:
    """True when a clause holds numbers and every one of them is zero."""
    numbers = _NUMBER.findall(part)
    return bool(numbers) and all(float(number) == 0 for number in numbers)


def _split_top(text: str, separators: tuple[str, ...]) -> list[str]:
    """Split on `separators` outside parentheses, keeping each separator."""
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            for sep in separators:
                if text.startswith(sep, index):
                    parts.append(text[start:index])
                    parts.append(sep)
                    index += len(sep)
                    start = index
                    break
            else:
                index += 1
                continue
            continue
        index += 1
    parts.append(text[start:])
    return parts


def _drop_zero_clauses(text: str) -> str:
    """Remove comma/semicolon clauses and sentences that are only zeros."""
    sentences = _split_top(text, (". ", " · "))
    kept_sentences: list[str] = []
    for index in range(0, len(sentences), 2):
        body = sentences[index]
        sep = sentences[index + 1] if index + 1 < len(sentences) else ""
        pieces = _split_top(body, (", ", "; "))
        head = ""
        first = pieces[0]
        # "Label: 0 x, 3 y" - the label is kept with whatever survives.
        if ": " in first and not _all_zero(first.split(": ", 1)[0]):
            head, pieces[0] = first.split(": ", 1)[0] + ": ", first.split(": ", 1)[1]
        clauses = [pieces[i] for i in range(0, len(pieces), 2)]
        joins = [pieces[i] for i in range(1, len(pieces), 2)]
        survivors = [
            (clause, joins[i - 1] if i else "")
            for i, clause in enumerate(clauses)
            if not _all_zero(clause)
        ]
        if not survivors:
            continue
        rebuilt = survivors[0][0]
        for clause, join in survivors[1:]:
            rebuilt += (join or ", ") + clause
        if body.rstrip().endswith(".") and not rebuilt.rstrip().endswith("."):
            rebuilt = rebuilt.rstrip() + "."
        kept_sentences.append(head + rebuilt + sep)
    out = "".join(kept_sentences).strip()
    for sep in (" ·", "·"):
        if out.endswith(sep):
            out = out[: -len(sep)].rstrip()
    return out


def plain_words(text: Any) -> str:
    """A report-card or congruence sentence as the trader should read it.

    Display only - the stored card and the night's pack keep their words.
    Internal ids become words, packet names go, clauses that are all zeros or
    "0 of 0" are hidden, and "unmeasured" reads "not measured" (unknown is
    never shown as zero).
    """
    out = str(text or "")
    for name, words in _PLAIN_NAMES:
        out = re.sub(rf"\b{re.escape(name)}\b", words, out)
    out = _PACKET.sub("", out)
    out = re.sub(r"\bunmeasured\b", "not measured", out)
    # The line's own label ("Did well:") survives even when its first
    # sentence is all zeros.
    label, _sep, rest = out.partition(": ")
    if rest and len(label) <= 24 and not re.search(r"\d", label):
        out = f"{label}: {_drop_zero_clauses(rest) or 'nothing to report.'}"
    else:
        out = _drop_zero_clauses(out)
    # A leading "Plus" whose first half was hidden reads oddly on its own.
    out = re.sub(r"(^|: )Plus (\d)", r"\1\2", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _money(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _card_line(card: Any, key: str) -> Mapping[str, Any]:
    rows = card.get("lines") if isinstance(card, Mapping) else getattr(card, "lines", ())
    for row in rows or ():
        if isinstance(row, Mapping) and row.get("key") == key:
            return row
    return {}


def glance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The numbers at the top of Day Review, from ONE payload. Pure.

    Every value is either measured or ``None`` ("not measured"); nothing
    unknown is shown as a zero.
    """
    trades = [row for row in (payload.get("trades") or ()) if isinstance(row, Mapping)]
    nets = [(row, _money(row.get("net_pnl"))) for row in trades]
    known = [(row, net) for row, net in nets if net is not None]
    pnl = sum(net for _row, net in known) if known else None
    risks = [_money(row.get("planned_risk")) for row, _net in known]
    r_value = (
        sum(net / risk for (_row, net), risk in zip(known, risks))
        if known and all(risk and risk > 0 for risk in risks)
        else None
    )
    wins = sum(1 for _row, net in known if net > 0)
    losses = sum(1 for _row, net in known if net < 0)
    best = max(known, key=lambda pair: pair[1], default=None)
    biggest_win = (
        {"symbol": str(best[0].get("symbol") or ""), "net_pnl": best[1],
         "trade_id": str(best[0].get("trade_id") or "")}
        if best and best[1] > 0 else None
    )

    process = _card_line(payload.get("report_card"), "process")
    reads = _card_line(payload.get("report_card"), "your_reads")
    planned = None
    if trades and process:
        planned = {
            "planned": int(process.get("planned") or 0),
            "unplanned": int(process.get("unplanned") or 0),
            "unmeasured": int(process.get("unmeasured") or 0),
            "lanes_unread": tuple(process.get("lanes_unread") or ()),
        }
    calls = None
    if reads and int(reads.get("n") or 0):
        calls = {name: int(reads.get(name) or 0) for name in ("right", "wrong", "flat", "pending")}

    miss = None
    walkaway = payload.get("walkaway")
    for population in ("rejected", "liked_not_traded"):
        for row in _rows_of(walkaway, population):
            if _text(getattr(row, "real_miss", "")) != real_miss.RUN:
                continue
            moved = getattr(row, "ran_after_pct", None)
            if moved is None:
                continue
            if miss is None or float(moved) > miss["ran_after_pct"]:
                miss = {
                    "symbol": _text(getattr(row, "symbol", "")),
                    "ran_after_pct": float(moved),
                    "population": population,
                }
    day_type = _text(payload.get("day_type"))
    return {
        "trades": len(trades),
        "pnl": pnl,
        "pnl_counted": len(known),
        "r": r_value,
        "wins": wins,
        "losses": losses,
        "planned": planned,
        "calls": calls,
        "day_type": day_type or None,
        "biggest_win": biggest_win,
        "biggest_miss": miss,
        "pnl_by_session": tuple(payload.get("pnl_by_session") or ()),
    }


__all__ = [
    "DESK_ORIGIN_LANES_READ",
    "LEDGER_TAIL_BYTES",
    "LEDGER_TAIL_ROWS",
    "LINE_KEYS",
    "LINE_TARGETS",
    "NO_FAMILY_FROM_CARDS",
    "ORIGIN_LANES",
    "PACK_LINE_KEYS",
    "ReportCard",
    "build",
    "glance",
    "plain_words",
    "congruence_line",
    "did_well_line",
    "how_fresh",
    "long_hold_lines",
    "missed_line",
    "pack_card",
    "process_line",
    "week",
    "week_from_cards",
    "your_reads_line",
]
