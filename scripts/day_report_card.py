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
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import evidence_stats
import real_miss
import trade_origin
import walkaway_day

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
    """The likes and claims that really ran, beside TJ-11's base rate."""
    rows = _rows_of(walkaway, "liked_not_traded") + _rows_of(walkaway, "claimed_d1")
    n, measured, runs = _verdict_counts(rows)
    if walkaway is None:
        return _line(
            "did_well",
            f"Did well: {_NO_STATEMENT} - no walk-away tables were read for this session.",
            0,
            0,
            runs=0,
            best_family=None,
        )
    window = _skill_window(walkaway)
    best = _best_family(window)
    parts = [_sentence(walkaway, "liked_not_traded") or f"You liked {n}."]
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
    """`prediction_ledger.your_reads`' own integers, printed as it wrote them.

    The owner names the baseline with the most ``right``. More right answers on
    a bigger base is NOT a win, so this line quotes both integers with their own
    ``n`` and never calls anything a victory - the trader was 3 of 4 on the day
    the best baseline was 2 of 2.
    """
    if not isinstance(tally, Mapping) or tally.get("empty"):
        return _line(
            "your_reads",
            "Your reads: no graded reads yet for this session.",
            0,
            0,
            right=0,
        )
    n = int(tally.get("n") or 0)
    pending = int(tally.get("pending") or 0)
    text = _text(tally.get("text"))
    return _line(
        "your_reads",
        f"Your reads: {text}" if not text.lower().startswith("your reads") else text,
        n,
        max(0, n - pending),
        right=int(tally.get("right") or 0),
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


def process_line(
    trades: Sequence[Mapping[str, Any]] | None,
    *,
    origin_lanes: Mapping[str, Any] | None = None,
    mentor_answers: Sequence[Mapping[str, Any]] | None = (),
    walkaway: Any = None,
    open_positions: Sequence[Mapping[str, Any]] | None = (),
) -> dict[str, Any]:
    """What the trader DID: planned or not, labelled or not, judged or not.

    Planned-vs-unplanned is `trade_origin.planned_state` and nothing else: a
    date-only broker fill is ``unmeasured``, never ``unplanned`` (a broker file
    is authoritative for money and blind to time). The label ages come from the
    row's own `label_provenance`, written by TJ-9; a row with the key PRESENT
    and EMPTY is an OLD row and counts as unlabelled, never as a third age.

    This is also the reader `mentor_questions.REGISTRY` names for the
    ``trade_origin`` question: where the desk could not tell where a trade came
    from and ASKED, the trader's own answer is filed under that key and is read
    back here. A row with no answer is not an answer.
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
    holds = long_hold_lines(open_positions, mentor_answers=mentor_answers)
    if not rows:
        return _line(
            "process",
            "Process: no trades on this session - nothing to judge.",
            0,
            0,
            planned=0,
            unplanned=0,
            unmeasured=0,
            unlabelled=0,
            label_provenance={name: 0 for name in trade_origin.LABEL_PROVENANCES},
            not_judged=0,
            told_us=0,
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
    for trade in rows:
        state = trade_origin.planned_state(trade, decisions, claims, focus_adds, armed)
        states[state] = states.get(state, 0) + 1
        if answers.get(_text(trade.get("trade_id"))):
            told_us += 1
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
    parts = [
        f"Process: {n} trade(s) - {states.get(trade_origin.PLANNED, 0)} planned, "
        f"{states.get(trade_origin.UNPLANNED, 0)} unplanned, "
        f"{unmeasured} unmeasured (a date-only fill has no time to plan against)."
    ]
    labelled = n - unlabelled
    parts.append(
        f"{labelled} labelled ("
        + ", ".join(f"{name} {count}" for name, count in provenance.items())
        + f"), {unlabelled} unlabelled."
    )
    if told_us:
        parts.append(f"{told_us} you told the desk where it came from.")
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
    return _line(
        "process",
        " ".join(parts),
        n,
        n - unmeasured,
        planned=states.get(trade_origin.PLANNED, 0),
        unplanned=states.get(trade_origin.UNPLANNED, 0),
        unmeasured=unmeasured,
        unlabelled=unlabelled,
        label_provenance=provenance,
        not_judged=len(not_judged),
        told_us=told_us,
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
def _tail_rows(path: Any, limit: int) -> list[dict[str, Any]]:
    """The LAST ``limit`` ledger rows, without reading a megabyte for a sentence.

    Small files go through `ai_jobs.ledger.recent_rows`, the owner's own bounded
    reader, with an EXPLICIT path - its default `ledger_path()` CREATES the AI
    store, and a reader that wants to say "unknown" about a missing store may
    not make one. A big file is SEEKED from the end here rather than read whole:
    `recent_rows` reads the whole file and then slices, which is right for the
    runner and wrong for a line on a page. `ai_jobs/ledger.py` is unchanged.
    """
    import ai_jobs.ledger as ledger

    from pathlib import Path

    target = Path(path)
    try:
        size = target.stat().st_size
    except OSError:
        return []
    if size <= LEDGER_TAIL_BYTES:
        return list(ledger.recent_rows(limit, path=target))
    rows: list[dict[str, Any]] = []
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
        return []
    return rows[-max(1, int(limit)):]


def how_fresh(freshness: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the card rests on: when the story was written, how far the fills
    reach, which session the reads were graded through, and any overnight slot
    whose LAST row for the session is not ``ok``.

    A missing AI store is ``night status unknown`` and CREATES NOTHING. Unknown
    is not "fine": a night nobody can see is a night nobody checked.
    """
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
    last: dict[str, str] = {}
    night_status = "unknown"
    if path is not None:
        from pathlib import Path

        target = Path(path)
        # `exists` never creates; `ledger_path()` would have made the folder.
        if target.exists():
            night_status = "read"
            for row in _tail_rows(target, LEDGER_TAIL_ROWS):
                if not isinstance(row, Mapping):
                    continue
                if row.get("noncanonical"):
                    # A correction is commentary about a run, not a run.
                    continue
                if session and _text(row.get("session_date"))[:10] != session:
                    continue
                job = _text(row.get("job"))
                if job:
                    last[job] = _text(row.get("status"))
    failed = tuple(sorted(job for job, status in last.items() if status != "ok"))
    if night_status == "unknown":
        parts.append("night status unknown - the desk has no AI job ledger to read")
    elif failed:
        parts.append(
            f"{len(last)} overnight slot(s) read; these did not finish ok: "
            + ", ".join(failed)
        )
    else:
        parts.append(f"{len(last)} overnight slot(s) read, every one finished ok")
    return _line(
        "how_fresh",
        "; ".join(parts) + ".",
        len(last),
        len(last),
        failed_slots=failed,
        night_status=night_status,
    )


# ---------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------
def build(day_inputs: Mapping[str, Any]) -> ReportCard:
    """The six lines for ONE session, from what the worker already read.

    PURE apart from :func:`how_fresh`'s bounded ledger tail: every other input
    is a value the Day Review worker opened once, inside the ONE payload.
    """
    inputs = dict(day_inputs or {})
    walkaway = inputs.get("walkaway")
    lines = (
        did_well_line(walkaway),
        missed_line(walkaway),
        your_reads_line(inputs.get("your_reads")),
        congruence_line(inputs.get("congruence")),
        process_line(
            inputs.get("trades") or (),
            origin_lanes=inputs.get("origin_lanes"),
            mentor_answers=inputs.get("mentor_answers") or (),
            walkaway=walkaway,
            open_positions=inputs.get("open_positions") or (),
        ),
        how_fresh(inputs.get("freshness")),
    )
    return ReportCard(session=_text(inputs.get("session"))[:10], lines=lines)


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


def _pooled(cards: Sequence[ReportCard], key: str) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for card in cards:
        for line in card.lines:
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


def week(sessions: Sequence[Mapping[str, Any]]) -> ReportCard:
    """The same six lines over a LIST of sessions - TJ-5's week / month strip.

    Same functions, longer window, ``n`` everywhere. A week POOLS counts and
    computes the ONE Wilson from the pooled pair; it never averages two days'
    rates, which on days of unequal length is a different number.
    """
    days = [dict(day) for day in (sessions or ()) if isinstance(day, Mapping)]
    cards = [build(day) for day in days]
    names = tuple(_text(day.get("session"))[:10] for day in days)
    count = len(names)
    label = f"over {count} session(s)"

    def _sum(key: str, field_name: str) -> int:
        return sum(int(line.get(field_name) or 0) for line in _pooled(cards, key))

    lines: list[dict[str, Any]] = []

    # -- Did well ----------------------------------------------------------
    n = _sum("did_well", "n")
    measured = _sum("did_well", "measured")
    runs = _sum("did_well", "runs")
    best = _pooled_families(days)
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
        text += (
            " No setup family has enough measured names over this window - too few "
            "to call."
        )
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
    if n:
        text = (
            f"Your reads {label}: {right} right of {n}" + _floor_clause(n) + "."
        )
    else:
        text = f"Your reads {label}: no graded reads yet."
    lines.append(_line("your_reads", text, n, measured, right=right, **_rate_keys(right, n)))

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

    return ReportCard(
        session=names[-1] if names else "",
        lines=tuple(lines),
        sessions=names,
    )


__all__ = [
    "LEDGER_TAIL_BYTES",
    "LEDGER_TAIL_ROWS",
    "LINE_KEYS",
    "LINE_TARGETS",
    "ReportCard",
    "build",
    "congruence_line",
    "did_well_line",
    "how_fresh",
    "long_hold_lines",
    "missed_line",
    "process_line",
    "week",
    "your_reads_line",
]
