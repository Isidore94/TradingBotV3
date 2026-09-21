"""The desk's AI has a voice: ideas the trader keeps or throws away (TJ-6).

`plan.md` §12.4 TJ-6 changes 1-3 with the **AMENDED 2026-09-19** block, decision
0021 answer 23. The pattern beside it is TJ-5's `week_review_narration.py`, and
every rule that packet learned the hard way is repeated here:

* **A JSON schema is a grammar hint, never a guard.** `maxItems`, `enum` and
  `additionalProperties` tell the constrained decoder what shape to write; they
  do not stop another shape arriving. :func:`check_ideas` re-checks every bound
  AFTER the answer comes back, and the bounds it checks come from the INPUT -
  the measurables the desk really computes, the ids this night really carries.
* **A status vocabulary is IMPORTED from its owner** (`ai_jobs.ledger`).
* **One call a night.** There is nothing to sweep: one night, one ask.

THE HARD INVARIANT
------------------
**An idea is a SUGGESTION.** Nothing written here is read by a detector, a
score, an alert, a watchlist, Focus, the review queue or `review_policy.json`;
no job writes `WISHLIST.md` or `plan.md`; and a KEEP is the TRADER's click -
:func:`keep_idea` and :func:`dismiss_idea` are the card's writers and no nightly
job may call them (`tests/test_tj6_keep_is_the_traders_act.py` proves it over the
source of every `ai_jobs` module).

THE TWO STORES
--------------
``AI_IDEAS_FILE`` is the night's half: JSONL, **append-only**. `plan.md` says a
repeat "increments ``seen_count``"; a store that REWROTE a row to do that would
lose the earlier sighting, which CLAUDE.md forbids of an evidence store. So a
repeat APPENDS a row carrying the same ``idea_id`` and a higher ``seen_count``,
and :func:`read_ideas` folds by id keeping the LAST row. One idea, one id, every
sighting still on disk.

``AI_IDEAS_STATE_FILE`` is the trader's half: ``{idea_id: {status, at}}`` plus,
for a kept `process` idea, the measurable's baseline FROZEN at the keep. It is
written temp-and-rename, one entry at a time, every other entry byte-identical.

ADVICE IS CHECKED
-----------------
A `process` idea must name ONE measurable the desk already computes or it is
dropped. :data:`MEASURABLES` is that CLOSED registry, and every entry names the
reader of its number, resolvable and callable (the rule
`mentor_questions.consumer_report` keeps for a question's answer). Two exist
today - a veto reason's real-miss rate, and a report-card line - because those
are the two the desk really has; the amendment's third example, a Mentor
question's answer mix, has no pooled reader and inventing one here would widen
the packet (lead decision, 2026-09-20).

:func:`measure` reads the LAST WRITTEN pack, never a 20-session build behind a
click, and an unreadable one is ``unmeasured`` with its reason - never a zero.
The baseline is frozen at the keep and never re-read; Week Review prints before
and after with both `n`, and under `evidence_stats.MIN_REPORTABLE_N` it says
"too few to call". **The model never grades its own advice**: nothing in the
checking path loads a model at all.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import evidence_stats
import project_paths
from ai_jobs import ledger

_log = logging.getLogger(__name__)

PROMPT_VERSION = "improvement_ideas_v1"
SCHEMA = "improvement_ideas_v1"

#: The name the structured-output validator reports under.
SCHEMA_NAME = "tradingbot_improvement_ideas"

#: The two kinds of idea, and there is no third. A `process` idea is about what
#: the TRADER does and must name a measurable; a `program` idea is about what the
#: DESK does and is listed for WISHLIST rather than measured.
IDEA_KINDS: tuple[str, ...] = ("process", "program")

#: *"up to three a night"* (`plan.md` TJ-6 change 1). A cap on what one night may
#: say, not on what the trader may keep.
MAX_IDEAS_PER_NIGHT = 3

#: *"text <= 280"*. A bound, so it is re-checked after the answer comes back.
MAX_IDEA_CHARS = 280

#: How many ids ONE idea may cite. An idea that needs ten rows to stand up is not
#: one idea.
MAX_EVIDENCE_PER_IDEA = 4

#: *"an idea whose normalised text matches one from the last 60 sessions
#: increments `seen_count` instead"*. SESSIONS, walked on the exchange calendar:
#: sixty calendar days is about forty-two sessions, so a window written in days
#: would forget eighteen sessions of ideas and start repeating itself.
DEDUPE_SESSIONS = 60

#: How many sessions of the trader's own evidence one night may look at. Five,
#: the same week TJ-5's story reads, ending at the session that just closed.
IDEAS_SESSIONS = 5

#: How many rows of one kind travel per day. What does not fit is COUNTED and
#: said (TJ-13A's bounded-package rule), never silently dropped.
MAX_ITEMS_PER_DAY = 12

#: How many contrast groups travel. The pack is already floor-aware and ordered
#: by SIZE; nothing here re-ranks it.
MAX_GROUPS = 8

#: One local call, the same 900 s every other session-scale local call uses.
TIMEOUT_SECONDS = 900

#: What the slot reserves (`plan.md` TJ-6 change 1: ``reserve_minutes=10``).
RESERVE_MINUTES = 10.0

#: How far back :func:`measure` may look for the last written contrast pack. The
#: night writes one per session; a trader clicking Keep on a Monday is reading
#: Friday's. Past this it is `unmeasured` and says so - never a stale number
#: presented as this week's.
PACK_LOOKBACK_SESSIONS = 5

STATUS_KEPT = "kept"
STATUS_DISMISSED = "dismissed"

#: Why one idea was dropped. Codes rather than sentences, because they are
#: COUNTED per reason into the slot's ledger row - a post-mortem that says "two
#: dropped" and not which two is a number nobody can act on (reviewer advisory
#: 2, 2026-09-20).
DROP_NOT_AN_OBJECT = "not_an_object"
DROP_UNKNOWN_KIND = "unknown_kind"
DROP_NO_TEXT = "no_text"
DROP_TOO_LONG = "too_long"
DROP_NO_EVIDENCE = "no_evidence"
DROP_NO_MEASURABLE = "no_measurable"
DROP_UNKNOWN_MEASURABLE = "unknown_measurable"
DROP_REPEATED_IN_ANSWER = "repeated_in_answer"

DROP_CODES: tuple[str, ...] = (
    DROP_NOT_AN_OBJECT,
    DROP_UNKNOWN_KIND,
    DROP_NO_TEXT,
    DROP_TOO_LONG,
    DROP_NO_EVIDENCE,
    DROP_NO_MEASURABLE,
    DROP_UNKNOWN_MEASURABLE,
    DROP_REPEATED_IN_ANSWER,
)

#: The marker the night leaves when it ASKED and stored nothing.
#:
#: A slot that called the model and then answered `skipped` with no artifact
#: re-asks on every one of the scheduled task's sixteen passes: `skipped` is in
#: neither `ledger.CANONICAL_COMPLETION_STATUSES` nor `ledger.ATTEMPT_STATUSES`,
#: so neither the runner's already-done check nor `max_attempts` ever bites
#: (reviewer, 2026-09-20, reproduced). So a night that ASKED is a night that is
#: DONE: it ends `ok`, and this marker is the artifact that says so - beside the
#: store, never inside it (two tester tests count the store's raw lines) and
#: never in the trader's state file, which the night may not write.
ASKED_MARKER_SCHEMA = "improvement_ideas_asked_v1"

#: What a checked idea can say. Every one of them is arithmetic over two stored
#: readings; none of them is a grade of the advice.
VERDICT_TOO_FEW = "too few to call"
VERDICT_NOT_CHECKED = "not checked here"
VERDICT_UNMEASURED = "unmeasured"
VERDICT_HIGHER = "higher than at the keep"
VERDICT_LOWER = "lower than at the keep"
VERDICT_SAME = "the same as at the keep"
#: Both sides over the floor, and their Wilson intervals OVERLAP. Two numbers
#: that differ by less than their own uncertainty have not moved.
VERDICT_NO_CHANGE = "no clear change"

#: What separates a session from the id it qualifies, exactly as TJ-5's week
#: story does it: each day pack mints its ids with its OWN minter, so
#: ``report_card:did_well`` repeats across five packs and an unqualified citation
#: would name five rows.
SOURCE_SEPARATOR = "/"


class IdeasRejected(ValueError):
    """The night's answer was not a set of ideas about the evidence it was given."""


# ---------------------------------------------------------------------------
# the program card - what a `program` idea is allowed to be about
# ---------------------------------------------------------------------------
#: A fixed, versioned description of THIS program's pages and fields, so a
#: `program` idea is about the desk the trader actually has rather than about
#: trading software in general. Checked in, exactly thirty lines, and never
#: built from the code at run time: a card that read the repository would
#: describe whatever it found, including a half-built page.
IDEAS_PROGRAM_CARD: tuple[str, ...] = (
    "TradingBotV3 is a Windows decision-support desk for one trader. It never places an order.",
    "Day Review: one page per session, read once on one worker - story, theses, trades, ideas.",
    "Day Review's report card is six lines: did well, missed, your reads, congruence, process, how fresh.",
    "Every report-card line carries its own n and how many of those n were measured.",
    "Week Review is the first page of Weekend Prep: five day cards, the week story, a week/month strip.",
    "A week card shows the day's headline, the were-you-right tally, a chased flag and said-vs-did.",
    "The Market Journal holds what the trader thought; the Journal holds what they traded.",
    "The Trade Mentor asks a present trader at whole hours; 09:00 is the forced trade check.",
    "A Mentor row is an observation (words) plus a forced prediction click: direction, horizon, confidence.",
    "The read grader measures a prediction against completed bars and answers right, wrong or unresolved.",
    "Walk-away measures a decision the trader did not act on: liked, claimed, rejected or untouched.",
    "A real miss is 1.0 ATR in the decision's favour before 0.5 ATR against it, completed bars only.",
    "The miss contrast groups D1 decisions by veto reason code and reports each group's real-miss rate.",
    "The prediction contrast puts the trader's calls beside three naive baselines on the same stamps.",
    "Alert Center reviews charts: a veto retires the chart, a claimed like places and advances it.",
    "A day-trade pass is a note with its own vocabulary and never retires a chart.",
    "M5 Focus is the intraday shortlist; a Focus pick fades after ten quiet trading days.",
    "The D1 scan is an anchored-VWAP swing scanner; BounceBot is the intraday 5-minute detector.",
    "The Strength boards rank names on relative volume and relative strength, batched, no IB traffic.",
    "The setup tracker replays every scanned thesis and stores the outcome by execution convention.",
    "Overnight jobs run in three stages: deterministic, narration, then the model-gated stage.",
    "Local inference is night-only and runs on a medium local model; the large one is measured first.",
    "Every overnight job writes one ledger row naming its status, its reason and what it wrote.",
    "Statistics use one Wilson lower bound, integer counts, and name nothing under thirty measured.",
    "Lately means twenty exchange sessions; a week means five; every surface says sessions.",
    "Missing data is uncertainty: the desk says unmeasured and never writes a zero in its place.",
    "The trader owns their tags; a machine may only write a provisional tag or a needs-review mark.",
    "Nothing the AI writes may reach a detector, a score, an alert, a watchlist, Focus or a policy.",
    "Ideas from the desk's AI are suggestions: the trader keeps or dismisses each one by hand.",
    "A kept process idea freezes its measurable's baseline so the advice itself can be checked later.",
)

PROGRAM_CARD_VERSION = "ideas_program_card_v1"

#: The CLOSED set of evidence sections. Nothing else may be sent, and a test
#: asserts no bar, tape, tick, lake or warehouse section can appear in it.
EVIDENCE_KEYS: tuple[str, ...] = (
    "package_id",
    "evidence_hash",
    "instructions",
    "allowed_source_ids",
    "session_date",
    "sessions",
    "sessions_with_facts",
    "sessions_missing",
    "days",
    "misses",
    "walkaway_totals",
    "measurables",
    "measurable_readings",
    "program_card",
    "program_card_version",
    # TJ-7's ONE addition, and the entry `plan.md` TJ-6 amendment (h) says this
    # packet owes TJ-6: what the trader said about THEMSELVES over the window,
    # under its own name. It is context an idea may cite, never a measurable -
    # `MEASURABLES` is closed and gains nothing here - and no idea may pair a
    # mood with a result, because no result is in this package at all.
    "mood",
)

#: How many mood rows one session contributes. A mood is one or two clicks a
#: day; what does not fit is COUNTED and said, never silently dropped.
MAX_MOODS_PER_DAY = 4

INSTRUCTIONS = (
    "Suggest at most three improvements from the evidence below and nothing "
    "else. You are advising ONE trader about their own recorded sessions and "
    "about this program. Every idea must cite at least one id copied exactly "
    "from allowed_source_ids. A 'process' idea is about what the trader does "
    "and MUST name one measurable from the list you were handed, so the idea "
    "can be checked later; a 'program' idea is about this desk's pages and "
    "leaves measurable empty. Do not calculate a statistic, do not grade a "
    "call, do not turn an unmeasured row into a fact, and do not name, rank or "
    "score a symbol. Say nothing rather than saying something the evidence "
    "does not carry."
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    return str(value or "").strip()


def _moment(now: datetime | None) -> str:
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat(timespec="seconds")


def normalise_text(text: Any) -> str:
    """The normal form two spellings of one thought share.

    Casefolded, punctuation turned into space, runs of space collapsed - and
    NOTHING else. No stemming and no stop-word stripping: those fold two
    different thoughts together, and this form decides both whether a repeat is
    the same idea and whether a DISMISSED idea is coming back.
    """
    body = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in str(text or "").casefold()
    )
    return " ".join(body.split())


def mint_idea_id(session: Any, text: Any) -> str:
    """``idea:<session>:<sha1 of the normal form, 12 hex>``.

    The MODEL never names an idea (a model that picked its own id could name a
    dismissed one back to life). The id carries the session the idea was FIRST
    seen in, so the same sentence returning after the dedupe window is a new
    idea rather than a resurrected one - which is why "a dismissed idea never
    returns" is checked on the TEXT and not on the id.
    """
    digest = hashlib.sha1(normalise_text(text).encode("utf-8")).hexdigest()[:12]
    return f"idea:{_text(session)[:10]}:{digest}"


def _previous_session(day: date) -> date:
    import market_calendar

    return market_calendar.previous_session(day)


def sessions_ending(session: Any, count: int) -> tuple[str, ...]:
    """`count` exchange sessions ending at `session`, oldest first.

    Walked on the exchange calendar. A calendar that cannot answer falls back to
    calendar days, which is wider than the truth and therefore never silently
    narrows a window.
    """
    try:
        cursor = date.fromisoformat(_text(session)[:10])
    except ValueError:
        return ()
    out = [cursor]
    for _step in range(max(0, int(count) - 1)):
        try:
            cursor = _previous_session(cursor)
        except Exception:  # noqa: BLE001 - an unanswerable calendar walks days
            _log.debug("The exchange calendar could not walk back.", exc_info=True)
            cursor = cursor.fromordinal(cursor.toordinal() - 1)
        out.append(cursor)
    return tuple(day.isoformat() for day in reversed(out))


def source_id(session: Any, item: Any) -> str:
    """``<session>/<source_id>`` - a citation that names ONE row of ONE day."""
    return f"{_text(session)[:10]}{SOURCE_SEPARATOR}{_text(item)}"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _ideas_path() -> Path:
    """Resolved at CALL time, never bound at import.

    ``from project_paths import AI_IDEAS_FILE`` would keep pointing at the
    trader's own home folder after a test moved the constant, which is the
    2026-09-05 and 2026-09-20 incidents in miniature.
    """
    return Path(project_paths.AI_IDEAS_FILE)


def _state_path() -> Path:
    return Path(project_paths.AI_IDEAS_STATE_FILE)


def _asked_path() -> Path:
    """``ai_ideas_asked.json``, beside the store. Resolved at CALL time too."""
    path = _ideas_path()
    return path.with_name(f"{path.stem}_asked.json")


# ---------------------------------------------------------------------------
# the measurables - what a `process` idea may name
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Measurable:
    """One number the desk ALREADY computes, and the reader that answers it."""

    #: The name an idea cites. Stored on the idea and on its frozen baseline.
    name: str
    #: The dotted path of the reader. Resolvable and callable - a measurable
    #: whose reader does not exist is a promise the desk cannot keep.
    reader: str
    #: One sentence for the trader, beside the two numbers.
    label: str
    #: How this measurable turns its reader into a reading. Kept beside the
    #: reader so the registry is one place rather than a name plus a branch.
    read: Callable[..., dict[str, Any]]


def _unmeasured(name: str, sessions: int, reason: str, **extra: Any) -> dict[str, Any]:
    """A reading nobody could take. `plan.md` sec 5: never a zero."""
    return {
        "measurable": name,
        "value": None,
        "n": 0,
        "measured": False,
        "window_sessions": int(sessions),
        "reason": reason,
        "source": dict(extra),
    }


def _report_card_rate(reader, *, name: str, end_session: str, sessions: int) -> dict[str, Any]:
    """The pooled `did_well` line over the window, from the packs' OWN lines.

    It builds nothing: a pack carries the BUILT card with all of its integers,
    and `day_report_card.week_from_cards` is the ONE place that knows what
    pooling a line means. A session with no pack contributes nothing and is
    NAMED - a missing day is never a zero day.
    """
    import day_review_pack

    window = sessions_ending(end_session, sessions)
    cards: list[dict[str, Any]] = []
    missing: list[str] = []
    for day in window:
        pack = day_review_pack.read_pack(day)
        card = pack.get("report_card") if isinstance(pack, Mapping) else None
        lines = card.get("lines") if isinstance(card, Mapping) else None
        if lines:
            cards.append({"session": day, "lines": list(lines)})
        else:
            missing.append(day)
    if not cards:
        return _unmeasured(
            name,
            sessions,
            f"no day pack in the {len(window)} session(s) ending {end_session} "
            "carries a report card",
            reader=reader_name(reader),
            sessions_missing=list(missing),
        )
    pooled = reader(cards)
    line = _line_named(pooled, "did_well")
    if line is None:
        return _unmeasured(
            name,
            sessions,
            "the pooled report card carries no did_well line",
            sessions_read=[card["session"] for card in cards],
        )
    runs = int(line.get("runs") or 0)
    measured = int(line.get("measured") or 0)
    return {
        "measurable": name,
        "value": (runs / measured) if measured else None,
        # The numerator travels with the rate: an interval needs the two
        # INTEGERS, and reconstructing hits from a stored float is a rounding
        # error waiting to be printed as a change.
        "hits": runs,
        "n": measured,
        "measured": bool(measured),
        "window_sessions": int(sessions),
        "reason": "" if measured else "no decision in this window was measured",
        "source": {
            "kind": "day_packs",
            "line": "did_well",
            "runs": runs,
            "sessions_read": [card["session"] for card in cards],
            "sessions_missing": missing,
        },
    }


def _veto_real_miss_rate(reader, *, name: str, end_session: str, sessions: int) -> dict[str, Any]:
    """The pooled real-miss rate of the VETO groups in the LAST WRITTEN pack.

    It reads the pack the night already wrote (`miss_contrast.read_latest`) and
    never runs a contrast build: that walks twenty sessions of decisions and
    daily bars, which is not something a Keep click may start (lead decision,
    2026-09-20). Which pack it read is recorded beside the number.
    """
    window = sessions_ending(end_session, PACK_LOOKBACK_SESSIONS)
    for day in reversed(window):
        pack = reader(day)
        if not isinstance(pack, Mapping) or not pack:
            continue
        groups = [
            group
            for group in pack.get("groups") or ()
            if isinstance(group, Mapping) and _text(group.get("verdict")) == "veto"
        ]
        misses = sum(int(group.get("misses") or 0) for group in groups)
        measured = sum(int(group.get("measured") or 0) for group in groups)
        return {
            "measurable": name,
            "value": (misses / measured) if measured else None,
            "hits": misses,
            "n": measured,
            "measured": bool(measured),
            "window_sessions": int(pack.get("window_sessions") or sessions),
            "reason": "" if measured else "no veto in this pack had a closed horizon",
            "source": {
                "kind": "miss_contrast_pack",
                "session_date": _text(pack.get("session_date")),
                "built_at": _text(pack.get("built_at")),
                "misses": misses,
                "groups": len(groups),
            },
        }
    return _unmeasured(
        name,
        sessions,
        f"no miss-contrast pack was written in the {len(window)} session(s) "
        f"ending {end_session}",
        reader=reader_name(reader),
    )


def reader_name(reader: Any) -> str:
    return _text(getattr(reader, "__name__", "")) or "the reader"


def _line_named(card: Any, key: str) -> Mapping[str, Any] | None:
    lines = getattr(card, "lines", None)
    if lines is None and isinstance(card, Mapping):
        lines = card.get("lines")
    for line in lines or ():
        if isinstance(line, Mapping) and _text(line.get("key")) == key:
            return line
    return None


#: The CLOSED registry. Two entries, because two is what the desk really
#: computes: the amendment's third example - a Mentor question's answer mix -
#: has no pooled reader (`day_report_card.process_line`'s `origin_answers` is
#: one kind, one session, and `_pool_cards` drops it), and inventing one here
#: would widen the packet (lead decision, 2026-09-20). Adding it later is one
#: entry in this tuple.
MEASURABLES: tuple[Measurable, ...] = (
    Measurable(
        name="report_card_did_well_rate",
        reader="day_report_card.week_from_cards",
        label=(
            "How often a pick you liked or claimed really ran your way - the "
            "report card's 'did well' line, pooled over the window from the day "
            "packs' own counts"
        ),
        read=_report_card_rate,
    ),
    Measurable(
        name="veto_reason_real_miss_rate",
        reader="ai_jobs.miss_contrast.read_latest",
        label=(
            "How often a veto turned out to be a real miss - every veto reason "
            "code in the last miss-contrast pack, pooled"
        ),
        read=_veto_real_miss_rate,
    ),
)


def measurable_named(name: Any) -> Measurable:
    """The measurable called `name`. An unknown name RAISES, never defaults.

    `mentor_questions.kind_named`'s rule: a default here would let an idea
    nobody can check through the drop that exists to stop it.
    """
    wanted = _text(name)
    for item in MEASURABLES:
        if item.name == wanted:
            return item
    raise KeyError(f"no measurable named {name!r}; the desk offers {measurable_names()}")


def measurable_names() -> tuple[str, ...]:
    return tuple(item.name for item in MEASURABLES)


def _resolve(dotted: str):
    """Import the longest importable prefix of `dotted` and walk the rest.

    Resolved at CALL time, so a measurable's reader is whatever its module holds
    now - which is also what makes an unreadable reader testable.
    """
    parts = [part for part in _text(dotted).split(".") if part]
    module = None
    index = 0
    for stop in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:stop]))
        except Exception:  # noqa: BLE001 - keep shortening until one imports
            continue
        index = stop
        break
    if module is None:
        raise ImportError(f"no prefix of {dotted!r} imports")
    target: Any = module
    for part in parts[index:]:
        target = getattr(target, part)
    return target


def measure(
    name: Any,
    *,
    end_session: Any = "",
    sessions: int = evidence_stats.LATELY_SESSIONS,
) -> dict[str, Any]:
    """ONE reading of ONE measurable, with its own `n` and its window.

    Never raises and never guesses: a reader that cannot answer comes back
    ``measured: False``, ``value: None``, ``n: 0`` and a reason. A store that
    will not open must not become "your real-miss rate is 0%".
    """
    item = measurable_named(name)
    session = _text(end_session)[:10] or date.today().isoformat()
    try:
        reader = _resolve(item.reader)
    except Exception as exc:  # noqa: BLE001 - an unreadable reader is uncertainty
        _log.debug("A measurable's reader could not be resolved.", exc_info=True)
        return _unmeasured(item.name, sessions, f"{item.reader} could not be read: {exc}")
    try:
        reading = item.read(reader, name=item.name, end_session=session, sessions=int(sessions))
    except Exception as exc:  # noqa: BLE001 - an unreadable store is uncertainty
        _log.debug("A measurable could not be read.", exc_info=True)
        return _unmeasured(item.name, sessions, f"{item.reader} could not be read: {exc}")
    reading.setdefault("label", item.label)
    reading.setdefault("end_session", session)
    return reading


# ---------------------------------------------------------------------------
# the night's inputs
# ---------------------------------------------------------------------------
def _root(root: Path | None) -> Path:
    import day_review_pack

    return Path(root) if root is not None else day_review_pack.default_root()


def _bounded(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> tuple[list[dict], int]:
    kept = [
        {key: row.get(key) for key in keys}
        for row in list(rows)[:MAX_ITEMS_PER_DAY]
        if isinstance(row, Mapping)
    ]
    return kept, max(0, len(list(rows)) - len(kept))


def _day_block(session: str, pack: Mapping[str, Any] | None, story: Mapping[str, Any]):
    """One day's evidence, bounded, every id session-qualified.

    Returns ``(block, ids)``: the ids are exactly what the block SHOWED, so a
    citation can never name a row the model was not handed.
    """
    if not isinstance(pack, Mapping) or not pack:
        return (
            {
                "session": session,
                "has_facts": False,
                "narrated": False,
                "note": "no day pack was built for this session; nobody measured it",
            },
            [],
        )
    card = pack.get("report_card") if isinstance(pack.get("report_card"), Mapping) else {}
    lines, lines_over = _bounded(
        [row for row in (card or {}).get("lines") or () if isinstance(row, Mapping)],
        ("key", "text", "n", "measured", "source_id"),
    )
    walkaway = pack.get("walkaway") if isinstance(pack.get("walkaway"), Mapping) else {}
    trades = pack.get("trades") if isinstance(pack.get("trades"), Mapping) else {}

    ids: list[str] = []
    for row in lines:
        row["source_id"] = source_id(session, row.get("source_id"))
        if row["source_id"] not in ids:
            ids.append(row["source_id"])

    block: dict[str, Any] = {
        "session": session,
        "has_facts": True,
        "narrated": bool(story),
        "report_card": lines,
        "trades": {"n": int(trades.get("n") or 0)},
        "walkaway_counts": dict(walkaway.get("counts") or {}),
        "omitted": {"report_card": lines_over},
    }
    if story:
        story_id = source_id(session, "story")
        block["story"] = {
            "source_id": story_id,
            "headline": _text(story.get("headline")),
            "what_happened": _text(story.get("what_happened")),
            "process": _text(story.get("process")),
            "chased_against_news": dict(story.get("chased_against_news") or {}),
        }
        ids.append(story_id)
    return block, ids


def _moods(packs: Mapping[str, Mapping[str, Any]], sessions: Sequence[str]):
    """TJ-7's `mood` section for the window. Returns ``(section, ids)``.

    Every row came out of a PACK this night already opened, with its id
    session-qualified so a cited mood names ONE row of ONE day. A session with
    no pack, or a pack with no mood, contributes nothing - and neither is a
    zero. Nothing here computes a statistic or pairs a mood with an outcome.
    """
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    omitted = 0
    for session in sessions:
        section = (packs.get(session) or {}).get("mood")
        recorded = list((section or {}).get("recorded") or ()) if isinstance(section, Mapping) else []
        omitted += max(0, len(recorded) - MAX_MOODS_PER_DAY)
        for item in recorded[:MAX_MOODS_PER_DAY]:
            if not isinstance(item, Mapping):
                continue
            cited = source_id(session, item.get("source_id"))
            rows.append(
                {
                    "session": session,
                    "at": _text(item.get("at")),
                    "score": item.get("score"),
                    "state_tags": list(item.get("state_tags") or ()),
                    "followed_plan": _text(item.get("followed_plan")),
                    "note": _text(item.get("note")),
                    "written_after_the_session": bool(item.get("written_after_the_session")),
                    "source_id": cited,
                }
            )
            if cited not in ids:
                ids.append(cited)
    return (
        {
            "n": len(rows),
            "sessions_with_a_mood": sorted({row["session"] for row in rows}),
            "recorded": rows,
            "omitted": omitted,
        },
        ids,
    )


def _day_story(session: str, root: Path) -> Mapping[str, Any]:
    from ai_jobs import day_review_narration

    try:
        stored = day_review_narration.read_narration(session, root=root)
    except Exception:  # noqa: BLE001 - a missing story is simply no story
        _log.debug("A day story could not be read.", exc_info=True)
        return {}
    if not isinstance(stored, Mapping):
        return {}
    body = stored.get("narration")
    return dict(body) if isinstance(body, Mapping) else {}


def _misses(session: str) -> dict[str, Any]:
    """TJ-15's contrast pack, bounded to its groups. The reason codes and their
    real-miss rates are the evidence a `process` idea about vetoing stands on."""
    from ai_jobs import miss_contrast

    try:
        pack = miss_contrast.read_latest(session)
    except Exception:  # noqa: BLE001 - a missing pack is simply no misses
        _log.debug("The miss-contrast pack could not be read.", exc_info=True)
        return {}
    if not isinstance(pack, Mapping) or not pack:
        return {}
    groups = [
        {
            "name": _text(group.get("name")),
            "verdict": _text(group.get("verdict")),
            "reason_code": _text(group.get("reason_code")),
            "n": int(group.get("n") or 0),
            "measured": int(group.get("measured") or 0),
            "misses": int(group.get("misses") or 0),
            "rate": group.get("rate"),
            "reportable": bool(group.get("reportable")),
        }
        for group in list(pack.get("groups") or ())[:MAX_GROUPS]
        if isinstance(group, Mapping)
    ]
    return {
        "session_date": _text(pack.get("session_date")),
        "source_id": f"misses:{_text(pack.get('session_date'))}",
        "groups": groups,
        "omitted": max(0, len(list(pack.get("groups") or ())) - len(groups)),
        "statement": _text(pack.get("statement")),
    }


def _walkaway_totals(packs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The window's A-D populations, SUMMED. Counts only, never a rate."""
    import day_review_pack

    totals = {name: 0 for name in day_review_pack.WALKAWAY_POPULATIONS}
    for pack in packs.values():
        walkaway = (pack or {}).get("walkaway")
        counts = walkaway.get("counts") if isinstance(walkaway, Mapping) else None
        for name in totals:
            try:
                totals[name] += int((counts or {}).get(name) or 0)
            except (TypeError, ValueError):
                continue
    return {"counts": totals, "sessions": len(packs), "n": sum(totals.values())}


def build_ideas_inputs(
    session_date: Any, *, root: Path | None = None, ledger_path: Any = None
) -> dict[str, Any]:
    """Everything tonight's ideas are allowed to see, already read.

    The last :data:`IDEAS_SESSIONS` packs and their stories, the window's
    walk-away totals, TJ-15's contrast groups, the measurables the desk really
    computes with their current readings, and the checked-in program card - and
    NOTHING else. No bars, no lake, no detector, no journal stream.

    ``ledger_path`` is accepted so a caller with its own ledger can hand one in;
    the ideas themselves read no ledger (that is the SLOT's record).

    ``inputs_hash`` is over the sections and never over the clock, so an
    unchanged night costs one model call and not one an hour.
    """
    import day_review_pack

    base = _root(root)
    session = _text(session_date)[:10]
    sessions = sessions_ending(session, IDEAS_SESSIONS)

    packs: dict[str, Mapping[str, Any]] = {}
    for day in sessions:
        pack = day_review_pack.read_pack(day, root=base)
        if isinstance(pack, Mapping) and pack:
            packs[day] = pack

    days: list[dict[str, Any]] = []
    allowed: list[str] = []
    for day in sessions:
        story = _day_story(day, base) if day in packs else {}
        block, ids = _day_block(day, packs.get(day), story)
        days.append(block)
        for item in ids:
            if item not in allowed:
                allowed.append(item)

    misses = _misses(sessions[-1] if sessions else session)
    if misses.get("source_id") and misses["source_id"] not in allowed:
        allowed.append(misses["source_id"])

    mood, mood_ids = _moods(packs, sessions)
    for item in mood_ids:
        if item not in allowed:
            allowed.append(item)

    readings = []
    for item in MEASURABLES:
        reading = measure(item.name, end_session=session)
        readings.append(
            {
                "measurable": item.name,
                "label": item.label,
                "value": reading.get("value"),
                "n": int(reading.get("n") or 0),
                "measured": bool(reading.get("measured")),
                "window_sessions": int(reading.get("window_sessions") or 0),
            }
        )

    body: dict[str, Any] = {
        "session_date": session,
        "sessions": list(sessions),
        "sessions_with_facts": [day for day in sessions if day in packs],
        "sessions_missing": [day for day in sessions if day not in packs],
        "days": days,
        "allowed_source_ids": allowed,
        "misses": misses,
        "walkaway_totals": _walkaway_totals(packs),
        "measurables": list(measurable_names()),
        "measurable_readings": readings,
        "program_card": list(IDEAS_PROGRAM_CARD),
        "program_card_version": PROGRAM_CARD_VERSION,
        "mood": mood,
    }
    body["inputs_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return body


def build_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The package as the model sees it. Keys are a subset of :data:`EVIDENCE_KEYS`."""
    body = {name: inputs.get(name) for name in EVIDENCE_KEYS if name in inputs}
    body["package_id"] = f"improvement-ideas:{_text(inputs.get('inputs_hash'))[:16]}"
    body["evidence_hash"] = _text(inputs.get("inputs_hash"))
    body["instructions"] = INSTRUCTIONS
    return body


IDEAS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ideas"],
    "properties": {
        "ideas": {
            "type": "array",
            "maxItems": MAX_IDEAS_PER_NIGHT,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "text", "measurable", "evidence"],
                "properties": {
                    "kind": {"type": "string", "enum": list(IDEA_KINDS)},
                    "text": {"type": "string", "maxLength": MAX_IDEA_CHARS},
                    "measurable": {"type": "string", "enum": list(measurable_names()) + [""]},
                    "evidence": {
                        "type": "array",
                        "maxItems": MAX_EVIDENCE_PER_IDEA,
                        "items": {"type": "string", "maxLength": 200},
                    },
                },
            },
        }
    },
}


def schema_for(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Tonight's schema: the bounds come FROM THE INPUT.

    The measurable enum is the registry the desk really has, and the citation
    cap is how many ids this night really carries. The item's
    ``additionalProperties`` is False and it declares no ``idea_id``, no
    ``seen_count`` and no verdict field: naming and grading are the DESK's.
    """
    body = json.loads(json.dumps(IDEAS_JSON_SCHEMA))
    item = body["properties"]["ideas"]["items"]
    offered = [_text(name) for name in inputs.get("measurables") or ()]
    item["properties"]["measurable"]["enum"] = offered + [""]
    item["properties"]["evidence"]["maxItems"] = min(
        MAX_EVIDENCE_PER_IDEA, len(list(inputs.get("allowed_source_ids") or ()))
    )
    return body


def _shape_only(item_schema: Mapping[str, Any]) -> dict[str, Any]:
    """The item schema with its PER-IDEA rules taken out.

    Two different things live in one schema. The SHAPE of an idea - an object,
    those four keys, no others - is a bound on the answer, and a breach of it
    rejects the answer whole. Which measurable an idea names and how long its
    text is are per-IDEA judgements the packet calls DROPS ("an idea without
    evidence is dropped ... a `process` idea must name one measurable ... or it
    is dropped like an idea without evidence"), so they are settled one idea at
    a time in :func:`usable_ideas` and a fourth bad thought never costs the
    three good ones.
    """
    body = json.loads(json.dumps(dict(item_schema)))
    for spec in (body.get("properties") or {}).values():
        spec.pop("enum", None)
        spec.pop("maxLength", None)
    return body


def _validate(payload: Any, schema: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """The closed schema, top level and one level into the ideas array.

    `ai_summary.validate_structured_output` is the SHARED validator and it stops
    at the top level's own strings: it enforces neither ``maxItems`` nor an
    array item's own properties. TJ-4 found that gap the hard way.
    """
    import ai_summary

    body = ai_summary.validate_structured_output(payload, schema, name=name)
    rows = list(body.get("ideas") or ())
    item_schema = _shape_only(schema["properties"]["ideas"]["items"])
    body["ideas"] = [
        ai_summary.validate_structured_output(row, item_schema, name=f"{name}.ideas[{index}]")
        for index, row in enumerate(rows)
    ]
    return body


def drop_reason(item: Any) -> str:
    """Why this one idea cannot be stored, or ``""`` when it can. Pure.

    The four per-IDEA drops the packet names, in one place so the verifier and
    the writer agree about what a usable idea is: no evidence behind it, no text
    (or more text than the cap), a kind nobody has, or - for a `process` idea -
    no measurable the desk really computes, which is advice nobody could ever
    check (AMENDED 2026-09-19).
    """
    if not isinstance(item, Mapping):
        return DROP_NOT_AN_OBJECT
    text = _text(item.get("text"))
    kind = _text(item.get("kind"))
    measurable = _text(item.get("measurable"))
    evidence = [_text(cited) for cited in item.get("evidence") or () if _text(cited)]
    if kind not in IDEA_KINDS:
        return DROP_UNKNOWN_KIND
    if not text:
        return DROP_NO_TEXT
    if len(text) > MAX_IDEA_CHARS:
        return DROP_TOO_LONG
    if not evidence:
        return DROP_NO_EVIDENCE
    if kind == "process" and not measurable:
        return DROP_NO_MEASURABLE
    if kind == "process" and measurable not in measurable_names():
        return DROP_UNKNOWN_MEASURABLE
    return ""


def check_ideas(reply_body: Any, inputs: Mapping[str, Any]) -> None:
    """Re-check every bound, against TONIGHT's input. Raises :class:`IdeasRejected`.

    The schema handed to the provider is a grammar hint; this is the guard. It
    rejects the answer WHOLE - a fabricated citation is not a thing to trim -
    and the store stays byte-identical. A well-formed idea that simply carries
    no evidence, names no measurable, or runs past the character cap is a
    different matter: those are DROPPED one at a time by :func:`usable_ideas`,
    which is the packet's own word for them.
    """
    if not isinstance(reply_body, Mapping):
        raise IdeasRejected("the night's answer was not an object")
    rows = reply_body.get("ideas")
    if rows is None or not isinstance(rows, (list, tuple)):
        raise IdeasRejected("the night's answer carried no ideas array")
    # The cap is on what the night would STORE. An idea that is dropped on its
    # own - no evidence, no measurable, too long - never reaches the store, so
    # counting it against the cap would throw away three good ideas because a
    # fourth thought could not be supported. A fourth USABLE idea is a bound
    # break and rejects the answer whole.
    usable = [row for row in rows if not drop_reason(row)]
    if len(usable) > MAX_IDEAS_PER_NIGHT:
        raise IdeasRejected(
            f"the night offered {len(usable)} usable ideas; at most "
            f"{MAX_IDEAS_PER_NIGHT} are allowed"
        )
    allowed = {_text(item) for item in inputs.get("allowed_source_ids") or ()}
    cap = min(MAX_EVIDENCE_PER_IDEA, len(allowed))
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise IdeasRejected(f"idea {index} was not an object")
        cited_rows = list(row.get("evidence") or ())
        if len(cited_rows) > cap:
            raise IdeasRejected(
                f"idea {index} cites {len(cited_rows)} ids; tonight allows at most {cap}"
            )
        for cited in cited_rows:
            if _text(cited) not in allowed:
                raise IdeasRejected(
                    f"idea {index} cited {_text(cited)!r}, which tonight does not carry"
                )


# ---------------------------------------------------------------------------
# reading the two stores
# ---------------------------------------------------------------------------
def read_ideas() -> tuple[dict[str, Any], ...]:
    """Every idea on disk, FOLDED by ``idea_id``, last row winning.

    Never raises and never CREATES the store: a reader that made a file in order
    to say "nothing yet" would leave an empty one in the trader's home folder on
    every desk launch. A torn last line costs that line and not the store.
    """
    path = _ideas_path()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ()
    folded: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, Mapping):
            continue
        key = _text(row.get("idea_id"))
        if not key:
            continue
        folded[key] = dict(row)
    return tuple(folded.values())


def read_state() -> dict[str, Any]:
    """The trader's keeps and dismissals. Never raises, never creates the file."""
    payload = _read_json(_state_path())
    if not isinstance(payload, Mapping):
        return {}
    return {
        _text(key): dict(value)
        for key, value in payload.items()
        if _text(key) and isinstance(value, Mapping)
    }


def _dismissed_forms(rows: Sequence[Mapping[str, Any]], state: Mapping[str, Any]) -> set[str]:
    """The NORMAL FORMS the trader threw away.

    Checked on the text rather than the id because the id carries the session an
    idea was first seen in: after the dedupe window the same sentence mints a
    NEW id, and a dismissal checked by id alone would expire after sixty
    sessions.
    """
    by_id = {_text(row.get("idea_id")): row for row in rows}
    out: set[str] = set()
    for key, record in state.items():
        if _text(record.get("status")) != STATUS_DISMISSED:
            continue
        row = by_id.get(key)
        if row is None:
            continue
        out.add(normalise_text(row.get("text")))
    return out


def ideas_for_session(session: Any) -> tuple[dict[str, Any], ...]:
    """The night's ideas for ONE session, as the Day Review card shows them.

    Dismissed ideas are removed - "a dismissed idea never returns", including to
    the card it was dismissed on, across a restart. Each row carries the
    trader's ``status`` for it, PRESENT and EMPTY when nothing is decided yet.
    """
    wanted = _text(session)[:10]
    rows = read_ideas()
    state = read_state()
    dismissed = _dismissed_forms(rows, state)
    out: list[dict[str, Any]] = []
    for row in rows:
        if _text(row.get("session_date"))[:10] != wanted:
            continue
        record = state.get(_text(row.get("idea_id"))) or {}
        status = _text(record.get("status"))
        if status == STATUS_DISMISSED or normalise_text(row.get("text")) in dismissed:
            continue
        item = dict(row)
        item["status"] = status
        out.append(item)
    return tuple(out)


def checked_ideas(*, end_session: Any = "") -> tuple[dict[str, Any], ...]:
    """Every KEPT idea with its before and after - the Week Review card's rows.

    Deterministic, and **no model is loaded**: the model never grades its own
    advice. ``before`` is the baseline FROZEN at the keep and is never re-read;
    ``after`` is the same measurable now. Under
    `evidence_stats.MIN_REPORTABLE_N` on either side the verdict is
    "too few to call" and nothing is named. A `program` idea has no measurable,
    so it carries no before, no after, and is listed for WISHLIST instead.
    """
    session = _text(end_session)[:10]
    rows = {_text(row.get("idea_id")): row for row in read_ideas()}
    state = read_state()
    out: list[dict[str, Any]] = []
    for key, record in sorted(state.items(), key=lambda pair: (_text(pair[1].get("at")), pair[0])):
        if _text(record.get("status")) != STATUS_KEPT:
            continue
        row = rows.get(key)
        if row is None:
            continue
        kind = _text(row.get("kind")) or IDEA_KINDS[0]
        item: dict[str, Any] = {
            "idea_id": key,
            "kind": kind,
            "text": _text(row.get("text")),
            "measurable": _text(row.get("measurable")),
            # Every row here IS kept - the card shows the same rows on both
            # pages and reads this key to count them.
            "status": STATUS_KEPT,
            "kept_at": _text(record.get("at")),
            "before": {},
            "after": {},
            "verdict": VERDICT_NOT_CHECKED,
            "for_wishlist": kind == "program",
        }
        baseline = record.get("baseline")
        if kind == "program" or not isinstance(baseline, Mapping) or not baseline:
            if kind != "program":
                item["verdict"] = VERDICT_UNMEASURED
            out.append(item)
            continue
        after = measure(
            _text(baseline.get("measurable")) or _text(row.get("measurable")),
            end_session=session or _text(row.get("session_date")),
        )
        item["before"] = with_interval(baseline)
        item["after"] = with_interval(after)
        item["verdict"] = _verdict(item["before"], item["after"])
        out.append(item)
    return tuple(out)


def with_interval(reading: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of one reading carrying the ONE Wilson interval its counts imply.

    `evidence_contrast.rate` is the desk's own interval (the ONE Wilson, z 1.96,
    through `walkaway_day._wilson`) and it is IMPORTED rather than re-derived.
    Nothing is re-measured here: `low` and `high` are arithmetic over the two
    integers the reading already carried, which is what makes it safe to add
    them to a baseline that was frozen before this code existed.
    """
    body = dict(reading or {})
    total = int(body.get("n") or 0)
    if not bool(body.get("measured")) or total <= 0:
        return body
    hits = body.get("hits")
    if hits is None:
        try:
            hits = round(float(body.get("value") or 0.0) * total)
        except (TypeError, ValueError):
            return body
    from evidence_contrast import rate as _rate

    cell = _rate(hits, total)
    body["hits"] = int(hits)
    body["low"] = cell["low"]
    body["high"] = cell["high"]
    body["reportable"] = bool(cell["reportable"])
    return body


def _verdict(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    """Two stored readings, compared. Arithmetic, never a grade of the advice.

    A difference is only SAID when the two Wilson intervals do not overlap.
    Before this rule 0.5000 (n 30) against 0.5001 (n 50,000) printed "higher
    than at the keep" on the Week Review card (reviewer, 2026-09-20) - two
    samples that disagree about nothing, one of them a thousand times the size
    of the other. Overlapping intervals are "no clear change", and nothing here
    is ever called an improvement: this compares one measurable with itself and
    says nothing about why it moved.
    """
    if not bool(before.get("measured")) or not bool(after.get("measured")):
        return VERDICT_UNMEASURED
    floor = evidence_stats.MIN_REPORTABLE_N
    if int(before.get("n") or 0) < floor or int(after.get("n") or 0) < floor:
        return VERDICT_TOO_FEW
    first = with_interval(before)
    second = with_interval(after)
    try:
        low_a, high_a = float(first["low"]), float(first["high"])
        low_b, high_b = float(second["low"]), float(second["high"])
    except (KeyError, TypeError, ValueError):
        return VERDICT_UNMEASURED
    if low_b > high_a:
        return VERDICT_HIGHER
    if high_b < low_a:
        return VERDICT_LOWER
    return VERDICT_NO_CHANGE


# ---------------------------------------------------------------------------
# the trader's two writes - the ONLY writers of the state file
# ---------------------------------------------------------------------------
def _write_state(state: Mapping[str, Any]) -> None:
    """Temp-and-rename, so a killed desk cannot leave half a state file.

    A failed rename RAISES and leaves the previous file exactly as it was; the
    card says so rather than showing "nothing kept" over a keep that happened.
    """
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(state), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    try:
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:  # pragma: no cover - the temp file is already gone
            pass
        raise


def _idea_or_raise(idea_id: Any) -> dict[str, Any]:
    """The stored idea behind a click. Fails CLOSED.

    A keep filed against an id with no row behind it would freeze a baseline
    over a measurable nobody named.
    """
    key = _text(idea_id)
    if not key:
        raise ValueError("a keep or a dismissal needs an idea id")
    for row in read_ideas():
        if _text(row.get("idea_id")) == key:
            return dict(row)
    raise KeyError(f"no idea called {key!r} is on disk")


def keep_idea(idea_id: Any, *, end_session: Any = "", now: datetime | None = None):
    """The trader keeps an idea. Freezes the measurable as it is TODAY.

    Idempotent: keeping an already-kept idea returns the stored record and
    writes nothing, because the baseline is frozen at the FIRST keep and a
    second click must never overwrite it (that is what makes before and after
    two different numbers).
    """
    row = _idea_or_raise(idea_id)
    key = _text(row.get("idea_id"))
    state = read_state()
    existing = state.get(key) or {}
    if _text(existing.get("status")) == STATUS_KEPT:
        return dict(existing)
    stamp = _moment(now)
    record: dict[str, Any] = {"status": STATUS_KEPT, "at": stamp}
    measurable = _text(row.get("measurable"))
    if _text(row.get("kind")) != "program" and measurable:
        reading = measure(
            measurable,
            end_session=_text(end_session) or _text(row.get("session_date")),
        )
        record["baseline"] = {**dict(reading), "at": stamp}
    state[key] = record
    _write_state(state)
    return dict(record)


def dismiss_idea(idea_id: Any, *, now: datetime | None = None):
    """The trader throws an idea away, for good. No measurable is read."""
    row = _idea_or_raise(idea_id)
    key = _text(row.get("idea_id"))
    state = read_state()
    record = {"status": STATUS_DISMISSED, "at": _moment(now)}
    state[key] = record
    _write_state(state)
    return dict(record)


# ---------------------------------------------------------------------------
# the night
# ---------------------------------------------------------------------------
def usable_ideas(
    offered: Sequence[Mapping[str, Any]],
    *,
    session: str,
    stored: Sequence[Mapping[str, Any]],
    dismissed: Sequence[str],
    now: datetime | None = None,
    model: str = "",
    inputs_hash: str = "",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Turn a checked answer into the rows to APPEND. Pure.

    Returns ``(rows, counts)``. An idea is DROPPED on its own - counted, never
    silently swallowed - when it has no evidence, names no measurable it can be
    checked by, carries no text, or runs past the cap. An idea the trader has
    already DISMISSED is not stored again at all. An idea already seen inside
    the dedupe window appends a row with the SAME id and a higher
    ``seen_count``, keeping its ``first_seen``.
    """
    window_start = sessions_ending(session, DEDUPE_SESSIONS)
    earliest = window_start[0] if window_start else session
    seen: dict[str, dict[str, Any]] = {}
    for row in stored:
        if _text(row.get("session_date"))[:10] < earliest:
            continue
        seen[normalise_text(row.get("text"))] = dict(row)

    thrown = set(dismissed)
    rows: list[dict[str, Any]] = []
    counts = {"offered": len(list(offered)), "dropped": 0, "dismissed": 0, "repeats": 0}
    minted: set[str] = set()
    stamp = _moment(now)
    reasons: dict[str, int] = {}

    def _drop(code: str) -> None:
        counts["dropped"] += 1
        reasons[code] = reasons.get(code, 0) + 1

    for item in offered:
        refused = drop_reason(item)
        if refused:
            _drop(refused)
            _log.debug("An idea was dropped: %s", refused)
            continue
        text = _text(item.get("text"))
        kind = _text(item.get("kind"))
        measurable = _text(item.get("measurable"))
        evidence = [_text(cited) for cited in item.get("evidence") or () if _text(cited)]
        if kind == "program":
            # PRESENT and EMPTY: a program idea names no measurable.
            measurable = ""
        normal = normalise_text(text)
        if normal in thrown:
            counts["dismissed"] += 1
            continue
        if normal in minted:
            _drop(DROP_REPEATED_IN_ANSWER)
            continue
        minted.add(normal)
        earlier = seen.get(normal)
        if earlier is not None:
            counts["repeats"] += 1
            rows.append(
                {
                    "idea_id": _text(earlier.get("idea_id")),
                    "session_date": session,
                    "kind": _text(earlier.get("kind")) or kind,
                    "text": _text(earlier.get("text")) or text,
                    "measurable": _text(earlier.get("measurable")) or measurable,
                    "evidence": evidence,
                    "first_seen": _text(earlier.get("first_seen"))
                    or _text(earlier.get("session_date")),
                    "seen_count": int(earlier.get("seen_count") or 1) + 1,
                    "created_at": stamp,
                    "prompt_version": PROMPT_VERSION,
                    "model": model,
                    "inputs_hash": inputs_hash,
                }
            )
            continue
        rows.append(
            {
                "idea_id": mint_idea_id(session, text),
                "session_date": session,
                "kind": kind,
                "text": text,
                "measurable": measurable,
                "evidence": evidence,
                "first_seen": session,
                "seen_count": 1,
                "created_at": stamp,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "inputs_hash": inputs_hash,
            }
        )
    # Per REASON, so the ledger row says WHICH two were dropped and not just
    # that two were (reviewer advisory 2). Present and empty when none were.
    counts["drop_reasons"] = dict(sorted(reasons.items()))
    return rows, counts


def _offer_counts(summary: Any) -> dict[str, Any]:
    """What a REJECTED answer held, counted. Never raises - it is a post-mortem.

    A rejection stores nothing, so `usable_ideas` never runs and its counts do
    not exist; these are read straight off the reply so the ledger row says how
    many ideas came back and how many of them were usable.
    """
    rows = summary.get("ideas") if isinstance(summary, Mapping) else None
    rows = list(rows) if isinstance(rows, (list, tuple)) else []
    reasons: dict[str, int] = {}
    usable = 0
    for row in rows:
        code = drop_reason(row)
        if code:
            reasons[code] = reasons.get(code, 0) + 1
        else:
            usable += 1
    return {
        "offered": len(rows),
        "usable": usable,
        "stored": 0,
        "rejected": True,
        "drop_reasons": dict(sorted(reasons.items())),
    }


def _append(rows: Sequence[Mapping[str, Any]]) -> Path:
    """Append-only, one JSON object per line. The earlier lines are untouched."""
    path = _ideas_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
    return path


def read_asked_marker() -> dict[str, Any]:
    """The record of the last night that ASKED. ``{}`` when there is none."""
    payload = _read_json(_asked_path())
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_asked_marker(
    session: str, inputs_hash: str, *, model: str, stored: int, counts: Mapping[str, Any], now
) -> Path | None:
    """Record that tonight was ASKED. Temp-and-rename, one file, superseding.

    Never fails the night: the runner's own already-done check is the first
    guard and this is the second, so a marker that could not be written costs a
    belt and keeps the braces.
    """
    path = _asked_path()
    payload = {
        "schema": ASKED_MARKER_SCHEMA,
        "session_date": session,
        "inputs_hash": inputs_hash,
        "prompt_version": PROMPT_VERSION,
        "asked_at": _moment(now),
        "model": model,
        "stored": int(stored),
        "counts": dict(counts),
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        _log.debug("The asked marker could not be written.", exc_info=True)
        # Review round 2: a failed rename left the temp file beside the store.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return path


def _already_tonight(rows: Sequence[Mapping[str, Any]], session: str, inputs_hash: str) -> bool:
    """Has tonight's evidence already been asked about?

    Two answers, because a night that STORED something and a night that stored
    nothing both count as asked: a stored row carries the hash it was minted
    from, and a night with nothing to store leaves the marker instead.
    """
    if any(
        _text(row.get("session_date"))[:10] == session
        and _text(row.get("inputs_hash")) == inputs_hash
        and _text(row.get("prompt_version")) == PROMPT_VERSION
        for row in rows
    ):
        return True
    marker = read_asked_marker()
    return (
        _text(marker.get("session_date"))[:10] == session
        and _text(marker.get("inputs_hash")) == inputs_hash
        and _text(marker.get("prompt_version")) == PROMPT_VERSION
    )


def run_improvement_ideas(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    ledger_path: Any = None,
    force: bool = False,
    **_ignored: Any,
) -> dict[str, Any]:
    """One night's ideas. Never raises: a crash in the last slot is a lost night.

    ``force`` re-spends the unchanged-hash skip and nothing else. It does not buy
    the night window - that is the runner's gate (TJ-13A item 1), and this slot
    declares ``uses_model`` so a forced daytime run records SKIPPED and loads
    nothing.

    It writes ONE file, the ideas store, and never the state file: the night
    proposes, and only the trader's card disposes.
    """
    base = _root(root)
    session = _text(session_date)[:10] or datetime.now().date().isoformat()
    try:
        inputs = build_ideas_inputs(session, root=base, ledger_path=ledger_path)
    except Exception as exc:  # noqa: BLE001 - an unreadable night is a recorded row
        _log.debug("The night's ideas inputs could not be read.", exc_info=True)
        return {
            "status": ledger.STATUS_FAILED,
            "model": "",
            "reason": f"tonight's evidence could not be read: {exc}",
            "outputs": [],
        }

    stored = read_ideas()
    digest = _text(inputs.get("inputs_hash"))
    if not force and _already_tonight(stored, session, digest):
        return {
            "status": ledger.STATUS_OK,
            "model": "",
            "reason": f"tonight's ideas for {session} are unchanged; no model was asked",
            "outputs": [str(_ideas_path())],
        }

    # NOTHING TO CITE -> NO MODEL LOAD. Every idea must cite an id the night
    # carries, so a window with no packs in it can produce nothing but a whole
    # rejection - and the honest answer is "tonight carries nothing to cite",
    # not "the model lied" (reviewer advisory 4, 2026-09-20). This is the state
    # the desk is in today: the live day-review folder holds zero packs. A
    # pre-model skip repeating every pass costs a ledger row and a few file
    # reads, which is what `week_review_narration`'s own floor branch costs.
    if not list(inputs.get("allowed_source_ids") or ()):
        with_facts = len(list(inputs.get("sessions_with_facts") or ()))
        total = len(list(inputs.get("sessions") or ()))
        return {
            "status": ledger.STATUS_SKIPPED,
            "model": "",
            "reason": (
                f"tonight carries nothing to cite: {with_facts} of {total} session(s) "
                f"ending {session} have facts, so no model was loaded"
            ),
            "outputs": [],
            "extra": {"sessions": total, "sessions_with_facts": with_facts, "asked": False},
        }

    schema = schema_for(inputs)
    evidence = build_evidence(inputs)
    if request is None:
        import ai_summary

        request = ai_summary.request_ai_summary
        model = ai_summary.local_model("medium")
    else:
        try:
            import ai_summary

            model = ai_summary.local_model("medium")
        except Exception:  # noqa: BLE001 - a test's request needs no configured model
            model = ""

    try:
        result = request(
            provider="local",
            model=model,
            api_key="",
            evidence=evidence,
            timeout_seconds=TIMEOUT_SECONDS,
            schema=schema,
            schema_name=SCHEMA_NAME,
            prompt_version=PROMPT_VERSION,
        )
    except Exception as exc:  # noqa: BLE001 - the store is untouched
        _log.debug("The ideas slot could not ask its model.", exc_info=True)
        return {
            "status": ledger.STATUS_DEGRADED,
            "model": "",
            "reason": f"no local model answered tonight's ideas: {exc}",
            "outputs": [],
        }

    answered = _text((result or {}).get("model")) or model
    try:
        body = _validate((result or {}).get("summary"), schema, name="ideas")
        check_ideas(body, inputs)
    except Exception as exc:  # noqa: BLE001 - a breach rejects the answer WHOLE
        _log.debug("Tonight's ideas were rejected.", exc_info=True)
        # A rejection is an ATTEMPT (`ledger.ATTEMPT_STATUSES`), so it is capped
        # at `max_attempts` by the runner and leaves NO marker - the next pass
        # is allowed to try again. Its counts travel with it, because "rejected"
        # without what it held is a post-mortem nobody can do (advisory 2).
        return {
            "status": ledger.STATUS_FAILED,
            "model": "",
            "reason": f"tonight's ideas were rejected and nothing was stored: {exc}",
            "outputs": [],
            "extra": _offer_counts((result or {}).get("summary")),
        }

    dismissed = _dismissed_forms(stored, read_state())
    rows, counts = usable_ideas(
        list(body.get("ideas") or ()),
        session=session,
        stored=stored,
        dismissed=sorted(dismissed),
        now=now,
        model=answered,
        inputs_hash=digest,
    )
    said = (
        f"{counts['offered']} idea(s) offered, {len(rows)} stored, "
        f"{counts['dropped']} dropped, {counts['dismissed']} already dismissed, "
        f"{counts['repeats']} seen before"
    )
    if not rows:
        # ASKED ONCE IS DONE. The model was loaded and answered; that this
        # night had nothing worth keeping is a finished night, not an unfinished
        # one, and `ok` is the only status the runner's already-done check
        # understands (`ledger.CANONICAL_COMPLETION_STATUSES`). The marker is
        # the artifact, so the slot's own unchanged-hash skip arms too.
        marker = _write_asked_marker(
            session, digest, model=answered, stored=0, counts=counts, now=now
        )
        return {
            "status": ledger.STATUS_OK,
            "model": answered,
            "reason": (
                f"asked once: 0 of {counts['offered']} ideas kept for {session} - {said}"
            ),
            "outputs": [str(marker)] if marker else [],
            "extra": dict(counts),
        }
    try:
        path = _append(rows)
    except OSError as exc:
        _log.debug("The night's ideas could not be appended.", exc_info=True)
        return {
            "status": ledger.STATUS_FAILED,
            "model": answered,
            "reason": f"the ideas store could not be written: {exc}",
            "outputs": [],
            "extra": dict(counts),
        }
    _write_asked_marker(
        session, digest, model=answered, stored=len(rows), counts=counts, now=now
    )
    return {
        "status": ledger.STATUS_OK,
        "model": answered,
        "reason": f"{said} for {session}",
        "outputs": [str(path)],
        "extra": dict(counts),
    }


__all__ = [
    "DEDUPE_SESSIONS",
    "EVIDENCE_KEYS",
    "IDEAS_JSON_SCHEMA",
    "IDEAS_PROGRAM_CARD",
    "IDEAS_SESSIONS",
    "IDEA_KINDS",
    "DROP_CODES",
    "IdeasRejected",
    "MAX_EVIDENCE_PER_IDEA",
    "MAX_IDEAS_PER_NIGHT",
    "MAX_IDEA_CHARS",
    "MEASURABLES",
    "Measurable",
    "PROGRAM_CARD_VERSION",
    "PROMPT_VERSION",
    "RESERVE_MINUTES",
    "SCHEMA",
    "SCHEMA_NAME",
    "STATUS_DISMISSED",
    "STATUS_KEPT",
    "TIMEOUT_SECONDS",
    "build_evidence",
    "build_ideas_inputs",
    "check_ideas",
    "checked_ideas",
    "dismiss_idea",
    "ideas_for_session",
    "keep_idea",
    "measurable_named",
    "measurable_names",
    "measure",
    "mint_idea_id",
    "normalise_text",
    "read_ideas",
    "read_state",
    "run_improvement_ideas",
    "schema_for",
    "sessions_ending",
    "source_id",
    "usable_ideas",
    "with_interval",
]
