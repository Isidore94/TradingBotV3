r"""Hand-built fixtures for the TJ-7 red tests. NOT a test module.

`plan.md` §12.4 "TJ-7 - Mood and process: the bones only" as AMENDED 2026-09-19,
packet `.claude/packets/TJ-5-6-7-13B.md` ("TJ-7 - mood and process fields").

Nothing here is produced by the code under test. Every mood row below is a
DICT LITERAL in the shape the tester pinned, so a builder who writes a
different shape fails on the shape and not on a fixture that followed them.
The one value read out of the owner is the state-tag vocabulary's
``vocab_version``, because a literal version in a test is forbidden
(CLAUDE.md, "Never assert a literal `vocab_version` in a test").

WHAT THE LIVE JOURNAL LOOKS LIKE TODAY (read-only count, 2026-09-20)
--------------------------------------------------------------------
``C:\TradingBotData\data\runtime\evidence_ledgers\market_journal-2026{08,09}.jsonl``

* **84 rows** in all - 15 in August, 69 in September.
* **41 rows carry a `mentor` key**: 13 of them PRESENT and EMPTY (`"mentor": {}`)
  and 28 carrying a real prompt block (`slot_id` / `prompt_kind` /
  `mentor_question`).
* **43 rows carry NO `mentor` key at all** - the key is ABSENT, which is what a
  row written before WISHLIST 10J looks like.
* **ZERO rows carry `mood`, `state_tags` or `followed_plan`.** That is the state
  TJ-7 starts from, and it is why :data:`NO_MOOD_YET_N` is 0: the first thing
  the trader sees must read "no mood recorded yet", with no invented percentage.

So every reader is fed BOTH absences here: the key missing (an old row) and the
key present and empty (a row written after TJ-7 ships with nothing clicked).

THE SESSION
-----------
`tj4_support.SESSION` - Friday **2026-09-18**, a real regular-close session
(13:00 Pacific). The stamps below are Pacific and hand-chosen:

    06:55  before the open        - a mood the 07:02 read COULD have known
    07:02  the read being graded  - `tj10_support.STAMP`
    08:55  mid-session
    12:55  five minutes before the close, and after the 07:02 read

THE PINNED CONTRACT (the builder may ADD keys, never remove one)
---------------------------------------------------------------
``scripts/trader_state_tags.py`` (new, pure, import-light)
    ``VOCABULARY_FAMILY = "state_tags"``, ``MAX_STATE_TAGS = 2``,
    ``load_vocabulary(*, directory=None) -> {"vocabulary_id", "vocab_version",
    "entries": [{"code", "label", "hint"}]}``, ``codes()``,
    ``StateTagError`` on a missing / malformed / mis-versioned file.
    Asset: ``scripts/ui/annotations/vocabularies/state_tags_v1.json``.

``scripts/market_journal.py``
    ``MOOD_SCHEMA = "trader_mood_v1"``, ``MOOD_SCALE = (1, 2, 3, 4, 5)``,
    ``FOLLOWED_PLAN_VALUES = ("yes", "partly", "no")``,
    ``PROCESS_NOTE_MAX = 200``, ``class MoodFieldError(ValueError)``,
    ``build_mood(*, score=None, state_tags=(), followed_plan="", note="")``,
    ``build_entry(..., mood=None, state_tags=(), process=None)`` storing ONE
    key ``mood`` - the block, or ``{}`` when nothing was clicked - and
    ``mood_of(entry) -> Mood | None`` / ``mood_at(entries, stamp) -> Mood | None``
    as its ONE readers.

``scripts/day_review_pack.py``
    the ``mood`` section filled from the session's entries, ``{}`` when there
    is none, ``MOOD_EMPTY_STATEMENT`` and ``mood_statement(pack)``.

``scripts/market_read_grades.py``
    ``context_for(..., mood_entries=())`` -> context key ``mood``, the score
    known AT THE STAMP or ``UNMEASURED``.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj4_support as fx4  # noqa: E402
import tj10_support as fx10  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")

SESSION = fx4.SESSION
#: The 07:02 Pacific stamp every TJ-10 read in these fixtures was made at.
READ_STAMP = fx10.STAMP

#: How many moods the live desk holds today. Zero, and the honest first state
#: says so without inventing a percentage.
NO_MOOD_YET_N = 0

#: The eight codes `plan.md` TJ-7 change 1 names, in its order. They are the
#: CONTENT of `state_tags_v1.json`; the version that ships them is read from
#: the file, never written here.
STATE_TAG_CODES = (
    "calm",
    "focused",
    "rushed",
    "fomo",
    "tilted",
    "bored",
    "tired",
    "confident",
)

#: The three answers to `Followed the plan:` plus the fourth state, "nothing
#: clicked", which is stored as an empty string and is never a default "yes".
FOLLOWED_PLAN = ("yes", "partly", "no")

#: The stored block's own name. A schema id is a permanent identifier and is
#: asserted by name; a VOCABULARY VERSION never is.
MOOD_SCHEMA = "trader_mood_v1"

#: The thirteen keys `build_entry` wrote at `e00520a3`, hand-read off
#: `scripts/market_journal.py:317-344`. TJ-7 adds exactly one: `mood`.
ENTRY_KEYS_BEFORE_TJ7 = (
    "event_type",
    "entry_id",
    "session_date",
    "created_at",
    "created_local_date",
    "written_after_the_session",
    "timeframe",
    "symbols",
    "origin",
    "text",
    "supersedes",
    "mentor",
    "reaffirms",
)


def pacific(hour: int, minute: int = 0, *, session: str = SESSION) -> datetime:
    """An aware Pacific stamp inside `session`'s day."""
    day = datetime.fromisoformat(str(session)[:10])
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=PACIFIC)


VOCABULARY_DIR = SCRIPTS_DIR / "ui" / "annotations" / "vocabularies"


def shipped_vocab_version() -> int | None:
    """The newest `state_tags_v<N>.json` present, read off its FILENAME.

    Read from the ASSET, never from the loader: a fixture must not be built by
    the code under test, and a literal version in a test is forbidden. `None`
    means the vocabulary has not shipped yet, and a row hand-built before it
    ships carries no `vocab_version` at all - which is itself a shape every
    reader has to tolerate.
    """
    import re

    found = [
        int(match.group(1))
        for match in (
            re.fullmatch(r"state_tags_v(\d+)\.json", path.name)
            for path in VOCABULARY_DIR.glob("state_tags_v*.json")
        )
        if match
    ]
    return max(found) if found else None


def current_vocab_version() -> int:
    """The shipped state-tag version, READ from the owner. Never a literal."""
    import trader_state_tags

    return int(trader_state_tags.load_vocabulary()["vocab_version"])


def mood_block(
    *,
    score: int | None = 3,
    state_tags: Iterable[str] = ("rushed",),
    followed_plan: str = "partly",
    note: str = "",
    vocab_version: int | None = None,
) -> dict[str, Any]:
    """The stored `mood` block, written by hand in the pinned shape."""
    block: dict[str, Any] = {
        "schema": MOOD_SCHEMA,
        "score": score,
        "state_tags": [str(code) for code in state_tags],
        "process": {"followed_plan": str(followed_plan or ""), "note": str(note or "")},
    }
    stamped = shipped_vocab_version() if vocab_version is None else int(vocab_version)
    if stamped is not None:
        block["vocab_version"] = stamped
    return block


def _base_row(
    *,
    entry_id: str,
    stamp: datetime,
    text: str,
    session: str = SESSION,
    timeframe: str = "M5",
    origin: str = "trade_mentor",
) -> dict[str, Any]:
    """A journal row in the shape `build_entry` writes it, by hand."""
    created = stamp.astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds")
    return {
        "event_type": "entry",
        "entry_id": entry_id,
        "session_date": session,
        "created_at": created,
        "created_local_date": stamp.astimezone().date().isoformat(),
        "written_after_the_session": False,
        "timeframe": timeframe,
        "symbols": [],
        "origin": origin,
        "text": text,
        "supersedes": "",
        "mentor": {},
        "reaffirms": "",
    }


def row_with_a_mood(
    *,
    entry_id: str = "mj-mood-1",
    stamp: datetime | None = None,
    score: int | None = 3,
    state_tags: Iterable[str] = ("rushed",),
    followed_plan: str = "partly",
    note: str = "",
    text: str = "Followed the plan? partly",
    after_the_session: bool = False,
    session: str = SESSION,
) -> dict[str, Any]:
    """One row carrying a mood block, hand-built."""
    row = _base_row(
        entry_id=entry_id,
        stamp=stamp or pacific(12, 55, session=session),
        text=text,
        session=session,
    )
    row["written_after_the_session"] = bool(after_the_session)
    row["mood"] = mood_block(
        score=score, state_tags=state_tags, followed_plan=followed_plan, note=note
    )
    return row


def row_with_the_key_present_and_empty(
    *, entry_id: str = "mj-empty-mood", stamp: datetime | None = None
) -> dict[str, Any]:
    """A row written AFTER TJ-7 ships with nothing clicked: `mood` is `{}`.

    The live journal's own precedent is `"mentor": {}` on 13 September rows.
    """
    row = _base_row(
        entry_id=entry_id,
        stamp=stamp or pacific(8, 55),
        text="Nothing to add.",
    )
    row["mood"] = {}
    return row


def old_row_without_the_key(
    *, entry_id: str = "mj-old", stamp: datetime | None = None
) -> dict[str, Any]:
    """A row written BEFORE TJ-7: the key is ABSENT (43 of the 84 live rows)."""
    return _base_row(
        entry_id=entry_id,
        stamp=stamp or pacific(7, 30),
        text="Sitting on my hands until the range breaks.",
        origin="journal_page",
    )


def three_kinds_of_absence() -> tuple[dict[str, Any], ...]:
    """The three rows that all mean "no mood was recorded"."""
    absent = old_row_without_the_key(entry_id="mj-absent")
    empty = row_with_the_key_present_and_empty(entry_id="mj-empty")
    nulled = old_row_without_the_key(entry_id="mj-null")
    nulled["mood"] = None
    return (absent, empty, nulled)


def day_close_subject(*, session: str = SESSION):
    """The registry's own `day_close` subject for `session` - never a literal."""
    import mentor_questions

    from datetime import date as _date

    day = _date.fromisoformat(str(session)[:10])
    subjects = mentor_questions._trigger_day_close(
        {"session": session, "slot": _last_slot(day)}
    )
    assert subjects, f"the registry offers no day_close subject on {session}"
    return subjects[0]


def _last_slot(day):
    from trade_mentor_schedule import slots_for_session

    slots = slots_for_session(day)
    assert slots, f"{day} carries no Mentor slots"
    return slots[-1]


class FakeJournal:
    """A stand-in for `shared_journal_service()`. Records what it was handed."""

    def __init__(self, *, ok: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        self.ok = ok

    def write_entry(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if not self.ok:
            return {"ok": False, "reason": "refused by the fake journal"}
        import market_journal

        entry = dict(kwargs)
        entry.setdefault("entry_id", f"mj-fake-{len(self.calls)}")
        entry.setdefault("mood", {})
        entry["schema"] = market_journal.SCHEMA_MARKET_JOURNAL_ENTRY
        return {"ok": True, "entry": entry}

    @property
    def last(self) -> Mapping[str, Any]:
        assert self.calls, "the journal was never written to"
        return self.calls[-1]
