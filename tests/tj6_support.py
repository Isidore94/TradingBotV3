r"""Hand-built fixtures for the TJ-6 (the AI's ideas) red tests. NOT a test module.

Nothing here is produced by the code under test: `scripts/ai_jobs/improvement_ideas.py`
and the ideas card do not exist on this branch (verified 2026-09-20 at
`1b9d77e0`: no `AI_IDEAS_*` name in `scripts/project_paths.py`, no
`improvement_ideas` anywhere but two lines of `plan.md`). Every stored row below
is typed out by hand in the shape the packet describes, and every expected
number in a test is hand-counted in that test's own docstring.

WHAT THE TRADER SEES FIRST
--------------------------
The live home folder holds NO ideas store at all (`C:\TradingBotData`, listed
read-only 2026-09-20: no file matching ``*idea*``, three session folders under
``day_review\sessions`` and ZERO ``pack.json``). So the first honest state is
"no ideas yet" and "kept 0 of 0" - never a zero that reads as a measurement.
:data:`EMPTY_STATE_COUNTS` is that state, hand-written.

THE WEEK is TJ-5's: Monday 2026-09-14 to Friday 2026-09-18, five real regular
sessions, three of which have packs. `tj5_support` builds them through TJ-4's
already-merged `day_review_pack.build_pack` (the OLD code TJ-6 only READS), so
this module never re-invents a pack.

WHAT IS MODELLED AS IT REALLY IS
--------------------------------
* A `program` idea carries ``measurable`` PRESENT and EMPTY (``""``) - never a
  missing key. Only a `process` idea names one.
* An OLD stored row (written before a later field existed) carries that field
  present and empty too: :func:`legacy_idea_row`.
* The store is JSONL and APPEND-ONLY. A repeat of an idea already seen inside
  the dedupe window appends a row with the SAME ``idea_id`` and a higher
  ``seen_count``; the earlier line stays byte-identical. `read_ideas` folds by
  ``idea_id``, LAST row wins.
* A dismissed idea lives in the STATE file, which is a mapping the trader's
  clicks own. A kept `process` idea carries the frozen baseline beside it.

THE HAND-COUNTED TABLES
-----------------------
Ideas offered by one night's reply (:func:`reply_with`)::

    #  kind      measurable                     evidence      verdict here
    1  process   <MEASURABLES[0].name>          1 allowed id   usable
    2  program   "" (present and empty)         1 allowed id   usable
    3  process   "not_a_measurable_the_desk_has" 1 allowed id  DROPPED (unknown measurable)
    4  process   <MEASURABLES[0].name>          []             DROPPED (no evidence)
    ------------------------------------------------------------------------
    usable = 2 of 4 offered

The kept-idea baseline (:data:`BASELINE`) and its after-window (:data:`AFTER`)::

    before   value 0.30   n 34     (34 >= MIN_REPORTABLE_N = 30, so reportable)
    after    value 0.48   n 41     (41 >= 30, so reportable)
    thin     value 0.50   n 29     = MIN_REPORTABLE_N - 1 -> "too few to call"

`evidence_stats.MIN_REPORTABLE_N` is 30 and `LATELY_SESSIONS` 20, both read back
out of `evidence_stats` by :func:`min_reportable_n` / :func:`lately_sessions`
rather than typed into a test.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj5_support as tj5  # noqa: E402 - the sys.path line above has to run first

#: TJ-5's week, reused rather than re-typed.
WEEK: tuple[str, ...] = tj5.WEEK
PACKED_SESSIONS: tuple[str, ...] = tj5.PACKED_SESSIONS
SESSION: str = tj5.FRIDAY
WEEK_ID: str = tj5.WEEK_ID

#: A weeknight inside the shipped 18:30-08:00 window, on the evening of a
#: session day - `runner.night_kind()`'s WEEKNIGHT.
WEEKNIGHT = datetime(2026, 9, 17, 23, 0, tzinfo=tj5.EASTERN)
#: A Friday AFTERNOON: outside the night window, the moment TJ-13A item 1 is
#: about.
FRIDAY_AFTERNOON = datetime(2026, 9, 18, 14, 0, tzinfo=tj5.EASTERN)

#: What the page says before anything exists. Hand-written, not computed.
EMPTY_STATE_COUNTS = {"kept": 0, "of": 0}

#: The baseline a Keep freezes, and the same measurable read again later. Both
#: `n` are ABOVE `evidence_stats.MIN_REPORTABLE_N` (30, measured 2026-09-20), so
#: this pair is reportable; :func:`thin_reading` is the one that is not.
BASELINE = {"value": 0.30, "n": 34}
AFTER = {"value": 0.48, "n": 41}


# ---------------------------------------------------------------------------
# identity - the id the code must mint, computed here from the definition
# ---------------------------------------------------------------------------
def normalised(text: Any) -> str:
    """The normal form this fixture expects: casefolded, punctuation-free, one
    space between words.

    Spelled out here rather than imported from the module under test, so a
    normaliser that folds two different sentences together fails instead of
    agreeing with itself.
    """
    body = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in str(text or "").casefold()
    )
    return " ".join(body.split())


def idea_id_for(session: Any, text: Any) -> str:
    """``idea:<session>:<sha1 of the normal form, 12 hex>``.

    The MODEL never names an idea. The id carries the session it was first seen
    in, so the same sentence returning after the dedupe window is a NEW idea
    rather than a resurrected one.
    """
    digest = hashlib.sha1(normalised(text).encode("utf-8")).hexdigest()[:12]
    return f"idea:{str(session)[:10]}:{digest}"


# ---------------------------------------------------------------------------
# the model's reply
# ---------------------------------------------------------------------------
def idea_payload(
    text: str,
    *,
    kind: str = "process",
    measurable: str = "",
    evidence: Sequence[str] = (),
) -> dict[str, Any]:
    """One idea as the MODEL returns it - no ``idea_id``, no counts, no grade."""
    return {
        "kind": kind,
        "text": text,
        "measurable": measurable,
        "evidence": list(evidence),
    }


def reply(ideas: Sequence[Mapping[str, Any]], *, model: str = "gemma3:12b-tbv3ctx-64k"):
    """A provider answer in `request(...)`'s own shape: ``summary`` plus a model."""
    return {"summary": {"ideas": [dict(item) for item in ideas]}, "model": model}


def fake_request(answer: Mapping[str, Any], *, calls: list | None = None):
    """A `request=` injection that returns `answer` and records its kwargs.

    No endpoint is contacted by any TJ-6 test; this is the only "model" they
    ever see.
    """

    def _request(**kwargs):
        if calls is not None:
            calls.append(dict(kwargs))
        return dict(answer)

    return _request


# ---------------------------------------------------------------------------
# the stores, written by hand
# ---------------------------------------------------------------------------
def stored_idea_row(
    text: str,
    *,
    session: str = SESSION,
    kind: str = "process",
    measurable: str = "",
    evidence: Sequence[str] = ("2026-09-18/card:did_well",),
    seen_count: int = 1,
    first_seen: str = "",
    idea_id: str = "",
) -> dict[str, Any]:
    """One row of `AI_IDEAS_FILE`, typed out - never produced by TJ-6's code."""
    return {
        "idea_id": idea_id or idea_id_for(session, text),
        "session_date": session,
        "kind": kind,
        "text": text,
        "measurable": measurable,
        "evidence": list(evidence),
        "first_seen": first_seen or session,
        "seen_count": int(seen_count),
        "created_at": f"{session}T23:40:00+00:00",
        "prompt_version": "improvement_ideas_v1",
        "model": "gemma3:12b-tbv3ctx-64k",
    }


def legacy_idea_row(text: str, *, session: str = "2026-06-01") -> dict[str, Any]:
    """An OLD row: every key PRESENT and EMPTY, the way a real old row is.

    `docs/AGENT_TEAM.md`: "Old rows have the key PRESENT and EMPTY, not absent."
    """
    row = stored_idea_row(text, session=session)
    row["measurable"] = ""
    row["evidence"] = []
    row["model"] = ""
    return row


def write_ideas(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """`AI_IDEAS_FILE` holding exactly `rows`, JSONL, in order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def install_stores(monkeypatch, tmp_path, *, write_the_week: bool = True) -> dict[str, Path]:
    """Point every TJ-6 store at `tmp_path` and (optionally) write TJ-5's week.

    The paths are set ON `project_paths`, never passed as arguments: a module
    that resolved them at import time would still be pointing at the trader's
    own home folder, which is the 2026-09-05 / 2026-09-20 incident.
    """
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = Path(tmp_path) / "day_review"
    ideas = Path(tmp_path) / "ai_ideas.jsonl"
    state = Path(tmp_path) / "ai_ideas_state.json"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    monkeypatch.setattr(project_paths, "AI_IDEAS_FILE", ideas, raising=False)
    monkeypatch.setattr(project_paths, "AI_IDEAS_STATE_FILE", state, raising=False)
    if write_the_week:
        write_week(root, narrated=PACKED_SESSIONS)
    return {"root": root, "ideas": ideas, "state": state}


def write_state(path: Path, state: Mapping[str, Any]) -> Path:
    """`AI_IDEAS_STATE_FILE` holding exactly `state`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def kept_record(
    idea_id: str,
    *,
    at: str = "2026-09-19T15:00:00+00:00",
    measurable: str = "",
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One `kept` entry of the state file, with its frozen baseline."""
    record: dict[str, Any] = {"status": "kept", "at": at}
    if measurable:
        record["baseline"] = {
            "measurable": measurable,
            "value": float((baseline or BASELINE)["value"]),
            "n": int((baseline or BASELINE)["n"]),
            "measured": True,
            "window_sessions": lately_sessions(),
            "at": at,
        }
    return record


def dismissed_record(idea_id: str, *, at: str = "2026-09-19T15:00:00+00:00"):
    return {"status": "dismissed", "at": at}


# ---------------------------------------------------------------------------
# numbers read back out of their owners, never typed
# ---------------------------------------------------------------------------
def lately_sessions() -> int:
    import evidence_stats

    return int(evidence_stats.LATELY_SESSIONS)


def min_reportable_n() -> int:
    import evidence_stats

    return int(evidence_stats.MIN_REPORTABLE_N)


def thin_reading(value: float = 0.50) -> dict[str, Any]:
    """A reading one row UNDER the floor - the "too few to call" side."""
    return {"value": float(value), "n": min_reportable_n() - 1}


def sessions_back(session: str, count: int) -> str:
    """The exchange session `count` sessions BEFORE `session`.

    Walked on the exchange calendar (`market_calendar.previous_session`), never
    in calendar days: a 60-session window counted in days would be a different
    window every holiday week.
    """
    import market_calendar

    cursor = date.fromisoformat(str(session)[:10])
    for _step in range(int(count)):
        cursor = market_calendar.previous_session(cursor)
    return cursor.isoformat()


# ---------------------------------------------------------------------------
# the week on disk, borrowed whole from TJ-5
# ---------------------------------------------------------------------------
def write_week(root: Path, **kwargs):
    """TJ-5's five-day week: packs, narrations and the ledger it reads."""
    return tj5.write_week(Path(root), **kwargs)


__all__ = [
    "AFTER",
    "BASELINE",
    "EMPTY_STATE_COUNTS",
    "FRIDAY_AFTERNOON",
    "PACKED_SESSIONS",
    "SESSION",
    "WEEK",
    "WEEKNIGHT",
    "WEEK_ID",
    "dismissed_record",
    "fake_request",
    "idea_id_for",
    "idea_payload",
    "install_stores",
    "kept_record",
    "lately_sessions",
    "legacy_idea_row",
    "min_reportable_n",
    "normalised",
    "reply",
    "sessions_back",
    "stored_idea_row",
    "thin_reading",
    "write_ideas",
    "write_state",
    "write_week",
]
