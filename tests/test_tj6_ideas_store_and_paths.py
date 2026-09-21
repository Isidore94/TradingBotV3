r"""TJ-6 - the two ideas stores, and the honest empty state (RED).

`plan.md` §12.4 TJ-6 changes 1 and 2: the night's ideas are appended to
``AI_IDEAS_FILE`` (``PERSISTENT_DATA_DIR / "ai_ideas.jsonl"``) and the trader's
Keep / Dismiss clicks live in ``AI_IDEAS_STATE_FILE``
(``ai_ideas_state.json``), whose *"only writer is the card"*.

VERIFIED ON THIS BRANCH (1b9d77e0, 2026-09-20)
----------------------------------------------
* `scripts/project_paths.py` holds NO ``AI_IDEAS`` name of any kind.
* There is no `scripts/ai_jobs/improvement_ideas.py`; the string
  ``improvement_ideas`` appears only at `plan.md:746` and `plan.md:1246`.
* `C:\TradingBotData` holds no file matching ``*idea*`` (listed read-only).

So every test here fails today on the missing module or the missing constant.

WHY THE STORE IS APPEND-ONLY AND FOLDED
---------------------------------------
TJ-6 change 1 says a repeat *"increments `seen_count` instead"* of appending a
new idea. A JSONL store that REWROTE a row to do that would lose the earlier
event, which CLAUDE.md forbids ("evidence stores are never allowed to cost the
thing they record"). The reading pinned here - and the one the builder may not
weaken - is: the file is APPEND-ONLY, a repeat appends a row carrying the SAME
``idea_id`` with a higher ``seen_count``, the earlier line stays byte-identical,
and :func:`read_ideas` FOLDS by ``idea_id`` keeping the LAST row. One idea, one
id, every sighting still on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj6_support as fx  # noqa: E402


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Scratch ideas stores, installed on `project_paths` itself.

    Nothing is passed as an argument: the point of setting them HERE is that a
    module which resolved the path at IMPORT time (``from project_paths import
    AI_IDEAS_FILE``) would keep pointing at the trader's own folder. Every
    reader and writer must resolve at CALL time.
    """
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    ideas = tmp_path / "ai_ideas.jsonl"
    state = tmp_path / "ai_ideas_state.json"
    monkeypatch.setattr(project_paths, "AI_IDEAS_FILE", ideas, raising=False)
    monkeypatch.setattr(project_paths, "AI_IDEAS_STATE_FILE", state, raising=False)
    return {"ideas": ideas, "state": state}


def test_the_two_ideas_paths_are_named_in_project_paths():
    """`plan.md` TJ-6 names both files and the folder they live in."""
    import project_paths

    assert project_paths.AI_IDEAS_FILE.name == "ai_ideas.jsonl"
    assert project_paths.AI_IDEAS_STATE_FILE.name == "ai_ideas_state.json"
    # The shared home folder, beside the other trader-facing stores - never a
    # per-machine cache: a kept idea is the trader's, not this machine's.
    assert project_paths.AI_IDEAS_FILE.parent == project_paths.PERSISTENT_DATA_DIR
    assert project_paths.AI_IDEAS_STATE_FILE.parent == project_paths.PERSISTENT_DATA_DIR


def test_no_ideas_yet_reads_as_no_ideas_and_never_as_a_zero(stores):
    """The state the trader is in TODAY: both files absent.

    Hand-counted: zero rows on disk, so zero ideas and zero kept - and the
    readers must SAY that with empty containers rather than raising or inventing
    a row. `C:\\TradingBotData` really holds neither file (listed 2026-09-20).
    """
    from ai_jobs import improvement_ideas

    assert not stores["ideas"].exists()
    assert not stores["state"].exists()
    assert tuple(improvement_ideas.read_ideas()) == ()
    assert dict(improvement_ideas.read_state()) == {}
    # A reader that CREATED its store in order to answer would put an empty file
    # in the trader's home folder on every desk launch.
    assert not stores["ideas"].exists()
    assert not stores["state"].exists()


def test_the_readers_resolve_their_path_at_call_time(stores, tmp_path):
    """Move the constant, and the next read follows it.

    A module-level ``from project_paths import AI_IDEAS_FILE`` passes every
    other test in this file and fails this one - which is the 2026-09-05 and
    2026-09-20 incident in miniature: an alias of a path is not the path.
    """
    import project_paths
    from ai_jobs import improvement_ideas

    fx.write_ideas(stores["ideas"], [fx.stored_idea_row("Stop chasing the open.")])
    assert len(improvement_ideas.read_ideas()) == 1

    moved = tmp_path / "moved" / "ai_ideas.jsonl"
    fx.write_ideas(
        moved,
        [
            fx.stored_idea_row("Stop chasing the open."),
            fx.stored_idea_row("Size the second entry smaller.", session="2026-09-17"),
        ],
    )
    project_paths.AI_IDEAS_FILE = moved
    assert len(improvement_ideas.read_ideas()) == 2


def test_two_rows_with_one_id_fold_into_one_idea_with_the_later_count(stores):
    """Hand-counted: 3 lines on disk, 2 distinct ids, so 2 folded ideas.

    The repeat carries ``seen_count`` 2 and the FIRST row's ``first_seen``; the
    fold keeps the LAST row, which is the one the card shows.
    """
    text = "Stop chasing the open."
    first = fx.stored_idea_row(text, session="2026-09-16", seen_count=1)
    repeat = fx.stored_idea_row(
        text,
        session="2026-09-18",
        seen_count=2,
        first_seen="2026-09-16",
        idea_id=fx.idea_id_for("2026-09-16", text),
    )
    other = fx.stored_idea_row("Size the second entry smaller.", session="2026-09-18")
    fx.write_ideas(stores["ideas"], [first, repeat, other])

    from ai_jobs import improvement_ideas

    folded = list(improvement_ideas.read_ideas())
    assert len(folded) == 2
    by_id = {row["idea_id"]: row for row in folded}
    kept = by_id[fx.idea_id_for("2026-09-16", text)]
    assert kept["seen_count"] == 2
    assert kept["first_seen"] == "2026-09-16"
    assert kept["session_date"] == "2026-09-18"


def test_an_old_row_with_empty_fields_is_read_not_skipped(stores):
    """An old row has its keys PRESENT and EMPTY (`docs/AGENT_TEAM.md`).

    Hand-counted: 2 rows, one of them written before the measurable field
    carried anything. Both come back; the old one is simply not a `process`
    idea anybody can check.
    """
    fx.write_ideas(
        stores["ideas"],
        [
            fx.legacy_idea_row("Read the tape before the first trade."),
            fx.stored_idea_row("Stop chasing the open.", measurable="x"),
        ],
    )

    from ai_jobs import improvement_ideas

    rows = list(improvement_ideas.read_ideas())
    assert len(rows) == 2
    old = next(row for row in rows if row["session_date"] == "2026-06-01")
    assert old["measurable"] == ""
    assert old["evidence"] == []


def test_a_half_written_line_costs_that_line_and_not_the_store(stores):
    """A torn last line is a lost row, never a lost store.

    Hand-counted: 2 good lines plus one truncated one -> 2 ideas.
    """
    fx.write_ideas(
        stores["ideas"],
        [
            fx.stored_idea_row("Stop chasing the open.", session="2026-09-16"),
            fx.stored_idea_row("Size the second entry smaller.", session="2026-09-17"),
        ],
    )
    with stores["ideas"].open("a", encoding="utf-8") as handle:
        handle.write('{"idea_id": "idea:2026-09-18:aaaaaaaa", "text": "tor')

    from ai_jobs import improvement_ideas

    assert len(list(improvement_ideas.read_ideas())) == 2
