r"""TJ-6 - Keep and Dismiss are the TRADER's writes, and nothing else writes them. RED.

`plan.md` §12.4 TJ-6 change 2: ``AI_IDEAS_STATE_FILE`` holds
``{idea_id: {status: kept|dismissed, at}}`` - *"the trader's clicks, the only
writer is the card"*.

The house rules this pins (packet TJ-5 "CORRECTED" item 6, and CLAUDE.md):

* a write the trader asks for VALIDATES its argument and FAILS CLOSED - it
  raises rather than filing a keep for an idea nobody has;
* every other entry survives a write byte-identical, and the write is
  temp-and-rename, so a killed desk cannot leave half a state file;
* no nightly job ever writes a keep: the night proposes, the trader disposes.
  `plan.md` sec 5's ask-first line and decision 0021 answer 23 both rest on
  that, and this file proves it structurally AND by running the slot over a
  state file that must come out unchanged.

**NO MODEL IS EVER CALLED HERE.**

VERIFIED ON THIS BRANCH (1b9d77e0): `ai_jobs.improvement_ideas` does not exist,
so every test below fails on the import.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj6_support as fx  # noqa: E402

TEXT = "Wait for the second test before sizing up."
WRITERS = ("keep_idea", "dismiss_idea")


@pytest.fixture
def night(tmp_path, monkeypatch):
    return fx.install_stores(monkeypatch, tmp_path)


def _one_idea(night, **kwargs):
    row = fx.stored_idea_row(TEXT, session=fx.SESSION, **kwargs)
    fx.write_ideas(night["ideas"], [row])
    return row


# ---------------------------------------------------------------------------
# the write validates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "idea:2026-01-01:nothinghere"])
def test_keeping_an_idea_nobody_has_raises_and_writes_nothing(night, bad):
    """Fails CLOSED: the click is refused and no state file appears.

    A keep filed against an id with no row behind it would be a baseline over a
    measurable nobody named.
    """
    from ai_jobs import improvement_ideas

    _one_idea(night)
    assert not night["state"].exists()
    with pytest.raises((KeyError, ValueError)):
        improvement_ideas.keep_idea(bad, end_session=fx.SESSION)
    assert not night["state"].exists()


def test_dismissing_an_idea_nobody_has_raises_and_writes_nothing(night):
    from ai_jobs import improvement_ideas

    _one_idea(night)
    with pytest.raises((KeyError, ValueError)):
        improvement_ideas.dismiss_idea("idea:2026-01-01:nothinghere")
    assert not night["state"].exists()


def test_a_status_the_state_does_not_know_is_refused(night):
    """Only the two statuses the packet names exist, and the writer is per
    status - there is no free-text status to file."""
    from ai_jobs import improvement_ideas

    assert improvement_ideas.STATUS_KEPT == "kept"
    assert improvement_ideas.STATUS_DISMISSED == "dismissed"
    row = _one_idea(night)
    improvement_ideas.dismiss_idea(row["idea_id"])
    stored = improvement_ideas.read_state()[row["idea_id"]]
    assert stored["status"] in (
        improvement_ideas.STATUS_KEPT,
        improvement_ideas.STATUS_DISMISSED,
    )


# ---------------------------------------------------------------------------
# the write keeps what was there
# ---------------------------------------------------------------------------
def test_a_second_click_leaves_the_first_record_exactly_as_it_was(night, monkeypatch):
    """Hand-counted: 2 entries afterwards, and entry one is byte-identical to
    what it was before entry two was written."""
    from ai_jobs import improvement_ideas

    first = fx.stored_idea_row(TEXT, session=fx.SESSION)
    second = fx.stored_idea_row(
        "Put the walk-away table beside the trades.",
        session=fx.SESSION,
        kind="program",
        measurable="",
    )
    fx.write_ideas(night["ideas"], [first, second])

    improvement_ideas.dismiss_idea(first["idea_id"])
    before = json.dumps(
        improvement_ideas.read_state()[first["idea_id"]], sort_keys=True
    ).encode("utf-8")

    improvement_ideas.keep_idea(second["idea_id"], end_session=fx.SESSION)
    state = improvement_ideas.read_state()
    assert len(state) == 2
    after = json.dumps(state[first["idea_id"]], sort_keys=True).encode("utf-8")
    assert after == before


def test_the_state_is_written_temp_and_rename_and_leaves_no_half_file(night, monkeypatch):
    """A killed desk must not leave a truncated state file where the card reads
    "nothing kept"."""
    import os

    from ai_jobs import improvement_ideas

    row = _one_idea(night)
    improvement_ideas.keep_idea(row["idea_id"], end_session=fx.SESSION)
    good = night["state"].read_bytes()
    assert not list(night["state"].parent.glob("*.tmp")), "a temp file was left behind"

    def _boom(*_args, **_kwargs):
        raise OSError("the disk went away mid-rename")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        improvement_ideas.dismiss_idea(row["idea_id"])
    assert night["state"].read_bytes() == good, "a failed write damaged the state"


# ---------------------------------------------------------------------------
# no job ever writes a keep
# ---------------------------------------------------------------------------
def test_the_nights_run_leaves_the_state_file_byte_identical(night):
    """The night proposes; only the card disposes.

    The state file already holds one kept and one dismissed idea. A night that
    wrote to it - to "confirm" its own advice, to expire a keep, to grade
    anything - would change these bytes.
    """
    from ai_jobs import improvement_ideas

    row = _one_idea(night)
    fx.write_state(
        night["state"],
        {
            row["idea_id"]: fx.kept_record(row["idea_id"]),
            "idea:2026-09-01:abcabcabcabc": fx.dismissed_record("idea:2026-09-01:abcabcabcabc"),
        },
    )
    before = night["state"].read_bytes()

    inputs = improvement_ideas.build_ideas_inputs(fx.SESSION, root=night["root"])
    allowed = list(inputs["allowed_source_ids"])
    measurable = str(improvement_ideas.MEASURABLES[0].name)
    improvement_ideas.run_improvement_ideas(
        session_date=fx.SESSION,
        now=fx.WEEKNIGHT,
        root=night["root"],
        request=fx.fake_request(
            fx.reply(
                [
                    fx.idea_payload(
                        "A brand new thought.", measurable=measurable, evidence=allowed[:1]
                    )
                ]
            )
        ),
    )
    assert night["state"].read_bytes() == before


def test_no_nightly_job_calls_the_keep_writers():
    """Structural, over the source of every `ai_jobs` module (TJ-12's idiom).

    `run_improvement_ideas` and every sibling slot must be unable to reach
    `keep_idea` / `dismiss_idea`; the only callers in the repository are the
    card and this program's tests.
    """
    from ai_jobs import improvement_ideas

    jobs_dir = ROOT_DIR / "scripts" / "ai_jobs"
    offenders: list[str] = []
    for path in sorted(jobs_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in WRITERS:
                offenders.append(f"{path.name}:{node.lineno} calls {name}")
    assert offenders == [], offenders

    # And the run function's own body, read directly, names neither.
    body = inspect.getsource(improvement_ideas.run_improvement_ideas)
    for name in WRITERS:
        assert name not in body, f"the night's run mentions {name}"
