r"""TJ-6 - the same thought twice, and the one the trader threw away. RED.

`plan.md` §12.4 TJ-6 change 1: *"an idea whose normalised text matches one from
the last 60 sessions increments `seen_count` instead; a dismissed idea never
returns"*.

Two traps this file exists for:

1. **Sixty SESSIONS, walked on the exchange calendar.** Sixty calendar days is
   about forty-two sessions, so a window measured in days silently forgets
   eighteen sessions of ideas and starts repeating itself. The fixtures here are
   stamped by walking `market_calendar.previous_session`, never by subtracting
   days.
2. **A dismissal is about the IDEA, not the row.** The id carries the session an
   idea was first seen in, so the same sentence returning after the dedupe
   window mints a NEW id - and if "dismissed" were checked by id alone, every
   dismissed idea would come back sixty-one sessions later. It is checked on the
   normalised TEXT of the dismissed ids' own rows.

**NO MODEL IS EVER CALLED HERE.**

VERIFIED ON THIS BRANCH (1b9d77e0): `ai_jobs.improvement_ideas` does not exist,
so every test below fails on the import.
"""

from __future__ import annotations

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
#: The same thought, typed by a model on another night: different case, an extra
#: space, and no full stop.
SAME_THOUGHT = "wait for the  SECOND test before sizing up"
#: A different thought that shares most of its words.
OTHER_THOUGHT = "Wait for the second test before cutting."


@pytest.fixture
def night(tmp_path, monkeypatch):
    return fx.install_stores(monkeypatch, tmp_path)


def _lines(path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _run(night, text):
    from ai_jobs import improvement_ideas

    inputs = improvement_ideas.build_ideas_inputs(fx.SESSION, root=night["root"])
    allowed = list(inputs["allowed_source_ids"])
    measurable = str(improvement_ideas.MEASURABLES[0].name)
    return improvement_ideas.run_improvement_ideas(
        session_date=fx.SESSION,
        now=fx.WEEKNIGHT,
        root=night["root"],
        request=fx.fake_request(
            fx.reply([fx.idea_payload(text, measurable=measurable, evidence=allowed[:1])])
        ),
    )


def test_the_normal_form_folds_case_spacing_and_punctuation_and_nothing_else():
    """Two spellings of one thought are one idea; two thoughts are two."""
    from ai_jobs import improvement_ideas

    normalise = improvement_ideas.normalise_text
    assert normalise(TEXT) == normalise(SAME_THOUGHT)
    assert normalise(TEXT) != normalise(OTHER_THOUGHT)
    assert normalise("  ") == ""


def test_a_repeat_inside_the_window_appends_a_seen_count_row_and_keeps_one_idea(night):
    """Hand-counted: 1 row on disk before, 2 after, 1 folded idea, seen_count 2.

    The earlier line is byte-identical afterwards - this store is append-only,
    and a store that rewrote a row to bump a counter would lose the first
    sighting's date and model.
    """
    from ai_jobs import improvement_ideas

    yesterday = fx.sessions_back(fx.SESSION, 1)
    fx.write_ideas(night["ideas"], [fx.stored_idea_row(TEXT, session=yesterday)])
    first_line = night["ideas"].read_text(encoding="utf-8").splitlines()[0]

    _run(night, SAME_THOUGHT)

    lines = night["ideas"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, lines
    assert lines[0] == first_line, "the earlier sighting was rewritten"

    folded = list(improvement_ideas.read_ideas())
    assert len(folded) == 1
    assert folded[0]["idea_id"] == fx.idea_id_for(yesterday, TEXT)
    assert folded[0]["seen_count"] == 2
    assert folded[0]["first_seen"] == yesterday


def test_the_dedupe_window_is_counted_in_exchange_sessions(night):
    """`DEDUPE_SESSIONS` sessions back, not days.

    Hand-built: one stored idea stamped ``DEDUPE_SESSIONS - 1`` SESSIONS before
    2026-09-18, walked with `market_calendar.previous_session`. That date is
    about eighty calendar days old, so a window written as
    ``timedelta(days=DEDUPE_SESSIONS)`` calls it forgotten and mints a second
    idea - which is this test failing.
    """
    from ai_jobs import improvement_ideas

    inside = fx.sessions_back(fx.SESSION, improvement_ideas.DEDUPE_SESSIONS - 1)
    fx.write_ideas(night["ideas"], [fx.stored_idea_row(TEXT, session=inside)])

    _run(night, TEXT)

    folded = list(improvement_ideas.read_ideas())
    assert len(folded) == 1, [row["idea_id"] for row in folded]
    assert folded[0]["idea_id"] == fx.idea_id_for(inside, TEXT)
    assert folded[0]["seen_count"] == 2


def test_the_same_thought_past_the_window_is_a_new_idea(night):
    """Hand-built: one stored idea ``DEDUPE_SESSIONS + 1`` sessions back.

    Two ideas afterwards, each with its own first-seen session: the desk is
    allowed to have the thought again a quarter later, and the count must not
    pretend it never went away.
    """
    from ai_jobs import improvement_ideas

    old = fx.sessions_back(fx.SESSION, improvement_ideas.DEDUPE_SESSIONS + 1)
    fx.write_ideas(night["ideas"], [fx.stored_idea_row(TEXT, session=old)])

    _run(night, TEXT)

    folded = {row["idea_id"]: row for row in improvement_ideas.read_ideas()}
    assert len(folded) == 2, sorted(folded)
    assert folded[fx.idea_id_for(old, TEXT)]["seen_count"] == 1
    assert folded[fx.idea_id_for(fx.SESSION, TEXT)]["seen_count"] == 1


def test_a_dismissed_idea_is_never_stored_again(night):
    """*"a dismissed idea never returns"* - across a restart, which is what a
    file-backed state means.

    Hand-counted: 1 row on disk before, 1 after; the night stored nothing.
    """
    from ai_jobs import improvement_ideas

    yesterday = fx.sessions_back(fx.SESSION, 1)
    idea_id = fx.idea_id_for(yesterday, TEXT)
    fx.write_ideas(night["ideas"], [fx.stored_idea_row(TEXT, session=yesterday)])
    fx.write_state(night["state"], {idea_id: fx.dismissed_record(idea_id)})
    before = night["ideas"].read_bytes()

    _run(night, SAME_THOUGHT)

    assert night["ideas"].read_bytes() == before
    folded = list(improvement_ideas.read_ideas())
    assert len(folded) == 1
    assert folded[0]["seen_count"] == 1, "a dismissed idea was counted again"


def test_a_dismissed_idea_does_not_come_back_after_the_dedupe_window(night):
    """The trap the id itself creates.

    The dismissed row is ``DEDUPE_SESSIONS + 1`` sessions old, so tonight's
    sentence would mint a DIFFERENT id. A dismissal checked by id alone lets the
    idea back in; checked on the normalised text it never returns.

    Hand-counted: 1 row on disk before, 1 after.
    """
    from ai_jobs import improvement_ideas

    old = fx.sessions_back(fx.SESSION, improvement_ideas.DEDUPE_SESSIONS + 1)
    idea_id = fx.idea_id_for(old, TEXT)
    fx.write_ideas(night["ideas"], [fx.stored_idea_row(TEXT, session=old)])
    fx.write_state(night["state"], {idea_id: fx.dismissed_record(idea_id)})

    _run(night, TEXT)

    assert len(_lines(night["ideas"])) == 1
    assert len(list(improvement_ideas.read_ideas())) == 1


def test_a_kept_idea_is_not_a_dismissed_one(night):
    """A Keep must not silence the idea it approved of.

    Hand-counted: 1 row before, 2 after (the repeat), 1 folded idea with
    seen_count 2.
    """
    from ai_jobs import improvement_ideas

    yesterday = fx.sessions_back(fx.SESSION, 1)
    idea_id = fx.idea_id_for(yesterday, TEXT)
    fx.write_ideas(night["ideas"], [fx.stored_idea_row(TEXT, session=yesterday)])
    fx.write_state(night["state"], {idea_id: fx.kept_record(idea_id)})

    _run(night, TEXT)

    assert len(_lines(night["ideas"])) == 2
    folded = list(improvement_ideas.read_ideas())
    assert len(folded) == 1
    assert folded[0]["seen_count"] == 2
