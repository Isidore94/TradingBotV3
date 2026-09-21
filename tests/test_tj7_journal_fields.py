r"""TJ-7 change 1 - the journal row gains a mood, and nothing else moves. RED.

`plan.md` §12.4 "TJ-7" change 1: *"`market_journal.build_entry` gains optional
`mood` (1-5), `state_tags` (<=2 from `ui/annotations/vocabularies/
state_tags_v1.json` ...), `process` ({`followed_plan`: yes|no|partly|null,
`note` <=200}). Every reader tolerates absence."*

WHY EACH TEST IS THE SHAPE IT IS
--------------------------------
* The field is **ADDITIVE**. An old row has the key ABSENT (43 of the 84 live
  rows); a row written after TJ-7 with nothing clicked has it PRESENT and EMPTY
  (the live journal's own precedent is `"mentor": {}` on 13 rows). Both mean "no
  mood", and `mood_of` answers `None` for both - never a default mood, never a
  neutral 3.
* The writer is **LOUD**. `build_entry` already RAISES on a row whose timeframe
  and prediction horizon disagree (`PredictionTimeframeError`,
  `market_journal.py:307`) because an append-only ledger has no second chance,
  and `is_publishable` asks the same question again at the gate every write
  passes through. A mood of 9, three state tags or a 201-character process note
  is the same class of defect and gets the same treatment.
* The cap of two and the code list belong to the **VOCABULARY**, not to the
  journal: the swap-the-owner test moves `MAX_STATE_TAGS` and watches
  `build_entry` follow it, so a second copy of the rule cannot pass.
* **No vocabulary version is written as a literal anywhere here.** The block's
  `vocab_version` is compared against what the loader reports.

RED FOR: `build_entry` takes no `mood` keyword (TypeError), and
`market_journal` has no `mood_of` / `mood_at` / `MoodFieldError` (AttributeError).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj7_support as fx  # noqa: E402


def _built(**kwargs):
    import market_journal

    return market_journal.build_entry(
        text=kwargs.pop("text", "Followed the plan? partly"),
        session_date=kwargs.pop("session_date", fx.SESSION),
        now=kwargs.pop("now", fx.pacific(12, 55)),
        origin=kwargs.pop("origin", "trade_mentor"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# the shape of a row
# ---------------------------------------------------------------------------
def test_a_note_saved_without_a_mood_carries_the_key_present_and_empty():
    """Exactly one new key, and it is falsy when nothing was clicked."""
    entry = _built()

    assert set(entry) == set(fx.ENTRY_KEYS_BEFORE_TJ7) | {"mood"}, (
        "TJ-7 adds ONE key to the row and removes none"
    )
    assert entry["mood"] == {}, "present and empty, never absent and never a default mood"


def test_a_mood_is_stored_exactly_as_the_trader_clicked_it():
    """Three clicks, one block, and the version comes from the vocabulary."""
    entry = _built(
        mood=4,
        state_tags=("rushed", "tired"),
        process={"followed_plan": "partly", "note": "chased the open"},
    )

    block = entry["mood"]
    assert block["schema"] == fx.MOOD_SCHEMA
    assert block["score"] == 4
    assert list(block["state_tags"]) == ["rushed", "tired"]
    assert block["process"] == {"followed_plan": "partly", "note": "chased the open"}
    assert block["vocab_version"] == fx.current_vocab_version(), (
        "the stamped version is the vocabulary's own, read at write time"
    )


def test_mood_of_is_the_one_reader_and_answers_none_for_three_kinds_of_absence():
    """A key missing, a key empty and a key null all mean the same thing."""
    import market_journal

    for row in fx.three_kinds_of_absence():
        assert market_journal.mood_of(row) is None, row["entry_id"]

    recorded = market_journal.mood_of(fx.row_with_a_mood(score=5, state_tags=("calm",)))
    assert recorded is not None
    assert recorded.score == 5
    assert tuple(recorded.state_tags) == ("calm",)
    assert recorded.followed_plan == "partly"


def test_a_mood_carries_the_row_s_own_after_the_session_label():
    """`written_after_the_session` is COMPUTED on the row and never backdated;
    the mood reader REPORTS it rather than re-deciding it (TJ-16's rule for
    `observation_tags.notes_for`, kept here)."""
    import market_journal

    during = market_journal.mood_of(fx.row_with_a_mood(after_the_session=False))
    after = market_journal.mood_of(
        fx.row_with_a_mood(entry_id="mj-evening", after_the_session=True)
    )

    assert during.recorded_after_the_session is False
    assert after.recorded_after_the_session is True


def test_the_entry_id_of_a_row_is_not_moved_by_a_mood():
    """`entry_id` is the identity every join over this store uses. A mood is a
    field ON the row, never part of what names it."""
    plain = _built()
    moody = _built(mood=2, state_tags=("tilted",))

    assert plain["entry_id"] == moody["entry_id"]


# ---------------------------------------------------------------------------
# the writer refuses rather than stores something nobody clicked
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, 6, -1, 3.5, "high", True])
def test_a_mood_outside_one_to_five_is_refused_at_the_writer(bad):
    import market_journal

    with pytest.raises(market_journal.MoodFieldError):
        _built(mood=bad)


def test_the_same_refusal_is_asked_again_at_the_publish_gate():
    """A row assembled as a dict literal never calls `build_entry` - the rule
    is asked again where every write passes through (`is_publishable`)."""
    import market_journal

    row = fx.row_with_a_mood()
    row["mood"]["score"] = 9

    ok, reason = market_journal.is_publishable(row)
    assert ok is False
    assert "mood" in reason.lower(), reason


def test_a_scale_of_five_is_what_the_writer_accepts():
    import market_journal

    assert tuple(market_journal.MOOD_SCALE) == (1, 2, 3, 4, 5)
    for score in market_journal.MOOD_SCALE:
        assert _built(mood=score)["mood"]["score"] == score


def test_more_than_two_state_tags_is_refused_and_the_cap_belongs_to_the_vocabulary(
    monkeypatch,
):
    """Swap the OWNER and the journal follows it - a second copy of the cap
    would keep refusing and fail this test."""
    import market_journal
    import trader_state_tags

    assert trader_state_tags.MAX_STATE_TAGS == 2
    with pytest.raises(market_journal.MoodFieldError):
        _built(mood=3, state_tags=("rushed", "tired", "fomo"))

    monkeypatch.setattr(trader_state_tags, "MAX_STATE_TAGS", 3)
    entry = _built(mood=3, state_tags=("rushed", "tired", "fomo"))
    assert len(entry["mood"]["state_tags"]) == 3


def test_an_unknown_state_tag_code_is_refused_and_all_eight_are_accepted():
    import market_journal

    with pytest.raises(market_journal.MoodFieldError):
        _built(mood=3, state_tags=("furious",))

    for code in fx.STATE_TAG_CODES:
        assert _built(mood=3, state_tags=(code,))["mood"]["state_tags"] == [code]


def test_followed_plan_takes_the_three_answers_or_nothing():
    """`yes|no|partly|null` - and null is stored as "", never as "no"."""
    import market_journal

    assert tuple(market_journal.FOLLOWED_PLAN_VALUES) == fx.FOLLOWED_PLAN
    for answer in fx.FOLLOWED_PLAN:
        block = _built(process={"followed_plan": answer})["mood"]
        assert block["process"]["followed_plan"] == answer

    unanswered = _built(mood=3, process={"followed_plan": None})["mood"]
    assert unanswered["process"]["followed_plan"] == ""

    with pytest.raises(market_journal.MoodFieldError):
        _built(process={"followed_plan": "maybe"})


def test_a_process_note_longer_than_two_hundred_characters_is_refused():
    """200 is the number `plan.md` TJ-7 change 1 names. 200 stores, 201 raises."""
    import market_journal

    assert market_journal.PROCESS_NOTE_MAX == 200
    at_the_limit = "x" * 200
    assert len(at_the_limit) == 200
    stored = _built(process={"followed_plan": "yes", "note": at_the_limit})
    assert stored["mood"]["process"]["note"] == at_the_limit

    with pytest.raises(market_journal.MoodFieldError):
        _built(process={"followed_plan": "yes", "note": "x" * 201})


def test_a_mood_alone_is_a_complete_answer_and_so_is_a_process_alone():
    """Optional, in both directions: a trader who clicks a face and nothing
    else, and one who answers the plan question and clicks no face."""
    face_only = _built(mood=5, process=None)["mood"]
    assert face_only["score"] == 5
    assert face_only["process"]["followed_plan"] == ""

    plan_only = _built(mood=None, process={"followed_plan": "yes"})["mood"]
    assert plan_only["score"] is None
    assert plan_only["process"]["followed_plan"] == "yes"


# ---------------------------------------------------------------------------
# every existing reader is unmoved
# ---------------------------------------------------------------------------
def test_every_existing_reader_behaves_identically_on_a_row_with_and_without_a_mood():
    """The readers `CLAUDE.md` names for this store, each asked twice.

    A mood is a field the desk REPORTS. No reader of the journal may change its
    answer because one arrived.
    """
    import ai_jobs.observation_tags as tagger
    import market_journal
    import market_read_grades as grades
    import market_story

    plain = _built(text="Breadth is better and SPY is over its VWAP.")
    moody = _built(
        text="Breadth is better and SPY is over its VWAP.",
        mood=2,
        state_tags=("fomo",),
        process={"followed_plan": "no", "note": "took it anyway"},
    )

    assert market_journal.is_machine_entry(plain) == market_journal.is_machine_entry(moody)
    assert market_journal.is_publishable(plain) == market_journal.is_publishable(moody)
    assert market_journal.prediction_of(plain) == market_journal.prediction_of(moody)
    assert market_journal.session_of_entry(plain) == market_journal.session_of_entry(moody)

    stripped = {key: value for key, value in moody.items() if key != "mood"}
    assert stripped == {key: value for key, value in plain.items() if key != "mood"}, (
        "a mood moves no other field on the row"
    )

    assert grades.read_rows([plain], session=fx.SESSION) == grades.read_rows(
        [moody], session=fx.SESSION
    )
    assert tagger.notes_for([plain]) == tagger.notes_for([moody])

    one = market_story.build_daily_story(fx.SESSION, entries=[plain])
    two = market_story.build_daily_story(fx.SESSION, entries=[moody])
    assert one == two


def test_a_second_mood_for_the_session_is_a_new_row_and_the_first_is_untouched():
    """Ground rule 5: the ledger is append-only. A change of heart at 12:55 is
    a NEW row, and the 06:55 row still says what it said."""
    import market_journal

    morning = fx.row_with_a_mood(
        entry_id="mj-morning", stamp=fx.pacific(6, 55), score=2, state_tags=("tired",)
    )
    close = fx.row_with_a_mood(
        entry_id="mj-close", stamp=fx.pacific(12, 55), score=5, state_tags=("calm",)
    )

    assert market_journal.mood_of(morning).score == 2
    assert market_journal.mood_at([morning, close], fx.pacific(13, 30)).score == 5
    assert market_journal.mood_at([morning, close], fx.pacific(7, 30)).score == 2
    assert market_journal.mood_of(morning).score == 2, "the earlier row is unrewritten"
