r"""TJ-7 into TJ-16 - a mood is point-in-time, and the tagger never hears it. RED.

`plan.md` §12.4 TJ-16 item 1: the graded prediction's context snapshot holds
*"... confidence, direction, and, **when TJ-7 exists, mood**. A field the desk
cannot measure is `unmeasured`, never guessed."* The packet's own line: *"Mood
is a context field in TJ-16's ledger from the day it exists."*

THE RULE THAT MAKES THIS HARD, AND WHY IT IS PINNED AS A NUMBER
---------------------------------------------------------------
The mood the trader clicks is on the session's LAST card - after the trade, and
often after the close. The 07:02 read was made by someone who had not felt it
yet. So `context_for` may only ever carry the mood that was already recorded AT
THE STAMP; a later one is `UNMEASURED`, exactly as a bar that has not finished
is not a bar. A context field filled from a row written afterwards would be
hindsight dressed as a measurement, and `prediction_contrast` would then rank
the trader's own outcome knowledge as a feature of their skill
(`ai_jobs/prediction_contrast.py:105-118` walks every SCALAR in `context`).

The same rule from the other side: a mood written after the session is LABELLED
and never dropped (`observation_tags.notes_for`'s precedent - the label rides on
the note and stays OUT of the payload), and the TAGGER is never told about a
mood at all. `build_evidence` "knows about two strings and a picklist and has no
way to reach anything else"; TJ-7 does not widen that.

RED FOR: `context_for` takes no `mood_entries` and emits no `mood` key (TypeError
then a real assertion), and `market_journal.mood_at` does not exist.
The tagger tests and the floor test are STATED GREEN GUARDS - they pass today
and exist so the builder cannot make them stop passing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj7_support as fx  # noqa: E402
import tj10_support as fx10  # noqa: E402


def _read_row():
    """One graded read of the 07:02 stamp, built by TJ-10's own reader."""
    import market_read_grades as grades

    entry = fx10.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    rows = grades.read_rows([entry], session=fx.SESSION)
    assert rows, "TJ-10 read no row from a clicked entry"
    return entry, rows[0]


def _context(entry, row, moods):
    import market_read_grades as grades

    return grades.context_for(entry, row=row, mood_entries=moods)


# ---------------------------------------------------------------------------
# point in time
# ---------------------------------------------------------------------------
def test_a_mood_clicked_at_the_close_is_not_context_for_an_eight_oclock_read():
    """12:55 is after 07:02. The read was made without it."""
    import market_read_grades as grades

    entry, row = _read_row()
    at_the_close = fx.row_with_a_mood(
        entry_id="mj-close", stamp=fx.pacific(12, 55), score=5
    )

    context = _context(entry, row, [at_the_close])

    assert context["mood"] == grades.UNMEASURED
    assert context["mood"] != 0, "an unmeasured field is never a zero"


def test_a_mood_already_recorded_is_the_context_the_read_was_made_in():
    entry, row = _read_row()
    before_the_open = fx.row_with_a_mood(
        entry_id="mj-0655", stamp=fx.pacific(6, 55), score=3
    )

    assert _context(entry, row, [before_the_open])["mood"] == 3


def test_the_context_mood_is_the_latest_one_at_or_before_the_stamp():
    """Two moods, one read: the one that was true THEN, by `astimezone`, never
    by stripping a zone."""
    import market_journal

    entry, row = _read_row()
    early = fx.row_with_a_mood(entry_id="mj-a", stamp=fx.pacific(6, 55), score=2)
    later = fx.row_with_a_mood(entry_id="mj-b", stamp=fx.pacific(8, 55), score=5)

    assert _context(entry, row, [early, later])["mood"] == 2, "07:02 knew only the 06:55"
    assert market_journal.mood_at([early, later], fx.pacific(9, 5)).score == 5
    assert market_journal.mood_at([early, later], fx.pacific(6, 0)) is None


def test_no_mood_at_all_reads_unmeasured_and_the_rest_of_the_snapshot_is_unmoved():
    """The field joins the block; nothing else in it changes."""
    import market_read_grades as grades

    entry, row = _read_row()
    with_none = _context(entry, row, [])
    old_keys = {
        "internals", "hour", "spy_vs_session_vwap", "spy_vs_prior_range", "gap_pct",
        "d1_environment", "last_hour_spy", "agrees_with_own_d1",
        "previous_call_verdict", "confidence", "direction",
    }

    assert with_none["mood"] == grades.UNMEASURED
    assert old_keys <= set(with_none)
    assert set(with_none) - old_keys == {"mood"}, "ONE new context field"


def test_a_mood_recorded_after_the_session_is_labelled_in_the_pack_never_dropped():
    """A mood typed in the evening is still the trader's mood. It is kept, it
    is counted, and it says when it was written - the partition a later reader
    needs is a LABEL, never a deletion."""
    import tj4_support as fx4

    inputs = fx4.pack_inputs()
    evening = fx.row_with_a_mood(
        entry_id="mj-evening",
        stamp=fx.pacific(20, 30),
        score=1,
        state_tags=("tilted",),
        after_the_session=True,
    )
    inputs["entries"] = list(inputs["entries"]) + [evening]

    import day_review_pack

    pack = day_review_pack.build_pack(fx.SESSION, now=fx4.AFTER_THE_CLOSE, **inputs)
    items = list(pack["mood"]["recorded"])

    assert len(items) == 1 and pack["mood"]["n"] == 1
    assert items[0]["written_after_the_session"] is True
    assert items[0]["score"] == 1


# ---------------------------------------------------------------------------
# the tagger never hears about it - STATED GREEN GUARDS
# ---------------------------------------------------------------------------
def test_the_tagger_is_never_handed_a_mood():
    """GREEN TODAY, and it must stay green. `notes_for` reads TWO fields, and
    `build_evidence` may not learn a third."""
    import ai_jobs.observation_tags as tagger

    moody = fx.row_with_a_mood(
        entry_id="mj-a",
        score=1,
        state_tags=("tilted",),
        note="I was furious about the gap",
        text="I am furious about this gap",
    )
    moody["mentor"] = {
        "observation": "SPY gapped over the range and held.",
        "prediction": {"direction": "up", "horizon": "rest_of_day",
                       "confidence": "high", "because": "breadth",
                       "schema": "mentor_prediction_v1"},
    }

    notes = tagger.notes_for([moody])
    assert {note["field"] for note in notes} == set(tagger.FIELDS)
    assert all("mood" not in note for note in notes)

    payload = json.dumps(tagger.build_evidence(notes), default=str)
    assert "tilted" not in payload
    assert "I was furious about the gap" not in payload, "the process note is not a note"
    assert '"mood"' not in payload
    assert "state_tags" not in payload


def test_an_unmeasured_mood_contributes_nothing_to_a_contrast():
    """GREEN TODAY (`prediction_contrast._is_unmeasured`), pinned because a
    mood is the first context field most rows will not have. `unmeasured` is
    NOT a category and NOT a zero: the row simply leaves that feature's count."""
    import ai_jobs.prediction_contrast as contrast
    import market_read_grades as grades

    rows = [
        {"verdict": "right", "context": {"mood": 4, "hour": 7}},
        {"verdict": "wrong", "context": {"mood": grades.UNMEASURED, "hour": 7}},
    ]
    encoded = contrast.encode_rows(rows)

    assert "mood" in encoded[0] and encoded[0]["mood"] == 4.0
    assert "mood" not in encoded[1], "an unmeasured mood is not a zero"


def test_a_thin_mood_feature_is_never_ranked_or_printed_as_a_rate():
    """GREEN TODAY through TJ-15's floors, pinned for the first field TJ-7 adds.
    Three rows is under `MIN_CONTRAST_SIDE_N`, so `mood` shows in
    `thin_features` with both counts and NO auc - never a ranked tendency."""
    import evidence_contrast

    assert evidence_contrast.MIN_CONTRAST_SIDE_N > 3, (
        "this fixture's three rows must be UNDER the feature floor"
    )
    groups = {
        "right": [{"mood": 4.0}, {"mood": 5.0}],
        "wrong": [{"mood": 2.0}],
    }
    report = evidence_contrast.contrast(groups["right"], groups["wrong"])
    thin = {row["feature"] for row in report.get("thin_features") or ()}
    ranked = {row["feature"] for row in report.get("features") or ()}

    assert "mood" in thin
    assert "mood" not in ranked
