r"""TJ-7 - the two closed evidence lists gain ONE named key each, and no more.

Packet `TJ-5-6-7-13B.md`, TJ-7 correction 6 and `plan.md` §12.4 TJ-6 amendment
**(h)**: *"TJ-7's mood/process fields are NOT in `EVIDENCE_KEYS` yet - that set
is closed and tested, so TJ-7 adds one key plus that test's list"*, and §12.5
item 12: TJ-7 *"owes TJ-6 one `EVIDENCE_KEYS` entry"*. `plan.md` TJ-7 change 3:
*"the day and week narrations may cite them"*.

WHY IT IS ONE NAMED KEY AND NOT A FIELD ON `days`
-------------------------------------------------
A closed list is only closed if a new fact has to be ADMITTED by name. Smuggling
a mood inside each day's block would grow the package without moving the list
that is supposed to bound it, and the next reviewer would have no single place
to look. So `mood` travels under its own name in both packages, session-
qualified so a citation names ONE row of ONE day, and the DAY narration needs no
new key at all because `mood` is already a `day_review_pack.SECTION`.

WHAT IS PINNED
--------------
* exactly one name is added to each list, and it is `mood`;
* the rows the package carries came out of a PACK, and their ids are in
  `allowed_source_ids` - a model may not cite what it was not handed;
* a window with no mood is ZERO ROWS, never a zero day and never a rate;
* a mood is never a MEASURABLE: TJ-6's closed registry is unchanged, so no
  `process` idea can be checked against how the trader felt.

RED BEFORE THE FIX (proven by restoring `week_review_narration.py` and
`improvement_ideas.py` from the merge base): neither `EVIDENCE_KEYS` holds
`mood` and neither package carries the section.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj5_support as fx5  # noqa: E402
import tj6_support as fx6  # noqa: E402
import tj7_support as fx  # noqa: E402

#: The fifteen week keys and the fifteen ideas keys as they stood when TJ-7
#: branched. Hand-read off the two modules; TJ-7 may add `mood` and nothing else.
WEEK_KEYS_BEFORE_TJ7 = (
    "package_id", "evidence_hash", "instructions", "allowed_source_ids", "week_id",
    "sessions", "sessions_with_facts", "sessions_missing", "narrated_sessions",
    "days", "were_you_right", "tendencies", "misses", "rollup", "walkaway_totals",
)
IDEAS_KEYS_BEFORE_TJ7 = (
    "package_id", "evidence_hash", "instructions", "allowed_source_ids",
    "session_date", "sessions", "sessions_with_facts", "sessions_missing", "days",
    "misses", "walkaway_totals", "measurables", "measurable_readings",
    "program_card", "program_card_version",
)

#: Which of TJ-5's three packed sessions gets a mood in these fixtures.
MOODY_SESSION = fx5.PACKED_SESSIONS[-1]


def _mood_row(session: str):
    """One hand-built mood row filed against `session`."""
    return fx.row_with_a_mood(
        entry_id=f"{session}-mood",
        stamp=fx.pacific(12, 55, session=session),
        score=2,
        state_tags=("rushed", "tired"),
        followed_plan="partly",
        note="chased the open",
        session=session,
    )


def _write_week(root: Path, *, with_a_mood: bool = True):
    """TJ-5's three packs on disk, one of them carrying a mood."""
    import day_review_pack

    for session in fx5.PACKED_SESSIONS:
        entries = list(fx5.entries_for(session))
        if with_a_mood and session == MOODY_SESSION:
            entries.append(_mood_row(session))
        pack = fx5.pack_for(session, entries=entries)
        day_review_pack.write_pack(pack, root=Path(root))


@pytest.fixture
def night(tmp_path, monkeypatch):
    stores = fx6.install_stores(monkeypatch, tmp_path, write_the_week=False)
    _write_week(stores["root"])
    return stores


# ---------------------------------------------------------------------------
# the week story
# ---------------------------------------------------------------------------
def test_the_week_list_gains_exactly_one_name_and_it_is_mood():
    import ai_jobs.week_review_narration as week

    added = set(week.EVIDENCE_KEYS) - set(WEEK_KEYS_BEFORE_TJ7)
    assert added == {"mood"}, sorted(added)
    assert set(WEEK_KEYS_BEFORE_TJ7) <= set(week.EVIDENCE_KEYS), "no key was removed"


def test_the_weeks_package_carries_the_mood_it_was_handed_with_a_citable_id(night):
    import ai_jobs.week_review_narration as week

    inputs = week.build_week_inputs(fx5.FRIDAY, root=night["root"])
    section = inputs["mood"]

    assert section["n"] == 1, section
    row = section["recorded"][0]
    assert row["session"] == MOODY_SESSION
    assert row["score"] == 2
    assert list(row["state_tags"]) == ["rushed", "tired"]
    assert row["followed_plan"] == "partly"
    assert row["source_id"].startswith(f"{MOODY_SESSION}{week.WEEK_SOURCE_SEPARATOR}")
    assert row["source_id"] in inputs["allowed_source_ids"], "it could never be cited"
    assert section["sessions_with_a_mood"] == [MOODY_SESSION]

    package = week._evidence(inputs)
    assert package["mood"] == section
    assert set(package) <= set(week.EVIDENCE_KEYS)


def test_a_week_with_no_mood_carries_zero_rows_and_never_a_zero_day(tmp_path, monkeypatch):
    import ai_jobs.week_review_narration as week

    stores = fx6.install_stores(monkeypatch, tmp_path, write_the_week=False)
    _write_week(stores["root"], with_a_mood=False)

    inputs = week.build_week_inputs(fx5.FRIDAY, root=stores["root"])

    assert inputs["mood"]["n"] == 0
    assert inputs["mood"]["recorded"] == []
    assert inputs["mood"]["sessions_with_a_mood"] == []
    # The missing days are still NAMED as missing, not as quiet moodless ones.
    assert set(inputs["sessions_missing"]) == set(fx5.sessions_missing())


# ---------------------------------------------------------------------------
# the ideas slot
# ---------------------------------------------------------------------------
def test_the_ideas_list_gains_exactly_one_name_and_it_is_mood():
    from ai_jobs import improvement_ideas

    added = set(improvement_ideas.EVIDENCE_KEYS) - set(IDEAS_KEYS_BEFORE_TJ7)
    assert added == {"mood"}, sorted(added)
    assert set(IDEAS_KEYS_BEFORE_TJ7) <= set(improvement_ideas.EVIDENCE_KEYS)


def test_the_ideas_package_carries_the_mood_and_nothing_else_new(night):
    from ai_jobs import improvement_ideas

    inputs = improvement_ideas.build_ideas_inputs(fx5.FRIDAY, root=night["root"])
    package = improvement_ideas.build_evidence(inputs)

    assert package["mood"]["n"] == 1, package["mood"]
    row = package["mood"]["recorded"][0]
    assert row["session"] == MOODY_SESSION
    assert row["source_id"] in package["allowed_source_ids"]
    assert set(package) - set(IDEAS_KEYS_BEFORE_TJ7) == {"mood"}, sorted(set(package))


def test_a_mood_is_never_a_measurable(night):
    """A `process` idea names a MEASURABLE the desk computes. A mood is not one:
    the registry is closed, and an idea checked against how the trader felt
    would be an outcome standing over a feeling."""
    from ai_jobs import improvement_ideas

    names = tuple(improvement_ideas.measurable_names())
    assert names, "the measurables registry is empty"
    assert not [name for name in names if "mood" in str(name).lower()], names

    inputs = improvement_ideas.build_ideas_inputs(fx5.FRIDAY, root=night["root"])
    assert not [
        row for row in inputs["measurable_readings"] if "mood" in str(row["measurable"]).lower()
    ]


def test_neither_package_carries_a_bar_a_tape_or_a_lake(night):
    """The reason both lists are closed in the first place."""
    from ai_jobs import improvement_ideas, week_review_narration

    forbidden = ("bar", "lake", "tape", "tick", "warehouse", "price")
    for names in (week_review_narration.EVIDENCE_KEYS, improvement_ideas.EVIDENCE_KEYS):
        for name in names:
            assert not [word for word in forbidden if word in str(name).lower()], name
