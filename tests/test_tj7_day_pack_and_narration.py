r"""TJ-7 change 3 - the day pack carries the mood, honestly. RED.

`plan.md` §12.4 "TJ-7" change 3: *"The day pack carries them (`mood` section);
the day and week narrations may cite them ('you said rushed at 07:30 and passed
on three names by 08:00')."*

The hook is already there and deliberately empty: `day_review_pack.SECTIONS`
holds `"mood"` and `build_pack` writes `{}` into it
(`scripts/day_review_pack.py:70,497`, "TJ-7's hook"). TJ-7 FILLS it, and what
it fills has to obey the three rules the pack was built around.

1. **`inputs_hash` is over the SECTIONS and never over the clock** (the pack's
   own docstring): the post-close tick and the nightly slot build the same pack
   hours apart, and change 2's skip depends on that. So the hash must not move
   when the clock does - and it MUST move when a mood moves, or the night would
   skip the story for a session that had changed.
2. **A citation is only legal if the id is in `allowed_source_ids`.** A story
   that says "you said rushed at 07:30" has to be able to name the row it read.
3. **Nothing is invented when there is nothing.** The live desk holds ZERO
   moods today (`tj7_support`'s read-only count), so the first thing the trader
   sees is "no mood recorded yet", `n 0`, and no percentage.

The week's evidence list stays CLOSED: `week_review_narration.EVIDENCE_KEYS` is
fenced here to the fifteen names it holds at `e00520a3` plus, at most, ONE new
name - `mood`. The DAY narration needs no new key at all: `_day_evidence` sends
`{name: pack.get(name) for name in day_review_pack.SECTIONS}`
(`ai_jobs/day_review_narration.py:495`), and `mood` is already a section.

RED FOR: `build_pack` writes `{}` into `mood` whatever the entries say (a real
assertion on `pack["mood"]`), and `day_review_pack` has no `mood_statement` /
`MOOD_EMPTY_STATEMENT` (AttributeError).
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj4_support as fx4  # noqa: E402
import tj7_support as fx  # noqa: E402

#: The fifteen evidence keys `week_review_narration.EVIDENCE_KEYS` held when
#: TJ-7 branched (`scripts/ai_jobs/week_review_narration.py:75-90`).
WEEK_EVIDENCE_AT_E00520A3 = (
    "package_id",
    "evidence_hash",
    "instructions",
    "allowed_source_ids",
    "week_id",
    "sessions",
    "sessions_with_facts",
    "sessions_missing",
    "narrated_sessions",
    "days",
    "were_you_right",
    "tendencies",
    "misses",
    "rollup",
    "walkaway_totals",
)


def _pack_with_moods(*rows, **overrides):
    """The TJ-4 pack, built over the SAME entries plus the mood rows."""
    import day_review_pack

    now = overrides.pop("now", fx4.AFTER_THE_CLOSE)
    inputs = fx4.pack_inputs()
    inputs["entries"] = list(inputs["entries"]) + list(rows)
    inputs.update(overrides)

    return day_review_pack.build_pack(fx.SESSION, now=now, **inputs)


# ---------------------------------------------------------------------------
# the section
# ---------------------------------------------------------------------------
def test_the_pack_carries_the_sessions_moods_with_their_own_source_ids():
    """Two mood rows in, two items out, each naming the row it came from."""
    morning = fx.row_with_a_mood(
        entry_id="mj-0700", stamp=fx.pacific(7, 30), score=2, state_tags=("rushed",),
        followed_plan="", text="Rushed.",
    )
    close = fx.row_with_a_mood(
        entry_id="mj-1255", stamp=fx.pacific(12, 55), score=4, state_tags=("calm", "tired"),
        followed_plan="partly", note="cut the third one early", text="Followed the plan? partly",
    )

    section = _pack_with_moods(morning, close)["mood"]

    assert section["n"] == 2, section
    items = list(section["recorded"])
    assert [item["entry_id"] for item in items] == ["mj-0700", "mj-1255"], "in time order"
    assert [item["score"] for item in items] == [2, 4]
    assert list(items[1]["state_tags"]) == ["calm", "tired"]
    assert items[1]["followed_plan"] == "partly"
    assert items[1]["note"] == "cut the third one early"
    assert all(item["source_id"] for item in items)
    assert len({item["source_id"] for item in items}) == 2, "one minter, no collisions"


def test_a_row_whose_mood_key_is_empty_is_not_a_mood():
    """Present and empty is the same as absent, and neither is a `3`."""
    section = _pack_with_moods(
        fx.row_with_the_key_present_and_empty(),
        fx.old_row_without_the_key(),
    )["mood"]

    assert section == {}, section


def test_a_session_with_no_mood_keeps_the_empty_hook_and_says_so():
    """The desk's honest first state: zero moods, no invented percentage."""
    import day_review_pack

    pack = fx4.build()
    assert pack["mood"] == {}, "TJ-4's hook shape is kept for a session with none"

    said = day_review_pack.mood_statement(pack)
    assert said == day_review_pack.MOOD_EMPTY_STATEMENT
    assert "no mood recorded yet" in said.lower()
    assert "0" in said, f"the count is stated: {said!r}"
    assert "%" not in said, "nothing is a rate of zero"
    assert fx.NO_MOOD_YET_N == 0


def test_the_statement_counts_what_is_there_and_still_prints_no_rate():
    import day_review_pack

    pack = _pack_with_moods(
        fx.row_with_a_mood(entry_id="mj-a", stamp=fx.pacific(7, 30)),
        fx.row_with_a_mood(entry_id="mj-b", stamp=fx.pacific(12, 55)),
    )
    said = day_review_pack.mood_statement(pack)

    assert "2" in said
    assert "%" not in said


# ---------------------------------------------------------------------------
# the hash, and what the night may cite
# ---------------------------------------------------------------------------
def test_the_inputs_hash_ignores_the_clock():
    """The post-close tick and the nightly slot build the same pack."""
    row = fx.row_with_a_mood(entry_id="mj-a")
    first = _pack_with_moods(row, now=fx4.AFTER_THE_CLOSE)
    second = _pack_with_moods(row, now=fx4.AFTER_THE_CLOSE + timedelta(hours=9))

    assert first["built_at"] != second["built_at"]
    assert first["inputs_hash"] == second["inputs_hash"]


def test_the_inputs_hash_moves_when_a_mood_moves():
    """Otherwise the night would skip a session that really had changed."""
    two = _pack_with_moods(fx.row_with_a_mood(entry_id="mj-a", score=2))
    four = _pack_with_moods(fx.row_with_a_mood(entry_id="mj-a", score=4))
    tagged = _pack_with_moods(
        fx.row_with_a_mood(entry_id="mj-a", score=2, state_tags=("tilted",))
    )
    none_at_all = fx4.build()

    assert two["inputs_hash"] != four["inputs_hash"]
    assert two["inputs_hash"] != tagged["inputs_hash"]
    assert two["inputs_hash"] != none_at_all["inputs_hash"]


def test_a_mood_may_be_cited_by_the_night():
    """Every id the story is allowed to name, once each, in pack order."""
    import day_review_pack

    pack = _pack_with_moods(fx.row_with_a_mood(entry_id="mj-a", stamp=fx.pacific(7, 30)))
    allowed = day_review_pack.allowed_source_ids(pack)
    mood_ids = [item["source_id"] for item in pack["mood"]["recorded"]]

    assert mood_ids
    for source_id in mood_ids:
        assert source_id in allowed, f"{source_id} could never be cited"
    assert len(allowed) == len(set(allowed))


def test_the_day_narration_sees_the_mood_through_the_sections_it_already_has():
    """No new key on the day side: `mood` is already in `SECTIONS`."""
    import ai_jobs.day_review_narration as narration
    import day_review_pack

    pack = _pack_with_moods(fx.row_with_a_mood(entry_id="mj-a", score=5))
    evidence = narration._day_evidence(pack, Path("."))

    assert "mood" in day_review_pack.SECTIONS
    assert evidence["pack"]["mood"] == pack["mood"]
    assert evidence["pack"]["mood"]["recorded"][0]["score"] == 5


def test_the_weeks_evidence_list_gains_at_most_one_named_mood_key():
    """A CLOSED list stays closed: if the week story ever carries a mood it
    does so under ONE name, never smuggled into `days` or `rollup`."""
    import ai_jobs.week_review_narration as week

    now = tuple(week.EVIDENCE_KEYS)
    assert set(WEEK_EVIDENCE_AT_E00520A3) <= set(now), "no key was removed"
    added = set(now) - set(WEEK_EVIDENCE_AT_E00520A3)
    assert added <= {"mood"}, f"TJ-7 may add only a `mood` key: {sorted(added)}"
    forbidden = {"bars", "lake", "research_lake", "prices"}
    assert not set(now) & forbidden
