r"""TJ-6 - an idea is grounded, bounded and checked before it is stored. RED.

`plan.md` §12.4 TJ-6 change 1 with its **AMENDED 2026-09-19** block:

* up to THREE ideas a night, each ``{idea_id, kind, text <= 280, evidence:
  [source_id], first_seen, seen_count}``;
* *"an idea without evidence is dropped"*;
* *"a `process` idea must name ONE measurable the desk already computes ... or
  it is dropped like an idea without evidence"*;
* the inputs are the last five packs and narrations, the week's walk-away
  totals, the mood/process fields when present (TJ-7) and the fixed, versioned
  30-line ``IDEAS_PROGRAM_CARD``.

The rules every grounded slot in this program learned the hard way (packet
TJ-5 "CORRECTED" item 6, each one a NO-GO on another packet the same week) bind
here too, and this file is where they are pinned:

* a JSON schema's ``maxItems`` / ``additionalProperties`` is a GRAMMAR HINT; the
  slot's own verifier re-checks every bound after the answer comes back;
* bounds come from the INPUT (the measurables the desk really has, the ids this
  night really carries), never from a fixed guess;
* a status vocabulary is IMPORTED from `ai_jobs.ledger`;
* a citation the input does not hold is a REJECTION of the whole answer, with
  the prior store byte-identical - the same rule `week_review_narration` and
  `day_review_narration` follow. A well-formed idea that simply carries no
  evidence, or names no measurable, is DROPPED on its own: that is the packet's
  own word for those two cases, and it is a bounded, per-idea judgement rather
  than a claim about a row nobody has.

**NO MODEL IS EVER CALLED HERE**: every run is handed a `request=` of this
module's own, from `tj6_support.fake_request`.

VERIFIED ON THIS BRANCH (1b9d77e0): `scripts/ai_jobs/improvement_ideas.py` does
not exist, so every test below fails on the import.
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


@pytest.fixture
def night(tmp_path, monkeypatch):
    """A scratch Day Review root with TJ-5's three packed sessions, and both
    ideas stores pointed at `tmp_path`."""
    return fx.install_stores(monkeypatch, tmp_path)


def _run(night, answer, **kwargs):
    from ai_jobs import improvement_ideas

    calls: list[dict] = []
    outcome = improvement_ideas.run_improvement_ideas(
        session_date=fx.SESSION,
        now=fx.WEEKNIGHT,
        root=night["root"],
        request=fx.fake_request(answer, calls=calls),
        **kwargs,
    )
    return outcome, calls


def _stored(night) -> list[dict]:
    text = night["ideas"].read_text(encoding="utf-8") if night["ideas"].exists() else ""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _first_measurable() -> str:
    from ai_jobs import improvement_ideas

    return str(improvement_ideas.MEASURABLES[0].name)


def _allowed(night) -> list[str]:
    from ai_jobs import improvement_ideas

    inputs = improvement_ideas.build_ideas_inputs(fx.SESSION, root=night["root"])
    return list(inputs["allowed_source_ids"])


# ---------------------------------------------------------------------------
# what the model may see
# ---------------------------------------------------------------------------
def test_the_evidence_is_a_closed_set_with_no_bars_and_no_lake_in_it(night):
    """The ideas slot reads packs, narrations, totals and the program card.

    A bar section here would put a session's tape into a prompt; a lake section
    would put the research warehouse in one. Both are forbidden by `plan.md`
    TJ-6 change 1's input list, and neither can appear if `EVIDENCE_KEYS` is
    closed.
    """
    from ai_jobs import improvement_ideas

    keys = tuple(improvement_ideas.EVIDENCE_KEYS)
    assert keys, "the evidence set is empty"
    for name in keys:
        lowered = str(name).lower()
        for forbidden in ("bar", "lake", "tape", "tick", "warehouse"):
            assert forbidden not in lowered, f"{name} is not allowed in the evidence"

    inputs = improvement_ideas.build_ideas_inputs(fx.SESSION, root=night["root"])
    package = improvement_ideas.build_evidence(inputs)
    assert set(package) <= set(keys), sorted(set(package) - set(keys))


def test_the_program_card_is_a_checked_in_versioned_thirty_line_description():
    """*"a fixed, versioned 30-line description of the program's pages and
    fields (`IDEAS_PROGRAM_CARD`, checked in) so a `program` idea is about THIS
    program"*. Hand-counted: exactly 30 lines, none of them blank."""
    from ai_jobs import improvement_ideas

    card = tuple(improvement_ideas.IDEAS_PROGRAM_CARD)
    assert len(card) == 30, f"the program card has {len(card)} lines"
    assert all(str(line).strip() for line in card), "a blank line is not a description"
    assert str(improvement_ideas.PROGRAM_CARD_VERSION).strip()


def test_the_night_tells_the_model_which_measurables_exist(night):
    """The bound comes from the INPUT: a `process` idea may only name a
    measurable the desk really computes, so the registry's names travel WITH the
    evidence and the schema's enum is built from them."""
    from ai_jobs import improvement_ideas

    inputs = improvement_ideas.build_ideas_inputs(fx.SESSION, root=night["root"])
    offered = [str(name) for name in inputs["measurables"]]
    assert offered == [str(item.name) for item in improvement_ideas.MEASURABLES]

    schema = improvement_ideas.schema_for(inputs)
    enum = schema["properties"]["ideas"]["items"]["properties"]["measurable"]["enum"]
    # "" is the `program` idea's answer: PRESENT and EMPTY, never absent.
    assert set(enum) == set(offered) | {""}
    assert schema["properties"]["ideas"]["maxItems"] == improvement_ideas.MAX_IDEAS_PER_NIGHT
    # The decoder may only write ids tonight carries: gemma shortened
    # "ideas_program_card_v1" to "program_card_v1" on 2026-09-24 and lost the night.
    cited = schema["properties"]["ideas"]["items"]["properties"]["evidence"]["items"]
    assert cited["enum"] == list(inputs["allowed_source_ids"])


# ---------------------------------------------------------------------------
# the verifier re-checks every bound
# ---------------------------------------------------------------------------
def test_a_fourth_idea_is_rejected_by_the_verifier_not_trimmed_by_the_schema(night):
    """Hand-counted: 4 ideas offered, cap 3 -> the answer is rejected WHOLE and
    the store keeps the two rows it already held, byte-identical.

    `maxItems` told the decoder what to write; it did not stop this reply
    arriving. The check that matters runs after the answer comes back.
    """
    from ai_jobs import improvement_ideas, ledger

    before_rows = [
        fx.stored_idea_row("Stop chasing the open.", session="2026-09-16"),
        fx.stored_idea_row("Size the second entry smaller.", session="2026-09-17"),
    ]
    fx.write_ideas(night["ideas"], before_rows)
    before = night["ideas"].read_bytes()

    allowed = _allowed(night)
    measurable = _first_measurable()
    offered = [
        fx.idea_payload(f"Idea number {index}.", measurable=measurable, evidence=allowed[:1])
        for index in range(4)
    ]
    outcome, _calls = _run(night, fx.reply(offered))

    assert outcome["status"] in ledger.RECOGNISED_JOB_STATUSES
    assert outcome["status"] != ledger.STATUS_OK
    assert night["ideas"].read_bytes() == before
    assert len(improvement_ideas.read_ideas()) == 2


def test_an_idea_citing_an_id_this_night_does_not_carry_is_rejected_whole(night):
    """A fabricated citation is the defect TJ-4 and TJ-5 both reject WHOLE.

    Hand-counted: 2 ideas, one of them citing `no-such-session/said:made-up`,
    which is in no pack -> nothing is written at all.
    """
    from ai_jobs import improvement_ideas, ledger

    allowed = _allowed(night)
    measurable = _first_measurable()
    outcome, _calls = _run(
        night,
        fx.reply(
            [
                fx.idea_payload("A real one.", measurable=measurable, evidence=allowed[:1]),
                fx.idea_payload(
                    "An invented one.",
                    measurable=measurable,
                    evidence=["no-such-session/said:made-up"],
                ),
            ]
        ),
    )

    assert outcome["status"] != ledger.STATUS_OK
    assert _stored(night) == []
    assert improvement_ideas.read_ideas() == () or list(improvement_ideas.read_ideas()) == []


def test_check_ideas_is_a_function_the_slot_calls_and_a_test_can_call(night):
    """The verifier is separable, and it RAISES rather than returning a flag.

    A bound checked only inside the run is a bound nobody can test at its edge.
    """
    from ai_jobs import improvement_ideas

    inputs = improvement_ideas.build_ideas_inputs(fx.SESSION, root=night["root"])
    allowed = list(inputs["allowed_source_ids"])
    measurable = _first_measurable()
    good = {
        "ideas": [
            fx.idea_payload("A real one.", measurable=measurable, evidence=allowed[:1])
        ]
    }
    improvement_ideas.check_ideas(good, inputs)  # does not raise

    with pytest.raises(improvement_ideas.IdeasRejected):
        improvement_ideas.check_ideas(
            {
                "ideas": [
                    fx.idea_payload(
                        "An invented one.", measurable=measurable, evidence=["nope"]
                    )
                ]
            },
            inputs,
        )


# ---------------------------------------------------------------------------
# what is dropped, one idea at a time
# ---------------------------------------------------------------------------
def test_an_idea_with_no_evidence_and_one_with_no_measurable_are_dropped(night):
    """The table in `tj6_support`'s docstring: 4 offered, 2 usable.

    Dropped, not rejected: the other two ideas are perfectly good and the night
    is not thrown away because the model added a fourth thought it could not
    support.
    """
    from ai_jobs import improvement_ideas, ledger

    allowed = _allowed(night)
    measurable = _first_measurable()
    outcome, _calls = _run(
        night,
        fx.reply(
            [
                fx.idea_payload(
                    "Wait for the second test before sizing up.",
                    measurable=measurable,
                    evidence=allowed[:1],
                ),
                fx.idea_payload(
                    "Put the walk-away table beside the trades on Day Review.",
                    kind="program",
                    measurable="",
                    evidence=allowed[:1],
                ),
                fx.idea_payload(
                    "Trade less on Mondays.",
                    measurable="not_a_measurable_the_desk_has",
                    evidence=allowed[:1],
                ),
                fx.idea_payload(
                    "Something I cannot show you.", measurable=measurable, evidence=[]
                ),
            ]
        ),
    )

    assert outcome["status"] == ledger.STATUS_OK, outcome.get("reason")
    rows = improvement_ideas.read_ideas()
    assert len(rows) == 2, [row["text"] for row in rows]
    kinds = sorted(row["kind"] for row in rows)
    assert kinds == ["process", "program"]
    # What was dropped is COUNTED and reaches the ledger row, never silently
    # swallowed (TJ-13A's bounded-package rule: what did not fit is counted and
    # said). Two of the four offered ideas were dropped.
    assert int((outcome.get("extra") or {}).get("dropped")) == 2, outcome.get("extra")


def test_a_process_idea_that_names_no_measurable_at_all_is_dropped(night):
    """AMENDED 2026-09-19: advice is CHECKED. A `process` idea with an empty
    measurable cannot be checked, so it is dropped like one with no evidence.

    Hand-counted: 1 offered, 0 usable - and a night with nothing usable is not
    an `ok` night with an empty store.
    """
    from ai_jobs import improvement_ideas, ledger

    allowed = _allowed(night)
    outcome, _calls = _run(
        night,
        fx.reply([fx.idea_payload("Be more patient.", measurable="", evidence=allowed[:1])]),
    )

    assert len(improvement_ideas.read_ideas()) == 0
    assert outcome["status"] in ledger.RECOGNISED_JOB_STATUSES
    assert str(outcome.get("reason") or "").strip()


def test_an_idea_longer_than_the_cap_is_not_stored(night):
    """``text <= 280`` is in the packet, so it is a bound - and a bound is
    re-checked after the answer comes back.

    Hand-counted: one idea of exactly ``MAX_IDEA_CHARS + 1`` characters.
    """
    from ai_jobs import improvement_ideas

    allowed = _allowed(night)
    measurable = _first_measurable()
    too_long = "x" * (improvement_ideas.MAX_IDEA_CHARS + 1)
    _outcome, _calls = _run(
        night,
        fx.reply([fx.idea_payload(too_long, measurable=measurable, evidence=allowed[:1])]),
    )
    assert [row for row in improvement_ideas.read_ideas() if row["text"] == too_long] == []


def test_the_model_never_names_an_idea_and_never_grades_one(night):
    """`idea_id`, `seen_count` and any verdict field are the DESK's.

    A model that picked its own id could resurrect a dismissed idea by naming
    it; one that graded its own advice would be marking its own homework, which
    the amendment forbids in as many words.
    """
    from ai_jobs import improvement_ideas

    inputs = improvement_ideas.build_ideas_inputs(fx.SESSION, root=night["root"])
    item = improvement_ideas.schema_for(inputs)["properties"]["ideas"]["items"]
    assert item["additionalProperties"] is False
    for forbidden in ("idea_id", "seen_count", "first_seen", "verdict", "result", "grade"):
        assert forbidden not in item["properties"], forbidden

    allowed = list(inputs["allowed_source_ids"])
    measurable = str(improvement_ideas.MEASURABLES[0].name)
    payload = fx.idea_payload(
        "Wait for the second test.", measurable=measurable, evidence=allowed[:1]
    )
    payload["idea_id"] = "idea:2026-01-01:deadbeefdead"
    _outcome, _calls = _run(night, fx.reply([payload]))

    stored = improvement_ideas.read_ideas()
    assert "idea:2026-01-01:deadbeefdead" not in [row["idea_id"] for row in stored]


def test_the_id_is_minted_from_the_session_and_the_normalised_text(night):
    """One sentence, one id, computed - so the same thought twice is one idea
    and the same thought a year later is a new one.

    The expected id is computed in `tj6_support.idea_id_for` from the definition
    (``idea:<session>:<sha1 of the normal form, 12 hex>``), never copied out of
    the code under test.
    """
    from ai_jobs import improvement_ideas

    allowed = _allowed(night)
    measurable = _first_measurable()
    text = "Wait for the second test before sizing up."
    _outcome, _calls = _run(
        night, fx.reply([fx.idea_payload(text, measurable=measurable, evidence=allowed[:1])])
    )
    rows = improvement_ideas.read_ideas()
    assert len(rows) == 1
    assert rows[0]["idea_id"] == fx.idea_id_for(fx.SESSION, text)


def test_the_same_night_run_twice_asks_once_and_appends_once(night):
    """The unchanged-hash skip, the way every grounded slot here has one.

    Hand-counted: two runs over the same inputs -> ONE model call and ONE stored
    row. `force=True` is what re-spends it.
    """
    from ai_jobs import improvement_ideas

    allowed = _allowed(night)
    measurable = _first_measurable()
    answer = fx.reply(
        [
            fx.idea_payload(
                "Wait for the second test.", measurable=measurable, evidence=allowed[:1]
            )
        ]
    )
    calls: list[dict] = []
    for _attempt in range(2):
        improvement_ideas.run_improvement_ideas(
            session_date=fx.SESSION,
            now=fx.WEEKNIGHT,
            root=night["root"],
            request=fx.fake_request(answer, calls=calls),
        )
    assert len(calls) == 1, f"{len(calls)} model calls for one unchanged night"
    assert len(_stored(night)) == 1
