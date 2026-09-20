"""TJ-14B item 1 - the registry, and the rule that gives it teeth.

RED BEFORE THE FIX. On `claude/tj14b-mentor-questions`'s base (`e8c04f88`)
`scripts/mentor_questions.py` does not exist, so every test here dies on the
import.

THE CONTRACT THESE PIN (plan.md 12.4 TJ-14 item 2; decision 0021 answer 28)
---------------------------------------------------------------------------
*"each kind names the reader that consumes its answer"* - so the registry is
walkable and a kind whose reader does not import, or does not read the key the
answer is stored under, FAILS. Nothing is asked that nothing reads.

The two teeth tests plant a bad kind with `dataclasses.replace` on a REAL
registered one, so no constructor signature is pinned here: a builder may add
fields to `QuestionKind` freely.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))


def _registry():
    import mentor_questions

    return mentor_questions


def test_every_registered_kind_names_a_consumer_that_reads_its_answer():
    """The whole point of the registry: nothing is asked that nothing reads.

    A kind is only allowed on a card when its `consumer` imports AND that
    reader actually touches the key the answer is filed under. A question whose
    answer lands in a store nobody opens is the trader's time spent for nothing,
    which is exactly what the trader's words rule out.
    """
    mentor_questions = _registry()

    report = {row["kind"]: row for row in mentor_questions.consumer_report()}

    assert report, "the registry is empty"
    # AMENDED by the builder under the lead's decision of 2026-09-19 (TJ-14B
    # decision 1): three kinds ship DORMANT because the reader that will consume
    # them is another packet's (`trade_origin` / `open_position_check` -> TJ-12,
    # `grader_gap` -> TJ-10). `pending()` never puts a dormant kind on a live
    # card, so the rule this test pins - nothing ASKED that nothing reads - is
    # unchanged; a dormant kind is excluded here instead of being given a shim
    # reader nobody calls, which is the lie the walk exists to catch.
    broken = {
        kind: row.get("reason") or row
        for kind, row in report.items()
        if not row.get("dormant") and not (row.get("imports") and row.get("reads"))
    }
    assert broken == {}, f"kinds whose consumer cannot use the answer: {broken}"
    dormant = {kind for kind, row in report.items() if row.get("dormant")}
    assert dormant == {
        "trade_origin",
        "open_position_check",
        "grader_gap",
        # Review blocker 2: its named consumer reads `claimed_setup_id` off
        # `trader_annotations.jsonl` rows, and the answer is filed as an
        # `opportunity_events` row - the KEY is read, the STORE is not joined.
        "quick_like_followup",
    }
    for kind in dormant:
        assert report[kind]["dormant_until"], f"{kind} is dormant with no packet named"


def test_the_registry_covers_every_kind_the_packet_names():
    """v1 is a CLOSED list. A kind that quietly disappeared is a question the
    desk stopped asking without anybody deciding to."""
    mentor_questions = _registry()

    names = {str(kind.kind) for kind in mentor_questions.REGISTRY}

    assert {
        "prediction_m5",
        "prediction_d1",
        "trade_label",
        "trade_origin",
        "open_position_check",
        "quick_like_followup",
        "day_close",
        "grader_gap",
        "ai_question",
    } <= names


def test_the_consumer_check_rejects_a_kind_whose_reader_does_not_exist():
    """The teeth. A check that cannot fail is not a check."""
    mentor_questions = _registry()
    real = mentor_questions.kind_named("trade_origin")

    planted = dataclasses.replace(real, consumer="nowhere_at_all.read_it")
    row = mentor_questions.consumer_report([planted])[0]

    assert row["imports"] is False
    assert row["reads"] is False


def test_the_consumer_check_rejects_a_reader_that_never_touches_the_answer_key():
    """`json.dumps` imports and is callable, and it will never read a Mentor
    answer. A registry that only checked the import would pass this."""
    mentor_questions = _registry()
    real = mentor_questions.kind_named("trade_origin")

    planted = dataclasses.replace(
        real, consumer="json.dumps", answer_key="a_key_json_dumps_never_reads"
    )
    row = mentor_questions.consumer_report([planted])[0]

    assert row["imports"] is True
    assert row["reads"] is False


def test_every_budgeted_question_offers_the_four_answer_states_and_stop_asking():
    """*"Every question offers the four answer states plus `Stop asking this`"*.

    The four states are TJ-9's, read from the module that owns them - never
    re-typed here, and never a fifth invented by a caller.
    """
    mentor_questions = _registry()
    import trade_mentor_trade_check as check

    required = set(check.ANSWER_STATES) | {mentor_questions.STOP_ASKING}

    missing = {
        str(kind.kind): sorted(required - set(kind.options))
        for kind in mentor_questions.REGISTRY
        if getattr(kind, "budgeted", True) and required - set(kind.options)
    }

    assert missing == {}


def test_the_two_prediction_rows_are_registered_and_never_budgeted():
    """TJ-14A's forced clicks are DESCRIBED by the registry so the card has one
    description - but they are outside the budget of three, because a card that
    spent its budget on the prediction would ask nothing else all day."""
    mentor_questions = _registry()

    for name in ("prediction_m5", "prediction_d1"):
        kind = mentor_questions.kind_named(name)
        assert kind.budgeted is False, f"{name} must not be budgeted"

    for name in ("trade_origin", "open_position_check", "quick_like_followup"):
        assert mentor_questions.kind_named(name).budgeted is True


def test_the_forced_trade_label_section_is_registered_outside_the_budget():
    """TJ-9 lists EVERY trade of the reviewed session and a same-session fill is
    always asked (plan.md TJ-14 item 3), so the budget covers the other kinds."""
    mentor_questions = _registry()

    assert mentor_questions.kind_named("trade_label").budgeted is False


def test_a_trigger_opens_no_store_of_its_own(monkeypatch):
    """`scripts/mentor_questions.py` is PURE: every lane arrives in `state`.

    A trigger that opened the journal would be a second opinion about it, and a
    read on whatever thread happened to call `pending`.
    """
    mentor_questions = _registry()
    from tj14b_support import SESSION, state

    import journal_store

    monkeypatch.setattr(
        journal_store,
        "JournalStore",
        lambda *a, **k: pytest.fail("a Mentor question trigger opened the journal"),
    )

    for kind in mentor_questions.REGISTRY:
        kind.trigger(state(session=SESSION))
