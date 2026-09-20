"""TJ-14B, BUILDER-ADDED on top of the tester's red suite.

Three things the tester's files could not pin, because they were the lead's
decision of the same night and the card's question area has no red test:

1. **A dormant kind is described and never asked.** Lead decision 1: a question
   is ASKED only when its answer has a reader (decision 0021 answer 28). The
   tester's files lift the dormancy of two kinds so their own arithmetic holds
   (`tests/tj14b_lift_dormancy.py`); NOTHING here lifts it, so these are the
   tests that prove the rule itself.
2. **The quick-like follow-up writes no annotation row at all.** The tester's
   last assertion in `test_a_quick_like_answer_writes_a_link_that_names_the_like`
   reads `annotation_state("T-1")["tag_status"] != "confirmed"`, and
   `JournalStore.annotation_state` answers `confirmed` for a trade with NO
   annotation row by design ("there is nothing provisional about an absence",
   `journal_store.py:2112-2122`) - so that line cannot pass without writing the
   very row it exists to forbid. The invariant it MEANT is pinned here instead:
   the machine writes nothing into `trade_annotations`.
3. **The card's question area.** At most three drawn, `Stop asking this`
   retiring through the service and storing no answer, and an answer filed
   under the key the registry declares.

Nothing here touches a broker, the live stores or the desk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_desk_isolation import fresh_mentor_pull_tally  # noqa: E402,F401
from tj14b_lift_dormancy import LIFTED  # noqa: E402
from tj14b_support import (  # noqa: E402
    SESSION,
    keys_of,
    like_row,
    new_store,
    pacific,
    slot_at,
    state,
    trade_row,
)


@pytest.fixture()
def lifted(monkeypatch):
    """Lift dormancy for ONE test, never for the whole module.

    The dormancy tests below are this file's whole point, so the lifting
    fixture is explicitly requested rather than autouse - the same
    `dataclasses.replace(kind, dormant_until="")` the lead authorised, over the
    same three kinds `tests/tj14b_lift_dormancy.py` lifts.
    """
    import dataclasses

    import mentor_questions

    monkeypatch.setattr(
        mentor_questions,
        "REGISTRY",
        tuple(
            dataclasses.replace(kind, dormant_until="") if kind.kind in LIFTED else kind
            for kind in mentor_questions.REGISTRY
        ),
    )


# ---------------------------------------------------------------------------
# 1. dormancy
# ---------------------------------------------------------------------------


def test_a_dormant_kind_never_reaches_a_live_card_however_loudly_it_triggers():
    """Two unplanned trades and a month-old open position are exactly what
    `trade_origin` and `open_position_check` fire on. Their readers are TJ-12's,
    so the trader is not asked - a click whose answer nothing reads is their
    time spent for nothing."""
    import mentor_questions

    payload = state(
        trades=[
            trade_row("T-1", symbol="AAPL", day="2026-09-11"),
            trade_row("T-2", symbol="MSFT", day="2026-09-11"),
        ],
        open_positions=[trade_row("P-1", symbol="TLT", day="2026-08-17", status="OPEN")],
        grader_gaps=[
            {"question_id": "gap-1", "subject_id": "gap-1", "options": ("yes", "no")}
        ],
        likes=[
            like_row(
                "like-quick-traded",
                symbol="AMD",
                day="2026-09-11",
                like_mode="quick",
                matched_trade_id="T-1",
            )
        ],
    )

    result = mentor_questions.pending(payload, slot_at(SESSION, 11))
    owed = keys_of(result.asked) | keys_of(result.carried)

    assert {kind for kind, _ in owed} & {
        "trade_origin",
        "open_position_check",
        "grader_gap",
        "quick_like_followup",
    } == set()


def test_a_dormant_kind_is_still_fully_described_and_names_its_packet():
    """*"REGISTERED with their trigger, options, `writes`, `answer_key` and the
    consumer they WILL have"*. A kind that quietly vanished would be a question
    nobody decided to stop asking, and the seam TJ-10 and TJ-12 need would be
    gone with it."""
    import mentor_questions

    report = {row["kind"]: row for row in mentor_questions.consumer_report()}

    for name, packet in (
        ("trade_origin", "TJ-12"),
        ("open_position_check", "TJ-12"),
        ("grader_gap", "TJ-10"),
        # Review blocker 2: the key IS read, but off another store's rows.
        ("quick_like_followup", "TJ-14C"),
    ):
        kind = mentor_questions.kind_named(name)
        assert report[name]["dormant"] is True
        assert report[name]["dormant_until"] == packet
        assert kind.consumer and kind.answer_key and kind.writes
        assert callable(kind.trigger)
        assert kind.options, f"{name} carries no clicks"


def test_a_dormant_kinds_trigger_still_answers_so_the_seam_stays_testable():
    """Dormant is a card rule, not a dead function: TJ-12 wakes it by clearing
    one field, not by writing a trigger."""
    import dataclasses

    import mentor_questions

    payload = state(trades=[trade_row("T-1", symbol="AAPL", day="2026-09-11")])
    kind = mentor_questions.kind_named("trade_origin")

    assert keys_of(kind.trigger(payload)) == {("trade_origin", "T-1")}

    woken = tuple(
        dataclasses.replace(item, dormant_until="") if item.kind == "trade_origin" else item
        for item in mentor_questions.REGISTRY
    )
    original = mentor_questions.REGISTRY
    try:
        mentor_questions.REGISTRY = woken
        result = mentor_questions.pending(payload, slot_at(SESSION, 11))
    finally:
        mentor_questions.REGISTRY = original

    assert ("trade_origin", "T-1") in keys_of(result.asked) | keys_of(result.carried)


# ---------------------------------------------------------------------------
# 1b. the consumer probe's teeth, after the review sharpened them
# ---------------------------------------------------------------------------


def reader_that_only_mentions_the_key_in_a_comment(row):
    # claimed_setup_id
    """A reader that names the key in a docstring: claimed_setup_id."""
    return row


def reader_that_only_returns_the_key_as_a_string(row):
    return "trade_origin"


def reader_that_actually_subscripts_the_key(row):
    return row["trade_origin"]


def test_the_probe_refuses_a_key_that_only_appears_in_a_comment_or_a_docstring():
    """The reviewer's first fooler. A text search passes it; a registry whose
    check can be satisfied by a comment is not a check."""
    import dataclasses

    import mentor_questions

    real = mentor_questions.kind_named("quick_like_followup")
    planted = dataclasses.replace(
        real,
        consumer=f"{__name__}.reader_that_only_mentions_the_key_in_a_comment",
        answer_key="claimed_setup_id",
    )

    row = mentor_questions.consumer_report([planted])[0]

    assert row["imports"] is True
    assert row["reads"] is False


def test_the_probe_refuses_a_key_that_is_only_a_bare_string_the_reader_returns():
    """The reviewer's second fooler: the body is `return "trade_origin"`. The
    key is in the source and nothing reads an answer with it."""
    import dataclasses

    import mentor_questions

    real = mentor_questions.kind_named("trade_origin")
    planted = dataclasses.replace(
        real, consumer=f"{__name__}.reader_that_only_returns_the_key_as_a_string"
    )

    row = mentor_questions.consumer_report([planted])[0]

    assert row["imports"] is True
    assert row["reads"] is False


def test_the_probe_still_accepts_a_reader_that_really_subscripts_the_key():
    """And the sharper probe must not refuse a real reader - a check that
    cannot pass is as useless as one that cannot fail."""
    import dataclasses

    import mentor_questions

    real = mentor_questions.kind_named("trade_origin")
    planted = dataclasses.replace(
        real, consumer=f"{__name__}.reader_that_actually_subscripts_the_key"
    )

    row = mentor_questions.consumer_report([planted])[0]

    assert row["imports"] is True
    assert row["reads"] is True


def test_every_live_kind_still_passes_the_sharper_probe():
    """The probe was tightened after the registry was written, so the live
    kinds are re-walked here rather than assumed."""
    import mentor_questions

    broken = {
        row["kind"]: row["reason"]
        for row in mentor_questions.consumer_report()
        if not row["dormant"] and not (row["imports"] and row["reads"])
    }

    assert broken == {}


# ---------------------------------------------------------------------------
# 2. the quick-like follow-up writes no tag
# ---------------------------------------------------------------------------


def test_a_quick_like_answer_writes_no_annotation_row_at_all(lifted, tmp_path):
    """The invariant behind the tester's last line, pinned the way the store
    can actually answer it.

    `annotation_state` reads `confirmed` for a trade with NO row, so the tag
    status cannot tell "nothing written" from "the trader confirmed it". What
    CAN: the setup tags are still empty and the annotation table is still empty.
    A machine writes only `provisional` / `needs_review`, and this one writes
    neither.
    """
    import mentor_questions

    store = new_store(tmp_path)
    row = like_row(
        "like-quick-traded",
        symbol="AMD",
        day="2026-09-11",
        like_mode="quick",
        matched_trade_id="T-1",
    )
    payload = state(likes=[row])
    result = mentor_questions.pending(payload, slot_at(SESSION, 11))
    subject = next(
        item
        for item in list(result.asked) + list(result.carried)
        if item.kind == "quick_like_followup"
    )

    mentor_questions.record_answer(
        subject, {"state": "avwap_reclaim"}, store=store, now=pacific(SESSION, 11)
    )

    assert str(store.annotation_state("T-1").get("setup_tags") or "") == ""
    with store.connection() as conn:
        rows = conn.execute("SELECT COUNT(*) FROM trade_annotations").fetchone()[0]
    assert rows == 0, "a machine wrote into trade_annotations, which the trader owns"


def test_stop_asking_this_is_never_stored_as_an_answer(lifted, tmp_path):
    """A retirement is a statement about the QUESTION, not about the subject.
    Filing it as the answer would put `stop_asking_this` where a setup name
    belongs and count it forever."""
    import mentor_questions

    store = new_store(tmp_path)
    payload = state(
        likes=[
            like_row(
                "like-quick-traded",
                symbol="AMD",
                day="2026-09-11",
                like_mode="quick",
                matched_trade_id="T-1",
            )
        ]
    )
    result = mentor_questions.pending(payload, slot_at(SESSION, 11))
    subject = next(
        item
        for item in list(result.asked) + list(result.carried)
        if item.kind == "quick_like_followup"
    )

    outcome = mentor_questions.record_answer(
        subject,
        {"state": mentor_questions.STOP_ASKING},
        store=store,
        now=pacific(SESSION, 11),
    )

    assert outcome["ok"] is False
    assert store.list_opportunity_events(limit=100) == []


def test_a_prediction_is_never_filed_by_the_registry(tmp_path):
    """TJ-14A's click is filed WITH the read, in one row, by the card's own
    `submit`. A second writer would be a second opinion about what was clicked
    - and would file a prediction with no observation beside it."""
    import mentor_questions

    store = new_store(tmp_path)
    subject = mentor_questions.Subject(kind="prediction_m5", subject_id="slot-1")

    outcome = mentor_questions.record_answer(
        subject, {"state": "up"}, store=store, now=pacific(SESSION, 11)
    )

    assert outcome["ok"] is False
    assert store.list_opportunity_events(limit=100) == []


# ---------------------------------------------------------------------------
# 3. the card's question area
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6", reason="the Mentor card is Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402


class _Service:
    def __init__(self) -> None:
        self.retired: list[tuple[str, str]] = []

    def stop_asking(self, kind: str, subject_id: str) -> None:
        self.retired.append((kind, subject_id))


@pytest.fixture()
def card(tmp_path):
    QApplication.instance() or QApplication([])
    from ui.widgets.trade_mentor_card import TradeMentorCard

    widget = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    widget._clock = lambda: pacific(SESSION, 11)
    try:
        yield widget
    finally:
        widget.deleteLater()


def _three_questions():
    return state(
        likes=[
            like_row(
                f"like-{index}",
                symbol=symbol,
                day="2026-09-11",
                like_mode="quick",
                matched_trade_id=f"T-{index}",
            )
            for index, symbol in enumerate(("AMD", "INTC", "NVDA", "SMCI"))
        ],
        ai_question={"question": "Did the open drive hold?", "options": ["Yes", "No"]},
    )


@pytest.mark.qt
def test_the_card_draws_at_most_three_questions_and_says_what_is_waiting(lifted, card, tmp_path):
    """Five owed, three drawn, the rest counted. Never a fourth widget."""
    import mentor_questions

    store = new_store(tmp_path)
    result = mentor_questions.pending(_three_questions(), slot_at(SESSION, 11))
    assert len(result.asked) == 3 and result.carried

    card.show_slot(slot_at(SESSION, 11))
    card.set_questions(result, store=store, service=_Service())

    assert len(card._question_inputs) == 3
    assert "waiting" in card.questions_label.text().lower()
    assert card.save_questions_button.isVisibleTo(card)


@pytest.mark.qt
def test_stop_asking_this_goes_to_the_service_and_stores_no_answer(lifted, card, tmp_path):
    """The service is the single writer of what the trader silenced, and a
    retirement leaves the answer stores untouched."""
    import mentor_questions

    store = new_store(tmp_path)
    service = _Service()
    result = mentor_questions.pending(_three_questions(), slot_at(SESSION, 11))
    subject = result.asked[0]

    card.show_slot(slot_at(SESSION, 11))
    card.set_questions(result, store=store, service=service)
    combo = card.question_box(subject.kind, subject.subject_id)
    combo.setCurrentIndex(combo.findData(mentor_questions.STOP_ASKING))
    outcome = card.save_questions()

    assert outcome["retired"] == 1 and outcome["saved"] == 0
    assert service.retired == [(subject.kind, subject.subject_id)]
    assert store.list_opportunity_events(limit=100) == []


@pytest.mark.qt
def test_one_click_files_the_answer_under_the_key_the_registry_declares(lifted, card, tmp_path):
    """*"I'm happy to click boxes ... but then I expect the AI to take it from
    there"*. One click, one stored row, no manual step."""
    import json

    import mentor_questions

    store = new_store(tmp_path)
    result = mentor_questions.pending(_three_questions(), slot_at(SESSION, 11))
    subject = next(item for item in result.asked if item.kind == "quick_like_followup")

    # A name off the CLAIM VOCABULARY the registry built this question's clicks
    # from, never one typed here: the vocabulary is read from the two stores
    # that own it and a literal would drift from both.
    import trade_mentor_trade_check as check

    chosen = next(name for name in check.setup_vocabulary() if name in subject.options)

    card.show_slot(slot_at(SESSION, 11))
    card.set_questions(result, store=store, service=_Service())
    combo = card.question_box(subject.kind, subject.subject_id)
    assert combo.findData(chosen) >= 0, "the claim vocabulary is not on the card"
    combo.setCurrentIndex(combo.findData(chosen))
    outcome = card.save_questions()

    assert outcome["saved"] == 1, outcome
    key = mentor_questions.kind_named("quick_like_followup").answer_key
    stored = [
        event
        for event in store.list_opportunity_events(limit=100)
        if key in json.dumps(event.get("payload") or {}, default=str)
    ]
    assert stored and stored[0]["payload"][key] == chosen
    assert subject.subject_id in json.dumps(stored[0]["payload"], default=str)


@pytest.mark.qt
def test_a_question_left_on_the_dash_files_nothing(lifted, card, tmp_path):
    """A card the trader closed without touching is not four answers."""
    import mentor_questions

    store = new_store(tmp_path)
    result = mentor_questions.pending(_three_questions(), slot_at(SESSION, 11))

    card.show_slot(slot_at(SESSION, 11))
    card.set_questions(result, store=store, service=_Service())
    outcome = card.save_questions()

    assert outcome["ok"] is False
    assert store.list_opportunity_events(limit=100) == []


# ---------------------------------------------------------------------------
# 4. the desk seam - where wave 1 lost two review rounds
# ---------------------------------------------------------------------------


@pytest.mark.qt
def test_an_ordinary_card_on_the_real_desk_carries_its_questions(tmp_path, monkeypatch):
    """Driven through `MainWindow._show_trade_mentor_prompt`, the slot
    `promptDue` is connected to - not through the widget.

    Last night's coaching question is the cheapest gap to stage - it needs no
    fill and no like - and it is the one the card printed forever with no way
    to answer it. NO MODEL RUNS HERE: the narration file is written by hand.
    """
    import json as _json

    import journal_store as journal_store_module
    import project_paths
    from ui.app import MainWindow
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    store = new_store(tmp_path)
    monkeypatch.setattr(journal_store_module, "JournalStore", lambda *a, **k: store)

    narrations = tmp_path / "narrations"
    narrations.mkdir()
    (narrations / f"{SESSION.isoformat()}.json").write_text(
        _json.dumps(
            {
                "narration": {
                    "mentor_question": "Did the open drive hold?",
                    "mentor_question_options": ["Yes", "No", "Partly"],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(project_paths, "MARKET_STORY_NARRATIONS_DIR", narrations)

    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        window._show_trade_mentor_prompt(slot_at(SESSION, 11))
        desk_card = window.trading_panel.alert_center.chart_review.mentor_card
        combo = desk_card.question_box("ai_question", SESSION.isoformat())
        assert combo is not None, "the ordinary 11:00 card carried no question"
        assert combo.findData("Partly") >= 0, "the night's click options are not on the card"
        assert desk_card.questions_box.isVisibleTo(desk_card)
    finally:
        window.close()


@pytest.mark.qt
def test_the_overnight_question_stops_being_printed_once_it_is_a_click(card, tmp_path):
    """Before TJ-14B `latest_coaching_question` put the same sentence on EVERY
    card forever, with no options and no answer. Printing it beside the click
    would ask it twice."""
    import mentor_questions

    store = new_store(tmp_path)
    card.show_slot(slot_at(SESSION, 11))
    card.coaching_label.setText("One thing to test: Did the open drive hold?")
    card.coaching_label.setVisible(True)

    payload = state(ai_question={"question": "Did the open drive hold?", "options": ["Yes"]})
    card.set_questions(
        mentor_questions.pending(payload, slot_at(SESSION, 11)),
        store=store,
        service=_Service(),
    )

    assert card.coaching_label.isVisibleTo(card) is False
    assert card.question_box("ai_question", SESSION.isoformat()) is not None
