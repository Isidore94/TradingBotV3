r"""TJ-7 change 2 - two clicks, optional, and never machine-filled. RED (Qt).

`plan.md` §12.4 "TJ-7" change 2 as AMENDED 2026-09-19: *"A two-click strip
(mood 1-5 + up to two chips) on the Trade Mentor popup and the desk's journal
tab; optional, never required, never asked twice for one row. ... on the Trade
Mentor the strip is not its own widget - it is TJ-14's `day_close` question,
asked once on the session's last card beside `Followed the plan`."*

Every test drives the REAL widgets offscreen: `TradeMentorCard.set_questions` /
`save_questions` (the click handlers TJ-14B ships) and `DayReviewPanel._save`
(the desk's journal tab, `day_review_panel.py:2469`). Nothing here builds a
payload by hand and calls a helper with it.

WHAT IS PINNED, AND WHY
-----------------------
* **The strip hangs off `day_close`, which the registry already describes.**
  The subject comes from `mentor_questions._trigger_day_close`, never a Subject
  typed here, so a strip drawn for a kind the registry does not offer fails.
* **Optional means Save never notices it.** TJ-9's Save gate greys out until
  every MATERIAL field of every listed trade holds a value or an answer state;
  a mood is not a material field, so a card with an untouched strip saves
  exactly as it did before. That is asserted through `save_answers_button
  .isEnabled()` on a real trade section, not through a helper.
* **A machine never pre-fills a mood.** The strip opens with nothing selected -
  no remembered face, no default 3, no "same as yesterday". The trader's own
  click or nothing.
* **A section with answer widgets is never rebuilt** (TJ-14B's card rule). A
  mood half-clicked and then re-offered keeps the SAME widget objects and their
  state; a rebuild would silently drop what the trader had already said.
* **Never asked twice for one row**: once the answer is filed the strip goes.

RED FOR: `TradeMentorCard` has no `mood_button` / `state_tag_button` /
`mood_answer` and `DayReviewPanel` has no mood strip (AttributeError), and the
card hands `record_answer` no mood (a real assertion on the payload).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj7_support as fx  # noqa: E402

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Mentor card and the journal tab are Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from tj14b_support import (  # noqa: E402
    REVIEWED,
    SESSION,
    add_round_trip,
    last_slot,
    mark_covered,
    new_store,
    pacific,
)


class _Service:
    """The Mentor service's one method the card uses for a retirement."""

    def __init__(self) -> None:
        self.retired: list[tuple[str, str]] = []

    def stop_asking(self, kind: str, subject_id: str) -> None:
        self.retired.append((str(kind), str(subject_id)))


class _Result:
    """`mentor_questions.pending`'s answer shape, with ONE asked subject."""

    def __init__(self, subject) -> None:
        self.asked = (subject,)
        self.forced = ()
        self.carried = ()
        self.waiting_note = ""


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def card(qapp, tmp_path):
    from ui.widgets.trade_mentor_card import TradeMentorCard

    widget = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    widget._clock = lambda: pacific(SESSION, 12)
    try:
        yield widget
    finally:
        widget.deleteLater()
        qapp.processEvents()


def _day_close_card(card, tmp_path, *, service=None):
    """The session's LAST card, carrying the registry's own day_close subject."""
    import mentor_questions  # noqa: F401 - imported so a failure names the module

    subject = fx.day_close_subject(session=SESSION.isoformat())
    store = new_store(tmp_path)
    card.show_slot(last_slot(SESSION))
    card.set_questions(_Result(subject), store=store, service=service or _Service())
    return subject, store


# ---------------------------------------------------------------------------
# the Trade Mentor's last card
# ---------------------------------------------------------------------------
def test_the_day_close_question_carries_a_mood_strip_with_nothing_pre_selected(
    card, tmp_path
):
    """Five faces, eight chips, and not one of them chosen for the trader."""
    subject, _store = _day_close_card(card, tmp_path)

    assert subject.kind == "day_close"
    for score in (1, 2, 3, 4, 5):
        assert card.mood_button(score) is not None, f"no face for {score}"
    for code in fx.STATE_TAG_CODES:
        assert card.state_tag_button(code) is not None, f"no chip for {code}"

    answer = card.mood_answer()
    assert answer["mood"] is None, "a machine never pre-fills a mood"
    assert tuple(answer["state_tags"]) == ()


def test_a_trader_who_ignores_the_strip_still_files_the_plan_answer(
    card, tmp_path, monkeypatch
):
    """Optional means optional: the plan answer alone is a complete answer."""
    import mentor_questions

    calls: list[tuple] = []

    def _record(subject, answer, **kwargs):
        calls.append((subject, dict(answer)))
        return {"ok": True, "row": {}, "answer_key": "text"}

    monkeypatch.setattr(mentor_questions, "record_answer", _record)

    subject, _store = _day_close_card(card, tmp_path)
    assert card.mood_button(3) is not None, "the strip is drawn beside the question"
    assert card.mood_answer()["mood"] is None, "and the trader ignores it"

    combo = card.question_box(subject.kind, subject.subject_id)
    combo.setCurrentIndex(combo.findData("yes"))
    outcome = card.save_questions()

    assert outcome["saved"] == 1, outcome
    assert len(calls) == 1
    answer = calls[0][1]
    assert answer["state"] == "yes"
    assert answer.get("mood") is None
    assert tuple(answer.get("state_tags") or ()) == ()


def test_a_mood_clicked_with_no_plan_answer_is_still_filed(card, tmp_path, monkeypatch):
    """The other direction. A click the trader made is never thrown away
    because a DIFFERENT question on the same row was left alone."""
    import mentor_questions

    calls: list[dict] = []
    monkeypatch.setattr(
        mentor_questions,
        "record_answer",
        lambda subject, answer, **kwargs: (
            calls.append(dict(answer)) or {"ok": True, "row": {}, "answer_key": "text"}
        ),
    )

    _subject, _store = _day_close_card(card, tmp_path)
    card.mood_button(4).click()
    card.state_tag_button("rushed").click()
    outcome = card.save_questions()

    assert outcome["saved"] == 1, outcome
    assert calls and calls[0]["mood"] == 4
    assert tuple(calls[0]["state_tags"]) == ("rushed",)
    assert not str(calls[0].get("state") or ""), "the plan question was not answered"


def test_at_most_two_chips_can_be_chosen(card, tmp_path):
    """The cap is the vocabulary's, and the STRIP holds it too - a third click
    cannot produce a row the writer will refuse."""
    import trader_state_tags

    _subject, _store = _day_close_card(card, tmp_path)
    card.state_tag_button("rushed").click()
    card.state_tag_button("tired").click()
    card.state_tag_button("fomo").click()

    chosen = tuple(card.mood_answer()["state_tags"])
    assert len(chosen) <= trader_state_tags.MAX_STATE_TAGS
    assert "fomo" not in chosen or len(chosen) == 2


def test_the_strip_never_greys_the_trade_check_save(card, tmp_path):
    """TJ-9's Save gate counts MATERIAL fields. A mood is not one of them, and
    adding the strip must not put a fourth thing in front of Save."""
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAPL", day=REVIEWED, entry_hour=7)

    card.show_slot(last_slot(SESSION))
    card.set_trade_check(check.build_task(store, SESSION), store=store)
    for combo, _text in card._answer_inputs[trade_id].values():
        combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))
    assert card.save_answers_button.isEnabled() is True

    subject = fx.day_close_subject(session=SESSION.isoformat())
    card.set_questions(_Result(subject), store=store, service=_Service())

    assert card.mood_button(3) is not None, "the strip is on this card"
    assert card.save_answers_button.isEnabled() is True, (
        "an untouched optional strip cannot grey out Save"
    )
    card.mood_button(2).click()
    assert card.save_answers_button.isEnabled() is True


def test_a_half_clicked_mood_survives_the_same_questions_being_offered_again(
    card, tmp_path
):
    """A section with ANSWER WIDGETS is never rebuilt (TJ-14B). The card merges
    a fresh task; a rebuild here would drop a face the trader had chosen."""
    subject, store = _day_close_card(card, tmp_path)
    button = card.mood_button(5)
    button.click()
    card.state_tag_button("calm").click()

    card.set_questions(_Result(subject), store=store, service=_Service())

    assert card.mood_button(5) is button, "the SAME widget object"
    assert card.mood_answer()["mood"] == 5
    assert tuple(card.mood_answer()["state_tags"]) == ("calm",)


def test_once_the_answer_is_filed_the_strip_is_not_asked_again(
    card, tmp_path, monkeypatch
):
    """"never asked twice for one row"."""
    import mentor_questions

    monkeypatch.setattr(
        mentor_questions,
        "record_answer",
        lambda subject, answer, **kwargs: {"ok": True, "row": {}, "answer_key": "text"},
    )

    _subject, _store = _day_close_card(card, tmp_path)
    card.mood_button(3).click()
    assert card.save_questions()["saved"] == 1

    assert card.mood_button(3) is None
    assert card.state_tag_button("calm") is None


def test_the_answer_reaches_the_journal_as_a_mood_not_as_a_sentence(monkeypatch):
    """`record_answer` -> `_record_in_journal` -> `write_entry`: the click is
    stored as FIELDS. A mood parsed back out of the row's text later would be a
    second, drifting reader of what the trader clicked."""
    import mentor_questions

    journal = fx.FakeJournal()
    subject = fx.day_close_subject(session=fx.SESSION)
    outcome = mentor_questions.record_answer(
        subject,
        {"state": "partly", "mood": 2, "state_tags": ("tilted",), "note": "chased"},
        journal=journal,
        now=fx.pacific(12, 55),
    )

    assert outcome["ok"] is True
    written = journal.last
    assert written["mood"] == 2
    assert tuple(written["state_tags"]) == ("tilted",)
    assert written["process"]["followed_plan"] == "partly"
    assert written["process"]["note"] == "chased"


# ---------------------------------------------------------------------------
# the desk's journal tab
# ---------------------------------------------------------------------------
class _RecordingService:
    """`market_journal_service` as the page sees it: one writer, one record."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def read_day(self, session_date, **_kwargs):
        return {"session_date": session_date}

    def write_entry(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"ok": True, "entry": {"entry_id": "mj-1", **kwargs}}


@pytest.fixture
def journal_tab(qapp, monkeypatch, tmp_path):
    from ui.panels.day_review_panel import DayReviewPanel

    service = _RecordingService()
    widget = DayReviewPanel(service=service, clock=lambda: fx.pacific(12, 55))
    monkeypatch.setattr(widget, "reload", lambda: None)
    monkeypatch.setattr(widget, "_refresh_if_loaded", lambda: None)
    try:
        yield widget, service
    finally:
        try:
            widget.shutdown()
        except Exception:  # noqa: BLE001
            pass
        widget.deleteLater()
        qapp.processEvents()


def test_the_journal_tab_saves_a_note_with_the_mood_beside_it(journal_tab):
    """The strip is on the tab too, and `_save` hands it to the ONE writer."""
    panel, service = journal_tab

    panel.entry_text.setPlainText("Rushed the open and paid for it.")
    panel.mood_button(2).click()
    panel.state_tag_button("rushed").click()
    panel._save()

    assert service.calls, "nothing was written"
    written = service.calls[-1]
    assert written["text"] == "Rushed the open and paid for it."
    assert written["mood"] == 2
    assert tuple(written["state_tags"]) == ("rushed",)


def test_the_journal_tab_saves_a_note_with_no_mood_exactly_as_before(journal_tab):
    """A trader who ignores the strip writes the row they always wrote."""
    panel, service = journal_tab

    panel.entry_text.setPlainText("Range day; nothing to do.")
    panel._save()

    written = service.calls[-1]
    assert written["text"] == "Range day; nothing to do."
    assert not written.get("mood"), "no mood is no mood - never a default face"
    assert not tuple(written.get("state_tags") or ())


def test_the_journal_tab_strip_starts_blank_on_every_new_note(journal_tab):
    """After a save the strip resets: yesterday's face is not tomorrow's."""
    panel, service = journal_tab

    panel.entry_text.setPlainText("First note.")
    panel.mood_button(5).click()
    panel._save()

    assert panel.mood_answer()["mood"] is None
    assert tuple(panel.mood_answer()["state_tags"]) == ()
