r"""TJ-7 review round 1 - the four advisories the lead sent back. RED FIRST.

Reviewer round 1 gave TJ-7 a GO with no blockers. These are its advisories, as
the lead scoped them:

**A1 - a clicked face can be taken back.** The faces were an exclusive
`QButtonGroup`, and Qt will not un-check the checked member of one. Meanwhile
`TradeMentorCard.save_questions` reads "a score is set" as "the trader touched
the strip", so a MIS-CLICK filed a mood the trader never meant and there was no
way back to "nothing selected" except saving it. A second click on the SELECTED
face now clears it, the way a chip toggles off, on BOTH surfaces - and after a
clear the strip is indistinguishable from one nobody touched.

**A4 - the machine may not file a mood, behaviourally.** The fence was purely
structural (an AST scan of the call sites). `build_entry` now refuses a mood on
a machine-written row through `is_machine_entry`'s OWN rule, and
`is_publishable` asks the same question again at the gate - the pattern TJ-14A
set for a mismatched prediction horizon.

**A6 - one id for one row.** The Day Review payload builds its mood section with
its own `_Minter`, so its `source_id`s had to be PROVEN equal to the pack's
rather than assumed. Both go through `day_review_pack.mood_source_id`, which
derives an id from the row and from nothing else.

**A2 - `MOOD_ITEM_FIELDS` is the truth and is USED.** It listed eight names
while the items carried nine. Each item is now projected THROUGH the constant,
so a field nobody named there does not ship.

RED BEFORE THE FIX (proven by restoring `mood_strip.py`, `market_journal.py`,
`day_review_pack.py` and `day_review_service.py` from `ae7c06c7`).
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
pytest.importorskip("PySide6", reason="the strip is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from tj14b_support import SESSION, last_slot, new_store, pacific  # noqa: E402


class _Service:
    def __init__(self) -> None:
        self.retired: list[tuple[str, str]] = []

    def stop_asking(self, kind: str, subject_id: str) -> None:
        self.retired.append((str(kind), str(subject_id)))


class _Result:
    def __init__(self, subject) -> None:
        self.asked = (subject,)
        self.forced = ()
        self.carried = ()
        self.waiting_note = ""


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


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
    subject = fx.day_close_subject(session=SESSION.isoformat())
    store = new_store(tmp_path)
    card.show_slot(last_slot(SESSION))
    card.set_questions(_Result(subject), store=store, service=service or _Service())
    return subject, store


# ---------------------------------------------------------------------------
# A1 - a misclick is not an answer
# ---------------------------------------------------------------------------
def test_a_second_click_on_the_chosen_face_clears_it(qapp):
    """Click, click = nothing selected. The state an untouched strip is in."""
    from ui.widgets.mood_strip import MoodStrip

    strip = MoodStrip()
    try:
        strip.mood_button(3).click()
        assert strip.answer()["mood"] == 3
        strip.mood_button(3).click()
        assert strip.answer()["mood"] is None, "a misclick must be takeable back"
        assert strip.is_touched() is False
        assert strip.mood_button(3).isChecked() is False
    finally:
        strip.deleteLater()
        qapp.processEvents()


def test_one_face_at_a_time_survives_the_change(qapp):
    """Clearing a face may not cost the "one score" rule."""
    from ui.widgets.mood_strip import MoodStrip

    strip = MoodStrip()
    try:
        strip.mood_button(2).click()
        strip.mood_button(5).click()
        assert strip.answer()["mood"] == 5
        assert strip.mood_button(2).isChecked() is False
        assert [
            score for score, button in strip._faces.items() if button.isChecked()
        ] == [5]
    finally:
        strip.deleteLater()
        qapp.processEvents()


def test_a_cleared_face_on_the_mentor_files_nothing_at_all(card, tmp_path, monkeypatch):
    """The advisory's own case: a mood the trader took back is NOT filed."""
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
    card.mood_button(4).click()

    assert card.mood_answer()["mood"] is None
    outcome = card.save_questions()

    assert calls == [], "a mood that was taken back was filed anyway"
    assert outcome.get("saved", 0) == 0
    assert outcome["ok"] is False


def test_a_cleared_strip_survives_the_same_questions_being_offered_again(card, tmp_path):
    """The merge keeps a CLEARED strip cleared - it does not resurrect a face."""
    subject, store = _day_close_card(card, tmp_path)
    button = card.mood_button(5)
    button.click()
    button.click()

    card.set_questions(_Result(subject), store=store, service=_Service())

    assert card.mood_button(5) is button, "the SAME widget object"
    assert card.mood_answer()["mood"] is None
    assert button.isChecked() is False


def test_on_the_journal_tab_a_cleared_face_writes_the_untouched_row(qapp, monkeypatch):
    """Byte for byte: a save after a clear is the row the trader always wrote."""
    from ui.panels.day_review_panel import DayReviewPanel

    class _Recording:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def read_day(self, session_date, **_kwargs):
            return {"session_date": session_date}

        def write_entry(self, **kwargs):
            self.calls.append(dict(kwargs))
            return {"ok": True, "entry": {"entry_id": "mj-1", **kwargs}}

    service = _Recording()
    panel = DayReviewPanel(service=service, clock=lambda: fx.pacific(12, 55))
    monkeypatch.setattr(panel, "reload", lambda: None)
    monkeypatch.setattr(panel, "_refresh_if_loaded", lambda: None)
    try:
        panel.entry_text.setPlainText("Range day; nothing to do.")
        panel._save()
        untouched = dict(service.calls[-1])

        panel.entry_text.setPlainText("Range day; nothing to do.")
        panel.mood_button(1).click()
        panel.mood_button(1).click()
        panel._save()
        after_a_clear = dict(service.calls[-1])

        assert after_a_clear == untouched
        assert not after_a_clear.get("mood")
        assert not tuple(after_a_clear.get("state_tags") or ())
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        qapp.processEvents()


def test_a_cleared_face_still_stores_the_key_present_and_empty():
    """The writer's half of A1: nothing clicked and nothing UN-clicked agree."""
    import market_journal

    untouched = market_journal.build_entry(
        text="Range day.", session_date=fx.SESSION, now=fx.pacific(12, 55)
    )
    cleared = market_journal.build_entry(
        text="Range day.",
        session_date=fx.SESSION,
        now=fx.pacific(12, 55),
        mood=None,
        state_tags=(),
    )

    assert cleared == untouched
    assert cleared["mood"] == {}


# ---------------------------------------------------------------------------
# A4 - the machine may not file a mood
# ---------------------------------------------------------------------------
def test_the_writer_refuses_a_mood_on_a_machine_row():
    """A mood is the trader's own click. An auto-mode flip has no feelings."""
    import market_journal

    for origin in market_journal.MACHINE_ORIGINS:
        with pytest.raises(market_journal.MoodFieldError):
            market_journal.build_entry(
                text="Auto mode: DESK -> AWAY",
                session_date=fx.SESSION,
                now=fx.pacific(12, 55),
                origin=origin,
                mood=3,
            )
        # ... and the same row with nothing clicked is written exactly as before.
        plain = market_journal.build_entry(
            text="Auto mode: DESK -> AWAY",
            session_date=fx.SESSION,
            now=fx.pacific(12, 55),
            origin=origin,
        )
        assert plain["mood"] == {}
        assert market_journal.is_machine_entry(plain) is True


def test_the_refusal_uses_the_one_rule_for_what_a_machine_row_is(monkeypatch):
    """`is_machine_entry` owns that rule (TJ-1), so a NEW machine origin is
    fenced the day it is added - not the day someone remembers this check."""
    import market_journal

    monkeypatch.setattr(
        market_journal, "MACHINE_ORIGINS", ("a_new_machine_origin",), raising=True
    )
    with pytest.raises(market_journal.MoodFieldError):
        market_journal.build_entry(
            text="Something the desk did.",
            session_date=fx.SESSION,
            now=fx.pacific(12, 55),
            origin="a_new_machine_origin",
            mood=2,
        )


def test_the_same_refusal_is_asked_again_at_the_publish_gate():
    """A row assembled as a dict literal never calls `build_entry`."""
    import market_journal

    row = fx.row_with_a_mood(entry_id="mj-machine")
    row["origin"] = market_journal.MACHINE_ORIGINS[0]

    ok, reason = market_journal.is_publishable(row)
    assert ok is False
    assert "mood" in reason.lower(), reason

    # The same machine row with the key present and EMPTY is publishable.
    row["mood"] = {}
    assert market_journal.is_publishable(row)[0] is True


def test_a_traders_own_row_is_untouched_by_the_machine_fence():
    import market_journal

    entry = market_journal.build_entry(
        text="Followed the plan? partly",
        session_date=fx.SESSION,
        now=fx.pacific(12, 55),
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        mood=4,
        state_tags=("calm",),
    )
    assert entry["mood"]["score"] == 4
    assert market_journal.is_publishable(entry)[0] is True


# ---------------------------------------------------------------------------
# A2 - the constant is the truth, and it is used
# ---------------------------------------------------------------------------
def test_every_mood_item_carries_exactly_the_named_fields():
    import day_review_pack

    section = day_review_pack.mood_section(
        [
            fx.row_with_a_mood(entry_id="mj-a", stamp=fx.pacific(7, 30)),
            fx.row_with_a_mood(entry_id="mj-b", stamp=fx.pacific(12, 55)),
        ]
    )

    assert section["n"] == 2
    for item in section["recorded"]:
        assert tuple(item) == tuple(day_review_pack.MOOD_ITEM_FIELDS), item


def test_a_field_nobody_named_does_not_ship(monkeypatch):
    """The constant is LOAD-BEARING, not decorative: drop a name from it and
    the rows lose that field. A constant the builder ignores drifts."""
    import day_review_pack

    narrowed = tuple(
        name for name in day_review_pack.MOOD_ITEM_FIELDS if name != "vocab_version"
    )
    monkeypatch.setattr(day_review_pack, "MOOD_ITEM_FIELDS", narrowed)

    item = day_review_pack.mood_section([fx.row_with_a_mood(entry_id="mj-a")])["recorded"][0]
    assert tuple(item) == narrowed
    assert "vocab_version" not in item


# ---------------------------------------------------------------------------
# A6 - one row, one id, wherever the section is built
# ---------------------------------------------------------------------------
def test_a_mood_id_is_derived_from_its_row_and_from_nothing_else():
    import day_review_pack

    assert day_review_pack.mood_source_id("mj-a") == "mood:mj-a"
    assert day_review_pack.mood_source_id("", 4) == "mood:4"


def test_the_pack_and_a_standalone_section_mint_the_same_ids():
    """The payload builds its own section with its own minter. The ids it gets
    must be the ids the PACK gives the same rows, or a citation resolves in one
    place and not the other - a trap the next packet would walk into."""
    import day_review_pack
    import tj4_support as fx4

    moods = [
        fx.row_with_a_mood(entry_id="mj-a", stamp=fx.pacific(7, 30)),
        fx.row_with_a_mood(entry_id="mj-b", stamp=fx.pacific(12, 55)),
    ]
    inputs = fx4.pack_inputs()
    entries = list(inputs["entries"]) + moods
    inputs["entries"] = entries

    pack = day_review_pack.build_pack(fx.SESSION, now=fx4.AFTER_THE_CLOSE, **inputs)
    standalone = day_review_pack.mood_section(entries)

    assert pack["mood"]["recorded"] == standalone["recorded"]
    assert [item["source_id"] for item in standalone["recorded"]] == [
        "mood:mj-a",
        "mood:mj-b",
    ]
    for item in standalone["recorded"]:
        assert item["source_id"] in day_review_pack.allowed_source_ids(pack)
