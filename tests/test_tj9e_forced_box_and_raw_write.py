r"""TJ-9E - the exit is ONE forced box, and the words hit disk FIRST.

Lead decisions 2 and 3: *"the exit ask is ONE box, and it is forced like the
rest ... No chips, no dropdowns on the exit - 'just write it out'"* and *"Raw
first, always ... A failed write fails LOUDLY ... A second note on the same exit
supersedes by append, never rewrites."*

The Save gate is driven through the REAL widget offscreen - the button's own
`isEnabled()` after real text entry - exactly as `tests/test_tj9_forced_trade_labels.py`
drives it for the four entry fields.

RED FOR (measured on `claude/tj9e-exit-notes` at `05988440`):

* `test_the_new_event_type_is_registered_with_the_journals_own_vocabulary` fails
  on a real assertion - `EXIT_NOTE_RAW` is not in
  `journal_store.OPPORTUNITY_EVENT_TYPES` (`scripts/journal_store.py:107-129`),
  so `record_opportunity_event` raises
  ``ValueError: unsupported opportunity event type`` (`:2513`);
* everything naming `save_exit_note`, `exit_notes`, `EXIT_PROMPT`,
  `EVENT_EXIT_NOTE_RAW` or `card.exit_note_box` fails with `AttributeError`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj9e_support as fx  # noqa: E402

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Trade Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402

_app = QApplication.instance() or QApplication([])

MONDAY_MORNING = datetime.fromisoformat("2026-09-14T09:05:00-04:00")


def _card(tmp_path):
    from ui.widgets.trade_mentor_card import TradeMentorCard

    return TradeMentorCard(drafts_path=Path(tmp_path) / "drafts.json")


def _raw_rows(store, trade_id: str) -> list[dict]:
    import trade_mentor_trade_check as check

    return store.list_opportunity_events(
        trade_id=str(trade_id), event_type=check.EVENT_EXIT_NOTE_RAW, limit=10000
    )


# ---------------------------------------------------------------------------
# the store's own vocabulary
# ---------------------------------------------------------------------------
def test_the_new_event_type_is_registered_with_the_journals_own_vocabulary():
    """ONE new name in `OPPORTUNITY_EVENT_TYPES`, beside `RECALLED_RAW`.

    `record_opportunity_event` refuses any type outside that set
    (`scripts/journal_store.py:2512-2514`), so an unregistered name is not a
    quiet nothing - it raises, and the trader's words are lost.
    """
    import journal_store
    import trade_mentor_trade_check as check

    assert check.EVENT_EXIT_NOTE_RAW in journal_store.OPPORTUNITY_EVENT_TYPES
    assert check.EVENT_EXIT_NOTE_RAW != check.EVENT_RECALLED_RAW
    # The two that already exist are untouched - this packet ADDS one name.
    assert {"RECALLED", "RECALLED_RAW"} <= journal_store.OPPORTUNITY_EVENT_TYPES


# ---------------------------------------------------------------------------
# raw first, append-only, loud
# ---------------------------------------------------------------------------
def test_the_words_are_on_disk_the_moment_save_is_pressed(tmp_path):
    """Hand-counted: ONE `EXIT_NOTE_RAW` row, holding the note character for
    character, the trade, the session the exit is in and its provenance."""
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)
    check.save_exit_note(
        store, ids[fx.SWING], fx.EXIT_NOTE, exit_session=fx.REVIEWED, now=MONDAY_MORNING
    )

    rows = _raw_rows(store, ids[fx.SWING])
    assert len(rows) == 1, rows
    payload = rows[0]["payload"]
    assert payload["raw_text"] == fx.EXIT_NOTE
    assert payload["exit_session"] == fx.REVIEWED
    assert "label_provenance" in payload and "written_after_the_session" in payload
    assert rows[0]["trade_id"] == ids[fx.SWING]


def test_a_second_note_supersedes_by_append_and_the_first_row_is_byte_identical(tmp_path):
    """Hand-counted: 2 rows, oldest first, and row 0 is UNCHANGED.

    `opportunity_events` is immutable by design (`scripts/journal_store.py:973`,
    `:1014`). A correction that rewrote the first row would delete the fact that
    the trader changed their mind, which is the only interesting thing about it.
    """
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)
    check.save_exit_note(
        store, ids[fx.SWING], fx.EXIT_NOTE, exit_session=fx.REVIEWED, now=MONDAY_MORNING
    )
    before = _raw_rows(store, ids[fx.SWING])
    assert len(before) == 1
    first = dict(before[0])

    check.save_exit_note(
        store,
        ids[fx.SWING],
        fx.SECOND_EXIT_NOTE,
        exit_session=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T09:31:00-04:00"),
    )
    after = _raw_rows(store, ids[fx.SWING])
    assert len(after) == 2, after
    assert dict(after[0]) == first, "the first note was rewritten"
    assert after[1]["payload"]["raw_text"] == fx.SECOND_EXIT_NOTE

    # And the reader hands them back oldest first, so "the note" is the LAST.
    notes = check.exit_notes(store, ids[fx.SWING])
    assert [row["raw_text"] for row in notes] == [fx.EXIT_NOTE, fx.SECOND_EXIT_NOTE]


def test_an_empty_note_is_refused_and_a_failed_write_is_loud(tmp_path):
    """A journal write is the ONE evidence store that may fail loudly (CLAUDE.md).

    Two refusals: a blank note is not a note, and a store that cannot append
    RAISES rather than returning a dict nobody checks. An exit note swallowed
    into a log is the trader's words gone.
    """
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)

    with pytest.raises(ValueError):
        check.save_exit_note(
            store, ids[fx.SWING], "   ", exit_session=fx.REVIEWED, now=MONDAY_MORNING
        )

    class _Broken:
        def __getattr__(self, name):
            return getattr(store, name)

        def record_opportunity_event(self, **_kwargs):
            raise OSError("the journal database is locked")

    with pytest.raises(OSError):
        check.save_exit_note(
            _Broken(), ids[fx.SWING], fx.EXIT_NOTE, exit_session=fx.REVIEWED, now=MONDAY_MORNING
        )


def test_the_note_survives_an_entry_answer_that_fails(tmp_path, monkeypatch):
    """RAW FIRST, proved by ORDER rather than by reading the source.

    The card saves the exit note and the four entry answers on one click. The
    entry half is made to raise; the note must already be on disk. Hand-counted:
    1 `EXIT_NOTE_RAW` row after a save that ended in an exception.
    """
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    card = _card(tmp_path)
    card.set_trade_check(task, store=store)
    card.exit_note_box(ids[fx.DAY_TRADE]).setPlainText(fx.EXIT_NOTE)
    for name in check.MATERIAL_FIELDS:
        combo = card._answer_inputs[ids[fx.DAY_TRADE]][name][0]
        combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))

    def _boom(*_args, **_kwargs):
        raise OSError("the annotation write failed")

    monkeypatch.setattr(check, "save_answers", _boom)
    try:
        card.save_trade_check()
    except OSError:
        pass

    rows = _raw_rows(store, ids[fx.DAY_TRADE])
    assert len(rows) == 1, "the words were lost when the entry half failed"
    assert rows[0]["payload"]["raw_text"] == fx.EXIT_NOTE


# ---------------------------------------------------------------------------
# the forced box, through the real widget
# ---------------------------------------------------------------------------
def test_save_stays_grey_until_the_exit_box_holds_words_or_an_answer_state(tmp_path):
    """SWNG is the exit-ONLY row: nothing to answer about the entry, so the exit
    box is the WHOLE gate.

    Hand-counted: Save is grey with an empty box, green once words are typed,
    grey again when they are deleted, and green on an explicit answer state -
    which is a complete answer, not a blank. The state names are read from the
    module, never spelled here.
    """
    import trade_mentor_trade_check as check

    store, trade_id = fx.swing_only(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)
    assert len(task.trades) == 1, [(q.symbol, q.missing) for q in task.trades]

    card = _card(tmp_path)
    card.set_trade_check(task, store=store)
    box = card.exit_note_box(trade_id)

    assert card.save_answers_button.isVisibleTo(card) is True
    assert card.save_answers_button.isEnabled() is False, "nothing written yet"

    box.setPlainText(fx.EXIT_NOTE)
    assert card.save_answers_button.isEnabled() is True

    box.setPlainText("   ")
    assert card.save_answers_button.isEnabled() is False, "whitespace is not an answer"

    card.set_exit_answer_state(trade_id, check.ANSWER_NOT_REMEMBERED)
    assert card.save_answers_button.isEnabled() is True

    card.set_exit_answer_state(trade_id, "")
    assert card.save_answers_button.isEnabled() is False, "the gate is a state, not a latch"


def test_the_exit_ask_is_one_prompt_and_carries_no_chips_or_dropdowns(tmp_path):
    """*"just write it out"*. The prompt is the trader's own three questions, and
    the exit half of a row holds a text box and nothing to pick from.

    The prompt string is read from the module rather than spelled here, and
    asserted to name all three things it asks about.
    """
    import trade_mentor_trade_check as check

    store, ids = fx.ready_store(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)

    card = _card(tmp_path)
    card.set_trade_check(task, store=store)

    prompt = str(check.EXIT_PROMPT)
    assert "exit" in prompt.lower() and "feel" in prompt.lower() and "watch" in prompt.lower()
    assert prompt in card.exit_prompt_text(ids[fx.SWING])

    box = card.exit_note_box(ids[fx.SWING])
    assert hasattr(box, "setPlainText"), type(box)
    # Nothing to PICK on the exit: the three fields are the night's job, and a
    # picklist in front of the trader at 09:00 is the form this packet refuses.
    assert box.parent() is not None
    assert box.parent().findChildren(QComboBox) == [], "a dropdown appeared on the exit"


def test_the_entry_half_of_a_row_is_untouched_by_the_exit_box(tmp_path):
    """STATED GUARD for *"entrys are good the way they are"*.

    DAYT carries both halves. Hand-counted: 4 entry combos, in `MATERIAL_FIELDS`
    order, and answering all four does NOT open Save while the exit box is empty.
    """
    import trade_mentor_trade_check as check

    store, trade_id = fx.day_trade_only(tmp_path)
    task = check.build_task(store, fx.SESSION_TODAY)
    assert len(task.trades) == 1

    card = _card(tmp_path)
    card.set_trade_check(task, store=store)

    fields = card._answer_inputs[trade_id]
    assert tuple(fields) == check.MATERIAL_FIELDS, tuple(fields)
    for name in check.MATERIAL_FIELDS:
        combo = fields[name][0]
        combo.setCurrentIndex(combo.findData(check.ANSWER_NOT_REMEMBERED))

    assert card.save_answers_button.isEnabled() is False, (
        "the exit box is still empty and Save opened anyway"
    )

    card.exit_note_box(trade_id).setPlainText(fx.EXIT_NOTE)
    assert card.save_answers_button.isEnabled() is True
