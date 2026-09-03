"""R4 A6 - the note box saves on Enter, and makes a newline on Shift+Enter.

Both note prompts on the desk are `QInputDialog`s in
`UsePlainTextEditForTextInput` mode, which is what makes them multi-line. That
option also hands Return to the editor, so Enter inserted a newline and the only
way to save a note was to reach for the mouse - while the packet that built them
said Enter saves.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _press(editor, *, shift: bool) -> None:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication

    modifier = (
        Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    )
    QApplication.sendEvent(
        editor, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, modifier)
    )


def _prompt(saved):
    from ui.widgets.note_prompt import open_note_prompt

    dialog = open_note_prompt(None, title="T", label="L", on_text=saved.append)
    from PySide6.QtWidgets import QPlainTextEdit

    editor = dialog.findChild(QPlainTextEdit)
    assert editor is not None, "plain-text mode is what makes the box multi-line"
    return dialog, editor


def test_enter_saves_the_note(app):
    saved: list[str] = []
    dialog, editor = _prompt(saved)
    editor.setPlainText("looked heavy into the close")

    _press(editor, shift=False)

    assert saved == ["looked heavy into the close"]
    assert not dialog.isVisible()


def test_shift_enter_makes_a_newline_and_saves_nothing(app):
    saved: list[str] = []
    dialog, editor = _prompt(saved)
    editor.setPlainText("one")
    editor.moveCursor(editor.textCursor().MoveOperation.End)

    _press(editor, shift=True)

    assert saved == [], "Shift+Enter is the multi-line case, not the save"
    assert editor.toPlainText() == "one\n"
    assert dialog.isVisible()


def test_both_note_boxes_on_the_desk_go_through_the_one_helper():
    """Two dialogs with two key rules is the failure this replaces."""
    for path in (
        "scripts/ui/panels/master_avwap_panel.py",
        "scripts/ui/panels/alert_center_panel.py",
    ):
        text = (ROOT / path).read_text(encoding="utf-8")
        assert "open_note_prompt(" in text, path
        assert "UsePlainTextEditForTextInput" not in text, (
            f"{path} still builds its own note dialog"
        )


def test_the_two_corrected_comments_no_longer_claim_a_modeless_deferred_dialog():
    """`open()` is window-modal, and the call is the handler's last statement.

    Both files now SAY window-modal. The word "MODELESS" survives only inside
    the sentence that corrects it, which is the shape every corrected comment on
    this desk takes - the old claim is quoted, not erased.
    """
    for path, prompt in (
        ("scripts/ui/panels/master_avwap_panel.py", "_prompt_for_verdict_note"),
        ("scripts/ui/panels/alert_center_panel.py", "_prompt_for_not_today_note"),
    ):
        text = (ROOT / path).read_text(encoding="utf-8")
        body = text.split(f"def {prompt}(", 1)[1].split("\n    def ", 1)[0]
        assert "WINDOW-MODAL" in body or "window-modal" in body, path
        assert "modeless dialog with no reference" not in body, path
        # The claim that the CALL is deferred to a later turn of the loop is
        # gone from the handler that makes it; it never was.
        assert "# DEFERRED" not in text, path


def test_the_master_avwap_note_box_itself_saves_on_enter(app, tmp_path, monkeypatch):
    """The real panel handler, not the helper in isolation.

    This is the one that fails on the un-fixed panel: it built its own
    `QInputDialog` with `UsePlainTextEditForTextInput` and no key rule, so Enter
    put a newline in the box and nothing was ever saved.
    """
    from PySide6.QtWidgets import QPlainTextEdit

    from ui.annotations import verdicts
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    panel = MasterAvwapPanel(None)
    written = verdicts.record_like(
        symbol="NVDA",
        side="LONG",
        surface=verdicts.SURFACE_MASTER_AVWAP,
        path=tmp_path / "trader_annotations.jsonl",
    )
    saved: list[tuple] = []
    monkeypatch.setattr(panel, "_save_verdict_note", lambda row, note: saved.append((row, note)))

    panel._prompt_for_verdict_note(written)
    dialog = panel._verdict_note_dialog
    editor = dialog.findChild(QPlainTextEdit)
    editor.setPlainText("held the 20 EMA all day")

    _press(editor, shift=False)

    assert [note for _row, note in saved] == ["held the 20 EMA all day"]
