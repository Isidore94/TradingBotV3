"""The optional note box, with ONE key rule: Enter saves, Shift+Enter newlines.

R4 A6. Both note prompts on the desk - the Master AVWAP star/cross and the
review pane's "Not today" - are `QInputDialog`s in
`UsePlainTextEditForTextInput` mode, which is what makes them multi-line. That
option also hands Return to the editor, so Enter inserted a newline and the only
way to save was to reach for the mouse. The packet that built them said Enter
saves; it did not.

A note the trader is typing one-handed between charts has to close on the key
their hand is already on. Shift+Enter keeps the multi-line case, which is the
reason the plain-text mode is there at all.

**What this module does NOT change.** The dialog is still opened with `open()`,
so it is window-modal and returns immediately, and the answer still arrives
through `textValueSelected`. Escape still leaves the click counted - the verdict
row is on disk before this box is ever shown, and cancelling records nothing but
loses nothing either.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QInputDialog, QPlainTextEdit, QWidget

__all__ = ["EnterSavesFilter", "open_note_prompt"]


class EnterSavesFilter(QObject):
    """Enter accepts the dialog; Shift+Enter falls through to the editor.

    Parented to the dialog on purpose: an event filter with no owner is garbage
    the moment the function that installed it returns, and a collected filter
    stops filtering silently.
    """

    def __init__(self, dialog: QInputDialog) -> None:
        super().__init__(dialog)
        self._dialog = dialog

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False  # a real newline, which is why we are plain text
                self._dialog.accept()
                return True
        return super().eventFilter(watched, event)


def open_note_prompt(
    parent: QWidget | None,
    *,
    title: str,
    label: str,
    on_text: Callable[[str], Any],
) -> QInputDialog:
    """Show the note box and return it. The caller must keep the reference.

    Window-modal and asynchronous (`open()`): a nested event loop here would sit
    between the click and the work the click still owes - the Focus placement,
    the review event, the queue advance - and in a headless test it would never
    return at all, so every test that clicks a star would HANG rather than fail.

    The dialog deletes itself on close; the caller holds it only so it is not
    collected before the trader answers.
    """
    dialog = QInputDialog(parent)
    dialog.setInputMode(QInputDialog.InputMode.TextInput)
    dialog.setOption(QInputDialog.InputDialogOption.UsePlainTextEditForTextInput, True)
    dialog.setWindowTitle(title)
    dialog.setLabelText(label)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dialog.textValueSelected.connect(on_text)

    editor = dialog.findChild(QPlainTextEdit)
    if editor is not None:
        editor.installEventFilter(EnterSavesFilter(dialog))
    dialog.open()
    return dialog
