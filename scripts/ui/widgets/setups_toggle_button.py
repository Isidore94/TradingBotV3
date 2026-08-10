"""The desk control that shows or hides the Master AVWAP setups column.

The setups half of the desk is not useful until several hours into the
session, so the desk opens with it hidden and this button is how it comes
back. It is a button rather than only a keyboard shortcut for the same
reason the paint-lines control counts its hidden groups in its label: a
panel that is deliberately missing must never look like a panel that broke.

Deliberately NOT persisted, which is the one place this departs from
``PaintLinesPrefs``. "Hidden at launch" is the product decision, so the
state resets every start rather than carrying an afternoon's reveal into
the next morning's open - the exact hour the column is least wanted.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QToolButton


class SetupsToggleButton(QToolButton):
    """Checkable show/hide for the desk's Master AVWAP column."""

    setupsVisibleChanged = Signal(bool)

    def __init__(self, parent=None, *, visible: bool = False) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(bool(visible))
        # The desk is keyboard-driven around the alert queue; this button must
        # never sit in the tab order and swallow a keystroke meant for a chart.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._sync_text()
        self.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        self._sync_text()
        self.setupsVisibleChanged.emit(bool(checked))

    def set_setups_visible(self, visible: bool) -> None:
        """Reflect a change made elsewhere (F9) without re-emitting.

        Without the signal block this would echo back into the desk's own
        setter and the two would ping-pong on every F9 press.
        """
        visible = bool(visible)
        if visible == self.isChecked():
            return
        blocked = self.blockSignals(True)
        self.setChecked(visible)
        self.blockSignals(blocked)
        self._sync_text()

    def _sync_text(self) -> None:
        showing = self.isChecked()
        self.setText("Setups shown" if showing else "Setups hidden")
        self.setToolTip(
            "Hide the Master AVWAP setups column and give the width to the charts."
            if showing
            else "Show the Master AVWAP setups column."
        )
