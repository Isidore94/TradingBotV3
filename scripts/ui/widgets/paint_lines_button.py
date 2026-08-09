"""The paint-lines control: one button that shows and hides level groups.

Deliberately one control rather than six checkboxes in the chart's header.
The chart pane is height-starved (the snapshot legends already measured 43%
of it at 2560x1440 before they were put on one line), and a row of six
labelled boxes would cost that space permanently to answer a question the
trader asks a few times a session.

Every group defaults ON and the state is machine-local
(:mod:`ui.services.paint_lines_prefs`), so the chart looks the same tomorrow
as it did when the trader last set it, on this desk, without a preference
file crossing Drive to a machine with a different screen.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QMenu, QToolButton, QWidget

from chart_levels import LEVEL_GROUPS
from ui.services.paint_lines_prefs import PaintLinesPrefs


class PaintLinesButton(QToolButton):
    """Checkable menu of level groups; emits the hidden set on every change."""

    #: (hidden_groups) - the groups the chart must NOT paint, after the change.
    groupsChanged = Signal(list)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        prefs: PaintLinesPrefs | None = None,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self._prefs = prefs if prefs is not None else PaintLinesPrefs()
        self._compact = bool(compact)
        if self._compact:
            # The desk's embedded snapshot pane is height-starved - its legends
            # were already put on one line to win back pixels for the candles.
            # A flat, short button rides the legend row without taking any of
            # them back: the rowChrome property drops the theme's button
            # padding and border, and the height cap below is the row's own
            # line height rather than a guessed constant, so the header row
            # measures exactly what the legend label alone would measure.
            self.setAutoRaise(True)
            self.setProperty("rowChrome", True)
            self._sync_row_height()
        self.setText("Lines")
        self.setToolTip(
            "Show or hide groups of chart lines. Saved on this machine only."
        )
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._menu = QMenu(self)
        self._actions: dict[str, object] = {}
        for group, label in LEVEL_GROUPS:
            action = self._menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self._prefs.is_visible(group))
            action.toggled.connect(
                lambda checked, name=group: self._on_toggled(name, checked)
            )
            self._actions[group] = action
        self.setMenu(self._menu)
        self._update_text()

    def _sync_row_height(self) -> None:
        """Cap the compact button at one line of text, in the current font.

        The neighbouring legend is a plain QLabel with no margins, so its
        height is exactly one line of the themed font. Measuring the same
        metric here keeps the pair equal through a font-size change (the UI
        scale setting) instead of pinning a number that was only right at
        scale 1.0.
        """
        self.setMaximumHeight(QFontMetrics(self.font()).height())

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        # The stylesheet's font arrives at polish time, after __init__, and the
        # scale setting can change it again while the desk is running. Both
        # land here, so the cap tracks the row instead of freezing at whatever
        # font happened to be active when the button was built.
        if self._compact and event.type() in (
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
        ):
            self._sync_row_height()

    def hidden_groups(self) -> list[str]:
        return self._prefs.hidden_groups()

    def _on_toggled(self, group: str, checked: bool) -> None:
        self._prefs.set_visible(group, bool(checked))
        self._update_text()
        self.groupsChanged.emit(self._prefs.hidden_groups())

    def _update_text(self) -> None:
        hidden = len(self._prefs.hidden_groups())
        # The count is the whole point of the label: a trader who hid the S/R
        # last week and forgot needs the button to say a chart is incomplete,
        # not to look identical to a chart with everything on.
        self.setText("Lines" if not hidden else f"Lines ({hidden} off)")
