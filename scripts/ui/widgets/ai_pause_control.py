"""Pause AI controls, shared by the desk and the Trade Mentor app.

``AiPauseButton``: "Pause AI" with a menu (2 h / 4 h / Until 06:00 / Until I resume,
Resume AI); every choice writes the one ``ai_paused_until`` setting. ``AiPauseRow``
adds a "paused until HH:MM" label. ``AiPausedPill``: a read-only status-bar pill,
shown only while paused. Reads are cheap (the settings file is re-stat'ed at most
once a second); the timers tick every 5 s and only while the widget is visible.
"""

from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QWidget

import ai_pause

REFRESH_MS = 5000
#: (menu text, ai_pause preset)
CHOICES = (
    ("For 2 hours", "2h"),
    ("For 4 hours", "4h"),
    ("Until 06:00", ai_pause.TONIGHT),
    ("Until I resume", ai_pause.FOREVER),
)
PAUSE_TIP = (
    "Stops every local-AI use of the GPU host (the Trade Mentor app, the desk's AI fill and the "
    "night AI's model jobs). Trade Mentor questions and the night's facts keep running."
)


def _paused_text(now: object = None) -> str:
    try:
        return ai_pause.reason(now)
    except Exception:  # noqa: BLE001 - a label never breaks the page
        logging.debug("Pause AI: the setting could not be read.", exc_info=True)
        return ""


class AiPauseButton(QPushButton):
    """"Pause AI" with a menu; ``changed`` fires after the setting is written."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None, *, now: Callable[[], object] | None = None) -> None:
        super().__init__("Pause AI", parent)
        self.setObjectName("AiPauseButton")
        self.setToolTip(PAUSE_TIP)
        self._now = now
        menu = QMenu(self)
        self.pause_actions = {}
        for text, choice in CHOICES:
            action = menu.addAction(text)
            action.triggered.connect(lambda _=False, c=choice: self.pause(c))
            self.pause_actions[choice] = action
        menu.addSeparator()
        self.resume_action = menu.addAction("Resume AI")
        self.resume_action.triggered.connect(self.resume)
        self.setMenu(menu)
        menu.aboutToShow.connect(self.refresh)
        self.refresh()

    def pause(self, choice: str) -> None:
        ai_pause.pause_for(choice, self._now() if self._now else None)
        self.refresh()
        self.changed.emit()

    def resume(self) -> None:
        ai_pause.resume()
        self.refresh()
        self.changed.emit()

    def refresh(self) -> None:
        # The owner's clock, the same one the pause was set on (the app's is injectable).
        paused = bool(_paused_text(self._now() if self._now else None))
        self.setText("AI paused" if paused else "Pause AI")
        self.resume_action.setEnabled(paused)


class AiPauseRow(QWidget):
    """The Settings row: the button and "AI paused until HH:MM" / "AI is on"."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.button = AiPauseButton(self)
        self.label = QLabel("")
        self.label.setObjectName("MutedLabel")
        layout.addWidget(self.button)
        layout.addWidget(self.label)
        layout.addStretch(1)
        self.button.changed.connect(self.refresh)
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        text = _paused_text()
        self.label.setText(text or "AI is on")
        self.button.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.refresh()
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._timer.stop()
        super().hideEvent(event)


class AiPausedPill(QLabel):
    """Status-bar pill: "AI paused" while paused (tooltip says until when), hidden otherwise."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("AI paused", parent)
        self.setObjectName("AiPausedPill")
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def refresh(self) -> None:
        text = _paused_text()
        self.setToolTip(text)
        self.setVisible(bool(text))
