"""The Mentor's "Today's news & econ" block, and the desk's econ warning toast.

Both only draw what they are handed: the view comes from
`EconReminderService` (built on a worker), the warning from its timer.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

TITLE = "Today's news & econ"


def _clock(day: str, time_et: str) -> str:
    """"10:00 ET (7:00 PT)"; "" when the time is unknown."""
    if not time_et:
        return ""
    try:
        base = date.fromisoformat(day)
        moment = datetime(base.year, base.month, base.day, int(time_et[:2]), int(time_et[3:]), tzinfo=EASTERN)
    except (TypeError, ValueError):
        return f"{time_et} ET"
    local = moment.astimezone(PACIFIC)
    return f"{time_et} ET ({local.hour}:{local.minute:02d} PT)"


def _weekday(day: str) -> str:
    try:
        return date.fromisoformat(day).strftime("%a %m-%d")
    except ValueError:
        return day


def format_view(view: Mapping[str, Any]) -> str:
    """The block's text. Plain words, one idea per line."""
    lines: list[str] = []
    note = str(view.get("note") or "")
    if note:
        lines.append(note)
    for line in view.get("summary_lines") or ():
        if str(line).strip():
            lines.append(f"• {str(line).strip()}")
    today = list(view.get("today") or ())
    if today:
        lines.append("")
        lines.append("Today:")
        for row in today:
            when = _clock(str(row.get("date") or ""), str(row.get("time_et") or "")) or "time not given"
            lines.append(f"  {when} — {row.get('label') or ''}")
    elif not note:
        lines.append("")
        lines.append("Today: no econ events in the brief.")
    week = list(view.get("week") or ())
    if week:
        lines.append("")
        lines.append("Rest of the week:")
        for row in week:
            time_et = str(row.get("time_et") or "")
            when = f"{_weekday(str(row.get('date') or ''))} {time_et + ' ET' if time_et else '(time not given)'}"
            lines.append(f"  {when} — {row.get('label') or ''}")
    unread = int(view.get("unread_lines") or 0)
    if unread:
        lines.append("")
        lines.append(f"{unread} calendar line{'s' if unread != 1 else ''} not read.")
    return "\n".join(lines).strip()


class EconBriefBlock(QFrame):
    """A small framed block at the top of the Mentor popup."""

    hideRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("EconBriefBlock")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._session = ""
        self.title_label = QLabel(TITLE)
        self.title_label.setStyleSheet("font-weight: 700;")
        self.origin_label = QLabel("")
        self.origin_label.setObjectName("MutedLabel")
        self.body_label = QLabel("")
        self.body_label.setWordWrap(True)
        self.body_label.setTextFormat(Qt.TextFormat.PlainText)
        self.body_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hide_button = QPushButton("Hide")
        self.hide_button.setToolTip("Hide the econ block for now.")
        self.hide_button.clicked.connect(self.hideRequested.emit)
        top = QHBoxLayout()
        top.addWidget(self.title_label)
        top.addWidget(self.origin_label, 1)
        top.addWidget(self.hide_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self.body_label)

    def set_view(self, view: Mapping[str, Any]) -> None:
        self._session = str(view.get("session") or "")
        origin = str(view.get("origin_text") or "")
        self.origin_label.setText(f"— {origin}" if origin else "")
        self.body_label.setText(format_view(view))

    def session(self) -> str:
        return self._session

    def text(self) -> str:
        return self.body_label.text()


class EconToast(QDialog):
    """A non-activating corner toast for one econ warning."""

    def __init__(self, payload: Mapping[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(str(payload.get("title") or "Econ"))
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setObjectName("EconToast")
        self.setStyleSheet(
            "QDialog#EconToast { background: #231d0e; border: 2px solid #d29922; "
            "border-radius: 8px; } QLabel { background: transparent; }"
        )
        where = "phone sent too" if payload.get("phone") else "desk"
        title = QLabel(f"{str(payload.get('title') or 'ECON').upper()} — {where}")
        title.setStyleSheet("color: #d29922; font-weight: 800; font-size: 15px;")
        self.body = QLabel(str(payload.get("message") or ""))
        self.body.setWordWrap(True)
        self.body.setMinimumWidth(360)
        dismiss = QPushButton("Dismiss")
        dismiss.clicked.connect(self.close)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(dismiss)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(self.body)
        layout.addLayout(bottom)
        self.adjustSize()


class EconToastManager(QObject):
    """Stacks econ toasts above the bottom-right corner; beeps once per warning."""

    def __init__(self, parent=None, *, cap: int = 3) -> None:
        super().__init__(parent)
        self.cap = max(1, int(cap))
        self.toasts: list[EconToast] = []

    def show_reminder(self, payload: Mapping[str, Any]) -> EconToast:
        self.toasts = [toast for toast in self.toasts if toast.isVisible()]
        while len(self.toasts) >= self.cap:
            self.toasts.pop(0).close()
        parent = self.parent()
        toast = EconToast(payload, parent if hasattr(parent, "isWindow") else None)
        toast.destroyed.connect(lambda *_args, target=toast: self._forget(target))
        self.toasts.append(toast)
        QApplication.beep()
        toast.show()
        toast.raise_()
        self._restack()
        return toast

    def _forget(self, toast: EconToast) -> None:
        self.toasts = [item for item in self.toasts if item is not toast]
        self._restack()

    def _restack(self) -> None:
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return
        area = screen.availableGeometry()
        margin = 18
        # Above the price-alert stack's usual spot, so the two never cover each other.
        y = area.bottom() - margin - 220
        for toast in reversed(self.toasts):
            try:
                if not toast.isVisible():
                    continue
                toast.adjustSize()
                y -= toast.height()
                toast.move(area.right() - toast.width() - margin, y)
                y -= 8
            except RuntimeError:  # pragma: no cover - deleted under us
                continue
