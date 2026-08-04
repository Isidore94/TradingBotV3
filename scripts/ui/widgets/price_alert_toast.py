"""Persistent, non-activating desktop presentation for price crossings."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

_DEFAULT_TOAST_CAP = 4


def price_alert_event_key(payload: dict[str, Any]) -> str:
    """Stable enough for live stream + sticky snapshot de-duplication."""
    return "|".join(
        str(payload.get(field) or "").strip()
        for field in ("date", "at", "symbol", "side", "level", "last")
    )


class PriceAlertToast(QDialog):
    def __init__(self, payload: dict[str, Any], parent=None, *, replayed: bool = False) -> None:
        super().__init__(parent)
        message = str(payload.get("message") or "Price alert fired")
        self.setWindowTitle("Missed price alert" if replayed else "PRICE ALERT")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setObjectName("PriceAlertToast")
        self.setStyleSheet(
            "QDialog#PriceAlertToast { background: #241113; border: 2px solid #f85149; "
            "border-radius: 8px; } QLabel { background: transparent; }"
        )

        if replayed:
            title_text = "MISSED PRICE ALERT"
        elif payload.get("push_ok") is False:
            title_text = "PRICE ALERT — PHONE PUSH FAILED"
        else:
            title_text = "PRICE ALERT — PHONE PUSH SENT"
        title = QLabel(title_text)
        title.setStyleSheet("color: #f85149; font-weight: 800; font-size: 15px;")
        body = QLabel(message)
        body.setWordWrap(True)
        body.setMinimumWidth(360)
        detail = QLabel(
            f"{payload.get('date', '')} {payload.get('at', '')} · priority "
            f"{payload.get('priority') or 'urgent'}"
        )
        detail.setObjectName("MutedLabel")
        dismiss = QPushButton("Dismiss")
        dismiss.clicked.connect(self.close)

        bottom = QHBoxLayout()
        bottom.addWidget(detail, 1)
        bottom.addWidget(dismiss)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addLayout(bottom)
        self.adjustSize()


class PriceAlertToastManager(QObject):
    def __init__(self, parent=None, *, cap: int = _DEFAULT_TOAST_CAP) -> None:
        super().__init__(parent)
        self.cap = max(1, int(cap))
        self.toasts: list[PriceAlertToast] = []
        self._seen: set[str] = set()

    def show_alert(self, payload: dict[str, Any], *, replayed: bool = False) -> bool:
        key = price_alert_event_key(payload)
        if key.strip("|") and key in self._seen:
            return False
        if key.strip("|"):
            self._seen.add(key)

        self.toasts = [toast for toast in self.toasts if toast.isVisible()]
        while len(self.toasts) >= self.cap:
            self.toasts.pop(0).close()

        toast = PriceAlertToast(payload, self.parent(), replayed=replayed)
        toast.destroyed.connect(lambda *_args, target=toast: self._forget(target))
        self.toasts.append(toast)
        QApplication.beep()
        toast.show()
        toast.raise_()
        self._restack()
        return True

    def _forget(self, toast: PriceAlertToast) -> None:
        self.toasts = [item for item in self.toasts if item is not toast]
        self._restack()

    def _restack(self) -> None:
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return
        area = screen.availableGeometry()
        margin = 18
        y = area.bottom() - margin
        for toast in reversed(self.toasts):
            if not toast.isVisible():
                continue
            toast.adjustSize()
            y -= toast.height()
            toast.move(area.right() - toast.width() - margin, y)
            y -= 8
