"""The M5 alert bar - a list, not a queue (trader, 2026-08-27).

"A lot of my charts to review are M5 charts. If I can instead just get a list I
can copy and paste into TC2000 that would be faster... a little sidebar in
between the master AVWAP setups and the chart... the ticker and the alert type
(new HOD, VWAP bounce etc) and I can choose what to look at. Then we can
totally purge M5 alerts from the waiting list and keep those for D1 alerts."

One line per alert, newest on top, oldest at the bottom - the trader's own
ordering rule, stated when asked. Clicking a line charts that alert in the
Alert Center exactly as a feed-row click does. "Copy all" puts the tickers on
the clipboard one per line (TC2000 paste), each ticker once, in bar order;
"Clear all" empties the bar ON SCREEN, and a clicked line leaves the bar the
moment it is charted - looked-at is done. Nothing here deletes, mutes, records
or withholds: the alert list, History, the feed and every evidence stream are
written before any alert reaches this bar, and none of them reads it.

Rows are plain QListWidget items with a foreground role for the side - no
per-widget stylesheet, no rebuild (fluidity rules, 2026-08-21).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.models.bounce import REGIME_PAUSE_TRIGGER_PREFIX

#: Oldest rows fall off past this; a session produced 72 M5 alerts in its
#: first 46 minutes on 2026-08-27, so this is a whole day with room.
MAX_ROWS = 400

_ALERT_ROLE = Qt.ItemDataRole.UserRole


def alert_type_label(alert: Any) -> str:
    """The short words for what fired: "new HOD", "VWAP reclaim", "lrsi_cross_20".

    The trigger is what the feed already shows; this only drops the tier tag
    in front of it and the regime-pause preamble, so the bar reads as a list
    of tickers and types rather than a second feed.
    """
    trigger = str(getattr(alert, "trigger", "") or "").strip()
    if trigger.startswith(REGIME_PAUSE_TRIGGER_PREFIX):
        tail = trigger[len(REGIME_PAUSE_TRIGGER_PREFIX) :].lstrip(" ··:-")
        trigger = tail or "regime pause"
    if trigger.startswith("[") and "]" in trigger:
        trigger = trigger.split("]", 1)[1].strip() or trigger
    if not trigger:
        trigger = str(getattr(alert, "timeframe", "") or "alert").strip() or "alert"
    return trigger[:48]


def row_text(alert: Any) -> str:
    time_text = str(getattr(alert, "time_text", "") or "")[:5]
    side = str(getattr(alert, "side", "") or "")
    mark = "▲" if side == "LONG" else "▼" if side == "SHORT" else "·"
    return f"{time_text}  {mark} {getattr(alert, 'symbol', '')}  {alert_type_label(alert)}"


class M5AlertBar(QWidget):
    """Newest alert on top. Click charts it; Copy/Clear act on the list only."""

    alertActivated = Signal(object)  # the BounceAlert behind the clicked row

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("M5AlertBar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.title_label = QLabel("M5 alerts")
        self.title_label.setObjectName("SectionTitle")
        header.addWidget(self.title_label, 1)
        self.copy_button = QPushButton("Copy all")
        self.copy_button.setToolTip(
            "Copy every ticker in this bar to the clipboard, one per line, "
            "each once, newest first - paste straight into a TC2000 watchlist."
        )
        self.copy_button.clicked.connect(self.copy_all)
        self.clear_button = QPushButton("Clear all")
        self.clear_button.setToolTip(
            "Empty this bar on screen. Nothing is deleted anywhere else - the "
            "feed, History and the evidence files keep every alert."
        )
        self.clear_button.clicked.connect(self.clear_all)
        header.addWidget(self.copy_button, 0)
        header.addWidget(self.clear_button, 0)
        layout.addLayout(header)

        self.list = QListWidget()
        self.list.setObjectName("M5AlertList")
        self.list.setUniformItemSizes(True)
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list, 1)
        self._refresh_title()

    # ---------------------------------------------------------------- data
    def post(self, alert: Any) -> None:
        """Add one alert at the TOP. Every alert is its own row."""
        symbol = str(getattr(alert, "symbol", "") or "").strip().upper()
        if not symbol:
            return
        item = QListWidgetItem(row_text(alert))
        item.setData(_ALERT_ROLE, alert)
        item.setToolTip(str(getattr(alert, "raw_text", "") or ""))
        side = str(getattr(alert, "side", "") or "")
        token = "long" if side == "LONG" else "short" if side == "SHORT" else "text_muted"
        try:
            item.setForeground(QColor(theme.color(token)))
        except Exception:
            pass
        self.list.insertItem(0, item)
        while self.list.count() > MAX_ROWS:
            self.list.takeItem(self.list.count() - 1)
        self._refresh_title()

    def alerts(self) -> list:
        """Top to bottom - newest first."""
        return [self.list.item(i).data(_ALERT_ROLE) for i in range(self.list.count())]

    def symbols(self) -> list[str]:
        """Each ticker once, in bar order (newest first)."""
        seen: set[str] = set()
        out: list[str] = []
        for alert in self.alerts():
            symbol = str(getattr(alert, "symbol", "") or "").strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                out.append(symbol)
        return out

    def count(self) -> int:
        return self.list.count()

    # ------------------------------------------------------------- actions
    def copy_all(self) -> str:
        """Tickers to the clipboard, one per line. Returns what was copied."""
        text = "\n".join(self.symbols())
        try:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)
        except Exception:
            pass
        return text

    def clear_all(self) -> None:
        self.list.clear()
        self._refresh_title()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Chart it, then take the line away (trader, 2026-08-27: "after I
        click on an alert it should go away"). Looked-at is done; the feed
        and History still have it."""
        alert = item.data(_ALERT_ROLE)
        row = self.list.row(item)
        if row >= 0:
            self.list.takeItem(row)
            self._refresh_title()
        if alert is not None:
            self.alertActivated.emit(alert)

    def _refresh_title(self) -> None:
        n = self.list.count()
        self.title_label.setText(f"M5 alerts ({n})" if n else "M5 alerts")
        self.copy_button.setEnabled(n > 0)
        self.clear_button.setEnabled(n > 0)
