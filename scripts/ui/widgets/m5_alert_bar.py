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

Since 2026-09-01 a REPEAT of the same symbol+side folds into the row it already
has, with a ×N badge, and the row returns to the top carrying the newest alert.
That is the main feed's own rule applied to this bar, and it is PRESENTATION
ONLY - the sentence above still holds exactly as written. N events are drawn on
one line instead of N lines; every one of them has already reached the review
queue door, the outcome CSV and the review-event store, and the tooltip says so
on any folded row. A row also carries the take rate the Alert Center measured
for it when there is one: context, never a filter.

Rows are plain QListWidget items with a foreground role for the side - no
per-widget stylesheet, no rebuild (fluidity rules, 2026-08-21).

Since R4 (2026-09-02) a row also carries ONE capture verb, on the right-click
menu: a quick like. `SURFACE_M5_ALERT_BAR` had been declared by P10 and never
written from anywhere, so the trader's opinion of a name on this bar had no
column to land in. It is a CAPTURE and not a control: the like writes one
annotation row and does nothing else - no Focus placement, no arm, no alert, no
change to what this bar shows or to what reaches the review queue. The paragraph
above still holds for the alert stream itself; what is new is that the trader can
now say something about a row, and be recorded saying it.
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
    QMenu,
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
#: How many alerts this row has folded. Display only - see `post`.
_REPEAT_ROLE = Qt.ItemDataRole.UserRole + 1


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


def take_probability(alert: Any) -> float | None:
    """The take probability the Alert Center already computed, or None.

    READ ONLY. The host attaches it to the alert before posting; nothing here
    computes, looks up, or stats a file for it. An alert the desk has no
    guidance for carries nothing, and the row simply does not mention a take
    rate - which is the honest rendering of "not measured", where a 0% would
    be a claim.
    """
    value = getattr(alert, "review_take_prob", None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 1.0 else None


def row_text(alert: Any, *, repeats: int = 1) -> str:
    """One line: time, side, ticker, what fired, and what the trader usually does.

    The take-rate suffix is P(take | shown) for this alert's segments, measured
    from the trader's own review decisions. It is CONTEXT on a row, never a
    filter: every alert is on the bar whatever it says.

    ``repeats`` is the ×N fold badge - see `M5AlertBar.post`. It counts what
    the bar has SHOWN, and the count is display only.
    """
    time_text = str(getattr(alert, "time_text", "") or "")[:5]
    side = str(getattr(alert, "side", "") or "")
    mark = "▲" if side == "LONG" else "▼" if side == "SHORT" else "·"
    line = f"{time_text}  {mark} {getattr(alert, 'symbol', '')}  {alert_type_label(alert)}"
    if repeats > 1:
        line += f"  ×{repeats}"
    probability = take_probability(alert)
    if probability is not None:
        line += f"  take {probability * 100:.0f}%"
    return line


class M5AlertBar(QWidget):
    """Newest alert on top. Click charts it; Copy/Clear act on the list only."""

    alertActivated = Signal(object)  # the BounceAlert behind the clicked row
    #: R4 A5 - one annotation row was written for this alert. Capture only; no
    #: listener may treat it as a placement, and nothing here reads it back.
    likeRecorded = Signal(object)

    def __init__(self, parent=None, *, annotations_path: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("M5AlertBar")
        # Where a quick like is written (R4 A5). None means the one live
        # stream, which is what the desk passes; the seam exists so a test can
        # exercise the real handler without touching the trader's file.
        self._annotations_path = annotations_path
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
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._open_row_menu)
        layout.addWidget(self.list, 1)
        self._refresh_title()

    # ---------------------------------------------------------------- data
    def post(self, alert: Any) -> None:
        """Show one alert. A repeat of the same name FOLDS into its own row.

        The fold is the main feed's rule (`alert_repetition.RepetitionLedger`,
        trader 2026-08-16: "less spam and more quality"), applied to this bar's
        far smaller question: one row per symbol+side, a ×N badge, and the row
        moves back to the top carrying the newest alert. A tier upgrade
        escalates - the row is rewritten with the stronger alert - because the
        thing that changed is exactly what the trader wanted to see.

        PRESENTATION ONLY, and this is the line that matters: every event has
        already reached `_enqueue_review_alert`, the outcome CSV and the
        review-event store before it arrives here. This bar still deletes
        nothing, mutes nothing, records nothing and withholds nothing - it
        draws N events on one line instead of N lines, and clicking that line
        charts the NEWEST of them. `alerts()` returns one object per row, so
        "Copy all" keeps listing one symbol per row exactly as before.
        """
        symbol = str(getattr(alert, "symbol", "") or "").strip().upper()
        if not symbol:
            return
        side = str(getattr(alert, "side", "") or "")
        existing = self._row_for(symbol, side)
        if existing is not None:
            row, item = existing
            repeats = int(item.data(_REPEAT_ROLE) or 1) + 1
            self.list.takeItem(row)
            self._write_item(item, alert, repeats)
            self.list.insertItem(0, item)
            self._refresh_title()
            return
        item = QListWidgetItem()
        self._write_item(item, alert, 1)
        self.list.insertItem(0, item)
        while self.list.count() > MAX_ROWS:
            self.list.takeItem(self.list.count() - 1)
        self._refresh_title()

    def _row_for(self, symbol: str, side: str):
        """The existing row for this symbol+side, or None. Linear over <=400.

        Keyed on symbol AND side deliberately: a name that flips direction is a
        different claim, and folding the two would hide the flip - the one
        thing on this bar most worth seeing.
        """
        for row in range(self.list.count()):
            item = self.list.item(row)
            held = item.data(_ALERT_ROLE)
            if held is None:
                continue
            held_symbol = str(getattr(held, "symbol", "") or "").strip().upper()
            held_side = str(getattr(held, "side", "") or "")
            if held_symbol == symbol and held_side == side:
                return row, item
        return None

    def _write_item(self, item: QListWidgetItem, alert: Any, repeats: int) -> None:
        """Fill one row IN PLACE - never a rebuilt widget (fluidity rules)."""
        item.setText(row_text(alert, repeats=repeats))
        item.setData(_ALERT_ROLE, alert)
        item.setData(_REPEAT_ROLE, repeats)
        raw = str(getattr(alert, "raw_text", "") or "")
        if repeats > 1:
            raw = (
                f"{repeats} alerts on this name this session; the newest is shown.\n"
                "Every one of them is in the feed, History and the evidence "
                f"files - this row folds them, it does not drop them.\n\n{raw}"
            )
        item.setToolTip(raw)
        side = str(getattr(alert, "side", "") or "")
        token = "long" if side == "LONG" else "short" if side == "SHORT" else "text_muted"
        try:
            item.setForeground(QColor(theme.color(token)))
        except Exception:
            pass

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

    # ----------------------------------------------------------- capture
    def _open_row_menu(self, point) -> None:
        """Right-click a row: one quick like, and nothing else."""
        item = self.list.itemAt(point)
        if item is None:
            return
        menu = QMenu(self)
        like = menu.addAction("Quick like")
        like.setToolTip(
            "Records that something about this alert was good. It places "
            "nothing, arms nothing and changes nothing on this bar."
        )
        if menu.exec(self.list.viewport().mapToGlobal(point)) is like:
            self.quick_like(item)

    def quick_like(self, item: QListWidgetItem | None = None) -> dict | None:
        """One QUICK like for a row of this bar. Returns the written row.

        QUICK because this bar has no claim picklist and never asks for a why -
        P9's Alt+L path, from a different screen. A like carries zero privileges
        (plan.md P3.1): it records, and it does not act. The row stays on the bar
        so the trader can still click through to the chart.

        Every failure is swallowed and reported nowhere but the return value: an
        evidence store never costs the event it records, and there is nothing
        here for it to cost anyway.
        """
        if item is None:
            item = self.list.currentItem()
        alert = item.data(_ALERT_ROLE) if item is not None else None
        if alert is None:
            return None
        symbol = str(getattr(alert, "symbol", "") or "").strip().upper()
        if not symbol:
            return None
        side = str(getattr(alert, "side", "") or "").strip().upper()
        try:
            from ui.annotations import verdicts

            written = verdicts.record_like(
                symbol=symbol,
                side="SHORT" if side.startswith("SHORT") else "LONG",
                surface=verdicts.SURFACE_M5_ALERT_BAR,
                timeframe=str(getattr(alert, "timeframe", "") or "M5"),
                **({} if self._annotations_path is None else {"path": self._annotations_path}),
            )
        except Exception:
            return None
        if written is not None:
            self.likeRecorded.emit(alert)
        return written

    def _refresh_title(self) -> None:
        n = self.list.count()
        self.title_label.setText(f"M5 alerts ({n})" if n else "M5 alerts")
        self.copy_button.setEnabled(n > 0)
        self.clear_button.setEnabled(n > 0)
