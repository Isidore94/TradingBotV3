"""The "Best right now" strip in the M5 alerts column (P1-5 5b). Display only.

One ranked list (`best_now.rank_best_now`) of today's M5 alerts, Movers
dip-strong names and D1 names with an M5 confirmation, each with why, entry
and stop. It refreshes on 5-minute bar boundaries (and when the Movers board
lands, which is itself bar-aligned). The ranking runs on this widget's own
worker thread; the Qt thread only diffs the rows and sets the labels whose
text changed - the row widgets are built once and never rebuilt. The look is
`theme.qss` (`#BestNowStrip`, `#BestNowRow`), never a per-tick stylesheet.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout

import best_now
from swallowed import note_swallowed

BAR_SECONDS = 300
#: After the bar closes, give the bot's cache and the Movers tick a moment.
BAR_GRACE_SECONDS = 25

ResultsProvider = Callable[[], Any]


def ms_to_next_bar(now: datetime) -> int:
    """Milliseconds to the next 5-minute boundary plus the grace."""
    seconds = now.minute * 60 + now.second + now.microsecond / 1_000_000
    remaining = BAR_SECONDS - (seconds % BAR_SECONDS) + BAR_GRACE_SECONDS
    return max(1_000, int(remaining * 1000))


class _RowLabel(QLabel):
    """One row. Elides to its width; the full text is the tooltip. Click = chart it."""

    clicked = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("BestNowRow")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._full = ""
        self._symbol = ""

    def set_row(self, text: str, symbol: str) -> None:
        self._full = text
        self._symbol = symbol
        self.setToolTip(text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def _elide(self) -> None:
        width = max(10, self.width())
        metrics = QFontMetrics(self.font())
        shown = "\n".join(
            metrics.elidedText(line, Qt.TextElideMode.ElideRight, width)
            for line in self._full.split("\n")
        )
        if self.text() != shown:
            self.setText(shown)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        self._elide()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt's name
        if self._symbol and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._symbol)
        super().mousePressEvent(event)


class BestNowStrip(QFrame):
    """Title plus a fixed pool of row labels. Nothing else lives here."""

    symbolActivated = Signal(str)
    #: (generation, entries) from the worker, queued onto the Qt thread.
    _entriesReady = Signal(int, object)

    def __init__(
        self,
        parent=None,
        *,
        threaded: bool = True,
        clock: Callable[[], datetime] | None = None,
        limit: int = best_now.BEST_NOW_LIMIT,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("BestNowStrip")
        self._threaded = bool(threaded)
        self._clock = clock or datetime.now
        self._limit = max(1, int(limit))
        self._results_provider: ResultsProvider | None = None
        self._movers_board: dict[str, Any] = {}
        self._swing_context: dict = {}
        self._entries: list[best_now.BestNowEntry] = []
        self._texts: list[str] = []
        self._generation = 0
        self._busy = False
        self._pending = False
        #: How many row labels were written by the last apply (0 = nothing changed).
        self.last_changed_rows = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(0)
        self.title_label = QLabel("Best right now")
        self.title_label.setObjectName("BestNowTitle")
        layout.addWidget(self.title_label)
        self.empty_label = QLabel("Nothing yet today.")
        self.empty_label.setObjectName("BestNowEmpty")
        layout.addWidget(self.empty_label)
        # Neither label may set the M5 column's minimum width.
        for label in (self.title_label, self.empty_label):
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._rows: list[_RowLabel] = []
        for _index in range(self._limit):
            row = _RowLabel(self)
            row.clicked.connect(self.symbolActivated)
            row.hide()
            layout.addWidget(row)
            self._rows.append(row)
        self.setToolTip(
            "Best right now: D1 setups with an M5 alert first, then M5 alerts by grade and "
            "live R, then Movers dip-strong names. e = entry, s = stop (dip-strong: last "
            "price and the day's low). Display only - nothing is filtered or changed."
        )

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_bar)
        self._entriesReady.connect(self._on_entries)

    # ---------------------------------------------------------------- inputs
    def set_results_provider(self, provider: ResultsProvider | None) -> None:
        """Where today's graded M5 alert results come from (the Working-now strip)."""
        self._results_provider = provider

    def set_movers_board(self, board: Any) -> None:
        """The Movers board landed (bar-aligned): keep it and re-rank."""
        self._movers_board = dict(board or {})
        self.refresh()

    def set_swing_context(self, mapping: Any) -> None:
        """The setups table's `{(SYMBOL, SIDE): context}` map. Used at the next refresh."""
        self._swing_context = dict(mapping or {})

    def clear_day(self) -> None:
        self._movers_board = {}
        self._generation += 1
        self._apply([])

    # --------------------------------------------------------------- refresh
    def refresh(self) -> None:
        """Re-rank off the Qt thread; the inputs are copied here first."""
        try:
            results = list(self._results_provider() or ()) if self._results_provider else []
        except Exception:  # noqa: BLE001 - a display strip never costs the desk
            results = []
        dips = best_now.dip_strong_rows(self._movers_board)
        context = dict(self._swing_context)
        if not self._threaded:
            self._on_entries(self._generation, self._compute(results, dips, context, self._limit))
            return
        if self._busy:
            self._pending = True
            return
        self._busy = True
        threading.Thread(
            target=self._work,
            args=(self._generation, results, dips, context, self._limit),
            name="best-now-strip",
            daemon=True,
        ).start()

    @staticmethod
    def _compute(results, dips, context, limit):
        return best_now.rank_best_now(results, dips, context, limit=limit)

    def _work(self, generation, results, dips, context, limit) -> None:
        try:
            entries = self._compute(results, dips, context, limit)
        except Exception:  # noqa: BLE001 - never let the worker die silently busy
            entries = None
        try:
            self._entriesReady.emit(generation, entries)
        except RuntimeError as exc:  # widget already destroyed at shutdown
            note_swallowed("best-now result after the widget was destroyed", exc, quiet=True)

    def _on_entries(self, generation: int, entries: Any) -> None:
        self._busy = False
        if generation == self._generation and entries is not None:
            self._apply(list(entries))
        if self._pending:
            self._pending = False
            self.refresh()

    def _apply(self, entries: list) -> None:
        """Diff against what is shown; touch only the rows that changed."""
        self._entries = list(entries)
        texts = [best_now.entry_text(entry) for entry in self._entries]
        changed = [index for index in best_now.diff_rows(self._texts, texts) if index < len(self._rows)]
        for index in changed:
            row = self._rows[index]
            if index < len(texts):
                row.set_row(texts[index], self._entries[index].symbol)
                row.show()
            else:
                row.set_row("", "")
                row.hide()
        self._texts = texts
        self.last_changed_rows = len(changed)
        has_rows = bool(texts)
        if self.empty_label.isHidden() == (not has_rows):
            self.empty_label.setVisible(not has_rows)
        if self.property("hasRows") != has_rows:
            self.setProperty("hasRows", has_rows)
            style = self.style()
            style.unpolish(self)
            style.polish(self)

    # ---------------------------------------------------------------- views
    def entries(self) -> list:
        return list(self._entries)

    def row_texts(self) -> list[str]:
        return list(self._texts)

    def row_widgets(self) -> list[_RowLabel]:
        return list(self._rows)

    # ---------------------------------------------------------------- timer
    def _arm(self) -> None:
        self._timer.start(ms_to_next_bar(self._clock()))

    def _on_bar(self) -> None:
        self.refresh()
        if self.isVisible():
            self._arm()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().showEvent(event)
        if not self._timer.isActive():
            self._arm()
        self.refresh()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt's name
        self._timer.stop()
        super().hideEvent(event)

    def timer_active(self) -> bool:
        return self._timer.isActive()
