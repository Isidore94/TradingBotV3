"""The AWAY day recap — R1 amendment 2026-08-24, decision record §5.

The return surface for a day the trader was not here. It replaces a 317-item
chart-review queue with one page: what the day produced, ranked the way the day
already ranked it, plus the Focus picks that needed a decision.

**It writes nothing.** Every mutation belongs to an existing owner:
`FocusService` adds, the Alert Center performs removals, and
`MarketJournalService` takes the after-the-fact D1 write-up. A trader-entered
add carries no auto-pick marker and is therefore trader-owned and never
auto-removed — that is structural, not a promise this page makes.

The R2 adoption-gate state is SURFACED at click time the way the Strength Board
already does, and **never blocks a trader action**: the gate governs the
machine's adoptions, not the trader's.

Every read is on a worker (ground rule 9).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from PySide6.QtCore import QEvent, QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ui import theme
from ui.widgets.data_table import apply_width_rule_to_table_widget

#: The text in the chart cell of a row that CAN be charted. A row that cannot is
#: blank there, so the affordance itself says which rows are openable.
CHART_CELL = "Chart ▸"

#: What the page says above the two chartable tables. §3.4 B: charting was wired
#: the whole time and nothing on the page said so, which read as "charting is
#: broken".
CHART_HINT = (
    "Select a row and press Enter, or click its Chart cell, to open it in the "
    "desk's one chart."
)


def is_scanner_status_row(row) -> bool:
    """True for a row that is about the SCANNER rather than about a symbol.

    `Scanning ...`, `Learning ...` and the entry-assist notes arrive on the Alert
    Center's backing list with no symbol and side `WATCH`. They belong to the
    day's record and are not deleted anywhere - but in a table of symbols they
    are indistinguishable from a symbol row whose chart is broken, which is
    exactly what the trader reported on 2026-08-26.

    The test is the blank symbol, not the side: a row with no symbol cannot be
    charted whatever its side says, and that is the property the page acts on.
    """
    if not isinstance(row, dict):
        return False
    return not str(row.get("symbol") or "").strip()


class _RecapWorker(QThread):
    """Assembles the recap off the GUI thread."""

    loaded = Signal(dict)

    def __init__(self, session_date: str, alerts: list, parent=None) -> None:
        super().__init__(parent)
        self._session = session_date
        self._alerts = alerts

    def run(self) -> None:  # pragma: no cover - exercised through its signal seam
        import away_recap

        unavailable: dict[str, str] = {}
        swings: list[str] = []
        staged: dict[str, Any] = {}
        focus: dict[str, Any] = {}
        try:
            from project_paths import AUTOPILOT_REPORT_FILE

            swings = away_recap.digest_swing_lines(
                open(AUTOPILOT_REPORT_FILE, encoding="utf-8").read()
            )
        except Exception as exc:  # noqa: BLE001
            unavailable["autopilot_today.txt"] = str(exc)
        try:
            from autopilot_core import load_auto_populate_pending_picks

            pending = load_auto_populate_pending_picks().get("pending") or {}
            staged = {
                side: sorted(str(sym).upper() for sym in (pending.get(side) or {}))
                for side in ("long", "short")
            }
        except Exception as exc:  # noqa: BLE001
            unavailable["staged picks"] = str(exc)
        try:
            import focus_picks

            # Keyword-only, and deliberately read ONCE: the no-argument form
            # unions the swing and m5 lists, which is what "what Focus held
            # today" means. Calling it per side both re-read the store and -
            # because it takes no positional argument - could only ever raise.
            by_side = focus_picks.load_focus_map()
            focus = {
                side: sorted(by_side.get(side) or [])
                for side in ("long", "short")
            }
        except Exception as exc:  # noqa: BLE001
            unavailable["Focus lists"] = str(exc)

        self.loaded.emit(
            away_recap.build_recap(
                session_date=self._session,
                alerts=self._alerts,
                staged_picks=staged,
                digest_swings=swings,
                focus_picks=focus,
                unavailable=unavailable,
            )
        )


class AwayRecapPanel(QFrame):
    """What the day produced, for a trader who was not watching it."""

    statusChanged = Signal(str)
    #: (symbol, side) - the host performs the Focus add through FocusService.
    focusAddRequested = Signal(str, str)
    symbolActivated = Signal(str)

    def __init__(self, focus_service=None, journal_service=None, parent=None) -> None:
        super().__init__(parent)
        self._focus_service = focus_service
        self._journal_service = journal_service
        self._worker: _RecapWorker | None = None
        self._alerts: list[dict[str, Any]] = []
        self._recap: dict[str, Any] = {}

        self.heading = QLabel("AWAY day recap")
        self.heading.setObjectName("SectionTitle")
        self.subtitle = QLabel(
            "An AWAY day ends here, not in a queue. This is what the day "
            "produced, ranked the way the day already ranked it - nothing on "
            "this page computes a new order."
        )
        self.subtitle.setObjectName("SectionSubtitle")
        self.subtitle.setWordWrap(True)

        self.session_picker = QComboBox()
        self.session_picker.setEditable(True)
        self.session_picker.addItem(date.today().isoformat())
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.reload)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.chart_hint = QLabel(CHART_HINT)
        self.chart_hint.setObjectName("SectionSubtitle")
        self.chart_hint.setWordWrap(True)
        self.swings = QTableWidget(0, 5)
        self.swings.setHorizontalHeaderLabels(["#", "Symbol", "Side", "Line", ""])
        self.swings.setEditTriggers(QTableWidget.NoEditTriggers)
        self.swings.itemActivated.connect(self._activate_swing)
        self.swings.itemDoubleClicked.connect(self._activate_swing)
        # The day's alerts. `build_recap` has always produced them and this page
        # never drew them, so a whole AWAY day's alerts left one trace: the word
        # "alert(s)" in the summary line. A recap that drops the thing it was
        # opened for is not a recap.
        self.alerts = QTableWidget(0, 7)
        self.alerts.setHorizontalHeaderLabels(
            ["Time", "Symbol", "Side", "Tier", "", "Trigger", ""]
        )
        self.alerts.setEditTriggers(QTableWidget.NoEditTriggers)
        self.alerts.itemActivated.connect(self._activate_alert)
        self.alerts.itemDoubleClicked.connect(self._activate_alert)
        self.alerts.cellClicked.connect(self._alert_cell_clicked)
        self.swings.cellClicked.connect(self._swing_cell_clicked)
        # Enter on the selected row. `itemActivated` covers it on most
        # platforms and a return key that does nothing on one of them is the
        # kind of "it is wired, nothing says so" defect this packet exists to
        # remove, so the key is handled explicitly as well.
        self.alerts.installEventFilter(self)
        self.swings.installEventFilter(self)

        # §8.3 decision 1: scanner status rows are hidden and COUNTED, never
        # deleted. One click reveals them for the session - the same
        # hide-and-count idiom as the movers-only review filter.
        self._show_status_rows = False
        self._status_rows: list[dict[str, Any]] = []
        self.status_rows_toggle = QPushButton("")
        self.status_rows_toggle.setObjectName("StatusRowsToggle")
        self.status_rows_toggle.setFlat(True)
        self.status_rows_toggle.clicked.connect(self._reveal_status_rows)
        self.status_rows_toggle.setVisible(False)
        self.staged = QTableWidget(0, 3)
        self.staged.setHorizontalHeaderLabels(["Symbol", "Side", "Gate at click time"])
        self.staged.setEditTriggers(QTableWidget.NoEditTriggers)
        self.focus_table = QTableWidget(0, 2)
        self.focus_table.setHorizontalHeaderLabels(["Symbol", "Side"])
        self.focus_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.add_button = QPushButton("Add selected staged pick to Focus")
        self.add_button.clicked.connect(self._add_selected)
        self.gate_note = QLabel("")
        self.gate_note.setWordWrap(True)

        self.journal_text = QPlainTextEdit()
        self.journal_text.setPlaceholderText(
            "The day's D1 analysis, written now. It is filed under the session "
            "it is about and stamped with when you actually wrote it."
        )
        self.journal_button = QPushButton("Save journal entry for this session")
        self.journal_button.clicked.connect(self._write_journal)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        header = QHBoxLayout()
        header.addWidget(QLabel("Session"))
        header.addWidget(self.session_picker, 1)
        header.addWidget(self.refresh_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.heading)
        layout.addWidget(self.subtitle)
        layout.addLayout(header)
        layout.addWidget(self.summary)
        layout.addWidget(self.chart_hint)
        layout.addWidget(QLabel("Best swing trades (as the day ranked them)"))
        layout.addWidget(self.swings, 2)
        layout.addWidget(
            QLabel("Alerts, in the order the day produced them - open one to chart it")
        )
        layout.addWidget(self.alerts, 3)
        layout.addWidget(self.status_rows_toggle)
        layout.addWidget(QLabel("Staged picks - never adopted while AWAY"))
        layout.addWidget(self.staged, 2)
        layout.addWidget(self.add_button)
        layout.addWidget(self.gate_note)
        layout.addWidget(QLabel("Focus picks that needed managing"))
        layout.addWidget(self.focus_table, 1)
        layout.addWidget(QLabel("Write the day up"))
        layout.addWidget(self.journal_text, 1)
        layout.addWidget(self.journal_button)
        layout.addWidget(self.status)

    # -- data -------------------------------------------------------------
    def set_alerts(self, alerts: list[dict[str, Any]]) -> None:
        """The day's alerts, handed in by the host that already holds them.

        Handed in rather than read: the Alert Center's backing list IS the
        record, and a second reader would be a second definition of what the
        day produced.
        """
        self._alerts = list(alerts or [])

    def session_date(self) -> str:
        return self.session_picker.currentText().strip() or date.today().isoformat()

    def reload(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._worker = _RecapWorker(self.session_date(), self._alerts, self)
        self._worker.loaded.connect(self._render)
        self._worker.start()

    def _render(self, recap: dict) -> None:
        self._recap = recap
        self.summary.setText(recap.get("summary", ""))
        self._fill(
            self.swings,
            [
                (str(row["rank"]), row["symbol"], row["side"], row["text"], "")
                for row in recap.get("best_swings") or []
            ],
            text_columns=(3,),
            chart_column=4,
            symbol_column=1,
        )
        self._render_alerts(recap.get("classified_alerts") or [])
        self._fill(
            self.staged,
            [(row["symbol"], row["side"], "") for row in recap.get("staged_picks") or []],
            text_columns=(2,),
        )
        self._fill(
            self.focus_table,
            [(row["symbol"], row["side"]) for row in recap.get("focus_to_manage") or []],
        )

    @staticmethod
    def _fill(
        table: QTableWidget,
        rows: list[tuple],
        *,
        text_columns=None,
        elide_columns=(),
        chart_column: int | None = None,
        symbol_column: int | None = None,
    ) -> None:
        """Write the rows, then apply the §12 width rule to the table.

        Every table on this page was a §12 violation: header labels and nothing
        else, so each column kept its default section width and the ranked-swing
        `Line` truncated to `1. FROG …` with two thirds of a 4K window empty.
        The rule is applied after the fill because it measures what is there.

        `chart_column` turns the last column into the per-row chart affordance
        (§8.3 decision 2). It is a plain item, not a cell widget: a widget per
        row is the shape the 2026-08-21 fluidity pass spent a day removing, and
        an AWAY day can produce hundreds of alert rows.
        """
        table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, value in enumerate(row):
                table.setItem(index, column, QTableWidgetItem(str(value)))
            if chart_column is None or symbol_column is None:
                continue
            symbol = str(row[symbol_column] or "").strip()
            cell = table.item(index, chart_column)
            if cell is None:
                cell = QTableWidgetItem("")
                table.setItem(index, chart_column, cell)
            cell.setText(CHART_CELL if symbol else "")
            if not symbol:
                AwayRecapPanel._mark_symbol_less(table, index)
        apply_width_rule_to_table_widget(
            table, text_columns=text_columns, elide_columns=elide_columns
        )

    @staticmethod
    def _mark_symbol_less(table: QTableWidget, row: int) -> None:
        """Render a symbol-less row distinctly, with no chart affordance.

        §8.3 decision 3: a blank `Symbol` cell must never read as a broken
        chart. The style is a theme TOKEN read through `ui.theme`, not a
        per-widget stylesheet - Qt style sheets do not reach view items at all,
        and the rule the fluidity pass established is about never making Qt
        parse CSS per widget, which this does not.
        """
        muted = QColor(theme.color("text_muted"))
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if item is None:
                continue
            item.setForeground(muted)
            font = item.font()
            font.setItalic(True)
            item.setFont(font)
            item.setData(Qt.ItemDataRole.UserRole + 1, "status")

    def _render_alerts(self, rows: list[dict]) -> None:
        """The day's alerts, with the scanner's own chatter hidden and counted.

        Nothing is deleted and nothing is muted: the Alert Center's backing list
        is untouched, `set_alerts` remains its one reader here, and one click
        puts the status rows back for the session.
        """
        symbol_rows = [row for row in rows if not is_scanner_status_row(row)]
        self._status_rows = [row for row in rows if is_scanner_status_row(row)]
        shown = symbol_rows + (self._status_rows if self._show_status_rows else [])
        self._fill(
            self.alerts,
            [
                (
                    str(row.get("time_text") or ""),
                    str(row.get("symbol") or ""),
                    str(row.get("side") or ""),
                    str(row.get("tier") or ""),
                    # Flagged, never merged away: the Alert Center keeps the D1
                    # feed separate because it is untiered, and a reader of this
                    # page has to be able to tell the two apart.
                    "D1" if row.get("is_d1") else "",
                    str(row.get("trigger") or ""),
                    "",
                )
                for row in shown
            ],
            text_columns=(5,),
            chart_column=6,
            symbol_column=1,
        )
        self._update_status_toggle()

    def _update_status_toggle(self) -> None:
        count = len(self._status_rows)
        if not count:
            self.status_rows_toggle.setVisible(False)
            self.status_rows_toggle.setText("")
            return
        self.status_rows_toggle.setVisible(True)
        noun = "message" if count == 1 else "messages"
        if self._show_status_rows:
            self.status_rows_toggle.setText(
                f"{count} scanner status {noun} shown - they are about the "
                "scanner, not a symbol, so they cannot be charted"
            )
            self.status_rows_toggle.setEnabled(False)
            return
        self.status_rows_toggle.setText(f"{count} scanner status {noun} - show")
        self.status_rows_toggle.setEnabled(True)

    def _reveal_status_rows(self) -> None:
        """Show them for the rest of the session. No re-read: the recap is held."""
        if self._show_status_rows:
            return
        self._show_status_rows = True
        self._render_alerts(list(self._recap.get("classified_alerts") or []))

    # -- charting (delegated; this page owns no chart) ---------------------
    def _alert_cell_clicked(self, row: int, column: int) -> None:
        if column == self.alerts.columnCount() - 1:
            self._ask_for_chart(self.alerts, self.alerts.item(row, 1))

    def _swing_cell_clicked(self, row: int, column: int) -> None:
        if column == self.swings.columnCount() - 1:
            self._ask_for_chart(self.swings, self.swings.item(row, 1))

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        """Enter on the selected row charts it.

        The page's charting was wired through `itemActivated` and nothing said
        so; the trader's verdict was "i also cant even check charts from here".
        The hint line says it now, and this makes the key work the same way on
        every platform rather than relying on Qt's per-platform activation.
        """
        if event is not None and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                for table in (self.alerts, self.swings):
                    if watched is table:
                        self._ask_for_chart(table, table.item(table.currentRow(), 1))
                        return True
        return super().eventFilter(watched, event)

    def _activate_alert(self, item) -> None:
        """Ask the host to chart the alert's symbol. Column 1 is the symbol."""
        self._ask_for_chart(self.alerts, item)

    def _activate_swing(self, item) -> None:
        self._ask_for_chart(self.swings, item)

    def _ask_for_chart(self, table: QTableWidget, item) -> None:
        """One chart surface for the whole desk (the R4 pattern).

        This page opens nothing itself. It emits the symbol and the host wires
        that to the Alert Center's existing snapshot popup - the same one the
        Strength Board, RS/RW and Industry boards use - so the chart carries the
        bot-backed series, the painted levels and the capture rail without a
        second chart widget existing anywhere.
        """
        if item is None:
            return
        cell = table.item(item.row(), 1)
        symbol = (cell.text() if cell is not None else "").strip().upper()
        # A blank cell is not a symbol. Asking for a chart of "" would open an
        # empty popup, which reads as a broken chart rather than an empty row.
        if not symbol:
            return
        self.symbolActivated.emit(symbol)

    # -- actions (delegated; this page owns no store) ----------------------
    def _selected_staged(self) -> tuple[str, str] | None:
        index = self.staged.currentRow()
        rows = self._recap.get("staged_picks") or []
        if index < 0 or index >= len(rows):
            self.status.setText("select a staged pick first")
            return None
        row = rows[index]
        return row["symbol"], row["side"]

    def _add_selected(self) -> None:
        """Adopt one staged pick. The GATE IS SHOWN, never enforced.

        The R2 adoption gate governs the MACHINE's adoptions. A trader clicking
        this button is making their own decision, and a surface that blocked it
        on a gate would be substituting the machine's judgement for theirs -
        which is the opposite of what the gate is for. The Strength Board set
        this pattern: re-check at click time, show the verdict, act anyway.
        """
        selected = self._selected_staged()
        if selected is None:
            return
        symbol, side = selected
        self.gate_note.setText(self._gate_text(symbol, side))
        if self._focus_service is None:
            # The host owns the Focus store; this page only asks.
            self.focusAddRequested.emit(symbol, side)
            self.status.setText(f"asked the desk to add {symbol} ({side}) to Focus")
            return
        try:
            self._focus_service.add(symbol, side, "swing")
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"could not add {symbol}: {exc}")
            return
        self.status.setText(
            f"added {symbol} ({side}) to swing Focus - trader-owned, so nothing "
            "will auto-remove it"
        )

    def _gate_text(self, symbol: str, side: str) -> str:
        """Say plainly that the gate was not measured here, and why.

        This used to call `mover_state(side, None, None, None)`. That signature
        is `(side, price, prev_high, prev_low)`, so with no price and no
        previous-day extremes it could only ever return UNKNOWN - and the page
        rendered that as a gate verdict for the symbol. A measurement that was
        never taken must not read like one that came back inconclusive.

        The recap has no bar source: it is an end-of-day page assembled from
        stores, and fetching the last completed M5 bar plus yesterday's extremes
        at click time would put a network/disk read on the Qt thread in a click
        handler - the exact defect this pass exists to remove. UNKNOWN stays
        UNKNOWN rather than being invented into a pass, and the trader is
        pointed at the surfaces that DO measure it, where the bars are already
        in hand.
        """
        return (
            f"R2 adoption gate: not measured on this page for {symbol} - the recap "
            "reads stores, not bars, so it has no completed M5 bar or previous-day "
            "extreme to measure against. The Strength Board and Focus surfaces "
            "re-check it at click time. It governs the machine's adoptions, never "
            "yours, so your action is unaffected."
        )

    def _write_journal(self) -> None:
        """Route the write-up to R10.H's service. Never backdated."""
        if self._journal_service is None:
            self.status.setText("the market journal service is not available here")
            return
        result = self._journal_service.write_entry(
            text=self.journal_text.toPlainText(),
            session_date=self.session_date(),
            timeframe="D1",
            origin="away_recap",
        )
        if result.get("ok"):
            self.journal_text.clear()
        else:
            self.status.setText(str(result.get("reason") or "entry not saved"))

    def shutdown(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(2000)
