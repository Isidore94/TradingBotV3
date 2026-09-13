"""The Daily Recap page - WISHLIST 10F + 5F, packet WS-DR.

The desk's return surface, for EVERY Auto mode rather than only AWAY. It
replaces the AWAY Recap page in the nav and differs from it in one structural
way: **it reads the day from the durable stores, not from the process.** The
AWAY Recap was handed `center._alerts` + `center._d1_alerts` - a process-scoped
list capped at 250 / 100 items - so a desk restarted mid-session, or left
running across midnight, reported what the PROCESS saw. This page hands
`daily_recap_reader.read_session` a session, a lookback and a set of PATHS, on a
worker, and the same session read tomorrow is the same answer.

**Tabs, not one flat page.** The Strength window's flat page is the precedent
for a set of small boards; four tables that each want a screen of rows are not
that, and stacking them puts the fourth below the fold at the desk's windowed
1640x980. One tab per view, each with its own population sentence and its own
sort control, and the AWAY staged-pick block kept in front of the trader under
the tabs where the return-to-the-desk action has always been.

**It writes nothing and it charts through the named door.** A row click emits
`chartRequested`; the host routes that to the Alert Center's
`show_board_symbol`, which is the door for a board that lives on ANOTHER page. A
board chart holds no place in the waiting list, so nothing here is ever
re-queued or skip-counted. The chart widget has no marker seam, so the decision
time travels in the row's own Time column and its tooltip rather than being
invented into one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.widgets.data_table import apply_width_rule_to_table_widget

#: How many completed sessions the picker offers behind today. Long enough to
#: read back a week the trader was away for, short enough that the list is a
#: list rather than a scroll.
PICKER_SESSIONS = 15

#: The 1/2/3 control. It selects the lookback WINDOW and the horizon reported
#: together, because "how far back" and "how far forward" are one question for a
#: swing pick (the reader enforces it; this only offers it).
LOOKBACK_CHOICES = (1, 2, 3)

#: What a cell reads when nobody measured it. Never a 0.00.
UNMEASURED = "—"

#: view name -> (tab label, column headers, the measure behind each column).
#: A column bound to `None` is text: identity, the trader's own words, or the
#: environment label. Every NUMERIC column names a measure the reader declared,
#: so a column that is not a declared measure cannot be sorted on by accident.
VIEW_COLUMNS: dict[str, tuple[str, tuple[tuple[str, str | None], ...]]] = {
    "worked_today": (
        "What worked today",
        (
            ("Time", None),
            ("Symbol", None),
            ("Side", None),
            ("Best move %", "mfe_pct"),
            ("Held at close %", "eod_move_pct"),
            ("Environment", None),
            ("Status", None),
        ),
    ),
    "recent_swings": (
        "Swings that followed through",
        (
            ("Scanned", None),
            ("Symbol", None),
            ("Side", None),
            ("Selected end %", "selected_end_pct"),
            ("Next close %", "next_close_pct"),
            ("First favorable %", "first_favorable_pct"),
            ("Environment", None),
        ),
    ),
    "my_decisions": (
        "My decisions",
        (
            ("Time", None),
            ("Symbol", None),
            ("Side", None),
            ("Verdict", None),
            ("Why", None),
            ("Clicks", None),
            ("Day best %", "day_mfe_pct"),
            ("After the decision %", "mfe_pct_after_decision"),
            ("Journal R", "journal_r"),
            ("Journal match", None),
            ("Environment", None),
        ),
    ),
    "rejected_that_worked": (
        "Rejected, and it worked",
        (
            ("Time", None),
            ("Symbol", None),
            ("Side", None),
            ("Verdict", None),
            ("My reason", None),
            ("It ran %", "favorable_pct"),
            ("Against me first %", "adverse_pct"),
            ("After the decision %", "favorable_pct_after_decision"),
            ("Environment", None),
        ),
    ),
}

#: The order the tabs appear in - the packet's order, which is the order the
#: four questions were asked in.
VIEW_ORDER: tuple[str, ...] = (
    "worked_today",
    "recent_swings",
    "my_decisions",
    "rejected_that_worked",
)

#: What the page says before its first read finishes. A page that says nothing
#: while a worker runs reads as a broken page.
LOADING_NOTE = "Reading the session from the stores… (0 rows so far, 0 pending)"


class _RecapReadWorker(QThread):
    """One session read, off the GUI thread (ground rule 9).

    The same shape as the AWAY Recap's `_RecapWorker`: a QThread that does the
    whole read in `run` and hands the result back through one signal. The read
    opens nine files and walks them; none of that belongs on the Qt thread.
    """

    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, session_date: str, lookback_sessions: int, parent=None) -> None:
        super().__init__(parent)
        self._session = str(session_date)
        self._lookback = int(lookback_sessions)

    def run(self) -> None:  # pragma: no cover - exercised through its signal seam
        try:
            import daily_recap_reader

            session = daily_recap_reader.read_session(
                self._session, lookback_sessions=self._lookback
            )
        except Exception as exc:  # noqa: BLE001 - a failed read never costs the page
            self.failed.emit(str(exc))
            return
        self.loaded.emit(session)


class DailyRecapPanel(QFrame):
    """The day, read back from the files that recorded it."""

    statusChanged = Signal(str)
    #: (symbol, side). The host charts it through `show_board_symbol`.
    chartRequested = Signal(str, str)
    #: (symbol, side) - the host performs the Focus add through FocusService.
    focusAddRequested = Signal(str, str)

    def __init__(self, focus_service=None, parent=None) -> None:
        super().__init__(parent)
        self._focus_service = focus_service
        self._worker: _RecapReadWorker | None = None
        self._session: Any = None
        self._view_rows: dict[str, tuple] = {name: () for name in VIEW_ORDER}
        self._staged_rows: list[tuple[str, str]] = []

        self.heading = QLabel("Daily Recap")
        self.heading.setObjectName("SectionTitle")
        self.subtitle = QLabel(
            "The day read back from the stores that recorded it, not from what "
            "this desk happened to be running. Best available movement - never "
            "money earned."
        )
        self.subtitle.setObjectName("SectionSubtitle")
        self.subtitle.setWordWrap(True)
        self.subtitle.setMaximumHeight(theme.px(34))

        self.session_picker = QComboBox()
        self.lookback_picker = QComboBox()
        for choice in LOOKBACK_CHOICES:
            self.lookback_picker.addItem(
                f"{choice} prior session" + ("" if choice == 1 else "s"), choice
            )
        self.lookback_picker.setCurrentIndex(len(LOOKBACK_CHOICES) - 1)
        self._fill_session_picker()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.reload)
        self.session_picker.currentIndexChanged.connect(lambda _index: self.reload())
        self.lookback_picker.currentIndexChanged.connect(lambda _index: self.reload())

        self.provisional_note = QLabel("")
        self.provisional_note.setObjectName("SectionSubtitle")

        self.tabs = QTabWidget()
        self._tables: dict[str, QTableWidget] = {}
        self._notes: dict[str, QLabel] = {}
        self._sorts: dict[str, QComboBox] = {}
        for name in VIEW_ORDER:
            self.tabs.addTab(self._build_view(name), VIEW_COLUMNS[name][0])

        # Named attributes as well as the map, because a page is read by name.
        self.worked_today_table = self._tables["worked_today"]
        self.recent_swings_table = self._tables["recent_swings"]
        self.my_decisions_table = self._tables["my_decisions"]
        self.rejected_that_worked_table = self._tables["rejected_that_worked"]
        self.worked_today_note = self._notes["worked_today"]
        self.recent_swings_note = self._notes["recent_swings"]
        self.my_decisions_note = self._notes["my_decisions"]
        self.rejected_that_worked_note = self._notes["rejected_that_worked"]

        # The AWAY staged-pick block, moved onto this page unchanged in
        # behaviour: AWAY still STAGES and never adopts, this page still only
        # ASKS, and the gate is SHOWN at click time rather than enforced.
        self.staged_heading = QLabel("Staged picks - never adopted while AWAY")
        self.staged = QTableWidget(0, 3)
        self.staged.setHorizontalHeaderLabels(["Symbol", "Side", "Gate at click time"])
        self.staged.setEditTriggers(QTableWidget.NoEditTriggers)
        self.staged.setMinimumHeight(theme.px(84))
        self.staged.setMaximumHeight(theme.px(150))
        self.add_button = QPushButton("Add selected staged pick to Focus")
        self.add_button.clicked.connect(self._add_selected)
        self.gate_note = QLabel("")
        self.gate_note.setWordWrap(True)
        self.gate_note.setMaximumHeight(theme.px(40))

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setMaximumHeight(theme.px(34))

        header = QHBoxLayout()
        header.addWidget(QLabel("Session"))
        header.addWidget(self.session_picker, 1)
        header.addWidget(QLabel("Swing lookback"))
        header.addWidget(self.lookback_picker)
        header.addWidget(self.refresh_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.heading)
        layout.addWidget(self.subtitle)
        layout.addLayout(header)
        layout.addWidget(self.provisional_note)
        layout.addWidget(self.tabs, 5)
        layout.addWidget(self.staged_heading)
        layout.addWidget(self.staged, 1)
        layout.addWidget(self.add_button)
        layout.addWidget(self.gate_note)
        layout.addWidget(self.status)

    # -- construction ------------------------------------------------------
    def _build_view(self, name: str) -> QWidget:
        label, columns = VIEW_COLUMNS[name]
        page = QWidget()
        note = QLabel(LOADING_NOTE)
        note.setObjectName("SectionSubtitle")
        note.setWordWrap(True)
        note.setMaximumHeight(theme.px(44))
        sort_picker = QComboBox()
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by"))
        sort_row.addWidget(sort_picker, 1)
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels([header for header, _measure in columns])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setMinimumHeight(theme.px(120))
        table.itemDoubleClicked.connect(
            lambda item, view=name: self._activate(view, item)
        )
        table.itemActivated.connect(lambda item, view=name: self._activate(view, item))
        sort_picker.currentIndexChanged.connect(
            lambda _index, view=name: self._resort(view)
        )
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(note)
        layout.addLayout(sort_row)
        layout.addWidget(table, 1)
        self._tables[name] = table
        self._notes[name] = note
        self._sorts[name] = sort_picker
        return page

    def _fill_session_picker(self) -> None:
        """Completed sessions, newest first, with Today offered as PROVISIONAL.

        Today is in the list because the trader asks about it, and it is marked
        provisional in the entry itself rather than only in a note: a session
        that has not closed cannot be compared with one that has, and the label
        is the only thing standing between those two readings.
        """
        import market_calendar

        now = datetime.now()
        self.session_picker.blockSignals(True)
        try:
            self.session_picker.clear()
            try:
                cursor = market_calendar.last_completed_session(now)
            except Exception:  # noqa: BLE001 - a calendar refusal never empties the page
                self.session_picker.addItem(now.date().isoformat(), now.date().isoformat())
                return
            for index in range(PICKER_SESSIONS):
                stamp = cursor.isoformat()
                self.session_picker.addItem(stamp, stamp)
                if index + 1 >= PICKER_SESSIONS:
                    break
                try:
                    cursor = market_calendar.previous_session(cursor)
                except Exception:  # noqa: BLE001
                    break
            today = now.date().isoformat()
            if self.session_picker.findData(today) < 0:
                self.session_picker.addItem(
                    f"Today ({today}) - provisional, the session is not closed", today
                )
            self.session_picker.setCurrentIndex(0)
        finally:
            self.session_picker.blockSignals(False)

    # -- the controls ------------------------------------------------------
    def session_date(self) -> str:
        data = self.session_picker.currentData()
        if data:
            return str(data)
        return self.session_picker.currentText().strip()[:10]

    def lookback_sessions(self) -> int:
        data = self.lookback_picker.currentData()
        try:
            return int(data)
        except (TypeError, ValueError):
            return LOOKBACK_CHOICES[-1]

    # -- the read ----------------------------------------------------------
    def reload(self) -> None:
        """Ask the worker for the selected session. Never blocks the page."""
        if self._worker is not None and self._worker.isRunning():
            return
        self._worker = _RecapReadWorker(self.session_date(), self.lookback_sessions(), self)
        self._worker.loaded.connect(self.render_session)
        self._worker.failed.connect(self._render_failure)
        self._worker.start()

    def _render_failure(self, reason: str) -> None:
        self.status.setText(f"the session could not be read: {reason}")
        self.statusChanged.emit(self.status.text())

    def render_session(self, session: Any) -> None:
        """Draw one `RecapSession`. Formatting only - it computes nothing."""
        self._session = session
        self.provisional_note.setText(
            "This session has NOT closed - every number on it is provisional."
            if getattr(session, "provisional", False)
            else f"Session {getattr(session, 'session_date', '')}, closed and measured."
        )
        for name in VIEW_ORDER:
            view = getattr(session, name, None)
            if view is None:
                continue
            self._sync_sort_picker(name, view)
            self._render_view(name, view)
        self._render_staged(session)
        self.statusChanged.emit(
            f"Daily Recap: {getattr(session, 'session_date', '')} read from "
            f"{len(getattr(session, 'coverage', {}) or {})} stores"
        )

    def _sync_sort_picker(self, name: str, view: Any) -> None:
        picker = self._sorts[name]
        keys = tuple(getattr(view, "sort_keys", ()) or ())
        current = [picker.itemData(index) for index in range(picker.count())]
        if tuple(current) == keys:
            return
        picker.blockSignals(True)
        try:
            picker.clear()
            for key in keys:
                picker.addItem(self._measure_label(name, key), key)
            wanted = picker.findData(getattr(view, "sort_key", ""))
            picker.setCurrentIndex(max(0, wanted))
        finally:
            picker.blockSignals(False)

    @staticmethod
    def _measure_label(name: str, key: str) -> str:
        for header, measure in VIEW_COLUMNS[name][1]:
            if measure == key:
                return header
        return key

    def _resort(self, name: str) -> None:
        view = getattr(self._session, name, None) if self._session is not None else None
        if view is None:
            return
        self._render_view(name, view)

    def _render_view(self, name: str, view: Any) -> None:
        key = self._sorts[name].currentData() or getattr(view, "sort_key", "")
        try:
            rows = view.sorted_by(str(key))
        except Exception:  # noqa: BLE001 - an undeclared key never empties a table
            rows = tuple(getattr(view, "rows", ()) or ())
        self._view_rows[name] = tuple(rows)
        table = self._tables[name]
        columns = VIEW_COLUMNS[name][1]
        table.setRowCount(len(rows))
        tooltips: list[tuple[int, int, str]] = []
        for index, row in enumerate(rows):
            for column, (header, measure) in enumerate(columns):
                text, tip = self._cell(row, header, measure)
                table.setItem(index, column, QTableWidgetItem(text))
                if tip:
                    tooltips.append((index, column, tip))
        apply_width_rule_to_table_widget(table)
        # After the width rule, which attaches its own full-value tooltips: the
        # reason a cell is blank is the thing the trader has to be able to read,
        # and an empty cell's full value is empty.
        for index, column, tip in tooltips:
            item = table.item(index, column)
            if item is not None:
                item.setToolTip(tip)
        self._notes[name].setText(self._population_sentence(view))

    def _population_sentence(self, view: Any) -> str:
        """Cohort, window, n, pending - and the coverage behind them."""
        try:
            sentence = view.population_sentence()
        except Exception:  # noqa: BLE001
            sentence = ""
        missing = [
            entry.name
            for entry in (getattr(self._session, "coverage", {}) or {}).values()
            if getattr(entry, "unavailable_reason", "")
        ]
        if missing:
            sentence += " Unavailable sources: " + ", ".join(sorted(missing)) + "."
        return sentence

    def _cell(self, row: Any, header: str, measure: str | None) -> tuple[str, str]:
        """One cell's text and its tooltip. `None` is a dash with its reason."""
        if measure is not None:
            value = (getattr(row, "measures", {}) or {}).get(measure)
            if value is None:
                return UNMEASURED, str(
                    (getattr(row, "unavailable", {}) or {}).get(measure, "")
                )
            if measure == "journal_r":
                return f"{float(value):+.2f}R", ""
            return f"{float(value):+.2f}%", ""
        detail = getattr(row, "detail", {}) or {}
        moment = getattr(row, "observed_at", None)
        if header in {"Time", "Scanned"}:
            if moment is None:
                return UNMEASURED, "this observation carries no timestamp"
            if header == "Scanned":
                return moment.date().isoformat(), moment.isoformat()
            return moment.strftime("%H:%M"), moment.isoformat()
        if header == "Symbol":
            return str(getattr(row, "symbol", "")), ""
        if header == "Side":
            return str(getattr(row, "side", "")), ""
        if header == "Verdict":
            return str(detail.get("verdict") or ""), str(detail.get("channel") or "")
        if header in {"Why", "My reason"}:
            codes = ", ".join(str(code) for code in (detail.get("reason_codes") or ()))
            return str(detail.get("reason") or codes or ""), codes
        if header == "Clicks":
            return str(getattr(row, "occurrences", 1)), "repeated clicks on one opportunity"
        if header == "Journal match":
            return str(detail.get("match_state") or UNMEASURED), str(
                detail.get("trade_id") or ""
            )
        if header == "Environment":
            # Packet WS-10I: the label the decision COULD KNOW, and the one the
            # fill happened in beside it where there is a fill. Two labels, an
            # arrow between them, never one blended into the other - and never
            # an entry label on an opportunity nobody took.
            observation = str(
                getattr(row, "observation_context", "")
                or getattr(row, "d1_environment", "")
            )
            entry = str(getattr(row, "entry_context", "") or "")
            certainty = str(getattr(row, "observation_certainty", "") or "")
            text = f"{observation} → {entry}" if entry else observation
            tip = f"observed in `{observation}`" + (f" ({certainty})" if certainty else "")
            if entry:
                entry_certainty = str(getattr(row, "entry_certainty", "") or "")
                tip += f"; entered in `{entry}`"
                if entry_certainty:
                    tip += f" ({entry_certainty})"
                if getattr(row, "entry_flagged", False):
                    tip += " - read with care: the fill's own time is not known"
            tip += (
                f". The session itself was labelled `{getattr(row, 'd1_environment', '')}`, "
                "which is published at its close."
            )
            return text, tip
        if header == "Status":
            return str(detail.get("status") or ""), ""
        return "", ""

    # -- staged picks ------------------------------------------------------
    def _render_staged(self, session: Any) -> None:
        staged = dict(getattr(session, "staged_picks", {}) or {})
        rows: list[tuple[str, str]] = []
        for side in ("long", "short"):
            for symbol in staged.get(side) or ():
                rows.append((str(symbol), side.upper()))
        self._staged_rows = rows
        self.staged.setRowCount(len(rows))
        for index, (symbol, side) in enumerate(rows):
            self.staged.setItem(index, 0, QTableWidgetItem(symbol))
            self.staged.setItem(index, 1, QTableWidgetItem(side))
            self.staged.setItem(index, 2, QTableWidgetItem(""))
        apply_width_rule_to_table_widget(self.staged, text_columns=(2,))

    def _add_selected(self) -> None:
        """Adopt one staged pick. The GATE IS SHOWN, never enforced.

        Unchanged from the AWAY Recap: the R2 adoption gate governs the
        MACHINE's adoptions, and a surface that blocked the trader on it would
        substitute the machine's judgement for theirs.
        """
        index = self.staged.currentRow()
        if index < 0 or index >= len(self._staged_rows):
            self.status.setText("select a staged pick first")
            return
        symbol, side = self._staged_rows[index]
        self.gate_note.setText(
            f"R2 adoption gate: not measured on this page for {symbol} - the recap "
            "reads stores, not bars, so it has no completed M5 bar or previous-day "
            "extreme to measure against. It governs the machine's adoptions, never "
            "yours, so your action is unaffected."
        )
        if self._focus_service is None:
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

    # -- charting (delegated; this page owns no chart) ---------------------
    def _activate(self, name: str, item) -> None:
        """Ask the host for a BOARD chart of this row's name.

        `show_board_symbol` is the door for a board on another page. A board
        chart holds no place in the waiting list, so nothing here is re-queued
        or skip-counted - this page is not the scanner talking.
        """
        if item is None:
            return
        rows = self._view_rows.get(name) or ()
        index = item.row()
        if index < 0 or index >= len(rows):
            return
        row = rows[index]
        symbol = str(getattr(row, "symbol", "") or "").strip().upper()
        if not symbol:
            return
        self.chartRequested.emit(symbol, str(getattr(row, "side", "") or ""))

    def shutdown(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(2000)


__all__ = ["DailyRecapPanel", "LOOKBACK_CHOICES", "VIEW_COLUMNS", "VIEW_ORDER"]
