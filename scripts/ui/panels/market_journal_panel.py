"""The left-nav Market Journal page — R10.H.

The sit-down review: six D1 charts, the entries, the environment timeline with
its auto-vs-manual agreement rate, the calendar strip, and the machine's own
day-context row beside the trader's words.

Two labels that look like a collision and are not: the existing left-nav
**"Journal"** is the trade/tax journal — what you traded. This is **"Market
Journal"** — what you thought. Merging them would turn the tax record into a
diary, so the difference is deliberate and stays.

Everything expensive is off the Qt thread (ground rule 9): entries and charts
load on a worker, and the page renders what it is handed.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

#: How many D1 charts the page shows. Six is the trader's own number.
CHART_COUNT = 6


class _EntriesWorker(QThread):
    """Loads entries, the regime timeline and the day context off the GUI thread."""

    loaded = Signal(dict)

    def __init__(self, service, session_date: str, parent=None) -> None:
        super().__init__(parent)
        self._service = service
        self._session = session_date

    def run(self) -> None:  # pragma: no cover - exercised through its signal seam
        payload: dict[str, Any] = {"session_date": self._session}
        try:
            payload["entries"] = self._service.entries_for(self._session)
            payload["sessions"] = self._service.sessions_with_entries()
            payload["timeline"] = self._service.regime_timeline()
            payload["context"] = self._service.day_context(self._session)
        except Exception as exc:  # noqa: BLE001
            payload["error"] = str(exc)
        self.loaded.emit(payload)


class MarketJournalPanel(QFrame):
    """What the trader thought, beside what the machine measured."""

    statusChanged = Signal(str)
    symbolActivated = Signal(str)

    def __init__(self, service=None, parent=None) -> None:
        super().__init__(parent)
        if service is None:
            from ui.services.market_journal_service import MarketJournalService

            service = MarketJournalService(self)
        self.service = service
        self._worker: _EntriesWorker | None = None

        self.heading = QLabel("Market Journal")
        self.heading.setObjectName("SectionTitle")
        self.subtitle = QLabel(
            "What you thought, beside what the desk measured. The left-nav "
            "“Journal” page is the trade and tax record; this one is not."
        )
        self.subtitle.setObjectName("SectionSubtitle")
        self.subtitle.setWordWrap(True)

        self.session_picker = QComboBox()
        self.session_picker.setEditable(True)
        self.session_picker.currentTextChanged.connect(lambda _text: self.reload())
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.reload)

        self.entry_text = QPlainTextEdit()
        self.entry_text.setPlaceholderText(
            "What happened today, and what you make of it. Ctrl+Enter saves."
        )
        self.timeframe_picker = QComboBox()
        self.save_button = QPushButton("Save entry")
        self.save_button.clicked.connect(self._save)
        self.after_the_fact = QLabel("")
        self.after_the_fact.setObjectName("CautionLabel")
        self.after_the_fact.setWordWrap(True)

        self.entries = QListWidget()
        self.timeline = QListWidget()
        self.agreement = QLabel("")
        self.agreement.setWordWrap(True)
        self.context_table = QTableWidget(0, 2)
        self.context_table.setHorizontalHeaderLabels(["Measured", "Value"])
        self.context_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.calendar_strip = QLabel("")
        self.calendar_strip.setWordWrap(True)
        self.charts_note = QLabel("")
        self.charts_note.setWordWrap(True)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        import market_journal

        self.timeframe_picker.addItems(list(market_journal.TIMEFRAMES))
        self.timeframe_picker.setCurrentText(market_journal.TIMEFRAME_D1)

        header = QHBoxLayout()
        header.addWidget(QLabel("Session"))
        header.addWidget(self.session_picker, 1)
        header.addWidget(self.refresh_button)

        compose = QVBoxLayout()
        compose.addWidget(QLabel("New entry"))
        compose.addWidget(self.entry_text, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Timeframe"))
        row.addWidget(self.timeframe_picker)
        row.addStretch(1)
        row.addWidget(self.save_button)
        compose.addLayout(row)
        compose.addWidget(self.after_the_fact)
        compose_widget = QWidget()
        compose_widget.setLayout(compose)

        review = QVBoxLayout()
        review.addWidget(QLabel("Entries"))
        review.addWidget(self.entries, 2)
        review.addWidget(QLabel("Environment timeline"))
        review.addWidget(self.agreement)
        review.addWidget(self.timeline, 1)
        review.addWidget(QLabel("What the desk measured that session"))
        review.addWidget(self.context_table, 1)
        review.addWidget(self.calendar_strip)
        review_widget = QWidget()
        review_widget.setLayout(review)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(compose_widget)
        splitter.addWidget(review_widget)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.heading)
        layout.addWidget(self.subtitle)
        layout.addLayout(header)
        layout.addWidget(self.charts_note)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status)

        self.service.statusChanged.connect(self.status.setText)
        self.service.entryWritten.connect(lambda _row: self.reload())
        self._sync_after_the_fact()

    # -- session ----------------------------------------------------------
    def session_date(self) -> str:
        text = self.session_picker.currentText().strip()
        return text or date.today().isoformat()

    def _sync_after_the_fact(self) -> None:
        """Say plainly when the entry being typed is about a past session.

        Decision record §5a: an entry about Friday written on Saturday says so.
        A reader weighing "what did you think at the time?" needs to know it was
        not written at the time, and the trader should see that before they
        write, not after.
        """
        session = self.session_date()
        today = date.today().isoformat()
        if session and session < today:
            self.after_the_fact.setText(
                f"This entry is ABOUT {session} and will be stamped as written "
                f"today ({today}). It is filed under the session, never backdated."
            )
        else:
            self.after_the_fact.setText("")

    # -- loading ----------------------------------------------------------
    def reload(self) -> None:
        self._sync_after_the_fact()
        if self._worker is not None and self._worker.isRunning():
            return
        self._worker = _EntriesWorker(self.service, self.session_date(), self)
        self._worker.loaded.connect(self._render)
        self._worker.start()

    def _render(self, payload: dict) -> None:
        if payload.get("error"):
            self.status.setText(f"Market journal unavailable: {payload['error']}")
            return
        self._render_sessions(payload.get("sessions") or [])
        self._render_entries(payload.get("entries") or [])
        self._render_timeline(payload.get("timeline") or {})
        self._render_context(payload.get("context") or {})
        self._render_calendar()
        self._render_charts_note(payload.get("entries") or [])

    def _render_sessions(self, sessions: list[str]) -> None:
        current = self.session_picker.currentText()
        known = {self.session_picker.itemText(i) for i in range(self.session_picker.count())}
        for session in sessions:
            if session not in known:
                self.session_picker.addItem(session)
        if current:
            self.session_picker.setCurrentText(current)

    def _render_entries(self, entries: list[dict]) -> None:
        self.entries.clear()
        if not entries:
            self.entries.addItem("No entries for this session yet.")
            return
        for entry in entries:
            marker = " [written after the session]" if entry.get("written_after_the_session") else ""
            stamp = str(entry.get("created_at") or "")[:19]
            self.entries.addItem(
                f"{entry.get('timeframe', '')} {stamp}{marker}: {entry.get('text', '')}"
            )

    def _render_timeline(self, timeline: dict) -> None:
        self.timeline.clear()
        for shift in timeline.get("shifts") or []:
            self.timeline.addItem(
                f"{str(shift.get('event_at') or '')[:19]} "
                f"{shift.get('from_regime', '')} -> {shift.get('to_regime', '')} "
                f"({shift.get('source', '')})"
            )
        agreement = timeline.get("agreement") or {}
        if agreement.get("rate") is None:
            self.agreement.setText(
                f"Auto-vs-manual agreement: UNMEASURED - {agreement.get('note', '')}"
            )
        else:
            self.agreement.setText(
                f"Auto-vs-manual agreement: {agreement['rate'] * 100:.0f}% over "
                f"{agreement.get('sessions_compared', 0)} session(s). "
                f"{agreement.get('note', '')}"
            )

    def _render_context(self, context: dict) -> None:
        self.context_table.setRowCount(0)
        if not context.get("measured"):
            self.context_table.setRowCount(1)
            self.context_table.setItem(0, 0, QTableWidgetItem("(absent)"))
            self.context_table.setItem(
                0, 1, QTableWidgetItem(str(context.get("reason") or "unmeasured"))
            )
            return
        row = context.get("row") or {}
        fields = [
            (name, value)
            for name, value in sorted(row.items())
            if name not in {"schema", "event_type", "writer_host", "writer_pid", "run_id"}
        ]
        self.context_table.setRowCount(len(fields))
        for index, (name, value) in enumerate(fields):
            self.context_table.setItem(index, 0, QTableWidgetItem(str(name)))
            self.context_table.setItem(index, 1, QTableWidgetItem(str(value)))

    def _render_calendar(self) -> None:
        try:
            import market_context_ledger

            overlay = market_context_ledger.load_calendar_overlay()
            coverage = market_context_ledger.calendar_coverage(overlay)
        except Exception as exc:  # noqa: BLE001
            self.calendar_strip.setText(f"Calendar coverage unavailable: {exc}")
            return
        self.calendar_strip.setText(f"Calendar: {coverage.get('note', '')}")

    def _render_charts_note(self, entries: list[dict]) -> None:
        """The six D1 charts.

        Deliberately a note rather than six live chart widgets in this build:
        the charts go through the existing `ChartDataService` worker, and
        wiring them is a separate, chart-owning change. What is here now is the
        symbol set the charts will draw, so the page is honest about what it is
        showing rather than presenting an empty grid as a rendered one.
        """
        import market_journal

        symbols: list[str] = []
        for entry in entries:
            for symbol in entry.get("symbols") or []:
                if symbol not in symbols:
                    symbols.append(symbol)
        shown = symbols[:CHART_COUNT]
        if shown:
            self.charts_note.setText(
                f"D1 charts for {', '.join(shown)} "
                f"(RVOL ≥ {market_journal.RVOL_OVERLAY_FLOOR} overlay is "
                "journal-only and never touches the canonical D1 level store)."
            )
        else:
            self.charts_note.setText(
                "No symbols named in this session's entries yet, so there is "
                "nothing to chart."
            )

    def shutdown(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(2000)

    # -- writing ----------------------------------------------------------
    def _save(self) -> None:
        result = self.service.write_entry(
            text=self.entry_text.toPlainText(),
            session_date=self.session_date(),
            timeframe=self.timeframe_picker.currentText(),
            origin="journal_page",
        )
        if result.get("ok"):
            self.entry_text.clear()
