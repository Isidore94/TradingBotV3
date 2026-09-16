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

**It fills itself in once a day (trader request, 2026-09-14).** One `QTimer`,
started by the host AFTER the window shows and never in the constructor, asks
`daily_recap_schedule.due_session` once a minute; at the configured Pacific
wall-clock time (default 12:00, `local_settings.json`
`daily_recap_auto_time`) on an exchange session it refills the session list,
selects TODAY and reads it once, then reads exactly once more after the
exchange-owned close. A desk started after the configured time reads today on
its first tick. Noon Pacific is an hour before the regular close, so that read
is labelled provisional by the reader itself; the close read is measured. The
read is a store read on a worker: no scan, no push, no write, so it runs in
every Auto mode and is not a starter under the quiet-hours rule
(`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`, amendment 2026-09-15).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Mapping

from PySide6.QtCore import QThread, QTimer, Signal
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

import daily_recap_schedule
from ui import theme
from ui.widgets.data_table import apply_width_rule_to_table_widget

#: How often the page asks whether its automatic read is due. A minute is the
#: Trade Mentor's cadence for the same question; the answer is a function of
#: the clock, so a late tick reads the same session a punctual one would.
AUTO_POLL_INTERVAL_MS = 60_000

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
            ("State", None),
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
            ("D1 result %", "d1_result_pct"),
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

#: Packet WS-RP: the FIFTH tab, added after the four views rather than swapped
#: in for one of them. It shows the session's published measured report - the
#: same cells, the same `report_id`, as the export carries.
REVIEW_TAB_TITLE = "Review"

#: What the local-AI review area says when nothing has been narrated for this
#: report id. This packet adds NO model call: it reads what the narration stage
#: left beside the report, and says so plainly when there is nothing.
NO_REVIEW_YET = "no review yet"

#: The Review tab's columns. `Cell` is the id the narration cites, the export
#: carries and this page prints - so it leads.
REVIEW_COLUMNS = (
    "Cell", "Value", "Unit", "State", "n", "Population", "Window", "Why not",
)


class _RecapReadWorker(QThread):
    """One session read, off the GUI thread (ground rule 9).

    The same shape as the AWAY Recap's `_RecapWorker`: a QThread that does the
    whole read in `run` and hands the result back through one signal. The read
    opens nine files and walks them; none of that belongs on the Qt thread.
    """

    loaded = Signal(object)
    failed = Signal(str)
    #: (report, narration) - packet WS-RP. The Review tab's payload is READ off
    #: the published files on this same worker; it is never rebuilt on the Qt
    #: thread and the page never computes a number of its own.
    reportLoaded = Signal(object, object)

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
        self._emit_report()

    def _emit_report(self) -> None:  # pragma: no cover - same signal seam
        """The published measured report for this session, or nothing at all.

        An absent report is not a failure: the first overnight run after this
        lands is what writes one, and the tab says so.
        """
        try:
            import measured_report
            from ai_jobs import digest, measured_report_publish

            root = digest._default_root()
            payload = measured_report_publish.latest_published(root, self._session)
            if not payload:
                self.reportLoaded.emit(None, None)
                return
            report = measured_report.report_from_payload(payload)
            narration = measured_report_publish.narration_for(root, report.report_id)
            # Proposal history is in the existing AI-store briefs namespace,
            # not the operational home.  This worker owns the read; the UI
            # merely renders its returned object.
            from ai_jobs import store
            import research_proposal

            proposal = research_proposal.published_display(store.briefs_dir(create=False) / "next_research_test", payload)
            self.reportLoaded.emit(report, {"narration": narration, "proposal": proposal})
        except Exception as exc:  # noqa: BLE001 - never costs the four views
            self.reportLoaded.emit(None, str(exc))


class DailyRecapPanel(QFrame):
    """The day, read back from the files that recorded it."""

    statusChanged = Signal(str)
    #: (symbol, side). The host charts it through `show_board_symbol`.
    chartRequested = Signal(str, str)
    #: (symbol, side) - the host performs the Focus add through FocusService.
    focusAddRequested = Signal(str, str)

    def __init__(
        self,
        focus_service=None,
        parent=None,
        *,
        clock: Callable[[], datetime] | None = None,
        auto_time_reader: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self._focus_service = focus_service
        self._worker: _RecapReadWorker | None = None
        self._session: Any = None
        # The automatic read (trader request 2026-09-14). `clock` and
        # `auto_time_reader` are injectable so a test can put the page at
        # 12:00 Pacific on a session without waiting for one.
        self._clock: Callable[[], datetime] = clock or datetime.now
        self._auto_time_reader: Callable[[], Any] = (
            auto_time_reader or daily_recap_schedule.auto_time_from_settings
        )
        self._auto_fired_session: str | None = None
        self._auto_post_close_session: str | None = None
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(AUTO_POLL_INTERVAL_MS)
        self._auto_timer.timeout.connect(self._on_auto_tick)
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
        self.summary_note = QLabel("")
        self.summary_note.setObjectName("SectionSubtitle")
        self.summary_note.setWordWrap(True)
        self.summary_note.setMaximumHeight(theme.px(54))

        self.tabs = QTabWidget()
        self._tables: dict[str, QTableWidget] = {}
        self._notes: dict[str, QLabel] = {}
        self._sorts: dict[str, QComboBox] = {}
        for name in VIEW_ORDER:
            self.tabs.addTab(self._build_view(name), VIEW_COLUMNS[name][0])
        # WS-RP: the fifth tab, after the four views.
        self._report: Any = None
        self._handoff: Any = None
        self._entry_quality_proposal: dict[str, Any] | None = None
        self.tabs.addTab(self._build_review_tab(), REVIEW_TAB_TITLE)

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
        layout.addWidget(self.summary_note)
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

    def _build_review_tab(self) -> QWidget:
        """The measured report, printed. It computes nothing and writes nothing.

        The numbers come off the PUBLISHED file through the recap's own worker,
        so this tab and the export cannot disagree: they are the same cells and
        the same `report_id`, which is printed here in full for exactly that
        reason.
        """
        page = QWidget()
        self.report_id_label = QLabel("no measured report has been published yet")
        self.report_id_label.setObjectName("SectionSubtitle")
        self.report_id_label.setWordWrap(True)
        self.report_id_label.setMaximumHeight(theme.px(44))

        self.review_table = QTableWidget(0, len(REVIEW_COLUMNS))
        self.review_table.setHorizontalHeaderLabels(list(REVIEW_COLUMNS))
        self.review_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.review_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.review_table.setMinimumHeight(theme.px(160))

        self.review_note = QLabel(NO_REVIEW_YET)
        self.review_note.setObjectName("SectionSubtitle")
        self.review_note.setWordWrap(True)

        self.next_test_card = QLabel("Next test: no validated proposal has been published yet")
        self.next_test_card.setObjectName("SectionSubtitle")
        self.next_test_card.setWordWrap(True)
        self.copy_next_test_button = QPushButton("Copy test brief")
        self.copy_next_test_button.clicked.connect(self.copy_next_test_brief)

        self.copy_handoff_button = QPushButton("Copy handoff")
        self.copy_handoff_button.clicked.connect(self.copy_handoff)
        self.export_handoff_button = QPushButton("Export handoff…")
        self.export_handoff_button.clicked.connect(self._export_handoff_clicked)
        self.handoff_note = QLabel(
            "The handoff is YOUR click: it writes three files (a capped "
            "readable brief, the full JSON payload and a manifest) or puts the "
            "brief on the clipboard. Nothing is uploaded and no model is called."
        )
        self.handoff_note.setObjectName("SectionSubtitle")
        self.handoff_note.setWordWrap(True)
        self.handoff_note.setMaximumHeight(theme.px(48))

        buttons = QHBoxLayout()
        buttons.addWidget(self.copy_handoff_button)
        buttons.addWidget(self.export_handoff_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.report_id_label)
        layout.addWidget(self.review_table, 1)
        layout.addWidget(QLabel("Local review of this report"))
        layout.addWidget(self.review_note)
        layout.addWidget(QLabel("Next test"))
        layout.addWidget(self.next_test_card)
        layout.addWidget(self.copy_next_test_button)
        layout.addLayout(buttons)
        layout.addWidget(self.handoff_note)
        return page

    def _fill_session_picker(self, select: str | None = None) -> None:
        """Completed sessions, newest first, with Today offered as PROVISIONAL.

        Today is in the list because the trader asks about it, and it is marked
        provisional in the entry itself rather than only in a note: a session
        that has not closed cannot be compared with one that has, and the label
        is the only thing standing between those two readings.

        Re-runnable: the list is a function of the clock, and a desk that was
        started before the close would otherwise still call today "provisional"
        in the picker at 21:00. `select` names the session to leave selected
        (the newest completed one when absent or unknown). Signals are blocked
        throughout, so a refill never triggers a read by itself.
        """
        import market_calendar

        now = self._clock()
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
            wanted = self.session_picker.findData(select) if select else -1
            self.session_picker.setCurrentIndex(max(0, wanted))
        finally:
            self.session_picker.blockSignals(False)

    def _refresh_session_picker(self) -> None:
        """Rebuild the list only when the newest completed session has moved.

        Called at the head of every read so a Refresh after the close relabels
        today from "provisional" to a plain completed entry. The selection is
        kept by its DATA (the ISO date), never by its index.
        """
        import market_calendar

        try:
            head = market_calendar.last_completed_session(self._clock()).isoformat()
        except Exception:  # noqa: BLE001 - a calendar refusal keeps the list as is
            return
        if self.session_picker.count() and self.session_picker.itemData(0) == head:
            return
        self._fill_session_picker(select=self.session_date())

    # -- the automatic read ------------------------------------------------
    def start(self) -> None:
        """Begin the once-a-minute due check. Called by the host after the
        window is up, never in the constructor: a timer started during
        construction runs while a test is still monkeypatching what it reads."""
        if not self._auto_timer.isActive():
            self._auto_timer.start()

    def auto_fired_session(self) -> str | None:
        """The session this process has already read automatically, if any."""
        return self._auto_fired_session

    def next_auto_read_at(self) -> datetime | None:
        """The next Pacific instant the automatic read is due; a label, never
        the decision (`daily_recap_schedule.due_session` is the decision)."""
        try:
            return daily_recap_schedule.next_fire_at(
                self._clock(), auto_time=self._configured_auto_time()
            )
        except Exception:  # noqa: BLE001
            return None

    def _configured_auto_time(self):
        try:
            raw = self._auto_time_reader()
        except Exception:  # noqa: BLE001 - settings unreadable: not due
            return None
        # A reader may hand back the parsed time or the raw setting text.
        if raw is None or hasattr(raw, "hour"):
            return raw
        return daily_recap_schedule.parse_auto_time(raw)

    def _on_auto_tick(self) -> None:
        try:
            self.poll_auto_read()
        except Exception:  # noqa: BLE001 - a timer slot never raises into Qt
            logging.debug("Daily Recap automatic read failed.", exc_info=True)

    def poll_auto_read(self) -> str | None:
        """One due check. Returns the session read, or `None` when nothing was
        due. Fires at most once per session per process; the memory is this
        process's, so a desk restarted after the hour reads today again, which
        is the behaviour the trader asked for (the page is ready when they
        come to it)."""
        due = daily_recap_schedule.due_session(
            self._clock(),
            auto_time=self._configured_auto_time(),
            last_fired_session=self._auto_fired_session,
        )
        if due is None:
            due = daily_recap_schedule.post_close_due_session(
                self._clock(),
                configured_session=self._auto_fired_session,
                last_post_close_session=self._auto_post_close_session,
            )
            if due is None:
                return None
            self._auto_post_close_session = due
        else:
            self._auto_fired_session = due
        self.show_session(due)
        return due

    def show_session(self, session_date: str) -> None:
        """Select `session_date` in the picker and read it. The picker is
        refilled first so today is in the list under its current label."""
        self._fill_session_picker(select=str(session_date))
        self.reload()

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
        self._refresh_session_picker()
        self._worker = _RecapReadWorker(self.session_date(), self.lookback_sessions(), self)
        self._worker.loaded.connect(self.render_session)
        self._worker.failed.connect(self._render_failure)
        self._worker.reportLoaded.connect(self._on_report_loaded)
        self._worker.start()

    def _on_report_loaded(self, report: Any, narration: Any) -> None:
        """The worker's measured report. An absent one is a sentence, not a hole."""
        if report is None:
            self._report = None
            self._handoff = None
            self.review_table.setRowCount(0)
            self.report_id_label.setText(
                "no measured report has been published for this session yet - "
                "the overnight run writes one"
                + (f" ({narration})" if narration else "")
            )
            self.review_note.setText(NO_REVIEW_YET)
            return
        proposal = None
        if isinstance(narration, Mapping):
            proposal = narration.get("proposal")
            narration = narration.get("narration")
        self.render_report(report, narration=narration)
        self.render_entry_quality_proposal(proposal)

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
        self.summary_note.setText(
            str(getattr(getattr(session, "summary", None), "text", ""))
        )
        self.statusChanged.emit(
            f"Daily Recap: {getattr(session, 'session_date', '')} read from "
            f"{len(getattr(session, 'coverage', {}) or {})} stores"
        )

    # -- the Review tab (WS-RP) -------------------------------------------
    def render_report(self, report: Any, narration: Any = None) -> None:
        """Draw one `MeasuredReport`. Formatting only - it computes nothing."""
        self._report = report
        self._handoff = None
        cells = tuple(report.cells())
        self.report_id_label.setText(
            f"report_id {report.report_id} · as of {report.as_of} · "
            f"{len(cells)} cells · the export carries this same id"
        )
        self.review_table.setRowCount(len(cells))
        for index, cell in enumerate(cells):
            value = cell.value
            values = (
                cell.cell_id,
                UNMEASURED if value is None else str(value),
                cell.unit,
                cell.state,
                str(cell.n),
                cell.population,
                f"{cell.window[0]}..{cell.window[1]}" if cell.window else "",
                cell.unavailable,
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(str(text))
                if cell.unavailable:
                    item.setToolTip(cell.unavailable)
                self.review_table.setItem(index, column, item)
        apply_width_rule_to_table_widget(self.review_table, text_columns=(5, 7))
        text = str(narration or "").strip()
        self.review_note.setText(text or NO_REVIEW_YET)

    def review_cells(self) -> tuple[tuple[str, Any], ...]:
        """`(cell_id, value)` for every row SHOWN, in the order shown.

        The parity seam: what this tab shows and what the export carries are
        the same cells read off the same report, and this is how a test proves
        the two can never drift.
        """
        out: list[tuple[str, Any]] = []
        for index in range(self.review_table.rowCount()):
            item = self.review_table.item(index, 0)
            if item is None or self._report is None:
                continue
            cell = self._report.cell(item.text())
            out.append((cell.cell_id, cell.value))
        return tuple(out)

    def render_entry_quality_proposal(self, payload: Mapping[str, Any] | None) -> None:
        """Show the already-published next-test object without recomputing it."""
        self._entry_quality_proposal = dict(payload) if isinstance(payload, Mapping) else None
        if not self._entry_quality_proposal:
            self.next_test_card.setText("Next test: no validated proposal has been published yet")
            return
        proposal = self._entry_quality_proposal.get("proposal") or {}
        control = proposal.get("control") or {}
        changed = proposal.get("changed_condition") or {}
        progress = self._entry_quality_proposal.get("current_progress") or {}
        source = self._entry_quality_proposal.get("proposal_source") or proposal.get("source") or {}
        self.next_test_card.setText(
            f"{proposal.get('question', 'No justified new test.')}\n"
            f"Control: {control.get('recipe_id', 'unknown')} · one change: "
            f"{changed.get('field', 'unknown')}={changed.get('value', 'unknown')}\n"
            f"Current progress: eligible {progress.get('eligible', 'unknown')} · no-trigger "
            f"{progress.get('no_trigger', 'unknown')} · missing {progress.get('missing_data', 'unknown')}\n"
            f"Limits: {self._entry_quality_proposal.get('unknown', 'not measured')} · "
            f"{proposal.get('status', 'proposed')} · report "
            f"{self._entry_quality_proposal.get('report_id', '')} · hash "
            f"{self._entry_quality_proposal.get('report_hash', '')}\n"
            f"Proposal source: {source.get('report_id', '')} · hash {source.get('report_hash', '')}"
        )

    def entry_quality_proposal_payload(self) -> dict[str, Any] | None:
        """The card's source object, useful for parity checks and no more."""
        return dict(self._entry_quality_proposal) if self._entry_quality_proposal else None

    def copy_next_test_brief(self) -> str:
        """Copy the validated brief.  It has no file, model, or runner effect."""
        if not self._entry_quality_proposal:
            self.status.setText("there is no validated next-test proposal yet")
            return ""
        import research_proposal
        from PySide6.QtWidgets import QApplication

        brief = research_proposal.copy_test_brief(self._entry_quality_proposal)
        app = QApplication.instance()
        if app is not None:
            app.clipboard().setText(brief)
        self.status.setText("copied the validated next-test brief; it does not register or run anything")
        self.statusChanged.emit(self.status.text())
        return brief

    def _build_handoff(self) -> Any:
        if self._report is None:
            return None
        if self._handoff is None:
            import measured_report

            self._handoff = measured_report.build_handoff(self._report)
        return self._handoff

    def copy_handoff(self) -> str:
        """The brief on the clipboard. Nothing on disk, nothing on the wire."""
        handoff = self._build_handoff()
        if handoff is None:
            self.status.setText("there is no measured report to hand off yet")
            return ""
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.clipboard().setText(handoff.markdown)
        self.status.setText(
            f"copied the brief for report {handoff.report_id} "
            f"({handoff.manifest['markdown_bytes']} bytes, "
            f"~{handoff.manifest['markdown_tokens_estimated']} estimated tokens)"
        )
        self.statusChanged.emit(self.status.text())
        return handoff.markdown

    def export_handoff(self, directory: Any) -> dict[str, str]:
        """Three files under `directory`: the brief, the payload, the manifest."""
        handoff = self._build_handoff()
        if handoff is None:
            self.status.setText("there is no measured report to hand off yet")
            return {}
        written = handoff.write(directory)
        self.status.setText(
            f"exported the handoff for report {handoff.report_id} to {directory}"
        )
        self.statusChanged.emit(self.status.text())
        return written

    def _export_handoff_clicked(self) -> None:
        """Ask where, then write. A dialog, because an export is a decision."""
        from PySide6.QtWidgets import QFileDialog

        directory = QFileDialog.getExistingDirectory(self, "Export the handoff to…")
        if not directory:
            return
        self.export_handoff(directory)

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
        if name == "recent_swings":
            rows = tuple(rows) + tuple(getattr(view, "pending", ()) or ())
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
        if header == "State":
            return ("Pending" if getattr(row, "pending", False) else "Measured"), ""
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
        try:
            self._auto_timer.stop()
        except RuntimeError:  # pragma: no cover - already torn down
            pass
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(2000)


__all__ = [
    "AUTO_POLL_INTERVAL_MS",
    "DailyRecapPanel",
    "LOOKBACK_CHOICES",
    "NO_REVIEW_YET",
    "REVIEW_COLUMNS",
    "REVIEW_TAB_TITLE",
    "VIEW_COLUMNS",
    "VIEW_ORDER",
]
