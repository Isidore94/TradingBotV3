"""The Day Review page — one day, top to bottom, on one worker (TJ-1 item 3).

Trader, 2026-09-17: *"daily recap and market journal feel less than ideal … I
feel like currently it's overcomplicated … Market journal just sucks, it has
information but it really should be compacting the days … there's just too much
shit in these tabs and it's laggy as all hell. this should be a simple 'what
worked what didn't and what was your process'."* Decision 0021 answer 1: **one
page, Day Review, replaces Market Journal and Daily Recap.**

What that means here, concretely:

* **One session at a time**, chosen in a picker (the Daily Recap's rule: the
  last 15 completed sessions, with Today offered and marked PROVISIONAL), with
  the same two schedule functions deciding the automatic read.
* **One read, one payload.** `DayReviewService.read_day` is called on ONE
  `QThread` and every section paints from what it returns. No section owns a
  read, so no section can start one on the Qt thread (ground rule 9).
* **One chart, built on first need.** The old Market Journal page built FOUR
  `CandleChart`s on the first entry click - 299 ms measured in the G0 baseline.
  This page builds one, only when there are SPY bars to draw, and reuses it.
* **No machine rows, ever.** The desk's own `Auto mode X -> Y` rows were 34 of
  77 on the live desk. The service filters them (`market_journal.is_machine_entry`
  in `entries_about`) and this page filters them again, because it is the surface
  the trader complained about.

What is deliberately NOT here (plan.md §12.2, "gone from the trader's screen"):
the environment timeline, "What the desk measured that session", the calendar
overlay, the thesis drafting pane and "Save interpretation", the four capture
panes, the five Daily Recap tabs. The stores all stay; the readers moved or are
simply not read by a page the trader reads every day. The Daily Recap's Review
tab is now a section on Research > Results and its Staged picks table is on Auto
Pilot (TJ-1 item 6).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Mapping

from PySide6.QtCore import QEvent, QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import daily_recap_schedule
from ui import theme
from ui.panels import desk_layout
from ui.widgets.data_table import MEASURE_PRECISION_ROWS

#: How often the page asks whether its automatic read is due (the Daily Recap's
#: cadence, kept: the answer is a function of the clock, so a late tick reads the
#: same session a punctual one would).
AUTO_POLL_INTERVAL_MS = 60_000

#: How many completed sessions the picker offers behind today.
PICKER_SESSIONS = 15

#: The walk-away read's window. The page offers no control for it: "how far back"
#: was a Daily Recap tab control and this page is one day at a time.
LOOKBACK_SESSIONS = 3

#: How much of a thought the list shows (the G3.1 rule, kept verbatim: 90
#: characters is about a sentence, which is enough to FIND an entry; the rest is
#: the reader box's job).
EXCERPT_LIMIT = 90

#: What the "What happened" section says until TJ-4 writes a story. Said plainly
#: rather than leaving an empty box that reads as a read that failed.
NO_STORY_YET = "No story yet - it is written overnight (TJ-4)."

#: What the SPY section says when the desk has no bars for the session. TJ-2
#: brings the stored bars for a past session; until then this is the truth.
NO_CHART_NOTE = (
    "chart after the close - the desk holds today's SPY bars in memory, and "
    "TJ-2 brings past sessions."
)

#: The ideas card, until TJ-6.
NO_IDEAS_YET = "Nothing yet - the desk's AI starts speaking in TJ-6."

#: The ONE walk-away table TJ-1 ships. TJ-2 brings the other three.
WALKAWAY_TITLE = "Passed, and it ran"

#: Its columns - the ones the Daily Recap's `rejected_that_worked` tab already
#: showed, under the new title.
WALKAWAY_COLUMNS: tuple[tuple[str, str | None], ...] = (
    ("Time", None),
    ("Symbol", None),
    ("Side", None),
    ("Verdict", None),
    ("My reason", None),
    ("It ran %", "favorable_pct"),
    ("Against me first %", "adverse_pct"),
    ("After the decision %", "favorable_pct_after_decision"),
    ("Environment", None),
)

#: The three tables TJ-2 adds, as (title, note) - each one a small titled frame
#: holding its place in the 2 x 2 grid. A labelled cell rather than an empty
#: widget: a table with no rows and no explanation reads as a table that failed,
#: and a placeholder that is one title line and one note line is not a tall box.
WALKAWAY_PLACEHOLDERS: tuple[tuple[str, str], ...] = (
    ("Liked but never traded", "TJ-2 measures it."),
    ("Traded, then left early", "TJ-2 measures it."),
    ("Claimed D1 picks", "TJ-2 measures it."),
)

#: Where each placeholder sits, in grid order. The one real table is (0, 0):
#: most-ran first (plan.md §12.2), reading left to right then down.
WALKAWAY_PLACEHOLDER_CELLS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1))

#: The two columns (trader, 2026-09-18: *"there's a lot of empty space
#: horizontally that's not being efficiently used"*, and, offered three shapes,
#: he chose two columns). LEFT is what happened and the chart, RIGHT is what he
#: said: a chart uses every pixel of width it is given, and a column of running
#: text does not.
COLUMNS_OBJECT_NAME = "DayReviewColumns"
COLUMN_WEIGHTS = (55, 45)
#: Per-machine, like every other `qt_*` setting and every other desk splitter -
#: a 3800 px desk and a MacBook do not want the same split.
COLUMN_SPLIT_KEY = "qt_day_review_columns_v1"

#: The right column's own vertical split: the entries list over the reader. A
#: splitter rather than two fixed heights, so an empty day is not 600 px of
#: nothing and a long thought can be given the room to be read.
SAID_SPLIT_WEIGHTS = (60, 40)
SAID_SPLIT_KEY = "qt_day_review_said_split_v1"

#: The SPY pane's floor. A candle chart in a 120 px strip is a smear.
SPY_MIN_HEIGHT_PX = 320

#: The story's floor. It GROWS with its text above this (TJ-4 writes paragraphs);
#: below it, an empty story reads as a broken section.
STORY_MIN_HEIGHT_PX = 120

#: The trade line's columns. Read-only: the Journal page is still where a trade
#: is tagged and corrected (decision 0021 consequences).
TRADE_COLUMNS: tuple[str, ...] = (
    "Time", "Symbol", "Direction", "Qty", "Net P&L", "Status",
)

#: What a cell reads when nobody measured it. Never a 0.00.
UNMEASURED = "—"

#: How many lines of a pasted forecast the block shows before "Show all". Three
#: in the column layout: someone else's commentary is the smallest thing on the
#: page and it sits under the trader's own words, never above them.
FORECAST_COLLAPSED_LINES = 3

#: What the page says while its first read is in flight.
LOADING_NOTE = "Reading the session…"

#: What it says while the post-close index build runs on its worker. The build
#: streams the big stores once, so it is worth a sentence rather than a silence.
BUILDING_INDEX_NOTE = "Building {session}'s index in the background…"


def _excerpt(text: str, limit: int = EXCERPT_LIMIT) -> str:
    """The first line of a thought, cut at `limit`, with `…` when there is more.

    The ellipsis is a CLAIM - "there is more text than this" - so it is never
    printed for a short single-line entry that is shown whole (G3.1).
    """
    body = str(text or "").strip()
    lines = body.splitlines()
    first = lines[0].strip() if lines else ""
    truncated = len(lines) > 1 or len(first) > limit
    if len(first) > limit:
        first = first[:limit].rstrip()
    return f"{first}…" if truncated else first


def _clock_text(created_at: Any) -> str:
    """`HH:MM` off the row's own stamp, or a dash. Never a guessed zone."""
    raw = str(created_at or "").strip()
    if len(raw) >= 16 and raw[10] in {"T", " "}:
        return raw[11:16]
    return UNMEASURED


def _fill_the_width(table: QTableWidget) -> None:
    """Make one table fill its cell, and stop clipping its own headers.

    Two rules, set ONCE at construction rather than on every render:

    * every column but the last measures its CONTENTS, which includes the
      header's own text - the shared width rule's 260 px ceiling cut "Against
      me first %" (which hints 273 px under the desk theme) at both ends, and a
      centred header clipped that way carries no ellipsis to say so;
    * the last section stretches, so a table on a 3800 px screen fills its cell
      instead of ending in the middle of it.

    `ResizeToContents` re-measures itself when the rows change, so this is not
    re-applied per paint. The measurement is bounded by the same row cap the
    shared rule uses (`MEASURE_PRECISION_ROWS`); these tables hold one day.
    """
    header = table.horizontalHeader()
    for side in (header, table.verticalHeader()):
        if side.resizeContentsPrecision() != MEASURE_PRECISION_ROWS:
            side.setResizeContentsPrecision(MEASURE_PRECISION_ROWS)
    for column in range(max(0, table.columnCount() - 1)):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    header.setStretchLastSection(True)


def _is_machine_row(row: Mapping[str, Any]) -> bool:
    try:
        import market_journal

        return market_journal.is_machine_entry(row)
    except Exception:  # noqa: BLE001 - the constant is in the same repo
        return str(row.get("origin") or "") == "auto_mode_flip"


class _DayReadWorker(QThread):
    """One `read_day`, off the GUI thread. The page's only reader.

    It hands back ONE payload and never touches a widget: the page renders, and
    a page that is refreshing goes on showing what it already had.

    The SPY bars are handed IN rather than read here. `journal_chart_bars` is the
    Alert Center's own cache accessor: it mutates `_m5_bar_dicts` and arms a
    `QTimer.singleShot`, and a `singleShot` armed from a thread with no event
    loop never fires - which latched `_d1_prefetch_flush_armed` True and killed
    D1 prefetch for the rest of the session (reviewer, 2026-09-17). It is a Qt
    THREAD accessor, so the slot that starts this worker reads it.
    """

    loaded = Signal(dict)
    failed = Signal(str)

    def __init__(self, service, session_date: str, parent=None, *, spy_m5_bars=None) -> None:
        super().__init__(parent)
        self._service = service
        self._session = str(session_date)
        self._spy_m5_bars = list(spy_m5_bars or ())

    def run(self) -> None:  # pragma: no cover - exercised through its signals
        try:
            payload = self._service.read_day(
                self._session,
                lookback_sessions=LOOKBACK_SESSIONS,
                spy_m5_bars=self._spy_m5_bars,
            )
        except Exception as exc:  # noqa: BLE001 - a failed read never costs the page
            self.failed.emit(str(exc))
            return
        self.loaded.emit(dict(payload or {}))


class _IndexBuildWorker(QThread):
    """One post-close index build, off the GUI thread.

    Blocker 1 (reviewer, 2026-09-17): the build was called straight from the
    60-second timer slot and froze the desk for **22.8 seconds** - it streams the
    476 MB intraday log and three other stores. A timer slot must return in
    milliseconds, so the slot starts this and the page says what is happening.
    """

    built = Signal(str)
    failed = Signal(str, str)

    def __init__(self, service, session_date: str, parent=None) -> None:
        super().__init__(parent)
        self._service = service
        self._session = str(session_date)

    def run(self) -> None:  # pragma: no cover - exercised through its signals
        try:
            self._service.build_index_for(
                self._session, lookback_sessions=LOOKBACK_SESSIONS
            )
        except Exception as exc:  # noqa: BLE001 - a cache never costs the page
            self.failed.emit(self._session, str(exc))
            return
        self.built.emit(self._session)


class DayReviewPanel(QFrame):
    """The day, read back: what happened, what you said, what you traded."""

    statusChanged = Signal(str)
    #: (symbol, side). The host charts it through `show_board_symbol`.
    chartRequested = Signal(str, str)

    def __init__(
        self,
        service=None,
        parent=None,
        *,
        clock: Callable[[], datetime] | None = None,
        auto_time_reader: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        if service is None:
            from ui.services.day_review_service import DayReviewService

            service = DayReviewService()
        self.service = service
        self._worker: _DayReadWorker | None = None
        self._index_worker: _IndexBuildWorker | None = None
        self._building_index = ""
        #: The desk's own M5 cache accessor (`alert_center.journal_chart_bars`).
        #: Called ONLY on the Qt thread, by `reload`, and only for a session that
        #: has not closed - see `_DayReadWorker` for what a worker call cost.
        self._bars_reader: Callable[[str], Any] | None = None
        self._payload: dict[str, Any] = {}
        self._entries: list[dict[str, Any]] = []
        self._walkaway_rows: tuple[Any, ...] = ()
        self._forecast_expanded = False
        self._loaded_once = False
        self._chart: Any = None
        self._clock: Callable[[], datetime] = clock or datetime.now
        self._auto_time_reader: Callable[[], Any] = (
            auto_time_reader or daily_recap_schedule.auto_time_from_settings
        )
        self._auto_fired_session: str | None = None
        self._auto_post_close_session: str | None = None
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(AUTO_POLL_INTERVAL_MS)
        self._auto_timer.timeout.connect(self._on_auto_tick)

        self._build_header()
        self._build_story()
        self._build_walkaway()
        self._build_said()
        self._build_traded()
        self._build_chart_section()
        self._build_ideas()
        self._build_layout()
        self.refresh_reader_measure()

    # -- construction ------------------------------------------------------
    def _build_header(self) -> None:
        self.heading = QLabel("Day Review")
        self.heading.setObjectName("SectionTitle")
        self.subtitle = QLabel(
            "One day: what happened, what you passed on, what you said and what "
            "you traded. Best available movement - never money earned."
        )
        self.subtitle.setObjectName("SectionSubtitle")
        self.subtitle.setWordWrap(True)

        self.session_picker = QComboBox()
        self._fill_session_picker()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.reload)
        self.session_picker.currentIndexChanged.connect(lambda _index: self.reload())

        self.provisional_note = QLabel("")
        self.provisional_note.setObjectName("SectionSubtitle")
        self.provisional_note.setWordWrap(True)
        self.status = QLabel("")
        self.status.setObjectName("SectionSubtitle")
        self.status.setWordWrap(True)

    def _build_story(self) -> None:
        self.story_note = QLabel(NO_STORY_YET)
        self.story_note.setObjectName("SectionSubtitle")
        self.story_note.setWordWrap(True)
        self.story_facts = QLabel("")
        self.story_facts.setWordWrap(True)
        self.story_facts.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # A floor, never a ceiling: TJ-4's story is paragraphs and a wrapped
        # QLabel grows with them. The floor is what stops an empty story from
        # reading as a section that failed.
        self.story_facts.setMinimumHeight(theme.px(STORY_MIN_HEIGHT_PX))
        self.story_facts.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # Read-only: the drafting pane and "Save interpretation" are gone from
        # the trader's screen (plan.md §12.2). The sidecar still holds them.
        self.theses = QListWidget()
        self.theses.setMinimumHeight(theme.px(70))
        self.theses.setMaximumHeight(theme.px(150))

    def _build_walkaway(self) -> None:
        self.rejected_that_worked_table = QTableWidget(0, len(WALKAWAY_COLUMNS))
        self.rejected_that_worked_table.setHorizontalHeaderLabels(
            [header for header, _measure in WALKAWAY_COLUMNS]
        )
        self.rejected_that_worked_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.rejected_that_worked_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.rejected_that_worked_table.setMinimumHeight(theme.px(120))
        self.rejected_that_worked_table.itemDoubleClicked.connect(self._activate_walkaway)
        self.rejected_that_worked_table.itemActivated.connect(self._activate_walkaway)
        _fill_the_width(self.rejected_that_worked_table)
        self.walkaway_note = QLabel(LOADING_NOTE)
        self.walkaway_note.setObjectName("SectionSubtitle")
        self.walkaway_note.setWordWrap(True)
        #: The three TJ-2 populations, by grid position. Built here so the grid
        #: is the same four cells whether or not anything has been read yet.
        self.walkaway_cells: dict[tuple[int, int], QWidget] = {
            cell: self._placeholder_cell(title, note)
            for cell, (title, note) in zip(
                WALKAWAY_PLACEHOLDER_CELLS, WALKAWAY_PLACEHOLDERS
            )
        }

    @staticmethod
    def _placeholder_cell(title: str, note: str) -> QFrame:
        """One empty population: a titled frame, one note line, nothing taller.

        `Panel` is the theme's existing bordered-card object name, so this costs
        a property set and not a stylesheet parse.
        """
        frame = QFrame()
        frame.setObjectName("Panel")
        body = QVBoxLayout(frame)
        body.setContentsMargins(10, 8, 10, 8)
        body.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("SectionTitle")
        heading.setWordWrap(True)
        body.addWidget(heading)
        subtitle = QLabel(note)
        subtitle.setObjectName("SectionSubtitle")
        subtitle.setWordWrap(True)
        body.addWidget(subtitle)
        frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        return frame

    def _build_said(self) -> None:
        self.entries = QListWidget()
        self.entries.setMinimumHeight(theme.px(110))
        self.entries.currentRowChanged.connect(self._on_entry_selected)
        self.entry_reader = QTextBrowser()
        self.entry_reader.setObjectName("ThoughtReader")
        self.entry_reader.setReadOnly(True)
        self.entry_reader.setOpenExternalLinks(False)
        self.entry_reader.setMinimumHeight(theme.px(90))
        self.entry_meta = QLabel("")
        self.entry_meta.setObjectName("ThoughtMeta")
        self.entry_meta.setWordWrap(True)

        self.entry_text = QPlainTextEdit()
        self.entry_text.setPlaceholderText(
            "What happened today, and what you make of it. Enter saves; "
            "Shift+Enter starts a new line."
        )
        self.entry_text.installEventFilter(self)
        self.entry_text.setMaximumHeight(theme.px(96))
        self.timeframe_picker = QComboBox()
        self.save_button = QPushButton("Save entry")
        self.save_button.clicked.connect(self._save)
        self.paste_forecast_button = QPushButton("Paste daily forecast…")
        self.paste_forecast_button.clicked.connect(self._paste_daily_forecast)
        self.after_the_fact = QLabel("")
        self.after_the_fact.setObjectName("CautionLabel")
        self.after_the_fact.setWordWrap(True)

        import market_journal

        self.timeframe_picker.addItems(list(market_journal.TIMEFRAMES))
        self.timeframe_picker.setCurrentText(market_journal.TIMEFRAME_D1)

        # Someone ELSE's words, under their own heading and never under what the
        # trader said (WISHLIST 10K's rule, kept).
        self.forecast_heading = QLabel("External forecast")
        self.forecast_heading.setObjectName("SectionTitle")
        self.forecast_note = QLabel("")
        self.forecast_note.setObjectName("SectionSubtitle")
        self.forecast_note.setWordWrap(True)
        self.forecast_box = QPlainTextEdit()
        self.forecast_box.setReadOnly(True)
        self.forecast_box.setMaximumHeight(theme.px(140))
        self.forecast_toggle = QPushButton("Show all")
        self.forecast_toggle.clicked.connect(self._toggle_forecast)

    def _build_traded(self) -> None:
        self.trades_table = QTableWidget(0, len(TRADE_COLUMNS))
        self.trades_table.setHorizontalHeaderLabels(list(TRADE_COLUMNS))
        self.trades_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.trades_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.trades_table.setMinimumHeight(theme.px(80))
        _fill_the_width(self.trades_table)
        self.trades_note = QLabel(
            "Read-only. The Journal page is where a trade is tagged and corrected."
        )
        self.trades_note.setObjectName("SectionSubtitle")
        self.trades_note.setWordWrap(True)

    def _build_chart_section(self) -> None:
        self.spy_note = QLabel(NO_CHART_NOTE)
        self.spy_note.setObjectName("SectionSubtitle")
        self.spy_note.setWordWrap(True)
        # G7.3's rule: a `CandleChart` is a pyqtgraph plot and this page opens
        # without one. `_chart_holder` is where the first one goes.
        self._chart_holder = QWidget()
        self._chart_holder.setMinimumHeight(theme.px(SPY_MIN_HEIGHT_PX))
        self._chart_layout = QVBoxLayout(self._chart_holder)
        self._chart_layout.setContentsMargins(0, 0, 0, 0)

    def _build_ideas(self) -> None:
        self.ideas_note = QLabel(NO_IDEAS_YET)
        self.ideas_note.setObjectName("SectionSubtitle")
        self.ideas_note.setWordWrap(True)

    def _section(self, title: str, *widgets, stretch_last: bool = False) -> QWidget:
        holder = QWidget()
        body = QVBoxLayout(holder)
        body.setContentsMargins(0, 0, 0, 0)
        if title:
            label = QLabel(title)
            label.setObjectName("SectionTitle")
            body.addWidget(label)
        for widget in widgets:
            if isinstance(widget, QWidget):
                body.addWidget(widget)
            else:
                body.addLayout(widget)
        if stretch_last and widgets:
            body.setStretch(body.count() - 1, 1)
        return holder

    def _left_column(self) -> QWidget:
        """What happened, the open theses under it, then the chart.

        Under, not beside (the trader's option 1, 2026-09-18): the theses are a
        short list ABOUT the story, and a chart is the one thing on this page
        that turns extra width into more information, so it takes the room the
        column has left.
        """
        self.story_section = self._section(
            "What happened", self.story_note, self.story_facts
        )
        theses_label = QLabel("Open theses")
        theses_label.setObjectName("SectionSubtitle")
        self.theses_section = self._section("", theses_label, self.theses)
        self.spy_section = self._section(
            "SPY, this session", self.spy_note, self._chart_holder, stretch_last=True
        )

        column = QWidget()
        body = QVBoxLayout(column)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addWidget(self.story_section)
        body.addWidget(self.theses_section)
        body.addWidget(self.spy_section, 1)
        return column

    def _right_column(self) -> QWidget:
        """What you said: the list over the reader, then the box you type in."""
        list_holder = QWidget()
        list_body = QVBoxLayout(list_holder)
        list_body.setContentsMargins(0, 0, 0, 0)
        list_body.addWidget(self.entries)
        reader_holder = QWidget()
        reader_body = QVBoxLayout(reader_holder)
        reader_body.setContentsMargins(0, 0, 0, 0)
        reader_body.addWidget(self.entry_meta)
        reader_body.addWidget(self.entry_reader, 1)

        self.said_split = QSplitter(Qt.Orientation.Vertical)
        self.said_split.setObjectName("DayReviewSaid")
        self.said_split.addWidget(list_holder)
        self.said_split.addWidget(reader_holder)
        self.said_split.setChildrenCollapsible(False)
        # Enough room for the 60/40 to BE 60/40: the reader's own floor is
        # 90 px plus its meta line, and a split whose smaller half is below a
        # child's minimum is not the split the preset asked for.
        self.said_split.setMinimumHeight(theme.px(300))
        desk_layout.apply_saved_sizes(self.said_split, SAID_SPLIT_KEY, SAID_SPLIT_WEIGHTS)
        desk_layout.track_preset(
            self, self.said_split, SAID_SPLIT_KEY, lambda _extent: SAID_SPLIT_WEIGHTS
        )
        desk_layout.persist_sizes(self, self.said_split, SAID_SPLIT_KEY)

        compose = QHBoxLayout()
        compose.addWidget(QLabel("Timeframe"))
        compose.addWidget(self.timeframe_picker)
        compose.addWidget(self.save_button)
        compose.addWidget(self.paste_forecast_button)
        compose.addStretch(1)

        forecast = QVBoxLayout()
        forecast.addWidget(self.forecast_heading)
        forecast.addWidget(self.forecast_note)
        forecast.addWidget(self.forecast_box)
        forecast.addWidget(self.forecast_toggle, 0, Qt.AlignLeft)
        self.forecast_section = QWidget()
        self.forecast_section.setLayout(forecast)

        self.said_section = self._section(
            "What you said",
            self.said_split,
            QLabel("New entry"),
            self.entry_text,
            compose,
            self.after_the_fact,
        )

        column = QWidget()
        body = QVBoxLayout(column)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addWidget(self.said_section, 1)
        body.addWidget(self.forecast_section)
        return column

    def _walkaway_row(self) -> QWidget:
        """The four populations as a 2 x 2 grid of equal cells.

        Equal COLUMNS, and rows that fit their content: three of the four are
        empty until TJ-2, and an empty population padded to the height of a
        table reads as a table that failed to load.
        """
        holder = QWidget()
        self.walkaway_grid = QGridLayout(holder)
        self.walkaway_grid.setContentsMargins(0, 0, 0, 0)
        self.walkaway_grid.setHorizontalSpacing(12)
        self.walkaway_grid.setVerticalSpacing(10)
        self.walkaway_grid.addWidget(
            self._section(WALKAWAY_TITLE, self.walkaway_note, self.rejected_that_worked_table),
            0,
            0,
        )
        for cell, widget in self.walkaway_cells.items():
            self.walkaway_grid.addWidget(widget, cell[0], cell[1], Qt.AlignTop)
        self.walkaway_grid.setColumnStretch(0, 1)
        self.walkaway_grid.setColumnStretch(1, 1)
        self.walkaway_grid.setRowStretch(0, 0)
        self.walkaway_grid.setRowStretch(1, 0)
        return holder

    def _bottom_row(self) -> QWidget:
        """What you traded, beside the desk's ideas. Two halves, full width."""
        self.traded_section = self._section(
            "What you traded", self.trades_note, self.trades_table
        )
        self.ideas_section = self._section("Ideas from the desk's AI", self.ideas_note)
        self.bottom_row = QWidget()
        row = QHBoxLayout(self.bottom_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        row.addWidget(self.traded_section, 1)
        row.addWidget(self.ideas_section, 1)
        return self.bottom_row

    def _build_layout(self) -> None:
        header = QHBoxLayout()
        header.addWidget(QLabel("Session"))
        header.addWidget(self.session_picker, 1)
        header.addWidget(self.refresh_button)

        # Row 2: the two columns. The ratio is the trader's to drag and it is
        # remembered per machine, like every other desk splitter.
        self.left_column = self._left_column()
        self.right_column = self._right_column()
        self.columns = QSplitter(Qt.Orientation.Horizontal)
        self.columns.setObjectName(COLUMNS_OBJECT_NAME)
        self.columns.addWidget(self.left_column)
        self.columns.addWidget(self.right_column)
        self.columns.setChildrenCollapsible(False)
        self.columns.setStretchFactor(0, 1)
        self.columns.setStretchFactor(1, 1)
        desk_layout.apply_saved_sizes(self.columns, COLUMN_SPLIT_KEY, COLUMN_WEIGHTS)
        desk_layout.track_preset(
            self, self.columns, COLUMN_SPLIT_KEY, lambda _extent: COLUMN_WEIGHTS
        )
        desk_layout.persist_sizes(self, self.columns, COLUMN_SPLIT_KEY)

        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addWidget(self.heading)
        body.addWidget(self.subtitle)
        body.addLayout(header)
        body.addWidget(self.provisional_note)
        body.addWidget(self.columns, 1)
        body.addWidget(self._walkaway_row())
        body.addWidget(self._bottom_row())
        body.addWidget(self.status)
        # No trailing stretch: the slack belongs to the two columns, and inside
        # them to the chart. A stretch here is what left a 3800 px screen with
        # 700 px of nothing under the last row.

        # ONE scroll area for the whole page (the Strength window's rule): every
        # section sized to its content, one scrollbar, nothing behind a tab.
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setWidget(page)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.addWidget(self.scroll)

    def refresh_reader_measure(self) -> None:
        """Keep the reader's column at a readable measure (the G3 fix round).

        A line of running text is readable at about 45-100 characters; the pane
        keeps its width and the TEXT is capped inside it.
        """
        try:
            for widget in (self.entry_reader, self.forecast_box):
                widget.ensurePolished()
                per_char = max(1, int(widget.fontMetrics().averageCharWidth()))
                widget.setMaximumWidth(max(theme.px(240), min(per_char * 100, theme.px(1200))))
        except Exception:  # noqa: BLE001 - a measure is never worth the page
            logging.debug("The Day Review reader measure failed.", exc_info=True)

    # -- the session picker ------------------------------------------------
    def _fill_session_picker(self, select: str | None = None) -> None:
        """Completed sessions, newest first, with Today offered as PROVISIONAL.

        The Daily Recap's rule, kept whole: today is in the list because the
        trader asks about it, and it is marked in the ENTRY rather than only in a
        note - a session that has not closed cannot be compared with one that
        has, and the label is the only thing standing between the two readings.
        """
        import market_calendar

        now = self._clock()
        self.session_picker.blockSignals(True)
        try:
            self.session_picker.clear()
            try:
                cursor = market_calendar.last_completed_session(now)
            except Exception:  # noqa: BLE001 - a calendar refusal never empties the page
                stamp = now.date().isoformat()
                self.session_picker.addItem(stamp, stamp)
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
        """Rebuild the list only when the newest completed session has moved."""
        import market_calendar

        try:
            head = market_calendar.last_completed_session(self._clock()).isoformat()
        except Exception:  # noqa: BLE001
            return
        if self.session_picker.count() and self.session_picker.itemData(0) == head:
            return
        self._fill_session_picker(select=self.session_date())

    def session_date(self) -> str:
        data = self.session_picker.currentData()
        if data:
            return str(data)
        return self.session_picker.currentText().strip()[:10]

    def show_session(self, session_date: str) -> None:
        """Select `session_date` in the picker and read it."""
        self._fill_session_picker(select=str(session_date))
        self.reload()

    # -- the automatic read ------------------------------------------------
    def start(self) -> None:
        """Begin the once-a-minute due check.

        Called by the host AFTER the window is up, never in the constructor: a
        timer started during construction runs while a test is still
        monkeypatching what it reads (the Daily Recap's rule, kept).
        """
        if not self._auto_timer.isActive():
            self._auto_timer.start()

    def auto_fired_session(self) -> str | None:
        """The session this process has already read automatically, if any."""
        return self._auto_fired_session

    def next_auto_read_at(self) -> datetime | None:
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
        if raw is None or hasattr(raw, "hour"):
            return raw
        return daily_recap_schedule.parse_auto_time(raw)

    def _on_auto_tick(self) -> None:
        try:
            self.poll_auto_read()
        except Exception:  # noqa: BLE001 - a timer slot never raises into Qt
            logging.debug("The Day Review automatic read failed.", exc_info=True)

    def poll_auto_read(self) -> str | None:
        """One due check. Returns the session read, or `None` when none was due.

        `daily_recap_schedule` stays the decision and this only obeys it: the
        noon read first, then the post-close one. The post-close tick also builds
        that session's index ONCE, through the service's one named seam, so the
        first open after the close is the fast one.
        """
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
            self._build_index_for(due)
        else:
            self._auto_fired_session = due
        self.show_session(due)
        return due

    def _build_index_for(self, session_date: str) -> None:
        """Start that session's index build on a WORKER and return at once.

        Never inline: the build streams the 476 MB intraday log and three other
        stores, and called from the timer slot it froze the desk for 22.8 s
        (reviewer, 2026-09-17). Single-flight - a second tick while one is in
        flight is ignored rather than queued - and quiet on every failure path,
        because the index is derived and rebuildable.

        `getattr` because the seam is the SERVICE's: a host that hands this page
        a reader without one still gets its read, just not the fast second one.
        """
        session = str(session_date or "")[:10]
        builder = getattr(self.service, "build_index_for", None)
        if not callable(builder) or not session:
            return
        if self._index_worker is not None and self._index_worker.isRunning():
            return
        try:
            worker = _IndexBuildWorker(self.service, session, self)
            worker.built.connect(self._on_index_built)
            worker.failed.connect(self._on_index_failed)
            self._index_worker = worker
            self._building_index = session
            self.status.setText(BUILDING_INDEX_NOTE.format(session=session))
            self.statusChanged.emit(self.status.text())
            worker.start()
        except Exception:  # noqa: BLE001 - a cache never costs the page
            self._index_worker = None
            self._building_index = ""
            logging.debug("The Day Review index build could not start.", exc_info=True)

    def _on_index_built(self, session_date: str) -> None:
        """The index landed. Repaint that session if it is the one on screen."""
        self._index_worker = None
        self._building_index = ""
        self.status.setText(f"Day Review: {session_date} is indexed.")
        self.statusChanged.emit(self.status.text())
        if str(session_date) == self.session_date():
            self.reload()

    def _on_index_failed(self, session_date: str, reason: str) -> None:
        self._index_worker = None
        self._building_index = ""
        logging.info("The Day Review index for %s was not built: %s", session_date, reason)
        self.status.setText(
            f"{session_date} could not be indexed ({reason}); the page reads the "
            "stores directly."
        )
        self.statusChanged.emit(self.status.text())

    # -- reading -----------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Read the first time the page is actually looked at.

        The desk builds every left-nav panel at startup and most are never
        opened, so the cost stays with the page that asked for it.
        """
        super().showEvent(event)
        if not self._loaded_once:
            self._loaded_once = True
            self.reload()

    def set_bars_reader(self, reader: Callable[[str], Any] | None) -> None:
        """Hand the page the desk's own M5 cache accessor.

        The PAGE holds it, not the service, because it may only be called on the
        Qt thread: `alert_center.journal_chart_bars` mutates the Alert Center's
        bar cache and arms a `QTimer.singleShot`, and a `singleShot` armed from a
        worker never fires, which latched the D1 prefetch flag and killed prefetch
        for the session (reviewer, 2026-09-17).
        """
        self._bars_reader = reader

    def _spy_bars_for(self, session_date: str) -> list[dict[str, Any]]:
        """Today's SPY M5 bars, read HERE, on the Qt thread. A cache read.

        Only for a session that has not closed: the accessor holds the running
        scanner's own chart, which is today's. TJ-2 brings the stored bars for a
        past session, and until then the page says so.
        """
        if self._bars_reader is None:
            return []
        if str(session_date) != self._clock().date().isoformat():
            return []
        try:
            bars = self._bars_reader("SPY")
        except Exception:  # noqa: BLE001 - no bars is a note, never a failed page
            logging.debug("The SPY bars could not be read.", exc_info=True)
            return []
        if isinstance(bars, tuple) and len(bars) == 2:
            # `journal_chart_bars` answers `(m5, d1)`; this page draws the M5.
            bars = bars[0]
        return [dict(bar) for bar in (bars or ()) if isinstance(bar, Mapping)]

    def reload(self) -> None:
        """Ask the worker for the selected session. Never blocks the page."""
        if self._worker is not None and self._worker.isRunning():
            return
        self._refresh_session_picker()
        self._sync_after_the_fact()
        self.status.setText(LOADING_NOTE)
        session = self.session_date()
        self._worker = _DayReadWorker(
            self.service, session, self, spy_m5_bars=self._spy_bars_for(session)
        )
        self._worker.loaded.connect(self.render)
        self._worker.failed.connect(self._render_failure)
        self._worker.start()

    def _render_failure(self, reason: str) -> None:
        self.status.setText(f"the session could not be read: {reason}")
        self.statusChanged.emit(self.status.text())

    def _refresh_if_loaded(self) -> None:
        if self._loaded_once:
            self.reload()

    # -- rendering ---------------------------------------------------------
    def render(self, payload: Mapping[str, Any]) -> None:
        """Draw one payload. Formatting only - it computes nothing.

        Tolerates a payload with nothing in it: a first paint before any read and
        a read that failed are the same shape.
        """
        payload = dict(payload or {})
        self._payload = payload
        session = str(payload.get("session_date") or self.session_date())
        self.provisional_note.setText(
            "This session has NOT closed - every number on it is provisional."
            if payload.get("provisional")
            else f"Session {session}, closed and measured."
        )
        self._render_story(payload.get("story"))
        self._render_theses(payload.get("theses") or [])
        self._render_walkaway(tuple(payload.get("rejected_that_worked") or ()))
        self._render_entries(list(payload.get("entries") or []))
        self._render_forecast(dict(payload.get("forecast") or {}))
        self._render_trades(list(payload.get("trades") or []))
        self._render_chart(list(payload.get("spy_m5_bars") or []))
        error = str(payload.get("error") or "")
        self.status.setText(error or f"Day Review: {session}")
        self.statusChanged.emit(self.status.text())

    def _render_story(self, story: Any) -> None:
        """The deterministic facts, under the fixed "no story yet" line.

        The line is FIXED and always shown until TJ-4: what is below it is what
        the desk MEASURED, and calling that a story would be the machine
        claiming a reading nobody wrote.
        """
        self.story_note.setText(NO_STORY_YET)
        if story is None:
            self.story_facts.setText("")
            return
        lines: list[str] = []
        for cell in tuple(getattr(story, "measured", ()) or ()):
            symbol = str(cell.get("symbol") or "")
            if str(cell.get("status") or "") != "measured":
                lines.append(f"{symbol}: unmeasured — {cell.get('reason') or 'no completed bars'}")
                continue
            change = cell.get("change_pct")
            span = cell.get("range_atr")
            text = f"{symbol}:"
            if isinstance(cell.get("close"), (int, float)):
                text += f" close {cell['close']:.2f}"
            if isinstance(change, (int, float)):
                text += f", {change:+.2f}%"
            if isinstance(span, (int, float)):
                text += f", range {span:.2f} ATR"
            lines.append(text)
        for note in tuple(getattr(story, "notes", ()) or ()):
            lines.append(str(note))
        self.story_facts.setText("\n".join(lines))

    def _render_theses(self, rows) -> None:
        self.theses.clear()
        for row in rows:
            claim = str(row.get("claim") or "").strip() or "(no claim found)"
            stance = str(row.get("stance") or "")
            horizon = str(row.get("horizon") or "")
            parts = [f"{stance}: {claim}" if stance else claim]
            if horizon:
                parts.append(f"horizon {horizon}")
            self.theses.addItem("  ·  ".join(parts))

    def _render_walkaway(self, rows) -> None:
        self._walkaway_rows = tuple(rows)
        table = self.rejected_that_worked_table
        table.setRowCount(len(self._walkaway_rows))
        for index, row in enumerate(self._walkaway_rows):
            for column, (header, measure) in enumerate(WALKAWAY_COLUMNS):
                text, tip = self._walkaway_cell(row, header, measure)
                item = QTableWidgetItem(text)
                if tip:
                    item.setToolTip(tip)
                table.setItem(index, column, item)
        # No width call here: the columns measure themselves (`_fill_the_width`,
        # set once at construction) and the last one stretches, so a repaint
        # costs the rows and nothing else.
        self.walkaway_note.setText(
            f"{len(self._walkaway_rows)} refusal(s) whose later path went the way "
            "you turned down. Double-click a row to chart it."
            if self._walkaway_rows
            else "Nothing you passed on ran, on this session's measured rows."
        )

    def _walkaway_cell(self, row: Any, header: str, measure: str | None) -> tuple[str, str]:
        """One cell's text and its tooltip. `None` is a dash WITH its reason."""
        if measure is not None:
            value = (getattr(row, "measures", {}) or {}).get(measure)
            if value is None:
                return UNMEASURED, str((getattr(row, "unavailable", {}) or {}).get(measure, ""))
            return f"{float(value):+.2f}%", ""
        detail = getattr(row, "detail", {}) or {}
        moment = getattr(row, "observed_at", None)
        if header == "Time":
            if moment is None:
                return UNMEASURED, "this observation carries no timestamp"
            return moment.strftime("%H:%M"), moment.isoformat()
        if header == "Symbol":
            return str(getattr(row, "symbol", "")), ""
        if header == "Side":
            return str(getattr(row, "side", "")), ""
        if header == "Verdict":
            return str(detail.get("verdict") or ""), str(detail.get("channel") or "")
        if header == "My reason":
            codes = ", ".join(str(code) for code in (detail.get("reason_codes") or ()))
            return str(detail.get("reason") or codes or ""), codes
        if header == "Environment":
            observation = str(
                getattr(row, "observation_context", "") or getattr(row, "d1_environment", "")
            )
            entry = str(getattr(row, "entry_context", "") or "")
            return (f"{observation} → {entry}" if entry else observation), ""
        return "", ""

    def _render_entries(self, rows) -> None:
        """The day's words, OLDEST FIRST, machine rows never.

        Oldest first because this page is read as a day: the morning thought
        comes before the answer to it. The old page's newest-first list was a
        list of every session at once, which is a different question.
        """
        previous = self._selected_entry_id()
        kept = [dict(row) for row in rows if not _is_machine_row(row)]
        kept.sort(key=lambda row: str(row.get("created_at") or ""))
        self._entries = kept
        blocked = self.entries.blockSignals(True)
        try:
            self.entries.clear()
            for entry in kept:
                marker = (
                    "  ·  [written after the session]"
                    if entry.get("written_after_the_session")
                    else ""
                )
                label = (
                    f"{_clock_text(entry.get('created_at'))}"
                    f"  ·  {entry.get('timeframe') or ''}{marker}"
                    f"  ·  {_excerpt(entry.get('text'))}"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, str(entry.get("entry_id") or ""))
                item.setToolTip(str(entry.get("text") or ""))
                self.entries.addItem(item)
            if not kept:
                self.entries.addItem("Nothing was written for this session.")
        finally:
            self.entries.blockSignals(blocked)
        if previous:
            for index, entry in enumerate(kept):
                if str(entry.get("entry_id") or "") == previous:
                    self.entries.setCurrentRow(index)
                    return
        self._fill_reader(None)

    def _selected_entry_id(self) -> str:
        item = self.entries.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.UserRole) or "")

    def _on_entry_selected(self, _row: int) -> None:
        """The WORDS, synchronously. No worker, no chart, no capture read."""
        entry_id = self._selected_entry_id()
        entry = next(
            (row for row in self._entries if str(row.get("entry_id") or "") == entry_id),
            None,
        )
        self._fill_reader(entry)

    def _fill_reader(self, entry: Mapping[str, Any] | None) -> None:
        if not entry:
            self.entry_meta.setText("")
            self.entry_reader.setPlainText("")
            return
        origin = str(entry.get("origin") or "")
        stamp = _clock_text(entry.get("created_at"))
        meta = f"written {stamp}  ·  {entry.get('timeframe') or ''}  ·  {origin}"
        if entry.get("written_after_the_session"):
            meta += "  ·  written after the session"
        self.entry_meta.setText(meta)
        self.entry_reader.setPlainText(str(entry.get("text") or ""))

    def _render_forecast(self, forecast: Mapping[str, Any]) -> None:
        text = str(forecast.get("text") or "")
        if not text.strip():
            self._forecast_expanded = False
            self.forecast_box.setPlainText("")
            self.forecast_note.setText(
                "No forecast has been pasted for this session."
            )
            self.forecast_toggle.setVisible(False)
            return
        import forecast_brief

        shown, hidden = forecast_brief.collapse(text, lines=FORECAST_COLLAPSED_LINES)
        self.forecast_box.setPlainText(text if self._forecast_expanded else shown)
        self.forecast_toggle.setVisible(bool(hidden))
        self.forecast_toggle.setText("Show less" if self._forecast_expanded else "Show all")
        source = str(forecast.get("source_model") or "unknown")
        brief = forecast.get("brief")
        headline = forecast_brief.headline(brief) if brief is not None else ""
        self.forecast_note.setText(
            f"Outside commentary from {source} - not your view unless you write an "
            f"entry adopting it."
            + (f" {headline}." if headline else "")
            + (f" {hidden} more line(s)." if hidden and not self._forecast_expanded else "")
        )

    def _toggle_forecast(self) -> None:
        self._forecast_expanded = not self._forecast_expanded
        self._render_forecast(dict(self._payload.get("forecast") or {}))

    def _render_trades(self, rows) -> None:
        self.trades_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            quantity = row.get("quantity")
            if quantity is None:
                quantity = row.get("quantity_opened")
            values = (
                _clock_text(row.get("opened_at")),
                str(row.get("symbol") or ""),
                str(row.get("direction") or ""),
                self._number(quantity, decimals=0),
                self._number(row.get("net_pnl"), signed=True),
                str(row.get("status") or ""),
            )
            for column, text in enumerate(values):
                self.trades_table.setItem(index, column, QTableWidgetItem(str(text)))

    @staticmethod
    def _number(value: Any, *, decimals: int = 2, signed: bool = False) -> str:
        """A number, or a dash. Never a 0.00 for something nobody measured."""
        if value is None or value == "":
            return UNMEASURED
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if signed:
            return f"{number:+.{decimals}f}"
        return f"{number:.{decimals}f}"

    # -- the one chart -----------------------------------------------------
    def _ensure_chart(self):
        """Build the SPY pane ONCE, the first time there are bars for it.

        A `CandleChart` is a pyqtgraph plot; the old page's four were its whole
        construction cost. Every reader here tolerates `None`, which is what
        "not built yet" looks like.
        """
        if self._chart is None:
            from ui.widgets.candle_chart import CandleChart

            self._chart = CandleChart()
            self._chart.setMinimumHeight(theme.px(200))
            self._chart_layout.addWidget(self._chart)
        return self._chart

    def _render_chart(self, bars) -> None:
        # A bar with no `dt` cannot be placed on a time axis, so it is DROPPED
        # and counted rather than drawn at an invented moment; the count is said
        # once per render, not once per bar.
        drawable = [bar for bar in bars or () if bar.get("dt") is not None]
        dropped = len(bars or ()) - len(drawable)
        if dropped:
            logging.info(
                "Day Review: %d SPY bar(s) carried no timestamp and are not drawn.",
                dropped,
            )
        if not drawable:
            self.spy_note.setText(
                NO_CHART_NOTE
                + (f" ({dropped} bar(s) carried no timestamp.)" if dropped else "")
            )
            if self._chart is not None:
                self._chart.set_data([])
                self._chart.setVisible(False)
            return
        chart = self._ensure_chart()
        chart.setVisible(True)
        chart.set_data(drawable, timeframe="m5")
        self.spy_note.setText(
            f"SPY M5 — {len(drawable)} completed bar(s) the desk holds."
            + (f" {dropped} carried no timestamp and are not drawn." if dropped else "")
        )

    def _activate_walkaway(self, item) -> None:
        """Ask the host for a BOARD chart of this row's name.

        `show_board_symbol` is the door for a board on another page; a board
        chart holds no place in the waiting list, so nothing here is re-queued
        or skip-counted.
        """
        if item is None:
            return
        index = item.row()
        if index < 0 or index >= len(self._walkaway_rows):
            return
        row = self._walkaway_rows[index]
        symbol = str(getattr(row, "symbol", "") or "").strip().upper()
        if not symbol:
            return
        self.chartRequested.emit(symbol, str(getattr(row, "side", "") or ""))

    # -- writing -----------------------------------------------------------
    def eventFilter(self, watched, event):  # noqa: N802 (Qt override)
        """Enter saves, Shift+Enter starts a new line (decision 0016 answer 11)."""
        try:
            if (
                watched is self.entry_text
                and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            ):
                self._save()
                return True
        except Exception:  # noqa: BLE001 - a key handler never breaks the page
            pass
        return super().eventFilter(watched, event)

    def _sync_after_the_fact(self) -> None:
        """Say plainly when the entry being typed is about a past session."""
        session = self.session_date()
        today = self._clock().date().isoformat()
        if session and session < today:
            self.after_the_fact.setText(
                f"This entry is ABOUT {session} and will be stamped as written "
                f"today ({today}). It is filed under the session, never backdated."
            )
        else:
            self.after_the_fact.setText("")

    def _save(self) -> None:
        result = self.service.write_entry(
            text=self.entry_text.toPlainText(),
            session_date=self.session_date(),
            timeframe=self.timeframe_picker.currentText(),
            origin="journal_page",
        )
        if result.get("ok"):
            self.entry_text.clear()
            self.status.setText("Entry saved.")
            self._refresh_if_loaded()
        else:
            self.status.setText(f"Entry NOT saved: {result.get('reason', '')}")
        self.statusChanged.emit(self.status.text())

    # -- the pasted forecast -----------------------------------------------
    def _paste_daily_forecast(self) -> None:
        """Ask for the text, the session it is about, and who wrote it."""
        payload = self._ask_for_forecast()
        if payload is None:
            return
        self._import_forecast(payload)

    def _ask_for_forecast(self) -> dict | None:
        """The paste dialog. Returns what was typed, or `None` when cancelled.

        The date field DEFAULTS to the date in the brief's own first heading
        (`forecast_brief.parse`), and stops defaulting the moment the trader
        types in it: the document knows which day it is about, and the trader
        outranks the document.
        """
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLineEdit,
        )

        import forecast_brief

        dialog = QDialog(self)
        dialog.setWindowTitle("Paste a daily forecast")
        text_box = QPlainTextEdit()
        text_box.setPlaceholderText("Paste the brief exactly as it was written.")
        session_box = QLineEdit(self.session_date())
        model_box = QLineEdit("chatgpt")
        created_box = QLineEdit()
        created_box.setPlaceholderText("When it was written, if you know (optional)")
        touched = {"session": False}
        session_box.textEdited.connect(lambda _text: touched.update(session=True))

        def _follow_the_brief() -> None:
            if touched["session"]:
                return
            parsed = forecast_brief.parse(text_box.toPlainText()).title_date
            session_box.setText(parsed or self.session_date())

        text_box.textChanged.connect(_follow_the_brief)

        form = QFormLayout(dialog)
        form.addRow(QLabel("The forecast, verbatim"))
        form.addRow(text_box)
        form.addRow("The session it is about", session_box)
        form.addRow("Source", model_box)
        form.addRow("Written at", created_box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None
        return {
            "text": text_box.toPlainText(),
            "target_session": session_box.text().strip() or self.session_date(),
            "source_model": model_box.text().strip(),
            "created_at_claimed": created_box.text().strip(),
        }

    def _import_forecast(self, payload: Mapping[str, Any]) -> dict:
        """The write half, separate from the dialog so it can be tested."""
        text = str(payload.get("text") or "").strip()
        if not text:
            self.status.setText("Nothing was pasted, so nothing was imported.")
            return {"ok": False, "reason": "empty forecast"}
        session = str(payload.get("target_session") or "").strip() or self.session_date()
        result = self.service.import_daily_forecast(
            text=text,
            target_session=session,
            source_model=str(payload.get("source_model") or ""),
            created_at_claimed=str(payload.get("created_at_claimed") or ""),
        )
        if result.get("ok"):
            self.status.setText(
                f"Forecast stored for {session} as outside commentary. It is not "
                "your view until you write an entry adopting it."
            )
            self._refresh_if_loaded()
        else:
            self.status.setText(f"Forecast NOT imported: {result.get('reason', '')}")
        self.statusChanged.emit(self.status.text())
        return result

    # -- teardown ----------------------------------------------------------
    def shutdown(self) -> None:
        try:
            self._auto_timer.stop()
        except RuntimeError:  # pragma: no cover - already torn down
            pass
        for worker in (self._worker, self._index_worker):
            if worker is not None and worker.isRunning():
                # Bounded: a desk that will not close is worse than an index
                # nobody collected, and the index is rebuildable.
                worker.wait(2000)


__all__ = [
    "AUTO_POLL_INTERVAL_MS",
    "COLUMNS_OBJECT_NAME",
    "COLUMN_SPLIT_KEY",
    "COLUMN_WEIGHTS",
    "DayReviewPanel",
    "EXCERPT_LIMIT",
    "FORECAST_COLLAPSED_LINES",
    "LOOKBACK_SESSIONS",
    "NO_CHART_NOTE",
    "NO_IDEAS_YET",
    "NO_STORY_YET",
    "PICKER_SESSIONS",
    "SAID_SPLIT_KEY",
    "SAID_SPLIT_WEIGHTS",
    "SPY_MIN_HEIGHT_PX",
    "STORY_MIN_HEIGHT_PX",
    "TRADE_COLUMNS",
    "WALKAWAY_COLUMNS",
    "WALKAWAY_PLACEHOLDERS",
    "WALKAWAY_PLACEHOLDER_CELLS",
    "WALKAWAY_TITLE",
]
