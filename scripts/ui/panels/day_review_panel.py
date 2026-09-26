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
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from PySide6.QtCore import QEvent, QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
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
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import daily_recap_schedule
from ai_jobs import window
from ui import theme
from ui.panels import desk_layout
from ui.widgets.data_table import MEASURE_PRECISION_ROWS
from slot_output_reads import note_slot_output_read
from swallowed import note_swallowed

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

#: What the "What happened" section says when the night has not written a story
#: for this session. Said plainly rather than leaving an empty box that reads as
#: a read that failed. The deterministic facts stay under it either way.
NO_STORY_YET = "No story yet. It is written overnight."

#: What a DAYTIME Redo says. Local inference is night-only, seven days a week
#: (trader, 2026-09-19; decision 0021 answer 19), so the button writes a
#: `redo_requested` marker the nightly slot honours and says so. A 14 GB model
#: load in front of the trader's own market prep is the thing the rule is about.
STORY_QUEUED_NOTE = (
    "Queued for tonight - the desk writes the story overnight, never by day."
)

#: What it says when the night window is open and the button did what it says.
STORY_REDO_STARTED_NOTE = "Rewriting {session}'s story in the background…"

#: What it says while the Redo's own pack build runs on its worker. The night
#: can only narrate a session it has a PACK for, so the click builds one first.
STORY_BUILDING_PACK_NOTE = "Building {session}'s facts before queuing the story…"

#: And when that build could not produce one. Said plainly, and NOTHING is
#: queued: a marker for a session with no pack is a request the night can never
#: answer (reviewer round 2, 2026-09-20 - the live home folder held three
#: session folders and zero packs).
REDO_NO_PACK_NOTE = "No pack could be built for {session} - nothing queued."

#: What it says when the session cannot be redone at all - a day the exchange
#: never opened, or one that has not CLOSED yet. Said on BOTH branches: by day
#: the marker is refused, and by night the process is too (reviewer round 3).
REDO_REFUSED_NOTE = "{session} cannot be redone: {reason}"

#: The slot the Redo button runs, by name. One spelling, used by the argv and
#: by the CLI that receives it.
REDO_SLOT = "day_review_narration"

#: What the SPY section says when the desk has no bars for the session.
NO_CHART_NOTE = (
    "No SPY chart for this session yet. The desk shows today's bars live and "
    "saves each day's bars after the close."
)

#: What the ideas section says on a session the night had nothing to say about.
#: Since TJ-6 the card itself is what speaks; this is the line that stands in
#: its place, and "nothing yet" is a different sentence from "nothing worked".
NO_IDEAS_YET = (
    "Nothing yet - the desk's AI writes up to three ideas a night, each one "
    "citing your own sessions, and you keep or dismiss each one here."
)

#: The heading of the "rejected" population.
WALKAWAY_TITLE = "Passed, and it ran"

#: The 16-column walk-away model: TJ-2B's columns plus TJ-11's three extra
#: moves, each in percent and in ATR, and the versioned real-miss verdict.
#: Spelled in FULL: the TJ-1L width rule exists because a clipped header
#: carries no ellipsis to say so.
TJ2B_WALKAWAY_COLUMNS = (
    "Time", "Symbol", "Side", "What you did",
    "Ran after %", "Against you first %", "At the close %",
    "Ran after (ATR)", "Against you first (ATR)", "At the close (ATR)",
    "Real miss", "Held at close %",
    "Traded?", "You made", "Left on the table %", "State",
)

#: The five walk-away populations and the heading each one carries. TJ-11 adds
#: the fifth: the D1 calls of the previous five sessions, measured to THIS
#: session's close.
TJ2B_WALKAWAY_TITLES: tuple[tuple[str, str], ...] = (
    ("rejected", WALKAWAY_TITLE),
    ("liked_not_traded", "Liked but never traded"),
    ("traded_left_early", "Traded, then left early"),
    ("claimed_d1", "Claimed D1 picks"),
    ("earlier_calls", "Earlier calls, now"),
)

#: The one miss table's filter chips, in order, with their short labels.
MISS_FILTERS: tuple[tuple[str, str], ...] = (
    ("rejected", "Passed & ran"),
    ("liked_not_traded", "Liked not traded"),
    ("traded_left_early", "Left early"),
    ("claimed_d1", "Claimed D1"),
    ("earlier_calls", "Earlier calls"),
)

#: The columns shown until "More columns" is on.
MISS_DEFAULT_COLUMNS: tuple[str, ...] = (
    "Time", "Symbol", "Side", "What you did",
    "Ran after %", "Against you first %", "At the close %", "Real miss",
)
#: A population whose useful columns differ from the default set.
MISS_DEFAULT_COLUMNS_BY_FILTER: dict[str, tuple[str, ...]] = {
    "traded_left_early": (
        "Time", "Symbol", "Side", "What you did",
        "Ran after %", "You made", "Left on the table %", "Real miss",
    ),
}

#: The miss table's height; the name chart beside it matches it.
MISS_TABLE_HEIGHT_PX = 340


def _tj2_pct(value: object) -> str:
    return UNMEASURED if value is None else f"{float(value):+.2f}%"


def _tj2_number(value: object) -> str:
    return UNMEASURED if value is None else f"{float(value):+.2f}"


def _button_text(text: object) -> str:
    """A button label with `&` shown, not read as a keyboard mnemonic."""
    return str(text or "").replace("&", "&&")


#: A night-story citation such as `[said:mj-2026-09-22-b6559bf2669f:prediction]`.
_CITATION = re.compile(r"\s*\[(?:said|cite|src|ref):[^\]]*\]")


def _without_citations(text: object) -> str:
    """Story text without the machine citation ids the night writes."""
    return _CITATION.sub("", str(text or "")).strip()


def _plain_cell(value: object) -> str:
    """A stored code as words: `no_run` -> `no run`, `unmeasured` -> `not measured`."""
    text = str(value or "").replace("_", " ").strip()
    return text.replace("unmeasured", "not measured")


def _median_ran_after(rows) -> float | None:
    moves = sorted(row.ran_after_pct for row in rows if row.ran_after_pct is not None)
    if not moves:
        return None
    middle = len(moves) // 2
    return moves[middle] if len(moves) % 2 else (moves[middle - 1] + moves[middle]) / 2

#: The two columns (trader, 2026-09-18: *"there's a lot of empty space
#: horizontally that's not being efficiently used"*, and, offered three shapes,
#: he chose two columns). LEFT is what happened and the chart, RIGHT is what he
#: said: a chart uses every pixel of width it is given, and a column of running
#: text does not.
#: TJ-12: the six-line report card that HEADS the page, above the story. It is
#: a THIRD row of the page's own column, never a cell inside either column -
#: what the trader missed and what they did well is the first thing on the page.
REPORT_CARD_OBJECT_NAME = "DayReviewReportCard"
REPORT_CARD_LINE_OBJECT_NAME = "DayReviewReportCardLine"

#: Where each card line's click goes, by target name. The two walk-away targets
#: are `walkaway_day.TABLES` entries and are resolved through the panel's own
#: table map, so a renamed table breaks the click loudly instead of silently.
REPORT_CARD_TITLE = "Your report card"

#: What a line says before anything has been read. Never a zero and never a
#: rate: a first paint has measured nothing.
REPORT_CARD_PLACEHOLDERS: dict[str, str] = {
    "did_well": "Did well: nothing read yet.",
    "missed": "Missed: nothing read yet.",
    "your_reads": "Your reads: nothing read yet.",
    "congruence": "Congruence: nothing read yet.",
    "process": "Process: nothing read yet.",
    "how_fresh": "How fresh: nothing read yet.",
}

#: How wide one card line may run before it is wrapped onto another visual row.
#: A `QPushButton` does not word-wrap, and a line the trader cannot finish
#: reading is a line that does not say what it measured.
REPORT_CARD_WRAP_CHARS = 120

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

#: The name pane that opens BESIDE the walk-away tables (TJ-3): the same floor,
#: and a width floor so the tables cannot squeeze it into a smear either.
NAME_CHART_MIN_HEIGHT_PX = 260
NAME_CHART_MIN_WIDTH_PX = 320

#: What the name pane says before a row is clicked, and when the session's bars
#: file never got that name. Said rather than drawn on somebody else's tape.
NAME_CHART_IDLE_NOTE = (
    "Click a row to see that name's session, with what you said on it."
)
NAME_CHART_MISSING_NOTE = (
    "{symbol}: no bars were saved for this name on this session, so there is "
    "no chart."
)

#: What the trade chart says before a trade is picked, and on a day with none.
TRADE_CHART_IDLE_NOTE = "Pick a trade to see its chart with your entry and exit."
TRADE_CHART_NO_TRADES_NOTE = "No trades this session."

#: The local setting that remembers the session on screen.
SESSION_SETTING_KEY = "qt_day_review_session_v1"

#: Where each glance tile's click goes (a `reveal_card_target` name).
GLANCE_TARGETS: dict[str, str] = {
    "pnl": "trades",
    "trades": "trades",
    "planned": "trades",
    "calls": "said",
    "day_type": "story",
    "market_axes": "story",
    "biggest_win": "trades",
    "biggest_miss": "miss",
    "sparkline": "trades",
}

#: What the SPY caption says when the session has no tape at all. A mark with no
#: bar to sit on is SAID, never silently dropped.
NO_TAPE_MARKER_NOTE = "No tape for this session - marks not drawn."

#: The story's floor. It GROWS with its text above this (TJ-4 writes paragraphs);
#: below it, an empty story reads as a broken section.
STORY_MIN_HEIGHT_PX = 120

#: The trade line's columns. Read-only: the Journal page is still where a trade
#: is tagged and corrected (decision 0021 consequences).
TRADE_COLUMNS: tuple[str, ...] = (
    "Time", "Symbol", "Direction", "Qty", "Whole trade net", "Status", "Bot grade",
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
FETCHING_BARS_NOTE = "Fetching that day's bars: {session}…"


def redo_command(session_date: str) -> list[str]:
    """The argv `plan.md` TJ-4 change 4 names. PURE - it runs nothing.

    `run_ai_jobs.py --slot day_review_narration --force --session <date>`.
    `--force` here re-spends the attempt caps and the already-done check; it
    does NOT buy the clock, which is why the button refuses by day and queues
    instead.
    """
    import sys

    script = Path(__file__).resolve().parents[3] / "scripts" / "run_ai_jobs.py"
    return [
        sys.executable,
        str(script),
        "--slot",
        REDO_SLOT,
        "--force",
        "--session",
        str(session_date or "")[:10],
    ]


def launch_redo_process(session_date: str) -> None:
    """Start the redo in a below-normal child PROCESS, and return at once.

    A process rather than a thread: a 14 GB model load can never share the
    desk's own heap, and a run that goes wrong must not be able to take down
    the window the trader watches charts in - the same reasoning that keeps the
    whole AI layer out of the GUI (`scripts/run_ai_jobs.py`).
    """
    import subprocess

    flags = 0
    if os.name == "nt":
        flags = int(
            getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    subprocess.Popen(  # noqa: S603 - argv is built here, never from user text
        redo_command(session_date), creationflags=flags, close_fds=True
    )


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


def _prediction_text(entry: Mapping[str, Any]) -> str:
    """The clicked call as one line, or `""`. Worded once, in `market_journal`."""
    try:
        from market_journal import prediction_line

        return prediction_line(entry)
    except Exception:  # noqa: BLE001 - a page never fails on a missing call
        return ""


def _display_zone():
    """The desk's one display clock: `market_session`'s market-local zone."""
    try:
        from market_session import get_market_local_timezone

        return get_market_local_timezone()[0]
    except Exception:  # noqa: BLE001 - a zone lookup never costs the page
        return None


def _zone_label(zone: Any = None, *, when: datetime | None = None) -> str:
    """A short name for the display zone, e.g. `PDT`, for the page header."""
    zone = zone if zone is not None else _display_zone()
    if zone is None:
        return "desk local time"
    try:
        moment = (when or datetime.now()).astimezone(zone)
        name = moment.tzname() or str(zone)
        # Windows spells some zones out ("Pacific Daylight Time"); use initials.
        if " " in name:
            name = "".join(word[0] for word in name.split() if word[:1].isalpha()).upper()
        return name
    except Exception:  # noqa: BLE001
        return str(zone)


def _as_moment(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    raw = str(value or "").strip()
    if len(raw) < 16:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _clock_text(created_at: Any, session: str = "", *, zone: Any = None) -> str:
    """`HH:MM` in the display zone, with the date only when it is not `session`.

    An aware stamp is converted; a naive one is already market-local (the bar
    rule). Unparseable is a dash, never a guess.
    """
    moment = _as_moment(created_at)
    if moment is None:
        return UNMEASURED
    if moment.tzinfo is not None:
        zone = zone if zone is not None else _display_zone()
        if zone is not None:
            moment = moment.astimezone(zone)
    text = moment.strftime("%H:%M")
    day = moment.date().isoformat()
    if session and day != str(session)[:10]:
        text = f"{moment.strftime('%b')} {moment.day} {text}"
    return text


def _status_problems(error: Any) -> str:
    """The payload's real read failures, minus facts about the day itself.

    A day with no trades has no exit notes to read; that is not a failure.
    """
    from ui.services.day_review_service import NO_TRADES_EXIT_NOTE

    parts = [
        part.strip() for part in str(error or "").split(" · ")
        if part.strip() and part.strip() != NO_TRADES_EXIT_NOTE
    ]
    return " · ".join(parts)


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
            # The durable tape is second and non-fatal: the index still lands
            # when yfinance is unavailable after the close.
            self._service.build_session_bars_for(
                self._session, lookback_sessions=LOOKBACK_SESSIONS
            )
        except Exception as exc:  # noqa: BLE001 - a cache never costs the page
            self.failed.emit(self._session, str(exc))
            return
        # TJ-10: the session's reads are graded HERE, on this worker, after the
        # tape it measures them against has landed. Through `getattr` because
        # this worker is handed a service by its host and a host that has no
        # grader must still get its index - and in its own guard, because a
        # failed grade costs the verdicts and nothing else.
        grade = getattr(self._service, "build_reads_for", None)
        if callable(grade):
            try:
                grade(self._session)
            except Exception:  # noqa: BLE001 - a verdict never costs the index
                logging.debug(
                    "The session's reads could not be graded.", exc_info=True
                )
        # TJ-4: the day pack, AFTER the grades, because its `reads` section IS
        # what the grader just wrote. Same shape as the line above it: through
        # `getattr`, because a host that hands this page a reader without the
        # seam must still get its index, and in its own guard, because a failed
        # pack costs the night's story and nothing else.
        build_pack = getattr(self._service, "build_pack_for", None)
        if callable(build_pack):
            try:
                build_pack(self._session)
            except Exception:  # noqa: BLE001 - a pack never costs the index
                logging.debug("The day pack could not be built.", exc_info=True)
        self.built.emit(self._session)


class _BarsBackfillWorker(QThread):
    """Fetch one past session's bars file, unless it already exists.

    The existence check is a parquet read, so it runs here and not on the Qt
    thread. `present` says the file was already there.
    """

    present = Signal(str)

    def __init__(self, callback, session: str, parent=None) -> None:
        super().__init__(parent)
        self.callback, self.value = callback, str(session)

    def run(self) -> None:  # pragma: no cover - asserted through its seams
        try:
            import day_review_bars

            if day_review_bars.read_session_bars(self.value) is not None:
                self.present.emit(self.value)
                return
        except Exception:  # noqa: BLE001 - an unreadable file is fetched again
            logging.debug("Day Review bars check failed.", exc_info=True)
        try:
            self.callback(self.value, lookback_sessions=LOOKBACK_SESSIONS)
        except Exception:  # noqa: BLE001
            logging.info("Day Review bars backfill failed.", exc_info=True)


class _RedoPackWorker(QThread):
    """One day pack, built off the Qt thread before a Redo is queued.

    The night can only narrate a session it HAS a pack for, and a marker for a
    session with no pack is a request nobody can answer (reviewer round 2,
    2026-09-20). The build is deterministic and calls no model - it is the same
    seam the post-close tick uses - but it streams the session's stores, so it
    may never run on the Qt thread.
    """

    done = Signal(str, bool)

    def __init__(self, builder, session_date: str, parent=None) -> None:
        super().__init__(parent)
        self._builder = builder
        self._session = str(session_date)

    def run(self) -> None:  # pragma: no cover - exercised through its signal
        built = False
        try:
            built = bool(self._builder(self._session))
        except Exception:  # noqa: BLE001 - a failed build is an answer, not a crash
            logging.debug("The Redo's day pack could not be built.", exc_info=True)
            built = False
        self.done.emit(self._session, built)


class DayReviewPanel(QFrame):
    """The day, read back: what happened, what you said, what you traded."""

    statusChanged = Signal(str)
    #: (symbol, side). The host charts it through `show_board_symbol`.
    chartRequested = Signal(str, str)
    openTradeRequested = Signal(str)
    #: True while a closed day's "Review my day" walk is waiting. The host badges its nav entry.
    walkReadyChanged = Signal(bool)

    def __init__(
        self,
        service=None,
        parent=None,
        *,
        clock: Callable[[], datetime] | None = None,
        auto_time_reader: Callable[[], Any] | None = None,
        redo_launcher: Callable[[str], Any] | None = None,
        remember_session: bool | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        #: Keep the shown session in local settings. Defaults on for the desk's
        #: own page (no injected service) and off for a host with its own.
        self._remember_session = (service is None) if remember_session is None else bool(
            remember_session
        )
        if service is None:
            from ui.services.day_review_service import DayReviewService

            service = DayReviewService()
        self.service = service
        self._worker: _DayReadWorker | None = None
        #: A date chosen while the one reader is still busy.  The reader is
        #: never terminated: its stale payload is ignored, then this request
        #: starts after it has finished.
        self._pending_day_read: tuple[str, bool] | None = None
        #: The A.I. Summary link reads the completed day's existing facts.  It
        #: must not turn a missing tape into a recovery write.
        self._next_read_skips_backfill = False
        self._index_worker: _IndexBuildWorker | None = None
        self._bars_worker: _IndexBuildWorker | None = None
        self._bars_backfill_queue: list[str] = []
        self._bars_backfill_queued: set[str] = set()
        #: Sessions a worker found already on disk; never checked again.
        self._bars_known_present: set[str] = set()
        self._building_index = ""
        #: The desk's own M5 cache accessor (`alert_center.journal_chart_bars`).
        #: Called ONLY on the Qt thread, by `reload`, and only for a session that
        #: has not closed - see `_DayReadWorker` for what a worker call cost.
        self._bars_reader: Callable[[str], Any] | None = None
        self._payload: dict[str, Any] = {}
        self._entries: list[dict[str, Any]] = []
        #: TJ-10: `entry_id -> the read row the worker graded`. The page reads
        #: this and computes no verdict of its own.
        self._reads: dict[str, dict[str, Any]] = {}
        self._walkaway_rows: tuple[Any, ...] = ()
        self._forecast_expanded = False
        self._loaded_once = False
        self._chart: Any = None
        #: The ONE name pane and what it is showing (TJ-3). "" means nothing.
        self._name_chart: Any = None
        self._name_chart_symbol = ""
        self._name_charts: dict[str, Any] = {}
        self._clock: Callable[[], datetime] = clock or datetime.now
        #: The one display zone, resolved once per render (item: one clock).
        self._zone: Any = _display_zone()
        self._auto_time_reader: Callable[[], Any] = (
            auto_time_reader or daily_recap_schedule.auto_time_from_settings
        )
        #: How a Redo actually starts (TJ-4 change 4). Injected so a test can
        #: assert the click without spawning a process.
        self._redo_launcher: Callable[[str], Any] = redo_launcher or launch_redo_process
        #: The Redo's own pack builds, held while they run (a QThread nobody
        #: holds is collected mid-run). ONE at a time; short-lived.
        self._redo_workers: list[_RedoPackWorker] = []
        #: Is a redo already in flight? Three clicks used to start three full
        #: `read_day` builds racing on one `pack.json` and three child
        #: processes (reviewer round 3, 2026-09-20).
        self._redo_busy = False
        self._auto_fired_session: str | None = None
        self._auto_post_close_session: str | None = None
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(AUTO_POLL_INTERVAL_MS)
        self._auto_timer.timeout.connect(self._on_auto_tick)
        #: "Review my day": the open walk, walks kept alive while a save runs, and the banner state.
        self._walk: Any = None
        self._retired_walks: list[Any] = []
        self._walk_ready = False
        self._walk_banner_dismissed: set[str] = set()
        #: R1: the open Day Review Show overlay, or None.
        self._show: Any = None
        #: The saved-clue layers on the name and trade charts (`clue_marker.ClueFlow`).
        self._name_clue_flow: Any = None
        self._trade_clue_flow: Any = None

        self._build_header()
        self._build_glance()
        self._build_report_card()
        self._build_story()
        self._build_walkaway()
        self._build_said()
        self._build_traded()
        self._build_chart_section()
        self._build_ideas()
        # P1-7: the trader's plan, read-only, read on its own worker.
        from ui.widgets.trading_plan_view import TradingPlanView

        self.plan_view = TradingPlanView(self)
        self._build_layout()
        self.refresh_reader_measure()

    # -- construction ------------------------------------------------------
    def _build_header(self) -> None:
        from PySide6.QtGui import QKeySequence, QShortcut

        self.heading = QLabel("Day Review")
        self.heading.setObjectName("SectionTitle")
        self.subtitle = QLabel(
            "One day: what happened, what you passed on, what you said and what "
            "you traded. Best available movement - never money earned."
        )
        self.subtitle.setObjectName("SectionSubtitle")
        self.subtitle.setWordWrap(True)
        #: Every time on the page is in this one zone; it is named once, here.
        self.zone_note = QLabel(f"Times in {_zone_label(self._zone)}")
        self.zone_note.setObjectName("SectionSubtitle")

        self.session_picker = QComboBox()
        saved = self._saved_session() if self._remember_session else ""
        self._fill_session_picker(select=saved or None)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.reload)
        self.session_picker.currentIndexChanged.connect(lambda _index: self.reload())
        # Day navigation: older on the left, newer on the right.
        self.prev_session_button = QPushButton("◀")
        self.prev_session_button.setToolTip("Previous session (Alt+Left)")
        self.prev_session_button.clicked.connect(lambda: self.step_session(-1))
        self.next_session_button = QPushButton("▶")
        self.next_session_button.setToolTip("Next session (Alt+Right)")
        self.next_session_button.clicked.connect(lambda: self.step_session(1))
        self.session_shortcuts: dict[str, QShortcut] = {}
        for keys, step in (("Alt+Left", -1), ("Alt+Right", 1)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda s=step: self.step_session(s))
            self.session_shortcuts[keys] = shortcut
        self._sync_step_buttons()

        self.provisional_note = QLabel("")
        self.provisional_note.setObjectName("SectionSubtitle")
        self.provisional_note.setWordWrap(True)
        self.status = QLabel("")
        self.status.setObjectName("SectionSubtitle")
        self.status.setWordWrap(True)

    # -- day navigation ----------------------------------------------------
    def step_session(self, step: int) -> None:
        """Move one session older (-1) or newer (+1). The picker is newest first."""
        if self.walk_is_open():
            return
        index = self.session_picker.currentIndex() - int(step)
        if 0 <= index < self.session_picker.count():
            self.session_picker.setCurrentIndex(index)

    def _sync_step_buttons(self) -> None:
        index = self.session_picker.currentIndex()
        count = self.session_picker.count()
        self.prev_session_button.setEnabled(0 <= index < count - 1)
        self.next_session_button.setEnabled(index > 0)

    @staticmethod
    def _saved_session() -> str:
        try:
            from project_paths import get_local_setting

            return str(get_local_setting(SESSION_SETTING_KEY, "") or "")[:10]
        except Exception:  # noqa: BLE001 - a setting never costs the page
            return ""

    def _remember(self, session: str) -> None:
        """Keep the shown session in local settings, written only on a change."""
        if not self._remember_session or not session:
            return
        try:
            from project_paths import get_local_setting, save_local_setting

            if get_local_setting(SESSION_SETTING_KEY, "") != session:
                save_local_setting(SESSION_SETTING_KEY, session)
        except Exception:  # noqa: BLE001
            logging.debug("The Day Review session was not remembered.", exc_info=True)

    # -- the glance strip --------------------------------------------------
    def _build_glance(self) -> None:
        """The numbers row at the top. Each tile has a tooltip and a click."""
        from ui.widgets.day_glance_strip import DayGlanceStrip

        self.glance_strip = DayGlanceStrip()
        self.glance_strip.tileClicked.connect(self._open_glance_target)
        self.details_toggle = QPushButton("Details ▸")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setFlat(True)
        self.details_toggle.setToolTip("Show the full report card sentences.")
        self.details_toggle.toggled.connect(self._toggle_details)
        self.walk_button = QPushButton("Review my day (5 min)")
        self.walk_button.setToolTip("A short guided walk through this day, one card at a time.")
        self.walk_button.clicked.connect(self.open_walk)
        self.walk_button.setEnabled(False)
        # R1: the Day Review Show - the day as slides over the whole window.
        self.show_button = QPushButton("Show")
        self.show_button.setToolTip(
            "Your day as slides. Right/Space next, Left back, A auto, Esc close."
        )
        self.show_button.clicked.connect(self.open_show)
        self.show_button.setEnabled(False)
        # The closed-day nudge: a small line on the page, never a popup.
        self.walk_banner = QFrame()
        self.walk_banner.setObjectName("DayReviewWalkBanner")
        banner_row = QHBoxLayout(self.walk_banner)
        banner_row.setContentsMargins(8, 4, 8, 4)
        self.walk_banner_label = QLabel("Your day is ready —")
        self.walk_banner_button = QPushButton("Review (5 min)")
        self.walk_banner_button.clicked.connect(self.open_walk)
        self.walk_banner_close = QPushButton("×")
        self.walk_banner_close.setFlat(True)
        self.walk_banner_close.setToolTip("Hide this for now.")
        self.walk_banner_close.clicked.connect(self._dismiss_walk_banner)
        banner_row.addWidget(self.walk_banner_label)
        banner_row.addWidget(self.walk_banner_button)
        banner_row.addStretch(1)
        banner_row.addWidget(self.walk_banner_close)
        self.walk_banner.setVisible(False)

    def _toggle_details(self, shown: bool) -> None:
        self.report_card_section.setVisible(bool(shown))
        self.details_toggle.setText("Details ▾" if shown else "Details ▸")

    def _open_glance_target(self, key: str) -> None:
        """A tile click scrolls to the section its number came from."""
        glance = dict(self._payload.get("glance") or {})
        if key == "biggest_miss" and isinstance(glance.get("biggest_miss"), Mapping):
            miss = glance["biggest_miss"]
            self.select_miss_population(str(miss.get("population") or ""))
            symbol = str(miss.get("symbol") or "")
            for index, row in enumerate(self._walkaway_rows):
                if str(getattr(row, "symbol", "") or "").upper() == symbol.upper():
                    self.miss_table.setCurrentCell(index, 1)
                    break
            self.reveal_card_target("miss")
            return
        if key == "biggest_win" and isinstance(glance.get("biggest_win"), Mapping):
            wanted = str(glance["biggest_win"].get("trade_id") or "")
            ids = getattr(self, "_trade_ids", ())
            if wanted in ids:
                self.trades_table.selectRow(ids.index(wanted))
        target = GLANCE_TARGETS.get(str(key), "")
        if target:
            self.reveal_card_target(target)

    def _build_report_card(self) -> None:
        """The six lines, built ONCE. A render only sets their text (TJ-12).

        Six fixed buttons rather than a rebuilt list: the page diffs and never
        rebuilds, and a card that re-created its widgets on every read would put
        six layout passes on the Qt thread for six sentences.
        """
        import day_report_card

        self.report_card_section = QFrame()
        self.report_card_section.setObjectName(REPORT_CARD_OBJECT_NAME)
        body = QVBoxLayout(self.report_card_section)
        body.setContentsMargins(10, 8, 10, 8)
        body.setSpacing(2)
        title = QLabel(REPORT_CARD_TITLE)
        title.setObjectName("SectionTitle")
        body.addWidget(title)
        #: key -> the clickable line. Ordered by `LINE_KEYS`, which is the
        #: packet's order and the order the payload arrives in.
        self._report_card_lines: dict[str, QPushButton] = {}
        for key in day_report_card.LINE_KEYS:
            button = QPushButton(REPORT_CARD_PLACEHOLDERS.get(key, key))
            button.setObjectName(REPORT_CARD_LINE_OBJECT_NAME)
            button.setFlat(True)
            button.setProperty("cardLine", key)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            button.setCursor(Qt.PointingHandCursor)
            # The target is read from `LINE_TARGETS` at CLICK time, not bound
            # here: the card owns what a line opens, and a second copy of that
            # map on the page would be a second opinion about it.
            button.clicked.connect(lambda _checked=False, name=key: self._open_card_target(name))
            body.addWidget(button)
            self._report_card_lines[key] = button
        # P8-P5: plain truth lines over the last 20 sessions, built on the worker.
        self.truth_note = QLabel("")
        self.truth_note.setObjectName("TruthNote")
        self.truth_note.setWordWrap(True)
        self.truth_note.setTextFormat(Qt.PlainText)
        body.addWidget(self.truth_note)
        # The full sentences live under "Details"; the glance strip leads.
        self.report_card_section.setVisible(False)

    @property
    def report_card_lines(self) -> tuple[QPushButton, ...]:
        """The six line widgets, in `day_report_card.LINE_KEYS` order."""
        import day_report_card

        return tuple(
            self._report_card_lines[key]
            for key in day_report_card.LINE_KEYS
            if key in self._report_card_lines
        )

    def _open_card_target(self, key: str) -> None:
        import day_report_card

        target = day_report_card.LINE_TARGETS.get(str(key), "")
        if target:
            self.reveal_card_target(target)

    def reveal_card_target(self, target: str):
        """Scroll to what a card line is ABOUT, and hand the widget back.

        Every `LINE_TARGETS` value resolves to a widget already on this page -
        a line that opened nothing would be a promise the card could not keep.
        """
        name = str(target or "")
        widget = None
        if name in self.miss_chips:
            # A walk-away target is a population of the one miss table.
            self.select_miss_population(name)
            widget = self.miss_section
        if widget is None:
            widget = {
                "miss": getattr(self, "miss_section", None),
                "said": getattr(self, "said_section", None),
                "congruence": self.congruence_note,
                "trades": getattr(self, "traded_section", None),
                "status": self.status,
                "story": getattr(self, "story_section", None),
            }.get(name)
        if widget is None:
            return None
        try:
            self.scroll.ensureWidgetVisible(widget)
        except Exception:  # noqa: BLE001 - a scroll never costs the page
            logging.debug("The card target could not be scrolled to.", exc_info=True)
        return widget

    @staticmethod
    def _wrapped_card_text(text: str) -> str:
        """One line, broken on WORDS so a `QPushButton` can show all of it."""
        words = str(text or "").split()
        if not words:
            return ""
        rows: list[str] = []
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) > REPORT_CARD_WRAP_CHARS:
                rows.append(current)
                current = word
            else:
                current = f"{current} {word}"
        rows.append(current)
        return "\n".join(rows)

    def _render_report_card(self, card: Any) -> None:
        """Set six texts. It builds nothing: the WORKER built this card."""
        lines = {}
        rows = card.get("lines") if isinstance(card, Mapping) else getattr(card, "lines", None)
        for line in rows or ():
            if isinstance(line, Mapping) and line.get("key"):
                lines[str(line["key"])] = line
        import day_report_card

        for key, button in self._report_card_lines.items():
            row = lines.get(key)
            text = day_report_card.plain_words((row or {}).get("text"))
            if not text:
                text = REPORT_CARD_PLACEHOLDERS.get(key, key)
            button.setText(self._wrapped_card_text(text))
            button.setToolTip(text)

    def _build_story(self) -> None:
        self.story_note = QLabel(NO_STORY_YET)
        self.story_note.setObjectName("SectionSubtitle")
        self.story_note.setWordWrap(True)
        self.story_warning = QLabel("")
        self.story_warning.setObjectName("SectionSubtitle")
        self.story_warning.setWordWrap(True)
        self.story_warning.setVisible(False)
        # TJ-4: the night's story, under its own headline. The page FORMATS it -
        # every verdict on it was measured by TJ-10's grader and copied by the
        # night; nothing here grades anything.
        self.story_body = QLabel("")
        self.story_body.setWordWrap(True)
        self.story_body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.story_body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.redo_story_button = QPushButton("Redo story")
        self.redo_story_button.setToolTip(
            "Ask the desk to write this day's story again. By day it is queued "
            "for tonight: local inference is night-only."
        )
        self.redo_story_button.clicked.connect(self.redo_story)
        # TJ-4 item 3: the rolling D1 view, above the open theses it is about.
        self.d1_view_note = QLabel("")
        self.d1_view_note.setObjectName("SectionSubtitle")
        self.d1_view_note.setWordWrap(True)
        self.d1_view_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.story_facts = QLabel("")
        self.story_facts.setWordWrap(True)
        self.story_facts.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # A floor, never a ceiling: TJ-4's story is paragraphs and a wrapped
        # QLabel grows with them. The floor is what stops an empty story from
        # reading as a section that failed.
        self.story_facts.setMinimumHeight(theme.px(STORY_MIN_HEIGHT_PX))
        self.story_facts.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # TJ-10: the three congruence lines, UNDER the story. Printed, never
        # pushed and never acted on (decision 0021 answer 15) - this label is
        # the whole surface. The worker builds the sentences; this shows them.
        self.congruence_note = QLabel("")
        self.congruence_note.setObjectName("CongruenceLines")
        self.congruence_note.setWordWrap(True)
        self.congruence_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # Read-only: the drafting pane and "Save interpretation" are gone from
        # the trader's screen (plan.md §12.2). The sidecar still holds them.
        self.theses = QListWidget()
        self.theses.setMinimumHeight(theme.px(70))
        self.theses.setMaximumHeight(theme.px(150))

    def _build_walkaway(self) -> None:
        """ONE miss table, five filter chips, and a "More columns" toggle.

        The 16-column model stays (`TJ2B_WALKAWAY_COLUMNS`); only the default
        columns show until the toggle is on. A single click (or J/K) draws the
        row's name beside the table; a double click also asks the host.
        """
        from PySide6.QtWidgets import QAbstractScrollArea, QButtonGroup

        self.miss_table = QTableWidget(0, len(TJ2B_WALKAWAY_COLUMNS))
        self.miss_table.setObjectName("DayReviewMissTable")
        self.miss_table.setHorizontalHeaderLabels(TJ2B_WALKAWAY_COLUMNS)
        self.miss_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.miss_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.miss_table.setSelectionMode(QTableWidget.SingleSelection)
        # Its width follows its visible columns, so the chart beside it gets
        # whatever is left.
        self.miss_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.miss_table.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.miss_table.setFixedHeight(theme.px(MISS_TABLE_HEIGHT_PX))
        self.miss_table.itemDoubleClicked.connect(self._activate_walkaway)
        self.miss_table.itemActivated.connect(self._activate_walkaway)
        self.miss_table.currentCellChanged.connect(self._on_miss_row_changed)
        self.miss_table.installEventFilter(self)
        _fill_the_width(self.miss_table)

        self.miss_chips: dict[str, QPushButton] = {}
        self._miss_chip_group = QButtonGroup(self)
        self._miss_chip_group.setExclusive(True)
        for name, label in MISS_FILTERS:
            chip = QPushButton(f"{_button_text(label)} (0)")
            chip.setObjectName("MissFilterChip")
            chip.setCheckable(True)
            chip.setProperty("population", name)
            chip.setToolTip(dict(TJ2B_WALKAWAY_TITLES).get(name, label))
            chip.clicked.connect(lambda _checked=False, n=name: self.select_miss_population(n))
            self._miss_chip_group.addButton(chip)
            self.miss_chips[name] = chip
        self.more_columns_toggle = QPushButton("More columns")
        self.more_columns_toggle.setCheckable(True)
        self.more_columns_toggle.setToolTip(
            "Show every measured column: the ATR moves, held at close, traded, "
            "what you made, left on the table and state."
        )
        self.more_columns_toggle.toggled.connect(lambda _on: self._apply_miss_columns())

        self.walkaway_note = QLabel(LOADING_NOTE)
        self.walkaway_note.setObjectName("SectionSubtitle")
        self.walkaway_note.setWordWrap(True)
        #: The selected population's one sentence (TJ-11 item 5).
        self.walkaway_sentence = QLabel("")
        self.walkaway_sentence.setObjectName("SectionSubtitle")
        self.walkaway_sentence.setWordWrap(True)
        #: The base rate the table is read against (TJ-11 item 4).
        self.walkaway_skill = QLabel("")
        self.walkaway_skill.setObjectName("SectionSubtitle")
        self.walkaway_skill.setWordWrap(True)

        self._miss_rows: dict[str, tuple[Any, ...]] = {name: () for name, _ in MISS_FILTERS}
        self._miss_sentences: dict[str, str] = {}
        self._miss_population = MISS_FILTERS[0][0]
        self.miss_chips[self._miss_population].setChecked(True)
        self._apply_miss_columns()

    def miss_population(self) -> str:
        """Which filter chip is on."""
        return self._miss_population

    def miss_counts(self) -> dict[str, int]:
        """Rows per population, as the chips show them."""
        return {name: len(self._miss_rows.get(name, ())) for name, _ in MISS_FILTERS}

    def miss_table_for(self, name: str) -> QTableWidget:
        """Turn on `name`'s chip and hand back the one table, now showing it."""
        self.select_miss_population(name)
        return self.miss_table

    def select_miss_population(self, name: str) -> None:
        """Show one population in the miss table. Formatting only."""
        name = str(name or "")
        if name not in self.miss_chips:
            return
        self._miss_population = name
        chip = self.miss_chips[name]
        if not chip.isChecked():
            chip.setChecked(True)
        self._show_miss_population()

    def _apply_miss_columns(self) -> None:
        """Hide the columns outside the default set unless "More columns" is on."""
        more = self.more_columns_toggle.isChecked()
        wanted = MISS_DEFAULT_COLUMNS_BY_FILTER.get(
            self._miss_population, MISS_DEFAULT_COLUMNS
        )
        for index, header in enumerate(TJ2B_WALKAWAY_COLUMNS):
            self.miss_table.setColumnHidden(index, not (more or header in wanted))
        self.more_columns_toggle.setText("Fewer columns" if more else "More columns")

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
        # TJ-10: the selected read's MEASURED verdict. Styled in `theme.qss` by
        # object name and a `verdict` dynamic property - never a widget
        # stylesheet, which is expensive on the Qt thread.
        self.verdict_chip = QLabel("")
        self.verdict_chip.setObjectName("VerdictChip")
        self.verdict_chip.setVisible(False)

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

        # TJ-7 change 2: the optional two-click strip beside the composer. The
        # SAME widget the Mentor's `day_close` question draws, so the cap, the
        # codes and the "nothing pre-selected" rule have one owner. It never
        # gates Save - a note with no mood is the note the trader always wrote.
        from ui.widgets.mood_strip import MoodStrip

        self.mood_strip = MoodStrip()

        # TJ-7 / live gate #151: ONE line saying what the trader recorded about
        # themselves this session. It is READ off the payload the worker already
        # builds - this page calls no builder and no model for it.
        self.mood_line = QLabel("")
        self.mood_line.setObjectName("SectionSubtitle")
        self.mood_line.setWordWrap(True)

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
        self.trades_table.itemSelectionChanged.connect(self._show_selected_trade)
        self.trades_table.cellDoubleClicked.connect(self._open_selected_trade)
        self.trades_table.installEventFilter(self)
        self.trade_detail = QPlainTextEdit()
        self.trade_detail.setReadOnly(True)
        self.trade_detail.setMaximumHeight(theme.px(175))
        self.trade_detail.setPlaceholderText("Select a trade to see your entry and exit notes.")
        # The selected trade's session chart, with its entry and exit marks.
        # Same widget class and the same worker-built `name_charts` payload as
        # the miss table's name pane; built on first need, then re-fed.
        self._trade_chart: Any = None
        self._trade_chart_symbol = ""
        self.trade_chart_note = QLabel(TRADE_CHART_IDLE_NOTE)
        self.trade_chart_note.setObjectName("SectionSubtitle")
        self.trade_chart_note.setWordWrap(True)
        self._trade_chart_holder = QWidget()
        self._trade_chart_holder.setMinimumWidth(theme.px(NAME_CHART_MIN_WIDTH_PX))
        self._trade_chart_holder.setMaximumHeight(theme.px(MISS_TABLE_HEIGHT_PX + 60))
        holder = QVBoxLayout(self._trade_chart_holder)
        holder.setContentsMargins(0, 0, 0, 0)
        holder.addWidget(self.trade_chart_note)
        self._trade_chart_layout = QVBoxLayout()
        self._trade_chart_layout.setContentsMargins(0, 0, 0, 0)
        holder.addLayout(self._trade_chart_layout, 1)
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

        # The ONE name pane (TJ-3 item 3). One `CandleChart` for every row of
        # every walk-away table: the old Market Journal page built four charts on
        # the first click and cost 299 ms, and a page that builds a widget per
        # name would be that mistake with more names. Built on the first row
        # click, then re-fed - never rebuilt.
        self.name_chart_heading = QLabel("That name, this session")
        self.name_chart_heading.setObjectName("SectionTitle")
        self.name_chart_note = QLabel(NAME_CHART_IDLE_NOTE)
        self.name_chart_note.setObjectName("SectionSubtitle")
        self.name_chart_note.setWordWrap(True)
        self._name_chart_holder = QWidget()
        self._name_chart_holder.setMinimumWidth(theme.px(NAME_CHART_MIN_WIDTH_PX))
        self._name_chart_holder.setFixedHeight(theme.px(MISS_TABLE_HEIGHT_PX))
        self._name_chart_holder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        body = QVBoxLayout(self._name_chart_holder)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(4)
        body.addWidget(self.name_chart_heading)
        body.addWidget(self.name_chart_note)
        self._name_chart_layout = QVBoxLayout()
        self._name_chart_layout.setContentsMargins(0, 0, 0, 0)
        body.addLayout(self._name_chart_layout, 1)

    def _build_ideas(self) -> None:
        """The note and the card. Exactly ONE of them is ever on screen.

        The card opens no store: its rows arrive in the same payload as
        everything else on this page (TJ-6), and its Keep / Dismiss write runs
        on its own worker.
        """
        from ui.widgets.ideas_card import IdeasCard

        self.ideas_note = QLabel(NO_IDEAS_YET)
        self.ideas_note.setObjectName("SectionSubtitle")
        self.ideas_note.setWordWrap(True)
        self.ideas_card = IdeasCard(self)
        self.ideas_card.setVisible(False)

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
        verbs = QHBoxLayout()
        verbs.setContentsMargins(0, 0, 0, 0)
        verbs.addWidget(self.redo_story_button)
        verbs.addStretch(1)
        self.story_section = self._section(
            "What happened",
            self.story_note,
            self.story_warning,
            self.story_body,
            self.story_facts,
            self.congruence_note,
            verbs,
        )
        theses_label = QLabel("Open theses")
        theses_label.setObjectName("SectionSubtitle")
        self.theses_section = self._section(
            "", theses_label, self.d1_view_note, self.theses
        )
        self.spy_section = self._section(
            "SPY, this session", self.spy_note, self._chart_holder, stretch_last=True
        )

        # B11: the night digest's narration, read on the day worker.
        self.digest_note = QLabel("Night digest: not read yet.")
        self.digest_note.setWordWrap(True)
        self.digest_note.setTextFormat(Qt.PlainText)
        self.digest_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.digest_section = self._section("Night digest", self.digest_note)

        column = QWidget()
        body = QVBoxLayout(column)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addWidget(self.story_section)
        body.addWidget(self.digest_section)
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
        reader_body.addWidget(self.verdict_chip)
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

        self.calls_table = QTableWidget(0, 5)
        self.calls_table.setHorizontalHeaderLabels(
            ["When", "Horizon", "Direction", "Confidence", "Result"]
        )
        self.calls_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.calls_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.calls_table.setMinimumHeight(theme.px(110))
        self.calls_table.setMaximumHeight(theme.px(190))
        _fill_the_width(self.calls_table)
        self.calls_table.cellDoubleClicked.connect(self._open_call_row)
        self.calls_table.installEventFilter(self)
        self.said_section = self._section(
            "Market calls and notes",
            QLabel("Market calls · double-click to see your exact words"),
            self.calls_table,
            self.said_split,
            self.mood_line,
            QLabel("New entry"),
            self.entry_text,
            self.mood_strip,
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
        """Picks and passes: the chips, then ONE table with its name chart beside.

        The skill line is the base rate the table is read against, so it sits
        above the chips: a miss count without a base rate is the number the
        trader would misread. The chart takes whatever width the table leaves.
        """
        chips = QHBoxLayout()
        chips.setContentsMargins(0, 0, 0, 0)
        chips.setSpacing(6)
        for chip in self.miss_chips.values():
            chips.addWidget(chip)
        chips.addStretch(1)
        chips.addWidget(self.more_columns_toggle)

        left = QWidget()
        left.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        left_body = QVBoxLayout(left)
        left_body.setContentsMargins(0, 0, 0, 0)
        left_body.setSpacing(4)
        left_body.addWidget(self.walkaway_note)
        left_body.addWidget(self.walkaway_sentence)
        left_body.addWidget(self.miss_table)

        beside = QWidget()
        side_by_side = QHBoxLayout(beside)
        side_by_side.setContentsMargins(0, 0, 0, 0)
        side_by_side.setSpacing(12)
        side_by_side.addWidget(left, 0, Qt.AlignTop)
        side_by_side.addWidget(self._name_chart_holder, 1, Qt.AlignBottom)

        self.miss_section = QWidget()
        body = QVBoxLayout(self.miss_section)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(6)
        picks_heading = QLabel("Picks and passes")
        picks_heading.setObjectName("SectionTitle")
        body.addWidget(picks_heading)
        picks_note = QLabel(
            "A pass that did not run is shown too. A later run does not by itself mean the pass was poor."
        )
        picks_note.setObjectName("SectionSubtitle")
        picks_note.setWordWrap(True)
        body.addWidget(picks_note)
        body.addWidget(self.walkaway_skill)
        body.addLayout(chips)
        body.addWidget(beside)
        return self.miss_section

    def _bottom_row(self) -> QWidget:
        """What you traded, with the selected trade's chart, beside the ideas."""
        trade_left = QWidget()
        left_body = QVBoxLayout(trade_left)
        left_body.setContentsMargins(0, 0, 0, 0)
        left_body.addWidget(self.trades_table)
        left_body.addWidget(self.trade_detail)
        trade_row = QHBoxLayout()
        trade_row.setContentsMargins(0, 0, 0, 0)
        trade_row.setSpacing(12)
        trade_row.addWidget(trade_left, 1)
        trade_row.addWidget(self._trade_chart_holder, 1)
        self.traded_section = self._section("Entries and exits", self.trades_note, trade_row)
        self.ideas_section = self._section(
            "Ideas from the desk's AI", self.ideas_note, self.ideas_card
        )
        self.bottom_row = QWidget()
        row = QHBoxLayout(self.bottom_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        row.addWidget(self.traded_section, 3)
        row.addWidget(self.ideas_section, 1)
        return self.bottom_row

    def _build_layout(self) -> None:
        header = QHBoxLayout()
        header.addWidget(QLabel("Session"))
        header.addWidget(self.prev_session_button)
        header.addWidget(self.session_picker, 1)
        header.addWidget(self.next_session_button)
        header.addWidget(self.refresh_button)
        header.addWidget(self.zone_note)

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

        # S5: longs and shorts vs SPY over the last 20 sessions. Display only.
        self.tape_side_note = QLabel("")
        self.tape_side_note.setObjectName("MutedLabel")
        self.tape_side_note.setWordWrap(True)
        self.tape_side_note.setTextFormat(Qt.PlainText)

        glance_row = QHBoxLayout()
        glance_row.setContentsMargins(0, 0, 0, 0)
        glance_row.addWidget(self.glance_strip, 1)
        glance_row.addWidget(self.walk_button, 0, Qt.AlignBottom)
        glance_row.addWidget(self.show_button, 0, Qt.AlignBottom)
        glance_row.addWidget(self.details_toggle, 0, Qt.AlignBottom)

        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addWidget(self.heading)
        body.addWidget(self.subtitle)
        body.addLayout(header)
        body.addWidget(self.provisional_note)
        body.addWidget(self.walk_banner)
        # The glance strip heads the page; the full report card sits under
        # "Details", still above the two columns that hold the story.
        body.addLayout(glance_row)
        body.addWidget(self.tape_side_note)
        body.addWidget(self.report_card_section)
        body.addWidget(self.columns, 1)
        body.addWidget(self._walkaway_row())
        body.addWidget(self._bottom_row())
        body.addWidget(self.plan_view)
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
        # The walk replaces the page body while it is open.
        self.body_stack = QStackedWidget()
        self.body_stack.addWidget(self.scroll)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.addWidget(self.body_stack)

    def refresh_reader_measure(self) -> None:
        """The reader and the forecast span the RIGHT column (TJ-1L).

        The G3 rule capped both at a 100-character measure, which was right
        when they sat across a page WIDE enough to need one. In the 45% column
        the cap is what puts the empty space back: measured at 3800x2000 both
        boxes stopped at about 420 px of an 890 px column while `New entry`
        under them ran the full width. The COLUMN is the measure now - the
        trader drags it, and the splitter is the control.

        The seam keeps its name and its caller: `MainWindow._apply_scaled_metrics`
        calls it on a scale change, and both widgets must come back spanning.
        """
        try:
            for widget in (self.entry_reader, self.forecast_box):
                widget.ensurePolished()
                # `QWIDGETSIZE_MAX`, which PySide6 does not export: the value
                # Qt uses for "no maximum", and what `setMaximumWidth` must be
                # given to UNDO an earlier cap.
                widget.setMaximumWidth(16_777_215)
                widget.setSizePolicy(
                    QSizePolicy.Policy.Expanding, widget.sizePolicy().verticalPolicy()
                )
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

    def show_latest_completed_session(self) -> bool:
        """Select the newest closed exchange session and read it once.

        This is the explicit route from A.I. Summary.  It asks the same
        calendar owner as the picker and refuses to replace the trader's
        historical selection when the calendar cannot answer.
        """
        import market_calendar

        try:
            session = market_calendar.last_completed_session(self._clock()).isoformat()
        except Exception:  # noqa: BLE001 - calendar uncertainty must not guess a session
            self.status.setText(
                "The latest completed session is uncertain. Day Review stayed where it was."
            )
            self.statusChanged.emit(self.status.text())
            return False
        self._next_read_skips_backfill = True
        self.show_session(session)
        return True

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

    def _backfill_bars_for(self, session_date: str) -> None:
        """Queue past-session recovery; one off-Qt worker drains FIFO."""
        import day_review_bars

        session = str(session_date or "")[:10]
        if not session or not day_review_bars.session_is_backfillable(session, now=self._clock()):
            return
        # Whether the file already exists is asked ON the worker: it is a
        # parquet read, and this runs on the Qt thread.
        if session in self._bars_known_present:
            return
        if session not in self._bars_backfill_queued:
            self._bars_backfill_queue.append(session)
            self._bars_backfill_queued.add(session)
        self._start_next_bars_backfill()

    def _start_next_bars_backfill(self) -> None:
        if self._bars_worker is not None and self._bars_worker.isRunning():
            return
        if not self._bars_backfill_queue:
            return
        session = self._bars_backfill_queue.pop(0)
        method = getattr(self.service, "backfill_session_bars_for", None)
        if not callable(method):
            self._bars_backfill_queued.discard(session)
            self._start_next_bars_backfill()
            return
        self._bars_worker = _BarsBackfillWorker(method, session, self)
        self._bars_worker.present.connect(lambda done: self._bars_known_present.add(str(done)))
        self._bars_worker.finished.connect(lambda: self._on_bars_backfill_finished(session))
        self.status.setText(FETCHING_BARS_NOTE.format(session=session))
        self.statusChanged.emit(self.status.text())
        self._bars_worker.start()

    def _on_bars_backfill_finished(self, session: str) -> None:
        self._bars_backfill_queued.discard(session)
        if self.status.text() == FETCHING_BARS_NOTE.format(session=session):
            self.status.setText(f"Day Review: {self.session_date()}")
            self.statusChanged.emit(self.status.text())
        self._start_next_bars_backfill()

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
        self._refresh_session_picker()
        self._sync_after_the_fact()
        session = self.session_date()
        self._sync_step_buttons()
        self._remember(session)
        backfill_bars = not self._next_read_skips_backfill
        self._next_read_skips_backfill = False
        self._request_day_read(session, backfill_bars=backfill_bars)
        self.plan_view.refresh()

    def _request_day_read(self, session: str, *, backfill_bars: bool) -> None:
        """Run one existing reader, or queue its replacement after it finishes."""
        if self._worker is not None and self._worker.isRunning():
            if str(getattr(self._worker, "_session", "")) != str(session):
                self._pending_day_read = (str(session), bool(backfill_bars))
            else:
                # The trader returned to the day this one reader already has.
                # A previously queued different date is obsolete; leaving it
                # here would read and then discard a day nobody selected.
                self._pending_day_read = None
            return
        self.status.setText(LOADING_NOTE)
        if backfill_bars:
            self._backfill_bars_for(session)
        self._worker = _DayReadWorker(
            self.service, session, self, spy_m5_bars=self._spy_bars_for(session)
        )
        worker = self._worker
        worker.loaded.connect(
            lambda payload, expected=str(session), allow_backfill=backfill_bars:
            self._render_loaded_day(expected, payload, allow_backfill=allow_backfill)
        )
        worker.failed.connect(
            lambda reason, expected=str(session): self._render_failed_day(expected, reason)
        )
        worker.finished.connect(lambda done=worker: self._on_day_read_finished(done))
        worker.start()

    def _render_loaded_day(
        self, expected: str, payload: Mapping[str, Any], *, allow_backfill: bool
    ) -> None:
        """Ignore a completed stale read after the picker moved to another day."""
        if self.session_date() != expected:
            return
        if str((payload or {}).get("session_date") or "") != expected:
            return
        self.render(payload, allow_backfill=allow_backfill)

    def _render_failed_day(self, expected: str, reason: str) -> None:
        if self.session_date() == expected:
            self._render_failure(reason)

    def _on_day_read_finished(self, worker: _DayReadWorker) -> None:
        if worker is not self._worker:
            return
        self._worker = None
        pending = self._pending_day_read
        self._pending_day_read = None
        if pending is not None:
            session, backfill_bars = pending
            self._request_day_read(session, backfill_bars=backfill_bars)

    def _render_failure(self, reason: str) -> None:
        self.status.setText(f"the session could not be read: {reason}")
        self.statusChanged.emit(self.status.text())

    def _refresh_if_loaded(self) -> None:
        if self._loaded_once:
            self.reload()

    # -- rendering ---------------------------------------------------------
    def render(self, payload: Mapping[str, Any], *, allow_backfill: bool = True) -> None:
        """Draw one payload. Formatting only - it computes nothing.

        Tolerates a payload with nothing in it: a first paint before any read and
        a read that failed are the same shape.
        """
        payload = dict(payload or {})
        self._payload = payload
        self._zone = _display_zone()
        self.zone_note.setText(f"Times in {_zone_label(self._zone)}")
        session = str(payload.get("session_date") or self.session_date())
        self.provisional_note.setText(
            "This session has NOT closed - every number on it is provisional."
            if payload.get("provisional")
            else f"Session {session}, closed and measured."
        )
        # TJ-10, BEFORE the entries: the list shows each read's verdict beside
        # the words it graded, and both come from this one payload.
        self._reads = {
            str(row.get("entry_id") or ""): dict(row)
            for row in (payload.get("reads") or ())
            if isinstance(row, Mapping) and row.get("entry_id")
        }
        # TJ-12, FIRST because it heads the page: six texts the worker already
        # built. `render` formats - it never calls `day_report_card.build` or
        # `how_fresh`, which are worker work inside the ONE payload.
        self._render_report_card(payload.get("report_card"))
        truth = payload.get("truth") if isinstance(payload.get("truth"), Mapping) else {}
        self.truth_note.setText("\n".join(str(line) for line in truth.get("lines") or ()))
        self._bot_grades = dict(truth.get("grades") or {})
        # The glance strip heads the page. The worker builds it; a hand-built
        # payload without one gets the same pure projection here (a few sums).
        if not payload.get("glance"):
            import day_report_card

            payload["glance"] = day_report_card.glance(payload)
        self.glance_strip.set_glance(payload["glance"])
        # Before the tables and trades: a row or trade click draws from these.
        self._name_charts = dict(payload.get("name_charts") or {})
        self._render_story(payload.get("story"))
        # TJ-4, AFTER the facts: the verified story replaces the "no story yet"
        # line when there is one, and leaves the facts exactly as they were when
        # there is not.
        self._render_day_story(
            payload.get("day_story"), session, payload.get("report_card"),
            payload.get("story_freshness"),
        )
        self._render_d1_view(payload.get("d1_view"))
        self._render_congruence(tuple(payload.get("congruence") or ()))
        self._render_theses(payload.get("theses") or [])
        self._render_tj2b_walkaway(payload.get("walkaway"))
        self._render_entries(list(payload.get("entries") or []))
        self._render_forecast(dict(payload.get("forecast") or {}))
        self._render_trades(list(payload.get("trades") or []))
        self._trade_reviews = {
            str(row.get("trade_id") or ""): dict(row)
            for row in payload.get("trade_reviews") or () if isinstance(row, Mapping)
        }
        self._render_calls(tuple(payload.get("reads") or ()))
        if self.trades_table.rowCount():
            self.trades_table.selectRow(0)
            self._show_selected_trade()
        else:
            self.trade_detail.setPlainText("No trades this session.")
            self._show_trade_chart("")
        self._render_chart(list(payload.get("spy_m5_bars") or []))
        # The markers were RESOLVED on the worker; this pushes them and computes
        # nothing (TJ-3). After `set_data`, because new bars drop the payload.
        self._render_spy_markers(
            tuple(payload.get("spy_markers") or ()),
            payload.get("spy_marker_placements"),
        )
        self._refresh_name_chart()
        self._render_ideas(session, list(payload.get("ideas") or []))
        self._render_mood(payload.get("mood"))
        self._render_tape_and_digest(payload)
        self._sync_walk_entry(session, payload)
        if allow_backfill:
            for exit_session in tuple(payload.get("walkaway_backfill_sessions") or ()):
                self._backfill_bars_for(str(exit_session))
        self.status.setText(_status_problems(payload.get("error")) or f"Day Review: {session}")
        self.statusChanged.emit(self.status.text())

    def _render_tape_and_digest(self, payload: Mapping[str, Any]) -> None:
        """S5's tape line and B11's digest narration. Formatting only: both were read on the worker."""
        import setup_grades

        tape = payload.get("side_by_tape")
        self.tape_side_note.setText(
            setup_grades.side_by_tape_line(tape if isinstance(tape, Mapping) else None)
        )
        digest = payload.get("digest_narration")
        text = str(digest.get("text") or "") if isinstance(digest, Mapping) else ""
        self.digest_note.setText(text or "Night digest: not read.")

    def _render_mood(self, section: Any) -> None:
        """TJ-7 / live gate #151: ONE line about the trader, from ONE payload.

        The line was FORMATTED on the worker (`day_review_pack.mood_statement`),
        so this pushes a string and computes nothing on the Qt thread: no
        builder call, no model call, no second read of the journal. A session
        with nothing clicked reads "No mood recorded yet ... (n 0)" - a count,
        never a percentage - and a mood typed in the evening says so on the
        line, because `written_after_the_session` is a LABEL and not a reason
        to hide one.
        """
        import day_review_pack

        line = ""
        if isinstance(section, Mapping):
            line = str(section.get("line") or "")
            if not line:
                line = day_review_pack.mood_statement({"mood": section})
        self.mood_line.setText(line or day_review_pack.MOOD_EMPTY_STATEMENT)

    def _render_ideas(self, session: str, rows: list) -> None:
        """TJ-6: the night's suggestions, or the line that says there are none.

        The rows were READ on the worker with the rest of the payload. A
        dismissed idea is already gone by the time they arrive - this formats
        what it is handed and decides nothing.
        """
        self.ideas_card.set_session(session)
        self.ideas_card.show_ideas(rows)
        self.ideas_card.setVisible(bool(rows))
        self.ideas_note.setVisible(not rows)

    def _render_story(self, story: Any) -> None:
        """The deterministic facts, under the fixed "no story yet" line.

        The line is FIXED and always shown until TJ-4: what is below it is what
        the desk MEASURED, and calling that a story would be the machine
        claiming a reading nobody wrote.
        """
        self.story_note.setText(NO_STORY_YET)
        self.story_note.setToolTip("")
        self.story_warning.setText("")
        self.story_warning.setVisible(False)
        if story is None:
            self.story_facts.setText("")
            return
        lines: list[str] = []
        for cell in tuple(getattr(story, "measured", ()) or ()):
            symbol = str(cell.get("symbol") or "")
            if str(cell.get("status") or "") != "measured":
                lines.append(f"{symbol}: not measured — {cell.get('reason') or 'no completed bars'}")
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

    @staticmethod
    def _story_attempt_state(card: Any) -> str:
        """Read the already-loaded `how_fresh` row; never open the ledger here."""
        lines = card.get("lines") if isinstance(card, Mapping) else getattr(card, "lines", ())
        for line in lines or ():
            if not isinstance(line, Mapping) or line.get("key") != "how_fresh":
                continue
            failed = set(line.get("slots_failed") or ())
            degraded = set(line.get("slots_degraded") or ())
            if "day_review_narration" in failed or "day_review_narration" in degraded:
                return "failed"
            return str(line.get("night_status") or "unknown")
        # A partial/older payload has no report-card row at all. Preserve the
        # normal pre-night message for it; only a loaded row may say unknown.
        return ""

    def _render_day_story(
        self, story: Any, session: str, card: Any, freshness: Any = None
    ) -> None:
        """The night's verified narration. Formatting only (TJ-4 item 4).

        Every verdict printed here was MEASURED by TJ-10's grader and copied by
        the night, which refuses the whole output rather than write one that
        disagrees. The page grades nothing, computes nothing and calls nothing.
        """
        self.story_body.setText("")
        self.story_body.setVisible(False)
        attempt_state = self._story_attempt_state(card)
        if isinstance(freshness, Mapping) and freshness.get("state") in {"stale", "unread", "missing"}:
            self.story_note.setText(
                f"Saved story is {freshness['state']}: {freshness.get('reason') or 'facts unavailable'}. "
                "Current measured results are below."
                + (" The latest AI story attempt failed its checks." if attempt_state == "failed" else "")
            )
            return
        if not isinstance(story, Mapping):
            if attempt_state == "failed":
                self.story_note.setText(
                    "The AI story failed its checks. Your measured results are still shown."
                )
            elif attempt_state == "no_rows":
                self.story_note.setText("The AI story has not run yet. Your measured results are still shown.")
            elif attempt_state == "unknown":
                self.story_note.setText(
                    "The AI story status is unknown. Your measured results are still shown."
                )
            return
        if str(story.get("session_date") or "") != str(session):
            return
        narration = story.get("narration")
        if not isinstance(narration, Mapping):
            return
        headline = str(narration.get("headline") or "").strip()
        if headline:
            self.story_note.setText(headline)
        if attempt_state == "failed":
            self.story_note.setToolTip("A new attempt failed its checks. These are the prior verified words.")
            self.story_warning.setText(
                "A new attempt failed its checks. These are the last verified words."
            )
            self.story_warning.setVisible(True)
        else:
            self.story_note.setToolTip("")
        lines: list[str] = []
        for key in ("what_happened", "what_you_thought"):
            text = str(narration.get(key) or "").strip()
            if text:
                lines.append(text)
        for claim in narration.get("were_you_right") or ():
            if not isinstance(claim, Mapping):
                continue
            said = str(claim.get("claim") or "").strip()
            verdict = self._verdict_text(str(claim.get("verdict") or ""))
            if said and verdict:
                lines.append(f"· {said} — {verdict}")
            elif said:
                lines.append(f"· {said}")
        chased = narration.get("chased_against_news")
        if isinstance(chased, Mapping):
            verdict = str(chased.get("verdict") or "").strip()
            if verdict:
                # `unknown` is printed exactly as it is: the desk does not
                # measure oil or the 10-year, and a condition it cannot see is
                # unknown, never assumed either way.
                lines.append(f"· Chased against the news: {verdict}")
        process = str(narration.get("process") or "").strip()
        if process:
            lines.append(process)
        # A SIZE statement, counted by the night and printed here: how many of
        # the session's measured reads this story graded. No result is involved
        # and nothing is ranked - it is said only when it is fewer than the
        # session held, so "the story covered everything" stays silent.
        graded = story.get("graded")
        if isinstance(graded, Mapping):
            try:
                covered = int(graded.get("reads_graded") or 0)
                held = int(graded.get("reads_in_pack") or 0)
            except (TypeError, ValueError):
                covered = held = 0
            if held and covered < held:
                lines.append(f"graded {covered} of {held} reads")
        body = "\n".join(_without_citations(line) for line in lines)
        self.story_body.setText(body)
        self.story_body.setVisible(bool(body))
        if headline or body:
            note_slot_output_read("day_review_narration")

    def _render_d1_view(self, view: Any) -> None:
        """The rolling D1 view, above the open theses. Formatting only."""
        self.d1_view_note.setText("")
        self.d1_view_note.setVisible(False)
        if not isinstance(view, Mapping):
            return
        narration = view.get("narration")
        if not isinstance(narration, Mapping):
            return
        lines: list[str] = []
        belief = str(narration.get("belief_now") or "").strip()
        if belief:
            lines.append(belief)
        for thesis in narration.get("open_theses") or ():
            if not isinstance(thesis, Mapping):
                continue
            claim = str(thesis.get("claim") or "").strip()
            if not claim:
                continue
            still = str(thesis.get("still_true") or "").strip()
            since = str(thesis.get("since") or "").strip()
            parts = [f"· {claim}"]
            if since:
                parts.append(f"since {since}")
            if still:
                parts.append(f"still true: {still}")
            lines.append(" — ".join(parts))
        text = "\n".join(_without_citations(line) for line in lines)
        self.d1_view_note.setText(text)
        self.d1_view_note.setVisible(bool(text))

    # -- Redo story --------------------------------------------------------
    def redo_story(self) -> None:
        """Ask for this session's story again - tonight, or now if it is night.

        The pack comes FIRST. The night narrates a session it has a pack for
        and skips one it does not, so a click that only wrote a marker could
        queue a request nobody would ever answer - which is what the live home
        folder, with three session folders and no packs, would have done to
        every Redo (reviewer round 2, 2026-09-20). The build is the page's own
        off-Qt seam, deterministic and model-free; only when it has produced a
        pack does the marker get written or the night's process get started.

        Local inference is still night-only, seven days a week (plan TJ-13 item
        5, TJ-4 change 4): outside the window this writes the `redo_requested`
        marker the nightly slot honours and SAYS it is queued; inside it, it
        starts one child process per click and writes no marker.

        ONE redo is in flight at a time. The click starts a full `read_day`
        and, at night, a child process; three impatient clicks used to start
        three of each, racing on one `pack.json` (reviewer round 3,
        2026-09-20). While a build is running this is a no-op that leaves the
        "Building…" note exactly as it is, and the button is grey from the
        click until the answer - every answer, including a build that raised.
        """
        if self._redo_busy:
            return
        session = self.session_date()
        if not session:
            return
        self._redo_busy = True
        self.redo_story_button.setEnabled(False)
        builder = getattr(self.service, "build_pack_for", None)
        if not callable(builder):
            # A host that hands this page a reader with no builder cannot build
            # one here either. The click still queues: a packless marker costs
            # the night no budget, keeps its place and is named in the ledger.
            self._redo_after_pack(session, True)
            return
        try:
            worker = _RedoPackWorker(builder, session, self)
            worker.done.connect(self._redo_after_pack)
            worker.finished.connect(lambda w=worker: self._drop_redo_worker(w))
            self._redo_workers.append(worker)
            self.status.setText(STORY_BUILDING_PACK_NOTE.format(session=session))
            self.statusChanged.emit(self.status.text())
            worker.start()
        except Exception:  # noqa: BLE001 - a failed start never raises into Qt
            logging.debug("The Redo's pack build could not start.", exc_info=True)
            self._redo_after_pack(session, False)

    def _drop_redo_worker(self, worker) -> None:
        """Let a finished build go. Held until then so Qt does not collect it.

        It also RELEASES the button, as a backstop: a worker that ended without
        its `done` reaching this page must not leave the verb grey for ever.
        """
        try:
            self._redo_workers.remove(worker)
        except ValueError as exc:
            note_swallowed("redo worker already released", exc, quiet=True)
        self._release_redo()
        worker.deleteLater()

    def _release_redo(self) -> None:
        """One redo has answered: let the next click through. Idempotent."""
        self._redo_busy = False
        try:
            self.redo_story_button.setEnabled(True)
        except RuntimeError as exc:  # pragma: no cover - the panel is being destroyed
            note_swallowed("redo button already destroyed", exc, quiet=True)

    def _redo_after_pack(self, session_date: str, built: bool) -> None:
        """Queue it for tonight, or start it - now that the facts exist.

        Every ending releases the button: queued, launched, no pack, refused,
        and a launcher that raised.
        """
        try:
            self._redo_outcome(session_date, built)
        finally:
            self._release_redo()

    def _redo_outcome(self, session_date: str, built: bool) -> None:
        import day_review_pack

        session = str(session_date or "")[:10]
        if not built:
            self.status.setText(REDO_NO_PACK_NOTE.format(session=session))
            self.statusChanged.emit(self.status.text())
            return
        # Checked on BOTH branches. A day the exchange never opened, or one
        # that has not closed, has no story to redo - and by night the launch
        # branch used to start a process for it and report it as under way
        # (reviewer round 3).
        try:
            session = day_review_pack.validated_session(session, now=self._clock())
        except ValueError as exc:
            self.status.setText(REDO_REFUSED_NOTE.format(session=session, reason=exc))
            self.statusChanged.emit(self.status.text())
            return
        try:
            allowed, reason = window.launch_allowed()
        except Exception as exc:  # noqa: BLE001 - an unreadable window queues
            allowed, reason = False, f"the night window could not be read ({exc})"
        if not allowed:
            try:
                day_review_pack.request_redo(session, now=self._clock())
                text = f"{STORY_QUEUED_NOTE} {reason}".strip()
            except Exception as exc:  # noqa: BLE001 - a marker never costs the page
                logging.debug("The story redo could not be queued.", exc_info=True)
                text = REDO_REFUSED_NOTE.format(session=session, reason=exc)
            self.status.setText(text)
            self.statusChanged.emit(self.status.text())
            return
        try:
            self._redo_launcher(session)
        except Exception as exc:  # noqa: BLE001 - a failed launch never raises into Qt
            logging.debug("The story redo could not be started.", exc_info=True)
            self.status.setText(f"the story redo could not be started: {exc}")
            self.statusChanged.emit(self.status.text())
            return
        self.status.setText(STORY_REDO_STARTED_NOTE.format(session=session))
        self.statusChanged.emit(self.status.text())

    def _render_congruence(self, lines) -> None:
        """The three lines, under the story. Formatting only (TJ-10 item 6).

        Printed, never pushed and never acted on: there is no button here, no
        threshold and no colour that means "do something". A line the desk could
        not measure SAYS which side was missing rather than going quiet.
        """
        import day_report_card

        rendered: list[str] = []
        for line in lines or ():
            if not isinstance(line, Mapping):
                continue
            text = str(line.get("text") or "").strip()
            if not text:
                continue
            verdict = str(line.get("verdict") or "")
            missing = str(line.get("missing") or "")
            if missing:
                text = f"{text} — missing: {missing}"
            elif verdict:
                text = f"{text} — {verdict}"
            rendered.append(f"· {day_report_card.plain_words(text)}")
        self.congruence_note.setText("\n".join(rendered))
        self.congruence_note.setVisible(bool(rendered))

    def congruence_text(self) -> str:
        """What the congruence block is showing. Read by tests and by nothing else."""
        return self.congruence_note.text()

    def verdict_chips(self) -> dict[str, str]:
        """`entry_id -> the MEASURED verdict` currently on the page.

        Only a read the worker actually graded is in it: a `No view` answer and
        an entry with no read carry no chip, because the page never shows a
        verdict nobody measured.
        """
        return {
            entry_id: str(row.get("verdict") or "")
            for entry_id, row in self._reads.items()
            if str(row.get("verdict") or "").strip()
        }

    @staticmethod
    def _verdict_text(verdict: str) -> str:
        """One short chip. `unmeasured:<reason>` reads as English, not a key."""
        text = str(verdict or "").strip()
        if text.startswith("unmeasured:"):
            return "unmeasured — " + text.split(":", 1)[1].replace("_", " ")
        return text

    @staticmethod
    def _read_chip(read: Mapping[str, Any] | None) -> str:
        """The chip beside an entry: WHAT was graded, then how it turned out.

        Live clicks are 0 and every live read row is an EXTRACTION, so a chip
        that said only "right" would present a stance the desk inferred from a
        sentence as the trader's own stated call (reviewer, 2026-09-20). The
        source is named in plain words, on every chip, always.
        """
        if not read:
            return ""
        verdict = DayReviewPanel._verdict_text(str(read.get("verdict") or ""))
        if not verdict:
            return ""
        direction = str(read.get("direction") or "")
        if str(read.get("source") or "") == "click":
            said = f"your call: {direction}" if direction else "your call"
        else:
            said = (
                f"we read your note as {direction}" if direction
                else "read from your note"
            )
        return f"{said} — {verdict}"

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

    def _walkaway_cells(self, row: Any) -> dict[str, str]:
        """One row, keyed by HEADER text.

        Keyed rather than positional: the populations share one column list,
        and a positional tuple is how a table came to print its "Ran after %"
        under "Held at close %".
        """
        session, zone = self.session_date(), self._zone
        return {
            "Time": _clock_text(row.time, session, zone=zone) if row.time else UNMEASURED,
            "Symbol": row.symbol,
            "Side": row.side,
            "What you did": _plain_cell(row.what_you_did),
            "Ran after %": _tj2_pct(row.ran_after_pct),
            "Against you first %": _tj2_pct(getattr(row, "against_first_pct", None)),
            "At the close %": _tj2_pct(getattr(row, "at_close_pct", None)),
            "Ran after (ATR)": _tj2_number(getattr(row, "ran_after_atr", None)),
            "Against you first (ATR)": _tj2_number(getattr(row, "against_first_atr", None)),
            "At the close (ATR)": _tj2_number(getattr(row, "at_close_atr", None)),
            "Real miss": _plain_cell(getattr(row, "real_miss", "")) or UNMEASURED,
            "Held at close %": _tj2_pct(row.held_at_close_pct),
            "Traded?": row.traded,
            "You made": _tj2_number(row.you_made),
            "Left on the table %": (
                str(getattr(row, "not_judged_reason", "") or "")
                or _tj2_pct(row.left_on_table_pct)
            ),
            "State": _plain_cell(row.state),
        }

    @staticmethod
    def _fill_walkaway_table(table: QTableWidget, rows, cells_for) -> None:
        """Fill one table without paying a column measurement per cell.

        `_fill_the_width` leaves every column but the last in
        `ResizeToContents`, and Qt re-measures those columns on EVERY
        `setItem` - bounded by `MEASURE_PRECISION_ROWS` (200) rows, through a
        styled delegate, on the GUI thread. Measured 2026-09-19 on a staged
        home: TJ-11's five tables and ~700 rows spent **90 seconds** inside one
        `processEvents`. The mode is suspended for the fill and restored once,
        so the measurement happens a single time and the columns still measure
        their own headers. Signals are blocked so a fill never "clicks" a row.
        """
        header = table.horizontalHeader()
        columns = table.columnCount()
        modes = [header.sectionResizeMode(index) for index in range(columns)]
        for index, mode in enumerate(modes):
            if mode == QHeaderView.ResizeMode.ResizeToContents:
                header.setSectionResizeMode(index, QHeaderView.ResizeMode.Interactive)
        blocked = table.blockSignals(True)
        table.setUpdatesEnabled(False)
        try:
            table.clearSelection()
            table.setRowCount(len(rows))
            for index, row in enumerate(rows):
                cells = cells_for(row)
                for column, name in enumerate(TJ2B_WALKAWAY_COLUMNS):
                    table.setItem(index, column, QTableWidgetItem(cells.get(name, "")))
            table.setCurrentCell(-1, -1)
        finally:
            table.setUpdatesEnabled(True)
            table.blockSignals(blocked)
            for index, mode in enumerate(modes):
                if header.sectionResizeMode(index) != mode:
                    header.setSectionResizeMode(index, mode)

    def _render_tj2b_walkaway(self, day) -> None:
        """Count the five populations onto their chips and show the chosen one.

        Formatting only: every number was measured on the worker, and a missing
        one is a dash with its reason, never a 0.00. `day` is None when the
        walk-away read failed: every chip then says so rather than "0".
        """
        import day_report_card

        read = day is not None
        sentences = dict(getattr(day, "sentences", {}) or {}) if read else {}
        for name, label in MISS_FILTERS:
            rows = tuple(getattr(day, name, ()) or ()) if read else ()
            self._miss_rows[name] = rows
            self._miss_sentences[name] = day_report_card.plain_words(sentences.get(name, ""))
            chip = self.miss_chips[name]
            chip.setText(f"{_button_text(label)} ({len(rows)})" if read else f"{_button_text(label)} (not read)")
            chip.setToolTip(
                f"{dict(TJ2B_WALKAWAY_TITLES).get(name, label)} - n={len(rows)}; "
                f"median Ran after {_tj2_pct(_median_ran_after(rows))}"
                if read else "The walk-away tables were not read for this session."
            )
        self.walkaway_skill.setText(
            "\n".join(
                day_report_card.plain_words(line)
                for line in self._skill_text(getattr(day, "skill", None)).splitlines()
            )
            if read else ""
        )
        self._show_miss_population()
        if not read:
            self.walkaway_note.setText("The walk-away tables were not read for this session.")

    def _show_miss_population(self) -> None:
        """Fill the one table with the selected chip's rows. Formatting only."""
        name = self._miss_population
        rows = self._miss_rows.get(name, ())
        self._walkaway_rows = rows
        self._apply_miss_columns()
        self._fill_walkaway_table(self.miss_table, rows, self._walkaway_cells)
        title = dict(TJ2B_WALKAWAY_TITLES).get(name, name)
        self.walkaway_sentence.setText(self._miss_sentences.get(name, ""))
        self.walkaway_sentence.setVisible(bool(self._miss_sentences.get(name)))
        self.walkaway_note.setText(
            f"{title}: {len(rows)} name(s), median ran after "
            f"{_tj2_pct(_median_ran_after(rows))}. Click a row to chart it here; "
            "double-click to open it on the board."
            if rows
            else f"{title}: nothing on this session."
        )

    def _on_miss_row_changed(self, row: int, _column: int, previous: int, _pc: int) -> None:
        """A single click, or J/K, draws that row's name beside the table."""
        if row == previous or row < 0 or row >= len(self._walkaway_rows):
            return
        symbol = str(getattr(self._walkaway_rows[row], "symbol", "") or "").strip().upper()
        if symbol:
            self._open_name_chart(symbol)

    @staticmethod
    def _skill_text(skill: Any) -> str:
        """Both base-rate windows, each saying its own window in SESSIONS.

        The session line alone is the one a fresh day cannot fill: its
        five-session horizons are still open, so it reads `measured 0, pending
        N` and names no rate. The lately line is what has actually closed.
        """
        if not isinstance(skill, Mapping):
            return ""
        lines: list[str] = []
        for key, fallback_label in (("session", "this session"), ("lately", "lately")):
            block = skill.get(key)
            if not isinstance(block, Mapping):
                continue
            sentence = str(block.get("sentence") or "").strip()
            if not sentence:
                continue
            sessions = block.get("window_sessions")
            count = f"{int(sessions)} session{'s' if int(sessions) != 1 else ''}" if isinstance(
                sessions, (int, float)
            ) else fallback_label
            lines.append(f"{count}: {sentence}")
        return "\n".join(lines)

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
                # TJ-14A: a Mentor card answered with a CLICK and no words is a
                # complete answer whose `text` is empty on purpose. The row
                # shows the call rather than a blank line, and nothing invents
                # a sentence the trader did not write.
                body = str(entry.get("text") or "") or _prediction_text(entry)
                # TJ-10: the verdict the WORKER measured, beside the words it
                # graded, and SAYING which it graded - a clicked call or a
                # stance the desk read out of the note. Nothing is computed
                # here and an ungraded row shows no chip at all.
                verdict = self._read_chip(
                    self._reads.get(str(entry.get("entry_id") or ""))
                )
                label = (
                    f"{_clock_text(entry.get('created_at'), self.session_date(), zone=self._zone)}"
                    f"  ·  {entry.get('timeframe') or ''}{marker}"
                    f"{('  ·  ' + verdict) if verdict else ''}"
                    f"  ·  {_excerpt(body)}"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, str(entry.get("entry_id") or ""))
                item.setToolTip(body)
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
            self._show_verdict_chip(None)
            return
        self._show_verdict_chip(self._reads.get(str(entry.get("entry_id") or "")))
        origin = str(entry.get("origin") or "")
        stamp = _clock_text(entry.get("created_at"), self.session_date(), zone=self._zone)
        meta = f"written {stamp}  ·  {entry.get('timeframe') or ''}  ·  {_plain_cell(origin)}"
        if entry.get("written_after_the_session"):
            meta += "  ·  written after the session"
        self.entry_meta.setText(meta)
        # The words, then the call beneath them (TJ-14A keeps the two apart).
        # A clicks-only answer shows its call alone; it never reads as blank.
        words = str(entry.get("text") or "")
        call = _prediction_text(entry)
        self.entry_reader.setPlainText(
            "\n\n".join(part for part in (words, call) if part)
        )

    def _show_verdict_chip(self, read: Mapping[str, Any] | None) -> None:
        """One chip for the selected read, styled by a DYNAMIC PROPERTY.

        `theme.qss` keys on `#VerdictChip[verdict="right"]` and friends, so no
        stylesheet is built on the Qt thread (CLAUDE.md: "expensive" includes a
        stylesheet). A property change needs an explicit repolish; that is one
        widget, not a page.
        """
        verdict = str((read or {}).get("verdict") or "")
        text = self._read_chip(read)
        kind = verdict.split(":", 1)[0].split(" ", 1)[0]
        self.verdict_chip.setText(text)
        self.verdict_chip.setVisible(bool(text))
        if self.verdict_chip.property("verdict") == kind:
            return
        self.verdict_chip.setProperty("verdict", kind)
        style = self.verdict_chip.style()
        style.unpolish(self.verdict_chip)
        style.polish(self.verdict_chip)

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
        self._trade_ids = [str(row.get("trade_id") or "") for row in rows]
        self._trade_symbols = [str(row.get("symbol") or "").strip().upper() for row in rows]
        self.trades_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            quantity = row.get("quantity")
            if quantity is None:
                quantity = row.get("quantity_opened")
            values = (
                _clock_text(row.get("opened_at"), self.session_date(), zone=self._zone),
                str(row.get("symbol") or ""),
                str(row.get("direction") or ""),
                self._number(quantity, decimals=0),
                self._number(row.get("net_pnl"), signed=True),
                str(row.get("status") or ""),
                self._bot_grade_cell(row.get("trade_id")),
            )
            for column, text in enumerate(values):
                self.trades_table.setItem(index, column, QTableWidgetItem(str(text)))

    def _bot_grade_cell(self, trade_id: Any) -> str:
        """The bot's grade of this trade's setup as of the entry (worker-built), or a dash."""
        grade = (getattr(self, "_bot_grades", {}) or {}).get(str(trade_id or ""))
        return str(grade.get("grade") or "") if isinstance(grade, Mapping) else UNMEASURED

    def _render_calls(self, rows) -> None:
        calls = [row for row in rows if isinstance(row, Mapping)]
        self._call_entry_ids = [str(row.get("entry_id") or "") for row in calls]
        header = self.calls_table.horizontalHeader()
        modes = [header.sectionResizeMode(column) for column in range(self.calls_table.columnCount())]
        for column, mode in enumerate(modes):
            if mode == QHeaderView.ResizeMode.ResizeToContents:
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        self.calls_table.setUpdatesEnabled(False)
        try:
            self.calls_table.setRowCount(len(calls))
            for index, row in enumerate(calls):
                values = (
                    _clock_text(row.get("stamp"), self.session_date(), zone=self._zone),
                    _plain_cell(row.get("horizon")),
                    str(row.get("direction") or ""),
                    str(row.get("confidence") or ""),
                    self._verdict_text(str(row.get("verdict") or "unmeasured")),
                )
                for column, value in enumerate(values):
                    self.calls_table.setItem(index, column, QTableWidgetItem(value))
        finally:
            self.calls_table.setUpdatesEnabled(True)
            for column, mode in enumerate(modes):
                header.setSectionResizeMode(column, mode)

    def _open_call_row(self, row: int, _column: int) -> None:
        refs = getattr(self, "_call_entry_ids", ())
        if 0 <= row < len(refs):
            self._select_entry_by_ref(refs[row])

    def _selected_trade_id(self) -> str:
        index = self.trades_table.currentRow()
        ids = getattr(self, "_trade_ids", ())
        return ids[index] if 0 <= index < len(ids) else ""

    def _open_selected_trade(self, *_args) -> None:
        trade_id = self._selected_trade_id()
        if trade_id:
            self.openTradeRequested.emit(trade_id)

    def _show_trade_chart(self, symbol: str) -> None:
        """Draw the selected trade's name, from the worker's `name_charts` alone.

        The chart carries that name's marks for the session - the entry and
        exit legs among them. No read, no store, no marker build here.
        """
        name = str(symbol or "").strip().upper()
        chart = dict(self._name_charts.get(name) or {}) if name else {}
        bars = [
            bar for bar in (chart.get("bars") or ())
            if isinstance(bar, Mapping) and bar.get("dt") is not None
        ]
        if not bars:
            self._trade_chart_symbol = ""
            if not name:
                self.trade_chart_note.setText(
                    TRADE_CHART_IDLE_NOTE if self.trades_table.rowCount()
                    else TRADE_CHART_NO_TRADES_NOTE
                )
            else:
                self.trade_chart_note.setText(NAME_CHART_MISSING_NOTE.format(symbol=name))
            if self._trade_chart is not None:
                self._trade_chart.set_data([])
                self._trade_chart.setVisible(False)
            return
        if self._trade_chart is None:
            from ui.widgets.candle_chart import CandleChart

            self._trade_chart = CandleChart()
            self._trade_chart.setMinimumHeight(theme.px(NAME_CHART_MIN_HEIGHT_PX))
            self._trade_chart.markerClicked.connect(self._select_entry_by_ref)
            self._trade_chart_layout.addWidget(self._trade_chart)
        self._trade_chart.setVisible(True)
        self._trade_chart.set_data(bars, timeframe="m5")
        self._trade_chart.set_note_markers(tuple(chart.get("markers") or ()))
        self._trade_chart_symbol = name
        self._draw_saved_clues(self._trade_chart, "_trade_clue_flow", name)
        caption = self._marker_caption(chart.get("placements"))
        self.trade_chart_note.setText(
            f"{name} M5 this session, with your entry and exit marked."
            + (f" {caption}" if caption else "")
        )

    def trade_chart_symbol(self) -> str:
        """Which name the trade chart is showing. "" when none."""
        return self._trade_chart_symbol

    def _show_selected_trade(self) -> None:
        trade_id = self._selected_trade_id()
        index = self.trades_table.currentRow()
        symbols = getattr(self, "_trade_symbols", ())
        self._show_trade_chart(symbols[index] if 0 <= index < len(symbols) else "")
        detail = getattr(self, "_trade_reviews", {}).get(trade_id)
        if detail is None:
            self.trade_detail.setPlainText("Trade answers were not read.")
            return
        if detail.get("status") == "unread":
            self.trade_detail.setPlainText("Trade answers could not be read.")
            return
        session, zone = self.session_date(), self._zone

        def _when(value: Any) -> str:
            text = _clock_text(value, session, zone=zone)
            return "" if text == UNMEASURED else text

        instrument = str(detail.get("instrument") or "").strip()
        lines = [" · ".join(part for part in (str(detail.get("symbol") or ""), instrument) if part)]
        opened, closed = _when(detail.get("opened_at")), _when(detail.get("closed_at"))
        lines.append(
            f"Opened {opened or 'not recorded'} · "
            + (f"Closed {closed}" if closed else "Still open")
        )
        money = detail.get("net_pnl")
        currency = str(detail.get("currency") or "").strip()
        lines.append(
            f"Whole-trade net: {self._number(money, signed=True)}"
            + (f" {currency}" if currency and money is not None else "")
        )
        if detail.get("review_pnl_note"):
            lines.append(str(detail["review_pnl_note"]))
        raw = detail.get("entry_raw") or {}
        recalled = _when(raw.get("recorded_at"))
        entry_words = str(raw.get("text") or "").strip()
        lines.append(
            f"Entry note{f' ({recalled})' if recalled and entry_words else ''}: "
            + (entry_words or "you did not write one")
        )
        for field, answer in (detail.get("entry_answers") or {}).items():
            value = answer.get("text") or (
                f"{answer['value']} {answer.get('unit') or ''}"
                if answer.get("value") is not None else ""
            )
            state = str(answer.get("state") or "").replace("_", " ")
            spoken = " · ".join(part for part in (str(value).strip(), state) if part) or "not answered"
            said_at = _when(answer.get("recorded_at"))
            lines.append(f"{field}: {spoken}" + (f" · {said_at}" if said_at else ""))
        exit_raw = detail.get("exit_raw") or {}
        exit_at = _when(exit_raw.get("recorded_at"))
        exit_words = str(
            exit_raw.get("text") or str(exit_raw.get("answer_state") or "").replace("_", " ")
        ).strip()
        lines.append(
            f"Exit note{f' ({exit_at})' if exit_at and exit_words else ''}: "
            + (exit_words or "you did not write one")
        )
        confirmed = detail.get("exit_fields") or {}
        fields = confirmed.get("fields") if confirmed.get("status") == "confirmed" else None
        if isinstance(fields, Mapping):
            lines.append("Confirmed exit: " + ", ".join(f"{key}: {value}" for key, value in fields.items()))
        else:
            lines.append("Exit reading: not confirmed")
        if detail.get("label_provenance"):
            lines.append(f"Label source: {detail['label_provenance']}")
        grade = (getattr(self, "_bot_grades", {}) or {}).get(trade_id)
        if isinstance(grade, Mapping):
            import journal_truth

            why = str(grade.get("why") or "")
            lines.append(f"Bot grade: {grade.get('grade') or journal_truth.NO_GRADE}" + (f" ({why})" if why else ""))
        self.trade_detail.setPlainText("\n".join(lines))

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
            self._chart.markerClicked.connect(self._select_entry_by_ref)
            self._chart_layout.addWidget(self._chart)
        return self._chart

    def _ensure_name_chart(self):
        """Build the name pane ONCE, the first time a row asks for it."""
        if self._name_chart is None:
            from ui.widgets.candle_chart import CandleChart

            self._name_chart = CandleChart()
            self._name_chart.setMinimumHeight(theme.px(NAME_CHART_MIN_HEIGHT_PX))
            self._name_chart.markerClicked.connect(self._select_entry_by_ref)
            self._name_chart_layout.addWidget(self._name_chart)
        return self._name_chart

    def name_chart_symbol(self) -> str:
        """Which name the side pane is showing. "" when it is showing none."""
        return self._name_chart_symbol

    @staticmethod
    def _marker_caption(placements: Mapping[str, Any] | None) -> str:
        """What the tape could NOT carry, in words. Counted on the worker.

        A mark the tape cannot carry is the one thing the trader must be told
        about: a fill two hours after the last candle used to be drawn ON that
        candle, which is the chart saying they acted at the close.
        """
        counts = dict(placements or {})

        def _count(name: str) -> int:
            try:
                return int(counts.get(name) or 0)
            except (TypeError, ValueError):
                return 0

        parts: list[str] = []
        after = _count("after_tape")
        between = _count("between_bars")
        if after:
            parts.append(f"{after} mark{'s' if after != 1 else ''} after the tape")
        if between:
            parts.append(f"{between} off a drawn bar")
        return f"{' · '.join(parts)} - not drawn." if parts else ""

    def _render_spy_markers(self, markers, placements=None) -> None:
        """Push the worker's marker payload onto the one SPY chart.

        Says out loud what could not be drawn - including the case where there
        is no tape at all, which used to be a silence.
        """
        caption = self._marker_caption(placements)
        if self._chart is None or not self._chart.bar_count():
            self.spy_note.setText(
                f"{self.spy_note.text()} {NO_TAPE_MARKER_NOTE}".strip()
            )
            return
        try:
            self._chart.set_note_markers(markers)
        except Exception:  # noqa: BLE001 - a marker never costs the page
            logging.debug("The Day Review markers could not be drawn.", exc_info=True)
        if caption:
            self.spy_note.setText(f"{self.spy_note.text()} {caption}".strip())

    def _refresh_name_chart(self) -> None:
        """Redraw whatever the side pane is already showing, from the new read.

        A name the new payload has no tape for stops being shown rather than
        going on showing yesterday's candles under today's heading.
        """
        if not self._name_chart_symbol:
            return
        self._open_name_chart(self._name_chart_symbol)

    def _open_name_chart(self, symbol: str) -> None:
        """Draw one name BESIDE the tables, from the payload alone.

        No read, no store, no builder: `name_charts` came off the same worker
        payload the tables did, with its markers already resolved against these
        very bars.
        """
        name = str(symbol or "").strip().upper()
        chart = dict(self._name_charts.get(name) or {})
        bars = [
            bar for bar in (chart.get("bars") or ())
            if isinstance(bar, Mapping) and bar.get("dt") is not None
        ]
        if not bars:
            self._name_chart_symbol = ""
            self.name_chart_note.setText(NAME_CHART_MISSING_NOTE.format(symbol=name))
            if self._name_chart is not None:
                self._name_chart.set_data([])
                self._name_chart.setVisible(False)
            return
        pane = self._ensure_name_chart()
        pane.setVisible(True)
        pane.set_data(bars, timeframe="m5")
        pane.set_note_markers(tuple(chart.get("markers") or ()))
        self._name_chart_symbol = name
        self._draw_saved_clues(pane, "_name_clue_flow", name)
        caption = self._marker_caption(chart.get("placements"))
        self.name_chart_note.setText(
            f"{name} M5 — {len(bars)} completed bar(s), and what you said about it."
            + (f" {caption}" if caption else "")
        )

    def _select_entry_by_ref(self, ref_id: str) -> None:
        """A marker click selects the note it names - the trader's own question.

        A marker for something that is not a note (a trade leg, a decision row)
        moves no selection: the reader is showing a thought, and replacing it
        with a blank would lose the one the trader was reading.
        """
        wanted = str(ref_id or "")
        if not wanted:
            return
        for index, entry in enumerate(self._entries):
            if str(entry.get("entry_id") or "") == wanted:
                self.entries.setCurrentRow(index)
                self._fill_reader(entry)
                return

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

    def _row_key_tables(self) -> tuple[QTableWidget, ...]:
        return tuple(
            table for table in (
                getattr(self, "miss_table", None),
                getattr(self, "trades_table", None),
                getattr(self, "calls_table", None),
            ) if table is not None
        )

    @staticmethod
    def step_row(table: QTableWidget, step: int) -> None:
        """Move `table`'s current row by `step`, staying inside the table."""
        count = table.rowCount()
        if not count:
            return
        current = table.currentRow()
        target = 0 if current < 0 else max(0, min(count - 1, current + int(step)))
        column = max(0, table.currentColumn())
        table.setCurrentCell(target, column)
        table.selectRow(target)

    def _activate_walkaway(self, item) -> None:
        """Open that name BESIDE the table, and tell the host about it too.

        Two different answers to one click, and TJ-3 adds the second without
        removing the first: the host's `show_board_symbol` is the door for a
        board on another page (a board chart holds no place in the waiting list,
        so nothing here is re-queued or skip-counted), while this page draws the
        name's own session with the trader's words on it.
        """
        if item is None:
            return
        index = item.row()
        rows = self._walkaway_rows
        if index < 0 or index >= len(rows):
            return
        row = rows[index]
        symbol = str(getattr(row, "symbol", "") or "").strip().upper()
        if not symbol:
            return
        self._open_name_chart(symbol)
        self.chartRequested.emit(symbol, str(getattr(row, "side", "") or ""))

    # -- writing -----------------------------------------------------------
    def eventFilter(self, watched, event):  # noqa: N802 (Qt override)
        """Enter saves, Shift+Enter starts a new line (decision 0016 answer 11).

        In the page's tables, J moves to the next row and K to the previous one.
        """
        try:
            if (
                watched is getattr(self, "entry_text", None)
                and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            ):
                self._save()
                return True
            if (
                event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_J, Qt.Key.Key_K)
                and not event.modifiers()
                and watched in self._row_key_tables()
            ):
                self.step_row(watched, 1 if event.key() == Qt.Key.Key_J else -1)
                return True
        except Exception as exc:  # noqa: BLE001 - a key handler never breaks the page
            note_swallowed("day review row key handler raised", exc)
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

    # -- TJ-7: the optional mood strip beside the composer ------------------
    def mood_button(self, score):
        """The face for `score`, or ``None``."""
        return self.mood_strip.mood_button(score)

    def state_tag_button(self, code: str):
        """The chip for `code`, or ``None``."""
        return self.mood_strip.state_tag_button(code)

    def mood_answer(self) -> dict:
        """What the trader clicked. Nothing is ever pre-filled for them."""
        return self.mood_strip.answer()

    def _save(self) -> None:
        mood = self.mood_answer()
        result = self.service.write_entry(
            text=self.entry_text.toPlainText(),
            session_date=self.session_date(),
            timeframe=self.timeframe_picker.currentText(),
            origin="journal_page",
            # TJ-7. `None` and `()` when the strip was not touched, which
            # stores the key present and EMPTY - never a default face.
            mood=mood["mood"],
            state_tags=tuple(mood["state_tags"]),
        )
        if result.get("ok"):
            self.entry_text.clear()
            # Yesterday's face is not tomorrow's: the strip goes back to
            # nothing selected on every successful save.
            self.mood_strip.reset()
            self.status.setText("Entry saved.")
            self._refresh_if_loaded()
        else:
            self.status.setText(f"Entry NOT saved: {result.get('reason', '')}")
        self.statusChanged.emit(self.status.text())

    # -- the pasted forecast -----------------------------------------------
    def _paste_daily_forecast(self) -> None:
        """Open the paste dialog. It is NON-modal: the desk keeps running."""
        dialog = getattr(self, "_forecast_dialog", None)
        try:
            if dialog is not None and dialog.isVisible():
                dialog.raise_()
                dialog.activateWindow()
                return
        except RuntimeError as exc:  # the last dialog was deleted on close
            note_swallowed("last forecast dialog already deleted", exc, quiet=True)
        self._forecast_dialog = self._ask_for_forecast(self._import_forecast)

    def _ask_for_forecast(self, on_accept: Callable[[dict], Any]):
        """Build and SHOW the paste dialog; `on_accept` gets what was typed.

        The date field DEFAULTS to the date in the brief's own first heading
        (`forecast_brief.parse`), and stops defaulting the moment the trader
        types in it. The source starts EMPTY: nothing is pre-filled for them.
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
        dialog.setModal(False)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        text_box = QPlainTextEdit()
        text_box.setPlaceholderText("Paste the brief exactly as it was written.")
        session_box = QLineEdit(self.session_date())
        model_box = QLineEdit()
        model_box.setPlaceholderText("Who wrote it, e.g. chatgpt (optional)")
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
        buttons.rejected.connect(dialog.reject)

        def _accept() -> None:
            values = {
                "text": text_box.toPlainText(),
                "target_session": session_box.text().strip() or self.session_date(),
                "source_model": model_box.text().strip(),
                "created_at_claimed": created_box.text().strip(),
            }
            dialog.accept()
            on_accept(values)

        buttons.accepted.connect(_accept)
        form.addRow(buttons)
        # Handles for tests; the dialog owns them.
        dialog.text_box, dialog.session_box = text_box, session_box
        dialog.model_box, dialog.buttons = model_box, buttons
        dialog.show()
        return dialog

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
    # -- "Review my day" -----------------------------------------------------
    def walk_is_open(self) -> bool:
        return self._walk is not None and self.body_stack.currentWidget() is self._walk

    def walk(self):
        """The open (or last) walk, or None."""
        return self._walk

    def _walk_ready_for(self, session: str, payload: Mapping[str, Any]) -> bool:
        """A closed session, the newest one, whose walk is not finished."""
        if not session or payload.get("provisional") or str(payload.get("session_date") or "") != session:
            return False
        try:
            import market_calendar

            if market_calendar.last_completed_session(self._clock()).isoformat() != session:
                return False
        except Exception:  # noqa: BLE001 - an unsure calendar shows no nudge
            return False
        from ui.widgets.recap_walk import walk_finished

        return not walk_finished(session)

    def _sync_walk_entry(self, session: str, payload: Mapping[str, Any]) -> None:
        """Enable the button for a loaded day; show the banner only after the close."""
        loaded = str(payload.get("session_date") or "") == session and bool(session)
        self.walk_button.setEnabled(loaded)
        self.show_button.setEnabled(loaded)
        ready = self._walk_ready_for(session, payload)
        self.walk_banner.setVisible(ready and session not in self._walk_banner_dismissed)
        if ready != self._walk_ready:
            self._walk_ready = ready
            self.walkReadyChanged.emit(ready)

    def _dismiss_walk_banner(self) -> None:
        self._walk_banner_dismissed.add(self.session_date())
        self.walk_banner.setVisible(False)

    def open_walk(self) -> None:
        """Replace the page body with the walk for the loaded session."""
        session = self.session_date()
        if str(self._payload.get("session_date") or "") != session:
            self.status.setText("The day is still loading - try again in a moment.")
            return
        walk = self._walk
        if walk is None or walk.session != session:
            if walk is not None:
                self._retire_walk(walk)
            from ui.widgets.recap_walk import RecapWalk

            walk = RecapWalk(session, self._payload, self, zone=self._zone)
            walk.exited.connect(self.close_walk)
            walk.walkFinished.connect(self._on_walk_finished)
            walk.statusChanged.connect(self.statusChanged.emit)
            self._walk = walk
            self.body_stack.addWidget(walk)
            walk.start()
        self.body_stack.setCurrentWidget(walk)
        walk.setFocus(Qt.FocusReason.OtherFocusReason)

    # -- R1 Day Review Show --------------------------------------------------
    def open_show(self) -> None:
        """Lay the loaded day's deck over the whole window (chosen on the worker)."""
        session = self.session_date()
        if str(self._payload.get("session_date") or "") != session:
            self.status.setText("The day is still loading - try again in a moment.")
            return
        chosen = self._payload.get("show")
        if not isinstance(chosen, Mapping) or not chosen.get("deck"):
            import day_review_show

            chosen = {
                "deck": day_review_show.fallback_deck(None, session_date=session),
                "facts_only": True,
                "reason": "the day's show was not read",
                "model": "",
            }
        from ui.widgets.day_review_show_overlay import DayReviewShow

        self.close_show()
        overlay = DayReviewShow(chosen, spy_bars=self._payload.get("spy_m5_bars") or ())
        overlay.closed.connect(self.close_show)
        self._show = overlay
        overlay.cover(self.window())

    def show_overlay(self):
        """The open Day Review Show overlay, or None."""
        return self._show

    def close_show(self) -> None:
        overlay, self._show = self._show, None
        if overlay is not None:
            overlay.hide()
            overlay.deleteLater()

    def close_walk(self) -> None:
        """Back to the page. The walk stays built, so reopening resumes it."""
        self.body_stack.setCurrentWidget(self.scroll)

    def _retire_walk(self, walk) -> None:
        """Drop an old session's walk once its saves have finished."""
        self.body_stack.removeWidget(walk)
        walk.setParent(None)
        self._retired_walks = [w for w in self._retired_walks if w.busy()]
        if walk.busy():
            self._retired_walks.append(walk)
        else:
            walk.deleteLater()

    def _on_walk_finished(self, session: str) -> None:
        if session == self.session_date():
            self.walk_banner.setVisible(False)
            if self._walk_ready:
                self._walk_ready = False
                self.walkReadyChanged.emit(False)

    # -- saved clues on the page's charts ---------------------------------------
    def _draw_saved_clues(self, chart, attr: str, symbol: str) -> None:
        """Draw the session's saved clues for `symbol` on `chart`, read on a worker."""
        try:
            from ui.widgets.clue_marker import mark_clue_flow
        except Exception:  # noqa: BLE001 - no clue module, no clues drawn
            return
        try:
            flow = getattr(self, attr)
            if flow is None:
                flow = mark_clue_flow(chart, self.session_date(), symbol, "M5")
                setattr(self, attr, flow)
            flow.session_date = self.session_date()
            flow.symbol = str(symbol or "").upper()
            flow.clues = []
            flow.redraw()
            flow.load()
        except Exception:  # noqa: BLE001 - a clue never costs the page
            logging.debug("Saved clues were not drawn.", exc_info=True)

    def shutdown(self) -> None:
        try:
            self._auto_timer.stop()
        except RuntimeError as swallowed_exc:  # pragma: no cover - already torn down
            note_swallowed("day review timer already torn down", swallowed_exc, quiet=True)
        for worker in (self._worker, self._index_worker, self._bars_worker):
            if worker is not None and worker.isRunning():
                # Bounded: a desk that will not close is worse than an index
                # nobody collected, and the index is rebuildable.
                worker.wait(2000)
        # The ideas card's write is the trader's own click, so it is waited for
        # (bounded) rather than abandoned.
        try:
            self.ideas_card.shutdown()
        except Exception as swallowed_exc:  # noqa: BLE001 - shutdown must not raise
            note_swallowed("day review ideas card shutdown failed", swallowed_exc)
        try:
            self.plan_view.shutdown()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            note_swallowed("day review plan view shutdown failed", exc)


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
    "REPORT_CARD_LINE_OBJECT_NAME",
    "REPORT_CARD_OBJECT_NAME",
    "REPORT_CARD_PLACEHOLDERS",
    "REPORT_CARD_WRAP_CHARS",
    "SAID_SPLIT_KEY",
    "SAID_SPLIT_WEIGHTS",
    "SPY_MIN_HEIGHT_PX",
    "STORY_MIN_HEIGHT_PX",
    "GLANCE_TARGETS",
    "MISS_DEFAULT_COLUMNS",
    "MISS_FILTERS",
    "SESSION_SETTING_KEY",
    "TJ2B_WALKAWAY_COLUMNS",
    "TRADE_COLUMNS",
    "WALKAWAY_TITLE",
]
