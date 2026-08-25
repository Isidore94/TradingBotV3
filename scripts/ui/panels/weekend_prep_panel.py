"""Weekend Prep: a guided five-step routine (R8 §9 steps 6-9, spec §1/§3/§6/§7).

A stepper rail down the left, one page per step on the right, and progress that
survives closing the app. The five steps are the trader's actual weekend ritual
in the order they do it: what happened, how the picks behaved, how the exits
went, what looks strong now, and what is coming.

Two things this tab deliberately does not do:

* **It never refreshes by itself.** Every fetch is behind a button. The service
  owns no timer, and the weekend quiet-hours gate is untouched.
* **It never removes anything.** Adopt adds to swing Focus through the existing
  membership-tracked injection. There is no remove path in this file at all, and
  a test asserts that rather than trusting it - the trader's own names stay the
  trader's.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import weekend_strength
from ui.services import journal_feed
from ui.services.weekend_prep_service import STEP_IDS, STEP_LABELS, WeekendPrepService

#: How many folded RS/RW rows one bucket may show in the week review. A cap
#: keeps the step readable; what it drops is printed, because a silent
#: top-N reads as "that was all of it".
RRS_ROWS_PER_BUCKET = 8

STATUS_MARKS = {"pending": "○", "done": "●", "skipped": "–"}


class _WalkawayWorker(QThread):
    finished_with = Signal(dict)
    failed = Signal(str)

    def __init__(self, since, until, parent=None) -> None:
        super().__init__(parent)
        self._since = since
        self._until = until

    def run(self) -> None:  # pragma: no cover - exercised through its signal seam
        try:
            self.finished_with.emit(journal_feed.walkaway_summary(self._since, self._until))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _StepPage(QFrame):
    """Shared furniture: a title, a body, and the two buttons every step has."""

    statusChanged = Signal(str)

    def __init__(self, step_id: str, service: WeekendPrepService, parent=None) -> None:
        super().__init__(parent)
        self.step_id = step_id
        self.service = service

        self.heading = QLabel(STEP_LABELS[step_id])
        self.heading.setObjectName("StepHeading")
        self.subtitle = QLabel("")
        self.subtitle.setWordWrap(True)

        self.done_button = QPushButton("Mark done")
        self.done_button.clicked.connect(lambda: self._set_status("done"))
        self.skip_button = QPushButton("Skip this week")
        self.skip_button.clicked.connect(lambda: self._set_status("skipped"))

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self.heading)
        self._layout.addWidget(self.subtitle)

    def _finish_layout(self) -> None:
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(self.skip_button)
        footer.addWidget(self.done_button)
        self._layout.addLayout(footer)

    def _set_status(self, status: str) -> None:
        self.service.set_step_status(self.step_id, status)
        self.statusChanged.emit(f"{STEP_LABELS[self.step_id]} {status}")

    def reload(self) -> None:  # pragma: no cover - overridden
        pass


class WeekReviewPage(_StepPage):
    """Step 1: what happened, from the review-learning state and the RS extremes."""

    def __init__(self, service, parent=None) -> None:
        super().__init__("week_review", service, parent)
        monday, friday = service.week_bounds
        self.subtitle.setText(f"Week of {monday} to {friday}. Refresh reads the week's decisions.")
        self.refresh_button = QPushButton("Refresh week")
        self.refresh_button.clicked.connect(self.reload)
        self.summary = QTextBrowser()
        self._layout.addWidget(self.refresh_button)
        self._layout.addWidget(self.summary, 1)
        self._finish_layout()

    def reload(self) -> None:
        try:
            from review_learning import build_review_learning_state

            state = build_review_learning_state(window_days=7)
        except Exception as exc:  # noqa: BLE001
            self.summary.setPlainText(f"Week review unavailable: {exc}")
            self.statusChanged.emit(f"week review unavailable: {exc}")
            return
        lines = [f"Week of {self.service.week_bounds[0]} to {self.service.week_bounds[1]}", ""]
        for key in ("takes", "skips", "rejects", "blind_spots", "leaks", "watch_conversion"):
            value = state.get(key) if isinstance(state, dict) else None
            if value is None:
                continue
            lines.append(f"{key.replace('_', ' ').title()}: {value if not isinstance(value, list) else len(value)}")
        # The recorded, accepted v1 limitation - stated where it is read, not
        # buried in a spec nobody opens on a Saturday.
        lines += ["", "Episodes fold on (trade_date, symbol): two setups in one name on one day read as one."]
        lines += self._rrs_lines()
        self.summary.setPlainText("\n".join(lines))


    def _rrs_lines(self) -> list[str]:
        """The week's RS/RW extremes beside the decisions (R8 retained scope).

        Folded per symbol, and capped, because the step is a review rather than
        a log dump - and what the cap drops is SAID, since a silent top-N reads
        as "that was all of it".
        """
        rows = _read_rrs_week(self.service.week_bounds)
        if not rows:
            return ["", "RS/RW extremes: nothing recorded this week (or the log is unreadable)."]
        lines = ["", "RS/RW extremes this week (folded per symbol):"]
        shown = 0
        for bucket in sorted({row["bucket"] for row in rows}):
            in_bucket = [row for row in rows if row["bucket"] == bucket]
            lines.append(f"  {bucket}: {len(in_bucket)} name(s)")
            for row in in_bucket[:RRS_ROWS_PER_BUCKET]:
                lines.append(
                    f"    {row['symbol']}: {row['days']} day(s), "
                    f"{row['sightings']} sighting(s), best RRS {row['best_rrs']:+.2f}, "
                    f"last {row['last_seen']}"
                )
                shown += 1
            if len(in_bucket) > RRS_ROWS_PER_BUCKET:
                lines.append(
                    f"    ... {len(in_bucket) - RRS_ROWS_PER_BUCKET} more in this bucket "
                    "not shown"
                )
        lines.append(f"  ({shown} of {len(rows)} folded rows shown.)")
        return lines


class FocusReviewPage(_StepPage):
    """Step 2: how the week's focus picks behaved."""

    def __init__(self, service, parent=None) -> None:
        super().__init__("focus_review", service, parent)
        self.subtitle.setText(
            "The week's focus picks, their outcomes, and both graded cohorts "
            "beside them - what you vetoed and what you liked."
        )
        self.refresh_button = QPushButton("Refresh picks")
        self.refresh_button.clicked.connect(self.reload)
        # One row per PICK, carrying its outcome - not picks and outcomes as
        # separate rows, which listed the same name twice and could not answer
        # "how did this pick do" (R8 retained scope, built 2026-08-18).
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["Date", "Symbol", "Side", "Source", "H1", "H3", "H5", "H10", "Matured"]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.note = QLabel("")
        self.note.setWordWrap(True)

        # AI-P1: the mirror cohort, which is what the subtitle has promised
        # since this step shipped. NOT week-scoped, unlike the table above -
        # the cohort is the whole graded record of what the trader threw away,
        # and a single week of it would answer nothing.
        self.cohort_caption = QLabel("")
        self.cohort_caption.setObjectName("SectionSubtitle")
        self.cohort_caption.setWordWrap(True)
        self.cohort_table = QTableWidget(0, 6)
        self.cohort_table.setHorizontalHeaderLabels(
            ["Veto reason", "Side", "n", "Win rate", "Avg return", "PF"]
        )
        self.cohort_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cohort_note = QLabel("")
        self.cohort_note.setWordWrap(True)

        # Packet 8b: R10.F's LIKE cohort, beside the veto one. The two are the
        # halves of one judgement - what you threw away and what you endorsed -
        # and reading either alone gives half an answer.
        self.like_table = QTableWidget(0, 6)
        self.like_table.setHorizontalHeaderLabels(
            ["Claimed setup", "Side", "n", "Win rate", "Avg return", "PF"]
        )
        self.like_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.like_note = QLabel("")
        self.like_note.setWordWrap(True)

        self._layout.addWidget(self.refresh_button)
        self._layout.addWidget(self.table, 1)
        self._layout.addWidget(self.note)
        self._layout.addWidget(QLabel("Vetoed picks, graded forward"))
        self._layout.addWidget(self.cohort_caption)
        self._layout.addWidget(self.cohort_table, 1)
        self._layout.addWidget(self.cohort_note)
        self._layout.addWidget(QLabel("Liked picks, graded forward"))
        self._layout.addWidget(self.like_table, 1)
        self._layout.addWidget(self.like_note)
        self._finish_layout()

    def reload(self) -> None:
        # The cohort first and unconditionally: it is not week-scoped, so a
        # week with no picks must not also hide the graded record of the
        # trader's vetoes.
        self._reload_cohort()
        self._reload_like_cohort()
        rows = _join_focus_week(self.service.week_bounds)
        self.table.setRowCount(len(rows))
        columns = ("date", "symbol", "side", "source", "h1", "h3", "h5", "h10", "matured")
        for index, row in enumerate(rows):
            for column, key in enumerate(columns):
                self.table.setItem(index, column, QTableWidgetItem(str(row.get(key) or "")))
        if not rows:
            self.note.setText(
                "No focus picks recorded for this week, or the CSVs are not readable."
            )
            return
        matured = sum(1 for row in rows if str(row.get("matured") or "").strip() not in ("", "0"))
        orphans = sum(1 for row in rows if "no pick snapshot" in str(row.get("source") or ""))
        note = (
            f"{len(rows)} pick(s) in the reviewed week; {matured} have at least one "
            "matured horizon. A blank horizon has not matured yet - it is not a zero."
        )
        if orphans:
            note += f" {orphans} row(s) have an outcome but no pick snapshot."
        self.note.setText(note)

    def _reload_like_cohort(self) -> None:
        """R10.F's cohort, rendered under the same honesty rules as the veto one."""
        rows = _read_like_cohort()
        self.like_table.setRowCount(len(rows))
        columns = ("cohort", "side", "n", "win_rate", "avg_return", "profit_factor")
        for index, row in enumerate(rows):
            for column, key in enumerate(columns):
                self.like_table.setItem(
                    index, column, QTableWidgetItem(str(row.get(key) or ""))
                )
        if not rows:
            self.like_note.setText(
                "No graded LIKE cohort yet. It is written by the overnight "
                "like_cohort_grading slot, and a claim needs forward sessions "
                "before it means anything - this is an absent measurement, not "
                "an empty record."
            )
            return
        self.like_note.setText(
            f"{len(rows)} cohort row(s), one per claimed setup family. Returns are "
            "side-adjusted, so POSITIVE means the pick you liked WORKED - the "
            "opposite reading from the veto table above, where positive means the "
            "one you rejected would have. Read as DISCOVERY, not confirmation: "
            "n is small per family and a blank is a horizon that has not matured. "
            "The two tables are the mirror pair - what you threw away, and what "
            "you endorsed."
        )

    def _reload_cohort(self) -> None:
        rows = _read_veto_cohort()
        self.cohort_table.setRowCount(len(rows))
        columns = ("cohort", "side", "n", "win_rate", "avg_return", "profit_factor")
        for index, row in enumerate(rows):
            for column, key in enumerate(columns):
                self.cohort_table.setItem(
                    index, column, QTableWidgetItem(str(row.get(key) or ""))
                )
        if not rows:
            self.cohort_caption.setText("")
            self.cohort_note.setText(
                "No graded veto cohort yet. It is written by the overnight "
                "veto_cohort_grading slot, and a veto needs forward sessions "
                "before it means anything - this is an absent measurement, "
                "not a clean record."
            )
            return
        # The label is fixed text, not a computed claim: it describes what the
        # rows ARE. Nothing here derives a statistic the CSV does not carry
        # (plan.md Phase 0.7 ground rule 6).
        horizons = sorted({str(row.get("horizon") or "").strip() for row in rows} - {""})
        span = ", ".join(horizons) if horizons else "unstated"
        self.cohort_caption.setText(
            f"Session horizon(s): {span}. Returns are side-adjusted, so POSITIVE "
            "means the pick you vetoed would have WORKED. Read as DISCOVERY, not "
            "confirmation: these are the trader's own rejections graded forward, "
            "n is small per reason, and a blank is a horizon that has not matured."
        )
        self.cohort_note.setText(
            f"{len(rows)} cohort row(s). Two capture facts travel with this "
            "table: 'Veto D1 - but M5 today' writes an ordinary veto row, so "
            "some of these names were day-traded the same day; and a reason "
            "introduced by a later vocabulary keeps its own cohort rather than "
            "back-filling the older one."
        )


class WalkawayPage(_StepPage):
    """Step 3: how the exits went, plus the weekly auto-tag review."""

    def __init__(self, service, parent=None) -> None:
        super().__init__("walkaway", service, parent)
        self.subtitle.setText(
            "Trades CLOSED inside the reviewed week. A position opened this week and still "
            "open is listed separately - it has no exit to learn from yet."
        )
        self.refresh_button = QPushButton("Run walk-away for this week")
        self.refresh_button.clicked.connect(self.reload)
        self.output = QTextBrowser()
        self.open_note = QLabel("")
        self.open_note.setWordWrap(True)

        self.tag_table = QTableWidget(0, 4)
        self.tag_table.setHorizontalHeaderLabels(["Date", "Symbol", "My tags", "Suggested"])
        self.tag_table.setEditTriggers(QTableWidget.NoEditTriggers)
        # AI-P2, trader-approved amendment 2026-08-24. R8 locked the journal
        # hook to the weekly review and that stays the DEFAULT; this opens the
        # whole backlog on request, because 220 proposals against one confirmed
        # annotation can never drain at a weekly trickle.
        self.backlog_toggle = QCheckBox("Show all pending proposals, not just this week")
        self.backlog_toggle.setChecked(False)
        self.backlog_toggle.toggled.connect(lambda _checked: self._reload_tags())
        self.tag_note = QLabel("")
        self.tag_note.setWordWrap(True)
        self.confirm_button = QPushButton("Confirm suggested tag")
        self.confirm_button.clicked.connect(self._confirm_tag)
        self.correct_button = QPushButton("Correct to my tags")
        self.correct_button.clicked.connect(self._correct_tag)
        self._tag_rows: list[dict[str, Any]] = []
        self._worker: _WalkawayWorker | None = None

        tag_buttons = QHBoxLayout()
        tag_buttons.addWidget(self.confirm_button)
        tag_buttons.addWidget(self.correct_button)
        tag_buttons.addStretch(1)

        self._layout.addWidget(self.refresh_button)
        self._layout.addWidget(self.output, 2)
        self._layout.addWidget(self.open_note)
        self._layout.addWidget(QLabel("Auto-tag review"))
        self._layout.addWidget(self.backlog_toggle)
        self._layout.addWidget(self.tag_table, 2)
        self._layout.addWidget(self.tag_note)
        self._layout.addLayout(tag_buttons)
        self._finish_layout()

    def reload(self) -> None:
        monday, friday = self.service.week_bounds
        if self._worker is None or not self._worker.isRunning():
            self.refresh_button.setEnabled(False)
            self.output.setPlainText("Running walk-away...")
            self._worker = _WalkawayWorker(monday, friday, self)
            self._worker.finished_with.connect(self._on_walkaway_done)
            self._worker.failed.connect(self._on_walkaway_failed)
            self._worker.start()
        self._reload_review_data()

    def _on_walkaway_done(self, result: dict) -> None:  # pragma: no cover - Qt signal seam
        self.refresh_button.setEnabled(True)
        self.output.setPlainText(journal_feed.render_walkaway_summary(result))

    def _on_walkaway_failed(self, message: str) -> None:  # pragma: no cover - Qt signal seam
        self.refresh_button.setEnabled(True)
        self.output.setPlainText(f"Walk-away unavailable: {message}")
        self.statusChanged.emit(f"walk-away failed: {message}")

    def _reload_review_data(self) -> None:
        monday, friday = self.service.week_bounds

        try:
            split = journal_feed.week_trades(monday, friday)
            still_open = split["still_open"]
            self.open_note.setText(
                f"{len(split['closed'])} closed this week."
                + (
                    f" {len(still_open)} opened this week and still open: "
                    + ", ".join(t.symbol for t in still_open)
                    + " - flagged, not counted."
                    if still_open
                    else ""
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.open_note.setText(f"Journal unavailable: {exc}")
        self._reload_tags()

    def _reload_tags(self) -> None:
        """Fill the auto-tag list from whichever scope is selected.

        Its own method so toggling the backlog never replays the walk-away:
        that is a market-history run behind a worker thread, and the tag list
        is a database read. Same reason `_confirm_tag` does not reload.
        """
        monday, friday = self.service.week_bounds
        try:
            if self.backlog_toggle.isChecked():
                self._tag_rows = journal_feed.pending_tag_candidates()
            else:
                self._tag_rows = journal_feed.week_tag_candidates(monday, friday)
        except Exception as exc:  # noqa: BLE001
            self.tag_note.setText(f"Auto-tag proposals unavailable: {exc}")
            self._tag_rows = []

        self.tag_table.setRowCount(len(self._tag_rows))
        for index, row in enumerate(self._tag_rows):
            suggested = "; ".join(str(c.get("tag")) for c in row["candidates"][:3])
            for column, text in enumerate((row["trade_date"], row["symbol"], row["current_tags"], suggested)):
                self.tag_table.setItem(index, column, QTableWidgetItem(str(text)))
        self.tag_note.setText(self._tag_note_text())

    def _tag_note_text(self) -> str:
        shown = len(self._tag_rows)
        if not self.backlog_toggle.isChecked():
            return f"{shown} proposal(s) from the reviewed week."
        if not shown:
            return "No pending auto-tag proposals in the backlog."
        total = int(self._tag_rows[0].get("backlog_total") or shown)
        tagged = sum(1 for row in self._tag_rows if row.get("already_tagged"))
        note = f"Backlog: showing {shown} of {total} pending proposal(s), newest first."
        if total > shown:
            # Said, never silent: a top-N the reader cannot see reads as
            # "that was all of it".
            note += f" {total - shown} more not shown - confirm these and refresh."
        if tagged:
            note += (
                f" {tagged} of these already carry your tags; accepting a"
                " suggestion does not remove its proposal, so they keep"
                " appearing until you tag past them."
            )
        return note

    def _selected_tag_row(self) -> dict[str, Any] | None:
        index = self.tag_table.currentRow()
        if index < 0 or index >= len(self._tag_rows):
            self.statusChanged.emit("select a trade in the auto-tag list first")
            return None
        return self._tag_rows[index]

    def _confirm_tag(self) -> None:
        row = self._selected_tag_row()
        if row is None:
            return
        tags = [str(c.get("tag")) for c in row["candidates"][:1]]
        journal_feed.accept_auto_tags(row["trade_id"], tags)
        self.service.record_tag_review(row["trade_id"])
        self.statusChanged.emit(f"confirmed {tags[0]} on {row['symbol']}")
        self._reload_review_data()

    def _correct_tag(self) -> None:
        row = self._selected_tag_row()
        if row is None:
            return
        journal_feed.correct_auto_tag(row["trade_id"], row["current_tags"])
        self.service.record_tag_review(row["trade_id"], corrected_to=row["current_tags"])
        self.statusChanged.emit(f"corrected {row['symbol']} to your tags")
        self._reload_review_data()

    def shutdown(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait()


class DiscoveryPage(_StepPage):
    """Step 4: strongest and weakest on H1, D1 and Monthly, then Adopt."""

    def __init__(self, service, parent=None, *, focus_service=None) -> None:
        super().__init__("discovery", service, parent)
        self.subtitle.setText(
            "Same strength formula as the M5 board, on three timeframes. Every refresh is "
            "manual and uses batched yfinance - no IB traffic."
        )
        self._focus_service = focus_service
        self.tabs = QTabWidget()
        self._boards: dict[str, dict[str, Any]] = {}
        for timeframe in weekend_strength.TIMEFRAMES:
            self.tabs.addTab(self._build_board_tab(timeframe), timeframe.label)
        self._layout.addWidget(self.tabs, 1)
        self._finish_layout()

        service.boardChanged.connect(self._on_board_changed)
        service.boardFailed.connect(self.show_failure)

    def _build_board_tab(self, timeframe) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        long_button = QPushButton("Refresh strongest")
        long_button.clicked.connect(lambda: self._refresh(timeframe.key, "long"))
        short_button = QPushButton("Refresh weakest")
        short_button.clicked.connect(lambda: self._refresh(timeframe.key, "short"))
        adopt_button = QPushButton("Adopt selected to swing Focus")
        adopt_button.clicked.connect(lambda: self._adopt(timeframe.key))

        buttons = QHBoxLayout()
        buttons.addWidget(long_button)
        buttons.addWidget(short_button)
        buttons.addStretch(1)
        buttons.addWidget(adopt_button)

        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(["Symbol", "Side", "Score", "Last", "Bars"])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        accounting = QLabel("Not refreshed yet.")
        accounting.setWordWrap(True)
        banner = QLabel("")
        banner.setObjectName("BoardFailureBanner")
        banner.setWordWrap(True)
        banner.setVisible(False)

        layout.addLayout(buttons)
        layout.addWidget(banner)
        layout.addWidget(table, 1)
        layout.addWidget(accounting)
        self._boards[timeframe.key] = {"table": table, "accounting": accounting, "banner": banner}
        return page

    def _refresh(self, timeframe: str, side: str) -> None:
        started = self.service.refresh_board(timeframe, side=side)
        if not started:
            self.statusChanged.emit(f"{timeframe} {side} board is already refreshing")

    def _on_board_changed(self, timeframe: str) -> None:
        board = self.service.board(timeframe)
        widgets = self._boards.get(timeframe)
        if board is None or widgets is None:
            return
        widgets["banner"].setVisible(False)
        table = widgets["table"]
        table.setRowCount(len(board.rows))
        for index, row in enumerate(board.rows):
            values = (
                row["symbol"],
                row["side"],
                f"{row['score']:.2f}",
                f"{row['last_close']:.2f}" if row.get("last_close") is not None else "-",
                row.get("bar_count", ""),
            )
            for column, text in enumerate(values):
                table.setItem(index, column, QTableWidgetItem(str(text)))
        widgets["accounting"].setText(f"{board.accounting}. As of {board.as_of}.")

    def show_failure(self, timeframe: str, message: str) -> None:
        """Keep the last good board and say what went wrong above it."""
        widgets = self._boards.get(timeframe)
        if widgets is None:
            return
        widgets["banner"].setText(f"{message} — showing the last good board.")
        widgets["banner"].setVisible(True)

    def _adopt(self, timeframe: str) -> None:
        widgets = self._boards.get(timeframe)
        board = self.service.board(timeframe)
        if widgets is None or board is None:
            return
        index = widgets["table"].currentRow()
        if index < 0 or index >= len(board.rows):
            self.statusChanged.emit("select a row to adopt")
            return
        row = board.rows[index]
        confirmed = QMessageBox.question(
            self,
            "Adopt to swing Focus?",
            f"Add {row['symbol']} ({row['side']}) to swing Focus and the swing watchlist?\n\n"
            "This adds only. Nothing in Weekend Prep removes an entry.",
        )
        if confirmed != QMessageBox.Yes:
            return
        service = self._focus_service or _default_focus_service()
        if service is None:
            self.statusChanged.emit("Focus service unavailable")
            return
        added = service.add(
            row["symbol"],
            row["side"],
            "swing",
            origin="weekend_prep",
            context=f"weekend_prep:{timeframe}:{self.service.weekend}",
        )
        self.service.record_adopted(row["symbol"], row["side"], timeframe)
        # A duplicate add is not an error: the trader may already hold the name,
        # and saying so is more useful than refusing.
        self.statusChanged.emit(
            f"adopted {row['symbol']} to swing Focus"
            if added
            else f"{row['symbol']} was already on the swing list"
        )


class WeekAheadPage(_StepPage):
    """Step 5: the forward-looking weekly prep."""

    def __init__(self, service, parent=None) -> None:
        super().__init__("week_ahead", service, parent)
        self.subtitle.setText("Earnings, economic calendar and risk windows for the week ahead.")
        self.refresh_button = QPushButton("Build week-ahead prep")
        self.refresh_button.clicked.connect(self.reload)
        self.report = QTextBrowser()
        self.report.setMarkdown("Not built yet for this weekend.")
        self._layout.addWidget(self.refresh_button)
        self._layout.addWidget(self.report, 1)
        self._finish_layout()
        service.weekAheadReady.connect(self._render)

    def reload(self) -> None:
        if not self.service.refresh_week_ahead():
            self.statusChanged.emit("week ahead is already building")

    def _render(self, markdown: str) -> None:
        self.report.setMarkdown(markdown or "The weekly prep returned nothing.")


class WeekendPrepPanel(QFrame):
    """The stepper rail and the five pages."""

    statusChanged = Signal(str)

    def __init__(self, parent=None, *, service: WeekendPrepService | None = None, focus_service=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.service = service or WeekendPrepService(self)

        self.rail = QListWidget()
        self.rail.setObjectName("StepRail")
        self.rail.setMaximumWidth(240)
        self.rail.currentRowChanged.connect(self._on_step_selected)

        self.pages = QStackedWidget()
        self.week_review = WeekReviewPage(self.service)
        self.focus_review = FocusReviewPage(self.service)
        self.walkaway = WalkawayPage(self.service)
        self.discovery = DiscoveryPage(self.service, focus_service=focus_service)
        self.week_ahead = WeekAheadPage(self.service)
        self._pages = {
            "week_review": self.week_review,
            "focus_review": self.focus_review,
            "walkaway": self.walkaway,
            "discovery": self.discovery,
            "week_ahead": self.week_ahead,
        }
        for step in STEP_IDS:
            page = self._pages[step]
            page.statusChanged.connect(self.statusChanged)
            self.pages.addWidget(page)

        self.header = QLabel("")
        self.header.setObjectName("WeekendHeader")

        body = QHBoxLayout()
        body.addWidget(self.rail)
        body.addWidget(self.pages, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.header)
        layout.addLayout(body, 1)

        self.service.stateChanged.connect(self._refresh_rail)
        self.service.statusChanged.connect(self.statusChanged)
        self._refresh_rail()
        self.rail.setCurrentRow(0)

    def _refresh_rail(self) -> None:
        current = self.rail.currentRow()
        self.rail.blockSignals(True)
        self.rail.clear()
        for step in STEP_IDS:
            status = self.service.step_status(step)
            item = QListWidgetItem(f"{STATUS_MARKS.get(status, '○')}  {STEP_LABELS[step]}")
            item.setData(Qt.UserRole, step)
            self.rail.addItem(item)
        self.rail.blockSignals(False)
        if 0 <= current < self.rail.count():
            self.rail.setCurrentRow(current)
        monday, friday = self.service.week_bounds
        done = "complete" if self.service.routine_complete else "in progress"
        self.header.setText(f"Weekend of {self.service.weekend} — reviewing {monday} to {friday} — {done}")

    def _on_step_selected(self, row: int) -> None:
        if 0 <= row < self.pages.count():
            self.pages.setCurrentIndex(row)

    def shutdown(self) -> None:
        self.walkaway.shutdown()
        self.service.shutdown()


# ---------------------------------------------------------------------------
# Small readers, kept out of the widgets so the panel stays about layout
# ---------------------------------------------------------------------------


def _default_focus_service():
    try:
        from ui.services.focus_service import FocusService

        return FocusService()
    except Exception:  # pragma: no cover - only on a broken install
        return None


def _read_rrs_week(bounds) -> list[dict[str, Any]]:
    """The week's RS/RW extremes, folded to one row per symbol and bucket.

    R8's retained future scope, built 2026-08-18. The extremes log is one row
    per sighting, so a name that led the tape all week would otherwise bury
    every other name in the review. Folding to (bucket, symbol) with a count
    and the best reading answers the question the step actually asks - "who
    was strong this week, and how consistently" - instead of reprinting the
    log.

    Read-only and forgiving, like every other weekend-prep reader: a missing
    or unreadable CSV is a quieter week, not an error worth stopping on.
    """
    import csv

    from project_paths import RRS_STRENGTH_LOG_FILE

    monday, friday = bounds
    folded: dict[tuple[str, str], dict[str, Any]] = {}
    path = Path(RRS_STRENGTH_LOG_FILE)
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                stamp = str(raw.get("timestamp_local") or "")[:10]
                symbol = str(raw.get("symbol") or "").strip().upper()
                if not stamp or not symbol:
                    continue
                try:
                    when = datetime.fromisoformat(stamp).date()
                except ValueError:
                    continue
                if not (monday <= when <= friday):
                    continue
                bucket = str(raw.get("bucket") or "").strip() or "unknown"
                try:
                    rrs = float(raw.get("rrs") or 0.0)
                except (TypeError, ValueError):
                    continue
                key = (bucket, symbol)
                entry = folded.get(key)
                if entry is None:
                    folded[key] = {
                        "bucket": bucket,
                        "symbol": symbol,
                        "sightings": 1,
                        "days": {stamp},
                        "best_rrs": rrs,
                        "last_seen": stamp,
                    }
                    continue
                entry["sightings"] += 1
                entry["days"].add(stamp)
                # "Best" means most extreme in the bucket's own direction: a
                # weak-side reading is most notable when it is most negative.
                if bucket.startswith("weak"):
                    entry["best_rrs"] = min(entry["best_rrs"], rrs)
                else:
                    entry["best_rrs"] = max(entry["best_rrs"], rrs)
                entry["last_seen"] = max(entry["last_seen"], stamp)
    except OSError:
        return []
    rows = []
    for entry in folded.values():
        rows.append(
            {
                "bucket": entry["bucket"],
                "symbol": entry["symbol"],
                "days": len(entry["days"]),
                "sightings": entry["sightings"],
                "best_rrs": round(float(entry["best_rrs"]), 4),
                "last_seen": entry["last_seen"],
            }
        )
    rows.sort(key=lambda row: (row["bucket"], -row["days"], -abs(row["best_rrs"])))
    return rows


def _join_focus_week(bounds) -> list[dict[str, Any]]:
    """The week's focus picks JOINED to their outcomes - one row per pick.

    R8's retained future scope, built 2026-08-18. The v1 step listed picks and
    outcomes as separate rows, so the same name appeared twice and the table
    could not answer "how did this pick do". The join is on
    (trade_date, symbol, side), which is the identity both CSVs are written
    with.

    An outcome with no matching pick is still shown - it is evidence that a
    pick existed on a day whose snapshot did not persist, and dropping it
    would quietly narrow the week. It is marked so the reader can tell.
    """
    import csv

    import project_paths

    monday, friday = bounds

    def _rows(path) -> list[dict[str, str]]:
        path = Path(path)
        if not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        except OSError:
            return []

    def _key(raw) -> tuple[str, str, str] | None:
        stamp = str(raw.get("trade_date") or raw.get("date") or "")[:10]
        symbol = str(raw.get("symbol") or "").strip().upper()
        if not stamp or not symbol:
            return None
        try:
            when = datetime.fromisoformat(stamp).date()
        except ValueError:
            return None
        if not (monday <= when <= friday):
            return None
        return (stamp, symbol, str(raw.get("side") or "").strip().lower())

    outcomes: dict[tuple[str, str, str], dict[str, str]] = {}
    for raw in _rows(project_paths.HUMAN_FOCUS_OUTCOMES_FILE):
        key = _key(raw)
        if key is not None:
            outcomes[key] = raw

    joined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in _rows(project_paths.HUMAN_FOCUS_DAILY_PICKS_FILE):
        key = _key(raw)
        if key is None:
            continue
        seen.add(key)
        joined.append(_focus_row(key, raw, outcomes.get(key), orphan=False))
    for key, raw in outcomes.items():
        if key not in seen:
            joined.append(_focus_row(key, None, raw, orphan=True))
    joined.sort(key=lambda row: (row["date"], row["symbol"]))
    return joined


def _read_veto_cohort() -> list[dict[str, str]]:
    """The graded veto cohort - R8 §6's last DEFERRED join (AI-P1).

    The Focus Pick Review subtitle has promised "the veto cohort beside them"
    since the step shipped, while nothing loaded any ``veto_cohort_*`` file.
    This is the loader that sentence was describing.

    It is the MIRROR of the picks table, and that is why it belongs on this
    page: the picks answer "how did what I took do", and only the cohort can
    answer "how did what I threw away do". A trader whose accepted picks look
    fine and whose vetoes were mostly wrong has a problem that the left-hand
    table cannot show.

    The rollup on disk is already pooled by :func:`canonical_veto_cohort`
    (``_rebuild_pooled_performance`` groups by it), so applying it here is
    idempotent. It is applied anyway, through the ONE canonical function -
    never a second pooling implementation - so a later vocabulary bump cannot
    leave this pane and the rollup disagreeing about which rows are the same
    reason.

    Read-only and forgiving like every other weekend-prep reader: an
    unreadable or missing file is a quieter page, not an error worth stopping
    the routine for. Nothing here computes a statistic the CSV does not carry
    (plan.md Phase 0.7 ground rule 6) - the columns are reformatted, never
    derived.
    """
    import csv

    import project_paths
    from ui.annotations import veto_cohort

    path = Path(project_paths.VETO_COHORT_PERFORMANCE_FILE)
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except OSError:
        return []

    def _pct(value: str) -> str:
        """A percentage, or BLANK. Never a substituted zero."""
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return f"{float(text) * 100:.1f}%"
        except (TypeError, ValueError):
            return text

    def _signed_pct(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return f"{float(text) * 100:+.2f}%"
        except (TypeError, ValueError):
            return text

    def _ratio(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return f"{float(text):.2f}"
        except (TypeError, ValueError):
            return text

    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        rows.append(
            {
                "cohort": veto_cohort.canonical_veto_cohort(raw.get("cohort") or ""),
                "side": str(raw.get("side") or "").strip(),
                "horizon": str(raw.get("horizon_sessions") or "").strip(),
                "n": str(raw.get("sample_count") or "").strip(),
                "win_rate": _pct(raw.get("win_rate")),
                "avg_return": _signed_pct(raw.get("avg_side_return")),
                "profit_factor": _ratio(raw.get("profit_factor")),
            }
        )
    return rows


def _read_like_cohort() -> list[dict[str, str]]:
    """The graded LIKE cohort - R10.F's output, given its surface (packet 8b).

    The mirror of :func:`_read_veto_cohort`, and read the same way: by NAMED
    CONSTANT, never by composing a filename, because AI-P1 found this step had
    been rendering an empty table for six days from exactly that mistake.

    The two cohorts are the halves of one judgement. The veto cohort answers
    "was I right to throw that away"; this one answers "was I right to like
    it". Reading either alone gives half an answer, and the half you get is
    the flattering one if you only kept the vetoes.

    No canonical pooling here, deliberately: pooling exists because the veto
    VOCABULARY is versioned and identical reasons had to be re-joined across a
    bump. A like's cohort is its claimed setup id, which is not versioned, so
    pooling it would be machinery with nothing to do - and machinery with
    nothing to do is machinery nobody notices going wrong.
    """
    import csv

    import project_paths

    path = Path(project_paths.LIKE_COHORT_PERFORMANCE_FILE)
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except OSError:
        return []

    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        rows.append(
            {
                "cohort": str(raw.get("cohort") or "").strip(),
                "side": str(raw.get("side") or "").strip(),
                "horizon": str(raw.get("horizon_sessions") or "").strip(),
                "n": str(raw.get("sample_count") or "").strip(),
                "win_rate": _cohort_pct(raw.get("win_rate")),
                "avg_return": _cohort_signed_pct(raw.get("avg_side_return")),
                "profit_factor": _cohort_ratio(raw.get("profit_factor")),
            }
        )
    return rows


def _cohort_pct(value) -> str:
    """A percentage, or BLANK. Never a substituted zero."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return f"{float(text) * 100:.1f}%"
    except (TypeError, ValueError):
        return text


def _cohort_signed_pct(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return f"{float(text) * 100:+.2f}%"
    except (TypeError, ValueError):
        return text


def _cohort_ratio(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return f"{float(text):.2f}"
    except (TypeError, ValueError):
        return text


def _focus_row(key, pick, outcome, *, orphan: bool) -> dict[str, Any]:
    stamp, symbol, side = key

    def _return(name: str) -> str:
        if not outcome:
            return ""
        value = str(outcome.get(name) or "").strip()
        if not value:
            return ""
        try:
            return f"{float(value) * 100:+.2f}%"
        except (TypeError, ValueError):
            return value

    source = str((pick or outcome or {}).get("source") or "")
    if orphan:
        # Said plainly in the row rather than dropped: an outcome whose pick
        # snapshot is missing is a real pick with a gap in its evidence.
        source = f"{source} (no pick snapshot)".strip()
    matured = str((outcome or {}).get("matured_horizons") or "")
    return {
        "date": stamp,
        "symbol": symbol,
        "side": side,
        "source": source,
        "h1": _return("h1_return"),
        "h3": _return("h3_return"),
        "h5": _return("h5_return"),
        "h10": _return("h10_return"),
        "matured": matured,
    }


def _read_focus_week(bounds) -> list[dict[str, Any]]:
    """The week's focus picks from the CSV evidence, or an empty list.

    Read-only and forgiving: this step is a review, and a missing CSV is a
    quieter week rather than an error worth stopping the routine for.
    """
    import csv

    from project_paths import PERSISTENT_DATA_DIR

    monday, friday = bounds
    rows: list[dict[str, Any]] = []
    for name, source in (("human_focus_daily_picks.csv", "pick"), ("human_focus_outcomes.csv", "outcome")):
        path = PERSISTENT_DATA_DIR / name
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    stamp = str(raw.get("date") or raw.get("trade_date") or "")[:10]
                    if not stamp:
                        continue
                    try:
                        when = datetime.fromisoformat(stamp).date()
                    except ValueError:
                        continue
                    if not (monday <= when <= friday):
                        continue
                    rows.append(
                        {
                            "date": stamp,
                            "symbol": str(raw.get("symbol") or "").upper(),
                            "side": str(raw.get("side") or ""),
                            "source": source,
                            "outcome": str(raw.get("h5") or raw.get("outcome") or ""),
                        }
                    )
        except OSError:
            continue
    return sorted(rows, key=lambda item: (item["date"], item["symbol"]))
