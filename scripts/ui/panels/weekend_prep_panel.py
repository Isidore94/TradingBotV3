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
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from ui.read_worker import ReadWorker, join_worker
from ui.widgets.data_table import apply_width_rule_to_table_widget
from ui.services.weekend_prep_service import STEP_IDS, STEP_LABELS, WeekendPrepService

#: How many folded RS/RW rows one bucket may show in the week review. A cap
#: keeps the step readable; what it drops is printed, because a silent
#: top-N reads as "that was all of it".
RRS_ROWS_PER_BUCKET = 8
#: Same idea for the group stream, and the same rule: what a cap drops is
#: PRINTED, because a silent top-N reads as "that was all of it".
RRS_GROUP_ROWS_PER_TYPE = 8
#: A week of verdicts is a table to read, not a log to scroll.
PICK_FEEDBACK_ROWS_SHOWN = 200

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


#: One owner for the read-off-the-Qt-thread shape; the private alias is kept
#: so nothing that already refers to `_ReadWorker` has to move.
_ReadWorker = ReadWorker


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

    def shutdown(self) -> None:
        """Join this page's worker, if it has one.

        On the base rather than per page: three pages now own a reader, and a
        thread that outlives the widget it was going to update is the failure
        this pass exists to avoid creating. `getattr` because a page without a
        worker is a normal page, not a broken one.
        """
        join_worker(getattr(self, "_worker", None))


#: The callout classes, in the order they are printed. `r_gaps` is the third
#: class (P1): a segment whose taken and passed halves measure far apart even
#: though its take rate looks ordinary. It is read DEFENSIVELY - a state file
#: written by a build without it simply has no such key, and this page must
#: keep working against both.
CALLOUT_CLASSES = (
    ("blind_spots", "BLIND SPOTS - you pass on these; they measure well"),
    ("leaks", "LEAKS - you take these; they measure poorly"),
    ("r_gaps", "R GAPS - taken and passed measure far apart, whatever the take rate"),
)


def _callout_measure(entry) -> str:
    """What the callout measured, in the units it was measured in.

    A blind spot is carried by the PASSED half, a leak by the TAKEN half, and
    an r_gap by both plus their difference. Each entry carries only the keys
    its own class produced, so this reads what is there rather than assuming a
    shape - and a class this build does not know still prints its segment and
    take rate instead of vanishing.
    """
    parts = []
    if entry.get("taken_r_avg") is not None:
        parts.append(f"taken {float(entry['taken_r_avg']):+.2f}R (n={entry.get('taken_r_n', '?')})")
    if entry.get("passed_r_avg") is not None:
        parts.append(f"passed {float(entry['passed_r_avg']):+.2f}R (n={entry.get('passed_r_n', '?')})")
    if entry.get("taken_fwd_avg_pct") is not None:
        parts.append(f"taken {float(entry['taken_fwd_avg_pct']):+.1f}% (n={entry.get('taken_fwd_n', '?')})")
    if entry.get("passed_fwd_avg_pct") is not None:
        parts.append(f"passed {float(entry['passed_fwd_avg_pct']):+.1f}% (n={entry.get('passed_fwd_n', '?')})")
    if entry.get("r_difference") is not None:
        parts.append(f"gap {float(entry['r_difference']):+.2f}R")
    return "; ".join(parts)


def callout_lines(state) -> list[str]:
    """The scoreboard's callouts BY NAME, not as two integers.

    This page printed "Blind Spots: 3" and "Leaks: 1" - counts a reader can do
    nothing with, over a store that has always known which segment, how often
    it was shown, how the trader's take rate on it compared with their overall
    one, and what the two halves measured. `review_learning.render_report`
    already prints exactly that; this builds the same rows for the page that is
    actually opened on a Saturday.

    Every number comes from the state file. Nothing is computed here (ground
    rule 6), and a class the state does not carry is simply absent rather than
    reported as zero.
    """
    if not isinstance(state, dict):
        return []
    overall = state.get("overall_take_rate")
    lines: list[str] = ["", "CALLOUTS - where what you do and what it measures disagree"]
    if overall is not None:
        lines.append(f"  Your overall take rate this window: {float(overall) * 100:.0f}%")
    found = False
    for key, title in CALLOUT_CLASSES:
        entries = state.get(key)
        if entries is None:
            continue
        lines.append("")
        lines.append(f"  == {title} ==")
        if not entries:
            lines.append("    none at current sample sizes.")
            continue
        found = True
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rate = entry.get("take_rate")
            rate_text = f"{float(rate) * 100:.0f}%" if rate is not None else "n/a"
            measure = _callout_measure(entry)
            lines.append(
                f"    {entry.get('dimension', '?')}={entry.get('segment', '?')}: "
                f"take {rate_text} of {entry.get('shown', '?')} shown"
                + (f"; {measure}" if measure else "")
            )
    if not found:
        lines.append("")
        lines.append(
            "  Nothing cleared a callout threshold this window. That is a "
            "quiet week, not a clean one - the thresholds need a minimum "
            "number of shown charts before any segment may be named."
        )
    return lines


class WeekReviewPage(_StepPage):
    """Step 1: what happened, from the review-learning state and the RS extremes."""

    def __init__(self, service, parent=None) -> None:
        super().__init__("week_review", service, parent)
        monday, friday = service.week_bounds
        self.subtitle.setText(f"Week of {monday} to {friday}. Refresh reads the week's decisions.")
        self.refresh_button = QPushButton("Refresh week")
        self.refresh_button.clicked.connect(self.reload)
        self.summary = QTextBrowser()
        # A separate slot for "refreshing" and for a stated failure, so neither
        # has to be written over the content the page is already showing.
        self.refresh_note = QLabel("")
        self.refresh_note.setWordWrap(True)
        self._worker: _ReadWorker | None = None
        self._layout.addWidget(self.refresh_button)
        self._layout.addWidget(self.refresh_note)
        self._layout.addWidget(self.summary, 1)
        self._finish_layout()

    def reload(self) -> None:
        """Start the week's reads on this page's worker. Single-flight.

        This method used to BE the read: `build_review_learning_state` plus two
        RS log scans, straight through the click that selected the page. It was
        the worst measured stall on the desk - 8.45 s with the whole GUI frozen
        (fluidity capture, 2026-08-25).
        """
        if self._worker is not None and self._worker.isRunning():
            return
        self.refresh_button.setEnabled(False)
        self.refresh_note.setText("Refreshing the week...")
        worker = _ReadWorker(self._build_summary_text, self)
        worker.finished_with.connect(self._on_summary_ready)
        worker.failed.connect(self._on_summary_failed)
        self._worker = worker
        worker.start()

    def _on_summary_ready(self, text: object) -> None:  # pragma: no cover - signal seam
        self.refresh_button.setEnabled(True)
        self.refresh_note.setText("")
        self.summary.setPlainText(str(text))

    def _on_summary_failed(self, message: str) -> None:  # pragma: no cover - signal seam
        self.refresh_button.setEnabled(True)
        stated = f"Week review unavailable: {message}"
        self.refresh_note.setText(stated)
        # Last-good survives. Only an empty page shows the failure as its body,
        # because there a blank would say "no decisions this week" - which is a
        # different claim from "the store could not be read".
        if not self.summary.toPlainText().strip():
            self.summary.setPlainText(stated)
        self.statusChanged.emit(f"week review unavailable: {message}")

    def _build_summary_text(self) -> str:
        """Every store this page reads, and no widget. Runs on the worker."""
        from review_learning import build_review_learning_state

        state = build_review_learning_state(window_days=7)
        lines = [f"Week of {self.service.week_bounds[0]} to {self.service.week_bounds[1]}", ""]
        for key in ("takes", "skips", "rejects", "watch_conversion"):
            value = state.get(key) if isinstance(state, dict) else None
            if value is None:
                continue
            lines.append(f"{key.replace('_', ' ').title()}: {value if not isinstance(value, list) else len(value)}")
        # The callouts are the point of this page and used to print as two
        # integers. "Blind Spots: 3" is a number a reader cannot act on; the
        # scoreboard has always known WHICH segments and by how much, and
        # `review_learning.render_report` already prints them. This is the same
        # information as a table, built here on the worker.
        lines += callout_lines(state)
        # The recorded, accepted v1 limitation - stated where it is read, not
        # buried in a spec nobody opens on a Saturday.
        lines += ["", "Episodes fold on (trade_date, symbol): two setups in one name on one day read as one."]
        lines += self._rrs_lines()
        lines += self._rrs_group_lines()
        return "\n".join(lines)


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

    def _rrs_group_lines(self) -> list[str]:
        """The week's SECTOR and INDUSTRY extremes, beside the symbol ones.

        R8 sec 6's last retained stream, built 2026-08-24. The symbol block
        above says which names led the tape; this says which parts of the
        market they came from, which is the difference between "a strong name"
        and "a strong name in a strong group".

        The group log stamps no bucket, so both extremes are printed and the
        SIGN is what the reader reads - nothing here invents a direction the
        file never recorded.
        """
        rows = _read_rrs_group_week(self.service.week_bounds)
        if not rows:
            return [
                "",
                "Sector/industry RS extremes: nothing recorded this week (or the "
                "log is unreadable).",
            ]
        lines = ["", "Sector/industry RS extremes this week (folded per group):"]
        shown = 0
        for group_type in sorted({row["group_type"] for row in rows}):
            in_type = [row for row in rows if row["group_type"] == group_type]
            lines.append(f"  {group_type}: {len(in_type)} group(s)")
            for row in in_type[:RRS_GROUP_ROWS_PER_TYPE]:
                etf = f" ({row['etf']})" if row["etf"] and row["etf"] != row["group_key"] else ""
                lines.append(
                    f"    {row['group_key']}{etf}: {row['days']} day(s), "
                    f"{row['sightings']} sighting(s), RRS {row['min_rrs']:+.2f} to "
                    f"{row['max_rrs']:+.2f}, last {row['last_seen']}"
                )
                shown += 1
            if len(in_type) > RRS_GROUP_ROWS_PER_TYPE:
                lines.append(
                    f"    ... {len(in_type) - RRS_GROUP_ROWS_PER_TYPE} more "
                    f"{group_type} group(s) not shown"
                )
        lines.append(f"  ({shown} of {len(rows)} folded group rows shown.)")
        return lines


class FocusReviewPage(_StepPage):
    """Step 2: how the week's focus picks behaved."""

    def __init__(self, service, parent=None) -> None:
        super().__init__("focus_review", service, parent)
        self.subtitle.setText(
            "The week's focus picks and their outcomes; both graded cohorts "
            "beside them - what you vetoed and what you liked; the picks' own "
            "forward record; and the week's like/dislike verdicts."
        )
        self.refresh_button = QPushButton("Refresh picks")
        self.refresh_button.clicked.connect(self.reload)
        self._worker: _ReadWorker | None = None
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
        # P2 item 1: the robust half of ground rule 10, which these two tables
        # were dropping while the Focus performance table below already showed
        # it. One horizon at a time, chosen here, so 13 columns stay readable.
        self.cohort_horizon_input = QComboBox()
        for horizon in COHORT_HORIZONS:
            self.cohort_horizon_input.addItem(f"{horizon}-session horizon", horizon)
        self.cohort_horizon_input.setCurrentIndex(
            max(0, self.cohort_horizon_input.findData(DEFAULT_COHORT_HORIZON))
        )
        self.cohort_horizon_input.currentIndexChanged.connect(self._on_cohort_horizon_changed)

        self.cohort_table = QTableWidget(0, len(COHORT_TABLE_COLUMNS))
        self.cohort_table.setHorizontalHeaderLabels(
            ["Veto reason", *COHORT_TABLE_HEADERS[1:]]
        )
        self.cohort_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cohort_note = QLabel("")
        self.cohort_note.setWordWrap(True)

        # Packet 8b: R10.F's LIKE cohort, beside the veto one. The two are the
        # halves of one judgement - what you threw away and what you endorsed -
        # and reading either alone gives half an answer.
        self.like_table = QTableWidget(0, len(COHORT_TABLE_COLUMNS))
        self.like_table.setHorizontalHeaderLabels(
            ["Claimed setup", *COHORT_TABLE_HEADERS[1:]]
        )
        self.like_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.like_note = QLabel("")
        self.like_note.setWordWrap(True)
        self.claim_caveat = QLabel("")
        self.claim_caveat.setObjectName("MutedLabel")
        self.claim_caveat.setWordWrap(True)
        #: The last full read, kept so the horizon selector re-renders without
        #: touching disk. Selecting a horizon is a VIEW change.
        self._cohort_rows: list[dict] = []
        self._like_rows: list[dict] = []

        # Packet W2: R8 sec 6's last two DEFERRED joins. The cohorts above are
        # the two judgement mirrors - what was thrown away, what was endorsed.
        # These are the picks THEMSELVES: how they behaved, and what the trader
        # said about them at the time.
        self.performance_table = QTableWidget(0, 11)
        self.performance_table.setHorizontalHeaderLabels(
            [
                "Cohort", "Side", "Horizon", "n", "Win rate", "Avg return",
                "Median", "PF", "Symbols", "Sessions", "Block CI",
            ]
        )
        self.performance_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.performance_note = QLabel("")
        self.performance_note.setWordWrap(True)

        self.feedback_table = QTableWidget(0, 7)
        self.feedback_table.setHorizontalHeaderLabels(
            ["Date", "Symbol", "Side", "Verdict", "Category", "Origin", "Reason"]
        )
        self.feedback_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.feedback_note = QLabel("")
        self.feedback_note.setWordWrap(True)

        self._layout.addWidget(self.refresh_button)
        self._layout.addWidget(self.table, 1)
        self._layout.addWidget(self.note)
        horizon_row = QHBoxLayout()
        horizon_row.setContentsMargins(0, 0, 0, 0)
        horizon_row.addWidget(QLabel("Vetoed picks, graded forward"), 1)
        horizon_row.addWidget(QLabel("Horizon"), 0)
        horizon_row.addWidget(self.cohort_horizon_input, 0)
        self._layout.addLayout(horizon_row)
        self._layout.addWidget(self.cohort_caption)
        self._layout.addWidget(self.cohort_table, 1)
        self._layout.addWidget(self.cohort_note)
        self._layout.addWidget(QLabel("Liked picks, graded forward"))
        self._layout.addWidget(self.like_table, 1)
        self._layout.addWidget(self.like_note)
        self._layout.addWidget(self.claim_caveat)
        self._layout.addWidget(QLabel("Focus picks, graded forward"))
        self._layout.addWidget(self.performance_table, 1)
        self._layout.addWidget(self.performance_note)
        self._layout.addWidget(QLabel("What you said about them at the time"))
        self._layout.addWidget(self.feedback_table, 1)
        self._layout.addWidget(self.feedback_note)
        self._finish_layout()

    def reload(self) -> None:
        """Start all five reads on this page's worker. Single-flight.

        This page ran five CSV/JSONL reads and then built every cell of five
        tables inside the click that selected it. The reads are now off the Qt
        thread and the render happens in one pass when they all land.

        The cohort still comes first and unconditionally when rendering: it is
        not week-scoped, so a week with no picks must not also hide the graded
        record of the trader's vetoes.
        """
        if self._worker is not None and self._worker.isRunning():
            return
        self.refresh_button.setEnabled(False)
        worker = _ReadWorker(self._read_everything, self)
        worker.finished_with.connect(self._on_focus_ready)
        worker.failed.connect(self._on_focus_failed)
        self._worker = worker
        worker.start()

    def _read_everything(self) -> dict:
        """Every store this page reads, and no widget. Runs on the worker."""
        return {
            "cohort": _read_veto_cohort(),
            "like": _read_like_cohort(),
            "claim_caveat": _claim_picklist_caveat(),
            "performance": _read_focus_performance(),
            "feedback": _read_pick_feedback_week(self.service.week_bounds),
            "week": _join_focus_week(self.service.week_bounds),
        }

    def _on_focus_ready(self, payload: object) -> None:  # pragma: no cover - signal seam
        self.refresh_button.setEnabled(True)
        data = payload if isinstance(payload, dict) else {}
        self._render_cohort(data.get("cohort") or [])
        self._render_like_cohort(data.get("like") or [])
        self.claim_caveat.setText(str(data.get("claim_caveat") or ""))
        self._render_performance(data.get("performance") or [])
        self._render_feedback(data.get("feedback") or [])
        self._render_week(data.get("week") or [])

    def _on_focus_failed(self, message: str) -> None:  # pragma: no cover - signal seam
        """State the failure; keep every row already on screen.

        Before this page had a worker it had no error handling at all, so an
        unreadable CSV propagated straight out of the click. Either way the
        graded cohorts are the whole forward record of the trader's own
        judgement - erasing them because one week would not load would be the
        worst possible response to a bad read.
        """
        self.refresh_button.setEnabled(True)
        self.note.setText(
            f"The focus review could not be refreshed: {message}. What is shown "
            "is the last good read, not this week's answer."
        )
        self.statusChanged.emit(f"focus review unavailable: {message}")

    def _render_week(self, rows) -> None:
        self.table.setRowCount(len(rows))
        columns = ("date", "symbol", "side", "source", "h1", "h3", "h5", "h10", "matured")
        for index, row in enumerate(rows):
            for column, key in enumerate(columns):
                self.table.setItem(index, column, QTableWidgetItem(str(row.get(key) or "")))
        apply_width_rule_to_table_widget(self.table, elide_columns=(3,))
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

    def _render_performance(self, rows) -> None:
        """The picks' own forward record (R8 sec 6, packet W2).

        Not week-scoped - see `_read_focus_performance`. Its as-of stamp is
        printed instead, so a stale rollup reads as stale rather than as this
        week's answer.
        """
        self.performance_table.setRowCount(len(rows))
        columns = (
            "cohort", "side", "horizon", "n", "win_rate", "avg_return",
            "median", "profit_factor", "symbols", "sessions", "ci",
        )
        for index, row in enumerate(rows):
            for column, key in enumerate(columns):
                self.performance_table.setItem(
                    index, column, QTableWidgetItem(str(row.get(key) or ""))
                )
        # §12: the cohort is the row's identity and it lives in the TAIL.
        apply_width_rule_to_table_widget(
            self.performance_table, text_columns=(0,), elide_columns=(0,)
        )
        if not rows:
            self.performance_note.setText(
                "No graded focus-pick rollup yet. It is written by the nightly "
                "human-focus tracking pass, and a pick needs forward sessions "
                "before it means anything - this is an absent measurement, not "
                "a flat week."
            )
            return
        stamps = sorted({row["updated_at"] for row in rows if row["updated_at"]})
        as_of = stamps[-1] if stamps else "unstated"
        blank_ci = sum(1 for row in rows if not row["ci"])
        note = (
            f"{len(rows)} rollup row(s), as of {as_of}. This table is the WHOLE "
            "graded record, not this week - the rollup carries no trade date, so "
            "scoping it to the week would have filtered on when it was last "
            "rebuilt. Returns are side-adjusted; a blank is a horizon that has "
            "not matured, never a zero. Read as DISCOVERY: n clears no floor by "
            "itself, and 'Symbols'/'Sessions' beside n are what say whether a "
            "large n is really one name on one day."
        )
        if blank_ci:
            note += (
                f" {blank_ci} row(s) carry no block interval - a sample spanning "
                "one session cannot have one."
            )
        self.performance_note.setText(note)

    def _render_feedback(self, rows) -> None:
        """The week's like/dislike verdicts in the trader's own words."""
        shown = rows[:PICK_FEEDBACK_ROWS_SHOWN]
        self.feedback_table.setRowCount(len(shown))
        columns = ("date", "symbol", "side", "verdict", "category", "origin", "reason")
        for index, row in enumerate(shown):
            for column, key in enumerate(columns):
                self.feedback_table.setItem(
                    index, column, QTableWidgetItem(str(row.get(key) or ""))
                )
        apply_width_rule_to_table_widget(self.feedback_table, text_columns=(6,))
        if not rows:
            self.feedback_note.setText(
                "No like/dislike verdicts recorded in the reviewed week, or the "
                "feedback log is not readable. That is an absent record, not a "
                "week without opinions."
            )
            return
        verdicts: dict[str, int] = {}
        for row in rows:
            verdicts[row["verdict"] or "unstated"] = verdicts.get(row["verdict"] or "unstated", 0) + 1
        tally = ", ".join(f"{count} {name}" for name, count in sorted(verdicts.items()))
        note = (
            f"{len(rows)} verdict(s) in the reviewed week ({tally}), dated by the "
            "session they are ABOUT rather than when they were typed. These are "
            "opinions, not outcomes - read them against the rollup above."
        )
        if len(rows) > len(shown):
            note += f" {len(rows) - len(shown)} row(s) beyond the first {len(shown)} are not shown."
        self.feedback_note.setText(note)

    def _cohort_horizon(self) -> str:
        """The selected horizon; the default if the widget is not built yet."""
        widget = getattr(self, "cohort_horizon_input", None)
        if widget is None:
            return DEFAULT_COHORT_HORIZON
        return str(widget.currentData() or DEFAULT_COHORT_HORIZON)

    def _on_cohort_horizon_changed(self) -> None:
        """Re-render both tables from what is already in memory.

        No disk, no worker: the full read is kept in `_cohort_rows` /
        `_like_rows`, so changing the horizon filters and re-sorts a list that
        is already here. A selector that re-read two CSVs would put a file read
        on the Qt thread for a view change.
        """
        self._render_cohort(self._cohort_rows)
        self._render_like_cohort(self._like_rows)

    def _render_like_cohort(self, rows) -> None:
        """R10.F's cohort, rendered under the same honesty rules as the veto one."""
        self._like_rows = list(rows or [])
        shown = _cohort_view(self._like_rows, self._cohort_horizon())
        _fill_cohort_table(self.like_table, shown)
        if not rows:
            self.like_note.setText(
                "No graded LIKE cohort yet. It is written by the overnight "
                "like_cohort_grading slot, and a claim needs forward sessions "
                "before it means anything - this is an absent measurement, not "
                "an empty record."
            )
            return
        if not shown:
            self.like_note.setText(
                _NO_ROWS_AT_HORIZON.format(
                    horizon=self._cohort_horizon(), total=len(rows)
                )
            )
            return
        self.like_note.setText(
            f"{len(shown)} row(s) at the {self._cohort_horizon()}-session horizon, "
            f"of {len(rows)} across all horizons, one per claimed setup family. "
            "Returns are side-adjusted, so POSITIVE means the pick you liked "
            "WORKED - the opposite reading from the veto table above, where "
            "positive means the one you rejected would have. "
            + _floor_sentence(shown)
            + " The two tables are the mirror pair - what you threw away, and "
            "what you endorsed."
        )

    def _render_cohort(self, rows) -> None:
        self._cohort_rows = list(rows or [])
        shown = _cohort_view(self._cohort_rows, self._cohort_horizon())
        _fill_cohort_table(self.cohort_table, shown)
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
        if not shown:
            self.cohort_caption.setText("")
            self.cohort_note.setText(
                _NO_ROWS_AT_HORIZON.format(
                    horizon=self._cohort_horizon(), total=len(rows)
                )
            )
            return
        self.cohort_caption.setText(
            f"Showing the {self._cohort_horizon()}-session horizon "
            f"({len(shown)} of {len(rows)} rows across all horizons). Returns are "
            "side-adjusted, so POSITIVE means the pick you vetoed would have "
            "WORKED. Read as DISCOVERY, not confirmation: these are the trader's "
            "own rejections graded forward, and a blank is a horizon that has not "
            "matured, never a zero. "
            + _floor_sentence(shown)
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
        # Every page, not a named one: this listed only `walkaway` while it was
        # the only page with a thread, and that is exactly the kind of list
        # that silently stops being complete.
        for index in range(self.pages.count()):
            page = self.pages.widget(index)
            shutdown = getattr(page, "shutdown", None)
            if callable(shutdown):
                shutdown()
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

    def _key(raw) -> tuple[str, str, str, str] | None:
        """The pick/outcome identity, INCLUDING the category slot.

        A name can sit on both the swing and the M5 list on the same day, and
        since 2026-09-01 that is two pick rows and two outcome rows. Joining on
        (date, symbol, side) alone would hand one of them the other's forward
        returns, so the family from `human_focus_tracking` - the one canonical
        implementation - is part of the key.
        """
        from human_focus_tracking import pick_source_family

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
        return (
            stamp,
            symbol,
            str(raw.get("side") or "").strip().lower(),
            pick_source_family(raw.get("source")),
        )

    outcomes: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for raw in _rows(project_paths.HUMAN_FOCUS_OUTCOMES_FILE):
        key = _key(raw)
        if key is not None:
            outcomes[key] = raw

    joined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
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
        row = {
            "cohort": veto_cohort.canonical_veto_cohort(raw.get("cohort") or ""),
            "side": str(raw.get("side") or "").strip(),
            "horizon": str(raw.get("horizon_sessions") or "").strip(),
            "n": str(raw.get("sample_count") or "").strip(),
            "win_rate": _pct(raw.get("win_rate")),
            "avg_return": _signed_pct(raw.get("avg_side_return")),
            "profit_factor": _ratio(raw.get("profit_factor")),
        }
        row.update(_cohort_robust_fields(raw))
        rows.append(row)
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
        row = {
            "cohort": str(raw.get("cohort") or "").strip(),
            "side": str(raw.get("side") or "").strip(),
            "horizon": str(raw.get("horizon_sessions") or "").strip(),
            "n": str(raw.get("sample_count") or "").strip(),
            "win_rate": _cohort_pct(raw.get("win_rate")),
            "avg_return": _cohort_signed_pct(raw.get("avg_side_return")),
            "profit_factor": _cohort_ratio(raw.get("profit_factor")),
        }
        row.update(_cohort_robust_fields(raw))
        rows.append(row)
    return rows


#: The two judgement tables' columns. `meets_n_floor` is deliberately NOT a
#: column: it decides the ORDER and the greying, which a reader takes in
#: without having to compare a yes/no cell against a sample count.
COHORT_TABLE_COLUMNS = (
    "cohort", "side", "n", "win_rate", "avg_return", "median", "trimmed",
    "profit_factor", "symbols", "sessions", "top_share", "ci", "evidence",
)
COHORT_TABLE_HEADERS = (
    "Cohort", "Side", "n", "Win rate", "Avg", "Median", "Trimmed", "PF",
    "Symbols", "Sessions", "Top share", "Block CI", "Evidence",
)


def _fill_cohort_table(table, rows) -> None:
    """Write one horizon's rows, greying anything under the n floor.

    The grey is a FOREGROUND ROLE on the item, never a per-widget stylesheet:
    a stylesheet per row is the exact cost the fluidity rules forbid, and this
    table is repainted every time the horizon selector moves.
    """
    from ui import theme

    try:
        muted = QColor(theme.color("text_muted"))
    except Exception:  # noqa: BLE001 - a missing token must not cost the table
        muted = None
    table.setRowCount(len(rows))
    for index, row in enumerate(rows):
        under_floor = not row.get("_meets_floor")
        for column, key in enumerate(COHORT_TABLE_COLUMNS):
            item = QTableWidgetItem(str(row.get(key) or ""))
            if under_floor and muted is not None:
                item.setForeground(muted)
            if key == "ci" and row.get("ci_basis"):
                item.setToolTip(str(row.get("ci_basis")))
            table.setItem(index, column, item)
    apply_width_rule_to_table_widget(table, text_columns=(0,), elide_columns=(0,))


def _floor_sentence(rows) -> str:
    """Say how many rows are below the reportable-n floor, and what that means.

    A count with no sentence reads as a footnote. The point of the floor is
    that a cohort under it is not a weak finding, it is not a finding.
    """
    under = sum(1 for row in rows if not row.get("_meets_floor"))
    if not rows:
        return ""
    if not under:
        return "Every row shown clears the reportable-n floor."
    return (
        f"{under} of {len(rows)} row(s) are UNDER the reportable-n floor: they are "
        "greyed and sorted last, because a cohort below the floor is not a weak "
        "finding, it is not a finding. Rows above it are ordered by TRIMMED mean, "
        "not the bare average."
    )


#: Horizons offered by the cohort selector, and the one it opens on. h3 is the
#: default because it is the shortest horizon at which the rollup carries a
#: block interval for most cohorts - h1 is almost all single-session samples.
COHORT_HORIZONS = ("1", "3", "5", "10")
#: Sentence used when the cohort exists but this HORIZON has no matured rows.
#: Distinguished from "no cohort at all" deliberately: they are different
#: absences, and printing the second for the first reads as a clean record.
_NO_ROWS_AT_HORIZON = (
    "No row has matured to the {horizon}-session horizon yet, though "
    "{total} row(s) exist at other horizons. Pick another horizon above - this "
    "is an unmatured measurement, not an empty record."
)
DEFAULT_COHORT_HORIZON = "3"


def _cohort_robust_fields(raw) -> dict[str, str]:
    """Ground rule 10's robust half, which both cohort tables were dropping.

    `human_focus_tracking` has written `median_return`, `trimmed_mean_return`,
    `p10/p90`, `symbols`, `sessions`, `top_symbol_share`, `ci_low`/`ci_high`,
    `ci_basis`, `evidence_label` and `meets_n_floor` since R10.C, and the Focus
    performance table on this same page already shows most of them. The two
    judgement tables kept six columns and threw the rest away, so the trader
    read a bare mean on a ratio - the statistic that produced
    `regime_pause_rw`'s -1.82R - with nothing beside it.

    Reformatted, never derived (ground rule 6). A blank stays blank: an
    interval that could not be computed, or a horizon that has not matured, is
    an absent measurement and a substituted zero would be a false lesson.
    `_sort_value` and `_meets_floor` are kept as raw floats/bools beside the
    display text so the view can order and grey rows without re-parsing them.
    """
    low = str(raw.get("ci_low") or "").strip()
    high = str(raw.get("ci_high") or "").strip()
    floor_text = str(raw.get("meets_n_floor") or "").strip().lower()
    meets_floor = floor_text in {"1", "true", "yes"}
    try:
        trimmed_value = float(str(raw.get("trimmed_mean_return") or "").strip())
    except (TypeError, ValueError):
        trimmed_value = None
    return {
        "median": _cohort_signed_pct(raw.get("median_return")),
        "trimmed": _cohort_signed_pct(raw.get("trimmed_mean_return")),
        "symbols": str(raw.get("symbols") or "").strip(),
        "sessions": str(raw.get("sessions") or "").strip(),
        "top_share": _cohort_pct(raw.get("top_symbol_share")),
        "ci": f"[{low}, {high}]" if low and high else "",
        "ci_basis": str(raw.get("ci_basis") or "").strip(),
        "evidence": str(raw.get("evidence_label") or "").strip(),
        "meets_n_floor": "1" if meets_floor else "",
        "_meets_floor": meets_floor,
        "_sort_value": trimmed_value,
    }


def _cohort_view(rows, horizon: str) -> list[dict]:
    """One horizon's rows, floor-clearing ones first, by trimmed mean.

    The ORDER is the honesty: a cohort that has not cleared the reportable-n
    floor sorts after every cohort that has, however flattering its average,
    and the view greys it. Sorting them together is what lets an n=3 row with a
    profit factor of 165 sit at the top of a table read on a Saturday.

    Within each group the key is the TRIMMED mean, not the bare one - the same
    reason R10.C published it. A row with no trimmed mean sorts last inside its
    own group rather than being promoted by a default of zero.
    """
    wanted = str(horizon or "").strip()
    kept = [row for row in rows if str(row.get("horizon") or "").strip() == wanted]
    return sorted(
        kept,
        key=lambda row: (
            0 if row.get("_meets_floor") else 1,
            0 if row.get("_sort_value") is not None else 1,
            -(row.get("_sort_value") or 0.0),
            str(row.get("cohort") or ""),
            str(row.get("side") or ""),
        ),
    )


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
    stamp, symbol, side, _family = key

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




def _claim_picklist_caveat() -> str:
    """The caveat the AI already gets, shown to the trader who wrote the data.

    `ai_summary._offered_claim_caveat` states that the like+claim control
    offers a BOUNDED picklist, so a claim type missing from the liked-cohort
    table is a fact about the user interface and not a preference. The model
    has been told that on every package; the trader reading the same table on a
    Saturday has not been. Reading the claim absence as "I never like those
    setups" is the confident wrong conclusion the caveat exists to stop.

    The one existing implementation is called - never a second copy of the
    sentence, which would drift the moment `MAIN_CLAIM_GROUP` changes. It runs
    on this page's worker with every other read, and any failure is a quieter
    page rather than a lost review.
    """
    try:
        from ai_summary import _offered_claim_caveat

        return str(_offered_claim_caveat() or "")
    except Exception:  # noqa: BLE001 - the caveat is context, never the review
        return (
            "The like+claim control offers a bounded picklist and it could not "
            "be read. Which claim types were reachable is UNKNOWN: do not read "
            "any claim's absence from the table above as a preference."
        )


def _read_focus_performance() -> list[dict[str, str]]:
    """The graded focus-pick rollup - R8 sec 6's last DEFERRED join (packet W2).

    The cohort tables answer "was I right to reject" and "was I right to
    endorse". This one answers the plainest question of the three: **how did the
    picks I actually took behave**, per cohort, side and horizon. Without it the
    page shows the two judgement mirrors and never the thing being judged.

    **NOT week-scoped, and that is deliberate**, exactly like the two cohort
    tables beside it. The rollup carries no `trade_date` - only `updated_at`,
    the stamp of when it was last rebuilt - so filtering it "to the week" the
    spec asked for would filter on the REBUILD time and empty the table on any
    week the nightly rollup happened not to run. That is a fabricated absence.
    The as-of stamp travels on every row instead, so a reader can see how old
    the measurement is.

    Read by NAMED CONSTANT, like every reader on this page, because this step
    shipped a blank table to the live desk for six days from exactly the
    opposite habit. Nothing here computes a statistic the CSV does not carry
    (ground rule 6); the columns are reformatted, never derived. A blank stays
    blank - a fabricated zero would be a false lesson about the trader's own
    picks.
    """
    import csv

    import project_paths

    path = Path(project_paths.HUMAN_FOCUS_PERFORMANCE_FILE)
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except OSError:
        return []

    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        low = str(raw.get("ci_low") or "").strip()
        high = str(raw.get("ci_high") or "").strip()
        rows.append(
            {
                "cohort": str(raw.get("cohort") or "").strip(),
                "side": str(raw.get("side") or "").strip(),
                "horizon": str(raw.get("horizon_sessions") or "").strip(),
                "n": str(raw.get("sample_count") or "").strip(),
                "win_rate": _cohort_pct(raw.get("win_rate")),
                "avg_return": _cohort_signed_pct(raw.get("avg_side_return")),
                "median": _cohort_signed_pct(raw.get("median_return")),
                "profit_factor": _cohort_ratio(raw.get("profit_factor")),
                "symbols": str(raw.get("symbols") or "").strip(),
                "sessions": str(raw.get("sessions") or "").strip(),
                # An interval that could not be computed is BLANK, and its
                # reason travels beside it: a blank with no explanation reads
                # as an oversight rather than a refusal.
                "ci": f"[{low}, {high}]" if low and high else "",
                "ci_basis": str(raw.get("ci_basis") or "").strip(),
                "updated_at": str(raw.get("updated_at") or "").strip(),
            }
        )
    return rows


def _read_pick_feedback_week(bounds) -> list[dict[str, str]]:
    """The week's like/dislike verdicts - R8 sec 6's other last DEFERRED join.

    The performance rollup says what the picks DID; this says what the trader
    THOUGHT of them at the time, in their own words. Read together on a
    Saturday they are the cheapest available check on whether a stated reason
    predicts anything.

    Week-scoped on `trade_date`, which is the session the verdict is ABOUT -
    never on `ts`, which is when it was typed. A verdict entered on Saturday
    about Friday belongs to Friday.

    Read through `pick_feedback.load_pick_feedback`, the one loader that owns
    this file, rather than a second JSONL parser on this page.
    """
    import project_paths

    monday, friday = bounds
    path = Path(project_paths.PICK_FEEDBACK_FILE)
    if not path.is_file():
        return []
    try:
        from pick_feedback import load_pick_feedback

        raw_rows = load_pick_feedback(path)
    except Exception:  # noqa: BLE001 - a review page never stops the routine
        return []

    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        stamp = str(raw.get("trade_date") or "")[:10]
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
                "symbol": str(raw.get("symbol") or "").strip().upper(),
                "side": str(raw.get("side") or "").strip(),
                "verdict": str(raw.get("verdict") or "").strip(),
                "category": str(raw.get("category") or "").strip(),
                "origin": str(raw.get("origin") or "").strip(),
                "reason": str(raw.get("reason") or "").strip(),
            }
        )
    rows.sort(key=lambda row: (row["date"], row["symbol"]))
    return rows


def _read_rrs_group_week(bounds) -> list[dict[str, Any]]:
    """The week's SECTOR and INDUSTRY RS extremes, folded per group.

    R8 sec 6's third DEFERRED join. The symbol stream beside this one says which
    NAMES led the tape; this says which parts of the market they came from, and
    a name that led a leading sector is a different observation from one that
    led a lagging one.

    Two differences from the symbol fold, both forced by the file:

    * The group log records **no bucket**. `_log_group_strength_extremes`
      writes the top and the bottom of each list with identical columns, unlike
      the symbol log which stamps `strongest`/`weakest`. So this keeps BOTH
      extremes it saw rather than inventing a direction the file never
      recorded, and the sign is what the reader reads.
    * Rows are keyed by `(group_type, group_key)`, so a sector and an industry
      ETF that share a ticker never fold into each other.

    Read-only and forgiving, like every reader here: a missing or unreadable
    CSV is a quieter week, not an error worth stopping the routine for.
    """
    import csv

    import project_paths

    monday, friday = bounds
    path = Path(project_paths.RRS_GROUP_STRENGTH_LOG_FILE)
    if not path.is_file():
        return []
    folded: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                stamp = str(raw.get("timestamp_local") or "")[:10]
                group_key = str(raw.get("group_key") or "").strip()
                group_type = str(raw.get("group_type") or "").strip() or "unknown"
                if not stamp or not group_key:
                    continue
                try:
                    when = datetime.fromisoformat(stamp).date()
                except ValueError:
                    continue
                if not (monday <= when <= friday):
                    continue
                try:
                    rrs = float(raw.get("rrs") or 0.0)
                except (TypeError, ValueError):
                    continue
                key = (group_type, group_key)
                entry = folded.get(key)
                if entry is None:
                    folded[key] = {
                        "group_type": group_type,
                        "group_key": group_key,
                        "etf": str(raw.get("etf") or "").strip(),
                        "sightings": 1,
                        "days": {stamp},
                        "max_rrs": rrs,
                        "min_rrs": rrs,
                        "last_seen": stamp,
                    }
                    continue
                entry["sightings"] += 1
                entry["days"].add(stamp)
                entry["max_rrs"] = max(entry["max_rrs"], rrs)
                entry["min_rrs"] = min(entry["min_rrs"], rrs)
                entry["last_seen"] = max(entry["last_seen"], stamp)
    except OSError:
        return []
    rows = [
        {
            "group_type": entry["group_type"],
            "group_key": entry["group_key"],
            "etf": entry["etf"],
            "sightings": entry["sightings"],
            "days": len(entry["days"]),
            "max_rrs": round(float(entry["max_rrs"]), 4),
            "min_rrs": round(float(entry["min_rrs"]), 4),
            "last_seen": entry["last_seen"],
        }
        for entry in folded.values()
    ]
    rows.sort(key=lambda row: (row["group_type"], -row["max_rrs"], row["group_key"]))
    return rows
