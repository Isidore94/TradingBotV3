"""Week Review coach section: your edge, your leaks, repeats, trend, Ask the AI.

Every store read and the question write run on a `ReadWorker`; this widget only
renders what `week_coach.read_view` returned. A citation that names a session
emits `openSessionRequested`, the signal Week Review already routes to Day Review.
"""

from __future__ import annotations

import html
import logging
from typing import Any, Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import week_coach
from ui.read_worker import ReadWorker, join_worker

#: How many questions the section lists (newest first).
QUESTIONS_SHOWN = 6


class WeekCoachCard(QFrame):
    openSessionRequested = Signal(str)

    def __init__(self, parent=None, *, read=None, ask=None, table_floor_px: int = 0) -> None:
        super().__init__(parent)
        self.setObjectName("WeekCoachCard")
        self._read = read or week_coach.read_view
        self._ask = ask or week_coach.record_question
        self._week = ""
        self._month = False
        self._worker: ReadWorker | None = None
        self._ask_worker: ReadWorker | None = None
        self._reading = False
        self._again = False
        self._view: dict[str, Any] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Your week, from your records")
        title.setObjectName("SectionTitle")
        self.prev_button = QPushButton("◀")
        self.next_button = QPushButton("▶")
        self.month_button = QPushButton("Month")
        self.month_button.setCheckable(True)
        self.week_label = QLabel("")
        for button in (self.prev_button, self.next_button):
            button.setFixedWidth(32)
        self.prev_button.setToolTip("Week before")
        self.next_button.setToolTip("Week after")
        self.month_button.setToolTip("Roll up every week of this calendar month")
        self.prev_button.clicked.connect(lambda: self._step(-1))
        self.next_button.clicked.connect(lambda: self._step(1))
        self.month_button.toggled.connect(self._month_toggled)
        picker = QHBoxLayout()
        picker.addWidget(title)
        picker.addStretch(1)
        picker.addWidget(self.prev_button)
        picker.addWidget(self.week_label)
        picker.addWidget(self.next_button)
        picker.addWidget(self.month_button)
        layout.addLayout(picker)

        self.note = QLabel("Reading your week...")
        self.note.setWordWrap(True)
        self.note.setObjectName("SectionSubtitle")
        layout.addWidget(self.note)
        columns = QHBoxLayout()
        self.edge_label = self._block(columns, "Your edge")
        self.leaks_label = self._block(columns, "Your leaks")
        self.repeats_label = self._block(columns, "Repeats")
        layout.addLayout(columns)
        #: P8-P5: the journal's truth lines for the week and the 4-week rollup.
        self.truth_label = QLabel("")
        self.truth_label.setObjectName("TruthNote")
        self.truth_label.setWordWrap(True)
        self.truth_label.setTextFormat(Qt.PlainText)
        self.truth_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.truth_label)

        self.trend = QTableWidget(0, 4)
        self.trend.setHorizontalHeaderLabels(("Week", "P&L", "Rules kept", "Calls right"))
        self.trend.setEditTriggers(QTableWidget.NoEditTriggers)
        self.trend.verticalHeader().setVisible(False)
        if table_floor_px:
            self.trend.setMinimumHeight(int(table_floor_px))
        layout.addWidget(self.trend)

        ask_title = QLabel("Ask the AI")
        ask_title.setObjectName("SectionTitle")
        layout.addWidget(ask_title)
        self.ask_box = QLineEdit()
        self.ask_box.setPlaceholderText("Ask in plain words, e.g. Do my afternoon trades lose?")
        self.ask_box.setMaxLength(week_coach.QUESTION_MAX)
        self.ask_box.returnPressed.connect(self.ask)
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.ask)
        ask_row = QHBoxLayout()
        ask_row.addWidget(self.ask_box, 1)
        ask_row.addWidget(self.ask_button)
        layout.addLayout(ask_row)
        self.ask_note = QLabel("Answered tonight, from your records only.")
        self.ask_note.setObjectName("SectionSubtitle")
        self.ask_note.setWordWrap(True)
        layout.addWidget(self.ask_note)
        self.questions = QWidget()
        self._questions_layout = QVBoxLayout(self.questions)
        self._questions_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.questions)
        layout.addStretch(1)

    @staticmethod
    def _block(columns: QHBoxLayout, title: str) -> QLabel:
        holder = QFrame()
        columns.addWidget(holder, 1)
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        heading = QLabel(title)
        heading.setObjectName("SectionTitle")
        body = QLabel("")
        body.setWordWrap(True)
        body.setTextFormat(Qt.PlainText)
        body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.addWidget(heading)
        box.addWidget(body)
        box.addStretch(1)
        return body

    # -- reading ------------------------------------------------------------
    def set_week(self, week: str) -> None:
        """Choose the week the next `load()` reads. Reads nothing."""
        self._week = str(week or "")

    def load(self, week: str | None = None) -> None:
        """Start the one read on a worker. A second call while reading queues one more."""
        if week is not None:
            self._week = str(week or "")
        if self._reading:
            self._again = True
            return
        self._reading = True
        self._again = False
        chosen, month = self._week, self._month
        worker = ReadWorker(lambda: self._read(chosen, month=month), self)
        worker.finished_with.connect(self._on_ready)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def _finish(self) -> bool:
        self._reading = False
        if self._again:
            self.load()
            return False
        return True

    def _on_ready(self, view: object) -> None:
        if not self._finish():
            return
        try:
            self.render(view if isinstance(view, dict) else {})
        except Exception as exc:  # noqa: BLE001 - a render that raises still answers
            logging.debug("The week coach could not be drawn.", exc_info=True)
            self.note.setText(f"Your week could not be drawn: {exc}")

    def _on_failed(self, message: str) -> None:
        if self._finish():
            self.note.setText(f"Your week could not be read: {message}")

    def _step(self, weeks: int) -> None:
        current = self._week or self._view.get("week")
        if not current:
            return
        target = week_coach.shift_week(current, weeks)
        if self._month:
            # In Month view the arrows move a whole calendar month.
            while week_coach.month_of(target) == week_coach.month_of(current):
                target = week_coach.shift_week(target, weeks)
        self._week = target
        self.load()

    def _month_toggled(self, checked: bool) -> None:
        self._month = bool(checked)
        self.load()

    # -- asking -------------------------------------------------------------
    def ask(self) -> None:
        text = self.ask_box.text().strip()
        if not text or self._ask_worker is not None and self._ask_worker.isRunning():
            return
        self.ask_button.setEnabled(False)
        week = self._view.get("week") or self._week
        worker = ReadWorker(lambda: self._ask(text, week=week), self)
        worker.finished_with.connect(self._on_asked)
        worker.failed.connect(self._on_ask_failed)
        self._ask_worker = worker
        worker.start()

    def _on_asked(self, _row: object) -> None:
        self.ask_button.setEnabled(True)
        self.ask_box.clear()
        self.ask_note.setText("Saved. Answered tonight, from your records only.")
        self.load()

    def _on_ask_failed(self, message: str) -> None:
        self.ask_button.setEnabled(True)
        self.ask_note.setText(f"Not saved: {message}")

    def shutdown(self) -> None:
        join_worker(self._worker)
        join_worker(self._ask_worker)

    # -- rendering ----------------------------------------------------------
    def render(self, view: Mapping[str, Any]) -> None:
        self._view = dict(view or {})
        week = str(self._view.get("week") or "")
        self._week = self._week or week
        month = str(self._view.get("month") or "")
        covered = list(self._view.get("covered_weeks") or ())
        self.week_label.setText(f"Month {month}" if month else week)
        n = int(self._view.get("trades_n") or 0)
        if not self._view.get("recorded"):
            self.note.setText(f"No record for {'month ' + month if month else 'week ' + week} yet.")
        else:
            scope = f"{len(covered)} week(s): {', '.join(covered)}" if month else f"week {week}"
            self.note.setText(
                f"{scope} · {len(self._view.get('sessions') or ())} session(s) · {n} trade(s). "
                f"Rows need n {week_coach.MIN_N}+; {self._view.get('thin_rows', 0)} row(s) are "
                f"{week_coach.TOO_FEW}. Unknown is not zero."
            )
        self.edge_label.setText(self._ranked_text("edge", positive=True))
        self.leaks_label.setText(self._ranked_text("leaks", positive=False))
        self.repeats_label.setText(self._repeats_text())
        self.truth_label.setText(self._truth_text())
        self._render_trend()
        self._render_questions()

    def _ranked_text(self, key: str, *, positive: bool) -> str:
        """Ranked rows (n 10+), then thin rows (n 5-9), for the view and its 4-week rollup."""
        none = f"No row has n {week_coach.MIN_N}+ yet ({week_coach.TOO_FEW})."

        def block(view: Mapping[str, Any]) -> list[str]:
            lines = [week_coach.row_line(row) for row in view.get(key) or ()] or [none]
            values = [(row, float(row.get("value") or 0.0)) for row in view.get("thin") or ()]
            thin = [row for row, value in values if (value > 0 if positive else value < 0)]
            lines.extend(week_coach.thin_line(row) for row in thin)
            return lines

        lines = block(self._view)
        rollup = dict(self._view.get("rollup") or {})
        weeks = list(rollup.get("weeks") or ())
        if weeks:
            lines.append(f"Last {len(weeks)} weeks ({weeks[0]} to {weeks[-1]}):")
            lines.extend(block(rollup))
        return "\n".join(lines)

    def _truth_text(self) -> str:
        truth = dict(self._view.get("truth") or {})
        if truth.get("error"):
            return f"In words: unknown ({truth['error']})."
        if not truth:
            return ""
        month = str(self._view.get("month") or "")
        lines = [f"Month {month}:" if month else "This week:"]
        lines.extend(str(line) for line in truth.get("lines") or ())
        if truth.get("worst_line"):
            lines.append(str(truth["worst_line"]))
        rollup = list(truth.get("rollup_weeks") or ())
        if rollup:
            lines.append(f"Last {len(rollup)} weeks ({rollup[0]} to {rollup[-1]}):")
            lines.extend(str(line) for line in truth.get("rollup_lines") or ())
        return "\n".join(lines)

    def _repeats_text(self) -> str:
        rep = dict(self._view.get("repeats") or {})
        lines = [f"{item['part']}: \"{item['text']}\" x{item['n']}" for item in rep.get("lessons") or ()]
        if not lines:
            lines.append("No lesson said twice yet.")
        for rule in (rep.get("rules") or ())[:5]:
            lines.append(f"Rule \"{rule['text']}\": kept {week_coach.fmt_rate(rule.get('rate'), int(rule.get('checked_n') or 0))}")
        kept = dict(self._view.get("rule_kept") or {})
        lines.append(f"All rules kept: {week_coach.fmt_rate(kept.get('rate'), int(kept.get('n') or 0))}")
        return "\n".join(lines)

    def _render_trend(self) -> None:
        rows = list(self._view.get("trend") or ())
        self.trend.setRowCount(len(rows))
        for index, row in enumerate(rows):
            if row.get("recorded"):
                calls = dict(row.get("calls") or {})
                values = (
                    str(row.get("week") or ""),
                    f"{week_coach.fmt_money(row.get('pnl_cad'))} (n {int(row.get('pnl_known_n') or 0)})",
                    week_coach.fmt_rate(row.get("rule_kept_rate"), int(row.get("rule_checks_n") or 0)),
                    week_coach.fmt_rate(calls.get("rate"), int(calls.get("n") or 0)),
                )
            else:
                values = (str(row.get("week") or ""), "no record", "no record", "no record")
            for column, value in enumerate(values):
                item = self.trend.item(index, column)
                if item is None:
                    self.trend.setItem(index, column, QTableWidgetItem(value))
                elif item.text() != value:
                    item.setText(value)
        self.trend.resizeColumnsToContents()

    def _render_questions(self) -> None:
        while self._questions_layout.count():
            item = self._questions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        questions = list(self._view.get("questions") or ())[:QUESTIONS_SHOWN]
        if not questions:
            self._questions_layout.addWidget(QLabel("No questions yet."))
        for question in questions:
            self._questions_layout.addWidget(self._question_widget(question))

    def _question_widget(self, question: Mapping[str, Any]) -> QWidget:
        box = QFrame()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 4, 0, 4)
        status = str(question.get("status") or week_coach.STATUS_PENDING)
        heading = QLabel(f"<b>Q:</b> {html.escape(str(question.get('text') or ''))}"
                         f" <i>({'answered tonight' if status == week_coach.STATUS_PENDING else status.replace('_', ' ')})</i>")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        answer = dict(question.get("answer") or {})
        claims = list(answer.get("claims") or ())
        if answer and not claims:
            layout.addWidget(QLabel("The records could not answer this with a cited claim."))
        for claim in claims:
            row = QHBoxLayout()
            flag = f" ({claim['flag']})" if claim.get("flag") else ""
            text = QLabel(f"• {claim.get('text', '')}{flag}")
            text.setWordWrap(True)
            text.setTextFormat(Qt.PlainText)
            row.addWidget(text, 1)
            for cite in claim.get("citations") or ():
                button = QToolButton()
                button.setText(str(cite.get("id") or ""))
                session = str(cite.get("session") or "")
                button.setEnabled(bool(session))
                button.setToolTip(f"Open {session} in Day Review" if session else "A week row, not one day")
                if session:
                    button.clicked.connect(lambda _checked=False, day=session: self.openSessionRequested.emit(day))
                row.addWidget(button)
            layout.addLayout(row)
        dropped = int(answer.get("dropped_n") or 0)
        notes = []
        if dropped:
            notes.append(f"{dropped} claim(s) {week_coach.UNCITED_NOTE}")
        if answer.get("small_sample"):
            notes.append(f"only {int(answer.get('trades_read') or 0)} trade(s) read: {week_coach.TOO_FEW}")
        if answer.get("model"):
            notes.append(f"model {answer['model']}")
        if notes:
            note = QLabel(" · ".join(notes))
            note.setObjectName("SectionSubtitle")
            layout.addWidget(note)
        return box
