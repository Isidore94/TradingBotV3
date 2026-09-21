"""The ideas card - what the desk's AI suggested, and what the trader keeps.

`plan.md` §12.4 TJ-6 change 3: *"Card on Day Review and Week Review: the night's
ideas, Keep / Dismiss"*, and on Week Review the kept ones with the measurable's
before and after. ONE widget serves both pages, because they show the same rows
with a different half filled in.

It computes nothing and reads no store. Both pages read their whole payload on
ONE worker (TJ-1 and TJ-5's rule) and hand the rows here; a card that fetched
its own would be a second read behind a tab click.

THE CLICK RULES (learned on TJ-4, and again on TJ-5)
----------------------------------------------------
A click that starts work is ONE at a time: every Keep and every Dismiss on the
card goes grey the moment one starts, and stays grey until EVERY ending has
answered - including a raise. The write itself runs OFF the Qt thread, because
keeping a `process` idea freezes a measurable read over
`evidence_stats.LATELY_SESSIONS` of evidence, and that is exactly the 8.45 s
freeze the Week Review worker exists to prevent.

HONEST WHEN EMPTY
-----------------
The trader's home folder holds no ideas store at all today, so the first thing
this card ever says is "no ideas yet" and ``kept 0 of 0`` - two counts, no rate,
no percentage of nothing.

A kept `program` idea is listed under "For WISHLIST - copy". The trader pastes
it; the AI never writes `WISHLIST.md` (`plan.md` TJ-6 change 3).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ai_jobs import improvement_ideas
from ui import theme
from ui.read_worker import ReadWorker, join_worker

_log = logging.getLogger(__name__)

#: What the card says with nothing in it. The store really is absent on the
#: desk today, and "no ideas yet" is a different sentence from "nothing worked".
NO_IDEAS_YET = (
    "No ideas yet. The desk's AI writes up to three a night, each one citing "
    "your own sessions; you keep or dismiss each one here."
)

#: The heading a kept `program` idea is listed under, verbatim from `plan.md`.
WISHLIST_HEADING = "For WISHLIST - copy"

#: The card's floor, stated as a SIZE HINT and never as a second
#: `setMinimumHeight` call: `test_r4_market_journal_page_and_tables.py` counts
#: those in the pages that host this widget, and the ten-row floor has ONE
#: owner there.
CARD_FLOOR_PX = 140


def _text(value: Any) -> str:
    return str(value or "").strip()


class _WriteWorker(ReadWorker):
    """One Keep or Dismiss, off the Qt thread.

    `ReadWorker`'s shape exactly - run one callable, emit the result or the
    string of the failure - and named for what it does here so nobody reads this
    as a read. Its two endings are the only two this card has.
    """


class _IdeaRow(QWidget):
    """One idea: what it says, what it cites, and the trader's two buttons."""

    keepRequested = Signal(str)
    dismissRequested = Signal(str)

    def __init__(self, idea_id: str, parent=None) -> None:
        super().__init__(parent)
        self.idea_id = str(idea_id)
        self.row: dict[str, Any] = {}
        self.setObjectName("IdeaRow")

        self.text_label = QLabel("")
        self.text_label.setWordWrap(True)
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("SectionSubtitle")
        self.detail_label.setWordWrap(True)

        self.keep_button = QPushButton("Keep")
        self.dismiss_button = QPushButton("Dismiss")
        self.keep_button.clicked.connect(lambda: self.keepRequested.emit(self.idea_id))
        self.dismiss_button.clicked.connect(lambda: self.dismissRequested.emit(self.idea_id))

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.keep_button)
        buttons.addWidget(self.dismiss_button)
        buttons.addStretch(1)

        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 4)
        body.setSpacing(2)
        body.addWidget(self.text_label)
        body.addWidget(self.detail_label)
        body.addLayout(buttons)

    # -- rendering ---------------------------------------------------------
    def show_row(self, row: Mapping[str, Any]) -> None:
        """Fill this row from the payload. Sets text, never rebuilds widgets."""
        self.row = dict(row)
        text = _text(row.get("text"))
        if self.text_label.text() != text:
            self.text_label.setText(text)
        detail = self._detail(row)
        if self.detail_label.text() != detail:
            self.detail_label.setText(detail)

    @property
    def status(self) -> str:
        return _text(getattr(self, "row", {}).get("status"))

    @property
    def kind(self) -> str:
        return _text(getattr(self, "row", {}).get("kind"))

    def _detail(self, row: Mapping[str, Any]) -> str:
        """The line under an idea: its kind, what it cites, and its two numbers.

        Every number here arrived measured. A reading nobody could take says
        `unmeasured` rather than showing a zero - which is the whole difference
        between "your rate is 0%" and "nobody measured it".
        """
        parts: list[str] = [_text(row.get("kind")) or "idea"]
        measurable = _text(row.get("measurable"))
        if measurable:
            parts.append(f"checked by {measurable}")
        seen = int(row.get("seen_count") or 0)
        if seen > 1:
            parts.append(f"suggested {seen} times")
        cited = [_text(item) for item in row.get("evidence") or () if _text(item)]
        if cited:
            parts.append("from " + ", ".join(cited))
        status = _text(row.get("status"))
        if status:
            parts.append(status)
        before = row.get("before") if isinstance(row.get("before"), Mapping) else {}
        after = row.get("after") if isinstance(row.get("after"), Mapping) else {}
        if before or after:
            parts.append(
                f"before {_reading(before)} · after {_reading(after)}"
                + (f" · {_text(row.get('verdict'))}" if _text(row.get("verdict")) else "")
            )
        return " · ".join(parts)


def _reading(reading: Mapping[str, Any]) -> str:
    """One stored reading, with its own `n`. Never a percentage of nothing."""
    if not reading or not bool(reading.get("measured")):
        return "unmeasured"
    value = reading.get("value")
    try:
        body = f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "unmeasured"
    return f"{body} (n {int(reading.get('n') or 0)})"


class IdeasCard(QWidget):
    """The night's ideas, with Keep and Dismiss. ONE write in flight, ever."""

    def __init__(self, parent=None, *, writer: Callable[[str, str], Any] | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("IdeasCard")
        self._writer = writer or self._default_writer
        self._end_session = ""
        self._writing = False
        self._worker: _WriteWorker | None = None
        self._by_id: dict[str, _IdeaRow] = {}

        self.rows: tuple[_IdeaRow, ...] = ()
        self.summary_label = QLabel("")
        self.summary_label.setObjectName("SectionSubtitle")
        self.summary_label.setWordWrap(True)
        self.status_label = QLabel("")
        self.status_label.setObjectName("SectionSubtitle")
        self.status_label.setWordWrap(True)
        self.empty_note = QLabel(NO_IDEAS_YET)
        self.empty_note.setObjectName("SectionSubtitle")
        self.empty_note.setWordWrap(True)
        self.wishlist_label = QLabel("")
        self.wishlist_label.setObjectName("SectionSubtitle")
        self.wishlist_label.setWordWrap(True)
        # The trader PASTES a kept program idea into WISHLIST.md themselves, so
        # the block has to be selectable. The AI never writes that file.
        self.wishlist_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._body = QVBoxLayout(self)
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(4)
        self._body.addWidget(self.summary_label)
        self._body.addWidget(self.empty_note)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        self._body.addLayout(self._rows_layout)
        self._body.addWidget(self.wishlist_label)
        self._body.addWidget(self.status_label)
        self._body.addStretch(1)
        self.show_ideas(())

    # -- the page's one hand-off -------------------------------------------
    def set_session(self, session: Any) -> None:
        """Which session a Keep should measure its baseline up to."""
        self._end_session = _text(session)[:10]

    def show_ideas(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Render the payload's rows. DIFFS - the same ideas keep their widgets.

        CLAUDE.md: lists diff, never rebuild. Two renders of one payload that
        threw away eight widgets and built eight more would be the cheapest
        stall on the page to avoid and the easiest to leave in.
        """
        wanted = [dict(row) for row in rows or () if isinstance(row, Mapping)]
        order: list[_IdeaRow] = []
        seen: set[str] = set()
        for row in wanted:
            key = _text(row.get("idea_id"))
            if not key or key in seen:
                continue
            seen.add(key)
            widget = self._by_id.get(key)
            if widget is None:
                widget = _IdeaRow(key, self)
                widget.keepRequested.connect(self._on_keep)
                widget.dismissRequested.connect(self._on_dismiss)
                self._by_id[key] = widget
                self._rows_layout.addWidget(widget)
            widget.show_row(row)
            widget.setVisible(True)
            order.append(widget)
        for key, widget in list(self._by_id.items()):
            if key in seen:
                continue
            self._rows_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
            self._by_id.pop(key, None)
        self.rows = tuple(order)
        self.empty_note.setVisible(not self.rows)
        self._refresh_summary()

    # -- what the card says ------------------------------------------------
    def summary_text(self) -> str:
        return self.summary_label.text()

    def status_text(self) -> str:
        return self.status_label.text()

    def wishlist_text(self) -> str:
        return self.wishlist_label.text()

    def is_writing(self) -> bool:
        return bool(self._writing)

    def _refresh_summary(self) -> None:
        total = len(self.rows)
        kept = sum(1 for row in self.rows if row.status == improvement_ideas.STATUS_KEPT)
        # Two integer counts and no rate: a percentage over nothing is the shape
        # this card must never take, and "kept 0 of 0" is the honest first state.
        self.summary_label.setText(f"Kept {kept} of {total}")
        lines = [
            row.row.get("text")
            for row in self.rows
            if row.kind == "program" and row.status == improvement_ideas.STATUS_KEPT
        ]
        if lines:
            self.wishlist_label.setText(
                WISHLIST_HEADING + "\n" + "\n".join(f"- {_text(line)}" for line in lines)
            )
        else:
            self.wishlist_label.setText("")
        self.wishlist_label.setVisible(bool(lines))

    # -- the two clicks ----------------------------------------------------
    def _on_keep(self, idea_id: str) -> None:
        self._start(idea_id, improvement_ideas.STATUS_KEPT)

    def _on_dismiss(self, idea_id: str) -> None:
        self._start(idea_id, improvement_ideas.STATUS_DISMISSED)

    def _start(self, idea_id: str, status: str) -> None:
        """ONE write at a time, and every button grey until it answers."""
        if self._writing:
            return
        self._writing = True
        self._set_enabled(False)
        self.status_label.setText(f"Saving {status}...")
        worker = _WriteWorker(lambda: self._writer(str(idea_id), str(status)), self)
        worker.finished_with.connect(
            lambda _result, key=str(idea_id), state=str(status): self._answered(key, state, "")
        )
        worker.failed.connect(
            lambda message, key=str(idea_id), state=str(status): self._answered(
                key, state, str(message)
            )
        )
        self._worker = worker
        worker.start()

    def _answered(self, idea_id: str, status: str, problem: str) -> None:
        """EVERY ending answers, including a raise.

        A card left grey by a write that failed silently is a card the trader
        has to restart the desk to use again.
        """
        self._writing = False
        self._set_enabled(True)
        if problem:
            self.status_label.setText(f"That {status} was not saved: {problem}")
            return
        self.status_label.setText("")
        widget = self._by_id.get(str(idea_id))
        if widget is not None:
            # The store is the truth and the next read confirms it; this is only
            # so the count under the card moves on the click that moved it.
            row = dict(getattr(widget, "row", {}))
            row["status"] = status
            widget.show_row(row)
        self._refresh_summary()

    def _set_enabled(self, enabled: bool) -> None:
        for row in self.rows:
            row.keep_button.setEnabled(bool(enabled))
            row.dismiss_button.setEnabled(bool(enabled))

    def _default_writer(self, idea_id: str, status: str) -> Any:
        """The state file's OWN writers, and there is no other door.

        Resolved on the module at call time, so the card and the store cannot
        drift about what keeping an idea means.
        """
        if status == improvement_ideas.STATUS_KEPT:
            return improvement_ideas.keep_idea(idea_id, end_session=self._end_session)
        return improvement_ideas.dismiss_idea(idea_id)

    # -- sizing and shutdown -----------------------------------------------
    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt's own name
        """The card's floor, as a HINT. Never a second minimum-height setter."""
        base = super().minimumSizeHint()
        return QSize(base.width(), max(base.height(), theme.px(CARD_FLOOR_PX)))

    def shutdown(self) -> None:
        """Wait for the write in flight, but never forever."""
        worker, self._worker = self._worker, None
        if worker is not None:
            join_worker(worker)
        self._writing = False


__all__ = ["CARD_FLOOR_PX", "IdeasCard", "NO_IDEAS_YET", "WISHLIST_HEADING"]
