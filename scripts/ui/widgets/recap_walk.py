"""Review my day: the guided walk, about five minutes, hosted in Day Review.

One card at a time with a progress bar ("3 of 9 · ~2 min left"). The cards
are built on a worker (`recap_walk_cards.load_walk_inputs`); every save runs on
a worker and says "saved" or "not saved: reason" on its card. Nothing here
reads a store on the Qt thread.

Keys: Right / Space next, Left back, 1-9 pick an option, Enter saves the text
you are typing, Esc exits (or leaves clue mode first).

The position is kept per session in local settings, so the walk resumes where
it was left. After the lesson card - or on exit, if anything was saved -
`day_session_record.rebuild(session)` runs on a worker.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Mapping

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.widgets import recap_walk_cards as cards_lib
from swallowed import note_swallowed

#: Local-settings key: `{session: {"index": int, "finished": bool}}`.
WALK_SETTING_KEY = "qt_day_review_walk_v1"
#: How many sessions' positions are kept.
WALK_SETTING_SESSIONS = 30
KEY_HINT = "→ / Space next · ← back · 1-9 choose · Enter saves text · Esc exits"
CHART_HEIGHT_PX = 300

try:  # the clue hook; the button hides when it is missing
    from ui.widgets.clue_marker import mark_clue_flow
except Exception:  # noqa: BLE001
    mark_clue_flow = None


# ---------------------------------------------------------------------------
# where the walk was left
# ---------------------------------------------------------------------------
class WalkPositions:
    """Per-session position in local settings. Small and cached by `project_paths`."""

    def get(self, session: str) -> dict[str, Any]:
        try:
            from project_paths import get_local_setting

            data = get_local_setting(WALK_SETTING_KEY, {}) or {}
            row = data.get(str(session)[:10]) if isinstance(data, Mapping) else None
            return dict(row) if isinstance(row, Mapping) else {}
        except Exception:  # noqa: BLE001 - a setting never costs the walk
            return {}

    def put(self, session: str, row: Mapping[str, Any]) -> None:
        try:
            from project_paths import get_local_setting, save_local_setting

            data = dict(get_local_setting(WALK_SETTING_KEY, {}) or {})
            key = str(session)[:10]
            if data.get(key) == dict(row):
                return
            data[key] = dict(row)
            for old in sorted(data)[:-WALK_SETTING_SESSIONS]:
                data.pop(old, None)
            save_local_setting(WALK_SETTING_KEY, data)
        except Exception:  # noqa: BLE001
            logging.debug("The walk position was not remembered.", exc_info=True)


def walk_finished(session: str, positions: WalkPositions | None = None) -> bool:
    return bool((positions or WalkPositions()).get(session).get("finished"))


# ---------------------------------------------------------------------------
# the writers (worker thread only)
# ---------------------------------------------------------------------------
class WalkWriter:
    """Every save the walk makes, through the stores' own writers.

    Raises `recap_store.RecapError` / `RecapWriteError` (or any error) on a
    save that did not happen; the walk says "not saved: reason".
    """

    def __init__(self, journal_store: Any = None) -> None:
        self._store = journal_store

    def _journal(self):
        if self._store is None:
            from journal_store import JournalStore

            self._store = JournalStore()
        return self._store

    def card_answer(self, *, session, card, option, text="", supersedes=""):
        import recap_store

        return recap_store.record_card_answer(
            session_date=session, card_id=card["id"], card_kind=card["kind"],
            subject=card.get("subject") or {}, option=option, text=text, supersedes=supersedes,
        )

    def mentor_answer(self, *, session, card, subject, option):
        import recap_store

        return recap_store.record_card_answer(
            session_date=session, card_id=card["id"], card_kind=card["kind"],
            subject=card.get("subject") or {}, option=option,
            mentor_subject=subject, journal_store=self._journal(),
        )

    def exit_note(self, *, trade_id, exit_session, text):
        import recap_store
        import trade_mentor_trade_check as check

        try:
            return check.save_exit_note(self._journal(), trade_id, text, exit_session=exit_session)
        except ValueError as exc:
            raise recap_store.RecapError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - a journal write fails loudly
            raise recap_store.RecapWriteError(f"the exit note was not saved: {exc}") from exc

    def _exit_result(self, result):
        import recap_store

        if not isinstance(result, Mapping) or not result.get("ok"):
            reason = result.get("reason") if isinstance(result, Mapping) else ""
            raise recap_store.RecapWriteError(f"the exit fields were not saved: {reason or 'nothing written'}")
        return dict(result)

    def exit_confirm(self, *, draft):
        from ui.widgets.trade_mentor_card import confirm_exit_draft

        return self._exit_result(confirm_exit_draft(self._journal(), draft))

    def exit_correct(self, *, draft, why, watching=()):
        from ui.widgets.trade_mentor_card import correct_exit_draft

        return self._exit_result(correct_exit_draft(self._journal(), draft, why=why, watching=tuple(watching)))

    def environment_verdict(self, *, session, auto_label, verdict, clue_ids=(), text="", supersedes=""):
        import recap_store

        return recap_store.record_environment_verdict(
            session_date=session, auto_label=auto_label, verdict=verdict,
            clue_ids=clue_ids, text=text, supersedes=supersedes,
        )

    def lesson(self, *, session, keep="", stop="", try_="", mood=None, supersedes=""):
        import recap_store

        return recap_store.record_lesson(
            session_date=session, keep=keep, stop=stop, try_=try_, mood=mood, supersedes=supersedes,
        )

    def rule(self, *, session, text, tag="", supersedes=""):
        import recap_store

        row = recap_store.record_rule(session_date=session, text=text, tag=tag, supersedes=supersedes)
        # P1-7 7d: the same rule goes under the plan's "What I am testing". The
        # recap row is already saved, so a plan write failure is logged, not raised.
        try:
            import recap_rule_loop

            recap_rule_loop.write_rule_to_plan(row)
        except Exception:  # noqa: BLE001 - the recap is the record; the plan copy is secondary
            logging.warning("The rule was saved but not copied into the trading plan.", exc_info=True)
        return row

    def rule_check(self, *, session, answer, rule_id="", supersedes=""):
        import recap_store

        return recap_store.record_rule_check(
            session_date=session, answer=answer, rule_id=rule_id, supersedes=supersedes,
        )

    def clue_ids(self, *, session):
        import recap_store

        return [
            str(row.get("id")) for row in recap_store.records_for(session, [recap_store.KIND_CLUE])
            if str(row.get("symbol") or "").upper() == "SPY"
        ]


def _default_loader(session: str, payload: Mapping[str, Any], zone: Any) -> dict[str, Any]:
    return cards_lib.load_walk_inputs(session, payload, today=datetime.now().date().isoformat(), zone=zone)


def _default_rebuilder(session: str) -> Any:
    import day_session_record

    return day_session_record.rebuild(session)


# ---------------------------------------------------------------------------
# a worker
# ---------------------------------------------------------------------------
class _JobSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Job(QRunnable):
    """One call off the Qt thread. Every ending emits exactly once."""

    def __init__(self, call: Callable[[], Any]) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self._call = call
        self.signals = _JobSignals()

    def run(self) -> None:
        try:
            result = self._call()
        except Exception as exc:  # noqa: BLE001 - the trader must see why
            self.signals.failed.emit(str(exc) or exc.__class__.__name__)
            return
        self.signals.done.emit(result)


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------
class RecapWalk(QFrame):
    """The walk. Emits `exited` when the trader leaves it (Exit, Esc or Done)."""

    exited = Signal()
    walkFinished = Signal(str)
    statusChanged = Signal(str)

    def __init__(
        self,
        session: str,
        payload: Mapping[str, Any],
        parent=None,
        *,
        loader: Callable[[str, Mapping[str, Any], Any], Mapping[str, Any]] | None = None,
        writer: Any = None,
        rebuilder: Callable[[str], Any] | None = None,
        positions: Any = None,
        zone: Any = None,
        pool: QThreadPool | None = None,
        clue_flow: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("RecapWalk")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.session = str(session or "")[:10]
        self._payload = dict(payload or {})
        self._loader = loader or _default_loader
        self._writer = writer if writer is not None else WalkWriter()
        self._rebuilder = rebuilder or _default_rebuilder
        self._positions = positions if positions is not None else WalkPositions()
        self._zone = zone or cards_lib.ET
        self._pool = pool or QThreadPool.globalInstance()
        # None: the real clue hook; False: no clue button (tests, or no hook).
        self._clue_flow = mark_clue_flow if clue_flow is None else (clue_flow or None)
        self._jobs: list[_Job] = []
        self.cards: list[dict[str, Any]] = []
        self.index = 0
        self._loaded = False
        self._card_pages: dict[int, QWidget] = {}
        self._primary: dict[int, list[QPushButton]] = {}
        self._status_labels: dict[int, QLabel] = {}
        self._flows: dict[int, Any] = {}
        self._clue_buttons: dict[int, QPushButton] = {}
        self._lesson_widgets: dict[str, Any] = {}
        self._charts: dict[int, Any] = {}
        #: `slot -> row id` of the last recap row saved, so a re-save supersedes it.
        self._last_ids: dict[str, str] = {}
        #: `card id -> [(what, words)]` for the finished screen.
        self.answers: dict[str, list[str]] = {}
        self.saved_count = 0
        self._saves_since_rebuild = 0
        self.rebuilds_started = 0
        self.tomorrow_rule = ""
        self._build()

    # -- frame ---------------------------------------------------------------
    def _build(self) -> None:
        self.heading = QLabel(f"Review my day · {self.session}")
        self.heading.setObjectName("SectionTitle")
        self.progress_label = QLabel("Building your walk…")
        self.progress_label.setObjectName("SectionSubtitle")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(theme.px(6))
        self.exit_button = QPushButton("Exit")
        self.exit_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.exit_button.setToolTip("Leave the walk (Esc). It resumes where you left it.")
        self.exit_button.clicked.connect(self.exit_walk)
        top = QHBoxLayout()
        top.addWidget(self.heading)
        top.addStretch(1)
        top.addWidget(self.progress_label)
        top.addWidget(self.exit_button)

        self.loading_label = QLabel("Building your walk…")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setWordWrap(True)
        self.cards_stack = QStackedWidget()
        self.finished_page = self._build_finished_page()
        self.body = QStackedWidget()
        self.body.addWidget(self.loading_label)
        self.body.addWidget(self.cards_stack)
        self.body.addWidget(self.finished_page)

        self.back_button = QPushButton("← Back")
        self.next_button = QPushButton("Next →")
        for button in (self.back_button, self.next_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.back_button.clicked.connect(self.back)
        self.next_button.clicked.connect(self.next)
        self.hint = QLabel(KEY_HINT)
        self.hint.setObjectName("SectionSubtitle")
        bottom = QHBoxLayout()
        bottom.addWidget(self.back_button)
        bottom.addStretch(1)
        bottom.addWidget(self.hint)
        bottom.addStretch(1)
        bottom.addWidget(self.next_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addLayout(top)
        layout.addWidget(self.progress)
        layout.addWidget(self.body, 1)
        layout.addLayout(bottom)
        self._sync_nav()

    def _build_finished_page(self) -> QWidget:
        page = QWidget()
        body = QVBoxLayout(page)
        title = QLabel("Done. Nice work.")
        title.setObjectName("SectionTitle")
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.rule_label = QLabel("")
        self.rule_label.setWordWrap(True)
        self.rule_label.setObjectName("SectionTitle")
        self.done_button = QPushButton("Done")
        self.done_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.done_button.clicked.connect(self.exit_walk)
        again = QPushButton("Walk it again")
        again.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        again.clicked.connect(lambda: self.go_to(0))
        row = QHBoxLayout()
        row.addWidget(again)
        row.addStretch(1)
        row.addWidget(self.done_button)
        body.addWidget(title)
        body.addWidget(self.rule_label)
        body.addWidget(self.summary_label)
        body.addStretch(1)
        body.addLayout(row)
        return page

    # -- loading -------------------------------------------------------------
    def start(self) -> None:
        """Build the cards on a worker. Returns at once."""
        session, payload, zone, loader = self.session, dict(self._payload), self._zone, self._loader
        self._run(lambda: loader(session, payload, zone), self._on_loaded, self._on_load_failed)

    def _run(self, call, on_done, on_failed) -> _Job:
        job = _Job(call)
        job.signals.done.connect(on_done)
        job.signals.failed.connect(on_failed)
        job.signals.done.connect(lambda _r, j=job: self._forget(j))
        job.signals.failed.connect(lambda _r, j=job: self._forget(j))
        self._jobs.append(job)
        self._pool.start(job)
        return job

    def _forget(self, job: _Job) -> None:
        if job in self._jobs:
            self._jobs.remove(job)

    def busy(self) -> bool:
        return bool(self._jobs)

    def _on_load_failed(self, reason: str) -> None:
        self.loading_label.setText(f"The walk could not be built: {reason}")
        self.progress_label.setText("not built")

    def _on_loaded(self, result: Any) -> None:
        data = dict(result or {}) if isinstance(result, Mapping) else {}
        self.cards = [dict(card) for card in data.get("cards") or () if isinstance(card, Mapping)]
        self.unread = list(data.get("unread") or ())
        self._loaded = True
        if not self.cards:
            self.loading_label.setText("Nothing to walk through for this session.")
            return
        self.progress.setMaximum(len(self.cards))
        saved = self._positions.get(self.session)
        if saved.get("finished"):
            self._show_finished(remember=False)
            return
        try:
            start = int(saved.get("index") or 0)
        except (TypeError, ValueError):
            start = 0
        self.go_to(max(0, min(start, len(self.cards) - 1)))

    # -- moving --------------------------------------------------------------
    def current_card(self) -> dict[str, Any] | None:
        if self.body.currentWidget() is not self.cards_stack:
            return None
        return self.cards[self.index] if 0 <= self.index < len(self.cards) else None

    def is_finished_screen(self) -> bool:
        return self.body.currentWidget() is self.finished_page

    def go_to(self, index: int) -> None:
        if not self.cards:
            return
        self._leave_clue_mode()
        self.index = max(0, min(int(index), len(self.cards) - 1))
        page = self._card_pages.get(self.index)
        if page is None:
            page = self._build_card_page(self.index, self.cards[self.index])
            self._card_pages[self.index] = page
            self.cards_stack.addWidget(page)
        self.cards_stack.setCurrentWidget(page)
        self.body.setCurrentWidget(self.cards_stack)
        self.progress.setValue(self.index + 1)
        self.progress_label.setText(cards_lib.progress_text(self.cards, self.index))
        self._positions.put(self.session, {"index": self.index, "finished": False})
        self._sync_nav()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def next(self) -> None:
        if not self._loaded or not self.cards:
            return
        if self.is_finished_screen():
            return
        if self.index >= len(self.cards) - 1:
            self._show_finished()
            return
        self.go_to(self.index + 1)

    def back(self) -> None:
        if not self.cards:
            return
        if self.is_finished_screen():
            self.go_to(len(self.cards) - 1)
            return
        if self.index > 0:
            self.go_to(self.index - 1)

    def _sync_nav(self) -> None:
        loaded = self._loaded and bool(self.cards)
        self.back_button.setEnabled(loaded and (self.index > 0 or self.is_finished_screen()))
        self.next_button.setEnabled(loaded and not self.is_finished_screen())
        last = loaded and self.index >= len(self.cards) - 1
        self.next_button.setText("Finish →" if last else "Next →")
        self.next_button.setVisible(not self.is_finished_screen())

    def _show_finished(self, *, remember: bool = True) -> None:
        self._leave_clue_mode()
        lines = []
        for card in self.cards:
            said = self.answers.get(card["id"])
            if said:
                lines.append(f"• {card.get('title')}: " + "; ".join(said))
        self.summary_label.setText("\n".join(lines) if lines else "You did not save any answers this time.")
        self.rule_label.setText(f"Tomorrow's rule: {self.tomorrow_rule or 'none set'}")
        self.body.setCurrentWidget(self.finished_page)
        self.progress.setValue(self.progress.maximum())
        self.progress_label.setText(f"{len(self.cards)} of {len(self.cards)} · done")
        self._sync_nav()
        if remember:
            self._positions.put(self.session, {"index": self.index, "finished": True})
            self.walkFinished.emit(self.session)
        self._rebuild_if_saved()

    def exit_walk(self) -> None:
        self._leave_clue_mode()
        self._rebuild_if_saved()
        self.exited.emit()

    def _rebuild_if_saved(self) -> None:
        """Rebuild the day's record on a worker when something new was saved."""
        if not self._saves_since_rebuild:
            return
        self._saves_since_rebuild = 0
        self.rebuilds_started += 1
        session, rebuilder = self.session, self._rebuilder
        self._run(
            lambda: rebuilder(session),
            lambda _result: None,
            lambda reason: logging.info("The day record was not rebuilt: %s", reason),
        )

    # -- keys ----------------------------------------------------------------
    def _leave_clue_mode(self) -> bool:
        left = False
        for flow in self._flows.values():
            try:
                if flow.marker.is_active():
                    flow.set_active(False)
                    left = True
            except Exception as exc:  # noqa: BLE001
                note_swallowed("clue mode could not be left for one chart", exc, quiet=True)
        for button in self._clue_buttons.values():
            button.setChecked(False)
        return left

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if not self._leave_clue_mode():
                self.exit_walk()
            event.accept()
            return
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Space):
            self.next()
            event.accept()
            return
        if key == Qt.Key.Key_Left:
            self.back()
            event.accept()
            return
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            self.choose(int(key) - int(Qt.Key.Key_1))
            event.accept()
            return
        super().keyPressEvent(event)

    def choose(self, position: int) -> bool:
        """Press option `position` (0-based) of the current card's first choice row."""
        card = self.current_card()
        if card is None:
            return False
        buttons = self._primary.get(self.index) or []
        if 0 <= position < len(buttons) and buttons[position].isEnabled():
            buttons[position].click()
            return True
        return False

    # -- saving --------------------------------------------------------------
    def _status(self, index: int, text: str) -> None:
        label = self._status_labels.get(index)
        if label is not None:
            label.setText(text)
        self.statusChanged.emit(text)

    def save(
        self, index: int, slot: str, call: Callable[[], Any], summary: str,
        on_saved: Callable[[], Any] | None = None,
    ) -> None:
        """Run one write on a worker; the card says saved / not saved."""
        card = self.cards[index]
        self._status(index, "saving…")

        def done(result: Any) -> None:
            row = result if isinstance(result, Mapping) else {}
            if row.get("id"):
                self._last_ids[slot] = str(row["id"])
            said = self.answers.setdefault(card["id"], [])
            prefix = slot.split(":", 1)[0] + ":"
            said[:] = [s for s in said if not s.startswith(prefix)] + [f"{prefix} {summary}"]
            self.saved_count += 1
            self._saves_since_rebuild += 1
            self._status(index, f"saved: {summary}")
            if on_saved is not None:
                on_saved()

        def failed(reason: str) -> None:
            self._status(index, f"not saved: {reason}")

        self._run(call, done, failed)

    def _supersedes(self, slot: str) -> str:
        return self._last_ids.get(slot, "")

    # -- the card pages --------------------------------------------------------
    def _build_card_page(self, index: int, card: Mapping[str, Any]) -> QWidget:
        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(6)
        title = QLabel(str(card.get("title") or ""))
        title.setObjectName("SectionTitle")
        title.setWordWrap(True)
        body.addWidget(title)
        self._add_chart(index, card, body)
        for line in card.get("lines") or ():
            label = QLabel(str(line))
            label.setWordWrap(True)
            body.addWidget(label)
        kind = card.get("kind")
        if kind == "trade":
            self._trade_parts(index, card, body)
        elif kind == "miss":
            look = card.get("where_to_look") or ()
            if look:
                label = QLabel("Where to look next time:\n" + "\n".join(f"• {line}" for line in look))
                label.setWordWrap(True)
                body.addWidget(label)
            self._answer_row(index, card, body)
        elif kind in ("good_pass", "call"):
            self._answer_row(index, card, body)
        elif kind == "environment":
            self._environment_parts(index, card, body)
        elif kind == "lesson":
            self._lesson_parts(index, card, body)
        status = QLabel("")
        status.setObjectName("SectionSubtitle")
        status.setWordWrap(True)
        self._status_labels[index] = status
        body.addWidget(status)
        body.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setWidget(page)
        return scroll

    def _add_chart(self, index: int, card: Mapping[str, Any], body: QVBoxLayout) -> None:
        chart_data = card.get("chart")
        if not chart_data:
            missing = str(card.get("chart_missing") or "")
            if missing:
                note = QLabel(missing)
                note.setObjectName("SectionSubtitle")
                body.addWidget(note)
            return
        from ui.widgets.candle_chart import CandleChart

        chart = CandleChart()
        chart.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        chart.setMinimumHeight(theme.px(CHART_HEIGHT_PX))
        try:
            chart.set_data(list(chart_data.get("bars") or ()), timeframe="m5")
        except Exception:  # noqa: BLE001 - a bad tape costs the chart, never the card
            logging.debug("Walk chart not drawn.", exc_info=True)
            chart.deleteLater()
            body.addWidget(QLabel("The chart could not be drawn."))
            return
        try:
            chart.set_note_markers(tuple(chart_data.get("markers") or ()))
        except Exception:  # noqa: BLE001 - a marker never costs the card
            logging.debug("Walk chart markers not drawn.", exc_info=True)
        self._charts[index] = chart
        body.addWidget(chart)
        if self._clue_flow is None:
            return
        symbol = str(chart_data.get("symbol") or card.get("symbol") or "")
        try:
            flow = self._clue_flow(
                chart, self.session, symbol, "M5",
                card_id=str(card.get("id") or ""), trade_id=str(card.get("trade_id") or ""),
                spy_bars=self._payload.get("spy_m5_bars") or (),
            )
        except Exception:  # noqa: BLE001 - no clue mode, the card still works
            logging.debug("Walk clue flow not wired.", exc_info=True)
            return
        self._flows[index] = flow
        try:
            flow.load()
        except Exception:  # noqa: BLE001
            logging.debug("Walk clues not loaded.", exc_info=True)
        button = QPushButton("Mark a clue on the SPY chart" if symbol == "SPY" else "Mark a clue")
        button.setCheckable(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setToolTip("Click a bar to say what you see there now. Esc leaves clue mode.")
        button.toggled.connect(lambda on, f=flow: f.set_active(on))
        try:
            flow.marker.activeChanged.connect(button.setChecked)
        except Exception as exc:  # noqa: BLE001
            note_swallowed("clue mode button not linked to its marker", exc, quiet=True)
        self._clue_buttons[index] = button
        row = QHBoxLayout()
        row.addWidget(button)
        row.addStretch(1)
        body.addLayout(row)

    def clue_button(self, index: int) -> QPushButton | None:
        return self._clue_buttons.get(index)

    def _buttons(self, index: int, options, on_click, *, primary: bool) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        made: list[QPushButton] = []
        for position, (value, label) in enumerate(options):
            number = position + 1
            numbered = primary and number <= 9 and str(label) != str(number)
            text = f"{number}  {label}" if numbered else str(label)
            button = QPushButton(text)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setProperty("walk_option", value)
            button.clicked.connect(lambda _checked=False, v=value: on_click(v))
            row.addWidget(button)
            made.append(button)
        row.addStretch(1)
        if primary and index not in self._primary:
            self._primary[index] = made
        return row

    def option_buttons(self, index: int) -> list[QPushButton]:
        return list(self._primary.get(index) or ())

    def _answer_row(self, index: int, card: Mapping[str, Any], body: QVBoxLayout) -> None:
        text = QLineEdit()
        text.setPlaceholderText("Anything to add? (optional, Enter saves)")
        chosen: dict[str, str] = {}

        def pick(option: str) -> None:
            chosen["option"] = option
            self._save_answer(index, option, text.text())

        body.addLayout(self._buttons(index, card.get("options") or (), pick, primary=True))
        text.returnPressed.connect(
            lambda: self._save_answer(index, chosen["option"], text.text()) if chosen.get("option")
            else self._status(index, "pick an answer first")
        )
        body.addWidget(text)

    def _save_answer(self, index: int, option: str, text: str) -> None:
        card = self.cards[index]
        slot = f"answer:{card['id']}"
        writer, session, supersedes = self._writer, self.session, self._supersedes(slot)
        self.save(
            index, slot,
            lambda: writer.card_answer(session=session, card=card, option=option, text=text, supersedes=supersedes),
            option.replace("_", " ") + (f" - {text}" if text else ""),
        )

    # trade -------------------------------------------------------------------
    def _trade_parts(self, index: int, card: Mapping[str, Any], body: QVBoxLayout) -> None:
        whatif = QLabel(str(card.get("what_if") or ""))
        whatif.setWordWrap(True)
        body.addWidget(whatif)
        writer, session = self._writer, self.session
        for number, question in enumerate(card.get("mentor") or ()):
            prompt = QLabel(str(question.get("prompt") or ""))
            prompt.setWordWrap(True)
            body.addWidget(prompt)
            subject = question.get("subject")

            def answer(option: str, s=subject, n=number) -> None:
                self.save(
                    index, f"mentor{n}:{card['id']}",
                    lambda: writer.mentor_answer(session=session, card=card, subject=s, option=option),
                    option.replace("_", " "),
                )

            body.addLayout(self._buttons(index, question.get("options") or (), answer, primary=number == 0))
        draft = card.get("exit_draft")
        if draft:
            self._exit_draft_parts(index, card, draft, body)
        note = card.get("exit_note")
        if note:
            label = QLabel("Why did you exit? A few words are enough.")
            body.addWidget(label)
            box = QLineEdit()
            box.setPlaceholderText("Your exit note (Enter saves)")
            box.returnPressed.connect(lambda: self._save_exit_note(index, note, box.text()))
            body.addWidget(box)

    def _save_exit_note(self, index: int, note: Mapping[str, Any], text: str) -> None:
        if not text.strip():
            self._status(index, "not saved: the exit note is empty")
            return
        writer = self._writer
        self.save(
            index, f"exit_note:{note.get('trade_id')}",
            lambda: writer.exit_note(trade_id=note["trade_id"], exit_session=note["exit_session"], text=text),
            f"exit note - {text}",
        )

    def _exit_draft_parts(self, index: int, card, draft: Mapping[str, Any], body: QVBoxLayout) -> None:
        said = str(draft.get("raw_text") or "")
        if said:
            words = QLabel(f"You wrote: {said}")
            words.setWordWrap(True)
            body.addWidget(words)
        sentence = QLabel(str(draft.get("sentence") or "The night read your exit note."))
        sentence.setWordWrap(True)
        body.addWidget(sentence)
        fix = QWidget()
        fix_row = QHBoxLayout(fix)
        fix_row.setContentsMargins(0, 0, 0, 0)
        why = QComboBox()
        why.addItem("- why you exited -", "")
        for code, label in draft.get("reasons") or ():
            why.addItem(str(label), str(code))
        watching = QLineEdit()
        watching.setPlaceholderText("What you were watching (use ; between)")
        save = QPushButton("Save my version")
        save.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        fix_row.addWidget(why)
        fix_row.addWidget(watching, 1)
        fix_row.addWidget(save)
        fix.setVisible(False)
        writer = self._writer

        def correct() -> None:
            code = str(why.currentData() or "")
            quotes = tuple(p.strip() for p in watching.text().split(";") if p.strip())
            self.save(
                index, f"exit_draft:{draft.get('key')}",
                lambda: writer.exit_correct(draft=draft, why=code, watching=quotes),
                "exit fixed" + (f" - {why.currentText()}" if code else ""),
            )

        def choose(option: str) -> None:
            if option == "yes":
                self.save(
                    index, f"exit_draft:{draft.get('key')}",
                    lambda: writer.exit_confirm(draft=draft), "exit reading confirmed",
                )
            else:
                fix.setVisible(True)

        save.clicked.connect(correct)
        watching.returnPressed.connect(correct)
        body.addLayout(self._buttons(index, (("yes", "Yes, right"), ("fix", "Fix it")), choose, primary=True))
        body.addWidget(fix)

    # environment -------------------------------------------------------------
    def _environment_parts(self, index: int, card: Mapping[str, Any], body: QVBoxLayout) -> None:
        chips: dict[str, QPushButton] = {}
        chip_row = QHBoxLayout()
        chip_row.setSpacing(4)
        chip_row.addWidget(QLabel("Clues:"))
        try:
            from ui.widgets.clue_marker import CLUE_TAG_LABELS
        except Exception:  # noqa: BLE001
            CLUE_TAG_LABELS = {}
        for tag in card.get("clue_tags") or ():
            chip = QPushButton(CLUE_TAG_LABELS.get(tag, str(tag).replace("_", " ")))
            chip.setCheckable(True)
            chip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            chips[tag] = chip
            chip_row.addWidget(chip)
        chip_row.addStretch(1)
        text = QLineEdit()
        text.setPlaceholderText("What did the market really do? (optional, Enter saves)")
        chosen: dict[str, str] = {}
        writer, session = self._writer, self.session
        auto = str(card.get("auto_label") or "unknown")

        def verdict(option: str) -> None:
            chosen["verdict"] = option
            tags = [tag for tag, chip in chips.items() if chip.isChecked()]
            words = text.text().strip()
            full = (f"Clues: {', '.join(tags)}. " if tags else "") + words
            slot = "environment"
            supersedes = self._supersedes(slot)

            def call():
                ids = writer.clue_ids(session=session) if hasattr(writer, "clue_ids") else []
                return writer.environment_verdict(
                    session=session, auto_label=auto, verdict=option, clue_ids=ids,
                    text=full.strip(), supersedes=supersedes,
                )

            self.save(index, slot, call, option.replace("_", " ") + (f" - {full.strip()}" if full.strip() else ""))

        body.addLayout(self._buttons(index, card.get("options") or (), verdict, primary=True))
        body.addLayout(chip_row)
        text.returnPressed.connect(
            lambda: verdict(chosen["verdict"]) if chosen.get("verdict")
            else self._status(index, "pick agree or a label first")
        )
        body.addWidget(text)

    # lesson ------------------------------------------------------------------
    def _lesson_parts(self, index: int, card: Mapping[str, Any], body: QVBoxLayout) -> None:
        writer, session = self._writer, self.session
        rule = card.get("rule")
        if rule:
            label = QLabel(f"Did you keep yesterday's rule? “{rule.get('text', '')}”")
            label.setWordWrap(True)
            body.addWidget(label)

            def check(answer: str) -> None:
                slot = "rule_check"
                supersedes = self._supersedes(slot)
                self.save(
                    index, slot,
                    lambda: writer.rule_check(
                        session=session, answer=answer, rule_id=str(rule.get("id") or ""), supersedes=supersedes,
                    ),
                    f"kept the rule: {answer}",
                )

            body.addLayout(self._buttons(index, card.get("rule_options") or (), check, primary=True))
        streak = int(card.get("streak") or 0)
        body.addWidget(QLabel(f"Rule streak: {streak} session{'s' if streak != 1 else ''} in a row"))

        boxes: dict[str, QLineEdit] = {}
        for key, words in (("keep", "Keep doing"), ("stop", "Stop doing"), ("try_", "Try tomorrow")):
            row = QHBoxLayout()
            row.addWidget(QLabel(words))
            box = QLineEdit()
            box.setMaxLength(280)
            box.setPlaceholderText("Enter saves")
            boxes[key] = box
            row.addWidget(box, 1)
            body.addLayout(row)
        mood: dict[str, int] = {}

        def save_lesson() -> None:
            values = {key: box.text().strip() for key, box in boxes.items()}
            score = mood.get("score")
            slot = "lesson"
            supersedes = self._supersedes(slot)
            shown = ", ".join(f"{k.rstrip('_')}: {v}" for k, v in values.items() if v)
            self.save(
                index, slot,
                lambda: writer.lesson(session=session, mood=score, supersedes=supersedes, **values),
                (shown + (f", mood {score}" if score else "")).strip(", ") or "lesson",
            )

        def set_mood(score: int) -> None:
            mood["score"] = int(score)
            save_lesson()

        mood_row = QHBoxLayout()
        mood_label = QLabel("Mood (1 rough - 5 great)")
        mood_row.addWidget(mood_label)
        mood_row.addLayout(self._buttons(
            index, [(m, str(m)) for m in card.get("moods") or ()], set_mood, primary=not rule,
        ))
        body.addLayout(mood_row)
        for box in boxes.values():
            box.returnPressed.connect(save_lesson)

        rule_row = QHBoxLayout()
        rule_row.addWidget(QLabel("One rule for tomorrow"))
        rule_text = QLineEdit()
        rule_text.setMaxLength(280)
        rule_text.setPlaceholderText("Enter saves")
        tag = QComboBox()
        tag.addItem("- tag -", "")
        for value in card.get("rule_tags") or ():
            tag.addItem(str(value).replace("_", " "), str(value))
        rule_row.addWidget(rule_text, 1)
        rule_row.addWidget(tag)
        body.addLayout(rule_row)

        def save_rule() -> None:
            words = rule_text.text().strip()
            if not words:
                self._status(index, "not saved: a rule needs text")
                return
            slot = "rule"
            supersedes = self._supersedes(slot)
            chosen_tag = str(tag.currentData() or "")

            def remember() -> None:
                self.tomorrow_rule = words

            self.save(
                index, slot,
                lambda: writer.rule(session=session, text=words, tag=chosen_tag, supersedes=supersedes),
                f"rule for tomorrow - {words}", on_saved=remember,
            )

        rule_text.returnPressed.connect(save_rule)
        self._lesson_widgets = {"boxes": boxes, "rule_text": rule_text, "rule_tag": tag}

    def lesson_widgets(self) -> dict[str, Any]:
        return dict(self._lesson_widgets)


__all__ = ["KEY_HINT", "RecapWalk", "WALK_SETTING_KEY", "WalkPositions", "WalkWriter", "walk_finished"]
