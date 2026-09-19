"""The Trade Mentor's card - WISHLIST 10J, packet WS-TM item 3.

Small, modeless, and shown in the reusable Trade Mentor popup only when needed.
It asks one question and gives three answers: **Submit**, **Read unchanged**,
**Skip**. Everything else about it is a refusal to get in the way.

**The scheduled show does not take focus.** The hour can turn while the trader
is typing a symbol into the arm bar's ticker box; a `setFocus()` here would eat
that keystroke, and a modal dialog would eat the whole minute. The popup is
shown with `WA_ShowWithoutActivating` and never calls `setFocus`, `raise_` or
`activateWindow`; a trader click still focuses its ordinary text box. The keys
it does own (`Ctrl+Enter` to submit) are read through an event filter ON THE
TEXT BOX, so they mean "submit" only while the cursor is in the card - a
`QShortcut` at window scope would fire for every widget in the page.

**The raw text goes to the store FIRST, through the store's one owner.**
`market_journal_service.write_entry` writes it; nothing is parsed, scored,
summarised or interpreted on the way in. Step 3 of the trader's brief (a local
model filling a form from the text) reads the stored row LATER and is not in
this packet - which is the whole reason the raw text is written first.

**An unanswered prompt is no observation.** Half-typed text is kept in
`trade_mentor_drafts.json` and is NEVER a journal row, never shown as a read and
never counted as an answer. It survives the next hour replacing the card,
because the trader typed it and the desk does not get to throw away what the
trader typed.

**"Read unchanged" is a NEW row, not a copy and not a correction.** It restates
the previous read at the current time with `reaffirms` naming it. It does not
supersede: the 09:00 read must still be readable beside the 11:00 one, or "my
view has not changed for two hours" becomes indistinguishable from "I only ever
said it once".

Nothing here reaches a detector, a score, a gate, an alert, a watchlist, Focus,
the review queue or `review_policy.json`, and nothing here pushes to a phone.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from PySide6.QtCore import QEvent, QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from trade_mentor_schedule import (
    KIND_M5_D1,
    KIND_M5_TRADES,
    KIND_MANUAL,
    PACIFIC,
    MentorSlot,
    manual_slot,
)

#: The trader's own reason when they dismiss a card by hand. Kept distinct from
#: every absence reason the service records: "I looked and had nothing to say"
#: is a different fact from "nobody was there".
SKIP_TRADER = "trader_skip"

_QUESTIONS = {
    "m5": "What do you see on the 5-minute tape right now?",
    KIND_M5_D1: "What do you see on the 5-minute tape right now?",
    KIND_M5_TRADES: "What do you see on the 5-minute tape right now?",
    KIND_MANUAL: "Your read, right now.",
}

_D1_QUESTION = "And the daily picture?"


class _MentorAIWorkerSignals(QObject):
    ready = Signal(str, object)
    failed = Signal(str, str)


class _MentorAIWorker(QRunnable):
    """One bounded local-model call, always outside the Qt thread."""

    def __init__(self, trade_id: str, raw_text: str, missing: tuple[str, ...], trade: dict):
        super().__init__()
        self.trade_id = trade_id
        self.raw_text = raw_text
        self.missing = missing
        self.trade = trade
        self.signals = _MentorAIWorkerSignals()

    def run(self) -> None:
        try:
            from trade_mentor_ai import extract_draft

            draft = extract_draft(self.raw_text, self.missing, self.trade)
        except Exception as exc:  # noqa: BLE001 - raw answer already survived
            self.signals.failed.emit(self.trade_id, str(exc))
            return
        self.signals.ready.emit(self.trade_id, draft)


class TradeMentorCard(QWidget):
    """One prompt, one answer, filed once."""

    #: (slot_id) - the trader answered. The host tells the service, which never
    #: shows the slot again.
    answered = Signal(str)
    #: (dict) - `{"slot_id", "skipped_reason"}`. The trader dismissed the card.
    skipped = Signal(dict)
    #: (str) - a line for the host's status area. Never a dialog.
    statusChanged = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        journal=None,
        clock: Callable[[], datetime] | None = None,
        drafts_path: Path | None = None,
        context_service=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("TradeMentorCard")
        self._journal = journal
        self._clock = clock or self._default_clock
        if drafts_path is None:
            from project_paths import TRADE_MENTOR_DRAFTS_FILE

            drafts_path = TRADE_MENTOR_DRAFTS_FILE
        self._drafts_path = Path(drafts_path)
        self._drafts: dict[str, str] = {}
        self._load_drafts()
        self._slot: MentorSlot | None = None
        self._previous: Mapping[str, Any] | None = None
        self._submitted: set[str] = set()
        self._context_service = None
        self._context_slot_id = ""
        self._current_context: dict[str, Any] | None = None
        self.set_context_service(context_service)

        # Never activates the window it appears in. This is the whole of the
        # "no focus stealing" promise and it costs one attribute.
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.prompt_label = QLabel("")
        self.prompt_label.setObjectName("SectionTitle")
        self.prompt_label.setWordWrap(True)
        self.previous_label = QLabel("")
        self.previous_label.setObjectName("MutedLabel")
        self.previous_label.setWordWrap(True)
        self.previous_label.setVisible(False)
        self.coaching_label = QLabel("")
        self.coaching_label.setObjectName("MutedLabel")
        self.coaching_label.setWordWrap(True)
        self.coaching_label.setVisible(False)

        self.text_box = QPlainTextEdit(self)
        self.text_box.setPlaceholderText(
            "In your own words. Nothing here is parsed or scored - it is stored "
            "exactly as you type it."
        )
        self.text_box.setMaximumHeight(84)
        self.text_box.installEventFilter(self)

        self.d1_label = QLabel(_D1_QUESTION)
        self.d1_label.setObjectName("MutedLabel")
        self.d1_box = QPlainTextEdit(self)
        self.d1_box.setMaximumHeight(84)
        self.d1_box.installEventFilter(self)
        self.d1_label.setVisible(False)
        self.d1_box.setVisible(False)

        # The 09:00 card's SECOND section, kept visually separate from the read
        # above it. Two different questions on one card is the trader's own
        # design; merging them into one box would produce a paragraph that is
        # neither a market read nor a record of a trade.
        self.trade_check_label = QLabel("")
        self.trade_check_label.setObjectName("MutedLabel")
        self.trade_check_label.setWordWrap(True)
        self.trade_check_label.setVisible(False)
        self.trade_check_box = QWidget(self)
        self._trade_check_layout = QVBoxLayout(self.trade_check_box)
        self._trade_check_layout.setContentsMargins(0, 0, 0, 0)
        self._trade_check_layout.setSpacing(3)
        self.trade_check_box.setVisible(False)
        #: trade_id -> field -> (state combo, free-text box)
        self._answer_inputs: dict[str, dict[str, tuple[QComboBox, QLineEdit]]] = {}
        self._trade_questions: dict[str, Any] = {}
        self._raw_trade_inputs: dict[str, QPlainTextEdit] = {}
        self._ai_draft_buttons: dict[str, QPushButton] = {}
        self._ai_drafts: dict[str, dict[str, dict[str, Any]]] = {}
        self._setup_confirm_buttons: dict[str, QPushButton] = {}
        #: trade_id -> the vocabulary list the confirm button sits beside.
        self._setup_choice_boxes: dict[str, QComboBox] = {}
        #: trade_ids whose setup the trader confirmed on THIS card. The combo
        #: for that field disappears, so the Save gate must stop waiting on it.
        self._setup_confirmed: set[str] = set()
        self._trade_store = None
        #: Which session's trades the section is asking about. TJ-9 item 2: an
        #: unanswered section RIDES on every later card of the same session and
        #: is cleared only when the session changes - a question about Friday's
        #: trades asked on Wednesday is a different question.
        self._trade_check_session = ""
        self.save_answers_button = QPushButton("Save answers")
        self.save_answers_button.setToolTip(
            "Files what you remember as a labelled next-morning note. It never "
            "becomes the plan you typed before the trade, and 'no stop' is "
            "never written as a stop at zero."
        )
        self.save_answers_button.clicked.connect(self.save_trade_check)
        self.save_answers_button.setVisible(False)
        self.save_answers_button.setEnabled(False)

        self.submit_button = QPushButton("Submit")
        self.submit_button.setToolTip("File this read now (Ctrl+Enter).")
        self.submit_button.clicked.connect(self.submit)
        self.unchanged_button = QPushButton("Read unchanged")
        self.unchanged_button.setToolTip(
            "Files a NEW observation at this time that restates your previous "
            "read. The earlier one stays exactly as you wrote it."
        )
        self.unchanged_button.clicked.connect(self.read_unchanged)
        self.skip_button = QPushButton("Skip")
        self.skip_button.setToolTip(
            "Nothing is filed. An unanswered prompt is not an observation, and "
            "anything you typed is kept as a draft."
        )
        self.skip_button.clicked.connect(self.skip)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        buttons.addWidget(self.submit_button)
        buttons.addWidget(self.unchanged_button)
        buttons.addWidget(self.skip_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addWidget(self.prompt_label)
        layout.addWidget(self.previous_label)
        layout.addWidget(self.coaching_label)
        layout.addWidget(self.text_box)
        layout.addWidget(self.d1_label)
        layout.addWidget(self.d1_box)
        layout.addWidget(self.trade_check_label)
        layout.addWidget(self.trade_check_box)
        layout.addWidget(self.save_answers_button)
        layout.addLayout(buttons)
        layout.addWidget(self.status_label)

        self.setVisible(False)

    # -- plumbing ---------------------------------------------------------
    @staticmethod
    def _default_clock() -> datetime:
        return datetime.now(PACIFIC)

    def _now(self) -> datetime:
        moment = self._clock()
        stamp = moment if moment.tzinfo else moment.astimezone()
        return stamp.astimezone(PACIFIC)

    def _service(self):
        if self._journal is None:
            from ui.services.market_journal_service import shared_journal_service

            self._journal = shared_journal_service()
        return self._journal

    def set_context_service(self, context_service) -> None:
        """Attach the window-owned reader once; cards never own a worker."""
        if context_service is self._context_service:
            return
        if self._context_service is not None:
            try:
                self._context_service.contextReady.disconnect(self._on_context_ready)
                self._context_service.contextUnavailable.disconnect(self._on_context_unavailable)
            except (RuntimeError, TypeError):
                pass
        self._context_service = context_service
        if context_service is not None:
            context_service.contextReady.connect(self._on_context_ready)
            context_service.contextUnavailable.connect(self._on_context_unavailable)

    # -- drafts -----------------------------------------------------------
    def _load_drafts(self) -> None:
        try:
            payload = json.loads(self._drafts_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(payload, dict):
            for slot_id, record in payload.items():
                if isinstance(record, Mapping):
                    self._drafts[str(slot_id)] = str(record.get("text") or "")

    def _save_drafts(self) -> None:
        """Never costs the card. A draft that could not be written is a lost
        half-thought; a card that refused to move on would be a lost hour."""
        try:
            self._drafts_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                slot_id: {"text": text} for slot_id, text in self._drafts.items() if text
            }
            tmp = self._drafts_path.with_name(self._drafts_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._drafts_path)
        except OSError:
            logging.debug("Trade Mentor draft not saved.", exc_info=True)

    def draft_for(self, slot_id: str) -> str:
        """Whatever was typed against `slot_id` and never submitted."""
        return self._drafts.get(str(slot_id), "")

    def _stash_draft(self) -> None:
        """Keep what is in the boxes, exactly as typed - trailing space and all."""
        if self._slot is None:
            return
        text = self.text_box.toPlainText()
        d1_text = self.d1_box.toPlainText()
        combined = text if not d1_text else f"{text}\n{d1_text}"
        if combined.strip():
            self._drafts[self._slot.slot_id] = combined
            self._save_drafts()

    def _drop_draft(self, slot_id: str) -> None:
        if self._drafts.pop(str(slot_id), None) is not None:
            self._save_drafts()

    # -- showing ----------------------------------------------------------
    def show_slot(self, slot: MentorSlot, previous: Mapping[str, Any] | None = None) -> None:
        """Put this prompt up, replacing whatever was there.

        A new hour REPLACES an untouched card rather than stacking beside it -
        one card is the trader's rule - and whatever was half-typed in the old
        one is stashed on the way out.
        """
        self._stash_draft()
        self._slot = slot
        # A prompt owns its own snapshot.  Saving while the worker is still
        # running records that absence now; a later result cannot mutate a
        # journal row that already exists.
        moment = self._now()
        self._context_slot_id = str(slot.slot_id)
        self._current_context = self._unavailable_context(
            moment, "context pending"
        )
        if self._context_service is not None:
            try:
                accepted = self._context_service.request_context(slot.slot_id, now=moment)
                if not accepted:
                    self._current_context = self._unavailable_context(
                        moment, "context unavailable or throttled"
                    )
            except Exception:  # noqa: BLE001 - context never costs a raw note
                self._current_context = self._unavailable_context(
                    moment, "context request failed"
                )
        self._previous = dict(previous) if previous else None
        kind = str(getattr(slot, "kind", "") or "")
        self.prompt_label.setText(_QUESTIONS.get(kind, _QUESTIONS[KIND_MANUAL]))
        restored = self.draft_for(slot.slot_id)
        self.text_box.setPlainText(restored)
        self.d1_box.setPlainText("")
        show_d1 = kind == KIND_M5_D1
        self.d1_label.setVisible(show_d1)
        self.d1_box.setVisible(show_d1)
        if str(getattr(slot, "session", "") or "") != self._trade_check_session:
            # TJ-9 item 2. The section RIDES on every later card of the same
            # session - an unanswered card must not expire into silence - and
            # is cleared the moment the session changes, so a stale question
            # from Friday can never be saved against Wednesday's morning.
            self._clear_trade_check()
            self.trade_check_label.setVisible(False)
            self.trade_check_box.setVisible(False)
            self.save_answers_button.setVisible(False)
            self._trade_check_session = ""
        if self._previous:
            self.previous_label.setText(
                "Your last read: " + str(self._previous.get("text") or "")
            )
            self.previous_label.setVisible(True)
        else:
            self.previous_label.setText("")
            self.previous_label.setVisible(False)
        self.unchanged_button.setEnabled(bool(self._previous))
        try:
            from ai_jobs.market_story_narration import latest_coaching_question

            coaching = latest_coaching_question()
        except Exception:  # noqa: BLE001 - coaching never costs the prompt
            coaching = ""
        self.coaching_label.setText(f"One thing to test: {coaching}" if coaching else "")
        self.coaching_label.setVisible(bool(coaching))
        self.status_label.setText(
            "Post-close read." if bool(getattr(slot, "post_close", False)) else ""
        )
        self.setVisible(True)

    @staticmethod
    def _unavailable_context(moment: datetime, reason: str) -> dict[str, Any]:
        from trade_mentor_context import unavailable_context

        return unavailable_context(now=moment, reason=reason)

    def _on_context_ready(self, request_id: str, context: object) -> None:
        if str(request_id) == self._context_slot_id and isinstance(context, Mapping):
            self._current_context = dict(context)

    def _on_context_unavailable(self, request_id: str, context: object) -> None:
        if str(request_id) == self._context_slot_id and isinstance(context, Mapping):
            self._current_context = dict(context)

    def _clear_trade_check(self) -> None:
        self._answer_inputs = {}
        self._trade_questions = {}
        self._raw_trade_inputs = {}
        self._ai_draft_buttons = {}
        self._ai_drafts = {}
        self._setup_confirm_buttons = {}
        self._setup_choice_boxes = {}
        self._setup_confirmed = set()
        self.save_answers_button.setEnabled(False)
        while self._trade_check_layout.count():
            item = self._trade_check_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def set_trade_check(self, task, store=None) -> None:
        """Build the 09:00 card's second section from `build_task`'s answer.

        Three states, all of them said out loud:

        * the statement has not landed - one line that NAMES the date the fills
          are current to, and NO questions. An empty questionnaire drawn from an
          incomplete list is a lie about the session, and the line rides to the
          next slot rather than asking nothing all day;
        * nothing is missing - one line saying so;
        * something is missing - EVERY trade of the reviewed session, each with
          a state combo and a free-text box per missing field, and a one-click
          confirm beside the setup when the machine has a suggestion.

        Save is disabled until every listed field holds one of the four
        explicit answer states. That is the whole of "forced": the trader can
        say `not remembered`, which is a complete answer, but they cannot leave
        the morning blank by closing the card.
        """
        import trade_mentor_trade_check as check

        self._clear_trade_check()
        self._trade_store = store
        if task is None:
            self.trade_check_label.setVisible(False)
            self.trade_check_box.setVisible(False)
            self.save_answers_button.setVisible(False)
            self._trade_check_session = ""
            return
        self._trade_check_session = str(
            getattr(self._slot, "session", "") or ""
        )
        if not getattr(task, "journal_ready", False):
            self.trade_check_label.setText(
                f"Yesterday's trades ({task.reviewed_session}): "
                f"{task.reason or check.REASON_NOT_READY} - "
                f"{self._freshness_phrase(task)}. The broker statement has not "
                "landed, so nothing is asked yet; this comes back on the next "
                "card. The day pull is Questrade only - IBKR has no day leg."
            )
            self.trade_check_label.setVisible(True)
            self.trade_check_box.setVisible(False)
            self.save_answers_button.setVisible(False)
            return
        if not task.trades:
            self.trade_check_label.setText(
                f"Yesterday's trades ({task.reviewed_session}): nothing is missing. "
                f"{self._freshness_phrase(task).capitalize()}."
            )
            self.trade_check_label.setVisible(True)
            self.trade_check_box.setVisible(False)
            self.save_answers_button.setVisible(False)
            return

        remainder = (
            f" {task.remaining} more still missing fields - they stay in the "
            "Journal's completeness view."
            if task.remaining
            else ""
        )
        self.trade_check_label.setText(
            f"Yesterday's trades ({task.reviewed_session}), missing fields only - "
            f"all {len(task.trades)}. Save stays off until each one is answered."
            + remainder
        )
        self.trade_check_label.setVisible(True)

        for question in task.trades:
            self._trade_questions[question.trade_id] = question
            heading = QLabel(
                f"{question.symbol} {question.direction}".strip() or question.trade_id
            )
            heading.setObjectName("MutedLabel")
            self._trade_check_layout.addWidget(heading)
            self._add_setup_confirm(question)
            raw_box = QPlainTextEdit(self.trade_check_box)
            raw_box.setMaximumHeight(72)
            raw_box.setPlaceholderText(
                "Tell me in one note: why, stop/invalidation, target, and setup. "
                "Your exact words are saved before local AI fills the draft."
            )
            self._trade_check_layout.addWidget(raw_box)
            ai_button = QPushButton("Fill missing fields with local AI", self.trade_check_box)
            ai_button.clicked.connect(
                lambda _checked=False, trade_id=question.trade_id: self._start_ai_draft(
                    trade_id
                )
            )
            self._trade_check_layout.addWidget(ai_button)
            self._raw_trade_inputs[question.trade_id] = raw_box
            self._ai_draft_buttons[question.trade_id] = ai_button
            fields: dict[str, tuple[QComboBox, QLineEdit]] = {}
            for name in question.missing:
                row = QWidget(self.trade_check_box)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                row_layout.addWidget(QLabel(name))
                combo = QComboBox(row)
                # "-" first, so a field the trader did not touch stays unasked
                # rather than being filed as whatever happened to be at index 0.
                combo.addItem("-", "")
                for state in check.ANSWER_STATES:
                    combo.addItem(state.replace("_", " "), state)
                # The Save gate is a STATE, not a latch: going back to "-"
                # closes it again, which is why this listens to the combo
                # rather than counting clicks.
                combo.currentIndexChanged.connect(self._refresh_save_gate)
                text_input = QLineEdit(row)
                text_input.setPlaceholderText("in your own words (optional)")
                row_layout.addWidget(combo)
                row_layout.addWidget(text_input, 1)
                self._trade_check_layout.addWidget(row)
                fields[name] = (combo, text_input)
            self._answer_inputs[question.trade_id] = fields

        self.trade_check_box.setVisible(True)
        self.save_answers_button.setVisible(True)
        self._refresh_save_gate()

    @staticmethod
    def _freshness_phrase(task) -> str:
        """"fills current to <date>" - the one line every surface prints.

        The DATE comes from the task, never from the widget: the Journal and
        the AWAY digest print the same sentence from the same number.
        """
        current = str(getattr(task, "fills_current_to", "") or "")
        return f"fills current to {current}" if current else "no verified import yet"

    def _add_setup_confirm(self, question) -> None:
        """One click for the setup, when the machine has something to suggest.

        The button is a SUGGESTION until it is pressed. Showing it writes
        nothing - the row stays exactly as the bulk tagger left it - and
        pressing it is the trader's write through the Journal's own writer. A
        trade whose setup the trader already confirmed is never offered one.
        """
        import trade_mentor_trade_check as check

        guess = str(getattr(question, "setup_guess", "") or "")
        if not guess or "setup" not in tuple(question.missing or ()):
            return
        lane = str(getattr(question, "setup_guess_lane", "") or "")
        row = QWidget(self.trade_check_box)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        # The VOCABULARY LIST the confirm button sits beside. The guess is
        # preselected, so one click is still one click - but a wrong guess is
        # CORRECTED here rather than confirmed, and the list carries no
        # rejection, so nothing outside it can be written from this card.
        choice = QComboBox(row)
        names = [guess]
        for name in check.setup_vocabulary():
            if name not in names:
                names.append(name)
        for name in names:
            choice.addItem(name, name)
        choice.setCurrentIndex(0)
        button = QPushButton("Confirm setup", row)
        button.setToolTip(
            "The machine's best guess"
            + (f", from {lane.replace('_', ' ')}" if lane else "")
            + ". Nothing is written until you press this, and what is written "
            "is whatever this list shows."
        )
        button.clicked.connect(
            lambda _checked=False, trade_id=question.trade_id: self._confirm_setup(trade_id)
        )
        row_layout.addWidget(QLabel("setup"))
        row_layout.addWidget(choice, 1)
        row_layout.addWidget(button)
        self._trade_check_layout.addWidget(row)
        self._setup_confirm_buttons[str(question.trade_id)] = button
        self._setup_choice_boxes[str(question.trade_id)] = choice

    def setup_confirm_button(self, trade_id: str):
        """The confirm button offered for one trade, or ``None``."""
        return self._setup_confirm_buttons.get(str(trade_id))

    def setup_choice_box(self, trade_id: str):
        """The vocabulary list that button sits beside, or ``None``."""
        return self._setup_choice_boxes.get(str(trade_id))

    def trade_check_session(self) -> str:
        """Which session's trade check is on the card, or ``""``.

        The host asks before rebuilding: a section the trader has half-answered
        rides untouched, and rebuilding it would throw their combos away.
        """
        return self._trade_check_session if self._has_trade_check() else ""

    def _has_trade_check(self) -> bool:
        return bool(
            self._answer_inputs
            or self.trade_check_label.isVisibleTo(self)
            or self.trade_check_box.isVisibleTo(self)
        )

    def _confirm_setup(self, trade_id: str) -> dict[str, Any]:
        """The trader's click. The pure function decides the provenance."""
        import trade_mentor_trade_check as check

        question = self._trade_questions.get(str(trade_id))
        store = self._trade_store
        if question is None or store is None:
            self._set_status("the trade journal is not available here")
            return {"ok": False, "reason": "the trade journal is not available here"}
        choice = self._setup_choice_boxes.get(str(trade_id))
        chosen = str(choice.currentData() or choice.currentText() or "") if choice else ""
        try:
            result = check.confirm_setup(store, question, now=self._now(), setup=chosen)
        except Exception as exc:  # noqa: BLE001 - journal writes fail loudly
            self._set_status(f"the setup was NOT saved: {exc}")
            return {"ok": False, "reason": str(exc)}
        if not result.get("ok"):
            self._set_status(str(result.get("reason") or "nothing to confirm"))
            return result
        button = self._setup_confirm_buttons.get(str(trade_id))
        if button is not None:
            button.setEnabled(False)
            button.setText(f"Confirmed: {result.get('setup')}")
        if choice is not None:
            choice.setEnabled(False)
        # The setup question is answered, so the Save gate stops waiting on it.
        self._setup_confirmed.add(str(trade_id))
        fields = self._answer_inputs.get(str(trade_id), {})
        controls = fields.get("setup")
        if controls is not None:
            controls[0].setEnabled(False)
            controls[1].setEnabled(False)
        self._refresh_save_gate()
        self._set_status(
            f"{question.symbol} setup confirmed as {result.get('setup')} "
            f"({result.get('label_provenance') or 'unrecorded'})."
        )
        return result

    def _pending_answers(self) -> int:
        """How many listed fields are still open. Zero means Save may arm."""
        open_fields = 0
        for trade_id, fields in self._answer_inputs.items():
            confirmed = str(trade_id) in self._setup_confirmed
            for name, (combo, _text_input) in fields.items():
                if name == "setup" and confirmed:
                    continue
                if not str(combo.currentData() or ""):
                    open_fields += 1
        return open_fields

    def _refresh_save_gate(self, *_args) -> None:
        """Forced means the button is grey until every field is answered."""
        try:
            self.save_answers_button.setEnabled(
                bool(self._answer_inputs) and self._pending_answers() == 0
            )
        except RuntimeError:  # pragma: no cover - widget already torn down
            pass

    def _start_ai_draft(self, trade_id: str) -> None:
        """Save raw words, then let the local model prepare editable controls."""
        import trade_mentor_trade_check as check

        question = self._trade_questions.get(trade_id)
        raw_box = self._raw_trade_inputs.get(trade_id)
        button = self._ai_draft_buttons.get(trade_id)
        body = raw_box.toPlainText() if raw_box is not None else ""
        if question is None or not body.strip() or self._trade_store is None:
            self._set_status("Type your answer first. Nothing was sent.")
            return
        try:
            check.save_raw_reply(
                self._trade_store,
                trade_id,
                body,
                missing=tuple(question.missing),
                now=self._now(),
            )
        except Exception as exc:  # noqa: BLE001 - journal writes fail loudly
            self._set_status(f"Your words were NOT saved: {exc}")
            return
        if button is not None:
            button.setEnabled(False)
            button.setText("Local AI is filling the draft…")
        trade = {
            "trade_id": trade_id,
            "symbol": str(question.symbol or ""),
            "direction": str(question.direction or ""),
        }
        worker = _MentorAIWorker(trade_id, body, tuple(question.missing), trade)
        worker.signals.ready.connect(self._apply_ai_draft)
        worker.signals.failed.connect(self._ai_draft_failed)
        QThreadPool.globalInstance().start(worker)
        self._set_status("Your exact words are saved. Local AI is making an editable draft.")

    def _apply_ai_draft(self, trade_id: str, payload: object) -> None:
        draft = dict(payload) if isinstance(payload, Mapping) else {}
        fields = self._answer_inputs.get(trade_id, {})
        kept: dict[str, dict[str, Any]] = {}
        conflicts: list[str] = []
        for answer in draft.get("answers") or []:
            if not isinstance(answer, Mapping):
                continue
            name = str(answer.get("field") or "")
            controls = fields.get(name)
            if controls is None:
                continue
            combo, text_input = controls
            if combo.currentData() or text_input.text().strip():
                conflicts.append(name)
                continue
            state = str(answer.get("state") or "")
            index = combo.findData(state)
            if index < 0:
                continue
            combo.setCurrentIndex(index)
            text_input.setText(str(answer.get("text") or answer.get("source_span") or ""))
            kept[name] = dict(answer)
        self._ai_drafts[trade_id] = kept
        button = self._ai_draft_buttons.get(trade_id)
        if button is not None:
            button.setEnabled(True)
            button.setText("Refill from a new raw answer")
        follow_up = str(draft.get("follow_up") or "").strip()
        message = f"Draft filled for {len(kept)} field(s). Check it, then Save answers."
        if conflicts:
            message += " I kept your existing " + ", ".join(conflicts) + "."
        if follow_up:
            message += " One question: " + follow_up
        self._set_status(message)

    def _ai_draft_failed(self, trade_id: str, reason: str) -> None:
        button = self._ai_draft_buttons.get(trade_id)
        if button is not None:
            button.setEnabled(True)
            button.setText("Try local AI again")
        self._set_status(
            "Your exact words are safe. Local AI could not fill the draft. "
            "You can use the fields by hand. " + str(reason or "")
        )

    def save_trade_check(self) -> dict[str, Any]:
        """File every field the trader actually answered, and nothing else."""
        import trade_mentor_trade_check as check

        store = self._trade_store
        if store is None:
            return {"ok": False, "reason": "the trade journal is not available here"}
        moment = self._now()
        saved = 0
        for trade_id, fields in self._answer_inputs.items():
            answers: dict[str, dict[str, Any]] = {}
            for name, (combo, text_input) in fields.items():
                state = str(combo.currentData() or "")
                if not state:
                    continue
                answer = {"state": state, "text": text_input.text().strip()}
                ai_answer = self._ai_drafts.get(trade_id, {}).get(name, {})
                if (
                    ai_answer
                    and str(ai_answer.get("state") or "") == state
                    and str(ai_answer.get("text") or ai_answer.get("source_span") or "").strip()
                    == text_input.text().strip()
                ):
                    answer.update(
                        value=ai_answer.get("value"),
                        unit=str(ai_answer.get("unit") or ""),
                        source_span=str(ai_answer.get("source_span") or ""),
                    )
                answers[name] = answer
            if not answers:
                continue
            try:
                check.save_answers(store, trade_id, answers, now=moment)
                saved += len(answers)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Recalled fields not saved for %s: %s", trade_id, exc)
                self._set_status(f"answers NOT saved: {exc}")
                return {"ok": False, "reason": str(exc)}
        if not saved:
            self._set_status("Nothing was answered, so nothing was filed.")
            return {"ok": False, "reason": "no field was answered"}
        self._clear_trade_check()
        self.trade_check_box.setVisible(False)
        self.save_answers_button.setVisible(False)
        # The section rides on later cards of the same session (TJ-9 item 2),
        # so the heading has to stop describing questions that are now answered
        # - a later hour no longer wipes it on its way in.
        self.trade_check_label.setText(
            f"Yesterday's trades: {saved} remembered field(s) filed, labelled as recalled."
        )
        self._set_status(f"{saved} remembered field(s) filed, labelled as recalled.")
        return {"ok": True, "fields": saved}

    def give_a_read(self, now: datetime | None = None) -> MentorSlot:
        """The manual door, open at all times - no slot has to be due.

        A manual read is a real observation at a real time; it is marked
        `manual` so a later reader never counts it as an answered prompt.
        """
        moment = now.astimezone(PACIFIC) if now and now.tzinfo else (now or self._now())
        slot = manual_slot(moment)
        self.show_slot(slot)
        return slot

    def hide_card(self) -> None:
        self._stash_draft()
        self._slot = None
        self.setVisible(False)

    # -- answering --------------------------------------------------------
    def _session_for(self, slot: MentorSlot, moment: datetime) -> str:
        if str(getattr(slot, "kind", "")) != KIND_MANUAL:
            return str(slot.session)
        try:
            import market_journal

            return market_journal.session_date_for(moment)
        except Exception:  # noqa: BLE001 - a read is never lost to a calendar
            return moment.date().isoformat()

    def _mentor_payload(self, slot: MentorSlot, moment: datetime) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "slot_id": str(slot.slot_id),
            "prompt_kind": str(slot.kind),
            "scheduled_at": slot.scheduled_at.isoformat(),
            # The moment the trader actually replied. Separate from
            # `scheduled_at` because a reply typed at 09:12 cannot claim to
            # describe the market at 09:00, and separate from the ledger's own
            # `created_at` because that is a UTC machine stamp of the same
            # instant, not the trader's wall clock.
            "responded_at": moment.isoformat(),
        }
        payload["context"] = (
            self._current_context
            if self._context_slot_id == str(slot.slot_id) and self._current_context is not None
            else self._unavailable_context(moment, "context pending")
        )
        return payload

    def submit(self) -> dict[str, Any]:
        """File the raw text. Once per slot, whatever the button does."""
        slot = self._slot
        if slot is None:
            return {"ok": False, "reason": "nothing is being asked"}
        if slot.slot_id in self._submitted:
            # A double click is one read. The guard is here rather than on the
            # button because Ctrl+Enter reaches the same verb.
            return {"ok": False, "reason": "this read is already filed"}
        moment = self._now()
        text = self.text_box.toPlainText().strip()
        d1_text = self.d1_box.toPlainText().strip() if self.d1_box.isVisible() else ""
        if not text and not d1_text:
            self._set_status("Nothing typed, so nothing was filed.")
            return {"ok": False, "reason": "an empty read is not an observation"}

        session = self._session_for(slot, moment)
        payload = self._mentor_payload(slot, moment)
        written: list[dict[str, Any]] = []
        # The M5 read and the D1 read are stored SEPARATELY even though one card
        # collected both (the trader's brief). Two timeframes in one row would
        # be one row that is true of neither.
        for body, timeframe in ((text, "M5"), (d1_text, "D1")):
            if not body:
                continue
            result = self._service().write_entry(
                text=body,
                session_date=session,
                timeframe=timeframe,
                origin="trade_mentor",
                now=moment,
                mentor=payload,
            )
            if not result.get("ok"):
                self._set_status(str(result.get("reason") or "entry NOT saved"))
                return result
            written.append(result.get("entry") or {})

        self._submitted.add(slot.slot_id)
        self._drop_draft(slot.slot_id)
        self.text_box.setPlainText("")
        self.d1_box.setPlainText("")
        self._set_status(f"Filed at {moment.strftime('%H:%M')}.")
        self.answered.emit(slot.slot_id)
        self.setVisible(False)
        return {"ok": True, "entries": written}

    def read_unchanged(self) -> dict[str, Any]:
        """File a NEW row restating the previous read, at this time."""
        slot = self._slot
        if slot is None:
            return {"ok": False, "reason": "nothing is being asked"}
        previous = self._previous or {}
        body = str(previous.get("text") or "").strip()
        if not body:
            return {"ok": False, "reason": "there is no earlier read to reaffirm"}
        if slot.slot_id in self._submitted:
            return {"ok": False, "reason": "this read is already filed"}
        moment = self._now()
        result = self._service().write_entry(
            text=body,
            session_date=self._session_for(slot, moment),
            timeframe=str(previous.get("timeframe") or "M5"),
            origin="trade_mentor",
            now=moment,
            mentor=self._mentor_payload(slot, moment),
            # Names the read it restates, and deliberately NOT `supersedes`:
            # superseding would hide the 09:00 read behind the 11:00 one.
            reaffirms=str(previous.get("entry_id") or ""),
        )
        if not result.get("ok"):
            self._set_status(str(result.get("reason") or "entry NOT saved"))
            return result
        self._submitted.add(slot.slot_id)
        self._drop_draft(slot.slot_id)
        self._set_status(f"Read unchanged, filed at {moment.strftime('%H:%M')}.")
        self.answered.emit(slot.slot_id)
        self.setVisible(False)
        return result

    def skip(self) -> dict[str, Any]:
        """Dismiss without filing. Whatever was typed is kept as a draft."""
        slot = self._slot
        if slot is None:
            return {"ok": False, "reason": "nothing is being asked"}
        self._stash_draft()
        record = {"slot_id": str(slot.slot_id), "skipped_reason": SKIP_TRADER}
        self._slot = None
        self.setVisible(False)
        self.skipped.emit(dict(record))
        return record

    def _set_status(self, text: str) -> None:
        self.status_label.setText(str(text or ""))
        self.statusChanged.emit(str(text or ""))

    # -- keys -------------------------------------------------------------
    def eventFilter(self, watched, event):  # noqa: N802 (Qt override)
        """Ctrl+Enter submits, and only while the cursor is in this card.

        An event filter on the two boxes rather than a `QShortcut`: a shortcut
        lives at window scope, and a hidden card's shortcut competing with a
        live one is the fault CLAUDE.md records for the rail bindings - two
        bindings for one sequence fire neither.
        """
        try:
            if (
                watched in (self.text_box, self.d1_box)
                and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            ):
                self.submit()
                return True
        except Exception:  # noqa: BLE001 - a key handler never breaks the desk
            logging.debug("Trade Mentor key handling failed.", exc_info=True)
        return super().eventFilter(watched, event)
