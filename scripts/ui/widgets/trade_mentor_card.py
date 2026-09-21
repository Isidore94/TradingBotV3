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

from market_journal import (
    CONFIDENCE_LEVELS,
    DIRECTION_NO_VIEW,
    DIRECTIONS,
    HORIZON_NEXT_5_SESSIONS,
    HORIZON_REST_OF_DAY,
    TIMEFRAME_D1,
    TIMEFRAME_M5,
    build_prediction,
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

#: TJ-7's optional strip hangs off exactly ONE question kind, and the name is
#: the registry's own (`mentor_questions.KIND_DAY_CLOSE`). It is spelled here
#: rather than imported at module scope because `mentor_questions` is imported
#: lazily on this card's seams; the two being the same string is asserted.
DAY_CLOSE_KIND = "day_close"

_QUESTIONS = {
    "m5": "What do you see on the 5-minute tape right now?",
    KIND_M5_D1: "What do you see on the 5-minute tape right now?",
    KIND_M5_TRADES: "What do you see on the 5-minute tape right now?",
    KIND_MANUAL: "Your read, right now.",
}

_D1_QUESTION = "And the daily picture?"

#: TJ-14A item 1. The card has TWO labelled parts and they never share a field.
SEE_HEADING = "What I see  ·  the tape right now, in your own words (optional)"
EXPECT_HEADING = "What I expect  ·  one call, and how sure you are"

#: What each horizon's button row says about itself. The horizon is PRINTED on
#: the row, so a call can never be read back against the wrong clock.
HORIZON_LABELS = {
    HORIZON_REST_OF_DAY: "Rest of day:",
    HORIZON_NEXT_5_SESSIONS: "Next 5 sessions:",
}
#: The order the two rows appear in, each under the words it belongs to.
HORIZONS = (HORIZON_REST_OF_DAY, HORIZON_NEXT_5_SESSIONS)
#: Which timeframe each horizon's answer is filed as (the M5 entry carries the
#: rest-of-day call, the D1 entry the five-session one).
TIMEFRAME_FOR_HORIZON = {
    HORIZON_REST_OF_DAY: TIMEFRAME_M5,
    HORIZON_NEXT_5_SESSIONS: TIMEFRAME_D1,
}
_DIRECTION_LABELS = {
    "up": "Up",
    "down": "Down",
    "chop": "Chop",
    "range": "Range",
    DIRECTION_NO_VIEW: "No view",
}


def _previous_by_timeframe(previous: Any) -> dict[str, Mapping[str, Any]]:
    """`{timeframe: row}` from whatever the host handed the card.

    The host now supplies the latest read of EACH timeframe the card can ask
    about (`MainWindow._previous_mentor_read`). A single row is still accepted
    - the manual door and every existing caller pass one - and is filed under
    its own timeframe, never under the one the card happens to be showing.
    """
    if not isinstance(previous, Mapping) or not previous:
        return {}
    if previous.get("entry_id") or previous.get("event_type") or previous.get("text"):
        row = dict(previous)
        timeframe = str(row.get("timeframe") or TIMEFRAME_M5).strip().upper()
        return {timeframe: row} if timeframe in TIMEFRAME_FOR_HORIZON.values() else {}
    answer: dict[str, Mapping[str, Any]] = {}
    for key, row in previous.items():
        timeframe = str(key or "").strip().upper()
        if isinstance(row, Mapping) and timeframe in TIMEFRAME_FOR_HORIZON.values():
            answer[timeframe] = dict(row)
    return answer


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
        #: timeframe -> the latest read of THAT timeframe this session.
        self._previous: dict[str, Mapping[str, Any]] = {}
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

        # TJ-14A item 5. What the desk ALREADY knows, above the box the trader
        # types into, so they write only what it cannot see. Built from the
        # context the service already delivered: it starts no read of its own.
        self.internals_strip = QWidget(self)
        self.internals_strip.setObjectName("MentorInternalsStrip")
        strip_layout = QVBoxLayout(self.internals_strip)
        strip_layout.setContentsMargins(6, 4, 6, 4)
        strip_layout.setSpacing(1)
        self.internals_text = QLabel("", self.internals_strip)
        self.internals_text.setObjectName("MentorInternalsText")
        self.internals_text.setWordWrap(True)
        strip_layout.addWidget(self.internals_text)

        self.see_label = QLabel(SEE_HEADING)
        self.see_label.setObjectName("MutedLabel")
        self.see_label.setWordWrap(True)
        self.text_box = QPlainTextEdit(self)
        self.text_box.setPlaceholderText(
            "In your own words. Nothing here is parsed or scored - it is stored "
            "exactly as you type it."
        )
        self.text_box.setMaximumHeight(84)
        self.text_box.installEventFilter(self)

        # TJ-14A items 1-2. The forced click, one row per horizon, each row
        # under the words it belongs to. `_prediction_rows` holds the widgets;
        # `_predictions` holds what has been clicked and is the ONE thing the
        # file verbs are gated on.
        self.expect_label = QLabel(EXPECT_HEADING)
        self.expect_label.setObjectName("MutedLabel")
        self.expect_label.setWordWrap(True)
        self._prediction_rows: dict[str, dict[str, Any]] = {}
        self._predictions: dict[str, dict[str, str]] = {}
        for horizon in HORIZONS:
            self._prediction_rows[horizon] = self._build_prediction_row(horizon)
            self._predictions[horizon] = {"direction": "", "confidence": ""}

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
        #: trade_id -> the heading that names its symbol, side and SESSION.
        self._trade_headings: dict[str, QLabel] = {}
        #: trade_id -> the ONE container widget holding that trade's block.
        #: A block exists so a later slot can ADD a trade beside the rows the
        #: trader has already touched, and drop one that is no longer owed,
        #: without rebuilding a single widget that already has a value in it.
        self._trade_blocks: dict[str, QWidget] = {}
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
        # TJ-14B item 3. The few extra questions the desk is missing an answer
        # to, at most three, each a CLICK. Kept apart from the market read above
        # and from TJ-9's trade section: three different questions merged into
        # one box would be one box that answers none of them.
        self.questions_label = QLabel("")
        self.questions_label.setObjectName("MutedLabel")
        self.questions_label.setWordWrap(True)
        self.questions_label.setVisible(False)
        self.questions_box = QWidget(self)
        self._questions_layout = QVBoxLayout(self.questions_box)
        self._questions_layout.setContentsMargins(0, 0, 0, 0)
        self._questions_layout.setSpacing(3)
        self.questions_box.setVisible(False)
        #: (kind, subject_id) -> (Subject, the combo holding its answer)
        self._question_inputs: dict[tuple[str, str], tuple[Any, QComboBox]] = {}
        # The row widget behind each question, so `set_questions` can MERGE:
        # a subject already on the card keeps its widgets and their values.
        self._question_rows: dict[tuple[str, str], QWidget] = {}
        # TJ-7's strip, when this card is asking `day_close`. `None` otherwise.
        self._mood_strip: Any = None
        self._question_store = None
        self._question_service = None
        self.save_questions_button = QPushButton("Save these answers")
        self.save_questions_button.setToolTip(
            "Files each answer through the store that owns it. Nothing is "
            "written for a question you leave on '-'."
        )
        self.save_questions_button.clicked.connect(self.save_questions)
        self.save_questions_button.setVisible(False)

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
        layout.addWidget(self.internals_strip)
        layout.addWidget(self.see_label)
        layout.addWidget(self.text_box)
        layout.addWidget(self.expect_label)
        layout.addWidget(self._prediction_rows[HORIZON_REST_OF_DAY]["widget"])
        layout.addWidget(self.d1_label)
        layout.addWidget(self.d1_box)
        layout.addWidget(self._prediction_rows[HORIZON_NEXT_5_SESSIONS]["widget"])
        layout.addWidget(self.trade_check_label)
        layout.addWidget(self.trade_check_box)
        layout.addWidget(self.save_answers_button)
        layout.addWidget(self.questions_label)
        layout.addWidget(self.questions_box)
        layout.addWidget(self.save_questions_button)
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

    # -- what I expect ----------------------------------------------------
    def _build_prediction_row(self, horizon: str) -> dict[str, Any]:
        """One horizon's forced click, with its clock printed on the row.

        A DESCRIPTION is not a PREDICTION (decision 0021 answer 29). The words
        above this row say what the tape is doing; this row is the only thing
        TJ-10 grades, which is why it is a closed set of buttons rather than a
        sentence somebody has to parse afterwards.
        """
        container = QWidget(self)
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        directions = QHBoxLayout()
        directions.setContentsMargins(0, 0, 0, 0)
        directions.setSpacing(4)
        heading = QLabel(HORIZON_LABELS[horizon], container)
        heading.setObjectName("MutedLabel")
        directions.addWidget(heading)
        direction_buttons: dict[str, QPushButton] = {}
        for name in DIRECTIONS[horizon]:
            button = QPushButton(_DIRECTION_LABELS[name], container)
            button.setObjectName("MentorPredictionButton")
            button.setCheckable(True)
            button.setToolTip(
                "No view is a complete answer and is never graded."
                if name == DIRECTION_NO_VIEW
                else f"{HORIZON_LABELS[horizon][:-1]} {_DIRECTION_LABELS[name].lower()}."
            )
            button.clicked.connect(
                lambda _checked=False, h=horizon, d=name: self._choose_direction(h, d)
            )
            directions.addWidget(button)
            direction_buttons[name] = button
        directions.addStretch(1)
        column.addLayout(directions)

        confidence_widget = QWidget(container)
        confidence_row = QHBoxLayout(confidence_widget)
        confidence_row.setContentsMargins(0, 0, 0, 0)
        confidence_row.setSpacing(4)
        sure_label = QLabel("How sure:", confidence_widget)
        sure_label.setObjectName("MutedLabel")
        confidence_row.addWidget(sure_label)
        confidence_buttons: dict[str, QPushButton] = {}
        for level in CONFIDENCE_LEVELS:
            button = QPushButton(level.capitalize(), confidence_widget)
            button.setObjectName("MentorConfidenceButton")
            button.setCheckable(True)
            button.clicked.connect(
                lambda _checked=False, h=horizon, name=level: self._choose_confidence(h, name)
            )
            confidence_row.addWidget(button)
            confidence_buttons[level] = button
        confidence_row.addStretch(1)
        # Nothing to be sure OF until a direction is clicked, and nothing at all
        # on a No view - asking how sure somebody is of nothing is the kind of
        # question that teaches a trader to click past the card.
        confidence_widget.setVisible(False)
        column.addWidget(confidence_widget)

        because = QLineEdit(container)
        because.setObjectName("MentorBecauseBox")
        because.setPlaceholderText("Because… (optional, one line)")
        column.addWidget(because)
        return {
            "widget": container,
            "directions": direction_buttons,
            "confidence_widget": confidence_widget,
            "confidence": confidence_buttons,
            "because": because,
        }

    def prediction_button(self, horizon: str, direction: str):
        """The direction button for one horizon, or ``None`` if it has none."""
        row = self._prediction_rows.get(str(horizon))
        return row["directions"].get(str(direction)) if row else None

    def confidence_button(self, horizon: str, level: str):
        """The `How sure` button for one horizon, or ``None``."""
        row = self._prediction_rows.get(str(horizon))
        return row["confidence"].get(str(level)) if row else None

    def because_box(self, horizon: str):
        """The optional one-line `Because…` for one horizon, or ``None``."""
        row = self._prediction_rows.get(str(horizon))
        return row["because"] if row else None

    def _choose_direction(self, horizon: str, direction: str) -> None:
        state = self._predictions.setdefault(horizon, {"direction": "", "confidence": ""})
        state["direction"] = direction
        row = self._prediction_rows[horizon]
        for name, button in row["directions"].items():
            button.setChecked(name == direction)
        if direction == DIRECTION_NO_VIEW:
            state["confidence"] = ""
            for button in row["confidence"].values():
                button.setChecked(False)
        row["confidence_widget"].setVisible(direction not in ("", DIRECTION_NO_VIEW))
        self._refresh_prediction_gate()

    def _choose_confidence(self, horizon: str, level: str) -> None:
        state = self._predictions.setdefault(horizon, {"direction": "", "confidence": ""})
        if state.get("direction") == DIRECTION_NO_VIEW:
            return
        state["confidence"] = level
        for name, button in self._prediction_rows[horizon]["confidence"].items():
            button.setChecked(name == level)
        self._refresh_prediction_gate()

    def _horizons_on_this_card(self) -> tuple[str, ...]:
        return tuple(
            horizon
            for horizon in HORIZONS
            if self._prediction_rows[horizon]["widget"].isVisibleTo(self)
        )

    def _prediction_complete(self, horizon: str) -> bool:
        """Forced means direction AND how sure - unless there is no view.

        Decision 0021 answer 29: a prediction IS direction, horizon and
        confidence; only `because` is optional. TJ-16's calibration is read by
        confidence, so a call filed without one could never join it.
        """
        state = self._predictions.get(horizon) or {}
        direction = str(state.get("direction") or "")
        if not direction:
            return False
        if direction == DIRECTION_NO_VIEW:
            return True
        return bool(state.get("confidence"))

    def _prediction_for(self, horizon: str) -> dict[str, Any]:
        state = self._predictions.get(horizon) or {}
        return build_prediction(
            direction=str(state.get("direction") or ""),
            horizon=horizon,
            confidence=str(state.get("confidence") or ""),
            because=self._prediction_rows[horizon]["because"].text(),
        )

    def _previous_for(self, horizon: str) -> Mapping[str, Any] | None:
        """The last read of THIS horizon's timeframe, or ``None``.

        There is no fallback. A silent one is what filed a D1-timeframe row
        carrying a rest-of-day call: `Read unchanged` on the 09:00 card was
        handed the 08:00 card's D1 row as "your last read", kept its timeframe,
        and took the only horizon the 09:00 card shows.
        """
        row = self._previous.get(TIMEFRAME_FOR_HORIZON[horizon])
        return row if isinstance(row, Mapping) and str(row.get("text") or "").strip() else None

    def _unchanged_refusal(self) -> str:
        """Why `Read unchanged` is unavailable, in the trader's own terms."""
        horizons = self._horizons_on_this_card()
        if not horizons:
            return "nothing is being asked"
        missing = [
            TIMEFRAME_FOR_HORIZON[horizon]
            for horizon in horizons
            if self._previous_for(horizon) is None
        ]
        if missing:
            # Never a partial file and never a substitute: the trader has
            # nothing to reaffirm for that timeframe, so they write it.
            return (
                "there is no earlier "
                + " or ".join(missing)
                + " read this session to reaffirm - Submit files this one"
            )
        return self._missing_prediction_reason(horizons)

    def _refresh_prediction_gate(self) -> None:
        """The file verbs stay grey until this hour's call has been clicked."""
        try:
            horizons = self._horizons_on_this_card()
            self.submit_button.setEnabled(
                bool(horizons) and all(self._prediction_complete(h) for h in horizons)
            )
            refusal = self._unchanged_refusal()
            self.unchanged_button.setEnabled(not refusal)
            self.unchanged_button.setToolTip(
                refusal
                or "Files a NEW observation at this time that restates your "
                "previous read, with THIS hour's call. The earlier one stays "
                "exactly as you wrote it."
            )
        except RuntimeError:  # pragma: no cover - widget already torn down
            pass

    def _missing_prediction_reason(self, horizons: tuple[str, ...]) -> str:
        """Why a file verb refused. The gate lives HERE as well as on the button.

        Forced means no code path files an hourly answer without its clicks -
        a keyboard shortcut, a host call and a future caller all reach this.
        """
        for horizon in horizons:
            state = self._predictions.get(horizon) or {}
            if not str(state.get("direction") or ""):
                return (
                    f"{HORIZON_LABELS[horizon]} still needs a call. "
                    "No view is a complete answer."
                )
            if not self._prediction_complete(horizon):
                return f"{HORIZON_LABELS[horizon]} still needs How sure."
        return ""

    def _reset_predictions(self) -> None:
        for horizon, row in self._prediction_rows.items():
            self._predictions[horizon] = {"direction": "", "confidence": ""}
            for button in row["directions"].values():
                button.setChecked(False)
            for button in row["confidence"].values():
                button.setChecked(False)
            row["confidence_widget"].setVisible(False)
            row["because"].setText("")

    def _restore_predictions(self, saved: Mapping[str, Any]) -> None:
        """Put unsaved clicks back, exactly as unsaved text is put back."""
        for horizon in HORIZONS:
            record = saved.get(horizon) if isinstance(saved, Mapping) else None
            if not isinstance(record, Mapping):
                continue
            direction = str(record.get("direction") or "")
            if direction in DIRECTIONS.get(horizon, ()):
                self._choose_direction(horizon, direction)
            level = str(record.get("confidence") or "")
            if level in CONFIDENCE_LEVELS:
                self._choose_confidence(horizon, level)
            self._prediction_rows[horizon]["because"].setText(str(record.get("because") or ""))

    def _clicked_predictions(self) -> dict[str, dict[str, str]]:
        """What is on the card right now, for the draft file."""
        answered: dict[str, dict[str, str]] = {}
        for horizon in HORIZONS:
            state = self._predictions.get(horizon) or {}
            because = self._prediction_rows[horizon]["because"].text()
            if str(state.get("direction") or "") or because.strip():
                answered[horizon] = {
                    "direction": str(state.get("direction") or ""),
                    "confidence": str(state.get("confidence") or ""),
                    "because": because,
                }
        return answered

    def _render_internals(self) -> None:
        """Draw the strip from the snapshot the card ALREADY holds.

        No fetch, no loader, no second request: the wording is computed by a
        pure function and this is one `setText`.
        """
        try:
            from trade_mentor_context import internals_lines

            self.internals_text.setText("\n".join(internals_lines(self._current_context)))
        except Exception:  # noqa: BLE001 - the strip never costs the prompt
            logging.debug("Trade Mentor internals strip not drawn.", exc_info=True)

    # -- drafts -----------------------------------------------------------
    def _load_drafts(self) -> None:
        try:
            payload = json.loads(self._drafts_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(payload, dict):
            for slot_id, record in payload.items():
                if isinstance(record, Mapping):
                    self._drafts[str(slot_id)] = {
                        "text": str(record.get("text") or ""),
                        # TJ-14A: an unsaved CLICK is kept exactly as unsaved
                        # text is, and across a restart - that is what the
                        # drafts file is for.
                        "predictions": dict(record.get("predictions") or {}),
                    }

    def _save_drafts(self) -> None:
        """Never costs the card. A draft that could not be written is a lost
        half-thought; a card that refused to move on would be a lost hour."""
        try:
            self._drafts_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                slot_id: {
                    "text": str(record.get("text") or ""),
                    "predictions": dict(record.get("predictions") or {}),
                }
                for slot_id, record in self._drafts.items()
                if str(record.get("text") or "") or record.get("predictions")
            }
            tmp = self._drafts_path.with_name(self._drafts_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._drafts_path)
        except OSError:
            logging.debug("Trade Mentor draft not saved.", exc_info=True)

    def draft_for(self, slot_id: str) -> str:
        """Whatever was typed against `slot_id` and never submitted."""
        return str((self._drafts.get(str(slot_id)) or {}).get("text") or "")

    def draft_predictions_for(self, slot_id: str) -> dict[str, Any]:
        """Whatever was CLICKED against `slot_id` and never submitted."""
        return dict((self._drafts.get(str(slot_id)) or {}).get("predictions") or {})

    def _stash_draft(self) -> None:
        """Keep what is in the boxes, exactly as typed - trailing space and all."""
        if self._slot is None:
            return
        text = self.text_box.toPlainText()
        d1_text = self.d1_box.toPlainText()
        combined = text if not d1_text else f"{text}\n{d1_text}"
        clicks = self._clicked_predictions()
        if combined.strip() or clicks:
            self._drafts[self._slot.slot_id] = {
                "text": combined,
                "predictions": clicks,
            }
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
        self._previous = _previous_by_timeframe(previous)
        kind = str(getattr(slot, "kind", "") or "")
        self.prompt_label.setText(_QUESTIONS.get(kind, _QUESTIONS[KIND_MANUAL]))
        restored = self.draft_for(slot.slot_id)
        self.text_box.setPlainText(restored)
        self.d1_box.setPlainText("")
        show_d1 = kind == KIND_M5_D1
        self.d1_label.setVisible(show_d1)
        self.d1_box.setVisible(show_d1)
        # Every card asks for the rest of the day; only a D1 card asks about the
        # next five sessions. A swing call offered six times a day would be the
        # same click about the same five sessions, over and over.
        self._prediction_rows[HORIZON_REST_OF_DAY]["widget"].setVisible(True)
        self._prediction_rows[HORIZON_NEXT_5_SESSIONS]["widget"].setVisible(show_d1)
        self._reset_predictions()
        self._restore_predictions(self.draft_predictions_for(slot.slot_id))
        self._render_internals()
        # TJ-14B: the questions belong to the CARD, not to the session. The host
        # rebuilds them for this slot right after this returns; an unanswered
        # one is not lost, it is CARRIED and asked again on the next card.
        self._clear_questions()
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
        # One line per timeframe this card asks about, each naming its own
        # earlier read: "your last read" on a card with two timeframes was one
        # sentence that could only be true of one of them.
        lines = []
        for horizon in self._horizons_on_this_card():
            row = self._previous_for(horizon)
            if row is not None:
                timeframe = TIMEFRAME_FOR_HORIZON[horizon]
                lines.append(f"Your last {timeframe} read: {row.get('text') or ''}")
        self.previous_label.setText("\n".join(lines))
        self.previous_label.setVisible(bool(lines))
        self._refresh_prediction_gate()
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
            self._render_internals()

    def _on_context_unavailable(self, request_id: str, context: object) -> None:
        if str(request_id) == self._context_slot_id and isinstance(context, Mapping):
            self._current_context = dict(context)
            self._render_internals()

    # -- TJ-14B: the few questions the desk is missing an answer to ---------
    def _clear_questions(self) -> None:
        self._question_inputs = {}
        self._question_rows = {}
        self._mood_strip = None
        self.questions_label.setVisible(False)
        self.questions_box.setVisible(False)
        self.save_questions_button.setVisible(False)
        while self._questions_layout.count():
            item = self._questions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _drop_question_row(self, key) -> None:
        """Take ONE question off the card and forget its widgets."""
        row = self._question_rows.pop(key, None)
        self._question_inputs.pop(key, None)
        if row is None:
            return
        if self._mood_strip is not None and self._mood_strip.parent() is row:
            self._mood_strip = None
        self._questions_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()

    def _build_question_row(self, subject) -> tuple[QWidget, QComboBox]:
        """One question's widgets: the prompt, the combo, and TJ-7's strip."""
        import mentor_questions

        row = QWidget(self.questions_box)
        stack = QVBoxLayout(row)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(2)

        line = QWidget(row)
        row_layout = QHBoxLayout(line)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        prompt = QLabel(str(getattr(subject, "prompt", "") or subject.kind), line)
        prompt.setWordWrap(True)
        combo = QComboBox(line)
        # "-" first, so a question the trader did not touch stays unasked
        # rather than being filed as whatever happened to be at index 0.
        combo.addItem("-", "")
        for option in tuple(getattr(subject, "options", ()) or ()):
            combo.addItem(str(option).replace("_", " "), str(option))
        if combo.findData(mentor_questions.STOP_ASKING) < 0:
            combo.addItem("stop asking this", mentor_questions.STOP_ASKING)
        row_layout.addWidget(prompt, 1)
        row_layout.addWidget(combo)
        stack.addWidget(line)

        if str(subject.kind) == DAY_CLOSE_KIND:
            # TJ-7 change 2 as AMENDED: on the Mentor the strip is NOT its own
            # widget - it is this question's hook, asked once on the session's
            # last card beside `Followed the plan`. It is OPTIONAL: it never
            # touches `save_answers_button`'s gate, which counts TJ-9's material
            # fields and nothing else.
            from ui.widgets.mood_strip import MoodStrip

            self._mood_strip = MoodStrip(row)
            stack.addWidget(self._mood_strip)
        return row, combo

    def set_questions(self, result, *, store=None, service=None) -> None:
        """Draw this card's questions - at most three, plus what is waiting.

        `result` is `mentor_questions.pending`'s answer. Only its `asked`
        subjects are drawn: `carried` is COUNTED in the waiting note and asked
        on a later card, never dropped and never a fourth here.

        Every question carries `Stop asking this`, which retires that ONE
        subject through `TradeMentorService.stop_asking` - the single writer -
        and writes no answer at all.

        It MERGES rather than rebuilds (TJ-14B's card rule, and TJ-7 is why it
        bites here): a subject already on the card keeps its SAME widgets and
        everything clicked into them. A rebuild would silently drop a face the
        trader had chosen and a combo they had set, and the trader would have
        no way to know it had happened.
        """
        self._question_store = store
        self._question_service = service
        asked = tuple(getattr(result, "asked", ()) or ()) if result is not None else ()
        note = str(getattr(result, "waiting_note", "") or "") if result is not None else ""
        if not asked:
            self._clear_questions()
            if note:
                self.questions_label.setText(note)
                self.questions_label.setVisible(True)
            return
        self.questions_label.setText(
            "A few things the desk cannot work out on its own"
            + (f". {note}" if note else ".")
        )
        self.questions_label.setVisible(True)

        wanted = {(str(subject.kind), str(subject.subject_id)): subject for subject in asked}
        for key in [key for key in self._question_inputs if key not in wanted]:
            self._drop_question_row(key)

        for index, subject in enumerate(asked):
            key = (str(subject.kind), str(subject.subject_id))
            existing = self._question_inputs.get(key)
            if existing is None:
                row, combo = self._build_question_row(subject)
                self._question_rows[key] = row
            else:
                row, combo = self._question_rows[key], existing[1]
            # A subject can change its prompt or its detail between cards; the
            # WIDGETS and what is in them do not.
            self._question_inputs[key] = (subject, combo)
            self._questions_layout.insertWidget(index, row)
            if str(subject.kind) == "ai_question":
                # The overnight question is now a CLICK with an answer and a
                # once-a-day rule. Before TJ-14B the same sentence was printed
                # as "One thing to test: ..." on EVERY card, forever, with no
                # way to answer it; printing both would ask it twice.
                self.coaching_label.setVisible(False)
        self.questions_box.setVisible(True)
        self.save_questions_button.setVisible(True)

    def question_box(self, kind: str, subject_id: str):
        """The combo one question is answered in, or ``None``."""
        entry = self._question_inputs.get((str(kind), str(subject_id)))
        return entry[1] if entry else None

    # -- TJ-7: the optional mood strip on the day_close question ------------
    def mood_button(self, score):
        """The face for `score` on this card's strip, or ``None``."""
        strip = self._mood_strip
        return strip.mood_button(score) if strip is not None else None

    def state_tag_button(self, code: str):
        """The chip for `code` on this card's strip, or ``None``."""
        strip = self._mood_strip
        return strip.state_tag_button(code) if strip is not None else None

    def mood_answer(self) -> dict[str, Any]:
        """What the trader clicked on the strip. Nothing is ever pre-filled."""
        strip = self._mood_strip
        return strip.answer() if strip is not None else {"mood": None, "state_tags": ()}

    def save_questions(self) -> dict[str, Any]:
        """File every question the trader answered, and nothing else.

        A retirement is not an answer and stores none: `Stop asking this` goes
        to the service, which is the single writer of what the trader silenced.
        A writer that refuses costs its own row and never the others.

        TJ-7: a mood clicked on the `day_close` strip with the plan question
        left alone is STILL FILED. A click the trader made is never thrown away
        because a different question on the same row went untouched.
        """
        import mentor_questions

        moment = self._now()
        saved = 0
        retired = 0
        failures: list[str] = []
        mood = self.mood_answer()
        touched_the_strip = mood["mood"] is not None or bool(mood["state_tags"])
        for (kind, subject_id), (subject, combo) in list(self._question_inputs.items()):
            chosen = str(combo.currentData() or "")
            carries_a_mood = kind == DAY_CLOSE_KIND and self._mood_strip is not None
            if not chosen and not (carries_a_mood and touched_the_strip):
                continue
            if chosen == mentor_questions.STOP_ASKING:
                service = self._question_service
                if service is None:
                    failures.append(f"{kind}: nothing to record the retirement with")
                    continue
                try:
                    service.stop_asking(kind, subject_id)
                    retired += 1
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{kind}: {exc}")
                continue
            answer: dict[str, Any] = {"state": chosen}
            if carries_a_mood:
                answer["mood"] = mood["mood"]
                answer["state_tags"] = tuple(mood["state_tags"])
            try:
                outcome = mentor_questions.record_answer(
                    subject,
                    answer,
                    store=self._question_store,
                    now=moment,
                )
            except Exception as exc:  # noqa: BLE001 - one refusal is not the card
                failures.append(f"{kind}: {exc}")
                continue
            if outcome.get("ok"):
                saved += 1
            else:
                failures.append(f"{kind}: {outcome.get('reason') or 'not stored'}")
        if not saved and not retired and not failures:
            self._set_status("Nothing was answered, so nothing was filed.")
            return {"ok": False, "reason": "no question was answered"}
        parts = []
        if saved:
            parts.append(f"{saved} answer(s) filed")
        if retired:
            parts.append(f"{retired} question(s) retired")
        if failures:
            parts.append(f"{len(failures)} NOT stored ({failures[0]})")
        self._set_status("; ".join(parts) + ".")
        if saved or retired:
            self._clear_questions()
        return {"ok": not failures, "saved": saved, "retired": retired, "failures": failures}

    def _clear_trade_check(self) -> None:
        self._answer_inputs = {}
        self._trade_questions = {}
        self._raw_trade_inputs = {}
        self._ai_draft_buttons = {}
        self._ai_drafts = {}
        self._setup_confirm_buttons = {}
        self._setup_choice_boxes = {}
        self._setup_confirmed = set()
        self._trade_blocks = {}
        self._trade_headings = {}
        self.save_answers_button.setEnabled(False)
        while self._trade_check_layout.count():
            item = self._trade_check_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _drop_trade_block(self, trade_id: str) -> None:
        """Take ONE trade's block off the card and forget its widgets.

        A trade leaves the card because it is no longer owed - its fields were
        answered and saved - and it leaves ALONE: every other block keeps the
        widgets the trader has been typing into.
        """
        key = str(trade_id)
        block = self._trade_blocks.pop(key, None)
        if block is not None:
            self._trade_check_layout.removeWidget(block)
            block.setParent(None)
            block.deleteLater()
        self._answer_inputs.pop(key, None)
        self._trade_questions.pop(key, None)
        self._raw_trade_inputs.pop(key, None)
        self._ai_draft_buttons.pop(key, None)
        self._ai_drafts.pop(key, None)
        self._setup_confirm_buttons.pop(key, None)
        self._setup_choice_boxes.pop(key, None)
        self._trade_headings.pop(key, None)
        self._setup_confirmed.discard(key)

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

        **Every delivered slot of the session hands this a FRESH task, and the
        section MERGES rather than choosing between keeping and rebuilding**
        (TJ-14B fix round). A row already on the card keeps its exact widgets
        and their values, a trade the fresh task names and the card does not
        is ADDED, a row that is no longer owed is dropped, and the heading is
        always rewritten from the fresh task. The host used to return early
        whenever the card held answer widgets, so a 09:00 card that had drawn
        today's own fills was never told the statement had landed: the reviewed
        session's trades were never asked about that day and the card went on
        printing `journal not ready` and a stale freshness date.
        """
        import trade_mentor_trade_check as check

        self._trade_store = store
        if task is None:
            self._clear_trade_check()
            self.trade_check_label.setVisible(False)
            self.trade_check_box.setVisible(False)
            self.save_answers_button.setVisible(False)
            self._trade_check_session = ""
            return
        session = str(getattr(self._slot, "session", "") or "")
        if not session or session != self._trade_check_session:
            # A different day's card - or a card with no session to be
            # identified by - has nothing to merge with. A question about
            # Friday's trades must never keep its widgets into Wednesday.
            self._clear_trade_check()
        self._trade_check_session = session
        same_session = tuple(getattr(task, "same_session_trade_ids", ()) or ())
        if not getattr(task, "journal_ready", False):
            opening = (
                f"Yesterday's trades ({task.reviewed_session}): "
                f"{task.reason or check.REASON_NOT_READY} - "
                f"{self._freshness_phrase(task)}."
            )
            if task.trades:
                # The heading may not contradict itself: a card that DRAWS
                # today's fills cannot also say nothing is asked yet.
                self.trade_check_label.setText(
                    opening
                    + " Yesterday's broker statement has not landed, so "
                    "yesterday's trades are not asked yet and come back on the "
                    f"next card. {len(same_session)} fill(s) seen TODAY are "
                    "asked below - today's statement never lands mid-session, "
                    "and a fill the desk has already seen is one you can still "
                    "label. The day pull is Questrade only - IBKR has no day leg."
                )
            else:
                # Nothing SEEN either. An empty questionnaire drawn from an
                # incomplete list is a lie about the session.
                self.trade_check_label.setText(
                    opening + " The broker statement has not landed, so nothing "
                    "is asked yet; this comes back on the next card. The day "
                    "pull is Questrade only - IBKR has no day leg."
                )
            self.trade_check_label.setVisible(True)
            self._merge_trade_questions(task)
            return
        if not task.trades:
            self.trade_check_label.setText(
                f"Yesterday's trades ({task.reviewed_session}): nothing is missing. "
                f"{self._freshness_phrase(task).capitalize()}."
            )
            self.trade_check_label.setVisible(True)
            self._merge_trade_questions(task)
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
            + (
                f" {len(same_session)} of them filled TODAY - labelling those "
                "now is the label made before the outcome is known."
                if same_session
                else ""
            )
        )
        self.trade_check_label.setVisible(True)
        self._merge_trade_questions(task)

    def _merge_trade_questions(self, task) -> None:
        """Make the card's rows the fresh task's rows, widget by widget.

        Three moves, in this order:

        1. a row whose trade the fresh task no longer names is DROPPED - it was
           answered and saved, and a question that is already answered must not
           come back;
        2. a row that is already here is LEFT EXACTLY AS IT IS. Not re-created,
           not re-read: the same combo objects with whatever the trader has
           chosen, and the same typed text. This is what the host's early
           return used to protect, and it is protected here instead;
        3. a trade the fresh task names and the card does not is ADDED after
           the rows already on it - the reviewed session's trades once the
           statement lands, or a fill seen later in the day.

        Then the Save gate is recomputed over ALL the rows now on the card, so
        adding a trade at 11:00 greys Save again until that trade is answered
        too.
        """
        owed = {str(question.trade_id): question for question in task.trades}
        for trade_id in [key for key in self._trade_blocks if key not in owed]:
            self._drop_trade_block(trade_id)
        for trade_id, question in owed.items():
            if trade_id in self._trade_blocks:
                continue
            self._add_trade_block(question, task)
        has_rows = bool(self._answer_inputs)
        self.trade_check_box.setVisible(has_rows)
        self.save_answers_button.setVisible(has_rows)
        self._refresh_save_gate()

    def _add_trade_block(self, question, task) -> None:
        """One block of widgets for ONE trade the card is asking about.

        Shared by the ready branch and the not-ready one: a statement that has
        not landed says nothing about the fills the desk has ALREADY SEEN today
        (TJ-14B), and a card that refused to draw them would make `same_session`
        unreachable on exactly the mornings it matters most.

        Every widget lives inside one container, so a later slot can drop this
        trade - or add another beside it - without touching anything the trader
        has already answered.
        """
        import trade_mentor_trade_check as check

        trade_id = str(question.trade_id)
        self._trade_questions[trade_id] = question
        block = QWidget(self.trade_check_box)
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(self._trade_check_layout.spacing())
        heading = QLabel(self._trade_heading(question, task))
        heading.setObjectName("MutedLabel")
        block_layout.addWidget(heading)
        self._trade_headings[trade_id] = heading
        self._add_setup_confirm(question, block, block_layout)
        raw_box = QPlainTextEdit(block)
        raw_box.setMaximumHeight(72)
        raw_box.setPlaceholderText(
            "Tell me in one note: why, stop/invalidation, target, and setup. "
            "Your exact words are saved before local AI fills the draft."
        )
        block_layout.addWidget(raw_box)
        ai_button = QPushButton("Fill missing fields with local AI", block)
        ai_button.clicked.connect(
            lambda _checked=False, trade_id=trade_id: self._start_ai_draft(trade_id)
        )
        block_layout.addWidget(ai_button)
        self._raw_trade_inputs[trade_id] = raw_box
        self._ai_draft_buttons[trade_id] = ai_button
        fields: dict[str, tuple[QComboBox, QLineEdit]] = {}
        for name in question.missing:
            row = QWidget(block)
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
            block_layout.addWidget(row)
            fields[name] = (combo, text_input)
        self._answer_inputs[trade_id] = fields
        self._trade_check_layout.addWidget(block)
        self._trade_blocks[trade_id] = block

    def _trade_heading(self, question, task) -> str:
        """`AAPL LONG - today (2026-09-14)` / `MSFT SHORT - 2026-09-11`.

        Which session a trade is from is the whole point of asking on the day:
        an answer given about today's fill is `same_session` and one about
        yesterday's is `recalled_after`, and the trader cannot tell the two
        apart from a row that only says the symbol.
        """
        name = f"{question.symbol} {question.direction}".strip() or str(question.trade_id)
        today_ids = {str(key) for key in (getattr(task, "same_session_trade_ids", ()) or ())}
        if str(question.trade_id) in today_ids:
            day = self._trade_check_session or str(getattr(question, "trade_date", "") or "")
            return f"{name} - today ({day})" if day else f"{name} - today"
        session = str(getattr(task, "reviewed_session", "") or "") or str(
            getattr(question, "trade_date", "") or ""
        )
        return f"{name} - {session}" if session else name

    @staticmethod
    def _freshness_phrase(task) -> str:
        """"fills current to <date>" - the one line every surface prints.

        The DATE comes from the task, never from the widget: the Journal and
        the AWAY digest print the same sentence from the same number.
        """
        current = str(getattr(task, "fills_current_to", "") or "")
        return f"fills current to {current}" if current else "no verified import yet"

    def _add_setup_confirm(self, question, parent=None, layout=None) -> None:
        """One click for the setup, when the machine has something to suggest.

        The button is a SUGGESTION until it is pressed. Showing it writes
        nothing - the row stays exactly as the bulk tagger left it - and
        pressing it is the trader's write through the Journal's own writer. A
        trade whose setup the trader already confirmed is never offered one.

        `parent` / `layout` are the trade's own block, so the row leaves the
        card with that trade and with nothing else.
        """
        import trade_mentor_trade_check as check

        guess = str(getattr(question, "setup_guess", "") or "")
        if not guess or "setup" not in tuple(question.missing or ()):
            return
        lane = str(getattr(question, "setup_guess_lane", "") or "")
        parent = parent if parent is not None else self.trade_check_box
        layout = layout if layout is not None else self._trade_check_layout
        row = QWidget(parent)
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
        layout.addWidget(row)
        self._setup_confirm_buttons[str(question.trade_id)] = button
        self._setup_choice_boxes[str(question.trade_id)] = choice

    def trade_heading_text(self, trade_id: str) -> str:
        """What ONE trade's block says it is - symbol, side and its session."""
        heading = self._trade_headings.get(str(trade_id))
        return heading.text() if heading is not None else ""

    def setup_confirm_button(self, trade_id: str):
        """The confirm button offered for one trade, or ``None``."""
        return self._setup_confirm_buttons.get(str(trade_id))

    def setup_choice_box(self, trade_id: str):
        """The vocabulary list that button sits beside, or ``None``."""
        return self._setup_choice_boxes.get(str(trade_id))

    def trade_check_session(self) -> str:
        """Which session's trade check is on the card in ANY form, or ``""``.

        Includes the one-line states - `journal not ready`, `nothing is
        missing`, `N field(s) filed` - which have no widgets to lose.
        """
        return self._trade_check_session if self._has_trade_check() else ""

    def open_answers_session(self) -> str:
        """Which session's trade check has ANSWER WIDGETS on the card, or ``""``.

        Deliberately narrower than :meth:`trade_check_session`: it answers only
        for a section the trader could already have TOUCHED. It is no longer
        the host's rebuild guard - the card MERGES a fresh task now, so the
        widgets are protected row by row instead of by refusing the whole
        update, which used to freeze a `journal not ready` line (and its stale
        date) on the card for the rest of the day.
        """
        return self._trade_check_session if self._answer_inputs else ""

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

    def _mentor_payload(self, slot: MentorSlot, moment: datetime, *, observation: str = "", horizon: str = "") -> dict[str, Any]:
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
        # TJ-14A item 2. Two keys that never share a field: the WORDS about now,
        # and the CLICK about what happens next. The entry's `text` stays the
        # observation, so every existing reader of this store keeps working.
        payload["observation"] = str(observation or "")
        if horizon:
            payload["prediction"] = self._prediction_for(horizon)
        return payload

    def submit(self) -> dict[str, Any]:
        """File the raw text and this hour's call. Once per slot, whatever the
        button does.

        The forced gate lives HERE as well as on the button (TJ-14A): forced
        means no code path files an hourly answer without its clicks, and
        `Ctrl+Enter` reaches this verb without touching a button.
        """
        slot = self._slot
        if slot is None:
            return {"ok": False, "reason": "nothing is being asked"}
        if slot.slot_id in self._submitted:
            # A double click is one read. The guard is here rather than on the
            # button because Ctrl+Enter reaches the same verb.
            return {"ok": False, "reason": "this read is already filed"}
        horizons = self._horizons_on_this_card()
        missing = self._missing_prediction_reason(horizons)
        if missing:
            self._set_status(missing)
            return {"ok": False, "reason": missing}
        moment = self._now()
        text = self.text_box.toPlainText().strip()
        d1_text = self.d1_box.toPlainText().strip() if self.d1_box.isVisibleTo(self) else ""

        session = self._session_for(slot, moment)
        written: list[dict[str, Any]] = []
        # The M5 read and the D1 read are stored SEPARATELY even though one card
        # collected both (the trader's brief). Two timeframes in one row would
        # be one row that is true of neither. Since TJ-14A a D1 card writes BOTH
        # rows once both calls are clicked - words or no words - because the
        # call, not the sentence, is what is graded.
        bodies = {HORIZON_REST_OF_DAY: text, HORIZON_NEXT_5_SESSIONS: d1_text}
        for horizon in horizons:
            body = bodies.get(horizon, "")
            result = self._service().write_entry(
                text=body,
                session_date=session,
                timeframe=TIMEFRAME_FOR_HORIZON[horizon],
                origin="trade_mentor",
                now=moment,
                mentor=self._mentor_payload(
                    slot, moment, observation=body, horizon=horizon
                ),
            )
            if not result.get("ok"):
                self._set_status(str(result.get("reason") or "entry NOT saved"))
                return result
            written.append(result.get("entry") or {})

        self._submitted.add(slot.slot_id)
        self._drop_draft(slot.slot_id)
        self.text_box.setPlainText("")
        self.d1_box.setPlainText("")
        self._reset_predictions()
        self._refresh_prediction_gate()
        self._set_status(f"Filed at {moment.strftime('%H:%M')}.")
        self.answered.emit(slot.slot_id)
        self.setVisible(False)
        return {"ok": True, "entries": written}

    def read_unchanged(self) -> dict[str, Any]:
        """File a NEW row per timeframe, restating THAT timeframe's last read.

        "My view has not changed" is a statement about the WORDS, and it is a
        statement about one timeframe: the M5 words are reaffirmed with this
        hour's rest-of-day call as an M5 row, the D1 words with this card's
        five-session call as a D1 row. A row's timeframe and its prediction's
        horizon always agree, and there is no fallback when a timeframe has no
        earlier read - the trader writes that one instead.
        """
        slot = self._slot
        if slot is None:
            return {"ok": False, "reason": "nothing is being asked"}
        if slot.slot_id in self._submitted:
            return {"ok": False, "reason": "this read is already filed"}
        refusal = self._unchanged_refusal()
        if refusal:
            self._set_status(refusal)
            return {"ok": False, "reason": refusal}
        moment = self._now()
        session = self._session_for(slot, moment)
        written: list[dict[str, Any]] = []
        for horizon in self._horizons_on_this_card():
            previous = self._previous_for(horizon)
            body = str(previous.get("text") or "").strip()
            result = self._service().write_entry(
                text=body,
                session_date=session,
                timeframe=TIMEFRAME_FOR_HORIZON[horizon],
                origin="trade_mentor",
                now=moment,
                mentor=self._mentor_payload(
                    slot, moment, observation=body, horizon=horizon
                ),
                # Names the read it restates, and deliberately NOT `supersedes`:
                # superseding would hide the 09:00 read behind the 11:00 one.
                reaffirms=str(previous.get("entry_id") or ""),
            )
            if not result.get("ok"):
                self._set_status(str(result.get("reason") or "entry NOT saved"))
                return result
            written.append(result.get("entry") or {})
        self._submitted.add(slot.slot_id)
        self._drop_draft(slot.slot_id)
        self._reset_predictions()
        self._refresh_prediction_gate()
        self._set_status(f"Read unchanged, filed at {moment.strftime('%H:%M')}.")
        self.answered.emit(slot.slot_id)
        self.setVisible(False)
        return {"ok": True, "entries": written}

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
