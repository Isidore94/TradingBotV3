"""The capture rail: four decisions, each in about two keystrokes.

Design constraint from the trader, and the reason this widget looks the way it
does: **every capture action is under five seconds and reachable without the
mouse.** A rail that costs ten seconds gets used twice and then abandoned, and
an abandoned rail produces no dataset. So the veto path is Alt+V, digit -
armed, chosen, written - and the note field only becomes mandatory for the one
reason whose value is entirely in the note.

What it writes:

* VETO      -> ui.annotations.store, plus a veto cohort row so forward returns
               accrue against the reason (ui.annotations.veto_cohort).
* LIKE      -> ui.annotations.store, one row carrying the claimed setup id.
* HYPO STOP -> ui.annotations.store. A price the trader would have used. No
               order is placed, ever; nothing downstream reads it.
* NOTE      -> ui.annotations.store.

What it never does: mute, suppress, score, gate, rank, or alert (plan.md
sec 5) - and it never writes a Focus list or watchlist either. An earlier
draft routed likes through FocusService.add, which put the symbol into a
swing watchlist and gave it Focus alert privileges; that crossed the
workspace plan's own boundary (a capture surface must stay analysis-only,
and adding a name to a list stays an explicit action on the surfaces that
own those lists), so it was removed. If likes ever need forward-return
grading, they get it the way vetoes do: a capture-side cohort file, not a
live list. The rail is a recorder.

Failures are shown, not swallowed. A capture that did not reach disk turns the
status line red and says so - the alternative is a trader who believes the
dataset has a decision in it that it does not.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.annotations.setup_claims import setup_claim_groups
from ui.annotations.store import (
    EVENT_HYPO_STOP,
    EVENT_LIKE_CLAIM,
    EVENT_NOTE,
    EVENT_VETO,
    AnnotationError,
    record_annotation,
)
from ui.annotations.vocabulary import VocabularyError, load_veto_vocabulary

_REASON_ROLE = Qt.ItemDataRole.UserRole


class CaptureRail(QFrame):
    """Veto / like+claim / hypothetical stop / note for the focused symbol."""

    #: (event_type, row) after a row reaches disk. Analysis-only consumers.
    captured = Signal(str, dict)
    #: (row) after a veto whose D1 chart the trader rejected but whose NAME
    #: they still want to day-trade. A REQUEST, not a write: the rail has
    #: never placed a name on a list and still does not (see the module
    #: docstring - an earlier draft routed likes through FocusService.add and
    #: had to be torn back out). The host that owns the Focus store performs
    #: the placement, so that store keeps exactly one writer.
    vetoDayTradeRequested = Signal(dict)

    def __init__(
        self,
        *,
        annotations_path: Any = None,
        veto_cohort_merge: Callable[..., dict] | None = None,
        bind_action_shortcuts: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("CaptureRail")
        self._annotations_path = annotations_path
        # A host that puts the rail on a hidden tab page has to own the four
        # Alt+ keys itself: a QShortcut bound inside a page the trader is not
        # looking at never fires, and binding one HERE as well as there makes
        # the sequence ambiguous in Qt, which fires neither. So such a host
        # passes False and rebinds `action_shortcuts()` at its own scope.
        self._bind_action_shortcuts = bool(bind_action_shortcuts)
        self._symbol = ""
        self._side = "LONG"
        self._last_price: float | None = None
        self._timeframe = "D1"
        self._ref_level_id = ""
        self._ref_level_family = ""
        # True only for the duration of a "veto but day-trade it" commit.
        # `captured` fires synchronously from inside commit_veto(), i.e.
        # BEFORE commit_veto_day_trade() can emit its own request, so a host
        # that retires the chart on any veto would retire this one too - and
        # the object the Focus placement needs would already be gone. This is
        # how that host is told to hold the chart for one commit.
        self._veto_keeps_chart = False

        if veto_cohort_merge is not None:
            self._merge_veto_cohort = veto_cohort_merge
        else:
            from ui.annotations.veto_cohort import merge_veto_cohort_picks

            self._merge_veto_cohort = merge_veto_cohort_picks

        try:
            self._vocabulary = load_veto_vocabulary()
            self._vocabulary_error = ""
        except VocabularyError as exc:
            # Fail visible: the veto action is disabled and says why, rather
            # than writing reason codes no later analysis would recognise.
            self._vocabulary = None
            self._vocabulary_error = str(exc)

        self._build()
        self._bind_shortcuts()
        self.set_context(symbol="", side="LONG")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*(theme.px(10),) * 4)
        layout.setSpacing(theme.px(8))

        self.symbol_label = QLabel("-")
        self.symbol_label.setObjectName("SectionTitle")
        layout.addWidget(self.symbol_label)

        side_row = QHBoxLayout()
        side_row.setSpacing(theme.px(4))
        side_row.addWidget(QLabel("Side"))
        self.side_input = QComboBox()
        self.side_input.addItems(["LONG", "SHORT"])
        self.side_input.currentTextChanged.connect(self._on_side_changed)
        side_row.addWidget(self.side_input, 1)
        layout.addLayout(side_row)

        layout.addWidget(self._veto_section())
        layout.addWidget(self._like_section())
        layout.addWidget(self._stop_section())
        layout.addWidget(self._note_section())
        layout.addStretch(1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("CaptureStatus")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _section(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("CaptureSection")
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(*(theme.px(8),) * 4)
        inner.setSpacing(theme.px(5))
        heading = QLabel(title)
        heading.setObjectName("SectionTitle")
        inner.addWidget(heading)
        return frame, inner

    def _veto_section(self) -> QFrame:
        frame, inner = self._section("Veto  (Alt+V, then 1-9)")
        self.reason_list = QListWidget()
        self.reason_list.setObjectName("VetoReasonList")
        self.reason_list.setAlternatingRowColors(False)
        if self._vocabulary is not None:
            for reason in self._vocabulary.reasons:
                item = QListWidgetItem(f"{reason.hotkey}  {reason.label}")
                item.setData(_REASON_ROLE, reason.code)
                if reason.hint:
                    item.setToolTip(reason.hint)
                self.reason_list.addItem(item)
            self.reason_list.itemActivated.connect(lambda _item: self.commit_veto())
            self.reason_list.currentItemChanged.connect(lambda *_: self._sync_note_requirement())
        else:
            self.reason_list.setEnabled(False)
            self.reason_list.addItem("vocabulary unavailable")
        self.reason_list.setMaximumHeight(theme.px(190))
        inner.addWidget(self.reason_list)

        self.veto_note_input = QLineEdit()
        self.veto_note_input.setPlaceholderText("note (optional)")
        self.veto_note_input.returnPressed.connect(self.commit_veto)
        inner.addWidget(self.veto_note_input)

        self.veto_button = QPushButton("Veto - not for today")
        self.veto_button.clicked.connect(self.commit_veto)
        # Trader, 2026-08-20: "it may be a shit D1 chart but its a good
        # daytrade." The veto is about the DAILY chart in front of them; the
        # name can still be worth a 5-minute trade. Without this the trader
        # has to choose between recording the honest D1 judgement and keeping
        # the day trade, and the dataset loses whichever one they drop.
        self.veto_day_trade_button = QPushButton("Veto D1 - but M5 today")
        self.veto_day_trade_button.setToolTip(
            "Record exactly the same D1 veto, then put the name on M5 Focus "
            "as a day trade. The veto is written by this rail; the Focus "
            "entry is made by the panel that owns that list, and it is yours "
            "- nothing auto-removes it."
        )
        self.veto_day_trade_button.clicked.connect(self.commit_veto_day_trade)
        if self._vocabulary is None:
            self.veto_button.setEnabled(False)
            self.veto_button.setToolTip(self._vocabulary_error)
            warning = QLabel(f"Veto disabled: {self._vocabulary_error}")
            warning.setWordWrap(True)
            inner.addWidget(warning)
        inner.addWidget(self.veto_button)
        inner.addWidget(self.veto_day_trade_button)
        return frame

    def _like_section(self) -> QFrame:
        frame, inner = self._section("Like + claim  (Alt+K)")
        self.setup_input = QComboBox()
        for group_name, claims in setup_claim_groups():
            for claim in claims:
                self.setup_input.addItem(f"{claim.label}   [{group_name}]", claim.setup_id)
                if claim.summary:
                    index = self.setup_input.count() - 1
                    self.setup_input.setItemData(index, claim.summary, Qt.ItemDataRole.ToolTipRole)
        inner.addWidget(self.setup_input)
        self.like_note_input = QLineEdit()
        self.like_note_input.setPlaceholderText("note (optional)")
        self.like_note_input.returnPressed.connect(self.commit_like)
        inner.addWidget(self.like_note_input)
        self.like_button = QPushButton("Like + claim setup")
        self.like_button.clicked.connect(self.commit_like)
        inner.addWidget(self.like_button)
        return frame

    def _stop_section(self) -> QFrame:
        frame, inner = self._section("Hypothetical stop  (Alt+S)")
        self.stop_input = QDoubleSpinBox()
        self.stop_input.setDecimals(4)
        self.stop_input.setRange(0.0, 1_000_000.0)
        self.stop_input.setSingleStep(0.05)
        self.stop_input.setSpecialValueText("")  # 0 reads as "unset"
        inner.addWidget(self.stop_input)
        self.stop_button = QPushButton("Record stop (no order)")
        self.stop_button.clicked.connect(self.commit_hypo_stop)
        inner.addWidget(self.stop_button)
        return frame

    def _note_section(self) -> QFrame:
        frame, inner = self._section("Note  (Alt+N)")
        self.note_input = QLineEdit()
        self.note_input.setPlaceholderText("freeform note")
        self.note_input.returnPressed.connect(self.commit_note)
        inner.addWidget(self.note_input)
        self.note_button = QPushButton("Save note")
        self.note_button.clicked.connect(self.commit_note)
        inner.addWidget(self.note_button)
        return frame

    def action_shortcuts(self) -> tuple[tuple[str, Callable[[], None]], ...]:
        """The rail's four key bindings, as (sequence, handler) pairs.

        Public so a host that took the rail onto a tab of its own can bind the
        identical keys at a scope the trader can actually reach, instead of
        hardcoding a second copy of this list that drifts.
        """
        return (
            ("Alt+V", self.focus_veto),
            ("Alt+K", self.focus_like),
            ("Alt+S", self.focus_hypo_stop),
            ("Alt+N", self.focus_note),
        )

    def _bind_shortcuts(self) -> None:
        """Alt+letter, not bare letters: the rail is full of text inputs and a
        bare 'v' has to stay a 'v' when the trader is typing a note."""
        if self._bind_action_shortcuts:
            for sequence, handler in self.action_shortcuts():
                shortcut = QShortcut(QKeySequence(sequence), self)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(handler)

        # Digits pick a reason, but only while the reason list has focus, so
        # they never swallow a keystroke meant for a note field.
        if self._vocabulary is not None:
            for reason in self._vocabulary.reasons:
                shortcut = QShortcut(QKeySequence(reason.hotkey), self.reason_list)
                shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
                shortcut.activated.connect(
                    lambda code=reason.code: self.select_reason(code)
                )

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------
    def set_context(
        self,
        *,
        symbol: str,
        side: str | None = None,
        last_price: float | None = None,
        timeframe: str | None = None,
        ref_level_id: str = "",
        ref_level_family: str = "",
    ) -> None:
        """Point the rail at a symbol. Capture actions apply to this context."""
        self._symbol = str(symbol or "").strip().upper()
        if side:
            resolved = "SHORT" if str(side).strip().upper().startswith("SHORT") else "LONG"
            self._side = resolved
            self.side_input.blockSignals(True)
            self.side_input.setCurrentText(resolved)
            self.side_input.blockSignals(False)
        if last_price is not None:
            self._last_price = float(last_price)
        if timeframe:
            self._timeframe = str(timeframe).strip().upper()
        self._ref_level_id = str(ref_level_id or "")
        self._ref_level_family = str(ref_level_family or "")
        self.symbol_label.setText(self._symbol or "-")
        armed = bool(self._symbol)
        for button in (
            self.veto_button,
            self.veto_day_trade_button,
            self.like_button,
            self.stop_button,
            self.note_button,
        ):
            button.setEnabled(armed)
        if self._vocabulary is None:
            self.veto_button.setEnabled(False)
            self.veto_day_trade_button.setEnabled(False)
        if self._last_price and self.stop_input.value() == 0.0:
            self.stop_input.setValue(float(self._last_price))
        self._set_status("" if armed else "Look up a symbol to start capturing.")

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def side(self) -> str:
        return self._side

    def _on_side_changed(self, text: str) -> None:
        self._side = "SHORT" if str(text).upper().startswith("SHORT") else "LONG"

    # ------------------------------------------------------------------
    # keyboard entry points
    # ------------------------------------------------------------------
    def focus_veto(self) -> None:
        if not self.reason_list.isEnabled():
            return
        if self.reason_list.currentRow() < 0 and self.reason_list.count():
            self.reason_list.setCurrentRow(0)
        self.reason_list.setFocus()

    def focus_like(self) -> None:
        self.setup_input.setFocus()

    def focus_hypo_stop(self) -> None:
        self.stop_input.setFocus()
        self.stop_input.selectAll()

    def focus_note(self) -> None:
        self.note_input.setFocus()

    def select_reason(self, code: str) -> None:
        """Select a reason by code. Commits immediately unless a note is due."""
        for row in range(self.reason_list.count()):
            item = self.reason_list.item(row)
            if item.data(_REASON_ROLE) == code:
                self.reason_list.setCurrentRow(row)
                break
        else:
            return
        self._sync_note_requirement()
        if self._selected_reason_requires_note():
            self.veto_note_input.setFocus()
            self._set_status("This reason needs a note - type it, then Enter.")
            return
        self.commit_veto()

    def selected_reason_code(self) -> str:
        item = self.reason_list.currentItem()
        return str(item.data(_REASON_ROLE)) if item is not None else ""

    def _selected_reason_requires_note(self) -> bool:
        if self._vocabulary is None:
            return False
        reason = self._vocabulary.reason(self.selected_reason_code())
        return bool(reason and reason.note_required)

    def _sync_note_requirement(self) -> None:
        required = self._selected_reason_requires_note()
        self.veto_note_input.setPlaceholderText("note (required)" if required else "note (optional)")

    # ------------------------------------------------------------------
    # captures
    # ------------------------------------------------------------------
    def _common_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "symbol": self._symbol,
            "timeframe": self._timeframe,
        }
        if self._last_price:
            fields["last_price"] = self._last_price
        if self._ref_level_id:
            fields["ref_level_id"] = self._ref_level_id
        if self._ref_level_family:
            fields["ref_level_family"] = self._ref_level_family
        if self._annotations_path is not None:
            fields["path"] = self._annotations_path
        return fields

    def _record(self, event_type: str, **fields: Any) -> dict | None:
        if not self._symbol:
            self._set_status("No symbol in focus.", ok=False)
            return None
        try:
            row = record_annotation(event_type, **{**self._common_fields(), **fields})
        except AnnotationError as exc:
            self._set_status(str(exc), ok=False)
            return None
        if row is None:
            self._set_status(
                "NOT SAVED - the annotation log could not be written.", ok=False
            )
            return None
        self.captured.emit(event_type, row)
        return row

    def commit_veto(self) -> dict | None:
        code = self.selected_reason_code()
        if not code:
            self._set_status("Pick a reason (1-9).", ok=False)
            return None
        row = self._record(
            EVENT_VETO,
            reason_code=code,
            side=self._side,
            note=self.veto_note_input.text(),
            vocabulary=self._vocabulary,
        )
        if row is None:
            return None
        self.veto_note_input.clear()
        detail = self._merge_veto_cohort_safely()
        self._set_status(f"VETO {row['symbol']} - {code}{detail}")
        return row

    def veto_keeps_chart(self) -> bool:
        """True while a day-trade veto is mid-commit; see ``_veto_keeps_chart``."""
        return self._veto_keeps_chart

    def commit_veto_day_trade(self) -> dict | None:
        """Veto the D1 chart, then ASK the host to day-trade the name.

        The veto row written here is an ordinary veto - identical bytes,
        identical validation, identical cohort merge. Nothing about the
        annotation schema changes and nothing new is written: the request
        that follows is a signal, and the Focus store's own writer decides
        what to do with it. If that placement fails the veto still stands,
        which is the right way round - the judgement is the evidence.
        """
        self._veto_keeps_chart = True
        try:
            row = self.commit_veto()
        finally:
            self._veto_keeps_chart = False
        if row is None:
            return None
        self.vetoDayTradeRequested.emit(dict(row))
        return row

    def _merge_veto_cohort_safely(self) -> str:
        """Forward tracking is capture-side and must never break a capture."""
        try:
            kwargs: dict[str, Any] = {}
            if self._annotations_path is not None:
                kwargs["annotations_path"] = self._annotations_path
            result = self._merge_veto_cohort(**kwargs)
        except Exception:
            return "  (cohort update deferred)"
        if isinstance(result, dict) and not result.get("written", True):
            return "  (cohort update deferred)"
        return ""

    def commit_like(self) -> dict | None:
        setup_id = self.setup_input.currentData()
        if not setup_id:
            self._set_status("Pick a setup to claim.", ok=False)
            return None
        row = self._record(
            EVENT_LIKE_CLAIM,
            claimed_setup_id=str(setup_id),
            side=self._side,
            note=self.like_note_input.text(),
        )
        if row is None:
            return None
        # The like is a recorded judgement, nothing more. It must not add the
        # symbol to Focus or any watchlist: Focus membership changes live
        # alerting, and this rail is analysis-only. Adding a name to a list
        # stays an explicit action on the Focus surfaces that own those files.
        self.like_note_input.clear()
        self._set_status(f"LIKE {row['symbol']} - {setup_id}")
        return row

    def commit_hypo_stop(self) -> dict | None:
        price = float(self.stop_input.value())
        if price <= 0:
            self._set_status("Enter a stop price.", ok=False)
            return None
        setup_id = self.setup_input.currentData()
        row = self._record(
            EVENT_HYPO_STOP,
            stop_price=price,
            side=self._side,
            claimed_setup_id=str(setup_id or ""),
        )
        if row is None:
            return None
        self._set_status(f"STOP {row['symbol']} {self._side} @ {price:g} (no order placed)")
        return row

    def commit_note(self) -> dict | None:
        text = self.note_input.text().strip()
        if not text:
            self._set_status("Nothing to save.", ok=False)
            return None
        row = self._record(EVENT_NOTE, note=text, side=self._side)
        if row is None:
            return None
        self.note_input.clear()
        self._set_status(f"NOTE {row['symbol']}")
        return row

    # ------------------------------------------------------------------
    def _set_status(self, message: str, *, ok: bool = True) -> None:
        self.status_label.setText(message)
        colour = theme.color("long" if ok else "short")
        self.status_label.setStyleSheet(f"color: {colour};" if message else "")

    def status_text(self) -> str:
        return self.status_label.text()
