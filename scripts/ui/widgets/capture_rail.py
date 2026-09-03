"""The capture rail: the trader's decisions, each in about two keystrokes.

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
* NOTE      -> ui.annotations.store.
* PASS      -> ui.annotations.store, one row carrying every ticked pass
               reason, plus the M5 bars the desk already held (sidecar).

The PASS section (2026-08-31) sits under Note because the trader asked for it
there, and because it shares that section's free-text field: a pass is a note
with the reason ticked. Trader: "many times I really like this stock for a
daytrade but it has this ONE issue" - so a pass is NOT a veto and, like a
note, it never retires the chart. The host decides that, and the host's rule
is the one in CLAUDE.md: a veto and a like each retire the chart, a note never
does. A pass is on the note side of that line.

The hypothetical stop was removed from this surface on 2026-08-20 (trader:
"get rid of hypothetical stop for now its not useful"). Only the CONTROL is
gone. `ui.annotations.store` still builds and validates `hypo_stop` rows,
because the stream is append-only evidence and rows already written have to
stay readable - deleting the schema would make history unparseable to buy
nothing. Re-adding the control is a layout change, not a migration.

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

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from ui.annotations import setup_claims, verdicts
from ui.annotations.setup_claims import (  # noqa: F401  (re-exported for hosts/tests)
    EXTRA_CLAIM_IDS,
    MAIN_CLAIM_GROUP,
    setup_claim_groups,
)
from ui.annotations.store import (
    AnnotationError,
    EVENT_LIKE_CLAIM,
    EVENT_NOTE,
    EVENT_PASS,
    EVENT_VETO,
    LIKE_MODE_CLAIMED,
    LIKE_MODE_QUICK,
    record_annotation,
    record_annotation_with_bars,
)
from ui.annotations.vocabulary import (
    VocabularyError,
    load_pass_vocabulary,
    load_veto_vocabulary,
)
from ui.widgets.flow_layout import FlowLayout

_REASON_ROLE = Qt.ItemDataRole.UserRole
_CLAIM_ROLE = Qt.ItemDataRole.UserRole

#: The note field's resting hint, restored whenever no verb is waiting on it.
NOTE_PLACEHOLDER = "freeform note"

#: One keystroke per claim, in list order. Digits first so the nine main-swing
#: claims keep the exact keys the trader already presses; letters continue the
#: run because there is no tenth digit and a two-key sequence would cost the
#: five-second contract this rail exists to keep. A row's label starts with its
#: own key, so QListWidget's type-search lands on the same row either way.
CLAIM_HOTKEYS = "1234567890qwertyuiop"


def offered_setup_claims() -> list:
    """The claims this rail offers, in display order.

    The definition moved to :mod:`ui.annotations.setup_claims` on 2026-08-24 so
    that ``ai_summary`` can state the offered list as a machine-written caveat
    without importing Qt. Delegated through the MODULE rather than bound at
    import time, so a test that patches the source patches this too - the rail
    and the caveat must never be able to disagree about what was offered.
    """
    return setup_claims.offered_setup_claims()


class CaptureRail(QFrame):
    """Veto / like+claim / note / day-trade pass for the focused symbol."""

    #: (event_type, row) after a row reaches disk. Analysis-only consumers.
    captured = Signal(str, dict)
    #: (row) after a veto whose D1 chart the trader rejected but whose NAME
    #: they still want to day-trade. A REQUEST, not a write: the rail has
    #: never placed a name on a list and still does not (see the module
    #: docstring - an earlier draft routed likes through FocusService.add and
    #: had to be torn back out). The host that owns the Focus store performs
    #: the placement, so that store keeps exactly one writer.
    vetoDayTradeRequested = Signal(dict)
    #: (event_type, verdict_row) once the trader has finished typing about a
    #: retiring verb - S1.2, trader 2026-09-03: *"when I hit like or not today
    #: or anything, it should keep the chart up UNTIL I finish typing."*
    #:
    #: The verb's own row is already on disk (``captured`` fired at the click).
    #: This says the trader is DONE with the chart, so the host may now retire
    #: it. A host that never calls :meth:`begin_follow_up` never sees this and
    #: keeps today's behaviour exactly - which is what the rail hosted in the
    #: snapshot popup and in the tests does.
    followUpSettled = Signal(str, dict)

    #: What the note field says while a verb is waiting for the trader.
    FOLLOW_UP_PLACEHOLDER = "Enter to advance - type first to add a note"
    FOLLOW_UP_HINT = "  ·  type a note then Enter, or Enter to move on."

    def __init__(
        self,
        *,
        annotations_path: Any = None,
        veto_cohort_merge: Callable[..., dict] | None = None,
        like_cohort_merge: Callable[..., dict] | None = None,
        pass_cohort_merge: Callable[..., dict] | None = None,
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
        # P10: which screen this rail's verbs report, and the scanner row behind
        # the chart when the host has one. `rail` is the honest default - a
        # verdict typed on the rail itself came from the rail - and a host that
        # owns a different screen overrides it through `set_scan_context`.
        self._surface = verdicts.SURFACE_RAIL
        self._scan_context: dict[str, Any] = {}
        # True only for the duration of a "veto but day-trade it" commit.
        # `captured` fires synchronously from inside commit_veto(), i.e.
        # BEFORE commit_veto_day_trade() can emit its own request, so a host
        # that retires the chart on any veto would retire this one too - and
        # the object the Focus placement needs would already be gone. This is
        # how that host is told to hold the chart for one commit.
        self._veto_keeps_chart = False
        # S1.2. (event_type, row) of the retiring verb whose chart is being
        # held up while the trader types. None means "no verb is waiting", and
        # then Enter in the note field is an ordinary note exactly as before.
        self._pending_follow_up: tuple[str, dict] | None = None
        # Supplied by the host that owns a chart, and called only at commit
        # time. It reads memory the desk already materialised and must never
        # fetch: no bars cached is an ordinary outcome, and the pass row is
        # written with its timestamp alone (trader, 2026-08-31: "if that is
        # hard just store the exact timestamp and the AI can read the charts").
        self._m5_bars_provider: Callable[[], list] | None = None

        if veto_cohort_merge is not None:
            self._merge_veto_cohort = veto_cohort_merge
        else:
            from ui.annotations.veto_cohort import merge_veto_cohort_picks

            self._merge_veto_cohort = merge_veto_cohort_picks

        # The LIKE half of the same decision, merged on the same click for the
        # same reason (2026-09-01). It was nightly-only: `like_cohort_picks.csv`
        # was last written 2026-08-27 against likes recorded through 09-01, so
        # a like was invisible to its own cohort for up to a day - and on any
        # day the overnight job did not run, indefinitely. The nightly merge
        # stays; both are idempotent, so running twice adds nothing.
        if like_cohort_merge is not None:
            self._merge_like_cohort = like_cohort_merge
        else:
            from ui.annotations.like_cohort import merge_like_cohort_picks

            self._merge_like_cohort = merge_like_cohort_picks
        # P5: the PASS cohort merged on the same click, for the same reason the
        # veto is. Idempotent, so the nightly slot re-running it adds nothing.
        if pass_cohort_merge is not None:
            self._merge_pass_cohort = pass_cohort_merge
        else:
            from ui.annotations.pass_cohort import merge_pass_cohort_picks

            self._merge_pass_cohort = merge_pass_cohort_picks

        try:
            self._vocabulary = load_veto_vocabulary()
            self._vocabulary_error = ""
        except VocabularyError as exc:
            # Fail visible: the veto action is disabled and says why, rather
            # than writing reason codes no later analysis would recognise.
            self._vocabulary = None
            self._vocabulary_error = str(exc)

        try:
            self._pass_vocabulary = load_pass_vocabulary()
            self._pass_vocabulary_error = ""
        except VocabularyError as exc:
            # Same fail-visible rule as the veto list, for the same reason.
            self._pass_vocabulary = None
            self._pass_vocabulary_error = str(exc)

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

        # Symbol and side share one line. They were two, which cost a row of
        # height on every host for one combo box and one word.
        head_row = QHBoxLayout()
        head_row.setSpacing(theme.px(6))
        self.symbol_label = QLabel("-")
        self.symbol_label.setObjectName("SectionTitle")
        head_row.addWidget(self.symbol_label)
        head_row.addSpacing(theme.px(10))
        head_row.addWidget(QLabel("Side"))
        self.side_input = QComboBox()
        self.side_input.addItems(["LONG", "SHORT"])
        self.side_input.currentTextChanged.connect(self._on_side_changed)
        head_row.addWidget(self.side_input)
        head_row.addStretch(1)
        layout.addLayout(head_row)

        # The sections FLOW rather than stack (trader, 2026-08-20: "I do think
        # we can sort some of these into columns so we can see more"). Stacked,
        # the rail was ~900px of single-column controls in a dialog 1700px
        # wide - two thirds of that width was blank while the trader scrolled
        # to reach Note. FlowLayout is the same primitive the arm bar uses for
        # the same reason: wide hosts get them side by side, and the narrow
        # Capture tab still gets a single column with nothing clipped.
        sections = FlowLayout(margin=0, spacing=theme.px(8))
        for section in (self._veto_section(), self._like_section(), self._note_section()):
            section.setMinimumWidth(theme.px(280))
            sections.addWidget(section)
        layout.addLayout(sections)
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
        frame, inner = self._section("Veto  (Alt+V, then the key shown)")
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
        # Show EVERY reason. A fixed 190px cap showed six of nine and scrolled
        # the rest, which is the opposite of a surface built for two
        # keystrokes: the trader cannot press the digit for a reason they
        # cannot see. Sized from the vocabulary rather than hardcoded, so
        # adding a reason does not quietly push one below the fold.
        #
        # Deliberately NOT a wrapped multi-column list: the labels ("Sector
        # mate earnings pending") are long enough that columns only fit by
        # eliding them, and a veto vocabulary the trader has to guess at is
        # worse than one that takes a few more pixels of a row that now sits
        # beside two other sections instead of above them.
        rows = max(1, min(self.reason_list.count(), 14))
        self.reason_list.setMaximumHeight(
            rows * theme.px(21) + theme.px(10)
        )
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
        """Same shape as the veto: a numbered picklist, not a dropdown.

        Trader, 2026-08-20: "layout the like + claim similiar to the veto but
        only do the main setups for now." A combo hides every option until it
        is opened and costs a click to read, which is the opposite of the
        five-second contract the veto list is built to. Main swing holds nine
        claims, so the same 1-9 digits work here, and double-click and Enter
        commit it exactly as they do a veto - meaning they ATTEMPT the commit
        and fall back to asking for the why, the way a veto falls back to
        asking for a `note_required` reason's note (trader, 2026-08-27).

        The nine main-swing claims keep digits 1-9 exactly as before; the
        claims added on 2026-08-21 (the three post-earnings families and the
        2nd-dev breakout) continue the run on 0 and then letters. What is
        offered is decided by MAIN_CLAIM_GROUP + EXTRA_CLAIM_IDS, so admitting
        another family stays a one-line change and never a migration - a claim
        id is valid the moment the registry names it.
        """
        frame, inner = self._section("Like + claim  (Alt+K, then the key shown)")
        self.setup_list = QListWidget()
        self.setup_list.setObjectName("SetupClaimList")
        self.setup_list.setAlternatingRowColors(False)
        self._claim_hotkeys: dict[str, str] = {}
        for position, claim in enumerate(offered_setup_claims()):
            hotkey = CLAIM_HOTKEYS[position] if position < len(CLAIM_HOTKEYS) else ""
            item = QListWidgetItem(f"{hotkey or ' '}  {claim.label}")
            item.setData(_CLAIM_ROLE, claim.setup_id)
            if claim.summary:
                item.setToolTip(claim.summary)
            self.setup_list.addItem(item)
            if hotkey:
                self._claim_hotkeys[hotkey] = claim.setup_id
        # Double-click and the digit do the same thing, and it is the same
        # thing the veto's reason list does: try to commit, and ask for the why
        # when there is not one yet (R9.2's required why is enforced inside
        # `commit_like`, not by refusing to call it).
        self.setup_list.itemActivated.connect(lambda item: self._claim_picked(item))
        rows = max(1, min(self.setup_list.count(), 14))
        self.setup_list.setMaximumHeight(rows * theme.px(21) + theme.px(10))
        inner.addWidget(self.setup_list)

        self.like_note_input = QLineEdit()
        # S1.1: OPTIONAL. The claim is the requirement; the why is the offer.
        self.like_note_input.setPlaceholderText("why (optional)")
        self.like_note_input.returnPressed.connect(self.commit_like)
        inner.addWidget(self.like_note_input)
        self.like_button = QPushButton("Like + claim setup")
        self.like_button.clicked.connect(self.commit_like)
        inner.addWidget(self.like_button)

        # The quick like's own button, beside the claimed one it is NOT. Every
        # other verb on this rail has a button as well as a key; this one had
        # only a key until the trader asked for both (2026-09-02).
        self.quick_like_button = QPushButton("♥ Quick like  (Alt+L)")
        self.quick_like_button.setToolTip(
            "Records that something about this chart was good, without naming "
            "the setup. Writes at once - Alt+L does exactly the same. The "
            "chart then waits: type a note and press Enter, or just press "
            "Enter. Nothing is added to Focus or any watchlist."
        )
        self.quick_like_button.clicked.connect(self.prompt_quick_like)
        inner.addWidget(self.quick_like_button)
        return frame

    def _note_section(self) -> QFrame:
        """Note, and directly under it the day-trade pass.

        Under, and inside the same section, by trader instruction
        (2026-08-31): "in the capture window add a section under the note area
        where I can tick a few reasons for passing... plus the existing note."
        Nesting it here rather than adding a fourth flow section is what makes
        "under" true at every host width - a fourth section flows to the RIGHT
        of Note on a wide host - and it is what lets the pass reuse the one
        free-text field instead of introducing a second one for the same job.
        """
        frame, inner = self._section("Note  (Alt+N)")
        self.note_input = QLineEdit()
        self.note_input.setPlaceholderText(NOTE_PLACEHOLDER)
        # S1.2: Enter means "save this note" normally, and "I have finished
        # typing about the verb I just clicked - move on" while a retiring verb
        # is waiting. One field, because the trader's hands are already on it
        # and a second box would be the pop-up they asked us to remove.
        self.note_input.returnPressed.connect(self._on_note_return)
        # Escape is not a QLineEdit gesture, so it needs a filter rather than a
        # signal. It only ever means "advance without the note"; with nothing
        # waiting the event is passed straight on.
        self.note_input.installEventFilter(self)
        inner.addWidget(self.note_input)
        self.note_button = QPushButton("Save note")
        self.note_button.clicked.connect(self.commit_note)
        inner.addWidget(self.note_button)
        inner.addWidget(self._pass_block())
        return frame

    def _pass_block(self) -> QFrame:
        """The "Passed - why?" block: the ONE issue that cost a liked trade.

        Checkboxes, not a picklist, because the trader said several reasons can
        be true of one pass. The digits still work the way they do on the veto
        list, scoped to this box alone, so a 3 typed into the note above stays
        a 3 - the box holds only the checkboxes for exactly that reason, and
        the shared note field sits outside it.
        """
        box = QFrame()
        box.setObjectName("PassReasonBox")
        inner = QVBoxLayout(box)
        inner.setContentsMargins(0, theme.px(6), 0, 0)
        inner.setSpacing(theme.px(3))
        heading = QLabel("Passed - why?  (Alt+P)")
        heading.setObjectName("SectionTitle")
        heading.setToolTip(
            "Liked it for a day trade and did not take it. Tick every reason "
            "that applied; the note above rides along. Nothing is muted, "
            "removed or scored - the chart stays up, exactly as it does for a "
            "note."
        )
        inner.addWidget(heading)

        self.pass_reason_box = QFrame()
        self.pass_reason_box.setObjectName("PassReasonChecks")
        checks = QVBoxLayout(self.pass_reason_box)
        checks.setContentsMargins(0, 0, 0, 0)
        checks.setSpacing(theme.px(2))
        self.pass_checkboxes: dict[str, QCheckBox] = {}
        self._pass_hotkeys: dict[str, str] = {}
        if self._pass_vocabulary is not None:
            for reason in self._pass_vocabulary.reasons:
                check = QCheckBox(f"{reason.hotkey}  {reason.label}")
                if reason.hint:
                    check.setToolTip(reason.hint)
                checks.addWidget(check)
                self.pass_checkboxes[reason.code] = check
                if reason.hotkey:
                    self._pass_hotkeys[reason.hotkey] = reason.code
        inner.addWidget(self.pass_reason_box)

        self.pass_button = QPushButton("Record pass")
        self.pass_button.setToolTip(
            "Write one row: the ticked reasons, the note above, the exact "
            "timestamp, and - only when the desk already holds them - this "
            "session's M5 bars, so the chart can be read back as it was."
        )
        self.pass_button.clicked.connect(self.commit_pass)
        if self._pass_vocabulary is None:
            self.pass_button.setEnabled(False)
            self.pass_button.setToolTip(self._pass_vocabulary_error)
            warning = QLabel(f"Pass disabled: {self._pass_vocabulary_error}")
            warning.setWordWrap(True)
            inner.addWidget(warning)
        inner.addWidget(self.pass_button)
        return box

    def action_shortcuts(self) -> tuple[tuple[str, Callable[[], None]], ...]:
        """The rail's key bindings, as (sequence, handler) pairs.

        Public so a host that took the rail onto a tab of its own can bind the
        identical keys at a scope the trader can actually reach, instead of
        hardcoding a second copy of this list that drifts.
        """
        return (
            ("Alt+V", self.focus_veto),
            ("Alt+K", self.focus_like),
            ("Alt+N", self.focus_note),
            ("Alt+P", self.focus_pass),
            # P9. Alt+L is UNBOUND everywhere in scripts/ui - the whole
            # inventory is Ctrl+F, Ctrl+J, Ctrl+R, Ctrl+Return, F9, Alt+E and
            # these four - and two live bindings for one sequence is an
            # ambiguous shortcut that fires NEITHER, so a clash would silently
            # cost the trader both verbs.
            ("Alt+L", self.commit_quick_like),
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
        # The claim list gets the identical treatment, scoped to itself, so
        # Alt+K then 3 is a whole like - and a 3 typed into a note stays a 3.
        for hotkey, setup_id in self._claim_hotkeys.items():
            shortcut = QShortcut(QKeySequence(hotkey), self.setup_list)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(lambda claim=setup_id: self.select_setup(claim))
        # And the pass checkboxes, scoped to the box that holds ONLY them -
        # WidgetWithChildren, because focus sits on a child QCheckBox rather
        # than on the box itself. The three digit maps can never be in context
        # at the same time (each needs the focus inside its own widget), so
        # this is not a second live binding for one sequence.
        for hotkey, code in self._pass_hotkeys.items():
            shortcut = QShortcut(QKeySequence(hotkey), self.pass_reason_box)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda pass_code=code: self.toggle_pass_reason(pass_code)
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
            self.note_button,
            self.pass_button,
        ):
            button.setEnabled(armed)
        if self._vocabulary is None:
            self.veto_button.setEnabled(False)
            self.veto_day_trade_button.setEnabled(False)
        if self._pass_vocabulary is None:
            self.pass_button.setEnabled(False)
        self._set_status("" if armed else "Look up a symbol to start capturing.")

    def set_m5_bars_provider(self, provider: Callable[[], list] | None) -> None:
        """Tell the rail where the charted symbol's M5 bars already live.

        The host passes a zero-argument callable that reads ITS OWN
        already-materialised bars (``SymbolSnapshotWidget.cached_m5_bars``).
        The rail never reaches for a bot, a service or a feed: attaching a
        chart to a pass is a bonus the desk can offer when it happens to be
        holding one, and it must not become a reason for a capture click to
        touch the network or block the Qt thread.
        """
        self._m5_bars_provider = provider

    def cached_m5_bars(self) -> list:
        """Whatever the host already has, or ``[]``. Never fetches, never raises."""
        provider = self._m5_bars_provider
        if provider is None:
            return []
        try:
            return list(provider() or [])
        except Exception:
            # A chart that cannot answer costs the attachment, never the row.
            return []

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
        if self.setup_list.currentRow() < 0 and self.setup_list.count():
            self.setup_list.setCurrentRow(0)
        self.setup_list.setFocus()

    def focus_note(self) -> None:
        self.note_input.setFocus()

    def focus_pass(self) -> None:
        """Alt+P: land on the pass checkboxes so 1-5 and space work at once."""
        for check in self.pass_checkboxes.values():
            check.setFocus()
            return
        self.note_input.setFocus()

    def toggle_pass_reason(self, code: str) -> None:
        """Tick or untick one pass reason. A digit is a toggle, never a commit.

        Deliberately unlike the veto digit, which commits on the spot: a pass
        is multi-select, so the trader has to be able to press 2 and 4 before
        anything is written.
        """
        check = self.pass_checkboxes.get(str(code or "").strip().lower())
        if check is not None:
            check.setChecked(not check.isChecked())

    def selected_pass_codes(self) -> list[str]:
        """Ticked reasons in VOCABULARY order, which is the order written."""
        return [code for code, check in self.pass_checkboxes.items() if check.isChecked()]

    def clear_pass_selection(self) -> None:
        for check in self.pass_checkboxes.values():
            check.setChecked(False)

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
    def set_scan_context(self, context: Any = None, *, surface: str = "") -> None:
        """The scanner row behind the chart, and which screen this rail serves.

        P10 B1. The host sets it when it charts something it HAS a row for; a
        bare symbol lookup sets nothing and the fields are simply absent, which
        is the honest answer rather than a row of empty strings.

        `surface` is an override for a host that is not the capture rail itself -
        the chart's own verb row reports `chart_review`, because that is the
        screen the trader clicked on, and a rollup asking "is the trader a better
        judge from the setups table or the chart?" needs that to be true.
        """
        self._scan_context = dict(context or {})
        if surface:
            self._surface = verdicts._validated_surface(surface)

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

    def _record(
        self,
        event_type: str,
        *,
        writer: Any = None,
        **fields: Any,
    ) -> dict | None:
        """Every non-like annotation this rail writes: veto, pass, note, stop.

        V3 item 4 closed a seam here. P10 gave the LIKE path a `surface` and this
        one kept writing without it, so a veto and a like from the same rail
        landed with different shapes and any rollup by screen silently omitted
        every veto. One writer means one row shape.

        R4 B5 closed the second half of it. `commit_pass` needed the SIDECAR
        writer, so it called `record_pass_annotation` directly and skipped this
        method entirely - and with it the two `setdefault` lines below. Every
        day-trade pass therefore landed with no `surface` and no scan context
        while the veto and the note beside it carried both. `writer` is that one
        difference and nothing else: which store function appends the row. The
        stamping happens here, once, for every verb.
        """
        if not self._symbol:
            self._set_status("No symbol in focus.", ok=False)
            return None
        common = {**self._common_fields(), **fields}
        common.setdefault("surface", self._surface)
        common.setdefault("scan_context", dict(self._scan_context or {}))
        try:
            row = (writer or record_annotation)(event_type, **common)
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
        self._append_follow_up_hint()
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
        return self._merge_cohort_safely(self._merge_veto_cohort)

    def _merge_like_cohort_safely(self) -> str:
        """The like's half, identical in shape to the veto's (2026-09-01)."""
        return self._merge_cohort_safely(self._merge_like_cohort)
    def _merge_pass_cohort_safely(self) -> str:
        """The PASS cohort's half, identical in shape to the veto's (P5)."""
        return self._merge_cohort_safely(self._merge_pass_cohort)

    def _merge_cohort_safely(self, merge) -> str:
        """Forward tracking is capture-side and must never break a capture.

        An evidence store never costs the event it records: the annotation row
        is already on disk when this runs, so every failure here degrades to a
        status suffix and the next merge - nightly or the next click - picks the
        row up. Both merges are idempotent, which is what makes that true.
        """
        try:
            kwargs: dict[str, Any] = {}
            if self._annotations_path is not None:
                kwargs["annotations_path"] = self._annotations_path
            result = merge(**kwargs)
        except Exception:
            return "  (cohort update deferred)"
        if isinstance(result, dict) and not result.get("written", True):
            return "  (cohort update deferred)"
        return ""

    def selected_setup_id(self) -> str:
        item = self.setup_list.currentItem()
        return str(item.data(_CLAIM_ROLE) or "") if item is not None else ""

    def select_setup(self, setup_id: str) -> None:
        """Select a claim by id, then try to commit it.

        This is `select_reason`'s shape, deliberately - trader, 2026-08-27:
        "i want to be able to double click the like and claim the same way i
        can double click the veto." The veto's gesture does not bypass its note
        rule; it ATTEMPTS the commit and `commit_veto` diverts to the note
        field when that reason's ``note_required`` is unmet. So the like's
        gesture now calls `commit_like`.

        S1.1 removed the why guard entirely (trader, 2026-09-03), so the gesture
        and the button now always write. The chart still waits afterwards, so a
        why the trader wants to give is typed into the note field rather than
        demanded before the judgement is recorded.
        """
        for row in range(self.setup_list.count()):
            if self.setup_list.item(row).data(_CLAIM_ROLE) == setup_id:
                self.setup_list.setCurrentRow(row)
                break
        else:
            return
        self.commit_like()

    def _claim_picked(self, item) -> None:
        """Double-click lands here so it behaves exactly like the digit."""
        if item is not None:
            self.setup_list.setCurrentItem(item)
        self.commit_like()

    def commit_quick_like(self, note: str = "") -> dict | None:
        """One key: "something about this was good", and nothing else.

        Trader, 2026-09-02: *"anytime I like and claim a setup or like a day
        trade setup I just want to let the bot and the future AI know
        'something about this was good' and then we can figure out what about
        it / what's the best entry later."*

        This SUPERSEDES R9.2(a)'s "why is required" for this path only. The
        claimed path - Alt+K, digit, why, Enter - is untouched, and the reason
        it still demands a why is unchanged: a claim without one is a label
        nobody can check later.

        Everything a claimed like does to the review, this does too. The chart
        RETIRES (a like retires, a note never does), the symbol is marked
        reviewed today through the existing `_ANNOTATION_DECISIONS`, and the
        review event is `like_advance` so the scoreboard counts it as a take.

        And everything a like has never done, this does not do either: NO Focus
        placement, no park, no watch, no alert, no watchlist. A like carries
        zero privileges (plan.md P3.1), and the whole value of a one-key verb is
        lost if the trader has to wonder what else it did.

        On an M5 chart it saves the bars the desk is already holding, exactly as
        a pass does - the trader asked for that explicitly. On a D1 chart it
        writes no sidecar: a D1 chart's bars are not what the intraday grade
        needs, and an empty sidecar would be a reference that lies.

        `note` is OPTIONAL and defaults to nothing, which is not a
        contradiction of R9.2(a): that rule REQUIRES a why on a claimed like,
        and this path has no claim to justify. The keystroke passes none - one
        key has to stay one key - and the chart button offers a box in case the
        trader has a sentence in mind (trader, 2026-09-02: *"maybe it can have a
        pop up with a note I can put in"*).
        """
        row = self._record_like(
            claimed_setup_id="", note=note, like_mode=LIKE_MODE_QUICK
        )
        if row is None:
            return None
        detail = self._merge_like_cohort_safely()
        attached = row.get("m5_bar_count")
        if attached:
            detail = f"  ({attached} M5 bars attached){detail}"
        self._set_status(
            f"Liked (quick) {row['symbol']} - claim it later with Alt+K{detail}"
        )
        self._append_follow_up_hint()
        return row

    def prompt_quick_like(self) -> dict | None:
        """The quick-like BUTTON. Writes at once; asks nothing.

        S1.1 retired the `QInputDialog` this used to open. Trader, 2026-09-03:
        *"when I hit something in the capture tab such as veto, or like and
        claim etc that is sufficient reason enough - these are quick buttons to
        get a note in essentially and do NOT require a pop up note."*

        That SUPERSEDES the 2026-09-02 "the BUTTON prompts" rule: the button and
        Alt+L are now the same verb, and the optional sentence is typed into the
        rail's own note field while the chart waits (S1.2) instead of into a
        modal that stopped the desk to ask.

        The name stays because both quick-like buttons - this rail's and the
        chart's verb row - are wired to it, and a route with one implementation
        cannot drift from itself.
        """
        return self.commit_quick_like()

    def _record_like(self, **fields: Any) -> dict | None:
        """Write a like row, with the M5 sidecar when the chart is an M5 one.

        One writer for both like paths so a claimed like and a quick like can
        never disagree about how a like is stored.
        """
        if not self._symbol:
            self._set_status("No symbol in focus.", ok=False)
            return None
        common = {**self._common_fields(), "side": self._side, **fields}
        common.setdefault("surface", self._surface)
        common.setdefault("scan_context", dict(self._scan_context or {}))
        try:
            # P10 A1: through the ONE writer, so the rail's like and a star in
            # Master AVWAP are the same row shape and grade in one bucket. The
            # M5 branch is unchanged - `record_like` passes the bars along to the
            # same `record_annotation_with_bars` this used directly.
            row = verdicts.record_like(
                m5_bars=self.cached_m5_bars()
                if str(self._timeframe or "").upper() == "M5"
                else (),
                **common,
            )
        except AnnotationError as exc:
            self._set_status(str(exc), ok=False)
            return None
        if row is None:
            self._set_status(
                "NOT SAVED - the annotation log could not be written.", ok=False
            )
            return None
        self.captured.emit(EVENT_LIKE_CLAIM, row)
        return row

    def commit_like(self) -> dict | None:
        setup_id = self.selected_setup_id()
        if not setup_id:
            self._set_status("Pick a setup to claim (1-9).", ok=False)
            return None
        # S1.1: THE CLAIM IS THE WHOLE REQUIREMENT. An empty why is accepted and
        # recorded as nothing rather than refused - trader, 2026-09-03: *"when I
        # hit something in the capture tab such as veto, or like and claim etc
        # that is sufficient reason enough."* This supersedes R9.2(a) for the
        # claimed path too; the quick path was already exempt (P9).
        #
        # Nothing is lost by it: the chart now stays up after the click (S1.2)
        # with the cursor in the note field, so the why the trader DOES want to
        # write still reaches the stream - as its own row, joined by
        # `supersedes` - and the one they do not is no longer the difference
        # between a recorded judgement and none at all.
        why = self.like_note_input.text().strip()
        row = self._record_like(
            claimed_setup_id=str(setup_id),
            note=why,
            like_mode=LIKE_MODE_CLAIMED,
        )
        if row is None:
            return None
        # The like is a recorded judgement, nothing more. It must not add the
        # symbol to Focus or any watchlist: Focus membership changes live
        # alerting, and this rail is analysis-only. Adding a name to a list
        # stays an explicit action on the Focus surfaces that own those files.
        self.like_note_input.clear()
        detail = self._merge_like_cohort_safely()
        self._set_status(f"LIKE {row['symbol']} - {setup_id}{detail}")
        self._append_follow_up_hint()
        return row


    def commit_pass(self) -> dict | None:
        """Record a day-trade pass. Writes one row; retires nothing.

        The chart stays up on purpose. A pass is a note about the name in
        front of the trader ("liked it, one issue"), and CLAUDE.md's rule is
        that only a veto and a like move the review on. Nothing here mutes,
        suppresses, scores or gates the symbol either - the pass reasons are
        evidence and nothing in the running desk reads them.
        """
        if self._pass_vocabulary is None:
            self._set_status(f"Pass disabled: {self._pass_vocabulary_error}", ok=False)
            return None
        codes = self.selected_pass_codes()
        if not codes:
            self._set_status("Tick at least one reason for passing.", ok=False)
            return None
        # R4 B5: THROUGH `_record`, like every other verb. This built its own
        # field dict and called `record_pass_annotation` directly, which meant it
        # never reached the two lines that stamp `surface` and the scan context -
        # so a pass was the one verb whose row could not say which screen it came
        # from. The sidecar writer is passed IN rather than forked around.
        row = self._record(
            EVENT_PASS,
            writer=record_annotation_with_bars,
            reason_codes=codes,
            vocabulary=self._pass_vocabulary,
            side=self._side,
            note=self.note_input.text(),
            # The bars are read HERE, at the moment of the decision, so what is
            # attached is the chart the trader was actually looking at.
            m5_bars=self.cached_m5_bars(),
        )
        if row is None:
            return None
        self.note_input.clear()
        self.clear_pass_selection()
        attached = row.get("m5_bar_count")
        detail = f"  ({attached} M5 bars attached)" if attached else "  (timestamp only)"
        detail += self._merge_pass_cohort_safely()
        self._set_status(f"PASS {row['symbol']} - {', '.join(codes)}{detail}")
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
    # S1.2 - the chart waits until the trader has finished typing
    # ------------------------------------------------------------------
    def begin_follow_up(self, event_type: str, row: dict) -> None:
        """Hold this chart open and put the cursor in the note field.

        Trader, 2026-09-03: *"when I hit like or not today or anything, it
        should keep the chart up UNTIL I finish typing."*

        Called by the HOST that owns the retire, on the same click that already
        wrote the verdict row - so the evidence is on disk before anything can
        go wrong with the sentence that explains it, exactly as it was before.
        What is deferred is only the queue move.

        A host that does not call this (the snapshot popup's rail, a bare rail
        in a test) never enters the waiting state, so Enter in the note field
        stays an ordinary note there.
        """
        if not isinstance(row, dict):
            return
        self._pending_follow_up = (str(event_type or ""), dict(row))
        self.note_input.clear()
        self.note_input.setPlaceholderText(self.FOLLOW_UP_PLACEHOLDER)
        self.note_input.setFocus()

    def follow_up_pending(self) -> bool:
        """True while a retiring verb is waiting for Enter or Escape."""
        return self._pending_follow_up is not None

    def cancel_follow_up(self) -> None:
        """Drop the waiting state, writing nothing and telling nobody.

        The host uses this when the trader clicks away to another chart: that
        click is a SKIP and it stays one, so the host emits its own retire for
        the chart being left and this must not emit a second one.
        """
        self._pending_follow_up = None
        self.note_input.clear()
        self.note_input.setPlaceholderText(NOTE_PLACEHOLDER)

    def _on_note_return(self) -> None:
        if self._pending_follow_up is not None:
            self._settle_follow_up(write=True)
            return
        self.commit_note()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        if (
            watched is self.note_input
            and self._pending_follow_up is not None
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
        ):
            # Escape discards the half-typed line. The verdict itself already
            # counted - it is on disk and is never rewritten.
            self._settle_follow_up(write=False)
            return True
        return super().eventFilter(watched, event)

    def _settle_follow_up(self, *, write: bool) -> None:
        """Write the optional note, then release the chart. Always releases.

        An evidence store never costs the event it records, and here the event
        is the trader moving on: a note that cannot be written degrades to a
        status line and the chart still advances.
        """
        pending = self._pending_follow_up
        self._pending_follow_up = None
        text = self.note_input.text().strip() if write else ""
        self.note_input.clear()
        self.note_input.setPlaceholderText(NOTE_PLACEHOLDER)
        if pending is None:
            return
        event_type, row = pending
        if text:
            try:
                self._record_follow_up_note(row, text)
            except Exception:
                self._set_status(
                    "Advanced, but the follow-up note was NOT saved.", ok=False
                )
        self.followUpSettled.emit(event_type, row)

    def _record_follow_up_note(self, row: dict, text: str) -> dict | None:
        """One NOTE row naming the verdict it follows.

        `supersedes` is the id the verdict row ALREADY carries (P10 A2's
        lineage key). No second opportunity id is invented - plan.md P5.3/P5.4
        own that - and the row is an ordinary `note`, so nothing that counts
        verdicts starts counting it (`pick_feedback._ANNOTATION_DECISIONS` is
        unchanged). Its ABSENCE on an older note means "not a follow-up".
        """
        event_id = str((row or {}).get("event_id") or "").strip()
        if not event_id:
            return None
        return self._record(
            EVENT_NOTE, note=text, side=self._side, supersedes=event_id
        )

    def _append_follow_up_hint(self) -> None:
        """Say how to move on, on the status line the verb just wrote."""
        if self._pending_follow_up is None:
            return
        self.status_label.setText(self.status_label.text() + self.FOLLOW_UP_HINT)

    # ------------------------------------------------------------------
    def _set_status(self, message: str, *, ok: bool = True) -> None:
        self.status_label.setText(message)
        colour = theme.color("long" if ok else "short")
        self.status_label.setStyleSheet(f"color: {colour};" if message else "")

    def status_text(self) -> str:
        return self.status_label.text()
