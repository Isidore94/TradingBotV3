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

from PySide6.QtCore import Qt, Signal
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
from ui.annotations import setup_claims
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
    record_pass_annotation,
)
from ui.annotations.vocabulary import (
    VocabularyError,
    load_pass_vocabulary,
    load_veto_vocabulary,
)
from ui.widgets.flow_layout import FlowLayout

_REASON_ROLE = Qt.ItemDataRole.UserRole
_CLAIM_ROLE = Qt.ItemDataRole.UserRole

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
        # True only for the duration of a "veto but day-trade it" commit.
        # `captured` fires synchronously from inside commit_veto(), i.e.
        # BEFORE commit_veto_day_trade() can emit its own request, so a host
        # that retires the chart on any veto would retire this one too - and
        # the object the Focus placement needs would already be gone. This is
        # how that host is told to hold the chart for one commit.
        self._veto_keeps_chart = False
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
        self.like_note_input.setPlaceholderText("why (required)")
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
            "the setup. Opens a box for an optional note. Alt+L does the same "
            "thing with no box. Nothing is added to Focus or any watchlist."
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
        self.note_input.setPlaceholderText("freeform note")
        self.note_input.returnPressed.connect(self.commit_note)
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
        gesture now calls `commit_like`, which already carries the identical
        guard for the why.

        The 2026-08-22 rule is therefore untouched - "if I like a chart I
        should always be prompted with why", and a like with no why still
        writes nothing and still holds the chart. What changes is only the case
        where the why is ALREADY typed: the gesture used to send the trader
        back to a field they had just filled in.
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

    def _prompt_for_why(self) -> None:
        self.like_note_input.setFocus()
        self._set_status("This like needs a why - type it, then Enter.")

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
        return row

    def prompt_quick_like(self) -> dict | None:
        """Ask for an optional note, then quick-like. The BUTTON's route.

        Trader, 2026-09-02: *"maybe it can have a pop up with a note I can put in
        similar to what we have in master avwapsetups"* - which is
        `QInputDialog.getMultiLineText`, the same control the setup tracker's
        dislike detail uses, so the gesture is one the trader already knows.

        The note is OPTIONAL: OK with an empty box is a plain quick like, and
        CANCEL records NOTHING. A dialog that wrote a row on cancel would make
        the button unusable for "let me look at this first".

        **Alt+L does not come through here.** A keystroke that stops to ask a
        question is not a one-key verb, and the whole point of the shortcut is
        that it costs nothing. The button is for the times the trader has a
        sentence in mind; the key is for the times they do not.
        """
        if not self._symbol:
            self._set_status("No symbol in focus.", ok=False)
            return None
        from PySide6.QtWidgets import QInputDialog

        note, accepted = QInputDialog.getMultiLineText(
            self,
            f"Like {self._symbol}",
            (
                "Something about this was good.\n\n"
                "Add a note if you want one - it is optional, and you can claim "
                "the setup later with Alt+K. Leave it empty to just record the "
                "like."
            ),
            "",
        )
        if not accepted:
            return None
        return self.commit_quick_like(note=str(note or "").strip())

    def _record_like(self, **fields: Any) -> dict | None:
        """Write a like row, with the M5 sidecar when the chart is an M5 one.

        One writer for both like paths so a claimed like and a quick like can
        never disagree about how a like is stored.
        """
        if not self._symbol:
            self._set_status("No symbol in focus.", ok=False)
            return None
        common = {**self._common_fields(), "side": self._side, **fields}
        try:
            if str(self._timeframe or "").upper() == "M5":
                row = record_annotation_with_bars(
                    EVENT_LIKE_CLAIM, m5_bars=self.cached_m5_bars(), **common
                )
            else:
                row = record_annotation(EVENT_LIKE_CLAIM, **common)
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
        why = self.like_note_input.text().strip()
        if not why:
            # Required, not merely offered. The `dislike` rows are the warning:
            # 31 of the most information-dense strings the trader ever wrote,
            # captured under a field nothing insisted on, and discarded.
            # A like without a why is not a like - the chart stays.
            self._prompt_for_why()
            return None
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
        if not self._symbol:
            self._set_status("No symbol in focus.", ok=False)
            return None
        fields = {
            **self._common_fields(),
            "reason_codes": codes,
            "vocabulary": self._pass_vocabulary,
            "side": self._side,
            "note": self.note_input.text(),
        }
        # The bars are read HERE, at the moment of the decision, so what is
        # attached is the chart the trader was actually looking at.
        fields["m5_bars"] = self.cached_m5_bars()
        try:
            row = record_pass_annotation(**fields)
        except AnnotationError as exc:
            self._set_status(str(exc), ok=False)
            return None
        if row is None:
            self._set_status(
                "NOT SAVED - the annotation log could not be written.", ok=False
            )
            return None
        self.captured.emit(EVENT_PASS, row)
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
    def _set_status(self, message: str, *, ok: bool = True) -> None:
        self.status_label.setText(message)
        colour = theme.color("long" if ok else "short")
        self.status_label.setStyleSheet(f"color: {colour};" if message else "")

    def status_text(self) -> str:
        return self.status_label.text()
