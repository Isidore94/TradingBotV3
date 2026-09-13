"""Research > Results - the landing page the trader may read (packet G5.2).

Trader, 2026-09-06 (decision 0016 answer 7, AMENDED that day): *"Research gains
a **Results** landing page the trader may read - Bot setups / My trades and
Swing / Day trading kept as four separate populations, never pooled. It is the
FULL readout; the Desk's 'what is working lately' line remains the primary
surface, and both read ONE evidence snapshot."* Decision 3 of that day: it opens
on **Bot setups x Swing x Recent 20 sessions**, last choice remembered.

This file renders and nothing else. Every number on it comes out of
`research_results.build_results_view`, which copies the evidence snapshot's own
cells and the journal's own totals and computes no statistic of its own. The
page has no "Chart it" button because `working_lately.EvidenceCell` carries no
example symbols and G5 adds no new read to go looking for them.

**Both reads are off the Qt thread** (`ReadWorker`): the snapshot is a JSON file
under `%LOCALAPPDATA%` and the journal is a SQLite query over the whole trade
history, and this is the tab's landing page, so both happen on the way in.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Mapping

from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QButtonGroup,
    QDateEdit,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import evidence_stats
import project_paths
import research_results
import swing_evidence
from d1_environment_join import attach_environment
from ui import theme
from ui.models.tracker_table_model import ROW_ROLE, TrackerTableModel
from ui.read_worker import ReadWorker, join_worker
from ui.services.journal_feed import load_trades
from ui.services.working_lately_service import read_persisted_snapshot
from ui.widgets.data_table import DataTable
from ui.widgets.research_explanation_view import ResearchExplanationView
from ui.widgets.section_header import SectionHeader

#: Where the last choice is remembered. Decision 3 of 2026-09-06.
RESULTS_SELECTION_KEY = "research_results_selection"

#: Bot setups x Swing x Recent 20 sessions - the trader's own default.
DEFAULT_SELECTION = ("bot", "swing", "recent")

POPULATIONS = (("bot", "Bot setups"), ("mine", "My trades"))
HORIZONS = (("swing", "Swing"), ("day", "Day trading"))
WINDOWS = (
    # "Lately" has ONE definition and this button reads it rather than
    # restating it: `evidence_stats.LATELY_SESSIONS`, walked on the exchange
    # calendar. A literal here would be a second definition that agreed with
    # the first only until somebody changed one of them.
    ("recent", f"Recent {evidence_stats.LATELY_SESSIONS} sessions"),
    ("all", "All history"),
    ("custom", "Custom…"),
)

#: Why the three window buttons are dead on the Bot page. Each snapshot cell
#: was measured over the window its own aggregator walked, so choosing another
#: one here would change the heading and not one number under it.
WINDOW_ON_BOT_TOOLTIP = (
    "The evidence snapshot owns its own window, so this control does nothing "
    "to Bot setups. It applies to My trades."
)

#: The reader's MEASURE, in characters of its own font - G3's rule, one page
#: later, and the same two constants as `market_journal_panel`. The first cut
#: of this page set the section text as ONE 1,922-character line across a
#: 3,456 px desk, where the eye loses the start of a line before it finds the
#: end. Running text is readable at about 45-100 characters; the labels keep
#: their place and the TEXT is capped, left-aligned, with the slack at the
#: right.
READER_MEASURE_CHARS = 100
READER_MEASURE_MAX_PX = 1200
READER_MEASURE_MIN_PX = 240


def _reader_measure(metrics: QFontMetrics) -> int:
    """The pixel width of `READER_MEASURE_CHARS` characters, capped and floored."""
    per_char = max(1, int(metrics.averageCharWidth()))
    wanted = min(per_char * READER_MEASURE_CHARS, theme.px(READER_MEASURE_MAX_PX))
    return max(theme.px(READER_MEASURE_MIN_PX), wanted)

#: The shortlist's columns, per population. One table, two shapes: a bot row is
#: a snapshot cell and a My-trades row is a confirmed-tag bucket, and printing
#: them under one set of headers would be the pooling this page refuses.
BOT_COLUMNS = (
    ("side", "Side"),
    ("family", "Family"),
    ("kind", "Measure"),
    # The study rows sit UNDER the live ones and are LABELLED. Without this
    # column a `study` cell in the shortlist is indistinguishable from a live
    # one, which is the packet's "listed under their own label" quietly not
    # happening - an unpromoted idea reading as a result.
    ("namespace", "Population"),
    ("sample", "Sample"),
    ("statistic", "Statistic"),
    ("lower_bound", "Lower bound"),
    ("n_symbols", "Names"),
    ("n_sessions", "Sessions"),
    ("coverage", "Coverage"),
    ("eligibility", "Eligibility"),
)

MINE_COLUMNS = (
    ("label", "Confirmed tag"),
    ("trades", "Trades"),
    ("closed", "Closed"),
    ("wins", "Wins"),
    ("losses", "Losses"),
    ("win_rate", "Win rate"),
    ("net_pnl", "Net P&L"),
)

BAND_TITLES = (
    ("stronger", "Stronger lately"),
    ("weaker", "Weaker lately"),
    ("not_enough", "Not enough evidence"),
)

#: How many rows one card prints per section. The shortlist below holds the
#: rest, and the card's count says so.
CARD_LINES = 3

#: What the detail pane is told it is explaining. An unknown kind lands in
#: `research_explanations`' generic branch, which is the right one: these rows
#: are aggregate measurements and the pane says so.
EXPLANATION_KIND = "research_results"


def _mine_columns(sections) -> tuple[tuple[str, str], ...]:
    """The My-trades headers, with the money column NAMING its currency.

    A number in a column called "Net P&L" is not money until it says what
    money it is; `research_results` reads the currency off `resolve_pnl_key`'s
    own choice, so the header cannot claim a conversion the journal never made.
    A refused total leaves the header bare rather than guessing.
    """
    currency = ""
    for section in sections or ():
        currency = str((section.stats or {}).get("currency") or "")
        if currency:
            break
    if not currency:
        return MINE_COLUMNS
    return tuple(
        (key, f"Net P&L ({currency})" if key == "net_pnl" else label)
        for key, label in MINE_COLUMNS
    )


def _valid_selection(value: Any) -> tuple[str, str, str] | None:
    """A remembered selection, or None if it is not one this page can show."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    population, horizon, window = (str(item) for item in value)
    if population not in dict(POPULATIONS):
        return None
    if horizon not in dict(HORIZONS):
        return None
    if window not in dict(WINDOWS):
        return None
    return population, horizon, window


class _BandCard(QFrame):
    """One band: its title, how many rows it holds, and up to three of them."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionTitle")
        self.count_label = QLabel("-")
        self.count_label.setObjectName("MutedLabel")
        self.body_label = QLabel("")
        self.body_label.setObjectName("SectionSubtitle")
        self.body_label.setWordWrap(True)
        self.body_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.title_label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.body_label, 1)

    def set_sections(self, pairs, *, empty_text: str) -> None:
        """`[(section title, rows)]` - one block per section, never merged.

        Bot x Swing shows two sections whose statistics measure different
        things, so the card stacks them under their own names rather than
        pouring them into one list of three.
        """
        pairs = [(title, list(rows)) for title, rows in pairs]
        total = sum(len(rows) for _title, rows in pairs)
        shown = sum(min(len(rows), CARD_LINES) for _title, rows in pairs)
        # `N of M shown`, not `M shown`. The card prints at most three rows per
        # section by design and the count above them used to state the BAND's
        # size, so a card holding 27 cells said "27 shown" over six printed
        # lines - a false count, and one that hid the fact that the rest of
        # them are in the table below.
        self.count_label.setText(f"{shown} of {total} shown" if total else "nothing to show")
        lines: list[str] = []
        full: list[str] = []
        for title, rows in pairs:
            # A heading with nothing under it reads as a read that failed.
            if len(pairs) > 1 and rows:
                lines.append(f"{title}:")
                full.append(f"{title}:")
            for row in rows[:CARD_LINES]:
                head = row.display.get("headline") or row.line
                if row.reason:
                    head = f"{head} - {row.reason}"
                lines.append(f"• {head}")
                full.append(f"• {row.line}")
        self.body_label.setText("\n".join(lines) if total else empty_text)
        # `EvidenceCell.line()` in full, where it was always meant to live: a
        # dozen clauses per cell is a tooltip, and every field in it is also a
        # column of the shortlist below and a paragraph of the detail pane.
        self.body_label.setToolTip("\n".join(full))


class ResearchResultsPanel(QFrame):
    """The Results page: four populations, one at a time, never pooled."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        self._worker: ReadWorker | None = None
        self._pending = False
        self._pushed_payload: dict[str, Any] | None = None
        self._selection = _valid_selection(
            project_paths.get_local_setting(RESULTS_SELECTION_KEY)
        ) or DEFAULT_SELECTION
        self._shortlist_rows: list[dict[str, Any]] = []

        self.population_buttons: dict[str, QToolButton] = {}
        self.horizon_buttons: dict[str, QToolButton] = {}
        self.window_buttons: dict[str, QToolButton] = {}
        self._population_group = self._make_group(POPULATIONS, self.population_buttons, 0)
        self._horizon_group = self._make_group(HORIZONS, self.horizon_buttons, 1)
        self._window_group = self._make_group(WINDOWS, self.window_buttons, 2)

        self.custom_start = QDateEdit()
        self.custom_end = QDateEdit()
        for edit in (self.custom_start, self.custom_end):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
        today = QDate.currentDate()
        self.custom_start.setDate(today.addDays(-30))
        self.custom_end.setDate(today)
        # Connected AFTER the two initial dates are set. `setDate` emits
        # `dateChanged`, and with a remembered `custom` selection that would
        # have started the page's first read from inside the constructor -
        # before the labels it renders into exist.
        for edit in (self.custom_start, self.custom_end):
            edit.dateChanged.connect(self._on_custom_dates_changed)

        self.freshness_label = QLabel("")
        self.freshness_label.setObjectName("MutedLabel")
        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.section_label = QLabel("")
        self.section_label.setObjectName("SectionSubtitle")
        # Every running-text label on this page reads at one measure, left,
        # with the slack on the right (`_reader_measure`). Wrapped, because a
        # capped line that could not wrap would just elide.
        self._reading_labels = (self.freshness_label, self.section_label, self.status_label)
        for label in self._reading_labels:
            label.setWordWrap(True)
            label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )

        self._cards = {key: _BandCard(title) for key, title in BAND_TITLES}
        self.shortlist = DataTable()
        # Declared order, not the header's: the page states that the shortlist
        # is the section's cells by lower bound with the ineligible ones under
        # them, and a stray click on a header would replace that with a sort the
        # page never explains.
        self.shortlist.setSortingEnabled(False)
        self.shortlist.setShowGrid(False)
        self.shortlist.clicked.connect(self._on_row_clicked)
        self.explanation_view = ResearchExplanationView(self)

        self._build_layout()
        self._apply_selection_to_buttons()
        self.refresh_reader_measure()
        self.refresh()

    def refresh_reader_measure(self) -> None:
        """Recompute the reading measure from the CURRENT font (G3b item 3).

        Polished first: the theme sizes fonts in the stylesheet, so an
        unpolished widget would be measured in the default font and the column
        would not follow the theme it claims to follow.
        """
        for label in self._reading_labels:
            label.ensurePolished()
            label.setMaximumWidth(_reader_measure(label.fontMetrics()))

    # -- construction ------------------------------------------------------

    def _make_group(self, specs, store: dict, index: int) -> QButtonGroup:
        group = QButtonGroup(self)
        group.setExclusive(True)
        for offset, (key, label) in enumerate(specs):
            button = QToolButton()
            button.setObjectName("WeekendViewButton")
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setText(label)
            button.setCheckable(True)
            group.addButton(button, offset)
            store[key] = button
        # `idToggled` FILTERED on `checked`, not `idClicked`: an exclusive
        # group un-checks the old button as it checks the new one, so an
        # unfiltered `toggled` would run the whole read twice for one click -
        # and `idClicked` never fires for a button the page disabled, which is
        # what the three window buttons are on the Bot page. The filter keeps
        # one reaction per change while leaving the state itself drivable.
        group.idToggled.connect(
            lambda _id, checked, position=index: (
                self._on_control_clicked(position) if checked else None
            )
        )
        return group

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(
            SectionHeader(
                "Results",
                "What is working, for whom, over which horizon, and on how much "
                "evidence. Bot setups and My trades, Swing and Day trading, are "
                "four separate populations and are never pooled.",
            )
        )

        controls = QHBoxLayout()
        controls.setSpacing(6)
        for store in (self.population_buttons, self.horizon_buttons, self.window_buttons):
            for button in store.values():
                controls.addWidget(button, 0)
            controls.addSpacing(12)
        self._custom_labels = (QLabel("from"), QLabel("to"))
        controls.addWidget(self._custom_labels[0], 0)
        controls.addWidget(self.custom_start, 0)
        controls.addWidget(self._custom_labels[1], 0)
        controls.addWidget(self.custom_end, 0)
        controls.addStretch(1)
        row = QWidget()
        row.setLayout(controls)
        layout.addWidget(row)

        layout.addWidget(self.freshness_label)
        layout.addWidget(self.section_label)

        cards = QHBoxLayout()
        cards.setSpacing(8)
        for key, _title in BAND_TITLES:
            cards.addWidget(self._cards[key], 1)
        card_row = QWidget()
        card_row.setLayout(cards)
        layout.addWidget(card_row)

        self.detail_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.detail_splitter.addWidget(self.shortlist)
        self.detail_splitter.addWidget(self.explanation_view)
        self.detail_splitter.setStretchFactor(0, 3)
        self.detail_splitter.setStretchFactor(1, 2)
        layout.addWidget(self.detail_splitter, 1)
        layout.addWidget(self.status_label)

    # -- selection ---------------------------------------------------------

    def selection(self) -> tuple[str, str, str]:
        """`(population, horizon, window)` - what the page is showing now."""
        return self._selection

    def _apply_selection_to_buttons(self) -> None:
        population, horizon, window = self._selection
        for store, wanted in (
            (self.population_buttons, population),
            (self.horizon_buttons, horizon),
            (self.window_buttons, window),
        ):
            for key, button in store.items():
                was = button.blockSignals(True)
                button.setChecked(key == wanted)
                button.blockSignals(was)
        self._update_custom_visibility()
        self._update_window_availability()

    def _update_window_availability(self) -> None:
        """The window control is live only where it changes a number.

        On Bot setups every cell was measured over the window its own
        aggregator walked, so the three buttons would move a heading and
        nothing under it. A dead control that says why is honest; a live one
        that does nothing is a lie the page tells once per click.
        """
        live = self._selection[0] == "mine"
        for button in self.window_buttons.values():
            button.setEnabled(live)
            button.setToolTip("" if live else WINDOW_ON_BOT_TOOLTIP)
        for widget in (self.custom_start, self.custom_end, *self._custom_labels):
            widget.setEnabled(live)
            widget.setToolTip("" if live else WINDOW_ON_BOT_TOOLTIP)

    def _update_custom_visibility(self) -> None:
        custom = self._selection[2] == "custom"
        # The two words go with the two fields: a bare "from" and "to" standing
        # beside nothing is a control the trader cannot use.
        for widget in (self.custom_start, self.custom_end, *self._custom_labels):
            widget.setVisible(custom)

    def _selected_key(self, store: dict) -> str:
        for key, button in store.items():
            if button.isChecked():
                return key
        return ""

    def _on_control_clicked(self, _position: int) -> None:
        selection = (
            self._selected_key(self.population_buttons) or self._selection[0],
            self._selected_key(self.horizon_buttons) or self._selection[1],
            self._selected_key(self.window_buttons) or self._selection[2],
        )
        if selection == self._selection:
            # Re-clicking the control already chosen changes nothing, and a
            # page that re-read and re-rendered anyway would take the open
            # explanation down for no reason (G1's rule).
            return
        self._selection = selection
        self._update_custom_visibility()
        self._update_window_availability()
        try:
            project_paths.save_local_setting(RESULTS_SELECTION_KEY, list(selection))
        except Exception:  # noqa: BLE001 - a preference is never worth the page
            logging.debug("Saving the Results selection failed.", exc_info=True)
        # A control change is a CONTEXT change: the pane described a row from a
        # population that is no longer on screen (packet G4's rule).
        self.explanation_view.clear()
        self.refresh()

    def _on_custom_dates_changed(self, _value) -> None:
        if self._selection[2] != "custom":
            return
        self.explanation_view.clear()
        self.refresh()

    # -- the snapshot seam -------------------------------------------------

    def set_working_lately_snapshot(self, payload: Mapping[str, Any] | None) -> None:
        """The service's own reading, pushed in. ONE snapshot, four surfaces."""
        self._pushed_payload = dict(payload) if isinstance(payload, Mapping) else None
        self.refresh()

    def freshness_text(self) -> str:
        """What the page is showing and where every number in it came from."""
        return self.freshness_label.text()

    # -- reading -----------------------------------------------------------

    def refresh(self) -> None:
        """Re-read and re-render. Every read is on a worker, never here."""
        if self._worker is not None and self._worker.isRunning():
            self._pending = True
            return
        selection = self._selection
        window: Any = selection[2]
        if window == "custom":
            window = (
                self.custom_start.date().toString("yyyy-MM-dd"),
                self.custom_end.date().toString("yyyy-MM-dd"),
            )
        payload = self._pushed_payload
        worker = ReadWorker(lambda: _read(selection, window, payload), self)
        worker.finished_with.connect(self._on_loaded)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def _on_failed(self, message: str) -> None:
        self.status_label.setText(f"The Results read failed: {message}")
        self._drain()

    def _on_loaded(self, payload: object) -> None:
        try:
            if isinstance(payload, dict) and payload.get("selection") == self._selection:
                self._render(payload["view"])
        finally:
            self._drain()

    def _drain(self) -> None:
        self._worker = None
        if self._pending:
            self._pending = False
            self.refresh()

    # -- rendering ---------------------------------------------------------

    def _render(self, view: research_results.ResultsView) -> None:
        self.freshness_label.setText(view.freshness_line)
        self.freshness_label.setToolTip(view.freshness_line)
        sections = list(view.sections)
        # ONE short verdict line per kind - the machine's own state and its own
        # reason. The leader, the policy line and the population sentence go in
        # the tooltip: all of it printed on the page came to 1,922 characters
        # of running text, which at 3,456 px is one line nobody reads.
        # A section with NO verdict contributes no line: the environment cut
        # (WS-ENV) names no leader - it is an observational cut - and a blank
        # row above the cards would read as a verdict that failed to print. Its
        # title and sentence still reach the tooltip, where the provenance lives.
        verdicts = [
            (section.verdict_short or section.verdict_line).strip()
            for section in sections
        ]
        self.section_label.setText("\n".join(line for line in verdicts if line))
        self.section_label.setToolTip(
            "\n\n".join(
                "\n".join(
                    part
                    for part in (section.title, section.sentence, section.verdict_line)
                    if str(part or "").strip()
                )
                for section in sections
            )
        )
        head = sections[0] if sections else None
        for key, _title in BAND_TITLES:
            self._cards[key].set_sections(
                [(section.title, getattr(section.bands, key)) for section in sections],
                empty_text=head.sentence if head is not None else research_results.NO_SNAPSHOT,
            )
        self._fill_shortlist(view, sections)
        self._reshow_or_clear_explanation()

    def _fill_shortlist(self, view, sections) -> None:
        columns = (
            _mine_columns(sections) if view.population == "mine" else BOT_COLUMNS
        )
        rows: list[dict[str, Any]] = []
        for section in sections:
            for row in list(section.rows) + list(section.studies):
                entry = dict(row.display)
                entry["_identity"] = (
                    EXPLANATION_KIND,
                    section.key,
                    str(row.values.get("side") or ""),
                    str(row.values.get("family") or row.values.get("label") or ""),
                )
                entry["_section"] = section.key
                entry["_payload"] = {
                    key: value
                    for key, value in row.values.items()
                    if not str(key).startswith("_")
                }
                entry["_payload"]["section"] = section.title
                entry["_payload"]["line"] = row.line
                if row.reason:
                    entry["_payload"]["not_eligible_because"] = row.reason
                # A STUDY is muted the same way an ineligible cell is. The
                # Population column alone told them apart, and a shortlist is
                # read at a glance - an unpromoted idea in the live rows' own
                # weight reads as a result.
                if not row.eligible or str(row.values.get("namespace") or "live") != "live":
                    entry["_muted_row"] = True
                rows.append(entry)
        self._shortlist_rows = rows
        numeric = {
            key
            for key, _label in columns
            if key
            in {"statistic", "lower_bound", "n_symbols", "n_sessions", "trades", "closed", "wins", "losses", "win_rate", "net_pnl"}
        }
        model = TrackerTableModel(columns, rows, numeric_keys=numeric, tooltip_keys={"coverage", "eligibility"})
        self.shortlist.setModel(model)
        self.shortlist.fit_columns()
        if not rows:
            head = sections[0] if sections else None
            self.status_label.setText(head.sentence if head is not None else research_results.NO_SNAPSHOT)
        else:
            # The window the numbers were MEASURED over, which on the Bot page
            # is the snapshot's and not the one the buttons name.
            self.status_label.setText(view.window_sentence or view.window_label)

    def _on_row_clicked(self, index) -> None:
        row = index.data(ROW_ROLE)
        if not isinstance(row, dict):
            return
        self.explanation_view.show_row(
            EXPLANATION_KIND,
            dict(row.get("_payload") or {}),
            identity=tuple(row.get("_identity") or ()),
        )

    def _reshow_or_clear_explanation(self) -> None:
        """After a re-read, re-open the SAME row from the new numbers, or drop it."""
        view = self.explanation_view
        identity = getattr(view, "shown_identity", None)
        if view.isHidden() or not identity:
            return
        for row in self._shortlist_rows:
            if tuple(row.get("_identity") or ()) == tuple(identity):
                view.show_row(
                    EXPLANATION_KIND, dict(row.get("_payload") or {}), identity=tuple(identity)
                )
                return
        view.clear()

    # -- shutdown ----------------------------------------------------------

    def shutdown(self) -> None:
        """Let no read outlive the page it was going to update."""
        self._pending = False
        join_worker(self._worker)
        self._worker = None


def _read_environment_rows(as_of) -> list[dict[str, Any]]:
    """Eligible swing observations, joined to the D1 environment of their SCAN date.

    On the WORKER, never the Qt thread (WS-ENV / WISHLIST 7): a 10 MB outcome
    CSV and a JSONL store, read once per Results redraw for the one selection
    that shows them.

    The eligibility is `swing_evidence`'s own (`POLICY_SCANROW_V1`: one horizon,
    deduplicated on `observation_id`, an explicit `stale_horizon` dropped) and
    the window is `evidence_stats.lately_window` at this readout's own declared
    length - see `research_results.ENVIRONMENT_WINDOW_SESSIONS` for why a cut
    by environment cannot live inside a 20-session window. The join is by
    `scan_date`, so a row is labelled with the tape it was DECIDED in.
    """
    window = evidence_stats.lately_window(
        as_of, sessions=research_results.ENVIRONMENT_WINDOW_SESSIONS
    )
    read = swing_evidence.read_eligible_rows(
        project_paths.MASTER_AVWAP_TIER_OUTCOMES_FILE,
        swing_evidence.POLICY_SCANROW_V1,
        window=window,
    )
    return attach_environment(read.rows, date_field=swing_evidence.POLICY_SCANROW_V1.clock_field)


def _read(selection, window, payload) -> dict[str, Any]:
    """The whole read, on the worker: the snapshot, the journal, and the view.

    The snapshot is read on every pass that has no pushed payload, under either
    population: it is one small per-machine JSON, it is the page's provenance,
    and holding it means switching back to Bot setups redraws instead of
    blanking. The JOURNAL is opened only for a My-trades selection - that one is
    a SQLite query over the whole trade history, and a Bot page has no business
    paying for it. Neither population ever renders the other's numbers.
    """
    population, horizon, _window_key = selection
    trades: list[Any] = []
    snapshot = dict(payload) if isinstance(payload, Mapping) else (read_persisted_snapshot() or {})
    if population == "mine":
        trades = list(load_trades())
    as_of = _as_of(snapshot)
    # The environment cut is opened ONLY where it is shown. A My-trades page
    # has no scan date to join on and a day-trade page is a different
    # population; neither pays for a 10 MB read it will not render.
    environment_rows: list[dict[str, Any]] | None = None
    if population == "bot" and horizon == "swing":
        try:
            environment_rows = _read_environment_rows(as_of)
        except Exception:
            # A readout never costs the page it sits on: the section renders
            # empty and says so, and the champion sections above it are already
            # built from the snapshot.
            logging.exception("Results: the D1 environment cut could not be read.")
            environment_rows = []
    view = research_results.build_results_view(
        population=population,
        horizon=horizon,
        window=window,
        snapshot=snapshot,
        journal_trades=trades,
        as_of=as_of,
        environment_rows=environment_rows,
        # No currency control on this page, so no mode is claimed:
        # `resolve_pnl_key` then sums a single-currency selection, sums the
        # converted column when everything converted, and REFUSES a total over
        # mixed unconverted rows. A default of "CAD" here would be this page
        # asserting a conversion the journal never made.
        currency_mode=None,
    )
    return {"selection": tuple(selection), "view": view}


def _as_of(snapshot: Mapping[str, Any]) -> date:
    """The session the reading is AS OF, so the window is the one it covers."""
    stamp = str((snapshot or {}).get("as_of") or "")[:10]
    try:
        return date.fromisoformat(stamp)
    except ValueError:
        return date.today()
