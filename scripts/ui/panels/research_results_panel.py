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

import project_paths
import research_results
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
    ("recent", "Recent 20 sessions"),
    ("all", "All history"),
    ("custom", "Custom…"),
)

#: The shortlist's columns, per population. One table, two shapes: a bot row is
#: a snapshot cell and a My-trades row is a confirmed-tag bucket, and printing
#: them under one set of headers would be the pooling this page refuses.
BOT_COLUMNS = (
    ("side", "Side"),
    ("family", "Family"),
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

#: What the detail pane is told it is explaining. An unknown kind lands in
#: `research_explanations`' generic branch, which is the right one: these rows
#: are aggregate measurements and the pane says so.
EXPLANATION_KIND = "research_results"


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
        self.count_label.setText(f"{total} shown" if total else "nothing to show")
        lines: list[str] = []
        full: list[str] = []
        for title, rows in pairs:
            if len(pairs) > 1:
                lines.append(f"{title}:")
                full.append(f"{title}:")
            for row in rows[:3]:
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
        self.freshness_label.setWordWrap(True)
        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        self.section_label = QLabel("")
        self.section_label.setObjectName("SectionSubtitle")
        self.section_label.setWordWrap(True)

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
        self.refresh()

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
        # `idClicked` and not `toggled`: an exclusive group un-checks the old
        # button as it checks the new one, so a `toggled` connection would run
        # the whole read twice for one click.
        group.idClicked.connect(lambda _id, position=index: self._on_control_clicked(position))
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
        sections = list(view.sections)
        self.section_label.setText(
            "\n".join(
                f"{section.title} - {section.sentence} [{section.verdict_line}]"
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
        columns = MINE_COLUMNS if view.population == "mine" else BOT_COLUMNS
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
                if not row.eligible:
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
            self.status_label.setText(view.window_label)

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
    view = research_results.build_results_view(
        population=population,
        horizon=horizon,
        window=window,
        snapshot=snapshot,
        journal_trades=trades,
        as_of=as_of,
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
