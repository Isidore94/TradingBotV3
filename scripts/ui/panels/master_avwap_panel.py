from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (
    QFileSystemWatcher,
    QItemSelection,
    QThread,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from project_paths import (
    ALERT_REVIEW_EVENTS_FILE,
    MASTER_AVWAP_FOCUS_FILE,
    MASTER_AVWAP_PRIORITY_SETUPS_FILE,
    get_local_setting,
    save_local_setting,
)
from review_events import record_review_event, setup_context_fields
from pick_feedback import reviewed_symbols_today
from market_session import get_default_hourly_scan_schedule, get_default_stop_time_label, get_market_session_window
from ui.annotations import verdicts
from ui.annotations.vocabulary import VocabularyError, load_veto_vocabulary
from ui.models.setup import DEFAULT_SETUP_BUCKET_FILTER_LABELS, SetupRow
from ui.models.setup_table_model import ROW_ROLE, SetupFilterProxyModel, SetupTableModel
from ui.services.data_feed import copy_symbols, load_latest_setup_rows_with_meta
from ui.services.scan_service import ScanService
from ui.timer_utils import SignalCoalescer, start_staggered
from ui.widgets.data_table import DataTable
from ui.widgets.setup_delegate import SetupTableDelegate
from ui.widgets.empty_state import EmptyState
from ui.widgets.setup_detail_view import SetupDetailView


# The segmented bucket selector. Values are sets of RAW bucket keys, so a
# selection can span several buckets - which the old single-label combo box
# could not express at all.
BUCKET_SELECTIONS = {
    "fav_hc_near": ("Fav + HC + Near", "Favorites, high conviction, and near-zone setups"),
    "fav_hc": ("Fav + HC", "Favorites and high conviction only"),
    "all": ("All", "Every bucket, including study and tracking rows"),
}
BUCKET_SELECTION_KEYS = {
    "fav_hc_near": {"favorite_setup", "high_conviction", "near_favorite_zone"},
    "fav_hc": {"favorite_setup", "high_conviction"},
    "all": set(),
}
DEFAULT_BUCKET_SELECTION = "fav_hc_near"
_SHADOW_SECTION_TITLE = "Stretched - shadow would demote (NO LIVE CHANGE)"
_SHADOW_ROW_RE = re.compile(r"^\s{2}(?P<symbol>[A-Z][A-Z0-9._\-]*)\s+(?:LONG|SHORT)\s+")


def _shadow_would_demote_symbols(
    path: Path = MASTER_AVWAP_PRIORITY_SETUPS_FILE,
) -> set[str]:
    """Read the additive R3 shadow section without treating it as a live tier."""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return set()
    active = False
    symbols: set[str] = set()
    for line in lines:
        if line == _SHADOW_SECTION_TITLE:
            active = True
            continue
        if not active:
            continue
        match = _SHADOW_ROW_RE.match(line)
        if match:
            symbols.add(match.group("symbol").upper())
    return symbols


def _apply_swing_quality_shadow_badges(rows: list[SetupRow]) -> None:
    report_symbols = _shadow_would_demote_symbols()
    for row in rows:
        raw = row.raw if isinstance(row.raw, dict) else {}
        if not (bool(raw.get("would_demote")) or row.symbol in report_symbols):
            continue
        badges = raw.setdefault("classification_badges", [])
        if not isinstance(badges, list):
            badges = []
            raw["classification_badges"] = badges
        label = "Stretched? (shadow)"
        if label not in badges:
            badges.append(label)


def _apply_reviewed_today_badges(
    rows: list[SetupRow],
    reviewed_symbols: set[str] | None = None,
) -> None:
    reviewed = reviewed_symbols if reviewed_symbols is not None else reviewed_symbols_today()
    for row in rows:
        raw = row.raw if isinstance(row.raw, dict) else {}
        badges = raw.get("classification_badges")
        badges = [
            str(label)
            for label in (badges if isinstance(badges, list) else [])
            if str(label) != "Reviewed today"
        ]
        if row.symbol.strip().upper() in reviewed:
            badges.append("Reviewed today")
        raw["classification_badges"] = badges

# Columns the compact profile hides. Sector/industry NAMES go, but both RS/RW
# readings stay: they are the group strength the trader actually reads, and in
# the old layout they were among the columns pushed off-screen entirely.
COMPACT_HIDDEN_COLUMNS = {"score", "supports", "sector", "last_trade_date"}

# Explicit widths for the compact profile. These sum to under the desk's
# setups viewport at 1640x980; DataTable.fit_columns' 80px floor cannot.
COMPACT_COLUMN_WIDTHS = {
    "favorite": 34,
    "dislike": 34,
    "symbol": 74,
    # The side cell is a LONG/SHORT pill, not bare text - it elides to "L..."
    # if it is squeezed under about 60px.
    "side": 66,
    "bucket": 96,
    "key_level": 116,
    "expected_r": 58,
    "setup_tags": 120,
    "industry": 130,
    "d1_vs_sector": 84,
    "d1_vs_industry": 88,
}


def _row_context(row: SetupRow) -> str:
    """One-line setup-row summary stored with a verdict for later AI review."""
    parts = [f"bucket={row.bucket_label or row.bucket}"]
    if row.score is not None:
        parts.append(f"score={row.score:.1f}")
    if row.tags_text:
        parts.append(f"tags={row.tags_text}")
    if row.key_level:
        parts.append(f"level={row.key_level}")
    if row.expected_r is not None:
        parts.append(f"expected_r={row.expected_r:.2f}")
    return "; ".join(parts)


class _FamilyRecordWorker(QThread):
    """One pass over the tracker outcomes for the Family Win % column - R4 B3.

    It never raises into Qt. A blank record column is a column that has not been
    measured yet; a swing screen that fails to load because a CSV moved would be
    a far worse trade.
    """

    done = Signal(object)

    def run(self) -> None:  # pragma: no cover - exercised through its seam
        try:
            from setup_docs import family_headline_rows

            records = family_headline_rows()
        except Exception:  # noqa: BLE001 - one column, never the table
            records = {}
        self.done.emit(records)


class MasterAvwapPanel(QWidget):
    setupSelected = Signal(object)
    rowsChanged = Signal(int, int, int)
    statusChanged = Signal(str)

    def __init__(self, focus_service=None, parent=None, *, review_events_path=None) -> None:
        super().__init__(parent)
        self.focus_service = focus_service
        # Swing-side decision log for the review-learning loop: the table's
        # ★/✕ are the trader's actual swing decisions, and a SetupRow carries
        # richer structured context (bucket/family/tags/expected R) than any
        # alert. Gated exactly like pick_feedback: only a real default focus
        # store writes, so test panels stay silent.
        focus_store = getattr(focus_service, "store", None)
        default_store = bool(
            focus_service is not None
            and getattr(focus_store, "uses_default_paths", lambda: False)()
        )
        self._uses_default_feedback_paths = default_store
        self._review_events_path = (
            Path(review_events_path)
            if review_events_path is not None
            else (ALERT_REVIEW_EVENTS_FILE if default_store else None)
        )
        self.scan_service = ScanService(self)
        self.scan_service.started.connect(self._on_scan_started)
        self.scan_service.finished.connect(self._on_scan_finished)
        self.scan_service.failed.connect(self._on_scan_failed)
        self.scheduler_enabled = False
        self.scheduler_active_slot = ""
        self.scheduler_day = ""
        self.scheduler_slots_state: dict[str, str] = {}
        self.scheduler_note = "Hourly scan scheduler is off."
        self.scheduler_covered_slots: list[str] = []
        self.external_scheduler_owner = ""

        self.model = SetupTableModel()
        self.proxy = SetupFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)

        self.table = DataTable()
        self.table.setModel(self.proxy)
        self.delegate = SetupTableDelegate(self.table)
        self.table.setItemDelegate(self.delegate)
        self.table.setShowGrid(False)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._bounce_service = None
        self._chart_watch_host = None
        self.table.clicked.connect(self._on_table_clicked)
        self.table.doubleClicked.connect(self._open_symbol_snapshot_from_double_click)
        self._next_snapshot_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Space),
            self.table,
        )
        self._next_snapshot_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._next_snapshot_shortcut.activated.connect(self._open_next_symbol_snapshot)
        self.table.add_row_action("D1+M5 Snapshot Chart", self._open_symbol_snapshot)
        if self.focus_service is not None:
            self.delegate.set_focus_lookup(self.focus_service.is_focus)
            # The ★ column: click to favorite into Swing Focus / click again to remove.
            self.table.add_row_action(
                "Add to Swing Focus Picks",
                lambda proxy_index: self._add_row_to_focus(proxy_index, "swing"),
            )
            self.table.add_row_action(
                "Add to M5 Focus Picks",
                lambda proxy_index: self._add_row_to_focus(proxy_index, "m5"),
            )
            # COALESCED (2026-08-31). The star column is painted from the
            # focus lookup, so a membership change does have to repaint the
            # table - but a repaint here is a full viewport pass through
            # `SetupTableDelegate` for every visible cell, and the DESK drain
            # that morning adopted 45 picks one at a time. `setup_delegate.py`
            # paint lines were the single hottest stack in the stall log.
            # One repaint per burst says exactly as much as 45 did.
            self._focus_repaint_coalescer = SignalCoalescer(
                lambda: self._repaint_focus_stars(), parent=self
            )
            self.focus_service.focusChanged.connect(
                self._focus_repaint_coalescer.request
            )

        self.empty_state = EmptyState(
            "Run a scan to see setups",
            "Master AVWAP results will appear here as sortable rows with side, score, "
            "bucket, support stack, and D1 sector/industry RS/RW context.",
            "Run Scan",
        )
        self.empty_state.action_button.clicked.connect(self.run_scan)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty_state)
        self.stack.addWidget(self.table)

        # Clicking a setup opens this pane to the right of the table with the
        # family mechanics and the symbol's concrete stop/TP levels.
        self.detail_view = SetupDetailView(self)
        self.detail_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.detail_splitter.addWidget(self.stack)
        self.detail_splitter.addWidget(self.detail_view)
        self.detail_splitter.setStretchFactor(0, 3)
        self.detail_splitter.setStretchFactor(1, 2)
        # Starts collapsed and opens on an explicit row click. Selection alone
        # must not open it: Space-to-advance moves the current index, so a pane
        # tied to selection would be open essentially always and permanently
        # cost the table a third of its width.
        self.detail_splitter.setSizes([1, 0])

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("MutedLabel")
        self.data_as_of_label = QLabel("")
        self.data_as_of_label.setObjectName("MutedLabel")
        self.last_run_label = QLabel("Last run: never")
        self.last_run_label.setObjectName("MutedLabel")
        self.scheduler_status_label = QLabel("")
        self.scheduler_status_label.setObjectName("MutedLabel")
        self.scheduler_status_label.setWordWrap(True)
        self.scheduler_button = QPushButton("Start Scheduler")
        self.scheduler_button.clicked.connect(self.toggle_scheduler)

        # No KPI tiles: app.py already renders "Setups: N | Favorites: N |
        # Near: N" permanently in the status bar from the same rowsChanged
        # signal, so the tiles were a second copy of the same three numbers
        # occupying prime vertical space above the table.
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter symbol, tag, level")
        self.search_input.textChanged.connect(self._apply_filters)

        self.min_score_input = QDoubleSpinBox()
        self.min_score_input.setRange(0.0, 200.0)
        self.min_score_input.setDecimals(1)
        self.min_score_input.setSingleStep(5.0)
        self.min_score_input.setPrefix("Min score ")
        self.min_score_input.valueChanged.connect(self._apply_filters)

        self.side_input = QComboBox()
        self.side_input.addItems(["ALL", "LONG", "SHORT"])
        self.side_input.currentTextChanged.connect(self._apply_filters)

        self.bucket_input = QComboBox()
        self.bucket_input.addItem("ALL")
        self.bucket_input.currentTextChanged.connect(self._apply_filters)

        self.max_dte_input = QSpinBox()
        self.max_dte_input.setRange(0, 365)
        self.max_dte_input.setSpecialValueText("Any DTE")
        self.max_dte_input.setValue(0)
        self.max_dte_input.valueChanged.connect(self._apply_filters)

        self._build_bucket_toggle()
        self._build_overflow_menu()
        self._column_profile = ""
        self._build_layout()
        self.set_column_profile("compact")
        self.set_bucket_selection(
            str(get_local_setting("qt_setups_bucket_filter", DEFAULT_BUCKET_SELECTION)
                or DEFAULT_BUCKET_SELECTION)
        )
        self._configure_report_watcher()
        self.refresh_from_reports(emit_empty=False)
        # QFileSystemWatcher can miss atomic replacements on synced/network
        # drives.  Poll file metadata as a cheap fallback so Auto Pilot scans
        # (which use a separate service) still refresh this page promptly.
        self.report_poll_timer = QTimer(self)
        self.report_poll_timer.setInterval(30_000)
        self.report_poll_timer.timeout.connect(self._poll_report_changes)
        start_staggered(self.report_poll_timer, 43_000)
        self.scheduler_timer = QTimer(self)
        self.scheduler_timer.setInterval(15_000)
        self.scheduler_timer.timeout.connect(self._scheduler_tick)
        self.scheduler_timer.start()
        self._refresh_scheduler_status()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        # Caught up once on the way back in, so a page that was hidden across a
        # scheduler slot shows the right status immediately.
        self._scheduler_tick()

    def _build_layout(self) -> None:
        """One control strip over the table.

        The old layout spent roughly 250px of the desk's most contested column
        on chrome: three KPI tiles that duplicate the permanent status-bar
        counts, a five-button Copy-visible row, a three-line scheduler status
        block, four filter widgets and a section header. All of it still
        exists - it moved into the overflow menu - but none of it earns
        standing height above a table the trader reads all day.
        """
        strip = QHBoxLayout()
        strip.setContentsMargins(0, 0, 0, 0)
        strip.setSpacing(6)
        for button in self.bucket_buttons.values():
            strip.addWidget(button)
        strip.addSpacing(6)
        strip.addWidget(self.search_input, 1)
        strip.addWidget(self.data_as_of_label)
        strip.addWidget(self.overflow_button)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        status_row.addWidget(self.last_run_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(strip)
        layout.addWidget(self.detail_splitter, 1)
        layout.addLayout(status_row)

    # ------------------------------------------------------------------
    def _build_bucket_toggle(self) -> None:
        """Segmented bucket selector replacing the combo box.

        The combo could only ever express ONE bucket label, so the desk's
        headline view - favourites and high-conviction together - was
        unselectable. These map to sets of raw bucket keys instead.
        """
        self.bucket_buttons: dict[str, QPushButton] = {}
        self._bucket_group = QButtonGroup(self)
        self._bucket_group.setExclusive(True)
        for key, (label, tip) in BUCKET_SELECTIONS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.clicked.connect(
                lambda _checked=False, selection=key: self.set_bucket_selection(selection)
            )
            self._bucket_group.addButton(button)
            self.bucket_buttons[key] = button

    def _build_overflow_menu(self) -> None:
        """Everything removed from standing height, one click away."""
        self.overflow_button = QPushButton("⋯")
        self.overflow_button.setToolTip("Scans, scheduler, copy lists and extra filters")
        menu = QMenu(self.overflow_button)
        menu.addAction("Run Scan", self.run_scan)
        menu.addAction("Refresh from reports", self.refresh_from_reports)
        menu.addSeparator()
        self._scheduler_action = menu.addAction("Start Scheduler", self.toggle_scheduler)
        self._scheduler_status_action = menu.addAction("")
        self._scheduler_status_action.setEnabled(False)
        menu.addSeparator()
        copy_menu = menu.addMenu("Copy visible")
        for label, kind in (
            ("Longs", "longs"),
            ("Shorts", "shorts"),
            ("Favorites", "favorites"),
            ("Active", "active"),
            ("Ranked", "ranked"),
        ):
            copy_menu.addAction(
                label, lambda copy_kind=kind: self.copy_list(copy_kind)
            )
        filters_menu = menu.addMenu("More filters")
        for widget, label in (
            (self.side_input, "Side"),
            (self.min_score_input, "Minimum score"),
            (self.max_dte_input, "Max days to earnings"),
            (self.bucket_input, "Exact bucket"),
        ):
            action = QWidgetAction(filters_menu)
            holder = QWidget()
            holder_layout = QHBoxLayout(holder)
            holder_layout.setContentsMargins(8, 2, 8, 2)
            holder_layout.setSpacing(8)
            caption = QLabel(label)
            caption.setObjectName("MutedLabel")
            holder_layout.addWidget(caption)
            holder_layout.addWidget(widget, 1)
            action.setDefaultWidget(holder)
            filters_menu.addAction(action)
        self.overflow_button.setMenu(menu)
        self._overflow_menu = menu

    def set_bucket_selection(self, selection: str) -> None:
        """Filter the table to a named group of buckets, and remember it."""
        selection = selection if selection in BUCKET_SELECTIONS else DEFAULT_BUCKET_SELECTION
        self._bucket_selection = selection
        button = self.bucket_buttons.get(selection)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        save_local_setting("qt_setups_bucket_filter", selection)
        self._apply_filters()

    def _repaint_focus_stars(self) -> None:
        """Repaint the table because Focus membership moved. Presentation only."""
        self.table.viewport().update()

    def flush_pending_refresh(self) -> None:
        """Run an owed coalesced repaint now. The seam the tests drive."""
        coalescer = getattr(self, "_focus_repaint_coalescer", None)
        if coalescer is not None:
            coalescer.flush()

    def _active_bucket_keys(self) -> set[str]:
        """The raw bucket keys the current selection allows.

        Falls back to no filter when the loaded rows carry none of the
        selected buckets, so a report of unbucketed or study-only rows shows
        them rather than an empty table the trader cannot explain.
        """
        keys = BUCKET_SELECTION_KEYS.get(getattr(self, "_bucket_selection", ""), set())
        if not keys:
            return set()
        available = {row.bucket.strip().lower() for row in self.model.rows()}
        return keys if (keys & available) else set()

    def set_column_profile(self, profile: str) -> None:
        """Switch between the compact reading set and every column.

        COLUMNS is never edited - the profile hides columns and pins widths.
        Hiding alone is not enough: DataTable.fit_columns enforces an 80px
        floor per column, so even the 9 compact columns at natural width
        overflow the desk's setups pane at 1640x980. The explicit widths are
        the fix.
        """
        profile = "full" if profile == "full" else "compact"
        if profile == self._column_profile:
            return
        self._column_profile = profile
        header = self.table.horizontalHeader()
        for column, (key, _label) in enumerate(self.model.COLUMNS):
            self.table.setColumnHidden(column, False)
            if profile == "compact" and key in COMPACT_HIDDEN_COLUMNS:
                self.table.setColumnHidden(column, True)
        if profile == "compact":
            header.setStretchLastSection(False)
            for column, (key, _label) in enumerate(self.model.COLUMNS):
                width = COMPACT_COLUMN_WIDTHS.get(key)
                if width:
                    header.resizeSection(column, width)
            # Exp R is appended last in COLUMNS (indices 0/1/2 are pinned click
            # targets), so move it into reading position beside the level.
            self._move_column_after("expected_r", "key_level")
            self._fit_compact_columns()
            header.setStretchLastSection(True)
        else:
            self.table.fit_columns()
            header.resizeSection(0, 36)
            header.resizeSection(1, 36)

    def _fit_compact_columns(self) -> None:
        """Squeeze the compact profile into the viewport, never past it.

        The whole point of the compact profile is that nothing hides behind a
        horizontal scrollbar, so fixed widths are not enough on their own - the
        setups pane is a splitter child and can be any width. Overflow is taken
        out of the elastic text columns down to a readable floor; only if that
        is not enough does a column drop out entirely.
        """
        header = self.table.horizontalHeader()
        keys = [key for key, _label in self.model.COLUMNS]
        viewport = self.table.viewport().width()
        if viewport <= 0:
            return

        def visible_total() -> int:
            return sum(
                header.sectionSize(column)
                for column in range(len(keys))
                if not self.table.isColumnHidden(column)
            )

        overflow = visible_total() - viewport
        if overflow <= 0:
            return
        # Elastic columns, widest-first, with the narrowest width each may take.
        for key, floor in (("setup_tags", 72), ("industry", 84), ("key_level", 88)):
            if overflow <= 0:
                break
            column = keys.index(key)
            if self.table.isColumnHidden(column):
                continue
            current = header.sectionSize(column)
            take = min(overflow, max(0, current - floor))
            if take:
                header.resizeSection(column, current - take)
                overflow -= take
        # Still over: drop columns in reverse value order rather than let a
        # scrollbar hide them silently.
        for key in ("d1_vs_sector", "industry", "setup_tags"):
            if overflow <= 0:
                break
            column = keys.index(key)
            if self.table.isColumnHidden(column):
                continue
            overflow -= header.sectionSize(column)
            self.table.setColumnHidden(column, True)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        if self._column_profile == "compact":
            profile, self._column_profile = self._column_profile, ""
            self.set_column_profile(profile)

    def _move_column_after(self, key: str, after_key: str) -> None:
        header = self.table.horizontalHeader()
        keys = [column_key for column_key, _label in self.model.COLUMNS]
        try:
            logical = keys.index(key)
            anchor = keys.index(after_key)
        except ValueError:
            return
        target = header.visualIndex(anchor)
        current = header.visualIndex(logical)
        if current < 0 or target < 0 or current == target + 1:
            return
        header.moveSection(current, target + 1 if current > target else target)

    def _configure_report_watcher(self) -> None:
        self.watcher = QFileSystemWatcher(self)
        for path in (MASTER_AVWAP_FOCUS_FILE, MASTER_AVWAP_PRIORITY_SETUPS_FILE):
            if Path(path).exists():
                self.watcher.addPath(str(path))
        self.watcher.fileChanged.connect(lambda _path: self.refresh_from_reports(emit_empty=False))
        self._report_signatures = self._current_report_signatures()

    @staticmethod
    def _path_signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _current_report_signatures(self) -> dict[str, tuple[int, int] | None]:
        return {
            str(path): self._path_signature(Path(path))
            for path in (MASTER_AVWAP_FOCUS_FILE, MASTER_AVWAP_PRIORITY_SETUPS_FILE)
        }

    def _poll_report_changes(self) -> None:
        signatures = self._current_report_signatures()
        if signatures != getattr(self, "_report_signatures", {}):
            self.refresh_from_reports(emit_empty=False)

    def run_scan(self) -> None:
        self.scan_service.run_watchlist_scan()

    def toggle_scheduler(self) -> None:
        if self.external_scheduler_owner:
            note = f"{self.external_scheduler_owner} owns scheduled scans while Auto mode is active."
            self._refresh_scheduler_status(note=note)
            self.statusChanged.emit(note)
            return
        self.scheduler_enabled = not self.scheduler_enabled
        note = (
            "Hourly scan scheduler started."
            if self.scheduler_enabled
            else "Hourly scan scheduler stopped."
        )
        self._refresh_scheduler_status(note=note)
        self.statusChanged.emit(note)
        if self.scheduler_enabled:
            self._scheduler_tick()

    def set_external_scheduler_owner(self, owner: str = "") -> None:
        """Make scheduled-scan ownership explicit across the main GUI.

        AutoPilot is the canonical scheduler while Auto mode is active.  The
        Setups-page scheduler remains available for manual-mode days, but the
        two can never both own the hourly slots.
        """
        owner = str(owner or "").strip()
        if owner == self.external_scheduler_owner:
            return
        self.external_scheduler_owner = owner
        if owner:
            self.scheduler_enabled = False
            note = f"{owner} owns hourly scans; Setups-page scheduler disabled."
        else:
            note = "AutoPilot is off; the Setups-page scheduler is available if needed."
        self._refresh_scheduler_status(note=note)

    def _scheduler_slot_datetime(self, slot: str, reference: datetime | None = None) -> datetime:
        now = reference or datetime.now()
        slot_time = datetime.strptime(str(slot).strip(), "%H:%M").time()
        return datetime.combine(now.date(), slot_time)

    def _reset_scheduler_state_for_day(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        today_iso = now.date().isoformat()
        if self.scheduler_day == today_iso and self.scheduler_slots_state:
            return False
        schedule = list(get_default_hourly_scan_schedule(reference=now))
        self.scheduler_day = today_iso
        self.scheduler_slots_state = {slot: "pending" for slot in schedule}
        self.scheduler_note = (
            "Scheduler ready for today's market window."
            if self.scheduler_enabled
            else "Hourly scan scheduler is off."
        )
        return True

    def _scheduler_schedule(self, now: datetime | None = None) -> list[str]:
        self._reset_scheduler_state_for_day(now)
        return list(self.scheduler_slots_state.keys())

    def _due_scheduler_slots(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now()
        due: list[str] = []
        for slot in self._scheduler_schedule(now):
            if self.scheduler_slots_state.get(slot) != "pending":
                continue
            if self._scheduler_slot_datetime(slot, reference=now) <= now:
                due.append(slot)
        return due

    def _next_scheduler_slot(self, now: datetime | None = None) -> str | None:
        now = now or datetime.now()
        for slot in self._scheduler_schedule(now):
            if self.scheduler_slots_state.get(slot) != "pending":
                continue
            if self._scheduler_slot_datetime(slot, reference=now) > now:
                return slot
        return None

    def _refresh_scheduler_status(self, note: str | None = None) -> None:
        now = datetime.now()
        self._reset_scheduler_state_for_day(now)
        if note is not None:
            self.scheduler_note = note

        schedule = self._scheduler_schedule(now)
        next_slot = self._next_scheduler_slot(now)
        stop_at = get_default_stop_time_label(reference=now)
        session = get_market_session_window(reference=now)
        completed = [slot for slot, status in self.scheduler_slots_state.items() if status == "completed"]
        failed = [slot for slot, status in self.scheduler_slots_state.items() if status == "failed"]
        state = "externally owned" if self.external_scheduler_owner else ("running" if self.scheduler_enabled else "stopped")
        active_task = self.scheduler_active_slot or ("manual scan" if self.scan_service.running else "None")
        self.scheduler_button.setEnabled(not bool(self.external_scheduler_owner))
        self.scheduler_button.setText(
            f"Owned by {self.external_scheduler_owner}"
            if self.external_scheduler_owner
            else ("Stop Scheduler" if self.scheduler_enabled else "Start Scheduler")
        )
        self.scheduler_status_label.setText(
            (
                f"Hourly scan scheduler: {state} | "
                f"Market session: {session.session_label} | "
                f"Today's slots: {', '.join(schedule) if schedule else 'None'} | Stop at: {stop_at}\n"
                f"Next slot: {next_slot or 'None'} | Completed: {len(completed)} | "
                f"Failed: {len(failed)} | Active task: {active_task}\n"
                f"Note: {self.scheduler_note}"
            )
        )
        # The three-line label no longer stands above the table; the same state
        # rides the overflow menu (and its tooltip carries the full text).
        self._scheduler_action.setEnabled(not bool(self.external_scheduler_owner))
        self._scheduler_action.setText(self.scheduler_button.text())
        self._scheduler_status_action.setText(
            f"Scheduler {state} · next {next_slot or 'none'} · done {len(completed)}"
        )
        self.overflow_button.setToolTip(self.scheduler_status_label.text())

    def _scheduler_tick(self) -> None:
        # Nothing this tick can do matters while the page is hidden: its only
        # outputs are the scheduler status label and the overflow tooltip, and
        # it cannot start a scan the trader is not looking at without saying so.
        # It is also inert when another process owns scheduled scans. The timer
        # keeps running; `showEvent` refreshes once when the page returns.
        if not self.isVisible() or self.external_scheduler_owner:
            return
        now = datetime.now()
        self._reset_scheduler_state_for_day(now)
        if self.scan_service.running:
            self._refresh_scheduler_status()
            return
        if not self.scheduler_enabled:
            self._refresh_scheduler_status()
            return
        if self.external_scheduler_owner:
            self.scheduler_enabled = False
            self._refresh_scheduler_status(
                note=f"{self.external_scheduler_owner} owns scheduled scans; no duplicate launched."
            )
            return
        stop_dt = self._scheduler_slot_datetime(get_default_stop_time_label(reference=now), reference=now)
        if now >= stop_dt:
            self._refresh_scheduler_status(note=f"Today's scheduler window ended at {stop_dt.strftime('%H:%M')}.")
            return

        due_slots = self._due_scheduler_slots(now)
        if due_slots:
            self._start_scheduled_scan(due_slots[-1], due_slots)
            return

        next_slot = self._next_scheduler_slot(now)
        self._refresh_scheduler_status(
            note=(
                f"Waiting for next hourly slot {next_slot}."
                if next_slot
                else "No pending hourly scan slots remain for today."
            )
        )

    def _start_scheduled_scan(self, trigger_slot: str, covered_slots: list[str]) -> None:
        self.scheduler_active_slot = trigger_slot
        self.scheduler_covered_slots = list(covered_slots)
        coverage = (
            f" (one catch-up scan covering {len(covered_slots)} due slots)"
            if len(covered_slots) > 1
            else ""
        )
        label = f"Running scheduled scan for {trigger_slot}{coverage}..."
        if not self.scan_service.run_watchlist_scan(label, scheduled_slot=trigger_slot):
            rejection = self.scan_service.last_rejection_reason
            if rejection == "scheduled slot already completed":
                for slot in self.scheduler_covered_slots:
                    if slot in self.scheduler_slots_state:
                        self.scheduler_slots_state[slot] = "completed"
                self.scheduler_active_slot = ""
                self.scheduler_covered_slots = []
                self._refresh_scheduler_status(
                    note=f"Scheduled slot {trigger_slot} was already completed; no duplicate scan launched."
                )
                return
            self.scheduler_active_slot = ""
            self.scheduler_covered_slots = []
            self._refresh_scheduler_status(note="Scheduler found a due slot, but another scan is already running.")
            return
        self._refresh_scheduler_status(note=label)

    def _finish_scheduler_run(self, success: bool, error_text: str = "") -> None:
        if not self.scheduler_active_slot:
            return
        trigger_slot = self.scheduler_active_slot
        covered_count = len(self.scheduler_covered_slots)
        for slot in self.scheduler_covered_slots:
            if slot in self.scheduler_slots_state:
                self.scheduler_slots_state[slot] = "completed" if success else "failed"
        self.scheduler_active_slot = ""
        self.scheduler_covered_slots = []
        note = (
            (
                f"Scheduled scan for {trigger_slot} completed; "
                f"one scan covered {covered_count} due slots."
                if covered_count > 1
                else f"Scheduled scan for {trigger_slot} completed."
            )
            if success
            else f"Scheduled scan for {trigger_slot} failed: {error_text}"
        )
        self._refresh_scheduler_status(note=note)

    def _start_family_record_read(self) -> None:
        """The per-family swing record, off the Qt thread - V3 item 1 / R4 B3.

        `setup_docs.family_headline_rows` opens
        `master_avwap_tier_outcomes.csv`, which is why this is a worker and not a
        call inside `set_rows`: the report refresh already runs on the click, and
        a store read there is a stall charged to the trader's own button.

        Started once per report refresh. The column simply reads "-" until the
        answer lands, which is the honest state for a record nobody has measured
        yet.
        """
        worker = getattr(self, "_family_record_worker", None)
        if worker is not None and worker.isRunning():
            return
        worker = _FamilyRecordWorker(self)
        worker.done.connect(self._on_family_records_ready)
        self._family_record_worker = worker
        worker.start()

    def _on_family_records_ready(self, payload: object) -> None:  # pragma: no cover - signal seam
        self.model.set_family_records(payload if isinstance(payload, dict) else {})

    def refresh_from_reports(self, emit_empty: bool = True) -> None:
        self._start_family_record_read()
        meta = load_latest_setup_rows_with_meta()
        rows = meta["rows"]
        _apply_swing_quality_shadow_badges(rows)
        if rows or emit_empty:
            self.set_rows(rows)
            self.status_label.setText("Loaded latest report rows." if rows else "No report rows found.")
            self.statusChanged.emit(self.status_label.text())
        self._apply_data_as_of(meta)
        self._refresh_watcher_paths()
        self._report_signatures = self._current_report_signatures()

    def _apply_data_as_of(self, meta: dict) -> None:
        data_date = meta.get("data_date")
        source = meta.get("source") or ""
        is_stale = bool(meta.get("is_stale"))
        if not data_date:
            self.data_as_of_label.setText("")
        else:
            source_note = " · priority report" if source == "priority_report" else ""
            if is_stale:
                text = f"⚠ Setups as of {data_date} — stale; run an after-close scan to refresh{source_note}"
            else:
                text = f"Setups as of {data_date}{source_note}"
            self.data_as_of_label.setText(text)
        object_name = "CautionLabel" if is_stale else "MutedLabel"
        if self.data_as_of_label.objectName() != object_name:
            self.data_as_of_label.setObjectName(object_name)
            # Re-apply the stylesheet so the objectName-scoped rule takes effect.
            style = self.data_as_of_label.style()
            style.unpolish(self.data_as_of_label)
            style.polish(self.data_as_of_label)

    def set_rows(self, rows: list[SetupRow]) -> None:
        if self._uses_default_feedback_paths:
            _apply_reviewed_today_badges(rows)
        self.model.set_rows(rows)
        self._refresh_bucket_filter(rows)
        self._apply_filters()
        self.stack.setCurrentWidget(self.table if rows else self.empty_state)
        if rows:
            # Rows arrive pre-ranked (conviction bucket, then tracker-led
            # Expected-R). Preserve that order instead of forcing a score sort so
            # the headline ranking is what the trader sees first; column headers
            # remain click-sortable.
            #
            # Re-apply the active profile rather than fit_columns unconditionally:
            # fit_columns' 80px floor is exactly what makes the compact profile
            # overflow, so letting it run here would undo the pinned widths on
            # every report refresh.
            profile = self._column_profile or "compact"
            self._column_profile = ""
            self.set_column_profile(profile)
        self.rowsChanged.emit(
            len(rows),
            sum(1 for row in rows if row.bucket.strip().lower() in {"favorite_setup", "high_conviction"}),
            sum(1 for row in rows if row.bucket.strip().lower() == "near_favorite_zone"),
        )

    def filtered_rows(self) -> list[SetupRow]:
        rows: list[SetupRow] = []
        for proxy_row in range(self.proxy.rowCount()):
            proxy_index = self.proxy.index(proxy_row, 0)
            source_index = self.proxy.mapToSource(proxy_index)
            row = self.model.data(source_index, ROW_ROLE)
            if isinstance(row, SetupRow):
                rows.append(row)
        return rows

    def all_rows(self) -> list[SetupRow]:
        return self.model.rows()

    def copy_list(self, kind: str) -> None:
        rows = self.filtered_rows()
        text = copy_symbols(rows, kind)
        QApplication.clipboard().setText(text)
        self.status_label.setText(f"Copied {kind.lower()} list: {len([s for s in text.split(',') if s.strip()])} symbol(s).")
        self.statusChanged.emit(self.status_label.text())

    def _apply_filters(self) -> None:
        max_dte = self.max_dte_input.value() or None
        self.proxy.set_filters(
            min_score=self.min_score_input.value(),
            side=self.side_input.currentText(),
            bucket=self.bucket_input.currentText(),
            buckets=self._active_bucket_keys(),
            max_dte=max_dte,
            search_text=self.search_input.text(),
        )

    def _refresh_bucket_filter(self, rows: list[SetupRow]) -> None:
        current = self.bucket_input.currentText()
        labels = list(DEFAULT_SETUP_BUCKET_FILTER_LABELS)
        extras = sorted({row.bucket_label for row in rows if row.bucket_label and row.bucket_label not in labels})
        self.bucket_input.blockSignals(True)
        self.bucket_input.clear()
        self.bucket_input.addItem("ALL")
        self.bucket_input.addItems(labels)
        self.bucket_input.addItems(extras)
        index = self.bucket_input.findText(current)
        self.bucket_input.setCurrentIndex(index if index >= 0 else 0)
        self.bucket_input.blockSignals(False)

    def _refresh_watcher_paths(self) -> None:
        watched = set(self.watcher.files())
        for path in (MASTER_AVWAP_FOCUS_FILE, MASTER_AVWAP_PRIORITY_SETUPS_FILE):
            path_text = str(path)
            if Path(path).exists() and path_text not in watched:
                self.watcher.addPath(path_text)

    def _on_scan_started(self, label: str) -> None:
        self.status_label.setText(label)
        self.statusChanged.emit(label)

    def _on_scan_finished(self, _run_result: dict, rows: list[SetupRow], stamp: str) -> None:
        self.set_rows(rows)
        self.last_run_label.setText(f"Last run: {stamp}")
        message = f"Scan complete at {stamp}; loaded {len(rows)} setup row(s)."
        self.status_label.setText(message)
        self.statusChanged.emit(message)
        self._apply_data_as_of(load_latest_setup_rows_with_meta())
        self._refresh_watcher_paths()
        self._finish_scheduler_run(success=True)

    def _on_scan_failed(self, message: str) -> None:
        summary = message.splitlines()[0] if message else "Scan failed."
        self.status_label.setText(f"Error: {summary}")
        self.statusChanged.emit(self.status_label.text())
        self._finish_scheduler_run(success=False, error_text=summary)
        QMessageBox.critical(self, "Master AVWAP Scan Failed", message)

    def set_bounce_service(self, service) -> None:
        """Optional: cached M5 bars for the snapshot popup's lower chart."""
        self._bounce_service = service

    def set_chart_watch_host(self, host) -> None:
        """Optional: the Alert Center panel that owns chart watches and the
        D1 Focus feed. With it, this panel's snapshot popups grow the same
        chart-only actions (D1 Focus pin + New HOD/LOD/VWAP-bounce arming)."""
        self._chart_watch_host = host

    def _open_symbol_snapshot(self, proxy_index) -> None:
        """Row double-click / context action: D1+M5 candle quick look."""
        if not proxy_index.isValid():
            return
        source_index = self.proxy.mapToSource(proxy_index)
        # The ★/✕ cells are click targets of their own; a fast double-click
        # there must not also pop the chart.
        if self.model.COLUMNS[source_index.column()][0] in {"favorite", "dislike"}:
            return
        row = self.model.row_at(source_index.row())
        if row is None or not row.symbol:
            return
        bot = None
        if self._bounce_service is not None:
            try:
                bot = self._bounce_service.current_bot()
            except Exception:
                bot = None
        from ui.widgets.symbol_snapshot_dialog import show_symbol_snapshot

        side = row.side if row.side in {"LONG", "SHORT"} else ""
        show_symbol_snapshot(
            self,
            row.symbol,
            bot=bot,
            side=side,
            watch_host=self._chart_watch_host,
            # The popup's ✕ and its advance-to-next-chart flow route back
            # through this panel, so a chart-by-chart pass over the table
            # never requires touching the table itself.
            review_host=self,
        )

    def _open_symbol_snapshot_from_double_click(self, proxy_index) -> None:
        """Keep the existing row double-click without reopening symbol clicks."""
        if not proxy_index.isValid():
            return
        source_index = self.proxy.mapToSource(proxy_index)
        if self.model.COLUMNS[source_index.column()][0] == "symbol":
            return  # the first single click already opened it
        self._open_symbol_snapshot(proxy_index)

    # ------------------------------------------------------------------
    # Snapshot review flow: the chart popup's ✕ / Add-to-D1-Focus buttons
    # call back here so each decision advances to the next visible chart.
    def snapshot_review_dislike(self, symbol: str) -> bool:
        """Popup ✕: dislike the row (reason prompt) and advance on accept."""
        row = self._visible_row_for_symbol(symbol)
        if row is None or self.focus_service is None:
            return False
        accepted = self._dislike_row(row)
        if accepted:
            self._open_next_symbol_snapshot()
        return accepted

    def snapshot_review_advance(self) -> None:
        """Popup follow-through (e.g. after Add to D1 Focus): next chart."""
        self._open_next_symbol_snapshot()

    def snapshot_review_previous(self) -> None:
        """Popup ◀ Prev: the previous visible row's chart. Records nothing."""
        self._open_next_symbol_snapshot(step=-1)

    def _visible_row_for_symbol(self, symbol: str) -> SetupRow | None:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return None
        for proxy_row in range(self.proxy.rowCount()):
            source = self.proxy.mapToSource(self.proxy.index(proxy_row, 0))
            row = self.model.row_at(source.row())
            if row is not None and row.symbol == symbol:
                return row
        return None

    def _open_next_symbol_snapshot(self, step: int = 1) -> None:
        """Step through visible setup rows and open that snapshot.

        ``step`` is +1 for the next row and -1 for the previous one; both
        wrap, so the walk never dead-ends at either edge of the table.
        """
        row_count = self.proxy.rowCount()
        if row_count <= 0:
            return
        current = self.table.currentIndex()
        step = -1 if int(step) < 0 else 1
        next_row = (current.row() + step) % row_count if current.isValid() else 0
        symbol_column = next(
            column
            for column, (key, _label) in enumerate(self.model.COLUMNS)
            if key == "symbol"
        )
        symbol_index = self.proxy.index(next_row, symbol_column)
        self.table.setCurrentIndex(symbol_index)
        self.table.scrollTo(symbol_index)
        self._open_symbol_snapshot(symbol_index)

    def _on_selection_changed(self, selected: QItemSelection, _deselected: QItemSelection) -> None:
        indexes = selected.indexes()
        if not indexes:
            return
        source_index = self.proxy.mapToSource(indexes[0])
        row = self.model.row_at(source_index.row())
        if row is not None:
            self.setupSelected.emit(row)
            self._show_setup_detail(row)

    def expand_detail_pane(self) -> None:
        """Open the detail pane if the trader has it collapsed."""
        sizes = self.detail_splitter.sizes()
        if len(sizes) == 2 and sizes[1] <= 0:
            total = max(sum(sizes), self.detail_splitter.width() or 600)
            self.detail_splitter.setSizes([int(total * 0.62), int(total * 0.38)])

    def _show_setup_detail(self, row: SetupRow) -> None:
        from setup_docs import resolve_setup_family_from_candidates

        raw = row.raw if isinstance(row.raw, dict) else {}
        signals = raw.get("favorite_signals") or row.setup_tags or []
        # Priority-report rows carry the family as a display label (and often
        # only inside the tags); resolve against the docs registry instead of
        # falling back to the bucket, which is not a setup family.
        family = resolve_setup_family_from_candidates(
            [raw.get("setup_family"), *row.setup_tags]
        )
        self.detail_view.show_setup(
            symbol=row.symbol,
            side=row.side or str(raw.get("side") or "LONG"),
            setup_family=family,
            favorite_signals=signals,
            last_close=raw.get("last_close") or raw.get("previous_close"),
        )

    def _on_table_clicked(self, proxy_index) -> None:
        """Symbol opens its snapshot; ★/✕ retain their existing actions."""
        if not proxy_index.isValid():
            return
        source_index = self.proxy.mapToSource(proxy_index)
        key = self.model.COLUMNS[source_index.column()][0]
        if key == "symbol":
            self._open_symbol_snapshot(proxy_index)
            return
        if key not in {"favorite", "dislike"}:
            # An explicit click on a data cell is the "tell me more" gesture;
            # selection changes alone leave the pane as the trader left it.
            self.expand_detail_pane()
        if self.focus_service is None:
            return
        if key not in {"favorite", "dislike"}:
            return
        row = self.model.row_at(source_index.row())
        if row is None or not row.symbol:
            return
        if key == "dislike":
            self._dislike_row(row)
        elif self.focus_service.is_focus(row.symbol):
            self.focus_service.remove_everywhere(row.symbol, origin="setups", context=_row_context(row))
            self._record_review_event("favorite", row, {"on": False, "origin": "setups"})
            message = f"Unfavorited {row.symbol}: removed from focus picks."
            self.status_label.setText(message)
            self.statusChanged.emit(message)
        else:
            self._add_row_to_focus(proxy_index, "swing")

    def _record_review_event(self, action: str, row: SetupRow, detail: dict) -> None:
        """Swing decision -> alert_review_events.jsonl. Best-effort, never UI-visible."""
        if self._review_events_path is None:
            return
        try:
            record_review_event(
                action,
                symbol=row.symbol,
                side=row.side,
                detail=detail,
                context_fields=setup_context_fields(row),
                path=self._review_events_path,
            )
        except Exception:
            pass

    def _dislike_row(self, row: SetupRow) -> bool:
        """Prompt for a versioned reason code plus optional detail."""
        try:
            vocabulary = load_veto_vocabulary()
        except VocabularyError as exc:
            QMessageBox.warning(self, "Dislike unavailable", str(exc))
            return False
        labels = [f"{reason.hotkey}. {reason.label} [{reason.code}]" for reason in vocabulary.reasons]
        selected, accepted = QInputDialog.getItem(
            self,
            f"Dislike {row.symbol}",
            "Choose the primary reason. The permanent code is counted by the review scoreboard.",
            labels,
            0,
            False,
        )
        if not accepted:
            return False
        selected_index = labels.index(selected) if selected in labels else -1
        if selected_index < 0:
            return False
        reason_choice = vocabulary.reasons[selected_index]
        detail, accepted = QInputDialog.getMultiLineText(
            self,
            f"Dislike {row.symbol} — optional detail",
            (
                f"{reason_choice.label}: add detail for later AI review."
                if not reason_choice.note_required
                else f"{reason_choice.label}: detail is required for this reason."
            ),
        )
        if not accepted:
            return False
        if not reason_choice.accepts(detail):
            QMessageBox.warning(self, "Detail required", "The Other reason requires a note.")
            return False
        self._record_dislike(
            row,
            detail,
            reason_code=reason_choice.code,
            vocab_version=vocabulary.vocab_version,
        )
        return True

    def _record_dislike(
        self,
        row: SetupRow,
        reason: str,
        *,
        reason_code: str = "",
        vocab_version: int | None = None,
    ) -> None:
        detail = {
            "reason": str(reason or "").strip(),
            "origin": "setups",
        }
        if reason_code:
            detail["reason_code"] = str(reason_code).strip().lower()
            detail["reason_codes"] = [detail["reason_code"]]
        if vocab_version is not None:
            detail["vocab_version"] = int(vocab_version)
        self._record_review_event(
            "dislike", row, detail
        )
        # P10 A1. A CODE WAS ALREADY CHOSEN on this path, so no note box opens
        # here - the picklist and its detail field are the quick buttons, and
        # the detail the trader typed is carried in as the note.
        self._record_verdict_annotation(
            row, kind="dislike", reason_code=reason_code, note=reason
        )
        feedback_reason = str(reason or "").strip()
        if reason_code:
            feedback_reason = f"[{str(reason_code).strip().lower()}] {feedback_reason}".strip()
        self.focus_service.record_feedback(
            row.symbol,
            row.side,
            "dislike",
            category=self.focus_service.focus_category(row.symbol) or "swing",
            origin="setups",
            reason=feedback_reason,
            context=_row_context(row),
        )
        message = f"✕ {row.symbol}: dislike logged for AI review."
        if self.focus_service.is_focus(row.symbol):
            self.focus_service.remove_everywhere(row.symbol)
            message = f"✕ {row.symbol}: dislike logged and removed from focus picks."
        self.status_label.setText(message)
        self.statusChanged.emit(message)

    def _record_verdict_annotation(
        self,
        row: SetupRow,
        *,
        kind: str,
        reason_code: str = "",
        note: str = "",
    ) -> None:
        """One annotation row for a star or an X, then the note box - P10 A1/A2.

        The ROW GOES FIRST and the dialog second. Trader: *"sometimes I may not
        want to write a note but the fact I clicked like should be processed by
        the bot eventually."* If the note box came first, Escape would mean the
        click never happened, and that is precisely the case they described.

        The box only opens when NO quick button was used - a coded dislike has
        already said why in the vocabulary the scoreboard counts, and asking
        again would be asking twice for one answer.

        Every failure here is swallowed. This panel's job is the Focus placement
        and the review event, both already done by the time this runs; an
        evidence store never costs the event it records.
        """
        try:
            side = "LONG" if str(row.side).upper().startswith("LONG") else "SHORT"
            context = verdicts.scan_context_from_row(row)
            if kind == "like":
                written = verdicts.record_like(
                    symbol=row.symbol,
                    side=side,
                    surface=verdicts.SURFACE_MASTER_AVWAP,
                    timeframe="D1",
                    scan_context=context,
                )
            else:
                written = verdicts.record_dislike(
                    symbol=row.symbol,
                    side=side,
                    surface=verdicts.SURFACE_MASTER_AVWAP,
                    timeframe="D1",
                    reason_code=reason_code,
                    note=note,
                    scan_context=context,
                )
        except Exception:
            return
        if written is None or reason_code:
            return
        # LAST, and asynchronous - which is not the same as deferred (R4 A6
        # corrected this comment, which claimed the call was on a later turn of
        # the event loop; it is not, it is the last statement of this handler).
        # What matters is that `open()` does not block: a modal opened with
        # `exec()` inside a handler never returns in a headless test, so every
        # existing test that clicks a star or a "Not today" would hang forever
        # rather than fail. The click's own work - the Focus placement, the
        # review event, the status line - is already finished above, so the
        # dialog can never sit between two halves of one action, and A2's rule
        # that the box must not block the 60 s poll holds.
        self._prompt_for_verdict_note(written)

    def _prompt_for_verdict_note(self, written: dict) -> None:
        """The optional note, as a SECOND row - never an edit of the first.

        Trader: *"I SHOULD get a little pop-up that lets me write a note if I am
        not using the quick buttons. same if I like a stock."*

        **WINDOW-MODAL and ASYNCHRONOUS** - R4 A6 corrected this docstring, which
        said MODELESS. `QDialog.open()` shows the dialog window-modal; what it
        does not do is block, which is the property this needs.
        `QInputDialog.getMultiLineText` runs a nested event loop and does not
        return until the trader answers, which is wrong twice over: it stops the
        click's own work finishing, and in a headless test it never returns at
        all - so every existing test that clicks a star would HANG rather than
        fail. `open()` shows the same dialog and returns immediately, delivering
        the answer through a signal.

        **Enter saves and Shift+Enter makes a newline** (R4 A6). The plain-text
        mode that makes the box multi-line also hands Return to the editor, so
        Enter used to insert a newline and the only way to save was the mouse.

        Empty or cancelled writes nothing, and the click already counted.
        """
        from ui.widgets.note_prompt import open_note_prompt

        symbol = str(written.get("symbol") or "")
        verb = "Liked" if written.get("event_type") == "like_claim" else "Disliked"
        # Kept alive until it closes: a dialog with no reference is garbage the
        # moment this method returns.
        self._verdict_note_dialog = open_note_prompt(
            self,
            title=f"{verb} {symbol}",
            label=(
                f"{verb} {symbol}. Add a note if you want one - it is optional, "
                "and the click is already saved either way. Enter saves; "
                "Shift+Enter starts a new line."
            ),
            on_text=lambda text, row=written: self._save_verdict_note(row, text),
        )

    def _save_verdict_note(self, written: dict, note: str) -> None:
        """The note row. Swallowed on failure, like every capture on this panel."""
        try:
            verdicts.record_note_on(written, note)
        except Exception:
            pass

    def _add_row_to_focus(self, proxy_index, category: str = "swing") -> None:
        if self.focus_service is None or not proxy_index.isValid():
            return
        row = self.model.row_at(self.proxy.mapToSource(proxy_index).row())
        bucket = "Swing" if category == "swing" else "M5"
        if row is None or row.side not in {"LONG", "SHORT"}:
            message = "Add to Focus needs a LONG or SHORT row."
        else:
            side = "long" if row.side == "LONG" else "short"
            added = self.focus_service.add(row.symbol, side, category, origin="setups", context=_row_context(row))
            self._record_review_event(
                "favorite",
                row,
                {"on": True, "origin": "setups", "category": category, "added": bool(added)},
            )
            # P10 A1: the star is a LIKE, and the same like as one made on the
            # chart-review rail. Until now it wrote a review event and no graded
            # row at all, so the most considered judgement the trader makes all
            # day - a star on a D1 setup - left no forward record while the same
            # opinion two panels away did. Written AFTER the Focus add and the
            # review event, both of which must not be blocked by it, and its own
            # failure is swallowed inside `_record_verdict_annotation`: an
            # evidence store never costs the event it records.
            self._record_verdict_annotation(row, kind="like")
            message = (
                f"Liked {row.symbol}: added to {bucket} Focus {side}s - its alerts now flag gold in the Alert Center."
                if added
                else f"{row.symbol} already in {bucket} Focus {side}s."
            )
        self.status_label.setText(message)
        self.statusChanged.emit(message)
