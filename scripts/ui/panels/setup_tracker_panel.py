from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from project_paths import (
    HUMAN_FOCUS_OUTCOMES_FILE,
    HUMAN_FOCUS_PERFORMANCE_FILE,
    MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE,
    MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE,
    MASTER_AVWAP_SETUP_STATS_FILE,
    MASTER_AVWAP_SETUP_TRACKER_FILE,
    MASTER_AVWAP_TIER_CATCH_RATE_FILE,
    MASTER_AVWAP_TIER_LIST_FILE,
    MASTER_AVWAP_TIER_PERFORMANCE_FILE,
)
from research_explanations import build_plain_english_whats_working
from ui import theme
from ui.timer_utils import SignalCoalescer
from ui.models.tracker_table_model import ROW_ROLE, TrackerSortProxyModel, TrackerTableModel
from ui.services.human_focus_tracker_feed import (
    build_human_focus_comparison_rows,
    load_human_focus_performance_rows,
)
from ui.widgets.data_table import DataTable
from ui.widgets.kpi_tile import KpiTile
from ui.widgets.section_header import SectionHeader
from ui.widgets.setup_detail_view import SetupDetailView


SETUP_TYPE_STATS_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name("master_avwap_setup_type_stats.csv")
RECENT_SETUP_TYPE_STATS_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name("master_avwap_setup_type_recent_stats.csv")
# Phase 0.10 shadow evidence: the AVWAP band challenger beside the champion.
# Read-only, like every other export on this page. Nothing here scores, ranks,
# alerts or gates - `calc_anchored_vwap_bands` is frozen (decision 0008) and the
# challenger is a candidate ADDITIONAL level family, never a swap.
BAND_VARIANT_STATS_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name("master_avwap_band_variant_stats.csv")
SETUP_PLAYBOOKS_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name("master_avwap_setup_playbooks.csv")
SHORT_HORIZON_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name("master_avwap_setup_short_horizon.csv")
SHORT_TERM_MIN_SAMPLES = 6

#: The attribute leaderboard the scanner has written every scan since it was
#: built, and which nothing on this desk has ever shown: ~190 attributes x
#: side x bucket, each with its own edge against the baseline. Until now its
#: only readers were the legacy Tk GUI and the offline tuner.
#:
#: **It is 19.7 MB / 38,617 rows on the live desk**, which is why this one
#: export is read OFF the Qt thread while its ten siblings stay inline. That is
#: not a style choice: the next largest is the playbook file at 5.5 MB and the
#: rest are under 150 KB, so parsing this one on the render path would freeze
#: the desk for seconds on every refresh and tab visit. Phase 0.9 G-P2.3 still
#: owns moving the whole page off-thread; this is the row that cannot wait.
ATTRIBUTE_LEADERBOARD_ROWS_SHOWN = 400


CURRENT_PICK_COLUMNS = (
    ("tier", "Tier"),
    ("symbol", "Symbol"),
    ("side", "Side"),
    ("priority_score", "Score"),
    ("setup_family", "Setup Family"),
    ("favorite_zone", "Favorite Zone"),
    ("current_band_zone", "Current Zone"),
    ("trend_20d", "20D Trend"),
    ("scan_factor_match_count", "Factor Hits"),
    ("scan_factor_matches", "Positive Factors"),
)

SETUP_TYPE_COLUMNS = (
    ("side", "Side"),
    ("priority_bucket", "Bucket"),
    ("setup_family", "Setup Family"),
    ("favorite_zone", "Zone"),
    ("retest_label", "Retest"),
    ("closed_setups", "Closed"),
    ("open_setups", "Open"),
    ("avg_closed_r", "Closed R"),
    ("avg_closed_r_edge", "R Edge"),
    ("target_hit_rate", "Target Hit"),
    ("stop_rate", "Stop"),
    ("score_delta", "Score Delta"),
    ("sample_setups", "Recent Samples"),
)

RECENT_TYPE_COLUMNS = (
    ("status", "Status"),
    ("namespace", "Source"),
    ("side", "Side"),
    ("priority_bucket", "Bucket"),
    ("setup_family", "Setup Family"),
    ("closed_setups", "Closed 30d"),
    ("tracked_setups", "Tracked 30d"),
    ("avg_closed_r", "Closed R"),
    ("avg_closed_r_edge", "R Edge"),
    ("target_hit_rate", "Target Hit"),
    ("stop_rate", "Stop"),
    ("representative_closed_r", "Repr R"),
    ("sample_setups", "Recent Samples"),
)

SHORT_TERM_COLUMNS = (
    ("side", "Side"),
    ("setup_family", "Setup Family"),
    ("samples_2d", "Samples"),
    ("win_rate_2d", "Win @2d"),
    ("avg_r_1d", "R @1d"),
    ("avg_r_2d", "R @2d"),
    ("median_r_2d", "Med R @2d"),
    ("avg_mfe_r_2d", "MFE 2d"),
    ("avg_mae_r_2d", "MAE 2d"),
    ("recent_samples_2d", "N 30d"),
    ("recent_avg_r_2d", "R @2d 30d"),
    ("short_term_score", "Rank"),
    ("sample_setups", "Recent Samples"),
)

PLAYBOOK_COLUMNS = (
    ("side", "Side"),
    ("priority_bucket", "Bucket"),
    ("setup_family", "Setup Family"),
    ("favorite_zone", "Zone"),
    ("stop_reference_label", "Stop"),
    ("profit_take_summary", "Exit Plan"),
    ("closed_setups", "Closed"),
    ("open_setups", "Open"),
    ("robust_closed_r", "Robust R"),
    ("robust_closed_r_edge", "R Edge"),
    ("win_rate_closed", "Win Rate"),
    ("target_hit_rate", "Target Hit"),
    ("ranking_score", "Rank"),
    ("sample_setups", "Recent Samples"),
)

SCAN_FACTOR_COLUMNS = (
    ("horizon_sessions", "Horizon"),
    ("side", "Side"),
    ("factor_label", "Factor"),
    ("value_label", "Value"),
    ("observation_count", "Obs"),
    ("symbol_count", "Symbols"),
    ("win_rate", "Win"),
    ("avg_side_return_pct", "Avg Side %"),
    ("side_return_edge_pct", "Edge %"),
    ("success_score", "Success"),
    ("sample_observations", "Samples"),
)

TIER_PERFORMANCE_COLUMNS = (
    ("horizon_sessions", "Horizon"),
    ("tier", "Tier"),
    ("side", "Side"),
    ("observation_count", "Obs"),
    ("symbol_count", "Symbols"),
    ("win_rate", "Win"),
    ("avg_side_return_pct", "Avg Side %"),
    ("side_return_edge_pct", "Edge %"),
    ("positive_scan_factor_match_rate", "Factor Hit Rate"),
    ("sample_observations", "Samples"),
)

CATCH_RATE_COLUMNS = (
    ("horizon_sessions", "Horizon"),
    ("side", "Side"),
    ("factor_opportunity_count", "Factor Opps"),
    ("factor_winner_count", "Factor Winners"),
    ("caught_winner_count", "Caught Winners"),
    ("caught_winner_rate", "Caught Winners"),
    ("missed_winner_count", "Missed Winners"),
    ("sample_caught_winners", "Caught Samples"),
    ("sample_missed_winners", "Missed Samples"),
)

BAND_VARIANT_COLUMNS = (
    ("setup_family", "Family"),
    ("side", "Side"),
    ("priority_bucket", "Bucket"),
    ("n", "n"),
    ("n_variant", "n Variant"),
    ("n_variant_unmeasured", "n Unmeasured"),
    ("avg_total_r_champion", "Champ R"),
    ("avg_total_r_variant", "Variant R"),
    ("stop_out_rate_champion", "Champ Stop%"),
    ("stop_out_rate_variant", "Variant Stop%"),
    ("target_hit_rate_champion", "Champ Target%"),
    ("target_hit_rate_variant", "Variant Target%"),
    ("mean_stop_distance_atr_champion", "Champ Stop ATR"),
    ("mean_stop_distance_atr_variant", "Variant Stop ATR"),
    ("exit_template_id", "Exit Template"),
)

ATTRIBUTE_LEADERBOARD_COLUMNS = (
    ("attribute_label", "Attribute"),
    ("value_label", "Value"),
    ("side", "Side"),
    ("priority_bucket", "Bucket"),
    ("setup_count", "n"),
    ("closed_tradeable_setup_count", "n Closed"),
    ("meets_n_floor_label", "Floor"),
    ("avg_closed_r", "Closed R"),
    ("avg_closed_r_edge", "Closed R Edge"),
    ("target_hit_rate_edge", "Target% Edge"),
    ("stop_rate_edge", "Stop% Edge"),
    ("sample_setups", "Examples"),
)

ATTRIBUTE_LEADERBOARD_EXPLANATION = (
    "Every attribute the scanner records at entry, graded against the baseline for its "
    "own side and bucket. EDGE columns are this value minus the baseline, so positive "
    "Closed R Edge means setups with this attribute closed better than the rest. "
    "Ranked by Closed R Edge, best first. "
    "ROWS UNDER THE SAMPLE FLOOR ARE GREYED AND SORTED LAST: a one-setup group with a "
    "huge edge is not a weak finding, it is not a finding. Nothing here scores, ranks, "
    "gates or alerts - it is the evidence the tuner reads, shown to the trader who "
    "generated it."
)

HUMAN_PICK_COLUMNS = (
    ("cohort", "Cohort"),
    ("side", "Side"),
    ("horizon_sessions", "Horizon"),
    ("sample_count", "Human N"),
    ("win_rate", "Human Win"),
    ("avg_side_return_pct", "Human Avg %"),
    ("profit_factor", "Human PF"),
    ("bot_sa_sample_count", "Bot S/A N"),
    ("bot_sa_win_rate", "Bot S/A Win"),
    ("bot_sa_avg_side_return_pct", "Bot S/A Avg %"),
    ("avg_side_return_delta_pct", "Delta %"),
)

PERCENT_KEYS = {
    "win_rate",
    "win_rate_closed",
    "win_rate_2d",
    "target_hit_rate",
    "stop_rate",
    "positive_scan_factor_match_rate",
    "caught_winner_rate",
    "caught_opportunity_rate",
    "bot_sa_win_rate",
    "stop_out_rate_champion",
    "stop_out_rate_variant",
    "target_hit_rate_champion",
    "target_hit_rate_variant",
}
SIGNED_KEYS = {
    "avg_total_r_champion",
    "avg_total_r_variant",
    "avg_closed_r",
    "avg_closed_r_edge",
    "representative_closed_r",
    "representative_total_r",
    "robust_closed_r",
    "robust_closed_r_edge",
    "avg_total_r",
    "avg_total_r_edge",
    "avg_r_1d",
    "avg_r_2d",
    "median_r_2d",
    "avg_mfe_r_2d",
    "avg_mae_r_2d",
    "recent_avg_r_2d",
    "short_term_score",
    "side_return_edge_pct",
    "win_rate_edge",
    "success_score",
    "score_delta",
    "avg_side_return_delta_pct",
}
TOOLTIP_KEYS = {
    "sample_setups",
    "sample_observations",
    "sample_caught_winners",
    "sample_missed_winners",
    "scan_factor_matches",
}


class SetupTrackerPanel(QFrame):
    statusChanged = Signal(str)
    #: The attribute leaderboard read lands here, off the worker thread.
    _attributesLoaded = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.current_pick_rows: list[dict[str, Any]] = []
        self.setup_type_rows: list[dict[str, Any]] = []
        self.recent_type_rows: list[dict[str, Any]] = []
        self.short_term_rows: list[dict[str, Any]] = []
        self.playbook_rows: list[dict[str, Any]] = []
        self.scan_factor_rows: list[dict[str, Any]] = []
        self.tier_performance_rows: list[dict[str, Any]] = []
        self.catch_rate_rows: list[dict[str, Any]] = []
        self.human_pick_rows: list[dict[str, Any]] = []
        self.band_variant_rows: list[dict[str, Any]] = []

        self.min_closed_input = QSpinBox()
        self.min_closed_input.setRange(1, 100)
        self.min_closed_input.setValue(5)
        # Held for ~250 ms: the spinbox arrows step one at a time and every
        # step re-ran the WHOLE page - ten CSV parses, ten model resets and ten
        # column fits. Same leading-edge window as everywhere else.
        self._refresh_coalescer = SignalCoalescer(self.refresh, 250, self)
        self.min_closed_input.valueChanged.connect(
            lambda *_args: self._refresh_coalescer.request()
        )

        self.refresh_button = QPushButton("Refresh Tracker")
        self.refresh_button.setObjectName("PrimaryButton")
        self.refresh_button.clicked.connect(self.refresh)

        self.tracked_tile = KpiTile("Tracked Setups", "0")
        self.current_tile = KpiTile("Current S/A Picks", "0", tone="favorite")
        self.best_type_tile = KpiTile("Best Type Edge", "-")
        self.best_short_term_tile = KpiTile("Best 1-2d Setup", "-", tone="favorite")
        self.best_factor_tile = KpiTile("Best Scan Factor", "-")

        self.summary_view = QTextBrowser()
        self.summary_view.setOpenExternalLinks(False)

        self.status_label = QLabel("Tracker exports have not been loaded yet.")
        self.status_label.setObjectName("MutedLabel")
        # The attribute tab's own line: it arrives after the rest of the page,
        # so a shared status label would either lie or overwrite.
        self.attribute_status_label = QLabel("")
        self.attribute_status_label.setObjectName("MutedLabel")
        self.attribute_status_label.setWordWrap(True)
        self.attribute_rows: list[dict[str, Any]] = []
        self._attributes_thread: threading.Thread | None = None

        self.tabs = QTabWidget()
        self.current_table, self.current_model = self._make_table(CURRENT_PICK_COLUMNS)
        self.setup_type_table, self.setup_type_model = self._make_table(SETUP_TYPE_COLUMNS)
        self.recent_type_table, self.recent_type_model = self._make_table(RECENT_TYPE_COLUMNS)
        self.short_term_table, self.short_term_model = self._make_table(SHORT_TERM_COLUMNS)
        self.playbook_table, self.playbook_model = self._make_table(PLAYBOOK_COLUMNS)
        self.scan_factor_table, self.scan_factor_model = self._make_table(SCAN_FACTOR_COLUMNS)
        self.tier_performance_table, self.tier_performance_model = self._make_table(TIER_PERFORMANCE_COLUMNS)
        self.catch_rate_table, self.catch_rate_model = self._make_table(CATCH_RATE_COLUMNS)
        self.human_pick_table, self.human_pick_model = self._make_table(HUMAN_PICK_COLUMNS)
        self.band_variant_table, self.band_variant_model = self._make_table(BAND_VARIANT_COLUMNS)
        self.attribute_table, self.attribute_model = self._make_table(
            ATTRIBUTE_LEADERBOARD_COLUMNS
        )

        self.tabs.addTab(self.current_table, "Current Picks")
        self.tabs.addTab(
            self._make_explained_tab(
                "Which setup families follow through in the FIRST 1-2 SESSIONS after entry (mark-to-market R, "
                "net of costs), independent of the swing outcome. Ranked best-first: the top row is the best "
                "short-term setup right now.",
                self.short_term_table,
            ),
            "Short-Term 1-2d",
        )
        self.tabs.addTab(
            self._make_explained_tab(
                "Compares snapshotted Focus Picks against bot S/A tier picks using the same side-return horizons.",
                self.human_pick_table,
            ),
            "Human Picks",
        )
        self.tabs.addTab(self.setup_type_table, "Setup Types")
        self.tabs.addTab(
            self._make_explained_tab(
                "What's worked in the last 30 days across live setups and measured-only study families. "
                "NEW = family first tracked within 3 weeks (fresh promotions); RISING = outperforming "
                "recently but NOT favorite-bucket yet (upgrade candidates). Both pin to the top.",
                self.recent_type_table,
            ),
            "Last 30 Days",
        )
        self.tabs.addTab(
            self._make_explained_tab(
                "Stop/exit combos per setup type, ranked best-first by robust closed R: "
                "the top row is the best-performing playbook right now.",
                self.playbook_table,
            ),
            "Playbooks",
        )
        self.tabs.addTab(self.scan_factor_table, "Scan Factors")
        self.tabs.addTab(
            self._make_explained_tab(
                "Realized outcome by S/A/B tier: win rate and side-return edge at each forward horizon.",
                self.tier_performance_table,
            ),
            "Tier Performance",
        )
        self.tabs.addTab(
            self._make_explained_tab(
                "Catch rate shows how often positive scan-factor opportunities became tier picks, and which winners were missed.",
                self.catch_rate_table,
            ),
            "Catch Rate",
        )
        self.tabs.addTab(
            self._make_explained_tab(
                "SHADOW EVIDENCE, plan.md Phase 0.10. The AVWAP band challenger - an anchored HLC/3 centre "
                "with a 20-close Bollinger sigma, replicated from OneOption - graded beside the champion's "
                "own protective stop on the SAME exit template. Nothing here scores, ranks or alerts. "
                "A blank cell means nothing was measured for it, never zero; n Unmeasured counts the setups "
                "whose challenger sigma could not be computed. A wider band is stopped out less often BY "
                "CONSTRUCTION when entry sits inside it, so read the stop-distance columns before the rates.",
                self.band_variant_table,
            ),
            "Band Variant",
        )
        self.tabs.addTab(
            self._make_explained_tab(
                ATTRIBUTE_LEADERBOARD_EXPLANATION,
                self.attribute_table,
                footer=self.attribute_status_label,
            ),
            "Attributes",
        )

        # Right-hand setup detail: appears when a row is clicked; for symbol
        # picks it shows the family mechanics plus THIS symbol's stop/target
        # prices from the current anchor bands.
        self.detail_view = SetupDetailView(self, playbook_lookup=self._best_playbook_row)
        self.current_table.clicked.connect(self._on_pick_clicked)
        explained_tables = (
            (self.setup_type_table, "setup_type"),
            (self.recent_type_table, "setup_recent"),
            (self.short_term_table, "setup_short_term"),
            (self.playbook_table, "setup_playbook"),
            (self.scan_factor_table, "setup_scan_factor"),
            (self.tier_performance_table, "setup_tier_performance"),
            (self.catch_rate_table, "setup_catch_rate"),
            (self.human_pick_table, "setup_human_pick"),
        )
        for table, kind in explained_tables:
            table.clicked.connect(
                lambda index, explanation_kind=kind: self._on_research_row_clicked(index, explanation_kind)
            )

        self._attributesLoaded.connect(self._on_attributes_loaded)
        self._build_layout()
        self.refresh()

    def shutdown(self) -> None:
        """Let no read outlive the panel it was going to update."""
        thread = getattr(self, "_attributes_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _build_layout(self) -> None:
        header = SectionHeader(
            "Setup Tracker",
            "Current tier picks plus evidence on which setup families, playbooks, and scan factors are working now.",
        )
        header.add_action(QLabel("Min closed"))
        header.add_action(self.min_closed_input)
        header.add_action(self.refresh_button)

        kpi_row = QHBoxLayout()
        kpi_row.setContentsMargins(0, 0, 0, 0)
        kpi_row.setSpacing(8)
        for tile in (
            self.tracked_tile,
            self.current_tile,
            self.best_type_tile,
            self.best_short_term_tile,
            self.best_factor_tile,
        ):
            kpi_row.addWidget(tile)
        kpi_row.addStretch(1)

        self.detail_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.detail_splitter.addWidget(self.tabs)
        self.detail_splitter.addWidget(self.detail_view)
        self.detail_splitter.setStretchFactor(0, 3)
        self.detail_splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addLayout(kpi_row)
        layout.addWidget(self.summary_view, 1)
        layout.addWidget(self.detail_splitter, 2)
        layout.addWidget(self.status_label)

    def _make_table(
        self,
        columns: tuple[tuple[str, str], ...],
    ) -> tuple[DataTable, TrackerTableModel]:
        numeric_keys = {key for key, _label in columns if _looks_numeric_key(key)}
        model = TrackerTableModel(
            columns,
            percent_keys=PERCENT_KEYS,
            signed_keys=SIGNED_KEYS,
            numeric_keys=numeric_keys,
            tooltip_keys=TOOLTIP_KEYS,
        )
        proxy = TrackerSortProxyModel(self)
        proxy.setSourceModel(model)
        table = DataTable()
        table.setModel(proxy)
        table.setShowGrid(False)
        return table, model

    def _make_explained_tab(self, description: str, table: DataTable, *, footer=None) -> QWidget:
        tab = QWidget()
        label = QLabel(description)
        label.setObjectName("MutedLabel")
        label.setWordWrap(True)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(label)
        layout.addWidget(table, 1)
        if footer is not None:
            layout.addWidget(footer)
        return tab

    def start_attribute_refresh(self) -> None:
        """Read the attribute leaderboard OFF the Qt thread. Single-flight.

        Every other export on this page is parsed inline, and deliberately so -
        Phase 0.9 G-P2.3 owns moving the page as a whole. This one is the
        exception because of its SIZE: 19.7 MB and 38,617 rows on the live
        desk, against 5.5 MB for the next largest and under 150 KB for the
        rest. Parsing it on the render path would freeze the desk for seconds
        on every refresh, spinbox step and tab visit.
        """
        thread = getattr(self, "_attributes_thread", None)
        if thread is not None and thread.is_alive():
            return
        self.attribute_status_label.setText("Reading the attribute leaderboard...")
        self._attributes_thread = threading.Thread(
            target=self._attributes_worker,
            name="tracker-attribute-leaderboard",
            daemon=True,
        )
        self._attributes_thread.start()

    def _attributes_worker(self) -> None:
        payload: dict[str, Any] = {"rows": [], "message": ""}
        try:
            payload["rows"] = _rank_attribute_leaderboard(
                _load_csv_rows_cached(MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE)
            )
        except Exception as exc:  # noqa: BLE001 - a table is never worth a panel
            payload["message"] = f"Attribute leaderboard unreadable: {exc}"
        try:
            self._attributesLoaded.emit(payload)
        except RuntimeError:
            # The panel was deleted while the read was in flight; nothing left
            # to update, so the payload is dropped rather than raised.
            pass

    def _on_attributes_loaded(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        rows = list(data.get("rows") or [])
        self.attribute_rows = rows
        self.attribute_model.set_rows(rows[:ATTRIBUTE_LEADERBOARD_ROWS_SHOWN])
        self.attribute_table.fit_columns()
        message = str(data.get("message") or "")
        if message:
            self.attribute_status_label.setText(message)
            return
        if not rows:
            self.attribute_status_label.setText(
                "No attribute leaderboard on disk yet. The scanner writes it every "
                "scan - this is an absent export, not a scan without attributes."
            )
            return
        under = sum(1 for row in rows if not row.get("_meets_floor"))
        shown = min(len(rows), ATTRIBUTE_LEADERBOARD_ROWS_SHOWN)
        self.attribute_status_label.setText(
            f"{len(rows):,} attribute/value group(s); showing the top {shown:,} by "
            f"closed-R edge. {under:,} are UNDER the reportable-n floor "
            f"(n < {_attribute_floor()}), greyed and sorted last."
        )

    def refresh(self) -> None:
        min_closed = int(self.min_closed_input.value())
        all_setup_type_rows = _load_csv_rows_cached(SETUP_TYPE_STATS_FILE)
        all_playbook_rows = _load_csv_rows_cached(SETUP_PLAYBOOKS_FILE)
        tier_performance_export_rows = _load_csv_rows_cached(MASTER_AVWAP_TIER_PERFORMANCE_FILE)
        self.current_pick_rows = _rank_current_picks(_load_csv_rows_cached(MASTER_AVWAP_TIER_LIST_FILE))
        self.setup_type_rows = _rank_setup_types(all_setup_type_rows, min_closed=min_closed)
        self.recent_type_rows = _rank_recent_types(_load_csv_rows_cached(RECENT_SETUP_TYPE_STATS_FILE))
        self.short_term_rows = _rank_short_term(_load_csv_rows_cached(SHORT_HORIZON_FILE))
        self.playbook_rows = _rank_playbooks(all_playbook_rows, min_closed=min_closed)
        self.scan_factor_rows = _rank_scan_factors(_load_csv_rows_cached(MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE))
        self.tier_performance_rows = _rank_tier_performance(tier_performance_export_rows)
        self.catch_rate_rows = _rank_catch_rates(_load_csv_rows_cached(MASTER_AVWAP_TIER_CATCH_RATE_FILE))
        self.human_pick_rows = build_human_focus_comparison_rows(
            load_human_focus_performance_rows(),
            tier_performance_export_rows,
        )
        # Shadow section. Same inline read as every other export on this page
        # (plan.md Phase 0.9 G-P2.3 owns moving this panel off the Qt thread as
        # a whole; doing it for one section would leave the page half on each).
        self.band_variant_rows = _rank_band_variants(_load_csv_rows_cached(BAND_VARIANT_STATS_FILE))

        self.current_model.set_rows(self.current_pick_rows[:300])
        self.human_pick_model.set_rows(self.human_pick_rows)
        self.setup_type_model.set_rows(self.setup_type_rows[:300])
        self.recent_type_model.set_rows(self.recent_type_rows[:300])
        self.short_term_model.set_rows(self.short_term_rows[:300])
        self.playbook_model.set_rows(self.playbook_rows[:300])
        self.scan_factor_model.set_rows(self.scan_factor_rows[:300])
        self.tier_performance_model.set_rows(self.tier_performance_rows)
        self.catch_rate_model.set_rows(self.catch_rate_rows)
        self.band_variant_model.set_rows(self.band_variant_rows[:300])
        # The attribute leaderboard is read on a worker (19.7 MB live); the
        # table fills when it arrives.
        self.start_attribute_refresh()
        for table in (
            self.current_table,
            self.human_pick_table,
            self.setup_type_table,
            self.recent_type_table,
            self.short_term_table,
            self.playbook_table,
            self.scan_factor_table,
            self.tier_performance_table,
            self.catch_rate_table,
            self.band_variant_table,
        ):
            table.fit_columns()

        tracked_setups = sum(_int(row.get("tracked_setups")) for row in all_setup_type_rows)
        current_sa = sum(1 for row in self.current_pick_rows if str(row.get("tier") or "").upper() in {"S", "A"})
        self.tracked_tile.set_value(str(tracked_setups))
        self.current_tile.set_value(str(current_sa))
        self.best_type_tile.set_value(_best_type_label(self.setup_type_rows))
        self.best_short_term_tile.set_value(_best_short_term_label(self.short_term_rows))
        self.best_factor_tile.set_value(_best_factor_label(self.scan_factor_rows))
        self.summary_view.setHtml(_summary_html(self))

        status = f"Setup tracker refreshed from exports. Last export: {_latest_mtime_text(_export_files())}"
        self.status_label.setText(status)
        self.statusChanged.emit(status)

    # ------------------------------------------------------------------
    # Click-to-detail: family mechanics + this symbol's stop/target prices
    # ------------------------------------------------------------------
    def _on_pick_clicked(self, index) -> None:
        row = index.data(ROW_ROLE)
        if not isinstance(row, dict):
            return
        self.detail_view.show_setup(
            symbol=str(row.get("symbol") or ""),
            side=str(row.get("side") or "LONG"),
            setup_family=str(row.get("setup_family") or ""),
            tier=str(row.get("tier") or ""),
            last_close=row.get("last_close"),
        )

    def _on_family_row_clicked(self, index) -> None:
        row = index.data(ROW_ROLE)
        if not isinstance(row, dict):
            return
        self.detail_view.show_family(str(row.get("setup_family") or ""), side=str(row.get("side") or ""))

    def _on_research_row_clicked(self, index, kind: str) -> None:
        row = index.data(ROW_ROLE)
        if not isinstance(row, dict):
            return
        self.detail_view.show_research_row(kind, row)

    def _best_playbook_row(self, side: str, family: str) -> dict[str, Any] | None:
        side = str(side or "").strip().upper()
        family = str(family or "").strip().lower()
        for row in self.playbook_rows:
            if (
                str(row.get("side") or "").strip().upper() == side
                and str(row.get("setup_family") or "").strip().lower() == family
            ):
                return row
        return None


#: Parsed export rows, keyed by path, with the (mtime_ns, size) they came from.
#: Bounded to one entry per export file - there are ten, and they are rewritten
#: by the scan, not by this page.
_CSV_ROW_CACHE: dict[str, tuple[tuple[int, int], list[dict]]] = {}


def clear_setup_tracker_csv_cache() -> None:
    """Forget every cached export. For tests and for a forced re-read."""
    _CSV_ROW_CACHE.clear()


def _csv_signature(path) -> tuple[int, int] | None:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _load_csv_rows_cached(path) -> list[dict]:
    """`_load_csv_rows`, parsed once per file version.

    One `refresh()` parses ten exports, and the page refreshes on a spinbox
    step, a button and every tab visit - for files a scan rewrites at most a
    few times a day. An unstampable file is not cached, so one that appears
    later is picked up.
    """
    key = str(path)
    signature = _csv_signature(path)
    if signature is not None:
        cached = _CSV_ROW_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
    rows = _load_csv_rows(path)
    if signature is not None:
        _CSV_ROW_CACHE[key] = (signature, rows)
    return rows


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _attribute_floor() -> int:
    """The reportable-n floor, from the ONE place that owns it.

    Computed in the panel only until B1 puts `meets_n_floor` in the CSV itself.
    A failed import falls back to the same constant's value rather than to no
    floor at all: showing an n=1 row ungreyed is the failure this exists to
    prevent.
    """
    try:
        from evidence_stats import MIN_REPORTABLE_N

        return int(MIN_REPORTABLE_N)
    except Exception:  # noqa: BLE001
        return 30


def _rank_attribute_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best closed-R edge first, with sub-floor groups greyed and last.

    The ORDER is the honesty. The export emits categorical, bool and list rows
    at `setup_count=1` with full averages and edges (only numeric bucketing has
    a floor today - B1 fixes the export itself), so sorting purely by edge puts
    a single lucky setup at the top of a 38,617-row table.

    Rows are KEPT, never dropped: this is visibility, not suppression. A group
    under the floor is not a weak finding, it is not a finding, and the label
    says which it is.

    Presentation only - nothing here re-reads the file or writes one.
    """
    floor = _attribute_floor()

    def _edge(row: dict[str, Any]):
        text = str(row.get("avg_closed_r_edge") or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    prepared: list[dict[str, Any]] = []
    for row in rows:
        closed = _float(row.get("closed_tradeable_setup_count"), 0.0)
        # THE FILE'S OWN VERDICT FIRST (R1). B1 made the export state
        # `meets_n_floor` per row; recomputing it here as well means two floors
        # that can disagree, and the one the reader would believe is the greyed
        # row rather than the column. The local comparison stays as the fallback
        # for a file written before B1 - which has no such column, and for which
        # a recomputed floor is the only answer available.
        stated = str(row.get("meets_n_floor") or "").strip()
        meets = stated in {"1", "true", "True"} if stated else closed >= floor
        prepared.append(
            {
                **row,
                "_meets_floor": meets,
                # Read by TrackerTableModel's ForegroundRole: the whole row is
                # muted, not just the count, because it is the EDGE a reader's
                # eye lands on.
                "_muted_row": not meets,
                # A word, not a tick: "n=4" reads as a measurement and "below
                # floor" reads as what it is.
                "meets_n_floor_label": "ok" if meets else f"below floor (<{floor})",
            }
        )
    return sorted(
        prepared,
        key=lambda row: (
            0 if row["_meets_floor"] else 1,
            _edge(row) is None,
            -(_edge(row) or 0.0),
            -_float(row.get("closed_tradeable_setup_count"), 0.0),
            str(row.get("attribute_label") or ""),
            str(row.get("value_label") or ""),
        ),
    )


def _rank_band_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Largest challenger-minus-champion R edge first, biggest n breaking ties.

    A row whose variant R is blank has no edge to rank on and sorts last rather
    than being treated as an edge of zero - the same rule the export itself uses
    when it refuses to print 0.0 for a cell nothing measured. Presentation only:
    this never re-reads the file and never writes one.
    """

    def _edge(row: dict[str, Any]):
        variant = str(row.get("avg_total_r_variant") or "").strip()
        if not variant:
            return None
        try:
            return float(variant) - float(row.get("avg_total_r_champion") or 0.0)
        except (TypeError, ValueError):
            return None

    return sorted(
        rows,
        key=lambda row: (
            _edge(row) is None,
            -(_edge(row) or 0.0),
            -_float(row.get("n"), 0.0),
            str(row.get("setup_family") or ""),
            str(row.get("side") or ""),
        ),
    )


def _rank_current_picks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _tier_rank(row.get("tier")),
            -_float(row.get("priority_score"), 0.0),
            str(row.get("symbol") or ""),
        ),
    )


def _rank_setup_types(rows: list[dict[str, Any]], *, min_closed: int) -> list[dict[str, Any]]:
    filtered = [row for row in rows if _int(row.get("closed_setups")) >= min_closed]
    return sorted(
        filtered,
        key=lambda row: (
            -_float(row.get("score_delta"), 0.0),
            -_float(row.get("ranking_score"), 0.0),
            -_float(row.get("avg_closed_r_edge"), 0.0),
            -_int(row.get("closed_setups")),
        ),
    )


def _rank_recent_types(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # NEW/RISING families with some closed evidence pin to the top (freshly
    # promoted ideas and not-yet-favorite outperformers must never drown under
    # high-sample veterans); below them, most closed evidence first, best
    # realized R next; open-only families fall to the bottom but stay visible.
    return sorted(
        rows,
        key=lambda row: (
            not (str(row.get("status") or "").strip() and _int(row.get("closed_setups")) >= 2),
            -_int(row.get("closed_setups")),
            -_float(row.get("avg_closed_r"), -1e9),
            -_int(row.get("tracked_setups")),
        ),
    )


def _rank_short_term(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Families with enough 2-session samples first, best short-term score next;
    # thin families stay visible at the bottom while evidence accrues.
    return sorted(
        rows,
        key=lambda row: (
            _int(row.get("samples_2d")) < SHORT_TERM_MIN_SAMPLES,
            -_float(row.get("short_term_score"), -1e9),
            -_float(row.get("avg_r_2d"), -1e9),
            -_int(row.get("samples_2d")),
        ),
    )


def _rank_playbooks(rows: list[dict[str, Any]], *, min_closed: int) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if _int(row.get("closed_setups")) >= min_closed
        and str(row.get("experimental") or "").strip().lower() != "true"
    ]
    return sorted(
        filtered,
        key=lambda row: (
            -_float(row.get("ranking_score"), 0.0),
            -_float(row.get("robust_closed_r_edge"), 0.0),
            -_float(row.get("robust_closed_r"), 0.0),
            -_int(row.get("closed_setups")),
        ),
    )


def _rank_scan_factors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [row for row in rows if _int(row.get("observation_count")) >= 8]
    return sorted(
        filtered,
        key=lambda row: (
            -_float(row.get("success_score"), 0.0),
            -_float(row.get("impact_score"), 0.0),
            -_int(row.get("observation_count")),
        ),
    )


def _rank_tier_performance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _int(row.get("horizon_sessions")),
            _tier_rank(row.get("tier")),
            str(row.get("side") or ""),
        ),
    )


def _rank_catch_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (_int(row.get("horizon_sessions")), str(row.get("side") or "")))


def _summary_html(panel: SetupTrackerPanel) -> str:
    body = theme.color("text_primary")
    muted = theme.color("text_secondary")
    long_c = theme.color("long")
    short_c = theme.color("short")
    favorite_c = theme.color("favorite")

    parts = [f"<body style='color:{body}; font-size:9pt'>"]
    plain = build_plain_english_whats_working(
        current_rows=panel.current_pick_rows,
        short_term_rows=panel.short_term_rows,
        recent_rows=panel.recent_type_rows,
        playbook_rows=panel.playbook_rows,
        short_term_min_samples=SHORT_TERM_MIN_SAMPLES,
    )
    parts.append(f"<div style='border:1px solid {favorite_c}; padding:7px; margin-bottom:7px'>")
    parts.append(f"<h3 style='margin:0; color:{favorite_c}'>{_esc(plain['headline'])}</h3><ul>")
    parts.extend(f"<li>{_esc(item)}</li>" for item in plain["bullets"])
    parts.append(f"</ul><div style='color:{muted}'>{_esc(plain['caution'])}</div></div>")
    parts.append(_best_now_banner_html(panel))
    parts.append("<table width='100%' cellspacing='0' cellpadding='4'><tr>")
    parts.append("<td valign='top' width='35%'>")
    parts.append(f"<h3 style='margin:0; color:{favorite_c}'>Current S/A picks</h3>")
    ready_rows = [row for row in panel.current_pick_rows if str(row.get("tier") or "").upper() in {"S", "A"}]
    if ready_rows:
        for row in ready_rows[:10]:
            color = long_c if str(row.get("side") or "").upper() == "LONG" else short_c
            parts.append(
                f"<div><b style='color:{favorite_c}'>{_esc(row.get('tier'))}</b> "
                f"<b>{_esc(row.get('symbol'))}</b> "
                f"<span style='color:{color}'>{_esc(row.get('side'))}</span> "
                f"{_fmt(row.get('priority_score'))} - {_esc(row.get('setup_family'))}</div>"
            )
            matches = str(row.get("scan_factor_matches") or "").strip()
            if matches:
                parts.append(f"<div style='color:{muted}; margin-left:14px'>{_esc(_shorten(matches, 140))}</div>")
    else:
        parts.append(f"<div style='color:{muted}'>No current stock clears the S/A quality gate.</div>")
    parts.append("</td>")

    parts.append("<td valign='top' width='32%'>")
    parts.append(f"<h3 style='margin:0; color:{long_c}'>Setup types working</h3>")
    for row in panel.setup_type_rows[:8]:
        edge = _float(row.get("avg_closed_r_edge"), 0.0)
        edge_color = long_c if edge >= 0 else short_c
        parts.append(
            f"<div><b>{_esc(row.get('side'))}</b> {_esc(row.get('setup_family'))} "
            f"<span style='color:{muted}'>closed {_int(row.get('closed_setups'))}</span> "
            f"<span style='color:{edge_color}'>edge {_signed(edge)}R</span> "
            f"delta {_signed(_float(row.get('score_delta'), 0.0), decimals=0)}</div>"
        )
    if not panel.setup_type_rows:
        parts.append(f"<div style='color:{muted}'>Not enough closed setup-type samples at this threshold.</div>")
    parts.append("</td>")

    parts.append("<td valign='top' width='33%'>")
    parts.append(f"<h3 style='margin:0; color:{long_c}'>Best playbooks</h3>")
    for row in panel.playbook_rows[:6]:
        parts.append(
            f"<div><b>{_esc(row.get('setup_family'))}</b> "
            f"{_esc(row.get('stop_reference_label'))} -> {_esc(_shorten(row.get('profit_take_summary'), 48))} "
            f"<span style='color:{long_c}'>{_signed(_float(row.get('robust_closed_r'), 0.0))}R</span></div>"
        )
    if not panel.playbook_rows:
        parts.append(f"<div style='color:{muted}'>No playbook rows cleared the min closed filter.</div>")
    parts.append("</td></tr></table>")

    parts.append(f"<h3 style='margin:8px 0 4px 0; color:{muted}'>Scan factors and tier quality</h3>")
    parts.append("<table width='100%' cellspacing='0' cellpadding='4'><tr>")
    parts.append("<td valign='top' width='50%'>")
    for row in panel.scan_factor_rows[:6]:
        parts.append(
            f"<div><b>{_esc(row.get('side'))} {_esc(row.get('horizon_sessions'))}d</b> "
            f"{_esc(row.get('factor_label'))} = {_esc(row.get('value_label'))} "
            f"<span style='color:{long_c}'>avg {_signed(_float(row.get('avg_side_return_pct'), 0.0))}%</span></div>"
        )
    parts.append("</td><td valign='top' width='50%'>")
    for row in sorted(panel.tier_performance_rows, key=lambda item: -_float(item.get("side_return_edge_pct"), 0.0))[:6]:
        parts.append(
            f"<div><b>{_esc(row.get('tier'))} {_esc(row.get('side'))} {_esc(row.get('horizon_sessions'))}d</b> "
            f"win {_pct(row.get('win_rate'))}, edge {_signed(_float(row.get('side_return_edge_pct'), 0.0))}% "
            f"<span style='color:{muted}'>n={_int(row.get('observation_count'))}</span></div>"
        )
    parts.append("</td></tr></table>")
    parts.append(f"<p style='color:{muted}'>Tracker source: {_esc(str(MASTER_AVWAP_SETUP_TRACKER_FILE))}</p>")
    parts.append("</body>")
    return "".join(parts)


def _best_now_banner_html(panel: SetupTrackerPanel) -> str:
    """One unmissable line per horizon: the best performing setup right now —
    swing (recent 30d realized R) and short-term (1-2 session follow-through)."""
    muted = theme.color("text_secondary")
    favorite_c = theme.color("favorite")
    long_c = theme.color("long")
    short_c = theme.color("short")

    def _side_color(side: Any) -> str:
        return long_c if str(side or "").upper() == "LONG" else short_c

    swing_row = next(
        (
            row
            for row in sorted(panel.recent_type_rows, key=lambda r: -_float(r.get("avg_closed_r"), -1e9))
            if _int(row.get("closed_setups")) >= 3 and _float(row.get("avg_closed_r")) is not None
        ),
        None,
    )
    short_row = next(
        (
            row
            for row in panel.short_term_rows
            if _int(row.get("samples_2d")) >= SHORT_TERM_MIN_SAMPLES and _float(row.get("avg_r_2d")) is not None
        ),
        None,
    )

    parts = [
        f"<div style='border:1px solid {favorite_c}; padding:6px; margin-bottom:6px'>",
        f"<b style='color:{favorite_c}; font-size:10pt'>BEST PERFORMING RIGHT NOW</b>",
    ]
    if short_row is not None:
        parts.append(
            f"<div><b>Short-term (1-2d):</b> "
            f"<span style='color:{_side_color(short_row.get('side'))}'><b>{_esc(short_row.get('side'))}</b></span> "
            f"<b>{_esc(short_row.get('setup_family'))}</b> "
            f"{_signed(_float(short_row.get('avg_r_2d')))}R@2d, win {_pct(short_row.get('win_rate_2d'))} "
            f"<span style='color:{muted}'>(n={_int(short_row.get('samples_2d'))}, "
            f"last 30d {_signed(_float(short_row.get('recent_avg_r_2d')))}R)</span></div>"
        )
    else:
        parts.append(
            f"<div style='color:{muted}'><b>Short-term (1-2d):</b> not enough 2-session samples yet "
            f"(accrues automatically each scan).</div>"
        )
    if swing_row is not None:
        parts.append(
            f"<div><b>Swing (30d realized):</b> "
            f"<span style='color:{_side_color(swing_row.get('side'))}'><b>{_esc(swing_row.get('side'))}</b></span> "
            f"<b>{_esc(swing_row.get('setup_family'))}</b> "
            f"{_signed(_float(swing_row.get('avg_closed_r')))}R closed, target hit {_pct(swing_row.get('target_hit_rate'))} "
            f"<span style='color:{muted}'>(closed {_int(swing_row.get('closed_setups'))})</span></div>"
        )
    else:
        parts.append(
            f"<div style='color:{muted}'><b>Swing (30d realized):</b> not enough closed setups in the last 30 days.</div>"
        )

    # Freshly promoted families + not-yet-favorite outperformers: the upgrade
    # candidates the trader asked to see without digging through the tab.
    highlighted = sorted(
        (
            row
            for row in panel.recent_type_rows
            if str(row.get("status") or "").strip() and _int(row.get("closed_setups")) >= 2
        ),
        key=lambda row: -_float(row.get("avg_closed_r"), -1e9),
    )
    if highlighted:
        chips = []
        for row in highlighted[:3]:
            chips.append(
                f"<span style='color:{_side_color(row.get('side'))}'><b>{_esc(row.get('side'))}</b></span> "
                f"<b>{_esc(row.get('setup_family'))}</b> {_signed(_float(row.get('avg_closed_r')))}R "
                f"<span style='color:{muted}'>({_esc(row.get('status'))}, closed {_int(row.get('closed_setups'))})</span>"
            )
        parts.append(f"<div><b>New &amp; rising (not favorites yet):</b> {' &middot; '.join(chips)}</div>")
    parts.append("</div>")
    return "".join(parts)


def _best_type_label(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "-"
    row = rows[0]
    delta = _float(row.get("score_delta"), 0.0)
    return f"{_esc(row.get('side'))} {delta:+.0f}"


def _best_short_term_label(rows: list[dict[str, Any]]) -> str:
    qualified = [row for row in rows if _int(row.get("samples_2d")) >= SHORT_TERM_MIN_SAMPLES]
    if not qualified:
        return "-"
    row = qualified[0]
    avg_r_2d = _float(row.get("avg_r_2d"))
    r_text = f" {avg_r_2d:+.2f}R" if avg_r_2d is not None else ""
    return f"{_esc(row.get('side'))}{r_text}@2d"


def _best_factor_label(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "-"
    row = rows[0]
    return f"{_esc(row.get('horizon_sessions'))}d {_esc(row.get('side'))}"


def _export_files() -> list[Path]:
    return [
        SETUP_TYPE_STATS_FILE,
        RECENT_SETUP_TYPE_STATS_FILE,
        BAND_VARIANT_STATS_FILE,
        SETUP_PLAYBOOKS_FILE,
        SHORT_HORIZON_FILE,
        MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE,
        MASTER_AVWAP_TIER_LIST_FILE,
        MASTER_AVWAP_TIER_PERFORMANCE_FILE,
        MASTER_AVWAP_TIER_CATCH_RATE_FILE,
        HUMAN_FOCUS_PERFORMANCE_FILE,
        HUMAN_FOCUS_OUTCOMES_FILE,
    ]


def _latest_mtime_text(paths: list[Path]) -> str:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return "never"
    latest = max(path.stat().st_mtime for path in existing)
    return datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S")


def _tier_rank(value: Any) -> int:
    return {"S": 0, "A": 1, "B": 2, "C": 3}.get(str(value or "").upper(), 9)


def _looks_numeric_key(key: str) -> bool:
    return (
        key.endswith("_count")
        or key.endswith("_setups")
        or key.endswith("_score")
        or key.endswith("_rate")
        or key.endswith("_pct")
        or key.endswith("_r")
        or key.endswith("_edge")
        or key.endswith("_1d")
        or key.endswith("_2d")
        or key in {"priority_score", "ranking_score", "horizon_sessions", "symbol_count", "score_delta"}
    )


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fmt(value: Any) -> str:
    numeric = _float(value)
    return "" if numeric is None else f"{numeric:.0f}"


def _signed(value: float | None, *, decimals: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:+.{decimals}f}"


def _pct(value: Any) -> str:
    numeric = _float(value)
    return "" if numeric is None else f"{numeric * 100:.1f}%"


def _shorten(value: Any, limit: int = 120) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _esc(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
