from __future__ import annotations

import csv
import hashlib
import logging
import threading
from datetime import datetime, timezone
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
from ui.read_worker import ReadWorker, join_worker
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
# Packet M5 (2026-09-05). Three populations the tracker has graded for months
# and shown nobody.
#
# CONTROLS are setups the scan REJECTED - the holdout that says whether the gate
# is throwing away edge. STUDIES are ideas that have never been promoted and
# touch no score. EXIT FRAMEWORKS puts the April `comparison_apr2026` templates
# beside the baseline ones on the same setups, which is the comparison those
# 91,674 scenario rows were written for and never read back.
#
# Read-only, like every other export on this page. Nothing here scores, ranks,
# gates or alerts, and a row from any of the three must never be read as a pick
# - which is why each tab carries a POPULATION SENTENCE above its table.
CONTROL_DISCOVERY_STATS_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name(
    "master_avwap_control_discovery.csv"
)
STUDY_DISCOVERY_STATS_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name(
    "master_avwap_study_discovery.csv"
)
EXIT_FRAMEWORK_STATS_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name(
    "master_avwap_exit_framework_stats.csv"
)

#: The ten-row floor Weekend Prep's tables use (R4 A18), applied to the three
#: tabs this packet adds. 260 px is ten rows plus a header. A separate constant
#: rather than an import from that panel: one number is cheaper than a
#: cross-panel dependency, and the reason is recorded in both places.
TABLE_TEN_ROWS_PX = 260
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
    # WIN RATE LEADS here too (V3 item 1, the owed seam, closed by ST2.2). It is
    # this row's OWN record at this row's own grain - `n_wins` / `n_losses` off
    # the export - with `n` and the ONE Wilson lower bound in the cell, and the
    # table sorts by that bound inside each side. Target Hit and Stop stay
    # beside it under their own names: they answer different questions and were
    # never a win rate, which is why this tab had none for so long.
    ("win_rate_headline", "Win %"),
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
    # WIN RATE LEADS (V3 item 1, decision 0016 answer 3), wired by R4 B3. It is
    # the first STATISTIC on the row - the five columns before it are the row's
    # identity, not a measurement - and it carries its Wilson lower bound and n
    # in one cell, `62% (>=52%, n=90)`, through `swing_headline.format_win_rate`.
    # Closed R stays right beside it and is never replaced: the two answer
    # different questions and dropping either is how a number starts lying.
    # ST2.1: the heading now says WHICH win rate. The cell is the row's own
    # integer counts; `win_rate_closed` beside it is the RECENCY-WEIGHTED mean
    # the scorer reads, kept on the table under its own name rather than
    # deleted, because it is a real number that answers a different question.
    # Reading one as the other is what printed "25% (n=4)" on a family that
    # went 2-2.
    ("win_rate_headline", "Win % (unweighted)"),
    ("win_rate_closed", "Win % (recency-weighted)"),
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

#: Packet M5.2. Win rate FIRST with its lower bound and its n beside it, then
#: mean R - `CLAUDE.md`'s headline rule for every trader-facing swing surface.
#: The window is a column because the export carries two blocks (all history and
#: `lately`), and a table that mixed them without saying so would double-count
#: every family.
DISCOVERY_COLUMNS = (
    ("window", "Window"),
    ("row_kind", "Kind"),
    ("cohort", "Cohort"),
    ("side", "Side"),
    ("setup_family", "Family"),
    ("win_rate", "Win %"),
    ("win_rate_lb", "Win % (low)"),
    ("n", "n"),
    ("wins", "Wins"),
    ("losses", "Losses"),
    ("avg_closed_r", "Avg R"),
    ("n_expired_unmeasured", "Expired"),
    ("flag", "Flag"),
)

#: Packet M5.3. `framework_family` and `experimental` lead, because the first
#: question about a row here is which framework it belongs to and whether it
#: ever happened.
EXIT_FRAMEWORK_COLUMNS = (
    ("framework_family", "Framework"),
    ("experimental", "Experimental"),
    ("exit_template_id", "Exit Template"),
    ("side", "Side"),
    ("priority_bucket", "Bucket"),
    ("win_rate", "Win %"),
    ("win_rate_lb", "Win % (low)"),
    ("n", "n"),
    ("n_closed", "n Closed"),
    ("avg_closed_r", "Avg R"),
    ("stop_out_rate", "Stop%"),
    ("target_hit_rate", "Target%"),
    ("n_expired_unmeasured", "Expired"),
    # Reviewer blocker 1: the column that explains a smaller denominator. A
    # template with `blocked_stop_rules` skips scenarios BY DEFINITION, and
    # `n + Filtered` is what reconciles to the baseline's n.
    ("n_filtered_by_experiment", "Filtered"),
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
    # M5: the Controls / Studies / Exit frameworks tabs. `win_rate` is already
    # here and covers all three.
    "win_rate_lb",
    "stop_out_rate",
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
    #: Emitted on the Qt thread once a refresh's rows have been APPLIED (G7.2).
    #: `refresh()` returns as soon as it has started a read, so this - not the
    #: call - is the moment the tables hold the new rows. A test awaits it; the
    #: desk uses it for nothing, and it carries no payload on purpose.
    refreshFinished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.current_pick_rows: list[dict[str, Any]] = []
        self.setup_type_rows: list[dict[str, Any]] = []
        self.recent_type_rows: list[dict[str, Any]] = []
        self.short_term_rows: list[dict[str, Any]] = []
        # The last verdict that named a leader off FRESH evidence, per horizon.
        # `working_lately.select_leader`'s `last_reliable_reading` state needs a
        # `previous` to carry, and without this it was unreachable from the
        # desk: every refresh started from nothing, so a tracker that stopped
        # writing produced "no clear leader" rather than "here is the last thing
        # we could honestly read, and it is N sessions old" - which is strictly
        # less information at exactly the moment the trader needs more.
        # In-memory only and deliberately so: it is a cache of a reading, never
        # evidence, and ST6 owns persisting it with the rest of the snapshot.
        self._last_fresh_verdicts: dict[str, Any] = {}
        self.playbook_rows: list[dict[str, Any]] = []
        self.scan_factor_rows: list[dict[str, Any]] = []
        self.tier_performance_rows: list[dict[str, Any]] = []
        self.catch_rate_rows: list[dict[str, Any]] = []
        self.human_pick_rows: list[dict[str, Any]] = []
        self.band_variant_rows: list[dict[str, Any]] = []
        # ST6.4. The SHARED Working-lately snapshot, handed over by the desk's
        # service. Empty means this page renders its own read and LABELS it
        # `panel read`, which is the difference between "the desk's answer" and
        # "this tab's answer" being visible instead of guessed at.
        self._working_lately_snapshot: dict[str, Any] = {}

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
        # The Band variant tab's coverage line (packet M1). It sits ABOVE the
        # table because it is the first thing that must be true about the table:
        # for ten days every row read n_variant = 0 and the empty comparison
        # looked like a comparison.
        self.band_variant_status_label = QLabel(BAND_VARIANT_NO_EXPORT_SENTENCE)
        self.band_variant_status_label.setObjectName("MutedLabel")
        self.band_variant_status_label.setWordWrap(True)

        # M3.3: the Setup Types tab says how many records aged out UNMEASURED
        # and were left out of its numbers - an exclusion nobody can see is a
        # second version of the defect it fixes.
        #
        # ONLY that tab. Current Picks renders `master_avwap_tier_list.csv`,
        # a different population entirely, and a count taken from the
        # setup-type export would have been describing rows that tab does not
        # show (reviewer, 2026-09-05).
        self.setup_type_status_label = QLabel("")
        self.setup_type_status_label.setObjectName("MutedLabel")
        self.setup_type_status_label.setWordWrap(True)

        # M5.2 / M5.3: one population sentence per new tab, ABOVE its table. A
        # control row and a pick look identical in a table; the sentence is the
        # only thing that keeps them apart.
        self.control_discovery_status_label = QLabel(CONTROL_DISCOVERY_NO_EXPORT_SENTENCE)
        self.control_discovery_status_label.setObjectName("MutedLabel")
        self.control_discovery_status_label.setWordWrap(True)
        self.study_discovery_status_label = QLabel(STUDY_DISCOVERY_NO_EXPORT_SENTENCE)
        self.study_discovery_status_label.setObjectName("MutedLabel")
        self.study_discovery_status_label.setWordWrap(True)
        self.exit_framework_status_label = QLabel(EXIT_FRAMEWORK_NO_EXPORT_SENTENCE)
        self.exit_framework_status_label.setObjectName("MutedLabel")
        self.exit_framework_status_label.setWordWrap(True)

        self.tabs = QTabWidget()
        # G2b.2: every tab NAMES the column that takes the slack and the
        # identifiers that elide, by key. See `_make_table`.
        self.current_table, self.current_model = self._make_table(
            CURRENT_PICK_COLUMNS, text_key="scan_factor_matches"
        )
        self.setup_type_table, self.setup_type_model = self._make_table(
            SETUP_TYPE_COLUMNS, text_key="sample_setups"
        )
        self.recent_type_table, self.recent_type_model = self._make_table(
            RECENT_TYPE_COLUMNS, text_key="sample_setups"
        )
        self.short_term_table, self.short_term_model = self._make_table(
            SHORT_TERM_COLUMNS, text_key="sample_setups"
        )
        # Exit Plan is what a playbook row is FOR; the samples beside it are
        # longer, so the measured rule gave them the width Exit Plan needed.
        self.playbook_table, self.playbook_model = self._make_table(
            PLAYBOOK_COLUMNS, text_key="profit_take_summary", elide_keys=("sample_setups",)
        )
        self.scan_factor_table, self.scan_factor_model = self._make_table(
            SCAN_FACTOR_COLUMNS, text_key="sample_observations"
        )
        self.tier_performance_table, self.tier_performance_model = self._make_table(
            TIER_PERFORMANCE_COLUMNS, text_key="sample_observations"
        )
        # The GUI review's finding verbatim: the CAUGHT examples ate the width
        # and the MISSED ones - the point of the tab - clipped.
        self.catch_rate_table, self.catch_rate_model = self._make_table(
            CATCH_RATE_COLUMNS,
            text_key="sample_missed_winners",
            elide_keys=("sample_caught_winners",),
        )
        # One identifier and ten measurements, and no free-text column at all,
        # so the slack stays EMPTY rather than going to `cohort` or `Delta %`.
        self.human_pick_table, self.human_pick_model = self._make_table(
            HUMAN_PICK_COLUMNS, elide_keys=("cohort",), stretch_last=False
        )
        self.band_variant_table, self.band_variant_model = self._make_table(
            BAND_VARIANT_COLUMNS, text_key="setup_family", elide_keys=("exit_template_id",)
        )
        # `Family` takes the slack on both discovery tabs, so `Win % (low)`
        # never takes it - which is what an EMPTY tab did, measuring headers.
        self.control_discovery_table, self.control_discovery_model = self._make_table(
            DISCOVERY_COLUMNS, text_key="setup_family", elide_keys=("cohort",)
        )
        self.study_discovery_table, self.study_discovery_model = self._make_table(
            DISCOVERY_COLUMNS, text_key="setup_family", elide_keys=("cohort",)
        )
        self.exit_framework_table, self.exit_framework_model = self._make_table(
            EXIT_FRAMEWORK_COLUMNS,
            text_key="framework_family",
            elide_keys=("exit_template_id",),
        )
        for table in (
            self.control_discovery_table,
            self.study_discovery_table,
            self.exit_framework_table,
        ):
            table.setMinimumHeight(TABLE_TEN_ROWS_PX)
        self.attribute_table, self.attribute_model = self._make_table(
            ATTRIBUTE_LEADERBOARD_COLUMNS,
            text_key="sample_setups",
            elide_keys=("value_label",),
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
        self.tabs.addTab(
            self._make_explained_tab(
                "Every tracked setup grouped by side, bucket, family, zone, retest and compression.",
                self.setup_type_table,
                status=self.setup_type_status_label,
            ),
            "Setup Types",
        )
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
                status=self.band_variant_status_label,
            ),
            "Band Variant",
        )
        self.tabs.addTab(
            self._make_explained_tab(
                "SHADOW EVIDENCE, packet M5. The control / holdout sample: setups the scan "
                "REJECTED, graded on their own scenarios beside the promoted ones, so a "
                "family the gate keeps throwing away can be seen. A row here is NEVER a pick "
                "and nothing on this tab scores, ranks, gates or alerts. Win rate leads with "
                "its n and its Wilson lower bound, and the sort is the BOUND - a 100% on two "
                "setups is not better than a 60% on ninety.",
                self.control_discovery_table,
                status=self.control_discovery_status_label,
            ),
            "Controls",
        )
        self.tabs.addTab(
            self._make_explained_tab(
                "SHADOW EVIDENCE, packet M5. The study namespace (docs/SETUPS_TEST.md): setup "
                "ideas measured for edge BEFORE they touch scoring - isolated from Expected-R, "
                "calibration and live ranking. Episode-deduped, representative-stop, "
                "net-of-cost closed R. Win rate leads, sorted by its Wilson lower bound.",
                self.study_discovery_table,
                status=self.study_discovery_status_label,
            ),
            "Studies",
        )
        self.tabs.addTab(
            self._make_explained_tab(
                "SHADOW EVIDENCE, packet M5. Exit templates compared on the SAME setups, one "
                "row per framework / template / side / bucket. The `comparison_apr2026` rows "
                "are EXPERIMENTAL: a 1.25R hard stop and an SMA_50 short-near-favorite skip, "
                "simulated since April and never taken. They are excluded from every champion "
                "aggregate by design and this is the first surface that reads them. THE PAIRING "
                "IS THE SAME SETUPS MINUS THE TEMPLATE'S OWN FILTER: a template that skips "
                "scenarios by definition has a smaller n, and Filtered carries the difference so "
                "n + Filtered equals the baseline's n - a smaller denominator here is the "
                "experiment working, not a worse result. Rows are grouped by side and bucket with "
                "the baseline above its twin. Nothing here scores, ranks, gates or alerts, and "
                "nothing here retires evidence.",
                self.exit_framework_table,
                status=self.exit_framework_status_label,
            ),
            "Exit frameworks",
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
        #: Packet G4b. `SetupDetailView.shown_identity` is deliberately coarse -
        #: `(kind, side, family, symbol, dimension)` - and on the Setup Types
        #: export two rows of one (side, family) live in different zones and
        #: buckets, so the identity alone would re-show the wrong row after a
        #: refresh. This is the SHOWN row's widening key, recorded beside the
        #: identity by the two show helpers below and trusted only while the
        #: pane is up. `None` means nothing is shown.
        self._detail_widened_key: tuple | None = None
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
        #: Every table that can open the pane, with the model that holds its
        #: rows and the `show_*` call its tab uses (`""` is `show_setup`).
        #: Looked up by TAB rather than by position, so adding or reordering a
        #: tab cannot silently re-point the re-show.
        self._detail_tables: tuple[tuple[DataTable, TrackerTableModel, str], ...] = (
            (self.current_table, self.current_model, ""),
            *(
                (table, model, kind)
                for (table, kind), model in zip(
                    explained_tables,
                    (
                        self.setup_type_model,
                        self.recent_type_model,
                        self.short_term_model,
                        self.playbook_model,
                        self.scan_factor_model,
                        self.tier_performance_model,
                        self.catch_rate_model,
                        self.human_pick_model,
                    ),
                    strict=True,
                )
            ),
        )
        # G4b.1: any tab move is a context change and retires the explanation.
        # The pane carries STOP AND TARGET PRICES on this page, so one left
        # standing beside another tab's table is a price plan read against the
        # wrong row.
        self.tabs.currentChanged.connect(self._on_context_tab_changed)

        self._attributesLoaded.connect(self._on_attributes_loaded)
        #: G7.2. The export read runs on ONE `ReadWorker` and is single-flight:
        #: a refresh asked for while one is in flight is coalesced into it, the
        #: same rule `start_attribute_refresh` has always used.
        self._read_worker: ReadWorker | None = None
        self._shutting_down = False
        #: A refresh asked for while one is in flight. The worker in flight
        #: takes it (one more pass, whatever the number of requests), so the
        #: request is never LOST and never doubles the reads either.
        self._refresh_pending = False
        self._refresh_lock = threading.Lock()
        #: The spinbox value the next pass reads with. Taken on the Qt thread -
        #: a `QSpinBox` is not a worker's to ask.
        self._refresh_min_closed = 5
        #: What each table was last rendered FROM, so an export nothing rewrote
        #: costs no model reset and no column fit. Keyed by the table's own
        #: attribute name; a table absent from here has never been rendered.
        self._rendered_from: dict[str, tuple] = {}
        #: G7.1: the first show pays for the first read, never the constructor.
        self._loaded_once = False
        self._build_layout()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Read the first time the page is actually looked at (G7.1).

        This was the 1.3 s op in the G0 baseline and it ran at startup for a
        Research tab nobody had selected. A `QTabWidget` child gets its
        `showEvent` only when its tab is chosen, so Research's first paint costs
        one child's load rather than nine.
        """
        super().showEvent(event)
        if self._loaded_once:
            return
        self._loaded_once = True
        self.refresh()

    def shutdown(self) -> None:
        """Let no read outlive the panel it was going to update."""
        self._shutting_down = True
        join_worker(self._read_worker)
        self._read_worker = None
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
        *,
        text_key: str | None = None,
        elide_keys: tuple[str, ...] = (),
        stretch_last: bool = True,
    ) -> tuple[DataTable, TrackerTableModel]:
        """Build one tab's table and NAME the column that takes the slack.

        Packet G2b.2. Every table here ran `apply_width_rule`'s measured path,
        which is content-dependent and moves: on a populated Catch Rate the
        caught samples outgrew the missed ones and ate the width the Missed
        Samples column exists to show, and on an EMPTY Controls tab the widest
        thing on screen is a HEADER, so `Win % (low)` stretched while `Family`
        sat at its floor. Naming the column fixes the answer to what the tab is
        for, populated or empty.

        `text_key` and `elide_keys` are KEYS, resolved through `_column_index`
        against this table's own tuple. A literal index would be a defect
        waiting for the next column insert - ST2 and M5 each added columns to
        these tuples this month.
        """
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
        table.set_width_rule(
            text_columns=None if text_key is None else (_column_index(columns, text_key),),
            elide_columns=tuple(_column_index(columns, key) for key in elide_keys),
            stretch_last=stretch_last,
        )
        return table, model

    def _make_explained_tab(
        self, description: str, table: DataTable, *, footer=None, status=None
    ) -> QWidget:
        tab = QWidget()
        label = QLabel(description)
        label.setObjectName("MutedLabel")
        label.setWordWrap(True)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(label)
        # `status` is what the table currently says about itself and belongs
        # ABOVE it; `footer` is how the read went and belongs below.
        if status is not None:
            layout.addWidget(status)
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

    def set_working_lately_snapshot(self, payload: Any) -> None:
        """Take the desk's shared reading (ST6.4). Formatting only, no read.

        The banner then renders THIS snapshot rather than its own CSV pass, so
        the tracker and the strip above the M5 list print the same
        `snapshot_id`. Handing over an empty payload puts the page back on its
        own labelled `panel read`.
        """
        self._working_lately_snapshot = dict(payload or {})
        try:
            self.summary_view.setHtml(_summary_html(self))
        except Exception:  # noqa: BLE001 - a banner is never worth a traceback
            logging.debug("Setup Tracker summary re-render skipped", exc_info=True)

    # ------------------------------------------------------------------
    # The refresh: one read on one worker, then one render on the Qt thread
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Ask for the twelve exports and the human-focus read. G7.2.

        Everything this used to do inline now happens on ONE `ReadWorker` -
        twelve `_load_csv_rows_cached` calls, `load_human_focus_performance_rows`
        and the pure ranking of what they returned - and the Qt thread does the
        rendering alone. It measured 1.3 s p95 at 3456 x 2160 in the G0 baseline
        and it fires on construction, on every spinbox step and on the button.

        Single-flight and COALESCED: a refresh asked for while one is in flight
        is taken by the worker in flight as one more pass, whatever the number
        of requests - so a spinbox step during a read is never lost, two reads
        are never in flight at once, and there is exactly one render and one
        `refreshFinished` per worker. `refreshFinished` is emitted when the ROWS
        ARE APPLIED, never when this call returns.
        """
        if self._shutting_down:
            return
        with self._refresh_lock:
            self._refresh_min_closed = int(self.min_closed_input.value())
            worker = self._read_worker
            if worker is not None and worker.isRunning():
                self._refresh_pending = True
                return
            self._refresh_pending = False
        worker = ReadWorker(self._read_until_nothing_is_pending, self)
        self._read_worker = worker
        worker.finished_with.connect(self._on_exports_loaded)
        worker.failed.connect(self._on_exports_failed)
        #: G7 fix round, items 1 and 2. `finished` is QThread's OWN signal,
        #: fired only once the thread has actually stopped - unlike
        #: `finished_with`/`failed`, it fires for every outcome, so it is the
        #: one place both fixes belong.
        worker.finished.connect(lambda: self._on_worker_finished(worker))
        worker.start()

    def _on_worker_finished(self, worker: ReadWorker) -> None:
        """G7 fix round items 1 and 2, both on the worker's own `finished`.

        Item 1: `_read_until_nothing_is_pending` releases `_refresh_lock`
        with `_refresh_pending` False the moment its loop decides to stop -
        but the `ReadWorker` thread has not actually finished at that
        instant, so a `refresh()` call landing in that window sees
        `worker.isRunning()` still True, sets `_refresh_pending = True`, and
        returns without starting anything. Nobody was going to check that
        flag again: the loop already left, and the page went on showing rows
        ranked at the previous `min_closed`. Restarting here, on `finished`,
        closes the window.

        Item 2: dropping the panel's reference to `worker` here - before
        `deleteLater()`, and before a possible restart hands `_read_worker` a
        new instance - is what keeps a later `refresh()` from calling
        `isRunning()` on a C++ object Qt has since destroyed.
        """
        if self._read_worker is worker:
            self._read_worker = None
        with self._refresh_lock:
            pending = self._refresh_pending
            if pending:
                self._refresh_pending = False
        if pending and not self._shutting_down:
            self.refresh()
        worker.deleteLater()

    def _read_until_nothing_is_pending(self) -> dict[str, Any]:
        """The worker's whole job: read, and read again if one was asked for.

        Running the coalesced pass HERE rather than from the Qt-thread slot is
        what makes `shutdown()`'s join enough - a request made a moment before
        the desk closes cannot leave a read starting after the panel is gone,
        because there is only ever the one thread and joining it is joining
        everything it was going to do.
        """
        while True:
            with self._refresh_lock:
                min_closed = int(self._refresh_min_closed)
            payload = _read_tracker_exports(min_closed)
            with self._refresh_lock:
                if not self._refresh_pending:
                    return payload
                self._refresh_pending = False

    def _on_exports_failed(self, message: str) -> None:
        """A read that could not run leaves the page showing what it had.

        `ReadWorker` never raises into Qt, and a tracker that blanked itself to
        announce a failed read would destroy the only copy of what it knew.
        """
        self.status_label.setText(f"Tracker exports could not be read: {message}")
        self.refreshFinished.emit()

    def _on_exports_loaded(self, payload: object) -> None:
        """Apply one read's rows. Qt thread, and the ONLY place that renders.

        A table is reset and re-fitted only when the export behind it CHANGED
        since the last render (`_rendered_from`). The mtime cache already
        skipped the parse on an unchanged file, but the thirteen model resets
        and thirteen column fits ran anyway - on every spinbox step, over files
        a scan rewrites a few times a day.
        """
        data = payload if isinstance(payload, dict) else {}
        if not data:
            self.refreshFinished.emit()
            return
        signatures: dict[str, Any] = data.get("signatures") or {}
        ranked: dict[str, Any] = data.get("ranked") or {}
        raw: dict[str, Any] = data.get("raw") or {}
        min_closed = int(data.get("min_closed") or 0)

        all_setup_type_rows = raw.get("setup_type") or []
        band_variant_export_rows = raw.get("band_variant") or []
        control_discovery_export_rows = raw.get("control_discovery") or []
        study_discovery_export_rows = raw.get("study_discovery") or []
        exit_framework_export_rows = raw.get("exit_framework") or []

        self.current_pick_rows = ranked.get("current") or []
        self.setup_type_rows = ranked.get("setup_type") or []
        self.recent_type_rows = ranked.get("recent_type") or []
        self.short_term_rows = ranked.get("short_term") or []
        self.playbook_rows = ranked.get("playbook") or []
        self.scan_factor_rows = ranked.get("scan_factor") or []
        self.tier_performance_rows = ranked.get("tier_performance") or []
        self.catch_rate_rows = ranked.get("catch_rate") or []
        self.human_pick_rows = ranked.get("human_pick") or []
        self.band_variant_rows = ranked.get("band_variant") or []
        self.control_discovery_rows = ranked.get("control_discovery") or []
        self.study_discovery_rows = ranked.get("study_discovery") or []
        self.exit_framework_rows = ranked.get("exit_framework") or []

        self.band_variant_status_label.setText(
            band_variant_coverage_sentence(band_variant_export_rows)
        )
        self.control_discovery_status_label.setText(
            discovery_population_sentence(control_discovery_export_rows, kind="control")
        )
        self.study_discovery_status_label.setText(
            discovery_population_sentence(study_discovery_export_rows, kind="study")
        )
        self.exit_framework_status_label.setText(
            exit_framework_population_sentence(exit_framework_export_rows)
        )

        rendered: dict[str, tuple] = {}
        for table_name, model_name, rows, memo in _table_render_plan(
            ranked, signatures, min_closed, str(data.get("human_focus_digest") or "")
        ):
            rendered[table_name] = memo
            if self._rendered_from.get(table_name) == memo:
                continue
            getattr(self, model_name).set_rows(rows)
            getattr(self, table_name).fit_columns()
        self._rendered_from = rendered

        # The attribute leaderboard is read on its own worker (19.7 MB live);
        # the table fills when it arrives. Left exactly as it was.
        self.start_attribute_refresh()

        tracked_setups = sum(_int(row.get("tracked_setups")) for row in all_setup_type_rows)
        current_sa = sum(1 for row in self.current_pick_rows if str(row.get("tier") or "").upper() in {"S", "A"})
        self.tracked_tile.set_value(str(tracked_setups))
        self.current_tile.set_value(str(current_sa))
        self.best_type_tile.set_value(_best_type_label(self.setup_type_rows))
        self.best_short_term_tile.set_value(_best_short_term_label(self.short_term_rows))
        self.best_factor_tile.set_value(_best_factor_label(self.scan_factor_rows))
        self.summary_view.setHtml(_summary_html(self))

        # M3.2: two clocks, named. The tracker snapshot's own `saved_at` /
        # `saved_by` come off the export the tracker pass stamped them onto -
        # never off the 1.1 GB JSON, which this panel must never open.
        self.setup_type_status_label.setText(
            setup_type_population_sentence(all_setup_type_rows)
        )
        status = tracker_clock_sentence(
            _first_non_empty(all_setup_type_rows, "tracker_saved_at"),
            _first_non_empty(all_setup_type_rows, "tracker_saved_by"),
            # Stat'ed on the worker with the reads (G7.2): a `stat` is a file
            # call and this one used to happen on the render path.
            str(data.get("scan_factor_mtime_text") or ""),
        )
        self.status_label.setText(status)
        self.statusChanged.emit(status)

        # G4b.2, LAST: every model above now holds the new rows, so an open
        # explanation is either re-drawn from the row that replaced it or taken
        # down. Running it here rather than earlier is what keeps the tables
        # themselves untouched by this packet.
        self._reshow_or_clear_detail()
        self.refreshFinished.emit()

    # ------------------------------------------------------------------
    # Click-to-detail: family mechanics + this symbol's stop/target prices
    # ------------------------------------------------------------------
    def _on_pick_clicked(self, index) -> None:
        row = index.data(ROW_ROLE)
        if not isinstance(row, dict):
            return
        self._show_pick_row(row)

    def _on_research_row_clicked(self, index, kind: str) -> None:
        row = index.data(ROW_ROLE)
        if not isinstance(row, dict):
            return
        self._show_research_row(kind, row)

    # -- the two show paths, shared by a click and by a refresh's re-show ---
    def _show_pick_row(self, row: dict[str, Any]) -> None:
        self.detail_view.show_setup(
            symbol=str(row.get("symbol") or ""),
            side=str(row.get("side") or "LONG"),
            setup_family=str(row.get("setup_family") or ""),
            tier=str(row.get("tier") or ""),
            last_close=row.get("last_close"),
        )
        self._detail_widened_key = _detail_widened_key(row)

    def _show_research_row(self, kind: str, row: dict[str, Any]) -> None:
        self.detail_view.show_research_row(kind, row)
        self._detail_widened_key = _detail_widened_key(row)

    def _clear_detail(self) -> None:
        self.detail_view.clear()
        self._detail_widened_key = None

    def _on_context_tab_changed(self, _index: int) -> None:
        """Any tab move retires the explanation (packet G4b.1)."""
        self._clear_detail()

    def _reshow_or_clear_detail(self) -> None:
        """After a re-read, redraw the open row from its NEW dict, or take it down.

        **The first question is whether the pane is up**, not whether a match
        exists: the trader reaches the hidden state by moving tabs, and a scan
        that re-showed on a match alone would pop an explanation open under
        someone who had closed it.

        The row is then looked up in the model that now holds the CURRENT tab's
        rows and the pane is redrawn from the new dict - never the cached one,
        or it would keep printing a stop, a target or a mean R the table has
        already revised. A row the re-read dropped takes its explanation with
        it. One linear scan of that one model, no dict copied per row.
        """
        view = self.detail_view
        identity = getattr(view, "shown_identity", None)
        if view.isHidden() or not identity:
            return
        source = self._detail_source_for_current_tab()
        if source is None:
            self._clear_detail()
            return
        model, kind = source
        wanted_widening = self._detail_widened_key
        for position in range(model.rowCount()):
            row = model.row_at(position)
            if not isinstance(row, dict):
                continue
            if _detail_row_identity(row, kind) != identity:
                continue
            if wanted_widening is not None and _detail_widened_key(row) != wanted_widening:
                continue
            if kind:
                self._show_research_row(kind, row)
            else:
                self._show_pick_row(row)
            return
        self._clear_detail()

    def _detail_source_for_current_tab(self) -> tuple[TrackerTableModel, str] | None:
        """The (model, show-kind) behind the tab the trader is looking at."""
        widget = self.tabs.currentWidget()
        if widget is None:
            return None
        for table, model, kind in self._detail_tables:
            if widget is table or widget.isAncestorOf(table):
                return model, kind
        return None

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


#: Packet G4b. The two columns that make two rows of one (side, family)
#: DIFFERENT rows on this page. Setup Types groups by side, bucket, family,
#: zone, retest and compression, so `shown_identity` alone collides there; these
#: two carry the collision that matters to the trader. A row that carries
#: neither reads `("", "")` on both sides of the comparison, so widening never
#: turns a real match into a miss. The lead widened it to the WHOLE Setup Types
#: grain at merge (2026-09-07): `retest_label` and `compression_label` joined,
#: so no two rows of one (side, family) can collide on this tuple.
DETAIL_WIDENING_KEYS = ("favorite_zone", "priority_bucket", "retest_label", "compression_label")


def _detail_widened_key(row: dict[str, Any]) -> tuple:
    return tuple(str(row.get(key) or "") for key in DETAIL_WIDENING_KEYS)


def _detail_row_identity(row: dict[str, Any], kind: str) -> tuple:
    """The identity `SetupDetailView` will publish for this row under `kind`.

    A MIRROR of `SetupDetailView._render`, kept here so the re-show can ask
    "is this the row the pane is showing?" without drawing anything. `kind`
    empty is the `show_setup` path (Current Picks); anything else is the
    `show_research_row` path.
    """
    if kind:
        return (
            str(kind or ""),
            str(row.get("side") or row.get("direction") or "LONG").strip().upper(),
            str(row.get("setup_family") or ""),
            "",
            str(row.get("dimension") or ""),
        )
    symbol = str(row.get("symbol") or "").strip().upper()
    return (
        "setup" if symbol else "family",
        str(row.get("side") or "LONG").strip().upper(),
        str(row.get("setup_family") or ""),
        symbol,
        "",
    )


#: The twelve exports `refresh()` reads, by the short name the payload, the
#: signatures and the per-table memo all use (G7.2). The attribute leaderboard
#: is deliberately absent: it is 19.7 MB and has had its own worker since
#: Phase 0.9, and the packet leaves it exactly where it is.
def tracker_export_files() -> tuple[tuple[str, Any], ...]:
    """Resolved at CALL time, never bound into a module constant.

    These twelve names are patched on this module by the tests that point a
    panel at a temporary home folder, so a tuple built at import would read the
    paths the live desk uses no matter what a test said.
    """
    return (
        ("setup_type", SETUP_TYPE_STATS_FILE),
        ("playbook", SETUP_PLAYBOOKS_FILE),
        ("tier_performance", MASTER_AVWAP_TIER_PERFORMANCE_FILE),
        ("tier_list", MASTER_AVWAP_TIER_LIST_FILE),
        ("recent_type", RECENT_SETUP_TYPE_STATS_FILE),
        ("short_term", SHORT_HORIZON_FILE),
        ("scan_factor", MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE),
        ("catch_rate", MASTER_AVWAP_TIER_CATCH_RATE_FILE),
        ("band_variant", BAND_VARIANT_STATS_FILE),
        ("control_discovery", CONTROL_DISCOVERY_STATS_FILE),
        ("study_discovery", STUDY_DISCOVERY_STATS_FILE),
        ("exit_framework", EXIT_FRAMEWORK_STATS_FILE),
    )


def _read_tracker_exports(min_closed: int) -> dict[str, Any]:
    """The whole of the Setup Tracker's read, on a worker thread. G7.2.

    Twelve cached CSV reads, the human-focus read and the pure ranking of what
    they returned - none of it touches a widget, and it is the same code in the
    same order the Qt thread used to run inline. Each file's `(mtime_ns, size)`
    is taken BEFORE its read: a file rewritten mid-pass then has a signature the
    NEXT refresh will see as changed, which is the safe direction to be wrong in.

    The human-focus store has no file signature to take, so its rows carry a
    content digest instead. Same question, answered from what was read.
    """
    signatures: dict[str, Any] = {}
    raw: dict[str, list[dict]] = {}
    for name, path in tracker_export_files():
        signatures[name] = _csv_signature(path)
        raw[name] = _load_csv_rows_cached(path)

    human_focus_rows = load_human_focus_performance_rows()
    digest = hashlib.sha1(
        repr(human_focus_rows).encode("utf-8", "replace")
    ).hexdigest()

    ranked = {
        "current": _rank_current_picks(raw["tier_list"]),
        "setup_type": _rank_setup_types(
            _setup_type_headline_rows(raw["setup_type"]), min_closed=min_closed
        ),
        "recent_type": _rank_recent_types(_recent_type_headline_rows(raw["recent_type"])),
        "short_term": _rank_short_term(raw["short_term"]),
        "playbook": _rank_playbooks(raw["playbook"], min_closed=min_closed),
        "scan_factor": _rank_scan_factors(raw["scan_factor"]),
        "tier_performance": _rank_tier_performance(raw["tier_performance"]),
        "catch_rate": _rank_catch_rates(raw["catch_rate"]),
        "human_pick": build_human_focus_comparison_rows(
            human_focus_rows, raw["tier_performance"]
        ),
        "band_variant": _rank_band_variants(raw["band_variant"]),
        "control_discovery": _rank_discovery_rows(raw["control_discovery"]),
        "study_discovery": _rank_discovery_rows(raw["study_discovery"]),
        "exit_framework": _rank_exit_frameworks(raw["exit_framework"]),
    }
    return {
        "min_closed": int(min_closed),
        "signatures": signatures,
        "human_focus_digest": digest,
        "raw": raw,
        "ranked": ranked,
        "scan_factor_mtime_text": _latest_mtime_text(
            [MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE]
        ),
    }


def _table_render_plan(
    ranked: dict[str, Any],
    signatures: dict[str, Any],
    min_closed: int,
    human_focus_digest: str,
) -> tuple[tuple[str, str, list[dict], tuple], ...]:
    """`(table attribute, model attribute, rows to show, what they came from)`.

    The memo is what the rows were BUILT from, never the rows themselves: the
    file's `(mtime_ns, size)`, plus `min_closed` for the two tables the spinbox
    actually re-ranks, plus the human-focus digest for the one table with no
    file behind it. Two tables read the same tier-performance export and both
    say so, so a rewrite of it re-fits both and nothing else.
    """
    return (
        ("current_table", "current_model", (ranked.get("current") or [])[:300],
         (signatures.get("tier_list"),)),
        ("human_pick_table", "human_pick_model", ranked.get("human_pick") or [],
         (signatures.get("tier_performance"), human_focus_digest)),
        ("setup_type_table", "setup_type_model", (ranked.get("setup_type") or [])[:300],
         (signatures.get("setup_type"), min_closed)),
        ("recent_type_table", "recent_type_model", (ranked.get("recent_type") or [])[:300],
         (signatures.get("recent_type"),)),
        ("short_term_table", "short_term_model", (ranked.get("short_term") or [])[:300],
         (signatures.get("short_term"),)),
        ("playbook_table", "playbook_model", (ranked.get("playbook") or [])[:300],
         (signatures.get("playbook"), min_closed)),
        ("scan_factor_table", "scan_factor_model", (ranked.get("scan_factor") or [])[:300],
         (signatures.get("scan_factor"),)),
        ("tier_performance_table", "tier_performance_model",
         ranked.get("tier_performance") or [], (signatures.get("tier_performance"),)),
        ("catch_rate_table", "catch_rate_model", ranked.get("catch_rate") or [],
         (signatures.get("catch_rate"),)),
        ("band_variant_table", "band_variant_model",
         (ranked.get("band_variant") or [])[:300], (signatures.get("band_variant"),)),
        ("control_discovery_table", "control_discovery_model",
         (ranked.get("control_discovery") or [])[:300],
         (signatures.get("control_discovery"),)),
        ("study_discovery_table", "study_discovery_model",
         (ranked.get("study_discovery") or [])[:300],
         (signatures.get("study_discovery"),)),
        ("exit_framework_table", "exit_framework_model",
         (ranked.get("exit_framework") or [])[:300],
         (signatures.get("exit_framework"),)),
    )


#: Parsed export rows, keyed by path, with the (mtime_ns, size) they came from.
#: Bounded to one entry per export file - fourteen since packet M5, and they are
#: rewritten by the scan, not by this page.
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


#: What the Band variant tab says when the export has never been written.
BAND_VARIANT_NO_EXPORT_SENTENCE = "No band-variant comparison has been written yet."


def band_variant_coverage_sentence(rows: list[dict[str, Any]]) -> str:
    """One sentence: how much of the comparison was actually measured.

    Packet M1, 2026-09-05. The challenger shipped on 2026-08-26 and measured
    NOTHING for the ten days after it - `n_variant` read 0 on all 40 rows of
    `master_avwap_band_variant_stats.csv` across 11,292 setups - and the table
    gave no sign of it: a family, a side, a champion R and blank challenger
    cells read as "no difference" rather than "never computed". This line makes
    an empty comparison impossible to mistake for a comparison.

    Pure, and built from the export's OWN counts. It never re-reads the file, it
    never opens the 1.1 GB tracker JSON, and it never writes anything.
    """
    if not rows:
        return BAND_VARIANT_NO_EXPORT_SENTENCE

    def _count(row: dict[str, Any], key: str) -> int:
        try:
            return int(float(str(row.get(key) or "0").strip() or 0))
        except (TypeError, ValueError):
            return 0

    total = sum(_count(row, "n") for row in rows)
    measured = sum(_count(row, "n_variant") for row in rows)
    unmeasured = sum(_count(row, "n_variant_unmeasured") for row in rows)

    sentence = f"Measured {measured} of {total} setups"
    if not unmeasured:
        return f"{sentence}."

    reasons: dict[str, int] = {}
    for row in rows:
        count = _count(row, "n_variant_unmeasured")
        reason = str(row.get("top_unmeasured_reason") or "").strip()
        if count and reason:
            reasons[reason] = reasons.get(reason, 0) + count
    if not reasons:
        return f"{sentence} ({unmeasured} unmeasured)."
    # Ties break alphabetically so the line is stable between refreshes.
    top = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return f"{sentence} ({unmeasured} unmeasured: {top})."


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


# ---------------------------------------------------------------------------
# Packet M5.2 / M5.3 - the Controls, Studies and Exit frameworks tabs.
#
# Three pure readers. Each sorts by the WILSON LOWER BOUND and each says what
# its population is, because these three are the easiest rows on the desk to
# misread as recommendations.
# ---------------------------------------------------------------------------

CONTROL_DISCOVERY_NO_EXPORT_SENTENCE = (
    "No control comparison has been written yet. The control sample is setups the "
    "scan REJECTED, graded on their own scenarios; nothing here is a pick."
)
STUDY_DISCOVERY_NO_EXPORT_SENTENCE = (
    "No study comparison has been written yet. Study setups are ideas that have "
    "never been promoted and touch no score; nothing here is a pick."
)
EXIT_FRAMEWORK_NO_EXPORT_SENTENCE = (
    "No exit-framework comparison has been written yet. Rows marked EXPERIMENTAL "
    "are what-if exits simulated on the same setups; nothing here is a pick."
)


def _lower_bound(row: dict[str, Any]) -> float | None:
    """The Wilson lower bound off an export row, or None when it has none.

    A blank is not a zero. A cell nothing graded has no bound to rank on and
    must sort LAST rather than below every graded cell as though it had lost.
    """
    text = str(row.get("win_rate_lb") or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _rank_discovery_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All-history block first, then best Wilson lower bound, then biggest n.

    Sorting by the LOWER BOUND rather than the raw rate is the headline rule and
    the whole reason the bound is computed: a 100% on two rejected setups would
    otherwise sit above a 60% on ninety and read as the strongest finding on the
    page. Presentation only - this never re-reads a file and never writes one.
    """
    return sorted(
        rows,
        key=lambda row: (
            0 if str(row.get("window") or "") == "all" else 1,
            0 if str(row.get("row_kind") or "") == "cohort" else 1,
            _lower_bound(row) is None,
            -(_lower_bound(row) or 0.0),
            -_float(row.get("n"), 0.0),
            str(row.get("cohort") or ""),
            str(row.get("setup_family") or ""),
            str(row.get("side") or ""),
        ),
    )


def _rank_exit_frameworks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Grouped by (side, bucket), then framework, ranked by the bound INSIDE.

    Reviewer advisory 3 (2026-09-05). A pure bound sort is right for a table
    whose rows are independent, and wrong for this one: every row here exists to
    be read AGAINST its twin, and sorting the whole table by the bound
    interleaves the sides - one live ordering came out SHORT, LONG, LONG, SHORT.
    Two rows a reader has to hunt for are two rows they will not compare, which
    would leave this tab as unread as the framework it was built to surface.

    So: the (side, bucket) blocks are ordered by the BEST bound in each, so the
    strongest pairing is still on top; inside a block the BASELINE comes first
    and its comparison twin follows, because the reference belongs before the
    challenger; and inside a framework the bound orders the templates. Blank
    bounds sort last at every level - a cell nothing graded has nothing to rank
    on and is not an edge of zero. Presentation only.
    """
    best_in_block: dict[tuple[str, str], float] = {}
    for row in rows:
        block = (str(row.get("side") or ""), str(row.get("priority_bucket") or ""))
        bound = _lower_bound(row)
        if bound is not None:
            best_in_block[block] = max(best_in_block.get(block, bound), bound)

    def _key(row: dict[str, Any]):
        block = (str(row.get("side") or ""), str(row.get("priority_bucket") or ""))
        block_best = best_in_block.get(block)
        family = str(row.get("framework_family") or "")
        return (
            block_best is None,
            -(block_best or 0.0),
            block,
            # The champion's own record is the reference and reads first.
            0 if family == "baseline" else 1,
            family,
            _lower_bound(row) is None,
            -(_lower_bound(row) or 0.0),
            str(row.get("exit_template_id") or ""),
        )

    return sorted(rows, key=_key)


def _graded_episodes(rows: list[dict[str, Any]]) -> int:
    """Total n across the all-history FAMILY rows - the GRADED EPISODE count.

    **Not the population size** (reviewer blocker 2, 2026-09-05): this is what
    was graded, and most records in either namespace never close. Live it is 308
    against 401 control records and 2,600 against 3,992 study ones, so printing
    it under the noun "setups" claims every record was graded and overstates the
    evidence by about a third. `_population_setups` is the other number and the
    sentence carries both.

    Family rows only: the control export's cohort rows partition the same
    episodes a second way, and adding the two together would double-count every
    one of them.
    """
    return sum(
        int(_float(row.get("n"), 0.0))
        for row in rows
        if str(row.get("window") or "") == "all" and str(row.get("row_kind") or "") == "family"
    )


def _population_setups(rows: list[dict[str, Any]]) -> int:
    """How many RECORDS the namespace holds, from the export's own column.

    Same value on every row of a window - it describes the file - so the first
    all-history row that carries one answers it. The panel must never count the
    namespace itself: that means opening the 1.1 GB tracker JSON.
    """
    for row in rows:
        if str(row.get("window") or "") != "all":
            continue
        text = str(row.get("population_setups") or "").strip()
        if text:
            try:
                return int(float(text))
            except (TypeError, ValueError):
                continue
    return 0


def discovery_population_sentence(rows: list[dict[str, Any]], *, kind: str) -> str:
    """One sentence naming the population, so a control is never read as a pick.

    Carries BOTH counts and names each: the graded episodes are the sample the
    numbers rest on, and the record count is the population they were drawn
    from. A sentence with only one of them is wrong whichever one it keeps.

    Pure, and built from the export's OWN counts - it never re-reads the file
    and never opens the 1.1 GB tracker JSON.
    """
    if not rows:
        return (
            CONTROL_DISCOVERY_NO_EXPORT_SENTENCE
            if kind == "control"
            else STUDY_DISCOVERY_NO_EXPORT_SENTENCE
        )
    episodes = _graded_episodes(rows)
    population = _population_setups(rows)
    if kind == "control":
        head = (
            f"{episodes} graded episodes from the {population} control setups the scan "
            "REJECTED, graded on their own scenarios - never picks, and nothing here "
            "scores, ranks or alerts."
        )
    else:
        head = (
            f"{episodes} graded episodes from the {population} study setups - ideas that "
            "have never been promoted and touch no score. Measured here BEFORE any of "
            "them could."
        )
    return f"{head} {_discovery_window_suffix(rows)}".strip()


def _discovery_window_suffix(rows: list[dict[str, Any]]) -> str:
    sessions = ""
    for row in rows:
        if str(row.get("window") or "") == "lately":
            sessions = str(row.get("window_sessions") or "").strip()
            if sessions:
                break
    if not sessions:
        return "Win rate leads, sorted by its Wilson lower bound."
    return (
        f"Two blocks: all history, and the last {sessions} SESSIONS. "
        "Win rate leads, sorted by its Wilson lower bound."
    )


def exit_framework_population_sentence(rows: list[dict[str, Any]]) -> str:
    """What the Exit frameworks table is, in one line.

    Names the EXPERIMENTAL rows explicitly: they are exits that were simulated,
    never taken. A reader who takes one for the champion's record has read a
    what-if as a result.

    **It no longer says "the SAME setups", because that was not true** (reviewer
    blocker 1, 2026-09-05). A template carrying `blocked_stop_rules` is DEFINED
    to skip some scenarios - live, `..._no_sma50_short_nearfav` skipped 98 of
    683 on SHORT / near_favorite_zone - so its `n` is legitimately smaller and
    reading that as a worse result is exactly backwards. The wording is "the
    same setups MINUS the template's own filter", with the filtered count
    printed so the two denominators reconcile.
    """
    if not rows:
        return EXIT_FRAMEWORK_NO_EXPORT_SENTENCE
    experimental = sum(
        1 for row in rows if str(row.get("experimental") or "").strip().lower() in {"true", "1"}
    )
    filtered = sum(int(_float(row.get("n_filtered_by_experiment"), 0.0)) for row in rows)
    families = sorted(
        {str(row.get("framework_family") or "").strip() for row in rows} - {""}
    )
    tail = (
        f" {filtered} scenario(s) were skipped by a template's own "
        "`blocked_stop_rules` and are counted in n_filtered_by_experiment, so n plus that "
        "column reconciles to the baseline's n."
        if filtered
        else " Where a template has no filter of its own, the two n's are EQUAL."
    )
    return (
        f"{len(rows)} exit-template groups across {len(families)} framework(s): "
        f"{', '.join(families)}. {experimental} row(s) are EXPERIMENTAL - what-if exits, "
        "never taken, simulated on the same setups as the baseline MINUS the ones each "
        f"template's own filter skips, and excluded from every champion aggregate.{tail} "
        "Nothing here scores, ranks or alerts."
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
    """Win rate leads, so the ORDER is the Wilson lower bound - inside each side.

    ST2.2 / lead decision 2026-09-06. The old key was the champion's
    `score_delta` then `ranking_score`, which is the SCORER's order and is not
    what "which of my setup types is working" asks; a reader given a Win %
    column sorted by something else reads the first row as the best one.

    **`score_delta` and `ranking_score` are untouched** - they keep their values
    and their column, and they stay in the key as tiebreaks. Sides are kept
    apart because LONG and SHORT are two books, not two ends of one list, and
    interleaving them by bound hides whichever side is quieter. A row with no
    graded closes has no bound and sorts under every row that has one: "not
    measured" is not "measured badly".
    """
    filtered = [row for row in rows if _int(row.get("closed_setups")) >= min_closed]

    def _key(row):
        bound = row.get("win_rate_lb")
        return (
            str(row.get("side") or ""),
            bound is None,
            -(float(bound) if bound is not None else 0.0),
            -_float(row.get("score_delta"), 0.0),
            -_float(row.get("ranking_score"), 0.0),
            -_float(row.get("avg_closed_r_edge"), 0.0),
            -_int(row.get("closed_setups")),
        )

    return sorted(filtered, key=_key)


#: What the cell says when the export carried no integer counts - ST2.1.
COUNTS_NOT_EXPORTED = "counts not exported yet"


def _recent_type_headline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the win-rate headline to each recent-type row - R4 B3, fixed by ST2.1.

    **The counts are READ, never rebuilt.** This used to hand the row's stored
    `win_rate_closed` to `swing_headline`'s rate-to-headline constructor, which
    recovers `round(rate * n)`. That is exact when the stored rate is
    `wins / n` - and this one is not: `win_rate_closed` is a RECENCY-WEIGHTED
    mean (`legacy.build_recent_tracker_setup_family_rows`, half life 14 days).
    Two 28-day-old wins at weight .25 and two same-day losses at weight 1.0 gave
    0.2, and this cell printed `25% (>=5%, n=4)` - a count nobody observed,
    carrying a Wilson bound computed from it, where the truth was 2-2 and 50%.

    So the pair comes off the row's own `n_wins` / `n_losses`
    (`swing_headline.headline_from_counts`), and a row whose export predates
    those columns - or carries them EMPTY - says so. It is never reconstructed:
    "we did not count this" is a fact, and a made-up 25% is not.

    The weighted rate keeps its place on the table under its own heading; see
    `RECENT_TYPE_COLUMNS`.

    `win_rate_lb` rides along as the sort key `_rank_recent_types` orders by and
    `working_lately.select_leader` ranks on, so the table and the banner cannot
    disagree.
    """
    return _counted_win_rate_rows(rows)


def _setup_type_headline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same treatment for the Setup Types tab - ST2.2, V3 item 1's owed seam.

    `master_avwap_setup_type_stats.csv` carried `target_hit_rate` and
    `stop_rate` - different questions - and no win column, so V3 could not put a
    win rate here without joining one from `master_avwap_tier_outcomes.csv`,
    whose 184 rows collapse to 71 (side, bucket, family, zone) groups: one
    joined rate would have repeated across up to six rows and read as each row's
    own. ST2.2 gave the export its own counts at its own grain, so the cell is
    now the row's own record and nothing is joined in.
    """
    return _counted_win_rate_rows(rows)


def _counted_win_rate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`win_rate_headline` + `win_rate_lb` from a row's INTEGER counts, or a refusal.

    One helper for both tables, so the two tabs can never drift into two
    definitions of a win. O(rows) and pure arithmetic - it runs on the
    `ReadWorker` thread, inside `_read_tracker_exports`, with the rest of the
    read `refresh()` now offloads (G7.2); nothing here touches a widget.
    """
    from swing_headline import format_win_rate, headline_from_counts
    from working_lately import counted_pair

    out: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        pair = counted_pair(row)
        if pair is None:
            enriched["win_rate_headline"] = COUNTS_NOT_EXPORTED
            enriched["win_rate_lb"] = None
            out.append(enriched)
            continue
        wins, losses = pair
        record = headline_from_counts(
            str(row.get("setup_family") or ""),
            wins=wins,
            losses=losses,
            # A scratch is not a loss and is not a win: `Headline.flats` keeps
            # it out of `n` while it stays a MEASURED outcome. Dropping it here
            # would leave the export counting flats and the cell not.
            flats=_int(row.get("n_flats")),
            avg_r=row.get("avg_closed_r"),
        )
        enriched["win_rate_headline"] = (
            format_win_rate(record.as_row()) if record.n else ""
        )
        enriched["win_rate_lb"] = record.win_rate_lb
        out.append(enriched)
    return out


def _rank_recent_types(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # NEW/RISING families with some closed evidence pin to the top (freshly
    # promoted ideas and not-yet-favorite outperformers must never drown under
    # high-sample veterans).
    #
    # R4 B3 changed what orders them BELOW that pin: the Wilson LOWER BOUND on
    # the win rate, not the raw closed count. Sorting by count put the biggest
    # sample on top whatever it measured, and sorting by the raw rate would put a
    # 100%-on-three family above a 62%-on-ninety every time. A family with no
    # graded closes has no bound and sorts under every family that has one -
    # "not measured" is not "measured badly" - with the old keys as tiebreaks.
    def _key(row):
        bound = row.get("win_rate_lb")
        return (
            not (str(row.get("status") or "").strip() and _int(row.get("closed_setups")) >= 2),
            bound is None,
            -(float(bound) if bound is not None else 0.0),
            -_int(row.get("closed_setups")),
            -_float(row.get("avg_closed_r"), -1e9),
            -_int(row.get("tracked_setups")),
        )

    return sorted(rows, key=_key)


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


def _summary_setup_type_rows(
    rows: list[dict[str, Any]], *, limit: int = 8
) -> list[dict[str, Any]]:
    """The Summary card's eight setup types: best by BOUND, across both sides.

    **ST2 fix round.** The tab's own table sorts side-first, which is right for
    a table you scroll - LONG and SHORT are two books. The Summary card shows
    only the first eight rows of that list, so side-first ordering meant it
    showed eight LONG rows and no SHORT one: on the live export the first SHORT
    row sat at index 68 of 117, and a reader of the card would have concluded
    the short book had nothing working. So this card takes its eight across BOTH
    sides by the same Wilson lower bound, and every line shows its side.

    The tab's order is untouched; this is a different question asked of the same
    rows. A row with no bound sorts under every row that has one.
    """

    def _key(row):
        bound = row.get("win_rate_lb")
        return (
            bound is None,
            -(float(bound) if bound is not None else 0.0),
            -_float(row.get("score_delta"), 0.0),
            -_int(row.get("closed_setups")),
        )

    return sorted(rows, key=_key)[:limit]


def _summary_html(panel: SetupTrackerPanel) -> str:
    body = theme.color("text_primary")
    muted = theme.color("text_secondary")
    long_c = theme.color("long")
    short_c = theme.color("short")
    favorite_c = theme.color("favorite")

    parts = [f"<body style='color:{body}; font-size:9pt'>"]
    # ONE computation per page, handed to BOTH renderers - the card and the
    # banner are three lines apart and may never disagree (re-review blocker 1).
    verdicts = panel_verdicts(panel)
    plain = build_plain_english_whats_working(
        current_rows=panel.current_pick_rows,
        short_term_rows=panel.short_term_rows,
        recent_rows=panel.recent_type_rows,
        playbook_rows=panel.playbook_rows,
        short_term_min_samples=SHORT_TERM_MIN_SAMPLES,
        verdicts=verdicts,
    )
    parts.append(f"<div style='border:1px solid {favorite_c}; padding:7px; margin-bottom:7px'>")
    parts.append(f"<h3 style='margin:0; color:{favorite_c}'>{_esc(plain['headline'])}</h3><ul>")
    parts.extend(f"<li>{_esc(item)}</li>" for item in plain["bullets"])
    parts.append(f"</ul><div style='color:{muted}'>{_esc(plain['caution'])}</div></div>")
    parts.append(_best_now_banner_html(panel, verdicts))
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
    for row in _summary_setup_type_rows(panel.setup_type_rows, limit=8):
        edge = _float(row.get("avg_closed_r_edge"), 0.0)
        edge_color = long_c if edge >= 0 else short_c
        parts.append(
            f"<div><b>{_esc(row.get('side'))}</b> {_esc(row.get('setup_family'))} "
            f"<span style='color:{muted}'>{_esc(row.get('win_rate_headline'))}</span> "
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


def _last_completed_session_or_today():
    """The last completed exchange session, or today when the calendar refuses.

    The calendar raises outside its validated range. A banner is not worth a
    traceback, and falling back to today only ever makes the freshness test
    STRICTER, never looser.
    """
    from datetime import datetime

    import market_calendar

    try:
        return market_calendar.last_completed_session(
            datetime.now(market_calendar.MARKET_TZ)
        )
    except Exception:
        return datetime.now(market_calendar.MARKET_TZ).date()


def _remembered_verdict(panel, rows, *, kind: str, last_completed_session, **kwargs):
    """`select_leader`, with the panel's last FRESH verdict as `previous`.

    This is what makes `last_reliable_reading` reachable from the desk. The
    remembered verdict is replaced only when a new one is a `leader` off fresh
    evidence, so a stale reading never overwrites the reading it is standing in
    for, and a `no_clear_leader` never becomes a leader by being remembered.
    """
    from working_lately import select_leader

    previous = panel._last_fresh_verdicts.get(kind)
    verdict = select_leader(
        rows,
        kind=kind,
        last_completed_session=last_completed_session,
        previous=previous,
        **kwargs,
    )
    if verdict.state == "leader":
        panel._last_fresh_verdicts[kind] = verdict
    return verdict


def _verdict_block_html(
    verdict, *, label: str, muted: str, side_color
) -> str:
    """One horizon's line, straight off a `working_lately.LeaderVerdict`.

    Four states, four sentences, and the word "leader" never appears beside a
    row that is not one.

    **`label` is the HORIZON only**; the state is appended here from the verdict
    (re-review blocker 2). The caller used to hardcode "2-session discovery",
    which then sat over a real `leader` verdict and called it discovery.
    """
    from working_lately import discovery_basis_phrase, discovery_note, verdict_label_suffix

    label = f"{label}, {verdict_label_suffix(verdict)}"
    if verdict.state in {"leader", "last_reliable_reading"} and verdict.leader is not None:
        row = verdict.leader
        stamp = (
            ""
            if verdict.state == "leader"
            else f" <span style='color:{muted}'>[as of {_esc(verdict.as_of)}]</span>"
        )
        return (
            f"<div><b>{_esc(label)}:</b> "
            f"<span style='color:{side_color(row.get('side'))}'><b>{_esc(row.get('side'))}</b></span> "
            f"<b>{_esc(row.get('setup_family'))}</b>{stamp}"
            f"<div style='color:{muted}; margin-left:14px'>{_esc(verdict.policy_line)}</div></div>"
        )

    discovery = verdict.coverage.get("discovery_leader")
    if discovery is not None:
        wins = _int(discovery.get("n_wins"))
        losses = _int(discovery.get("n_losses"))
        reason = verdict.coverage.get("discovery_reason") or ""
        head = (
            f"No leader at the n={verdict.coverage.get('min_n')} floor"
            if reason == "floor"
            else "No leader"
        )
        # A row kept out for being OLD is not thin - it has the evidence, it is
        # just not current, and calling it thin misnames the gate (re-review
        # advisory 1).
        basis = discovery_basis_phrase(reason)
        # ...and the extra sentence a discovery row earns, which for exactly
        # one reason is "the export carries no measured session". It lives
        # beside the phrase that names the same gate, so a caller can no longer
        # pass one that does not match (re-check advisory 2). Printing it beside
        # "58 sessions behind" was two contradictory facts in one line.
        note = discovery_note(reason)
        return (
            f"<div style='color:{muted}'><b>{_esc(label)}:</b> {_esc(head)} - "
            f"{_esc(basis)}: "
            f"<span style='color:{side_color(discovery.get('side'))}'>"
            f"<b>{_esc(discovery.get('side'))}</b></span> "
            f"<b>{_esc(discovery.get('setup_family'))}</b> "
            f"(n={wins + losses}), discovery only."
            f"{(' ' + _esc(note)) if note else ''}"
            f"<div style='margin-left:14px'>{_esc(verdict.reason)}</div></div>"
        )
    return (
        f"<div style='color:{muted}'><b>{_esc(label)}:</b> "
        f"{_esc(_VERDICT_HEADLINES.get(verdict.state, 'No verdict'))} - "
        f"{_esc(verdict.reason)}</div>"
    )


#: The one spelling of each state, so the banner never invents a fifth.
_VERDICT_HEADLINES = {
    "leader": "Leader",
    "no_clear_leader": "No clear leader",
    "last_reliable_reading": "Last reliable reading",
    "no_evidence": "No clear leader yet",
}


def panel_verdicts(panel: SetupTrackerPanel) -> dict[str, Any]:
    """Both horizons' verdicts, computed ONCE for the whole Summary page.

    **Re-review blocker 1.** The banner went through `_remembered_verdict` (which
    carries a `previous`, so a stale refresh becomes `last_reliable_reading`)
    while the plain-English card called `select_leader` directly with no
    `previous` - so on the stale path the card printed *"no clear leader.
    Leading on thin evidence, SHORT general on n=88 - discovery only"* directly
    above the banner's *"SHORT general [last reliable reading, as of
    2026-09-04]"*. Two renderers computing the same thing will disagree the
    moment one of them gains an argument; there is now exactly one computation
    per page and both renderers are handed the same objects.
    """
    from working_lately import short_term_evidence_rows, verdicts_from_payload

    last_session = _last_completed_session_or_today()
    # **ST6.4 re-review.** The desk's SHARED snapshot feeds THIS function, so
    # there is still exactly one computation per page: the card, the banner and
    # the strip above the M5 list cannot split, whichever of them renders first.
    # The panel's own `select_leader` read is the labelled FALLBACK and only
    # ever answers the horizons the snapshot does not carry.
    payload = getattr(panel, "_working_lately_snapshot", None) or {}
    shared = verdicts_from_payload(payload) if payload else {}
    swing = shared.get("swing_trade_r")
    if swing is not None:
        return {
            "swing": swing,
            # The 2-session block is NOT in the snapshot (the short-horizon
            # export is a different grain), so it stays a panel read. It is the
            # only thing on this page that is one when a snapshot is present,
            # and the banner says so beside it rather than under one label for
            # the whole page.
            "swing_short_term": _remembered_verdict(
                panel,
                short_term_evidence_rows(panel.short_term_rows),
                kind="swing_short_term",
                last_completed_session=last_session,
                min_n=SHORT_TERM_MIN_SAMPLES,
            ),
            "swing_favorable": shared.get("swing_favorable"),
            "daytrade_held_run": shared.get("daytrade_held_run"),
            "snapshot": payload,
        }
    return {
        "swing": _remembered_verdict(
            panel,
            panel.recent_type_rows,
            kind="swing",
            last_completed_session=last_session,
        ),
        # The two-session block reads the same function with its own floor - the
        # panel's `SHORT_TERM_MIN_SAMPLES`, passed in rather than re-declared.
        # Since the ST2 fix round the short-horizon export carries its own
        # counts and its own MEASURED session, so this block is judged for
        # freshness too; a row from an OLDER file stays undated.
        "swing_short_term": _remembered_verdict(
            panel,
            short_term_evidence_rows(panel.short_term_rows),
            kind="swing_short_term",
            last_completed_session=last_session,
            min_n=SHORT_TERM_MIN_SAMPLES,
        ),
    }


def _best_now_banner_html(panel: SetupTrackerPanel, verdicts: dict[str, Any] | None = None) -> str:
    """One unmissable line per horizon, and it agrees with the table beneath it.

    **ST2.3.** This used to pick `max(avg_closed_r)` over any row with three
    closed setups, across the live AND study namespaces, while the table
    underneath ranked by the Wilson lower bound - so on the same screen the
    banner crowned `fat_but_wide` (+2.50R on 90, bound 0.497) and the table
    listed `tight_and_hot` first (bound 0.627 on 30), and a three-example STUDY
    with a big R could be presented as the desk's best performer. It now reads
    `working_lately.select_leader` on the SAME rows in the SAME order, so the
    two cannot disagree; a study is never the leader and its exclusion is
    counted out loud; and when nothing is eligible the banner says which gate
    closed instead of crowning whoever was left.
    """
    muted = theme.color("text_secondary")
    favorite_c = theme.color("favorite")
    long_c = theme.color("long")
    short_c = theme.color("short")

    def _side_color(side: Any) -> str:
        return long_c if str(side or "").upper() == "LONG" else short_c

    if verdicts is None:
        verdicts = panel_verdicts(panel)
    swing_verdict = verdicts["swing"]
    short_verdict = verdicts["swing_short_term"]

    parts = [
        f"<div style='border:1px solid {favorite_c}; padding:6px; margin-bottom:6px'>",
        f"<b style='color:{favorite_c}; font-size:10pt'>BEST PERFORMING RIGHT NOW</b>",
    ]
    # **No conditional.** `_verdict_block_html` renders all four states, and the
    # guard that used to sit here ("a discovery row OR a leader, else a
    # hardcoded sentence") dropped `no_clear_leader` - today's live short-term
    # state, 12 eligible families with the top two 0.001 of bound apart - into
    # "not enough 2-session samples yet", which was false AND contradicted the
    # card three lines above saying "no clear leader". A renderer that has a
    # verdict must render the verdict.
    parts.append(
        _verdict_block_html(
            short_verdict,
            label="Short-term (1-2d)",
            muted=muted,
            side_color=_side_color,
        )
    )
    parts.append(
        _verdict_block_html(
            swing_verdict,
            label="Swing (30d realized)",
            muted=muted,
            side_color=_side_color,
        )
    )
    excluded = _int(swing_verdict.coverage.get("studies_excluded"))
    if excluded:
        parts.append(
            f"<div style='color:{muted}'>{excluded} study famil"
            f"{'y' if excluded == 1 else 'ies'} excluded from the leader - an "
            f"unpromoted idea never leads, whatever its R.</div>"
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
    parts.append(_banner_source_html(verdicts, muted=muted))
    parts.append("</div>")
    return "".join(parts)


def _banner_source_html(verdicts: dict[str, Any], *, muted: str) -> str:
    """Which reading this banner rendered, and its id when there is one (ST6.4).

    An unlabelled fallback is how the desk grew two answers to one question in
    the first place, so the panel's own pass says `panel read` out loud. When
    the shared snapshot is present the two extra kinds it carries are printed
    here too - the favorable-direction rate and the day-trade held x ran - so
    the page shows the whole reading rather than the half of it that happens to
    match the tables underneath.
    """
    import working_lately

    payload = verdicts.get("snapshot") or {}
    if not payload:
        return (
            f"<div style='color:{muted}'>Source: <b>panel read</b> - this page read the "
            f"exports itself because the desk's Working-lately service has not handed "
            f"it a snapshot. A desk session shows the shared reading and its id.</div>"
        )
    bits: list[str] = []
    for kind, label in (
        ("swing_favorable", "Swing (favorable direction, percent move)"),
        ("daytrade_held_run", "Day trade (held x ran)"),
    ):
        verdict = verdicts.get(kind)
        if verdict is None:
            continue
        bits.append(
            f"<div style='color:{muted}'><b>{_esc(label)}:</b> "
            f"{_esc(working_lately.kind_phrase(payload, kind, label))}</div>"
        )
    caveats = ", ".join(
        working_lately.observational_caveat(payload, kind)
        for kind in working_lately.SNAPSHOT_KINDS
        if verdicts.get(kind) is not None or kind == "swing_trade_r"
    )
    bits.append(
        f"<div style='color:{muted}'>Source: the desk's shared Working-lately reading "
        f"({_esc(working_lately.snapshot_stamp(payload))}). The 2-session block above is "
        f"this page's OWN read - the short-horizon export is a different grain and the "
        f"snapshot does not carry it. {_esc(caveats)} - nothing here is proven.</div>"
    )
    return "".join(bits)


def _best_type_label(rows: list[dict[str, Any]]) -> str:
    """The tile is "Best Type EDGE", so it picks the biggest `score_delta`.

    **It reads its own meaning, not the table's first row** (ST2 fix round).
    It used to take `rows[0]`, which was fine only while the tab happened to be
    sorted by `score_delta`; ST2.2 changed that sort to the Wilson lower bound
    inside each side, and the tile silently changed with it - on the live export
    from `SHORT +23` to `LONG +14`, a number the trader would have read as the
    edge falling by nine points when nothing had moved at all. A tile that
    borrows another surface's ordering has no meaning of its own.
    """
    if not rows:
        return "-"
    row = max(rows, key=lambda item: _float(item.get("score_delta"), 0.0) or 0.0)
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
        CONTROL_DISCOVERY_STATS_FILE,
        STUDY_DISCOVERY_STATS_FILE,
        EXIT_FRAMEWORK_STATS_FILE,
        SETUP_PLAYBOOKS_FILE,
        SHORT_HORIZON_FILE,
        MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE,
        MASTER_AVWAP_TIER_LIST_FILE,
        MASTER_AVWAP_TIER_PERFORMANCE_FILE,
        MASTER_AVWAP_TIER_CATCH_RATE_FILE,
        HUMAN_FOCUS_PERFORMANCE_FILE,
        HUMAN_FOCUS_OUTCOMES_FILE,
    ]


#: Packet M3.2 (2026-09-05). This page reads TWO stores with two different
#: clocks: the setup-tracker snapshot (written only by a pass that actually
#: replayed the tracker) and the `scan_factor_*` exports (rewritten by every
#: scan). Showing one mtime for both made the tables look as fresh as the last
#: scan even on the days the tracker write was refused outright - which, with
#: the daily-bar pin in force, was every day. Two clocks, both named.
TRACKER_CLOCK_UNKNOWN = "unknown"


def tracker_clock_sentence(saved_at: str, saved_by: str, scan_factor_text: str) -> str:
    """`Tracker as of <saved_at> (<saved_by>); scan factors as of <mtime>`.

    Pure, and blank-preserving: an export with no stamp reads ``unknown``, never
    the current time and never the scan-factor clock. A page that borrows one
    store's freshness for another is the defect this replaces.
    """
    stamp = str(saved_at or "").strip() or TRACKER_CLOCK_UNKNOWN
    writer = str(saved_by or "").strip() or TRACKER_CLOCK_UNKNOWN
    factors = str(scan_factor_text or "").strip() or TRACKER_CLOCK_UNKNOWN
    return f"Tracker as of {stamp} ({writer}); scan factors as of {factors}"


def _first_non_empty(rows: list[dict[str, Any]], key: str) -> str:
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def expired_unmeasured_sentence(rows: list[dict[str, Any]]) -> str:
    """`N expired unmeasured, excluded` - or nothing when there are none.

    The count comes from the export's OWN column; this never opens the 1.1 GB
    tracker JSON.
    """
    total = 0
    for row in rows:
        try:
            total += int(float(str(row.get("n_expired_unmeasured") or "0").strip() or 0))
        except (TypeError, ValueError):
            continue
    if total <= 0:
        return ""
    return f"{total} expired unmeasured, excluded"


def setup_type_population_sentence(rows: list[dict[str, Any]]) -> str:
    """What the Setup Types tab's Win % column MEANS, above the table - ST2.2.

    The trader's requirement: *"Show side, outcome meaning, actual measurement
    horizon, window, and coverage."* A win column with no stated outcome
    definition is the same defect as a rate with no n - a control row, a study
    row and a pick row all look identical in a table, and so do a target-hit
    rate and a win rate.

    Every number here is summed from the export's OWN columns; this never opens
    the 1.1 GB tracker JSON. M3's `expired unmeasured, excluded` clause is kept
    verbatim at the end (gate #72 clause 5 reads it).
    """
    from working_lately import counted_pair

    wins = losses = flats = unmeasured = pending = 0
    counted_rows = 0
    for row in rows:
        # `counted_pair` is the ONE reader of these columns, and it is strict:
        # a MISSING or BLANK cell is None, while an exported integer 0 is a
        # count. A `str(...) or ""` truthiness check reads a real 0 as "not
        # exported" and would silently drop every row that went 0-0.
        pair = counted_pair(row)
        if pair is None:
            continue
        counted_rows += 1
        wins += pair[0]
        losses += pair[1]
        flats += _int(row.get("n_flats"))
        unmeasured += _int(row.get("n_unmeasured"))
        pending += _int(row.get("n_pending"))
    expired = expired_unmeasured_sentence(rows) or "0 expired unmeasured"
    if not counted_rows:
        head = (
            f"{len(rows):,} setup-type row(s); no win/loss counts in this export "
            f"yet - the Win % column fills on the next tracker write."
        )
        return f"{head} {expired}".strip()
    kind = _first_non_empty(rows, "outcome_kind") or "trade_r_representative_exit"
    head = (
        f"{counted_rows:,} setup-type row(s) at (side, bucket, family, zone, "
        f"retest, compression) grain, over ALL HISTORY in the tracker (this tab "
        f"has no lately window; the Last 30 Days tab is the windowed view); "
        f"Win % is each row's OWN {wins:,} win(s) / {losses:,} loss(es) - "
        f"outcome {kind} (the sign of the representative protective-stop "
        f"scenario's closed R) - with {flats:,} flat, {unmeasured:,} "
        f"closed-unmeasured and {pending:,} still open, none of them counted as "
        f"a loss. Sorted by the Wilson lower bound inside each side."
    )
    return f"{head} {expired}".strip()


def _latest_mtime_text(paths: list[Path]) -> str:
    """The newest mtime among ``paths``, rendered MARKET-LOCAL with its offset.

    It is shown beside the tracker's own `saved_at`, which is market-local with
    an offset. Rendering this one machine-local and unlabelled put the two
    clocks in two zones on one line: on this desk (PT) the pair read three
    hours apart when they were the same instant, which is worse than showing
    one clock. One zone, and the offset is printed so it cannot be mistaken.
    """
    existing = [path for path in paths if path.exists()]
    if not existing:
        return "never"
    latest = max(path.stat().st_mtime for path in existing)
    from market_calendar import MARKET_TZ

    return (
        datetime.fromtimestamp(latest, tz=timezone.utc)
        .astimezone(MARKET_TZ)
        .replace(microsecond=0)
        .isoformat()
    )


def _tier_rank(value: Any) -> int:
    return {"S": 0, "A": 1, "B": 2, "C": 3}.get(str(value or "").upper(), 9)


def _column_index(columns: tuple[tuple[str, str], ...], key: str) -> int:
    """Where a column sits in its own tuple, BY KEY (packet G2b.2).

    Raises rather than guessing: a width rule that silently pointed at the
    wrong column would be invisible until the trader read a clipped table, and
    a construction-time `KeyError` names the typo on the spot.
    """
    for index, (column_key, _label) in enumerate(columns):
        if column_key == key:
            return index
    raise KeyError(f"{key!r} is not a column of this table")


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
