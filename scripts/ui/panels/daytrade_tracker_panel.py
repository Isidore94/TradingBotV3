from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
)

from bounce_bot_lib.learning import BOUNCE_LEARNING_STATE_FILE, load_bounce_learning_state
from project_paths import INTRADAY_BOUNCE_OUTCOMES_FILE
from ui.models.tracker_table_model import ROW_ROLE, TrackerSortProxyModel, TrackerTableModel
from ui.widgets.data_table import DataTable
from ui.widgets.kpi_tile import KpiTile
from ui.widgets.research_explanation_view import ResearchExplanationView
from ui.widgets.section_header import SectionHeader
from swallowed import note_swallowed

#: V3 item 2 (decision 0016 answer 4): *"the intraday level holds, then the name
#: runs. Rank by maximum favourable excursion - the most the move offered - not
#: by any exit; exiting well is the trader's job."*
#:
#: So HELD and RAN lead this table, and the champion tier stays as a column. The
#: two are different questions - the tier says whether the desk should alert on
#: the segment at all, and these say what the alert offered once the level held -
#: and the Verdict column already says which is which.
#:
#: R4 A10: the column is "Held 30m" now, and the label is not a rename but a
#: correction. V3 labelled it "Held" because the number came from the
#: aggregator's own `stop_rate`, over ITS window and over every row rather than
#: the ones that held - a second formula under the headline key. The number now
#: comes from `held_run_score`, which asks the thirty-minute question of the raw
#: outcome log, so the column may finally say what it measures.
PERFORMANCE_COLUMNS = (
    ("direction", "Side"),
    ("segment", "Segment"),
    ("held_rate", "Held 30m"),
    ("held_run_score", "Held x Ran"),
    # Packet Q1: coverage BESIDE the headline - n_measured / n. A hold rate over
    # 35 measured of 41 alerts is a different fact from one over 41 of 41, and
    # until 2026-09-04 the unmeasured ones were silently counted as held.
    ("measured", "Measured"),
    # R4 B4: the champion tier the header comment above has promised since V3 and
    # the table never carried. PROVEN / MUTED / active from the bounce learning
    # state, joined on the SAME (dimension, direction, segment) key the headline
    # uses; blank for a segment the state has never seen, because "not tracked"
    # and "tracked and unremarkable" are different facts.
    ("champion_tier", "Tier"),
    # S1 + S10c: the setup grades on the Bounce Types tab - the 1:1 bracket
    # grade (+1R before -1R) the badges sort by, the 2R grade (+2R before -1R),
    # EOD close R and the share that reached +2R, from `setup_grades`.
    ("grade_1r", "1:1 bracket"),
    ("grade_2r", "2R grade"),
    ("eod_r_mean", "EOD R"),
    ("eod_r_median", "EOD med R"),
    ("reach_2r_rate", "Reach 2R"),
    ("grade_n", "Grade n"),
    # S16: the same 1:1 grade inside the trader's current regime ("untested in
    # this regime" when it has no alerts there), then every regime, current
    # first, ending with "all regimes" = the pooled grade the two columns above show.
    ("regime_now", "This regime"),
    ("regime_all", "By regime"),
    # S11: when the family usually peaks and +1R-or-60-min vs holding, from
    # `exit_windows.json` (facts only, never enforced).
    ("exit_by", "Exit by"),
    ("sample_count", "N"),
    ("avg_close_r", "Avg R"),
    ("median_close_r", "Med R"),
    ("avg_mfe_r", "MFE R"),
    ("avg_mae_r", "MAE R"),
    ("positive_eod_rate", "Win"),
    ("target_1r_rate", "1R Hit"),
    ("target_2r_rate", "2R Hit"),
    ("stop_rate", "Stop"),
    # R4 B4: NAMED. This is the aggregator's `edge_score` verdict, computed from
    # average R over its own window - a different question from Held x Ran, which
    # sits three columns to the left. Two verdicts under one table with neither
    # naming its basis is how a reader ends up believing they agree.
    ("recommendation", "Verdict (edge score)"),
    ("example_symbols", "Examples"),
)

LEARNING_COLUMNS = (
    ("dimension", "Dimension"),
    ("direction", "Side"),
    ("segment", "Segment"),
    ("sample_count", "N"),
    ("avg_close_r", "Avg R"),
    ("score_delta", "Delta"),
    ("stop_rate", "Stop"),
    ("target_1r_rate", "1R Hit"),
    ("status", "Status"),
)

#: S9: the shadow tier beside the live tier, per recent alert (evidence only;
#: never read by an alert, a sort, the Show filter or the phone).
SHADOW_S9_COLUMNS = (
    ("trade_date", "Date"),
    ("time_local", "Time"),
    ("symbol", "Symbol"),
    ("direction", "Side"),
    ("bounce_types", "Bounce types"),
    ("tier", "Tier"),
    ("composite_r", "Tier R"),
    ("shadow_s9_tier", "Shadow S9"),
    ("shadow_s9_composite_r", "Shadow S9 R"),
)
#: How many recent alerts the Shadow S9 tab lists (newest first).
SHADOW_S9_MAX_ROWS = 300

PERCENT_KEYS = {
    "positive_eod_rate",
    "target_1r_rate",
    "target_2r_rate",
    "stop_rate",
    "held_rate",
    "reach_2r_rate",
}
SIGNED_KEYS = {
    "eod_r_mean",
    "eod_r_median",
    "avg_close_r",
    "median_close_r",
    "avg_mfe_r",
    "avg_mae_r",
    "score_delta",
    "held_run_score",
}

#: The column this table sorts by when it is first shown. V3 item 2 makes it the
#: day-trade headline rather than the sample count - a tracker that opens sorted
#: by N answers "what has the most data", which is not a question the trader has.
DEFAULT_PERFORMANCE_SORT_KEY = "held_run_score"

# ---------------------------------------------------------------------------
# "My decisions" (P2 item 3): the same shape, over the trader's OWN choices.
#
# Everything above is what the BOT measured. `review_preference_state.json`
# carries what the TRADER did with it - P(take | shown) per segment, and the R
# of what they took against the R of what they passed - and until now it had no
# surface at all outside a text report nobody opens. It is the same question
# these tabs already answer, asked of the other half of the loop, so it belongs
# in the same tab strip rather than a new page.
#
# Read-only over a file the review-learning pass writes. Nothing here reaches a
# detector, score, alert, Focus list, review queue or review_policy.json.
# ---------------------------------------------------------------------------
DECISION_COLUMNS = (
    ("segment", "Segment"),
    ("shown", "Shown"),
    ("take", "Takes"),
    ("take_rate", "Take rate"),
    # R4 B4: the DAY-TRADE HEADLINE, on the trader's own decisions. These tabs
    # graded what was taken and what was passed in mean R alone, and decision
    # 0016 answer 4 makes MFE-after-a-held-level the headline on this side. From
    # the one helper (`apply_held_and_ran`) and the one module, joined on the
    # pooled-direction cell because this state records no side within a
    # dimension.
    ("held_rate", "Held 30m"),
    ("held_run_score", "Held x Ran"),
    ("measured", "Measured"),
    ("taken_r", "Taken R"),
    ("taken_n", "(n)"),
    ("passed_r", "Passed R"),
    ("passed_n", "(n)"),
    ("gap", "Gap"),
    ("probation", "Probation"),
)

#: Dimensions carried by the scoreboard, in display order. A dimension the
#: state does not carry yields an empty tab rather than being dropped: an
#: absent dimension is a measurement that has not happened, and a tab that
#: silently disappears reads as one that never existed.
DECISION_TABS = (
    ("bounce_type", "Bounce Types"),
    ("alert_kind", "Alert Kind"),
    ("tier", "Tier"),
    ("side", "Side"),
    ("time_bucket", "Time of Day"),
    ("market_environment", "Environment"),
    ("rrs_alignment", "RRS"),
    ("rvol_bucket", "RVOL"),
    ("setup_family", "Swing Family"),
    ("setup_tag", "Swing Tags"),
    ("bucket", "Swing Bucket"),
    ("dislike_reason", "Veto Reasons"),
    ("expected_r_band", "Expected R"),
)

DECISION_PERCENT_KEYS = {"take_rate"}
DECISION_SIGNED_KEYS = {"taken_r", "passed_r", "gap"}


#: Bounce types the desk no longer records, with the day recording stopped. Their
#: history stays on disk; the family row says why no new rows arrive.
RECORDING_OFF_SINCE = {"h1_blue_after_red": "2026-09-26"}


def recording_off_text(dimension: str, segment: str) -> str:
    """'off since <day>' for a bounce-type family that is no longer recorded, else ''."""
    if str(dimension or "").strip() != "bounce_type":
        return ""
    since = RECORDING_OFF_SINCE.get(str(segment or "").strip())
    return f"off since {since}" if since else ""


def _probation_types() -> frozenset[str]:
    """The M5 signal engines, which are on probation and not champions.

    Set membership over the two dicts that already exist - `BOUNCE_TYPE_DEFAULTS`
    is the established taxonomy, `M5_SIGNAL_TYPE_DEFAULTS` the R5 engines still
    earning their place. A row's badge is which dict its bounce_type is in and
    nothing else: no threshold, no judgement, no second list to maintain.

    An import failure yields an empty set, so every row simply carries no badge
    - the wrong direction to guess in would be labelling a champion "probation".
    """
    try:
        from bounce_bot_lib.legacy import BOUNCE_TYPE_DEFAULTS, M5_SIGNAL_TYPE_DEFAULTS

        return frozenset(set(M5_SIGNAL_TYPE_DEFAULTS) - set(BOUNCE_TYPE_DEFAULTS))
    except Exception:  # noqa: BLE001 - a badge is never worth a broken tab
        return frozenset()


def decision_rows(state, dimension: str, probation=frozenset()) -> list[dict]:
    """One dimension's segments from the scoreboard state, ready to render.

    Reformatted, never derived - with ONE exception that is stated: `gap` is
    `taken.r_avg - passed.r_avg`, the subtraction the trader would otherwise do
    by eye across two columns. It is shown only when BOTH sides carry a
    measured average, so it can never be a difference against an absent number.

    A missing average stays None and renders blank. A segment with no measured
    R on either side is still listed, because "you saw 40 of these and none has
    a graded outcome yet" is a real answer and dropping the row would hide it.
    """
    if not isinstance(state, dict):
        return []
    dimensions = state.get("dimensions")
    if not isinstance(dimensions, dict):
        return []
    table = dimensions.get(dimension)
    if not isinstance(table, dict):
        return []
    rows: list[dict] = []
    for segment, stats in table.items():
        if not isinstance(stats, dict):
            continue
        taken = stats.get("taken") if isinstance(stats.get("taken"), dict) else {}
        passed = stats.get("passed") if isinstance(stats.get("passed"), dict) else {}
        taken_r = taken.get("r_avg")
        passed_r = passed.get("r_avg")
        gap = None
        if isinstance(taken_r, (int, float)) and isinstance(passed_r, (int, float)):
            gap = round(float(taken_r) - float(passed_r), 3)
        rows.append(
            {
                "dimension": dimension,
                "segment": str(segment),
                "shown": stats.get("shown"),
                "take": stats.get("take"),
                "take_rate": stats.get("take_rate"),
                "taken_r": taken_r,
                "taken_n": taken.get("r_n"),
                "passed_r": passed_r,
                "passed_n": passed.get("r_n"),
                "gap": gap,
                "probation": "probation" if str(segment) in probation else "",
            }
        )
    rows.sort(key=lambda row: -_float(row.get("shown"), -1.0))
    return rows


# Dimension tabs shown, in display order (dimension key -> tab label).
DIMENSION_TABS = (
    ("bounce_type", "Bounce Types"),
    ("bounce_combo", "Combos"),
    ("time_bucket", "Time of Day"),
    ("market_environment", "Environment"),
    ("rrs_alignment", "RRS"),
    ("master_avwap_focus", "Swing Focus"),
    ("master_avwap_priority_bucket", "Swing Bucket"),
    ("master_avwap_setup_family", "Swing Family"),
    ("master_avwap_swing_trait", "Swing Traits"),
)


def _explanation_identity(kind: str, row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """What names the row an explanation is about (packet G4.2).

    The columns that name the SEGMENT, read from the row dict and never from the
    display text: the dimension, plus the `direction|segment` pair the learning
    store itself keys a segment by (a `long vwap` row and a `short vwap` row are
    two different measurements). Values are stringified because a performance
    row arrives from `csv.DictReader` as strings and a decision row does not.
    """
    return (
        str(kind or ""),
        str(row.get("dimension") or ""),
        str(row.get("direction") or ""),
        str(row.get("segment") or ""),
    )


class DaytradeTrackerPanel(QFrame):
    """Research tab: BounceBot's measured performance and the live learning state.

    Everything the alert-time learning loop knows is on display here: per-segment
    R stats from the outcome tracker, and the tiers/mutes/deltas currently applied
    to alerts. Refresh re-aggregates the full candidate/outcome history.
    """

    statusChanged = Signal(str)
    _refreshFinished = Signal(str)
    #: The decisions read lands here, off the worker thread. `object` because
    #: the payload is a plain dict and Qt must not try to marshal it.
    _decisionsLoaded = Signal(object)
    #: R4 A10: `held_run_score`'s own marginals, read on a worker for the same
    #: reason - the outcome log is ~90 MB and the panel opens on the Qt thread.
    _heldRunLoaded = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        #: G7.1: the first show pays for the first read, never the constructor.
        self._loaded_once = False
        self._refresh_thread: threading.Thread | None = None

        self._decisions_thread: threading.Thread | None = None
        self._held_run_thread: threading.Thread | None = None
        #: `{(dimension, direction, segment): summary}`. Empty until the first
        #: read lands, and empty means BLANK - never a substitute number.
        self._held_run_summaries: dict = {}
        #: The bounce learning state, cached from `reload_from_disk` so the
        #: held/ran worker's callback can re-join the tier without a file read on
        #: the Qt thread (R4 B4).
        self._learning_state: dict = {}
        self._performance_rows: list[dict[str, Any]] = []
        #: The persisted setup grades payload, read on the held/ran worker.
        self._setup_grades: dict = {}
        self._regime_grades: dict = {}
        #: S11: the persisted exit-window payload, read on the same worker.
        self._exit_windows: dict = {}

        self.refresh_button = QPushButton("Re-aggregate Outcomes")
        self.refresh_button.setObjectName("PrimaryButton")
        self.refresh_button.clicked.connect(self.start_refresh)
        self.decisions_button = QPushButton("Refresh my decisions")
        self.decisions_button.setToolTip(
            "Rebuild the review-preference scoreboard from the decision log if "
            "it has gone stale, then reload these tabs. Local file reads only - "
            "nothing is fetched and nothing about the bot changes."
        )
        self.decisions_button.clicked.connect(self.start_decisions_refresh)
        self.decisions_status = QLabel("")
        self.decisions_status.setObjectName("MutedLabel")
        self.decisions_status.setWordWrap(True)
        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")

        self.episodes_tile = KpiTile("Measured Segments", "0")
        self.proven_tile = KpiTile("Proven Live Triggers", "0", tone="favorite")
        self.muted_tile = KpiTile("Muted Segments", "0", tone="short")
        self.best_tile = KpiTile("Best Segment", "-", tone="long")
        self.fresh_tile = KpiTile("Outcomes Updated", "-")

        self.tabs = QTabWidget()
        self._dimension_tables: dict[str, tuple[DataTable, TrackerTableModel]] = {}
        for key, label in DIMENSION_TABS:
            table, model = self._make_table(PERFORMANCE_COLUMNS)
            self._dimension_tables[key] = (table, model)
            # V3 item 2: OPEN ON THE HEADLINE. Sorted descending by
            # `held_run_score` - did the level hold, and then how far did it run -
            # rather than by the sample count. A tracker that opens sorted by N
            # answers "what has the most data", which is not a question the
            # trader has.
            self._apply_default_sort(table)
            self.tabs.addTab(table, label)
            table.clicked.connect(
                lambda index, dimension=key: self._show_row_explanation(
                    index, "daytrade_performance", dimension=dimension
                )
            )
        self.learning_table, self.learning_model = self._make_table(LEARNING_COLUMNS)
        self.tabs.addTab(self.learning_table, "Live Alert Rules")
        self.learning_table.clicked.connect(
            lambda index: self._show_row_explanation(index, "daytrade_learning")
        )
        self.shadow_s9_model = TrackerTableModel(
            SHADOW_S9_COLUMNS,
            signed_keys={"composite_r", "shadow_s9_composite_r"},
            numeric_keys={"composite_r", "shadow_s9_composite_r"},
        )
        shadow_proxy = TrackerSortProxyModel(self)
        shadow_proxy.setSourceModel(self.shadow_s9_model)
        self.shadow_s9_table = DataTable()
        self.shadow_s9_table.setModel(shadow_proxy)
        self.shadow_s9_table.setShowGrid(False)
        self.tabs.addTab(self.shadow_s9_table, "Shadow S9")

        # "My decisions" - the same question over the trader's own choices,
        # in its own tab strip beside the bot's. One tab per scoreboard
        # dimension, mirroring the tracker tabs above.
        self.decisions_tabs = QTabWidget()
        self._decision_tables: dict[str, tuple[DataTable, TrackerTableModel]] = {}
        for key, label in DECISION_TABS:
            table, model = self._make_decision_table()
            self._decision_tables[key] = (table, model)
            self.decisions_tabs.addTab(table, label)
        self.tabs.addTab(self._decisions_page(), "My Decisions")

        self.explanation_view = ResearchExplanationView(self)
        # Packet G4.2: a tab change is a CONTEXT change, and an explanation
        # never outlives its context - the GUI review of 2026-09-06 found the
        # `lrsi_cross50` explanation still standing over the Combos tab with no
        # combo selected. Connected after both strips are built so the addTab
        # calls above do not fire it during construction.
        self.tabs.currentChanged.connect(self._on_context_tab_changed)
        # Belt and braces: in the live GUI the OUTER strip changes first when
        # the trader leaves My Decisions, so this second connection is only
        # reached for a move BETWEEN the My Decisions sub-tabs - which is a
        # context change of its own and must clear the pane too.
        self.decisions_tabs.currentChanged.connect(self._on_context_tab_changed)

        self._refreshFinished.connect(self._on_refresh_finished)
        self._decisionsLoaded.connect(self._on_decisions_loaded)
        self._heldRunLoaded.connect(self._on_held_run_loaded)
        self._build_layout()

    # ------------------------------------------------------------------
    # First load on first show (G7.1, the Market Journal idiom)
    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Read the first time the page is actually looked at.

        `reload_from_disk()` is a CSV parse plus a JSON read plus every
        dimension model rebuilt, and the scoreboard read behind it is a 34 KB
        JSON on the home folder. The desk builds every left-nav panel at
        startup and most are never opened, so this page paid all of it for a
        tab nobody selected. A `QTabWidget` child gets its `showEvent` only
        when its tab is chosen, so Research's first paint costs one child's
        load rather than nine.
        """
        super().showEvent(event)
        if self._loaded_once:
            return
        self._loaded_once = True
        self.reload_from_disk()
        # Off the Qt thread from the first paint: the scoreboard is a 34 KB
        # JSON on the home folder, and reading it on the render path is the
        # drip these panels have been audited for twice.
        self.start_decisions_refresh(rebuild=False)

    def _build_layout(self) -> None:
        header = SectionHeader(
            "Day Trade Tracker",
            "BounceBot outcomes by segment (R-based, from the intraday tracker) and the live tier/mute rules applied to alerts.",
        )
        header.add_action(self.refresh_button)

        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(8)
        for tile in (self.episodes_tile, self.proven_tile, self.muted_tile, self.best_tile, self.fresh_tile):
            kpi_row.addWidget(tile)
        kpi_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addLayout(kpi_row)
        self.detail_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.detail_splitter.addWidget(self.tabs)
        self.detail_splitter.addWidget(self.explanation_view)
        self.detail_splitter.setStretchFactor(0, 3)
        self.detail_splitter.setStretchFactor(1, 2)
        layout.addWidget(self.detail_splitter, 1)
        layout.addWidget(self.status_label)

    def _decisions_page(self) -> QFrame:
        """The tab body: a caption saying what these numbers ARE, then the strip."""
        page = QFrame()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        caption = QLabel(
            "What YOU did with the alerts, from the review-decision log - the "
            "mirror of the tabs beside this one, which are what the BOT "
            "measured. Take rate is P(take | shown) for that segment. Taken R "
            "and Passed R are the outcomes of the charts you took and the ones "
            "you did not; a blank is an absent measurement, never a zero. Gap "
            "is Taken minus Passed and is shown only when both sides have one. "
            "Read as DISCOVERY: nothing here changes a score, an alert or a list."
        )
        caption.setObjectName("MutedLabel")
        caption.setWordWrap(True)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.decisions_button, 0)
        row.addWidget(self.decisions_status, 1)
        layout.addWidget(caption)
        layout.addLayout(row)
        layout.addWidget(self.decisions_tabs, 1)
        return page

    def _make_decision_table(self) -> tuple[DataTable, TrackerTableModel]:
        numeric = {key for key, _label in DECISION_COLUMNS
                   if key not in {"segment", "probation"}}
        model = TrackerTableModel(
            DECISION_COLUMNS,
            percent_keys=DECISION_PERCENT_KEYS,
            signed_keys=DECISION_SIGNED_KEYS,
            numeric_keys=numeric,
        )
        proxy = TrackerSortProxyModel(self)
        proxy.setSourceModel(model)
        table = DataTable()
        table.setModel(proxy)
        table.setShowGrid(False)
        return table, model

    def start_decisions_refresh(self, *, rebuild: bool = True) -> None:
        """Read the scoreboard on a daemon thread. Single-flight.

        `rebuild=True` (the button) asks `review_learning` to rebuild the state
        first if it has gone stale - the same call `app.py` makes at startup,
        in the same shape. `rebuild=False` (construction) only READS, so
        opening the desk never triggers a rebuild it did not ask for.

        Every file touch is inside the worker. The Qt thread gets one signal
        with a finished dict.
        """
        if self._decisions_thread is not None and self._decisions_thread.is_alive():
            return
        self.decisions_button.setEnabled(False)
        if rebuild:
            self.decisions_status.setText("Reading your decisions...")
        self._decisions_thread = threading.Thread(
            target=self._decisions_worker,
            args=(bool(rebuild),),
            name="tracker-decisions-read",
            daemon=True,
        )
        self._decisions_thread.start()

    def _decisions_worker(self, rebuild: bool) -> None:
        payload: dict[str, Any] = {"state": None, "message": "", "probation": frozenset()}
        try:
            from review_learning import load_review_learning_state

            if rebuild:
                from review_learning import refresh_review_learning_if_stale

                refresh_review_learning_if_stale()
            payload["state"] = load_review_learning_state()
            payload["probation"] = _probation_types()
        except Exception as exc:  # noqa: BLE001 - a scoreboard is advisory
            payload["message"] = f"Your decisions could not be read: {exc}"
        try:
            self._decisionsLoaded.emit(payload)
        except RuntimeError as swallowed_exc:
            # The panel was deleted while this read was in flight. `shutdown`
            # joins the thread, but deletion can still win the race, and a
            # worker must never touch a widget that is gone - there is nothing
            # left to update, so the payload is simply dropped.
            note_swallowed("decisions read finished after the panel was deleted", swallowed_exc, quiet=True)

    def _on_decisions_loaded(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        self.decisions_button.setEnabled(True)
        state = data.get("state")
        probation = data.get("probation") or frozenset()
        for key, (table, model) in self._decision_tables.items():
            # R4 B4: through the SAME join the performance table uses, so the
            # trader's own decisions carry the day-trade headline rather than
            # mean R alone. These rows name no side, so they land on the pooled
            # cell.
            model.set_rows(
                apply_held_and_ran(
                    decision_rows(state, key, probation), self._held_run_summaries
                )
            )
            table.fit_columns()
        message = str(data.get("message") or "")
        if message:
            self.decisions_status.setText(message)
            return
        if not isinstance(state, dict):
            self.decisions_status.setText(
                "No review-preference scoreboard on disk yet. It is built from "
                "the decision log - press 'Refresh my decisions' once you have "
                "reviewed some charts. This is an absent measurement, not a "
                "session without decisions."
            )
            return
        self.decisions_status.setText(
            f"{state.get('shown', 0)} chart(s) shown, {state.get('takes', 0)} taken "
            f"over the last {state.get('window_sessions', '?')} sessions; scoreboard "
            f"generated {state.get('generated_at', 'unknown')}."
        )

    def _apply_default_sort(self, table) -> None:
        """Sort the performance table by the day-trade headline, descending.

        By COLUMN NAME rather than index: this table has gained a column in three
        packets, and an index would move under the next one.
        """
        keys = [key for key, _label in PERFORMANCE_COLUMNS]
        try:
            column = keys.index(DEFAULT_PERFORMANCE_SORT_KEY)
        except ValueError:  # pragma: no cover - the column was renamed
            return
        from PySide6.QtCore import Qt as _Qt

        table.sortByColumn(column, _Qt.SortOrder.DescendingOrder)

    def _make_table(self, columns) -> tuple[DataTable, TrackerTableModel]:
        text_keys = {"direction", "segment", "dimension", "recommendation", "status", "example_symbols", "grade_1r", "grade_2r", "exit_by"}
        numeric = {key for key, _label in columns if key not in text_keys}
        model = TrackerTableModel(
            columns,
            percent_keys=PERCENT_KEYS,
            signed_keys=SIGNED_KEYS,
            numeric_keys=numeric,
            tooltip_keys={"example_symbols"},
        )
        proxy = TrackerSortProxyModel(self)
        proxy.setSourceModel(model)
        table = DataTable()
        table.setModel(proxy)
        table.setShowGrid(False)
        return table, model

    def _show_row_explanation(self, index, kind: str, *, dimension: str = "") -> None:
        row = index.data(ROW_ROLE)
        if not isinstance(row, dict):
            return
        payload = dict(row)
        if dimension and not payload.get("dimension"):
            payload["dimension"] = dimension
        self.explanation_view.show_row(
            kind, payload, identity=_explanation_identity(kind, payload)
        )

    def _on_context_tab_changed(self, _index: int) -> None:
        """Any tab move retires the explanation (packet G4.2)."""
        self.explanation_view.clear()

    def _reshow_or_clear_explanation(self) -> None:
        """After a data revision, re-read the open row or take the pane down.

        The identity is looked up in the model that now holds the tab's rows and
        the pane is redrawn from the NEW row dict - never the cached one, or the
        pane would keep showing a number the table has already revised. A
        segment the revision dropped takes its explanation with it. One dict
        lookup over the current dimension's rows; nothing else lands on the Qt
        thread here.
        """
        view = self.explanation_view
        identity = getattr(view, "shown_identity", None)
        if view.isHidden() or not identity:
            return
        kind, dimension = identity[0], identity[1]
        if kind == "daytrade_learning":
            model = self.learning_model
        else:
            entry = self._dimension_tables.get(dimension)
            model = entry[1] if entry else None
        if model is None:
            view.clear()
            return
        for row in model.rows():
            if not isinstance(row, dict):
                continue
            payload = dict(row)
            if dimension and not payload.get("dimension"):
                payload["dimension"] = dimension
            if _explanation_identity(kind, payload) == identity:
                view.show_row(kind, payload, identity=identity)
                return
        view.clear()

    # ------------------------------------------------------------------
    def reload_from_disk(self) -> None:
        # The held/ran numbers come from `held_run_score` and its read is
        # expensive, so the table renders with whatever this panel already has
        # and the worker below fills the two columns in when it lands. Blank
        # first is honest: it is what an unmeasured cell looks like.
        self._performance_rows = _load_performance_rows()
        self._start_held_run_read()
        # R4 B4: the champion tier, from the ONE learning-state read this method
        # already does. Cached on the panel so the held/ran worker's callback can
        # re-join it without opening the file again - a signal handler runs on
        # the Qt thread, and nothing expensive belongs there.
        state = load_bounce_learning_state() or {}
        self._learning_state = state
        perf_rows = apply_regime_grades(
            apply_exit_windows(
                apply_setup_grades(
                    apply_champion_tier(
                        apply_held_and_ran(self._performance_rows, self._held_run_summaries),
                        state,
                    ),
                    self._setup_grades,
                ),
                self._exit_windows,
            ),
            self._regime_grades,
        )
        by_dimension: dict[str, list[dict]] = {}
        for row in perf_rows:
            by_dimension.setdefault(str(row.get("dimension") or ""), []).append(row)
        for key, (_table, model) in self._dimension_tables.items():
            model.set_rows(_by_headline(by_dimension.get(key, [])))

        learning_rows = []
        muted_count = 0
        proven_count = 0
        for dimension, segments in (state.get("segments") or {}).items():
            for seg_key, entry in segments.items():
                direction, _, segment = seg_key.partition("|")
                muted = bool(entry.get("muted"))
                proven = bool(entry.get("proven"))
                muted_count += int(muted)
                proven_count += int(proven)
                learning_rows.append(
                    {
                        "dimension": dimension,
                        "direction": direction,
                        "segment": segment,
                        "sample_count": entry.get("sample_count"),
                        "avg_close_r": entry.get("avg_close_r"),
                        "score_delta": entry.get("score_delta"),
                        "stop_rate": entry.get("stop_rate"),
                        "target_1r_rate": entry.get("target_1r_rate"),
                        "status": recording_off_text(dimension, segment)
                        or ("MUTED" if muted else ("PROVEN" if proven else "active")),
                    }
                )
        status_order = {"PROVEN": 0, "MUTED": 1, "active": 2}
        learning_rows.sort(
            key=lambda r: (status_order.get(r["status"], 3), -_float(r.get("avg_close_r"), -999.0))
        )
        self.learning_model.set_rows(learning_rows)

        for table, _model in self._dimension_tables.values():
            table.fit_columns()
        self.learning_table.fit_columns()

        self.episodes_tile.set_value(str(len(learning_rows)))
        self.proven_tile.set_value(str(proven_count))
        self.muted_tile.set_value(str(muted_count))
        best = max(learning_rows, key=lambda r: _float(r.get("avg_close_r"), -999.0), default=None)
        if best:
            self.best_tile.set_value(f"{best['direction']} {best['segment']} {_float(best.get('avg_close_r'), 0.0):+.2f}R")
        self.fresh_tile.set_value(_mtime_text(INTRADAY_BOUNCE_OUTCOMES_FILE))
        generated = str(state.get("generated_at") or "never")
        self.status_label.setText(
            f"Learning state generated {generated} ({BOUNCE_LEARNING_STATE_FILE.name}); "
            f"outcome file updated {_mtime_text(INTRADAY_BOUNCE_OUTCOMES_FILE)}. "
            f"{getattr(self, '_held_run_window_text', '')} "
            f"{getattr(self, '_outcome_coverage_text', '')}".rstrip()
        )

    def start_refresh(self) -> None:
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return
        self.refresh_button.setEnabled(False)
        self.status_label.setText("Re-aggregating bounce outcomes (full history)...")
        self._refresh_thread = threading.Thread(target=self._refresh_worker, daemon=True)
        self._refresh_thread.start()

    def _refresh_worker(self) -> None:
        try:
            from bounce_bot_lib.learning import refresh_bounce_learning_state

            state = refresh_bounce_learning_state()
            segments = sum(len(v) for v in state.get("segments", {}).values())
            message = f"Bounce learning refreshed: {segments} segments with enough evidence."
        except Exception as exc:
            message = f"Bounce learning refresh failed: {exc}"
        self._refreshFinished.emit(message)

    def _on_refresh_finished(self, message: str) -> None:
        self.refresh_button.setEnabled(True)
        self.reload_from_disk()
        # The models under the open explanation just changed (G4.2).
        self._reshow_or_clear_explanation()
        self.status_label.setText(message)
        self.statusChanged.emit(message)

    # ------------------------------------------------------------------
    def _start_held_run_read(self) -> None:
        """One read at a time, on a worker, never blocking the panel."""
        if self._held_run_thread is not None and self._held_run_thread.is_alive():
            return
        self._held_run_thread = threading.Thread(
            target=self._held_run_worker,
            name="tracker-held-run-read",
            daemon=True,
        )
        self._held_run_thread.start()

    def _held_run_worker(self) -> None:
        summaries = load_held_run_report()
        try:
            self._heldRunLoaded.emit(summaries)
        except RuntimeError as exc:
            # The panel went away mid-read. Nothing left to update; drop it.
            note_swallowed("held-run read finished after the panel was deleted", exc, quiet=True)

    def _on_held_run_loaded(self, summaries) -> None:
        window_text = ""
        coverage_text = ""
        if isinstance(summaries, dict) and "summaries" in summaries:
            grades = summaries.get("setup_grades")
            self._setup_grades = grades if isinstance(grades, dict) else {}
            exits = summaries.get("exit_windows")
            self._exit_windows = exits if isinstance(exits, dict) else {}
            by_regime = summaries.get("regime_grades")
            self._regime_grades = by_regime if isinstance(by_regime, dict) else {}
            shadow_rows = summaries.get("shadow_s9")
            self.shadow_s9_model.set_rows(shadow_rows if isinstance(shadow_rows, list) else [])
            self.shadow_s9_table.fit_columns()
            window_text = held_run_window_text(summaries.get("window"))
            coverage_text = outcome_coverage_text(summaries.get("outcome_coverage"))
            summaries = summaries.get("summaries")
        self._held_run_summaries = summaries if isinstance(summaries, dict) else {}
        # Q1's window sentence, then M2's coverage sentence. Each is appended
        # only once: `reload_from_disk` rebuilds the label from
        # `_held_run_window_text`, and this guard keeps a second worker result
        # from doubling either clause.
        for text, attribute in (
            (window_text, "_held_run_window_text"),
            (coverage_text, "_outcome_coverage_text"),
        ):
            if not text:
                continue
            setattr(self, attribute, text)
            current = self.status_label.text()
            if text not in current:
                self.status_label.setText(f"{current} {text}".strip())
        rows = apply_regime_grades(
            apply_exit_windows(
                apply_setup_grades(
                    apply_champion_tier(
                        apply_held_and_ran(self._performance_rows, self._held_run_summaries),
                        getattr(self, "_learning_state", {}) or {},
                    ),
                    self._setup_grades,
                ),
                getattr(self, "_exit_windows", {}) or {},
            ),
            getattr(self, "_regime_grades", {}) or {},
        )
        by_dimension: dict[str, list[dict]] = {}
        for row in rows:
            by_dimension.setdefault(str(row.get("dimension") or ""), []).append(row)
        for key, (table, model) in self._dimension_tables.items():
            model.set_rows(_by_headline(by_dimension.get(key, [])))
            table.fit_columns()
        # The held/ran columns are a data revision too (G4.2).
        self._reshow_or_clear_explanation()

    def shutdown(self) -> None:
        """Let no read outlive the panel it was going to update."""
        for thread in (self._decisions_thread, self._held_run_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)


def _load_performance_rows() -> list[dict[str, Any]]:
    try:
        from bounce_bot_lib.legacy import INTRADAY_BOUNCE_PERFORMANCE_CSV as perf_path
    except Exception:
        return []
    if not Path(perf_path).exists():
        return []
    try:
        with open(perf_path, newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def _by_headline(rows) -> list[dict]:
    """V3 item 2's ordering, in one place because two callers now need it.

    Ordered by the HEADLINE - did the level hold, then how far did it run -
    rather than by average R. Decision 0016 answer 4: *"rank by maximum
    favourable excursion, not by any exit; exiting well is the trader's job."*
    A row that cannot be measured sorts LAST rather than at the bottom of the
    scale, which is a different claim.
    """
    return sorted(
        rows,
        key=lambda r: (
            r.get("held_run_score") is None,
            -_float(r.get("held_run_score"), -999.0),
        ),
    )


def _float(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mtime_text(path: Path) -> str:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "never"


def apply_held_and_ran(rows, summaries) -> list[dict]:
    """Join `held_run_score`'s own numbers onto the aggregator's rows - R4 A10.

    V3 shipped a SECOND FORMULA here under the same column key: `1 - stop_rate`
    times `avg_mfe_r`, both taken from the aggregator over ITS window and over
    ALL rows rather than the ones that held, with no thirty-minute question
    anywhere in it. Two different numbers under one heading is worse than one
    blank, because the trader reads the column as an ordering.

    So the arithmetic now comes from the module that owns it and nothing is
    computed in this file. `summaries` is
    `held_run_score.dimension_summaries(...)`, keyed `(dimension, direction,
    segment)`, and a row it cannot answer gets None - which the default sort
    already puts last, and which the six unmeasurable tabs will show, because
    `intraday_bounce_outcomes.csv` does not record the alert context those
    dimensions are cut on.
    """
    lookup = summaries or {}
    out: list[dict] = []
    for raw in rows or ():
        row = dict(raw)
        dimension = str(row.get("dimension") or "").strip()
        segment = str(row.get("segment") or "").strip()
        # R4 B4: a row that names no side joins the POOLED cell, which
        # `held_run_score` accumulates from the episodes exactly as it does the
        # sided ones. The "My Decisions" tabs are such rows -
        # `review_preference_state.json` records take and pass per segment and
        # carries no side within a dimension - and averaging the long cell with
        # the short cell here would be a mean of trimmed means, which is not a
        # trimmed mean and would be a second formula in this file again.
        direction = str(row.get("direction") or "").strip().lower()
        if not direction:
            import held_run_score

            direction = held_run_score.ALL_DIRECTIONS
        cell = lookup.get((dimension, direction, segment))
        row["held_rate"] = (cell or {}).get("hold_rate")
        row["held_run_score"] = (cell or {}).get("held_run_score")
        row["measured"] = measured_text(cell)
        out.append(row)
    return out


def apply_setup_grades(rows, payload) -> list[dict]:
    """Join `setup_grades`' day-trade cells onto the Bounce Types rows (S1 + S10c).

    Keyed ``(bounce type, side)`` as `setup_grades.daytrade_key`. Other
    dimensions and a type with no cell get blanks, never a substitute number.
    """
    import setup_grades

    lookup = setup_grades.daytrade_lookup(payload)
    out: list[dict] = []
    for raw in rows or ():
        row = dict(raw)
        cell = None
        if str(row.get("dimension") or "").strip() == "bounce_type":
            cell = lookup.get(setup_grades.daytrade_key(row.get("segment"), row.get("direction")))
        cell = cell or {}
        row["grade_1r"] = setup_grades.badge(cell.get("grade")) if cell else ""
        row["grade_2r"] = setup_grades.badge(cell.get("grade_2r")) if "grade_2r" in cell else ""
        row["eod_r_mean"] = cell.get("eod_r_mean")
        row["eod_r_median"] = cell.get("eod_r_median")
        row["reach_2r_rate"] = cell.get("reach_2r_rate")
        row["grade_n"] = cell.get("n") if cell else None
        out.append(row)
    return out


def apply_regime_grades(rows, payload) -> list[dict]:
    """S16: this regime's grade and every regime's, on Bounce Types rows. Formats only.

    Keyed ``(bounce type, side)`` as `setup_grades.daytrade_key`; other
    dimensions get blanks. The pooled grade is labelled "all regimes".
    """
    import regime_grades
    import setup_grades

    payload = payload or {}
    cells = payload.get("daytrade") or {}
    out: list[dict] = []
    for raw in rows or ():
        row = dict(raw)
        row["regime_now"] = ""
        row["regime_all"] = ""
        if payload and str(row.get("dimension") or "").strip() == "bounce_type":
            entry = cells.get(setup_grades.daytrade_key(row.get("segment"), row.get("direction"))) or {}
            by_regime = entry.get("by_regime") or {}
            row["regime_now"] = regime_grades.this_regime_text(by_regime, payload.get("current"))
            row["regime_all"] = regime_grades.by_regime_text(
                by_regime, payload, pooled=entry.get("all") or {}
            )
        out.append(row)
    return out


def apply_exit_windows(rows, payload) -> list[dict]:
    """S11: the "Exit by" text on Bounce Types rows, keyed (bounce type, side). Formats only."""
    import exit_windows

    cells = exit_windows.lookup(payload)
    out: list[dict] = []
    for raw in rows or ():
        row = dict(raw)
        cell = None
        if str(row.get("dimension") or "").strip() == "bounce_type":
            cell = cells.get(exit_windows.key_for(row.get("segment"), row.get("direction")))
        row["exit_by"] = exit_windows.tracker_text(cell)
        out.append(row)
    return out


def measured_text(cell) -> str:
    """`"35 / 41"` - measured episodes over all episodes - or blank (packet Q1)."""
    if not isinstance(cell, dict):
        return ""
    total = cell.get("n")
    measured = cell.get("n_measured")
    if total is None or measured is None:
        return ""
    return f"{int(measured)} / {int(total)}"


def outcome_coverage_text(counts) -> str:
    """The window's outcome coverage, in one sentence (packet M2.3).

    `Outcomes: measured N (eod A / swept B), unmeasured C, open D over the
    window.` Rendered by `outcome_semantics` so this line, the AWAY digest's
    and the sweep's log all say the same thing. Blank when there is nothing to
    say, so the status line never grows an empty clause.
    """
    import outcome_semantics

    return outcome_semantics.format_terminal_coverage(counts)


def held_run_window_text(report) -> str:
    """One status-line sentence for `held_run_score.window_report` (packet Q1)."""
    if not isinstance(report, dict) or not report.get("start"):
        return ""
    missing = len(report.get("missing_sessions") or ())
    return (
        f"Held/ran window: {report.get('sessions', 0)} sessions "
        f"({report.get('start')} to {report.get('end')}), "
        f"{report.get('sessions_with_data', 0)} with data, {missing} missing."
    )


def apply_champion_tier(rows, state) -> list[dict]:
    """Join the champion's own tier onto the aggregator's rows - R4 B4.

    The panel's header comment has said "the champion tier stays as a column"
    since V3 and there was no such column. The tier answers a different question
    from the headline: it says whether the desk should ALERT on the segment at
    all, while Held x Ran says what the alert offered once the level held. Beside
    each other they are two facts; without the tier the reader has one number
    carrying both meanings.

    `state` is `load_bounce_learning_state()`, whose segments are keyed
    `dimension -> "<direction>|<segment>"` - the same identity the headline joins
    on, spelled the state's way.

    A segment the state has never seen gets a BLANK, never "active": "not
    tracked" and "tracked and unremarkable" are different facts, and this file
    has already paid once for a column that filled an absent measurement in.

    Nothing here changes a tier. It is read-only over a file the champion writes.
    """
    segments = (state or {}).get("segments") or {}
    out: list[dict] = []
    for raw in rows or ():
        row = dict(raw)
        dimension = str(row.get("dimension") or "").strip()
        direction = str(row.get("direction") or "").strip().lower()
        segment = str(row.get("segment") or "").strip()
        entry = (segments.get(dimension) or {}).get(f"{direction}|{segment}")
        if not isinstance(entry, dict):
            row["champion_tier"] = ""
        elif entry.get("muted"):
            row["champion_tier"] = "MUTED"
        elif entry.get("proven"):
            row["champion_tier"] = "PROVEN"
        else:
            row["champion_tier"] = "active"
        row["champion_tier"] = recording_off_text(dimension, segment) or row["champion_tier"]
        out.append(row)
    return out


def load_held_run_summaries() -> dict:
    """`held_run_score`'s marginals, read off disk. NEVER on the Qt thread.

    The outcome log is ~300,000 rows and ~90 MB and the setups snapshot is
    ~19 MB, so this is the panel's expensive read and it runs on a worker.
    A failure yields an empty mapping, which shows blanks - the panel still
    opens, and a blank is what an unmeasured cell should look like anyway.
    """
    return load_held_run_report().get("summaries") or {}


def load_held_run_report() -> dict:
    """`{"summaries": ..., "window": ...}` from ONE read of the outcome log.

    The window report and the outcome coverage ride along so the status line
    can say which sessions the headline is measured over, which are missing
    (packet Q1) and how many of the window's outcomes were measured at all
    (packet M2), without a second 300 MB pass. NEVER on the Qt thread.
    """
    try:
        import held_run_score

        episodes = held_run_score.load_episodes()
        return {
            "summaries": held_run_score.dimension_summaries(episodes),
            "window": held_run_score.window_report(episodes),
            # Packet M2.3: coverage is shown, never hidden - and it rides THIS
            # read. A second pass over the 308 MB file to count four kinds
            # would be the panel's expensive read done twice.
            "outcome_coverage": held_run_score.terminal_coverage(episodes),
            "setup_grades": _read_setup_grades(),
            "exit_windows": _read_exit_windows(),
            "regime_grades": _read_regime_grades(),
            "shadow_s9": read_shadow_s9_rows(),
        }
    except Exception:
        return {
            "summaries": {}, "window": {}, "outcome_coverage": {},
            "setup_grades": _read_setup_grades(), "exit_windows": _read_exit_windows(),
            "regime_grades": _read_regime_grades(), "shadow_s9": read_shadow_s9_rows(),
        }


def read_shadow_s9_rows(path: Path | None = None, limit: int = SHADOW_S9_MAX_ROWS) -> list[dict]:
    """S9: recent alerts that carry a shadow tier, newest first. NEVER on the Qt thread."""
    try:
        from project_paths import INTRADAY_BOUNCES_FILE

        source = Path(path) if path else Path(INTRADAY_BOUNCES_FILE)
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.DictReader(handle) if str(row.get("shadow_s9_tier") or "").strip()]
    except Exception:  # noqa: BLE001 - the tab is then empty
        return []
    out = []
    for row in reversed(rows[-int(limit):]):
        item = {key: str(row.get(key) or "") for key, _label in SHADOW_S9_COLUMNS}
        for key in ("composite_r", "shadow_s9_composite_r"):
            item[key] = _float(item[key], None)
        out.append(item)
    return out


def _read_exit_windows() -> dict:
    """S11: the night's exit-window payload, `{}` when absent. NEVER on the Qt thread."""
    try:
        import exit_windows

        return exit_windows.read_payload()
    except Exception:  # noqa: BLE001 - the Exit by column is then blank
        return {}


def _read_regime_grades() -> dict:
    """S16: the grades by regime the working-lately build last published. NEVER on the Qt thread."""
    try:
        from ui.services.working_lately_service import read_persisted_regime_grades

        return read_persisted_regime_grades()
    except Exception:  # noqa: BLE001 - the regime columns are then blank
        return {}


def _read_setup_grades() -> dict:
    """The grades the working-lately build last published, `{}` when absent. NEVER on the Qt thread."""
    try:
        from ui.services.working_lately_service import read_persisted_grades

        return read_persisted_grades()
    except Exception:  # noqa: BLE001 - the grade columns are then blank
        return {}
