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

PERCENT_KEYS = {
    "positive_eod_rate",
    "target_1r_rate",
    "target_2r_rate",
    "stop_rate",
    "held_rate",
}
SIGNED_KEYS = {
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

        self._refreshFinished.connect(self._on_refresh_finished)
        self._decisionsLoaded.connect(self._on_decisions_loaded)
        self._heldRunLoaded.connect(self._on_held_run_loaded)
        self._build_layout()
        self.reload_from_disk()
        # Off the Qt thread from the first paint: the scoreboard is a 34 KB
        # JSON on the home folder, and reading it in the constructor is the
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
        except RuntimeError:
            # The panel was deleted while this read was in flight. `shutdown`
            # joins the thread, but deletion can still win the race, and a
            # worker must never touch a widget that is gone - there is nothing
            # left to update, so the payload is simply dropped.
            pass

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
        numeric = {key for key, _label in columns if key not in {"direction", "segment", "dimension", "recommendation", "status", "example_symbols"}}
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
        self.explanation_view.show_row(kind, payload)

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
        perf_rows = apply_champion_tier(
            apply_held_and_ran(self._performance_rows, self._held_run_summaries),
            state,
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
                        "status": "MUTED" if muted else ("PROVEN" if proven else "active"),
                    }
                )
        status_order = {"PROVEN": 0, "MUTED": 1, "active": 2}
        learning_rows.sort(
            key=lambda r: (status_order.get(r["status"], 3), -_float(r.get("avg_close_r"), -999.0))
        )
        self.learning_model.set_rows(learning_rows)

        for key, (table, _model) in self._dimension_tables.items():
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
            f"{getattr(self, '_held_run_window_text', '')}".rstrip()
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
        except RuntimeError:
            # The panel went away mid-read. Nothing left to update; drop it.
            pass

    def _on_held_run_loaded(self, summaries) -> None:
        window_text = ""
        if isinstance(summaries, dict) and "summaries" in summaries:
            window_text = held_run_window_text(summaries.get("window"))
            summaries = summaries.get("summaries")
        self._held_run_summaries = summaries if isinstance(summaries, dict) else {}
        if window_text:
            self._held_run_window_text = window_text
            current = self.status_label.text()
            if window_text not in current:
                self.status_label.setText(f"{current} {window_text}".strip())
        rows = apply_champion_tier(
            apply_held_and_ran(self._performance_rows, self._held_run_summaries),
            getattr(self, "_learning_state", {}) or {},
        )
        by_dimension: dict[str, list[dict]] = {}
        for row in rows:
            by_dimension.setdefault(str(row.get("dimension") or ""), []).append(row)
        for key, (table, model) in self._dimension_tables.items():
            model.set_rows(_by_headline(by_dimension.get(key, [])))
            table.fit_columns()

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


def measured_text(cell) -> str:
    """`"35 / 41"` - measured episodes over all episodes - or blank (packet Q1)."""
    if not isinstance(cell, dict):
        return ""
    total = cell.get("n")
    measured = cell.get("n_measured")
    if total is None or measured is None:
        return ""
    return f"{int(measured)} / {int(total)}"


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

    The window report rides along so the status line can say which sessions
    the headline is measured over and which are missing (packet Q1), without a
    second 300 MB pass. NEVER on the Qt thread.
    """
    try:
        import held_run_score

        episodes = held_run_score.load_episodes()
        return {
            "summaries": held_run_score.dimension_summaries(episodes),
            "window": held_run_score.window_report(episodes),
        }
    except Exception:
        return {"summaries": {}, "window": {}}
