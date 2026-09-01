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

PERFORMANCE_COLUMNS = (
    ("direction", "Side"),
    ("segment", "Segment"),
    ("sample_count", "N"),
    ("avg_close_r", "Avg R"),
    ("median_close_r", "Med R"),
    ("avg_mfe_r", "MFE R"),
    ("avg_mae_r", "MAE R"),
    ("positive_eod_rate", "Win"),
    ("target_1r_rate", "1R Hit"),
    ("target_2r_rate", "2R Hit"),
    ("stop_rate", "Stop"),
    ("recommendation", "Verdict"),
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

PERCENT_KEYS = {"positive_eod_rate", "target_1r_rate", "target_2r_rate", "stop_rate"}
SIGNED_KEYS = {"avg_close_r", "median_close_r", "avg_mfe_r", "avg_mae_r", "score_delta"}

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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._refresh_thread: threading.Thread | None = None

        self._decisions_thread: threading.Thread | None = None

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
        self._decisionsLoaded.emit(payload)

    def _on_decisions_loaded(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        self.decisions_button.setEnabled(True)
        state = data.get("state")
        probation = data.get("probation") or frozenset()
        for key, (table, model) in self._decision_tables.items():
            model.set_rows(decision_rows(state, key, probation))
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
            f"over the last {state.get('window_days', '?')} days; scoreboard "
            f"generated {state.get('generated_at', 'unknown')}."
        )

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
        perf_rows = _load_performance_rows()
        by_dimension: dict[str, list[dict]] = {}
        for row in perf_rows:
            by_dimension.setdefault(str(row.get("dimension") or ""), []).append(row)
        for key, (_table, model) in self._dimension_tables.items():
            rows = sorted(
                by_dimension.get(key, []),
                key=lambda r: -_float(r.get("avg_close_r"), -999.0),
            )
            model.set_rows(rows)

        state = load_bounce_learning_state() or {}
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
            f"outcome file updated {_mtime_text(INTRADAY_BOUNCE_OUTCOMES_FILE)}."
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

    def shutdown(self) -> None:
        """Let no read outlive the panel it was going to update."""
        thread = self._decisions_thread
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
