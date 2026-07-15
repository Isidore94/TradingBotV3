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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._refresh_thread: threading.Thread | None = None

        self.refresh_button = QPushButton("Re-aggregate Outcomes")
        self.refresh_button.setObjectName("PrimaryButton")
        self.refresh_button.clicked.connect(self.start_refresh)
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
        self.explanation_view = ResearchExplanationView(self)

        self._refreshFinished.connect(self._on_refresh_finished)
        self._build_layout()
        self.reload_from_disk()

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
        pass


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
