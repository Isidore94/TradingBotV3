from __future__ import annotations

import csv
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
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from project_paths import (
    HUMAN_FOCUS_OUTCOMES_FILE,
    HUMAN_FOCUS_PERFORMANCE_FILE,
    MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE,
    MASTER_AVWAP_SETUP_STATS_FILE,
    MASTER_AVWAP_SETUP_TRACKER_FILE,
    MASTER_AVWAP_TIER_CATCH_RATE_FILE,
    MASTER_AVWAP_TIER_LIST_FILE,
    MASTER_AVWAP_TIER_PERFORMANCE_FILE,
)
from ui import theme
from ui.models.tracker_table_model import TrackerSortProxyModel, TrackerTableModel
from ui.services.human_focus_tracker_feed import (
    build_human_focus_comparison_rows,
    load_human_focus_performance_rows,
)
from ui.widgets.data_table import DataTable
from ui.widgets.kpi_tile import KpiTile
from ui.widgets.section_header import SectionHeader


SETUP_TYPE_STATS_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name("master_avwap_setup_type_stats.csv")
RECENT_SETUP_TYPE_STATS_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name("master_avwap_setup_type_recent_stats.csv")
SETUP_PLAYBOOKS_FILE = MASTER_AVWAP_SETUP_STATS_FILE.with_name("master_avwap_setup_playbooks.csv")


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
    "target_hit_rate",
    "stop_rate",
    "positive_scan_factor_match_rate",
    "caught_winner_rate",
    "caught_opportunity_rate",
    "bot_sa_win_rate",
}
SIGNED_KEYS = {
    "avg_closed_r",
    "avg_closed_r_edge",
    "representative_closed_r",
    "representative_total_r",
    "robust_closed_r",
    "robust_closed_r_edge",
    "avg_total_r",
    "avg_total_r_edge",
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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.current_pick_rows: list[dict[str, Any]] = []
        self.setup_type_rows: list[dict[str, Any]] = []
        self.recent_type_rows: list[dict[str, Any]] = []
        self.playbook_rows: list[dict[str, Any]] = []
        self.scan_factor_rows: list[dict[str, Any]] = []
        self.tier_performance_rows: list[dict[str, Any]] = []
        self.catch_rate_rows: list[dict[str, Any]] = []
        self.human_pick_rows: list[dict[str, Any]] = []

        self.min_closed_input = QSpinBox()
        self.min_closed_input.setRange(1, 100)
        self.min_closed_input.setValue(5)
        self.min_closed_input.valueChanged.connect(self.refresh)

        self.refresh_button = QPushButton("Refresh Tracker")
        self.refresh_button.setObjectName("PrimaryButton")
        self.refresh_button.clicked.connect(self.refresh)

        self.tracked_tile = KpiTile("Tracked Setups", "0")
        self.current_tile = KpiTile("Current S/A Picks", "0", tone="favorite")
        self.best_type_tile = KpiTile("Best Type Edge", "-")
        self.best_factor_tile = KpiTile("Best Scan Factor", "-")

        self.summary_view = QTextBrowser()
        self.summary_view.setOpenExternalLinks(False)

        self.status_label = QLabel("Tracker exports have not been loaded yet.")
        self.status_label.setObjectName("MutedLabel")

        self.tabs = QTabWidget()
        self.current_table, self.current_model = self._make_table(CURRENT_PICK_COLUMNS)
        self.setup_type_table, self.setup_type_model = self._make_table(SETUP_TYPE_COLUMNS)
        self.recent_type_table, self.recent_type_model = self._make_table(RECENT_TYPE_COLUMNS)
        self.playbook_table, self.playbook_model = self._make_table(PLAYBOOK_COLUMNS)
        self.scan_factor_table, self.scan_factor_model = self._make_table(SCAN_FACTOR_COLUMNS)
        self.tier_performance_table, self.tier_performance_model = self._make_table(TIER_PERFORMANCE_COLUMNS)
        self.catch_rate_table, self.catch_rate_model = self._make_table(CATCH_RATE_COLUMNS)
        self.human_pick_table, self.human_pick_model = self._make_table(HUMAN_PICK_COLUMNS)

        self.tabs.addTab(self.current_table, "Current Picks")
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
                "What's worked in the last 30 days: per-family closed count, realized R, target/stop rates "
                "across live setups and measured-only study families (incl. 2nd-dev breakouts).",
                self.recent_type_table,
            ),
            "Last 30 Days",
        )
        self.tabs.addTab(self.playbook_table, "Playbooks")
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

        self._build_layout()
        self.refresh()

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
        for tile in (self.tracked_tile, self.current_tile, self.best_type_tile, self.best_factor_tile):
            kpi_row.addWidget(tile)
        kpi_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addLayout(kpi_row)
        layout.addWidget(self.summary_view, 1)
        layout.addWidget(self.tabs, 2)
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

    def _make_explained_tab(self, description: str, table: DataTable) -> QWidget:
        tab = QWidget()
        label = QLabel(description)
        label.setObjectName("MutedLabel")
        label.setWordWrap(True)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(label)
        layout.addWidget(table, 1)
        return tab

    def refresh(self) -> None:
        min_closed = int(self.min_closed_input.value())
        all_setup_type_rows = _load_csv_rows(SETUP_TYPE_STATS_FILE)
        all_playbook_rows = _load_csv_rows(SETUP_PLAYBOOKS_FILE)
        tier_performance_export_rows = _load_csv_rows(MASTER_AVWAP_TIER_PERFORMANCE_FILE)
        self.current_pick_rows = _rank_current_picks(_load_csv_rows(MASTER_AVWAP_TIER_LIST_FILE))
        self.setup_type_rows = _rank_setup_types(all_setup_type_rows, min_closed=min_closed)
        self.recent_type_rows = _rank_recent_types(_load_csv_rows(RECENT_SETUP_TYPE_STATS_FILE))
        self.playbook_rows = _rank_playbooks(all_playbook_rows, min_closed=min_closed)
        self.scan_factor_rows = _rank_scan_factors(_load_csv_rows(MASTER_AVWAP_SCAN_FACTOR_LEADERBOARD_FILE))
        self.tier_performance_rows = _rank_tier_performance(tier_performance_export_rows)
        self.catch_rate_rows = _rank_catch_rates(_load_csv_rows(MASTER_AVWAP_TIER_CATCH_RATE_FILE))
        self.human_pick_rows = build_human_focus_comparison_rows(
            load_human_focus_performance_rows(),
            tier_performance_export_rows,
        )

        self.current_model.set_rows(self.current_pick_rows[:300])
        self.human_pick_model.set_rows(self.human_pick_rows)
        self.setup_type_model.set_rows(self.setup_type_rows[:300])
        self.recent_type_model.set_rows(self.recent_type_rows[:300])
        self.playbook_model.set_rows(self.playbook_rows[:300])
        self.scan_factor_model.set_rows(self.scan_factor_rows[:300])
        self.tier_performance_model.set_rows(self.tier_performance_rows)
        self.catch_rate_model.set_rows(self.catch_rate_rows)
        for table in (
            self.current_table,
            self.human_pick_table,
            self.setup_type_table,
            self.recent_type_table,
            self.playbook_table,
            self.scan_factor_table,
            self.tier_performance_table,
            self.catch_rate_table,
        ):
            table.fit_columns()

        tracked_setups = sum(_int(row.get("tracked_setups")) for row in all_setup_type_rows)
        current_sa = sum(1 for row in self.current_pick_rows if str(row.get("tier") or "").upper() in {"S", "A"})
        self.tracked_tile.set_value(str(tracked_setups))
        self.current_tile.set_value(str(current_sa))
        self.best_type_tile.set_value(_best_type_label(self.setup_type_rows))
        self.best_factor_tile.set_value(_best_factor_label(self.scan_factor_rows))
        self.summary_view.setHtml(_summary_html(self))

        status = f"Setup tracker refreshed from exports. Last export: {_latest_mtime_text(_export_files())}"
        self.status_label.setText(status)
        self.statusChanged.emit(status)


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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
    # Surface families with the most recent closed evidence first, best realized R
    # next; open-only families (no closed yet) fall to the bottom but stay visible.
    return sorted(
        rows,
        key=lambda row: (
            -_int(row.get("closed_setups")),
            -_float(row.get("avg_closed_r"), -1e9),
            -_int(row.get("tracked_setups")),
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
    parts.append("<table width='100%' cellspacing='0' cellpadding='4'><tr>")
    parts.append("<td valign='top' width='35%'>")
    parts.append(f"<h3 style='margin:0; color:{favorite_c}'>Current S/A picks</h3>")
    if panel.current_pick_rows:
        for row in [row for row in panel.current_pick_rows if str(row.get("tier") or "").upper() in {"S", "A"}][:10]:
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
        parts.append(f"<div style='color:{muted}'>No current tier export found yet.</div>")
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


def _best_type_label(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "-"
    row = rows[0]
    delta = _float(row.get("score_delta"), 0.0)
    return f"{_esc(row.get('side'))} {delta:+.0f}"


def _best_factor_label(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "-"
    row = rows[0]
    return f"{_esc(row.get('horizon_sessions'))}d {_esc(row.get('side'))}"


def _export_files() -> list[Path]:
    return [
        SETUP_TYPE_STATS_FILE,
        RECENT_SETUP_TYPE_STATS_FILE,
        SETUP_PLAYBOOKS_FILE,
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
