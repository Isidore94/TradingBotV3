from __future__ import annotations

"""RS Window tab: pick a stretch of the SPY M5 tape, see who led and lagged.

Chart on top (bot's cached SPY 5m bars, gold draggable region = the
measurement window); table below ranks the bot's long/short candidate pools
by excess move vs SPY over that window, joined with current-pick context
(tier / setup family / favorite), industry & sector board strength, and
D1/weekly strength so the list can be filtered to favorite setups and
re-sorted by any strength axis.
"""

from typing import Any

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
)
from PySide6.QtCore import Qt

from ui.models.tracker_table_model import TrackerSortProxyModel, TrackerTableModel
from ui.services import rs_window_feed
from ui.widgets.data_table import DataTable
from ui.widgets.section_header import SectionHeader
from ui.widgets.spy_m5_chart import SpyM5Chart

COLUMNS = (
    ("symbol", "Symbol"),
    ("side", "Side"),
    ("window_pct", "Window %"),
    ("excess", "Excess vs SPY"),
    ("tier", "Tier"),
    ("setup_family", "Setup Family"),
    ("favorite_zone", "Fav Zone"),
    ("sector", "Sector"),
    ("sector_rs", "Sector RS"),
    ("industry", "Industry"),
    ("industry_rs", "Ind RS"),
    ("industry_rank", "Ind Rank"),
    ("d1_rs_5d", "D1 RS 5d"),
    ("d1_rs_20d", "D1 RS 20d"),
    ("weekly_streak", "Wkly 8EMA"),
)
SIGNED_KEYS = {
    "window_pct",
    "excess",
    "sector_rs",
    "industry_rs",
    "d1_rs_5d",
    "d1_rs_20d",
    "weekly_streak",
}
NUMERIC_KEYS = SIGNED_KEYS | {"industry_rank"}

SIDE_CHOICES = (("Both sides", ""), ("Longs (RS)", "LONG"), ("Shorts (RW)", "SHORT"))


class RsWindowPanel(QFrame):
    statusChanged = Signal(str)

    def __init__(self, bounce_service=None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.bounce_service = bounce_service
        self._decorated_rows: list[dict[str, Any]] = []
        # Auto mode (hands-off default): the panel refreshes the chart and
        # re-ranks on its own; the region tracks the trailing hour until the
        # trader drags it, after which their window is preserved.
        self._region_customized = False

        self.refresh_chart_button = QPushButton("Refresh SPY Chart")
        self.refresh_chart_button.clicked.connect(self.refresh_chart)
        self.rank_button = QPushButton("Rank Selected Window")
        self.rank_button.setObjectName("PrimaryButton")
        self.rank_button.clicked.connect(self.rank_selected_window)

        self.side_input = QComboBox()
        for label, value in SIDE_CHOICES:
            self.side_input.addItem(label, value)
        self.side_input.currentIndexChanged.connect(self._apply_view)

        self.favorites_input = QCheckBox("Favorite setups only")
        self.favorites_input.setToolTip(
            "Keep only names that are a current bot pick with the favorite_setup bucket."
        )
        self.favorites_input.toggled.connect(self._apply_view)

        self.sort_input = QComboBox()
        for label, value in rs_window_feed.SORT_CHOICES:
            self.sort_input.addItem(label, value)
        self.sort_input.setToolTip(
            "Strength sorts are side-aligned: longs rank by strength, shorts by weakness, "
            "so the most trade-aligned names always float to the top."
        )
        self.sort_input.currentIndexChanged.connect(self._apply_view)

        self.status_label = QLabel(
            "Auto mode: the chart and ranking refresh themselves every 5 minutes tracking the trailing "
            "hour. Drag the gold region to pin your own window (it then survives refreshes)."
        )
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)

        self.chart = SpyM5Chart()
        self.chart.regionChanged.connect(self._on_region_changed)

        self.model = TrackerTableModel(
            COLUMNS,
            percent_keys=set(),
            signed_keys=SIGNED_KEYS,
            numeric_keys=NUMERIC_KEYS,
            tooltip_keys={"industry", "sector", "setup_family"},
        )
        proxy = TrackerSortProxyModel(self)
        proxy.setSourceModel(self.model)
        self.table = DataTable()
        self.table.setModel(proxy)
        self.table.setShowGrid(False)

        self._build_layout()

        # Everything automatic: fill soon after launch, then re-rank every
        # 5 minutes from cached bars (skipped quietly while disconnected).
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(5 * 60_000)
        self._auto_timer.timeout.connect(self._auto_tick)
        self._auto_timer.start()
        QTimer.singleShot(20_000, self._auto_tick)

    def _build_layout(self) -> None:
        header = SectionHeader(
            "RS Window",
            "Select from-where-to-where on the SPY M5 chart; rank which names stayed strongest (RS) "
            "or weakest (RW) through that window, with favorite-setup, industry/sector, and D1/weekly context.",
        )
        header.add_action(self.refresh_chart_button)
        header.add_action(self.rank_button)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(QLabel("Show"))
        controls.addWidget(self.side_input)
        controls.addWidget(self.favorites_input)
        controls.addWidget(QLabel("Sort by"))
        controls.addWidget(self.sort_input)
        controls.addStretch(1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.chart)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(header)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    def _current_bot(self):
        if self.bounce_service is None:
            return None
        return self.bounce_service.current_bot()

    def _auto_tick(self) -> None:
        """Hands-off refresh: new bars + re-rank, no clicks. The region keeps
        tracking the trailing hour until the trader customizes it."""
        if self._current_bot() is None:
            return  # quiet: the bot auto-connects; next tick will catch it
        self.refresh_chart()
        if self.chart.bar_count():
            self.rank_selected_window()

    def refresh_chart(self) -> None:
        bot = self._current_bot()
        if bot is None:
            self._set_status("BounceBot is not connected yet - start it on the Trading Desk first.")
            return
        try:
            bars = bot.spy_m5_chart_bars(max_sessions=2)
        except Exception as exc:
            self._set_status(f"Could not read SPY bars: {exc}")
            return
        self.chart.set_bars(bars, preserve_selection=self._region_customized)
        if not bars:
            self._set_status(
                "No cached SPY 5m bars yet - the bot fills them once scanning (or the paused "
                "regime refresh) has run."
            )
            return
        first = bars[0]["dt"].strftime("%m/%d %H:%M")
        last = bars[-1]["dt"].strftime("%m/%d %H:%M")
        self._set_status(
            f"SPY M5 loaded: {len(bars)} bars ({first} -> {last}). Drag the gold region, then rank."
        )

    def rank_selected_window(self) -> None:
        bot = self._current_bot()
        if bot is None:
            self._set_status("BounceBot is not connected yet - start it on the Trading Desk first.")
            return
        if self.chart.bar_count() == 0:
            self.refresh_chart()
        selected = self.chart.selected_range()
        if selected is None:
            self._set_status("No chart selection yet - refresh the chart first.")
            return
        start_dt, end_dt = selected
        try:
            result = bot.rank_window_movers(start_dt, end_dt)
        except Exception as exc:
            self._set_status(f"Ranking failed: {exc}")
            return
        if not result.get("ok"):
            self._set_status(str(result.get("note") or "Ranking produced no output."))
            return
        rows = result.get("rows") or []
        try:
            self._decorated_rows = rs_window_feed.decorate_mover_rows(rows)
        except Exception as exc:
            # Context joins are best-effort: the window ranking itself must
            # never be blocked by a missing board CSV or bar file.
            self._decorated_rows = rows
            self._set_status(f"Ranked (context join skipped: {exc})")
        spy_pct = result.get("spy_pct")
        window_text = f"{start_dt.strftime('%H:%M')} -> {end_dt.strftime('%H:%M')}"
        spy_text = f"SPY {spy_pct:+.2f}%" if isinstance(spy_pct, (int, float)) else "SPY n/a"
        self._apply_view(
            status=(
                f"Window {window_text}: {spy_text}, {len(self._decorated_rows)} candidate(s) ranked."
            )
        )

    def _apply_view(self, *_args, status: str | None = None) -> None:
        rows = rs_window_feed.filter_mover_rows(
            self._decorated_rows,
            side=str(self.side_input.currentData() or ""),
            favorites_only=self.favorites_input.isChecked(),
        )
        rows = rs_window_feed.sort_mover_rows(rows, str(self.sort_input.currentData() or "excess"))
        self.model.set_rows(rows[:300])
        self.table.fit_columns()
        if status is not None:
            self._set_status(status)
        elif self._decorated_rows:
            self._set_status(f"{len(rows)} of {len(self._decorated_rows)} ranked name(s) shown.")

    def _on_region_changed(self) -> None:
        selected = self.chart.selected_range()
        if selected is None:
            return
        self._region_customized = True  # the trader's window now wins over trailing-hour tracking
        start_dt, end_dt = selected
        self._set_status(
            f"Window selected: {start_dt.strftime('%H:%M')} -> {end_dt.strftime('%H:%M')} - "
            "click Rank Selected Window."
        )

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"RS Window: {message}")
