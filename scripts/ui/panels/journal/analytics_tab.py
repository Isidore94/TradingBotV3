"""Analytics: the equity curve, the per-group breakdowns, and walk-away (§9 step 12).

Two things here are corrections rather than features.

The **honest total**: when the selection mixes currencies and anything is
unconverted, this tab shows the reason instead of a number. Adding a USD win to
a CAD loss produces 60 of nothing, and quietly dropping the unconverted rows
produces a total that looks right and is not.

The **honest curve**: a trade that cannot be converted is skipped and counted,
never absorbed as a flat step. A curve that swallows a real position as zero is
a lie in the shape a chart makes easy to believe.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.services import journal_feed

try:  # pragma: no cover - the desk has pyqtgraph; a headless box may not.
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    pg = None
    PYQTGRAPH_AVAILABLE = False


#: Buckets with fewer closed trades than this are drawn, but labeled thin. The
#: number is not a threshold anything is decided on - it is the point at which
#: a bar chart starts to look like evidence when it is not.
THIN_SAMPLE_TRADES = 5

#: How many buckets one chart shows. Beyond this the axis is unreadable, and
#: what is dropped is SAID rather than silently trimmed.
GROUP_CHART_MAX_BARS = 12


def group_breakdown_rows(summary: dict, group_name: str) -> list[dict]:
    """The chosen group's buckets, sorted the way the table already sorts."""
    groups = summary.get("groups") or {}
    return list(groups.get(group_name) or [])


def group_chart_series(rows: list[dict]) -> tuple[list[str], list[float], int]:
    """(labels with honest n, net values, dropped count) for one breakdown.

    A bucket whose net is ``None`` is EXCLUDED, never plotted as zero: None
    here means "mixed currencies with unconverted rows", and a zero bar would
    read as "this setup broke even" - a claim the data refuses to make. The
    count of what was excluded is returned so the caller can say it.
    """
    plottable = [row for row in rows if row.get("net_pnl") is not None]
    dropped = len(rows) - len(plottable)
    shown = plottable[:GROUP_CHART_MAX_BARS]
    dropped += max(0, len(plottable) - len(shown))
    labels = []
    values = []
    for row in shown:
        closed = int(row.get("closed", 0) or 0)
        label = f"{row.get('label', '')} (n={closed})"
        if closed < THIN_SAMPLE_TRADES:
            label += " thin"
        labels.append(label)
        values.append(float(row.get("net_pnl") or 0.0))
    return labels, values, dropped


#: The stat cards, in reading order: (key, title). Two rows of six.
STAT_CARDS = (
    ("net_pnl", "Net P&L"),
    ("closed", "Closed trades"),
    ("win_rate", "Win rate"),
    ("profit_factor", "Profit factor"),
    ("expectancy", "Expectancy / trade"),
    ("avg_r", "Average R"),
    ("avg_win", "Average win"),
    ("avg_loss", "Average loss"),
    ("largest_win", "Largest win"),
    ("largest_loss", "Largest loss"),
    ("max_drawdown", "Max drawdown"),
    ("streaks", "Longest streaks"),
)

#: Keys whose sign colours the card green or red.
SIGNED_CARD_KEYS = {"net_pnl", "expectancy", "avg_r"}

#: Rows of the Long vs Short table: (label, stats key, kind).
SIDE_ROWS = (
    ("Closed trades", "closed", "count"),
    ("Net P&L", "net_pnl", "money"),
    ("Win rate", "win_rate", "pct"),
    ("Profit factor", "profit_factor", "ratio"),
    ("Expectancy / trade", "expectancy", "money"),
    ("Average win", "avg_win", "money"),
    ("Average loss", "avg_loss", "money"),
    ("Largest win", "largest_win", "money"),
    ("Largest loss", "largest_loss", "money"),
    ("Max drawdown", "max_drawdown", "money"),
)

#: The breakdown table's columns.
GROUP_TABLE_COLUMNS = (
    "Bucket", "Trades", "Closed", "Win rate", "Profit factor",
    "Avg win", "Avg loss", "Expectancy", "Net",
)

#: What each breakdown means, shown as the picker's tooltip and under the chart.
GROUP_DESCRIPTIONS = {
    "my setups": "Tags you typed or confirmed. Machine guesses are never counted here.",
    "provisional setups": "Tags a machine applied that you have not confirmed yet.",
    "auto tags": "The automatic tags on every trade (time of day, shape, scanner match).",
    "weekday (entry)": "Day of the week you ENTERED the trade, market time.",
    "hour of entry": "Hour you entered the trade, New York time.",
    "hold time": "How long each closed trade was held.",
}


def format_stat(value, kind: str) -> str:
    """Text for one stat value; a dash when unknown."""
    if value is None:
        return "-"
    if kind == "count":
        return f"{int(value):,}"
    if kind == "pct":
        return f"{float(value):.0%}"
    if kind == "ratio":
        return f"{float(value):.2f}"
    if kind == "r":
        return f"{float(value):+.2f}R"
    return f"{float(value):+,.2f}"


def stat_card_values(stats: dict) -> dict[str, tuple[str, str, str]]:
    """(value text, detail text, tone) for every card in `STAT_CARDS`."""
    wins = int(stats.get("wins") or 0)
    losses = int(stats.get("losses") or 0)
    breakeven = int(stats.get("breakeven") or 0)
    open_count = max(0, int(stats.get("trades") or 0) - int(stats.get("closed") or 0)
                     - int(stats.get("unpriced") or 0))
    avg_win, avg_loss = stats.get("avg_win"), stats.get("avg_loss")
    payoff = stats.get("payoff_ratio")

    def tone(key):
        value = stats.get(key)
        if value is None or key not in SIGNED_CARD_KEYS | {"largest_win", "largest_loss",
                                                           "avg_win", "avg_loss", "max_drawdown"}:
            return "none"
        if abs(float(value)) < 0.005:
            return "flat"
        return "win" if float(value) > 0 else "loss"

    pf = stats.get("profit_factor")
    values = {
        "net_pnl": (format_stat(stats.get("net_pnl"), "money"), "", tone("net_pnl")),
        "closed": (
            format_stat(stats.get("closed"), "count"),
            f"{wins}W {losses}L" + (f" {breakeven} even" if breakeven else "")
            + (f", {open_count} open" if open_count else ""),
            "none",
        ),
        "win_rate": (format_stat(stats.get("win_rate"), "pct"), f"{wins} of {stats.get('closed') or 0}", "none"),
        "profit_factor": (
            format_stat(pf, "ratio"),
            "gross win / gross loss" if pf is not None else "no losing trades yet",
            "none" if pf is None else ("win" if pf >= 1 else "loss"),
        ),
        "expectancy": (format_stat(stats.get("expectancy"), "money"), "net per closed trade", tone("expectancy")),
        "avg_r": (
            format_stat(stats.get("avg_r"), "r"),
            f"{stats.get('r_trades') or 0} trade(s) with a planned risk",
            tone("avg_r"),
        ),
        "avg_win": (format_stat(avg_win, "money"), "", tone("avg_win")),
        "avg_loss": (
            format_stat(avg_loss, "money"),
            f"win/loss size {payoff:.2f}" if payoff is not None else "",
            tone("avg_loss"),
        ),
        "largest_win": (format_stat(stats.get("largest_win"), "money"), "", tone("largest_win")),
        "largest_loss": (format_stat(stats.get("largest_loss"), "money"), "", tone("largest_loss")),
        "max_drawdown": (
            format_stat(stats.get("max_drawdown"), "money"),
            "worst drop from a peak",
            "loss" if (stats.get("max_drawdown") or 0) < -0.005 else "none",
        ),
        "streaks": (
            f"{int(stats.get('max_win_streak') or 0)}W / {int(stats.get('max_loss_streak') or 0)}L",
            _current_streak_text(int(stats.get("current_streak") or 0)),
            "none",
        ),
    }
    return values


def _current_streak_text(streak: int) -> str:
    if streak > 0:
        return f"now {streak} win(s) in a row"
    if streak < 0:
        return f"now {-streak} loss(es) in a row"
    return ""


def group_table_row(stats: dict) -> list[str]:
    """One breakdown bucket as the table's text cells, in `GROUP_TABLE_COLUMNS` order."""
    from journal_analytics import group_expectancy

    expectancy = stats.get("expectancy")
    if expectancy is None and "expectancy" not in stats:
        expectancy = group_expectancy(stats)
    net = stats.get("net_pnl")
    return [
        str(stats.get("label", "")),
        str(stats.get("trades", 0)),
        str(stats.get("closed", 0)),
        format_stat(stats.get("win_rate"), "pct"),
        format_stat(stats.get("profit_factor"), "ratio"),
        format_stat(stats.get("avg_win") if net is not None else None, "money"),
        format_stat(stats.get("avg_loss") if net is not None else None, "money"),
        format_stat(expectancy, "money"),
        format_stat(net, "money"),
    ]


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


class StatCard(QFrame):
    """One headline number with a title and a small detail line."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("JournalStatCard")
        self.setProperty("tone", "none")
        self._tone = "none"
        self.title_label = QLabel(title)
        self.title_label.setObjectName("JournalStatTitle")
        self.value_label = QLabel("-")
        self.value_label.setObjectName("JournalStatValue")
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("JournalStatDetail")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(1)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def show_value(self, value: str, detail: str, tone: str) -> None:
        self.value_label.setText(value)
        self.detail_label.setText(detail)
        if tone != self._tone:
            self._tone = tone
            self.setProperty("tone", tone)
            _repolish(self)


class _WalkawayWorker(QThread):
    """Walk-away replays daily bars, so it never runs on the GUI thread."""

    finished_with = Signal(dict)
    failed = Signal(str)

    def __init__(self, since, until, parent=None) -> None:
        super().__init__(parent)
        self._since = since
        self._until = until

    def run(self) -> None:  # pragma: no cover - exercised on the desk
        try:
            self.finished_with.emit(journal_feed.walkaway_summary(self._since, self._until))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class AnalyticsTab(QFrame):
    statusChanged = Signal(str)

    def __init__(self, header, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._header = header
        self._worker: _WalkawayWorker | None = None

        self.headline = QLabel("")
        self.headline.setObjectName("SectionTitle")
        self.headline.setWordWrap(True)
        self.currency_badge = QLabel("")
        self.currency_badge.setObjectName("JournalCurrencyBadge")
        self.currency_badge.setToolTip(
            "Every money number on this tab is in this currency. Change it in the header."
        )
        self.currency_note = QLabel("")
        self.currency_note.setObjectName("CurrencyNote")
        self.currency_note.setWordWrap(True)
        #: ST5.4. Two sentences the charts below cannot say for themselves: how
        #: much of this journal is the trader's own answer, and how much of it
        #: carries a plan. Both are counted over CLOSED trades - the same
        #: denominator every number on this tab uses.
        self.evidence_note = QLabel("")
        self.evidence_note.setObjectName("MutedLabel")
        self.evidence_note.setWordWrap(True)
        #: Closed trades with a made-up entry: kept, but not in any total.
        self.not_counted_note = QLabel("")
        self.not_counted_note.setObjectName("CurrencyNote")
        self.not_counted_note.setWordWrap(True)
        self.not_counted_note.setVisible(False)

        cards_host = QWidget()
        cards_grid = QGridLayout(cards_host)
        cards_grid.setContentsMargins(0, 0, 0, 0)
        cards_grid.setSpacing(6)
        self.stat_cards: dict[str, StatCard] = {}
        for index, (key, title) in enumerate(STAT_CARDS):
            card = StatCard(title)
            cards_grid.addWidget(card, index // 6, index % 6)
            self.stat_cards[key] = card

        self.side_table = QTableWidget(len(SIDE_ROWS), 2)
        self.side_table.setHorizontalHeaderLabels(["Long", "Short"])
        self.side_table.setVerticalHeaderLabels([label for label, _key, _kind in SIDE_ROWS])
        self.side_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.side_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.curve = pg.PlotWidget(
            title="Cumulative P&L (closed trades)", background=theme.color("bg_panel")
        ) if PYQTGRAPH_AVAILABLE else QLabel(
            "pyqtgraph is not installed; the table below carries the same numbers."
        )
        self.curve_table = QTableWidget(0, 2)
        self.curve_table.setHorizontalHeaderLabels(["Date", "Cumulative"])
        self.curve_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.curve_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.curve_table.setMinimumHeight(200)

        self.groups_table = QTableWidget(0, len(GROUP_TABLE_COLUMNS))
        self.groups_table.setHorizontalHeaderLabels(list(GROUP_TABLE_COLUMNS))
        self.groups_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.groups_table.setMinimumHeight(240)
        self.groups_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # R7's deferred per-group charts, built 2026-08-18. The table below
        # already carries every number; what was missing is the SHAPE, and the
        # n count beside each bar is what stops a two-trade setup from looking
        # like a finding.
        self.group_picker = QComboBox()
        self.group_picker.setMinimumWidth(220)
        self.group_chart = (
            pg.PlotWidget(title="Net by group", background=theme.color("bg_panel"))
            if PYQTGRAPH_AVAILABLE
            else QLabel("pyqtgraph is not installed; the table below carries the same numbers.")
        )
        self.group_chart.setFixedHeight(260)
        self.group_note = QLabel("")
        self.group_note.setWordWrap(True)
        self.group_picker.currentTextChanged.connect(lambda _text: self._on_group_picked())
        self.group_csv_button = QPushButton("Export this breakdown CSV")
        self.group_csv_button.clicked.connect(self._export_group_csv)
        self._summary: dict = {}

        self.walkaway_button = QPushButton("Run walk-away for this range")
        self.walkaway_button.clicked.connect(self._run_walkaway)
        self.walkaway_output = QLabel("Walk-away has not been run for this range yet.")
        self.walkaway_output.setWordWrap(True)

        self.export_button = QPushButton("Export trades CSV")
        self.export_button.clicked.connect(self._export)

        buttons = QHBoxLayout()
        buttons.addWidget(self.walkaway_button)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.group_csv_button)
        buttons.addStretch(1)

        title_row = QHBoxLayout()
        title_row.addWidget(self.headline, 1)
        title_row.addWidget(self.currency_badge, 0, Qt.AlignTop)

        side_box = QVBoxLayout()
        side_title = QLabel("Long vs Short")
        side_title.setObjectName("SectionTitle")
        side_box.addWidget(side_title)
        side_box.addWidget(self.side_table, 1)
        curve_host = QWidget()
        curve_host.setFixedHeight(400)
        curve_row = QHBoxLayout(curve_host)
        curve_row.setContentsMargins(0, 0, 0, 0)
        curve_row.addWidget(self.curve, 3)
        curve_row.addLayout(side_box, 2)

        picker_row = QHBoxLayout()
        breakdown_title = QLabel("Break down by")
        breakdown_title.setObjectName("SectionTitle")
        picker_row.addWidget(breakdown_title)
        picker_row.addWidget(self.group_picker)
        picker_row.addStretch(1)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 4, 8, 0)
        layout.addLayout(title_row)
        layout.addWidget(self.currency_note)
        layout.addWidget(self.not_counted_note)
        layout.addWidget(cards_host)
        layout.addWidget(self.evidence_note)
        layout.addWidget(curve_host)
        layout.addLayout(picker_row)
        layout.addWidget(self.group_note)
        layout.addWidget(self.group_chart)
        layout.addWidget(self.groups_table)
        daily_title = QLabel("Cumulative P&L by date")
        daily_title.setObjectName("SectionTitle")
        layout.addWidget(daily_title)
        layout.addWidget(self.curve_table)
        layout.addLayout(buttons)
        layout.addWidget(self.walkaway_output)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setWidget(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scroll)

    def reload(self) -> None:
        try:
            trades = journal_feed.load_trades(**self._header.query())
        except Exception as exc:  # noqa: BLE001
            self.statusChanged.emit(f"analytics unavailable: {exc}")
            return
        from journal_analytics import (
            direction_split_stats,
            pnl_currency_label,
            time_breakdown_groups,
            trade_performance_stats,
        )

        mode = self._header.currency_mode
        summary = journal_feed.analytics_summary(trades, mode)
        raw = [trade.raw for trade in trades]
        pnl_key = str(summary.get("pnl_key") or "")
        currency = pnl_currency_label(mode, pnl_key, summary.get("currencies"))
        self.currency_badge.setText(f"Numbers in {currency}")

        overall = summary.get("overall") or {}
        net = overall.get("net_pnl")
        win_rate = overall.get("win_rate")
        self.headline.setText(
            f"{overall.get('trades', 0)} trades, {overall.get('closed', 0)} closed"
            + (f", win rate {win_rate:.0%}" if win_rate is not None else "")
            + (f", net {net:,.2f} {currency}" if net is not None else ", net not shown")
        )
        # The refusal, rendered. `pnl_note` carries the reason the total is
        # missing, and a missing total with a reason beats a wrong one.
        note = str(summary.get("pnl_note") or "")
        unconvertible = journal_feed.unconvertible_count(trades, mode)
        if unconvertible:
            note = (note + " " if note else "") + (
                f"{unconvertible} closed trade(s) have no booked {mode} rate and are not in the curve."
            )
        self.currency_note.setText(note)
        self.currency_note.setVisible(bool(note))
        not_counted_line = str((summary.get("not_counted") or {}).get("line") or "")
        self.not_counted_note.setText(not_counted_line)
        self.not_counted_note.setVisible(bool(not_counted_line))

        stats = trade_performance_stats(raw, pnl_key)
        for key, (value, detail, tone) in stat_card_values(stats).items():
            self.stat_cards[key].show_value(value, detail, tone)
        sides = direction_split_stats(raw, pnl_key)
        for column, side in enumerate(("LONG", "SHORT")):
            side_stats = sides.get(side) or {}
            for row, (_label, key, kind) in enumerate(SIDE_ROWS):
                item = QTableWidgetItem(format_stat(side_stats.get(key), kind))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.side_table.setItem(row, column, item)

        # ST5.4: the coverage line and the refusal that stands where a "best
        # personal setup" would otherwise be named. In memory over the rows
        # already loaded - no second query, nothing new on the Qt thread.
        evidence = summary.get("personal_evidence") or {}
        coverage_line = str((evidence.get("coverage") or {}).get("line") or "")
        headline_line = str(evidence.get("headline") or "")
        self.evidence_note.setText(
            " ".join(part for part in (coverage_line, headline_line) if part)
        )
        self.evidence_note.setVisible(bool(coverage_line or headline_line))

        points = journal_feed.equity_curve(trades, mode)
        if PYQTGRAPH_AVAILABLE:
            self.curve.clear()
            if points:
                self.curve.plot(
                    list(range(len(points))),
                    [value for _day, value in points],
                    pen=pg.mkPen(theme.color("accent"), width=2),
                )
        self.curve_table.setRowCount(len(points))
        for row, (day, value) in enumerate(points):
            self.curve_table.setItem(row, 0, QTableWidgetItem(day))
            self.curve_table.setItem(row, 1, QTableWidgetItem(f"{value:,.2f}"))

        summary["groups"] = {**(summary.get("groups") or {}), **time_breakdown_groups(raw, pnl_key)}
        self._summary = summary
        self._sync_group_picker(summary["groups"])
        self._on_group_picked()

    def _on_group_picked(self) -> None:
        self._draw_group_chart()
        self._fill_groups_table()

    def _fill_groups_table(self) -> None:
        """The picked breakdown only, so confirmed and provisional tags never share a table."""
        rows = group_breakdown_rows(self._summary, self.group_picker.currentText())
        self.groups_table.setRowCount(len(rows))
        for row, stats in enumerate(rows):
            for column, text in enumerate(group_table_row(stats)):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.groups_table.setItem(row, column, item)

    def _sync_group_picker(self, groups: dict) -> None:
        names = list(groups.keys())
        existing = [self.group_picker.itemText(i) for i in range(self.group_picker.count())]
        if names == existing:
            return
        current = self.group_picker.currentText()
        self.group_picker.blockSignals(True)
        self.group_picker.clear()
        self.group_picker.addItems(names)
        for index, name in enumerate(names):
            description = GROUP_DESCRIPTIONS.get(name)
            if description:
                self.group_picker.setItemData(index, description, Qt.ToolTipRole)
        if current in names:
            self.group_picker.setCurrentText(current)
        self.group_picker.blockSignals(False)

    def _draw_group_chart(self) -> None:
        group_name = self.group_picker.currentText()
        rows = group_breakdown_rows(self._summary, group_name)
        # R1: `group_notes` was written and nothing read it. The note is the
        # whole point of the coverage check - a bar chart of five tagged trades
        # beside a full one, at the same width, with nothing saying which is
        # which. PREPENDED to the group's own label, as the packet asked.
        coverage_note = str((self._summary.get("group_notes") or {}).get(group_name) or "")
        labels, values, dropped = group_chart_series(rows)
        if PYQTGRAPH_AVAILABLE:
            self.group_chart.clear()
            if values:
                brushes = [
                    pg.mkBrush(theme.color("long" if value >= 0 else "short")) for value in values
                ]
                bars = pg.BarGraphItem(
                    x=list(range(len(values))), height=values, width=0.6, brushes=brushes
                )
                self.group_chart.addItem(bars)
                self.group_chart.getAxis("bottom").setTicks([list(enumerate(labels))])
            self.group_chart.setTitle(f"Net by {group_name}" if group_name else "Net by group")
        parts = []
        if coverage_note:
            parts.append(coverage_note)
        if GROUP_DESCRIPTIONS.get(group_name):
            parts.append(GROUP_DESCRIPTIONS[group_name])
        if not rows:
            parts.append("No trades in this range for that grouping.")
        else:
            thin = sum(1 for row in rows if int(row.get("closed", 0) or 0) < THIN_SAMPLE_TRADES)
            parts.append(
                f"{len(labels)} bucket(s) charted; n is closed trades. "
                f"{thin} bucket(s) have fewer than {THIN_SAMPLE_TRADES} closed trades "
                "and are labeled thin."
            )
        if dropped:
            # Said out loud: a chart that silently drops buckets reads as
            # "that was all of them".
            parts.append(
                f"{dropped} bucket(s) not charted (no convertible total, or beyond the "
                f"{GROUP_CHART_MAX_BARS}-bar cap). They are still in the table below."
            )
        if group_name in (self._summary.get("nonexclusive_groups") or []):
            parts.append(
                "One trade can carry several tags here, so these buckets overlap and "
                "do not sum to the headline."
            )
        if group_name in (self._summary.get("provisional_groups") or []):
            # P6a. Said on the chart itself rather than only in the group's name:
            # this is the one breakdown on the page whose buckets nobody has
            # agreed to yet, and a bar is a bar.
            parts.append(
                "These tags were applied for you and are still waiting for review - "
                "confirm or correct them in the Trades tab, where they are marked "
                "provisional. They are never counted under \"my setups\"."
            )
        self.group_note.setText(" ".join(parts))

    def _export_group_csv(self) -> None:
        """The charted breakdown, exactly as shown, as a CSV beside it."""
        group_name = self.group_picker.currentText()
        rows = group_breakdown_rows(self._summary, group_name)
        if not rows:
            self.statusChanged.emit("nothing to export for that grouping")
            return
        try:
            from project_paths import JOURNAL_EXPORT_DIR

            slug = "".join(ch if ch.isalnum() else "_" for ch in group_name) or "group"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = Path(JOURNAL_EXPORT_DIR) / f"journal_by_{slug}_{stamp}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            from journal_analytics import group_expectancy

            columns = [
                "label", "trades", "closed", "win_rate", "profit_factor",
                "avg_win", "avg_loss", "expectancy", "net_pnl",
            ]
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    values = {**row, "expectancy": row.get("expectancy", group_expectancy(row))}
                    writer.writerow({column: values.get(column, "") for column in columns})
        except Exception as exc:  # noqa: BLE001
            self.statusChanged.emit(f"breakdown export failed: {exc}")
            return
        self.statusChanged.emit(f"exported {target}")

    def _run_walkaway(self) -> None:  # pragma: no cover - worker path
        if self._worker is not None and self._worker.isRunning():
            return
        since, until = self._header.date_bounds()
        self.walkaway_button.setEnabled(False)
        self.walkaway_output.setText("Running walk-away...")
        self._worker = _WalkawayWorker(since, until, self)
        self._worker.finished_with.connect(self._on_walkaway_done)
        self._worker.failed.connect(self._on_walkaway_failed)
        self._worker.start()

    def _on_walkaway_done(self, result: dict) -> None:  # pragma: no cover
        self.walkaway_button.setEnabled(True)
        self.walkaway_output.setText(journal_feed.render_walkaway_summary(result))

    def _on_walkaway_failed(self, message: str) -> None:  # pragma: no cover
        self.walkaway_button.setEnabled(True)
        self.walkaway_output.setText(f"Walk-away failed: {message}")
        self.statusChanged.emit(f"walk-away failed: {message}")

    def _export(self) -> None:
        try:
            path = journal_feed.export_trades_csv()
        except Exception as exc:  # noqa: BLE001
            self.statusChanged.emit(f"export failed: {exc}")
            return
        self.statusChanged.emit(f"exported {path}")

    def shutdown(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(2000)
