"""Calendar: a Mon-Fri month grid with weekly totals, and a 12-month strip.

Each day cell shows the day's net P&L large, with the trade count and wins /
losses under it, tinted green or red by result. A day with no trades is plain;
a breakeven day has its own grey tint. Clicking a day filters the Trades tab.

The calendar shows the whole selected year. The header's Range filter does not
apply here; its account, symbol, status, direction and tag filters do.
"""

from __future__ import annotations

import calendar as calendar_module
from datetime import date

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.services import journal_feed

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")

#: A day within this of zero is breakeven: grey, never green or red.
FLAT_EPSILON = 0.005

#: A day at or above this share of the month's biggest day gets the strong tint.
STRONG_TONE_SHARE = 0.5


def year_heatmap_matrix(by_day: dict, year: int) -> tuple[list[list[float | None]], float]:
    """A 12x31 grid of daily P&L for ``year``, and the largest absolute day.

    ``None`` marks a day with no trading, which is never drawn as zero.
    """
    matrix: list[list[float | None]] = [[None] * 31 for _ in range(12)]
    scale = 0.0
    for key, value in (by_day or {}).items():
        try:
            when = date.fromisoformat(str(key))
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if when.year != int(year):
            continue
        matrix[when.month - 1][when.day - 1] = amount
        scale = max(scale, abs(amount))
    return matrix, scale


def pnl_tone(value: float | None, scale: float = 0.0) -> str:
    """Tint for one P&L cell: none, flat, win, win_strong, loss or loss_strong."""
    if value is None:
        return "none"
    if abs(value) <= FLAT_EPSILON:
        return "flat"
    strong = scale > 0 and abs(value) >= STRONG_TONE_SHARE * scale
    if value > 0:
        return "win_strong" if strong else "win"
    return "loss_strong" if strong else "loss"


def format_pnl(value: float | None, *, decimals: int = 0) -> str:
    """Signed money text, e.g. ``+1,234`` or ``-56``; a dash for unknown."""
    if value is None:
        return "-"
    if abs(value) <= FLAT_EPSILON:
        return f"{0:.{decimals}f}"
    return f"{value:+,.{decimals}f}"


def month_layout(day_stats: dict, year: int, month: int) -> dict:
    """The month as Mon-Fri weeks with weekly totals, plus a summary.

    Weekend trades are never dropped: they count in their week's total and in
    the month summary, which says how many weekend days had trades.
    """
    weeks = []
    scale = max(
        (
            abs(float(entry.get("net") or 0.0))
            for key, entry in (day_stats or {}).items()
            if str(key).startswith(f"{year:04d}-{month:02d}-")
        ),
        default=0.0,
    )
    summary = {
        "net": None, "trades": 0, "wins": 0, "losses": 0, "days_traded": 0,
        "green_days": 0, "red_days": 0, "flat_days": 0, "best": None, "worst": None,
        "weekend_days": 0,
    }
    for week in calendar_module.Calendar(firstweekday=0).monthdayscalendar(year, month):
        cells: list[dict | None] = []
        week_net: float | None = None
        week_trades = 0
        for weekday, day in enumerate(week):
            if not day:
                if weekday < 5:
                    cells.append(None)
                continue
            key = date(year, month, day).isoformat()
            entry = (day_stats or {}).get(key)
            net = None if entry is None else float(entry.get("net") or 0.0)
            trades = 0 if entry is None else int(entry.get("trades") or 0)
            if net is not None:
                week_net = (week_net or 0.0) + net
                week_trades += trades
                summary["net"] = (summary["net"] or 0.0) + net
                summary["trades"] += trades
                summary["wins"] += int(entry.get("wins") or 0)
                summary["losses"] += int(entry.get("losses") or 0)
                summary["days_traded"] += 1
                tone = pnl_tone(net)
                if tone == "flat":
                    summary["flat_days"] += 1
                elif net > 0:
                    summary["green_days"] += 1
                else:
                    summary["red_days"] += 1
                if summary["best"] is None or net > summary["best"][1]:
                    summary["best"] = (key, net)
                if summary["worst"] is None or net < summary["worst"][1]:
                    summary["worst"] = (key, net)
                if weekday >= 5:
                    summary["weekend_days"] += 1
            if weekday < 5:
                cells.append(
                    {
                        "date": key,
                        "day": day,
                        "net": net,
                        "trades": trades,
                        "wins": 0 if entry is None else int(entry.get("wins") or 0),
                        "losses": 0 if entry is None else int(entry.get("losses") or 0),
                        "tone": pnl_tone(net, scale),
                    }
                )
        if any(cells) or week_net is not None:
            weeks.append({"days": cells, "net": week_net, "trades": week_trades})
    return {"weeks": weeks, "summary": summary}


def year_month_totals(day_stats: dict, year: int) -> list[dict]:
    """Twelve month cards for ``year``: net, trading days, green and red days."""
    months = [
        {"month": m, "label": calendar_module.month_abbr[m], "net": None, "days": 0,
         "green": 0, "red": 0, "trades": 0}
        for m in range(1, 13)
    ]
    for key, entry in (day_stats or {}).items():
        try:
            when = date.fromisoformat(str(key))
        except ValueError:
            continue
        if when.year != int(year):
            continue
        card = months[when.month - 1]
        net = float(entry.get("net") or 0.0)
        card["net"] = (card["net"] or 0.0) + net
        card["days"] += 1
        card["trades"] += int(entry.get("trades") or 0)
        if net > FLAT_EPSILON:
            card["green"] += 1
        elif net < -FLAT_EPSILON:
            card["red"] += 1
    return months


def month_summary_text(summary: dict, currency: str) -> str:
    """One line over the grid: net, green/red days, best and worst day."""
    if not summary.get("days_traded"):
        return "No closed trades this month."
    parts = [
        f"Net {format_pnl(summary['net'], decimals=2)} {currency}",
        f"{summary['days_traded']} day(s) traded: {summary['green_days']} green, "
        f"{summary['red_days']} red"
        + (f", {summary['flat_days']} flat" if summary.get("flat_days") else ""),
        f"{summary['trades']} trade(s)",
    ]
    for label, key in (("Best", "best"), ("Worst", "worst")):
        found = summary.get(key)
        if found:
            when = date.fromisoformat(found[0])
            parts.append(f"{label} {when.strftime('%b')} {when.day} {format_pnl(found[1])}")
    if summary.get("weekend_days"):
        parts.append(f"{summary['weekend_days']} weekend day(s) with trades, counted in the week totals")
    return "  |  ".join(parts)


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


class _ClickableCell(QFrame):
    """A tinted cell whose tone is a `theme.qss` property, never a stylesheet."""

    clicked = Signal(str)

    def __init__(self, object_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setProperty("tone", "none")
        self._key = ""
        self._tone = "none"

    def set_key(self, key: str) -> None:
        self._key = key
        self.setCursor(Qt.PointingHandCursor if key else Qt.ArrowCursor)

    def set_tone(self, tone: str) -> None:
        if tone != self._tone:
            self._tone = tone
            self.setProperty("tone", tone)
            _repolish(self)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._key and event.button() == Qt.LeftButton:
            self.clicked.emit(self._key)
        super().mousePressEvent(event)


class DayCell(_ClickableCell):
    def __init__(self, parent: QWidget | None = None, *, week: bool = False) -> None:
        super().__init__("JournalWeekCell" if week else "JournalDayCell", parent)
        self.day_label = QLabel("")
        self.day_label.setObjectName("JournalCellDay")
        self.net_label = QLabel("")
        self.net_label.setObjectName("JournalCellNet")
        self.net_label.setAlignment(Qt.AlignCenter)
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("JournalCellDetail")
        self.detail_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(0)
        layout.addWidget(self.day_label)
        layout.addStretch(1)
        layout.addWidget(self.net_label)
        layout.addWidget(self.detail_label)
        layout.addStretch(1)
        self.setMinimumSize(110, 74)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def show_day(self, cell: dict | None) -> None:
        if cell is None:
            self.set_key("")
            self.day_label.setText("")
            self.net_label.setText("")
            self.detail_label.setText("")
            self.set_tone("outside")
            return
        self.set_key(cell["date"])
        self.day_label.setText(str(cell["day"]))
        if cell["net"] is None:
            self.net_label.setText("")
            self.detail_label.setText("")
        else:
            self.net_label.setText(format_pnl(cell["net"]))
            self.detail_label.setText(
                f"{cell['trades']} trade{'s' if cell['trades'] != 1 else ''}  "
                f"{cell['wins']}W {cell['losses']}L"
            )
        self.set_tone(cell["tone"])
        self.setToolTip(
            f"{cell['date']}: net {format_pnl(cell['net'], decimals=2)}. Click to see these trades."
            if cell["net"] is not None
            else f"{cell['date']}: no closed trades."
        )

    def show_week(self, week: dict | None, index: int) -> None:
        self.set_key("")
        if week is None:
            self.day_label.setText("")
            self.net_label.setText("")
            self.detail_label.setText("")
            self.set_tone("outside")
            return
        self.day_label.setText(f"Week {index + 1}")
        self.net_label.setText(format_pnl(week["net"]) if week["net"] is not None else "-")
        self.detail_label.setText(
            f"{week['trades']} trade{'s' if week['trades'] != 1 else ''}" if week["trades"] else ""
        )
        self.set_tone(pnl_tone(week["net"]))


class MonthCard(_ClickableCell):
    def __init__(self, month: int, parent: QWidget | None = None) -> None:
        super().__init__("JournalMonthCard", parent)
        self.month = month
        self.name_label = QLabel(calendar_module.month_abbr[month])
        self.name_label.setObjectName("JournalCellDay")
        self.net_label = QLabel("-")
        self.net_label.setObjectName("JournalMonthNet")
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("JournalCellDetail")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(0)
        layout.addWidget(self.name_label)
        layout.addWidget(self.net_label)
        layout.addWidget(self.detail_label)
        self.set_key(str(month))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def show_month(self, card: dict, selected: bool) -> None:
        self.net_label.setText(format_pnl(card["net"]) if card["net"] is not None else "-")
        self.detail_label.setText(
            f"{card['days']}d  {card['green']}G {card['red']}R" if card["days"] else "no trades"
        )
        self.set_tone(pnl_tone(card["net"]))
        if bool(self.property("selected")) != selected:
            self.setProperty("selected", selected)
            _repolish(self)


class CalendarTab(QFrame):
    daySelected = Signal(str)
    statusChanged = Signal(str)

    def __init__(self, header, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._header = header
        self._by_day: dict[str, float] = {}
        self._day_stats: dict[str, dict] = {}
        self._loaded_year: int | None = None
        self._currency = ""
        today = date.today()

        self.prev_button = QPushButton("<")
        self.prev_button.setToolTip("Previous month")
        self.prev_button.clicked.connect(lambda: self._step_month(-1))
        self.next_button = QPushButton(">")
        self.next_button.setToolTip("Next month")
        self.next_button.clicked.connect(lambda: self._step_month(1))
        self.today_button = QPushButton("This month")
        self.today_button.clicked.connect(self._go_to_today)
        self.year_input = QComboBox()
        self.year_input.addItems([str(year) for year in range(today.year - 4, today.year + 1)])
        self.year_input.setCurrentText(str(today.year))
        self.year_input.currentTextChanged.connect(self._on_year_changed)
        self.month_input = QComboBox()
        self.month_input.addItems([calendar_module.month_name[m] for m in range(1, 13)])
        self.month_input.setCurrentIndex(today.month - 1)
        self.month_input.currentIndexChanged.connect(self._render)
        self.currency_badge = QLabel("")
        self.currency_badge.setObjectName("JournalCurrencyBadge")
        self.currency_badge.setToolTip("Every number on this tab is in this currency. Change it in the header.")

        controls = QHBoxLayout()
        controls.addWidget(self.prev_button)
        controls.addWidget(self.month_input)
        controls.addWidget(self.year_input)
        controls.addWidget(self.next_button)
        controls.addWidget(self.today_button)
        controls.addStretch(1)
        controls.addWidget(self.currency_badge)

        self.summary = QLabel("")
        self.summary.setObjectName("JournalSummaryStrip")
        self.summary.setWordWrap(True)
        self.range_note = QLabel(
            "Shows the whole year. The header's Range does not apply here; its other filters do. "
            "Click a day to see its trades."
        )
        self.range_note.setObjectName("MutedLabel")
        self.range_note.setWordWrap(True)

        grid_host = QWidget()
        self.grid_layout = QGridLayout(grid_host)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(4)
        for column, name in enumerate((*WEEKDAYS, "Week")):
            label = QLabel(name)
            label.setObjectName("JournalCalendarHead")
            label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(label, 0, column)
        self.day_cells: list[list[DayCell]] = []
        self.week_cells: list[DayCell] = []
        for row in range(6):
            cells = []
            for column in range(5):
                cell = DayCell()
                cell.clicked.connect(self.daySelected.emit)
                self.grid_layout.addWidget(cell, row + 1, column)
                cells.append(cell)
            self.day_cells.append(cells)
            week_cell = DayCell(week=True)
            self.grid_layout.addWidget(week_cell, row + 1, 5)
            self.week_cells.append(week_cell)
        for column in range(5):
            self.grid_layout.setColumnStretch(column, 3)
        self.grid_layout.setColumnStretch(5, 2)

        year_host = QWidget()
        year_layout = QGridLayout(year_host)
        year_layout.setContentsMargins(0, 0, 0, 0)
        year_layout.setSpacing(4)
        self.month_cards: list[MonthCard] = []
        for index in range(12):
            card = MonthCard(index + 1)
            card.clicked.connect(self._on_month_card_clicked)
            year_layout.addWidget(card, 0, index)
            self.month_cards.append(card)
        self.year_title = QLabel("Year")
        self.year_title.setObjectName("SectionTitle")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.summary)
        layout.addWidget(grid_host, 1)
        layout.addWidget(self.year_title)
        layout.addWidget(year_host)
        layout.addWidget(self.range_note)

    # -- data ----------------------------------------------------------------

    def _selected_year(self) -> int:
        try:
            return int(self.year_input.currentText())
        except ValueError:
            return date.today().year

    def reload(self) -> None:
        year = self._selected_year()
        query = dict(self._header.query())
        query["date_from"] = date(year, 1, 1)
        query["date_to"] = date(year, 12, 31)
        try:
            data = journal_feed.calendar_month_data(
                currency_mode=self._header.currency_mode, **query
            )
        except Exception as exc:  # noqa: BLE001
            data = {"days": {}, "note": "", "currency": "", "not_counted": ""}
            self.statusChanged.emit(f"calendar unavailable: {exc}")
        self._day_stats = dict(data.get("days") or {})
        self._by_day = {key: float(entry.get("net") or 0.0) for key, entry in self._day_stats.items()}
        self._currency = str(data.get("currency") or "")
        self._not_counted_line = str(data.get("not_counted") or "")
        note = str(data.get("note") or "")
        self._loaded_year = year
        self.currency_badge.setText(f"Numbers in {self._currency}" if self._currency else "")
        if note and not self._day_stats:
            self.statusChanged.emit(f"calendar: {note}")
        self._render()

    # -- navigation ----------------------------------------------------------

    def _on_year_changed(self, _text: str = "") -> None:
        if self._loaded_year is not None and self._selected_year() != self._loaded_year:
            self.reload()
        else:
            self._render()

    def _step_month(self, delta: int) -> None:
        month = self.month_input.currentIndex() + delta
        year = self._selected_year()
        if month < 0:
            month, year = 11, year - 1
        elif month > 11:
            month, year = 0, year + 1
        if self.year_input.findText(str(year)) < 0:
            return
        self.month_input.blockSignals(True)
        self.month_input.setCurrentIndex(month)
        self.month_input.blockSignals(False)
        if str(year) != self.year_input.currentText():
            self.year_input.setCurrentText(str(year))
        else:
            self._render()

    def _go_to_today(self) -> None:
        today = date.today()
        self.month_input.blockSignals(True)
        self.month_input.setCurrentIndex(today.month - 1)
        self.month_input.blockSignals(False)
        if self.year_input.currentText() != str(today.year):
            self.year_input.setCurrentText(str(today.year))
        else:
            self._render()

    def _on_month_card_clicked(self, key: str) -> None:
        self.month_input.setCurrentIndex(int(key) - 1)

    # -- drawing -------------------------------------------------------------

    def _render(self, *_args) -> None:
        year = self._selected_year()
        month = self.month_input.currentIndex() + 1
        layout = month_layout(self._day_stats, year, month)
        weeks = layout["weeks"]
        for row in range(6):
            week = weeks[row] if row < len(weeks) else None
            visible = week is not None
            for column in range(5):
                cell = self.day_cells[row][column]
                cell.setVisible(visible)
                if visible:
                    cell.show_day(week["days"][column])
            self.week_cells[row].setVisible(visible)
            if visible:
                self.week_cells[row].show_week(week, row)
        currency = self._currency or self._header.currency_mode
        text = f"{calendar_module.month_name[month]} {year}:  " + month_summary_text(layout["summary"], currency)
        not_counted = getattr(self, "_not_counted_line", "")
        self.summary.setText(f"{text}  |  {not_counted}" if not_counted else text)
        month_totals = year_month_totals(self._day_stats, year)
        for card, totals in zip(self.month_cards, month_totals):
            card.show_month(totals, totals["month"] == month)
        year_net = sum(card["net"] or 0.0 for card in month_totals)
        self.year_title.setText(
            f"{year} by month  (net {format_pnl(year_net, decimals=2)} {currency}, click a month to open it)"
        )
