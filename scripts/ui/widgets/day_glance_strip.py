"""The Day Review glance strip: one row of numbers at the top of the page.

Formatting only. The numbers come from `day_report_card.glance`, built on the
Day Review worker; this widget turns them into short tiles, each with a tooltip,
and says which tile was clicked. Unknown is "not measured", never a zero.
"""

from __future__ import annotations

from typing import Any, Mapping

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QSizePolicy, QWidget

#: Tile order, left to right, with the small label under each number.
TILES: tuple[tuple[str, str], ...] = (
    ("pnl", "day P&L"),
    ("trades", "trades W / L"),
    ("planned", "planned / unplanned"),
    ("calls", "calls right / wrong / flat"),
    ("day_type", "day type"),
    ("market_axes", "market read right / wrong"),
    ("biggest_win", "biggest win"),
    ("biggest_miss", "biggest miss"),
)

NOT_MEASURED = "not measured"


def _amp(text: str) -> str:
    """Show `&` on a button instead of reading it as a mnemonic."""
    return str(text).replace("&", "&&")


def _money(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def tile_texts(glance: Mapping[str, Any] | None) -> dict[str, tuple[str, str]]:
    """`key -> (value, tooltip)` for every tile. Pure, so it is tested alone."""
    data = dict(glance or {})
    trades = int(data.get("trades") or 0)
    out: dict[str, tuple[str, str]] = {}

    pnl = data.get("pnl")
    if not data:
        out["pnl"] = ("…", "Not read yet.")
    elif trades == 0:
        out["pnl"] = ("no trades", "No trade opened or closed on this session.")
    elif pnl is None:
        out["pnl"] = (NOT_MEASURED, "No trade on this session has a whole-trade net yet.")
    else:
        r_value = data.get("r")
        text = _money(float(pnl)) + (f" · {float(r_value):+.2f}R" if r_value is not None else "")
        counted = int(data.get("pnl_counted") or 0)
        tip = f"Sum of whole-trade net over {counted} of {trades} trade(s)."
        tip += (
            " R is the net over each trade's planned risk."
            if r_value is not None
            else " R: not measured - a trade has no planned risk."
        )
        out["pnl"] = (text, tip)

    if trades == 0:
        out["trades"] = ("—", "No trades on this session.")
    else:
        wins, losses = int(data.get("wins") or 0), int(data.get("losses") or 0)
        out["trades"] = (
            f"{wins}W / {losses}L",
            f"{trades} trade(s): {wins} with a gain, {losses} with a loss; "
            f"{trades - wins - losses} flat or not measured.",
        )

    planned = data.get("planned")
    if trades == 0:
        out["planned"] = ("—", "No trades on this session.")
    elif not isinstance(planned, Mapping):
        out["planned"] = (NOT_MEASURED, "The report card did not say where the trades came from.")
    else:
        tip = (
            f"Planned: a claim or like before the first fill. {planned.get('unmeasured', 0)} "
            "not measured (a date-only fill has no time to plan against)."
        )
        if planned.get("lanes_unread"):
            tip += (
                " Unplanned here means no claim or like was seen before the fill; "
                "some plan sources are not read yet."
            )
        out["planned"] = (f"{planned.get('planned', 0)} / {planned.get('unplanned', 0)}", tip)

    calls = data.get("calls")
    if not isinstance(calls, Mapping):
        out["calls"] = ("no calls", "No graded market calls for this session.")
    else:
        out["calls"] = (
            f"{calls.get('right', 0)} / {calls.get('wrong', 0)} / {calls.get('flat', 0)}",
            f"Your market calls, graded. {calls.get('pending', 0)} still waiting on their horizon.",
        )

    day_type = str(data.get("day_type") or "").strip()
    out["day_type"] = (
        (day_type.replace("_", " "), "The desk's D1 market environment label for this session.")
        if day_type
        else ("not labelled", "The desk stored no market environment label for this session.")
    )

    out["market_axes"] = _market_axes_tile(data.get("market_axes"))

    win = data.get("biggest_win")
    out["biggest_win"] = (
        (f"{win.get('symbol', '')} {_money(float(win.get('net_pnl') or 0.0))}",
         "Your best trade on this session, by whole-trade net.")
        if isinstance(win, Mapping)
        else ("none", "No trade with a gain on this session.")
    )
    miss = data.get("biggest_miss")
    out["biggest_miss"] = (
        (f"{miss.get('symbol', '')} {float(miss.get('ran_after_pct') or 0.0):+.2f}%",
         "The biggest real miss: a name you passed on or liked and did not trade, "
         "that ran the most afterwards.")
        if isinstance(miss, Mapping)
        else ("none", "No real miss was measured on this session.")
    )
    return out


def _market_axes_tile(axes: Any) -> tuple[str, str]:
    """The morning's SPY / breadth / internals read, graded against SPY (P2-8)."""
    if not isinstance(axes, Mapping) or not axes.get("grades"):
        return (NOT_MEASURED, "No market read was recorded for the morning of this session.")
    summary = axes.get("summary") if isinstance(axes.get("summary"), Mapping) else {}
    right, wrong = int(summary.get("right") or 0), int(summary.get("wrong") or 0)
    flat, pending = int(summary.get("flat") or 0), int(summary.get("pending") or 0)
    if not (right or wrong or flat) and pending:
        value = "pending"
    elif not (right or wrong or flat):
        value = "no call"
    else:
        value = f"{right} / {wrong}"
    lines = [str(axes.get("line") or "").strip()]
    for grade in axes.get("grades") or ():
        move = grade.get("move_atr")
        moved = f" (SPY {float(move):+.2f} ATR)" if isinstance(move, (int, float)) else ""
        lines.append(f"{grade.get('axis')}: {str(grade.get('verdict') or '').replace('_', ' ')}{moved}")
    lines.append("Graded close to close against SPY; a move inside 0.25 ATR is flat.")
    return (value, "\n".join(line for line in lines if line))


class Sparkline(QWidget):
    """Five sessions of day P&L as small bars. A missing day is a gap."""

    clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._values: list[float | None] = []
        self.setMinimumSize(QSize(90, 36))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_series(self, series) -> None:
        pairs = [pair for pair in (series or ()) if isinstance(pair, (tuple, list)) and len(pair) == 2]
        self._values = [None if value is None else float(value) for _day, value in pairs]
        tip = "Day P&L, last sessions:\n" + "\n".join(
            f"{day}: {'no trades' if value is None else _money(float(value))}" for day, value in pairs
        )
        self.setToolTip(tip if pairs else "No earlier sessions read.")
        self.update()

    def values(self) -> list[float | None]:
        return list(self._values)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.clicked.emit()
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        known = [value for value in self._values if value is not None]
        if not self._values:
            return
        painter = QPainter(self)
        try:
            width, height = self.width(), self.height()
            middle = height / 2.0
            scale = max((abs(value) for value in known), default=0.0) or 1.0
            step = width / max(1, len(self._values))
            painter.setPen(QPen(self.palette().mid().color(), 1))
            painter.drawLine(QPointF(0, middle), QPointF(width, middle))
            for index, value in enumerate(self._values):
                if value is None:
                    continue
                bar = (abs(value) / scale) * (middle - 2)
                colour = QColor("#3fb950") if value >= 0 else QColor("#f85149")
                top = middle - bar if value >= 0 else middle
                painter.fillRect(
                    int(index * step + 2), int(top), max(2, int(step - 4)), max(1, int(bar)), colour
                )
        finally:
            painter.end()


class DayGlanceStrip(QFrame):
    """The tiles and the sparkline. Emits the key of a clicked tile."""

    tileClicked = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("DayGlanceStrip")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.tiles: dict[str, QPushButton] = {}
        self._labels = dict(TILES)
        for key, label in TILES:
            tile = QPushButton(f"…\n{_amp(label)}")
            tile.setObjectName("GlanceTile")
            tile.setFlat(True)
            tile.setProperty("glance", key)
            tile.setCursor(Qt.CursorShape.PointingHandCursor)
            tile.clicked.connect(lambda _checked=False, name=key: self.tileClicked.emit(name))
            row.addWidget(tile)
            self.tiles[key] = tile
        self.sparkline = Sparkline()
        self.sparkline.clicked.connect(lambda: self.tileClicked.emit("sparkline"))
        row.addWidget(self.sparkline)
        row.addStretch(1)

    def set_glance(self, glance: Mapping[str, Any] | None) -> None:
        for key, (value, tip) in tile_texts(glance).items():
            tile = self.tiles.get(key)
            if tile is None:
                continue
            tile.setText(f"{_amp(value)}\n{_amp(self._labels[key])}")
            tile.setToolTip(tip)
        self.sparkline.set_series((glance or {}).get("pnl_by_session") or ())

    def tile_value(self, key: str) -> str:
        """The number line of one tile (what a test reads)."""
        tile = self.tiles.get(key)
        return tile.text().split("\n", 1)[0].replace("&&", "&") if tile is not None else ""


__all__ = ["DayGlanceStrip", "NOT_MEASURED", "Sparkline", "TILES", "tile_texts"]
