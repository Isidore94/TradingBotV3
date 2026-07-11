from __future__ import annotations

"""SPY M5 candle chart with a draggable from->to selection region.

Built from the bot's cached 5m bars (no IB fetch). Candles are drawn at
integer bar indexes (no overnight gaps); the bottom axis translates indexes
back to HH:MM labels. The gold LinearRegionItem is the RS/RW measurement
window - drag its edges to choose "from where to where".
"""

from datetime import datetime

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPicture, QPen

from ui import theme


class _CandleItem(pg.GraphicsObject):
    def __init__(self, bars: list[dict]) -> None:
        super().__init__()
        self._bars = bars
        self._picture = QPicture()
        self._render()

    def _render(self) -> None:
        up = QColor(theme.color("long"))
        down = QColor(theme.color("short"))
        painter = QPainter(self._picture)
        width = 0.35
        for index, bar in enumerate(self._bars):
            color = up if bar["close"] >= bar["open"] else down
            painter.setPen(QPen(color))
            painter.setBrush(color)
            painter.drawLine(
                pg.QtCore.QPointF(index, bar["low"]), pg.QtCore.QPointF(index, bar["high"])
            )
            body_top = max(bar["open"], bar["close"])
            body_bottom = min(bar["open"], bar["close"])
            painter.drawRect(
                QRectF(index - width, body_bottom, width * 2, max(body_top - body_bottom, 1e-9))
            )
        painter.end()

    def paint(self, painter, *_args) -> None:
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:
        if not self._bars:
            return QRectF()
        lows = [bar["low"] for bar in self._bars]
        highs = [bar["high"] for bar in self._bars]
        return QRectF(-1, min(lows), len(self._bars) + 1, max(highs) - min(lows))


class SpyM5Chart(pg.PlotWidget):
    """Candles + selection region; exposes the selection as datetimes."""

    regionChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent, background=theme.color("bg_panel"))
        self._bars: list[dict] = []
        self._region: pg.LinearRegionItem | None = None
        self.showGrid(x=False, y=True, alpha=0.15)
        self.setMouseEnabled(x=True, y=False)
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().hideButtons()

    def set_bars(self, bars: list[dict], *, preserve_selection: bool = True) -> None:
        """Replace the chart contents; keeps (or re-seeds) the selection.

        preserve_selection=False re-anchors the region to the trailing hour -
        the auto-refresh path uses it while the trader has not customized the
        window, so the default view always tracks "the last hour"."""
        previous = self.selected_range() if preserve_selection else None
        self._bars = [dict(bar) for bar in bars or []]
        plot = self.getPlotItem()
        plot.clear()
        self._region = None
        if not self._bars:
            return
        plot.addItem(_CandleItem(self._bars))

        axis = plot.getAxis("bottom")
        step = max(1, len(self._bars) // 10)
        ticks = []
        for index in range(0, len(self._bars), step):
            stamp = self._bars[index]["dt"]
            label = stamp.strftime("%H:%M")
            if index == 0 or stamp.date() != self._bars[index - 1]["dt"].date():
                label = stamp.strftime("%m/%d %H:%M")
            ticks.append((index, label))
        axis.setTicks([ticks])

        color = QColor(theme.color("favorite"))
        color.setAlpha(45)
        hoverColor = QColor(theme.color("favorite"))
        hoverColor.setAlpha(70)
        self._region = pg.LinearRegionItem(
            values=self._default_region(previous),
            brush=color,
            hoverBrush=hoverColor,
            pen=pg.mkPen(QColor(theme.color("favorite")), width=1),
        )
        self._region.setZValue(10)
        self._region.sigRegionChangeFinished.connect(lambda *_: self.regionChanged.emit())
        plot.addItem(self._region)
        plot.setXRange(-1, len(self._bars), padding=0.01)
        lows = [bar["low"] for bar in self._bars]
        highs = [bar["high"] for bar in self._bars]
        plot.setYRange(min(lows), max(highs), padding=0.05)

    def _default_region(self, previous: tuple[datetime, datetime] | None) -> tuple[float, float]:
        if previous is not None:
            start = self._index_for_dt(previous[0])
            end = self._index_for_dt(previous[1])
            if start is not None and end is not None and end > start:
                return (start, end)
        # Default: the trailing hour (12 five-minute bars).
        end = len(self._bars) - 1
        return (max(0, end - 12), end)

    def _index_for_dt(self, stamp: datetime) -> float | None:
        for index, bar in enumerate(self._bars):
            if bar["dt"] >= stamp:
                return float(index)
        return None

    def selected_range(self) -> tuple[datetime, datetime] | None:
        """The selection as (start_dt, end_dt) clamped to the loaded bars."""
        if self._region is None or not self._bars:
            return None
        lo, hi = self._region.getRegion()
        last = len(self._bars) - 1
        start_index = min(max(int(round(lo)), 0), last)
        end_index = min(max(int(round(hi)), 0), last)
        if end_index < start_index:
            start_index, end_index = end_index, start_index
        return (self._bars[start_index]["dt"], self._bars[end_index]["dt"])

    def bar_count(self) -> int:
        return len(self._bars)
