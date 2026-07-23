from __future__ import annotations

"""Reusable candle chart with optional overlay lines (SMAs/EMAs/VWAP bands).

Shares the SPY M5 chart's rendering approach: candles drawn at integer bar
indexes (no overnight/weekend gaps) with the bottom axis translating indexes
back to time labels. Overlays follow the chart_snapshot contract: values
align 1:1 with the bars, ``color`` is a ui.theme role, None values break the
line (plotted as NaN with connect="finite").
"""

import math

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPicture, QPen

from ui import theme


class CandleItem(pg.GraphicsObject):
    """Candles at integer x-indexes, colored by the theme's long/short roles."""

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


class CandleChart(pg.PlotWidget):
    """Candles + overlay lines; y-range follows the candles (overlays clip)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent, background=theme.color("bg_panel"))
        self._bars: list[dict] = []
        self.showGrid(x=False, y=True, alpha=0.15)
        self.setMouseEnabled(x=True, y=False)
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().hideButtons()

    def set_data(self, bars: list[dict], overlays: list[dict] = (), *, timeframe: str = "m5") -> None:
        self._bars = [dict(bar) for bar in bars or []]
        plot = self.getPlotItem()
        plot.clear()
        if not self._bars:
            return
        plot.addItem(CandleItem(self._bars))
        for overlay in overlays or []:
            values = [
                float(value) if value is not None else math.nan
                for value in (overlay.get("values") or [])
            ]
            if len(values) != len(self._bars) or all(math.isnan(value) for value in values):
                continue
            pen = pg.mkPen(
                QColor(theme.color(str(overlay.get("color") or "neutral"))),
                width=float(overlay.get("width") or 1.0),
                style=Qt.PenStyle.DashLine if overlay.get("dash") else Qt.PenStyle.SolidLine,
            )
            plot.plot(list(range(len(values))), values, pen=pen, connect="finite")
        self._set_ticks(timeframe)
        plot.setXRange(-1, len(self._bars), padding=0.01)
        lows = [bar["low"] for bar in self._bars]
        highs = [bar["high"] for bar in self._bars]
        plot.setYRange(min(lows), max(highs), padding=0.05)

    def _set_ticks(self, timeframe: str) -> None:
        axis = self.getPlotItem().getAxis("bottom")
        daily = str(timeframe).lower().startswith("d")
        step = max(1, len(self._bars) // 8)
        ticks = []
        for index in range(0, len(self._bars), step):
            stamp = self._bars[index]["dt"]
            if daily:
                label = stamp.strftime("%m/%d")
            else:
                label = stamp.strftime("%H:%M")
                if index == 0 or stamp.date() != self._bars[index - 1]["dt"].date():
                    label = stamp.strftime("%m/%d %H:%M")
            ticks.append((index, label))
        axis.setTicks([ticks])

    def bar_count(self) -> int:
        return len(self._bars)
