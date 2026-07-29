from __future__ import annotations

"""Reusable candle chart with optional overlay lines (SMAs/EMAs/VWAP bands).

Shares the SPY M5 chart's rendering approach: candles drawn at integer bar
indexes (no overnight/weekend gaps) with the bottom axis translating indexes
back to time labels. Overlays follow the chart_snapshot contract: values
align 1:1 with the bars, ``color`` is a ui.theme role, None values break the
line (plotted as NaN with connect="finite").

Prices plot on a log10 scale by default, so equal percentage moves get equal
vertical distance: a run from 40 to 60 reads the same as 60 to 90, and a
trendline drawn across months keeps its meaning. The chart's view coordinates
are therefore log10(price) - ``_to_log_price`` maps into that space and the
left axis / click handling map back out, so every price crossing the widget
boundary stays a real price.
"""

import math

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPicture, QPen

from ui import theme


_CANDLE_HALF_WIDTH = 0.27
# A log axis is undefined at or below zero. Prices are positive in practice
# (CandleChart falls back to linear if they are not), so this floor only keeps
# a bad cache row from raising mid-render.
_LOG_PRICE_FLOOR = 1e-6
# Round steps traders actually read off a price axis, in units of 10^k.
_TICK_STEP_MULTIPLES = (1.0, 2.0, 2.5, 5.0, 10.0)


def _to_log_price(value: float) -> float:
    """Price -> log10 chart space, clamped so non-positive input cannot raise."""
    return math.log10(max(float(value), _LOG_PRICE_FLOOR))


def _nice_price_ticks(low: float, high: float, *, target: int = 6) -> tuple[list[float], float]:
    """Round price levels spanning [low, high], with the step used to pick them.

    Ticks are chosen in price space, not axis space: evenly spaced log10
    coordinates would label the axis 39.8 / 44.7 / 50.1, and a price axis has
    to show round numbers to be worth reading.
    """
    low = float(low)
    high = float(high)
    span = high - low
    if not math.isfinite(span) or span <= 0:
        return ([low], max(abs(low), 1.0))
    raw_step = span / max(1, int(target))
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    step = magnitude * _TICK_STEP_MULTIPLES[-1]
    for multiple in _TICK_STEP_MULTIPLES:
        candidate = magnitude * multiple
        if candidate >= raw_step:
            step = candidate
            break
    ticks = []
    value = math.ceil(low / step) * step
    while value <= high + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    return (ticks, step)


def _format_price(value: float, step: float) -> str:
    """Label a tick with just enough decimals to keep adjacent ticks distinct."""
    if step >= 1:
        decimals = 0
    elif step >= 0.1:
        decimals = 1
    else:
        decimals = 2
    return f"{value:,.{decimals}f}"


class PriceAxis(pg.AxisItem):
    """Left axis that labels log10 chart coordinates with real prices.

    pyqtgraph's own log mode is not used: it only transforms items that
    implement ``setLogMode`` (so the candles would stay linear) and labels
    decades as powers of ten, which is unreadable across a 40-to-90 range.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._log_prices = False
        self._step = 1.0

    def set_log_prices(self, enabled: bool) -> None:
        self._log_prices = bool(enabled)
        self.picture = None
        self.update()

    def tickValues(self, minVal, maxVal, size):  # noqa: N802 (pyqtgraph override)
        if not self._log_prices:
            return super().tickValues(minVal, maxVal, size)
        prices, step = _nice_price_ticks(
            10.0 ** min(minVal, maxVal), 10.0 ** max(minVal, maxVal)
        )
        self._step = step
        return [(step, [_to_log_price(price) for price in prices])]

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph override)
        if not self._log_prices:
            return super().tickStrings(values, scale, spacing)
        step = spacing or self._step
        return [_format_price(10.0 ** value, step) for value in values]


def _time_ticks(bars: list[dict], timeframe: str, *, max_ticks: int = 7) -> list[tuple[int, str]]:
    """Sparse, session-aware labels for the integer-indexed time axis."""
    if not bars:
        return []
    count = len(bars)
    target = max(2, min(int(max_ticks), count))
    if count <= target:
        positions = list(range(count))
    else:
        positions = sorted(
            {round(slot * (count - 1) / (target - 1)) for slot in range(target)}
        )

    daily = str(timeframe).lower().startswith("d")
    ticks = []
    previous_tick_date = None
    for index in positions:
        stamp = bars[index]["dt"]
        if daily:
            label = stamp.strftime("%m/%d")
        elif previous_tick_date is None or stamp.date() != previous_tick_date:
            label = stamp.strftime("%m/%d %H:%M")
        else:
            label = stamp.strftime("%H:%M")
        ticks.append((index, label))
        previous_tick_date = stamp.date()
    return ticks


class CandleItem(pg.GraphicsObject):
    """Candles at integer x-indexes, colored by the theme's long/short roles.

    ``log_y`` draws the bars in log10 price space. It defaults off so the
    plain linear callers (the SPY M5 selection chart) are unaffected.
    """

    def __init__(self, bars: list[dict], *, log_y: bool = False) -> None:
        super().__init__()
        self._bars = bars
        self._log_y = bool(log_y)
        self._picture = QPicture()
        self._render()

    def _y(self, value: float) -> float:
        return _to_log_price(value) if self._log_y else float(value)

    def _render(self) -> None:
        up = QColor(theme.color("long"))
        down = QColor(theme.color("short"))
        painter = QPainter(self._picture)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for index, bar in enumerate(self._bars):
            color = up if bar["close"] >= bar["open"] else down
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidthF(1.0)
            painter.setPen(pen)
            # A forming (preview) candle - e.g. today's D1 synthesized from
            # cached M5 bars - draws hollow: colored outline, no body fill,
            # so a completed bar and a still-moving one never read the same.
            if bar.get("preview"):
                painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                painter.setBrush(color)
            painter.drawLine(
                pg.QtCore.QPointF(index, self._y(bar["low"])),
                pg.QtCore.QPointF(index, self._y(bar["high"])),
            )
            # Take the log after the max/min: log10 is monotonic, so the body
            # edges are the same bars either way, but this keeps the compare
            # in price space where the values are exact.
            body_top = self._y(max(bar["open"], bar["close"]))
            body_bottom = self._y(min(bar["open"], bar["close"]))
            if body_top == body_bottom:
                painter.drawLine(
                    pg.QtCore.QPointF(index - _CANDLE_HALF_WIDTH, body_top),
                    pg.QtCore.QPointF(index + _CANDLE_HALF_WIDTH, body_top),
                )
            else:
                painter.drawRect(
                    QRectF(
                        index - _CANDLE_HALF_WIDTH,
                        body_bottom,
                        _CANDLE_HALF_WIDTH * 2,
                        body_top - body_bottom,
                    )
                )
        painter.end()

    def paint(self, painter, *_args) -> None:
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:
        if not self._bars:
            return QRectF()
        low = self._y(min(bar["low"] for bar in self._bars))
        high = self._y(max(bar["high"] for bar in self._bars))
        return QRectF(-1, low, len(self._bars) + 1, high - low)


class CandleChart(pg.PlotWidget):
    """Candles + overlay lines; y-range follows the candles (overlays clip).

    A left click emits ``barClicked(index)`` for the candle nearest the click
    (hosts use it e.g. to arm a D1 level alert off that candle's high/low);
    panning still works because the press is not consumed.
    """

    barClicked = Signal(int)
    # The y coordinate of the same click, in price. mousePressEvent already
    # maps the click into view space to find the bar, so the price is free -
    # it turns "the level I can see" into "the level I armed" without the
    # trader reading it off the axis and retyping it.
    priceClicked = Signal(int, float)

    def __init__(self, parent=None, *, log_y: bool = True) -> None:
        self._price_axis = PriceAxis(orientation="left")
        super().__init__(
            parent,
            background=theme.color("bg_panel"),
            axisItems={"left": self._price_axis},
        )
        self._bars: list[dict] = []
        self._overlays: list[dict] = []
        self._timeframe = "m5"
        # Requested scaling vs. what the current bars actually allow.
        self._log_y = bool(log_y)
        self._log_active = False
        self.showGrid(x=False, y=True, alpha=0.15)
        self.setMouseEnabled(x=True, y=False)
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().hideButtons()
        axis_font = QFont()
        axis_font.setPointSizeF(9.5)
        for name in ("bottom", "left"):
            axis = self.getPlotItem().getAxis(name)
            axis.setTickFont(axis_font)
            axis.setTextPen(pg.mkPen(theme.color("text_secondary")))
            axis.setPen(pg.mkPen(theme.color("border")))
            axis.setStyle(hideOverlappingLabels=True, tickTextOffset=7)

    def set_data(self, bars: list[dict], overlays: list[dict] = (), *, timeframe: str = "m5") -> None:
        self._bars = [dict(bar) for bar in bars or []]
        # Retained so a log/linear toggle can re-render without the caller
        # having to re-fetch the snapshot.
        self._overlays = [dict(overlay) for overlay in overlays or []]
        self._timeframe = timeframe
        plot = self.getPlotItem()
        plot.clear()
        if not self._bars:
            self._apply_log_active(False)
            return
        lows = [bar["low"] for bar in self._bars]
        highs = [bar["high"] for bar in self._bars]
        # Log scaling needs strictly positive prices. A non-positive bar means
        # a bad cache row, and a silently clamped candle would misdraw the
        # whole chart - fall back to linear and stay honest instead.
        self._apply_log_active(self._log_y and min(lows) > 0)
        plot.addItem(CandleItem(self._bars, log_y=self._log_active))
        for overlay in self._overlays:
            values = [
                float(value) if value is not None else math.nan
                for value in (overlay.get("values") or [])
            ]
            if len(values) != len(self._bars) or all(math.isnan(value) for value in values):
                continue
            if self._log_active:
                # An overlay may dip non-positive where the bars do not (a
                # sigma band on a low-priced name); break the line there
                # rather than clamping it onto the floor.
                values = [
                    _to_log_price(value) if value > 0 else math.nan for value in values
                ]
            pen = pg.mkPen(
                QColor(theme.color(str(overlay.get("color") or "neutral"))),
                width=float(overlay.get("width") or 1.0),
                style=Qt.PenStyle.DashLine if overlay.get("dash") else Qt.PenStyle.SolidLine,
            )
            plot.plot(
                list(range(len(values))),
                values,
                pen=pen,
                connect="finite",
                antialias=True,
            )
        self._set_ticks(timeframe)
        plot.setXRange(-1, len(self._bars), padding=0.01)
        plot.setYRange(self._y(min(lows)), self._y(max(highs)), padding=0.05)

    def _apply_log_active(self, active: bool) -> None:
        self._log_active = bool(active)
        self._price_axis.set_log_prices(self._log_active)

    def _y(self, value: float) -> float:
        """Price -> chart space (the inverse of :meth:`price_at`)."""
        return _to_log_price(value) if self._log_active else float(value)

    def price_at(self, y: float) -> float:
        """Chart space -> price, for anything reading a click or a cursor."""
        return 10.0 ** float(y) if self._log_active else float(y)

    def is_log_scaled(self) -> bool:
        """Whether the drawn bars are currently on a log price axis."""
        return self._log_active

    def set_log_y(self, enabled: bool) -> None:
        """Switch between log and linear price scaling, redrawing the bars."""
        enabled = bool(enabled)
        if enabled == self._log_y:
            return
        self._log_y = enabled
        self.set_data(self._bars, self._overlays, timeframe=self._timeframe)

    def _set_ticks(self, timeframe: str) -> None:
        axis = self.getPlotItem().getAxis("bottom")
        axis.setTicks([_time_ticks(self._bars, timeframe)])

    def bar_count(self) -> int:
        return len(self._bars)

    def bar_at(self, index: int) -> dict | None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        if 0 <= index < len(self._bars):
            return dict(self._bars[index])
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton and self._bars:
            price = None
            try:
                scene_pos = self.mapToScene(event.position().toPoint())
                view_pos = self.getPlotItem().vb.mapSceneToView(scene_pos)
                index = int(round(view_pos.x()))
                # The view is in log space when log scaling is on, and this
                # price arms a real level - map it back before it escapes.
                price = self.price_at(view_pos.y())
            except Exception:
                index = -1
            if 0 <= index < len(self._bars):
                self.barClicked.emit(index)
                if price is not None and price > 0:
                    self.priceClicked.emit(index, price)
        super().mousePressEvent(event)
