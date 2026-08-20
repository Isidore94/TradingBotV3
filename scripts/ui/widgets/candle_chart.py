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

PAINT LINES (A4) are a second, separate layer: the D1 S/R the scan stores,
prev-day extremes, and the projected D1 trendline. They are NOT overlays -
an overlay is a per-bar series, and forcing a horizontal level through that
contract would mean inventing an array of one repeated number per bar and
then having nothing to click. ``set_levels`` takes the snapshot's ``levels``
payload (see :mod:`chart_levels`) and draws each entry with the primitive it
actually is: an infinite horizontal line for a level, a curve for a sloped
one. Every level carries a stable id, so a click can name the line it hit
and ``levelSelected`` can hand that id to a capture rail.

Levels never influence the view range. They are added with
``ignoreBounds=True`` and the y-range is set from the candles alone, so a
level far above the visible prices simply is not seen - it never stretches
the chart to reach itself.
"""

import math

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPicture, QPen

from ui import theme


_CANDLE_HALF_WIDTH = 0.27
# How long after the last pan/zoom event antialiasing comes back. Long enough
# that a continuous drag never re-enables it mid-gesture, short enough that
# the lines look right again as soon as the trader lets go.
_AA_RESTORE_MS = 150
# A log axis is undefined at or below zero. Prices are positive in practice
# (CandleChart falls back to linear if they are not), so this floor only keeps
# a bad cache row from raising mid-render.
_LOG_PRICE_FLOOR = 1e-6
# Round steps traders actually read off a price axis, in units of 10^k.
_TICK_STEP_MULTIPLES = (1.0, 2.0, 2.5, 5.0, 10.0)
# How close a click has to land, in screen pixels, to count as hitting a
# painted level. Generous enough to hit a 1px line with a trackpad, tight
# enough that two levels a few cents apart stay separately selectable.
LEVEL_HIT_TOLERANCE_PX = 6.0
# What selection does to a level's pen. Deliberately weight and opacity only:
# recoloring would break the one thing the level palette is for, which is
# telling green-bucket S/R from red at a glance.
_LEVEL_SELECTED_EXTRA_WIDTH = 1.6


def hover_readout(
    bars: list[dict], x_position: float, timeframe: str
) -> tuple[int, str] | None:
    """Return the nearest bar index and its display-only OHLCV readout.

    This is deliberately plain Python. Mouse movement must never trigger a
    frame conversion, file read, provider call, or any other work beyond one
    indexed lookup and formatting the already-drawn bar.
    """
    if not bars:
        return None
    try:
        position = float(x_position)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(position) or position < -0.5 or position > len(bars) - 0.5:
        return None
    index = int(round(position))
    if not 0 <= index < len(bars):
        return None
    bar = bars[index]
    stamp = bar.get("dt")
    if not hasattr(stamp, "strftime"):
        return None
    stamp_text = stamp.strftime(
        "%Y-%m-%d" if str(timeframe).lower().startswith("d") else "%Y-%m-%d %H:%M"
    )

    def price(name: str) -> str:
        value = float(bar[name])
        return f"{value:,.4f}" if 0 < abs(value) < 1 else f"{value:,.2f}"

    try:
        volume = float(bar.get("volume") or 0.0)
        text = (
            f"{stamp_text}   O {price('open')}   H {price('high')}   "
            f"L {price('low')}   C {price('close')}   V {volume:,.0f}"
        )
    except (KeyError, TypeError, ValueError):
        return None
    return index, text


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

    def __init__(self, bars: list[dict] = (), *, log_y: bool = False) -> None:
        super().__init__()
        self._bars = list(bars or [])
        self._log_y = bool(log_y)
        self._picture = QPicture()
        self._bounds = QRectF()
        self._render()

    def set_bars(self, bars: list[dict], *, log_y: bool | None = None) -> None:
        """Re-record this item's candles in place (C5: never rebuild items).

        One CandleItem lives for the chart's lifetime; a symbol switch
        re-records its picture instead of destroying a scene item and adding
        a fresh one. Rebuilding was measured at 10.6ms of a 13ms set_data.
        """
        self._bars = list(bars or [])
        if log_y is not None:
            self._log_y = bool(log_y)
        # The bar count and price span both move, so the cached geometry Qt
        # holds for this item is stale until _render recomputes it.
        self.prepareGeometryChange()
        self._render()
        self.informViewBoundsChanged()
        self.update()

    def _y(self, value: float) -> float:
        return _to_log_price(value) if self._log_y else float(value)

    def _render(self) -> None:
        # A fresh QPicture per render: re-opening a painter on a recorded
        # picture is not a documented reset, and allocation is negligible
        # next to the drawing loop (~7us/bar).
        self._picture = QPicture()
        self._bounds = self._compute_bounds()
        if not self._bars:
            return
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

    def _compute_bounds(self) -> QRectF:
        if not self._bars:
            return QRectF()
        low = self._y(min(bar["low"] for bar in self._bars))
        high = self._y(max(bar["high"] for bar in self._bars))
        return QRectF(-1, low, len(self._bars) + 1, high - low)

    def boundingRect(self) -> QRectF:
        # Cached, not recomputed: Qt calls this on every paint and on every
        # view-range change, and an O(bars) min/max per call is exactly the
        # kind of work that shows up as a pan/zoom stall.
        return self._bounds


class VolumeItem(pg.GraphicsObject):
    """Translucent volume columns hugging the bottom of the view.

    Deliberately NOT a second stacked plot. The desk's alert column is short
    of vertical space (that is the whole reason the capture rail moved to a
    tab), and a volume sub-panel would take 20-25% of the candles to show a
    series the trader reads as context rather than studies. Drawn as an
    underlay it costs zero height.

    The columns are laid out against the CURRENT view, not the data: the
    picture is recorded once in a normalized 0..1 band and ``paint`` maps that
    band onto the bottom ``height_fraction`` of whatever the y-range happens
    to be. So a pan or a log/linear flip is a transform, never a re-render,
    and the item never has an opinion about the price range - it reads it.

    Scaling is by the PEAK of the drawn bars. That makes the columns a
    relative read ("today is heavy for this name"), which is what a volume
    underlay is for; it is not an axis and it deliberately has no ticks.
    """

    def __init__(self, bars: list[dict] = (), *, height_fraction: float = 0.18) -> None:
        super().__init__()
        self._bars: list[dict] = []
        self._fraction = float(height_fraction)
        self._picture = QPicture()
        self._peak = 0.0
        self.setZValue(-20)  # under the candles, the overlays and the levels
        self.set_bars(bars)

    def set_bars(self, bars: list[dict]) -> None:
        """Re-record in place (C5: never rebuild scene items)."""
        self._bars = list(bars or [])
        self.prepareGeometryChange()
        self._render()
        self.update()

    def has_volume(self) -> bool:
        """False when nothing measurable was supplied, so nothing is drawn.

        Missing volume is uncertainty, never zero: a store that carries no
        volume column, or a symbol whose rows are all 0, draws NO columns
        rather than a flat row of nothing, which would read as "no volume
        traded" when the truth is "not measured here".
        """
        return self._peak > 0.0

    def _render(self) -> None:
        self._picture = QPicture()
        volumes = []
        for bar in self._bars:
            try:
                value = float(bar.get("volume") or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            # NaN fails this compare, which is the intent.
            volumes.append(value if value > 0 else 0.0)
        self._peak = max(volumes) if volumes else 0.0
        if not self._peak:
            return
        up = QColor(theme.color("long"))
        down = QColor(theme.color("short"))
        # Translucent: the candles, the overlays and the paint lines all draw
        # over this, and none of them may become harder to read for it.
        up.setAlphaF(0.38)
        down.setAlphaF(0.38)
        painter = QPainter(self._picture)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(Qt.PenStyle.NoPen)
        for index, bar in enumerate(self._bars):
            value = volumes[index]
            if value <= 0.0:
                continue
            painter.setBrush(up if bar.get("close", 0) >= bar.get("open", 0) else down)
            painter.drawRect(
                QRectF(
                    index - _CANDLE_HALF_WIDTH,
                    0.0,
                    _CANDLE_HALF_WIDTH * 2,
                    value / self._peak,
                )
            )
        painter.end()

    def _band(self) -> tuple[float, float] | None:
        """(bottom, height) of the volume band in current view coordinates."""
        if not self._peak:
            return None
        view = self.getViewBox()
        if view is None:
            return None
        try:
            (_x_min, _x_max), (y_min, y_max) = view.viewRange()
        except Exception:
            return None
        height = (float(y_max) - float(y_min)) * self._fraction
        return (float(y_min), height) if height > 0 else None

    def paint(self, painter, *_args) -> None:
        band = self._band()
        if band is None:
            return
        bottom, height = band
        painter.translate(0.0, bottom)
        painter.scale(1.0, height)
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:
        band = self._band()
        if band is None:
            return QRectF()
        bottom, height = band
        return QRectF(-1.0, bottom, len(self._bars) + 1.0, height)

    def viewRangeChanged(self) -> None:  # noqa: N802 (pyqtgraph override)
        """The band is defined by the view, so its geometry moved with it."""
        self.prepareGeometryChange()
        self.update()


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
    # (id, family, price) for a painted level the click landed on. The id is
    # the stable one chart_levels derived, so a capture rail can record WHICH
    # line the trader was looking at, not just a number that happened to be
    # nearby. Emitted in addition to barClicked/priceClicked, never instead:
    # a click on a level is still a click on the chart.
    levelSelected = Signal(str, str, float)

    def __init__(self, parent=None, *, log_y: bool = True) -> None:
        self._price_axis = PriceAxis(orientation="left")
        super().__init__(
            parent,
            background=theme.color("bg_panel"),
            axisItems={"left": self._price_axis},
        )
        self._bars: list[dict] = []
        self._overlays: list[dict] = []
        self._levels: list[dict] = []
        self._selected_level_id = ""
        self._timeframe = "m5"
        # Requested scaling vs. what the current bars actually allow.
        self._log_y = bool(log_y)
        self._log_active = False
        self._antialias = True
        self.showGrid(x=False, y=True, alpha=0.15)
        self.setMouseEnabled(x=True, y=False)
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().hideButtons()

        # --- C5 render discipline -------------------------------------
        # One candle item and a reused pool of curve items live for the
        # widget's lifetime. set_data pushes new numbers into them; it never
        # calls plot.clear() and never constructs scene items, because
        # rebuilding 14 curves per symbol switch measured 10.6ms of a 13ms
        # set_data - the single largest cost on the chart path.
        plot = self.getPlotItem()
        self._candles = CandleItem()
        plot.addItem(self._candles)
        # Volume underlay. ignoreBounds because it reads the view range to
        # place itself - letting it vote on that range would be circular, and
        # the price scale must come from the candles and nothing else (the
        # same rule the paint lines follow).
        self._volume = VolumeItem()
        self._show_volume = False
        self._volume.setVisible(False)
        plot.addItem(self._volume, ignoreBounds=True)
        # Crosshair/readout items are deliberately NOT built here. They are a
        # hover decoration, and building them at construction gave every
        # CandleChart in the app three extra native scene items plus a
        # scene-level sigMouseMoved connection whose teardown order Qt does not
        # guarantee - enough accumulated state to segfault the suite (SIGSEGV
        # in 9 of 13 full runs; 0 of 8 without them). They are created on the
        # first hover that has something to show and released the moment the
        # chart is hidden or closed, so a chart nobody hovers costs nothing and
        # never has a scene item outliving the widget that owns it. See
        # _ensure_crosshair / _release_crosshair.
        self._crosshair_v: pg.InfiniteLine | None = None
        self._crosshair_h: pg.InfiniteLine | None = None
        self._hover_label: pg.TextItem | None = None
        self._overlay_items: list[pg.PlotDataItem] = []
        # Paint-line pools, kept apart from the overlay pool: they hold
        # different primitives and a symbol switch changes their counts
        # independently. Same reuse discipline - hidden, never destroyed.
        self._level_line_items: list[pg.InfiniteLine] = []
        self._level_curve_items: list[pg.PlotDataItem] = []
        #: What each drawn item is currently showing, so a click can name it.
        #: [(level dict, "line"|"curve", pooled item)]
        self._drawn_levels: list[tuple[dict, str, object]] = []
        # Clip to the visible range and let pyqtgraph decimate when a series
        # is denser than the pixels available. At today's 90-500 bars auto
        # downsampling resolves to 1 (a no-op); both settings earn their keep
        # when a longer history is zoomed into.
        plot.setClipToView(True)
        plot.setDownsampling(auto=True, mode="peak")
        # Antialiasing off while the trader is dragging, restored shortly
        # after they stop. Only manual range changes count as interaction -
        # set_data's own setXRange/setYRange must not trip it.
        self._aa_restore_timer = QTimer(self)
        self._aa_restore_timer.setSingleShot(True)
        self._aa_restore_timer.setInterval(_AA_RESTORE_MS)
        self._aa_restore_timer.timeout.connect(lambda: self._set_overlay_antialias(True))
        plot.vb.sigRangeChangedManually.connect(self._on_manual_range_change)

        axis_font = QFont()
        axis_font.setPointSizeF(9.5)
        for name in ("bottom", "left"):
            axis = self.getPlotItem().getAxis(name)
            axis.setTickFont(axis_font)
            axis.setTextPen(pg.mkPen(theme.color("text_secondary")))
            axis.setPen(pg.mkPen(theme.color("border")))
            axis.setStyle(hideOverlappingLabels=True, tickTextOffset=7)

    def set_volume_visible(self, visible: bool) -> None:
        """Draw the volume underlay (D1 today; see VolumeItem for why it is
        an underlay and not a stacked sub-plot)."""
        self._show_volume = bool(visible)
        self._sync_volume()

    def volume_is_drawn(self) -> bool:
        """True only when volume is both wanted AND actually measurable."""
        return self._show_volume and self._volume.has_volume() and bool(self._bars)

    def _sync_volume(self) -> None:
        self._volume.set_bars(self._bars if self._show_volume else [])
        self._volume.setVisible(self.volume_is_drawn())

    def set_data(self, bars: list[dict], overlays: list[dict] = (), *, timeframe: str = "m5") -> None:
        self._bars = [dict(bar) for bar in bars or []]
        self._set_crosshair_visible(False)
        # Retained so a log/linear toggle can re-render without the caller
        # having to re-fetch the snapshot.
        self._overlays = [dict(overlay) for overlay in overlays or []]
        self._timeframe = timeframe
        plot = self.getPlotItem()
        if not self._bars:
            self._apply_log_active(False)
            self._candles.set_bars([], log_y=False)
            self._sync_overlays(0)
            self._sync_volume()
            self._push_levels()  # nothing to hang a level on; hide them all
            return
        lows = [bar["low"] for bar in self._bars]
        highs = [bar["high"] for bar in self._bars]
        # Log scaling needs strictly positive prices. A non-positive bar means
        # a bad cache row, and a silently clamped candle would misdraw the
        # whole chart - fall back to linear and stay honest instead.
        self._apply_log_active(self._log_y and min(lows) > 0)
        self._candles.set_bars(self._bars, log_y=self._log_active)
        self._sync_overlays(self._push_overlays())
        self._set_ticks(timeframe)
        plot.setXRange(-1, len(self._bars), padding=0.01)
        # The y-range comes from the candles and nothing else. Every level and
        # overlay is drawn inside whatever range this produces; none of them
        # gets a vote in what it is.
        plot.setYRange(self._y(min(lows)), self._y(max(highs)), padding=0.05)
        self._sync_volume()
        # Levels last: a log/linear flip or a bar change moves where they sit.
        self._push_levels()

    def set_overlays(self, overlays: list[dict] = ()) -> None:
        """Replace the overlay series without touching the candles or the view.

        The paint-lines toggle changes only WHICH lines are drawn, and routing
        that through set_data would re-range the plot and throw away the pan
        and zoom the trader had set up.
        """
        self._overlays = [dict(overlay) for overlay in overlays or ()]
        self._sync_overlays(self._push_overlays() if self._bars else 0)

    def set_levels(self, levels: list[dict] = ()) -> None:
        """Draw the snapshot's paint-lines (chart_levels' ``levels`` payload).

        Independent of :meth:`set_data` on purpose: the toggle shows and hides
        level groups many times between symbol switches, and re-pushing a
        handful of lines must not mean re-rendering the candles.
        """
        self._levels = [dict(level) for level in levels or ()]
        known = {str(level.get("id") or "") for level in self._levels}
        if self._selected_level_id not in known:
            # The selected line is not on this chart any more (symbol switch,
            # or the group holding it was switched off).
            self._selected_level_id = ""
        self._push_levels()

    def _push_levels(self) -> None:
        """Feed the retained levels into the pooled line/curve items."""
        self._drawn_levels = []
        lines = curves = 0
        if self._bars:
            x_values = list(range(len(self._bars)))
            for level in self._levels:
                selected = str(level.get("id") or "") == self._selected_level_id
                pen = self._level_pen(level, selected)
                values = level.get("values")
                if values is None:
                    price = level.get("price")
                    try:
                        price = float(price)
                    except (TypeError, ValueError):
                        continue
                    if price <= 0:
                        continue
                    item = self._level_line_item(lines)
                    item.setPen(pen)
                    item.setPos(self._y(price))
                    self._drawn_levels.append((level, "line", item))
                    lines += 1
                    continue
                if len(values) != len(self._bars):
                    continue  # a series that does not align is not drawable
                plotted = [
                    self._y(float(value))
                    if value is not None and float(value) > 0
                    else math.nan
                    for value in values
                ]
                if all(math.isnan(value) for value in plotted):
                    continue
                item = self._level_curve_item(curves)
                item.setData(
                    x_values,
                    plotted,
                    pen=pen,
                    connect="finite",
                    antialias=self._antialias,
                )
                self._drawn_levels.append((level, "curve", item))
                curves += 1
        for item in self._level_line_items[lines:]:
            item.setVisible(False)
        for item in self._level_curve_items[curves:]:
            item.setVisible(False)

    def _level_pen(self, level: dict, selected: bool):
        dash = level.get("dash")
        if dash == "dot":
            style = Qt.PenStyle.DotLine
        elif dash:
            style = Qt.PenStyle.DashLine
        else:
            style = Qt.PenStyle.SolidLine
        width = float(level.get("width") or 1.0)
        if selected:
            width += _LEVEL_SELECTED_EXTRA_WIDTH
            style = Qt.PenStyle.SolidLine
        return pg.mkPen(
            QColor(theme.color(str(level.get("color") or "neutral"))),
            width=width,
            style=style,
        )

    def _level_line_item(self, index: int) -> pg.InfiniteLine:
        while len(self._level_line_items) <= index:
            item = pg.InfiniteLine(angle=0, movable=False)
            # ignoreBounds: a level must never be able to stretch the view to
            # bring itself into it. Off-screen means off-screen.
            self.getPlotItem().addItem(item, ignoreBounds=True)
            self._level_line_items.append(item)
        item = self._level_line_items[index]
        item.setVisible(True)
        return item

    def _level_curve_item(self, index: int) -> pg.PlotDataItem:
        while len(self._level_curve_items) <= index:
            item = pg.PlotDataItem()
            item.setClipToView(True)
            self.getPlotItem().addItem(item, ignoreBounds=True)
            self._level_curve_items.append(item)
        item = self._level_curve_items[index]
        item.setVisible(True)
        return item

    # -- level selection ---------------------------------------------------
    def drawn_levels(self) -> list[dict]:
        """The levels currently on screen, in draw order."""
        return [dict(level) for level, _kind, _item in self._drawn_levels]

    def selected_level_id(self) -> str:
        return self._selected_level_id

    def select_level(self, level_id: str) -> bool:
        """Highlight a level by id. "" clears. Returns whether it is drawn."""
        level_id = str(level_id or "")
        if level_id == self._selected_level_id:
            return bool(level_id)
        self._selected_level_id = level_id
        self._push_levels()
        return any(
            str(level.get("id") or "") == level_id
            for level, _kind, _item in self._drawn_levels
        )

    def _level_y(self, level: dict, index: int) -> float | None:
        """A level's chart-space y at bar ``index``, or None if undefined there."""
        values = level.get("values")
        if values is None:
            try:
                price = float(level.get("price"))
            except (TypeError, ValueError):
                return None
            return self._y(price) if price > 0 else None
        if not 0 <= index < len(values):
            return None
        value = values[index]
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return self._y(value) if value > 0 else None

    def level_at(
        self, index: int, view_y: float, *, tolerance_px: float = LEVEL_HIT_TOLERANCE_PX
    ) -> dict | None:
        """The drawn level nearest ``view_y`` at bar ``index``, within tolerance.

        Tolerance is in SCREEN pixels, converted through the viewbox: on a log
        price axis a fixed price tolerance would be several times looser at
        the bottom of the chart than at the top, and the trader is aiming with
        a cursor, not with a price.
        """
        if not self._drawn_levels:
            return None
        try:
            pixel_height = float(self.getPlotItem().vb.viewPixelSize()[1])
        except Exception:
            pixel_height = 0.0
        if not math.isfinite(pixel_height) or pixel_height <= 0:
            return None
        limit = float(tolerance_px) * pixel_height
        best = None
        best_distance = limit
        for level, _kind, _item in self._drawn_levels:
            y = self._level_y(level, index)
            if y is None:
                continue
            distance = abs(y - float(view_y))
            if distance <= best_distance:
                best_distance = distance
                best = level
        return dict(best) if best is not None else None

    def _push_overlays(self) -> int:
        """Feed the overlay series into pooled curves; return how many drew."""
        x_values = list(range(len(self._bars)))
        used = 0
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
            # "dash" is the overlay contract's line style: falsy = solid,
            # "dot" = dotted (the SMAs), any other truthy = dashed.
            dash = overlay.get("dash")
            if dash == "dot":
                style = Qt.PenStyle.DotLine
            elif dash:
                style = Qt.PenStyle.DashLine
            else:
                style = Qt.PenStyle.SolidLine
            pen = pg.mkPen(
                QColor(theme.color(str(overlay.get("color") or "neutral"))),
                width=float(overlay.get("width") or 1.0),
                style=style,
            )
            self._overlay_item(used).setData(
                x_values,
                values,
                pen=pen,
                connect="finite",
                antialias=self._antialias,
            )
            used += 1
        return used

    def _overlay_item(self, index: int) -> pg.PlotDataItem:
        """The pooled curve at ``index``, created once and reused thereafter."""
        while len(self._overlay_items) <= index:
            item = pg.PlotDataItem()
            item.setClipToView(True)
            self.getPlotItem().addItem(item)
            self._overlay_items.append(item)
        item = self._overlay_items[index]
        item.setVisible(True)
        return item

    def _sync_overlays(self, used: int) -> None:
        """Hide pooled curves the current snapshot did not fill.

        Hidden rather than removed: the next symbol almost always needs the
        same count back, and keeping them costs one idle item each.
        """
        for item in self._overlay_items[used:]:
            item.setVisible(False)

    def _on_manual_range_change(self, *_args) -> None:
        """User-driven pan/zoom: drop antialiasing until they settle."""
        self._set_overlay_antialias(False)
        self._aa_restore_timer.start()

    def _set_overlay_antialias(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._antialias:
            return
        self._antialias = enabled
        for item in (*self._overlay_items, *self._level_curve_items):
            # Poke the underlying curve's option directly: PlotDataItem only
            # reads opts["antialias"] when it rebuilds, and a full setData
            # here would re-process the samples on every drag event.
            curve = getattr(item, "curve", None)
            if curve is None:
                continue
            curve.opts["antialias"] = enabled
            curve.update()

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

    def _ensure_crosshair(self) -> bool:
        """Build the hover decoration on demand. False means "cannot draw it".

        ``_hover_label`` is assigned last and is the built/not-built sentinel,
        so a half-finished build cannot be mistaken for a usable one.
        """
        if self._hover_label is not None:
            return True
        try:
            plot = self.getPlotItem()
            pen = pg.mkPen(theme.color("text_muted"), width=1, style=Qt.PenStyle.DotLine)
            self._crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            self._crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pen)
            label = pg.TextItem(
                color=theme.color("text_primary"),
                fill=pg.mkBrush(theme.color("bg_elevated")),
                border=pg.mkPen(theme.color("border")),
                anchor=(0, 1),
            )
            label.setZValue(100)
            plot.addItem(self._crosshair_v, ignoreBounds=True)
            plot.addItem(self._crosshair_h, ignoreBounds=True)
            plot.addItem(label, ignoreBounds=True)
            self._hover_label = label
        except Exception:
            self._release_crosshair()
            return False
        return True

    def _release_crosshair(self) -> None:
        """Take the hover items out of the scene and forget them.

        Idempotent and never raising: this runs from hide/close, which Qt can
        deliver while the C++ side is already going away. The handles are
        dropped BEFORE the removals so a re-entrant call - or a hover that
        arrives mid-teardown - cannot touch an item twice or reach into memory
        that pyqtgraph has already released. PlotItem.removeItem is itself a
        no-op for an item it no longer holds.
        """
        items = (self._crosshair_v, self._crosshair_h, self._hover_label)
        self._crosshair_v = None
        self._crosshair_h = None
        self._hover_label = None
        try:
            plot = self.getPlotItem()
        except Exception:
            return
        for item in items:
            if item is None:
                continue
            try:
                plot.removeItem(item)
            except Exception:
                pass

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # A hidden chart cannot be hovered, so this is the natural deterministic
        # teardown point - and the one Qt reliably delivers before the widget
        # and its scene start coming apart. The next hover rebuilds.
        self._release_crosshair()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._release_crosshair()
        super().closeEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._set_crosshair_visible(False)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Drive the readout from this widget's own event, not the scene's.

        The scene's sigMouseMoved carried the crosshair before; a connection
        into a per-widget bound method that Qt may emit while the widget is
        being destroyed is exactly the lifetime hazard this file no longer
        takes. GraphicsView already enables mouse tracking, so the override
        sees the same moves the signal did.
        """
        super().mouseMoveEvent(event)
        try:
            position = event.position() if hasattr(event, "position") else event.localPos()
            self._on_mouse_moved(self.mapToScene(position.toPoint()))
        except Exception:
            self._set_crosshair_visible(False)

    def _set_crosshair_visible(self, visible: bool) -> None:
        items = (self._crosshair_v, self._crosshair_h, self._hover_label)
        try:
            for item in items:
                if item is not None:
                    item.setVisible(bool(visible))
        except RuntimeError:
            # The C++ items went away underneath us. Drop the stale handles so
            # the next hover rebuilds instead of touching deleted memory.
            self._release_crosshair()

    def _on_mouse_moved(self, scene_position) -> None:
        """Move the pure-Qt crosshair and show the nearest drawn bar."""
        plot = self.getPlotItem()
        try:
            if not self._bars or not plot.sceneBoundingRect().contains(scene_position):
                self._set_crosshair_visible(False)
                return
            view = plot.vb.mapSceneToView(scene_position)
            payload = hover_readout(self._bars, view.x(), self._timeframe)
            if payload is None:
                self._set_crosshair_visible(False)
                return
            # Built only now: a hover with nothing to read out - an empty chart,
            # a move outside the plot - must not leave scene items behind.
            if not self._ensure_crosshair():
                return
            index, text = payload
            self._crosshair_v.setPos(index)
            self._crosshair_h.setPos(view.y())
            (x_low, x_high), (y_low, y_high) = plot.vb.viewRange()
            self._hover_label.setText(text)
            self._hover_label.setPos(
                x_low + (x_high - x_low) * 0.01,
                y_high - (y_high - y_low) * 0.02,
            )
            self._set_crosshair_visible(True)
        except Exception:
            # A transient scene teardown or invalid cached row hides the
            # decoration; mouse movement must never take down a chart.
            self._set_crosshair_visible(False)

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
            view_y = None
            try:
                scene_pos = self.mapToScene(event.position().toPoint())
                view_pos = self.getPlotItem().vb.mapSceneToView(scene_pos)
                index = int(round(view_pos.x()))
                view_y = view_pos.y()
                # The view is in log space when log scaling is on, and this
                # price arms a real level - map it back before it escapes.
                price = self.price_at(view_y)
            except Exception:
                index = -1
            if 0 <= index < len(self._bars):
                self.barClicked.emit(index)
                if price is not None and price > 0:
                    self.priceClicked.emit(index, price)
                if view_y is not None:
                    self._select_level_at(index, view_y)
        super().mousePressEvent(event)

    def _select_level_at(self, index: int, view_y: float) -> None:
        """Select the painted level the click hit; a miss clears the selection.

        A miss clearing is deliberate: the highlight says "this is the line
        the next capture will reference", so it has to stop saying that the
        moment the trader clicks somewhere else.
        """
        hit = self.level_at(index, view_y)
        if hit is None:
            self.select_level("")
            return
        self.select_level(str(hit.get("id") or ""))
        try:
            price = float(hit.get("price"))
        except (TypeError, ValueError):
            price = 0.0
        self.levelSelected.emit(
            str(hit.get("id") or ""), str(hit.get("family") or ""), price
        )
