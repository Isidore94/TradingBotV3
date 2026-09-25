"""Hindsight clue marking on the desk's `CandleChart` (Day Recap coach).

`enable_clue_marking(chart, on_mark)` adds a clue mode: a left click snaps to
the nearest COMPLETED bar and calls ``on_mark(bar_time, price)`` with a
tz-aware bar time. Esc or ``set_active(False)`` leaves the mode.

`ClueForm` picks a clue tag and a short note and saves through
`recap_store.record_clue` on a worker thread. `draw_clues` shows saved clues
as small labelled markers. `mark_clue_flow` wires all three for the Walk.

Evidence only: nothing here detects, scores, ranks or alerts.
"""

from __future__ import annotations

import math
import threading
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable, Mapping

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPoint, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

import recap_store
from ui import theme
from swallowed import note_swallowed

#: Plain words for each clue tag, in `recap_store.CLUE_TAGS` order.
CLUE_TAG_LABELS: dict[str, str] = {
    "volume_dry_up": "Volume dried up",
    "volume_surge": "Volume surge",
    "vwap_reclaim": "VWAP reclaim",
    "vwap_reject": "VWAP reject",
    "level_break": "Level broke",
    "level_hold": "Level held",
    "rs_vs_spy": "Strong vs SPY",
    "sector_move": "Sector moved",
    "news": "News",
    "gap": "Gap",
    "trendline_break": "Trendline broke",
    "sma_reclaim": "SMA reclaim",
    "other": "Other",
}

#: The short glyph a saved clue draws on a chart. Unique per tag.
CLUE_TAG_INITIALS: dict[str, str] = {
    "volume_dry_up": "VD",
    "volume_surge": "VS",
    "vwap_reclaim": "VR",
    "vwap_reject": "VX",
    "level_break": "LB",
    "level_hold": "LH",
    "rs_vs_spy": "RS",
    "sector_move": "SM",
    "news": "N",
    "gap": "G",
    "trendline_break": "TB",
    "sma_reclaim": "SR",
    "other": "O",
}

CLUE_NOTE_MAX = recap_store.SHORT_TEXT_MAX
_INTRADAY_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}
_CLUE_COLOR = "caution"


# ---------------------------------------------------------------------------
# time helpers (pure)
# ---------------------------------------------------------------------------
def _market_tz():
    import market_session

    return market_session.get_market_local_timezone()[0]


def aware_bar_time(stamp: Any) -> datetime | None:
    """A bar's `dt` as a tz-aware datetime. Naive bars are market-local time."""
    if isinstance(stamp, str):
        try:
            stamp = datetime.fromisoformat(stamp.strip())
        except ValueError:
            return None
    if isinstance(stamp, date) and not isinstance(stamp, datetime):
        stamp = datetime.combine(stamp, datetime.min.time())
    if not isinstance(stamp, datetime):
        return None
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        return stamp.replace(tzinfo=_market_tz())
    return stamp


def bar_end(bar_time: datetime, timeframe: str) -> datetime:
    """When the bar that opened at ``bar_time`` is complete."""
    frame = str(timeframe or "").upper()
    if frame in _INTRADAY_MINUTES:
        return bar_time + timedelta(minutes=_INTRADAY_MINUTES[frame])
    if frame == "W1":
        return bar_time + timedelta(days=7)
    import market_calendar

    try:
        return market_calendar.session_close(bar_time.date())
    except Exception:  # noqa: BLE001 - outside the calendar: treat as the next day
        return bar_time + timedelta(days=1)


def is_completed(bar: Mapping[str, Any], timeframe: str, now: datetime) -> bool:
    """True when the bar has closed at ``now``. Unknown time is not complete."""
    stamp = aware_bar_time(bar.get("dt"))
    if stamp is None:
        return False
    return bar_end(stamp, timeframe) <= now


def snap_to_completed(
    bars: list[Mapping[str, Any]], index: int, timeframe: str, now: datetime
) -> int | None:
    """The completed bar nearest ``index``; a forming bar is never chosen."""
    if not bars:
        return None
    start = min(max(int(index), 0), len(bars) - 1)
    for step in range(len(bars)):
        for candidate in (start - step, start + step):
            if 0 <= candidate < len(bars) and is_completed(bars[candidate], timeframe, now):
                return candidate
    return None


def _now() -> datetime:
    return datetime.now().astimezone()


def _chart_bars(chart) -> list[dict]:
    return [chart.bar_at(index) for index in range(chart.bar_count())]


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clue_context(bar: Mapping[str, Any] | None, spy_bars: Iterable[Mapping[str, Any]] | None = None) -> dict:
    """The per-clue snapshot for the AI: the bar's OHLCV and SPY's same-bar close.

    Missing data stays ``None`` (unknown), never a guess.
    """
    snap: dict[str, Any] = {"bar": None, "spy_close": None}
    if bar:
        stamp = aware_bar_time(bar.get("dt"))
        snap["bar"] = {
            "time": stamp.isoformat() if stamp else None,
            **{key: _number(bar.get(key)) for key in ("open", "high", "low", "close", "volume")},
        }
        if stamp is not None:
            for spy in spy_bars or ():
                if aware_bar_time(spy.get("dt")) == stamp:
                    snap["spy_close"] = _number(spy.get("close"))
                    break
    return snap


# ---------------------------------------------------------------------------
# clue mode
# ---------------------------------------------------------------------------
class ClueMarker(QObject):
    """Clue mode on one chart. Built by `enable_clue_marking`."""

    #: (tz-aware bar time, price at the click, snapped bar index)
    marked = Signal(object, float, int)
    activeChanged = Signal(bool)

    def __init__(
        self,
        chart,
        on_mark: Callable[[datetime, float], Any] | None,
        *,
        timeframe: str = "M5",
        clock: Callable[[], datetime] = _now,
    ) -> None:
        super().__init__(chart)
        self._chart = chart
        self._on_mark = on_mark
        self.timeframe = str(timeframe or "M5").upper()
        self._clock = clock
        self._active = False
        chart.installEventFilter(self)
        chart.viewport().installEventFilter(self)

    def is_active(self) -> bool:
        return self._active

    def set_active(self, active: bool) -> None:
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        viewport = self._chart.viewport()
        if active:
            viewport.setCursor(Qt.CursorShape.CrossCursor)
            self._chart.setFocus(Qt.FocusReason.OtherFocusReason)
        else:
            viewport.unsetCursor()
        self.activeChanged.emit(active)

    def toggle(self) -> None:
        self.set_active(not self._active)

    def mark_at_view(self, view_x: float, view_y: float) -> bool:
        """Mark the completed bar nearest a point in view coordinates."""
        count = self._chart.bar_count()
        if not count or not math.isfinite(view_x) or not math.isfinite(view_y):
            return False
        bars = _chart_bars(self._chart)
        index = snap_to_completed(bars, int(round(view_x)), self.timeframe, self._clock())
        if index is None:
            return False
        stamp = aware_bar_time(bars[index].get("dt"))
        price = float(self._chart.price_at(view_y))
        if stamp is None or not (price > 0 and math.isfinite(price)):
            return False
        if self._on_mark is not None:
            self._on_mark(stamp, price)
        self.marked.emit(stamp, price, index)
        return True

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        if not self._active:
            return False
        kind = event.type()
        if kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self.set_active(False)
            return True
        if (
            kind == QEvent.Type.MouseButtonPress
            and watched is self._chart.viewport()
            and event.button() == Qt.MouseButton.LeftButton
        ):
            try:
                scene = self._chart.mapToScene(event.position().toPoint())
                view = self._chart.getPlotItem().vb.mapSceneToView(scene)
                self.mark_at_view(float(view.x()), float(view.y()))
            except Exception as exc:  # noqa: BLE001 - a bad click never takes the chart down
                note_swallowed("clue click could not be mapped to the chart", exc, quiet=True)
            # Consumed: in clue mode a click marks, it does not pan or open menus.
            return True
        return False


def enable_clue_marking(
    chart,
    on_mark: Callable[[datetime, float], Any] | None,
    *,
    timeframe: str = "M5",
    clock: Callable[[], datetime] = _now,
) -> ClueMarker:
    """Add clue mode to ``chart``. Call ``set_active(True)`` on the result to start."""
    return ClueMarker(chart, on_mark, timeframe=timeframe, clock=clock)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
class _CallSignals(QObject):
    done = Signal(object)
    failed = Signal(str, bool)  # (message, was the write itself lost)


class _CallWorker(QRunnable):
    """One store call off the Qt thread. Every ending emits."""

    def __init__(self, call: Callable[[], Any]) -> None:
        super().__init__()
        self._call = call
        self.signals = _CallSignals()

    def run(self) -> None:
        try:
            result = self._call()
        except recap_store.RecapWriteError as exc:
            self.signals.failed.emit(str(exc), True)
            return
        except Exception as exc:  # noqa: BLE001 - the trader must see "not saved"
            self.signals.failed.emit(str(exc), False)
            return
        self.signals.done.emit(result)


def clues_for(session_date: Any, symbol: Any, *, path=None) -> list[dict]:
    """One symbol's saved clues for a session, oldest first. Reads disk: call on a worker."""
    ticker = str(symbol or "").strip().upper()
    rows = recap_store.records_for(session_date, [recap_store.KIND_CLUE], path=path)
    return [row for row in rows if str(row.get("symbol") or "").upper() == ticker]


# ---------------------------------------------------------------------------
# the form
# ---------------------------------------------------------------------------
class ClueForm(QFrame):
    """Tag chips, a short note, Save / Cancel. Saving runs on a worker."""

    saved = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        parent=None,
        *,
        writer: Callable[..., dict] | None = None,
        pool: QThreadPool | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("clueForm")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self._writer = writer or recap_store.record_clue
        self._pool = pool or QThreadPool.globalInstance()
        self._payload: dict[str, Any] = {}
        self._workers: list[_CallWorker] = []
        self.write_threads: list[threading.Thread] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.title = QLabel("Mark a clue")
        layout.addWidget(self.title)
        chips = QGridLayout()
        chips.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.chips: dict[str, QPushButton] = {}
        for position, tag in enumerate(recap_store.CLUE_TAGS):
            button = QPushButton(CLUE_TAG_LABELS.get(tag, tag))
            button.setCheckable(True)
            button.setProperty("clue_tag", tag)
            self._group.addButton(button)
            chips.addWidget(button, position // 3, position % 3)
            self.chips[tag] = button
        self._group.buttonToggled.connect(lambda *_args: self._refresh_save())
        layout.addLayout(chips)
        self.note = QLineEdit()
        self.note.setMaxLength(CLUE_NOTE_MAX)
        self.note.setPlaceholderText("What do you see? (optional)")
        layout.addWidget(self.note)
        row = QHBoxLayout()
        self.status = QLabel("")
        row.addWidget(self.status, 1)
        self.cancel_button = QPushButton("Cancel")
        self.save_button = QPushButton("Save")
        self.save_button.setDefault(True)
        row.addWidget(self.cancel_button)
        row.addWidget(self.save_button)
        layout.addLayout(row)
        self.save_button.clicked.connect(self.save)
        self.cancel_button.clicked.connect(self._cancel)
        self._refresh_save()

    def open_for(self, **payload: Any) -> None:
        """Start a new clue: the `record_clue` fields except the tag and text."""
        self._payload = dict(payload)
        self._group.setExclusive(False)
        for button in self.chips.values():
            button.setChecked(False)
        self._group.setExclusive(True)
        self.note.clear()
        self.status.setText("")
        stamp = payload.get("bar_time")
        when = stamp.strftime("%m/%d %H:%M") if hasattr(stamp, "strftime") else ""
        price = _number(payload.get("price"))
        self.title.setText(
            f"Clue: {payload.get('symbol', '')} {when}"
            + (f" @ {price:,.2f}" if price is not None else "")
        )
        self._refresh_save()
        self.show()
        self.raise_()

    def chosen_tag(self) -> str:
        button = self._group.checkedButton()
        return str(button.property("clue_tag")) if button is not None else ""

    def _refresh_save(self) -> None:
        self.save_button.setEnabled(bool(self.chosen_tag()) and bool(self._payload))

    def save(self) -> None:
        tag = self.chosen_tag()
        if not tag or not self._payload:
            return
        fields = dict(self._payload)
        fields["clue_tag"] = tag
        fields["text"] = self.note.text().strip()
        writer = self._writer
        threads = self.write_threads

        def call() -> dict:
            threads.append(threading.current_thread())
            return writer(**fields)

        worker = _CallWorker(call)
        worker.setAutoDelete(False)
        worker.signals.done.connect(self._on_saved)
        worker.signals.failed.connect(self._on_failed)
        self._workers.append(worker)
        self.save_button.setEnabled(False)
        self.status.setText("saving...")
        self._pool.start(worker)

    def _on_saved(self, row: Any) -> None:
        self._workers = [w for w in self._workers if w.signals is not self.sender()]
        self.status.setText("saved")
        self._payload = {}
        self._refresh_save()
        self.saved.emit(dict(row) if isinstance(row, Mapping) else {})

    def _on_failed(self, message: str, _lost: bool) -> None:
        self._workers = [w for w in self._workers if w.signals is not self.sender()]
        self.status.setText("not saved")
        self.status.setToolTip(message)
        self._refresh_save()
        self.failed.emit(message)

    def _cancel(self) -> None:
        self._payload = {}
        self.hide()
        self.cancelled.emit()


# ---------------------------------------------------------------------------
# drawing saved clues
# ---------------------------------------------------------------------------
class _ClueLayer:
    """The pooled items one chart uses for clues. Built on the first clue."""

    def __init__(self, chart) -> None:
        self.chart = chart
        self.dots = pg.ScatterPlotItem(
            size=9,
            symbol="d",
            pen=pg.mkPen(theme.color("bg_panel"), width=1),
            brush=pg.mkBrush(theme.color(_CLUE_COLOR)),
        )
        self.dots.setZValue(22)
        chart.getPlotItem().addItem(self.dots, ignoreBounds=True)
        self.labels: list[pg.TextItem] = []
        self.drawn: list[tuple[int, float, dict]] = []


def _bar_index_map(chart, timeframe: str) -> dict[Any, int]:
    daily = str(timeframe or "").upper() in ("D1", "W1")
    out: dict[Any, int] = {}
    for index in range(chart.bar_count()):
        stamp = aware_bar_time((chart.bar_at(index) or {}).get("dt"))
        if stamp is not None:
            out.setdefault(stamp.date() if daily else stamp, index)
    return out


def _clue_key(clue: Mapping[str, Any], daily: bool):
    stamp = aware_bar_time(clue.get("bar_time"))
    if stamp is None:
        return None
    if daily:
        return stamp.astimezone(_market_tz()).date() if stamp.tzinfo else stamp.date()
    return stamp


def clue_tooltip(clue: Mapping[str, Any]) -> str:
    tag = str(clue.get("clue_tag") or "")
    words = CLUE_TAG_LABELS.get(tag, tag or "Clue")
    stamp = aware_bar_time(clue.get("bar_time"))
    when = stamp.strftime("%m/%d %H:%M") if stamp else ""
    price = _number(clue.get("price"))
    head = " ".join(part for part in (words, when, f"@ {price:,.2f}" if price else "") if part)
    text = str(clue.get("text") or "").strip()
    return f"{head}\n{text}" if text else head


def draw_clues(chart, clues: Iterable[Mapping[str, Any]], *, timeframe: str = "M5") -> int:
    """Draw saved clues on ``chart`` at their bar and price. Returns how many drew.

    A clue whose bar is not on this chart is skipped. Items are pooled and
    hidden, never rebuilt; a chart with no clues grows no items. Call again
    after `set_data`, like the note markers.
    """
    clues = [dict(clue) for clue in clues or ()]
    layer: _ClueLayer | None = getattr(chart, "_clue_layer", None)
    if not clues and layer is None:
        return 0
    daily = str(timeframe or "").upper() in ("D1", "W1")
    positions = _bar_index_map(chart, timeframe) if clues else {}
    placed: list[tuple[int, float, dict]] = []
    for clue in clues:
        index = positions.get(_clue_key(clue, daily))
        price = _number(clue.get("price"))
        if index is None or price is None or price <= 0:
            continue
        placed.append((index, price, clue))
    if layer is None:
        if not placed:
            return 0
        layer = _ClueLayer(chart)
        chart._clue_layer = layer
    log = bool(chart.is_log_scaled())
    ys = [math.log10(price) if log else price for _index, price, _clue in placed]
    layer.dots.setData(
        x=[float(index) for index, _price, _clue in placed],
        y=ys,
        data=[str(clue.get("id") or "") for _i, _p, clue in placed],
    )
    layer.dots.setVisible(bool(placed))
    while len(layer.labels) < len(placed):
        fill = QColor(theme.color("bg_elevated"))
        fill.setAlphaF(0.85)
        label = pg.TextItem(
            "",
            anchor=(0.5, 1.25),
            color=theme.color(_CLUE_COLOR),
            fill=pg.mkBrush(fill),
            border=pg.mkPen(theme.color(_CLUE_COLOR)),
        )
        label.setZValue(23)
        chart.getPlotItem().addItem(label, ignoreBounds=True)
        layer.labels.append(label)
    for position, label in enumerate(layer.labels):
        if position >= len(placed):
            label.setVisible(False)
            continue
        index, _price, clue = placed[position]
        tag = str(clue.get("clue_tag") or "")
        label.setText(CLUE_TAG_INITIALS.get(tag, "?"))
        label.setToolTip(clue_tooltip(clue))
        label.setPos(float(index), ys[position])
        label.setVisible(True)
    layer.drawn = placed
    return len(placed)


def drawn_clue_count(chart) -> int:
    layer = getattr(chart, "_clue_layer", None)
    if layer is None:
        return 0
    return sum(1 for label in layer.labels if label.isVisible())


# ---------------------------------------------------------------------------
# the flow the Walk calls
# ---------------------------------------------------------------------------
class ClueFlow(QObject):
    """Clue mode + form + drawn clues on one chart, for one session and symbol."""

    def __init__(
        self,
        chart,
        session_date: Any,
        symbol: Any,
        timeframe: str,
        *,
        card_id: str = "",
        trade_id: str = "",
        pick_id: str = "",
        spy_bars: Iterable[Mapping[str, Any]] | None = None,
        writer: Callable[..., dict] | None = None,
        loader: Callable[[Any, Any], list[dict]] | None = None,
        pool: QThreadPool | None = None,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        super().__init__(chart)
        self.chart = chart
        self.session_date = str(session_date or "")[:10]
        self.symbol = str(symbol or "").strip().upper()
        self.timeframe = str(timeframe or "M5").upper()
        self.links = {"card_id": card_id, "trade_id": trade_id, "pick_id": pick_id}
        self.spy_bars = list(spy_bars or ())
        self.clues: list[dict] = []
        self._pool = pool or QThreadPool.globalInstance()
        self._loader = loader or clues_for
        self._load_workers: list[_CallWorker] = []
        self.marker = enable_clue_marking(chart, None, timeframe=self.timeframe, clock=clock)
        self.marker.marked.connect(self._on_marked)
        self.form = ClueForm(chart, writer=writer, pool=self._pool)
        self.form.hide()
        self.form.saved.connect(self._on_saved)

    def set_active(self, active: bool) -> None:
        self.marker.set_active(active)
        if not active:
            self.form.hide()

    def set_spy_bars(self, bars: Iterable[Mapping[str, Any]] | None) -> None:
        self.spy_bars = list(bars or ())

    def load(self) -> None:
        """Read this symbol's saved clues on a worker, then draw them."""
        session, symbol, loader = self.session_date, self.symbol, self._loader
        worker = _CallWorker(lambda: loader(session, symbol))
        worker.setAutoDelete(False)
        worker.signals.done.connect(self._on_loaded)
        self._load_workers.append(worker)
        self._pool.start(worker)

    def redraw(self) -> int:
        return draw_clues(self.chart, self.clues, timeframe=self.timeframe)

    def _on_loaded(self, rows: Any) -> None:
        self._load_workers = [w for w in self._load_workers if w.signals is not self.sender()]
        self.clues = [dict(row) for row in rows or ()]
        self.redraw()

    def _on_marked(self, bar_time: datetime, price: float, index: int) -> None:
        bar = self.chart.bar_at(index)
        self.form.open_for(
            session_date=self.session_date,
            symbol=self.symbol,
            timeframe=self.timeframe,
            bar_time=bar_time,
            price=round(float(price), 4),
            context=clue_context(bar, self.spy_bars),
            **self.links,
        )
        self.form.adjustSize()
        self.place_form(index, price)

    def place_form(self, index: int, price: float) -> None:
        """Put the form next to the marked bar, inside the chart."""
        try:
            view = self.chart.getPlotItem().vb
            y = math.log10(price) if self.chart.is_log_scaled() else price
            point = self.chart.mapFromScene(view.mapViewToScene(pg.QtCore.QPointF(index, y)))
        except Exception:  # noqa: BLE001
            point = QPoint(12, 12)
        width, height = self.form.width(), self.form.height()
        x = point.x() + 16
        if x + width > self.chart.width():
            x = point.x() - width - 16
        y = min(max(point.y() - height // 2, 4), max(self.chart.height() - height - 4, 4))
        self.form.move(max(x, 4), y)

    def _on_saved(self, row: dict) -> None:
        if row:
            self.clues.append(row)
            self.redraw()


def mark_clue_flow(
    chart,
    session_date: Any,
    symbol: Any,
    timeframe: str,
    *,
    card_id: str = "",
    trade_id: str = "",
    pick_id: str = "",
    **options: Any,
) -> ClueFlow:
    """Wire clue mode, the form and the saved-clue markers onto ``chart``.

    Returns the `ClueFlow`; call ``set_active(True)`` to start marking and
    ``load()`` to draw the clues already saved. ``options`` passes
    ``spy_bars``, ``writer``, ``loader``, ``pool`` and ``clock`` through.
    """
    return ClueFlow(
        chart, session_date, symbol, timeframe,
        card_id=card_id, trade_id=trade_id, pick_id=pick_id, **options,
    )
