from __future__ import annotations

"""Click-a-ticker quick look: D1 candles on top, M5 candles below.

D1 comes from the durable daily parquet store (always available offline);
M5 from BounceBot's cached bars (only names in the current scan set). Both
are local reads, so the popup fills synchronously on click. Overlays are
fixed per the desk's reading style: D1 = SMA50/100/200 + EMA8/15/21, M5 =
session VWAP with +/-1 sigma bands + EMA15/21 - just the candles otherwise.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from chart_watch import WATCH_KINDS
from ui import theme
from ui.widgets.candle_chart import CandleChart


def _legend_html(
    title: str,
    overlays: list[dict],
    *,
    missing_reason: str = "needs deeper stored history",
) -> str:
    parts = [f"<b>{title}</b>"]
    seen = set()
    missing = []
    for overlay in overlays:
        label = str(overlay.get("label") or "")
        if not label or label in seen:
            continue
        seen.add(label)
        if not any(value is not None for value in overlay.get("values") or []):
            # e.g. SMA200 while the durable daily store is still shorter than
            # 200 sessions: say why the line is absent instead of lying.
            missing.append(label)
            continue
        color = theme.color(str(overlay.get("color") or "neutral"))
        parts.append(f"<span style='color:{color};'>— {label}</span>")
    if missing:
        parts.append(
            f"<span style='color:{theme.color('text_muted')};'>"
            f"({', '.join(missing)}: {missing_reason})</span>"
        )
    # Keep each marker visually grouped, but leave ordinary spaces between
    # entries so a narrow popup can wrap instead of forcing one giant line.
    return " &nbsp; ".join(parts)


class SymbolSnapshotWidget(QWidget):
    """Reusable embedded D1-over-M5 snapshot view.

    Clicking a D1 candle opens a small menu offering a persistent level
    alert off that candle's high or low; the choice is emitted as
    ``d1LevelAlertRequested(symbol, direction, level, candle_date)`` for the
    hosting surface to arm (it stays on across sessions until it flags).
    """

    d1LevelAlertRequested = Signal(str, str, float, str)
    # Click-to-price from either chart, for hosts with a level box to fill.
    pricePicked = Signal(float)

    def __init__(self, parent=None, *, compact: bool = False) -> None:
        """``compact`` trades legend wrapping for chart height.

        The standalone popup is 1180px wide with height to spare, so its
        legends keep wrapping (a narrow popup showing one long unwrapped line
        was a real complaint). The desk's embedded pane is the opposite: it is
        height-starved, and wrapped legends measured 59px EACH at 2560x1440 -
        43% of the whole snapshot - so there it stays on one line.
        """
        super().__init__(parent)
        self._symbol = ""
        self._compact = bool(compact)
        # Latest snapshot dicts, retained so callers can quick-fill a price from
        # a drawn overlay (VWAP, +/-1 sigma). set_data plots overlays and drops
        # them, so without this the values are unrecoverable after rendering.
        self._d1: dict = {}
        self._m5: dict = {}
        # The whole snapshot must be Expanding: it is the thing that should eat
        # the column's spare height. It previously declared Preferred with
        # verticalStretch 0, so the chart pane's stretch factor never reached
        # the charts themselves.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        legend_v = QSizePolicy.Policy.Fixed if self._compact else QSizePolicy.Policy.Preferred
        self.d1_legend = QLabel()
        self.d1_legend.setTextFormat(Qt.TextFormat.RichText)
        self.d1_legend.setWordWrap(not self._compact)
        self.d1_legend.setSizePolicy(QSizePolicy.Policy.Expanding, legend_v)
        self.d1_chart = CandleChart()
        self.d1_chart.barClicked.connect(self._on_d1_bar_clicked)
        self.d1_chart.setMinimumHeight(120)
        self.d1_note = QLabel()
        self.d1_note.setObjectName("MutedLabel")
        self.d1_note.setWordWrap(True)
        self.d1_note.setVisible(False)

        self.m5_legend = QLabel()
        self.m5_legend.setTextFormat(Qt.TextFormat.RichText)
        self.m5_legend.setWordWrap(not self._compact)
        self.m5_legend.setSizePolicy(QSizePolicy.Policy.Expanding, legend_v)
        self.m5_chart = CandleChart()
        # M5 candle clicks used to be inert: only the D1 chart was wired, so an
        # opening-range high, a premarket high, or any intraday level could not
        # be armed by clicking the bar that shows it.
        self.m5_chart.barClicked.connect(self._on_m5_bar_clicked)
        self.m5_chart.setMinimumHeight(120)
        for chart in (self.d1_chart, self.m5_chart):
            chart.priceClicked.connect(
                lambda _index, price: self.pricePicked.emit(price)
            )
        self.m5_note = QLabel()
        self.m5_note.setObjectName("MutedLabel")
        self.m5_note.setWordWrap(True)
        self.m5_note.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self.d1_legend)
        layout.addWidget(self.d1_chart, 1)
        layout.addWidget(self.d1_note)
        layout.addWidget(self.m5_legend)
        layout.addWidget(self.m5_chart, 1)
        layout.addWidget(self.m5_note)

    def set_symbol(self, symbol: str, *, bot=None) -> None:
        import chart_snapshot

        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        self._symbol = symbol
        d1 = chart_snapshot.build_d1_snapshot(symbol)
        self._d1 = d1
        self.d1_legend.setText(_legend_html(f"{symbol} · D1", d1["overlays"]))
        self.d1_chart.set_data(d1["bars"], d1["overlays"], timeframe="d1")
        self.d1_chart.setVisible(bool(d1["bars"]))
        self.d1_note.setVisible(not d1["bars"])
        if not d1["bars"]:
            self.d1_note.setText(
                f"No daily store for {symbol} - it is outside the built universe "
                "(Universe tab rebuilds fill the store)."
            )
        else:
            last = d1["bars"][-1]["dt"]
            self.d1_legend.setText(
                self.d1_legend.text()
                + f" &nbsp; <span style='color:{theme.color('text_muted')};'>"
                + f"through {last.strftime('%m/%d')}</span>"
            )

        m5_bars = []
        if bot is not None:
            try:
                m5_bars = bot.m5_chart_bars(symbol, max_sessions=2)
            except Exception:
                m5_bars = []
        m5 = chart_snapshot.build_m5_snapshot(symbol, m5_bars)
        self._m5 = m5
        self.m5_legend.setText(
            _legend_html(
                f"{symbol} · M5",
                m5["overlays"],
                missing_reason="needs positive cached volume",
            )
        )
        self.m5_chart.set_data(m5["bars"], m5["overlays"], timeframe="m5")
        self.m5_chart.setVisible(bool(m5["bars"]))
        self.m5_note.setVisible(not m5["bars"])
        if not m5["bars"]:
            self.m5_note.setText(
                f"No cached M5 bars for {symbol} - it is not in the current scan set, "
                "or the bot has not completed a scan cycle yet."
            )
        else:
            last = m5["bars"][-1]["dt"]
            self.m5_legend.setText(
                self.m5_legend.text()
                + f" &nbsp; <span style='color:{theme.color('text_muted')};'>"
                + f"last bar {last.strftime('%m/%d %H:%M')}</span>"
            )

    def quick_fill(self, source: str) -> float | None:
        """Resolve a quick-fill source against the M5 chart's drawn series.

        HOD/LOD/Last come from the intraday bars; VWAP and the sigma bands come
        from the overlays already plotted on them, so the number that fills is
        the line the trader is looking at.
        """
        from ui.widgets.arm_bar import quick_fill_value

        snapshot = self._m5 or {}
        return quick_fill_value(
            source, snapshot.get("bars") or [], snapshot.get("overlays") or []
        )

    def _on_d1_bar_clicked(self, index: int) -> None:
        self._popup_level_menu(self.d1_chart, index, "%m/%d")

    def _on_m5_bar_clicked(self, index: int) -> None:
        self._popup_level_menu(self.m5_chart, index, "%m/%d %H:%M")

    def _popup_level_menu(self, chart, index: int, stamp_format: str) -> None:
        """Offer a persistent break-above/below alert off a clicked candle.

        Both timeframes route into the same persistent D1 level watch: the
        watch is a price level, not a bar, so an M5 candle's high is as valid
        an anchor as a daily candle's.
        """
        bar = chart.bar_at(index)
        if bar is None or not self._symbol:
            return
        stamp = bar["dt"].strftime(stamp_format)
        menu = QMenu(self)
        above = menu.addAction(f"Alert: break above {bar['high']:.2f} ({stamp} high)")
        above.triggered.connect(
            lambda: self._emit_level_alert(chart, "above", index)
        )
        below = menu.addAction(f"Alert: break below {bar['low']:.2f} ({stamp} low)")
        below.triggered.connect(
            lambda: self._emit_level_alert(chart, "below", index)
        )
        menu.popup(QCursor.pos())

    def request_d1_level_alert(self, direction: str, index: int) -> None:
        """Emit the persistent level alert for a clicked D1 candle's high/low."""
        self._emit_level_alert(self.d1_chart, direction, index)

    def request_m5_level_alert(self, direction: str, index: int) -> None:
        """Emit the persistent level alert for a clicked M5 candle's high/low."""
        self._emit_level_alert(self.m5_chart, direction, index)

    def _emit_level_alert(self, chart, direction: str, index: int) -> None:
        bar = chart.bar_at(index)
        if bar is None or not self._symbol or direction not in ("above", "below"):
            return
        level = float(bar["high"] if direction == "above" else bar["low"])
        self.d1LevelAlertRequested.emit(
            self._symbol, direction, level, bar["dt"].strftime("%Y-%m-%d")
        )


class SymbolSnapshotDialog(QDialog):
    """Non-modal two-chart snapshot, reused across clicks (one per panel).

    When opened with a ``watch_host`` (the Alert Center panel, or anything
    exposing ``armed_watch_kinds`` / ``arm_chart_watch_for`` /
    ``disarm_chart_watch_for`` / ``is_d1_focus_active`` /
    ``toggle_d1_focus`` / ``is_m5_focus`` / ``toggle_m5_focus``) the popup
    grows the chart-only action row: "Add to D1 Focus" (Swing Focus picks +
    D1 Focus feed pin) and "Add to M5 Focus" (day-trade list) toggles plus
    the one-shot watch toggles whose hits flag red in the Alert Center.
    Everything is a toggle - a second click removes or disarms. Without a
    host the popup stays a pure quick look.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.resize(1180, 760)
        self.watch_host = None
        # Review-flow host (the setups panel): supplies the popup's ✕ dislike
        # and the advance-to-next-chart follow-through, so the trader can
        # work the whole table chart by chart without touching it.
        self.review_host = None
        self._symbol = ""
        self._side = ""

        self.snapshot = SymbolSnapshotWidget(self)
        self.snapshot.d1LevelAlertRequested.connect(self._on_d1_level_alert)
        # Compatibility aliases for existing callers and tests.
        for name in (
            "d1_legend",
            "d1_chart",
            "d1_note",
            "m5_legend",
            "m5_chart",
            "m5_note",
        ):
            setattr(self, name, getattr(self.snapshot, name))

        self.d1_focus_button = QPushButton("Add to D1 Focus")
        self.d1_focus_button.setCheckable(True)
        self.d1_focus_button.setToolTip(
            "Toggle this pick into Swing Focus (it lands on the Focus Picks "
            "tab and the swing watchlists) and pin it in the Alert Center's "
            "D1 Focus feed. Click again to remove both."
        )
        self.d1_focus_button.clicked.connect(self._toggle_d1_focus)
        self.m5_focus_button = QPushButton("Add to M5 Focus")
        self.m5_focus_button.setCheckable(True)
        self.m5_focus_button.setToolTip(
            "Toggle this symbol onto the M5 Focus day-trade list (BounceBot "
            "M5-scans it immediately). Click again to remove."
        )
        self.m5_focus_button.clicked.connect(self._toggle_m5_focus)
        self.watch_buttons: dict[str, QPushButton] = {}
        for kind, label in WATCH_KINDS.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(
                f"Toggle a one-shot {label} watch for this symbol. The first "
                "completed M5 bar that meets it fires a red alert in the "
                "Alert Center (bypasses the tier gate and sounds). Click "
                "again to disarm."
            )
            button.clicked.connect(
                lambda _checked=False, k=kind: self._toggle_watch(k)
            )
            self.watch_buttons[kind] = button

        # Review-flow ✕: log a dislike (with the typed reason) and advance to
        # the next visible setup row's chart. Only shown when a review host
        # (the setups panel) opened this popup.
        self.dislike_button = QPushButton("✕ Dislike")
        self.dislike_button.setToolTip(
            "Log a dislike for this setup - you'll be asked why, and the "
            "reason feeds the review-learning loop - then advance to the "
            "next chart in the setups table."
        )
        self.dislike_button.clicked.connect(self._review_dislike)

        self.action_row = QWidget()
        action_layout = QHBoxLayout(self.action_row)
        action_layout.setContentsMargins(10, 0, 10, 8)
        action_layout.setSpacing(6)
        action_layout.addWidget(self.dislike_button)
        action_layout.addWidget(self.d1_focus_button)
        action_layout.addWidget(self.m5_focus_button)
        for button in self.watch_buttons.values():
            action_layout.addWidget(button)
        action_layout.addStretch(1)
        self.action_row.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.snapshot)
        layout.addWidget(self.action_row)

    def show_symbol(
        self, symbol: str, *, bot=None, side: str = "", watch_host=None, review_host=None
    ) -> None:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        self._symbol = symbol
        self._side = side if side in ("LONG", "SHORT") else ""
        self.watch_host = watch_host
        self.review_host = review_host
        side_text = f" ({side})" if side in ("LONG", "SHORT") else ""
        self.setWindowTitle(f"{symbol}{side_text} — D1 + M5 snapshot")
        self.snapshot.set_symbol(symbol, bot=bot)
        self._refresh_watch_actions()
        # show + raise only (no activateWindow): the popup must never steal
        # typing focus from a watchlist editor or the live feed.
        self.show()
        self.raise_()

    def _refresh_watch_actions(self) -> None:
        host = self.watch_host
        reviewing = self.review_host is not None
        self.action_row.setVisible(host is not None or reviewing)
        self.dislike_button.setVisible(reviewing)
        # Watch/focus toggles need a watch host to act through; hide them
        # rather than showing dead buttons in a review-only popup.
        for button in (self.d1_focus_button, self.m5_focus_button, *self.watch_buttons.values()):
            button.setVisible(host is not None)
        if host is None:
            return
        try:
            armed = set(host.armed_watch_kinds(self._symbol))
        except Exception:
            armed = set()
        for kind, button in self.watch_buttons.items():
            label = WATCH_KINDS[kind]
            is_armed = kind in armed
            button.setText(f"{label} ✓ armed" if is_armed else label)
            button.setChecked(is_armed)
        pinned = bool(host.is_d1_focus_active(self._symbol, self._side))
        self.d1_focus_button.setText("✓ In D1 Focus" if pinned else "Add to D1 Focus")
        self.d1_focus_button.setChecked(pinned)
        in_m5 = bool(host.is_m5_focus(self._symbol, self._side))
        self.m5_focus_button.setText("✓ In M5 Focus" if in_m5 else "Add to M5 Focus")
        self.m5_focus_button.setChecked(in_m5)

    def _toggle_watch(self, kind: str) -> None:
        if self.watch_host is None or not self._symbol:
            return
        if kind in set(self.watch_host.armed_watch_kinds(self._symbol)):
            self.watch_host.disarm_chart_watch_for(self._symbol, kind)
        else:
            self.watch_host.arm_chart_watch_for(
                self._symbol,
                self._side or "WATCH",
                kind,
                source_text=f"chart snapshot: {self.windowTitle()}",
            )
        self._refresh_watch_actions()

    def _review_dislike(self) -> None:
        """The review-flow ✕: the host prompts for the reason, logs it, and
        advances to the next chart; a cancelled prompt leaves this one up."""
        if self.review_host is None or not self._symbol:
            return
        self.review_host.snapshot_review_dislike(self._symbol)

    def _toggle_d1_focus(self) -> None:
        if self.watch_host is None or not self._symbol:
            return
        added = self.watch_host.toggle_d1_focus(
            self._symbol,
            self._side,
            origin="chart",
            context=f"chart snapshot: {self.windowTitle()}",
        )
        self._refresh_watch_actions()
        # Review flow: filing the pick into D1 Focus is a decision made -
        # move on to the next chart. Toggling OFF is a correction, not a
        # decision; it stays on this chart.
        if added and self.review_host is not None:
            self.review_host.snapshot_review_advance()

    def _toggle_m5_focus(self) -> None:
        if self.watch_host is None or not self._symbol:
            return
        self.watch_host.toggle_m5_focus(
            self._symbol,
            self._side,
            origin="chart",
            context=f"chart snapshot: {self.windowTitle()}",
        )
        self._refresh_watch_actions()

    def _on_d1_level_alert(self, symbol: str, direction: str, level: float, candle_date: str) -> None:
        if self.watch_host is None:
            return
        self.watch_host.arm_d1_level_watch(
            symbol, direction, level, candle_date=candle_date
        )


def show_symbol_snapshot(
    owner, symbol: str, *, bot=None, side: str = "", watch_host=None, review_host=None
) -> SymbolSnapshotDialog:
    """Panel helper: lazily create one reusable dialog per owner widget."""
    dialog = getattr(owner, "_symbol_snapshot_dialog", None)
    if dialog is None:
        dialog = SymbolSnapshotDialog(owner)
        owner._symbol_snapshot_dialog = dialog
    dialog.show_symbol(
        symbol, bot=bot, side=side, watch_host=watch_host, review_host=review_host
    )
    return dialog
