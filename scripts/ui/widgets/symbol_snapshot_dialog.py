from __future__ import annotations

"""Click-a-ticker quick look: D1 candles on top, M5 candles below.

D1 comes from the durable daily parquet store (always available offline);
M5 from BounceBot's cached bars (only names in the current scan set). Both
are local reads, so the popup fills synchronously on click. Overlays are
fixed per the desk's reading style: D1 = SMA50/100/200 + EMA8/15/21, M5 =
session VWAP with +/-1 sigma bands + EMA15/21 - just the candles otherwise.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QSizePolicy, QVBoxLayout, QWidget

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
    """Reusable embedded D1-over-M5 snapshot view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.d1_legend = QLabel()
        self.d1_legend.setTextFormat(Qt.TextFormat.RichText)
        self.d1_legend.setWordWrap(True)
        self.d1_legend.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.d1_chart = CandleChart()
        self.d1_note = QLabel()
        self.d1_note.setObjectName("MutedLabel")
        self.d1_note.setWordWrap(True)
        self.d1_note.setVisible(False)

        self.m5_legend = QLabel()
        self.m5_legend.setTextFormat(Qt.TextFormat.RichText)
        self.m5_legend.setWordWrap(True)
        self.m5_legend.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.m5_chart = CandleChart()
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
        d1 = chart_snapshot.build_d1_snapshot(symbol)
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


class SymbolSnapshotDialog(QDialog):
    """Non-modal two-chart snapshot, reused across clicks (one per panel)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.resize(1180, 760)

        self.snapshot = SymbolSnapshotWidget(self)
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.snapshot)

    def show_symbol(self, symbol: str, *, bot=None, side: str = "") -> None:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        side_text = f" ({side})" if side in ("LONG", "SHORT") else ""
        self.setWindowTitle(f"{symbol}{side_text} — D1 + M5 snapshot")
        self.snapshot.set_symbol(symbol, bot=bot)
        # show + raise only (no activateWindow): the popup must never steal
        # typing focus from a watchlist editor or the live feed.
        self.show()
        self.raise_()


def show_symbol_snapshot(owner, symbol: str, *, bot=None, side: str = "") -> SymbolSnapshotDialog:
    """Panel helper: lazily create one reusable dialog per owner widget."""
    dialog = getattr(owner, "_symbol_snapshot_dialog", None)
    if dialog is None:
        dialog = SymbolSnapshotDialog(owner)
        owner._symbol_snapshot_dialog = dialog
    dialog.show_symbol(symbol, bot=bot, side=side)
    return dialog
