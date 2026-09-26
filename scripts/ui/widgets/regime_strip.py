"""The desk's regime strip (S17): six auto-regime cells (M5..W) per index.

It only formats. The readings come from `BounceService.regimeStripChanged`,
computed on a worker kicked by the existing auto-regime timer. Styles live in
`ui/theme.qss` (object names below, `tone` property); a cell is re-polished
only when its tone changes.
"""

from __future__ import annotations

from typing import Any, Mapping

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

INDEXES = ("SPY", "QQQ", "IWM")
TIMEFRAMES = ("M5", "M30", "H1", "H4", "D1", "W")
STRIP_OBJECT_NAME = "RegimeStrip"
SYMBOL_OBJECT_NAME = "RegimeStripSymbol"
CELL_OBJECT_NAME = "RegimeCell"

#: env_key -> (short word, tone the stylesheet keys on).
_WORDS = {
    "bullish_strong": ("Up+", "bull"),
    "bullish_weak": ("Up", "bull"),
    "neutral_chop": ("Chop", "chop"),
    "bearish_weak": ("Dn", "bear"),
    "bearish_strong": ("Dn+", "bear"),
}


def format_cell(timeframe: str, env_key: Any) -> tuple[str, str]:
    """`(text, tone)` for one cell; anything but a champion env_key is unknown."""
    word, tone = _WORDS.get(str(env_key or "").strip().lower(), ("?", "unknown"))
    return f"{timeframe} {word}", tone


def format_strip(payload: Any) -> dict[str, list[tuple[str, str, str]]]:
    """Per index, six `(text, tone, tooltip)` cells from the service payload."""
    payload = payload if isinstance(payload, Mapping) else {}
    readings = payload.get("symbols") if isinstance(payload.get("symbols"), Mapping) else {}
    as_of = str(payload.get("as_of") or "")
    cells: dict[str, list[tuple[str, str, str]]] = {}
    for symbol in INDEXES:
        row = readings.get(symbol) if isinstance(readings.get(symbol), Mapping) else {}
        out = []
        for timeframe in TIMEFRAMES:
            env_key = str(row.get(timeframe) or "unknown")
            text, tone = format_cell(timeframe, env_key)
            tip = f"{symbol} {timeframe}: {env_key}" + (f" (completed bars to {as_of})" if as_of else "")
            out.append((text, tone, tip))
        cells[symbol] = out
    return cells


class RegimeStrip(QFrame):
    """SPY / QQQ / IWM, each with six auto-regime cells."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(STRIP_OBJECT_NAME)
        self.setToolTip(
            "Auto regimes (champion Auto Market Bias) on completed bars: M5, M30, H1, H4, D1, W. "
            "? = not enough bars. Context only."
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.cells: dict[str, list[QLabel]] = {}
        for symbol in INDEXES:
            name = QLabel(symbol, self)
            name.setObjectName(SYMBOL_OBJECT_NAME)
            row.addWidget(name)
            labels = []
            for timeframe in TIMEFRAMES:
                cell = QLabel(f"{timeframe} ?", self)
                cell.setObjectName(CELL_OBJECT_NAME)
                cell.setProperty("tone", "unknown")
                row.addWidget(cell)
                labels.append(cell)
            self.cells[symbol] = labels
        row.addStretch(1)

    def set_readings(self, payload: Any) -> None:
        for symbol, cells in format_strip(payload).items():
            for label, (text, tone, tip) in zip(self.cells[symbol], cells, strict=True):
                if label.text() != text:
                    label.setText(text)
                label.setToolTip(tip)
                if label.property("tone") != tone:
                    label.setProperty("tone", tone)
                    style = label.style()
                    style.unpolish(label)
                    style.polish(label)
