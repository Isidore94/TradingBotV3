"""Always-on sector / industry strength tape across the top of the desk.

The intraday group read already existed, but it lived in the lower half of the
second tab of a splitter section, and `_group_strength_html` rendered only the
top-2 and bottom-2 per group type per timeframe - discarding most of what the
scan computed. "Occasionally look at industry/sector RS/RW" is what that access
cost produces.

Zero new data: this renders the same `group_strength` payload the Alert Center
already receives every scan cycle. No new service, thread, timer, or IB request.

Each chip carries a three-bar D1 | H1 | M5 sparkline, which is the read no
existing surface shows at a glance: a red D1 bar over a green M5 bar is a group
turning up today against a weak base - fresh rotation in.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui import theme

# Ordered weakest-timeframe-last so the sparkline reads left-to-right as
# "where it has been" -> "where it is now".
SPARK_TIMEFRAMES = ("D1", "H1", "M5")
STRIP_HEIGHT = 76
MAX_SECTOR_CHIPS = 11
MAX_INDUSTRY_CHIPS_NARROW = 9
MAX_INDUSTRY_CHIPS_WIDE = 23
WIDE_THRESHOLD = 2000


def rotation_callout(groups: dict[str, Any]) -> tuple[str, str]:
    """The one line worth reading: what is turning up, and what is fading.

    Rotating IN = strongest on M5 while still negative on D1 (today's move is
    not yet in the daily read). FADING = the mirror. Returns ("", "") when the
    payload cannot support the call rather than inventing one.
    """
    m5 = _rows(groups, "M5")
    d1 = {row["etf"]: row for row in _rows(groups, "D1")}
    if not m5 or not d1:
        return "", ""

    rotating_in = ""
    fading = ""
    for row in sorted(m5, key=lambda item: -(item["rrs"])):
        daily = d1.get(row["etf"])
        if daily is not None and row["rrs"] > 0 and daily["rrs"] < 0:
            rotating_in = f"{row['etf']} {row['rrs']:+.1f} M5 (D1 {daily['rrs']:+.1f})"
            break
    for row in sorted(m5, key=lambda item: item["rrs"]):
        daily = d1.get(row["etf"])
        if daily is not None and row["rrs"] < 0 and daily["rrs"] > 0:
            fading = f"{row['etf']} {row['rrs']:+.1f} M5 (D1 {daily['rrs']:+.1f})"
            break
    return rotating_in, fading


def _rows(groups: dict[str, Any], timeframe: str) -> list[dict[str, Any]]:
    frame = groups.get(timeframe) if isinstance(groups.get(timeframe), dict) else {}
    out = []
    for key in ("sectors", "industries"):
        for item in frame.get(key) or []:
            if not isinstance(item, dict):
                continue
            etf = str(item.get("etf") or "").strip().upper()
            rrs = _f(item.get("rrs"))
            if etf and rrs is not None:
                out.append({"etf": etf, "rrs": rrs, "kind": key})
    return out


def build_chips(groups: dict[str, Any], kind: str, limit: int) -> list[dict[str, Any]]:
    """One chip per group, with its RRS across every timeframe.

    Ranked by the M5 read (what is happening now) and truncated symmetrically:
    the strongest AND the weakest both matter, so a plain head-of-list cut
    would hide every short candidate.
    """
    by_etf: dict[str, dict[str, Any]] = {}
    for timeframe in SPARK_TIMEFRAMES:
        for row in _rows(groups, timeframe):
            if row["kind"] != kind:
                continue
            chip = by_etf.setdefault(row["etf"], {"etf": row["etf"], "spark": {}})
            chip["spark"][timeframe] = row["rrs"]
    chips = [chip for chip in by_etf.values() if chip["spark"]]
    for chip in chips:
        chip["rrs"] = chip["spark"].get("M5")
        if chip["rrs"] is None:
            chip["rrs"] = next(
                (chip["spark"][tf] for tf in SPARK_TIMEFRAMES if tf in chip["spark"]),
                0.0,
            )
    chips.sort(key=lambda chip: -chip["rrs"])
    if len(chips) <= limit:
        return chips
    head = (limit + 1) // 2
    tail = limit - head
    return chips[:head] + (chips[-tail:] if tail else [])


def _f(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


class Sparkline(QWidget):
    """Three signed bars, one per timeframe, centred on zero."""

    def __init__(self, spark: dict[str, float], parent=None) -> None:
        super().__init__(parent)
        self._spark = spark
        self.setFixedSize(26, 18)
        self.setToolTip(
            " · ".join(
                f"{tf} {spark[tf]:+.2f}" for tf in SPARK_TIMEFRAMES if tf in spark
            )
            or "no read"
        )

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        width = self.width() / max(len(SPARK_TIMEFRAMES), 1)
        mid = self.height() / 2
        scale = max(
            [abs(value) for value in self._spark.values() if value is not None] or [1.0]
        )
        for index, timeframe in enumerate(SPARK_TIMEFRAMES):
            value = self._spark.get(timeframe)
            if value is None:
                continue
            colour = QColor(theme.color("long" if value >= 0 else "short"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colour)
            height = (abs(value) / scale) * (mid - 1)
            top = mid - height if value >= 0 else mid
            painter.drawRect(
                int(index * width) + 1, int(top), max(2, int(width) - 3), max(1, int(height))
            )
        painter.end()


class GroupChip(QFrame):
    activated = Signal(str)

    def __init__(self, chip: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.etf = chip["etf"]
        rrs = chip.get("rrs") or 0.0
        accent = theme.color("long" if rrs >= 0 else "short")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {theme.with_alpha(accent, 0.45)};"
            f" border-radius: 6px; background: {theme.with_alpha(accent, 0.10)}; }}"
            f" QLabel {{ border: none; background: transparent; }}"
        )
        name = QLabel(self.etf)
        name.setStyleSheet(f"color: {accent}; font-weight: 700;")
        value = QLabel(f"{rrs:+.1f}")
        value.setObjectName("MutedLabel")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 4, 1)
        layout.setSpacing(4)
        layout.addWidget(name)
        layout.addWidget(value)
        layout.addWidget(Sparkline(chip.get("spark") or {}))
        self.setToolTip(
            f"{self.etf} — click to chart it. Bars are D1 | H1 | M5 intraday RRS."
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.etf)
        super().mousePressEvent(event)


class GroupTapeStrip(QFrame):
    """The always-visible sector/industry tape."""

    symbolActivated = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setFixedHeight(STRIP_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.sector_row = QWidget()
        self.sector_layout = QHBoxLayout(self.sector_row)
        self.sector_layout.setContentsMargins(0, 0, 0, 0)
        self.sector_layout.setSpacing(4)

        self.industry_row = QWidget()
        self.industry_layout = QHBoxLayout(self.industry_row)
        self.industry_layout.setContentsMargins(0, 0, 0, 0)
        self.industry_layout.setSpacing(4)

        self.callout = QLabel("")
        self.callout.setObjectName("MutedLabel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(2)
        layout.addWidget(self.sector_row)
        layout.addWidget(self.industry_row)
        layout.addWidget(self.callout)
        self.update_groups({})

    def update_groups(self, payload: Any) -> None:
        """Render from an RRS payload (or its bare `group_strength` mapping)."""
        payload = payload if isinstance(payload, dict) else {}
        groups = payload.get("group_strength")
        if not isinstance(groups, dict):
            groups = payload if any(tf in payload for tf in SPARK_TIMEFRAMES) else {}

        wide = self.width() >= WIDE_THRESHOLD
        industry_limit = MAX_INDUSTRY_CHIPS_WIDE if wide else MAX_INDUSTRY_CHIPS_NARROW
        sectors = build_chips(groups, "sectors", MAX_SECTOR_CHIPS)
        industries = build_chips(groups, "industries", industry_limit)

        self._fill(self.sector_layout, sectors, "SECTORS")
        self._fill(self.industry_layout, industries, "INDUSTRIES")

        rotating_in, fading = rotation_callout(groups)
        if rotating_in or fading:
            bits = []
            if rotating_in:
                bits.append(f"ROTATING IN: {rotating_in}")
            if fading:
                bits.append(f"FADING: {fading}")
            self.callout.setText("   ·   ".join(bits))
        elif groups:
            self.callout.setText("No clean rotation split right now.")
        else:
            self.callout.setText(
                "Waiting for BounceBot's group scan — sector/industry RRS fills in "
                "automatically once it has cached ETF bars."
            )

    def _fill(self, layout, chips, caption: str) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Detach now, free later: deleteLater alone leaves the old chip
                # parented until the event loop turns, so a stale group can
                # still be found (and clicked) after the tape has moved on.
                widget.setParent(None)
                widget.deleteLater()
        label = QLabel(caption)
        label.setObjectName("MutedLabel")
        label.setMinimumWidth(74)
        layout.addWidget(label)
        for chip in chips:
            widget = GroupChip(chip)
            widget.activated.connect(self.symbolActivated)
            layout.addWidget(widget)
        layout.addStretch(1)
