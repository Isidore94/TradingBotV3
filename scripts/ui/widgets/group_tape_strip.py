"""Always-on sector / industry strength tape across the top of the desk.

The intraday group read already existed, but it lived in the lower half of the
second tab of a splitter section, and `_group_strength_html` rendered only the
top-2 and bottom-2 per group type per timeframe - discarding most of what the
scan computed. "Occasionally look at industry/sector RS/RW" is what that access
cost produces.

**Rebuilt 2026-08-27** (plan.md Phase 0.5 item 11). The tape used to render
whatever BounceBot's last RRS pass had left behind: 10-30 minutes stale, once
31 minutes late on a flip, and its one intraday number was a 60-minute window
off a 5-day fetch that reached across the overnight gap for the whole first
hour. It now renders `ui.services.group_tape_service`'s payload - today's
completed bars only, refreshed every five minutes - and each chip carries a
**90 | 60 | 30** minute sparkline, which is the read the trader asked for
("what is actually strong over the last 30-60-90 minutes"). Left-to-right that
reads "where it has been" -> "where it is now"; the chips rank by the 30.

A window without enough completed bars is BLANK, never a zero bar: on a tape
0.0 reads as "exactly in line with SPY", which is a claim, while a blank reads
as "no answer yet", which is the truth.

The strip diffs rather than rebuilds, and its variants live in `theme.qss`
keyed on object names and a dynamic property. Both were per-widget
`setStyleSheet` calls in a full teardown-and-recreate, which is exactly the
pattern the 2026-08-21 fluidity pass measured at 1843 stalls / 1008 s blocked.
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

# Ordered longest-window-first so the sparkline reads left-to-right as
# "where it has been" -> "where it is now". Kept in step with
# `group_rrs.WINDOW_ORDER`, which a test pins.
SPARK_TIMEFRAMES = ("90", "60", "30")
#: The window the chips are ranked by - what is happening NOW.
RANK_TIMEFRAME = "30"
STRIP_HEIGHT = 76
MAX_SECTOR_CHIPS = 11
MAX_INDUSTRY_CHIPS_NARROW = 9
MAX_INDUSTRY_CHIPS_WIDE = 23
WIDE_THRESHOLD = 2000


def rotation_callout(groups: dict[str, Any]) -> tuple[str, str]:
    """The one line worth reading: what is turning up, and what is fading.

    Rotating IN = up on the 30-minute read while still down on the 90 (the
    move is this half hour's, not the morning's). FADING is the mirror.
    Returns ("", "") when the payload cannot support the call rather than
    inventing one.
    """
    fast = _rows(groups, RANK_TIMEFRAME)
    slow = {row["etf"]: row for row in _rows(groups, SPARK_TIMEFRAMES[0])}
    if not fast or not slow:
        return "", ""

    rotating_in = ""
    fading = ""
    for row in sorted(fast, key=lambda item: -(item["rrs"])):
        older = slow.get(row["etf"])
        if older is not None and row["rrs"] > 0 and older["rrs"] < 0:
            rotating_in = f"{row['etf']} {row['rrs']:+.1f} on 30, still {older['rrs']:+.1f} on 90"
            break
    for row in sorted(fast, key=lambda item: item["rrs"]):
        older = slow.get(row["etf"])
        if older is not None and row["rrs"] < 0 and older["rrs"] > 0:
            fading = f"{row['etf']} {row['rrs']:+.1f} on 30, still {older['rrs']:+.1f} on 90"
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
    """One chip per group, with its RRS across every window.

    Ranked by the 30-minute read (what is happening now) and truncated
    symmetrically: the strongest AND the weakest both matter, so a plain
    head-of-list cut would hide every short candidate.
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
        chip["rrs"] = chip["spark"].get(RANK_TIMEFRAME)
        if chip["rrs"] is None:
            # Unreachable while 30 needs the fewest bars of the three, but a
            # ranking key of None would sort-crash rather than degrade, so the
            # fallback is kept.
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


def _side(value: float | None) -> str:
    if value is None:
        return "unknown"
    return "long" if value >= 0 else "short"


def _restyle(widget: QWidget, name: str, value: str) -> None:
    """Set a dynamic property and re-polish - the cheap way to change a variant.

    A `setStyleSheet` call here would make Qt parse CSS and re-polish per
    widget, per update, which is what the fluidity pass measured. The rules
    live once in `theme.qss`.
    """
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


class Sparkline(QWidget):
    """One signed bar per window, centred on zero. A missing window is BLANK."""

    def __init__(self, spark: dict[str, float], parent=None) -> None:
        super().__init__(parent)
        self._spark: dict[str, float] = {}
        self.setFixedSize(26, 18)
        self.set_spark(spark)

    def set_spark(self, spark: dict[str, float]) -> None:
        self._spark = dict(spark or {})
        missing = [tf for tf in SPARK_TIMEFRAMES if tf not in self._spark]
        read = " · ".join(
            f"{tf} min {self._spark[tf]:+.2f}"
            for tf in SPARK_TIMEFRAMES
            if tf in self._spark
        )
        if missing:
            waiting = f"{'/'.join(missing)} min: not enough completed bars yet"
            read = f"{read} · {waiting}" if read else waiting
        self.setToolTip(read or "no read")
        self.update()

    def values(self) -> tuple[float | None, ...]:
        """The three segments in draw order; None is a blank, not a zero."""
        return tuple(self._spark.get(tf) for tf in SPARK_TIMEFRAMES)

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
                # UNKNOWN draws nothing. A zero-height bar on the zero line
                # would be indistinguishable from "exactly in line with SPY".
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
        self.setObjectName("GroupChip")
        # The variant rules live in theme.qss now, not in a per-widget
        # stylesheet. A widget that had its own sheet got a styled background
        # for free; one styled from the app sheet has to ask for it, or the
        # chips arrive on the desk with no fill and no border.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.etf = str(chip["etf"])
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._name = QLabel(self.etf)
        self._name.setObjectName("GroupChipName")
        self._value = QLabel("")
        self._value.setObjectName("GroupChipValue")
        self._spark = Sparkline(chip.get("spark") or {})
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 4, 1)
        layout.setSpacing(4)
        layout.addWidget(self._name)
        layout.addWidget(self._value)
        layout.addWidget(self._spark)
        self.update_chip(chip)

    def update_chip(self, chip: dict[str, Any]) -> None:
        """Re-render in place. Called instead of rebuilding the whole row."""
        self.etf = str(chip["etf"])
        rrs = _f(chip.get("rrs"))
        side = _side(rrs)
        if self._name.text() != self.etf:
            self._name.setText(self.etf)
        text = f"{rrs:+.1f}" if rrs is not None else "--"
        if self._value.text() != text:
            self._value.setText(text)
        _restyle(self, "side", side)
        _restyle(self._name, "side", side)
        self._spark.set_spark(chip.get("spark") or {})
        self.setToolTip(
            f"{self.etf} - click to chart it. Bars are 90 | 60 | 30 minute RRS "
            "vs SPY, today's completed bars only."
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

        # Live chips, keyed by ETF, so an update moves and re-labels them
        # instead of destroying and re-creating up to 34 widgets every five
        # minutes.
        self._sector_chips: dict[str, GroupChip] = {}
        self._industry_chips: dict[str, GroupChip] = {}
        self._captions: dict[str, QLabel] = {}
        self._service_status = ""
        self._as_of_text = ""
        self._last_groups: dict[str, Any] = {}
        self.update_groups({})

    def set_status(self, text: str) -> None:
        """The owning service's `status_text` - freshness and failures.

        Kept beside the as-of on the callout line so a stale or failed read is
        visible rather than silent; a tape that looks current when its last
        refresh failed is worse than no tape.
        """
        self._service_status = str(text or "")
        self._render_callout(self._last_groups)

    def update_groups(self, payload: Any) -> None:
        """Render a `group_tape_service` payload (or a bare groups mapping)."""
        payload = payload if isinstance(payload, dict) else {}
        groups = payload.get("group_strength")
        if not isinstance(groups, dict):
            groups = payload if any(tf in payload for tf in SPARK_TIMEFRAMES) else {}
        self._last_groups = groups
        self._as_of_text = str(payload.get("as_of_text") or "")
        status = payload.get("status")
        if status:
            self._service_status = str(status)

        wide = self.width() >= WIDE_THRESHOLD
        industry_limit = MAX_INDUSTRY_CHIPS_WIDE if wide else MAX_INDUSTRY_CHIPS_NARROW
        self._fill(
            self.sector_layout,
            build_chips(groups, "sectors", MAX_SECTOR_CHIPS),
            "SECTORS",
            self._sector_chips,
        )
        self._fill(
            self.industry_layout,
            build_chips(groups, "industries", industry_limit),
            "INDUSTRIES",
            self._industry_chips,
        )
        self._render_callout(groups)

    # ------------------------------------------------------------------ paint
    def _render_callout(self, groups: dict[str, Any]) -> None:
        rotating_in, fading = rotation_callout(groups)
        bits = []
        if rotating_in:
            bits.append(f"ROTATING IN: {rotating_in}")
        if fading:
            bits.append(f"FADING: {fading}")
        if not bits and groups:
            bits.append("No clean rotation split right now.")
        if not bits and not groups:
            bits.append(
                "Waiting for the first group read - 90 | 60 | 30 minute RRS vs "
                "SPY fills in once today's session has printed enough bars."
            )
        freshness = " · ".join(
            part
            for part in (
                f"as of {self._as_of_text}" if self._as_of_text else "",
                self._service_status,
            )
            if part
        )
        if freshness:
            bits.append(freshness)
        text = "   ·   ".join(bits)
        if self.callout.text() != text:
            self.callout.setText(text)

    def _fill(self, layout, chips, caption: str, cache: dict[str, GroupChip]) -> None:
        """Diff the row: reuse every chip that survives, move it, relabel it.

        The old implementation deleted and re-created every chip on every
        payload. With 34 chips carrying a stylesheet each, that is the exact
        shape the 2026-08-21 fluidity pass measured and fixed elsewhere.
        """
        wanted = [chip["etf"] for chip in chips]
        for etf in list(cache):
            if etf not in wanted:
                widget = cache.pop(etf)
                # Detach now, free later: deleteLater alone leaves the old chip
                # parented until the event loop turns, so a stale group can
                # still be found (and clicked) after the tape has moved on.
                widget.setParent(None)
                widget.deleteLater()

        # Taken items are held until this returns; taking a widget out of a
        # layout does not delete it, and re-adding below restores management.
        taken = []
        while layout.count():
            taken.append(layout.takeAt(0))

        label = self._captions.get(caption)
        if label is None:
            label = QLabel(caption)
            label.setObjectName("MutedLabel")
            label.setMinimumWidth(74)
            self._captions[caption] = label
        layout.addWidget(label)
        label.setVisible(True)

        for chip in chips:
            widget = cache.get(chip["etf"])
            if widget is None:
                widget = GroupChip(chip)
                widget.activated.connect(self.symbolActivated)
                cache[chip["etf"]] = widget
            else:
                widget.update_chip(chip)
            layout.addWidget(widget)
            widget.setVisible(True)
        layout.addStretch(1)
        taken.clear()
