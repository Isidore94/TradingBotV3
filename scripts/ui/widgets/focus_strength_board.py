"""Compact strongest/weakest board that pins the trader's Focus names first.

Sits beside the Alert Center's tab stack, which is wider than the alert feed
needs. Zero new data: it renders the same `rrsSnapshotChanged` payload the
RS/RW Board tab and the RRS snapshot already receive - no new service, thread,
timer, or IB request.

The deep read stays on the RS/RW Board tab. This is the peripheral-vision
version: which of MY names are leading or lagging right now, and how does that
compare to the field. Focus names ranking against their own thesis (a long
pick in relative weakness) lead the lane, because that disagreement is the
most decision-relevant thing the board can say.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.models.focus_strength import StrengthBoard, StrengthRow, build_strength_board
from ui.timer_utils import SignalCoalescer

# Narrow enough that the tab stack and this board still both fit inside the
# alert column's 360px floor, wide enough for "SYMBOL +1.23 #4 IND · m5"
# without wrapping.
MIN_BOARD_WIDTH = 170


class FocusStrengthBoard(QWidget):
    """Focus picks pinned above the strongest/weakest field."""

    symbolActivated = Signal(str, str)
    reviewAllRequested = Signal()  # walk every Focus pick through the chart

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._payload: dict[str, Any] = {}
        self._focus_service = None
        self._render_coalescer = SignalCoalescer(lambda: self._render(), parent=self)

        self.title_label = QLabel("Strength")
        self.title_label.setObjectName("SectionTitle")
        self.meta_label = QLabel("--:--")
        self.meta_label.setObjectName("MutedLabel")
        # One-click chart walkthrough of every Focus pick (2026-07-31 user
        # request: review Focus picks from the desk itself).
        self.review_button = QPushButton("Review ▶")
        self.review_button.setToolTip(
            "Queue every Focus pick (Swing + M5) onto the review chart - "
            "walk them one by one with the usual buttons."
        )
        self.review_button.clicked.connect(self.reviewAllRequested)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.review_button)
        header.addWidget(self.meta_label)

        self.board = QTextBrowser()
        self.board.setOpenLinks(False)
        self.board.setOpenExternalLinks(False)
        self.board.anchorClicked.connect(self._on_anchor_clicked)
        # This column is narrow by design. A horizontal scrollbar here would
        # hide the RRS numbers behind a drag, which defeats a glance board.
        self.board.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.board.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.board, 1)

        self.apply_scaled_metrics()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.board.setHtml(_empty_html())

    def apply_scaled_metrics(self) -> None:
        """Board floor at the current UI scale (see ui.theme.px)."""
        self.setMinimumWidth(theme.px(MIN_BOARD_WIDTH))

    # ------------------------------------------------------------------
    def set_focus_service(self, service) -> None:
        """Attach the Focus store and re-render on every membership change.

        COALESCED (2026-08-31): `_render` rebuilds the whole board as an HTML
        document and hands it to `setHtml`, which re-parses and re-lays it out.
        One membership change is worth that; the 45 the DESK drain made that
        morning are one board, not 45.
        """
        self._focus_service = service
        if service is not None:
            service.focusChanged.connect(self._render_coalescer.request)
        self._render()

    def flush_pending_refresh(self) -> None:
        """Run an owed coalesced render now. The seam the tests drive."""
        self._render_coalescer.flush()

    def update_snapshot(self, payload: Any) -> None:
        self._payload = payload if isinstance(payload, dict) else {}
        self._render()

    def current_board(self) -> StrengthBoard:
        """The rendered model - the seam the tests read."""
        return build_strength_board(self._payload, self._focus_by_category())

    # ------------------------------------------------------------------
    def _focus_by_category(self) -> dict[str, dict[str, list[str]]]:
        if self._focus_service is None:
            return {}
        try:
            return self._focus_service.all_focus_by_category()
        except Exception:
            # A board is decoration on top of a decision surface; a store hiccup
            # must never take the alert column down with it.
            return {}

    def _render(self) -> None:
        board = self.current_board()
        self.meta_label.setText(_meta_text(board))
        self.board.setHtml(_board_html(board))

    def _on_anchor_clicked(self, url) -> None:
        if url.scheme().lower() != "snapshot":
            return
        symbol = url.path().strip("/").upper()
        side = url.host().upper()
        if symbol:
            self.symbolActivated.emit(symbol, side if side in {"LONG", "SHORT"} else "")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _meta_text(board: StrengthBoard) -> str:
    stamp = _stamp(board.timestamp)
    return f"{board.timeframe} {stamp}".strip() if board.timeframe else stamp


def _board_html(board: StrengthBoard) -> str:
    if board.is_empty and not board.unranked_focus:
        return _empty_html()

    head_c = theme.color("text_secondary")
    body_c = theme.color("text_primary")
    panel_c = theme.color("bg_panel")
    parts = [
        f"<html><body style='color:{body_c}; background:{panel_c}; font-size:9pt'>"
    ]

    parts.append(_lane_title("Your Focus", head_c))
    if board.focus:
        parts.append("<table width='100%' cellspacing='0' cellpadding='1'>")
        for row in board.focus:
            parts.append(_focus_row_html(row))
        parts.append("</table>")
    else:
        parts.append(
            f"<div style='color:{head_c}; margin-left:2px'>No Focus name is ranked yet.</div>"
        )
    if board.unranked_focus:
        # Honest gap: not ranked is not the same as ranked poorly.
        names = ", ".join(board.unranked_focus[:8])
        more = f" +{len(board.unranked_focus) - 8}" if len(board.unranked_focus) > 8 else ""
        parts.append(
            f"<div style='color:{head_c}; margin:3px 0 0 2px'>Unranked: {_esc(names)}{more}</div>"
        )

    parts.append(_lane_title("Field", head_c))
    parts.append(_field_html(board))
    parts.append("</body></html>")
    return "".join(parts)


def _lane_title(text: str, color: str) -> str:
    return (
        f"<div style='color:{color}; font-weight:600; letter-spacing:0.5px; "
        f"margin:6px 0 2px 0'>{_esc(text.upper())}</div>"
    )


# Swing and M5 are independent memberships, so the lane says which one it is
# rather than letting a single star imply "focused" without a horizon.
_CATEGORY_SHORT = {"swing": "sw", "m5": "m5", "both": "sw+m5"}
# Abbreviated so the whole line clears the board's narrow minimum width. The
# model keeps the long form in `rank_text()` for wider surfaces.
_SCOPE_SHORT = {"SPY": "SPY", "Sector": "SEC", "Industry": "IND"}


def _focus_row_html(row: StrengthRow) -> str:
    """One line per pick: marker, symbol, RRS, and where it ranks.

    Deliberately single-line - the lane shares a short column with the field
    below it, and a two-line row pushed the field off the board entirely.
    """
    color = theme.color("long" if row.side == "RS" else "short")
    head_c = theme.color("text_secondary")
    if row.aligned:
        marker = f"<span style='color:{theme.color('favorite')}'>&#9733;</span>"
        note_c = head_c
    else:
        # The long pick sitting in relative weakness. The glyph carries it, so
        # the callout survives for a trader who cannot read the colour.
        marker = f"<span style='color:{theme.color('caution')}'>&#9888;</span>"
        note_c = theme.color("caution")
    category = _CATEGORY_SHORT.get(row.focus_category, row.focus_category)
    note = f"#{row.rank} {_SCOPE_SHORT.get(row.scope, row.scope)}"
    if category:
        note += f" · {category}"
    side = row.focus_side or ("long" if row.side == "RS" else "short")
    return (
        "<tr>"
        f"<td style='font-weight:600; white-space:nowrap'>{marker} "
        f"{_symbol_link(row.symbol, side, color)}</td>"
        f"<td align='right' style='color:{color}; white-space:nowrap'>{row.rrs:+.2f}</td>"
        f"<td align='right' style='color:{note_c}; white-space:nowrap; padding-left:6px'>"
        f"{_esc(note)}</td>"
        "</tr>"
    )


def _field_html(board: StrengthBoard) -> str:
    long_c = theme.color("long")
    short_c = theme.color("short")
    parts = ["<table width='100%' cellspacing='0' cellpadding='1'>"]
    parts.append(
        f"<tr><th align='left' style='color:{long_c}'>Strong</th>"
        f"<th align='right' style='color:{long_c}'></th>"
        f"<th align='left' style='color:{short_c}; padding-left:8px'>Weak</th>"
        f"<th align='right' style='color:{short_c}'></th></tr>"
    )
    for index in range(max(len(board.strong), len(board.weak), 1)):
        strong = board.strong[index] if index < len(board.strong) else None
        weak = board.weak[index] if index < len(board.weak) else None
        parts.append("<tr>")
        parts.append(_field_cell(strong, long_c, "long"))
        parts.append(_field_number(strong, long_c))
        parts.append(_field_cell(weak, short_c, "short", left_pad=True))
        parts.append(_field_number(weak, short_c))
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _field_cell(row: StrengthRow | None, color: str, side: str, *, left_pad: bool = False) -> str:
    padding = "padding-left:8px;" if left_pad else ""
    link = _symbol_link(row.symbol, side, color) if row is not None else ""
    return f"<td style='{padding} font-weight:600; white-space:nowrap'>{link}</td>"


def _field_number(row: StrengthRow | None, color: str) -> str:
    text = f"{row.rrs:+.2f}" if row is not None else ""
    return f"<td align='right' style='color:{color}; white-space:nowrap'>{text}</td>"


def _symbol_link(symbol: str, side: str, color: str) -> str:
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return ""
    return (
        f"<a href='snapshot://{_esc(str(side or '').lower())}/{_esc(symbol)}' "
        f"style='color:{color}; text-decoration:none'>{_esc(symbol)}</a>"
    )


def _empty_html() -> str:
    return (
        f"<body style='color:{theme.color('text_secondary')}; font-size:9pt'>"
        "No relative-strength read yet. Focus names appear here as the sweep ranks them."
        "</body>"
    )


def _stamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%H:%M:%S")
    return str(value or "").strip() or "--:--"


def _esc(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
