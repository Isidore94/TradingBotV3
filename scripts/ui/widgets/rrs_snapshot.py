from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from ui import theme
from ui.models.rrs import rrs_rows


_SCOPES = ("SPY", "Sector", "Industry")
_GROUP_TIMEFRAMES = ("M5", "H1", "D1")


class RrsSnapshotWidget(QWidget):
    """Compact relative-strength board for BounceBot snapshots."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._payload: dict[str, Any] = {}

        self.env_label = QLabel("RRS Snapshot")
        self.env_label.setObjectName("SectionTitle")
        self.meta_label = QLabel("Connect BounceBot to stream relative-strength scans.")
        self.meta_label.setObjectName("MutedLabel")
        self.meta_label.setWordWrap(True)

        self.copy_rs_button = QPushButton("Copy All RS")
        self.copy_rs_button.clicked.connect(lambda: self._copy_side("RS"))
        self.copy_rw_button = QPushButton("Copy All RW")
        self.copy_rw_button.clicked.connect(lambda: self._copy_side("RW"))

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        top_row.addWidget(self.env_label)
        top_row.addStretch(1)
        top_row.addWidget(self.copy_rs_button)
        top_row.addWidget(self.copy_rw_button)

        self.board = QTextBrowser()
        self.board.setOpenExternalLinks(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(top_row)
        layout.addWidget(self.meta_label)
        layout.addWidget(self.board, 1)
        self.board.setHtml(_empty_html())

    _focus_service = None

    def set_focus_service(self, service) -> None:
        self._focus_service = service

    def _focus_map(self) -> dict[str, set]:
        if self._focus_service is None:
            return {"long": set(), "short": set()}
        focus = self._focus_service.all_focus()
        return {"long": set(focus.get("long", [])), "short": set(focus.get("short", []))}

    def update_snapshot(self, payload: Any) -> None:
        self._payload = payload if isinstance(payload, dict) else {}
        env = self._payload.get("market_environment_label") or self._payload.get("market_environment") or "Environment"
        self.env_label.setText(str(env))
        self.meta_label.setText(self._meta_text())
        self.board.setHtml(_board_html(self._payload, self._focus_map()))

    def _copy_side(self, side: str) -> None:
        symbols: list[str] = []
        seen: set[str] = set()
        for scope in _SCOPES:
            rows = rrs_rows(self._payload, scope)
            if side == "RS":
                scoped_rows = sorted((row for row in rows if row.side == side), key=lambda row: -row.rrs)
            else:
                scoped_rows = sorted((row for row in rows if row.side == side), key=lambda row: row.rrs)
            for row in scoped_rows:
                if row.symbol and row.symbol not in seen:
                    symbols.append(row.symbol)
                    seen.add(row.symbol)
        QApplication.clipboard().setText(", ".join(symbols))

    def _meta_text(self) -> str:
        timeframe = self._payload.get("timeframe_key") or "?"
        threshold = self._payload.get("threshold")
        threshold_text = f"{float(threshold):.2f}" if isinstance(threshold, (int, float)) else "?"
        return f"Timeframe {timeframe} - Threshold {threshold_text} - Updated {_stamp(self._payload.get('timestamp'))}"


def _board_html(payload: dict[str, Any], focus: dict[str, set] | None = None) -> str:
    if not payload:
        return _empty_html()

    focus = focus or {"long": set(), "short": set()}
    body_c = theme.color("text_primary")
    head_c = theme.color("text_secondary")
    border_c = theme.color("border")
    panel_c = theme.color("bg_panel")
    parts = [
        "<html><body",
        f" style='color:{body_c}; background:{panel_c}; font-size:9pt;'>",
    ]
    parts.append("<table width='100%' cellspacing='0' cellpadding='0'><tr>")
    for scope in _SCOPES:
        parts.append(f"<td valign='top' width='33%' style='padding-right:8px'>{_scope_html(payload, scope, focus)}</td>")
    parts.append("</tr></table>")
    parts.append(f"<div style='height:8px; border-bottom:1px solid {border_c}'></div>")
    parts.append(_group_strength_html(payload))
    parts.append(_environment_html(payload))
    parts.append(f"<p style='color:{head_c}; margin-top:8px'>RS = relative strength; RW = relative weakness.</p>")
    parts.append("</body></html>")
    return "".join(parts)


def _scope_html(payload: dict[str, Any], scope: str, focus: dict[str, set] | None = None) -> str:
    focus = focus or {"long": set(), "short": set()}
    rows = rrs_rows(payload, scope)
    strong = sorted((row for row in rows if row.side == "RS"), key=lambda row: -row.rrs)[:8]
    weak = sorted((row for row in rows if row.side == "RW"), key=lambda row: row.rrs)[:8]
    long_c = theme.color("long")
    short_c = theme.color("short")
    head_c = theme.color("text_secondary")

    title = "vs SPY" if scope == "SPY" else f"vs {scope}"
    parts = [f"<h3 style='margin:0 0 5px 0; color:{head_c}'>{_esc(title)}</h3>"]
    parts.append("<table width='100%' cellspacing='0' cellpadding='2'>")
    parts.append(
        f"<tr><th align='left' style='color:{long_c}'>Strongest</th>"
        f"<th align='right' style='color:{long_c}'>RRS</th>"
        f"<th align='left' style='color:{short_c}; padding-left:10px'>Weakest</th>"
        f"<th align='right' style='color:{short_c}'>RRS</th></tr>"
    )
    for index in range(max(len(strong), len(weak), 1)):
        rs = strong[index] if index < len(strong) else None
        rw = weak[index] if index < len(weak) else None
        parts.append("<tr>")
        parts.append(_symbol_cell(rs.symbol if rs else "", long_c, focus_aligned=bool(rs and rs.symbol in focus["long"])))
        parts.append(_number_cell(rs.rrs if rs else None, long_c))
        parts.append(_symbol_cell(rw.symbol if rw else "", short_c, left_pad=True, focus_aligned=bool(rw and rw.symbol in focus["short"])))
        parts.append(_number_cell(rw.rrs if rw else None, short_c))
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _group_strength_html(payload: dict[str, Any]) -> str:
    groups = payload.get("group_strength") if isinstance(payload.get("group_strength"), dict) else {}
    if not groups:
        return ""

    head_c = theme.color("text_secondary")
    parts = [f"<h3 style='margin:8px 0 4px 0; color:{head_c}'>Sector / Industry Tape</h3>"]
    parts.append("<table width='100%' cellspacing='0' cellpadding='2'>")
    parts.append("<tr>")
    for timeframe in _GROUP_TIMEFRAMES:
        parts.append(f"<td valign='top' width='33%'>{_group_timeframe_html(groups, timeframe)}</td>")
    parts.append("</tr></table>")
    return "".join(parts)


def _group_timeframe_html(groups: dict[str, Any], timeframe: str) -> str:
    frame = groups.get(timeframe) if isinstance(groups.get(timeframe), dict) else {}
    head_c = theme.color("text_secondary")
    if not frame:
        return f"<div style='color:{head_c}'>{_esc(timeframe)}: no read</div>"
    parts = [f"<div style='color:{head_c}; font-weight:600'>{_esc(timeframe)}</div>"]
    for label, key in (("Sectors", "sectors"), ("Industries", "industries")):
        items = [item for item in (frame.get(key) or []) if isinstance(item, dict)]
        if not items:
            continue
        ranked = sorted(items, key=lambda row: -(_f(row.get("rrs")) or 0.0))
        parts.append(f"<div style='color:{head_c}; margin-top:3px'>{_esc(label)}</div>")
        for item in ranked[:2]:
            parts.append(_group_line(item, theme.color("long")))
        for item in reversed(ranked[-2:]):
            if item not in ranked[:2]:
                parts.append(_group_line(item, theme.color("short")))
    return "".join(parts)


def _environment_html(payload: dict[str, Any]) -> str:
    highlights = payload.get("environment_highlights") if isinstance(payload.get("environment_highlights"), list) else []
    if not highlights:
        return ""

    head_c = theme.color("text_secondary")
    parts = [f"<h3 style='margin:8px 0 4px 0; color:{head_c}'>Environment Focus</h3>"]
    for section in highlights[:4]:
        if not isinstance(section, dict):
            continue
        parts.append(f"<div style='color:{head_c}; font-weight:600'>{_esc(section.get('title', 'Section'))}</div>")
        rows = section.get("rows") if isinstance(section.get("rows"), list) else []
        if not rows:
            parts.append(f"<div style='color:{head_c}; margin-left:8px'>None</div>")
            continue
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            tag = str(row.get("tag") or "")
            color = theme.color("short") if tag == "rrs_rw" else head_c if tag == "rrs_hdr" else theme.color("long")
            parts.append(f"<div style='color:{color}; margin-left:8px'>{_esc(row.get('text', ''))}</div>")
    return "".join(parts)


def _empty_html() -> str:
    return (
        f"<body style='color:{theme.color('text_secondary')}; font-size:9pt'>"
        "No relative-strength reads yet. Start scanning to rank symbols by RRS."
        "</body>"
    )


def _symbol_cell(value: str, color: str, *, left_pad: bool = False, focus_aligned: bool = False) -> str:
    padding = "padding-left:10px;" if left_pad else ""
    marker = f"<span style='color:{theme.color('favorite')}'>&#9733;</span> " if (focus_aligned and value) else ""
    return f"<td style='color:{color}; {padding} font-weight:600'>{marker}{_esc(value)}</td>"


def _number_cell(value: float | None, color: str) -> str:
    text = f"{value:+.2f}" if value is not None else ""
    return f"<td align='right' style='color:{color}; white-space:nowrap'>{text}</td>"


def _group_line(item: dict[str, Any], color: str) -> str:
    group_key = _esc(item.get("group_key", ""))
    etf = _esc(item.get("etf", ""))
    rrs = _f(item.get("rrs"))
    power = _f(item.get("power_index"))
    rrs_text = f"{rrs:+.2f}" if rrs is not None else ""
    power_text = f"{power:+.2f}" if power is not None else ""
    muted = theme.color("text_muted")
    return (
        f"<div style='color:{color}; margin-left:8px'>{group_key} "
        f"<span style='color:{muted}'>{etf}</span> {rrs_text} {power_text}</div>"
    )


def _stamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%H:%M:%S")
    text = str(value or "").strip()
    return text or datetime.now().strftime("%H:%M:%S")


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _esc(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
