from __future__ import annotations

"""Always-on entry-assist RS/RW board.

Renders the bot's ``entry_assist_board_snapshot()``: the auto regime, live
SPY pause detection, the active pullback/bounce window's ranking AS IT
STANDS (or a pause-preview ranking when SPY is pausing with no window open),
and the trailing strongest/weakest movers for both sides. Everything the
entry-assist buttons can produce, recomputed automatically - no clicks."""

from typing import Any

from PySide6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget

from ui import theme


class EntryAssistBoard(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._board: dict[str, Any] = {}

        self.title_label = QLabel("Auto RS/RW Board")
        self.title_label.setObjectName("SectionTitle")
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.title_label)
        layout.addWidget(self.view, 1)
        self.view.setHtml(_empty_html())

    def update_board(self, board: Any) -> None:
        self._board = board if isinstance(board, dict) else {}
        if not self._board:
            self.view.setHtml(_empty_html())
            self.title_label.setText("Auto RS/RW Board")
            return
        env = self._board.get("env_label") or self._board.get("env_key") or "?"
        stamp = self._board.get("bar_time") or ""
        self.title_label.setText(f"Auto RS/RW Board - {env} (bar {stamp})")
        self.view.setHtml(_board_html(self._board))


def _board_html(board: dict[str, Any]) -> str:
    body_c = theme.color("text_primary")
    muted = theme.color("text_secondary")
    long_c = theme.color("long")
    short_c = theme.color("short")
    favorite_c = theme.color("favorite")

    parts = [f"<body style='color:{body_c}; font-size:9pt'>"]

    pause = board.get("pause") if isinstance(board.get("pause"), dict) else {}
    trend_side = str(pause.get("trend_side") or "long")
    if pause.get("detected"):
        against = "pullback" if trend_side == "long" else "bounce"
        parts.append(
            f"<div><b style='color:{favorite_c}'>SPY {against.upper()} DETECTED</b> "
            f"<span style='color:{muted}'>since {_esc(pause.get('since'))} - watching which "
            f"{'longs hold up (RS)' if trend_side == 'long' else 'shorts stay weak (RW)'}</span></div>"
        )
    else:
        parts.append(
            f"<div style='color:{muted}'>Tape running with the trend - no SPY pause detected right now.</div>"
        )

    window = board.get("window") if isinstance(board.get("window"), dict) else {}
    if window.get("active"):
        sides = "/".join(window.get("sides") or [])
        spy_pct = window.get("spy_pct")
        spy_text = f"SPY {spy_pct:+.2f}%" if isinstance(spy_pct, (int, float)) else "SPY n/a"
        parts.append(
            f"<h3 style='margin:8px 0 2px 0; color:{favorite_c}'>Live window ({_esc(sides)}) "
            f"since {_esc(window.get('started'))} [{_esc(window.get('source'))}] - {spy_text}</h3>"
        )
        rankings = window.get("rankings") if isinstance(window.get("rankings"), dict) else {}
        for side, rows in rankings.items():
            color = long_c if side == "long" else short_c
            held = "holding strongest so far" if side == "long" else "staying weakest so far"
            parts.append(f"<div style='color:{color}'><b>{_esc(side.upper())}</b> {held}:</div>")
            parts.append(_ranked_rows_html(rows, color, pct_key="window_pct"))
    elif isinstance(board.get("pause_preview"), dict):
        preview = board["pause_preview"]
        side = str(preview.get("side") or "long")
        color = long_c if side == "long" else short_c
        spy_pct = preview.get("spy_pct")
        spy_text = f"SPY {spy_pct:+.2f}%" if isinstance(spy_pct, (int, float)) else "SPY n/a"
        parts.append(
            f"<h3 style='margin:8px 0 2px 0; color:{favorite_c}'>Pause preview ({_esc(side)}) "
            f"since {_esc(preview.get('since'))} - {spy_text}</h3>"
        )
        parts.append(_ranked_rows_html(preview.get("rows") or [], color, pct_key="window_pct"))

    movers = board.get("movers") if isinstance(board.get("movers"), dict) else {}
    minutes = board.get("movers_minutes") or 30
    parts.append("<table width='100%' cellspacing='0' cellpadding='2'><tr>")
    for side, color, title in (
        ("long", long_c, f"Strongest {minutes}m (longs)"),
        ("short", short_c, f"Weakest {minutes}m (shorts)"),
    ):
        parts.append("<td valign='top' width='50%'>")
        parts.append(f"<h3 style='margin:8px 0 2px 0; color:{color}'>{_esc(title)}</h3>")
        parts.append(_ranked_rows_html(movers.get(side) or [], color, pct_key="change_pct"))
        parts.append("</td>")
    parts.append("</tr></table>")

    parts.append(
        f"<p style='color:{muted}; margin-top:6px'>Auto-refreshed every minute from cached bars; "
        f"windows also open/close automatically via SPY pause detection in auto mode.</p>"
    )
    parts.append("</body>")
    return "".join(parts)


def _ranked_rows_html(rows: list, color: str, *, pct_key: str) -> str:
    if not rows:
        return f"<div style='color:{theme.color('text_muted')}; margin-left:8px'>none with fresh bars</div>"
    parts = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        symbol = _esc(row.get("symbol"))
        pct = row.get(pct_key)
        excess = row.get("excess")
        pct_text = f"{pct:+.2f}%" if isinstance(pct, (int, float)) else ""
        excess_text = f" (x{excess:+.2f})" if isinstance(excess, (int, float)) else ""
        parts.append(
            f"<div style='margin-left:8px'><b style='color:{color}'>{symbol}</b> "
            f"{pct_text}<span style='color:{theme.color('text_muted')}'>{excess_text}</span></div>"
        )
    return "".join(parts)


def _empty_html() -> str:
    return (
        f"<body style='color:{theme.color('text_secondary')}; font-size:9pt'>"
        "Waiting for the bot: the board fills automatically once SPY bars are cached "
        "(regime, pause detection, live window rankings, strongest/weakest movers)."
        "</body>"
    )


def _esc(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
