from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from human_focus_tracking import snapshot_human_focus_picks
from ui import theme
from ui.models.bounce import BounceAlert
from ui.models.rrs import rrs_rows
from ui.services.focus_service import FocusService
from ui.widgets.flow_layout import FlowLayout
from ui.widgets.section_header import SectionHeader


class FocusPicksPanel(QFrame):
    """Two side-by-side editable focus lists (Longs / Shorts) of handpicked names."""

    statusChanged = Signal(str)

    def __init__(self, focus_service: FocusService, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.service = focus_service
        self._bounce_state: dict[str, dict[str, str]] = {}
        self._rrs_state: dict[str, dict[str, str]] = {}

        self.long_editor = FocusSideEditor("Focus Longs", "long", focus_service, self._live_state_for, tone="long")
        self.short_editor = FocusSideEditor("Focus Shorts", "short", focus_service, self._live_state_for, tone="short")
        self.long_editor.statusChanged.connect(self.statusChanged)
        self.short_editor.statusChanged.connect(self.statusChanged)
        self.snapshot_status_label = QLabel("")
        self.snapshot_status_label.setObjectName("MutedLabel")

        splitter = QSplitter()
        splitter.addWidget(self.long_editor)
        splitter.addWidget(self.short_editor)
        splitter.setSizes([500, 500])

        header = SectionHeader(
            "Focus Picks",
            "Handpicked daily longs/shorts the bot watches closely. Adds sync into the shared "
            "longs/shorts watchlists; removing only un-injects what Focus Picks added.",
        )
        snapshot_button = QPushButton("Snapshot Today")
        snapshot_button.clicked.connect(lambda: self.snapshot_today(force=True))
        header.add_action(snapshot_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addWidget(self.snapshot_status_label)
        layout.addWidget(splitter, 1)

        # One signal rebuilds both sides (covers edits from anywhere, incl. Step 3).
        self.service.focusChanged.connect(self._refresh_all)
        self.snapshot_today(force=False, emit_status=False)

    def _refresh_all(self) -> None:
        self.long_editor.refresh()
        self.short_editor.refresh()

    def record_bounce_alert(self, alert: BounceAlert) -> None:
        """Surface BounceBot alerts directly on matching Focus Picks chips."""
        if alert.is_d1 or not alert.symbol or not self.service.is_focus(alert.symbol):
            return
        symbol = alert.symbol.upper()
        detail = " ".join(part for part in (alert.side, alert.timeframe, alert.trigger) if part).strip()
        self._bounce_state[symbol] = {
            "tone": "long" if alert.side == "LONG" else "short" if alert.side == "SHORT" else "favorite",
            "text": f"{alert.time_text} bounce" + (f" - {detail}" if detail else ""),
        }
        self._refresh_symbol(symbol)

    def record_rrs_snapshot(self, payload: Any) -> None:
        """Mark focus longs that are RS and focus shorts that are RW."""
        focus = self.service.all_focus()
        aligned: dict[str, dict[str, str]] = {}
        for scope in ("SPY", "Sector", "Industry"):
            for row in rrs_rows(payload, scope):
                symbol = row.symbol.upper()
                if row.side == "RS" and symbol in focus.get("long", []):
                    aligned[symbol] = {"tone": "long", "text": f"RS {row.rrs:+.2f} vs {scope}"}
                elif row.side == "RW" and symbol in focus.get("short", []):
                    aligned[symbol] = {"tone": "short", "text": f"RW {row.rrs:+.2f} vs {scope}"}
        self._rrs_state = aligned
        self._refresh_all()

    def snapshot_today(self, *, force: bool, emit_status: bool = True) -> None:
        if not getattr(self.service.store, "uses_default_paths", lambda: False)():
            self.snapshot_status_label.setText("Snapshot: custom focus store")
            return
        result = snapshot_human_focus_picks(
            focus_map=self.service.all_focus(),
            force=force,
        )
        trade_date = result.get("trade_date", "today")
        added = int(result.get("added") or 0)
        total = int(result.get("total_for_date") or 0)
        if result.get("snapshotted"):
            message = f"Snapshot {trade_date}: {total} pick(s), {added} new."
        else:
            message = f"Snapshot {trade_date}: already captured ({total} pick(s))."
        self.snapshot_status_label.setText(message)
        if emit_status:
            self.statusChanged.emit(message)

    def _live_state_for(self, symbol: str) -> dict[str, dict[str, str]]:
        symbol = str(symbol or "").upper()
        return {
            "bounce": self._bounce_state.get(symbol, {}),
            "rrs": self._rrs_state.get(symbol, {}),
        }

    def _refresh_symbol(self, symbol: str) -> None:
        side = self.service.focus_side(symbol)
        if side == "LONG":
            self.long_editor.refresh()
        elif side == "SHORT":
            self.short_editor.refresh()
        else:
            self._refresh_all()


class FocusSideEditor(QFrame):
    statusChanged = Signal(str)

    def __init__(
        self,
        title: str,
        side: str,
        focus_service: FocusService,
        live_state_for,
        *,
        tone: str,
    ) -> None:
        super().__init__()
        self.setObjectName("Panel")
        self.side = side
        self.service = focus_service
        self.live_state_for = live_state_for
        self.tone = tone

        self.title_label = QLabel(title)
        self.title_label.setObjectName("SectionTitle")
        self.count_label = QLabel("0")
        self.count_label.setObjectName("MutedLabel")

        self.add_input = QLineEdit()
        self.add_input.setPlaceholderText("Add ticker(s)")
        self.add_input.returnPressed.connect(self.add_from_input)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_from_input)

        self.chip_host = QWidget()
        self.chip_flow = FlowLayout(self.chip_host, margin=2, spacing=6)
        chip_scroll = QScrollArea()
        chip_scroll.setWidgetResizable(True)
        chip_scroll.setWidget(self.chip_host)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)

        self._build_layout(add_button, chip_scroll)
        self.refresh()

    def _build_layout(self, add_button, chip_scroll) -> None:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title_label)
        header.addWidget(self.count_label)
        header.addStretch(1)

        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        add_row.setSpacing(6)
        add_row.addWidget(self.add_input, 1)
        add_row.addWidget(add_button)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)
        for label, slot in (
            ("Paste", self.paste),
            ("Copy", self.copy),
            ("Clear All", self.clear_all),
            ("Refresh", self.refresh),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            action_row.addWidget(button)
        action_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addLayout(add_row)
        layout.addLayout(action_row)
        layout.addWidget(chip_scroll, 1)
        layout.addWidget(self.status_label)

    def refresh(self) -> None:
        while self.chip_flow.count():
            item = self.chip_flow.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        symbols = self.service.focus_symbols(self.side)
        for symbol in symbols:
            chip = FocusStatusChip(symbol, tone=self.tone, state=self.live_state_for(symbol))
            chip.removed.connect(self._remove)
            self.chip_flow.addWidget(chip)
        self.count_label.setText(str(len(symbols)))

    def add_from_input(self) -> None:
        added = self.service.add_many(self.add_input.text(), self.side)
        self.add_input.clear()
        if added:
            self._set_status(f"Added {', '.join(added)}.")

    def paste(self) -> None:
        added = self.service.add_many(QApplication.clipboard().text(), self.side)
        self._set_status(f"Pasted {len(added)} symbol(s)." if added else "Nothing new to paste.")

    def copy(self) -> None:
        symbols = self.service.focus_symbols(self.side)
        QApplication.clipboard().setText(", ".join(symbols))
        self._set_status(f"Copied {len(symbols)} symbol(s).")

    def clear_all(self) -> None:
        removed = self.service.clear(self.side)
        self._set_status(f"Cleared {removed} symbol(s).")

    def _remove(self, symbol: str) -> None:
        if self.service.remove(symbol, self.side):
            self._set_status(f"Removed {symbol}.")

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"Focus {self.side}s: {message}")


class FocusStatusChip(QFrame):
    """Ticker chip with optional live BounceBot/RRS status."""

    removed = Signal(str)

    def __init__(self, symbol: str, *, tone: str, state: dict[str, dict[str, str]], parent=None) -> None:
        super().__init__(parent)
        self.symbol = symbol
        self.setObjectName("FocusStatusChip")

        bounce = state.get("bounce") or {}
        rrs = state.get("rrs") or {}
        has_bounce = bool(bounce.get("text"))
        has_rrs = bool(rrs.get("text"))
        accent_tone = str(bounce.get("tone") or rrs.get("tone") or tone)
        accent = theme.color("favorite" if has_bounce else accent_tone)
        side_color = theme.color(tone)
        bg_alpha = 0.20 if has_bounce else 0.14 if has_rrs else 0.10
        border_alpha = 0.78 if has_bounce else 0.55
        self.setStyleSheet(
            f"""
            QFrame#FocusStatusChip {{
                background: {theme.with_alpha(accent, bg_alpha)};
                border: 1px solid {theme.with_alpha(accent, border_alpha)};
                border-radius: 8px;
            }}
            QFrame#FocusStatusChip QLabel {{ background: transparent; }}
            QFrame#FocusStatusChip QToolButton {{
                color: {side_color}; border: none; background: transparent; font-weight: 700; padding: 0 2px;
            }}
            QFrame#FocusStatusChip QToolButton:hover {{ color: {theme.color('text_primary')}; }}
            """
        )

        title = QLabel(symbol)
        title.setStyleSheet(f"color: {side_color}; font-weight: 700;")
        remove_button = QToolButton()
        remove_button.setText("x")
        remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_button.setToolTip(f"Remove {symbol}")
        remove_button.clicked.connect(lambda: self.removed.emit(self.symbol))

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        top.addWidget(title)
        if has_bounce:
            flag = QLabel("BOUNCE")
            flag.setStyleSheet(f"color: {theme.color('favorite')}; font-weight: 700;")
            top.addWidget(flag)
        elif has_rrs:
            flag = QLabel("RRS")
            flag.setStyleSheet(f"color: {accent}; font-weight: 700;")
            top.addWidget(flag)
        top.addWidget(remove_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 4, 5, 4)
        layout.setSpacing(2)
        layout.addLayout(top)
        for item in (bounce, rrs):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            status = QLabel(text)
            status.setObjectName("MutedLabel")
            status.setWordWrap(True)
            layout.addWidget(status)
