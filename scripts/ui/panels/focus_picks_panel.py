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
from pick_feedback import latest_like_origins
from ui import theme
from ui.models.bounce import BounceAlert
from ui.models.rrs import rrs_rows
from ui.services.focus_service import FocusService
from ui.widgets.flow_layout import FlowLayout
from ui.widgets.section_header import SectionHeader


class FocusPicksPanel(QFrame):
    """Focus picks in two categories, each with editable Longs/Shorts lists.

    SWING (top, the headline bucket): anything that remotely looks good for a
    multi-day hold. Synced into the swing watchlists so every master scan
    covers it, and graded 1/3/5/10 sessions forward so the bot learns what
    the trader likes. M5 (below): day-trade names synced into the intraday
    longs/shorts watchlists that BounceBot sweeps.
    """

    statusChanged = Signal(str)

    def __init__(self, focus_service: FocusService, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.service = focus_service
        self._bounce_state: dict[str, dict[str, str]] = {}
        self._rrs_state: dict[str, dict[str, str]] = {}

        self.editors: list[FocusSideEditor] = []
        swing_section = self._build_category_section(
            "Swing Focus",
            "swing",
            "Multi-day picks the bot learns from - synced into the swing watchlists, graded 1/3/5/10 sessions.",
            accent=theme.color("favorite"),
        )
        m5_section = self._build_category_section(
            "M5 / Day-Trade Focus",
            "m5",
            "Intraday picks for the DT patterns - synced into longs/shorts.txt for BounceBot M5 sweeps.",
        )

        self.snapshot_status_label = QLabel("")
        self.snapshot_status_label.setObjectName("MutedLabel")

        category_splitter = QSplitter(Qt.Orientation.Vertical)
        category_splitter.addWidget(swing_section)
        category_splitter.addWidget(m5_section)
        category_splitter.setSizes([550, 450])

        header = SectionHeader(
            "Focus Picks",
            "Handpicked names in two buckets: Swing (multi-day, tracker-graded) and M5 (day-trade). "
            "Adds sync into the matching shared watchlists; removing only un-injects what Focus Picks added.",
        )
        snapshot_button = QPushButton("Snapshot Today")
        snapshot_button.clicked.connect(lambda: self.snapshot_today(force=True))
        header.add_action(snapshot_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addWidget(self.snapshot_status_label)
        layout.addWidget(category_splitter, 1)

        # One signal rebuilds all editors (covers edits from anywhere, incl. the
        # like buttons on the Alert Center / setups table), and force-merges the
        # day's snapshot so a mid-day like still lands in today's cohort.
        self.service.focusChanged.connect(self._on_focus_changed)
        self.snapshot_today(force=False, emit_status=False)

    def _build_category_section(self, title: str, category: str, hint: str, accent: str | None = None) -> QWidget:
        long_editor = FocusSideEditor(
            f"{title} - Longs", "long", category, self.service, self._live_state_for, tone="long"
        )
        short_editor = FocusSideEditor(
            f"{title} - Shorts", "short", category, self.service, self._live_state_for, tone="short"
        )
        setattr(self, f"{category}_long_editor", long_editor)
        setattr(self, f"{category}_short_editor", short_editor)
        for editor in (long_editor, short_editor):
            editor.statusChanged.connect(self.statusChanged)
            self.editors.append(editor)

        splitter = QSplitter()
        splitter.addWidget(long_editor)
        splitter.addWidget(short_editor)
        splitter.setSizes([500, 500])

        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")
        if accent:
            title_label.setStyleSheet(f"color: {accent}; font-weight: 700;")
        hint_label = QLabel(hint)
        hint_label.setObjectName("MutedLabel")
        hint_label.setWordWrap(True)

        section = QFrame()
        section.setObjectName("Panel")
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(8, 8, 8, 8)
        section_layout.setSpacing(4)
        section_layout.addWidget(title_label)
        section_layout.addWidget(hint_label)
        section_layout.addWidget(splitter, 1)
        return section

    def _on_focus_changed(self) -> None:
        self._refresh_all()
        # Merge new names into today's cohort immediately (no-op on removals;
        # snapshots only ever add rows for the date).
        self.snapshot_today(force=True, emit_status=False)

    def _refresh_all(self) -> None:
        for editor in self.editors:
            editor.refresh()

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
        self._refresh_all()

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
            focus_maps_by_category=self.service.all_focus_by_category(),
            like_origins=latest_like_origins(),
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


class FocusSideEditor(QFrame):
    statusChanged = Signal(str)

    def __init__(
        self,
        title: str,
        side: str,
        category: str,
        focus_service: FocusService,
        live_state_for,
        *,
        tone: str,
    ) -> None:
        super().__init__()
        self.setObjectName("Panel")
        self.side = side
        self.category = category
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
        symbols = self.service.focus_symbols(self.side, self.category)
        for symbol in symbols:
            chip = FocusStatusChip(symbol, tone=self.tone, state=self.live_state_for(symbol))
            chip.removed.connect(self._remove)
            self.chip_flow.addWidget(chip)
        self.count_label.setText(str(len(symbols)))

    def add_from_input(self) -> None:
        added = self.service.add_many(self.add_input.text(), self.side, self.category, origin="manual")
        self.add_input.clear()
        if added:
            self._set_status(f"Added {', '.join(added)}.")

    def paste(self) -> None:
        added = self.service.add_many(
            QApplication.clipboard().text(), self.side, self.category, origin="manual"
        )
        self._set_status(f"Pasted {len(added)} symbol(s)." if added else "Nothing new to paste.")

    def copy(self) -> None:
        symbols = self.service.focus_symbols(self.side, self.category)
        QApplication.clipboard().setText(", ".join(symbols))
        self._set_status(f"Copied {len(symbols)} symbol(s).")

    def clear_all(self) -> None:
        removed = self.service.clear(self.side, self.category)
        self._set_status(f"Cleared {removed} symbol(s).")

    def _remove(self, symbol: str) -> None:
        if self.service.remove(symbol, self.side, self.category):
            self._set_status(f"Removed {symbol}.")

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"{self.category.upper()} focus {self.side}s: {message}")


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
