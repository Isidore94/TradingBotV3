from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from human_focus_tracking import snapshot_human_focus_picks
from ui.services.focus_service import FocusService
from ui.widgets.flow_layout import FlowLayout
from ui.widgets.section_header import SectionHeader
from ui.widgets.symbol_chip import SymbolChip


class FocusPicksPanel(QFrame):
    """Two side-by-side editable focus lists (Longs / Shorts) of handpicked names."""

    statusChanged = Signal(str)

    def __init__(self, focus_service: FocusService, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.service = focus_service

        self.long_editor = FocusSideEditor("Focus Longs", "long", focus_service, tone="long")
        self.short_editor = FocusSideEditor("Focus Shorts", "short", focus_service, tone="short")
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


class FocusSideEditor(QFrame):
    statusChanged = Signal(str)

    def __init__(self, title: str, side: str, focus_service: FocusService, *, tone: str) -> None:
        super().__init__()
        self.setObjectName("Panel")
        self.side = side
        self.service = focus_service
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
            chip = SymbolChip(symbol, tone=self.tone)
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
