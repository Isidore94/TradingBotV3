from __future__ import annotations

from PySide6.QtCore import QFileSystemWatcher, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from project_paths import (
    HUMAN_FOCUS_DAILY_PICKS_FILE,
    MASTER_AVWAP_MARKET_PREP_FILE,
    MASTER_AVWAP_MARKET_PREP_REPORT_FILE,
    open_path_in_file_manager,
)
from ui.services.market_prep_feed import (
    MARKET_PREP_SECTION_DEFINITIONS,
    human_focus_pick_count,
    human_focus_pick_text,
    load_human_focus_daily_picks,
    load_market_prep_payload,
    load_market_prep_report,
    market_prep_sections,
    section_copy_text,
    section_symbol_count,
)
from ui.widgets.section_header import SectionHeader


class MasterMarketPrepPanel(QFrame):
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.section_widgets: dict[str, QTextEdit] = {}
        self.section_labels: dict[str, QLabel] = {}
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("Run Master AVWAP to populate market prep output.")
        self.human_picks_label = QLabel("Today's Human Picks (0)")
        self.human_picks_label.setObjectName("SectionTitle")
        self.human_picks_text = QTextEdit()
        self.human_picks_text.setReadOnly(True)
        self.human_picks_text.setMaximumHeight(84)
        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self._build_layout()
        self._configure_watcher()
        self.refresh()

    def _build_layout(self) -> None:
        header = SectionHeader(
            "Master AVWAP Market Prep",
            "Copy focused lists from the latest Master AVWAP scan and inspect the full report.",
        )
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        copy_all_button = QPushButton("Copy All Sections")
        copy_all_button.clicked.connect(self.copy_all_sections)
        open_button = QPushButton("Open Reports Folder")
        open_button.clicked.connect(lambda: open_path_in_file_manager(MASTER_AVWAP_MARKET_PREP_REPORT_FILE.parent))
        header.add_action(refresh_button)
        header.add_action(copy_all_button)
        header.add_action(open_button)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for index, definition in enumerate(MARKET_PREP_SECTION_DEFINITIONS):
            section = self._build_section_card(definition)
            grid.addWidget(section, index // 2, index % 2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_host)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addWidget(scroll, 3)
        body.addWidget(self.report_text, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(header)
        layout.addWidget(self.status_label)
        layout.addWidget(self._build_human_picks_panel())
        layout.addLayout(body, 1)

    def _build_section_card(self, definition: dict[str, str]) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Panel")
        title = QLabel(definition["title"])
        title.setObjectName("SectionTitle")
        self.section_labels[definition["id"]] = title
        text = QTextEdit()
        text.setReadOnly(True)
        text.setMinimumHeight(92)
        self.section_widgets[definition["id"]] = text
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(lambda _checked=False, section_id=definition["id"]: self.copy_section(section_id))
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(copy_button)
        layout.addLayout(top)
        layout.addWidget(text)
        return frame

    def _build_human_picks_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("StatusStrip")
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self.copy_human_picks)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.human_picks_label)
        top.addStretch(1)
        top.addWidget(copy_button)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addWidget(self.human_picks_text)
        return frame

    def _configure_watcher(self) -> None:
        self.watcher = QFileSystemWatcher(self)
        for path in (MASTER_AVWAP_MARKET_PREP_FILE, MASTER_AVWAP_MARKET_PREP_REPORT_FILE, HUMAN_FOCUS_DAILY_PICKS_FILE):
            if path.exists():
                self.watcher.addPath(str(path))
        self.watcher.fileChanged.connect(lambda _path: self.refresh())

    def refresh(self) -> None:
        payload = load_market_prep_payload()
        section_map = market_prep_sections(payload)
        total_symbols = 0
        for definition in MARKET_PREP_SECTION_DEFINITIONS:
            section = section_map.get(definition["id"], {})
            text = section_copy_text(section, definition)
            count = section_symbol_count(section, text)
            total_symbols += count
            label = self.section_labels.get(definition["id"])
            if label is not None:
                label.setText(f"{definition['title']} ({count})")
            widget = self.section_widgets.get(definition["id"])
            if widget is not None:
                widget.setPlainText(text)
        self.report_text.setPlainText(load_market_prep_report())
        generated = payload.get("generated_at") if isinstance(payload, dict) else ""
        self._refresh_human_picks(payload)
        self.status_label.setText(
            f"Generated: {generated or 'n/a'} | Focus symbols across sections: {total_symbols}"
        )
        self.statusChanged.emit(self.status_label.text())
        self._refresh_watcher_paths()

    def copy_section(self, section_id: str) -> None:
        widget = self.section_widgets.get(section_id)
        if widget is None:
            return
        text = widget.toPlainText().strip()
        if not text or text == "None":
            self.status_label.setText("No symbols in that market prep section.")
        else:
            QApplication.clipboard().setText(text)
            self.status_label.setText("Copied market prep section.")
        self.statusChanged.emit(self.status_label.text())

    def copy_all_sections(self) -> None:
        chunks: list[str] = []
        for definition in MARKET_PREP_SECTION_DEFINITIONS:
            widget = self.section_widgets.get(definition["id"])
            if widget is None:
                continue
            text = widget.toPlainText().strip()
            if text and text != "None":
                chunks.append(f"{definition['title']}: {text}")
        if not chunks:
            self.status_label.setText("No market prep symbols to copy.")
        else:
            QApplication.clipboard().setText("\n".join(chunks))
            self.status_label.setText("Copied all market prep sections.")
        self.statusChanged.emit(self.status_label.text())

    def copy_human_picks(self) -> None:
        text = self.human_picks_text.toPlainText().strip()
        if not text or text.startswith("No human focus picks"):
            self.status_label.setText("No human focus picks to copy.")
        else:
            QApplication.clipboard().setText(text)
            self.status_label.setText("Copied today's human picks.")
        self.statusChanged.emit(self.status_label.text())

    def _refresh_human_picks(self, payload: dict) -> None:
        trade_date = payload.get("run_date") if isinstance(payload, dict) else ""
        rows = load_human_focus_daily_picks(trade_date=trade_date)
        count = human_focus_pick_count(rows)
        self.human_picks_label.setText(f"Today's Human Picks ({count})")
        self.human_picks_text.setPlainText(human_focus_pick_text(rows))

    def _refresh_watcher_paths(self) -> None:
        watched = set(self.watcher.files())
        for path in (MASTER_AVWAP_MARKET_PREP_FILE, MASTER_AVWAP_MARKET_PREP_REPORT_FILE, HUMAN_FOCUS_DAILY_PICKS_FILE):
            if path.exists() and str(path) not in watched:
                self.watcher.addPath(str(path))
