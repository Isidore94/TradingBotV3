from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from datetime import datetime

from project_paths import (
    AUTO_LONGS_FILE,
    AUTO_SHORTS_FILE,
    LONGS_FILE,
    SHORTS_FILE,
    SWING_LONGS_FILE,
    SWING_SHORTS_FILE,
    open_path_in_file_manager,
)
from watchlist_utils import extract_watchlist_symbols
from ui.widgets.section_header import SectionHeader


class _SymbolTextEdit(QPlainTextEdit):
    """Watchlist text area; double-clicking a symbol line opens the D1+M5
    snapshot popup (one symbol per line, so the whole line is the ticker)."""

    symbolActivated = Signal(str)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mouseDoubleClickEvent(event)
        cursor = self.cursorForPosition(event.position().toPoint())
        cursor.select(QTextCursor.SelectionType.LineUnderCursor)
        symbols = extract_watchlist_symbols(cursor.selectedText())
        if len(symbols) == 1:
            self.symbolActivated.emit(symbols[0])


class WatchlistsPanel(QFrame):
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._bounce_service = None
        self.shared = WatchlistEditorArea(
            "Shared BounceBot + Master AVWAP",
            "Shared Longs",
            LONGS_FILE,
            "Shared Shorts",
            SHORTS_FILE,
        )
        self.master = WatchlistEditorArea(
            "Master AVWAP Swing Lists",
            "Swing Longs",
            SWING_LONGS_FILE,
            "Short Swings",
            SWING_SHORTS_FILE,
        )
        self.auto = AutoWatchlistViewerArea()
        self.shared.statusChanged.connect(self.statusChanged)
        self.master.statusChanged.connect(self.statusChanged)

        tabs = QTabWidget()
        tabs.addTab(self.shared, "Shared Longs / Shorts")
        tabs.addTab(self.master, "Master Swing Lists")
        tabs.addTab(self.auto, "Auto Lists (bot-owned)")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(
            SectionHeader(
                "Watchlists",
                "Edit shared lists used by BounceBot and Master AVWAP, plus Master-only swing lists. "
                "Double-click any symbol line for its D1+M5 snapshot chart.",
            )
        )
        layout.addWidget(tabs, 1)

        for area in (self.shared, self.master, self.auto):
            for leaf in area.symbol_panels():
                leaf.text.symbolActivated.connect(self._open_symbol_snapshot)

    def set_bounce_service(self, service) -> None:
        """Optional: cached M5 bars for the popup's lower chart."""
        self._bounce_service = service

    def _open_symbol_snapshot(self, symbol: str) -> None:
        bot = None
        if self._bounce_service is not None:
            try:
                bot = self._bounce_service.current_bot()
            except Exception:
                bot = None
        from ui.widgets.symbol_snapshot_dialog import show_symbol_snapshot

        show_symbol_snapshot(self, symbol, bot=bot)


class AutoWatchlistViewerArea(QWidget):
    """Read-only view of the bot's own pick files (autolongs/autoshorts).

    The bot rewrites these intraday, so they are display-only here: editing
    would silently lose to the next auto write. The view re-reads from disk
    on a timer so the trader can watch names land without touching anything.
    """

    _REFRESH_MS = 30_000

    def __init__(self) -> None:
        super().__init__()
        self.long_viewer = AutoWatchlistViewerPanel("Auto Longs", AUTO_LONGS_FILE)
        self.short_viewer = AutoWatchlistViewerPanel("Auto Shorts", AUTO_SHORTS_FILE)

        splitter = QSplitter()
        splitter.addWidget(self.long_viewer)
        splitter.addWidget(self.short_viewer)
        splitter.setSizes([600, 600])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QLabel("Auto Pilot's own picks (read-only; the bot rewrites these intraday)")
        header.setObjectName("SectionTitle")
        layout.addWidget(header)
        layout.addWidget(splitter, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(self._REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def refresh(self) -> None:
        self.long_viewer.refresh_from_disk()
        self.short_viewer.refresh_from_disk()

    def symbol_panels(self) -> tuple:
        return (self.long_viewer, self.short_viewer)


class AutoWatchlistViewerPanel(QFrame):
    def __init__(self, title: str, path: Path) -> None:
        super().__init__()
        self.setObjectName("Panel")
        self.title = title
        self.path = Path(path)

        self.text = _SymbolTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("Empty - the bot has not written any picks here yet today.")
        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(self.title)
        title_label.setObjectName("SectionTitle")
        header.addWidget(title_label)
        header.addStretch(1)
        refresh_button = QPushButton("Refresh")
        copy_button = QPushButton("Copy")
        open_button = QPushButton("Open Folder")
        refresh_button.clicked.connect(self.refresh_from_disk)
        copy_button.clicked.connect(self._copy_symbols)
        open_button.clicked.connect(lambda: open_path_in_file_manager(self.path.parent))
        for button in (refresh_button, copy_button, open_button):
            header.addWidget(button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addWidget(self.status_label)
        layout.addWidget(self.text, 1)
        self.refresh_from_disk()

    def current_symbols(self) -> list[str]:
        return extract_watchlist_symbols(self.text.toPlainText())

    def refresh_from_disk(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        except OSError:
            raw = ""
        symbols = extract_watchlist_symbols(raw)
        self.text.setPlainText("\n".join(symbols))
        written_at = ""
        try:
            if self.path.exists():
                stamp = datetime.fromtimestamp(self.path.stat().st_mtime)
                written_at = f" | written {stamp.strftime('%H:%M:%S')}"
        except OSError:
            pass
        label = "symbol" if len(symbols) == 1 else "symbols"
        self.status_label.setText(f"{self.path.name} | {len(symbols)} {label}{written_at} | {self.path}")

    def _copy_symbols(self) -> None:
        QApplication.clipboard().setText(", ".join(self.current_symbols()))


class WatchlistEditorArea(QWidget):
    statusChanged = Signal(str)

    def __init__(self, title: str, long_title: str, long_path: Path, short_title: str, short_path: Path) -> None:
        super().__init__()
        self.long_editor = WatchlistEditorPanel(long_title, long_path, self._handle_symbols_saved)
        self.short_editor = WatchlistEditorPanel(short_title, short_path, self._handle_symbols_saved)
        self.long_editor.statusChanged.connect(self.statusChanged)
        self.short_editor.statusChanged.connect(self.statusChanged)

        splitter = QSplitter()
        splitter.addWidget(self.long_editor)
        splitter.addWidget(self.short_editor)
        splitter.setSizes([600, 600])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QLabel(title)
        header.setObjectName("SectionTitle")
        layout.addWidget(header)
        layout.addWidget(splitter, 1)

    def _handle_symbols_saved(self, source: "WatchlistEditorPanel", symbols: list[str]) -> None:
        peer = self.short_editor if source is self.long_editor else self.long_editor
        peer.remove_symbols(set(symbols))

    def symbol_panels(self) -> tuple:
        return (self.long_editor, self.short_editor)


class WatchlistEditorPanel(QFrame):
    statusChanged = Signal(str)

    def __init__(self, title: str, path: Path, on_symbols_saved: Callable[["WatchlistEditorPanel", list[str]], None]) -> None:
        super().__init__()
        self.setObjectName("Panel")
        self.title = title
        self.path = path
        self.on_symbols_saved = on_symbols_saved
        self._loading = False

        self.add_symbol_input = QLineEdit()
        self.add_symbol_input.setPlaceholderText("Add ticker")
        self.text = _SymbolTextEdit()
        self.text.setPlaceholderText("One symbol per line")
        self.status_label = QLabel("Autosave is on.")
        self.status_label.setObjectName("MutedLabel")

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(350)
        self._save_timer.timeout.connect(lambda: self._save_current(notify=True))

        self._build_layout()
        self.refresh_from_disk()

    def _build_layout(self) -> None:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel(self.title)
        title.setObjectName("SectionTitle")
        header.addWidget(title)
        header.addStretch(1)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)
        add_button = QPushButton("Add")
        dedupe_button = QPushButton("Dedupe Now")
        sort_button = QPushButton("Sort A-Z")
        paste_button = QPushButton("Paste")
        copy_button = QPushButton("Copy")
        refresh_button = QPushButton("Refresh")
        open_button = QPushButton("Open Folder")

        self.add_symbol_input.returnPressed.connect(self.add_symbol)
        add_button.clicked.connect(self.add_symbol)
        dedupe_button.clicked.connect(self.force_save)
        sort_button.clicked.connect(self.sort_symbols)
        paste_button.clicked.connect(self.paste_symbols)
        copy_button.clicked.connect(self.copy_symbols)
        refresh_button.clicked.connect(self.refresh_from_disk)
        open_button.clicked.connect(lambda: open_path_in_file_manager(self.path.parent))
        self.text.textChanged.connect(self._on_text_changed)

        action_row.addWidget(self.add_symbol_input)
        for button in (add_button, dedupe_button, sort_button, paste_button, copy_button, refresh_button, open_button):
            action_row.addWidget(button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addLayout(action_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.text, 1)

    def refresh_from_disk(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            raw = ""
        symbols = extract_watchlist_symbols(raw)
        self._set_symbols(symbols)
        self._set_status(symbols, "loaded")

    def force_save(self) -> None:
        self._save_timer.stop()
        self._save_current(notify=True)

    def add_symbol(self) -> None:
        incoming = extract_watchlist_symbols(self.add_symbol_input.text())
        if not incoming:
            return
        symbols = _merge_symbols(self.current_symbols(), incoming)
        self._write_symbols(symbols, notify=True)
        self.add_symbol_input.clear()

    def paste_symbols(self) -> None:
        incoming = extract_watchlist_symbols(QApplication.clipboard().text())
        if not incoming:
            return
        self._write_symbols(_merge_symbols(self.current_symbols(), incoming), notify=True)

    def copy_symbols(self) -> None:
        symbols = self.current_symbols()
        QApplication.clipboard().setText(", ".join(symbols))
        self._set_status(symbols, "copied")

    def sort_symbols(self) -> None:
        self._write_symbols(sorted(self.current_symbols()), notify=True)

    def remove_symbols(self, symbols_to_remove: set[str]) -> None:
        current = self.current_symbols()
        filtered = [symbol for symbol in current if symbol not in symbols_to_remove]
        if filtered != current:
            self._write_symbols(filtered, notify=False)

    def current_symbols(self) -> list[str]:
        return extract_watchlist_symbols(self.text.toPlainText())

    def _on_text_changed(self) -> None:
        if self._loading:
            return
        self._save_timer.start()

    def _save_current(self, notify: bool) -> None:
        self._write_symbols(self.current_symbols(), notify=notify)

    def _write_symbols(self, symbols: list[str], notify: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(symbols), encoding="utf-8")
        self._set_symbols(symbols)
        self._set_status(symbols, "saved")
        if notify:
            self.on_symbols_saved(self, symbols)

    def _set_symbols(self, symbols: list[str]) -> None:
        self._loading = True
        try:
            self.text.setPlainText("\n".join(symbols))
        finally:
            self._loading = False

    def _set_status(self, symbols: list[str], action: str) -> None:
        label = "symbol" if len(symbols) == 1 else "symbols"
        message = f"{self.path.name} | {len(symbols)} {label} | {action} | {self.path}"
        self.status_label.setText(message)
        self.statusChanged.emit(message)


def _merge_symbols(current: list[str], incoming: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for symbol in [*current, *incoming]:
        if symbol in seen:
            continue
        seen.add(symbol)
        merged.append(symbol)
    return merged
