from __future__ import annotations

import threading
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from universe_builder import (
    DEFAULT_OPTIONS_FILTER,
    OPTIONS_FILTER_CHOICES,
    UNIVERSE_ALL_FILE,
    UNIVERSE_LONGS_FILE,
    UNIVERSE_SHORTS_FILE,
    build_universe,
    compare_symbol_lists,
)
from watchlist_utils import extract_watchlist_symbols
from ui.widgets.section_header import SectionHeader

UNIVERSE_FILES = {
    "All (quality screen)": UNIVERSE_ALL_FILE,
    "Longs (above SMA100+200)": UNIVERSE_LONGS_FILE,
    "Shorts (below SMA100+200)": UNIVERSE_SHORTS_FILE,
}


class UniversePanel(QFrame):
    """Today's self-built scan universe + a paste-box diff against TC2000.

    The point of the diff is validation: paste the TC2000 watchlist this
    universe is meant to replace and see exactly which names differ, so the
    local screen can be trusted (or tuned) before the scanners consume it.
    """

    statusChanged = Signal(str)
    _buildFinished = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._build_thread: threading.Thread | None = None

        self.list_selector = QComboBox()
        self.list_selector.addItems(list(UNIVERSE_FILES))
        self.list_selector.currentTextChanged.connect(lambda _text: self.refresh_from_disk())

        # "optionable" mirrors TC2000's 'Optionable Stocks Is True'; "weeklies"
        # narrows to weekly-options names; "none" screens every listed stock.
        self.options_filter_selector = QComboBox()
        self.options_filter_selector.addItems(list(OPTIONS_FILTER_CHOICES))
        self.options_filter_selector.setCurrentText(DEFAULT_OPTIONS_FILTER)
        self.options_filter_selector.setToolTip(
            "optionable = any listed options (TC2000 'Optionable Stocks Is True')\n"
            "weeklies = weekly options only\nnone = every listed stock"
        )

        self.build_button = QPushButton("Build Universe Now")
        self.build_button.clicked.connect(self.start_build)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_from_disk)
        copy_button = QPushButton("Copy Tickers")
        copy_button.clicked.connect(self.copy_symbols)

        self.build_status = QLabel("")
        self.build_status.setObjectName("MutedLabel")

        self.universe_text = QPlainTextEdit()
        self.universe_text.setReadOnly(True)
        self.universe_text.setPlaceholderText(
            "No universe built yet. Click 'Build Universe Now' (takes a few minutes on the "
            "first run; cached and fast afterwards)."
        )

        self.compare_input = QPlainTextEdit()
        self.compare_input.setPlaceholderText(
            "Paste your TC2000 watchlist here (any format: commas, spaces or one per line), "
            "then click Compare."
        )
        compare_button = QPushButton("Compare With Pasted List")
        compare_button.clicked.connect(self.run_compare)
        self.compare_output = QPlainTextEdit()
        self.compare_output.setReadOnly(True)
        self.compare_output.setPlaceholderText("Comparison results appear here.")

        self._buildFinished.connect(self._on_build_finished)
        self._build_layout(refresh_button, copy_button, compare_button)
        self.refresh_from_disk()

    def _build_layout(self, refresh_button: QPushButton, copy_button: QPushButton, compare_button: QPushButton) -> None:
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        action_row.addWidget(self.list_selector)
        action_row.addWidget(QLabel("Options:"))
        action_row.addWidget(self.options_filter_selector)
        action_row.addWidget(self.build_button)
        action_row.addWidget(refresh_button)
        action_row.addWidget(copy_button)
        action_row.addStretch(1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        universe_title = QLabel("Today's Universe")
        universe_title.setObjectName("SectionTitle")
        left_layout.addWidget(universe_title)
        left_layout.addLayout(action_row)
        left_layout.addWidget(self.build_status)
        left_layout.addWidget(self.universe_text, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        compare_title = QLabel("Does it match your TC2000 list?")
        compare_title.setObjectName("SectionTitle")
        right_layout.addWidget(compare_title)
        right_layout.addWidget(self.compare_input, 1)
        right_layout.addWidget(compare_button)
        right_layout.addWidget(self.compare_output, 2)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([620, 620])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(
            SectionHeader(
                "Universe",
                "Self-built scan universe (optionable/weeklies filter + price/volume/cap/trend "
                "screen via yfinance -- no IBKR pacing). Validate it against your TC2000 list on the right.",
            )
        )
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Universe list display
    # ------------------------------------------------------------------
    def _selected_path(self) -> Path:
        return UNIVERSE_FILES[self.list_selector.currentText()]

    def current_symbols(self) -> list[str]:
        return extract_watchlist_symbols(self.universe_text.toPlainText())

    def refresh_from_disk(self) -> None:
        path = self._selected_path()
        if not path.exists():
            self.universe_text.setPlainText("")
            self.build_status.setText(f"{path.name}: not built yet.")
            return
        try:
            symbols = extract_watchlist_symbols(path.read_text(encoding="utf-8"))
        except OSError:
            symbols = []
        built_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes")
        self.universe_text.setPlainText("\n".join(symbols))
        message = f"{path.name}: {len(symbols)} symbols | built {built_at}"
        self.build_status.setText(message)
        self.statusChanged.emit(message)

    def copy_symbols(self) -> None:
        symbols = self.current_symbols()
        QApplication.clipboard().setText(", ".join(symbols))
        self.build_status.setText(f"Copied {len(symbols)} symbols to clipboard.")

    # ------------------------------------------------------------------
    # Build (background thread; yfinance-only so the UI stays responsive)
    # ------------------------------------------------------------------
    def start_build(self) -> None:
        if self._build_thread is not None and self._build_thread.is_alive():
            return
        self.build_button.setEnabled(False)
        options_filter = self.options_filter_selector.currentText()
        self.build_status.setText(
            f"Building universe ({options_filter})... (listing directory -> options -> prices -> caps)"
        )
        self._build_thread = threading.Thread(
            target=self._build_worker, args=(options_filter,), daemon=True
        )
        self._build_thread.start()

    def _build_worker(self, options_filter: str) -> None:
        try:
            result = build_universe(options_filter=options_filter)
            applied = "applied" if result["options_filter_applied"] else "SKIPPED (source unreachable)"
            message = (
                f"Universe built: {len(result['all'])} total / {len(result['longs'])} longs / "
                f"{len(result['shorts'])} shorts | {result['options_filter']} options filter {applied}"
            )
        except Exception as exc:  # surfaced in the status label, never crashes the UI
            traceback.print_exc()
            message = f"Universe build failed: {exc}"
        self._buildFinished.emit(message)

    def _on_build_finished(self, message: str) -> None:
        self.build_button.setEnabled(True)
        self.refresh_from_disk()
        self.build_status.setText(message)
        self.statusChanged.emit(message)

    # ------------------------------------------------------------------
    # TC2000 comparison
    # ------------------------------------------------------------------
    def run_compare(self) -> None:
        theirs = extract_watchlist_symbols(self.compare_input.toPlainText())
        ours = self.current_symbols()
        if not theirs:
            self.compare_output.setPlainText("Paste your TC2000 list on the left of this box first.")
            return
        if not ours:
            self.compare_output.setPlainText("No universe list loaded. Build or refresh it first.")
            return
        result = compare_symbol_lists(ours, theirs)
        lines = [
            f"Selected list: {self.list_selector.currentText()}",
            f"Ours: {result['ours_count']}  |  Yours: {result['theirs_count']}  |  "
            f"Matched: {result['matched_count']}  ({result['overlap_pct']}% of your list)",
            "",
            f"-- In YOUR list but missing from ours ({len(result['only_theirs'])}) --",
            ", ".join(result["only_theirs"]) or "(none)",
            "",
            f"-- In OUR list but not in yours ({len(result['only_ours'])}) --",
            ", ".join(result["only_ours"]) or "(none)",
        ]
        self.compare_output.setPlainText("\n".join(lines))
        self.statusChanged.emit(
            f"Universe compare: {result['overlap_pct']}% of the pasted list matched."
        )

    def shutdown(self) -> None:
        # The build thread is a daemon and finishes on its own; nothing to stop.
        pass
