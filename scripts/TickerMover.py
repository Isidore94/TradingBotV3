#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QTextEdit, QVBoxLayout, QPushButton,
    QMessageBox, QLabel, QScrollArea, QFrame, QHBoxLayout
)
from PyQt5.QtCore import Qt, QFileSystemWatcher, QTimer, QDateTime
from PyQt5.QtGui import QPalette, QColor

# ──────────────────────────────────────────────────────────────────────────────
# Files & Settings
# ──────────────────────────────────────────────────────────────────────────────
SHORTS_FILE        = "shorts.txt"
LONGS_FILE         = "longs.txt"
BOUNCERS_FILE      = "bouncers.txt"  # legacy live bouncer feed (HH:MM:SS | SYM | types | side)
MASTER_EVENTS_FILE = os.path.join("output", "master_avwap_events.txt")

# Files that can be opened via "Start New Instance"
# Order: start with shorts, then open others
FILE_LIST = [
    SHORTS_FILE,
    LONGS_FILE,
    BOUNCERS_FILE,
    MASTER_EVENTS_FILE,
]

next_file_index = 1
windows = []

# Theme
DARK_GREY  = "#2D2D2D"
TEXT_COLOR = "#E0E0E0"

# ──────────────────────────────────────────────────────────────────────────────
# UI Widgets
# ──────────────────────────────────────────────────────────────────────────────
class TickerWidget(QFrame):
    def __init__(self, ticker, parent=None):
        super().__init__(parent)
        self.ticker = ticker
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"background-color:{DARK_GREY};"
            f"color:{TEXT_COLOR};"
            "border:1px solid #555;"
        )
        layout = QVBoxLayout()
        lbl = QLabel(ticker)
        lbl.setStyleSheet(f"color:{TEXT_COLOR};font-size:10pt;")
        layout.addWidget(lbl)
        self.setLayout(layout)

class TickerListWidget(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.layout.setAlignment(Qt.AlignTop)
        self.setWidget(container)
        self.widgets = []

    def set_tickers(self, tickers):
        for w in self.widgets:
            self.layout.removeWidget(w)
            w.deleteLater()
        self.widgets.clear()
        for t in tickers:
            if t.strip():
                w = TickerWidget(t, self)
                self.layout.addWidget(w)
                self.widgets.append(w)

    def get_tickers(self):
        return [w.ticker for w in self.widgets]

class DragDropTextEdit(QTextEdit):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        pal = self.palette()
        pal.setColor(QPalette.Base, QColor(DARK_GREY))
        pal.setColor(QPalette.Text, QColor(TEXT_COLOR))
        self.setPalette(pal)
        self.setAcceptDrops(True)
        self.textChanged.connect(self.on_text_changed)
        self.last_content = ""

    def on_text_changed(self):
        cur = self.toPlainText()
        if cur != self.last_content:
            self.last_content = cur
            self.parent.save_tickers()
            self.parent.last_save = QDateTime.currentMSecsSinceEpoch()

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        text = event.mimeData().text().strip()
        if text:
            if self.parent.newest_at_top and "|" in text:
                parts = text.split("|")
                text = parts[1].strip() if len(parts) > 1 else text
            if self.parent.newest_at_top:
                self.setPlainText(text + "\n" + self.toPlainText())
            else:
                self.append(text)
        event.acceptProposedAction()

class MainWindow(QWidget):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

        # newest-at-top for list-style windows
        self.newest_at_top = filename in (
            BOUNCERS_FILE,
            MASTER_EVENTS_FILE,
        )

        self.setWindowTitle(f"TickerMover — {os.path.basename(filename)}")
        self.resize(560, 420)

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(DARK_GREY))
        pal.setColor(QPalette.WindowText, QColor(TEXT_COLOR))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

        layout = QVBoxLayout()

        # Clock
        self.clock = QLabel()
        self.clock.setAlignment(Qt.AlignCenter)
        self.clock.setStyleSheet(f"color:{TEXT_COLOR}; font-size:10pt;")
        layout.addWidget(self.clock)

        # Content widget
        if self.newest_at_top:
            self.list_widget = TickerListWidget(self)
            layout.addWidget(self.list_widget)
        else:
            self.text_edit = DragDropTextEdit(self)
            font = self.text_edit.font()
            font.setPointSize(int(font.pointSize() * 0.67))
            self.text_edit.setFont(font)
            layout.addWidget(self.text_edit)

        # Buttons row
        btn_row = QHBoxLayout()

        # Copy everything as visible
        copy_btn = QPushButton("Copy Tickers")
        copy_btn.clicked.connect(self.copy_tickers)
        copy_btn.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
        btn_row.addWidget(copy_btn)

        # Extra filters ONLY for master_avwap_events.txt
        if self.filename == MASTER_EVENTS_FILE:
            xbtn = QPushButton("Copy Crossers")
            xbtn.clicked.connect(self.copy_crossers)
            xbtn.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(xbtn)

            xup_btn = QPushButton("Copy Cross Ups")
            xup_btn.clicked.connect(lambda: self.copy_crosses_by_direction("UP"))
            xup_btn.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(xup_btn)

            xdn_btn = QPushButton("Copy Cross Downs")
            xdn_btn.clicked.connect(lambda: self.copy_crosses_by_direction("DOWN"))
            xdn_btn.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(xdn_btn)

            vwap_btn = QPushButton("Copy VWAP")
            vwap_btn.clicked.connect(self.copy_vwap)
            vwap_btn.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(vwap_btn)

            b1 = QPushButton("Copy 1SD")
            b1.clicked.connect(lambda: self.copy_sd_band(1))
            b1.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(b1)

            b2 = QPushButton("Copy 2SD")
            b2.clicked.connect(lambda: self.copy_sd_band(2))
            b2.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(b2)

            b3 = QPushButton("Copy 3SD")
            b3.clicked.connect(lambda: self.copy_sd_band(3))
            b3.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(b3)

            # NEW: Copy all bounce-style signals (BOUNCE_* and PREV_BOUNCE_*)
            bb = QPushButton("Copy Bounces")
            bb.clicked.connect(self.copy_bounces)
            bb.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(bb)

            bvu = QPushButton("Copy Bounce Upper")
            bvu.clicked.connect(lambda: self.copy_bounces_by_level("UPPER"))
            bvu.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(bvu)

            bvl = QPushButton("Copy Bounce Lower")
            bvl.clicked.connect(lambda: self.copy_bounces_by_level("LOWER"))
            bvl.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(bvl)

            bvv = QPushButton("Copy Bounce VWAP")
            bvv.clicked.connect(lambda: self.copy_bounces_by_level("VWAP"))
            bvv.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(bvv)

            cur_btn = QPushButton("Copy Current")
            cur_btn.clicked.connect(lambda: self.copy_avwap_by_scope("current"))
            cur_btn.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(cur_btn)

            prev_btn = QPushButton("Copy Previous")
            prev_btn.clicked.connect(lambda: self.copy_avwap_by_scope("previous"))
            prev_btn.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(prev_btn)

        layout.addLayout(btn_row)

        # Start New Instance button for editable files
        if not self.newest_at_top:
            new_btn = QPushButton("Start New Instance")
            new_btn.clicked.connect(self.start_new_instance)
            new_btn.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            layout.addWidget(new_btn)

        self.setLayout(layout)

        # Watcher & timer
        self.watcher = QFileSystemWatcher()
        if not os.path.exists(self.filename):
            open(self.filename, "a", encoding="utf-8").close()
        self.watcher.addPath(self.filename)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(1000)

        self.last_save = 0
        self.load_tickers(force=True)
        self.update_clock()

    # ───── UI tick/update ─────
    def on_timer_tick(self):
        self.update_clock()
        self.load_tickers(force=True)

    def update_clock(self):
        now = datetime.now()
        self.clock.setText(now.strftime("%H:%M:%S"))

    # ───── file <-> view ─────
    def load_tickers(self, force=False):
        try:
            with open(self.filename, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception:
            return
        if self.newest_at_top:
            self.list_widget.set_tickers(list(reversed(lines)))
        else:
            content = "\n".join(lines)
            if content != self.text_edit.toPlainText():
                self.text_edit.textChanged.disconnect()
                self.text_edit.setPlainText(content)
                self.text_edit.textChanged.connect(self.text_edit.on_text_changed)

    def save_tickers(self):
        if self.newest_at_top:
            lines = list(reversed(self.list_widget.get_tickers()))
            data = "\n".join(lines)
        else:
            data = self.text_edit.toPlainText()
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write(data)
        except Exception:
            pass

    # ───── copy helpers ─────
    def copy_tickers(self):
        if self.newest_at_top:
            content = "\n".join(self.list_widget.get_tickers())
        else:
            content = self.text_edit.toPlainText()
        QApplication.clipboard().setText(content)

    # visible master events rows (for MASTER_EVENTS_FILE window)
    def _visible_master_rows(self):
        if self.filename != MASTER_EVENTS_FILE or not self.newest_at_top:
            return []
        rows = []
        for line in self.list_widget.get_tickers():  # visible order
            parsed = _parse_master_event_line(line)
            if parsed:
                rows.append((line, parsed))
        return rows

    # Copy crossers only (unchanged)
    def copy_crossers(self):
        rows = self._visible_master_rows()
        keep = []
        for original, (_, _, level, _) in rows:
            if level.startswith("CROSS_UP_") or level.startswith("CROSS_DOWN_"):
                keep.append(original)
        QApplication.clipboard().setText("\n".join(keep))

    def copy_crosses_by_direction(self, direction: str):
        direction = direction.upper()
        if direction not in {"UP", "DOWN"}:
            return
        rows = self._visible_master_rows()
        keep = []
        prefix = f"CROSS_{direction}_"
        for original, (_, _, level, _) in rows:
            normalized = level.replace("PREV_", "")
            if normalized.startswith(prefix):
                keep.append(original)
        QApplication.clipboard().setText("\n".join(keep))

    # Copy VWAP taps
    def copy_vwap(self):
        rows = self._visible_master_rows()
        keep = [original for original, (_, _, level, _) in rows if level == "VWAP"]
        QApplication.clipboard().setText("\n".join(keep))

    # Copy Nσ band proximity (UPPER_n / LOWER_n exact)
    def copy_sd_band(self, n: int):
        target_upper = f"UPPER_{n}"
        target_lower = f"LOWER_{n}"
        rows = self._visible_master_rows()
        keep = []
        for original, (_, _, level, _) in rows:
            if level == target_upper or level == target_lower:
                keep.append(original)
        QApplication.clipboard().setText("\n".join(keep))

    # NEW: Copy all bounce signals from master_avwap_events
    def copy_bounces(self):
        rows = self._visible_master_rows()
        keep = []
        for original, (_, _, level, _) in rows:
            if level.startswith("BOUNCE_") or level.startswith("PREV_BOUNCE_"):
                keep.append(original)
        QApplication.clipboard().setText("\n".join(keep))

    def copy_bounces_by_level(self, target_level: str):
        target_level = target_level.upper()
        rows = self._visible_master_rows()
        keep = []
        for original, (_, _, level, _) in rows:
            normalized = level.replace("PREV_", "")
            if not normalized.startswith("BOUNCE_"):
                continue
            if target_level == "VWAP" and normalized == "BOUNCE_VWAP":
                keep.append(original)
            elif target_level == "UPPER" and "BOUNCE_UPPER" in normalized:
                keep.append(original)
            elif target_level == "LOWER" and "BOUNCE_LOWER" in normalized:
                keep.append(original)
        QApplication.clipboard().setText("\n".join(keep))

    def copy_avwap_by_scope(self, scope: str):
        scope = scope.lower()
        rows = self._visible_master_rows()
        keep = []
        for original, (_, _, level, _) in rows:
            if scope == "current" and not level.startswith("PREV_"):
                keep.append(original)
            elif scope == "previous" and level.startswith("PREV_"):
                keep.append(original)
        QApplication.clipboard().setText("\n".join(keep))

    def start_new_instance(self):
        global next_file_index
        if next_file_index < len(FILE_LIST):
            fn = FILE_LIST[next_file_index]
            next_file_index += 1
            w = MainWindow(fn)
            windows.append(w)
            w.show()
        else:
            QMessageBox.information(self, "Info", "No more files.")

# ──────────────────────────────────────────────────────────────────────────────
# Matching / Parsing Helpers
# ──────────────────────────────────────────────────────────────────────────────
SYMBOL_RE = r"[A-Z0-9.\-]+"

def _re_fullmatch(pattern: str, text: str) -> bool:
    return re.fullmatch(pattern, text) is not None

def _parse_master_event_line(line: str):
    """
    master_avwap_events.txt lines:
      SYMBOL,MM/DD,LEVEL,SIDE

    LEVEL can be:
      - UPPER_n / LOWER_n (tier classifications)
      - VWAP
      - CROSS_UP_* / CROSS_DOWN_*
      - BOUNCE_* (current-earnings bounces)
      - PREV_BOUNCE_* (prev-earnings bounces, if you ever co-locate)

    Supports both current and previous anchors; previous entries use the
    "PREV_" prefix. Returns (sym, mmdd, level, side) or None.
    """
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 4:
        return None
    sym = parts[0].upper()
    if not _re_fullmatch(SYMBOL_RE, sym):
        return None
    mmdd  = parts[1]
    level = parts[2]
    side  = parts[3].upper()
    if side not in ("LONG", "SHORT"):
        return None
    return sym, mmdd, level, side

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Ensure the master events directory exists so we can create the file if needed.
    events_dir = os.path.dirname(MASTER_EVENTS_FILE)
    if events_dir and not os.path.exists(events_dir):
        os.makedirs(events_dir, exist_ok=True)

    # Start with Shorts window; open others via “Start New Instance”
    w = MainWindow(FILE_LIST[0])
    windows.append(w)
    w.show()

    sys.exit(app.exec_())
