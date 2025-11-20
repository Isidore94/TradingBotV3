#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
import time
import threading
import hashlib
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QWidget, QTextEdit, QVBoxLayout, QPushButton,
    QMessageBox, QLabel, QScrollArea, QFrame, QHBoxLayout
)
from PyQt5.QtCore import Qt, QFileSystemWatcher, QTimer, QDateTime
from PyQt5.QtGui import QPalette, QColor

# ──────────────────────────────────────────────────────────────────────────────
# Files & Settings
# ──────────────────────────────────────────────────────────────────────────────
SHORTS_FILE          = "shorts.txt"
LONGS_FILE           = "longs.txt"
BOUNCERS_FILE        = "bouncers.txt"              # old/live bouncer feed (HH:MM:SS | SYM | types | side)
PREV_BOUNCERS_FILE   = "prev_avwap_bouncers.txt"   # new: prev-earnings AVWAP bounces script output
COMBINED_FILE        = "combined_avwap.txt"        # main AVWAP2 output incl. BOUNCE_*
ALERTS_FILE          = "alerts_feed.txt"           # this script writes alerts here

# Freshness / matching behavior
COOLDOWN_SECONDS   = 300                 # per-symbol cooldown between alerts
BACKFILL_ON_START  = False               # if True, sweep today’s combined on launch

# Bouncers read mode for BOUNCERS_FILE (unchanged behavior)
#   "latest_batch" -> use only the most recent timestamp batch in bouncers.txt
#   "window"       -> use entries newer than now - BOUNCER_WINDOW_SECONDS
BOUNCER_MODE           = "latest_batch"
BOUNCER_WINDOW_SECONDS = 420             # only for "window" mode

# Drive/FS heartbeat sweep interval
HEARTBEAT_SECONDS = 60

# Files that can be opened via "Start New Instance"
# Order: start with shorts, then open others
FILE_LIST = [
    SHORTS_FILE,
    LONGS_FILE,
    BOUNCERS_FILE,
    PREV_BOUNCERS_FILE,
    COMBINED_FILE,
    ALERTS_FILE,
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
            PREV_BOUNCERS_FILE,
            COMBINED_FILE,
            ALERTS_FILE,
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

        # Extra filters ONLY for combined_avwap.txt
        if self.filename == COMBINED_FILE:
            xbtn = QPushButton("Copy Crossers")
            xbtn.clicked.connect(self.copy_crossers)
            xbtn.setStyleSheet(f"background-color:#3D3D3D; color:{TEXT_COLOR};")
            btn_row.addWidget(xbtn)

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

    # visible combined rows (for COMBINED_FILE window)
    def _visible_combined_rows(self):
        if self.filename != COMBINED_FILE or not self.newest_at_top:
            return []
        rows = []
        for line in self.list_widget.get_tickers():  # visible order
            parsed = _parse_combined_line(line)
            if parsed:
                rows.append((line, parsed))
        return rows

    # Copy crossers only (unchanged)
    def copy_crossers(self):
        rows = self._visible_combined_rows()
        keep = []
        for original, (_, _, level, _) in rows:
            if level.startswith("CROSS_UP_") or level.startswith("CROSS_DOWN_"):
                keep.append(original)
        QApplication.clipboard().setText("\n".join(keep))

    # Copy VWAP taps
    def copy_vwap(self):
        rows = self._visible_combined_rows()
        keep = [original for original, (_, _, level, _) in rows if level == "VWAP"]
        QApplication.clipboard().setText("\n".join(keep))

    # Copy Nσ band proximity (UPPER_n / LOWER_n exact)
    def copy_sd_band(self, n: int):
        target_upper = f"UPPER_{n}"
        target_lower = f"LOWER_{n}"
        rows = self._visible_combined_rows()
        keep = []
        for original, (_, _, level, _) in rows:
            if level == target_upper or level == target_lower:
                keep.append(original)
        QApplication.clipboard().setText("\n".join(keep))

    # NEW: Copy all bounce signals from combined_avwap
    def copy_bounces(self):
        rows = self._visible_combined_rows()
        keep = []
        for original, (_, _, level, _) in rows:
            if level.startswith("BOUNCE_") or level.startswith("PREV_BOUNCE_"):
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

def _today_mmdd():
    return datetime.now().strftime("%m/%d")

def _re_fullmatch(pattern: str, text: str) -> bool:
    return re.fullmatch(pattern, text) is not None

def _parse_combined_line(line: str):
    """
    combined_avwap.txt lines:
      SYMBOL,MM/DD,LEVEL,SIDE

    LEVEL can be:
      - UPPER_n / LOWER_n (tier classifications)
      - VWAP
      - CROSS_UP_* / CROSS_DOWN_*
      - BOUNCE_* (current-earnings bounces)
      - PREV_BOUNCE_* (prev-earnings bounces, if you ever co-locate)

    Returns (sym, mmdd, level, side) or None.
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

def _parse_bouncer_line(line: str):
    """
    bouncers.txt legacy format:
      "HH:MM:SS | SYM | types | long"
      "HH:MM:SS | SYM | vwap, eod_vwap | short"
    Returns (timestamp, sym, types, side_match, side_disp) or None.
    """
    parts = [p.strip() for p in line.strip().split("|")]
    if len(parts) < 4:
        return None
    time_txt = parts[0]
    sym      = parts[1].upper()
    types    = parts[2]
    side_disp = parts[3].lower()
    side_match = side_disp.upper()

    if not _re_fullmatch(SYMBOL_RE, sym) or side_match not in ("LONG", "SHORT"):
        return None

    try:
        t = datetime.strptime(time_txt, "%H:%M:%S")
        ts = datetime.now().replace(
            hour=t.hour, minute=t.minute, second=t.second, microsecond=0
        )
    except Exception:
        return None

    return (ts, sym, types, side_match, side_disp)

def _load_bouncers_latest_batch_state(path: str):
    """
    For bouncers.txt (legacy live feed):
      bmap = {SYM: {'types': <display_str>, 'side_match': 'LONG/SHORT', 'side_disp': 'long/short'}}
      batch_tag = latest "HH:MM:SS"
    """
    latest_ts = None
    rows = []
    if not os.path.exists(path):
        return {}, ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_bouncer_line(line)
            if parsed:
                rows.append(parsed)
                if latest_ts is None or parsed[0] > latest_ts:
                    latest_ts = parsed[0]
    if latest_ts is None:
        return {}, ""
    batch_tag = latest_ts.strftime("%H:%M:%S")
    bmap = {}
    for ts, sym, types, side_match, side_disp in rows:
        if ts.strftime("%H:%M:%S") == batch_tag:
            bmap[sym] = {
                "types": types,
                "side_match": side_match,
                "side_disp": side_disp,
            }
    return bmap, batch_tag

def _load_bouncers_window_state(path: str, window_seconds: int):
    """
    Rolling-window view of bouncers.txt.
    """
    cutoff = datetime.now() - timedelta(seconds=window_seconds)
    bmap = {}
    if not os.path.exists(path):
        return {}, ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_bouncer_line(line)
            if not parsed:
                continue
            ts, sym, types, side_match, side_disp = parsed
            if ts >= cutoff:
                bmap[sym] = {
                    "types": types,
                    "side_match": side_match,
                    "side_disp": side_disp,
                }
    anchor = datetime.now().strftime("%Y%m%d%H%M")
    return bmap, f"window:{anchor}"

def _load_bouncers_state(path: str):
    if BOUNCER_MODE == "window":
        return _load_bouncers_window_state(path, BOUNCER_WINDOW_SECONDS)
    return _load_bouncers_latest_batch_state(path)

def _scan_file(path: str):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
    except Exception:
        return

def _truncate_file(path: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass

def _append_alert(msg: str):
    try:
        with open(ALERTS_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    try:
        QApplication.beep()
    except Exception:
        pass

def _handle_one_combined_line(line: str,
                              bouncers_map: dict,
                              batch_tag: str,
                              last_alert_time: dict,
                              emitted_keys: set,
                              cooldown: timedelta):
    """
    Alert if:
      - line is for TODAY
      - symbol present in freshest bouncers_map (legacy live bouncer feed)
      - side matches
      - cooldown respected
      - not already emitted for (sym, level, side, batch_tag)
    Works with any LEVEL (including BOUNCE_*).
    """
    parsed = _parse_combined_line(line)
    if not parsed:
        return
    sym, mmdd, level, combined_side = parsed
    if mmdd != _today_mmdd():
        return

    binfo = bouncers_map.get(sym)
    if not binfo:
        return
    if binfo["side_match"] != combined_side:
        return

    key = (sym, level, combined_side, batch_tag)
    if key in emitted_keys:
        return

    now = datetime.now()
    prev = last_alert_time.get(sym)
    if prev is not None and (now - prev) < cooldown:
        return

    last_alert_time[sym] = now
    emitted_keys.add(key)

    stamp = now.strftime("%H:%M:%S")
    types_disp = binfo["types"]
    msg = f"{stamp} | {sym} | {types_disp} | {level} | {combined_side}"
    _append_alert(msg)

def _sweep_current_for_matches(combined_path: str,
                               bouncers_map: dict,
                               batch_tag: str,
                               last_alert_time: dict,
                               emitted_keys: set,
                               cooldown: timedelta):
    """Sweep all TODAY lines in combined using the current bouncers_map."""
    today = _today_mmdd()
    for line in _scan_file(combined_path):
        parsed = _parse_combined_line(line)
        if not parsed:
            continue
        sym, mmdd, level, side = parsed
        if mmdd != today:
            continue
        binfo = bouncers_map.get(sym)
        if not binfo or binfo["side_match"] != side:
            continue

        key = (sym, level, side, batch_tag)
        if key in emitted_keys:
            continue

        now = datetime.now()
        prev = last_alert_time.get(sym)
        if prev is not None and (now - prev) < cooldown:
            continue

        last_alert_time[sym] = now
        emitted_keys.add(key)

        stamp = now.strftime("%H:%M:%S")
        types_disp = binfo["types"]
        msg = f"{stamp} | {sym} | {types_disp} | {level} | {side}"
        _append_alert(msg)

# ──────────────────────────────────────────────────────────────────────────────
# Drive-friendly Watcher
# ──────────────────────────────────────────────────────────────────────────────
def _fast_sig(path):
    """Fast fingerprint using head+tail blocks."""
    try:
        sz = os.path.getsize(path)
        head = b""
        tail = b""
        with open(path, "rb") as f:
            head = f.read(4096)
            if sz > 8192:
                f.seek(-4096, os.SEEK_END)
                tail = f.read(4096)
        h = hashlib.md5()
        h.update(head)
        h.update(tail)
        return h.hexdigest(), sz
    except Exception:
        return "", 0

def avwap_watcher_thread(base_dir: str):
    """
    Watches:
      - combined_avwap.txt (rewritten)
      - bouncers.txt (legacy live feed)
    Generates alerts_feed.txt when symbols & sides align.
    BOUNCE_* levels are supported automatically via _parse_combined_line.
    """
    combined_path = os.path.join(base_dir, COMBINED_FILE)
    bouncers_path = os.path.join(base_dir, BOUNCERS_FILE)
    alerts_path   = os.path.join(base_dir, ALERTS_FILE)

    # Ensure files exist
    for p in (combined_path, bouncers_path, alerts_path):
        if not os.path.exists(p):
            open(p, "a", encoding="utf-8").close()

    _truncate_file(alerts_path)
    last_alert_time = {}
    emitted_keys = set()
    cooldown = timedelta(seconds=COOLDOWN_SECONDS)

    # Initial bouncers load
    last_bouncer_mtime = os.path.getmtime(bouncers_path) if os.path.exists(bouncers_path) else 0.0
    bouncers_map, batch_tag = _load_bouncers_state(bouncers_path)

    # Optional backfill
    if BACKFILL_ON_START:
        _sweep_current_for_matches(
            combined_path, bouncers_map, batch_tag,
            last_alert_time, emitted_keys, cooldown
        )

    # Track combined file metadata
    def _stat_ns(path):
        try:
            st = os.stat(path)
            mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
            return mtime_ns, st.st_size
        except FileNotFoundError:
            return 0, 0

    last_mtime_ns, last_size = _stat_ns(combined_path)
    last_sig, _ = _fast_sig(combined_path)

    # Initial sweep
    _sweep_current_for_matches(
        combined_path, bouncers_map, batch_tag,
        last_alert_time, emitted_keys, cooldown
    )

    POLL_INTERVAL = 0.5
    last_heartbeat = time.time()
    last_mmdd = _today_mmdd()

    while True:
        # New day reset
        cur_mmdd = _today_mmdd()
        if cur_mmdd != last_mmdd:
            last_mmdd = cur_mmdd
            last_alert_time.clear()
            emitted_keys.clear()
            _truncate_file(alerts_path)

        # Reload bouncers.txt if changed
        try:
            if os.path.exists(bouncers_path):
                b_mtime = os.path.getmtime(bouncers_path)
                if b_mtime > last_bouncer_mtime:
                    last_bouncer_mtime = b_mtime
                    bouncers_map, batch_tag = _load_bouncers_state(bouncers_path)
                    _sweep_current_for_matches(
                        combined_path, bouncers_map, batch_tag,
                        last_alert_time, emitted_keys, cooldown
                    )
        except Exception:
            pass

        # Detect combined_avwap changes
        try:
            cur_mtime_ns, cur_size = _stat_ns(combined_path)
            cur_sig, _ = _fast_sig(combined_path)
            if (cur_mtime_ns > last_mtime_ns) or (cur_size != last_size) or (cur_sig != last_sig):
                last_mtime_ns, last_size, last_sig = cur_mtime_ns, cur_size, cur_sig
                _sweep_current_for_matches(
                    combined_path, bouncers_map, batch_tag,
                    last_alert_time, emitted_keys, cooldown
                )
        except Exception:
            pass

        # Heartbeat sweep
        now_t = time.time()
        if now_t - last_heartbeat >= HEARTBEAT_SECONDS:
            last_heartbeat = now_t
            _sweep_current_for_matches(
                combined_path, bouncers_map, batch_tag,
                last_alert_time, emitted_keys, cooldown
            )

        time.sleep(POLL_INTERVAL)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)

    base_dir = os.path.abspath(os.getcwd())
    watcher = threading.Thread(target=avwap_watcher_thread, args=(base_dir,), daemon=True)
    watcher.start()

    # Start with Shorts window; open others via “Start New Instance”
    w = MainWindow(FILE_LIST[0])
    windows.append(w)
    w.show()

    sys.exit(app.exec_())
