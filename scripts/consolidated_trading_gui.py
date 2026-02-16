#!/usr/bin/env python3
"""Unified GUI for AVWAP manager + BounceBot RRS panel + longs/shorts lists."""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bounce_bot import create_rrs_confirmed_panel, run_bot_with_gui
from master_avwap import LONGS_FILE, SHORTS_FILE, MasterAvwapGUI

DARK_GREY = "#2E2E2E"
TEXT_COLOR = "#E0E0E0"


class ConsolidatedTradingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Consolidated Trading GUI")
        self.root.geometry("1800x980")

        self.rrs_queue: queue.Queue = queue.Queue()
        self.bot_instance = None
        self.selected_symbol = tk.StringVar(value="")

        self._build_layout()
        self._start_bounce_rrs_feed()
        self.refresh_watchlists()
        self._process_rrs_queue()

    def _build_layout(self):
        main = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg=DARK_GREY)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: full Master AVWAP GUI embedded intact.
        avwap_root = ttk.Frame(main)
        self.avwap_gui = MasterAvwapGUI(avwap_root, standalone=False)
        main.add(avwap_root, stretch="always", minsize=980)

        # Right: Bounce RRS + longs/shorts text displays.
        right = tk.Frame(main, bg=DARK_GREY)
        main.add(right, stretch="always", minsize=700)

        self.rrs_panel = create_rrs_confirmed_panel(right, bot_instance=self._bot_proxy(), dark_grey=DARK_GREY, text_color=TEXT_COLOR)
        self.rrs_panel["container"].pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        watch_frame = tk.LabelFrame(right, text="TickerMover Watchlists", bg=DARK_GREY, fg=TEXT_COLOR, padx=8, pady=8)
        watch_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        controls = tk.Frame(watch_frame, bg=DARK_GREY)
        controls.pack(fill=tk.X)
        tk.Label(controls, text="Selected:", bg=DARK_GREY, fg=TEXT_COLOR).pack(side=tk.LEFT)
        tk.Entry(controls, textvariable=self.selected_symbol, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Refresh", command=self.refresh_watchlists).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Add to longs", command=lambda: self.add_symbol_to_file(LONGS_FILE)).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Add to shorts", command=lambda: self.add_symbol_to_file(SHORTS_FILE)).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Load into AVWAP ticker", command=self.load_symbol_into_avwap).pack(side=tk.LEFT, padx=4)

        lists = tk.PanedWindow(watch_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg=DARK_GREY)
        lists.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        longs_panel = tk.LabelFrame(lists, text="longs.txt", bg=DARK_GREY, fg=TEXT_COLOR)
        self.longs_text = scrolledtext.ScrolledText(longs_panel, wrap=tk.NONE, font=("Courier", 11), bg="#222", fg=TEXT_COLOR)
        self.longs_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.longs_text.bind("<Double-Button-1>", lambda e: self._pick_symbol_from_text(self.longs_text))
        lists.add(longs_panel, stretch="always")

        shorts_panel = tk.LabelFrame(lists, text="shorts.txt", bg=DARK_GREY, fg=TEXT_COLOR)
        self.shorts_text = scrolledtext.ScrolledText(shorts_panel, wrap=tk.NONE, font=("Courier", 11), bg="#222", fg=TEXT_COLOR)
        self.shorts_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.shorts_text.bind("<Double-Button-1>", lambda e: self._pick_symbol_from_text(self.shorts_text))
        lists.add(shorts_panel, stretch="always")

    def _bot_proxy(self):
        outer = self

        class _Proxy:
            rrs_threshold = 2.0
            rrs_timeframe_key = "5m"

            def set_rrs_threshold(self, value):
                self.rrs_threshold = float(value)
                if outer.bot_instance:
                    outer.bot_instance.set_rrs_threshold(value)

            def set_rrs_timeframe(self, key):
                self.rrs_timeframe_key = key
                if outer.bot_instance:
                    outer.bot_instance.set_rrs_timeframe(key)

        self.bot_proxy = _Proxy()
        return self.bot_proxy

    def _start_bounce_rrs_feed(self):
        def gui_callback(message, tag):
            if tag.startswith("rrs"):
                self.rrs_queue.put((message, tag))

        def run_bot():
            try:
                self.bot_instance = run_bot_with_gui(gui_callback)
                self.bot_proxy.rrs_threshold = self.bot_instance.rrs_threshold
                self.bot_proxy.rrs_timeframe_key = self.bot_instance.rrs_timeframe_key
            except Exception as exc:
                self.rrs_queue.put((f"BounceBot start failed: {exc}", "rrs_status"))

        threading.Thread(target=run_bot, daemon=True).start()

    def _process_rrs_queue(self):
        while True:
            try:
                msg, tag = self.rrs_queue.get_nowait()
            except queue.Empty:
                break

            if tag == "rrs_status":
                self.rrs_panel["rrs_status_var"].set(str(msg))
            elif tag == "rrs_snapshot":
                self.rrs_panel["render_rrs_snapshot"](msg)

        self.root.after(150, self._process_rrs_queue)

    def refresh_watchlists(self):
        self._load_file_into_text(LONGS_FILE, self.longs_text)
        self._load_file_into_text(SHORTS_FILE, self.shorts_text)

    def _load_file_into_text(self, path: Path, widget: scrolledtext.ScrolledText):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            text = ""
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)

    def _pick_symbol_from_text(self, widget: scrolledtext.ScrolledText):
        idx = widget.index(f"@{widget.winfo_pointerx()-widget.winfo_rootx()},{widget.winfo_pointery()-widget.winfo_rooty()}")
        line = widget.get(f"{idx} linestart", f"{idx} lineend").strip().upper()
        if not line:
            return
        symbol = line.split(",")[0].split()[0]
        self.selected_symbol.set(symbol)

    def add_symbol_to_file(self, path: Path):
        symbol = self.selected_symbol.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Missing symbol", "Select or type a symbol first.")
            return

        existing = []
        if path.exists():
            existing = [ln.strip().upper() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if symbol not in existing:
            existing.append(symbol)
            path.write_text("\n".join(existing) + "\n", encoding="utf-8")
        self.refresh_watchlists()

    def load_symbol_into_avwap(self):
        symbol = self.selected_symbol.get().strip().upper()
        if not symbol:
            return
        self.avwap_gui.ticker_var.set(symbol)

    def on_close(self):
        try:
            if self.bot_instance:
                self.bot_instance.disconnect()
        finally:
            self.root.destroy()


def launch():
    root = tk.Tk()
    app = ConsolidatedTradingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    launch()
