#!/usr/bin/env python3
"""Unified GUI for AVWAP manager + BounceBot RRS panel."""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bounce_bot import create_rrs_confirmed_panel, run_bot_with_gui
from master_avwap import MasterAvwapGUI

DARK_GREY = "#2E2E2E"
TEXT_COLOR = "#E0E0E0"
PANEL_GREY = "#3A3A3A"
INPUT_GREY = "#252525"


class ConsolidatedTradingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Consolidated Trading GUI")
        self.root.geometry("1800x980")
        self.root.configure(bg=DARK_GREY)

        self._configure_theme()

        self.rrs_queue: queue.Queue = queue.Queue()
        self.bot_instance = None
        self._build_layout()
        self._start_bounce_rrs_feed()
        self._process_rrs_queue()

    def _build_layout(self):
        main = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashrelief=tk.RAISED, bg=DARK_GREY)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main)
        self.avwap_gui = MasterAvwapGUI(top, standalone=False)
        main.add(top, stretch="always", minsize=620)

        bottom = tk.Frame(main, bg=DARK_GREY)
        self.rrs_panel = create_rrs_confirmed_panel(
            bottom,
            bot_instance=self._bot_proxy(),
            dark_grey=DARK_GREY,
            text_color=TEXT_COLOR,
        )
        self.rrs_panel["container"].pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        main.add(bottom, stretch="always", minsize=320)

        self._apply_dark_widget_theme()

    def _configure_theme(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        self.root.option_add("*Background", DARK_GREY)
        self.root.option_add("*Foreground", TEXT_COLOR)
        self.root.option_add("*Label*Background", DARK_GREY)
        self.root.option_add("*Entry*Background", INPUT_GREY)
        self.root.option_add("*Entry*Foreground", TEXT_COLOR)
        self.root.option_add("*Text*Background", INPUT_GREY)
        self.root.option_add("*Text*Foreground", TEXT_COLOR)

        style.configure(".", background=DARK_GREY, foreground=TEXT_COLOR)
        style.configure("TFrame", background=DARK_GREY)
        style.configure("TLabel", background=DARK_GREY, foreground=TEXT_COLOR)
        style.configure("TButton", background=PANEL_GREY, foreground=TEXT_COLOR)
        style.map("TButton", background=[("active", "#4A4A4A")])
        style.configure("TEntry", fieldbackground=INPUT_GREY, foreground=TEXT_COLOR)
        style.configure("TSpinbox", fieldbackground=INPUT_GREY, foreground=TEXT_COLOR)
        style.configure("TCombobox", fieldbackground=INPUT_GREY, foreground=TEXT_COLOR)
        style.configure("TNotebook", background=DARK_GREY, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_GREY, foreground=TEXT_COLOR)
        style.map("TNotebook.Tab", background=[("selected", "#4A4A4A")])
        style.configure("TLabelframe", background=DARK_GREY, foreground=TEXT_COLOR)
        style.configure("TLabelframe.Label", background=DARK_GREY, foreground=TEXT_COLOR)
        style.configure("Treeview", background=INPUT_GREY, fieldbackground=INPUT_GREY, foreground=TEXT_COLOR)
        style.map("Treeview", background=[("selected", "#4A4A4A")], foreground=[("selected", TEXT_COLOR)])
        style.configure("Treeview.Heading", background=PANEL_GREY, foreground=TEXT_COLOR)
        style.configure("Horizontal.TScrollbar", background=PANEL_GREY, troughcolor=INPUT_GREY)
        style.configure("Vertical.TScrollbar", background=PANEL_GREY, troughcolor=INPUT_GREY)

    def _apply_dark_widget_theme(self):
        dark_defaults = {
            "bg": DARK_GREY,
            "fg": TEXT_COLOR,
            "insertbackground": TEXT_COLOR,
            "highlightbackground": DARK_GREY,
            "highlightcolor": PANEL_GREY,
            "selectbackground": "#4A4A4A",
            "selectforeground": TEXT_COLOR,
        }
        input_overrides = {"bg": INPUT_GREY, "fg": TEXT_COLOR}

        def _apply(widget):
            for key, val in dark_defaults.items():
                try:
                    widget.configure(**{key: val})
                except tk.TclError:
                    pass

            if isinstance(widget, (tk.Text, tk.Listbox, tk.Entry, tk.Spinbox)):
                for key, val in input_overrides.items():
                    try:
                        widget.configure(**{key: val})
                    except tk.TclError:
                        pass

            if isinstance(widget, tk.Button):
                try:
                    widget.configure(bg=PANEL_GREY, activebackground="#4A4A4A", relief=tk.RAISED)
                except tk.TclError:
                    pass

            for child in widget.winfo_children():
                _apply(child)

        _apply(self.root)

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
