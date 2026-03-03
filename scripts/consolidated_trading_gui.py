#!/usr/bin/env python3
"""Unified GUI for Master AVWAP and BounceBot RRS controls."""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

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
        self.bot_controller = BounceBotController(self.rrs_queue)
        self._queue_after_id = None
        self._build_layout()
        self.bot_controller.start()
        self._process_rrs_queue()

    def _build_layout(self):
        main = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashrelief=tk.RAISED, bg=DARK_GREY)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main)
        self.avwap_gui = MasterAvwapGUI(top, standalone=False)
        main.add(top, stretch="always", minsize=620)

        bottom = tk.Frame(main, bg=DARK_GREY)
        controls = ttk.Frame(bottom)
        controls.pack(fill=tk.X, padx=8, pady=(8, 2))

        ttk.Label(controls, text="BounceBot:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(controls, textvariable=self.bot_controller.status_var).pack(side=tk.LEFT)
        ttk.Button(controls, text="Reconnect", command=self.bot_controller.restart).pack(side=tk.RIGHT)

        self.rrs_panel = create_rrs_confirmed_panel(
            bottom,
            bot_instance=self.bot_controller.gui_proxy,
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

        self._queue_after_id = self.root.after(150, self._process_rrs_queue)

    def on_close(self):
        try:
            if self._queue_after_id:
                self.root.after_cancel(self._queue_after_id)
            self.bot_controller.stop()
        finally:
            self.root.destroy()


class BounceBotController:
    """Lifecycle + GUI bridge for BounceBot in the consolidated window."""

    class GUIProxy:
        def __init__(self, controller: "BounceBotController"):
            self._controller = controller
            self.rrs_threshold = 2.0
            self.rrs_timeframe_key = "5m"

        def set_rrs_threshold(self, value: float) -> None:
            self.rrs_threshold = float(value)
            self._controller._forward_call("set_rrs_threshold", value)

        def set_rrs_timeframe(self, key: str) -> None:
            self.rrs_timeframe_key = key
            self._controller._forward_call("set_rrs_timeframe", key)

    def __init__(self, rrs_queue: queue.Queue):
        self.rrs_queue = rrs_queue
        self.status_var = tk.StringVar(value="starting...")
        self.bot_instance = None
        self._lock = threading.Lock()
        self.gui_proxy = self.GUIProxy(self)

    def _forward_call(self, method_name: str, *args: Any) -> None:
        with self._lock:
            bot = self.bot_instance
        if bot:
            getattr(bot, method_name)(*args)

    def _emit(self, message: str, tag: str = "rrs_status") -> None:
        self.rrs_queue.put((message, tag))
        self.status_var.set(message)

    def _make_callback(self) -> Callable[[Any, str], None]:
        def gui_callback(message, tag):
            if tag.startswith("rrs"):
                self.rrs_queue.put((message, tag))

        return gui_callback

    def start(self) -> None:
        def run_bot() -> None:
            self._emit("connecting")
            try:
                bot = run_bot_with_gui(self._make_callback())
                with self._lock:
                    self.bot_instance = bot
                self.gui_proxy.rrs_threshold = bot.rrs_threshold
                self.gui_proxy.rrs_timeframe_key = bot.rrs_timeframe_key
                self._emit("connected")
            except Exception as exc:
                self._emit(f"start failed: {exc}")

        threading.Thread(target=run_bot, daemon=True).start()

    def restart(self) -> None:
        self.stop()
        self.start()

    def stop(self) -> None:
        with self._lock:
            bot = self.bot_instance
            self.bot_instance = None
        if bot:
            try:
                bot.disconnect()
            except Exception:
                pass
        self._emit("stopped")


def launch():
    root = tk.Tk()
    app = ConsolidatedTradingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    launch()
