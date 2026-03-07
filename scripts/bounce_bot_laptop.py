#!/usr/bin/env python3
"""Lightweight BounceBot GUI for laptop use."""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import scrolledtext, ttk

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bounce_bot import MARKET_ENVIRONMENTS, RRS_TIMEFRAMES, run_bot_with_gui

DARK_GREY = "#2E2E2E"
TEXT_COLOR = "#E0E0E0"
PANEL_GREY = "#3A3A3A"
INPUT_GREY = "#252525"


class LaptopBounceBotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BounceBot Laptop")
        self.root.geometry("980x680")
        self.root.configure(bg=DARK_GREY)

        self._queue_after_id = None
        self.alert_queue: queue.Queue = queue.Queue()
        self.bot_instance = None
        self._bot_lock = threading.Lock()

        self.status_var = tk.StringVar(value="starting...")
        self.connection_var = tk.StringVar(value="IB: connecting")
        self.rrs_threshold_var = tk.DoubleVar(value=2.0)
        self.rrs_timeframe_var = tk.StringVar(value="5m")
        self.market_environment_var = tk.StringVar(value="bullish_strong")

        self._configure_theme()
        self._build_layout()
        self.start_bot()
        self.process_queue()

    def _configure_theme(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(".", background=DARK_GREY, foreground=TEXT_COLOR)
        style.configure("TFrame", background=DARK_GREY)
        style.configure("TLabel", background=DARK_GREY, foreground=TEXT_COLOR)
        style.configure("TButton", background=PANEL_GREY, foreground=TEXT_COLOR)
        style.map("TButton", background=[("active", "#4A4A4A")])
        style.configure("TNotebook", background=DARK_GREY, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_GREY, foreground=TEXT_COLOR, padding=(12, 6))
        style.map("TNotebook.Tab", background=[("selected", "#4A4A4A")])
        style.configure("Horizontal.TScrollbar", background=PANEL_GREY, troughcolor=INPUT_GREY)
        style.configure("Vertical.TScrollbar", background=PANEL_GREY, troughcolor=INPUT_GREY)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(container)
        header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(header, text="BounceBot Laptop Alerts").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.connection_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(header, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(header, text="Reconnect", command=self.restart_bot).pack(side=tk.RIGHT)
        ttk.Button(header, text="Disconnect", command=self.stop_bot).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(header, text="Clear", command=self.clear_alerts).pack(side=tk.RIGHT, padx=(0, 8))

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(controls, text="RRS Sensitivity").pack(side=tk.LEFT)
        rrs_scale = tk.Scale(
            controls,
            from_=0.0,
            to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.rrs_threshold_var,
            length=180,
            bg=DARK_GREY,
            fg=TEXT_COLOR,
            highlightthickness=0,
        )
        rrs_scale.pack(side=tk.LEFT, padx=(8, 14))
        self.rrs_threshold_var.trace_add("write", self._on_rrs_threshold_change)

        ttk.Label(controls, text="Timeframe").pack(side=tk.LEFT)
        timeframe_frame = ttk.Frame(controls)
        timeframe_frame.pack(side=tk.LEFT, padx=(8, 14))
        for key in ("5m", "15m", "30m", "1h"):
            label = RRS_TIMEFRAMES[key]["label"]
            tk.Radiobutton(
                timeframe_frame,
                text=label,
                variable=self.rrs_timeframe_var,
                value=key,
                indicatoron=0,
                command=self._on_rrs_timeframe_change,
                padx=6,
                pady=2,
                bg=DARK_GREY,
                fg=TEXT_COLOR,
                selectcolor="#444444",
                activebackground="#444444",
                activeforeground=TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=2)

        env_frame = ttk.Frame(container)
        env_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(env_frame, text="Market Environment").pack(side=tk.LEFT, padx=(0, 8))
        for key, info in MARKET_ENVIRONMENTS.items():
            tk.Radiobutton(
                env_frame,
                text=info["label"],
                variable=self.market_environment_var,
                value=key,
                indicatoron=0,
                command=self._on_market_environment_change,
                padx=8,
                pady=3,
                bg=DARK_GREY,
                fg=TEXT_COLOR,
                selectcolor="#444444",
                activebackground="#444444",
                activeforeground=TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=2)

        notebook = ttk.Notebook(container)
        notebook.pack(fill=tk.BOTH, expand=True)

        alerts_tab = ttk.Frame(notebook)
        notebook.add(alerts_tab, text="BounceBot Alerts")

        self.text_area = scrolledtext.ScrolledText(
            alerts_tab,
            wrap=tk.WORD,
            font=("Courier", 11),
            state="disabled",
            bg=INPUT_GREY,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)

        self.text_area.tag_config("green", foreground="#50FA7B", font=("Courier", 11))
        self.text_area.tag_config("red", foreground="#FF5555", font=("Courier", 11))
        self.text_area.tag_config("pink_symbol", foreground="#FF79C6", font=("Courier", 11, "bold"))
        self.text_area.tag_config("orange_symbol", foreground="#FFB86C", font=("Courier", 11, "bold"))
        self.text_area.tag_config("blue", foreground="#8BE9FD", font=("Courier", 11))
        self.text_area.tag_config("candle_line", foreground="#BD93F9", overstrike=1)

    def _emit_status(self, message: str, connected: bool | None = None) -> None:
        self.status_var.set(message)
        if connected is True:
            self.connection_var.set("IB: connected")
        elif connected is False:
            self.connection_var.set("IB: disconnected")

    def _make_callback(self):
        def gui_callback(message, tag):
            if tag.startswith("rrs"):
                return
            if tag == "approaching" or tag.startswith("approaching_"):
                return
            if tag == "blue" and "removed from" in str(message):
                return
            self.alert_queue.put((str(message), str(tag)))

        return gui_callback

    def _with_bot(self, callback) -> None:
        with self._bot_lock:
            bot = self.bot_instance
        if bot:
            callback(bot)

    def _sync_controls_from_bot(self, bot) -> None:
        self.root.after(0, lambda: self.rrs_threshold_var.set(float(bot.rrs_threshold)))
        self.root.after(0, lambda: self.rrs_timeframe_var.set(str(bot.rrs_timeframe_key)))
        self.root.after(0, lambda: self.market_environment_var.set(str(bot.get_market_environment())))

    def _on_rrs_threshold_change(self, *_args) -> None:
        value = float(self.rrs_threshold_var.get())
        self._with_bot(lambda bot: bot.set_rrs_threshold(value))

    def _on_rrs_timeframe_change(self) -> None:
        key = self.rrs_timeframe_var.get()
        self._with_bot(lambda bot: bot.set_rrs_timeframe(key))
        self._emit_status(f"RRS timeframe set to {RRS_TIMEFRAMES[key]['label']}")

    def _on_market_environment_change(self) -> None:
        env_key = self.market_environment_var.get()
        self._with_bot(lambda bot: bot.set_market_environment(env_key))
        self._emit_status(f"Environment: {MARKET_ENVIRONMENTS[env_key]['label']}")

    def start_bot(self) -> None:
        def run_bot() -> None:
            self._emit_status("starting...", connected=False)
            try:
                bot = run_bot_with_gui(self._make_callback())
                with self._bot_lock:
                    self.bot_instance = bot
                self._sync_controls_from_bot(bot)
                self._emit_status("listening for alerts", connected=True)
            except Exception as exc:
                self._emit_status(f"start failed: {exc}", connected=False)

        threading.Thread(target=run_bot, daemon=True).start()

    def restart_bot(self) -> None:
        self.stop_bot()
        self.start_bot()

    def stop_bot(self) -> None:
        with self._bot_lock:
            bot = self.bot_instance
            self.bot_instance = None
        if bot:
            try:
                bot.disconnect()
            except Exception:
                pass
        self._emit_status("stopped", connected=False)

    def clear_alerts(self) -> None:
        self.text_area.config(state="normal")
        self.text_area.delete("1.0", tk.END)
        self.text_area.config(state="disabled")

    def append_alert(self, message: str, tag: str) -> None:
        self.text_area.config(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")

        if "Bounce confirmed" in message:
            parts = message.split(":", 1)
            if len(parts) == 2:
                symbol = parts[0].strip()
                rest = ":" + parts[1]
                self.text_area.insert(tk.END, f"{timestamp} - ", tag)
                if "(long)" in rest:
                    self.text_area.insert(tk.END, symbol, "pink_symbol")
                    self.text_area.insert(tk.END, rest + "\n", "green")
                elif "(short)" in rest:
                    self.text_area.insert(tk.END, symbol, "orange_symbol")
                    self.text_area.insert(tk.END, rest + "\n", "red")
                else:
                    self.text_area.insert(tk.END, f"{message}\n", tag)
            else:
                self.text_area.insert(tk.END, f"{timestamp} - {message}\n", tag)
        else:
            self.text_area.insert(tk.END, f"{timestamp} - {message}\n", tag)

        self.text_area.config(state="disabled")
        self.text_area.see(tk.END)

    def process_queue(self) -> None:
        while True:
            try:
                message, tag = self.alert_queue.get_nowait()
            except queue.Empty:
                break
            self.append_alert(message, tag)

        self._queue_after_id = self.root.after(150, self.process_queue)

    def on_close(self) -> None:
        try:
            if self._queue_after_id:
                self.root.after_cancel(self._queue_after_id)
            self.stop_bot()
        finally:
            self.root.destroy()


def launch() -> None:
    root = tk.Tk()
    app = LaptopBounceBotApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    launch()
