#!/usr/bin/env python3
"""Unified GUI with simple and full workspaces for BounceBot and Master AVWAP."""

from __future__ import annotations

import argparse
import os
import re
import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

from project_paths import (
    LOCAL_SETTINGS_FILE,
    LONGS_FILE,
    SHORTS_FILE,
    get_shared_watchlist_details,
    get_tracker_storage_details,
    get_local_setting,
    open_path_in_file_manager,
    save_tracker_storage_dir,
    save_local_setting,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ROOT_DIR = SCRIPT_DIR.parent
WATCHLIST_SYMBOL_RE = re.compile(r"[A-Z0-9.\-]+")

from bounce_bot import (
    BOUNCE_TYPE_DEFAULTS,
    BOUNCE_TYPE_LABELS,
    MARKET_ENVIRONMENTS,
    RRS_TIMEFRAMES,
    append_alert_message,
    configure_alert_tags,
    create_rrs_confirmed_panel,
    run_bot_with_gui,
)
from master_avwap import (
    EVENT_TICKERS_FILE,
    PRIORITY_SETUPS_FILE,
    STDEV_RANGE_FILE,
    MasterAvwapGUI,
    run_master,
    run_master_with_shared_watchlists,
)

DARK_GREY = "#2E2E2E"
TEXT_COLOR = "#E0E0E0"
PANEL_GREY = "#3A3A3A"
INPUT_GREY = "#252525"

BOUNCE_TOGGLE_ORDER = [
    "10_candle",
    "vwap",
    "dynamic_vwap",
    "eod_vwap",
    "vwap_eod_confluence",
    "impulse_retest_vwap_eod",
    "ema_8",
    "ema_15",
    "ema_21",
    "vwap_upper_band",
    "vwap_lower_band",
    "dynamic_vwap_upper_band",
    "dynamic_vwap_lower_band",
    "eod_vwap_upper_band",
    "eod_vwap_lower_band",
    "prev_day_high",
    "prev_day_low",
]


def _open_folder(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        open_path_in_file_manager(path)
    except Exception as exc:
        messagebox.showerror("Open Folder", f"Could not open folder:\n{path}\n\n{exc}")


class TrackerStorageControls:
    def __init__(self, parent: tk.Misc, compact: bool = False):
        self.parent = parent
        self.compact = compact
        self.info_var = tk.StringVar()
        self._build(parent)
        self.refresh()

    def _build(self, parent: tk.Misc) -> None:
        container = ttk.LabelFrame(parent, text="Home Folder")
        container.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.container = container

        description = (
            "Use a Google Drive or OneDrive folder here so watchlists, caches, reports, logs, and AVWAP tracker data stay in sync across devices."
        )
        ttk.Label(container, text=description, wraplength=900, justify=tk.LEFT).pack(
            anchor="w",
            padx=10,
            pady=(8, 6),
        )

        self.info_label = ttk.Label(container, textvariable=self.info_var, justify=tk.LEFT, wraplength=900)
        self.info_label.pack(anchor="w", padx=10, pady=(0, 6))

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Button(button_row, text="Change Home Folder", command=self.choose_folder).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Open Home Folder", command=self.open_folder).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Open Settings File", command=self.open_settings_file).pack(side=tk.LEFT, padx=(8, 0))

        if not self.compact:
            hint = ttk.Label(
                container,
                text="If you change this setting, restart the GUI so every AVWAP tab picks up the new location.",
                wraplength=900,
                justify=tk.LEFT,
            )
            hint.pack(anchor="w", padx=10, pady=(0, 8))

    def refresh(self) -> None:
        details = get_tracker_storage_details()
        shared_watchlists = get_shared_watchlist_details()
        self.shared_root_dir = Path(details["data_dir"])
        self.settings_file = Path(details["settings_file"])
        self.info_var.set(
            f"Home folder: {details['data_dir']}\n"
            f"Mutable data: {details['mutable_data_dir']}\n"
            f"Logs: {details['logs_dir']}\n"
            f"Reports: {details['output_dir']}\n"
            f"Runtime tracker data: {details['runtime_dir']}\n"
            f"Home-folder longs.txt: {shared_watchlists['longs_path']} ({shared_watchlists['longs_exists']})\n"
            f"Home-folder shorts.txt: {shared_watchlists['shorts_path']} ({shared_watchlists['shorts_exists']})\n"
            f"Source: {details['source_label']}"
        )

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose home folder",
            initialdir=str(self.shared_root_dir if self.shared_root_dir.exists() else Path.home()),
            mustexist=False,
        )
        if not selected:
            return
        target = save_tracker_storage_dir(selected)
        self.refresh()
        messagebox.showinfo(
            "Home Folder Saved",
            "Saved this computer's home folder.\n\n"
            f"Folder: {target}\n"
            f"Settings file: {LOCAL_SETTINGS_FILE}\n\n"
            "Place longs.txt and shorts.txt in that folder root to share watchlists across devices.\n\n"
            "Restart the GUI to start using the new home folder.",
        )

    def open_folder(self) -> None:
        _open_folder(self.shared_root_dir)

    def open_settings_file(self) -> None:
        _open_folder(self.settings_file.parent)


def configure_theme(root: tk.Misc) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")

    root.option_add("*Background", DARK_GREY)
    root.option_add("*Foreground", TEXT_COLOR)
    root.option_add("*Label*Background", DARK_GREY)
    root.option_add("*Entry*Background", INPUT_GREY)
    root.option_add("*Entry*Foreground", TEXT_COLOR)
    root.option_add("*Text*Background", INPUT_GREY)
    root.option_add("*Text*Foreground", TEXT_COLOR)

    style.configure(".", background=DARK_GREY, foreground=TEXT_COLOR)
    style.configure("TFrame", background=DARK_GREY)
    style.configure("TLabel", background=DARK_GREY, foreground=TEXT_COLOR)
    style.configure("TButton", background=PANEL_GREY, foreground=TEXT_COLOR)
    style.map("TButton", background=[("active", "#4A4A4A")])
    style.configure("TEntry", fieldbackground=INPUT_GREY, foreground=TEXT_COLOR)
    style.configure("TSpinbox", fieldbackground=INPUT_GREY, foreground=TEXT_COLOR)
    style.configure("TCombobox", fieldbackground=INPUT_GREY, foreground=TEXT_COLOR)
    style.configure("TNotebook", background=DARK_GREY, borderwidth=0)
    style.configure("TNotebook.Tab", background=PANEL_GREY, foreground=TEXT_COLOR, padding=(12, 6))
    style.map("TNotebook.Tab", background=[("selected", "#4A4A4A")])
    style.configure("TLabelframe", background=DARK_GREY, foreground=TEXT_COLOR)
    style.configure("TLabelframe.Label", background=DARK_GREY, foreground=TEXT_COLOR)
    style.configure("Treeview", background=INPUT_GREY, fieldbackground=INPUT_GREY, foreground=TEXT_COLOR)
    style.map("Treeview", background=[("selected", "#4A4A4A")], foreground=[("selected", TEXT_COLOR)])
    style.configure("Treeview.Heading", background=PANEL_GREY, foreground=TEXT_COLOR)
    style.configure("Horizontal.TScrollbar", background=PANEL_GREY, troughcolor=INPUT_GREY)
    style.configure("Vertical.TScrollbar", background=PANEL_GREY, troughcolor=INPUT_GREY)


def choose_gui_mode() -> str:
    preferred_mode = str(get_local_setting("gui_mode", "full") or "full").strip().lower()
    if preferred_mode not in {"full", "simple"}:
        preferred_mode = "full"
    selection = {"mode": preferred_mode}

    picker = tk.Tk()
    picker.title("Consolidated GUI Mode")
    picker.geometry("400x170")
    picker.configure(bg=DARK_GREY)
    picker.resizable(False, False)

    tk.Label(
        picker,
        text="Choose startup mode",
        bg=DARK_GREY,
        fg=TEXT_COLOR,
        font=("Arial", 12, "bold"),
    ).pack(pady=(18, 10))

    tk.Label(
        picker,
        text=(
            "Full mode uses the full BounceBot layout.\n"
            "Simple mode uses the laptop BounceBot layout.\n"
            "Master AVWAP stays full in both modes.\n"
            f"Default on this computer: {preferred_mode.title()}"
        ),
        bg=DARK_GREY,
        fg=TEXT_COLOR,
        justify=tk.CENTER,
    ).pack(pady=(0, 14))

    button_row = tk.Frame(picker, bg=DARK_GREY)
    button_row.pack()

    def select_mode(mode: str) -> None:
        selection["mode"] = mode
        save_local_setting("gui_mode", mode)
        picker.destroy()

    tk.Button(button_row, text="Full", width=12, command=lambda: select_mode("full")).pack(side=tk.LEFT, padx=8)
    tk.Button(button_row, text="Simple", width=12, command=lambda: select_mode("simple")).pack(side=tk.LEFT, padx=8)

    picker.protocol("WM_DELETE_WINDOW", lambda: select_mode("full"))
    picker.mainloop()
    return selection["mode"]


class BounceBotController:
    class GUIProxy:
        def __init__(self, controller: "BounceBotController"):
            self._controller = controller
            self.rrs_threshold = controller.rrs_threshold
            self.rrs_timeframe_key = controller.rrs_timeframe_key

        def set_rrs_threshold(self, value: float) -> None:
            self._controller.set_rrs_threshold(value)

        def set_rrs_timeframe(self, key: str) -> None:
            self._controller.set_rrs_timeframe(key)

        def set_market_environment(self, env_key: str) -> None:
            self._controller.set_market_environment(env_key)

        def get_market_environment(self) -> str:
            return self._controller.get_market_environment()

    def __init__(
        self,
        include_approaching: bool,
        ui_parent: tk.Misc,
        *,
        start_scanning_enabled: bool = False,
    ):
        self.include_approaching = include_approaching
        self.ui_parent = ui_parent
        self.rrs_queue: queue.Queue = queue.Queue()
        self.bounce_queue: queue.Queue = queue.Queue()
        self.status_var = tk.StringVar(value="starting...")
        self.connection_var = tk.StringVar(value="IB: disconnected")
        self.active_bounce_var = tk.StringVar(value="active bounces: 0")
        self.bot_instance = None
        self._lock = threading.Lock()

        self.rrs_threshold = 2.0
        self.rrs_timeframe_key = "5m"
        self.market_environment = "bullish_strong"
        self.scanning_enabled = bool(start_scanning_enabled)
        self.bounce_type_settings = dict(BOUNCE_TYPE_DEFAULTS)
        self.gui_proxy = self.GUIProxy(self)

    def _run_on_ui_thread(self, callback) -> None:
        if threading.current_thread() is threading.main_thread():
            callback()
            return
        try:
            self.ui_parent.after(0, callback)
        except RuntimeError:
            pass

    def _set_var(self, variable: tk.Variable, value: str) -> None:
        self._run_on_ui_thread(lambda: variable.set(value))

    def _with_bot(self, callback):
        with self._lock:
            bot = self.bot_instance
        if bot is None:
            return None
        return callback(bot)

    def _sync_state_from_bot(self, bot) -> None:
        self.rrs_threshold = float(getattr(bot, "rrs_threshold", self.rrs_threshold))
        self.rrs_timeframe_key = str(getattr(bot, "rrs_timeframe_key", self.rrs_timeframe_key))
        self.market_environment = str(bot.get_market_environment())
        self.scanning_enabled = bool(bot.is_scanning_enabled())
        for bounce_key in self.bounce_type_settings:
            self.bounce_type_settings[bounce_key] = bool(bot.is_bounce_type_enabled(bounce_key))
        self.gui_proxy.rrs_threshold = self.rrs_threshold
        self.gui_proxy.rrs_timeframe_key = self.rrs_timeframe_key

    def _apply_saved_state(self, bot) -> None:
        bot.set_rrs_threshold(self.rrs_threshold)
        bot.set_rrs_timeframe(self.rrs_timeframe_key)
        bot.set_market_environment(self.market_environment)
        bot.set_scanning_enabled(self.scanning_enabled)
        for bounce_key, enabled in self.bounce_type_settings.items():
            bot.set_bounce_type_enabled(bounce_key, enabled)

    def _emit(self, message: str) -> None:
        self._set_var(self.status_var, message)

    def _make_callback(self):
        def gui_callback(message, tag):
            if tag.startswith("rrs"):
                self.rrs_queue.put((message, tag))
                return
            if tag == "blue" and "removed from" in str(message):
                return
            if not self.include_approaching and (tag == "approaching" or str(tag).startswith("approaching_")):
                return
            self.bounce_queue.put((message, tag))

        return gui_callback

    def start(self) -> None:
        def run_bot() -> None:
            self._emit("connecting")
            self._set_var(self.connection_var, "IB: connecting")
            try:
                bot = run_bot_with_gui(
                    self._make_callback(),
                    start_scanning_enabled=self.scanning_enabled,
                )
                self._apply_saved_state(bot)
                self._sync_state_from_bot(bot)
                with self._lock:
                    self.bot_instance = bot
                self._run_on_ui_thread(self.refresh_active_bounces)
                self._set_var(self.connection_var, "IB: connected")
                self._emit("connected")
            except Exception as exc:
                self._set_var(self.connection_var, "IB: disconnected")
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
        self._set_var(self.connection_var, "IB: disconnected")
        self._set_var(self.active_bounce_var, "active bounces: 0")
        self._emit("stopped")

    def set_rrs_threshold(self, value: float) -> None:
        self.rrs_threshold = float(value)
        self.gui_proxy.rrs_threshold = self.rrs_threshold
        self._with_bot(lambda bot: bot.set_rrs_threshold(self.rrs_threshold))

    def set_rrs_timeframe(self, key: str) -> None:
        if key not in RRS_TIMEFRAMES:
            return
        self.rrs_timeframe_key = key
        self.gui_proxy.rrs_timeframe_key = key
        self._with_bot(lambda bot: bot.set_rrs_timeframe(key))

    def set_market_environment(self, env_key: str) -> None:
        if env_key not in MARKET_ENVIRONMENTS:
            return
        self.market_environment = env_key
        self._with_bot(lambda bot: bot.set_market_environment(env_key))

    def get_market_environment(self) -> str:
        return self.market_environment

    def set_bounce_type_enabled(self, bounce_type: str, enabled: bool) -> None:
        if bounce_type not in self.bounce_type_settings:
            return
        self.bounce_type_settings[bounce_type] = bool(enabled)
        self._with_bot(lambda bot: bot.set_bounce_type_enabled(bounce_type, enabled))

    def is_bounce_type_enabled(self, bounce_type: str) -> bool:
        return bool(self.bounce_type_settings.get(bounce_type, False))

    def set_scanning_enabled(self, enabled: bool) -> None:
        self.scanning_enabled = bool(enabled)
        self._with_bot(lambda bot: bot.set_scanning_enabled(self.scanning_enabled))

    def is_scanning_enabled(self) -> bool:
        return self.scanning_enabled

    def start_scanning(self) -> None:
        self.set_scanning_enabled(True)

    def stop_scanning(self) -> None:
        self.set_scanning_enabled(False)

    def refresh_active_bounces(self) -> None:
        def _read_count(bot):
            return len(bot.find_active_master_avwap_bounces())

        count = self._with_bot(_read_count)
        if count is None:
            self._set_var(self.active_bounce_var, "active bounces: 0")
            return
        self._set_var(self.active_bounce_var, f"active bounces: {count}")

    def run_manual_check(self, method_name: str, heading: str) -> None:
        def worker() -> None:
            with self._lock:
                bot = self.bot_instance
            if bot is None:
                self.bounce_queue.put((f"{heading}: BounceBot not connected.", "red"))
                return
            try:
                results = getattr(bot, method_name)()
            except Exception as exc:
                self.bounce_queue.put((f"{heading}: {exc}", "red"))
                return

            self.bounce_queue.put((f"=== {heading} ===", "blue"))
            if results:
                for result in results:
                    self.bounce_queue.put((str(result), "green"))
            else:
                self.bounce_queue.put(("No symbols flagged.", "blue"))

        threading.Thread(target=worker, daemon=True).start()


class BaseBounceBotPanel:
    def __init__(self, parent: tk.Misc, controller: BounceBotController, switch_mode_callback=None):
        self.parent = parent
        self.controller = controller
        self.switch_mode_callback = switch_mode_callback
        self.container = ttk.Frame(parent)
        self._queue_after_id = None
        self.alert_text: scrolledtext.ScrolledText | None = None

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _create_alerts_widget(self, parent: tk.Misc, font_size: int) -> scrolledtext.ScrolledText:
        text_area = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=("Courier", font_size),
            state="disabled",
            bg=INPUT_GREY,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
        )
        configure_alert_tags(text_area, font_size=font_size)
        return text_area

    def _append_alert_with_timestamp(self, message: str, tag: str) -> None:
        if self.alert_text is None:
            return
        self.alert_text.config(state="normal")
        append_alert_message(
            self.alert_text,
            message,
            tag,
            datetime.now().strftime("%H:%M:%S"),
        )
        self.alert_text.config(state="disabled")
        self.alert_text.see(tk.END)

    def clear_alerts(self) -> None:
        if self.alert_text is None:
            return
        self.alert_text.config(state="normal")
        self.alert_text.delete("1.0", tk.END)
        self.alert_text.config(state="disabled")

    def start(self) -> None:
        self.controller.start()
        self._process_queues()

    def _process_queues(self) -> None:
        raise NotImplementedError

    def on_close(self) -> None:
        try:
            if self._queue_after_id:
                self.container.after_cancel(self._queue_after_id)
        except Exception:
            pass
        self.controller.stop()


class SimpleBounceBotPanel(BaseBounceBotPanel):
    def __init__(self, parent: tk.Misc, switch_mode_callback=None):
        super().__init__(
            parent,
            BounceBotController(
                include_approaching=False,
                ui_parent=parent,
                start_scanning_enabled=False,
            ),
            switch_mode_callback=switch_mode_callback,
        )
        self._syncing_controls = False
        self.toggle_vars: dict[str, tk.BooleanVar] = {}
        self.rrs_threshold_var = tk.DoubleVar(value=self.controller.rrs_threshold)
        self.timeframe_var = tk.StringVar(value=self.controller.rrs_timeframe_key)
        self.environment_var = tk.StringVar(value=self.controller.get_market_environment())
        self._build_layout()
        self.start()

    def _build_layout(self) -> None:
        header = ttk.Frame(self.container)
        header.pack(fill=tk.X, padx=10, pady=(10, 8))

        ttk.Label(header, text="BounceBot Simple").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.controller.connection_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(header, textvariable=self.controller.status_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(header, textvariable=self.controller.active_bounce_var).pack(side=tk.LEFT, padx=(12, 0))
        if self.switch_mode_callback:
            ttk.Button(header, text="Switch to Full", command=self.switch_mode_callback).pack(side=tk.RIGHT)
        ttk.Button(header, text="Reconnect", command=self.controller.restart).pack(side=tk.RIGHT)
        self.stop_scanning_button = ttk.Button(header, text="Stop Scanning", command=self.controller.stop_scanning)
        self.stop_scanning_button.pack(side=tk.RIGHT, padx=(0, 8))
        self.start_scanning_button = ttk.Button(header, text="Start Scanning", command=self.controller.start_scanning)
        self.start_scanning_button.pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(header, text="Disconnect", command=self.controller.stop).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(header, text="Clear", command=self.clear_alerts).pack(side=tk.RIGHT, padx=(0, 8))

        controls = tk.Frame(self.container, bg=DARK_GREY)
        controls.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Label(controls, text="RRS Sensitivity", bg=DARK_GREY, fg=TEXT_COLOR).pack(side=tk.LEFT)
        self.rrs_threshold_var.trace_add("write", self._on_rrs_threshold_change)
        tk.Scale(
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
        ).pack(side=tk.LEFT, padx=(8, 14))

        tk.Label(controls, text="Timeframe", bg=DARK_GREY, fg=TEXT_COLOR).pack(side=tk.LEFT)
        for key in ("5m", "15m", "30m", "1h"):
            tk.Radiobutton(
                controls,
                text=RRS_TIMEFRAMES[key]["label"],
                variable=self.timeframe_var,
                value=key,
                indicatoron=0,
                command=self._on_timeframe_change,
                padx=6,
                pady=2,
                bg=DARK_GREY,
                fg=TEXT_COLOR,
                selectcolor="#444444",
                activebackground="#444444",
                activeforeground=TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=2)

        env_frame = tk.Frame(self.container, bg=DARK_GREY)
        env_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Label(env_frame, text="Market Environment", bg=DARK_GREY, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=(0, 8))
        for key, info in MARKET_ENVIRONMENTS.items():
            tk.Radiobutton(
                env_frame,
                text=info["label"],
                variable=self.environment_var,
                value=key,
                indicatoron=0,
                command=self._on_environment_change,
                padx=8,
                pady=3,
                bg=DARK_GREY,
                fg=TEXT_COLOR,
                selectcolor="#444444",
                activebackground="#444444",
                activeforeground=TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=2)

        alerts_frame = ttk.Frame(self.container)
        alerts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.alert_text = self._create_alerts_widget(alerts_frame, font_size=11)
        self.alert_text.pack(fill=tk.BOTH, expand=True)

        bounce_toggle_frame = tk.LabelFrame(
            self.container,
            text="Bounce Filters",
            bg=DARK_GREY,
            fg=TEXT_COLOR,
            padx=8,
            pady=6,
            highlightbackground="#444444",
            highlightcolor="#444444",
        )
        bounce_toggle_frame.pack(fill=tk.X, padx=10, pady=(0, 10), side=tk.BOTTOM)

        for idx, bounce_key in enumerate(BOUNCE_TOGGLE_ORDER):
            var = tk.BooleanVar(value=self.controller.is_bounce_type_enabled(bounce_key))
            self.toggle_vars[bounce_key] = var
            tk.Checkbutton(
                bounce_toggle_frame,
                text=BOUNCE_TYPE_LABELS.get(bounce_key, bounce_key),
                variable=var,
                command=lambda k=bounce_key, v=var: self.controller.set_bounce_type_enabled(k, bool(v.get())),
                bg=DARK_GREY,
                fg=TEXT_COLOR,
                selectcolor="#444444",
                activebackground="#444444",
                activeforeground=TEXT_COLOR,
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=6, pady=2)

    def _sync_controls_from_controller(self) -> None:
        self._syncing_controls = True
        try:
            if float(self.rrs_threshold_var.get()) != float(self.controller.rrs_threshold):
                self.rrs_threshold_var.set(self.controller.rrs_threshold)
            if self.timeframe_var.get() != self.controller.rrs_timeframe_key:
                self.timeframe_var.set(self.controller.rrs_timeframe_key)
            if self.environment_var.get() != self.controller.get_market_environment():
                self.environment_var.set(self.controller.get_market_environment())
            for bounce_key, var in self.toggle_vars.items():
                expected = self.controller.is_bounce_type_enabled(bounce_key)
                if bool(var.get()) != bool(expected):
                    var.set(expected)
        finally:
            self._syncing_controls = False
        self._sync_scanning_controls()

    def _sync_scanning_controls(self) -> None:
        scanning_enabled = self.controller.is_scanning_enabled()
        self.start_scanning_button.configure(state=("disabled" if scanning_enabled else "normal"))
        self.stop_scanning_button.configure(state=("normal" if scanning_enabled else "disabled"))

    def _on_rrs_threshold_change(self, *_args) -> None:
        if self._syncing_controls:
            return
        self.controller.set_rrs_threshold(self.rrs_threshold_var.get())

    def _on_timeframe_change(self) -> None:
        if self._syncing_controls:
            return
        self.controller.set_rrs_timeframe(self.timeframe_var.get())

    def _on_environment_change(self) -> None:
        if self._syncing_controls:
            return
        self.controller.set_market_environment(self.environment_var.get())

    def _process_queues(self) -> None:
        self._sync_controls_from_controller()

        while True:
            try:
                message, tag = self.controller.rrs_queue.get_nowait()
            except queue.Empty:
                break
            if tag == "rrs_status":
                self.controller.status_var.set(str(message))

        while True:
            try:
                message, tag = self.controller.bounce_queue.get_nowait()
            except queue.Empty:
                break
            self._append_alert_with_timestamp(str(message), str(tag))

        self.controller.refresh_active_bounces()
        self._queue_after_id = self.container.after(150, self._process_queues)


class FullBounceBotPanel(BaseBounceBotPanel):
    def __init__(self, parent: tk.Misc, switch_mode_callback=None):
        super().__init__(
            parent,
            BounceBotController(
                include_approaching=False,
                ui_parent=parent,
                start_scanning_enabled=False,
            ),
            switch_mode_callback=switch_mode_callback,
        )
        self.toggle_vars: dict[str, tk.BooleanVar] = {}
        self._build_layout()
        self.start()

    def _build_layout(self) -> None:
        header = ttk.Frame(self.container)
        header.pack(fill=tk.X, padx=10, pady=(10, 8))

        ttk.Label(header, text="BounceBot Full").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.controller.connection_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(header, textvariable=self.controller.status_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(header, textvariable=self.controller.active_bounce_var).pack(side=tk.LEFT, padx=(12, 0))
        controls_frame = ttk.Frame(header)
        controls_frame.pack(side=tk.RIGHT)
        if self.switch_mode_callback:
            ttk.Button(controls_frame, text="Switch to Simple", command=self.switch_mode_callback).pack(side=tk.LEFT, padx=(0, 8))
        self.start_scanning_button = tk.Button(
            controls_frame,
            text="Start Scanning",
            command=self.controller.start_scanning,
            relief=tk.RAISED,
            padx=10,
            bg=PANEL_GREY,
            fg=TEXT_COLOR,
        )
        self.start_scanning_button.pack(side=tk.LEFT)
        self.stop_scanning_button = tk.Button(
            controls_frame,
            text="Stop Scanning",
            command=self.controller.stop_scanning,
            relief=tk.RAISED,
            padx=10,
            bg=PANEL_GREY,
            fg=TEXT_COLOR,
        )
        self.stop_scanning_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls_frame, text="Clear", command=self.clear_alerts).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls_frame, text="Reconnect", command=self.controller.restart).pack(side=tk.LEFT, padx=(8, 0))

        content_pane = tk.PanedWindow(
            self.container,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=10,
            showhandle=True,
            bg=DARK_GREY,
        )
        content_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        alerts_frame = tk.Frame(content_pane, bg=DARK_GREY)
        tk.Label(alerts_frame, text="BounceBot Alerts", bg=DARK_GREY, fg=TEXT_COLOR).pack(anchor="w", padx=5, pady=(6, 2))
        self.alert_text = self._create_alerts_widget(alerts_frame, font_size=11)
        self.alert_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        content_pane.add(alerts_frame, stretch="always")

        self.rrs_panel = create_rrs_confirmed_panel(
            content_pane,
            bot_instance=self.controller.gui_proxy,
            dark_grey=DARK_GREY,
            text_color=TEXT_COLOR,
        )
        content_pane.add(self.rrs_panel["container"], stretch="always")

        bounce_toggle_frame = tk.LabelFrame(
            self.container,
            text="Bounce Filters",
            bg=DARK_GREY,
            fg=TEXT_COLOR,
            padx=8,
            pady=6,
            highlightbackground="#444444",
            highlightcolor="#444444",
        )
        bounce_toggle_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        for idx, bounce_key in enumerate(BOUNCE_TOGGLE_ORDER):
            var = tk.BooleanVar(value=self.controller.is_bounce_type_enabled(bounce_key))
            self.toggle_vars[bounce_key] = var
            tk.Checkbutton(
                bounce_toggle_frame,
                text=BOUNCE_TYPE_LABELS.get(bounce_key, bounce_key),
                variable=var,
                command=lambda k=bounce_key, v=var: self.controller.set_bounce_type_enabled(k, bool(v.get())),
                bg=DARK_GREY,
                fg=TEXT_COLOR,
                selectcolor="#444444",
                activebackground="#444444",
                activeforeground=TEXT_COLOR,
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=6, pady=2)

    def _sync_toggle_state(self) -> None:
        for bounce_key, var in self.toggle_vars.items():
            expected = self.controller.is_bounce_type_enabled(bounce_key)
            if bool(var.get()) != bool(expected):
                var.set(expected)
        scanning_enabled = self.controller.is_scanning_enabled()
        self.start_scanning_button.configure(state=(tk.DISABLED if scanning_enabled else tk.NORMAL))
        self.stop_scanning_button.configure(state=(tk.NORMAL if scanning_enabled else tk.DISABLED))

    def _process_queues(self) -> None:
        self._sync_toggle_state()

        while True:
            try:
                message, tag = self.controller.rrs_queue.get_nowait()
            except queue.Empty:
                break
            if tag == "rrs_status":
                self.controller.status_var.set(str(message))
                self.rrs_panel["rrs_status_var"].set(str(message))
            elif tag == "rrs_snapshot":
                self.rrs_panel["render_rrs_snapshot"](message)

        while True:
            try:
                message, tag = self.controller.bounce_queue.get_nowait()
            except queue.Empty:
                break
            self._append_alert_with_timestamp(str(message), str(tag))

        self.controller.refresh_active_bounces()
        self._queue_after_id = self.container.after(150, self._process_queues)


class SimpleMasterAvwapPanel:
    def __init__(self, parent: tk.Misc):
        self.parent = parent
        self.container = ttk.Frame(parent)
        self.status_var = tk.StringVar(value="Ready")
        self._build_layout()
        self.refresh_output_view()

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self.container)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 8))

        ttk.Label(toolbar, text="Master AVWAP Simple").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Run Shared Watchlist Scan", command=self.run_master_once).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(toolbar, text="Run Local Watchlist Scan", command=self.run_local_watchlists_once).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Refresh Output", command=self.refresh_output_view).pack(side=tk.LEFT, padx=4)

        hint = ttk.Label(
            self.container,
            text="Focused on longs.txt / shorts.txt AVWAP event searches. Shared home-folder watchlists are the default here.",
        )
        hint.pack(anchor="w", padx=10, pady=(0, 8))

        self.tracker_storage_controls = TrackerStorageControls(self.container, compact=True)

        self.text_area = scrolledtext.ScrolledText(
            self.container,
            wrap=tk.WORD,
            font=("Courier New", 10),
            bg=INPUT_GREY,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        status = ttk.Label(self.container, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _read_text_file(self, path: Path) -> str:
        if not path.exists():
            return f"[Missing file] {path.name}"
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            return f"[Error reading {path.name}] {exc}"

    def refresh_output_view(self) -> None:
        combined = (
            "MASTER AVWAP PRIORITY SETUPS\n"
            + "=" * 80
            + "\n"
            + (self._read_text_file(PRIORITY_SETUPS_FILE) or "No priority setup output yet.")
            + "\n\n"
            + "MASTER AVWAP EVENT TICKERS\n"
            + "=" * 80
            + "\n"
            + (self._read_text_file(EVENT_TICKERS_FILE) or "No event ticker output yet.")
            + "\n\n"
            + "MASTER AVWAP STDEV 2-3 OUTPUT\n"
            + "=" * 80
            + "\n"
            + (self._read_text_file(STDEV_RANGE_FILE) or "No stdev output yet.")
            + "\n"
        )
        self.text_area.configure(state="normal")
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert("1.0", combined)
        self.text_area.configure(state="disabled")

    def _run_background(self, target, running_msg: str, done_msg: str) -> None:
        self.status_var.set(running_msg)

        def task() -> None:
            try:
                target()
                self.container.after(0, lambda: self.status_var.set(done_msg))
                self.container.after(0, self.refresh_output_view)
            except Exception as exc:
                self.container.after(0, lambda: self.status_var.set(f"Error: {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def run_master_once(self) -> None:
        self._run_background(
            run_master_with_shared_watchlists,
            "Running Master AVWAP scan from shared home-folder longs.txt / shorts.txt...",
            "Shared-watchlist Master AVWAP scan complete.",
        )

    def run_local_watchlists_once(self) -> None:
        self._run_background(
            run_master,
            "Running Master AVWAP scan from local project watchlists...",
            "Local-watchlist Master AVWAP scan complete.",
        )


class WatchlistEditorPanel:
    def __init__(self, parent: tk.Misc, title: str, path: Path, on_symbols_saved):
        self.parent = parent
        self.title = title
        self.path = path
        self.on_symbols_saved = on_symbols_saved
        self.container = ttk.Frame(parent)
        self._loading = False
        self._save_after_id = None
        self.add_symbol_var = tk.StringVar()
        self._build_layout()
        self.refresh_from_disk()

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _build_layout(self) -> None:
        header = ttk.Frame(self.container)
        header.pack(fill=tk.X, padx=10, pady=(10, 6))

        ttk.Label(header, text=self.title).pack(side=tk.LEFT)
        actions = ttk.Frame(header)
        actions.pack(side=tk.RIGHT)
        self.add_symbol_entry = ttk.Entry(actions, textvariable=self.add_symbol_var, width=12)
        self.add_symbol_entry.pack(side=tk.LEFT)
        self.add_symbol_entry.bind("<Return>", self._on_add_symbol_return)
        ttk.Button(actions, text="Add", command=self.add_symbol).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Button(actions, text="Dedupe", command=self.force_save).pack(side=tk.LEFT)
        ttk.Button(actions, text="Paste", command=self.paste_symbols).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(actions, text="Copy", command=self.copy_symbols).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(actions, text="Refresh", command=self.refresh_from_disk).pack(side=tk.LEFT, padx=(6, 0))

        hint = ttk.Label(self.container, text=f"{self.path.name} auto-saves and removes duplicates.")
        hint.pack(anchor="w", padx=10, pady=(0, 6))

        self.text_area = scrolledtext.ScrolledText(
            self.container,
            wrap=tk.WORD,
            font=("Courier New", 10),
            bg=INPUT_GREY,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.text_area.bind("<<Modified>>", self._on_modified)

    def _normalize_symbols(self, raw_text: str) -> list[str]:
        symbols: list[str] = []
        seen = set()
        for line in raw_text.splitlines():
            upper = line.strip().upper()
            if not upper or upper.startswith("SYMBOLS FROM TC2000"):
                continue
            for symbol in WATCHLIST_SYMBOL_RE.findall(upper):
                if symbol not in seen:
                    seen.add(symbol)
                    symbols.append(symbol)
        return symbols

    def _set_text_from_symbols(self, symbols: list[str]) -> None:
        text = "\n".join(symbols)
        self._loading = True
        try:
            self.text_area.delete("1.0", tk.END)
            if text:
                self.text_area.insert("1.0", text)
            self.text_area.edit_modified(False)
        finally:
            self._loading = False

    def _write_symbols(self, symbols: list[str], notify: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(symbols), encoding="utf-8")
        self._set_text_from_symbols(symbols)
        if notify:
            self.on_symbols_saved(self, symbols)

    def _save_current(self, notify: bool) -> None:
        self._save_after_id = None
        symbols = self._normalize_symbols(self.text_area.get("1.0", tk.END))
        self._write_symbols(symbols, notify=notify)

    def _on_modified(self, _event=None) -> None:
        if self._loading or not self.text_area.edit_modified():
            return
        self.text_area.edit_modified(False)
        if self._save_after_id:
            self.container.after_cancel(self._save_after_id)
        self._save_after_id = self.container.after(250, lambda: self._save_current(notify=True))

    def refresh_from_disk(self) -> None:
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        symbols = self._normalize_symbols(self.path.read_text(encoding="utf-8"))
        self._write_symbols(symbols, notify=False)

    def force_save(self) -> None:
        if self._save_after_id:
            self.container.after_cancel(self._save_after_id)
            self._save_after_id = None
        self._save_current(notify=True)

    def copy_symbols(self) -> None:
        symbols = self._normalize_symbols(self.text_area.get("1.0", tk.END))
        self.container.clipboard_clear()
        self.container.clipboard_append(", ".join(symbols))

    def paste_symbols(self) -> None:
        try:
            payload = self.container.clipboard_get()
        except tk.TclError:
            return
        current_symbols = self._normalize_symbols(self.text_area.get("1.0", tk.END))
        incoming_symbols = self._normalize_symbols(str(payload))
        merged = []
        seen = set()
        for symbol in current_symbols + incoming_symbols:
            if symbol not in seen:
                seen.add(symbol)
                merged.append(symbol)
        self._write_symbols(merged, notify=True)

    def add_symbol(self) -> None:
        incoming = self._normalize_symbols(self.add_symbol_var.get())
        if not incoming:
            return
        current_symbols = self._normalize_symbols(self.text_area.get("1.0", tk.END))
        merged = []
        seen = set()
        for symbol in current_symbols + incoming:
            if symbol not in seen:
                seen.add(symbol)
                merged.append(symbol)
        self._write_symbols(merged, notify=True)
        self.add_symbol_var.set("")

    def _on_add_symbol_return(self, _event=None):
        self.add_symbol()
        return "break"

    def remove_symbols(self, symbols_to_remove: set[str]) -> None:
        current_symbols = self._normalize_symbols(self.text_area.get("1.0", tk.END))
        filtered = [symbol for symbol in current_symbols if symbol not in symbols_to_remove]
        if filtered != current_symbols:
            self._write_symbols(filtered, notify=False)


class WatchlistEditorArea:
    def __init__(self, parent: tk.Misc):
        self.parent = parent
        self.container = ttk.Frame(parent)
        self._build_layout()

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _build_layout(self) -> None:
        pane = tk.PanedWindow(self.container, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg=DARK_GREY)
        pane.pack(fill=tk.BOTH, expand=True)

        longs_frame = ttk.Frame(pane)
        shorts_frame = ttk.Frame(pane)

        self.longs_panel = WatchlistEditorPanel(longs_frame, "Longs Watchlist", LONGS_FILE, self._handle_symbols_saved)
        self.longs_panel.pack(fill=tk.BOTH, expand=True)

        self.shorts_panel = WatchlistEditorPanel(shorts_frame, "Shorts Watchlist", SHORTS_FILE, self._handle_symbols_saved)
        self.shorts_panel.pack(fill=tk.BOTH, expand=True)

        pane.add(longs_frame, stretch="always")
        pane.add(shorts_frame, stretch="always")

    def _handle_symbols_saved(self, source_panel: WatchlistEditorPanel, symbols: list[str]) -> None:
        peer = self.shorts_panel if source_panel is self.longs_panel else self.longs_panel
        peer.remove_symbols(set(symbols))


class ConsolidatedTradingGUI:
    def __init__(self, root: tk.Tk, mode: str):
        self.root = root
        self.mode = mode
        self.root.title("Consolidated Trading GUI")
        self.root.geometry("1880x1040" if mode == "full" else "1380x900")
        self.root.configure(bg=DARK_GREY)
        configure_theme(self.root)

        self.bounce_panel: BaseBounceBotPanel | None = None
        self.avwap_gui: MasterAvwapGUI | None = None
        self._build_layout()

    def _build_layout(self) -> None:
        main_pane = tk.PanedWindow(
            self.root,
            orient=tk.VERTICAL,
            sashrelief=tk.RAISED,
            sashwidth=10,
            showhandle=True,
            bg=DARK_GREY,
        )
        main_pane.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main_pane)

        bounce_tab = ttk.Frame(notebook)
        notebook.add(bounce_tab, text="BounceBot")

        master_tab = ttk.Frame(notebook)
        notebook.add(master_tab, text="Master AVWAP")

        if self.mode == "full":
            self.bounce_panel = FullBounceBotPanel(bounce_tab, switch_mode_callback=lambda: self.switch_mode("simple"))
            self.bounce_panel.pack(fill=tk.BOTH, expand=True)
        else:
            self.bounce_panel = SimpleBounceBotPanel(bounce_tab, switch_mode_callback=lambda: self.switch_mode("full"))
            self.bounce_panel.pack(fill=tk.BOTH, expand=True)

        self.avwap_gui = MasterAvwapGUI(master_tab, standalone=False)

        main_pane.add(notebook, stretch="always")

        watchlist_container = ttk.Frame(main_pane)
        self.watchlist_area = WatchlistEditorArea(watchlist_container)
        self.watchlist_area.pack(fill=tk.BOTH, expand=True)
        main_pane.add(watchlist_container)

    def on_close(self) -> None:
        try:
            if self.bounce_panel:
                self.bounce_panel.on_close()
        finally:
            self.root.destroy()

    def switch_mode(self, next_mode: str) -> None:
        if next_mode == self.mode:
            return
        save_local_setting("gui_mode", next_mode)
        try:
            if self.bounce_panel:
                self.bounce_panel.on_close()
        finally:
            self.root.destroy()
        launch(mode=next_mode)


def launch(mode: str = "prompt") -> None:
    if mode == "prompt":
        mode = str(get_local_setting("gui_mode", "full") or "full").strip().lower()
        if mode not in {"full", "simple"}:
            mode = "full"

    root = tk.Tk()
    app = ConsolidatedTradingGUI(root, mode=mode)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the consolidated BounceBot + Master AVWAP GUI.")
    parser.add_argument(
        "--mode",
        choices=("prompt", "full", "simple"),
        default="full",
        help="Launch directly in full/simple mode, or use prompt to choose and save a preference.",
    )
    args = parser.parse_args()
    if args.mode in {"full", "simple"}:
        save_local_setting("gui_mode", args.mode)
    launch(mode=args.mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
