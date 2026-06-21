#!/usr/bin/env python3
"""Unified GUI with simple and full workspaces for BounceBot and Master AVWAP."""

from __future__ import annotations

import argparse
import logging
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
    MASTER_AVWAP_D1_UPGRADE_ALERTS_REPORT_FILE,
    SHORTS_FILE,
    SWING_LONGS_FILE,
    SWING_SHORTS_FILE,
    get_shared_watchlist_details,
    get_tracker_storage_details,
    get_local_setting,
    open_path_in_file_manager,
    save_tracker_storage_dir,
    save_local_setting,
)

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ROOT_DIR = SCRIPT_DIR.parent

from bounce_bot import (
    BOUNCE_TYPE_DEFAULTS,
    BOUNCE_TYPE_LABELS,
    MARKET_ENVIRONMENTS,
    RRS_TIMEFRAMES,
    append_alert_message,
    configure_alert_tags,
    create_rrs_confirmed_panel,
    record_bounce_feedback,
    run_bot_with_gui,
)
from master_avwap import (
    EVENT_TICKERS_FILE,
    PRIORITY_SETUPS_FILE,
    STDEV_RANGE_FILE,
    THETA_PUTS_FILE,
    MasterAvwapGUI,
    build_combined_avwap_output_text,
    run_master,
    run_master_with_shared_watchlists,
)
from gui_output import (
    MAIN_GUI_OUTPUT_DEBOUNCE_MS,
    MAIN_GUI_OUTPUT_FILE,
    MAIN_GUI_OUTPUT_REFRESH_MS,
    build_consolidated_gui_output,
)
from gui_text_highlighter import configure_market_text_tags, set_highlighted_text
from market_prep_tab import MarketPrepTab, TickerLookupTab
from watchlist_utils import extract_watchlist_symbols

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

from .storage_controls import TrackerStorageControls
from .bounce_panel import BounceBotController, BaseBounceBotPanel, FullBounceBotPanel, SimpleBounceBotPanel
from .master_panel import SimpleMasterAvwapPanel
from .watchlist_editor import WatchlistEditorArea, WatchlistEditorPanel


class ConsolidatedTradingGUI:
    def __init__(self, root: tk.Tk, mode: str):
        self.root = root
        self.mode = mode
        self.root.title("Consolidated Trading GUI")
        self.root.geometry("1880x1040" if mode == "full" else "1380x900")
        self.root.minsize(1280, 760)
        self.root.configure(bg=DARK_GREY)
        configure_theme(self.root)

        self.bounce_panel: BaseBounceBotPanel | None = None
        self.avwap_gui: MasterAvwapGUI | None = None
        self.market_prep_panel: MarketPrepTab | None = None
        self.ticker_lookup_panel: TickerLookupTab | None = None
        self._output_write_after_id = None
        self._output_refresh_after_id = None
        self._output_traces: list[tuple[tk.Variable, str]] = []
        self._build_layout()
        self._configure_output_snapshot_updates()

    def _build_layout(self) -> None:
        notebook = ttk.Notebook(self.root)
        self.main_notebook = notebook
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        trading_tab = ttk.Frame(notebook)
        self.trading_tab = trading_tab
        notebook.add(trading_tab, text="Trading")

        market_prep_tab = ttk.Frame(notebook)
        self.market_prep_tab = market_prep_tab
        notebook.add(market_prep_tab, text="Market Prep")

        ticker_lookup_tab = ttk.Frame(notebook)
        self.ticker_lookup_tab = ticker_lookup_tab
        notebook.add(ticker_lookup_tab, text="Ticker Lookup")

        trading_pane = tk.PanedWindow(
            trading_tab,
            orient=tk.VERTICAL,
            sashrelief=tk.RAISED,
            sashwidth=10,
            showhandle=True,
            bg=DARK_GREY,
        )
        trading_pane.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        trading_notebook = ttk.Notebook(trading_pane)
        self.trading_notebook = trading_notebook

        bounce_tab = ttk.Frame(trading_notebook)
        self.bounce_tab = bounce_tab
        trading_notebook.add(bounce_tab, text="BounceBot")

        master_tab = ttk.Frame(trading_notebook)
        self.master_tab = master_tab
        trading_notebook.add(master_tab, text="Master AVWAP")

        if self.mode == "full":
            self.bounce_panel = FullBounceBotPanel(bounce_tab, switch_mode_callback=lambda: self.switch_mode("simple"))
            self.bounce_panel.pack(fill=tk.BOTH, expand=True)
        else:
            self.bounce_panel = SimpleBounceBotPanel(bounce_tab, switch_mode_callback=lambda: self.switch_mode("full"))
            self.bounce_panel.pack(fill=tk.BOTH, expand=True)

        self.avwap_gui = MasterAvwapGUI(master_tab, standalone=False)
        self.market_prep_panel = MarketPrepTab(market_prep_tab, text_bg=INPUT_GREY, text_fg=TEXT_COLOR)
        self.market_prep_panel.pack(fill=tk.BOTH, expand=True)
        self.ticker_lookup_panel = TickerLookupTab(ticker_lookup_tab, text_bg=INPUT_GREY, text_fg=TEXT_COLOR)
        self.ticker_lookup_panel.pack(fill=tk.BOTH, expand=True)

        trading_pane.add(trading_notebook, stretch="always", minsize=420)
        trading_notebook.bind("<<NotebookTabChanged>>", self._on_trading_tab_changed)
        notebook.bind("<<NotebookTabChanged>>", self._on_main_tab_changed)

        watchlist_container = ttk.LabelFrame(trading_pane, text="Trading Watchlists")
        self.watchlist_container = watchlist_container

        watchlist_toolbar = ttk.Frame(watchlist_container)
        watchlist_toolbar.pack(fill=tk.X, padx=10, pady=(8, 0))
        ttk.Label(
            watchlist_toolbar,
            text="Shared lists feed BounceBot and Master AVWAP. Master swing lists are AVWAP-only.",
        ).pack(side=tk.LEFT)
        ttk.Button(watchlist_toolbar, text="Refresh All", command=self._refresh_watchlist_editors).pack(side=tk.RIGHT)

        watchlist_notebook = ttk.Notebook(watchlist_container)
        self.watchlist_notebook = watchlist_notebook
        watchlist_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 10))

        shared_watchlist_tab = ttk.Frame(watchlist_notebook)
        self.shared_watchlist_tab = shared_watchlist_tab
        watchlist_notebook.add(shared_watchlist_tab, text="Shared longs / shorts")

        master_watchlist_tab = ttk.Frame(watchlist_notebook)
        self.master_watchlist_tab = master_watchlist_tab
        watchlist_notebook.add(master_watchlist_tab, text="Master swing lists")

        self.bounce_watchlist_area = WatchlistEditorArea(
            shared_watchlist_tab,
            long_title="BounceBot Longs",
            long_path=LONGS_FILE,
            short_title="BounceBot Shorts",
            short_path=SHORTS_FILE,
        )
        self.master_watchlist_area = WatchlistEditorArea(
            master_watchlist_tab,
            long_title="Master Swing Longs",
            long_path=SWING_LONGS_FILE,
            short_title="Master Short Swings",
            short_path=SWING_SHORTS_FILE,
        )
        self.bounce_watchlist_area.pack(fill=tk.BOTH, expand=True)
        self.master_watchlist_area.pack(fill=tk.BOTH, expand=True)
        self.watchlist_area = self.bounce_watchlist_area
        self._visible_watchlist_area = self.bounce_watchlist_area
        self._sync_watchlist_editor_to_selected_tab()
        trading_pane.add(watchlist_container, stretch="never", minsize=230, height=300)

    def _sync_watchlist_editor_to_selected_tab(self) -> None:
        selected_tab = self.trading_notebook.select()
        if selected_tab == str(self.master_tab):
            next_area = self.master_watchlist_area
            next_watchlist_tab = self.master_watchlist_tab
        else:
            next_area = self.bounce_watchlist_area
            next_watchlist_tab = self.shared_watchlist_tab

        try:
            if self.watchlist_notebook.select() != str(next_watchlist_tab):
                self.watchlist_notebook.select(next_watchlist_tab)
        except tk.TclError:
            pass
        if self._visible_watchlist_area is next_area:
            return
        next_area.longs_panel.refresh_from_disk()
        next_area.shorts_panel.refresh_from_disk()
        self.watchlist_area = next_area
        self._visible_watchlist_area = next_area

    def _refresh_watchlist_editors(self) -> None:
        for area in (self.bounce_watchlist_area, self.master_watchlist_area):
            area.longs_panel.refresh_from_disk()
            area.shorts_panel.refresh_from_disk()

    def _on_main_tab_changed(self, _event=None) -> None:
        if self.main_notebook.select() == str(self.trading_tab):
            self._sync_watchlist_editor_to_selected_tab()

    def _on_trading_tab_changed(self, _event=None) -> None:
        self._sync_watchlist_editor_to_selected_tab()

    def _configure_output_snapshot_updates(self) -> None:
        if self.bounce_panel:
            self.bounce_panel.on_output_changed = self.request_output_snapshot
            controller = getattr(self.bounce_panel, "controller", None)
            if controller:
                self._bind_output_var(controller.status_var)
                self._bind_output_var(controller.connection_var)
                self._bind_output_var(controller.active_bounce_var)
        if self.avwap_gui:
            self.avwap_gui.on_output_changed = self.request_output_snapshot
            self._bind_output_var(self.avwap_gui.status_var)
            self._bind_output_var(self.avwap_gui.tracker_storage_var)
        self.request_output_snapshot(delay_ms=250)
        self._schedule_periodic_output_snapshot()

    def _bind_output_var(self, variable: tk.Variable | None) -> None:
        if variable is None:
            return
        trace_id = variable.trace_add("write", lambda *_args: self.request_output_snapshot())
        self._output_traces.append((variable, trace_id))

    def request_output_snapshot(self, delay_ms: int = MAIN_GUI_OUTPUT_DEBOUNCE_MS) -> None:
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return

        if self._output_write_after_id:
            try:
                self.root.after_cancel(self._output_write_after_id)
            except Exception:
                pass
        self._output_write_after_id = self.root.after(delay_ms, self._write_output_snapshot)

    def _write_output_snapshot(self) -> None:
        self._output_write_after_id = None
        try:
            payload = build_consolidated_gui_output(self.mode, self.bounce_panel, self.avwap_gui)
            MAIN_GUI_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            MAIN_GUI_OUTPUT_FILE.write_text(payload, encoding="utf-8")
        except Exception:
            logging.exception("Failed writing consolidated GUI snapshot to %s", MAIN_GUI_OUTPUT_FILE)

    def _schedule_periodic_output_snapshot(self) -> None:
        try:
            self._output_refresh_after_id = self.root.after(
                MAIN_GUI_OUTPUT_REFRESH_MS,
                self._run_periodic_output_snapshot,
            )
        except tk.TclError:
            self._output_refresh_after_id = None

    def _run_periodic_output_snapshot(self) -> None:
        self._output_refresh_after_id = None
        self.request_output_snapshot(delay_ms=0)
        self._schedule_periodic_output_snapshot()

    def _cancel_output_snapshot_updates(self) -> None:
        if self._output_write_after_id:
            try:
                self.root.after_cancel(self._output_write_after_id)
            except Exception:
                pass
            self._output_write_after_id = None
        if self._output_refresh_after_id:
            try:
                self.root.after_cancel(self._output_refresh_after_id)
            except Exception:
                pass
            self._output_refresh_after_id = None
        for variable, trace_id in self._output_traces:
            try:
                variable.trace_remove("write", trace_id)
            except Exception:
                pass
        self._output_traces.clear()

    def on_close(self) -> None:
        self._write_output_snapshot()
        self._cancel_output_snapshot_updates()
        try:
            if self.bounce_panel:
                self.bounce_panel.on_close()
        finally:
            self.root.destroy()

    def switch_mode(self, next_mode: str) -> None:
        if next_mode == self.mode:
            return
        save_local_setting("gui_mode", next_mode)
        self._write_output_snapshot()
        self._cancel_output_snapshot_updates()
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
