#!/usr/bin/env python3
"""Unified GUI for AVWAP manager + BounceBot RRS panel + longs/shorts lists."""

from __future__ import annotations

import queue
import re
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
        self.selected_symbol = tk.StringVar(value="")

        self._build_layout()
        self._start_bounce_rrs_feed()
        self.refresh_watchlists()
        self._process_rrs_queue()

    def _build_layout(self):
        main = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg=DARK_GREY)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: Master AVWAP + TickerMover watchlists stacked vertically.
        left = tk.PanedWindow(main, orient=tk.VERTICAL, sashrelief=tk.RAISED, bg=DARK_GREY)
        main.add(left, stretch="always", minsize=980)

        avwap_root = ttk.Frame(left)
        self.avwap_gui = MasterAvwapGUI(avwap_root, standalone=False)
        left.add(avwap_root, stretch="always", minsize=600)

        watch_frame = tk.LabelFrame(left, text="TickerMover Watchlists", bg=DARK_GREY, fg=TEXT_COLOR, padx=8, pady=8)
        # Keep watchlists visible, but don't let this section aggressively consume
        # height on very large monitors.
        left.add(watch_frame, stretch="never", minsize=150, height=220)

        # Right: Bounce RRS panel with extra vertical room.
        right = tk.Frame(main, bg=DARK_GREY)
        main.add(right, stretch="always", minsize=700)

        self.rrs_panel = create_rrs_confirmed_panel(right, bot_instance=self._bot_proxy(), dark_grey=DARK_GREY, text_color=TEXT_COLOR)
        self.rrs_panel["container"].pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

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
        self._configure_drop_target(self.longs_text, LONGS_FILE)
        lists.add(longs_panel, stretch="always")

        shorts_panel = tk.LabelFrame(lists, text="shorts.txt", bg=DARK_GREY, fg=TEXT_COLOR)
        self.shorts_text = scrolledtext.ScrolledText(shorts_panel, wrap=tk.NONE, font=("Courier", 11), bg="#222", fg=TEXT_COLOR)
        self.shorts_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.shorts_text.bind("<Double-Button-1>", lambda e: self._pick_symbol_from_text(self.shorts_text))
        self._configure_drop_target(self.shorts_text, SHORTS_FILE)
        lists.add(shorts_panel, stretch="always")

        self._apply_dark_widget_theme()

    def _configure_drop_target(self, widget: scrolledtext.ScrolledText, destination: Path):
        """Register text widgets as drop targets (best effort) for TC2000 symbols."""
        # tkdnd uses Tk's DND package; if unavailable we silently skip.
        try:
            self.root.tk.call("package", "require", "tkdnd")
            widget.drop_target_register("DND_Text")
            widget.dnd_bind("<<Drop>>", lambda event, p=destination: self._handle_symbol_drop(event, p))
        except tk.TclError:
            return

    def _handle_symbol_drop(self, event, destination: Path):
        raw = getattr(event, "data", "") or ""
        symbols = self._extract_symbols(raw)
        if symbols:
            self._append_symbols_to_file(destination, symbols)
            self.refresh_watchlists()
        return "break"

    def _extract_symbols(self, raw: str) -> list[str]:
        # Accept drops like "AAPL", "AAPL\nMSFT", or TC2000 style rows with
        # pipes/tabs. Only keep all-caps ticker-like tokens.
        candidates = re.split(r"[\s,;|{}\t\n\r]+", raw.upper())
        return [c for c in candidates if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", c)]

    def _append_symbols_to_file(self, path: Path, symbols: list[str]):
        existing = []
        if path.exists():
            existing = [ln.strip().upper() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

        changed = False
        for sym in symbols:
            if sym not in existing:
                existing.append(sym)
                changed = True

        if changed:
            path.write_text("\n".join(existing) + "\n", encoding="utf-8")

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

        self._append_symbols_to_file(path, [symbol])
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
