from __future__ import annotations

from . import app as _app

globals().update(
    {
        name: value
        for name, value in vars(_app).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
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
        self.status_var = tk.StringVar()
        self._build_layout()
        self.refresh_from_disk()

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _build_layout(self) -> None:
        header = ttk.Frame(self.container)
        header.pack(fill=tk.X, padx=10, pady=(10, 6))

        ttk.Label(header, text=self.title, font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        actions = ttk.Frame(self.container)
        actions.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.add_symbol_entry = ttk.Entry(actions, textvariable=self.add_symbol_var, width=12)
        self.add_symbol_entry.pack(side=tk.LEFT)
        self.add_symbol_entry.bind("<Return>", self._on_add_symbol_return)
        ttk.Button(actions, text="Add", command=self.add_symbol).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Button(actions, text="Save / Dedupe", command=self.force_save).pack(side=tk.LEFT)
        ttk.Button(actions, text="Sort A-Z", command=self.sort_symbols).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(actions, text="Paste", command=self.paste_symbols).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(actions, text="Copy", command=self.copy_symbols).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(actions, text="Refresh", command=self.refresh_from_disk).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(actions, text="Open Folder", command=self.open_folder).pack(side=tk.LEFT, padx=(6, 0))

        status = ttk.Label(self.container, textvariable=self.status_var, anchor="w", justify=tk.LEFT, wraplength=620)
        status.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.text_area = scrolledtext.ScrolledText(
            self.container,
            wrap=tk.WORD,
            font=("Courier New", 11),
            bg=INPUT_GREY,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            undo=True,
            maxundo=100,
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.text_area.bind("<<Modified>>", self._on_modified)

    def _normalize_symbols(self, raw_text: str) -> list[str]:
        return extract_watchlist_symbols(raw_text)

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

    def _set_status_from_symbols(self, symbols: list[str], action: str) -> None:
        count = len(symbols)
        label = "symbol" if count == 1 else "symbols"
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_var.set(f"{self.path.name} | {count} {label} | {action} {timestamp} | {self.path}")

    def _write_symbols(self, symbols: list[str], notify: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(symbols), encoding="utf-8")
        self._set_text_from_symbols(symbols)
        self._set_status_from_symbols(symbols, "saved")
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
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
        symbols = self._normalize_symbols(self.path.read_text(encoding="utf-8"))
        self._write_symbols(symbols, notify=False)
        self._set_status_from_symbols(symbols, "loaded")

    def force_save(self) -> None:
        if self._save_after_id:
            self.container.after_cancel(self._save_after_id)
            self._save_after_id = None
        self._save_current(notify=True)

    def copy_symbols(self) -> None:
        symbols = self._normalize_symbols(self.text_area.get("1.0", tk.END))
        self.container.clipboard_clear()
        self.container.clipboard_append(", ".join(symbols))
        self._set_status_from_symbols(symbols, "copied")

    def sort_symbols(self) -> None:
        symbols = sorted(self._normalize_symbols(self.text_area.get("1.0", tk.END)))
        self._write_symbols(symbols, notify=True)

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

    def open_folder(self) -> None:
        _open_folder(self.path.parent)


class WatchlistEditorArea:
    def __init__(
        self,
        parent: tk.Misc,
        long_title: str = "Longs Watchlist",
        long_path: Path = LONGS_FILE,
        short_title: str = "Shorts Watchlist",
        short_path: Path = SHORTS_FILE,
    ):
        self.parent = parent
        self.long_title = long_title
        self.long_path = long_path
        self.short_title = short_title
        self.short_path = short_path
        self.container = ttk.Frame(parent)
        self._build_layout()

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _build_layout(self) -> None:
        pane = tk.PanedWindow(self.container, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg=DARK_GREY)
        pane.pack(fill=tk.BOTH, expand=True)

        longs_frame = ttk.Frame(pane)
        shorts_frame = ttk.Frame(pane)

        self.longs_panel = WatchlistEditorPanel(
            longs_frame,
            self.long_title,
            self.long_path,
            self._handle_symbols_saved,
        )
        self.longs_panel.pack(fill=tk.BOTH, expand=True)

        self.shorts_panel = WatchlistEditorPanel(
            shorts_frame,
            self.short_title,
            self.short_path,
            self._handle_symbols_saved,
        )
        self.shorts_panel.pack(fill=tk.BOTH, expand=True)

        pane.add(longs_frame, stretch="always")
        pane.add(shorts_frame, stretch="always")

    def _handle_symbols_saved(self, source_panel: WatchlistEditorPanel, symbols: list[str]) -> None:
        peer = self.shorts_panel if source_panel is self.longs_panel else self.longs_panel
        peer.remove_symbols(set(symbols))

__all__ = ["WatchlistEditorPanel", "WatchlistEditorArea"]
