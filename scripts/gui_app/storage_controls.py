from __future__ import annotations

from . import app as _app

globals().update(
    {
        name: value
        for name, value in vars(_app).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)


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
            "Use a Google Drive or OneDrive folder here so watchlists, reports, logs, and AVWAP tracker data stay in sync across devices. Replaceable download caches stay local on each computer so the cloud folder stays lightweight."
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
            f"Local machine cache: {details['local_cache_dir']}\n"
            f"Home-folder longs.txt: {shared_watchlists['longs_path']} ({shared_watchlists['longs_exists']})\n"
            f"Home-folder shorts.txt: {shared_watchlists['shorts_path']} ({shared_watchlists['shorts_exists']})\n"
            f"Master swinglongs.txt: {SWING_LONGS_FILE} ({'yes' if SWING_LONGS_FILE.exists() else 'no'})\n"
            f"Master shortswings.txt: {SWING_SHORTS_FILE} ({'yes' if SWING_SHORTS_FILE.exists() else 'no'})\n"
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
            "Master AVWAP also reads swinglongs.txt and shortswings.txt from that folder; BounceBot does not.\n\n"
            "Replaceable download caches stay local to each computer so the shared folder stays small.\n\n"
            "Restart the GUI to start using the new home folder.",
        )

    def open_folder(self) -> None:
        _open_folder(self.shared_root_dir)

    def open_settings_file(self) -> None:
        _open_folder(self.settings_file.parent)

__all__ = ["TrackerStorageControls"]
