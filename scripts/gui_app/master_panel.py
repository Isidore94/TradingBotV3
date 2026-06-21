from __future__ import annotations

from . import app as _app

globals().update(
    {
        name: value
        for name, value in vars(_app).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)


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
        configure_market_text_tags(self.text_area)

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
        combined = build_combined_avwap_output_text(
            self._read_text_file(PRIORITY_SETUPS_FILE),
            self._read_text_file(THETA_PUTS_FILE),
            self._read_text_file(EVENT_TICKERS_FILE),
            self._read_text_file(STDEV_RANGE_FILE),
            self._read_text_file(MASTER_AVWAP_D1_UPGRADE_ALERTS_REPORT_FILE),
        )
        set_highlighted_text(self.text_area, combined, state_after="disabled")

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

__all__ = ["SimpleMasterAvwapPanel"]
