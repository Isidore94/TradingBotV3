from __future__ import annotations

from . import legacy as _legacy

# The GUI is extracted first while the scanner logic is still being migrated.
# Copy legacy globals, including private helpers, so existing method bodies keep
# their behavior during the package split.
globals().update(
    {
        name: value
        for name, value in vars(_legacy).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)


# ============================================================================
# GUI
# ============================================================================

class MasterAvwapGUI:
    def __init__(self, root, standalone=True):
        self.root = root
        self.standalone = standalone
        if standalone:
            self.root.title("Master AVWAP Manager")
            self.root.geometry("1200x760")

        # Embedded mode may pass a ttk container (e.g. ttk.Frame), which
        # does not expose a `bg`/`background` configure option.
        try:
            self.root.configure(bg=GUI_DARK_BG)
        except tk.TclError:
            try:
                self.root.configure(background=GUI_DARK_BG)
            except tk.TclError:
                pass
        self._configure_dark_theme()

        self.status_var = tk.StringVar(value="Ready")
        self.tracker_storage_var = tk.StringVar(value="")
        self.user_favorite_notes_var = tk.StringVar()
        self.tracker_backfill_sessions_var = tk.IntVar(value=5)
        self.shared_scheduler_button_var = tk.StringVar(value="Start Scheduler")
        self.shared_scheduler_status_var = tk.StringVar(value="")
        self.focus_side_map = {}
        self.setup_tracker_row_map = {}
        self.on_output_changed = None
        self.background_task_active = False
        self.current_background_label = ""
        self.shared_scheduler_enabled = False
        self.shared_scheduler_active_slot = ""
        self.shared_scheduler_note = "Hourly shared-watchlist scheduler is off."
        self.shared_scheduler_day = ""
        self.shared_scheduler_slots_state = {}
        self.setup_tracker_payload = _default_setup_tracker_payload()
        self.setup_tracker_setup_type_rows = []
        self.setup_tracker_playbook_rows = []
        self.setup_tracker_best_playbook_rows = []
        self.setup_tracker_factor_rows = []
        self.setup_tracker_view_loaded = False
        self.setup_tracker_view_stale = True
        self.theta_sort_column = "score"
        self.theta_sort_descending = True
        self.theta_table_rows = []
        self.theta_pcs_table_rows = []
        self.theta_min_score_var = tk.StringVar(value="")
        self.theta_min_support_count_var = tk.StringVar(value="")
        self.theta_max_earnings_days_var = tk.StringVar(value="")
        self.theta_symbol_filter_var = tk.StringVar(value="")

        self._build_layout()
        self._show_setup_tracker_lazy_placeholder()
        self.refresh_avwap_output_view()
        self.refresh_theta_output_view()
        self.refresh_market_prep_view()
        self.notebook.select(self.avwap_tab)
        self._refresh_shared_watchlist_scheduler_status()
        self.root.after(SHARED_WATCHLIST_SCHEDULER_TICK_MS, self._shared_watchlist_scheduler_tick)

    def _configure_dark_theme(self):
        style = ttk.Style(self.root)

        try:
            current_theme = style.theme_use()
            if current_theme == "default":
                style.theme_use("clam")
        except Exception:
            pass

        self.root.option_add("*Background", GUI_DARK_BG)
        self.root.option_add("*Foreground", GUI_DARK_TEXT)
        self.root.option_add("*Entry*Background", GUI_DARK_INPUT)
        self.root.option_add("*Entry*Foreground", GUI_DARK_TEXT)
        self.root.option_add("*Text*Background", GUI_DARK_INPUT)
        self.root.option_add("*Text*Foreground", GUI_DARK_TEXT)

        style.configure(".", background=GUI_DARK_BG, foreground=GUI_DARK_TEXT)
        style.configure("TFrame", background=GUI_DARK_BG)
        style.configure("TLabel", background=GUI_DARK_BG, foreground=GUI_DARK_TEXT)

        style.configure("TButton", background=GUI_DARK_PANEL, foreground=GUI_DARK_TEXT)
        style.map(
            "TButton",
            background=[("active", "#4A4A4A"), ("disabled", GUI_DARK_PANEL)],
            foreground=[("disabled", "#9AA0A6")],
        )

        style.configure("TEntry", fieldbackground=GUI_DARK_INPUT, foreground=GUI_DARK_TEXT)
        style.map(
            "TEntry",
            fieldbackground=[("readonly", GUI_DARK_INPUT), ("disabled", GUI_DARK_PANEL)],
            foreground=[("readonly", GUI_DARK_TEXT), ("disabled", "#9AA0A6")],
        )

        style.configure("TSpinbox", fieldbackground=GUI_DARK_INPUT, foreground=GUI_DARK_TEXT)
        style.map(
            "TSpinbox",
            fieldbackground=[("readonly", GUI_DARK_INPUT), ("disabled", GUI_DARK_PANEL)],
            foreground=[("readonly", GUI_DARK_TEXT), ("disabled", "#9AA0A6")],
        )

        style.configure("TCombobox", fieldbackground=GUI_DARK_INPUT, foreground=GUI_DARK_TEXT)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", GUI_DARK_INPUT), ("disabled", GUI_DARK_PANEL)],
            foreground=[("readonly", GUI_DARK_TEXT), ("disabled", "#9AA0A6")],
        )

        style.configure("TNotebook", background=GUI_DARK_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=GUI_DARK_PANEL, foreground=GUI_DARK_TEXT)
        style.map("TNotebook.Tab", background=[("selected", "#4A4A4A")])
        style.configure("TLabelframe", background=GUI_DARK_BG, foreground=GUI_DARK_TEXT)
        style.configure("TLabelframe.Label", background=GUI_DARK_BG, foreground=GUI_DARK_TEXT)

        # Use explicit style names so Treeview + heading colors are stable on
        # Windows where the default native theme can ignore generic style keys.
        style.configure(
            "Dark.Treeview",
            background=GUI_DARK_INPUT,
            fieldbackground=GUI_DARK_INPUT,
            foreground=GUI_DARK_TEXT,
            bordercolor=GUI_DARK_BG,
            rowheight=24,
        )
        style.map(
            "Dark.Treeview",
            background=[("selected", "#4A4A4A")],
            foreground=[("selected", GUI_DARK_TEXT)],
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=GUI_DARK_PANEL,
            foreground=GUI_DARK_TEXT,
            relief="flat",
        )
        style.map("Dark.Treeview.Heading", background=[("active", "#4A4A4A")])

        style.configure("Vertical.TScrollbar", background=GUI_DARK_PANEL, troughcolor=GUI_DARK_BG)

    def _notify_output_changed(self):
        callback = getattr(self, "on_output_changed", None)
        if callable(callback):
            callback()

    def _apply_dark_theme_to_text_widgets(self):
        widgets = [
            self.avwap_text,
            self.favorite_symbols_text,
            self.near_favorite_symbols_text,
            self.user_favorite_symbols_text,
            self.long_focus_symbols_text,
            self.short_focus_symbols_text,
            self.setup_type_symbols_text,
            self.theta_text,
            self.theta_symbols_text,
            self.theta_reason_risk_text,
            self.setup_tracker_stats_text,
            self.setup_tracker_playbook_text,
            self.setup_tracker_setup_type_text,
            self.setup_tracker_factor_text,
        ]
        widgets.extend(getattr(self, "market_prep_text_widgets", []))
        if getattr(self, "market_prep_report_text", None) is not None:
            widgets.append(self.market_prep_report_text)

        for widget in widgets:
            widget.configure(
                bg=GUI_DARK_INPUT,
                fg=GUI_DARK_TEXT,
                insertbackground=GUI_DARK_TEXT,
                selectbackground="#4A4A4A",
                selectforeground=GUI_DARK_TEXT,
                highlightbackground=GUI_DARK_BG,
                highlightcolor=GUI_DARK_PANEL,
            )
            configure_market_text_tags(widget)
        self.avwap_text.tag_configure("trendline_bold", font=("Courier New", 10, "bold"), foreground="#FFFFFF")
        for tree in (
            getattr(self, "setup_tracker_table", None),
            getattr(self, "setup_tracker_scenario_table", None),
            getattr(self, "setup_tracker_playbook_table", None),
            getattr(self, "setup_tracker_setup_type_table", None),
            getattr(self, "setup_tracker_factor_table", None),
            getattr(self, "theta_table", None),
            getattr(self, "theta_pcs_table", None),
        ):
            configure_treeview_market_tags(tree)

    def _build_layout(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=10, pady=8)

        ttk.Button(toolbar, text="Run Shared Watchlist Scan", command=self.run_master_once).pack(side="left", padx=4)
        ttk.Button(
            toolbar,
            textvariable=self.shared_scheduler_button_var,
            command=self.toggle_shared_watchlist_scheduler,
        ).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Run Local Watchlist Scan", command=self.run_local_watchlist_scan_once).pack(side="left", padx=4)

        ttk.Label(
            self.root,
            text="Views refresh automatically when scans finish or when you open a tab.",
            justify="left",
        ).pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(
            self.root,
            textvariable=self.shared_scheduler_status_var,
            justify="left",
            wraplength=1100,
        ).pack(fill="x", padx=10, pady=(0, 8))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=8)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        tracker_tab = ttk.Frame(self.notebook)
        self.tracker_tab = tracker_tab
        self.notebook.add(tracker_tab, text="Setup Tracker")

        tracker_toolbar = ttk.Frame(tracker_tab)
        tracker_toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(tracker_toolbar, text="Copy Active Symbols", command=self.copy_setup_tracker_symbols).pack(side="left", padx=(8, 0))
        ttk.Button(tracker_toolbar, text="Refresh Tracker", command=self.refresh_setup_tracker_view).pack(side="left", padx=(8, 0))
        ttk.Label(tracker_toolbar, text="Backfill sessions:").pack(side="left", padx=(16, 4))
        tracker_backfill_spin = ttk.Spinbox(
            tracker_toolbar,
            from_=1,
            to=30,
            width=4,
            textvariable=self.tracker_backfill_sessions_var,
        )
        tracker_backfill_spin.pack(side="left")
        ttk.Button(
            tracker_toolbar,
            text="Backfill Tracker",
            command=self.backfill_setup_tracker_history,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            tracker_toolbar,
            text="Analyze Scoring",
            command=self.run_scoring_analysis_once,
        ).pack(side="left", padx=(16, 0))
        ttk.Button(
            tracker_toolbar,
            text="Apply Scoring",
            command=self.apply_scoring_analysis_once,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            tracker_toolbar,
            text="Open Tuner Report",
            command=self.open_scoring_tuner_report,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            tracker_toolbar,
            text="Open Scoring Config",
            command=self.open_scoring_config_file,
        ).pack(side="left", padx=(8, 0))
        ttk.Frame(tracker_toolbar).pack(side="left", fill="x", expand=True)
        ttk.Button(tracker_toolbar, text="Change Home Folder", command=self.choose_tracker_storage_dir).pack(side="left", padx=(16, 0))
        ttk.Button(tracker_toolbar, text="Open Home Folder", command=self.open_tracker_storage_dir).pack(side="left", padx=(8, 0))
        ttk.Button(tracker_toolbar, text="Open Settings File", command=self.open_tracker_settings_file).pack(side="left", padx=(8, 0))

        tracker_storage_label = ttk.Label(
            tracker_tab,
            textvariable=self.tracker_storage_var,
            justify="left",
            wraplength=1100,
        )
        tracker_storage_label.pack(fill="x", pady=(0, 8))

        tracker_body = ttk.Frame(tracker_tab)
        tracker_body.pack(fill="both", expand=True)

        tracker_left = ttk.Frame(tracker_body)
        tracker_left.pack(side="left", fill="both", expand=True)

        setup_frame = ttk.LabelFrame(tracker_left, text="Tracked Setups")
        setup_frame.pack(fill="both", expand=True, pady=(0, 8))
        setup_columns = (
            "scan_date",
            "age_d",
            "symbol",
            "side",
            "bucket",
            "status",
            "closed_scenarios",
            "open_scenarios",
            "avg_closed_r",
            "open_bias",
            "priority_score",
            "retest",
            "compression",
        )
        self.setup_tracker_table = ttk.Treeview(setup_frame, columns=setup_columns, show="headings", style="Dark.Treeview", height=12)
        tracker_headings = {
            "scan_date": "Scan",
            "age_d": "Age",
            "symbol": "Symbol",
            "side": "Side",
            "bucket": "Bucket",
            "status": "Status",
            "closed_scenarios": "Closed",
            "open_scenarios": "Open",
            "avg_closed_r": "Avg Closed R",
            "open_bias": "Open Bias",
            "priority_score": "Score",
            "retest": "Retest",
            "compression": "Comp",
        }
        tracker_col_widths = {
            "scan_date": 96,
            "age_d": 54,
            "symbol": 80,
            "side": 60,
            "bucket": 120,
            "status": 86,
            "closed_scenarios": 70,
            "open_scenarios": 96,
            "avg_closed_r": 94,
            "open_bias": 88,
            "priority_score": 90,
            "retest": 110,
            "compression": 64,
        }
        for col in setup_columns:
            self.setup_tracker_table.heading(col, text=tracker_headings.get(col, col))
            self.setup_tracker_table.column(col, width=tracker_col_widths.get(col, 100), anchor="w")
        setup_scroll = ttk.Scrollbar(setup_frame, orient="vertical", command=self.setup_tracker_table.yview)
        self.setup_tracker_table.configure(yscrollcommand=setup_scroll.set)
        self.setup_tracker_table.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        setup_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.setup_tracker_table.bind("<<TreeviewSelect>>", self._on_setup_tracker_selected)

        scenario_frame = ttk.LabelFrame(tracker_left, text="Scenario Outcomes")
        scenario_frame.pack(fill="both", expand=True)
        scenario_columns = (
            "stop",
            "exit",
            "shares",
            "status",
            "total_r",
            "total_pnl",
            "last_action",
        )
        self.setup_tracker_scenario_table = ttk.Treeview(
            scenario_frame,
            columns=scenario_columns,
            show="headings",
            style="Dark.Treeview",
            height=10,
        )
        scenario_col_widths = {
            "stop": 90,
            "exit": 300,
            "shares": 70,
            "status": 86,
            "total_r": 80,
            "total_pnl": 90,
            "last_action": 300,
        }
        for col in scenario_columns:
            self.setup_tracker_scenario_table.heading(col, text=col)
            self.setup_tracker_scenario_table.column(col, width=scenario_col_widths.get(col, 110), anchor="w")
        scenario_scroll = ttk.Scrollbar(scenario_frame, orient="vertical", command=self.setup_tracker_scenario_table.yview)
        self.setup_tracker_scenario_table.configure(yscrollcommand=scenario_scroll.set)
        self.setup_tracker_scenario_table.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scenario_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)

        tracker_right = ttk.Frame(tracker_body, width=640)
        tracker_right.pack(side="right", fill="y", padx=(12, 0))
        tracker_right.pack_propagate(False)
        tracker_insights_notebook = ttk.Notebook(tracker_right)
        tracker_insights_notebook.pack(fill="both", expand=True)

        tracker_summary_tab = ttk.Frame(tracker_insights_notebook)
        tracker_insights_notebook.add(tracker_summary_tab, text="Summary")
        tracker_stats_frame = ttk.LabelFrame(tracker_summary_tab, text="Tracker Stats / Details")
        tracker_stats_frame.pack(fill="both", expand=True)
        self.setup_tracker_stats_text = tk.Text(tracker_stats_frame, wrap="word", font=("Courier New", 10))
        self.setup_tracker_stats_text.pack(fill="both", expand=True, padx=8, pady=8)

        playbook_tab = ttk.Frame(tracker_insights_notebook)
        tracker_insights_notebook.add(playbook_tab, text="Best Playbooks")
        playbook_frame = ttk.LabelFrame(playbook_tab, text="Best Stop / Profit-Take Playbooks")
        playbook_frame.pack(fill="both", expand=True, pady=(0, 8))
        playbook_columns = (
            "context",
            "stop",
            "profit_take",
            "closed",
            "robust_r",
            "r_edge",
            "win_rate",
        )
        self.setup_tracker_playbook_table = ttk.Treeview(
            playbook_frame,
            columns=playbook_columns,
            show="headings",
            style="Dark.Treeview",
            height=12,
        )
        playbook_col_widths = {
            "context": 210,
            "stop": 68,
            "profit_take": 150,
            "closed": 52,
            "robust_r": 66,
            "r_edge": 66,
            "win_rate": 56,
        }
        playbook_headings = {
            "context": "Setup Type",
            "stop": "Stop",
            "profit_take": "Take Profit",
            "closed": "Closed",
            "robust_r": "Robust R",
            "r_edge": "R Edge",
            "win_rate": "Win",
        }
        for col in playbook_columns:
            self.setup_tracker_playbook_table.heading(col, text=playbook_headings.get(col, col))
            self.setup_tracker_playbook_table.column(
                col,
                width=playbook_col_widths.get(col, 90),
                anchor="w",
            )
        playbook_scroll = ttk.Scrollbar(
            playbook_frame,
            orient="vertical",
            command=self.setup_tracker_playbook_table.yview,
        )
        self.setup_tracker_playbook_table.configure(yscrollcommand=playbook_scroll.set)
        self.setup_tracker_playbook_table.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(8, 0),
            pady=8,
        )
        playbook_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.setup_tracker_playbook_table.bind("<<TreeviewSelect>>", self._on_playbook_selected)

        playbook_detail_frame = ttk.LabelFrame(playbook_tab, text="Playbook Details")
        playbook_detail_frame.pack(fill="both", expand=True)
        self.setup_tracker_playbook_text = tk.Text(
            playbook_detail_frame,
            wrap="word",
            font=("Courier New", 10),
            height=11,
        )
        self.setup_tracker_playbook_text.pack(fill="both", expand=True, padx=8, pady=8)

        setup_types_tab = ttk.Frame(tracker_insights_notebook)
        tracker_insights_notebook.add(setup_types_tab, text="Setup Types")
        setup_type_frame = ttk.LabelFrame(setup_types_tab, text="Setup Type Performance")
        setup_type_frame.pack(fill="both", expand=True, pady=(0, 8))
        setup_type_columns = (
            "type",
            "tracked",
            "closed",
            "avg_closed_r",
            "r_edge",
            "hit_rate",
        )
        self.setup_tracker_setup_type_table = ttk.Treeview(
            setup_type_frame,
            columns=setup_type_columns,
            show="headings",
            style="Dark.Treeview",
            height=12,
        )
        setup_type_col_widths = {
            "type": 315,
            "tracked": 64,
            "closed": 64,
            "avg_closed_r": 78,
            "r_edge": 72,
            "hit_rate": 68,
        }
        for col in setup_type_columns:
            self.setup_tracker_setup_type_table.heading(col, text=col)
            self.setup_tracker_setup_type_table.column(
                col,
                width=setup_type_col_widths.get(col, 90),
                anchor="w",
            )
        setup_type_scroll = ttk.Scrollbar(
            setup_type_frame,
            orient="vertical",
            command=self.setup_tracker_setup_type_table.yview,
        )
        self.setup_tracker_setup_type_table.configure(yscrollcommand=setup_type_scroll.set)
        self.setup_tracker_setup_type_table.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(8, 0),
            pady=8,
        )
        setup_type_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.setup_tracker_setup_type_table.bind("<<TreeviewSelect>>", self._on_setup_type_selected)

        setup_type_detail_frame = ttk.LabelFrame(setup_types_tab, text="Setup Type Details")
        setup_type_detail_frame.pack(fill="both", expand=True)
        self.setup_tracker_setup_type_text = tk.Text(
            setup_type_detail_frame,
            wrap="word",
            font=("Courier New", 10),
            height=11,
        )
        self.setup_tracker_setup_type_text.pack(fill="both", expand=True, padx=8, pady=8)

        factor_tab = ttk.Frame(tracker_insights_notebook)
        tracker_insights_notebook.add(factor_tab, text="Factor Impact")
        factor_frame = ttk.LabelFrame(factor_tab, text="Factor Impact")
        factor_frame.pack(fill="both", expand=True, pady=(0, 8))
        factor_columns = (
            "factor",
            "value",
            "setups",
            "closed",
            "avg_closed_r",
            "r_edge",
            "success",
            "impact",
        )
        self.setup_tracker_factor_table = ttk.Treeview(
            factor_frame,
            columns=factor_columns,
            show="headings",
            style="Dark.Treeview",
            height=12,
        )
        factor_col_widths = {
            "factor": 170,
            "value": 150,
            "setups": 58,
            "closed": 58,
            "avg_closed_r": 78,
            "r_edge": 72,
            "success": 74,
            "impact": 68,
        }
        for col in factor_columns:
            self.setup_tracker_factor_table.heading(col, text=col)
            self.setup_tracker_factor_table.column(
                col,
                width=factor_col_widths.get(col, 90),
                anchor="w",
            )
        factor_scroll = ttk.Scrollbar(
            factor_frame,
            orient="vertical",
            command=self.setup_tracker_factor_table.yview,
        )
        self.setup_tracker_factor_table.configure(yscrollcommand=factor_scroll.set)
        self.setup_tracker_factor_table.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(8, 0),
            pady=8,
        )
        factor_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.setup_tracker_factor_table.bind("<<TreeviewSelect>>", self._on_factor_selected)

        factor_detail_frame = ttk.LabelFrame(factor_tab, text="Factor Details")
        factor_detail_frame.pack(fill="both", expand=True)
        self.setup_tracker_factor_text = tk.Text(
            factor_detail_frame,
            wrap="word",
            font=("Courier New", 10),
            height=11,
        )
        self.setup_tracker_factor_text.pack(fill="both", expand=True, padx=8, pady=8)

        avwap_tab = ttk.Frame(self.notebook)
        self.avwap_tab = avwap_tab
        self.notebook.add(avwap_tab, text="AVWAP Results")

        ttk.Label(
            avwap_tab,
            text="Priority setups, event tickers, and stdev groups refresh automatically after each scan and when you open this tab.",
            justify="left",
            wraplength=1100,
        ).pack(fill="x", pady=(0, 8))

        avwap_body = ttk.Frame(avwap_tab)
        avwap_body.pack(fill="both", expand=True)

        avwap_main = ttk.Frame(avwap_body)
        avwap_main.pack(side="left", fill="both", expand=True)
        self.avwap_text = tk.Text(avwap_main, wrap="word", font=("Courier New", 10))
        self.avwap_text.pack(side="left", fill="both", expand=True)
        output_scroll = ttk.Scrollbar(avwap_main, orient="vertical", command=self.avwap_text.yview)
        self.avwap_text.configure(yscrollcommand=output_scroll.set)
        output_scroll.pack(side="right", fill="y")

        avwap_side = ttk.Frame(avwap_body, width=340)
        avwap_side.pack(side="right", fill="y", padx=(12, 0))
        avwap_side.pack_propagate(False)
        ttk.Label(
            avwap_side,
            text="Comma-separated ticker groups for TradingView paste.",
            justify="left",
            wraplength=300,
        ).pack(fill="x", pady=(0, 8))

        favorite_frame = ttk.LabelFrame(avwap_side, text="Favorite Setups")
        favorite_frame.pack(fill="x", pady=(0, 10))
        self.favorite_symbols_text = tk.Text(favorite_frame, wrap="word", height=5, font=("Courier New", 10))
        self.favorite_symbols_text.pack(fill="both", expand=True, padx=8, pady=(8, 8))
        ttk.Button(favorite_frame, text="Copy Favorites", command=self.copy_favorite_symbols).pack(anchor="w", padx=8, pady=(0, 8))

        near_favorite_frame = ttk.LabelFrame(avwap_side, text="Near Favorite Zones")
        near_favorite_frame.pack(fill="x", pady=(0, 10))
        self.near_favorite_symbols_text = tk.Text(near_favorite_frame, wrap="word", height=5, font=("Courier New", 10))
        self.near_favorite_symbols_text.pack(fill="both", expand=True, padx=8, pady=(8, 8))
        ttk.Button(
            near_favorite_frame,
            text="Copy Near Favorites",
            command=self.copy_near_favorite_symbols,
        ).pack(anchor="w", padx=8, pady=(0, 8))

        user_favorite_frame = ttk.LabelFrame(avwap_side, text="My Favorite Tickers Log")
        user_favorite_frame.pack(fill="x", pady=(0, 10))
        self.user_favorite_symbols_text = tk.Text(
            user_favorite_frame,
            wrap="word",
            height=4,
            font=("Courier New", 10),
        )
        self.user_favorite_symbols_text.pack(fill="both", expand=True, padx=8, pady=(8, 6))
        ttk.Label(user_favorite_frame, text="Notes").pack(anchor="w", padx=8, pady=(0, 2))
        user_favorite_notes = ttk.Entry(
            user_favorite_frame,
            textvariable=self.user_favorite_notes_var,
        )
        user_favorite_notes.pack(fill="x", padx=8, pady=(0, 6))
        user_favorite_actions = ttk.Frame(user_favorite_frame)
        user_favorite_actions.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(
            user_favorite_actions,
            text="Log Favorites",
            command=self.log_user_favorite_symbols,
        ).pack(side="left")
        ttk.Button(
            user_favorite_actions,
            text="Clear",
            command=self.clear_user_favorite_symbols,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            user_favorite_actions,
            text="Open Log",
            command=self.open_user_favorites_log,
        ).pack(side="left", padx=(8, 0))

        directional_frame = ttk.LabelFrame(avwap_side, text="Directional Copy Lists")
        directional_frame.pack(fill="x", pady=(0, 10))
        directional_toolbar = ttk.Frame(directional_frame)
        directional_toolbar.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Button(directional_toolbar, text="Copy Longs", command=self.copy_long_focus_symbols).pack(side="left")
        ttk.Button(directional_toolbar, text="Copy Shorts", command=self.copy_short_focus_symbols).pack(side="left", padx=(8, 0))
        ttk.Label(directional_frame, text="LONG").pack(anchor="w", padx=8, pady=(8, 0))
        self.long_focus_symbols_text = tk.Text(directional_frame, wrap="word", height=3, font=("Courier New", 10))
        self.long_focus_symbols_text.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Label(directional_frame, text="SHORT").pack(anchor="w", padx=8, pady=(0, 0))
        self.short_focus_symbols_text = tk.Text(directional_frame, wrap="word", height=3, font=("Courier New", 10))
        self.short_focus_symbols_text.pack(fill="x", padx=8, pady=(4, 8))

        setup_type_copy_frame = ttk.LabelFrame(avwap_side, text="Score-Ranked Setups")
        setup_type_copy_frame.pack(fill="both", expand=True)
        setup_type_toolbar = ttk.Frame(setup_type_copy_frame)
        setup_type_toolbar.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Button(setup_type_toolbar, text="Copy Ranked Setups", command=self.copy_setup_type_symbols).pack(side="left")
        self.setup_type_symbols_text = tk.Text(setup_type_copy_frame, wrap="word", height=12, font=("Courier New", 10))
        self.setup_type_symbols_text.pack(fill="both", expand=True, padx=8, pady=(8, 8))

        theta_tab = ttk.Frame(self.notebook)
        self.theta_tab = theta_tab
        self.notebook.add(theta_tab, text="Theta Plays")

        ttk.Label(
            theta_tab,
            text="Theta option candidates are recommendation-only and are not added to the setup tracker.",
            justify="left",
            wraplength=1100,
        ).pack(fill="x", pady=(0, 8))

        theta_body = ttk.Frame(theta_tab)
        theta_body.pack(fill="both", expand=True)

        theta_main = ttk.Frame(theta_body)
        theta_main.pack(side="left", fill="both", expand=True)
        theta_filter_frame = ttk.LabelFrame(theta_main, text="Filters")
        theta_filter_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(theta_filter_frame, text="Min score").grid(row=0, column=0, padx=(8, 4), pady=8, sticky="w")
        ttk.Entry(theta_filter_frame, textvariable=self.theta_min_score_var, width=8).grid(row=0, column=1, padx=(0, 8), pady=8, sticky="w")
        ttk.Label(theta_filter_frame, text="Min supports").grid(row=0, column=2, padx=(0, 4), pady=8, sticky="w")
        ttk.Entry(theta_filter_frame, textvariable=self.theta_min_support_count_var, width=8).grid(row=0, column=3, padx=(0, 8), pady=8, sticky="w")
        ttk.Label(theta_filter_frame, text="Max days to earnings").grid(row=0, column=4, padx=(0, 4), pady=8, sticky="w")
        ttk.Entry(theta_filter_frame, textvariable=self.theta_max_earnings_days_var, width=10).grid(row=0, column=5, padx=(0, 8), pady=8, sticky="w")
        ttk.Label(theta_filter_frame, text="Symbol").grid(row=0, column=6, padx=(0, 4), pady=8, sticky="w")
        ttk.Entry(theta_filter_frame, textvariable=self.theta_symbol_filter_var, width=12).grid(row=0, column=7, padx=(0, 8), pady=8, sticky="w")
        ttk.Button(theta_filter_frame, text="Apply", command=self._refresh_theta_table).grid(row=0, column=8, padx=(0, 8), pady=8, sticky="w")
        ttk.Button(theta_filter_frame, text="Clear", command=self._clear_theta_filters).grid(row=0, column=9, padx=(0, 8), pady=8, sticky="w")

        theta_table_frame = ttk.LabelFrame(theta_main, text="Theta Option Candidates")
        theta_table_frame.pack(fill="both", expand=True, pady=(0, 8))
        theta_table_notebook = ttk.Notebook(theta_table_frame)
        theta_table_notebook.pack(fill="both", expand=True, padx=8, pady=8)

        theta_sold_put_frame = ttk.Frame(theta_table_notebook)
        theta_pcs_frame = ttk.Frame(theta_table_notebook)
        theta_table_notebook.add(theta_sold_put_frame, text="Sold Puts")
        theta_table_notebook.add(theta_pcs_frame, text="PCS")

        columns = (
            "symbol",
            "score",
            "support_count",
            "close",
            "atr",
            "next_earnings_days",
            "recommended_strike",
            "recommended_credit",
            "primary_strike_band",
            "liquidity_score",
        )
        self.theta_table = ttk.Treeview(theta_sold_put_frame, columns=columns, show="headings", style="Dark.Treeview")
        headings = {
            "symbol": "Symbol",
            "score": "Score",
            "support_count": "Support Count",
            "close": "Close",
            "atr": "ATR",
            "next_earnings_days": "Next Earnings (days)",
            "recommended_strike": "Sell Strike",
            "recommended_credit": "Approx Credit",
            "primary_strike_band": "Strike Context",
            "liquidity_score": "Quote Source",
        }
        for key, label in headings.items():
            self.theta_table.heading(key, text=label, command=lambda c=key: self._sort_theta_table(c))
        self.theta_table.column("symbol", width=90, anchor="w")
        self.theta_table.column("score", width=70, anchor="center")
        self.theta_table.column("support_count", width=120, anchor="center")
        self.theta_table.column("close", width=85, anchor="e")
        self.theta_table.column("atr", width=85, anchor="e")
        self.theta_table.column("next_earnings_days", width=145, anchor="center")
        self.theta_table.column("recommended_strike", width=105, anchor="e")
        self.theta_table.column("recommended_credit", width=115, anchor="e")
        self.theta_table.column("primary_strike_band", width=260, anchor="w")
        self.theta_table.column("liquidity_score", width=115, anchor="center")
        self.theta_table.pack(side="left", fill="both", expand=True)
        theta_table_scroll = ttk.Scrollbar(theta_sold_put_frame, orient="vertical", command=self.theta_table.yview)
        self.theta_table.configure(yscrollcommand=theta_table_scroll.set)
        theta_table_scroll.pack(side="right", fill="y")

        self.theta_pcs_table = ttk.Treeview(theta_pcs_frame, columns=columns, show="headings", style="Dark.Treeview")
        pcs_headings = dict(headings)
        pcs_headings["primary_strike_band"] = "Spread Context"
        pcs_headings["liquidity_score"] = "Quote Source"
        for key, label in pcs_headings.items():
            self.theta_pcs_table.heading(key, text=label, command=lambda c=key: self._sort_theta_table(c))
        self.theta_pcs_table.column("symbol", width=90, anchor="w")
        self.theta_pcs_table.column("score", width=70, anchor="center")
        self.theta_pcs_table.column("support_count", width=120, anchor="center")
        self.theta_pcs_table.column("close", width=85, anchor="e")
        self.theta_pcs_table.column("atr", width=85, anchor="e")
        self.theta_pcs_table.column("next_earnings_days", width=145, anchor="center")
        self.theta_pcs_table.column("recommended_strike", width=105, anchor="e")
        self.theta_pcs_table.column("recommended_credit", width=115, anchor="e")
        self.theta_pcs_table.column("primary_strike_band", width=260, anchor="w")
        self.theta_pcs_table.column("liquidity_score", width=115, anchor="center")
        self.theta_pcs_table.pack(side="left", fill="both", expand=True)
        theta_pcs_scroll = ttk.Scrollbar(theta_pcs_frame, orient="vertical", command=self.theta_pcs_table.yview)
        self.theta_pcs_table.configure(yscrollcommand=theta_pcs_scroll.set)
        theta_pcs_scroll.pack(side="right", fill="y")

        theta_details_frame = ttk.LabelFrame(theta_main, text="Details")
        theta_details_frame.pack(fill="both", expand=True)
        self.theta_text = tk.Text(theta_details_frame, wrap="word", font=("Courier New", 10), height=12)
        self.theta_text.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        theta_scroll = ttk.Scrollbar(theta_details_frame, orient="vertical", command=self.theta_text.yview)
        self.theta_text.configure(yscrollcommand=theta_scroll.set)
        theta_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)

        theta_side = ttk.Frame(theta_body, width=340)
        theta_side.pack(side="right", fill="y", padx=(12, 0))
        theta_side.pack_propagate(False)
        theta_symbols_frame = ttk.LabelFrame(theta_side, text="Theta Symbols")
        theta_symbols_frame.pack(fill="x")
        self.theta_symbols_text = tk.Text(theta_symbols_frame, wrap="word", height=10, font=("Courier New", 10))
        self.theta_symbols_text.pack(fill="both", expand=True, padx=8, pady=(8, 8))
        ttk.Button(
            theta_symbols_frame,
            text="Copy Theta Symbols",
            command=self.copy_theta_symbols,
        ).pack(anchor="w", padx=8, pady=(0, 8))
        theta_reason_risk_frame = ttk.LabelFrame(theta_side, text="Reason / Risk")
        theta_reason_risk_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.theta_reason_risk_text = tk.Text(theta_reason_risk_frame, wrap="word", height=14, font=("Courier New", 10))
        self.theta_reason_risk_text.pack(fill="both", expand=True, padx=8, pady=(8, 8))
        self.theta_reason_risk_text.tag_configure("risk_green", foreground="#8FD19E")
        self.theta_reason_risk_text.tag_configure("risk_yellow", foreground="#F0C76E")
        self.theta_reason_risk_text.tag_configure("risk_red", foreground="#FF9A9A")

        market_prep_tab = ttk.Frame(self.notebook)
        self.market_prep_tab = market_prep_tab
        self.notebook.add(market_prep_tab, text="Market Prep")

        market_prep_toolbar = ttk.Frame(market_prep_tab)
        market_prep_toolbar.pack(fill="x", pady=(0, 8))
        ttk.Label(
            market_prep_toolbar,
            text="Market prep copy lists refresh after Master AVWAP scans and when you open this tab.",
            justify="left",
            wraplength=820,
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            market_prep_toolbar,
            text="Copy All Lists",
            command=self.copy_all_market_prep_sections,
        ).pack(side="right", padx=(8, 0))
        ttk.Button(
            market_prep_toolbar,
            text="Refresh",
            command=self.refresh_market_prep_view,
        ).pack(side="right")

        market_prep_body = ttk.Frame(market_prep_tab)
        market_prep_body.pack(fill="both", expand=True)

        market_prep_grid = ttk.Frame(market_prep_body)
        market_prep_grid.pack(side="left", fill="both", expand=True)
        market_prep_grid.columnconfigure(0, weight=1)
        market_prep_grid.columnconfigure(1, weight=1)
        self.market_prep_section_widgets = {}
        self.market_prep_section_frames = {}
        self.market_prep_text_widgets = []
        for idx, definition in enumerate(MARKET_PREP_SECTION_DEFINITIONS):
            row = idx // 2
            column = idx % 2
            market_prep_grid.rowconfigure(row, weight=1)
            section_frame = ttk.LabelFrame(market_prep_grid, text=definition["title"])
            section_frame.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 8 if column == 0 else 0), pady=(0, 8))
            text_height = 5 if definition["id"] in {"earnings_last_night_or_today", "post_earnings_potential_plays"} else 4
            section_text = tk.Text(section_frame, wrap="word", height=text_height, font=("Courier New", 10))
            section_text.pack(fill="both", expand=True, padx=8, pady=(8, 6))
            section_actions = ttk.Frame(section_frame)
            section_actions.pack(fill="x", padx=8, pady=(0, 8))
            section_id = definition["id"]
            ttk.Button(
                section_actions,
                text=definition.get("copy_label", "Copy"),
                command=lambda sid=section_id: self.copy_market_prep_section(sid),
            ).pack(side="left")
            self.market_prep_section_widgets[section_id] = section_text
            self.market_prep_section_frames[section_id] = section_frame
            self.market_prep_text_widgets.append(section_text)

        market_prep_detail_frame = ttk.LabelFrame(market_prep_body, text="Market Prep Details")
        market_prep_detail_frame.pack(side="right", fill="both", expand=True, padx=(12, 0))
        self.market_prep_report_text = tk.Text(market_prep_detail_frame, wrap="word", font=("Courier New", 10), width=56)
        self.market_prep_report_text.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        market_prep_scroll = ttk.Scrollbar(
            market_prep_detail_frame,
            orient="vertical",
            command=self.market_prep_report_text.yview,
        )
        self.market_prep_report_text.configure(yscrollcommand=market_prep_scroll.set)
        market_prep_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)

        self._apply_dark_theme_to_text_widgets()

        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill="x", padx=10, pady=(0, 10))
        self.refresh_tracker_storage_summary()

    def refresh_tracker_storage_summary(self):
        details = get_tracker_storage_details()
        shared_longs_path, shared_shorts_path = get_shared_watchlist_paths()
        self.tracker_storage_dir = Path(details["data_dir"])
        self.tracker_storage_runtime_dir = Path(details["runtime_dir"])
        self.tracker_storage_settings_file = Path(details["settings_file"])
        self.tracker_storage_var.set(
            f"Home folder: {details['data_dir']}\n"
            f"Mutable data: {details['mutable_data_dir']}\n"
            f"Logs: {details['logs_dir']}\n"
            f"Reports: {details['output_dir']}\n"
            f"Runtime tracker data: {details['runtime_dir']}\n"
            f"Local machine cache: {details['local_cache_dir']}\n"
            f"Home-folder longs.txt: {'OK' if shared_longs_path.exists() else 'missing'}\n"
            f"Home-folder shorts.txt: {'OK' if shared_shorts_path.exists() else 'missing'}\n"
            f"Master swinglongs.txt: {'OK' if SWING_LONGS_FILE.exists() else 'missing'}\n"
            f"Master shortswings.txt: {'OK' if SWING_SHORTS_FILE.exists() else 'missing'}\n"
            f"Source: {details['source_label']}"
        )
        self._notify_output_changed()

    def _open_folder_in_explorer(self, path: Path):
        try:
            path.mkdir(parents=True, exist_ok=True)
            open_path_in_file_manager(path)
        except Exception as exc:
            if messagebox:
                messagebox.showerror("Open Folder", f"Could not open folder:\n{path}\n\n{exc}")
            else:
                raise

    def choose_tracker_storage_dir(self):
        if filedialog is None:
            self.status_var.set("Folder picker is not available in this environment.")
            return
        initial_dir = self.tracker_storage_dir if getattr(self, "tracker_storage_dir", None) else Path.home()
        selected = filedialog.askdirectory(
            title="Choose home folder",
            initialdir=str(initial_dir if initial_dir.exists() else Path.home()),
            mustexist=False,
        )
        if not selected:
            return
        target = save_tracker_storage_dir(selected)
        self.refresh_tracker_storage_summary()
        self.status_var.set("Saved home folder. Restart the GUI to use the new location.")
        if messagebox:
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

    def open_tracker_storage_dir(self):
        self.refresh_tracker_storage_summary()
        self._open_folder_in_explorer(self.tracker_storage_dir)

    def open_tracker_settings_file(self):
        self.refresh_tracker_storage_summary()
        self._open_folder_in_explorer(self.tracker_storage_settings_file.parent)

    def _reveal_path_in_explorer(self, path: Path):
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                subprocess.Popen(["explorer", "/select,", str(target)])
            else:
                open_path_in_file_manager(target.parent)
        except Exception as exc:
            if messagebox:
                messagebox.showerror("Open File", f"Could not open file location:\n{path}\n\n{exc}")
            else:
                raise

    def run_scoring_analysis_once(self):
        self.notebook.select(self.tracker_tab)
        self._run_background(
            lambda: run_priority_scoring_tuner(apply_changes=False, min_setups=8, suppress_failures=False),
            "Analyzing setup-tracker scoring signals...",
            "Scoring analysis complete.",
            done_callback=self.refresh_setup_tracker_view,
        )

    def apply_scoring_analysis_once(self):
        self.notebook.select(self.tracker_tab)
        self._run_background(
            lambda: run_priority_scoring_tuner(apply_changes=True, min_setups=8, suppress_failures=False),
            "Applying scoring suggestions from setup-tracker history...",
            "Scoring suggestions applied.",
            done_callback=self.refresh_setup_tracker_view,
        )

    def open_scoring_tuner_report(self):
        self._reveal_path_in_explorer(SCORING_TUNER_REPORT_FILE)

    def open_scoring_config_file(self):
        self._reveal_path_in_explorer(SCORING_CONFIG_FILE)

    def _refresh_active_tab(self):
        selected_tab = self.notebook.select()
        if selected_tab == str(self.tracker_tab):
            self._refresh_setup_tracker_view_if_needed()
        elif selected_tab == str(self.avwap_tab):
            self.refresh_avwap_output_view()
        elif selected_tab == str(self.theta_tab):
            self.refresh_theta_output_view()
        elif selected_tab == str(self.market_prep_tab):
            self.refresh_market_prep_view()

    def _on_notebook_tab_changed(self, _event=None):
        self._refresh_active_tab()

    def _read_text_file(self, path: Path):
        if not path.exists():
            return f"[Missing file] {path.name}"
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            return f"[Error reading {path.name}] {exc}"

    def _set_text_widget_contents(self, widget, text: str):
        set_highlighted_text(widget, text, state_after="normal")

    def _load_ai_state_payload(self) -> dict:
        payload = load_json(AI_STATE_FILE, default={})
        return payload if isinstance(payload, dict) else {}

    def _load_setup_tracker_payload(self) -> dict:
        payload = load_setup_tracker_payload()
        return payload if isinstance(payload, dict) else _default_setup_tracker_payload()

    def _show_setup_tracker_lazy_placeholder(self):
        self.refresh_tracker_storage_summary()
        self.setup_tracker_row_map = {}
        self.setup_tracker_payload = _default_setup_tracker_payload()
        self.setup_tracker_setup_type_rows = []
        self.setup_tracker_playbook_rows = []
        self.setup_tracker_best_playbook_rows = []
        self.setup_tracker_factor_rows = []
        for table in (
            getattr(self, "setup_tracker_table", None),
            getattr(self, "setup_tracker_scenario_table", None),
            getattr(self, "setup_tracker_setup_type_table", None),
            getattr(self, "setup_tracker_playbook_table", None),
            getattr(self, "setup_tracker_factor_table", None),
        ):
            if table is None:
                continue
            for item in table.get_children():
                table.delete(item)
        placeholder = (
            "Setup tracker dashboard is lazy-loaded to keep GUI startup fast.\n\n"
            "Open this tab or click Refresh Tracker to load the historical dashboard. "
            "Live scan scoring still reads setup-tracker history inside run_master even if this dashboard is never opened."
        )
        for widget_name in (
            "setup_tracker_stats_text",
            "setup_tracker_setup_type_text",
            "setup_tracker_playbook_text",
            "setup_tracker_factor_text",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                self._set_text_widget_contents(widget, placeholder)

    def _mark_setup_tracker_view_stale(self):
        self.setup_tracker_view_stale = True
        if not self.setup_tracker_view_loaded:
            return
        if self.notebook.select() == str(self.tracker_tab):
            self.refresh_setup_tracker_view()

    def _refresh_setup_tracker_view_if_needed(self):
        if self.setup_tracker_view_loaded and not self.setup_tracker_view_stale:
            return
        self.refresh_setup_tracker_view()

    def copy_setup_tracker_symbols(self):
        symbols = []
        for item_id in self.setup_tracker_table.get_children():
            values = self.setup_tracker_table.item(item_id, "values")
            if len(values) >= 3 and values[2]:
                symbols.append(str(values[2]).strip().upper())
        symbols = sorted(set(symbols))
        if not symbols:
            self.status_var.set("No setup tracker symbols to copy.")
            return
        self._copy_to_clipboard(", ".join(symbols))
        self.status_var.set("Copied setup tracker symbols to clipboard.")

    def _get_playbooks_for_context(
        self,
        context_id: str,
        preferred_min_closed: int = TRACKER_PLAYBOOK_MIN_CLOSED_SAMPLES,
    ) -> tuple[list[dict], int]:
        playbook_rows = getattr(self, "setup_tracker_playbook_rows", [])
        matching_rows = [
            row
            for row in playbook_rows
            if str(row.get("context_id") or "") == str(context_id or "")
            and int(row.get("closed_setups", 0) or 0) > 0
        ]
        display_rows, min_closed = _filter_tracker_summary_rows(
            matching_rows,
            closed_key="closed_setups",
            preferred_min_closed=preferred_min_closed,
        )
        ranked_rows = sorted(display_rows, key=_tracker_playbook_sort_key)
        return ranked_rows, min_closed

    def _populate_playbook_table(self, rows: list[dict]) -> None:
        self.setup_tracker_playbook_row_map = {}
        for item in self.setup_tracker_playbook_table.get_children():
            self.setup_tracker_playbook_table.delete(item)
        if not rows:
            self._render_playbook_details(None)
            return

        for row in rows:
            robust_closed_r = _coerce_float(row.get("robust_closed_r"))
            robust_closed_r_edge = _coerce_float(row.get("robust_closed_r_edge"))
            win_rate = _coerce_float(row.get("win_rate_closed"))
            values = (
                row.get("type_label", ""),
                row.get("stop_reference_label", ""),
                row.get("profit_take_summary", ""),
                int(row.get("closed_setups", 0) or 0),
                "" if robust_closed_r is None else f"{robust_closed_r:.2f}",
                "" if robust_closed_r_edge is None else f"{robust_closed_r_edge:+.2f}",
                "" if win_rate is None else f"{win_rate * 100:.0f}%",
            )
            item_id = self.setup_tracker_playbook_table.insert(
                "",
                "end",
                values=values,
                tags=tree_tags_for_values(values),
            )
            self.setup_tracker_playbook_row_map[item_id] = row

        first_item = self.setup_tracker_playbook_table.get_children()
        if first_item:
            self.setup_tracker_playbook_table.selection_set(first_item[0])
            self.setup_tracker_playbook_table.focus(first_item[0])
            self._on_playbook_selected()
        else:
            self._render_playbook_details(None)

    def _render_playbook_details(self, row: dict | None) -> None:
        if not row:
            self._set_text_widget_contents(
                self.setup_tracker_playbook_text,
                "No playbook recommendations yet.",
            )
            return

        def _fmt_r(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric:.2f}R" if numeric is not None else "n/a"

        def _fmt_r_edge(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric:+.2f}R" if numeric is not None else "n/a"

        def _fmt_pct(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric * 100:.0f}%" if numeric is not None else "n/a"

        matching_rows, min_closed = self._get_playbooks_for_context(
            str(row.get("context_id") or "")
        )
        framework_label = "Experimental comparison" if row.get("experimental") else "Baseline"
        lines = [
            str(row.get("type_label") or "Playbook"),
            "-" * 80,
            f"Recommended playbook: {row.get('playbook_label')}",
            (
                f"Framework={framework_label}"
                f" | closed filter used={max(0, int(row.get('context_min_closed_used', min_closed) or 0))}"
                f" | alternatives={max(0, int(row.get('context_option_count', len(matching_rows)) or 0) - 1)}"
            ),
            (
                f"Tracked={int(row.get('tracked_setups', 0) or 0)} | "
                f"with_closed={int(row.get('closed_setups', 0) or 0)} | "
                f"open={int(row.get('open_setups', 0) or 0)}"
            ),
            (
                f"Robust closed R={_fmt_r(row.get('robust_closed_r'))} "
                f"vs context {_fmt_r(row.get('baseline_robust_closed_r'))} "
                f"({_fmt_r_edge(row.get('robust_closed_r_edge'))})"
            ),
            (
                f"Median closed R={_fmt_r(row.get('median_closed_r'))} "
                f"| raw avg closed R={_fmt_r(row.get('raw_avg_closed_r'))}"
            ),
            (
                f"Win rate={_fmt_pct(row.get('win_rate_closed'))} "
                f"| target hit={_fmt_pct(row.get('target_hit_rate'))} "
                f"| stop rate={_fmt_pct(row.get('stop_rate'))}"
            ),
        ]
        if _coerce_float(row.get("robust_total_r")) is not None:
            lines.append(f"Robust total R including open setups: {_fmt_r(row.get('robust_total_r'))}")
        if row.get("framework_notes"):
            lines.append(f"Notes: {row.get('framework_notes')}")
        if row.get("outlier_warning"):
            lines.append(f"Note: {row.get('outlier_warning')}")

        alternatives = [
            item
            for item in matching_rows
            if str(item.get("playbook_label") or "") != str(row.get("playbook_label") or "")
        ]
        if alternatives:
            lines.append("")
            lines.append("Next-best options in this setup type")
            lines.append("-" * 80)
            for item in alternatives[:4]:
                lines.append(
                    f"{item.get('playbook_label')}: "
                    f"closed={int(item.get('closed_setups', 0) or 0)} "
                    f"robust={_fmt_r(item.get('robust_closed_r'))} "
                    f"edge={_fmt_r_edge(item.get('robust_closed_r_edge'))} "
                    f"win={_fmt_pct(item.get('win_rate_closed'))}"
                )

        if row.get("sample_setups"):
            lines.append("")
            lines.append("Recent examples")
            lines.append("-" * 80)
            for item in str(row.get("sample_setups") or "").split("; "):
                if item:
                    lines.append(item)

        self._set_text_widget_contents(self.setup_tracker_playbook_text, "\n".join(lines))

    def _on_playbook_selected(self, _event=None) -> None:
        selected = self.setup_tracker_playbook_table.selection()
        row = self.setup_tracker_playbook_row_map.get(selected[0]) if selected else None
        self._render_playbook_details(row)

    def _render_setup_tracker_stats_text(self, payload: dict, selected_setup: dict | None = None):
        stats_rows = payload.get("stats", []) if isinstance(payload.get("stats"), list) else []
        setup_type_rows = getattr(
            self,
            "setup_tracker_setup_type_rows",
            payload.get("setup_type_stats", []) if isinstance(payload.get("setup_type_stats"), list) else [],
        )
        factor_rows = getattr(self, "setup_tracker_factor_rows", [])
        playbook_rows = getattr(self, "setup_tracker_playbook_rows", [])
        best_playbook_rows = getattr(self, "setup_tracker_best_playbook_rows", [])
        tracker_setups = payload.get("setups", {}) if isinstance(payload.get("setups"), dict) else {}
        daily_watchlists = payload.get("daily_watchlists", {}) if isinstance(payload.get("daily_watchlists"), dict) else {}
        attribute_registry = payload.get("attribute_registry", {}) if isinstance(payload.get("attribute_registry"), dict) else {}
        recent_family_rows = build_recent_tracker_setup_family_rows(tracker_setups)

        def _fmt_r(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric:.2f}R" if numeric is not None else "n/a"

        def _fmt_r_edge(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric:+.2f}R" if numeric is not None else "n/a"

        def _fmt_pct(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric * 100:.0f}%" if numeric is not None else "n/a"

        def _fmt_pct_edge(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric * 100:+.0f} pts" if numeric is not None else "n/a"

        lines = []
        lines.append("Tracker summary")
        lines.append("-" * 80)
        lines.append(f"Updated: {payload.get('updated_at') or 'n/a'}")
        lines.append(f"Tracked setups: {len(tracker_setups)}")
        lines.append("Auto scoring tuner: enabled after tracker updates/backfills")
        if daily_watchlists:
            latest_date = max(daily_watchlists.keys())
            latest_watchlist = daily_watchlists.get(latest_date, {})
            lines.append(
                f"Latest final scan day: {latest_date} | symbols={len(latest_watchlist.get('symbols', []) or [])}"
            )
            lines.append("")
            lines.append("Recent daily watchlists")
            lines.append("-" * 80)
            for watchlist_date in sorted(daily_watchlists.keys(), reverse=True)[:8]:
                watchlist = daily_watchlists.get(watchlist_date, {})
                symbols = [str(symbol).strip().upper() for symbol in watchlist.get("symbols", []) if str(symbol).strip()]
                preview = ", ".join(symbols[:10]) if symbols else "None"
                if len(symbols) > 10:
                    preview += f" (+{len(symbols) - 10} more)"
                lines.append(f"{watchlist_date}: {preview}")
        lines.append("")
        lines.append("Recent setup families driving scan rank")
        lines.append("-" * 80)
        positive_recent_family_rows = [
            row for row in recent_family_rows if int(row.get("score_delta", 0) or 0) > 0
        ]
        if not positive_recent_family_rows:
            lines.append("Not enough recent tracker history yet for family-level score boosts.")
        else:
            for row in positive_recent_family_rows[:6]:
                lines.append(
                    f"{row.get('type_label')}: delta={int(row.get('score_delta', 0) or 0):+d} "
                    f"closed={int(row.get('closed_setups', 0) or 0)} "
                    f"tracked={int(row.get('tracked_setups', 0) or 0)} "
                    f"avg_closed={_fmt_r(row.get('avg_closed_r'))} "
                    f"edge={_fmt_r_edge(row.get('avg_closed_r_edge'))}"
                )
        lines.append("")
        lines.append("Best playbooks right now")
        lines.append("-" * 80)
        if not best_playbook_rows:
            lines.append("No playbook recommendations yet.")
        else:
            display_best_rows, best_min_closed = _filter_tracker_summary_rows(
                best_playbook_rows,
                closed_key="closed_setups",
                preferred_min_closed=TRACKER_PLAYBOOK_MIN_CLOSED_SAMPLES,
            )
            if best_min_closed > 0:
                lines.append(
                    f"Showing setup-type recommendations with >= {best_min_closed} closed setups when possible."
                )
            for row in display_best_rows[:6]:
                lines.append(
                    f"{row.get('type_label')}:"
                )
                lines.append(
                    f"  {row.get('playbook_label')} | "
                    f"closed={int(row.get('closed_setups', 0) or 0)} "
                    f"robust={_fmt_r(row.get('robust_closed_r'))} "
                    f"edge={_fmt_r_edge(row.get('robust_closed_r_edge'))} "
                    f"win={_fmt_pct(row.get('win_rate_closed'))}"
                )

        lines.append("")
        lines.append("Top setup types")
        lines.append("-" * 80)
        display_setup_type_rows, setup_type_min_closed = _filter_tracker_summary_rows(
            setup_type_rows,
            closed_key="closed_setups",
        )
        if not display_setup_type_rows:
            lines.append("No setup-type stats yet.")
        else:
            if setup_type_min_closed > 0:
                lines.append(
                    f"Showing rows with >= {setup_type_min_closed} closed setups."
                )
            for row in display_setup_type_rows[:6]:
                lines.append(
                    f"{row.get('type_label')} (#"
                    f"{int(row.get('rank_within_side_bucket', 0) or 0)}/"
                    f"{int(row.get('rank_group_size', 0) or 0)}): tracked={row.get('tracked_setups', 0)} "
                    f"closed={row.get('closed_setups', 0)} avg_closed={_fmt_r(row.get('avg_closed_r'))} "
                    f"edge={_fmt_r_edge(row.get('avg_closed_r_edge'))} hit={_fmt_pct(row.get('target_hit_rate'))}"
                )

        lines.append("")
        lines.append("Top scenario groups")
        lines.append("-" * 80)
        display_stats_rows, stats_min_closed = _filter_tracker_summary_rows(
            stats_rows,
            closed_key="closed_setups",
        )
        if not display_stats_rows:
            lines.append("No scenario-group stats yet.")
        else:
            if stats_min_closed > 0:
                lines.append(
                    f"Showing rows with >= {stats_min_closed} closed setups, ranked by avg closed R."
                )
            for row in display_stats_rows[:6]:
                scenario_label = str(row.get("exit_template_label") or row.get("exit_template_id") or "").strip()
                lines.append(
                    f"{row.get('stop_reference_label')} / {scenario_label}: "
                    f"closed={row.get('closed_setups', 0)} open={row.get('open_setups', 0)} "
                    f"avg_closed={_fmt_r(row.get('avg_closed_r'))} "
                    f"avg_total={_fmt_r(row.get('avg_total_r'))} "
                    f"bias={_fmt_r_edge(row.get('open_distortion'))} "
                    f"win={_fmt_pct(row.get('win_rate_closed'))}"
                )

        positive_factor_rows = [
            row
            for row in factor_rows
            if int(row.get("closed_tradeable_setup_count", 0) or 0) >= TRACKER_SUMMARY_MIN_CLOSED_SAMPLES
            if (_coerce_float(row.get("success_edge")) or 0.0) > 0
        ]
        negative_factor_rows = [
            row
            for row in factor_rows
            if int(row.get("closed_tradeable_setup_count", 0) or 0) >= TRACKER_SUMMARY_MIN_CLOSED_SAMPLES
            if (_coerce_float(row.get("success_edge")) or 0.0) < 0
        ]
        if not positive_factor_rows:
            positive_factor_rows = [
                row
                for row in factor_rows
                if (_coerce_float(row.get("success_edge")) or 0.0) > 0
            ]
        if not negative_factor_rows:
            negative_factor_rows = [
                row
                for row in factor_rows
                if (_coerce_float(row.get("success_edge")) or 0.0) < 0
            ]
        positive_factor_rows = sorted(
            positive_factor_rows,
            key=lambda row: (
                -(_coerce_float(row.get("success_edge")) or 0.0),
                -int(row.get("closed_tradeable_setup_count", 0) or 0),
            ),
        )
        negative_factor_rows = sorted(
            negative_factor_rows,
            key=lambda row: (
                (_coerce_float(row.get("success_edge")) or 0.0),
                -int(row.get("closed_tradeable_setup_count", 0) or 0),
            ),
        )

        lines.append("")
        lines.append("Positive factor edges")
        lines.append("-" * 80)
        if not positive_factor_rows:
            lines.append("No positive factor edges yet.")
        else:
            for row in positive_factor_rows[:5]:
                lines.append(
                    f"{row.get('side')} {row.get('priority_bucket')} | "
                    f"{row.get('attribute_label')}={row.get('value_label')}: "
                    f"closed={row.get('closed_tradeable_setup_count', 0)} "
                    f"success={_fmt_r_edge(row.get('success_edge'))} "
                    f"r_edge={_fmt_r_edge(row.get('avg_closed_r_edge'))} "
                    f"hit_edge={_fmt_pct_edge(row.get('target_hit_rate_edge'))} "
                    f"stop_edge={_fmt_pct_edge(row.get('stop_rate_edge'))}"
                )

        lines.append("")
        lines.append("Negative factor edges")
        lines.append("-" * 80)
        if not negative_factor_rows:
            lines.append("No negative factor edges yet.")
        else:
            for row in negative_factor_rows[:5]:
                lines.append(
                    f"{row.get('side')} {row.get('priority_bucket')} | "
                    f"{row.get('attribute_label')}={row.get('value_label')}: "
                    f"closed={row.get('closed_tradeable_setup_count', 0)} "
                    f"success={_fmt_r_edge(row.get('success_edge'))} "
                    f"r_edge={_fmt_r_edge(row.get('avg_closed_r_edge'))} "
                    f"hit_edge={_fmt_pct_edge(row.get('target_hit_rate_edge'))} "
                    f"stop_edge={_fmt_pct_edge(row.get('stop_rate_edge'))}"
                )

        if selected_setup:
            lines.append("")
            lines.append("Selected setup")
            lines.append("-" * 80)
            lines.append(
                f"{selected_setup.get('symbol')} {selected_setup.get('side')} "
                f"{selected_setup.get('priority_bucket', '')} scan={selected_setup.get('scan_date')}"
            )
            lines.append(
                f"entry={_coerce_float(selected_setup.get('entry_price')) or 0:.2f} "
                f"anchor={selected_setup.get('anchor_date')} "
                f"score={_coerce_float(selected_setup.get('priority_score')) or 0:.1f}"
            )
            selected_outcome = _summarize_tracker_setup_outcome(selected_setup)
            selected_days_since_scan = _tracker_days_since_scan(str(selected_setup.get("scan_date") or ""))
            lines.append(
                f"age={selected_days_since_scan if selected_days_since_scan is not None else 'n/a'}d "
                f"closed={int(selected_outcome.get('closed_tradeable_scenario_count', 0) or 0)} "
                f"open={int(selected_outcome.get('open_tradeable_scenario_count', 0) or 0)} "
                f"avg_closed={_fmt_r(selected_outcome.get('avg_closed_r'))} "
                f"avg_total={_fmt_r(selected_outcome.get('avg_total_r'))} "
                f"bias={_fmt_r_edge(selected_outcome.get('open_distortion'))}"
            )
            max_days_held = int(selected_outcome.get("max_days_held", 0) or 0)
            if max_days_held > 0:
                lines.append(f"max scenario hold={max_days_held} day(s)")
            if selected_setup.get("favorite_zone"):
                lines.append(f"zone={selected_setup.get('favorite_zone')}")
            if selected_setup.get("retest_note"):
                lines.append(f"retest={selected_setup.get('retest_note')}")
            if selected_setup.get("extreme_move_note"):
                lines.append(f"extreme_move={selected_setup.get('extreme_move_note')}")
            if selected_setup.get("extension_note"):
                lines.append(f"extension={selected_setup.get('extension_note')}")
            if selected_setup.get("first_dev_note"):
                lines.append(f"1stdev={selected_setup.get('first_dev_note')}")
            if selected_setup.get("score_bonus_note"):
                lines.append(f"bonus={selected_setup.get('score_bonus_note')}")
            if selected_setup.get("adaptive_score_note"):
                lines.append(f"adaptive={selected_setup.get('adaptive_score_note')}")
            if selected_setup.get("recent_tracker_score_note"):
                lines.append(f"recent_tracker={selected_setup.get('recent_tracker_score_note')}")
            if selected_setup.get("setup_type_score_note"):
                lines.append(f"setup_type={selected_setup.get('setup_type_score_note')}")
            if selected_setup.get("ranking_note"):
                lines.append(f"filters={selected_setup.get('ranking_note')}")
            if selected_setup.get("compression_note"):
                lines.append(f"compression={selected_setup.get('compression_note')}")

            selected_context_id = _tracker_setup_context(selected_setup).get("context_id", "")
            matching_playbooks, matching_min_closed = self._get_playbooks_for_context(str(selected_context_id))
            if matching_playbooks:
                lines.append("")
                lines.append("Best matching stop / take-profit playbooks")
                lines.append("-" * 80)
                if matching_min_closed > 0:
                    lines.append(
                        f"Ranking this setup against playbooks with >= {matching_min_closed} closed setups when possible."
                    )
                for row in matching_playbooks[:4]:
                    lines.append(
                        f"{row.get('playbook_label')}: "
                        f"closed={int(row.get('closed_setups', 0) or 0)} "
                        f"robust={_fmt_r(row.get('robust_closed_r'))} "
                        f"median={_fmt_r(row.get('median_closed_r'))} "
                        f"edge={_fmt_r_edge(row.get('robust_closed_r_edge'))} "
                        f"win={_fmt_pct(row.get('win_rate_closed'))}"
                    )
            elif playbook_rows:
                broader_matches = [
                    row
                    for row in playbook_rows
                    if str(row.get("side") or "") == str(selected_setup.get("side") or "")
                    and str(row.get("priority_bucket") or "") == str(selected_setup.get("priority_bucket") or "")
                    and int(row.get("closed_setups", 0) or 0) > 0
                ]
                broader_matches, _ = _filter_tracker_summary_rows(
                    broader_matches,
                    closed_key="closed_setups",
                    preferred_min_closed=TRACKER_PLAYBOOK_MIN_CLOSED_SAMPLES,
                )
                broader_matches = sorted(broader_matches, key=_tracker_playbook_sort_key)
                if broader_matches:
                    lines.append("")
                    lines.append("Best broader-context playbooks")
                    lines.append("-" * 80)
                    lines.append(
                        "No exact setup-type match had enough history yet, so this is using the broader side + bucket context."
                    )
                    for row in broader_matches[:3]:
                        lines.append(
                            f"{row.get('type_label')} -> {row.get('playbook_label')}: "
                            f"closed={int(row.get('closed_setups', 0) or 0)} "
                            f"robust={_fmt_r(row.get('robust_closed_r'))} "
                            f"win={_fmt_pct(row.get('win_rate_closed'))}"
                        )

            entry_attributes = selected_setup.get("entry_attributes", {})
            if isinstance(entry_attributes, dict) and entry_attributes:
                lines.append("")
                lines.append("Entry attributes")
                lines.append("-" * 80)
                rendered_count = 0
                for attribute_key in sorted(entry_attributes.keys()):
                    meta = attribute_registry.get(attribute_key, {}) if isinstance(attribute_registry.get(attribute_key), dict) else {}
                    label = meta.get("label") or attribute_key
                    group = meta.get("group") or attribute_key.split(".", 1)[0]
                    lines.append(f"[{group}] {label}: {_format_tracker_attribute_text(entry_attributes[attribute_key])}")
                    rendered_count += 1
                    if rendered_count >= 18:
                        remaining = len(entry_attributes) - rendered_count
                        if remaining > 0:
                            lines.append(f"... {remaining} more attributes logged in {SETUP_ATTRIBUTES_FILE.name}")
                        break

        lines.append("")
        lines.append("Exports")
        lines.append("-" * 80)
        lines.append(f"Tracker JSON: {SETUP_TRACKER_FILE}")
        lines.append(f"Scenario CSV: {SETUP_SCENARIOS_FILE}")
        lines.append(f"Daily CSV: {SETUP_DAILY_FILE}")
        lines.append(f"Stats CSV: {SETUP_STATS_FILE}")
        lines.append(f"Setup type stats CSV: {SETUP_TYPE_STATS_FILE}")
        lines.append(f"Playbook stats CSV: {SETUP_PLAYBOOKS_FILE}")
        lines.append(f"Attributes CSV: {SETUP_ATTRIBUTES_FILE}")
        lines.append(f"Attribute leaderboard CSV: {SETUP_ATTRIBUTE_LEADERBOARD_FILE}")
        lines.append(f"Scoring config JSON: {SCORING_CONFIG_FILE}")
        lines.append(f"Scoring recommendations JSON: {SCORING_RECOMMENDATIONS_FILE}")
        lines.append(f"Scoring tuner report: {SCORING_TUNER_REPORT_FILE}")
        self._set_text_widget_contents(self.setup_tracker_stats_text, "\n".join(lines))

    def _populate_setup_type_table(self, rows: list[dict]) -> None:
        self.setup_tracker_setup_type_row_map = {}
        for item in self.setup_tracker_setup_type_table.get_children():
            self.setup_tracker_setup_type_table.delete(item)
        if not rows:
            self._render_setup_type_details(None)
            return

        for row in rows:
            avg_closed_r = _coerce_float(row.get("avg_closed_r"))
            avg_closed_r_edge = _coerce_float(row.get("avg_closed_r_edge"))
            hit_rate = _coerce_float(row.get("target_hit_rate"))
            values = (
                row.get("type_label", ""),
                int(row.get("tracked_setups", 0) or 0),
                int(row.get("closed_setups", 0) or 0),
                "" if avg_closed_r is None else f"{avg_closed_r:.2f}",
                "" if avg_closed_r_edge is None else f"{avg_closed_r_edge:+.2f}",
                "" if hit_rate is None else f"{hit_rate * 100:.0f}%",
            )
            item_id = self.setup_tracker_setup_type_table.insert(
                "",
                "end",
                values=values,
                tags=tree_tags_for_values(values),
            )
            self.setup_tracker_setup_type_row_map[item_id] = row

        first_item = self.setup_tracker_setup_type_table.get_children()
        if first_item:
            self.setup_tracker_setup_type_table.selection_set(first_item[0])
            self.setup_tracker_setup_type_table.focus(first_item[0])
            self._on_setup_type_selected()
        else:
            self._render_setup_type_details(None)

    def _render_setup_type_details(self, row: dict | None) -> None:
        if not row:
            self._set_text_widget_contents(
                self.setup_tracker_setup_type_text,
                "No setup-type stats yet.",
            )
            return

        def _fmt_r(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric:.2f}R" if numeric is not None else "n/a"

        def _fmt_r_edge(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric:+.2f}R" if numeric is not None else "n/a"

        def _fmt_pct(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric * 100:.0f}%" if numeric is not None else "n/a"

        def _fmt_pct_edge(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric * 100:+.0f} pts" if numeric is not None else "n/a"

        avg_priority_score = _coerce_float(row.get("avg_priority_score"))
        avg_priority_score_text = f"{avg_priority_score:.1f}" if avg_priority_score is not None else "n/a"
        lines = [
            str(row.get("type_label") or "Setup type"),
            "-" * 80,
            (
                f"Historical rank=#{int(row.get('rank_within_side_bucket', 0) or 0)}/"
                f"{int(row.get('rank_group_size', 0) or 0)} within "
                f"{row.get('side', '')} {row.get('priority_bucket', '')}"
                f" | tracker delta={int(row.get('score_delta', 0) or 0):+d}"
            ),
            (
                f"Tracked={int(row.get('tracked_setups', 0) or 0)} | "
                f"tradeable={int(row.get('tradeable_setups', 0) or 0)} | "
                f"with_closed={int(row.get('closed_setups', 0) or 0)} | "
                f"open={int(row.get('open_setups', 0) or 0)}"
            ),
            (
                f"Avg closed R={_fmt_r(row.get('avg_closed_r'))} "
                f"vs baseline {_fmt_r(row.get('baseline_avg_closed_r'))} "
                f"({_fmt_r_edge(row.get('avg_closed_r_edge'))})"
            ),
            (
                f"Target hit={_fmt_pct(row.get('target_hit_rate'))} "
                f"vs baseline {_fmt_pct(row.get('baseline_target_hit_rate'))} "
                f"({_fmt_pct_edge(row.get('target_hit_rate_edge'))})"
            ),
            (
                f"Stop rate={_fmt_pct(row.get('stop_rate'))} "
                f"vs baseline {_fmt_pct(row.get('baseline_stop_rate'))} "
                f"({_fmt_pct_edge(row.get('stop_rate_edge'))})"
            ),
            (
                f"Avg priority score={avg_priority_score_text}"
                f" | avg total R={_fmt_r(row.get('avg_total_r'))}"
            ),
        ]
        if row.get("current_band_zone_summary"):
            lines.append(f"Common band zones: {row.get('current_band_zone_summary')}")
        if row.get("trend_summary"):
            lines.append(f"Common trends: {row.get('trend_summary')}")
        if _coerce_float(row.get("previous_day_range_break_rate")) is not None:
            lines.append(
                f"Previous-day range break rate: {_fmt_pct(row.get('previous_day_range_break_rate'))}"
            )
        if _coerce_float(row.get("extreme_move_rate")) is not None:
            lines.append(f"Extreme-move rate: {_fmt_pct(row.get('extreme_move_rate'))}")
        matching_playbooks, matching_min_closed = self._get_playbooks_for_context(
            str(row.get("setup_type_id") or "")
        )
        if matching_playbooks:
            lines.append("")
            lines.append("Best stop / take-profit playbooks")
            lines.append("-" * 80)
            if matching_min_closed > 0:
                lines.append(
                    f"Showing playbooks with >= {matching_min_closed} closed setups when possible."
                )
            for item in matching_playbooks[:4]:
                lines.append(
                    f"{item.get('playbook_label')}: "
                    f"closed={int(item.get('closed_setups', 0) or 0)} "
                    f"robust={_fmt_r(item.get('robust_closed_r'))} "
                    f"median={_fmt_r(item.get('median_closed_r'))} "
                    f"edge={_fmt_r_edge(item.get('robust_closed_r_edge'))} "
                    f"win={_fmt_pct(item.get('win_rate_closed'))}"
                )
        if row.get("sample_setups"):
            lines.append("")
            lines.append("Recent examples")
            lines.append("-" * 80)
            for item in str(row.get("sample_setups") or "").split("; "):
                if item:
                    lines.append(item)
        self._set_text_widget_contents(self.setup_tracker_setup_type_text, "\n".join(lines))

    def _on_setup_type_selected(self, _event=None) -> None:
        selected = self.setup_tracker_setup_type_table.selection()
        row = self.setup_tracker_setup_type_row_map.get(selected[0]) if selected else None
        self._render_setup_type_details(row)

    def _populate_factor_table(self, rows: list[dict]) -> None:
        self.setup_tracker_factor_row_map = {}
        for item in self.setup_tracker_factor_table.get_children():
            self.setup_tracker_factor_table.delete(item)
        if not rows:
            self._render_factor_details(None)
            return

        for row in rows:
            avg_closed_r = _coerce_float(row.get("avg_closed_r"))
            avg_closed_r_edge = _coerce_float(row.get("avg_closed_r_edge"))
            success_edge = _coerce_float(row.get("success_edge"))
            impact_score = _coerce_float(row.get("impact_score"))
            values = (
                row.get("attribute_label", ""),
                row.get("value_label", ""),
                int(row.get("setup_count", 0) or 0),
                int(row.get("closed_tradeable_setup_count", 0) or 0),
                "" if avg_closed_r is None else f"{avg_closed_r:.2f}",
                "" if avg_closed_r_edge is None else f"{avg_closed_r_edge:+.2f}",
                "" if success_edge is None else f"{success_edge:+.2f}",
                "" if impact_score is None else f"{impact_score:.2f}",
            )
            item_id = self.setup_tracker_factor_table.insert(
                "",
                "end",
                values=values,
                tags=tree_tags_for_values(values),
            )
            self.setup_tracker_factor_row_map[item_id] = row

        first_item = self.setup_tracker_factor_table.get_children()
        if first_item:
            self.setup_tracker_factor_table.selection_set(first_item[0])
            self.setup_tracker_factor_table.focus(first_item[0])
            self._on_factor_selected()
        else:
            self._render_factor_details(None)

    def _render_factor_details(self, row: dict | None) -> None:
        if not row:
            self._set_text_widget_contents(
                self.setup_tracker_factor_text,
                "No factor-impact stats yet.",
            )
            return

        def _fmt_r(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric:.2f}R" if numeric is not None else "n/a"

        def _fmt_r_edge(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric:+.2f}R" if numeric is not None else "n/a"

        def _fmt_pct(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric * 100:.0f}%" if numeric is not None else "n/a"

        def _fmt_pct_edge(value) -> str:
            numeric = _coerce_float(value)
            return f"{numeric * 100:+.0f} pts" if numeric is not None else "n/a"

        coverage_pct = _coerce_float(row.get("coverage_pct"))
        coverage_text = f"{coverage_pct:.1f}%" if coverage_pct is not None else "n/a"
        success_edge = _coerce_float(row.get("success_edge"))
        success_text = f"{success_edge:+.2f}R" if success_edge is not None else "n/a"
        impact_score = _coerce_float(row.get("impact_score"))
        impact_text = f"{impact_score:.2f}" if impact_score is not None else "n/a"
        lines = [
            f"[{row.get('attribute_group')}] {row.get('attribute_label')} = {row.get('value_label')}",
            "-" * 80,
            f"Context: {row.get('side')} {row.get('priority_bucket')}",
            (
                f"Setups={int(row.get('setup_count', 0) or 0)} | "
                f"tradeable={int(row.get('tradeable_setup_count', 0) or 0)} | "
                f"with_closed={int(row.get('closed_tradeable_setup_count', 0) or 0)} | "
                f"coverage={coverage_text}"
            ),
            (
                f"Avg closed R={_fmt_r(row.get('avg_closed_r'))} "
                f"vs baseline {_fmt_r(row.get('baseline_avg_closed_r'))} "
                f"({_fmt_r_edge(row.get('avg_closed_r_edge'))})"
            ),
            (
                f"Target hit={_fmt_pct(row.get('target_hit_rate'))} "
                f"vs baseline {_fmt_pct(row.get('baseline_target_hit_rate'))} "
                f"({_fmt_pct_edge(row.get('target_hit_rate_edge'))})"
            ),
            (
                f"Stop rate={_fmt_pct(row.get('stop_rate'))} "
                f"vs baseline {_fmt_pct(row.get('baseline_stop_rate'))} "
                f"({_fmt_pct_edge(row.get('stop_rate_edge'))})"
            ),
            f"Success edge={success_text}",
            f"Impact score={impact_text}",
        ]
        if row.get("attribute_description"):
            lines.append(f"Description: {row.get('attribute_description')}")
        if row.get("sample_setups"):
            lines.append("")
            lines.append("Recent examples")
            lines.append("-" * 80)
            for item in str(row.get("sample_setups") or "").split("; "):
                if item:
                    lines.append(item)
        self._set_text_widget_contents(self.setup_tracker_factor_text, "\n".join(lines))

    def _on_factor_selected(self, _event=None) -> None:
        selected = self.setup_tracker_factor_table.selection()
        row = self.setup_tracker_factor_row_map.get(selected[0]) if selected else None
        self._render_factor_details(row)

    def _populate_setup_tracker_scenarios(self, setup: dict | None):
        for item in self.setup_tracker_scenario_table.get_children():
            self.setup_tracker_scenario_table.delete(item)
        if not setup:
            return

        scenarios = list((setup.get("scenarios") or {}).values())
        scenarios.sort(key=lambda item: (not _scenario_is_open(item.get("status", "")), -(float(item.get("total_r", 0.0) or 0.0)), item.get("scenario_id", "")))
        for scenario in scenarios:
            total_r = _coerce_float(scenario.get("total_r"))
            total_pnl = _coerce_float(scenario.get("total_pnl"))
            exit_summary = _format_tracker_exit_summary(
                partial_target_label=scenario.get("partial_target_label"),
                final_target_label=scenario.get("final_target_label"),
                trail_after_partial_label=scenario.get("trail_after_partial_label"),
                fallback_label=str(scenario.get("exit_template_label") or ""),
                hard_stop_r_multiple=scenario.get("hard_stop_r_multiple"),
            )
            values = (
                scenario.get("stop_reference_label", ""),
                exit_summary,
                int(scenario.get("shares", 0) or 0),
                scenario.get("status", ""),
                "" if total_r is None else f"{total_r:.2f}",
                "" if total_pnl is None else f"{total_pnl:.2f}",
                scenario.get("last_action", ""),
            )
            self.setup_tracker_scenario_table.insert(
                "",
                "end",
                values=values,
                tags=tree_tags_for_values(values),
            )

    def _on_setup_tracker_selected(self, _event=None):
        selected = self.setup_tracker_table.selection()
        setup = self.setup_tracker_row_map.get(selected[0]) if selected else None
        self._populate_setup_tracker_scenarios(setup)
        self._render_setup_tracker_stats_text(self.setup_tracker_payload, setup)

    def refresh_setup_tracker_view(self):
        self.status_var.set("Loading setup tracker dashboard...")
        self.root.update_idletasks()
        self.refresh_tracker_storage_summary()
        payload = self._load_setup_tracker_payload()
        self.setup_tracker_payload = payload
        self.setup_tracker_row_map = {}
        tracker_setups = payload.get("setups", {}) if isinstance(payload.get("setups"), dict) else {}
        attribute_registry = payload.get("attribute_registry", {}) if isinstance(payload.get("attribute_registry"), dict) else {}
        attribute_rows = _flatten_tracker_attributes(tracker_setups, attribute_registry)
        attribute_leaderboard_rows = _build_tracker_attribute_leaderboard_rows(attribute_rows)
        self.setup_tracker_setup_type_rows = (
            payload.get("setup_type_stats", [])
            if isinstance(payload.get("setup_type_stats"), list) and payload.get("setup_type_stats")
            else build_tracker_setup_type_rows(tracker_setups)
        )
        self.setup_tracker_playbook_rows = build_tracker_playbook_rows(tracker_setups)
        self.setup_tracker_best_playbook_rows = build_tracker_playbook_recommendation_rows(
            self.setup_tracker_playbook_rows
        )
        self.setup_tracker_factor_rows = build_tracker_factor_view_rows(attribute_leaderboard_rows)
        self._populate_playbook_table(self.setup_tracker_best_playbook_rows)
        self._populate_setup_type_table(self.setup_tracker_setup_type_rows)
        self._populate_factor_table(self.setup_tracker_factor_rows)

        for item in self.setup_tracker_table.get_children():
            self.setup_tracker_table.delete(item)

        setups = list((payload.get("setups") or {}).values())
        setups.sort(
            key=lambda setup: (
                setup.get("setup_status") != "OPEN",
                str(setup.get("scan_date") or ""),
                str(setup.get("symbol") or ""),
            ),
            reverse=False,
        )
        setups = sorted(
            setups,
            key=lambda setup: (
                setup.get("setup_status") != "OPEN",
                str(setup.get("scan_date") or ""),
                str(setup.get("symbol") or ""),
            ),
        )
        setups = sorted(
            setups,
            key=lambda setup: (
                setup.get("setup_status") != "OPEN",
                -(float(setup.get("priority_score", 0) or 0.0)),
                str(setup.get("symbol") or ""),
            ),
        )

        today = datetime.now().date()
        for setup in setups:
            outcome_summary = _summarize_tracker_setup_outcome(setup)
            days_since_scan = _tracker_days_since_scan(str(setup.get("scan_date") or ""), reference=today)
            avg_closed_r = _coerce_float(outcome_summary.get("avg_closed_r"))
            open_bias = _coerce_float(outcome_summary.get("open_distortion"))
            values = (
                setup.get("scan_date", ""),
                "" if days_since_scan is None else str(days_since_scan),
                setup.get("symbol", ""),
                setup.get("side", ""),
                setup.get("priority_bucket", ""),
                setup.get("setup_status", ""),
                int(outcome_summary.get("closed_tradeable_scenario_count", 0) or 0),
                int(setup.get("open_scenario_count", 0) or 0),
                "" if avg_closed_r is None else f"{avg_closed_r:.2f}",
                "" if open_bias is None else f"{open_bias:+.2f}",
                "" if _coerce_float(setup.get("priority_score")) is None else f"{float(setup.get('priority_score')):.1f}",
                setup.get("retest_reference_level", ""),
                "Y" if setup.get("compression_flag") else "",
            )
            item_id = self.setup_tracker_table.insert(
                "",
                "end",
                values=values,
                tags=tree_tags_for_values(values),
            )
            self.setup_tracker_row_map[item_id] = setup

        first_item = self.setup_tracker_table.get_children()
        if first_item:
            self.setup_tracker_table.selection_set(first_item[0])
            self.setup_tracker_table.focus(first_item[0])
            self._on_setup_tracker_selected()
        else:
            self._populate_setup_tracker_scenarios(None)
            self._render_setup_tracker_stats_text(payload, None)
        self.setup_tracker_view_loaded = True
        self.setup_tracker_view_stale = False
        self.status_var.set("Setup tracker dashboard refreshed.")
        self._notify_output_changed()

    def _highlight_trendline_candidates_in_avwap_output(self):
        self.avwap_text.tag_remove("trendline_bold", "1.0", tk.END)
        payload = self._load_ai_state_payload()
        raw_symbols = payload.get("symbols", {}) if isinstance(payload, dict) else {}
        highlighted = {
            str(symbol).strip().upper()
            for symbol, entry in (raw_symbols.items() if isinstance(raw_symbols, dict) else [])
            if isinstance(entry, dict) and (
                entry.get("priority_trendline_within_alert_range")
                or entry.get("priority_trendline_break_recent")
            )
        }
        if not highlighted:
            return

        section_end = self.avwap_text.search("MASTER AVWAP EVENT TICKERS", "1.0", stopindex=tk.END)
        if section_end:
            total_lines = max(0, int(section_end.split(".")[0]) - 1)
        else:
            total_lines = int(self.avwap_text.index("end-1c").split(".")[0])

        for line_no in range(1, total_lines + 1):
            line_start = f"{line_no}.0"
            line_end = f"{line_no}.end"
            line_text = self.avwap_text.get(line_start, line_end).strip()
            if not line_text or "score=" not in line_text:
                continue
            parts = line_text.split()
            if len(parts) < 2 or parts[1].upper() not in {"LONG", "SHORT"}:
                continue
            symbol = parts[0].strip(",").upper()
            if symbol in highlighted:
                self.avwap_text.tag_add("trendline_bold", line_start, line_end)

    def _load_tradingview_groups(self) -> dict | None:
        text = self._read_text_file(TRADINGVIEW_REPORT_FILE)
        if not text or text.startswith("[Missing file]") or text.startswith("[Error reading"):
            return None

        section_lookup = {
            "Best current favorite setups": "favorites",
            "Near favorite zones": "near_favorite_zones",
        }
        groups = {
            "favorites": {"LONG": [], "SHORT": []},
            "near_favorite_zones": {"LONG": [], "SHORT": []},
        }

        current_section = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                current_section = None
                continue
            if line in section_lookup:
                current_section = section_lookup[line]
                continue
            if line.startswith("-"):
                continue
            if current_section not in groups or ":" not in line:
                continue

            side_label, values = line.split(":", 1)
            side = side_label.strip().upper()
            if side not in ("LONG", "SHORT"):
                continue
            groups[current_section][side] = _extract_symbols_from_text(values)

        return groups

    def _load_focus_payload(self) -> dict:
        payload = load_json(MASTER_AVWAP_FOCUS_FILE, default={})
        return payload if isinstance(payload, dict) else {}

    def refresh_focus_group_boxes(self):
        self.focus_side_map = {}
        payload = self._load_focus_payload()
        favorites = payload.get("favorites", [])
        near_favorites = payload.get("near_favorite_zones", [])
        symbol_map = payload.get("symbols", {})
        tradingview_groups = self._load_tradingview_groups()

        if isinstance(symbol_map, dict):
            for symbol, row in symbol_map.items():
                ticker = str(symbol).strip().upper()
                if not ticker or not isinstance(row, dict):
                    continue
                self.focus_side_map[ticker] = normalize_side(row.get("side", "LONG"))

        favorite_symbols = []
        near_favorite_symbols = []
        if tradingview_groups:
            favorite_longs = tradingview_groups["favorites"]["LONG"]
            favorite_shorts = tradingview_groups["favorites"]["SHORT"]
            near_longs = tradingview_groups["near_favorite_zones"]["LONG"]
            near_shorts = tradingview_groups["near_favorite_zones"]["SHORT"]

            for symbol in favorite_longs + near_longs:
                self.focus_side_map[symbol] = "LONG"
            for symbol in favorite_shorts + near_shorts:
                self.focus_side_map[symbol] = "SHORT"

            favorite_symbols = favorite_longs + favorite_shorts
            near_favorite_symbols = near_longs + near_shorts

        if not favorite_symbols and not near_favorite_symbols:
            favorite_symbols = [
                str(entry.get("symbol", "")).strip().upper()
                for entry in favorites
                if isinstance(entry, dict) and str(entry.get("symbol", "")).strip()
            ]
            near_favorite_symbols = [
                str(entry.get("symbol", "")).strip().upper()
                for entry in near_favorites
                if isinstance(entry, dict) and str(entry.get("symbol", "")).strip()
            ]

        favorite_text = _format_symbol_group(favorite_symbols) if favorite_symbols else "None"
        near_favorite_text = _format_symbol_group(near_favorite_symbols) if near_favorite_symbols else "None"

        side_groups = build_master_avwap_focus_side_groups(payload)
        if not side_groups["LONG"] and not side_groups["SHORT"]:
            if tradingview_groups:
                side_groups = {
                    "LONG": _ordered_unique_symbols(
                        tradingview_groups["favorites"]["LONG"] + tradingview_groups["near_favorite_zones"]["LONG"]
                    ),
                    "SHORT": _ordered_unique_symbols(
                        tradingview_groups["favorites"]["SHORT"] + tradingview_groups["near_favorite_zones"]["SHORT"]
                    ),
                }

        setup_type_text = build_master_avwap_focus_setup_type_text(payload)
        self._set_text_widget_contents(self.favorite_symbols_text, favorite_text)
        self._set_text_widget_contents(self.near_favorite_symbols_text, near_favorite_text)
        self._set_text_widget_contents(self.long_focus_symbols_text, _format_symbol_group(side_groups.get("LONG", [])))
        self._set_text_widget_contents(self.short_focus_symbols_text, _format_symbol_group(side_groups.get("SHORT", [])))
        self._set_text_widget_contents(self.setup_type_symbols_text, setup_type_text or "None")

    def _copy_text_widget_contents(self, widget, empty_message: str, success_message: str):
        text = widget.get("1.0", tk.END).strip()
        if not text or text == "None":
            self.status_var.set(empty_message)
            return
        self._copy_to_clipboard(text)
        self.status_var.set(success_message)

    def copy_favorite_symbols(self):
        self._copy_text_widget_contents(
            self.favorite_symbols_text,
            "No favorite setup symbols to copy.",
            "Copied favorite setup symbols to clipboard.",
        )

    def copy_near_favorite_symbols(self):
        self._copy_text_widget_contents(
            self.near_favorite_symbols_text,
            "No near favorite zone symbols to copy.",
            "Copied near favorite zone symbols to clipboard.",
        )

    def copy_long_focus_symbols(self):
        self._copy_text_widget_contents(
            self.long_focus_symbols_text,
            "No long focus symbols to copy.",
            "Copied long focus symbols to clipboard.",
        )

    def copy_short_focus_symbols(self):
        self._copy_text_widget_contents(
            self.short_focus_symbols_text,
            "No short focus symbols to copy.",
            "Copied short focus symbols to clipboard.",
        )

    def copy_setup_type_symbols(self):
        self._copy_text_widget_contents(
            self.setup_type_symbols_text,
            "No score-ranked setup rows to copy.",
            "Copied score-ranked setup rows to clipboard.",
        )

    def copy_theta_symbols(self):
        self._copy_text_widget_contents(
            self.theta_symbols_text,
            "No theta symbols to copy.",
            "Copied theta symbols to clipboard.",
        )

    def _current_user_favorite_output_context(self) -> dict:
        payload = self._load_focus_payload()
        return {
            "run_date": payload.get("run_date", "") if isinstance(payload, dict) else "",
            "scan_generated_at": payload.get("generated_at", "") if isinstance(payload, dict) else "",
            "favorites": self.favorite_symbols_text.get("1.0", tk.END).strip(),
            "near_favorites": self.near_favorite_symbols_text.get("1.0", tk.END).strip(),
            "long_focus": self.long_focus_symbols_text.get("1.0", tk.END).strip(),
            "short_focus": self.short_focus_symbols_text.get("1.0", tk.END).strip(),
            "setup_types": self.setup_type_symbols_text.get("1.0", tk.END).strip(),
            "source": "master_avwap_gui",
        }

    def log_user_favorite_symbols(self):
        raw_text = self.user_favorite_symbols_text.get("1.0", tk.END).strip()
        result = append_master_avwap_user_favorites(
            raw_text,
            source="master_avwap_gui",
            notes=self.user_favorite_notes_var.get(),
            output_context=self._current_user_favorite_output_context(),
        )
        if not result.get("saved"):
            self.status_var.set(str(result.get("message") or "No tickers found in the pasted input."))
            return

        symbols = result.get("user_symbols", [])
        missing = result.get("missing_from_bot_symbols", [])
        missing_note = f" {len(missing)} were not in the current bot output." if missing else ""
        self.status_var.set(
            f"Logged {len(symbols)} favorite ticker(s) to {result.get('path')}.{missing_note}"
        )
        self._notify_output_changed()

    def clear_user_favorite_symbols(self):
        self.user_favorite_symbols_text.delete("1.0", tk.END)
        self.user_favorite_notes_var.set("")
        self.status_var.set("Cleared user favorite ticker input.")

    def open_user_favorites_log(self):
        self._reveal_path_in_explorer(USER_FAVORITES_FILE)

    def _load_market_prep_payload(self) -> dict:
        payload = load_json(MARKET_PREP_FILE, default={})
        return payload if isinstance(payload, dict) else {}

    def _market_prep_section_symbol_count(self, section: dict, text: str) -> int:
        symbols = section.get("symbols") if isinstance(section, dict) else []
        if isinstance(symbols, list):
            return len([symbol for symbol in symbols if str(symbol).strip()])
        clean_text = str(text or "").strip()
        if not clean_text or clean_text == "None":
            return 0
        return len(_ordered_unique_symbols(WATCHLIST_SYMBOL_RE.findall(clean_text.upper())))

    def _style_market_prep_section_widget(self, section_id: str, widget, count: int) -> None:
        if count <= 0:
            widget.configure(bg="#202428", fg="#7F8A96")
            return
        if "short" in section_id:
            widget.configure(bg=GUI_DARK_INPUT, fg="#FF9A9A")
        elif "earnings" in section_id:
            widget.configure(bg=GUI_DARK_INPUT, fg="#F0C76E")
        else:
            widget.configure(bg=GUI_DARK_INPUT, fg="#8FD19E")

    def refresh_market_prep_view(self):
        payload = self._load_market_prep_payload()
        sections = payload.get("sections", []) if isinstance(payload, dict) else []
        section_map = {
            str(section.get("id") or ""): section
            for section in sections
            if isinstance(section, dict)
        }

        for definition in MARKET_PREP_SECTION_DEFINITIONS:
            widget = self.market_prep_section_widgets.get(definition["id"])
            if widget is None:
                continue
            section = section_map.get(definition["id"], {})
            text = str(section.get("copy_text") or definition.get("empty_message") or "None").strip() or "None"
            count = self._market_prep_section_symbol_count(section, text)
            frame = getattr(self, "market_prep_section_frames", {}).get(definition["id"])
            if frame is not None:
                frame.configure(text=f"{definition['title']} ({count})")
            self._set_text_widget_contents(widget, text)
            self._style_market_prep_section_widget(definition["id"], widget, count)

        if MARKET_PREP_REPORT_FILE.exists():
            report_text = self._read_text_file(MARKET_PREP_REPORT_FILE)
        else:
            report_text = ""
        if not report_text or report_text.startswith("[Missing file]") or report_text.startswith("[Error reading"):
            report_text = format_market_prep_payload_report(payload) if payload else "No market prep output yet."
        self._set_text_widget_contents(self.market_prep_report_text, report_text)
        self._notify_output_changed()

    def copy_market_prep_section(self, section_id: str):
        widget = self.market_prep_section_widgets.get(section_id)
        definition = MARKET_PREP_SECTION_BY_ID.get(section_id, {})
        if widget is None:
            self.status_var.set("Market prep section is not available.")
            return
        self._copy_text_widget_contents(
            widget,
            f"No {definition.get('title', 'market prep')} symbols to copy.",
            f"Copied {definition.get('title', 'market prep')} symbols to clipboard.",
        )

    def copy_all_market_prep_sections(self):
        chunks = []
        for definition in MARKET_PREP_SECTION_DEFINITIONS:
            widget = self.market_prep_section_widgets.get(definition["id"])
            if widget is None:
                continue
            text = widget.get("1.0", tk.END).strip()
            if not text or text == "None":
                continue
            chunks.append(f"{definition['title']}: {text}")
        if not chunks:
            self.status_var.set("No market prep symbols to copy.")
            return
        self._copy_to_clipboard("\n".join(chunks))
        self.status_var.set("Copied all market prep lists to clipboard.")
    def refresh_avwap_output_view(self):
        priority_section = self._read_text_file(PRIORITY_SETUPS_FILE)
        theta_section = self._read_text_file(THETA_PUTS_FILE)
        event_section = self._read_text_file(EVENT_TICKERS_FILE)
        stdev_section = self._read_text_file(STDEV_RANGE_FILE)
        combined = build_combined_avwap_output_text(
            priority_section,
            theta_section,
            event_section,
            stdev_section,
        )

        self._set_text_widget_contents(self.avwap_text, combined)
        self._highlight_trendline_candidates_in_avwap_output()
        self.refresh_focus_group_boxes()
        self._notify_output_changed()

    def refresh_theta_output_view(self):
        theta_section = self._read_text_file(THETA_PUTS_FILE)
        text = theta_section or "No theta option output yet."
        self._set_text_widget_contents(self.theta_text, text)
        parsed_theta_rows = extract_theta_rows_from_report(text)
        self.theta_table_rows = [
            row for row in parsed_theta_rows if str(row.get("play_type") or "sold_put") != "pcs"
        ]
        self.theta_pcs_table_rows = [
            row for row in parsed_theta_rows if str(row.get("play_type") or "") == "pcs"
        ]
        self._refresh_theta_table()

        symbols = extract_theta_symbols_from_report(text)
        symbol_text = _format_symbol_group(symbols) if symbols else "None"
        self._set_text_widget_contents(self.theta_symbols_text, symbol_text)
        reason_widget = getattr(self, "theta_reason_risk_text", None)
        if reason_widget is None:
            self._notify_output_changed()
            return
        reason_rows = extract_theta_reason_risk_rows(text)
        reason_widget.configure(state="normal")
        reason_widget.delete("1.0", tk.END)
        if reason_rows:
            for row in reason_rows:
                risk_text = str(row.get("risk") or "none")
                risk_lower = risk_text.lower()
                tag = "risk_green"
                if "none" in risk_lower:
                    tag = "risk_green"
                elif "soon" in risk_lower or "high" in risk_lower:
                    tag = "risk_red"
                else:
                    tag = "risk_yellow"
                reason_widget.insert(tk.END, f"{row.get('symbol')}\n")
                reason_widget.insert(tk.END, f"  Reason: {row.get('reason') or 'n/a'}\n", "risk_green")
                reason_widget.insert(tk.END, "  Risk: ")
                reason_widget.insert(tk.END, f"{risk_text}\n\n", tag)
        else:
            reason_widget.insert("1.0", "No theta reason/risk details yet.")
        reason_widget.configure(state="normal")
        self._notify_output_changed()

    def _clear_theta_filters(self):
        self.theta_min_score_var.set("")
        self.theta_min_support_count_var.set("")
        self.theta_max_earnings_days_var.set("")
        self.theta_symbol_filter_var.set("")
        self._refresh_theta_table()

    def _sort_theta_table(self, column: str):
        if self.theta_sort_column == column:
            self.theta_sort_descending = not self.theta_sort_descending
        else:
            self.theta_sort_column = column
            self.theta_sort_descending = True
        self._refresh_theta_table()

    def _safe_int_filter(self, value: str) -> int | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _refresh_theta_table(self):
        table = getattr(self, "theta_table", None)
        pcs_table = getattr(self, "theta_pcs_table", None)
        if table is None:
            return
        for row_id in table.get_children():
            table.delete(row_id)
        if pcs_table is not None:
            for row_id in pcs_table.get_children():
                pcs_table.delete(row_id)

        min_score = self._safe_int_filter(self.theta_min_score_var.get())
        min_supports = self._safe_int_filter(self.theta_min_support_count_var.get())
        max_earnings_days = self._safe_int_filter(self.theta_max_earnings_days_var.get())
        symbol_filter = str(self.theta_symbol_filter_var.get() or "").strip().upper()

        def filter_rows(source_rows):
            filtered_rows = []
            for row in source_rows or []:
                if min_score is not None and int(row.get("score", 0) or 0) < min_score:
                    continue
                if min_supports is not None and int(row.get("support_count", 0) or 0) < min_supports:
                    continue
                next_days = row.get("next_earnings_days")
                if max_earnings_days is not None and isinstance(next_days, int) and next_days > max_earnings_days:
                    continue
                if symbol_filter and symbol_filter not in str(row.get("symbol", "")).upper():
                    continue
                filtered_rows.append(row)
            return filtered_rows

        def sort_key(entry: dict):
            if self.theta_sort_column in {"score", "support_count", "next_earnings_days"}:
                value = entry.get(self.theta_sort_column)
                return -10**9 if value is None else int(value)
            if self.theta_sort_column in {"close", "atr", "recommended_strike", "recommended_credit"}:
                value = entry.get(self.theta_sort_column)
                return float(value or 0.0)
            return str(entry.get(self.theta_sort_column, "") or "").upper()

        def format_option_table_decimal(value) -> str:
            if value is None or value == "":
                return ""
            try:
                return _format_option_decimal(float(value))
            except (TypeError, ValueError):
                return str(value)

        def populate_table(target_table, source_rows):
            filtered_rows = filter_rows(source_rows)
            filtered_rows.sort(key=sort_key, reverse=self.theta_sort_descending)
            for row in filtered_rows:
                values = (
                    row.get("symbol", ""),
                    row.get("score", ""),
                    row.get("support_count", ""),
                    f"{float(row.get('close', 0.0) or 0.0):.2f}" if row.get("close") is not None else "",
                    row.get("atr", ""),
                    row.get("next_earnings_days", "unknown"),
                    format_option_table_decimal(row.get("recommended_strike")),
                    format_option_table_decimal(row.get("recommended_credit")),
                    row.get("primary_strike_band", ""),
                    row.get("liquidity_score", ""),
                )
                target_table.insert(
                    "",
                    "end",
                    values=values,
                    tags=tree_tags_for_values(values),
                )

        populate_table(table, self.theta_table_rows)
        if pcs_table is not None:
            populate_table(pcs_table, self.theta_pcs_table_rows)

    def _copy_to_clipboard(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()

    def _shared_scheduler_slot_datetime(self, slot: str, reference: datetime | None = None) -> datetime:
        now = reference or datetime.now()
        slot_time = datetime.strptime(str(slot).strip(), "%H:%M").time()
        return datetime.combine(now.date(), slot_time)

    def _reset_shared_watchlist_scheduler_state_for_day(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        today_iso = now.date().isoformat()
        if self.shared_scheduler_day == today_iso and self.shared_scheduler_slots_state:
            return False

        schedule = list(get_default_hourly_scan_schedule(reference=now))
        self.shared_scheduler_day = today_iso
        self.shared_scheduler_slots_state = {slot: "pending" for slot in schedule}
        if self.shared_scheduler_enabled:
            self.shared_scheduler_note = "Scheduler ready for today's market window."
        else:
            self.shared_scheduler_note = "Hourly shared-watchlist scheduler is off."
        return True

    def _get_shared_watchlist_scheduler_schedule(self, now: datetime | None = None) -> list[str]:
        self._reset_shared_watchlist_scheduler_state_for_day(now)
        return list(self.shared_scheduler_slots_state.keys())

    def _get_due_pending_shared_watchlist_scheduler_slots(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now()
        schedule = self._get_shared_watchlist_scheduler_schedule(now)
        due_slots = []
        for slot in schedule:
            if self.shared_scheduler_slots_state.get(slot) != "pending":
                continue
            if self._shared_scheduler_slot_datetime(slot, reference=now) <= now:
                due_slots.append(slot)
        return due_slots

    def _get_next_pending_shared_watchlist_scheduler_slot(self, now: datetime | None = None) -> str | None:
        now = now or datetime.now()
        schedule = self._get_shared_watchlist_scheduler_schedule(now)
        for slot in schedule:
            if self.shared_scheduler_slots_state.get(slot) != "pending":
                continue
            if self._shared_scheduler_slot_datetime(slot, reference=now) > now:
                return slot
        return None

    def _refresh_shared_watchlist_scheduler_status(self, note: str | None = None):
        now = datetime.now()
        self._reset_shared_watchlist_scheduler_state_for_day(now)
        if note is not None:
            self.shared_scheduler_note = note

        schedule = self._get_shared_watchlist_scheduler_schedule(now)
        next_slot = self._get_next_pending_shared_watchlist_scheduler_slot(now)
        stop_at = get_default_stop_time_label(reference=now)
        market_session = get_market_session_window(reference=now)
        completed_slots = [
            slot for slot, status in self.shared_scheduler_slots_state.items()
            if status == "completed"
        ]
        failed_slots = [
            slot for slot, status in self.shared_scheduler_slots_state.items()
            if status == "failed"
        ]
        active_task = self.shared_scheduler_active_slot or self.current_background_label or "None"
        scheduler_state = "running" if self.shared_scheduler_enabled else "stopped"
        self.shared_scheduler_button_var.set(
            "Stop Scheduler" if self.shared_scheduler_enabled else "Start Scheduler"
        )
        self.shared_scheduler_status_var.set(
            "\n".join(
                [
                    (
                        f"Hourly shared-watchlist scheduler: {scheduler_state} | "
                        f"Market session: {market_session.session_label} | "
                        f"Today's slots: {', '.join(schedule) if schedule else 'None'} | Stop at: {stop_at}"
                    ),
                    (
                        f"Next slot: {next_slot or 'None'} | Completed: {len(completed_slots)} | "
                        f"Failed: {len(failed_slots)} | Active task: {active_task}"
                    ),
                    f"Note: {self.shared_scheduler_note}",
                ]
            )
        )

    def toggle_shared_watchlist_scheduler(self):
        self.shared_scheduler_enabled = not self.shared_scheduler_enabled
        note = (
            "Hourly shared-watchlist scheduler started."
            if self.shared_scheduler_enabled
            else "Hourly shared-watchlist scheduler stopped."
        )
        self._refresh_shared_watchlist_scheduler_status(note=note)
        self.status_var.set(note)

    def _finish_shared_watchlist_scheduler_run(
        self,
        covered_slots: list[str],
        trigger_slot: str,
        success: bool,
        error_text: str = "",
    ):
        for slot in covered_slots:
            if slot not in self.shared_scheduler_slots_state:
                continue
            self.shared_scheduler_slots_state[slot] = "completed" if success else "failed"
        self.shared_scheduler_active_slot = ""
        note = (
            f"Scheduled shared-watchlist scan for {trigger_slot} completed."
            if success
            else f"Scheduled shared-watchlist scan for {trigger_slot} failed: {error_text}"
        )
        self._refresh_shared_watchlist_scheduler_status(note=note)

    def _start_scheduled_shared_watchlist_scan(self, trigger_slot: str, covered_slots: list[str]) -> bool:
        if self.background_task_active:
            self.status_var.set("Another background task is already running. Please wait for it to finish.")
            return False

        self.notebook.select(self.avwap_tab)
        self.shared_scheduler_active_slot = trigger_slot
        running_msg = f"Running scheduled shared-watchlist scan for {trigger_slot}..."
        started = self._run_background(
            run_master_with_shared_watchlists,
            running_msg,
            f"Scheduled shared-watchlist scan for {trigger_slot} complete.",
            done_callback=lambda: (
                self.refresh_avwap_output_view(),
                self.refresh_theta_output_view(),
                self.refresh_market_prep_view(),
                self._mark_setup_tracker_view_stale(),
            ),
            finish_callback=lambda success, error_text, slots=list(covered_slots), slot=trigger_slot: (
                self._finish_shared_watchlist_scheduler_run(slots, slot, success, error_text)
            ),
        )
        if not started:
            self.shared_scheduler_active_slot = ""
            return False
        self._refresh_shared_watchlist_scheduler_status(note=running_msg)
        return True

    def _shared_watchlist_scheduler_tick(self):
        try:
            now = datetime.now()
            self._reset_shared_watchlist_scheduler_state_for_day(now)

            if self.background_task_active:
                self._refresh_shared_watchlist_scheduler_status()
                return

            if not self.shared_scheduler_enabled:
                self._refresh_shared_watchlist_scheduler_status()
                return

            stop_dt = self._shared_scheduler_slot_datetime(
                get_default_stop_time_label(reference=now),
                reference=now,
            )
            if now >= stop_dt:
                self._refresh_shared_watchlist_scheduler_status(
                    note=f"Today's scheduler window ended at {stop_dt.strftime('%H:%M')}.",
                )
                return

            due_slots = self._get_due_pending_shared_watchlist_scheduler_slots(now)
            if due_slots:
                self._start_scheduled_shared_watchlist_scan(due_slots[-1], due_slots)
                return

            next_slot = self._get_next_pending_shared_watchlist_scheduler_slot(now)
            self._refresh_shared_watchlist_scheduler_status(
                note=(
                    f"Waiting for next hourly slot {next_slot}."
                    if next_slot else "No pending hourly shared-watchlist slots remain for today."
                ),
            )
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(
                        SHARED_WATCHLIST_SCHEDULER_TICK_MS,
                        self._shared_watchlist_scheduler_tick,
                    )
            except Exception:
                pass

    def _run_background(self, target, running_msg, done_msg, done_callback=None, finish_callback=None):
        if self.background_task_active:
            self.status_var.set("Another background task is already running. Please wait for it to finish.")
            return False

        self.background_task_active = True
        self.current_background_label = running_msg
        self.status_var.set(running_msg)
        self._refresh_shared_watchlist_scheduler_status()

        def _task():
            error_text = ""
            try:
                target()
            except Exception as exc:
                logging.exception("GUI background task failed")
                error_text = str(exc)

            def _finish():
                self.background_task_active = False
                self.current_background_label = ""
                if error_text:
                    self.status_var.set(f"Error: {error_text}")
                else:
                    self.status_var.set(done_msg)
                    if done_callback:
                        done_callback()
                if finish_callback:
                    finish_callback(not bool(error_text), error_text)
                else:
                    self._refresh_shared_watchlist_scheduler_status()

            self.root.after(0, _finish)

        threading.Thread(target=_task, daemon=True).start()
        return True

    def run_master_once(self):
        self.notebook.select(self.avwap_tab)
        self._run_background(
            run_master_with_shared_watchlists,
            "Running Master AVWAP scan from shared home-folder longs.txt / shorts.txt...",
            "Shared-watchlist Master AVWAP scan complete.",
            done_callback=lambda: (
                self.refresh_avwap_output_view(),
                self.refresh_theta_output_view(),
                self.refresh_market_prep_view(),
                self._mark_setup_tracker_view_stale(),
            ),
        )

    def run_local_watchlist_scan_once(self):
        self.notebook.select(self.avwap_tab)
        self._run_background(
            run_master,
            "Running Master AVWAP scan from local project watchlists...",
            "Local-watchlist Master AVWAP scan complete.",
            done_callback=lambda: (
                self.refresh_avwap_output_view(),
                self.refresh_theta_output_view(),
                self.refresh_market_prep_view(),
                self._mark_setup_tracker_view_stale(),
            ),
        )

    def backfill_setup_tracker_history(self):
        try:
            lookback_sessions = max(1, int(self.tracker_backfill_sessions_var.get()))
        except Exception:
            lookback_sessions = 5
            self.tracker_backfill_sessions_var.set(lookback_sessions)

        self.notebook.select(self.tracker_tab)
        self._run_background(
            lambda: backfill_setup_tracker_from_recent_sessions(lookback_sessions=lookback_sessions),
            f"Backfilling tracker from last {lookback_sessions} session(s)...",
            f"Setup tracker backfill complete for last {lookback_sessions} session(s).",
            done_callback=self.refresh_setup_tracker_view,
        )

    def run_shared_watchlist_scan_once(self):
        self.run_master_once()

def launch_gui():
    if tk is None or ttk is None:
        logging.error("tkinter is unavailable in this Python environment; cannot launch GUI.")
        return

    root = tk.Tk()
    MasterAvwapGUI(root, standalone=True)
    root.mainloop()


# ============================================================================
# ENTRYPOINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run Master AVWAP scanner")
    tracker_window_start, tracker_window_end = get_setup_tracker_update_window_labels()
    parser.add_argument("--once", action="store_true", help="Run a single AVWAP scan and exit.")
    parser.add_argument("--loop", action="store_true", help="Run AVWAP scan in hourly loop.")
    parser.add_argument(
        "--force-setup-tracker-update",
        action="store_true",
        help=(
            "Update the setup tracker even outside the default "
            f"{tracker_window_start}-{tracker_window_end} local live-update window."
        ),
    )
    parser.add_argument("--gui", action="store_true", help="Launch the Master AVWAP management GUI.")
    args = parser.parse_args()

    if args.once:
        run_master(update_setup_tracker=True if args.force_setup_tracker_update else None)
        return

    if args.loop:
        logging.info("Starting hourly Master AVWAP loop (once per hour)…")
        while True:
            start = time.time()
            run_master(update_setup_tracker=True if args.force_setup_tracker_update else None)
            elapsed = time.time() - start
            sleep_seconds = max(0, 3600 - elapsed)
            logging.info(f"Sleeping {int(sleep_seconds)} seconds until next run…")
            time.sleep(sleep_seconds)

    if args.gui or (not args.once and not args.loop):
        launch_gui()


if __name__ == "__main__":
    main()
