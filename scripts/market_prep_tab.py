from __future__ import annotations

import sys
import threading
import tkinter as tk
import os
import re
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from project_paths import (
    LOCAL_SETTINGS_FILE,
    OUTPUT_DIR,
    ROOT_DIR,
    get_local_setting,
    open_path_in_file_manager,
    save_local_setting,
)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from market_prep import get_market_prep_logger, load_market_prep_config
from market_prep.orchestrator import MarketPrepOrchestrator
from market_prep.report_builder import build_catalyst_clock
from market_prep.services.ticker_lookup_service import lookup_ticker_context


MARKET_PREP_PLACEHOLDER_TEXT = "Market Prep tab loaded. Phase 1 skeleton ready."
MARKET_PREP_NAV_ITEMS = (
    ("Command Center", "overview"),
    ("AI Summary", "ai_summary"),
    ("API Key", "api_key"),
    ("Catalyst Clock", "catalyst"),
    ("Watchlist Risk", "watchlist"),
    ("Earnings", "earnings"),
    ("Macro/Fed/Treasury", "macro"),
    ("News/SEC", "news_sec"),
    ("Raw Markdown", "raw"),
)
HIGH_PRIORITY_VALUES = {"HIGH", "MEGA"}
MEDIUM_PRIORITY_VALUES = {"MEDIUM"}
TEXT_FONT = ("Courier New", 10)
OPENAI_LOCAL_SETTING_KEY = "market_prep_openai_api_key"

ACCENT_BLUE = "#8AB4F8"
ACCENT_CYAN = "#7DD3FC"
ACCENT_GREEN = "#8FD19E"
ACCENT_YELLOW = "#F0C76E"
ACCENT_RED = "#FF9A9A"
ACCENT_PURPLE = "#C7B8FF"
MUTED_TEXT = "#8A939E"
TREE_SELECTED_BG = "#53606C"
TREE_HEADER_BG = "#36414B"
TREE_HEADER_ACTIVE_BG = "#45525F"
TREE_RISK_HIGH_BG = "#3A2527"
TREE_RISK_MEDIUM_BG = "#3A3322"
TREE_RISK_LOW_BG = "#252B31"
TREE_RISK_CLEAN_BG = "#223027"
TEXT_TAG_NAMES = (
    "mp_heading",
    "mp_separator",
    "mp_bullet",
    "mp_ticker",
    "mp_link",
    "mp_high",
    "mp_medium",
    "mp_clean",
    "mp_muted",
    "mp_positive",
    "mp_negative",
    "mp_alert_line",
    "mp_warning_line",
    "mp_clean_line",
)
HEADING_HINTS = {
    "ai",
    "brief",
    "catalyst",
    "checklist",
    "company",
    "details",
    "earnings",
    "fed",
    "headline",
    "landmine",
    "lookup",
    "macro",
    "market",
    "overview",
    "posture",
    "prompt",
    "risk",
    "sec",
    "snapshot",
    "source",
    "status",
    "treasury",
    "warnings",
    "watchlist",
}
TICKER_STOPWORDS = {
    "AI",
    "AMC",
    "API",
    "BMO",
    "ET",
    "ETF",
    "FED",
    "FOMC",
    "HIGH",
    "LOW",
    "MEGA",
    "MEDIUM",
    "RSS",
    "SEC",
    "TBD",
    "USD",
    "URL",
}
TICKER_RE = re.compile(r"\b[A-Z][A-Z0-9.\-]{1,5}\b")
URL_RE = re.compile(r"https?://\S+")
SIGNED_PCT_RE = re.compile(r"(?<!\w)([+-]\d+(?:\.\d+)?%)")
HIGH_RISK_RE = re.compile(r"\b(MEGA|HIGH|NO-TRADE|TODAY/TOMORROW|ERRORS?|FAIL(?:ED)?)\b", re.IGNORECASE)
MEDIUM_RISK_RE = re.compile(r"\b(MEDIUM|WATCH|WARNING|WARNINGS|REDUCED-SIZE|EARNINGS WITHIN)\b", re.IGNORECASE)
CLEAN_RE = re.compile(r"\b(CLEAN|READY|OK|AVAILABLE|GENERATED)\b", re.IGNORECASE)


def _configure_readable_tree_style(parent: tk.Misc, text_bg: str, text_fg: str) -> None:
    style = ttk.Style(parent)
    style.configure(
        "Readable.Treeview",
        background=text_bg,
        fieldbackground=text_bg,
        foreground=text_fg,
        bordercolor="#15191E",
        rowheight=25,
    )
    style.map(
        "Readable.Treeview",
        background=[("selected", TREE_SELECTED_BG)],
        foreground=[("selected", "#FFFFFF")],
    )
    style.configure(
        "Readable.Treeview.Heading",
        background=TREE_HEADER_BG,
        foreground="#F2F5F8",
        relief="flat",
    )
    style.map("Readable.Treeview.Heading", background=[("active", TREE_HEADER_ACTIVE_BG)])


def _configure_text_tags(widget: tk.Text) -> None:
    bold_font = (TEXT_FONT[0], TEXT_FONT[1], "bold")
    widget.tag_configure("mp_heading", foreground=ACCENT_CYAN, font=bold_font, spacing1=4, spacing3=3)
    widget.tag_configure("mp_separator", foreground=MUTED_TEXT)
    widget.tag_configure("mp_bullet", foreground=MUTED_TEXT)
    widget.tag_configure("mp_ticker", foreground=ACCENT_BLUE, font=bold_font)
    widget.tag_configure("mp_link", foreground=ACCENT_BLUE, underline=True)
    widget.tag_configure("mp_high", foreground=ACCENT_RED, font=bold_font)
    widget.tag_configure("mp_medium", foreground=ACCENT_YELLOW, font=bold_font)
    widget.tag_configure("mp_clean", foreground=ACCENT_GREEN, font=bold_font)
    widget.tag_configure("mp_muted", foreground=MUTED_TEXT)
    widget.tag_configure("mp_positive", foreground=ACCENT_GREEN)
    widget.tag_configure("mp_negative", foreground=ACCENT_RED)
    widget.tag_configure("mp_alert_line", background=TREE_RISK_HIGH_BG)
    widget.tag_configure("mp_warning_line", background=TREE_RISK_MEDIUM_BG)
    widget.tag_configure("mp_clean_line", background=TREE_RISK_CLEAN_BG)
    for tag in ("mp_heading", "mp_ticker", "mp_link", "mp_high", "mp_medium", "mp_clean"):
        widget.tag_raise(tag)


def _clear_text_tags(widget: tk.Text) -> None:
    for tag in TEXT_TAG_NAMES:
        widget.tag_remove(tag, "1.0", tk.END)


def _add_regex_tag(widget: tk.Text, line_no: int, line: str, pattern: re.Pattern, tag: str) -> None:
    for match in pattern.finditer(line):
        widget.tag_add(tag, f"{line_no}.{match.start()}", f"{line_no}.{match.end()}")


def _looks_like_heading(line: str, next_line: str = "") -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if re.fullmatch(r"[-=]{3,}", stripped):
        return False
    if next_line.strip() and re.fullmatch(r"[-=]{3,}", next_line.strip()):
        return True
    if stripped.startswith(("-", "*")) or stripped.endswith(".") or len(stripped) > 54:
        return False
    if ":" in stripped and not stripped.endswith(":"):
        return False
    words = {word.lower().strip("/:") for word in re.findall(r"[A-Za-z/]+", stripped)}
    return bool(words & HEADING_HINTS)


def _apply_text_highlights(widget: tk.Text, text: str) -> None:
    _clear_text_tags(widget)
    lines = str(text or "").splitlines()
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        line_start = f"{line_no}.0"
        line_end = f"{line_no}.end"
        next_line = lines[line_no] if line_no < len(lines) else ""
        if re.fullmatch(r"[-=]{3,}", stripped):
            widget.tag_add("mp_separator", line_start, line_end)
            continue
        if _looks_like_heading(line, next_line):
            widget.tag_add("mp_heading", line_start, line_end)
        if HIGH_RISK_RE.search(line):
            widget.tag_add("mp_alert_line", line_start, line_end)
        elif MEDIUM_RISK_RE.search(line):
            widget.tag_add("mp_warning_line", line_start, line_end)
        elif CLEAN_RE.search(line):
            widget.tag_add("mp_clean_line", line_start, line_end)
        bullet_pos = line.find("- ")
        if 0 <= bullet_pos <= 4:
            widget.tag_add("mp_bullet", f"{line_no}.{bullet_pos}", f"{line_no}.{bullet_pos + 1}")
        _add_regex_tag(widget, line_no, line, URL_RE, "mp_link")
        _add_regex_tag(widget, line_no, line, HIGH_RISK_RE, "mp_high")
        _add_regex_tag(widget, line_no, line, MEDIUM_RISK_RE, "mp_medium")
        _add_regex_tag(widget, line_no, line, CLEAN_RE, "mp_clean")
        for match in TICKER_RE.finditer(line):
            token = match.group(0).strip(".-")
            if token in TICKER_STOPWORDS:
                continue
            widget.tag_add("mp_ticker", f"{line_no}.{match.start()}", f"{line_no}.{match.end()}")
        for match in SIGNED_PCT_RE.finditer(line):
            tag = "mp_positive" if match.group(1).startswith("+") else "mp_negative"
            widget.tag_add(tag, f"{line_no}.{match.start()}", f"{line_no}.{match.end()}")


class MarketPrepTab:
    """Market Prep workspace for scheduled-risk and watchlist catalysts."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        output_dir: Path = OUTPUT_DIR,
        text_bg: str = "#252525",
        text_fg: str = "#E0E0E0",
        compact_layout: bool = False,
    ):
        self.parent = parent
        self.output_dir = Path(output_dir)
        self.text_bg = text_bg
        self.text_fg = text_fg
        self.compact_layout = bool(compact_layout)
        self.config = load_market_prep_config()
        resolved_paths = self.config.resolved_paths()
        self.output_dir = resolved_paths.get("output_dir", self.output_dir)
        self.logger = get_market_prep_logger()
        self.orchestrator = MarketPrepOrchestrator()
        self.status_var = tk.StringVar(value="Ready")
        self.source_status_var = tk.StringVar(value="Sources: waiting for first run.")
        self.openai_key_var = tk.StringVar(value=self._saved_openai_key())
        self.openai_key_status_var = tk.StringVar()
        self.summary_vars: dict[str, tk.StringVar] = {}
        self.container = ttk.Frame(parent)
        self.display_text: scrolledtext.ScrolledText | None = None
        self.overview_text: scrolledtext.ScrolledText | None = None
        self.ai_summary_text: scrolledtext.ScrolledText | None = None
        self.earnings_text: scrolledtext.ScrolledText | None = None
        self.macro_text: scrolledtext.ScrolledText | None = None
        self.news_sec_text: scrolledtext.ScrolledText | None = None
        self.catalyst_tree: ttk.Treeview | None = None
        self.watchlist_tree: ttk.Treeview | None = None
        self.catalyst_status_var = tk.StringVar(value="No catalyst clock loaded.")
        self.watchlist_status_var = tk.StringVar(value="No watchlist risk loaded.")
        self.buttons: dict[str, ttk.Button] = {}
        self.background_task_active = False
        self.latest_report: dict | None = None
        self._nav_syncing = False
        _configure_readable_tree_style(parent, self.text_bg, self.text_fg)
        self._build_layout()
        self._render_plain_text(MARKET_PREP_PLACEHOLDER_TEXT)
        self.logger.info("Market Prep loaded")

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self.container)
        toolbar.pack(fill=tk.X, padx=6 if self.compact_layout else 10, pady=(4 if self.compact_layout else 10, 4))

        buttons = [
            ("Start Day", self.start_day),
            ("Start Week", self.start_week),
            ("Run AI Summary", self.run_ai_summary),
            ("Run Daily Prep", self.run_daily_prep),
            ("Run Weekly Prep", self.run_weekly_prep),
            ("Refresh Earnings", self.refresh_earnings),
            ("Refresh Economic Calendar", self.refresh_economic_calendar),
            ("Toggle ForexFactory", self.toggle_forexfactory),
            ("Refresh News", self.refresh_news),
            ("Refresh YouTube Links", self.refresh_youtube_links),
            ("Scan Watchlists", self.scan_watchlists),
            ("Export Markdown", self.export_markdown),
            ("Open Output Folder", self.open_output_folder),
        ]
        if self.compact_layout:
            buttons = [
                ("Start Day", self.start_day),
                ("Start Week", self.start_week),
                ("Run AI Summary", self.run_ai_summary),
                ("Scan Watchlists", self.scan_watchlists),
                ("Export", self.export_markdown),
                ("Folder", self.open_output_folder),
            ]
        for label, command in buttons:
            button = ttk.Button(toolbar, text=label, command=command)
            button.pack(side=tk.LEFT, padx=(0, 4 if self.compact_layout else 8), pady=(0, 2 if self.compact_layout else 4))
            self.buttons[label] = button
        self._refresh_forexfactory_button()

        self._build_command_center()

        body = ttk.Frame(self.container)
        body.pack(fill=tk.BOTH, expand=True, padx=6 if self.compact_layout else 10, pady=(0, 4 if self.compact_layout else 8))

        nav_frame = ttk.LabelFrame(body, text="Sections")
        if not self.compact_layout:
            nav_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.nav_list = tk.Listbox(
            nav_frame,
            width=24 if not self.compact_layout else 18,
            height=len(MARKET_PREP_NAV_ITEMS),
            exportselection=False,
            bg="#202428",
            fg=self.text_fg,
            selectbackground=TREE_SELECTED_BG,
            selectforeground="#FFFFFF",
            activestyle="none",
            highlightthickness=1,
            highlightbackground="#15191E",
            highlightcolor=ACCENT_BLUE,
        )
        for label, _tab_id in MARKET_PREP_NAV_ITEMS:
            self.nav_list.insert(tk.END, label)
        self.nav_list.pack(fill=tk.Y, expand=False, padx=8, pady=8)
        self.nav_list.selection_set(0)
        self.nav_list.bind("<<ListboxSelect>>", self._on_nav_selected)

        content = ttk.Frame(body)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.view_notebook = ttk.Notebook(content)
        self.view_notebook.pack(fill=tk.BOTH, expand=True)
        self.view_notebook.bind("<<NotebookTabChanged>>", self._on_view_tab_changed)

        self._build_overview_tab()
        self._build_ai_summary_tab()
        self._build_api_key_tab()
        self._build_catalyst_tab()
        self._build_watchlist_tab()
        self._build_text_tab("earnings", "Earnings")
        self._build_text_tab("macro", "Macro/Fed/Treasury")
        self._build_text_tab("news_sec", "News/SEC")
        self._build_raw_tab()

        status = ttk.Label(self.container, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill=tk.X, padx=6 if self.compact_layout else 10, pady=(0, 6 if self.compact_layout else 10))

    def _build_command_center(self) -> None:
        center = ttk.Frame(self.container)
        center.pack(fill=tk.X, padx=6 if self.compact_layout else 10, pady=(0, 4 if self.compact_layout else 8))

        cards = [
            ("report", "Report"),
            ("market_regime", "Market Regime"),
            ("risk_level", "Week/Day Risk"),
            ("next_catalyst", "Next Landmine"),
            ("watchlist_holds", "Watchlist Holds"),
            ("ai_brief", "AI Brief"),
        ]
        if self.compact_layout:
            cards = [
                ("next_catalyst", "Next Landmine"),
                ("watchlist_holds", "Watchlist Holds"),
                ("ai_brief", "AI Brief"),
            ]
            for key in {"report", "market_regime", "risk_level"}:
                self.summary_vars[key] = tk.StringVar(value="Waiting for run")
        for key, title in cards:
            frame = ttk.LabelFrame(center, text=title)
            frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
            var = tk.StringVar(value="Waiting for run")
            self.summary_vars[key] = var
            ttk.Label(frame, textvariable=var, justify="left", wraplength=220 if self.compact_layout else 270).pack(
                fill=tk.X,
                padx=6 if self.compact_layout else 8,
                pady=(3 if self.compact_layout else 6, 4 if self.compact_layout else 8),
            )

        source_frame = ttk.LabelFrame(self.container, text="Source Health")
        if self.compact_layout:
            return
        source_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(source_frame, textvariable=self.source_status_var, justify="left", wraplength=1600).pack(
            fill=tk.X,
            padx=8,
            pady=(5, 7),
        )

    def _build_overview_tab(self) -> None:
        frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(frame, text="Command Center")
        self.overview_text = self._make_text(frame)

    def _build_ai_summary_tab(self) -> None:
        frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(frame, text="AI Summary")
        self.ai_summary_text = self._make_text(frame)

    def _build_api_key_tab(self) -> None:
        frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(frame, text="API Key")

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(row, text="OpenAI API Key").pack(side=tk.LEFT, padx=(0, 8))
        self.openai_key_entry = ttk.Entry(row, textvariable=self.openai_key_var, show="*", width=54)
        self.openai_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(row, text="Save", command=self.save_openai_api_key).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Clear", command=self.clear_openai_api_key).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Reload", command=self.reload_openai_api_key).pack(side=tk.LEFT)

        status = ttk.LabelFrame(frame, text="Status")
        status.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        ttk.Label(status, textvariable=self.openai_key_status_var, justify="left", wraplength=900).pack(
            fill=tk.X,
            padx=8,
            pady=(6, 8),
        )
        self._refresh_openai_key_status()

    def _saved_openai_key(self) -> str:
        return str(get_local_setting(OPENAI_LOCAL_SETTING_KEY, "") or "").strip()

    def _config_openai_key(self) -> str:
        keys = self.config.api_keys if isinstance(self.config.api_keys, dict) else {}
        return str(keys.get("openai") or "").strip()

    def _refresh_openai_key_status(self) -> None:
        env_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
        local_key = self._saved_openai_key()
        config_key = self._config_openai_key()
        lines = []
        if env_key:
            lines.append(f"OPENAI_API_KEY is set: {self._mask_secret(env_key)}")
            lines.append("Environment key takes priority.")
        if local_key:
            lines.append(f"Saved local key: {self._mask_secret(local_key)}")
        if config_key:
            lines.append(f"Repo config fallback key: {self._mask_secret(config_key)}")
        if not env_key and not local_key and not config_key:
            lines.append("No OpenAI API key saved.")
        lines.append(f"Local settings: {LOCAL_SETTINGS_FILE}")
        self.openai_key_status_var.set("\n".join(lines))

    def _mask_secret(self, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            return ""
        if len(cleaned) <= 8:
            return "*" * len(cleaned)
        return f"{cleaned[:4]}...{cleaned[-4:]}"

    def save_openai_api_key(self) -> None:
        key = self.openai_key_var.get().strip()
        try:
            save_local_setting(OPENAI_LOCAL_SETTING_KEY, key)
            self.orchestrator = MarketPrepOrchestrator()
        except Exception as exc:
            self.logger.exception("Failed saving OpenAI API key.")
            messagebox.showerror("OpenAI API Key", f"Could not save OpenAI API key:\n\n{exc}")
            self.openai_key_status_var.set(f"Save failed: {exc}")
            return
        self.openai_key_var.set(self._saved_openai_key())
        self._refresh_openai_key_status()
        self.status_var.set("OpenAI API key saved.")

    def clear_openai_api_key(self) -> None:
        self.openai_key_var.set("")
        self.save_openai_api_key()
        self.status_var.set("OpenAI API key cleared.")

    def reload_openai_api_key(self) -> None:
        self.config = load_market_prep_config(self.config.config_path)
        self.output_dir = self.config.resolved_paths().get("output_dir", self.output_dir)
        self.openai_key_var.set(self._saved_openai_key())
        self._refresh_openai_key_status()
        self.status_var.set("OpenAI API key reloaded.")

    def _build_catalyst_tab(self) -> None:
        frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(frame, text="Catalyst Clock")
        ttk.Label(frame, textvariable=self.catalyst_status_var, anchor="w").pack(fill=tk.X, padx=8, pady=(8, 4))
        columns = ("date", "time", "bucket", "priority", "text")
        self.catalyst_tree = self._make_tree(
            frame,
            columns,
            {
                "date": ("Date", 100, "w"),
                "time": ("ET", 70, "w"),
                "bucket": ("Bucket", 100, "w"),
                "priority": ("Risk", 80, "w"),
                "text": ("Catalyst", 760, "w"),
            },
        )

    def _build_watchlist_tab(self) -> None:
        frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(frame, text="Watchlist Risk")
        ttk.Label(frame, textvariable=self.watchlist_status_var, anchor="w").pack(fill=tk.X, padx=8, pady=(8, 4))
        columns = ("ticker", "classification", "source_list", "company", "market_cap", "sector", "reason")
        self.watchlist_tree = self._make_tree(
            frame,
            columns,
            {
                "ticker": ("Ticker", 80, "w"),
                "classification": ("Risk", 190, "w"),
                "source_list": ("List", 90, "w"),
                "company": ("Company", 180, "w"),
                "market_cap": ("Market Cap", 100, "e"),
                "sector": ("Sector", 145, "w"),
                "reason": ("Reason", 520, "w"),
            },
        )

    def _build_text_tab(self, tab_id: str, title: str) -> None:
        frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(frame, text=title)
        widget = self._make_text(frame)
        if tab_id == "earnings":
            self.earnings_text = widget
        elif tab_id == "macro":
            self.macro_text = widget
        elif tab_id == "news_sec":
            self.news_sec_text = widget

    def _build_raw_tab(self) -> None:
        frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(frame, text="Raw Markdown")
        self.display_text = self._make_text(frame)

    def _make_text(self, parent: tk.Misc) -> scrolledtext.ScrolledText:
        widget = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=TEXT_FONT,
            bg=self.text_bg,
            fg=self.text_fg,
            insertbackground=self.text_fg,
            selectbackground="#4A4A4A",
            selectforeground=self.text_fg,
        )
        widget.pack(fill=tk.BOTH, expand=True, padx=6 if self.compact_layout else 8, pady=4 if self.compact_layout else 8)
        _configure_text_tags(widget)
        widget.configure(state="disabled")
        return widget

    def _make_tree(
        self,
        parent: tk.Misc,
        columns: tuple[str, ...],
        headings: dict[str, tuple[str, int, str]],
    ) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=6 if self.compact_layout else 8, pady=(0, 4 if self.compact_layout else 8))
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse", style="Readable.Treeview")
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        for column in columns:
            label, width, anchor = headings[column]
            tree.heading(column, text=label, command=lambda col=column, t=tree: self._sort_tree(t, col, False))
            tree.column(column, width=width, anchor=anchor, stretch=column in {"text", "reason", "company"})
        tree.tag_configure("risk_high", foreground=ACCENT_RED, background=TREE_RISK_HIGH_BG)
        tree.tag_configure("risk_medium", foreground=ACCENT_YELLOW, background=TREE_RISK_MEDIUM_BG)
        tree.tag_configure("risk_low", foreground="#C8D0DA", background=TREE_RISK_LOW_BG)
        tree.tag_configure("risk_clean", foreground=ACCENT_GREEN, background=TREE_RISK_CLEAN_BG)
        return tree

    def _on_nav_selected(self, _event=None) -> None:
        if self._nav_syncing:
            return
        selected = self.nav_list.curselection()
        if not selected:
            return
        index = selected[0]
        if 0 <= index < len(MARKET_PREP_NAV_ITEMS):
            self.view_notebook.select(index)

    def _on_view_tab_changed(self, _event=None) -> None:
        if not hasattr(self, "nav_list"):
            return
        index = self.view_notebook.index(self.view_notebook.select())
        self._nav_syncing = True
        try:
            self.nav_list.selection_clear(0, tk.END)
            self.nav_list.selection_set(index)
            self.nav_list.see(index)
        finally:
            self._nav_syncing = False

    def set_display_text(self, text: str) -> None:
        if self.display_text is None:
            return
        self._set_text(self.display_text, text)

    def _set_text(self, widget: scrolledtext.ScrolledText | None, text: str) -> None:
        if widget is None:
            return
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        _apply_text_highlights(widget, text)
        widget.configure(state="disabled")

    def _set_placeholder_status(self, action: str) -> None:
        result = self.orchestrator.run_placeholder(action)
        timestamp = result.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status_var.set(f"{action} placeholder ready.")
        report = str(result.get("report") or MARKET_PREP_PLACEHOLDER_TEXT)
        self._render_plain_text(f"{report}\n\n{action}: placeholder handler invoked at {timestamp}.")

    def _run_background_task(
        self,
        *,
        button_label: str,
        running_status: str,
        loading_text: str,
        worker_func,
        done_status: str,
        report_key: str | None = None,
        export_prefix: str | None = None,
        report_type: str | None = None,
    ) -> None:
        if self.background_task_active:
            self.status_var.set("Market Prep task already running.")
            return

        self.background_task_active = True
        self.latest_report = None
        button = self.buttons.get(button_label)
        if button is not None:
            button.configure(state="disabled")
        self.status_var.set(running_status)
        self._render_plain_text(loading_text)

        def worker() -> None:
            failed = False
            try:
                result = worker_func()
            except Exception as exc:
                failed = True
                self.logger.exception("%s failed.", button_label)
                result = {
                    "report": f"{button_label} failed.\n\n{exc}\n\nSee logs/market_prep.log for details.",
                }

            def finish() -> None:
                self.background_task_active = False
                if button is not None:
                    button.configure(state="normal")
                active_report = result.get(report_key) if report_key and isinstance(result, dict) else None
                report = str(result.get("report") or "No Market Prep output available.")
                if isinstance(active_report, dict):
                    self.latest_report = active_report
                elif report.strip():
                    self.latest_report = self._make_exportable_report(
                        report,
                        report_type=report_type or "market_prep",
                        export_prefix=export_prefix or "market_prep_report",
                    )
                if isinstance(self.latest_report, dict):
                    self._render_report(self.latest_report, fallback_markdown=report)
                else:
                    self._render_plain_text(report)
                status = f"{button_label} finished with errors." if failed else done_status
                self.status_var.set(status if self.latest_report else f"{button_label} finished with errors.")

            try:
                self.container.after(0, finish)
            except RuntimeError:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _make_exportable_report(self, markdown: str, *, report_type: str, export_prefix: str) -> dict:
        generated_at = datetime.now().isoformat(timespec="seconds")
        return {
            "report_type": report_type,
            "report_date": datetime.now().date().isoformat(),
            "export_prefix": export_prefix,
            "generated_at": generated_at,
            "markdown": markdown,
        }

    def _render_plain_text(self, text: str) -> None:
        self._reset_summary_cards()
        self.source_status_var.set("Sources: waiting for structured Market Prep report.")
        self.catalyst_status_var.set("No catalyst clock loaded.")
        self.watchlist_status_var.set("No watchlist risk loaded.")
        self._clear_tree(self.catalyst_tree)
        self._clear_tree(self.watchlist_tree)
        self._set_text(self.overview_text, text)
        self._set_text(
            self.ai_summary_text,
            "AI summary will appear here after Start Day or Start Week runs.",
        )
        self._set_text(self.earnings_text, "")
        self._set_text(self.macro_text, "")
        self._set_text(self.news_sec_text, "")
        self.set_display_text(text)

    def _render_report(self, report: dict, *, fallback_markdown: str = "") -> None:
        markdown = str(report.get("markdown") or fallback_markdown or "")
        self.set_display_text(markdown)
        self._render_summary(report, markdown)
        self._render_overview(report, markdown)
        self._render_ai_summary(report)
        self._render_catalyst_clock(report)
        self._render_watchlist(report)
        self._render_earnings(report)
        self._render_macro(report)
        self._render_news_sec(report)

    def _reset_summary_cards(self) -> None:
        defaults = {
            "report": "Waiting for run",
            "market_regime": "Waiting for run",
            "risk_level": "Waiting for run",
            "next_catalyst": "Waiting for run",
            "watchlist_holds": "Waiting for run",
            "ai_brief": "Waiting for run",
        }
        for key, value in defaults.items():
            if key in self.summary_vars:
                self.summary_vars[key].set(value)

    def _render_summary(self, report: dict, markdown: str) -> None:
        report_type = str(report.get("report_type") or "market_prep").replace("_", " ").title()
        report_date = str(report.get("report_date") or datetime.now().date().isoformat())
        generated_at = str(report.get("generated_at") or "n/a")
        clock_items = self._report_catalyst_clock(report)
        watchlist_rows = self._watchlist_rows_for_report(report)
        hold_warning_count = len(report.get("overnight_hold_warnings") or [])
        next_item = clock_items[0] if clock_items else {}
        next_catalyst = self._format_clock_item_compact(next_item) if next_item else "None in current window"
        market_regime = self._market_regime_text(report)
        risk_level = self._risk_level_text(report)
        ai_status = self._ai_status_text(report)

        self.summary_vars["report"].set(f"{report_type}\n{report_date}\n{generated_at}")
        self.summary_vars["market_regime"].set(market_regime or "Market snapshot unavailable")
        self.summary_vars["risk_level"].set(risk_level or "No risk level generated")
        self.summary_vars["next_catalyst"].set(next_catalyst)
        self.summary_vars["watchlist_holds"].set(f"{hold_warning_count} warning(s)\n{self._watchlist_hold_text(watchlist_rows)}")
        self.summary_vars["ai_brief"].set(ai_status)
        self.source_status_var.set(self._source_health_text(report, markdown))

    def _render_overview(self, report: dict, markdown: str) -> None:
        sections: list[str] = []
        ai_brief = report.get("ai_brief") if isinstance(report.get("ai_brief"), dict) else {}
        ai_summary = str(ai_brief.get("summary") or "").strip()
        if ai_summary:
            sections.append("AI Brief\n" + ai_summary)
        elif ai_brief:
            prompt = str(ai_brief.get("prompt") or "").strip()
            status = str(ai_brief.get("status_label") or ai_brief.get("status") or "").strip()
            if prompt:
                sections.append("AI Prompt / Status\n" + (status or "AI brief unavailable") + "\n\n" + prompt)

        for title, key in (
            ("Weekly Thesis Checklist", "weekly_thesis_checklist"),
            ("Daily Landmine Checklist", "daily_landmine_checklist"),
            ("No-Trade / Reduced-Size Windows", "no_trade_windows"),
            ("Overnight Hold Warnings", "overnight_hold_warnings"),
        ):
            rows = report.get(key)
            if isinstance(rows, list) and rows:
                sections.append(title + "\n" + "\n".join(f"- {row}" for row in rows if str(row).strip()))

        market = report.get("market_snapshot") if isinstance(report.get("market_snapshot"), dict) else {}
        market_text = self._market_snapshot_text(market)
        if market_text:
            sections.append("Market Snapshot\n" + market_text)

        focus = self._markdown_section(markdown, "## 1. Highest Importance Focus", "## 2.")
        if focus:
            sections.append("Highest Importance Focus\n" + focus)

        if str(report.get("report_type") or "").lower() == "weekly":
            week_risk = report.get("week_risk_level") if isinstance(report.get("week_risk_level"), dict) else {}
            if week_risk:
                sections.append(
                    "Week Risk Level\n"
                    f"- Level: {week_risk.get('level') or 'LOW'}\n"
                    f"- Reason: {week_risk.get('reason') or 'No meaningful scheduled events found.'}"
                )
        else:
            landmines = self._markdown_section(markdown, "## 3. Scheduled Landmines Today", "## 4.")
            if landmines:
                sections.append("Scheduled Landmines Today\n" + landmines)

        posture = self._posture_lines(report)
        if posture:
            sections.append("Trading Posture\n" + "\n".join(f"- {line}" for line in posture))

        if not sections:
            sections.append(markdown or "No Market Prep output available.")
        self._set_text(self.overview_text, "\n\n".join(sections).strip())

    def _render_ai_summary(self, report: dict) -> None:
        ai_brief = report.get("ai_brief") if isinstance(report.get("ai_brief"), dict) else {}
        if not ai_brief:
            self._set_text(
                self.ai_summary_text,
                "AI summary was not requested for this report. Run Start Day or Start Week to request it.",
            )
            return

        sections: list[str] = []
        summary = str(ai_brief.get("summary") or "").strip()
        status = str(ai_brief.get("status_label") or ai_brief.get("status") or "").strip()
        model = str(ai_brief.get("model") or "").strip()
        generated_at = str(ai_brief.get("generated_at") or "").strip()
        prompt = str(ai_brief.get("prompt") or "").strip()
        warnings = [
            str(item).strip()
            for item in (ai_brief.get("warnings") or [])
            if str(item).strip()
        ]

        if summary:
            sections.append(summary)
        else:
            sections.append(status or "AI summary unavailable.")

        details = []
        if model:
            details.append(f"Model: {model}")
        if generated_at:
            details.append(f"Generated: {generated_at}")
        if status and summary:
            details.append(f"Status: {status}")
        if details:
            sections.append("Details\n" + "\n".join(details))
        if warnings:
            sections.append("Warnings\n" + "\n".join(f"- {warning}" for warning in warnings))
        if prompt and not summary and not warnings:
            sections.append("Prompt\n" + prompt)

        self._set_text(self.ai_summary_text, "\n\n".join(sections).strip())

    def _render_catalyst_clock(self, report: dict) -> None:
        self._clear_tree(self.catalyst_tree)
        items = self._report_catalyst_clock(report)
        self.catalyst_status_var.set(f"{len(items)} catalyst(s) in the current window.")
        if self.catalyst_tree is None:
            return
        for item in items:
            priority = str(item.get("priority") or "").upper()
            self.catalyst_tree.insert(
                "",
                tk.END,
                values=(
                    item.get("date") or "",
                    item.get("time_et") or "TBD",
                    item.get("bucket") or "Catalyst",
                    priority,
                    item.get("text") or "",
                ),
                tags=(self._risk_tag(priority),),
            )

    def _render_watchlist(self, report: dict) -> None:
        self._clear_tree(self.watchlist_tree)
        rows = self._watchlist_rows_for_report(report)
        flagged = [
            row
            for row in rows
            if "Clean" not in str(row.get("classification") or "")
        ]
        self.watchlist_status_var.set(f"{len(flagged)} flagged ticker(s), {len(rows)} total ticker(s).")
        if self.watchlist_tree is None:
            return
        for row in rows:
            classification = str(row.get("classification") or "")
            self.watchlist_tree.insert(
                "",
                tk.END,
                values=(
                    row.get("ticker") or "",
                    classification,
                    row.get("source_list") or ", ".join(row.get("source_lists") or []),
                    row.get("company") or "",
                    row.get("market_cap_fmt") or self._market_cap_text(row.get("market_cap")),
                    row.get("sector") or row.get("industry") or "",
                    row.get("reason") or "",
                ),
                tags=(self._classification_tag(classification),),
            )

    def _render_earnings(self, report: dict) -> None:
        lines: list[str] = []
        watchlist_rows = [
            row
            for row in self._watchlist_rows_for_report(report)
            if "Earnings" in str(row.get("classification") or "")
        ]
        if watchlist_rows:
            lines.append("Watchlist Earnings Risk")
            lines.extend(self._format_watchlist_row(row) for row in watchlist_rows[:40])
            lines.append("")

        today_tomorrow = report.get("today_tomorrow_earnings") if isinstance(report.get("today_tomorrow_earnings"), dict) else {}
        next_7 = report.get("next_7_earnings") if isinstance(report.get("next_7_earnings"), dict) else {}
        major = report.get("major_earnings") if isinstance(report.get("major_earnings"), dict) else {}
        today_rows = self._payload_rows(today_tomorrow, "earnings")
        next_rows = self._payload_rows(next_7, "earnings")
        major_rows = self._payload_rows(major, "earnings")

        if today_rows:
            lines.append("Today/Tomorrow Earnings")
            lines.extend(self._format_earnings_row(row) for row in today_rows[:40])
            lines.append("")
        if next_rows:
            notable = [
                row
                for row in next_rows
                if str(row.get("importance") or "").upper() in {"MEGA", "HIGH", "MEDIUM"}
            ]
            lines.append("Next 7 Days Notable Earnings")
            lines.extend(self._format_earnings_row(row, include_date=True) for row in (notable or next_rows)[:40])
            lines.append("")
        if major_rows:
            lines.append("Major Earnings This Week")
            lines.extend(self._format_earnings_row(row, include_date=True) for row in major_rows[:60])

        self._set_text(self.earnings_text, "\n".join(lines).strip() or "No earnings rows found.")

    def _render_macro(self, report: dict) -> None:
        lines: list[str] = []
        sections = (
            ("Today", report.get("todays_events")),
            ("Next 7 Days", report.get("next_7_events")),
            ("Weekly Economic Calendar", report.get("economic_calendar")),
            ("Fed Calendar", report.get("fed_calendar")),
            ("Treasury Auction Calendar", report.get("treasury_calendar")),
        )
        for title, payload in sections:
            if not isinstance(payload, dict):
                continue
            rows = self._payload_rows(payload, "events")
            if not rows:
                continue
            lines.append(title)
            lines.extend(self._format_event_row(row) for row in rows[:80])
            lines.append("")
        self._set_text(self.macro_text, "\n".join(lines).strip() or "No macro/Fed/Treasury events found.")

    def _render_news_sec(self, report: dict) -> None:
        lines: list[str] = []
        sec_filings = report.get("sec_filings") if isinstance(report.get("sec_filings"), dict) else {}
        filings = self._payload_rows(sec_filings, "filings")
        if filings:
            lines.append("SEC Filing Risk")
            lines.extend(self._format_sec_row(row) for row in filings[:40])
            lines.append("")
        elif sec_filings:
            lines.append("SEC Filing Risk")
            lines.append(str(sec_filings.get("message") or "No SEC filing risk found."))
            lines.append("")

        headlines_payload = report.get("rss_headlines") if isinstance(report.get("rss_headlines"), dict) else {}
        headlines = self._payload_rows(headlines_payload, "headlines")
        if headlines:
            lines.append("Google News/RSS Headline Risk")
            lines.extend(self._format_headline_row(row) for row in headlines[:35])
            lines.append("")
        elif headlines_payload:
            lines.append("Google News/RSS Headline Risk")
            lines.append(str(headlines_payload.get("message") or "No RSS headlines found."))
            lines.append("")

        youtube_payload = report.get("youtube_links") if isinstance(report.get("youtube_links"), dict) else {}
        videos = self._payload_rows(youtube_payload, "videos")
        if videos:
            lines.append("YouTube Links")
            lines.extend(self._format_video_row(row) for row in videos[:25])
        elif youtube_payload:
            lines.append("YouTube Links")
            lines.append(str(youtube_payload.get("message") or "No configured YouTube links found."))

        self._set_text(self.news_sec_text, "\n".join(lines).strip() or "No News/SEC rows found.")

    def _report_catalyst_clock(self, report: dict) -> list[dict]:
        clock = report.get("catalyst_clock") if isinstance(report, dict) else []
        if isinstance(clock, list) and clock:
            return [row for row in clock if isinstance(row, dict)]
        economic = report.get("next_7_events") or report.get("economic_calendar") or {}
        earnings = report.get("next_7_earnings") or report.get("major_earnings") or {}
        watchlist = report.get("watchlist_risk") or report.get("watchlist_earnings_risk") or {}
        try:
            return build_catalyst_clock(
                economic if isinstance(economic, dict) else {},
                earnings if isinstance(earnings, dict) else {},
                watchlist if isinstance(watchlist, dict) else {},
                fed_calendar=report.get("fed_calendar") if isinstance(report.get("fed_calendar"), dict) else {},
                treasury_calendar=report.get("treasury_calendar") if isinstance(report.get("treasury_calendar"), dict) else {},
                sec_filings=report.get("sec_filings") if isinstance(report.get("sec_filings"), dict) else {},
            )
        except Exception:
            self.logger.exception("Failed building Market Prep catalyst clock view.")
            return []

    def _watchlist_rows_for_report(self, report: dict) -> list[dict]:
        payload = report.get("watchlist_risk") or report.get("watchlist_earnings_risk") or {}
        return self._payload_rows(payload if isinstance(payload, dict) else {}, "risks")

    def _posture_lines(self, report: dict) -> list[str]:
        posture = report.get("trading_posture")
        if not isinstance(posture, list):
            posture = report.get("swing_trading_conditions")
        return [str(item).strip() for item in posture if str(item).strip()] if isinstance(posture, list) else []

    def _posture_text(self, report: dict) -> str:
        lines = self._posture_lines(report)
        if lines:
            return lines[0]
        week_risk = report.get("week_risk_level") if isinstance(report.get("week_risk_level"), dict) else {}
        if week_risk:
            return f"{week_risk.get('level') or 'LOW'}: {week_risk.get('reason') or ''}".strip()
        return ""

    def _risk_level_text(self, report: dict) -> str:
        week_risk = report.get("week_risk_level") if isinstance(report.get("week_risk_level"), dict) else {}
        if week_risk:
            return f"{week_risk.get('level') or 'LOW'}\n{week_risk.get('reason') or ''}".strip()
        landmines = report.get("daily_landmine_checklist") if isinstance(report.get("daily_landmine_checklist"), list) else []
        no_trade = report.get("no_trade_windows") if isinstance(report.get("no_trade_windows"), list) else []
        if no_trade:
            return f"HIGH\n{len(no_trade)} timing window(s)"
        if landmines:
            return f"WATCH\n{len(landmines)} checklist item(s)"
        return ""

    def _market_regime_text(self, report: dict) -> str:
        market = report.get("market_snapshot") if isinstance(report.get("market_snapshot"), dict) else {}
        classification = market.get("classification") if isinstance(market.get("classification"), dict) else {}
        if not classification:
            return ""
        return f"{classification.get('label') or 'n/a'}\n{classification.get('reason') or 'n/a'}"

    def _ai_status_text(self, report: dict) -> str:
        ai = report.get("ai_brief") if isinstance(report.get("ai_brief"), dict) else {}
        if not ai:
            return "Not requested"
        if str(ai.get("summary") or "").strip():
            return "Ready\nOpenAI brief generated"
        return str(ai.get("status_label") or ai.get("status") or "Unavailable")

    def _watchlist_hold_text(self, rows: list[dict]) -> str:
        flagged = [
            str(row.get("ticker") or "").strip().upper()
            for row in rows
            if "Clean" not in str(row.get("classification") or "")
        ]
        return ", ".join(flagged[:10]) if flagged else "No flagged holds"

    def _market_snapshot_text(self, snapshot: dict) -> str:
        classification = snapshot.get("classification") if isinstance(snapshot.get("classification"), dict) else {}
        lines = []
        if classification:
            lines.append(f"- {classification.get('label') or 'n/a'}: {classification.get('reason') or 'n/a'}")
        rows = self._payload_rows(snapshot, "rows")
        for row in rows[:12]:
            ticker = str(row.get("ticker") or "").strip()
            ret_5d = self._pct_text(row.get("return_5d_pct"))
            ret_20d = self._pct_text(row.get("return_20d_pct"))
            above_21 = "above 21SMA" if row.get("above_21_sma") is True else "below 21SMA" if row.get("above_21_sma") is False else "21SMA n/a"
            lines.append(f"- {ticker}: 5D {ret_5d}, 20D {ret_20d}, {above_21}")
        errors = snapshot.get("errors") if isinstance(snapshot, dict) else []
        if isinstance(errors, list) and errors:
            lines.append(f"- Warning: {errors[0]}")
        return "\n".join(lines)

    def _pct_text(self, value) -> str:
        try:
            return f"{float(value):+.2f}%"
        except (TypeError, ValueError):
            return "n/a"

    def _source_health_text(self, report: dict, markdown: str) -> str:
        items: list[str] = []
        for payload in (
            report.get("todays_events"),
            report.get("next_7_events"),
            report.get("economic_calendar"),
        ):
            if isinstance(payload, dict):
                self._append_nested_status(items, "ForexFactory", payload.get("forexfactory_status"))

        for label, payload in (
            ("Federal Reserve", report.get("fed_calendar")),
            ("Treasury", report.get("treasury_calendar")),
            ("SEC", report.get("sec_filings")),
            ("RSS", report.get("rss_headlines")),
            ("Market Snapshot", report.get("market_snapshot")),
            ("OpenAI", report.get("ai_brief")),
        ):
            if isinstance(payload, dict):
                self._append_status(items, label, payload)

        for payload in (
            report.get("today_tomorrow_earnings"),
            report.get("next_7_earnings"),
            report.get("major_earnings"),
            report.get("watchlist_risk"),
            report.get("watchlist_earnings_risk"),
        ):
            if isinstance(payload, dict):
                self._append_nested_status(items, "yfinance", payload.get("yfinance_status"))

        if not items:
            return "Sources: structured source status unavailable." if markdown else "Sources: waiting for first run."
        return "Sources: " + " | ".join(self._dedupe(items))

    def _append_status(self, items: list[str], label: str, payload: dict) -> None:
        status = str(payload.get("status_label") or payload.get("status") or "").strip()
        message = str(payload.get("message") or "").strip()
        warning = ""
        warnings = payload.get("warnings")
        if isinstance(warnings, list) and warnings:
            warning = str(warnings[0]).strip()
        value = status or message
        if value:
            items.append(f"{label}: {value}")
        if warning:
            items.append(f"{label} warning: {warning}")

    def _append_nested_status(self, items: list[str], label: str, status_payload) -> None:
        if not isinstance(status_payload, dict):
            return
        status = str(status_payload.get("status_label") or status_payload.get("status") or "").strip()
        if status and status != "No metadata requested":
            items.append(f"{label}: {status}")
        warnings = status_payload.get("warnings")
        if isinstance(warnings, list) and warnings:
            items.append(f"{label} warning: {warnings[0]}")

    def _clear_tree(self, tree: ttk.Treeview | None) -> None:
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)

    def _sort_tree(self, tree: ttk.Treeview, column: str, reverse: bool) -> None:
        rows = [(self._sort_value(tree.set(item, column)), item) for item in tree.get_children("")]
        rows.sort(reverse=reverse)
        for index, (_value, item) in enumerate(rows):
            tree.move(item, "", index)
        tree.heading(column, command=lambda: self._sort_tree(tree, column, not reverse))

    def _sort_value(self, value: str):
        text = str(value or "").strip()
        try:
            return float(text.replace("$", "").replace(",", "").replace("B", "").replace("M", ""))
        except ValueError:
            return text.lower()

    def _risk_tag(self, priority: str) -> str:
        value = str(priority or "").upper()
        if value in HIGH_PRIORITY_VALUES:
            return "risk_high"
        if value in MEDIUM_PRIORITY_VALUES:
            return "risk_medium"
        return "risk_low"

    def _classification_tag(self, classification: str) -> str:
        text = str(classification or "")
        if "Clean" in text:
            return "risk_clean"
        if "Today/Tomorrow" in text:
            return "risk_high"
        if "Earnings" in text or "Risk" in text:
            return "risk_medium"
        return "risk_low"

    def _markdown_section(self, markdown: str, header: str, next_prefix: str) -> str:
        if not markdown or header not in markdown:
            return ""
        after = markdown.split(header, 1)[1]
        marker_index = after.find(next_prefix)
        if marker_index >= 0:
            after = after[:marker_index]
        return after.strip()

    def _payload_rows(self, payload: dict, key: str) -> list[dict]:
        rows = payload.get(key) if isinstance(payload, dict) else []
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    def _format_clock_item_compact(self, item: dict) -> str:
        date_value = str(item.get("date") or "").strip()
        time_value = str(item.get("time_et") or "TBD").strip()
        bucket = str(item.get("bucket") or "Catalyst").strip()
        text = str(item.get("text") or "").strip()
        return f"{date_value} {time_value} ET\n{bucket}: {text}".strip()

    def _format_event_row(self, row: dict) -> str:
        priority = str(row.get("priority") or "").strip().upper()
        currency = str(row.get("currency") or "").strip()
        time_value = str(row.get("time_et") or "TBD").strip()
        parts = [
            str(row.get("date") or "").strip(),
            f"{time_value} ET",
            f"[{priority}]" if priority else "",
            currency,
            str(row.get("event") or "").strip(),
        ]
        stats = self._event_stats_text(row)
        notes = str(row.get("notes") or "").strip()
        line = " ".join(part for part in parts if part)
        if stats:
            line += f" | {stats}"
        if notes:
            line += f" | {notes}"
        return "- " + line

    def _format_earnings_row(self, row: dict, *, include_date: bool = False) -> str:
        parts = []
        if include_date:
            parts.append(str(row.get("date") or "").strip())
        parts.extend(
            [
                str(row.get("ticker") or "").strip().upper(),
                str(row.get("company") or row.get("company_yfinance") or "").strip(),
                str(row.get("time") or "").strip().upper(),
                row.get("market_cap_fmt") or self._market_cap_text(row.get("market_cap")),
                str(row.get("sector") or row.get("industry") or "").strip(),
                str(row.get("importance") or "").strip().upper(),
                str(row.get("source") or "").strip(),
            ]
        )
        notes = str(row.get("notes") or "").strip()
        line = " | ".join(part if part else "n/a" for part in parts)
        if notes:
            line += f" | {notes}"
        return "- " + line

    def _format_watchlist_row(self, row: dict) -> str:
        parts = [
            str(row.get("ticker") or "").strip().upper(),
            str(row.get("classification") or "").strip(),
            str(row.get("source_list") or ", ".join(row.get("source_lists") or [])).strip(),
            str(row.get("company") or "").strip(),
            row.get("market_cap_fmt") or self._market_cap_text(row.get("market_cap")),
            str(row.get("reason") or "").strip(),
        ]
        return "- " + " | ".join(part if part else "n/a" for part in parts)

    def _format_sec_row(self, row: dict) -> str:
        parts = [
            str(row.get("ticker") or "").strip().upper(),
            str(row.get("form") or "").strip(),
            str(row.get("filing_date") or "").strip(),
            str(row.get("risk_classification") or "").strip().upper(),
            ", ".join(row.get("matched_keywords") or []),
            str(row.get("url") or "").strip(),
        ]
        return "- " + " | ".join(part if part else "n/a" for part in parts)

    def _format_headline_row(self, row: dict) -> str:
        tags = ", ".join(row.get("tags") or [])
        prefix = f"[{tags}] " if tags else ""
        parts = [
            prefix + str(row.get("title") or "").strip(),
            str(row.get("source") or "").strip(),
            str(row.get("query") or "").strip(),
            str(row.get("published") or "").strip(),
            str(row.get("url") or "").strip(),
        ]
        return "- " + " | ".join(part for part in parts if part)

    def _format_video_row(self, row: dict) -> str:
        keywords = ", ".join(row.get("matched_keywords") or [])
        parts = [
            str(row.get("creator") or "").strip(),
            str(row.get("title") or "").strip(),
            str(row.get("published") or "").strip(),
            keywords,
            str(row.get("url") or "").strip(),
        ]
        return "- " + " | ".join(part for part in parts if part)

    def _event_stats_text(self, row: dict) -> str:
        parts = []
        for label, key in (("Actual", "actual"), ("Forecast", "forecast"), ("Previous", "previous")):
            value = str(row.get(key) or "").strip()
            if value:
                parts.append(f"{label}: {value}")
        return ", ".join(parts)

    def _market_cap_text(self, value) -> str:
        try:
            market_cap = float(value)
        except (TypeError, ValueError):
            return ""
        if market_cap <= 0:
            return ""
        for suffix, divisor in (("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000)):
            if abs(market_cap) >= divisor:
                return f"${market_cap / divisor:.2f}{suffix}"
        return f"${market_cap:,.0f}"

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                deduped.append(text)
        return deduped

    def _forexfactory_enabled(self) -> bool:
        return bool(self.config.features.get("forexfactory_calendar")) and bool(
            self.config.forexfactory.get("enabled")
        )

    def _refresh_forexfactory_button(self) -> None:
        button = self.buttons.get("Toggle ForexFactory")
        if button is None:
            return
        button.configure(text="Disable ForexFactory" if self._forexfactory_enabled() else "Enable ForexFactory")

    def toggle_forexfactory(self) -> None:
        next_enabled = not self._forexfactory_enabled()
        try:
            result = self.orchestrator.set_forexfactory_enabled(next_enabled)
            self.config = load_market_prep_config()
            self.output_dir = self.config.resolved_paths().get("output_dir", self.output_dir)
            self._refresh_forexfactory_button()
        except Exception as exc:
            self.logger.exception("Failed toggling ForexFactory calendar.")
            messagebox.showerror("ForexFactory", f"Could not update ForexFactory setting:\n\n{exc}")
            self.status_var.set(f"ForexFactory setting update failed: {exc}")
            return

        state = "enabled" if result.get("enabled") else "disabled"
        config_path = result.get("config_path") or "config/market_prep_config.json"
        message = (
            f"ForexFactory calendar {state}.\n\n"
            f"Config updated: {config_path}\n\n"
            "Click Refresh Economic Calendar to test it now."
        )
        self._render_plain_text(message)
        self.status_var.set(f"ForexFactory calendar {state}.")

    def run_daily_prep(self) -> None:
        self._run_background_task(
            button_label="Run Daily Prep",
            running_status="Building daily market prep report...",
            loading_text="Fetching configured scheduled-risk sources...",
            worker_func=self.orchestrator.run_daily_prep,
            done_status="Daily market prep report ready.",
            report_key="daily_report",
        )

    def run_weekly_prep(self) -> None:
        self._run_background_task(
            button_label="Run Weekly Prep",
            running_status="Building weekly market prep report...",
            loading_text="Fetching weekly scheduled-risk inputs...",
            worker_func=self.orchestrator.run_weekly_prep,
            done_status="Weekly market prep report ready.",
            report_key="weekly_report",
        )

    def start_day(self) -> None:
        self._run_background_task(
            button_label="Start Day",
            running_status="Building start-of-day command center...",
            loading_text="Fetching daily landmines, market snapshot, watchlist risk, and optional AI brief...",
            worker_func=self.orchestrator.start_day_prep,
            done_status="Start-of-day command center ready.",
            report_key="daily_report",
        )

    def start_week(self) -> None:
        self._run_background_task(
            button_label="Start Week",
            running_status="Building start-of-week thesis...",
            loading_text="Fetching weekly catalysts, market snapshot, watchlist risk, and optional AI brief...",
            worker_func=self.orchestrator.start_week_prep,
            done_status="Start-of-week thesis ready.",
            report_key="weekly_report",
        )

    def run_ai_summary(self) -> None:
        if self.background_task_active:
            self.status_var.set("Market Prep task already running.")
            return
        if not isinstance(self.latest_report, dict):
            message = "Run Start Day or Start Week before running the AI summary."
            self.status_var.set(message)
            self._set_text(self.ai_summary_text, message)
            self._select_market_prep_tab("AI Summary")
            return
        report_type = str(self.latest_report.get("report_type") or "").strip().lower()
        if report_type not in {"daily", "weekly"}:
            message = "Run Start Day or Start Week before running the AI summary."
            self.status_var.set(message)
            self._set_text(self.ai_summary_text, message)
            self._select_market_prep_tab("AI Summary")
            return

        self.background_task_active = True
        button = self.buttons.get("Run AI Summary")
        if button is not None:
            button.configure(state="disabled")
        self.status_var.set("Generating Market Prep AI summary...")
        self.summary_vars["ai_brief"].set("Running\nOpenAI summary requested")
        self._set_text(self.ai_summary_text, "Generating Market Prep AI summary...")
        self._select_market_prep_tab("AI Summary")
        report = dict(self.latest_report)

        def worker() -> None:
            try:
                updated_report = self.orchestrator.run_ai_summary(report)
                error = ""
            except Exception as exc:
                self.logger.exception("Run AI Summary failed.")
                updated_report = None
                error = str(exc)

            def finish() -> None:
                self.background_task_active = False
                if button is not None:
                    button.configure(state="normal")
                if updated_report is None:
                    message = f"Run AI Summary failed: {error}"
                    self.status_var.set(message)
                    self.summary_vars["ai_brief"].set(error or "AI summary failed")
                    self._set_text(self.ai_summary_text, message)
                    self._select_market_prep_tab("AI Summary")
                    return
                self.latest_report = updated_report
                self._render_report(updated_report)
                self._select_market_prep_tab("AI Summary")
                self.status_var.set("Market Prep AI summary ready.")

            try:
                self.container.after(0, finish)
            except RuntimeError:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _select_market_prep_tab(self, title: str) -> None:
        if not hasattr(self, "view_notebook"):
            return
        try:
            for tab_id in self.view_notebook.tabs():
                if str(self.view_notebook.tab(tab_id, "text")) == title:
                    self.view_notebook.select(tab_id)
                    return
        except tk.TclError:
            return

    def refresh_earnings(self) -> None:
        self._run_background_task(
            button_label="Refresh Earnings",
            running_status="Refreshing earnings calendar...",
            loading_text="Loading Nasdaq earnings and yfinance market-cap metadata...",
            worker_func=lambda: self.orchestrator.refresh_earnings(days=7),
            done_status="Earnings calendar refreshed.",
            report_type="earnings",
            export_prefix="earnings_calendar",
        )

    def refresh_economic_calendar(self) -> None:
        self._run_background_task(
            button_label="Refresh Economic Calendar",
            running_status="Refreshing economic calendar...",
            loading_text="Loading manual economic calendar and optional ForexFactory enrichment...",
            worker_func=lambda: self.orchestrator.refresh_economic_calendar(days=7),
            done_status="Economic calendar refreshed.",
            export_prefix="economic_calendar",
            report_type="economic_calendar",
        )

    def refresh_news(self) -> None:
        self._run_background_task(
            button_label="Refresh News",
            running_status="Refreshing RSS macro headlines...",
            loading_text="Fetching RSS headlines from configured feeds...",
            worker_func=lambda: self.orchestrator.refresh_news(limit=25),
            done_status="RSS macro headlines refreshed.",
            export_prefix="rss_news",
            report_type="rss_news",
        )

    def refresh_youtube_links(self) -> None:
        self._run_background_task(
            button_label="Refresh YouTube Links",
            running_status="Refreshing YouTube RSS links...",
            loading_text="Fetching YouTube channel RSS links from configured creators...",
            worker_func=lambda: self.orchestrator.refresh_youtube_links(limit=25),
            done_status="YouTube RSS links refreshed.",
            export_prefix="youtube_links",
            report_type="youtube_links",
        )

    def scan_watchlists(self) -> None:
        self._run_background_task(
            button_label="Scan Watchlists",
            running_status="Scanning watchlists for scheduled risk...",
            loading_text="Reading longs.txt, shorts.txt, and manual calendars...",
            worker_func=self.orchestrator.scan_watchlists,
            done_status="Watchlist risk scan complete.",
            export_prefix="watchlist_risk_scan",
            report_type="watchlist_risk",
        )

    def export_markdown(self) -> None:
        if not self.latest_report:
            message = "Run Daily Prep or Weekly Prep before exporting markdown."
            self.status_var.set(message)
            self._render_plain_text(message)
            return

        try:
            result = self.orchestrator.export_markdown(self.latest_report)
        except Exception as exc:
            self.logger.exception("Export Markdown failed.")
            messagebox.showerror("Export Markdown", str(exc))
            self.status_var.set(f"Export Markdown failed: {exc}")
            return

        markdown = str(result.get("markdown") or self.latest_report.get("markdown") or "")
        if markdown:
            self.set_display_text(markdown)
        self.status_var.set(f"Exported markdown: {result.get('path')}")

    def open_output_folder(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            open_path_in_file_manager(self.output_dir)
        except Exception as exc:
            messagebox.showerror(
                "Open Output Folder",
                f"Could not open output folder:\n{self.output_dir}\n\n{exc}",
            )
            return
        self.status_var.set(f"Opened output folder: {self.output_dir}")


class TickerLookupTab:
    """Single-symbol event, earnings, filing, and industry-news lookup."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        output_dir: Path = OUTPUT_DIR,
        text_bg: str = "#252525",
        text_fg: str = "#E0E0E0",
        compact_layout: bool = False,
    ):
        self.parent = parent
        self.output_dir = Path(output_dir)
        self.text_bg = text_bg
        self.text_fg = text_fg
        self.compact_layout = bool(compact_layout)
        self.config = load_market_prep_config()
        self.output_dir = self.config.resolved_paths().get("output_dir", self.output_dir)
        self.logger = get_market_prep_logger()
        self.container = ttk.Frame(parent)
        self.ticker_var = tk.StringVar()
        self.days_var = tk.StringVar(value=str(self.config.ticker_lookup.get("days_ahead", 10)))
        self.status_var = tk.StringVar(value="Enter a ticker and run lookup.")
        self.summary_var = tk.StringVar(value="No ticker loaded.")
        self.latest_lookup: dict | None = None
        self.background_task_active = False
        self.lookup_button: ttk.Button | None = None
        self.earnings_tree: ttk.Treeview | None = None
        self.peer_tree: ttk.Treeview | None = None
        self.headline_tree: ttk.Treeview | None = None
        self.sec_tree: ttk.Treeview | None = None
        self.overview_text: scrolledtext.ScrolledText | None = None
        self.ticker_ai_text: scrolledtext.ScrolledText | None = None
        self.raw_text: scrolledtext.ScrolledText | None = None
        _configure_readable_tree_style(parent, self.text_bg, self.text_fg)
        self._build_layout()

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self.container)
        toolbar.pack(fill=tk.X, padx=6 if self.compact_layout else 10, pady=(4 if self.compact_layout else 10, 4))
        ttk.Label(toolbar, text="Ticker").pack(side=tk.LEFT, padx=(0, 6))
        entry = ttk.Entry(toolbar, textvariable=self.ticker_var, width=10 if self.compact_layout else 14)
        entry.pack(side=tk.LEFT, padx=(0, 6 if self.compact_layout else 8))
        entry.bind("<Return>", lambda _event: self.run_lookup())
        ttk.Label(toolbar, text="Days").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Spinbox(toolbar, from_=7, to=180, increment=1, textvariable=self.days_var, width=4 if self.compact_layout else 6).pack(
            side=tk.LEFT,
            padx=(0, 6 if self.compact_layout else 8),
        )
        self.lookup_button = ttk.Button(toolbar, text="Lookup", command=self.run_lookup)
        self.lookup_button.pack(side=tk.LEFT, padx=(0, 6 if self.compact_layout else 8))
        ttk.Button(toolbar, text="Export" if self.compact_layout else "Export Markdown", command=self.export_markdown).pack(
            side=tk.LEFT,
            padx=(0, 6 if self.compact_layout else 8),
        )
        ttk.Button(toolbar, text="Folder" if self.compact_layout else "Open Output Folder", command=self.open_output_folder).pack(side=tk.LEFT)

        summary = ttk.LabelFrame(self.container, text="Lookup Summary")
        summary.pack(fill=tk.X, padx=6 if self.compact_layout else 10, pady=(0, 4 if self.compact_layout else 8))
        ttk.Label(summary, textvariable=self.summary_var, justify="left", wraplength=700 if self.compact_layout else 1500).pack(
            fill=tk.X,
            padx=6 if self.compact_layout else 8,
            pady=(3 if self.compact_layout else 6, 4 if self.compact_layout else 8),
        )

        self.notebook = ttk.Notebook(self.container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6 if self.compact_layout else 10, pady=(0, 4 if self.compact_layout else 8))
        self._build_overview_tab()
        self._build_ai_tab()
        self._build_earnings_tab()
        self._build_news_tab()
        self._build_sec_tab()
        self._build_raw_tab()

        status = ttk.Label(self.container, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill=tk.X, padx=6 if self.compact_layout else 10, pady=(0, 6 if self.compact_layout else 10))

    def _build_overview_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Overview")
        self.overview_text = self._make_text(frame)
        self._set_text(self.overview_text, "Ticker Lookup is ready. Enter a symbol such as TSM, NVDA, AAPL, or AMD.")

    def _build_ai_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="AI Brief")
        self.ticker_ai_text = self._make_text(frame)
        self._set_text(
            self.ticker_ai_text,
            "Ticker AI brief will appear here after lookup. Save an OpenAI key in Market Prep > API Key first.",
        )

    def _build_earnings_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Earnings / Events")
        target_frame = ttk.LabelFrame(frame, text="Ticker Earnings")
        target_frame.pack(fill=tk.BOTH, expand=True, padx=6 if self.compact_layout else 8, pady=(4 if self.compact_layout else 8, 2 if self.compact_layout else 4))
        self.earnings_tree = self._make_tree(
            target_frame,
            ("date", "ticker", "company", "time", "importance", "market_cap", "notes"),
            {
                "date": ("Date", 100, "w"),
                "ticker": ("Ticker", 80, "w"),
                "company": ("Company", 220, "w"),
                "time": ("Time", 70, "w"),
                "importance": ("Risk", 90, "w"),
                "market_cap": ("Market Cap", 110, "e"),
                "notes": ("Notes", 520, "w"),
            },
        )
        peer_frame = ttk.LabelFrame(frame, text="Major Peer Earnings / Events")
        peer_frame.pack(fill=tk.BOTH, expand=True, padx=6 if self.compact_layout else 8, pady=(2 if self.compact_layout else 4, 4 if self.compact_layout else 8))
        self.peer_tree = self._make_tree(
            peer_frame,
            ("date", "ticker", "company", "time", "importance", "market_cap", "notes"),
            {
                "date": ("Date", 100, "w"),
                "ticker": ("Ticker", 80, "w"),
                "company": ("Company", 220, "w"),
                "time": ("Time", 70, "w"),
                "importance": ("Risk", 90, "w"),
                "market_cap": ("Market Cap", 110, "e"),
                "notes": ("Notes", 520, "w"),
            },
        )

    def _build_news_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="News / Industry")
        self.headline_tree = self._make_tree(
            frame,
            ("scope", "title", "source", "query", "published", "url"),
            {
                "scope": ("Scope", 95, "w"),
                "title": ("Headline", 520, "w"),
                "source": ("Source", 170, "w"),
                "query": ("Query", 220, "w"),
                "published": ("Published", 180, "w"),
                "url": ("URL", 360, "w"),
            },
        )

    def _build_sec_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="SEC / Announcements")
        self.sec_tree = self._make_tree(
            frame,
            ("ticker", "form", "date", "risk", "keywords", "url"),
            {
                "ticker": ("Ticker", 80, "w"),
                "form": ("Form", 80, "w"),
                "date": ("Filing Date", 105, "w"),
                "risk": ("Risk", 90, "w"),
                "keywords": ("Matched Keywords", 260, "w"),
                "url": ("URL", 680, "w"),
            },
        )

    def _build_raw_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Raw Markdown")
        self.raw_text = self._make_text(frame)

    def _make_text(self, parent: tk.Misc) -> scrolledtext.ScrolledText:
        widget = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=TEXT_FONT,
            bg=self.text_bg,
            fg=self.text_fg,
            insertbackground=self.text_fg,
            selectbackground="#4A4A4A",
            selectforeground=self.text_fg,
        )
        widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        _configure_text_tags(widget)
        widget.configure(state="disabled")
        return widget

    def _make_tree(
        self,
        parent: tk.Misc,
        columns: tuple[str, ...],
        headings: dict[str, tuple[str, int, str]],
    ) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=6 if self.compact_layout else 8, pady=4 if self.compact_layout else 8)
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse", style="Readable.Treeview")
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        for column in columns:
            label, width, anchor = headings[column]
            tree.heading(column, text=label, command=lambda col=column, t=tree: self._sort_tree(t, col, False))
            tree.column(column, width=width, anchor=anchor, stretch=column in {"title", "notes", "url"})
        tree.tag_configure("risk_high", foreground=ACCENT_RED, background=TREE_RISK_HIGH_BG)
        tree.tag_configure("risk_medium", foreground=ACCENT_YELLOW, background=TREE_RISK_MEDIUM_BG)
        tree.tag_configure("risk_low", foreground="#C8D0DA", background=TREE_RISK_LOW_BG)
        tree.tag_configure("scope_ticker", foreground=ACCENT_BLUE, background="#202B38")
        tree.tag_configure("scope_industry", foreground=ACCENT_PURPLE, background="#2B2838")
        return tree

    def run_lookup(self) -> None:
        if self.background_task_active:
            self.status_var.set("Ticker lookup already running.")
            return
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            self.status_var.set("Enter a ticker symbol before running lookup.")
            return
        try:
            days = int(self.days_var.get())
        except ValueError:
            days = 10
            self.days_var.set("10")

        self.background_task_active = True
        if self.lookup_button is not None:
            self.lookup_button.configure(state="disabled")
        self.status_var.set(f"Looking up {ticker}...")
        self.summary_var.set(f"{ticker}: lookup running.")
        self._set_text(
            self.overview_text,
            "Fetching ticker metadata, earnings, filings, broad Google News RSS context, and optional AI brief...",
        )
        self._set_text(self.ticker_ai_text, "Ticker AI brief running...")

        def worker() -> None:
            try:
                result = lookup_ticker_context(ticker, config=self.config, days_ahead=days)
                error = ""
            except Exception as exc:
                self.logger.exception("Ticker lookup failed for %s.", ticker)
                result = None
                error = str(exc)

            def finish() -> None:
                self.background_task_active = False
                if self.lookup_button is not None:
                    self.lookup_button.configure(state="normal")
                if result is None:
                    self.status_var.set(f"Ticker lookup failed: {error}")
                    self._set_text(self.overview_text, f"Ticker lookup failed.\n\n{error}")
                    return
                self.latest_lookup = result
                self._render_lookup(result)
                self.status_var.set(f"{ticker} lookup ready.")

            try:
                self.container.after(0, finish)
            except RuntimeError:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _render_lookup(self, payload: dict) -> None:
        ticker = str(payload.get("ticker") or "").strip().upper()
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        target_earnings = payload.get("target_earnings") if isinstance(payload.get("target_earnings"), list) else []
        peer_earnings = payload.get("peer_earnings") if isinstance(payload.get("peer_earnings"), list) else []
        target_headlines = payload.get("target_headlines") if isinstance(payload.get("target_headlines"), list) else []
        industry_headlines = payload.get("industry_headlines") if isinstance(payload.get("industry_headlines"), list) else []
        landmine_headlines = payload.get("landmine_headlines") if isinstance(payload.get("landmine_headlines"), list) else []
        peers = payload.get("peer_tickers") if isinstance(payload.get("peer_tickers"), list) else []
        statuses = payload.get("source_status") if isinstance(payload.get("source_status"), list) else []
        ai_brief = payload.get("ai_brief") if isinstance(payload.get("ai_brief"), dict) else {}
        self.summary_var.set(
            f"{ticker} | {metadata.get('company_name') or 'Company unknown'} | "
            f"{metadata.get('sector') or 'sector n/a'} / {metadata.get('industry') or 'industry n/a'} | "
            f"Market cap: {metadata.get('market_cap_fmt') or 'n/a'} | "
            f"Peers: {', '.join(peers) or 'None'} | "
            f"Earnings: {len(target_earnings)} ticker, {len(peer_earnings)} peer | "
            f"Headlines: {len(target_headlines)} ticker, {len(industry_headlines)} industry, {len(landmine_headlines)} landmine"
        )
        overview = [
            f"{ticker} Lookup",
            "=" * 80,
            f"Company: {metadata.get('company_name') or 'n/a'}",
            f"Sector: {metadata.get('sector') or 'n/a'}",
            f"Industry: {metadata.get('industry') or 'n/a'}",
            f"Market cap: {metadata.get('market_cap_fmt') or 'n/a'}",
            f"Peer context: {payload.get('peer_reason') or 'n/a'}",
            f"Peer tickers: {', '.join(peers) or 'None'}",
            f"Landmine-tagged headlines: {len(landmine_headlines)}",
            "",
            "AI Brief Status",
            "-" * 80,
            str(ai_brief.get("status_label") or ai_brief.get("status") or "AI brief not generated.").strip(),
            "",
            "Source Status",
            "-" * 80,
        ]
        overview.extend(f"- {status}" for status in statuses)
        self._set_text(self.overview_text, "\n".join(overview).rstrip())
        self._render_ticker_ai_brief(ai_brief, payload)
        self._set_text(self.raw_text, str(payload.get("markdown") or ""))
        self._render_earnings_tree(self.earnings_tree, target_earnings)
        self._render_earnings_tree(self.peer_tree, peer_earnings)
        self._render_headlines(target_headlines, industry_headlines, landmine_headlines)
        self._render_sec(payload.get("sec_filings") if isinstance(payload.get("sec_filings"), dict) else {})

    def _render_ticker_ai_brief(self, ai_brief: dict, payload: dict) -> None:
        if not ai_brief:
            self._set_text(self.ticker_ai_text, "AI brief was not generated.")
            return
        sections: list[str] = []
        summary = str(ai_brief.get("summary") or "").strip()
        status = str(ai_brief.get("status_label") or ai_brief.get("status") or "").strip()
        model = str(ai_brief.get("model") or "").strip()
        generated_at = str(ai_brief.get("generated_at") or "").strip()
        prompt = str(ai_brief.get("prompt") or "").strip()
        warnings = [
            str(item).strip()
            for item in (ai_brief.get("warnings") or [])
            if str(item).strip()
        ]
        if summary:
            sections.append(summary)
        else:
            sections.append(status or "AI brief unavailable.")
        details = []
        if model:
            details.append(f"Model: {model}")
        if generated_at:
            details.append(f"Generated: {generated_at}")
        if status and summary:
            details.append(f"Status: {status}")
        if details:
            sections.append("Details\n" + "\n".join(details))
        if warnings:
            sections.append("Warnings\n" + "\n".join(f"- {warning}" for warning in warnings))
        if not summary and str(payload.get("ai_swing_query") or "").strip():
            sections.append("Manual AI Query\n" + str(payload.get("ai_swing_query") or "").strip())
        self._set_text(self.ticker_ai_text, "\n\n".join(sections).strip())

    def _render_earnings_tree(self, tree: ttk.Treeview | None, rows: list[dict]) -> None:
        self._clear_tree(tree)
        if tree is None:
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            importance = str(row.get("importance") or "").upper()
            tree.insert(
                "",
                tk.END,
                values=(
                    row.get("date") or "",
                    row.get("ticker") or "",
                    row.get("company") or row.get("company_yfinance") or "",
                    row.get("time") or "",
                    importance,
                    row.get("market_cap_fmt") or "",
                    row.get("notes") or "",
                ),
                tags=(self._risk_tag(importance),),
            )

    def _render_headlines(
        self,
        target_rows: list[dict],
        industry_rows: list[dict],
        landmine_rows: list[dict] | None = None,
    ) -> None:
        self._clear_tree(self.headline_tree)
        if self.headline_tree is None:
            return
        for scope, rows, tag in (
            ("Ticker", target_rows, "scope_ticker"),
            ("Industry", industry_rows, "scope_industry"),
            ("Landmine", landmine_rows or [], "risk_high"),
        ):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                self.headline_tree.insert(
                    "",
                    tk.END,
                    values=(
                        scope,
                        row.get("title") or "",
                        row.get("source") or "",
                        row.get("query") or "",
                        row.get("published") or "",
                        row.get("url") or "",
                    ),
                    tags=(tag,),
                )

    def _render_sec(self, payload: dict) -> None:
        self._clear_tree(self.sec_tree)
        if self.sec_tree is None:
            return
        rows = payload.get("filings") if isinstance(payload, dict) else []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            risk = str(row.get("risk_classification") or "").upper()
            self.sec_tree.insert(
                "",
                tk.END,
                values=(
                    row.get("ticker") or "",
                    row.get("form") or "",
                    row.get("filing_date") or "",
                    risk,
                    ", ".join(row.get("matched_keywords") or []),
                    row.get("url") or "",
                ),
                tags=(self._risk_tag(risk),),
            )

    def export_markdown(self) -> None:
        if not isinstance(self.latest_lookup, dict):
            self.status_var.set("Run a ticker lookup before exporting markdown.")
            return
        ticker = str(self.latest_lookup.get("ticker") or "ticker").lower()
        report_date = str(self.latest_lookup.get("report_date") or datetime.now().date().isoformat())
        markdown = str(self.latest_lookup.get("markdown") or "").rstrip()
        if not markdown:
            self.status_var.set("No ticker lookup markdown available.")
            return
        path = self.output_dir / f"ticker_lookup_{ticker}_{report_date}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown + "\n", encoding="utf-8")
        self.status_var.set(f"Exported ticker lookup: {path}")

    def open_output_folder(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            open_path_in_file_manager(self.output_dir)
        except Exception as exc:
            messagebox.showerror("Open Output Folder", f"Could not open output folder:\n{self.output_dir}\n\n{exc}")
            return
        self.status_var.set(f"Opened output folder: {self.output_dir}")

    def _set_text(self, widget: scrolledtext.ScrolledText | None, text: str) -> None:
        if widget is None:
            return
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        _apply_text_highlights(widget, text)
        widget.configure(state="disabled")

    def _clear_tree(self, tree: ttk.Treeview | None) -> None:
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)

    def _sort_tree(self, tree: ttk.Treeview, column: str, reverse: bool) -> None:
        rows = [(str(tree.set(item, column)).lower(), item) for item in tree.get_children("")]
        rows.sort(reverse=reverse)
        for index, (_value, item) in enumerate(rows):
            tree.move(item, "", index)
        tree.heading(column, command=lambda: self._sort_tree(tree, column, not reverse))

    def _risk_tag(self, value: str) -> str:
        normalized = str(value or "").upper()
        if normalized in {"HIGH", "MEGA"}:
            return "risk_high"
        if normalized == "MEDIUM":
            return "risk_medium"
        return "risk_low"
