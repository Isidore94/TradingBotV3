from __future__ import annotations

import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from project_paths import OUTPUT_DIR, ROOT_DIR, open_path_in_file_manager
from gui_text_highlighter import set_highlighted_text

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from market_prep import get_market_prep_logger, load_market_prep_config
from market_prep.config_loader import (
    get_market_prep_openai_key_source,
    save_llm_summary_settings,
    save_market_prep_openai_api_key,
)
from market_prep.orchestrator import MarketPrepOrchestrator
from market_prep.report_builder import build_catalyst_clock
from market_prep.services.ticker_lookup_service import lookup_ticker_context


MARKET_PREP_PLACEHOLDER_TEXT = "Market Prep tab loaded. Phase 1 skeleton ready."
MARKET_PREP_NAV_ITEMS = (
    ("Overview", "overview"),
    ("Catalyst Clock", "catalyst"),
    ("Watchlist Risk", "watchlist"),
    ("Earnings", "earnings"),
    ("Macro/Fed/Treasury", "macro"),
    ("News/SEC", "news_sec"),
    ("AI Summary", "ai_summary"),
    ("Raw Markdown", "raw"),
)
HIGH_PRIORITY_VALUES = {"HIGH", "MEGA"}
MEDIUM_PRIORITY_VALUES = {"MEDIUM"}
TEXT_FONT = ("Courier New", 10)
CATALYST_FILTER_OPTIONS = (
    ("all", "All"),
    ("medium_plus", "Medium+"),
    ("high", "High/Mega"),
)


class TickerLookupTab:
    """Single-symbol event, earnings, filing, and industry-news lookup."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        output_dir: Path = OUTPUT_DIR,
        text_bg: str = "#252525",
        text_fg: str = "#E0E0E0",
    ):
        self.parent = parent
        self.output_dir = Path(output_dir)
        self.text_bg = text_bg
        self.text_fg = text_fg
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
        self.raw_text: scrolledtext.ScrolledText | None = None
        self._build_layout()

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self.container)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 8))
        ttk.Label(toolbar, text="Ticker").pack(side=tk.LEFT, padx=(0, 6))
        entry = ttk.Entry(toolbar, textvariable=self.ticker_var, width=14)
        entry.pack(side=tk.LEFT, padx=(0, 8))
        entry.bind("<Return>", lambda _event: self.run_lookup())
        ttk.Label(toolbar, text="Days").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Spinbox(toolbar, from_=7, to=180, increment=1, textvariable=self.days_var, width=6).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        self.lookup_button = ttk.Button(toolbar, text="Lookup", command=self.run_lookup)
        self.lookup_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Export Markdown", command=self.export_markdown).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Open Output Folder", command=self.open_output_folder).pack(side=tk.LEFT)

        summary = ttk.LabelFrame(self.container, text="Lookup Summary")
        summary.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(summary, textvariable=self.summary_var, justify="left", wraplength=1500).pack(
            fill=tk.X,
            padx=8,
            pady=(6, 8),
        )

        self.notebook = ttk.Notebook(self.container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self._build_overview_tab()
        self._build_earnings_tab()
        self._build_news_tab()
        self._build_sec_tab()
        self._build_raw_tab()

        status = ttk.Label(self.container, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _build_overview_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Overview")
        self.overview_text = self._make_text(frame)
        self._set_text(self.overview_text, "Ticker Lookup is ready. Enter a symbol such as TSM, NVDA, AAPL, or AMD.")

    def _build_earnings_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Earnings / Events")
        target_frame = ttk.LabelFrame(frame, text="Ticker Earnings")
        target_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
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
        peer_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
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
        widget.configure(state="disabled")
        return widget

    def _make_tree(
        self,
        parent: tk.Misc,
        columns: tuple[str, ...],
        headings: dict[str, tuple[str, int, str]],
    ) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        for column in columns:
            label, width, anchor = headings[column]
            tree.heading(column, text=label, command=lambda col=column, t=tree: self._sort_tree(t, col, False))
            tree.column(column, width=width, anchor=anchor, stretch=column in {"title", "notes", "url"})
        tree.tag_configure("risk_high", foreground="#FF9A9A")
        tree.tag_configure("risk_medium", foreground="#F0C76E")
        tree.tag_configure("risk_low", foreground="#A9B4C0")
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
        self._set_text(self.overview_text, "Fetching ticker metadata, earnings, filings, and Google News RSS context...")

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
        peers = payload.get("peer_tickers") if isinstance(payload.get("peer_tickers"), list) else []
        statuses = payload.get("source_status") if isinstance(payload.get("source_status"), list) else []
        self.summary_var.set(
            f"{ticker} | {metadata.get('company_name') or 'Company unknown'} | "
            f"{metadata.get('sector') or 'sector n/a'} / {metadata.get('industry') or 'industry n/a'} | "
            f"Market cap: {metadata.get('market_cap_fmt') or 'n/a'} | "
            f"Peers: {', '.join(peers) or 'None'} | "
            f"Earnings: {len(target_earnings)} ticker, {len(peer_earnings)} peer | "
            f"Headlines: {len(target_headlines)} ticker, {len(industry_headlines)} industry"
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
            "",
            "Source Status",
            "-" * 80,
        ]
        overview.extend(f"- {status}" for status in statuses)
        self._set_text(self.overview_text, "\n".join(overview).rstrip())
        self._set_text(self.raw_text, str(payload.get("markdown") or ""))
        self._render_earnings_tree(self.earnings_tree, target_earnings)
        self._render_earnings_tree(self.peer_tree, peer_earnings)
        self._render_headlines(target_headlines, industry_headlines)
        self._render_sec(payload.get("sec_filings") if isinstance(payload.get("sec_filings"), dict) else {})

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

    def _render_headlines(self, target_rows: list[dict], industry_rows: list[dict]) -> None:
        self._clear_tree(self.headline_tree)
        if self.headline_tree is None:
            return
        for scope, rows in (("Ticker", target_rows), ("Industry", industry_rows)):
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
        set_highlighted_text(widget, text, state_after="disabled")

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
