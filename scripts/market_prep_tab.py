from __future__ import annotations

import tkinter as tk
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from project_paths import OUTPUT_DIR, ROOT_DIR, open_path_in_file_manager

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from market_prep import get_market_prep_logger, load_market_prep_config
from market_prep.orchestrator import MarketPrepOrchestrator


MARKET_PREP_PLACEHOLDER_TEXT = "Market Prep tab loaded. Phase 1 skeleton ready."


class MarketPrepTab:
    """Phase 1 skeleton for the top-level Market Prep workspace."""

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
        resolved_paths = self.config.resolved_paths()
        self.output_dir = resolved_paths.get("output_dir", self.output_dir)
        self.logger = get_market_prep_logger()
        self.orchestrator = MarketPrepOrchestrator()
        self.status_var = tk.StringVar(value="Ready")
        self.container = ttk.Frame(parent)
        self.display_text: scrolledtext.ScrolledText | None = None
        self.buttons: dict[str, ttk.Button] = {}
        self.background_task_active = False
        self.latest_report: dict | None = None
        self._build_layout()
        self.set_display_text(MARKET_PREP_PLACEHOLDER_TEXT)
        self.logger.info("Market Prep loaded")

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self.container)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 8))

        buttons = (
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
        )
        for label, command in buttons:
            button = ttk.Button(toolbar, text=label, command=command)
            button.pack(side=tk.LEFT, padx=(0, 8))
            self.buttons[label] = button
        self._refresh_forexfactory_button()

        body = ttk.Frame(self.container)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        self.display_text = scrolledtext.ScrolledText(
            body,
            wrap=tk.WORD,
            font=("Courier New", 10),
            bg=self.text_bg,
            fg=self.text_fg,
            insertbackground=self.text_fg,
        )
        self.display_text.pack(fill=tk.BOTH, expand=True)

        status = ttk.Label(self.container, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill=tk.X, padx=10, pady=(0, 10))

    def set_display_text(self, text: str) -> None:
        if self.display_text is None:
            return
        self.display_text.configure(state="normal")
        self.display_text.delete("1.0", tk.END)
        self.display_text.insert("1.0", text)
        self.display_text.configure(state="disabled")

    def _set_placeholder_status(self, action: str) -> None:
        result = self.orchestrator.run_placeholder(action)
        timestamp = result.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status_var.set(f"{action} placeholder ready.")
        report = str(result.get("report") or MARKET_PREP_PLACEHOLDER_TEXT)
        self.set_display_text(f"{report}\n\n{action}: placeholder handler invoked at {timestamp}.")

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
        self.set_display_text(loading_text)

        def worker() -> None:
            try:
                result = worker_func()
            except Exception as exc:
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
                self.set_display_text(report)
                self.status_var.set(done_status if self.latest_report else f"{button_label} finished with errors.")

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
        self.set_display_text(message)
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
            self.set_display_text(message)
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


class TickerLookupTabStub:
    """Dormant placeholder for a future Ticker Lookup workspace."""

    PLACEHOLDER_TEXT = "Ticker Lookup tab stub loaded. Future phase placeholder."

    def __init__(
        self,
        parent: tk.Misc,
        *,
        text_bg: str = "#252525",
        text_fg: str = "#E0E0E0",
    ):
        self.container = ttk.Frame(parent)
        display = scrolledtext.ScrolledText(
            self.container,
            wrap=tk.WORD,
            font=("Courier New", 10),
            bg=text_bg,
            fg=text_fg,
            insertbackground=text_fg,
        )
        display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        display.insert("1.0", self.PLACEHOLDER_TEXT)
        display.configure(state="disabled")

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)
