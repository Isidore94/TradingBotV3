from __future__ import annotations

import calendar
import threading
import tkinter as tk
from datetime import date, datetime, time as dt_time
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any
import zoneinfo

from journal_analytics import build_analytics_text, calendar_pnl_by_day
from journal_importers import (
    IBKR_CLIENT_ID_SETTING,
    IBKR_DEFAULT_CLIENT_ID,
    IBKR_ENABLED_SETTING,
    IBKR_HOST_SETTING,
    IBKR_PORT_SETTING,
    PACIFIC_TZ_NAME,
    QUESTRADE_REFRESH_TOKEN_SETTING,
    QuestradeImporter,
    manual_execution_from_fields,
    mask_secret,
    pacific_now,
    parse_csv_executions,
    resolve_ibkr_client_id,
)
from journal_runner import run_journal_import_for_date
from journal_store import JournalStore, REGIME_PRESETS
from project_paths import LOCAL_SETTINGS_FILE, get_local_setting, open_path_in_file_manager, save_local_setting


JOURNAL_SCHEDULER_ENABLED_SETTING = "journal_scheduler_enabled"
JOURNAL_SCHEDULER_LAST_RUN_SETTING = "journal_scheduler_last_run_date"
JOURNAL_AUTO_PULL_TIME = dt_time(hour=13, minute=0)

MONEY_COLUMNS = {"gross_pnl", "commission", "fees", "net_pnl", "pnl_usd"}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _safe_date(text: str | None = None) -> date:
    value = str(text or "").strip()
    if value:
        try:
            return datetime.fromisoformat(value[:10]).date()
        except ValueError:
            pass
    return pacific_now().date()


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{numeric:.{digits}f}"


def _fmt_money(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{numeric:,.2f}"


class JournalTab(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        text_bg: str = "#252525",
        text_fg: str = "#E0E0E0",
        compact_layout: bool = False,
    ) -> None:
        super().__init__(parent)
        self.text_bg = text_bg
        self.text_fg = text_fg
        self.compact_layout = bool(compact_layout)
        self.store = JournalStore()
        self.trade_row_map: dict[str, dict[str, Any]] = {}
        self.selected_trade_id = ""
        self.import_active = False
        self.current_month = _safe_date().replace(day=1)

        self.selected_date_var = tk.StringVar(value=_safe_date().isoformat())
        self.broker_filter_var = tk.StringVar(value="All")
        self.account_filter_var = tk.StringVar(value="All")
        self.status_var = tk.StringVar(value="Journal ready.")
        self.scheduler_enabled_var = tk.BooleanVar(
            value=_coerce_bool(get_local_setting(JOURNAL_SCHEDULER_ENABLED_SETTING, True), default=True)
        )
        self.scheduler_status_var = tk.StringVar()

        self.tag_var = tk.StringVar()
        self.notes_var = tk.StringVar()
        self.regime_date_var = tk.StringVar(value=self.selected_date_var.get())
        self.mid_regime_var = tk.StringVar()
        self.short_regime_var = tk.StringVar()
        self.intraday_regime_var = tk.StringVar()
        self.regime_notes_var = tk.StringVar()

        self.questrade_refresh_var = tk.StringVar(
            value=str(get_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, "") or "")
        )
        self.ibkr_enabled_var = tk.BooleanVar(
            value=_coerce_bool(get_local_setting(IBKR_ENABLED_SETTING, False), default=False)
        )
        self.ibkr_host_var = tk.StringVar(value=str(get_local_setting(IBKR_HOST_SETTING, "127.0.0.1") or "127.0.0.1"))
        self.ibkr_port_var = tk.StringVar(value=str(get_local_setting(IBKR_PORT_SETTING, 7496) or 7496))
        self.ibkr_client_id_var = tk.StringVar(
            value=str(resolve_ibkr_client_id(get_local_setting(IBKR_CLIENT_ID_SETTING, IBKR_DEFAULT_CLIENT_ID)))
        )

        self._build()
        self.refresh_all()
        self.after(1000, self._scheduler_tick)

    def _build(self) -> None:
        self._build_toolbar()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        self._build_daily_tab()
        self._build_calendar_tab()
        self._build_analytics_tab()
        self._build_regime_tab()
        self._build_imports_tab()
        ttk.Label(self, textvariable=self.status_var).pack(fill=tk.X, padx=10, pady=(0, 6))

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=10, pady=(8, 6))
        ttk.Label(toolbar, text="Date").pack(side=tk.LEFT)
        date_entry = ttk.Entry(toolbar, textvariable=self.selected_date_var, width=12)
        date_entry.pack(side=tk.LEFT, padx=(6, 8))
        date_entry.bind("<Return>", lambda _event: self.refresh_all())
        ttk.Button(toolbar, text="Today", command=self._select_today).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(toolbar, text="Broker").pack(side=tk.LEFT)
        self.broker_combo = ttk.Combobox(toolbar, textvariable=self.broker_filter_var, width=14, state="readonly")
        self.broker_combo.pack(side=tk.LEFT, padx=(6, 8))
        self.broker_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_all())
        ttk.Label(toolbar, text="Account").pack(side=tk.LEFT)
        self.account_combo = ttk.Combobox(toolbar, textvariable=self.account_filter_var, width=20, state="readonly")
        self.account_combo.pack(side=tk.LEFT, padx=(6, 10))
        self.account_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_all())
        ttk.Button(toolbar, text="Get Latest Trades", command=self.get_latest_trades).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Add Manual Trade", command=self.add_manual_trade).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Import CSV", command=self.import_csv).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT)
        ttk.Checkbutton(
            toolbar,
            text="13:00 Pacific Auto Pull",
            variable=self.scheduler_enabled_var,
            command=self._save_scheduler_setting,
        ).pack(side=tk.RIGHT, padx=(12, 0))
        ttk.Label(toolbar, textvariable=self.scheduler_status_var).pack(side=tk.RIGHT)

    def _build_daily_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Daily Journal")
        pane = tk.PanedWindow(frame, orient=tk.VERTICAL, sashwidth=8, showhandle=True)
        pane.pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.Frame(pane)
        columns = (
            "opened",
            "closed",
            "broker",
            "account",
            "symbol",
            "type",
            "direction",
            "status",
            "qty",
            "pnl",
            "currency",
            "tags",
        )
        self.trade_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        headings = {
            "opened": "Opened",
            "closed": "Closed",
            "broker": "Broker",
            "account": "Account",
            "symbol": "Symbol",
            "type": "Type",
            "direction": "Dir",
            "status": "Status",
            "qty": "Qty",
            "pnl": "Net PnL",
            "currency": "Ccy",
            "tags": "Setup Tags",
        }
        widths = {
            "opened": 130,
            "closed": 130,
            "broker": 90,
            "account": 120,
            "symbol": 110,
            "type": 70,
            "direction": 58,
            "status": 70,
            "qty": 72,
            "pnl": 90,
            "currency": 48,
            "tags": 360,
        }
        for col in columns:
            self.trade_table.heading(col, text=headings[col])
            self.trade_table.column(col, width=widths[col], anchor="w")
        trade_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.trade_table.yview)
        self.trade_table.configure(yscrollcommand=trade_scroll.set)
        self.trade_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trade_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.trade_table.bind("<<TreeviewSelect>>", self._on_trade_selected)

        detail_frame = ttk.Frame(pane)
        left = ttk.LabelFrame(detail_frame, text="Raw Fills")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        leg_columns = ("time", "side", "role", "qty", "price", "commission", "fees")
        self.leg_table = ttk.Treeview(left, columns=leg_columns, show="headings", height=7)
        leg_widths = {"time": 150, "side": 70, "role": 70, "qty": 70, "price": 80, "commission": 90, "fees": 70}
        for col in leg_columns:
            self.leg_table.heading(col, text=col)
            self.leg_table.column(col, width=leg_widths[col], anchor="w")
        self.leg_table.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        right = ttk.LabelFrame(detail_frame, text="Tag / Notes")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right.configure(width=520)
        edit = ttk.Frame(right)
        edit.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(edit, text="Tags").grid(row=0, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.tag_var, width=62).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(edit, text="Notes").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(edit, textvariable=self.notes_var, width=62).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0))
        edit.columnconfigure(1, weight=1)
        ttk.Button(right, text="Save Tags / Notes", command=self.save_selected_annotation).pack(anchor="e", padx=8, pady=(0, 6))
        self.candidate_text = tk.Text(right, height=5, wrap="word", bg=self.text_bg, fg=self.text_fg, font=("Courier New", 9))
        self.candidate_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        pane.add(table_frame, minsize=220)
        pane.add(detail_frame, minsize=170)

    def _build_calendar_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Calendar")
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(toolbar, text="<", width=4, command=lambda: self._shift_month(-1)).pack(side=tk.LEFT)
        self.month_label_var = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.month_label_var, width=24, anchor="center").pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text=">", width=4, command=lambda: self._shift_month(1)).pack(side=tk.LEFT)
        self.calendar_frame = ttk.Frame(frame)
        self.calendar_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _build_analytics_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Analytics")
        ttk.Button(frame, text="Refresh Analytics", command=self.refresh_analytics).pack(anchor="w", padx=8, pady=8)
        self.analytics_text = scrolledtext.ScrolledText(
            frame,
            wrap="word",
            bg=self.text_bg,
            fg=self.text_fg,
            font=("Courier New", 10),
        )
        self.analytics_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _build_regime_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Regimes")
        form = ttk.LabelFrame(frame, text="Market Regime")
        form.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(form, text="Date").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        ttk.Entry(form, textvariable=self.regime_date_var, width=14).grid(row=0, column=1, sticky="w", pady=(8, 4))
        ttk.Button(form, text="Load", command=self.load_regime_form).grid(row=0, column=2, padx=6, pady=(8, 4))
        ttk.Label(form, text="Mid Term").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(form, textvariable=self.mid_regime_var, values=REGIME_PRESETS["mid_term"], width=30).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(form, text="Short Term").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(form, textvariable=self.short_regime_var, values=REGIME_PRESETS["short_term"], width=30).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(form, text="Intraday").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(form, textvariable=self.intraday_regime_var, values=REGIME_PRESETS["intraday"], width=30).grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(form, text="Notes").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(form, textvariable=self.regime_notes_var, width=70).grid(row=4, column=1, columnspan=3, sticky="ew", pady=4)
        ttk.Button(form, text="Save Regime", command=self.save_regime_form).grid(row=5, column=1, sticky="e", pady=(6, 10))
        form.columnconfigure(1, weight=1)
        help_text = (
            "Mid-term and short-term regimes carry forward until changed. "
            "Intraday regime is specific to the selected date."
        )
        ttk.Label(frame, text=help_text, wraplength=900).pack(fill=tk.X, padx=12, pady=(0, 8))

    def _build_imports_tab(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Imports / Settings")

        qt = ttk.LabelFrame(frame, text="Questrade")
        qt.pack(fill=tk.X, padx=10, pady=(10, 6))
        row = ttk.Frame(qt)
        row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(row, text="Refresh Token").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.questrade_refresh_var, show="*", width=64).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row, text="Save", command=self.save_questrade_settings).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Clear", command=self.clear_questrade_settings).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Refresh Token Now", command=self.refresh_questrade_token).pack(side=tk.LEFT)
        self.questrade_status_text = tk.Text(qt, height=4, wrap="word", bg=self.text_bg, fg=self.text_fg)
        self.questrade_status_text.pack(fill=tk.X, padx=8, pady=(0, 8))

        ib = ttk.LabelFrame(frame, text="IBKR")
        ib.pack(fill=tk.X, padx=10, pady=6)
        ib_row = ttk.Frame(ib)
        ib_row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Checkbutton(ib_row, text="Enable IBKR import", variable=self.ibkr_enabled_var, command=self.save_ibkr_settings).pack(side=tk.LEFT)
        ttk.Label(ib_row, text="Host").pack(side=tk.LEFT, padx=(14, 4))
        ttk.Entry(ib_row, textvariable=self.ibkr_host_var, width=16).pack(side=tk.LEFT)
        ttk.Label(ib_row, text="Port").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Entry(ib_row, textvariable=self.ibkr_port_var, width=7).pack(side=tk.LEFT)
        ttk.Label(ib_row, text="Journal Client ID").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Entry(ib_row, textvariable=self.ibkr_client_id_var, width=7).pack(side=tk.LEFT)
        ttk.Button(ib_row, text="Save", command=self.save_ibkr_settings).pack(side=tk.LEFT, padx=(10, 0))

        runs = ttk.LabelFrame(frame, text="Import Runs")
        runs.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))
        toolbar = ttk.Frame(runs)
        toolbar.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(toolbar, text="Refresh", command=self.refresh_import_status).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Open Settings File", command=lambda: open_path_in_file_manager(LOCAL_SETTINGS_FILE)).pack(side=tk.LEFT, padx=(8, 0))
        self.import_runs_text = scrolledtext.ScrolledText(runs, height=10, wrap="word", bg=self.text_bg, fg=self.text_fg, font=("Courier New", 10))
        self.import_runs_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _select_today(self) -> None:
        self.selected_date_var.set(pacific_now().date().isoformat())
        self.regime_date_var.set(self.selected_date_var.get())
        self.refresh_all()

    def _save_scheduler_setting(self) -> None:
        save_local_setting(JOURNAL_SCHEDULER_ENABLED_SETTING, bool(self.scheduler_enabled_var.get()))
        self._refresh_scheduler_status()

    def _scheduler_tick(self) -> None:
        try:
            self._refresh_scheduler_status()
            if bool(self.scheduler_enabled_var.get()) and not self.import_active:
                now = datetime.now(zoneinfo.ZoneInfo(PACIFIC_TZ_NAME))
                today = now.date().isoformat()
                last_run = str(get_local_setting(JOURNAL_SCHEDULER_LAST_RUN_SETTING, "") or "")
                if now.time() >= JOURNAL_AUTO_PULL_TIME and last_run != today:
                    self.selected_date_var.set(today)
                    save_local_setting(JOURNAL_SCHEDULER_LAST_RUN_SETTING, today)
                    self.get_latest_trades(trigger="scheduler")
        finally:
            try:
                if self.winfo_exists():
                    self.after(60_000, self._scheduler_tick)
            except Exception:
                pass

    def _refresh_scheduler_status(self) -> None:
        state = "on" if bool(self.scheduler_enabled_var.get()) else "off"
        last_run = str(get_local_setting(JOURNAL_SCHEDULER_LAST_RUN_SETTING, "") or "never")
        self.scheduler_status_var.set(f"Auto pull {state}; last {last_run}")

    def refresh_all(self) -> None:
        self._refresh_filters()
        self.refresh_daily()
        self.refresh_calendar()
        self.refresh_analytics()
        self.load_regime_form()
        self.refresh_import_status()
        self._refresh_scheduler_status()

    def _refresh_filters(self) -> None:
        brokers = ["All"] + self.store.distinct_values("broker")
        accounts = ["All"] + self.store.distinct_values("account_label")
        self.broker_combo.configure(values=brokers)
        self.account_combo.configure(values=accounts)
        if self.broker_filter_var.get() not in brokers:
            self.broker_filter_var.set("All")
        if self.account_filter_var.get() not in accounts:
            self.account_filter_var.set("All")

    def refresh_daily(self) -> None:
        self.trade_row_map = {}
        for item in self.trade_table.get_children():
            self.trade_table.delete(item)
        trades = self.store.list_trades(
            trade_date=self.selected_date_var.get(),
            broker=self.broker_filter_var.get(),
            account=self.account_filter_var.get(),
        )
        for trade in trades:
            values = (
                str(trade.get("opened_at") or "")[:19],
                str(trade.get("closed_at") or "")[:19],
                trade.get("broker", ""),
                trade.get("account_label") or trade.get("account_number", ""),
                trade.get("symbol", ""),
                trade.get("security_type", ""),
                trade.get("direction", ""),
                trade.get("status", ""),
                _fmt_float(trade.get("quantity_opened"), 2),
                _fmt_money(trade.get("net_pnl")),
                trade.get("currency", ""),
                trade.get("display_tags", ""),
            )
            item_id = self.trade_table.insert("", tk.END, values=values)
            self.trade_row_map[item_id] = trade
        self.status_var.set(f"Loaded {len(trades)} journal trades for {self.selected_date_var.get()}.")

    def _on_trade_selected(self, _event=None) -> None:
        selected = self.trade_table.selection()
        trade = self.trade_row_map.get(selected[0]) if selected else None
        self.selected_trade_id = str(trade.get("trade_id") or "") if trade else ""
        self.tag_var.set(str((trade or {}).get("setup_tags") or (trade or {}).get("auto_tag_summary") or ""))
        self.notes_var.set(str((trade or {}).get("notes") or ""))
        self._refresh_legs()
        self._refresh_candidates()

    def _refresh_legs(self) -> None:
        for item in self.leg_table.get_children():
            self.leg_table.delete(item)
        if not self.selected_trade_id:
            return
        for leg in self.store.list_trade_legs(self.selected_trade_id):
            self.leg_table.insert(
                "",
                tk.END,
                values=(
                    str(leg.get("timestamp") or "")[:19],
                    leg.get("side", ""),
                    leg.get("role", ""),
                    _fmt_float(leg.get("quantity"), 2),
                    _fmt_float(leg.get("price"), 4),
                    _fmt_money(leg.get("commission")),
                    _fmt_money(leg.get("fees")),
                ),
            )

    def _refresh_candidates(self) -> None:
        lines = []
        if self.selected_trade_id:
            for row in self.store.list_auto_tag_candidates(self.selected_trade_id):
                lines.append(
                    f"{float(row.get('confidence', 0.0) or 0.0):.0%}  {row.get('tag')}  "
                    f"[{row.get('source')}] {row.get('rationale')}"
                )
        self._set_text(self.candidate_text, "\n".join(lines) if lines else "No auto-tag candidates yet.")

    def save_selected_annotation(self) -> None:
        if not self.selected_trade_id:
            messagebox.showinfo("Journal", "Select a trade first.")
            return
        trade = self.store.get_trade(self.selected_trade_id)
        if not trade:
            return
        tags = self.tag_var.get().strip()
        notes = self.notes_var.get().strip()
        previous_tags = str(trade.get("setup_tags") or "").strip()
        self.store.save_trade_annotation(self.selected_trade_id, setup_tags=tags, notes=notes)
        if tags and tags != previous_tags:
            self.store.record_tag_corrections(trade, tags)
            self.store.refresh_auto_tags()
        self.refresh_daily()
        self.status_var.set("Trade annotation saved.")

    def refresh_calendar(self) -> None:
        for child in self.calendar_frame.winfo_children():
            child.destroy()
        month_start = self.current_month
        self.month_label_var.set(month_start.strftime("%B %Y"))
        month_trades = self.store.list_trades()
        pnl_by_day = calendar_pnl_by_day(month_trades)
        for col, label in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
            ttk.Label(self.calendar_frame, text=label, anchor="center").grid(row=0, column=col, sticky="ew", padx=2, pady=2)
            self.calendar_frame.columnconfigure(col, weight=1)
        for row_idx, week in enumerate(calendar.monthcalendar(month_start.year, month_start.month), start=1):
            for col_idx, day_num in enumerate(week):
                if day_num == 0:
                    ttk.Label(self.calendar_frame, text="").grid(row=row_idx, column=col_idx, sticky="nsew", padx=2, pady=2)
                    continue
                day = date(month_start.year, month_start.month, day_num)
                pnl = float(pnl_by_day.get(day.isoformat(), 0.0) or 0.0)
                text = f"{day_num}\n{pnl:,.0f}" if abs(pnl) > 0.005 else f"{day_num}\n"
                bg = "#25452D" if pnl > 0 else "#4A2528" if pnl < 0 else "#2F3338"
                fg = "#E8F4EA" if pnl >= 0 else "#FFECEC"
                btn = tk.Button(
                    self.calendar_frame,
                    text=text,
                    bg=bg,
                    fg=fg,
                    height=4,
                    relief=tk.FLAT,
                    command=lambda d=day: self._select_calendar_day(d),
                )
                btn.grid(row=row_idx, column=col_idx, sticky="nsew", padx=2, pady=2)
            self.calendar_frame.rowconfigure(row_idx, weight=1)

    def _select_calendar_day(self, selected: date) -> None:
        self.selected_date_var.set(selected.isoformat())
        self.regime_date_var.set(selected.isoformat())
        self.notebook.select(0)
        self.refresh_all()

    def _shift_month(self, months: int) -> None:
        year = self.current_month.year + ((self.current_month.month - 1 + months) // 12)
        month = ((self.current_month.month - 1 + months) % 12) + 1
        self.current_month = date(year, month, 1)
        self.refresh_calendar()

    def refresh_analytics(self) -> None:
        trades = self.store.list_trades()
        self._set_text(self.analytics_text, build_analytics_text(trades) if trades else "No journal trades yet.")

    def load_regime_form(self) -> None:
        if not self.regime_date_var.get().strip():
            self.regime_date_var.set(self.selected_date_var.get())
        regime = self.store.get_regime_for_date(self.regime_date_var.get())
        self.mid_regime_var.set(regime.get("mid_term_regime", ""))
        self.short_regime_var.set(regime.get("short_term_regime", ""))
        self.intraday_regime_var.set(regime.get("intraday_regime", ""))
        self.regime_notes_var.set(regime.get("regime_notes", ""))

    def save_regime_form(self) -> None:
        target_date = self.regime_date_var.get().strip() or self.selected_date_var.get()
        self.store.upsert_regime(
            target_date,
            mid_term_regime=self.mid_regime_var.get(),
            short_term_regime=self.short_regime_var.get(),
            intraday_regime=self.intraday_regime_var.get(),
            notes=self.regime_notes_var.get(),
        )
        self.refresh_all()
        self.status_var.set(f"Regime saved for {target_date}.")

    def refresh_import_status(self) -> None:
        importer = QuestradeImporter()
        qt_lines = importer.status_lines()
        saved_refresh = str(get_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, "") or "")
        if saved_refresh:
            qt_lines.append(f"Saved local refresh token: {mask_secret(saved_refresh)}")
        self._set_text(self.questrade_status_text, "\n".join(qt_lines))
        lines = []
        for row in self.store.list_import_runs():
            lines.append(
                f"{row.get('import_run_id')}: {row.get('source')} {row.get('status')} "
                f"started={row.get('started_at')} finished={row.get('finished_at') or ''} "
                f"count={row.get('imported_executions')} {row.get('message') or ''}"
            )
        self._set_text(self.import_runs_text, "\n".join(lines) if lines else "No import runs yet.")

    def save_questrade_settings(self) -> None:
        save_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, self.questrade_refresh_var.get().strip())
        self.refresh_import_status()
        self.status_var.set("Questrade settings saved.")

    def clear_questrade_settings(self) -> None:
        self.questrade_refresh_var.set("")
        save_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, "")
        self.refresh_import_status()
        self.status_var.set("Questrade refresh token cleared.")

    def refresh_questrade_token(self) -> None:
        self.save_questrade_settings()

        def task() -> None:
            run_id = self.store.start_import_run("QUESTRADE_TOKEN")
            try:
                QuestradeImporter().refresh_access_token()
            except Exception as exc:
                self.store.finish_import_run(run_id, status="FAILED", imported_executions=0, message=str(exc))
                self.after(0, lambda: messagebox.showerror("Questrade", f"Token refresh failed:\n\n{exc}"))
            else:
                self.store.finish_import_run(run_id, status="OK", imported_executions=0, message="Token refreshed.")
            self.after(0, self.refresh_import_status)

        threading.Thread(target=task, daemon=True).start()

    def save_ibkr_settings(self) -> None:
        save_local_setting(IBKR_ENABLED_SETTING, bool(self.ibkr_enabled_var.get()))
        save_local_setting(IBKR_HOST_SETTING, self.ibkr_host_var.get().strip() or "127.0.0.1")
        save_local_setting(IBKR_PORT_SETTING, int(self.ibkr_port_var.get() or 7496))
        client_id = resolve_ibkr_client_id(self.ibkr_client_id_var.get())
        self.ibkr_client_id_var.set(str(client_id))
        save_local_setting(IBKR_CLIENT_ID_SETTING, client_id)
        self.status_var.set("IBKR settings saved.")

    def get_latest_trades(self, trigger: str = "manual") -> None:
        if self.import_active:
            self.status_var.set("Journal import is already running.")
            return
        self.save_ibkr_settings()
        self.import_active = True
        target_date = _safe_date(self.selected_date_var.get())
        self.status_var.set(f"Importing latest trades for {target_date.isoformat()}...")

        def task() -> None:
            try:
                summary = run_journal_import_for_date(
                    target_date,
                    trigger=trigger,
                    store=self.store,
                    include_ibkr=bool(self.ibkr_enabled_var.get()),
                    ibkr_host=self.ibkr_host_var.get().strip() or "127.0.0.1",
                    ibkr_port=int(self.ibkr_port_var.get() or 7496),
                    ibkr_client_id=resolve_ibkr_client_id(self.ibkr_client_id_var.get()),
                )
            except Exception as exc:
                summary = {
                    "status": "FAILED",
                    "total_imported": 0,
                    "messages": [str(exc)],
                }
            total_imported = int(summary.get("total_imported") or 0)
            messages = list(summary.get("messages") or [])

            def finish() -> None:
                self.import_active = False
                self.refresh_all()
                label = "Import finished with errors" if summary.get("status") == "FAILED" else "Import complete"
                self.status_var.set(f"{label} ({total_imported} executions): " + "; ".join(messages))

            self.after(0, finish)

        threading.Thread(target=task, daemon=True).start()

    def add_manual_trade(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Add Manual Execution")
        dialog.transient(self)
        dialog.grab_set()
        fields = {
            "broker": tk.StringVar(value="MANUAL"),
            "account_number": tk.StringVar(value="MANUAL"),
            "symbol": tk.StringVar(),
            "side": tk.StringVar(value="BUY"),
            "quantity": tk.StringVar(),
            "price": tk.StringVar(),
            "currency": tk.StringVar(value="USD"),
            "security_type": tk.StringVar(value="STK"),
            "timestamp": tk.StringVar(value=f"{self.selected_date_var.get()}T12:00:00"),
            "commission": tk.StringVar(value="0"),
            "fees": tk.StringVar(value="0"),
        }
        for idx, (key, var) in enumerate(fields.items()):
            ttk.Label(dialog, text=key.replace("_", " ").title()).grid(row=idx, column=0, sticky="w", padx=8, pady=4)
            if key == "side":
                ttk.Combobox(dialog, textvariable=var, values=("BUY", "SELL"), width=20, state="readonly").grid(row=idx, column=1, sticky="ew", padx=8, pady=4)
            else:
                ttk.Entry(dialog, textvariable=var, width=34).grid(row=idx, column=1, sticky="ew", padx=8, pady=4)
        dialog.columnconfigure(1, weight=1)

        def save() -> None:
            try:
                execution = manual_execution_from_fields({key: var.get() for key, var in fields.items()})
                if not execution.symbol:
                    raise ValueError("Symbol is required.")
                if execution.quantity <= 0 or execution.price <= 0:
                    raise ValueError("Quantity and price must be positive.")
                self.store.upsert_accounts(
                    execution.broker,
                    [
                        {
                            "account_number": execution.account_number,
                            "account_label": execution.account_label,
                            "account_type": execution.account_type,
                            "currency": execution.currency,
                        }
                    ],
                )
                self.store.upsert_executions([execution])
                self.store.rebuild_trades()
            except Exception as exc:
                messagebox.showerror("Manual Trade", str(exc), parent=dialog)
                return
            dialog.destroy()
            self.refresh_all()
            self.status_var.set("Manual execution added.")

        buttons = ttk.Frame(dialog)
        buttons.grid(row=len(fields), column=0, columnspan=2, sticky="e", padx=8, pady=8)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Save", command=save).pack(side=tk.RIGHT, padx=(0, 8))

    def import_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Import journal executions CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if not path:
            return
        run_id = self.store.start_import_run("CSV")
        try:
            executions = parse_csv_executions(Path(path))
            self.store.upsert_executions(executions)
            accounts = [
                {
                    "account_number": item.account_number,
                    "account_label": item.account_label,
                    "account_type": item.account_type,
                    "currency": item.currency,
                }
                for item in executions
            ]
            self.store.upsert_accounts("MANUAL", accounts)
            trade_count = self.store.rebuild_trades()
            self.store.finish_import_run(run_id, status="OK", imported_executions=len(executions), message=f"Grouped trades={trade_count}")
        except Exception as exc:
            self.store.finish_import_run(run_id, status="FAILED", imported_executions=0, message=str(exc))
            messagebox.showerror("CSV Import", str(exc))
            return
        self.refresh_all()
        self.status_var.set(f"Imported {len(executions)} executions from CSV.")

    def export_csv(self) -> None:
        try:
            path = self.store.export_trades_csv()
        except Exception as exc:
            messagebox.showerror("Journal Export", str(exc))
            return
        self.status_var.set(f"Journal export written to {path}.")
        try:
            open_path_in_file_manager(path.parent)
        except Exception:
            pass

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)
