from __future__ import annotations

import importlib
import sys

_legacy = sys.modules.get("bounce_bot_lib.legacy")
if _legacy is None:
    candidate = sys.modules.get("__main__")
    if getattr(candidate, "__package__", None) == "bounce_bot_lib":
        _legacy = candidate
if _legacy is None:
    _legacy = importlib.import_module("bounce_bot_lib.legacy")

globals().update(
    {
        name: value
        for name, value in vars(_legacy).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def build_environment_focus_copy_text(snapshot):
    sections = snapshot.get("environment_highlights", []) if isinstance(snapshot, dict) else []
    label = "Environment Focus Lists"
    if isinstance(snapshot, dict):
        env_label = snapshot.get("market_environment_label", "Environment")
        label = f"{env_label} Focus Lists"

    lines = [label]
    if not sections:
        lines.extend(["", "None"])
        return "\n".join(lines)

    for section in sections:
        title = str(section.get("title", "Section")).strip() or "Section"
        seen = set()
        symbols = []
        for row in section.get("rows", []):
            symbol = str(row.get("symbol") or "").strip().upper()
            if not symbol:
                text = str(row.get("text") or "").strip()
                if text:
                    first_token = text.split()[0].strip(",")
                    if ENVIRONMENT_FOCUS_SYMBOL_RE.fullmatch(first_token):
                        symbol = first_token
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)

        lines.append("")
        lines.append(title)
        lines.append(", ".join(symbols) if symbols else "None")

    return "\n".join(lines).strip()


def copy_text_to_clipboard(widget: tk.Misc, text: str) -> None:
    widget.clipboard_clear()
    widget.clipboard_append(text)
    widget.update_idletasks()


def create_rrs_confirmed_panel(parent, bot_instance, dark_grey="#2E2E2E", text_color="#E0E0E0"):
    """Create the BounceBot RS/RW confirmed screen (industry + sector) inside `parent`."""
    rrs_container = tk.Frame(parent, bg=dark_grey)

    rrs_controls = tk.Frame(rrs_container, padx=10, pady=10, bg=dark_grey)
    rrs_controls.pack(fill=tk.X)

    rrs_status_var = tk.StringVar(value="RRS ready")
    tk.Label(rrs_controls, textvariable=rrs_status_var, fg=text_color, bg=dark_grey).pack(side=tk.LEFT, padx=(0, 10))

    rrs_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)

    def on_rrs_threshold_change(*_):
        bot_instance.set_rrs_threshold(rrs_threshold_var.get())

    rrs_threshold_var.trace_add("write", on_rrs_threshold_change)

    tk.Scale(
        rrs_controls,
        from_=0.0,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        label="RRS Sensitivity",
        variable=rrs_threshold_var,
        length=220,
        bg=dark_grey,
        fg=text_color,
        highlightthickness=0,
    ).pack(side=tk.LEFT, padx=(0, 10))

    timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)

    def on_timeframe_change() -> None:
        bot_instance.set_rrs_timeframe(timeframe_var.get())

    for key in ("5m", "15m", "30m", "1h"):
        label = RRS_TIMEFRAMES[key]["label"]
        tk.Radiobutton(
            rrs_controls,
            text=label,
            variable=timeframe_var,
            value=key,
            indicatoron=0,
            command=on_timeframe_change,
            padx=6,
            pady=2,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        ).pack(side=tk.LEFT, padx=2)

    env_selection_var = tk.StringVar(value=bot_instance.get_market_environment())
    env_label_var = tk.StringVar(
        value=f"Environment: {MARKET_ENVIRONMENTS.get(env_selection_var.get(), {}).get('label', env_selection_var.get())}"
    )

    def on_environment_change():
        selected = env_selection_var.get()
        env_label_var.set(
            f"Environment: {MARKET_ENVIRONMENTS.get(selected, {}).get('label', selected)}"
        )
        bot_instance.set_market_environment(selected)

    env_mode_frame = tk.Frame(rrs_container, bg=dark_grey, pady=4)
    env_mode_frame.pack(fill=tk.X, padx=10)
    tk.Label(env_mode_frame, textvariable=env_label_var, fg=text_color, bg=dark_grey).pack(side=tk.LEFT)
    env_button_frame = tk.Frame(env_mode_frame, bg=dark_grey)
    env_button_frame.pack(side=tk.RIGHT)
    for key, info in MARKET_ENVIRONMENTS.items():
        tk.Radiobutton(
            env_button_frame,
            text=info["label"],
            variable=env_selection_var,
            value=key,
            indicatoron=0,
            command=on_environment_change,
            padx=6,
            pady=2,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        ).pack(side=tk.LEFT, padx=2)

    rrs_frame = tk.Frame(rrs_container, padx=10, pady=10, bg=dark_grey)
    rrs_frame.pack(fill=tk.BOTH, expand=True)

    vertical_pane = tk.PanedWindow(
        rrs_frame,
        orient=tk.VERTICAL,
        sashrelief=tk.RAISED,
        sashwidth=10,
        showhandle=True,
        bg=dark_grey,
    )
    vertical_pane.pack(fill=tk.BOTH, expand=True)

    env_focus_pane = tk.PanedWindow(
        vertical_pane,
        orient=tk.HORIZONTAL,
        sashrelief=tk.RAISED,
        sashwidth=10,
        showhandle=True,
        bg=dark_grey,
    )
    vertical_pane.add(env_focus_pane, minsize=150)

    env_focus_frame = tk.LabelFrame(env_focus_pane, text="Environment Focus", bg=dark_grey, fg=text_color)
    env_focus_text = scrolledtext.ScrolledText(
        env_focus_frame,
        wrap=tk.NONE,
        width=80,
        height=10,
        font=('Courier', 10),
        state='disabled',
        bg=dark_grey,
        fg=text_color,
    )
    env_focus_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    env_focus_text.tag_config("rrs_hdr", foreground="#BD93F9", font=('Courier', 11, 'bold'))
    env_focus_text.tag_config("rrs_rs", foreground="#50FA7B", font=('Courier', 11, 'bold'))
    env_focus_text.tag_config("rrs_rw", foreground="#FF5555", font=('Courier', 11, 'bold'))
    env_focus_pane.add(env_focus_frame, stretch="always")

    env_copy_frame = tk.LabelFrame(env_focus_pane, text="Environment Focus Lists", bg=dark_grey, fg=text_color)
    env_copy_toolbar = tk.Frame(env_copy_frame, bg=dark_grey)
    env_copy_toolbar.pack(fill=tk.X, padx=4, pady=(4, 0))

    env_copy_text = scrolledtext.ScrolledText(
        env_copy_frame,
        wrap=tk.WORD,
        width=52,
        height=10,
        font=('Courier', 10),
        state='disabled',
        bg=dark_grey,
        fg=text_color,
    )

    def copy_env_focus_lists():
        text = env_copy_text.get("1.0", tk.END).strip()
        if not text:
            rrs_status_var.set("Environment focus lists: nothing to copy.")
            return
        copy_text_to_clipboard(env_copy_text, text)
        rrs_status_var.set("Copied environment focus lists to clipboard.")

    tk.Button(
        env_copy_toolbar,
        text="Copy",
        command=copy_env_focus_lists,
        relief=tk.RAISED,
        padx=10,
        bg=dark_grey,
        fg=text_color,
    ).pack(side=tk.LEFT)

    env_copy_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    env_focus_pane.add(env_copy_frame, stretch="always")

    notebook_host = tk.Frame(vertical_pane, bg=dark_grey)
    vertical_pane.add(notebook_host, stretch="always")

    notebook = ttk.Notebook(notebook_host)
    notebook.pack(fill=tk.BOTH, expand=True)

    def _make_text(parent_widget, width, height):
        widget = scrolledtext.ScrolledText(
            parent_widget,
            wrap=tk.NONE,
            width=width,
            height=height,
            font=("Courier", 10),
            state="disabled",
            bg=dark_grey,
            fg=text_color,
        )
        widget.tag_config("rrs_hdr", foreground="#BD93F9", font=("Courier", 11, "bold"))
        widget.tag_config("rrs_rs", foreground="#50FA7B", font=("Courier", 11, "bold"))
        widget.tag_config("rrs_rw", foreground="#FF5555", font=("Courier", 11, "bold"))
        return widget

    def _create_split_signal_tab(parent_widget, title, threshold_var=None, timeframe_var=None):
        tab = ttk.Frame(parent_widget)
        parent_widget.add(tab, text=title)

        if threshold_var is not None or timeframe_var is not None:
            controls = tk.Frame(tab, bg=dark_grey, padx=6, pady=6)
            controls.pack(fill=tk.X)
            if threshold_var is not None:
                tk.Scale(
                    controls,
                    from_=0.0,
                    to=5.0,
                    resolution=0.1,
                    orient=tk.HORIZONTAL,
                    label="Sensitivity",
                    variable=threshold_var,
                    length=180,
                    bg=dark_grey,
                    fg=text_color,
                    highlightthickness=0,
                ).pack(side=tk.LEFT, padx=(0, 10))
            if timeframe_var is not None:
                tf_frame = tk.Frame(controls, bg=dark_grey)
                tf_frame.pack(side=tk.LEFT)
                for key in ("5m", "15m", "30m", "1h"):
                    tk.Radiobutton(
                        tf_frame,
                        text=key,
                        variable=timeframe_var,
                        value=key,
                        indicatoron=0,
                        padx=4,
                        pady=1,
                        bg=dark_grey,
                        fg=text_color,
                        selectcolor="#444444",
                        activebackground="#444444",
                        activeforeground=text_color,
                    ).pack(side=tk.LEFT, padx=1)

        split = tk.PanedWindow(
            tab,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=10,
            showhandle=True,
            bg=dark_grey,
        )
        split.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        strong_frame = tk.LabelFrame(split, text="Strong", bg=dark_grey, fg=text_color)
        weak_frame = tk.LabelFrame(split, text="Weak", bg=dark_grey, fg=text_color)
        strong_text = _make_text(strong_frame, width=40, height=18)
        weak_text = _make_text(weak_frame, width=40, height=18)
        strong_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        weak_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        split.add(strong_frame, stretch="always")
        split.add(weak_frame, stretch="always")
        return {
            "tab": tab,
            "strong_text": strong_text,
            "weak_text": weak_text,
        }

    industry_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)
    sector_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)

    industry_timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)
    sector_timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)

    spy_view = _create_split_signal_tab(notebook, "VS SPY")
    industry_view = _create_split_signal_tab(notebook, "Industry Ref", industry_threshold_var, industry_timeframe_var)
    sector_view = _create_split_signal_tab(notebook, "Sector", sector_threshold_var, sector_timeframe_var)

    groups_tab = ttk.Frame(notebook)
    notebook.add(groups_tab, text="Top Industries/Sectors")
    groups_split = tk.PanedWindow(
        groups_tab,
        orient=tk.HORIZONTAL,
        sashrelief=tk.RAISED,
        sashwidth=10,
        showhandle=True,
        bg=dark_grey,
    )
    groups_split.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
    sectors_frame = tk.LabelFrame(groups_split, text="Sectors", bg=dark_grey, fg=text_color)
    industries_frame = tk.LabelFrame(groups_split, text="Industries", bg=dark_grey, fg=text_color)
    sectors_text = _make_text(sectors_frame, width=44, height=18)
    industries_text = _make_text(industries_frame, width=44, height=18)
    sectors_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    industries_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    groups_split.add(sectors_frame, stretch="always")
    groups_split.add(industries_frame, stretch="always")

    def render_rrs_snapshot(snapshot):
        threshold = snapshot.get("threshold", RRS_DEFAULT_THRESHOLD)
        timeframe_key = snapshot.get("timeframe_key", "5m")
        timeframe_label = RRS_TIMEFRAMES.get(timeframe_key, {}).get("label", timeframe_key)
        results = snapshot.get("results", [])
        sector_results = snapshot.get("results_sector", [])
        industry_results = snapshot.get("results_industry", [])
        group_strength = snapshot.get("group_strength", {})
        timestamp = snapshot.get("timestamp", datetime.now())
        env_label_var.set(f"Environment: {snapshot.get('market_environment_label', 'Environment')}")

        def render_split_table(view, title, rows, local_threshold, selected_tf):
            filtered = [r for r in rows if abs(r[2]) >= local_threshold]
            strong_rows = [r for r in filtered if r[0] == "RS"]
            weak_rows = [r for r in filtered if r[0] == "RW"]
            for widget, heading, subset, tag in (
                (view["strong_text"], "Strong", strong_rows, "rrs_rs"),
                (view["weak_text"], "Weak", weak_rows, "rrs_rw"),
            ):
                widget.config(state="normal")
                widget.delete("1.0", tk.END)
                widget.insert(
                    tk.END,
                    f"{title} {heading}  TF:{selected_tf}  Threshold:{local_threshold:.2f}\n",
                    "rrs_hdr",
                )
                widget.insert(tk.END, "SYMBOL  SIDE  RRS    POWER\n")
                widget.insert(tk.END, "--------------------------\n")
                if not subset:
                    widget.insert(tk.END, "No symbols flagged.\n")
                for signal, symbol, rrs_value, power in subset:
                    line = f"{symbol:<6}  {signal:<4}  {rrs_value:+.2f}  {power if power is not None else 0:>6.2f}\n"
                    widget.insert(tk.END, line, tag)
                widget.config(state="disabled")
                widget.see("1.0")

        render_split_table(spy_view, "RS/RW vs SPY", results, threshold, timeframe_key)
        render_split_table(
            industry_view,
            "RS/RW vs Industry Ref",
            industry_results,
            industry_threshold_var.get(),
            industry_timeframe_var.get(),
        )
        render_split_table(
            sector_view,
            "RS/RW vs Sector",
            sector_results,
            sector_threshold_var.get(),
            sector_timeframe_var.get(),
        )

        def render_group_column(widget, label, key_name):
            widget.config(state="normal")
            widget.delete("1.0", tk.END)
            widget.insert(
                tk.END,
                f"{label}  Last scan: {timestamp.strftime('%H:%M:%S')}  Base TF:{timeframe_label}\n",
                "rrs_hdr",
            )
            for tf in ("M5", "H1", "D1"):
                payload = group_strength.get(tf, {})
                items = payload.get(key_name, [])
                widget.insert(tk.END, f"\n[{tf}] Strongest\n", "rrs_hdr")
                if not items:
                    widget.insert(tk.END, "  No data\n")
                for item in items[:SCAN_EXTREME_COUNT]:
                    widget.insert(
                        tk.END,
                        f"  + {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n",
                        "rrs_rs",
                    )
                widget.insert(tk.END, f"[{tf}] Weakest\n", "rrs_hdr")
                if not items:
                    widget.insert(tk.END, "  No data\n")
                for item in list(reversed(items[-SCAN_EXTREME_COUNT:])):
                    widget.insert(
                        tk.END,
                        f"  - {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n",
                        "rrs_rw",
                    )
            widget.config(state="disabled")
            widget.see("1.0")

        render_group_column(sectors_text, "Sectors", "sectors")
        render_group_column(industries_text, "Industries", "industries")

        env_focus_text.config(state='normal')
        env_focus_text.delete("1.0", tk.END)
        env_highlights = snapshot.get("environment_highlights", [])
        env_focus_text.insert(tk.END, f"{snapshot.get('market_environment_label', 'Environment')} Focus\n", "rrs_hdr")
        for section in env_highlights:
            env_focus_text.insert(tk.END, f"\n{section.get('title', 'Section')}\n", "rrs_hdr")
            rows = section.get("rows", [])
            if not rows:
                env_focus_text.insert(tk.END, "  None\n")
            for row in rows:
                env_focus_text.insert(tk.END, f"  {row.get('text', '')}\n", row.get("tag", "rrs_rs"))
        env_focus_text.config(state='disabled')
        env_focus_text.see("1.0")

        env_copy_text.config(state='normal')
        env_copy_text.delete("1.0", tk.END)
        env_copy_text.insert(tk.END, build_environment_focus_copy_text(snapshot))
        env_copy_text.config(state='disabled')
        env_copy_text.see("1.0")

    return {
        "container": rrs_container,
        "rrs_status_var": rrs_status_var,
        "render_rrs_snapshot": render_rrs_snapshot,
    }


def choose_gui_mode():
    preferred_mode = str(get_local_setting("bounce_bot_gui_mode", "full") or "full").strip().lower()
    if preferred_mode not in {"full", "lightweight"}:
        preferred_mode = "full"
    selection = {"mode": preferred_mode}
    picker = tk.Tk()
    picker.title("BounceBot Mode")
    picker.geometry("360x160")
    picker.configure(bg="#2E2E2E")
    picker.resizable(False, False)

    tk.Label(
        picker,
        text="Choose BounceBot startup mode",
        bg="#2E2E2E",
        fg="#E0E0E0",
        font=("Arial", 12, "bold"),
    ).pack(pady=(18, 10))

    tk.Label(
        picker,
        text=(
            "Full mode keeps the RS/RW panels.\n"
            "Lightweight mode keeps alerts and core bounce controls only.\n"
            f"Default on this computer: {preferred_mode.title()}"
        ),
        bg="#2E2E2E",
        fg="#E0E0E0",
        justify=tk.CENTER,
    ).pack(pady=(0, 14))

    button_row = tk.Frame(picker, bg="#2E2E2E")
    button_row.pack()

    def select_mode(mode):
        selection["mode"] = mode
        save_local_setting("bounce_bot_gui_mode", mode)
        picker.destroy()

    tk.Button(button_row, text="Full", width=12, command=lambda: select_mode("full")).pack(side=tk.LEFT, padx=8)
    tk.Button(button_row, text="Lightweight", width=12, command=lambda: select_mode("lightweight")).pack(side=tk.LEFT, padx=8)

    picker.protocol("WM_DELETE_WINDOW", lambda: select_mode("full"))
    picker.mainloop()
    return selection["mode"]


def prompt_change_home_folder(root, cleanup_callback=None):
    details = get_tracker_storage_details()
    current_dir = Path(details["data_dir"])
    selected = filedialog.askdirectory(
        title="Choose home folder",
        initialdir=str(current_dir if current_dir.exists() else Path.home()),
        mustexist=False,
    )
    if not selected:
        return

    target = save_tracker_storage_dir(selected)
    restart_now = messagebox.askyesno(
        "Home Folder Saved",
        "Saved this computer's home folder.\n\n"
        f"Folder: {target}\n\n"
        "Restart BounceBot now so it starts using the new home folder?",
    )
    if restart_now:
        if cleanup_callback is not None:
            cleanup_callback()
        root.destroy()
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _save_bounce_feedback_from_widget(text_area, feedback_context, rating, reason, feedback_callback, source):
    context = _normalize_bounce_feedback_context(feedback_context)
    if callable(feedback_callback):
        feedback_callback(context, rating, reason, source)
    else:
        record_bounce_feedback(context, rating, reason, source=source)


def _open_bounce_feedback_reason_window(text_area, feedback_context, feedback_callback, source):
    context = _normalize_bounce_feedback_context(feedback_context)
    parent = text_area.winfo_toplevel()
    dialog = tk.Toplevel(parent)
    dialog.title("Bounce Feedback")
    dialog.configure(bg="#2E2E2E")
    dialog.transient(parent)
    dialog.grab_set()

    symbol = context.get("symbol") or "Bounce"
    direction = context.get("direction", "").upper()
    bounce_types = context.get("bounce_types") or "unknown levels"
    header = f"{symbol} {direction} | {bounce_types}"
    tk.Label(
        dialog,
        text=header,
        bg="#2E2E2E",
        fg="#E0E0E0",
        font=("Arial", 10, "bold"),
        wraplength=520,
        justify=tk.LEFT,
    ).pack(fill=tk.X, padx=12, pady=(12, 6))
    tk.Label(
        dialog,
        text="What is wrong with this bounce?",
        bg="#2E2E2E",
        fg="#E0E0E0",
        justify=tk.LEFT,
    ).pack(anchor="w", padx=12, pady=(0, 4))
    reason_text = tk.Text(
        dialog,
        width=64,
        height=7,
        wrap=tk.WORD,
        bg="#252525",
        fg="#E0E0E0",
        insertbackground="#E0E0E0",
    )
    reason_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

    button_row = tk.Frame(dialog, bg="#2E2E2E")
    button_row.pack(fill=tk.X, padx=12, pady=(0, 12))

    def save_issue():
        reason = reason_text.get("1.0", tk.END).strip()
        if not reason:
            messagebox.showerror("Missing Reason", "Enter a short reason before saving this as an issue.")
            return
        _save_bounce_feedback_from_widget(
            text_area,
            context,
            "issue",
            reason,
            feedback_callback,
            source,
        )
        dialog.destroy()

    tk.Button(button_row, text="Save Issue", command=save_issue, padx=10).pack(side=tk.LEFT)
    tk.Button(button_row, text="Cancel", command=dialog.destroy, padx=10).pack(side=tk.LEFT, padx=(8, 0))
    reason_text.focus_set()


def _handle_bounce_issue_click(text_area, feedback_context, feedback_callback, source):
    _open_bounce_feedback_reason_window(text_area, feedback_context, feedback_callback, source)


def _insert_bounce_feedback_controls(text_area, feedback_context, feedback_callback, source):
    seq = getattr(append_alert_message, "_feedback_seq", 0) + 1
    append_alert_message._feedback_seq = seq
    issue_tag = f"bounce_feedback_issue_{seq}"

    text_area.insert(tk.END, "!", (issue_tag, "bounce_feedback_issue"))
    text_area.insert(tk.END, "  ")

    text_area.tag_config("bounce_feedback_issue", foreground="#FF5555", font=("Courier", 10, "bold"))
    text_area.tag_bind(
        issue_tag,
        "<Button-1>",
        lambda _event, ctx=dict(feedback_context): _handle_bounce_issue_click(
            text_area,
            ctx,
            feedback_callback,
            source,
        ),
    )
    text_area.tag_bind(issue_tag, "<Enter>", lambda _event: text_area.config(cursor="hand2"))
    text_area.tag_bind(issue_tag, "<Leave>", lambda _event: text_area.config(cursor=""))


def append_alert_message(
    text_area,
    msg,
    tag,
    timestamp,
    feedback_callback=None,
    feedback_source="bounce_bot_gui",
):
    display_msg, feedback_context = _normalize_alert_message_payload(msg)
    if not feedback_context and "Bounce confirmed" in display_msg:
        feedback_context = _normalize_bounce_feedback_context({"alert_message": display_msg})
    if feedback_context:
        feedback_context["alert_time"] = timestamp
    feedback_available = bool(feedback_context) and "Bounce confirmed" in display_msg
    prefix_inserted = False

    def insert_prefix(prefix_tag=tag):
        nonlocal prefix_inserted
        if prefix_inserted:
            return
        if feedback_available:
            _insert_bounce_feedback_controls(
                text_area,
                feedback_context,
                feedback_callback,
                feedback_source,
            )
        text_area.insert(tk.END, f"{timestamp} - ", prefix_tag)
        prefix_inserted = True

    if display_msg.startswith("MASTER_AVWAP_D1_FLAG:"):
        insert_prefix(tag)
        text_area.insert(tk.END, f"{display_msg}\n", tag)
    elif display_msg.startswith("MASTER_AVWAP_FAVORITE_BOUNCE:"):
        insert_prefix(tag)
        text_area.insert(tk.END, f"{display_msg}\n", tag)
    elif "Bounce confirmed" in display_msg:
        parts = display_msg.split(":", 1)
        if len(parts) == 2:
            symbol = parts[0].strip()
            rest = ":" + parts[1]
            if "(long)" in rest:
                insert_prefix(tag)
                text_area.insert(tk.END, symbol, "pink_symbol")
                text_area.insert(tk.END, rest + "\n", "green")
            elif "(short)" in rest:
                insert_prefix(tag)
                text_area.insert(tk.END, symbol, "orange_symbol")
                text_area.insert(tk.END, rest + "\n", "red")
            else:
                insert_prefix(tag)
                text_area.insert(tk.END, f"{display_msg}\n", tag)
        else:
            insert_prefix(tag)
            text_area.insert(tk.END, f"{display_msg}\n", tag)
    elif "Price approaching levels" in display_msg:
        parts = display_msg.split(":", 1)
        if len(parts) == 2:
            symbol = parts[0].strip()
            rest = ":" + parts[1]
            if "(long)" in rest:
                insert_prefix(tag)
                text_area.insert(tk.END, symbol, "pink_symbol")
                text_area.insert(tk.END, rest + "\n", "approaching_green")
            elif "(short)" in rest:
                insert_prefix(tag)
                text_area.insert(tk.END, symbol, "orange_symbol")
                text_area.insert(tk.END, rest + "\n", "approaching_red")
            else:
                insert_prefix(tag)
                text_area.insert(tk.END, f"{display_msg}\n", tag)
        else:
            insert_prefix(tag)
            text_area.insert(tk.END, f"{display_msg}\n", tag)
    else:
        insert_prefix(tag)
        text_area.insert(tk.END, f"{display_msg}\n", tag)


def configure_alert_tags(text_area, font_size=12):
    text_area.tag_config("green", foreground="#50FA7B", font=("Courier", font_size))
    text_area.tag_config("red", foreground="#FF5555", font=("Courier", font_size))
    text_area.tag_config("pink_symbol", foreground="#FF79C6", font=("Courier", font_size, "bold"))
    text_area.tag_config("orange_symbol", foreground="#FFB86C", font=("Courier", font_size, "bold"))
    text_area.tag_config("blue", foreground="#8BE9FD", font=("Courier", font_size))
    text_area.tag_config("master_avwap_favorite_long", foreground="#00E5FF", font=("Courier", font_size, "bold"))
    text_area.tag_config("master_avwap_favorite_short", foreground="#FFD166", font=("Courier", font_size, "bold"))
    text_area.tag_config("master_avwap_focus_long", foreground="#7DF9FF", font=("Courier", font_size, "bold"))
    text_area.tag_config("master_avwap_focus_short", foreground="#FFB000", font=("Courier", font_size, "bold"))
    text_area.tag_config("d1_flag_long", foreground="#7DF9FF", font=("Courier", font_size, "bold"))
    text_area.tag_config("d1_flag_short", foreground="#FFB000", font=("Courier", font_size, "bold"))
    text_area.tag_config("d1_flag_watch", foreground="#BD93F9", font=("Courier", font_size, "bold"))
    text_area.tag_config("candle_line", foreground="#BD93F9", overstrike=1)
    text_area.tag_config("approaching", foreground="#FF79C6", font=("Courier", font_size))
    text_area.tag_config("approaching_green", foreground="#50FA7B", font=("Courier", font_size))
    text_area.tag_config("approaching_red", foreground="#FF5555", font=("Courier", font_size))


def start_lightweight_gui():
    bounce_queue = queue.Queue()
    dark_grey = "#2E2E2E"
    text_color = "#E0E0E0"
    input_grey = "#252525"
    panel_grey = "#3A3A3A"

    def gui_callback(message, tag):
        if tag.startswith("rrs"):
            return
        if tag == "approaching" or tag.startswith("approaching_"):
            return
        if tag == "blue" and "removed from" in str(message):
            return
        bounce_queue.put((message, tag))

    bot_instance = run_bot_with_gui(gui_callback, start_scanning_enabled=False)

    root = tk.Tk()
    root.title("BounceBot Lightweight")
    root.geometry("980x680")
    root.configure(background=dark_grey)

    container = tk.Frame(root, padx=10, pady=10, bg=dark_grey)
    container.pack(fill=tk.BOTH, expand=True)

    header = tk.Frame(container, bg=dark_grey)
    header.pack(fill=tk.X, pady=(0, 8))

    status_var = tk.StringVar(value="scanning paused")
    connection_var = tk.StringVar(value="IB: connected")
    tk.Label(header, text="BounceBot Lightweight", bg=dark_grey, fg=text_color, font=("Arial", 11, "bold")).pack(side=tk.LEFT)
    tk.Label(header, textvariable=connection_var, bg=dark_grey, fg=text_color).pack(side=tk.LEFT, padx=(12, 0))
    tk.Label(header, textvariable=status_var, bg=dark_grey, fg=text_color).pack(side=tk.LEFT, padx=(12, 0))

    def disconnect_bot():
        nonlocal bot_instance
        try:
            bot_instance.disconnect()
        except Exception:
            pass
        bot_instance = None
        connection_var.set("IB: disconnected")
        status_var.set("alerts paused")

    def restart_bot():
        nonlocal bot_instance
        disconnect_bot()
        connection_var.set("IB: reconnecting")
        status_var.set("starting...")
        bot_instance = run_bot_with_gui(gui_callback, start_scanning_enabled=False)
        rrs_threshold_var.set(bot_instance.rrs_threshold)
        timeframe_var.set(bot_instance.rrs_timeframe_key)
        env_selection_var.set(bot_instance.get_market_environment())
        connection_var.set("IB: connected")
        status_var.set("scanning paused")

    def start_scanning():
        if bot_instance is None:
            return
        bot_instance.set_scanning_enabled(True)
        status_var.set("scanning enabled")

    def stop_scanning():
        if bot_instance is None:
            return
        bot_instance.set_scanning_enabled(False)
        status_var.set("scanning paused")

    def switch_mode(new_mode):
        save_local_setting("bounce_bot_gui_mode", new_mode)
        disconnect_bot()
        root.destroy()
        start_gui(mode=new_mode)

    tk.Button(
        header,
        text="Change Home Folder",
        command=lambda: prompt_change_home_folder(root, cleanup_callback=disconnect_bot),
        relief=tk.RAISED,
        padx=10,
        bg=panel_grey,
        fg=text_color,
    ).pack(side=tk.RIGHT)
    tk.Button(header, text="Switch to Full", command=lambda: switch_mode("full"), relief=tk.RAISED, padx=10, bg=panel_grey, fg=text_color).pack(side=tk.RIGHT)
    tk.Button(header, text="Reconnect", command=restart_bot, relief=tk.RAISED, padx=10, bg=panel_grey, fg=text_color).pack(side=tk.RIGHT)
    stop_scanning_button = tk.Button(header, text="Stop Scanning", command=stop_scanning, relief=tk.RAISED, padx=10, bg=panel_grey, fg=text_color)
    stop_scanning_button.pack(side=tk.RIGHT, padx=(0, 8))
    start_scanning_button = tk.Button(header, text="Start Scanning", command=start_scanning, relief=tk.RAISED, padx=10, bg=panel_grey, fg=text_color)
    start_scanning_button.pack(side=tk.RIGHT, padx=(0, 8))
    tk.Button(header, text="Disconnect", command=disconnect_bot, relief=tk.RAISED, padx=10, bg=panel_grey, fg=text_color).pack(side=tk.RIGHT, padx=(0, 8))

    def clear_alerts():
        for widget in (text_area, d1_flags_text):
            widget.config(state="normal")
            widget.delete("1.0", tk.END)
            widget.config(state="disabled")

    tk.Button(header, text="Clear", command=clear_alerts, relief=tk.RAISED, padx=10, bg=panel_grey, fg=text_color).pack(side=tk.RIGHT, padx=(0, 8))

    controls = tk.Frame(container, bg=dark_grey)
    controls.pack(fill=tk.X, pady=(0, 8))

    tk.Label(controls, text="RRS Sensitivity", bg=dark_grey, fg=text_color).pack(side=tk.LEFT)
    rrs_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)

    def on_rrs_threshold_change(*_):
        bot_instance.set_rrs_threshold(rrs_threshold_var.get())

    rrs_threshold_var.trace_add("write", on_rrs_threshold_change)

    tk.Scale(
        controls,
        from_=0.0,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        variable=rrs_threshold_var,
        length=180,
        bg=dark_grey,
        fg=text_color,
        highlightthickness=0,
    ).pack(side=tk.LEFT, padx=(8, 14))

    tk.Label(controls, text="Timeframe", bg=dark_grey, fg=text_color).pack(side=tk.LEFT)
    timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)

    def on_timeframe_change():
        selected = timeframe_var.get()
        bot_instance.set_rrs_timeframe(selected)
        status_var.set(f"RRS timeframe set to {RRS_TIMEFRAMES[selected]['label']}")

    for key in ("5m", "15m", "30m", "1h"):
        tk.Radiobutton(
            controls,
            text=RRS_TIMEFRAMES[key]["label"],
            variable=timeframe_var,
            value=key,
            indicatoron=0,
            command=on_timeframe_change,
            padx=6,
            pady=2,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        ).pack(side=tk.LEFT, padx=2)

    env_frame = tk.Frame(container, bg=dark_grey)
    env_frame.pack(fill=tk.X, pady=(0, 8))
    tk.Label(env_frame, text="Market Environment", bg=dark_grey, fg=text_color).pack(side=tk.LEFT, padx=(0, 8))
    env_selection_var = tk.StringVar(value=bot_instance.get_market_environment())

    def on_environment_change():
        selected = env_selection_var.get()
        bot_instance.set_market_environment(selected)
        status_var.set(f"Environment: {MARKET_ENVIRONMENTS[selected]['label']}")

    for key, info in MARKET_ENVIRONMENTS.items():
        tk.Radiobutton(
            env_frame,
            text=info["label"],
            variable=env_selection_var,
            value=key,
            indicatoron=0,
            command=on_environment_change,
            padx=8,
            pady=3,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        ).pack(side=tk.LEFT, padx=2)

    alerts_frame = tk.PanedWindow(
        container,
        orient=tk.HORIZONTAL,
        sashrelief=tk.RAISED,
        sashwidth=8,
        showhandle=True,
        bg=dark_grey,
    )
    alerts_frame.pack(fill=tk.BOTH, expand=True)
    bounce_frame = tk.LabelFrame(alerts_frame, text="BounceBot Alerts", bg=dark_grey, fg=text_color)
    text_area = scrolledtext.ScrolledText(
        bounce_frame,
        wrap=tk.WORD,
        width=80,
        height=30,
        font=("Courier", 11),
        state="disabled",
        bg=input_grey,
        fg=text_color,
        insertbackground=text_color,
    )
    text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    alerts_frame.add(bounce_frame, stretch="always")

    d1_frame = tk.LabelFrame(alerts_frame, text="D1 Master AVWAP Events", bg=dark_grey, fg=text_color)
    d1_flags_text = scrolledtext.ScrolledText(
        d1_frame,
        wrap=tk.WORD,
        width=58,
        height=30,
        font=("Courier", 10),
        state="disabled",
        bg=input_grey,
        fg=text_color,
        insertbackground=text_color,
    )
    d1_flags_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    alerts_frame.add(d1_frame, stretch="always")
    configure_alert_tags(text_area, font_size=11)
    configure_alert_tags(d1_flags_text, font_size=10)

    def process_bounce_queue():
        scanning_active = bool(bot_instance is not None and bot_instance.is_scanning_enabled())
        start_scanning_button.config(state=(tk.DISABLED if scanning_active else tk.NORMAL))
        stop_scanning_button.config(state=(tk.NORMAL if scanning_active else tk.DISABLED))
        while True:
            try:
                msg, tag = bounce_queue.get_nowait()
            except queue.Empty:
                break
            target_area = d1_flags_text if str(tag).startswith("d1_flag") else text_area
            target_area.config(state="normal")
            append_alert_message(
                target_area,
                msg,
                str(tag),
                datetime.now().strftime("%H:%M:%S"),
                feedback_callback=lambda ctx, rating, reason, source: (
                    record_bounce_feedback(ctx, rating, reason, source=source),
                    status_var.set(
                        "Saved bounce feedback: "
                        f"{ctx.get('symbol', 'bounce')} -> {rating}"
                    ),
                ),
                feedback_source="bounce_bot_lightweight_gui",
            )
            target_area.config(state="disabled")
            target_area.see(tk.END)
            root.update()
        root.after(150, process_bounce_queue)

    def on_closing():
        disconnect_bot()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    process_bounce_queue()
    root.mainloop()


def start_gui(mode="prompt"):
    if mode == "prompt":
        mode = str(get_local_setting("bounce_bot_gui_mode", "full") or "full").strip().lower()
        if mode not in {"full", "lightweight"}:
            mode = "full"
    if mode in {"full", "lightweight"}:
        save_local_setting("bounce_bot_gui_mode", mode)
    if mode == "lightweight":
        start_lightweight_gui()
        return

    bounce_queue = queue.Queue()
    rrs_queue = queue.Queue()
    # Replace light_grey with dark_grey
    dark_grey = "#2E2E2E"  # Dark grey color code
    text_color = "#E0E0E0"  # Light text color for dark background

    def gui_callback(message, tag):
        if tag.startswith("rrs"):
            rrs_queue.put((message, tag))
        elif tag == "candle_line":
            bounce_queue.put((message, tag))
        elif tag == "blue" and "removed from" in str(message):
            pass
        else:
            bounce_queue.put((message, tag))

    bot_instance = run_bot_with_gui(gui_callback, start_scanning_enabled=False)

    # Main bounce alerts window
    root = tk.Tk()
    root.title("BounceBot Alerts")
    root.geometry("800x600")
    root.configure(background=dark_grey)

    frame = tk.Frame(root, padx=10, pady=10, bg=dark_grey)
    frame.pack(fill=tk.BOTH, expand=True)

    header = tk.Frame(frame, bg=dark_grey)
    header.pack(fill=tk.X, pady=(0, 8))
    tk.Label(header, text="BounceBot Full", bg=dark_grey, fg=text_color, font=("Arial", 11, "bold")).pack(side=tk.LEFT)

    def switch_mode(new_mode):
        save_local_setting("bounce_bot_gui_mode", new_mode)
        try:
            bot_instance.disconnect()
        except Exception:
            pass
        root.destroy()
        start_gui(mode=new_mode)

    def start_scanning():
        bot_instance.set_scanning_enabled(True)

    def stop_scanning():
        bot_instance.set_scanning_enabled(False)

    tk.Button(
        header,
        text="Switch to Lightweight",
        command=lambda: switch_mode("lightweight"),
        relief=tk.RAISED,
        padx=10,
        bg="#3A3A3A",
        fg=text_color,
    ).pack(side=tk.RIGHT)
    tk.Button(
        header,
        text="Change Home Folder",
        command=lambda: prompt_change_home_folder(root, cleanup_callback=lambda: bot_instance.disconnect()),
        relief=tk.RAISED,
        padx=10,
        bg="#3A3A3A",
        fg=text_color,
    ).pack(side=tk.RIGHT, padx=(0, 8))

    main_notebook = ttk.Notebook(frame)
    main_notebook.pack(fill=tk.BOTH, expand=True)

    day_tab = tk.Frame(main_notebook, bg=dark_grey)
    tracker_tab = tk.Frame(main_notebook, bg=dark_grey)
    main_notebook.add(day_tab, text="Day Trading")
    main_notebook.add(tracker_tab, text="Bounce Setup Tracker")

    content_pane = tk.PanedWindow(day_tab, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg=dark_grey)
    content_pane.pack(fill=tk.BOTH, expand=True)

    alerts_frame = tk.Frame(content_pane, bg=dark_grey)
    alerts_split = tk.PanedWindow(
        alerts_frame,
        orient=tk.HORIZONTAL,
        sashrelief=tk.RAISED,
        sashwidth=8,
        showhandle=True,
        bg=dark_grey,
    )
    alerts_split.pack(fill=tk.BOTH, expand=True)

    confirmed_frame = tk.LabelFrame(alerts_split, text="Confirmed Bounces", bg=dark_grey, fg=text_color)
    text_area = scrolledtext.ScrolledText(
        confirmed_frame,
        wrap=tk.WORD,
        width=68,
        height=30,
        font=('Courier', 12),
        state='disabled',
        bg=dark_grey,
        fg=text_color  # Add text color
    )
    text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    alerts_split.add(confirmed_frame, stretch="always")

    d1_flags_frame = tk.LabelFrame(alerts_split, text="D1 Master AVWAP Flags", bg=dark_grey, fg=text_color)
    d1_flags_text = scrolledtext.ScrolledText(
        d1_flags_frame,
        wrap=tk.WORD,
        width=58,
        height=30,
        font=('Courier', 11),
        state='disabled',
        bg=dark_grey,
        fg=text_color,
    )
    d1_flags_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    alerts_split.add(d1_flags_frame, stretch="always")
    content_pane.add(alerts_frame, stretch="always")

    configure_alert_tags(text_area, font_size=12)
    configure_alert_tags(d1_flags_text, font_size=11)

    # Create RRS panel inside main window
    rrs_container = tk.Frame(content_pane, bg=dark_grey)
    content_pane.add(rrs_container, stretch="always")

    rrs_controls = tk.Frame(rrs_container, padx=10, pady=10, bg=dark_grey)
    rrs_controls.pack(fill=tk.X)

    rrs_status_var = tk.StringVar(value="RRS ready")
    rrs_status_label = tk.Label(rrs_controls, textvariable=rrs_status_var, fg=text_color, bg=dark_grey)
    rrs_status_label.pack(side=tk.LEFT, padx=(0, 10))

    rrs_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)

    def on_rrs_threshold_change(*_):
        bot_instance.set_rrs_threshold(rrs_threshold_var.get())

    rrs_threshold_var.trace_add("write", on_rrs_threshold_change)

    rrs_scale = tk.Scale(
        rrs_controls,
        from_=0.0,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        label="RRS Sensitivity",
        variable=rrs_threshold_var,
        length=220,
        bg=dark_grey,
        fg=text_color,
        highlightthickness=0,
    )
    rrs_scale.pack(side=tk.LEFT, padx=(0, 10))

    timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)

    def on_timeframe_change() -> None:
        bot_instance.set_rrs_timeframe(timeframe_var.get())

    for key in ("5m", "15m", "30m", "1h"):
        label = RRS_TIMEFRAMES[key]["label"]
        btn = tk.Radiobutton(
            rrs_controls,
            text=label,
            variable=timeframe_var,
            value=key,
            indicatoron=0,
            command=on_timeframe_change,
            padx=6,
            pady=2,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        )
        btn.pack(side=tk.LEFT, padx=2)

    env_selection_var = tk.StringVar(value=bot_instance.get_market_environment())
    env_label_var = tk.StringVar(
        value=f"Environment: {MARKET_ENVIRONMENTS.get(env_selection_var.get(), {}).get('label', env_selection_var.get())}"
    )

    def on_environment_change():
        selected = env_selection_var.get()
        env_label_var.set(
            f"Environment: {MARKET_ENVIRONMENTS.get(selected, {}).get('label', selected)}"
        )
        bot_instance.set_market_environment(selected)

    env_mode_frame = tk.Frame(rrs_container, bg=dark_grey, pady=4)
    env_mode_frame.pack(fill=tk.X, padx=10)
    tk.Label(env_mode_frame, textvariable=env_label_var, fg=text_color, bg=dark_grey).pack(side=tk.LEFT)
    env_button_frame = tk.Frame(env_mode_frame, bg=dark_grey)
    env_button_frame.pack(side=tk.RIGHT)
    for key, info in MARKET_ENVIRONMENTS.items():
        tk.Radiobutton(
            env_button_frame,
            text=info["label"],
            variable=env_selection_var,
            value=key,
            indicatoron=0,
            command=on_environment_change,
            padx=6,
            pady=2,
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        ).pack(side=tk.LEFT, padx=2)

    rrs_frame = tk.Frame(rrs_container, padx=10, pady=10, bg=dark_grey)
    rrs_frame.pack(fill=tk.BOTH, expand=True)

    rrs_main_text = scrolledtext.ScrolledText(
        rrs_frame,
        wrap=tk.NONE,
        width=80,
        height=12,
        font=('Courier', 11),
        state='disabled',
        bg=dark_grey,
        fg=text_color
    )
    rrs_main_text.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

    env_focus_row = tk.PanedWindow(
        rrs_frame,
        orient=tk.HORIZONTAL,
        sashrelief=tk.RAISED,
        sashwidth=10,
        showhandle=True,
        bg=dark_grey,
    )
    env_focus_row.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

    env_focus_frame = tk.LabelFrame(env_focus_row, text="Environment Focus", bg=dark_grey, fg=text_color)
    env_focus_text = scrolledtext.ScrolledText(
        env_focus_frame,
        wrap=tk.NONE,
        width=80,
        height=8,
        font=('Courier', 10),
        state='disabled',
        bg=dark_grey,
        fg=text_color,
    )
    env_focus_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    env_focus_text.tag_config("rrs_hdr", foreground="#BD93F9", font=('Courier', 11, 'bold'))
    env_focus_text.tag_config("rrs_rs", foreground="#50FA7B", font=('Courier', 11, 'bold'))
    env_focus_text.tag_config("rrs_rw", foreground="#FF5555", font=('Courier', 11, 'bold'))
    env_focus_row.add(env_focus_frame, stretch="always")

    env_copy_frame = tk.LabelFrame(env_focus_row, text="Environment Focus Lists", bg=dark_grey, fg=text_color)
    env_copy_toolbar = tk.Frame(env_copy_frame, bg=dark_grey)
    env_copy_toolbar.pack(fill=tk.X, padx=4, pady=(4, 0))
    env_copy_text = scrolledtext.ScrolledText(
        env_copy_frame,
        wrap=tk.WORD,
        width=52,
        height=8,
        font=('Courier', 10),
        state='disabled',
        bg=dark_grey,
        fg=text_color,
    )

    def copy_env_focus_lists():
        text = env_copy_text.get("1.0", tk.END).strip()
        if not text:
            rrs_status_var.set("Environment focus lists: nothing to copy.")
            return
        copy_text_to_clipboard(env_copy_text, text)
        rrs_status_var.set("Copied environment focus lists to clipboard.")

    tk.Button(
        env_copy_toolbar,
        text="Copy",
        command=copy_env_focus_lists,
        relief=tk.RAISED,
        padx=10,
        bg=dark_grey,
        fg=text_color,
    ).pack(side=tk.LEFT)
    env_copy_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    env_focus_row.add(env_copy_frame, stretch="always")

    rrs_compare_frame = tk.Frame(rrs_frame, bg=dark_grey)
    rrs_compare_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    industry_col = tk.LabelFrame(rrs_compare_frame, text="RS/RW vs Industry Ref", bg=dark_grey, fg=text_color)
    industry_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
    sector_col = tk.LabelFrame(rrs_compare_frame, text="RS/RW vs Sector", bg=dark_grey, fg=text_color)
    sector_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

    industry_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)
    sector_threshold_var = tk.DoubleVar(value=bot_instance.rrs_threshold)

    tk.Scale(industry_col, from_=0.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, label="Sensitivity",
             variable=industry_threshold_var, length=180, bg=dark_grey, fg=text_color, highlightthickness=0).pack(fill=tk.X, padx=4)
    tk.Scale(sector_col, from_=0.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, label="Sensitivity",
             variable=sector_threshold_var, length=180, bg=dark_grey, fg=text_color, highlightthickness=0).pack(fill=tk.X, padx=4)

    industry_timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)
    sector_timeframe_var = tk.StringVar(value=bot_instance.rrs_timeframe_key)

    industry_tf = tk.Frame(industry_col, bg=dark_grey)
    industry_tf.pack(fill=tk.X)
    sector_tf = tk.Frame(sector_col, bg=dark_grey)
    sector_tf.pack(fill=tk.X)
    for key in ("5m", "15m", "30m", "1h"):
        for parent, var in ((industry_tf, industry_timeframe_var), (sector_tf, sector_timeframe_var)):
            tk.Radiobutton(parent, text=key, variable=var, value=key, indicatoron=0, padx=4, pady=1,
                           bg=dark_grey, fg=text_color, selectcolor="#444444", activebackground="#444444",
                           activeforeground=text_color).pack(side=tk.LEFT, padx=1)

    industry_text = scrolledtext.ScrolledText(industry_col, wrap=tk.NONE, width=45, height=14, font=('Courier', 10), state='disabled', bg=dark_grey, fg=text_color)
    industry_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    sector_text = scrolledtext.ScrolledText(sector_col, wrap=tk.NONE, width=45, height=14, font=('Courier', 10), state='disabled', bg=dark_grey, fg=text_color)
    sector_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    group_frame = tk.LabelFrame(rrs_frame, text="Top Industries/Sectors", bg=dark_grey, fg=text_color)
    group_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    group_text = scrolledtext.ScrolledText(group_frame, wrap=tk.NONE, width=90, height=14, font=('Courier', 10), state='disabled', bg=dark_grey, fg=text_color)
    group_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    for widget in (rrs_main_text, industry_text, sector_text, group_text):
        widget.tag_config("rrs_hdr", foreground="#BD93F9", font=('Courier', 11, 'bold'))
        widget.tag_config("rrs_rs", foreground="#50FA7B", font=('Courier', 11, 'bold'))
        widget.tag_config("rrs_rw", foreground="#FF5555", font=('Courier', 11, 'bold'))


    tracker_loaded = {"value": False}

    tracker_host = tk.Frame(tracker_tab, bg=dark_grey, padx=8, pady=8)
    tracker_host.pack(fill=tk.BOTH, expand=True)

    tracker_toolbar = tk.Frame(tracker_host, bg=dark_grey)
    tracker_toolbar.pack(fill=tk.X, pady=(0, 6))
    tracker_status_var = tk.StringVar(value="Open this tab or click Refresh Tracker to load bounce stats.")
    tracker_min_samples_var = tk.IntVar(value=BOUNCE_PERFORMANCE_MIN_SAMPLES)

    tk.Label(tracker_toolbar, text="Min samples", bg=dark_grey, fg=text_color).pack(side=tk.LEFT, padx=(0, 4))
    tk.Spinbox(
        tracker_toolbar,
        from_=1,
        to=100,
        width=4,
        textvariable=tracker_min_samples_var,
        bg="#3A3A3A",
        fg=text_color,
        buttonbackground="#3A3A3A",
        insertbackground=text_color,
    ).pack(side=tk.LEFT, padx=(0, 8))

    tracker_columns = (
        "dimension",
        "direction",
        "segment",
        "sample_count",
        "avg_eod_r",
        "median_eod_r",
        "avg_mfe_r",
        "avg_mae_r",
        "positive_eod_rate",
        "target_1r_rate",
        "stop_rate",
        "recommendation",
        "example_symbols",
    )
    tracker_col_labels = {
        "dimension": "Dimension",
        "direction": "Dir",
        "segment": "Segment",
        "sample_count": "N",
        "avg_eod_r": "Avg EOD",
        "median_eod_r": "Med EOD",
        "avg_mfe_r": "Avg Peak",
        "avg_mae_r": "Avg MAE",
        "positive_eod_rate": "Green",
        "target_1r_rate": "1R Seen",
        "stop_rate": "Stop",
        "recommendation": "Rec",
        "example_symbols": "Examples",
    }
    tracker_col_widths = {
        "dimension": 165,
        "direction": 55,
        "segment": 190,
        "sample_count": 45,
        "avg_eod_r": 75,
        "median_eod_r": 75,
        "avg_mfe_r": 75,
        "avg_mae_r": 75,
        "positive_eod_rate": 65,
        "target_1r_rate": 65,
        "stop_rate": 65,
        "recommendation": 95,
        "example_symbols": 180,
    }

    tracker_table_host = tk.Frame(tracker_host, bg=dark_grey)
    tracker_table_host.pack(fill=tk.BOTH, expand=True)
    tracker_tree = ttk.Treeview(
        tracker_table_host,
        columns=tracker_columns,
        show="headings",
        height=14,
    )
    tracker_y_scroll = ttk.Scrollbar(tracker_table_host, orient=tk.VERTICAL, command=tracker_tree.yview)
    tracker_x_scroll = ttk.Scrollbar(tracker_table_host, orient=tk.HORIZONTAL, command=tracker_tree.xview)
    tracker_tree.configure(yscrollcommand=tracker_y_scroll.set, xscrollcommand=tracker_x_scroll.set)
    tracker_tree.grid(row=0, column=0, sticky="nsew")
    tracker_y_scroll.grid(row=0, column=1, sticky="ns")
    tracker_x_scroll.grid(row=1, column=0, sticky="ew")
    tracker_table_host.rowconfigure(0, weight=1)
    tracker_table_host.columnconfigure(0, weight=1)
    for col in tracker_columns:
        tracker_tree.heading(col, text=tracker_col_labels[col])
        tracker_tree.column(
            col,
            width=tracker_col_widths[col],
            minwidth=45,
            stretch=(col in {"segment", "example_symbols"}),
            anchor=tk.W if col in {"dimension", "segment", "recommendation", "example_symbols"} else tk.CENTER,
        )

    tracker_report_frame = tk.LabelFrame(
        tracker_host,
        text="Current Bounce Performance Report",
        bg=dark_grey,
        fg=text_color,
    )
    tracker_report_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
    tracker_report_text = scrolledtext.ScrolledText(
        tracker_report_frame,
        wrap=tk.NONE,
        height=11,
        font=("Courier", 10),
        state="disabled",
        bg=dark_grey,
        fg=text_color,
    )
    tracker_report_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    tk.Label(tracker_host, textvariable=tracker_status_var, bg=dark_grey, fg=text_color).pack(fill=tk.X, anchor=tk.W)

    def refresh_bounce_setup_tracker_view():
        try:
            try:
                min_samples = max(1, int(tracker_min_samples_var.get() or BOUNCE_PERFORMANCE_MIN_SAMPLES))
            except Exception:
                min_samples = BOUNCE_PERFORMANCE_MIN_SAMPLES
                tracker_min_samples_var.set(min_samples)

            rows = build_intraday_bounce_performance_rows(min_samples=min_samples)
            INTRADAY_BOUNCE_PERFORMANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(INTRADAY_BOUNCE_PERFORMANCE_CSV, index=False)
            write_intraday_bounce_performance_report(rows, report_path=INTRADAY_BOUNCE_PERFORMANCE_REPORT)

            rec_rank = {"focus": 0, "neutral": 1, "watch_more_samples": 2, "avoid": 3}
            dimension_rank = {
                "bounce_type": 0,
                "bounce_combo": 1,
                "master_avwap_swing_trait": 2,
                "master_avwap_focus": 3,
                "master_avwap_priority_bucket": 4,
                "master_avwap_setup_family": 5,
                "master_avwap_h1_focus_type": 6,
                "top_pattern_entry_timing": 7,
                "market_environment": 8,
                "rrs_alignment": 9,
                "time_bucket": 10,
            }
            display_rows = sorted(
                rows,
                key=lambda row: (
                    rec_rank.get(str(row.get("recommendation") or ""), 9),
                    dimension_rank.get(str(row.get("dimension") or ""), 99),
                    str(row.get("direction") or ""),
                    -float(row.get("edge_score", 0.0) or 0.0),
                    -int(row.get("sample_count", 0) or 0),
                    str(row.get("segment") or ""),
                ),
            )

            for item_id in tracker_tree.get_children():
                tracker_tree.delete(item_id)
            for row in display_rows:
                tracker_tree.insert(
                    "",
                    tk.END,
                    values=(
                        row.get("dimension", ""),
                        row.get("direction", ""),
                        row.get("segment", ""),
                        int(row.get("sample_count", 0) or 0),
                        _format_bounce_perf_r(row.get("avg_eod_r", row.get("avg_close_r"))),
                        _format_bounce_perf_r(row.get("median_eod_r", row.get("median_close_r"))),
                        _format_bounce_perf_r(row.get("avg_mfe_r")),
                        _format_bounce_perf_r(row.get("avg_mae_r")),
                        _format_bounce_perf_pct(row.get("positive_eod_rate")),
                        _format_bounce_perf_pct(row.get("target_1r_rate")),
                        _format_bounce_perf_pct(row.get("stop_rate")),
                        row.get("recommendation", ""),
                        row.get("example_symbols", ""),
                    ),
                )

            report_text = INTRADAY_BOUNCE_PERFORMANCE_REPORT.read_text(encoding="utf-8")
            tracker_report_text.config(state="normal")
            tracker_report_text.delete("1.0", tk.END)
            tracker_report_text.insert(tk.END, report_text)
            tracker_report_text.config(state="disabled")
            tracker_report_text.see("1.0")
            tracker_status_var.set(
                f"Rows: {len(rows)} | Min samples: {min_samples} | CSV: {INTRADAY_BOUNCE_PERFORMANCE_CSV}"
            )
        except Exception as exc:
            tracker_status_var.set(f"Tracker refresh failed: {exc}")

    def refresh_tracker_from_button():
        tracker_loaded["value"] = True
        refresh_bounce_setup_tracker_view()

    tk.Button(
        tracker_toolbar,
        text="Refresh Tracker",
        command=refresh_tracker_from_button,
        relief=tk.RAISED,
        padx=10,
        bg="#3A3A3A",
        fg=text_color,
    ).pack(side=tk.LEFT, padx=(0, 8))
    tk.Label(tracker_toolbar, text="Peak = MFE, EOD = final regular-session close.", bg=dark_grey, fg=text_color).pack(
        side=tk.LEFT
    )

    def on_main_tab_changed(_event=None):
        try:
            selected_index = main_notebook.index(main_notebook.select())
        except Exception:
            selected_index = -1
        if selected_index == 1 and not tracker_loaded["value"]:
            tracker_loaded["value"] = True
            refresh_bounce_setup_tracker_view()

    main_notebook.bind("<<NotebookTabChanged>>", on_main_tab_changed)

    button_frame = tk.Frame(day_tab, bg=dark_grey)  # Add background color to button frame
    button_frame.pack(fill=tk.X, pady=10)

    bounce_toggle_frame = tk.LabelFrame(
        day_tab,
        text="Bounce Filters",
        bg=dark_grey,
        fg=text_color,
        padx=8,
        pady=6,
        highlightbackground="#444444",
        highlightcolor="#444444",
    )
    bounce_toggle_frame.pack(fill=tk.X, pady=(0, 8))

    bounce_toggle_vars = {}

    def on_toggle_bounce(bounce_key, var):
        bot_instance.set_bounce_type_enabled(bounce_key, bool(var.get()))

    toggle_order = [
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

    for idx, bounce_key in enumerate(toggle_order):
        var = tk.BooleanVar(value=bot_instance.is_bounce_type_enabled(bounce_key))
        bounce_toggle_vars[bounce_key] = var
        chk = tk.Checkbutton(
            bounce_toggle_frame,
            text=BOUNCE_TYPE_LABELS.get(bounce_key, bounce_key),
            variable=var,
            command=lambda k=bounce_key, v=var: on_toggle_bounce(k, v),
            bg=dark_grey,
            fg=text_color,
            selectcolor="#444444",
            activebackground="#444444",
            activeforeground=text_color,
        )
        row = idx // 4
        col = idx % 4
        chk.grid(row=row, column=col, sticky="w", padx=6, pady=2)


    def check_dvwap_touches():
        results = bot_instance.check_dynamic_vwap_touches()
        text_area.config(state='normal')
        text_area.insert(tk.END, "\n=== DVWAP Touch Check Results ===\n", "blue")
        for result in results:
            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {result}\n", "green")
        text_area.config(state='disabled')
        text_area.see(tk.END)
        root.update()

    def check_dvwap2_touches():
        results = bot_instance.check_dynamic_vwap2_touches()
        text_area.config(state='normal')
        text_area.insert(tk.END, "\n=== DVWAP2 Touch Check Results ===\n", "blue")
        for result in results:
            text_area.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {result}\n", "green")
        text_area.config(state='disabled')
        text_area.see(tk.END)
        root.update()

    dvwap_button = tk.Button(
        button_frame,
        text="Check DVWAP Touches",
        command=check_dvwap_touches,
        relief=tk.RAISED,
        padx=10
    )
    dvwap_button.pack(side=tk.LEFT, padx=5)

    dvwap2_button = tk.Button(
        button_frame,
        text="Check DVWAP2 Touches",
        command=check_dvwap2_touches,
        relief=tk.RAISED,
        padx=10
    )
    dvwap2_button.pack(side=tk.LEFT, padx=5)

    start_scanning_button = tk.Button(
        button_frame,
        text="Start Scanning",
        command=start_scanning,
        relief=tk.RAISED,
        padx=10,
    )
    start_scanning_button.pack(side=tk.LEFT, padx=5)
    stop_scanning_button = tk.Button(
        button_frame,
        text="Stop Scanning",
        command=stop_scanning,
        relief=tk.RAISED,
        padx=10,
    )
    stop_scanning_button.pack(side=tk.LEFT, padx=5)

    def process_bounce_queue():
        scanning_active = bot_instance.is_scanning_enabled()
        start_scanning_button.config(state=(tk.DISABLED if scanning_active else tk.NORMAL))
        stop_scanning_button.config(state=(tk.NORMAL if scanning_active else tk.DISABLED))
        while True:
            try:
                msg, tag = bounce_queue.get_nowait()
                target_area = d1_flags_text if str(tag).startswith("d1_flag") else text_area
                target_area.config(state='normal')
                append_alert_message(
                    target_area,
                    msg,
                    str(tag),
                    datetime.now().strftime('%H:%M:%S'),
                    feedback_source="bounce_bot_full_gui",
                )
                target_area.config(state='disabled')
                target_area.see(tk.END)
                root.update()
            except queue.Empty:
                break
        root.after(100, process_bounce_queue)

    def render_rrs_snapshot(snapshot):
        threshold = snapshot.get("threshold", RRS_DEFAULT_THRESHOLD)
        timeframe_key = snapshot.get("timeframe_key", "5m")
        timeframe_label = RRS_TIMEFRAMES.get(timeframe_key, {}).get("label", timeframe_key)
        results = snapshot.get("results", [])
        sector_results = snapshot.get("results_sector", [])
        industry_results = snapshot.get("results_industry", [])
        group_strength = snapshot.get("group_strength", {})
        timestamp = snapshot.get("timestamp", datetime.now())

        def render_table(widget, title, rows, local_threshold):
            selected_tf = timeframe_key
            if title.startswith("RS/RW vs Industry"):
                selected_tf = industry_timeframe_var.get()
            elif title.startswith("RS/RW vs Sector"):
                selected_tf = sector_timeframe_var.get()
            widget.config(state='normal')
            widget.delete("1.0", tk.END)
            widget.insert(tk.END, f"{title}  TF:{selected_tf}  Threshold:{local_threshold:.2f}\n", "rrs_hdr")
            widget.insert(tk.END, "SYMBOL  SIDE  RRS    POWER\n")
            widget.insert(tk.END, "--------------------------\n")
            filtered = [r for r in rows if abs(r[2]) >= local_threshold]
            if not filtered:
                widget.insert(tk.END, "No symbols flagged.\n")
            for signal, symbol, rrs_value, power in filtered:
                line = f"{symbol:<6}  {signal:<4}  {rrs_value:+.2f}  {power if power is not None else 0:>6.2f}\n"
                widget.insert(tk.END, line, "rrs_rs" if signal == "RS" else "rrs_rw")
            widget.config(state='disabled')
            widget.see("1.0")

        render_table(rrs_main_text, "RS/RW vs SPY", results, threshold)
        render_table(industry_text, "RS/RW vs Industry Ref", industry_results, industry_threshold_var.get())
        render_table(sector_text, "RS/RW vs Sector", sector_results, sector_threshold_var.get())

        group_text.config(state='normal')
        group_text.delete("1.0", tk.END)
        group_text.insert(tk.END, f"Last scan: {timestamp.strftime('%H:%M:%S')}   Timeframe: {timeframe_label}\n", "rrs_hdr")
        for tf in ("D1", "H1", "M5"):
            payload = group_strength.get(tf, {})
            sectors = payload.get("sectors", [])
            industries = payload.get("industries", [])
            group_text.insert(tk.END, f"\n[{tf}] Sectors\n", "rrs_hdr")
            for item in sectors[:SCAN_EXTREME_COUNT]:
                group_text.insert(tk.END, f"  + {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n", "rrs_rs")
            for item in list(reversed(sectors[-SCAN_EXTREME_COUNT:])):
                group_text.insert(tk.END, f"  - {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n", "rrs_rw")
            group_text.insert(tk.END, f"[{tf}] Industries\n", "rrs_hdr")
            for item in industries[:SCAN_EXTREME_COUNT]:
                group_text.insert(tk.END, f"  + {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n", "rrs_rs")
            for item in list(reversed(industries[-SCAN_EXTREME_COUNT:])):
                group_text.insert(tk.END, f"  - {item['group_key']:<26} {item['etf']:<6} {item['rrs']:+.2f} {item['power_index']:+.2f}\n", "rrs_rw")
        group_text.config(state='disabled')
        group_text.see("1.0")

        env_focus_text.config(state='normal')
        env_focus_text.delete("1.0", tk.END)
        env_highlights = snapshot.get("environment_highlights", [])
        env_focus_text.insert(tk.END, f"{snapshot.get('market_environment_label', 'Environment')} Focus\n", "rrs_hdr")
        for section in env_highlights:
            env_focus_text.insert(tk.END, f"\n{section.get('title', 'Section')}\n", "rrs_hdr")
            rows = section.get("rows", [])
            if not rows:
                env_focus_text.insert(tk.END, "  None\n")
            for row in rows:
                env_focus_text.insert(tk.END, f"  {row.get('text', '')}\n", row.get("tag", "rrs_rs"))
        env_focus_text.config(state='disabled')
        env_focus_text.see("1.0")

        env_copy_text.config(state='normal')
        env_copy_text.delete("1.0", tk.END)
        env_copy_text.insert(tk.END, build_environment_focus_copy_text(snapshot))
        env_copy_text.config(state='disabled')
        env_copy_text.see("1.0")

    def process_rrs_queue():
        while True:
            try:
                msg, tag = rrs_queue.get_nowait()
            except queue.Empty:
                break

            if tag == "rrs_status":
                rrs_status_var.set(str(msg))
            elif tag == "rrs_snapshot":
                render_rrs_snapshot(msg)
        root.after(150, process_rrs_queue)


    def on_closing():
        bot_instance.disconnect()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    process_bounce_queue()
    process_rrs_queue()
    root.mainloop()


__all__ = [
    "append_alert_message",
    "build_environment_focus_copy_text",
    "choose_gui_mode",
    "configure_alert_tags",
    "copy_text_to_clipboard",
    "create_rrs_confirmed_panel",
    "prompt_change_home_folder",
    "start_gui",
    "start_lightweight_gui",
]
