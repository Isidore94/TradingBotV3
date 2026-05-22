from __future__ import annotations

import time

from . import app as _app

globals().update(
    {
        name: value
        for name, value in vars(_app).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)


class BounceBotController:
    class GUIProxy:
        def __init__(self, controller: "BounceBotController"):
            self._controller = controller
            self.rrs_threshold = controller.rrs_threshold
            self.rrs_timeframe_key = controller.rrs_timeframe_key

        def set_rrs_threshold(self, value: float) -> None:
            self._controller.set_rrs_threshold(value)

        def set_rrs_timeframe(self, key: str) -> None:
            self._controller.set_rrs_timeframe(key)

        def set_market_environment(self, env_key: str) -> None:
            self._controller.set_market_environment(env_key)

        def get_market_environment(self) -> str:
            return self._controller.get_market_environment()

    def __init__(
        self,
        include_approaching: bool,
        ui_parent: tk.Misc,
        *,
        start_scanning_enabled: bool = False,
    ):
        self.include_approaching = include_approaching
        self.ui_parent = ui_parent
        self.rrs_queue: queue.Queue = queue.Queue()
        self.bounce_queue: queue.Queue = queue.Queue()
        self.status_var = tk.StringVar(value="starting...")
        self.connection_var = tk.StringVar(value="IB: disconnected")
        self.active_bounce_var = tk.StringVar(value="active bounces: 0")
        self.bot_instance = None
        self._lock = threading.Lock()

        self.rrs_threshold = 2.0
        self.rrs_timeframe_key = "5m"
        self.market_environment = "bullish_strong"
        self.scanning_enabled = bool(start_scanning_enabled)
        self.bounce_type_settings = dict(BOUNCE_TYPE_DEFAULTS)
        self.gui_proxy = self.GUIProxy(self)

    def _run_on_ui_thread(self, callback) -> None:
        if threading.current_thread() is threading.main_thread():
            callback()
            return
        try:
            self.ui_parent.after(0, callback)
        except RuntimeError:
            pass

    def _set_var(self, variable: tk.Variable, value: str) -> None:
        self._run_on_ui_thread(lambda: variable.set(value))

    def _with_bot(self, callback):
        with self._lock:
            bot = self.bot_instance
        if bot is None:
            return None
        return callback(bot)

    def _sync_state_from_bot(self, bot) -> None:
        self.rrs_threshold = float(getattr(bot, "rrs_threshold", self.rrs_threshold))
        self.rrs_timeframe_key = str(getattr(bot, "rrs_timeframe_key", self.rrs_timeframe_key))
        self.market_environment = str(bot.get_market_environment())
        self.scanning_enabled = bool(bot.is_scanning_enabled())
        for bounce_key in self.bounce_type_settings:
            self.bounce_type_settings[bounce_key] = bool(bot.is_bounce_type_enabled(bounce_key))
        self.gui_proxy.rrs_threshold = self.rrs_threshold
        self.gui_proxy.rrs_timeframe_key = self.rrs_timeframe_key

    def _apply_saved_state(self, bot) -> None:
        bot.set_rrs_threshold(self.rrs_threshold)
        bot.set_rrs_timeframe(self.rrs_timeframe_key)
        bot.set_market_environment(self.market_environment)
        bot.set_scanning_enabled(self.scanning_enabled)
        for bounce_key, enabled in self.bounce_type_settings.items():
            bot.set_bounce_type_enabled(bounce_key, enabled)

    def _emit(self, message: str) -> None:
        self._set_var(self.status_var, message)

    def _make_callback(self):
        def gui_callback(message, tag):
            if tag.startswith("rrs"):
                self.rrs_queue.put((message, tag))
                return
            if tag == "blue" and "removed from" in str(message):
                return
            if not self.include_approaching and (tag == "approaching" or str(tag).startswith("approaching_")):
                return
            self.bounce_queue.put((message, tag))

        return gui_callback

    def start(self) -> None:
        def run_bot() -> None:
            self._emit("connecting")
            self._set_var(self.connection_var, "IB: connecting")
            try:
                bot = run_bot_with_gui(
                    self._make_callback(),
                    start_scanning_enabled=self.scanning_enabled,
                )
                self._apply_saved_state(bot)
                self._sync_state_from_bot(bot)
                with self._lock:
                    self.bot_instance = bot
                self._run_on_ui_thread(self.refresh_active_bounces)
                self._set_var(self.connection_var, "IB: connected")
                self._emit("connected")
            except Exception as exc:
                self._set_var(self.connection_var, "IB: disconnected")
                self._emit(f"start failed: {exc}")

        threading.Thread(target=run_bot, daemon=True).start()

    def restart(self) -> None:
        self.stop()
        self.start()

    def stop(self) -> None:
        with self._lock:
            bot = self.bot_instance
            self.bot_instance = None
        if bot:
            try:
                bot.disconnect()
            except Exception:
                pass
        self._set_var(self.connection_var, "IB: disconnected")
        self._set_var(self.active_bounce_var, "active bounces: 0")
        self._emit("stopped")

    def set_rrs_threshold(self, value: float) -> None:
        self.rrs_threshold = float(value)
        self.gui_proxy.rrs_threshold = self.rrs_threshold
        self._with_bot(lambda bot: bot.set_rrs_threshold(self.rrs_threshold))

    def set_rrs_timeframe(self, key: str) -> None:
        if key not in RRS_TIMEFRAMES:
            return
        self.rrs_timeframe_key = key
        self.gui_proxy.rrs_timeframe_key = key
        self._with_bot(lambda bot: bot.set_rrs_timeframe(key))

    def set_market_environment(self, env_key: str) -> None:
        if env_key not in MARKET_ENVIRONMENTS:
            return
        self.market_environment = env_key
        self._with_bot(lambda bot: bot.set_market_environment(env_key))

    def get_market_environment(self) -> str:
        return self.market_environment

    def set_bounce_type_enabled(self, bounce_type: str, enabled: bool) -> None:
        if bounce_type not in self.bounce_type_settings:
            return
        self.bounce_type_settings[bounce_type] = bool(enabled)
        self._with_bot(lambda bot: bot.set_bounce_type_enabled(bounce_type, enabled))

    def is_bounce_type_enabled(self, bounce_type: str) -> bool:
        return bool(self.bounce_type_settings.get(bounce_type, False))

    def set_scanning_enabled(self, enabled: bool) -> None:
        self.scanning_enabled = bool(enabled)
        self._with_bot(lambda bot: bot.set_scanning_enabled(self.scanning_enabled))

    def is_scanning_enabled(self) -> bool:
        return self.scanning_enabled

    def start_scanning(self) -> None:
        self.set_scanning_enabled(True)

    def stop_scanning(self) -> None:
        self.set_scanning_enabled(False)

    def refresh_active_bounces(self) -> None:
        def _read_count(bot):
            return len(bot.find_active_master_avwap_bounces())

        count = self._with_bot(_read_count)
        if count is None:
            self._set_var(self.active_bounce_var, "active bounces: 0")
            return
        self._set_var(self.active_bounce_var, f"active bounces: {count}")

    def run_manual_check(self, method_name: str, heading: str) -> None:
        def worker() -> None:
            with self._lock:
                bot = self.bot_instance
            if bot is None:
                self.bounce_queue.put((f"{heading}: BounceBot not connected.", "red"))
                return
            try:
                results = getattr(bot, method_name)()
            except Exception as exc:
                self.bounce_queue.put((f"{heading}: {exc}", "red"))
                return

            self.bounce_queue.put((f"=== {heading} ===", "blue"))
            if results:
                for result in results:
                    self.bounce_queue.put((str(result), "green"))
            else:
                self.bounce_queue.put(("No symbols flagged.", "blue"))

        threading.Thread(target=worker, daemon=True).start()


class BaseBounceBotPanel:
    def __init__(self, parent: tk.Misc, controller: BounceBotController, switch_mode_callback=None):
        self.parent = parent
        self.controller = controller
        self.switch_mode_callback = switch_mode_callback
        self.container = ttk.Frame(parent)
        self._queue_after_id = None
        self.alert_text: scrolledtext.ScrolledText | None = None
        self.d1_alert_text: scrolledtext.ScrolledText | None = None
        self.on_output_changed = None
        self._last_active_bounce_refresh = 0.0

    def pack(self, **kwargs) -> None:
        self.container.pack(**kwargs)

    def _create_alerts_widget(self, parent: tk.Misc, font_size: int) -> scrolledtext.ScrolledText:
        text_area = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=("Courier", font_size),
            state="disabled",
            bg=INPUT_GREY,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
        )
        configure_alert_tags(text_area, font_size=font_size)
        return text_area

    def _alert_widget_for_tag(self, tag: str) -> scrolledtext.ScrolledText | None:
        if str(tag).startswith("d1_flag") and self.d1_alert_text is not None:
            return self.d1_alert_text
        return self.alert_text

    def _append_alert_with_timestamp(self, message: Any, tag: str) -> None:
        target = self._alert_widget_for_tag(tag)
        if target is None:
            return
        target.config(state="normal")
        append_alert_message(
            target,
            message,
            tag,
            datetime.now().strftime("%H:%M:%S"),
            feedback_callback=self._record_bounce_feedback,
            feedback_source="consolidated_gui",
        )
        target.config(state="disabled")
        target.see(tk.END)
        self._notify_output_changed()

    def _record_bounce_feedback(self, context: dict, rating: str, reason: str, source: str) -> None:
        record_bounce_feedback(context, rating, reason, source=source)
        symbol = str(context.get("symbol") or "bounce").strip().upper() or "bounce"
        self.controller.status_var.set(f"Saved bounce feedback: {symbol} -> {rating}")
        self._notify_output_changed()

    def clear_alerts(self) -> None:
        for widget in (self.alert_text, self.d1_alert_text):
            if widget is None:
                continue
            widget.config(state="normal")
            widget.delete("1.0", tk.END)
            widget.config(state="disabled")
        self._notify_output_changed()

    def _notify_output_changed(self) -> None:
        callback = getattr(self, "on_output_changed", None)
        if callable(callback):
            callback()

    def start(self) -> None:
        self.controller.start()
        self._refresh_active_bounces_if_due(force=True)
        self._process_queues()

    def _refresh_active_bounces_if_due(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_active_bounce_refresh < 5.0:
            return
        self._last_active_bounce_refresh = now
        self.controller.refresh_active_bounces()

    def _process_queues(self) -> None:
        raise NotImplementedError

    def on_close(self) -> None:
        try:
            if self._queue_after_id:
                self.container.after_cancel(self._queue_after_id)
        except Exception:
            pass
        self.controller.stop()


class SimpleBounceBotPanel(BaseBounceBotPanel):
    def __init__(self, parent: tk.Misc, switch_mode_callback=None):
        super().__init__(
            parent,
            BounceBotController(
                include_approaching=False,
                ui_parent=parent,
                start_scanning_enabled=False,
            ),
            switch_mode_callback=switch_mode_callback,
        )
        self._syncing_controls = False
        self.toggle_vars: dict[str, tk.BooleanVar] = {}
        self.rrs_threshold_var = tk.DoubleVar(value=self.controller.rrs_threshold)
        self.timeframe_var = tk.StringVar(value=self.controller.rrs_timeframe_key)
        self.environment_var = tk.StringVar(value=self.controller.get_market_environment())
        self._build_layout()
        self.start()

    def _build_layout(self) -> None:
        header = ttk.Frame(self.container)
        header.pack(fill=tk.X, padx=10, pady=(10, 8))

        ttk.Label(header, text="BounceBot Simple").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.controller.connection_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(header, textvariable=self.controller.status_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(header, textvariable=self.controller.active_bounce_var).pack(side=tk.LEFT, padx=(12, 0))
        if self.switch_mode_callback:
            ttk.Button(header, text="Switch to Full", command=self.switch_mode_callback).pack(side=tk.RIGHT)
        ttk.Button(header, text="Reconnect", command=self.controller.restart).pack(side=tk.RIGHT)
        self.stop_scanning_button = ttk.Button(header, text="Stop Scanning", command=self.controller.stop_scanning)
        self.stop_scanning_button.pack(side=tk.RIGHT, padx=(0, 8))
        self.start_scanning_button = ttk.Button(header, text="Start Scanning", command=self.controller.start_scanning)
        self.start_scanning_button.pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(header, text="Disconnect", command=self.controller.stop).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(header, text="Clear", command=self.clear_alerts).pack(side=tk.RIGHT, padx=(0, 8))

        controls = tk.Frame(self.container, bg=DARK_GREY)
        controls.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Label(controls, text="RRS Sensitivity", bg=DARK_GREY, fg=TEXT_COLOR).pack(side=tk.LEFT)
        self.rrs_threshold_var.trace_add("write", self._on_rrs_threshold_change)
        tk.Scale(
            controls,
            from_=0.0,
            to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.rrs_threshold_var,
            length=180,
            bg=DARK_GREY,
            fg=TEXT_COLOR,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=(8, 14))

        tk.Label(controls, text="Timeframe", bg=DARK_GREY, fg=TEXT_COLOR).pack(side=tk.LEFT)
        for key in ("5m", "15m", "30m", "1h"):
            tk.Radiobutton(
                controls,
                text=RRS_TIMEFRAMES[key]["label"],
                variable=self.timeframe_var,
                value=key,
                indicatoron=0,
                command=self._on_timeframe_change,
                padx=6,
                pady=2,
                bg=DARK_GREY,
                fg=TEXT_COLOR,
                selectcolor="#444444",
                activebackground="#444444",
                activeforeground=TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=2)

        env_frame = tk.Frame(self.container, bg=DARK_GREY)
        env_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Label(env_frame, text="Market Environment", bg=DARK_GREY, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=(0, 8))
        for key, info in MARKET_ENVIRONMENTS.items():
            tk.Radiobutton(
                env_frame,
                text=info["label"],
                variable=self.environment_var,
                value=key,
                indicatoron=0,
                command=self._on_environment_change,
                padx=8,
                pady=3,
                bg=DARK_GREY,
                fg=TEXT_COLOR,
                selectcolor="#444444",
                activebackground="#444444",
                activeforeground=TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=2)

        alerts_split = ttk.PanedWindow(self.container, orient=tk.HORIZONTAL)
        alerts_split.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        alerts_frame = ttk.LabelFrame(alerts_split, text="BounceBot Alerts")
        self.alert_text = self._create_alerts_widget(alerts_frame, font_size=11)
        self.alert_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        alerts_split.add(alerts_frame, weight=3)

        d1_frame = ttk.LabelFrame(alerts_split, text="D1 Master AVWAP Events")
        self.d1_alert_text = self._create_alerts_widget(d1_frame, font_size=10)
        self.d1_alert_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        alerts_split.add(d1_frame, weight=2)

        bounce_toggle_frame = tk.LabelFrame(
            self.container,
            text="Bounce Filters",
            bg=DARK_GREY,
            fg=TEXT_COLOR,
            padx=8,
            pady=6,
            highlightbackground="#444444",
            highlightcolor="#444444",
        )
        bounce_toggle_frame.pack(fill=tk.X, padx=10, pady=(0, 10), side=tk.BOTTOM)

        for idx, bounce_key in enumerate(BOUNCE_TOGGLE_ORDER):
            var = tk.BooleanVar(value=self.controller.is_bounce_type_enabled(bounce_key))
            self.toggle_vars[bounce_key] = var
            tk.Checkbutton(
                bounce_toggle_frame,
                text=BOUNCE_TYPE_LABELS.get(bounce_key, bounce_key),
                variable=var,
                command=lambda k=bounce_key, v=var: self.controller.set_bounce_type_enabled(k, bool(v.get())),
                bg=DARK_GREY,
                fg=TEXT_COLOR,
                selectcolor="#444444",
                activebackground="#444444",
                activeforeground=TEXT_COLOR,
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=6, pady=2)

    def _sync_controls_from_controller(self) -> None:
        self._syncing_controls = True
        try:
            if float(self.rrs_threshold_var.get()) != float(self.controller.rrs_threshold):
                self.rrs_threshold_var.set(self.controller.rrs_threshold)
            if self.timeframe_var.get() != self.controller.rrs_timeframe_key:
                self.timeframe_var.set(self.controller.rrs_timeframe_key)
            if self.environment_var.get() != self.controller.get_market_environment():
                self.environment_var.set(self.controller.get_market_environment())
            for bounce_key, var in self.toggle_vars.items():
                expected = self.controller.is_bounce_type_enabled(bounce_key)
                if bool(var.get()) != bool(expected):
                    var.set(expected)
        finally:
            self._syncing_controls = False
        self._sync_scanning_controls()

    def _sync_scanning_controls(self) -> None:
        scanning_enabled = self.controller.is_scanning_enabled()
        self.start_scanning_button.configure(state=("disabled" if scanning_enabled else "normal"))
        self.stop_scanning_button.configure(state=("normal" if scanning_enabled else "disabled"))

    def _on_rrs_threshold_change(self, *_args) -> None:
        if self._syncing_controls:
            return
        self.controller.set_rrs_threshold(self.rrs_threshold_var.get())

    def _on_timeframe_change(self) -> None:
        if self._syncing_controls:
            return
        self.controller.set_rrs_timeframe(self.timeframe_var.get())

    def _on_environment_change(self) -> None:
        if self._syncing_controls:
            return
        self.controller.set_market_environment(self.environment_var.get())

    def _process_queues(self) -> None:
        self._sync_controls_from_controller()

        while True:
            try:
                message, tag = self.controller.rrs_queue.get_nowait()
            except queue.Empty:
                break
            if tag == "rrs_status":
                self.controller.status_var.set(str(message))

        while True:
            try:
                message, tag = self.controller.bounce_queue.get_nowait()
            except queue.Empty:
                break
            self._append_alert_with_timestamp(message, str(tag))

        self._refresh_active_bounces_if_due()
        self._queue_after_id = self.container.after(150, self._process_queues)


class FullBounceBotPanel(BaseBounceBotPanel):
    def __init__(self, parent: tk.Misc, switch_mode_callback=None):
        super().__init__(
            parent,
            BounceBotController(
                include_approaching=False,
                ui_parent=parent,
                start_scanning_enabled=False,
            ),
            switch_mode_callback=switch_mode_callback,
        )
        self.toggle_vars: dict[str, tk.BooleanVar] = {}
        self._build_layout()
        self.start()

    def _build_layout(self) -> None:
        header = ttk.Frame(self.container)
        header.pack(fill=tk.X, padx=10, pady=(10, 8))

        ttk.Label(header, text="BounceBot Full").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.controller.connection_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(header, textvariable=self.controller.status_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(header, textvariable=self.controller.active_bounce_var).pack(side=tk.LEFT, padx=(12, 0))
        controls_frame = ttk.Frame(header)
        controls_frame.pack(side=tk.RIGHT)
        if self.switch_mode_callback:
            ttk.Button(controls_frame, text="Switch to Simple", command=self.switch_mode_callback).pack(side=tk.LEFT, padx=(0, 8))
        self.start_scanning_button = tk.Button(
            controls_frame,
            text="Start Scanning",
            command=self.controller.start_scanning,
            relief=tk.RAISED,
            padx=10,
            bg=PANEL_GREY,
            fg=TEXT_COLOR,
        )
        self.start_scanning_button.pack(side=tk.LEFT)
        self.stop_scanning_button = tk.Button(
            controls_frame,
            text="Stop Scanning",
            command=self.controller.stop_scanning,
            relief=tk.RAISED,
            padx=10,
            bg=PANEL_GREY,
            fg=TEXT_COLOR,
        )
        self.stop_scanning_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls_frame, text="Clear", command=self.clear_alerts).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls_frame, text="Reconnect", command=self.controller.restart).pack(side=tk.LEFT, padx=(8, 0))

        content_pane = tk.PanedWindow(
            self.container,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=10,
            showhandle=True,
            bg=DARK_GREY,
        )
        content_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        alerts_frame = tk.Frame(content_pane, bg=DARK_GREY)
        alerts_split = tk.PanedWindow(
            alerts_frame,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=8,
            showhandle=True,
            bg=DARK_GREY,
        )
        alerts_split.pack(fill=tk.BOTH, expand=True)

        bounce_alerts_frame = tk.LabelFrame(alerts_split, text="BounceBot Alerts", bg=DARK_GREY, fg=TEXT_COLOR)
        self.alert_text = self._create_alerts_widget(bounce_alerts_frame, font_size=11)
        self.alert_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        alerts_split.add(bounce_alerts_frame, stretch="always")

        d1_frame = tk.LabelFrame(alerts_split, text="D1 Master AVWAP Events", bg=DARK_GREY, fg=TEXT_COLOR)
        self.d1_alert_text = self._create_alerts_widget(d1_frame, font_size=10)
        self.d1_alert_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        alerts_split.add(d1_frame, stretch="always")

        content_pane.add(alerts_frame, stretch="always")

        self.rrs_panel = create_rrs_confirmed_panel(
            content_pane,
            bot_instance=self.controller.gui_proxy,
            dark_grey=DARK_GREY,
            text_color=TEXT_COLOR,
        )
        content_pane.add(self.rrs_panel["container"], stretch="always")

        bounce_toggle_frame = tk.LabelFrame(
            self.container,
            text="Bounce Filters",
            bg=DARK_GREY,
            fg=TEXT_COLOR,
            padx=8,
            pady=6,
            highlightbackground="#444444",
            highlightcolor="#444444",
        )
        bounce_toggle_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        for idx, bounce_key in enumerate(BOUNCE_TOGGLE_ORDER):
            var = tk.BooleanVar(value=self.controller.is_bounce_type_enabled(bounce_key))
            self.toggle_vars[bounce_key] = var
            tk.Checkbutton(
                bounce_toggle_frame,
                text=BOUNCE_TYPE_LABELS.get(bounce_key, bounce_key),
                variable=var,
                command=lambda k=bounce_key, v=var: self.controller.set_bounce_type_enabled(k, bool(v.get())),
                bg=DARK_GREY,
                fg=TEXT_COLOR,
                selectcolor="#444444",
                activebackground="#444444",
                activeforeground=TEXT_COLOR,
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=6, pady=2)

    def _sync_toggle_state(self) -> None:
        for bounce_key, var in self.toggle_vars.items():
            expected = self.controller.is_bounce_type_enabled(bounce_key)
            if bool(var.get()) != bool(expected):
                var.set(expected)
        scanning_enabled = self.controller.is_scanning_enabled()
        self.start_scanning_button.configure(state=(tk.DISABLED if scanning_enabled else tk.NORMAL))
        self.stop_scanning_button.configure(state=(tk.NORMAL if scanning_enabled else tk.DISABLED))

    def _process_queues(self) -> None:
        self._sync_toggle_state()

        while True:
            try:
                message, tag = self.controller.rrs_queue.get_nowait()
            except queue.Empty:
                break
            if tag == "rrs_status":
                self.controller.status_var.set(str(message))
                self.rrs_panel["rrs_status_var"].set(str(message))
            elif tag == "rrs_snapshot":
                self.rrs_panel["render_rrs_snapshot"](message)

        while True:
            try:
                message, tag = self.controller.bounce_queue.get_nowait()
            except queue.Empty:
                break
            self._append_alert_with_timestamp(message, str(tag))

        self._refresh_active_bounces_if_due()
        self._queue_after_id = self.container.after(150, self._process_queues)

__all__ = ["BounceBotController", "BaseBounceBotPanel", "SimpleBounceBotPanel", "FullBounceBotPanel"]
