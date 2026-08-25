"""Research -> Price Alerts: the sleep-in wake-up watchlist.

Enter current positions (or SPY itself) as tickers with alert levels above
and/or below price. While the GUI runs, PriceAlertService watches last prices
(pre/post market included) and pushes an urgent phone + Apple Watch
notification the moment a level crosses. Each side fires once per arm, then
needs re-arming.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from project_paths import get_local_setting, save_local_setting
from push_notify import (
    DEFAULT_NTFY_SERVER,
    PUSH_SERVER_SETTING,
    PUSH_TOKEN_SETTING,
    PUSH_TOPIC_SETTING,
)
from ui.services.price_alert_service import ALWAYS_ON_SETTING, PriceAlertService
from ui.widgets.section_header import SectionHeader

_COLUMNS = ("Symbol", "Alert Above", "Alert Below", "Armed ^", "Armed v", "Note", "Last Trigger")


class PriceAlertsPanel(QFrame):
    def __init__(
        self,
        service: PriceAlertService | None = None,
        *,
        read_only: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._owns_service = service is None
        self.service = service or PriceAlertService(self)
        self.read_only = bool(read_only)
        self._loading = False

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._save_table)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COLUMNS.index("Note"), QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COLUMNS.index("Last Trigger"), QHeaderView.ResizeMode.Stretch)
        self.table.cellChanged.connect(self._on_cell_changed)
        if self.read_only:
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        add_button = QPushButton("Add Row")
        remove_button = QPushButton("Remove Selected")
        rearm_button = QPushButton("Re-arm All")
        check_button = QPushButton("Check Now")
        test_button = QPushButton("Test Push")
        # The one that answers the overnight question. Ordinary "Test Push"
        # goes out at priority "high", which is NOT what the two
        # EVENING-permitted senders use - so passing it proved nothing about
        # whether a real 3am alert breaks through Sleep Focus. This sends at
        # the same "urgent" the price alerts and the SPY wake alarm send at,
        # and says so on the phone. See docs/EVENING_MODE_RUNBOOK.md.
        wake_button = QPushButton("Test wake alert (urgent)")
        wake_button.setToolTip(
            "Send one push at the SAME priority as your price alerts and the "
            "SPY wake alarm. Run it with Sleep Focus ON: ntfy has no Apple "
            "critical-alert entitlement, so urgent priority alone does not "
            "override Sleep Focus - the app also has to be allowed in "
            "Settings > Focus > Sleep, and the topic must not be set to "
            "Deliver Quietly."
        )
        add_button.clicked.connect(self._add_row)
        remove_button.clicked.connect(self._remove_selected)
        rearm_button.clicked.connect(self._rearm_all)
        check_button.clicked.connect(self.service.check_now)
        test_button.clicked.connect(self._test_push)
        wake_button.clicked.connect(self._test_wake_push)
        self.test_button = test_button
        self.wake_button = wake_button
        buttons = (add_button, remove_button, rearm_button, check_button, test_button, wake_button)
        if self.read_only:
            for button in buttons:
                button.setEnabled(False)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        for button in buttons:
            action_row.addWidget(button)
        action_row.addStretch(1)

        self.server_input = QLineEdit(str(get_local_setting(PUSH_SERVER_SETTING, DEFAULT_NTFY_SERVER) or DEFAULT_NTFY_SERVER))
        self.topic_input = QLineEdit(str(get_local_setting(PUSH_TOPIC_SETTING, "") or ""))
        self.topic_input.setPlaceholderText("your private ntfy topic (empty = pushes off)")
        self.token_input = QLineEdit(str(get_local_setting(PUSH_TOKEN_SETTING, "") or ""))
        self.token_input.setPlaceholderText("access token (optional)")
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.always_on_check = QCheckBox("Monitor in every mode (not just Auto EVENING)")
        self.always_on_check.setChecked(bool(get_local_setting(ALWAYS_ON_SETTING, True)))
        for widget in (self.server_input, self.topic_input, self.token_input):
            widget.editingFinished.connect(self._save_push_settings)
        self.always_on_check.toggled.connect(self._save_push_settings)
        if self.read_only:
            for widget in (
                self.server_input,
                self.topic_input,
                self.token_input,
                self.always_on_check,
            ):
                widget.setEnabled(False)

        push_row = QHBoxLayout()
        push_row.setSpacing(6)
        push_row.addWidget(QLabel("ntfy server:"))
        push_row.addWidget(self.server_input, 2)
        push_row.addWidget(QLabel("topic:"))
        push_row.addWidget(self.topic_input, 2)
        push_row.addWidget(QLabel("token:"))
        push_row.addWidget(self.token_input, 1)
        push_row.addWidget(self.always_on_check)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.service.statusChanged.connect(lambda _snapshot: self._refresh_status())
        self.service.triggered.connect(self._on_triggered)
        self.service.entriesChanged.connect(self._load_table)

        help_label = QLabel(
            "Phone/watch setup: install the ntfy app on the iPhone, subscribe it to the topic above "
            "(Apple Watch mirrors iPhone notifications automatically), then hit Test Push. "
            "In the ntfy app, enable critical alerting for the topic so urgent alerts break "
            "through phone Focus modes. Each side fires once, then shows here until you re-arm it."
        )
        help_label.setObjectName("MutedLabel")
        help_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(
            SectionHeader(
                "Price Alerts",
                "Positions + levels that should wake you up. Checked every minute, pre/post market included.",
            )
        )
        layout.addLayout(action_row)
        layout.addLayout(push_row)
        layout.addWidget(help_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table, 1)

        self._load_table()
        self._refresh_status()

    # ------------------------------------------------------------------
    # Table <-> store
    # ------------------------------------------------------------------
    def _load_table(self) -> None:
        self._loading = True
        try:
            entries = self.service.entries()
            self.table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                self._set_row(row, entry)
        finally:
            self._loading = False

    def _set_row(self, row: int, entry: dict) -> None:
        def _text_item(value: str, editable: bool = True) -> QTableWidgetItem:
            item = QTableWidgetItem(value)
            if not editable:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            return item

        def _check_item(checked: bool) -> QTableWidgetItem:
            item = QTableWidgetItem()
            item.setFlags(
                (item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable
            )
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            return item

        history = entry.get("history") or []
        last_trigger = ""
        if history:
            newest = history[-1]
            last_trigger = (
                f"{newest.get('date', '')} {newest.get('at', '')} "
                f"{newest.get('side', '')} {newest.get('level', '')} @ {newest.get('last', '')}"
            ).strip()
        self.table.setItem(row, 0, _text_item(str(entry.get("symbol") or "")))
        self.table.setItem(row, 1, _text_item(_level_text(entry.get("above"))))
        self.table.setItem(row, 2, _text_item(_level_text(entry.get("below"))))
        self.table.setItem(row, 3, _check_item(bool(entry.get("armed_above"))))
        self.table.setItem(row, 4, _check_item(bool(entry.get("armed_below"))))
        self.table.setItem(row, 5, _text_item(str(entry.get("note") or "")))
        self.table.setItem(row, 6, _text_item(last_trigger, editable=False))

    def _table_entries(self) -> list[dict]:
        # History must survive round-trips through the table, so merge the
        # editable cells onto the stored entries keyed by symbol.
        stored = {entry["symbol"]: entry for entry in self.service.entries()}
        entries: list[dict] = []
        for row in range(self.table.rowCount()):
            symbol = _cell_text(self.table.item(row, 0)).upper()
            if not symbol:
                continue
            base = dict(stored.get(symbol) or {})
            base.update(
                {
                    "symbol": symbol,
                    "above": _parse_level(_cell_text(self.table.item(row, 1))),
                    "below": _parse_level(_cell_text(self.table.item(row, 2))),
                    "armed_above": _cell_checked(self.table.item(row, 3)),
                    "armed_below": _cell_checked(self.table.item(row, 4)),
                    "note": _cell_text(self.table.item(row, 5)),
                }
            )
            entries.append(base)
        return entries

    def _on_cell_changed(self, _row: int, _column: int) -> None:
        if self._loading or self.read_only:
            return
        self._save_timer.start()

    def _save_table(self) -> None:
        if self._loading or self.read_only:
            return
        self.service.save_entries(self._table_entries())
        self._load_table()
        self._refresh_status()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _add_row(self) -> None:
        if self.read_only:
            return
        self._loading = True
        try:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._set_row(
                row,
                {"symbol": "", "above": None, "below": None, "armed_above": True, "armed_below": True, "note": ""},
            )
        finally:
            self._loading = False

    def _remove_selected(self) -> None:
        if self.read_only:
            return
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._loading = True
        try:
            for row in rows:
                self.table.removeRow(row)
        finally:
            self._loading = False
        self._save_table()

    def _rearm_all(self) -> None:
        if self.read_only:
            return
        entries = self._table_entries()
        for entry in entries:
            entry["armed_above"] = entry.get("above") is not None
            entry["armed_below"] = entry.get("below") is not None
        self.service.save_entries(entries)
        self._load_table()
        self._refresh_status()

    def _test_push(self) -> None:
        if self.read_only:
            return
        self._save_push_settings()
        result = self.service.test_push()
        if result.get("ok"):
            self.status_label.setText("Test push sent - check the phone (and watch).")
        else:
            self.status_label.setText(f"Test push FAILED: {result.get('error') or 'unknown'}")

    def _test_wake_push(self) -> None:
        """Same fail-quiet contract, at the priority that matters overnight."""
        if self.read_only:
            return
        self._save_push_settings()
        result = self.service.test_push(urgent=True)
        if result.get("ok"):
            self.status_label.setText(
                "Wake alert sent at urgent priority - it should have sounded "
                "THROUGH Sleep Focus. If it did not, ntfy is not allowed in "
                "iOS Settings > Focus > Sleep, or the topic is set to Deliver "
                "Quietly."
            )
        else:
            self.status_label.setText(
                f"Wake alert FAILED: {result.get('error') or 'unknown'}"
            )

    def _save_push_settings(self) -> None:
        if self.read_only:
            return
        save_local_setting(PUSH_SERVER_SETTING, self.server_input.text().strip() or DEFAULT_NTFY_SERVER)
        save_local_setting(PUSH_TOPIC_SETTING, self.topic_input.text().strip())
        save_local_setting(PUSH_TOKEN_SETTING, self.token_input.text().strip())
        save_local_setting(ALWAYS_ON_SETTING, bool(self.always_on_check.isChecked()))
        self._refresh_status()

    def _on_triggered(self, message: str) -> None:
        self.status_label.setText(f"ALERT: {message}")
        self._load_table()

    def _refresh_status(self) -> None:
        snapshot = self.service.status_snapshot()
        if self.read_only:
            self.status_label.setText(
                "Read-only. Monitoring, edits, and phone pushes are disabled for this view."
            )
            return
        push_state = "configured" if snapshot.get("push_configured") else "NOT configured (set a topic)"
        checked = snapshot.get("last_check_at") or "never"
        note = snapshot.get("note") or ""
        error = snapshot.get("push_error") or ""
        error_text = f" | push error: {error}" if error else ""
        self.status_label.setText(f"Push: {push_state} | last check: {checked} | {note}{error_text}")

    def shutdown(self) -> None:
        if self._owns_service:
            self.service.shutdown()


def _cell_text(item: QTableWidgetItem | None) -> str:
    return str(item.text()).strip() if item is not None else ""


def _cell_checked(item: QTableWidgetItem | None) -> bool:
    return item is not None and item.checkState() == Qt.CheckState.Checked


def _parse_level(text: str) -> float | None:
    text = text.replace("$", "").replace(",", "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _level_text(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"
