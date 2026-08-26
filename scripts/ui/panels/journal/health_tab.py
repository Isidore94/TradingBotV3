"""Health: coverage, reconciliation, FX, the nightly slot, and broker sync (§9 step 13).

This is the tab that answers the trader's actual question - *is the journal
complete?* - and it answers it with the ledger rather than with a reassuring
number. A red cell in the coverage grid is a day nobody imported; a NEEDS_REVIEW
row is a position the broker disagrees about; an unconverted count is money that
cannot honestly be added up yet.

It is also where A1 and A9 are closed: the Qt panel had no IBKR control, no Flex
fields and no backfill button at all, so the only complete import path in the
system was a CLI the trader never ran.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from journal_importers import (
    IBKR_FLEX_QUERY_ID_SETTING,
    IBKR_FLEX_TOKEN_SETTING,
    QUESTRADE_REFRESH_TOKEN_SETTING,
    mask_secret,
)
from project_paths import get_local_setting, save_local_setting
from ui.services import journal_feed

STATUS_COLOURS = {
    "COVERED": "#1f6f3f",
    "FAILED": "#8c2f2f",
    "PENDING": "#7a6a1f",
    "NO_SESSION": "#3a3a3a",
}


class _JournalTask(QThread):
    """Broker work never runs on the GUI thread."""

    finished_with = Signal(dict)
    failed = Signal(str)

    def __init__(self, action: str, parent=None) -> None:
        super().__init__(parent)
        self.action = action

    def run(self) -> None:  # pragma: no cover - exercised on the desk
        try:
            if self.action == "pull":
                result = journal_feed.pull_today()
            else:
                failed_only = self.action == "failed"
                # "Retry failed Questrade days" is an explicit trader decision,
                # so it reaches days that burned their attempt budget while the
                # credential chain was broken. "Backfill gaps" keeps the cap.
                result = journal_feed.self_heal_gaps(
                    failed_only=failed_only, include_exhausted=failed_only
                )
            self.finished_with.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class HealthTab(QFrame):
    statusChanged = Signal(str)

    def __init__(self, header, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._header = header
        self._task: _JournalTask | None = None
        self._suggestions: list[dict] = []

        # AI-P4: a broken Questrade chain explains everything below it, so it
        # goes above everything below it. Hidden entirely when there is nothing
        # to say - a permanently-present "all good" strip is furniture, and the
        # trader stops seeing furniture.
        self.chain_banner = QFrame()
        self.chain_banner.setObjectName("BrokerChainBanner")
        self.chain_label = QLabel("")
        self.chain_label.setObjectName("CautionLabel")
        self.chain_label.setWordWrap(True)
        self.chain_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        banner_layout = QVBoxLayout(self.chain_banner)
        banner_layout.setContentsMargins(10, 8, 10, 8)
        banner_layout.addWidget(self.chain_label)
        self.chain_banner.setVisible(False)

        self.coverage_table = QTableWidget(0, 0)
        self.coverage_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.gaps_label = QLabel("")
        self.gaps_label.setWordWrap(True)

        self.pull_button = QPushButton("Pull today now")
        self.pull_button.clicked.connect(lambda: self._start_task("pull"))
        self.backfill_button = QPushButton("Backfill Questrade gaps")
        self.backfill_button.clicked.connect(lambda: self._start_task("gaps"))
        self.retry_button = QPushButton("Retry failed Questrade days")
        self.retry_button.clicked.connect(lambda: self._start_task("failed"))

        self.reconcile_label = QLabel("No reconciliation has run yet.")
        self.reconcile_label.setWordWrap(True)
        self.suggestions_list = QListWidget()
        self.confirm_button = QPushButton("Confirm selected force-close")
        self.confirm_button.clicked.connect(self._confirm_suggestion)

        self.fx_label = QLabel("")
        self.slot_list = QListWidget()
        self.runs_table = QTableWidget(0, 5)
        self.runs_table.setHorizontalHeaderLabels(["Source", "Status", "Account", "Span", "Message"])
        self.runs_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # A1/A9: the controls the Qt panel never had.
        self.flex_token_input = QLineEdit()
        self.flex_token_input.setEchoMode(QLineEdit.Password)
        self.flex_query_input = QLineEdit()
        self.questrade_token_input = QLineEdit()
        self.questrade_token_input.setEchoMode(QLineEdit.Password)
        self.save_credentials_button = QPushButton("Save broker credentials")
        self.save_credentials_button.clicked.connect(self._save_credentials)
        self.env_warning = QLabel("")
        self.env_warning.setWordWrap(True)
        self.env_warning.setVisible(False)

        heal_row = QHBoxLayout()
        heal_row.addWidget(self.pull_button)
        heal_row.addWidget(self.backfill_button)
        heal_row.addWidget(self.retry_button)
        heal_row.addStretch(1)

        sync_row = QHBoxLayout()
        sync_row.addWidget(QLabel("Flex token"))
        sync_row.addWidget(self.flex_token_input)
        sync_row.addWidget(QLabel("Query id"))
        sync_row.addWidget(self.flex_query_input)
        sync_row.addWidget(QLabel("Questrade refresh token"))
        sync_row.addWidget(self.questrade_token_input)
        sync_row.addWidget(self.save_credentials_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.chain_banner)
        layout.addWidget(QLabel("Coverage"))
        layout.addWidget(self.coverage_table, 2)
        layout.addWidget(self.gaps_label)
        layout.addLayout(heal_row)
        layout.addWidget(QLabel("Reconciliation"))
        layout.addWidget(self.reconcile_label)
        layout.addWidget(self.suggestions_list, 1)
        layout.addWidget(self.confirm_button)
        layout.addWidget(QLabel("Currency"))
        layout.addWidget(self.fx_label)
        layout.addWidget(QLabel("Nightly journal_import"))
        layout.addWidget(self.slot_list, 1)
        layout.addWidget(QLabel("Import runs"))
        layout.addWidget(self.runs_table, 1)
        layout.addWidget(QLabel("Broker sync"))
        layout.addLayout(sync_row)
        layout.addWidget(self.env_warning)

    # -- loading -----------------------------------------------------------

    def reload(self) -> None:
        self._load_chain_health()
        self._load_coverage()
        self._load_reconciliation()
        self._load_fx()
        self._load_slot()
        self._load_runs()
        self._load_credentials()

    def _load_chain_health(self) -> None:
        """Show the Questrade credential chain only when it needs a hand.

        Best-effort by design: this tab's job is showing the journal's health,
        and a check that could not run must not stop it. A failure here leaves
        the banner hidden and says so in the status line rather than raising -
        but it never renders "fine", because it does not know that.
        """
        try:
            import journal_health

            verdict = journal_health.questrade_chain_health()
        except Exception as exc:  # noqa: BLE001
            self.chain_banner.setVisible(False)
            self.statusChanged.emit(f"Questrade chain check unavailable: {exc}")
            return
        if verdict["state"] not in journal_health.ALERTING_STATES:
            self.chain_banner.setVisible(False)
            return
        self.chain_label.setText(
            "\n\n".join([verdict["headline"], verdict["action"]])
        )
        self.chain_banner.setVisible(True)

    def _load_coverage(self) -> None:
        grid = journal_feed.coverage_grid(days=30)
        days = grid.get("days") or []
        accounts = grid.get("accounts") or {}
        self.coverage_table.setRowCount(len(accounts))
        self.coverage_table.setColumnCount(len(days))
        self.coverage_table.setHorizontalHeaderLabels([day[5:] for day in days])
        self.coverage_table.setVerticalHeaderLabels(
            [f"{broker} {number}" for broker, number in accounts]
        )
        for row, key in enumerate(accounts):
            statuses = accounts[key]
            for column, day in enumerate(days):
                status = statuses.get(day, "")
                item = QTableWidgetItem("" if status in {"COVERED", "NO_SESSION"} else status[:1])
                item.setToolTip(f"{day}: {status or 'no import has covered this day'}")
                colour = STATUS_COLOURS.get(status)
                if colour:
                    from PySide6.QtGui import QColor

                    item.setBackground(QColor(colour))
                self.coverage_table.setItem(row, column, item)

        gaps = journal_feed.find_coverage_gaps(days=365)
        if not gaps:
            self.gaps_label.setText("No gaps in the last 365 days.")
        else:
            total = sum(len(entry["days"]) for entry in gaps)
            detail = "; ".join(
                f"{entry['broker']} {entry['account_number']}: {len(entry['days'])} day(s)"
                for entry in gaps
            )
            self.gaps_label.setText(f"{total} uncovered session day(s) - {detail}")

    def _load_reconciliation(self) -> None:
        report = journal_feed.last_reconciliation()
        self.suggestions_list.clear()
        self._suggestions = []
        if not report:
            self.reconcile_label.setText("No reconciliation has run yet.")
            return
        mismatched = report.get("mismatched") or []
        self.reconcile_label.setText(
            f"Checked {report.get('positions_checked', 0)} position(s) at "
            f"{report.get('checked_at', '')}: {len(mismatched)} mismatch(es)."
        )
        suggestions = {
            str(item.get("group_key") or ""): item
            for item in report.get("suggestions") or []
        }
        for mismatch in mismatched:
            row = QListWidgetItem(
                f"{mismatch.get('kind')} {mismatch.get('broker')} {mismatch.get('symbol')}: "
                f"journal {mismatch.get('journal_quantity')} vs broker {mismatch.get('broker_quantity')}"
            )
            suggestion = suggestions.get(str(mismatch.get("group_key") or ""))
            row.setData(Qt.UserRole, suggestion)
            self.suggestions_list.addItem(row)

    def _load_fx(self) -> None:
        coverage = journal_feed.fx_coverage()
        unconverted = coverage.get("unconverted") or []
        if not unconverted:
            self.fx_label.setText(
                f"{coverage.get('converted', 0)} of {coverage.get('trades', 0)} trades converted; "
                f"{coverage.get('booked_rates', 0)} rate(s) booked."
            )
            return
        detail = ", ".join(f"{row['trades']} {row['currency']}" for row in unconverted)
        self.fx_label.setText(
            f"{coverage.get('converted', 0)} of {coverage.get('trades', 0)} converted. "
            f"Unconverted: {detail}. These are excluded from cross-currency totals rather than "
            "counted as zero."
        )

    def _load_slot(self) -> None:
        status = journal_feed.nightly_slot_status()
        self.slot_list.clear()
        if not status.get("available"):
            self.slot_list.addItem("The AI jobs ledger is not readable from here.")
            return
        rows = status.get("rows") or []
        if not rows:
            self.slot_list.addItem("The nightly slot has not run yet.")
        for row in rows:
            self.slot_list.addItem(
                f"{row.get('session_date')} {row.get('status')} - {row.get('detail') or row.get('message') or ''}"
            )

    def _load_runs(self) -> None:
        runs = journal_feed.list_import_runs(limit=25)
        self.runs_table.setRowCount(len(runs))
        for row, record in enumerate(runs):
            span = f"{record.get('coverage_start') or ''}..{record.get('coverage_end') or ''}".strip(".")
            for column, text in enumerate(
                [
                    record.get("source"),
                    record.get("status"),
                    record.get("account_number"),
                    span,
                    record.get("message"),
                ]
            ):
                self.runs_table.setItem(row, column, QTableWidgetItem(str(text or "")))

    def _load_credentials(self) -> None:
        import os

        self.flex_token_input.setPlaceholderText(
            mask_secret(str(get_local_setting(IBKR_FLEX_TOKEN_SETTING, "") or "")) or "not set"
        )
        self.flex_query_input.setText(str(get_local_setting(IBKR_FLEX_QUERY_ID_SETTING, "") or ""))
        self.questrade_token_input.setPlaceholderText(
            mask_secret(str(get_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, "") or "")) or "not set"
        )
        # A8: the env var is a first-boot seed only, and a stale one silently
        # loses the rotated single-use token on the next refresh.
        stale = [name for name in ("QUESTRADE_REFRESH_TOKEN",) if os.environ.get(name)]
        if stale:
            self.env_warning.setText(
                f"{', '.join(stale)} is still set in the environment. Local settings win, but the "
                "env var is a first-boot seed only - clear it so a stale copy cannot be mistaken "
                "for the live rotating token."
            )
        self.env_warning.setVisible(bool(stale))

    # -- actions -----------------------------------------------------------

    def _save_credentials(self) -> None:
        saved = []
        if self.flex_token_input.text().strip():
            save_local_setting(IBKR_FLEX_TOKEN_SETTING, self.flex_token_input.text().strip())
            saved.append("Flex token")
        if self.flex_query_input.text().strip():
            save_local_setting(IBKR_FLEX_QUERY_ID_SETTING, self.flex_query_input.text().strip())
            saved.append("query id")
        if self.questrade_token_input.text().strip():
            save_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, self.questrade_token_input.text().strip())
            saved.append("Questrade token")
        self.flex_token_input.clear()
        self.questrade_token_input.clear()
        self.statusChanged.emit(f"saved {', '.join(saved)}" if saved else "nothing to save")
        self._load_credentials()

    def _start_task(self, action: str) -> None:  # pragma: no cover - worker path
        if self._task is not None and self._task.isRunning():
            return
        self.pull_button.setEnabled(False)
        self.backfill_button.setEnabled(False)
        self.retry_button.setEnabled(False)
        labels = {"pull": "pulling today...", "gaps": "backfilling Questrade gaps...",
                  "failed": "retrying failed Questrade days..."}
        self.statusChanged.emit(labels[action])
        self._task = _JournalTask(action, self)
        self._task.finished_with.connect(self._on_heal_done)
        self._task.failed.connect(self._on_heal_failed)
        self._task.start()

    def _on_heal_done(self, summary: dict) -> None:  # pragma: no cover
        self.pull_button.setEnabled(True)
        self.backfill_button.setEnabled(True)
        self.retry_button.setEnabled(True)
        if "source_results" in summary:
            message = "; ".join(summary.get("messages") or []) or summary.get("status", "pull complete")
        else:
            message = (
                f"repaired {len(summary.get('repaired') or [])}, "
                f"failed {len(summary.get('failed') or [])}, "
                f"skipped {len(summary.get('exhausted') or [])}"
            )
            reopened = int(summary.get("reopened_exhausted") or 0)
            if reopened:
                message += f"; {reopened} day(s) reopened past the attempt cap"
            if "IBKR" in (summary.get("unsupported_brokers") or []):
                message += "; IBKR historical gaps require a Flex backfill"
        self.statusChanged.emit(message)
        self.reload()

    def _on_heal_failed(self, message: str) -> None:  # pragma: no cover
        self.pull_button.setEnabled(True)
        self.backfill_button.setEnabled(True)
        self.retry_button.setEnabled(True)
        self.statusChanged.emit(f"gap repair failed: {message}")

    def _confirm_suggestion(self) -> None:
        item = self.suggestions_list.currentItem()
        suggestion = item.data(Qt.UserRole) if item is not None else None
        if not isinstance(suggestion, dict):
            self.statusChanged.emit("select a mismatch that offers a force-close")
            return
        confirmed = QMessageBox.question(
            self,
            "Force-close this position?",
            f"{suggestion.get('reason')}\n\n"
            "This records an append-only correction. It books no profit or loss unless you "
            "supply a price - the system does not know what the position was worth.",
        )
        if confirmed != QMessageBox.Yes:
            return
        try:
            journal_feed.confirm_reconciliation_suggestion(
                suggestion, reason=str(suggestion.get("reason") or "confirmed from reconciliation")
            )
            journal_feed.rebuild_trades()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Refused", str(exc))
            return
        self.statusChanged.emit("force-close recorded")
        self.reload()

    def shutdown(self) -> None:
        task = self._task
        if task is not None and task.isRunning():
            task.wait(2000)
