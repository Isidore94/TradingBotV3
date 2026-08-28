"""The Trades tab: the plan, the legs, the tags, and the corrections (§9 step 11).

Everything a trade needs a human for lives here. The pieces that are new in R7
are the ones that answer the trader's report directly: an R-multiple that is
computed rather than eyeballed, a legs view that shows *where each fill came
from* (including the synthetic ones the journal had to invent), and a
corrections launcher, because until now there was no way at any layer to tell
the journal it was wrong.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from journal_store import TRADE_SHAPE_SOURCE
from ui.models.journal import JournalTrade
from ui.services import journal_feed

#: Actions the corrections dialog offers, with the wording the trader reads.
CORRECTION_ACTIONS = (
    ("VOID_EXECUTION", "Void an execution the broker sent twice"),
    ("EDIT_EXECUTION", "Correct an execution's price, quantity or fees"),
    ("ADD_EXECUTION", "Add a fill the broker never reported"),
    ("REASSIGN_GROUP", "Move an execution into a different account or symbol"),
    ("FORCE_CLOSE", "Force-close this position"),
)


class _MoneyBox(QDoubleSpinBox):
    def __init__(self, maximum: float = 1_000_000.0) -> None:
        super().__init__()
        self.setRange(0.0, maximum)
        self.setDecimals(4)
        self.setSpecialValueText("")  # 0 reads as "not set", never as a real 0.


class ManualExecutionDialog(QDialog):
    """Enter a fill by hand - into a real broker and account.

    Spec §5 fix 3, the half deferred out of step 3 because the only manual-entry
    dialog lived in the legacy Tk tab. ``broker="MANUAL"`` used to be the default
    and made every hand-entered fill an orphan that could never attach to the
    position it belonged to (B3), so the pickers are the point of this dialog
    rather than a convenience on it.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add execution")
        self.broker_input = QComboBox()
        self.account_input = QComboBox()
        self._accounts = journal_feed.accounts()
        brokers = sorted({str(a.get("broker") or "") for a in self._accounts if a.get("broker")})
        self.broker_input.addItems(brokers or ["QUESTRADE", "IBKR"])
        self.broker_input.currentTextChanged.connect(self._refresh_accounts)

        self.symbol_input = QLineEdit()
        self.security_type_input = QComboBox()
        self.security_type_input.addItems(["STK", "OPT", "FUT", "CASH"])
        self.currency_input = QComboBox()
        self.currency_input.addItems(["USD", "CAD"])
        self.side_input = QComboBox()
        self.side_input.addItems(["BUY", "SELL"])
        self.quantity_input = _MoneyBox(1_000_000.0)
        self.price_input = _MoneyBox()
        self.commission_input = _MoneyBox(100_000.0)
        self.timestamp_input = QLineEdit()
        self.timestamp_input.setPlaceholderText("2026-08-05T09:31:00-07:00")
        self.reason_input = QLineEdit()
        self.reason_input.setPlaceholderText("why this fill is being entered by hand")

        form = QFormLayout()
        form.addRow("Broker", self.broker_input)
        form.addRow("Account", self.account_input)
        form.addRow("Symbol", self.symbol_input)
        form.addRow("Type", self.security_type_input)
        form.addRow("Currency", self.currency_input)
        form.addRow("Side", self.side_input)
        form.addRow("Quantity", self.quantity_input)
        form.addRow("Price", self.price_input)
        form.addRow("Commission", self.commission_input)
        form.addRow("Timestamp", self.timestamp_input)
        form.addRow("Reason", self.reason_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "A hand-entered fill is a real execution: it nets against the position "
                "in the account you choose here."
            )
        )
        layout.addWidget(buttons)
        self._refresh_accounts(self.broker_input.currentText())

    def _refresh_accounts(self, broker: str) -> None:
        self.account_input.clear()
        for account in self._accounts:
            if str(account.get("broker") or "") != str(broker):
                continue
            number = str(account.get("account_number") or "")
            label = str(account.get("account_label") or number)
            self.account_input.addItem(f"{label} ({number})", number)
        if not self.account_input.count():
            self.account_input.addItem("(no imported accounts)", "")

    def fields(self) -> dict:
        return {
            "broker": self.broker_input.currentText(),
            "account_number": self.account_input.currentData() or "",
            "symbol": self.symbol_input.text().strip().upper(),
            "security_type": self.security_type_input.currentText(),
            "currency": self.currency_input.currentText(),
            "side": self.side_input.currentText(),
            "quantity": self.quantity_input.value(),
            "price": self.price_input.value(),
            "commission": self.commission_input.value(),
            "timestamp": self.timestamp_input.text().strip(),
        }

    @property
    def reason(self) -> str:
        return self.reason_input.text().strip()


class CorrectionsDialog(QDialog):
    """Tell the journal it is wrong, on the record.

    Every correction needs a reason and the OK button stays disabled without
    one. That is not politeness: this table is the audit trail behind a tax
    filing, and six months later an unexplained correction is indistinguishable
    from a mistake.
    """

    def __init__(self, trade: JournalTrade, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Correct {trade.symbol}")
        self._trade = trade

        self.action_input = QComboBox()
        for value, label in CORRECTION_ACTIONS:
            self.action_input.addItem(label, value)
        self.action_input.currentIndexChanged.connect(self._refresh_targets)

        self.target_input = QComboBox()
        self.price_input = _MoneyBox()
        self.quantity_input = _MoneyBox(1_000_000.0)
        self.reason_input = QPlainTextEdit()
        self.reason_input.setPlaceholderText("Why is this correction being made?")
        self.reason_input.textChanged.connect(self._refresh_ok)

        form = QFormLayout()
        form.addRow("Action", self.action_input)
        form.addRow("Target", self.target_input)
        form.addRow("Price", self.price_input)
        form.addRow("Quantity", self.quantity_input)
        form.addRow("Reason", self.reason_input)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "Corrections are append-only and are re-applied on every rebuild, so the "
                "next import cannot undo one. Undo adds a superseding record."
            )
        )
        layout.addWidget(self.buttons)
        self._refresh_targets()
        self._refresh_ok()

    def _refresh_targets(self) -> None:
        self.target_input.clear()
        action = self.action_input.currentData()
        if action == "FORCE_CLOSE":
            self.target_input.addItem(
                f"whole position: {self._trade.symbol}", journal_feed.group_key_for(self._trade)
            )
            self.target_input.setEnabled(False)
            return
        self.target_input.setEnabled(True)
        for leg in journal_feed.trade_legs(self._trade.trade_id):
            uid = str(leg.get("execution_uid") or "")
            self.target_input.addItem(
                f"{leg.get('role')} {leg.get('side')} {leg.get('quantity')} @ {leg.get('price')} - {uid}",
                uid,
            )
        if not self.target_input.count():
            self.target_input.addItem("(no legs)", "")

    def _refresh_ok(self) -> None:
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self.reason_input.toPlainText().strip()))

    def payload(self) -> dict:
        action = self.action_input.currentData()
        payload: dict = {}
        if self.price_input.value():
            payload["price"] = self.price_input.value()
        if self.quantity_input.value():
            payload["quantity"] = self.quantity_input.value()
        if action == "ADD_EXECUTION":
            raw = self._trade.raw
            payload.setdefault("broker", raw.get("broker"))
            payload.setdefault("account_number", raw.get("account_number"))
            payload.setdefault("symbol", raw.get("symbol"))
            payload.setdefault("security_type", raw.get("security_type"))
            payload.setdefault("currency", raw.get("currency"))
        return payload

    def request(self) -> dict:
        return {
            "action": self.action_input.currentData(),
            "target_uid": self.target_input.currentData() or "",
            "payload": self.payload(),
            "reason": self.reason_input.toPlainText().strip(),
        }


class TagManagerDialog(QDialog):
    """Rename or retire one tag across every trade that carries it.

    Fixing a typo used to mean retyping the tags field on each affected trade,
    which is why a store ends up carrying "gap-and-go", "gap and go" and
    "gapngo" as three separate setups with a third of the evidence each.

    Only tags the trader typed are offered. A derived tag (``midday``,
    ``swing``) is re-computed from the trade on every refresh, so renaming one
    here would be silently undone by the next rebuild -- the list marks those
    rows and the dialog refuses them, rather than accepting a rename that will
    not survive.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage tags")
        self._entries = journal_feed.tags_in_use()

        self.tag_list = QListWidget()
        for entry in self._entries:
            tag = str(entry.get("tag") or "")
            own = int(entry.get("own") or 0)
            auto = int(entry.get("auto") or 0)
            if entry.get("derived"):
                label = f"{tag}  -  {auto} trade(s), automatic"
            else:
                label = f"{tag}  -  {own} trade(s)"
                if auto:
                    label += f", {auto} automatic"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, tag)
            item.setData(Qt.UserRole + 1, bool(entry.get("derived")) or own == 0)
            self.tag_list.addItem(item)
        self.tag_list.currentItemChanged.connect(self._on_selection_changed)

        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("New name (leave blank to remove the tag)")

        self.note = QLabel("")
        self.note.setWordWrap(True)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Apply")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Your tags"))
        layout.addWidget(self.tag_list)
        layout.addWidget(self.new_name)
        layout.addWidget(self.note)
        layout.addWidget(self.buttons)
        if not self._entries:
            self.note.setText("No tags yet.")

    def _on_selection_changed(self, current: QListWidgetItem | None, _previous=None) -> None:
        derived = bool(current.data(Qt.UserRole + 1)) if current is not None else True
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(current is not None and not derived)
        if current is None:
            self.note.setText("")
        elif derived:
            self.note.setText(
                "This tag is worked out from the trade itself, so renaming it here "
                "would be undone the next time tags refresh. Accept it onto a trade "
                "first if you want to keep your own wording."
            )
        else:
            self.note.setText("")

    def selection(self) -> tuple[str, str]:
        """``(old, new)``. An empty ``new`` means remove the tag."""
        item = self.tag_list.currentItem()
        old = str(item.data(Qt.UserRole)) if item is not None else ""
        return old, self.new_name.text().strip()


class TradesTab(QFrame):
    """The trade table, its detail pane, and everything a human does to a trade."""

    statusChanged = Signal(str)
    dataChanged = Signal()

    def __init__(self, header, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._header = header
        self._trades: list[JournalTrade] = []
        self._current: JournalTrade | None = None

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Date", "Symbol", "Dir", "Status", "Qty", "P&L", "R", "Tags"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        self.detail = self._build_detail()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # -- detail pane -------------------------------------------------------

    def _build_detail(self) -> QWidget:
        self.review_banner = QLabel("")
        self.review_banner.setObjectName("NeedsReviewBanner")
        self.review_banner.setWordWrap(True)
        self.review_banner.setVisible(False)

        self.planned_entry = _MoneyBox()
        self.planned_stop = _MoneyBox()
        self.planned_risk = _MoneyBox(1_000_000.0)
        self.r_readout = QLabel("R: -")
        self.prefill_button = QPushButton("Prefill from alert")
        self.prefill_button.clicked.connect(self._prefill_risk)
        self.save_risk_button = QPushButton("Save plan")
        self.save_risk_button.clicked.connect(self._save_risk)

        risk_form = QFormLayout()
        risk_form.addRow("Planned entry", self.planned_entry)
        risk_form.addRow("Planned stop", self.planned_stop)
        risk_form.addRow("Planned risk", self.planned_risk)
        risk_row = QHBoxLayout()
        risk_row.addWidget(self.r_readout)
        risk_row.addStretch(1)
        risk_row.addWidget(self.prefill_button)
        risk_row.addWidget(self.save_risk_button)

        self.legs_table = QTableWidget(0, 5)
        self.legs_table.setHorizontalHeaderLabels(["Role", "Side", "Qty", "Price", "Source"])
        self.legs_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.auto_tags = QListWidget()
        self.auto_tags.setSelectionMode(QListWidget.ExtendedSelection)
        self.accept_tags_button = QPushButton("Accept selected tags")
        self.accept_tags_button.clicked.connect(self._accept_tags)
        self.accept_all_tags_button = QPushButton("Accept all")
        self.accept_all_tags_button.setToolTip(
            "Accept every suggestion listed for this trade."
        )
        self.accept_all_tags_button.clicked.connect(self._accept_all_tags)
        self.manage_tags_button = QPushButton("Manage tags...")
        self.manage_tags_button.setToolTip(
            "Rename or remove one of your tags across every trade that carries it."
        )
        self.manage_tags_button.clicked.connect(self._open_tag_manager)

        tag_buttons = QHBoxLayout()
        tag_buttons.addWidget(self.accept_tags_button)
        tag_buttons.addWidget(self.accept_all_tags_button)
        tag_buttons.addStretch(1)
        tag_buttons.addWidget(self.manage_tags_button)
        self._tag_buttons_row = tag_buttons

        self.tags_input = QLineEdit()
        self.notes_input = QPlainTextEdit()
        self.save_notes_button = QPushButton("Save tags and notes")
        self.save_notes_button.clicked.connect(self._save_annotation)

        self.review_outcome = QComboBox()
        self.review_outcome.addItems(
            [
                "Followed plan",
                "Good process, bad outcome",
                "Poor entry discipline",
                "Poor exit discipline",
                "Risk or sizing mistake",
                "Other",
            ]
        )
        self.decision_reason = QLineEdit()
        self.decision_reason.setPlaceholderText("Why did you make that decision?")
        self.save_review_button = QPushButton("Save structured review")
        self.save_review_button.clicked.connect(self._save_review)

        self.correct_button = QPushButton("Correct this trade...")
        self.correct_button.clicked.connect(self._open_corrections)
        self.add_execution_button = QPushButton("Add execution...")
        self.add_execution_button.clicked.connect(self._open_manual_execution)
        self.adjustments_list = QListWidget()

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(self.review_banner)
        layout.addWidget(QLabel("Plan and R"))
        layout.addLayout(risk_form)
        layout.addLayout(risk_row)
        layout.addWidget(QLabel("Legs"))
        layout.addWidget(self.legs_table)
        layout.addWidget(QLabel("Suggested tags"))
        layout.addWidget(self.auto_tags)
        layout.addLayout(self._tag_buttons_row)
        layout.addWidget(QLabel("My tags"))
        layout.addWidget(self.tags_input)
        layout.addWidget(self.notes_input)
        layout.addWidget(self.save_notes_button)
        review_form = QFormLayout()
        review_form.addRow("Review outcome", self.review_outcome)
        review_form.addRow("Decision reason", self.decision_reason)
        layout.addLayout(review_form)
        layout.addWidget(self.save_review_button)
        corrections_row = QHBoxLayout()
        corrections_row.addWidget(self.correct_button)
        corrections_row.addWidget(self.add_execution_button)
        layout.addLayout(corrections_row)
        layout.addWidget(QLabel("Corrections on this trade"))
        layout.addWidget(self.adjustments_list)
        return body

    # -- loading -----------------------------------------------------------

    def reload(self) -> None:
        try:
            self._trades = journal_feed.load_trades(**self._header.query())
        except Exception as exc:  # noqa: BLE001 - a broken read is a status line
            self._trades = []
            self.statusChanged.emit(f"could not load trades: {exc}")
        self._populate_table()

    def _populate_table(self) -> None:
        mode = self._header.currency_mode
        self.table.setRowCount(len(self._trades))
        for row, trade in enumerate(self._trades):
            value, label = journal_feed.convert_amount(trade, mode)
            r_value = journal_feed.r_multiple(trade)
            cells = [
                trade.trade_date,
                trade.symbol,
                trade.direction,
                trade.status,
                f"{trade.quantity:g}" if trade.quantity is not None else "",
                # "unconverted" rather than a number: I5 at the render seam.
                f"{value:,.2f} {label}" if value is not None else label or "-",
                f"{r_value:.2f}R" if r_value is not None else "-",
                trade.tags,
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if str(trade.raw.get("reconcile_status") or "") == "NEEDS_REVIEW":
                    item.setToolTip("Does not match the broker's reported position")
                self.table.setItem(row, column, item)
        self.dataChanged.emit()

    def _on_selection_changed(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        if not rows:
            self._current = None
            return
        self._current = self._trades[min(rows)]
        self._show_trade(self._current)

    def _show_trade(self, trade: JournalTrade) -> None:
        raw = trade.raw
        status = str(raw.get("reconcile_status") or "")
        if status == "NEEDS_REVIEW":
            self.review_banner.setText(
                "Needs review: the broker's reported position does not match this trade."
            )
        elif status == "FORCED_CLOSED":
            self.review_banner.setText("Force-closed by a correction, not by a broker fill.")
        self.review_banner.setVisible(bool(status))

        self.planned_entry.setValue(float(raw.get("planned_entry") or 0.0))
        self.planned_stop.setValue(float(raw.get("planned_stop") or 0.0))
        self.planned_risk.setValue(float(raw.get("planned_risk") or 0.0))
        r_value = journal_feed.r_multiple(trade)
        self.r_readout.setText(f"R: {r_value:.2f}" if r_value is not None else "R: - (needs risk and a booked FX rate)")

        legs = journal_feed.trade_legs(trade.trade_id)
        self.legs_table.setRowCount(len(legs))
        for row, leg in enumerate(legs):
            role = str(leg.get("role") or "")
            source = str(leg.get("source") or "")
            if role == "SYNTHETIC_OPEN":
                source = "inferred - opening fill missing"
            elif role == "SYNTHETIC_CLOSE":
                source = "correction"
            for column, text in enumerate(
                [role, leg.get("side"), leg.get("quantity"), leg.get("price"), source]
            ):
                self.legs_table.setItem(row, column, QTableWidgetItem(str(text if text is not None else "")))

        # Suggestions this trade's own tags already carry are dropped, so
        # accepting one removes it from the list instead of leaving it there
        # to be accepted again forever.
        self.auto_tags.clear()
        for candidate in journal_feed.unaccepted_auto_tag_candidates(
            trade.trade_id, raw.get("setup_tags")
        ):
            source = str(candidate.get("source") or "")
            if source.startswith(f"{TRADE_SHAPE_SOURCE}:"):
                # A derived tag is a measured fact, not a guess, and showing it
                # as "100%" beside a scanner match reads as a stronger opinion
                # about the setup than it is.
                label = f"{candidate.get('tag')}  ({source.split(':', 1)[1]})"
            else:
                label = f"{candidate.get('tag')}  ({float(candidate.get('confidence') or 0):.0%})"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, candidate.get("tag"))
            item.setToolTip(str(candidate.get("rationale") or ""))
            self.auto_tags.addItem(item)

        self.tags_input.setText(str(raw.get("setup_tags") or ""))
        self.notes_input.setPlainText(str(raw.get("notes") or ""))
        self.review_outcome.setCurrentIndex(0)
        latest_review = journal_feed.latest_trade_review(trade.trade_id) or {}
        review_payload = latest_review.get("payload") or {}
        outcome = str(review_payload.get("review_outcome") or "")
        if outcome and self.review_outcome.findText(outcome) < 0:
            self.review_outcome.addItem(outcome)
        if outcome:
            self.review_outcome.setCurrentText(outcome)
        self.decision_reason.setText(str(latest_review.get("reason") or ""))

        self.adjustments_list.clear()
        for record in journal_feed.list_adjustments(limit=25):
            if record.get("target_uid") not in {
                journal_feed.group_key_for(trade),
                *[str(leg.get("execution_uid") or "") for leg in legs],
            }:
                continue
            superseded = " (undone)" if record.get("superseded_by") else ""
            self.adjustments_list.addItem(
                f"{record.get('created_at')} {record.get('action')}{superseded} - {record.get('reason')}"
            )

    # -- actions -----------------------------------------------------------

    def _prefill_risk(self) -> None:
        if self._current is None:
            return
        suggestion = journal_feed.suggest_planned_risk(self._current)
        if not suggestion:
            # Declining out loud. A silent no-op would read as "no alert found"
            # when the real answer is "more than one, and I will not guess".
            self.statusChanged.emit(
                "no single armed alert matches this trade - prefill needs an unambiguous match"
            )
            return
        self.planned_entry.setValue(float(suggestion.get("planned_entry") or 0.0))
        self.planned_stop.setValue(float(suggestion.get("planned_stop") or 0.0))
        self.planned_risk.setValue(float(suggestion.get("planned_risk") or 0.0))
        self.statusChanged.emit("prefilled from the matching alert - not saved until you save the plan")

    def _save_risk(self) -> None:
        if self._current is None:
            return
        journal_feed.save_risk_fields(
            self._current.trade_id,
            planned_entry=self.planned_entry.value() or None,
            planned_stop=self.planned_stop.value() or None,
            planned_risk=self.planned_risk.value() or None,
        )
        self.statusChanged.emit("plan saved")
        self.reload()

    def _save_annotation(self) -> None:
        if self._current is None:
            return
        journal_feed.save_annotation(
            self._current.trade_id,
            setup_tags=self.tags_input.text().strip(),
            notes=self.notes_input.toPlainText().strip(),
        )
        self.statusChanged.emit("tags and notes saved")
        self._refresh_header_tags()
        self.reload()

    def _save_review(self) -> None:
        if self._current is None:
            return
        journal_feed.record_trade_review(
            self._current.trade_id,
            review_outcome=self.review_outcome.currentText(),
            decision_reason=self.decision_reason.text().strip(),
            setup_tags=self.tags_input.text().strip(),
            notes=self.notes_input.toPlainText().strip(),
        )
        self.statusChanged.emit("structured review saved")
        self.reload()

    def _accept_tags(self) -> None:
        if self._current is None:
            return
        tags = [item.data(Qt.UserRole) for item in self.auto_tags.selectedItems()]
        self._accept(tags)

    def _accept_all_tags(self) -> None:
        if self._current is None:
            return
        tags = [
            self.auto_tags.item(row).data(Qt.UserRole)
            for row in range(self.auto_tags.count())
        ]
        self._accept(tags)

    def _accept(self, tags: list) -> None:
        if self._current is None:
            return
        wanted = [str(tag).strip() for tag in tags if str(tag or "").strip()]
        if not wanted:
            return
        combined = journal_feed.accept_auto_tags(self._current.trade_id, wanted)
        self.tags_input.setText(combined)
        self.statusChanged.emit(f"accepted {len(wanted)} suggestion(s)")
        self._refresh_header_tags()
        self.reload()

    def _open_tag_manager(self) -> None:
        dialog = TagManagerDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        old, new = dialog.selection()
        if not old:
            return
        if new and new == old:
            return
        if not new:
            confirm = QMessageBox.question(
                self,
                "Remove tag",
                f"Remove '{old}' from every trade that carries it?",
            )
            if confirm != QMessageBox.Yes:
                return
        try:
            changed = journal_feed.rename_tag(old, new)
        except Exception as exc:  # noqa: BLE001
            self.statusChanged.emit(f"tag rename failed: {exc}")
            return
        verb = "renamed" if new else "removed"
        self.statusChanged.emit(f"{verb} '{old}' on {changed} trade(s)")
        self._refresh_header_tags()
        self.reload()
        self.dataChanged.emit()

    def _refresh_header_tags(self) -> None:
        """Keep the shared header's tag picker honest after a tag edit.

        The header owns the filter and this tab owns the writes, so nothing
        else would notice a tag that just came into existence or stopped
        existing. Guarded because the tab is constructed against a stub header
        in tests.
        """
        refresh = getattr(self._header, "refresh_tags", None)
        if callable(refresh):
            refresh()

    def _open_corrections(self) -> None:
        if self._current is None:
            return
        dialog = CorrectionsDialog(self._current, self)
        if dialog.exec() != QDialog.Accepted:
            return
        request = dialog.request()
        try:
            journal_feed.record_adjustment(**request)
            journal_feed.rebuild_trades()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Correction refused", str(exc))
            return
        self.statusChanged.emit(f"correction recorded: {request['action']}")
        self.reload()

    def _open_manual_execution(self) -> None:
        dialog = ManualExecutionDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            journal_feed.add_manual_execution(dialog.fields())
            journal_feed.rebuild_trades()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Execution refused", str(exc))
            return
        self.statusChanged.emit("execution added")
        self.reload()
