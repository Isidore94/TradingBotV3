"""The shared header above every Journal sub-tab (R7 §9 step 11, spec §7).

Three controls, one signal. The account tree, the currency toggle and the date
range apply to whichever tab is showing, so a number the trader reads on the
Analytics tab is the same selection they were just looking at on Trades - which
is the whole reason this is one header and not three.

I6 lives here: the tree is grouped by tax treatment, and a selection that spans
groups gets a badge. It is a badge and not a refusal - the trader is allowed to
look at everything at once, and is not allowed to do it by accident.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ui.services import journal_feed

DATE_PRESETS = ("7d", "30d", "QTD", "YTD", "All", "Custom")
CURRENCY_MODES = ("CAD", "Native", "USD")


class JournalHeader(QFrame):
    """Account selection, currency and date range, shared by every sub-tab."""

    selectionChanged = Signal()

    def __init__(self, parent: QWidget | None = None, *, autoload: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("JournalHeader")
        self._accounts: list[tuple[str, str]] = []
        self._selected: set[tuple[str, str]] = set()
        self._loading = False

        self.account_button = QPushButton("All accounts")
        self.account_menu = QMenu(self)
        self.account_button.setMenu(self.account_menu)

        self.blend_badge = QLabel("")
        self.blend_badge.setObjectName("BlendBadge")
        self.blend_badge.setVisible(False)

        self.currency_input = QComboBox()
        self.currency_input.addItems(CURRENCY_MODES)
        self.currency_input.currentTextChanged.connect(self._on_currency_changed)

        # Manual USD display rate. Booked CAD conversion is point-in-time and
        # automatic (journal_fx); this is the separate, deliberately manual
        # knob that lets a MIXED selection show a USD total at all. It is an
        # estimate - it never touches fx_rates or net_pnl_cad - so it only
        # appears in USD mode and says so in its own tooltip.
        self.usd_rate_input = QLineEdit()
        self.usd_rate_input.setPlaceholderText("USD/CAD")
        self.usd_rate_input.setMaximumWidth(90)
        self.usd_rate_input.setToolTip(
            "Estimate only. Non-USD trades are converted from their booked CAD "
            "value at this one rate, not at each trade's own booked rate. "
            "Never a tax figure; leave blank to refuse mixed USD totals."
        )
        self.usd_rate_input.editingFinished.connect(self._on_usd_rate_entered)
        self.usd_rate_status = QLabel("")
        self.usd_rate_status.setObjectName("UsdRateStatus")
        self._load_usd_rate()

        self.range_input = QComboBox()
        self.range_input.addItems(DATE_PRESETS)
        self.range_input.setCurrentText("30d")
        self.range_input.currentTextChanged.connect(self._on_range_changed)

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Symbol")
        self.symbol_input.editingFinished.connect(self._emit_changed)
        self.status_input = QComboBox()
        self.status_input.addItems(["All", "OPEN", "CLOSED_PARTIAL", "CLOSED"])
        self.status_input.currentTextChanged.connect(self._emit_changed)
        self.direction_input = QComboBox()
        self.direction_input.addItems(["All", "LONG", "SHORT"])
        self.direction_input.currentTextChanged.connect(self._emit_changed)

        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        for widget in (self.date_from, self.date_to):
            widget.setDisplayFormat("yyyy-MM-dd")
            widget.setVisible(False)
            widget.dateChanged.connect(self._emit_changed)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Accounts"))
        row.addWidget(self.account_button)
        row.addWidget(self.blend_badge)
        row.addWidget(QLabel("Symbol"))
        row.addWidget(self.symbol_input)
        row.addWidget(QLabel("Status"))
        row.addWidget(self.status_input)
        row.addWidget(QLabel("Direction"))
        row.addWidget(self.direction_input)
        row.addStretch(1)
        row.addWidget(QLabel("Currency"))
        row.addWidget(self.currency_input)
        row.addWidget(self.usd_rate_input)
        row.addWidget(self.usd_rate_status)
        row.addWidget(QLabel("Range"))
        row.addWidget(self.range_input)
        row.addWidget(self.date_from)
        row.addWidget(self.date_to)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.addLayout(row)

        if autoload:
            self.refresh_accounts()

    # -- manual USD display rate -------------------------------------------
    def _load_usd_rate(self) -> None:
        from journal_fx import manual_usd_rate

        stored = manual_usd_rate()
        if stored:
            self.usd_rate_input.setText(f"{stored['rate_cad_per_usd']:.4f}")
            stamp = str(stored.get("entered_at") or "")[:10]
            self.usd_rate_status.setText(f"est. rate set {stamp}" if stamp else "est. rate set")
        else:
            self.usd_rate_status.setText("")
        self._sync_usd_rate_visibility()

    def _sync_usd_rate_visibility(self) -> None:
        showing = self.currency_mode.upper() == "USD"
        self.usd_rate_input.setVisible(showing)
        self.usd_rate_status.setVisible(showing and bool(self.usd_rate_status.text()))

    def _on_currency_changed(self, _text: str = "") -> None:
        self._sync_usd_rate_visibility()
        self._emit_changed()

    def _on_usd_rate_entered(self) -> None:
        from journal_fx import set_manual_usd_rate

        try:
            stored = set_manual_usd_rate(self.usd_rate_input.text().strip())
        except ValueError as exc:
            # Refused, not stored. A silently accepted 13.5 would rescale every
            # USD total by an order of magnitude and look like a real number.
            self.usd_rate_status.setText(str(exc))
            self.usd_rate_status.setVisible(True)
            return
        if stored is None:
            self.usd_rate_input.setText("")
            self.usd_rate_status.setText("")
        else:
            self.usd_rate_status.setText(f"est. rate set {str(stored['entered_at'])[:10]}")
        self._sync_usd_rate_visibility()
        self._emit_changed()

    # -- accounts ----------------------------------------------------------

    def refresh_accounts(self) -> None:
        """Rebuild the tree from the feed, keeping whatever is still selectable."""
        self._loading = True
        try:
            tree = journal_feed.account_tree()
        except Exception as exc:
            self.account_menu.clear()
            failed = self.account_menu.addAction(f"Journal unavailable: {exc}")
            failed.setEnabled(False)
            self._accounts = []
            self._selected = set()
            self._loading = False
            self.account_button.setText("Journal unavailable")
            return
        self.account_menu.clear()
        self._accounts = []
        checkboxes: list[tuple[tuple[str, str], QWidget]] = []

        for group in tree:
            heading = self.account_menu.addSection(group["label"])
            if heading is not None:
                heading.setEnabled(False)
            for account in group["accounts"]:
                key = (str(account.get("broker") or "").upper(), str(account.get("account_number") or ""))
                self._accounts.append(key)
                label = str(account.get("account_label") or key[1]) or key[1]
                action = self.account_menu.addAction(f"{key[0]} {label} ({key[1]})")
                action.setCheckable(True)
                action.setChecked(not self._selected or key in self._selected)
                action.toggled.connect(lambda checked, k=key: self._on_account_toggled(k, checked))
                checkboxes.append((key, action))

        if not self._accounts:
            empty = self.account_menu.addAction("No accounts imported yet")
            empty.setEnabled(False)
        if not self._selected:
            self._selected = set(self._accounts)
        else:
            self._selected &= set(self._accounts)
        self._loading = False
        self._update_account_summary()

    def _on_account_toggled(self, key: tuple[str, str], checked: bool) -> None:
        if checked:
            self._selected.add(key)
        else:
            self._selected.discard(key)
        self._update_account_summary()
        if not self._loading:
            self.selectionChanged.emit()

    def _update_account_summary(self) -> None:
        total = len(self._accounts)
        chosen = len(self._selected)
        if not total:
            self.account_button.setText("No accounts")
        elif chosen == total:
            self.account_button.setText(f"All accounts ({total})")
        elif chosen == 1:
            only = next(iter(self._selected))
            self.account_button.setText(f"{only[0]} {only[1]}")
        else:
            self.account_button.setText(f"{chosen} of {total} accounts")

        blended = False
        if self._selected:
            try:
                blended = journal_feed.selection_spans_tax_groups(self._selected)
            except Exception:
                blended = False
        # I6: never silently. The badge is the "explicit" the invariant asks for.
        self.blend_badge.setText("Blended: taxable + tax-free" if blended else "")
        self.blend_badge.setVisible(blended)

    # -- range -------------------------------------------------------------

    def _on_range_changed(self, preset: str) -> None:
        custom = str(preset) == "Custom"
        self.date_from.setVisible(custom)
        self.date_to.setVisible(custom)
        self._emit_changed()

    def _emit_changed(self, *_args) -> None:
        if not self._loading:
            self.selectionChanged.emit()

    # -- what the tabs ask for --------------------------------------------

    @property
    def currency_mode(self) -> str:
        return self.currency_input.currentText()

    @property
    def selected_accounts(self) -> list[tuple[str, str]] | None:
        """None means "everything", which is not the same as "nothing selected".

        A tab handed an empty list would show an empty journal, which is exactly
        what the trader sees after unchecking the last account - and that is
        correct, so the two cases stay distinguishable.
        """
        if not self._accounts or self._selected == set(self._accounts):
            return None
        return sorted(self._selected)

    def date_bounds(self):
        preset = self.range_input.currentText()
        if preset == "Custom":
            return (
                self.date_from.date().toPython(),
                self.date_to.date().toPython(),
            )
        return journal_feed.date_range_bounds(preset)

    def query(self) -> dict:
        """The one dict every tab passes to ``journal_feed.load_trades``."""
        date_from, date_to = self.date_bounds()
        return {
            "date_from": date_from,
            "date_to": date_to,
            "accounts_filter": self.selected_accounts,
            "symbol": self.symbol_input.text().strip(),
            "status": self.status_input.currentText(),
            "direction": self.direction_input.currentText(),
        }
