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

from PySide6.QtCore import Signal

from ui.timer_utils import SignalCoalescer
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
)

from ui.services import journal_feed

DATE_PRESETS = ("7d", "30d", "QTD", "YTD", "All", "Custom")
CURRENCY_MODES = ("CAD", "Native", "USD")


#: The filter header's reload window. Long enough that walking a four-account
#: menu with the mouse is one query, short enough that a single tick still feels
#: immediate.
JOURNAL_FILTER_COALESCE_MS = 250


class JournalHeader(QFrame):
    """Account selection, currency and date range, shared by every sub-tab."""

    selectionChanged = Signal()

    def __init__(self, parent: QWidget | None = None, *, autoload: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("JournalHeader")
        self._accounts: list[tuple[str, str]] = []
        self._selected: set[tuple[str, str]] = set()
        self._loading = False
        # Every checkbox, combo and date widget in this header re-queries the
        # journal, and ticking four accounts is four full reloads - each one a
        # `list_trades` over the whole store. The window is the house pattern:
        # leading-edge, folding, and unable to starve, so a burst of toggles is
        # ONE reload and it is never more than the window late.
        self._change_coalescer = SignalCoalescer(
            self.selectionChanged.emit, JOURNAL_FILTER_COALESCE_MS, self
        )

        self.account_button = QPushButton("All accounts")
        self.account_menu = QMenu(self)
        self.account_button.setMenu(self.account_menu)

        self.blend_badge = QLabel("")
        self.blend_badge.setObjectName("BlendBadge")
        self.blend_badge.setVisible(False)

        self.currency_input = QComboBox()
        self.currency_input.addItems(CURRENCY_MODES)
        self.currency_input.currentTextChanged.connect(self._on_currency_changed)

        # Manual USD display rate. Since 2026-08-24 a mixed selection is
        # normally converted from each trade's OWN booked session rate
        # (JournalStore.book_currency_values), so this is now the FALLBACK for a
        # selection holding a session with no booked BoC observation at all. It
        # is an estimate - it never touches fx_rates, net_pnl_cad or
        # net_pnl_usd - so it only appears in USD mode and says so.
        self.usd_rate_input = QLineEdit()
        self.usd_rate_input.setPlaceholderText("USD/CAD")
        self.usd_rate_input.setMaximumWidth(90)
        self.usd_rate_input.setToolTip(
            "Fallback estimate. Trades are normally shown in USD at each "
            "trade's own booked Bank of Canada rate; this one rate is used only "
            "when a selected session has no booked observation. Never a tax "
            "figure; leave blank to refuse a total that cannot be booked."
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

        # Tag lives on the SHARED header rather than on the Trades tab, so
        # "show me only my gap-and-go trades" narrows the calendar, the equity
        # curve and the fee totals too. Analytics could already group BY tag;
        # nothing could filter TO one.
        self.tag_input = QComboBox()
        self.tag_input.setMinimumWidth(140)
        self.tag_input.setToolTip(
            "Filter to one tag. Lists tags you typed and, for trades you have "
            "not tagged, the automatic ones."
        )
        self.tag_input.currentTextChanged.connect(self._emit_changed)
        self.tag_input.addItem("All")

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
        row.addWidget(QLabel("Tag"))
        row.addWidget(self.tag_input)
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
            self._reload_tags()
        # Building and populating the widgets fires their own change signals.
        # Those used to emit synchronously - before anyone had connected - and
        # were harmless; a coalesced one would land AFTER construction and read
        # as a real filter change. Nothing is owed when the header is born.
        self._change_coalescer.cancel()

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
            self._change_coalescer.cancel()
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
        # The tree was just rebuilt from the feed, so anything the rebuild's own
        # widget signals asked for is already answered.
        self._change_coalescer.cancel()

    def _on_account_toggled(self, key: tuple[str, str], checked: bool) -> None:
        if checked:
            self._selected.add(key)
        else:
            self._selected.discard(key)
        self._update_account_summary()
        # Ticking several accounts in a row is one reload, not one per tick.
        self._emit_changed()

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
        """Ask for a reload, coalesced. See `_change_coalescer`."""
        if not self._loading:
            self._change_coalescer.request()

    def flush_pending_change(self) -> None:
        """Run an owed reload now. For tests and for a deliberate hurry."""
        self._change_coalescer.flush()

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

    def _reload_tags(self) -> None:
        """Repopulate the tag picker, keeping the current choice if it survives.

        A rename or a tag-refresh changes what exists, and a picker still
        offering a tag nobody carries filters every tab to zero rows with no
        explanation. Selection is restored by NAME, so renaming the selected
        tag drops the filter rather than silently pointing at the old word.
        """
        previous = self.tag_input.currentText() if self.tag_input.count() else "All"
        blocked = self.tag_input.blockSignals(True)
        try:
            self.tag_input.clear()
            self.tag_input.addItem("All")
            for name in journal_feed.tag_names():
                self.tag_input.addItem(name)
            index = self.tag_input.findText(previous)
            self.tag_input.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.tag_input.blockSignals(blocked)

    def refresh_tags(self) -> None:
        """Reload the tag picker and re-run the tabs if the selection moved."""
        before = self.tag_input.currentText()
        self._reload_tags()
        if self.tag_input.currentText() != before:
            self._emit_changed()

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
            "tag": self.tag_input.currentText() or "All",
        }
