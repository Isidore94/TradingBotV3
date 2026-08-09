"""Chart Review: the workspace the trader lives in, built around capture.

Layout follows the one constraint that matters - the chart gets the room and
the capture rail is always reachable:

    [ lookup | recents | Setups ]
    [ setups drawer (hidden) | chart area | capture rail ]

The Master AVWAP setups panel is **not** shown here by default. The trader's
reason: AVWAP setups matter less early in the day, and the chart matters more.
The "Setups" button slides a read-only summary in and out. That drawer reads
the setup tracker file on open and nothing else - no service, no timer, no
second owner of anything - so showing or hiding it cannot change what the
scanners find or what the alerting does.

Looking up a symbol never writes a watchlist (see ui.services.symbol_lookup).
Nothing on this page mutes, suppresses, scores, gates, or alerts (plan.md
sec 5); the rail records, and that is all.

CHART AREA: not yet wired. The chart data path is being rebuilt off the GUI
thread (ui.services.chart_data_service / bar_cache), and drawing here against
the old synchronous path would have meant a second chart loader and a
guaranteed conflict. The placeholder states that plainly rather than showing
an empty frame that looks broken. Everything else on this page is live.
"""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from project_paths import MASTER_AVWAP_SETUP_TRACKER_FILE
from ui import theme
from ui.services.symbol_lookup import RecentLookups, normalize_symbol
from ui.widgets.capture_rail import CaptureRail
from ui.widgets.flow_layout import FlowLayout


class ChartReviewPanel(QFrame):
    """Ticker lookup + chart area + capture rail, with a Setups drawer."""

    #: Emitted when the focused symbol changes. Read-only broadcast: no
    #: consumer of this signal may add the symbol to a watchlist.
    symbolChanged = Signal(str)

    def __init__(
        self,
        *,
        focus_service: Any = None,
        recent_lookups: RecentLookups | None = None,
        annotations_path: Any = None,
        setup_tracker_path: Any = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ChartReviewPanel")
        self._recents = recent_lookups if recent_lookups is not None else RecentLookups()
        self._setup_tracker_path = setup_tracker_path or MASTER_AVWAP_SETUP_TRACKER_FILE
        self._symbol = ""

        self.capture_rail = CaptureRail(
            focus_service=focus_service, annotations_path=annotations_path
        )
        self._build()
        self._bind_shortcuts()
        self._render_recents()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(*(theme.px(8),) * 4)
        outer.setSpacing(theme.px(8))
        outer.addWidget(self._lookup_bar())

        body = QHBoxLayout()
        body.setSpacing(theme.px(8))
        body.addWidget(self._setups_drawer())
        body.addWidget(self._chart_area(), 1)

        self.capture_rail.setFixedWidth(theme.px(268))
        self.capture_rail.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        body.addWidget(self.capture_rail)
        outer.addLayout(body, 1)

    def _lookup_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("ChartReviewLookupBar")
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(*(theme.px(8),) * 4)
        layout.setSpacing(theme.px(6))

        row = QHBoxLayout()
        row.setSpacing(theme.px(6))
        self.lookup_input = QLineEdit()
        self.lookup_input.setPlaceholderText("Look up any symbol  (Ctrl+L)")
        self.lookup_input.setClearButtonEnabled(True)
        self.lookup_input.returnPressed.connect(self._on_lookup_submitted)
        row.addWidget(self.lookup_input, 1)

        open_button = QPushButton("Open")
        open_button.clicked.connect(self._on_lookup_submitted)
        row.addWidget(open_button)

        self.setups_button = QPushButton("Setups")
        self.setups_button.setCheckable(True)
        self.setups_button.setChecked(False)
        self.setups_button.setToolTip("Show/hide the setups drawer (Alt+E). Display only.")
        self.setups_button.toggled.connect(self.set_setups_visible)
        row.addWidget(self.setups_button)
        layout.addLayout(row)

        self.lookup_status = QLabel("")
        self.lookup_status.setWordWrap(True)
        layout.addWidget(self.lookup_status)

        recents_row = QFrame()
        self._recents_layout = FlowLayout(recents_row)
        self._recents_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(recents_row)
        return bar

    def _setups_drawer(self) -> QFrame:
        self.setups_drawer = QFrame()
        self.setups_drawer.setObjectName("SetupsDrawer")
        self.setups_drawer.setFixedWidth(theme.px(250))
        layout = QVBoxLayout(self.setups_drawer)
        layout.setContentsMargins(*(theme.px(8),) * 4)
        layout.setSpacing(theme.px(6))
        heading = QLabel("Setups (read-only)")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)
        self.setups_body = QLabel("")
        self.setups_body.setWordWrap(True)
        self.setups_body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setups_body.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.setups_body, 1)
        self.setups_drawer.setVisible(False)
        return self.setups_drawer

    def _chart_area(self) -> QFrame:
        area = QFrame()
        area.setObjectName("ChartReviewChartArea")
        layout = QVBoxLayout(area)
        layout.setContentsMargins(*(theme.px(12),) * 4)
        layout.setSpacing(theme.px(6))

        self.chart_symbol_label = QLabel("-")
        self.chart_symbol_label.setObjectName("TitleLabel")
        layout.addWidget(self.chart_symbol_label)

        # Provenance is a permanent fixture of this area, not a chart detail:
        # the workspace must always be able to say where its numbers came from
        # and how old they are. It reads "no feed" until the chart lands.
        self.provenance_label = QLabel("Feed: none - chart not yet wired")
        self.provenance_label.setObjectName("FeedProvenance")
        layout.addWidget(self.provenance_label)

        placeholder = QLabel(
            "Charts arrive with the off-GUI-thread chart data path.\n\n"
            "The capture rail on the right is live now: look up a symbol and "
            "the veto / like / stop / note actions all record."
        )
        placeholder.setWordWrap(True)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder, 1)
        return area

    def _bind_shortcuts(self) -> None:
        for sequence, handler in (
            ("Ctrl+L", self.focus_lookup),
            ("Alt+E", self.toggle_setups),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(handler)

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------
    def focus_lookup(self) -> None:
        self.lookup_input.setFocus()
        self.lookup_input.selectAll()

    def _on_lookup_submitted(self) -> None:
        self.open_symbol(self.lookup_input.text())

    def open_symbol(self, text: object) -> str:
        """Focus a symbol for review. Returns "" when it is not a symbol.

        Read-only: this records the name in a machine-local recents list and
        points the rail at it. It does not add it to any watchlist, focus
        list, or the CandidateRegistry.
        """
        symbol = normalize_symbol(text)
        if not symbol:
            self.lookup_status.setText(f"Not a symbol: {str(text or '').strip()!r}")
            return ""
        self._symbol = symbol
        self.lookup_input.setText(symbol)
        self.lookup_status.setText("")
        self.chart_symbol_label.setText(symbol)
        self._recents.remember(symbol)
        self._render_recents()
        self.capture_rail.set_context(symbol=symbol)
        self.symbolChanged.emit(symbol)
        return symbol

    @property
    def symbol(self) -> str:
        return self._symbol

    def recent_symbols(self) -> list[str]:
        return self._recents.symbols()

    def _render_recents(self) -> None:
        while self._recents_layout.count():
            item = self._recents_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        for symbol in self._recents.symbols():
            chip = QPushButton(symbol)
            chip.setObjectName("RecentLookupChip")
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.clicked.connect(lambda _checked=False, name=symbol: self.open_symbol(name))
            self._recents_layout.addWidget(chip)

    # ------------------------------------------------------------------
    # setups drawer (display only)
    # ------------------------------------------------------------------
    def toggle_setups(self) -> bool:
        self.setups_button.setChecked(not self.setups_button.isChecked())
        return self.setups_button.isChecked()

    def set_setups_visible(self, visible: bool) -> None:
        visible = bool(visible)
        self.setups_drawer.setVisible(visible)
        if self.setups_button.isChecked() != visible:
            self.setups_button.blockSignals(True)
            self.setups_button.setChecked(visible)
            self.setups_button.blockSignals(False)
        if visible:
            self.setups_body.setText(self._setups_summary())

    def setups_visible(self) -> bool:
        # isHidden(), not isVisible(): a child of a window that has not been
        # shown yet reports isVisible() False even when it is set to show, so
        # isVisible() would answer a question about the window, not the drawer.
        return not self.setups_drawer.isHidden()

    def _setups_summary(self) -> str:
        """A read of the tracker file, on open. No service, no timer."""
        try:
            payload = json.loads(
                self._setup_tracker_path.read_text(encoding="utf-8")
                if hasattr(self._setup_tracker_path, "read_text")
                else open(self._setup_tracker_path, encoding="utf-8").read()
            )
        except (OSError, json.JSONDecodeError, TypeError):
            return "Setup tracker not readable from here."
        setups = payload.get("setups") if isinstance(payload, dict) else None
        if not isinstance(setups, dict) or not setups:
            return "No tracked setups."
        updated = str(payload.get("updated_at") or "unknown")
        lines = [f"as of {updated}", ""]
        for symbol in sorted(setups)[:40]:
            entry = setups.get(symbol)
            family = ""
            if isinstance(entry, dict):
                family = str(entry.get("setup_family") or entry.get("family") or "")
            lines.append(f"{symbol}  {family}".rstrip())
        if len(setups) > 40:
            lines.append(f"... and {len(setups) - 40} more")
        return "\n".join(lines)
