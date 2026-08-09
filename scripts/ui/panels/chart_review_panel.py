"""Chart Review: the workspace the trader lives in, built around capture.

Layout follows the one constraint that matters - the chart gets the room and
the capture rail is always reachable:

    [ lookup | recents | Setups ]
    [ setups drawer (hidden) | chart area | capture rail ]

The Master AVWAP setups panel is **not** shown here by default. The trader's
reason: AVWAP setups matter less early in the day, and the chart matters more.
The "Setups" button slides a read-only summary in and out. The drawer reads
the compact tracker SCORING SNAPSHOT (~11.5MB, refused past a ceiling), never
the raw setup tracker - that file was measured at 762MB on the live desk and
must never be read unbounded (ai_summary learned this the hard way). The read
happens on a pool thread and lands by signal, so opening the drawer cannot
stall the GUI; it runs on open and nothing else - no service, no timer, no
second owner of anything - so showing or hiding it cannot change what the
scanners find or what the alerting does.

Looking up a symbol never writes a watchlist (see ui.services.symbol_lookup).
Nothing on this page mutes, suppresses, scores, gates, or alerts (plan.md
sec 5); the rail records, and that is all.

The chart area embeds the same SymbolSnapshotWidget used elsewhere: one
ChartDataService worker path, one CandleChart implementation, and no file or
provider reads on the paint path. This workspace disables the shared widget's
alert menus; painted-level clicks feed annotation provenance only.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
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

from project_paths import MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE
from ui import theme
from ui.services.symbol_lookup import RecentLookups, normalize_symbol
from ui.widgets.capture_rail import CaptureRail
from ui.widgets.flow_layout import FlowLayout
from ui.widgets.symbol_snapshot_dialog import REFRESH_INTERVAL_MS, SymbolSnapshotWidget

#: The snapshot was measured at 11.5MB; this refuses one that has grown into
#: the raw tracker's problem instead of parsing it anyway (same ceiling as
#: ai_summary's tracker extract).
MAX_SETUPS_SNAPSHOT_BYTES = 64 * 1024 * 1024
#: Rows shown in the drawer, newest scan date first.
SETUPS_DRAWER_ROWS = 40
# Two NYSE years are about 504 sessions. Keep a little more than that so the
# requested D1 view is genuinely 2y+ rather than a calendar approximation.
CHART_REVIEW_D1_SESSIONS = 520


def provenance_state(
    meta: dict[str, Any], *, now: datetime | None = None
) -> tuple[str, bool]:
    """Human-readable feed/bar-age strip and whether it is degraded.

    The source and timestamp were assembled on the chart worker. This helper
    formats only those values; it performs no probing or I/O on the GUI path.
    """
    source = str((meta or {}).get("source") or "none").strip().lower()
    labels = {
        "ibkr-cache": "IBKR live cache",
        "yfinance-fallback": "YFINANCE FALLBACK",
        "durable-store": "durable D1 store",
        "memory": "memory cache",
        "local": "local mirror",
        "shared": "shared store",
    }
    degraded = source in {"yahoo", "yfinance", "yfinance-fallback"}
    label = labels.get(source, source or "none")
    text = f"Feed: {label}"
    tier = str((meta or {}).get("storage_tier") or "").strip()
    if tier and tier != source:
        text += f" · storage {tier}"

    stamp = (meta or {}).get("bar_timestamp")
    if isinstance(stamp, str):
        try:
            stamp = datetime.fromisoformat(stamp)
        except ValueError:
            stamp = None
    if isinstance(stamp, datetime):
        moment = now
        if moment is None:
            moment = datetime.now(tz=stamp.tzinfo) if stamp.tzinfo else datetime.now()
        if stamp.tzinfo is None and moment.tzinfo is not None:
            moment = moment.replace(tzinfo=None)
        elif stamp.tzinfo is not None and moment.tzinfo is None:
            moment = moment.replace(tzinfo=stamp.tzinfo)
        seconds = max(0, int((moment - stamp).total_seconds()))
        if seconds < 3600:
            age = f"{seconds // 60}m"
        elif seconds < 48 * 3600:
            age = f"{seconds // 3600}h"
        else:
            age = f"{seconds // 86400}d"
        timeframe = str((meta or {}).get("bar_timeframe") or "bar")
        text += f" · {timeframe} age {age}"
    else:
        text += " · bar age unknown"
    return text, degraded


def read_setups_summary(
    path: Path | str,
    *,
    max_bytes: int = MAX_SETUPS_SNAPSHOT_BYTES,
    max_rows: int = SETUPS_DRAWER_ROWS,
) -> str:
    """The drawer's text, from the compact tracker scoring snapshot.

    BLOCKING - worker threads (or tests) only. The snapshot's ``setups`` keys
    are stable setup ids (``date:symbol:side:anchor:bucket``), not symbols;
    each row carries its own ``symbol``/``setup_family``/``scan_date``, and
    the drawer shows the newest rows rather than the alphabetically-first
    ids. Never raises: the drawer must degrade to a message, not an error.
    """
    target = Path(path)
    try:
        size = target.stat().st_size
    except OSError:
        return "Setups snapshot not readable from here."
    if size > max_bytes:
        return (
            f"Setups snapshot is {size:,} bytes - past the {max_bytes:,} byte "
            "ceiling; refusing to parse it. The scan that writes it may be "
            "misbehaving."
        )
    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return "Setups snapshot not readable from here."
    setups = payload.get("setups") if isinstance(payload, dict) else None
    if not isinstance(setups, dict) or not setups:
        return "No tracked setups."
    updated = str(
        payload.get("source_updated_at") or payload.get("generated_at") or "unknown"
    )

    def _row_line(setup_id: str, row: Any) -> tuple[str, str]:
        """(sort_key, display_line); the setup id is only a fallback."""
        if not isinstance(row, dict):
            row = {}
        parts = str(setup_id).split(":")
        symbol = str(row.get("symbol") or (parts[1] if len(parts) >= 2 else setup_id))
        family = str(row.get("setup_family") or row.get("tracker_setup_family") or "")
        scan_date = str(row.get("scan_date") or (parts[0] if parts else ""))
        return scan_date, f"{symbol}  {family}  {scan_date}".rstrip()

    lines = sorted(
        (_row_line(setup_id, row) for setup_id, row in setups.items()),
        key=lambda pair: (pair[0], pair[1]),
        reverse=True,
    )
    body = [f"as of {updated}", ""]
    body.extend(line for _, line in lines[:max_rows])
    if len(lines) > max_rows:
        body.append(f"... and {len(lines) - max_rows} more")
    return "\n".join(body)


class _SetupsSummaryBridge(QObject):
    """Carries a worker's summary back to the GUI thread by queued signal.

    Deliberately unparented: a worker may emit after the panel is destroyed,
    and emitting on a dead QObject is an access violation. The panel-side
    connection is severed automatically when the panel goes away; the bridge
    itself stays alive until the task releases it.
    """

    ready = Signal(int, str)


class _SetupsSummaryTask(QRunnable):
    def __init__(self, bridge: _SetupsSummaryBridge, request_id: int, path: Path) -> None:
        super().__init__()
        self._bridge = bridge
        self._request_id = request_id
        self._path = path

    def run(self) -> None:  # noqa: D401 (Qt override)
        text = read_setups_summary(self._path)
        try:
            self._bridge.ready.emit(self._request_id, text)
        except RuntimeError:
            pass


class ChartReviewPanel(QFrame):
    """Ticker lookup + chart area + capture rail, with a Setups drawer."""

    #: Emitted when the focused symbol changes. Read-only broadcast: no
    #: consumer of this signal may add the symbol to a watchlist.
    symbolChanged = Signal(str)

    def __init__(
        self,
        *,
        recent_lookups: RecentLookups | None = None,
        annotations_path: Any = None,
        setups_snapshot_path: Any = None,
        bot_provider: Callable[[], object | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ChartReviewPanel")
        self._recents = recent_lookups if recent_lookups is not None else RecentLookups()
        self._setups_snapshot_path = Path(
            setups_snapshot_path or MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE
        )
        self._symbol = ""
        self._bot_provider = bot_provider
        self._setups_request = 0
        self._setups_bridge = _SetupsSummaryBridge()
        self._setups_bridge.ready.connect(self._on_setups_summary)

        self.capture_rail = CaptureRail(annotations_path=annotations_path)
        self._build()
        self._bind_shortcuts()
        self._render_recents()
        # Same cadence and same refresh method as the existing snapshot popup.
        # This panel owns this timer and runs it only while its page is visible.
        self._chart_refresh_timer = QTimer(self)
        self._chart_refresh_timer.setInterval(REFRESH_INTERVAL_MS)
        self._chart_refresh_timer.timeout.connect(self._refresh_visible_chart)
        self._chart_refresh_timer.start()

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

        # Provenance is a permanent fixture of this area, not a chart detail.
        self.provenance_label = QLabel("Feed: none · bar age unknown")
        self.provenance_label.setObjectName("FeedProvenance")
        layout.addWidget(self.provenance_label)

        self.snapshot = SymbolSnapshotWidget(
            area,
            compact=True,
            d1_sessions=CHART_REVIEW_D1_SESSIONS,
            allow_alerts=False,
        )
        self.snapshot.d1LevelSelected.connect(self._on_d1_level_selected)
        self.snapshot.snapshotMetaChanged.connect(self._on_snapshot_meta)
        layout.addWidget(self.snapshot, 1)
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
        self.snapshot.set_symbol(symbol, bot=self._current_bot())
        self.symbolChanged.emit(symbol)
        return symbol

    def _current_bot(self):
        if self._bot_provider is None:
            return None
        try:
            return self._bot_provider()
        except Exception:
            return None

    def _refresh_visible_chart(self) -> None:
        if not self._symbol or not self.isVisible():
            return
        try:
            self.snapshot.refresh(bot=self._current_bot())
        except Exception:
            pass  # display-only refresh; the next owned tick retries

    def _on_d1_level_selected(
        self, symbol: str, level_id: str, family: str, _price: float
    ) -> None:
        if symbol != self._symbol:
            return
        self.capture_rail.set_context(
            symbol=symbol,
            timeframe="D1",
            ref_level_id=level_id,
            ref_level_family=family,
        )

    def _on_snapshot_meta(self, symbol: str, meta: object) -> None:
        if symbol != self._symbol:
            return
        text, degraded = provenance_state(meta if isinstance(meta, dict) else {})
        self.provenance_label.setText(text)
        self.provenance_label.setProperty("degraded", degraded)
        if degraded:
            self.provenance_label.setStyleSheet(
                f"background: {theme.color('short')}; color: {theme.color('bg_app')}; "
                "font-weight: 800; padding: 6px;"
            )
        else:
            self.provenance_label.setStyleSheet("")

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
            self._refresh_setups_summary()

    def setups_visible(self) -> bool:
        # isHidden(), not isVisible(): a child of a window that has not been
        # shown yet reports isVisible() False even when it is set to show, so
        # isVisible() would answer a question about the window, not the drawer.
        return not self.setups_drawer.isHidden()

    def _refresh_setups_summary(self) -> None:
        """Queue the snapshot read on a pool thread. Never reads on the GUI
        thread: the snapshot lives in the Drive-backed home folder, and a
        cloud-synced read can stall for seconds."""
        self._setups_request += 1
        self.setups_body.setText("Reading setups snapshot...")
        QThreadPool.globalInstance().start(
            _SetupsSummaryTask(
                self._setups_bridge, self._setups_request, self._setups_snapshot_path
            )
        )

    def _on_setups_summary(self, request_id: int, text: str) -> None:
        if request_id == self._setups_request:
            self.setups_body.setText(text)
