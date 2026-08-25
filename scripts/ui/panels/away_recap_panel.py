"""The AWAY day recap — R1 amendment 2026-08-24, decision record §5.

The return surface for a day the trader was not here. It replaces a 317-item
chart-review queue with one page: what the day produced, ranked the way the day
already ranked it, plus the Focus picks that needed a decision.

**It writes nothing.** Every mutation belongs to an existing owner:
`FocusService` adds, the Alert Center performs removals, and
`MarketJournalService` takes the after-the-fact D1 write-up. A trader-entered
add carries no auto-pick marker and is therefore trader-owned and never
auto-removed — that is structural, not a promise this page makes.

The R2 adoption-gate state is SURFACED at click time the way the Strength Board
already does, and **never blocks a trader action**: the gate governs the
machine's adoptions, not the trader's.

Every read is on a worker (ground rule 9).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class _RecapWorker(QThread):
    """Assembles the recap off the GUI thread."""

    loaded = Signal(dict)

    def __init__(self, session_date: str, alerts: list, parent=None) -> None:
        super().__init__(parent)
        self._session = session_date
        self._alerts = alerts

    def run(self) -> None:  # pragma: no cover - exercised through its signal seam
        import away_recap

        unavailable: dict[str, str] = {}
        swings: list[str] = []
        staged: dict[str, Any] = {}
        focus: dict[str, Any] = {}
        try:
            from project_paths import AUTOPILOT_REPORT_FILE

            swings = away_recap.digest_swing_lines(
                open(AUTOPILOT_REPORT_FILE, encoding="utf-8").read()
            )
        except Exception as exc:  # noqa: BLE001
            unavailable["autopilot_today.txt"] = str(exc)
        try:
            from autopilot_core import load_auto_populate_pending_picks

            pending = load_auto_populate_pending_picks().get("pending") or {}
            staged = {
                side: sorted(str(sym).upper() for sym in (pending.get(side) or {}))
                for side in ("long", "short")
            }
        except Exception as exc:  # noqa: BLE001
            unavailable["staged picks"] = str(exc)
        try:
            from focus_picks import load_focus_map

            focus = {
                side: sorted(load_focus_map(side) or [])
                for side in ("long", "short")
            }
        except Exception as exc:  # noqa: BLE001
            unavailable["Focus lists"] = str(exc)

        self.loaded.emit(
            away_recap.build_recap(
                session_date=self._session,
                alerts=self._alerts,
                staged_picks=staged,
                digest_swings=swings,
                focus_picks=focus,
                unavailable=unavailable,
            )
        )


class AwayRecapPanel(QFrame):
    """What the day produced, for a trader who was not watching it."""

    statusChanged = Signal(str)
    #: (symbol, side) - the host performs the Focus add through FocusService.
    focusAddRequested = Signal(str, str)
    symbolActivated = Signal(str)

    def __init__(self, focus_service=None, journal_service=None, parent=None) -> None:
        super().__init__(parent)
        self._focus_service = focus_service
        self._journal_service = journal_service
        self._worker: _RecapWorker | None = None
        self._alerts: list[dict[str, Any]] = []
        self._recap: dict[str, Any] = {}

        self.heading = QLabel("AWAY day recap")
        self.heading.setObjectName("SectionTitle")
        self.subtitle = QLabel(
            "An AWAY day ends here, not in a queue. This is what the day "
            "produced, ranked the way the day already ranked it - nothing on "
            "this page computes a new order."
        )
        self.subtitle.setObjectName("SectionSubtitle")
        self.subtitle.setWordWrap(True)

        self.session_picker = QComboBox()
        self.session_picker.setEditable(True)
        self.session_picker.addItem(date.today().isoformat())
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.reload)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.swings = QTableWidget(0, 4)
        self.swings.setHorizontalHeaderLabels(["#", "Symbol", "Side", "Line"])
        self.swings.setEditTriggers(QTableWidget.NoEditTriggers)
        self.staged = QTableWidget(0, 3)
        self.staged.setHorizontalHeaderLabels(["Symbol", "Side", "Gate at click time"])
        self.staged.setEditTriggers(QTableWidget.NoEditTriggers)
        self.focus_table = QTableWidget(0, 2)
        self.focus_table.setHorizontalHeaderLabels(["Symbol", "Side"])
        self.focus_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.add_button = QPushButton("Add selected staged pick to Focus")
        self.add_button.clicked.connect(self._add_selected)
        self.gate_note = QLabel("")
        self.gate_note.setWordWrap(True)

        self.journal_text = QPlainTextEdit()
        self.journal_text.setPlaceholderText(
            "The day's D1 analysis, written now. It is filed under the session "
            "it is about and stamped with when you actually wrote it."
        )
        self.journal_button = QPushButton("Save journal entry for this session")
        self.journal_button.clicked.connect(self._write_journal)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        header = QHBoxLayout()
        header.addWidget(QLabel("Session"))
        header.addWidget(self.session_picker, 1)
        header.addWidget(self.refresh_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.heading)
        layout.addWidget(self.subtitle)
        layout.addLayout(header)
        layout.addWidget(self.summary)
        layout.addWidget(QLabel("Best swing trades (as the day ranked them)"))
        layout.addWidget(self.swings, 2)
        layout.addWidget(QLabel("Staged picks - never adopted while AWAY"))
        layout.addWidget(self.staged, 2)
        layout.addWidget(self.add_button)
        layout.addWidget(self.gate_note)
        layout.addWidget(QLabel("Focus picks that needed managing"))
        layout.addWidget(self.focus_table, 1)
        layout.addWidget(QLabel("Write the day up"))
        layout.addWidget(self.journal_text, 1)
        layout.addWidget(self.journal_button)
        layout.addWidget(self.status)

    # -- data -------------------------------------------------------------
    def set_alerts(self, alerts: list[dict[str, Any]]) -> None:
        """The day's alerts, handed in by the host that already holds them.

        Handed in rather than read: the Alert Center's backing list IS the
        record, and a second reader would be a second definition of what the
        day produced.
        """
        self._alerts = list(alerts or [])

    def session_date(self) -> str:
        return self.session_picker.currentText().strip() or date.today().isoformat()

    def reload(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._worker = _RecapWorker(self.session_date(), self._alerts, self)
        self._worker.loaded.connect(self._render)
        self._worker.start()

    def _render(self, recap: dict) -> None:
        self._recap = recap
        self.summary.setText(recap.get("summary", ""))
        self._fill(
            self.swings,
            [
                (str(row["rank"]), row["symbol"], row["side"], row["text"])
                for row in recap.get("best_swings") or []
            ],
        )
        self._fill(
            self.staged,
            [(row["symbol"], row["side"], "") for row in recap.get("staged_picks") or []],
        )
        self._fill(
            self.focus_table,
            [(row["symbol"], row["side"]) for row in recap.get("focus_to_manage") or []],
        )

    @staticmethod
    def _fill(table: QTableWidget, rows: list[tuple]) -> None:
        table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, value in enumerate(row):
                table.setItem(index, column, QTableWidgetItem(str(value)))

    # -- actions (delegated; this page owns no store) ----------------------
    def _selected_staged(self) -> tuple[str, str] | None:
        index = self.staged.currentRow()
        rows = self._recap.get("staged_picks") or []
        if index < 0 or index >= len(rows):
            self.status.setText("select a staged pick first")
            return None
        row = rows[index]
        return row["symbol"], row["side"]

    def _add_selected(self) -> None:
        """Adopt one staged pick. The GATE IS SHOWN, never enforced.

        The R2 adoption gate governs the MACHINE's adoptions. A trader clicking
        this button is making their own decision, and a surface that blocked it
        on a gate would be substituting the machine's judgement for theirs -
        which is the opposite of what the gate is for. The Strength Board set
        this pattern: re-check at click time, show the verdict, act anyway.
        """
        selected = self._selected_staged()
        if selected is None:
            return
        symbol, side = selected
        self.gate_note.setText(self._gate_text(symbol, side))
        if self._focus_service is None:
            # The host owns the Focus store; this page only asks.
            self.focusAddRequested.emit(symbol, side)
            self.status.setText(f"asked the desk to add {symbol} ({side}) to Focus")
            return
        try:
            self._focus_service.add(symbol, side, "swing")
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"could not add {symbol}: {exc}")
            return
        self.status.setText(
            f"added {symbol} ({side}) to swing Focus - trader-owned, so nothing "
            "will auto-remove it"
        )

    def _gate_text(self, symbol: str, side: str) -> str:
        try:
            from focus_adoption_gate import mover_state

            state, reason = mover_state(side, None, None, None)
        except Exception as exc:  # noqa: BLE001
            return f"adoption gate unavailable ({exc}); your action is unaffected"
        return (
            f"R2 adoption gate for {symbol}: {state} ({reason}). "
            "Shown for context - it governs the machine's adoptions, not yours."
        )

    def _write_journal(self) -> None:
        """Route the write-up to R10.H's service. Never backdated."""
        if self._journal_service is None:
            self.status.setText("the market journal service is not available here")
            return
        result = self._journal_service.write_entry(
            text=self.journal_text.toPlainText(),
            session_date=self.session_date(),
            timeframe="D1",
            origin="away_recap",
        )
        if result.get("ok"):
            self.journal_text.clear()
        else:
            self.status.setText(str(result.get("reason") or "entry not saved"))

    def shutdown(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(2000)
