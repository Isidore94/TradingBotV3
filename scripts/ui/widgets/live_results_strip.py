"""The "Working now" line at the top of the M5 alerts column.

Trader, 2026-09-22 (change #3): instant feedback on which trades are working
right now. One line - today's M5 alerts grouped by their setup grade, how many,
and their average R since they fired - and a tooltip with one row per alert.

**Display only.** It records nothing, withholds nothing and changes no alert,
detector or score. The math is `live_alert_results` (pure). The bars come from
a provider the desk hands in - the bot's CACHED `m5_chart_bars`, never an IB
fetch - and are read on this widget's own worker thread, because on the
process-proxy bot each read is a pipe round trip and fifty of them do not
belong on the Qt thread. One worker at a time; the widget owns it and its
60-second timer, which runs only while the strip is on screen.

The variant is a dynamic property in `theme.qss`, never a per-tick stylesheet.

P1-6 6a: the same worker also reads each alert's entry state (`entry_state`:
valid / improved / gone / unknown) - for the first alert per name and side
(this strip's tooltip) and for the NEWEST one (the M5 bar's row chip, sent out
as `entryStatesChanged`).
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

import entry_state as es
import live_alert_results as lar
from swallowed import note_swallowed

REFRESH_MS = 60_000

BarsProvider = Callable[[str], Any]


def _now_ny() -> datetime:
    return datetime.now(tz=lar.NY)


class LiveResultsStrip(QFrame):
    """One line, one tooltip. Nothing else lives here."""

    #: (generation, results) from the worker - queued onto the Qt thread.
    _resultsReady = Signal(int, object)
    #: `{(SYMBOL, SIDE): entry_state dict}` for the NEWEST alert per name+side (6a).
    entryStatesChanged = Signal(object)

    def __init__(self, parent=None, *, threaded: bool = True, clock: Callable[[], datetime] | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LiveResultsStrip")
        self._threaded = bool(threaded)
        self._clock = clock or _now_ny
        self._book = lar.AlertBook()
        self._grade_keys: dict[tuple[str, str, Any], str] = {}
        self._grades: dict[str, Any] = {}
        self._provider: BarsProvider | None = None
        self._results: list[dict[str, Any]] = []
        #: The newest alert per (symbol, side), and its state (6a).
        self._latest: dict[tuple[str, str], dict[str, Any]] = {}
        self._first_states: list[dict[str, Any]] = []
        self._latest_states: dict[tuple[str, str], dict[str, Any]] = {}
        self._generation = 0
        self._busy = False
        self._pending = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)
        self.line_label = QLabel()
        self.line_label.setObjectName("LiveResultsLine")
        self.line_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        layout.addWidget(self.line_label, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._resultsReady.connect(self._on_results)
        self._render()

    # ---------------------------------------------------------------- inputs
    def record(self, alert: Any) -> None:
        """Take one posted M5 alert. The first per symbol+side per day counts."""
        try:
            from ui.models.bounce import is_regime_pause_alert

            if is_regime_pause_alert(alert):
                return
        except Exception as exc:  # noqa: BLE001 - a display strip never costs the desk
            note_swallowed("regime pause check failed for a live result", exc, quiet=True)
        received_at = self._clock()
        latest = lar.entry_from_alert(alert, received_at)
        if latest is not None:
            self._latest[(latest["symbol"], latest["side"])] = latest
        if not self._book.add(alert, received_at):
            if latest is not None and self.isVisible():
                self.refresh()
            return
        entry = self._book.entries()[-1]
        self._grade_keys[self._key(entry)] = self._grade_key(alert)
        # Shown at once as "no data" until the first bars are read.
        self._results.append(lar.result_for(entry, (), received_at, naive_zone=lar.NY))
        self._render()
        if self.isVisible():
            self.refresh()

    def clear_day(self) -> None:
        """The M5 day rolled: start the book again. Stale worker results are dropped."""
        self._book.clear()
        self._grade_keys.clear()
        self._results = []
        self._latest.clear()
        self._first_states = []
        self._latest_states = {}
        self._generation += 1
        self._render()
        self.entryStatesChanged.emit({})

    def set_setup_grades(self, payload: Any) -> None:
        import setup_grades

        self._grades = setup_grades.daytrade_lookup(payload)
        self._render()

    def set_bars_provider(self, provider: BarsProvider | None) -> None:
        self._provider = provider

    # --------------------------------------------------------------- refresh
    def refresh(self) -> None:
        """Re-measure every entry. The bar reads run off the Qt thread."""
        entries = self._book.entries()
        if not entries:
            return
        latest = dict(self._latest)
        if not self._threaded:
            self._on_results(
                self._generation, self._measure(entries, latest, self._provider, self._clock())
            )
            return
        if self._busy:
            self._pending = True
            return
        self._busy = True
        worker = threading.Thread(
            target=self._work,
            args=(self._generation, entries, latest, self._provider, self._clock()),
            name="live-results-strip",
            daemon=True,
        )
        worker.start()

    def _work(self, generation: int, entries, latest, provider, now) -> None:
        try:
            results = self._measure(entries, latest, provider, now)
        except Exception:  # noqa: BLE001 - never let the worker die silently busy
            results = None
        try:
            self._resultsReady.emit(generation, results)
        except RuntimeError as exc:  # widget already destroyed at shutdown
            note_swallowed("live results after the widget was destroyed", exc, quiet=True)

    @staticmethod
    def _bars_for(bars_by_symbol: dict[str, Any], entry, provider) -> list:
        """One cached-bar read per symbol per pass; no bars is "no data"."""
        symbol = entry["symbol"]
        if symbol not in bars_by_symbol:
            bars = []
            if provider is not None and entry.get("status") == lar.OK:
                try:
                    bars = list(provider(symbol) or ())
                except Exception:  # noqa: BLE001 - no bars is "no data", never an error
                    bars = []
            bars_by_symbol[symbol] = bars
        return bars_by_symbol[symbol]

    @classmethod
    def _compute(cls, entries, provider, now) -> list[dict[str, Any]]:
        return cls._measure(entries, {}, provider, now)["results"]

    @classmethod
    def _measure(cls, entries, latest, provider, now) -> dict[str, Any]:
        """Results and entry states for the first alerts; states for the newest."""
        zone = lar.desk_zone()
        bars_by_symbol: dict[str, Any] = {}
        results, first_states = [], []
        for entry in entries:
            bars = cls._bars_for(bars_by_symbol, entry, provider)
            results.append(lar.result_for(entry, bars, now, naive_zone=zone))
            first_states.append(es.entry_state(entry, bars, now, naive_zone=zone))
        latest_states = {
            key: es.entry_state(
                entry, cls._bars_for(bars_by_symbol, entry, provider), now, naive_zone=zone
            )
            for key, entry in (latest or {}).items()
        }
        return {"results": results, "first_states": first_states, "latest_states": latest_states}

    def _on_results(self, generation: int, measured: Any) -> None:
        self._busy = False
        if generation == self._generation and measured is not None:
            results = list(measured["results"])
            self._first_states = list(measured["first_states"])
            if measured["latest_states"] != self._latest_states:
                self._latest_states = dict(measured["latest_states"])
                self.entryStatesChanged.emit(dict(self._latest_states))
            # An alert recorded while the worker ran keeps its "no data" row
            # until the next read, rather than vanishing for a minute.
            for entry in self._book.entries()[len(results):]:
                results.append(lar.result_for(entry, (), entry["received_at"], naive_zone=lar.NY))
            self._results = results
            self._render()
        if self._pending:
            self._pending = False
            self.refresh()

    # ---------------------------------------------------------------- render
    @staticmethod
    def _key(entry: dict[str, Any]) -> tuple[str, str, Any]:
        return (entry["symbol"], entry["side"], entry["received_at"].date())

    @staticmethod
    def _grade_key(alert: Any) -> str:
        """The bounce types the grade is read from - `M5AlertBar._grade_for`'s rule."""
        payload = getattr(alert, "payload", None)
        feedback = payload.get("feedback") if isinstance(payload, dict) else None
        types = str((feedback or {}).get("bounce_types") or "")
        if not types:
            try:
                import working_lately

                types = str(working_lately.alert_priority_key(alert)[0] or "")
            except Exception:  # noqa: BLE001
                types = ""
        return types

    def _graded(self) -> list[dict[str, Any]]:
        import setup_grades

        rows = []
        for row in self._results:
            types = self._grade_keys.get(self._key(row), row.get("bounce_types", ""))
            cell = setup_grades.daytrade_cell_for_alert(self._grades, types, row.get("side"))
            grade = str(cell.get("grade") or setup_grades.NEW) if cell else setup_grades.NEW
            rows.append(
                {**row, "grade": grade, "grade_line": setup_grades.cell_line(cell) if cell else ""}
            )
        return rows

    def results(self) -> list[dict[str, Any]]:
        return self._graded()

    def entry_states(self) -> dict[tuple[str, str], dict[str, Any]]:
        """The newest alert's entry state per (symbol, side) - 6a."""
        return dict(self._latest_states)

    def _states_for(self, rows) -> list[Any]:
        """The first alert's entry state per row; None until it is measured."""
        states = list(self._first_states[: len(rows)])
        return states + [None] * (len(rows) - len(states))

    def line_text(self) -> str:
        return self.line_label.text()

    def _render(self) -> None:
        rows = self._graded()
        text = lar.strip_text(lar.summarize(rows))
        takeable = sum(
            1 for state in self._states_for(rows) if es.chip_text(state) in (es.VALID, es.IMPROVED)
        )
        if takeable:
            text += f" · {takeable} entry valid"
        if rows:
            tooltip = "\n".join(
                [
                    "How each of today's M5 alerts has done since it fired, in R",
                    "(completed M5 bars only; first alert per name and side).",
                    "",
                ]
                + [
                    f"{lar.tooltip_line(row)} · {es.chip_detail(state)}"
                    + (f"\n    grade {row['grade_line']}" if row.get("grade_line") else "")
                    for row, state in zip(rows, self._states_for(rows), strict=False)
                ]
            )
        else:
            tooltip = "No M5 alerts yet today."
        if self.line_label.text() != text:
            self.line_label.setText(text)
        if self.toolTip() != tooltip:
            self.line_label.setToolTip(tooltip)
            self.setToolTip(tooltip)
        has_rows = bool(rows)
        if self.property("hasRows") != has_rows:
            self.setProperty("hasRows", has_rows)
            style = self.style()
            style.unpolish(self)
            style.polish(self)

    # ---------------------------------------------------------------- timer
    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()
        self.refresh()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt's name
        self._timer.stop()
        super().hideEvent(event)

    def timer_active(self) -> bool:
        return self._timer.isActive()
