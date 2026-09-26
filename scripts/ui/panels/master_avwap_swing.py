"""The swing table half of the Master AVWAP panel (p9, 2026-09-26).

The trader: "I want everything from today that will help me find the best swing trades
in the master avwap table. That is basically my 'swing' table."

A mixin, so `master_avwap_panel.py` stays one screen of wiring: the Long leaders merge,
the regime line, the "Best swing" order, the swing columns' visibility and the ONE
worker that reads the swing context (`ui.services.swing_table_context`). Every file read
happens on that worker; the Qt thread only stats files and formats in-memory payloads.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QCheckBox, QLabel

import project_paths
from swallowed import note_swallowed
from ui.models import swing_columns

#: "Best swing" display order (default off: the scan's own order stays the default).
SETTING_BEST_SWING = "qt_setups_best_swing_sort"
#: The swing columns (default on; the overflow menu hides them all).
SETTING_SWING_COLUMNS = "qt_setups_swing_columns"
#: The compact profile shows only this swing column; the full profile shows every one with values.
COMPACT_SWING_COLUMNS = ("regime_grade",)
#: Its compact width (`B n42` or `untested in this regime`, elided; the tooltip has it all).
SWING_COMPACT_WIDTHS = {"regime_grade": 92}
_SPY = "SPY"
_TIMEFRAMES = ("M5", "M30", "H1", "H4", "D1", "W")


class _SwingContextWorker(QThread):
    """`swing_context.read_swing_context` off the Qt thread. Never raises into Qt."""

    done = Signal(object)

    def __init__(self, data_date: str, parent=None) -> None:
        super().__init__(parent)
        self._data_date = data_date

    def run(self) -> None:  # pragma: no cover - exercised through its seam
        try:
            from ui.services import swing_table_context as swing_context

            payload = swing_context.read_swing_context(self._data_date)
        except Exception:  # noqa: BLE001 - the swing columns stay as they were
            payload = None
        self.done.emit(payload)


def regime_line(regime_payload, gate, spy_payload, *, banner_shown: bool) -> str:
    """``Regime: Uptrend, day 12 · Longs: market working · SPY M5 Up · ... · W Up``.

    The longs part is left out while the longs-off banner already says it (one banner).
    """
    parts = []
    current = (regime_payload or {}).get("current") or None
    if current:
        parts.append(f"Regime: {current.get('label')}, day {current.get('day_count')}")
    elif regime_payload:
        parts.append("Regime: not typed yet")
    if not banner_shown:
        verdict = str(getattr(gate, "verdict", "") or "unknown")
        words = {"yes": "market working", "no": "market not working"}.get(verdict, "market unknown")
        parts.append(f"Longs: {words}")
    readings = ((spy_payload or {}).get("symbols") or {}).get(_SPY) or {}
    if readings:
        from ui.widgets.regime_strip import format_cell

        parts.append(f"{_SPY} " + " ".join(format_cell(tf, readings.get(tf))[0] for tf in _TIMEFRAMES))
    return " · ".join(parts)


class SwingTableMixin:
    """Mixed into `MasterAvwapPanel`; needs its `model`, `proxy`, `table` and `_longs_gate`."""

    def _build_swing_controls(self) -> None:
        self._swing_signature: object = object()
        self._swing_worker = None
        self._swing_pending = False
        self._spy_regime: dict = {}
        self._swing_columns_on = bool(project_paths.get_local_setting(SETTING_SWING_COLUMNS, True))
        self.best_swing_toggle = QCheckBox("Best swing")
        self.best_swing_toggle.setToolTip(
            "Order the table for swing trades: longs last while the market is not working "
            "for longs, then promoted Long leaders and the proven short families by their "
            "grade in the current regime, then the rest in the scan's order. Presentation "
            "only - no row is hidden and no score, bucket or point changes."
        )
        self.best_swing_toggle.setChecked(bool(project_paths.get_local_setting(SETTING_BEST_SWING, False)))
        self.best_swing_toggle.toggled.connect(self._on_best_swing_toggled)
        self.swing_regime_label = QLabel("")
        self.swing_regime_label.setObjectName("SwingRegimeLine")
        self.swing_regime_label.setToolTip(
            "Your structural regime and its day count, whether the market is working for longs, "
            "and SPY's M5..W regime (the desk's regime strip)."
        )

    def swing_columns_enabled(self) -> bool:
        return self._swing_columns_on

    def _on_swing_columns_toggled(self, checked: bool) -> None:
        self._swing_columns_on = bool(checked)
        try:
            project_paths.save_local_setting(SETTING_SWING_COLUMNS, bool(checked))
        except Exception as exc:  # noqa: BLE001 - a preference never costs the table
            note_swallowed("swing columns setting write failed", exc)
        self._reapply_column_profile()

    def _reapply_column_profile(self) -> None:
        profile = self._column_profile or "compact"
        self._column_profile = ""
        self.set_column_profile(profile)

    def _swing_column_hidden(self, key: str, profile: str) -> bool:
        """The swing columns' share of `set_column_profile`: off, compact-only set, or empty."""
        if not self.swing_columns_enabled():
            return True
        if profile == "compact" and key not in COMPACT_SWING_COLUMNS:
            return True
        return not self.model.has_swing_values(key)

    def _on_best_swing_toggled(self, checked: bool) -> None:
        try:
            project_paths.save_local_setting(SETTING_BEST_SWING, bool(checked))
        except Exception as exc:  # noqa: BLE001 - a preference never costs the table
            note_swallowed("best swing setting write failed", exc)
        self._resort_from_source()

    def _resort_from_source(self) -> None:
        source = getattr(self, "_working_lately_source_rows", None)
        if source is not None:
            self.set_rows(list(source))

    def _swing_gate_verdict(self) -> str:
        return str(getattr(getattr(self, "_longs_gate", None), "verdict", "") or "")

    def _merge_long_leaders(self, rows):
        try:
            payload = self.model.swing_context().get("long_setups")
            data_date = getattr(self, "_data_date", "")
            # One grouping per payload, not one per re-sort (set_rows runs on every toggle).
            key = (id(payload), data_date)
            memo = getattr(self, "_leaders_memo", None)
            if memo is None or memo[0] != key or memo[2] is not payload:
                memo = (key, swing_columns.leaders_by_symbol(payload, data_date), payload, {})
                self._leaders_memo = memo
            if len(memo[3]) > 4 * (len(rows) + len(memo[1])):
                memo[3].clear()  # rows from older reports; rebuilt on demand
            return swing_columns.merge_long_leaders(rows, payload, data_date, leaders=memo[1], memo=memo[3])
        except Exception as exc:  # noqa: BLE001 - a leader never costs the scan's rows
            note_swallowed("long leaders not merged", exc)
            return list(rows)

    def _best_swing(self, rows):
        if len(rows) < 2 or not self.best_swing_toggle.isChecked():
            return list(rows)
        return swing_columns.best_swing_order(
            rows, gate_verdict=self._swing_gate_verdict(), regime_payload=self.model.regime_grades()
        )

    # -- the three inputs ------------------------------------------------------
    def set_swing_context(self, payload) -> None:
        """The worker's payload (or a test's). Re-merges the Long leaders from the scan's rows."""
        if not isinstance(payload, dict):
            return
        before = self.model.swing_context()
        self.model.set_swing_context(payload)
        if before.get("long_setups") != payload.get("long_setups"):
            self._resort_from_source()
        else:
            self._reapply_column_profile()

    def set_regime_grades(self, payload) -> None:
        """`setup_grades_by_regime` from the Working-lately snapshot (built on its worker)."""
        before = self.model.regime_grades()
        self.model.set_regime_grades(payload or {})
        if self.model.regime_grades() != before:
            if self.best_swing_toggle.isChecked():
                self._resort_from_source()
            else:
                self._reapply_column_profile()
        self.refresh_swing_regime_line()

    def set_spy_regime(self, payload) -> None:
        """The regime strip's readings (`BounceService.regimeStripChanged`); SPY is shown."""
        self._spy_regime = dict(payload or {}) if isinstance(payload, dict) else {}
        self.refresh_swing_regime_line()

    def refresh_swing_regime_line(self) -> None:
        banner = getattr(self, "longs_off_banner", None)
        shown = bool(banner is not None and banner.text())
        text = regime_line(self.model.regime_grades(), getattr(self, "_longs_gate", None),
                           self._spy_regime, banner_shown=shown)
        if text != self.swing_regime_label.text():
            self.swing_regime_label.setText(text)
        self.swing_regime_label.setVisible(bool(text))

    # -- the worker --------------------------------------------------------------
    def _start_swing_context_read(self) -> None:
        """One read at a time, and only when an input moved. `stat` calls here, reads there."""
        if not getattr(self, "_reads_swing_context", False):
            return
        worker = self._swing_worker
        if worker is not None and worker.isRunning():
            self._swing_pending = True
            return
        data_date = str(getattr(self, "_data_date", "") or "")
        try:
            from ui.services import swing_table_context as swing_context

            signature = swing_context.signature(data_date)
        except Exception as exc:  # noqa: BLE001 - no signature reads every time
            note_swallowed("swing context signature failed", exc, quiet=True)
            signature = object()
        if signature == self._swing_signature:
            return
        self._swing_signature = signature
        worker = _SwingContextWorker(data_date, self)
        worker.done.connect(self._on_swing_context_ready)
        worker.finished.connect(worker.deleteLater)
        self._swing_worker = worker
        worker.start()

    def _on_swing_context_ready(self, payload: object) -> None:  # pragma: no cover - signal seam
        self._swing_worker = None
        if payload is None:
            self._swing_signature = object()
        else:
            self.set_swing_context(payload)
        if self._swing_pending:
            self._swing_pending = False
            self._start_swing_context_read()
