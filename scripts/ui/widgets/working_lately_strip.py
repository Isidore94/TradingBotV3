"""The one Working-lately line, at the top of the M5 alerts column - ST6.4.

Decision 0016: *"what is working lately"* belongs on the Trading Desk and never
only in Research. This is that line. It says which swing family and which
bounce type are leading, the bound and the n behind each, the session it is as
of, and - always - that the leader is OBSERVATIONAL: `observational leader among
K cells` is printed rather than corrected away, because K cells were read and
the best of K was named.

**It renders; it decides nothing and reads nothing.** The payload arrives from
`ui.services.working_lately_service`'s `snapshotChanged`, built on a worker.
This widget formats a dict. Clicking it opens the Setup Tracker page, where the
same `snapshot_id` is printed above the table the line came from.

The switch beside it is the priority switch (ST6.5): display-only, default OFF,
persisted in `local_settings` as `prioritise_working_lately`, read AT SORT TIME
by the three lists that honour it. It REORDERS and never withholds - the tier
gate, movers-only and the repetition fold are untouched, and a test asserts the
three lists show the same rows either way.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel

import working_lately


class WorkingLatelyStrip(QFrame):
    """One line, one tooltip, one switch. Nothing else lives here."""

    #: The trader clicked the line - open the Setup Tracker page.
    openRequested = Signal()
    #: The priority switch moved. The host re-pushes the order to its lists.
    prioritiseToggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkingLatelyStrip")
        self._payload: dict[str, Any] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)

        self.line_label = QLabel(working_lately.snapshot_line(None))
        self.line_label.setObjectName("WorkingLatelyLine")
        self.line_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.line_label.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.line_label, 1)

        self.prioritise_box = QCheckBox("Prioritise what is working")
        self.prioritise_box.setToolTip(
            "Sort the M5 list, the waiting review list and the setups table so "
            "the cells that are leading come first.\n\n"
            "It REORDERS and never withholds: the same rows are shown either "
            "way, the tier gate, the movers-only filter and the repeat fold are "
            "untouched, and nothing is written differently. Turn it off and "
            "today's order comes back exactly."
        )
        self.prioritise_box.setChecked(working_lately.prioritise_enabled())
        self.prioritise_box.toggled.connect(self._on_toggled)
        layout.addWidget(self.prioritise_box, 0)

        self._refresh()

    # -- data --------------------------------------------------------------

    def set_snapshot(self, payload: Any) -> None:
        """Render THE snapshot the service holds. Formatting only."""
        self._payload = dict(payload or {})
        self._refresh()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._payload)

    def line_text(self) -> str:
        return working_lately.snapshot_line(self._payload or None)

    def tooltip_text(self) -> str:
        """The identity first, then every cell - what the line summarised."""
        if not self._payload:
            return (
                "No Working-lately snapshot yet. The desk builds one after the "
                "window shows, on the day roll, after every persisted tracker "
                "export and every 30 minutes."
            )
        lines = [
            working_lately.snapshot_stamp(self._payload),
            "",
            "Click to open the Setup Tracker, which prints the same snapshot.",
            "",
        ]
        lines.extend(working_lately.snapshot_cell_lines(self._payload))
        return "\n".join(lines)

    def _refresh(self) -> None:
        """Format once, set twice. Nothing else on the Qt thread.

        **Re-review advisory 4.** This cost 14.65 ms per call: it built the
        tooltip TWICE (once per `setToolTip`) - and the tooltip is one
        `EvidenceCell.line()` per cell over a live snapshot's hundreds - and it
        called `setStyleSheet("")` on the label, which is a stylesheet
        recomputation for the whole subtree to say nothing. The variant lives in
        `theme.qss` keyed on a dynamic property, which is the rule the fluidity
        pass wrote and this widget was breaking.
        """
        tooltip = self.tooltip_text()
        self.line_label.setText(self.line_text())
        self.line_label.setToolTip(tooltip)
        self.setToolTip(tooltip)
        has_snapshot = bool(self._payload)
        if self.property("hasSnapshot") != has_snapshot:
            self.setProperty("hasSnapshot", has_snapshot)
            # Re-polish THIS widget only - no stylesheet is set or recomputed.
            style = self.style()
            style.unpolish(self)
            style.polish(self)

    # -- interaction -------------------------------------------------------

    def _on_toggled(self, checked: bool) -> None:
        try:
            import project_paths

            project_paths.save_local_setting("prioritise_working_lately", bool(checked))
            project_paths.invalidate_local_settings_cache()
        except Exception:  # noqa: BLE001 - a preference never costs the desk
            pass
        self.prioritiseToggled.emit(bool(checked))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            self.openRequested.emit()
        super().mouseReleaseEvent(event)
