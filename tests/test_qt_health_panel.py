from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _payload(status: str = "degraded") -> dict:
    return {
        "status": status,
        "generated_at": "2026-07-13T12:30:00-07:00",
        "market_phase": "regular",
        "market_session": "06:30-13:00",
        "summary": {"healthy": 1, "degraded": 1, "unhealthy": 0, "total": 2},
        "checks": [
            {"id": "heartbeat", "label": "Runtime heartbeat", "status": "healthy", "summary": "PID 123; idle.", "updated_at": "12:29", "source": "heartbeat.json", "details": {"pid": 123}},
            {"id": "greatness_shadow", "label": "Greatness shadow", "status": "degraded", "summary": "Last evaluation is 21m old.", "updated_at": "12:09", "source": "greatness_candidates.json", "details": {"evaluations": 20}},
        ],
    }


def test_health_panel_renders_overall_checks_and_evidence():
    from ui.panels.health_panel import HealthPanel

    panel = HealthPanel(refresh_interval_ms=60_000)
    panel.set_payload(_payload())

    assert panel.overall_tile.value_label.text() == "DEGRADED"
    assert panel.table.rowCount() == 2
    assert panel.table.item(0, 1).text() == "Runtime heartbeat"
    panel.table.selectRow(1)
    panel._show_selected_check(1)
    assert "Greatness shadow" in panel.details.toPlainText()
    assert '"evaluations": 20' in panel.details.toPlainText()
    panel.shutdown()


def test_health_panel_emits_status_for_status_bar():
    from ui.panels.health_panel import HealthPanel

    panel = HealthPanel(refresh_interval_ms=60_000)
    statuses = []
    panel.statusChanged.connect(statuses.append)
    panel.set_payload(_payload("healthy"))

    assert statuses[-1] == "healthy"
    panel.shutdown()
