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
        "summary": {"healthy": 1, "degraded": 1, "unhealthy": 0, "unknown": 0, "total": 2},
        "checks": [
            {"id": "heartbeat", "label": "Runtime heartbeat", "status": "healthy", "summary": "PID 123; idle.", "updated_at": "12:29", "source": "heartbeat.json", "details": {"pid": 123}},
            {"id": "greatness_shadow", "label": "Greatness shadow", "status": "degraded", "summary": "Last evaluation is 21m old.", "updated_at": "12:09", "source": "greatness_candidates.json", "details": {"evaluations": 20}},
        ],
    }


def _partial_payload() -> dict:
    """Every implemented check green, required dimensions unmeasured.

    This is the payload shape the old panel rendered as a green tile.
    """
    return {
        "status": "unknown",
        "generated_at": "2026-07-13T12:30:00-07:00",
        "market_phase": "regular",
        "market_session": "06:30-13:00",
        "summary": {"healthy": 1, "degraded": 0, "unhealthy": 0, "unknown": 1, "total": 2},
        "checks": [
            {"id": "heartbeat", "label": "Runtime heartbeat", "status": "healthy", "summary": "PID 123; idle.", "updated_at": "12:29", "source": "heartbeat.json", "details": {"pid": 123}},
            {
                "id": "disk_storage_warnings",
                "label": "Disk/storage warnings",
                "status": "unknown",
                "summary": "Not measured: disk/storage warnings.",
                "updated_at": "",
                "source": "plan.md#sec-6.3",
                "details": {"requirement": "disk/storage warnings"},
            },
        ],
        "required_checks": [
            {"id": "heartbeat_age", "label": "Heartbeat age", "status": "healthy", "implemented": True},
            {
                "id": "disk_storage_warnings",
                "label": "Disk/storage warnings",
                "status": "unknown",
                "implemented": False,
            },
        ],
    }


def _payload_with_jobs_and_phases() -> dict:
    """A payload carrying the two things the panel used to drop on the floor."""
    payload = _payload()
    payload["checks"].append(
        {
            "id": "run_manifest",
            "label": "Latest scan manifest",
            "status": "healthy",
            "summary": "master_scan ok; 18.0m; 5.0m old. Slowest phase tracker update 10.0m.",
            "updated_at": "12:18",
            "source": "run_manifests",
            "details": {
                "phases": [
                    {"label": "tracker update", "seconds": 600.0, "aggregate": False, "share_pct": 83.3},
                    {"label": "universe", "seconds": 120.0, "aggregate": False, "share_pct": 16.7},
                    {"label": "TOTAL", "seconds": 1080.0, "aggregate": True, "share_pct": None},
                ]
            },
        }
    )
    payload["jobs"] = [
        {
            "slot": "12:00",
            "job_type": "swing_scan",
            "state": "COMPLETED",
            "attempt": 1,
            "started_at": "2026-07-13T12:00:01-07:00",
            "ended_at": "2026-07-13T12:18:00-07:00",
            "run_id": "run-1",
            "error": "",
            "error_class": "",
        },
        {
            "slot": "13:00",
            "job_type": "swing_scan",
            "state": "FAILED",
            "attempt": 3,
            "started_at": "2026-07-13T13:00:01-07:00",
            "ended_at": "2026-07-13T13:05:00-07:00",
            "run_id": "run-2",
            "error": "provider said no",
            "error_class": "bad_local_state",
        },
    ]
    payload["latest_manifest"] = {"run_id": "run-1", "phases": []}
    return payload


def test_per_job_and_per_phase_evidence_is_rendered_as_rows():
    """plan.md sec 6.3 bullets 4 and 11 reached the payload and stopped there.

    ``set_payload`` read only ``payload["checks"]``, so per-job attempts and the
    manifest's phase timings - both already computed - were never displayed and
    only one aggregate total ever reached the trader.
    """
    from ui.panels.health_panel import HealthPanel

    panel = HealthPanel(refresh_interval_ms=60_000)
    panel.set_payload(_payload_with_jobs_and_phases())

    assert panel.jobs_table.rowCount() == 2
    assert panel.jobs_table.item(1, 0).text() == "FAILED"
    assert panel.jobs_table.item(1, 2).text() == "13:00"
    assert panel.jobs_table.item(1, 3).text() == "3"
    assert "bad_local_state" in panel.jobs_table.item(1, 6).text()
    assert panel.detail_tabs.tabText(1) == "Jobs (2)"

    assert panel.phases_table.rowCount() == 3
    assert panel.phases_table.item(0, 0).text() == "tracker update"
    assert panel.phases_table.item(0, 1).text() == "10.0"
    assert panel.phases_table.item(0, 3).text() == "83.3%"
    # The aggregate TOTAL row is shown but claims no share of itself.
    assert panel.phases_table.item(2, 3).text() == "-"
    assert panel.detail_tabs.tabText(2) == "Phase timings (3)"
    panel.shutdown()


def test_job_and_phase_tables_empty_cleanly_when_nothing_ran():
    from ui.panels.health_panel import HealthPanel

    panel = HealthPanel(refresh_interval_ms=60_000)
    panel.set_payload(_payload())

    assert panel.jobs_table.rowCount() == 0
    assert panel.phases_table.rowCount() == 0
    assert panel.detail_tabs.tabText(1) == "Jobs (0)"
    panel.shutdown()


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


def test_unknown_is_rendered_as_its_own_status_not_as_green_or_bad():
    """UNKNOWN gets its own count, tone and row treatment (plan.md sec 6.3)."""
    from ui import theme
    from ui.panels.health_panel import _STATUS_TONES, HealthPanel

    panel = HealthPanel(refresh_interval_ms=60_000)
    statuses = []
    panel.statusChanged.connect(statuses.append)
    panel.set_payload(_partial_payload())

    assert panel.overall_tile.value_label.text() == "UNKNOWN"
    assert panel.unknown_tile.value_label.text() == "1"
    assert panel.healthy_tile.value_label.text() == "1"
    assert statuses[-1] == "unknown"
    # The unmeasured dimensions are named on the page, not just counted.
    assert "UNKNOWN" in panel.meta_label.text()
    assert "Disk/storage warnings" in panel.meta_label.text()

    unknown_cell = panel.table.item(1, 0)
    assert unknown_cell.text() == "UNKNOWN"
    # Visually distinct from healthy, degraded and unhealthy - by colour and,
    # because colour alone is not a distinction, by italics.
    tones = {theme.color(_STATUS_TONES[name]) for name in ("healthy", "degraded", "unhealthy", "unknown")}
    assert len(tones) == 4
    assert unknown_cell.foreground().color().name().lower() == theme.color(_STATUS_TONES["unknown"]).lower()
    assert unknown_cell.font().italic() is True
    assert panel.table.item(0, 0).font().italic() is False
    panel.shutdown()


def test_a_status_the_panel_does_not_recognize_renders_as_unknown():
    from ui.panels.health_panel import HealthPanel

    panel = HealthPanel(refresh_interval_ms=60_000)
    payload = _payload("")
    payload["checks"][0]["status"] = ""
    statuses = []
    panel.statusChanged.connect(statuses.append)
    panel.set_payload(payload)

    assert panel.overall_tile.value_label.text() == "UNKNOWN"
    assert panel.table.item(0, 0).text() == "UNKNOWN"
    assert statuses[-1] == "unknown"
    panel.shutdown()


def test_hidden_page_audits_slowly_and_visible_page_audits_fast(monkeypatch):
    """Desk snappiness packet 1 item 1b: the 15 s cadence is for a trader
    LOOKING at the page. Hidden, the audit keeps running - the shell's status
    chip must keep updating - but at 120 s. The timer itself never stops."""
    import ui.panels.health_panel as hp

    monkeypatch.setattr(hp, "build_operations_audit", lambda: _payload())
    panel = hp.HealthPanel(refresh_interval_ms=15_000)
    try:
        # Construction is hidden: the shell decides when the page shows.
        assert panel._timer.interval() == 120_000
        panel.show()
        _app.processEvents()
        assert panel._timer.interval() == 15_000
        panel.hide()
        _app.processEvents()
        assert panel._timer.interval() == 120_000
        panel.wait_for_audit()
    finally:
        panel.shutdown()
        panel.deleteLater()
        _app.processEvents()


def test_refresh_never_blocks_the_gui_thread_even_when_the_audit_is_slow(monkeypatch):
    """The live regression: build_operations_audit streams multi-MB shadow logs
    (~0.4s measured on the real machine, worse mid-scan) and ran synchronously
    on the GUI thread every 15s - a recurring visible freeze.  refresh() must
    return immediately and deliver the payload from a worker thread."""
    import time

    import ui.panels.health_panel as hp

    marker = _payload("degraded")

    def slow_audit():
        time.sleep(0.5)
        return marker

    monkeypatch.setattr(hp, "build_operations_audit", slow_audit)
    panel = hp.HealthPanel(refresh_interval_ms=3_600_000)
    try:
        started = time.perf_counter()
        panel.refresh()
        elapsed = time.perf_counter() - started
        assert elapsed < 0.2, (
            f"refresh() blocked the GUI thread for {elapsed:.3f}s; the audit "
            "must run on a worker thread"
        )
        # A tick landing while an audit is in flight is skipped, not queued.
        panel.refresh()
        panel.wait_for_audit()
        _app.processEvents()
        assert panel.overall_tile.value_label.text() == "DEGRADED"
    finally:
        # shutdown, not just deleteLater: the construction-time
        # singleShot(0, refresh) fires during processEvents here and, without
        # _closing set, starts one more 0.5 s slow-audit thread that outlives
        # this test - which is exactly what
        # test_shutdown_joins_the_audit_thread's process-wide sweep then
        # catches in any file order where it runs next.
        panel.shutdown()
        panel.deleteLater()
        _app.processEvents()
