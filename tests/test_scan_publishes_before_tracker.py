"""P0-2 2b: the close slot publishes its outputs before the tracker write.

09-23 07:30 died with ``MemoryError`` while the write slot held the 1.35 GB
tracker; because the tracker ran BEFORE the output writes, the scan lost its
signals, reports and state too. Now a tracker failure is a ledger row and a
stamp, and the scan still returns published.

The scan runs in a child with ``TRADINGBOTV3_DATA_DIR``, ``LOCALAPPDATA`` and the
diagnostics dir pointed at ``tmp_path`` (the same idiom as
``test_pct3_runner_publishes_compression.py``), so nothing touches a live store
or the suite's shared test home.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

RESULT_MARKER = "P02B-RESULT::"

CHILD_SOURCE = r'''
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

SCRATCH = Path(sys.argv[1])
SCRIPTS_DIR = sys.argv[2]
MODE = sys.argv[3]
SYMBOL = "PUBX"
RESULT_MARKER = "P02B-RESULT::"

os.environ["TRADINGBOTV3_DATA_DIR"] = str(SCRATCH / "home")
os.environ["LOCALAPPDATA"] = str(SCRATCH / "localappdata")
os.environ["TRADINGBOT_DIAGNOSTICS_DIR"] = str(SCRATCH / "diag")
os.environ["TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE"] = "1"
os.environ["TRADINGBOT_RUN_TRIGGER"] = "Auto Pilot swing scan 13:00"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, SCRIPTS_DIR)

import pandas as pd

import project_paths

for root in (project_paths.DATA_DIR, project_paths.LOCAL_SETTINGS_DIR, project_paths.get_diagnostics_dir()):
    if not str(root).startswith(str(SCRATCH)):
        raise SystemExit(f"ABORT: a project_paths root is outside the scratch dir: {root}")

from master_avwap_lib import runner


def _sessions(count):
    end = date.today() - timedelta(days=1)
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    return list(pd.bdate_range(end=pd.Timestamp(end), periods=count))


stamps = _sessions(120)
rows = []
for index, stamp in enumerate(stamps):
    base = 80.0 + index * 0.25
    rows.append((stamp, base, base + 1.2, base - 1.2, base + 0.3))
frame = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])
frame["volume"] = 1_000_000
anchor_iso = stamps[90].date().isoformat()

runner.resolve_master_scan_watchlist_paths = lambda **kwargs: (
    [SCRATCH / "longs.txt"],
    [SCRATCH / "shorts.txt"],
    "test watchlist",
)
runner.load_tickers_from_paths = lambda paths, **kw: (
    [SYMBOL] if "longs" in str(list(paths)[0]) else []
)
runner.append_master_avwap_d1_watchlist_symbols = lambda longs, shorts: (longs, shorts, 0)
runner.load_theta_long_symbols = lambda *a, **k: []
runner.load_scan_earnings_context = lambda symbols: ({SYMBOL: [anchor_iso]}, {SYMBOL: {}})
runner.collect_upcoming_earnings_dates = lambda symbols: {}
runner._maybe_run_setup_tracker_catchup = lambda **kwargs: {}
runner.connect_daily_data_client = lambda **kwargs: None
runner.build_market_regime_snapshot = lambda ib, day: {"label": "mixed"}
runner.fetch_daily_bars = lambda ib, sym, days, **kw: frame.copy()
runner.record_theta_picks = lambda *a, **k: None
runner.refresh_playbook_study_if_stale = lambda *a, **k: False
runner.calibrate_expected_r_prior_anchors = lambda *a, **k: None
# The deferred theta pass opens its own IB connection; never from a test.
runner._schedule_deferred_theta_enrichment = lambda **kwargs: False

if MODE == "fail":
    def _boom():
        raise MemoryError("tracker too big")
    runner.load_setup_tracker_payload = _boom
else:
    runner.load_setup_tracker_payload = lambda: {}
    runner.update_setup_tracker_from_scan = lambda *a, **k: None

result = runner._run_master_impl(update_setup_tracker=True)

import tracker_store
from job_ledger import default_ledger_path

ledger = default_ledger_path()
ledger_rows = []
if ledger.exists():
    ledger_rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
payload = {
    "setup_tracker_updated": result.get("setup_tracker_updated"),
    "setup_tracker_write_error": result.get("setup_tracker_write_error", ""),
    "setup_tracker_skip_reason": result.get("setup_tracker_skip_reason", ""),
    "priority_report_exists": Path(runner.PRIORITY_SETUPS_FILE).exists(),
    "ai_state_exists": Path(project_paths.MASTER_AVWAP_AI_STATE_FILE).exists(),
    "ledger_rows": ledger_rows,
    "write_state": tracker_store.read_write_state(),
    "failure_line": tracker_store.tracker_write_failure_line(),
    "health_line": tracker_store.tracker_last_written_line(),
}
print(RESULT_MARKER + json.dumps(payload, default=str))
'''


def _run_child(tmp_path: Path, mode: str) -> dict:
    scratch = tmp_path / mode
    for name in ("home", "localappdata", "diag"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    child = scratch / "child_scan.py"
    child.write_text(CHILD_SOURCE, encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(child), str(scratch), str(SCRIPTS_DIR), mode],
        capture_output=True,
        text=True,
        timeout=900,
        env=environment,
        cwd=str(SCRIPTS_DIR),
    )
    assert completed.returncode == 0, (
        f"the scan child failed ({completed.returncode})\n"
        f"stdout tail:\n{completed.stdout[-3000:]}\nstderr tail:\n{completed.stderr[-3000:]}"
    )
    line = next(
        (
            text[len(RESULT_MARKER):]
            for text in reversed(completed.stdout.splitlines())
            if text.startswith(RESULT_MARKER)
        ),
        "",
    )
    assert line, f"the child printed no result payload:\n{completed.stdout[-3000:]}"
    return json.loads(line)


@pytest.fixture(scope="module")
def failed_tracker(tmp_path_factory) -> dict:
    return _run_child(tmp_path_factory.mktemp("p02b"), "fail")


@pytest.fixture(scope="module")
def good_tracker(tmp_path_factory) -> dict:
    return _run_child(tmp_path_factory.mktemp("p02b"), "ok")


def test_a_tracker_exception_leaves_published_outputs_on_disk(failed_tracker):
    assert failed_tracker["priority_report_exists"] is True
    assert failed_tracker["ai_state_exists"] is True
    assert failed_tracker["setup_tracker_updated"] is False
    assert "MemoryError" in failed_tracker["setup_tracker_write_error"]
    assert failed_tracker["setup_tracker_skip_reason"].startswith("write failed: MemoryError")


def test_a_tracker_exception_writes_one_keyless_ledger_row(failed_tracker):
    rows = [row for row in failed_tracker["ledger_rows"] if row.get("event") == "setup_tracker_write_failed"]
    assert len(rows) == 1
    row = rows[0]
    assert "key" not in row
    assert "MemoryError" in row["error"]
    assert row["slot"] == "Auto Pilot swing scan 13:00"
    assert "+" in row["ts"] or row["ts"].endswith("Z") or "-" in row["ts"][19:]


def test_a_tracker_exception_stamps_the_failure_for_the_digest(failed_tracker):
    state = failed_tracker["write_state"]
    assert state["last_result"] == "failed"
    assert failed_tracker["failure_line"].startswith("setup tracker write failed ")
    assert failed_tracker["failure_line"].endswith("; last good unknown")


def test_a_good_tracker_write_stamps_last_written(good_tracker):
    assert good_tracker["setup_tracker_updated"] is True
    state = good_tracker["write_state"]
    assert state["last_result"] == "ok" and state["last_written_at"]
    assert good_tracker["failure_line"] == ""
    assert good_tracker["health_line"].startswith("tracker last written 20")
    assert not [row for row in good_tracker["ledger_rows"] if row.get("event") == "setup_tracker_write_failed"]


# ---------------------------------------------------------------------------
# Where the stamp is shown: the phone digest's OPERATIONS and the Health page.
# ---------------------------------------------------------------------------


def _report_payload(**extra) -> dict:
    return {
        "generated_at": "2026-09-24 13:30:00",
        "ib_status": "connected",
        "regime": "mixed",
        "longs": [],
        "shorts": [],
        "swing_picks": [],
        "alerts": [],
        "slots_done": [],
        "next_slot": "",
        "log_lines": [],
        "auto_longs": [],
        "auto_shorts": [],
        "enabled": True,
        "auto_mode": "AWAY",
        **extra,
    }


def test_the_digest_operations_section_names_a_failed_tracker_write():
    import autopilot_core as core

    line = "setup tracker write failed 2026-09-23 07:31; last good 2026-09-22 07:51"
    report = core.render_away_report(_report_payload(tracker_write_failure_line=line))
    operations = report[report.index("OPERATIONS"):]
    assert line in operations
    assert "setup tracker write failed" not in core.render_away_report(_report_payload())


def test_failure_and_health_lines_from_the_stamp(tmp_path):
    from datetime import datetime, timedelta, timezone

    import tracker_store

    path = tmp_path / "state.json"
    assert tracker_store.tracker_last_written_line(tracker_store.read_write_state(path)) == (
        "tracker last written unknown"
    )
    tz = timezone(timedelta(hours=-4))
    tracker_store.record_write_success(path=path, now=datetime(2026, 9, 22, 7, 51, tzinfo=tz))
    tracker_store.record_write_failure(
        "MemoryError",
        slot="13:00",
        path=path,
        ledger_path=tmp_path / "ledger.jsonl",
        now=datetime(2026, 9, 23, 7, 31, tzinfo=tz),
    )
    state = tracker_store.read_write_state(path)
    assert tracker_store.tracker_write_failure_line(state) == (
        "setup tracker write failed 2026-09-23 07:31; last good 2026-09-22 07:51"
    )
    assert tracker_store.tracker_last_written_line(state) == "tracker last written 2026-09-22 07:51"
    row = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["event"] == "setup_tracker_write_failed" and row["slot"] == "13:00"


@pytest.mark.qt
def test_the_health_page_shows_tracker_last_written(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])  # noqa: F841
    import tracker_store
    from ui.panels import health_panel

    monkeypatch.setattr(
        tracker_store,
        "read_write_state",
        lambda path=None: {"last_written_at": "2026-09-22T07:51:00-04:00", "last_result": "ok"},
    )
    payload = health_panel._with_tracker_write_line({"status": "healthy", "summary": {}, "checks": []})
    assert payload["tracker_last_written_line"] == "tracker last written 2026-09-22 07:51"
    panel = health_panel.HealthPanel(refresh_interval_ms=60_000)
    try:
        panel.set_payload(payload)
        assert "tracker last written 2026-09-22 07:51" in panel.meta_label.text()
    finally:
        panel.shutdown()
