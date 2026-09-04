"""The live M5 tee wiring (BD-20, plan sec 8.4 / LD-01).

What must hold, and what this file pins:

* the tee reads the bar cache the champion already populated and issues **no
  provider request** - it cannot fail or delay a champion fetch;
* nothing in ``bounce_bot_lib`` imports or knows about the warehouse;
* the ``latest_bars`` snapshot is taken on the calling (GUI) thread, so a
  concurrent resize cannot raise mid-iteration;
* capture is **spool-only**: no lake I/O at all on the GUI thread;
* an unconfigured warehouse constructs nothing, and a capture failure is
  swallowed - research evidence never breaks the desk;
* the Health page renders the six tiles on its existing audit refresh.
"""

import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from ui.services.warehouse_service import WarehouseTeeCapture  # noqa: E402

UTC = timezone.utc
OPEN_UTC = datetime(2026, 8, 3, 13, 30, tzinfo=UTC)


class _FakeBot:
    """Only the one attribute the tee is allowed to touch."""

    def __init__(self, bars):
        self.latest_bars = bars


@dataclass(frozen=True)
class FakeIbBar:
    """Shaped like the champion's IbBar: a naive *market-local* bar start."""

    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


#: The champion's IB bars carry naive *desk-local* starts (the tee localizes
#: them with ``market_local_timezone()``), so the fixture derives the local
#: wall-clock for OPEN_UTC instead of hard-coding one machine's offset.
def _open_local() -> datetime:
    from research_warehouse.bar_archive import market_local_timezone

    return OPEN_UTC.astimezone(market_local_timezone()).replace(tzinfo=None)


def _bars(count=3, *, start_offset=0, symbol_base=100.0):
    return [
        FakeIbBar(
            dt=_open_local() + timedelta(minutes=5 * (index + start_offset)),
            open=symbol_base + index,
            high=symbol_base + index + 1,
            low=symbol_base + index - 1,
            close=symbol_base + index + 0.5,
            volume=1000 + index,
        )
        for index in range(count)
    ]


def _cache(symbols=("AAPL", "MSFT"), count=3):
    return {f"{symbol}|5 D|5 mins": _bars(count) for symbol in symbols}


class _RecordingSpool:
    def __init__(self):
        self.writes = []

    def write(self, dataset, rows, **kwargs):
        rows = list(rows)
        self.writes.append((dataset, rows))
        return len(rows)


@pytest.fixture()
def enabled(monkeypatch, tmp_path):
    import research_warehouse.config as config

    monkeypatch.setattr(config, "warehouse_enabled", lambda: True)
    monkeypatch.setattr(config, "get_research_store_dir", lambda: tmp_path / "lake")
    return config


def test_the_tee_spools_and_never_touches_the_lake(enabled, monkeypatch):
    """Spool-only: the GUI thread does zero lake I/O and zero provider work."""
    import research_warehouse.store as store_module

    opened = []
    monkeypatch.setattr(
        store_module.ResearchStore, "open", classmethod(lambda cls, root=None: opened.append(1))
    )

    spool = _RecordingSpool()
    capture = WarehouseTeeCapture(spool=spool)
    report = capture.capture(_FakeBot(_cache()), now=OPEN_UTC + timedelta(minutes=20))

    assert opened == [], "no ResearchStore is opened on the capture path"
    assert report is not None and report.status == "SPOOLED"
    assert [dataset for dataset, _rows in spool.writes] == ["bar_m5"]
    assert capture.rows_spooled == report.rows_published > 0


def test_the_same_bars_are_not_spooled_twice(enabled):
    """The session ``seen`` set is what de-duplicates without reading the lake."""
    spool = _RecordingSpool()
    capture = WarehouseTeeCapture(spool=spool)
    cache = _cache()

    first = capture.capture(_FakeBot(cache), now=OPEN_UTC + timedelta(minutes=20))
    again = capture.capture(_FakeBot(cache), now=OPEN_UTC + timedelta(minutes=21))
    assert first.rows_published > 0
    assert again is None or again.rows_published == 0

    # A new completed bar does publish, and only that bar.
    grown = {key: value + _bars(1, start_offset=3) for key, value in cache.items()}
    third = capture.capture(_FakeBot(grown), now=OPEN_UTC + timedelta(minutes=25))
    assert third.rows_published == len(grown)


def test_the_bar_cache_is_snapshotted_on_the_calling_thread(enabled, monkeypatch):
    """A dict the champion may resize is copied before anything iterates it."""
    from research_warehouse import bar_archive

    cache = _cache()
    seen_object = {}

    def spy(store, bars_by_symbol, **kwargs):
        seen_object["passed"] = bars_by_symbol
        # Mutating the champion's dict now must not disturb the capture.
        cache["NVDA|5 D|5 mins"] = _bars(2)
        return bar_archive.CaptureReport(status="SPOOLED")

    monkeypatch.setattr(bar_archive, "capture_m5_tee", spy)
    capture = WarehouseTeeCapture(spool=_RecordingSpool())
    capture.capture(_FakeBot(cache), now=OPEN_UTC)

    passed = seen_object["passed"]
    assert passed is not cache, "the tee must not hold the champion's own dict"
    assert "NVDA|5 D|5 mins" not in passed


def test_an_unconfigured_warehouse_captures_nothing(monkeypatch):
    import research_warehouse.config as config

    monkeypatch.setattr(config, "warehouse_enabled", lambda: False)
    spool = _RecordingSpool()
    capture = WarehouseTeeCapture(spool=spool)
    assert capture.capture(_FakeBot(_cache())) is None
    assert spool.writes == []


def test_a_capture_failure_never_breaks_the_desk(enabled, monkeypatch):
    from research_warehouse import bar_archive

    def explode(*args, **kwargs):
        raise RuntimeError("DAS unplugged mid-session")

    monkeypatch.setattr(bar_archive, "capture_m5_tee", explode)
    capture = WarehouseTeeCapture(spool=_RecordingSpool())
    assert capture.capture(_FakeBot(_cache())) is None  # no exception escapes
    assert "DAS unplugged" in capture.last_error


def test_no_champion_module_imports_the_warehouse():
    """R1/BD-15 structurally: the tee is a reader, never a champion import."""
    for module in ("bounce_bot_lib.legacy", "master_avwap_lib.legacy"):
        source = (SCRIPTS_DIR / module.replace(".", "/")).with_suffix(".py").read_text(encoding="utf-8")
        assert "research_warehouse" not in source
        assert "warehouse_service" not in source


# --- the service-layer wiring ---------------------------------------------
def test_the_bounce_service_owns_one_warehouse_timer_and_stops_it():
    from ui.services.bounce_service import BounceService

    service = BounceService()
    try:
        assert service._warehouse_timer.interval() == 60_000
        assert not service._warehouse_timer.isActive()  # armed only once started
        assert service._warehouse_capture is None  # constructed lazily
        # Shutdown stops it along with every other timer this service owns.
        service.stop()
        assert not service._warehouse_timer.isActive()
    finally:
        service.shutdown()


def test_the_service_slot_is_inert_without_a_bot():
    from ui.services.bounce_service import BounceService

    service = BounceService()
    try:
        service.capture_warehouse_tee()  # no bot: returns, constructs nothing
        assert service._warehouse_capture is None
    finally:
        service.shutdown()


# --- the Health page -------------------------------------------------------
def test_the_health_page_renders_the_warehouse_tiles(enabled, monkeypatch):
    from ui.panels import health_panel

    payload = {
        "status": "healthy",
        "summary": {"healthy": 2, "degraded": 0, "unhealthy": 0, "unknown": 0, "total": 2},
        "checks": [{"id": "a", "label": "A", "status": "healthy", "summary": "", "details": {}}],
    }
    monkeypatch.setattr(
        health_panel,
        "warehouse_checks",
        lambda now=None: [
            {"id": "warehouse_das_mount", "label": "Warehouse: DAS", "status": "healthy", "summary": "ok", "details": {}},
            {"id": "warehouse_backup", "label": "Warehouse: Backup", "status": "degraded", "summary": "old", "details": {}},
        ],
    )

    merged = health_panel._with_warehouse_checks(payload)
    assert [row["id"] for row in merged["checks"]] == ["a", "warehouse_das_mount", "warehouse_backup"]
    assert merged["summary"]["total"] == 4 and merged["summary"]["degraded"] == 1
    # A degraded tile moves a healthy page to degraded; it never improves one.
    assert merged["status"] == "degraded"


def test_an_unconfigured_warehouse_reads_as_unknown_not_healthy(monkeypatch):
    """OFF is an unmeasured dimension; this page never paints that green."""
    import research_warehouse.config as config
    from ui.panels import health_panel

    monkeypatch.setattr(config, "get_research_store_dir", lambda: None)
    rows = health_panel.warehouse_checks()
    assert rows and {row["status"] for row in rows} == {"unknown"}


# --- D21: no filesystem work on the GUI timer path ------------------------
class _ThreadRecordingSpool(_RecordingSpool):
    """A spool writer that records which thread constructed and wrote it."""

    constructed_on: list = []
    wrote_on: list = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        type(self).constructed_on.append(threading.current_thread())

    def write(self, dataset, rows, **kwargs):
        type(self).wrote_on.append(threading.current_thread())
        return super().write(dataset, rows, **kwargs)


@pytest.fixture()
def recording_writer(monkeypatch):
    import research_warehouse.spool as spool_module

    _ThreadRecordingSpool.constructed_on = []
    _ThreadRecordingSpool.wrote_on = []
    monkeypatch.setattr(spool_module, "ResearchSpoolWriter", _ThreadRecordingSpool)
    return _ThreadRecordingSpool


def test_submit_touches_no_filesystem_on_the_gui_thread(enabled, recording_writer):
    """D21: writer construction adopts stale segments and renames files.

    `ResearchSpoolWriter.__init__` mkdirs and adopts every stale `.open`
    segment, and each `write` runs `enforce_cap` (glob + stat + read) and
    fsyncs. None of that may happen on the 60s GUI timer.
    """
    capture = WarehouseTeeCapture()
    main = threading.current_thread()

    assert capture.submit(_FakeBot(_cache())) is True
    # Nothing was built on this thread, then or ever.
    assert main not in recording_writer.constructed_on

    assert capture.wait_idle(5.0)
    capture.close()
    assert len(recording_writer.constructed_on) == 1
    assert recording_writer.constructed_on[0] is not main
    assert recording_writer.wrote_on and all(t is not main for t in recording_writer.wrote_on)


def test_submit_still_spools_through_the_spool_only_path(enabled, recording_writer, monkeypatch):
    """The work still happens - just not here. And still no lake I/O."""
    import research_warehouse.store as store_module

    opened_on = []
    monkeypatch.setattr(
        store_module.ResearchStore,
        "open",
        classmethod(lambda cls, root=None: opened_on.append(threading.current_thread())),
    )

    capture = WarehouseTeeCapture()
    capture.submit(_FakeBot(_cache()), now=OPEN_UTC + timedelta(minutes=20))
    worker = capture._worker
    assert capture.wait_idle(5.0)
    capture.close()

    assert worker not in opened_on, "this tee worker never opens a ResearchStore"
    assert capture.last_report is not None and capture.last_report.status == "SPOOLED"
    assert capture.rows_spooled > 0
    # The rows went through the spool writer the worker built, as bar_m5.
    writer = capture._spool
    assert isinstance(writer, recording_writer)
    assert [dataset for dataset, _rows in writer.writes] == ["bar_m5"]


def test_submit_is_memory_only_when_the_warehouse_is_disabled(monkeypatch, recording_writer):
    """Disabled: the worker looks once, builds nothing, and retires."""
    import research_warehouse.config as config

    monkeypatch.setattr(config, "warehouse_enabled", lambda: False)
    capture = WarehouseTeeCapture()

    assert capture.submit(_FakeBot(_cache())) is True  # accepted, then discarded
    assert capture.wait_idle(5.0)
    assert capture.disabled is True
    assert recording_writer.constructed_on == []
    assert capture.rows_spooled == 0
    # Once disabled, further submits are refused outright.
    assert capture.submit(_FakeBot(_cache())) is False
    capture.close()


def test_the_mailbox_is_latest_wins_and_cannot_backlog(enabled, recording_writer):
    """A slow disk must not build a queue; the next tick re-offers the bars."""
    capture = WarehouseTeeCapture()
    for index in range(50):
        capture.submit(_FakeBot(_cache()), now=OPEN_UTC + timedelta(minutes=20 + index))
    assert capture.wait_idle(5.0)
    capture.close()
    # 50 submits, one mailbox: far fewer captures than submits, and no growth.
    assert capture.captures <= 50
    assert capture._pending is None


def test_a_stop_start_resumes_the_same_capture_and_keeps_its_dedupe(enabled):
    """The lake's publish does not dedupe, so `seen` must survive a restart."""
    spool = _RecordingSpool()
    capture = WarehouseTeeCapture(spool=spool)
    cache = _cache()

    capture.submit(_FakeBot(cache), now=OPEN_UTC + timedelta(minutes=20))
    assert capture.wait_idle(5.0)
    first_rows = capture.rows_spooled
    assert first_rows > 0

    capture.close()  # Stop

    capture.submit(_FakeBot(cache), now=OPEN_UTC + timedelta(minutes=21))  # Start
    assert capture.wait_idle(5.0)
    capture.close()
    assert capture.rows_spooled == first_rows, "the same bars must not be re-spooled"


# --- D21: the service-layer lifecycle -------------------------------------
def test_the_service_slot_submits_without_building_a_writer(enabled, recording_writer):
    from ui.services.bounce_service import BounceService

    service = BounceService()
    try:
        service._bot = _FakeBot(_cache())
        service.capture_warehouse_tee()
        # The GUI slot returned with nothing constructed on this thread.
        assert threading.current_thread() not in recording_writer.constructed_on
        capture = service._warehouse_capture
        assert capture is not None
        assert capture.wait_idle(5.0)
        assert capture.rows_spooled > 0
    finally:
        service._bot = None
        service.shutdown()


def test_shutdown_closes_the_tee_worker(enabled, recording_writer):
    from ui.services.bounce_service import BounceService

    service = BounceService()
    service._bot = _FakeBot(_cache())
    service.capture_warehouse_tee()
    capture = service._warehouse_capture
    assert capture.wait_idle(5.0)

    service._bot = None
    service.shutdown()
    worker = capture._worker
    assert worker is None or not worker.is_alive(), "the worker is retired on shutdown"


def test_a_disabled_warehouse_stops_the_service_timer(monkeypatch):
    import research_warehouse.config as config
    from ui.services.bounce_service import BounceService

    monkeypatch.setattr(config, "warehouse_enabled", lambda: False)
    service = BounceService()
    try:
        service._bot = _FakeBot(_cache())
        service.capture_warehouse_tee()
        capture = service._warehouse_capture
        assert capture.wait_idle(5.0)
        # Second tick sees `disabled` and stops the timer rather than spinning.
        service.capture_warehouse_tee()
        assert capture.disabled is True
        assert not service._warehouse_timer.isActive()
    finally:
        service._bot = None
        service.shutdown()


# --- the post-scan build hook ---------------------------------------------
# These three used to assert the build ran on a THREAD inside the desk, with
# `run_build` monkeypatched in-process. Packet F1 (2026-09-03) moved it to a
# child process, because that thread held the GIL in 82.7% of py-spy samples
# for 27-57 minutes per scan and froze the desk for a morning. What each test
# asks is unchanged - the build is invoked, it is not inline, only one runs,
# and a disabled warehouse does nothing - only the mechanism it asks about is.
# The child's argv, priority class and environment are pinned separately, in
# tests/test_warehouse_build_child.py.
class _FakeBuildChild:
    """Alive until the test releases it, like the real build child."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pid = 4242
        self.returncode = None
        self._exit = threading.Event()

    def poll(self):
        return 0 if self._exit.is_set() else None

    def wait(self, timeout=None):
        self.returncode = 0
        self._exit.set()
        return 0

    def communicate(self, *a, **k):
        self._exit.wait(30.0)
        return ("", "")

    def terminate(self):
        self.wait()


def _capture_build_children(monkeypatch):
    from ui.services import scan_service as scan_service_mod

    children = []

    def fake_popen(*args, **kwargs):
        child = _FakeBuildChild(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(scan_service_mod.subprocess, "Popen", fake_popen)
    return children


def test_a_finished_scan_starts_the_warehouse_build(enabled, monkeypatch):
    """LD-01's 'post-scan build job' had no invoker at all.

    Without it the tee spools M5 bars every minute and nothing seals them - and
    M5 segments are PROTECTED, so the backlog grows until Health goes red.
    """
    from ui.services.scan_service import ScanService, warehouse_build_command

    children = _capture_build_children(monkeypatch)
    service = ScanService()
    try:
        assert service.start_warehouse_build("run-42") is True
        assert len(children) == 1
        assert children[0].args[0] == warehouse_build_command("run-42")
        assert children[0].kwargs["env"]["TRADINGBOT_RUN_ID"] == "run-42"
    finally:
        service.wait_for_warehouse_build(10.0)


def test_the_build_runs_out_of_process_and_never_breaks_the_scan(enabled, monkeypatch):
    """It must not run inline, and a child that cannot start must not escape."""
    from ui.services import scan_service as scan_service_mod
    from ui.services.scan_service import ScanService
    import research_warehouse.cli as warehouse_cli

    ran_inline = []
    monkeypatch.setattr(
        warehouse_cli, "run_build", lambda *a, **k: ran_inline.append(1)
    )

    def refuse(*args, **kwargs):
        raise OSError("no such executable")

    monkeypatch.setattr(scan_service_mod.subprocess, "Popen", refuse)

    service = ScanService()
    try:
        # A build that cannot even be spawned is a False, never an exception:
        # research evidence must never be able to break a scan.
        assert service.start_warehouse_build("run-1") is False
        assert ran_inline == [], "the build ran inside the desk process"
    finally:
        service.wait_for_warehouse_build(10.0)


def test_only_one_warehouse_build_runs_at_a_time(enabled, monkeypatch):
    """A second scan finishing mid-build is skipped, not stacked."""
    from ui.services.scan_service import ScanService

    children = _capture_build_children(monkeypatch)
    service = ScanService()
    try:
        assert service.start_warehouse_build("first") is True
        assert service.start_warehouse_build("second") is False, "not stacked"
        assert len(children) == 1
    finally:
        service.wait_for_warehouse_build(10.0)


def test_a_disabled_warehouse_builds_nothing_after_a_scan(monkeypatch):
    import research_warehouse.config as config
    from ui.services.scan_service import ScanService

    monkeypatch.setattr(config, "warehouse_enabled", lambda: False)
    children = _capture_build_children(monkeypatch)

    service = ScanService()
    try:
        assert service.start_warehouse_build("run-1") is False
        assert children == []
    finally:
        service.wait_for_warehouse_build(10.0)


# --- BD-96: the mark is persisted, and the clock never resets it -----------
def test_a_utc_midnight_does_not_re_spool_the_cache(enabled):
    """The old dedupe state was a set keyed on the UTC date, so at 17:00 PT
    every evening the whole five-day cache went to the spool again (346,111
    rows / 240 MB on 2026-09-03) and, because the seal did not dedupe, into
    the lake. A capture just after 00:00 UTC must add nothing."""
    spool = _RecordingSpool()
    capture = WarehouseTeeCapture(spool=spool)
    late = OPEN_UTC.replace(hour=23, minute=59)
    capture.capture(_FakeBot(_cache()), now=late)
    spooled = capture.rows_spooled
    assert spooled > 0

    again = capture.capture(_FakeBot(_cache()), now=late + timedelta(minutes=2))  # 00:01 UTC, next date

    assert capture.rows_spooled == spooled
    assert again is not None and again.rows_published == 0


def test_the_high_water_mark_survives_a_desk_restart(enabled, tmp_path):
    """A restart used to mean a fresh `seen` set and a full re-spool (107,119
    rows at 13:05 PT on 2026-09-03). The mark lives beside the spool, so a NEW
    capture object over the same directory resumes where the last one stopped."""
    from research_warehouse.spool import ResearchSpoolWriter

    first = WarehouseTeeCapture(spool=ResearchSpoolWriter(tmp_path / "spool"))
    report = first.capture(_FakeBot(_cache()), now=OPEN_UTC + timedelta(minutes=20))
    assert report is not None and report.rows_published == 6
    assert (tmp_path / "spool" / WarehouseTeeCapture.HIGH_WATER_NAME).exists()

    restarted = WarehouseTeeCapture(spool=ResearchSpoolWriter(tmp_path / "spool"))
    again = restarted.capture(_FakeBot(_cache()), now=OPEN_UTC + timedelta(minutes=21))

    assert again is not None and again.rows_published == 0 and again.symbols_unchanged == 2


def test_an_unreadable_high_water_file_starts_empty_and_never_raises(enabled, tmp_path):
    import json

    from research_warehouse.spool import ResearchSpoolWriter

    writer = ResearchSpoolWriter(tmp_path / "spool")
    (tmp_path / "spool" / WarehouseTeeCapture.HIGH_WATER_NAME).write_text("{not json", encoding="utf-8")
    capture = WarehouseTeeCapture(spool=writer)
    report = capture.capture(_FakeBot(_cache()), now=OPEN_UTC + timedelta(minutes=20))
    assert report is not None and report.rows_published == 6
    # ...and the bad file is replaced by a good one on the way out.
    payload = json.loads((tmp_path / "spool" / WarehouseTeeCapture.HIGH_WATER_NAME).read_text(encoding="utf-8"))
    assert set(payload["high_water"]) == {"AAPL", "MSFT"}
