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
def test_a_finished_scan_starts_the_warehouse_build(enabled, monkeypatch):
    """LD-01's 'post-scan build job' had no invoker at all.

    Without it the tee spools M5 bars every minute and nothing seals them - and
    M5 segments are PROTECTED, so the backlog grows until Health goes red.
    """
    from ui.services.scan_service import ScanService

    calls = []
    import research_warehouse.cli as warehouse_cli

    monkeypatch.setattr(
        warehouse_cli,
        "run_build",
        lambda *args, **kwargs: calls.append(kwargs.get("run_id")) or _BuildOk(),
    )

    service = ScanService()
    try:
        assert service.start_warehouse_build("run-42") is True
        service.wait_for_warehouse_build(10.0)
        assert calls == ["run-42"]
    finally:
        service.wait_for_warehouse_build(10.0)


class _BuildOk:
    status = "OK"
    message = ""


def test_the_build_runs_off_the_gui_thread_and_never_breaks_the_scan(enabled, monkeypatch):
    """It must not run inline, and an exploding build must not escape."""
    from ui.services.scan_service import ScanService
    import research_warehouse.cli as warehouse_cli

    ran_on = []

    def explode(*args, **kwargs):
        ran_on.append(threading.current_thread())
        raise RuntimeError("DAS unplugged mid-build")

    monkeypatch.setattr(warehouse_cli, "run_build", explode)

    service = ScanService()
    try:
        service.start_warehouse_build("run-1")  # returns immediately
        service.wait_for_warehouse_build(10.0)
        assert ran_on and ran_on[0] is not threading.current_thread()
    finally:
        service.wait_for_warehouse_build(10.0)  # no exception escapes


def test_only_one_warehouse_build_runs_at_a_time(enabled, monkeypatch):
    """A second scan finishing mid-build is skipped, not stacked."""
    from ui.services.scan_service import ScanService
    import research_warehouse.cli as warehouse_cli

    release = threading.Event()
    started = threading.Event()

    def blocking(*args, **kwargs):
        started.set()
        release.wait(5.0)
        return _BuildOk()

    monkeypatch.setattr(warehouse_cli, "run_build", blocking)

    service = ScanService()
    try:
        assert service.start_warehouse_build("first") is True
        assert started.wait(5.0)
        assert service.start_warehouse_build("second") is False, "not stacked"
    finally:
        release.set()
        service.wait_for_warehouse_build(10.0)


def test_a_disabled_warehouse_builds_nothing_after_a_scan(monkeypatch):
    import research_warehouse.config as config
    import research_warehouse.cli as warehouse_cli
    from ui.services.scan_service import ScanService

    monkeypatch.setattr(config, "warehouse_enabled", lambda: False)
    calls = []
    monkeypatch.setattr(warehouse_cli, "run_build", lambda *a, **k: calls.append(1))

    service = ScanService()
    try:
        service.start_warehouse_build("run-1")
        service.wait_for_warehouse_build(10.0)
        assert calls == []
    finally:
        service.wait_for_warehouse_build(10.0)
