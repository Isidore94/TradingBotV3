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
