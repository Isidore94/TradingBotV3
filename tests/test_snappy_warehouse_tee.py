"""The warehouse tee never pulls the scanner child's bar cache on the GUI thread.

On the process proxy `bot.latest_bars` is a full-cache pickle over the child
pipe. The GUI slot must only hand the worker the bot; the worker reads it.
"""

import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
TESTS_DIR = ROOT_DIR / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from test_qt_warehouse_tee import _RecordingSpool, _cache  # noqa: E402
from ui.services.warehouse_service import WarehouseTeeCapture  # noqa: E402

OPEN_UTC = datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc)


class _SlowProxy:
    """A process proxy whose `latest_bars` is a slow RPC. Records the reading thread."""

    is_process_proxy = True

    def __init__(self, cache, delay=2.0):
        self._cache = cache
        self._delay = delay
        self.read_on: list[threading.Thread] = []

    @property
    def latest_bars(self):
        self.read_on.append(threading.current_thread())
        time.sleep(self._delay)
        return dict(self._cache)


@pytest.fixture()
def enabled(monkeypatch, tmp_path):
    import research_warehouse.config as config

    monkeypatch.setattr(config, "warehouse_enabled", lambda: True)
    monkeypatch.setattr(config, "get_research_store_dir", lambda: tmp_path / "lake")
    return config


def test_submit_on_a_proxy_returns_at_once_and_reads_bars_on_the_worker(enabled):
    spool = _RecordingSpool()
    capture = WarehouseTeeCapture(spool=spool)
    bot = _SlowProxy(_cache(), delay=2.0)
    main = threading.current_thread()

    started = time.perf_counter()
    assert capture.submit(bot, now=OPEN_UTC + timedelta(minutes=20)) is True
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2, f"GUI thread blocked {elapsed:.2f}s on the child RPC"
    assert capture.wait_idle(10.0)
    capture.close()
    assert bot.read_on, "the worker still reads the cache"
    assert all(thread is not main for thread in bot.read_on)
    assert capture.rows_spooled > 0


def test_a_proxy_read_failure_is_swallowed_on_the_worker(enabled):
    class _Broken:
        is_process_proxy = True

        @property
        def latest_bars(self):
            raise RuntimeError("BounceBot child is not running")

    capture = WarehouseTeeCapture(spool=_RecordingSpool())
    assert capture.submit(_Broken()) is True
    assert capture.wait_idle(5.0)
    capture.close()
    assert capture.rows_spooled == 0
