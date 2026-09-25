"""P2-8 - Market Prep names the morning read on three axes, read off the Qt thread."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402

READ = {
    "basis": "2026-09-24",
    "line": "Market read (from 2026-09-24 close): SPY mixed · breadth weak (27% > SMA20) · internals vol bid + narrow tape",
    "axes": [
        {"axis": "spy", "text": "SPY mixed"},
        {"axis": "breadth", "text": "breadth weak (27% > SMA20)"},
        {"axis": "internals", "text": "internals vol bid + narrow tape"},
    ],
}


def _app():
    return QApplication.instance() or QApplication([])


def test_the_banner_reads_on_a_worker_and_shows_three_axes():
    _app()
    from ui.widgets.market_read_banner import MarketReadBanner

    threads: list[str] = []

    def loader():
        threads.append(threading.current_thread().name)
        return READ

    banner = MarketReadBanner(loader=loader)
    assert threads == []  # the constructor reads nothing
    banner.refresh()
    banner._worker.wait(5000)
    QApplication.processEvents()
    assert threads and threads[0] != threading.main_thread().name
    assert "SPY mixed" in banner.text() and "breadth weak" in banner.text()
    assert "internals vol bid + narrow tape" in banner.text()


def test_no_read_says_so():
    _app()
    from ui.widgets.market_read_banner import NO_READ_TEXT, MarketReadBanner

    banner = MarketReadBanner(loader=lambda: {})
    banner.apply({})
    assert banner.text() == NO_READ_TEXT


def test_market_prep_carries_the_banner():
    _app()
    from ui.panels.master_market_prep_panel import MasterMarketPrepPanel

    panel = MasterMarketPrepPanel()
    assert panel.market_read_banner is not None
    panel.market_read_banner.apply(READ)
    assert panel.market_read_banner.text().startswith("Market read (from 2026-09-24 close)")


def test_latest_morning_read_uses_the_last_completed_session(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import market_axes

    seen: list[str] = []
    monkeypatch.setattr(
        market_axes, "morning_read_for",
        lambda basis: seen.append(basis) or market_axes.morning_read(
            basis, d1_label="trending_down", breadth_row=None, internals_snapshot=None
        ),
    )
    read = market_axes.latest_morning_read(datetime(2026, 9, 25, 8, 0, tzinfo=ZoneInfo("America/New_York")))
    assert seen == ["2026-09-24"]
    assert read["line"].startswith("Market read (from 2026-09-24 close): SPY trending down")
