"""R4 section 3: the early-morning D1 gap distortion.

Found mechanism (spec recon 2026-08-15): the forming D1 preview candle is built
from IB RTH M5 bars when an M5 cache exists, which is accurate -- but falls back
to a Yahoo `interval="1d"` "today" row taken verbatim as OHLC whenever no M5
cache exists yet. In the first minutes after the open that row is a thin
pre-market/early print, and it both mis-states the gap and drives the chart's Y
autoscale. The trader read this as a broken axis; it is the data.

The fix is not to guess a better bar. It is to refuse to draw one during the
window where the Yahoo row is known to be untrustworthy, and to say which source
the preview came from when one IS drawn. Missing data renders as absence with a
caveat, never as a confident candle (plan.md sec 5).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")

from ui.widgets.symbol_snapshot_dialog import (  # noqa: E402
    YAHOO_FORMING_SUPPRESS_MINUTES,
    yahoo_forming_bar_is_trustworthy,
)


def _open_at(day: str = "2026-08-17") -> datetime:
    """Today's regular open on the market-local clock (06:30 PT)."""
    from market_session import get_market_session_window

    window = get_market_session_window(datetime.fromisoformat(f"{day}T12:00:00").date())
    return window.open_local.replace(tzinfo=None)


# --------------------------------------------------------------------------
# the suppression window
# --------------------------------------------------------------------------
def test_a_yahoo_row_inside_the_window_is_not_trusted():
    opened = _open_at()
    assert not yahoo_forming_bar_is_trustworthy(opened + timedelta(minutes=1))
    assert not yahoo_forming_bar_is_trustworthy(opened + timedelta(minutes=14))


def test_a_yahoo_row_after_the_window_is_trusted():
    opened = _open_at()
    assert yahoo_forming_bar_is_trustworthy(opened + timedelta(minutes=16))
    assert yahoo_forming_bar_is_trustworthy(opened + timedelta(hours=3))


def test_the_boundary_minute_is_pinned_exactly():
    """A mutation check: flipping the comparison or going off by one minute
    moves exactly this assertion, and nothing else in the file catches it."""
    opened = _open_at()
    edge = opened + timedelta(minutes=YAHOO_FORMING_SUPPRESS_MINUTES)
    assert not yahoo_forming_bar_is_trustworthy(edge - timedelta(seconds=1))
    assert yahoo_forming_bar_is_trustworthy(edge)


def test_before_the_open_there_is_no_session_to_distort():
    """Pre-market: nothing has opened, so the Yahoo row is not an early print
    of today's session at all and the suppression window does not apply."""
    opened = _open_at()
    assert yahoo_forming_bar_is_trustworthy(opened - timedelta(minutes=30))


def test_a_weekend_is_trusted_because_nothing_is_forming():
    """Saturday: session_has_opened is False, so there is no early-print
    window to be inside of. Fail-open - the caller finds no today-dated bar."""
    assert yahoo_forming_bar_is_trustworthy(datetime(2026, 8, 15, 8, 0))


def test_zero_minutes_disables_the_suppression(monkeypatch):
    import ui.widgets.symbol_snapshot_dialog as mod

    monkeypatch.setattr(mod, "_suppress_minutes", lambda: 0)
    opened = _open_at()
    assert yahoo_forming_bar_is_trustworthy(opened + timedelta(seconds=1))


def test_a_configured_window_is_honored(monkeypatch):
    import ui.widgets.symbol_snapshot_dialog as mod

    monkeypatch.setattr(mod, "_suppress_minutes", lambda: 45)
    opened = _open_at()
    assert not yahoo_forming_bar_is_trustworthy(opened + timedelta(minutes=30))
    assert yahoo_forming_bar_is_trustworthy(opened + timedelta(minutes=46))


def test_an_unreadable_setting_falls_back_to_the_default(monkeypatch):
    """A corrupt settings file must not silently disable the guard."""
    import ui.widgets.symbol_snapshot_dialog as mod

    def boom(*_a, **_k):
        raise OSError("settings unreadable")

    monkeypatch.setattr("project_paths.get_local_setting", boom)
    assert mod._suppress_minutes() == YAHOO_FORMING_SUPPRESS_MINUTES


def test_a_session_lookup_failure_fails_open(monkeypatch):
    """Missing data is uncertainty. If we cannot tell whether we are inside
    the early window, suppressing every preview all day would be worse than
    the distortion - so this one fails OPEN and the source label carries the
    caveat instead."""
    import chart_snapshot

    def boom(*_a, **_k):
        raise RuntimeError("no session calendar")

    monkeypatch.setattr(chart_snapshot, "session_has_opened", boom)
    assert yahoo_forming_bar_is_trustworthy(datetime(2026, 8, 17, 6, 31))


# --------------------------------------------------------------------------
# the label
# --------------------------------------------------------------------------
def test_a_yahoo_preview_says_so():
    from ui.widgets.symbol_snapshot_dialog import forming_preview_caveat

    assert "Yahoo" in forming_preview_caveat("yfinance-fallback")


def test_an_ib_preview_needs_no_caveat():
    from ui.widgets.symbol_snapshot_dialog import forming_preview_caveat

    assert forming_preview_caveat("ibkr-cache") == ""


def test_a_suppressed_preview_says_it_is_absent_not_flat():
    """The distinction the invariant turns on: no candle, and a reason."""
    from ui.widgets.symbol_snapshot_dialog import forming_preview_caveat

    text = forming_preview_caveat("yfinance-fallback", suppressed=True)
    assert "not shown" in text.lower()
    assert "open" in text.lower()


# --------------------------------------------------------------------------
# through the real request seam
#
# A helper that returns the right boolean proves nothing about whether the
# chart stops drawing the bar. These drive SymbolSnapshotWidget._request_
# snapshots and read what it actually handed the data service.
# --------------------------------------------------------------------------
_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")

pytestmark = pytest.mark.qt


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


@pytest.fixture
def widget(monkeypatch):
    import ui.widgets.symbol_snapshot_dialog as mod

    mod._FORMING_BARS.clear()
    mod._FORMING_ATTEMPTS.clear()
    w = mod.SymbolSnapshotWidget()
    requests: list[dict] = []

    class _Data:
        def request(self, symbol, m5_bars, **kwargs):
            requests.append({"symbol": symbol, "m5_bars": m5_bars, **kwargs})
            return 1

    w._data = _Data()
    w.requests = requests
    yield w
    mod._FORMING_BARS.clear()
    w.deleteLater()


def _cache_forming_bar(symbol: str = "AAPL") -> dict:
    import ui.widgets.symbol_snapshot_dialog as mod

    bar = {
        "dt": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        "open": 100.0,
        "high": 100.2,
        "low": 99.9,
        "close": 100.1,
        "volume": 1_000.0,
        "preview": True,
    }
    mod._FORMING_BARS[symbol] = (datetime.now(), bar)
    return bar


def test_the_early_yahoo_row_is_not_sent_to_the_chart(widget, monkeypatch):
    import ui.widgets.symbol_snapshot_dialog as mod

    monkeypatch.setattr(mod, "yahoo_forming_bar_is_trustworthy", lambda *a, **k: False)
    _cache_forming_bar()
    widget._symbol = "AAPL"
    widget._request_snapshots()
    assert widget.requests[-1]["d1_preview_bars"] == []
    assert widget._forming_suppressed is True


def test_a_settled_yahoo_row_is_sent(widget, monkeypatch):
    import ui.widgets.symbol_snapshot_dialog as mod

    monkeypatch.setattr(mod, "yahoo_forming_bar_is_trustworthy", lambda *a, **k: True)
    _cache_forming_bar()
    widget._symbol = "AAPL"
    widget._request_snapshots()
    assert len(widget.requests[-1]["d1_preview_bars"]) == 1
    assert widget._forming_suppressed is False


def test_ib_m5_bars_are_never_suppressed(widget, monkeypatch):
    """The window only ever gates the Yahoo fallback. When real RTH bars
    exist the preview is accurate and must keep rendering at 06:31."""
    import ui.widgets.symbol_snapshot_dialog as mod

    monkeypatch.setattr(mod, "yahoo_forming_bar_is_trustworthy", lambda *a, **k: False)
    _cache_forming_bar()

    class _Bot:
        def m5_chart_bars(self, _symbol, max_sessions=2):
            return [{"dt": datetime.now(), "close": 100.0, "volume": 5.0}]

    widget._symbol = "AAPL"
    widget._bot = _Bot()
    widget._request_snapshots()
    assert len(widget.requests[-1]["d1_preview_bars"]) == 1
    assert widget.requests[-1]["source"] == "ibkr-cache"
    assert widget._forming_suppressed is False


def test_a_suppressed_preview_still_reports_the_yahoo_source(widget, monkeypatch):
    """Not 'durable-store'. The trader has to be able to tell 'no forming bar
    because none exists' from 'no forming bar because I refused to draw it'."""
    import ui.widgets.symbol_snapshot_dialog as mod

    monkeypatch.setattr(mod, "yahoo_forming_bar_is_trustworthy", lambda *a, **k: False)
    _cache_forming_bar()
    widget._symbol = "AAPL"
    widget._request_snapshots()
    assert widget.requests[-1]["source"] == "yfinance-fallback"


def test_a_symbol_with_no_forming_bar_reports_the_durable_store(widget, monkeypatch):
    import ui.widgets.symbol_snapshot_dialog as mod

    monkeypatch.setattr(mod, "yahoo_forming_bar_is_trustworthy", lambda *a, **k: False)
    widget._symbol = "AAPL"
    widget._request_snapshots()
    assert widget.requests[-1]["source"] == "durable-store"
    assert widget._forming_suppressed is False
