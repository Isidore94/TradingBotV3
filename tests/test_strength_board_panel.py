"""The strength board surface and its add-to-Focus path (R2 Part B.3.4)."""

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _row(symbol, *, last, prev_high, prev_low, vwap, strength=10.0):
    return {
        "symbol": symbol, "strength": strength, "last": last,
        "prev_high": prev_high, "prev_low": prev_low, "session_vwap": vwap,
        "day_pct": 1.5, "vwap_distance_pct": 0.8,
    }


class _Signal:
    """Just enough Signal for the panel's connects."""

    def connect(self, _slot):
        pass


class _Service:
    def __init__(self, board):
        self._board = board
        self.refreshed = 0
        self.boardChanged = _Signal()
        self.statusChanged = _Signal()

    def board(self):
        return dict(self._board)

    def status_text(self):
        return "Strength board: test"

    def refresh_now(self):
        self.refreshed += 1
        return True


def _panel(board, tmp_path):
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - PySide6 is on the desk
        pytest.skip("PySide6 is not installed")
    QApplication.instance() or QApplication([])
    from focus_picks import FocusPickStore
    from ui.panels.strength_board_panel import StrengthBoardPanel
    from ui.services.focus_service import FocusService

    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "membership.json",
    )
    service = _Service(board)
    panel = StrengthBoardPanel(service=service, focus_service=FocusService(store))
    return panel, store, service


def test_the_board_renders_both_sides(tmp_path):
    board = {
        "long": [_row("NVDA", last=105.0, prev_high=100.0, prev_low=98.0, vwap=101.0)],
        "short": [_row("SOXS", last=95.0, prev_high=102.0, prev_low=100.0, vwap=97.0)],
    }
    panel, _store, _service = _panel(board, tmp_path)
    assert panel.longs.table.rowCount() == 1
    assert panel.shorts.table.rowCount() == 1
    assert panel.longs.table.item(0, 0).text() == "NVDA"


def test_a_qualifying_row_adds_to_focus(tmp_path):
    board = {"long": [_row("NVDA", last=105.0, prev_high=100.0, prev_low=98.0, vwap=101.0)],
             "short": []}
    panel, store, _service = _panel(board, tmp_path)
    messages: list[str] = []
    panel.statusChanged.connect(messages.append)

    panel._add_one("NVDA", "long")
    assert store.focus_symbols("long", "m5") == ["NVDA"]
    assert any("added to M5 Focus" in text for text in messages)


def test_a_row_that_has_fallen_back_is_refused_with_its_reason(tmp_path):
    """The board is up to 15 minutes stale, so the gate runs at click time."""
    board = {"long": [_row("NVDA", last=105.0, prev_high=100.0, prev_low=98.0, vwap=106.0)],
             "short": []}
    panel, store, _service = _panel(board, tmp_path)
    messages: list[str] = []
    panel.statusChanged.connect(messages.append)

    panel._add_one("NVDA", "long")
    assert store.focus_symbols("long", "m5") == []
    assert any("not above session VWAP" in text for text in messages)


def test_add_all_names_the_refusals_rather_than_counting_them(tmp_path):
    board = {
        "long": [
            _row("GOOD", last=105.0, prev_high=100.0, prev_low=98.0, vwap=101.0),
            _row("UNDER", last=105.0, prev_high=100.0, prev_low=98.0, vwap=106.0),
            _row("INSIDE", last=99.0, prev_high=100.0, prev_low=98.0, vwap=97.0),
        ],
        "short": [],
    }
    panel, store, _service = _panel(board, tmp_path)
    messages: list[str] = []
    panel.statusChanged.connect(messages.append)

    panel._add_all("long")
    assert store.focus_symbols("long", "m5") == ["GOOD"]
    text = " ".join(messages)
    assert "UNDER (not above session VWAP)" in text
    assert "INSIDE (not above yesterday's high)" in text


def test_an_unmeasurable_row_is_refused_not_added(tmp_path):
    """Missing data is uncertainty, never confirmation."""
    board = {"long": [_row("NVDA", last=105.0, prev_high=100.0, prev_low=98.0, vwap=None)],
             "short": []}
    panel, store, _service = _panel(board, tmp_path)
    messages: list[str] = []
    panel.statusChanged.connect(messages.append)

    panel._add_one("NVDA", "long")
    assert store.focus_symbols("long", "m5") == []
    assert any("session VWAP" in text for text in messages)


def test_a_board_add_is_the_traders_like_not_the_machines(tmp_path, monkeypatch):
    """The trader clicked it, so it belongs in the pick-feedback log - unlike
    auto-adoption, which writes through the store to stay out of it."""
    board = {"long": [_row("NVDA", last=105.0, prev_high=100.0, prev_low=98.0, vwap=101.0)],
             "short": []}
    panel, store, _service = _panel(board, tmp_path)
    rows: list[dict] = []
    monkeypatch.setattr(
        "ui.services.focus_service.record_pick_feedback", lambda **kw: rows.append(kw)
    )
    monkeypatch.setattr(store, "uses_default_paths", lambda: True)

    panel._add_one("NVDA", "long")
    assert len(rows) == 1
    assert rows[0]["verdict"] == "like" and rows[0]["origin"] == "strength_board"


def test_the_refresh_button_asks_the_service(tmp_path):
    panel, _store, service = _panel({"long": [], "short": []}, tmp_path)
    panel._refresh()
    assert service.refreshed == 1


def test_a_missing_number_shows_a_dash_not_a_zero(tmp_path):
    """An unmeasured field must not read as a measured zero."""
    from ui.panels.strength_board_panel import _fmt

    assert _fmt(None, 2) == "—"
    assert _fmt(float("nan"), 2) == "—"
    assert _fmt(1.5, 2, signed=True, suffix="%") == "+1.50%"
