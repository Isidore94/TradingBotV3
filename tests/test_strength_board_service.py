"""The strength board's owner: fetch, single flight, last good (R2 Part B)."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

NOW = datetime(2026, 7, 2, 11, 0)


class _Frame:
    """Minimal stand-in for the per-symbol frame `_frame_rows` reads."""

    def __init__(self, rows):
        self._rows = rows
        self.empty = not rows

    def iterrows(self):
        for row in self._rows:
            yield row["dt"], {
                "Open": row["open"], "High": row["high"],
                "Low": row["low"], "Close": row["close"], "Volume": row["volume"],
            }


def _bars(close, *, prev_high, prev_low, opening):
    bars = []
    day_one = datetime(2026, 7, 1, 6, 30)
    mid = (prev_high + prev_low) / 2
    for index in range(60):
        bars.append({"dt": day_one + timedelta(minutes=5 * index), "open": mid,
                     "high": prev_high, "low": prev_low, "close": mid, "volume": 1000.0})
    day_two = datetime(2026, 7, 2, 6, 30)
    for index in range(20):
        bars.append({"dt": day_two + timedelta(minutes=5 * index), "open": opening,
                     "high": max(opening, close) + 0.1, "low": min(opening, close) - 0.1,
                     "close": close, "volume": 1000.0})
    return bars


def _downloader(mapping):
    class _Data:
        def __getitem__(self, symbol):
            return _Frame(mapping.get(symbol) or [])

    return lambda chunk, **_kwargs: _Data()


def test_the_pipeline_turns_bars_into_a_board():
    from ui.services.strength_board_service import build_board

    mapping = {
        "STRONG": _bars(105.0, prev_high=100.0, prev_low=98.0, opening=101.0),
        "WEAK": _bars(95.0, prev_high=102.0, prev_low=100.0, opening=99.0),
    }
    board = build_board(symbols=list(mapping), downloader=_downloader(mapping),
                        fraction=1.0, now=NOW)
    assert board["offered"] == 2
    assert board["measured"] == 2
    assert [row["symbol"] for row in board["long"]] == ["STRONG"]
    assert [row["symbol"] for row in board["short"]] == ["WEAK"]


def test_the_forming_bar_is_excluded():
    """A board ranked on a forming bar would reshuffle every few seconds."""
    from ui.services.strength_board_service import _completed_bars

    rows = [
        {"dt": datetime(2026, 7, 2, 10, 50)},
        {"dt": datetime(2026, 7, 2, 10, 55)},  # ends 11:00 - complete at 11:00
        {"dt": datetime(2026, 7, 2, 11, 0)},   # ends 11:05 - still forming
    ]
    kept = _completed_bars(rows, NOW)
    assert [row["dt"].strftime("%H:%M") for row in kept] == ["10:50", "10:55"]


def test_a_failed_chunk_costs_its_symbols_not_the_board():
    from ui.services.strength_board_service import build_board

    def explode(chunk, **_kwargs):
        raise RuntimeError("yfinance is down")

    board = build_board(symbols=["AAA"], downloader=explode, now=NOW)
    assert board["measured"] == 0 and board["long"] == []
    assert board["offered"] == 1, "the count still says what was attempted"


def test_an_empty_universe_is_an_honest_empty_board():
    from ui.services.strength_board_service import build_board

    board = build_board(symbols=[], now=NOW)
    assert board["offered"] == 0 and board["long"] == [] and board["short"] == []


def _service(monkeypatch):
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - PySide6 is on the desk
        pytest.skip("PySide6 is not installed")
    QApplication.instance() or QApplication([])
    from ui.services import strength_board_service as module

    service = module.StrengthBoardService()
    service._timer.stop()
    return service, module


def test_quiet_hours_gate_the_scheduled_refresh_but_not_the_manual_one(monkeypatch):
    service, module = _service(monkeypatch)
    monkeypatch.setattr(module.core, "auto_scanning_due", lambda *_a, **_k: (False, "quiet"))
    assert service._due(NOW) is False

    started: list[str] = []
    monkeypatch.setattr(
        "threading.Thread",
        lambda *a, **k: started.append(k.get("name") or "t")
        or type("T", (), {"start": lambda self: None})(),
    )
    assert service.refresh_now() is True, "the trader's button is never gated"
    assert started == ["strength-board"]


def test_only_one_refresh_runs_at_a_time(monkeypatch):
    service, _module = _service(monkeypatch)
    monkeypatch.setattr(
        "threading.Thread",
        lambda *a, **k: type("T", (), {"start": lambda self: None})(),
    )
    assert service.refresh_now() is True
    assert service.refresh_now() is False, "single flight"
    assert service._due(NOW) is False


def test_a_failed_refresh_keeps_the_last_good_board(monkeypatch):
    service, module = _service(monkeypatch)
    good = {"long": [{"symbol": "NVDA"}], "short": [], "offered": 1, "measured": 1}
    monkeypatch.setattr(module, "build_board", lambda **_k: good)
    service._worker()
    assert service.board()["long"] == [{"symbol": "NVDA"}]

    def explode(**_kwargs):
        raise RuntimeError("network gone")

    monkeypatch.setattr(module, "build_board", explode)
    service._worker()
    assert service.board()["long"] == [{"symbol": "NVDA"}], "last good survives"
    # ... and the failure is visible, so a stale board cannot look current.
    assert "FAILED" in service.status_text()
