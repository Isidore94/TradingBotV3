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


#: V1's relative volume compares each of the last 12 bars with the same offset
#: over the PRIOR 15 SESSIONS, so a fixture needs sixteen sessions or every row
#: is greyed with "relative volume not measurable" and the test is about the
#: fixture rather than the board.
_SESSIONS = 16
_BARS_PER_SESSION = 78


def _bars(close, *, prev_high, prev_low, opening, today_volume=1000.0):
    """Sixteen sessions of flat history, then today's move.

    Today's volume is a parameter because the board's second volume cut asks
    whether the SESSION is busy, and a fixture where every symbol traded exactly
    the same amount cannot exercise it.
    """
    bars = []
    mid = (prev_high + prev_low) / 2
    for session in range(_SESSIONS - 1):
        start = datetime(2026, 6, 1, 6, 30) + timedelta(days=session)
        for index in range(_BARS_PER_SESSION):
            bars.append({"dt": start + timedelta(minutes=5 * index), "open": mid,
                         "high": prev_high, "low": prev_low, "close": mid,
                         "volume": 1000.0})
    today = datetime(2026, 7, 2, 6, 30)
    for index in range(20):
        bars.append({"dt": today + timedelta(minutes=5 * index), "open": opening,
                     "high": max(opening, close) + 0.1, "low": min(opening, close) - 0.1,
                     "close": close, "volume": today_volume})
    return bars


def _daily(level):
    """210 daily closes at `level` - enough for the 200 SMA the V1 floors need."""
    start = datetime(2025, 9, 1)
    return [
        {"dt": start + timedelta(days=index), "open": level, "high": level,
         "low": level, "close": level, "volume": 1_000_000.0}
        for index in range(210)
    ]


def _downloader(mapping, daily=None):
    """Answers the 5m and the 1d call separately.

    V1 added a second batched download for the D1 SMA floors, so a stub that
    returned M5 bars to both would leave every row unable to measure its 200 SMA
    - which fails the floor, correctly, and would make this test about the stub.
    """
    daily = daily or {}

    class _Data:
        def __init__(self, source):
            self._source = source

        def __getitem__(self, symbol):
            return _Frame(self._source.get(symbol) or [])

    def _call(chunk, **kwargs):
        if str(kwargs.get("interval") or "") == "1d":
            return _Data(daily)
        return _Data(mapping)

    return _call


def test_the_pipeline_turns_bars_into_a_board():
    from ui.services.strength_board_service import build_board

    mapping = {
        "STRONG": _bars(105.0, prev_high=100.0, prev_low=98.0, opening=101.0),
        "WEAK": _bars(95.0, prev_high=102.0, prev_low=100.0, opening=99.0),
    }
    # STRONG sits above its daily SMAs and WEAK below its own, so each clears
    # the floors on the side it belongs to.
    # The volume cuts are OFF here (fraction 1.0). They keep the busier HALF of
    # the measured population, and with two symbols that is one of them - so on
    # a two-name fixture they decide the test rather than the pipeline does.
    # `test_the_volume_cuts_keep_the_busier_half` exercises them on a population.
    board = build_board(
        symbols=list(mapping),
        downloader=_downloader(mapping, {"STRONG": _daily(50.0), "WEAK": _daily(150.0)}),
        fraction=1.0,
        rvol_fraction=1.0,
        session_volume_fraction=1.0,
        now=NOW,
    )
    assert board["offered"] == 2
    assert board["measured"] == 2
    # CHANGED BY V1: the board keeps the rows that miss a filter, greyed and
    # carrying why, so `long` holds both and `long_picks` counts the picks.
    picks = [row["symbol"] for row in board["long"] if row["passes_floors"]]
    assert picks == ["STRONG"]
    assert board["long_picks"] == 1
    # Both rows appear on both sides now - the strength cut is 1.0 here - and
    # what separates them is `passes_floors`, not membership.
    assert sorted(row["symbol"] for row in board["short"]) == ["STRONG", "WEAK"]
    short_picks = [row["symbol"] for row in board["short"] if row["passes_floors"]]
    assert short_picks == ["WEAK"]


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


def test_the_volume_cuts_keep_the_busier_half():
    """The trader's second filter, on a population rather than two names.

    Both halves of it: the per-bar relative volume, and today's session volume.
    A name can clear the first on twelve quiet bars that are merely less quiet
    than usual, which is why the trader asks for both.
    """
    from ui.services.strength_board_service import build_board

    # Four names, identical in every way except how much they traded today.
    mapping = {
        f"N{index}": _bars(
            105.0, prev_high=100.0, prev_low=98.0, opening=101.0,
            today_volume=1000.0 * (index + 1),
        )
        for index in range(4)
    }
    daily = {symbol: _daily(50.0) for symbol in mapping}

    board = build_board(
        symbols=list(mapping), downloader=_downloader(mapping, daily),
        fraction=1.0, now=NOW,
    )

    by_symbol = {row["symbol"]: row for row in board["long"]}
    assert len(by_symbol) == 4, "nothing is hidden by a volume cut"
    # The two busiest are in; the two quietest are greyed and say why.
    # Ordered by STRENGTH, which is identical here, so the order is the input's.
    # What the cut decides is membership, not rank.
    assert [row["symbol"] for row in board["long"] if row["passes_floors"]] == ["N2", "N3"]
    for symbol in ("N0", "N1"):
        reasons = by_symbol[symbol]["failed_floors"]
        assert any("busier half" in reason for reason in reasons), reasons

