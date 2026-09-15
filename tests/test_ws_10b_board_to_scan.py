"""Packet WS-10B - the TC2000 board's picks reach the M5 scan, and the board says why not.

WISHLIST 10B. Two separate questions, and this file answers both END TO END
rather than at one seam:

1. **The trace.** Board publication -> eligible rows -> the mode/adoption gate
   -> the persisted Focus membership -> `longs.txt` / `shorts.txt` -> the
   universe BounceBot actually scans. T1.4 (2026-09-04) built the first three
   links and `test_t1_capture_and_board_focus.py` pins them; nothing pinned the
   last two, so a break anywhere past the store would have been invisible.
   `test_the_board_reaches_the_bouncebot_scan_set` is the verification.
2. **The reason.** Before WS-10B the only record of why a board row did NOT
   join Focus was a counted string inside one review-event row; the trader
   reading the board saw a name sitting there with no explanation. Every row
   now carries an `adoption` verdict, computed at the moment
   `_auto_adopt_strength_board` decides, and the M5 Strength Board shows it in
   a last column called `Scan`.

Scanner inclusion and Focus adoption stay DISTINCT contracts: nothing here
scans a row the adoption gate refused. AWAY still STAGES rather than adopting,
through the existing auto-populate queue, so the drain on the return to DESK is
the only thing that ever adopts an unattended pick.

Hermetic: a tmp `FocusPickStore`, a tmp auto-populate pending file, tmp review
events, `_auto_mode_now` monkeypatched. The BounceBot leg reads the tmp
watchlist files with the engine's own `read_tickers` and calls the real
`get_scan_symbol_set` unbound on a stub - no IB, no network, no BounceBot
instance.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _BoardService(QObject):
    """The shape `StrengthBoardService` presents: one dict, one signal."""

    boardChanged = Signal(dict)
    statusChanged = Signal(str)

    def __init__(self, board: dict) -> None:
        super().__init__()
        self._board = dict(board)

    def board(self) -> dict:
        return dict(self._board)

    def publish(self, board: dict) -> None:
        self._board = dict(board)
        self.boardChanged.emit(self.board())

    def status_text(self) -> str:
        return "Strength board: test"

    def refresh_now(self) -> bool:
        return True


def _row(symbol, *, last, prev_high, prev_low, vwap, strength=10.0, failed=()):
    """A board row exactly as `strength_scan.build_strength_board` writes one."""
    failed = list(failed)
    return {
        "symbol": symbol,
        "strength": strength,
        "last": last,
        "prev_high": prev_high,
        "prev_low": prev_low,
        "session_vwap": vwap,
        "day_pct": 1.5,
        "vwap_distance_pct": 0.8,
        "rvol": 1.4,
        "sma200_d1": 10.0,
        "sma100_d1": 10.0,
        "failed_floors": failed,
        "passes_floors": not failed,
    }


# Two parity longs, one parity short, one floor failure, one gate failure, one
# name the trader parked ("Not today"), one they took off today, and one the
# gate cannot measure. Exactly three may ever be adopted.
CLEAN_LONG_NVDA = _row("NVDA", last=105.0, prev_high=100.0, prev_low=98.0, vwap=101.0)
CLEAN_LONG_AMD = _row("AMD", last=55.0, prev_high=50.0, prev_low=48.0, vwap=52.0)
CLEAN_SHORT_XOM = _row("XOM", last=90.0, prev_high=105.0, prev_low=95.0, vwap=92.0)
FLOOR_FAIL_MSFT = _row(
    "MSFT",
    last=310.0,
    prev_high=300.0,
    prev_low=295.0,
    vwap=305.0,
    failed=["not above the D1 200 SMA"],
)
GATE_FAIL_TSLA = _row("TSLA", last=99.0, prev_high=100.0, prev_low=98.0, vwap=97.0)
PARKED_IGN = _row("IGN", last=25.0, prev_high=20.0, prev_low=18.0, vwap=21.0)
DECLINED_DEC = _row("DEC", last=35.0, prev_high=30.0, prev_low=28.0, vwap=31.0)
UNKNOWN_UNK = _row("UNK", last=50.0, prev_high=45.0, prev_low=44.0, vwap=None)

FULL_BOARD = {
    "long": [
        CLEAN_LONG_NVDA,
        CLEAN_LONG_AMD,
        FLOOR_FAIL_MSFT,
        GATE_FAIL_TSLA,
        PARKED_IGN,
        DECLINED_DEC,
        UNKNOWN_UNK,
    ],
    "short": [CLEAN_SHORT_XOM],
    "as_of": "2026-09-12T10:15:00",
}

#: The verdict every row must carry, verbatim. The gate reasons are the strings
#: `focus_adoption_gate` itself writes - asserted, never paraphrased, so a
#: reworded gate shows up here instead of quietly changing what the trader reads.
EXPECTED_VERDICTS = {
    ("long", "NVDA"): "adopted",
    ("long", "AMD"): "adopted",
    ("short", "XOM"): "adopted",
    ("long", "MSFT"): "not adopted: floor not above the D1 200 SMA",
    ("long", "TSLA"): "not adopted: not above yesterday's high",
    ("long", "IGN"): "not today",
    ("long", "DEC"): "declined today",
    ("long", "UNK"): "not adopted: cannot verify session VWAP",
}


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """(panel, store, focus_service, paths) with nothing attached yet."""
    import pick_feedback
    from focus_picks import FocusPickStore
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.services.focus_service import FocusService
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    pick_feedback.clear_reviewed_today_cache()
    longs_path = tmp_path / "longs.txt"
    shorts_path = tmp_path / "shorts.txt"
    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=tmp_path / "membership.json",
    )
    assert not store.uses_default_paths(), "a test store must never be the live one"
    focus_service = FocusService(store)

    panel = AlertCenterPanel(
        focus_service=focus_service,
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
        auto_pick_pending_path=tmp_path / "auto_pick_pending.json",
    )
    monkeypatch.setattr(panel, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(panel, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(panel, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(panel.chart_review, "_reviewed_symbols", lambda: set())
    # IGN is parked ("Not today"); DEC was on Focus and the trader took it off
    # today. Both clear every floor and pass the gate, so the ONLY thing that
    # may keep them off Focus is the trader's own answer.
    panel._ignore_alert_symbol("IGN")
    assert store.add("DEC", "long", "m5") is True
    assert store.remove("DEC", "long", "m5") is True
    assert store.declined_today("DEC", "long", "m5") is True

    paths = {
        "long": longs_path,
        "short": shorts_path,
        "pending": tmp_path / "auto_pick_pending.json",
    }
    yield panel, store, focus_service, paths
    panel.close()
    panel.deleteLater()


def _attach(desk, board: dict | None = None) -> _BoardService:
    panel, _store, focus_service, _paths = desk
    service = _BoardService(FULL_BOARD if board is None else board)
    panel.attach_strength_board(service, focus_service)
    return service


def _focus(store) -> dict[str, list[str]]:
    return {
        "long": sorted(store.focus_symbols("long", "m5")),
        "short": sorted(store.focus_symbols("short", "m5")),
    }


def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verdicts(panel) -> dict[tuple[str, str], str]:
    """Every board row's `adoption` field, as the panel's board model holds it."""
    board = panel.strength_board
    out: dict[tuple[str, str], str] = {}
    for side, table in (("long", board.longs), ("short", board.shorts)):
        for row in table.rows():
            out[(side, str(row.get("symbol") or ""))] = str(row.get("adoption") or "")
    return out


# ---------------------------------------------------------------------------
# item 1 - the trace
# ---------------------------------------------------------------------------
def test_the_board_reaches_the_bouncebot_scan_set(desk):
    """THE verification: board -> gate -> store -> longs.txt -> scan universe.

    Every link is asserted in order, so a break names itself instead of showing
    up as "the name is not being scanned".
    """
    panel, store, _service, paths = desk
    import bounce_bot_lib.legacy as legacy

    _attach(desk)

    # link 1-3: the gate's survivors are the persisted Focus membership.
    assert _focus(store) == {"long": ["AMD", "NVDA"], "short": ["XOM"]}
    assert store.is_auto_adopted("NVDA", "long", "m5")
    assert store.is_auto_adopted("AMD", "long", "m5")
    assert store.is_auto_adopted("XOM", "short", "m5")

    # link 4: the shared watchlists the scanner re-reads every cycle.
    assert sorted(_lines(paths["long"])) == ["AMD", "NVDA"]
    assert _lines(paths["short"]) == ["XOM"]

    # link 5: the universe BounceBot builds from those files.
    class _Stub:
        longs = legacy.read_tickers(paths["long"])
        shorts = legacy.read_tickers(paths["short"])
        _human_focus_symbols = staticmethod(lambda: set())
        _auto_watch_symbols = staticmethod(lambda side=None: set())
        get_master_avwap_d1_watch_symbols = staticmethod(lambda: [])
        _chart_watch_symbols = staticmethod(lambda: set())

    scanned = legacy.BounceBot.get_scan_symbol_set(_Stub())
    assert {"NVDA", "AMD", "XOM"} <= scanned
    # Scanner inclusion and Focus adoption are DISTINCT contracts: a row the
    # gate refused is not smuggled into the scan set by this packet.
    assert "TSLA" not in scanned
    assert "MSFT" not in scanned


def test_a_second_identical_board_appends_nothing(desk):
    """`longs.txt` gains each adopted name ONCE. The 15-minute refresh must not
    append a name that is already on the line."""
    panel, store, _service, paths = desk

    service = _attach(desk)
    first_long = _lines(paths["long"])
    first_short = _lines(paths["short"])

    service.publish(FULL_BOARD)
    service.publish(FULL_BOARD)

    assert _lines(paths["long"]) == first_long
    assert _lines(paths["short"]) == first_short
    assert sorted(first_long) == ["AMD", "NVDA"]


def test_a_trader_entered_watchlist_line_survives_every_refresh(desk):
    """plan.md sec 5: user-entered watchlist names are never auto-removed."""
    panel, store, _service, paths = desk
    paths["long"].write_text("MANUAL\n", encoding="utf-8")

    service = _attach(desk)
    service.publish(FULL_BOARD)

    lines = _lines(paths["long"])
    assert lines.count("MANUAL") == 1, lines
    assert {"AMD", "NVDA"} <= set(lines)


def test_away_stages_and_adopts_nothing(desk):
    """AWAY stages, never adopts (the auto-mode matrix). The staging path is the
    EXISTING auto-populate queue, so the return to DESK adopts through the drain
    that already owns it - no second door into Focus."""
    panel, store, _service, paths = desk
    import autopilot_core

    panel._auto_mode_now = lambda: "AWAY"

    _attach(desk)

    assert _focus(store) == {"long": [], "short": []}
    assert _lines(paths["long"]) == []
    pending = autopilot_core.load_auto_populate_pending_picks(paths["pending"])
    assert set(pending["pending"]["long"]) == {"NVDA", "AMD"}
    assert set(pending["pending"]["short"]) == {"XOM"}
    assert _verdicts(panel)[("long", "NVDA")] == "staged (AWAY)"


@pytest.mark.parametrize("mode", ["EVENING", "OFF"])
def test_evening_and_off_do_nothing(desk, mode):
    """Neither adopts nor stages; the board says which mode stopped it."""
    panel, store, _service, paths = desk
    import autopilot_core

    panel._auto_mode_now = lambda: mode

    _attach(desk)

    assert _focus(store) == {"long": [], "short": []}
    assert _lines(paths["long"]) == []
    pending = autopilot_core.load_auto_populate_pending_picks(paths["pending"])
    assert pending["pending"]["long"] == {}
    assert pending["pending"]["short"] == {}
    assert _verdicts(panel)[("long", "NVDA")] == f"mode {mode}"


# ---------------------------------------------------------------------------
# item 2 - the board says why
# ---------------------------------------------------------------------------
def test_every_board_row_carries_its_adoption_verdict(desk):
    """One verdict per row, in the gate's own words."""
    panel, _store, _service, _paths = desk

    _attach(desk)

    assert _verdicts(panel) == EXPECTED_VERDICTS


def test_a_name_already_on_focus_reads_already_in_focus(desk):
    """"nothing happened" is not an acceptable answer for a name the trader
    already owns - the board says it is already there."""
    panel, store, _service, _paths = desk
    assert store.add("NVDA", "long", "m5") is True

    _attach(desk)

    assert _verdicts(panel)[("long", "NVDA")] == "already_in_focus"
    assert store.is_auto_adopted("NVDA", "long", "m5") is False


def test_the_scan_column_is_the_last_column_and_renders(desk):
    """The trader reads this on the Strength page, so it has to be ON the table."""
    panel, _store, _service, _paths = desk

    _attach(desk)
    table = panel.strength_board.longs
    table.set_parity_only(False)  # so the greyed row is visible too

    headers = [
        table.table.horizontalHeaderItem(index).text()
        for index in range(table.table.columnCount())
        if table.table.horizontalHeaderItem(index) is not None
    ]
    assert "Scan" in headers
    scan_column = headers.index("Scan")
    assert scan_column == max(
        index for index, text in enumerate(headers) if text
    ), "Scan is the LAST named column"

    seen = {}
    for index in range(table.table.rowCount()):
        symbol = table.table.item(index, 0).text()
        cell = table.table.item(index, scan_column)
        seen[symbol] = cell.text() if cell is not None else None
    assert seen["NVDA"] == "adopted"
    assert seen["TSLA"] == "not adopted: not above yesterday's high"
    assert seen["MSFT"] == "not adopted: floor not above the D1 200 SMA"


def test_the_scan_column_never_reorders_the_board(desk):
    """Text only, no reorder: clicking `Scan` leaves the ranking alone."""
    panel, _store, _service, _paths = desk

    _attach(desk)
    table = panel.strength_board.longs
    before = table.sort_state()
    order_before = [
        table.table.item(index, 0).text() for index in range(table.table.rowCount())
    ]

    headers = [
        table.table.horizontalHeaderItem(index).text()
        for index in range(table.table.columnCount())
        if table.table.horizontalHeaderItem(index) is not None
    ]
    table._on_header_clicked(headers.index("Scan"))

    assert table.sort_state() == before
    assert [
        table.table.item(index, 0).text() for index in range(table.table.rowCount())
    ] == order_before


# ---------------------------------------------------------------------------
# item 3 - one log line per refresh
# ---------------------------------------------------------------------------
def test_one_line_per_board_refresh_names_the_counts_and_the_reasons(desk, caplog):
    panel, _store, _service, _paths = desk

    with caplog.at_level(logging.INFO):
        _attach(desk)

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Strength board: ")
    ]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "8 rows" in line
    assert "3 adopted" in line
    assert "0 staged" in line
    assert "5 not adopted" in line
    assert "reasons:" in line
    assert "not above yesterday's high" in line
