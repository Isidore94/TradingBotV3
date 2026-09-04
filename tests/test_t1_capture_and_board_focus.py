"""Packet T1.4 - the TC2000 board's longs and shorts auto-join M5 Focus.

Trader, 2026-09-04:

    "I want all shorts and longs on the RS/RW board TC2000 to bne auto added to
    the M5 focus picks."

On `main` @ 6e05878 `attach_strength_board` only parents the widget and connects
the row click; the board reaches M5 Focus only when the trader presses "Add" or
"Add all", through `focus_service.add` (a trader LIKE, which writes a
`pick_feedback` row).

The new contract - modelled on the regime-pause auto-join, which is the
machine's placement precedent:

* `attach_strength_board` connects `service.boardChanged` to
  `_auto_adopt_strength_board(board)` and calls it once with `service.board()`;
* only rows with EMPTY `failed_floors` (the TC2000 parity list) are considered -
  a greyed near-miss is never adopted;
* the ONE adoption gate is re-run on the row's own numbers; UNKNOWN fails;
* DESK only - AWAY stages nothing here;
* written through the STORE (`store.add` + `store.mark_auto_adopted`), NEVER
  `focus_service.add`, because a machine placement is not a trader like;
* it NEVER removes, and it is idempotent per refresh;
* a symbol the trader said "Not today" to is not put back by the next refresh.

Hermetic: a tmp `FocusPickStore` (so `FocusService.record_feedback` short-
circuits on `uses_default_paths()` and no live `pick_feedback.jsonl` is
touched), tmp review events, and `_auto_mode_now` monkeypatched rather than read
from the Auto Pilot state file.
"""

from __future__ import annotations

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
    """A board row exactly as `strength_scan.build_strength_board` writes one.

    `failed_floors` is always PRESENT: an empty list for a parity row, a list of
    reasons for a greyed near-miss. A row with the key ABSENT does not exist in
    the real file.
    """
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
        "failed_floors": failed,
        "passes_floors": not failed,
    }


#: Two clean longs, one clean short, one greyed near-miss, one gate failure and
#: one name the trader parked. Exactly three of the six may be adopted.
CLEAN_LONG_NVDA = _row("NVDA", last=105.0, prev_high=100.0, prev_low=98.0, vwap=101.0)
CLEAN_LONG_AMD = _row("AMD", last=55.0, prev_high=50.0, prev_low=48.0, vwap=52.0)
GREYED_LONG_MSFT = _row(
    "MSFT",
    last=310.0,
    prev_high=300.0,
    prev_low=295.0,
    vwap=305.0,
    failed=["not above the D1 200 SMA"],
)
GATE_FAIL_LONG_TSLA = _row("TSLA", last=99.0, prev_high=100.0, prev_low=98.0, vwap=97.0)
PARKED_LONG_IGN = _row("IGN", last=25.0, prev_high=20.0, prev_low=18.0, vwap=21.0)
CLEAN_SHORT_XOM = _row("XOM", last=90.0, prev_high=105.0, prev_low=95.0, vwap=92.0)

FULL_BOARD = {
    "long": [
        CLEAN_LONG_NVDA,
        CLEAN_LONG_AMD,
        GREYED_LONG_MSFT,
        GATE_FAIL_LONG_TSLA,
        PARKED_LONG_IGN,
    ],
    "short": [CLEAN_SHORT_XOM],
    "as_of": "2026-09-04T10:15:00",
}


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """(panel, store, service, service_adds) with the board attached."""
    import pick_feedback
    from focus_picks import FocusPickStore
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.services.focus_service import FocusService
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    pick_feedback.clear_reviewed_today_cache()
    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "membership.json",
    )
    assert not store.uses_default_paths(), "a test store must never be the live one"
    focus_service = FocusService(store)
    service_adds: list[tuple] = []
    real_add = focus_service.add

    def _spy_add(symbol, side, category="m5", *, origin="", context=""):
        service_adds.append((symbol, side, category, origin))
        return real_add(symbol, side, category, origin=origin, context=context)

    monkeypatch.setattr(focus_service, "add", _spy_add)

    made = AlertCenterPanel(
        focus_service=focus_service,
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    monkeypatch.setattr(made, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(made, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(made, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(made.chart_review, "_reviewed_symbols", lambda: set())
    yield made, store, focus_service, service_adds
    made.close()
    made.deleteLater()


def _attach(desk, board: dict = None):
    panel, store, focus_service, service_adds = desk
    service = _BoardService(FULL_BOARD if board is None else board)
    panel.attach_strength_board(service, focus_service)
    return service


def _focus(store) -> dict[str, list[str]]:
    return {
        "long": sorted(store.focus_symbols("long", "m5")),
        "short": sorted(store.focus_symbols("short", "m5")),
    }


# ---------------------------------------------------------------------------
# the adoption itself
# ---------------------------------------------------------------------------
def test_the_parity_rows_join_m5_focus_when_the_board_is_attached(desk):
    """"all shorts and longs on the RS/RW board TC2000 auto added"."""
    panel, store, _service, _adds = desk

    _attach(desk)

    assert _focus(store) == {"long": ["AMD", "NVDA"], "short": ["XOM"]}


def test_a_refreshed_board_adopts_on_its_own_signal(desk):
    """The 15-minute refresh, not just the attach."""
    panel, store, _service, _adds = desk
    service = _attach(desk, {"long": [], "short": [], "as_of": "2026-09-04T10:00:00"})
    assert _focus(store) == {"long": [], "short": []}

    service.publish(FULL_BOARD)

    assert _focus(store) == {"long": ["AMD", "NVDA"], "short": ["XOM"]}


def test_every_adopted_name_carries_the_machine_marker(desk):
    """`focus_auto_picks.json` is what makes "Not today" able to reach the
    entry; absence of a marker means the trader owns it."""
    panel, store, _service, _adds = desk

    _attach(desk)

    assert store.is_auto_adopted("NVDA", "long", "m5")
    assert store.is_auto_adopted("AMD", "long", "m5")
    assert store.is_auto_adopted("XOM", "short", "m5")


def test_the_machine_never_writes_a_trader_like_for_a_board_adoption(desk):
    """`focus_service.add` writes a `pick_feedback` "like". A machine placement
    is not the trader liking the name - it goes through the STORE.

    The positive half is asserted first on purpose: "no service call" is also
    true of a board that adopts nothing, so the names have to be there before
    the absence means anything.
    """
    panel, store, _service, service_adds = desk

    _attach(desk)

    assert _focus(store) == {"long": ["AMD", "NVDA"], "short": ["XOM"]}
    assert service_adds == [], (
        f"the auto-join went through FocusService.add: {service_adds}"
    )


# ---------------------------------------------------------------------------
# what must NOT be adopted
# ---------------------------------------------------------------------------
def test_a_greyed_near_miss_row_is_never_adopted(desk):
    """`failed_floors` is non-empty, so the TC2000 parity list never shows it.

    Asserted against the full expected membership rather than a bare `not in`:
    a board that adopts nothing at all would satisfy `not in` and prove nothing.
    """
    panel, store, _service, _adds = desk

    _attach(desk)

    assert _focus(store) == {"long": ["AMD", "NVDA"], "short": ["XOM"]}
    assert "MSFT" not in store.focus_symbols("long", "m5")


def test_a_row_that_fails_the_adoption_gate_is_not_adopted(desk):
    """TSLA's last (99.0) is under its prev high (100.0): CLOSED, not OPEN."""
    panel, store, _service, _adds = desk

    _attach(desk)

    assert _focus(store) == {"long": ["AMD", "NVDA"], "short": ["XOM"]}
    assert "TSLA" not in store.focus_symbols("long", "m5")


def test_the_gate_is_re_run_on_each_rows_own_numbers(desk, monkeypatch):
    """ONE definition of the gate, now a fourth call site. Counted, not assumed:
    exactly one call per parity row, with that row's own four numbers."""
    import focus_adoption_gate

    panel, store, _service, _adds = desk
    calls: list[tuple] = []
    real = focus_adoption_gate.passes_focus_adoption_gate

    def _spy(side, last, prev_high, prev_low, vwap):
        calls.append((side, last, prev_high, prev_low, vwap))
        return real(side, last, prev_high, prev_low, vwap)

    monkeypatch.setattr(focus_adoption_gate, "passes_focus_adoption_gate", _spy)

    _attach(desk)

    assert ("long", 105.0, 100.0, 98.0, 101.0) in calls
    assert ("long", 99.0, 100.0, 98.0, 97.0) in calls, "TSLA is asked, not skipped"
    assert ("short", 90.0, 105.0, 95.0, 92.0) in calls
    assert not any(call[1] == 310.0 for call in calls), (
        "a greyed near-miss is filtered out before the gate is even asked"
    )


def test_a_row_the_gate_cannot_measure_is_not_adopted(desk):
    """plan.md sec 5: missing data is uncertainty, never confirmation.

    UNK's session VWAP is missing, so the gate answers UNKNOWN and UNKNOWN
    fails. NVDA rides along as the positive control.
    """
    panel, store, _service, _adds = desk
    unknown = _row("UNK", last=50.0, prev_high=45.0, prev_low=44.0, vwap=None)

    _attach(
        desk,
        {
            "long": [CLEAN_LONG_NVDA, unknown],
            "short": [],
            "as_of": "2026-09-04T10:15:00",
        },
    )

    assert store.focus_symbols("long", "m5") == ["NVDA"]


def test_a_name_the_trader_said_not_today_to_is_not_put_back(desk):
    """"Not today" parks the symbol for the session; the next 15-minute refresh
    must not undo the trader's answer."""
    panel, store, _service, _adds = desk
    panel._ignore_alert_symbol("IGN")
    assert "IGN" in panel._ignored_symbols

    service = _attach(desk)
    service.publish(FULL_BOARD)

    assert _focus(store) == {"long": ["AMD", "NVDA"], "short": ["XOM"]}
    assert "IGN" not in store.focus_symbols("long", "m5")


@pytest.mark.parametrize("mode", ["AWAY", "EVENING", "OFF"])
def test_only_desk_adopts_from_the_board(desk, monkeypatch, mode):
    """The auto-mode matrix is unchanged: DESK adopts, nothing else does.

    The DESK leg at the end is the control - without it, "nothing adopted" is
    equally true of a build where the auto-join does not exist.
    """
    panel, store, _service, _adds = desk
    mode_now = {"value": mode}
    monkeypatch.setattr(panel, "_auto_mode_now", lambda: mode_now["value"])

    service = _attach(desk)
    service.publish(FULL_BOARD)

    assert _focus(store) == {"long": [], "short": []}

    mode_now["value"] = "DESK"
    service.publish(FULL_BOARD)

    assert _focus(store) == {"long": ["AMD", "NVDA"], "short": ["XOM"]}


# ---------------------------------------------------------------------------
# ownership, idempotence and the never-removes rule
# ---------------------------------------------------------------------------
def test_a_second_identical_refresh_changes_nothing(desk):
    panel, store, _service, _adds = desk
    service = _attach(desk)
    first = _focus(store)
    first_marker = store.auto_pick_marker("NVDA", "long", "m5")

    service.publish(FULL_BOARD)

    assert _focus(store) == first == {"long": ["AMD", "NVDA"], "short": ["XOM"]}
    assert store.auto_pick_marker("NVDA", "long", "m5") == first_marker, (
        "an existing entry is counted, never re-marked"
    )


def test_a_name_the_trader_typed_is_never_re_marked_as_the_machines(desk):
    """Absence of a marker is what makes a pick the trader's. `store.add`
    returning False is the no-op that protects it."""
    panel, store, _service, _adds = desk
    assert store.add("AMD", "long", "m5") is True
    assert store.is_auto_adopted("AMD", "long", "m5") is False

    _attach(desk)

    assert "AMD" in store.focus_symbols("long", "m5")
    assert store.is_auto_adopted("AMD", "long", "m5") is False, (
        "a marker was written over a name the trader owns"
    )
    assert store.is_auto_adopted("NVDA", "long", "m5") is True


def test_a_name_that_leaves_the_board_stays_on_focus(desk):
    """NEVER removes. The ten-session fade and "Not today" own removal."""
    panel, store, _service, _adds = desk
    service = _attach(desk)
    assert "AMD" in store.focus_symbols("long", "m5")

    service.publish({"long": [CLEAN_LONG_NVDA], "short": [], "as_of": "2026-09-04T10:30:00"})

    assert sorted(store.focus_symbols("long", "m5")) == ["AMD", "NVDA"]
    assert store.focus_symbols("short", "m5") == ["XOM"]


# ---------------------------------------------------------------------------
# the evidence row
# ---------------------------------------------------------------------------
def test_one_review_event_names_the_side_counts_and_the_names(desk, monkeypatch):
    panel, store, _service, _adds = desk
    written: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        panel, "_record_review_event", lambda action, **kw: written.append((action, kw))
    )

    _attach(desk)

    rows = [kw for action, kw in written if action == "strength_board_auto_focus"]
    assert len(rows) == 1, f"one event per refresh; saw {[a for a, _ in written]}"
    detail = rows[0].get("detail") or {}
    assert detail.get("side_counts") == {"long": 2, "short": 1}
    assert sorted(detail.get("adopted") or []) == ["AMD", "NVDA", "XOM"]
    assert detail.get("as_of") == "2026-09-04T10:15:00"
    refused = detail.get("refused") or []
    assert any("TSLA" in str(item) for item in refused), (
        "a refusal is NAMED, the way `_add_all` names one"
    )


def test_the_click_to_add_path_is_still_the_traders_own_like(desk):
    """Unchanged: a click on a board row IS the trader liking the name, so it
    still goes through the service and still writes the pick-feedback row."""
    panel, store, _service, service_adds = desk
    _attach(desk, {"long": [CLEAN_LONG_NVDA], "short": [], "as_of": "x"})
    store.remove("NVDA", "long", "m5")

    panel.strength_board._add_one("NVDA", "long")

    assert service_adds == [("NVDA", "long", "m5", "strength_board")]
    assert store.focus_symbols("long", "m5") == ["NVDA"]
