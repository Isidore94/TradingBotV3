"""P8 P8 item 2: new Movers names feed the M5 Focus watch (AUTO lane).

Trader, 2026-09-25: "My only thing with the movers is longs need to be above
previous day high and over vwap. Shorts need to be below previous day [low] and
below vwap. Outside of that I'm ok with those being put into the M5 list."

The levels are the board row's own M5 prior session (the +F click's source),
and only when that session IS the previous NY trading session (review
2026-09-25: stale daily files gave 09-22 / 09-23 as "previous day").
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import movers_notify as mn  # noqa: E402

NY = ZoneInfo("America/New_York")
T0 = datetime(2026, 9, 25, 10, 35, 20, tzinfo=NY)  # Friday
BAR = datetime(2026, 9, 25, 10, 30, tzinfo=NY).isoformat(timespec="seconds")
PREV = "2026-09-24"  # the previous NY trading session
OPEN = mn.focus_adoption_gate.OPEN
UNKNOWN = mn.focus_adoption_gate.UNKNOWN


def _row(symbol, *, pop, last, vwap, prev_high=100.0, prev_low=90.0, prev_session=PREV):
    return {"symbol": symbol, "pop_score": pop, "rvol": 2.0, "last": last, "session_vwap": vwap,
            "prev_high": prev_high, "prev_low": prev_low, "prev_session": prev_session}


def _board(long_rows=(), short_rows=()):
    return {"as_of": BAR, "as_of_stale": False, "state": {"state": "up_day"},
            "pop": {"long": list(long_rows), "short": list(short_rows)},
            "dip": {"long": [], "short": []}, "rip": {"long": [], "short": []}}


def _candidates(board, mode="DESK"):
    notices = mn.MoversNotifier().decide(board, mode=mode, now=T0, local_tz=NY)
    return mn.adoption_candidates(notices)


# ------------------------------------------------------------------ level gate
@pytest.mark.parametrize("last, vwap, passes", [
    (101.0, 100.5, True),    # above prev high, above VWAP
    (101.0, 102.0, False),   # above prev high, below VWAP
    (99.0, 98.0, False),     # below prev high, above VWAP
    (99.0, 99.5, False),     # below both
])
def test_long_needs_prev_high_and_vwap(last, vwap, passes):
    cands = _candidates(_board([_row("AAA", pop=1.0, last=last, vwap=vwap)]))
    assert [(c["symbol"], c["side"], c["passes"]) for c in cands] == [("AAA", "long", passes)]
    assert cands[0]["level_gate"] == {"prev_high": 100.0, "prev_low": 90.0,
                                      "prev_session": PREV, "vwap": vwap, "last": last}


@pytest.mark.parametrize("last, vwap, passes", [
    (89.0, 89.5, True),      # below prev low, below VWAP
    (89.0, 88.0, False),     # below prev low, above VWAP
    (91.0, 92.0, False),     # above prev low, below VWAP
    (91.0, 90.5, False),     # above both
])
def test_short_needs_prev_low_and_vwap(last, vwap, passes):
    cands = _candidates(_board(short_rows=[_row("AAA", pop=-1.0, last=last, vwap=vwap)]))
    assert [(c["side"], c["passes"]) for c in cands] == [("short", passes)]


def test_stale_prior_session_is_unknown_and_never_adopted():
    # The name's bars stop two sessions back: its "previous day" is 09-23.
    stale = _row("ADI", pop=1.0, last=101.0, vwap=100.5, prev_session="2026-09-23")
    cand = _candidates(_board([stale]))[0]
    assert cand["passes"] is False and cand["gate"] == UNKNOWN
    assert "2026-09-23" in cand["gate_reason"] and PREV in cand["gate_reason"]


def test_fresh_prior_session_is_gated_normally():
    cand = _candidates(_board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)]))[0]
    assert cand["passes"] is True and cand["gate"] == OPEN


def test_missing_prior_session_is_no_adoption():
    row = _row("ZZZ", pop=1.0, last=101.0, vwap=100.0, prev_high=None, prev_low=None,
               prev_session="")
    cand = _candidates(_board([row]))[0]
    assert cand["passes"] is False and cand["gate"] == UNKNOWN


def test_monday_prior_session_is_friday():
    row = _row("AAA", pop=1.0, last=101.0, vwap=100.5, prev_session="2026-09-25")
    state, _reason, _gate = mn.row_level_gate(row, "long", "2026-09-28")
    assert state == OPEN


def test_unknown_vwap_is_no_adoption():
    cands = _candidates(_board([_row("AAA", pop=1.0, last=101.0, vwap=None)]))
    assert cands[0]["passes"] is False


@pytest.mark.parametrize("mode", ["DESK", "AWAY", "EVENING"])
def test_desk_away_and_evening_all_feed_the_watch(mode):
    cands = _candidates(_board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)]), mode)
    assert cands and cands[0]["mode"] == mode


def test_off_mode_feeds_nothing():
    assert _candidates(_board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)]), "OFF") == []


def test_adopt_record_shape():
    cands = _candidates(_board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)]), "AWAY")
    rows = mn.adopt_records([dict(cands[0], result="adopted")], at=T0.isoformat())
    assert rows == [{
        "kind": "adopt", "session": "2026-09-25", "symbol": "AAA", "side": "long",
        "list": "pop", "state": "up_day", "mode": "AWAY", "bar": BAR,
        "level_gate": {"prev_high": 100.0, "prev_low": 90.0, "prev_session": PREV,
                       "vwap": 100.5, "last": 101.0},
        "gate": OPEN, "gate_reason": cands[0]["gate_reason"], "result": "adopted",
        "adopted_at": T0.isoformat()}]


def test_board_rows_carry_the_prior_session_date():
    import movers_scan

    bars = []
    for day, base in ((24, 100.0), (25, 101.0)):
        for i in range(4):
            dt = datetime(2026, 9, day, 9, 30 + 5 * i, tzinfo=NY)
            bars.append({"dt": dt, "open": base, "high": base + 1, "low": base - 1,
                         "close": base, "volume": 1e5})
    prior, today = movers_scan.split_today(bars)
    levels = movers_scan._levels(prior, today, 1.0)
    assert levels["prev_session"] == PREV and levels["prev_high"] == 101.0


# ------------------------------------------------------------------ service
@pytest.fixture
def run_service(tmp_path, monkeypatch):
    from ui.services import movers_service as svc

    made = []

    def run(mode, board):
        monkeypatch.setattr(svc, "_market_local_tz", lambda: NY)
        monkeypatch.setattr(svc.movers_scan, "build_movers_board", lambda *a, **k: dict(board))
        service = svc.MoversService(
            clock=lambda: T0, autostart=False,
            outcomes_path=tmp_path / "movers_dip_outcomes.jsonl",
            mode_provider=lambda: mode, push_sender=lambda *_a: {"ok": True, "kind": "delivered"},
            hidden_provider=lambda: ("", set()), sector_hidden=lambda _s: False,
        )
        made.append(service)
        offered = []
        service.moversAdopt.connect(offered.append)
        service._publish({}, [], T0, {"long": [], "short": []}, NY, final=True)
        return offered

    yield run
    for service in made:
        service.shutdown()
        service.deleteLater()


def test_service_offers_gated_names_from_the_board_row(run_service):
    offered = run_service("AWAY", _board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)]))
    cand = offered[0]["candidates"][0]
    assert (cand["symbol"], cand["passes"], cand["mode"]) == ("AAA", True, "AWAY")
    assert offered[0]["log_paths"]["dip"].endswith("movers_dip_outcomes.jsonl")


def test_service_off_mode_offers_nothing(run_service):
    assert run_service("OFF", _board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)])) == []


def test_service_never_reads_the_daily_store(run_service, monkeypatch):
    import autopilot_core

    def boom(*_a, **_k):
        raise AssertionError("durable daily context must not decide the Movers gate")

    monkeypatch.setattr(autopilot_core, "load_daily_context", boom)
    offered = run_service("DESK", _board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)]))
    assert offered[0]["candidates"][0]["passes"] is True


def test_gate_patched_to_refuse_blocks_adoption(run_service, monkeypatch):
    monkeypatch.setattr(mn.focus_adoption_gate, "focus_adoption_gate_state",
                        lambda *a, **k: (mn.focus_adoption_gate.CLOSED, "refused by the gate"))
    offered = run_service("DESK", _board([_row("AAA", pop=1.0, last=101.0, vwap=100.5)]))
    cand = offered[0]["candidates"][0]
    assert cand["passes"] is False and cand["gate_reason"] == "refused by the gate"


# ------------------------------------------------------------------ Alert Center
class FakeStore:
    def __init__(self, existing=(), declined=()):
        self.lists = {"long": list(existing), "short": []}
        self.declined = set(declined)
        self.markers = []
        self.removed = []

    def add_many(self, symbols, side, category="m5"):
        added = [s for s in symbols if s not in self.lists[side]]
        self.lists[side].extend(added)
        return added

    def mark_auto_adopted(self, symbol, side, category="m5", *, staged_at="", reason=""):
        self.markers.append((symbol, side, staged_at, reason))

    def declined_today(self, symbol, side, category="m5"):
        return symbol in self.declined

    def remove(self, *a, **k):  # never called
        self.removed.append(a)


class FakeFocus:
    def __init__(self, store):
        self.store = store
        self.added = []

    def add(self, symbol, side, category="m5", **_k):  # the +F (trader) path only
        self.added.append((symbol, side))
        return True


@pytest.fixture
def panel(tmp_path):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from ui.panels.alert_center_panel import AlertCenterPanel

    widget = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    yield widget
    thread = getattr(widget, "_movers_adopt_log_thread", None)
    if thread is not None:
        thread.join(5)
    widget.close()
    widget.deleteLater()
    app.processEvents()


def _payload(tmp_path, cands):
    return {"candidates": cands, "at": T0.isoformat(),
            "log_paths": {"pop": str(tmp_path / "pop.jsonl"), "dip": str(tmp_path / "dip.jsonl")}}


def _rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def test_panel_adopts_only_gated_names_marks_them_and_logs_rows(panel, tmp_path):
    store = FakeStore(existing=["OWN"])
    focus = FakeFocus(store)
    panel.focus_service = focus
    board = _board([_row("AAA", pop=2.0, last=101.0, vwap=100.5),
                    _row("BBB", pop=1.5, last=99.0, vwap=98.0),
                    _row("OWN", pop=1.2, last=101.0, vwap=100.5)])
    panel.adopt_movers_picks(_payload(tmp_path, _candidates(board, "AWAY")))
    panel._movers_adopt_log_thread.join(5)
    assert store.lists["long"] == ["OWN", "AAA"]
    assert [m[0] for m in store.markers] == ["AAA"]  # the trader's OWN is never re-marked
    assert store.removed == [] and focus.added == []  # never the trader-like path
    rows = _rows(tmp_path / "pop.jsonl")
    results = {r["symbol"]: r["result"] for r in rows}
    assert results["AAA"] == "adopted" and results["OWN"] == "already in M5 Focus"
    assert results["BBB"].startswith("gated:")
    assert all(r["kind"] == "adopt" and r["mode"] == "AWAY" for r in rows)


def test_panel_skips_a_stale_prior_session(panel, tmp_path):
    store = FakeStore()
    panel.focus_service = FakeFocus(store)
    board = _board([_row("AFRM", pop=2.0, last=101.0, vwap=100.5, prev_session="2026-09-23")])
    panel.adopt_movers_picks(_payload(tmp_path, _candidates(board)))
    panel._movers_adopt_log_thread.join(5)
    assert store.lists["long"] == []


def test_panel_honours_not_today_and_taken_off_today(panel, tmp_path):
    store = FakeStore(declined=["BBB"])
    panel.focus_service = FakeFocus(store)
    panel._ignored_symbols.add("AAA")
    board = _board([_row("AAA", pop=2.0, last=101.0, vwap=100.5),
                    _row("BBB", pop=1.5, last=101.0, vwap=100.5)])
    panel.adopt_movers_picks(_payload(tmp_path, _candidates(board)))
    panel._movers_adopt_log_thread.join(5)
    assert store.lists["long"] == []
    results = {r["symbol"]: r["result"] for r in _rows(tmp_path / "pop.jsonl")}
    assert results == {"AAA": "refused: you said not today",
                       "BBB": "refused: you took it off today"}


def test_panel_refuses_a_non_ticker(panel, tmp_path):
    store = FakeStore()
    panel.focus_service = FakeFocus(store)
    cand = _candidates(_board([_row("AAA", pop=2.0, last=101.0, vwap=100.5)]))[0]
    cand["symbol"] = "BAD SYM!"
    panel.adopt_movers_picks(_payload(tmp_path, [cand]))
    panel._movers_adopt_log_thread.join(5)
    assert store.lists["long"] == []
    assert _rows(tmp_path / "pop.jsonl")[0]["result"] == "refused: not a ticker"


@pytest.mark.parametrize("prev_session, last, vwap", [
    (PREV, 101.0, 100.5),          # passes both ways
    (PREV, 99.0, 98.0),            # inside yesterday's range both ways
    ("2026-09-23", 101.0, 100.5),  # stale prior session both ways
])
def test_plus_f_and_the_auto_feed_agree_on_the_same_row(panel, tmp_path, prev_session, last,
                                                        vwap):
    row = _row("AAA", pop=2.0, last=last, vwap=vwap, prev_session=prev_session)
    board = _board([row])
    auto = _candidates(board)[0]["passes"]
    focus = FakeFocus(FakeStore())
    panel.focus_service = focus
    panel.movers_board.update_board(board)
    panel._add_movers_row_to_focus("AAA", "long")
    assert bool(focus.added) is auto


def test_attach_wires_the_adopt_signal(panel, monkeypatch):
    from ui.services import movers_service as svc

    heard = []
    monkeypatch.setattr(panel, "adopt_movers_picks", heard.append)
    service = svc.MoversService(autostart=False, mode_provider=lambda: "DESK")
    try:
        panel.attach_movers_service(service)
        service.moversAdopt.emit({"candidates": []})
        assert heard == [{"candidates": []}]
    finally:
        service.shutdown()
        service.deleteLater()
